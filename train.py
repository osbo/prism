"""
python train.py [--data_root ...] [--batch_size N] [--n_epochs N]
python train.py --overfit   # single-object memorization (many random views / step; see PRISMConfig overfit_*)

Checkpointing:
  • By default, **does not** load ``model.pt`` even if it exists: fresh weights,
    stale ``model.pt`` is removed so the new run overwrites it on first save.
  • ``--resume`` — continue from ``model.pt`` (no path).
  • ``--resume path/to.pt`` — continue from that file.
"""

import argparse
import logging
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from config import PRISMConfig
from prism  import PRISM
from data.omniobject3d import OmniObject3DDataset, enumerate_object_pairs

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("train")


def lr_scale(step, warmup, total, min_ratio=0.01):
    if step < warmup:
        return step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p))


def save(path, model, opt, scaler, epoch, step, best_val):
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "scaler": scaler.state_dict(), "epoch": epoch,
                "step": step, "best_val": best_val}, path)


def load(path, model, opt=None, scaler=None):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(ck["model"], strict=False)
    if opt    and "opt"    in ck: opt.load_state_dict(ck["opt"])
    if scaler and "scaler" in ck: scaler.load_state_dict(ck["scaler"])
    return ck.get("epoch", 0), ck.get("step", 0), ck.get("best_val", float("inf"))


def validate(model, loader, device):
    """No ``@torch.no_grad()``: ``PRISM.forward`` uses ``torch.autograd.grad`` for ∇SDF."""
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        losses = model(
            batch["images"].to(device),
            batch["c2w"].to(device),
            batch["K"].to(device),
            batch["depth"].to(device),
            batch["normal"].to(device),
            gt_mask=batch["mask"].to(device),
            input_c2ws=batch["input_c2ws"].to(device),
            input_Ks=batch["input_Ks"].to(device),
            input_masks=batch.get("input_masks", batch["mask"].unsqueeze(1)).to(device),
        )
        total += losses["total"].item()
        n += 1
    model.train()
    return total / max(n, 1)


def train(
    cfg: PRISMConfig,
    resume_path: str | None = None,
    restrict_object_ids: list[str] | None = None,
    overfit: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)
    if device.type == "cuda":
        torch.cuda.init()

    run_epochs = cfg.n_epochs
    if overfit:
        run_epochs = max(cfg.n_epochs, cfg.overfit_min_epochs)
        if run_epochs > cfg.n_epochs:
            log.info("--overfit: raising n_epochs from %d to %d", cfg.n_epochs, run_epochs)

    model = PRISM(cfg).to(device)

    enc_params   = list(model.encoder.parameters())
    other_params = [p for p in model.parameters() if not any(p is e for e in enc_params)]
    lr_enc = cfg.lr if overfit else cfg.lr_encoder
    wd = cfg.overfit_weight_decay if overfit else cfg.weight_decay
    opt = torch.optim.AdamW([
        {"params": enc_params,   "lr": lr_enc, "initial_lr": lr_enc},
        {"params": other_params, "lr": cfg.lr, "initial_lr": cfg.lr},
    ], weight_decay=wd)
    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))

    train_virtual = cfg.overfit_samples_per_epoch if overfit else None
    if overfit:
        log.info(
            "--overfit: %d random multi-view samples per epoch (warmup_steps=%d, weight_decay=%s, encoder_lr=%.1e)",
            cfg.overfit_samples_per_epoch,
            cfg.overfit_warmup_steps,
            wd,
            lr_enc,
        )
    train_ds = OmniObject3DDataset(
        cfg.data_root, split="train", image_size=cfg.image_size, n_input_views=cfg.n_input_views,
        restrict_object_ids=restrict_object_ids,
        virtual_epoch_len=train_virtual,
    )
    val_ds = OmniObject3DDataset(
        cfg.data_root, split="val", image_size=cfg.image_size, n_input_views=cfg.n_input_views,
        restrict_object_ids=restrict_object_ids,
    )
    bs_train = min(cfg.batch_size, max(1, len(train_ds)))
    bs_val = min(cfg.batch_size, max(1, len(val_ds)))
    if bs_train < cfg.batch_size or bs_val < cfg.batch_size:
        log.info("Batch size clamped to train=%d val=%d (dataset smaller than cfg.batch_size=%d)",
                 bs_train, bs_val, cfg.batch_size)
    train_loader = DataLoader(train_ds, batch_size=bs_train, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
                              persistent_workers=cfg.num_workers > 0)
    val_loader   = DataLoader(val_ds,   batch_size=bs_val, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=True,
                              persistent_workers=cfg.num_workers > 0)
    log.info(
        "Data: train len=%d (%d objects) / val len=%d",
        len(train_ds),
        len(train_ds.objects),
        len(val_ds),
    )

    start_epoch, step, best_val = 0, 0, float("inf")
    ckpt_path = Path(cfg.checkpoint)
    if resume_path is not None:
        src = Path(resume_path)
        start_epoch, step, best_val = load(src, model, opt, scaler)
        log.info("Resumed %s (epoch %d, step %d)", src, start_epoch, step)
    else:
        if ckpt_path.exists():
            ckpt_path.unlink()
            log.info("Fresh run: removed stale %s", ckpt_path)

    total_steps = len(train_loader) * run_epochs
    warmup_steps = cfg.overfit_warmup_steps if overfit else cfg.warmup_steps
    model.train()

    t_prev = time.perf_counter()
    end_epoch = start_epoch + run_epochs
    for epoch in range(start_epoch, end_epoch):
        for batch in train_loader:
            run_step = step - (start_epoch * len(train_loader))
            scale = lr_scale(run_step, warmup_steps, total_steps)
            for pg in opt.param_groups:
                pg["lr"] = pg["initial_lr"] * scale

            images       = batch["images"].to(device, non_blocking=True)
            depth        = batch["depth"].to(device, non_blocking=True)
            normal       = batch["normal"].to(device, non_blocking=True)
            mask         = batch["mask"].to(device, non_blocking=True)
            c2w          = batch["c2w"].to(device, non_blocking=True)
            K            = batch["K"].to(device, non_blocking=True)
            input_c2ws   = batch["input_c2ws"].to(device, non_blocking=True)
            input_Ks     = batch["input_Ks"].to(device, non_blocking=True)
            input_masks  = batch.get("input_masks", batch["mask"].unsqueeze(1)).to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=(device.type == "cuda")):
                losses = model(images, c2w, K, depth, normal, gt_mask=mask,
                               input_c2ws=input_c2ws, input_Ks=input_Ks,
                               input_masks=input_masks)

            loss = losses["total"]
            if not torch.isfinite(loss).all():
                log.warning("non-finite loss at step %d — skipping batch", step)
                step += 1
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gn = nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            if not torch.isfinite(torch.as_tensor(gn, device=device)):
                log.warning("non-finite grad norm at step %d — skipping optimizer step", step)
                scaler.update()
                step += 1
                continue

            scaler.step(opt)
            scaler.update()

            with torch.no_grad():
                if not torch.isfinite(model.log_beta):
                    model.log_beta.zero_()
                # Anneal the beta upper bound: start soft (large beta), sharpen over training.
                t_frac    = min(step / max(total_steps, 1), 1.0)
                beta_max  = cfg.beta_anneal_start * (1.0 - t_frac) + cfg.beta_anneal_end * t_frac
                log_lo    = math.log(cfg.beta_min)
                log_hi    = math.log(max(beta_max, cfg.beta_min))
                model.log_beta.clamp_(log_lo, log_hi)

            if step % cfg.log_every == 0:
                t_now = time.perf_counter()
                ms_per_step = (
                    1000.0 * (t_now - t_prev) / max(cfg.log_every, 1) if step > 0 else float("nan")
                )
                t_prev = t_now
                log.info(
                    "[%d/%d] step=%d  total=%.4f  render=%.4f  depth=%.4f  "
                    "normal=%.4f  eik=%.4f  curv=%.4f  sdf0=%.4f  sdf_sign=%.4f  sdf_band=%.4f  bg_sdf=%.4f  "
                    "bg_alpha=%.4f  sil_bce=%.4f  sil_dice=%.4f  hull=%.4f  lface=%.4f  close=%.4f  β=%.3f  lr=%.1e  %s",
                    epoch, end_epoch, step,
                    losses["total"].item(), losses["render"].item(),
                    losses["depth"].item(), losses["normal"].item(),
                    losses["eikonal"].item(),
                    losses["curvature"].item(),
                    losses["sdf_surface"].item(), losses["sdf_sign"].item(),
                    losses["sdf_band"].item(),
                    losses["bg_sdf"].item(),
                    losses["bg_alpha"].item(),
                    losses["sil_bce"].item(),
                    losses["sil_dice"].item(),
                    losses["visual_hull"].item(),
                    losses["light_facing"].item(),
                    losses["closure"].item(),
                    model.beta.item(),
                    opt.param_groups[1]["lr"],
                    (f"{ms_per_step:.1f}ms/step" if step > 0 else "—"),
                )
            step += 1

        if (epoch + 1) % cfg.eval_every == 0:
            val_loss = validate(model, val_loader, device)
            log.info("  Val loss: %.4f", val_loss)
            if val_loss < best_val:
                best_val = val_loss
                save(ckpt_path, model, opt, scaler, epoch, step, best_val)
                log.info("  → Best checkpoint saved")

        if (epoch + 1) % cfg.save_every == 0:
            save(ckpt_path, model, opt, scaler, epoch, step, best_val)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",  type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--n_epochs",   type=int)
    parser.add_argument("--lr",         type=float)
    parser.add_argument("--image_size", type=int)
    parser.add_argument(
        "--resume",
        nargs="?",
        default=None,
        const="",
        metavar="PATH",
        help="Continue training: optional checkpoint path; default is cfg.checkpoint. "
        "Omit entirely for a fresh run (deletes existing checkpoint file first).",
    )
    parser.add_argument(
        "--overfit",
        action="store_true",
        help="Memorize one object: restrict data + many random view batches per epoch + overfit-friendly LR/WD.",
    )
    parser.add_argument(
        "--overfit_object",
        type=str,
        default=None,
        metavar="ID",
        help="With --overfit, which object_id (default: first from enumerate_object_pairs).",
    )
    args = parser.parse_args()

    cfg = PRISMConfig()
    for k in ("data_root", "batch_size", "n_epochs", "lr", "image_size"):
        v = getattr(args, k)
        if v is not None:
            setattr(cfg, k, v)

    if args.resume is None:
        resume_path = None
    elif args.resume == "":
        resume_path = str(Path(cfg.checkpoint))
    else:
        resume_path = args.resume

    restrict: list[str] | None = None
    if args.overfit:
        pairs = enumerate_object_pairs(cfg.data_root)
        if not pairs:
            raise RuntimeError(f"--overfit: no objects under data_root={cfg.data_root!r}")
        if args.overfit_object is not None:
            oid = args.overfit_object
            if not any(o[1] == oid for o in pairs):
                raise RuntimeError(
                    f"--overfit_object {oid!r} not found under {cfg.data_root!r} "
                    f"(have {len(pairs)} ids, e.g. {pairs[0][1]!r})"
                )
            cat0 = next(c for c, o in pairs if o == oid)
            oid0 = oid
        else:
            cat0, oid0 = pairs[0]
        restrict = [oid0]
        log.info("--overfit: memorizing %s (category %s)", oid0, cat0)

    train(cfg, resume_path=resume_path, restrict_object_ids=restrict, overfit=args.overfit)
