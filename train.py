"""
python train.py [--data_root ...] [--batch_size N] [--n_epochs N]

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
from data.omniobject3d import OmniObject3DDataset

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
            batch["image"].to(device),
            batch["c2w"].to(device),
            batch["K"].to(device),
            batch["depth"].to(device),
            batch["normal"].to(device),
            gt_mask=batch["mask"].to(device),
        )
        total += losses["total"].item()
        n += 1
    model.train()
    return total / max(n, 1)


def train(cfg: PRISMConfig, resume_path: str | None = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)
    if device.type == "cuda":
        torch.cuda.init()

    model = PRISM(cfg).to(device)

    enc_params   = list(model.encoder.parameters())
    other_params = [p for p in model.parameters() if not any(p is e for e in enc_params)]
    opt = torch.optim.AdamW([
        {"params": enc_params,   "lr": cfg.lr_encoder, "initial_lr": cfg.lr_encoder},
        {"params": other_params, "lr": cfg.lr,         "initial_lr": cfg.lr},
    ], weight_decay=cfg.weight_decay)
    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))

    train_ds = OmniObject3DDataset(cfg.data_root, split="train", image_size=cfg.image_size)
    val_ds   = OmniObject3DDataset(cfg.data_root, split="val",   image_size=cfg.image_size)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
                              persistent_workers=cfg.num_workers > 0)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=True,
                              persistent_workers=cfg.num_workers > 0)
    log.info("Data: %d train / %d val objects", len(train_ds), len(val_ds))

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

    run_epochs = cfg.n_epochs
    total_steps = len(train_loader) * run_epochs
    model.train()

    t_prev = time.perf_counter()
    end_epoch = start_epoch + run_epochs
    for epoch in range(start_epoch, end_epoch):
        for batch in train_loader:
            run_step = step - (start_epoch * len(train_loader))
            scale = lr_scale(run_step, cfg.warmup_steps, total_steps)
            for pg in opt.param_groups:
                pg["lr"] = pg["initial_lr"] * scale

            image  = batch["image"].to(device, non_blocking=True)
            depth  = batch["depth"].to(device, non_blocking=True)
            normal = batch["normal"].to(device, non_blocking=True)
            mask   = batch["mask"].to(device, non_blocking=True)
            c2w    = batch["c2w"].to(device, non_blocking=True)
            K      = batch["K"].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=(device.type == "cuda")):
                losses = model(image, c2w, K, depth, normal, gt_mask=mask)

            loss = losses["total"]
            if not torch.isfinite(loss).all():
                log.warning("non-finite loss at step %d — skipping batch", step)
                scaler.update()
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
                model.log_beta.clamp_(-10.0, 6.0)

            if step % cfg.log_every == 0:
                t_now = time.perf_counter()
                ms_per_step = (
                    1000.0 * (t_now - t_prev) / max(cfg.log_every, 1) if step > 0 else float("nan")
                )
                t_prev = t_now
                log.info(
                    "[%d/%d] step=%d  total=%.4f  render=%.4f  render_bg=%.4f  depth=%.4f  "
                    "normal=%.4f  eik=%.4f  sdf0=%.4f  sdf_sign=%.4f  bg_sdf=%.4f  occ=%.4f  "
                    "sil_bce=%.4f  sil_dice=%.4f  bg_inf=%.4f  lface=%.4f  close=%.4f  β=%.3f  lr=%.1e  %s",
                    epoch, end_epoch, step,
                    losses["total"].item(), losses["render"].item(),
                    losses["render_bg"].item(),
                    losses["depth"].item(), losses["normal"].item(),
                    losses["eikonal"].item(),
                    losses["sdf_surface"].item(), losses["sdf_sign"].item(),
                    losses["bg_sdf"].item(),
                    losses["opacity"].item(),
                    losses["sil_bce"].item(),
                    losses["sil_dice"].item(),
                    losses["bg_inf"].item(),
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

    train(cfg, resume_path=resume_path)
