"""
PRISM training script — PyTorch DDP + AMP on MIT ORCD Engaging.

Launch (single node, N GPUs):
    torchrun --nproc_per_node=N train.py [--overrides key=value ...]

Launch (SLURM sbatch):
    See slurm_train.sh for a ready-to-go job script.

Key design decisions:
  • Mixed-precision (fp16) via torch.cuda.amp.GradScaler
  • Gradient clipping to prevent exploding gradients through the sphere tracer
  • Separate LR for the encoder (smaller, to preserve pretrained features)
  • NeuS scale s is annealed every step: s ← min(s * 1.01, s_max)
  • Checkpoints saved every `save_every_n_epochs` and on val-loss improvement
"""

import os
import sys
import math
import argparse
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast

import wandb

# Make prism/ and data/ importable from the project root
sys.path.insert(0, str(Path(__file__).parent))

from config import PRISMConfig
from prism import PRISM
from data import build_dataloaders


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("prism")


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------

def setup_ddp():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, local_rank

def is_main(rank: int) -> bool:
    return rank == 0

def cleanup_ddp():
    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# LR schedule helpers
# ---------------------------------------------------------------------------

def cosine_lr(step: int, warmup_steps: int, total_steps: int,
               lr_min_ratio: float = 0.01) -> float:
    """Cosine decay with linear warmup. Returns a multiplier in [lr_min_ratio, 1]."""
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_min_ratio + (1.0 - lr_min_ratio) * cosine


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: Path,
    model: PRISM,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    epoch: int,
    step: int,
    val_loss: float,
):
    # Unwrap DDP before saving
    state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
    torch.save({
        "model":     state,
        "optimizer": optimizer.state_dict(),
        "scaler":    scaler.state_dict(),
        "epoch":     epoch,
        "step":      step,
        "val_loss":  val_loss,
    }, path)


def load_checkpoint(path: Path, model: PRISM, optimizer=None, scaler=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt.get("epoch", 0), ckpt.get("step", 0), ckpt.get("val_loss", float("inf"))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model, loader, device, rank, cfg):
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        image  = batch["image"].to(device, non_blocking=True)
        depth  = batch["depth"].to(device, non_blocking=True)
        normal = batch["normal"].to(device, non_blocking=True)
        c2w    = batch["c2w"].to(device, non_blocking=True)
        K      = batch["K"].to(device, non_blocking=True)

        with autocast(enabled=cfg.train.mixed_precision):
            losses, _, _ = model(
                image=image,
                camera_extrinsics=c2w,
                camera_intrinsics=K,
                gt_image=image,    # render the same view
                gt_depth=depth,
                gt_normal=normal,
                n_rays=cfg.renderer.n_rays_eval // 4,  # fewer rays for val speed
            )

        total_loss += losses["total"].item()
        n_batches  += 1

    # Average across all ranks
    avg = torch.tensor(total_loss / max(n_batches, 1), device=device)
    dist.all_reduce(avg, op=dist.ReduceOp.AVG)
    model.train()
    return avg.item()


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(cfg: PRISMConfig, resume_from: str | None = None):
    rank, local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    torch.manual_seed(cfg.seed + rank)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = PRISM(cfg).to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    if cfg.train.compile:
        model = torch.compile(model)

    # ------------------------------------------------------------------
    # Optimizer: two param groups — encoder (lower LR) + rest
    # ------------------------------------------------------------------
    encoder_params = list(model.module.encoder.parameters())
    other_params   = [p for p in model.parameters()
                      if not any(p is ep for ep in encoder_params)]

    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": cfg.train.lr_encoder},
            {"params": other_params,   "lr": cfg.train.lr},
        ],
        weight_decay=cfg.train.weight_decay,
    )
    scaler = GradScaler(enabled=cfg.train.mixed_precision)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    # Build datasets (all ranks) but use DistributedSampler
    from data.omniobject3d import OmniObject3DDataset
    from torch.utils.data import DataLoader

    train_ds = OmniObject3DDataset(cfg.train.data_root, split="train")
    val_ds   = OmniObject3DDataset(cfg.train.data_root, split="val")

    train_sampler = DistributedSampler(train_ds, shuffle=True)
    val_sampler   = DistributedSampler(val_ds,   shuffle=False)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.train.batch_size,
        sampler=train_sampler, num_workers=cfg.train.num_workers,
        pin_memory=True, drop_last=True, persistent_workers=cfg.train.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.train.batch_size,
        sampler=val_sampler, num_workers=cfg.train.num_workers,
        pin_memory=True, drop_last=False, persistent_workers=cfg.train.num_workers > 0,
    )

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------
    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")

    if resume_from:
        start_epoch, global_step, best_val_loss = load_checkpoint(
            Path(resume_from), model.module, optimizer, scaler
        )
        if is_main(rank):
            log.info(f"Resumed from {resume_from} (epoch {start_epoch}, step {global_step})")

    # ------------------------------------------------------------------
    # LR schedule
    # ------------------------------------------------------------------
    total_steps = len(train_loader) * cfg.train.n_epochs

    def get_lr_scale(step: int) -> float:
        return cosine_lr(step, cfg.train.warmup_steps, total_steps)

    # ------------------------------------------------------------------
    # wandb (main rank only)
    # ------------------------------------------------------------------
    if is_main(rank):
        wandb.init(project=cfg.project_name, config=cfg.__dict__, resume="allow")
        output_dir = Path(cfg.train.output_dir) / wandb.run.id
        output_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"Output dir: {output_dir}")
    else:
        output_dir = None

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    model.train()
    for epoch in range(start_epoch, cfg.train.n_epochs):
        train_sampler.set_epoch(epoch)

        for batch in train_loader:
            # LR update
            scale = get_lr_scale(global_step)
            for pg in optimizer.param_groups:
                pg["lr"] = pg.get("initial_lr", cfg.train.lr) * scale

            image  = batch["image"].to(device, non_blocking=True)
            depth  = batch["depth"].to(device, non_blocking=True)
            normal = batch["normal"].to(device, non_blocking=True)
            c2w    = batch["c2w"].to(device, non_blocking=True)
            K      = batch["K"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=cfg.train.mixed_precision):
                losses, render_out, extras = model(
                    image=image,
                    camera_extrinsics=c2w,
                    camera_intrinsics=K,
                    gt_image=image,
                    gt_depth=depth,
                    gt_normal=normal,
                )

            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            # Anneal NeuS scale
            renderer = model.module.renderer if isinstance(model, DDP) else model.renderer
            renderer.anneal_neus_s(factor=1.005)

            # ----------------------------------------------------------
            # Logging
            # ----------------------------------------------------------
            if is_main(rank) and global_step % cfg.train.log_every_n_steps == 0:
                log_dict = {
                    "train/loss":      losses["total"].item(),
                    "train/render":    losses["render"].item(),
                    "train/depth":     losses["depth"].item(),
                    "train/normal":    losses["normal"].item(),
                    "train/eikonal":   losses["eikonal"].item(),
                    "train/neus_s":    renderer.neus_s.item(),
                    "train/lr":        optimizer.param_groups[1]["lr"],
                    "train/hit_frac":  render_out["hit_mask"].float().mean().item(),
                    "epoch":           epoch,
                    "step":            global_step,
                }
                wandb.log(log_dict, step=global_step)
                log.info(
                    f"[{epoch}/{cfg.train.n_epochs}] step={global_step:6d} "
                    f"loss={losses['total'].item():.4f} "
                    f"render={losses['render'].item():.4f} "
                    f"eik={losses['eikonal'].item():.4f}"
                )

            global_step += 1

        # ------------------------------------------------------------------
        # Epoch-end: validation + checkpoint
        # ------------------------------------------------------------------
        if (epoch + 1) % cfg.train.eval_every_n_epochs == 0:
            val_loss = validate(model, val_loader, device, rank, cfg)
            if is_main(rank):
                wandb.log({"val/loss": val_loss, "epoch": epoch}, step=global_step)
                log.info(f"  Val loss: {val_loss:.4f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(
                        output_dir / "best.pt",
                        model.module, optimizer, scaler,
                        epoch, global_step, val_loss,
                    )
                    log.info(f"  → New best model saved.")

        if is_main(rank) and (epoch + 1) % cfg.train.save_every_n_epochs == 0:
            save_checkpoint(
                output_dir / f"epoch_{epoch+1:04d}.pt",
                model.module, optimizer, scaler,
                epoch, global_step, best_val_loss,
            )

    cleanup_ddp()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PRISM")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--n_epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--n_rays_train", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()

    cfg = PRISMConfig()

    # Apply CLI overrides
    if args.data_root:    cfg.train.data_root       = args.data_root
    if args.output_dir:   cfg.train.output_dir      = args.output_dir
    if args.batch_size:   cfg.train.batch_size       = args.batch_size
    if args.n_epochs:     cfg.train.n_epochs         = args.n_epochs
    if args.lr:           cfg.train.lr               = args.lr
    if args.n_rays_train: cfg.renderer.n_rays_train  = args.n_rays_train

    if args.no_wandb:
        os.environ["WANDB_MODE"] = "disabled"

    train(cfg, resume_from=args.resume)
