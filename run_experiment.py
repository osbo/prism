"""
Unified train + evaluate + visualize runner with config overrides.

Usage:
  # Full model (30 epochs)
  python run_experiment.py --exp_name full_model --n_epochs 30

  # Ablation: no photometric loss (reduced model)
  python run_experiment.py --exp_name no_photometric --lambda_render 0 --lambda_perceptual 0 --reduced_model

  # Ablation: no depth supervision
  python run_experiment.py --exp_name no_depth --lambda_depth 0 --reduced_model

  # Skip training (just re-evaluate an existing checkpoint)
  python run_experiment.py --exp_name full_model --skip_train

All outputs land in experiments/{exp_name}/
  model.pt, metrics/metrics.json, visuals/, meshes/
"""

import argparse
import logging
import sys
from pathlib import Path

from config import PRISMConfig
from train import train
from evaluate import evaluate
from visualize import visualize

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("run_experiment")


def build_config(args) -> PRISMConfig:
    cfg = PRISMConfig()

    if args.reduced_model:
        cfg.latent_dim      = 64
        cfg.sdf_hidden      = 128
        cfg.sdf_layers      = 4
        cfg.feat_dim        = 8
        cfg.n_rays          = 384
        cfg.n_samples       = 64
        cfg.n_importance    = 16
        log.info("Reduced model: latent=%d  sdf_hidden=%d  sdf_layers=%d",
                 cfg.latent_dim, cfg.sdf_hidden, cfg.sdf_layers)

    if args.n_epochs is not None:
        cfg.n_epochs = args.n_epochs

    # Lambda overrides
    lambda_fields = [
        "lambda_render", "lambda_depth", "lambda_normal", "lambda_eikonal",
        "lambda_perceptual", "lambda_sdf_surface", "lambda_sdf_sign",
        "lambda_sdf_band", "lambda_bg_sdf", "lambda_bg_alpha",
        "lambda_sil_bce", "lambda_sil_dice", "lambda_visual_hull",
    ]
    for field in lambda_fields:
        val = getattr(args, field, None)
        if val is not None:
            setattr(cfg, field, val)
            log.info("Override: %s = %s", field, val)

    # Checkpoint lives inside the experiment directory
    exp_dir = Path("experiments") / args.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    cfg.checkpoint = str(exp_dir / "model.pt")

    return cfg


def main():
    parser = argparse.ArgumentParser(description="Run one PRISM experiment end-to-end.")
    parser.add_argument("--exp_name",     required=True, help="Experiment name (used as directory).")
    parser.add_argument("--n_epochs",     type=int, default=30)
    parser.add_argument("--n_eval_objects", type=int, default=None,
                        help="Cap number of test objects during eval (default: all).")
    parser.add_argument("--reduced_model", action="store_true",
                        help="Use smaller model (latent=64, hidden=128, layers=4) to save time.")
    parser.add_argument("--skip_train",   action="store_true")
    parser.add_argument("--skip_eval",    action="store_true")
    parser.add_argument("--skip_vis",     action="store_true")

    # Lambda overrides (None = use PRISMConfig default)
    for name in ["lambda_render", "lambda_depth", "lambda_normal", "lambda_eikonal",
                 "lambda_perceptual", "lambda_sdf_surface", "lambda_sdf_sign",
                 "lambda_sdf_band", "lambda_bg_sdf", "lambda_bg_alpha",
                 "lambda_sil_bce", "lambda_sil_dice", "lambda_visual_hull"]:
        parser.add_argument(f"--{name}", type=float, default=None)

    args = parser.parse_args()

    cfg = build_config(args)
    exp_dir = Path("experiments") / args.exp_name

    # ------------------------------------------------------------------
    # 1. Train
    # ------------------------------------------------------------------
    if not args.skip_train:
        log.info("=== TRAINING: %s  (%d epochs) ===", args.exp_name, cfg.n_epochs)
        train(cfg)
    else:
        log.info("Skipping training.")

    ckpt = str(exp_dir / "model.pt")
    if not Path(ckpt).exists():
        log.error("Checkpoint not found: %s  (did training complete?)", ckpt)
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Evaluate  (pass cfg directly so architecture matches checkpoint)
    # ------------------------------------------------------------------
    if not args.skip_eval:
        log.info("=== EVALUATING: %s ===", args.exp_name)
        evaluate(cfg, n_objects=args.n_eval_objects, out_dir=exp_dir / "metrics")
    else:
        log.info("Skipping evaluation.")

    # ------------------------------------------------------------------
    # 3. Visualize  (pass cfg directly so architecture matches checkpoint)
    # ------------------------------------------------------------------
    if not args.skip_vis:
        log.info("=== VISUALIZING: %s ===", args.exp_name)
        n_vis = min(args.n_eval_objects or 10, 10)   # visualize at most 10 objects
        visualize(cfg, n_objects=n_vis,
                  out_dir=exp_dir / "visuals",
                  mesh_dir=exp_dir / "meshes")
    else:
        log.info("Skipping visualization.")

    log.info("=== DONE: %s ===  Results in %s/", args.exp_name, exp_dir)


if __name__ == "__main__":
    main()
