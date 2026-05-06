"""
Quantitative evaluation: Chamfer Distance, F-Score @ τ, PSNR.

For each test object the script:
  1. Encodes the input views and extracts a mesh via marching cubes.
  2. Loads the GT raw-scan mesh from batch["mesh_path"].
  3. Normalises both meshes to the GT bounding-box scale and computes
     Chamfer distance and F-Score in that normalised space.
  4. Renders each input view with model.render_image and computes PSNR
     against the GT image on foreground pixels.

Results are saved to --out_dir/metrics.json and printed as a summary table.

Usage:
  python evaluate.py [--n_objects N] [--checkpoint model.pt]
  python evaluate.py --overfit [--overfit_object bottle_001]
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader

import trimesh
import trimesh.sample

from config import PRISMConfig
from prism import PRISM
from prism.mesh_extract import extract_sdf_mesh
from data.omniobject3d import OmniObject3DDataset, enumerate_object_pairs

log = logging.getLogger("evaluate")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _load_gt_mesh(mesh_path: str) -> trimesh.Trimesh | None:
    p = Path(mesh_path)
    if not p.exists():
        log.warning("GT mesh not found: %s", p)
        return None
    try:
        m = trimesh.load(str(p), force="mesh", process=False)
        if isinstance(m, trimesh.Scene):
            m = m.dump(concatenate=True)
        return m
    except Exception as exc:
        log.warning("Failed to load GT mesh %s: %s", p, exc)
        return None


def _sample_surface(mesh: trimesh.Trimesh, n: int) -> np.ndarray:
    pts, _ = trimesh.sample.sample_surface(mesh, n)
    return np.array(pts, dtype=np.float64)


def _bbox_normalisation(mesh: trimesh.Trimesh) -> tuple[np.ndarray, float]:
    """Return (centroid, scale) where scale = bounding-box diagonal length."""
    v = np.array(mesh.vertices, dtype=np.float64)
    centroid = v.mean(axis=0)
    diag = float(np.linalg.norm(v.max(axis=0) - v.min(axis=0)))
    return centroid, max(diag, 1e-8)


def chamfer_distance(pred: np.ndarray, gt: np.ndarray) -> float:
    """Bidirectional mean nearest-neighbour distance (lower is better)."""
    t_pred = cKDTree(pred)
    t_gt   = cKDTree(gt)
    d_pg, _ = t_gt.query(pred)
    d_gp, _ = t_pred.query(gt)
    return float((d_pg.mean() + d_gp.mean()) / 2.0)


def f_score(pred: np.ndarray, gt: np.ndarray, tau: float) -> float:
    """F-Score at threshold tau: harmonic mean of precision and recall (higher is better)."""
    t_pred = cKDTree(pred)
    t_gt   = cKDTree(gt)
    d_pg, _ = t_gt.query(pred)    # each pred point  → nearest GT
    d_gp, _ = t_pred.query(gt)    # each GT   point  → nearest pred
    precision = float((d_pg < tau).mean())
    recall    = float((d_gp < tau).mean())
    denom = precision + recall
    return 2.0 * precision * recall / denom if denom > 1e-8 else 0.0


# ---------------------------------------------------------------------------
# Image metric
# ---------------------------------------------------------------------------

def psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    """PSNR in dB; arrays in [0, 1]."""
    mse = ((pred.astype(np.float64) - gt.astype(np.float64)) ** 2).mean()
    return float(10.0 * np.log10(1.0 / mse)) if mse > 1e-10 else float("inf")


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    cfg: PRISMConfig,
    n_objects: int | None,
    out_dir: Path,
    restrict_object_ids: list[str] | None = None,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PRISM(cfg).to(device)
    ckpt  = torch.load(cfg.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    test_ds = OmniObject3DDataset(
        cfg.data_root,
        split="test",
        image_size=cfg.image_size,
        n_input_views=cfg.n_input_views,
        restrict_object_ids=restrict_object_ids,
    )
    loader  = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)
    n_total = min(len(test_ds), n_objects) if n_objects else len(test_ds)
    log.info("Evaluating %d test objects → %s", n_total, out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_object: list[dict] = []

    for i, batch in enumerate(loader):
        if n_objects is not None and i >= n_objects:
            break

        obj_id    = batch["object_id"][0]
        mesh_path = batch["mesh_path"][0]
        images    = batch["images"].to(device)      # (1, N, 3, H, W)
        input_c2ws = batch["input_c2ws"].to(device)
        input_Ks   = batch["input_Ks"].to(device)
        N = images.shape[1]

        log.info("[%d/%d]  %s", i + 1, n_total, obj_id)
        t_start = time.perf_counter()

        # ---- Encode -------------------------------------------------------
        with torch.no_grad():
            z_latent, feat_maps = model.encoder(images)
        z_single = z_latent[0]

        # ---- Extract predicted mesh ---------------------------------------
        H, W = images.shape[-2], images.shape[-1]
        mc = extract_sdf_mesh(
            model, z_single, cfg, device,
            mask_hw=batch["mask"][0],
            c2w=batch["c2w"].to(device),
            K=batch["K"].to(device),
            input_masks=batch["input_masks"].to(device) if "input_masks" in batch else None,
            input_c2ws=input_c2ws,
            input_Ks=input_Ks,
            feat_maps=feat_maps,
            img_hw=(H, W),
        )

        # ---- Geometry metrics --------------------------------------------
        rec: dict = {"object_id": obj_id, "mesh_path": mesh_path}
        gt_mesh = _load_gt_mesh(mesh_path)

        if mc is None:
            log.warning("  No iso-surface for %s — geometry metrics skipped", obj_id)
            rec.update({"chamfer": None, "fscore": None})
        elif gt_mesh is None:
            rec.update({"chamfer": None, "fscore": None})
        else:
            verts, faces = mc
            pred_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

            # Normalise both meshes to the GT bounding-box scale so Chamfer and
            # F-Score are comparable across objects of different physical sizes.
            # Both meshes share the world coordinate frame of the training cameras
            # (OmniObject3D raw scans and blender renders originate from the same
            # Blender scene), so we subtract the GT centroid from both and divide
            # by the GT diagonal rather than normalising independently.
            gt_centroid, gt_scale = _bbox_normalisation(gt_mesh)
            pred_pts = (_sample_surface(pred_mesh, cfg.n_eval_pts) - gt_centroid) / gt_scale
            gt_pts   = (_sample_surface(gt_mesh,   cfg.n_eval_pts) - gt_centroid) / gt_scale

            cd = chamfer_distance(pred_pts, gt_pts)
            fs = f_score(pred_pts, gt_pts, tau=cfg.fscore_tau)
            log.info("  Chamfer ↓ %.5f   F-Score@%.3f ↑ %.4f", cd, cfg.fscore_tau, fs)
            rec.update({"chamfer": cd, "fscore": fs})

        # ---- PSNR (rendered novel views vs GT images) --------------------
        psnr_vals: list[float] = []
        for vi in range(N):
            view_c2w = input_c2ws[:, vi]
            view_K   = input_Ks[:, vi]
            with torch.no_grad():
                r = model.render_image(images, input_c2ws, input_Ks, view_c2w, view_K)

            hit      = r["hit"].cpu().numpy().astype(bool)
            pred_img = r["color"].cpu().numpy()                        # (H, W, 3)
            gt_img   = images[0, vi].permute(1, 2, 0).cpu().numpy()   # (H, W, 3)

            # Restrict to foreground pixels — background is trivially black on both
            # sides and would inflate PSNR by rewarding easy agreement.
            if hit.any():
                psnr_vals.append(psnr(pred_img[hit], gt_img[hit]))

        mean_psnr = float(np.mean(psnr_vals)) if psnr_vals else float("nan")
        log.info("  PSNR ↑ %.2f dB  (mean over %d views)", mean_psnr, len(psnr_vals))

        rec["psnr_db"]       = mean_psnr
        rec["psnr_per_view"] = psnr_vals
        rec["elapsed_s"]     = round(time.perf_counter() - t_start, 2)

        per_object.append(rec)

    # ---- Aggregate -------------------------------------------------------
    valid_cd   = [r["chamfer"] for r in per_object if r.get("chamfer") is not None]
    valid_fs   = [r["fscore"]  for r in per_object if r.get("fscore")  is not None]
    valid_psnr = [r["psnr_db"] for r in per_object
                  if r.get("psnr_db") is not None and not np.isnan(r["psnr_db"])]

    aggregate = {
        "n_objects":    len(per_object),
        "fscore_tau":   cfg.fscore_tau,
        "n_eval_pts":   cfg.n_eval_pts,
        "chamfer_mean": float(np.mean(valid_cd))   if valid_cd   else None,
        "chamfer_std":  float(np.std(valid_cd))    if valid_cd   else None,
        "fscore_mean":  float(np.mean(valid_fs))   if valid_fs   else None,
        "fscore_std":   float(np.std(valid_fs))    if valid_fs   else None,
        "psnr_mean_db": float(np.mean(valid_psnr)) if valid_psnr else None,
        "psnr_std_db":  float(np.std(valid_psnr))  if valid_psnr else None,
    }

    output = {"aggregate": aggregate, "per_object": per_object}
    out_file = out_dir / "metrics.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    log.info("Saved → %s", out_file)

    # ---- Summary table ---------------------------------------------------
    w = 56
    print("\n" + "─" * w)
    print("  Aggregate Results")
    print("─" * w)
    print(f"  Objects evaluated : {aggregate['n_objects']}")
    print(f"  Point samples     : {aggregate['n_eval_pts']:,}")
    if aggregate["chamfer_mean"] is not None:
        print(f"  Chamfer ↓         : {aggregate['chamfer_mean']:.5f} ± {aggregate['chamfer_std']:.5f}")
    else:
        print("  Chamfer           : n/a  (GT meshes not found)")
    if aggregate["fscore_mean"] is not None:
        print(f"  F-Score@{cfg.fscore_tau:.3f} ↑   : {aggregate['fscore_mean']:.4f} ± {aggregate['fscore_std']:.4f}")
    else:
        print("  F-Score           : n/a")
    if aggregate["psnr_mean_db"] is not None:
        print(f"  PSNR ↑            : {aggregate['psnr_mean_db']:.2f} ± {aggregate['psnr_std_db']:.2f} dB")
    else:
        print("  PSNR              : n/a")
    print("─" * w + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Quantitative evaluation: Chamfer, F-Score, PSNR."
    )
    parser.add_argument("--n_objects",  type=int, default=None,
                        help="Cap the number of test objects (default: all).")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--out_dir",    type=str, default="eval_results/metrics")
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--data_root",  type=str, default=None)
    parser.add_argument("--overfit",    action="store_true",
                        help="Evaluate the single object used during --overfit training.")
    parser.add_argument("--overfit_object", type=str, default=None, metavar="ID")
    args = parser.parse_args()

    cfg = PRISMConfig()
    for k in ("checkpoint", "image_size", "data_root"):
        if getattr(args, k) is not None:
            setattr(cfg, k, getattr(args, k))

    restrict: list[str] | None = None
    if args.overfit:
        pairs = enumerate_object_pairs(cfg.data_root)
        if not pairs:
            raise RuntimeError(f"No objects found under data_root={cfg.data_root!r}")
        oid = args.overfit_object or pairs[0][1]
        if not any(o[1] == oid for o in pairs):
            raise RuntimeError(f"--overfit_object {oid!r} not found")
        restrict = [oid]
        log.info("--overfit: evaluating %s", oid)

    evaluate(
        cfg,
        n_objects=args.n_objects,
        out_dir=Path(args.out_dir),
        restrict_object_ids=restrict,
    )
