"""
PRISM evaluation script.

Metrics (evaluated against raw_scans GT meshes):
  • Chamfer Distance    — bidirectional mean point-to-point L2
  • F-Score @ τ         — harmonic mean of precision/recall at threshold τ
  • PSNR               — on rendered novel views (same 24 Blender views)

Usage:
    python evaluate.py --checkpoint runs/<run_id>/best.pt \
                       --data_root /orcd/pool/007/osbo/omniobject3d \
                       [--split test] [--output_dir ./eval_results]
"""

import sys
import json
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from config import PRISMConfig
from prism import PRISM
from data.omniobject3d import OmniObject3DDataset

log = logging.getLogger("prism.eval")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")


# ---------------------------------------------------------------------------
# Mesh extraction from SDF via Marching Cubes
# ---------------------------------------------------------------------------

def extract_mesh(
    model: PRISM,
    z: torch.Tensor,       # (1, latent_dim)
    resolution: int = 256,
    threshold: float = 0.0,
    bound: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run Marching Cubes on a dense SDF grid extracted from the model.
    Returns (vertices: (V,3), faces: (F,3)) as numpy arrays.
    """
    import skimage.measure as skm

    device = z.device
    # Build a regular grid of query points in [-bound, bound]³
    lin = torch.linspace(-bound, bound, resolution, device=device)
    grid_x, grid_y, grid_z = torch.meshgrid(lin, lin, lin, indexing="ij")
    pts = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)  # (R³, 3)

    # Query SDF in chunks to avoid OOM
    chunk = 65536
    sdf_vals = []
    z_exp = z.expand(chunk, -1)
    with torch.no_grad():
        for start in range(0, pts.shape[0], chunk):
            end = min(start + chunk, pts.shape[0])
            p = pts[start:end]
            z_chunk = z.expand(p.shape[0], -1)
            sdf_vals.append(model.sdf_mlp(p, z_chunk).squeeze(-1).cpu().float())
    sdf_grid = torch.cat(sdf_vals).reshape(resolution, resolution, resolution).numpy()

    # Marching Cubes
    verts, faces, normals, _ = skm.marching_cubes(
        sdf_grid, level=threshold, spacing=[2 * bound / (resolution - 1)] * 3
    )
    # Shift from [0, 2*bound] to [-bound, bound]
    verts = verts - bound

    return verts.astype(np.float32), faces.astype(np.int32)


# ---------------------------------------------------------------------------
# Chamfer Distance
# ---------------------------------------------------------------------------

def chamfer_distance(
    pts_pred: np.ndarray,   # (N, 3)
    pts_gt:   np.ndarray,   # (M, 3)
) -> float:
    """
    Mean bidirectional point-to-point Chamfer distance.
    Uses open3d's KD-tree for efficiency.
    """
    import open3d as o3d

    pcd_p = o3d.geometry.PointCloud()
    pcd_p.points = o3d.utility.Vector3dVector(pts_pred)
    pcd_g = o3d.geometry.PointCloud()
    pcd_g.points = o3d.utility.Vector3dVector(pts_gt)

    d_p_to_g = np.asarray(pcd_p.compute_point_cloud_distance(pcd_g))
    d_g_to_p = np.asarray(pcd_g.compute_point_cloud_distance(pcd_p))

    return float((d_p_to_g.mean() + d_g_to_p.mean()) / 2.0)


# ---------------------------------------------------------------------------
# F-Score @ τ
# ---------------------------------------------------------------------------

def fscore(
    pts_pred: np.ndarray,
    pts_gt:   np.ndarray,
    tau: float = 0.01,
) -> tuple[float, float, float]:
    """
    F-Score at distance threshold τ.
    Returns (precision, recall, f_score).
    """
    import open3d as o3d

    pcd_p = o3d.geometry.PointCloud()
    pcd_p.points = o3d.utility.Vector3dVector(pts_pred)
    pcd_g = o3d.geometry.PointCloud()
    pcd_g.points = o3d.utility.Vector3dVector(pts_gt)

    d_p_to_g = np.asarray(pcd_p.compute_point_cloud_distance(pcd_g))
    d_g_to_p = np.asarray(pcd_g.compute_point_cloud_distance(pcd_p))

    precision = float((d_p_to_g < tau).mean())
    recall    = float((d_g_to_p < tau).mean())
    if precision + recall < 1e-8:
        return 0.0, 0.0, 0.0
    f = 2 * precision * recall / (precision + recall)
    return precision, recall, float(f)


# ---------------------------------------------------------------------------
# PSNR
# ---------------------------------------------------------------------------

def psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """PSNR between two (H, W, 3) tensors in [0, 1]."""
    mse = F.mse_loss(pred, gt).item()
    if mse < 1e-10:
        return 100.0
    return float(10.0 * np.log10(1.0 / mse))


# ---------------------------------------------------------------------------
# Sample points from mesh
# ---------------------------------------------------------------------------

def sample_points_from_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    n_samples: int = 100_000,
) -> np.ndarray:
    """Uniformly sample points from mesh surface using area-weighted triangle sampling."""
    import trimesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    pts, _ = trimesh.sample.sample_surface(mesh, n_samples)
    return pts.astype(np.float32)


def load_gt_mesh(obj_path: str) -> tuple[np.ndarray, np.ndarray]:
    import trimesh
    mesh = trimesh.load(obj_path, force="mesh")
    return np.asarray(mesh.vertices, dtype=np.float32), np.asarray(mesh.faces, dtype=np.int32)


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    checkpoint_path: str,
    data_root: str,
    split: str = "test",
    output_dir: str = "./eval_results",
    n_point_samples: int = 100_000,
    mc_resolution: int = 256,
    fscore_tau: float = 0.01,
    device_str: str = "cuda",
):
    device = torch.device(device_str)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    cfg = PRISMConfig()
    model = PRISM(cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    log.info(f"Loaded checkpoint from {checkpoint_path}")

    # Dataset
    dataset = OmniObject3DDataset(data_root=data_root, split=split)
    log.info(f"Evaluating {len(dataset)} objects from split='{split}'")

    results = []

    for idx in tqdm(range(len(dataset)), desc="Evaluating"):
        sample = dataset[idx]
        obj_id = sample["object_id"]
        cat    = sample["category"]

        image  = sample["image"].unsqueeze(0).to(device)   # (1, 3, H, W)
        c2w    = sample["c2w"].unsqueeze(0).to(device)
        K      = sample["K"].unsqueeze(0).to(device)

        # ---- Encode → z ----
        with torch.no_grad():
            image_norm = (image - model.img_mean) / model.img_std
            z = model.encoder(image_norm)   # (1, latent_dim)

        # ---- Geometry: extract mesh via Marching Cubes ----
        verts_pred, faces_pred = extract_mesh(
            model, z, resolution=mc_resolution, threshold=0.0
        )

        # ---- Load GT mesh ----
        if not Path(sample["mesh_path"]).exists():
            log.warning(f"GT mesh not found: {sample['mesh_path']}, skipping geometry metrics.")
            cd, prec, rec, fs = None, None, None, None
        else:
            verts_gt, faces_gt = load_gt_mesh(sample["mesh_path"])

            # Normalise GT mesh to [-1, 1] bounding box (same as predicted)
            centre = (verts_gt.max(0) + verts_gt.min(0)) / 2
            scale  = (verts_gt.max(0) - verts_gt.min(0)).max() / 2
            verts_gt = (verts_gt - centre) / (scale + 1e-8)

            pts_pred = sample_points_from_mesh(verts_pred, faces_pred, n_point_samples)
            pts_gt   = sample_points_from_mesh(verts_gt,   faces_gt,   n_point_samples)

            cd = chamfer_distance(pts_pred, pts_gt)
            prec, rec, fs = fscore(pts_pred, pts_gt, tau=fscore_tau)

        # ---- Appearance: render and compute PSNR ----
        rendered = model.render_image(image, c2w, K)
        gt_img   = sample["image"].to(device)   # (3, H, W)

        pred_rgb = rendered["colour"].clamp(0, 1)         # (H, W, 3)
        gt_rgb   = gt_img.permute(1, 2, 0)               # (H, W, 3)
        psnr_val = psnr(pred_rgb, gt_rgb)

        entry = {
            "category": cat,
            "object_id": obj_id,
            "psnr":      psnr_val,
            "chamfer":   cd,
            "precision": prec,
            "recall":    rec,
            "fscore":    fs,
        }
        results.append(entry)
        log.info(
            f"  [{idx+1}/{len(dataset)}] {obj_id}  "
            f"PSNR={psnr_val:.2f}  "
            f"CD={cd:.5f}" if cd else f"PSNR={psnr_val:.2f}"
        )

    # ---- Aggregate ----
    valid_cd = [r["chamfer"] for r in results if r["chamfer"] is not None]
    valid_fs = [r["fscore"]  for r in results if r["fscore"]  is not None]
    psnr_vals = [r["psnr"] for r in results]

    summary = {
        "n_objects":       len(results),
        "mean_psnr":       float(np.mean(psnr_vals)),
        "mean_chamfer":    float(np.mean(valid_cd))  if valid_cd else None,
        "mean_fscore":     float(np.mean(valid_fs))  if valid_fs else None,
        "fscore_tau":      fscore_tau,
    }

    log.info("=" * 60)
    log.info(f"Mean PSNR:      {summary['mean_psnr']:.3f} dB")
    log.info(f"Mean Chamfer:   {summary['mean_chamfer']:.5f}"  if summary['mean_chamfer'] else "Mean Chamfer:   N/A")
    log.info(f"Mean F-Score:   {summary['mean_fscore']:.4f} @ τ={fscore_tau}" if summary['mean_fscore'] else "Mean F-Score:   N/A")
    log.info("=" * 60)

    # Save
    out_file = output_dir / "results.json"
    with open(out_file, "w") as f:
        json.dump({"summary": summary, "per_object": results}, f, indent=2)
    log.info(f"Results saved to {out_file}")
    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PRISM")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_root",  type=str, required=True)
    parser.add_argument("--split",      type=str, default="test")
    parser.add_argument("--output_dir", type=str, default="./eval_results")
    parser.add_argument("--mc_resolution", type=int, default=256)
    parser.add_argument("--fscore_tau",    type=float, default=0.01)
    parser.add_argument("--n_point_samples", type=int, default=100_000)
    parser.add_argument("--device",     type=str, default="cuda")
    args = parser.parse_args()

    evaluate(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        split=args.split,
        output_dir=args.output_dir,
        mc_resolution=args.mc_resolution,
        fscore_tau=args.fscore_tau,
        n_point_samples=args.n_point_samples,
        device_str=args.device,
    )
