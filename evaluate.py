"""
Extract SDF mesh via marching cubes and compute Chamfer + F-score vs GT.

python evaluate.py [--data_root ...] [--checkpoint model.pt] [--n_objects N]
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import PRISMConfig
from prism  import PRISM
from data.omniobject3d import OmniObject3DDataset

log = logging.getLogger("evaluate")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")


# ---------------------------------------------------------------------------
# Marching cubes mesh extraction
# ---------------------------------------------------------------------------

def extract_mesh(model, z, cfg, device):
    """Marching cubes over a uniform grid in [-bound, bound]^3."""
    try:
        from skimage.measure import marching_cubes
    except ImportError:
        raise ImportError("pip install scikit-image")

    res   = cfg.mc_resolution
    bound = cfg.mc_bound
    lin   = torch.linspace(-bound, bound, res, device=device)
    gx, gy, gz = torch.meshgrid(lin, lin, lin, indexing="ij")   # (res, res, res)
    pts   = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)

    chunk    = 32768
    sdf_vals = []
    z_single = z.unsqueeze(0).expand(chunk, -1)

    with torch.no_grad():
        for i in range(0, pts.shape[0], chunk):
            p  = pts[i:i+chunk]
            zc = z_single[:p.shape[0]]
            sdf_vals.append(model.sdf_mlp(p, zc).squeeze(-1))

    sdf_grid = torch.cat(sdf_vals).reshape(res, res, res).cpu().numpy()

    if sdf_grid.min() >= cfg.mc_threshold or sdf_grid.max() <= cfg.mc_threshold:
        return None

    verts, faces, *_ = marching_cubes(sdf_grid, level=cfg.mc_threshold)
    # Map from grid coords [0, res-1] to world coords [-bound, bound]
    verts = verts / (res - 1) * (2 * bound) - bound
    return verts.astype(np.float32), faces


# ---------------------------------------------------------------------------
# Chamfer distance + F-score
# ---------------------------------------------------------------------------

def sample_surface(verts, faces, n_points):
    """Uniformly sample points from a triangle mesh surface."""
    import trimesh
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    pts, _ = trimesh.sample.sample_surface(mesh, n_points)
    return pts.astype(np.float32)


def chamfer_fscore(pred_pts, gt_pts, tau):
    p = torch.from_numpy(pred_pts).float()
    g = torch.from_numpy(gt_pts).float()

    chunk = 4096
    d_pg, d_gp = [], []
    for i in range(0, p.shape[0], chunk):
        d_pg.append(((p[i:i+chunk, None] - g[None]) ** 2).sum(-1).min(-1).values)
    for i in range(0, g.shape[0], chunk):
        d_gp.append(((g[i:i+chunk, None] - p[None]) ** 2).sum(-1).min(-1).values)

    d_pg = torch.cat(d_pg).sqrt()
    d_gp = torch.cat(d_gp).sqrt()

    chamfer   = (d_pg.mean() + d_gp.mean()).item() / 2
    precision = (d_pg <= tau).float().mean().item()
    recall    = (d_gp <= tau).float().mean().item()
    fscore    = 2 * precision * recall / (precision + recall + 1e-8)
    return chamfer, fscore, precision, recall


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate(cfg: PRISMConfig, n_objects=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PRISM(cfg).to(device)
    ckpt  = torch.load(cfg.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    test_ds = OmniObject3DDataset(cfg.data_root, split="test", image_size=cfg.image_size)
    loader  = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)
    n_split = len(test_ds)
    n_run = min(n_split, n_objects) if n_objects is not None else n_split
    if n_objects is not None:
        log.info("Evaluating %d of %d test objects (--n_objects=%d)", n_run, n_split, n_objects)
    else:
        log.info("Evaluating all %d test objects", n_split)

    results = []
    for i, batch in enumerate(loader):
        if n_objects and i >= n_objects:
            break

        obj_id    = batch["object_id"][0]
        image     = batch["image"].to(device)
        mesh_path = batch["mesh_path"][0]

        with torch.no_grad():
            z = model.encoder(image)

        mc = extract_mesh(model, z[0], cfg, device)
        if mc is None:
            log.warning("No surface found for %s — skipping", obj_id)
            continue

        pred_verts, pred_faces = mc
        try:
            import trimesh
            gt_mesh   = trimesh.load(mesh_path, force="mesh")
            gt_pts, _ = trimesh.sample.sample_surface(gt_mesh, cfg.n_eval_pts)
            gt_pts    = gt_pts.astype(np.float32)
        except Exception as e:
            log.warning("Could not load GT mesh for %s: %s", obj_id, e)
            continue

        pred_pts = sample_surface(pred_verts, pred_faces, cfg.n_eval_pts)

        # Normalize both point clouds to [-1, 1] before metric computation.
        # GT raw scans and Blender render space use different coordinate systems /
        # scales, so raw Chamfer is meaningless.  Each cloud is independently
        # centered and scaled so max |coord| = 1.
        def _normalize(pts):
            pts = pts - pts.mean(0)
            scale = np.abs(pts).max() + 1e-8
            return pts / scale

        pred_pts_n = _normalize(pred_pts)
        gt_pts_n   = _normalize(gt_pts)
        chamfer, fscore, prec, rec = chamfer_fscore(pred_pts_n, gt_pts_n, cfg.fscore_tau)

        log.info("%s  chamfer=%.5f  F@%.3f=%.4f", obj_id, chamfer, cfg.fscore_tau, fscore)
        results.append({"object_id": obj_id, "chamfer": chamfer,
                         "fscore": fscore, "precision": prec, "recall": rec})

    if not results:
        log.error("No objects evaluated")
        return

    mean_c = np.mean([r["chamfer"] for r in results])
    mean_f = np.mean([r["fscore"]  for r in results])
    log.info("\n=== %d objects ===  mean Chamfer %.5f  mean F@%.3f %.4f",
             len(results), mean_c, cfg.fscore_tau, mean_f)

    out = Path("eval_results")
    out.mkdir(exist_ok=True)
    with open(out / "results.json", "w") as f:
        json.dump({"mean_chamfer": mean_c, "mean_fscore": mean_f,
                   "tau": cfg.fscore_tau, "per_object": results}, f, indent=2)
    log.info("Saved to eval_results/results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",  type=str)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--n_objects",  type=int)
    parser.add_argument("--image_size", type=int)
    args = parser.parse_args()

    cfg = PRISMConfig()
    for k in ("data_root", "checkpoint", "image_size"):
        v = getattr(args, k)
        if v is not None:
            setattr(cfg, k, v)

    evaluate(cfg, n_objects=args.n_objects)
