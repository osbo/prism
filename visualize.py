"""
Render GT vs PRISM comparison images.

For each test object: loads one view, runs model.render_image(), and saves a
multi-row 16:9 dashboard, including object/background separation.

By default, also export the **neural SDF** as a mesh (marching cubes) for 3-D preview in
VS Code / Cursor:

  • Install an extension such as **“3D Viewer”** (Microsoft) or **“glTF Viewer”**.
  • Writes ``.obj`` or ``.glb`` (see ``--mesh-dir`` / ``--mesh-format``).
  • Pass ``--no-export-mesh`` to skip mesh export.

python visualize.py [--n_objects N] [--checkpoint model.pt] [--out_dir eval_results/visuals]
python visualize.py --mesh-format obj
python visualize.py --no-export-mesh
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

import trimesh

from config import PRISMConfig
from prism import PRISM
from prism.mesh_extract import extract_sdf_mesh
from data.omniobject3d import OmniObject3DDataset

log = logging.getLogger("visualize")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")


def to_uint8(t: torch.Tensor) -> np.ndarray:
    return (t.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)


def depth_to_rgb(depth: torch.Tensor, opacity: torch.Tensor, alpha_thresh: float = 0.2) -> np.ndarray:
    """Visualize depth only on confident (non-background) pixels."""
    d = depth.cpu().float()
    a = opacity.cpu().float()
    fg = a > alpha_thresh
    if fg.any():
        lo, hi = d[fg].min(), d[fg].max()
    else:
        lo, hi = d.min(), d.max()
    d = ((d - lo) / (hi - lo + 1e-6)).clamp(0, 1)
    d = 1.0 - d                         # near = white, far = dark
    d[~fg] = 0.0
    arr = (d.numpy() * 255).astype(np.uint8)
    return np.stack([arr, arr, arr], axis=-1)


def normal_to_rgb(normal: torch.Tensor, opacity: torch.Tensor, alpha_thresh: float = 0.2) -> np.ndarray:
    """Map normals to RGB and hide low-opacity background."""
    n = (normal * 0.5 + 0.5).clamp(0, 1)
    fg = (opacity.cpu().float() > alpha_thresh)
    n = n.cpu()
    n[~fg] = 0.0
    return to_uint8(n)

def opacity_to_rgb(opacity: torch.Tensor) -> np.ndarray:
    a = opacity.cpu().float().clamp(0, 1)
    arr = (a.numpy() * 255).astype(np.uint8)
    return np.stack([arr, arr, arr], axis=-1)


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    m = (mask.astype(np.uint8) * 255)
    return np.stack([m, m, m], axis=-1)


def mask_diff_rgb(gt_mask: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
    """TP=green, FP=red, FN=blue."""
    tp = pred_mask & gt_mask
    fp = pred_mask & (~gt_mask)
    fn = (~pred_mask) & gt_mask
    out = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
    out[tp] = np.array([0, 220, 80], dtype=np.uint8)
    out[fp] = np.array([220, 40, 40], dtype=np.uint8)
    out[fn] = np.array([60, 120, 255], dtype=np.uint8)
    return out


def make_panel_16x9(
    images: list[np.ndarray],
    labels: list[str],
    out_w: int = 1600,
    out_h: int = 900,
    n_cols: int = 4,
) -> Image.Image:
    """Grid dashboard with target 16:9 output aspect."""
    n = len(images)
    n_rows = int(np.ceil(n / n_cols))
    pad = 12
    label_h = 24
    cell_w = (out_w - pad * (n_cols + 1)) // n_cols
    cell_h = (out_h - pad * (n_rows + 1)) // n_rows
    img_h = max(1, cell_h - label_h)

    canvas = Image.new("RGB", (out_w, out_h), color=(238, 238, 238))
    draw = ImageDraw.Draw(canvas)

    for i, (arr, label) in enumerate(zip(images, labels)):
        r, c = divmod(i, n_cols)
        x0 = pad + c * (cell_w + pad)
        y0 = pad + r * (cell_h + pad)
        tile = Image.fromarray(arr).resize((cell_w, img_h), Image.Resampling.BILINEAR)
        canvas.paste(tile, (x0, y0 + label_h))
        draw.text((x0 + 4, y0 + 3), label, fill=(24, 24, 24))
    return canvas


def _default_mesh_dir(out_dir: Path) -> Path:
    return out_dir.parent / "meshes"


def visualize(
    cfg: PRISMConfig,
    n_objects: int | None,
    out_dir: Path,
    export_mesh: bool = False,
    mesh_dir: Path | None = None,
    mesh_format: str = "obj",
    alpha_thresh: float = 0.2,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PRISM(cfg).to(device)
    ckpt  = torch.load(cfg.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    test_ds = OmniObject3DDataset(cfg.data_root, split="test", image_size=cfg.image_size)
    loader  = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)
    log.info("Rendering %d test objects → %s", len(test_ds), out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, batch in enumerate(loader):
        if n_objects and i >= n_objects:
            break

        obj_id = batch["object_id"][0]
        image  = batch["image"].to(device)
        c2w    = batch["c2w"].to(device)
        K      = batch["K"].to(device)
        gt_mask = (batch["mask"][0, 0].cpu().numpy() > 0.5)

        log.info("Rendering %s …", obj_id)
        rendered = model.render_image(image, c2w, K)

        if export_mesh:
            mdir = mesh_dir if mesh_dir is not None else _default_mesh_dir(out_dir)
            mdir.mkdir(parents=True, exist_ok=True)
            ext = mesh_format.lower()
            if ext not in {"obj", "glb"}:
                ext = "obj"
            mesh_path = mdir / f"{obj_id}.{ext}"
            with torch.no_grad():
                z_latent = model.encoder(image)[0]
            mc = extract_sdf_mesh(
                model,
                z_latent,
                cfg,
                device,
                mask_hw=batch["mask"][0],
                c2w=batch["c2w"].to(device),
                K=batch["K"].to(device),
            )
            if mc is None:
                log.warning("  no SDF iso-surface for %s (skipped mesh)", obj_id)
            else:
                verts, faces = mc
                trimesh.Trimesh(vertices=verts, faces=faces, process=False).export(
                    str(mesh_path)
                )
                log.info("  saved mesh → %s (open in a 3D viewer extension)", mesh_path)

        gt_rgb   = to_uint8(image[0].permute(1, 2, 0))
        pred_rgb = to_uint8(rendered["color"])
        pred_d   = depth_to_rgb(rendered["depth"], rendered["opacity"], alpha_thresh=alpha_thresh)
        pred_n   = normal_to_rgb(rendered["normal"], rendered["opacity"], alpha_thresh=alpha_thresh)
        pred_o   = opacity_to_rgb(rendered["opacity"])
        pred_mask = (rendered["opacity"].cpu().numpy() > alpha_thresh)
        gt_mask_rgb = mask_to_rgb(gt_mask)
        pred_mask_rgb = mask_to_rgb(pred_mask)
        sep_rgb = mask_diff_rgb(gt_mask, pred_mask)

        # Black where GT marks background (same silhouette assumption as mesh carving).
        pred_on_gt_mask = pred_rgb.copy()
        pred_on_gt_mask[~gt_mask] = 0
        pred_bg = pred_rgb.copy()
        pred_bg[gt_mask] = 0

        panel = make_panel_16x9(
            [
                gt_rgb, gt_mask_rgb, pred_rgb, pred_o,
                pred_on_gt_mask, pred_bg, pred_d, sep_rgb,
            ],
            [
                "GT image", "GT object mask", "Predicted color", "Predicted opacity",
                "Pred (GT mask, bg=black)", "Pred on GT bg only", "Pred depth (FG)", "Mask diff TP/FP/FN",
            ],
        )
        panel.save(out_dir / f"{obj_id}.png")
        log.info("  saved %s", out_dir / f"{obj_id}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_objects",  type=int, default=1)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--out_dir",    type=str, default="eval_results/visuals")
    parser.add_argument("--image_size", type=int)
    parser.add_argument("--data_root",  type=str)
    parser.add_argument(
        "--export-mesh",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Marching cubes on neural SDF → mesh (OBJ/GLB). Default: on; use --no-export-mesh to skip.",
    )
    parser.add_argument(
        "--mesh_dir",
        type=str,
        default=None,
        help="Mesh output folder (default: sibling of --out_dir, e.g. eval_results/meshes).",
    )
    parser.add_argument(
        "--mesh_format",
        type=str,
        choices=["obj", "glb"],
        default="obj",
        help="File format for exported mesh (both work with common VS Code 3D extensions).",
    )
    parser.add_argument(
        "--alpha_thresh",
        type=float,
        default=0.2,
        help="Opacity threshold used for foreground/background separation in visualizations.",
    )
    args = parser.parse_args()

    cfg = PRISMConfig()
    for k in ("checkpoint", "image_size", "data_root"):
        if getattr(args, k) is not None:
            setattr(cfg, k, getattr(args, k))

    mesh_dir_arg = Path(args.mesh_dir) if args.mesh_dir else None
    visualize(
        cfg,
        n_objects=args.n_objects,
        out_dir=Path(args.out_dir),
        export_mesh=args.export_mesh,
        mesh_dir=mesh_dir_arg,
        mesh_format=args.mesh_format,
        alpha_thresh=args.alpha_thresh,
    )
