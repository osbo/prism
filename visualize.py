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


def color_to_uint8(color: torch.Tensor, opacity: torch.Tensor, alpha_thresh: float = 0.2) -> np.ndarray:
    """Predicted RGB as uint8; pixels with opacity ≤ thresh are black (same FG rule as depth)."""
    c = color.clamp(0, 1).cpu().numpy()
    fg = (opacity.cpu().float().numpy() > alpha_thresh)[..., None]
    c = np.where(fg, c, 0.0)
    return (c * 255).astype(np.uint8)


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


def compute_fg_mask(rgb_uint8: np.ndarray) -> np.ndarray:
    """True = foreground. Background = dark pixels connected to the image border."""
    dark = rgb_uint8.astype(np.int32).sum(-1) <= 10
    try:
        from scipy.ndimage import label as nd_label
        labeled, _ = nd_label(dark)
        border_lbls: set = set()
        for arr in (labeled[0], labeled[-1], labeled[:, 0], labeled[:, -1]):
            border_lbls.update(arr.tolist())
        border_lbls.discard(0)
        bg = np.zeros(dark.shape, dtype=bool)
        for lbl in border_lbls:
            bg |= labeled == lbl
        return ~bg
    except ImportError:
        return ~dark


def make_rows_grid(
    rows: "list[list[np.ndarray]]",
    labels: "list[list[str]]",
    cell_w: int = 240,
    label_h: int = 18,
    pad: int = 6,
) -> Image.Image:
    """One row per view, fixed columns. Each cell is cell_w × cell_w pixels."""
    n_rows = len(rows)
    n_cols = max(len(r) for r in rows)
    cell_h = cell_w
    out_w = pad + n_cols * (cell_w + pad)
    out_h = pad + n_rows * (cell_h + label_h + pad)
    canvas = Image.new("RGB", (out_w, out_h), color=(238, 238, 238))
    draw = ImageDraw.Draw(canvas)
    for ri, (row_panels, row_lbls) in enumerate(zip(rows, labels)):
        for ci, (arr, lbl) in enumerate(zip(row_panels, row_lbls)):
            x0 = pad + ci * (cell_w + pad)
            y0 = pad + ri * (cell_h + label_h + pad)
            tile = Image.fromarray(arr).resize((cell_w, cell_h), Image.Resampling.BILINEAR)
            canvas.paste(tile, (x0, y0 + label_h))
            draw.text((x0 + 2, y0 + 2), lbl, fill=(24, 24, 24))
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

    test_ds = OmniObject3DDataset(cfg.data_root, split="test", image_size=cfg.image_size, n_input_views=cfg.n_input_views)
    loader  = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)
    log.info("Rendering %d test objects → %s", len(test_ds), out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, batch in enumerate(loader):
        if n_objects and i >= n_objects:
            break

        obj_id      = batch["object_id"][0]
        images      = batch["images"].to(device)          # (1, N, 3, H, W)
        input_c2ws  = batch["input_c2ws"].to(device)      # (1, N, 4, 4)
        input_Ks    = batch["input_Ks"].to(device)        # (1, N, 3, 3)
        N = images.shape[1]

        log.info("Rendering %s (%d views) …", obj_id, N)

        if export_mesh:
            mdir = mesh_dir if mesh_dir is not None else _default_mesh_dir(out_dir)
            mdir.mkdir(parents=True, exist_ok=True)
            ext = mesh_format.lower()
            if ext not in {"obj", "glb"}:
                ext = "obj"
            mesh_path = mdir / f"{obj_id}.{ext}"
            with torch.no_grad():
                z_latent = model.encoder(images)[0]
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

        # One row per input view: GT image | GT mask | pred color | pred opacity | pred depth | mask diff
        grid_rows:  list[list[np.ndarray]] = []
        grid_labels: list[list[str]] = []
        for vi in range(N):
            gt_rgb  = to_uint8(images[0, vi].permute(1, 2, 0))
            gt_mask = compute_fg_mask(gt_rgb)
            gt_mask_rgb = mask_to_rgb(gt_mask)

            view_c2w = input_c2ws[:, vi]   # (1, 4, 4)
            view_K   = input_Ks[:, vi]     # (1, 3, 3)
            with torch.no_grad():
                r = model.render_image(images, view_c2w, view_K)

            pred_rgb  = color_to_uint8(r["color"], r["opacity"], alpha_thresh=alpha_thresh)
            pred_op   = opacity_to_rgb(r["opacity"])
            pred_d    = depth_to_rgb(r["depth"], r["opacity"], alpha_thresh=alpha_thresh)
            pred_mask = r["opacity"].cpu().numpy() > alpha_thresh
            diff      = mask_diff_rgb(gt_mask, pred_mask)

            tag = f"v{vi}"
            grid_rows.append([gt_rgb, gt_mask_rgb, pred_rgb, pred_op, pred_d, diff])
            grid_labels.append([
                f"{tag} GT", f"{tag} mask",
                f"{tag} pred", f"{tag} opacity",
                f"{tag} depth", f"{tag} diff",
            ])

        panel = make_rows_grid(grid_rows, grid_labels)
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
