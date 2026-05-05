"""
Render GT vs PRISM side-by-side comparison images.

For each test object: loads one view, runs model.render_image(), and saves a
4-panel PNG (GT image | predicted color | predicted depth | predicted normals).

python visualize.py [--n_objects N] [--checkpoint model.pt] [--out_dir eval_results/visuals]
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from config import PRISMConfig
from prism  import PRISM
from data.omniobject3d import OmniObject3DDataset

log = logging.getLogger("visualize")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")


def to_uint8(t: torch.Tensor) -> np.ndarray:
    return (t.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)


def depth_to_rgb(depth: torch.Tensor) -> np.ndarray:
    """Normalise depth to [0,1] and apply a simple colormap (near=bright, far=dark)."""
    d = depth.cpu().float()
    lo, hi = d[d > 0].min() if (d > 0).any() else d.min(), d.max()
    d = ((d - lo) / (hi - lo + 1e-6)).clamp(0, 1)
    d = 1.0 - d                         # near = white, far = dark
    arr = (d.numpy() * 255).astype(np.uint8)
    return np.stack([arr, arr, arr], axis=-1)


def normal_to_rgb(normal: torch.Tensor) -> np.ndarray:
    """Map normals from [-1, 1] to [0, 255]."""
    return to_uint8((normal * 0.5 + 0.5))


def make_panel(images: list[np.ndarray], labels: list[str]) -> Image.Image:
    """Stack H×W×3 uint8 arrays horizontally with labels."""
    H, W = images[0].shape[:2]
    label_h = 20
    total_w = W * len(images)
    canvas = np.ones((H + label_h, total_w, 3), dtype=np.uint8) * 240

    for i, (img, label) in enumerate(zip(images, labels)):
        canvas[label_h:, i*W:(i+1)*W] = img

    pil = Image.fromarray(canvas)
    try:
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(pil)
        for i, label in enumerate(labels):
            draw.text((i * W + 4, 2), label, fill=(30, 30, 30))
    except Exception:
        pass
    return pil


def visualize(cfg: PRISMConfig, n_objects: int | None, out_dir: Path):
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

        log.info("Rendering %s …", obj_id)
        rendered = model.render_image(image, c2w, K)

        gt_rgb   = to_uint8(image[0].permute(1, 2, 0))
        pred_rgb = to_uint8(rendered["color"])
        pred_d   = depth_to_rgb(rendered["depth"])
        pred_n   = normal_to_rgb(rendered["normal"])

        panel = make_panel(
            [gt_rgb, pred_rgb, pred_d, pred_n],
            ["GT image", "Predicted color", "Predicted depth", "Predicted normals"],
        )
        panel.save(out_dir / f"{obj_id}.png")
        log.info("  saved %s", out_dir / f"{obj_id}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_objects",  type=int, default=10)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--out_dir",    type=str, default="eval_results/visuals")
    parser.add_argument("--image_size", type=int)
    parser.add_argument("--data_root",  type=str)
    args = parser.parse_args()

    cfg = PRISMConfig()
    for k in ("checkpoint", "image_size", "data_root"):
        if getattr(args, k) is not None:
            setattr(cfg, k, getattr(args, k))

    visualize(cfg, n_objects=args.n_objects, out_dir=Path(args.out_dir))
