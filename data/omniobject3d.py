"""
OmniObject3D dataset loader for PRISM.

Expected on-disk layout (blender_renders_24_views):

    {data_root}/blender_renders_24_views/
        {category}/
            {object_id}/
                {view_id:04d}/          # 0000 … 0023
                    rgb.png             # 800×800 sRGB
                    depth.npy           # (800, 800) float32  metric depth in scene units
                    normal.npy          # (800, 800, 3) float32  world-space unit normals
                    camera.npz          # 'c2w': (4,4), 'K': (3,3)

    {data_root}/raw_scans/
        {category}/
            {object_id}/
                {object_id}.obj         # textured mesh (GT for evaluation)

The dataset returns one (object, view) pair per sample. During training we
randomly pick one of the 24 views as the input; the render loss is computed
against the same view.  (Multi-view consistency across views is a potential
future extension.)

If the exact file naming differs from the above, adjust the path helpers
(_rgb_path, _depth_path, etc.) at the top of this file.
"""

import os
import json
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


# ---------------------------------------------------------------------------
# Path helpers — edit these if the actual on-disk naming differs
# ---------------------------------------------------------------------------

def _rgb_path(obj_dir: Path, view_id: int) -> Path:
    return obj_dir / f"{view_id:04d}" / "rgb.png"

def _depth_path(obj_dir: Path, view_id: int) -> Path:
    return obj_dir / f"{view_id:04d}" / "depth.npy"

def _normal_path(obj_dir: Path, view_id: int) -> Path:
    return obj_dir / f"{view_id:04d}" / "normal.npy"

def _camera_path(obj_dir: Path, view_id: int) -> Path:
    return obj_dir / f"{view_id:04d}" / "camera.npz"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class OmniObject3DDataset(Dataset):
    """
    Args:
        data_root:    Path to the OmniObject3D root (contains blender_renders_24_views/)
        split:        'train' | 'val' | 'test'
        n_views:      number of Blender views available per object (default 24)
        image_size:   resize images to this square size (default 800 — no resize)
        split_file:   optional JSON file listing {split: [object_ids]}
                      if None, a deterministic 90/5/5 split is inferred from
                      the directory listing.
        categories:   optional list of category names to restrict to.
    """

    N_VIEWS = 24

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        n_views: int = 24,
        image_size: int = 800,
        split_file: Optional[str] = None,
        categories: Optional[list[str]] = None,
    ):
        super().__init__()
        self.data_root  = Path(data_root)
        self.render_root = self.data_root / "blender_renders_24_views"
        self.scan_root   = self.data_root / "raw_scans"
        self.split  = split
        self.n_views = n_views
        self.image_size = image_size

        # ImageNet normalisation (applied to the input image only)
        self.to_tensor   = transforms.ToTensor()           # PIL → [0,1] (C,H,W)
        self.img_resize  = (
            transforms.Resize((image_size, image_size), antialias=True)
            if image_size != 800 else None
        )

        # Enumerate all (category, object_id) pairs
        all_objects = self._enumerate_objects(categories)

        # Train / val / test split
        if split_file is not None:
            with open(split_file) as f:
                split_map = json.load(f)
            ids_in_split = set(split_map[split])
            self.objects = [o for o in all_objects if o[1] in ids_in_split]
        else:
            self.objects = self._auto_split(all_objects, split)

        if len(self.objects) == 0:
            raise RuntimeError(
                f"No objects found for split='{split}' under {self.render_root}. "
                "Check that the blender_renders_24_views directory exists and "
                "matches the expected structure."
            )

    # ------------------------------------------------------------------
    # Enumeration helpers
    # ------------------------------------------------------------------

    def _enumerate_objects(self, categories: Optional[list[str]]) -> list[tuple[str, str]]:
        objects = []
        if not self.render_root.exists():
            raise FileNotFoundError(f"Render root not found: {self.render_root}")
        for cat_dir in sorted(self.render_root.iterdir()):
            if not cat_dir.is_dir():
                continue
            if categories and cat_dir.name not in categories:
                continue
            for obj_dir in sorted(cat_dir.iterdir()):
                if not obj_dir.is_dir():
                    continue
                # Quick sanity: at least view 0000 must exist
                if _rgb_path(obj_dir, 0).exists():
                    objects.append((cat_dir.name, obj_dir.name))
        return objects

    def _auto_split(self, objects: list, split: str) -> list:
        """Deterministic 90 / 5 / 5 split using a fixed seed."""
        rng = random.Random(42)
        shuffled = objects[:]
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(0.90 * n)
        n_val   = int(0.05 * n)
        if split == "train":
            return shuffled[:n_train]
        elif split == "val":
            return shuffled[n_train:n_train + n_val]
        else:  # test
            return shuffled[n_train + n_val:]

    # ------------------------------------------------------------------
    # __len__ / __getitem__
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.objects)

    def __getitem__(self, idx: int) -> dict:
        cat, obj_id = self.objects[idx]
        obj_dir = self.render_root / cat / obj_id

        # Pick a random view
        view_id = random.randint(0, self.n_views - 1)

        # ---- RGB image ----
        rgb_img = Image.open(_rgb_path(obj_dir, view_id)).convert("RGB")
        if self.img_resize:
            rgb_img = self.img_resize(rgb_img)
        image = self.to_tensor(rgb_img)   # (3, H, W) in [0, 1]

        # ---- Depth map ----
        depth_np = np.load(_depth_path(obj_dir, view_id)).astype(np.float32)
        depth = torch.from_numpy(depth_np).unsqueeze(0)   # (1, H, W)

        # ---- Normal map ----
        normal_np = np.load(_normal_path(obj_dir, view_id)).astype(np.float32)
        normal = torch.from_numpy(normal_np).permute(2, 0, 1)  # (3, H, W)
        # Normals should already be unit vectors; re-normalise for safety
        normal = torch.nn.functional.normalize(normal, dim=0)

        # ---- Camera parameters ----
        cam = np.load(_camera_path(obj_dir, view_id))
        c2w = torch.from_numpy(cam["c2w"].astype(np.float32))  # (4, 4)
        K   = torch.from_numpy(cam["K"].astype(np.float32))    # (3, 3)

        # ---- GT mesh path (for eval only) ----
        mesh_path = str(self.scan_root / cat / obj_id / f"{obj_id}.obj")

        return {
            "image":    image,       # (3, H, W)
            "depth":    depth,       # (1, H, W)
            "normal":   normal,      # (3, H, W)
            "c2w":      c2w,         # (4, 4)
            "K":        K,           # (3, 3)
            "category": cat,
            "object_id": obj_id,
            "view_id":  view_id,
            "mesh_path": mesh_path,  # str (used in evaluate.py)
        }


# ---------------------------------------------------------------------------
# DataLoader builder
# ---------------------------------------------------------------------------

def build_dataloaders(
    data_root: str,
    batch_size: int,
    num_workers: int = 4,
    image_size: int = 800,
    split_file: Optional[str] = None,
    categories: Optional[list[str]] = None,
    pin_memory: bool = True,
) -> dict[str, DataLoader]:
    datasets = {
        split: OmniObject3DDataset(
            data_root=data_root,
            split=split,
            image_size=image_size,
            split_file=split_file,
            categories=categories,
        )
        for split in ("train", "val")
    }

    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
            persistent_workers=num_workers > 0,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            persistent_workers=num_workers > 0,
        ),
    }
    return loaders
