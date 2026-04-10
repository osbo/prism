"""
PRISM — Physics-Informed Reconstruction via Implicit Surfaces and Materials

Full feed-forward model that, given a single RGB image, produces:
  • A neural SDF representing object geometry
  • Cook-Torrance GGX BRDF parameters (albedo, roughness, metalness)
  • A point light position and intensity

These are jointly constrained by differentiable rendering: the predicted
geometry + materials + lighting must reproduce the input image.

Model forward pass:
    image  →  z = Encoder(image)
           →  BRDF params  = BRDFHead(z)
           →  Light params  = LightHead(z)
           →  [given camera rays]
           →  render_out   = Renderer(SDF_MLP(·; z), BRDF, Light, rays)
           →  losses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from config import PRISMConfig
from .encoder  import ImageEncoder
from .sdf_mlp  import SDFMLP
from .heads    import BRDFHead, LightHead
from .renderer import DifferentiableRenderer
from .loss     import PRISMLoss


def _sample_rays(
    camera_extrinsics: torch.Tensor,  # (B, 4, 4)  world-to-camera (or c2w)
    camera_intrinsics: torch.Tensor,  # (B, 3, 3)  K matrix
    image_hw: tuple[int, int],
    n_rays: int,
    device: torch.device,
    c2w: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample `n_rays` random pixel rays from B cameras.

    Returns:
        rays_o:    (B*n_rays, 3)  ray origins in world space
        rays_d:    (B*n_rays, 3)  unit ray directions in world space
        pixel_idx: (B*n_rays, 2)  (row, col) pixel coordinates per ray
    """
    H, W = image_hw
    B = camera_extrinsics.shape[0]

    # Random pixel coordinates  (B, n_rays)
    rows = torch.randint(0, H, (B, n_rays), device=device)
    cols = torch.randint(0, W, (B, n_rays), device=device)

    # Pixel centres in NDC: (u - cx) / fx, (v - cy) / fy
    fx = camera_intrinsics[:, 0, 0].unsqueeze(1)   # (B, 1)
    fy = camera_intrinsics[:, 1, 1].unsqueeze(1)
    cx = camera_intrinsics[:, 0, 2].unsqueeze(1)
    cy = camera_intrinsics[:, 1, 2].unsqueeze(1)

    u = (cols.float() - cx) / fx   # (B, n_rays)
    v = (rows.float() - cy) / fy

    # Directions in camera space (OpenCV convention: +z into scene)
    ones = torch.ones_like(u)
    dirs_cam = torch.stack([u, v, ones], dim=-1)   # (B, n_rays, 3)

    if c2w:
        R = camera_extrinsics[:, :3, :3]   # (B, 3, 3)  camera-to-world rotation
        t = camera_extrinsics[:, :3, 3]    # (B, 3)     camera position in world

        # Rotate directions to world space
        dirs_world = torch.einsum("bij,bkj->bki", R, dirs_cam)  # (B, n_rays, 3)
        dirs_world = F.normalize(dirs_world, dim=-1)

        # All rays from this camera share the same origin
        origins = t.unsqueeze(1).expand(B, n_rays, 3)  # (B, n_rays, 3)
    else:
        # world-to-camera: invert
        R = camera_extrinsics[:, :3, :3].transpose(-1, -2)
        t = -torch.einsum("bij,bj->bi", R, camera_extrinsics[:, :3, 3])
        dirs_world = torch.einsum("bij,bkj->bki", R, dirs_cam)
        dirs_world = F.normalize(dirs_world, dim=-1)
        origins = t.unsqueeze(1).expand(B, n_rays, 3)

    pixel_idx = torch.stack([rows, cols], dim=-1)   # (B, n_rays, 2)

    # Flatten batch dimension
    rays_o    = origins.reshape(B * n_rays, 3)
    rays_d    = dirs_world.reshape(B * n_rays, 3)
    pixel_idx = pixel_idx.reshape(B * n_rays, 2)

    return rays_o, rays_d, pixel_idx


def _gather_gt_rays(
    gt_image:  torch.Tensor,   # (B, 3, H, W)
    gt_depth:  torch.Tensor,   # (B, 1, H, W)
    gt_normal: torch.Tensor,   # (B, 3, H, W)
    pixel_idx: torch.Tensor,   # (B*R, 2)  (row, col)
    B: int,
    R: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather GT colour / depth / normal at the sampled pixel locations."""
    # pixel_idx: (B*R, 2)
    idx = pixel_idx.reshape(B, R, 2)
    rows = idx[:, :, 0]   # (B, R)
    cols = idx[:, :, 1]

    # Using advanced indexing: gather per-image
    batch_i = torch.arange(B, device=gt_image.device).unsqueeze(1).expand(B, R)

    gt_col = gt_image[batch_i, :, rows, cols]      # (B, R, 3)
    gt_dep = gt_depth[batch_i, 0, rows, cols]      # (B, R)
    gt_nor = gt_normal[batch_i, :, rows, cols]     # (B, R, 3)

    return (
        gt_col.reshape(B * R, 3),
        gt_dep.reshape(B * R),
        gt_nor.reshape(B * R, 3),
    )


# ---------------------------------------------------------------------------
# PRISM model
# ---------------------------------------------------------------------------

class PRISM(nn.Module):
    """
    Args:
        cfg: PRISMConfig
    """

    def __init__(self, cfg: PRISMConfig):
        super().__init__()
        self.cfg = cfg

        self.encoder  = ImageEncoder(cfg.encoder)
        self.sdf_mlp  = SDFMLP(cfg.sdf)
        self.brdf_head = BRDFHead(cfg.brdf)
        self.light_head = LightHead(cfg.light)
        self.renderer  = DifferentiableRenderer(cfg.renderer)
        self.loss_fn   = PRISMLoss(cfg.loss)

        # ImageNet normalisation constants (applied inside forward)
        self.register_buffer(
            "img_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "img_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

    # ------------------------------------------------------------------
    # SDF convenience wrapper (expands z to match point batch size)
    # ------------------------------------------------------------------
    def _make_sdf_fn(self, z: torch.Tensor, batch_idx: torch.Tensor):
        """Return a closure that queries the SDF MLP for a given z/batch."""
        def sdf_fn(pts: torch.Tensor) -> torch.Tensor:
            # pts: (N, 3)  batch_idx: (N,) mapping points to object in batch
            z_exp = z[batch_idx[:pts.shape[0]]]   # (N, latent_dim)
            return self.sdf_mlp(pts, z_exp)
        return sdf_fn

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        image:             torch.Tensor,   # (B, 3, H, W)  in [0, 1]
        camera_extrinsics: torch.Tensor,   # (B, 4, 4)
        camera_intrinsics: torch.Tensor,   # (B, 3, 3)
        gt_image:          torch.Tensor,   # (B, 3, H, W)
        gt_depth:          torch.Tensor,   # (B, 1, H, W)
        gt_normal:         torch.Tensor,   # (B, 3, H, W)
        n_rays: Optional[int] = None,
    ) -> dict:
        """
        Training forward pass: samples rays, renders them, and computes losses.

        Returns:
            losses: dict with 'total', 'render', 'depth', 'normal', 'eikonal'
            extras: dict with rendered output for visualisation / logging
        """
        B = image.shape[0]
        H, W = image.shape[2], image.shape[3]
        device = image.device
        n_rays = n_rays or self.cfg.renderer.n_rays_train

        # ----------------------------------------------------------------
        # Stage 1: Encode image → z
        # ----------------------------------------------------------------
        image_norm = (image - self.img_mean) / self.img_std
        z = self.encoder(image_norm)   # (B, latent_dim)

        # ----------------------------------------------------------------
        # Stage 2: Predict BRDF + Light
        # ----------------------------------------------------------------
        brdf  = self.brdf_head(z)    # {albedo, roughness, metalness}
        light = self.light_head(z)   # {light_pos, light_intensity}

        # ----------------------------------------------------------------
        # Sample rays and gather GT
        # ----------------------------------------------------------------
        rays_o, rays_d, pixel_idx = _sample_rays(
            camera_extrinsics, camera_intrinsics,
            image_hw=(H, W),
            n_rays=n_rays,
            device=device,
        )  # (B*n_rays, 3), (B*n_rays, 3), (B*n_rays, 2)

        gt_col, gt_dep, gt_nor = _gather_gt_rays(
            gt_image, gt_depth, gt_normal, pixel_idx, B, n_rays
        )

        # batch_idx: which object each ray belongs to
        batch_idx = torch.arange(B, device=device).repeat_interleave(n_rays)

        # ----------------------------------------------------------------
        # Stage 3: SDF fn closure (expands z per ray)
        # ----------------------------------------------------------------
        def sdf_fn(pts: torch.Tensor) -> torch.Tensor:
            N = pts.shape[0]
            z_exp = z[batch_idx[:N]]
            return self.sdf_mlp(pts, z_exp)

        # ----------------------------------------------------------------
        # Stage 4: Differentiable rendering
        # ----------------------------------------------------------------
        render_out = self.renderer(
            sdf_fn=sdf_fn,
            brdf=brdf,
            light=light,
            rays_o=rays_o,
            rays_d=rays_d,
            batch_idx=batch_idx,
        )

        # ----------------------------------------------------------------
        # Loss
        # ----------------------------------------------------------------
        losses = self.loss_fn(
            render_out=render_out,
            gt_colour=gt_col,
            gt_depth=gt_dep,
            gt_normal=gt_nor,
            sdf_fn=sdf_fn,
        )

        return losses, render_out, {"brdf": brdf, "light": light, "z": z}

    # ------------------------------------------------------------------
    # Inference: render full image (no ray-subsampling, no loss)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def render_image(
        self,
        image:             torch.Tensor,   # (1, 3, H, W)
        camera_extrinsics: torch.Tensor,   # (1, 4, 4)
        camera_intrinsics: torch.Tensor,   # (1, 3, 3)
        n_rays_per_chunk:  int = 4096,
    ) -> dict:
        """
        Render all pixels of the image in chunks (avoids OOM).
        Returns {colour, depth, normal} as (H, W, C) tensors.
        """
        B = 1
        H, W = image.shape[2], image.shape[3]
        device = image.device

        image_norm = (image - self.img_mean) / self.img_std
        z = self.encoder(image_norm)
        brdf  = self.brdf_head(z)
        light = self.light_head(z)

        # Build all pixel rays
        ys, xs = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing="ij",
        )
        fx = camera_intrinsics[0, 0, 0]
        fy = camera_intrinsics[0, 1, 1]
        cx = camera_intrinsics[0, 0, 2]
        cy = camera_intrinsics[0, 1, 2]

        u = (xs - cx) / fx
        v = (ys - cy) / fy
        dirs_cam = torch.stack([u, v, torch.ones_like(u)], dim=-1).reshape(-1, 3)

        R = camera_extrinsics[0, :3, :3]
        t = camera_extrinsics[0, :3, 3]
        dirs_world = F.normalize(dirs_cam @ R.T, dim=-1)
        origins = t.unsqueeze(0).expand_as(dirs_world)

        N_pixels = H * W
        colours = torch.zeros(N_pixels, 3, device=device)
        depths  = torch.zeros(N_pixels, device=device)
        normals = torch.zeros(N_pixels, 3, device=device)

        for start in range(0, N_pixels, n_rays_per_chunk):
            end = min(start + n_rays_per_chunk, N_pixels)
            chunk_o = origins[start:end]
            chunk_d = dirs_world[start:end]
            chunk_n = end - start

            def sdf_fn(pts: torch.Tensor) -> torch.Tensor:
                return self.sdf_mlp(pts, z.expand(pts.shape[0], -1))

            out = self.renderer(
                sdf_fn=sdf_fn, brdf=brdf, light=light,
                rays_o=chunk_o, rays_d=chunk_d, batch_idx=None,
            )
            colours[start:end] = out["colour"]
            depths[start:end]  = out["depth"]
            normals[start:end] = out["normal"]

        return {
            "colour": colours.reshape(H, W, 3),
            "depth":  depths.reshape(H, W),
            "normal": normals.reshape(H, W, 3),
            "brdf":   brdf,
            "light":  light,
        }
