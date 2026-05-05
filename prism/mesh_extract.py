"""
Marching-cubes extraction of the neural SDF (eval / visualize only).

The SDF is queried on a uniform grid in ``[-cfg.mc_bound, cfg.mc_bound]^3``.
Requires ``scikit-image`` for ``skimage.measure.marching_cubes``.

Optional **view-conditioned carving** (``cfg.mc_carve_background``): grid points
that project outside the image, behind the camera, or onto GT background pixels
have their SDF blended toward ``max(sdf, mc_carve_sdf_min)``. A lightly **blurred**
mask plus this blend softens the carve boundary so marching cubes is less prone
to harsh axis-aligned stairsteps than a hard per-voxel ``where``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _blur_mask_hw(mask_hw: torch.Tensor, radius: int) -> torch.Tensor:
    """Light Gaussian blur on (H,W) mask for softer carve weights."""
    if radius <= 0:
        return mask_hw
    k = 2 * radius + 1
    sig = 0.5 * float(radius) + 0.25
    ax = torch.arange(k, device=mask_hw.device, dtype=mask_hw.dtype) - radius
    g = torch.exp(-0.5 * (ax / sig) ** 2)
    g = g / g.sum()
    k2 = (g[:, None] * g[None, :]) / (g[:, None] * g[None, :]).sum()
    k2 = k2.view(1, 1, k, k)
    x = mask_hw.unsqueeze(0).unsqueeze(0)
    x = F.pad(x, (radius, radius, radius, radius), mode="replicate")
    return F.conv2d(x, k2).squeeze(0).squeeze(0)


def _squeeze_pose(c2w: torch.Tensor, K: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if c2w.dim() == 3:
        c2w = c2w[0]
    if K.dim() == 3:
        K = K[0]
    return c2w, K


def _mask_hw(mask: torch.Tensor) -> torch.Tensor:
    """Return (H, W) float mask in [0, 1]."""
    m = mask
    if m.dim() == 3:
        m = m[0]
    if m.dim() == 2:
        return m
    raise ValueError(f"mask must be (H,W) or (1,H,W); got {tuple(mask.shape)}")


def _carve_blend_weight(
    pts: torch.Tensor,
    c2w: torch.Tensor,
    K: torch.Tensor,
    mask_hw: torch.Tensor,
    mask_blur_radius: int,
) -> torch.Tensor:
    """
    Per grid point: blend weight w in [0, 1]. At w=1 the SDF is pushed to ``max(sdf, smin)``;
    at w=0 it is unchanged. Soft weights (blurred mask) reduce voxel stairsteps at the silhouette.
    """
    device = pts.device
    dtype = pts.dtype
    c2w, K = _squeeze_pose(c2w.to(device=device, dtype=dtype), K.to(device=device, dtype=dtype))
    H, W = int(mask_hw.shape[0]), int(mask_hw.shape[1])
    mask_hw = mask_hw.to(device=device, dtype=dtype)
    mask_use = _blur_mask_hw(mask_hw, mask_blur_radius)

    R = c2w[:3, :3]
    t = c2w[:3, 3]
    # Row-vector camera coords: p_c_row = (p_w - t) @ R  (matches p_w = R @ p_c + t).
    pc = torch.matmul(pts - t.unsqueeze(0), R)
    xc, yc, zc = pc[:, 0], pc[:, 1], pc[:, 2]
    # Rays use dirs_cam with z = -1 (into scene); in-front points have zc < 0.
    eps = 1e-5
    in_front = zc < -eps

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    col = cx + fx * (xc / zc.clamp(max=-eps))
    row = cy + fy * (yc / zc.clamp(max=-eps))

    in_bounds = (col >= 0) & (col < W - 1e-6) & (row >= 0) & (row < H - 1e-6)
    col_n = (col / max(W - 1, 1)) * 2.0 - 1.0
    row_n = (row / max(H - 1, 1)) * 2.0 - 1.0
    # grid_sample: x=col, y=row in normalized [-1,1]; batch mode (1,1,H,W).
    grid = torch.stack([col_n, row_n], dim=-1).view(1, 1, -1, 2)
    m = mask_use.view(1, 1, H, W)
    sampled = F.grid_sample(
        m,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).view(-1)
    sampled = sampled.clamp(0.0, 1.0)
    # Background cone: push SDF; soft (1 - sampled) so mask edges are gradual in 3D.
    w = (1.0 - sampled).clamp(0.0, 1.0)
    w = torch.where(~in_front | ~in_bounds, torch.ones_like(w), w)
    return w


def extract_sdf_mesh(
    model,
    z: torch.Tensor,
    cfg,
    device: torch.device,
    mask_hw: torch.Tensor | None = None,
    c2w: torch.Tensor | None = None,
    K: torch.Tensor | None = None,
):
    """
    Build a triangle mesh at SDF iso-level ``cfg.mc_threshold``.

    Parameters
    ----------
    model :
        Must expose ``model.sdf_mlp(pts, z_batch) -> (N, 1)`` SDF values.
    z :
        Latent ``(latent_dim,)`` on ``device``.
    mask_hw, c2w, K :
        If all set and ``cfg.mc_carve_background``, SDF is raised where the grid
        point projects to GT background (or outside the frustum), removing
        spurious sheets from ``.obj`` export for that view.
    """
    try:
        from skimage.measure import marching_cubes
    except ImportError as e:
        raise ImportError("Mesh extraction requires scikit-image: pip install scikit-image") from e

    res = cfg.mc_resolution
    bound = cfg.mc_bound
    lin = torch.linspace(-bound, bound, res, device=device)
    gx, gy, gz = torch.meshgrid(lin, lin, lin, indexing="ij")
    pts = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)

    chunk = 32768
    z_single = z.unsqueeze(0).expand(chunk, -1)

    sdf_chunks = []
    with torch.no_grad():
        for i in range(0, pts.shape[0], chunk):
            p = pts[i : i + chunk]
            zc = z_single[: p.shape[0]]
            sdf_chunks.append(model.sdf_mlp(p, zc).squeeze(-1))

        sdf_flat = torch.cat(sdf_chunks)

        do_carve = (
            getattr(cfg, "mc_carve_background", False)
            and mask_hw is not None
            and c2w is not None
            and K is not None
        )
        if do_carve:
            m = _mask_hw(mask_hw)
            blur_r = int(getattr(cfg, "mc_carve_mask_blur_radius", 2))
            w = _carve_blend_weight(pts, c2w, K, m, blur_r)
            smin = float(getattr(cfg, "mc_carve_sdf_min", 0.35))
            smin_t = sdf_flat.new_tensor(smin)
            sdf_pushed = torch.maximum(sdf_flat, smin_t)
            # Convex blend avoids a single-voxel SDF cliff at the carve boundary.
            sdf_flat = (1.0 - w) * sdf_flat + w * sdf_pushed

    sdf_grid = sdf_flat.reshape(res, res, res).cpu().numpy()

    thr = getattr(cfg, "mc_threshold", 0.0)
    if sdf_grid.min() >= thr or sdf_grid.max() <= thr:
        return None

    verts, faces, *_ = marching_cubes(sdf_grid, level=thr)
    verts = verts / (res - 1) * (2 * bound) - bound
    verts = verts.astype(np.float32)

    if getattr(cfg, "mc_keep_largest_component", False):
        import trimesh

        tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        parts = tm.split(only_watertight=False)
        if len(parts) > 1:
            tm = max(parts, key=lambda x: int(x.faces.shape[0]))
            verts, faces = tm.vertices.astype(np.float32), tm.faces

    return verts, faces
