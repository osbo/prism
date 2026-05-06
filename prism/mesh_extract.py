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


def _multiview_carve_weight(
    pts: torch.Tensor,
    input_c2ws: torch.Tensor,
    input_Ks: torch.Tensor,
    input_masks: torch.Tensor,
    mask_blur_radius: int,
) -> torch.Tensor:
    """
    Per-grid-point carve weight from all input views.  A point outside the foreground
    mask in ANY view gets w=1 (fully carved); inside all views gets w=0 (unchanged).

    input_c2ws:  (Nv, 4, 4)
    input_Ks:    (Nv, 3, 3)
    input_masks: (Nv, 1, H, W)
    Returns:     (N_pts,) in [0, 1]
    """
    Nv = input_c2ws.shape[0]
    dtype = pts.dtype
    # min mask value across all views (0 = outside in ≥1 view)
    min_sampled = torch.ones(pts.shape[0], device=pts.device, dtype=dtype)
    for k in range(Nv):
        c2w_k = input_c2ws[k].to(device=pts.device, dtype=dtype)
        K_k   = input_Ks[k].to(device=pts.device, dtype=dtype)
        mask_k = input_masks[k].to(device=pts.device, dtype=dtype)   # (1, H, W)
        _, H, W = mask_k.shape
        if mask_blur_radius > 0:
            mask_k = _blur_mask_hw(mask_k[0], mask_blur_radius).unsqueeze(0)

        R = c2w_k[:3, :3]
        t = c2w_k[:3, 3]
        pc = torch.matmul(pts - t.unsqueeze(0), R)

        # Blender: forward = -Z; visible points have pc[:,2] < 0
        eps = 1e-5
        in_front = pc[:, 2] < -eps
        z_c = (-pc[:, 2]).clamp(min=eps)
        fx, fy = K_k[0, 0], K_k[1, 1]
        cx, cy = K_k[0, 2], K_k[1, 2]
        u = cx + fx * (pc[:, 0] / z_c)
        v = cy - fy * (pc[:, 1] / z_c)   # negate y: camera +Y up, image +Y down

        u_n = u / (W - 1) * 2.0 - 1.0
        v_n = v / (H - 1) * 2.0 - 1.0
        grid = torch.stack([u_n, v_n], dim=-1).view(1, 1, -1, 2)
        sampled = F.grid_sample(
            mask_k.unsqueeze(0), grid, mode="bilinear",
            padding_mode="zeros", align_corners=True,
        ).view(-1).clamp(0.0, 1.0)
        # Behind-camera or out-of-frustum: treat as foreground (don't carve)
        sampled = torch.where(in_front, sampled, torch.ones_like(sampled))
        min_sampled = torch.minimum(min_sampled, sampled)

    # carve weight: 1 where outside hull, 0 where inside all views
    return (1.0 - min_sampled).clamp(0.0, 1.0)


def extract_sdf_mesh(
    model,
    z: torch.Tensor,
    cfg,
    device: torch.device,
    mask_hw: torch.Tensor | None = None,
    c2w: torch.Tensor | None = None,
    K: torch.Tensor | None = None,
    input_masks: torch.Tensor | None = None,
    input_c2ws: torch.Tensor | None = None,
    input_Ks: torch.Tensor | None = None,
    feat_maps: torch.Tensor | None = None,
    img_hw: tuple[int, int] | None = None,
):
    """
    Build a triangle mesh at SDF iso-level ``cfg.mc_threshold``.

    Parameters
    ----------
    feat_maps : (1, Nv, C, Hf, Wf) or (Nv, C, Hf, Wf) — per-view feature maps from
        the encoder.  When provided (along with input_c2ws / input_Ks / img_hw), SDF
        grid points are evaluated with the same per-point projected features used
        during training, avoiding a train/eval mismatch when feat_dim > 0.
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

    # Prepare feature projection if feat_maps provided (avoids train/eval mismatch).
    use_feats = (
        feat_maps is not None
        and input_c2ws is not None
        and input_Ks is not None
        and img_hw is not None
        and hasattr(model, "_project_features")
    )
    if use_feats:
        # Ensure (1, Nv, C, Hf, Wf) shape for _project_features
        fm = feat_maps if feat_maps.dim() == 5 else feat_maps.unsqueeze(0)
        ic = input_c2ws if input_c2ws.dim() == 4 else input_c2ws.unsqueeze(0)
        ik = input_Ks   if input_Ks.dim() == 4   else input_Ks.unsqueeze(0)

    sdf_chunks = []
    with torch.no_grad():
        for i in range(0, pts.shape[0], chunk):
            p = pts[i : i + chunk]
            zc = z_single[: p.shape[0]]
            if use_feats:
                bidx = torch.zeros(p.shape[0], dtype=torch.long, device=device)
                lf = model._project_features(p, bidx, ic.float(), ik.float(), fm.float(), img_hw)
            else:
                lf = None
            sdf_chunks.append(model.sdf_mlp(p, zc, lf).squeeze(-1))

        sdf_flat = torch.cat(sdf_chunks)

        blur_r = int(getattr(cfg, "mc_carve_mask_blur_radius", 2))
        smin   = float(getattr(cfg, "mc_carve_sdf_min", 0.35))
        smin_t = sdf_flat.new_tensor(smin)

        multi_view_available = (
            input_masks is not None
            and input_c2ws is not None
            and input_Ks is not None
        )
        single_view_available = (
            getattr(cfg, "mc_carve_background", False)
            and mask_hw is not None
            and c2w is not None
            and K is not None
        )

        if multi_view_available:
            # Preferred: carve from all input views (respects visual hull).
            im = input_masks[0] if input_masks.dim() == 5 else input_masks  # (Nv,1,H,W)
            ic = input_c2ws[0] if input_c2ws.dim() == 4 else input_c2ws      # (Nv,4,4)
            ik = input_Ks[0]   if input_Ks.dim() == 4   else input_Ks        # (Nv,3,3)
            w = _multiview_carve_weight(pts, ic, ik, im, blur_r)
            sdf_pushed = torch.maximum(sdf_flat, smin_t)
            sdf_flat = (1.0 - w) * sdf_flat + w * sdf_pushed
        elif single_view_available:
            m = _mask_hw(mask_hw)
            w = _carve_blend_weight(pts, c2w, K, m, blur_r)
            sdf_pushed = torch.maximum(sdf_flat, smin_t)
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

    n_smooth = int(getattr(cfg, "mc_laplacian_iters", 0))
    if n_smooth > 0:
        import trimesh
        import trimesh.smoothing

        tm_s = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        trimesh.smoothing.filter_laplacian(tm_s, lamb=0.5, iterations=n_smooth)
        verts = np.array(tm_s.vertices, dtype=np.float32)
        faces = tm_s.faces

    return verts, faces
