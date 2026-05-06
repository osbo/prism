"""
NeuS-style volume rendering.

Instead of hard sphere tracing, we use NeuS's logistic density approximation
(Wang et al., NeurIPS 2021) which converts SDF values along a ray into volume
rendering weights. This gives smooth gradient flow through silhouettes and
doesn't require implementing the implicit function theorem backward.

Reference: https://arxiv.org/abs/2106.10689
"""

import torch
import torch.nn.functional as F


def sample_rays(c2w, K, H, W, n_rays, device):
    """
    Randomly sample n_rays pixels per image and build world-space rays.

    Returns:
        rays_o  (B*n_rays, 3)  world-space origins
        rays_d  (B*n_rays, 3)  unit world-space directions
        pix_rc  (B*n_rays, 2)  (row, col) of sampled pixel
        bidx    (B*n_rays,)    batch index
    """
    B = c2w.shape[0]
    rows = torch.randint(0, H, (B, n_rays), device=device)
    cols = torch.randint(0, W, (B, n_rays), device=device)

    fx = K[:, 0, 0].unsqueeze(1)
    fy = K[:, 1, 1].unsqueeze(1)
    cx = K[:, 0, 2].unsqueeze(1)
    cy = K[:, 1, 2].unsqueeze(1)

    # Blender/OpenGL convention: camera looks along -Z
    dirs_cam = torch.stack([
        (cols.float() - cx) / fx,
        -(rows.float() - cy) / fy,
        -torch.ones_like(rows, dtype=torch.float32),
    ], dim=-1)                                             # (B, n_rays, 3)

    R = c2w[:, :3, :3]
    dirs_world = F.normalize(
        torch.einsum("bij,bkj->bki", R, dirs_cam), dim=-1
    )                                                      # (B, n_rays, 3)
    origins = c2w[:, :3, 3].unsqueeze(1).expand(B, n_rays, 3)

    bidx = torch.arange(B, device=device).repeat_interleave(n_rays)
    return (
        origins.reshape(B * n_rays, 3),
        dirs_world.reshape(B * n_rays, 3),
        torch.stack([rows, cols], dim=-1).reshape(B * n_rays, 2),
        bidx,
    )


def sample_pdf(
    bins: torch.Tensor,
    weights: torch.Tensor,
    n_samples: int,
    det: bool = False,
) -> torch.Tensor:
    """
    Inverse-CDF sampling from piecewise-constant PDF.
    bins:    (N_rays, N_bins) midpoints
    weights: (N_rays, N_bins) non-negative
    Returns: (N_rays, n_samples)
    """
    if n_samples <= 0:
        return bins.new_empty((bins.shape[0], 0))
    weights = weights + 1e-5
    pdf = weights / weights.sum(dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], dim=-1)  # (N, N_bins+1)
    if det:
        u = torch.linspace(0.0, 1.0, n_samples, device=bins.device, dtype=bins.dtype)
        u = u.unsqueeze(0).expand(bins.shape[0], -1)
    else:
        u = torch.rand(bins.shape[0], n_samples, device=bins.device, dtype=bins.dtype)
    cdf = cdf.contiguous()
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = (inds - 1).clamp(min=0)
    above = inds.clamp(max=cdf.shape[-1] - 1)
    cdf_g0 = torch.gather(cdf, 1, below)
    cdf_g1 = torch.gather(cdf, 1, above)
    bins_pad = torch.cat([bins[:, :1], bins], dim=-1)
    bins_g0 = torch.gather(bins_pad, 1, below)
    bins_g1 = torch.gather(bins_pad, 1, above)
    denom = (cdf_g1 - cdf_g0).clamp(min=1e-6)
    t = (u - cdf_g0) / denom
    return bins_g0 + t * (bins_g1 - bins_g0)


def neus_weights(sdf: torch.Tensor, beta: torch.Tensor, t_vals: torch.Tensor) -> torch.Tensor:
    """
    Convert sorted SDF values along rays to NeuS volume rendering weights.

    sdf:    (N_rays, N_samples)  — sorted by increasing t
    beta: scalar               — surface sharpness (learned)
    t_vals: (N_rays, N_samples) — sorted ray distances
    Returns weights (N_rays, N_samples) that peak at the zero-crossing.
    """
    inv_s = (1.0 / beta.clamp_min(1e-6)).clamp(max=1e6)
    dists = t_vals[:, 1:] - t_vals[:, :-1]
    dists = torch.cat([dists, dists[:, -1:]], dim=-1).clamp(min=1e-6)

    # NeuS-style section estimation with monotonic cosine (outside->inside along ray).
    true_cos = (sdf[:, 1:] - sdf[:, :-1]) / dists[:, :-1].clamp(min=1e-6)
    prev_cos = torch.cat([true_cos[:, :1], true_cos[:, :-1]], dim=-1)
    iter_cos = torch.minimum(prev_cos, true_cos).clamp(max=0.0)
    iter_cos = torch.cat([iter_cos, iter_cos[:, -1:]], dim=-1)

    est_prev = sdf - iter_cos * dists * 0.5
    est_next = sdf + iter_cos * dists * 0.5
    prev_cdf = torch.sigmoid(est_prev * inv_s)
    next_cdf = torch.sigmoid(est_next * inv_s)
    p = (prev_cdf - next_cdf).clamp(min=0.0)
    c = prev_cdf.clamp(min=1e-6)
    alpha = (p / c).clamp(0.0, 1.0)
    alpha = torch.nan_to_num(alpha, nan=0.0, posinf=1.0, neginf=0.0)
    # T_i = ∏_{j<i}(1 − α_j)
    T = torch.cumprod(
        torch.cat([torch.ones_like(alpha[:, :1]), 1.0 - alpha[:, :-1]], dim=1),
        dim=1,
    )
    w = T * alpha
    return torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)  # (N_rays, N_samples)
