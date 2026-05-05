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


def neus_weights(sdf: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """
    Convert sorted SDF values along rays to NeuS volume rendering weights.

    sdf:  (N_rays, N_samples)  — sorted by increasing t
    beta: scalar               — surface sharpness (learned)
    Returns weights (N_rays, N_samples) that peak at the zero-crossing.
    """
    beta = beta.clamp_min(1e-6)
    # Φ_s(x) = sigmoid(+x/β)  — NeuS eq.13; high outside (sdf>0), low inside (sdf<0).
    # Along a ray from outside→inside, sdf decreases → Φ decreases → α > 0.
    phi      = torch.sigmoid(sdf / beta)
    phi_next = torch.cat([phi[:, 1:], phi[:, -1:]], dim=1)
    # α_i = max(Φ(s_i) − Φ(s_{i+1}), 0) / Φ(s_i)
    alpha = ((phi - phi_next) / (phi + 1e-6)).clamp(0.0, 1.0)
    alpha = torch.nan_to_num(alpha, nan=0.0, posinf=1.0, neginf=0.0)
    # T_i = ∏_{j<i}(1 − α_j)
    T = torch.cumprod(
        torch.cat([torch.ones_like(alpha[:, :1]), 1.0 - alpha[:, :-1]], dim=1),
        dim=1,
    )
    w = T * alpha
    return torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)  # (N_rays, N_samples)
