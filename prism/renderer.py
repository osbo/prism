"""
Differentiable renderer combining:

1. Sphere tracing   — marches along each ray querying the SDF MLP until
                       |f(x)| < ε, finding the surface intersection x*.

2. IDR gradient trick — after tracing (which runs without gradients for
                         speed), we re-evaluate the SDF at the converged
                         point so that gradients can flow back through x*
                         into the SDF weights and the encoder:

                             x*(θ) ≈ x_0 − f(x_0; θ) / (d · ∇f(x_0; θ)) · d

                         where x_0 = detached surface point (from sphere
                         trace), and f(x_0; θ) ≈ 0 so the correction term
                         is nearly zero forward but carries dL/dθ backward.

3. NeuS silhouette  — rays that do NOT converge (near boundaries / background)
                       fall back to volume rendering with the NeuS logistic
                       density ρ = s·sigmoid(-s·f)·sigmoid(s·f).  This
                       guarantees smooth gradient flow at silhouettes during
                       early training when the SDF is imprecise.

4. BRDF shading     — at hit points, normals are n = ∇f(x*)/||∇f(x*)|| and
                       Cook-Torrance GGX is evaluated with the predicted
                       material + light parameters.

References:
    IDR   — Yariv et al., NeurIPS 2020
    NeuS  — Wang et al., NeurIPS 2021
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional
from config import RendererConfig
from .brdf import cook_torrance_ggx


# ---------------------------------------------------------------------------
# Sphere tracer  (no-grad forward, IFT-based backward)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _sphere_trace(
    sdf_fn: Callable[[torch.Tensor], torch.Tensor],
    rays_o: torch.Tensor,   # (N, 3)
    rays_d: torch.Tensor,   # (N, 3) unit vectors
    near: float,
    far: float,
    n_steps: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sphere trace a batch of rays.

    Returns:
        t:    (N,)  ray parameter at (approximate) surface or `far` if miss
        hit:  (N,)  bool — True if |f| < eps at convergence
    """
    N = rays_o.shape[0]
    t = torch.full((N,), near, device=rays_o.device, dtype=rays_o.dtype)
    hit = torch.zeros(N, dtype=torch.bool, device=rays_o.device)

    for _ in range(n_steps):
        active = ~hit & (t < far)
        if not active.any():
            break
        pts = rays_o[active] + t[active].unsqueeze(-1) * rays_d[active]  # (M, 3)
        sdf = sdf_fn(pts).squeeze(-1)                                      # (M,)

        # Advance by the SDF value (Lipschitz sphere step).
        # Use clamp(min=0): if sdf < 0 (overshoot), don't advance further —
        # advancing by |sdf| would push deeper into the surface.
        t[active] = t[active] + sdf.clamp(min=0)
        # Mark converged
        hit[active] = sdf.abs() < eps

    return t.clamp(near, far), hit


def _ift_surface_points(
    sdf_fn: Callable[[torch.Tensor], torch.Tensor],
    rays_o: torch.Tensor,   # (N, 3)
    rays_d: torch.Tensor,   # (N, 3)
    t: torch.Tensor,        # (N,) detached t from sphere trace
) -> torch.Tensor:
    """
    Apply the IDR implicit function theorem correction so gradients flow
    from the loss into the SDF MLP through x*:

        x*(θ) ≈ x_0 − [f(x_0; θ) / (d · ∇f(x_0; θ))] · d

    where x_0 is the detached (no-grad) surface point found by sphere tracing.
    The first-order correction term is ≈ 0 numerically (since f(x*) ≈ 0),
    but its gradient wrt θ is non-zero and is the key gradient signal.

    Returns:
        x_star: (N, 3)  with gradient-attached coordinates
    """
    x0 = (rays_o + t.unsqueeze(-1) * rays_d).detach().requires_grad_(True)

    sdf_val = sdf_fn(x0)             # (N, 1)

    # ∇_x f at x0
    grad_x = torch.autograd.grad(
        outputs=sdf_val,
        inputs=x0,
        grad_outputs=torch.ones_like(sdf_val),
        create_graph=True,
        retain_graph=True,
    )[0]                             # (N, 3)

    # d · ∇f  (directional derivative along the ray)
    d_dot_grad = (rays_d * grad_x).sum(-1, keepdim=True).clamp(min=1e-5)  # (N, 1)

    # IFT correction: subtract the tangential residual
    x_star = x0 - (sdf_val / d_dot_grad) * rays_d    # (N, 3), grad attached

    return x_star, grad_x.detach()


# ---------------------------------------------------------------------------
# NeuS silhouette volume rendering (for non-hit / boundary rays)
# ---------------------------------------------------------------------------

def _neus_volume_render(
    sdf_fn: Callable[[torch.Tensor], torch.Tensor],
    rays_o: torch.Tensor,    # (M, 3)
    rays_d: torch.Tensor,    # (M, 3)
    near: float,
    far: float,
    n_samples: int,
    s: torch.Tensor,         # () — NeuS logistic scale (learnable)
    shading_fn: Callable,    # (pts: (M*K,3)) → rgb (M*K,3)
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Volume render M rays using the NeuS density formulation.

    Density: ρ(t) = max(-df/dt, 0) / (sigmoid(s*f))  (Eq. 11, NeuS)
    Simplified to: φ_s(f) = s * sigmoid(s*f) * sigmoid(-s*f)

    Returns:
        colour: (M, 3)
        depth:  (M,)    expected depth
    """
    M = rays_o.shape[0]
    K = n_samples

    # Stratified samples along each ray
    t_vals = torch.linspace(near, far, K, device=rays_o.device, dtype=rays_o.dtype)
    t_vals = t_vals.unsqueeze(0).expand(M, K)   # (M, K)
    # Add uniform noise during training for anti-aliasing
    if K > 1:
        dt = (far - near) / K
        t_vals = t_vals + torch.rand_like(t_vals) * dt

    pts = rays_o.unsqueeze(1) + t_vals.unsqueeze(-1) * rays_d.unsqueeze(1)  # (M,K,3)
    pts_flat = pts.reshape(-1, 3)   # (M*K, 3)

    sdf_flat = sdf_fn(pts_flat).squeeze(-1)    # (M*K,)
    sdf = sdf_flat.reshape(M, K)              # (M, K)

    # NeuS density φ_s(f) = s * sigmoid(s*f) * sigmoid(-s*f)
    phi = s * torch.sigmoid(s * sdf) * torch.sigmoid(-s * sdf)  # (M, K)

    # Approximate alpha from density (trapezoidal intervals)
    delta = torch.diff(t_vals, dim=-1)  # (M, K-1)
    delta = torch.cat([delta, torch.full_like(delta[:, :1], 1e10)], dim=-1)  # (M,K)
    alpha = 1.0 - torch.exp(-phi * delta)  # (M, K)

    # Transmittance T_i = prod_{j<i}(1 - α_j)
    T = torch.cumprod(torch.cat([torch.ones_like(alpha[:, :1]), 1 - alpha + 1e-7], dim=-1), dim=-1)[:, :-1]
    weights = T * alpha   # (M, K)

    # Shading
    rgb_flat = shading_fn(pts_flat)           # (M*K, 3)
    rgb = rgb_flat.reshape(M, K, 3)           # (M, K, 3)

    colour = (weights.unsqueeze(-1) * rgb).sum(dim=1)   # (M, 3)
    depth  = (weights * t_vals).sum(dim=-1)              # (M,)

    return colour, depth


# ---------------------------------------------------------------------------
# Main differentiable renderer
# ---------------------------------------------------------------------------

class DifferentiableRenderer(nn.Module):
    """
    End-to-end differentiable renderer.

    Usage inside PRISM.forward:
        output = renderer(
            sdf_fn    = lambda pts: sdf_mlp(pts, z_expanded),
            brdf      = brdf_params,    # dict from BRDFHead
            light     = light_params,   # dict from LightHead
            rays_o    = ...,            # (B*N_rays, 3)
            rays_d    = ...,            # (B*N_rays, 3)
        )
    """

    def __init__(self, cfg: RendererConfig):
        super().__init__()
        self.cfg = cfg
        # NeuS logistic scale — annealed upward during training by the trainer
        self.register_buffer(
            "neus_s",
            torch.tensor(cfg.neus_s_init, dtype=torch.float32),
        )
        self.bg_colour = torch.tensor(cfg.bg_colour, dtype=torch.float32)

    def forward(
        self,
        sdf_fn: Callable[[torch.Tensor], torch.Tensor],
        brdf: dict,        # albedo(B,3), roughness(B,1), metalness(B,1)
        light: dict,       # light_pos(B,3), light_intensity(B,3)
        rays_o: torch.Tensor,   # (B*R, 3)
        rays_d: torch.Tensor,   # (B*R, 3)
        batch_idx: Optional[torch.Tensor] = None,  # (B*R,) which object per ray
    ) -> dict:
        """
        Returns dict with keys:
            colour    (B*R, 3)
            depth     (B*R,)
            normal    (B*R, 3)   — zero for background rays
            hit_mask  (B*R,)     — bool
            sdf_pts   (N_hit, 3) — surface points (for eikonal loss)
        """
        cfg = self.cfg
        N = rays_o.shape[0]
        device = rays_o.device

        # ----------------------------------------------------------------
        # 1. Sphere trace (no gradient)
        # ----------------------------------------------------------------
        t, hit = _sphere_trace(
            sdf_fn=sdf_fn,
            rays_o=rays_o,
            rays_d=rays_d,
            near=cfg.near,
            far=cfg.far,
            n_steps=cfg.n_sphere_trace_steps,
            eps=cfg.sphere_trace_eps,
        )

        # ----------------------------------------------------------------
        # 2. Surface points via IFT (gradient-attached)
        # ----------------------------------------------------------------
        colour   = self.bg_colour.to(device).expand(N, 3).clone()
        depth    = torch.full((N,), cfg.far, device=device)
        normal   = torch.zeros(N, 3, device=device)
        sdf_pts  = torch.zeros(0, 3, device=device)   # for eikonal loss

        if hit.any():
            h_o = rays_o[hit]   # (H, 3)
            h_d = rays_d[hit]   # (H, 3)
            h_t = t[hit]        # (H,)

            x_star, grad_at_x = _ift_surface_points(sdf_fn, h_o, h_d, h_t)

            # Surface normals: ∇f / ||∇f||
            # Re-compute with create_graph=True for normal loss
            x_for_normal = x_star.detach().requires_grad_(True)
            sdf_for_normal = sdf_fn(x_for_normal)
            grad_for_normal = torch.autograd.grad(
                sdf_for_normal,
                x_for_normal,
                grad_outputs=torch.ones_like(sdf_for_normal),
                create_graph=self.training,
            )[0]
            n = F.normalize(grad_for_normal, dim=-1)  # (H, 3)

            # ----------------------------------------------------------------
            # 3. BRDF shading at hit points
            # ----------------------------------------------------------------
            if batch_idx is not None:
                b_idx = batch_idx[hit]   # (H,)
                alb  = brdf["albedo"][b_idx]        # (H, 3)
                rou  = brdf["roughness"][b_idx]     # (H, 1)
                met  = brdf["metalness"][b_idx]     # (H, 1)
                lpos = light["light_pos"][b_idx]    # (H, 3)
                lint = light["light_intensity"][b_idx]  # (H, 3)
            else:
                # Single object in batch
                alb  = brdf["albedo"].expand(hit.sum(), -1)
                rou  = brdf["roughness"].expand(hit.sum(), -1)
                met  = brdf["metalness"].expand(hit.sum(), -1)
                lpos = light["light_pos"].expand(hit.sum(), -1)
                lint = light["light_intensity"].expand(hit.sum(), -1)

            # Direction from surface to light
            l_dir = F.normalize(lpos - x_star.detach(), dim=-1)    # (H, 3)
            # Direction from surface to camera
            v_dir = F.normalize(-h_d, dim=-1)                        # (H, 3)

            rgb = cook_torrance_ggx(
                normals=n,
                view_dirs=v_dir,
                light_dirs=l_dir,
                light_intensity=lint,
                albedo=alb,
                roughness=rou,
                metalness=met,
            )  # (H, 3)

            colour[hit]  = rgb
            depth[hit]   = (x_star.detach() - h_o).norm(dim=-1)
            normal[hit]  = n.detach()
            sdf_pts      = x_star   # keep grad for eikonal

        # ----------------------------------------------------------------
        # 4. NeuS volume rendering for non-hit rays (boundary / background)
        # ----------------------------------------------------------------
        miss = ~hit
        if miss.any() and self.training:
            m_o = rays_o[miss]
            m_d = rays_d[miss]

            if batch_idx is not None:
                b_idx = batch_idx[miss]
            else:
                b_idx = None

            def _shade_pts(pts: torch.Tensor) -> torch.Tensor:
                """Quick Lambertian shading for volume-rendered pts."""
                M_K = pts.shape[0]
                if b_idx is not None:
                    # Replicate batch_idx for the K samples per ray
                    _b = b_idx.repeat_interleave(
                        M_K // b_idx.shape[0] if b_idx.shape[0] > 0 else 1
                    )
                    alb  = brdf["albedo"][_b]
                    lpos = light["light_pos"][_b]
                    lint = light["light_intensity"][_b]
                else:
                    alb  = brdf["albedo"].expand(M_K, -1)
                    lpos = light["light_pos"].expand(M_K, -1)
                    lint = light["light_intensity"].expand(M_K, -1)
                l_dir = F.normalize(lpos - pts, dim=-1)
                n_approx = torch.zeros_like(pts)
                n_approx[:, 1] = 1.0   # placeholder normal for volume rendering
                v_approx = torch.zeros_like(pts)
                v_approx[:, 2] = 1.0
                rou = brdf["roughness"].expand(M_K, -1) if b_idx is None else brdf["roughness"][_b]
                met = brdf["metalness"].expand(M_K, -1) if b_idx is None else brdf["metalness"][_b]
                return cook_torrance_ggx(n_approx, v_approx, l_dir, lint, alb, rou, met)

            neus_col, neus_dep = _neus_volume_render(
                sdf_fn=sdf_fn,
                rays_o=m_o,
                rays_d=m_d,
                near=cfg.near,
                far=cfg.far,
                n_samples=32,
                s=self.neus_s,
                shading_fn=_shade_pts,
            )
            colour[miss] = neus_col
            depth[miss]  = neus_dep

        return {
            "colour":   colour,    # (N, 3)
            "depth":    depth,     # (N,)
            "normal":   normal,    # (N, 3)
            "hit_mask": hit,       # (N,) bool
            "sdf_pts":  sdf_pts,   # (H, 3) grad-attached surface points
        }

    def anneal_neus_s(self, factor: float = 1.01):
        """Call once per optimiser step to sharpen the NeuS density."""
        self.neus_s.clamp_(max=self.cfg.neus_s_max)
        self.neus_s.mul_(factor)
