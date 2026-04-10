"""
Cook-Torrance GGX BRDF evaluation (fully differentiable, batch-capable).

Implements the microfacet BRDF:

    f_r = (D * G * F) / (4 * (n·l) * (n·v))

where:
    D  = GGX normal distribution function (Trowbridge-Reitz)
    G  = Smith height-correlated masking-shadowing function
    F  = Schlick Fresnel approximation

Full shading model (one point light, no indirect illumination):

    L_o(v) = [k_d * albedo/π  +  f_r] * L_i * saturate(n·l)

where k_d = (1 - metalness) enforces energy conservation
and the specular F0 is lerp(0.04, albedo, metalness) (standard PBR convention).

All inputs and outputs are (N, C) tensors so the entire pixel batch is
evaluated in one vectorised forward pass.
"""

import torch
import torch.nn.functional as F

EPS = 1e-6


# ---------------------------------------------------------------------------
# GGX / Trowbridge-Reitz distribution  D
# ---------------------------------------------------------------------------

def _ggx_distribution(n_dot_h: torch.Tensor, roughness: torch.Tensor) -> torch.Tensor:
    """
    D_GGX(n, h; α) = α² / (π * ((n·h)² * (α²-1) + 1)²)

    Args:
        n_dot_h:   (N,) clamped to [0, 1]
        roughness: (N,) α ∈ (0, 1]
    Returns:
        D: (N,)
    """
    a2 = (roughness ** 2).clamp(min=EPS)
    denom = (n_dot_h ** 2) * (a2 - 1.0) + 1.0
    return a2 / (torch.pi * denom ** 2 + EPS)


# ---------------------------------------------------------------------------
# Smith geometry function  G
# ---------------------------------------------------------------------------

def _smith_g1_ggx(n_dot_x: torch.Tensor, roughness: torch.Tensor) -> torch.Tensor:
    """
    G1_Smith(n, x; α) using the GGX form.
    k = α / 2  (direct lighting remap from Schlick)
    """
    k = roughness / 2.0
    return n_dot_x / (n_dot_x * (1.0 - k) + k + EPS)


def _smith_geometry(n_dot_l: torch.Tensor, n_dot_v: torch.Tensor,
                    roughness: torch.Tensor) -> torch.Tensor:
    """
    Height-correlated Smith G = G1(n,l) * G1(n,v)
    """
    return _smith_g1_ggx(n_dot_l, roughness) * _smith_g1_ggx(n_dot_v, roughness)


# ---------------------------------------------------------------------------
# Schlick Fresnel  F
# ---------------------------------------------------------------------------

def _schlick_fresnel(f0: torch.Tensor, v_dot_h: torch.Tensor) -> torch.Tensor:
    """
    F(v, h) = F0 + (1 - F0) * (1 - v·h)^5

    Args:
        f0:     (N, 3) reflectance at normal incidence
        v_dot_h:(N,)   clamped to [0, 1]
    Returns:
        F: (N, 3)
    """
    return f0 + (1.0 - f0) * (1.0 - v_dot_h.unsqueeze(-1)).clamp(min=0) ** 5


# ---------------------------------------------------------------------------
# Full Cook-Torrance GGX evaluation
# ---------------------------------------------------------------------------

def cook_torrance_ggx(
    normals: torch.Tensor,        # (N, 3) surface normals (unit)
    view_dirs: torch.Tensor,      # (N, 3) direction FROM surface TO camera (unit)
    light_dirs: torch.Tensor,     # (N, 3) direction FROM surface TO light (unit)
    light_intensity: torch.Tensor,# (N, 3) light radiance
    albedo: torch.Tensor,         # (N, 3) diffuse albedo ∈ [0,1]
    roughness: torch.Tensor,      # (N, 1) or (N,) ∈ [0,1]
    metalness: torch.Tensor,      # (N, 1) or (N,) ∈ [0,1]
) -> torch.Tensor:
    """
    Evaluate outgoing radiance L_o for each surface point given:
      - shading geometry (n, v, l)
      - material parameters (albedo, roughness, metalness)
      - incident light radiance L_i from the predicted point light

    Returns:
        L_o: (N, 3) outgoing RGB radiance (linear), clipped to [0, ∞)

    Notes:
        • roughness is clamped to [0.04, 1] to prevent GGX singularity.
        • All vectors are assumed unit-length; dot products are clamped to [0, 1].
        • Energy conservation: k_d = (1 - metalness) * (1 - F).
    """
    roughness = roughness.squeeze(-1).clamp(0.04, 1.0)  # (N,)
    metalness = metalness.squeeze(-1)                    # (N,)

    # Half-vector
    h = F.normalize(view_dirs + light_dirs, dim=-1)     # (N, 3)

    # Dot products — clamped to avoid NaN / negative values
    n_dot_l = (normals * light_dirs).sum(-1).clamp(0.0, 1.0)   # (N,)
    n_dot_v = (normals * view_dirs).sum(-1).clamp(0.0, 1.0)    # (N,)
    n_dot_h = (normals * h).sum(-1).clamp(0.0, 1.0)            # (N,)
    v_dot_h = (view_dirs * h).sum(-1).clamp(0.0, 1.0)          # (N,)

    # F0: lerp(dielectric=0.04, albedo, metalness)  — standard PBR
    f0 = 0.04 * (1.0 - metalness.unsqueeze(-1)) + albedo * metalness.unsqueeze(-1)  # (N,3)

    # Microfacet terms
    D = _ggx_distribution(n_dot_h, roughness)   # (N,)
    G = _smith_geometry(n_dot_l, n_dot_v, roughness)  # (N,)
    Fres = _schlick_fresnel(f0, v_dot_h)         # (N, 3)

    # Specular BRDF
    denom = (4.0 * n_dot_v * n_dot_l).clamp(min=EPS)  # (N,)
    specular = (D * G).unsqueeze(-1) * Fres / denom.unsqueeze(-1)  # (N, 3)

    # Diffuse BRDF (Lambertian, energy-conserving)
    k_d = (1.0 - Fres) * (1.0 - metalness.unsqueeze(-1))
    diffuse = k_d * albedo / torch.pi               # (N, 3)

    # Outgoing radiance
    L_o = (diffuse + specular) * light_intensity * n_dot_l.unsqueeze(-1)  # (N, 3)
    return L_o.clamp(min=0.0)
