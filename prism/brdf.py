import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-6


def cook_torrance_ggx(
    n:               torch.Tensor,  # (..., 3) unit surface normal
    v:               torch.Tensor,  # (..., 3) unit view direction (toward camera)
    l:               torch.Tensor,  # (..., 3) unit light direction (toward light)
    albedo:          torch.Tensor,  # (..., 3) in [0, 1]
    roughness:       torch.Tensor,  # (..., 1) in [0, 1]
    metalness:       torch.Tensor,  # (..., 1) in [0, 1]
    light_intensity: torch.Tensor,  # (..., 3) point-light radiance scale
    ambient:         torch.Tensor,  # (..., 3) isotropic ambient strength (× diffuse albedo)
) -> torch.Tensor:
    """Cook-Torrance GGX BRDF × point light + diffuse ambient fill.  Returns (..., 3) RGB."""
    h   = F.normalize(v + l, dim=-1, eps=EPS)
    ndl = (n * l).sum(-1, keepdim=True)                   # can be ≤ 0 (back-face)
    ndv = (n * v).sum(-1, keepdim=True).clamp(EPS, 1.0)
    ndh = (n * h).sum(-1, keepdim=True).clamp(EPS, 1.0)
    hdv = (h * v).sum(-1, keepdim=True).clamp(0.0, 1.0)

    a  = roughness.clamp(0.04, 1.0) ** 2
    a2 = a ** 2

    # GGX normal distribution
    D = a2 / (torch.pi * ((ndh**2 * (a2 - 1.0) + 1.0) ** 2 + EPS))

    # Schlick Fresnel
    F0 = 0.04 * (1.0 - metalness) + albedo * metalness
    Fr = F0 + (1.0 - F0) * (1.0 - hdv) ** 5

    # Smith GGX geometry
    k  = a / 2.0
    G  = (ndv / (ndv * (1.0 - k) + k + EPS)) * (ndl.clamp(EPS, 1.0) / (ndl.clamp(EPS, 1.0) * (1.0 - k) + k + EPS))

    specular = D * Fr * G / (4.0 * ndv * ndl.clamp(EPS, 1.0) + EPS)
    diffuse  = (albedo / torch.pi) * (1.0 - metalness) * (1.0 - Fr)

    direct = (diffuse + specular) * light_intensity * ndl.clamp(min=0.0)
    # Isotropic ambient: tint albedo on the diffuse lobe only (no specular in flat fill).
    amb_fill = ambient * (albedo / torch.pi) * (1.0 - metalness)
    out = (direct + amb_fill).clamp(0.0, 1.0)
    return torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)


class BRDFHead(nn.Module):
    """z → (albedo, roughness, metalness) — spatially uniform per object."""

    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, 128),        nn.ReLU(),
            nn.Linear(128, 5),
        )
        # Init: albedo→0.5 (bias=0), roughness→0.5 (bias=0), metalness→0.1 (bias≈-2.2)
        # Zero the last layer's weights so initial output depends only on bias, not z.
        nn.init.zeros_(self.net[-1].weight)
        bias = torch.zeros(5)
        bias[4] = -2.2   # metalness starts low (~0.1); most surfaces are dielectric
        self.net[-1].bias = nn.Parameter(bias)

    def forward(self, z: torch.Tensor):
        out       = self.net(z)
        albedo    = out[..., :3].sigmoid()
        roughness = out[..., 3:4].sigmoid()
        metalness = out[..., 4:5].sigmoid()
        return albedo, roughness, metalness


class LightHead(nn.Module):
    """z → (light_pos, point_light_intensity, ambient) — point + isotropic ambient."""

    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, 128),        nn.ReLU(),
            nn.Linear(128, 9),
        )
        # Init intensity bias to +2.0 → softplus(2)≈2.1, sigmoid(2)≈0.88 gradient.
        # Without this, random init can push intensity to softplus(very_negative)≈0
        # and the gradient (sigmoid(very_negative)≈0) vanishes — unrecoverable.
        nn.init.zeros_(self.net[-1].weight)
        bias = torch.zeros(9)
        bias[2] = 4.0   # light_pos z: on camera side at init
        bias[3:6] = 2.0   # point light → softplus(2) ≈ 2.1
        bias[6:9] = 0.0   # ambient → softplus(0) ≈ 0.69 (gentle fill in shadows)
        self.net[-1].bias = nn.Parameter(bias)

    def forward(self, z: torch.Tensor):
        out = self.net(z)
        light_pos = out[..., :3]
        # Avoid runaway highlights that can destabilize early optimization.
        light_int = F.softplus(out[..., 3:6]).clamp(max=20.0)
        ambient = F.softplus(out[..., 6:9]).clamp(max=10.0)
        return light_pos, light_int, ambient
