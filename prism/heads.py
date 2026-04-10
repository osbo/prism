"""
Prediction heads conditioned on the global latent z.

BRDFHead  → per-object Cook-Torrance GGX material parameters
              albedo  (R, G, B)  ∈ [0, 1]³
              roughness           ∈ [0, 1]
              metalness           ∈ [0, 1]

LightHead → single point-light source
              position  (x, y, z) ∈ R³  (scene units)
              intensity (R, G, B) ∈ R₊³

Both heads receive no direct supervision — they are trained entirely
through the render loss, forced to be physically consistent with the
geometry and the observed image.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import BRDFConfig, LightConfig


# ---------------------------------------------------------------------------
# Shared MLP builder
# ---------------------------------------------------------------------------

def _build_mlp(in_dim: int, hidden_dim: int, out_dim: int,
               n_layers: int) -> nn.Sequential:
    """Simple MLP with GELU activations and layer norm on hidden layers."""
    layers: list[nn.Module] = []
    dim = in_dim
    for i in range(n_layers - 1):
        layers += [nn.Linear(dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()]
        dim = hidden_dim
    layers.append(nn.Linear(dim, out_dim))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# BRDF Head
# ---------------------------------------------------------------------------

class BRDFHead(nn.Module):
    """
    Maps z → (albedo, roughness, metalness).

    Output activations:
      albedo    — sigmoid  → [0, 1]
      roughness — sigmoid  → [0, 1]  (clamped to [0.04, 1] in BRDF eval
                                       to avoid the singularity at r=0)
      metalness — sigmoid  → [0, 1]
    """

    def __init__(self, cfg: BRDFConfig):
        super().__init__()
        self.mlp = _build_mlp(cfg.latent_dim, cfg.hidden_dim,
                              out_dim=5,           # 3 albedo + 1 roughness + 1 metalness
                              n_layers=cfg.n_layers)

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            z: (B, latent_dim)
        Returns:
            dict with keys 'albedo' (B,3), 'roughness' (B,1), 'metalness' (B,1)
        """
        out = self.mlp(z)                        # (B, 5)
        albedo    = torch.sigmoid(out[:, :3])    # (B, 3)  ∈ [0,1]
        roughness = torch.sigmoid(out[:, 3:4])   # (B, 1)  ∈ [0,1]
        metalness = torch.sigmoid(out[:, 4:5])   # (B, 1)  ∈ [0,1]
        return {"albedo": albedo, "roughness": roughness, "metalness": metalness}


# ---------------------------------------------------------------------------
# Light Head
# ---------------------------------------------------------------------------

class LightHead(nn.Module):
    """
    Maps z → (light_position, light_intensity).

    The position is passed through tanh and scaled so predictions stay
    within a few scene units (configurable via cfg.position_scale).
    Intensity uses softplus to enforce positivity with smooth gradients.
    """

    def __init__(self, cfg: LightConfig):
        super().__init__()
        self.position_scale = cfg.position_scale
        self.mlp = _build_mlp(cfg.latent_dim, cfg.hidden_dim,
                              out_dim=6,   # 3 position + 3 intensity
                              n_layers=cfg.n_layers)

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            z: (B, latent_dim)
        Returns:
            dict with keys 'light_pos' (B,3), 'light_intensity' (B,3)
        """
        out = self.mlp(z)                                     # (B, 6)
        light_pos       = torch.tanh(out[:, :3]) * self.position_scale  # (B, 3)
        light_intensity = F.softplus(out[:, 3:])              # (B, 3) ∈ R₊
        return {"light_pos": light_pos, "light_intensity": light_intensity}
