import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def fourier_encode(x: torch.Tensor, n_freqs: int = 6) -> torch.Tensor:
    """NeRF-style Fourier positional encoding. Returns (*, 3 + 6*n_freqs)."""
    freqs = (2.0 ** torch.arange(n_freqs, dtype=x.dtype, device=x.device)) * math.pi
    xf = (x.unsqueeze(-1) * freqs).reshape(*x.shape[:-1], x.shape[-1] * n_freqs)
    return torch.cat([x, xf.sin(), xf.cos()], dim=-1)


class SDFMLP(nn.Module):
    """
    (x, z [, local_feat]) → SDF value.

    8-layer MLP with a skip connection at layer 4.
    z is the global latent (from the encoder's mean-pool).
    local_feat is an optional per-point feature from image projection (feat_dim dims).
    """

    def __init__(self, latent_dim: int = 128, hidden: int = 256,
                 n_layers: int = 8, n_freqs: int = 6,
                 sphere_init_radius: float = 0.35,
                 feat_dim: int = 0):
        super().__init__()
        self.feat_dim  = feat_dim
        enc_dim = 3 + 2 * n_freqs * 3   # fourier(x)
        in_dim  = enc_dim + latent_dim + feat_dim
        skip    = n_layers // 2

        layers = []
        d = in_dim
        for i in range(n_layers):
            if i == skip:
                d = hidden + in_dim
            layers.append(nn.Linear(d, hidden))
            d = hidden
        self.layers  = nn.ModuleList(layers)
        self.out     = nn.Linear(hidden, 1)
        self.skip    = skip
        self.n_freqs = n_freqs
        self.sphere_init_radius = float(sphere_init_radius)

        self.sphere_scale = nn.Parameter(torch.tensor(1.0))
        self.sphere_bias  = nn.Parameter(torch.tensor(-self.sphere_init_radius))
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor, z: torch.Tensor,
                local_feat: torch.Tensor | None = None) -> torch.Tensor:
        """x: (..., 3), z: (..., latent_dim), local_feat: (..., feat_dim) → (..., 1)"""
        enc = fourier_encode(x, self.n_freqs)
        parts: list[torch.Tensor] = [enc, z]
        if self.feat_dim > 0:
            if local_feat is None:
                local_feat = enc.new_zeros(*enc.shape[:-1], self.feat_dim)
            parts.append(local_feat)
        inp = torch.cat(parts, dim=-1)
        h = inp
        for i, layer in enumerate(self.layers):
            if i == self.skip:
                h = torch.cat([h, inp], dim=-1)
            h = F.relu(layer(h))
        r = x.norm(dim=-1, keepdim=True)
        return self.out(h) + self.sphere_scale * r + self.sphere_bias
