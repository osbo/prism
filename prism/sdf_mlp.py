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
    (x, z) → SDF value.

    8-layer MLP with a skip connection at layer 4 (concatenate input again).
    z is injected by concatenation at the input — clean and simple.
    """

    def __init__(self, latent_dim: int = 128, hidden: int = 256,
                 n_layers: int = 8, n_freqs: int = 6):
        super().__init__()
        enc_dim = 3 + 2 * n_freqs * 3   # fourier(x)
        in_dim  = enc_dim + latent_dim
        skip    = n_layers // 2

        layers = []
        d = in_dim
        for i in range(n_layers):
            if i == skip:
                d = hidden + in_dim
            layers.append(nn.Linear(d, hidden))
            d = hidden
        self.layers = nn.ModuleList(layers)
        self.out    = nn.Linear(hidden, 1)
        self.skip   = skip
        self.n_freqs = n_freqs

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """x: (..., 3), z: (..., latent_dim) → (..., 1)"""
        inp = torch.cat([fourier_encode(x, self.n_freqs), z], dim=-1)
        h = inp
        for i, layer in enumerate(self.layers):
            if i == self.skip:
                h = torch.cat([h, inp], dim=-1)
            h = F.relu(layer(h))
        return self.out(h)
