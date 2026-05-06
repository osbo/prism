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

    z conditions the network via FiLM (feature-wise linear modulation): a single
    linear layer maps z → (γ_i, β_i) for every hidden layer, then applies
    h_i ← γ_i ⊙ ReLU(W_i h_{i-1}) + β_i.  This lets z modulate every layer's
    activations rather than only influencing the first layer through concatenation.

    Initialised so γ = 1, β = 0 everywhere, making FiLM a no-op at the start of
    training — the network begins as a plain sphere-init MLP and the FiLM generator
    learns to deviate from that.
    """

    def __init__(self, latent_dim: int = 128, hidden: int = 256,
                 n_layers: int = 8, n_freqs: int = 6,
                 sphere_init_radius: float = 0.35,
                 feat_dim: int = 0):
        super().__init__()
        self.feat_dim  = feat_dim
        self.n_freqs   = n_freqs
        self.n_layers  = n_layers
        self.hidden    = hidden

        enc_dim = 3 + 2 * n_freqs * 3   # fourier(x): 3 raw + 2*n_freqs*3 sinusoids
        in_dim  = enc_dim + feat_dim     # z is NOT in the input — handled by FiLM
        skip    = n_layers // 2
        self.skip = skip

        layers = []
        d = in_dim
        for i in range(n_layers):
            if i == skip:
                d = hidden + in_dim      # skip concatenates [h, inp_no_z]
            layers.append(nn.Linear(d, hidden))
            d = hidden
        self.layers = nn.ModuleList(layers)
        self.out    = nn.Linear(hidden, 1)

        # FiLM generator: single linear z → (γ, β) for all layers simultaneously.
        # One linear is cheaper than a deep generator and avoids adding a new
        # optimisation problem on top of the SDF.
        self.film_gen = nn.Linear(latent_dim, 2 * n_layers * hidden)
        nn.init.zeros_(self.film_gen.weight)
        bias = torch.zeros(2 * n_layers * hidden)
        bias[:n_layers * hidden] = 1.0   # γ channels → 1  (identity at init)
        self.film_gen.bias = nn.Parameter(bias)

        # Sphere initialisation: SDF(x) ≈ ‖x‖ − R at the start.
        self.sphere_scale = nn.Parameter(torch.tensor(1.0))
        self.sphere_bias  = nn.Parameter(torch.tensor(-float(sphere_init_radius)))
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor, z: torch.Tensor,
                local_feat: torch.Tensor | None = None) -> torch.Tensor:
        """x: (N, 3), z: (N, latent_dim), local_feat: (N, feat_dim) | None → (N, 1)"""
        enc = fourier_encode(x, self.n_freqs)
        parts: list[torch.Tensor] = [enc]
        if self.feat_dim > 0:
            if local_feat is None:
                local_feat = enc.new_zeros(*enc.shape[:-1], self.feat_dim)
            parts.append(local_feat)
        inp = torch.cat(parts, dim=-1)   # (N, enc_dim + feat_dim)

        # FiLM parameters from z — one pass through the generator.
        film = self.film_gen(z)          # (N, 2 * n_layers * hidden)
        nh   = self.n_layers * self.hidden
        gamma = film[:, :nh].reshape(-1, self.n_layers, self.hidden)   # (N, L, H)
        beta  = film[:, nh:].reshape(-1, self.n_layers, self.hidden)   # (N, L, H)

        h = inp
        for i, layer in enumerate(self.layers):
            if i == self.skip:
                h = torch.cat([h, inp], dim=-1)
            h = F.relu(layer(h))
            h = gamma[:, i, :] * h + beta[:, i, :]   # FiLM modulation

        r = x.norm(dim=-1, keepdim=True)
        return self.out(h) + self.sphere_scale * r + self.sphere_bias
