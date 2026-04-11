"""
Neural SDF MLP with:
  - NeRF-style Fourier positional encoding on 3D query point x
    (optionally accelerated by tiny-cuda-nn's frequency encoding kernel)
  - FiLM conditioning of the global latent z into every MLP layer
  - Geometric initialisation so the SDF starts as a sphere (avoids
    degenerate early training where the sphere tracer never converges)
  - Optional weight normalisation for stable gradients through the eikonal loss

Architecture per layer:
    h_l = γ_l(z) * Linear(h_{l-1}) + β_l(z)     # FiLM
    h_l = Softplus(h_l)

Output: scalar signed distance value s = f(x; θ, z).

References:
  - IDR (Yariv et al., NeurIPS 2020) — geometric init + IFT backward
  - NeuS (Wang et al., NeurIPS 2021) — SDF-to-density
  - pi-GAN / SIREN — FiLM conditioning for implicit networks
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import SDFConfig


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------

class FourierEncoding(nn.Module):
    """
    NeRF-style sinusoidal encoding:
        γ(x) = [x, sin(2^0 π x), cos(2^0 π x), ..., sin(2^{L-1} π x), cos(2^{L-1} π x)]
    Output dimension: 3 + 3 * 2 * n_freqs
    """

    def __init__(self, n_freqs: int = 6, include_input: bool = True):
        super().__init__()
        self.n_freqs = n_freqs
        self.include_input = include_input
        # Register frequency bands as a buffer (not learned)
        freqs = 2.0 ** torch.arange(n_freqs, dtype=torch.float32) * math.pi
        self.register_buffer("freqs", freqs)

    @property
    def out_dim(self) -> int:
        return 3 * (1 + 2 * self.n_freqs) if self.include_input else 3 * 2 * self.n_freqs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., 3)
        Returns:
            encoded: (..., out_dim)
        """
        # x: (..., 3)  freqs: (n_freqs,)
        xf = x.unsqueeze(-1) * self.freqs               # (..., 3, n_freqs)
        encoded = torch.cat([torch.sin(xf), torch.cos(xf)], dim=-1)  # (..., 3, 2*n_freqs)
        encoded = encoded.flatten(-2)                    # (..., 3*2*n_freqs)
        if self.include_input:
            encoded = torch.cat([x, encoded], dim=-1)   # (..., out_dim)
        return encoded


class TCNNFourierEncoding(nn.Module):
    """
    Drop-in replacement for FourierEncoding that uses tiny-cuda-nn's
    fused frequency-encoding CUDA kernel for ~4× faster throughput.
    Falls back to the PyTorch implementation if tcnn is not available.
    """

    def __init__(self, n_freqs: int = 6, include_input: bool = True):
        super().__init__()
        self.n_freqs = n_freqs
        self.include_input = include_input
        self._tcnn_enc = None

        try:
            import tinycudann as tcnn
            self._tcnn_enc = tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": "Frequency",
                    "n_frequencies": n_freqs,
                },
            )
        except (ImportError, Exception):
            self._fallback = FourierEncoding(n_freqs, include_input=False)

    @property
    def out_dim(self) -> int:
        return 3 * (1 + 2 * self.n_freqs) if self.include_input else 3 * 2 * self.n_freqs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape[:-1]
        x_flat = x.reshape(-1, 3)

        if self._tcnn_enc is not None:
            # tcnn expects float32 in [0, 1]; our x is typically in [-1, 1]
            # Remap: x_01 = (x + 1) / 2  (approximate; true range depends on scene)
            x_01 = (x_flat + 1.0) / 2.0
            enc = self._tcnn_enc(x_01).float()  # tcnn may output half
        else:
            enc = self._fallback(x_flat)

        if self.include_input:
            enc = torch.cat([x_flat, enc], dim=-1)

        return enc.reshape(*shape, -1)


# ---------------------------------------------------------------------------
# FiLM conditioning utilities
# ---------------------------------------------------------------------------

class FiLMLayer(nn.Module):
    """
    One FiLM-conditioned hidden layer:
        h = γ(z) ⊙ W x + β(z)
    followed by Softplus activation.

    γ and β are produced by small linear layers from the latent z.
    """

    def __init__(self, in_features: int, out_features: int, z_dim: int,
                 weight_norm: bool = True):
        super().__init__()
        linear = nn.Linear(in_features, out_features, bias=True)
        self.linear = nn.utils.weight_norm(linear) if weight_norm else linear

        # Separate projections for scale (γ) and shift (β)
        self.gamma = nn.Linear(z_dim, out_features, bias=True)
        self.beta  = nn.Linear(z_dim, out_features, bias=True)

        # Initialise γ → 1 and β → 0 (identity FiLM at the start)
        # γ(z) = W_γ z + b_γ; for γ=1 regardless of z: W_γ=0, b_γ=1
        nn.init.zeros_(self.gamma.weight)
        nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (N, in_features)  — point features
            z: (N, z_dim)        — latent codes (already expanded to match N)
        Returns:
            (N, out_features) after FiLM + Softplus
        """
        gamma = self.gamma(z)   # (N, out_features)
        beta  = self.beta(z)    # (N, out_features)
        h = self.linear(h)      # (N, out_features)
        h = gamma * h + beta
        return F.softplus(h, beta=100)  # sharp Softplus ≈ ReLU but differentiable


# ---------------------------------------------------------------------------
# SDF MLP
# ---------------------------------------------------------------------------

class SDFMLP(nn.Module):
    """
    Coordinate MLP mapping (x ∈ R³, z) → s ∈ R (signed distance).

    Args:
        cfg: SDFConfig
    """

    def __init__(self, cfg: SDFConfig):
        super().__init__()
        self.cfg = cfg

        # Positional encoding
        if cfg.use_tcnn_encoding:
            self.encoding = TCNNFourierEncoding(cfg.n_fourier_freqs, include_input=True)
        else:
            self.encoding = FourierEncoding(cfg.n_fourier_freqs, include_input=True)

        enc_dim = self.encoding.out_dim  # 3 + 3*2*n_freqs

        # Input projection (no FiLM on the first layer — just a linear map)
        first_linear = nn.Linear(enc_dim, cfg.hidden_dim, bias=True)
        self.input_proj = (
            nn.utils.weight_norm(first_linear) if cfg.weight_norm else first_linear
        )

        # FiLM-conditioned hidden layers
        self.layers = nn.ModuleList([
            FiLMLayer(cfg.hidden_dim, cfg.hidden_dim, cfg.latent_dim,
                      weight_norm=cfg.weight_norm)
            for _ in range(cfg.n_layers - 1)
        ])

        # Output head: hidden_dim → 1 (SDF value)
        self.output = nn.Linear(cfg.hidden_dim, 1, bias=True)

        if cfg.geometric_init:
            self._geometric_init()

    # ------------------------------------------------------------------
    # Geometric initialisation (IDR / SAL style)
    # The network is initialised so that f(x) ≈ ||x|| - r (sphere of
    # radius r), giving the sphere tracer a reasonable starting surface.
    # ------------------------------------------------------------------
    def _geometric_init(self):
        r = self.cfg.sphere_radius
        # Zero all layer weights first
        for layer in self.layers:
            nn.init.normal_(layer.linear.weight, mean=0.0,
                            std=math.sqrt(2) / math.sqrt(layer.linear.out_features))
            nn.init.zeros_(layer.linear.bias)

        # Output layer: initialise to approximate sphere SDF
        nn.init.normal_(self.output.weight, mean=math.sqrt(math.pi) /
                         math.sqrt(self.cfg.hidden_dim), std=1e-4)
        nn.init.constant_(self.output.bias, -r)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, 3) 3D query points  (scene units, object centred ~[-1,1])
            z: (B, latent_dim) or (N, latent_dim) latent codes.
               If (B, latent_dim) and B != N, z is expanded along the ray
               dimension inside the renderer before this call.
        Returns:
            sdf: (N, 1) signed distance values
        """
        enc = self.encoding(x)                         # (N, enc_dim)
        h = F.softplus(self.input_proj(enc), beta=100) # (N, hidden_dim)
        for layer in self.layers:
            h = layer(h, z)                            # (N, hidden_dim)
        sdf = self.output(h)                           # (N, 1)
        return sdf

    # ------------------------------------------------------------------
    # Gradient utility — used for normals + eikonal loss
    # ------------------------------------------------------------------
    def gradient(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Compute ∇_x f(x; z) via autograd.  x must have requires_grad=True.

        Args:
            x: (N, 3) with requires_grad=True
            z: (N, latent_dim)
        Returns:
            grad: (N, 3)
        """
        x = x.requires_grad_(True)
        sdf = self.forward(x, z)                       # (N, 1)
        grad = torch.autograd.grad(
            outputs=sdf,
            inputs=x,
            grad_outputs=torch.ones_like(sdf),
            create_graph=self.training,
            retain_graph=True,
        )[0]                                           # (N, 3)
        return grad
