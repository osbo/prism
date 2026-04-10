"""
Image encoder: ResNet-34 backbone → global latent vector z.

The final average-pool + FC classification head is replaced by a small
projection MLP that maps the 512-d ResNet feature to the latent_dim z.
The backbone is fine-tuned end-to-end (or with BN frozen, configurable).
"""

import torch
import torch.nn as nn
import torchvision.models as tvm
from config import EncoderConfig


class ImageEncoder(nn.Module):
    """
    ResNet-34 encoder producing a global latent z ∈ R^{latent_dim}.

    Args:
        cfg: EncoderConfig
    """

    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg

        # Load backbone
        weights = tvm.ResNet34_Weights.IMAGENET1K_V1 if cfg.pretrained else None
        backbone = tvm.resnet34(weights=weights)

        # Keep all layers except the final classification FC
        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # → (B, 512)

        backbone_out_dim = 512  # ResNet-34 layer4 channels

        # Projection head: 512 → latent_dim with layer norm + activation
        self.proj = nn.Sequential(
            nn.Linear(backbone_out_dim, backbone_out_dim),
            nn.LayerNorm(backbone_out_dim),
            nn.GELU(),
            nn.Linear(backbone_out_dim, cfg.latent_dim),
            nn.LayerNorm(cfg.latent_dim),
        )

        if cfg.freeze_bn:
            self._freeze_bn()

    def _freeze_bn(self):
        for m in self.features.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.eval()
                for p in m.parameters():
                    p.requires_grad_(False)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: (B, 3, H, W) normalised RGB in [0, 1] or ImageNet-normalised
        Returns:
            z: (B, latent_dim) global latent code
        """
        feat = self.features(image)        # (B, 512, H/32, W/32)
        feat = self.pool(feat)             # (B, 512, 1, 1)
        feat = feat.flatten(1)             # (B, 512)
        z = self.proj(feat)                # (B, latent_dim)
        return z
