import torch
import torch.nn as nn
import torchvision.models as tvm


class ImageEncoder(nn.Module):
    """ResNet-34 → global latent vector z."""

    def __init__(self, latent_dim: int = 128, pretrained: bool = True):
        super().__init__()
        weights = tvm.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        bb = tvm.resnet34(weights=weights)
        self.backbone = nn.Sequential(
            bb.conv1, bb.bn1, bb.relu, bb.maxpool,
            bb.layer1, bb.layer2, bb.layer3, bb.layer4,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.proj = nn.Sequential(
            nn.Linear(512, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        # ImageNet normalisation buffers so callers can just pass [0,1] images.
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B, N, 3, H, W) in [0, 1].  Returns z: (B, latent_dim)."""
        B, N, C, H, W = images.shape
        x = images.reshape(B * N, C, H, W)
        x = (x - self.mean) / self.std
        z = self.proj(self.backbone(x))          # (B*N, latent_dim)
        return z.reshape(B, N, -1).mean(dim=1)   # (B, latent_dim) — mean over views
