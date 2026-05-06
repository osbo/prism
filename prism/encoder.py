import torch
import torch.nn as nn
import torchvision.models as tvm


class ImageEncoder(nn.Module):
    """
    ResNet-34 encoder with two outputs:
      z         — global latent (B, latent_dim), mean-pooled over N views; used for BRDF/light.
      feat_maps — per-pixel feature maps (B, N, feat_dim, H/8, W/8); used for per-point SDF conditioning.

    feat_dim=0 disables spatial features and returns feat_maps=None (global-only mode).
    """

    def __init__(self, latent_dim: int = 128, feat_dim: int = 32, pretrained: bool = True):
        super().__init__()
        self.feat_dim = feat_dim

        weights = tvm.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        bb = tvm.resnet34(weights=weights)

        # Shallow stem → layer2: output (B*N, 128, H/8, W/8) — good spatial resolution for projection.
        self.stem = nn.Sequential(
            bb.conv1, bb.bn1, bb.relu, bb.maxpool,
            bb.layer1, bb.layer2,
        )
        # Deep path: layer3 → layer4 → global pool → projection.
        self.deep = nn.Sequential(
            bb.layer3, bb.layer4,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.proj = nn.Sequential(
            nn.Linear(512, latent_dim),
            nn.LayerNorm(latent_dim),
        )

        # 1×1 conv to project 128-channel stem features to feat_dim.
        if feat_dim > 0:
            self.feat_proj = nn.Conv2d(128, feat_dim, kernel_size=1)
        else:
            self.feat_proj = None

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, images: torch.Tensor):
        """
        images: (B, N, 3, H, W) in [0, 1].
        Returns:
          z         — (B, latent_dim)
          feat_maps — (B, N, feat_dim, H/8, W/8) or None if feat_dim == 0
        """
        B, N, C, H, W = images.shape
        x = images.reshape(B * N, C, H, W)
        x = (x - self.mean) / self.std

        stem_out = self.stem(x)                              # (B*N, 128, H/8, W/8)
        z_flat   = self.proj(self.deep(stem_out))            # (B*N, latent_dim)
        z        = z_flat.reshape(B, N, -1).mean(dim=1)      # (B, latent_dim)

        if self.feat_proj is not None:
            fm = self.feat_proj(stem_out)                    # (B*N, feat_dim, H/8, W/8)
            _, _, Hf, Wf = fm.shape
            feat_maps = fm.reshape(B, N, self.feat_dim, Hf, Wf)
        else:
            feat_maps = None

        return z, feat_maps
