"""
Perceptual (VGG) loss for patch-level render supervision.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


class VGGPerceptualLoss(nn.Module):
    """
    L1 loss in VGG-16 relu1_2 and relu2_2 feature space.

    Weights are frozen.  Input images should be in [0, 1]; they are
    normalised to ImageNet statistics internally before passing to VGG.
    """

    def __init__(self):
        super().__init__()
        vgg = tvm.vgg16(weights=tvm.VGG16_Weights.IMAGENET1K_V1)
        feats = list(vgg.features)
        self.slice1 = nn.Sequential(*feats[:4])   # up to relu1_2
        self.slice2 = nn.Sequential(*feats[4:9])  # up to relu2_2
        for p in self.parameters():
            p.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred, target: (B, 3, H, W) in [0, 1].
        Returns scalar L1 loss in VGG feature space.
        """
        pred   = (pred.clamp(0, 1).float()   - self.mean) / self.std
        target = (target.clamp(0, 1).float() - self.mean) / self.std

        f1_p = self.slice1(pred)
        f1_t = self.slice1(target)
        f2_p = self.slice2(f1_p)
        f2_t = self.slice2(f1_t)
        return F.l1_loss(f1_p, f1_t) + F.l1_loss(f2_p, f2_t)
