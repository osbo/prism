"""
PRISM training losses:

    L_total = λ₁·L_render  +  λ₂·L_depth  +  λ₃·L_normal  +  λ₄·L_eikonal

L_render   — L1 pixel loss + optional LPIPS perceptual loss
             (sole supervision signal for BRDF + light parameters)
L_depth    — L1 between sphere-traced depth and GT depth map
L_normal   — cosine similarity loss between ∇f(x*) and GT normals
L_eikonal  — ||∇f(x)|| = 1 regulariser, essential for valid SDF

All losses operate on the ray-batch tensors produced by the renderer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import LossConfig


class PRISMLoss(nn.Module):
    def __init__(self, cfg: LossConfig):
        super().__init__()
        self.cfg = cfg

        if cfg.use_perceptual:
            try:
                import lpips
                self.lpips_fn = lpips.LPIPS(net="vgg")
                # Freeze LPIPS — it is a fixed perceptual metric
                for p in self.lpips_fn.parameters():
                    p.requires_grad_(False)
            except ImportError:
                print("Warning: lpips not installed. Disabling perceptual loss.")
                self.lpips_fn = None
        else:
            self.lpips_fn = None

    # ------------------------------------------------------------------
    # Individual loss terms
    # ------------------------------------------------------------------

    def render_loss(
        self,
        pred_colour: torch.Tensor,   # (N, 3)  rendered pixels (ray batch)
        gt_colour: torch.Tensor,     # (N, 3)  GT pixels
        # Optional: full image tensors for perceptual loss — only used
        # when the caller reconstructs the full image from the ray batch.
        pred_image: torch.Tensor | None = None,  # (B, 3, H, W)
        gt_image:   torch.Tensor | None = None,  # (B, 3, H, W)
    ) -> torch.Tensor:
        """
        L_render = L1(pred, gt) + perceptual_weight * LPIPS(pred_img, gt_img)
        """
        l1 = F.l1_loss(pred_colour, gt_colour)

        perceptual = torch.tensor(0.0, device=pred_colour.device)
        if (self.lpips_fn is not None
                and pred_image is not None
                and gt_image is not None):
            # LPIPS expects inputs in [-1, 1]
            p_img = pred_image * 2.0 - 1.0
            g_img = gt_image  * 2.0 - 1.0
            perceptual = self.lpips_fn(p_img, g_img).mean()

        return l1 + self.cfg.perceptual_weight * perceptual

    def depth_loss(
        self,
        pred_depth: torch.Tensor,   # (N,)
        gt_depth:   torch.Tensor,   # (N,)
        hit_mask:   torch.Tensor,   # (N,) bool  — only supervise hit rays
    ) -> torch.Tensor:
        """L1 depth loss over hit rays (rays that actually struck the surface)."""
        if not hit_mask.any():
            return pred_depth.sum() * 0.0   # differentiable zero
        return F.l1_loss(pred_depth[hit_mask], gt_depth[hit_mask])

    def normal_loss(
        self,
        pred_normal: torch.Tensor,  # (N, 3)
        gt_normal:   torch.Tensor,  # (N, 3)  unit vectors in camera / world frame
        hit_mask:    torch.Tensor,  # (N,) bool
    ) -> torch.Tensor:
        """
        1 - cosine_similarity(pred, gt), averaged over hit rays.
        This is in [0, 2] but in practice stays near 0 for well-aligned normals.
        """
        if not hit_mask.any():
            return pred_normal.sum() * 0.0
        cos_sim = F.cosine_similarity(
            pred_normal[hit_mask], gt_normal[hit_mask], dim=-1
        )
        return (1.0 - cos_sim).mean()

    def eikonal_loss(
        self,
        sdf_pts: torch.Tensor,   # (H, 3) surface points with grad attached
        sdf_fn,                  # callable: (pts) → sdf (H, 1)
    ) -> torch.Tensor:
        """
        Eikonal regulariser: E[( ||∇f(x)|| − 1 )²]

        We also sample a small set of random space points in [-1, 1]³ to
        encourage the SDF to be a valid distance function off-surface too.
        """
        if sdf_pts.shape[0] == 0:
            return sdf_pts.sum() * 0.0

        # Surface points (already have grad via IFT)
        pts_surface = sdf_pts.requires_grad_(True)
        sdf_s = sdf_fn(pts_surface)

        # Off-surface random points
        n_random = min(sdf_pts.shape[0], 512)
        pts_rand = torch.rand(n_random, 3, device=sdf_pts.device) * 2.0 - 1.0
        pts_rand = pts_rand.requires_grad_(True)
        sdf_r = sdf_fn(pts_rand)

        pts_all = torch.cat([pts_surface, pts_rand], dim=0)
        sdf_all = torch.cat([sdf_s, sdf_r], dim=0)

        grad = torch.autograd.grad(
            outputs=sdf_all,
            inputs=pts_all,
            grad_outputs=torch.ones_like(sdf_all),
            create_graph=True,
        )[0]   # (H + n_random, 3)

        eikonal = ((grad.norm(dim=-1) - 1.0) ** 2).mean()
        return eikonal

    # ------------------------------------------------------------------
    # Combined loss
    # ------------------------------------------------------------------

    def forward(
        self,
        render_out: dict,           # from DifferentiableRenderer
        gt_colour:  torch.Tensor,   # (N, 3)
        gt_depth:   torch.Tensor,   # (N,)
        gt_normal:  torch.Tensor,   # (N, 3)
        sdf_fn,
        pred_image: torch.Tensor | None = None,
        gt_image:   torch.Tensor | None = None,
    ) -> dict:
        """
        Compute all loss terms and their weighted sum.

        Returns dict with 'total' and individual terms for logging.
        """
        cfg = self.cfg
        pred_colour = render_out["colour"]
        pred_depth  = render_out["depth"]
        pred_normal = render_out["normal"]
        hit_mask    = render_out["hit_mask"]
        sdf_pts     = render_out["sdf_pts"]

        l_render   = self.render_loss(pred_colour, gt_colour, pred_image, gt_image)
        l_depth    = self.depth_loss(pred_depth, gt_depth, hit_mask)
        l_normal   = self.normal_loss(pred_normal, gt_normal, hit_mask)
        l_eikonal  = self.eikonal_loss(sdf_pts, sdf_fn)

        total = (
            cfg.lambda_render   * l_render
            + cfg.lambda_depth  * l_depth
            + cfg.lambda_normal * l_normal
            + cfg.lambda_eikonal * l_eikonal
        )

        return {
            "total":     total,
            "render":    l_render.detach(),
            "depth":     l_depth.detach(),
            "normal":    l_normal.detach(),
            "eikonal":   l_eikonal.detach(),
        }
