"""
PRISM: Physics-Informed Reconstruction via Implicit Surfaces and Materials.

Forward pass (training):
  1. image → encoder → z
  2. Sample n_rays pixels; build world-space rays from (c2w, K)
  3. Stratified sampling along each ray → 3D points
  4. SDF MLP(pts, z) → SDF values + gradient (via autograd)
  5. NeuS weights(SDF) → expected depth, normal, surface point
  6. BRDF head(z) → albedo, roughness, metalness
  7. Light head(z) → light_pos, light_intensity
  8. Cook-Torrance GGX → predicted pixel color
  9. Loss = λ_render·L1(color) + λ_depth·L1(depth) + λ_normal·cosine(normal) + λ_eik·eikonal
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder  import ImageEncoder
from .sdf_mlp  import SDFMLP
from .brdf     import BRDFHead, LightHead, cook_torrance_ggx
from .renderer import sample_rays, neus_weights


class PRISM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        ld = cfg.latent_dim
        self.encoder    = ImageEncoder(latent_dim=ld, pretrained=cfg.pretrained_encoder)
        self.sdf_mlp    = SDFMLP(latent_dim=ld, hidden=cfg.sdf_hidden,
                                 n_layers=cfg.sdf_layers, n_freqs=cfg.n_freqs)
        self.brdf_head  = BRDFHead(latent_dim=ld)
        self.light_head = LightHead(latent_dim=ld)
        # NeuS β: controls surface sharpness.  Learned; started soft, sharpens during training.
        self.log_beta   = nn.Parameter(torch.tensor(0.0))   # β = exp(log_β), init = 1.0
        self.cfg        = cfg

    @property
    def beta(self):
        return self.log_beta.exp().clamp(min=1e-4)

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------
    def forward(self, image, c2w, K, gt_depth, gt_normal):
        """
        image:     (B, 3, H, W) in [0, 1]
        c2w:       (B, 4, 4)
        K:         (B, 3, 3)
        gt_depth:  (B, 1, H, W)
        gt_normal: (B, 3, H, W) unit normals; zero on background
        """
        cfg    = self.cfg
        B, _, H, W = image.shape
        device = image.device

        z = self.encoder(image)   # (B, latent_dim)

        # ---- Sample rays -----------------------------------------------
        rays_o, rays_d, pix_rc, bidx = sample_rays(
            c2w, K, H, W, cfg.n_rays, device
        )  # each (B*n_rays, 3/2)
        N = rays_o.shape[0]       # B * n_rays

        # ---- Stratified point sampling ---------------------------------
        near, far = cfg.near, cfg.far
        t_base  = torch.linspace(near, far, cfg.n_samples, device=device)
        t_noise = torch.rand(N, cfg.n_samples, device=device) * (far - near) / cfg.n_samples
        t_vals  = (t_base + t_noise)                          # (N, n_samples)

        pts = rays_o[:, None] + t_vals[:, :, None] * rays_d[:, None]  # (N, n_samples, 3)

        z_rays = z[bidx]                                      # (N, latent_dim)
        z_pts  = z_rays[:, None].expand(-1, cfg.n_samples, -1).reshape(-1, z.shape[-1])

        # ---- SDF + gradient (create_graph for eikonal BP) --------------
        pts_flat = pts.reshape(-1, 3).requires_grad_(True)
        sdf_flat = self.sdf_mlp(pts_flat, z_pts).squeeze(-1)  # (N*n_samples,)

        sdf_grad = torch.autograd.grad(
            sdf_flat, pts_flat,
            grad_outputs=torch.ones_like(sdf_flat),
            create_graph=self.training,
            retain_graph=True,
        )[0]                                                   # (N*n_samples, 3)

        sdf_vals = sdf_flat.reshape(N, cfg.n_samples)
        normals  = sdf_grad.reshape(N, cfg.n_samples, 3)

        # ---- NeuS weights → depth, normal, surface point ---------------
        # Keep SDF attached so render/depth/normal losses can shape geometry.
        # Detaching here makes SDF learn only from eikonal regularization.
        weights     = neus_weights(sdf_vals, self.beta)  # (N, n_samples)
        pred_depth  = (weights * t_vals).sum(-1)                  # (N,)
        pred_normal = F.normalize(
            (weights[:, :, None] * normals).sum(1), dim=-1
        )                                                          # (N, 3)
        x_surf = rays_o + pred_depth[:, None] * rays_d            # (N, 3)

        # ---- BRDF + light ----------------------------------------------
        albedo, roughness, metalness = self.brdf_head(z_rays)     # (N, 3/1/1)
        light_pos, light_int         = self.light_head(z_rays)    # (N, 3/3)

        l_dir = F.normalize(light_pos - x_surf, dim=-1)
        v_dir = F.normalize(-rays_d, dim=-1)

        pred_color = cook_torrance_ggx(
            pred_normal, v_dir, l_dir, albedo, roughness, metalness, light_int
        )                                                          # (N, 3)

        # ---- Losses ----------------------------------------------------
        # Index GT at sampled pixels
        img_hwc    = image.permute(0, 2, 3, 1)           # (B, H, W, 3)
        depth_hw   = gt_depth[:, 0]                       # (B, H, W)
        normal_hwc = gt_normal.permute(0, 2, 3, 1)        # (B, H, W, 3)
        r, c       = pix_rc[:, 0], pix_rc[:, 1]

        gt_color  = img_hwc[bidx, r, c]                   # (N, 3)
        gt_d      = depth_hw[bidx, r, c]                  # (N,)
        gt_n      = normal_hwc[bidx, r, c]                # (N, 3)

        # Render loss — only where light is on the right side (ndl > 0)
        # and weight mass is meaningful (ray hit something)
        hit  = weights.sum(-1) > 0.1
        l_render = F.l1_loss(pred_color[hit], gt_color[hit]) if hit.any() else pred_color.sum() * 0.0

        # Depth loss — valid GT only; normalize by ``far`` so L1 is O(1) like RGB
        # (raw world distances ~[near, far] were ~3× larger than render terms, so
        # ``total`` looked flat / noisy even when render was improving).
        valid_d = (gt_d > near) & (gt_d < far)
        if valid_d.any():
            inv = 1.0 / (far + 1e-6)
            l_depth = F.l1_loss(pred_depth[valid_d] * inv, gt_d[valid_d] * inv)
        else:
            l_depth = pred_depth.sum() * 0.0

        # Normal loss — valid (non-background) GT normals
        valid_n = gt_n.norm(dim=-1) > 0.5
        if valid_n.any():
            pn, gn = pred_normal[valid_n], gt_n[valid_n]
            l_normal = (1.0 - F.cosine_similarity(pn, gn, dim=-1)).mean() \
                     + F.l1_loss(pn, F.normalize(gn, dim=-1))
        else:
            l_normal = pred_normal.sum() * 0.0

        # Eikonal loss — ||∇f|| = 1
        l_eikonal = ((sdf_grad.norm(dim=-1) - 1.0) ** 2).mean()

        total = (
            cfg.lambda_render   * l_render
            + cfg.lambda_depth  * l_depth
            + cfg.lambda_normal * l_normal
            + cfg.lambda_eik    * l_eikonal
        )
        return {
            "total":    total,
            "render":   l_render.detach(),
            "depth":    l_depth.detach(),
            "normal":   l_normal.detach(),
            "eikonal":  l_eikonal.detach(),
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @torch.no_grad()
    def render_image(self, image, c2w, K):
        """Render depth + normal + shaded color for a full image (no GT needed)."""
        B, _, H, W = image.shape
        device     = image.device
        cfg        = self.cfg

        z                            = self.encoder(image)
        albedo, roughness, metalness = self.brdf_head(z)
        light_pos, light_int         = self.light_head(z)

        # Build all rays for the full image
        ys = torch.arange(H, device=device, dtype=torch.float32)
        xs = torch.arange(W, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")   # (H, W)

        fx, fy = K[0, 0, 0], K[0, 1, 1]
        cx, cy = K[0, 0, 2], K[0, 1, 2]
        dirs_cam = torch.stack([
            (xx - cx) / fx, -(yy - cy) / fy, -torch.ones_like(xx)
        ], dim=-1).reshape(-1, 3)                          # (H*W, 3)
        dirs_world = F.normalize(dirs_cam @ c2w[0, :3, :3].T, dim=-1)
        origins    = c2w[0, :3, 3].unsqueeze(0).expand(H * W, 3)

        # Render in chunks to stay within GPU memory
        chunk   = 4096
        colors  = torch.zeros(H * W, 3, device=device)
        depths  = torch.zeros(H * W,    device=device)
        norms   = torch.zeros(H * W, 3, device=device)

        z_exp      = z.expand(chunk, -1)
        ab_exp     = albedo.expand(chunk, -1)
        ro_exp     = roughness.expand(chunk, -1)
        me_exp     = metalness.expand(chunk, -1)
        lp_exp     = light_pos.expand(chunk, -1)
        li_exp     = light_int.expand(chunk, -1)

        near, far   = cfg.near, cfg.far
        n_s         = cfg.n_samples

        for i in range(0, H * W, chunk):
            ro = origins[i:i+chunk]
            rd = dirs_world[i:i+chunk]
            n  = ro.shape[0]

            t  = torch.linspace(near, far, n_s, device=device)
            t  = t.unsqueeze(0).expand(n, -1)
            p  = ro[:, None] + t[:, :, None] * rd[:, None]

            zc = z_exp[:n]; ab = ab_exp[:n]; ro_ = ro_exp[:n]
            me = me_exp[:n]; lp = lp_exp[:n]; li = li_exp[:n]

            zp = zc[:, None].expand(-1, n_s, -1).reshape(-1, z.shape[-1])
            sf = self.sdf_mlp(p.reshape(-1, 3), zp).squeeze(-1).reshape(n, n_s)

            # Finite-difference normals for inference (no create_graph needed)
            with torch.enable_grad():
                pf = p.reshape(-1, 3).requires_grad_(True)
                sf2 = self.sdf_mlp(pf, zp).squeeze(-1)
                g = torch.autograd.grad(sf2, pf, torch.ones_like(sf2))[0]
            g = g.reshape(n, n_s, 3)

            w  = neus_weights(sf.detach(), self.beta)
            d  = (w * t).sum(-1)
            nm = F.normalize((w[:, :, None] * g).sum(1), dim=-1)
            xs_ = ro + d[:, None] * rd

            l_dir = F.normalize(lp - xs_, dim=-1)
            v_dir = F.normalize(-rd, dim=-1)
            col   = cook_torrance_ggx(nm, v_dir, l_dir, ab, ro_, me, li)

            colors[i:i+n] = col
            depths[i:i+n] = d
            norms[i:i+n]  = nm

        return {
            "color":  colors.reshape(H, W, 3).clamp(0, 1),
            "depth":  depths.reshape(H, W),
            "normal": norms.reshape(H, W, 3),
        }
