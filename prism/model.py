"""
PRISM: Physics-Informed Reconstruction via Implicit Surfaces and Materials.

Forward pass (training):
  1. image → encoder → z
  2. Sample n_rays pixels; build world-space rays from (c2w, K)
  3. Stratified sampling along each ray → 3D points
  4. SDF MLP(pts, z) → SDF values + gradient (via autograd)
  5. NeuS weights(SDF) → depth (conditional mean if w_sum above thresh else far), normal, surface point
  6. BRDF head(z) → albedo, roughness, metalness
  7. Light head(z) → light_pos, point intensity, ambient (RGB)
  8. Cook-Torrance GGX → predicted pixel color
  9. Loss = λ_render·L1(color) + λ_depth·L1(depth) + λ_normal·cosine(normal) + λ_eik·eikonal
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

from .encoder  import ImageEncoder
from .sdf_mlp  import SDFMLP
from .brdf     import BRDFHead, LightHead, cook_torrance_ggx
from .renderer import sample_rays, neus_weights, sample_pdf


class PRISM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        ld = cfg.latent_dim
        fd = getattr(cfg, "feat_dim", 0)
        self.encoder    = ImageEncoder(latent_dim=ld, feat_dim=fd, pretrained=cfg.pretrained_encoder)
        self.sdf_mlp    = SDFMLP(latent_dim=ld, hidden=cfg.sdf_hidden,
                                 n_layers=cfg.sdf_layers, n_freqs=cfg.n_freqs,
                                 sphere_init_radius=cfg.sdf_init_radius,
                                 feat_dim=fd)
        self.brdf_head  = BRDFHead(latent_dim=ld)
        self.light_head = LightHead(latent_dim=ld)
        # NeuS β: controls surface sharpness.  Learned; started soft, sharpens during training.
        self.log_beta   = nn.Parameter(torch.tensor(0.0))   # β = exp(log_β), init = 1.0
        self.cfg        = cfg

    @property
    def beta(self):
        # Clamp log-domain first to prevent exp overflow -> inf -> NaNs downstream.
        return self.log_beta.clamp(-10.0, 6.0).exp().clamp(min=self.cfg.beta_min)

    # ------------------------------------------------------------------
    # Per-point local feature projection (PixelNeRF-style)
    # ------------------------------------------------------------------
    def _project_features(self, pts, bidx_pts, input_c2ws, input_Ks, feat_maps, img_hw):
        """
        For each 3-D point, project into every input view and bilinearly sample the
        feature map at that location.  Features are averaged across views.

        pts:         (N_pts, 3)        — world-space sample positions
        bidx_pts:    (N_pts,)          — which batch element each point belongs to
        input_c2ws:  (B, Nv, 4, 4)    — camera-to-world for each input view
        input_Ks:    (B, Nv, 3, 3)    — intrinsics for each input view
        feat_maps:   (B, Nv, C, Hf, Wf)
        img_hw:      (H, W)            — original image dimensions (for coordinate normalisation)
        Returns:     (N_pts, C)
        """
        B, Nv, C, Hf, Wf = feat_maps.shape
        H, W = img_hw
        device = pts.device
        out = torch.zeros(pts.shape[0], C, device=device, dtype=pts.dtype)

        for b in range(B):
            mb = bidx_pts == b
            if not mb.any():
                continue
            pts_b = pts[mb]          # (N_b, 3)
            N_b   = pts_b.shape[0]

            view_feats = []
            for k in range(Nv):
                c2w_k = input_c2ws[b, k]    # (4, 4)
                K_k   = input_Ks[b, k]      # (3, 3)

                # World → camera (c2w is camera-to-world, so w2c: R^T(p - t))
                R   = c2w_k[:3, :3]
                t   = c2w_k[:3, 3]
                pc  = (pts_b - t) @ R       # (N_b, 3)  row-vector convention

                z_c = pc[:, 2].clamp(min=1e-4)
                u   = pc[:, 0] / z_c * K_k[0, 0] + K_k[0, 2]   # pixel x
                v   = pc[:, 1] / z_c * K_k[1, 1] + K_k[1, 2]   # pixel y

                # Normalise to [-1, 1] for grid_sample; out-of-view → zero via padding_mode.
                u_n = u / (W - 1) * 2.0 - 1.0
                v_n = v / (H - 1) * 2.0 - 1.0
                grid = torch.stack([u_n, v_n], dim=-1).reshape(1, 1, N_b, 2)

                # feat_maps[b, k]: (C, Hf, Wf) → add batch dim → (1, C, Hf, Wf)
                fm = feat_maps[b, k].unsqueeze(0)
                # grid_sample output: (1, C, 1, N_b) → (N_b, C)
                sampled = F.grid_sample(fm, grid, align_corners=True,
                                        mode="bilinear", padding_mode="zeros")
                view_feats.append(sampled.squeeze(0).squeeze(1).T)   # (N_b, C)

            out[mb] = torch.stack(view_feats, dim=0).mean(0)

        return out   # (N_pts, C)

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------
    def forward(self, images, c2w, K, gt_depth, gt_normal, gt_mask=None,
                input_c2ws=None, input_Ks=None):
        """
        images:      (B, N, 3, H, W) in [0, 1] — N context views
        c2w:         (B, 4, 4)  — target view camera (supervision)
        K:           (B, 3, 3)
        gt_depth:    (B, 1, H, W)
        gt_normal:   (B, 3, H, W) unit normals; zero on background
        gt_mask:     (B, 1, H, W) float — 1=foreground, 0=background
        input_c2ws:  (B, N, 4, 4) — cameras for all input views (for feature projection)
        input_Ks:    (B, N, 3, 3)
        """
        cfg    = self.cfg
        B, N_views, _, H, W = images.shape
        device = images.device

        # Encoder only under AMP — SDF + positional encoding + ∇SDF need fp32 to avoid
        # overflow / NaNs when autocast uses fp16 (especially with create_graph=True).
        if device.type == "cuda":
            with autocast("cuda", enabled=True):
                z, feat_maps = self.encoder(images)   # (B, ld), (B, N, fd, Hf, Wf) or None
        else:
            z, feat_maps = self.encoder(images)
        z_fp32 = z.float()
        if feat_maps is not None:
            feat_maps = feat_maps.float()

        # ---- Sample rays -----------------------------------------------
        rays_o, rays_d, pix_rc, bidx = sample_rays(
            c2w, K, H, W, cfg.n_rays, device
        )  # each (B*n_rays, 3/2)
        Nr = rays_o.shape[0]       # B * n_rays

        # ---- Stratified + hierarchical point sampling ------------------
        near, far = cfg.near, cfg.far
        beta_v = self.beta.float().clamp(min=float(cfg.beta_min))
        z_rays = z_fp32[bidx]  # (Nr, latent_dim)

        t_edges = torch.linspace(near, far, cfg.n_samples + 1, device=device)
        lower = t_edges[:-1].unsqueeze(0).expand(Nr, -1)
        upper = t_edges[1:].unsqueeze(0).expand(Nr, -1)
        t_coarse = lower + torch.rand(Nr, cfg.n_samples, device=device) * (upper - lower)

        if getattr(cfg, "n_importance", 0) > 0:
            with torch.no_grad():
                pts_c = rays_o[:, None] + t_coarse[:, :, None] * rays_d[:, None]
                zc = z_rays[:, None].expand(-1, cfg.n_samples, -1).reshape(-1, z.shape[-1])
                bidx_c = bidx[:, None].expand(-1, cfg.n_samples).reshape(-1)
                pf_c = pts_c.reshape(-1, 3)
                if feat_maps is not None and input_c2ws is not None:
                    lf_c = self._project_features(
                        pf_c, bidx_c,
                        input_c2ws.float(), input_Ks.float(),
                        feat_maps, (H, W),
                    )
                else:
                    lf_c = None
                sdf_c = self.sdf_mlp(pf_c, zc, lf_c).squeeze(-1).reshape(Nr, cfg.n_samples)
                w_c = neus_weights(sdf_c, beta_v, t_coarse).detach()
                t_mid = 0.5 * (t_coarse[:, :-1] + t_coarse[:, 1:])
                w_pdf = w_c[:, 1:-1].clamp_min(1e-6)
                t_fine = sample_pdf(t_mid, w_pdf, int(cfg.n_importance), det=False)
                t_vals = torch.sort(torch.cat([t_coarse, t_fine], dim=-1), dim=-1).values
        else:
            t_vals = t_coarse

        n_total = t_vals.shape[1]
        pts = rays_o[:, None] + t_vals[:, :, None] * rays_d[:, None]
        z_pts = z_rays[:, None].expand(-1, n_total, -1).reshape(-1, z.shape[-1])
        bidx_pts = bidx[:, None].expand(-1, n_total).reshape(-1)

        # ---- SDF + gradient (create_graph for eikonal BP) --------------
        with autocast("cuda", enabled=False):
            pts_flat = pts.reshape(-1, 3).detach().requires_grad_(True)
            if feat_maps is not None and input_c2ws is not None:
                local_feat = self._project_features(
                    pts_flat.detach(), bidx_pts,
                    input_c2ws.float(), input_Ks.float(),
                    feat_maps, (H, W),
                )
            else:
                local_feat = None

            sdf_flat = self.sdf_mlp(pts_flat, z_pts, local_feat).squeeze(-1)
            lim = cfg.sdf_clamp
            sdf_flat = sdf_flat.clamp(-lim, lim)
            sdf_grad = torch.autograd.grad(
                sdf_flat,
                pts_flat,
                grad_outputs=torch.ones_like(sdf_flat),
                create_graph=self.training,
                retain_graph=True,
            )[0]

        sdf_vals = sdf_flat.reshape(Nr, n_total)
        normals  = sdf_grad.reshape(Nr, n_total, 3)

        # ---- NeuS weights → depth, normal, surface point ---------------
        weights = neus_weights(sdf_vals, beta_v, t_vals)
        w_sum = weights.sum(-1)
        hit = w_sum > cfg.depth_hit_w_sum_thresh
        # No compositing with a far-plane constant: either there is mass along the ray
        # (depth = conditional mean) or the ray is treated as empty (depth = far).
        w_safe = w_sum.clamp(min=1e-8)
        pred_depth = torch.where(
            hit,
            (weights * t_vals).sum(-1) / w_safe,
            torch.full_like(w_sum, far),
        )
        acc_n = (weights[:, :, None] * normals).sum(1)
        pred_normal = torch.where(
            hit[:, None],
            F.normalize(acc_n, dim=-1, eps=1e-6),
            F.normalize(-rays_d, dim=-1, eps=1e-6),
        )
        # Keep training-time shading stable by orienting normals toward camera.
        view_dot = (pred_normal * (-rays_d)).sum(-1, keepdim=True)
        pred_normal = torch.where(view_dot < 0, -pred_normal, pred_normal)
        x_surf = rays_o + pred_depth[:, None] * rays_d            # (Nr, 3)

        # ---- BRDF + light ----------------------------------------------
        if device.type == "cuda":
            with autocast("cuda", enabled=True):
                albedo, roughness, metalness = self.brdf_head(z_rays)     # (Nr, 3/1/1)
                light_pos, light_int, amb = self.light_head(z_rays)       # (Nr, 3) each
        else:
            albedo, roughness, metalness = self.brdf_head(z_rays)
            light_pos, light_int, amb = self.light_head(z_rays)

        l_dir = F.normalize(light_pos - x_surf, dim=-1)
        v_dir = F.normalize(-rays_d, dim=-1)
        ndl_vals = (pred_normal.detach() * l_dir).sum(-1)   # (Nr,) — for light-facing penalty

        pred_color = cook_torrance_ggx(
            pred_normal, v_dir, l_dir, albedo, roughness, metalness, light_int, amb
        )                                                          # (Nr, 3)

        # ---- Losses ----------------------------------------------------
        # Index GT at sampled pixels (target view = index 0; matches c2w / K / depth / mask).
        img_hwc    = images[:, 0].permute(0, 2, 3, 1)    # (B, H, W, 3)
        depth_hw   = gt_depth[:, 0]                       # (B, H, W)
        normal_hwc = gt_normal.permute(0, 2, 3, 1)        # (B, H, W, 3)
        r, c       = pix_rc[:, 0], pix_rc[:, 1]

        gt_color  = img_hwc[bidx, r, c]                   # (Nr, 3)
        gt_d      = depth_hw[bidx, r, c]                  # (Nr,)
        gt_n      = normal_hwc[bidx, r, c]                # (Nr, 3)

        # fg: foreground pixels — use GT mask when available (exact alpha/threshold),
        # otherwise fall back to the depth-bounds proxy.
        if gt_mask is not None:
            fg = gt_mask[:, 0][bidx, r, c] > 0.5
        else:
            fg = (gt_d > near) & (gt_d < far)
        bg = ~fg
        valid_d = fg & (gt_d > near) & (gt_d < far)   # fg pixels with usable depth

        # Push light to illuminated side; when n·l ≤ 0, BRDF=0 and all render gradients vanish.
        l_light_facing = (
            F.relu(-ndl_vals[fg]).mean() if fg.any() else l_dir.sum() * 0.0
        )

        # Render loss on object pixels.
        l_render_obj = (
            F.l1_loss(pred_color[fg], gt_color[fg])
            if fg.any()
            else pred_color.sum() * 0.0
        )
        # Object-only optimization: ignore background RGB.
        l_render_bg = pred_color.sum() * 0.0
        l_render = l_render_obj

        # Depth loss — valid GT only. Use metric-space L1 directly so large
        # geometric errors produce strong corrective gradients.
        if valid_d.any():
            l_depth = F.l1_loss(pred_depth[valid_d], gt_d[valid_d])
        else:
            l_depth = pred_depth.sum() * 0.0

        # Background rays are known empty from mask: keep SDF positive along the full ray.
        if bg.any():
            l_bg_sdf = F.softplus(cfg.bg_sdf_margin - sdf_vals[bg]).mean()
            l_bg_alpha = w_sum[bg].mean()
        else:
            l_bg_sdf = sdf_vals.sum() * 0.0
            l_bg_alpha = w_sum.sum() * 0.0

        # Silhouette losses on sampled rays (mask-driven contour shaping).
        # Use SDF directly (soft-min along the ray), not accumulated opacity:
        # silhouette is "does the ray intersect the SDF surface?".
        tau = max(float(getattr(cfg, "sil_sdf_tau", 0.05)), 1e-4)
        sdf_softmin = -tau * torch.logsumexp(-sdf_vals / tau, dim=-1)   # approx min_t SDF(t)
        sil_logit = -sdf_softmin / tau                                   # >0 when min SDF < 0
        s_prob = torch.sigmoid(sil_logit).clamp(1e-4, 1 - 1e-4)
        gt_fg = fg.float()
        l_sil_bce = F.binary_cross_entropy_with_logits(sil_logit, gt_fg)
        inter = (s_prob * gt_fg).sum()
        dice = (2.0 * inter + 1e-6) / (s_prob.sum() + gt_fg.sum() + 1e-6)
        l_sil_dice = 1.0 - dice

        # SDF supervision from GT depth:
        #  - SDF(x(gt_depth)) ≈ 0
        #  - points before the surface should be outside (SDF > 0)
        #  - points after the surface should be inside  (SDF < 0)
        # This keeps SDF sign/scale grounded even when NeuS weights saturate.
        if valid_d.any():
            t_valid = t_vals[valid_d]            # (Nv, n_samples)
            sdf_valid = sdf_vals[valid_d]        # (Nv, n_samples)
            gd_valid = gt_d[valid_d]             # (Nv,)

            # Surface-zero constraint at sample closest to GT depth.
            k = (t_valid - gd_valid[:, None]).abs().argmin(dim=-1)           # (Nv,)
            sdf_at_surface = sdf_valid.gather(1, k[:, None]).squeeze(1)      # (Nv,)
            l_sdf_surface = sdf_at_surface.abs().mean()

            # Sign constraints: all samples clearly in front of / behind the GT surface.
            # Use half-bin margin so the surface sample itself isn't penalised.
            dt = (far - near) / max(cfg.n_samples, 1)
            front_mask = t_valid < (gd_valid[:, None] - dt * 0.5)
            back_mask  = t_valid > (gd_valid[:, None] + dt * 0.5)

            if front_mask.any():
                # Penalize negative SDF in front of the observed surface.
                l_sdf_front = F.softplus(-sdf_valid[front_mask]).mean()
            else:
                l_sdf_front = sdf_valid.sum() * 0.0
            if back_mask.any():
                # Penalize positive SDF behind the observed surface.
                l_sdf_back = F.softplus(sdf_valid[back_mask]).mean()
            else:
                l_sdf_back = sdf_valid.sum() * 0.0
            l_sdf_sign = l_sdf_front + l_sdf_back
        else:
            l_sdf_surface = sdf_vals.sum() * 0.0
            l_sdf_sign = sdf_vals.sum() * 0.0

        # Normal loss — valid (non-background) GT normals
        valid_n = gt_n.norm(dim=-1) > 0.5
        if valid_n.any():
            pn, gn = pred_normal[valid_n], gt_n[valid_n]
            pn = F.normalize(pn, dim=-1, eps=1e-6)
            gn = F.normalize(gn, dim=-1, eps=1e-6)
            l_normal = (1.0 - F.cosine_similarity(pn, gn, dim=-1)).mean()
        else:
            l_normal = pred_normal.sum() * 0.0

        # Eikonal loss — ||∇f|| = 1 (cap extreme norms so late-training spikes don't explode)
        gnrm = sdf_grad.norm(dim=-1)
        l_eikonal = ((gnrm.clamp(max=100.0) - 1.0) ** 2).mean()

        # Closure prior (lightweight): keep center inside and far boundary outside.
        with autocast("cuda", enabled=False):
            c_pts = torch.zeros(B, 3, device=device, dtype=torch.float32)
            c_sdf = self.sdf_mlp(c_pts, z_fp32).squeeze(-1)
            l_center_inside = F.softplus(c_sdf + cfg.closure_center_margin).mean()

            n_b = 64
            d = torch.randn(B, n_b, 3, device=device, dtype=torch.float32)
            d = F.normalize(d, dim=-1, eps=1e-6)
            b_pts = d * cfg.mc_bound
            zb = z_fp32[:, None, :].expand(-1, n_b, -1).reshape(-1, z_fp32.shape[-1])
            b_sdf = self.sdf_mlp(b_pts.reshape(-1, 3), zb).squeeze(-1)
            l_boundary_outside = F.softplus(cfg.closure_boundary_margin - b_sdf).mean()
        l_closure = l_center_inside + l_boundary_outside

        total = (
            cfg.lambda_render        * l_render
            + cfg.lambda_bg_render   * l_render_bg
            + cfg.lambda_bg_sdf      * l_bg_sdf
            + cfg.lambda_bg_alpha    * l_bg_alpha
            + cfg.lambda_depth       * l_depth
            + cfg.lambda_normal      * l_normal
            + cfg.lambda_eik         * l_eikonal
            + cfg.lambda_sdf_surface * l_sdf_surface
            + cfg.lambda_sdf_sign    * l_sdf_sign
            + cfg.lambda_sil_bce     * l_sil_bce
            + cfg.lambda_sil_dice    * l_sil_dice
            + cfg.lambda_light_facing * l_light_facing
            + cfg.lambda_closure     * l_closure
        )
        return {
            "total":       total,
            "render":      l_render.detach(),
            "bg_sdf":      l_bg_sdf.detach(),
            "bg_alpha":    l_bg_alpha.detach(),
            "depth":       l_depth.detach(),
            "normal":      l_normal.detach(),
            "eikonal":     l_eikonal.detach(),
            "sdf_surface": l_sdf_surface.detach(),
            "sdf_sign":    l_sdf_sign.detach(),
            "sil_bce":     l_sil_bce.detach(),
            "sil_dice":    l_sil_dice.detach(),
            "light_facing": l_light_facing.detach(),
            "closure":     l_closure.detach(),
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @torch.no_grad()
    def render_image(self, images, input_c2ws, input_Ks, render_c2w, render_K):
        """
        Render depth + normal + shaded color for a full image (no GT needed).
        images:      (B, N, 3, H, W)
        input_c2ws:  (B, N, 4, 4) — cameras for input views (for feature projection)
        input_Ks:    (B, N, 3, 3)
        render_c2w:  (B, 4, 4)    — camera to render from
        render_K:    (B, 3, 3)
        """
        B, N_views, _, H, W = images.shape
        device     = images.device
        cfg        = self.cfg

        z, feat_maps                 = self.encoder(images)
        albedo, roughness, metalness = self.brdf_head(z)
        light_pos, light_int, amb    = self.light_head(z)

        # Build all rays for the full image
        ys = torch.arange(H, device=device, dtype=torch.float32)
        xs = torch.arange(W, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")   # (H, W)

        fx, fy = render_K[0, 0, 0], render_K[0, 1, 1]
        cx, cy = render_K[0, 0, 2], render_K[0, 1, 2]
        dirs_cam = torch.stack([
            (xx - cx) / fx, -(yy - cy) / fy, -torch.ones_like(xx)
        ], dim=-1).reshape(-1, 3)                          # (H*W, 3)
        dirs_world = F.normalize(dirs_cam @ render_c2w[0, :3, :3].T, dim=-1)
        origins    = render_c2w[0, :3, 3].unsqueeze(0).expand(H * W, 3)

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
        amb_exp    = amb.expand(chunk, -1)

        near, far   = cfg.near, cfg.far
        n_s         = cfg.n_samples

        for i in range(0, H * W, chunk):
            ro = origins[i:i+chunk]
            rd = dirs_world[i:i+chunk]
            n  = ro.shape[0]

            zc = z_exp[:n]; ab = ab_exp[:n]; ro_ = ro_exp[:n]
            me = me_exp[:n]; lp = lp_exp[:n]; li = li_exp[:n]
            am = amb_exp[:n]

            # Coarse samples (deterministic), then hierarchical fine resampling.
            t_coarse = torch.linspace(near, far, n_s, device=device).unsqueeze(0).expand(n, -1)
            p_coarse = ro[:, None] + t_coarse[:, :, None] * rd[:, None]
            zp_c = zc[:, None].expand(-1, n_s, -1).reshape(-1, z.shape[-1])
            pts_c = p_coarse.reshape(-1, 3)
            if feat_maps is not None:
                bidx_c = torch.zeros(pts_c.shape[0], dtype=torch.long, device=device)
                lf_c = self._project_features(
                    pts_c, bidx_c,
                    input_c2ws.float(), input_Ks.float(),
                    feat_maps.float(), (H, W),
                )
            else:
                lf_c = None
            sf_c = self.sdf_mlp(pts_c, zp_c, lf_c).squeeze(-1).reshape(n, n_s)
            if getattr(cfg, "n_importance", 0) > 0:
                w_c = neus_weights(sf_c, self.beta, t_coarse)
                t_mid = 0.5 * (t_coarse[:, :-1] + t_coarse[:, 1:])
                w_pdf = w_c[:, 1:-1].clamp_min(1e-6)
                t_fine = sample_pdf(t_mid, w_pdf, int(cfg.n_importance), det=True)
                t = torch.sort(torch.cat([t_coarse, t_fine], dim=-1), dim=-1).values
            else:
                t = t_coarse

            n_tot = t.shape[1]
            p = ro[:, None] + t[:, :, None] * rd[:, None]
            zp = zc[:, None].expand(-1, n_tot, -1).reshape(-1, z.shape[-1])
            pts_chunk = p.reshape(-1, 3)
            if feat_maps is not None:
                bidx_chunk = torch.zeros(pts_chunk.shape[0], dtype=torch.long, device=device)
                lf = self._project_features(
                    pts_chunk, bidx_chunk,
                    input_c2ws.float(), input_Ks.float(),
                    feat_maps.float(), (H, W),
                )
            else:
                lf = None
            sf = self.sdf_mlp(pts_chunk, zp, lf).squeeze(-1).reshape(n, n_tot)

            # Finite-difference normals for inference (no create_graph needed)
            with torch.enable_grad():
                pf  = pts_chunk.requires_grad_(True)
                sf2 = self.sdf_mlp(pf, zp, lf).squeeze(-1)
                g   = torch.autograd.grad(sf2, pf, torch.ones_like(sf2))[0]
            g = g.reshape(n, n_s, 3)

            w  = neus_weights(sf.detach(), self.beta, t)
            w_sum = w.sum(-1)
            hit = w_sum > cfg.depth_hit_w_sum_thresh
            # Dominant-bin depth when there is real mass; otherwise far (no spurious sheet).
            k = w.argmax(dim=-1)  # (n,)
            ridx = torch.arange(n, device=device)
            d = torch.where(hit, t[ridx, k], torch.full((n,), far, device=device, dtype=t.dtype))
            nm_hit = F.normalize(g[ridx, k], dim=-1)
            nm = torch.where(
                hit[:, None],
                nm_hit,
                F.normalize(-rd, dim=-1, eps=1e-6),
            )
            # Orient normals toward the camera for stable visualisation.
            view_dot = (nm * (-rd)).sum(-1, keepdim=True)
            nm = torch.where(view_dot < 0, -nm, nm)
            xs_ = ro + d[:, None] * rd

            l_dir = F.normalize(lp - xs_, dim=-1)
            v_dir = F.normalize(-rd, dim=-1)
            col   = cook_torrance_ggx(nm, v_dir, l_dir, ab, ro_, me, li, am)

            colors[i:i+n] = col
            depths[i:i+n] = d
            norms[i:i+n]  = nm
            # Store visibility in alpha channel proxy for downstream masking.
            # (reuse depths tensor not needed; create lazily below)
            if i == 0:
                opac = torch.zeros(H * W, device=device)
            opac[i:i+n] = w_sum

        return {
            "color":  colors.reshape(H, W, 3).clamp(0, 1),
            "depth":  depths.reshape(H, W),
            "normal": norms.reshape(H, W, 3),
            "opacity": opac.reshape(H, W).clamp(0, 1),
            "hit": (depths < far).reshape(H, W),
        }
