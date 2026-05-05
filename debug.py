"""
Diagnostic checks for PRISM.  Run after training to find failure modes.

python debug.py [--checkpoint model.pt] [--data_root ...]

Checks (in order):
  1. Z diversity          — encoder producing different codes per object?
  2. SDF along a ray      — is there a zero crossing in [near, far]?
  3. NeuS weight mass     — is the surface localized or spread?
  4. Hit-mask fraction    — how many rays actually "hit" something?
  5. BRDF outputs         — albedo / roughness / light position sanity
  6. n·l sign             — why is the render black?
  7. GT mesh scale        — scan coords vs. render world coords
  8. Depth accuracy       — predicted vs. GT depth numerically
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import PRISMConfig
from prism  import PRISM
from prism.renderer import sample_rays, neus_weights
from data.omniobject3d import OmniObject3DDataset

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("debug")

SEP = "=" * 60


def banner(title):
    log.info(f"\n{SEP}\n  {title}\n{SEP}")


def run_debug(cfg: PRISMConfig, n_objects: int = 5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PRISM(cfg).to(device)
    ckpt  = torch.load(cfg.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    log.info(f"Loaded checkpoint: {cfg.checkpoint}")
    log.info(f"  Training epoch: {ckpt.get('epoch', '?')}  step: {ckpt.get('step', '?')}")
    log.info(f"  β = {model.beta.item():.4f}")

    ds     = OmniObject3DDataset(cfg.data_root, split="test", image_size=cfg.image_size)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    # Accumulate stats across objects
    z_list, obj_ids = [], []

    for i, batch in enumerate(loader):
        if i >= n_objects:
            break

        obj_id = batch["object_id"][0]
        image  = batch["image"].to(device)
        c2w    = batch["c2w"].to(device)
        K      = batch["K"].to(device)
        gt_d   = batch["depth"].to(device)      # (1, 1, H, W)
        gt_n   = batch["normal"].to(device)
        mesh_p = batch["mesh_path"][0]

        B, _, H, W = image.shape
        obj_ids.append(obj_id)

        with torch.no_grad():
            z = model.encoder(image)            # (1, latent_dim)
        z_list.append(z.cpu())
        obj_diag = {}

        banner(f"Object {i+1}/{n_objects}: {obj_id}")

        # ------------------------------------------------------------------
        # CHECK 2: SDF zero crossing along one central ray
        # ------------------------------------------------------------------
        with torch.no_grad():
            cam_origin = c2w[0, :3, 3]
            # Ray through image centre
            fx, fy = K[0, 0, 0], K[0, 1, 1]
            cx, cy = K[0, 0, 2], K[0, 1, 2]
            dir_cam = torch.tensor([0.0, 0.0, -1.0], device=device)   # centre ray
            dir_w   = F.normalize(c2w[0, :3, :3] @ dir_cam, dim=-1)

            near, far = cfg.near, cfg.far
            n_s = 128
            t_vals = torch.linspace(near, far, n_s, device=device)
            pts    = cam_origin + t_vals[:, None] * dir_w              # (128, 3)
            z_exp  = z.expand(n_s, -1)
            sdf    = model.sdf_mlp(pts, z_exp).squeeze(-1)             # (128,)

            gt_depth_centre = gt_d[0, 0, H // 2, W // 2].item()
            log.info(f"\n[SDF along centre ray]")
            log.info(f"  Camera origin:     {cam_origin.cpu().numpy().round(3)}")
            log.info(f"  GT depth (centre): {gt_depth_centre:.4f}")
            log.info(f"  SDF range:         min={sdf.min():.4f}  max={sdf.max():.4f}")
            log.info(f"  SDF/β range:       min={(sdf.min() / model.beta):.2f}  max={(sdf.max() / model.beta):.2f}")
            log.info(f"  SDF at near/far:   {sdf[0].item():.4f}  →  {sdf[-1].item():.4f}")

            crossings = ((sdf[:-1] * sdf[1:]) < 0).nonzero(as_tuple=True)[0]
            obj_diag["centre_crossing"] = len(crossings) > 0
            if len(crossings):
                t_cross = t_vals[crossings[0]].item()
                log.info(f"  Zero crossing at:  t = {t_cross:.4f}  (GT depth = {gt_depth_centre:.4f})")
                log.info(f"  -> Depth error:    {abs(t_cross - gt_depth_centre):.4f}")
            else:
                log.info(f"  *** NO zero crossing in [{near}, {far}] — SDF stays {'positive' if sdf.min() > 0 else 'negative'}")
                log.info(f"      SDF at GT depth: {sdf[(t_vals - gt_depth_centre).abs().argmin()].item():.4f}")

        # ------------------------------------------------------------------
        # CHECK 3 & 4: NeuS weight mass and hit fraction
        # ------------------------------------------------------------------
        with torch.no_grad():
            rays_o, rays_d, pix_rc, bidx = sample_rays(c2w, K, H, W, 512, device)
            # Match training stratified sampling (sorted near→far per ray).
            t_edges = torch.linspace(near, far, cfg.n_samples + 1, device=device)
            lower = t_edges[:-1].unsqueeze(0).expand(rays_o.shape[0], -1)
            upper = t_edges[1:].unsqueeze(0).expand(rays_o.shape[0], -1)
            t_v = lower + torch.rand(rays_o.shape[0], cfg.n_samples, device=device) * (upper - lower)
            pts_r   = rays_o[:, None] + t_v[:, :, None] * rays_d[:, None]
            z_r     = z.expand(rays_o.shape[0], -1)
            z_p     = z_r[:, None].expand(-1, cfg.n_samples, -1).reshape(-1, z.shape[-1])
            sdf_r   = model.sdf_mlp(pts_r.reshape(-1, 3), z_p).squeeze(-1)
            sdf_v   = sdf_r.reshape(rays_o.shape[0], cfg.n_samples)

            w = neus_weights(sdf_v, model.beta)        # (N_rays, N_samples)
            # Diagnostic: sign-flipped SDF. If this suddenly "works", NeuS sign
            # convention and SDF sign are inconsistent somewhere in the pipeline.
            w_flip = neus_weights(-sdf_v, model.beta)
            w_sum = w.sum(-1)                           # (N_rays,)
            w_sum_flip = w_flip.sum(-1)
            hit   = (w_sum > 0.1)
            hit_flip = (w_sum_flip > 0.1)
            crossing_any = ((sdf_v[:, :-1] * sdf_v[:, 1:]) < 0).any(dim=-1)
            sat_frac = ((sdf_v.abs() / model.beta) > 8.0).float().mean()
            neg_frac = (sdf_v < 0).float().mean()
            pos_frac = (sdf_v > 0).float().mean()
            near_sdf = sdf_v[:, 0]
            far_sdf = sdf_v[:, -1]

            obj_diag["hit_frac"] = hit.float().mean().item()
            obj_diag["hit_frac_flip"] = hit_flip.float().mean().item()
            obj_diag["ray_cross_frac"] = crossing_any.float().mean().item()
            obj_diag["sat_frac"] = sat_frac.item()
            obj_diag["neg_frac"] = neg_frac.item()

            log.info(f"\n[NeuS weights  (512 sampled rays)]")
            log.info(f"  Weight sum:  mean={w_sum.mean():.4f}  max={w_sum.max():.4f}")
            log.info(f"  Hit fraction (w_sum > 0.1):  {hit.float().mean():.3f}  ({hit.sum()}/{len(hit)})")
            log.info(f"  SDF values:  mean={sdf_v.mean():.4f}  std={sdf_v.std():.4f}")
            log.info(f"  SDF sign:    neg={neg_frac:.3f}  pos={pos_frac:.3f}")
            log.info(f"  SDF near/far means: {near_sdf.mean():.4f}  →  {far_sdf.mean():.4f}")
            log.info(f"  Ray sign-change fraction: {crossing_any.float().mean():.3f}")
            log.info(f"  Sigmoid saturation frac (|sdf|/β > 8): {sat_frac:.3f}")
            log.info(f"  Flipped-sign hit fraction: {hit_flip.float().mean():.3f}  ({hit_flip.sum()}/{len(hit_flip)})")
            if not hit.any():
                log.info(f"  *** ALL RAYS MISS — render loss = 0 during training!")
                if hit_flip.float().mean() > 0.5:
                    log.info(f"  *** SIGN MISMATCH SUSPECTED: using -SDF gives many hits")
                if sat_frac > 0.9:
                    log.info(f"  *** LOGISTIC SATURATION: |SDF| >> β on almost all samples")
                if neg_frac > 0.99:
                    log.info(f"  *** SDF is almost entirely negative (camera likely inside implicit interior)")

        # ------------------------------------------------------------------
        # CHECK 5 & 6: BRDF outputs and n·l sign
        # ------------------------------------------------------------------
        with torch.no_grad():
            albedo, roughness, metalness = model.brdf_head(z)
            light_pos, light_int        = model.light_head(z)
            pred_d = (w[:1].cpu() * t_v[:1].cpu()).sum(-1).item() if hit.any() else None

        # n_hat = grad SDF needs autograd; cannot run inside ``no_grad`` above.
        if hit.any() and pred_d is not None:
            x_surf = cam_origin + pred_d * dir_w
            l_dir  = F.normalize(light_pos[0] - x_surf, dim=-1)
            v_dir  = F.normalize(-dir_w, dim=-1)
            pf = x_surf.unsqueeze(0).detach().clone().requires_grad_(True)
            z_f = z.detach()
            with torch.enable_grad():
                sf_s = model.sdf_mlp(pf, z_f)
                g_s = torch.autograd.grad(
                    sf_s, pf, torch.ones_like(sf_s), create_graph=False, retain_graph=False,
                )[0][0]
            n_hat = F.normalize(g_s.detach(), dim=-1)
            ndl = (n_hat * l_dir).sum().item()
            ndv = (n_hat * v_dir).sum().item()
            log.info(f"\n[BRDF / lighting at surface point]")
            log.info(f"  Surface point:    {x_surf.cpu().numpy().round(3)}")
            log.info(f"  Light position:   {light_pos[0].cpu().numpy().round(3)}")
            log.info(f"  Light intensity:  {light_int[0].cpu().numpy().round(3)}")
            log.info(f"  n·l (should be >0 for lit surface): {ndl:.4f}")
            log.info(f"  n·v (should be >0):                 {ndv:.4f}")
            if ndl <= 0:
                log.info(f"  *** n·l ≤ 0 → Cook-Torrance returns BLACK")
        else:
            log.info(f"\n[BRDF / lighting]  (skipped — no hit rays)")

        with torch.no_grad():
            log.info(f"\n[BRDF params]")
            log.info(f"  Albedo:     {albedo[0].cpu().numpy().round(3)}")
            log.info(f"  Roughness:  {roughness[0].item():.4f}")
            log.info(f"  Metalness:  {metalness[0].item():.4f}")

        # ------------------------------------------------------------------
        # CHECK 7: GT mesh scale
        # ------------------------------------------------------------------
        try:
            import trimesh
            gt_mesh = trimesh.load(mesh_p, force="mesh")
            bounds  = gt_mesh.bounds              # [[xmin,ymin,zmin],[xmax,ymax,zmax]]
            extent  = bounds[1] - bounds[0]
            centre  = (bounds[0] + bounds[1]) / 2
            log.info(f"\n[GT mesh coordinate range]")
            log.info(f"  Bounds:   {bounds[0].round(3)} → {bounds[1].round(3)}")
            log.info(f"  Extent:   {extent.round(3)}  (total: {extent.max():.3f})")
            log.info(f"  Centre:   {centre.round(3)}")
            log.info(f"  Marching cubes samples: [{-cfg.mc_bound}, {cfg.mc_bound}]^3")
            if extent.max() > 5 * cfg.mc_bound:
                log.info(f"  *** GT mesh is {extent.max():.1f} units wide but marching "
                         f"cubes is only ±{cfg.mc_bound} — SCALE MISMATCH")
        except Exception as e:
            log.info(f"\n[GT mesh]  Could not load: {e}")

        # ------------------------------------------------------------------
        # CHECK 8: Predicted vs GT depth
        # ------------------------------------------------------------------
        with torch.no_grad():
            pred_depths = (w * t_v).sum(-1)                       # (N_rays,)
            r, c = pix_rc[:, 0], pix_rc[:, 1]
            gt_depths_sampled = gt_d[0, 0, r, c]                  # (N_rays,)
            valid_gt = (gt_depths_sampled > near) & (gt_depths_sampled < far)
            valid = valid_gt & hit

            log.info(f"\n[Depth accuracy  (on hit rays with valid GT)]")
            log.info(
                f"  GT depth in [{near:.1f}, {far:.1f}]: {valid_gt.float().mean():.3f}  "
                f"({valid_gt.sum()}/{len(valid_gt)})"
            )
            if valid.any():
                pd = pred_depths[valid].cpu()
                gd = gt_depths_sampled[valid].cpu()
                err = (pd - gd).abs()
                log.info(f"  Pred depth:  mean={pd.mean():.4f}  std={pd.std():.4f}")
                log.info(f"  GT depth:    mean={gd.mean():.4f}  std={gd.std():.4f}")
                log.info(f"  L1 error:    mean={err.mean():.4f}  median={err.median():.4f}")
                rel = err / gd
                log.info(f"  Rel error:   mean={rel.mean():.3f}  (1.0 = 100%)")
            else:
                log.info(f"  *** No valid (hit + GT) rays to compare")

        # ------------------------------------------------------------------
        # Per-object diagnosis summary
        # ------------------------------------------------------------------
        log.info(f"\n[Diagnosis]")
        if obj_diag.get("hit_frac", 0.0) == 0.0:
            if obj_diag.get("hit_frac_flip", 0.0) > 0.5:
                log.info("  Primary failure: NeuS/SDF sign convention likely inconsistent")
            elif obj_diag.get("sat_frac", 0.0) > 0.9:
                log.info("  Primary failure: SDF magnitude saturation (|sdf|/β too large)")
            elif obj_diag.get("neg_frac", 0.0) > 0.99:
                log.info("  Primary failure: field collapsed to all-negative values")
            else:
                log.info("  Primary failure: no rendered hits (see SDF/weight stats above)")
        else:
            log.info("  Geometry is producing hit rays; inspect depth/normal errors")

    # ------------------------------------------------------------------
    # CHECK 1: Z diversity across objects
    # ------------------------------------------------------------------
    banner("Z diversity across objects")
    if len(z_list) >= 2:
        Z = torch.cat(z_list, dim=0)                          # (N, latent_dim)
        pairwise = torch.cdist(Z, Z)                          # (N, N) L2
        off_diag = pairwise[~torch.eye(len(Z), dtype=bool)]
        log.info(f"  Z pairwise L2:  mean={off_diag.mean():.4f}  min={off_diag.min():.4f}")
        log.info(f"  Z norm per obj: {Z.norm(dim=-1).numpy().round(3)}")
        if off_diag.mean() < 0.5:
            log.info(f"  *** z vectors are very similar — encoder may not be "
                     f"differentiating objects")
        else:
            log.info(f"  z vectors are diverse  (encoder is working)")
    else:
        log.info("  Need ≥ 2 objects")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--data_root",  type=str)
    parser.add_argument("--image_size", type=int)
    parser.add_argument("--n_objects",  type=int, default=3)
    args = parser.parse_args()

    cfg = PRISMConfig()
    for k in ("checkpoint", "data_root", "image_size"):
        if getattr(args, k) is not None:
            setattr(cfg, k, getattr(args, k))

    run_debug(cfg, n_objects=args.n_objects)
