"""
debug.py  —  PRISM diagnostics.  Ends with a concrete action plan.

    python debug.py [--checkpoint model.pt] [--n_objects N]
    python debug.py --overfit [--overfit_object ID]   # same single-object scope as train/visualize
"""
import argparse, logging, numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from config import PRISMConfig
from prism import PRISM
from prism.brdf import cook_torrance_ggx
from prism.renderer import sample_rays, neus_weights
from data.omniobject3d import OmniObject3DDataset, enumerate_object_pairs

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger()

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"

# ── measurements ──────────────────────────────────────────────────────────────

def m_eikonal(model, z, cfg, device):
    n = 2048
    pts = (torch.rand(n, 3, device=device) * 2 - 1) * cfg.mc_bound
    pg = pts.requires_grad_(True)
    with torch.enable_grad():
        s = model.sdf_mlp(pg, z.expand(n, -1)).squeeze(-1)
        g = torch.autograd.grad(s, pg, torch.ones_like(s))[0]
    norms = g.norm(dim=-1).detach().cpu()
    mean = norms.mean().item()
    tight = ((norms > 0.9) & (norms < 1.1)).float().mean().item()
    status = PASS if (0.8 < mean < 1.3 and tight > 0.5) else \
             FAIL if (mean > 2.0 or tight < 0.2) else WARN
    return {"status": status, "mean": mean, "tight": tight}


def m_sharpness(weights, t_vals):
    w_peak = weights.max(dim=-1).values
    mean_peak = w_peak.mean().item()
    status = PASS if mean_peak > 0.55 else FAIL if mean_peak < 0.30 else WARN
    return {"status": status, "w_peak": mean_peak}


def m_sdf_isotropy(model, z, cfg, device):
    """Blob diagnostic: check if SDF has essentially no angular variation (= sphere).

    Evaluates SDF at N random directions on a sphere of each test radius.
    Low std/|mean| ratio means SDF is nearly spherically symmetric.
    """
    n_dirs = 512
    radii = [0.2, 0.35, 0.5]
    iso_scores = {}
    with torch.no_grad():
        dirs = F.normalize(torch.randn(n_dirs, 3, device=device), dim=-1)
        for r in radii:
            pts = dirs * r
            vals = model.sdf_mlp(pts, z.expand(n_dirs, -1)).squeeze(-1).cpu()
            std  = vals.std().item()
            mean = vals.mean().item()
            ratio = abs(std / (mean + 1e-8))
            iso_scores[r] = {"std": std, "mean": mean, "ratio": ratio}
    # At r=sdf_init_radius, a healthy SDF should vary significantly across angles.
    r_key = 0.35
    ratio = iso_scores[r_key]["ratio"]
    status = PASS if ratio > 0.15 else FAIL if ratio < 0.05 else WARN
    return {"status": status, "radii": iso_scores}


def m_depth_hit(w_sum, gt_d, near, far, hit_thresh):
    """What fraction of foreground rays (valid GT depth) have w_sum > hit_thresh?

    Low hit_frac means depth loss is barely receiving gradients (no surface mass where
    the object is) — the primary mechanism keeping the model stuck as a blob.
    """
    fg = (gt_d > near) & (gt_d < far)
    if not fg.any():
        return {"status": WARN, "note": "no fg rays"}
    ws_fg = w_sum[fg].cpu()
    hit_frac = (ws_fg > hit_thresh).float().mean().item()
    mean_wsum = ws_fg.mean().item()
    status = PASS if hit_frac > 0.60 else FAIL if hit_frac < 0.30 else WARN
    return {"status": status, "hit_frac": hit_frac, "mean_wsum_fg": mean_wsum}


def m_sdf_at_surface(sdf, t, gt_d, near, far):
    """SDF value at the sample closest to GT depth — should be ~0 if surface is placed correctly."""
    valid = (gt_d > near) & (gt_d < far)
    if not valid.any():
        return {"status": WARN, "note": "no valid GT rays"}
    sv, tv, gd = sdf[valid], t[valid], gt_d[valid]
    k = (tv - gd[:, None]).abs().argmin(dim=-1)
    sdf_surf = sv.gather(1, k[:, None]).squeeze(1).cpu()
    mean_abs = sdf_surf.abs().mean().item()
    mean_val = sdf_surf.mean().item()
    status = PASS if mean_abs < 0.05 else FAIL if mean_abs > 0.20 else WARN
    return {"status": status, "mean_abs": mean_abs, "mean": mean_val}


def m_sdf_sign(sdf, t, gt_d, near, far):
    valid = (gt_d > near) & (gt_d < far)
    if not valid.any():
        return {"status": WARN, "note": "no valid GT rays", "correct_frac": float("nan")}
    sv, tv, gd = sdf[valid], t[valid], gt_d[valid]
    dt = (far - near) / sv.shape[1]
    correct = []
    front_means, back_means = [], []
    for i in range(sv.shape[0]):
        f = tv[i] < gd[i] - dt;  b = tv[i] > gd[i] + dt
        fm = sv[i, f].mean().item() if f.any() else float("nan")
        bm = sv[i, b].mean().item() if b.any() else float("nan")
        front_means.append(fm); back_means.append(bm)
        correct.append((not np.isnan(fm) and fm > 0) and (not np.isnan(bm) and bm < 0))
    cf = float(np.mean(correct))
    fm = float(np.nanmean(front_means)); bm = float(np.nanmean(back_means))
    status = PASS if cf > 0.65 else FAIL if cf < 0.35 else WARN
    inverted = fm < 0 and bm > 0
    return {"status": status, "correct_frac": cf, "front": fm, "back": bm, "inverted": inverted}


def m_sdf_band_sign(sdf_front: torch.Tensor, sdf_back: torch.Tensor):
    """
    Training-aligned local sign check at d-δ / d+δ:
      front should be positive, back should be negative.
    """
    if sdf_front.numel() == 0:
        return {"status": WARN, "note": "no valid GT rays"}
    sf = sdf_front.detach().cpu()
    sb = sdf_back.detach().cpu()
    correct = ((sf > 0) & (sb < 0)).float().mean().item()
    margin = (sf - sb).mean().item()
    status = PASS if correct > 0.65 else FAIL if correct < 0.35 else WARN
    return {
        "status": status,
        "correct_frac": correct,
        "front": sf.mean().item(),
        "back": sb.mean().item(),
        "margin": margin,
    }


def m_depth(pred_d, gt_d, near, far):
    hit = pred_d > near * 0.9
    valid = (gt_d > near) & (gt_d < far)
    both = hit & valid
    if not both.any():
        return {"status": FAIL, "note": "no valid+hit rays"}
    pd = pred_d[both].cpu().numpy(); gd = gt_d[both].cpu().numpy()
    err = np.abs(pd - gd)
    bias = float((pd - gd).mean())
    corr = float(np.corrcoef(pd, gd)[0, 1]) if len(pd) > 2 else float("nan")
    l1 = float(err.mean())
    status = PASS if (l1 < 0.15 and corr > 0.70) else \
             FAIL if (l1 > 0.35 or corr < 0.40) else WARN
    return {"status": status, "l1": l1, "corr": corr, "bias": bias, "n": int(both.sum())}


def m_normals(pred_n, gt_n):
    valid = gt_n.norm(dim=-1) > 0.5
    if not valid.any():
        return {"status": WARN, "note": "no valid GT normals"}
    cos = F.cosine_similarity(
        F.normalize(pred_n[valid], dim=-1),
        F.normalize(gt_n[valid],  dim=-1), dim=-1
    ).abs()
    mean_cos = cos.mean().item()
    angle = float(torch.acos(cos.clamp(0, 1)).rad2deg().mean())
    status = PASS if mean_cos > 0.80 else FAIL if mean_cos < 0.60 else WARN
    return {"status": status, "cos": mean_cos, "angle_deg": angle}


def m_light(model, z, pred_n, pred_d, rays_o, rays_d, gt_col, valid_d):
    N = pred_n.shape[0]
    with torch.no_grad():
        ab, ro, me = model.brdf_head(z.unsqueeze(0))
        lp, li, amb = model.light_head(z.unsqueeze(0))
    ab = ab.expand(N,-1); ro = ro.expand(N,-1); me = me.expand(N,-1)
    lp_v = lp.expand(N,-1); li_v = li.expand(N,-1); amb_v = amb.expand(N,-1)
    x    = rays_o + pred_d[:,None] * rays_d
    v    = F.normalize(-rays_d, dim=-1)
    l    = F.normalize(lp_v - x, dim=-1)
    ndl  = (pred_n * l).sum(-1)
    l_ora   = F.normalize(rays_o[0:1].expand(N,-1) - x, dim=-1)
    with torch.no_grad():
        col_pred = cook_torrance_ggx(pred_n, v, l,     ab, ro, me, li_v, amb_v)
        col_ora  = cook_torrance_ggx(pred_n, v, l_ora, ab, ro, me, li_v, amb_v)
    frac_pos = (ndl > 0).float().mean().item()
    status = PASS if frac_pos > 0.75 else FAIL if frac_pos < 0.50 else WARN
    result = {"status": status, "frac_pos": frac_pos, "mean_ndl": ndl.mean().item(),
              "light_pos": lp[0].cpu().numpy().round(3)}
    if valid_d.any():
        result["l1_pred"] = F.l1_loss(col_pred[valid_d], gt_col[valid_d]).item()
        result["l1_ora"]  = F.l1_loss(col_ora[valid_d],  gt_col[valid_d]).item()
    return result


def m_mask(model, batch, cfg, device):
    images      = batch["images"].to(device)
    input_c2ws  = batch["input_c2ws"].to(device)
    input_Ks    = batch["input_Ks"].to(device)
    c2w         = batch["c2w"].to(device)
    K           = batch["K"].to(device)
    gt_d        = batch["depth"]
    B, N, C, H, W = images.shape
    scale = 4; Hs, Ws = H // scale, W // scale
    img_s = F.interpolate(
        images.reshape(B * N, C, H, W), (Hs, Ws),
        mode="bilinear", align_corners=False,
    ).reshape(B, N, C, Hs, Ws)
    Ks = K.clone()
    for i, j in [(0,0),(1,1),(0,2),(1,2)]: Ks[:,i,j] /= scale
    iKs_s = input_Ks.clone()
    for i, j in [(0,0),(1,1),(0,2),(1,2)]: iKs_s[:,:,i,j] /= scale
    with torch.no_grad():
        out = model.render_image(img_s, input_c2ws, iKs_s, c2w, Ks)
    pred_d = out["depth"].cpu().numpy()
    gd = gt_d[0,0].numpy()[::scale, ::scale][:Hs, :Ws]
    gt_fg = (gd > cfg.near) & (gd < cfg.far)
    # Match visualize.py: predicted foreground is where rendered depth hits before far.
    pred_fg = pred_d < (cfg.far * 0.98)
    tp = int((pred_fg & gt_fg).sum()); fp = int((pred_fg & ~gt_fg).sum())
    fn = int((~pred_fg & gt_fg).sum())
    iou = tp / (tp + fp + fn + 1e-8)

    def _erode3x3(mask: np.ndarray) -> np.ndarray:
        h, w = mask.shape
        p = np.pad(mask, 1, mode="constant", constant_values=False)
        out = np.ones_like(mask, dtype=bool)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                out &= p[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
        return out

    def _dilate3x3(mask: np.ndarray) -> np.ndarray:
        h, w = mask.shape
        p = np.pad(mask, 1, mode="constant", constant_values=False)
        out = np.zeros_like(mask, dtype=bool)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                out |= p[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
        return out

    gt_b = gt_fg & (~_erode3x3(gt_fg))
    pr_b = pred_fg & (~_erode3x3(pred_fg))
    gt_b_d = _dilate3x3(gt_b)
    pr_b_d = _dilate3x3(pr_b)
    b_prec = float((pr_b & gt_b_d).sum() / (pr_b.sum() + 1e-8))
    b_rec = float((gt_b & pr_b_d).sum() / (gt_b.sum() + 1e-8))
    b_f1 = float((2 * b_prec * b_rec) / (b_prec + b_rec + 1e-8))

    status = PASS if iou > 0.60 else FAIL if iou < 0.35 else WARN
    return {"status": status, "iou": iou,
            "prec": tp/(tp+fp+1e-8), "rec": tp/(tp+fn+1e-8),
            "b_prec": b_prec, "b_rec": b_rec, "b_f1": b_f1}


def m_z_diversity(z_list):
    if len(z_list) < 2:
        return {"status": WARN, "note": "need ≥2 objects"}
    Z  = torch.cat(z_list, dim=0)
    pw = torch.cdist(Z, Z)
    od = pw[~torch.eye(len(Z), dtype=bool)]
    mean_l2 = od.mean().item()
    status = PASS if mean_l2 > 1.0 else FAIL if mean_l2 < 0.3 else WARN
    return {"status": status, "mean_l2": mean_l2}


# ── reporting ─────────────────────────────────────────────────────────────────

def fmt(status):
    return {"PASS": "✓", "WARN": "~", "FAIL": "✗"}[status]


def row(name, m, detail):
    return f"  {fmt(m['status'])} {name:<16} {detail}"


def print_results(obj_id, metrics):
    log.info(f"\n{obj_id}")
    mk = metrics

    log.info(row("eikonal",   mk["eikonal"],
                 f"‖∇f‖={mk['eikonal']['mean']:.2f}  tight={mk['eikonal']['tight']:.0%}"))
    log.info(row("sharpness", mk["sharpness"],
                 f"w_peak={mk['sharpness']['w_peak']:.3f}"))

    # ── Blob diagnostics ───────────────────────────────────────────────────────
    iso = mk["sdf_isotropy"]
    iso_detail = "  ".join(
        f"r={r:.2f}→(std={v['std']:.3f},mean={v['mean']:+.3f},ratio={v['ratio']:.3f})"
        for r, v in iso["radii"].items()
    )
    log.info(row("sdf_isotropy", iso, iso_detail))

    dh = mk["depth_hit"]
    if "note" in dh:
        log.info(row("depth_hit", dh, dh["note"]))
    else:
        log.info(row("depth_hit", dh,
                     f"hit_frac={dh['hit_frac']:.0%}  mean_wsum_fg={dh['mean_wsum_fg']:.3f}"))

    ss = mk["sdf_at_surf"]
    if "note" in ss:
        log.info(row("sdf_at_surf", ss, ss["note"]))
    else:
        log.info(row("sdf_at_surf", ss,
                     f"mean_abs={ss['mean_abs']:.3f}  mean={ss['mean']:+.3f}"))

    s = mk["sdf_sign"]
    if "note" in s:
        log.info(row("sdf_sign", s, s["note"]))
    else:
        inv = "  *** INVERTED" if s.get("inverted") else ""
        log.info(row("sdf_sign", s,
                     f"correct={s['correct_frac']:.0%}  front={s['front']:+.2f}  back={s['back']:+.2f}{inv}"))

    sb = mk["sdf_band"]
    if "note" in sb:
        log.info(row("sdf_band", sb, sb["note"]))
    else:
        log.info(row("sdf_band", sb,
                     f"correct={sb['correct_frac']:.0%}  front={sb['front']:+.2f}  "
                     f"back={sb['back']:+.2f}  margin={sb['margin']:+.2f}"))

    d = mk["depth"]
    if "note" in d:
        log.info(row("depth", d, d["note"]))
    else:
        log.info(row("depth", d,
                     f"L1={d['l1']:.3f}  corr={d['corr']:.2f}  bias={d['bias']:+.3f}  n={d['n']}"))

    n = mk["normals"]
    if "note" in n:
        log.info(row("normals", n, n["note"]))
    else:
        log.info(row("normals", n, f"cos={n['cos']:.2f}  angle={n['angle_deg']:.0f}°"))

    lt = mk["light"]
    detail = f"n·l frac>0={lt['frac_pos']:.0%}  mean={lt['mean_ndl']:+.2f}"
    if "l1_pred" in lt:
        detail += f"  render_L1 pred={lt['l1_pred']:.3f} oracle={lt['l1_ora']:.3f}"
    log.info(row("light", lt, detail))

    msk = mk["mask"]
    if "note" in msk:
        log.info(row("mask", msk, msk["note"]))
    else:
        log.info(row("mask", msk,
                     f"IoU={msk['iou']:.2f}  prec={msk['prec']:.2f}  rec={msk['rec']:.2f}  "
                     f"bF1={msk['b_f1']:.2f}"))


def action_plan(all_metrics, cfg):
    def avg(key, subkey):
        vals = [m[key].get(subkey, float("nan")) for m in all_metrics]
        return float(np.nanmean([v for v in vals if not np.isnan(v)]))

    eik_mean  = avg("eikonal",    "mean")
    eik_tight = avg("eikonal",    "tight")
    ndl_frac  = avg("light",      "frac_pos")
    sdf_cf    = avg("sdf_sign",   "correct_frac")
    sdf_inv   = any(m["sdf_sign"].get("inverted", False) for m in all_metrics)
    hit_frac  = avg("depth_hit",  "hit_frac")
    surf_abs  = avg("sdf_at_surf","mean_abs")

    iso_ratios = [m["sdf_isotropy"]["radii"].get(0.35, {}).get("ratio", float("nan"))
                  for m in all_metrics]
    iso_ratio = float(np.nanmean([v for v in iso_ratios if not np.isnan(v)]))

    eik_bad  = eik_mean > 1.5 or eik_tight < 0.35
    light_bad = ndl_frac < 0.60
    sign_bad  = sdf_cf < 0.45 and not eik_bad
    blob_iso  = iso_ratio < 0.10
    no_hit    = hit_frac < 0.40

    causes, actions = [], []

    if blob_iso:
        causes.append(
            f"SDF is nearly spherical (angular variation ratio={iso_ratio:.3f} at r=0.35) "
            f"→ root cause of blob rendering"
        )
        if surf_abs > 0.15:
            causes.append(
                f"  SDF at GT surface = {surf_abs:.3f} (should be ~0) "
                f"→ sphere init not pulled to actual shape yet"
            )
            actions.append(
                f"  config.py: lambda_sdf_surface {cfg.lambda_sdf_surface:.2f}"
                f" → {min(2.0, cfg.lambda_sdf_surface * 4):.2f}"
            )
            actions.append(
                f"  config.py: lambda_depth {cfg.lambda_depth:.2f}"
                f" → {min(2.0, cfg.lambda_depth * 2):.2f}"
            )
        if sdf_cf < 0.50:
            actions.append(
                f"  config.py: lambda_sdf_sign {cfg.lambda_sdf_sign:.2f}"
                f" → {min(1.0, cfg.lambda_sdf_sign * 5):.2f}"
            )

    if no_hit:
        causes.append(
            f"w_sum on fg rays too low (hit_frac={hit_frac:.0%}) "
            f"→ depth/normal loss gradients barely reach the surface"
        )
        actions.append(
            "  config.py: beta_anneal_start 2.0 → 0.5  "
            "(start sharper so weights concentrate earlier)"
        )
        actions.append(
            f"  config.py: lambda_sil_bce {cfg.lambda_sil_bce:.2f}"
            f" → {min(4.0, cfg.lambda_sil_bce * 2):.2f}"
        )

    if eik_bad:
        causes.append(
            f"eikonal violated (‖∇f‖={eik_mean:.2f}, {eik_tight:.0%} tight) "
            f"→ noisy normals, diffuse surface"
        )
        actions.append(
            f"  config.py: lambda_eik {cfg.lambda_eik:.2f}"
            f" → {min(1.0, cfg.lambda_eik * 5):.2f}"
        )
        actions.append(
            f"  config.py: n_freqs {cfg.n_freqs} → {max(1, cfg.n_freqs - 2)}"
        )

    if light_bad:
        causes.append(
            f"light behind surface (n·l frac>0={ndl_frac:.0%}) "
            f"→ BRDF black → zero render gradient"
        )
        actions.append(
            f"  config.py: lambda_light_facing {cfg.lambda_light_facing:.2f}"
            f" → {min(1.0, cfg.lambda_light_facing * 3):.2f}"
        )

    if sdf_inv:
        causes.append("SDF sign is INVERTED (negative outside, positive inside)")
        actions.append("  prism/sdf_mlp.py: negate output of self.out(h)")

    if sign_bad and not sdf_inv:
        causes.append(f"SDF sign ordering incorrect ({sdf_cf:.0%} correct)")
        actions.append(
            f"  config.py: lambda_sdf_sign {cfg.lambda_sdf_sign:.2f}"
            f" → {min(0.5, cfg.lambda_sdf_sign * 5):.2f}"
        )
        actions.append(
            f"  config.py: lambda_sdf_surface {cfg.lambda_sdf_surface:.2f}"
            f" → {min(1.0, cfg.lambda_sdf_surface * 3):.2f}"
        )

    log.info("\n" + "─"*60)
    log.info("DIAGNOSIS")
    if not causes:
        log.info("  No critical failures detected")
    else:
        for c in causes:
            log.info(f"  • {c}")
    log.info("\nACTION PLAN")
    if not actions:
        log.info("  Nothing to do — check eval metrics")
    else:
        for i, a in enumerate(actions, 1):
            log.info(f"  {i}.{a}")
    log.info("")


# ── main ──────────────────────────────────────────────────────────────────────

def run_debug(cfg, n_objects=1, restrict_object_ids=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PRISM(cfg).to(device)
    ckpt  = torch.load(cfg.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    log.info(f"{cfg.checkpoint}  epoch={ckpt.get('epoch','?')}  "
             f"step={ckpt.get('step','?')}  β={model.beta.item():.3f}")

    ds     = OmniObject3DDataset(
        cfg.data_root, split="test",
        image_size=cfg.image_size, n_input_views=cfg.n_input_views,
        restrict_object_ids=restrict_object_ids,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    z_list, all_metrics = [], []

    for i, batch in enumerate(loader):
        if i >= n_objects: break

        obj_id      = batch["object_id"][0]
        images      = batch["images"].to(device)       # (B, N_views, 3, H, W)
        input_c2ws  = batch["input_c2ws"].to(device)   # (B, N_views, 4, 4)
        input_Ks    = batch["input_Ks"].to(device)     # (B, N_views, 3, 3)
        c2w         = batch["c2w"].to(device)          # (B, 4, 4)
        K_mat       = batch["K"].to(device)            # (B, 3, 3)
        gt_d        = batch["depth"].to(device)        # (B, 1, H, W)
        gt_n        = batch["normal"].to(device)       # (B, 3, H, W)
        B, N_views, _, H, W = images.shape

        with torch.no_grad():
            z, feat_maps = model.encoder(images)    # z: (B, latent_dim)
        z_list.append(z.cpu())

        near, far = cfg.near, cfg.far
        with torch.no_grad():
            rays_o, rays_d, pix_rc, bidx = sample_rays(c2w, K_mat, H, W, 512, device)
        Nr = rays_o.shape[0]

        t_e = torch.linspace(near, far, cfg.n_samples + 1, device=device)
        t_v = (t_e[:-1].unsqueeze(0).expand(Nr, -1)
               + torch.rand(Nr, cfg.n_samples, device=device)
               * (t_e[1:] - t_e[:-1]).unsqueeze(0))
        pts = rays_o[:, None] + t_v[:, :, None] * rays_d[:, None]   # (Nr, n_samples, 3)

        # Per-ray latent: index by bidx (which batch element each ray belongs to).
        z_pts = z[bidx][:, None].expand(-1, cfg.n_samples, -1).reshape(-1, z.shape[-1])
        bidx_pts = bidx[:, None].expand(-1, cfg.n_samples).reshape(-1)

        pf = pts.reshape(-1, 3).requires_grad_(True)
        if feat_maps is not None:
            lf = model._project_features(
                pf.detach(), bidx_pts,
                input_c2ws.float(), input_Ks.float(),
                feat_maps.float(), (H, W),
            )
        else:
            lf = None
        with torch.enable_grad():
            sf = model.sdf_mlp(pf, z_pts.detach(), lf).squeeze(-1)
            sg = torch.autograd.grad(sf, pf, torch.ones_like(sf))[0]
        sdf_v = sf.detach().reshape(Nr, cfg.n_samples)
        norms = sg.detach().reshape(Nr, cfg.n_samples, 3)

        with torch.no_grad():
            w      = neus_weights(sdf_v, model.beta, t_v)
            w_sum  = w.sum(-1)
            w_safe = w_sum.clamp(min=1e-8)
            pred_d = (w * t_v).sum(-1) / w_safe
            pred_n = F.normalize((w[:, :, None] * norms).sum(1), dim=-1)
            vdot   = (pred_n * (-rays_d)).sum(-1, keepdim=True)
            pred_n = torch.where(vdot < 0, -pred_n, pred_n)

            r_idx, c_idx = pix_rc[:, 0], pix_rc[:, 1]
            gt_col = images[:, 0].permute(0, 2, 3, 1)[bidx, r_idx, c_idx]   # (Nr, 3)
            gt_dep = gt_d[:, 0][bidx, r_idx, c_idx]                          # (Nr,)
            gt_nor = gt_n.permute(0, 2, 3, 1)[bidx, r_idx, c_idx]           # (Nr, 3)
            valid  = (gt_dep > near) & (gt_dep < far)

        delta = float(getattr(cfg, "sdf_band_delta", 0.03))
        if valid.any():
            d_front = (gt_dep[valid] - delta).clamp(min=near, max=far)
            d_back = (gt_dep[valid] + delta).clamp(min=near, max=far)
            p_front = rays_o[valid] + d_front[:, None] * rays_d[valid]
            p_back = rays_o[valid] + d_back[:, None] * rays_d[valid]
            z_v = z[bidx][valid]
            bidx_v = bidx[valid]
            with torch.no_grad():
                if feat_maps is not None:
                    lf_front = model._project_features(
                        p_front, bidx_v,
                        input_c2ws.float(), input_Ks.float(),
                        feat_maps.float(), (H, W),
                    )
                    lf_back = model._project_features(
                        p_back, bidx_v,
                        input_c2ws.float(), input_Ks.float(),
                        feat_maps.float(), (H, W),
                    )
                else:
                    lf_front = None
                    lf_back = None
                sdf_front = model.sdf_mlp(p_front, z_v, lf_front).squeeze(-1)
                sdf_back = model.sdf_mlp(p_back, z_v, lf_back).squeeze(-1)
        else:
            sdf_front = torch.empty(0, device=device)
            sdf_back = torch.empty(0, device=device)

        metrics = {
            "eikonal":     m_eikonal(model, z[0], cfg, device),
            "sharpness":   m_sharpness(w.cpu(), t_v.cpu()),
            "sdf_isotropy": m_sdf_isotropy(model, z[0], cfg, device),
            "depth_hit":   m_depth_hit(w_sum, gt_dep, near, far, cfg.depth_hit_w_sum_thresh),
            "sdf_at_surf": m_sdf_at_surface(sdf_v.cpu(), t_v.cpu(), gt_dep.cpu(), near, far),
            "sdf_sign":    m_sdf_sign(sdf_v.cpu(), t_v.cpu(), gt_dep.cpu(), near, far),
            "sdf_band":    m_sdf_band_sign(sdf_front, sdf_back),
            "depth":       m_depth(pred_d, gt_dep, near, far),
            "normals":     m_normals(pred_n, gt_nor),
            "light":       m_light(model, z[0], pred_n, pred_d,
                                   rays_o, rays_d, gt_col, valid),
            "mask":        m_mask(model, batch, cfg, device),
        }
        print_results(obj_id, metrics)
        all_metrics.append(metrics)

    zdiv = m_z_diversity(z_list)
    log.info(row("z_diversity", zdiv,
                 zdiv.get("note") or f"mean pairwise L2={zdiv['mean_l2']:.3f}"))

    action_plan(all_metrics, cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--data_root",  type=str)
    parser.add_argument("--image_size", type=int)
    parser.add_argument("--n_objects",  type=int, default=1)
    parser.add_argument(
        "--overfit",
        action="store_true",
        help="Restrict diagnostics to one object (default id matches train.py --overfit).",
    )
    parser.add_argument(
        "--overfit_object",
        type=str,
        default=None,
        metavar="ID",
        help="With --overfit, which object_id to use (default: first on disk).",
    )
    args = parser.parse_args()
    cfg = PRISMConfig()
    for k in ("checkpoint", "data_root", "image_size"):
        if getattr(args, k) is not None:
            setattr(cfg, k, getattr(args, k))

    restrict = None
    if args.overfit:
        pairs = enumerate_object_pairs(cfg.data_root)
        if not pairs:
            raise RuntimeError(f"--overfit: no objects under data_root={cfg.data_root!r}")
        if args.overfit_object is not None:
            oid = args.overfit_object
            if not any(o[1] == oid for o in pairs):
                raise RuntimeError(
                    f"--overfit_object {oid!r} not found under {cfg.data_root!r} "
                    f"(have {len(pairs)} ids, e.g. {pairs[0][1]!r})"
                )
            oid0 = oid
        else:
            oid0 = pairs[0][1]
        restrict = [oid0]
        log.info("--overfit: diagnostics on %s", oid0)

    run_debug(cfg, n_objects=args.n_objects, restrict_object_ids=restrict)
