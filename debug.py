"""
debug.py  —  PRISM diagnostics.  Ends with a concrete action plan.

    python debug.py [--checkpoint model.pt] [--n_objects N]
"""
import argparse, logging, numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from config import PRISMConfig
from prism import PRISM
from prism.brdf import cook_torrance_ggx
from prism.renderer import sample_rays, neus_weights
from data.omniobject3d import OmniObject3DDataset

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger()

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
_STATUS_PAD = 4   # width of status column

# ── measurements (each returns a flat dict with a 'status' key) ───────────────

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
        lp, li    = model.light_head(z.unsqueeze(0))
    ab = ab.expand(N,-1); ro = ro.expand(N,-1); me = me.expand(N,-1)
    lp_v = lp.expand(N,-1); li_v = li.expand(N,-1)
    x    = rays_o + pred_d[:,None] * rays_d
    v    = F.normalize(-rays_d, dim=-1)
    l    = F.normalize(lp_v - x, dim=-1)
    ndl  = (pred_n * l).sum(-1)
    # oracle: light at camera
    l_ora   = F.normalize(rays_o[0:1].expand(N,-1) - x, dim=-1)
    ndl_ora = (pred_n * l_ora).sum(-1)
    with torch.no_grad():
        col_pred = cook_torrance_ggx(pred_n, v, l,     ab, ro, me, li_v)
        col_ora  = cook_torrance_ggx(pred_n, v, l_ora, ab, ro, me, li_v)
    frac_pos = (ndl > 0).float().mean().item()
    status = PASS if frac_pos > 0.75 else FAIL if frac_pos < 0.50 else WARN
    result = {"status": status, "frac_pos": frac_pos, "mean_ndl": ndl.mean().item(),
              "light_pos": lp[0].cpu().numpy().round(3)}
    if valid_d.any():
        result["l1_pred"] = F.l1_loss(col_pred[valid_d], gt_col[valid_d]).item()
        result["l1_ora"]  = F.l1_loss(col_ora[valid_d],  gt_col[valid_d]).item()
    return result


def m_mask(model, batch, cfg, device):
    try:
        from scipy.ndimage import label as nd_label
    except ImportError:
        return {"status": WARN, "note": "scipy missing"}
    image = batch["image"].to(device); c2w = batch["c2w"].to(device)
    K = batch["K"].to(device);         gt_d = batch["depth"]
    B, C, H, W = image.shape
    scale = 4; Hs, Ws = H // scale, W // scale
    img_s = F.interpolate(image, (Hs, Ws), mode="bilinear", align_corners=False)
    Ks = K.clone()
    for i, j in [(0,0),(1,1),(0,2),(1,2)]: Ks[:,i,j] /= scale
    with torch.no_grad():
        out = model.render_image(img_s, c2w, Ks)
    op = out["opacity"].cpu().numpy()
    gd = gt_d[0,0].numpy()[::scale, ::scale][:Hs, :Ws]
    gt_fg = (gd > cfg.near) & (gd < cfg.far)
    # flood-fill background from edges
    below = op < 0.05
    labeled, _ = nd_label(below)
    edge_lbls = set()
    for row in (labeled[0], labeled[-1]): edge_lbls.update(row.tolist())
    for col in (labeled[:,0], labeled[:,-1]): edge_lbls.update(col.tolist())
    edge_lbls.discard(0)
    bg = np.zeros_like(below)
    for lbl in edge_lbls: bg |= labeled == lbl
    pred_fg = ~bg
    tp = int((pred_fg & gt_fg).sum()); fp = int((pred_fg & ~gt_fg).sum())
    fn = int((~pred_fg & gt_fg).sum())
    iou = tp / (tp + fp + fn + 1e-8)
    status = PASS if iou > 0.60 else FAIL if iou < 0.35 else WARN
    return {"status": status, "iou": iou,
            "prec": tp/(tp+fp+1e-8), "rec": tp/(tp+fn+1e-8)}


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
    return f"  {fmt(m['status'])} {name:<12} {detail}"


def print_results(obj_id, metrics):
    log.info(f"\n{obj_id}")
    mk = metrics
    log.info(row("eikonal",   mk["eikonal"],
                 f"‖∇f‖={mk['eikonal']['mean']:.2f}  tight={mk['eikonal']['tight']:.0%}"))
    log.info(row("sharpness", mk["sharpness"],
                 f"w_peak={mk['sharpness']['w_peak']:.3f}"))
    s = mk["sdf_sign"]
    if "note" in s:
        log.info(row("sdf_sign", s, s["note"]))
    else:
        inv = "  *** INVERTED" if s.get("inverted") else ""
        log.info(row("sdf_sign", s,
                     f"correct={s['correct_frac']:.0%}  front={s['front']:+.2f}  back={s['back']:+.2f}{inv}"))
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
                     f"IoU={msk['iou']:.2f}  prec={msk['prec']:.2f}  rec={msk['rec']:.2f}"))


def action_plan(all_metrics, cfg):
    """Aggregate across objects, identify root causes, emit specific fixes."""
    def avg_status(key, subkey):
        vals = [m[key].get(subkey, float("nan")) for m in all_metrics]
        return float(np.nanmean([v for v in vals if not np.isnan(v)]))

    eik_mean  = avg_status("eikonal",   "mean")
    eik_tight = avg_status("eikonal",   "tight")
    w_peak    = avg_status("sharpness", "w_peak")
    ndl_frac  = avg_status("light",     "frac_pos")
    depth_l1  = avg_status("depth",     "l1")
    depth_c   = avg_status("depth",     "corr")
    cos_mean  = avg_status("normals",   "cos")
    sdf_cf    = avg_status("sdf_sign",  "correct_frac")
    sdf_inv   = any(m["sdf_sign"].get("inverted", False) for m in all_metrics)
    mask_iou  = avg_status("mask",      "iou")

    eik_bad   = eik_mean > 1.5 or eik_tight < 0.35
    light_bad = ndl_frac < 0.60
    sign_bad  = sdf_cf < 0.45 and not eik_bad   # sign issues independent of eikonal

    causes, actions = [], []

    if eik_bad:
        causes.append(
            f"eikonal violated (‖∇f‖={eik_mean:.2f}, {eik_tight:.0%} tight) "
            f"→ noisy normals, diffuse surface"
        )
        actions.append(f"  config.py: lambda_eik   {cfg.lambda_eik:.2f} → {min(1.0, cfg.lambda_eik * 5):.2f}")
        actions.append(f"  config.py: n_freqs      {cfg.n_freqs}    → {max(1, cfg.n_freqs - 2)}")

    if light_bad:
        causes.append(
            f"light behind surface (n·l frac>0={ndl_frac:.0%}) "
            f"→ BRDF black → zero render gradient"
        )
        actions.append(
            f"  config.py: lambda_light_facing "
            f"{cfg.lambda_light_facing:.2f} → {min(1.0, cfg.lambda_light_facing * 3):.2f}"
        )

    if sdf_inv:
        causes.append("SDF sign is INVERTED (negative outside, positive inside)")
        actions.append("  prism/sdf_mlp.py: negate output of self.out(h)")

    if sign_bad and not sdf_inv:
        causes.append(f"SDF sign ordering incorrect ({sdf_cf:.0%} correct)")
        actions.append(f"  config.py: lambda_sdf_sign    {cfg.lambda_sdf_sign:.2f} → {min(0.5, cfg.lambda_sdf_sign * 5):.2f}")
        actions.append(f"  config.py: lambda_sdf_surface {cfg.lambda_sdf_surface:.2f} → {min(1.0, cfg.lambda_sdf_surface * 3):.2f}")

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

def run_debug(cfg, n_objects=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PRISM(cfg).to(device)
    ckpt  = torch.load(cfg.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    log.info(f"{cfg.checkpoint}  epoch={ckpt.get('epoch','?')}  "
             f"step={ckpt.get('step','?')}  β={model.beta.item():.3f}")

    ds     = OmniObject3DDataset(cfg.data_root, split="test", image_size=cfg.image_size)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    z_list, all_metrics = [], []

    for i, batch in enumerate(loader):
        if i >= n_objects: break
        obj_id = batch["object_id"][0]
        image  = batch["image"].to(device)
        c2w    = batch["c2w"].to(device)
        K_mat  = batch["K"].to(device)
        gt_d   = batch["depth"].to(device)
        gt_n   = batch["normal"].to(device)
        B, _, H, W = image.shape

        with torch.no_grad():
            z = model.encoder(image)
        z_list.append(z.cpu())

        near, far = cfg.near, cfg.far
        with torch.no_grad():
            rays_o, rays_d, pix_rc, bidx = sample_rays(c2w, K_mat, H, W, 512, device)
        N = rays_o.shape[0]
        t_e = torch.linspace(near, far, cfg.n_samples + 1, device=device)
        t_v = (t_e[:-1].unsqueeze(0).expand(N,-1)
               + torch.rand(N, cfg.n_samples, device=device)
               * (t_e[1:] - t_e[:-1]).unsqueeze(0))
        pts = rays_o[:,None] + t_v[:,:,None] * rays_d[:,None]
        z_pts = z.expand(N,-1)[:,None].expand(-1,cfg.n_samples,-1).reshape(-1, z.shape[-1])

        pf = pts.reshape(-1,3).requires_grad_(True)
        with torch.enable_grad():
            sf = model.sdf_mlp(pf, z_pts.detach()).squeeze(-1)
            sg = torch.autograd.grad(sf, pf, torch.ones_like(sf))[0]
        sdf_v = sf.detach().reshape(N, cfg.n_samples)
        norms = sg.detach().reshape(N, cfg.n_samples, 3)

        with torch.no_grad():
            w      = neus_weights(sdf_v, model.beta)
            pred_d = (w * t_v).sum(-1)
            pred_n = F.normalize((w[:,:,None]*norms).sum(1), dim=-1)
            vdot   = (pred_n * (-rays_d)).sum(-1, keepdim=True)
            pred_n = torch.where(vdot < 0, -pred_n, pred_n)
            r, c   = pix_rc[:,0], pix_rc[:,1]
            gt_col = image.permute(0,2,3,1)[bidx, r, c]
            gt_dep = gt_d[:,0][bidx, r, c]
            gt_nor = gt_n.permute(0,2,3,1)[bidx, r, c]
            valid  = (gt_dep > near) & (gt_dep < far)

        metrics = {
            "eikonal":  m_eikonal(model, z[0], cfg, device),
            "sharpness": m_sharpness(w.cpu(), t_v.cpu()),
            "sdf_sign": m_sdf_sign(sdf_v.cpu(), t_v.cpu(), gt_dep.cpu(), near, far),
            "depth":    m_depth(pred_d, gt_dep, near, far),
            "normals":  m_normals(pred_n, gt_nor),
            "light":    m_light(model, z[0], pred_n, pred_d,
                                rays_o, rays_d, gt_col, valid),
            "mask":     m_mask(model, batch, cfg, device),
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
    parser.add_argument("--n_objects",  type=int, default=3)
    args = parser.parse_args()
    cfg = PRISMConfig()
    for k in ("checkpoint", "data_root", "image_size"):
        if getattr(args, k) is not None:
            setattr(cfg, k, getattr(args, k))
    run_debug(cfg, n_objects=args.n_objects)
