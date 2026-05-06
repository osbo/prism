from dataclasses import dataclass


@dataclass
class PRISMConfig:
    # Data
    data_root:      str = "/home/osbo/orcd/pool/omniobject3d/extracted"
    image_size:     int = 256          # resize for training speed (800 for full quality)
    num_workers:    int = 8
    n_input_views:  int = 5            # context views fed to encoder; 1 = classic single-view

    # Model
    latent_dim:        int   = 128
    feat_dim:          int   = 16     # per-point local feature dim projected from input views (0 = global-only)
    pretrained_encoder: bool = True
    sdf_hidden:        int   = 256
    sdf_layers:        int   = 8
    n_freqs:           int   = 5    # Fourier positional encoding frequencies (high values → checkerboard)
    # Sphere initialization (not a structural restriction).
    sdf_init_radius:   float = 0.35

    # Rendering
    n_rays:    int   = 512          # rays sampled per image per training step
    n_samples: int   = 96           # samples along each ray
    n_importance: int = 24          # hierarchical fine samples per ray (NeUS importance resampling)
    near:      float = 0.5
    far:       float = 6.0
    # Depth / inference: no surface mass along the ray → depth clamped to far
    # (avoids picking a spurious bin when weights are noise).
    depth_hit_w_sum_thresh: float = 0.15
    # Hard clamp raw SDF logits before NeuS sigmoid (fp32 path; improves AMP stability).
    sdf_clamp: float = 5.0

    # Loss weights: each lambda scales its matching l_* term in prism/model.py total.
    # Tuned for depth + normals as primary objectives; RGB (render/perc) secondary.
    # Log column names from train.py when they differ from lambda_* names:
    #   render depth normal eik sdf0 sdf_sign sdf_band close bg_sdf bg_alpha
    #   sil_bce sil_dice hull perc lface
    #
    #  lambda_render           render       L1 shaded RGB (Cook-Torrance) vs GT on FG rays (GT mask).
    #  lambda_depth            depth        L1 rendered depth vs GT depth where GT depth valid.
    #  lambda_normal           normal       1 - cosine similarity vs GT normals where GT valid.
    #  lambda_eik              eik          Mean squared (||grad SDF|| - 1) with capped grad norm.
    #  lambda_sdf_surface      sdf0         |SDF| at ray sample closest to GT depth (surface hits data).
    #  lambda_sdf_sign         sdf_sign     In front of GT depth SDF should be +; behind should be -.
    #  lambda_sdf_band         sdf_band     Extra samples offset along ray: outside + / inside - bands.
    #  lambda_closure          close        Weak prior: origin inside object; random dirs at mc_bound outside.
    #  lambda_bg_render        (unused)     Kept at 0: no RGB loss on background pixels.
    #  lambda_bg_sdf           bg_sdf       BG rays: softplus keeps SDF samples >= margin (empty space).
    #  lambda_bg_alpha         bg_alpha     BG rays: NeuS weight sum -> 0 (no density in background).
    #  lambda_sil_bce          sil_bce      BCE on soft-min SDF silhouette vs FG/BKG from mask.
    #  lambda_sil_dice         sil_dice     1 - Dice between predicted silhouette and GT mask.
    #  lambda_visual_hull      hull         Points outside any input-view mask -> positive SDF (carving).
    #  lambda_perceptual       perc         VGG feature L1 on random rendered patch vs GT patch.
    #  lambda_light_facing     lface        Penalize normals with n dot l <= 0 for BRDF gradient flow.
    lambda_render: float = 0.35
    lambda_depth:  float = 4.0
    lambda_normal: float = 2.5
    lambda_eik:    float = 1.0
    lambda_sdf_surface: float = 2.0
    lambda_sdf_sign:    float = 1.0
    lambda_sdf_band:    float = 1.5
    sdf_band_delta:     float = 0.03  # meters along ray for sdf_band probes
    lambda_closure:     float = 0.01
    closure_center_margin: float = 0.05
    closure_boundary_margin: float = 0.05
    lambda_bg_render:   float = 0.0
    lambda_bg_sdf:      float = 2.0
    lambda_bg_alpha:    float = 2.0
    bg_sdf_margin:      float = 0.02
    lambda_sil_bce:     float = 3.0
    lambda_sil_dice:    float = 0.5
    lambda_visual_hull:  float = 1.5
    visual_hull_margin:  float = 0.02  # same scale as bg_sdf_margin
    lambda_perceptual:    float = 0.5
    perceptual_patch_size: int = 32    # patch_size^2 extra rays per step
    lambda_light_facing: float = 0.5
    # Temperature for SDF soft-min silhouette logits (smaller = crisper but less stable).
    sil_sdf_tau:        float = 0.02
    # NeuS β: learned log_β is clamped to [beta_min, beta_max]. beta_max falls linearly from
    # beta_anneal_start (training start) to beta_anneal_end over the first ``beta_anneal_epochs``
    # worth of optimizer steps (step / (beta_anneal_epochs * len(train_loader))), then stays at end.
    # This horizon is independent of ``n_epochs`` so changing run length or --resume does not
    # re-stretch the anneal by changing the denominator.
    beta_min:            float = 1e-6
    beta_anneal_epochs: int   = 30
    beta_anneal_start:   float = 0.25   # max β at start of anneal (softer surface)
    beta_anneal_end:    float = 1e-6    # max β after anneal (sharper surface)

    # Training
    n_epochs:       int   = 100
    batch_size:     int   = 4
    lr:             float = 5e-4
    lr_encoder:     float = 1e-4
    weight_decay:   float = 1e-4
    grad_clip:      float = 1.0
    warmup_steps:   int   = 500
    # --overfit: many random (view,target) draws per epoch; short LR warmup; no WD for memorization.
    overfit_samples_per_epoch: int = 512
    overfit_warmup_steps:     int = 100
    overfit_weight_decay:    float = 0.0
    overfit_min_epochs:       int = 120   # floor when --overfit (full cfg.n_epochs used if larger)
    checkpoint:     str   = "model.pt"
    log_every:      int   = 50
    save_every:     int   = 5       # epochs — periodic checkpoint
    eval_every:     int   = 5       # epochs — val + best-on-val (same cadence)

    # Evaluation / mesh export
    # Higher resolution reduces axis-aligned marching-cubes stairsteps on curved surfaces.
    mc_resolution: int   = 192      # was 128; ↑ for smoother meshes (slower ~ (res/128)³)
    mc_threshold:  float = 0.0
    mc_bound:      float = 2.5      # [-bound, bound]^3 for marching cubes
    mc_laplacian_iters: int = 5     # Laplacian smoothing passes post-MC (0 = off)
    # If True: push SDF outside where grid points project to GT background (view-conditioned).
    # Set False to export raw marching cubes without mask carve.
    mc_carve_background: bool = False
    mc_carve_sdf_min: float = 0.35   # carved voxels: max(sdf, this) so iso=0 surface cannot pass there
    mc_keep_largest_component: bool = True
    # Blur GT mask before carving + soft blend (reduces harsh voxel-aligned “steps” at carve boundary).
    mc_carve_mask_blur_radius: int = 2   # 0 = hard silhouette (more stairsteps); 2 ≈ 5×5 Gaussian

    # Evaluation metrics (not training losses)
    fscore_tau:    float = 0.01
    n_eval_pts:    int   = 100_000
