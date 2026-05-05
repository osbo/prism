from dataclasses import dataclass, field


@dataclass
class PRISMConfig:
    # Data
    data_root:      str = "/home/osbo/orcd/pool/omniobject3d/extracted"
    image_size:     int = 256          # resize for training speed (800 for full quality)
    num_workers:    int = 4
    n_input_views:  int = 5            # context views fed to encoder; 1 = classic single-view

    # Model
    latent_dim:        int   = 128
    pretrained_encoder: bool = True
    sdf_hidden:        int   = 256
    sdf_layers:        int   = 8
    n_freqs:           int   = 8    # Fourier positional encoding frequencies (high values → checkerboard)
    # Sphere initialization (not a structural restriction).
    sdf_init_radius:   float = 0.35

    # Rendering
    n_rays:    int   = 256          # rays sampled per image per training step
    n_samples: int   = 64           # samples along each ray
    near:      float = 0.5
    far:       float = 6.0
    # Depth / inference: no surface mass along the ray → depth clamped to far
    # (avoids picking a spurious bin when weights are noise).
    depth_hit_w_sum_thresh: float = 0.05
    # Hard clamp raw SDF logits before NeuS sigmoid (fp32 path; improves AMP stability).
    sdf_clamp: float = 80.0

    # Loss weights
    lambda_render: float = 1.0
    lambda_depth:  float = 1.0
    lambda_normal: float = 0.5
    lambda_eik:    float = 0.5
    # Direct SDF supervision from GT depth (prevents all-negative saturation collapse)
    lambda_sdf_surface: float = 0.50
    lambda_sdf_sign:    float = 0.10
    # Closure priors: keep origin inside and boundary outside.
    lambda_closure:     float = 0.01
    closure_center_margin: float = 0.05
    closure_boundary_margin: float = 0.05
    # Background constraints from mask (outside object should stay empty).
    lambda_bg_render:   float = 0.0
    lambda_bg_sdf:      float = 0.20
    bg_sdf_margin:      float = 0.02
    # Silhouette matching (sampled rays): drives object contour away from spherical blob.
    lambda_sil_bce:     float = 2.0
    lambda_sil_dice:    float = 0.5
    # Prevent NeuS sharpness from collapsing to overly brittle values.
    beta_min:           float = 0.30

    # Training
    n_epochs:       int   = 100
    batch_size:     int   = 4
    lr:             float = 5e-4
    lr_encoder:     float = 1e-4
    weight_decay:   float = 1e-4
    grad_clip:      float = 1.0
    warmup_steps:   int   = 500
    checkpoint:     str   = "model.pt"
    log_every:      int   = 50
    save_every:     int   = 5       # epochs — periodic checkpoint
    eval_every:     int   = 5       # epochs — val + best-on-val (same cadence)

    # Evaluation / mesh export
    # Higher resolution reduces axis-aligned marching-cubes stairsteps on curved surfaces.
    mc_resolution: int   = 192      # was 128; ↑ for smoother meshes (slower ~ (res/128)³)
    mc_threshold:  float = 0.0
    mc_bound:      float = 2.5      # [-bound, bound]^3 for marching cubes
    # If True: push SDF outside where grid points project to GT background (view-conditioned).
    # Set False to export raw marching cubes without mask carve.
    mc_carve_background: bool = True
    mc_carve_sdf_min: float = 0.35   # carved voxels: max(sdf, this) so iso=0 surface cannot pass there
    mc_keep_largest_component: bool = True
    # Blur GT mask before carving + soft blend (reduces harsh voxel-aligned “steps” at carve boundary).
    mc_carve_mask_blur_radius: int = 2   # 0 = hard silhouette (more stairsteps); 2 ≈ 5×5 Gaussian

    # Light-facing penalty: push n·l > 0 so render loss provides non-zero gradients.
    lambda_light_facing: float = 0.2
    fscore_tau:    float = 0.01
    n_eval_pts:    int   = 100_000
