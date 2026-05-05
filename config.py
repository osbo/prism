from dataclasses import dataclass, field


@dataclass
class PRISMConfig:
    # Data
    data_root:   str = "/home/osbo/orcd/pool/omniobject3d/extracted"
    image_size:  int = 256          # resize for training speed (800 for full quality)
    num_workers: int = 4

    # Model
    latent_dim:        int   = 128
    pretrained_encoder: bool = True
    sdf_hidden:        int   = 256
    sdf_layers:        int   = 8
    n_freqs:           int   = 6    # Fourier positional encoding frequencies

    # Rendering
    n_rays:    int   = 256          # rays sampled per image per training step
    n_samples: int   = 64           # samples along each ray
    near:      float = 0.5
    far:       float = 6.0

    # Loss weights
    lambda_render: float = 1.0
    lambda_depth:  float = 1.0
    lambda_normal: float = 0.5
    lambda_eik:    float = 0.1
    # Direct SDF supervision from GT depth (prevents all-negative saturation collapse)
    lambda_sdf_surface: float = 0.20
    lambda_sdf_sign:    float = 0.02
    # Opacity supervision from GT depth validity:
    # object pixels should accumulate mass, background should remain empty.
    lambda_opacity:     float = 0.25
    # Keep photometric supervision on background to suppress "hallucinated" surfaces.
    lambda_bg_render:   float = 0.20
    lambda_bg_sdf:      float = 0.20
    # Prevent NeuS sharpness from collapsing to overly brittle values.
    beta_min:           float = 0.50

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

    # Evaluation
    mc_resolution: int   = 128      # marching cubes grid resolution
    mc_threshold:  float = 0.0
    mc_bound:      float = 1.5      # [-bound, bound]^3 for marching cubes
    fscore_tau:    float = 0.01
    n_eval_pts:    int   = 100_000
