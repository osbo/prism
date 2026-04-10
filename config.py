from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class EncoderConfig:
    latent_dim: int = 256
    backbone: str = "resnet34"          # torchvision pretrained
    pretrained: bool = True
    freeze_bn: bool = False


@dataclass
class SDFConfig:
    latent_dim: int = 256
    hidden_dim: int = 256
    n_layers: int = 8
    n_fourier_freqs: int = 6            # positional encoding frequencies for x
    use_tcnn_encoding: bool = True      # use tiny-cuda-nn for Fourier encoding
    geometric_init: bool = True         # initialise SDF as a sphere
    sphere_radius: float = 0.5          # initial sphere radius for geometric init
    weight_norm: bool = True


@dataclass
class BRDFConfig:
    latent_dim: int = 256
    hidden_dim: int = 128
    n_layers: int = 3


@dataclass
class LightConfig:
    latent_dim: int = 256
    hidden_dim: int = 128
    n_layers: int = 2
    # Scale applied to predicted position — keeps output in scene unit cube
    position_scale: float = 3.0


@dataclass
class RendererConfig:
    # Sphere tracing
    near: float = 0.5
    far: float = 4.0
    n_sphere_trace_steps: int = 64
    sphere_trace_eps: float = 5e-4      # convergence threshold
    # NeuS silhouette
    neus_s_init: float = 64.0           # initial value of logistic scale
    neus_s_max: float = 512.0
    # Ray sampling
    n_rays_train: int = 512             # rays per image during training
    n_rays_eval: int = 4096
    image_size: int = 800
    # Background colour (black for object-centric renders)
    bg_colour: Tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class LossConfig:
    lambda_render: float = 1.0
    lambda_depth: float = 0.1
    lambda_normal: float = 0.05
    lambda_eikonal: float = 0.1
    use_perceptual: bool = True         # add LPIPS on top of L1 render loss
    perceptual_weight: float = 0.1


@dataclass
class TrainConfig:
    data_root: str = "/orcd/pool/007/osbo/omniobject3d"
    output_dir: str = "./runs"
    n_epochs: int = 100
    batch_size: int = 4                 # objects per GPU
    lr: float = 5e-4
    lr_encoder: float = 1e-4           # encoder trained slower
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    warmup_steps: int = 1000
    lr_schedule: str = "cosine"
    num_workers: int = 4
    save_every_n_epochs: int = 5
    log_every_n_steps: int = 50
    eval_every_n_epochs: int = 10
    mixed_precision: bool = True        # fp16 training
    compile: bool = False               # torch.compile (experimental)
    # DDP
    port: str = "29500"


@dataclass
class EvalConfig:
    marching_cubes_resolution: int = 256
    marching_cubes_threshold: float = 0.0
    fscore_tau: float = 0.01            # 1% of bounding box diagonal
    n_point_samples: int = 100_000     # points sampled from predicted/GT mesh


@dataclass
class PRISMConfig:
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    sdf: SDFConfig = field(default_factory=SDFConfig)
    brdf: BRDFConfig = field(default_factory=BRDFConfig)
    light: LightConfig = field(default_factory=LightConfig)
    renderer: RendererConfig = field(default_factory=RendererConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    seed: int = 42
    project_name: str = "prism-6s058"
