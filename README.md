# PRISM

Physics-Informed Reconstruction via Implicit Surfaces and Materials.

PRISM is a feed-forward multi-view inverse-rendering system that jointly estimates:

- a neural signed distance function (SDF) for geometry,
- Cook-Torrance GGX BRDF parameters for material,
- and a point-light + ambient lighting model.

Unlike radiance-field-only pipelines, PRISM keeps geometry and appearance physically interpretable by coupling an implicit surface with differentiable NeuS-style rendering.

## Highlights

- Joint geometry + material + lighting reconstruction from sparse RGB views.
- PixelNeRF-style local feature projection for multi-view conditioning.
- Physically based shading (Cook-Torrance GGX) in the training loop.
- Strong geometric supervision (depth, normals, silhouette, SDF sign/band, visual hull, eikonal).
- Evaluated on OmniObject3D with both geometric and photometric metrics.

## Main Results (from report)

On 19 held-out OmniObject3D objects (6 categories), the full model reports:

- **Chamfer distance**: `0.209 ± 0.043` (lower is better)
- **Foreground PSNR**: `14.55 ± 2.52 dB` (higher is better)

Loss-group ablations showed the largest PSNR impact from removing photometric supervision (`-4.48 dB`), followed by depth supervision (`-1.14 dB`), with smaller effects from normal (`-0.23 dB`) and eikonal (`-0.08 dB`) terms.

## Example Figures

Five input views used for `clock_029`:

![clock_029 input views](../Final%20Project%20Report/Screenshot%202026-05-07%20at%208.25.02%E2%80%AFPM.png)

Extracted mesh rendering (`clock_029`):

![clock_029 reconstructed mesh](../Final%20Project%20Report/Screenshot%202026-05-07%20at%208.35.09%E2%80%AFPM.png)

## Method Overview

1. **Image encoder** (`ResNet-34`): produces shallow feature maps and a global latent code.
2. **SDF network**: predicts signed distance for 3D query points, conditioned on latent + projected local features.
3. **Material and light heads**: decode BRDF (albedo, roughness, metalness) and light parameters.
4. **Differentiable rendering (NeuS)**: ray sampling + hierarchical resampling, then GGX shading at expected hit points.
5. **Multi-term objective**: photometric, geometric, regularization, silhouette/free-space, and optimization helper losses.

## Repository Layout

- `prism/model.py` - full PRISM model and forward pass
- `prism/renderer.py` - differentiable rendering utilities
- `prism/sdf_mlp.py` - SDF MLP
- `prism/brdf.py` - Cook-Torrance GGX shading
- `prism/encoder.py` - image encoder
- `prism/losses.py` - training losses
- `prism/mesh_extract.py` - marching cubes mesh extraction
- `train.py` - training entrypoint
- `evaluate.py` - quantitative evaluation (Chamfer / F-score / PSNR)
- `config.py` - hyperparameters and loss weights

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

PRISM expects OmniObject3D data with Blender renders and raw scans. Set the dataset root via `config.py` or CLI flags:

```bash
python train.py --data_root /path/to/omniobject3d/extracted
```

## Training

Fresh training run:

```bash
python train.py --data_root /path/to/omniobject3d/extracted
```

Resume from default checkpoint (`model.pt`):

```bash
python train.py --resume
```

Overfit one object (debug mode):

```bash
python train.py --overfit --overfit_object bottle_001
```

## Evaluation

Run quantitative evaluation:

```bash
python evaluate.py --checkpoint model.pt --out_dir eval_results/metrics
```

Results are written to `eval_results/metrics/metrics.json` with aggregate and per-object metrics.

## Current Limitations

- Spatially uniform BRDF per object (no spatially varying albedo/roughness maps yet).
- Point-light approximation cannot fully capture HDRI environment illumination.
- Thin-structure reconstruction is constrained by finite ray sampling budget.
- Performance is compute-limited; longer training / higher resolution is expected to improve detail.
