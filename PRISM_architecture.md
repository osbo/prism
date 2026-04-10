# PRISM
## Physics-Informed Reconstruction via Implicit Surfaces and Materials

**Course:** 6.S058 — Introduction to Computer Vision  
**Author:** Carl Osborne  
**Dataset:** OmniObject3D (`blender_renders_24_views`, `raw_scans`)

---

## Overview

PRISM is a feed-forward, single-image 3D reconstruction network that infers object geometry, surface materials, and scene lighting jointly from a single RGB image. Rather than treating reconstruction as a purely geometric regression problem, PRISM bakes physical rendering constraints directly into the training objective: the model must produce a neural SDF, BRDF parameters, and a light position that together — when evaluated against the rendering equation — reproduce the input image. Geometry, appearance, and illumination are implicitly co-supervised through differentiable rendering, with no direct supervision on BRDF parameters or light positions.

---

## Dataset

| Split | Source | Contents |
|---|---|---|
| Training inputs | `blender_renders_24_views` | RGB images (800×800), depth maps, normal maps, camera extrinsics |
| Evaluation GT | `raw_scans` | Textured OBJ meshes from real high-fidelity scans |

OmniObject3D provides ~6,000 real-world objects across 190 categories, rendered under known Blender lighting conditions. Ground-truth depth and normal maps are available per frame, enabling direct supervision of intermediate predictions. Crucially, because the renders derive from real scans rather than artist-modeled geometry, the dataset avoids the synthetic domain gap present in ShapeNetCore.

---

## Architecture

### Stage 1 — Image Encoder

A **ResNet-34** backbone (pretrained on ImageNet) encodes the 800×800 input RGB image into a global latent vector **z**. This vector conditions all downstream components. The encoder is fine-tuned end-to-end during training.

### Stage 2 — Neural SDF MLP (tiny-cuda-nn)

A coordinate MLP implemented in **tiny-cuda-nn** maps a 3D query point **x**, conditioned on **z**, to a scalar signed distance value f(**x**):

```
f : (x ∈ ℝ³, z) → s ∈ ℝ
```

- **Positional encoding** on **x** (NeRF-style Fourier features) to recover fine geometric detail
- **FiLM conditioning** (feature-wise linear modulation) of **z** into MLP layers, preferred over naive concatenation for shape conditioning
- tiny-cuda-nn is used for MLP speed — sphere tracing requires hundreds of SDF queries per ray during both forward and backward passes

### Stage 3 — Prediction Heads

Two small MLP heads, both conditioned on **z**:

**BRDF Head**
Outputs per-object Cook-Torrance GGX material parameters:
- Albedo (RGB)
- Roughness scalar
- Metalness scalar

Cook-Torrance GGX is used instead of Blinn-Phong to correctly handle the metallic, specular, and glossy objects prevalent in OmniObject3D. These are global per-object predictions (spatially uniform BRDF); spatially-varying BRDF is left as future work.

**Light Head**
Outputs a single point light source:
- Position (3D)
- Intensity (RGB)

Consistent with the known single-light Blender rendering setup in OmniObject3D. BRDF parameters and light position receive **no direct supervision** — they are constrained entirely through the render loss.

### Stage 4 — Differentiable Rendering

For each sampled camera ray:

1. **Sphere tracing** — march along the ray, querying the SDF MLP, until the surface is found: |f(**x**)| < ε. Surface point **x\*** is recorded.

2. **Surface normal** — computed analytically as the gradient of the SDF:
   ```
   n = ∇f(x*) / ||∇f(x*)||
   ```
   This is exact, differentiable, and free — no finite differencing required.

3. **BRDF evaluation** — evaluate Cook-Torrance GGX at **x\*** using:
   - Surface normal **n**
   - Predicted albedo, roughness, metalness
   - Predicted light position and intensity
   - Camera view direction

4. **NeuS silhouette formulation** — at object boundaries, the hard surface intersection is replaced with a logistic density approximation, ensuring smooth gradient flow through silhouettes during early training when the SDF is imprecise.

5. **Backward pass through sphere tracer** — gradients flow from the render loss back through **x\*** into the SDF weights via the implicit function theorem:
   ```
   ∂x* / ∂θ = −∇f(x*) / ||∇f(x*)||²  ·  ∂f/∂θ
   ```
   This is the key equation enabling full end-to-end gradient flow from rendered pixel → BRDF params → surface point → SDF weights → encoder.

---

## Loss Function

```
L_total = λ₁ · L_render    +   λ₂ · L_depth    +   λ₃ · L_normal    +   λ₄ · L_eikonal
```

| Term | Description | Supervision Source |
|---|---|---|
| `L_render` | L1/perceptual loss between rendered and GT image | `blender_renders_24_views` RGB |
| `L_depth` | L1 loss between sphere-traced depth and GT depth | `blender_renders_24_views` depth maps |
| `L_normal` | Cosine loss between ∇f(x*) and GT normals | `blender_renders_24_views` normal maps |
| `L_eikonal` | Enforces \|\|∇f(x)\|\| = 1 (valid SDF regularization) | No GT needed — analytic |

**Notes:**
- `L_render` is the sole supervision signal for BRDF parameters and light position
- `L_eikonal` is essential to prevent the SDF MLP from learning degenerate non-distance functions; typical weight λ₄ ≈ 0.1
- BRDF params and light position have no dedicated loss term — they are latent variables constrained implicitly by physical consistency with the render

---

## Evaluation

Evaluation is performed against `raw_scans` ground truth meshes. At evaluation time only, a mesh is extracted from the trained SDF via **Marching Cubes**. Marching Cubes is not part of the training loop and does not need to be differentiable.

| Metric | Description |
|---|---|
| Chamfer Distance | Mean bidirectional point-to-point distance between predicted and GT mesh |
| F-Score @ τ | Harmonic mean of precision and recall at distance threshold τ |
| PSNR | Peak signal-to-noise ratio on rendered novel views |

Baselines: dense voxel grid reconstruction (ShapeNet-style), and the ICCV 2023 OmniObject3D Challenge Track-1 winner (Pixel-NeRF + depth supervision).

---

## Implementation Stack

| Component | Tool |
|---|---|
| Image encoder | ResNet-34 (torchvision, pretrained) |
| SDF MLP | tiny-cuda-nn |
| Sphere tracer | Custom PyTorch/CUDA (~100 lines), implicit function theorem backward |
| Silhouette gradients | NeuS density formulation |
| BRDF evaluation | Custom PyTorch (Cook-Torrance GGX) |
| Mesh extraction (eval only) | Marching Cubes (PyMCubes or skimage) |
| Training cluster | MIT ORCD Engaging (`/orcd/pool/007/osbo/omniobject3d/`) |

---

## Key Novelty

Prior work on single-image reconstruction either (a) regresses geometry only with no appearance model, or (b) uses rendering losses without physical BRDF constraints. PRISM's contribution is that **BRDF parameters and light position are inferred jointly with geometry**, constrained only by having to satisfy the rendering equation. The model cannot produce a geometrically correct surface with physically wrong materials, or vice versa — the physics enforce joint consistency across all predicted quantities.

This is distinct from the ICCV 2023 OmniObject3D challenge winner, which added depth supervision as a loss term on top of Pixel-NeRF. PRISM instead builds the physics into the forward pass itself.

---

## Ablation Studies

| Ablation | What it tests |
|---|---|
| Remove `L_render` | Does physics-informed rendering actually help geometry? |
| Replace Cook-Torrance with Blinn-Phong | Impact of BRDF model expressiveness |
| Remove `L_eikonal` | Importance of SDF regularization |
| Remove light head (fixed light) | Does predicted lighting improve reconstruction? |
| Single λ weighting vs. tuned λ | Sensitivity of loss balance |

---

## References

- Wu et al., "OmniObject3D," CVPR 2023
- Wang et al., "NeuS," NeurIPS 2021
- Yariv et al., "IDR," NeurIPS 2020
- Mildenhall et al., "NeRF," ECCV 2020
- Du et al., "1st Place Solution, ICCV 2023 OmniObject3D Challenge," arXiv 2404.10441
- Yang et al., "Learning Effective NeRFs and SDFs," arXiv 2309.16110
- Müller et al., "tiny-cuda-nn," 2022
