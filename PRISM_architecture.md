# PRISM
## Physics-Informed Reconstruction via Implicit Surfaces and Materials

**Course:** 6.S058 — Introduction to Computer Vision  
**Author:** Carl Osborne  
**Dataset:** OmniObject3D (`blender_renders_24_views`, `raw_scans`)

---

## Overview

PRISM is a **feed-forward multi-view** 3D reconstruction model: one or more RGB input views are encoded into a global latent **z** (and optional per-pixel features), which conditions a neural SDF, material parameters, and lighting. A **single-view** configuration is the special case **N = 1**.

Reconstruction is not treated as pure geometric regression. The forward pass evaluates a **differentiable** image formation model (implicit surface + Cook–Torrance shading), so geometry, appearance, and illumination are **coupled** through the render. **BRDF parameters and light parameters still have no direct labels**; they are identified through image consistency, augmented by **direct depth and normal supervision** from the dataset where available.

---

## Dataset

| Split | Source | Contents |
|---|---|---|
| Training inputs | `blender_renders_24_views` | RGB images (800×800), depth maps, normal maps, camera extrinsics |
| Evaluation GT | `raw_scans` | Textured OBJ meshes from real high-fidelity scans |

OmniObject3D provides ~6,000 real-world objects across 190 categories, rendered under known Blender lighting conditions. Ground-truth depth and normal maps are available per frame, enabling **direct supervision** of rendered depth and normals as well as auxiliary SDF and silhouette terms. Because the renders derive from real scans rather than artist-modeled geometry, the dataset avoids the synthetic domain gap present in ShapeNetCore.

---

## Architecture

### Stage 1 — Image Encoder

A **ResNet-34** backbone (pretrained on ImageNet) processes each input view. The implementation supports **multiple views**: shallow features are kept at **H/8 × W/8** for optional **PixelNeRF-style** projection to 3D query points, while a deeper path produces a **global latent z** per batch element (views are pooled for conditioning BRDF and light heads). The encoder is fine-tuned end-to-end during training. Input resolution is configurable at train time (not required to be 800×800).

### Stage 2 — Neural SDF MLP (PyTorch)

A coordinate MLP in **plain PyTorch** maps a 3D query **x**, conditioned on **z** (and optionally on **locally projected image features**), to a scalar signed distance **f(x)**:

```
f : (x ∈ ℝ³, z [, local features]) → s ∈ ℝ
```

- **Positional encoding** on **x** (NeRF-style Fourier features) for geometric detail  
- **FiLM conditioning** of **z** on hidden layers (feature-wise linear modulation), rather than only concatenating **z** at the input  

**Design choice:** A PyTorch MLP is used instead of tiny-cuda-nn. It integrates cleanly with the rest of the stack (AMP, autograd through the SDF, second-order-free curvature-free training), is easier to modify, and is **sufficient** for this project’s ray sample counts when combined with **NeuS** (below) rather than per-pixel sphere marching at high iteration counts.

### Stage 3 — Prediction Heads

Two small MLP heads, both conditioned on **z**:

**BRDF head**  
Global Cook–Torrance GGX parameters:

- Albedo (RGB)  
- Roughness (scalar)  
- Metalness (scalar)  

Spatially uniform BRDF per object; spatially varying BRDF remains future work.

**Light head**  
A **point light** plus **isotropic ambient** RGB:

- Light position (3D)  
- Point intensity (RGB)  
- Ambient (RGB)  

This matches real Blender-style shading better than a bare point source alone: shadows and grazing regions still receive stable gradients. **No direct supervision** on BRDF or light parameters — they are tied to observations through the render loss (and indirectly through normals and depth that depend on surface and shading).

### Stage 4 — Differentiable rendering (NeuS)

For each sampled camera ray, the model **does not** use discrete sphere tracing with an implicit-function-theorem backward through a hit point. Instead it uses **NeuS**-style volume rendering (Wang et al., NeurIPS 2021):

1. **Stratified samples** along the ray (plus optional **hierarchical importance** resampling driven by coarse NeuS weights).  
2. **SDF values** and **gradients ∇f** at sample locations via autograd.  
3. **NeuS weights** from the SDF and a learned sharpness **β**, yielding **accumulated opacity**, **expected depth**, and **normals** as weighted combinations of samples (with a hit threshold so empty rays do not pick spurious depth).  
4. **Shading** — Cook–Torrance GGX at the effective surface using predicted BRDF, light position, intensity, and ambient, view direction, and the predicted normal.  

**Why NeuS over sphere tracing here:** NeuS gives **smooth, stable gradients through silhouettes and thin structures** while the SDF is still coarse, avoids maintaining a custom CUDA sphere tracer and IFT backward, and couples naturally with **multi-view feature projection** at arbitrary 3D points. Sphere tracing with IFT remains a valid alternative for other systems; this codebase standardizes on NeuS.

---

## Loss function

The training objective is a **weighted sum** of terms (see `config.py` for current λ). Conceptually:

| Group | Role |
|---|---|
| **Photometric** | `L_render` (L1 on shaded RGB, foreground rays), optional **`L_perceptual`** (VGG features on a random patch) — primary signal for BRDF and light. |
| **Geometry** | `L_depth`, `L_normal` vs ground-truth maps; **`L_sdf_surface`**, **`L_sdf_sign`**, **`L_sdf_band`** to anchor the implicit field to observed depth along rays. |
| **Regularity** | **`L_eikonal`** (‖∇f‖ ≈ 1); **`L_closure`** (weak cage: origin inside, boundary samples outside). |
| **Freespace / contour** | **`L_bg_sdf`**, **`L_bg_alpha`** on mask-background rays; **`L_sil_bce`**, **`L_sil_dice`** on silhouette; **`L_visual_hull`** carving using multi-view masks. |
| **Optimization helper** | **`L_light_facing`** (encourage normals toward the light so BRDF does not saturate to zero gradient). |

`L_render` alone does not carry all geometric information: **depth, normals, and SDF auxiliary terms** improve shape and reduce ambiguity, while the render still **ties materials and lighting** to pixels. Weights are tuned for the project (e.g. stronger depth/normal when geometry is the priority).

---

## Evaluation

Evaluation uses **`raw_scans`** meshes as geometric ground truth. At evaluation time only, a mesh is extracted from the trained SDF via **marching cubes** (`skimage.measure.marching_cubes`). Marching cubes is **not** in the training loop and need not be differentiable.

| Metric | Description |
|---|---|
| Chamfer distance | Mean bidirectional point-to-point distance between predicted and GT mesh |
| F-Score @ τ | Harmonic mean of precision and recall at distance threshold τ |
| PSNR | Peak signal-to-noise ratio on rendered novel views |

Optional comparison baselines (e.g. dense voxels, challenge-winning NeRF variants) are **out of scope** of this repository unless added separately.

---

## Implementation stack

| Component | Tool |
|---|---|
| Image encoder | ResNet-34 (torchvision, pretrained) |
| SDF MLP | PyTorch (`prism/sdf_mlp.py`), Fourier encoding + FiLM |
| Differentiable renderer | NeuS weights + stratified (and optional importance) sampling (`prism/renderer.py`, `prism/model.py`) |
| BRDF / shading | Custom PyTorch Cook–Torrance GGX (`prism/brdf.py`) |
| Multi-view conditioning | Optional projected features (`prism/model.py`) |
| Mesh extraction (eval only) | Marching cubes (**scikit-image**), optional mask-aware carve (`prism/mesh_extract.py`) |

---

## Key novelty

Prior single-image work often regresses geometry only, or uses neural fields without a compact BRDF/light parameterization tied to a **single** forward shading model. PRISM keeps a **small, interpretable** material and lighting head conditioned on the same latent that drives the SDF, and trains with **differentiable shading** plus **rich geometric supervision** (depth, normals, hull, SDF-from-depth), with **NeuS** for stable inverse rendering under sparse views.

The ICCV 2023 OmniObject3D challenge winner stacks depth on top of Pixel-NeRF-style radiance fields; PRISM instead uses an **implicit surface + GGX** path with **explicit depth/normal losses** and **silhouette / background** constraints suited to object-centric scans.

---

## Ablation studies

| Ablation | What it tests |
|---|---|
| Remove or down-weight `L_render` / `L_perceptual` | How much appearance supervision drives geometry vs. depth/normal alone |
| Single-view (`N=1`, no spatial features) vs. multi-view | Benefit of extra context views and local feature projection |
| Remove `L_eikonal` | Stability and quality of the learned SDF |
| Fixed light vs. learned light + ambient | Identifiability and shading fit |
| Loss weighting (geometry-first vs. balanced) | Sensitivity of Chamfer / PSNR trade-offs |

---

## References

- Wu et al., "OmniObject3D," CVPR 2023  
- Wang et al., "NeuS," NeurIPS 2021  
- Yariv et al., "IDR," NeurIPS 2020  
- Mildenhall et al., "NeRF," ECCV 2020  
- Du et al., "1st Place Solution, ICCV 2023 OmniObject3D Challenge," arXiv 2404.10441  
- Yang et al., "Learning Effective NeRFs and SDFs," arXiv 2309.16110  
