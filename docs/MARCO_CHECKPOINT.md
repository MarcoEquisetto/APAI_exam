# Marco's Progress Checkpoint & Implementation Tracker

> **Role**: Marco — Vision Modality Lead (CLIP-Adapter)  
> **Project**: Comparing Parameter-Efficient Adaptation Strategies on Frozen CLIP for EuroSAT Satellite Classification  
> **Last Updated**: 2026-08-12  

---

## 1. Overview & Work Status

This document tracks Marco's contribution to the project: implementing and evaluating the visual modality adapter (**CLIP-Adapter**) and its optional extension (**Tip-Adapter**).

### Progress Checklist

- [x] **Theory & Architecture Alignment**
  - [x] Analyzed CLIP-Adapter architecture (Gao et al., IJCV 2024).
  - [x] Defined visual feature residual blending formula: $f_{\text{new}} = \alpha \cdot \text{MLP}(f_{\text{orig}}) + (1 - \alpha) \cdot f_{\text{orig}}$.
  - [x] Verified parameter efficiency (~65K to 130K trainable parameters vs 86M total CLIP parameters).
- [x] **Modular Repository Setup**
  - [x] Created `clip_adapter.py`: Implemented `VisionAdapterModule` and `CLIPAdapterModel(BaseCLIPWrapper)`.
  - [x] Subclassed `BaseCLIPWrapper` from `base_model.py` to ensure zero breaking changes to Mattia's engine.
  - [x] Created `clip_adapter_experiments.py`: Built experiment pipeline for hyperparameter sweeps and plotting.
  - [x] Populated `docs/sources.txt` with full literature and software citations for report inclusion.
- [ ] **Hyperparameter Sweeps & Benchmark Experiments**
  - [ ] Run bottleneck reduction ratio sweep ($r \in [2, 4, 8, 16]$).
  - [ ] Run residual blending ratio sweep ($\alpha \in [0.1, 0.2, 0.5, 0.8]$).
  - [ ] Collect metrics (Top-1 Acc, Top-5 Acc, Peak GPU Memory, Training Time).
  - [ ] Generate comparative plots (`plots/clip_adapter_reduction_sweep.png`).
- [ ] **Cross-Modal Integration**
  - [ ] Test joint inference combining Marco's vision adapter (`CLIPAdapterModel`) with Carlo's text prompt adapter (`CoOpModel`).
- [ ] **Optional Extension**
  - [ ] Implement Training-Free Tip-Adapter key-value cache if time permits.
- [ ] **Final Report Section**
  - [ ] Write Vision Modality results section in LaTeX (`docs/report.tex`).

---

## 2. Technical Architecture & Design Decisions

### A. Vision Bottleneck MLP (`VisionAdapterModule`)
- **Structure**: Linear Down-projection ($D \to R$) $\to$ ReLU $\to$ Linear Up-projection ($R \to D$).
  - For ViT-B/32 ($D = 512$):
    - Reduction ratio $r = 4 \implies R = 128 \implies$ ~131,200 trainable params.
    - Reduction ratio $r = 8 \implies R = 64 \implies$ ~65,600 trainable params.
- **Residual Blending**:
  $$f_{\text{blended}} = \alpha \cdot \text{Adapter}(f_{\text{orig}}) + (1 - \alpha) \cdot f_{\text{orig}}$$
  Followed by L2-normalization: $f_{\text{final}} = \frac{f_{\text{blended}}}{\|f_{\text{blended}}\|_2}$.
- **Weight Initialization Strategy**:
  - `down_proj`: Kaiming Uniform initialization with ReLU nonlinearity.
  - `up_proj`: Normal initialization near zero ($\sigma = 1e-4$) and zero bias.
  - **Rationale**: Ensures that at epoch 0, the adapter's initial contribution is near zero, preserving pre-trained CLIP zero-shot representations and avoiding gradient shocks.

### B. Gradient Flow & Interface Compliance
- `BaseCLIPWrapper` sets `requires_grad = False` for all standard CLIP parameters.
- In `CLIPAdapterModel`, only `self.adapter` parameters have `requires_grad = True`.
- `get_image_features(self, images)` runs `self.model.encode_image(images)` inside `torch.no_grad()` to avoid building backprop graphs for frozen backbone layers, then passes original features through `self.adapter` (gradient enabled).
- `get_text_features(self, class_names)` inherits standard frozen text encoding from `BaseCLIPWrapper`, keeping it ready for Carlo's text adapter override.

---

## 3. Modular Interfacing Matrix

| Partner | Integration File / Function | Interface Contract | Marco's Implementation |
|---|---|---|---|
| **Mattia** (Lead / Baseline) | `src/base_model.py` (`BaseCLIPWrapper`) | Subclass `BaseCLIPWrapper` | Implemented `CLIPAdapterModel(BaseCLIPWrapper)` in `src/clip_adapter.py`. |
| **Mattia** (Dataset) | `src/dataset.py` (`get_dataloaders`) | Yields `(images, labels, text)` | Consumed in `src/clip_adapter_experiments.py`. |
| **Mattia** (Engine) | `src/engine.py` (`train`, `evaluate`) | Calls `get_image_features()`, `get_text_features()`, `predict()` | `CLIPAdapterModel` exposes `get_image_features()` and `predict()`. |
| **Carlo** (Text Modality) | `src/coop.py` (CoOp subclass) | Overrides `get_text_features()` | Seamlessly stackable: a joint model can inherit both `get_image_features()` from Marco and `get_text_features()` from Carlo. |

---

## 4. Hyperparameter Search Space

| Hyperparameter | Values to Search | Recommended Default |
|---|---|---|
| **Reduction Ratio ($r$)** | 2, 4, 8, 16 | **4** (Hidden Dim = 128) |
| **Residual Alpha ($\alpha$)** | 0.1, 0.2, 0.5, 0.8 | **0.2** |
| **Learning Rate** | 1e-4, 5e-4, 1e-3, 2e-3 | **1e-3** (AdamW) |
| **Weight Decay** | 1e-4, 5e-4, 1e-3 | **5e-4** |
| **Scheduler** | Cosine Annealing with Warmup | 1 epoch warmup, 10 epochs total |

---

## 5. Summary of Created Files

1. `clip_adapter.py`: Production-ready PyTorch module containing `VisionAdapterModule` and `CLIPAdapterModel`.
2. `clip_adapter_experiments.py`: CLI experiment runner for reduction sweeps, alpha tuning, and automated graph saving.
3. `docs/sources.txt`: Complete bibliography of academic papers, dataset references, and libraries.
4. `docs/MARCO_CHECKPOINT.md`: This active checkpoint tracker.
