# Carlo's Progress Checkpoint & Implementation Tracker

> **Role**: Carlo — Text Modality Lead (CoOp, Context Optimization)
> **Project**: Comparing Parameter-Efficient Adaptation Strategies on Frozen CLIP for EuroSAT Satellite Classification
> **Branch**: `carlo`
> **Last Updated**: 2026-08-26
> **Scope of this document**: the first CoOp commit — `src/coop.py` and `src/few_shot.py`

---

## 1. Overview & Work Status

This document tracks Carlo's contribution: replacing CLIP's hand-written text prompts with **learnable continuous context vectors** (CoOp, Zhou et al., IJCV 2022), leaving the vision tower entirely to Marco and the infrastructure entirely to Mattia.

The two files committed here are the *model* and the *data protocol*. The experiment driver, the results and the plots are deliberately kept out of this commit and land separately, so that the reusable code and the run artifacts have independent histories.

### Progress Checklist

- [x] **Theory & Architecture Alignment**
  - [x] Analyzed CoOp (Zhou et al., IJCV 2022) and its relation to CLIP's prompt sensitivity.
  - [x] Defined the prompt layout: `[SOS] [v_1] ... [v_M] [tok(class)] [EOS] [pad]`, with only the $v_i$ trainable.
  - [x] Verified parameter efficiency: 2,048 / 8,192 trainable parameters (unified, $M=4$ / $M=16$) and 81,920 (class-specific, $M=16$, $C=10$) against 86M frozen backbone parameters.
- [x] **Modular Repository Setup**
  - [x] Created `src/coop.py`: implemented `PromptLearner` and `CoOpModel(BaseCLIPWrapper)`.
  - [x] Subclassed `BaseCLIPWrapper` from `base_model.py`, overriding **only** `get_text_features()`, so Marco's `get_image_features()` remains composable.
  - [x] Created `src/few_shot.py`: dataset-agnostic $K$-shot sampler, so the few-shot protocol does not require editing Mattia's `dataset.py`.
  - [x] Verified gradient flow: non-zero gradient on the context vectors, zero trainable parameters elsewhere (constructor assertion).
- [ ] **Hyperparameter Sweeps & Benchmark Experiments** *(next commit)*
  - [x] Learning-rate search over the text embeddings.
  - [ ] Context-length sweep $M \in \{4, 8, 16\}$.
  - [ ] Unified Context vs. Class-Specific Context.
  - [ ] Few-shot sweep $K \in \{1, 2, 4, 8, 16\}$.
  - [ ] Initialization ablation: random $\mathcal{N}(0, 0.02)$ vs. embeddings of `"a satellite image of"`.
- [ ] **Cross-Modal Integration**
  - [ ] Joint model `CoOpAdapterModel(CoOpModel, CLIPAdapterModel)` combining Carlo's learned prompts with Marco's vision adapter.
- [ ] **Final Report Section**
  - [ ] Write the Text Modality results section in `docs/report.tex`.

---

## 2. Technical Architecture & Design Decisions

### A. Prompt Learner (`PromptLearner`)

- **Structure**: the token sequence fed to the text transformer is assembled as
  $$\text{concat}\big(\underbrace{E_{[SOS]}}_{(C,1,D)},\ \underbrace{\text{ctx}}_{(C,M,D)},\ \underbrace{E_{\text{class}} \| E_{[EOS]} \| E_{\text{pad}}}_{(C,\,77-1-M,\,D)}\big) \in \mathbb{R}^{C \times 77 \times D}$$
  with $D = 512$ for ViT-B/32's text tower.
- **Two context modes**:
  - *Unified Context*: `ctx` of shape $(M, D)$, shared across all classes → $M \cdot D$ parameters (8,192 for $M=16$). Broadcast to all classes with `expand`, which creates a view rather than copying memory.
  - *Class-Specific Context (CSC)*: `ctx` of shape $(C, M, D)$ → $C \cdot M \cdot D$ parameters (81,920 for $M=16$, $C=10$).
- **Placeholder tokenization trick**: prefix and suffix are extracted by tokenizing `"X X ... X <class name>."` with exactly $M$ placeholders. Each `"X"` is a single BPE token, so the sequence has the intended layout **and** the `[EOS]` token lands at the index it would occupy in a real prompt. This matters because CLIP pools the sequence at the `[EOS]` position; an off-by-$M$ error there would silently read a meaningless row.
- **Buffer registration**: `token_prefix`, `token_suffix` and `tokenized_prompts` are registered with `register_buffer`. They must follow the module across `.to(device)` and into checkpoints, but must **not** appear in `parameters()`, or the optimizer would attempt to train frozen embeddings.
- **Weight Initialization Strategy**:
  - default: `ctx ~ N(0, 0.02)`, the recipe of the original paper;
  - optional: initialize from the embeddings of a real phrase (`ctx_init="a satellite image of"`), validated to tokenize to exactly $M$ tokens.
  - **Rationale**: the second variant starts the optimization from the hand-written prompt that zero-shot CLIP already uses, which makes "learned context beats engineered context" a controlled comparison rather than a comparison against noise.
- **No duplicate registration of frozen weights**: `token_embedding` is passed to the constructor but never assigned to `self`. Assigning it would register CLIP's frozen embedding table a second time inside the module and corrupt the trainable-parameter count, which is a first-class metric of this project.

### B. Manual Text-Encoder Forward (`_encode_text_from_embeddings`)

`open_clip`'s `encode_text()` takes token **ids** and performs the embedding lookup internally, leaving no insertion point for the learned context. The forward pass is therefore reproduced one step later, starting from token **embeddings**:

```
x = prompt_embeddings + positional_embedding
x = transformer(x, attn_mask=attn_mask)     # causal mask, 12 blocks
x = ln_final(x)
x = x[arange(C), tokenized_prompts.argmax(-1)]   # pool at [EOS]
x = x @ text_projection
```

- `argmax(-1)` locates `[EOS]` because it holds the highest id in CLIP's BPE vocabulary.
- The projection is applied through an `isinstance` check, since `open_clip` stores `text_projection` either as a plain matrix or as an `nn.Linear` depending on the model config.
- This procedure mirrors the manual forward Mattia already wrote in `optimal_transport.py:156-206`, reusing his handling of `attn_mask` and `cast_dtype`.

### C. Gradient Flow & Interface Compliance

- `BaseCLIPWrapper` sets `requires_grad = False` on all CLIP parameters; `CoOpModel` re-asserts this after building the prompt learner and adds a constructor assertion that the trainable-parameter count equals `prompt_learner.ctx.numel()` exactly.
- `get_text_features()` **omits** the `@torch.no_grad()` decorator carried by the base-class version. Inheriting it would let training run to completion without errors and without learning anything, since no gradient would reach the context vectors.
- Unlike a post-encoder adapter, the gradient here must traverse all 12 text transformer blocks. The frozen weights receive no updates, but the backward graph is still built — CoOp is therefore **more expensive per trainable parameter** than CLIP-Adapter, a cost worth reporting alongside accuracy.
- `predict()` recomputes the text prototypes on every call rather than caching them: the context vectors change after each optimizer step, so a cache built once in `__init__` (as `CLIPAdapterModel` does, correctly for its own case) would go stale during training. This is the failure mode the joint model must avoid.
- **Known contract deviation, handled locally**: `engine.train()` passes already-formatted prompts (`"a satellite image of forest"`) to `get_text_features()`, whereas the project brief specifies raw class names — which is what CoOp needs, the template being precisely what it replaces. `CoOpModel` stores the class names at construction and ignores the argument, raising an explicit error if the class count disagrees. This keeps Mattia's and Marco's code untouched; realigning the signature is a team decision, not a unilateral edit.

### D. Few-Shot Protocol (`few_shot.py`)

- **Motivation**: CoOp, CLIP-Adapter and Tip-Adapter are all benchmarked in the few-shot regime in their original papers. Training on all 21,600 EuroSAT images measures model capacity, not parameter efficiency, which is the question this project asks. CoOp runs use **16 shots per class** (160 images).
- **Evaluation is never subsampled**: only the training split is reduced. Test accuracy is always measured on the full 5,400-image split, so numbers stay comparable with the results Mattia and Marco already have.
- **Seeded, per-class sampling**: a global random draw would produce unbalanced classes, and an unseeded one would make two runs differ because of the draw rather than because of the method.
- **Labels without image decoding**: `extract_labels()` reads the label lists stored on the underlying torchvision objects (`targets`, `_labels`, `samples`) instead of iterating the dataset, which would decode and resize 21,600 JPEGs to obtain information already held in memory. A slow fallback with a printed warning covers unknown dataset types.
- **Dataset-agnostic by design**: the sampler wraps any dataset from `dataset.py`, so EuroSAT, DTD and Flowers102 all work, and Mattia and Marco can re-run their own methods in the same regime without anyone editing shared files.

---

## 3. Modular Interfacing Matrix

| Partner | Integration File / Function | Interface Contract | Carlo's Implementation |
|---|---|---|---|
| **Mattia** (Lead / Base class) | `src/base_model.py` (`BaseCLIPWrapper`) | Subclass and override one feature method | `CoOpModel(BaseCLIPWrapper)` overrides `get_text_features()` only; `get_image_features()` is inherited unchanged. |
| **Mattia** (Engine — training) | `src/engine.py` (`train`) | Calls `get_image_features()` and `get_text_features(prompts)` per batch | Compatible as-is. Text prototypes are recomputed every step, which is exactly what learned prompts require. The formatted-prompt argument is tolerated and ignored (see §2C). |
| **Mattia** (Engine — evaluation) | `src/engine.py` (`evaluate`) | Expects `model.predict(images) -> (preds, scores)` | `CoOpModel.predict()` returns cosine similarities; no `sinkhorn_reg` attribute, so the engine treats scores as similarities, correctly. |
| **Mattia** (Dataset) | `src/dataset.py` (`get_dataloaders`) | Yields `(images, labels, text_descriptions)` | `few_shot.build_few_shot_loader()` wraps the dataset and preserves the same tuple contract. |
| **Marco** (Vision Modality) | `src/clip_adapter.py` (`CLIPAdapterModel`) | Overrides `get_image_features()` | Disjoint override sets, so `CoOpAdapterModel(CoOpModel, CLIPAdapterModel)` inherits one method from each. Two items to resolve first: both classes define `predict()`, and Marco caches text prototypes in `__init__` while CoOp's change every step. |

---

## 4. Hyperparameter Search Space

| Hyperparameter | Values to Search | Recommended Default |
|---|---|---|
| **Context length ($M$)** | 4, 8, 16 | **16** (8,192 params) |
| **Context mode** | Unified, Class-Specific | **Unified** (fewer parameters; CSC is the capacity comparison) |
| **Shots per class ($K$)** | 1, 2, 4, 8, 16 | **16** |
| **Learning rate** | 2e-6 … 2e-1 | **2e-3** (paper value) |
| **Optimizer** | SGD + momentum | **SGD**, momentum 0.9, weight decay 5e-4 |
| **Scheduler** | Cosine annealing with warmup | 1 epoch warmup, 100 epochs total |
| **Initialization** | $\mathcal{N}(0, 0.02)$, or `"a satellite image of"` | **$\mathcal{N}(0, 0.02)$** |

**Note on warmup**: it is not optional. The gradient norm on the context vectors is of order $10^3$ while the vectors themselves are initialized at scale $0.02$; without a warmup epoch the first updates are catastrophic.

---

## 5. Summary of Created Files

Committed here:

1. `src/coop.py`: `PromptLearner` (learnable context, prompt assembly, placeholder tokenization) and `CoOpModel(BaseCLIPWrapper)` (manual text-encoder forward, `get_text_features()` override, `predict()`). A `__main__` smoke test verifies parameter counts for $M=4$, $M=16$ and CSC, and asserts a non-zero gradient on the context vectors.
2. `src/few_shot.py`: `extract_labels()`, `few_shot_indices()` and `build_few_shot_loader()`. A `__main__` smoke test verifies balanced $K$-shot draws for $K = 1, 4, 16$.

Deliberately excluded from this commit, landing separately:

3. `src/coop_experiments.py`: CLI experiment runner for the four sweeps, JSON result persistence and plotting.
4. `plots/coop_results.json`, `plots/coop_sweeps.png`, `plots/coop_training_curves.png`: run artifacts.
