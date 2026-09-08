# Beyond Single Prompts: Comparing Vision Adapters, CoOp, and Optimal Transport for Domain-Specific Image Classification

APAI exam project — comparing parameter-efficient adaptation strategies on a **frozen** CLIP backbone.

## Goal

The question is not only *which method is most accurate*, but **how much accuracy each trainable parameter buys**. Every method plugs into the same base class and the same evaluation loop, over the same frozen backbone, so accuracy, parameter count, GPU memory and wall-clock time are directly comparable.

| Method | Modality | Trainable params (EuroSAT) | Owner |
|---|---|---|---|
| Zero-Shot CLIP | — | 0 | Mattia |
| Zero-Shot + prompt ensembling | Text | 0 | Mattia |
| Linear Probe | Vision (sklearn) | 5,130 | Mattia |
| Optimal Transport (Sinkhorn) | Vision + text tokens | 0 | Mattia |
| CoOp | Text (learnable prompts) | 8,192 | Carlo |
| CLIP-Adapter | Vision (bottleneck MLP) | 131,712 | Marco |
| CoOp + CLIP-Adapter (joint) | Text + vision | 139,904 | Carlo |
| Tip-Adapter | Vision (key-value cache) | 0 | Marco |
| Tip-Adapter-F | Vision (fine-tuned cache) | 16 · C · 512 | Marco |
| LoRA on the vision tower | Vision (inside the backbone) | 368,640 | Marco |

> The parameter counts of the Linear Probe and of the Tip-Adapter cache scale with the number of classes, so they differ per dataset. `run_all.py` prints one table for accuracy and a separate one for parameters, precisely because these two do not share a column.

**Datasets:** EuroSAT (10 classes, satellite), DTD (47, textures), Flowers102 (102, fine-grained).

## Repository structure

```
├── src/
│   ├── base_model.py               # BaseCLIPWrapper: frozen backbone, get_image_features,
│   │                               #   get_text_features, get_image_patch_tokens, predict
│   ├── dataset.py                  # EuroSAT / DTD / Flowers102 → (images, labels, text)
│   ├── few_shot.py                 # seeded K-shot sampling, method-agnostic
│   ├── baselines.py                # ZeroShotCLIP, ZeroShotEnsembleCLIP, LinearProbeCLIP
│   ├── optimal_transport.py        # OptimalTransportCLIP (Sinkhorn on patch/text tokens)
│   ├── coop.py                     # CoOpModel — learnable text context          (Carlo)
│   ├── clip_adapter.py             # CLIPAdapterModel, TipAdapterModel,
│   │                               #   VisionLoRAModel                            (Marco)
│   ├── coop_adapter.py             # CoOpAdapterModel — text + vision jointly     (Carlo)
│   ├── engine.py                   # train(), evaluate(), comparative plots
│   ├── run_all.py                  # THE experiment driver: every method × every dataset
│   ├── coop_experiments.py         # CoOp ablations: M, CSC, LR, shots
│   └── clip_adapter_experiments.py # CLIP-Adapter sweeps: reduction ratio, alpha
├── docs/                           # integration report, command reference
├── plots/                          # figures and result JSON (written by run_all.py only)
├── data/                           # datasets (auto-download, git-ignored)
├── requirements.txt
└── environment.yml
```

## Setup

```bash
# 1. PyTorch first, from the official index — the PyPI wheel has no CUDA.
pip install torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu121

# 2. Everything else.
pip install -r requirements.txt

# 3. Check that CUDA really came through.
python -c "import torch; print(torch.cuda.is_available())"
```

Conda alternative: `conda env create -f environment.yml && conda activate APAI`.

> `cu121` is deliberate. One of the project machines is a Quadro P2200 (Pascal, sm_61), and recent PyTorch builds no longer compile for Pascal: upgrading `torch` leaves CUDA silently unavailable.

## Running the experiments

```bash
# The main grid: 16-shot, matched epoch budget, three seeds.
python src/run_all.py --shots 16 --seeds 0 1 2 --epochs 10

# Full-shot reference (whole training split, one seed).
python src/run_all.py --shots 0 --seeds 0 --epochs 10

# One dataset at a time — an interruption then costs one dataset, not all three.
python src/run_all.py --dataset eurosat --shots 16 --seeds 0 1 2

# Quick end-to-end check, a few minutes.
python src/run_all.py --dataset eurosat --shots 16 --seeds 0 --epochs 2 --no-tsne

# Redraw every figure from the results already on disk. No GPU.
python src/run_all.py --plots-only
```

Per-workstream ablations:

```bash
python src/coop_experiments.py --sweep lr        # also: ctx_len, csc, shots, all
python src/clip_adapter_experiments.py --epochs 5
python src/engine.py                             # baselines only → plots/baselines_only/
```

Smoke tests, all cheap:

```bash
python src/dataset.py        # dataloaders yield (images, labels, texts)
python src/few_shot.py       # K-shot sampling is seeded and balanced
python src/coop.py           # parameter counts + gradient reaches ctx
python src/coop_adapter.py   # joint model, optimizer groups, MRO trap
python src/clip_adapter.py   # adapter, Tip-Adapter, LoRA
```

`docs/COMANDI.md` is the full command reference, with measured timings and a symptom-by-symptom diagnostic section.

## Integrating a new method

```python
from src.base_model import BaseCLIPWrapper
from src.dataset import get_dataloaders
from src.engine import train, evaluate


class MyAdapter(BaseCLIPWrapper):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # ... add trainable parameters here ...

    def get_image_features(self, images):      # vision-side methods override this
        ...

    def get_text_features(self, class_names):  # text-side methods override this
        ...


model = MyAdapter(device="cuda")
train_loader, test_loader = get_dataloaders()
history = train(model, train_loader, val_loader=test_loader, model_name="MyAdapter")
metrics = evaluate(model, test_loader, model_name="MyAdapter")
```

Three rules that are easy to break:

1. **Do not put `@torch.no_grad()` on an overridden feature method.** The base-class version has it; an override that inherits it trains without learning anything, and without raising.
2. **Do not cache text prototypes in `__init__`** if the text side is trainable. `BaseCLIPWrapper.predict()` recomputes them per call for exactly this reason.
3. **A method needing a different learning rate per module** defines `trainable_param_groups(lr)`; `engine.train()` picks it up automatically.

## Results

Full-shot, single seed, ViT-B/32 — Top-1 accuracy (%):

| Method | EuroSAT | DTD | Flowers102 |
|---|---|---|---|
| Zero-Shot | 46.35 | 54.89 | 71.46 |
| Zero-Shot + ensembling | 50.61 | 57.34 | 71.93 |
| Linear Probe | 93.81 | 70.90 | 86.97 |
| Optimal Transport | 20.31 | 23.30 | 19.06 |
| CoOp | 95.37 | 73.46 | 82.35 |
| CLIP-Adapter | 96.59 | 75.21 | 83.88 |
| CoOp + Adapter | 95.44 | 74.31 | 76.32 |
| Tip-Adapter | 76.13 | 68.03 | 81.17 |
| Tip-Adapter-F | 96.56 | 78.24 | 94.32 |
| LoRA | 98.17 | 65.90 | 74.47 |

The headline: on EuroSAT, **8,192 trainable parameters — 0.0095 % of the backbone — are worth +49 points over zero-shot**, and the adapter's 131,712 (sixteen times as many) add only 1.2 more.

> ⚠️ These numbers predate the protocol fixes described below: they were produced with unmatched epoch budgets, a Tip-Adapter cache built over the entire training split, and the joint model trained with CoOp's optimizer for both of its halves. Re-run `run_all.py` before quoting them in the report.

## Known limitations

Stated here so they are stated somewhere, and so the report can repeat them rather than discover them:

- **No validation split.** `val_loader` is the test set, so the per-epoch curves and the "Best Val" line are test accuracy, and the hyperparameters were chosen with the test set visible. Common in the few-shot CLIP literature, but it means these numbers are optimistic.
- **The weights are `laion2b_s34b_b79k`, not OpenAI's.** The brief says "OpenAI CLIP via `open_clip`"; the LAION-2B checkpoint is what is actually loaded, and zero-shot numbers differ between the two.
- **ViT-L/14 was never run.** Only ViT-B/32.
- **No data augmentation** during training (no random crop, no flip), unlike the CoOp and CLIP-Adapter papers.
- **Optimal Transport scores below zero-shot on all three datasets.** This is a property of the method as implemented, not a crash: patch tokens are not in the contrastively aligned space, the marginals weight background patches as heavily as informative ones, and the shared template words dominate the text-token set. It needs explaining in the report, not hiding.
- **LoRA is not in the project brief** and is the only method that reaches inside the backbone. The pretrained weights do stay frozen, but it should be framed as a deliberate out-of-brief comparison.

## Authors

- **Mattia** — infrastructure, baselines, Optimal Transport
- **Carlo** — CoOp, few-shot protocol, joint model
- **Marco** — CLIP-Adapter, Tip-Adapter, LoRA
