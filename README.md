# Beyond Single Prompts: Comparing Vision Adapters, CoOp, and Optimal Transport for Domain-Specific Image Classification

APAI exam project — Comparing parameter-efficient adaptation strategies on frozen CLIP for satellite image classification (EuroSAT).

## Goal

Evaluate the **performance vs. trainable parameters** trade-off across different CLIP adaptation strategies:

| Method | Modality | Trainable Params | Owner |
|---|---|---|---|
| Zero-Shot CLIP | — | 0 | Mattia |
| Linear Probe | Vision (sklearn) | ~5K | Mattia |
| Optimal Transport (Sinkhorn) | Vision+Text tokens | 0 | Mattia |
| CoOp | Text (learnable prompts) | ~2-8K | Carlo |
| CLIP-Adapter | Vision (bottleneck MLP) | ~65-130K | Marco |

## Repository Structure

```
├── src/
│   ├── base_model.py          # BaseCLIPWrapper — base interface (get_image_features, get_text_features)
│   ├── baselines.py           # ZeroShotCLIP, LinearProbeCLIP
│   ├── clip_adapter.py        # Vision CLIP-Adapter (Marco)
│   ├── clip_adapter_experiments.py # Hyperparameter exploration script
│   ├── dataset.py             # EuroSATDataset, get_dataloaders → (images, labels, text)
│   ├── engine.py              # evaluate(), train(), plot_comparative_results()
│   └── optimal_transport.py   # OptimalTransportCLIP (Sinkhorn on patch/text tokens)
├── environment.yaml           # Conda environment
├── data/                      # EuroSAT (auto-download)
├── docs/                      # Progress checkpoints & bibliography
└── plots/                     # Generated comparative plots
```

## Setup

```bash
# Option 1: Conda
conda env create -f environment.yaml
conda activate APAI

# Option 2: pip
pip install -r requirements.txt
```

## Usage

### Run baselines (Mattia)

```bash
python src/engine.py
```

Runs Zero-Shot, Linear Probe and Optimal Transport on EuroSAT, then saves comparative plots to `plots/`.

### Run Marco's CLIP-Adapter Experiments

```bash
python src/clip_adapter_experiments.py --epochs 5
```

### Integrate an adapter (Carlo / Marco)

```python
from src.base_model import BaseCLIPWrapper
from src.engine import train, evaluate
from src.dataset import get_dataloaders
```

class MyAdapter(BaseCLIPWrapper):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Add trainable parameters here

    def get_image_features(self, images):     # Marco overrides this
        ...

    def get_text_features(self, class_names): # Carlo overrides this
        ...

# Training
model = MyAdapter(device="cuda")
train_loader, test_loader = get_dataloaders()
history = train(model, train_loader, val_loader=test_loader, model_name="MyAdapter")
```

## Baseline Results

```
[ZeroShot]          Top-1: 44.56% | Top-5: 89.78% | Params: 0
[LinearProbe]       Top-1: 93.61% | Top-5: 99.91% | Params: 5,130
[OptimalTransport]  Top-1: 21.74% | Top-5: 73.59% | Params: 0
```

## Key Dependencies

- PyTorch ≥ 2.0
- OpenCLIP (`open_clip_torch`)
- POT (Python Optimal Transport)
- Weights & Biases (`wandb`)
- scikit-learn
- torchvision

## Authors

- **Mattia** — Infrastructure, baselines, Optimal Transport
- **Carlo** — CoOp (text adapter)
- **Marco** — CLIP-Adapter (vision adapter)