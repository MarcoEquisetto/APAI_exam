"""
run_all.py — Confronto unificato di tutti i metodi su tutti i dataset
=====================================================================

Allena e valuta ogni metodo sullo stesso split, per ogni dataset, e produce
la tabella comparativa finale, i plot unificati e una t-SNE visualization.

Protocollo
----------

Perché il confronto significhi qualcosa, tre cose devono essere uguali per
tutti i metodi messi sullo stesso grafico:

1. **La supervisione.**  ``--shots K`` addestra ogni metodo sullo *stesso*
   support set K-shot, estratto con un seed fisso da ``src/few_shot.py``.
   ``--shots 0`` usa l'intero training split (regime full-shot).
   Confrontare CoOp addestrato su 160 immagini con un Linear Probe
   addestrato su 21.600 misura anche un fattore 135 di supervisione.
2. **Il budget.**  ``--epochs`` vale per tutti i metodi addestrabili.
   L'*ottimizzatore* resta invece quello del paper di ciascun metodo (SGD
   per CoOp, AdamW per adapter e LoRA): quello fa parte del metodo, le
   epoche no.
3. **La varianza.**  ``--seeds`` ripete ogni metodo addestrabile su più
   semi e riporta media ± deviazione standard.  I metodi training-free
   (Zero-Shot, Ensemble, Optimal Transport) sono deterministici e girano
   una volta sola.

**Limite noto, da dichiarare nel report:** non esiste uno split di
validazione.  ``val_loader`` è il test set, quindi le curve per epoca e il
"Best Val" stampato sono accuratezza di test, e gli iperparametri sono stati
scelti con il test visibile.  È una pratica diffusa nella letteratura
few-shot su CLIP, ma va scritta, non nascosta.

Usage
-----
::

    # Griglia principale: 16-shot, budget appaiato, tre semi
    python src/run_all.py --shots 16 --seeds 0 1 2 --epochs 10

    # Riferimento full-shot (un solo seme)
    python src/run_all.py --shots 0 --seeds 0 --epochs 10

    # Un dataset alla volta (consigliato: un'interruzione non perde tutto)
    python src/run_all.py --dataset eurosat --shots 16 --seeds 0 1 2

    # Prova rapida end-to-end, pochi minuti
    python src/run_all.py --dataset eurosat --shots 16 --seeds 0 --epochs 2 --no-tsne

    # Salta Optimal Transport (molto lento su >10 classi)
    python src/run_all.py --skip-slow

    # Rigenera SOLO le figure dai risultati già su disco, senza GPU
    python src/run_all.py --plots-only
"""

import sys
import os
import io
import json
import time
import argparse
import contextlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path setup (same convention as the rest of the project).
# ---------------------------------------------------------------------------
FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parent
if str(FILE_DIR) not in sys.path:
    sys.path.insert(0, str(FILE_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# SSL fix (same as engine.py — needed on Windows for dataset downloads).
# ---------------------------------------------------------------------------
import ssl
import certifi

def _create_ssl_context(purpose=ssl.Purpose.SERVER_AUTH, *, cafile=None,
                        capath=None, cadata=None):
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.load_verify_locations(cafile=certifi.where())
    return ctx

ssl.create_default_context = _create_ssl_context
ssl._create_default_https_context = _create_ssl_context

# ---------------------------------------------------------------------------
# Project imports.
# ---------------------------------------------------------------------------
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from src import engine
    from src.base_model import BaseCLIPWrapper
    from src.baselines import ZeroShotCLIP, ZeroShotEnsembleCLIP, LinearProbeCLIP
    from src.optimal_transport import OptimalTransportCLIP
    from src.coop import CoOpModel
    from src.clip_adapter import (
        CLIPAdapterModel, TipAdapterModel, VisionLoRAModel,
    )
    from src.coop_adapter import CoOpAdapterModel
    from src.few_shot import build_few_shot_loader
    from src.dataset import (
        EuroSATDataset, get_dataloaders,
        DTDDataset, get_dtd_dataloaders,
        Flowers102Dataset, get_flowers102_dataloaders,
        EUROSAT_CLASS_NAMES, PROMPT_TEMPLATE, EUROSAT_PROMPT_TEMPLATES,
        DTD_CLASS_NAMES, DTD_PROMPT_TEMPLATE, DTD_PROMPT_TEMPLATES,
        FLOWERS102_CLASS_NAMES, FLOWERS102_PROMPT_TEMPLATE,
        FLOWERS102_PROMPT_TEMPLATES,
    )
except ModuleNotFoundError:
    import engine
    from base_model import BaseCLIPWrapper
    from baselines import ZeroShotCLIP, ZeroShotEnsembleCLIP, LinearProbeCLIP
    from optimal_transport import OptimalTransportCLIP
    from coop import CoOpModel
    from clip_adapter import (
        CLIPAdapterModel, TipAdapterModel, VisionLoRAModel,
    )
    from coop_adapter import CoOpAdapterModel
    from few_shot import build_few_shot_loader
    from dataset import (
        EuroSATDataset, get_dataloaders,
        DTDDataset, get_dtd_dataloaders,
        Flowers102Dataset, get_flowers102_dataloaders,
        EUROSAT_CLASS_NAMES, PROMPT_TEMPLATE, EUROSAT_PROMPT_TEMPLATES,
        DTD_CLASS_NAMES, DTD_PROMPT_TEMPLATE, DTD_PROMPT_TEMPLATES,
        FLOWERS102_CLASS_NAMES, FLOWERS102_PROMPT_TEMPLATE,
        FLOWERS102_PROMPT_TEMPLATES,
    )


RESULTS_DIR = PROJECT_ROOT / "plots"
RESULTS_JSON = RESULTS_DIR / "unified_results.json"


# ============================================================================
# Dataset registry
# ============================================================================

DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {
    "eurosat": {
        "class_names": EUROSAT_CLASS_NAMES,
        "prompt_template": PROMPT_TEMPLATE,
        "ensemble_templates": EUROSAT_PROMPT_TEMPLATES,
        "get_loaders": get_dataloaders,
        "train_dataset_cls": EuroSATDataset,
        "short_name": "EuroSAT",
    },
    "dtd": {
        "class_names": DTD_CLASS_NAMES,
        "prompt_template": DTD_PROMPT_TEMPLATE,
        "ensemble_templates": DTD_PROMPT_TEMPLATES,
        "get_loaders": get_dtd_dataloaders,
        "train_dataset_cls": DTDDataset,
        "short_name": "DTD",
    },
    "flowers102": {
        "class_names": FLOWERS102_CLASS_NAMES,
        "prompt_template": FLOWERS102_PROMPT_TEMPLATE,
        "ensemble_templates": FLOWERS102_PROMPT_TEMPLATES,
        "get_loaders": get_flowers102_dataloaders,
        "train_dataset_cls": Flowers102Dataset,
        "short_name": "Flowers102",
    },
}


def load_data(
    dataset_name: str,
    batch_size: int = 64,
    num_workers: int = 0,
    shots: int = 0,
    seed: int = 0,
) -> Tuple[Any, Any, Any, Dict[str, Any]]:
    """
    Load train/test loaders, the raw train dataset, and dataset metadata.

    Parameters
    ----------
    shots : int
        ``0`` → full training split.  ``K > 0`` → a K-shot support set drawn
        by ``few_shot.build_few_shot_loader()``, the same one for **every**
        method, so the comparison holds supervision fixed.
    seed : int
        Seed of the support-set draw (ignored when ``shots == 0``).

    Returns
    -------
    train_loader, test_loader, train_dataset, info
    """
    cfg = DATASET_REGISTRY[dataset_name]

    full_train_loader, test_loader = cfg["get_loaders"](
        batch_size=batch_size, num_workers=num_workers, download=True,
    )
    train_dataset = cfg["train_dataset_cls"](split="train", download=True)

    if shots > 0:
        # Few-shot batches are small; 32 is CoOp's setting and gives a
        # sane number of optimizer steps per epoch on 160 images.
        train_loader = build_few_shot_loader(
            train_dataset,
            n_shots=shots,
            batch_size=min(batch_size, 32),
            num_workers=num_workers,
            seed=seed,
        )
    else:
        train_loader = full_train_loader

    info = {
        "class_names": cfg["class_names"],
        "prompt_template": cfg["prompt_template"],
        "ensemble_templates": cfg["ensemble_templates"],
        "short_name": cfg["short_name"],
        "n_classes": len(cfg["class_names"]),
        "n_train_full": len(train_dataset),
        "n_train_used": len(train_loader.dataset),
        "n_test": len(test_loader.dataset),
        "shots": shots,
    }
    return train_loader, test_loader, train_dataset, info


# ============================================================================
# Per-method training recipes
# ============================================================================
# The optimizer, its learning rate and the weight decay are part of the
# method as published; the epoch count is not, and is therefore driven by a
# single ``--epochs`` so that no method is handed a larger budget than the
# one it is being compared against.
# ============================================================================

RECIPES: Dict[str, Dict[str, Any]] = {
    # CoOp: SGD + momentum, as in Zhou et al. (2022).
    "CoOp": {"optimizer_type": "sgd", "lr": 2e-3, "momentum": 0.9},
    # CLIP-Adapter: AdamW, as in Gao et al. (2024).
    "CLIP-Adapter": {"optimizer_type": "adamw", "lr": 1e-3},
    # Joint model: AdamW, the recipe of the half with the most parameters.
    # Training it with CoOp's SGD (the first version of this script) left the
    # adapter's near-zero-initialized weights essentially where they started
    # and made the joint model lose to the plain adapter on all three
    # datasets.  See CoOpAdapterModel.trainable_param_groups().
    "CoOp+Adapter": {"optimizer_type": "adamw", "lr": 1e-3},
    "LoRA": {"optimizer_type": "adamw", "lr": 1e-4},
    "Tip-Adapter-F": {"lr": 1e-3},
}


def _finish(
    metrics: Dict[str, Any],
    history: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Attach the training history to a metrics dict.

    The per-epoch GPU memory series (``gpu_mb`` and ``gpu_epoch_mb``) are
    **kept**.  An earlier version of this file stripped them right here,
    which is why the brief's "Memory Usage vs. Epochs" figure could not be
    drawn: the data was collected by ``engine.train()`` and thrown away one
    function call later.
    """
    if history is not None:
        metrics["history"] = history
        metrics["train_time_seconds"] = history.get("total_time_seconds")
        metrics["peak_gpu_mb"] = history.get("peak_gpu_mb")
    return metrics


def _release(model) -> None:
    """Drop a model and give its VRAM back — 5 GB cards need this."""
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================================
# Method runners
# ============================================================================
# Each function trains (if needed) and evaluates a single method, returning
# the metrics dict from engine.evaluate().  Any function that builds a model
# extracts its t-SNE features *before* releasing it: the features of a
# deleted model cannot be recovered afterwards, which is exactly how the
# t-SNE figure ended up with a single panel.
# ============================================================================

def run_zero_shot(
    clip_wrapper: BaseCLIPWrapper,
    test_loader,
    info: dict,
) -> Dict[str, Any]:
    """Zero-Shot CLIP (single template)."""
    model = ZeroShotCLIP(
        clip_wrapper,
        class_names=info["class_names"],
        prompt_template=info["prompt_template"],
    )
    return engine.evaluate(
        model, test_loader, model_name="ZeroShot", use_wandb=False,
    )


def run_zero_shot_ensemble(
    clip_wrapper: BaseCLIPWrapper,
    test_loader,
    info: dict,
) -> Dict[str, Any]:
    """Zero-Shot CLIP with prompt ensembling."""
    model = ZeroShotEnsembleCLIP(
        clip_wrapper,
        class_names=info["class_names"],
        templates=info["ensemble_templates"],
    )
    return engine.evaluate(
        model, test_loader, model_name="ZeroShot-Ensemble", use_wandb=False,
    )


def run_linear_probe(
    clip_wrapper: BaseCLIPWrapper,
    train_loader,
    test_loader,
) -> Dict[str, Any]:
    """
    Linear Probe (sklearn Logistic Regression on frozen features).

    Fitted on whatever ``train_loader`` carries — the K-shot support set in
    few-shot mode, the whole split in full-shot mode.  That is the point:
    a probe fitted on 21.600 images is not a baseline for a prompt learner
    fitted on 160.
    """
    model = LinearProbeCLIP(clip_wrapper)
    model.fit(train_loader)
    return engine.evaluate(
        model, test_loader, model_name="LinearProbe", use_wandb=False,
    )


def run_optimal_transport(
    clip_wrapper: BaseCLIPWrapper,
    test_loader,
    info: dict,
) -> Dict[str, Any]:
    """Optimal Transport (Sinkhorn on patch/text tokens). SLOW."""
    model = OptimalTransportCLIP(
        clip_wrapper,
        class_names=info["class_names"],
        prompt_template=info["prompt_template"],
    )
    return engine.evaluate(
        model, test_loader, model_name="OptimalTransport", use_wandb=False,
    )


def run_coop(
    device: str,
    train_loader,
    test_loader,
    info: dict,
    epochs: int,
    seed: int,
    tsne_sink: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """CoOp: Context Optimization (M=16, unified context)."""
    torch.manual_seed(seed)
    recipe = RECIPES["CoOp"]

    model = CoOpModel(
        device=device,
        class_names=info["class_names"],
        n_ctx=16,
        class_specific=False,
    )
    history = engine.train(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        class_names=info["class_names"],
        epochs=epochs,
        lr=recipe["lr"],
        optimizer_type=recipe["optimizer_type"],
        weight_decay=5e-4,
        momentum=recipe.get("momentum", 0.9),
        scheduler_type="cosine",
        warmup_epochs=1,
        use_wandb=False,
        model_name="CoOp",
    )
    metrics = engine.evaluate(
        model, test_loader, model_name="CoOp", use_wandb=False,
    )
    _finish(metrics, history)

    if tsne_sink is not None:
        tsne_sink["CoOp"] = extract_features(model, test_loader)

    _release(model)
    return metrics


def run_clip_adapter(
    device: str,
    train_loader,
    test_loader,
    info: dict,
    epochs: int,
    seed: int,
    reduction_ratio: int = 4,
    alpha: float = 0.2,
    tsne_sink: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """CLIP-Adapter: vision bottleneck MLP (r=4, alpha=0.2)."""
    torch.manual_seed(seed)
    recipe = RECIPES["CLIP-Adapter"]

    model = CLIPAdapterModel(
        device=device,
        reduction_ratio=reduction_ratio,
        alpha=alpha,
        class_names=info["class_names"],
        prompt_template=info["prompt_template"],
    )
    history = engine.train(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        class_names=info["class_names"],
        epochs=epochs,
        lr=recipe["lr"],
        optimizer_type=recipe["optimizer_type"],
        weight_decay=5e-4,
        scheduler_type="cosine",
        warmup_epochs=1,
        use_wandb=False,
        model_name="CLIP-Adapter",
    )
    metrics = engine.evaluate(
        model, test_loader, model_name="CLIP-Adapter", use_wandb=False,
    )
    _finish(metrics, history)

    # Extract the ADAPTED features while the trained model still exists.
    # This is the panel the t-SNE figure is for: frozen features next to
    # the ones the adapter produced.
    if tsne_sink is not None:
        tsne_sink["CLIP-Adapter"] = extract_features(model, test_loader)

    _release(model)
    return metrics


def run_joint(
    device: str,
    train_loader,
    test_loader,
    info: dict,
    epochs: int,
    seed: int,
    adapter_lr_scale: float = 1.0,
    tsne_sink: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Joint CoOp + CLIP-Adapter: text and vision adapted together.

    The two halves go into two optimizer groups (see
    ``CoOpAdapterModel.trainable_param_groups``), under AdamW rather than
    CoOp's SGD.  With one SGD rate for everything the adapter contributed
    essentially nothing and the joint model scored *below* the plain
    adapter — a configuration artefact that read like a finding.
    """
    torch.manual_seed(seed)
    recipe = RECIPES["CoOp+Adapter"]

    model = CoOpAdapterModel(
        device=device,
        class_names=info["class_names"],
        n_ctx=16,
        class_specific=False,
        reduction_ratio=4,
        alpha=0.2,
        adapter_lr_scale=adapter_lr_scale,
    )
    history = engine.train(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        class_names=info["class_names"],
        epochs=epochs,
        lr=recipe["lr"],
        optimizer_type=recipe["optimizer_type"],
        weight_decay=5e-4,
        scheduler_type="cosine",
        warmup_epochs=1,
        use_wandb=False,
        model_name="CoOp+Adapter",
    )
    metrics = engine.evaluate(
        model, test_loader, model_name="CoOp+Adapter", use_wandb=False,
    )
    _finish(metrics, history)
    metrics["parameter_breakdown"] = model.parameter_breakdown()

    if tsne_sink is not None:
        tsne_sink["CoOp+Adapter"] = extract_features(model, test_loader)

    _release(model)
    return metrics


def run_tip_adapter(
    device: str,
    train_dataset,
    train_loader,
    test_loader,
    info: dict,
    epochs: int,
    seed: int,
    cache_shots: int = 16,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Tip-Adapter (training-free cache) + Tip-Adapter-F (fine-tuned cache).

    The cache is built from a **K-shot support set**, not from the whole
    training split.  Building it from everything turns Tip-Adapter into a
    nearest-neighbour index over 21.600 images and makes Tip-Adapter-F
    report 11.059.200 "trainable parameters" — a number that is technically
    correct (the keys really are optimized) and completely incomparable with
    the 8.192 of CoOp on the same axis.

    The support set comes from ``few_shot.build_few_shot_loader`` with the
    run's seed, so it is the *same* one the trained methods saw and it does
    not depend on the shuffle order of the training loader, as it did when
    the cache was filled by consuming ``train_loader`` until every class had
    enough samples.

    Returns two metrics dicts: ``(tip_metrics, tipf_metrics)``.
    """
    torch.manual_seed(seed)

    model = TipAdapterModel(
        device=device,
        class_names=info["class_names"],
        prompt_template=info["prompt_template"],
    )

    cache_loader = build_few_shot_loader(
        train_dataset,
        n_shots=cache_shots,
        batch_size=64,
        num_workers=0,
        seed=seed,
        shuffle=False,
    )
    model.build_cache(cache_loader, num_shots=cache_shots)

    tip_metrics = engine.evaluate(
        model, test_loader, model_name="Tip-Adapter", use_wandb=False,
    )

    # Tip-Adapter-F: the cache keys become nn.Parameter and are optimized.
    # Same epoch budget as every other trained method.
    t0 = time.time()
    model.finetune_cache(
        train_loader, epochs=epochs, lr=RECIPES["Tip-Adapter-F"]["lr"],
    )
    tipf_metrics = engine.evaluate(
        model, test_loader, model_name="Tip-Adapter-F", use_wandb=False,
    )
    tipf_metrics["train_time_seconds"] = time.time() - t0

    _release(model)
    return tip_metrics, tipf_metrics


def run_lora(
    device: str,
    train_loader,
    test_loader,
    info: dict,
    epochs: int,
    seed: int,
    r: int = 4,
    tsne_sink: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Vision LoRA: low-rank adaptation of the ViT MLP layers."""
    torch.manual_seed(seed)
    recipe = RECIPES["LoRA"]

    model = VisionLoRAModel(
        device=device,
        r=r,
        class_names=info["class_names"],
        prompt_template=info["prompt_template"],
    )
    history = engine.train(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        class_names=info["class_names"],
        epochs=epochs,
        lr=recipe["lr"],
        optimizer_type=recipe["optimizer_type"],
        weight_decay=5e-4,
        scheduler_type="cosine",
        warmup_epochs=1,
        use_wandb=False,
        model_name="LoRA",
    )
    metrics = engine.evaluate(
        model, test_loader, model_name="LoRA", use_wandb=False,
    )
    _finish(metrics, history)

    if tsne_sink is not None:
        tsne_sink["LoRA"] = extract_features(model, test_loader)

    _release(model)
    return metrics


# ============================================================================
# Aggregation over seeds
# ============================================================================

def aggregate_runs(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collapse several seeds of the same method into one record.

    A single run reports a number; several runs report a number **and how
    much it moves**.  Without the second one, a 1,2-point gap between two
    methods cannot be called a result.

    The aggregate keeps the mean of every scalar metric, the standard
    deviation of the accuracies, the full list of per-seed accuracies (so
    the report can quote the spread), and the artefacts of the *first* seed
    only — history, predictions and labels — because averaging a confusion
    matrix across seeds would mean something different from what the figure
    claims to show.

    Parameters
    ----------
    runs : List[dict]
        Metrics dicts from ``engine.evaluate()``, one per seed.

    Returns
    -------
    aggregate : dict
        Same keys as a single run, plus ``top1_std``, ``top5_std``,
        ``top1_runs`` and ``n_seeds``.
    """
    if not runs:
        return {}

    base = dict(runs[0])
    top1 = [r["top1_accuracy"] for r in runs]
    top5 = [r["top5_accuracy"] for r in runs]

    base["top1_accuracy"] = float(np.mean(top1))
    base["top5_accuracy"] = float(np.mean(top5))
    # ddof=1 (sample std) with a single seed would be NaN; report 0 instead,
    # which is what "we did not measure the spread" should look like on a
    # plot with error bars.
    base["top1_std"] = float(np.std(top1, ddof=1)) if len(top1) > 1 else 0.0
    base["top5_std"] = float(np.std(top5, ddof=1)) if len(top5) > 1 else 0.0
    base["top1_runs"] = [float(x) for x in top1]
    base["n_seeds"] = len(runs)

    for key in ("eval_time_seconds", "train_time_seconds", "peak_gpu_mb"):
        values = [r[key] for r in runs if r.get(key) is not None]
        if values:
            base[key] = float(np.mean(values))

    return base


# ============================================================================
# t-SNE feature extraction
# ============================================================================

@torch.no_grad()
def extract_features(
    model, dataloader, max_samples: int = 500,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract image features for t-SNE visualization.

    Works with any object that has ``get_image_features(images)`` — the
    frozen wrapper as well as any trained subclass.  Call it **before**
    releasing the model.
    """
    features_list: List[torch.Tensor] = []
    labels_list: List[torch.Tensor] = []
    collected = 0

    was_training = getattr(model, "training", False)
    if hasattr(model, "eval"):
        model.eval()

    for images, labels, _ in dataloader:
        feats = model.get_image_features(images)
        features_list.append(feats.detach().cpu())
        labels_list.append(labels)
        collected += images.shape[0]
        if collected >= max_samples:
            break

    if was_training and hasattr(model, "train"):
        model.train()

    feats_all = torch.cat(features_list)[:max_samples]
    labs_all = torch.cat(labels_list)[:max_samples]
    return feats_all.numpy(), labs_all.numpy()


# ============================================================================
# Plotting
# ============================================================================

# Methods in presentation order for the plots.
METHOD_ORDER = [
    "ZeroShot", "ZeroShot-Ensemble", "LinearProbe", "OptimalTransport",
    "CoOp", "CLIP-Adapter", "CoOp+Adapter", "Tip-Adapter", "Tip-Adapter-F",
    "LoRA",
]

# A distinct colour for each method, reused across all figures.
METHOD_COLORS = {
    "ZeroShot": "#7f8c8d",
    "ZeroShot-Ensemble": "#95a5a6",
    "LinearProbe": "#2ecc71",
    "OptimalTransport": "#e67e22",
    "CoOp": "#3498db",
    "CLIP-Adapter": "#e74c3c",
    "CoOp+Adapter": "#9b59b6",
    "Tip-Adapter": "#1abc9c",
    "Tip-Adapter-F": "#16a085",
    "LoRA": "#f39c12",
}

DATASET_MARKERS = {"eurosat": "o", "dtd": "s", "flowers102": "D"}


def _ordered_methods(methods: List[str]) -> List[str]:
    """Sort methods according to the canonical presentation order."""
    return [m for m in METHOD_ORDER if m in methods]


def _all_methods(results: Dict[str, Dict[str, Dict[str, Any]]]) -> List[str]:
    seen = set()
    for ds_results in results.values():
        seen.update(ds_results.keys())
    return _ordered_methods(list(seen))


def plot_main_comparison(
    results: Dict[str, Dict[str, Dict[str, Any]]],
    save_dir: Path,
) -> None:
    """
    Grouped bar chart: Top-1 accuracy per method, one group per dataset.

    Error bars are the standard deviation across seeds.  A method run on a
    single seed gets a zero-height bar, which reads correctly as "spread not
    measured" rather than as "spread is zero".
    """
    datasets = list(results.keys())
    methods = _all_methods(results)

    n_datasets = len(datasets)
    n_methods = len(methods)
    if n_methods == 0:
        return

    fig, ax = plt.subplots(figsize=(max(12, 2 * n_methods), 6))

    x = np.arange(n_methods)
    bar_width = 0.8 / n_datasets
    dataset_colors = ["#3498db", "#e74c3c", "#2ecc71"]

    for di, ds in enumerate(datasets):
        short = DATASET_REGISTRY[ds]["short_name"]
        accs, errs = [], []
        for m in methods:
            met = results[ds].get(m)
            accs.append(met["top1_accuracy"] if met else 0.0)
            errs.append(met.get("top1_std", 0.0) if met else 0.0)

        offset = (di - n_datasets / 2 + 0.5) * bar_width
        bars = ax.bar(
            x + offset, accs, bar_width,
            yerr=errs, capsize=2, error_kw={"linewidth": 1, "ecolor": "#333"},
            label=short, color=dataset_colors[di % len(dataset_colors)],
            edgecolor="white", linewidth=0.5,
        )
        for bar, acc, err in zip(bars, accs, errs):
            if acc > 0:
                ax.annotate(
                    f"{acc:.1f}",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height() + err),
                    textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=7, fontweight="bold",
                )

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=25, ha="right", fontsize=10)
    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=12)
    ax.set_title(
        "Unified Comparison — All Methods × All Datasets",
        fontsize=14, fontweight="bold", pad=15,
    )
    ax.legend(fontsize=10)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = save_dir / "unified_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved: {out}")


def plot_accuracy_vs_params(
    results: Dict[str, Dict[str, Dict[str, Any]]],
    save_dir: Path,
) -> None:
    """
    Scatter plot: Accuracy vs Trainable Parameters (log scale).

    This is THE figure of the project — it answers "how much accuracy does
    each parameter buy?".  Colour encodes the method, marker shape the
    dataset, and both get their own legend: reading the dataset off a text
    annotation next to every point, as this figure used to require, does not
    survive ten methods on three datasets.
    """
    fig, ax = plt.subplots(figsize=(11, 6.5))

    for ds_name, ds_results in results.items():
        marker = DATASET_MARKERS.get(ds_name, "o")
        for m in _ordered_methods(list(ds_results.keys())):
            met = ds_results[m]
            params = max(met["trainable_params"], 0.5)  # avoid log(0)
            acc = met["top1_accuracy"]
            err = met.get("top1_std", 0.0)
            color = METHOD_COLORS.get(m, "#333333")

            if err > 0:
                ax.errorbar(
                    params, acc, yerr=err, fmt="none",
                    ecolor=color, elinewidth=1, capsize=3, alpha=0.8, zorder=2,
                )
            ax.scatter(
                params, acc, c=color, marker=marker, s=110,
                edgecolors="white", linewidths=0.6, zorder=3,
            )

    # Two legends: one for the colours (methods), one for the shapes
    # (datasets).  Matplotlib keeps only the last one added unless the first
    # is re-attached by hand.
    method_handles = [
        plt.Line2D([], [], marker="o", linestyle="", markersize=8,
                   markerfacecolor=METHOD_COLORS[m], markeredgecolor="white",
                   label=m)
        for m in _all_methods(results)
    ]
    dataset_handles = [
        plt.Line2D([], [], marker=DATASET_MARKERS.get(ds, "o"), linestyle="",
                   markersize=8, markerfacecolor="#555555",
                   markeredgecolor="white",
                   label=DATASET_REGISTRY[ds]["short_name"])
        for ds in results
    ]
    # Both legends sit OUTSIDE the axes: with ten methods on three datasets
    # every corner of the plotting area holds points, and a legend box
    # anywhere inside covers some of them.
    method_legend = ax.legend(
        handles=method_handles, fontsize=8, title="Method",
        loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0,
    )
    ax.add_artist(method_legend)
    dataset_legend = ax.legend(
        handles=dataset_handles, fontsize=8, title="Dataset",
        loc="lower left", bbox_to_anchor=(1.02, 0.0), borderaxespad=0,
    )

    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("Trainable Parameters", fontsize=12)
    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=12)
    ax.set_title(
        "Accuracy vs. Trainable Parameters",
        fontsize=14, fontweight="bold", pad=15,
    )
    ax.grid(alpha=0.3)

    # No tight_layout() here: it lays out the axes without knowing about the
    # two legends parked outside them, and savefig's tight bounding box then
    # crops the longer method names. Naming the legends as extra artists is
    # what makes the saved box include them.
    out = save_dir / "accuracy_vs_params_unified.png"
    fig.savefig(
        out, dpi=150, bbox_inches="tight",
        bbox_extra_artists=(method_legend, dataset_legend),
    )
    plt.close(fig)
    print(f"[Plot] Saved: {out}")


def plot_training_cost(
    results: Dict[str, Dict[str, Dict[str, Any]]],
    save_dir: Path,
) -> None:
    """
    Training wall-clock time per method — the resource axis that actually
    discriminates.

    Peak GPU memory does not: every method in this project sits within a few
    tens of MB of the others, because the frozen 86M-parameter backbone
    dominates and none of the adaptation modules is large enough to matter.
    That is a finding, and it belongs in the report — but it means the
    "resource investment" the brief asks about has to be read on the clock,
    not on the memory gauge.
    """
    datasets = list(results.keys())
    methods = [
        m for m in _all_methods(results)
        if any(results[ds].get(m, {}).get("train_time_seconds")
               for ds in datasets)
    ]
    if not methods:
        print("[Plot] No trained methods with timings — 'training_cost' not written.")
        return

    fig, ax = plt.subplots(figsize=(max(9, 1.6 * len(methods)), 5.5))
    x = np.arange(len(methods))
    bar_width = 0.8 / max(len(datasets), 1)
    dataset_colors = ["#3498db", "#e74c3c", "#2ecc71"]

    for di, ds in enumerate(datasets):
        times = [
            results[ds].get(m, {}).get("train_time_seconds") or 0.0
            for m in methods
        ]
        offset = (di - len(datasets) / 2 + 0.5) * bar_width
        ax.bar(
            x + offset, times, bar_width,
            label=DATASET_REGISTRY[ds]["short_name"],
            color=dataset_colors[di % len(dataset_colors)],
            edgecolor="white", linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=25, ha="right", fontsize=10)
    ax.set_ylabel("Training time (s, mean over seeds)", fontsize=12)
    ax.set_title(
        "Training Cost at Matched Epoch Budget",
        fontsize=14, fontweight="bold", pad=15,
    )
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = save_dir / "training_cost.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved: {out}")


def plot_tsne(
    features_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
    class_names: List[str],
    save_dir: Path,
    dataset_name: str = "eurosat",
) -> None:
    """
    t-SNE of image features, one panel per method.

    The comparison the figure exists for is *frozen vs adapted*: the
    Zero-Shot panel shows what the backbone produces on its own, the others
    what each adaptation method makes of it.  Each panel is annotated with
    its **silhouette score** on the class labels, so the reader is not asked
    to judge cluster separation by eye — t-SNE layouts are not comparable
    across panels, silhouette scores computed in the original feature space
    are.
    """
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score

    n = len(features_dict)
    if n == 0:
        print("[Plot] No features collected — 't-SNE' not written.")
        return

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5.4), squeeze=False)
    n_classes = len(class_names)
    cmap = plt.get_cmap("tab10" if n_classes <= 10 else "tab20", n_classes)

    for idx, (method_name, (feats, labs)) in enumerate(features_dict.items()):
        ax = axes[0][idx]

        # Silhouette on the ORIGINAL features, not on the 2-D embedding:
        # t-SNE distances are not metric and two panels' layouts have no
        # common scale.
        try:
            sil = silhouette_score(feats, labs, metric="cosine")
            sil_txt = f"silhouette {sil:.3f}"
        except Exception:
            sil_txt = "silhouette n/a"

        perplexity = min(30, max(5, len(feats) // 4))
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        coords = tsne.fit_transform(feats)

        for c in range(n_classes):
            mask = labs == c
            if mask.sum() == 0:
                continue
            short_name = class_names[c]
            if len(short_name) > 15:
                short_name = short_name[:13] + "…"
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                c=[cmap(c)], s=12, alpha=0.7,
                label=short_name if idx == 0 else None,
            )

        ax.set_title(f"{method_name}\n{sil_txt}", fontsize=12, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

    # Shared legend on the right.
    if n_classes <= 20:
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(
            handles, labels, loc="center right",
            fontsize=7, markerscale=2,
            bbox_to_anchor=(1.02, 0.5),
        )

    short = DATASET_REGISTRY[dataset_name]["short_name"]
    fig.suptitle(
        f"t-SNE of Image Features — {short}",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 0.88 if n_classes <= 20 else 1.0, 0.95])
    out = save_dir / "tsne_features.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved: {out}")


# ============================================================================
# Console tables + JSON export
# ============================================================================

def print_results_table(
    results: Dict[str, Dict[str, Dict[str, Any]]],
    run_config: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Print two Markdown tables for direct paste into the report.

    **Two** tables, not one.  The previous version carried a single
    ``Params`` column filled inside the loop over datasets, so it survived
    only for the last one: with EuroSAT, DTD and Flowers102 in that order,
    every row showed Flowers102's parameter count — the Linear Probe read
    52.326 on the EuroSAT row where it is 5.130.  Accuracy varies by
    dataset *and* parameter count varies by dataset (the classifier and the
    cache both scale with the number of classes), so they need one table
    each.
    """
    datasets = list(results.keys())
    methods = _all_methods(results)
    ds_shorts = [DATASET_REGISTRY[ds]["short_name"] for ds in datasets]

    if run_config:
        shots = run_config.get("shots", 0)
        regime = f"{shots}-shot" if shots else "full-shot"
        print(f"\n  Protocol: {regime} | epochs={run_config.get('epochs')} | "
              f"seeds={run_config.get('seeds')} | "
              f"eval on the full test split")

    # ---------------- Table 1: accuracy ----------------
    header = f"| {'Method':<17} |"
    sep = f"|{'-' * 19}|"
    for short in ds_shorts:
        header += f" {short:>16} |"
        sep += f"{'':->18}|"

    print("\n" + "=" * len(header))
    print("  TOP-1 ACCURACY (%) — mean ± std over seeds")
    print("=" * len(header))
    print(header)
    print(sep)

    for m in methods:
        row = f"| {m:<17} |"
        for ds in datasets:
            met = results[ds].get(m)
            if met is None:
                row += f" {'—':>16} |"
            elif met.get("n_seeds", 1) > 1:
                row += f" {met['top1_accuracy']:>8.2f} ± {met['top1_std']:<5.2f} |"
            else:
                row += f" {met['top1_accuracy']:>10.2f}     |"
        print(row)
    print("=" * len(header))

    # ---------------- Table 2: trainable parameters ----------------
    print("\n" + "=" * len(header))
    print("  TRAINABLE PARAMETERS (the CLIP backbone is frozen throughout)")
    print("=" * len(header))
    print(header)
    print(sep)

    for m in methods:
        row = f"| {m:<17} |"
        for ds in datasets:
            met = results[ds].get(m)
            if met is None:
                row += f" {'—':>16} |"
            else:
                row += f" {met['trainable_params']:>16,} |"
        print(row)
    print("=" * len(header) + "\n")


def save_results_json(
    results: Dict[str, Dict[str, Dict[str, Any]]],
    path: Path,
    run_config: Optional[Dict[str, Any]] = None,
) -> None:
    """Save results to JSON, stripping non-serializable numpy arrays."""
    clean: Dict[str, Any] = {"_config": run_config or {}}
    for ds, ds_results in results.items():
        clean[ds] = {}
        for method, metrics in ds_results.items():
            clean[ds][method] = {
                k: v for k, v in metrics.items()
                if not isinstance(v, np.ndarray)
            }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean, indent=2, default=str), encoding="utf-8")
    print(f"[Results] Saved to {path}")


def save_predictions_npz(
    results: Dict[str, Dict[str, Dict[str, Any]]],
    save_dir: Path,
) -> None:
    """
    Persist the raw predictions and labels of every method.

    ``save_results_json`` drops them — they are numpy arrays and would bloat
    the file — but the confusion matrices and the per-class accuracy chart
    are built from exactly these two arrays.  Without them on disk, those
    figures can only be regenerated by re-running the whole grid on a GPU.
    With them, ``--plots-only`` redraws everything in seconds.
    """
    for ds, ds_results in results.items():
        payload = {}
        for method, metrics in ds_results.items():
            if "all_predictions" in metrics and "all_labels" in metrics:
                payload[f"{method}__pred"] = metrics["all_predictions"]
                payload[f"{method}__true"] = metrics["all_labels"]
        if payload:
            out = save_dir / f"predictions_{ds}.npz"
            np.savez_compressed(out, **payload)
            print(f"[Results] Saved: {out}")


def load_predictions_npz(
    results: Dict[str, Dict[str, Dict[str, Any]]],
    save_dir: Path,
) -> None:
    """Put predictions/labels back into ``results`` (used by --plots-only)."""
    for ds, ds_results in results.items():
        path = save_dir / f"predictions_{ds}.npz"
        if not path.exists():
            continue
        with np.load(path) as data:
            for method in ds_results:
                if f"{method}__pred" in data:
                    ds_results[method]["all_predictions"] = data[f"{method}__pred"]
                    ds_results[method]["all_labels"] = data[f"{method}__true"]


# ============================================================================
# Figures that need predictions, delegated to engine.py
# ============================================================================

def plot_per_dataset_diagnostics(
    results: Dict[str, Dict[str, Dict[str, Any]]],
    dataset_name: str,
    save_dir: Path,
) -> None:
    """
    Confusion matrices and per-class accuracy for one dataset.

    Delegated to ``engine.plot_confusion_matrices`` and
    ``engine.plot_per_class_metrics``, the same functions ``engine.py``'s
    own entry point uses — but fed the **full ten-method** result set rather
    than the four baselines, so the figures in ``plots/`` finally describe
    the same experiment as the tables next to them.
    """
    ds_results = results.get(dataset_name)
    if not ds_results:
        return
    usable = {
        m: met for m, met in ds_results.items()
        if met.get("all_predictions") is not None
    }
    if not usable:
        print("[Plot] No stored predictions — confusion matrices skipped.")
        return

    class_names = DATASET_REGISTRY[dataset_name]["class_names"]
    if len(class_names) > 20:
        # A 47×47 or 102×102 heatmap with a number printed in every cell,
        # times ten methods, is not a figure anybody can read.
        print(f"[Plot] {dataset_name}: {len(class_names)} classes — "
              "confusion matrices and per-class chart skipped (unreadable).")
        return

    engine.plot_confusion_matrices(
        usable, class_names=class_names, save_dir=str(save_dir),
    )
    engine.plot_per_class_metrics(
        usable, class_names=class_names, save_dir=str(save_dir),
    )


def plot_memory_from_results(
    results: Dict[str, Dict[str, Dict[str, Any]]],
    save_dir: Path,
) -> None:
    """
    "Memory Usage vs. Epochs" — one figure per dataset.

    A deliverable the brief asks for that never existed, for a reason worth
    remembering: ``engine.train()`` collected the per-epoch series all along,
    and this script used to delete it from ``history`` before saving. The
    plotting function existed too, and nothing called it.

    One figure per dataset because the curves are not comparable across
    datasets: a full-shot EuroSAT epoch is 338 batches and a 16-shot DTD one
    is 24, so the peaks are reached at different points for reasons that
    have nothing to do with the methods.
    """
    for ds_name, ds_results in results.items():
        histories = {
            m: met["history"] for m, met in ds_results.items()
            if isinstance(met.get("history"), dict)
            and met["history"].get("gpu_epoch_mb")
        }
        if not histories:
            continue
        out_dir = save_dir / "_memory_tmp"
        # engine.plot_memory_vs_epochs() writes a fixed file name and
        # announces it; swallow that message and print the final path
        # instead, otherwise the console advertises a temporary directory
        # that no longer exists by the time anyone reads the log.
        with contextlib.redirect_stdout(io.StringIO()):
            engine.plot_memory_vs_epochs(
                histories, save_dir=str(out_dir), per_epoch=True,
            )
        produced = out_dir / "memory_vs_epochs.png"
        if produced.exists():
            final = save_dir / f"memory_vs_epochs_{ds_name}.png"
            os.replace(produced, final)
            print(f"[Plot] Saved: {final}")
    tmp = save_dir / "_memory_tmp"
    if tmp.exists() and not any(tmp.iterdir()):
        tmp.rmdir()


# ============================================================================
# Main
# ============================================================================

def run_dataset(
    dataset_name: str,
    device: str,
    batch_size: int,
    num_workers: int,
    skip_slow: bool,
    do_tsne: bool,
    epochs: int,
    seeds: List[int],
    shots: int,
    adapter_lr_scale: float,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """
    Run all methods on a single dataset. Returns ``(results, tsne_features)``.

    Training-free methods (Zero-Shot, Ensemble, Optimal Transport) are
    deterministic and run once.  Everything with trainable parameters runs
    once per seed and is aggregated by ``aggregate_runs``.  The Linear Probe
    is in between: sklearn's L-BFGS is deterministic given the features, but
    in few-shot mode the *support set* depends on the seed, so it repeats
    too.
    """
    short = DATASET_REGISTRY[dataset_name]["short_name"]
    print(f"\n{'#' * 70}")
    print(f"#  DATASET: {short}")
    print(f"{'#' * 70}")

    results: Dict[str, Dict[str, Any]] = {}
    tsne_feats: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    # ------------------------------------------------------------------
    # Data.  In full-shot mode the loaders do not depend on the seed, so we
    # build them once; in few-shot mode the support set is re-drawn per seed
    # inside the loop below.
    # ------------------------------------------------------------------
    train_loader, test_loader, train_dataset, info = load_data(
        dataset_name, batch_size=batch_size, num_workers=num_workers,
        shots=shots, seed=seeds[0],
    )
    regime = f"{shots}-shot" if shots else "full-shot"
    print(f"  Classes: {info['n_classes']}  |  Train: {info['n_train_used']} "
          f"({regime}, of {info['n_train_full']})  |  Test: {info['n_test']}")

    def data_for(seed: int):
        """Loaders for one seed (only the support set changes)."""
        if shots <= 0 or seed == seeds[0]:
            return train_loader, test_loader, train_dataset, info
        tl, te, td, nfo = load_data(
            dataset_name, batch_size=batch_size, num_workers=num_workers,
            shots=shots, seed=seed,
        )
        return tl, te, td, nfo

    # ------------------------------------------------------------------
    # Training-free baselines (share one frozen backbone to save memory).
    # ------------------------------------------------------------------
    print(f"\n--- Training-free baselines ({short}) ---")
    clip_wrapper = BaseCLIPWrapper(device=device)

    try:
        t0 = time.time()
        results["ZeroShot"] = run_zero_shot(clip_wrapper, test_loader, info)
        print(f"  ✓ ZeroShot ({time.time() - t0:.1f}s)")
        if do_tsne:
            tsne_feats["Zero-Shot (frozen)"] = extract_features(
                clip_wrapper, test_loader,
            )
    except Exception as e:
        print(f"  ✗ ZeroShot failed: {e}")

    try:
        t0 = time.time()
        results["ZeroShot-Ensemble"] = run_zero_shot_ensemble(
            clip_wrapper, test_loader, info,
        )
        print(f"  ✓ ZeroShot-Ensemble ({time.time() - t0:.1f}s)")
    except Exception as e:
        print(f"  ✗ ZeroShot-Ensemble failed: {e}")

    if skip_slow and info["n_classes"] > 10:
        print(f"  ⊘ OptimalTransport skipped "
              f"({info['n_classes']} classes, --skip-slow)")
    else:
        try:
            t0 = time.time()
            results["OptimalTransport"] = run_optimal_transport(
                clip_wrapper, test_loader, info,
            )
            print(f"  ✓ OptimalTransport ({time.time() - t0:.1f}s)")
        except Exception as e:
            print(f"  ✗ OptimalTransport failed: {e}")

    # Linear Probe: repeated per seed because the support set moves.
    lp_runs: List[Dict[str, Any]] = []
    for seed in seeds:
        tl, te, _td, nfo = data_for(seed)
        try:
            t0 = time.time()
            lp_runs.append(run_linear_probe(clip_wrapper, tl, te))
            print(f"  ✓ LinearProbe seed={seed} ({time.time() - t0:.1f}s)")
        except Exception as e:
            print(f"  ✗ LinearProbe seed={seed} failed: {e}")
        if shots <= 0:
            break  # deterministic without a support-set draw
    if lp_runs:
        results["LinearProbe"] = aggregate_runs(lp_runs)

    _release(clip_wrapper)

    # ------------------------------------------------------------------
    # Trainable methods (each loads its own CLIP copy).
    # ------------------------------------------------------------------
    print(f"\n--- Trainable methods ({short}) "
          f"| {epochs} epochs | seeds {seeds} ---")

    trained: Dict[str, List[Dict[str, Any]]] = {}

    def record(name: str, metrics: Dict[str, Any], seed: int) -> None:
        trained.setdefault(name, []).append(metrics)

    for seed in seeds:
        tl, te, td, nfo = data_for(seed)
        first = seed == seeds[0]
        sink = tsne_feats if (do_tsne and first) else None

        for name, fn in (
            ("CoOp", lambda: run_coop(device, tl, te, nfo, epochs, seed, sink)),
            ("CLIP-Adapter", lambda: run_clip_adapter(
                device, tl, te, nfo, epochs, seed, tsne_sink=sink)),
            ("CoOp+Adapter", lambda: run_joint(
                device, tl, te, nfo, epochs, seed,
                adapter_lr_scale=adapter_lr_scale, tsne_sink=sink)),
            ("LoRA", lambda: run_lora(
                device, tl, te, nfo, epochs, seed, tsne_sink=sink)),
        ):
            try:
                t0 = time.time()
                record(name, fn(), seed)
                print(f"  ✓ {name} seed={seed} ({time.time() - t0:.1f}s)")
            except Exception as e:
                print(f"  ✗ {name} seed={seed} failed: {e}")

        # Tip-Adapter and Tip-Adapter-F come out of a single model.
        try:
            t0 = time.time()
            tip_met, tipf_met = run_tip_adapter(
                device, td, tl, te, nfo, epochs, seed,
                cache_shots=shots if shots > 0 else 16,
            )
            record("Tip-Adapter", tip_met, seed)
            record("Tip-Adapter-F", tipf_met, seed)
            print(f"  ✓ Tip-Adapter + Tip-Adapter-F seed={seed} "
                  f"({time.time() - t0:.1f}s)")
        except Exception as e:
            print(f"  ✗ Tip-Adapter seed={seed} failed: {e}")

    for name, runs in trained.items():
        results[name] = aggregate_runs(runs)

    return results, tsne_feats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified comparison of ALL methods on ALL datasets.",
    )
    parser.add_argument(
        "--dataset", nargs="+",
        choices=list(DATASET_REGISTRY.keys()),
        default=list(DATASET_REGISTRY.keys()),
        help="Datasets to evaluate (default: all).",
    )
    parser.add_argument(
        "--shots", type=int, default=16,
        help="Labelled images per class for every trainable method. "
             "0 = use the whole training split (full-shot). Default: 16.",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[0],
        help="Seeds for the trainable methods; results are reported as "
             "mean ± std over them. Default: a single seed.",
    )
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="Epoch budget, the SAME for every trainable method. Default: 10.",
    )
    parser.add_argument(
        "--adapter-lr-scale", type=float, default=1.0,
        help="LR multiplier for the adapter half of the joint CoOp+Adapter "
             "model, relative to the context vectors. Default: 1.0.",
    )
    parser.add_argument(
        "--skip-slow", action="store_true",
        help="Skip Optimal Transport on datasets with >10 classes.",
    )
    parser.add_argument("--no-tsne", action="store_true",
                        help="Skip t-SNE visualization.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--out", type=str, default=None,
        help="Result JSON file name inside plots/ "
             "(default: unified_results.json).",
    )
    parser.add_argument(
        "--plots-only", action="store_true",
        help="Redraw every figure from the results already on disk. "
             "No GPU, no training.",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS_DIR / (args.out or "unified_results.json")

    # ------------------------------------------------------------------
    # --plots-only: no model is ever constructed.
    # ------------------------------------------------------------------
    if args.plots_only:
        if not out_json.exists():
            raise SystemExit(f"{out_json} not found — run the grid first.")
        raw = json.loads(out_json.read_text(encoding="utf-8"))
        run_config = raw.pop("_config", {})
        all_results = {k: v for k, v in raw.items() if k in DATASET_REGISTRY}
        load_predictions_npz(all_results, RESULTS_DIR)

        print_results_table(all_results, run_config)
        plot_main_comparison(all_results, RESULTS_DIR)
        plot_accuracy_vs_params(all_results, RESULTS_DIR)
        plot_training_cost(all_results, RESULTS_DIR)
        plot_memory_from_results(all_results, RESULTS_DIR)
        first_ds = next(iter(all_results))
        plot_per_dataset_diagnostics(all_results, first_ds, RESULTS_DIR)
        print("[run_all] Figures regenerated from disk. "
              "(t-SNE needs the trained models, so it is not redrawn here.)")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[run_all] Device: {device}")
    if device == "cpu":
        print("[run_all] ⚠ Running on CPU — this will be very slow.")

    regime = f"{args.shots}-shot" if args.shots else "full-shot"
    print(f"[run_all] Protocol: {regime}, {args.epochs} epochs for every "
          f"trainable method, seeds {args.seeds}")
    print("[run_all] Note: val_loader is the test set — model selection is "
          "not held out. Declare this in the report.")

    run_config = {
        "shots": args.shots,
        "epochs": args.epochs,
        "seeds": args.seeds,
        "batch_size": args.batch_size,
        "adapter_lr_scale": args.adapter_lr_scale,
        "recipes": {k: v for k, v in RECIPES.items()},
        "backbone": "ViT-B-32 / laion2b_s34b_b79k (NOT the OpenAI weights)",
        "validation_split": "none — val_loader is the test set",
    }

    do_tsne = not args.no_tsne
    all_results: Dict[str, Dict[str, Dict[str, Any]]] = {}
    all_tsne: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    total_start = time.time()

    for ds_name in args.dataset:
        ds_results, ds_tsne = run_dataset(
            dataset_name=ds_name,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            skip_slow=args.skip_slow,
            do_tsne=do_tsne and ds_name == args.dataset[0],
            epochs=args.epochs,
            seeds=args.seeds,
            shots=args.shots,
            adapter_lr_scale=args.adapter_lr_scale,
        )
        all_results[ds_name] = ds_results
        if ds_tsne:
            all_tsne.update(ds_tsne)

    # ------------------------------------------------------------------
    # Output.  Every figure in plots/ comes out of this one block, so the
    # directory can never again hold two generations that disagree.
    # ------------------------------------------------------------------
    print_results_table(all_results, run_config)
    save_results_json(all_results, out_json, run_config)
    save_predictions_npz(all_results, RESULTS_DIR)

    plot_main_comparison(all_results, RESULTS_DIR)
    plot_accuracy_vs_params(all_results, RESULTS_DIR)
    plot_training_cost(all_results, RESULTS_DIR)
    plot_memory_from_results(all_results, RESULTS_DIR)
    plot_per_dataset_diagnostics(all_results, args.dataset[0], RESULTS_DIR)

    if all_tsne:
        first_ds = args.dataset[0]
        class_names = DATASET_REGISTRY[first_ds]["class_names"]
        plot_tsne(all_tsne, class_names, RESULTS_DIR, first_ds)

    total_time = time.time() - total_start
    print(f"\n[run_all] Total time: {total_time / 60:.1f} minutes")
    print("[run_all] Done.")


if __name__ == "__main__":
    main()
