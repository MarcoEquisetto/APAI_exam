"""
run_all.py — Confronto unificato di tutti i metodi su tutti i dataset
=====================================================================

Allena e valuta ogni metodo sullo stesso split, per ogni dataset, e produce
la tabella comparativa finale, i plot unificati e una t-SNE visualization.

Usage
-----
::

    # Tutto (EuroSAT + DTD + Flowers102, tutti i metodi)
    python src/run_all.py

    # Solo un dataset
    python src/run_all.py --dataset eurosat

    # Due dataset
    python src/run_all.py --dataset eurosat dtd

    # Salta Optimal Transport (molto lento su >10 classi)
    python src/run_all.py --skip-slow

    # Senza t-SNE
    python src/run_all.py --no-tsne

    # Batch size e num_workers personalizzati
    python src/run_all.py --batch-size 32 --num-workers 0
"""

import sys
import os
import json
import time
import argparse
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
) -> Tuple[Any, Any, Any, Dict[str, Any]]:
    """
    Load train/test loaders, the raw train dataset, and dataset metadata.

    Returns
    -------
    train_loader, test_loader, train_dataset, info
    """
    cfg = DATASET_REGISTRY[dataset_name]

    train_loader, test_loader = cfg["get_loaders"](
        batch_size=batch_size, num_workers=num_workers, download=True,
    )
    train_dataset = cfg["train_dataset_cls"](split="train", download=True)

    info = {
        "class_names": cfg["class_names"],
        "prompt_template": cfg["prompt_template"],
        "ensemble_templates": cfg["ensemble_templates"],
        "short_name": cfg["short_name"],
        "n_classes": len(cfg["class_names"]),
        "n_train": len(train_dataset),
        "n_test": len(test_loader.dataset),
    }
    return train_loader, test_loader, train_dataset, info


# ============================================================================
# Method runners
# ============================================================================
# Each function trains (if needed) and evaluates a single method, returning
# the metrics dict from engine.evaluate().  Models are deleted after
# evaluation to free GPU memory for the next method.
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
    """Linear Probe (sklearn Logistic Regression on frozen features)."""
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
    epochs: int = 10,
    lr: float = 2e-3,
) -> Dict[str, Any]:
    """CoOp: Context Optimization (M=16, unified)."""
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
        lr=lr,
        optimizer_type="sgd",
        weight_decay=5e-4,
        momentum=0.9,
        scheduler_type="cosine",
        warmup_epochs=1,
        use_wandb=False,
        model_name="CoOp",
    )
    metrics = engine.evaluate(
        model, test_loader, model_name="CoOp", use_wandb=False,
    )
    metrics["history"] = {
        k: v for k, v in history.items()
        if k not in ("gpu_mb", "gpu_epoch_mb")
    }
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def run_clip_adapter(
    device: str,
    train_loader,
    test_loader,
    info: dict,
    epochs: int = 5,
    lr: float = 1e-3,
    reduction_ratio: int = 4,
    alpha: float = 0.2,
) -> Dict[str, Any]:
    """CLIP-Adapter: Vision bottleneck MLP (r=4, alpha=0.2)."""
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
        lr=lr,
        optimizer_type="adamw",
        weight_decay=5e-4,
        scheduler_type="cosine",
        warmup_epochs=1,
        use_wandb=False,
        model_name="CLIP-Adapter",
    )
    metrics = engine.evaluate(
        model, test_loader, model_name="CLIP-Adapter", use_wandb=False,
    )
    metrics["history"] = {
        k: v for k, v in history.items()
        if k not in ("gpu_mb", "gpu_epoch_mb")
    }
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def run_joint(
    device: str,
    train_loader,
    test_loader,
    info: dict,
    epochs: int = 10,
    lr: float = 2e-3,
) -> Dict[str, Any]:
    """Joint CoOp + CLIP-Adapter: text + vision adaptation."""
    model = CoOpAdapterModel(
        device=device,
        class_names=info["class_names"],
        n_ctx=16,
        class_specific=False,
        reduction_ratio=4,
        alpha=0.2,
    )
    history = engine.train(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        class_names=info["class_names"],
        epochs=epochs,
        lr=lr,
        optimizer_type="sgd",
        weight_decay=5e-4,
        momentum=0.9,
        scheduler_type="cosine",
        warmup_epochs=1,
        use_wandb=False,
        model_name="CoOp+Adapter",
    )
    metrics = engine.evaluate(
        model, test_loader, model_name="CoOp+Adapter", use_wandb=False,
    )
    metrics["history"] = {
        k: v for k, v in history.items()
        if k not in ("gpu_mb", "gpu_epoch_mb")
    }
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def run_tip_adapter(
    device: str,
    train_loader,
    test_loader,
    info: dict,
    finetune_epochs: int = 20,
    finetune_lr: float = 1e-3,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Tip-Adapter (training-free cache) + Tip-Adapter-F (fine-tuned cache).

    Returns two metrics dicts: (tip_metrics, tipf_metrics).
    """
    model = TipAdapterModel(
        device=device,
        class_names=info["class_names"],
        prompt_template=info["prompt_template"],
    )
    # Build cache from the full training set.
    model.build_cache(train_loader, num_shots=99999)

    tip_metrics = engine.evaluate(
        model, test_loader, model_name="Tip-Adapter", use_wandb=False,
    )

    # Fine-tune the cache keys (Tip-Adapter-F).
    model.finetune_cache(
        train_loader, epochs=finetune_epochs, lr=finetune_lr,
    )
    tipf_metrics = engine.evaluate(
        model, test_loader, model_name="Tip-Adapter-F", use_wandb=False,
    )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return tip_metrics, tipf_metrics


def run_lora(
    device: str,
    train_loader,
    test_loader,
    info: dict,
    epochs: int = 10,
    lr: float = 1e-4,
    r: int = 4,
) -> Dict[str, Any]:
    """Vision LoRA: Low-Rank Adaptation of the ViT MLP layers."""
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
        lr=lr,
        optimizer_type="adamw",
        weight_decay=5e-4,
        scheduler_type="cosine",
        warmup_epochs=1,
        use_wandb=False,
        model_name="LoRA",
    )
    metrics = engine.evaluate(
        model, test_loader, model_name="LoRA", use_wandb=False,
    )
    metrics["history"] = {
        k: v for k, v in history.items()
        if k not in ("gpu_mb", "gpu_epoch_mb")
    }
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


# ============================================================================
# t-SNE feature extraction
# ============================================================================

@torch.no_grad()
def extract_features(
    model, dataloader, max_samples: int = 500,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract image features for t-SNE visualization.

    Works with any object that has ``get_image_features(images)``.
    """
    features_list: List[torch.Tensor] = []
    labels_list: List[torch.Tensor] = []
    collected = 0

    for images, labels, _ in dataloader:
        feats = model.get_image_features(images)
        features_list.append(feats.cpu())
        labels_list.append(labels)
        collected += images.shape[0]
        if collected >= max_samples:
            break

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


def _ordered_methods(methods: List[str]) -> List[str]:
    """Sort methods according to the canonical presentation order."""
    return [m for m in METHOD_ORDER if m in methods]


def plot_main_comparison(
    results: Dict[str, Dict[str, Dict[str, Any]]],
    save_dir: Path,
) -> None:
    """
    Grouped bar chart: Top-1 accuracy per method, one group per dataset.
    """
    datasets = list(results.keys())
    all_methods = set()
    for ds in datasets:
        all_methods.update(results[ds].keys())
    methods = _ordered_methods(list(all_methods))

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
        accs = []
        for m in methods:
            if m in results[ds]:
                accs.append(results[ds][m]["top1_accuracy"])
            else:
                accs.append(0)
        offset = (di - n_datasets / 2 + 0.5) * bar_width
        bars = ax.bar(
            x + offset, accs, bar_width,
            label=short, color=dataset_colors[di % len(dataset_colors)],
            edgecolor="white", linewidth=0.5,
        )
        for bar, acc in zip(bars, accs):
            if acc > 0:
                ax.annotate(
                    f"{acc:.1f}",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
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

    This is THE figure for the paper — it answers the project's central
    question: "how much accuracy does each parameter buy?"
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    markers = ["o", "s", "D", "^", "v", "<", ">", "p", "*", "h"]

    dataset_markers = {"eurosat": "o", "dtd": "s", "flowers102": "D"}
    dataset_labels_added = set()

    for ds_name, ds_results in results.items():
        short = DATASET_REGISTRY[ds_name]["short_name"]
        marker = dataset_markers.get(ds_name, "o")

        methods = _ordered_methods(list(ds_results.keys()))
        for m in methods:
            met = ds_results[m]
            params = max(met["trainable_params"], 0.5)  # avoid log(0)
            acc = met["top1_accuracy"]
            color = METHOD_COLORS.get(m, "#333333")

            # Only add legend entry once per method.
            label = m if ds_name == list(results.keys())[0] else None
            ax.scatter(
                params, acc, c=color, marker=marker, s=100,
                edgecolors="white", linewidths=0.5, label=label, zorder=3,
            )
            ax.annotate(
                f"{short}" if len(results) > 1 else "",
                (params, acc), textcoords="offset points",
                xytext=(6, -3), fontsize=6, color="gray",
            )

    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("Trainable Parameters", fontsize=12)
    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=12)
    ax.set_title(
        "Accuracy vs. Trainable Parameters",
        fontsize=14, fontweight="bold", pad=15,
    )

    # Deduplicate legend.
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out = save_dir / "accuracy_vs_params_unified.png"
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
    t-SNE visualization of image features from different methods.

    Parameters
    ----------
    features_dict : dict
        Mapping from method name to (features, labels) arrays.
    class_names : list
        Class names for the legend.
    """
    from sklearn.manifold import TSNE

    n = len(features_dict)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)
    n_classes = len(class_names)
    cmap = plt.cm.get_cmap("tab10" if n_classes <= 10 else "tab20", n_classes)

    for idx, (method_name, (feats, labs)) in enumerate(features_dict.items()):
        ax = axes[0][idx]

        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
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

        ax.set_title(method_name, fontsize=12, fontweight="bold")
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
# Console table + JSON export
# ============================================================================

def print_results_table(
    results: Dict[str, Dict[str, Dict[str, Any]]],
) -> None:
    """Print a Markdown-formatted table for direct paste into the report."""
    datasets = list(results.keys())
    all_methods = set()
    for ds in datasets:
        all_methods.update(results[ds].keys())
    methods = _ordered_methods(list(all_methods))

    ds_shorts = [DATASET_REGISTRY[ds]["short_name"] for ds in datasets]

    # Header.
    header = "| Method           |"
    sep = "|------------------|"
    for short in ds_shorts:
        header += f" {short:>10} |"
        sep += f"{'':->12}|"
    header += "    Params |"
    sep += "-----------|"

    print("\n" + "=" * len(header))
    print("  UNIFIED RESULTS TABLE")
    print("=" * len(header))
    print(header)
    print(sep)

    for m in methods:
        row = f"| {m:<16} |"
        params = "—"
        for ds in datasets:
            if m in results[ds]:
                acc = results[ds][m]["top1_accuracy"]
                row += f" {acc:>9.2f}% |"
                p = results[ds][m]["trainable_params"]
                params = f"{p:>9,}" if p > 0 else "        0"
            else:
                row += f" {'—':>10} |"
        row += f" {params} |"
        print(row)

    print("=" * len(header) + "\n")


def save_results_json(
    results: Dict[str, Dict[str, Dict[str, Any]]],
    path: Path,
) -> None:
    """Save results to JSON, stripping non-serializable numpy arrays."""
    clean = {}
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
    coop_epochs: int,
    adapter_epochs: int,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """
    Run all methods on a single dataset. Returns (results, tsne_features).
    """
    short = DATASET_REGISTRY[dataset_name]["short_name"]
    print(f"\n{'#' * 70}")
    print(f"#  DATASET: {short}")
    print(f"{'#' * 70}")

    train_loader, test_loader, train_dataset, info = load_data(
        dataset_name, batch_size=batch_size, num_workers=num_workers,
    )
    print(f"  Classes: {info['n_classes']}  |  "
          f"Train: {info['n_train']}  |  Test: {info['n_test']}")

    results: Dict[str, Dict[str, Any]] = {}
    tsne_feats: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    # ------------------------------------------------------------------
    # Baselines (share one frozen backbone to save memory).
    # ------------------------------------------------------------------
    print(f"\n--- Baselines ({short}) ---")
    clip_wrapper = BaseCLIPWrapper(device=device)

    # Zero-Shot
    try:
        t0 = time.time()
        results["ZeroShot"] = run_zero_shot(clip_wrapper, test_loader, info)
        print(f"  ✓ ZeroShot ({time.time() - t0:.1f}s)")
        if do_tsne:
            tsne_feats["Zero-Shot"] = extract_features(
                clip_wrapper, test_loader,
            )
    except Exception as e:
        print(f"  ✗ ZeroShot failed: {e}")

    # Zero-Shot Ensemble
    try:
        t0 = time.time()
        results["ZeroShot-Ensemble"] = run_zero_shot_ensemble(
            clip_wrapper, test_loader, info,
        )
        print(f"  ✓ ZeroShot-Ensemble ({time.time() - t0:.1f}s)")
    except Exception as e:
        print(f"  ✗ ZeroShot-Ensemble failed: {e}")

    # Linear Probe
    try:
        t0 = time.time()
        results["LinearProbe"] = run_linear_probe(
            clip_wrapper, train_loader, test_loader,
        )
        print(f"  ✓ LinearProbe ({time.time() - t0:.1f}s)")
    except Exception as e:
        print(f"  ✗ LinearProbe failed: {e}")

    # Optimal Transport (very slow on many classes)
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

    # Free the shared backbone before loading method-specific models.
    del clip_wrapper
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Trainable methods (each loads its own CLIP copy).
    # ------------------------------------------------------------------
    print(f"\n--- Trainable methods ({short}) ---")

    # CoOp
    try:
        t0 = time.time()
        results["CoOp"] = run_coop(
            device, train_loader, test_loader, info, epochs=coop_epochs,
        )
        print(f"  ✓ CoOp ({time.time() - t0:.1f}s)")
    except Exception as e:
        print(f"  ✗ CoOp failed: {e}")

    # CLIP-Adapter
    try:
        t0 = time.time()
        results["CLIP-Adapter"] = run_clip_adapter(
            device, train_loader, test_loader, info, epochs=adapter_epochs,
        )
        print(f"  ✓ CLIP-Adapter ({time.time() - t0:.1f}s)")
    except Exception as e:
        print(f"  ✗ CLIP-Adapter failed: {e}")

    # Collect Adapter features for t-SNE before deleting.
    # We rebuild a quick model just for feature extraction.
    if do_tsne and "CLIP-Adapter" in results:
        try:
            tmp = CLIPAdapterModel(
                device=device, reduction_ratio=4, alpha=0.2,
                class_names=info["class_names"],
                prompt_template=info["prompt_template"],
            )
            # Load the same adapter state by re-training with 0 epochs...
            # Actually, for t-SNE we just want to show the adapted features.
            # Since the model was deleted, let's skip adapter t-SNE for now
            # and only show frozen vs LoRA if available.
            del tmp
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # Joint CoOp + Adapter
    try:
        t0 = time.time()
        results["CoOp+Adapter"] = run_joint(
            device, train_loader, test_loader, info,
            epochs=coop_epochs,
        )
        print(f"  ✓ CoOp+Adapter ({time.time() - t0:.1f}s)")
    except Exception as e:
        print(f"  ✗ CoOp+Adapter failed: {e}")

    # Tip-Adapter + Tip-Adapter-F
    try:
        t0 = time.time()
        tip_met, tipf_met = run_tip_adapter(
            device, train_loader, test_loader, info,
        )
        results["Tip-Adapter"] = tip_met
        results["Tip-Adapter-F"] = tipf_met
        print(f"  ✓ Tip-Adapter + Tip-Adapter-F ({time.time() - t0:.1f}s)")
    except Exception as e:
        print(f"  ✗ Tip-Adapter failed: {e}")

    # LoRA
    try:
        t0 = time.time()
        results["LoRA"] = run_lora(
            device, train_loader, test_loader, info, epochs=adapter_epochs,
        )
        print(f"  ✓ LoRA ({time.time() - t0:.1f}s)")
    except Exception as e:
        print(f"  ✗ LoRA failed: {e}")

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
        "--skip-slow", action="store_true",
        help="Skip Optimal Transport on datasets with >10 classes.",
    )
    parser.add_argument("--no-tsne", action="store_true",
                        help="Skip t-SNE visualization.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--coop-epochs", type=int, default=10,
                        help="Training epochs for CoOp and Joint (default: 10).")
    parser.add_argument("--adapter-epochs", type=int, default=5,
                        help="Training epochs for CLIP-Adapter and LoRA (default: 5).")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[run_all] Device: {device}")
    if device == "cpu":
        print("[run_all] ⚠ Running on CPU — this will be very slow.")

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
            coop_epochs=args.coop_epochs,
            adapter_epochs=args.adapter_epochs,
        )
        all_results[ds_name] = ds_results
        if ds_tsne:
            all_tsne.update(ds_tsne)

    # ------------------------------------------------------------------
    # Output.
    # ------------------------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print_results_table(all_results)
    save_results_json(all_results, RESULTS_DIR / "unified_results.json")

    plot_main_comparison(all_results, RESULTS_DIR)
    plot_accuracy_vs_params(all_results, RESULTS_DIR)

    if all_tsne:
        first_ds = args.dataset[0]
        class_names = DATASET_REGISTRY[first_ds]["class_names"]
        plot_tsne(all_tsne, class_names, RESULTS_DIR, first_ds)

    total_time = time.time() - total_start
    print(f"\n[run_all] Total time: {total_time / 60:.1f} minutes")
    print("[run_all] Done.")


if __name__ == "__main__":
    main()
