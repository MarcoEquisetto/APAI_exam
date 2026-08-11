"""
engine.py — Training Loop, Evaluation Loop & W&B Logging
==========================================================

This module provides the unified infrastructure that ALL models in the
project share:

  1. ``evaluate()``  — model-agnostic evaluation (Top-1, Top-5, GPU memory).
  2. ``train()``     — generic training loop for Carlo's CoOp and Marco's
                       CLIP-Adapter (any BaseCLIPWrapper subclass with
                       trainable parameters).
  3. ``run_all_evaluations()`` — convenience: runs all baselines at once.
  4. ``plot_comparative_results()`` — generates Accuracy vs. Params,
                                      Memory Usage, and combined plots.

**Integration contract for Carlo & Marco:**

    Carlo subclasses ``BaseCLIPWrapper`` and overrides ``get_text_features()``.
    Marco subclasses ``BaseCLIPWrapper`` and overrides ``get_image_features()``.
    Both call ``engine.train(model, train_loader, ...)`` to optimize their
    trainable parameters, then ``engine.evaluate(model_or_wrapper, ...)``
    for final metrics.
"""

import ssl
import certifi
import os

# Fix SSL certificate error on Windows: the Windows Certificate Store
# contains a malformed certificate that crashes OpenSSL. We monkey-patch
# ssl.create_default_context to use certifi's CA bundle instead.
def _create_ssl_context(purpose=ssl.Purpose.SERVER_AUTH, *, cafile=None, capath=None, cadata=None):
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.load_verify_locations(cafile=certifi.where())
    return ctx

ssl.create_default_context = _create_ssl_context
ssl._create_default_https_context = _create_ssl_context

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import wandb
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from typing import Dict, Any, Optional, List

# Use non-interactive backend so plots can be saved without a display.
matplotlib.use("Agg")


# ============================================================================
# Utility functions
# ============================================================================

def count_trainable_parameters(model: Any) -> int:
    """
    Count the number of trainable parameters in a model.

    Works with:
      - nn.Module subclasses (BaseCLIPWrapper and adapters):
        counts parameters with requires_grad=True.
      - LinearProbeCLIP (scikit-learn based):
        counts the number of coefficients in the logistic regression.
      - OptimalTransportCLIP:
        returns 0 (no trainable parameters — it's a distance-based method).
    """
    # Case 1: nn.Module (BaseCLIPWrapper, future adapters)
    if hasattr(model, "count_trainable_params"):
        return model.count_trainable_params()

    # Case 2: LinearProbeCLIP (has a scikit-learn classifier inside)
    if hasattr(model, "classifier") and hasattr(model.classifier, "coef_"):
        n_weights = model.classifier.coef_.size
        n_biases = model.classifier.intercept_.size
        return n_weights + n_biases

    # Case 3: OptimalTransportCLIP or any other parameter-free method
    return 0


def _get_gpu_memory_mb() -> Dict[str, float]:
    """
    Query current GPU memory usage.

    Returns a dict with:
      - allocated_mb: memory currently used by tensors
      - reserved_mb:  memory reserved by the caching allocator
      - max_allocated_mb: peak memory usage since last reset

    Returns zeros if CUDA is not available.
    """
    if not torch.cuda.is_available():
        return {
            "gpu/allocated_mb": 0.0,
            "gpu/reserved_mb": 0.0,
            "gpu/max_allocated_mb": 0.0,
        }

    return {
        "gpu/allocated_mb": torch.cuda.memory_allocated() / (1024 ** 2),
        "gpu/reserved_mb": torch.cuda.memory_reserved() / (1024 ** 2),
        "gpu/max_allocated_mb": torch.cuda.max_memory_allocated() / (1024 ** 2),
    }


# ============================================================================
# Evaluation Loop
# ============================================================================

@torch.no_grad()
def evaluate(
    model: Any,
    dataloader: DataLoader,
    model_name: str = "model",
    use_wandb: bool = True,
    wandb_project: str = "clip-eurosat",
    wandb_config: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Evaluate a model on a dataset and log results to wandb.

    This function is model-agnostic: it calls ``model.predict(images)``
    and expects back a tuple of ``(predictions, scores)``.  The scores
    can be similarities (higher = better, for ZeroShot) or distances
    (lower = better, for OT) — we handle both cases.

    Parameters
    ----------
    model : Any
        Must have a ``.predict(images)`` method that returns
        ``(predictions: Tensor, scores: Tensor)``.
        - predictions: (B,) int tensor of predicted class indices.
        - scores: (B, num_classes) float tensor.
    dataloader : DataLoader
        Test set dataloader from dataset.py.
    model_name : str
        Human-readable name for logging (e.g., "ZeroShot", "LinearProbe").
    use_wandb : bool
        Whether to log metrics to Weights & Biases.
    wandb_project : str
        W&B project name.
    wandb_config : Optional[Dict]
        Extra configuration to log to W&B (e.g., hyperparameters).

    Returns
    -------
    metrics : Dict[str, Any]
        Dictionary containing:
        - "top1_accuracy": float
        - "top5_accuracy": float
        - "trainable_params": int
        - "total_samples": int
        - "eval_time_seconds": float
        - "gpu/*": GPU memory metrics
    """
    # ----------------------------------------------------------------
    # Initialize wandb run (if enabled).
    # ----------------------------------------------------------------
    if use_wandb:
        config = {"model_name": model_name}
        if wandb_config:
            config.update(wandb_config)

        wandb.init(
            project=wandb_project,
            name=f"{model_name}_eval",
            config=config,
            reinit=True,
        )

    trainable_params = count_trainable_parameters(model)

    # ----------------------------------------------------------------
    # Evaluation loop
    # ----------------------------------------------------------------
    correct_top1 = 0
    correct_top5 = 0
    total_samples = 0
    start_time = time.time()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for batch_idx, (images, labels, _class_texts) in enumerate(dataloader):
        batch_size = images.shape[0]

        # All models implement: predict(images) -> (preds, scores)
        predictions, scores = model.predict(images)

        labels = labels.cpu()
        predictions = predictions.cpu()
        scores = scores.cpu()

        # Top-1 accuracy
        correct_top1 += (predictions == labels).sum().item()

        # Top-5 accuracy
        num_classes = scores.shape[1]
        k = min(5, num_classes)

        # Heuristic: OT models use distances (lower = better).
        is_distance_based = hasattr(model, "sinkhorn_reg")

        if is_distance_based:
            _, top5_indices = torch.topk(-scores, k=k, dim=-1)
        else:
            _, top5_indices = torch.topk(scores, k=k, dim=-1)

        labels_expanded = labels.unsqueeze(1).expand_as(top5_indices)
        correct_top5 += (top5_indices == labels_expanded).any(dim=1).sum().item()

        total_samples += batch_size

        # Log per-batch GPU memory to wandb.
        if use_wandb and torch.cuda.is_available():
            gpu_mem = _get_gpu_memory_mb()
            wandb.log({"batch_idx": batch_idx, **gpu_mem})



    # ----------------------------------------------------------------
    # Compute final metrics
    # ----------------------------------------------------------------
    eval_time = time.time() - start_time
    top1_accuracy = correct_top1 / total_samples * 100
    top5_accuracy = correct_top5 / total_samples * 100
    gpu_memory = _get_gpu_memory_mb()

    metrics = {
        "top1_accuracy": top1_accuracy,
        "top5_accuracy": top5_accuracy,
        "trainable_params": trainable_params,
        "total_samples": total_samples,
        "eval_time_seconds": eval_time,
        **gpu_memory,
    }

    print(f"[{model_name}] Top-1: {top1_accuracy:.2f}% | Top-5: {top5_accuracy:.2f}% | Params: {trainable_params:,} | {eval_time:.1f}s")

    # Log final summary metrics to wandb
    if use_wandb:
        wandb.log({
            f"{model_name}/top1_accuracy": top1_accuracy,
            f"{model_name}/top5_accuracy": top5_accuracy,
            f"{model_name}/trainable_params": trainable_params,
            f"{model_name}/eval_time_seconds": eval_time,
            **{f"{model_name}/{k}": v for k, v in gpu_memory.items()},
        })
        wandb.run.summary["top1_accuracy"] = top1_accuracy
        wandb.run.summary["top5_accuracy"] = top5_accuracy
        wandb.run.summary["trainable_params"] = trainable_params
        wandb.finish()

    return metrics


# ============================================================================
# Training Loop (for Carlo's CoOp and Marco's CLIP-Adapter)
# ============================================================================

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    class_names: Optional[List[str]] = None,
    epochs: int = 20,
    lr: float = 2e-3,
    optimizer_type: str = "sgd",
    weight_decay: float = 5e-4,
    momentum: float = 0.9,
    scheduler_type: str = "cosine",
    warmup_epochs: int = 1,
    use_wandb: bool = True,
    wandb_project: str = "clip-eurosat",
    wandb_config: Optional[Dict] = None,
    model_name: str = "model",
) -> Dict[str, Any]:
    """
    Generic training loop for CLIP adapter models.

    Designed for subclasses of ``BaseCLIPWrapper`` that add a small number
    of trainable parameters (learnable prompts, adapter MLPs, etc.) while
    keeping the CLIP backbone frozen.

    **Training protocol** (standard for CoOp / CLIP-Adapter):

    1. Compute image features via ``model.get_image_features(images)``.
    2. Compute text features via ``model.get_text_features(prompts)``.
    3. Compute scaled cosine-similarity logits using CLIP's learned
       temperature (``logit_scale``).
    4. Cross-entropy loss against ground-truth labels.
    5. Backprop through **only** the trainable parameters.

    **Important for Carlo / Marco**:  Your overridden ``get_text_features``
    or ``get_image_features`` methods must **NOT** use ``@torch.no_grad()``.
    Gradients need to flow through your trainable parameters.  The frozen
    backbone parameters have ``requires_grad=False``, so PyTorch will not
    waste memory or compute on them.

    Parameters
    ----------
    model : nn.Module
        A ``BaseCLIPWrapper`` subclass with trainable parameters.
        Must expose ``model.device``, ``model.model.logit_scale``,
        ``model.get_image_features()``, and ``model.get_text_features()``.
    train_loader : DataLoader
        Training set dataloader (from ``dataset.py``).
    val_loader : Optional[DataLoader]
        Validation / test set dataloader.  If provided, Top-1 and Top-5
        accuracy are computed after each epoch.
    class_names : Optional[List[str]]
        Raw class names (e.g., ``["forest", "river", ...]``).  If None,
        defaults to ``EUROSAT_CLASS_NAMES``.
    epochs : int
        Number of training epochs.
    lr : float
        Learning rate.  CoOp typically uses 2e-3 with SGD;  CLIP-Adapter
        uses 1e-3 with AdamW.  Tune as needed.
    optimizer_type : str
        One of ``"sgd"``, ``"adam"``, ``"adamw"``.
    weight_decay : float
        Weight decay (L2 regularization).
    momentum : float
        Momentum for SGD (ignored for Adam / AdamW).
    scheduler_type : str
        ``"cosine"`` for cosine annealing with linear warmup, or
        ``"none"`` to disable scheduling.
    warmup_epochs : int
        Number of linear warmup epochs (only used with cosine scheduler).
    use_wandb : bool
        Whether to log training curves to Weights & Biases.
    wandb_project : str
        W&B project name.
    wandb_config : Optional[Dict]
        Extra configuration to log.
    model_name : str
        Human-readable name for logging.

    Returns
    -------
    history : Dict[str, Any]
        Training history containing:
        - ``"train_loss"``: list of per-epoch average loss
        - ``"train_acc"``:  list of per-epoch training accuracy (%)
        - ``"val_top1"``:   list of per-epoch validation Top-1 accuracy (%)
        - ``"val_top5"``:   list of per-epoch validation Top-5 accuracy (%)
        - ``"lr"``:         list of per-epoch learning rates
        - ``"best_val_top1"``: best validation Top-1 accuracy
        - ``"peak_gpu_mb"``:   peak GPU memory during training
    """
    from dataset import EUROSAT_CLASS_NAMES, PROMPT_TEMPLATE

    if class_names is None:
        class_names = EUROSAT_CLASS_NAMES

    # Build full text prompts from raw class names.
    prompts = [PROMPT_TEMPLATE.format(name) for name in class_names]

    # ----------------------------------------------------------------
    # Collect trainable parameters.  Only these will receive gradients.
    # ----------------------------------------------------------------
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise ValueError(
            "No trainable parameters found in the model. "
            "Did you forget to add nn.Parameter or register a submodule?"
        )

    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"[{model_name}] Training {n_trainable:,} params | {epochs} epochs | LR={lr} | {optimizer_type}")

    # ----------------------------------------------------------------
    # Optimizer
    # ----------------------------------------------------------------
    if optimizer_type == "sgd":
        optimizer = SGD(
            trainable_params, lr=lr,
            momentum=momentum, weight_decay=weight_decay,
        )
    elif optimizer_type == "adam":
        optimizer = Adam(
            trainable_params, lr=lr, weight_decay=weight_decay,
        )
    elif optimizer_type == "adamw":
        optimizer = AdamW(
            trainable_params, lr=lr, weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer_type: '{optimizer_type}'")

    # ----------------------------------------------------------------
    # Learning rate scheduler
    # Cosine annealing with linear warmup is standard for both CoOp
    # and CLIP-Adapter.
    # ----------------------------------------------------------------
    if scheduler_type == "cosine":
        if warmup_epochs > 0 and epochs > warmup_epochs:
            warmup = LinearLR(
                optimizer, start_factor=0.01, total_iters=warmup_epochs,
            )
            cosine = CosineAnnealingLR(
                optimizer, T_max=epochs - warmup_epochs,
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_epochs],
            )
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    elif scheduler_type == "none":
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler_type: '{scheduler_type}'")

    # ----------------------------------------------------------------
    # W&B initialization
    # ----------------------------------------------------------------
    if use_wandb:
        config = {
            "model_name": model_name,
            "epochs": epochs,
            "lr": lr,
            "optimizer": optimizer_type,
            "scheduler": scheduler_type,
            "weight_decay": weight_decay,
            "warmup_epochs": warmup_epochs,
            "trainable_params": n_trainable,
        }
        if wandb_config:
            config.update(wandb_config)
        wandb.init(
            project=wandb_project,
            name=f"{model_name}_train",
            config=config,
            reinit=True,
        )

    # Reset peak GPU memory for accurate tracking.
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # ----------------------------------------------------------------
    # Training loop
    # ----------------------------------------------------------------
    # Put the model in train mode for adapter layers (dropout, etc.),
    # but keep the frozen CLIP backbone in eval mode to avoid enabling
    # dropout in pretrained transformer layers.
    model.train()
    model.model.eval()

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_top1": [],
        "val_top5": [],
        "lr": [],
    }
    best_val_acc = 0.0
    start_time = time.time()

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for batch_idx, (images, labels, _class_texts) in enumerate(train_loader):
            images = images.to(model.device)
            labels = labels.to(model.device)

            # ----------------------------------------------------------
            # Forward pass
            # ----------------------------------------------------------
            # Image features: (B, D) — goes through adapter for Marco.
            image_features = model.get_image_features(images)
            # Text features:  (C, D) — goes through learnable prompts
            # for Carlo.  For base/Marco models, this is the standard
            # frozen text encoding.
            text_features = model.get_text_features(prompts)

            # Scale by CLIP's learned temperature parameter.
            logit_scale = model.model.logit_scale.exp()
            logits = logit_scale * (image_features @ text_features.T)

            # ----------------------------------------------------------
            # Loss & backward
            # ----------------------------------------------------------
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ----------------------------------------------------------
            # Track batch metrics
            # ----------------------------------------------------------
            batch_size = images.shape[0]
            epoch_loss += loss.item() * batch_size
            epoch_correct += (logits.argmax(dim=-1) == labels).sum().item()
            epoch_total += batch_size

        # ==============================================================
        # End of epoch
        # ==============================================================
        avg_loss = epoch_loss / epoch_total
        train_acc = epoch_correct / epoch_total * 100
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(avg_loss)
        history["train_acc"].append(train_acc)
        history["lr"].append(current_lr)

        # Step the scheduler after each epoch.
        if scheduler is not None:
            scheduler.step()

        # ----------------------------------------------------------
        # Validation (if a val_loader is provided)
        # ----------------------------------------------------------
        val_msg = ""
        if val_loader is not None:
            model.eval()

            val_correct_1 = 0
            val_correct_5 = 0
            val_total = 0

            with torch.no_grad():
                for images_v, labels_v, _ in val_loader:
                    images_v = images_v.to(model.device)

                    img_feats = model.get_image_features(images_v)
                    txt_feats = model.get_text_features(prompts)
                    logit_scale = model.model.logit_scale.exp()
                    val_logits = (logit_scale * (img_feats @ txt_feats.T)).cpu()

                    labels_cpu = labels_v

                    # Top-1
                    val_correct_1 += (
                        val_logits.argmax(dim=-1) == labels_cpu
                    ).sum().item()

                    # Top-5
                    k = min(5, val_logits.shape[1])
                    _, top5_idx = torch.topk(val_logits, k=k, dim=-1)
                    labels_exp = labels_cpu.unsqueeze(1).expand_as(top5_idx)
                    val_correct_5 += (
                        (top5_idx == labels_exp).any(dim=1).sum().item()
                    )

                    val_total += images_v.shape[0]

            val_top1 = val_correct_1 / val_total * 100
            val_top5 = val_correct_5 / val_total * 100
            history["val_top1"].append(val_top1)
            history["val_top5"].append(val_top5)
            val_msg = f" | Val Top-1: {val_top1:.2f}%"

            if val_top1 > best_val_acc:
                best_val_acc = val_top1

            # Return to train mode (adapter layers), but keep
            # backbone frozen in eval mode.
            model.train()
            model.model.eval()

        print(f"  Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Acc: {train_acc:.2f}%{val_msg}")

        # ----------------------------------------------------------
        # Log to W&B
        # ----------------------------------------------------------
        if use_wandb:
            log_dict = {
                "epoch": epoch + 1,
                "train/loss": avg_loss,
                "train/accuracy": train_acc,
                "train/lr": current_lr,
            }
            if val_loader is not None:
                log_dict["val/top1_accuracy"] = val_top1
                log_dict["val/top5_accuracy"] = val_top5
            if torch.cuda.is_available():
                log_dict.update(_get_gpu_memory_mb())
            wandb.log(log_dict)

    # ================================================================
    # Training complete — final summary
    # ================================================================
    total_time = time.time() - start_time
    gpu_memory = _get_gpu_memory_mb()

    history["best_val_top1"] = best_val_acc
    history["peak_gpu_mb"] = gpu_memory.get("gpu/max_allocated_mb", 0.0)
    history["total_time_seconds"] = total_time

    val_info = f" | Best Val: {best_val_acc:.2f}%" if val_loader is not None else ""
    print(f"[{model_name}] Done in {total_time:.1f}s | Final Loss: {history['train_loss'][-1]:.4f}{val_info}")

    if use_wandb:
        wandb.run.summary["best_val_top1"] = best_val_acc
        wandb.run.summary["final_train_loss"] = history["train_loss"][-1]
        wandb.run.summary["total_time_seconds"] = total_time
        wandb.finish()

    # Return to eval mode for downstream evaluation.
    model.eval()

    return history


# ============================================================================
# Convenience: run all baselines
# ============================================================================

def run_all_evaluations(
    test_loader: DataLoader,
    device: str = "cuda",
    use_wandb: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate all available models and return a results dictionary.

    This is a convenience function that instantiates ZeroShot, LinearProbe,
    and OptimalTransport models, runs evaluation on each, and collects the
    results.

    Parameters
    ----------
    test_loader : DataLoader
        Test set dataloader.
    device : str
        "cuda" or "cpu".
    use_wandb : bool
        Whether to log to W&B.

    Returns
    -------
    all_results : Dict[str, Dict[str, Any]]
        Mapping from model name to its metrics dictionary.
    """
    from base_model import BaseCLIPWrapper
    from baselines import ZeroShotCLIP, LinearProbeCLIP
    from optimal_transport import OptimalTransportCLIP
    from dataset import get_dataloaders

    # Shared frozen CLIP backbone.
    clip_wrapper = BaseCLIPWrapper(device=device)

    all_results = {}


    zs_model = ZeroShotCLIP(clip_wrapper)
    all_results["ZeroShot"] = evaluate(
        zs_model, test_loader, model_name="ZeroShot", use_wandb=use_wandb,
    )


    train_loader, _ = get_dataloaders(batch_size=256, num_workers=4)
    lp_model = LinearProbeCLIP(clip_wrapper)
    lp_model.fit(train_loader)
    all_results["LinearProbe"] = evaluate(
        lp_model, test_loader, model_name="LinearProbe", use_wandb=use_wandb,
    )


    ot_model = OptimalTransportCLIP(clip_wrapper)
    all_results["OptimalTransport"] = evaluate(
        ot_model, test_loader, model_name="OptimalTransport", use_wandb=use_wandb,
    )



    return all_results


# ============================================================================
# Comparative Plotting
# ============================================================================

def plot_comparative_results(
    all_results: Dict[str, Dict[str, Any]],
    save_dir: str = "./plots",
) -> None:
    """
    Generate and save comparative plots from evaluation results.

    Produces three figures:
      1. **Accuracy vs. Trainable Parameters** — bar chart comparing Top-1
         and Top-5 accuracy for each model, with trainable param count
         annotated on each bar.
      2. **GPU Peak Memory & Eval Time** — side-by-side bar charts showing
         peak GPU memory and inference time for each model.
      3. **Combined Summary** — a single figure with all three panels
         for easy inclusion in reports/presentations.
    """
    os.makedirs(save_dir, exist_ok=True)

    model_names = list(all_results.keys())
    top1_accs = [all_results[m]["top1_accuracy"] for m in model_names]
    top5_accs = [all_results[m]["top5_accuracy"] for m in model_names]
    trainable_params = [all_results[m]["trainable_params"] for m in model_names]
    peak_mem = [
        all_results[m].get("gpu/max_allocated_mb", 0.0) for m in model_names
    ]
    eval_times = [all_results[m]["eval_time_seconds"] for m in model_names]

    plt.style.use("seaborn-v0_8-darkgrid")
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]

    # ================================================================
    # Figure 1: Accuracy vs. Trainable Parameters
    # ================================================================
    fig1, ax1 = plt.subplots(figsize=(10, 6))

    x = np.arange(len(model_names))
    bar_width = 0.35

    bars_top1 = ax1.bar(
        x - bar_width / 2, top1_accs, bar_width,
        label="Top-1 Accuracy", color=colors[0], edgecolor="white", linewidth=0.8,
    )
    bars_top5 = ax1.bar(
        x + bar_width / 2, top5_accs, bar_width,
        label="Top-5 Accuracy", color=colors[1], edgecolor="white", linewidth=0.8,
    )

    for bar in bars_top1:
        height = bar.get_height()
        ax1.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4), textcoords="offset points",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )
    for bar in bars_top5:
        height = bar.get_height()
        ax1.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4), textcoords="offset points",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    param_labels = []
    for name, params in zip(model_names, trainable_params):
        if params == 0:
            param_labels.append(f"{name}\n(0 params)")
        elif params < 1000:
            param_labels.append(f"{name}\n({params} params)")
        elif params < 1_000_000:
            param_labels.append(f"{name}\n({params / 1000:.1f}K params)")
        else:
            param_labels.append(f"{name}\n({params / 1_000_000:.2f}M params)")

    ax1.set_xlabel("Model (Trainable Parameters)", fontsize=12)
    ax1.set_ylabel("Accuracy (%)", fontsize=12)
    ax1.set_title(
        "Accuracy vs. Trainable Parameters",
        fontsize=14, fontweight="bold", pad=15,
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(param_labels, fontsize=10)
    ax1.legend(fontsize=11, loc="lower right")
    ax1.set_ylim(0, 105)
    ax1.grid(axis="y", alpha=0.3)

    fig1.tight_layout()
    fig1.savefig(
        os.path.join(save_dir, "accuracy_vs_params.png"),
        dpi=150, bbox_inches="tight",
    )
    print(f"[Plot] Saved: {save_dir}/accuracy_vs_params.png")

    # ================================================================
    # Figure 2: GPU Peak Memory & Eval Time
    # ================================================================
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(12, 5))

    bars_mem = ax2a.bar(
        model_names, peak_mem,
        color=colors[2], edgecolor="white", linewidth=0.8,
    )
    for bar in bars_mem:
        height = bar.get_height()
        if height > 0:
            ax2a.annotate(
                f"{height:.0f} MB",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
            )
    ax2a.set_ylabel("Peak GPU Memory (MB)", fontsize=11)
    ax2a.set_title("GPU Memory Usage", fontsize=13, fontweight="bold")
    ax2a.grid(axis="y", alpha=0.3)

    bars_time = ax2b.bar(
        model_names, eval_times,
        color=colors[3], edgecolor="white", linewidth=0.8,
    )
    for bar in bars_time:
        height = bar.get_height()
        ax2b.annotate(
            f"{height:.1f}s",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4), textcoords="offset points",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )
    ax2b.set_ylabel("Evaluation Time (seconds)", fontsize=11)
    ax2b.set_title("Inference Time", fontsize=13, fontweight="bold")
    ax2b.grid(axis="y", alpha=0.3)

    fig2.suptitle(
        "Resource Investment Comparison",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig2.tight_layout()
    fig2.savefig(
        os.path.join(save_dir, "resource_usage.png"),
        dpi=150, bbox_inches="tight",
    )
    print(f"[Plot] Saved: {save_dir}/resource_usage.png")

    # ================================================================
    # Figure 3: Combined summary (for report / presentation)
    # ================================================================
    fig3, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Accuracy
    axes[0].bar(
        x - bar_width / 2, top1_accs, bar_width,
        label="Top-1", color=colors[0], edgecolor="white",
    )
    axes[0].bar(
        x + bar_width / 2, top5_accs, bar_width,
        label="Top-5", color=colors[1], edgecolor="white",
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(model_names, fontsize=9)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("Classification Accuracy", fontweight="bold")
    axes[0].legend(fontsize=9)
    axes[0].set_ylim(0, 105)
    axes[0].grid(axis="y", alpha=0.3)

    # Panel 2: Trainable Parameters (log scale for readability)
    param_values = [max(p, 0.5) for p in trainable_params]  # avoid log(0)
    axes[1].bar(
        model_names, param_values,
        color=colors[4], edgecolor="white",
    )
    for i, (name, params) in enumerate(zip(model_names, trainable_params)):
        axes[1].annotate(
            f"{params:,}",
            xy=(i, max(params, 0.5)),
            xytext=(0, 4), textcoords="offset points",
            ha="center", va="bottom", fontsize=8, fontweight="bold",
        )
    axes[1].set_ylabel("Trainable Parameters")
    axes[1].set_title("Model Complexity", fontweight="bold")
    axes[1].set_yscale("symlog", linthresh=1)
    axes[1].grid(axis="y", alpha=0.3)

    # Panel 3: GPU Memory
    axes[2].bar(
        model_names, peak_mem,
        color=colors[2], edgecolor="white",
    )
    axes[2].set_ylabel("Peak GPU Memory (MB)")
    axes[2].set_title("Memory Footprint", fontweight="bold")
    axes[2].grid(axis="y", alpha=0.3)

    fig3.suptitle(
        "CLIP Adaptation - Comparative Analysis",
        fontsize=15, fontweight="bold", y=1.03,
    )
    fig3.tight_layout()
    fig3.savefig(
        os.path.join(save_dir, "combined_summary.png"),
        dpi=150, bbox_inches="tight",
    )
    print(f"[Plot] Saved: {save_dir}/combined_summary.png")

    plt.close("all")
    print(f"\n[Plot] All figures saved to '{save_dir}/'")


# ============================================================================
# Entry point: run this file directly to evaluate all baselines
# ============================================================================
if __name__ == "__main__":
    from dataset import get_dataloaders

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    _, test_loader = get_dataloaders(
        batch_size=64, num_workers=4, download=True
    )

    results = run_all_evaluations(
        test_loader, device=device, use_wandb=False
    )

    # Generate and save comparative plots.
    plot_comparative_results(results, save_dir="./plots")
