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

import sys
import os
from pathlib import Path

# Add project root and src directory to sys.path
FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parent
if str(FILE_DIR) not in sys.path:
    sys.path.insert(0, str(FILE_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import ssl
import certifi


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
from sklearn.metrics import confusion_matrix, classification_report
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
    all_predictions = []
    all_labels = []
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

        # Collect all predictions and labels for per-class analysis.
        all_predictions.append(predictions)
        all_labels.append(labels)

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

    # Concatenate all batch-level predictions and labels.
    all_predictions = torch.cat(all_predictions, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    metrics = {
        "top1_accuracy": top1_accuracy,
        "top5_accuracy": top5_accuracy,
        "trainable_params": trainable_params,
        "total_samples": total_samples,
        "eval_time_seconds": eval_time,
        "all_predictions": all_predictions,
        "all_labels": all_labels,
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
    prompt_template: Optional[str] = None,
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
    prompt_template : Optional[str]
        Template applied to each class name, e.g.
        ``"a satellite image of {}"``.  If None, falls back to the model's
        own ``prompt_template`` attribute, and finally to the EuroSAT
        default — so existing call sites are unaffected.

        **Pass this whenever you train on something other than EuroSAT.**
        Each dataset in ``dataset.py`` carries its own template
        (``train_loader.dataset.prompt_template``): DTD wants
        ``"a photo of a {} texture"`` and Flowers102 wants
        ``"a photo of a {}, a type of flower"``.  Using the EuroSAT
        template on DTD builds the prompt "a satellite image of banded",
        which is meaningless — the run does not crash, it just quietly
        loses accuracy.
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
        - ``"gpu_mb"``:      list of per-epoch GPU memory (MB)
        - ``"peak_gpu_mb"``:   peak GPU memory during training
    """
    try:
        from src.dataset import EUROSAT_CLASS_NAMES, PROMPT_TEMPLATE
    except ModuleNotFoundError:
        from dataset import EUROSAT_CLASS_NAMES, PROMPT_TEMPLATE

    # ----------------------------------------------------------------
    # Resolve the class set and the prompt template.
    #
    # Precedence: explicit argument → whatever the model was built with →
    # EuroSAT default.  The middle step matters: a model constructed for
    # DTD already knows its own class names and template, and forcing the
    # caller to repeat them here is exactly how the two drift apart.
    # ----------------------------------------------------------------
    if class_names is None:
        class_names = getattr(model, "class_names", None) or EUROSAT_CLASS_NAMES
    if prompt_template is None:
        prompt_template = getattr(model, "prompt_template", None) or PROMPT_TEMPLATE

    # Build full text prompts from raw class names.
    prompts = [prompt_template.format(name) for name in class_names]

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
        # Per-epoch GPU memory.  The brief asks for a "Memory Usage vs.
        # Epochs" plot, and a single end-of-run scalar cannot draw a curve.
        # Two series are recorded because they answer different questions:
        #   • gpu_mb      — running peak since the start of training,
        #                   monotonically non-decreasing.  This is the
        #                   number that tells you whether the run fits in
        #                   5 GB of VRAM.
        #   • gpu_epoch_mb — peak *within* that epoch alone, obtained by
        #                   resetting the CUDA counter at the top of every
        #                   epoch.  This is the one that shows a method's
        #                   steady-state cost and makes methods comparable.
        "gpu_mb": [],
        "gpu_epoch_mb": [],
    }
    best_val_acc = 0.0
    running_peak_mb = 0.0
    start_time = time.time()

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        # Reset the CUDA peak counter so that ``max_memory_allocated()``
        # measures this epoch alone.  The peak over the whole run is kept
        # separately in ``running_peak_mb``: resetting the counter is what
        # makes the per-epoch series meaningful, but it would otherwise
        # throw away the global peak the brief also asks for.
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

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

        # Memory for this epoch, recorded *before* validation so the number
        # reflects the training step (forward + backward + optimizer state)
        # and is not contaminated by the validation batch size.
        epoch_peak_mb = _get_gpu_memory_mb()["gpu/max_allocated_mb"]
        running_peak_mb = max(running_peak_mb, epoch_peak_mb)
        history["gpu_epoch_mb"].append(epoch_peak_mb)
        history["gpu_mb"].append(running_peak_mb)

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
    # NOT ``gpu_memory["gpu/max_allocated_mb"]``: the counter is reset at the
    # top of every epoch, so reading it here would report the *last* epoch's
    # peak rather than the run's.  ``running_peak_mb`` is the true maximum.
    history["peak_gpu_mb"] = max(
        running_peak_mb, gpu_memory.get("gpu/max_allocated_mb", 0.0)
    )
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
    extra_models: Optional[Dict[str, Any]] = None,
    clip_wrapper: Optional[Any] = None,
    include_baselines: bool = True,
    train_loader: Optional[DataLoader] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate the baselines — and any adapted models handed in — in one pass.

    The baselines (Zero-Shot, Zero-Shot Ensemble, Linear Probe, Optimal
    Transport) are parameter-free or cheap to fit, so this function builds
    them itself.  The three workstreams' methods (CoOp, CLIP-Adapter,
    Tip-Adapter, LoRA) must be **trained first** and passed in through
    ``extra_models``: re-training them inside a function called
    "run_all_evaluations" would be a surprising amount of work to trigger
    by accident, and the training hyperparameters belong to whoever owns
    the method.

    Without ``extra_models`` the comparative figures show only the
    baselines — two of which sit at zero trainable parameters — and the
    "Accuracy vs. Trainable Parameters" trade-off that the project is
    actually about does not appear at all.

    Parameters
    ----------
    test_loader : DataLoader
        Test set dataloader.
    device : str
        "cuda" or "cpu".
    use_wandb : bool
        Whether to log to W&B.
    extra_models : Optional[Dict[str, Any]]
        Already-trained models to evaluate alongside the baselines, keyed
        by the name to show in the plots, e.g.::

            coop = CoOpModel(device=device)
            train(coop, few_shot_loader, epochs=100, lr=2e-3)

            adapter = CLIPAdapterModel(device=device, reduction_ratio=4)
            train(adapter, few_shot_loader, epochs=20, lr=1e-3,
                  optimizer_type="adamw")

            results = run_all_evaluations(
                test_loader, device=device,
                extra_models={"CoOp (M=16, 16-shot)": coop,
                              "CLIP-Adapter (r=4)": adapter},
            )

        Each value only has to expose ``predict(images) -> (preds, scores)``,
        which every subclass of ``BaseCLIPWrapper`` inherits.
    clip_wrapper : Optional[Any]
        Reuse an existing frozen backbone instead of loading a second copy.
        On a 5 GB card this is the difference between fitting and not.
    include_baselines : bool
        Set to ``False`` to evaluate only ``extra_models`` — useful when
        the baseline numbers are already on disk and only a new method
        needs re-measuring.
    train_loader : Optional[DataLoader]
        Training data for the Linear Probe.  If None, a full-shot EuroSAT
        loader is built.  **Pass a few-shot loader here** if the comparison
        is meant to be at matched supervision: a Linear Probe fitted on
        21,600 images is not comparable with CoOp fitted on 160.

    Returns
    -------
    all_results : Dict[str, Dict[str, Any]]
        Mapping from model name to its metrics dictionary.
    """
    try:
        from src.base_model import BaseCLIPWrapper
        from src.baselines import ZeroShotCLIP, ZeroShotEnsembleCLIP, LinearProbeCLIP
        from src.optimal_transport import OptimalTransportCLIP
        from src.dataset import get_dataloaders
    except ModuleNotFoundError:
        from base_model import BaseCLIPWrapper
        from baselines import ZeroShotCLIP, ZeroShotEnsembleCLIP, LinearProbeCLIP
        from optimal_transport import OptimalTransportCLIP
        from dataset import get_dataloaders

    all_results: Dict[str, Dict[str, Any]] = {}

    if include_baselines:
        # Shared frozen CLIP backbone — one copy for all four baselines.
        if clip_wrapper is None:
            clip_wrapper = BaseCLIPWrapper(device=device)

        zs_model = ZeroShotCLIP(clip_wrapper)
        all_results["ZeroShot"] = evaluate(
            zs_model, test_loader, model_name="ZeroShot", use_wandb=use_wandb,
        )

        zs_ensemble = ZeroShotEnsembleCLIP(clip_wrapper)
        all_results["ZeroShot-Ensemble"] = evaluate(
            zs_ensemble, test_loader, model_name="ZeroShot-Ensemble",
            use_wandb=use_wandb,
        )

        if train_loader is None:
            train_loader, _ = get_dataloaders(batch_size=256, num_workers=4)
        lp_model = LinearProbeCLIP(clip_wrapper)
        lp_model.fit(train_loader)
        all_results["LinearProbe"] = evaluate(
            lp_model, test_loader, model_name="LinearProbe", use_wandb=use_wandb,
        )

        ot_model = OptimalTransportCLIP(clip_wrapper)
        all_results["OptimalTransport"] = evaluate(
            ot_model, test_loader, model_name="OptimalTransport",
            use_wandb=use_wandb,
        )

    # ----------------------------------------------------------------
    # The three workstreams' methods.  Already trained by their owner —
    # this loop only measures them, through the very same evaluate() the
    # baselines went through, so the numbers are directly comparable.
    # ----------------------------------------------------------------
    for name, model in (extra_models or {}).items():
        if not hasattr(model, "predict"):
            raise TypeError(
                f"extra_models['{name}'] has no .predict(images) method; "
                "evaluate() cannot score it.  Every BaseCLIPWrapper "
                "subclass inherits one."
            )
        # Adapter modules must not be left in train mode: dropout and the
        # like would make the reported accuracy noisy and irreproducible.
        if hasattr(model, "eval"):
            model.eval()
        all_results[name] = evaluate(
            model, test_loader, model_name=name, use_wandb=use_wandb,
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
    if save_dir == "./plots" and not os.path.exists("./plots"):
        save_dir = str(PROJECT_ROOT / "plots")

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
# Confusion Matrices
# ============================================================================

def plot_memory_vs_epochs(
    histories: Dict[str, Dict[str, Any]],
    save_dir: str = "./plots",
    per_epoch: bool = True,
) -> None:
    """
    Plot GPU memory usage against training epochs — a brief deliverable.

    The project treats resource cost as a first-class metric, not an
    afterthought, and this is the figure that shows it over time rather
    than as a single bar.

    Parameters
    ----------
    histories : Dict[str, Dict[str, Any]]
        Mapping from a method's name to the ``history`` dict returned by
        ``train()``.  Any entry without a memory series is skipped with a
        warning rather than crashing the whole figure.
    save_dir : str
        Output directory.  Same fallback logic as the other plots.
    per_epoch : bool
        ``True``  → plot ``history["gpu_epoch_mb"]``, the peak *within*
        each epoch.  This is the fair comparison between methods: it shows
        steady-state cost and it is flat for a well-behaved training loop.

        ``False`` → plot ``history["gpu_mb"]``, the running peak since the
        start.  Monotonically non-decreasing by construction, so the curve
        is a staircase; useful to answer "does this run fit in 5 GB?".

    Notes
    -----
    The two series are genuinely different measurements and mixing them up
    produces a plot that looks like a memory leak when nothing is leaking.
    The axis label states which one is being drawn.
    """
    if save_dir == "./plots" and not os.path.exists("./plots"):
        save_dir = str(PROJECT_ROOT / "plots")
    os.makedirs(save_dir, exist_ok=True)

    key = "gpu_epoch_mb" if per_epoch else "gpu_mb"
    plt.style.use("seaborn-v0_8-darkgrid")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    plotted = 0

    for idx, (name, hist) in enumerate(histories.items()):
        series = hist.get(key)
        if not series:
            print(
                f"[Plot] Warning: '{name}' has no '{key}' series "
                f"(trained before per-epoch memory logging existed?) — skipped."
            )
            continue
        epochs_axis = range(1, len(series) + 1)
        ax.plot(
            epochs_axis, series,
            marker="o", markersize=3, linewidth=1.8,
            label=f"{name} (peak {max(series):.0f} MB)",
            color=f"C{idx % 10}",
        )
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        print("[Plot] No memory series available — 'memory_vs_epochs' not written.")
        return

    ylabel = (
        "Peak GPU memory within epoch (MB)" if per_epoch
        else "Peak GPU memory since start (MB)"
    )
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(
        "Memory Usage vs. Epochs", fontsize=14, fontweight="bold", pad=15,
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    out = os.path.join(save_dir, "memory_vs_epochs.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved: {out}")


def plot_confusion_matrices(
    all_results: Dict[str, Dict[str, Any]],
    class_names: Optional[List[str]] = None,
    save_dir: str = "./plots",
) -> None:
    """
    Generate and save a confusion matrix heatmap for each evaluated model.

    Each matrix shows predicted class (x-axis) vs. true class (y-axis).
    Values are **normalized per row** (i.e., each row sums to 1.0) so that
    the diagonal shows per-class recall.  This makes it easy to spot which
    classes are frequently confused.

    Parameters
    ----------
    all_results : Dict[str, Dict[str, Any]]
        Output from ``run_all_evaluations()``.  Each entry must contain
        ``"all_predictions"`` and ``"all_labels"`` arrays.
    class_names : Optional[List[str]]
        Human-readable class names for axis labels.
        Defaults to ``EUROSAT_CLASS_NAMES``.
    save_dir : str
        Directory where plots are saved.
    """
    try:
        from src.dataset import EUROSAT_CLASS_NAMES
    except ModuleNotFoundError:
        from dataset import EUROSAT_CLASS_NAMES

    if class_names is None:
        class_names = EUROSAT_CLASS_NAMES

    if save_dir == "./plots" and not os.path.exists("./plots"):
        save_dir = str(PROJECT_ROOT / "plots")
    os.makedirs(save_dir, exist_ok=True)

    # Shortened labels for readability on the axes.
    short_names = [n.replace(" land", "").replace(" buildings", "").title()
                   for n in class_names]

    n_models = len(all_results)
    # Arrange in a grid: up to 3 columns.
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(6 * n_cols, 5.5 * n_rows),
        squeeze=False,
    )

    for idx, (model_name, metrics) in enumerate(all_results.items()):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]

        preds = metrics["all_predictions"]
        labels = metrics["all_labels"]

        # Compute row-normalized confusion matrix (each row sums to 1.0).
        cm = confusion_matrix(labels, preds, labels=range(len(class_names)))
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        # Handle classes with zero samples (avoid NaN).
        cm_norm = np.nan_to_num(cm_norm, nan=0.0)

        im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues",
                       vmin=0, vmax=1)
        ax.set_title(
            f"{model_name}\n(Top-1: {metrics['top1_accuracy']:.1f}%)",
            fontsize=11, fontweight="bold",
        )

        # Add text annotations on each cell.
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                val = cm_norm[i, j]
                color = "white" if val > 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color=color)

        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(short_names, fontsize=8)
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("True", fontsize=9)

    # Hide unused subplots.
    for idx in range(n_models, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    fig.suptitle(
        "Confusion Matrices (Row-Normalized)",
        fontsize=14, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    fig.savefig(
        os.path.join(save_dir, "confusion_matrices.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)
    print(f"[Plot] Saved: {save_dir}/confusion_matrices.png")


# ============================================================================
# Per-Class Accuracy Comparison
# ============================================================================

def plot_per_class_metrics(
    all_results: Dict[str, Dict[str, Any]],
    class_names: Optional[List[str]] = None,
    save_dir: str = "./plots",
) -> None:
    """
    Generate a grouped bar chart comparing per-class accuracy across models.

    For each class, we show the recall (= per-class accuracy) of every model
    side by side.  This makes it easy to answer questions like:
      - "Which classes does ZeroShot struggle with the most?"
      - "Does Prompt Ensembling help uniformly, or only on specific classes?"
      - "Is there a class where OT actually beats cosine similarity?"

    Also prints a concise classification report to the console.

    Parameters
    ----------
    all_results : Dict[str, Dict[str, Any]]
        Output from ``run_all_evaluations()``.
    class_names : Optional[List[str]]
        Class names for labeling.
    save_dir : str
        Directory where plots are saved.
    """
    try:
        from src.dataset import EUROSAT_CLASS_NAMES
    except ModuleNotFoundError:
        from dataset import EUROSAT_CLASS_NAMES

    if class_names is None:
        class_names = EUROSAT_CLASS_NAMES

    if save_dir == "./plots" and not os.path.exists("./plots"):
        save_dir = str(PROJECT_ROOT / "plots")
    os.makedirs(save_dir, exist_ok=True)

    short_names = [n.replace(" land", "").replace(" buildings", "").title()
                   for n in class_names]
    n_classes = len(class_names)
    model_names = list(all_results.keys())
    n_models = len(model_names)

    # Compute per-class accuracy (recall) for each model.
    per_class_acc = {}  # model_name -> array of shape (n_classes,)

    for model_name, metrics in all_results.items():
        preds = metrics["all_predictions"]
        labels = metrics["all_labels"]

        # Per-class accuracy = diagonal of row-normalized confusion matrix.
        cm = confusion_matrix(labels, preds, labels=range(n_classes))
        row_sums = cm.sum(axis=1)
        # Avoid division by zero for classes with no samples.
        row_sums[row_sums == 0] = 1
        class_acc = cm.diagonal().astype(float) / row_sums * 100
        per_class_acc[model_name] = class_acc

        # Print a concise classification report.
        print(f"\n{'='*60}")
        print(f"Classification Report: {model_name}")
        print(f"{'='*60}")
        report = classification_report(
            labels, preds,
            target_names=short_names,
            digits=2,
            zero_division=0,
        )
        print(report)

    # ----------------------------------------------------------------
    # Grouped bar chart
    # ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(n_classes)
    bar_width = 0.8 / n_models
    colors = plt.cm.Set2(np.linspace(0, 1, max(n_models, 3)))

    for i, model_name in enumerate(model_names):
        offset = (i - n_models / 2 + 0.5) * bar_width
        ax.bar(
            x + offset,
            per_class_acc[model_name],
            bar_width,
            label=model_name,
            color=colors[i],
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Per-Class Accuracy (%)", fontsize=12)
    ax.set_title(
        "Per-Class Accuracy Comparison Across Models",
        fontsize=14, fontweight="bold", pad=15,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=30, ha="right", fontsize=10)
    ax.legend(fontsize=9, loc="lower right", ncol=2)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        os.path.join(save_dir, "per_class_accuracy.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)
    print(f"\n[Plot] Saved: {save_dir}/per_class_accuracy.png")


# ============================================================================
# Entry point: run this file directly to evaluate all baselines
# ============================================================================
if __name__ == "__main__":
    try:
        from src.dataset import get_dataloaders
    except ModuleNotFoundError:
        from dataset import get_dataloaders

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    _, test_loader = get_dataloaders(
        batch_size=64, num_workers=4, download=True
    )

    # Baselines only.  To put CoOp / CLIP-Adapter / Tip-Adapter / LoRA on
    # the same figures, train them first and hand them over:
    #
    #     from src.coop import CoOpModel
    #     from src.clip_adapter import CLIPAdapterModel
    #     from src.few_shot import build_few_shot_loader
    #
    #     support = build_few_shot_loader(train_loader.dataset, n_shots=16)
    #
    #     coop = CoOpModel(device=device)
    #     h_coop = train(coop, support, epochs=100, lr=2e-3, use_wandb=False,
    #                    model_name="CoOp")
    #
    #     adapter = CLIPAdapterModel(device=device, reduction_ratio=4)
    #     h_adapter = train(adapter, support, epochs=20, lr=1e-3,
    #                       optimizer_type="adamw", use_wandb=False,
    #                       model_name="CLIP-Adapter")
    #
    #     results = run_all_evaluations(
    #         test_loader, device=device, use_wandb=False,
    #         train_loader=support,          # matched supervision, see below
    #         extra_models={"CoOp (M=16)": coop, "CLIP-Adapter (r=4)": adapter},
    #     )
    #     plot_memory_vs_epochs({"CoOp": h_coop, "CLIP-Adapter": h_adapter})
    #
    # Note the ``train_loader=support``: fitting the Linear Probe on all
    # 21,600 images and CoOp on 160 puts two different experiments on one
    # axis.  Whatever protocol is chosen, it has to be the same for every
    # bar in the figure.
    results = run_all_evaluations(
        test_loader, device=device, use_wandb=False
    )

    # Generate and save comparative plots.
    plot_comparative_results(results, save_dir="./plots")

    # Generate confusion matrices and per-class analysis.
    plot_confusion_matrices(results, save_dir="./plots")
    plot_per_class_metrics(results, save_dir="./plots")
