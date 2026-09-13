
# Use: python clip_adapter_experiments.py --epochs 10 --batch_size 64 --lr 1e-3 --no_wandb

import os
import json
import argparse
import torch
import matplotlib.pyplot as plt
import matplotlib

import sys
from pathlib import Path

# Add project root and src directory to sys.path
FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parent
if str(FILE_DIR) not in sys.path:
    sys.path.insert(0, str(FILE_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


PLOTS_DIR = PROJECT_ROOT / "plots"

from src.base_model import BaseCLIPWrapper
from src.dataset import get_dataloaders
from src.engine import train, evaluate, count_trainable_parameters
from src.clip_adapter import CLIPAdapterModel

matplotlib.use("Agg")


def run_single_experiment(
    reduction_ratio: int = 4,
    alpha: float = 0.2,
    learnable_alpha: bool = False,
    epochs: int = 10,
    lr: float = 1e-3,
    batch_size: int = 64,
    device: str = "cuda",
    use_wandb: bool = False,
):
    mode = "learned" if learnable_alpha else "fixed"
    print(f"\n==================================================================")
    print(f"Running CLIP-Adapter Experiment: Reduction Ratio = {reduction_ratio}, "
          f"Alpha = {alpha} ({mode})")
    print(f"==================================================================")

    train_loader, test_loader = get_dataloaders(batch_size=batch_size, num_workers=0)

    model = CLIPAdapterModel(
        model_name="ViT-B-32",
        pretrained="laion2b_s34b_b79k",
        device=device,
        reduction_ratio=reduction_ratio,
        alpha=alpha,
        learnable_alpha=learnable_alpha,
    )

    model_name = f"CLIPAdapter_R{reduction_ratio}_a{alpha}_{mode}"

    # Train model using standard engine loop
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        epochs=epochs,
        lr=lr,
        optimizer_type="adamw",
        weight_decay=5e-4,
        scheduler_type="cosine",
        warmup_epochs=1,
        use_wandb=use_wandb,
        wandb_project="clip-eurosat-marco",
        model_name=model_name,
    )

    # Final evaluation metrics
    metrics = evaluate(
        model=model,
        dataloader=test_loader,
        model_name=model_name,
        use_wandb=use_wandb,
        wandb_project="clip-eurosat-marco",
    )

    metrics["history"] = history
    metrics["reduction_ratio"] = reduction_ratio
    metrics["alpha"] = alpha
    metrics["learnable_alpha"] = learnable_alpha
    # A learned alpha is unconstrained unless constrain_alpha is on, so the
    # value it settled at is a result in its own right and has to be logged.
    metrics["alpha_final"] = float(model.alpha)

    return model, metrics


def plot_marco_sweeps(sweep_results: list, alpha_sweep_results: list = None, save_dir: str = str(PLOTS_DIR)):
    """Generate comparative visualization plots for Marco's experiments."""
    os.makedirs(save_dir, exist_ok=True)
    plt.style.use("seaborn-v0_8-darkgrid")

    # Plot 1: Accuracy vs Reduction Ratio
    ratios = [res["reduction_ratio"] for res in sweep_results]
    accs = [res["top1_accuracy"] for res in sweep_results]
    params = [res["trainable_params"] for res in sweep_results]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    color = "tab:blue"
    ax1.set_xlabel("Reduction Ratio (r)")
    ax1.set_ylabel("Top-1 Accuracy (%)", color=color)
    ax1.plot(ratios, accs, marker="o", linewidth=2, color=color, label="Top-1 Acc")
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("Trainable Parameters", color=color)
    ax2.plot(ratios, params, marker="s", linestyle="--", color=color, label="Params")
    ax2.tick_params(axis="y", labelcolor=color)

    plt.title("CLIP-Adapter: Impact of Reduction Ratio on Accuracy & Parameters")
    fig.tight_layout()
    plt.savefig(os.path.join(save_dir, "clip_adapter_reduction_sweep.png"), dpi=150)
    plt.close()
    print(f"[Plot] Saved: {save_dir}/clip_adapter_reduction_sweep.png")

    # Plot 2: Accuracy vs Residual Alpha
    if alpha_sweep_results:
        alphas = [res["alpha"] for res in alpha_sweep_results]
        alpha_accs = [res["top1_accuracy"] for res in alpha_sweep_results]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_xlabel("Residual Blending Alpha")
        ax.set_ylabel("Top-1 Accuracy (%)", color="tab:blue")
        ax.plot(alphas, alpha_accs, marker="o", linewidth=2, color="tab:blue", label="Top-1 Acc")
        ax.tick_params(axis="y", labelcolor="tab:blue")

        plt.title("CLIP-Adapter: Impact of Residual Blending Alpha on Accuracy")
        fig.tight_layout()
        plt.savefig(os.path.join(save_dir, "clip_adapter_alpha_sweep.png"), dpi=150)
        plt.close()
        print(f"[Plot] Saved: {save_dir}/clip_adapter_alpha_sweep.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLIP-Adapter Hyperparameter Search")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B logging")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Run baseline sweep over reduction ratios
    ratios_to_test = [2, 4, 8, 16]
    results = []

    for r in ratios_to_test:
        _, metrics = run_single_experiment(
            reduction_ratio=r,
            alpha=0.2,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            device=device,
            use_wandb=not args.no_wandb,
        )
        results.append(metrics)

    # Save metrics JSON
    os.makedirs(PLOTS_DIR, exist_ok=True)
    with open(PLOTS_DIR / "clip_adapter_results.json", "w") as f:
        json.dump(results, f, indent=4)

    # Run alpha sweep
    print("\n--- Starting Alpha Sweep ---")
    alphas_to_test = [0.1, 0.2, 0.5, 0.8]
    alpha_results = []

    for a in alphas_to_test:
        _, metrics = run_single_experiment(
            reduction_ratio=4,  # default recommended
            alpha=a,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            device=device,
            use_wandb=not args.no_wandb,
        )
        alpha_results.append(metrics)

    with open(PLOTS_DIR / "clip_adapter_alpha_results.json", "w") as f:
        json.dump(alpha_results, f, indent=4)

    # Plot results
    plot_marco_sweeps(results, alpha_sweep_results=alpha_results)
    print("\nCLIP-Adapter hyperparameter exploration complete!")
