
# Use: python clip_adapter_experiments.py --epochs 10 --batch_size 64 --lr 1e-3

import os
import json
import argparse
import torch
import matplotlib.pyplot as plt
import matplotlib

from src.base_model import BaseCLIPWrapper
from src.dataset import get_dataloaders
from src.engine import train, evaluate, count_trainable_parameters
from src.clip_adapter import CLIPAdapterModel

matplotlib.use("Agg")


def run_single_experiment(
    reduction_ratio: int = 4,
    alpha: float = 0.2,
    epochs: int = 10,
    lr: float = 1e-3,
    batch_size: int = 64,
    device: str = "cuda",
    use_wandb: bool = False,
):
    print(f"\n==================================================================")
    print(f"Running CLIP-Adapter Experiment: Reduction Ratio = {reduction_ratio}, Alpha = {alpha}")
    print(f"==================================================================")

    train_loader, test_loader = get_dataloaders(batch_size=batch_size, num_workers=0)

    model = CLIPAdapterModel(
        model_name="ViT-B-32",
        pretrained="laion2b_s34b_b79k",
        device=device,
        reduction_ratio=reduction_ratio,
        alpha=alpha,
    )

    model_name = f"CLIPAdapter_R{reduction_ratio}_a{alpha}"

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

    return model, metrics


def plot_marco_sweeps(sweep_results: list, save_dir: str = "./plots"):
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
    os.makedirs("./plots", exist_ok=True)
    with open("./plots/clip_adapter_results.json", "w") as f:
        json.dump(results, f, indent=4)

    # Plot results
    plot_marco_sweeps(results)
    print("\nCLIP-Adapter hyperparameter exploration complete!")
