"""
Runs the three ablations the brief asks for on the text side, plus the
few-shot sweep that gives them meaning:

* **context length** ``M in {4, 8, 16}``;
* **Unified Context vs. Class-Specific Context (CSC)** — which generalizes
  better with the same amount of supervision;
* **learning rate**, tuned specifically for the text embeddings. 
* **shots** ``K in {1, 2, 4, 8, 16}`` — the axis along which few-shot
  adaptation is supposed to pay off.

Every run reports Top-1 / Top-5 accuracy, the number of trainable parameters,
peak GPU memory and wall-clock time, because in this project resource cost is
a first-class metric and not an afterthought.

Training goes through ``engine.train()`` and evaluation through
``engine.evaluate()`` — no method reimplements the loop.

Usage
-----
::

    python src/coop_experiments.py --sweep ctx_len     # M = 4, 8, 16
    python src/coop_experiments.py --sweep csc         # unified vs CSC
    python src/coop_experiments.py --sweep lr          # LR search
    python src/coop_experiments.py --sweep shots       # K = 1..16
    python src/coop_experiments.py --sweep all
    python src/coop_experiments.py --single --n-ctx 16 --shots 16 --epochs 50
"""

import sys
from pathlib import Path

FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parent
if str(FILE_DIR) not in sys.path:
    sys.path.insert(0, str(FILE_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import time
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Subset

try:
    from src import engine
    from src.coop import CoOpModel
    from src.dataset import EUROSAT_CLASS_NAMES, EuroSATDataset
    from src.few_shot import build_few_shot_loader
except ModuleNotFoundError:
    import engine
    from coop import CoOpModel
    from dataset import EUROSAT_CLASS_NAMES, EuroSATDataset
    from few_shot import build_few_shot_loader


RESULTS_DIR = PROJECT_ROOT / "src/plots"
RESULTS_JSON = RESULTS_DIR / "coop_results.json"


# ============================================================================
# Data plumbing
# ============================================================================

def build_test_loaders(
    batch_size: int = 64,
    num_workers: int = 0,
    val_subset: int = 1000,
    seed: int = 42,
) -> tuple:
    """
    Build the full test loader plus a small validation loader.

    Two loaders, two jobs:

    * ``test_loader`` — the **entire** test split (5,400 images).  Used once,
      at the end of a run, by ``engine.evaluate()``.  Never subsampled, or
      accuracies stop being comparable with Mattia's and Marco's numbers.
    * ``val_loader`` — a fixed random slice of the same split, used *during*
      training to draw the per-epoch accuracy curve.  A full pass every epoch
      would dominate the runtime of a 160-image few-shot run.

    The slice is drawn with a fixed seed so every run sees the same one.

    Parameters
    ----------
    batch_size, num_workers : int
        Standard DataLoader settings.  ``num_workers=0`` is the safe default
        on Windows.
    val_subset : int
        Size of the per-epoch validation slice.  ``0`` disables it, and
        training runs without a validation curve.
    seed : int
        Seed of the slice.

    Returns
    -------
    (test_loader, val_loader) : Tuple[DataLoader, Optional[DataLoader]]
    """
    test_dataset = EuroSATDataset(split="test", download=True)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader: Optional[DataLoader] = None
    if val_subset > 0:
        generator = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(test_dataset), generator=generator)
        indices = perm[: min(val_subset, len(test_dataset))].tolist()
        val_loader = DataLoader(
            Subset(test_dataset, indices),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    return test_loader, val_loader


# ============================================================================
# A single run
# ============================================================================

def run_single_experiment(
    train_dataset: EuroSATDataset,
    test_loader: DataLoader,
    val_loader: Optional[DataLoader],
    n_shots: int = 16,
    n_ctx: int = 16,
    class_specific: bool = False,
    ctx_init: Optional[str] = None,
    lr: float = 2e-3,
    epochs: int = 50,
    batch_size: int = 32,
    seed: int = 42,
    device: str = "cuda",
    use_wandb: bool = False,
    tag: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Train one CoOp configuration and evaluate it on the full test set.

    Returns
    -------
    result : Dict[str, Any]
        Configuration, training history and final metrics, flattened into a
        single JSON-serializable dictionary.
    """
    if tag is None:
        # The tag is the key under which the run is stored, so it must name
        # every axis that changes the result — epochs included, or a short
        # debug run would silently overwrite a full one.
        kind = "csc" if class_specific else "unified"
        tag = f"CoOp_M{n_ctx}_{kind}_K{n_shots}_lr{lr:g}_e{epochs}"

    print("\n" + "=" * 78)
    print(f"  {tag}")
    print("=" * 78)

    # Reproducibility: the model init (random context vectors) and the
    # support-set draw both depend on the seed.
    torch.manual_seed(seed)

    train_loader = build_few_shot_loader(
        train_dataset,
        n_shots=n_shots,
        batch_size=batch_size,
        seed=seed,
    )

    model = CoOpModel(
        device=device,
        class_names=EUROSAT_CLASS_NAMES,
        n_ctx=n_ctx,
        class_specific=class_specific,
        ctx_init=ctx_init,
    )

    start = time.time()
    history = engine.train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_names=EUROSAT_CLASS_NAMES,
        epochs=epochs,
        lr=lr,
        optimizer_type="sgd",       # CoOp's recipe: SGD + momentum.
        weight_decay=5e-4,
        momentum=0.9,
        scheduler_type="cosine",
        warmup_epochs=1,            # Not optional: see the module notes.
        use_wandb=use_wandb,
        model_name=tag,
    )
    train_time = time.time() - start

    metrics = engine.evaluate(
        model=model,
        dataloader=test_loader,
        model_name=tag,
        use_wandb=use_wandb,
    )

    result: Dict[str, Any] = {
        "tag": tag,
        "n_shots": n_shots,
        "n_ctx": n_ctx,
        "class_specific": class_specific,
        "ctx_init": ctx_init,
        "lr": lr,
        "epochs": epochs,
        "batch_size": batch_size,
        "seed": seed,
        "top1_accuracy": metrics["top1_accuracy"],
        "top5_accuracy": metrics["top5_accuracy"],
        "trainable_params": metrics["trainable_params"],
        "train_time_seconds": train_time,
        "peak_gpu_mb": history.get("peak_gpu_mb", 0.0),
        "best_val_top1": history.get("best_val_top1", 0.0),
        "train_loss": history["train_loss"],
        "train_acc": history["train_acc"],
        "val_top1": history["val_top1"],
        "lr_curve": history["lr"],
    }

    # Free the backbone before the next configuration is built: a fresh
    # CoOpModel loads another full copy of CLIP onto the GPU.
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# ============================================================================
# Sweeps
# ============================================================================

def sweep_ctx_len(common: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Context length M in {4, 8, 16}, Unified Context."""
    return [
        run_single_experiment(n_ctx=m, class_specific=False, **common)
        for m in (4, 8, 16)
    ]


def sweep_csc(common: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Unified Context vs. Class-Specific Context at the same M."""
    return [
        run_single_experiment(n_ctx=16, class_specific=csc, **common)
        for csc in (False, True)
    ]


def sweep_lr(common: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Learning-rate search for the context vectors."""
    common = {k: v for k, v in common.items() if k != "lr"}
    return [
        run_single_experiment(n_ctx=16, class_specific=False, lr=lr, **common)
        for lr in (2e-4, 2e-3, 2e-2, 2e-1)
    ]


def sweep_shots(common: Dict[str, Any]) -> List[Dict[str, Any]]:
    """K-shot sweep: the axis on which few-shot adaptation is judged."""
    common = {k: v for k, v in common.items() if k != "n_shots"}
    return [
        run_single_experiment(
            n_ctx=16, class_specific=False, n_shots=k, **common
        )
        for k in (1, 2, 4, 8, 16)
    ]


# ============================================================================
# Persistence
# ============================================================================

def save_results(results: List[Dict[str, Any]], path: Path = RESULTS_JSON) -> None:
    """
    Append results to the project's JSON file, keyed by run tag.

    Appending rather than overwriting means separate sweeps accumulate into
    one file that the final comparative plots can read.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    existing: Dict[str, Any] = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"[coop] {path.name} was corrupt; starting a fresh file.")

    for result in results:
        existing[result["tag"]] = result

    path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(f"\n[coop] {len(results)} run(s) saved to {path}")


def load_results(path: Path = RESULTS_JSON) -> List[Dict[str, Any]]:
    """Load every run recorded so far, in insertion order."""
    if not path.exists():
        return []
    return list(json.loads(path.read_text(encoding="utf-8")).values())


# ============================================================================
# Plots
# ============================================================================
# Zero-shot CLIP needs no training data at all, so its accuracy is the one
# baseline that is directly comparable against a few-shot run.  The Linear
# Probe (93.61%) and CLIP-Adapter (97.28%) numbers already in the project were
# measured on the *full* 21,600-image training set and are deliberately NOT
# drawn here: putting them on a few-shot axis would compare two different
# protocols on the same picture.
ZERO_SHOT_TOP1 = 44.56


def _select(
    results: List[Dict[str, Any]], **constraints: Any
) -> List[Dict[str, Any]]:
    """Return the runs whose fields match every given constraint."""
    return [
        r for r in results
        if all(r.get(key) == value for key, value in constraints.items())
    ]


def _annotate_params(ax, xs, ys, runs) -> None:
    """Label each point with its trainable-parameter count."""
    for x, y, r in zip(xs, ys, runs):
        ax.annotate(
            f"{r['trainable_params']:,}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
            color="dimgray",
        )


def plot_coop_results(
    results: Optional[List[Dict[str, Any]]] = None,
    save_dir: Path = RESULTS_DIR,
) -> None:
    """
    Draw every panel for which results exist.

    Two figures are produced:

    * ``coop_sweeps.png`` — one panel per ablation (learning rate, context
      length, unified vs CSC, number of shots).  Panels with no data are left
      out, so the function is safe to call after a single sweep.
    * ``coop_training_curves.png`` — loss and validation accuracy per epoch,
      the "training curves" the brief asks every run to log.

    Points are annotated with their trainable-parameter count, because in this
    project accuracy alone is only half of a result.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if results is None:
        results = load_results()
    if not results:
        print("[coop] No results to plot yet.")
        return

    save_dir.mkdir(parents=True, exist_ok=True)
    max_shots = max(r["n_shots"] for r in results)

    # Every panel varies exactly one axis, so all the *other* axes have to be
    # pinned — otherwise the LR sweep (whose runs are all M=16, K=16) leaks
    # into the context-length, CSC and shots panels and piles several points
    # onto the same x position.  The reference value of an axis is the one
    # shared by the largest number of runs.
    def _modal(field: str) -> Any:
        counts: Dict[Any, int] = {}
        for r in results:
            counts[r[field]] = counts.get(r[field], 0) + 1
        return max(counts, key=counts.get)

    ref_lr = _modal("lr")
    ref_epochs = _modal("epochs")

    # ------------------------------------------------------------------
    # Collect the panels that actually have data.
    # ------------------------------------------------------------------
    panels = []

    lr_runs = sorted(
        _select(
            results, n_ctx=16, class_specific=False,
            n_shots=max_shots, epochs=ref_epochs,
        ),
        key=lambda r: r["lr"],
    )
    if len({r["lr"] for r in lr_runs}) > 1:
        panels.append(("lr", lr_runs))

    ctx_runs = sorted(
        _select(
            results, class_specific=False, n_shots=max_shots,
            lr=ref_lr, epochs=ref_epochs,
        ),
        key=lambda r: r["n_ctx"],
    )
    if len({r["n_ctx"] for r in ctx_runs}) > 1:
        panels.append(("ctx", ctx_runs))

    csc_runs = sorted(
        _select(
            results, n_ctx=16, n_shots=max_shots,
            lr=ref_lr, epochs=ref_epochs,
        ),
        key=lambda r: r["class_specific"],
    )
    if len({r["class_specific"] for r in csc_runs}) > 1:
        panels.append(("csc", csc_runs))

    shot_runs = sorted(
        _select(
            results, n_ctx=16, class_specific=False,
            lr=ref_lr, epochs=ref_epochs,
        ),
        key=lambda r: r["n_shots"],
    )
    if len({r["n_shots"] for r in shot_runs}) > 1:
        panels.append(("shots", shot_runs))

    if not panels:
        print("[coop] Not enough runs yet to draw a comparison panel.")
    else:
        fig, axes = plt.subplots(
            1, len(panels), figsize=(5.2 * len(panels), 4.4), squeeze=False
        )
        for ax, (kind, runs) in zip(axes[0], panels):
            ys = [r["top1_accuracy"] for r in runs]

            if kind == "lr":
                xs = [r["lr"] for r in runs]
                ax.semilogx(xs, ys, "o-", color="tab:blue")
                ax.set_xlabel("Learning rate (SGD)")
                ax.set_title(f"LR sweep — M=16, unified, K={max_shots}")
                _annotate_params(ax, xs, ys, runs)

            elif kind == "ctx":
                xs = [r["n_ctx"] for r in runs]
                ax.plot(xs, ys, "o-", color="tab:green")
                ax.set_xticks(xs)
                ax.set_xlabel("Context length M (tokens)")
                ax.set_title(f"Context length — unified, K={max_shots}")
                _annotate_params(ax, xs, ys, runs)

            elif kind == "csc":
                labels = [
                    "Class-Specific" if r["class_specific"] else "Unified"
                    for r in runs
                ]
                bars = ax.bar(labels, ys, color=["tab:blue", "tab:orange"])
                ax.set_title(f"Unified vs CSC — M=16, K={max_shots}")
                # Headroom so the per-bar labels do not run into the title.
                ax.set_ylim(top=max(ys) * 1.20)
                for bar, r in zip(bars, runs):
                    ax.annotate(
                        f"{r['top1_accuracy']:.2f}%\n{r['trainable_params']:,} params",
                        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        textcoords="offset points",
                        xytext=(0, 4),
                        ha="center",
                        fontsize=8,
                    )

            elif kind == "shots":
                xs = [r["n_shots"] for r in runs]
                ax.semilogx(xs, ys, "o-", base=2, color="tab:red")
                ax.set_xticks(xs)
                ax.set_xticklabels([str(x) for x in xs])
                ax.set_xlabel("Shots per class (K)")
                ax.set_title("Few-shot sweep — M=16, unified")

            ax.axhline(
                ZERO_SHOT_TOP1, ls="--", lw=1, color="gray",
                label=f"Zero-Shot ({ZERO_SHOT_TOP1:.2f}%)",
            )
            ax.set_ylabel("Top-1 accuracy (%)")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)

        fig.suptitle("CoOp on EuroSAT — few-shot ablations", fontsize=13)
        fig.tight_layout()
        out = save_dir / "coop_sweeps.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"[coop] Saved {out}")

    # ------------------------------------------------------------------
    # Training curves.
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for r in results:
        epochs = range(1, len(r["train_loss"]) + 1)
        axes[0].plot(epochs, r["train_loss"], label=r["tag"], lw=1.2)
        if r["val_top1"]:
            axes[1].plot(epochs, r["val_top1"], label=r["tag"], lw=1.2)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Training loss")
    axes[0].set_title("Training loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation Top-1 (%)")
    axes[1].set_title("Validation accuracy")
    axes[1].axhline(ZERO_SHOT_TOP1, ls="--", lw=1, color="gray")
    for ax in axes:
        ax.grid(alpha=0.3)
        if len(results) <= 8:
            ax.legend(fontsize=7)

    fig.suptitle("CoOp training curves", fontsize=13)
    fig.tight_layout()
    out = save_dir / "coop_training_curves.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[coop] Saved {out}")


def print_summary(results: List[Dict[str, Any]]) -> None:
    """Print the results table in the same shape as the report's table."""
    print("\n" + "=" * 78)
    print(f"{'Run':<34}{'Top-1':>8}{'Top-5':>8}{'Params':>10}{'GPU MB':>9}{'Time s':>9}")
    print("-" * 78)
    for r in results:
        print(
            f"{r['tag']:<34}{r['top1_accuracy']:>7.2f}%{r['top5_accuracy']:>7.2f}%"
            f"{r['trainable_params']:>10,}{r['peak_gpu_mb']:>9.0f}{r['train_time_seconds']:>9.1f}"
        )
    print("=" * 78)


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="CoOp experiments on EuroSAT")
    parser.add_argument(
        "--sweep",
        choices=["ctx_len", "csc", "lr", "shots", "all"],
        default=None,
        help="Which ablation to run.",
    )
    parser.add_argument("--single", action="store_true", help="Run one configuration.")
    parser.add_argument("--shots", type=int, default=16)
    parser.add_argument("--n-ctx", type=int, default=16)
    parser.add_argument("--csc", action="store_true", help="Class-Specific Context.")
    parser.add_argument(
        "--ctx-init",
        type=str,
        default=None,
        help="Initialize the context from a phrase, e.g. 'a satellite image of'.",
    )
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-subset", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--wandb", action="store_true", help="Log to Weights & Biases.")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Only redraw the plots from the saved results, without training.",
    )
    args = parser.parse_args()

    if args.plot:
        plot_coop_results()
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[coop] Device: {device}")
    if device == "cpu":
        print("[coop] Warning: running on CPU. Expect this to be very slow.")

    train_dataset = EuroSATDataset(split="train", download=True)
    test_loader, val_loader = build_test_loaders(
        num_workers=args.num_workers,
        val_subset=args.val_subset,
        seed=args.seed,
    )

    common = {
        "train_dataset": train_dataset,
        "test_loader": test_loader,
        "val_loader": val_loader,
        "n_shots": args.shots,
        "lr": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "device": device,
        "use_wandb": args.wandb,
        "ctx_init": args.ctx_init,
    }

    results: List[Dict[str, Any]] = []

    if args.single or args.sweep is None:
        results.append(
            run_single_experiment(
                n_ctx=args.n_ctx, class_specific=args.csc, **common
            )
        )
    else:
        if args.sweep in ("ctx_len", "all"):
            results += sweep_ctx_len(common)
        if args.sweep in ("csc", "all"):
            results += sweep_csc(common)
        if args.sweep in ("lr", "all"):
            results += sweep_lr(common)
        if args.sweep in ("shots", "all"):
            results += sweep_shots(common)

    save_results(results)
    print_summary(results)
    plot_coop_results()


if __name__ == "__main__":
    main()
