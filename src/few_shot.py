import sys
from pathlib import Path

FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parent
if str(FILE_DIR) not in sys.path:
    sys.path.insert(0, str(FILE_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import random
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset


# Label extraction
def extract_labels(dataset: Dataset) -> List[int]:
    """
    Recover the integer label of every sample **without decoding images**.

    Iterating a dataset to read its labels would open, resize and normalize
    every single JPEG — minutes of work to obtain information that is already
    sitting in a list on the torchvision object.  This function digs it out,
    handling the three wrappers defined in ``dataset.py``:

    """
    # Unwrap: (underlying torchvision object, indices into it).
    if hasattr(dataset, "subset") and isinstance(dataset.subset, Subset):
        base = dataset.subset.dataset
        indices: Sequence[int] = dataset.subset.indices
    elif hasattr(dataset, "dataset"):
        base = dataset.dataset
        indices = range(len(base))
    else:
        base = dataset
        indices = range(len(dataset))

    # Find the label list on the torchvision object.
    base_labels = None
    for attr in ("targets", "_labels", "labels"):
        if hasattr(base, attr):
            base_labels = list(getattr(base, attr))
            break
    if base_labels is None and hasattr(base, "samples"):
        base_labels = [label for _, label in base.samples]

    if base_labels is not None:
        return [int(base_labels[i]) for i in indices]

    print(
        "[few_shot] Warning: could not find a label list on "
        f"{type(base).__name__}; falling back to decoding every image."
    )
    return [int(dataset[i][1]) for i in range(len(dataset))]


# K-shot sampling
def few_shot_indices(
    labels: Sequence[int], n_shots: int, seed: int = 42
) -> List[int]:
    """
    Pick ``n_shots`` sample indices for each class.

    Sampling is **seeded and per-class**, so the same seed always yields the
    same support set: without that, a difference between two runs could come
    from the draw rather than from the method being tested.  Classes with
    fewer than ``n_shots`` examples contribute everything they have.
    """
    by_class: Dict[int, List[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        by_class[int(label)].append(idx)

    rng = random.Random(seed)
    selected: List[int] = []
    for label in sorted(by_class):
        pool = by_class[label]
        k = min(n_shots, len(pool))
        if k < n_shots:
            print(
                f"[few_shot] Warning: class {label} has only {len(pool)} "
                f"samples, requested {n_shots}."
            )
        selected.extend(rng.sample(pool, k))

    return sorted(selected)


def build_few_shot_loader(
    train_dataset: Dataset,
    n_shots: int = 16,
    batch_size: int = 32,
    num_workers: int = 0,
    seed: int = 42,
    shuffle: bool = True,
) -> DataLoader:
    """
    Wrap a full training set into a K-shot ``DataLoader``.
    """
    labels = extract_labels(train_dataset)
    indices = few_shot_indices(labels, n_shots=n_shots, seed=seed)
    subset = Subset(train_dataset, indices)

    print(
        f"[few_shot] {n_shots}-shot support set: {len(subset)} images "
        f"across {len(set(labels))} classes (seed={seed})."
    )

    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


# Smoke test
if __name__ == "__main__":
    try:
        from src.dataset import EuroSATDataset
    except ModuleNotFoundError:
        from dataset import EuroSATDataset

    train_dataset = EuroSATDataset(split="train", download=True)
    labels = extract_labels(train_dataset)
    print(f"Train split: {len(labels)} samples, {len(set(labels))} classes")

    for k in (1, 4, 16):
        indices = few_shot_indices(labels, n_shots=k)
        counts = defaultdict(int)
        for i in indices:
            counts[labels[i]] += 1
        assert all(c == k for c in counts.values()), counts
        print(f"  K={k:<3} -> {len(indices)} images, {dict(sorted(counts.items()))}")

    loader = build_few_shot_loader(train_dataset, n_shots=16, batch_size=32)
    images, batch_labels, texts = next(iter(loader))
    print(
        f"Batch: images {tuple(images.shape)}, labels {tuple(batch_labels.shape)}, "
        f"first text: {texts[0]!r}"
    )
