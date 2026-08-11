"""
dataset.py — EuroSAT Dataset & DataLoader Pipeline
====================================================

This module wraps the EuroSAT remote-sensing dataset (available via torchvision)
into a custom PyTorch Dataset that returns a *triple* per sample:

    (image_tensor, label_index, class_name_as_text)

The third element is critical: it lets us feed human-readable class names directly
to CLIP's text encoder without maintaining a separate lookup table downstream.

EuroSAT contains 27,000 geo-referenced Sentinel-2 satellite images across
10 land-use / land-cover classes (e.g., "Forest", "Highway", "River", etc.).
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from typing import Tuple, Optional, List


# ============================================================================
# Human-readable class names for EuroSAT
# ============================================================================
# These correspond to the 10 EuroSAT categories in label-index order.
# We prefix each with "a satellite image of " to create a natural-language
# prompt, which is the standard CLIP practice for zero-shot classification.
# The prefix helps CLIP's text encoder produce more discriminative features
# because CLIP was trained on (image, caption) pairs — not single words.
# ============================================================================
EUROSAT_CLASS_NAMES: List[str] = [
    "annual crop land",
    "forest",
    "herbaceous vegetation land",
    "highway or road",
    "industrial buildings",
    "pasture land",
    "permanent crop land",
    "residential buildings",
    "river",
    "sea or lake",
]

# Template used to convert bare class names into full text prompts for CLIP.
# This is a well-known technique from the original CLIP paper (Radford et al.)
# that significantly boosts zero-shot performance over using raw label strings.
PROMPT_TEMPLATE: str = "a satellite image of {}"


class EuroSATDataset(Dataset):
    """
    Custom PyTorch Dataset wrapping torchvision's EuroSAT.

    Key design decision: __getitem__ returns (image, label, class_name_text).
    This allows downstream models to access both the integer label (for loss
    computation / accuracy) and the text prompt (for CLIP text encoding)
    without needing any external mapping.

    Parameters
    ----------
    root : str
        Root directory where the dataset will be downloaded / cached.
    split : str
        One of "train" or "test".  We use torchvision's built-in split
        functionality, which creates a 70/30 train/test partition.
    transform : Optional[transforms.Compose]
        Image transformations to apply.  If None, a sensible default
        (resize → center-crop → normalize for CLIP) is used.
    download : bool
        Whether to download the dataset if not already present.
    """

    def __init__(
        self,
        root: str = "./data",
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
        download: bool = True,
    ) -> None:
        super().__init__()

        # ----------------------------------------------------------------
        # Default transform: designed to match OpenCLIP's expected input.
        #   1. Resize to 224×224 (CLIP's native resolution for ViT-B/32).
        #   2. Convert PIL image → tensor (scales pixel values to [0, 1]).
        #   3. Normalize with ImageNet mean/std — the same normalization
        #      that OpenCLIP models were trained with.
        # ----------------------------------------------------------------
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize(
                    (224, 224),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),   # CLIP/OpenAI stats
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ])

        # ----------------------------------------------------------------
        # Load EuroSAT via torchvision.
        # `split` must be "train" or "test".  torchvision handles the
        # partitioning internally using a fixed random seed for
        # reproducibility.
        # ----------------------------------------------------------------
        self.dataset = datasets.EuroSAT(
            root=root,
            transform=transform,
            download=download,
        )

        # ----------------------------------------------------------------
        # Perform a manual train/test split.
        # torchvision's EuroSAT does not have a built-in split argument,
        # so we create one ourselves using a fixed generator seed for
        # reproducibility.  We use an 80/20 split, which is standard for
        # remote sensing benchmarks.
        # ----------------------------------------------------------------
        total_len = len(self.dataset)
        train_len = int(0.8 * total_len)
        test_len = total_len - train_len

        # Use a fixed seed so every instantiation gets the same split,
        # regardless of the global random state.
        generator = torch.Generator().manual_seed(42)
        train_subset, test_subset = torch.utils.data.random_split(
            self.dataset,
            [train_len, test_len],
            generator=generator,
        )

        # Select the appropriate subset based on the requested split.
        if split == "train":
            self.subset = train_subset
        elif split == "test":
            self.subset = test_subset
        else:
            raise ValueError(
                f"Unknown split '{split}'. Must be 'train' or 'test'."
            )

        # Store the class names and prompt template for text generation.
        self.class_names = EUROSAT_CLASS_NAMES
        self.prompt_template = PROMPT_TEMPLATE

    def __len__(self) -> int:
        """Return the number of samples in the selected split."""
        return len(self.subset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """
        Retrieve a single sample.

        Returns
        -------
        image : torch.Tensor
            Preprocessed image tensor of shape (3, 224, 224).
        label : int
            Integer class index in [0, 9].
        class_name_text : str
            Human-readable text prompt, e.g.
            "a satellite image of forest".
            This is fed directly to CLIP's text encoder.
        """
        # Fetch the image and its integer label from the underlying dataset.
        image, label = self.subset[idx]

        # Convert the integer label to a natural-language prompt.
        # Example: label=1 → "forest" → "a satellite image of forest"
        class_name_text = self.prompt_template.format(
            self.class_names[label]
        )

        return image, label, class_name_text


# ============================================================================
# DataLoader Factory
# ============================================================================
def get_dataloaders(
    root: str = "./data",
    batch_size: int = 64,
    num_workers: int = 4,
    download: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and test DataLoaders for EuroSAT.

    Parameters
    ----------
    root : str
        Where to store / find the dataset on disk.
    batch_size : int
        Mini-batch size.  64 works well on a single GPU with 8 GB VRAM.
    num_workers : int
        Number of parallel data-loading workers.
        Set to 0 on Windows if you encounter multiprocessing errors.
    download : bool
        Whether to download EuroSAT if it's missing.

    Returns
    -------
    train_loader, test_loader : Tuple[DataLoader, DataLoader]
    """
    train_dataset = EuroSATDataset(
        root=root, split="train", download=download
    )
    test_dataset = EuroSATDataset(
        root=root, split="test", download=download
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,           # Shuffle training data every epoch.
        num_workers=num_workers,
        pin_memory=True,        # Speeds up host → GPU transfer.
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,          # No shuffling for evaluation — deterministic.
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, test_loader




# ============================================================================
# Quick sanity check (run this file directly to verify the pipeline)
# ============================================================================
if __name__ == "__main__":
    train_loader, test_loader = get_dataloaders(
        batch_size=4, num_workers=0, download=True,
    )

    images, labels, class_texts = next(iter(train_loader))
    print(f"Image batch shape : {images.shape}")      # (4, 3, 224, 224)
    print(f"Label batch       : {labels.tolist()}")    # e.g. [2, 7, 0, 5]
    print(f"Text prompts      : {class_texts}")        # list of 4 strings
    print(f"\nTrain samples: {len(train_loader.dataset)}")
    print(f"Test  samples: {len(test_loader.dataset)}")

