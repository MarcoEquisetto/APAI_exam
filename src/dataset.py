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

# ============================================================================
# Prompt Ensembling Templates
# ============================================================================
# The CLIP paper (Radford et al., 2021, Section 3.1.4) showed that averaging
# text features across multiple prompt templates improves zero-shot accuracy
# by 3-5% on average.  The idea is that different phrasings activate
# different parts of CLIP's learned text-image alignment, and averaging
# them produces a more robust text prototype.
#
# These templates are designed for the satellite / remote sensing domain
# of EuroSAT.  They vary along three axes:
#   1. Viewpoint:  "satellite", "aerial", "remote sensing", "overhead"
#   2. Framing:    "image of", "photo of", "view of", "showing"
#   3. Specificity: generic vs. domain-specific ("land use category: {}")
# ============================================================================
EUROSAT_PROMPT_TEMPLATES: List[str] = [
    "a satellite image of {}",
    "a satellite photo of {}",
    "a centered satellite photo of {}",
    "an aerial view of {}",
    "a remote sensing image of {}",
    "a satellite image showing {}",
    "land use category: {}",
]


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

        # Resolve data path dynamically if default relative path doesn't exist in CWD
        if root == "./data" and not os.path.exists("./data") and (PROJECT_ROOT / "data").exists():
            root = str(PROJECT_ROOT / "data")

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
# DTD (Describable Textures Dataset)
# ============================================================================
# DTD contains 5,640 images across 47 texture categories (e.g., "banded",
# "bubbly", "cracked", "woven").  It is a standard few-shot evaluation
# benchmark for CLIP and a very different domain from EuroSAT, making it
# ideal for testing whether adaptation methods generalize.
#
# Reference: Cimpoi et al. (2014). "Describing Textures in the Wild." CVPR.
# ============================================================================

DTD_CLASS_NAMES: List[str] = [
    "banded", "blotchy", "braided", "bubbly", "bumpy",
    "chequered", "cobwebbed", "cracked", "crosshatched", "crystalline",
    "dotted", "fibrous", "flecked", "freckled", "frilly",
    "gauzy", "grid", "grooved", "honeycombed", "interlaced",
    "knitted", "lacelike", "lined", "marbled", "matted",
    "meshed", "paisley", "perforated", "pitted", "pleated",
    "polka-dotted", "porous", "potholed", "scaly", "smeared",
    "spiralled", "sprinkled", "stained", "stratified", "striped",
    "studded", "swirly", "veined", "waffled", "woven",
    "wrinkled", "zigzagged",
]

DTD_PROMPT_TEMPLATE: str = "a photo of a {} texture"

DTD_PROMPT_TEMPLATES: List[str] = [
    "a photo of a {} texture",
    "a photo of a {} surface",
    "a photo of a {} pattern",
    "a close-up photo of a {} texture",
    "a {} texture",
    "a {} surface",
    "a {} pattern",
]


class DTDDataset(Dataset):
    """
    Custom PyTorch Dataset wrapping torchvision's DTD.

    Same interface as ``EuroSATDataset``: __getitem__ returns
    ``(image, label, class_name_text)``.

    DTD has official train/val/test splits (10 partitions), so we use
    the built-in ``split`` argument instead of random_split.

    Parameters
    ----------
    root : str
        Root directory for dataset storage.
    split : str
        One of "train" or "test".  "test" uses the official test split.
    transform : Optional[transforms.Compose]
        Image transformations.  Default matches CLIP preprocessing.
    download : bool
        Whether to download the dataset if not present.
    partition : int
        DTD partition index (1-10).  Default 1 for reproducibility.
    """

    def __init__(
        self,
        root: str = "./data",
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
        download: bool = True,
        partition: int = 1,
    ) -> None:
        super().__init__()

        if transform is None:
            transform = transforms.Compose([
                transforms.Resize(
                    (224, 224),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ])

        # Resolve data path dynamically.
        if root == "./data" and not os.path.exists("./data") and (PROJECT_ROOT / "data").exists():
            root = str(PROJECT_ROOT / "data")

        # Map "test" to the official test split.
        dtd_split = split if split in ("train", "val", "test") else "train"

        self.dataset = datasets.DTD(
            root=root,
            split=dtd_split,
            partition=partition,
            transform=transform,
            download=download,
        )

        # DTD class names are stored in dataset._classes (sorted alphabetically).
        self.class_names = DTD_CLASS_NAMES
        self.prompt_template = DTD_PROMPT_TEMPLATE

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """
        Returns (image, label, class_name_text).

        The text prompt uses the DTD template, e.g.:
        "a photo of a cracked texture".
        """
        image, label = self.dataset[idx]
        class_name_text = self.prompt_template.format(
            self.class_names[label]
        )
        return image, label, class_name_text


# ============================================================================
# DTD DataLoader Factory
# ============================================================================
def get_dtd_dataloaders(
    root: str = "./data",
    batch_size: int = 64,
    num_workers: int = 4,
    download: bool = True,
    partition: int = 1,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and test DataLoaders for DTD.

    Parameters
    ----------
    root : str
        Where to store / find the dataset on disk.
    batch_size : int
        Mini-batch size.
    num_workers : int
        Number of parallel data-loading workers.
    download : bool
        Whether to download DTD if it's missing.
    partition : int
        DTD partition (1-10).

    Returns
    -------
    train_loader, test_loader : Tuple[DataLoader, DataLoader]
    """
    train_dataset = DTDDataset(
        root=root, split="train", download=download, partition=partition,
    )
    test_dataset = DTDDataset(
        root=root, split="test", download=download, partition=partition,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, test_loader



# ============================================================================
# Oxford Flowers102 Dataset
# ============================================================================
# Oxford 102 Flower is a fine-grained image classification dataset consisting
# of 102 flower categories commonly found in the United Kingdom.  Each class
# contains between 40 and 258 images with large scale, pose, and lighting
# variations.  It is a standard few-shot and zero-shot benchmark for CLIP
# and related vision-language models.
#
# Reference: Nilsback & Zisserman (2008). "Automated Flower Classification
#            over a Large Number of Classes." ICVGIP.
#
# NOTE: This dataset requires scipy for loading target files from .mat format.
# ============================================================================

FLOWERS102_CLASS_NAMES: List[str] = [
    "pink primrose",             # 0  (original label 1)
    "hard-leaved pocket orchid", # 1
    "canterbury bells",          # 2
    "sweet pea",                 # 3
    "english marigold",          # 4
    "tiger lily",                # 5
    "moon orchid",               # 6
    "bird of paradise",          # 7
    "monkshood",                 # 8
    "globe thistle",             # 9
    "snapdragon",                # 10
    "colts foot",                # 11
    "king protea",               # 12
    "spear thistle",             # 13
    "yellow iris",               # 14
    "globe-flower",              # 15
    "purple coneflower",         # 16
    "peruvian lily",             # 17
    "balloon flower",            # 18
    "giant white arum lily",     # 19
    "fire lily",                 # 20
    "pincushion flower",         # 21
    "fritillary",                # 22
    "red ginger",                # 23
    "grape hyacinth",            # 24
    "corn poppy",                # 25
    "prince of wales feathers",  # 26
    "stemless gentian",          # 27
    "artichoke",                 # 28
    "sweet william",             # 29
    "carnation",                 # 30
    "garden phlox",              # 31
    "love in the mist",          # 32
    "mexican aster",             # 33
    "alpine sea holly",          # 34
    "ruby-lipped cattleya",      # 35
    "cape flower",               # 36
    "great masterwort",          # 37
    "siam tulip",                # 38
    "lenten rose",               # 39
    "barbeton daisy",            # 40
    "daffodil",                  # 41
    "sword lily",                # 42
    "poinsettia",                # 43
    "bolero deep blue",          # 44
    "wallflower",                # 45
    "marigold",                  # 46
    "buttercup",                 # 47
    "oxeye daisy",               # 48
    "common dandelion",          # 49
    "petunia",                   # 50
    "wild pansy",                # 51
    "primula",                   # 52
    "sunflower",                 # 53
    "pelargonium",               # 54
    "bishop of llandaff",        # 55
    "gaura",                     # 56
    "geranium",                  # 57
    "orange dahlia",             # 58
    "pink-yellow dahlia",        # 59
    "cautleya spicata",          # 60
    "japanese anemone",          # 61
    "black-eyed susan",          # 62
    "silverbush",                # 63
    "californian poppy",         # 64
    "osteospermum",              # 65
    "spring crocus",             # 66
    "bearded iris",              # 67
    "windflower",                # 68
    "tree poppy",                # 69
    "gazania",                   # 70
    "azalea",                    # 71
    "water lily",                # 72
    "rose",                      # 73
    "thorn apple",               # 74
    "morning glory",             # 75
    "passion flower",            # 76
    "lotus",                     # 77
    "toad lily",                 # 78
    "anthurium",                 # 79
    "frangipani",                # 80
    "clematis",                  # 81
    "hibiscus",                  # 82
    "columbine",                 # 83
    "desert-rose",               # 84
    "tree mallow",               # 85
    "magnolia",                  # 86
    "cyclamen",                  # 87
    "watercress",                # 88
    "canna lily",                # 89
    "hippeastrum",               # 90
    "bee balm",                  # 91
    "ball moss",                 # 92
    "foxglove",                  # 93
    "bougainvillea",             # 94
    "camellia",                  # 95
    "mallow",                    # 96
    "mexican petunia",           # 97
    "bromelia",                  # 98
    "blanket flower",            # 99
    "trumpet creeper",           # 100
    "blackberry lily",           # 101
]

FLOWERS102_PROMPT_TEMPLATE: str = "a photo of a {}, a type of flower"

FLOWERS102_PROMPT_TEMPLATES: List[str] = [
    "a photo of a {}, a type of flower",
    "a close-up photo of a {}",
    "a photo of a {} flower",
    "a macro photo of a {}",
    "a bright photo of a {}, a type of flower",
    "a photo of the {}",
    "a good photo of a {}, a type of flower",
]


class Flowers102Dataset(Dataset):
    """
    Custom PyTorch Dataset wrapping torchvision's Flowers102.

    Same interface as ``EuroSATDataset`` and ``DTDDataset``:
    __getitem__ returns ``(image, label, class_name_text)``.

    Flowers102 has official train/val/test splits, so we use
    the built-in ``split`` argument.

    Parameters
    ----------
    root : str
        Root directory for dataset storage.
    split : str
        One of "train" or "test".  "test" uses the official test split.
    transform : Optional[transforms.Compose]
        Image transformations.  Default matches CLIP preprocessing.
    download : bool
        Whether to download the dataset if not present.
    """

    def __init__(
        self,
        root: str = "./data",
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
        download: bool = True,
    ) -> None:
        super().__init__()

        if transform is None:
            transform = transforms.Compose([
                transforms.Resize(
                    (224, 224),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ])

        # Resolve data path dynamically.
        if root == "./data" and not os.path.exists("./data") and (PROJECT_ROOT / "data").exists():
            root = str(PROJECT_ROOT / "data")

        # Map our "test" split to the official test split.
        flowers_split = split if split in ("train", "val", "test") else "train"

        self.dataset = datasets.Flowers102(
            root=root,
            split=flowers_split,
            transform=transform,
            download=download,
        )

        self.class_names = FLOWERS102_CLASS_NAMES
        self.prompt_template = FLOWERS102_PROMPT_TEMPLATE

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """
        Returns (image, label, class_name_text).

        The text prompt uses the Flowers102 template, e.g.:
        "a photo of a sunflower, a type of flower".
        """
        image, label = self.dataset[idx]
        class_name_text = self.prompt_template.format(
            self.class_names[label]
        )
        return image, label, class_name_text


# ============================================================================
# Flowers102 DataLoader Factory
# ============================================================================
def get_flowers102_dataloaders(
    root: str = "./data",
    batch_size: int = 64,
    num_workers: int = 4,
    download: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and test DataLoaders for Oxford Flowers102.

    Parameters
    ----------
    root : str
        Where to store / find the dataset on disk.
    batch_size : int
        Mini-batch size.
    num_workers : int
        Number of parallel data-loading workers.
    download : bool
        Whether to download Flowers102 if it's missing.

    Returns
    -------
    train_loader, test_loader : Tuple[DataLoader, DataLoader]
    """
    train_dataset = Flowers102Dataset(
        root=root, split="train", download=download,
    )
    test_dataset = Flowers102Dataset(
        root=root, split="test", download=download,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, test_loader




# ============================================================================
# Quick sanity check (run this file directly to verify the pipeline)
# ============================================================================
if __name__ == "__main__":
    print("=" * 50)
    print("EuroSAT")
    print("=" * 50)
    train_loader, test_loader = get_dataloaders(
        batch_size=4, num_workers=0, download=True,
    )

    images, labels, class_texts = next(iter(train_loader))
    print(f"Image batch shape : {images.shape}")      # (4, 3, 224, 224)
    print(f"Label batch       : {labels.tolist()}")    # e.g. [2, 7, 0, 5]
    print(f"Text prompts      : {class_texts}")        # list of 4 strings
    print(f"\nTrain samples: {len(train_loader.dataset)}")
    print(f"Test  samples: {len(test_loader.dataset)}")

    print(f"\n{'=' * 50}")
    print("DTD")
    print("=" * 50)
    dtd_train, dtd_test = get_dtd_dataloaders(
        batch_size=4, num_workers=0, download=True,
    )

    images, labels, class_texts = next(iter(dtd_train))
    print(f"Image batch shape : {images.shape}")
    print(f"Label batch       : {labels.tolist()}")
    print(f"Text prompts      : {class_texts}")
    print(f"\nTrain samples: {len(dtd_train.dataset)}")
    print(f"Test  samples: {len(dtd_test.dataset)}")

    print(f"\n{'=' * 50}")
    print("Flowers102")
    print("=" * 50)
    fl_train, fl_test = get_flowers102_dataloaders(
        batch_size=4, num_workers=0, download=True,
    )

    images, labels, class_texts = next(iter(fl_train))
    print(f"Image batch shape : {images.shape}")
    print(f"Label batch       : {labels.tolist()}")
    print(f"Text prompts      : {class_texts}")
    print(f"\nTrain samples: {len(fl_train.dataset)}")
    print(f"Test  samples: {len(fl_test.dataset)}")
