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
import torch.nn as nn
import numpy as np
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from typing import List, Tuple

try:
    from src.base_model import BaseCLIPWrapper
    from src.dataset import EUROSAT_CLASS_NAMES, PROMPT_TEMPLATE, EUROSAT_PROMPT_TEMPLATES
except ModuleNotFoundError:
    from base_model import BaseCLIPWrapper
    from dataset import EUROSAT_CLASS_NAMES, PROMPT_TEMPLATE, EUROSAT_PROMPT_TEMPLATES



# Baseline 1: Zero-Shot CLIP
class ZeroShotCLIP:
    """
    Zero-shot classification with CLIP.

    This is the simplest possible baseline — no training whatsoever.
    The idea is:
      1. Encode all class names through CLIP's text encoder → text prototypes.
      2. Encode each test image through CLIP's vision encoder → image embedding.
      3. Compute cosine similarity between the image embedding and every
         text prototype.
      4. Predict the class whose text prototype is most similar.

    This works because CLIP was trained with a contrastive objective that
    aligns images and their captions in a shared embedding space.
    """

    def __init__(
        self,
        clip_wrapper: BaseCLIPWrapper,
        class_names: List[str] = EUROSAT_CLASS_NAMES,
        prompt_template: str = PROMPT_TEMPLATE,
    ) -> None:
        self.clip_wrapper = clip_wrapper

        prompts = [prompt_template.format(name) for name in class_names]
        # text_prototypes shape: (num_classes, D)
        self.text_prototypes = clip_wrapper.get_text_features(prompts)

    @torch.no_grad()
    def predict(
        self, images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict class labels for a batch of images.
        """
    
        # 1: Encode images → (B, D), L2-normalized.
        image_features = self.clip_wrapper.get_image_features(images)

        # 2: Cosine similarity = dot product of L2-normalized vectors.
        similarities = image_features @ self.text_prototypes.T

        # 3: The predicted class is the one with highest similarity.
        predictions = similarities.argmax(dim=-1)

        return predictions, similarities


# Baseline 1b: Zero-Shot CLIP with Prompt Ensembling
class ZeroShotEnsembleCLIP:
    """
    Zero-shot classification with prompt ensembling.

    The procedure for each class c is:
      1. For each template t ∈ T, compute text_features(t.format(c)).
      2. Average all T feature vectors:  mean_feat = (1/|T|) Σ_t feat_t.
      3. L2-normalize the averaged vector.
    """

    def __init__(
        self,
        clip_wrapper: BaseCLIPWrapper,
        class_names: List[str] = EUROSAT_CLASS_NAMES,
        templates: List[str] = EUROSAT_PROMPT_TEMPLATES,
    ) -> None:
        self.clip_wrapper = clip_wrapper
        self.templates = templates

        # Build ensembled text prototypes.
        self.text_prototypes = self._build_ensemble_prototypes(class_names)

    @torch.no_grad()
    def _build_ensemble_prototypes(
        self, class_names: List[str]
    ) -> torch.Tensor:
        """
        Compute ensembled text prototypes by averaging across templates.
        """
        all_class_features = []

        for class_name in class_names:
            # Generate one prompt per template for this class.
            prompts = [t.format(class_name) for t in self.templates]

            # Encode all prompts → (num_templates, D), already L2-normed.
            features = self.clip_wrapper.get_text_features(prompts)

            # Average across templates → (D,)
            mean_feature = features.mean(dim=0)

            # Re-normalize after averaging (the mean of unit vectors is
            # generally NOT a unit vector).
            mean_feature = mean_feature / mean_feature.norm()

            all_class_features.append(mean_feature)

        # Stack into (num_classes, D).
        return torch.stack(all_class_features, dim=0)

    @torch.no_grad()
    def predict(
        self, images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict class labels using ensembled text prototypes.
        """
        image_features = self.clip_wrapper.get_image_features(images)
        similarities = image_features @ self.text_prototypes.T
        predictions = similarities.argmax(dim=-1)

        return predictions, similarities


# 2: Linear Probe CLIP
class LinearProbeCLIP:
    """
    Linear probe on frozen CLIP image features.
    """

    def __init__(
        self,
        clip_wrapper: BaseCLIPWrapper,
        C: float = 0.316,
        max_iter: int = 1000,
    ) -> None:
        self.clip_wrapper = clip_wrapper

        # Initialize the Logistic Regression classifier.
        self.classifier = LogisticRegression(
            C=C,
            max_iter=max_iter,
            solver="lbfgs",
            verbose=1,
        )

        # Will be set to True after fit() is called.
        self._is_fitted = False

    @torch.no_grad()
    def _extract_features(
        self, dataloader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract image features for an entire dataset.

        Iterates through the dataloader, encodes every image through the
        frozen CLIP vision encoder, and collects the results.
        """

        all_features = []
        all_labels = []

        for images, labels, _class_texts in dataloader:
            # Encode batch → (B, D) tensor on GPU.
            features = self.clip_wrapper.get_image_features(images)

            # Move to CPU and convert to numpy for scikit-learn.
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())

        # Concatenate all batches into single arrays.
        all_features = np.concatenate(all_features, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        return all_features, all_labels

    def fit(self, train_loader: DataLoader) -> None:
        """Extract training features and fit the linear classifier."""
        features, labels = self._extract_features(train_loader)
        self.classifier.fit(features, labels)
        self._is_fitted = True

    @torch.no_grad()
    def predict(
        self, images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict class labels for a batch of images.
        """

        if not self._is_fitted:
            raise RuntimeError(
                "LinearProbeCLIP has not been fitted yet. Call fit() first."
            )

        # Extract features using the frozen CLIP vision encoder.
        features = self.clip_wrapper.get_image_features(images)
        features_np = features.cpu().numpy()

        # Get predictions and probability estimates from scikit-learn.
        preds_np = self.classifier.predict(features_np)
        probs_np = self.classifier.predict_proba(features_np)

        # Convert back to PyTorch tensors for compatibility with the
        # evaluation engine (engine.py).
        predictions = torch.tensor(preds_np, dtype=torch.long)
        probabilities = torch.tensor(probs_np, dtype=torch.float32)

        return predictions, probabilities


# test
if __name__ == "__main__":
    try:
        from src.dataset import get_dataloaders
    except ModuleNotFoundError:
        from dataset import get_dataloaders

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_wrapper = BaseCLIPWrapper(device=device)

    # --- Zero-Shot ---
    zs = ZeroShotCLIP(clip_wrapper)
    dummy = torch.randn(2, 3, 224, 224)
    preds, sims = zs.predict(dummy)
    print(f"[ZeroShot] Predictions: {preds.tolist()}")
    print(f"[ZeroShot] Similarities shape: {sims.shape}")

    # --- Linear Probe ---
    train_loader, test_loader = get_dataloaders(
        batch_size=64, num_workers=0
    )
    lp = LinearProbeCLIP(clip_wrapper)
    lp.fit(train_loader)

    test_images, test_labels, _ = next(iter(test_loader))
    preds, probs = lp.predict(test_images)
    print(f"[LinearProbe] Predictions: {preds[:10].tolist()}")
