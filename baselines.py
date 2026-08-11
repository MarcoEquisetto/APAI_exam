"""
baselines.py — Zero-Shot & Linear Probe Baselines
===================================================

This module implements two standard baselines for evaluating CLIP on
downstream classification tasks:

1. **ZeroShotCLIP** — Pure inference, no training.
   Computes cosine similarity between image features and text features
   for every class, then predicts the class with the highest similarity.

2. **LinearProbeCLIP** — Minimal training on top of frozen features.
   Extracts image features for the entire training set using the frozen
   CLIP vision encoder, then fits a simple classifier (Logistic Regression
   from scikit-learn) on those features.

These two baselines represent the lower and upper bounds of what CLIP can
achieve without architectural modifications.  Any adapter that Carlo or
Marco builds should ideally outperform ZeroShot and approach or exceed
LinearProbe.
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from typing import List, Tuple

from base_model import BaseCLIPWrapper
from dataset import EUROSAT_CLASS_NAMES, PROMPT_TEMPLATE


# ============================================================================
# Baseline 1: Zero-Shot CLIP
# ============================================================================
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

    Parameters
    ----------
    clip_wrapper : BaseCLIPWrapper
        A (possibly subclassed) CLIP wrapper that provides
        ``get_image_features`` and ``get_text_features``.
    class_names : List[str]
        Raw class names (without the prompt template).
        The template is applied internally.
    """

    def __init__(
        self,
        clip_wrapper: BaseCLIPWrapper,
        class_names: List[str] = EUROSAT_CLASS_NAMES,
    ) -> None:
        self.clip_wrapper = clip_wrapper

        # ----------------------------------------------------------------
        # Build text prototypes once and cache them.
        # We apply the prompt template (e.g., "a satellite image of {}")
        # to each class name, then encode all prompts in a single forward
        # pass through the text encoder.
        # ----------------------------------------------------------------
        prompts = [PROMPT_TEMPLATE.format(name) for name in class_names]
        # text_prototypes shape: (num_classes, D)
        self.text_prototypes = clip_wrapper.get_text_features(prompts)

    @torch.no_grad()
    def predict(
        self, images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict class labels for a batch of images.

        Parameters
        ----------
        images : torch.Tensor
            Batch of preprocessed images, shape (B, 3, 224, 224).

        Returns
        -------
        predictions : torch.Tensor
            Predicted class indices, shape (B,).
        similarities : torch.Tensor
            Full similarity matrix, shape (B, num_classes).
            Useful for computing top-k accuracy downstream.
        """
        # Step 1: Encode images → (B, D), L2-normalized.
        image_features = self.clip_wrapper.get_image_features(images)

        # Step 2: Cosine similarity = dot product of L2-normalized vectors.
        # image_features: (B, D),  text_prototypes: (num_classes, D)
        # Result: (B, num_classes) — each entry is the cosine similarity
        # between one image and one class's text prototype.
        similarities = image_features @ self.text_prototypes.T

        # Step 3: The predicted class is the one with highest similarity.
        predictions = similarities.argmax(dim=-1)

        return predictions, similarities


# ============================================================================
# Baseline 2: Linear Probe CLIP
# ============================================================================
class LinearProbeCLIP:
    """
    Linear probe on frozen CLIP image features.

    This baseline:
      1. Extracts image features for every sample in the training set
         using the frozen CLIP vision encoder (one-time cost).
      2. Fits a Logistic Regression classifier (scikit-learn) on those
         features.
      3. At test time, extracts image features and runs them through the
         trained classifier.

    Why Logistic Regression instead of nn.Linear?
    - It's simpler: no need for a training loop, optimizer, or scheduler.
    - scikit-learn's solver handles regularization automatically.
    - It's the standard protocol used in the original CLIP paper
      (Radford et al., 2021) for linear probe evaluation.

    Parameters
    ----------
    clip_wrapper : BaseCLIPWrapper
        Frozen CLIP model for feature extraction.
    C : float
        Regularization strength for Logistic Regression.
        Smaller values → stronger regularization.  Default 0.316 is
        the value used in the CLIP paper.
    max_iter : int
        Maximum iterations for the L-BFGS solver.
    """

    def __init__(
        self,
        clip_wrapper: BaseCLIPWrapper,
        C: float = 0.316,
        max_iter: int = 1000,
    ) -> None:
        self.clip_wrapper = clip_wrapper

        # ----------------------------------------------------------------
        # Initialize the Logistic Regression classifier.
        # - solver="lbfgs" is a quasi-Newton method that works well for
        #   small-to-medium datasets (EuroSAT has ~27k samples).
        # - Multinomial (softmax) multi-class is the default in modern
        #   scikit-learn versions.
        # ----------------------------------------------------------------
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

        Returns
        -------
        all_features : np.ndarray, shape (N, D)
            Pooled image features for every sample.
        all_labels : np.ndarray, shape (N,)
            Corresponding integer labels.
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

        Parameters
        ----------
        images : torch.Tensor
            Batch of preprocessed images, shape (B, 3, 224, 224).

        Returns
        -------
        predictions : torch.Tensor
            Predicted class indices, shape (B,).
        probabilities : torch.Tensor
            Class probability distribution, shape (B, num_classes).
            These come from the Logistic Regression's softmax output.
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


# ============================================================================
# Quick smoke test
# ============================================================================
if __name__ == "__main__":
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
