"""
NOTE:
standard CLIP = visual features extracted by image encoder -> directly used for cosine similarity matching against text embeddings. 

CLIP-Adapter= insert lightweight trainable bottleneck MLP after frozen visual encoder.

Frozen CLIP backbone (requires_grad = False for all parameters)

Bottleneck MLP with configurable reduction ratio (e.g. 512 -> 128 -> 512)

Residual blending parameter alpha (usually 0.2)

Zero/Near-zero initialization for the adapter's output projection layer

Modular subclassing of `BaseCLIPWrapper` from `base_model.py`
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
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

try:
    from src.base_model import BaseCLIPWrapper
    from src.dataset import EUROSAT_CLASS_NAMES, PROMPT_TEMPLATE
except ModuleNotFoundError:
    from base_model import BaseCLIPWrapper
    from dataset import EUROSAT_CLASS_NAMES, PROMPT_TEMPLATE



class VisionAdapterModule(nn.Module):
    """
    Bottleneck MLP Adapter with Residual Blending for Vision Features.

    Params:
    embed_dim : int
        Input and output embedding dimension (e.g., 512 for ViT-B/32).
    reduction_ratio : int
        Bottleneck reduction factor. Hidden dim = embed_dim // reduction_ratio.
        Default is 4 (512 -> 128 -> 512).
    alpha : float
        Blending parameter between adapted features and original features.
        f_final = alpha * MLP(f_orig) + (1 - alpha) * f_orig
    """

    def __init__(
        self,
        embed_dim: int = 512,
        reduction_ratio: int = 4,
        alpha: float = 0.2,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.reduction_ratio = reduction_ratio
        self.hidden_dim = embed_dim // reduction_ratio
        self.alpha = alpha

        self.down_proj = nn.Linear(self.embed_dim, self.hidden_dim, bias=True)
        self.act = nn.ReLU()
        self.up_proj = nn.Linear(self.hidden_dim, self.embed_dim, bias=True)

        # Using near-zero weights initialization so that f_adapted ~= f_origin (keeps pretrained visual representations mostly intact in the first training steps)
        self._init_weights()


    def _init_weights(self) -> None:
        nn.init.kaiming_uniform_(self.down_proj.weight, a=0, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(self.down_proj.bias)

        # Initialize final layer weights and bias near zero
        nn.init.normal_(self.up_proj.weight, std=1e-4)
        nn.init.zeros_(self.up_proj.bias)


    def forward(self, f_orig: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for vision feature adaptation.

        Params:
        f_orig : torch.Tensor
            Original L2-normalized image features from frozen CLIP vision encoder, shape (B, D).

        Returns
        -------
        f_final : torch.Tensor
            Adapted and L2-normalized feature vectors, shape (B, D).
        """
        # Pass through bottleneck MLP
        f_mlp = self.act(self.down_proj(f_orig))
        f_adapter = self.up_proj(f_mlp)

        # Residual Blending: alpha * Adapter(f) + (1 - alpha) * f
        f_blended = self.alpha * f_adapter + (1.0 - self.alpha) * f_orig

        # L2-normalize final adapted features
        f_final = f_blended / f_blended.norm(dim=-1, keepdim=True)

        return f_final


class CLIPAdapterModel(BaseCLIPWrapper):
    """
    Subclass of BaseCLIPWrapper with Vision CLIP-Adapter.

    Overrides `get_image_features()` to route image embeddings through
    `VisionAdapterModule`. `get_text_features()` remains the frozen CLIP
    text encoding baseline.

    Params:
    model_name : str
        OpenCLIP model architecture, e.g. "ViT-B-32".
    pretrained : str
        Pretrained weights tag.
    device : str
        "cuda" or "cpu".
    reduction_ratio : int
        Bottleneck reduction ratio for the vision adapter.
    alpha : float
        Residual blending hyperparameter.
    """


    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str = "cuda",
        reduction_ratio: int = 4,
        alpha: float = 0.2,
        class_names: List[str] = EUROSAT_CLASS_NAMES,
    ) -> None:
        super().__init__(
            model_name=model_name,
            pretrained=pretrained,
            device=device,
        )

        self.reduction_ratio = reduction_ratio
        self.alpha = alpha
        self.class_names = class_names

        # Extract vision embedding dimension to perform a dummy pass or inspect visual projection to get exact dim
        embed_dim = self.model.visual.output_tokens if hasattr(self.model.visual, "output_tokens") else 512
        if hasattr(self.model, "visual") and hasattr(self.model.visual, "proj") and self.model.visual.proj is not None:
            embed_dim = self.model.visual.proj.shape[1]

        # Instanciate trainable Vision Adapter
        self.adapter = VisionAdapterModule(
            embed_dim=embed_dim,
            reduction_ratio=reduction_ratio,
            alpha=alpha,
        ).to(device)

        # Cache text prototypes for predict() evaluation loop
        self._update_text_prototypes()


    def _update_text_prototypes(self) -> None:
        """Cache text feature embeddings for EuroSAT classes."""
        prompts = [PROMPT_TEMPLATE.format(name) for name in self.class_names]
        with torch.no_grad():
            self.text_prototypes = self.get_text_features(prompts)


    def get_image_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract image features from frozen CLIP vision encoder, then pass them through the trainable VisionAdapterModule.

        Params:
        images : torch.Tensor
            Batch of preprocessed images, shape (B, 3, 224, 224).

        Return:
        adapted_image_features : torch.Tensor
            L2-normalized feature vectors, shape (B, D).
        """
        images = images.to(self.device)

        # Compute original features from frozen backbone without recording backbone gradients
        with torch.no_grad():
            orig_features = self.model.encode_image(images)
            orig_features = orig_features / orig_features.norm(dim=-1, keepdim=True)

        # Pass through trainable Vision Adapter (gradients enabled for adapter weights)
        adapted_features = self.adapter(orig_features)

        return adapted_features

    def set_alpha(self, new_alpha: float) -> None:
        """Dynamically update the residual blending factor alpha."""
        self.alpha = new_alpha
        self.adapter.alpha = new_alpha

    @torch.no_grad()
    def predict(
        self, images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict class labels for evaluation compatibility with `engine.evaluate()`

        Params:
        images : torch.Tensor
            Batch of images, shape (B, 3, 224, 224).

        Return:
        predictions : torch.Tensor
            Predicted class indices (B,).
        similarities : torch.Tensor
            Cosine similarity matrix (B, num_classes).
        """
        image_features = self.get_image_features(images)

        # Refresh text prototypes if needed
        if not hasattr(self, "text_prototypes") or self.text_prototypes is None:
            self._update_text_prototypes()

        similarities = image_features @ self.text_prototypes.T
        predictions = similarities.argmax(dim=-1)

        return predictions, similarities



# Test
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing CLIPAdapterModel on device: {device}")

    model = CLIPAdapterModel(
        model_name="ViT-B-32",
        pretrained="laion2b_s34b_b79k",
        device=device,
        reduction_ratio=4,
        alpha=0.2,
    )

    dummy_images = torch.randn(4, 3, 224, 224).to(device)
    features = model.get_image_features(dummy_images)
    print(f"Adapted Image Features shape: {features.shape}")  # (4, 512)
    print(f"Trainable Parameters count: {model.count_trainable_params():,}")

    preds, sims = model.predict(dummy_images)
    print(f"Predictions shape: {preds.shape}, Similarities shape: {sims.shape}")
