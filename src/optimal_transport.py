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
import ot  # POT: Python Optimal Transport
from typing import List, Tuple, Optional

try:
    from src.base_model import BaseCLIPWrapper
    from src.dataset import EUROSAT_CLASS_NAMES, PROMPT_TEMPLATE
except ModuleNotFoundError:
    from base_model import BaseCLIPWrapper
    from dataset import EUROSAT_CLASS_NAMES, PROMPT_TEMPLATE



class OptimalTransportCLIP:
    """
    Classification via Sinkhorn Optimal Transport on token-level features.

    Instead of comparing single pooled vectors, we compare the *distribution*
    of image patch tokens against the *distribution* of text word tokens
    for each candidate class, and predict the class with the lowest
    transport cost.
    """

    def __init__(
        self,
        clip_wrapper: BaseCLIPWrapper,
        class_names: List[str] = EUROSAT_CLASS_NAMES,
        prompt_template: str = PROMPT_TEMPLATE,
        sinkhorn_reg: float = 0.05,
        sinkhorn_max_iter: int = 100,
    ) -> None:
        self.clip_wrapper = clip_wrapper
        self.device = clip_wrapper.device
        self.sinkhorn_reg = sinkhorn_reg
        self.sinkhorn_max_iter = sinkhorn_max_iter

        prompts = [prompt_template.format(name) for name in class_names]

        self.text_token_features = self._extract_text_tokens(prompts)

    # Token-level feature extraction
    @torch.no_grad()
    def _extract_image_patch_tokens(
        self, images: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract intermediate patch tokens from CLIP's vision transformer.

        Delegates to ``BaseCLIPWrapper.get_image_patch_tokens()`` to extract
        patch tokens.
        """
        return self.clip_wrapper.get_image_patch_tokens(images)

    @torch.no_grad()
    def _extract_text_tokens(
        self, prompts: List[str]
    ) -> List[torch.Tensor]:
        """
        Extract intermediate word-level tokens from CLIP's text transformer.
        """

        tokenizer = self.clip_wrapper.tokenizer
        model = self.clip_wrapper.model

        # Tokenize all prompts → (N, context_length) padded tensor.
        tokens = tokenizer(prompts).to(self.device)  # (N, 77)

        cast_dtype = model.transformer.get_cast_dtype()
        x = model.token_embedding(tokens).to(cast_dtype)  # (N, 77, D)
        x = x + model.positional_embedding.to(cast_dtype)  # (N, 77, D)
        attn_mask = model.attn_mask  # (77, 77) causal mask

        # Run through transformer (batch-first in open_clip >= 3.x).
        x = model.transformer(x, attn_mask=attn_mask)  # (N, L, D)

        # Step 3: Apply the final LayerNorm.
        x = model.ln_final(x)  # (N, 77, D)

        # Step 4: Apply the text projection matrix (if present).
        # This maps from transformer hidden dim to the shared embedding dim.
        if model.text_projection is not None:
            x = x @ model.text_projection  # (N, 77, D_embed)

        all_token_features = []
        for i in range(len(prompts)):
            # Find the indices of non-padding tokens.
            # tokens[i] is a 1D tensor of token IDs for prompt i.
            non_pad_mask = tokens[i] != 0  # (77,) boolean mask
            token_feats = x[i][non_pad_mask]  # (T_i, D_embed)

            # L2-normalize each token independently.
            token_feats = token_feats / token_feats.norm(
                dim=-1, keepdim=True
            )

            all_token_features.append(token_feats)

        return all_token_features

    # Sinkhorn Optimal Transport distance
    def _compute_sinkhorn_distance(
        self,
        image_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
    ) -> float:
        """
        Compute the Sinkhorn distance between one image's patch tokens
        and one class's text tokens.
        """
        # Move to CPU and convert to float64 for numerical stability.
        # The POT library works with numpy arrays.
        img_np = image_tokens.cpu().to(torch.float64).numpy()  # (P, D)
        txt_np = text_tokens.cpu().to(torch.float64).numpy()   # (T, D)

        # STEP 1: Build the cost matrix.
        cost_matrix = ot.dist(img_np, txt_np, metric="cosine")  # (P, T)

        # STEP 2: Define the marginal distributions.
        P = img_np.shape[0]  # number of image patches (e.g., 50)
        T = txt_np.shape[0]  # number of text tokens (varies by class)

        a = np.ones(P) / P  # uniform distribution over patches
        b = np.ones(T) / T  # uniform distribution over text tokens

        # STEP 3: Run the Sinkhorn algorithm.
        sinkhorn_distance = ot.sinkhorn2(
            a, b, cost_matrix,
            reg=self.sinkhorn_reg,
            numItermax=self.sinkhorn_max_iter,
        )

        # sinkhorn2 may return a numpy array with a single element
        # depending on the POT version.  Convert to a plain Python float.
        return float(sinkhorn_distance)

    # Prediction
    @torch.no_grad()
    def predict(
        self, images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict class labels by finding the class with the lowest
        Sinkhorn distance for each image.
        """
        # Step 1: Extract patch tokens for the entire batch.
        # patch_tokens shape: (B, num_patches+1, D), e.g. (B, 50, 512)
        patch_tokens = self._extract_image_patch_tokens(images)

        batch_size = patch_tokens.shape[0]
        num_classes = len(self.text_token_features)

        # Step 2: Compute Sinkhorn distance for each (image, class) pair.
        distances = torch.zeros(batch_size, num_classes)

        for i in range(batch_size):
            for j in range(num_classes):
                distances[i, j] = self._compute_sinkhorn_distance(
                    patch_tokens[i],          # (P, D) — patches for image i
                    self.text_token_features[j],  # (T_j, D) — tokens for class j
                )

        # Step 3: Predict the class with the *lowest* transport cost.
        # (In contrast, zero-shot CLIP predicts the class with the
        # *highest* cosine similarity.  The semantics are flipped because
        # Sinkhorn distance is a cost, not a similarity.)
        predictions = distances.argmin(dim=-1)

        return predictions, distances


# test
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_wrapper = BaseCLIPWrapper(device=device)

    ot_clip = OptimalTransportCLIP(clip_wrapper)

    # Dummy batch of 2 images
    dummy_images = torch.randn(2, 3, 224, 224)
    preds, dists = ot_clip.predict(dummy_images)
    print(f"Predictions: {preds.tolist()}")
    print(f"Distance matrix shape: {dists.shape}")
    print(f"Distances (image 0): {dists[0].tolist()}")
