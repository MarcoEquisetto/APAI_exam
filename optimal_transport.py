"""
optimal_transport.py — Sinkhorn-Based CLIP Classification
==========================================================

This module implements a fine-grained classification method using
*Optimal Transport* (OT) between CLIP's intermediate token representations.

**Core Idea:**

Standard CLIP zero-shot classification uses the *pooled* [CLS] token from
the vision encoder and the *pooled* [EOS] token from the text encoder.
These are single-vector summaries of the entire image / text.

But a ViT-B/32 actually produces 50 patch tokens (7×7 grid + 1 CLS token),
and the text encoder produces one token per sub-word.  These intermediate
tokens carry rich *local* information that the pooled vector throws away.

Optimal Transport lets us compute a structured distance between the full
set of image patch tokens and the full set of text tokens, effectively
asking: "What is the minimum cost of transporting the mass of image patches
to match the text token distribution?"

We use the **Sinkhorn algorithm** (an entropy-regularized variant of the
classical Wasserstein / Earth Mover's Distance) because:
  1. It is differentiable (useful if we ever want to backprop through it).
  2. It is much faster than exact OT solvers (O(n² / ε) vs. O(n³ log n)).
  3. The regularization parameter ε controls the smoothness of the
     transport plan, acting as a natural temperature.

**Reference:**
  Cuturi, M. (2013). "Sinkhorn Distances: Lightspeed Computation of
  Optimal Transport." NeurIPS.
"""

import torch
import torch.nn as nn
import numpy as np
import ot  # POT: Python Optimal Transport
from typing import List, Tuple, Optional

from base_model import BaseCLIPWrapper
from dataset import EUROSAT_CLASS_NAMES, PROMPT_TEMPLATE


class OptimalTransportCLIP:
    """
    Classification via Sinkhorn Optimal Transport on token-level features.

    Instead of comparing single pooled vectors, we compare the *distribution*
    of image patch tokens against the *distribution* of text word tokens
    for each candidate class, and predict the class with the lowest
    transport cost.

    Parameters
    ----------
    clip_wrapper : BaseCLIPWrapper
        Frozen CLIP model.  We access its internal layers to extract
        intermediate token representations.
    class_names : List[str]
        Raw EuroSAT class names (prompt template applied internally).
    sinkhorn_reg : float
        Entropy regularization parameter (ε) for the Sinkhorn algorithm.
        - Smaller ε → solution closer to true Wasserstein distance, but
          slower convergence and potential numerical instability.
        - Larger ε  → smoother transport plan, faster convergence, but
          a less precise distance.
        Default 0.05 is a good balance for 512-dim CLIP features.
    sinkhorn_max_iter : int
        Maximum number of Sinkhorn iterations.  100 is usually enough
        for convergence with ε ≥ 0.01.
    """

    def __init__(
        self,
        clip_wrapper: BaseCLIPWrapper,
        class_names: List[str] = EUROSAT_CLASS_NAMES,
        sinkhorn_reg: float = 0.05,
        sinkhorn_max_iter: int = 100,
    ) -> None:
        self.clip_wrapper = clip_wrapper
        self.device = clip_wrapper.device
        self.sinkhorn_reg = sinkhorn_reg
        self.sinkhorn_max_iter = sinkhorn_max_iter

        # ----------------------------------------------------------------
        # Pre-compute text token features for all classes.
        # We cache these because the class set is fixed; recomputing
        # them for every image would be wasteful.
        # ----------------------------------------------------------------
        prompts = [PROMPT_TEMPLATE.format(name) for name in class_names]
        # text_token_features is a list of tensors, one per class.
        # Each tensor has shape (T_c, D) where T_c is the number of
        # non-padding tokens for class c.
        self.text_token_features = self._extract_text_tokens(prompts)

    # ====================================================================
    # Token-level feature extraction
    # ====================================================================

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

        In CLIP's text encoder:
          1. Text is tokenized into sub-word tokens (BPE).
          2. Token embeddings + positional embeddings are added.
          3. The sequence is passed through L transformer blocks.
          4. The [EOS] token's output is used as the pooled representation.

        Here we extract ALL token outputs (not just [EOS]) and discard
        the padding tokens, keeping only meaningful word tokens.

        Parameters
        ----------
        prompts : List[str]
            Text prompts, e.g. ["a satellite image of forest", ...].

        Returns
        -------
        all_token_features : List[torch.Tensor]
            One tensor per prompt.  Each has shape (T_i, D) where T_i
            is the number of non-padding tokens in prompt i.
        """
        tokenizer = self.clip_wrapper.tokenizer
        model = self.clip_wrapper.model

        # Tokenize all prompts → (N, context_length) padded tensor.
        tokens = tokenizer(prompts).to(self.device)  # (N, 77)

        # ----------------------------------------------------------------
        # Run the text transformer manually to get all token outputs.
        # We replicate the first two steps of encode_text():
        #   1. Token embedding lookup
        #   2. Positional embedding addition
        # The causal attention mask is stored as model.attn_mask.
        # ----------------------------------------------------------------
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

        # ----------------------------------------------------------------
        # Step 5: Remove padding tokens.
        # Padding tokens have token ID = 0.  We keep only the tokens
        # that correspond to actual sub-words (and the special [SOS]/[EOS]
        # tokens).  This is important because padding tokens would dilute
        # the optimal transport computation.
        # ----------------------------------------------------------------
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

    # ====================================================================
    # Sinkhorn Optimal Transport distance
    # ====================================================================

    def _compute_sinkhorn_distance(
        self,
        image_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
    ) -> float:
        """
        Compute the Sinkhorn distance between one image's patch tokens
        and one class's text tokens.

        **How the cost matrix is built:**

        We have:
          - image_tokens: (P, D) — P patch token embeddings from the image.
          - text_tokens:  (T, D) — T word token embeddings from the text.

        The cost matrix C has shape (P, T), where:
          C[i, j] = 1 - cos_sim(image_token_i, text_token_j)

        We use cosine distance (= 1 − cosine similarity) because:
          1. CLIP features are L2-normalized, so cos_sim = dot product.
          2. Cosine distance is non-negative and equals 0 when tokens
             are identical, which satisfies the requirements for an OT
             cost function.
          3. It matches CLIP's training objective (contrastive loss on
             cosine similarity).

        **How Sinkhorn is applied:**

        Given the cost matrix C, we define two uniform distributions:
          - a = (1/P, 1/P, ..., 1/P)  over image patches
          - b = (1/T, 1/T, ..., 1/T)  over text tokens

        The Sinkhorn algorithm finds the optimal transport plan γ* that
        minimizes:

            Σ_ij  γ_ij · C_ij  +  ε · KL(γ || a ⊗ b)

        where:
          - γ_ij is the amount of "mass" transported from patch i to
            text token j.
          - The first term is the total transport cost.
          - The second term (KL divergence) is the entropic regularization
            that makes the optimization tractable.
          - ε (self.sinkhorn_reg) controls the trade-off: smaller ε gives
            a sharper (more Wasserstein-like) distance; larger ε gives a
            smoother solution.

        The optimal transport cost (Sinkhorn distance) is then:

            d_Sinkhorn = Σ_ij  γ*_ij · C_ij

        A lower Sinkhorn distance means the image patches can be
        "cheaply" aligned with the text tokens, suggesting that the
        image belongs to that class.

        Parameters
        ----------
        image_tokens : torch.Tensor
            Patch token features for a single image, shape (P, D).
        text_tokens : torch.Tensor
            Word token features for a single class, shape (T, D).

        Returns
        -------
        distance : float
            The Sinkhorn distance (lower = better match).
        """
        # Move to CPU and convert to float64 for numerical stability.
        # The POT library works with numpy arrays.
        img_np = image_tokens.cpu().to(torch.float64).numpy()  # (P, D)
        txt_np = text_tokens.cpu().to(torch.float64).numpy()   # (T, D)

        # ----------------------------------------------------------------
        # STEP 1: Build the cost matrix.
        #
        # ot.dist() computes pairwise distances.  With metric="cosine",
        # it computes:  C[i,j] = 1 - (img_np[i] · txt_np[j]) / (‖img_np[i]‖ · ‖txt_np[j]‖)
        #
        # Since we've already L2-normalized all tokens, this simplifies
        # to:  C[i,j] = 1 - img_np[i] · txt_np[j]
        #
        # The resulting matrix has shape (P, T) with values in [0, 2].
        # Values near 0 mean the patch and word token are very similar;
        # values near 2 mean they point in opposite directions.
        # ----------------------------------------------------------------
        cost_matrix = ot.dist(img_np, txt_np, metric="cosine")  # (P, T)

        # ----------------------------------------------------------------
        # STEP 2: Define the marginal distributions.
        #
        # We use uniform distributions:
        #   a[i] = 1/P  for each image patch  (every patch matters equally)
        #   b[j] = 1/T  for each text token   (every word matters equally)
        #
        # This means we treat the image as a uniform distribution over its
        # patches and the text as a uniform distribution over its tokens.
        # Alternative: weight patches by attention scores, or weight text
        # tokens by TF-IDF.  Uniform is the simplest starting point.
        # ----------------------------------------------------------------
        P = img_np.shape[0]  # number of image patches (e.g., 50)
        T = txt_np.shape[0]  # number of text tokens (varies by class)

        a = np.ones(P) / P  # uniform distribution over patches
        b = np.ones(T) / T  # uniform distribution over text tokens

        # ----------------------------------------------------------------
        # STEP 3: Run the Sinkhorn algorithm.
        #
        # ot.sinkhorn2() computes the *Sinkhorn divergence* (a scalar):
        #   d = <γ*, C>  = Σ_ij γ*_ij · C_ij
        #
        # where γ* is the entropy-regularized optimal transport plan.
        #
        # Arguments:
        #   a           — source marginal (uniform over patches)
        #   b           — target marginal (uniform over tokens)
        #   cost_matrix — pairwise cosine distances
        #   reg         — ε, the entropic regularization strength
        #   numItermax  — maximum Sinkhorn iterations
        #
        # Note: ot.sinkhorn2 returns the *transport cost* as a scalar.
        #       ot.sinkhorn  returns the *transport plan* (matrix γ*).
        #       We only need the scalar cost for classification.
        # ----------------------------------------------------------------
        sinkhorn_distance = ot.sinkhorn2(
            a, b, cost_matrix,
            reg=self.sinkhorn_reg,
            numItermax=self.sinkhorn_max_iter,
        )

        # sinkhorn2 may return a numpy array with a single element
        # depending on the POT version.  Convert to a plain Python float.
        return float(sinkhorn_distance)

    # ====================================================================
    # Prediction
    # ====================================================================

    @torch.no_grad()
    def predict(
        self, images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict class labels by finding the class with the lowest
        Sinkhorn distance for each image.

        For each image:
          1. Extract all 50 patch tokens from the vision transformer.
          2. Compute the Sinkhorn distance to every class's text tokens.
          3. Predict the class with the *smallest* distance.

        Parameters
        ----------
        images : torch.Tensor
            Batch of images, shape (B, 3, 224, 224).

        Returns
        -------
        predictions : torch.Tensor
            Predicted class indices, shape (B,).
        distances : torch.Tensor
            Full distance matrix, shape (B, num_classes).
            Lower values → better match.  To convert to "similarities"
            for top-k accuracy, negate these values.
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


# ============================================================================
# Quick smoke test
# ============================================================================
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
