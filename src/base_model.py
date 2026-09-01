"""
base_model.py — Abstract CLIP Wrapper (Frozen Backbone)
========================================================

This module defines ``BaseCLIPWrapper``, the central abstraction for the entire
project.  It wraps a frozen OpenCLIP model and exposes two key methods:

    • get_image_features(images)  → pooled image embeddings
    • get_text_features(class_names) → pooled text embeddings

**Why this abstraction exists:**

Carlo and Marco will later build *adapter modules* that modify or augment
the features produced by CLIP's vision and text encoders, respectively.
By putting the feature-extraction logic behind virtual methods in a base
class, they can simply subclass ``BaseCLIPWrapper`` and override
``get_image_features`` / ``get_text_features`` without touching any
evaluation or inference code.

For example, Carlo might write:

    class VisionAdapterCLIP(BaseCLIPWrapper):
        def __init__(self, ...):
            super().__init__(...)
            self.adapter = nn.Sequential(...)   # lightweight adapter

        def get_image_features(self, images):
            # Get the standard pooled features from the frozen backbone…
            feats = super().get_image_features(images)
            # …then pass them through the trainable adapter.
            return self.adapter(feats)

This pattern keeps the codebase modular and avoids code duplication.
"""

import torch
import torch.nn as nn
import open_clip
from typing import List, Optional, Tuple


class BaseCLIPWrapper(nn.Module):
    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str = "cuda",
        class_names: Optional[List[str]] = None,
        prompt_template: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.device = device
        self.model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = self.model.to(device)

        # Store the tokenizer so we can convert text → token IDs on the fly.
        self.tokenizer = open_clip.get_tokenizer(model_name)

        # ----------------------------------------------------------------
        # Class set and prompt template.
        #
        # These live on the base class because ``predict()`` needs them:
        # in this project there is no classification head, the "classes"
        # *are* the text prototypes, so turning a class name into a prompt
        # is part of the classifier itself.
        #
        # They default to EuroSAT so that every existing call site keeps
        # working unchanged, but any subclass or caller can point them at
        # DTD or Flowers102 — whose templates are different and whose
        # accuracy collapses silently if the EuroSAT one is used instead
        # ("a satellite image of banded" for a texture class).
        #
        # The import is deferred to call time: ``dataset.py`` pulls in
        # torchvision, and this module must stay importable on its own.
        # ----------------------------------------------------------------
        if class_names is None or prompt_template is None:
            try:
                from src.dataset import EUROSAT_CLASS_NAMES, PROMPT_TEMPLATE
            except ModuleNotFoundError:
                from dataset import EUROSAT_CLASS_NAMES, PROMPT_TEMPLATE
            if class_names is None:
                class_names = EUROSAT_CLASS_NAMES
            if prompt_template is None:
                prompt_template = PROMPT_TEMPLATE

        self.class_names = list(class_names)
        self.prompt_template = prompt_template

        # ----------------------------------------------------------------
        # FREEZE all parameters of the pretrained CLIP model.
        # This is essential: we treat CLIP as a fixed feature extractor.
        # Only parameters added by subclasses (adapters, linear probes,
        # learnable prompts, etc.) should be trainable.
        # ----------------------------------------------------------------
        for param in self.model.parameters():
            param.requires_grad = False

        # Put the model in eval mode to disable dropout / batchnorm updates.
        self.model.eval()

    # ====================================================================
    # Feature-extraction methods
    # ====================================================================
    # These are the two methods that Carlo and Marco should override
    # in their adapter subclasses.  The default implementations below
    # simply return the standard pooled CLS-token features produced by
    # OpenCLIP.  Overriding them allows injecting adapter layers,
    # learned prompts, or any other modification without changing the
    # evaluation pipeline.
    # ====================================================================

    @torch.no_grad()
    def get_image_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of images into pooled feature vectors.

        Parameters
        ----------
        images : torch.Tensor
            Batch of preprocessed images, shape (B, 3, 224, 224).

        Returns
        -------
        image_features : torch.Tensor
            L2-normalized feature vectors, shape (B, D) where D is the
            embedding dimension of the CLIP model (e.g., 512 for ViT-B/32).

        Notes
        -----
        **For Carlo / Marco**: Override this method to insert your vision
        adapter between the frozen backbone and the returned features.
        Call ``super().get_image_features(images)`` to get the baseline
        features, then transform them through your adapter.
        """
        images = images.to(self.device)

        # ``self.model.encode_image`` runs the full vision transformer
        # and returns the pooled [CLS] token embedding.
        image_features = self.model.encode_image(images)

        # L2-normalize so that cosine similarity = dot product.
        # This is standard practice in contrastive learning and is
        # required for correct zero-shot classification.
        image_features = image_features / image_features.norm(
            dim=-1, keepdim=True
        )

        return image_features

    @torch.no_grad()
    def get_text_features(
        self, class_names: List[str]
    ) -> torch.Tensor:
        """
        Encode a list of text prompts into pooled feature vectors.

        Parameters
        ----------
        class_names : List[str]
            Human-readable text prompts, e.g.
            ["a satellite image of forest", "a satellite image of river"].

        Returns
        -------
        text_features : torch.Tensor
            L2-normalized feature vectors, shape (N, D) where N is the
            number of class names and D is the embedding dimension.

        Notes
        -----
        **For Carlo / Marco**: Override this method to insert your text
        adapter.  Call ``super().get_text_features(class_names)`` for the
        baseline embeddings, then transform them.
        """
        # Tokenize: converts strings → integer token IDs → pads to
        # the model's maximum context length (typically 77 tokens).
        tokens = self.tokenizer(class_names).to(self.device)

        # ``self.model.encode_text`` runs the text transformer and
        # returns the pooled [EOS] token embedding (analogous to [CLS]
        # on the vision side).
        text_features = self.model.encode_text(tokens)

        # L2-normalize for cosine similarity.
        text_features = text_features / text_features.norm(
            dim=-1, keepdim=True
        )

        return text_features

    # ====================================================================
    # Patch-level feature extraction (for Optimal Transport)
    # ====================================================================

    @torch.no_grad()
    def get_image_patch_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract intermediate patch tokens from CLIP's vision transformer.

        Parameters
        ----------
        images : torch.Tensor
            Batch of preprocessed images, shape (B, 3, 224, 224).

        Returns
        -------
        patch_tokens : torch.Tensor
            Shape (B, num_patches + 1, D_embed).
        """
        images = images.to(self.device)
        visual = self.model.visual

        # Steps 1-4: Patchify → class token → positional embeddings → LN.
        x = visual._embeds(images)           # (B, num_patches+1, D_hidden)

        # Step 5: Run through all transformer blocks.
        x = visual.transformer(x)            # (B, num_patches+1, D_hidden)

        # Step 6: Apply the final LayerNorm to ALL tokens.
        x = visual.ln_post(x)               # (B, num_patches+1, D_hidden)

        # Step 7: Apply the output projection (D_hidden → D_embed).
        if visual.proj is not None:
            x = x @ visual.proj              # (B, num_patches+1, D_embed)

        # L2-normalize each token independently.
        x = x / x.norm(dim=-1, keepdim=True)

        return x

    # ====================================================================
    # Shared inference: one predict() for the whole project
    # ====================================================================
    # The brief is explicit: "Every method plugs into the same base class
    # and the same evaluation loop.  No method reimplements evaluation."
    # Before this method existed, ``predict()`` was written five times —
    # three of them character-for-character identical — and the joint
    # CoOp + CLIP-Adapter model inherited two of them in conflict.
    # ====================================================================

    def build_prompts(self) -> List[str]:
        """
        Turn the stored class names into full text prompts.

        Returns
        -------
        prompts : List[str]
            e.g. ``["a satellite image of forest", ...]``.

        Notes
        -----
        Kept as a separate method so that a subclass which does not build
        its prompts from strings at all can override it — or, like
        ``CoOpModel``, simply ignore the result because its prompts are
        continuous vectors rather than text.
        """
        return [self.prompt_template.format(name) for name in self.class_names]

    @torch.no_grad()
    def predict(
        self, images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Classify a batch of images against the text prototypes.

        Parameters
        ----------
        images : torch.Tensor
            Batch of preprocessed images, shape (B, 3, 224, 224).

        Returns
        -------
        predictions : torch.Tensor
            ``(B,)`` predicted class indices.
        similarities : torch.Tensor
            ``(B, num_classes)`` cosine similarities.  ``engine.evaluate()``
            expects exactly this pair.

        Notes
        -----
        **The text prototypes are recomputed on every call, never cached.**
        This looks wasteful — it is ten short prompts through the text
        tower, negligible next to a batch of images — but it is what makes
        the method correct for *every* subclass.  ``CoOpModel`` changes its
        prompts after each optimizer step, so a cache built in ``__init__``
        would go stale mid-training and silently report the accuracy of the
        initial random context.

        Methods with a genuinely different scoring rule still override this:
        ``TipAdapterModel`` blends a key-value cache with the zero-shot
        logits, and ``OptimalTransportCLIP`` returns *distances* rather than
        similarities (which is why ``engine.evaluate()`` sniffs for
        ``sinkhorn_reg`` before taking a top-k).
        """
        image_features = self.get_image_features(images)
        text_features = self.get_text_features(self.build_prompts())

        similarities = image_features @ text_features.T
        return similarities.argmax(dim=-1), similarities

    # ====================================================================
    # Utility: count trainable parameters
    # ====================================================================
    def count_trainable_params(self) -> int:
        """
        Return the number of parameters with ``requires_grad=True``.

        For the base model this will be 0 (everything is frozen).
        Subclasses with adapters will report only the adapter parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# Quick smoke test
# ============================================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wrapper = BaseCLIPWrapper(device=device)

    # Dummy image batch
    dummy_images = torch.randn(2, 3, 224, 224)
    img_feats = wrapper.get_image_features(dummy_images)
    print(f"Image features shape: {img_feats.shape}")  # (2, 512)

    # Text features for all EuroSAT classes
    class_prompts = [
        "a satellite image of forest",
        "a satellite image of river",
    ]
    txt_feats = wrapper.get_text_features(class_prompts)
    print(f"Text features shape : {txt_feats.shape}")   # (2, 512)

    print(f"Trainable params    : {wrapper.count_trainable_params()}")  # 0

    # The shared predict(): every method in the project inherits this one
    # unless it has a genuinely different scoring rule.
    print(f"Default prompts     : {wrapper.build_prompts()[:2]}")
    preds, sims = wrapper.predict(dummy_images)
    print(f"predict() shapes    : preds {tuple(preds.shape)}, sims {tuple(sims.shape)}")

    # Same wrapper, different dataset: only the template and names change.
    dtd_wrapper = BaseCLIPWrapper(
        device=device,
        class_names=["banded", "bubbly", "cracked"],
        prompt_template="a photo of a {} texture",
    )
    print(f"DTD prompts         : {dtd_wrapper.build_prompts()}")
