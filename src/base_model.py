"""
base_model.py — Abstract CLIP Wrapper (Frozen Backbone)
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

        for param in self.model.parameters():
            param.requires_grad = False

        # Put the model in eval mode to disable dropout / batchnorm updates.
        self.model.eval()


    @torch.no_grad()
    def get_image_features(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(self.device)

        # ``self.model.encode_image`` runs the full vision transformer
        # and returns the pooled [CLS] token embedding.
        image_features = self.model.encode_image(images)

        # L2-normalize so that cosine similarity = dot product... standard practice
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
        """

        # Tokenize: converts strings → integer token IDs → pads to
        # the model's maximum context length (typically 77 tokens).
        tokens = self.tokenizer(class_names).to(self.device)

        text_features = self.model.encode_text(tokens)

        # L2-normalize for cosine similarity.
        text_features = text_features / text_features.norm(
            dim=-1, keepdim=True
        )

        return text_features

    @torch.no_grad()
    def get_image_patch_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract intermediate patch tokens from CLIP's vision transformer.
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

    def build_prompts(self) -> List[str]:
        """
        Turn the stored class names into full text prompts.
        """
        return [self.prompt_template.format(name) for name in self.class_names]

    @torch.no_grad()
    def predict(
        self, images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Classify a batch of images against the text prototypes.
        """
        image_features = self.get_image_features(images)
        text_features = self.get_text_features(self.build_prompts())

        similarities = image_features @ text_features.T
        return similarities.argmax(dim=-1), similarities

    def count_trainable_params(self) -> int:
        """
        Return the number of parameters with ``requires_grad=True``.

        For the base model this will be 0 (everything is frozen).
        Subclasses with adapters will report only the adapter parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# test
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
