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
import math
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
    learnable_alpha : bool
        False (default) -> alpha is a fixed hyperparameter, as in the
        CLIP-Adapter paper and as required by the alpha sweep.
        True -> alpha is trained together with the MLP.
    constrain_alpha : bool
        Only meaningful when learnable_alpha is True. Keeps alpha inside
        (0, 1) by learning its logit and applying a sigmoid, so the residual
        blend stays an interpolation. Set False to reproduce older runs
        that trained an unconstrained alpha.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        reduction_ratio: int = 4,
        alpha: float = 0.2,
        learnable_alpha: bool = False,
        constrain_alpha: bool = True,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.reduction_ratio = reduction_ratio
        self.hidden_dim = embed_dim // reduction_ratio
        self.learnable_alpha = learnable_alpha
        self.constrain_alpha = constrain_alpha

        # ------------------------------------------------------------------
        # Alpha: fixed hyperparameter or learned scalar?
        #
        # Both are legitimate experiments and the project reports them as
        # two separate ones.  They cannot be merged: an alpha sweep only
        # means something if alpha stays where it is put, and the sweep
        # numbers in the results tables were produced with a fixed alpha.
        #
        # When learned, alpha is stored as a *logit* and squashed through a
        # sigmoid in forward().  The residual blend
        #     f = a * MLP(f) + (1 - a) * f
        # is an interpolation only for a in [0, 1]; an unconstrained
        # parameter is free to leave that range and turn the blend into an
        # extrapolation, which is a different model from the one the brief
        # describes.  Parameterizing the logit keeps a in (0, 1) by
        # construction, with no clamping and no gradient discontinuity.
        # ------------------------------------------------------------------
        if learnable_alpha:
            if constrain_alpha:
                a = float(min(max(alpha, 1e-4), 1.0 - 1e-4))
                logit = math.log(a / (1.0 - a))
                self.alpha_logit = nn.Parameter(torch.tensor(logit))
            else:
                # Legacy behaviour: a raw, unconstrained parameter.  Kept so
                # that runs recorded before the constraint existed remain
                # reproducible.
                self.alpha_raw = nn.Parameter(torch.tensor(float(alpha)))
        else:
            # Plain Python float: contributes nothing to
            # count_trainable_params(), which keeps the parameter counts in
            # the "Accuracy vs. Trainable Parameters" plot honest.
            self._alpha_value = float(alpha)

        self.down_proj = nn.Linear(self.embed_dim, self.hidden_dim, bias=True)
        self.act = nn.ReLU()
        self.up_proj = nn.Linear(self.hidden_dim, self.embed_dim, bias=True)

        # Using near-zero weights initialization so that f_adapted ~= f_origin (keeps pretrained visual representations mostly intact in the first training steps)
        self._init_weights()


    @property
    def alpha(self):
        """
        The effective blending factor, whatever the storage mode.

        Returns a 0-dim tensor when alpha is learned (so gradients flow) and
        a plain float when it is fixed. Both broadcast correctly in
        forward(); float(module.alpha) works in either case, which is what
        logging and the experiment scripts want.
        """
        if not self.learnable_alpha:
            return self._alpha_value
        if self.constrain_alpha:
            return torch.sigmoid(self.alpha_logit)
        return self.alpha_raw

    @alpha.setter
    def alpha(self, new_alpha: float) -> None:
        """Set alpha in place, in whichever representation is active."""
        if not self.learnable_alpha:
            self._alpha_value = float(new_alpha)
            return
        with torch.no_grad():
            if self.constrain_alpha:
                a = float(min(max(new_alpha, 1e-4), 1.0 - 1e-4))
                self.alpha_logit.fill_(math.log(a / (1.0 - a)))
            else:
                self.alpha_raw.fill_(float(new_alpha))

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
    learnable_alpha : bool
        Train alpha alongside the MLP instead of holding it fixed.
        Default False: the alpha sweep is only meaningful with a fixed
        alpha, so "fixed alpha" and "learned alpha" stay two distinct
        experiments.
    constrain_alpha : bool
        Keep a learned alpha inside (0, 1) via a sigmoid. Ignored when
        learnable_alpha is False.
    class_names : List[str]
        Raw class names, e.g. ["forest", "river", ...].
    prompt_template : str
        Template wrapped around each class name. Defaults to the EuroSAT
        one; pass the dataset's own template (available as
        `loader.dataset.prompt_template`) when running on DTD or
        Flowers102, otherwise the prompts are nonsense and accuracy drops
        with no error.
    """


    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str = "cuda",
        reduction_ratio: int = 4,
        alpha: float = 0.2,
        learnable_alpha: bool = False,
        constrain_alpha: bool = True,
        class_names: List[str] = EUROSAT_CLASS_NAMES,
        prompt_template: str = PROMPT_TEMPLATE,
    ) -> None:
        super().__init__(
            model_name=model_name,
            pretrained=pretrained,
            device=device,
            class_names=class_names,
            prompt_template=prompt_template,
        )

        self.reduction_ratio = reduction_ratio

        # Extract vision embedding dimension to perform a dummy pass or inspect visual projection to get exact dim
        embed_dim = self.model.visual.output_tokens if hasattr(self.model.visual, "output_tokens") else 512
        if hasattr(self.model, "visual") and hasattr(self.model.visual, "proj") and self.model.visual.proj is not None:
            embed_dim = self.model.visual.proj.shape[1]

        # Instanciate trainable Vision Adapter
        self.adapter = VisionAdapterModule(
            embed_dim=embed_dim,
            reduction_ratio=reduction_ratio,
            alpha=alpha,
            learnable_alpha=learnable_alpha,
            constrain_alpha=constrain_alpha,
        ).to(device)

        # Cache text prototypes for anyone reading them directly.  Note that
        # predict() no longer uses this cache: it is inherited from
        # BaseCLIPWrapper, which recomputes the prototypes on every call so
        # that the same method also works for models whose text side is
        # being trained (CoOp, and the joint CoOp + adapter model).
        self._update_text_prototypes()


    @property
    def alpha(self):
        """Effective blending factor, read straight from the adapter."""
        return self.adapter.alpha


    def _update_text_prototypes(self) -> None:
        """Cache text feature embeddings for the model's classes."""
        with torch.no_grad():
            self.text_prototypes = self.get_text_features(self.build_prompts())


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
        self.adapter.alpha = new_alpha

    # predict() is deliberately NOT defined here.
    #
    # It used to be a verbatim copy of the base-class implementation, and
    # the joint CoOp + CLIP-Adapter model ended up inheriting two rival
    # copies of the same three lines.  BaseCLIPWrapper.predict() now does
    # the job for every method whose scoring rule is "cosine similarity
    # against the text prototypes" — which is this one.
    #
    # The one behavioural difference is that the base version recomputes
    # the text prototypes on each call instead of reading the cache built
    # in __init__.  That costs ten short prompts through the text tower and
    # it is what makes the same method correct when the text side is also
    # being trained.



class TipAdapterModel(BaseCLIPWrapper):
    """
    Training-Free Tip-Adapter for CLIP.

    Constructs a key-value cache from a few-shot support set and blends its predictions with zero-shot predictions to improve accuracy without any gradient updates.
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str = "cuda",
        alpha: float = 1.0,
        beta: float = 5.5,
        class_names: List[str] = EUROSAT_CLASS_NAMES,
        prompt_template: str = PROMPT_TEMPLATE,
    ) -> None:
        super().__init__(
            model_name=model_name,
            pretrained=pretrained,
            device=device,
            class_names=class_names,
            prompt_template=prompt_template,
        )
        self.alpha = alpha
        self.beta = beta

        self.cache_keys = None   # F_train
        self.cache_values = None # L_train (one-hot)

        self._update_text_prototypes()

    def _update_text_prototypes(self) -> None:
        """Cache text feature embeddings for the model's classes."""
        with torch.no_grad():
            self.text_prototypes = self.get_text_features(self.build_prompts())

    @torch.no_grad()
    def build_cache(self, dataloader, num_shots: int = 16) -> None:
        """
        Build the few-shot key-value cache.
        """
        self.model.eval()
        keys_list = []
        values_list = []

        class_counts = {i: 0 for i in range(len(self.class_names))}

        for images, labels, _ in dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            features = self.model.encode_image(images)
            features = features / features.norm(dim=-1, keepdim=True)

            for i in range(images.size(0)):
                lbl = labels[i].item()
                if class_counts[lbl] < num_shots:
                    keys_list.append(features[i].unsqueeze(0))
                    one_hot = torch.zeros(len(self.class_names), device=self.device)
                    one_hot[lbl] = 1.0
                    values_list.append(one_hot.unsqueeze(0))
                    class_counts[lbl] += 1

            if all(count >= num_shots for count in class_counts.values()):
                break

        self.cache_keys = torch.cat(keys_list, dim=0) # (N, D)
        self.cache_values = torch.cat(values_list, dim=0) # (N, C)
        print(f"Tip-Adapter cache built: keys {self.cache_keys.shape}, values {self.cache_values.shape}")

    def finetune_cache(self, dataloader, epochs: int = 20, lr: float = 1e-3) -> None:
        """
        Fine-tune the key cache (Tip-Adapter-F).
        Transforms the cache into learnable parameters and optimizes them using Cross Entropy Loss.
        """
        if self.cache_keys is None:
            raise ValueError("Cache not built. Call build_cache() first.")
        
        self.cache_keys = nn.Parameter(self.cache_keys.clone())
        optimizer = torch.optim.AdamW([self.cache_keys], lr=lr)
        
        self.model.eval()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for images, labels, _ in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                with torch.no_grad():
                    test_features = self.model.encode_image(images)
                    test_features = test_features / test_features.norm(dim=-1, keepdim=True)
                    clip_logits = 100.0 * test_features @ self.text_prototypes.T
                
                affinity = test_features @ self.cache_keys.T
                cache_logits = ((-1) * (self.beta - self.beta * affinity)).exp() @ self.cache_values
                logits = clip_logits + cache_logits * self.alpha
                
                loss = F.cross_entropy(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            # print(f"Tip-Adapter-F Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloader):.4f}")

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict class labels using Tip-Adapter formula.
        """
        images = images.to(self.device)

        test_features = self.model.encode_image(images)
        test_features = test_features / test_features.norm(dim=-1, keepdim=True)

        # Zero-shot predictions (B, C)
        clip_logits = 100.0 * test_features @ self.text_prototypes.T

        if self.cache_keys is not None and self.cache_values is not None:
            # Affinity (B, N)
            affinity = test_features @ self.cache_keys.T
            # Tip-Adapter activation
            cache_logits = ((-1) * (self.beta - self.beta * affinity)).exp() @ self.cache_values

            # Combine
            logits = clip_logits + cache_logits * self.alpha
        else:
            logits = clip_logits

        predictions = logits.argmax(dim=-1)
        return predictions, logits


class LinearLoRA(nn.Module):
    """
    Wraps an existing nn.Linear layer to inject Low-Rank Adaptation (LoRA) matrices.
    """
    def __init__(self, linear_layer: nn.Linear, r: int = 4, alpha: float = 1.0):
        super().__init__()
        self.linear = linear_layer # The frozen pretrained layer
        self.r = r
        self.alpha = alpha
        
        in_features = self.linear.in_features
        out_features = self.linear.out_features
        
        self.lora_A = nn.Parameter(torch.zeros(in_features, r))
        self.lora_B = nn.Parameter(torch.zeros(r, out_features))
        self.scaling = alpha / r
        
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        lora_out = (x @ self.lora_A @ self.lora_B) * self.scaling
        return out + lora_out


class VisionLoRAModel(BaseCLIPWrapper):
    """
    Subclass of BaseCLIPWrapper with Low-Rank Adaptation (LoRA) injected into the Vision Transformer.
    """
    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str = "cuda",
        r: int = 4,
        lora_alpha: float = 1.0,
        class_names: List[str] = EUROSAT_CLASS_NAMES,
        prompt_template: str = PROMPT_TEMPLATE,
    ) -> None:
        super().__init__(
            model_name=model_name,
            pretrained=pretrained,
            device=device,
            class_names=class_names,
            prompt_template=prompt_template,
        )
        self.r = r
        self.lora_alpha = lora_alpha

        # Inject LoRA into the vision transformer's MLP layers
        for block in self.model.visual.transformer.resblocks:
            if hasattr(block.mlp, "c_fc") and isinstance(block.mlp.c_fc, nn.Linear):
                block.mlp.c_fc = LinearLoRA(block.mlp.c_fc, r=r, alpha=lora_alpha).to(device)
            if hasattr(block.mlp, "c_proj") and isinstance(block.mlp.c_proj, nn.Linear):
                block.mlp.c_proj = LinearLoRA(block.mlp.c_proj, r=r, alpha=lora_alpha).to(device)
                
        self._update_text_prototypes()

    def _update_text_prototypes(self) -> None:
        with torch.no_grad():
            self.text_prototypes = self.get_text_features(self.build_prompts())

    def get_image_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract image features allowing gradients to flow through LoRA layers."""
        images = images.to(self.device)
        # No torch.no_grad() here to allow gradients for LoRA parameters
        image_features = self.model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    # predict() inherited from BaseCLIPWrapper: same scoring rule, same
    # three lines.  See the note in CLIPAdapterModel.


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
    owner = next(k.__name__ for k in type(model).__mro__ if "predict" in k.__dict__)
    print(f"predict() inherited from: {owner}")   # expected: BaseCLIPWrapper
    print(f"alpha (fixed): {float(model.alpha):.4f} | set_alpha(0.5) -> ", end="")
    model.set_alpha(0.5)
    print(f"{float(model.alpha):.4f}")

    # Same adapter, alpha now trained and constrained to (0, 1).
    # Note the parameter count: +1 compared to the fixed-alpha variant.
    learned = CLIPAdapterModel(device=device, reduction_ratio=4, alpha=0.2,
                               learnable_alpha=True)
    print(f"Learnable alpha: start {float(learned.alpha):.4f} | "
          f"trainable {learned.count_trainable_params():,} "
          f"(fixed-alpha variant: {model.count_trainable_params():,})")

    # Different dataset: only the class names and the template change.
    dtd = CLIPAdapterModel(device=device,
                           class_names=["banded", "bubbly", "cracked"],
                           prompt_template="a photo of a {} texture")
    print(f"DTD prompts: {dtd.build_prompts()}")

    print(f"\nTesting TipAdapterModel on device: {device}")
    tip_model = TipAdapterModel(
        model_name="ViT-B-32",
        pretrained="laion2b_s34b_b79k",
        device=device,
    )
    # mock dataloader with 1 batch
    dummy_labels = torch.randint(0, 10, (4,))
    tip_model.build_cache([(dummy_images, dummy_labels, None)], num_shots=1)
    preds, sims = tip_model.predict(dummy_images)
    print(f"Tip-Adapter Predictions shape: {preds.shape}, Similarities shape: {sims.shape}")

    print(f"\nTesting VisionLoRAModel on device: {device}")
    lora_model = VisionLoRAModel(
        model_name="ViT-B-32",
        pretrained="laion2b_s34b_b79k",
        device=device,
        r=4,
    )
    features = lora_model.get_image_features(dummy_images)
    print(f"LoRA Adapted Image Features shape: {features.shape}")
    print(f"Trainable Parameters count: {lora_model.count_trainable_params():,}")
    preds, sims = lora_model.predict(dummy_images)
    print(f"LoRA Predictions shape: {preds.shape}, Similarities shape: {sims.shape}")
