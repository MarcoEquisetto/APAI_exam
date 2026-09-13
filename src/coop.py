import sys
from pathlib import Path

FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parent
if str(FILE_DIR) not in sys.path:
    sys.path.insert(0, str(FILE_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from typing import List, Optional

try:
    from src.base_model import BaseCLIPWrapper
    from src.dataset import EUROSAT_CLASS_NAMES
except ModuleNotFoundError:
    from base_model import BaseCLIPWrapper
    from dataset import EUROSAT_CLASS_NAMES


class PromptLearner(nn.Module):
    """
    Holds the learnable context vectors and assembles full prompt embeddings.
    """

    def __init__(
        self,
        token_embedding: nn.Embedding,
        tokenizer,
        class_names: List[str],
        n_ctx: int = 16,
        class_specific: bool = False,
        ctx_init: Optional[str] = None,
        ctx_std: float = 0.02,
    ) -> None:
        super().__init__()

        self.class_names = list(class_names)
        self.n_cls = len(self.class_names)
        self.n_ctx = n_ctx
        self.class_specific = class_specific

        ctx_dim = token_embedding.weight.shape[1]
        self.ctx_dim = ctx_dim

        if ctx_init is not None:
            init_tokens = tokenizer([ctx_init])
            # Non-padding tokens include [SOS] and [EOS], hence the -2.
            n_words = int((init_tokens[0] != 0).sum().item()) - 2
            if n_words != n_ctx:
                raise ValueError(
                    f"ctx_init={ctx_init!r} tokenizes to {n_words} tokens "
                    f"but n_ctx={n_ctx}.  Pick a phrase of the right length "
                    f"or change n_ctx."
                )
            with torch.no_grad():
                init_emb = token_embedding(
                    init_tokens.to(token_embedding.weight.device)
                )
            # Skip [SOS] at position 0, take the n_ctx word embeddings.
            ctx_vectors = init_emb[0, 1: 1 + n_ctx, :].clone()  # (M, D)
            if class_specific:
                ctx_vectors = ctx_vectors.unsqueeze(0).repeat(self.n_cls, 1, 1)
        else:
            shape = (
                (self.n_cls, n_ctx, ctx_dim) if class_specific
                else (n_ctx, ctx_dim)
            )
            ctx_vectors = torch.empty(*shape)
            nn.init.normal_(ctx_vectors, std=ctx_std)

        # THE trainable parameter of this whole file.
        self.ctx = nn.Parameter(ctx_vectors)

        placeholder = " ".join(["X"] * n_ctx)
        prompts = [
            placeholder + " " + name.replace("_", " ") + "."
            for name in self.class_names
        ]
        tokenized = tokenizer(prompts)

        with torch.no_grad():
            embedding = token_embedding(
                tokenized.to(token_embedding.weight.device)
            )

        # (C, 1, D) — the [SOS] embedding.
        self.register_buffer("token_prefix", embedding[:, :1, :].clone())
        # (C, 77-1-M, D) — class name + [EOS] + padding.
        self.register_buffer(
            "token_suffix", embedding[:, 1 + n_ctx:, :].clone()
        )
        # Kept so the caller can locate [EOS] via argmax over token ids.
        self.register_buffer("tokenized_prompts", tokenized.clone())

    def forward(self) -> torch.Tensor:
        """
        Assemble the full prompt embeddings.
        """
        ctx = self.ctx
        if not self.class_specific:
            # Unified context: the same (M, D) block for every class.
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        return torch.cat([self.token_prefix, ctx, self.token_suffix], dim=1)


# CoOp model
class CoOpModel(BaseCLIPWrapper):
    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str = "cuda",
        class_names: List[str] = EUROSAT_CLASS_NAMES,
        n_ctx: int = 16,
        class_specific: bool = False,
        ctx_init: Optional[str] = None,
    ) -> None:
        super().__init__(
            model_name=model_name,
            pretrained=pretrained,
            device=device,
            class_names=class_names,
            prompt_template="{}",
        )

        self.n_ctx = n_ctx
        self.class_specific = class_specific

        self.prompt_learner = PromptLearner(
            token_embedding=self.model.token_embedding,
            tokenizer=self.tokenizer,
            class_names=self.class_names,
            n_ctx=n_ctx,
            class_specific=class_specific,
            ctx_init=ctx_init,
        ).to(device)

        for param in self.model.parameters():
            param.requires_grad = False
        assert self.count_trainable_params() == self.prompt_learner.ctx.numel()

    def _encode_text_from_embeddings(
        self, prompt_embeddings: torch.Tensor, tokenized_prompts: torch.Tensor
    ) -> torch.Tensor:
        """
        Run CLIP's text transformer starting from token *embeddings*.
        """
        m = self.model
        cast_dtype = m.transformer.get_cast_dtype()

        x = prompt_embeddings.to(cast_dtype)
        x = x + m.positional_embedding.to(cast_dtype)

        # Causal attention mask — the text tower is autoregressively masked
        # even though we only ever read the [EOS] position.
        x = m.transformer(x, attn_mask=m.attn_mask)
        x = m.ln_final(x)

        # Pool at [EOS].
        eos_idx = tokenized_prompts.argmax(dim=-1)
        x = x[torch.arange(x.shape[0], device=x.device), eos_idx]

        # Project into the shared image-text embedding space.  open_clip
        # stores this either as a plain matrix or as an nn.Linear depending
        # on the model config, so handle both.
        if m.text_projection is not None:
            if isinstance(m.text_projection, nn.Linear):
                x = m.text_projection(x)
            else:
                x = x @ m.text_projection

        return x

    def get_text_features(
        self, class_names: Optional[List[str]] = None
    ) -> torch.Tensor:
        """
        Return the learned text prototypes, one per class.
        """
        if class_names is not None and len(class_names) != len(self.class_names):
            raise ValueError(
                f"CoOpModel was built for {len(self.class_names)} classes but "
                f"got {len(class_names)} prompts.  The class set is fixed at "
                f"construction time; rebuild the model to change it."
            )

        prompt_embeddings = self.prompt_learner()
        text_features = self._encode_text_from_embeddings(
            prompt_embeddings, self.prompt_learner.tokenized_prompts
        )
        return text_features / text_features.norm(dim=-1, keepdim=True)

# test
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    for n_ctx, csc in [(4, False), (16, False), (16, True)]:
        model = CoOpModel(device=device, n_ctx=n_ctx, class_specific=csc)
        txt = model.get_text_features()
        tag = "CSC" if csc else "unified"
        print(
            f"M={n_ctx:<3} {tag:<8} | text feats {tuple(txt.shape)} "
            f"| trainable {model.count_trainable_params():,} "
            f"| requires_grad={txt.requires_grad}"
        )

    # Gradient check: a backward pass must reach ctx, and only ctx.
    model = CoOpModel(device=device, n_ctx=4, ctx_init="a satellite image of")
    dummy_images = torch.randn(2, 3, 224, 224)
    img_feats = model.get_image_features(dummy_images)
    txt_feats = model.get_text_features()
    logits = model.model.logit_scale.exp() * (img_feats @ txt_feats.T)
    logits.sum().backward()

    grad_norm = model.prompt_learner.ctx.grad.norm().item()
    print(f"ctx grad norm: {grad_norm:.4e}  (must be > 0)")
    print(f"ctx_init prompts OK | trainable {model.count_trainable_params():,}")
