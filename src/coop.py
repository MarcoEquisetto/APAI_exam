import sys
from pathlib import Path

# Add project root and src directory to sys.path (same convention as the
# rest of the project, so the module runs both as ``python src/coop.py``
# and as ``from src.coop import CoOpModel``).
FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parent
if str(FILE_DIR) not in sys.path:
    sys.path.insert(0, str(FILE_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from typing import List, Optional, Tuple

try:
    from src.base_model import BaseCLIPWrapper
    from src.dataset import EUROSAT_CLASS_NAMES
except ModuleNotFoundError:
    from base_model import BaseCLIPWrapper
    from dataset import EUROSAT_CLASS_NAMES


# ============================================================================
# Prompt Learner
# ============================================================================

class PromptLearner(nn.Module):
    """
    Holds the learnable context vectors and assembles full prompt embeddings.

    The module stores three things:

    * ``ctx`` — the trainable ``nn.Parameter``;
    * ``token_prefix`` — the frozen [SOS] embedding, shape ``(C, 1, D)``;
    * ``token_suffix`` — the frozen embeddings of the class name, the [EOS]
      token and the padding, shape ``(C, 77 - 1 - M, D)``.

    Prefix and suffix are registered as *buffers*: they are constants derived
    once from the frozen embedding table, but they must follow the module
    across ``.to(device)`` calls, and they must not appear in
    ``parameters()`` or the optimizer would try to train them.

    **The placeholder trick.**  To obtain prefix and suffix we tokenize the
    string ``"X X X ... X <class name>."`` with exactly ``M`` copies of
    ``"X"``.  Each ``"X"`` is a single BPE token, so the tokenized sequence
    has precisely the layout we want — and, crucially, the [EOS] token lands
    at the *same index* it would occupy in the real prompt.  That matters
    because CLIP pools the sequence by taking the position of the highest
    token id, which is [EOS] in CLIP's BPE vocabulary.  The embeddings of the
    ``"X"`` placeholders are then thrown away and replaced by ``ctx``.

    Params:
    ----------
    token_embedding : nn.Embedding
        CLIP's frozen token-embedding table.  Used only inside ``__init__``
        to precompute the buffers; **not** stored as an attribute, otherwise
        the frozen CLIP weights would be registered a second time inside
        this module and would show up in ``parameters()``.
    tokenizer : Callable
        OpenCLIP tokenizer, ``List[str] -> (N, 77)`` int tensor.
    class_names : List[str]
        Raw class names, e.g. ``["forest", "river", ...]``.  Raw, *not*
        wrapped in a prompt template: the template is exactly what CoOp
        replaces.
    n_ctx : int
        Number of context tokens ``M``.  The brief asks for 4, 8 and 16.
    class_specific : bool
        ``False`` → Unified Context ``(M, D)``.  ``True`` → CSC ``(C, M, D)``.
    ctx_init : Optional[str]
        If given (e.g. ``"a satellite image of"``), the context is
        initialized from the embeddings of those words instead of random
        noise.  The phrase must tokenize to exactly ``n_ctx`` tokens.
        Starting from a sensible prompt usually converges faster and makes a
        nice ablation for the report.
    ctx_std : float
        Standard deviation of the random init (the CoOp paper uses 0.02).
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

        # Width of the *transformer* embedding space (512 for ViT-B/32's
        # text tower).  Note this is the dimension *before* the final text
        # projection: the context vectors live in the input space of the
        # transformer, not in the shared image-text space.
        ctx_dim = token_embedding.weight.shape[1]
        self.ctx_dim = ctx_dim

        # ------------------------------------------------------------------
        # 1. Initialize the context vectors.
        # ------------------------------------------------------------------
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

        # ------------------------------------------------------------------
        # 2. Build the frozen prefix / suffix around it.
        # ------------------------------------------------------------------
        placeholder = " ".join(["X"] * n_ctx)
        prompts = [
            placeholder + " " + name.replace("_", " ") + "."
            for name in self.class_names
        ]
        tokenized = tokenizer(prompts)                         # (C, 77)

        with torch.no_grad():
            embedding = token_embedding(
                tokenized.to(token_embedding.weight.device)
            )                                                  # (C, 77, D)

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

        Returns
        -------
        prompts : torch.Tensor
            Shape ``(C, 77, D)`` — ready to be fed to the text transformer.
        """
        ctx = self.ctx
        if not self.class_specific:
            # Unified context: the same (M, D) block for every class.
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        return torch.cat([self.token_prefix, ctx, self.token_suffix], dim=1)


# ============================================================================
# CoOp model
# ============================================================================

class CoOpModel(BaseCLIPWrapper):
    """
    ``BaseCLIPWrapper`` subclass that learns its text prompts.

    Overrides ``get_text_features()``; ``get_image_features()`` is inherited
    untouched, so this model composes cleanly with Marco's vision adapter in
    a later joint model.

    Parameters
    ----------
    model_name, pretrained, device
        Forwarded to ``BaseCLIPWrapper``.
    class_names : List[str]
        Raw class names.  Stored at construction time — see the note in
        ``get_text_features`` about why its argument is ignored.
    n_ctx : int
        Context length ``M`` (brief: 4 / 8 / 16).
    class_specific : bool
        Unified Context (default) vs. Class-Specific Context.
    ctx_init : Optional[str]
        Optional phrase to initialize the context from.
    """

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
        )

        self.class_names = list(class_names)
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

        # Safety net: the base class froze the backbone before the prompt
        # learner existed, so re-assert the invariant explicitly.  Only the
        # context vectors may train.
        for param in self.model.parameters():
            param.requires_grad = False
        assert self.count_trainable_params() == self.prompt_learner.ctx.numel()

    # ------------------------------------------------------------------
    # Manual text-encoder forward
    # ------------------------------------------------------------------
    def _encode_text_from_embeddings(
        self, prompt_embeddings: torch.Tensor, tokenized_prompts: torch.Tensor
    ) -> torch.Tensor:
        """
        Run CLIP's text transformer starting from token *embeddings*.

        This mirrors ``open_clip``'s ``encode_text()`` exactly, minus the
        embedding lookup which the prompt learner has already done for us.

        Parameters
        ----------
        prompt_embeddings : torch.Tensor
            ``(C, 77, D)`` prompt embeddings from ``PromptLearner``.
        tokenized_prompts : torch.Tensor
            ``(C, 77)`` token ids of the *placeholder* prompts.  Only used
            to locate the [EOS] position: in CLIP's BPE vocabulary [EOS]
            has the largest id, so ``argmax(-1)`` finds it.

        Returns
        -------
        text_features : torch.Tensor
            ``(C, D_embed)`` — unnormalized pooled text embeddings.
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

    # ------------------------------------------------------------------
    # Interface method (overridden)
    # ------------------------------------------------------------------
    def get_text_features(
        self, class_names: Optional[List[str]] = None
    ) -> torch.Tensor:
        """
        Return the learned text prototypes, one per class.

        Params:
        ----------
        class_names : Optional[List[str]]
            **Ignored.**  The prompts are not built from strings at call
            time: the class-name embeddings were baked into the prompt
            learner's frozen suffix buffer at construction time, and the
            context is a tensor, not text.  The argument is kept only so the
            signature stays compatible with ``BaseCLIPWrapper`` and with
            ``engine.train()``, which currently hands over already-formatted
            prompts (``"a satellite image of forest"``) rather than the raw
            names the brief specifies.  A *different number* of classes is
            rejected loudly rather than silently ignored.

        Returns
        -------
        text_features : torch.Tensor
            ``(C, D)`` L2-normalized prototypes, recomputed from the current
            value of the context vectors.

        Notes
        -----
        No ``@torch.no_grad()`` here — on purpose.  See the module docstring.
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

    # ------------------------------------------------------------------
    # Evaluation hook expected by engine.evaluate()
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(
        self, images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Classify a batch of images against the learned prototypes.

        Text features are recomputed on every call rather than cached: the
        context vectors change after every optimizer step, so a cache would
        go stale mid-training (contract #4 in the project notes).  The cost
        is ten prompts through the text tower — negligible next to the image
        batch.

        Returns
        -------
        predictions : torch.Tensor
            ``(B,)`` predicted class indices.
        similarities : torch.Tensor
            ``(B, C)`` cosine similarities.
        """
        image_features = self.get_image_features(images)
        text_features = self.get_text_features()
        similarities = image_features @ text_features.T
        return similarities.argmax(dim=-1), similarities


# ============================================================================
# Smoke test
# ============================================================================
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
