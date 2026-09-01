"""
coop_adapter.py — Joint text + vision adaptation (CoOp × CLIP-Adapter)
=====================================================================

This module combines the two adaptation methods the project develops
separately:

* **CoOp** (Carlo, ``src/coop.py``) adapts the *text* side by replacing the
  hand-written prompt ``"a satellite image of {class}"`` with ``M``
  continuous vectors that are learned by gradient descent;
* **CLIP-Adapter** (Marco, ``src/clip_adapter.py``) adapts the *vision*
  side by appending a bottleneck MLP after the frozen vision encoder and
  blending its output back into the original features.

They are complementary by construction: one overrides
``get_text_features()``, the other overrides ``get_image_features()``, and
the brief assigns exactly one of the two to each workstream precisely so
that they can be composed. The interesting question for the report is
whether the gains add up, or whether both methods are chasing the same
few percentage points of headroom.

The backbone stays frozen. Only the context vectors and the adapter MLP
train — roughly 140K parameters against CLIP's 86M.


Why this file exists instead of a one-line subclass
---------------------------------------------------

The obvious spelling does not work::

    class CoOpAdapterModel(CoOpModel, CLIPAdapterModel):
        pass

    CoOpAdapterModel(device="cuda")
    # AttributeError: 'CoOpAdapterModel' object has no attribute 'prompt_learner'

Python resolves attributes along the MRO
``CoOpAdapterModel → CoOpModel → CLIPAdapterModel → BaseCLIPWrapper``.
Construction then goes wrong in a way that is worth understanding, because
the same trap catches any pair of sibling classes written independently:

1. ``CoOpModel.__init__`` starts and immediately calls ``super().__init__()``,
   believing it is talking to ``BaseCLIPWrapper``;
2. under the new MRO its ``super()`` is ``CLIPAdapterModel``, not the base;
3. ``CLIPAdapterModel.__init__`` runs to completion and, as its last act,
   calls ``self._update_text_prototypes()``;
4. that call resolves *on the instance*, so it lands on CoOp's text path,
   which reads ``self.prompt_learner``;
5. but the prompt learner is only created *after* ``super().__init__()``
   returns — i.e. after step 1 finishes. It does not exist yet.

Neither parent is at fault: each is correct on its own. The problem is that
neither ``__init__`` was written to cooperate along an MRO (neither forwards
``**kwargs`` up the chain), and cooperative multiple inheritance only works
when *every* class in the chain plays along.

The fix is to stop using the MRO for construction: call
``BaseCLIPWrapper.__init__`` explicitly, build the two modules by hand in an
order we control, and then *borrow* the two feature methods from their
owners as plain function attributes. No behaviour is duplicated — if Carlo
or Marco changes their method, this model picks up the change — but nothing
is inherited either, so there is no MRO to get wrong.


Two traps that must survive any future clean-up
------------------------------------------------

1. **No ``@torch.no_grad()`` on the borrowed feature methods.** The base
   class decorates its own versions; the overrides deliberately do not. If
   somebody "restores consistency" by adding the decorator, training still
   runs, the loss still wobbles, and not a single parameter learns
   anything — silently.
2. **No cache of the text prototypes.** ``BaseCLIPWrapper.predict()``
   recomputes them on every call. That is not an oversight: ``ctx`` changes
   after every optimizer step, so a cache built in ``__init__`` would
   report the accuracy of the initial *random* context forever.
"""

import sys
from pathlib import Path

# Same sys.path convention as the rest of the project, so the module works
# both as ``python src/coop_adapter.py`` and as
# ``from src.coop_adapter import CoOpAdapterModel``.
FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parent
if str(FILE_DIR) not in sys.path:
    sys.path.insert(0, str(FILE_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from typing import List, Optional

try:
    from src.base_model import BaseCLIPWrapper
    from src.clip_adapter import CLIPAdapterModel, VisionAdapterModule
    from src.coop import CoOpModel, PromptLearner
    from src.dataset import EUROSAT_CLASS_NAMES
except ModuleNotFoundError:
    from base_model import BaseCLIPWrapper
    from clip_adapter import CLIPAdapterModel, VisionAdapterModule
    from coop import CoOpModel, PromptLearner
    from dataset import EUROSAT_CLASS_NAMES


class CoOpAdapterModel(BaseCLIPWrapper):
    """
    Learned text prompts *and* a vision adapter, over one frozen backbone.

    Parameters
    ----------
    model_name, pretrained, device
        Forwarded to ``BaseCLIPWrapper``.
    class_names : List[str]
        Raw class names, e.g. ``["forest", "river", ...]``.  As in
        ``CoOpModel``, the class set is fixed at construction time: the
        class-name embeddings are baked into the prompt learner's frozen
        suffix buffer.
    n_ctx : int
        Number of learnable context tokens ``M`` (the brief sweeps 4/8/16).
        Unified context costs ``M × 512`` parameters; class-specific costs
        ``C × M × 512``.
    class_specific : bool
        ``False`` → one shared context for every class (Unified Context).
        ``True``  → a separate context per class (CSC).
    ctx_init : Optional[str]
        Optional phrase to initialize the context from, e.g.
        ``"a satellite image of"``.  Must tokenize to exactly ``n_ctx``
        tokens.  Starting from a meaningful prompt usually converges faster
        than starting from Gaussian noise.
    reduction_ratio : int
        Vision adapter bottleneck ratio.  ``4`` means 512 → 128 → 512.
    alpha : float
        Residual blending factor for the adapter.
    learnable_alpha : bool
        Train ``alpha`` alongside the MLP.  Default ``False``, matching
        ``CLIPAdapterModel``, so that an alpha sweep stays meaningful.
    constrain_alpha : bool
        Keep a learned ``alpha`` inside ``(0, 1)`` via a sigmoid.

    Notes
    -----
    There is no ``prompt_template`` argument, unlike Marco's models. The
    text side is CoOp's, and CoOp *has* no template — learning a
    replacement for it is the method. This is also why the joint model,
    like plain CoOp, is immune to the EuroSAT-template-on-DTD bug that
    affects every method with hard-coded prompts.
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
        reduction_ratio: int = 4,
        alpha: float = 0.2,
        learnable_alpha: bool = False,
        constrain_alpha: bool = True,
    ) -> None:
        # ------------------------------------------------------------------
        # EXPLICIT call to the base class, not ``super().__init__()``.
        #
        # This is the whole point of the file.  Going through ``super()``
        # would route the call into CLIPAdapterModel.__init__, which ends by
        # computing text prototypes — before the prompt learner that those
        # prototypes depend on has been created.  See the module docstring.
        #
        # ``prompt_template="{}"`` because CoOp does not use one; the base
        # class stores it only so that ``build_prompts()`` stays well
        # defined, and nothing here reads the result.
        # ------------------------------------------------------------------
        BaseCLIPWrapper.__init__(
            self,
            model_name=model_name,
            pretrained=pretrained,
            device=device,
            class_names=class_names,
            prompt_template="{}",
        )

        self.n_ctx = n_ctx
        self.class_specific = class_specific
        self.reduction_ratio = reduction_ratio

        # ------------------------------------------------------------------
        # Text side: CoOp's prompt learner.
        # Built first, so that anything touching the text path afterwards
        # finds it in place.
        # ------------------------------------------------------------------
        self.prompt_learner = PromptLearner(
            token_embedding=self.model.token_embedding,
            tokenizer=self.tokenizer,
            class_names=self.class_names,
            n_ctx=n_ctx,
            class_specific=class_specific,
            ctx_init=ctx_init,
        ).to(device)

        # ------------------------------------------------------------------
        # Vision side: Marco's bottleneck MLP.
        # The embedding dimension is read off the visual projection matrix
        # rather than hard-coded to 512, so ViT-L/14 would work unchanged.
        # ------------------------------------------------------------------
        embed_dim = 512
        if getattr(self.model.visual, "proj", None) is not None:
            embed_dim = self.model.visual.proj.shape[1]

        self.adapter = VisionAdapterModule(
            embed_dim=embed_dim,
            reduction_ratio=reduction_ratio,
            alpha=alpha,
            learnable_alpha=learnable_alpha,
            constrain_alpha=constrain_alpha,
        ).to(device)

        # ------------------------------------------------------------------
        # Safety net.  ``BaseCLIPWrapper.__init__`` froze the backbone before
        # either module existed, so re-assert the invariant and then check
        # it: the only trainable tensors must be the context vectors and the
        # adapter.  An assert here is cheap and catches, at construction
        # time, the class of bug that otherwise shows up as "training runs
        # but nothing improves".
        # ------------------------------------------------------------------
        for param in self.model.parameters():
            param.requires_grad = False

        expected = self.prompt_learner.ctx.numel() + sum(
            p.numel() for p in self.adapter.parameters() if p.requires_grad
        )
        assert self.count_trainable_params() == expected, (
            f"Expected {expected:,} trainable parameters (context + adapter) "
            f"but found {self.count_trainable_params():,}.  Something in the "
            "backbone was left unfrozen."
        )

    # ----------------------------------------------------------------------
    # Borrowed methods.
    #
    # Plain function attributes, not inheritance.  Each one stays the single
    # source of truth in its owner's file: a change to CoOp's text encoder or
    # to Marco's adapter path is picked up here automatically, and there is
    # no MRO to resolve.
    #
    # NEITHER has @torch.no_grad().  That is deliberate and load-bearing —
    # see the module docstring.
    # ----------------------------------------------------------------------
    _encode_text_from_embeddings = CoOpModel._encode_text_from_embeddings
    get_text_features = CoOpModel.get_text_features
    get_image_features = CLIPAdapterModel.get_image_features

    # ``predict()`` is inherited from BaseCLIPWrapper, which recomputes the
    # text prototypes on every call — exactly what a model with a trainable
    # text side needs.  Do not add a cached override here.

    @property
    def alpha(self):
        """Effective residual blending factor, read from the adapter."""
        return self.adapter.alpha

    def set_alpha(self, new_alpha: float) -> None:
        """Update the residual blending factor in place."""
        self.adapter.alpha = new_alpha

    def parameter_breakdown(self) -> dict:
        """
        Trainable parameters split by module.

        Useful for the "Accuracy vs. Trainable Parameters" figure: the joint
        model's cost is the sum of the two methods', and the report should
        be able to say which half is paying for which gain.
        """
        ctx = self.prompt_learner.ctx.numel()
        adapter = sum(p.numel() for p in self.adapter.parameters() if p.requires_grad)
        return {"context": ctx, "adapter": adapter, "total": ctx + adapter}


# ============================================================================
# Smoke test
# ============================================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # 1. The naive spelling must fail — documented here so nobody
    #    "simplifies" this file back into a one-liner.
    # ------------------------------------------------------------------
    class _Naive(CoOpModel, CLIPAdapterModel):
        pass

    try:
        _Naive(device=device)
        print("[warn] the naive multiple-inheritance model constructed; "
              "the MRO trap may have been fixed upstream.")
    except AttributeError as exc:
        print(f"[ok] naive multiple inheritance fails as expected: {exc}")

    # ------------------------------------------------------------------
    # 2. The explicit model, in three configurations.
    # ------------------------------------------------------------------
    for n_ctx, csc, r in [(4, False, 4), (16, False, 4), (16, True, 16)]:
        model = CoOpAdapterModel(
            device=device, n_ctx=n_ctx, class_specific=csc, reduction_ratio=r,
        )
        b = model.parameter_breakdown()
        tag = "CSC" if csc else "unified"
        print(f"M={n_ctx:<3} {tag:<8} r={r:<3} | "
              f"context {b['context']:>7,} + adapter {b['adapter']:>7,} "
              f"= {b['total']:>7,}")
        del model

    # ------------------------------------------------------------------
    # 3. Gradient check: a backward pass must reach BOTH modules.
    #    This is the test that catches a stray @torch.no_grad().
    # ------------------------------------------------------------------
    model = CoOpAdapterModel(device=device, n_ctx=16, reduction_ratio=4)
    dummy_images = torch.randn(2, 3, 224, 224)

    img_feats = model.get_image_features(dummy_images)
    txt_feats = model.get_text_features()
    logits = model.model.logit_scale.exp() * (img_feats @ txt_feats.T)
    logits.sum().backward()

    ctx_grad = model.prompt_learner.ctx.grad
    adapter_grad = model.adapter.up_proj.weight.grad
    print(f"\nctx grad norm     : {ctx_grad.norm().item():.4e}  (must be > 0)")
    print(f"adapter grad norm : {adapter_grad.norm().item():.4e}  (must be > 0)")

    # ------------------------------------------------------------------
    # 4. predict() must come from the base class, not from a copy.
    # ------------------------------------------------------------------
    owner = next(k.__name__ for k in type(model).__mro__ if "predict" in k.__dict__)
    print(f"predict() inherited from: {owner}")     # expected: BaseCLIPWrapper
    preds, sims = model.predict(dummy_images)
    print(f"predict() shapes  : preds {tuple(preds.shape)}, sims {tuple(sims.shape)}")

    # ------------------------------------------------------------------
    # 5. A different class set: 47 DTD classes, no template needed.
    # ------------------------------------------------------------------
    dtd = CoOpAdapterModel(
        device=device,
        class_names=["banded", "bubbly", "cracked", "dotted", "woven"],
        n_ctx=4,
    )
    print(f"5-class model     : text feats {tuple(dtd.get_text_features().shape)} "
          f"| trainable {dtd.count_trainable_params():,}")
