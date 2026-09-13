"""
coop_adapter.py — Joint text + vision adaptation (CoOp x CLIP-Adapter)
"""

import sys
from pathlib import Path

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
        adapter_lr_scale: float = 1.0,
    ) -> None:

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
        self.adapter_lr_scale = adapter_lr_scale

        # Text side: CoOp's prompt learner.
        self.prompt_learner = PromptLearner(
            token_embedding=self.model.token_embedding,
            tokenizer=self.tokenizer,
            class_names=self.class_names,
            n_ctx=n_ctx,
            class_specific=class_specific,
            ctx_init=ctx_init,
        ).to(device)

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

    _encode_text_from_embeddings = CoOpModel._encode_text_from_embeddings
    get_text_features = CoOpModel.get_text_features
    get_image_features = CLIPAdapterModel.get_image_features

    @property
    def alpha(self):
        """Effective residual blending factor, read from the adapter."""
        return self.adapter.alpha

    def set_alpha(self, new_alpha: float) -> None:
        """Update the residual blending factor in place."""
        self.adapter.alpha = new_alpha

    # Optimizer hook
    def trainable_param_groups(self, lr: float) -> List[dict]:
        """
        Split the trainable tensors into two optimizer groups.
        """
        return [
            {
                "name": "context",
                "params": [self.prompt_learner.ctx],
                "lr": lr,
            },
            {
                "name": "adapter",
                "params": [p for p in self.adapter.parameters() if p.requires_grad],
                "lr": lr * self.adapter_lr_scale,
            },
        ]

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


# test
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    class _Naive(CoOpModel, CLIPAdapterModel):
        pass

    try:
        _Naive(device=device)
        print("[warn] the naive multiple-inheritance model constructed; "
              "the MRO trap may have been fixed upstream.")
    except AttributeError as exc:
        print(f"[ok] naive multiple inheritance fails as expected: {exc}")

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

    owner = next(k.__name__ for k in type(model).__mro__ if "predict" in k.__dict__)
    print(f"predict() inherited from: {owner}")     # expected: BaseCLIPWrapper
    preds, sims = model.predict(dummy_images)
    print(f"predict() shapes  : preds {tuple(preds.shape)}, sims {tuple(sims.shape)}")

    groups = model.trainable_param_groups(lr=1e-3)
    grouped = sum(p.numel() for g in groups for p in g["params"])
    print("\nparam groups      : " + ", ".join(
        f"{g['name']} {sum(p.numel() for p in g['params']):,} @ lr={g['lr']:g}"
        for g in groups
    ))
    print(f"grouped total     : {grouped:,} "
          f"(count_trainable_params: {model.count_trainable_params():,})")
    assert grouped == model.count_trainable_params(), "a trainable tensor is missing from the groups"

    dtd = CoOpAdapterModel(
        device=device,
        class_names=["banded", "bubbly", "cracked", "dotted", "woven"],
        n_ctx=4,
    )
    print(f"5-class model     : text feats {tuple(dtd.get_text_features().shape)} "
          f"| trainable {dtd.count_trainable_params():,}")
