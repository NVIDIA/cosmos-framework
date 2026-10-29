# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Opt-in fused SFT objective for dense and MoE Qwen3-VL models."""

from functools import wraps
from types import MethodType
from typing import Any, cast

import torch.nn as nn
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeForConditionalGeneration

from cosmos_framework.model.generator.algorithm.loss.cross_entropy import fused_weighted_cross_entropy_loss
from cosmos_framework.model.generator.reasoner.qwen3_vl_rope import configure_qwen3_vl_fp32_rope
from cosmos_framework.model.generator.reasoner.qwen35_caption import Qwen35CaptionLoss


def configure_qwen3_vl_fused_ce(
    model: nn.Module, *, exponent: float, recompute: bool = False, accumulate_head_grad_fp32: bool = True
) -> None:
    """Evaluate LM projection and CE while the FSDP root's weight is gathered.

    The existing Qwen3-VL backbone patch owns text-only vision dependencies.
    MoE routing hooks remain on the original decoder blocks, so VLMModel still
    computes its configured load-balancing objective outside this wrapper.
    """
    if getattr(model, "config").model_type not in {"qwen3_vl", "qwen3_vl_moe"}:
        raise ValueError("Qwen3-VL fused CE requires a dense or MoE Qwen3-VL model")
    if getattr(model, "_qwen3_vl_fused_ce", False):
        raise ValueError("Qwen3-VL fused CE is already configured")
    if not 0.0 <= exponent <= 1.0:
        raise ValueError("Weighted CE exponent must be in [0,1]")
    # The family check establishes the native Qwen3-VL model interface.
    model = cast(Qwen3VLForConditionalGeneration | Qwen3VLMoeForConditionalGeneration, model)
    # Evaluate native position phases in FP32 on the fused path, using the same
    # phases in CP=1 and CP>1 when comparing the fused path.
    configure_qwen3_vl_fp32_rope(model.model.language_model.rotary_emb)
    original_forward = model.forward

    @wraps(original_forward.__func__)
    def forward(
        self: Qwen3VLForConditionalGeneration | Qwen3VLMoeForConditionalGeneration, *args: Any, **kwargs: Any
    ) -> Qwen35CaptionLoss:
        labels = kwargs.pop("labels", None)
        if labels is None:
            raise ValueError("Fused Qwen3-VL SFT requires labels")
        kwargs.pop("logits_to_keep", None)
        kwargs["use_cache"] = False
        kwargs["return_dict"] = True
        hidden = self.model(*args, **kwargs).last_hidden_state
        loss, stats = fused_weighted_cross_entropy_loss(
            hidden,
            labels,
            self.lm_head.weight,
            exponent=exponent,
            context_parallel_group=getattr(self, "_cosmos_cp_group", None),
            cu_seq_lens=kwargs.get("cu_seq_lens_q"),
            recompute=recompute,
            accumulate_head_grad_fp32=accumulate_head_grad_fp32 and kwargs.get("cu_seq_lens_q") is not None,
        )
        return Qwen35CaptionLoss(loss, stats)

    model.forward = MethodType(forward, model)
    setattr(model, "_qwen3_vl_fused_ce", True)
