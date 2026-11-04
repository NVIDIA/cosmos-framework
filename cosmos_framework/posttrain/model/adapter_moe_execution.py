# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Opt-in native/stock MoE arithmetic contract; never changes router selection.

The default native kernels remain unchanged. This profile deliberately pays for
stock expert weight layout and a fixed pair of vision passes so native training,
portable inference and the pinned, unmodified HF loader execute the same graph.
"""

from importlib.metadata import version
from types import MethodType, SimpleNamespace

import torch

PROFILE = "qwen3_vl_moe_stock_v1"
_NATIVE = "cosmos_framework.model.generator.reasoner.qwen3_vl_moe.qwen3_vl_moe"
_STOCK = "transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe"


def validate_model_execution(net, config, family):
    """Validate the entire profile before adapter or model mutation."""
    declared = getattr(getattr(net, "config", None), "cosmos_model_execution", None)
    if declared is not None and declared != config.model_execution:
        raise ValueError("base declares a different model execution profile; preserve its explicit execution contract")
    if config.model_execution == "default_v1":
        return
    if family != "qwen3_vl_moe" or config.model_execution != PROFILE:
        raise ValueError(f"{PROFILE} requires the Qwen3-VL-MoE family")
    if version("transformers") != "5.8.1" or version("torch").split("+")[0] != "2.10.0":
        raise ValueError(f"{PROFILE} requires the qualified Transformers 5.8.1 / PyTorch 2.10.0 runtime")
    model_type = type(net)
    if model_type.__name__ != "Qwen3VLMoeForConditionalGeneration" or model_type.__module__ not in (_NATIVE, _STOCK):
        raise ValueError(f"{PROFILE} requires an exact native or stock Qwen3-VL-MoE model")
    configs = [net.config, net.config.text_config, net.config.vision_config]
    configs.extend((net.model.language_model.config, net.model.visual.config))
    configs.extend(
        module.config
        for module in net.modules()
        if type(module).__name__ in ("Qwen3VLMoeTextAttention", "Qwen3VLMoeVisionAttention")
    )
    if any(config._attn_implementation != "sdpa" for config in configs):
        raise ValueError(f"{PROFILE} requires SDPA attention")
    if model_type.__module__ == _STOCK:
        if net.config.text_config._experts_implementation != "grouped_mm" or any(
            module.config._experts_implementation != "grouped_mm"
            for module in net.modules()
            if type(module).__name__ == "Qwen3VLMoeTextExperts"
        ):
            raise ValueError(f"{PROFILE} requires stock experts_implementation='grouped_mm'")
        return
    if not getattr(net.model, "_posttrain_forward_override", False):
        raise ValueError(f"{PROFILE} requires the native posttrain multimodal forward")
    for layer in net.model.language_model.layers:
        block = layer.mlp
        if type(block).__name__ != "Qwen3VLMoeTextSparseMoeBlock" or type(block).__module__ != _NATIVE:
            raise ValueError(f"{PROFILE} requires sparse blocks in every language layer")
        router = block.cosine_router
        if (
            block.noisy_gating
            or block.aux_loss_free_load_balancing_config.enabled
            or block.shared_expert is not None
            or router.use_cosine_similarity
            or router.activation != "softmax"
            or router.input_centering != "none"
            or block.top_k != block.config.num_experts_per_tok
        ):
            raise ValueError(f"{PROFILE} requires the checkpoint's original softmax/top-k router")


def stock_experts_forward(self, hidden_states, topk_scores, expert_indices, num_tokens_per_expert, token_mask):
    """Reuse stock grouped execution with original native parameter ownership.

    Contiguous transposed copies are temporary differentiable tensors, never
    registered leaves or cached weights. They match the pinned loader's layout.
    Padding sentinels affect only inactive rows; the router's outputs stay intact.
    """
    from transformers.integrations.moe import grouped_mm_experts_forward

    del num_tokens_per_expert
    if token_mask is not None:
        hidden_states = torch.where(token_mask[:, None], hidden_states, 0.0)
        expert_indices = expert_indices.masked_fill(~token_mask[:, None], self.num_experts)
        topk_scores = torch.where(token_mask[:, None], topk_scores, 0.0)

    def apply_gate(value):
        gate, up = value.chunk(2, dim=-1)
        return self.act_fn(gate) * up

    proxy = SimpleNamespace(
        num_experts=self.num_experts,
        has_gate=True,
        has_bias=False,
        is_transposed=False,
        gate_up_proj=self.gate_up_proj.transpose(-1, -2).contiguous(),
        down_proj=self.down_proj.transpose(-1, -2).contiguous(),
        _apply_gate=apply_gate,
    )
    return grouped_mm_experts_forward(proxy, hidden_states, expert_indices, topk_scores)


def install_model_execution(net, config):
    """Called only after validation and trainability policy, before FSDP."""
    if config.model_execution == "default_v1":
        return
    if type(net).__module__ == _NATIVE:
        for layer in net.model.language_model.layers:
            experts = layer.mlp.experts
            experts.forward = MethodType(stock_experts_forward, experts)
        net.model._posttrain_separate_modalities = True
    net._posttrain_model_execution = config.model_execution
