# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from collections.abc import Iterator

import torch
from torch.distributed.device_mesh import DeviceMesh
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
    Qwen3VLMoeTextSparseMoeBlock as HFQwen3VLMoeTextSparseMoeBlock,
)

from cosmos_framework.model.generator.reasoner.qwen3_vl_moe.qwen3_vl_moe import (
    LBLMetadata,
    Qwen3VLMoeTextSparseMoeBlock,
)


def _iter_generation_moe_blocks(
    net: torch.nn.Module,
) -> Iterator[tuple[str, Qwen3VLMoeTextSparseMoeBlock]]:
    """Yield generation-tower sparse MoE blocks and their module names."""
    for name, module in net.named_modules():
        if isinstance(module, Qwen3VLMoeTextSparseMoeBlock) and "moe_gen" in name:
            yield name, module


def set_hf_moe_token_mask(net: torch.nn.Module, attention_mask: torch.Tensor | None) -> None:
    """Publish which rows are real tokens to every patched HF sparse MoE block.

    The patched block forward (see ``monkey_patch.patch_qwen3_vl_moe_grouped_mm_experts``)
    receives only ``hidden_states``, so it cannot tell the collate's trailing padding from
    content on its own. Without this, its routing statistics count the padding: the padded
    rows are dispatched to experts, land in the token counts and in the mean router
    probability, and the auxiliary loss then trains the router to balance rows that carry no
    supervision — ``ignore_index`` keeps them out of the cross-entropy, not out of this. The
    native ``Qwen3VLMoeTextSparseMoeBlock`` takes the same mask as a forward argument.

    The mirror image of :func:`collect_hf_moe_lbl_metadata`: this publishes before the
    forward, that collects after it. A plain attribute for the same reason — it holds one
    step's tensor and must never reach a state_dict.

    Published per step and overwritten by the next one rather than cleared after the forward:
    under non-reentrant activation checkpointing the block forward is re-executed during
    backward and has to mask the same rows, or the recomputed activations would not match the
    ones the forward produced.

    Args:
        net: Any module containing the patched HF ``Qwen3VLMoeTextSparseMoeBlock`` submodules
            (e.g. ``HFModel`` or the raw HF model).
        attention_mask: The collate's ``[B, N]`` mask, true on real tokens. ``None`` clears
            the blocks, which makes them count every row — what an unpadded stream wants.
    """
    if attention_mask is not None and attention_mask.ndim != 2:
        raise ValueError(
            f"attention_mask must be [B, N] to flatten alongside hidden_states, got {tuple(attention_mask.shape)}"
        )
    # Flattened once per step rather than once per layer: the block sees hidden_states already
    # reshaped to [B*N, hidden_size].
    token_mask = None if attention_mask is None else attention_mask.reshape(-1).bool()  # [B*N]
    for module in net.modules():
        if isinstance(module, HFQwen3VLMoeTextSparseMoeBlock):
            module.token_mask = token_mask


def collect_hf_moe_lbl_metadata(net: torch.nn.Module) -> LBLMetadata | None:
    """Pop the per-layer load-balancing statistics stashed on HF sparse MoE blocks.

    The statistics live on a plain ``block.lbl_metadata`` attribute set by the patched
    forward (see monkey_patch.patch_qwen3_vl_moe_grouped_mm_experts) rather than a buffer:
    it holds a graph-carrying tensor for the lifetime of one forward and must never reach a
    state_dict.

    Returns them stacked into the ``[num_layers, ...]`` layout that
    :func:`~cosmos_framework.model.generator.algorithm.loss.load_balancing.compute_load_balancing_loss`
    expects, or ``None`` for a backbone with no patched MoE block (or before its first
    forward).

    Popping is the point: ``mean_router_prob_per_expert`` carries the router's autograd
    graph, so leaving it on the module would pin a whole step's activations past backward.
    Call it exactly once per forward, before backward. The recompute that non-reentrant
    activation checkpointing runs during backward does NOT re-stash — ``monkey_patch``'s
    patched MoE block forward skips the stash there precisely because this collector has
    already run by then and would never clear it.

    Args:
        net: Any module containing the patched HF ``Qwen3VLMoeTextSparseMoeBlock``
            submodules (e.g. ``HFModel`` or the raw HF model).
    """
    per_layer: list[LBLMetadata] = []
    for module in net.modules():
        if not isinstance(module, HFQwen3VLMoeTextSparseMoeBlock):
            continue
        metadata = getattr(module, "lbl_metadata", None)
        if metadata is None:
            continue
        module.lbl_metadata = None
        per_layer.append(metadata)

    if not per_layer:
        return None
    return LBLMetadata(
        num_tokens_per_expert=torch.stack([m.num_tokens_per_expert for m in per_layer]),  # [L,E]
        num_tokens=torch.stack([m.num_tokens for m in per_layer]),  # [L,1]
        mean_router_prob_per_expert=torch.stack([m.mean_router_prob_per_expert for m in per_layer]),  # [L,E]
        top_k=torch.stack([m.top_k for m in per_layer]),  # [L,1]
    )


def update_expert_biases(
    net: torch.nn.Module,
    device_mesh: DeviceMesh | None = None,
) -> None:
    """Update routing-load biases on every enabled generation-tower MoE block."""
    for _, module in _iter_generation_moe_blocks(net):
        if module.aux_loss_free_load_balancing_config.enabled:
            module.update_bias(device_mesh=device_mesh)


def update_router_biases(net: torch.nn.Module, device_mesh: DeviceMesh | None = None) -> None:
    """Update EMA router biases on every enabled generation-tower MoE block."""
    for _, module in _iter_generation_moe_blocks(net):
        if module.cosine_router.input_centering == "ema":
            module.update_router_bias(device_mesh=device_mesh)


def uses_aux_loss_free_load_balancing(net: torch.nn.Module) -> bool:
    """Return whether any generation-tower MoE block uses aux-loss-free load balancing."""
    return any(module.aux_loss_free_load_balancing_config.enabled for _, module in _iter_generation_moe_blocks(net))


def uses_ema_router_bias(net: torch.nn.Module) -> bool:
    """Return whether any generation-tower MoE block uses an EMA router bias."""
    return any(module.cosine_router.input_centering == "ema" for _, module in _iter_generation_moe_blocks(net))


@torch.no_grad()
def sync_expert_biases_to_ema(net: torch.nn.Module, net_ema: torch.nn.Module) -> None:
    """Mirror generation-tower expert-bias buffers from ``net`` into ``net_ema``."""
    ema_blocks = {
        name: module
        for name, module in _iter_generation_moe_blocks(net_ema)
        if module.aux_loss_free_load_balancing_config.enabled
    }
    for name, source in _iter_generation_moe_blocks(net):
        target = ema_blocks.get(name)
        if source.aux_loss_free_load_balancing_config.enabled and target is not None and hasattr(target, "expert_bias"):
            target.expert_bias.copy_(source.expert_bias)  # [E]


@torch.no_grad()
def sync_router_biases_to_ema(net: torch.nn.Module, net_ema: torch.nn.Module) -> None:
    """Mirror generation-tower router-bias buffers from ``net`` into ``net_ema``."""
    ema_blocks = {
        name: module
        for name, module in _iter_generation_moe_blocks(net_ema)
        if module.cosine_router.input_centering == "ema"
    }
    for name, source in _iter_generation_moe_blocks(net):
        target = ema_blocks.get(name)
        if source.cosine_router.input_centering == "ema" and target is not None:
            target.cosine_router.router_bias.copy_(source.cosine_router.router_bias)  # [D]
