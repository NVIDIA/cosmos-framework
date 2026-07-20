# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from collections.abc import Iterator

import torch
from torch.distributed.device_mesh import DeviceMesh

from cosmos_framework.model.generator.reasoner.qwen3_vl_moe.qwen3_vl_moe import Qwen3VLMoeTextSparseMoeBlock


def _iter_generation_moe_blocks(
    net: torch.nn.Module,
) -> Iterator[tuple[str, Qwen3VLMoeTextSparseMoeBlock]]:
    """Yield generation-tower sparse MoE blocks and their module names."""
    for name, module in net.named_modules():
        if isinstance(module, Qwen3VLMoeTextSparseMoeBlock) and "moe_gen" in name:
            yield name, module


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
