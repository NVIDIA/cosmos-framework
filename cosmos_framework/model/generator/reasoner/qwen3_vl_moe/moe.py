# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import torch
import torch.nn as nn
from transformers.activations import ACT2FN

from cosmos_framework.model.generator.moe import native
from cosmos_framework.model.generator.moe.interface import MoEBackendConfig, MoEExperts
from cosmos_framework.model.generator.moe.native import _run_experts_grouped_mm as _run_experts_grouped_mm
from cosmos_framework.model.generator.reasoner.qwen3_vl_moe.configuration_qwen3_vl_moe import (
    Qwen3VLMoeTextConfig,
)


class Qwen3VLMoeTextExpertsGroupedMm(native.NativeMoEExperts):
    """Legacy Qwen entry point for the shared native grouped-MM implementation.

    Keep the config-based constructor and parameter names for existing callers.
    The inherited forward and reorder methods also support binding to HF expert instances.
    """

    def __init__(self, config: Qwen3VLMoeTextConfig, top_k: int | None = None) -> None:
        selected_top_k = config.num_experts_per_tok if top_k is None else top_k
        assert 1 <= selected_top_k <= config.num_experts, (
            f"top_k must be in [1, {config.num_experts}], got {selected_top_k}."
        )
        super().__init__(
            config.hidden_size,
            config.moe_intermediate_size,
            config.num_experts,
            selected_top_k,
            config.hidden_act,
        )


class Qwen3VLMoeTextExpertsNaive(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_up_proj = nn.Parameter(
            torch.empty(config.num_experts, config.hidden_size, 2 * config.moe_intermediate_size)
        )
        self.down_proj = nn.Parameter(torch.empty(config.num_experts, config.moe_intermediate_size, config.hidden_size))
        self.act_fn = ACT2FN[config.hidden_act]

        self.num_experts = config.num_experts
        self.hidden_size = config.hidden_size

    def forward(
        self,
        hidden_states: torch.Tensor,  # [num_tokens,hidden_size]
        topk_scores: torch.Tensor,  # [num_tokens,top_k]
        expert_indices: torch.Tensor,  # [num_tokens,top_k]
        num_tokens_per_expert: torch.Tensor,  # [num_experts]
        token_mask: torch.Tensor | None,  # [num_tokens]
    ) -> torch.Tensor:  # [num_tokens,hidden_size]
        """
        When training it is more efficient to just loop over the experts and compute the output for each expert
        as otherwise the memory would explode.

        For inference we can sacrifice some memory and compute the output for all experts at once. By repeating the inputs.

        Args:
            hidden_states (torch.Tensor): (batch_size * token_num, hidden_size)
            routing_weights (torch.Tensor): (batch_size * token_num, top_k)
            expert_indices (torch.Tensor): (batch_size * token_num, top_k)
            num_tokens_per_expert (torch.Tensor): (num_experts,)
            token_mask (torch.Tensor | None): (batch_size * token_num,) boolean, false on padding
                rows. Ignored here: this reference path computes every row and relies on the caller
                having given the padding rows a zero score, which contributes nothing to the sum.
                Skipping their compute is a property of the grouped-GEMM path only.

        Returns:
            torch.Tensor: (batch_size * seq_len, hidden_size)
        """
        del num_tokens_per_expert
        del token_mask
        assert hidden_states.ndim == 2, "hidden_states must be of shape (batch_size * seq_len, hidden_size)"
        assert hidden_states.shape[1] == self.hidden_size, (
            "hidden_states must be of shape (batch_size * seq_len, hidden_size)"
        )
        routing_weights = torch.zeros(
            hidden_states.shape[0],
            self.num_experts,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )  # [num_tokens,num_experts]
        routing_weights = routing_weights.scatter_(1, expert_indices, topk_scores)  # [num_tokens,num_experts]

        if self.training:
            next_states = torch.zeros_like(hidden_states)  # [num_tokens,hidden_size]
            with torch.no_grad():
                expert_mask = torch.nn.functional.one_hot(
                    expert_indices, num_classes=self.num_experts
                )  # [num_tokens,top_k,num_experts]
                expert_mask = expert_mask.permute(2, 1, 0)  # [num_experts,top_k,num_tokens]
                # we sum on the top_k and on the sequence length to get which experts
                # are hit this time around
                expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
            for expert_idx in expert_hit[:]:
                with torch.no_grad():
                    _, token_idx = torch.where(expert_mask[expert_idx[0]])
                current_state = hidden_states[token_idx]  # [num_expert_tokens,hidden_size]
                gate_up = current_state @ self.gate_up_proj[expert_idx]  # [num_expert_tokens,2*moe_intermediate_size]
                gate, up = gate_up.chunk(2, dim=-1)  # 2x [num_expert_tokens,moe_intermediate_size]
                gated_output = up * self.act_fn(gate)  # [num_expert_tokens,moe_intermediate_size]
                out = gated_output @ self.down_proj[expert_idx]  # [num_expert_tokens,hidden_size]
                weighted_output = out[0] * routing_weights[token_idx, expert_idx, None]
                assert weighted_output.dtype == hidden_states.dtype
                next_states.index_add_(0, token_idx, weighted_output)
        else:
            hidden_states = hidden_states.repeat(self.num_experts, 1)  # [num_experts*num_tokens,hidden_size]
            hidden_states = hidden_states.view(
                self.num_experts, -1, self.hidden_size
            )  # [num_experts,num_tokens,hidden_size]
            gate_up = torch.bmm(hidden_states, self.gate_up_proj)  # [num_experts,num_tokens,2*moe_intermediate_size]
            gate, up = gate_up.chunk(
                2, dim=-1
            )  # not supported for DTensors  # 2x [num_experts,num_tokens,moe_intermediate_size]
            next_states = torch.bmm((up * self.act_fn(gate)), self.down_proj)  # [num_experts,num_tokens,hidden_size]
            next_states = next_states * routing_weights.transpose(0, 1).unsqueeze(
                dim=-1
            )  # [num_experts,num_tokens,hidden_size]
            next_states = next_states.sum(dim=0)  # [num_tokens,hidden_size]
        return next_states

    def init_weights(self, buffer_device: torch.device):
        nn.init.normal_(self.gate_up_proj, mean=0.0, std=0.02)
        nn.init.normal_(self.down_proj, mean=0.0, std=0.02)


def create_text_experts(
    config: Qwen3VLMoeTextConfig,
    implementation_type: str = "naive",
    top_k: int | None = None,
) -> nn.Module:
    if implementation_type == "naive":
        return Qwen3VLMoeTextExpertsNaive(config)
    elif implementation_type == "grouped_mm":
        return MoEExperts(
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok if top_k is None else top_k,
            activation=config.hidden_act,
            backend_config=MoEBackendConfig(
                backend=getattr(config, "moe_backend", "native"),
                precision=getattr(config, "moe_precision", "bf16"),
            ),
        )
    else:
        raise ValueError(f"Invalid implementation: {implementation_type}")
