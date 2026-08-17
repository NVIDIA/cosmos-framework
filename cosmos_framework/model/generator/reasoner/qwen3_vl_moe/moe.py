# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from typing import Callable

import torch
import torch.nn as nn
from transformers.activations import ACT2FN

from cosmos_framework.model.generator.reasoner.qwen3_vl_moe.configuration_qwen3_vl_moe import (
    Qwen3VLMoeTextConfig,
)
from cosmos_framework.model.generator.reasoner.qwen3_vl_moe.moe_kernels import (
    TOKEN_GROUP_ALIGN_SIZE_M,
    _generate_permute_indices,
)


def _run_experts_grouped_mm(
    gate_up_proj: torch.Tensor,  # [num_experts,hidden_size,2*moe_intermediate_size]
    down_proj: torch.Tensor,  # [num_experts,moe_intermediate_size,hidden_size]
    act_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,  # [num_tokens,hidden_size]  (tokens sorted by expert)
    num_tokens_per_expert: torch.Tensor,  # [num_experts]
    scores: torch.Tensor,  # [padded_len]
) -> torch.Tensor:  # [num_tokens,hidden_size]
    """
    This function runs the gate/up/down projection in a grouped matrix multiplication fashion.

    Args:
        gate_up_proj (torch.Tensor): (num_experts, hidden_size, 2 * moe_intermediate_size)
        down_proj (torch.Tensor): (num_experts, moe_intermediate_size, hidden_size)
        x (torch.Tensor): (batch_size * seq_len, hidden_size)
        num_tokens_per_expert (torch.Tensor): (num_experts,)
        scores (torch.Tensor): (num_tokens,)

    Returns:
        torch.Tensor: (batch_size * seq_len, hidden_size)
    """
    offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)  # [num_experts]
    h = torch._grouped_mm(x, gate_up_proj, offs=offsets)  # [num_tokens,2*moe_intermediate_size]
    h = torch.chunk(h, chunks=2, dim=-1)  # 2x [num_tokens,moe_intermediate_size]
    h = act_fn(h[0]) * h[1] * scores.unsqueeze(-1)  # [num_tokens,moe_intermediate_size]
    return torch._grouped_mm(h, down_proj, offs=offsets)  # [num_tokens,hidden_size]


class Qwen3VLMoeTextExpertsGroupedMm(nn.Module):
    def __init__(self, config: Qwen3VLMoeTextConfig, top_k: int | None = None) -> None:
        super().__init__()
        self.gate_up_proj = nn.Parameter(
            torch.empty(config.num_experts, config.hidden_size, 2 * config.moe_intermediate_size)
        )
        self.down_proj = nn.Parameter(torch.empty(config.num_experts, config.moe_intermediate_size, config.hidden_size))
        self.act_fn = ACT2FN[config.hidden_act]

        self.num_experts = config.num_experts
        self.hidden_size = config.hidden_size
        self.top_k = config.num_experts_per_tok if top_k is None else top_k
        assert 1 <= self.top_k <= self.num_experts, f"top_k must be in [1, {self.num_experts}], got {self.top_k}."

    def forward(
        self,
        hidden_states: torch.Tensor,  # [num_tokens,hidden_size]
        topk_scores: torch.Tensor,  # [num_tokens,top_k]
        expert_indices: torch.Tensor,  # [num_tokens,top_k]
        num_tokens_per_expert: torch.Tensor,  # [num_experts]
        token_mask: torch.Tensor | None,  # [num_tokens]
    ) -> torch.Tensor:  # [num_tokens,hidden_size]
        """
        This module obtains the output of the experts by routing the tokens
        to the experts and then performing a weighted sum of the output of the experts.

        Args:
            hidden_states (torch.Tensor): (batch_size * seq_len, hidden_size)
            topk_scores (torch.Tensor): (batch_size * seq_len, top_k)
            expert_indices (torch.Tensor): (batch_size * seq_len, top_k)
            num_tokens_per_expert (torch.Tensor): (num_experts,) number of tokens each
                expert is to process. It counts exactly the number of real tokens that reach
                the GEMM.
            token_mask (torch.Tensor | None): (batch_size * seq_len,) boolean, false on the tokens
                that are padding rather than real tokens. Those tokens are sent to a sentinel group
                past the last expert, which the stable sort places after every real group and the
                permutation therefore never reaches, so neither their compute nor their values reach
                the GEMM. ``None`` treats every token as a real token.

        Returns:
            torch.Tensor: (batch_size * seq_len, hidden_size)
        """
        assert topk_scores.shape == expert_indices.shape, (
            f"topk_scores and expert_indices must have matching shapes, got "
            f"{tuple(topk_scores.shape)} and {tuple(expert_indices.shape)}."
        )
        assert expert_indices.shape[-1] == self.top_k, (
            f"Grouped-MM top_k={self.top_k} does not match runtime routing top_k={expert_indices.shape[-1]}."
        )
        num_tokens, dim = hidden_states.shape
        if token_mask is not None:
            # A group index past the last expert, rather than -1: the sort has to leave the real
            # groups starting where ``num_tokens_per_expert`` says they do, and a negative key
            # would push the padding in front of expert 0 instead of behind expert E-1.
            expert_indices = expert_indices.masked_fill(~token_mask.unsqueeze(1), self.num_experts)
        topk_scores_sorted, token_indices_sorted = self._reorder_tokens(
            topk_scores,
            expert_indices,
        )
        # topk_scores_sorted: [num_tokens*top_k]
        # token_indices_sorted: [num_tokens*top_k]

        # Build padded permutation indices
        num_experts = num_tokens_per_expert.shape[0]
        alignment = TOKEN_GROUP_ALIGN_SIZE_M
        padded_size = num_tokens * self.top_k + num_experts * alignment
        padded_size = ((padded_size + alignment - 1) // alignment) * alignment

        permuted_indices, padded_num_tokens_per_expert = _generate_permute_indices(
            num_tokens_per_expert,
            num_experts,
            padded_size,
            alignment,
        )

        # Compose: permuted_indices indexes into sorted order,
        # token_indices_sorted maps sorted→original. Compose them:
        #
        # torch.full, not torch.tensor([num_tokens], device=...): the latter stages the value in
        # pageable host memory, and a pageable H2D copy is cudaMemcpyAsync plus a blocking
        # cudaStreamSynchronize. That synchronize drains everything this layer has queued, once
        # per MoE layer (94 of them per forward on the 235B), which costs the CPU the run-ahead
        # FSDP2 needs: with no explicit forward prefetch, a block's all-gather is only issued when
        # the CPU reaches its pre-forward hook, so a CPU pinned to the GPU serializes comm behind
        # compute. full() fills on device with the scalar as a kernel argument — nothing waits.
        sentinel = torch.full(
            (1,), num_tokens, dtype=token_indices_sorted.dtype, device=hidden_states.device
        )  # for padding slots
        token_indices_ext = torch.cat([token_indices_sorted, sentinel])
        combined_indices = token_indices_ext[permuted_indices.long()]
        combined_indices = combined_indices.unsqueeze(-1).expand(-1, dim)

        # Pad scores with a zero sentinel so padding slots contribute nothing
        scores_ext = torch.cat([topk_scores_sorted, topk_scores_sorted.new_zeros(1)])
        combined_scores = scores_ext[permuted_indices.long()]  # [padded_len]

        # Single gather (with a zero-padded sentinel row)
        input_padded = torch.cat([hidden_states, hidden_states.new_zeros(1, dim)])
        routed_input = input_padded.gather(dim=0, index=combined_indices)

        # Run experts
        routed_output = _run_experts_grouped_mm(
            self.gate_up_proj,
            self.down_proj,
            self.act_fn,
            routed_input,
            padded_num_tokens_per_expert,
            combined_scores,
        )

        output_padded = torch.zeros_like(input_padded)
        output_padded.scatter_add_(dim=0, index=combined_indices, src=routed_output)
        return output_padded[:-1]

    def _reorder_tokens(
        self,
        topk_scores: torch.Tensor,  # [num_tokens,top_k]
        expert_indices: torch.Tensor,  # [num_tokens,top_k]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reorder tokens into expert-grouped order via argsort.

        Returns:
            topk_scores_sorted: [num_tokens*top_k] scores in expert-grouped order.
            token_indices_sorted: [num_tokens*top_k] original token indices in
                expert-grouped order.
        """
        token_indices_sorted = torch.argsort(expert_indices.view(-1), stable=True)  # [num_tokens*top_k]
        topk_scores_sorted = topk_scores.view(-1)[token_indices_sorted]  # [num_tokens*top_k]
        token_indices_sorted = token_indices_sorted // self.top_k  # [num_tokens*top_k]
        return topk_scores_sorted, token_indices_sorted

    def init_weights(self, buffer_device: torch.device):
        nn.init.normal_(self.gate_up_proj, mean=0.0, std=0.02)
        nn.init.normal_(self.down_proj, mean=0.0, std=0.02)


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
        return Qwen3VLMoeTextExpertsGroupedMm(config, top_k=top_k)
    else:
        raise ValueError(f"Invalid implementation: {implementation_type}")
