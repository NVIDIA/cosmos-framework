# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from collections.abc import Callable

import torch
from torch import nn
from transformers.activations import ACT2FN

from cosmos_framework.model.generator.moe.native_kernels import TOKEN_GROUP_ALIGN_SIZE_M, _generate_permute_indices


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


def grouped_mm(
    gate_up_proj: torch.Tensor,  # [E,D,2I]
    down_proj: torch.Tensor,  # [E,I,D]
    act_fn: Callable[[torch.Tensor], torch.Tensor],
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
    num_experts = gate_up_proj.shape[0]
    top_k = expert_indices.shape[1]
    assert topk_scores.shape == expert_indices.shape, (
        f"topk_scores and expert_indices must have matching shapes, got "
        f"{tuple(topk_scores.shape)} and {tuple(expert_indices.shape)}."
    )
    num_tokens, dim = hidden_states.shape
    if token_mask is not None:
        # A group index past the last expert, rather than -1: the sort has to leave the real
        # groups starting where ``num_tokens_per_expert`` says they do, and a negative key
        # would push the padding in front of expert 0 instead of behind expert E-1.
        expert_indices = expert_indices.masked_fill(~token_mask.unsqueeze(1), num_experts)  # [T,K]
    topk_scores_sorted, token_indices_sorted = _reorder_tokens(
        top_k,
        topk_scores,
        expert_indices,
    )
    # topk_scores_sorted: [num_tokens*top_k]
    # token_indices_sorted: [num_tokens*top_k]

    # Build padded permutation indices
    num_experts = num_tokens_per_expert.shape[0]
    alignment = TOKEN_GROUP_ALIGN_SIZE_M
    padded_size = num_tokens * top_k + num_experts * alignment
    padded_size = ((padded_size + alignment - 1) // alignment) * alignment

    permuted_indices, padded_num_tokens_per_expert = _generate_permute_indices(
        num_tokens_per_expert,
        num_experts,
        padded_size,
        alignment,
    )  # [padded_len], [E]

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
    )  # [1], for padding slots
    token_indices_ext = torch.cat([token_indices_sorted, sentinel])  # [T*K+1]
    combined_indices = token_indices_ext[permuted_indices.long()]  # [padded_len]
    combined_indices = combined_indices.unsqueeze(-1).expand(-1, dim)  # [padded_len,D]

    # Pad scores with a zero sentinel so padding slots contribute nothing
    scores_ext = torch.cat([topk_scores_sorted, topk_scores_sorted.new_zeros(1)])  # [T*K+1]
    combined_scores = scores_ext[permuted_indices.long()]  # [padded_len]

    # Single gather (with a zero-padded sentinel row)
    input_padded = torch.cat([hidden_states, hidden_states.new_zeros(1, dim)])  # [T+1,D]
    routed_input = input_padded.gather(dim=0, index=combined_indices)  # [padded_len,D]

    # Run experts
    routed_output = _run_experts_grouped_mm(
        gate_up_proj,
        down_proj,
        act_fn,
        routed_input,
        padded_num_tokens_per_expert,
        combined_scores,
    )  # [padded_len,D]

    output_padded = torch.zeros_like(input_padded)  # [T+1,D]
    output_padded.scatter_add_(dim=0, index=combined_indices, src=routed_output)  # [T+1,D]
    return output_padded[:-1]  # [T,D]


def _reorder_tokens(
    top_k: int,
    topk_scores: torch.Tensor,  # [num_tokens,top_k]
    expert_indices: torch.Tensor,  # [num_tokens,top_k]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reorder tokens into expert-grouped order via argsort.

    Returns:
        topk_scores_sorted: [num_tokens*top_k] scores in expert-grouped order.
        token_indices_sorted: [num_tokens*top_k] original token indices in
            expert-grouped order.
    """
    token_indices_sorted = torch.argsort(expert_indices.reshape(-1), stable=True)  # [num_tokens*top_k]
    topk_scores_sorted = topk_scores.reshape(-1)[token_indices_sorted]  # [num_tokens*top_k]
    token_indices_sorted = token_indices_sorted // top_k  # [num_tokens*top_k]
    return topk_scores_sorted, token_indices_sorted


def naive(
    gate_up_proj: torch.Tensor,  # [E,D,2I]
    down_proj: torch.Tensor,  # [E,I,D]
    act_fn: Callable[[torch.Tensor], torch.Tensor],
    hidden_states: torch.Tensor,  # [T,D]
    scores: torch.Tensor,  # [T,K]
    indices: torch.Tensor,  # [T,K]
) -> torch.Tensor:  # [T,D]
    """Portable PyTorch reference, also used for CPU and empty inputs.

    Like Qwen's naive experts, evaluate each expert and combine its weighted
    outputs. A loop bounds intermediate memory; scatter_add handles repeated IDs.
    """
    routing_weights = scores.new_zeros((hidden_states.shape[0], gate_up_proj.shape[0]))  # [T,E]
    routing_weights = routing_weights.scatter_add(1, indices.long(), scores)  # [T,E]
    output = torch.zeros_like(hidden_states)  # [T,D]
    for expert in range(gate_up_proj.shape[0]):
        gate, up = (hidden_states @ gate_up_proj[expert]).chunk(2, dim=-1)  # 2x[T,I]
        expert_output = (act_fn(gate) * up) @ down_proj[expert]  # [T,D]
        output = output + expert_output * routing_weights[:, expert, None].to(hidden_states.dtype)  # [T,D]
    return output  # [T,D]


class NativeMoEExperts(nn.Module):
    """Canonical expert parameters and the shared native grouped-MM implementation.

    The Qwen wrapper retains its config-based constructor and parameter names.
    The forward and reorder methods also support binding to HF expert instances.
    """

    gate_up_proj: nn.Parameter
    down_proj: nn.Parameter
    act_fn: Callable[[torch.Tensor], torch.Tensor]
    num_experts: int
    hidden_size: int
    intermediate_size: int
    activation: str
    top_k: int

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        if min(hidden_size, intermediate_size, num_experts, top_k) <= 0 or top_k > num_experts:
            raise ValueError("Expert dimensions must be positive and top_k must not exceed num_experts")
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.activation = activation
        self.act_fn = ACT2FN[activation]
        self.gate_up_proj = nn.Parameter(torch.empty(num_experts, hidden_size, 2 * intermediate_size))  # [E,D,2I]
        self.down_proj = nn.Parameter(torch.empty(num_experts, intermediate_size, hidden_size))  # [E,I,D]

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
        assert expert_indices.shape[-1] == self.top_k, (
            f"Grouped-MM top_k={self.top_k} does not match runtime routing top_k={expert_indices.shape[-1]}."
        )
        return grouped_mm(
            self.gate_up_proj,
            self.down_proj,
            self.act_fn,
            hidden_states,
            topk_scores,
            expert_indices,
            num_tokens_per_expert,
            token_mask,
        )  # [num_tokens,hidden_size]

    def _reorder_tokens(
        self,
        topk_scores: torch.Tensor,  # [num_tokens,top_k]
        expert_indices: torch.Tensor,  # [num_tokens,top_k]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compatibility entry point for existing Qwen expert callers.

        Reorder tokens into expert-grouped order via argsort.

        Returns:
            topk_scores_sorted: [num_tokens*top_k] scores in expert-grouped order.
            token_indices_sorted: [num_tokens*top_k] original token indices in
                expert-grouped order.
        """
        return _reorder_tokens(self.top_k, topk_scores, expert_indices)  # 2x[num_tokens*top_k]

    def init_weights(self, buffer_device: torch.device) -> None:
        nn.init.normal_(self.gate_up_proj, mean=0.0, std=0.02)  # [E,D,2I]
        nn.init.normal_(self.down_proj, mean=0.0, std=0.02)  # [E,I,D]
