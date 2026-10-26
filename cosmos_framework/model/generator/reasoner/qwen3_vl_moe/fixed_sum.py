# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Opt-in fixed-order FP32 dispatch/combine for grouped Qwen3-VL MoE experts.

The expert GEMMs and parameters are unchanged. Only top-k accumulation order
and accumulation precision change; this does not promise deterministic attention.

A token is routed to K experts. Combining their outputs, and adding the K input
gradients in dispatch backward, each requires a sum over those assignments.
This module gathers them in top-k order and accumulates in FP32 instead of using
unordered low-precision scatter additions. It is an optional Qwen MoE numerical
control, useful for CP comparisons but equally applicable to CP1. CP's shared
sequence/head communication does not perform this expert reduction.

``assignments[row] = token * K + slot`` maps an expert-grouped, padded row back
to a top-k assignment. ``tokens * K`` denotes a padding row. ``inverse`` maps
each real assignment back to its grouped row, or -1 for a masked assignment.
"""

from collections.abc import Callable
from typing import Any, cast

import torch
import triton
import triton.language as tl
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeTextSparseMoeBlock

from cosmos_framework.model.generator.moe.native import NativeMoEExperts
from cosmos_framework.model.generator.reasoner.qwen3_vl_moe.moe import (
    _run_experts_grouped_mm,
)
from cosmos_framework.model.generator.reasoner.qwen3_vl_moe.moe_kernels import (
    TOKEN_GROUP_ALIGN_SIZE_M,
    _generate_permute_indices,
)


@triton.jit
def _inverse_kernel(
    assignments: torch.Tensor, inverse: torch.Tensor, N: tl.constexpr, P: tl.constexpr, BLOCK: tl.constexpr
) -> None:
    """Invert real assignment IDs; padded rows must not overwrite the inverse."""
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    assignment = tl.load(assignments + row, row < P, other=N)
    tl.store(inverse + assignment, row, (row < P) & (assignment < N))


@triton.jit
def _combine_kernel(
    values: torch.Tensor,
    inverse: torch.Tensor,
    result: torch.Tensor,
    N: tl.constexpr,
    D: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    """Sum each token's K rows in slot order and FP32, then cast on output store."""
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    token, column = offset // D, offset % D
    total = tl.full((BLOCK,), 0, tl.float32)
    for k in range(K):
        row = tl.load(inverse + token * K + k, token < N, other=-1)
        value = tl.load(values + row * D + column, (token < N) & (row >= 0), other=0).to(tl.float32)
        total = total + value
    tl.store(result + offset, total, token < N)


def inverse_assignments(assignments: torch.Tensor, tokens: int, top_k: int) -> torch.Tensor:
    """Build the [tokens * top_k] lookup, leaving absent/masked assignments at -1."""
    inverse = torch.full((tokens * top_k,), -1, dtype=torch.int64, device=assignments.device)
    # Triton specializes Python integer launch arguments annotated tl.constexpr in the kernel.
    launch = cast(Callable[..., None], _inverse_kernel[(triton.cdiv(assignments.numel(), 256),)])
    launch(assignments, inverse, tokens * top_k, assignments.numel(), 256)
    return inverse


def fixed_sum(values: torch.Tensor, inverse: torch.Tensor, tokens: int, top_k: int) -> torch.Tensor:
    """Reduce grouped [padded_rows, hidden] values to [tokens, hidden] in their input dtype.

    Only accumulation is FP32; absent assignments contribute zero. Autograd is
    supplied by Dispatch/Combine because this Triton call has no backward itself.
    """
    result = values.new_empty((tokens, values.shape[1]))
    launch = cast(Callable[..., None], _combine_kernel[(triton.cdiv(result.numel(), 256),)])
    launch(values.contiguous(), inverse, result, tokens, values.shape[1], top_k, 256)
    return result


class Dispatch(torch.autograd.Function):
    """Gather token inputs into expert order; backward sums their K gradients.

    A sentinel zero token supplies alignment padding without routing a real token.
    """

    @staticmethod
    def forward(
        ctx: Any,
        hidden: torch.Tensor,
        assignments: torch.Tensor,
        inverse: torch.Tensor,
        top_k: int,
    ) -> torch.Tensor:
        ctx.save_for_backward(assignments, inverse)
        ctx.tokens, ctx.top_k = hidden.shape[0], top_k
        padded = torch.cat([hidden, hidden.new_zeros((1, hidden.shape[1]))])
        return padded.index_select(0, assignments // top_k)

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[torch.Tensor, None, None, None]:
        _, inverse = ctx.saved_tensors
        return fixed_sum(grad, inverse, ctx.tokens, ctx.top_k), None, None, None


class Combine(torch.autograd.Function):
    """Sum expert outputs per token; backward copies its gradient to each assignment.

    Routing scores have already been applied by the grouped expert computation.
    """

    @staticmethod
    def forward(
        ctx: Any,
        values: torch.Tensor,
        assignments: torch.Tensor,
        inverse: torch.Tensor,
        tokens: int,
        top_k: int,
    ) -> torch.Tensor:
        ctx.save_for_backward(assignments)
        ctx.top_k = top_k
        return fixed_sum(values, inverse, tokens, top_k)

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[torch.Tensor, None, None, None, None]:
        (assignments,) = ctx.saved_tensors
        padded = torch.cat([grad, grad.new_zeros((1, grad.shape[1]))])
        return padded.index_select(0, assignments // ctx.top_k), None, None, None, None


def deterministic_experts_forward(
    self: NativeMoEExperts,
    hidden_states: torch.Tensor,
    topk_scores: torch.Tensor,
    expert_indices: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    token_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Reuse native grouped expert GEMMs with fixed-order dispatch/combine reductions.

    The stable expert ordering and inverse lookup preserve top-k slot identity
    through alignment padding. Masked tokens have no real expert assignments.
    Expert weights, routing scores and the auxiliary objective are unchanged;
    the name refers to the reductions, not whole-model determinism.
    """
    tokens, _ = hidden_states.shape
    top_k = self.top_k
    if token_mask is not None:
        expert_indices = expert_indices.masked_fill(~token_mask.unsqueeze(1), self.num_experts)
    order = torch.argsort(expert_indices.reshape(-1), stable=True)
    sorted_scores = topk_scores.reshape(-1)[order]
    alignment = TOKEN_GROUP_ALIGN_SIZE_M
    experts = num_tokens_per_expert.shape[0]
    padded_size = ((tokens * top_k + experts * alignment + alignment - 1) // alignment) * alignment
    permutation, counts = _generate_permute_indices(num_tokens_per_expert, experts, padded_size, alignment)
    order_ext = torch.cat([order, order.new_full((1,), tokens * top_k)])
    assignments = order_ext[permutation.long()]
    inverse = inverse_assignments(assignments, tokens, top_k)
    scores_ext = torch.cat([sorted_scores, sorted_scores.new_zeros(1)])
    scores = scores_ext[permutation.long()]
    routed = Dispatch.apply(hidden_states, assignments, inverse, top_k)
    output = _run_experts_grouped_mm(self.gate_up_proj, self.down_proj, self.act_fn, routed, counts, scores)
    return Combine.apply(output, assignments, inverse, tokens, top_k)


def configure_qwen3_vl_moe_fixed_sum(model: torch.nn.Module) -> int:
    """Enable the optional expert reduction after grouped-mm setup, before AC/FSDP.

    Only this model instance changes. The default recipe does not opt in, and no
    parameter or checkpoint layout changes when the option is enabled.
    """
    blocks = [module for module in model.modules() if isinstance(module, Qwen3VLMoeTextSparseMoeBlock)]
    if not blocks:
        raise ValueError("Fixed MoE summation requires Qwen3-VL-MoE blocks")
    if any(not hasattr(block.experts, "_reorder_tokens") for block in blocks):
        raise ValueError("Configure grouped-mm experts before fixed MoE summation")
    for block in blocks:
        experts = block.experts
        experts.forward = deterministic_experts_forward.__get__(experts, type(experts))
    return len(blocks)
