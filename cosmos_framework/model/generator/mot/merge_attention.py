# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Activation-checkpoint-compatible LSE merge shared by interactive attention paths."""

from typing import Any

import torch


class _ACSafeMergeAttentionsFn(torch.autograd.Function):
    """AC-compatible drop-in for NATTEN's ``MergeAttentionsAutogradFn``.

    NATTEN's backward indexes ``ctx.saved_tensors`` in multiple slices
    (``[:2]``, ``[2 : N+2]``, ``[N+2:]``), and each indexing access fires
    the non-reentrant ``torch.utils.checkpoint`` unpack hook for *every*
    saved tensor. The hook only permits one unpack per saved tensor, so
    activation checkpointing + NATTEN merge_attentions raises
    ``CheckpointError: Unpack is being triggered for a tensor that was
    already unpacked once`` (see the replayed-LSE + AC=full long-video
    TF path).

    This version preserves the same forward math and the same
    storage-patching backward contract (see :class:`MergeAttentionsBridge`
    docstring for the full description), but reads ``ctx.saved_tensors``
    exactly once.  Numerics match ``naive_merge_attentions`` (iterative
    pairwise LSE rescale) — which is what NATTEN's kernel implements up
    to reduction order.
    """

    @staticmethod
    def forward(
        ctx: Any,
        num_components: int,
        *tensors: torch.Tensor,  # outputs: [B,Q,H,D], LSEs: [B,Q,H] or [B,Q,H,1]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = tensors[:num_components]
        lses = tensors[num_components:]
        output_dtype = outputs[0].dtype
        normalized_lses = [lse.squeeze(-1) if lse.ndim == 4 else lse for lse in lses]  # list[[B,Q,H]]

        merged_lse = normalized_lses[0]  # [B,Q,H]
        merged_out = outputs[0]  # [B,Q,H,D]
        for i in range(1, num_components):
            new_lse = torch.logaddexp(merged_lse, normalized_lses[i])  # [B,Q,H]
            w_old = torch.exp(merged_lse - new_lse).unsqueeze(-1)  # [B,Q,H,1]
            w_new = torch.exp(normalized_lses[i] - new_lse).unsqueeze(-1)  # [B,Q,H,1]
            merged_out = w_old * merged_out + w_new * outputs[i]  # [B,Q,H,D]
            merged_lse = new_lse  # [B,Q,H]
        merged_out = merged_out.to(output_dtype)  # [B,Q,H,D]

        ctx.save_for_backward(merged_out, merged_lse, *outputs, *lses)
        ctx.num_components = num_components
        return merged_out, merged_lse  # [B,Q,H,D], [B,Q,H]

    @staticmethod
    def backward(
        ctx: Any,
        grad_merged_out: torch.Tensor,  # [B,Q,H,D]
        grad_merged_lse: torch.Tensor,  # [B,Q,H]
    ) -> tuple[torch.Tensor | None, ...]:
        # Single access — avoid retriggering the AC unpack hook for any saved tensor.
        saved = ctx.saved_tensors
        merged_out = saved[0]  # [B,Q,H,D]
        merged_lse = saved[1]  # [B,Q,H]
        num = ctx.num_components
        outputs = saved[2 : 2 + num]
        lses = saved[2 + num : 2 + 2 * num]

        # Patch each component's storage with the merged O / LSE.  The
        # upstream attention kernel's backward will read these as its
        # saved O / LSE and compute gradients as if it had produced the
        # merged output.  The original LSE shape is preserved (the
        # forward squeezes a trailing singleton, so we re-broadcast).
        for o in outputs:
            o.data.copy_(merged_out.data)  # [B,Q,H,D]
        for l in lses:
            if l.ndim == merged_lse.ndim + 1 and l.shape[-1] == 1:
                l.data.copy_(merged_lse.data.unsqueeze(-1))  # [B,Q,H,1]
            else:
                l.data.copy_(merged_lse.data)  # [B,Q,H]

        # Same upstream-grad contract as NATTEN: dL/dO_i = dL/dO_merged
        # for every component; dL/dLSE_i is forwarded unchanged for
        # parity (i4 attention treats LSE as non-differentiable, so this
        # gradient is silently dropped at the kernel boundary).
        grads = (None,) + (grad_merged_out,) * num + (grad_merged_lse,) * num
        return grads


def merge_attentions_ac_safe(
    outputs: list[torch.Tensor],  # list[[B,Q,H,D]]
    lse_tensors: list[torch.Tensor],  # list[[B,Q,H] or [B,Q,H,1]]
) -> tuple[torch.Tensor, torch.Tensor]:
    """AC-safe drop-in for ``cosmos_framework.model.attention.merge_attentions``.

    Use at call sites that live inside an activation-checkpointed module
    boundary.  Matches NATTEN's storage-patching backward contract so
    upstream i4 attention kernels (whose LSE is not differentiable)
    still receive correct gradients via their own saved O / LSE
    backward formulas.
    """
    assert len(outputs) == len(lse_tensors) >= 2
    return _ACSafeMergeAttentionsFn.apply(len(outputs), *outputs, *lse_tensors)  # [B,Q,H,D], [B,Q,H]
