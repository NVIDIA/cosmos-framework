# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Keep ``merge_attentions``' data-pointer contract across a shape change.

Its own module because two unrelated attention paths need it -- the interactive tree's
``three_way_attention_with_kv_cache`` and ``multiview_maskless_attention``'s two folds -- and because
putting it in either one would make the other import that one for a utility that belongs to
neither.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

BridgeFn = Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


class MergeAttentionsBridge(torch.autograd.Function):
    """Autograd bridge that preserves ``merge_attentions``' data-pointer
    contract across an arbitrary invertible shape-changing op.

    ``merge_attentions`` (NATTEN's ``MergeAttentionsAutogradFn``, see
    ``data_local/attn_merge.py``) implements its backward via a hack:
    instead of computing real gradients w.r.t. its inputs, it writes the
    *merged* output and LSE back into each input tensor's storage via
    ``.data.copy_()`` and returns the upstream gradient unchanged.  The
    attention kernel that produced the input then reads the patched
    storage as its saved ``O`` / ``LSE`` during its own backward, and its
    standard backward formula then computes the gradient *as if* the
    kernel had produced the merged output.

    This contract is broken whenever a tensor-allocating op (e.g.
    ``torch.cat`` to insert a zero-padded frame 0) sits between the
    attention kernel and ``merge_attentions``: the op's result has its
    own storage, so ``merge_attentions``' ``.data.copy_()`` patches the
    op's output storage, not the kernel's saved output → the kernel's
    backward then runs against unpatched data and produces gradients
    that don't account for the merge.

    This Function rebridges the contract across any invertible action
    on the inner ``(out, lse)`` pair.  The action is supplied as two
    callables:

    - ``forward_fn(out_inner, lse_inner) -> (out_full, lse_full)``: the
      invertible action applied in the forward pass (e.g. cat-pad a
      frame, permute, scatter, …).  ``out_full`` / ``lse_full`` are the
      tensors that ``merge_attentions`` will receive (and later patch in
      its backward).
    - ``inverse_fn(out_full, lse_full) -> (out_inner, lse_inner)``: the
      exact inverse — undoes ``forward_fn`` so that ``inverse_fn ∘
      forward_fn`` is the identity on the inner tensors.

    For ``forward_fn`` that is linear with constant-fill (cat-pad,
    permutation, scatter with zeros, …), the *gradient* w.r.t. the
    inner input is also ``inverse_fn`` applied to the upstream gradient
    — so the same callable serves both backward roles below.  If your
    forward is not in this class (e.g. it has trainable parameters, or
    is non-linear), do not use this bridge.

    Backward:
      Runs *after* ``merge_attentions``' backward (autograd is
      reverse-order), at which point the outer tensors have already
      been patched.  We then apply ``inverse_fn`` to the patched outer
      data and ``.data.copy_()`` it into the inner kernel's saved
      output / LSE storage.  When the inner kernel's backward runs
      next, it reads the patched data and produces gradients relative
      to the merged attention.  We also return ``inverse_fn`` of the
      upstream gradient as the gradient w.r.t. the inner inputs.
    """

    @staticmethod
    def forward(
        ctx,
        out_inner: torch.Tensor,
        lse_inner: torch.Tensor,
        forward_fn: BridgeFn,
        inverse_fn: BridgeFn,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out_full, lse_full = forward_fn(out_inner, lse_inner)
        out_full = out_full.contiguous()
        lse_full = lse_full.contiguous()
        # Save BOTH the inner kernel outputs (target of the .data.copy_ back)
        # AND the outer tensors (source of the patched data, as patched
        # by merge_attentions.backward before our backward runs).
        ctx.save_for_backward(out_inner, lse_inner, out_full, lse_full)
        ctx.inverse_fn = inverse_fn
        return out_full, lse_full

    @staticmethod
    def backward(
        ctx,
        grad_out_full: torch.Tensor,
        grad_lse_full: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, None, None]:
        out_inner, lse_inner, out_full, lse_full = ctx.saved_tensors
        inverse_fn: BridgeFn = ctx.inverse_fn
        # By now merge_attentions.backward has already run and patched
        # out_full.data / lse_full.data with the merged output / LSE.
        # Apply inverse_fn to recover the data corresponding to the inner
        # attention's range and write it into the inner kernel's saved
        # output / LSE so the kernel's backward (which runs after ours)
        # reads the merged data.
        patched_out_inner, patched_lse_inner = inverse_fn(out_full, lse_full)
        out_inner.data.copy_(patched_out_inner.data)
        lse_inner.data.copy_(patched_lse_inner.data)
        # For linear-with-constant-fill forward_fn (cat-pad, permute,
        # scatter-with-zeros, …), the backward gradient operator equals
        # inverse_fn.  (Constant rows added by forward_fn are not
        # functions of the inner inputs, so their gradient does not flow
        # back; the remaining rows pass through.)
        grad_out_inner, grad_lse_inner = inverse_fn(grad_out_full, grad_lse_full)
        return grad_out_inner, grad_lse_inner, None, None
