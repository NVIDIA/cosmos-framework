# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Keep ``merge_attentions``' data-pointer contract across a shape change.

Its own module because two unrelated attention paths need it -- the interactive tree's
``three_way_attention_with_memory`` and ``multiview_maskless_attention``'s two folds -- and because
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

    For shape-only ``forward_fn`` (cat-pad, permutation, scatter with zeros,
    …), the gradient operator is also ``inverse_fn``. For a fixed invertible
    linear feature transform, supply ``gradient_fn`` separately: restoring
    saved outputs needs the inverse, whereas gradients need the transpose.
    Transforms with trainable parameters or nonlinear transforms are unsupported.

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
        gradient_fn: BridgeFn | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out_full, lse_full = forward_fn(out_inner, lse_inner)
        out_full = out_full.contiguous()
        lse_full = lse_full.contiguous()
        # Save BOTH the inner kernel outputs (target of the .data.copy_ back)
        # AND the outer tensors (source of the patched data, as patched
        # by merge_attentions.backward before our backward runs).
        ctx.save_for_backward(out_inner, lse_inner, out_full, lse_full)
        ctx.inverse_fn = inverse_fn
        ctx.gradient_fn = inverse_fn if gradient_fn is None else gradient_fn
        ctx.num_inputs = len(ctx.needs_input_grad)
        return out_full, lse_full

    @staticmethod
    def backward(
        ctx,
        grad_out_full: torch.Tensor,
        grad_lse_full: torch.Tensor,
    ) -> tuple[torch.Tensor | None, ...]:
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
        # Shape-only transforms share their inverse and gradient operator.
        # Camera SE(3) output transforms instead use the transpose for gradients.
        # Constant rows added by forward_fn do not propagate gradients.
        grad_out_inner, grad_lse_inner = ctx.gradient_fn(grad_out_full, grad_lse_full)
        return (grad_out_inner, grad_lse_inner, None, None, None)[: ctx.num_inputs]


class DisjointQueriesBridge(torch.autograd.Function):
    """Rejoin windowing's sensor and control query partitions in packed order.

    Temporal windowing splits same-view attention into four kernels because each query/key
    edge can have a different frame window. The two kernels sharing sensor queries are first
    merged over their alternative key sets, as are the two kernels sharing control queries::

        Sensor->Sensor -----+
                            +-- inner merge_attentions --> sensor_out ----+
        Sensor->Control ----+                                            |
                                                                         +-- DisjointQueriesBridge
        Control->Sensor ----+                                            |       |
                            +-- inner merge_attentions --> control_out ---+       v
        Control->Control ---+                                             same_view_out
                                                                                 |
        cross_view_out ----------------------------------------------------------+-- outer
        caption_out -------------------------------------------------------------+   merge_attentions

    ``sensor_out`` and ``control_out`` cover disjoint *query* rows, so combining them is not an
    attention merge: each packed row has exactly one source. This bridge scatters the two
    populations into ``same_view_out`` so the pre-existing outer merge can combine same-view,
    cross-view, and caption key sets for every query.

    The current multiview path needs this bridge only when temporal windowing has split paired
    sensor/control attention into separate kernels. The ordinary non-windowed path keeps both
    query roles in one same-view kernel.

    The custom backward is required by :func:`merge_attentions`' storage-patching contract.
    The outer merge patches ``same_view_out`` and its LSE with the globally merged values.
    Backward gathers those patched rows into ``sensor_out`` and ``control_out`` before their inner
    merges run, allowing the patch to continue to the four attention kernels. It separately
    gathers the ordinary upstream gradients into the two query populations.
    """

    @staticmethod
    def forward(
        ctx,
        first_out: torch.Tensor,  # [1,N_first,H,D]
        first_lse: torch.Tensor,  # [1,N_first,H]
        second_out: torch.Tensor,  # [1,N_second,H,D]
        second_lse: torch.Tensor,  # [1,N_second,H]
        first_gather: torch.Tensor,  # [N_first]
        second_gather: torch.Tensor,  # [N_second]
        output_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:  # [1,N,H,D], [1,N,H]
        """Scatter the two query populations into one packed output and LSE.

        In the windowed multiview caller, ``first_*`` contains sensor-query results after the
        Sensor-to-Sensor/Sensor-to-Control inner merge, while ``second_*`` contains control-query
        results after the Control-to-Sensor/Control-to-Control inner merge. ``first_gather`` and
        ``second_gather`` give each population's rows in the original packed GEN stream.

        The returned tensors have ``output_tokens`` rows in packed order. Rows not covered by
        either population, such as padding, receive zero output and minimum-finite LSE so they
        contribute no weight to the subsequent outer ``merge_attentions`` call. The original
        inputs and the packed results are saved so backward can propagate both storage patches
        and ordinary gradients across this scatter.
        """
        out_full = first_out.new_zeros(1, output_tokens, first_out.shape[-2], first_out.shape[-1])  # [1,N,H,D]
        lse_full = first_lse.new_full(  # [1,N,H]
            (1, output_tokens, first_lse.shape[-1]), torch.finfo(first_lse.dtype).min
        )
        out_full[0, first_gather] = first_out[0]  # [N_first,H,D]
        lse_full[0, first_gather] = first_lse[0]  # [N_first,H]
        out_full[0, second_gather] = second_out[0]  # [N_second,H,D]
        lse_full[0, second_gather] = second_lse[0]  # [N_second,H]
        ctx.save_for_backward(
            first_out,
            first_lse,
            second_out,
            second_lse,
            out_full,
            lse_full,
            first_gather,
            second_gather,
        )
        return out_full, lse_full

    @staticmethod
    def backward(
        ctx,
        grad_out_full: torch.Tensor,  # [1,N,H,D]
        grad_lse_full: torch.Tensor,  # [1,N,H]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None, None, None]:
        """Gather patched merge state and gradients back to each query population.

        This method runs after the outer ``merge_attentions`` backward has patched the saved
        packed ``out_full`` and ``lse_full`` with the final same-view + cross-view + caption
        output and normalization. It gathers RGB rows and control rows from those packed
        tensors and copies them into the corresponding inner-merge output/LSE storage. When
        the two inner merges subsequently run backward, they can therefore pass the globally
        merged state on to their four attention kernels.

        Independently, ``grad_out_full`` and ``grad_lse_full`` arrive in packed query order.
        Gathering them with the same indices produces the ordinary gradients returned for the
        first and second output/LSE inputs. The index tensors and integer output size are
        metadata and receive no gradients.
        """
        (
            first_out,
            first_lse,
            second_out,
            second_lse,
            out_full,
            lse_full,
            first_gather,
            second_gather,
        ) = ctx.saved_tensors
        patched_first_out = out_full[:, first_gather]  # [1,N_first,H,D]
        patched_first_lse = lse_full[:, first_gather]  # [1,N_first,H]
        patched_second_out = out_full[:, second_gather]  # [1,N_second,H,D]
        patched_second_lse = lse_full[:, second_gather]  # [1,N_second,H]
        first_out.data.copy_(patched_first_out.data)
        first_lse.data.copy_(patched_first_lse.data)
        second_out.data.copy_(patched_second_out.data)
        second_lse.data.copy_(patched_second_lse.data)
        grad_first_out = grad_out_full[:, first_gather]  # [1,N_first,H,D]
        grad_first_lse = grad_lse_full[:, first_gather]  # [1,N_first,H]
        grad_second_out = grad_out_full[:, second_gather]  # [1,N_second,H,D]
        grad_second_lse = grad_lse_full[:, second_gather]  # [1,N_second,H]
        return grad_first_out, grad_first_lse, grad_second_out, grad_second_lse, None, None, None
