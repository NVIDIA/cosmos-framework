# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Marking individual call sites for selective activation checkpointing.

``save_ops_regex`` selects by dispatched op name, which cannot separate calls
that run the same kernel. The decomposed multiview attention runs four FMHA
calls per layer -- three sensor folds plus the causal pass -- and they are worth
very different amounts to keep: the same-view fold is ~96% of forward attention
time and ~94% of backward, while the others are cheap to recompute. A regex
either keeps all four or none.

A call site marks itself instead::

    k = mark_next_activation(k)
    out, lse = attention(q, k, v, ...)

The policy sees ``cosmos3::keep_next_activation`` before the op it marks and keeps
that one. What counts as markable is ``save_ops_regex``: a mark stays pending
until an op matching it arrives, so the gathers, reshapes and the clone the
attention frontend makes in between cannot take the mark -- four such ops sit
between the marker and the kernel at the real call site.

Marking only decides anything when ``save_only_marked_ops`` is set. Without it the
regex keeps every op it matches, marked or not, which is what existing configs do
-- and then the marker is pure cost, so it is not emitted at all: ``enable_marking``
is called from ``_apply_selective_ac`` when that config field is on, and until it is,
``mark_next_activation`` hands its argument straight back. The switch is read while
Dynamo traces, so it is baked into the compiled graph rather than tested per step.

The switch only ever turns on, and production has no way to turn it off -- see
``enable_marking`` for why turning it off from a second model's config is a silent
whole-run regression rather than a saving.

The marker must be functional and its result must be consumed, which means it
copies what it marks. Mark the smallest tensor the call takes. On the AV shape
that is K or V: Cosmos3 16B runs 32 query heads against 8 KV heads, and under
CP16 that is 2 against 1 per rank, so the copy is a quarter to a half of the
output the fold would otherwise stash. The arithmetic still comes out ahead: on a
three-call stand-in, marking one rather than saving all three moved per-step peak
from 2574 MiB to 1928 MiB, clone included.

The copy is transient. The marker is ``MUST_RECOMPUTE``, so inside a checkpointed
region it is rebuilt during recompute rather than stored, and nothing carries
across steps -- measured at 0 MiB resident after each of four consecutive steps.
"""

import torch

_MARK_OP_NAME = "keep_next_activation"

MARK_OP_QUALNAME = f"cosmos3::{_MARK_OP_NAME}"


@torch.library.custom_op(MARK_OP_QUALNAME, mutates_args=())
def _keep_next_activation(tensor: torch.Tensor) -> torch.Tensor:
    # A copy, not a view: inductor rejects custom ops whose output aliases an
    # input, and a functional op's result has to be consumed for the marker to
    # keep its place in the trace.
    return tensor.clone()


@_keep_next_activation.register_fake
def _(tensor: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(tensor)


def _backward(ctx, grad: torch.Tensor) -> torch.Tensor:
    return grad


_keep_next_activation.register_autograd(_backward)


_MARKING_ENABLED = False


def enable_marking() -> None:
    """Turn call-site marking on for this process. There is deliberately no way back.

    Called from ``parallelize_unified_mot._apply_selective_ac`` when the model's
    ``activation_checkpointing.save_only_marked_ops`` is set -- at setup, before the
    first forward, so the answer is fixed by the time anything traces.

    Process-level because the call sites that mark are deep inside attention and have
    no view of the checkpointing config, and one-way because the two mistakes are not
    the same size. Dynamo installs an equality guard on the switch, so turning it off
    for a second model invalidates the first model's compiled code on its very next
    forward -- no shape change needed -- and the recompiled graph has no marker in it.
    A policy built with ``save_only_marked_ops`` then keeps *nothing*: the model silently
    reverts to recomputing all four folds, ~50 ms/step, with no error. Measured, not
    reasoned about: the same block goes from saving ``[False, True, False]`` to
    ``[False, False, False]`` across that transition.

    The cost of the opposite mistake is a clone a later model does not need. It is
    bounded and visible: only the maskless folds mark at all, the marker is
    ``MUST_RECOMPUTE`` so it is 0 bytes resident under selective or full AC, and it is
    a quarter to a half of one fold's output in transient bandwidth. So the switch
    latches, and a model that would rather not pay it scopes marking to itself.
    """
    global _MARKING_ENABLED
    _MARKING_ENABLED = True


def reset_marking_for_tests() -> None:
    """Turn marking back off. Tests only -- see ``enable_marking`` for why.

    Named for its one legitimate caller so that a config-driven ``enable_marking(False)``
    cannot be written by accident; that call is the regression described above.
    """
    global _MARKING_ENABLED
    _MARKING_ENABLED = False


def marking_enabled() -> bool:
    """Whether ``mark_next_activation`` currently emits anything."""
    return _MARKING_ENABLED


def mark_next_activation(tensor: torch.Tensor) -> torch.Tensor:
    """Ask selective AC to keep the output of the next op that reads ``tensor``.

    Pass the smallest tensor the marked call takes, and use its return value::

        k = mark_next_activation(k)
        out, lse = attention(q, k, v, ...)

    Binds to the next op matching ``save_ops_regex``, in trace order, rather than
    to an index: a call site behind a branch cannot shift the mark onto its
    neighbour, and the reshapes between the mark and the kernel cannot absorb it.

    A pass-through unless ``enable_marking`` has been called, so a model that does not
    use ``save_only_marked_ops`` pays nothing for the call sites that mark. When it is
    on the marker is a copy, which the checkpoint recomputes and discards.
    """
    if not _MARKING_ENABLED:
        return tensor
    return _keep_next_activation(tensor)


def is_mark_op(op_name: str) -> bool:
    """Whether a dispatched op name is the marker."""
    return _MARK_OP_NAME in op_name
