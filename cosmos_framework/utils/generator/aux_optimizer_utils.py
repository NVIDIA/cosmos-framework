# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Shared parameter-categorization for the orthogonalizing optimizers (Muon / Dion2).

Both ``MuonWithAuxAdamW`` and ``Dion2WithAuxAdamW`` only apply their
orthogonalized (Newton-Schulz) update to *hidden* ``nn.Linear`` weight matrices.
Everything else -- token / positional embeddings, the output head (``lm_head``),
layer-norm scales, biases, and any non-Linear parameter -- must use the auxiliary
AdamW.

The tricky part is reliably identifying the **embeddings** and the **output head**
across every model architecture in the repo (Qwen3, GPT-OSS, DeepSeekV3, Gemma4,
Qwen3-VL-MoE, unified MoT, ...). We do not rely on a single hard-coded name.
Instead :func:`split_orthogonalizable_params` combines four signals:

1. **Module type** -- ``nn.Embedding`` weights always go to AdamW (and they are
   never ``nn.Linear`` so they would never reach Muon anyway).
2. **Name keywords** -- a configurable set of substrings (default ``{"lm_head"}``,
   the convention used by every LLM in this repo) marks output-head Linears.
3. **Tied weights** -- a Linear whose ``weight`` tensor is *the same object* as an
   ``nn.Embedding`` weight (``tie_word_embeddings=True``) is the tied output head.
4. **Vocabulary shape** -- a Linear whose output dimension equals some
   ``nn.Embedding.num_embeddings`` (the vocab size) is treated as an output
   projection.

Signals (2)-(4) are OR-ed, so an output head is excluded from Muon/Dion2 even if a
new architecture names it differently or ties it to the embedding. The failure
mode of the heuristics is conservative: a misclassified weight falls back to
AdamW (always safe) rather than being orthogonalized.
"""

import contextlib
import math
import re
from collections.abc import Callable, Generator, MutableMapping
from dataclasses import dataclass, field
from typing import Any, cast

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor

from cosmos_framework.utils import log
from cosmos_framework.utils.misc import get_local_tensor_if_DTensor

# Default regex patterns (matched with ``re.search`` against the dotted parameter
# name) that force an ``nn.Linear`` weight onto the AdamW side -- i.e. output heads.
# Every LLM in the repo names its head ``lm_head``; the extra entries cover common
# alternative names used elsewhere. Users add more (e.g. MoE routers) via
# ``orthogonalize_skip_patterns``; the two are concatenated into a single check.
DEFAULT_ORTHOGONALIZE_SKIP_PATTERNS: tuple[str, ...] = (r"lm_head", r"embed_out", r"output_layer")


def split_orthogonalizable_params(
    model: nn.Module,
    optimizer_param_ids: set[int],
    expert_param_keywords: tuple[str, ...] = (),
    orthogonalize_skip_patterns: tuple[str, ...] = (),
) -> tuple[list[nn.Parameter], list[nn.Parameter], dict[nn.Parameter, str]]:
    """Split ``model``'s trainable params by whether they can be orthogonalized.

    Args:
        model: The (sub)module to walk -- typically the trainable ``net``.
        optimizer_param_ids: ``id()`` of every parameter actually owned by the
            optimizer's param groups. Only these are categorized, so that
            ``keys_to_select`` filtering is respected and ``state_dict`` stays
            consistent.
        expert_param_keywords: Name substrings that mark **stacked MoE expert**
            parameters -- raw ``nn.Parameter`` tensors of shape ``[num_experts, M,
            N]`` (e.g. ``gate_up_proj`` / ``down_proj``). When a 3-D+ param matches,
            it is routed to the orthogonalizable side so the optimizer can treat
            each expert slice as its own matrix. Empty (default) keeps expert
            params on AdamW, i.e. no behavior change.

            NOTE (sharding assumption): the optimizer orthogonalizes these per
            expert slice assuming the tensor is sharded on dim 0 (the expert axis),
            which holds for the FSDP2 ``fully_shard`` path used here (FSDP2 shards
            every parameter on dim 0) and makes the update communication-free. It is
            NOT guaranteed if tensor/expert parallelism shards *within* an expert
            matrix (dim 1/2); that case is unsupported and rejected at step time by
            ``MuonWithAuxAdamW._step_stacked_muon`` / ``Dion2WithAuxAdamW._step_stacked_dion2``.
            This was not exhaustively audited across every parallelization config,
            hence the runtime guard there.

            TODO(expert-parallelism): support 2-D sharding of stacked expert params.
            With expert parallelism, dim 0 would be sharded along the expert axis and
            dim 1 along the FSDP axis simultaneously, so each rank holds a shard of a
            *slice* of each expert matrix rather than whole expert matrices. The
            per-expert-slice orthogonalization here assumes dim-0-only sharding (whole
            expert matrices are local), so it would need to reconstruct each expert
            matrix across the FSDP axis (an all-gather along dim 1, like the dense
            2-D path) before running Newton-Schulz. Doable, but needs changes.
        orthogonalize_skip_patterns: Extra regex patterns matched (``re.search``)
            against each 2-D ``nn.Linear`` weight's parameter name. Concatenated with
            the built-in ``DEFAULT_ORTHOGONALIZE_SKIP_PATTERNS`` (output heads) into a
            single check: any match forces that weight onto the auxiliary AdamW side
            (skips orthogonalization). The embedding-by-type / tied-weight (``id()``
            dedup) / stacked-expert rules are separate and still apply. This is the
            general, name-based way to keep specific Linear weights off Muon/Dion2 --
            e.g.
            ``r"\\.gate\\.weight$"`` for MoE router/gate weights (matches
            ``...mlp.gate.weight`` / ``...mlp_moe_gen.gate.weight`` but not
            ``gate_proj`` / ``gate_noise``). Empty (default) skips nothing extra
            (no behavior change).

    Returns:
        ``(orthogonalizable, unorthogonalizable, param_to_name)`` where
        ``orthogonalizable`` holds the hidden ``nn.Linear`` weights (2-D) and, when
        ``expert_param_keywords`` is set, the stacked expert tensors (3-D); the
        optimizer splits those by rank. ``unorthogonalizable`` is everything else
        (embeddings, output head, biases, norms, other non-Linear params), which
        uses the auxiliary AdamW.
    """
    orthogonalizable: list[nn.Parameter] = []
    unorthogonalizable: list[nn.Parameter] = []
    param_to_name: dict[nn.Parameter, str] = {}
    categorized: set[int] = set()

    # Precompile the orthogonalization-skip regexes once (matched against param names):
    # the built-in defaults (output heads) plus any user-specified extra patterns.
    ortho_skip_res = [
        re.compile(pat) for pat in DEFAULT_ORTHOGONALIZE_SKIP_PATTERNS + tuple(orthogonalize_skip_patterns)
    ]

    for name, param in model.named_parameters():
        if param.requires_grad:
            param_to_name[param] = name

    def _eligible(p: nn.Parameter) -> bool:
        return p.requires_grad and id(p) in optimizer_param_ids and id(p) not in categorized

    def _is_stacked_expert(param: nn.Parameter) -> bool:
        if not expert_param_keywords or param.ndim < 3:
            return False
        name = param_to_name.get(param, "")
        return any(keyword in name for keyword in expert_param_keywords)

    for module_name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # A Linear weight goes to AdamW if its parameter name matches any skip
            # pattern (built-in output-head defaults + user-specified extras such as
            # r"\.gate\.weight$" for MoE routers); otherwise it is orthogonalized.
            if _eligible(module.weight):
                skip_ortho = any(r.search(param_to_name[module.weight]) for r in ortho_skip_res)
                if skip_ortho:
                    unorthogonalizable.append(module.weight)
                else:
                    orthogonalizable.append(module.weight)
                categorized.add(id(module.weight))
            if module.bias is not None and _eligible(module.bias):
                unorthogonalizable.append(module.bias)
                categorized.add(id(module.bias))
        else:
            # Embeddings (nn.Embedding), norms, conv, and any raw nn.Parameter ->
            # unorthogonalizable (AdamW), except stacked MoE expert tensors when
            # opted in. ``recurse=False`` so each owning module handles its own
            # params exactly once.
            for param in module.parameters(recurse=False):
                if not _eligible(param):
                    continue
                (orthogonalizable if _is_stacked_expert(param) else unorthogonalizable).append(param)
                categorized.add(id(param))

    return orthogonalizable, unorthogonalizable, param_to_name


# -----------------------------------------------------------------------------
# Shared orthogonalization / momentum math (used by both MuonWithAuxAdamW and
# Dion2WithAuxAdamW). Kept here so the two optimizers do not duplicate them.
# -----------------------------------------------------------------------------


@torch.compile(fullgraph=True)
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.

    Uses a quintic iteration whose coefficients are selected to maximize the slope at zero.
    This produces US'V^T where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which
    empirically does not hurt model performance relative to exact UV^T.

    Args:
        G: Input tensor of shape (m, n) where m, n >= 1.
        steps: Number of Newton-Schulz iterations.

    Returns:
        Orthogonalized tensor of same shape as G.
    """
    assert G.ndim == 2, "Input must be 2D"
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()

    # Transpose if tall matrix for numerical stability
    if G.size(0) > G.size(1):
        X = X.T

    # Ensure spectral norm is at most 1 (global norm, matching Moonlight)
    X = X / (X.norm() + 1e-7)

    # Perform Newton-Schulz iterations
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X

    # Transpose back if we transposed earlier
    if G.size(0) > G.size(1):
        X = X.T

    return X


@torch.compile(fullgraph=True)
def zeropower_via_newtonschulz5_batched(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Batched Newton-Schulz over a stack of matrices.

    Same quintic iteration as :func:`zeropower_via_newtonschulz5`, but applied to a
    batch ``G`` of shape ``[..., M, N]`` (e.g. stacked MoE experts ``[E, M, N]``)
    using batched matmuls. All matrices in the batch share ``M, N``, so the
    transpose-if-tall decision is uniform across the batch.
    """
    assert G.ndim >= 3, "Batched input must be at least 3D ([..., M, N])"
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()

    transposed = False
    if X.size(-2) > X.size(-1):
        X = X.mT
        transposed = True

    # Per-matrix spectral-norm normalization (norm over the last two dims).
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if transposed:
        X = X.mT

    return X


def _compute_pre_ns_update_impl(
    grad: torch.Tensor,
    momentum_buffer: torch.Tensor,
    momentum: float,
    nesterov: bool,
    output_dtype: torch.dtype | None,
) -> torch.Tensor:  # grad/momentum_buffer: [M,N], returns [M,N]
    """Tensor implementation shared by the single- and multi-parameter compiled entry points."""
    # SGD-style momentum: buf = momentum * buf + grad (matching Moonlight)
    momentum_buffer.mul_(momentum).add_(grad)  # [M,N]

    # Nesterov: g = g + momentum * buf, else just use buf
    if nesterov:
        pre_ns = grad.add(momentum_buffer, alpha=momentum)  # [M,N]
    else:
        pre_ns = momentum_buffer.clone()  # [M,N]
    if output_dtype is not None:
        pre_ns = pre_ns.to(output_dtype)  # [M,N]
    return pre_ns


@torch.compile(fullgraph=True)
def compute_pre_ns_update(
    grad: torch.Tensor,
    momentum_buffer: torch.Tensor,
    momentum: float = 0.95,
    nesterov: bool = True,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:  # grad/momentum_buffer: [M,N], returns [M,N]
    """
    Compute the pre-Newton-Schulz update (momentum + optional Nesterov).

    This is separated from NS so that momentum/Nesterov can be applied on shards
    before all-gathering for distributed NS.

    Args:
        grad: Gradient tensor.
        momentum_buffer: Momentum buffer (modified in-place).
        momentum: Momentum coefficient.
        nesterov: Whether to use Nesterov momentum.
        output_dtype: Optional dtype conversion fused into the compiled update.

    Returns:
        Pre-NS update tensor (same shape as grad).
    """
    return _compute_pre_ns_update_impl(grad, momentum_buffer, momentum, nesterov, output_dtype)  # [M,N]


_PRE_NS_COMPILE_OPTIONS: dict[str, bool | int] = {
    name: value
    for name, value in {
        "combo_kernels": True,
        "benchmark_combo_kernel": False,
        "combo_kernel_max_num_nodes": 64,
        "combo_kernel_max_num_args": 250,
        "aggressive_fusion": True,
    }.items()
    if hasattr(torch._inductor.config, name)
}


@torch.compile(fullgraph=True, options=_PRE_NS_COMPILE_OPTIONS)
def compute_pre_ns_updates_and_pack(
    grads: list[torch.Tensor],
    momentum_buffers: list[torch.Tensor],
    active_indices: list[int],
    local_param_shard: torch.Tensor,
    actual_batch_size: int,
    batch_capacity: int,
    momentum: float = 0.95,
    nesterov: bool = True,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:  # grads/momentum_buffers/local_param_shard: [M,N] each, returns [S,M,N]
    """Compute active pre-NS updates and pack active, inactive, and padding slots in one graph."""
    pre_ns_updates = [
        _compute_pre_ns_update_impl(grad, momentum_buffer, momentum, nesterov, output_dtype)
        for grad, momentum_buffer in zip(grads, momentum_buffers)
    ]  # [M,N] each
    pre_ns_by_index = dict(zip(active_indices, pre_ns_updates, strict=True))
    zero = local_param_shard.new_zeros(local_param_shard.shape, dtype=output_dtype)  # [M,N]
    slots = [
        pre_ns_by_index[index] if index < actual_batch_size and index in pre_ns_by_index else zero
        for index in range(batch_capacity)
    ]  # [M,N] each
    return torch.stack(slots, dim=0)  # [S,M,N]


def _compute_pre_ns_update_moe_expert_impl(
    grad: torch.Tensor,
    momentum_buffer: torch.Tensor,
    momentum: float,
    nesterov: bool,
) -> tuple[torch.Tensor, torch.Tensor]:  # grad/momentum_buffer: [E,M,N], returns ([E,M,N], [E])
    """Tensor implementation shared by the single- and multi-matrix compiled entry points.

    Kept undecorated so it inlines into whichever compiled graph calls it (mirrors
    ``_compute_pre_ns_update_impl`` on the dense path); nesting ``torch.compile``
    inside ``torch.compile`` would otherwise force a graph break per matrix.
    """
    active = (grad != 0).flatten(1).any(dim=1)  # [E] bool
    a = active.view(-1, 1, 1).to(momentum_buffer.dtype)  # [E,1,1] in {0, 1}
    momentum_buffer.mul_(1 - a * (1 - momentum)).add_(grad)  # [E,M,N]
    if nesterov:
        pre_ns = grad.add(momentum_buffer, alpha=momentum)  # [E,M,N]
    else:
        pre_ns = momentum_buffer.clone()  # [E,M,N]
    return pre_ns, active


@torch.compile(fullgraph=True, options=_PRE_NS_COMPILE_OPTIONS)
def compute_pre_ns_update_moe_expert(
    grad: torch.Tensor,
    momentum_buffer: torch.Tensor,
    momentum: float = 0.95,
    nesterov: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:  # grad/momentum_buffer: [E,M,N], returns ([E,M,N], [E])
    """
    Per-expert masked momentum for stacked MoE experts.

    Identical to :func:`compute_pre_ns_update` for *active* experts (those whose
    gradient slice has any nonzero element, i.e. tokens were routed to them this
    step), but FREEZES the momentum of *inactive* experts -- no momentum decay
    and no accumulation -- instead of decaying it.

    This distinction matters for Muon/Dion but not for Adam: Newton-Schulz
    renormalizes whatever it is handed to unit spectral norm, so an inactive
    expert's (decayed but nonzero) stale momentum would be blown back up into a
    full-strength update in a stale direction. Freezing keeps the momentum in
    reserve until the expert is routed again, matching the reference
    (microsoft/dion), which sits out params/experts that received no gradient.

    This function only handles the momentum recurrence. The caller MUST also, for
    inactive experts: (1) zero the Newton-Schulz output, and (2) skip the weight
    update and weight decay. The returned ``active`` mask is provided for exactly
    that.

    Args:
        grad: Local expert gradients, shape ``[E, M, N]`` (E = local experts).
        momentum_buffer: Momentum buffer ``[E, M, N]``, modified in place.
        momentum: Momentum coefficient.
        nesterov: Whether to use Nesterov momentum.

    Returns:
        pre_ns: Pre-Newton-Schulz update, shape ``[E, M, N]``.
        active: Per-expert bool mask ``[E]``; True where the expert got a gradient.
    """
    # Per-expert active mask: True iff the expert's gradient slice has any nonzero
    # element. An expert with no tokens routed produces an exactly-zero gradient
    # slice, so this is an exact "was this expert updated" test. Masked SGD-style
    # momentum keeps inactive experts frozen; see the impl for the arithmetic.
    return _compute_pre_ns_update_moe_expert_impl(grad, momentum_buffer, momentum, nesterov)


@torch.compile(fullgraph=True, options=_PRE_NS_COMPILE_OPTIONS)
def compute_pre_ns_updates_moe_expert(
    grads: list[torch.Tensor],
    momentum_buffers: list[torch.Tensor],
    momentum: float = 0.95,
    nesterov: bool = True,
) -> tuple[
    list[torch.Tensor], list[torch.Tensor]
]:  # grads/momentum_buffers: [E,M,N] each, returns ([E,M,N] each, [E] each)
    """Batched :func:`compute_pre_ns_update_moe_expert` over many expert matrices.

    Mirrors the dense path's :func:`compute_pre_ns_updates_and_pack`: a plain Python loop over
    the per-matrix impl, compiled as a single graph so inductor's ``combo_kernels``
    fuse the independent per-matrix work (including the active-mask reductions) into
    a handful of kernels. Deliberately NOT written with ``torch._foreach_*``: foreach
    cannot fuse the mask/multiplier construction, so that work would stay eager and
    cost more launches than the batching saves.

    ``momentum_buffers`` are mutated in place; torch.compile preserves in-place
    mutation of its inputs, so optimizer state persists across steps (covered by
    ``test_batched_pre_ns_momentum_persists_across_steps``).

    Every tensor in ``grads`` (and likewise in ``momentum_buffers``) MUST share one
    shape and one stride. A call mixing layouts -- e.g. gate/up ``[E, H, I]`` slices
    together with ``down`` ``[E, I, H]`` -- compiles an artifact whose argument
    binding depends on how the inputs happen to alias each other. FSDP re-buckets
    expert gradients between the first and second optimizer step after a checkpoint
    resume (per-parameter buffers become views of one flat per-layer buffer), and
    dynamo does not recompile on that change, so arguments get bound to the wrong
    slots. Call this once per layout instead; see
    ``test_megabatch_survives_grad_alias_topology_change``.

    Note: ``fullgraph=True`` specializes on the list length, so a differently sized
    tail batch costs one extra compile (48 layers at K=8 divides evenly, so none).

    Args:
        grads: Local expert gradients, each ``[E, M, N]``.
        momentum_buffers: Matching momentum buffers, modified in place.
        momentum: Momentum coefficient.
        nesterov: Whether to use Nesterov momentum.

    Returns:
        pre_ns_list: Pre-Newton-Schulz updates, one per input.
        active_list: Per-expert bool masks ``[E]``, one per input.
    """
    results = [
        _compute_pre_ns_update_moe_expert_impl(grad, buf, momentum, nesterov)
        for grad, buf in zip(grads, momentum_buffers)
    ]
    return [r[0] for r in results], [r[1] for r in results]


def _apply_stacked_expert_update_impl(
    target: torch.Tensor,
    ortho: torch.Tensor,
    active: torch.Tensor,
    wd_factor: torch.Tensor,
    neg_adjusted_lr: torch.Tensor,
) -> None:  # target/ortho: [E,M,N], active: [E], wd_factor/neg_adjusted_lr: [] scalars
    """Masked weight decay + scaled update for one matrix, in place on ``target``.

    Zeroes inactive experts in ``ortho`` here rather than in the caller: the mask is
    exactly 0 or 1, so applying it before or after the dtype cast is bit-identical,
    and doing it inside the graph saves a per-matrix eager multiply and cast.

    Undecorated so it inlines into the compiled batched entry point below.
    """
    a = active.view(-1, 1, 1).to(target.dtype)  # [E,1,1] in {0, 1}
    target.mul_(1 - a * wd_factor)  # [E,M,N]
    target.add_(ortho.to(target.dtype) * a * neg_adjusted_lr)  # [E,M,N]


@torch.compile(fullgraph=True, options=_PRE_NS_COMPILE_OPTIONS)
def _apply_stacked_expert_updates_compiled(
    targets: list[torch.Tensor],
    orthos: list[torch.Tensor],
    actives: list[torch.Tensor],
    wd_factors: torch.Tensor,
    neg_adjusted_lrs: torch.Tensor,
) -> None:  # targets/orthos: [E,M,N] each, actives: [E] each, wd_factors/neg_adjusted_lrs: [N_mat]
    """Apply a whole expert-matrix batch in ONE graph (see combo_kernels note above)."""
    for i, (target, ortho, active) in enumerate(zip(targets, orthos, actives)):
        _apply_stacked_expert_update_impl(target, ortho, active, wd_factors[i], neg_adjusted_lrs[i])


def apply_stacked_expert_updates_batched(
    local_params: list[torch.Tensor],
    orthos: list[torch.Tensor],
    actives: list[torch.Tensor],
    base_lrs: list[float | torch.Tensor],
    weight_decays: list[float],
    adjusted_lrs: list[float | torch.Tensor],
    masters: list[torch.Tensor | None],
) -> None:
    """Batched form of :func:`_apply_stacked_expert_matrix_update`.

    Same per-matrix semantics (per-expert masked weight decay, then the scaled
    orthogonalized update, inactive experts untouched), run as a single compiled
    graph so combo_kernels fuse across matrices.

    The learning-rate-derived scalars are passed as **tensors**, not Python floats:
    the LR changes every scheduler step, and a float would be baked into the graph
    as a constant and force a recompile on every change.

    ``masters`` is either all-None (no FP32 master weights) or all-set -- an
    optimizer-wide setting -- so the two cases are handled as whole lists.

    As with :func:`compute_pre_ns_updates_moe_expert`, every tensor within
    ``local_params`` (and within ``orthos``) must share one shape and stride, so
    callers batch one role at a time.
    """
    if not local_params:
        return

    use_master = masters[0] is not None
    if use_master and any(m is None for m in masters):
        raise ValueError("master weights must be all-set or all-None across a megabatch")
    targets: list[torch.Tensor] = cast(list[torch.Tensor], masters) if use_master else local_params
    device = targets[0].device

    def _as_tensor(values: list) -> torch.Tensor:  # returns [N_mat]
        if any(isinstance(v, torch.Tensor) for v in values):
            return torch.stack([torch.as_tensor(v, device=device, dtype=torch.float32).reshape(()) for v in values])
        return torch.tensor(values, device=device, dtype=torch.float32)

    wd_factors = _as_tensor([blr * wd for blr, wd in zip(base_lrs, weight_decays, strict=True)])  # [N_mat]
    neg_adjusted_lrs = -_as_tensor(list(adjusted_lrs))  # [N_mat]

    _apply_stacked_expert_updates_compiled(targets, orthos, actives, wd_factors, neg_adjusted_lrs)

    if use_master:
        # Pure copy with nothing to fuse, so foreach is the right tool here (unlike
        # the masked math above, which needs a compiled graph).
        torch._foreach_copy_(local_params, targets)


@dataclass
class _ExpertApplyGroup:
    """Matrices sharing one tensor layout, accumulated for a single batched apply.

    Exists so the megabatch apply issues one
    :func:`apply_stacked_expert_updates_batched` call per layout, which is what
    guarantees a single layout per compiled graph.
    """

    params: list[torch.Tensor] = field(default_factory=list)  # [E,M,N] each
    orthos: list[torch.Tensor] = field(default_factory=list)  # [E,M,N] each
    actives: list[torch.Tensor] = field(default_factory=list)  # [E] each
    base_lrs: list[float | torch.Tensor] = field(default_factory=list)
    weight_decays: list[float] = field(default_factory=list)
    adjusted_lrs: list[float | torch.Tensor] = field(default_factory=list)
    masters: list[torch.Tensor | None] = field(default_factory=list)

    def add(
        self,
        param: torch.Tensor,  # [E,M,N]
        ortho: torch.Tensor,  # [E,M,N]
        active: torch.Tensor,  # [E]
        base_lr: float | torch.Tensor,
        weight_decay: float,
        adjusted_lr: float | torch.Tensor,
        master: torch.Tensor | None,  # [E,M,N] or None
    ) -> None:
        self.params.append(param)
        self.orthos.append(ortho)
        self.actives.append(active)
        self.base_lrs.append(base_lr)
        self.weight_decays.append(weight_decay)
        self.adjusted_lrs.append(adjusted_lr)
        self.masters.append(master)

    def apply(self) -> None:
        apply_stacked_expert_updates_batched(
            self.params,
            self.orthos,
            self.actives,
            self.base_lrs,
            self.weight_decays,
            self.adjusted_lrs,
            self.masters,
        )


# -----------------------------------------------------------------------------
# MoE expert gate_up / down pairing and split-NS helpers
# -----------------------------------------------------------------------------


def pair_moe_gate_up_down_params(
    stacked_params: list[nn.Parameter],
    param_to_name: dict[nn.Parameter, str],
) -> list[tuple[nn.Parameter, nn.Parameter]]:
    """Pair gate_up_proj [E,H,2I] params with their matching down_proj [E,I,H] params.

    Matches by replacing the ``gate_up_proj`` suffix in the parameter name with
    ``down_proj`` and looking up the result in the same stacked-param list. Validates
    shapes to confirm the pair is consistent (H and I must align across the two).

    Args:
        stacked_params: List of stacked MoE expert parameters (ndim >= 3).
        param_to_name: Mapping from parameter to its dotted name (e.g. from
            ``split_orthogonalizable_params``).

    Returns:
        List of ``(gate_up_param, down_param)`` tuples in the order they appear in
        ``stacked_params``.

    Raises:
        ValueError: If any ``gate_up_proj`` parameter has no matching ``down_proj``
            in the list, or if shapes are inconsistent.
    """
    stacked_ids = {id(p) for p in stacked_params}
    name_to_param: dict[str, nn.Parameter] = {v: k for k, v in param_to_name.items() if id(k) in stacked_ids}

    pairs: list[tuple[nn.Parameter, nn.Parameter]] = []
    unmatched: list[str] = []

    for p in stacked_params:
        name = param_to_name.get(p, "")
        if "gate_up_proj" not in name:
            continue
        down_name = name.replace("gate_up_proj", "down_proj")
        down_param = name_to_param.get(down_name)
        if down_param is None:
            unmatched.append(name)
            continue

        # Validate shapes: gate_up [E, H, 2I], down [E, I, H]
        gu_shape = tuple(p.shape)
        d_shape = tuple(down_param.shape)
        if (
            len(gu_shape) != 3
            or len(d_shape) != 3
            or gu_shape[0] != d_shape[0]  # E == E
            or gu_shape[1] != d_shape[2]  # H == H
            or gu_shape[2] != 2 * d_shape[1]  # 2I == 2I
        ):
            raise ValueError(
                f"Shape mismatch for MoE gate_up/down pair '{name}' / '{down_name}': "
                f"gate_up {gu_shape}, down {d_shape}. "
                f"Expected gate_up [E, H, 2I] and down [E, I, H] with matching E, H, I."
            )
        pairs.append((p, down_param))

    if unmatched:
        raise ValueError(
            f"split_expert_gate_up=True but gate_up_proj params have no matching "
            f"down_proj counterpart in stacked_params: {unmatched}"
        )

    return pairs


def validate_split_expert_ns_config(
    split_expert_gate_up: bool,
    batch_split_expert_ns: bool,
    *,
    fraction: float | None = None,
) -> None:
    """Validate split-expert Newton-Schulz configuration options.

    Args:
        split_expert_gate_up: Whether to split gate_up into gate+up before NS.
        batch_split_expert_ns: Whether to batch gate+up+down into one NS call.
        fraction: DION2 submatrix fraction; must be 1.0 when splitting experts.

    Raises:
        ValueError: If the combination of options is invalid.
    """
    if batch_split_expert_ns and not split_expert_gate_up:
        raise ValueError(
            "batch_split_expert_ns=True requires split_expert_gate_up=True "
            "(cannot batch NS for unsplit gate_up matrices)."
        )
    if fraction is not None and fraction != 1.0 and split_expert_gate_up:
        raise ValueError(
            f"split_expert_gate_up=True requires fraction=1.0 (full-matrix NS), "
            f"got fraction={fraction}. Submatrix selection is unsupported for split experts."
        )


def create_moe_megabatches(
    split_expert_pairs: list[tuple[nn.Parameter, nn.Parameter]],
    world_size: int,
    max_moe_expert_ns_matrices: int,
) -> list[list[tuple[nn.Parameter, nn.Parameter]]]:
    """Group split expert pairs into K-layer Newton-Schulz batches.

    Pairs with the same ``(gate_up_shape, down_shape)`` are grouped into batches of
    K consecutive pairs. K is derived from the matrix budget:

        K = max(1, max_moe_expert_ns_matrices // (3 * E_local))

    where ``E_local = ceil(gate_up.shape[0] / world_size)``. Expressing the knob as a
    matrix budget rather than a layer count keeps the NS working set stable when the
    expert shard count changes, so one config does not silently blow up memory on a
    different mesh.

    Called once at ``categorize_params`` time; the plan is frozen for the run.

    Args:
        split_expert_pairs: ``(gate_up, down)`` parameter pairs to group.
        world_size: Expert shard count used to derive ``E_local``.
        max_moe_expert_ns_matrices: Matrix budget per NS call; ``<= 0`` means K=1
            (one pair per NS call, equivalent to single-layer split batching but
            routed through the megabatch dispatch).

    Returns:
        Batches of pairs, each batch being one NS round.
    """
    if not split_expert_pairs:
        return []

    # Group pairs by shape (identical on all ranks; use global shape).
    shape_groups: dict[tuple, list[tuple[nn.Parameter, nn.Parameter]]] = {}
    for gate_up, down in split_expert_pairs:
        key = (tuple(gate_up.shape), tuple(down.shape))
        shape_groups.setdefault(key, []).append((gate_up, down))

    megabatches: list[list[tuple[nn.Parameter, nn.Parameter]]] = []
    for (gu_shape, _down_shape), pairs in shape_groups.items():
        E_total = gu_shape[0]
        E_local = max(1, math.ceil(E_total / max(1, world_size)))
        if max_moe_expert_ns_matrices <= 0:
            K = 1
        else:
            K = max(1, max_moe_expert_ns_matrices // (3 * E_local))
        for i in range(0, len(pairs), K):
            megabatches.append(pairs[i : i + K])
        log.info(
            f"MoE megabatch: shape {gu_shape}, {len(pairs)} pairs, "
            f"E_local={E_local}, K={K}, {math.ceil(len(pairs) / K)} NS rounds"
        )

    log.info(
        f"MoE expert megabatch plan: {len(split_expert_pairs)} pairs -> "
        f"{len(megabatches)} NS rounds "
        f"(max_moe_expert_ns_matrices={max_moe_expert_ns_matrices})"
    )
    return megabatches


@torch.compile(fullgraph=True)
def zeropower_via_newtonschulz5_batched_groups(
    matrices: tuple[torch.Tensor, ...],
    steps: int,
) -> tuple[torch.Tensor, ...]:  # matrices: [E,M,N] each, returns [E,M,N] each
    """Cat equal-shaped [E, M, N] stacks along dim 0, run one batched NS, split back.

    All tensors in ``matrices`` must share the same ``[E, M, N]`` shape. They are
    concatenated into ``[len(matrices)*E, M, N]``, orthogonalized in a single
    :func:`zeropower_via_newtonschulz5_batched` call, then split back so the i-th
    output tensor corresponds to the i-th input tensor.

    Args:
        matrices: Tuple of same-shape ``[E, M, N]`` tensors.
        steps: Number of Newton-Schulz iterations.

    Returns:
        Tuple of orthogonalized tensors, one per input, each ``[E, M, N]``.
    """
    E = matrices[0].shape[0]
    packed = torch.cat(matrices, dim=0)  # [len(matrices)*E, M, N]
    packed_ortho = zeropower_via_newtonschulz5_batched(packed, steps=steps)  # [len(matrices)*E, M, N]
    return tuple(packed_ortho.split(E, dim=0))  # [E,M,N] each


# -----------------------------------------------------------------------------
# Dataclasses and per-parameter helpers shared across _step_stacked_* helpers
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class _StackedExpertStepContext:
    """Immutable bundle of optimizer context passed through stacked-expert step helpers."""

    optimizer_state: MutableMapping  # self.state (param -> state dict)
    param_to_name: dict[Any, str]  # param -> dotted name
    param_to_master: dict[int, torch.Tensor]  # id(param) -> FP32 master tensor
    master_weights: bool
    momentum: float
    nesterov: bool
    ns_steps: int
    batch_split_expert_ns: bool
    profile_phases: bool  # emit torch.profiler + NVTX ranges for moe.megabatch.{pre_ns,ns,apply}
    base_lr_for: Callable[[nn.Parameter], "float | torch.Tensor"]
    weight_decay_for: Callable[[nn.Parameter], float]
    adjusted_lr_for: Callable[["tuple[int, ...]", "float | torch.Tensor"], "float | torch.Tensor"]


@dataclass(frozen=True)
class _PreparedStackedExpertParam:
    """Validated local tensors for one stacked-expert parameter, ready for NS."""

    local_param: torch.Tensor  # local model weight shard (not DTensor)
    local_grad: torch.Tensor  # local gradient shard
    momentum: torch.Tensor  # momentum buffer, FP32 (local)
    master: torch.Tensor | None  # local FP32 master weight, or None


def _prepare_stacked_expert_param(
    p: nn.Parameter,
    context: _StackedExpertStepContext,
) -> _PreparedStackedExpertParam:
    """Validate DTensor placement, initialize optimizer state, and return local tensors.

    Args:
        p: Stacked MoE expert parameter (may be a DTensor).
        context: Optimizer context (state dict, flags, etc.).

    Returns:
        :class:`_PreparedStackedExpertParam` with validated local tensor views.

    Raises:
        NotImplementedError: If the DTensor is sharded on a non-expert dim (dim != 0).
        NotImplementedError: If the local tensor is not 3-D.
    """
    if isinstance(p, DTensor):
        for placement in p.placements:
            if placement.is_shard() and placement.dim != 0:
                raise NotImplementedError(
                    "Stacked-expert orthogonalization requires sharding on the expert "
                    f"dim (0); got placement {placement} for "
                    f"'{context.param_to_name.get(p, 'unknown')}'."
                )

    local_grad = get_local_tensor_if_DTensor(p.grad)
    local_param = get_local_tensor_if_DTensor(p)

    if local_grad.ndim != 3:
        raise NotImplementedError(
            f"Stacked-expert orthogonalization supports 3D params [E, M, N]; "
            f"got shape {tuple(local_grad.shape)} for '{context.param_to_name.get(p, 'unknown')}'."
        )

    # setdefault (not ``[p]``) so any MutableMapping works, not just the
    # defaultdict(dict) that torch.optim.Optimizer.state happens to be.
    state = context.optimizer_state.setdefault(p, {})
    if len(state) == 0:
        state["momentum_buffer"] = torch.zeros_like(p).float()

    master = None
    if context.master_weights:
        master = get_local_tensor_if_DTensor(context.param_to_master[id(p)])

    return _PreparedStackedExpertParam(
        local_param=local_param,
        local_grad=local_grad,
        momentum=get_local_tensor_if_DTensor(state["momentum_buffer"]),
        master=master,
    )


def _apply_stacked_expert_matrix_update(
    local_param: torch.Tensor,
    ortho: torch.Tensor,
    active: torch.Tensor,
    matrix_shape: tuple[int, int],
    p: nn.Parameter,
    context: _StackedExpertStepContext,
    master: torch.Tensor | None = None,
) -> None:
    """Apply per-expert masked weight decay and orthogonalized update.

    Inactive experts (``active[e] == False``) receive neither weight decay nor an
    update -- their weight tensor is left completely untouched.

    Args:
        local_param: Local model weight view to update in-place, shape ``[E, M, N]``.
        ortho: Orthogonalized update, same shape as ``local_param``. Must already be
            zeroed for inactive experts (``ortho * active.view(-1,1,1)``).
        active: Per-expert bool mask ``[E]``.
        matrix_shape: ``(M, N)`` tuple used for LR scaling (the per-expert matrix dims).
        p: The original ``nn.Parameter`` (used only for per-group lr / wd lookup).
        context: Optimizer context.
        master: FP32 master weight for ``local_param``, or ``None``.
    """
    base_lr = context.base_lr_for(p)
    wd = context.weight_decay_for(p)
    adjusted_lr = context.adjusted_lr_for(matrix_shape, base_lr)

    if master is not None:
        a_wd = active.view(-1, 1, 1).to(master.dtype)
        master.mul_(1 - a_wd * (base_lr * wd))
        master.add_(ortho.float() * (-adjusted_lr))
        local_param.copy_(master)
    else:
        a_wd = active.view(-1, 1, 1).to(local_param.dtype)
        local_param.mul_(1 - a_wd * (base_lr * wd))
        local_param.add_(ortho.to(local_param.dtype) * (-adjusted_lr))


def _step_one_stacked_expert_param(
    p: nn.Parameter,
    context: _StackedExpertStepContext,
) -> None:
    """Historical whole-param path: pre_NS -> batched NS over all local experts -> apply.

    Used for stacked expert params that are NOT part of a gate_up / down pair (i.e.
    ``split_expert_gate_up`` is False, or the param has no matching counterpart).
    """
    prep = _prepare_stacked_expert_param(p, context)
    pre_ns, active = compute_pre_ns_update_moe_expert(
        prep.local_grad,
        prep.momentum,
        momentum=context.momentum,
        nesterov=context.nesterov,
    )
    ortho = zeropower_via_newtonschulz5_batched(pre_ns, steps=context.ns_steps)
    ortho = ortho * active.view(-1, 1, 1).to(ortho.dtype)
    _apply_stacked_expert_matrix_update(
        prep.local_param,
        ortho,
        active,
        tuple(p.shape[-2:]),
        p,
        context,
        prep.master,
    )


def _step_split_stacked_expert_pair(
    gate_up_param: nn.Parameter,
    down_param: nn.Parameter,
    context: _StackedExpertStepContext,
) -> None:
    """Split gate_up into gate+up, run pre_NS on each plus down, optionally batch NS.

    When ``context.batch_split_expert_ns`` is True, the three [E, H, I] pre-NS
    matrices (gate, up, transposed-down) are orthogonalized in a single
    :func:`zeropower_via_newtonschulz5_batched_groups` call. When False, three
    separate NS calls are made.

    Args:
        gate_up_param: Combined gate+up projection, shape ``[E, H, 2I]``.
        down_param: Down projection, shape ``[E, I, H]``.
        context: Optimizer context.
    """
    if gate_up_param.grad is None and down_param.grad is None:
        return

    gate_up_prep: _PreparedStackedExpertParam | None = None
    gate_pre: torch.Tensor | None = None
    up_pre: torch.Tensor | None = None
    gate_active: torch.Tensor | None = None
    up_active: torch.Tensor | None = None

    if gate_up_param.grad is not None:
        gate_up_prep = _prepare_stacked_expert_param(gate_up_param, context)
        gate_grad, up_grad = gate_up_prep.local_grad.chunk(2, dim=-1)
        gate_mom, up_mom = gate_up_prep.momentum.chunk(2, dim=-1)
        gate_pre, gate_active = compute_pre_ns_update_moe_expert(
            gate_grad, gate_mom, momentum=context.momentum, nesterov=context.nesterov
        )
        up_pre, up_active = compute_pre_ns_update_moe_expert(
            up_grad, up_mom, momentum=context.momentum, nesterov=context.nesterov
        )

    down_prep: _PreparedStackedExpertParam | None = None
    down_pre: torch.Tensor | None = None
    down_active: torch.Tensor | None = None

    if down_param.grad is not None:
        down_prep = _prepare_stacked_expert_param(down_param, context)
        down_pre, down_active = compute_pre_ns_update_moe_expert(
            down_prep.local_grad,
            down_prep.momentum,
            momentum=context.momentum,
            nesterov=context.nesterov,
        )

    if context.batch_split_expert_ns:
        # Pack gate, up, transposed-down into one batched NS call.
        matrices: list[torch.Tensor] = []
        if gate_up_prep is not None:
            matrices.extend([gate_pre, up_pre])  # type: ignore[list-item]
        if down_prep is not None:
            matrices.append(down_pre.mT)  # type: ignore[union-attr]  # [E,H,I]

        if matrices:
            orthos = zeropower_via_newtonschulz5_batched_groups(tuple(matrices), steps=context.ns_steps)
            o_idx = 0
            if gate_up_prep is not None:
                gate_ortho = orthos[o_idx] * gate_active.view(-1, 1, 1).to(orthos[o_idx].dtype)  # type: ignore[union-attr]
                up_ortho = orthos[o_idx + 1] * up_active.view(-1, 1, 1).to(orthos[o_idx + 1].dtype)  # type: ignore[union-attr]
                o_idx += 2
                gate_width = gate_up_prep.local_param.shape[-1] // 2
                matrix_shape = (gate_up_prep.local_param.shape[-2], gate_width)
                gate_master = gate_up_prep.master[..., :gate_width] if gate_up_prep.master is not None else None
                up_master = gate_up_prep.master[..., gate_width:] if gate_up_prep.master is not None else None
                _apply_stacked_expert_matrix_update(
                    gate_up_prep.local_param[..., :gate_width],
                    gate_ortho,
                    gate_active,
                    matrix_shape,
                    gate_up_param,
                    context,
                    gate_master,  # type: ignore[arg-type]
                )
                _apply_stacked_expert_matrix_update(
                    gate_up_prep.local_param[..., gate_width:],
                    up_ortho,
                    up_active,
                    matrix_shape,
                    gate_up_param,
                    context,
                    up_master,  # type: ignore[arg-type]
                )
            if down_prep is not None:
                d_ortho_t = orthos[o_idx]  # still transposed: [E, H, I]
                d_ortho = d_ortho_t.mT  # back to [E, I, H]
                d_ortho = d_ortho * down_active.view(-1, 1, 1).to(d_ortho.dtype)  # type: ignore[union-attr]
                _apply_stacked_expert_matrix_update(
                    down_prep.local_param,
                    d_ortho,
                    down_active,
                    tuple(down_param.shape[-2:]),
                    down_param,
                    context,
                    down_prep.master,  # type: ignore[arg-type]
                )
    else:
        # Sequential: separate NS calls for gate, up, and down.
        if gate_up_prep is not None:
            gate_width = gate_up_prep.local_param.shape[-1] // 2
            matrix_shape = (gate_up_prep.local_param.shape[-2], gate_width)
            gate_master = gate_up_prep.master[..., :gate_width] if gate_up_prep.master is not None else None
            up_master = gate_up_prep.master[..., gate_width:] if gate_up_prep.master is not None else None

            gate_ortho = zeropower_via_newtonschulz5_batched(gate_pre, steps=context.ns_steps)  # type: ignore[arg-type]
            gate_ortho = gate_ortho * gate_active.view(-1, 1, 1).to(gate_ortho.dtype)  # type: ignore[union-attr]
            up_ortho = zeropower_via_newtonschulz5_batched(up_pre, steps=context.ns_steps)  # type: ignore[arg-type]
            up_ortho = up_ortho * up_active.view(-1, 1, 1).to(up_ortho.dtype)  # type: ignore[union-attr]

            _apply_stacked_expert_matrix_update(
                gate_up_prep.local_param[..., :gate_width],
                gate_ortho,
                gate_active,
                matrix_shape,
                gate_up_param,
                context,
                gate_master,  # type: ignore[arg-type]
            )
            _apply_stacked_expert_matrix_update(
                gate_up_prep.local_param[..., gate_width:],
                up_ortho,
                up_active,
                matrix_shape,
                gate_up_param,
                context,
                up_master,  # type: ignore[arg-type]
            )

        if down_prep is not None:
            d_ortho = zeropower_via_newtonschulz5_batched(down_pre, steps=context.ns_steps)  # type: ignore[arg-type]
            d_ortho = d_ortho * down_active.view(-1, 1, 1).to(d_ortho.dtype)  # type: ignore[union-attr]
            _apply_stacked_expert_matrix_update(
                down_prep.local_param,
                d_ortho,
                down_active,
                tuple(down_param.shape[-2:]),
                down_param,
                context,
                down_prep.master,  # type: ignore[arg-type]
            )


@contextlib.contextmanager
def _nvtx_phase(name: str, enabled: bool) -> Generator[None, None, None]:
    """Context manager that wraps a block with profiler + NVTX range when enabled."""
    if enabled:
        with torch.profiler.record_function(name), torch.cuda.nvtx.range(name):
            yield
    else:
        yield


def _step_stacked_moe_megabatch(
    pairs: list[tuple[nn.Parameter, nn.Parameter]],
    context: _StackedExpertStepContext,
) -> None:
    """Orthogonalize K consecutive (gate_up, down) pairs in one batched NS call.

    All K*3 matrices (gate, up, transposed-down) share the same [E, H, I] shape
    and are concatenated into [3K*E, H, I] for one
    :func:`zeropower_via_newtonschulz5_batched` call, reducing sequential NS
    invocations from K to 1.

    None-grad pairs are excluded entirely (no zero-padding), since NS is local
    and no collective requires uniform batch size.

    Args:
        pairs: List of ``(gate_up_param, down_param)`` tuples; all must share the
            same gate_up / down shapes.
        context: Optimizer context.
    """
    gate_pre_list: list[torch.Tensor] = []
    up_pre_list: list[torch.Tensor] = []
    down_pre_list: list[torch.Tensor] = []
    gate_active_list: list[torch.Tensor] = []
    up_active_list: list[torch.Tensor] = []
    down_active_list: list[torch.Tensor] = []
    gate_up_preps: list[_PreparedStackedExpertParam | None] = []
    down_preps: list[_PreparedStackedExpertParam | None] = []

    # Phase 1 — pre-NS: momentum update + Nesterov for all K pairs, batched across
    # layers with one call per tensor layout (gate+up, then down) so each compiled
    # graph sees a single layout.
    with _nvtx_phase("moe.megabatch.pre_ns", context.profile_phases):
        gate_grads: list[torch.Tensor] = []
        gate_moms: list[torch.Tensor] = []
        up_grads: list[torch.Tensor] = []
        up_moms: list[torch.Tensor] = []
        down_grads: list[torch.Tensor] = []
        down_moms: list[torch.Tensor] = []

        for gate_up_param, down_param in pairs:
            if gate_up_param.grad is not None:
                gu = _prepare_stacked_expert_param(gate_up_param, context)
                gate_grad, up_grad = gu.local_grad.chunk(2, dim=-1)
                gate_mom, up_mom = gu.momentum.chunk(2, dim=-1)
                gate_grads.append(gate_grad)
                gate_moms.append(gate_mom)
                up_grads.append(up_grad)
                up_moms.append(up_mom)
                gate_up_preps.append(gu)
            else:
                gate_up_preps.append(None)

            if down_param.grad is not None:
                d = _prepare_stacked_expert_param(down_param, context)
                down_grads.append(d.local_grad)
                down_moms.append(d.momentum)
                down_preps.append(d)
            else:
                down_preps.append(None)

        # Two batched momentum recurrences, one per layout: gate and up are both
        # [E, H, I] halves of the fused gate_up buffer and share a layout, while down
        # is [E, I, H]. Mixing the two in one compiled call is what breaks on resume
        # (see compute_pre_ns_updates_moe_expert).
        def _pre_ns(
            grads: list[torch.Tensor], moms: list[torch.Tensor]
        ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
            if not grads:
                return [], []
            return compute_pre_ns_updates_moe_expert(grads, moms, momentum=context.momentum, nesterov=context.nesterov)

        n_gate = len(gate_grads)
        gate_up_pre, gate_up_act = _pre_ns(gate_grads + up_grads, gate_moms + up_moms)
        gate_pre_list = gate_up_pre[:n_gate]
        up_pre_list = gate_up_pre[n_gate:]
        gate_active_list = gate_up_act[:n_gate]
        up_active_list = gate_up_act[n_gate:]

        down_pre, down_active_list = _pre_ns(down_grads, down_moms)
        down_pre_list = [t.mT for t in down_pre]  # [E, H, I] aligned with gate/up

    if not gate_pre_list and not down_pre_list:
        return

    all_pre = gate_pre_list + up_pre_list + down_pre_list
    assert all(t.shape[-2:] == all_pre[0].shape[-2:] for t in all_pre), (
        f"Shape mismatch in megabatch: expected all matrices to share shape "
        f"{all_pre[0].shape[-2:]}, got shapes {[tuple(t.shape) for t in all_pre]}."
    )

    # Phase 2 — NS: cat all [E, H, I] pre-NS tensors and run one batched NS call.
    with _nvtx_phase("moe.megabatch.ns", context.profile_phases):
        E = all_pre[0].shape[0]
        # Capture the block counts before dropping the lists (Phase 3 only needs
        # their lengths, not the tensors).
        n_g = len(gate_pre_list)
        n_d = len(down_pre_list)
        packed = torch.cat(all_pre, dim=0)  # [N_total * E, H, I]
        # ``torch.cat`` already copied every pre-NS tensor into ``packed``, but the
        # lists keep the originals alive for the whole NS call -- an extra buffer the
        # size of the NS input (~0.14 GB per K at [E=16, H=2048, I=768] bf16). Drop
        # them here so peak memory during NS is the iteration working set only.
        del all_pre, gate_pre_list, up_pre_list, down_pre_list
        packed_ortho = zeropower_via_newtonschulz5_batched(packed, steps=context.ns_steps)

    # Phase 3 — apply: split output and write back per-pair weight updates.
    with _nvtx_phase("moe.megabatch.apply", context.profile_phases):
        gate_block, up_block, down_block = packed_ortho.split([n_g * E, n_g * E, n_d * E], dim=0)
        gate_per = gate_block.split(E, dim=0) if n_g else ()
        up_per = up_block.split(E, dim=0) if n_g else ()
        down_per = down_block.split(E, dim=0) if n_d else ()

        # Group by layout, then apply one layout per batched call, for the same reason
        # the pre-NS above is split: gate/up slices and down must not share a graph.
        gate_up_group = _ExpertApplyGroup()
        down_group = _ExpertApplyGroup()

        g_idx = d_idx = 0
        for k, (gate_up_param, down_param) in enumerate(pairs):
            gu = gate_up_preps[k]
            if gu is not None:
                gate_width = gu.local_param.shape[-1] // 2
                matrix_shape = (gu.local_param.shape[-2], gate_width)
                base_lr = context.base_lr_for(gate_up_param)
                wd = context.weight_decay_for(gate_up_param)
                adj_lr = context.adjusted_lr_for(matrix_shape, base_lr)

                # Inactive-expert zeroing happens inside the compiled apply.
                gate_up_group.add(
                    gu.local_param[..., :gate_width],
                    gate_per[g_idx],
                    gate_active_list[g_idx],
                    base_lr,
                    wd,
                    adj_lr,
                    gu.master[..., :gate_width] if gu.master is not None else None,
                )
                gate_up_group.add(
                    gu.local_param[..., gate_width:],
                    up_per[g_idx],
                    up_active_list[g_idx],
                    base_lr,
                    wd,
                    adj_lr,
                    gu.master[..., gate_width:] if gu.master is not None else None,
                )
                g_idx += 1

            d = down_preps[k]
            if d is not None:
                base_lr = context.base_lr_for(down_param)
                down_group.add(
                    d.local_param,
                    down_per[d_idx].mT,
                    down_active_list[d_idx],
                    base_lr,
                    context.weight_decay_for(down_param),
                    context.adjusted_lr_for(tuple(down_param.shape[-2:]), base_lr),
                    d.master,
                )
                d_idx += 1

        for group in (gate_up_group, down_group):
            group.apply()


def step_stacked_expert_params(
    stacked_params: list[nn.Parameter],
    split_expert_pairs: list[tuple[nn.Parameter, nn.Parameter]],
    split_expert_param_ids: set[int],
    *,
    optimizer_state: MutableMapping,
    param_to_name: dict,
    param_to_master: dict,
    master_weights: bool,
    momentum: float,
    nesterov: bool,
    ns_steps: int,
    batch_split_expert_ns: bool,
    base_lr_for: Callable,
    weight_decay_for: Callable,
    adjusted_lr_for: Callable,
    moe_megabatches: "list[list[tuple[nn.Parameter, nn.Parameter]]] | None" = None,
    profile_phases: bool = False,
) -> None:
    """Orthogonalize all stacked MoE expert parameters.

    Dispatches to the appropriate path based on whether MoE megabatching is active:

    * ``moe_megabatches`` provided (non-None): Process groups of K pairs per NS round
      via :func:`_step_stacked_moe_megabatch` (K=1 falls back to single-pair path).
    * ``split_expert_pairs`` provided (no megabatches): Process each (gate_up, down)
      pair independently via :func:`_step_split_stacked_expert_pair`.
    * Remaining stacked params not in ``split_expert_param_ids``: processed via
      :func:`_step_one_stacked_expert_param` (the pre-split historical path).

    Args:
        stacked_params: All stacked ``[E, M, N]`` expert parameters managed by the
            optimizer (superset of paired params).
        split_expert_pairs: Paired ``(gate_up, down)`` tuples; empty if
            ``split_expert_gate_up`` is False.
        split_expert_param_ids: ``id()`` of every param that belongs to a pair
            (used to skip them in the unpaired loop).
        optimizer_state: ``self.state`` mapping (param -> state dict).
        param_to_name: Param -> dotted name mapping.
        param_to_master: ``id(param)`` -> FP32 master tensor (may be empty when
            ``master_weights=False``).
        master_weights: Whether FP32 masters are in use.
        momentum: Momentum coefficient.
        nesterov: Whether to use Nesterov momentum.
        ns_steps: Number of Newton-Schulz iterations.
        batch_split_expert_ns: Whether to batch gate+up+down into one NS call
            (used by :func:`_step_split_stacked_expert_pair` when megabatching is off).
        base_lr_for: Callable ``(p) -> lr``.
        weight_decay_for: Callable ``(p) -> wd``.
        adjusted_lr_for: Callable ``(matrix_shape, base_lr) -> adjusted_lr``.
        moe_megabatches: Pre-built list of K-pair batches for megabatch NS (built by
            ``_create_moe_megabatches``). Pass ``None`` to use the split-pair path.
        profile_phases: When True, emit ``torch.profiler.record_function`` and NVTX
            ranges for ``moe.megabatch.{pre_ns,ns,apply}`` inside megabatch steps.
    """
    context = _StackedExpertStepContext(
        optimizer_state=optimizer_state,
        param_to_name=param_to_name,
        param_to_master=param_to_master,
        master_weights=master_weights,
        momentum=momentum,
        nesterov=nesterov,
        ns_steps=ns_steps,
        batch_split_expert_ns=batch_split_expert_ns,
        profile_phases=profile_phases,
        base_lr_for=base_lr_for,
        weight_decay_for=weight_decay_for,
        adjusted_lr_for=adjusted_lr_for,
    )

    if moe_megabatches is not None:
        # Megabatch path: process groups of K pairs, one NS call per group.
        for batch in moe_megabatches:
            if len(batch) == 1:
                _step_split_stacked_expert_pair(batch[0][0], batch[0][1], context)
            else:
                _step_stacked_moe_megabatch(batch, context)
    elif split_expert_pairs:
        for gate_up_param, down_param in split_expert_pairs:
            _step_split_stacked_expert_pair(gate_up_param, down_param, context)

    for p in stacked_params:
        if id(p) in split_expert_param_ids or p.grad is None:
            continue
        _step_one_stacked_expert_param(p, context)
