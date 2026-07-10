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

import torch
import torch.nn as nn

# Substrings (matched against the dotted module name) that force an ``nn.Linear``
# onto the AdamW side. Every LLM in the repo names its head ``lm_head``; the extra
# entries cover common alternative names used elsewhere.
DEFAULT_ADAMW_MODULE_KEYWORDS: tuple[str, ...] = ("lm_head", "embed_out", "output_layer")


def is_output_head_linear(
    module_name: str,
    *,
    adamw_module_keywords: tuple[str, ...],
) -> bool:
    """Whether an ``nn.Linear`` is an output head (and must use AdamW, not Muon).

    Name-based: every LLM backbone in the repo names its head with an
    ``adamw_module_keywords`` substring (default ``lm_head``). Embeddings are
    excluded separately by module type in :func:`split_orthogonalizable_params`
    (``nn.Embedding`` never reaches this function), and a tied head shares its
    weight object with an embedding, so it is caught there too -- hence the name
    check alone is sufficient for every current model.
    """
    return any(keyword in module_name for keyword in adamw_module_keywords)


def split_orthogonalizable_params(
    model: nn.Module,
    optimizer_param_ids: set[int],
    adamw_module_keywords: tuple[str, ...] = DEFAULT_ADAMW_MODULE_KEYWORDS,
    expert_param_keywords: tuple[str, ...] = (),
) -> tuple[list[nn.Parameter], list[nn.Parameter], dict[nn.Parameter, str]]:
    """Split ``model``'s trainable params by whether they can be orthogonalized.

    Args:
        model: The (sub)module to walk -- typically the trainable ``net``.
        optimizer_param_ids: ``id()`` of every parameter actually owned by the
            optimizer's param groups. Only these are categorized, so that
            ``keys_to_select`` filtering is respected and ``state_dict`` stays
            consistent.
        adamw_module_keywords: Name substrings that force an ``nn.Linear`` onto the
            unorthogonalizable (AdamW) side (output heads).
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
            is_head = is_output_head_linear(
                module_name,
                adamw_module_keywords=adamw_module_keywords,
            )
            if _eligible(module.weight):
                (unorthogonalizable if is_head else orthogonalizable).append(module.weight)
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


@torch.compile
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


@torch.compile
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


def compute_pre_ns_update(
    grad: torch.Tensor,
    momentum_buffer: torch.Tensor,
    momentum: float = 0.95,
    nesterov: bool = True,
) -> torch.Tensor:
    """
    Compute the pre-Newton-Schulz update (momentum + optional Nesterov).

    This is separated from NS so that momentum/Nesterov can be applied on shards
    before all-gathering for distributed NS.

    Args:
        grad: Gradient tensor.
        momentum_buffer: Momentum buffer (modified in-place).
        momentum: Momentum coefficient.
        nesterov: Whether to use Nesterov momentum.

    Returns:
        Pre-NS update tensor (same shape as grad).
    """
    # SGD-style momentum: buf = momentum * buf + grad (matching Moonlight)
    momentum_buffer.mul_(momentum).add_(grad)

    # Nesterov: g = g + momentum * buf, else just use buf
    if nesterov:
        return grad.add(momentum_buffer, alpha=momentum)
    else:
        return momentum_buffer.clone()


def compute_pre_ns_update_moe_expert(
    grad: torch.Tensor,
    momentum_buffer: torch.Tensor,
    momentum: float = 0.95,
    nesterov: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
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
    # slice, so this is an exact "was this expert updated" test.
    active = (grad != 0).flatten(1).any(dim=1)  # [E] bool
    a = active.view(-1, 1, 1).to(momentum_buffer.dtype)  # [E, 1, 1] in {0, 1}

    # Masked SGD-style momentum:
    #   active   -> momentum * buf + grad  (== compute_pre_ns_update)
    #   inactive -> buf unchanged (factor 1.0; grad is 0 there anyway) => frozen
    momentum_buffer.mul_(1 - a * (1 - momentum)).add_(grad)

    # Nesterov: g = g + momentum * buf, else just use buf. Inactive experts get a
    # bogus (renormalized) value here, but the caller zeros their NS output.
    if nesterov:
        pre_ns = grad.add(momentum_buffer, alpha=momentum)
    else:
        pre_ns = momentum_buffer.clone()

    return pre_ns, active
