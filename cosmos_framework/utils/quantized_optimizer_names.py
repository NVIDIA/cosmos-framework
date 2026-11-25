# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Names of the opt-in bitsandbytes eight-bit AdamW optimizers (non-paged, paged).

The single definition, torch-free and import-light, so configuration code can name the
optimizers without importing torch, bitsandbytes or ``cosmos_framework.utils.quantized_optimizer``.
"""

import math
from collections.abc import Sequence
from numbers import Real

QUANTIZED_OPTIMIZER_NAMES: tuple[str, ...] = ("adamw_bnb_8bit", "paged_adamw_8bit")


def validate_quantized_options(
    *,
    lr: object,
    eps: object,
    weight_decay: object,
    betas: Sequence[object],
    bnb_min_8bit_size: object,
    bnb_fp32_state_names: Sequence[object],
    owner: str = "OptimizerConfig",
) -> None:
    """Refuse options the eight-bit backend cannot run, before any model or backend allocation.

    The one check shared by ``OptimizerConfig`` and the generator ``OptimizersContainer``.
    """
    if type(bnb_min_8bit_size) is not int or bnb_min_8bit_size < 1:
        raise ValueError("bnb_min_8bit_size must be a positive integer")
    if any(not isinstance(n, str) or not n for n in bnb_fp32_state_names):
        raise ValueError("bnb_fp32_state_names must contain exact nonempty parameter names")
    if len(set(bnb_fp32_state_names)) != len(bnb_fp32_state_names):
        raise ValueError("duplicate bnb_fp32_state_names")
    for key, value in (("lr", lr), ("eps", eps), ("weight_decay", weight_decay)):
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{owner}.{key} must be finite and non-negative for bitsandbytes")
    if len(betas) != 2 or any(
        isinstance(beta, bool) or not isinstance(beta, Real) or not math.isfinite(beta) or not 0 <= beta < 1
        for beta in betas
    ):
        raise ValueError(f"{owner}.betas must contain two finite values in [0, 1) for bitsandbytes")


def is_quantized_optimizer(optimizer: object) -> bool:
    """Whether ``optimizer`` runs the bitsandbytes eight-bit backend (directly or as an FP32-master backend)."""
    return bool(getattr(optimizer, "cosmos_quantized_optimizer", False))


def is_local_master_optimizer(optimizer: object) -> bool:
    """Whether ``optimizer`` is the FP32-master update owner (``update_policy=fp32_master``).

    Its state is rank-local (masters, moments) and uses its own checkpoint schema.
    """
    return bool(getattr(optimizer, "cosmos_local_master_optimizer", False))


def supports_host_offload(optimizer: object) -> bool:
    """Whether generic trainer host offload may move ``optimizer`` state (the opt-in optimizers refuse it)."""
    return not (is_quantized_optimizer(optimizer) or is_local_master_optimizer(optimizer))


def refuse_rank_local_optimizer(
    optimizer: object,
    checkpointer: str,
    remedy: str = (
        "use cosmos_framework.checkpoint.dcp.DistributedCheckpointer with checkpoint.dcp_async_mode_enabled=false"
    ),
) -> None:
    """Refuse an optimizer whose state is rank-local, in a checkpointer that cannot write it.

    The eight-bit generator container and the FP32-master owner keep per-rank masters and
    moments. Checkpointers that save only rank 0's state, or reshape optimizer state through
    torch's generic DCP conversion, would write a checkpoint that cannot resume, so they call
    this from both ``save`` and ``load``. The trainer always loads (or finds nothing to load)
    before its first save, so the refusal happens before any training. ``optimizer`` may be a
    keyed collection of optimizers (anything with ``items()``: a dict, or the distillation
    ``PhaseOptimizer``), and each entry may be a container holding raw ``optimizers``.
    """
    items = getattr(optimizer, "items", None)
    entries = [value for _, value in items()] if callable(items) else [optimizer]
    for entry in entries:
        inner = getattr(entry, "optimizers", None)
        for candidate in (entry, *(inner if isinstance(inner, (list, tuple)) else ())):
            if getattr(candidate, "rank_local_checkpoint", False) or is_local_master_optimizer(candidate):
                raise ValueError(
                    f"{checkpointer} cannot checkpoint rank-local (eight-bit / FP32-master) optimizer state; {remedy}"
                )
