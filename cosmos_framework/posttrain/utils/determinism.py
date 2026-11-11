# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""The one determinism recipe for posttrain, so cross-run equivalence gates are provable.

Hoisted verbatim from `tests/parity_cli._enable_determinism`, which `test_parity_cosmos3.py`
already relies on for a bit-identical 8B loss on GB200 — this is the single source of truth;
`parity_cli` now delegates here instead of carrying a second copy of the recipe.

No `torch.compile` gate here because nothing on the GRPO path compiles today: the HF backend
(`model.hf.HFModel`) has no compile toggle and the vLLM rollout defaults `enforce_eager=True`.
(`parity_cli`'s `compile.enabled = False` targets the DPO diffusion backbone, which GRPO never
builds.) A future compiled GRPO backend should gate on `RunConfig.deterministic`, as DPO does.
"""

from __future__ import annotations

import os

from cosmos_framework.posttrain.utils.log import info0


def enable_determinism(*, strict: bool = False) -> None:
    """Set PyTorch's determinism knobs so the loss is bit-exact across reruns on the same GPU and
    torch build. Lifted from `tests/parity_cli._enable_determinism`, which `test_parity_cosmos3.py`
    already relies on for a bit-identical 8B loss.

    MUST run before any CUDA / flash-attn init, in EVERY process that touches CUDA — the driver AND
    each Ray actor (separate processes that do not inherit the driver's `os.environ`). cuBLAS reads
    `CUBLAS_WORKSPACE_CONFIG` when it creates its handle, so setting it after init silently does
    nothing: the run then LOOKS configured while drifting. The `Raises` guard makes that loud, which
    is why only the launch environment (not a late mutation) reliably reaches the actors.

    The legacy default warns and proceeds when a deterministic implementation is unavailable.
    ``strict=True`` requires deterministic algorithms and raises instead. In PyTorch 2.10,
    warning-only mode also permits nondeterministic SDPA FlashAttention backward even when a
    deterministic backward exists. Strict mode is therefore necessary for controlled continuation
    checks; it does not promise bitwise equality across different devices or world sizes.

    Raises:
        RuntimeError: `CUBLAS_WORKSPACE_CONFIG` was not already set (e.g. inherited from the launch
            environment) AND CUDA is already initialized, so setting it now would silently not take
            effect — the run would appear configured while remaining non-deterministic.
    """
    import torch

    if "PYTHONHASHSEED" not in os.environ:
        # Python fixes the hash seed at interpreter startup, so it must come from the launch env.
        # `info0` (not `print`) tags the line — this runs in every Ray actor process.
        info0("[determinism] PYTHONHASHSEED unset — prepend PYTHONHASHSEED=42 for a fully bit-exact run")
    if "CUBLAS_WORKSPACE_CONFIG" not in os.environ and torch.cuda.is_initialized():
        raise RuntimeError(
            "CUBLAS_WORKSPACE_CONFIG is unset and CUDA is already initialized: cuBLAS reads this "
            "variable when it creates its handle, so setting it now would silently do nothing and the "
            "run would remain non-deterministic while appearing configured. Export "
            "CUBLAS_WORKSPACE_CONFIG=:4096:8 in the LAUNCH environment (before the process starts, so "
            "Ray actors — separate processes — inherit it too) and rerun."
        )
    os.environ["FLASH_ATTENTION_DETERMINISTIC"] = "1"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=not strict)
