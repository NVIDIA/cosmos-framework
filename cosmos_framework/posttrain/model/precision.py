# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""The forward-time half of `rl.mixed_precision_mode`: how the compute dtype reaches a forward.

"fsdp": nothing here -- `model/parallelize.py` attaches a `MixedPrecisionPolicy` whose parameter cast delivers
`ShardPlan.param_dtype`. "autocast": `torch.autocast(param_dtype)` around the forward with parameters left in the
storage dtype; the plan then carries `param_dtype=None` (`fsdp_param_dtype`), so the FSDP wrap casts neither
parameters nor forward inputs and a wrapped forward equals an unwrapped one at every world size.
"""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext

import torch


def cast_floating_buffers(
    root: torch.nn.Module,
    dtype: torch.dtype,
    *,
    originals: list[tuple[torch.nn.Module, str, torch.Tensor]] | None = None,
) -> None:
    """Cast registered bindings like Module.to, leaving ordinary aliases untouched.

    A temporary caller supplies its restoration journal before conversion, so even
    a failed conversion can restore all earlier bindings. Permanent callers omit it.
    """
    for module in root.modules():
        for name, buffer in module.named_buffers(recurse=False, remove_duplicate=False):
            if buffer.is_floating_point() and buffer.dtype != dtype:
                if originals is not None:
                    originals.append((module, name, buffer))
                module._buffers[name] = buffer.to(dtype)


def compute_context(mode: str, dtype: torch.dtype, device_type: str = "cuda") -> AbstractContextManager:
    """Return the context a forward runs under. Build it once and re-enter it every forward."""
    if mode == "fsdp":
        return nullcontext()
    if mode == "autocast":
        return torch.autocast(device_type, dtype=dtype)
    raise ValueError(f"unknown mixed precision mode {mode!r}; expected 'fsdp' or 'autocast'")


def fsdp_param_dtype(mode: str, compute_dtype: torch.dtype) -> torch.dtype | None:
    """What `ShardPlan.param_dtype` carries: the compute dtype the FSDP2 parameter cast delivers in mode "fsdp", or
    `None` in mode "autocast", where the context above owns the compute dtype and the wrap must stay out of the way.
    `None` is FSDP2's documented "no cast", and it also disarms `MixedPrecisionPolicy.cast_forward_inputs`, which
    otherwise casts every wrapped module's floating inputs to `param_dtype`; carrying the storage dtype instead
    would make a wrapped block see fp32 where the unwrapped one sees bf16."""
    if mode == "fsdp":
        return compute_dtype
    if mode == "autocast":
        return None
    raise ValueError(f"unknown mixed precision mode {mode!r}; expected 'fsdp' or 'autocast'")
