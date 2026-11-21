# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""Fail closed on compiler fallback for the opt-in, single-policy native path."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn


def guard_native_compiled_call(module: nn.Module) -> nn.Module:
    """Keep Dynamo limit/error fallback fatal only while this compiled call runs.

    ``torch.compile`` is lazy, so a config patch around its construction is not
    sufficient. Patch the compiled entrypoint at invocation instead. No new
    module wrapper, parameter owner, or state-dict prefix is introduced. This is
    restricted to the qualified single-policy process; concurrent unrelated
    compilation in another thread is not part of that execution contract.
    """
    if module.__dict__.get("_cosmos_compile_guarded", False):
        return module
    compiled_forward = module.forward

    def forward(*args: Any, **kwargs: Any) -> Any:
        object.__setattr__(module, "_cosmos_compile_invocations", module.__dict__["_cosmos_compile_invocations"] + 1)
        with torch._dynamo.config.patch(suppress_errors=False, fail_on_recompile_limit_hit=True):
            return compiled_forward(*args, **kwargs)

    module.forward = forward
    object.__setattr__(module, "_cosmos_compile_guarded", True)
    object.__setattr__(module, "_cosmos_compile_invocations", 0)
    return module


def native_compile_diagnostics(modules: Iterable[nn.Module] = ()) -> dict[str, Any]:
    """Aggregate process counters without publishing source paths or break text.

    Counts include earlier kernel preflights in this process. A called compiled
    entrypoint can contain declared graph breaks; these diagnostics do not claim
    every operation was fused or establish a speed benefit.
    """
    from torch._dynamo.utils import counters, guard_failures

    guarded = [module for module in modules if module.__dict__.get("_cosmos_compile_guarded", False)]
    calls = [int(module.__dict__["_cosmos_compile_invocations"]) for module in guarded]
    return {
        "scope": "process_cumulative",
        "frames_total": int(counters["frames"]["total"]),
        "frames_compiled": int(counters["frames"]["ok"]),
        "captured_calls": int(counters["stats"]["calls_captured"]),
        "unique_graphs": int(counters["stats"]["unique_graphs"]),
        "graph_breaks": int(sum(counters["graph_break"].values())),
        # Guard failures trigger specialization/recompilation attempts; they
        # are not an exact count of newly generated machine-code artifacts.
        "guard_failures": int(sum(len(reasons) for reasons in guard_failures.values())),
        "guarded_modules": len(guarded),
        "guarded_modules_called": sum(count > 0 for count in calls),
        "guarded_module_forward_calls": calls,
    }
