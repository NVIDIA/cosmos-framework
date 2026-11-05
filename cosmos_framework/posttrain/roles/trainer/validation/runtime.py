# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Algorithm-independent module inventory and FSDP cleanup for trainer validation."""

from __future__ import annotations

from collections.abc import Mapping

import torch


def validation_module_inventory(
    model: object,
) -> tuple[dict[str, torch.nn.Module], list[torch.nn.Module]]:
    """Return eval roots and every module whose mode affects held-out evaluation."""
    candidates: dict[str, torch.nn.Module] = {}
    for name in ("net", "net_ref", "net_ema"):
        value = getattr(model, name, None)
        if isinstance(value, torch.nn.Module) and all(value is not candidate for candidate in candidates.values()):
            candidates[name] = value
    for name, value in vars(model).items():
        if isinstance(value, torch.nn.Module) and all(value is not candidate for candidate in candidates.values()):
            candidates[name] = value
    roots = {
        name: candidate
        for name, candidate in candidates.items()
        if not any(
            candidate is not possible_parent and candidate in possible_parent.modules()
            for possible_parent in candidates.values()
        )
    }
    modules: list[torch.nn.Module] = []
    seen: set[int] = set()
    for root in roots.values():
        for module in root.modules():
            if id(module) in seen:
                continue
            seen.add(id(module))
            modules.append(module)
    return roots, modules


def reshard_fsdp_modules(roots: Mapping[str, torch.nn.Module | None]) -> None:
    """Best-effort reshard every unique FSDP2 unit below validation roots."""
    from torch.distributed.fsdp import FSDPModule

    seen: set[int] = set()
    failures: list[str] = []
    for root_name, root in roots.items():
        if not isinstance(root, torch.nn.Module):
            continue
        for module_name, module in root.named_modules():
            if id(module) in seen:
                continue
            seen.add(id(module))
            if not isinstance(module, FSDPModule):
                continue
            try:
                module.reshard()
            except BaseException as error:  # noqa: BLE001 -- best effort across every FSDP unit
                label = f"{root_name}.{module_name}" if module_name else root_name
                failures.append(f"{label}: {type(error).__name__}: {error}")
    if failures:
        raise RuntimeError("; ".join(failures))
