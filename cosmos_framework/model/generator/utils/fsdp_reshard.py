# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Scoped FSDP2 reshard deferral for repeated model calls."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch


@contextmanager
def defer_fsdp_reshard(
    model: torch.nn.Module,
    *,
    after_forward: bool = False,
    after_backward: bool = False,
    reshard_root_on_exit: bool = False,
) -> Iterator[None]:
    """Keep FSDP2 parameters unsharded across repeated forwards or backwards.

    This scope is opt-in because retaining full parameters increases peak device
    memory. Descendant reshard policies are restored on exit and those modules
    are explicitly resharded. Modules are selected by capability so tests and
    non-FSDP configurations do not require importing PyTorch's private
    composable-FSDP type. The model root is not changed for forward deferral:
    FSDP2 resolves its default policy to ``False`` so root parameters remain
    gathered through backward. Policies that are already disabled are left
    untouched, including their gathered parameters. Partial (integer) forward
    policies, mixed policies within a module, and unknown FSDP state layouts are
    rejected before any policies are changed. Callers must not nest these scopes.

    ``reshard_root_on_exit`` is intended for frozen/no-grad networks that have
    no subsequent backward to release the root parameters.
    """
    root_reshard = getattr(model, "reshard", None)
    forward_modules: list[Any] = (
        [
            module
            for module in model.modules()
            if module is not model
            and callable(getattr(module, "set_reshard_after_forward", None))
            and callable(getattr(module, "reshard", None))
        ]
        if after_forward
        else []
    )
    backward_modules: list[Any] = (
        [
            module
            for module in model.modules()
            if callable(getattr(module, "set_reshard_after_backward", None))
            and callable(getattr(module, "reshard", None))
        ]
        if after_backward
        else []
    )
    # FSDP2 exposes setters but no public getters. Inspect the same state they
    # update, and validate every candidate before mutating any policies.
    forward_policies: list[tuple[Any, Any, bool | None]] = []
    backward_policies: list[Any] = []
    for module in dict.fromkeys([*forward_modules, *backward_modules]):
        get_state = getattr(module, "_get_fsdp_state", None)
        if not callable(get_state):
            raise RuntimeError("Cannot preserve reshard policies without FSDP2 state access")
        state = get_state()
        # PyTorch 2.12 supports multiple parameter groups per FSDP module;
        # earlier versions expose only the singular parameter-group attribute.
        if hasattr(state, "_fsdp_param_groups"):
            groups = state._fsdp_param_groups
            if not isinstance(groups, list):
                raise RuntimeError("Cannot preserve reshard policies with an unknown FSDP2 parameter-group layout")
        elif hasattr(state, "_fsdp_param_group"):
            group = state._fsdp_param_group
            groups = [group] if group is not None else []
        else:
            raise RuntimeError("Cannot preserve reshard policies with an unknown FSDP2 state layout")
        if not groups:
            continue  # Parameterless FSDP modules have no reshard policy to change.
        if module in forward_modules:
            policies: set[bool] = set()
            for group in groups:
                if not hasattr(group, "post_forward_mesh_info") or not hasattr(group, "mesh_info"):
                    raise RuntimeError("Cannot preserve the FSDP2 forward reshard policy")
                if group.post_forward_mesh_info is not None and group.post_forward_mesh_info != group.mesh_info:
                    raise ValueError("Reshard deferral does not support partial (integer) forward reshard policies")
                policies.add(group.post_forward_mesh_info is not None)
            if len(policies) != 1:
                raise ValueError("Reshard deferral does not support mixed forward reshard policies within a module")
            if True in policies:
                auto_policy = getattr(state, "_auto_reshard_after_forward", None)
                if auto_policy is not None and not isinstance(auto_policy, bool):
                    raise RuntimeError("Cannot preserve the FSDP2 automatic forward reshard policy")
                forward_policies.append((module, state, auto_policy))
        if module in backward_modules:
            policies = set()
            for group in groups:
                policy = getattr(group, "reshard_after_backward", None)
                if not isinstance(policy, bool):
                    raise RuntimeError("Cannot preserve the FSDP2 backward reshard policy")
                policies.add(policy)
            if len(policies) != 1:
                raise ValueError("Reshard deferral does not support mixed backward reshard policies within a module")
            if True in policies:
                backward_policies.append(module)

    disabled_forward_modules: list[tuple[Any, Any, bool | None]] = []
    disabled_backward_modules: list[Any] = []
    entered = False
    try:
        for module, state, auto_policy in forward_policies:
            disabled_forward_modules.append((module, state, auto_policy))
            module.set_reshard_after_forward(False, recurse=False)
        for module in backward_policies:
            disabled_backward_modules.append(module)
            module.set_reshard_after_backward(False, recurse=False)
        entered = True
        yield
    finally:
        for module, state, auto_policy in disabled_forward_modules:
            module.set_reshard_after_forward(True, recurse=False)
            if auto_policy is not None:
                state._auto_reshard_after_forward = auto_policy
        for module in disabled_backward_modules:
            module.set_reshard_after_backward(True, recurse=False)
        reshard_modules = list(
            dict.fromkeys([*(module for module, _, _ in disabled_forward_modules), *disabled_backward_modules])
        )
        for module in reshard_modules:
            module.reshard()
        if entered and reshard_root_on_exit and callable(root_reshard):
            root_reshard()
