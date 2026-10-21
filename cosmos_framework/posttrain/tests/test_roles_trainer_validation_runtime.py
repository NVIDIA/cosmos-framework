# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Shared validation inventory does not depend on a preference-policy contract."""

from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.posttrain.roles.trainer.validation.runtime import validation_module_inventory

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def test_inventory_deduplicates_aliases_and_nested_roots_without_changing_modes() -> None:
    child = torch.nn.Linear(2, 2)
    net = torch.nn.Sequential(child, torch.nn.Dropout())
    auxiliary = torch.nn.Sequential(child, torch.nn.ReLU())
    child.eval()
    model = SimpleNamespace(net=net, net_ref=net, net_ema=child, auxiliary=auxiliary)
    modes = {module: module.training for root in (net, auxiliary) for module in root.modules()}

    roots, modules = validation_module_inventory(model)

    assert roots == {"net": net, "auxiliary": auxiliary}
    assert len(modules) == len(set(modules)) == len(modes)
    assert {module: module.training for module in modules} == modes


def test_inventory_ignores_non_module_attributes() -> None:
    assert validation_module_inventory(SimpleNamespace(net=None, metadata={})) == ({}, [])
