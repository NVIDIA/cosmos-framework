# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import importlib

import pytest

from cosmos_framework.utils.lazy_config import lazy


@pytest.mark.L0
@pytest.mark.CPU
def test_lazy_config_module_can_be_loaded_after_equivalent_resolvers_are_registered() -> None:
    importlib.reload(lazy)
