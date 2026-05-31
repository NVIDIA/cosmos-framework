# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from pathlib import Path

import pytest

from cosmos_framework.inference.fixtures.script import ScriptConfig, script_test

_CURRENT_DIR = Path(__file__).parent.absolute()
_TEST_DIR = _CURRENT_DIR / "_test"

_script_configs: list[ScriptConfig] = [
    ScriptConfig(
        script=_TEST_DIR / "convert_model_to_dcp.sh",
    ),
]


@script_test(_script_configs)
class TestScript: ...
