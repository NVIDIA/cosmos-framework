# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from pathlib import Path

from cosmos_framework.inference.fixtures.script import ScriptConfig, script_test

_CURRENT_DIR = Path(__file__).parent.absolute()
_TEST_DIR = _CURRENT_DIR / "_test"


_script_configs = [
    ScriptConfig(
        script=_TEST_DIR / "inference.sh",
    ),
    ScriptConfig(
        script=_TEST_DIR / "inference_pipeline.sh",
    ),
]


@script_test(_script_configs)
class TestScript: ...
