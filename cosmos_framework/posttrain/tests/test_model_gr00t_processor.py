# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""The real Isaac processor remains selected when Omni registers its processor last."""

import os
import subprocess
import sys

import pytest

pytestmark = [pytest.mark.L1, pytest.mark.CPU]


def test_native_processor_is_independent_of_omni_registration() -> None:
    checkpoint = os.environ["GR00T_PROCESSOR_CHECKPOINT"]
    script = """
import importlib
import sys
modules = [
    'gr00t.model.gr00t_n1d7.setup',
    'vllm_omni.diffusion.models.gr00t.modeling.processing_gr00t_n1d7',
]
for module in modules:
    importlib.import_module(module)
from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor
from cosmos_framework.posttrain.model.gr00t.policy import load_processor
processor = load_processor(sys.argv[1])
assert type(processor) is Gr00tN1d7Processor, type(processor)
processor.eval()
assert callable(processor.collator)
print(type(processor).__module__)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, checkpoint],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
