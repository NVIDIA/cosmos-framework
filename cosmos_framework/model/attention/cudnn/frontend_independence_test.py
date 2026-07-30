# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""
Unit test guarding the cuDNN backend's independence from the standalone cuDNN Python frontend.

The backend runs attention through PyTorch's own cuDNN SDPA dispatch
(``torch.ops.aten._scaled_dot_product_cudnn_attention``), so no module in this package may
``import cudnn`` (the ``nvidia-cudnn-frontend`` package). Checked statically so it holds even
where the frontend happens to be installed.
"""

import ast
from pathlib import Path

import pytest

CUDNN_BACKEND_DIR = Path(__file__).parent


def _imported_top_level_modules(source: str) -> set[str]:
    modules = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        # Relative imports (node.level > 0) can never name the standalone frontend.
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.add(node.module.split(".")[0])
    return modules


@pytest.mark.L0
class TestCudnnFrontendIndependence:
    """The cuDNN backend must not depend on the standalone cuDNN Python frontend."""

    def test_no_module_imports_the_cudnn_frontend(self):
        offenders = [
            path.name
            for path in sorted(CUDNN_BACKEND_DIR.glob("*.py"))
            if "cudnn" in _imported_top_level_modules(path.read_text())
        ]
        assert not offenders, (
            f"{offenders} import the standalone cuDNN frontend; the backend must go through "
            "torch.ops.aten._scaled_dot_product_cudnn_attention instead."
        )
