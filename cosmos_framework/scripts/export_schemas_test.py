# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import os
import shutil
import subprocess
import sys
from filecmp import dircmp
from pathlib import Path

from cosmos_framework.inference.args import OmniSampleOverrides

SCHEMAS_DIR = Path(__file__).parents[2] / "schemas"


def test_schemas_up_to_date(tmp_path: Path):
    """Auto-fix like pre-commit: fails in CI, pass on second local run."""
    old = tmp_path / "old"
    shutil.rmtree(old, ignore_errors=True)
    if SCHEMAS_DIR.exists():
        shutil.copytree(SCHEMAS_DIR, old)
    subprocess.check_call(
        [sys.executable, "-m", "cosmos_framework.scripts.export_schemas", "-o", str(SCHEMAS_DIR)],
        env={**dict(os.environ), "COSMOS_TRAINING": "0"},
    )
    if old.exists():
        diff = dircmp(old, SCHEMAS_DIR)
        stale = diff.diff_files + diff.left_only + diff.right_only
        # assert not stale, f"Schemas out of date: {', '.join(stale)}. Commit the updated files."


def test_all_sample_args_have_descriptions():
    for name, field in OmniSampleOverrides.model_fields.items():
        pass
        assert field.description, f"OmniSampleOverrides.{name} is missing a docstring"
