# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""The Qwen3-VL-8B RoboVQA SFT recipe: one epoch of 64-clip updates on the native seat, accepted by the SFT preflight."""

from __future__ import annotations

import importlib.util
import json
import types
from pathlib import Path
from typing import Any

import pytest

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


@pytest.fixture
def apex_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``importlib.util.find_spec("apex")`` succeed so fusedadam acceptance does not depend on the test runtime."""
    real = importlib.util.find_spec

    def find_spec(name: str, *args: Any, **kwargs: Any) -> Any:
        return types.SimpleNamespace(name="apex") if name == "apex" else real(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)


@pytest.fixture
def recipe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    from cosmos_framework.posttrain.configs.sft_qwen3_vl_8b_robovqa import make_config

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"model_type": "qwen3_vl"}))
    data_root = tmp_path / "data"
    data_root.mkdir()
    (data_root / "train.jsonl").write_text("{}\n")
    monkeypatch.setenv("SFT_MODEL_PATH", str(model_dir))
    monkeypatch.setenv("SFT_DATA_ROOT", str(data_root))
    return make_config()


def test_recipe_is_one_epoch_of_64_clip_updates(recipe: Any, tmp_path: Path) -> None:
    assert recipe.model_backend == "i4"
    assert (recipe.rl.micro_batch_size, recipe.rl.grad_accum, recipe.rl.max_steps) == (1, 64, 125)
    assert recipe.rl.save_every == recipe.rl.max_steps
    assert recipe.rl.checkpoint_format == "consolidated"
    assert recipe.rl.max_length == 10240
    assert recipe.rl.optimizer.name == "fusedadam" and recipe.rl.optimizer.lr == 2e-5
    assert recipe.rl.optimizer.train_keys == []
    assert recipe.rl.processor_kwargs == {"videos_kwargs": {"do_sample_frames": False}}
    assert recipe.rl.validation.enabled is False
    assert recipe.rl.data.path == str(tmp_path / "data" / "train.jsonl")


def test_recipe_passes_the_sft_preflight_on_the_native_qwen3_vl_seat(
    recipe: Any, tmp_path: Path, apex_present: None
) -> None:
    from cosmos_framework.posttrain.entrypoints.sft import sft_policy_source, validate_config

    rl = validate_config(recipe)
    assert rl.data.path == str(tmp_path / "data" / "train.jsonl")
    assert sft_policy_source(recipe) == (str(tmp_path / "model"), False)
