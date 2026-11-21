# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""CPU contract checks; only native accelerator runs establish recovery support."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location("qualification_lifecycle", Path(__file__).with_name("lifecycle.py"))
lifecycle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lifecycle)


def test_overrides_preserve_hyperparameters_and_reject_hidden_protocol_changes():
    requested = ["model.model_config.adapter.rank=32", "rl.optimizer.lr=0.00005", "rl.seed=7"]
    result = lifecycle.controlled_overrides(requested)
    assert result[:3] == requested
    assert "rl.max_steps=4" in result
    for wrong in (["rl.max_steps=300"], ["rl.resume=auto"], ["rl.seed=7", "rl.seed=8"], ["+rl.seed=7"]):
        with pytest.raises(ValueError):
            lifecycle.controlled_overrides(wrong)


def update_row(step=3):
    row = dict.fromkeys(lifecycle.NUMERICAL_METRICS, 0.0)
    row.update(
        {
            "step": step,
            "train/loss_avg": 0.2,
            "train/grad_norm": 1.2,
            "train/learning_rate": 0.0001,
            "train/valid_token_count": 16.0,
            "train/valid_sequence_count": 8.0,
            "optim/optimizer_step": step,
            "optim/attempt_id": step,
            "optim/finite_gradient_check_enabled": 1,
            "optim/update_outcome": "applied",
        }
    )
    return row


def test_next_update_compares_numerics_but_not_cold_cache_or_timing(tmp_path):
    first, second = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    row = update_row()
    first.write_text(json.dumps({**row, "timer/e2e_step_s": 10, "train/preprocessing_videos_hits": 2}) + "\n")
    second.write_text(json.dumps({**row, "timer/e2e_step_s": 20, "train/preprocessing_videos_hits": 0}) + "\n")
    assert lifecycle.compare_next_metrics(first, second, 3)["passed"]
    row["train/grad_norm"] += 0.0000001
    second.write_text(json.dumps(row) + "\n")
    with pytest.raises(AssertionError, match="numerical"):
        lifecycle.compare_next_metrics(first, second, 3)


@pytest.mark.parametrize("defect", ["duplicate", "missing", "skip", "nonfinite", "clock"])
def test_next_update_cannot_hide_invalid_metrics(tmp_path, defect):
    path = tmp_path / "metrics.jsonl"
    row = update_row()
    if defect == "skip":
        row["optim/step_skipped"] = 1
    elif defect == "nonfinite":
        row["train/loss_avg"] = float("nan")
    elif defect == "clock":
        row["optim/attempt_id"] = 4
    rows = [] if defect == "missing" else [row, row] if defect == "duplicate" else [row]
    path.write_text("".join(json.dumps(value) + "\n" for value in rows))
    with pytest.raises(ValueError):
        lifecycle.compare_next_metrics(path, path, 3)


def test_subprocess_failure_preserves_own_attempt_and_forbids_overwrite(tmp_path):
    stages = lifecycle.Stages(tmp_path, {}, 10)
    with pytest.raises(RuntimeError, match="unexpected status"):
        stages.run("failure", [sys.executable, "-c", "print('preserved failure'); raise SystemExit(3)"])
    assert (tmp_path / "stages/failure/output.log").read_text().strip() == "preserved failure"
    assert json.loads((tmp_path / "stages/failure/outcome.json").read_text())["exit_code"] == 3
    with pytest.raises(FileExistsError):
        stages.run("failure", [sys.executable, "-c", "pass"])


def test_expected_interruption_must_not_return_success(tmp_path):
    stages = lifecycle.Stages(tmp_path, {}, 10)
    with pytest.raises(RuntimeError, match="unexpected status"):
        stages.run("lost_injection", [sys.executable, "-c", "pass"], interrupted=True)


def test_plan_only_records_selected_native_controls_without_launch_and_rejects_changed_spec(tmp_path, monkeypatch):
    from transformers import Qwen3VLConfig

    from cosmos_framework.posttrain.utils import adapter_source

    source = tmp_path / "source"
    Qwen3VLConfig().save_pretrained(source)
    monkeypatch.setenv("SFT_MODEL_PATH", str(source))
    monkeypatch.setenv("COSMOS_TRAINING_CODE_REVISION", "a" * 40)
    monkeypatch.setenv("COSMOS_TRAINING_WHEEL_SHA256", "b" * 64)
    monkeypatch.setattr(lifecycle.subprocess, "Popen", lambda *a, **k: pytest.fail("plan-only launched a process"))
    # Source-byte validation has its own real-manifest suite; this fixture isolates
    # immutable planning and must not require downloading an 8B checkpoint.
    monkeypatch.setattr(adapter_source, "prepare_sft_adapter_source", lambda _: None)
    data = tmp_path / "validation.jsonl"
    data.write_text("{}\n")
    args = SimpleNamespace(
        config="cosmos_framework/posttrain/configs/sft_qwen3_vl_8b_qlora.py",
        output=str(tmp_path / "plan"),
        data=str(data),
        image=None,
        spec=None,
        plan_only=True,
        stage_timeout=1,
        override=["model.model_config.adapter.rank=32", "model.model_config.adapter.alpha=64", "rl.seed=7"],
    )
    result = lifecycle.run(args)
    assert result["plan_only"] and not result["complete"]
    spec = json.loads((tmp_path / "plan/experiment.json").read_text())
    assert spec["selected"]["rank"] == 32 and spec["selected"]["alpha"] == 64 and spec["selected"]["seed"] == 7
    assert spec["selected"]["adapter_initialization_seed"] == 42  # Distinct, explicitly recorded seed policy.
    args.spec = str(tmp_path / "plan/experiment.json")
    args.output = str(tmp_path / "changed")
    args.override = ["rl.seed=8"]
    with pytest.raises(ValueError, match="immutable recovery experiment"):
        lifecycle.run(args)
    assert json.loads((tmp_path / "changed/failure.json").read_text())["complete"] is False
