# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""CPU checks for comparison/measurement tools; these do not qualify NF4 kernels."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest
import torch

from cosmos_framework.posttrain.tests.qlora_integration import (
    _assert_close_tree,
    _load_flat_dcp,
    _portable_trainer_controls,
    _probe_rows,
)
from cosmos_framework.posttrain.utils.qlora_metrics import record_qlora_phase


def test_flat_checkpoint_comparison_preserves_tensor_and_control_state(tmp_path):
    import torch.distributed.checkpoint as dcp

    path = tmp_path / "optim"
    state = {"state": {"factor": {"exp_avg": torch.arange(4).float(), "step": torch.tensor(3.0)}}, "groups": [1, 2]}
    dcp.save(state, checkpoint_id=path)
    restored = _load_flat_dcp(path)
    assert restored["groups"] == [1, 2]
    assert torch.equal(restored["state.factor.exp_avg"], torch.arange(4).float())
    _assert_close_tree(restored, copy.deepcopy(restored), atol=0, rtol=0, path="optim")
    changed = copy.deepcopy(restored)
    changed["state.factor.exp_avg"][0] += 0.1
    with pytest.raises(AssertionError):
        _assert_close_tree(restored, changed, atol=0, rtol=0, path="optim")
    _assert_close_tree(restored, changed, atol=0.11, rtol=0, path="optim")
    changed["groups"] = [1, 3]
    with pytest.raises(AssertionError, match="scalar/control"):
        _assert_close_tree(restored, changed, atol=0.11, rtol=0, path="optim")


def test_resume_controls_compare_rng_exactly_and_only_exclude_storage_identity():
    original = {
        "optimizer_step": 3,
        "rng": {"numpy": np.arange(5), "torch": torch.tensor([1, 2], dtype=torch.uint8)},
        "validation": {
            "progress": {"pending_step": None, "pending_metrics": {}},
            "publication": {
                "step": 3,
                "archive_required": False,
                "id": "first",
                "destination": {"root": "/a"},
                "manifest": "first.json",
            },
        },
    }
    other = copy.deepcopy(original)
    other["validation"]["publication"].update(id="second", destination={"root": "/b"}, manifest="second.json")
    _assert_close_tree(
        _portable_trainer_controls(original), _portable_trainer_controls(other), atol=0, rtol=0, path="trainer"
    )
    other["rng"]["numpy"][0] = 7
    with pytest.raises(AssertionError, match="array state"):
        _assert_close_tree(
            _portable_trainer_controls(original), _portable_trainer_controls(other), atol=0, rtol=0, path="trainer"
        )
    other["validation"]["progress"]["pending_step"] = 3
    with pytest.raises(ValueError, match="completed validation"):
        _portable_trainer_controls(other)
    assert original["validation"]["publication"]["id"] == "first"


def test_probe_rows_use_actual_video_and_existing_image_without_transcoding(tmp_path):
    from cosmos_framework.posttrain.data.datasets.sft import row_identity

    video, image = tmp_path / "clip.mp4", tmp_path / "frame.png"
    video.write_bytes(b"fixture file identity, decoding tested on GPU integration")
    image.write_bytes(b"fixed image bytes")
    row = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "video", "video": "clip.mp4"}, {"type": "text", "text": "Describe the action."}],
            },
            {"role": "assistant", "content": "A person walks."},
        ]
    }
    path = tmp_path / "validation.jsonl"
    path.write_text(json.dumps(row) + "\n")
    rows = _probe_rows(str(path), 0, str(image), tmp_path)
    assert set(rows) == {"text", "image", "video"}
    assert rows["video"]["messages"][0]["content"][0]["video"] == str(video)
    copied = tmp_path / "copy.png"
    copied.write_bytes(image.read_bytes())
    second = _probe_rows(str(path), 0, str(copied), tmp_path)
    assert row_identity(rows["image"]) == row_identity(second["image"])
    with pytest.raises(ValueError, match="actual local video"):
        _probe_rows(str(path), 1, str(image), tmp_path)


def test_memory_telemetry_is_opt_in_and_does_not_initialize_cuda(tmp_path, monkeypatch):
    monkeypatch.delenv("COSMOS_QLORA_METRICS_FILE", raising=False)
    assert record_qlora_phase("disabled", reset_peak=True) is None
    path = tmp_path / "metrics.jsonl"
    monkeypatch.setenv("COSMOS_QLORA_METRICS_FILE", str(path))
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    record = record_qlora_phase("cpu_loading", reset_peak=True)
    assert record["host_status"]["VmRSS"] > 0
    assert record["host_peak_rss_bytes"] > 0
    assert "cuda" not in record
    assert json.loads(path.read_text()) == record
