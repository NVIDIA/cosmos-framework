# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Tests for inference_prompts_to_json (dense prompt -> structured-JSON prompt)."""

import json

import pytest

from cosmos_framework.scripts import inference_prompts_to_json as mod


def _build_val(tmp_path, variants=("inference_prompt", "inference_prompt_i2v")):
    val = tmp_path / "val"
    cap = val / "captions" / "ep0"
    cap.mkdir(parents=True)
    caption_json = {"background_setting": "kitchen", "temporal_caption": "An arm.", "fps": 5}
    (cap / "caption.json").write_text(json.dumps(caption_json))
    for v in variants:
        d = val / v
        d.mkdir(parents=True)
        rec = {"name": f"{v}/ep0", "prompt": "OLD DENSE PROMPT", "resolution": "256", "fps": 5}
        if v != "inference_prompt":
            rec["vision_path"] = "../images/ep0.jpg"
        (d / "ep0.json").write_text(json.dumps(rec))
    return val, caption_json


def test_replaces_prompt_with_structured_json_preserving_fields(tmp_path):
    val, caption_json = _build_val(tmp_path)
    mod.main(val_dir=val)

    for v in ("inference_prompt", "inference_prompt_i2v"):
        rec = json.loads((val / v / "ep0.json").read_text())
        assert json.loads(rec["prompt"]) == caption_json  # prompt is now the serialized JSON
        assert rec["name"] == f"{v}/ep0"  # preserved
        assert rec["resolution"] == "256" and rec["fps"] == 5  # preserved
    # i2v keeps its vision_path
    assert json.loads((val / "inference_prompt_i2v" / "ep0.json").read_text())["vision_path"] == "../images/ep0.jpg"


def test_dry_run_does_not_modify(tmp_path):
    val, _ = _build_val(tmp_path, variants=("inference_prompt",))
    before = (val / "inference_prompt" / "ep0.json").read_text()
    mod.main(val_dir=val, dry_run=True)
    assert (val / "inference_prompt" / "ep0.json").read_text() == before


def test_missing_caption_json_is_skipped(tmp_path):
    val, _ = _build_val(tmp_path, variants=("inference_prompt",))
    # Add a second prompt with no matching caption.json.
    rec = {"name": "inference_prompt/ep_missing", "prompt": "DENSE"}
    (val / "inference_prompt" / "ep_missing.json").write_text(json.dumps(rec))
    mod.main(val_dir=val)
    # ep_missing is untouched (still dense), ep0 is updated.
    assert json.loads((val / "inference_prompt" / "ep_missing.json").read_text())["prompt"] == "DENSE"
    assert (val / "inference_prompt" / "ep0.json").read_text() != json.dumps({"prompt": "DENSE"})


def test_errors_when_no_prompt_dirs(tmp_path):
    (tmp_path / "val").mkdir()
    with pytest.raises(SystemExit):
        mod.main(val_dir=tmp_path / "val")
