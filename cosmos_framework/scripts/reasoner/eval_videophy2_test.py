# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Unit tests for the model-type dispatch helpers in ``eval_videophy2``.

Synthetic-dir tests are pure filesystem checks — no model weights, no hub
traffic. The tests against the real ``videophy2_sft_edge`` HF export are
read-only and auto-skip when the export dir is unavailable; the full
model-load smoke lives in the export-pipeline validation plan
(``SPEC_edge_export_pipeline_ux.md`` §4.5).
"""

import json
import os

import pytest

from cosmos_framework.scripts.reasoner.eval_videophy2 import (
    _EDGE_PROCESSOR_FILES,
    _edge_processor_source,
    _read_model_type,
)

# Existing Edge HF export (model_type="cosmos3_edge", bundled processor files).
_EDGE_EXPORT_DIR = (
    "/lustre/fsw/portfolios/cosmos/projects/cosmos_base_training/users/simonz/work/dev/20260709/"
    "cosmos-framework-private/outputs/renewal2_videophy2/cosmos3/vlm_videophy2_sft/videophy2_sft_edge/"
    "hf_exports/iter_000000050"
)

requires_edge_export = pytest.mark.skipif(
    not os.path.isdir(_EDGE_EXPORT_DIR),
    reason=f"videophy2_sft_edge HF export not available at {_EDGE_EXPORT_DIR}",
)


def test_read_model_type_edge(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "cosmos3_edge"}))
    assert _read_model_type(str(tmp_path)) == "cosmos3_edge"


def test_read_model_type_qwen(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_vl"}))
    assert _read_model_type(str(tmp_path)) == "qwen3_vl"


def test_read_model_type_missing_or_invalid(tmp_path):
    assert _read_model_type(str(tmp_path)) is None  # no config.json
    assert _read_model_type("Qwen/Qwen3-VL-2B-Init") is None  # hub id, not a dir
    (tmp_path / "config.json").write_text("not json")
    assert _read_model_type(str(tmp_path)) is None


def test_edge_processor_source_bundled(tmp_path):
    for name in _EDGE_PROCESSOR_FILES:
        (tmp_path / name).write_text("{}")
    assert _edge_processor_source(str(tmp_path)) == str(tmp_path)


def test_edge_processor_source_incomplete_falls_back(tmp_path):
    # Any one missing processor file must trigger the recipe-snapshot fallback.
    for name in _EDGE_PROCESSOR_FILES[:-1]:
        (tmp_path / name).write_text("{}")
    assert _edge_processor_source(str(tmp_path)) is None


@requires_edge_export
def test_real_edge_export_dispatches_to_edge():
    assert _read_model_type(_EDGE_EXPORT_DIR) == "cosmos3_edge"
    assert _edge_processor_source(_EDGE_EXPORT_DIR) == _EDGE_EXPORT_DIR


@requires_edge_export
def test_real_edge_export_processor_builds():
    # CPU-only, weight-free: tokenizer + sub-processor configs from the export.
    from cosmos_framework.data.generator.processors.cosmos3_edge_processing import build_cosmos3_edge_processor

    processor = build_cosmos3_edge_processor(_EDGE_EXPORT_DIR)
    assert processor.tokenizer is not None
    assert callable(getattr(processor, "batch_decode", None))
    assert processor.image_token_id is not None and processor.video_token_id is not None
