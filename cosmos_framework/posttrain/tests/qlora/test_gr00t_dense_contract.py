# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""GR00T dense export task identity and sidecars."""

import json
from pathlib import Path

import pytest

from cosmos_framework.posttrain.configs.gr00t_n1d7_libero_qlora import make_config
from cosmos_framework.posttrain.model.gr00t.adaptation import ISAAC_GR00T_REVISION, task_identity
from cosmos_framework.posttrain.model.gr00t.dense import sidecars
from cosmos_framework.posttrain.model.gr00t.qlora_source import verify_metadata

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def write_metadata(root):
    native, backbone = root / "native", root / "backbone"
    native.mkdir(parents=True)
    backbone.mkdir()
    for name, content in {
        "config.json": {"model_type": "Gr00tN1d7", "select_layer": 16},
        "processor_config.json": {"processor_kwargs": {"model_name": "immutable-backbone"}},
        "statistics.json": {"libero_sim": {"action": "fixture"}},
        "embodiment_id.json": {"libero_sim": 24},
    }.items():
        (native / name).write_text(json.dumps(content))
    (backbone / "config.json").write_text(json.dumps({"model_type": "qwen3_vl"}))
    (backbone / "tokenizer_config.json").write_text(json.dumps({"padding_side": "left"}))
    return native, backbone


def fixture_pins(monkeypatch, native, backbone):
    from cosmos_framework.posttrain.model.gr00t import qlora_source
    from cosmos_framework.posttrain.model.gr00t.source import metadata_files

    monkeypatch.setattr(qlora_source, "BASE_FILES", metadata_files(native, gr00t=True))
    monkeypatch.setattr(qlora_source, "BACKBONE_FILES", metadata_files(backbone))


def test_dense_sidecars_require_no_source_weights_and_exclude_weight_payloads(tmp_path, monkeypatch):
    native, backbone = write_metadata(tmp_path)
    fixture_pins(monkeypatch, native, backbone)
    (native / "model.safetensors").write_bytes(b"this must never be opened")
    (backbone / "model.safetensors").write_bytes(b"nor this")
    config = make_config()
    config.model.checkpoint_path = str(native)
    config.model.backbone_adaptation.backbone_path = str(backbone)
    mapping = sidecars(config)
    assert "native/statistics.json" in mapping
    assert "native/embodiment_id.json" in mapping
    assert "backbone/tokenizer_config.json" in mapping
    assert all("safetensors" not in name for name in mapping)
    assert all(Path(path).is_file() for path in mapping.values())


def test_dense_task_identity_binds_stats_tokenizer_and_ignores_old_machine_paths(tmp_path, monkeypatch):
    native, backbone = write_metadata(tmp_path)
    fixture_pins(monkeypatch, native, backbone)
    config = make_config().model

    def _task_identity(config, root):
        verify_metadata(root / "native", root / "backbone")
        return task_identity(config, root / "native", root / "backbone")

    # This proves identity reconstruction cannot require opening a former source checkpoint.
    config.checkpoint_path = "/does/not/exist/old/base"
    config.backbone_adaptation.backbone_path = "/does/not/exist/old/tokenizer"
    identity = _task_identity(config, tmp_path)
    assert "checkpoint_path" not in identity["config"]
    # Machine paths and training-only seed/dropout are not model-defining.
    assert not {"backbone_path", "seed", "dropout"} & set(identity["config"]["backbone_adaptation"])
    # Only the pinned upstream revision, not source digests, process TF32 flags or library versions.
    assert identity["external"] == {"revision": ISAAC_GR00T_REVISION}
    assert not {"matmul_allow_tf32", "cudnn_allow_tf32"} & set(identity["precision"])
    # Neither the supervised prefix (training contract) nor the rollout horizon defines the model:
    # a policy trained with execution_horizon=16 still runs the H8 LIBERO rollout.
    assert not {"training_prefix", "execution_horizon"} & set(identity["config"])
    config.backbone_adaptation.seed += 1
    config.backbone_adaptation.dropout = 0.1
    config.training_prefix, config.execution_horizon = 16, 16
    assert _task_identity(config, tmp_path) == identity
    assert identity["statistics"] == {"libero_sim": {"action": "fixture"}}
    assert identity["embodiment_id"] == {"libero_sim": 24}
    original = identity["backbone_metadata"]
    (backbone / "tokenizer_config.json").write_text(json.dumps({"padding_side": "right"}))
    with pytest.raises(ValueError, match="backbone"):
        _task_identity(config, tmp_path)
    (backbone / "tokenizer_config.json").write_text(json.dumps({"padding_side": "left"}))
    assert _task_identity(config, tmp_path)["backbone_metadata"] == original
    (native / "statistics.json").unlink()
    with pytest.raises(FileNotFoundError):
        _task_identity(config, tmp_path)


def test_recorded_source_cannot_record_different_weight_payload(monkeypatch, tmp_path):
    from cosmos_framework.posttrain.model.gr00t import qlora_source, source

    native, backbone = write_metadata(tmp_path)
    fixture_pins(monkeypatch, native, backbone)
    qlora_source.BASE_FILES["model.safetensors"] = "a" * 64
    monkeypatch.setattr(source, "weight_files", lambda path: {"model.safetensors": "a" * 64})
    identity = qlora_source.pinned_source_identity(native, backbone)
    assert identity["base"]["revision"] == "2ea293aa20ba7cf5bbf3ba17a5fbcb1a01cbfe21"
    monkeypatch.setattr(source, "weight_files", lambda path: {"model.safetensors": "b" * 64})
    with pytest.raises(ValueError, match="weights differ"):
        qlora_source.pinned_source_identity(native, backbone)
