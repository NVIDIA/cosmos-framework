# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Alpamayo adapter targets, PhysicalAI observation validation, trajectory metrics and config checks."""

from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.posttrain.model.alpamayo.sft import (
    PHYSICALAI_CAMERAS,
    AlpamayoSFTModelConfig,
    physicalai_frames,
)
from cosmos_framework.posttrain.model.alpamayo.source import backbone_targets
from cosmos_framework.posttrain.model.alpamayo.training import trajectory_metrics
from cosmos_framework.posttrain.model.quantization.family import FamilyBackboneConfig
from cosmos_framework.posttrain.tests.qlora.action_helpers import physicalai_sample

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def test_exact_gradient_reachable_language_allowlist():
    net = SimpleNamespace(
        vlm=SimpleNamespace(model=SimpleNamespace(language_model=SimpleNamespace(layers=[None] * 36)))
    )
    targets = backbone_targets(net)
    assert len(targets) == 247
    assert {name.rsplit(".", 1)[1] for name in targets if ".layers.35." in name} == {"k_proj", "v_proj"}
    assert all(name.startswith("vlm.model.language_model.layers.") for name in targets)
    net.vlm.model.language_model.layers = [None] * 32
    with pytest.raises(ValueError, match="36-layer"):
        backbone_targets(net)


def test_physicalai_camera_time_units_and_calibration_guards():
    row = physicalai_sample()
    assert physicalai_frames(row) == [f"{c}-{i}" for c in PHYSICALAI_CAMERAS for i in range(4)]
    row["observation"]["frame_timestamps_us"][0] = [4800000, 4900000, 5000000, 5100001]
    with pytest.raises(ValueError, match="precede"):
        physicalai_frames(row)
    row = physicalai_sample()
    row["observation"]["coordinate_frame"] = "world"
    with pytest.raises(ValueError, match="vehicle frame"):
        physicalai_frames(row)
    row = physicalai_sample()
    row["observation"]["calibration"]["sensor_extrinsics"][PHYSICALAI_CAMERAS[0]]["qw"] = 0.1
    with pytest.raises(ValueError, match="unit norm"):
        physicalai_frames(row)


def test_openloop_metrics_preserve_meters_horizons_and_example_weighting():
    expected = torch.zeros(2, 64, 3)
    predicted = expected.clone()
    predicted[0, :, 0] = 3
    predicted[0, :, 1] = 4
    predicted[1, :, 2] = 100  # explicitly planar XY, not an unexplained 3D distance
    for metric in trajectory_metrics(predicted, expected).values():
        assert metric == {"sum": 5.0, "count": 2}
    with pytest.raises(ValueError, match="nonfinite"):
        trajectory_metrics(predicted * float("nan"), expected)


def test_native_configuration_rejects_missing_checkpoint_and_no_trainables():
    AlpamayoSFTModelConfig("base", "processor")
    with pytest.raises(ValueError, match="checkpoint_path"):
        AlpamayoSFTModelConfig("", "processor")
    with pytest.raises(ValueError, match="requires train_action_head"):
        AlpamayoSFTModelConfig(
            "base", "processor", backbone_adaptation=FamilyBackboneConfig(mode="nf4_frozen"), train_action_head=False
        )


def test_dense_source_check_rejects_wrong_checkpoint_identity():
    from cosmos_framework.posttrain.model.alpamayo.dense import recorded_base
    from cosmos_framework.posttrain.model.alpamayo.source import BASE_CONFIG_SHA256, BASE_REVISION, BASE_WEIGHTS
    from cosmos_framework.posttrain.utils.identity import canonical_json_sha256

    files = {"config.json": BASE_CONFIG_SHA256, **BASE_WEIGHTS}
    base = {
        "repository": "nvidia/Alpamayo-R1-10B",
        "revision": BASE_REVISION,
        "files": files,
        "sha256": canonical_json_sha256(files, context="Alpamayo snapshot"),
    }
    identity = {"base": {"base": base}}
    assert recorded_base(identity) == base
    base["revision"] = "main"
    with pytest.raises(ValueError, match="supported immutable base"):
        recorded_base(identity)
