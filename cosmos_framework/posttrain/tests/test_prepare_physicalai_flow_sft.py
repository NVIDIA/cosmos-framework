# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Native four-camera data, immutable preparation and legacy-profile compatibility."""

import copy
import json

import numpy as np
import pytest
import torch

from cosmos_framework.posttrain.model.alpamayo.sft import (
    PHYSICALAI_CAMERAS,
    PHYSICALAI_PROFILE,
    AlpamayoSFTModelConfig,
    physicalai_frames,
)
from cosmos_framework.posttrain.scripts.prepare_physicalai_flow_sft import FORMAT, verify_clip, write_clip


def source():
    sample = {
        "clip_id": "first-clip",
        "t0_us": 5_100_000,
        "image_frames": torch.arange(4 * 4 * 3 * 2 * 4).reshape(4, 4, 3, 2, 4).to(torch.uint8),
        "camera_indices": torch.tensor([0, 1, 2, 6]),
        "absolute_timestamps": torch.tensor([[4_800_000, 4_900_000, 5_000_000, 5_100_000]]).repeat(4, 1),
        "ego_history_xyz": torch.zeros(1, 1, 16, 3),
        "ego_history_rot": torch.eye(3).repeat(1, 1, 16, 1, 1),
        "ego_future_xyz": torch.arange(192).reshape(1, 1, 64, 3).float() / 100,
        "ego_future_rot": torch.eye(3).repeat(1, 1, 64, 1, 1),
    }
    calibration = {
        "camera_intrinsics": {
            camera: {
                "width": 4,
                "height": 2,
                "cx": 1,
                "cy": 1,
                **{f"{side}_poly_{i}": float(i == 1) for side in ("bw", "fw") for i in range(5)},
            }
            for camera in PHYSICALAI_CAMERAS
        },
        "sensor_extrinsics": {
            camera: {"qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}
            for camera in PHYSICALAI_CAMERAS
        },
    }
    identity = {
        "format": FORMAT,
        "profile": PHYSICALAI_PROFILE,
        "clip_id": "first-clip",
        "source_revision": "a" * 40,
        "t0_us": 5_100_000,
    }
    return sample, calibration, identity


def test_native_rgb_poses_calibration_and_idempotent_clip_commit(tmp_path):
    from PIL import Image

    sample, calibration, identity = source()
    destination = tmp_path / "clips/first-clip"
    row, record = write_clip(sample, calibration, destination, identity)
    assert physicalai_frames(row) == [f"clips/first-clip/{c}-{i}.png" for c in PHYSICALAI_CAMERAS for i in range(4)]
    for c, camera in enumerate(PHYSICALAI_CAMERAS):
        for i in range(4):
            with Image.open(destination / f"{camera}-{i}.png") as image:
                np.testing.assert_array_equal(
                    np.asarray(image), sample["image_frames"][c, i].numpy().transpose(1, 2, 0)
                )
    assert row["actions"] == sample["ego_future_xyz"][0, 0].tolist()
    assert row["observation"]["calibration"] == calibration
    assert write_clip(sample, calibration, destination, identity) == (row, record)
    with pytest.raises(ValueError, match="identity differs"):
        verify_clip(destination, {**identity, "source_revision": "b" * 40})
    (destination / "row.json").write_text("{}")
    with pytest.raises(ValueError, match="bytes differ"):
        verify_clip(destination, identity)


def test_interrupted_preparation_preserves_failure_and_restarts_to_complete(tmp_path, monkeypatch):
    from PIL import Image

    sample, calibration, identity = source()
    destination = tmp_path / "clips/first-clip"
    save = Image.Image.save
    monkeypatch.setattr(Image.Image, "save", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("interruption")))
    with pytest.raises(RuntimeError, match="interruption"):
        write_clip(sample, calibration, destination, identity)
    assert not destination.exists()
    partial = list(destination.parent.glob(".first-clip.incomplete-*"))
    assert len(partial) == 1 and not (partial[0] / "complete.json").exists()
    with pytest.raises(ValueError, match="incomplete"):
        verify_clip(partial[0], identity)
    monkeypatch.setattr(Image.Image, "save", save)
    write_clip(sample, calibration, destination, identity)
    assert partial[0].exists() and verify_clip(destination, identity)[1]["format"] == FORMAT
    record = json.loads((destination / "complete.json").read_text())
    record["files"] = {}
    (destination / "complete.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="file inventory"):
        verify_clip(destination, identity)


@pytest.mark.parametrize(
    "fault",
    [
        "camera_order",
        "camera_indices",
        "missing_camera",
        "duplicate_frame",
        "future_frame",
        "negative_frame",
        "decreasing_frame",
        "interval",
        "frame",
        "calibration",
        "nonfinite_calibration",
        "quaternion",
    ],
)
def test_four_camera_contract_refuses_invalid_observation(tmp_path, fault):
    sample, calibration, identity = source()
    row, _ = write_clip(sample, calibration, tmp_path / "clips/first-clip", identity)
    row = copy.deepcopy(row)
    obs = row["observation"]
    if fault == "camera_order":
        obs["camera_order"] = list(reversed(obs["camera_order"]))
    elif fault == "camera_indices":
        obs["camera_indices"][-1] = 3
    elif fault == "missing_camera":
        del row["media"][PHYSICALAI_CAMERAS[-1]]
    elif fault == "duplicate_frame":
        row["media"][PHYSICALAI_CAMERAS[0]][1] = row["media"][PHYSICALAI_CAMERAS[0]][0]
    elif fault == "future_frame":
        obs["frame_timestamps_us"][0][-1] += 1
    elif fault == "negative_frame":
        obs["frame_timestamps_us"][0][0] = -1
    elif fault == "decreasing_frame":
        obs["frame_timestamps_us"][0][1] = obs["frame_timestamps_us"][0][0] - 1
    elif fault == "interval":
        obs["trajectory_dt_seconds"] = 0.05
    elif fault == "frame":
        obs["coordinate_frame"] = "global"
    elif fault == "calibration":
        del obs["calibration"]["camera_intrinsics"][PHYSICALAI_CAMERAS[0]]
    elif fault == "nonfinite_calibration":
        obs["calibration"]["camera_intrinsics"][PHYSICALAI_CAMERAS[0]]["cx"] = float("nan")
    elif fault == "quaternion":
        obs["calibration"]["sensor_extrinsics"][PHYSICALAI_CAMERAS[0]]["qw"] = 0
    with pytest.raises(ValueError):
        physicalai_frames(row)


def test_observation_profile_identity_roundtrip_preserves_legacy_default():
    assert AlpamayoSFTModelConfig("processor").observation_profile == "front_frames_v1"
    config = AlpamayoSFTModelConfig("processor", observation_profile=PHYSICALAI_PROFILE)
    assert config.observation_profile == PHYSICALAI_PROFILE
    with pytest.raises(ValueError, match="observation_profile"):
        AlpamayoSFTModelConfig("processor", observation_profile="invented")


def test_native_preceding_selection_may_repeat_a_frame_after_a_source_gap(tmp_path):
    sample, calibration, identity = source()
    sample["absolute_timestamps"][0, 1] = sample["absolute_timestamps"][0, 0]
    sample["absolute_timestamps"][0, 2] = sample["absolute_timestamps"][0, 0]
    sample["image_frames"][0, 1:3] = sample["image_frames"][0, :1]
    row, record = write_clip(sample, calibration, tmp_path / "clips/first-clip", identity)
    assert row["observation"]["frame_timestamps_us"][0][:3] == [4_800_000] * 3
    assert len(physicalai_frames(row)) == 16
    assert write_clip(sample, calibration, tmp_path / "clips/first-clip", identity) == (row, record)
