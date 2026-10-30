# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Known physical motions, frame timing and drive separation for public KITTI."""

import json

import numpy as np
import pytest
from PIL import Image
from scipy.spatial.transform import Rotation

from cosmos_framework.posttrain.scripts.prepare_kitti_flow_sft import aligned_window, oxts_poses, prepare


def test_oxts_projection_rotation_and_relative_pose():
    latitude = 49.0
    meters_per_degree = np.cos(np.deg2rad(latitude)) * 6378137 * np.pi / 180
    packets = np.array([[latitude, 8 + x / meters_per_degree, 100, 0, 0, 0] for x in [0, 1, 2]])
    xyz, rot = oxts_poses(packets)
    np.testing.assert_allclose(xyz, [[0, 0, 0], [1, 0, 0], [2, 0, 0]], atol=1e-8)
    np.testing.assert_allclose(rot, np.broadcast_to(np.eye(3), (3, 3, 3)), atol=1e-14)
    # A vehicle pointing north moves along global y; its local forward remains x.
    times = np.arange(100) * 0.1
    positions = np.stack((np.zeros(100), times * 5, np.zeros(100)), axis=-1)
    rotations = Rotation.from_euler("z", np.full((100, 1), np.pi / 2)).as_matrix()
    local, local_rot = aligned_window(times, positions, rotations, current_time=2.0)
    np.testing.assert_allclose(local[:, 0], np.arange(-15, 65) * 0.5, atol=1e-12)
    np.testing.assert_allclose(local[:, 1:], 0, atol=1e-12)
    np.testing.assert_allclose(local_rot, np.broadcast_to(np.eye(3), (80, 3, 3)), atol=1e-14)
    with pytest.raises(ValueError, match="without extrapolation"):
        aligned_window(times, positions, rotations, current_time=1.0)


def write_drive(root, name, color):
    directory = root / name
    for sub in ["image_02", "oxts"]:
        (directory / sub / "data").mkdir(parents=True)
    times = np.datetime64("2011-09-26T13:00:00", "ns") + np.arange(112) * np.timedelta64(100, "ms")
    (directory / "image_02/timestamps.txt").write_text("\n".join(str(x) for x in times + np.timedelta64(1, "ms")))
    (directory / "oxts/timestamps.txt").write_text("\n".join(str(x) for x in times))
    scale = np.cos(np.deg2rad(49)) * 6378137 * np.pi / 180
    for i in range(112):
        Image.new("RGB", (8, 8), (color, i, 0)).save(directory / f"image_02/data/{i:010d}.png")
        np.savetxt(directory / f"oxts/data/{i:010d}.txt", np.array([49, 8 + i * 0.5 / scale, 100, 0, 0, 0])[None])


def test_preparation_keeps_real_history_and_disjoint_drives(tmp_path):
    write_drive(tmp_path / "raw", "drive_a", 30)
    write_drive(tmp_path / "raw", "drive_b", 80)
    result = prepare(tmp_path / "raw", tmp_path / "out", training_drive="drive_a", validation_drive="drive_b")
    assert result["counts"]["train"] >= 8
    rows = [json.loads(x) for x in (tmp_path / "out/train.jsonl").read_text().splitlines()]
    row = rows[0]
    np.testing.assert_allclose(np.array(row["actions"])[:, 0], np.arange(1, 65) * 0.5, atol=1e-7)
    np.testing.assert_allclose(row["observation"]["ego_history_xyz"][-1], [0, 0, 0], atol=0)
    assert len(row["media"]["front"]) == 4
    assert all((tmp_path / "out" / n).is_file() for n in row["media"]["front"])
    assert result["maximum_camera_alignment_seconds"]["train"] < 1e-12
    with pytest.raises(FileExistsError):
        prepare(tmp_path / "raw", tmp_path / "out", training_drive="drive_a", validation_drive="drive_b")
    with pytest.raises(ValueError, match="different drives"):
        prepare(tmp_path / "raw", tmp_path / "bad", training_drive="drive_a", validation_drive="drive_a")
    (tmp_path / "raw/drive_a/image_02/data/0000000001.png").unlink()
    with pytest.raises(ValueError, match="inventory"):
        prepare(tmp_path / "raw", tmp_path / "incomplete", training_drive="drive_a", validation_drive="drive_b")
