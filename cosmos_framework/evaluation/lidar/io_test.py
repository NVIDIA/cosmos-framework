# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Exercise metric range-view preparation and precise reference alignment."""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from cosmos_framework.evaluation.lidar.io import prepare_framework_outputs, read_rangeview

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    ranges = np.full((2, 2, 4), 10.0, dtype=np.float32)  # [T,H,W]
    save_file(
        {
            "range_m": ranges,
            "intensity": np.full_like(ranges, 0.25).astype(np.float16),
            "frame_ids": np.array([10, 13], dtype=np.int64),
        },
        tmp_path / "generated.safetensors",
    )
    reference = np.full((3, 2, 8), 10.0, dtype=np.float32)  # [T_gt,H,W_native]
    # Adjacent native columns have a farther return; pooling must keep the near one and its intensity.
    reference[..., 1::2] = 12.0  # [T_gt,H,W_native/2]
    intensity = np.full_like(reference, 0.25)  # [T_gt,H,W_native]
    intensity[..., 1::2] = 0.9  # [T_gt,H,W_native/2]
    np.savez(
        tmp_path / "reference.npz",
        range_m=reference,
        valid=np.ones_like(reference, dtype=bool),
        intensity=intensity,
        sweep_ids=np.array([7, 10, 13]),
    )
    poses = np.repeat(np.eye(4)[None], 2, axis=0)  # [T,4,4]
    poses[1, 0, 3] = 1.0
    np.save(tmp_path / "poses.npy", poses)
    manifest = tmp_path / "input.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "clip_id": "clip",
                    "generated": "generated.safetensors",
                    "reference": "reference.npz",
                    "lidar_to_world": "poses.npy",
                }
            ]
        )
    )
    projection = tmp_path / "projection.json"
    projection.write_text(json.dumps({"elevation_degrees": [0, 0], "semantic_width": 4, "model_width": 4}))
    return manifest, projection


def test_prepare_native_artifact_pool_select_and_cloud_metrics(tmp_path: Path) -> None:
    manifest, projection = _fixture(tmp_path)
    output = tmp_path / "prepared"
    result = prepare_framework_outputs(manifest, output, projection, True, True)
    assert result["frames"] == 2
    xyzi = np.load(output / "clip/reference_xyzi/frame_000000.npy")  # [N,4]
    np.testing.assert_allclose(xyzi[:, 3], 0.25)
    np.testing.assert_allclose(np.linalg.norm(xyzi[:, :3], axis=1), 10)
    clouds = json.loads((output / "clouds.json").read_text())
    assert clouds[0]["frame_ids"] == [10, 13]
    assert Path(clouds[0]["lidar_to_world"]).samefile(tmp_path / "poses.npy")
    with np.load(output / "clip/reference.npz") as reference:
        np.testing.assert_array_equal(reference["sweep_ids"], [10, 13])


def test_source_ids_not_matched_by_position(tmp_path: Path) -> None:
    manifest, projection = _fixture(tmp_path)
    with pytest.raises(ValueError, match="source frame IDs"):
        prepare_framework_outputs(manifest, tmp_path / "rejected", projection, False, True)


def test_sparse_tar_projection_requires_explicit_option(tmp_path: Path) -> None:
    manifest, projection = _fixture(tmp_path)
    entries = json.loads(manifest.read_text())
    entries[0]["reference"] = "reference.tar"
    manifest.write_text(json.dumps(entries))
    with pytest.raises(ValueError, match="explicitly pass --pool-reference"):
        prepare_framework_outputs(manifest, tmp_path / "rejected", projection)


def test_saved_generator_is_explicitly_masked_and_no_intensity_fabrication(tmp_path: Path) -> None:
    ranges = np.array([[[10.0, 0.0]]], dtype=np.float32)  # [T,H,W]
    save_file({"range_m": ranges, "frame_ids": np.array([0])}, tmp_path / "range.safetensors")
    values = read_rangeview(tmp_path / "range.safetensors", {})
    assert values["valid"].tolist() == [[[True, False]]]
    assert "intensity" not in values


def test_framework_sparse_tar_pooling_preserves_measured_intensity(tmp_path: Path) -> None:
    with tarfile.open(tmp_path / "recording.tar", "w") as archive:
        for frame, distance in ((0, 25.0), (3, 30.0)):
            for kind, value in (("row", 64), ("col", 900), ("range", distance), ("intensity", 0.25)):
                buffer = io.BytesIO()
                np.savez(buffer, np.array([value]))
                payload = buffer.getvalue()
                member = tarfile.TarInfo(f"scene.{frame:06d}.lidar_{kind}.npz")
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))
    ranges = np.full((2, 128, 16), 25, dtype=np.float32)  # [T,H,W]
    save_file({"range_m": ranges, "frame_ids": np.array([0, 3], dtype=np.int64)}, tmp_path / "gen.safetensors")
    manifest = tmp_path / "pairs.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "clip_id": "clip",
                    "generated": "gen.safetensors",
                    "reference": "recording.tar",
                }
            ]
        )
    )
    prepare_framework_outputs(manifest, tmp_path / "prepared", pool_reference=True)
    for frame, distance in ((0, 25), (1, 30)):
        xyzi = np.load(tmp_path / f"prepared/clip/reference_xyzi/frame_{frame:06d}.npy")  # [N,4]
        assert xyzi.shape == (1, 4)
        np.testing.assert_allclose(np.linalg.norm(xyzi[:, :3], axis=1), distance, atol=1e-5)
        np.testing.assert_allclose(xyzi[:, 3], 0.25)
