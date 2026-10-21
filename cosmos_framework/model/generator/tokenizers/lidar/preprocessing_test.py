# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Round-trip the actual WSM writer through Framework and the standalone cookbook."""

import importlib.util
import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import load_file

from cosmos_framework.model.generator.tokenizers.lidar.preprocessing import (
    load_lidar_frames,
    load_prepared_lidar_video_3ch,
    load_prepared_range_video,
)
from cosmos_framework.model.generator.tokenizers.lidar.range_projection import (
    V0_TRANSFER_RANGE_PROJECTION,
    V1_TRANSFER_RANGE_PROJECTION,
    V1P2_TRANSFER_RANGE_PROJECTION,
)

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


@pytest.fixture
def exporters(monkeypatch):
    auto = Path(__file__).resolve().parents[5] / ".agents/skills/cookbooks/cosmos3/simulation/auto"
    monkeypatch.syspath_prepend(str(auto / "wsm"))
    from cosmos_auto.lidar import rangemap
    from cosmos_auto.lidar.control import save_control

    spec = importlib.util.spec_from_file_location("cookbook_lidar_rangemap", auto / "helpers/lidar_rangemap.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return rangemap, save_control, helper


def read_control(path, **kwargs):
    return load_prepared_lidar_video_3ch(
        path,
        min_source_frame_id=0,
        max_source_frame_id=6,
        expected_clip_key="scene",
        with_intensity=False,
        range_projection=V1_TRANSFER_RANGE_PROJECTION,
        **kwargs,
    )


def replace_metadata(path, metadata, *, name="scene.lidar_metadata.json", copies=1):
    replacement = path.with_suffix(".rewritten.tar")
    with tarfile.open(path) as source, tarfile.open(replacement, "w") as target:
        for member in source.getmembers():
            if not member.name.endswith(".lidar_metadata.json"):
                target.addfile(member, source.extractfile(member))
        if metadata is not None:
            payload = json.dumps(metadata).encode()
            for _ in range(copies):
                member = tarfile.TarInfo(name)
                member.size = len(payload)
                target.addfile(member, io.BytesIO(payload))
    replacement.replace(path)


@pytest.mark.parametrize("width", [1800, 3600])
@pytest.mark.parametrize("as_bytes", [False, True])
def test_tar_matches_direct_and_cookbook_exports(tmp_path, exporters, width, as_bytes):
    rangemap, save_control, cookbook = exporters
    ranges = np.zeros((3, 128, width), dtype=np.float32)
    ranges[0, 64, width // 2] = 25.0
    ranges[1, 10, 0] = 12.0
    if width == 3600:
        ranges[0, 64, width // 2 + 1] = 30.0
    path = tmp_path / "control.tar"
    rangemap.save_range_maps_to_tar(ranges, [0, 3, 6], "scene", path)
    with tarfile.open(path) as archive:
        assert json.load(archive.extractfile("scene.lidar_metadata.json")) == {
            "format_version": 2,
            "n_rows": 128,
            "n_columns": width,
        }
    restored = rangemap.load_range_maps_from_tar(path, 128, width)
    np.testing.assert_array_equal(restored["000000"][0], ranges[0])
    with pytest.raises(ValueError, match="metadata"):
        rangemap.load_range_maps_from_tar(path, 128, 5400 - width)

    source = path.read_bytes() if as_bytes else path
    actual, frame_ids = read_control(source, max_frames_per_chunk=1)
    assert frame_ids == [0, 3, 6]
    assert actual.shape == (3, 3, 128, 1800)
    assert actual[0, 0, 64, 900].item() == 25.0
    assert actual[0, 0, 64, 450].item() == 0.0
    assert not actual[:, 2].any()
    direct, converted = tmp_path / "direct.safetensors", tmp_path / "cookbook.safetensors"
    save_control(ranges, frame_ids, direct, "scene")
    cookbook.convert_control(path, converted)
    for output in (direct, converted):
        np.testing.assert_array_equal(actual.numpy(), load_file(output)["frames"])

    padded, _ = load_prepared_lidar_video_3ch(
        source,
        min_source_frame_id=0,
        max_source_frame_id=6,
        with_intensity=False,
        range_projection=V1P2_TRANSFER_RANGE_PROJECTION,
        prepend_first_frame=True,
        max_frames_per_chunk=1,
    )
    assert padded.shape == (3, 4, 128, 1808)
    np.testing.assert_array_equal(padded[:, 0, :, 4:-4], actual[:, 0])
    np.testing.assert_array_equal(padded[:, 1:, :, 4:-4], actual)
    assert V1_TRANSFER_RANGE_PROJECTION.native_width == 3600  # No mutation of model config.


@pytest.mark.parametrize("width", [1800, 3600])
def test_intensity_follows_nearest_return(tmp_path, exporters, width):
    rangemap, _, _ = exporters
    ranges = np.zeros((1, 128, width), dtype=np.float32)
    intensities = np.zeros_like(ranges)
    ranges[0, 64, width // 2] = 25.0
    intensities[0, 64, width // 2] = 0.25
    if width == 3600:
        ranges[0, 64, width // 2 + 1] = 30.0
        intensities[0, 64, width // 2 + 1] = 0.75
    path = tmp_path / "measured.tar"
    rangemap.save_range_maps_to_tar(ranges, [0], "scene", path, intensities)
    actual, _ = load_prepared_lidar_video_3ch(
        path,
        min_source_frame_id=0,
        max_source_frame_id=0,
        range_projection=V1_TRANSFER_RANGE_PROJECTION,
    )
    np.testing.assert_array_equal(actual[:, 0, 64, 900], [25.0, 0.25, 1.0])


def test_unversioned_3600_archive_remains_compatible(tmp_path, exporters):
    rangemap, _, cookbook = exporters
    ranges = np.zeros((1, 128, 3600), dtype=np.float32)
    ranges[0, 64, 1800] = 25.0
    path = tmp_path / "legacy.tar"
    rangemap.save_range_maps_to_tar(ranges, [0], "scene", path)
    expected, _ = read_control(path)
    replace_metadata(path, None)
    actual, _ = read_control(path)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(load_lidar_frames(path), ranges)
    legacy, _, ids = load_prepared_range_video(path, min_source_frame_id=0, max_source_frame_id=0)
    assert legacy.shape == (1, 1, 128, 1024)
    assert ids == [0]
    output = tmp_path / "legacy.safetensors"
    cookbook.convert_control(path, output)
    np.testing.assert_array_equal(actual, load_file(output)["frames"])


@pytest.mark.parametrize(
    "metadata",
    [
        [],
        {},
        {"format_version": 3, "n_rows": 128, "n_columns": 1800},
        {"format_version": 2, "n_rows": 64, "n_columns": 1800},
        {"format_version": 2, "n_rows": 128, "n_columns": "1800"},
        {"format_version": 2, "n_rows": 128, "n_columns": 900},
    ],
)
def test_invalid_metadata_is_not_treated_as_legacy(tmp_path, exporters, metadata):
    rangemap, _, cookbook = exporters
    path = tmp_path / "bad.tar"
    rangemap.save_range_maps_to_tar(np.ones((1, 128, 1800)), [0], "scene", path)
    replace_metadata(path, metadata)
    for reader in (lambda: read_control(path), lambda: cookbook.convert_control(path, tmp_path / "bad.safetensors")):
        with pytest.raises(ValueError, match="metadata"):
            reader()


@pytest.mark.parametrize("name,copies", [("other.lidar_metadata.json", 1), ("scene.lidar_metadata.json", 2)])
def test_metadata_must_uniquely_match_clip(tmp_path, exporters, name, copies):
    rangemap, _, cookbook = exporters
    path = tmp_path / "bad.tar"
    rangemap.save_range_maps_to_tar(np.ones((1, 128, 1800)), [0], "scene", path)
    replace_metadata(path, {"format_version": 2, "n_rows": 128, "n_columns": 1800}, name=name, copies=copies)
    for reader in (lambda: read_control(path), lambda: cookbook.convert_control(path, tmp_path / "bad.safetensors")):
        with pytest.raises(ValueError, match="metadata"):
            reader()


def test_coordinates_must_fit_declared_width(tmp_path, exporters):
    rangemap, _, cookbook = exporters
    path = tmp_path / "bad.tar"
    ranges = np.zeros((1, 128, 3600))
    ranges[0, 64, 2700] = 25.0
    rangemap.save_range_maps_to_tar(ranges, [0], "scene", path)
    replace_metadata(path, {"format_version": 2, "n_rows": 128, "n_columns": 1800})
    with pytest.raises(ValueError, match="columns"):
        read_control(path)
    with pytest.raises(ValueError, match="coordinates"):
        cookbook.convert_control(path, tmp_path / "bad.safetensors")


def test_1800_archive_rejects_incompatible_legacy_paths(tmp_path, exporters):
    rangemap, _, _ = exporters
    path = tmp_path / "control.tar"
    rangemap.save_range_maps_to_tar(np.ones((1, 128, 1800)), [0], "scene", path)
    with pytest.raises(ValueError, match="legacy reader"):
        load_lidar_frames(path)
    with pytest.raises(ValueError, match="legacy reader"):
        load_prepared_range_video(path, min_source_frame_id=0, max_source_frame_id=0)
    with pytest.raises(ValueError, match="divisor"):
        load_prepared_lidar_video_3ch(
            path,
            min_source_frame_id=0,
            max_source_frame_id=0,
            range_projection=V0_TRANSFER_RANGE_PROJECTION,
        )
