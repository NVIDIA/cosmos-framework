# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import pytest

from cosmos_framework.scripts.multiview_collage import (
    _MADS_11VIEW_CAMERA_CELLS,
    MAX_GIF_BYTES,
    _parse_ffprobe_size,
    cell_xy,
    choose_camera_grid,
    choose_vehicle_cells,
    collect_views,
    find_sample_dirs,
    gif_target_width,
    tile_filter,
    tile_size,
)

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def test_grid_near_4x3() -> None:
    expected = {
        1: (1, 1),
        2: (2, 1),
        3: (2, 2),
        4: (2, 2),
        5: (3, 2),
        6: (3, 2),
        7: (3, 3),
        8: (3, 3),
        9: (3, 3),
        10: (4, 3),
        11: (4, 3),
    }
    for n, grid in expected.items():
        assert choose_camera_grid(n) == grid


def test_mads_cameras_use_vehicle_spatial_cells() -> None:
    cameras = [
        "camera_left_fisheye_200fov",
        "camera_front_fisheye_200fov",
        "camera_front_wide_120fov",
        "camera_right_fisheye_200fov",
        "camera_cross_left_120fov",
        "camera_front_tele_30fov",
        "camera_cross_right_120fov",
        "camera_rear_left_70fov",
        "camera_rear_right_70fov",
        "camera_rear_fisheye_200fov",
        "camera_rear_tele_30fov",
    ]
    rows, cols, cells = choose_vehicle_cells(cameras)
    assert (rows, cols) == (4, 4)
    assert cells == _MADS_11VIEW_CAMERA_CELLS
    assert cells["camera_left_fisheye_200fov"] == (0, 0.5)
    assert cells["camera_front_fisheye_200fov"] == (0, 1.5)
    assert cells["camera_right_fisheye_200fov"] == (0, 2.5)
    assert cells["camera_cross_left_120fov"] == (1, 0)
    assert cells["camera_front_wide_120fov"] == (1, 1)
    assert cells["camera_front_tele_30fov"] == (1, 2)
    assert cells["camera_cross_right_120fov"] == (1, 3)
    assert cells["camera_rear_left_70fov"] == (2, 0.5)
    assert cells["camera_rear_tele_30fov"] == (2, 1.5)
    assert cells["camera_rear_right_70fov"] == (2, 2.5)
    assert cells["camera_rear_fisheye_200fov"] == (3, 1.5)
    assert len(set(cells.values())) == len(cameras)
    origins = {cell_xy(row, col, 320, 180) for row, col in cells.values()}
    assert origins == {
        (160, 0),
        (480, 0),
        (800, 0),
        (0, 180),
        (320, 180),
        (640, 180),
        (960, 180),
        (160, 360),
        (480, 360),
        (800, 360),
        (480, 540),
    }


def test_seven_mads_cameras_pack_tight_3x3() -> None:
    cameras = [
        "camera_front_wide_120fov",
        "camera_cross_right_120fov",
        "camera_rear_right_70fov",
        "camera_rear_tele_30fov",
        "camera_rear_left_70fov",
        "camera_cross_left_120fov",
        "camera_front_tele_30fov",
    ]
    rows, cols, cells = choose_vehicle_cells(cameras)
    assert (rows, cols) == (3, 3)
    assert cells == {
        "camera_front_wide_120fov": (0, 1),
        "camera_cross_left_120fov": (1, 0),
        "camera_front_tele_30fov": (1, 1),
        "camera_cross_right_120fov": (1, 2),
        "camera_rear_left_70fov": (2, 0),
        "camera_rear_tele_30fov": (2, 1),
        "camera_rear_right_70fov": (2, 2),
    }


def test_mads_subsets_drop_missing_camera_tiles() -> None:
    one = choose_vehicle_cells(["camera_front_wide_120fov"])
    assert one[:2] == (1, 1)
    assert one[2] == {"camera_front_wide_120fov": (0, 0)}

    four = choose_vehicle_cells(
        [
            "camera_front_wide_120fov",
            "camera_cross_right_120fov",
            "camera_rear_tele_30fov",
            "camera_cross_left_120fov",
        ]
    )
    assert four[:2] == (2, 3)
    assert four[2]["camera_cross_left_120fov"] == (0, 0)
    assert four[2]["camera_front_wide_120fov"] == (0, 1)
    assert four[2]["camera_cross_right_120fov"] == (0, 2)
    assert four[2]["camera_rear_tele_30fov"] == (1, 1)

    eight = choose_vehicle_cells(
        [
            "camera_front_wide_120fov",
            "camera_cross_right_120fov",
            "camera_rear_right_70fov",
            "camera_rear_tele_30fov",
            "camera_rear_left_70fov",
            "camera_cross_left_120fov",
            "camera_front_tele_30fov",
            "camera_front_fisheye_200fov",
        ]
    )
    assert eight[:2] == (3, 3)
    assert eight[2]["camera_front_fisheye_200fov"] == (0, 0)
    assert eight[2]["camera_front_wide_120fov"] == (0, 1)
    assert eight[2]["camera_rear_tele_30fov"] == (2, 1)

    ten = [
        "camera_left_fisheye_200fov",
        "camera_front_fisheye_200fov",
        "camera_front_wide_120fov",
        "camera_right_fisheye_200fov",
        "camera_cross_left_120fov",
        "camera_front_tele_30fov",
        "camera_cross_right_120fov",
        "camera_rear_left_70fov",
        "camera_rear_right_70fov",
        "camera_rear_tele_30fov",
    ]
    ten_layout = choose_vehicle_cells(ten)
    assert ten_layout[:2] == (3, 4)
    assert len(set(ten_layout[2].values())) == 10

    for rows, cols, cells in (one, four, eight, ten_layout):
        occupied = set(cells.values())
        for row in range(rows):
            assert any(r == row for r, _ in occupied)
        for col in range(cols):
            assert any(c == col for _, c in occupied)
        assert (rows, cols) != (4, 4)


def test_unknown_cameras_pack_near_4x3() -> None:
    rows, cols, cells = choose_vehicle_cells(["cam_a", "cam_b", "cam_c"])
    assert (rows, cols) == (2, 2)
    assert cells == {"cam_a": (0, 0), "cam_b": (0, 1), "cam_c": (1, 0)}


def test_gif_target_width_keeps_full_size_when_under_budget() -> None:
    assert gif_target_width(1280, MAX_GIF_BYTES - 1, MAX_GIF_BYTES) == 1280


def test_gif_target_width_scales_area_when_over_budget() -> None:
    width = gif_target_width(1280, 40 * 1024 * 1024, MAX_GIF_BYTES)
    assert 160 <= width < 1280
    assert width % 2 == 0


def test_tile_size_matches_first_view_aspect() -> None:
    assert tile_size(320, 1920, 1080) == (320, 180)
    assert tile_size(320, 1000, 1000) == (320, 320)
    fit = tile_filter(320, 180)
    assert "force_original_aspect_ratio=decrease" in fit
    assert "pad=320:180" in fit


def test_parse_ffprobe_size_reads_the_first_video_stream() -> None:
    assert _parse_ffprobe_size("1920,1080\n") == (1920, 1080)
    assert _parse_ffprobe_size("1920,1080\n640,480\n") == (1920, 1080)


def test_parse_ffprobe_size_returns_none_so_video_size_can_fall_back() -> None:
    # ffprobe exits 0 with empty stdout on a file with no video stream; the caller then uses ffmpeg,
    # whose error names the file.
    assert _parse_ffprobe_size("") is None
    assert _parse_ffprobe_size("N/A,N/A\n") is None
    assert _parse_ffprobe_size("1920\n") is None


def test_find_sample_dirs(tmp_path) -> None:
    sample = tmp_path / "clip" / "0"
    sample.mkdir(parents=True)
    (sample / "vision_view00_camera_front.mp4").write_bytes(b"x")
    (sample / "vision_view01_camera_left.mp4").write_bytes(b"x")
    (sample / "control_view00_camera_front.mp4").write_bytes(b"x")
    (sample / "vision_view00_camera_front_preview.mp4").write_bytes(b"x")
    assert find_sample_dirs(tmp_path) == [sample]
    assert [idx for idx, _, _ in collect_views(sample)] == [0, 1]
    assert [idx for idx, _, _ in collect_views(sample, "control")] == [0]

    other = tmp_path / "clip_b" / "0"
    other.mkdir(parents=True)
    (other / "vision_view00_camera_front.mp4").write_bytes(b"x")
    assert find_sample_dirs(tmp_path) == [sample, other]
