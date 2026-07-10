# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from io import BytesIO

import numpy as np
import pytest

from cosmos_framework.utils.easy_io.handlers.imageio_video_handler import ImageioVideoHandler


@pytest.mark.L0
def test_dump_to_fileobj_pads_odd_dimensions_for_mp4(monkeypatch):
    calls = {}

    def fake_mimsave(file, obj, format, **kwargs):  # pylint: disable=redefined-builtin
        calls["file"] = file
        calls["obj"] = obj
        calls["format"] = format
        calls["kwargs"] = kwargs

    monkeypatch.setattr(
        "cosmos_framework.utils.easy_io.handlers.imageio_video_handler.imageio.mimsave",
        fake_mimsave,
    )

    video = np.arange(2 * 3 * 5 * 3, dtype=np.uint8).reshape(2, 3, 5, 3)
    ImageioVideoHandler().dump_to_fileobj(video, BytesIO(), crf=25)

    saved = calls["obj"]
    assert saved.shape == (2, 4, 6, 3)
    np.testing.assert_array_equal(saved[:, :3, :5, :], video)
    np.testing.assert_array_equal(saved[:, 3:, :5, :], video[:, 2:3, :, :])
    np.testing.assert_array_equal(saved[:, :3, 5:, :], video[:, :, 4:5, :])
    assert calls["kwargs"]["ffmpeg_params"][-2:] == ["-s", "6x4"]


@pytest.mark.L0
def test_dump_to_fileobj_appends_padded_size_to_custom_non_crf_ffmpeg_params(monkeypatch):
    calls = {}

    def fake_mimsave(file, obj, format, **kwargs):  # pylint: disable=redefined-builtin
        calls["file"] = file
        calls["obj"] = obj
        calls["format"] = format
        calls["kwargs"] = kwargs

    monkeypatch.setattr(
        "cosmos_framework.utils.easy_io.handlers.imageio_video_handler.imageio.mimsave",
        fake_mimsave,
    )

    video = np.zeros((2, 3, 5, 3), dtype=np.uint8)
    ImageioVideoHandler().dump_to_fileobj(video, BytesIO(), ffmpeg_params=["-an"])

    assert calls["obj"].shape == (2, 4, 6, 3)
    assert calls["kwargs"]["ffmpeg_params"] == ["-an", "-s", "6x4"]
