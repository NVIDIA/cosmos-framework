# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Tests for the local-dataset ffprobe helper."""

import json
import shutil
import subprocess

import pytest

from cosmos_framework.data.generator.local_datasets import helper

# ffprobe -show_streams -show_format -select_streams v:0 output (trimmed) for the same
# 90-frame clip in different containers.
_MP4 = {
    "streams": [{"width": 320, "height": 240, "r_frame_rate": "30/1", "duration": "3.000000", "nb_frames": "90"}],
    "format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2", "duration": "3.000000"},
}
_MKV = {
    "streams": [{"width": 320, "height": 240, "r_frame_rate": "30/1", "tags": {"DURATION": "00:00:03.000000000"}}],
    "format": {"format_name": "matroska,webm", "duration": "3.000000"},
}
_WEBM = {
    "streams": [{"width": 320, "height": 240, "r_frame_rate": "30/1", "tags": {"DURATION": "00:00:03.000000000"}}],
    "format": {"format_name": "matroska,webm", "duration": "3.000000"},
}
# 29.97 fps video track (3.003 s) muxed with a 5 s audio track.
_MKV_LONGER_AUDIO = {
    "streams": [
        {"width": 160, "height": 120, "r_frame_rate": "30000/1001", "tags": {"DURATION": "00:00:03.003000000"}}
    ],
    "format": {"format_name": "matroska,webm", "duration": "5.008000"},
}
# Derived from _MKV: no usable per-track tag, so the container duration is used.
_MKV_NO_TRACK_TAG = {
    "streams": [{"width": 320, "height": 240, "r_frame_rate": "30/1"}],
    "format": {"format_name": "matroska,webm", "duration": "3.000000"},
}
_MKV_BAD_TRACK_TAG = {
    "streams": [{"width": 320, "height": 240, "r_frame_rate": "30/1", "tags": {"DURATION": "N/A"}}],
    "format": {"format_name": "matroska,webm", "duration": "3.000000"},
}


@pytest.mark.parametrize(
    "probe",
    [_MP4, _MKV, _WEBM, _MKV_LONGER_AUDIO, _MKV_NO_TRACK_TAG, _MKV_BAD_TRACK_TAG],
    ids=["mp4", "mkv", "webm", "mkv_longer_audio", "mkv_no_track_tag", "mkv_bad_track_tag"],
)
def test_get_video_metadata_total_frames(monkeypatch: pytest.MonkeyPatch, probe: dict) -> None:
    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(probe), stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert helper.get_video_metadata("clip")["total_frames"] == 90


def test_get_video_metadata_real_mkv(tmp_path) -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg and ffprobe are required to encode and probe a real clip")
    path = tmp_path / "clip.mkv"
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=64x64:rate=30"]
        + ["-frames:v", "90", "-c:v", "mpeg4", str(path)],
        check=True,
    )
    assert helper.get_video_metadata(str(path))["total_frames"] == 90
