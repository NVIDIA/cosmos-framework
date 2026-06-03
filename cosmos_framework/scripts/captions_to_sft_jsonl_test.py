# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Tests for captions_to_sft_jsonl (caption_json + dense emission, filters, summary)."""

import json

import pytest

from cosmos_framework.inference.structured_caption import CAPTION_JSON_KEY
from cosmos_framework.scripts import captions_to_sft_jsonl as mod


def _meta(width=256, height=256, duration=17.0, fps=5.0, total_frames=85):
    return {"width": width, "height": height, "duration": duration, "fps": fps, "total_frames": total_frames}


def _make_clip(captions_dir, videos_dir, name, dense="A robot arm.", caption_json=None):
    d = captions_dir / name
    d.mkdir(parents=True)
    if dense is not None:
        (d / "caption.txt").write_text(dense)
    if caption_json is not None:
        (d / "caption.json").write_text(json.dumps(caption_json))
    (videos_dir / f"{name}.mp4").write_bytes(b"\x00")  # presence only; ffprobe is mocked


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.fixture
def dirs(tmp_path):
    captions_dir = tmp_path / "captions"
    videos_dir = tmp_path / "videos"
    captions_dir.mkdir()
    videos_dir.mkdir()
    return captions_dir, videos_dir


def test_emits_caption_json_and_dense(dirs, tmp_path, monkeypatch):
    captions_dir, videos_dir = dirs
    cj = {"background_setting": "kitchen", "temporal_caption": "A robot arm.", "fps": 5}
    _make_clip(captions_dir, videos_dir, "ep0", dense="A robot arm.", caption_json=cj)
    monkeypatch.setattr(mod, "probe_video_metadata", lambda p: _meta())

    out = tmp_path / "ds.jsonl"
    mod.main(captions_dir=captions_dir, videos_dir=videos_dir, output=out)

    rows = _read_jsonl(out)
    assert len(rows) == 1
    window = rows[0]["t2w_windows"][0]
    assert window[CAPTION_JSON_KEY] == cj  # structured, as a dict object
    assert window["caption"] == "A robot arm."  # dense backup
    assert window["start_frame"] == 0 and window["end_frame"] == 84
    assert rows[0]["uuid"] == "ep0"
    # vision_path is relative to the output JSONL dir.
    assert rows[0]["vision_path"] == "videos/ep0.mp4"

    summary = json.loads((tmp_path / "ds.jsonl.summary.json").read_text())
    assert summary["records_kept"] == 1 and summary["records_with_caption_json"] == 1


def test_dense_only_when_no_caption_json(dirs, tmp_path, monkeypatch):
    captions_dir, videos_dir = dirs
    _make_clip(captions_dir, videos_dir, "ep0", dense="Only dense.", caption_json=None)
    monkeypatch.setattr(mod, "probe_video_metadata", lambda p: _meta())

    out = tmp_path / "ds.jsonl"
    mod.main(captions_dir=captions_dir, videos_dir=videos_dir, output=out)
    window = _read_jsonl(out)[0]["t2w_windows"][0]
    assert CAPTION_JSON_KEY not in window
    assert window["caption"] == "Only dense."


def test_drops_long_and_short_clips(dirs, tmp_path, monkeypatch):
    captions_dir, videos_dir = dirs
    _make_clip(captions_dir, videos_dir, "good", caption_json={"x": 1})
    _make_clip(captions_dir, videos_dir, "toolong", caption_json={"x": 1})
    _make_clip(captions_dir, videos_dir, "tooshort", caption_json={"x": 1})

    def fake_probe(path):
        if "toolong" in str(path):
            return _meta(duration=99.0)
        if "tooshort" in str(path):
            return _meta(total_frames=10)
        return _meta()

    monkeypatch.setattr(mod, "probe_video_metadata", fake_probe)
    out = tmp_path / "ds.jsonl"
    mod.main(captions_dir=captions_dir, videos_dir=videos_dir, output=out)

    rows = _read_jsonl(out)
    assert {r["uuid"] for r in rows} == {"good"}
    summary = json.loads((tmp_path / "ds.jsonl.summary.json").read_text())
    assert summary["drops_by_reason"]["duration_too_long"] == 1
    assert summary["drops_by_reason"]["too_few_frames"] == 1


def test_num_video_frames_filter_drops_85_frame_clip(dirs, tmp_path, monkeypatch):
    captions_dir, videos_dir = dirs
    _make_clip(captions_dir, videos_dir, "ep0", caption_json={"x": 1})
    monkeypatch.setattr(mod, "probe_video_metadata", lambda p: _meta(total_frames=85))

    out = tmp_path / "ds.jsonl"
    # With num_video_frames=93, an 85-frame clip must be dropped (matches decode-time
    # filtering); with the default -1 it is kept.
    with pytest.raises(SystemExit):
        mod.main(captions_dir=captions_dir, videos_dir=videos_dir, output=out, num_video_frames=93)
    summary = json.loads((tmp_path / "ds.jsonl.summary.json").read_text())
    assert summary["drops_by_reason"]["too_few_frames"] == 1


def test_caption_json_falls_back_to_temporal_caption_for_dense(dirs, tmp_path, monkeypatch):
    captions_dir, videos_dir = dirs
    # No caption.txt; dense should be recovered from caption.json temporal_caption.
    cj = {"temporal_caption": "Recovered dense.", "fps": 5}
    d = captions_dir / "ep0"
    d.mkdir(parents=True)
    (d / "caption.json").write_text(json.dumps(cj))
    (videos_dir / "ep0.mp4").write_bytes(b"\x00")
    monkeypatch.setattr(mod, "probe_video_metadata", lambda p: _meta())

    out = tmp_path / "ds.jsonl"
    mod.main(captions_dir=captions_dir, videos_dir=videos_dir, output=out)
    window = _read_jsonl(out)[0]["t2w_windows"][0]
    assert window[CAPTION_JSON_KEY] == cj
    assert window["caption"] == "Recovered dense."
