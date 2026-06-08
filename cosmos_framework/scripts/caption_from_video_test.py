# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Tests for caption_from_video input handling (remote URLs + manifest media)."""

import json

from cosmos_framework.scripts.caption_from_video import (
    _PACKAGE_DIR,
    _build_vlm_messages,
    _is_remote_ref,
    _read_manifest_entries,
    _video_url,
)


def test_default_template_path_resolves():
    # Guards against the default prompt-template path regressing (it must point at
    # inference/defaults/, not cosmos_framework/defaults/).
    assert (_PACKAGE_DIR / "inference/defaults/video_captioner.txt").is_file()


def test_is_remote_ref():
    assert _is_remote_ref("https://h/a.mp4")
    assert _is_remote_ref("http://h/a.mp4")
    assert _is_remote_ref("data:video/mp4;base64,AAAA")
    assert not _is_remote_ref("/abs/a.mp4")
    assert not _is_remote_ref("videos/a.mp4")


def test_video_url_passthrough_and_file():
    assert _video_url("https://h/a.mp4") == "https://h/a.mp4"
    assert _video_url("/abs/a.mp4") == "file:///abs/a.mp4"
    assert _video_url("data:video/mp4;base64,AB") == "data:video/mp4;base64,AB"


def test_build_messages_uses_remote_url_verbatim():
    msgs = _build_vlm_messages("https://h/a.mp4", "PROMPT")
    content = msgs[0]["content"]
    assert content[0]["video_url"]["url"] == "https://h/a.mp4"
    assert content[1]["text"] == "PROMPT"


def test_read_manifest_jsonl_with_url_and_media(tmp_path):
    media = {"resolution": {"H": 256, "W": 256}, "aspect_ratio": "1,1", "duration": "17s", "fps": 5}
    p = tmp_path / "m.jsonl"
    p.write_text(
        json.dumps({"name": "ep0", "vision_path": "https://h/ep0.mp4", "media": media})
        + "\n"
        + json.dumps({"vision_path": "https://h/ep1.mp4"})  # no name -> derive stem; no media
        + "\n"
    )
    items = _read_manifest_entries([p])
    assert items[0] == ("ep0", "https://h/ep0.mp4", media)
    assert items[1] == ("ep1", "https://h/ep1.mp4", None)


def test_read_manifest_skips_bad_entries(tmp_path):
    p = tmp_path / "m.jsonl"
    p.write_text(
        json.dumps({"name": "novp"})  # missing vision_path
        + "\n"
        + json.dumps({"vision_path": "https://h/notavideo.txt"})  # wrong suffix
        + "\n"
        + json.dumps({"vision_path": "https://h/good.mp4"})
        + "\n"
    )
    items = _read_manifest_entries([p])
    assert items == [("good", "https://h/good.mp4", None)]


def test_read_manifest_json_object_and_list(tmp_path):
    obj = tmp_path / "o.json"
    obj.write_text(json.dumps({"vision_path": "/local/a.mp4"}))
    lst = tmp_path / "l.json"
    lst.write_text(json.dumps([{"vision_path": "/local/b.mp4"}, {"vision_path": "/local/c.mp4"}]))
    assert _read_manifest_entries([obj]) == [("a", "/local/a.mp4", None)]
    assert [i[0] for i in _read_manifest_entries([lst])] == ["b", "c"]
