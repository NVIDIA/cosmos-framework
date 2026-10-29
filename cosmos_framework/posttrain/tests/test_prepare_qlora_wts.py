# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""CPU conversion contracts; fixtures do not claim video-decoder qualification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from cosmos_framework.posttrain.scripts.prepare_qlora_wts import _array_rows, prepare_wts

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def _inputs(tmp_path: Path, count: int = 3) -> dict:
    options = {"output_dir": tmp_path / "converted"}
    for split in ("train", "validation"):
        media = tmp_path / split / "videos"
        media.mkdir(parents=True)
        rows = []
        for index in range(count):
            video = f"nested/video{index}.mp4"
            path = media / video
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(f"distinct-video-content-{split}-{index}".encode())
            rows.append(
                {
                    "id": f"{split}-{index}",
                    "wts_id": index,
                    "video": video,
                    "conversations": [
                        {"from": "human", "value": "<video>\n  Which letter?\nA. α\nB. β\n"},
                        {"from": "gpt", "value": "A"},
                        {"from": "human", "value": "Repeat verbatim. "},
                        {"from": "gpt", "value": " A\n"},
                    ],
                }
            )
        annotations = tmp_path / split / "annotations.json"
        annotations.write_text(json.dumps(rows, ensure_ascii=False))
        options[f"{split}_annotations"] = annotations
        options[f"{split}_media"] = media
    return options


def test_conversion_preserves_text_answers_and_real_video_handles(tmp_path):
    options = _inputs(tmp_path)
    manifest = prepare_wts(**options, system_prompt="Answer with its letter.")
    row = json.loads((options["output_dir"] / "train.jsonl").read_text().splitlines()[0])
    messages = row["messages"]
    assert messages[0] == {"role": "system", "content": "Answer with its letter."}
    assert messages[1]["content"] == [
        {"type": "video", "video": str((options["train_media"] / "nested/video0.mp4").resolve())},
        {"type": "text", "text": "\n  Which letter?\nA. α\nB. β\n"},
    ]
    assert [message["content"] for message in messages[2:]] == ["A", "Repeat verbatim. ", " A\n"]
    assert manifest["full_split_disjointness"] == {"row_ids": True, "video_sha256": True}
    assert (
        manifest["splits"]["train"]["output"]["sha256"]
        == hashlib.sha256((options["output_dir"] / "train.jsonl").read_bytes()).hexdigest()
    )
    assert len((options["output_dir"] / "media_inventory.jsonl").read_text().splitlines()) == 6


def test_subset_keeps_full_split_inventory_and_identical_reruns_are_safe(tmp_path):
    options = _inputs(tmp_path)
    first = prepare_wts(**options, train_limit=1, validation_limit=1, seed=42)
    assert first == prepare_wts(**options, train_limit=1, validation_limit=1, seed=42)
    assert first["splits"]["train"]["source_rows"] == 3
    assert first["splits"]["train"]["selected_rows"] == 1
    assert len((options["output_dir"] / "media_inventory.jsonl").read_text().splitlines()) == 6
    row = json.loads((options["output_dir"] / "train.jsonl").read_text())
    assert row["messages"][0]["role"] == "user"  # no implicit system prompt
    accepted = (options["output_dir"] / "manifest.json").read_bytes()
    with pytest.raises(ValueError, match="existing different"):
        prepare_wts(**options, train_limit=2, validation_limit=1)
    assert (options["output_dir"] / "manifest.json").read_bytes() == accepted


def test_overlapping_media_outside_selected_subset_is_rejected(tmp_path):
    options = _inputs(tmp_path)
    (options["validation_media"] / "nested/video2.mp4").write_bytes(
        (options["train_media"] / "nested/video2.mp4").read_bytes()
    )
    with pytest.raises(ValueError, match="video content overlaps across full"):
        prepare_wts(**options, train_limit=1, validation_limit=1)
    assert not options["output_dir"].exists()


def test_overlapping_ids_across_full_splits_are_rejected(tmp_path):
    options = _inputs(tmp_path)
    path = options["validation_annotations"]
    rows = json.loads(path.read_text())
    rows[-1]["id"] = "train-2"
    path.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="overlapping row id"):
        prepare_wts(**options, train_limit=1, validation_limit=1)
    assert not options["output_dir"].exists()


@pytest.mark.parametrize("text", ["No video marker", "<video><video>", "<video>question"])
def test_bad_markers_fail_without_silent_row_drops(tmp_path, text):
    options = _inputs(tmp_path)
    path = options["train_annotations"]
    rows = json.loads(path.read_text())
    rows[-1]["conversations"][0]["value"] = text
    if text == "<video>question":
        rows[-1]["conversations"][1]["value"] = "<video>"
    path.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="exactly one <video>"):
        prepare_wts(**options, train_limit=1)
    assert not options["output_dir"].exists()


def test_media_path_cannot_escape_its_declared_root(tmp_path):
    options = _inputs(tmp_path)
    path = options["train_annotations"]
    rows = json.loads(path.read_text())
    rows[0]["video"] = "../annotations.json"
    path.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="inside its media root"):
        prepare_wts(**options)


@pytest.mark.parametrize("content", ['[{"id":1},]', '[{"id":1}] false', '[{"id":1}', "{}"])
def test_streaming_array_rejects_invalid_json(tmp_path, content):
    path = tmp_path / "annotations.json"
    path.write_text(content)
    with pytest.raises(ValueError):
        list(_array_rows(path))


def test_streaming_array_crosses_small_chunk_boundaries(tmp_path, monkeypatch):
    monkeypatch.setattr("cosmos_framework.posttrain.scripts.prepare_qlora_wts._CHUNK", 7)
    path = tmp_path / "annotations.json"
    values = [{"id": i, "text": 'α,\\"[]' * 10} for i in range(20)]
    path.write_text(json.dumps(values, ensure_ascii=False))
    assert list(_array_rows(path)) == values


def test_incomplete_existing_output_is_never_overwritten(tmp_path):
    options = _inputs(tmp_path)
    options["output_dir"].mkdir()
    sentinel = options["output_dir"] / "train.jsonl"
    sentinel.write_text("interrupted")
    with pytest.raises(ValueError, match="incomplete output"):
        prepare_wts(**options)
    assert sentinel.read_text() == "interrupted"
