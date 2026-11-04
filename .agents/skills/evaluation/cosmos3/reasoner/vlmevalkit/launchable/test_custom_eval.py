# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic annotation, inference-boundary, scoring and identity contracts."""

import copy
import json
from pathlib import Path
import shutil
import subprocess

import pytest

import custom_eval as custom


@pytest.fixture
def row():
    return {"id": "invented-001", "video": "clip.mp4", "question": "What happens next?",
            "candidates": ["A box moves left", "A box moves right", "A box stays still", "A box vanishes"],
            "answer": "A box moves right", "category": "motion", "start": 0, "end": 1}


@pytest.fixture
def root(tmp_path):
    media = tmp_path / "media"
    media.mkdir()
    # Annotation/path unit tests need only a nonempty file, never decode it.
    (media / "clip.mp4").write_bytes(b"invented path-only fixture; not media")
    return media


@pytest.mark.parametrize("change", [
    {"answer": "B"}, {"answer": "A box moves right "}, {"answer": None},
    {"candidates": ["same", "same"], "answer": "same"},
    {"candidates": ["only one"], "answer": "only one"},
    {"candidates": [str(i) for i in range(27)], "answer": "0"},
    {"start": 1, "end": 1}, {"start": -1}, {"end": float("nan")},
    {"start": True}, {"video": "../clip.mp4"}, {"video": "/tmp/clip.mp4"},
    {"video": "https://example.invalid/video.mp4"}, {"video": "missing.mp4"},
])
def test_reject_bad_canonical_rows(row, root, change):
    row.update(change)
    with pytest.raises(ValueError):
        custom.validate_row(row, root.resolve(), set())


def test_labels_ids_ranges_and_symlinks(row, root, tmp_path):
    seen = set()
    validated, path = custom.validate_row(row, root.resolve(), seen)
    assert validated == row and path == root / "clip.mp4"
    with pytest.raises(ValueError, match="Duplicate id"):
        custom.validate_row(row, root.resolve(), seen)
    del row["end"]
    with pytest.raises(ValueError, match="supplied together"):
        custom.validate_row(row, root.resolve(), set())
    row.pop("start")
    del row["answer"]
    with pytest.raises(ValueError, match="answer"):
        custom.validate_row(row, root.resolve(), set())
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"invented")
    (root / "link.mp4").symlink_to(outside)
    row.update(answer=row["candidates"][0], video="link.mp4")
    with pytest.raises(ValueError, match="escaping"):
        custom.validate_row(row, root.resolve(), set())


def test_two_and_twenty_six_options_are_supported(row, root):
    for count in (2, 26):
        row.update(candidates=[f"Option {i}" for i in range(count)], answer=f"Option {count - 1}")
        custom.validate_row(row, root.resolve(), set())
        assert custom.score_answer(row, chr(64 + count), "stop")["correct"]


def test_inference_payload_is_label_independent(row):
    settings = {"model": {"name": "invented-model"}, "sampling": {"nframes": 8},
                "generation": {"temperature": 0, "seed": 42, "max_tokens": 32, "enable_thinking": False}}
    payload = custom.build_payload(row, "data:video/mp4;base64,INVENTED", {"fps": 2}, settings)
    changed = {**row, "answer": row["candidates"][0], "category": "SECRET_CATEGORY", "id": "SECRET_ID", "start": 50, "end": 60}
    assert custom.build_payload(changed, "data:video/mp4;base64,INVENTED", {"fps": 2}, settings) == payload
    text = json.dumps(payload)
    assert "SECRET" not in text and '"answer"' not in text and '"category"' not in text
    assert all(candidate in text for candidate in row["candidates"])


def test_invalid_and_truncated_answers_stay_in_denominator(row):
    rows = [{**row, "id": str(i), "category": "motion" if i < 2 else "other"} for i in range(5)]
    answers = [("B", "stop"), ("B", "length"), ("B. explanation", "stop"), (None, "stop"), ("A", "stop")]
    results = [{"id": r["id"], **custom.score_answer(r, answer, finish)} for r, (answer, finish) in zip(rows, answers)]
    summary = custom.summarize(rows, results)
    assert summary["status"] == "completed"
    assert summary["coverage"] == {"expected": 5, "scored": 5}
    assert summary["primary_metric"] == {"name": "accuracy_percent", "value": 20, "unit": "percent", "correct": 1, "total": 5}
    assert summary["categories"]["motion"]["accuracy_percent"] == 50
    assert summary["categories"]["other"]["total"] == 3
    assert results[2]["prediction"] is None


def test_transport_failure_or_missing_rows_cannot_complete(row):
    rows = [row, {**row, "id": "second"}]
    good = {"id": row["id"], "correct": True}
    for results in ([good], [good, {"id": "second", "error": "ConnectionError"}]):
        summary = custom.summarize(rows, results)
        assert summary["status"] == "failed"
        assert summary["coverage"] == {"expected": 2, "scored": 1}
        assert summary["primary_metric"]["value"] is None
        assert summary["primary_metric"]["total"] == 2
    with pytest.raises(ValueError, match="duplicate"):
        custom.summarize(rows, [good, good])


@pytest.fixture
def real_config(tmp_path, row):
    pytest.importorskip("decord")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg is required to generate invented media")
    media = tmp_path / "media"
    media.mkdir()
    subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                    "color=c=blue:s=64x64:r=8:d=2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    str(media / "clip.mp4")], check=True)
    manifest = tmp_path / "rows.jsonl"
    manifest.write_text(json.dumps(row) + "\n")
    config = {"dataset": {"name": "invented", "revision": "test-v1", "source": "synthetic test metadata"},
              "manifest": "rows.jsonl", "media_root": "media", "nframes": 8,
              "min_pixels": 1024, "max_pixels": 4096, "image_patch_size": 16,
              "model": "invented-model-one", "model_revision": "invented-revision"}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return path, manifest, config


def test_real_validation_identity_is_model_and_cwd_independent(real_config, tmp_path, monkeypatch):
    path, manifest, config = real_config
    first = custom.validate(path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    changed = copy.deepcopy(config)
    changed["model"] = "invented-model-two"
    changed["generation"] = {"seed": 123}
    path.write_text(json.dumps(changed))
    second = custom.validate(path)
    assert first["dataset_identity"] == second["dataset_identity"]
    # Independently serialized equivalent canonical JSONL keeps semantic data identity.
    annotation = json.loads(manifest.read_text())
    manifest.write_text(json.dumps(annotation, sort_keys=True) + "\n")
    assert custom.validate(path)["dataset_identity"] == first["dataset_identity"]
    annotation["answer"] = annotation["candidates"][0]
    manifest.write_text(json.dumps(annotation) + "\n")
    assert custom.validate(path)["dataset_identity"] != first["dataset_identity"]
    annotation["end"] = 10
    manifest.write_text(json.dumps(annotation) + "\n")
    with pytest.raises(ValueError, match="exceeds video duration"):
        custom.validate(path)


def test_real_media_bytes_participate_in_identity(real_config):
    path, _, _ = real_config
    first = custom.validate(path)
    with (path.parent / "media" / "clip.mp4").open("ab") as stream:
        stream.write(b"invented trailing metadata")
    assert custom.validate(path)["dataset_identity"] != first["dataset_identity"]


def test_unicode_line_separators_preserve_a_single_jsonl_question(real_config):
    path, manifest, _ = real_config
    annotation = json.loads(manifest.read_text())
    annotation["question"] = "What moves\u2028after the box\u2029stops?"
    manifest.write_text(json.dumps(annotation, ensure_ascii=False) + "\n")
    assert custom._load(path)[2][0]["question"] == annotation["question"]


def test_ancestor_media_root_cannot_include_source_media(tmp_path, monkeypatch, row):
    source = tmp_path / "checkout"
    (source / ".git").mkdir(parents=True)
    (source / "launchable").mkdir()
    (source / "clip.mp4").write_bytes(b"invented nonempty path fixture")
    monkeypatch.setattr(custom, "__file__", str(source / "launchable" / "custom_eval.py"))
    row["video"] = "checkout/clip.mp4"
    with pytest.raises(ValueError, match="outside the source"):
        custom.validate_row(row, tmp_path, set())
