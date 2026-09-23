# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the cosmos-video-reasoning-v1.0 validator."""

import json
from pathlib import Path

import pytest

from cosmos_framework.data.reasoner.formats.validate_reasoning import VideoReasoningV1_0Validator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _minimal_dataset(tmp_path: Path, *, with_media: bool = True) -> Path:
    """Build a minimal valid cosmos-video-reasoning-v1.0 dataset under ``tmp_path``."""
    dataset = tmp_path / "ds"
    if with_media:
        (dataset / "videos").mkdir(parents=True)
        (dataset / "videos" / "v.mp4").write_bytes(b"")
    _write(
        dataset / "bcq.json",
        {
            "format": "cosmos-video-reasoning-v1.0",
            "metadata": {"type": "annotation", "license": "CC BY-NC-ND 4.0", "task": "bcq"},
            "media_root": None,
            "items": [
                {
                    "video_id": "videos/v.mp4",
                    "question": "Does anything happen?",
                    "answer": "Yes",
                }
            ],
        },
    )
    return dataset


# ---------------------------------------------------------------------------
# Validator instantiation + class-level invariants
# ---------------------------------------------------------------------------


class TestValidatorVideoReasoningV1_0Instantiation:
    def test_format_constant(self):
        assert VideoReasoningV1_0Validator.format == "cosmos-video-reasoning-v1.0"

    def test_instantiate(self):
        v = VideoReasoningV1_0Validator()
        assert isinstance(v, VideoReasoningV1_0Validator)


# ---------------------------------------------------------------------------
# Positive: minimal valid dataset
# ---------------------------------------------------------------------------


class TestMinimalValid:
    def test_passes(self, tmp_path):
        ds = _minimal_dataset(tmp_path)
        result = VideoReasoningV1_0Validator().validate_dataset(ds)
        assert result.is_valid(), result.errors
        assert result.files_checked == 1
        assert result.files_passed == 1

    def test_no_warnings_in_lenient(self, tmp_path):
        ds = _minimal_dataset(tmp_path)
        result = VideoReasoningV1_0Validator().validate_dataset(ds, permissive=True)
        assert result.warnings == []


# ---------------------------------------------------------------------------
# Negative: schema violations
# ---------------------------------------------------------------------------


class TestSchemaViolations:
    def test_wrong_format_string_rejected(self, tmp_path):
        ds = tmp_path / "ds"
        ds.mkdir()
        _write(
            ds / "x.json",
            {
                "format": "tao-vl-reason-v999",
                "metadata": {"type": "annotation", "license": "CC BY-NC-ND 4.0", "task": "bcq"},
                "items": [],
            },
        )
        # Files with the wrong `format` aren't recognized as part of the dataset,
        # so find_datasets returns nothing.
        result = VideoReasoningV1_0Validator().validate_dataset(ds)
        assert result.files_checked == 0

    def test_wrong_metadata_type_rejected(self, tmp_path):
        """metadata.type is a const — anything other than 'annotation' fails."""
        ds = tmp_path / "ds"
        ds.mkdir()
        _write(
            ds / "x.json",
            {
                "format": "cosmos-video-reasoning-v1.0",
                "metadata": {"type": "bcq"},  # legacy free-form value, no longer accepted
                "items": [],
            },
        )
        result = VideoReasoningV1_0Validator().validate_dataset(ds)
        assert not result.is_valid()
        assert any("annotation" in e for e in result.errors)

    def test_missing_metadata_type_rejected(self, tmp_path):
        ds = tmp_path / "ds"
        (ds / "videos").mkdir(parents=True)
        (ds / "videos" / "v.mp4").write_bytes(b"")
        _write(
            ds / "x.json",
            {
                "format": "cosmos-video-reasoning-v1.0",
                "metadata": {},  # missing required `type`
                "items": [],
            },
        )
        result = VideoReasoningV1_0Validator().validate_dataset(ds)
        assert not result.is_valid()
        assert any("type" in e for e in result.errors)

    def test_item_must_have_video_or_image(self, tmp_path):
        ds = tmp_path / "ds"
        ds.mkdir()
        _write(
            ds / "x.json",
            {
                "format": "cosmos-video-reasoning-v1.0",
                "metadata": {"type": "annotation", "license": "CC BY-NC-ND 4.0", "task": "open_qa"},
                "items": [{"question": "Q?", "answer": "A"}],
            },
        )
        result = VideoReasoningV1_0Validator().validate_dataset(ds)
        assert not result.is_valid()

    def test_both_video_and_image_rejected(self, tmp_path):
        """oneOf enforces exactly one of video_id / image_id."""
        ds = tmp_path / "ds"
        ds.mkdir()
        _write(
            ds / "x.json",
            {
                "format": "cosmos-video-reasoning-v1.0",
                "metadata": {"type": "annotation", "license": "CC BY-NC-ND 4.0", "task": "open_qa"},
                "items": [
                    {
                        "video_id": "v.mp4",
                        "image_id": "i.jpg",
                        "question": "Q?",
                        "answer": "A",
                    }
                ],
            },
        )
        result = VideoReasoningV1_0Validator().validate_dataset(ds)
        assert not result.is_valid()


# ---------------------------------------------------------------------------
# Cross-references: media path resolution
# ---------------------------------------------------------------------------


class TestMediaReferences:
    def test_missing_media_errors_in_lenient(self, tmp_path):
        """A referenced media file that doesn't exist on disk is always an
        error, even in permissive (default) mode. The annotation's
        ``video_id`` / ``image_id`` is a claim that the file exists, so its
        absence breaks the format contract regardless of ``--strict``."""
        ds = _minimal_dataset(tmp_path, with_media=False)
        result = VideoReasoningV1_0Validator().validate_dataset(ds, permissive=True)
        assert not result.is_valid()
        assert any("media not found" in e for e in result.errors)

    def test_missing_media_errors_in_strict(self, tmp_path):
        ds = _minimal_dataset(tmp_path, with_media=False)
        result = VideoReasoningV1_0Validator().validate_dataset(ds, permissive=False)
        assert not result.is_valid()
        assert any("media not found" in e for e in result.errors)

    def test_media_root_relative_subdir(self, tmp_path):
        """Items reference filenames under the configured media_root subdir."""
        ds = tmp_path / "ds"
        (ds / "media").mkdir(parents=True)
        (ds / "media" / "v.mp4").write_bytes(b"")
        _write(
            ds / "x.json",
            {
                "format": "cosmos-video-reasoning-v1.0",
                "metadata": {"type": "annotation", "license": "CC BY-NC-ND 4.0", "task": "open_qa"},
                "media_root": "media/",
                "items": [{"video_id": "v.mp4", "question": "Q?", "answer": "A"}],
            },
        )
        result = VideoReasoningV1_0Validator().validate_dataset(ds)
        assert result.is_valid(), result.errors

    def test_image_id_resolves(self, tmp_path):
        ds = tmp_path / "ds"
        (ds / "images").mkdir(parents=True)
        (ds / "images" / "i.jpg").write_bytes(b"")
        _write(
            ds / "x.json",
            {
                "format": "cosmos-video-reasoning-v1.0",
                "metadata": {
                    "type": "annotation",
                    "license": "CC BY-NC-ND 4.0",
                    "task": "grounding",
                },
                "media_root": None,
                "items": [{"image_id": "images/i.jpg", "question": "Q?", "answer": "A"}],
            },
        )
        result = VideoReasoningV1_0Validator().validate_dataset(ds)
        assert result.is_valid(), result.errors


# ---------------------------------------------------------------------------
# Multi-file datasets
# ---------------------------------------------------------------------------


class TestMultipleAnnotationFiles:
    def test_multiple_files_aggregated(self, tmp_path):
        """A dataset can have several annotation files; all are validated."""
        ds = tmp_path / "ds"
        (ds / "videos").mkdir(parents=True)
        (ds / "videos" / "v.mp4").write_bytes(b"")
        for task in ("bcq", "mcq", "open_qa"):
            _write(
                ds / f"{task}.json",
                {
                    "format": "cosmos-video-reasoning-v1.0",
                    "metadata": {"type": "annotation", "license": "CC BY-NC-ND 4.0", "task": task},
                    "media_root": None,
                    "items": [{"video_id": "videos/v.mp4", "question": "Q?", "answer": "A"}],
                },
            )
        result = VideoReasoningV1_0Validator().validate_dataset(ds)
        assert result.is_valid(), result.errors
        assert result.files_checked == 3
        assert result.files_passed == 3


# ---------------------------------------------------------------------------
# Real example datasets
# ---------------------------------------------------------------------------


class TestSchemaToCrossRefGating:
    def test_schema_failed_annotation_skipped_in_media_phase(self, tmp_path):
        """An annotation file that fails schema validation is omitted from
        ``_validate_schemas``'s return value — its items therefore never
        reach ``_validate_media_references``, so missing-media complaints
        are not piled on top of the schema error.

        We trigger the schema violation through ``metadata.type`` (not
        ``format``) because ``format`` is the discovery key — a wrong
        ``format`` value would cause the file to be silently ignored as
        a non-tao-vl JSON rather than enter the validator at all."""
        ds = tmp_path / "ds"
        _write(
            ds / "bad.json",
            {
                "format": "cosmos-video-reasoning-v1.0",  # discovery passes
                "metadata": {
                    "type": "wrong-metadata-type",  # schema violation
                    "license": "CC BY-NC-ND 4.0",
                    "task": "bcq",
                },
                "media_root": None,
                "items": [
                    {
                        "video_id": "videos/does_not_exist.mp4",
                        "question": "Q?",
                        "answer": "A",
                    }
                ],
            },
        )
        result = VideoReasoningV1_0Validator().validate_dataset(ds)
        assert not result.is_valid()
        joined = " ".join(result.errors)
        # Schema error for the bad ``metadata.type`` value is present.
        assert "bad.json" in joined
        # No media-reference error for the (would-be-missing) video file:
        # the annotation file was skipped before the media phase.
        assert "does_not_exist.mp4" not in joined


# ---------------------------------------------------------------------------
# Type-fuzz: non-string values on schema-typed item fields must produce
# clean structured errors, never tracebacks.
# ---------------------------------------------------------------------------


class TestTypeFuzz:
    @pytest.mark.parametrize("field", ["video_id", "question", "answer"])
    @pytest.mark.parametrize("bad_value", [[], {}, 42])
    def test_non_string_item_field_produces_clean_error(self, tmp_path, field, bad_value):
        ds = _minimal_dataset(tmp_path)
        ann_path = ds / "bcq.json"
        ann = json.loads(ann_path.read_text())
        ann["items"][0][field] = bad_value
        _write(ann_path, ann)
        # Must not raise; must return a structured failure result.
        result = VideoReasoningV1_0Validator().validate_dataset(ds)
        assert not result.is_valid()
