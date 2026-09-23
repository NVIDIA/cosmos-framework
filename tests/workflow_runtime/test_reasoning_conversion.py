# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the metropolis-v3.0 → cosmos-video-reasoning-v1.0 converter."""

import json
from pathlib import Path

from cosmos_framework.data.reasoner.formats.metropolis_to_reasoning import MetropolisV3_0ToVideoReasoningV1_0Converter
from cosmos_framework.data.reasoner.formats.validate_reasoning import VideoReasoningV1_0Validator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_scene(tmp_path: Path, task_type: str, items: list, *, with_media: bool = True) -> Path:
    """Create a minimal metropolis-v3.0 scene with one task file plus the
    matching ``contextual/`` entries and ``raw/`` media files needed for
    the converter's schema-contract lookup (``{media_id}.{format}``).

    The converter relies on each ``video_id`` / ``image_id`` being declared
    in a ``contextual/`` file with a sibling ``format``, so the helper
    auto-derives the set of declared media from the ``items`` list. Videos
    are recorded as ``format: "mp4"`` and images as ``format: "jpg"`` —
    individual tests can override by writing their own scene.
    """
    scene = tmp_path / "test_scene"
    (scene / "task").mkdir(parents=True)
    (scene / "contextual").mkdir()
    if with_media:
        (scene / "raw").mkdir()

    video_ids = sorted({i["video_id"] for i in items if "video_id" in i})
    image_ids = sorted({i["image_id"] for i in items if "image_id" in i})

    for vid in video_ids:
        (scene / "contextual" / f"video_{vid}.json").write_text(
            json.dumps({"metadata": {"type": "video"}, "video_id": vid, "format": "mp4"})
        )
        if with_media:
            (scene / "raw" / f"{vid}.mp4").write_bytes(b"")
    for img in image_ids:
        (scene / "contextual" / f"image_{img}.json").write_text(
            json.dumps({"metadata": {"type": "image"}, "image_id": img, "format": "jpg"})
        )
        if with_media:
            (scene / "raw" / f"{img}.jpg").write_bytes(b"")

    (scene / "task" / f"{task_type}.json").write_text(
        json.dumps(
            {
                "version": "metropolis-v3.0",
                "metadata": {"type": task_type},
                "items": items,
            },
            indent=2,
        )
    )
    return scene


# ---------------------------------------------------------------------------
# Class-level invariants
# ---------------------------------------------------------------------------


class TestClassInvariants:
    def test_format_strings(self):
        assert MetropolisV3_0ToVideoReasoningV1_0Converter.source_format == "metropolis-v3.0"
        assert MetropolisV3_0ToVideoReasoningV1_0Converter.target_format == "cosmos-video-reasoning-v1.0"

    def test_pair_name(self):
        assert (
            MetropolisV3_0ToVideoReasoningV1_0Converter.pair_name() == "metropolis-v3.0_to_cosmos-video-reasoning-v1.0"
        )

    def test_supports_all_metropolis_task_types(self):
        # Same 10 task types as the cosmos-reason converter.
        expected = {
            "open_qa",
            "bcq",
            "bcq_openended",
            "mcq",
            "mcq_openended",
            "video_summarization",
            "scene_description",
            "temporal_localization",
            "temporal_description",
            "causal_linkage",
        }
        assert MetropolisV3_0ToVideoReasoningV1_0Converter.SUPPORTED_TASKS == frozenset(expected)


# ---------------------------------------------------------------------------
# Single-task synthetic conversions — one test per "shape" of answer formatting.
# ---------------------------------------------------------------------------


class TestPerTaskTypeSynthetic:
    def test_open_qa_passes_through(self, tmp_path):
        scene = _make_scene(
            tmp_path,
            "open_qa",
            [{"video_id": "v1", "question": "What happens?", "answer": "A crash."}],
        )
        out = tmp_path / "out"
        result = MetropolisV3_0ToVideoReasoningV1_0Converter().convert_scene(scene, out)
        assert result.is_success()
        assert result.samples_written == 1

        ann = json.loads((out / "open_qa.json").read_text())
        assert ann["format"] == "cosmos-video-reasoning-v1.0"
        assert ann["metadata"]["type"] == "annotation"
        assert ann["metadata"]["task"] == "open_qa"
        assert ann["media_root"] is None
        item = ann["items"][0]
        assert item["question"] == "What happens?"
        assert item["answer"] == "A crash."
        # Copied media is named by the source's path-relative-to-root with
        # ``"/"→"--"`` flattening: here the scene path itself is the root,
        # so the relpath is ``raw/v1.mp4`` → ``raw--v1.mp4``.
        assert item["video_id"] == "videos/raw--v1.mp4"
        assert "reasoning" not in item  # source had none

    def test_bcq_with_explanation(self, tmp_path):
        scene = _make_scene(
            tmp_path,
            "bcq",
            [
                {
                    "video_id": "v1",
                    "question": "Crash?",
                    "answer": "Yes",
                    "explanation": "A sedan rear-ends a truck.",
                }
            ],
        )
        out = tmp_path / "out"
        MetropolisV3_0ToVideoReasoningV1_0Converter().convert_scene(scene, out)
        ann = json.loads((out / "bcq.json").read_text())
        item = ann["items"][0]
        # BCQ instruction is appended to the question.
        assert "Answer with only Yes or No." in item["question"]
        # Answer is composed as "<Yes/No>. <explanation>".
        assert item["answer"] == "Yes. A sedan rear-ends a truck."

    def test_mcq_options_appended_and_letter_resolved(self, tmp_path):
        scene = _make_scene(
            tmp_path,
            "mcq",
            [
                {
                    "video_id": "v1",
                    "question": "Which vehicle?",
                    "options": {"A": "sedan", "B": "truck", "C": "bus"},
                    "answer": "B",
                }
            ],
        )
        out = tmp_path / "out"
        MetropolisV3_0ToVideoReasoningV1_0Converter().convert_scene(scene, out)
        ann = json.loads((out / "mcq.json").read_text())
        item = ann["items"][0]
        # Canonical options block in the question.
        assert "A) sedan" in item["question"]
        assert "B) truck" in item["question"]
        assert "Choose the correct option by letter only." in item["question"]
        # Answer letter is resolved to "B) truck".
        assert item["answer"] == "B) truck"

    def test_temporal_localization_wraps_json(self, tmp_path):
        scene = _make_scene(
            tmp_path,
            "temporal_localization",
            [
                {
                    "video_id": "v1",
                    "question": "When does the collision occur?",
                    "answer": {"start": "00:04", "end": "00:06"},
                }
            ],
        )
        out = tmp_path / "out"
        MetropolisV3_0ToVideoReasoningV1_0Converter().convert_scene(scene, out)
        ann = json.loads((out / "temporal_localization.json").read_text())
        item = ann["items"][0]
        # Format instruction is appended.
        assert "Provide the result in json format" in item["question"]
        # Answer is wrapped in a ```json ...``` code block.
        assert item["answer"].startswith("```json\n")
        assert '"start": "00:04"' in item["answer"]

    def test_reasoning_passes_through_verbatim(self, tmp_path):
        scene = _make_scene(
            tmp_path,
            "open_qa",
            [
                {
                    "video_id": "v1",
                    "question": "Q?",
                    "answer": "A",
                    "reasoning": "This is the chain of thought.",
                }
            ],
        )
        out = tmp_path / "out"
        MetropolisV3_0ToVideoReasoningV1_0Converter().convert_scene(scene, out)
        ann = json.loads((out / "open_qa.json").read_text())
        assert ann["items"][0]["reasoning"] == "This is the chain of thought."

    def test_image_item_uses_image_id_and_images_subdir(self, tmp_path):
        scene = tmp_path / "img_scene"
        (scene / "task").mkdir(parents=True)
        (scene / "contextual").mkdir()
        (scene / "contextual" / "image_i1.json").write_text(
            json.dumps({"metadata": {"type": "image"}, "image_id": "i1", "format": "jpg"})
        )
        (scene / "raw").mkdir()
        (scene / "raw" / "i1.jpg").write_bytes(b"")
        (scene / "task" / "open_qa.json").write_text(
            json.dumps(
                {
                    "version": "metropolis-v3.0",
                    "metadata": {"type": "open_qa"},
                    "items": [{"image_id": "i1", "question": "Q?", "answer": "A"}],
                }
            )
        )
        out = tmp_path / "out"
        MetropolisV3_0ToVideoReasoningV1_0Converter().convert_scene(scene, out)
        ann = json.loads((out / "open_qa.json").read_text())
        item = ann["items"][0]
        assert "image_id" in item and "video_id" not in item
        assert item["image_id"] == "images/raw--i1.jpg"
        assert (out / "images" / "raw--i1.jpg").exists()


# ---------------------------------------------------------------------------
# Aggregation across multiple scenes
# ---------------------------------------------------------------------------


class TestAggregation:
    def test_items_aggregate_across_scenes(self, tmp_path):
        """convert_dataset merges items from multiple scenes into one annotation file per task type."""
        root = tmp_path / "root"
        for i in range(2):
            scene = root / f"scene_{i}"
            (scene / "task").mkdir(parents=True)
            (scene / "contextual").mkdir()
            (scene / "contextual" / f"video_v{i}.json").write_text(
                json.dumps({"metadata": {"type": "video"}, "video_id": f"v{i}", "format": "mp4"})
            )
            (scene / "raw").mkdir()
            (scene / "raw" / f"v{i}.mp4").write_bytes(b"")
            (scene / "task" / "open_qa.json").write_text(
                json.dumps(
                    {
                        "version": "metropolis-v3.0",
                        "metadata": {"type": "open_qa"},
                        "items": [{"video_id": f"v{i}", "question": "Q?", "answer": "A"}],
                    }
                )
            )
        out = tmp_path / "out"
        result = MetropolisV3_0ToVideoReasoningV1_0Converter().convert_dataset(root, out)
        assert result.is_success()
        assert result.samples_written == 2
        ann = json.loads((out / "open_qa.json").read_text())
        assert len(ann["items"]) == 2

    def test_no_scenes_returns_error(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        result = MetropolisV3_0ToVideoReasoningV1_0Converter().convert_dataset(empty, tmp_path / "out")
        assert not result.is_success()
        assert any("metropolis-v3.0 scenes" in e for e in result.errors)

    def test_same_basename_across_scenes_does_not_collide(self, tmp_path):
        """Regression: two scenes shipping ``camera_01.mp4`` in their raw/
        directories must land at distinct output files. The flat naming rule
        is "the source-relative path with ``/`` replaced by ``--``", so files
        from different scenes carry distinct prefixes.
        """
        root = tmp_path / "root"
        for category in ("transportation", "its_collision"):
            scene = root / category / "scene_001"
            (scene / "task").mkdir(parents=True)
            (scene / "contextual").mkdir()
            (scene / "contextual" / "video_camera_01.json").write_text(
                json.dumps({"metadata": {"type": "video"}, "video_id": "camera_01", "format": "mp4"})
            )
            (scene / "raw").mkdir()
            # Same basename, distinct content.
            (scene / "raw" / "camera_01.mp4").write_bytes(category.encode())
            (scene / "task" / "open_qa.json").write_text(
                json.dumps(
                    {
                        "version": "metropolis-v3.0",
                        "metadata": {"type": "open_qa"},
                        "items": [{"video_id": "camera_01", "question": "Q?", "answer": category}],
                    }
                )
            )

        out = tmp_path / "out"
        result = MetropolisV3_0ToVideoReasoningV1_0Converter().convert_dataset(root, out)
        assert result.is_success()
        assert result.samples_written == 2

        # Two distinct output files exist with the original content preserved,
        # both flat under videos/ (no recursive directory structure).
        f_transport = out / "videos" / "transportation--scene_001--raw--camera_01.mp4"
        f_collision = out / "videos" / "its_collision--scene_001--raw--camera_01.mp4"
        assert f_transport.read_bytes() == b"transportation"
        assert f_collision.read_bytes() == b"its_collision"

        # Item video_id values are likewise distinct.
        ann = json.loads((out / "open_qa.json").read_text())
        video_ids = sorted(item["video_id"] for item in ann["items"])
        assert video_ids == [
            "videos/its_collision--scene_001--raw--camera_01.mp4",
            "videos/transportation--scene_001--raw--camera_01.mp4",
        ]

    def test_naming_uniform_across_convert_scene_and_convert_dataset(self, tmp_path):
        """Both entry points apply the same flat naming rule: when the caller
        passes the scene path itself as the root, the relpath is
        ``raw/<file>`` → ``raw--<file>``. There is no scene-vs-dataset
        special-case in the rename helper.
        """
        scene = _make_scene(
            tmp_path,
            "open_qa",
            [{"video_id": "v1", "question": "Q?", "answer": "A"}],
        )

        for fn, label in (
            (lambda c, o: c.convert_scene(scene, o), "convert_scene"),
            (lambda c, o: c.convert_dataset(scene, o), "convert_dataset"),
        ):
            out = tmp_path / f"out_{label}"
            fn(MetropolisV3_0ToVideoReasoningV1_0Converter(), out)
            ann = json.loads((out / "open_qa.json").read_text())
            assert ann["items"][0]["video_id"] == "videos/raw--v1.mp4", label
            assert (out / "videos" / "raw--v1.mp4").exists(), label

    def test_dataset_passes_validator_after_flattening(self, tmp_path):
        """End-to-end: a multi-scene conversion with collision-prone basenames
        must still produce a dataset that passes the tao-vl-reason validator.
        The validator's media-existence check is the canary — if the flattened
        item path and the copy destination disagree, validation fails.
        """
        root = tmp_path / "root"
        for category in ("a", "b"):
            scene = root / category / "scene"
            (scene / "task").mkdir(parents=True)
            (scene / "raw").mkdir()
            (scene / "raw" / "shared.mp4").write_bytes(b"x")
            (scene / "task" / "open_qa.json").write_text(
                json.dumps(
                    {
                        "version": "metropolis-v3.0",
                        "metadata": {"type": "open_qa"},
                        "items": [{"video_id": "shared", "question": "Q?", "answer": "A"}],
                    }
                )
            )

        out = tmp_path / "out"
        MetropolisV3_0ToVideoReasoningV1_0Converter().convert_dataset(root, out)
        result = VideoReasoningV1_0Validator().validate_dataset(out)
        assert result.is_valid(), f"errors: {result.errors}\nwarnings: {result.warnings}"


# ---------------------------------------------------------------------------
# --no-copy-media
# ---------------------------------------------------------------------------


class TestNoCopyMedia:
    def test_default_sets_media_root_to_source_and_relative_item_paths(self, tmp_path):
        """--no-copy-media: media_root = absolute source; item paths are relative."""
        scene = _make_scene(
            tmp_path,
            "open_qa",
            [{"video_id": "v1", "question": "Q?", "answer": "A"}],
        )
        out = tmp_path / "out"
        MetropolisV3_0ToVideoReasoningV1_0Converter().convert_scene(scene, out, copy_media=False)
        ann = json.loads((out / "open_qa.json").read_text())
        # media_root is the absolute source scene path.
        assert ann["media_root"] == str(scene.resolve())
        # Item path is bare-relative under that root.
        assert ann["items"][0]["video_id"] == "raw/v1.mp4"
        # No media subdirectory was created in the output.
        assert not (out / "videos").exists()

    def test_emit_media_root_nulls_media_root(self, tmp_path):
        """--no-copy-media + emit_media_root_as_null=True: media_root is null,
        item paths are still source-relative (consumer overrides at load time)."""
        scene = _make_scene(
            tmp_path,
            "open_qa",
            [{"video_id": "v1", "question": "Q?", "answer": "A"}],
        )
        out = tmp_path / "out"
        MetropolisV3_0ToVideoReasoningV1_0Converter().convert_scene(
            scene, out, copy_media=False, emit_media_root_as_null=True
        )
        ann = json.loads((out / "open_qa.json").read_text())
        assert ann["media_root"] is None
        assert ann["items"][0]["video_id"] == "raw/v1.mp4"
        assert not (out / "videos").exists()

    def test_emit_media_root_ignored_when_copy_media(self, tmp_path):
        """emit_media_root_as_null is meaningful only with copy_media=False;
        in copy mode media_root is already null and items reference output paths."""
        scene = _make_scene(
            tmp_path,
            "open_qa",
            [{"video_id": "v1", "question": "Q?", "answer": "A"}],
        )
        out = tmp_path / "out"
        MetropolisV3_0ToVideoReasoningV1_0Converter().convert_scene(
            scene, out, copy_media=True, emit_media_root_as_null=True
        )
        ann = json.loads((out / "open_qa.json").read_text())
        assert ann["media_root"] is None
        assert ann["items"][0]["video_id"] == "videos/raw--v1.mp4"


# ---------------------------------------------------------------------------
# Real example: its_collision end-to-end
# ---------------------------------------------------------------------------
