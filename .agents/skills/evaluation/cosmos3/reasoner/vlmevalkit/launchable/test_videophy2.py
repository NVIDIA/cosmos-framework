# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only scoring, media integrity, and audited-response boundaries."""

import io
import json
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import videophy2


def response(output="4", finish="stop", count=784):
    groups, remainder = divmod(count, 49)
    sizes = [49] * groups + ([remainder] if remainder else [])
    return {"prompt_token_ids": [token for size in sizes for token in [151652, *([151656] * size), 151653]],
            "choices": [{"message": {"content": output}, "finish_reason": finish}]}


def rating_row(sa_truth, pc_truth, sa_output, pc_output):
    return {"truth": {"sa": sa_truth, "pc": pc_truth},
            "responses": {task: {"score": videophy2.official_parse(output)[0]}
                          for task, output in (("sa", sa_output), ("pc", pc_output))}}


class ScoringTests(unittest.TestCase):
    def test_official_parser_retains_order_and_invalid_zero(self):
        cases = (("", (0, True)), ("invalid", (0, True)), ("5", (5, False)),
                 ("five", (5, False)), ("one or five", (1, False)),
                 ("0", (0, False)), ("9", (0, True)))
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(videophy2.official_parse(text), expected)

    def test_sa_and_pc_are_independent_and_mean_is_correlation_not_accuracy(self):
        rows = [rating_row(1, 1, "1", "5"), rating_row(3, 3, "3", "3"), rating_row(5, 5, "5", "1")]
        scores = videophy2.correlations(rows)
        self.assertAlmostEqual(scores["sa_pearson_x100"], 100)
        self.assertAlmostEqual(scores["pc_pearson_x100"], -100)
        self.assertAlmostEqual(scores["mean_sa_pc_pearson_x100"], 0)

    def test_blank_predictions_stay_in_denominator(self):
        scores = videophy2.correlations([rating_row(1, 1, "1", "1"), rating_row(3, 3, "3", "3"), rating_row(5, 5, "", "")])
        self.assertEqual(scores["rows"], 3)
        self.assertLess(scores["mean_sa_pc_pearson_x100"], 0)

    def test_undefined_correlations_are_json_null(self):
        scores = videophy2.correlations([rating_row(4, 2, "5", "4"), rating_row(2, 2, "1", "4")])
        self.assertAlmostEqual(scores["sa_pearson_x100"], 100)
        self.assertIsNone(scores["pc_pearson_x100"])
        self.assertIsNone(scores["mean_sa_pc_pearson_x100"])
        json.dumps(scores, allow_nan=False)

    def test_truncated_or_blank_completed_transport_is_not_dropped(self):
        self.assertEqual(videophy2.audit_response(response("", "length")), ("", "length", 784))
        self.assertEqual(videophy2.audit_response(response(None)), ("", "stop", 784))

    def test_four_frame_and_incomplete_transport_are_rejected(self):
        for raw in (response(count=98), response(finish=None), response(finish="error"),
                    {"choices": []}, {"prompt_token_ids": [151656] * 784, "choices": [{"finish_reason": "stop"}]}):
            with self.subTest(raw_shape=list(raw)):
                with self.assertRaises(ValueError):
                    videophy2.audit_response(raw)

    def test_equal_total_tokens_cannot_hide_per_group_pixel_mismatch(self):
        raw = response()
        raw["prompt_token_ids"] = [token for size in [48, 50] + [49] * 14
                                   for token in [151652, *([151656] * size), 151653]]
        self.assertEqual(raw["prompt_token_ids"].count(151656), 784)
        with self.assertRaises(ValueError):
            videophy2.audit_response(raw)

    def test_unbounded_single_run_does_not_prove_32_frame_slots(self):
        raw = response()
        raw["prompt_token_ids"] = [151656] * 784
        with self.assertRaises(ValueError):
            videophy2.audit_response(raw)


class MediaTests(unittest.TestCase):
    def test_archive_mapping_uses_exact_path_then_model_not_caption(self):
        row = {"video_url": "https://public.example/hunyuan_xdit/video%3Bname.mp4", "model_name": "hunyuan"}
        self.assertEqual(videophy2.archive_match(row, ["hunyuan_xdit/video;name.mp4", "wan/video;name.mp4"]), "hunyuan_xdit/video;name.mp4")
        row["video_url"] = "https://public.example/unavailable/video.mp4"
        self.assertEqual(videophy2.archive_match(row, ["hunyuan_xdit/video.mp4", "wan/video.mp4"]), "hunyuan_xdit/video.mp4")
        self.assertIsNone(videophy2.archive_match(row, ["hunyuan_xdit/different.mp4"]))
        with self.assertRaises(ValueError):
            videophy2.archive_match(row, ["hunyuan_xdit/video.mp4", "hunyuan_other/video.mp4"])

    def test_safe_extraction_rejects_traversal_and_symlink(self):
        for name, mode in (("../escape.mp4", 0), ("/absolute.mp4", 0), ("safe\\escape.mp4", 0), ("link.mp4", stat.S_IFLNK | 0o777)):
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w") as archive:
                info = zipfile.ZipInfo(name)
                info.external_attr = mode << 16
                archive.writestr(info, b"outside")
            buffer.seek(0)
            with zipfile.ZipFile(buffer) as archive, self.assertRaises(ValueError):
                videophy2._archive_files(archive)

    def test_all_selected_media_must_have_matching_bytes_and_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "media").mkdir()
            (root / "media" / "00000.mp4").write_bytes(b"video")
            row = {"video_url": "https://public.example/video.mp4"}
            entry = {"index": 0, "sample_id": "videophy2_test:00000", "video_url": row["video_url"],
                     "bytes": 5, "sha256": videophy2._sha256(root / "media" / "00000.mp4"),
                     "dataset_sha256": videophy2.DATASET_SHA256, "archive_entry": None}
            (root / "media-manifest.jsonl").write_text(json.dumps(entry) + "\n")
            self.assertEqual(len(videophy2._preflight(root, [(0, row)])), 1)
            with self.assertRaises(ValueError):
                videophy2._preflight(root, [(0, row), (1, row)])
            (root / "media" / "00000.mp4").write_bytes(b"other")
            with self.assertRaises(ValueError):
                videophy2._preflight(root, [(0, row)])

    def test_worker_bounds_and_model_revision_do_not_infer_endpoint_identity(self):
        with patch.dict("os.environ", {}, clear=True):
            for workers in (0, 5, 20, 1.5, True):
                with self.subTest(workers=workers), self.assertRaises(ValueError):
                    videophy2._settings(workers)
            _, _, settings = videophy2._settings(4)
            self.assertEqual(settings["workers"], 4)
            self.assertEqual(settings["model_revision"], "unknown")


if __name__ == "__main__":
    unittest.main()
