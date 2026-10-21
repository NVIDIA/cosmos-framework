# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Focused Skill-wrapper tests; run in the supported Cosmos Framework environment."""

from __future__ import annotations

import json
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from typing import Any

from paibench_local_bundle import (
    file_hash,
    prepare_custom,
    probe,
    training_caption,
    write_json,
)


class TransferBundleTest(unittest.TestCase):
    def test_caption_preserves_nested_fields_and_rejects_empty_dense(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "caption.json"
            caption = {
                "subjects": [{"description": "Robot", "details": {"color": "blue"}}],
                "temporal_caption": "The robot moves.",
                "fps": 30,
            }
            write_json(path, caption)
            self.assertEqual(training_caption(path), {key: value for key, value in caption.items() if key != "fps"})
            write_json(path, {"temporal_caption": " "})
            with self.assertRaisesRegex(ValueError, "empty"):
                training_caption(path)

    @staticmethod
    def video(
        path: Path, width: int = 638, height: int = 480, frames: int = 9, start_pts: int = 0, frame_step: int = 512
    ) -> None:
        import av
        import numpy as np

        pixels = np.zeros((height, width, 3), dtype=np.uint8)  # [H,W,3]
        pixels[:, :, 2] = 180  # [H,W]
        with av.open(str(path), "w") as container:
            stream = container.add_stream("libx264", rate=30)
            stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
            stream.codec_context.thread_count = 1
            stream.time_base = stream.codec_context.time_base = Fraction(1, 15360)
            for index in range(frames):
                frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
                frame.pts = start_pts + index * frame_step
                frame.time_base = Fraction(1, 15360)
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)

    def test_alignment_ignores_container_start_but_preserves_frame_timing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.video(root / "control.mp4", width=32, height=32)
            self.video(root / "rgb.mp4", width=32, height=32, start_pts=507)
            self.video(root / "drift.mp4", width=32, height=32, frame_step=513)
            self.assertEqual(probe(root / "rgb.mp4"), probe(root / "control.mp4"))
            self.assertNotEqual(probe(root / "drift.mp4")["timestamps"], probe(root / "control.mp4")["timestamps"])

    def test_custom_native_builder_preserves_caption_and_excludes_invalid_samples(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokenizer = root / "tokenizer"
            tokenizer.mkdir()
            for name in ("vocab.json", "merges.txt", "tokenizer_config.json"):
                (tokenizer / name).write_text("{}\n")
            self.video(root / "valid.mp4")
            self.video(root / "short.mp4", frames=5)
            self.video(root / "wide.mp4", width=960, height=480)
            self.video(root / "long.mp4", width=32, height=32, frames=601)
            caption = {
                "subjects": [{"description": "Blue square", "extra": {"preserve": [1, 2]}}],
                "temporal_caption": "A blue square remains still.",
                "fps": 30,
            }
            write_json(root / "caption.json", caption)
            write_json(root / "invalid.json", {"temporal_caption": ""})
            base: dict[str, Any] = {
                "rgb": "valid.mp4",
                "depth": "valid.mp4",
                "seg": "valid.mp4",
                "caption_path": "caption.json",
                "caption_start": 0,
                "caption_end": 9,
            }
            rows = [
                {**base, "identity": "valid"},
                {**base, "identity": "bad_interval", "caption_end": 8},
                {**base, "identity": "bad_caption", "caption_path": "invalid.json"},
                {**base, "identity": "bad_alignment", "depth": "short.mp4"},
                {
                    **base,
                    "identity": "short",
                    "rgb": "short.mp4",
                    "depth": "short.mp4",
                    "seg": "short.mp4",
                    "caption_end": 5,
                },
                {**base, "identity": "bad_aspect", "rgb": "wide.mp4", "depth": "wide.mp4", "seg": "wide.mp4"},
                {
                    **base,
                    "identity": "long",
                    "rgb": "long.mp4",
                    "depth": "long.mp4",
                    "seg": "long.mp4",
                    "caption_end": 601,
                },
            ]
            manifest_path = root / "train.jsonl"
            manifest_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            output = root / "bundle"
            prepare_custom(manifest_path, tokenizer, output, "test_transfer")
            records = json.loads((output / "samples.json").read_text())
            self.assertEqual([record["identity"] for record in records], ["valid"])
            record = records[0]
            self.assertEqual(
                json.loads(record["entry"]["caption_json"]),
                {key: value for key, value in caption.items() if key != "fps"},
            )
            self.assertEqual(record["entry"]["aspect_ratio"], "4,3")
            self.assertEqual((record["entry"]["source_width"], record["entry"]["source_height"]), (638, 480))
            self.assertEqual(file_hash(output / record["assets"]["rgb"]), file_hash(root / "valid.mp4"))
            manifest = json.loads((output / "manifest.json").read_text())
            transform = manifest["transforms"][record["stratum"]]
            self.assertEqual(
                (transform["resolution"], transform["cfg_dropout_rate"], transform["dense_caption_probability"]),
                ("480", 0.0, 0.07),
            )
            summary = json.loads((root / "bundle.summary.json").read_text())
            self.assertEqual(
                summary["excluded"],
                {
                    "caption_interval": 1,
                    "invalid_caption": 1,
                    "unaligned_control": 1,
                    "shorter_than_9_frames": 1,
                    "aspect_distortion": 1,
                    "stage0_720_window": 1,
                },
            )
            for relative, expected in manifest["provenance"]["sha256"].items():
                self.assertEqual(file_hash(output / relative), expected)
            manifest_path.write_text(json.dumps({**base, "identity": "missing", "rgb": "absent.mp4"}) + "\n")
            with self.assertRaisesRegex(FileNotFoundError, "missing: missing rgb file: .*absent.mp4"):
                prepare_custom(manifest_path, tokenizer, root / "missing-bundle", "test_transfer")
            self.assertFalse((root / "missing-bundle").exists())


if __name__ == "__main__":
    unittest.main()
