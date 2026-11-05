# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Shared physical LiDAR controls retain values and a usable camera timeline."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from cosmos_framework.inference.lidar_transfer import (
    _tensor_lidar_frame_ids,
    load_lidar_transfer_control,
    save_lidar_range_view,
)
from cosmos_framework.model.generator.tokenizers.lidar.range_projection import LidarRangeProjectionConfig


class SharedLidarInputTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "control.safetensors"
        self.projection = LidarRangeProjectionConfig(
            native_height=8, native_width=16, semantic_height=8, semantic_width=16, model_width=16
        )
        self.frames = torch.ones(3, 7, 8, 16)
        self.frames[0] *= 20
        self.frames[1] *= 0.25

    def load(self, frames, count=5):
        save_file({"frames": frames}, self.path)
        return load_lidar_transfer_control(self.path, max_frames=count, range_projection=self.projection)

    def test_load_preserves_physical_values_and_truncates(self):
        clip, ids = self.load(self.frames)
        self.assertTrue(torch.equal(clip, self.frames[:, :5]))
        self.assertEqual(clip.dtype, torch.float32)
        self.assertIsNone(ids)
        full, _ = self.load(self.frames, None)
        self.assertTrue(torch.equal(full, self.frames))

    def test_rejects_invalid_shape_dtype_units_or_short_input(self):
        invalid = [self.frames.double(), self.frames[:2], self.frames[:, :2], self.frames[:, :, :, :8]]
        negative = self.frames.clone()
        negative[0, 0, 0, 0] = -1
        invalid.append(negative)
        normalized = self.frames.clone()
        normalized[1] *= 10
        invalid.append(normalized)
        nan = self.frames.clone()
        nan[0, 0, 0, 0] = float("nan")
        invalid.append(nan)
        for value in invalid:
            with self.subTest(shape=value.shape, dtype=value.dtype), self.assertRaises(ValueError):
                self.load(value)

    def test_rejects_output_schema_as_control(self):
        save_file({"range_m": self.frames[0]}, self.path)
        with self.assertRaisesRegex(ValueError, "physical 'frames'"):
            load_lidar_transfer_control(self.path, max_frames=5, range_projection=self.projection)

    def test_tensor_timeline_reaches_output_saver(self):
        clip, _ = self.load(self.frames)
        ids = _tensor_lidar_frame_ids(5, camera_fps=30, lidar_fps=10)
        self.assertEqual(ids.tolist(), [0, 3, 6, 9, 12])
        output = Path(self.directory.name) / "lidar.safetensors"
        save_lidar_range_view(clip.unsqueeze(0), output, range_projection=self.projection, frame_ids=ids)
        with safe_open(output, framework="pt") as handle:
            self.assertEqual(set(handle.keys()), {"range_m", "intensity", "frame_ids"})
            self.assertTrue(torch.equal(handle.get_tensor("frame_ids"), ids))
            self.assertEqual(handle.get_tensor("range_m").shape, (5, 8, 16))

    def test_nonintegral_or_invalid_clocks_fail_before_generation(self):
        for camera_fps, lidar_fps in [(25, 10), (0, 10), (30, float("nan")), (30, float("inf"))]:
            with self.subTest(camera_fps=camera_fps, lidar_fps=lidar_fps), self.assertRaises(ValueError):
                _tensor_lidar_frame_ids(5, camera_fps=camera_fps, lidar_fps=lidar_fps)


@pytest.mark.parametrize("suffix,camera_fps", [(".pt", 25), (".pt", 30), (".safetensors", 30)])
def test_joint_generation_only_synthesizes_ids_for_shared_safetensors(
    monkeypatch: pytest.MonkeyPatch, suffix: str, camera_fps: int
) -> None:
    from cosmos_framework.inference import inference, lidar_transfer

    class ReachedPromptPreparation(Exception):
        pass

    sample = SimpleNamespace(
        is_joint_cam_lidar_transfer=True,
        resolution="480",
        vision_size=(8, 4),
        num_frames=7,
        fps=camera_fps,
        lidar=object(),
        multiview=SimpleNamespace(
            views=[SimpleNamespace(control_path="camera.mp4")],
            load_caption_from_data=False,
            has_condition_vision=False,
        ),
    )
    sample.model_copy = lambda update: SimpleNamespace(**(vars(sample) | update))
    model = SimpleNamespace(
        tokenizer_vision_gen=object(),
        config=SimpleNamespace(
            diffusion_expert_config=SimpleNamespace(num_view_embeddings=0),
            tokenizer=SimpleNamespace(temporal_compression_factor=2),
            lidar_fps=10,
        ),
    )
    monkeypatch.setattr(lidar_transfer, "load_multiview_media_pixels", lambda *args, **kwargs: torch.zeros(3, 7, 4, 8))
    monkeypatch.setattr(
        lidar_transfer,
        "load_lidar_control",
        lambda *args, **kwargs: (Path("control" + suffix), torch.ones(3, kwargs["max_frames"], 4, 8), None),
    )
    timeline = Mock(wraps=lidar_transfer._tensor_lidar_frame_ids)
    monkeypatch.setattr(lidar_transfer, "_tensor_lidar_frame_ids", timeline)
    monkeypatch.setattr(inference, "_get_prompt_sample_data", Mock(side_effect=ReachedPromptPreparation))

    # Run the real input/frontend path, stopping before CUDA batch/model work.
    with pytest.raises(ReachedPromptPreparation):
        lidar_transfer.generate_joint_cam_lidar_sample(sample, model)
    assert timeline.call_count == (1 if suffix == ".safetensors" else 0)


if __name__ == "__main__":
    unittest.main()
