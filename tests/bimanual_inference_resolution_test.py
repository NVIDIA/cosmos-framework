# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""CPU regressions for native recorded-action sizing and temporal alignment."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec("torch"), "Requires the inference dependencies")
class InferenceResolutionTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        import torch

        self.np = np
        self.torch = torch
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.actions = [[float(index)] * 29 for index in range(4)]
        self.actions_path = self.root / "actions.json"
        self.actions_path.write_text(json.dumps(self.actions))

    def batch(self, observation, content_hw, *, content_crop_box=None):
        from cosmos_framework.inference.action import get_action_sample_data
        from cosmos_framework.inference.args import ModelMode

        result = get_action_sample_data(
            model_config=SimpleNamespace(input_video_key="video"),
            batch_size=1,
            prompt="Move both arms.",
            vision_path=observation,
            model_mode=ModelMode.FORWARD_DYNAMICS,
            action_path=self.actions_path,
            domain_name="agibotworld",
            resolution="480",
            content_crop_box=content_crop_box,
            action_chunk_size=4,
            max_action_dim=64,
            fps=30,
            device="cpu",
        )
        video = result["video"][0][0]
        self.assertEqual(tuple(video.shape), (3, 5, 480, 832))
        # Latent cropping reads the last two entries, not just the canvas size.
        self.assertEqual(result["image_size"].tolist(), [[480, 832, *content_hw]])
        self.assertEqual(result["action"][0][0][:, :29].tolist(), self.actions)
        self.assertEqual(int(result["raw_action_dim"][0]), 29)
        self.assertEqual(int(result["domain_id"][0]), 15)
        self.assertEqual(result["sequence_plan"][0].condition_frame_indexes_vision, [0])
        return result

    def test_smaller_image_keeps_native_pixels_with_bottom_right_padding(self):
        from PIL import Image

        pixels = self.np.random.default_rng(42).integers(0, 256, (360, 640, 3), dtype=self.np.uint8)
        image = self.root / "initial.png"
        Image.fromarray(pixels).save(image)
        result = self.batch(image, (360, 640))
        video = result["video"][0][0]
        expected = self.torch.from_numpy(pixels).permute(2, 0, 1)
        for frame in range(5):
            self.assertTrue(self.torch.equal(video[:, frame, :360, :640], expected))
        # Reflection starts beyond the content's right and bottom boundaries;
        # it must not shift or resample the conditioning image.
        self.assertTrue(self.torch.equal(video[:, 0, :360, 640], expected[:, :, -2]))
        self.assertTrue(self.torch.equal(video[:, 0, 360, :640], expected[:, -2, :]))

    def test_native_content_metadata_crops_latents_before_generation(self):
        from PIL import Image

        from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel

        image = self.root / "initial.png"
        Image.new("RGB", (640, 360), color=(32, 64, 96)).save(image)
        result = self.batch(image, (360, 640))
        latent = self.torch.arange(30 * 52).reshape(1, 1, 1, 30, 52)
        # Exercise the production crop without constructing a model or loading
        # a checkpoint. The tokenizer's spatial factor is 16 for this recipe.
        model = SimpleNamespace(tokenizer_vision_gen=SimpleNamespace(spatial_compression_factor=16))
        (cropped,) = OmniMoTModel._remove_padding_from_latent(model, [latent], [result["image_size"]])
        self.assertEqual(tuple(cropped.shape), (1, 1, 1, 22, 40))
        self.assertTrue(self.torch.equal(cropped, latent[..., :22, :40]))

    def test_larger_image_is_downscaled_without_distorting_aspect_ratio(self):
        from PIL import Image

        # A 16:9 input fits inside the 832x480 canvas at 832x468, with padding
        # below it. It must not be stretched or center-cropped to 832x480.
        image = self.root / "initial.png"
        Image.new("RGB", (1280, 720), color=(32, 64, 96)).save(image)
        self.batch(image, (468, 832))

    def test_existing_target_sized_image_preserves_pixels(self):
        from PIL import Image

        pixels = self.np.random.default_rng(42).integers(0, 256, (480, 832, 3), dtype=self.np.uint8)
        image = self.root / "initial.png"
        Image.fromarray(pixels).save(image)
        video = self.batch(image, (480, 832))["video"][0][0]
        expected = self.torch.from_numpy(pixels).permute(2, 0, 1)
        self.assertTrue(self.torch.equal(video[:, 0], expected))
        self.assertTrue(self.torch.equal(video[:, -1], expected))

    def test_declared_content_crop_removes_transport_canvas_before_padding(self):
        from PIL import Image

        pixels = self.np.full((480, 832, 3), 255, dtype=self.np.uint8)
        content = self.np.random.default_rng(42).integers(0, 128, (360, 640, 3), dtype=self.np.uint8)
        pixels[:360, :640] = content
        image = self.root / "transport_canvas.png"
        Image.fromarray(pixels).save(image)
        raw_image = self.root / "raw_content.png"
        Image.fromarray(content).save(raw_image)

        cropped_batch = self.batch(image, (360, 640), content_crop_box=(0, 0, 640, 360))
        raw_batch = self.batch(raw_image, (360, 640))
        video = cropped_batch["video"][0][0]
        expected = self.torch.from_numpy(content).permute(2, 0, 1)
        self.assertTrue(self.torch.equal(video, raw_batch["video"][0][0]))
        self.assertTrue(self.torch.equal(cropped_batch["image_size"], raw_batch["image_size"]))
        self.assertTrue(self.torch.equal(video[:, 0, :360, :640], expected))
        # The white transport canvas is gone; padding is reflected from the
        # declared content instead of becoming part of the visual condition.
        self.assertTrue(self.torch.equal(video[:, 0, :360, 640], expected[:, :, -2]))
        self.assertLess(int(video.max()), 128)

    def test_smaller_mp4_preserves_first_frame_and_action_alignment(self):
        import av

        from cosmos_framework.inference.vision import read_media_frames

        path = self.root / "observations.mp4"
        with av.open(str(path), mode="w") as container:
            stream = container.add_stream("libx264", rate=30)
            stream.width, stream.height, stream.pix_fmt = 640, 360, "yuv420p"
            stream.options = {"crf": "0", "preset": "ultrafast"}
            for index in range(7):
                pixels = self.np.full((360, 640, 3), index * 30, dtype=self.np.uint8)
                frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        # Use the same decoder for the reference: RGB conversion can round
        # differently across backends, independently of spatial preprocessing.
        decoded, _ = read_media_frames(path, max_frames=5, video_decoder="pyav")
        self.assertEqual(tuple(decoded.shape), (3, 5, 360, 640))
        # Keep this spatial/action-alignment regression independent of the
        # host's TorchCodec shared-library installation. Decoder integration is
        # covered separately; both calls here intentionally use PyAV.
        with patch(
            "cosmos_framework.inference.action.read_media_frames",
            side_effect=lambda media_path, max_frames: read_media_frames(
                media_path, max_frames, video_decoder="pyav"
            ),
        ):
            video = self.batch(path, (360, 640))["video"][0][0]
        # The five retained observations must stay paired with the original
        # four actions, with their native pixels unchanged before padding.
        for index in range(5):
            expected = decoded[:, index]
            self.assertTrue(self.torch.equal(video[:, index, :360, :640], expected))
            observed = video[:, index, 240, 416].to(self.torch.int16)
            self.assertLessEqual(int((observed - index * 30).abs().max()), 2)


if __name__ == "__main__":
    unittest.main()
