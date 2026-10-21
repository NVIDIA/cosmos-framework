# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""VAE compilation flags preserve defaults and configure the actual vision tokenizer."""

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import tyro

from cosmos_framework.inference.common.args import ParallelismOverrides
from cosmos_framework.inference.inference import _configure_vae_compile
from cosmos_framework.model.generator.tokenizers.wan2pt2_vae_4x16x16 import Wan2pt2VAEInterface


class VaeCompileSetupTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = Mock(spec=Wan2pt2VAEInterface)
        self.model = SimpleNamespace(tokenizer_vision_gen=self.tokenizer, config=SimpleNamespace(resolution="720"))

    def settings(self, encode=False, decode=False, separate=False, resolution="480", aspect_ratio="16,9"):
        return SimpleNamespace(
            use_vae_encode_compile=encode,
            use_vae_decode_compile=decode,
            use_separate_pipeline_vision_decode_gpu=separate,
            sample_overrides=SimpleNamespace(resolution=resolution, aspect_ratio=aspect_ratio),
            output_dir=Path("outputs/vae-compile"),
        )

    def test_cli_flags_round_trip_independently_of_diffusion_compile(self):
        defaults = ParallelismOverrides().build_parallelism()
        self.assertFalse(defaults.use_vae_encode_compile)
        self.assertFalse(defaults.use_vae_decode_compile)
        for flags in (
            [],
            ["--use-vae-encode-compile"],
            ["--use-vae-decode-compile"],
            ["--use-vae-encode-compile", "--use-vae-decode-compile"],
        ):
            with self.subTest(flags=flags):
                overrides = tyro.cli(ParallelismOverrides, args=["--no-use-torch-compile", *flags])
                built = overrides.build_parallelism()
                self.assertFalse(built.use_torch_compile)
                self.assertEqual(built.use_vae_encode_compile, "--use-vae-encode-compile" in flags)
                self.assertEqual(built.use_vae_decode_compile, "--use-vae-decode-compile" in flags)
                self.assertEqual(ParallelismOverrides(**built.model_dump()).build_parallelism(), built)

    def test_disabled_flags_leave_tokenizers_untouched(self):
        self.model.tokenizer_vision_gen = None
        _configure_vae_compile(self.model, self.settings())

    def test_encode_only_uses_cli_shape_and_output_directory(self):
        _configure_vae_compile(self.model, self.settings(encode=True, aspect_ratio="1,1"))
        self.tokenizer.compile_encode.assert_called_once_with(
            ["480"], output_dir="outputs/vae-compile", aspect_ratio="1,1"
        )
        self.tokenizer.active_decoder.enable_decoder_compile.assert_not_called()

    def test_decode_only_targets_active_decoder_and_allows_spare_gpu(self):
        _configure_vae_compile(self.model, self.settings(decode=True, separate=True))
        self.tokenizer.active_decoder.enable_decoder_compile.assert_called_once_with()
        self.tokenizer.compile_encode.assert_not_called()

    def test_both_flags_use_model_resolution_when_no_cli_shape_is_set(self):
        _configure_vae_compile(self.model, self.settings(encode=True, decode=True, resolution=None, aspect_ratio=None))
        self.tokenizer.compile_encode.assert_called_once_with(
            ["720"], output_dir="outputs/vae-compile", aspect_ratio="16,9"
        )
        self.tokenizer.active_decoder.enable_decoder_compile.assert_called_once_with()

    def test_encoder_spare_gpu_conflict_fails_before_any_compilation(self):
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            ParallelismOverrides(
                use_vae_encode_compile=True, use_separate_pipeline_vision_decode_gpu=True
            ).build_parallelism()
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            _configure_vae_compile(self.model, self.settings(encode=True, decode=True, separate=True))
        self.assertEqual(self.tokenizer.mock_calls, [])

    def test_unsupported_tokenizer_rejected_for_either_flag(self):
        for tokenizer in (None, object()):
            for encode, decode in ((True, False), (False, True)):
                with self.subTest(tokenizer=tokenizer, encode=encode, decode=decode):
                    self.model.tokenizer_vision_gen = tokenizer
                    with self.assertRaisesRegex(ValueError, "Wan 2.2"):
                        _configure_vae_compile(self.model, self.settings(encode=encode, decode=decode))

    def test_unsupported_active_decoder_rejected(self):
        self.tokenizer.active_decoder = object()
        with self.assertRaisesRegex(ValueError, "active vision decoder"):
            _configure_vae_compile(self.model, self.settings(decode=True))


if __name__ == "__main__":
    unittest.main()
