# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Check local asset binding, parent selection and native CLI argument parsing."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import qa_transfer_stage as qa
from hydra.core.override_parser.overrides_parser import OverridesParser


class LocalStageTest(unittest.TestCase):
    temporary: tempfile.TemporaryDirectory[str]
    root: Path
    env: dict[str, str]

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.env = {}
        framework = self.root / "framework"
        (framework / qa.LOCAL_CONFIG).parent.mkdir(parents=True)
        (framework / qa.LOCAL_CONFIG).write_text("# Native config fixture\n")
        self.env["FRAMEWORK_DIR"] = str(framework)
        for name in ("BASE_CKPT_ROOT", "STAGE0_CKPT_ROOT", "STAGE1_CKPT_ROOT", "STAGE2_CKPT_ROOT"):
            path = self.root / name
            (path / "model").mkdir(parents=True)
            (path / "model/.metadata").write_bytes(b"metadata fixture")
            self.env[name] = str(path)
        nano = self.root / "nano"
        (nano / "text_tokenizer").mkdir(parents=True)
        self.env["NANO_HF_DIR"] = str(nano)
        bundle = self.root / "bundle with spaces,=[]"
        bundle.mkdir()
        for name in ("manifest.json", "samples.json"):
            (bundle / name).write_text("{}\n")
        self.env["TRANSFER_BUNDLE_PATH"] = str(bundle)
        for name in ("VIDEO_VAE_PATH", "AUDIO_VAE_PATH", "AUDIO_VAE_CONFIG_PATH"):
            path = self.root / name
            path.write_text("fixture\n")
            self.env[name] = str(path)

    def parse(self, stage: int) -> dict[str, object]:
        arguments = qa.training_arguments(stage)
        self.assertEqual(arguments[:2], [f"--config={qa.LOCAL_CONFIG}", "--"])
        # Use Hydra's real parser so quoting regressions fail before model imports.
        return {
            override.key_or_group: override.value()
            for override in OverridesParser.create().parse_overrides(arguments[2:])
        }

    def test_native_stages_bind_local_assets_and_their_own_parents(self) -> None:
        with patch.dict(os.environ, self.env, clear=True):
            for stage in range(4):
                with self.subTest(stage=stage):
                    values = self.parse(stage)
                    self.assertEqual(values["experiment"], qa.EXPERIMENTS[stage])
                    self.assertEqual(values["model.config.tokenizer.vae_path"], self.env["VIDEO_VAE_PATH"])
                    self.assertEqual(values["model.config.vlm_config.tokenizer.config_variant"], "hf")
                    self.assertFalse(values["model.config.vlm_config.pretrained_weights.enabled"])
                    self.assertFalse(values["checkpoint.load_training_state"])
                    self.assertEqual(values["checkpoint.load_ema_to_reg"], stage in (1, 2))
                    bundles = {key: value for key, value in values.items() if key.endswith(".bundle_path")}
                    streams = (
                        ("video_transfer_4modality_data_480", "video_transfer_4modality_data_720")
                        if stage == 0
                        else ("local_data",)
                    )
                    self.assertEqual(
                        bundles,
                        {
                            f"dataloader_train.dataloaders.{stream}.dataloader.dataset.bundle_path": self.env[
                                "TRANSFER_BUNDLE_PATH"
                            ]
                            for stream in streams
                        },
                    )
                    if stage < 3:
                        key = ("BASE_CKPT_ROOT", "STAGE0_CKPT_ROOT", "STAGE1_CKPT_ROOT")[stage]
                        self.assertEqual(values["checkpoint.load_path"], self.env[key])
                    if stage == 0:
                        self.assertEqual(values["model.config.sound_tokenizer.avae_path"], self.env["AUDIO_VAE_PATH"])
                        self.assertNotIn("checkpoint.keys_to_skip_loading", values)
                    elif stage == 2:
                        self.assertEqual(
                            values["model.config.teacher_forcing_dcm_teacher_load_from.load_path"],
                            self.env["STAGE1_CKPT_ROOT"] + "/model",
                        )
                    elif stage == 3:
                        self.assertEqual(values["checkpoint.load_path"], "")
                        self.assertEqual(
                            values["model.config.student_load_from.load_path"], self.env["STAGE2_CKPT_ROOT"] + "/model"
                        )
                        self.assertEqual(
                            values["model.config.teacher_load_from.load_path"], self.env["STAGE0_CKPT_ROOT"] + "/model"
                        )
                    self.assertFalse(any("dropout" in key or "resolution" in key or "chunk" in key for key in values))

    def test_asset_symlinks_keep_filenames_for_native_format_detection(self) -> None:
        snapshot = self.root / "snapshots" / "revision"
        snapshot.mkdir(parents=True)
        blobs = self.root / "blobs"
        blobs.mkdir()
        assets = {
            "VIDEO_VAE_PATH": ("model.config.tokenizer.vae_path", "Wan2.2_VAE.pth"),
            "AUDIO_VAE_PATH": ("model.config.sound_tokenizer.avae_path", "audio.ckpt"),
            "AUDIO_VAE_CONFIG_PATH": ("model.config.sound_tokenizer.avae_config_path", "audio.json"),
        }
        for index, (name, (_, filename)) in enumerate(assets.items()):
            blob = blobs / f"{index:064x}"
            blob.write_text("fixture\n")
            link = snapshot / filename
            link.symlink_to(blob)
            self.env[name] = os.path.relpath(link) if name == "AUDIO_VAE_PATH" else str(link)
        with patch.dict(os.environ, self.env, clear=True):
            values = self.parse(0)
        for name, (key, filename) in assets.items():
            with self.subTest(asset=name):
                self.assertEqual(values[key], str(snapshot / filename))

    def test_model_only_stage0_skips_only_ema_without_disabling_strict_loading(self) -> None:
        with patch.dict(os.environ, {**self.env, "TRANSFER_QA_BASE_MODEL_ONLY": "1"}, clear=True):
            values = self.parse(0)
        self.assertEqual(values["checkpoint.keys_to_skip_loading"], ["net_ema."])
        self.assertNotIn("checkpoint.strict_resume", values)
        self.assertEqual(values["model.config.ema.iteration_shift"], 0)

    def test_missing_completed_parent_fails_before_emitting_arguments(self) -> None:
        (Path(self.env["STAGE1_CKPT_ROOT"]) / "model/.metadata").unlink()
        with (
            patch.dict(os.environ, self.env, clear=True),
            self.assertRaisesRegex(FileNotFoundError, "STAGE1_CKPT_ROOT"),
        ):
            qa.training_arguments(2)

    def test_later_stages_do_not_require_stage0_audio_assets_or_base(self) -> None:
        environment = {
            key: value
            for key, value in self.env.items()
            if key not in {"BASE_CKPT_ROOT", "AUDIO_VAE_PATH", "AUDIO_VAE_CONFIG_PATH"}
        }
        with patch.dict(os.environ, environment, clear=True):
            for stage in (1, 2, 3):
                self.assertFalse(any("sound_tokenizer" in argument for argument in qa.training_arguments(stage)))


if __name__ == "__main__":
    unittest.main()
