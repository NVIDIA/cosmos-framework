# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Checkpoint-driven mixed-domain action preprocessing, without model weights."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from cosmos_framework.inference.action import get_action_sample_data
from cosmos_framework.inference.args import ModelMode, SamplingOverrides
from cosmos_framework.inference.bimanual import (
    bimanual_action_metadata_for_model_config,
    build_bimanual_action_metadata,
    load_bimanual_action_contract,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/data/bimanual_unified_action_test"
LEGACY_FIXTURE = ROOT / "tests/data/bimanual_legacy_action_test"

_RELEASED_SCHEMA5_DOMAIN_IDS_BY_REVISION: dict[str, dict[str, int]] = {
    "37364df6f0e9711ff685ea0870c7aa18da6ac042": {
        "abc_yam": 16,
        "agibot_gear_gripper": 15,
        "agibot_gear_gripper_ext": 15,
        "agibotworld": 15,
        "behavior1k_lerobot": 22,
        "camera_pose": 2,
        "hand_pose": 3,
        "molmoact2_yam": 16,
        "xdof_yam": 16,
    },
    "abda464aa2c42af0d5b0bf4b6d6e9da53e4f0956": {
        "abc_yam": 16,
        "agibot_gear_gripper": 15,
        "agibot_gear_gripper_ext": 15,
        "agibotworld": 15,
        "behavior1k_lerobot": 22,
        "camera_pose": 2,
        "hand_pose": 3,
        "molmoact2_yam": 16,
        "xdof_yam": 16,
    },
}


class BimanualUnifiedActionTests(unittest.TestCase):
    _VALID_RANGES: dict[str, tuple[tuple[int, int], ...]] = {
        "camera_pose": ((0, 9),),
        "hand_pose": ((0, 33), (34, 58)),
        "agibotworld": ((0, 18), (33, 34), (34, 43), (58, 59)),
        "agibot_gear_gripper": ((0, 18), (33, 34), (34, 43), (58, 59)),
        "agibot_gear_gripper_ext": ((0, 18), (33, 34), (34, 43), (58, 59)),
        "embodiment_c_gripper": ((0, 18), (33, 34), (34, 43), (58, 59)),
        "embodiment_c_gripper_ext": ((0, 18), (33, 34), (34, 43), (58, 59)),
        "behavior1k_lerobot": ((0, 18), (33, 34), (34, 43), (58, 59)),
        "abc_yam": ((9, 18), (33, 34), (34, 43), (58, 59)),
        "molmoact2_yam": ((9, 18), (33, 34), (34, 43), (58, 59)),
        "xdof_yam": ((9, 18), (33, 34), (34, 43), (58, 59)),
    }

    @classmethod
    def valid_row(cls, domain_name: str, value: float = 0.1) -> list[float]:
        row = [0.0] * 59
        for start, stop in cls._VALID_RANGES[domain_name]:
            row[start:stop] = [value] * (stop - start)
        return row

    def test_sampling_uses_checkpoint_schedule_with_or_without_terminal_zero(self):
        for schedule in ([1.0, 0.5], [1.0, 0.5, 0.0], [1.0, 0.9375, 0.8333, 0.625]):
            with self.subTest(schedule=schedule):
                sample = SamplingOverrides(num_steps=30)
                model = SimpleNamespace(fixed_step_sampler_config=SimpleNamespace(t_list=schedule))
                sample._build_sampling(model, SimpleNamespace(model_mode=ModelMode.FORWARD_DYNAMICS))
                self.assertEqual(sample.num_steps, len(schedule) - int(schedule[-1] == 0.0))

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.contract = load_bimanual_action_contract(FIXTURE)
        domain_names = ["agibotworld"] * 2 + ["camera_pose"] * 4 + ["agibotworld"] * 2
        self.payload = {
            "action_space": "normalized_unified_v1",
            "action": [self.valid_row(name, (index + 1) / 10) for index, name in enumerate(domain_names)],
            "domain_names": domain_names,
        }
        self.action_path = self.root / "actions.json"
        self.image_path = self.root / "initial.png"
        Image.new("RGB", (832, 480), (24, 48, 72)).save(self.image_path)
        self.model = SimpleNamespace(input_video_key="video", _bimanual_action_contract=self.contract)

    def batch(self, **overrides):
        self.action_path.write_text(json.dumps(self.payload))
        arguments = dict(
            model_config=self.model,
            batch_size=1,
            prompt="Spread the shorts flat.",
            vision_path=self.image_path,
            model_mode=ModelMode.FORWARD_DYNAMICS,
            action_path=self.action_path,
            domain_name="agibotworld",
            resolution="480",
            action_chunk_size=8,
            max_action_dim=64,
            fps=30,
            device="cpu",
            raw_action_dim=59,
        )
        arguments.update(overrides)
        return get_action_sample_data(**arguments)

    def test_unified_actions_keep_values_domains_and_plain_prompt(self):
        batch = self.batch()
        action = batch["action"][0][0]
        torch.testing.assert_close(action[:, :59], torch.tensor(self.payload["action"]), rtol=0, atol=0)
        self.assertEqual(tuple(action.shape), (8, 64))
        self.assertEqual(action[:, 59:].count_nonzero().item(), 0)
        self.assertEqual(batch["domain_id"][0].tolist(), [15, 15, 2, 2, 2, 2, 15, 15])
        self.assertEqual(batch["raw_action_dim"][0].item(), 59)
        self.assertEqual(batch["ai_caption"], ["Spread the shorts flat."])
        self.assertEqual(tuple(batch["video"][0][0].shape), (3, 9, 480, 832))

    def test_explicit_content_crop_excludes_transport_canvas_before_padding(self):
        batch = self.batch(content_crop_box=(0, 0, 640, 360))
        self.assertEqual(batch["image_size"][0].tolist(), [480.0, 832.0, 360.0, 640.0])
        video = batch["video"][0][0]
        self.assertEqual(tuple(video.shape), (3, 9, 480, 832))

    def test_invalid_content_crop_fails_before_encoding(self):
        for crop_box in ((0, 0, 0, 360), (0, 0, 900, 480), (640, 0, 320, 360)):
            with self.subTest(crop_box=crop_box), self.assertRaisesRegex(ValueError, "content_crop_box"):
                self.batch(content_crop_box=crop_box)

    def test_native_frame_is_padded_like_vllm_omni_and_cropped_to_aligned_content(self):
        from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel

        # vLLM-Omni pads the uploaded frame bottom/right to the tier canvas with
        # numpy, reflecting unless a pad reaches the content size, then encodes
        # the canvas and keeps the top-left floor(content / 16) latents.
        model = SimpleNamespace(tokenizer_vision_gen=SimpleNamespace(spatial_compression_factor=16))
        for (height, width), mode, latent_hw in (((360, 640), "reflect", (22, 40)), ((180, 320), "edge", (11, 20))):
            with self.subTest(size=(width, height)):
                pixels = np.random.default_rng(height).integers(0, 256, (height, width, 3), dtype=np.uint8)
                Image.fromarray(pixels).save(self.image_path)
                batch = self.batch()
                canvas = np.pad(pixels, ((0, 480 - height), (0, 832 - width), (0, 0)), mode=mode)
                video = batch["video"][0][0]
                self.assertEqual(tuple(video.shape), (3, 9, 480, 832))
                self.assertTrue(torch.equal(video[:, 0], torch.from_numpy(canvas).permute(2, 0, 1)))
                self.assertEqual(batch["image_size"].tolist(), [[480, 832, height, width]])
                latent = torch.zeros(1, 1, 1, 30, 52)
                (cropped,) = OmniMoTModel._remove_padding_from_latent(model, [latent], [batch["image_size"]])
                self.assertEqual(tuple(cropped.shape[-2:]), latent_hw)

    def test_export_metadata_matches_the_consumer_contract(self):
        expected = json.loads((FIXTURE / "transformer/config.json").read_text())
        self.assertEqual(build_bimanual_action_metadata(), expected)
        bimanual_config = {
            "action_gen": True,
            "action_schema": "unified_v1",
            "action_global_asinh": True,
            "action_metric_gripper": True,
            "max_action_dim": 64,
            "num_embodiment_domains": 32,
        }
        self.assertEqual(
            bimanual_action_metadata_for_model_config(bimanual_config),
            expected,
        )
        self.assertIsNone(bimanual_action_metadata_for_model_config({**bimanual_config, "action_schema": "native"}))
        with self.assertRaisesRegex(ValueError, "action_metric_gripper"):
            bimanual_action_metadata_for_model_config({**bimanual_config, "action_metric_gripper": False})

    def test_action_contract_resolves_native_export_root_config(self):
        metadata = (FIXTURE / "transformer/config.json").read_text()
        (self.root / "config.json").write_text(metadata)

        contract = load_bimanual_action_contract(self.root)

        self.assertIsNotNone(contract)
        assert contract is not None
        self.assertEqual(contract.action_dim, 59)
        self.assertEqual(contract.model_action_dim, 64)
        self.assertEqual(contract.domain_ids, self.contract.domain_ids)

    def test_identical_root_and_transformer_contracts_are_accepted(self):
        metadata = (FIXTURE / "transformer/config.json").read_text()
        (self.root / "config.json").write_text(metadata)
        transformer_config = self.root / "transformer/config.json"
        transformer_config.parent.mkdir()
        transformer_config.write_text(metadata)

        contract = load_bimanual_action_contract(self.root)

        self.assertIsNotNone(contract)
        assert contract is not None
        self.assertEqual(contract.domain_ids, self.contract.domain_ids)

    def test_diffusers_root_config_may_omit_transformer_conditioning(self):
        metadata = json.loads((FIXTURE / "transformer/config.json").read_text())
        root_metadata = copy.deepcopy(metadata)
        del root_metadata["cosmos3_nano_sim_bimanual"]["conditioning"]
        (self.root / "config.json").write_text(json.dumps(root_metadata))
        transformer_config = self.root / "transformer/config.json"
        transformer_config.parent.mkdir()
        transformer_config.write_text(json.dumps(metadata))

        contract = load_bimanual_action_contract(self.root)

        self.assertIsNotNone(contract)
        assert contract is not None
        self.assertEqual(contract.domain_ids, self.contract.domain_ids)

    def test_non_conditioning_metadata_conflict_is_rejected(self):
        metadata = json.loads((FIXTURE / "transformer/config.json").read_text())
        (self.root / "config.json").write_text(json.dumps(metadata))
        transformer_config = self.root / "transformer/config.json"
        transformer_config.parent.mkdir()
        metadata["cosmos3_nano_sim_bimanual"]["chunk_size"] = 32
        transformer_config.write_text(json.dumps(metadata))

        with self.assertRaisesRegex(ValueError, "Conflicting Bimanual action metadata"):
            load_bimanual_action_contract(self.root)

    def test_conflicting_root_and_transformer_contracts_are_rejected(self):
        metadata = json.loads((FIXTURE / "transformer/config.json").read_text())
        (self.root / "config.json").write_text(json.dumps(metadata))
        transformer_config = self.root / "transformer/config.json"
        transformer_config.parent.mkdir()
        metadata["cosmos3_nano_sim_bimanual"]["conditioning"]["input_contract"]["action_dim"] = 58
        transformer_config.write_text(json.dumps(metadata))

        with self.assertRaisesRegex(ValueError, "Conflicting Bimanual action metadata"):
            load_bimanual_action_contract(self.root)

    def test_released_schema5_revisions_have_registered_validity_masks(self) -> None:
        template = json.loads((FIXTURE / "transformer/config.json").read_text())
        checkpoint_root = self.root / "released-checkpoint"
        config_path = checkpoint_root / "transformer/config.json"
        config_path.parent.mkdir(parents=True)

        for revision, domain_ids in _RELEASED_SCHEMA5_DOMAIN_IDS_BY_REVISION.items():
            metadata = copy.deepcopy(template)
            conditioning = metadata["cosmos3_nano_sim_bimanual"]["conditioning"]
            conditioning["embodiments"] = {
                name: {"domain_id": domain_id, "input_action_dim": 59} for name, domain_id in domain_ids.items()
            }
            config_path.write_text(json.dumps(metadata))

            with self.subTest(revision=revision):
                contract = load_bimanual_action_contract(checkpoint_root)
                self.assertIsNotNone(contract)
                assert contract is not None
                self.assertEqual(contract.domain_ids, domain_ids)
                for domain_name in domain_ids:
                    contract.validate(
                        {
                            "action_space": "normalized_unified_v1",
                            "action": [self.valid_row(domain_name)],
                            "domain_names": domain_name,
                        },
                        action_count=1,
                        domain_name=domain_name,
                    )

    def test_full_rollout_preserves_domain_transitions_inside_chunks(self):
        from cosmos_framework.model.generator.omni_mot_causal_model import OmniMoTCausalModel

        domain_names = ["agibotworld"] * 300 + ["camera_pose"] * 300 + ["agibotworld"] * 300
        payload = {
            "action_space": "normalized_unified_v1",
            "action": [self.valid_row(name) for name in domain_names],
            "domain_names": domain_names,
        }
        actions, domains = self.contract.validate(payload, action_count=900, domain_name="agibotworld")
        self.assertEqual(len(actions), 900)
        self.assertEqual(domains, [15] * 300 + [2] * 300 + [15] * 300)
        domain_tensor = torch.tensor(domains)
        self.assertEqual(
            OmniMoTCausalModel._slice_action_domain_id(domain_tensor, 296, 304).tolist(), [15] * 4 + [2] * 4
        )
        self.assertEqual(
            OmniMoTCausalModel._slice_action_domain_id(domain_tensor, 596, 604).tolist(), [2] * 4 + [15] * 4
        )

    def test_single_domain_can_use_the_record_domain(self):
        self.payload.pop("domain_names")
        self.payload["action"] = [self.valid_row("camera_pose")] * 8
        _, domains = self.contract.validate(self.payload, action_count=8, domain_name="camera_pose")
        self.assertEqual(domains, [2] * 8)

    def test_single_domain_can_be_declared_in_the_sidecar(self):
        self.payload["domain_names"] = "camera_pose"
        self.payload["action"] = [self.valid_row("camera_pose")] * 8
        _, domains = self.contract.validate(self.payload, action_count=8, domain_name="agibotworld")
        self.assertEqual(domains, [2] * 8)

    def test_every_supported_domain_enforces_its_unified_validity_mask(self):
        for domain_name in sorted(self.contract.domain_ids):
            payload = {
                "action_space": "normalized_unified_v1",
                "action": [self.valid_row(domain_name)],
                "domain_names": domain_name,
            }
            self.contract.validate(payload, action_count=1, domain_name="agibotworld")

            valid_slots = {slot for start, stop in self._VALID_RANGES[domain_name] for slot in range(start, stop)}
            invalid_slot = next(slot for slot in range(59) if slot not in valid_slots)
            payload["action"][0][invalid_slot] = 0.25
            with self.subTest(domain_name=domain_name), self.assertRaisesRegex(ValueError, "invalid unified-v1"):
                self.contract.validate(payload, action_count=1, domain_name="agibotworld")

    def test_single_entry_domain_list_is_not_implicitly_broadcast(self):
        self.payload["domain_names"] = ["xdof_yam"]
        with self.assertRaisesRegex(ValueError, "one name per action row"):
            self.contract.validate(self.payload, action_count=8, domain_name="agibotworld")

    def test_legacy_array_is_rejected_for_the_new_checkpoint(self):
        self.payload = [[0.0] * 29 for _ in range(8)]
        with self.assertRaisesRegex(ValueError, "legacy native action arrays are incompatible"):
            self.batch()

    def test_unified_sidecar_is_rejected_without_checkpoint_contract(self):
        with self.assertRaisesRegex(ValueError, "schema-5 metadata"):
            self.batch(model_config=SimpleNamespace(input_video_key="video"), raw_action_dim=None)

    def test_schema3_export_preserves_native_actions_and_json_prompt(self):
        contract = load_bimanual_action_contract(LEGACY_FIXTURE)
        self.assertIsNone(contract)
        self.payload = [[(i + 1) / 10] * 29 for i in range(8)]
        batch = self.batch(
            model_config=SimpleNamespace(input_video_key="video", _bimanual_action_contract=contract),
            raw_action_dim=29,
        )
        action = batch["action"][0][0]
        torch.testing.assert_close(action[:, :29], torch.tensor(self.payload), rtol=0, atol=0)
        self.assertEqual(tuple(action.shape), (8, 64))
        self.assertEqual(action[:, 29:].count_nonzero().item(), 0)
        self.assertEqual(batch["domain_id"][0].item(), 15)
        self.assertEqual(batch["raw_action_dim"][0].item(), 29)
        self.assertIsInstance(json.loads(batch["ai_caption"][0]), dict)
        self.assertIn("Spread the shorts flat.", batch["ai_caption"][0])

    def test_schema3_export_rejects_unified_sidecar(self):
        model = SimpleNamespace(
            input_video_key="video", _bimanual_action_contract=load_bimanual_action_contract(LEGACY_FIXTURE)
        )
        with self.assertRaisesRegex(ValueError, "schema-5 metadata"):
            self.batch(model_config=model, raw_action_dim=29)

    def test_incompatible_schema3_contract_is_not_silently_treated_as_native(self):
        original = json.loads((LEGACY_FIXTURE / "transformer/config.json").read_text())
        path = self.root / "transformer/config.json"
        path.parent.mkdir()
        mutations = [
            lambda s: s.update(model_action_dim=32),
            lambda s: s["embodiments"]["agibotworld"].update(raw_action_dim=59),
            lambda s: s["embodiments"]["agibotworld"].update(domain_id=2),
            lambda s: s["embodiments"]["agibotworld"]["normalizer"].update(method="global_asinh"),
        ]
        for index, mutate in enumerate(mutations):
            data = copy.deepcopy(original)
            mutate(data["cosmos3_nano_sim_bimanual"]["conditioning"])
            path.write_text(json.dumps(data))
            with self.subTest(index=index), self.assertRaisesRegex(ValueError, "Unsupported legacy Bimanual"):
                load_bimanual_action_contract(self.root)

    def test_incompatible_flags_fail_before_generation(self):
        for options in (dict(raw_action_dim=29), dict(max_action_dim=32), dict(format_prompt_as_json=True)):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.batch(**options)

    def test_invalid_sidecars_fail_before_tensor_conversion(self):
        mutations = [
            lambda p: p.update(action_space="raw"),
            lambda p: p["action"].pop(),
            lambda p: p["action"][0].pop(),
            lambda p: p["action"][0].__setitem__(0, float("nan")),
            lambda p: p["action"][0].__setitem__(0, True),
            lambda p: p["domain_names"].pop(),
            lambda p: p["domain_names"].__setitem__(0, "unknown"),
        ]
        for index, mutate in enumerate(mutations):
            payload = copy.deepcopy(self.payload)
            mutate(payload)
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.contract.validate(payload, action_count=8, domain_name="agibotworld")

    def test_unsupported_schema_fails_and_legacy_metadata_is_optional(self):
        self.assertIsNone(load_bimanual_action_contract(self.root))
        path = self.root / "transformer/config.json"
        path.parent.mkdir()
        data = json.loads((FIXTURE / "transformer/config.json").read_text())
        data["cosmos3_nano_sim_bimanual"]["conditioning"]["schema_version"] = 99
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "Unsupported Bimanual action contract"):
            load_bimanual_action_contract(self.root)


if __name__ == "__main__":
    unittest.main()
