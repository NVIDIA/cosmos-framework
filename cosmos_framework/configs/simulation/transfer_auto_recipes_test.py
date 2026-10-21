# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU contracts; real training and model compatibility require GPU validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from cosmos_framework.callbacks.stage_handoff import ResetNewStageLearningRate
from cosmos_framework.configs.base.defaults.multiview_attention import MultiviewAttentionConfig
from cosmos_framework.configs.simulation.common.auto_profiles import make_auto_experiment
from cosmos_framework.configs.simulation.common.transfer_auto_recipes import make_transfer_auto_experiment
from cosmos_framework.data.generator.augmentors.merge_datadict import NonRigVideoMetadata
from cosmos_framework.data.generator.augmentors.text_tokenizer import (
    TextTokenizerTransform,
    TextTokenizerTransformForTransfer,
)
from cosmos_framework.data.generator.local_simulation.auto_bundle import build_bundle, default_transforms
from cosmos_framework.data.generator.local_simulation.core import decode_config
from cosmos_framework.data.generator.local_simulation.dataloader import (
    LocalSimulationDataLoader,
    LocalSimulationJointDataLoader,
)
from cosmos_framework.data.generator.local_simulation.dataset import LocalSimulationDataset, _model_constructors
from cosmos_framework.data.generator.local_simulation.resume import (
    LOCAL_SIMULATION_RESUME_KEY,
    LocalSimulationResumeCallback,
    consume_local_resume_state,
)
from cosmos_framework.data.generator.local_simulation.video_prediction import LocalVideoPredictionDataset
from cosmos_framework.data.generator.multiview.local_pipeline import make_auto_augmentations
from cosmos_framework.data.generator.multiview.rig_sampler import ExplicitRigSampler
from cosmos_framework.data.generator.multiview.transforms import (
    ExtractFramesAndCaptions,
    ExtractLidarTransferWindow,
    JointCamLidarToTrainingFormat,
)
from cosmos_framework.data.imaginaire.webdataset.augmentors.image.padding import ReflectionPadding
from cosmos_framework.data.imaginaire.webdataset.augmentors.image.resize import ResizeLargestSideAspectPreserving
from cosmos_framework.model.generator.tokenizers.lidar.range_projection import V1P2_TRANSFER_RANGE_PROJECTION
from cosmos_framework.utils.lazy_config import lazy_call
from cosmos_framework.utils.lazy_config.registry import locate

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


@pytest.mark.parametrize("phase", [1, 2, 3, 4])
def test_phase_contract(phase: int, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COSMOS_TRANSFER_AUTO_NANO_DCP", "/checkpoints/nano")
    for predecessor in (1, 2, 3):
        monkeypatch.setenv(f"COSMOS_TRANSFER_AUTO_PHASE{predecessor}_CHECKPOINT", f"/checkpoints/phase{predecessor}")
    config = make_transfer_auto_experiment(phase)
    streams = config.dataloader_train.dataloaders
    assert config.job.name == f"transfer_auto_phase{phase}"
    assert config.checkpoint.load_path == ("/checkpoints/nano" if phase == 1 else f"/checkpoints/phase{phase - 1}")
    assert config.trainer.max_iter == {1: 70_000, 2: 10_000, 3: 22_000, 4: 50_000}[phase]
    assert config.model.config.ema.iteration_shift == (0 if phase == 1 else 70_000)
    assert config.model.config.joint_attn_implementation == ("two_way" if phase == 1 else "multiview")
    assert sum(child.ratio for child in streams.values()) == pytest.approx(1)
    assert config.model.config.causal_training_strategy == "none"
    assert config.model.config.lidar_state_ch == 128
    assert config.model.config.activation_checkpointing.mode == "selective"
    assert config.model.config.activation_checkpointing.get("save_only_marked_ops", False) == (phase >= 2)
    assert config.model.config.lidar_tokenizer.streaming_chunk_frames == (20 if phase >= 3 else 21)
    assert config.model.config.rectified_flow_training_config.loss_reduction_mode == "local_item_mean"
    assert config.model.config.rectified_flow_training_config.normalize_loss_by_active is False
    assert config.checkpoint.load_training_state == (phase > 1)
    assert config.checkpoint.load_ema_to_reg is False
    for stream in streams.values():
        dataset = stream.dataloader.dataset
        policy = dataset.transform_overrides
        # Ordinary single-camera attention must never receive text_token_lengths;
        # only the joint multiview path scopes independently tokenized captions.
        assert policy.separate_view_text_tokenization == (phase >= 2)
        assert policy.cfg_dropout_rate == 0.1
        assert policy.per_view_caption_dropout_rate == (0.1 if phase >= 2 else 0.0)
        assert policy.lidar_dropout_rate == (0.1 if phase >= 3 else 0.0)
        assert policy.rgb_view_completion_probability == policy.control_dropout_rate == 0.0
        if phase >= 2:
            assert policy.num_video_frames == (65 if phase == 2 else 400)
            assert policy.joint_cam_lidar_fixed_camera_frames == (61 if phase == 2 else None)
            assert "camera_sampler" not in policy  # Retain the local bundle's rig/subset sampling.
    assert list(config.checkpoint.keys_not_to_resume) == (
        [] if phase == 1 else ["dataloader", "scheduler", "trainer"] if phase == 2 else ["dataloader"]
    )
    if phase == 1:
        assert set(streams) == {"rgb_480", "rgb_720", "lidar"}
        assert streams.rgb_480.ratio + streams.rgb_720.ratio == pytest.approx(0.3)
        assert streams.lidar.ratio == pytest.approx(0.7)
        assert streams.rgb_480.dataloader.dataset.transform_overrides.num_cameras_to_keep == 1
        assert streams.rgb_480.dataloader.dataset.transform_overrides.num_video_frames == 400
        assert streams.lidar.dataloader.dataset.transform_overrides.num_video_frames == 21
    else:
        assert set(streams) == {"joint_480", "joint_720"}
        assert streams.joint_480.ratio == pytest.approx(0.6)
        assert streams.joint_720.ratio == pytest.approx(0.4)


@pytest.mark.parametrize("phase", [1, 2, 3, 4])
def test_every_auto_stream_constructs_its_caption_and_sensor_pipeline(phase: int) -> None:
    for stream in make_transfer_auto_experiment(phase).dataloader_train.dataloaders.values():
        dataset = stream.dataloader.dataset
        if "transform_overrides" not in dataset:
            continue
        policy = default_transforms(["camera_front_wide_120fov", "camera_cross_right_120fov"], 480)
        policy.update(OmegaConf.to_container(dataset.transform_overrides, resolve=False))
        policy.update(camera_sampler=None, lidar_range_projection=V1P2_TRANSFER_RANGE_PROJECTION, tokenizer_config={})
        pipeline = make_auto_augmentations(decode_config(policy, constructors=_model_constructors("auto")))
        assert pipeline
        tokenizer = next(
            item for item in pipeline if isinstance(item, (TextTokenizerTransform, TextTokenizerTransformForTransfer))
        )
        assert tokenizer.args["use_system_prompt"] is True


def test_driving_extractor_keeps_reference_fractional_fps_guard(monkeypatch) -> None:
    def decode(video, indices, resolution, **kwargs):
        return torch.zeros((len(indices), 3, *resolution), dtype=torch.uint8), 41.5, resolution

    monkeypatch.setattr(ExtractFramesAndCaptions, "_extract_frames", staticmethod(decode))
    extractor = ExtractFramesAndCaptions(
        camera_order=("video",),
        num_frames=61,
        resolution_hw=(8, 8),
        fps_downsample_factor=1,
        caption_probability={"caption_structured": 1.0},
        camera_view_mapping={"video": -1},
        camera_caption_key_mapping={"video": "metadata"},
        camera_video_key_mapping={"video": "video"},
    )
    sample = extractor(
        dict(
            video=b"decoded-by-test",
            metadata={"caption_structured": {"window": dict(start_frame=0, end_frame=200, caption="A robot moves.")}},
            __key__="fractional-fps",
            __url__="local://test",
        )
    )
    assert sample is None


@pytest.mark.parametrize("resolution", ["480", "720"])
@pytest.mark.parametrize("convert_target_to_string", [False, True])
def test_general_video_uses_reference_augmentors_and_resume_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, resolution: str, convert_target_to_string: bool
) -> None:
    monkeypatch.setattr(lazy_call, "_CONVERT_TARGET_TO_STRING", convert_target_to_string)
    (tmp_path / "manifest.json").write_text(
        json.dumps(dict(format_version=1, complete=True, model="transfer", sample_count=0, transforms={}))
    )
    (tmp_path / "samples.json").write_text("[]")
    dataset = LocalVideoPredictionDataset(tmp_path, resolution_hw=(480, 832) if resolution == "480" else (720, 1280))
    policy = dataset.manifest["prediction_policy"]
    assert policy["augmentor"] == "video_basic_augmentor_v3_json_caption"
    assert policy["preprocessing_version"] == 2
    assert policy["use_system_prompt"] is False
    # Inspect the exact LazyCalls without constructing a real text tokenizer.
    monkeypatch.setattr(
        "cosmos_framework.data.generator.local_simulation.video_prediction.instantiate", lambda config: config
    )
    pipeline = dataset._pipeline("unused")
    targets = [item["_target_"] for item in pipeline[:-1]]
    assert all(isinstance(target, str) == convert_target_to_string for target in targets)
    resolved_targets = [locate(target) if isinstance(target, str) else target for target in targets]
    assert [target.__name__ for target in resolved_targets] == [
        "TextTransformForVideoJsonCaption",
        "VideoParsingChunkedFrames",
        "DataDictMerger",
        "SequencePlanAugmentor",
        "ReflectionPadding",
        "DurationFPSTextTimeStamps",
        "ResolutionTextInfo",
        "TextTokenizerTransform",
        "SoundSequencePlanBuilder",
    ]
    assert isinstance(pipeline[-1], NonRigVideoMetadata)
    parser = pipeline[1]["args"]
    assert parser["max_num_frames"] == 397
    assert parser["min_stride"] == parser["max_stride"] == 1
    assert parser["use_dynamic_fps"] is False
    assert parser["extract_audio"] is False
    assert parser["dataset_resolution_type"] == f"gt{resolution}p"
    assert parser["size"]["16,9"] == ((832, 480) if resolution == "480" else (1280, 720))
    assert pipeline[5]["args"]["fps_key"] == "conditioning_fps"
    assert pipeline[7]["args"]["use_system_prompt"] is False
    assert pipeline[7]["args"]["cfg_dropout_rate"] == 0.1


def test_general_video_spatial_policy_preserves_content_and_padding_mask() -> None:
    # A wide source within a taller bucket must retain both ends, not center-crop.
    source = torch.arange(30, dtype=torch.uint8).reshape(1, 1, 3, 10).expand(3, 5, 3, 10).clone()
    sample = {"video": source, "aspect_ratio": "test"}
    size = {"test": (10, 6)}
    sample = ResizeLargestSideAspectPreserving(input_keys=["video"], args={"size": size})(sample)
    sample = ReflectionPadding(input_keys=["video"], args={"size": size})(sample)
    assert sample["video"].shape == (3, 5, 6, 10)
    torch.testing.assert_close(sample["video"][:, :, :3, :], source)
    assert sample["image_size"].tolist() == [6, 10, 3, 10]


def test_long_joint_continuation_preserves_model_and_data_policy() -> None:
    original = make_transfer_auto_experiment(3)
    continuation = make_transfer_auto_experiment(4)
    assert original.trainer.max_iter == 22_000
    assert continuation.trainer.max_iter == 50_000
    assert continuation.model == original.model
    assert continuation.dataloader_train == original.dataloader_train
    assert continuation.scheduler == original.scheduler
    assert list(continuation.checkpoint.keys_not_to_resume) == ["dataloader"]


@pytest.mark.parametrize(
    "model, phase", [("transfer", 2), ("transfer", 3), ("transfer", 4), ("sim", 1), ("sim", 2), ("sim", 3)]
)
def test_released_auto_training_attention_matches_runtime_schema(model: str, phase: int) -> None:
    config = make_transfer_auto_experiment(phase) if model == "transfer" else make_auto_experiment(phase)
    # Merge against the real structured schema, not another untyped recipe dict:
    # retired keys must fail here just as they do in native Hydra composition.
    attention = OmegaConf.merge(
        OmegaConf.structured(MultiviewAttentionConfig(), flags={"allow_objects": True}),
        config.model.config.multiview_attention,
    )
    assert attention.backend == "maskless"
    assert attention.mask.attention_scope == "decomposed"
    assert tuple(attention.mask.cross_view_temporal_window.bounds) == (-0.4, 0.0)
    assert attention.mask.cross_view_temporal_window.includes_first_frame is False
    assert attention.mask.control_attends_sensor is True
    assert all(bounds is None for bounds in attention.mask.same_view_temporal_window.values())
    if model == "sim":
        # Replay policy is a different schema; its existing field names remain valid.
        replay = config.model.config.teacher_forcing_replay_policy
        assert replay.decomposed_temporal_window_seconds == 0.4
        assert replay.deduplicate_cross_view is True
        assert replay.controls_read_strict_past_clean_rgb is True


@pytest.mark.parametrize("task", ["rgb", "joint", "lidar"])
def test_front_caption_metadata_does_not_invent_an_rgb_camera(task: str) -> None:
    camera = "camera_cross_right_120fov"
    policy = default_transforms([camera], 480)
    policy.update(
        camera_sampler=None,
        joint_cam_lidar_transfer=task == "joint",
        lidar_transfer=task == "lidar",
        load_lidar=task != "rgb",
        separate_view_text_tokenization=task == "joint",
        add_view_prefix_to_caption=task != "lidar",
        per_view_caption_dropout_rate=0.1 if task == "joint" else 0.0,
    )
    decoded = decode_config(policy, constructors=_model_constructors("auto"))
    decoded["lidar_range_projection"] = V1P2_TRANSFER_RANGE_PROJECTION
    pipeline = make_auto_augmentations(decoded)
    cls = ExtractLidarTransferWindow if task == "lidar" else ExtractFramesAndCaptions
    extractor = next(item for item in pipeline if isinstance(item, cls))
    if task == "lidar":
        assert "camera_front_wide_120fov" in extractor.camera_caption_key_mapping
    else:
        assert list(extractor.camera_caption_key_mapping) == [camera]


def test_allowlisted_legacy_constructor_without_i4_import() -> None:
    config = decode_config(
        {
            "_target_": "cosmos_framework.data.generator.multiview.rig_sampler.ExplicitRigSampler",
            "rigs": [["camera_front_wide_120fov"]],
        },
        constructors=_model_constructors("auto"),
    )
    assert config["_target_"] is ExplicitRigSampler
    with pytest.raises(ValueError, match="Unsupported local transform constructor"):
        decode_config({"_target_": "os.system", "command": "not-executed"}, constructors=_model_constructors("auto"))


def test_joint_pipeline_rejects_unsupported_modes() -> None:
    with pytest.raises(ValueError, match="radar"):
        make_auto_augmentations({"joint_cam_radar_transfer": True})
    pipeline = make_auto_augmentations(
        dict(
            camera_keys=["camera_front_wide_120fov"],
            joint_cam_lidar_transfer=True,
            num_video_frames=65,
            joint_cam_lidar_fixed_camera_frames=61,
            camera_sampler=None,
            tokenizer_config=None,
        )
    )
    formatter = next(item for item in pipeline if isinstance(item, JointCamLidarToTrainingFormat))
    assert formatter.fixed_camera_frames == 61
    assert formatter.lidar_fps == 10


def test_changed_auto_policy_is_in_resume_identity(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text(
        json.dumps({"format_version": 1, "complete": True, "model": "auto", "sample_count": 0, "transforms": {}})
    )
    (tmp_path / "samples.json").write_text("[]")
    left = LocalSimulationDataset(tmp_path, model="auto", transform_overrides={"cfg_dropout_rate": 0.1})
    right = LocalSimulationDataset(tmp_path, model="auto", transform_overrides={"cfg_dropout_rate": 0.2})
    assert left.manifest != right.manifest
    assert "local_auto_transform_overrides" not in json.loads((tmp_path / "manifest.json").read_text())


def test_full_rig_default_sampling_and_native_constructor() -> None:
    from cosmos_framework.data.generator.multiview.camera_attributes import CAMERA_VIEW_MAPPING

    policy = default_transforms(list(CAMERA_VIEW_MAPPING), 480)
    sampler = policy["camera_sampler"]
    assert sum(sampler["weights"]) == pytest.approx(1)
    assert sum(
        weight for rig, weight in zip(sampler["rigs"], sampler["weights"], strict=True) if len(rig) == 11
    ) == pytest.approx(0.3)
    decoded = decode_config(sampler, constructors=_model_constructors("auto"))
    assert decoded["_target_"] is ExplicitRigSampler


def test_bundle_builder_preserves_inputs_and_refuses_overwrite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cameras = ["camera_front_wide_120fov"]
    assets = {
        "video_camera_front_wide_120fov": "rgb.mp4",
        "wsm_camera_front_wide_120fov": "wsm.mp4",
        "caption_camera_front_wide_120fov": "caption.json",
        "lidar/lidar_rangemap": "real.npz",
        "lidar/hdmap_rangemap": "map.npz",
    }
    for filename in assets.values():
        (tmp_path / filename).write_bytes(b"packaging-fixture-not-decoded-media")
    (tmp_path / "caption.json").write_text(
        json.dumps(
            {"caption_structured": {"clip": {"start_frame": 0, "end_frame": 65, "caption": "Recorded driving scene."}}}
        )
    )
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer_config.json").write_text("{}")
    index = tmp_path / "input.jsonl"
    index.write_text(json.dumps({"id": "scene-1", "stratum": "480", "assets": assets}) + "\n")
    output = tmp_path / "bundle"
    build_bundle(index, tokenizer, output, cameras)
    dataset = LocalSimulationDataset(output, model="auto")
    assert len(dataset) == 1
    # Construct through the dataset dispatch, not just the augmentation factory.
    # The fixture contains no real tokenizer assets, and no media is decoded here.
    monkeypatch.setattr("cosmos_framework.data.generator.local_simulation.dataset.tokenizer_config", lambda root: {})
    pipeline = dataset._pipeline("480")
    assert any(isinstance(item, ExtractFramesAndCaptions) for item in pipeline)
    assert any(isinstance(item, JointCamLidarToTrainingFormat) for item in pipeline)
    assert dataset._pipeline("480") is pipeline
    assert (tmp_path / "rgb.mp4").read_bytes() == b"packaging-fixture-not-decoded-media"
    assert (output / "resources/tokenizer/tokenizer_config.json").is_file()
    with pytest.raises(FileExistsError):
        build_bundle(index, tokenizer, output, cameras)


def test_stage_initial_lr_preserves_adam_and_same_stage_resume() -> None:
    value = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([value], lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 0.1)
    value.grad = torch.tensor(0.5)
    optimizer.step()
    moments = optimizer.state[value]["exp_avg"].clone()
    optimizer.param_groups[0]["lr"] = 8e-5
    callback = ResetNewStageLearningRate()
    callback.on_before_optimizer_step(None, optimizer, scheduler, None, iteration=0)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-5)
    torch.testing.assert_close(optimizer.state[value]["exp_avg"], moments)
    optimizer.param_groups[0]["lr"] = 3e-5
    callback.on_before_optimizer_step(None, optimizer, scheduler, None, iteration=1)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(3e-5)


def test_retained_counter_can_restart_data_then_resume_its_new_cursor(tmp_path: Path) -> None:
    consume_local_resume_state()
    (tmp_path / "manifest.json").write_text(
        json.dumps({"format_version": 1, "complete": True, "model": "transfer", "sample_count": 1, "transforms": {}})
    )
    (tmp_path / "samples.json").write_text(json.dumps([{"identity": "s", "stratum": "31"}]))

    def make_loader() -> LocalSimulationJointDataLoader:
        return LocalSimulationJointDataLoader(
            restart_data_iterator_on_handoff=True,
            dataloaders={
                name: {
                    "ratio": 0.5,
                    "dataloader": LocalSimulationDataLoader(LocalSimulationDataset(tmp_path), rank=0, world_size=1),
                }
                for name in ("a", "b")
            },
            tokenizer_spatial_compression_factor=16,
            tokenizer_temporal_compression_factor=4,
            patch_spatial=2,
            max_sequence_length=None,
            max_samples_per_batch=1,
            default_lookahead_limit=1,
            prewarm=False,
        )

    first = make_loader()
    first.set_start_iteration(2)
    assert first._fetch_count == 2 and first._fetch_origin == 2
    assert first._stream_fetch_counts == [0, 0]
    callback = LocalSimulationResumeCallback()
    callback.load_state_dict(
        {
            LOCAL_SIMULATION_RESUME_KEY: {
                "format_version": 1,
                "fetch_count": 3,
                "fetch_origin": 2,
                "stream_fetch_counts": [1, 0],
                "identity": first._resume_identity,
            }
        }
    )
    resumed = make_loader()
    resumed.set_start_iteration(3)
    assert resumed._fetch_origin == 2
    assert resumed._fetch_count == 3
    assert resumed._stream_fetch_counts == [1, 0]
    consume_local_resume_state()
