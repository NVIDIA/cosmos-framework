# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json
import os
import types
from pathlib import Path

import omegaconf
import pydantic
import pytest
from typing_extensions import TYPE_CHECKING

from cosmos_framework.inference.args import (
    DEFAULT_CHECKPOINT_NAME,
    MODEL_MEMORY_BYTES_BY_SIZE,
    ModelMode,
    OmniSampleOverrides,
    OmniSetupOverrides,
    SoundDataOverrides,
    _get_nvml_device_memory_info,
    is_reasoner_only,
)
from cosmos_framework.inference.common.config import structure_config

if TYPE_CHECKING:
    from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel

_H100_MEMORY_BYTES = 80 * 1024**3
# Reserved for future use (paired with the reserved memory-based `_get_dp_shard_size`
# heuristic in args.py); not currently exercised.
_GB200_MEMORY_BYTES = 192 * 1024**3


def test_reasoner_only_detection() -> None:
    reasoner = OmniSampleOverrides(model_mode=ModelMode.REASONER)
    generator = OmniSampleOverrides(model_mode=ModelMode.TEXT2VIDEO)

    assert is_reasoner_only([reasoner])
    assert is_reasoner_only([reasoner, reasoner])
    assert not is_reasoner_only([reasoner, generator])
    assert not is_reasoner_only([])


def test_reasoner_only_override_disables_vision_tokenizer_in_model_config(tmp_path: Path) -> None:
    setup_args = OmniSetupOverrides(
        checkpoint_path=DEFAULT_CHECKPOINT_NAME,
        output_dir=tmp_path / "outputs",
    ).build_setup(world_size=1, local_world_size=1, device_memory_bytes=_H100_MEMORY_BYTES)

    model_dict = structure_config(setup_args.load_model_config_dict(), omegaconf.DictConfig)
    assert model_dict.config.load_vision_tokenizer is True

    setup_args.experiment_overrides.append("model.config.load_vision_tokenizer=false")

    model_dict = structure_config(setup_args.load_model_config_dict(), omegaconf.DictConfig)
    assert model_dict.config.load_vision_tokenizer is False


def test_build_parallelism(monkeypatch: pytest.MonkeyPatch):
    parallelism_args = OmniSetupOverrides(
        checkpoint_path=DEFAULT_CHECKPOINT_NAME,
        output_dir="outputs",
        model_memory_bytes=MODEL_MEMORY_BYTES_BY_SIZE["8B"],
        parallelism_preset="latency",
    ).build_parallelism(world_size=16, device_memory_bytes=_H100_MEMORY_BYTES)
    assert parallelism_args.dp_shard_size == 16
    assert parallelism_args.dp_replicate_size == 1
    assert parallelism_args.cp_size == 8
    assert parallelism_args.cfgp_size == 2

    parallelism_args = OmniSetupOverrides(
        checkpoint_path=DEFAULT_CHECKPOINT_NAME,
        output_dir="outputs",
        model_memory_bytes=MODEL_MEMORY_BYTES_BY_SIZE["8B"],
        parallelism_preset="throughput",
    ).build_parallelism(world_size=16, device_memory_bytes=_H100_MEMORY_BYTES)
    assert parallelism_args.dp_shard_size == 16
    assert parallelism_args.dp_replicate_size == 1
    assert parallelism_args.cp_size == 1
    assert parallelism_args.cfgp_size == 1

    parallelism_args = OmniSetupOverrides(
        checkpoint_path=DEFAULT_CHECKPOINT_NAME,
        output_dir="outputs",
        model_memory_bytes=MODEL_MEMORY_BYTES_BY_SIZE["32B"],
        parallelism_preset="latency",
    ).build_parallelism(world_size=16, device_memory_bytes=_H100_MEMORY_BYTES)
    assert parallelism_args.dp_shard_size == 16
    assert parallelism_args.dp_replicate_size == 1
    assert parallelism_args.cp_size == 8
    assert parallelism_args.cfgp_size == 2

    parallelism_args = OmniSetupOverrides(
        checkpoint_path=DEFAULT_CHECKPOINT_NAME,
        output_dir="outputs",
        model_memory_bytes=MODEL_MEMORY_BYTES_BY_SIZE["32B"],
        parallelism_preset="throughput",
    ).build_parallelism(world_size=16, device_memory_bytes=_H100_MEMORY_BYTES)
    assert parallelism_args.dp_shard_size == 16
    assert parallelism_args.dp_replicate_size == 1
    assert parallelism_args.cp_size == 1
    assert parallelism_args.cfgp_size == 1

    parallelism_args = OmniSetupOverrides(
        checkpoint_path=DEFAULT_CHECKPOINT_NAME,
        output_dir="outputs",
        model_memory_bytes=MODEL_MEMORY_BYTES_BY_SIZE["32B"],
        parallelism_preset="latency",
    ).build_parallelism(world_size=0, device_memory_bytes=_H100_MEMORY_BYTES)
    assert parallelism_args.dp_shard_size == 1
    assert parallelism_args.dp_replicate_size == 1
    assert parallelism_args.cp_size == 1
    assert parallelism_args.cfgp_size == 1

    parallelism_args = OmniSetupOverrides(
        checkpoint_path=DEFAULT_CHECKPOINT_NAME,
        output_dir="outputs",
        model_memory_bytes=MODEL_MEMORY_BYTES_BY_SIZE["8B"],
        parallelism_preset="latency",
        compile_dynamic=False,
    ).build_parallelism(world_size=16, device_memory_bytes=_H100_MEMORY_BYTES)
    assert parallelism_args.compile_dynamic is False


def test_get_nvml_device_memory_info_prefers_v2(monkeypatch: pytest.MonkeyPatch):
    from cosmos_framework.inference import args

    expected_info = types.SimpleNamespace(total=96 * 1024**3)

    def fail_v1(_handle):
        raise args.pynvml.NVMLError_NotSupported()

    monkeypatch.setattr(args.pynvml, "nvmlDeviceGetMemoryInfo_v2", lambda _handle: expected_info, raising=False)
    monkeypatch.setattr(args.pynvml, "nvmlDeviceGetMemoryInfo", fail_v1)

    assert _get_nvml_device_memory_info(object()) is expected_info


def test_get_nvml_device_memory_info_falls_back_when_v2_unavailable(monkeypatch: pytest.MonkeyPatch):
    from cosmos_framework.inference import args

    expected_info = types.SimpleNamespace(total=80 * 1024**3)

    monkeypatch.delattr(args.pynvml, "nvmlDeviceGetMemoryInfo_v2", raising=False)
    monkeypatch.setattr(args.pynvml, "nvmlDeviceGetMemoryInfo", lambda _handle: expected_info)

    assert _get_nvml_device_memory_info(object()) is expected_info


def test_checkpoints():
    for name, ckpt in OmniSetupOverrides.CHECKPOINTS.items():
        assert ckpt.hf.repository.split("/")[0] == "nvidia"

        # The released 4-step repositories are gated. Their pinned registry
        # metadata is covered below without requiring CI to hold an HF token.
        if ckpt.hf.repository.endswith("-4Step") and not os.environ.get("HF_TOKEN"):
            continue

        # Download a file to ensure that the repository/revision is valid.
        # Native checkpoints store ``config.json`` at the root; Diffusers
        # checkpoints store the transformer config in ``transformer/``.
        ckpt_hf = ckpt.hf.model_copy(update=dict(include=("config.json", "transformer/config.json")))
        checkpoint_dir = Path(ckpt_hf.download())
        config_paths = [
            path
            for path in (checkpoint_dir / "config.json", checkpoint_dir / "transformer/config.json")
            if path.is_file()
        ]
        assert config_paths, f"No model config found for {name}"
        json.loads(config_paths[0].read_text())


@pytest.mark.parametrize(
    ("checkpoint_name", "repository", "revision", "expected_resolution", "expected_base_fps"),
    [
        (
            "Cosmos3-Super-Text2Image-4Step",
            "nvidia/Cosmos3-Super-Text2Image-4Step",
            "1ba94110bc118f479bbd5e461e79d685d74b2554",
            "768",
            24,
        ),
        (
            "Cosmos3-Super-Image2Video-4Step",
            "nvidia/Cosmos3-Super-Image2Video-4Step",
            "f85d3335d2ad8b352462cecbd637aa980cec9688",
            "480",
            16,
        ),
    ],
)
def test_distilled_checkpoint_uses_published_fixed_step_schedule(
    tmp_path: Path,
    checkpoint_name: str,
    repository: str,
    revision: str,
    expected_resolution: str,
    expected_base_fps: int,
) -> None:
    args = OmniSetupOverrides(
        checkpoint_path=checkpoint_name,
        output_dir=tmp_path / "outputs",
    ).build_setup(world_size=4, local_world_size=4, device_memory_bytes=_H100_MEMORY_BYTES)

    assert args.checkpoint_hf is not None
    assert args.checkpoint_hf.repository == repository
    assert args.checkpoint_hf.revision == revision
    assert args.vlm_processor_from_checkpoint is False

    model_config = args.load_model_config_dict()["config"]
    assert model_config["action_gen"] is False
    assert model_config["sound_gen"] is False
    assert model_config["sound_dim"] is None
    assert model_config["sound_tokenizer"] is None
    assert model_config["resolution"] == expected_resolution
    assert model_config["diffusion_expert_config"]["base_fps"] == expected_base_fps

    if checkpoint_name == "Cosmos3-Super-Text2Image-4Step":
        assert model_config["rectified_flow_training_config"]["shift"]["720"] == 5
        assert model_config["rectified_flow_training_config"]["shift"]["768"] == 5
        assert model_config["tokenizer"]["encode_chunk_frames"]["768"] == 8
        assert model_config["tokenizer"]["encode_exact_durations"] is None

    fixed_step_sampler_config = model_config["fixed_step_sampler_config"]
    assert fixed_step_sampler_config is not None
    assert fixed_step_sampler_config["t_list"] == [1.0, 0.9375, 0.8333333333333334, 0.625]
    assert fixed_step_sampler_config["sample_type"] == "sde"


def test_setup_args(tmp_path: Path):
    overrides = OmniSetupOverrides(
        checkpoint_path=DEFAULT_CHECKPOINT_NAME,
        output_dir=tmp_path / "outputs",
    )
    args = overrides.build_setup()

    def check_model_equal(actual: pydantic.BaseModel, expected: pydantic.BaseModel):
        # Check json first, since the pytest failure diff is more readable.
        assert actual.model_dump() == expected.model_dump()
        assert actual == expected

    # Check idempotent
    check_model_equal(overrides.build_setup(), args)
    check_model_equal(OmniSetupOverrides.model_validate(args.model_dump()).build_setup(), args)


def test_sample_args(tmp_path: Path):
    setup_args = OmniSetupOverrides(
        checkpoint_path=DEFAULT_CHECKPOINT_NAME,
        output_dir=tmp_path / "outputs",
    ).build_setup()
    model_dict: "OmniMoTModel" = structure_config(
        setup_args.load_model_config_dict(),
        omegaconf.DictConfig,
    )

    # Check that all fields are optional
    for name, field in OmniSampleOverrides.model_fields.items():
        assert field.default is None, name

    overrides = OmniSampleOverrides(
        name="test",
    )
    overrides.output_dir = tmp_path / "inputs"
    args = overrides.build_sample(model_config=model_dict.config)

    # Check idempotent
    assert overrides.build_sample(model_config=model_dict.config) == args
    overrides_dump = {k: v for k, v in args.model_dump().items() if k in OmniSampleOverrides.model_fields}
    assert OmniSampleOverrides.model_validate(overrides_dump).build_sample(model_config=model_dict.config) == args

    text2image_args = OmniSampleOverrides(
        name="text2image",
        output_dir=tmp_path / "text2image",
        model_mode=ModelMode.TEXT2IMAGE,
    ).build_sample(model_config=model_dict.config)
    assert text2image_args.aspect_ratio == "1,1"
    assert text2image_args.num_steps == 50
    assert text2image_args.guidance == 4.0
    assert text2image_args.shift == 3.0


def test_edge_num_frames_default(tmp_path: Path):
    def _num_frames(
        checkpoint: str,
        label: str,
        model_mode: ModelMode = ModelMode.TEXT2VIDEO,
        **overrides: object,
    ) -> int:
        setup_args = OmniSetupOverrides(
            checkpoint_path=checkpoint,
            output_dir=tmp_path / f"outputs_{label}",
        ).build_setup()
        model_dict: "OmniMoTModel" = structure_config(
            setup_args.load_model_config_dict(),
            omegaconf.DictConfig,
        )
        args = OmniSampleOverrides(
            name=label,
            output_dir=tmp_path / label,
            model_mode=model_mode,
            **overrides,
        ).build_sample(model_config=model_dict.config)
        return args.num_frames

    # Video generation: Cosmos3-Edge defaults to a shorter 121-frame clip.
    assert _num_frames("Cosmos3-Edge", "edge_default") == 121
    # Other models keep the per-modality JSON default (189).
    assert _num_frames("Cosmos3-Nano", "nano_default") == 189
    # An explicit user value always wins over the model-specific default.
    assert _num_frames("Cosmos3-Edge", "edge_override", num_frames=189) == 189

    # Regression: the Edge 121 default is scoped to plain video generation only.
    # Image modes stay single-frame; action modes keep their own default (189);
    # the reasoner (which reports VIDEO vision_mode) keeps its inert 1 -- none of
    # these should be rewritten to 121.
    assert _num_frames("Cosmos3-Edge", "edge_t2i", model_mode=ModelMode.TEXT2IMAGE) == 1
    assert _num_frames("Cosmos3-Edge", "edge_policy", model_mode=ModelMode.POLICY) == 189
    assert _num_frames("Cosmos3-Edge", "edge_reasoner", model_mode=ModelMode.REASONER, prompt="Describe.") == 1


def test_build_sound_data_requires_sound_path_for_a2v():
    model_config = types.SimpleNamespace(sound_gen=True)
    sample_meta = types.SimpleNamespace(model_mode=ModelMode.AUDIO_IMAGE2VIDEO)

    overrides = SoundDataOverrides(sound_path=None)
    with pytest.raises(ValueError, match="sound_path"):
        overrides._build_sound_data(model_config=model_config, sample_meta=sample_meta)

    overrides = SoundDataOverrides(sound_path="https://example.com/clip.wav")
    overrides._build_sound_data(model_config=model_config, sample_meta=sample_meta)
    assert overrides.enable_sound is True


def test_build_sound_data_rejects_model_without_sound_gen():
    model_config = types.SimpleNamespace(sound_gen=False)
    sample_meta = types.SimpleNamespace(model_mode=ModelMode.AUDIO_IMAGE2VIDEO)
    overrides = SoundDataOverrides(sound_path="https://example.com/clip.wav")
    with pytest.raises(ValueError, match="sound tokenizer"):
        overrides._build_sound_data(model_config=model_config, sample_meta=sample_meta)


def test_audio_image2video_conditions_image_and_sound(tmp_path: Path):
    import omegaconf

    from cosmos_framework.inference.common.config import structure_config

    setup_args = OmniSetupOverrides(
        checkpoint_path=DEFAULT_CHECKPOINT_NAME,
        output_dir=tmp_path / "outputs",
    ).build_setup()
    model_dict = structure_config(setup_args.load_model_config_dict(), omegaconf.DictConfig)

    img = tmp_path / "robot.jpg"
    img.write_bytes(b"\xff\xd8\xff\xe0")  # minimal non-empty file; not actually decoded here
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"RIFF")

    args = OmniSampleOverrides(
        name="a2v",
        output_dir=tmp_path / "a2v",
        model_mode=ModelMode.AUDIO_IMAGE2VIDEO,
        vision_path=str(img),
        sound_path=str(clip),
    ).build_sample(model_config=model_dict.config)

    assert args.condition_vision_mode.value == "image"
    assert args.condition_frame_indexes_vision == [0]
    assert args.enable_sound is True
    assert Path(args.sound_path).name == "clip.wav"


def test_reasoner_video_fps_defaults_none():
    from cosmos_framework.inference.args import ReasonerDataOverrides

    assert ReasonerDataOverrides().video_fps is None


def test_reasoner_video_fps_accepts_positive_float():
    from cosmos_framework.inference.args import ReasonerDataOverrides

    assert ReasonerDataOverrides(video_fps=2.0).video_fps == 2.0
