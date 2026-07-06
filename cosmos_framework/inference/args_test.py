# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json
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
)
from cosmos_framework.inference.common.config import structure_config

if TYPE_CHECKING:
    from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel

_H100_MEMORY_BYTES = 80 * 1024**3
# Reserved for future use (paired with the reserved memory-based `_get_dp_shard_size`
# heuristic in args.py); not currently exercised.
_GB200_MEMORY_BYTES = 192 * 1024**3


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

        # Download a file to ensure that the repository/revision is valid.
        # (The published checkpoint.json no longer encodes the source S3 URI —
        # it is empty for most checkpoints and a stale local path for the rest —
        # so we only validate that the repo/revision resolves and parses.)
        ckpt_hf = ckpt.hf.model_copy(update=dict(include=("checkpoint.json",)))
        json.loads((Path(ckpt_hf.download()) / "checkpoint.json").read_text())


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
