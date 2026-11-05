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

import cosmos_framework.inference.args as args_module
from cosmos_framework.inference.args import (
    DEFAULT_CHECKPOINT_NAME,
    MODEL_MEMORY_BYTES_BY_SIZE,
    ModelMode,
    OmniSampleOverrides,
    OmniSetupOverrides,
    SoundDataOverrides,
    VisionDataOverrides,
    VisionMode,
    _get_nvml_device_memory_info,
    is_reasoner_only,
    reasoner_only_overrides,
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


def test_reasoner_only_overrides_disable_every_generation_side_module(tmp_path: Path) -> None:
    """All three overrides must land on the live model config, not just parse.

    They are orthogonal on purpose: dropping any one of them degrades to the
    previous behaviour for that module alone.
    """
    setup_args = OmniSetupOverrides(
        checkpoint_path=DEFAULT_CHECKPOINT_NAME,
        output_dir=tmp_path / "outputs",
    ).build_setup(world_size=1, local_world_size=1, device_memory_bytes=_H100_MEMORY_BYTES)

    model_dict = structure_config(setup_args.load_model_config_dict(), omegaconf.DictConfig)
    assert model_dict.config.load_vision_tokenizer is True
    assert model_dict.config.vision_gen is True
    # ``include_gen_pathway`` is absent until overridden: the LazyDict only carries
    # kwargs the experiment config passed explicitly, so the True default lives in
    # ``_MoTConfigBase.__init__``. That is also why enabling this flag cannot leak
    # into an exported checkpoint config.
    assert "include_gen_pathway" not in model_dict.config.vlm_config.model_instance.config

    setup_args.experiment_overrides.extend(reasoner_only_overrides())

    model_dict = structure_config(setup_args.load_model_config_dict(), omegaconf.DictConfig)
    assert model_dict.config.load_vision_tokenizer is False
    assert model_dict.config.vision_gen is False
    assert model_dict.config.vlm_config.model_instance.config.include_gen_pathway is False
    # Cosmos3VFMNetworkConfig asserts action/sound generation each imply vision
    # generation, so the three modality flags must be switched off together.
    assert model_dict.config.action_gen is False
    assert model_dict.config.sound_gen is False


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
            "main",
            "768",
            24,
        ),
        (
            "Cosmos3-Super-Image2Video-4Step",
            "nvidia/Cosmos3-Super-Image2Video-4Step",
            "main",
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


def test_diffusion_cache_max_consecutive_cached_round_trip(tmp_path: Path) -> None:
    overrides = OmniSetupOverrides(
        checkpoint_path=DEFAULT_CHECKPOINT_NAME,
        output_dir=tmp_path / "outputs",
        diffusion_cache=True,
        diffusion_cache_max_consecutive_cached=3,
    )

    args = overrides.build_setup()

    assert args.diffusion_cache is True
    assert args.diffusion_cache_max_consecutive_cached == 3
    assert OmniSetupOverrides.model_validate(args.model_dump()).diffusion_cache_max_consecutive_cached == 3


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
    # Compare the public fields: a built sample also records which fields were requested.
    assert overrides.build_sample(model_config=model_dict.config).model_dump() == args.model_dump()
    overrides_dump = {k: v for k, v in args.model_dump().items() if k in OmniSampleOverrides.model_fields}
    rebuilt = OmniSampleOverrides.model_validate(overrides_dump).build_sample(model_config=model_dict.config)
    assert rebuilt.model_dump() == args.model_dump()

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
    assert _num_frames("Cosmos3-Edge", "edge_policy", model_mode=ModelMode.WAM) == 189
    assert _num_frames("Cosmos3-Edge", "edge_reasoner", model_mode=ModelMode.REASONER, prompt="Describe.") == 1


@pytest.mark.parametrize("num_frames", [5, 301, 901, 1801])
@pytest.mark.parametrize("contract_source", ["training_config", "checkpoint_metadata"])
def test_bimanual_rollout_skips_generic_video_warning(
    monkeypatch: pytest.MonkeyPatch, num_frames: int, contract_source: str
) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(args_module.log, "warning", warnings.append)
    model_config = types.SimpleNamespace(
        resolution="480",
        tokenizer=types.SimpleNamespace(temporal_compression_factor=4),
        action_gen=True,
        action_schema="unified_v1",
        action_global_asinh=True,
        action_metric_gripper=True,
        max_action_dim=64,
        num_embodiment_domains=32,
    )
    if contract_source == "checkpoint_metadata":
        model_config.action_schema = "native"
        model_config._bimanual_action_contract = object()
    sample_meta = types.SimpleNamespace(
        condition_vision_mode=None,
        model_mode=ModelMode.FORWARD_DYNAMICS,
        vision_mode=VisionMode.VIDEO,
    )

    sample = VisionDataOverrides(resolution="480", fps=30, num_frames=num_frames)
    sample._build_vision_data(model_config=model_config, sample_meta=sample_meta)

    assert sample.num_frames == num_frames
    assert warnings == []

    model_config.action_schema = "native"
    model_config._bimanual_action_contract = None
    VisionDataOverrides(resolution="480", fps=30, num_frames=num_frames)._build_vision_data(
        model_config=model_config, sample_meta=sample_meta
    )
    assert any("outside the recommended range [24, 300]" in warning for warning in warnings)


def test_auto_multiview_skips_generic_single_view_duration_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(args_module.log, "warning", warnings.append)
    config = types.SimpleNamespace(resolution="720", tokenizer=types.SimpleNamespace(temporal_compression_factor=4))
    sample_meta = types.SimpleNamespace(
        condition_vision_mode=None, model_mode=ModelMode.TEXT2VIDEO, vision_mode=VisionMode.VIDEO
    )
    sample = OmniSampleOverrides(
        resolution="720",
        fps=30,
        num_frames=297,
        multiview={"views": [{"camera_key": "camera_front_wide_120fov"}]},
    )
    sample._build_vision_data(model_config=config, sample_meta=sample_meta)
    assert sample.num_frames == 297
    assert warnings == []

    # The generic recommendation still applies to ordinary single-view input.
    sample.multiview = None
    sample._build_vision_data(model_config=config, sample_meta=sample_meta)
    assert any("outside the recommended range [24, 200]" in warning for warning in warnings)


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


def _multiview_sample_meta(tmp_path: Path, *, with_vision: bool):
    """Build sample_meta for a 2-view WSM-transfer sample, with or without per-view first frames."""
    from cosmos_framework.inference.args import (
        MultiviewCameraOverrides,
        MultiviewOverrides,
        TransferOverrides,
    )

    views = []
    for index, camera_key in enumerate(("camera_front_wide_120fov", "camera_cross_right_120fov")):
        control = tmp_path / f"control_{index}.mp4"
        control.write_bytes(b"\x00")
        vision_path = None
        if with_vision:
            frame = tmp_path / f"frame_{index}.jpg"
            frame.write_bytes(b"\xff\xd8\xff\xe0")
            vision_path = str(frame)
        views.append(
            MultiviewCameraOverrides(camera_key=camera_key, vision_path=vision_path, control_path=str(control))
        )

    overrides = OmniSampleOverrides(
        name="mv",
        output_dir=tmp_path / "mv",
        model_mode=ModelMode.IMAGE2VIDEO if with_vision else ModelMode.TEXT2VIDEO,
        multiview=MultiviewOverrides(views=views),
        wsm=TransferOverrides(),
    )
    return overrides.sample_meta


def test_multiview_condition_vision_mode_comes_from_the_views(tmp_path: Path):
    """A multiview i2v sample must resolve a conditioning mode from its per-view frames.

    Regression: sample_meta read only the top-level vision_path, which multiview never sets --
    its first frames live in multiview.views[].vision_path. The mode resolved to None,
    _get_multiview_condition_pixel_frames returned [None, ...], and the conditioning was dropped,
    so an image2video sample produced text2video output byte-identical to the t2v run with no
    error anywhere (measured: all 11 views identical to the t2v run).
    """
    sample_meta = _multiview_sample_meta(tmp_path, with_vision=True)

    assert sample_meta.condition_vision_mode is not None, (
        "multiview i2v resolved no conditioning mode; conditioning would be silently dropped"
    )
    assert sample_meta.condition_vision_mode.value == "image"


def test_multiview_text2video_still_resolves_no_conditioning(tmp_path: Path):
    """The multiview branch must not invent conditioning for t2v.

    With no vision_path on any view, MultiviewOverrides.condition_vision_mode returns None --
    the same value the top-level property it replaced would have returned -- so the t2v path is
    unchanged. Verified empirically too: cosmos-framework t2v output is byte-identical before and
    after this fix (11/11 views).
    """
    assert _multiview_sample_meta(tmp_path, with_vision=False).condition_vision_mode is None


# --- camera defaults must track the upstream codebase -----------------------------------
#
# These three are a single coherent recipe, not independent knobs: global_asinh
# requires translation_scale=1.0. The upstream codebase changed all three together in
# 2b258f77340 ("Add production Lance camera stage-0 recipe"), and this repo kept the
# previous values for a while afterwards. Nothing failed -- packages/cosmos3 is not
# release-mapped, so an upstream change to it cannot propagate here, and a camera
# run that does not pass these flags silently produced a different rollout in each
# runtime. Pin them so the next divergence is a failing test rather than a bad video.

_I4_CAMERA_DEFAULTS = {
    "camera_translation_scale": 1.0,
    "camera_action_normalization": "global_asinh",
    "camera_pose_convention": "backward_chunk_anchored_16f",
}


def test_camera_defaults_match_upstream() -> None:
    from cosmos_framework.inference.args import InteractiveDataArgs

    for field, expected in _I4_CAMERA_DEFAULTS.items():
        actual = InteractiveDataArgs.model_fields[field].default
        assert actual == expected, (
            f"{field} is {actual!r} but upstream resolves {expected!r}; "
            "a camera run without this flag would diverge between the two runtimes"
        )


def test_unset_camera_overrides_fall_back_to_the_same_values(tmp_path: Path) -> None:
    """The Overrides fallback must agree with the Args default.

    Two places encode the same recipe, so they can drift apart from each other as
    well as from the upstream codebase.
    """
    import types

    from cosmos_framework.inference.args import InteractiveDataOverrides

    overrides = InteractiveDataOverrides()
    for field in _I4_CAMERA_DEFAULTS:
        assert getattr(overrides, field) is None, f"{field} should be unset before resolution"

    overrides._build_interactive_data(
        model_config=types.SimpleNamespace(video_temporal_causal=False),
        sample_meta=types.SimpleNamespace(),  # discarded by the method
    )
    for field, expected in _I4_CAMERA_DEFAULTS.items():
        assert getattr(overrides, field) == expected, f"{field} fallback disagrees with the Args default"


def _caption_mode_transfer_overrides(tmp_path: Path, *, load_caption_from_data: bool) -> OmniSampleOverrides:
    """A multiview WSM-transfer sample, in caption mode or not, with nothing else set."""
    control = tmp_path / "c0.mp4"
    control.touch()
    return OmniSampleOverrides.model_validate(
        {
            "name": "mv",
            "wsm": {},
            "multiview": {
                "views": [{"camera_key": "camera_front_wide_120fov", "control_path": str(control)}],
                "load_caption_from_data": load_caption_from_data,
            },
        }
    )


def test_caption_mode_withholds_the_generic_transfer_chunk_length(tmp_path: Path) -> None:
    """A caption run derives its chunk length from the span each caption describes.

    The generic 93-frame transfer chunk must not stand in for it: the multiview captions cover 0-300,
    which derives to 297, so letting the default win generates less than a third of what the text
    covers -- with no error, and a well-formed video.
    """
    overrides = _caption_mode_transfer_overrides(tmp_path, load_caption_from_data=True)
    overrides._build_transfer_data(None, None)

    assert overrides.num_video_frames_per_chunk is None
    # Only that one field is withheld; the rest of the transfer defaults still apply.
    assert overrides.num_conditional_frames == 1


def test_a_non_caption_transfer_sample_still_gets_the_default_chunk_length(tmp_path: Path) -> None:
    overrides = _caption_mode_transfer_overrides(tmp_path, load_caption_from_data=False)
    overrides._build_transfer_data(None, None)

    assert overrides.num_video_frames_per_chunk == 93


def _mv_overrides(tmp_path: Path, *, per_view_captions: bool) -> dict:
    """Two cameras in an evaluation-bundle layout, with or without explicit caption paths."""
    sample = tmp_path / "samples" / "clip"
    views = []
    for index, camera in enumerate(("camera_front_wide_120fov", "camera_cross_right_120fov")):
        control = sample / "controls" / f"c{index}.mp4"
        caption = sample / "captions" / "raw" / f"c{index}.json"
        control.parent.mkdir(parents=True, exist_ok=True)
        caption.parent.mkdir(parents=True, exist_ok=True)
        control.touch()
        caption.write_text("{}")
        view = {"camera_key": camera, "control_path": str(control)}
        if per_view_captions:
            view["caption_path"] = str(caption)
        views.append(view)
    return {"name": "mv", "wsm": {}, "multiview": {"views": views, "load_caption_from_data": True}}


def test_each_camera_can_carry_its_own_caption_path(tmp_path: Path) -> None:
    """The schema gap that made the upstream codebase's shipped per-view inputs unloadable here.

    OverridesBase is extra="forbid", so caption_path on a view was a hard validation error rather
    than an ignored key -- 11 of them for an 11-camera rig.
    """
    overrides = OmniSampleOverrides.model_validate(_mv_overrides(tmp_path, per_view_captions=True))

    assert overrides.multiview is not None
    assert [Path(v.caption_path).name for v in overrides.multiview.views] == ["c0.json", "c1.json"]


def test_download_fills_one_caption_per_camera_from_the_bundle(tmp_path: Path) -> None:
    """Inference is per camera, not per sample.

    Inferring one path from views[0] and handing it to every view conditions ten cameras on the
    eleventh's caption, which reads as a plausible video rather than an error.
    """
    overrides = OmniSampleOverrides.model_validate(_mv_overrides(tmp_path, per_view_captions=False))
    assert overrides.multiview is not None
    overrides.multiview.download(tmp_path / "out")

    assert [Path(v.caption_path).name for v in overrides.multiview.views] == ["c0.json", "c1.json"]
    # The sample-level path stays unset: it is the "one shared caption" override, not a default.
    assert overrides.multiview.caption_path is None


def test_a_sample_level_caption_path_still_drives_every_camera(tmp_path: Path) -> None:
    """How lidar transfer works: front-wide text drives all views."""
    shared = tmp_path / "shared.json"
    shared.write_text("{}")
    raw = _mv_overrides(tmp_path, per_view_captions=False)
    raw["multiview"]["caption_path"] = str(shared)

    overrides = OmniSampleOverrides.model_validate(raw)
    assert overrides.multiview is not None
    overrides.multiview.download(tmp_path / "out")

    assert {Path(v.caption_path).name for v in overrides.multiview.views} == {"shared.json"}


def _fake_lidar_model_config() -> types.SimpleNamespace:
    """The bidirectional model-config fields LiDAR sample building reads."""
    return types.SimpleNamespace(
        action_gen=False,
        sound_gen=False,
        resolution="480",
        tokenizer=types.SimpleNamespace(temporal_compression_factor=1),
        video_temporal_causal=False,
        vlm_config=types.SimpleNamespace(model_name="Qwen/Qwen3-VL-8B-Instruct"),
    )


@pytest.mark.parametrize("sweep_count", [None, 3])
def test_joint_lidar_prefix_args_preserve_paths_and_sweep_count(tmp_path: Path, sweep_count: int | None) -> None:
    control = tmp_path / "hdmap.pt"
    condition = tmp_path / "measured.pt"
    wsm = tmp_path / "wsm.mp4"
    for path in (control, condition, wsm):
        path.write_bytes(b"not decoded during argument validation")
    overrides = OmniSampleOverrides(
        name="joint_prefix",
        output_dir=tmp_path / "out",
        prompt="A driving scene.",
        wsm={},
        lidar={"control_path": str(control), "condition_path": str(condition), "num_conditional_sweeps": sweep_count},
        multiview={"views": [{"camera_key": "camera_front_wide_120fov", "control_path": str(wsm)}]},
    )
    args = overrides.build_sample(model_config=_fake_lidar_model_config())
    assert args.lidar.condition_path == str(condition)
    assert args.lidar.num_conditional_sweeps == (1 if sweep_count is None else sweep_count)
    assert args.model_dump()["lidar"]["condition_path"] == str(condition)
    # LiDAR-only generation has no measured-prefix path in Framework, so the prefix is refused there.
    with pytest.raises(ValueError, match="only supported for joint camera\\+LiDAR transfer"):
        OmniSampleOverrides(
            name="standalone_prefix",
            output_dir=tmp_path / "standalone",
            prompt="A driving scene.",
            lidar={"control_path": str(control), "condition_path": str(condition)},
        ).build_sample(model_config=_fake_lidar_model_config())
    with pytest.raises(ValueError, match="greater than 0"):
        OmniSampleOverrides(lidar={"condition_path": str(condition), "num_conditional_sweeps": 0})


def test_lidar_without_a_prefix_resolves_the_default_sweep_count(tmp_path: Path) -> None:
    """Default behaviour is unchanged: no condition_path, nothing measured is read."""
    control = tmp_path / "hdmap.pt"
    control.write_bytes(b"not decoded during argument validation")
    args = OmniSampleOverrides(
        name="no_prefix",
        output_dir=tmp_path / "out",
        prompt="A driving scene.",
        lidar={"control_path": str(control)},
    ).build_sample(model_config=_fake_lidar_model_config())
    assert args.lidar.condition_path is None
    assert args.lidar.num_conditional_sweeps == 1


def test_lidar_only_transfer_refuses_a_measured_prefix(tmp_path: Path) -> None:
    control = tmp_path / "hdmap.tar"
    condition = tmp_path / "measured.tar"
    for path in (control, condition):
        path.write_bytes(b"not decoded during argument validation")
    OmniSampleOverrides(lidar={"control_path": str(control)})._validate_lidar_transfer()
    with_prefix = OmniSampleOverrides(lidar={"control_path": str(control), "condition_path": str(condition)})
    with pytest.raises(ValueError, match="only supported for joint camera\\+LiDAR transfer"):
        with_prefix._validate_lidar_transfer()


def test_lidar_prefix_download_resolves_remote_condition_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import cosmos_framework.inference.args as args_module

    calls: list[tuple[str, str]] = []

    def fake_download(url: str | None, output_dir: Path, output_name: str) -> str | None:
        if url is None:
            return None
        calls.append((url, output_name))
        local = output_dir / f"{output_name}{Path(url).suffix}"
        local.parent.mkdir(parents=True, exist_ok=True)
        local.touch()
        return str(local)

    monkeypatch.setattr(args_module, "download_file", fake_download)
    overrides = OmniSampleOverrides(
        lidar={"control_path": "https://example.com/hdmap.tar", "condition_path": "https://example.com/measured.tar"}
    )
    overrides.download(tmp_path)
    assert overrides.lidar is not None
    assert calls == [
        ("https://example.com/hdmap.tar", "transfer_control"),
        ("https://example.com/measured.tar", "lidar_condition"),
    ]
    assert overrides.lidar.condition_path == str(tmp_path / "lidar_condition.tar")


def test_first_caption_paragraph_only_defaults_off_and_is_accepted_on_multiview(tmp_path: Path) -> None:
    """Training trims captions for Transfer-Auto stage 3 / LiDAR; inference must be able to match."""
    from cosmos_framework.inference.args import MultiviewArgs

    assert MultiviewArgs.model_fields["first_caption_paragraph_only"].default is False
    raw = _mv_overrides(tmp_path, per_view_captions=True)
    overrides = OmniSampleOverrides.model_validate(raw)
    assert overrides.multiview is not None
    assert overrides.multiview.first_caption_paragraph_only is False

    raw["multiview"]["first_caption_paragraph_only"] = True
    overrides = OmniSampleOverrides.model_validate(raw)
    assert overrides.multiview is not None
    assert overrides.multiview.first_caption_paragraph_only is True
    assert overrides.multiview.model_dump()["first_caption_paragraph_only"] is True


def test_save_preview_defaults_off_and_is_resolved_on_build(tmp_path: Path) -> None:
    setup_args = OmniSetupOverrides(
        checkpoint_path=DEFAULT_CHECKPOINT_NAME,
        output_dir=tmp_path / "outputs",
    ).build_setup(world_size=1, local_world_size=1, device_memory_bytes=_H100_MEMORY_BYTES)
    model_dict: "OmniMoTModel" = structure_config(setup_args.load_model_config_dict(), omegaconf.DictConfig)

    args = OmniSampleOverrides(name="video", output_dir=tmp_path / "video").build_sample(model_config=model_dict.config)
    assert args.num_frames > 1
    assert args.save_preview is False
    assert args.preview_crf == 28
    assert args.preview_preset == "medium"

    args = OmniSampleOverrides(
        name="preview",
        output_dir=tmp_path / "preview",
        save_preview=True,
        preview_crf=32,
        preview_preset="fast",
    ).build_sample(model_config=model_dict.config)
    assert args.save_preview is True
    assert args.preview_crf == 32
    assert args.preview_preset == "fast"

    with pytest.raises(pydantic.ValidationError):
        OmniSampleOverrides(preview_crf=52)


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        (["model.config.a=1"], ["model.config.a=1"]),
        (["[model.config.a=1,model.config.b=2]"], ["model.config.a=1", "model.config.b=2"]),
        ([" [model.config.a=1] ", "model.config.c=3"], ["model.config.a=1", "model.config.c=3"]),
    ],
)
def test_normalize_dotlist_overrides_accepts_the_bracketed_cli_form(overrides: list[str], expected: list[str]) -> None:
    from cosmos_framework.inference.common.args import _normalize_dotlist_overrides

    assert _normalize_dotlist_overrides(overrides) == expected


def test_bracketed_experiment_overrides_reach_the_model_config(tmp_path: Path) -> None:
    setup_args = OmniSetupOverrides(
        checkpoint_path=DEFAULT_CHECKPOINT_NAME,
        output_dir=tmp_path / "outputs",
        experiment_overrides=["[model.config.load_vision_tokenizer=false]"],
    ).build_setup(world_size=1, local_world_size=1, device_memory_bytes=_H100_MEMORY_BYTES)

    model_dict = structure_config(setup_args.load_model_config_dict(), omegaconf.DictConfig)
    assert model_dict.config.load_vision_tokenizer is False


def test_batched_cfg_is_off_by_default_and_yields_to_cfg_parallelism(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Parallelism validation reads the launcher's WORLD_SIZE; match the two-rank setup built below.
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")

    def build(**kwargs: object):
        return OmniSetupOverrides(
            checkpoint_path=DEFAULT_CHECKPOINT_NAME,
            output_dir=tmp_path / "outputs",
            **kwargs,
        ).build_setup(world_size=2, local_world_size=2, device_memory_bytes=_H100_MEMORY_BYTES)

    assert build(cp_size=1, cfgp_size=1).use_batched_cfg is False
    assert build(cp_size=1, cfgp_size=1, use_batched_cfg=True).use_batched_cfg is True
    assert build(cp_size=1, cfgp_size=2, use_batched_cfg=True).use_batched_cfg is False


@pytest.mark.parametrize("name", ["sample", "inference_prompt/episode_000", "a/b/c"])
def test_sample_output_dir_keeps_relative_names_under_the_root(tmp_path: Path, name: str) -> None:
    from cosmos_framework.inference.common.args import sample_output_dir

    assert sample_output_dir(tmp_path, name) == tmp_path / name


@pytest.mark.parametrize("name", ["", ".", "..", "../escape", "a/../../escape", "/abs", "a//b", "a\\b", "a\x00b"])
def test_sample_output_dir_rejects_traversal(tmp_path: Path, name: str) -> None:
    from cosmos_framework.inference.common.args import sample_output_dir

    with pytest.raises(ValueError):
        sample_output_dir(tmp_path, name)


def test_sample_output_dir_rejects_a_symlink_out_of_the_root(tmp_path: Path) -> None:
    from cosmos_framework.inference.common.args import sample_output_dir

    root = tmp_path / "root"
    root.mkdir()
    (root / "link").symlink_to(tmp_path)

    with pytest.raises(ValueError, match="outside the requested output root"):
        sample_output_dir(root, "link")
