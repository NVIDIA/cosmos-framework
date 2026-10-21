# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Sample and setup contracts for causal multiview (Cosmos3-Nano-Sim-Auto) inference.

The sample cases port the reference implementation's causal simulation argument tests, minus the
fields this repository does not have (``joint_generation``, the top-level ``lidar_condition_path``
prefix tensor options, the persistent joint cache). A measured LiDAR prefix is requested through
``lidar.condition_path`` instead.
"""

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import cosmos_framework.inference.args as args_module
from cosmos_framework.inference.args import OmniSampleOverrides, OmniSetupOverrides

_GB200_MEMORY_BYTES = 192 * 1024**3
_CAMERA_KEYS = ("camera_front_wide_120fov", "camera_cross_left_120fov")

# Causal fields of an exported Sim-Auto student whose export recorded the three rolling-history
# settings.
_SIM_AUTO_EXPORTED_CONFIG: dict[str, Any] = {
    "video_temporal_causal": True,
    "teacher_forcing_kv_implementation": "multiview_maskless_kv",
    "teacher_forcing_frames_per_chunk": 3,
    "teacher_forcing_replay_policy": {
        "_type": "teacher_forcing_replay_policy_config",
        "clean_pass_causality": "chunk",
        "control_visibility": "causal",
        "controls_read_strict_past_clean_rgb": True,
        "decomposed_temporal_window_seconds": 0.4,
        "deduplicate_cross_view": True,
        "multiview_attention_scope": "decomposed",
    },
    "fixed_step_sampler_config": {
        "_type": "fixed_step_sampler_config",
        "sample_type": "sde",
        "t_list": [1.0, 0.9375, 0.8333333333333334, 0.625],
    },
    "diffusion_expert_config": {"base_fps": 30, "enable_fps_modulation": True},
    "rolling_kv_cache_chunks": 2,
    "rolling_kv_sink_chunks": 1,
    "rolling_prompt_policy": "segmented",
    "kv_cache_inference_size": None,
    "attention_sink_size": 0,
    "kv_cache_dtype": None,
}

# The SF-DMD Transfer (Sim-Transfer) student: single-view, one latent frame per chunk.
_SIM_TRANSFER_EXPORTED_CONFIG: dict[str, Any] = {
    "resolution": "480",
    "video_temporal_causal": True,
    "teacher_forcing_kv_implementation": "singleview_threeway_kv",
    "teacher_forcing_frames_per_chunk": 1,
    "teacher_forcing_replay_policy": {
        "_type": "teacher_forcing_replay_policy_config",
        "clean_pass_causality": "frame",
        "control_visibility": "causal",
        "controls_read_strict_past_clean_rgb": True,
        "multiview_attention_scope": "all_views",
    },
    "fixed_step_sampler_config": {"sample_type": "sde", "t_list": [1.0, 0.9375, 0.8333333333333334, 0.625]},
    "kv_cache_inference_size": 51,
    "attention_sink_size": 1,
}


# --------------------------------------------------------------------------------------------------
# Sample contract
# --------------------------------------------------------------------------------------------------


@pytest.fixture
def sample_input(tmp_path: Path) -> dict[str, Any]:
    for key in _CAMERA_KEYS:
        (tmp_path / f"{key}.mp4").touch()
    (tmp_path / "hdmap_rangemap.tar").touch()
    return {
        "name": "scene",
        "output_dir": str(tmp_path / "artifacts"),
        "prompt": "A car.",
        "num_frames": 33,
        "fps": 30,
        "autoregressive": True,
        "wsm": {},
        "multiview": {
            "load_caption_from_data": False,
            "views": [{"camera_key": key, "control_path": str(tmp_path / f"{key}.mp4")} for key in _CAMERA_KEYS],
        },
    }


def _with_lidar(sample_input: dict[str, Any]) -> dict[str, Any]:
    control = Path(sample_input["output_dir"]).parent / "hdmap_rangemap.tar"
    return {**sample_input, "lidar": {"control_path": str(control)}}


def _without(sample_input: dict[str, Any], *fields: str) -> dict[str, Any]:
    return {key: value for key, value in sample_input.items() if key not in fields}


@pytest.fixture
def model_config() -> SimpleNamespace:
    exported = _SIM_AUTO_EXPORTED_CONFIG
    return SimpleNamespace(
        action_gen=False,
        sound_gen=False,
        resolution="480",
        tokenizer=SimpleNamespace(temporal_compression_factor=4),
        vlm_config=SimpleNamespace(model_name="Qwen/Qwen3-VL-8B-Instruct"),
        video_temporal_causal=exported["video_temporal_causal"],
        teacher_forcing_kv_implementation=exported["teacher_forcing_kv_implementation"],
        teacher_forcing_frames_per_chunk=exported["teacher_forcing_frames_per_chunk"],
        teacher_forcing_replay_policy=SimpleNamespace(
            **{k: v for k, v in exported["teacher_forcing_replay_policy"].items() if k != "_type"}
        ),
        diffusion_expert_config=SimpleNamespace(base_fps=exported["diffusion_expert_config"]["base_fps"]),
        fixed_step_sampler_config=None,
        lidar_tokenizer=None,
    )


def _distilled(model_config: SimpleNamespace) -> SimpleNamespace:
    model_config.fixed_step_sampler_config = SimpleNamespace(
        t_list=_SIM_AUTO_EXPORTED_CONFIG["fixed_step_sampler_config"]["t_list"]
    )
    return model_config


def _record_warnings(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    warnings: list[str] = []

    def record_warning(message: str, *_args: object, **_kwargs: object) -> None:
        warnings.append(message)

    monkeypatch.setattr(args_module.log, "warning", record_warning)
    return warnings


@pytest.mark.parametrize("backend", ["multiview_maskless_kv", "multiview_flex_kv"])
def test_causal_defaults_and_explicit_overrides(
    sample_input: dict[str, Any], model_config: SimpleNamespace, backend: str
) -> None:
    preset = args_module._load_modality_defaults("multiview")
    model_config.teacher_forcing_kv_implementation = backend
    _distilled(model_config)
    unset = _without(sample_input, "autoregressive", "fps")

    sample = OmniSampleOverrides(**unset).build_sample(model_config=model_config)

    # The bidirectional multiview preset would run one-shot at its own clock and CFG scale.
    assert preset["autoregressive"] is False
    assert sample.autoregressive is True
    assert sample.fps == model_config.diffusion_expert_config.base_fps != preset["fps"]
    assert sample.guidance == 1.0 != preset["guidance"]
    assert sample.is_multiview_transfer and not sample.is_joint_cam_lidar_transfer

    explicit = OmniSampleOverrides(**unset, fps=24, guidance=3.0).build_sample(model_config=model_config)
    assert explicit.fps == 24 and explicit.guidance == 1.0

    with pytest.raises(ValueError, match="requires autoregressive=True"):
        OmniSampleOverrides(**unset, autoregressive=False).build_sample(model_config=model_config)


def test_custom_defaults_file_values_count_as_configured(
    sample_input: dict[str, Any], model_config: SimpleNamespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    defaults = copy.deepcopy(args_module._load_modality_defaults("multiview"))
    defaults["fps"] = 24
    path = tmp_path / "defaults.json"
    path.write_text(json.dumps(defaults))
    _distilled(model_config)
    warnings = _record_warnings(monkeypatch)

    sample = OmniSampleOverrides(**_without(sample_input, "autoregressive", "fps"), defaults_file=path).build_sample(
        model_config=model_config
    )

    # A defaults file is a deliberate choice, so its clock stands and its CFG scale is overridden loudly.
    assert sample.autoregressive is True
    assert sample.fps == 24
    assert sample.guidance == 1.0
    assert [message for message in warnings if "overriding guidance" in message] == [
        f"Sample 'scene': overriding guidance={defaults['guidance']} with guidance=1.0 for distilled causal "
        "multiview inference; guidance is already distilled into the student."
    ]


@pytest.mark.parametrize("joint", [False, True])
@pytest.mark.parametrize("guidance", [None, 0.0, 1.0, 6.0])
def test_distilled_guidance_is_forced_to_one_with_warning(
    sample_input: dict[str, Any],
    model_config: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    joint: bool,
    guidance: float | None,
) -> None:
    _distilled(model_config)
    model_config.lidar_tokenizer = object()
    if joint:
        sample_input = _with_lidar(sample_input)
    warnings = _record_warnings(monkeypatch)

    sample = OmniSampleOverrides(**sample_input, guidance=guidance).build_sample(model_config=model_config)

    assert sample.is_joint_cam_lidar_transfer is joint
    assert sample.guidance == 1.0
    guidance_warnings = [message for message in warnings if "overriding guidance" in message]
    if guidance in (None, 1.0):
        assert guidance_warnings == []
    else:
        assert len(guidance_warnings) == 1
        assert f"overriding guidance={guidance} with guidance=1.0" in guidance_warnings[0]
    assert sample.model_dump()["guidance"] == 1.0


@pytest.mark.parametrize("trailing_zero", [False, True])
@pytest.mark.parametrize("num_steps", [None, 4, 2, 35])
def test_distilled_num_steps_warns_only_when_configured(
    sample_input: dict[str, Any],
    model_config: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    num_steps: int | None,
    trailing_zero: bool,
) -> None:
    _distilled(model_config)
    if trailing_zero:
        model_config.fixed_step_sampler_config.t_list = [*model_config.fixed_step_sampler_config.t_list, 0.0]
    warnings = _record_warnings(monkeypatch)

    sample = OmniSampleOverrides(**sample_input, num_steps=num_steps).build_sample(model_config=model_config)

    # The student always runs its fixed schedule; the built-in default of 35 steps is not a request.
    assert sample.num_steps == 4
    step_warnings = [message for message in warnings if "is ignored for this distilled checkpoint" in message]
    if num_steps in (None, 4):
        assert step_warnings == []
    else:
        assert step_warnings == [
            f"num_steps={num_steps} is ignored for this distilled checkpoint; using its fixed 4-step sampling schedule."
        ]


def test_distilled_num_steps_from_custom_defaults_file_warns(
    sample_input: dict[str, Any], model_config: SimpleNamespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    defaults = copy.deepcopy(args_module._load_modality_defaults("multiview"))
    assert defaults["num_steps"] != 4
    path = tmp_path / "defaults.json"
    path.write_text(json.dumps(defaults))
    _distilled(model_config)
    warnings = _record_warnings(monkeypatch)

    sample = OmniSampleOverrides(**sample_input, defaults_file=path).build_sample(model_config=model_config)

    # A defaults file is a deliberate choice, so its step count is overridden loudly.
    assert sample.num_steps == 4
    assert [message for message in warnings if "is ignored for this distilled checkpoint" in message] == [
        f"num_steps={defaults['num_steps']} is ignored for this distilled checkpoint; "
        "using its fixed 4-step sampling schedule."
    ]


def test_non_distilled_num_steps_is_preserved(sample_input: dict[str, Any], model_config: SimpleNamespace) -> None:
    sample = OmniSampleOverrides(**sample_input, num_steps=7).build_sample(model_config=model_config)

    assert sample.num_steps == 7


@pytest.mark.parametrize("causal", [False, True])
def test_non_distilled_guidance_is_preserved(
    sample_input: dict[str, Any], model_config: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, causal: bool
) -> None:
    model_config.video_temporal_causal = causal
    warnings = _record_warnings(monkeypatch)

    sample = OmniSampleOverrides(**_without(sample_input, "autoregressive"), guidance=3.0).build_sample(
        model_config=model_config
    )

    assert sample.autoregressive is causal
    assert sample.guidance == 3.0
    assert [message for message in warnings if "guidance" in message] == []


@pytest.mark.parametrize(
    "video_temporal_causal,kv_implementation",
    [(False, "multiview_maskless_kv"), (True, "singleview_threeway_kv"), (True, None)],
)
def test_bidirectional_defaults_remain_unchanged(
    sample_input: dict[str, Any],
    model_config: SimpleNamespace,
    video_temporal_causal: bool,
    kv_implementation: str | None,
) -> None:
    """Only a causal *multiview* checkpoint changes the multiview preset (e.g. not a single-view Sim student)."""
    preset = args_module._load_modality_defaults("multiview")
    model_config.video_temporal_causal = video_temporal_causal
    model_config.teacher_forcing_kv_implementation = kv_implementation
    _distilled(model_config)

    sample = OmniSampleOverrides(**_without(sample_input, "autoregressive", "fps")).build_sample(
        model_config=model_config
    )

    assert sample.autoregressive is preset["autoregressive"]
    assert sample.fps == preset["fps"]
    assert sample.guidance == preset["guidance"]
    with pytest.raises(ValueError, match="autoregressive=True"):
        OmniSampleOverrides(**sample_input).build_sample(model_config=model_config)


def test_rgb_sample_skips_the_joint_lidar_requirements(
    sample_input: dict[str, Any], model_config: SimpleNamespace
) -> None:
    """An RGB-only sample on a causal multiview checkpoint is held to none of the LiDAR checks."""
    rgb = {**sample_input, "fps": 24, "normalize_cfg": True}

    sample = OmniSampleOverrides(**rgb).build_sample(model_config=model_config)

    assert sample.is_multiview_transfer and not sample.is_joint_cam_lidar_transfer
    assert sample.lidar is None
    assert sample.fps == 24 and sample.normalize_cfg is True
    with pytest.raises(ValueError, match="requires a checkpoint with a LiDAR tokenizer"):
        OmniSampleOverrides(**_with_lidar(rgb)).build_sample(model_config=model_config)


def test_joint_manifest_stays_joint_and_requires_a_lidar_tokenizer(
    sample_input: dict[str, Any], model_config: SimpleNamespace
) -> None:
    model_config.lidar_tokenizer = object()
    joint = _with_lidar(_without(sample_input, "autoregressive", "fps"))

    sample = OmniSampleOverrides(**joint).build_sample(model_config=model_config)

    assert sample.is_joint_cam_lidar_transfer and not sample.is_multiview_transfer
    assert sample.autoregressive is True
    assert sample.fps == model_config.diffusion_expert_config.base_fps
    model_config.lidar_tokenizer = None
    with pytest.raises(ValueError, match="requires a checkpoint with a LiDAR tokenizer"):
        OmniSampleOverrides(**joint).build_sample(model_config=model_config)


@pytest.mark.parametrize("causal", [False, True])
def test_nested_lidar_prefix_options_support_one_initial_sweep(
    sample_input: dict[str, Any], model_config: SimpleNamespace, causal: bool
) -> None:
    prefix_path = Path(sample_input["output_dir"]).parent / "measured_prefix.pt"
    prefix_path.touch()
    joint = _with_lidar(sample_input)
    joint["lidar"].update(condition_path=str(prefix_path), num_conditional_sweeps=1)
    joint["autoregressive"] = causal
    model_config.video_temporal_causal = causal
    model_config.lidar_tokenizer = object()

    sample = OmniSampleOverrides(**joint).build_sample(model_config=model_config)
    assert sample.is_joint_cam_lidar_transfer
    assert str(sample.lidar.condition_path) == str(prefix_path)
    assert sample.lidar.num_conditional_sweeps == 1
    joint["lidar"]["num_conditional_sweeps"] = 2
    if causal:
        with pytest.raises(ValueError, match="only one initial LiDAR sweep"):
            OmniSampleOverrides(**joint).build_sample(model_config=model_config)
    else:
        sample = OmniSampleOverrides(**joint).build_sample(model_config=model_config)
        assert sample.lidar.num_conditional_sweeps == 2


@pytest.mark.parametrize("causal", [False, True])
def test_lidar_sweep_count_without_a_prefix(
    sample_input: dict[str, Any], model_config: SimpleNamespace, causal: bool
) -> None:
    # Matches i4: only the causal joint path checks an explicit sweep count, before defaults fill
    # it in; the bidirectional path ignores a count without a prefix.
    joint = _with_lidar(sample_input)
    joint["autoregressive"] = causal
    model_config.video_temporal_causal = causal
    model_config.lidar_tokenizer = object()
    sample = OmniSampleOverrides(**joint).build_sample(model_config=model_config)
    assert sample.lidar.condition_path is None
    for count, message in ((2, "only one initial LiDAR sweep"), (1, "requires lidar.condition_path")):
        joint["lidar"]["num_conditional_sweeps"] = count
        if causal:
            with pytest.raises(ValueError, match=message):
                OmniSampleOverrides(**joint).build_sample(model_config=model_config)
        else:
            assert OmniSampleOverrides(**joint).build_sample(model_config=model_config).lidar.condition_path is None


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("fps", 24, "requires fps=30 to match the MADS/Hyperion source-frame clock"),
        ("normalize_cfg", True, "requires normalize_cfg=False"),
    ],
)
def test_joint_sample_refuses_off_contract_options(
    sample_input: dict[str, Any], model_config: SimpleNamespace, field: str, value: Any, message: str
) -> None:
    model_config.lidar_tokenizer = object()
    joint = _with_lidar(sample_input)
    OmniSampleOverrides(**joint).build_sample(model_config=model_config)
    with pytest.raises(ValueError, match=message):
        OmniSampleOverrides(**{**joint, field: value}).build_sample(model_config=model_config)


@pytest.mark.parametrize("joint", [False, True])
def test_causal_multiview_refuses_control_guidance(
    sample_input: dict[str, Any], model_config: SimpleNamespace, joint: bool
) -> None:
    model_config.lidar_tokenizer = object()
    sample = _with_lidar(sample_input) if joint else sample_input
    with pytest.raises(ValueError, match="requires control_guidance=1.0"):
        OmniSampleOverrides(**sample, control_guidance=2.0).build_sample(model_config=model_config)


@pytest.mark.parametrize("autoregressive", [None, True])
def test_multiview_without_controls_names_the_causal_requirement(
    sample_input: dict[str, Any], model_config: SimpleNamespace, autoregressive: bool | None
) -> None:
    """Causal multiview AR covers per-camera WSM transfer and joint camera+LiDAR only.

    AR is turned on for every multiview sample of this checkpoint, so a sample without controls is
    told what it lacks, not that AR, which its user may never have asked for, is unsupported.
    """
    text_only = _without(sample_input, "autoregressive", "wsm")
    text_only["multiview"] = {"views": [{"camera_key": key} for key in _CAMERA_KEYS]}
    if autoregressive is not None:
        text_only["autoregressive"] = autoregressive

    with pytest.raises(ValueError, match="requires per-camera transfer controls") as error:
        OmniSampleOverrides(**text_only).build_sample(model_config=model_config)
    assert "autoregressive=True" not in str(error.value)

    # A bidirectional multiview checkpoint still rejects an AR request outright.
    bidirectional = SimpleNamespace(**{**vars(model_config), "video_temporal_causal": False})
    with pytest.raises(ValueError, match="- autoregressive=True"):
        OmniSampleOverrides(**{**text_only, "autoregressive": True}).build_sample(model_config=bidirectional)


def test_save_latents_requires_causal_multiview(
    sample_input: dict[str, Any], model_config: SimpleNamespace, tmp_path: Path
) -> None:
    assert OmniSampleOverrides(**sample_input).build_sample(model_config=model_config).save_latents is False
    assert OmniSampleOverrides(**sample_input, save_latents=True).build_sample(model_config=model_config).save_latents

    # The same request on a bidirectional multiview checkpoint has no causal latents to save.
    bidirectional = SimpleNamespace(**{**vars(model_config), "video_temporal_causal": False})
    with pytest.raises(ValueError, match="Latent saving requires causal multiview inference"):
        OmniSampleOverrides(**_without(sample_input, "autoregressive"), save_latents=True).build_sample(
            model_config=bidirectional
        )

    # A JSON input file carries the flag through the shared CLI defaults.
    path = tmp_path / "samples.jsonl"
    path.write_text(json.dumps({**sample_input, "save_latents": True}) + "\n")
    [loaded] = OmniSampleOverrides.from_files([path], overrides=OmniSampleOverrides())
    assert loaded.build_sample(model_config=model_config).save_latents is True


# --------------------------------------------------------------------------------------------------
# Setup contract
# --------------------------------------------------------------------------------------------------


def _write_hf_checkpoint(checkpoint_dir: Path, config: dict[str, Any], **top_level: Any) -> Path:
    checkpoint_dir.mkdir()
    (checkpoint_dir / "model.safetensors.index.json").write_text('{"weight_map": {}}')
    (checkpoint_dir / "config.json").write_text(
        json.dumps({"model": {"_target": "omni_mot_causal_model", "config": config}, **top_level})
    )
    return checkpoint_dir


def _sim_auto_checkpoint(tmp_path: Path, **config_changes: Any) -> Path:
    return _write_hf_checkpoint(
        tmp_path / "sim_auto",
        {**copy.deepcopy(_SIM_AUTO_EXPORTED_CONFIG), **config_changes},
        per_view_captions=True,
        per_view_caption_task="av_joint_camera_lidar_transfer",
    )


def _build_setup(checkpoint_dir: Path, world_size: int, **setup: Any):
    overrides = OmniSetupOverrides(
        checkpoint_path=str(checkpoint_dir), output_dir=checkpoint_dir.parent / "out", **setup
    )
    return overrides, overrides.build_setup(
        world_size=world_size, local_world_size=world_size, device_memory_bytes=_GB200_MEMORY_BYTES
    )


def _contract_warnings(warnings: list[str]) -> list[str]:
    return [message for message in warnings if "causal multiview runtime contract" in message]


def test_sim_auto_setup_pins_the_eager_rolling_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The documented CLI defaults (compile on, diffusion cache on, latency preset) resolve to the rolling runtime."""
    checkpoint_dir = _sim_auto_checkpoint(tmp_path)
    warnings = _record_warnings(monkeypatch)

    overrides, setup = _build_setup(checkpoint_dir, world_size=4, diffusion_cache=True)

    assert setup.config_file == str(checkpoint_dir / "config.json")
    assert setup.use_torch_compile is False
    assert setup.use_cuda_graphs is False
    assert setup.diffusion_cache is False
    assert (setup.cp_size, setup.cfgp_size) == (4, 1)
    assert _contract_warnings(warnings) == [
        "Applying the causal multiview runtime contract; overriding use_torch_compile=True->False, "
        "diffusion_cache=True->False, cp_size=0->4, cfgp_size=0->1."
    ]
    rebuilt = overrides.build_setup(world_size=4, local_world_size=4, device_memory_bytes=_GB200_MEMORY_BYTES)
    assert rebuilt == setup
    assert len(_contract_warnings(warnings)) == 1


@pytest.mark.parametrize(
    "field,requested,transition",
    [
        ("use_torch_compile", True, "use_torch_compile=True->False"),
        ("use_cuda_graphs", True, "use_cuda_graphs=True->False"),
        ("diffusion_cache", True, "diffusion_cache=True->False"),
        ("cp_size", 0, "cp_size=0->4"),
        ("cfgp_size", 2, "cfgp_size=2->1"),
        ("cfgp_size", 0, "cfgp_size=0->1"),
    ],
)
def test_sim_auto_setup_contract_transition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, requested: Any, transition: str
) -> None:
    compliant = dict(use_torch_compile=False, use_cuda_graphs=False, diffusion_cache=False, cp_size=4, cfgp_size=1)
    checkpoint_dir = _sim_auto_checkpoint(tmp_path)
    warnings = _record_warnings(monkeypatch)

    _, setup = _build_setup(checkpoint_dir, world_size=4, **{**compliant, field: requested})

    assert {name: getattr(setup, name) for name in compliant} == compliant
    assert _contract_warnings(warnings) == [f"Applying the causal multiview runtime contract; overriding {transition}."]


@pytest.mark.parametrize("world_size,cp_size,expected_cp_size", [(0, 0, 1), (1, 0, 1), (8, 0, 8), (4, 2, 2)])
def test_sim_auto_setup_cp_size_spans_the_world_unless_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, world_size: int, cp_size: int, expected_cp_size: int
) -> None:
    checkpoint_dir = _sim_auto_checkpoint(tmp_path)
    warnings = _record_warnings(monkeypatch)

    _, setup = _build_setup(
        checkpoint_dir, world_size=world_size, use_torch_compile=False, cp_size=cp_size, cfgp_size=1
    )

    assert setup.cp_size == expected_cp_size
    assert setup.cfgp_size == 1
    assert _contract_warnings(warnings) == (
        []
        if cp_size == expected_cp_size
        else [f"Applying the causal multiview runtime contract; overriding cp_size={cp_size}->{expected_cp_size}."]
    )


@pytest.mark.parametrize(
    "config_changes",
    [
        {"rolling_kv_cache_chunks": None, "rolling_prompt_policy": None},  # the plain export template
        {"rolling_kv_cache_chunks": None},
        {"rolling_prompt_policy": None},
    ],
)
def test_sim_auto_setup_refuses_an_export_without_the_rolling_profile(
    tmp_path: Path, config_changes: dict[str, Any]
) -> None:
    checkpoint_dir = _sim_auto_checkpoint(tmp_path, **config_changes)
    with pytest.raises(ValueError, match="exported without the rolling KV profile") as error:
        _build_setup(checkpoint_dir, world_size=1)
    for override in (
        "model.config.rolling_kv_cache_chunks=",
        "model.config.rolling_kv_sink_chunks=",
        "model.config.rolling_prompt_policy=segmented",
    ):
        assert override in str(error.value)


@pytest.mark.parametrize(
    "setup,flags",
    [
        ({"kv_cache_inference_size": 30}, "--kv-cache-inference-size"),
        ({"kv_cache_inference_size": 0}, "--kv-cache-inference-size"),
        ({"attention_sink_size": 0}, "--attention-sink-size"),
        (
            {"kv_cache_inference_size": 30, "attention_sink_size": 1},
            "--kv-cache-inference-size and --attention-sink-size",
        ),
    ],
)
def test_sim_auto_setup_refuses_the_framewise_cache_flags(tmp_path: Path, setup: dict[str, int], flags: str) -> None:
    checkpoint_dir = _sim_auto_checkpoint(tmp_path)
    with pytest.raises(ValueError, match=f"^{flags} configure the framewise causal cache"):
        _build_setup(checkpoint_dir, world_size=1, **setup)


def test_sim_auto_setup_refuses_a_separate_vae_decode_gpu(tmp_path: Path) -> None:
    """Refused before the model loads, instead of by the causal frontend at the first sample."""
    checkpoint_dir = _sim_auto_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="does not support a separate VAE decode GPU"):
        _build_setup(checkpoint_dir, world_size=4, use_separate_pipeline_vision_decode_gpu=True)

    # The bidirectional sibling keeps the option.
    transfer_auto = _write_hf_checkpoint(
        tmp_path / "transfer_auto", {"video_temporal_causal": False, "fixed_step_sampler_config": None}
    )
    _, setup = _build_setup(transfer_auto, world_size=4, use_separate_pipeline_vision_decode_gpu=True)
    assert setup.use_separate_pipeline_vision_decode_gpu is True


def test_sim_auto_does_not_take_the_sim_transfer_contract(tmp_path: Path) -> None:
    checkpoint_dir = _sim_auto_checkpoint(tmp_path)
    overrides = OmniSetupOverrides(checkpoint_path=str(checkpoint_dir))
    overrides._build_checkpoint(checkpoints=overrides.CHECKPOINTS)

    assert overrides._is_sf_dmd_transfer_hf_checkpoint() is False
    _, setup = _build_setup(checkpoint_dir, world_size=1)
    assert setup.compiled_region == "all"  # the Sim-Transfer contract would pin "language"


def test_sim_transfer_keeps_its_own_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A Sim-Transfer student is not causal multiview: no rolling refusal, and its own runtime stands."""
    checkpoint_dir = _write_hf_checkpoint(tmp_path / "sim_transfer", copy.deepcopy(_SIM_TRANSFER_EXPORTED_CONFIG))
    warnings = _record_warnings(monkeypatch)

    _, setup = _build_setup(checkpoint_dir, world_size=4, diffusion_cache=True, kv_cache_inference_size=30)

    assert setup.use_torch_compile is True
    assert setup.use_cuda_graphs is True
    assert setup.compiled_region == "language"
    assert setup.diffusion_cache is False
    assert (setup.cp_size, setup.cfgp_size) == (1, 1)
    assert "model.config.kv_cache_inference_size=30" in setup.experiment_overrides
    assert _contract_warnings(warnings) == []
    assert [message for message in warnings if "SF-DMD Transfer runtime contract" in message] == [
        "Applying the SF-DMD Transfer runtime contract; overriding use_cuda_graphs=False->True, "
        "compiled_region='all'->'language', diffusion_cache=True->False, cp_size=0->1, cfgp_size=0->1."
    ]


def test_bidirectional_transfer_auto_keeps_the_requested_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_dir = _write_hf_checkpoint(
        tmp_path / "transfer_auto",
        {"video_temporal_causal": False, "fixed_step_sampler_config": None},
        per_view_captions=True,
        per_view_caption_task="av_joint_camera_lidar_transfer",
    )
    warnings = _record_warnings(monkeypatch)

    _, setup = _build_setup(checkpoint_dir, world_size=4, diffusion_cache=True, cp_size=2, cfgp_size=2)

    assert setup.use_torch_compile is True
    assert setup.diffusion_cache is True
    assert (setup.cp_size, setup.cfgp_size) == (2, 2)
    assert _contract_warnings(warnings) == []
