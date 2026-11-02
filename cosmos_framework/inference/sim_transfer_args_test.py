# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import cosmos_framework.inference.args as args_module
from cosmos_framework.inference.args import OmniSampleOverrides, OmniSetupOverrides

_GB200_MEMORY_BYTES = 192 * 1024**3


@pytest.mark.parametrize("requested_ar,expected_ar", [(None, True), (False, False), (True, True)])
def test_build_sample_resolves_transfer_ar_from_checkpoint(
    tmp_path: Path, requested_ar: bool | None, expected_ar: bool
) -> None:
    control = tmp_path / "depth.mp4"
    control.touch()
    model_config = SimpleNamespace(
        action_gen=False,
        sound_gen=False,
        resolution="480",
        tokenizer=SimpleNamespace(temporal_compression_factor=4),
        video_temporal_causal=True,
        vlm_config=SimpleNamespace(model_name="Qwen/Qwen3-VL-8B-Instruct"),
        fixed_step_sampler_config=SimpleNamespace(t_list=[1.0, 0.9375, 0.8333333333333334, 0.625]),
        teacher_forcing_frames_per_chunk=1,
        teacher_forcing_replay_policy=SimpleNamespace(
            control_visibility="causal", controls_read_strict_past_clean_rgb=True, clean_pass_causality="frame"
        ),
    )
    sample = OmniSampleOverrides(
        name="depth",
        prompt="A robot places an object on a table.",
        model_mode="video2video",
        output_dir=tmp_path / "output",
        depth={"control_path": str(control)},
        autoregressive=requested_ar,
        resolution="480",
        aspect_ratio="16,9",
    ).build_sample(model_config=model_config)

    assert sample.autoregressive is expected_ar
    assert sample.control_guidance == (1.0 if expected_ar else 1.5)


@pytest.mark.parametrize("resolution,expected_cap,expected_shift", [("480", 201, 5.0), ("720", 81, 10.0)])
@pytest.mark.parametrize("contract_source", ("native_self_forcing", "exported_distilled"))
def test_causal_transfer_defaults_match_student_recipe(
    tmp_path: Path, resolution: str, expected_cap: int, expected_shift: float, contract_source: str
) -> None:
    control = tmp_path / "depth.mp4"
    control.touch()
    sample = OmniSampleOverrides(
        name="depth",
        output_dir=tmp_path / "output",
        depth={"control_path": str(control)},
        autoregressive=True,
        resolution=resolution,
        guidance=1.0,
    )
    model_config = SimpleNamespace(
        fixed_step_sampler_config=SimpleNamespace(t_list=[1.0, 0.9375, 0.8333333333333334, 0.625]),
        simulation_mode="self_forcing",
        video_temporal_causal=True,
        teacher_forcing_frames_per_chunk=1,
        teacher_forcing_replay_policy=SimpleNamespace(
            control_visibility="causal", controls_read_strict_past_clean_rgb=True, clean_pass_causality="frame"
        ),
    )
    if contract_source == "native_self_forcing":
        model_config.fixed_step_sampler_config = None
    else:
        model_config.simulation_mode = None

    sample._build_transfer_data(model_config, sample.sample_meta, user_fields=frozenset())
    sample._apply_sf_dmd_transfer_inference_contract(model_config=model_config, user_fields=frozenset())

    assert sample.max_frames == expected_cap
    assert sample.shift == expected_shift
    assert sample.guidance == 1.0
    assert sample.control_guidance == 1.0

    sample.max_frames = 401
    sample.shift = 7.0
    sample._apply_sf_dmd_transfer_inference_contract(
        model_config=model_config, user_fields=frozenset({"max_frames", "shift"})
    )
    assert sample.max_frames == 401
    assert sample.shift == 7.0


def _write_sf_dmd_student_checkpoint(checkpoint_dir: Path, public_aliases: bool) -> None:
    checkpoint_dir.mkdir()
    (checkpoint_dir / "model.safetensors.index.json").write_text('{"weight_map": {}}')
    target = (
        {"_target": "omni_mot_causal_model"}
        if public_aliases
        else {"_target_": "cosmos_framework.model.generator.omni_mot_causal_model.OmniMoTCausalModel"}
    )
    (checkpoint_dir / "config.json").write_text(
        json.dumps(
            {
                "model": {
                    **target,
                    "config": {
                        "fixed_step_sampler_config": {"t_list": [1.0, 0.9375, 0.8333333333333334, 0.625]},
                        "video_temporal_causal": True,
                        "teacher_forcing_frames_per_chunk": 1,
                        "teacher_forcing_replay_policy": {
                            "control_visibility": "causal",
                            "controls_read_strict_past_clean_rgb": True,
                            "clean_pass_causality": "frame",
                        },
                    },
                }
            }
        )
    )


def _record_contract_warnings(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    warnings: list[str] = []

    def record_warning(message: str, *_args: object, **_kwargs: object) -> None:
        warnings.append(message)

    monkeypatch.setattr(args_module.log, "warning", record_warning)
    return warnings


@pytest.mark.parametrize("public_aliases", [False, True])
def test_sf_dmd_transfer_hf_setup_uses_artifact_contract(
    tmp_path: Path, public_aliases: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a CUDA-graph request the student runs eager; the shared runtime limits always apply."""
    checkpoint_dir = tmp_path / "student"
    _write_sf_dmd_student_checkpoint(checkpoint_dir, public_aliases)
    overrides = OmniSetupOverrides(
        checkpoint_path=str(checkpoint_dir),
        output_dir=tmp_path / "outputs",
        use_torch_compile=True,
        use_cuda_graphs=False,
        diffusion_cache=True,
        cp_size=2,
        cfgp_size=2,
    )
    warnings = _record_contract_warnings(monkeypatch)

    setup = overrides.build_setup(world_size=1, local_world_size=1, device_memory_bytes=_GB200_MEMORY_BYTES)

    assert setup.experiment == ""
    assert setup.config_file == str(checkpoint_dir / "config.json")
    assert setup.use_torch_compile is True
    assert setup.use_cuda_graphs is True  # the default runtime: compile + CUDA graphs
    assert setup.compiled_region == "language"
    assert setup.diffusion_cache is False
    assert setup.cp_size == 1
    assert setup.cfgp_size == 1
    assert overrides.build_setup(world_size=1, local_world_size=1, device_memory_bytes=_GB200_MEMORY_BYTES) == setup
    assert [message for message in warnings if "SF-DMD Transfer runtime contract" in message] == [
        "Applying the SF-DMD Transfer runtime contract; overriding "
        "use_cuda_graphs=False->True, compiled_region='all'->'language', diffusion_cache=True->False, "
        "cp_size=2->1, cfgp_size=2->1."
    ]


def test_sf_dmd_transfer_hf_setup_without_compile_runs_eager(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``--no-use-torch-compile`` selects the eager runtime; CUDA graphs are switched off with it."""
    checkpoint_dir = tmp_path / "student"
    _write_sf_dmd_student_checkpoint(checkpoint_dir, public_aliases=False)
    overrides = OmniSetupOverrides(
        checkpoint_path=str(checkpoint_dir),
        output_dir=tmp_path / "outputs",
        use_torch_compile=False,
        use_cuda_graphs=False,
        diffusion_cache=True,
        cp_size=2,
        cfgp_size=2,
    )
    warnings = _record_contract_warnings(monkeypatch)

    setup = overrides.build_setup(world_size=1, local_world_size=1, device_memory_bytes=_GB200_MEMORY_BYTES)

    assert setup.use_torch_compile is False
    assert setup.use_cuda_graphs is False
    assert setup.diffusion_cache is False
    assert setup.cp_size == 1
    assert setup.cfgp_size == 1
    assert [message for message in warnings if "SF-DMD Transfer runtime contract" in message] == [
        "Applying the SF-DMD Transfer runtime contract; overriding compiled_region='all'->'language', "
        "diffusion_cache=True->False, cp_size=2->1, cfgp_size=2->1."
    ]


def test_sf_dmd_transfer_hf_setup_rejects_cuda_graphs_without_compile(tmp_path: Path) -> None:
    """CUDA graphs ride on torch.compile; asking for graphs while disabling compile is a user error."""
    checkpoint_dir = tmp_path / "student"
    _write_sf_dmd_student_checkpoint(checkpoint_dir, public_aliases=False)
    overrides = OmniSetupOverrides(
        checkpoint_path=str(checkpoint_dir),
        output_dir=tmp_path / "outputs",
        use_torch_compile=False,
        use_cuda_graphs=True,
    )
    with pytest.raises(ValueError, match="require torch.compile"):
        overrides.build_setup(world_size=1, local_world_size=1, device_memory_bytes=_GB200_MEMORY_BYTES)


def test_sf_dmd_transfer_contract_reads_registered_and_local_configs_without_hf_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_download(*_args: object, **_kwargs: object) -> str:
        raise AssertionError("SF-DMD Transfer detection unexpectedly downloaded Hugging Face metadata")

    monkeypatch.setattr(args_module.CheckpointDirHf, "download", fail_download)

    for checkpoint_name in OmniSetupOverrides.CHECKPOINTS:
        registered = OmniSetupOverrides(checkpoint_path=checkpoint_name)
        registered._build_checkpoint(checkpoints=registered.CHECKPOINTS)
        assert registered._is_sf_dmd_transfer_hf_checkpoint() is False

    checkpoint_dir = tmp_path / "student"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "model.safetensors.index.json").write_text('{"weight_map": {}}')
    (checkpoint_dir / "config.json").write_text(
        json.dumps(
            {
                "model": {
                    "config": {
                        "fixed_step_sampler_config": {"t_list": [1.0, 0.625]},
                        "video_temporal_causal": True,
                        "teacher_forcing_frames_per_chunk": 1,
                        "teacher_forcing_replay_policy": {
                            "control_visibility": "causal",
                            "controls_read_strict_past_clean_rgb": True,
                            "clean_pass_causality": "frame",
                        },
                    }
                }
            }
        )
    )
    local = OmniSetupOverrides(checkpoint_path=str(checkpoint_dir))
    local._build_checkpoint(checkpoints=local.CHECKPOINTS)
    assert local.checkpoint_hf is None
    assert local._is_sf_dmd_transfer_hf_checkpoint() is True


@pytest.mark.parametrize(
    "config",
    [
        {},
        {
            "fixed_step_sampler_config": {"t_list": [1.0, 0.9375, 0.8333333333333334, 0.625]},
            "video_temporal_causal": False,
        },
        {
            "fixed_step_sampler_config": {"t_list": [1.0, 0.9375, 0.8333333333333334, 0.625]},
            "video_temporal_causal": True,
            "teacher_forcing_frames_per_chunk": 4,
            "teacher_forcing_replay_policy": {
                "control_visibility": "causal",
                "controls_read_strict_past_clean_rgb": True,
                "clean_pass_causality": "frame",
            },
        },
        {
            "fixed_step_sampler_config": {"t_list": [1.0, 0.9375, 0.8333333333333334, 0.625]},
            "video_temporal_causal": True,
            "teacher_forcing_frames_per_chunk": 1,
            "teacher_forcing_replay_policy": None,
        },
    ],
)
def test_other_hf_models_preserve_requested_runtime(tmp_path: Path, config: dict) -> None:
    checkpoint_dir = tmp_path / "student"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "model.safetensors.index.json").write_text('{"weight_map": {}}')
    (checkpoint_dir / "config.json").write_text(json.dumps({"model": {"config": config}}))
    setup = OmniSetupOverrides(
        checkpoint_path=str(checkpoint_dir),
        output_dir=tmp_path / "outputs",
        use_torch_compile=True,
        diffusion_cache=True,
        cp_size=2,
        cfgp_size=2,
    ).build_setup(world_size=4, local_world_size=4, device_memory_bytes=_GB200_MEMORY_BYTES)

    assert setup.use_torch_compile is True
    assert setup.diffusion_cache is True
    assert setup.cp_size == 2
    assert setup.cfgp_size == 2


@pytest.mark.parametrize("resolution", ["256", "720"])
@pytest.mark.parametrize("autoregressive", [False, True])
def test_480p_transfer_checkpoint_rejects_other_resolution_tiers(
    tmp_path: Path, resolution: str, autoregressive: bool
) -> None:
    control = tmp_path / "depth.mp4"
    control.touch()
    model_config = SimpleNamespace(
        resolution="480",
        fixed_step_sampler_config=SimpleNamespace(t_list=[1.0, 0.9375, 5 / 6, 0.625]),
        video_temporal_causal=True,
        teacher_forcing_frames_per_chunk=1,
        teacher_forcing_replay_policy=SimpleNamespace(
            control_visibility="causal", controls_read_strict_past_clean_rgb=True, clean_pass_causality="frame"
        ),
    )
    sample = OmniSampleOverrides(
        depth={"control_path": str(control)}, autoregressive=autoregressive, resolution=resolution
    )
    with pytest.raises(ValueError, match="supports only resolution='480'"):
        sample._apply_sf_dmd_transfer_inference_contract(model_config=model_config, user_fields=frozenset())
    # Omitting the required control must not bypass the model's resolution limit.
    without_control = OmniSampleOverrides(autoregressive=autoregressive, resolution=resolution)
    with pytest.raises(ValueError, match="supports only resolution='480'"):
        without_control._apply_sf_dmd_transfer_inference_contract(model_config=model_config, user_fields=frozenset())


@pytest.mark.parametrize("field", ["dp_shard_size", "cp_size", "cfgp_size"])
def test_streaming_transfer_rejects_sharded_execution(field: str) -> None:
    with pytest.raises(ValueError, match="requires dp_shard_size=1, cp_size=1, and cfgp_size=1"):
        OmniSetupOverrides(
            checkpoint_path=args_module.DEFAULT_CHECKPOINT_NAME, condition_stream_encode=True, **{field: 2}
        )


@pytest.mark.parametrize("window,sink,expected_window", [(30, 3, "30"), (0, 0, "null")])
def test_public_causal_cache_overrides_preserve_exported_config(
    tmp_path: Path, window: int, sink: int, expected_window: str
) -> None:
    checkpoint_dir = tmp_path / "student"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "model.safetensors.index.json").write_text('{"weight_map": {}}')
    source_config = {
        "model": {
            "_target": "omni_mot_causal_model",
            "config": {
                "resolution": "480",
                "video_temporal_causal": True,
                "teacher_forcing_frames_per_chunk": 1,
                "teacher_forcing_replay_policy": {
                    "control_visibility": "causal",
                    "controls_read_strict_past_clean_rgb": True,
                    "clean_pass_causality": "frame",
                },
                "fixed_step_sampler_config": {"sample_type": "sde", "t_list": [1.0, 0.9375, 5 / 6, 0.625]},
                "kv_cache_inference_size": 51,
                "attention_sink_size": 1,
            },
        }
    }
    config_path = checkpoint_dir / "config.json"
    config_path.write_text(json.dumps(source_config))
    overrides = OmniSetupOverrides(
        checkpoint_path=str(checkpoint_dir),
        output_dir=tmp_path / "outputs",
        condition_stream_encode=True,
        kv_cache_inference_size=window,
        attention_sink_size=sink,
        dp_shard_size=1,
    )
    setup = overrides.build_setup(world_size=1, local_world_size=1, device_memory_bytes=_GB200_MEMORY_BYTES)
    assert setup.condition_stream_encode is True
    assert setup.experiment_overrides.count(f"model.config.kv_cache_inference_size={expected_window}") == 1
    assert setup.experiment_overrides.count(f"model.config.attention_sink_size={sink}") == 1
    config = setup.load_model_config_dict()["config"]
    assert config["kv_cache_inference_size"] == (window or None)
    assert config["attention_sink_size"] == sink
    assert json.loads(config_path.read_text()) == source_config
    assert overrides.build_setup(world_size=1, local_world_size=1, device_memory_bytes=_GB200_MEMORY_BYTES) == setup


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("resolution", ["256", "704", "720", "768"])
def test_480_only_checkpoint_names_its_own_limit_for_every_rejected_tier(tmp_path: Path, resolution: str) -> None:
    control = tmp_path / "depth.mp4"
    control.touch()
    model_config = SimpleNamespace(
        resolution="480",
        fixed_step_sampler_config=SimpleNamespace(t_list=[1.0, 0.9375, 5 / 6, 0.625]),
        video_temporal_causal=True,
        causal_training_strategy="none",
        teacher_forcing_frames_per_chunk=1,
        teacher_forcing_replay_policy=SimpleNamespace(
            control_visibility="causal", controls_read_strict_past_clean_rgb=True, clean_pass_causality="frame"
        ),
    )
    sample = OmniSampleOverrides(
        depth={"control_path": str(control)},
        autoregressive=True,
        resolution=resolution,
        transfer_resize_mode="direct_resize",
        control_guidance=1.0,
        num_first_chunk_conditional_frames=0,
        share_vision_temporal_positions=True,
    )
    with pytest.raises(ValueError, match=f"supports only resolution='480', got resolution='{resolution}'"):
        sample._build_transfer_data(model_config, sample.sample_meta, user_fields=frozenset())
        sample._apply_sf_dmd_transfer_inference_contract(model_config=model_config, user_fields=frozenset())


@pytest.mark.L0
@pytest.mark.CPU
def test_bidirectional_480_model_keeps_the_teacher_tier_message(tmp_path: Path) -> None:
    # Only the causal SF-DMD student is 480-only; a bidirectional model recording 480 still runs 720.
    control = tmp_path / "depth.mp4"
    control.touch()
    model_config = SimpleNamespace(resolution="480", fixed_step_sampler_config=None, video_temporal_causal=False)
    sample = OmniSampleOverrides(
        depth={"control_path": str(control)}, resolution="256", transfer_resize_mode="direct_resize"
    )
    with pytest.raises(ValueError, match="teacher's trained 480 and 720 tiers only, got resolution='256'"):
        sample._build_transfer_data(model_config, sample.sample_meta, user_fields=frozenset())
