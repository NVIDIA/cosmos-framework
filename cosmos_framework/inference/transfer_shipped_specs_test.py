# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Shipped Transfer specs and requests must build without errors or ignored-setting warnings."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from cosmos_framework.inference import args as args_module
from cosmos_framework.inference import transfer
from cosmos_framework.inference.args import OmniSampleOverrides

pytestmark = [pytest.mark.L0, pytest.mark.CPU]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_COOKBOOK_SPECS = _REPO_ROOT / ".agents/skills/cookbooks/cosmos3/generator/transfer/specs"
_SIM_TRANSFER_SKILL = _REPO_ROOT / ".agents/skills/cosmos3-nano-sim-transfer"


def _nano_model_config() -> SimpleNamespace:
    return SimpleNamespace(
        action_gen=False,
        sound_gen=False,
        resolution="720",
        video_temporal_causal=False,
        fixed_step_sampler_config=None,
        tokenizer=SimpleNamespace(temporal_compression_factor=4),
        vlm_config=SimpleNamespace(model_name="Qwen/Qwen3-VL-8B-Instruct"),
        rectified_flow_inference_config=SimpleNamespace(scheduler_type="unipc"),
    )


def _sim_transfer_model_config() -> SimpleNamespace:
    return SimpleNamespace(
        action_gen=False,
        sound_gen=False,
        resolution="480",
        video_temporal_causal=True,
        causal_training_strategy="none",
        simulation_mode=None,
        fixed_step_sampler_config=SimpleNamespace(t_list=[1.0, 0.9375, 0.8333333333333334, 0.625]),
        teacher_forcing_frames_per_chunk=1,
        teacher_forcing_replay_policy=SimpleNamespace(
            control_visibility="causal", controls_read_strict_past_clean_rgb=True, clean_pass_causality="frame"
        ),
        tokenizer=SimpleNamespace(temporal_compression_factor=4),
        vlm_config=SimpleNamespace(model_name="Qwen/Qwen3-VL-8B-Instruct"),
        rectified_flow_inference_config=SimpleNamespace(scheduler_type="unipc"),
    )


def _record_warnings(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    warnings: list[str] = []
    monkeypatch.setattr(args_module.log, "warning", lambda message, *_a, **_k: warnings.append(message))
    return warnings


@pytest.mark.parametrize("spec", sorted(_COOKBOOK_SPECS.glob("*.json")), ids=lambda path: path.stem)
def test_cookbook_transfer_spec_builds_without_ignored_setting_warnings(
    spec: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (overrides,) = OmniSampleOverrides.from_files([spec])
    if str(overrides.vision_path).startswith("https://"):
        # Keep the test offline; only the request fields matter for building the sample.
        stand_in = tmp_path / "vision.mp4"
        stand_in.touch()
        overrides.vision_path = str(stand_in)
    warnings = _record_warnings(monkeypatch)
    # The cookbook runs from the repository root, so the spec's relative negative_prompt_file does not
    # resolve; it must not be needed, because the defaults already supply the negative caption.
    monkeypatch.chdir(tmp_path)
    overrides.output_dir = tmp_path / "out"
    overrides.download(overrides.output_dir / "inputs")

    sample = overrides.build_sample(model_config=_nano_model_config())

    assert not [w for w in warnings if w.startswith("[transfer]")]
    assert {"fps", "num_frames"} <= sample.requested_fields


def _sim_transfer_requests(tmp_path: Path) -> list[Path]:
    """Write the skill's official requests with prepare_hf_sample.py, on empty stand-in media."""
    module_spec = importlib.util.spec_from_file_location(
        "prepare_hf_sample", _SIM_TRANSFER_SKILL / "scripts/prepare_hf_sample.py"
    )
    assert module_spec is not None and module_spec.loader is not None
    prepare_hf_sample = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(prepare_hf_sample)
    requests: list[Path] = []
    for example in ("example1", "example2"):
        sample_dir = tmp_path / "sample" / example
        sample_dir.mkdir(parents=True)
        for suffix in (".mp4", "_depth.mp4", "_segmentation.mp4"):
            (sample_dir / f"{example}{suffix}").touch()
        prompt = sample_dir / f"{example}_upsampled_prompt.json"
        prompt.write_text(json.dumps({"prompt": "a robot"}) if example == "example1" else "a street")
        prepare_hf_sample.prepare(example, sample_dir, tmp_path / example)
        requests += sorted((tmp_path / example / "requests").glob("*.json"))
    return requests


def test_sim_transfer_skill_requests_build_without_ignored_setting_warnings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests = _sim_transfer_requests(tmp_path)
    assert len(requests) == 6
    # The skill's command: its defaults file plus these CLI flags.
    cli = OmniSampleOverrides(
        defaults_file=_SIM_TRANSFER_SKILL / "inference/assets/sample_args.json",
        autoregressive=True,
        seed=2025,
        num_steps=4,
        guidance=1.0,
        control_guidance=1.0,
        prompt_upsampling=False,
    )
    warnings = _record_warnings(monkeypatch)
    for request in requests:
        for overrides in OmniSampleOverrides.from_files([request], overrides=cli):
            overrides.output_dir = tmp_path / "out" / request.stem
            overrides.download(overrides.output_dir / "inputs")
            overrides.build_sample(model_config=_sim_transfer_model_config())

    assert not [w for w in warnings if w.startswith("[transfer]")]


@pytest.mark.parametrize(
    ("requested", "total_frames", "input_fps", "expected"),
    [
        ({"fps": 30, "num_frames": 121}, 121, 30.0, []),
        ({"fps": 30, "num_frames": 121}, 121, 29.97, []),
        ({"fps": 10}, 121, 30.0, ["'fps'=10 has no effect with a video input"]),
        ({"fps": 10}, 1, 0.0, []),
        ({"num_frames": 61}, 121, 30.0, ["'num_frames'=61 does not set the Transfer output length"]),
        ({"num_frames": 601}, 600, 30.0, []),
        ({}, 93, 16.0, []),
    ],
)
def test_requested_fps_and_num_frames_warn_only_when_the_source_disagrees(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    requested: dict,
    total_frames: int,
    input_fps: float,
    expected: list[str],
) -> None:
    control = tmp_path / "depth.mp4"
    control.touch()
    overrides = OmniSampleOverrides(
        name="sample", depth={"control_path": str(control)}, output_dir=tmp_path, **requested
    )
    overrides.download(tmp_path / "inputs")
    sample = overrides.build_sample(model_config=_nano_model_config())
    warnings: list[str] = []
    monkeypatch.setattr(transfer.log, "warning", lambda message, *_a, **_k: warnings.append(message))

    transfer._warn_unused_length_and_fps(sample, total_frames, input_fps, temporal_compression_factor=4)

    assert len(warnings) == len(expected)
    for message, text in zip(warnings, expected, strict=True):
        assert text in message
