# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Auto-only clock validation and output retiming, without loading model weights."""

import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import tyro
from torch.utils._pytree import tree_flatten

from cosmos_framework.inference.args import OmniSampleArgs, OmniSampleOverrides, check_auto_fps_options
from cosmos_framework.inference.common.auto_fps import auto_camera_fps
from cosmos_framework.inference.inference import OmniInference
from cosmos_framework.inference.multiview import get_multiview_sample_data

pytestmark = [pytest.mark.level(0), pytest.mark.gpus(0)]


@pytest.fixture
def request_data(tmp_path: Path) -> dict[str, Any]:
    control = tmp_path / "wsm.mp4"
    control.touch()
    return {
        "name": "scene",
        "output_dir": str(tmp_path),
        "prompt": "An overcast road.",
        "num_frames": 33,
        "seed": 0,
        "wsm": {},
        "multiview": {
            "views": [{"camera_key": "camera_front_wide_120fov", "control_path": str(control)}],
        },
    }


@pytest.fixture(params=[False, True], ids=["transfer-auto", "sim-auto"])
def model_config(request: pytest.FixtureRequest) -> SimpleNamespace:
    # Existing fields in both published Auto configs; no new checkpoint metadata.
    metadata = {"per_view_captions": True, "per_view_caption_task": "av_joint_camera_lidar_transfer"}
    causal = request.param
    return SimpleNamespace(
        _auto_camera_fps=auto_camera_fps(metadata),
        action_gen=False,
        sound_gen=False,
        resolution="480",
        tokenizer=SimpleNamespace(temporal_compression_factor=4),
        vlm_config=SimpleNamespace(model_name="Qwen/Qwen3-VL-8B-Instruct"),
        video_temporal_causal=causal,
        teacher_forcing_kv_implementation="multiview_maskless_kv" if causal else None,
        diffusion_expert_config=SimpleNamespace(base_fps=30),
        fixed_step_sampler_config=None,
        lidar_tokenizer=object(),
    )


@pytest.mark.parametrize("output_fps", [None, 10, 30])
def test_auto_defaults_and_output_only_change(
    request_data: dict[str, Any], model_config: SimpleNamespace, output_fps: int | None
) -> None:
    baseline = OmniSampleOverrides(**request_data).build_sample(model_config=model_config)
    sample = OmniSampleOverrides(**request_data, output_fps=output_fps).build_sample(model_config=model_config)
    assert sample.fps == 30
    assert sample.video_output_fps == (output_fps or 30)
    assert sample.output_fps == sample.video_output_fps
    assert sample.model_dump(exclude={"output_fps"}) == baseline.model_dump(exclude={"output_fps"})
    assert sample.duration == baseline.duration


@pytest.mark.parametrize("source", ["request", "cli", "defaults"])
def test_explicit_bad_fps_is_not_silently_overwritten(
    tmp_path: Path, request_data: dict[str, Any], model_config: SimpleNamespace, source: str
) -> None:
    data = dict(request_data)
    cli = None
    if source == "request":
        data["fps"] = 10
    elif source == "cli":
        cli = tyro.cli(OmniSampleOverrides, args=["--fps", "10"])
    else:
        defaults = tmp_path / "defaults.json"
        defaults.write_text(json.dumps({"fps": 10}))
        data["defaults_file"] = str(defaults)
    input_file = tmp_path / "input.jsonl"
    input_file.write_text(json.dumps(data) + "\n")
    sample = OmniSampleOverrides.from_files([input_file], overrides=cli)[0]
    with pytest.raises(ValueError, match="requires model conditioning fps=30.*--output-fps 10"):
        sample.build_sample(model_config=model_config)


def test_cli_output_fps_and_cli_precedence(
    tmp_path: Path, request_data: dict[str, Any], model_config: SimpleNamespace
) -> None:
    input_file = tmp_path / "input.json"
    input_file.write_text(json.dumps(request_data | {"fps": 10}))
    cli = tyro.cli(OmniSampleOverrides, args=["--fps", "30", "--output-fps", "10"])
    sample = OmniSampleOverrides.from_files([input_file], overrides=cli)[0].build_sample(model_config=model_config)
    assert (sample.fps, sample.output_fps) == (30, 10)


def test_joint_and_lidar_only_clocks(
    tmp_path: Path, request_data: dict[str, Any], model_config: SimpleNamespace
) -> None:
    control = tmp_path / "lidar.tar"
    control.touch()
    joint = request_data | {"lidar": {"control_path": str(control)}}
    sample = OmniSampleOverrides(**joint).build_sample(model_config=model_config)
    assert (sample.fps, sample.output_fps) == (30, 30)
    with pytest.raises(ValueError, match="Joint RGB/LiDAR retiming is not supported"):
        OmniSampleOverrides(**joint, output_fps=10).build_sample(model_config=model_config)
    lidar_only = {key: value for key, value in joint.items() if key not in {"wsm", "multiview"}}
    sample = OmniSampleOverrides(**lidar_only, fps=10).build_sample(model_config=model_config)
    assert sample.fps == 10
    assert sample.output_fps is None
    with pytest.raises(ValueError, match="only supported for identified Auto camera models"):
        OmniSampleOverrides(**lidar_only, output_fps=10).build_sample(model_config=model_config)


@pytest.mark.parametrize("fps", [10, 24, 30])
def test_other_models_keep_existing_fps_behavior(
    request_data: dict[str, Any], model_config: SimpleNamespace, fps: int
) -> None:
    del model_config._auto_camera_fps
    sample = OmniSampleOverrides(**request_data, fps=fps).build_sample(model_config=model_config)
    assert sample.fps == fps
    assert sample.output_fps is None
    with pytest.raises(ValueError, match="only supported for identified Auto camera models"):
        OmniSampleOverrides(**request_data, fps=fps, output_fps=10).build_sample(model_config=model_config)


def test_modified_built_sample_is_also_validated(request_data: dict[str, Any], model_config: SimpleNamespace) -> None:
    sample = OmniSampleOverrides(**request_data).build_sample(model_config=model_config)
    with pytest.raises(ValueError, match="requires model conditioning fps=30"):
        check_auto_fps_options(sample.model_copy(update={"fps": 10}), model_config)


def test_identification_is_not_a_path_or_architecture_guess() -> None:
    assert auto_camera_fps({"per_view_caption_task": "av_multiview_transfer"}) == 30
    for name in ("nvidia/Cosmos3-Nano-Transfer-Auto", "nvidia/Cosmos3-Nano-Sim-Auto"):
        assert auto_camera_fps(None, repository=name) == 30
    assert auto_camera_fps(None, repository="/checkpoints/Cosmos3-Nano-Sim-Auto") is None
    assert auto_camera_fps({"per_view_captions": True, "base_fps": 30, "num_view_embeddings": 12}) is None


def test_output_fps_does_not_change_model_batch_or_caption_clock(
    request_data: dict[str, Any], model_config: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(OmniSampleArgs, "vision_size", property(lambda self: (16, 16)))
    model = SimpleNamespace(config=model_config, input_caption_key="caption")
    control = torch.arange(33, dtype=torch.uint8).reshape(1, 33, 1, 1).expand(3, 33, 16, 16)  # [3,F,H,W]
    batches: list[dict[str, Any]] = []
    for output_fps in (30, 10):
        sample = OmniSampleOverrides(**request_data, output_fps=output_fps).build_sample(model_config=model_config)
        batches.append(
            get_multiview_sample_data(
                sample,
                model,
                device="cpu",
                condition_frames=[None],
                control_frames=[control],
                view_prompts=["[0s-1s] A car follows the road."],
            )
        )
    # Includes actual FPS tensors, frame counts, WSM pixels, prompt wrappers and timestamp text.
    before, before_structure = tree_flatten(batches[0])
    after, after_structure = tree_flatten(batches[1])
    assert before_structure == after_structure
    for left, right in zip(before, after, strict=True):
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        else:
            assert left == right
    assert batches[1]["fps"].item() == batches[1]["conditioning_fps"].item() == 30


def test_real_rgb_and_wsm_saver_preserves_frames_at_output_fps(
    tmp_path: Path, request_data: dict[str, Any], model_config: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    if shutil.which("ffprobe") is None:
        pytest.skip("ffprobe is required to inspect encoded frame count and playback rate")
    sample = OmniSampleOverrides(**request_data, output_fps=10).build_sample(model_config=model_config)
    pipe = OmniInference(
        setup_args=SimpleNamespace(debug=False),
        model=SimpleNamespace(input_video_key="video"),
        guardrails=None,
        _timer=None,
    )
    monkeypatch.setattr(pipe, "_run_video_guardrail", lambda name, video: video)
    video = torch.linspace(0, 1, 300).reshape(1, 300, 1, 1).expand(3, 300, 16, 16).contiguous()  # [3,F,H,W]
    data = {
        "sample_n_views": torch.tensor([1]),  # [1]
        "num_video_frames_per_view": torch.tensor([300]),  # [1]
        "video": [(video * 255).to(torch.uint8).unsqueeze(0)],  # list[[1,3,F,H,W]]
    }
    outputs = pipe._save_multiview_sample(sample, data, {}, video)
    content = outputs.outputs[0].content
    for field in ("vision_files_by_camera", "control_files_by_camera"):
        path = content[field]["camera_front_wide_120fov"][0]
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-count_frames", "-show_streams", "-of", "json", path],
            check=True,
            capture_output=True,
            text=True,
        )
        stream = json.loads(probe.stdout)["streams"][0]
        assert int(stream["nb_read_frames"]) == 300
        assert stream["avg_frame_rate"] == "10/1"
        assert float(stream["duration"]) == pytest.approx(30.0)
    recorded = json.loads((tmp_path / "sample_outputs.json").read_text())["args"]
    assert (recorded["fps"], recorded["output_fps"]) == (30, 10)
