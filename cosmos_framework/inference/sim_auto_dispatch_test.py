# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Routing of causal multiview (Cosmos3-Nano-Sim-Auto) samples through the shared inference pipeline.

The dispatch and guardrail cases port the reference implementation's causal simulation tests.
The end-to-end case drives ``generate_batch`` through the real
causal frontend and saver with a stub model, and the input cases build local Sim-Auto JSONL
requests through this repo's own argument models.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
import torch

from cosmos_framework.data.generator.augmentors.text_tokenizer import _SYSTEM_PROMPT_AV_JOINT_CAMERA_LIDAR_TRANSFER
from cosmos_framework.data.generator.multiview.camera_attributes import CAMERA_VIEW_MAPPING
from cosmos_framework.inference import inference
from cosmos_framework.inference import multiview_causal as causal_entry
from cosmos_framework.inference.args import OmniSampleArgs, OmniSampleOverrides
from cosmos_framework.inference.common.args import CheckpointType, ConfigFileType, SampleOutputs
from cosmos_framework.inference.inference import OmniInference, SampleDataset
from cosmos_framework.model.generator.tokenizers.lidar.range_projection import LidarRangeProjectionConfig
from cosmos_framework.utils.misc import TrainingTimer

pytestmark = [pytest.mark.L0, pytest.mark.CPU]

_REPO_DIR = Path(__file__).resolve().parents[2]
_CAMERAS = ("camera_front_wide_120fov", "camera_cross_left_120fov")

# The fields of the Cosmos3-Nano-Sim-Auto student's exported model config (an export that recorded
# the three rolling-history settings) that sample building, dispatch and the causal frontend read.
_SIM_AUTO_CONFIG: dict[str, Any] = {
    "resolution": "720",
    "video_temporal_causal": True,
    "teacher_forcing_kv_implementation": "multiview_maskless_kv",
    "fixed_step_sampler_config": {"sample_type": "sde", "t_list": [1.0, 0.9375, 0.8333333333333334, 0.625]},
    "rolling_kv_cache_chunks": 2,
    "rolling_kv_sink_chunks": 1,
    "rolling_prompt_policy": "segmented",
    "rolling_text_sink_tokens": 4,
    "rolling_prompt_context_before": 0,
    "rolling_prompt_context_after": 0,
    "rolling_media_text_extent": None,
    "kv_cache_dtype": None,
    "kv_cache_inference_size": None,
    "attention_sink_size": 0,
    "lidar_fps": 10.0,
    "diffusion_expert_config": {"base_fps": 30, "num_view_embeddings": 12},
}

# The same fields of the bidirectional Cosmos3-Nano-Transfer-Auto export, which shares Sim-Auto's rig
# and sensor clocks and has no causal or rolling settings.
_TRANSFER_AUTO_CONFIG: dict[str, Any] = {
    "resolution": "720",
    "video_temporal_causal": False,
    "teacher_forcing_kv_implementation": None,
    "fixed_step_sampler_config": None,
    "lidar_fps": 10.0,
    "diffusion_expert_config": {"base_fps": 30, "num_view_embeddings": 12},
}


def _model_config(exported: dict[str, Any]) -> SimpleNamespace:
    fields = {
        key: SimpleNamespace(**value) if isinstance(value, dict) else value
        for key, value in copy.deepcopy(exported).items()
    }
    return SimpleNamespace(
        action_gen=False,
        sound_gen=False,
        tokenizer=SimpleNamespace(temporal_compression_factor=4),
        vlm_config=SimpleNamespace(model_name="Qwen/Qwen3-VL-8B-Instruct"),
        lidar_tokenizer=object(),
        **fields,
    )


def _caption_file(path: Path, camera: str, spans: list[tuple[int, int]]) -> str:
    path.write_text(
        json.dumps(
            {
                "caption_structured": {
                    f"chunk_{start}_{end}": {
                        "caption": f"The {camera} view at frame {start}.",
                        "start_frame": start,
                        "end_frame": end,
                    }
                    for start, end in spans
                }
            }
        )
    )
    return str(path)


def _sample(
    tmp_path: Path,
    config: SimpleNamespace,
    *,
    joint: bool = False,
    captions: bool = True,
    num_frames: int = 9,
    **overrides: Any,
) -> OmniSampleArgs:
    """A per-camera WSM transfer sample, optionally with a LiDAR control (joint camera+LiDAR)."""
    views = []
    for index, camera in enumerate(_CAMERAS):
        control = tmp_path / f"control_{index}.mp4"
        control.touch()
        view = {"camera_key": camera, "control_path": str(control)}
        if captions:
            # Two caption segments per camera: the horizon consumes both.
            view["caption_path"] = _caption_file(tmp_path / f"{camera}.json", camera, [(0, 5), (5, 33)])
        views.append(view)
    if joint:
        lidar = tmp_path / "hdmap_rangemap.tar"
        lidar.touch()
        overrides["lidar"] = {"control_path": str(lidar)}
    return OmniSampleOverrides(
        name="sample",
        output_dir=tmp_path / "out",
        prompt="A car.",
        num_frames=num_frames,
        seed=0,
        wsm={},
        multiview={"load_caption_from_data": captions, "use_first_chunk_only": False, "views": views},
        **overrides,
    ).build_sample(model_config=config)


def _dispatch_pipe(config: SimpleNamespace) -> SimpleNamespace:
    """Every pipeline attribute ``generate_batch`` reads on its multiview branches."""
    return SimpleNamespace(
        model=SimpleNamespace(config=config),
        per_view_captions=True,
        per_view_caption_task="av_joint_camera_lidar_transfer",
        _generate_causal_multiview_batch=MagicMock(return_value=["causal"]),
        _generate_joint_cam_lidar_batch=MagicMock(return_value=["joint-bidi"]),
        _generate_multiview_caption_chunks_batch=MagicMock(return_value=["rgb-bidi"]),
    )


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("joint", [False, True])
def test_shared_pipeline_dispatches_before_bidirectional_branches(tmp_path: Path, causal: bool, joint: bool) -> None:
    config = _model_config(_SIM_AUTO_CONFIG if causal else _TRANSFER_AUTO_CONFIG)
    sample = _sample(tmp_path, config, joint=joint)
    pipe = _dispatch_pipe(config)
    result = OmniInference.generate_batch(cast(OmniInference, pipe), [sample], {}, save_outputs=False)
    assert result == (["causal"] if causal else ["joint-bidi" if joint else "rgb-bidi"])
    assert pipe._generate_causal_multiview_batch.called is causal
    if causal:
        pipe._generate_causal_multiview_batch.assert_called_once_with(sample, save_outputs=False)


@pytest.mark.parametrize("causal", [False, True])
def test_causal_dispatch_precedes_the_caption_chunk_refusal(tmp_path: Path, causal: bool) -> None:
    """The per-camera caption refusal names the bidirectional caption-chunk path.

    A causal sample must reach its own frontend, which refuses the same input with the causal
    runtime's requirement, instead of being told about a path it never takes.
    """
    config = _model_config(_SIM_AUTO_CONFIG if causal else _TRANSFER_AUTO_CONFIG)
    sample = _sample(tmp_path, config, captions=False)
    pipe = _dispatch_pipe(config)
    if causal:
        assert OmniInference.generate_batch(cast(OmniInference, pipe), [sample], {}) == ["causal"]
    else:
        with pytest.raises(NotImplementedError, match="caption-chunk path"):
            OmniInference.generate_batch(cast(OmniInference, pipe), [sample], {})
        pipe._generate_causal_multiview_batch.assert_not_called()


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("joint", [False, True])
def test_sample_dataset_leaves_causal_batches_to_the_causal_sampler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, causal: bool, joint: bool
) -> None:
    """``create_batches`` reads every sample through the dataset before ``generate_batch`` runs.

    A causal sample must not be assembled there: the one-shot builder would decode every camera's
    full media for a batch the causal sampler never reads.
    """
    one_shot_batch = {"video": "one-shot batch"}
    builder = MagicMock(return_value=one_shot_batch)
    monkeypatch.setattr("cosmos_framework.inference.multiview.get_multiview_sample_data", builder)
    config = _model_config(_SIM_AUTO_CONFIG if causal else _TRANSFER_AUTO_CONFIG)
    sample = _sample(tmp_path, config, joint=joint)
    _, data = SampleDataset([sample], cast(Any, SimpleNamespace(config=config)))[0]
    builds_one_shot_batch = not causal and not joint
    assert data == (one_shot_batch if builds_one_shot_batch else {})
    assert builder.called is builds_one_shot_batch


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("causal", [False, True])
def test_diffusion_cache_is_not_installed_on_causal_multiview_replay(
    monkeypatch: pytest.MonkeyPatch, causal: bool, enabled: bool
) -> None:
    from cosmos_framework.model.generator.mot import diffusion_cache

    install = MagicMock()
    info = MagicMock()
    monkeypatch.setattr(diffusion_cache, "install_diffusion_cache", install)
    monkeypatch.setattr(inference.log, "info", info)
    pipe = SimpleNamespace(
        model=SimpleNamespace(config=_model_config(_SIM_AUTO_CONFIG if causal else _TRANSFER_AUTO_CONFIG))
    )
    setup_args = SimpleNamespace(
        diffusion_cache=enabled,
        diffusion_cache_thresh=None,
        diffusion_cache_residual_order=None,
        diffusion_cache_max_consecutive_cached=None,
    )
    OmniInference._maybe_install_diffusion_cache(cast(OmniInference, pipe), cast(Any, setup_args))
    if enabled and not causal:
        install.assert_called_once_with(pipe=pipe, enabled=True, sample_args_list=[], config_overrides=None)
    else:
        install.assert_not_called()
    # Skipping a requested cache is reported; an unrequested one is simply absent.
    skip_reported = [call for call in info.call_args_list if "causal multiview" in call.args[0]]
    assert len(skip_reported) == int(enabled and causal)


def _result(chunks: list[dict[str, Any]]) -> causal_entry.CausalMultiviewResult:
    return causal_entry.CausalMultiviewResult(
        vision=[torch.zeros(3, 9, 2, 2) for _ in _CAMERAS],  # list[[3,F,H,W]]
        controls=[torch.zeros(3, 9, 2, 2, dtype=torch.uint8) for _ in _CAMERAS],  # list[[3,F,H,W]]
        lidar=None,
        lidar_control=None,
        chunks=chunks,
        rolling_prompt={"policy": "segmented"},
        latents=[],
        save_latents=False,
    )


def _wrapper_pipe(*, processes_sample: bool) -> SimpleNamespace:
    """Every pipeline attribute the causal batch wrapper reads, with the frontend patched out."""
    return SimpleNamespace(
        setup_args=SimpleNamespace(benchmark=False),
        _get_timer=lambda name: nullcontext(),
        should_process_sample=MagicMock(return_value=processes_sample),
        _run_text_guardrail=MagicMock(),
        _handle_sample_exception=MagicMock(return_value="handled"),
    )


@pytest.mark.parametrize("blocked", [False, True])
def test_causal_text_guardrail_checks_consumed_view_captions_before_saving(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, blocked: bool
) -> None:
    sample = _sample(tmp_path, _model_config(_SIM_AUTO_CONFIG))
    prompts = ["front caption", "side caption", "later front caption", "later side caption"]
    monkeypatch.setattr(
        causal_entry,
        "generate_causal_multiview_sample",
        MagicMock(return_value=_result([{"prompt": prompts[:2]}, {"prompt": prompts[2:]}])),
    )
    saved = SampleOutputs(args=sample.model_dump(mode="json"), outputs=[])
    save = MagicMock(return_value=saved)
    monkeypatch.setattr(causal_entry, "save_causal_multiview_outputs", save)
    monkeypatch.setattr(inference, "sync_distributed_errors", nullcontext)
    pipe = _wrapper_pipe(processes_sample=True)
    if blocked:
        pipe._run_text_guardrail.side_effect = [None, None, None, ValueError("blocked caption")]
    outputs = OmniInference._generate_causal_multiview_batch(cast(OmniInference, pipe), sample, save_outputs=True)
    assert [call.args for call in pipe._run_text_guardrail.call_args_list] == [
        (str(sample.output_dir), prompt) for prompt in prompts
    ]
    assert save.called is not blocked
    assert pipe._handle_sample_exception.called is blocked
    assert outputs == (["handled"] if blocked else [saved])


@pytest.mark.parametrize("save_outputs", [False, True])
def test_dummy_replica_participates_in_save_barrier_without_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, save_outputs: bool
) -> None:
    """The padding batch a short replica runs: it samples (collectives) and joins both barriers, but writes nothing."""
    sample = _sample(tmp_path, _model_config(_SIM_AUTO_CONFIG), save_latents=True).model_copy(
        update={"output_dir": None}
    )
    generate = MagicMock(return_value=_result([{"prompt": ["front", "side"]}]))
    save = MagicMock()
    sync = MagicMock(side_effect=nullcontext)
    monkeypatch.setattr(causal_entry, "generate_causal_multiview_sample", generate)
    monkeypatch.setattr(causal_entry, "save_causal_multiview_outputs", save)
    monkeypatch.setattr(inference, "sync_distributed_errors", sync)
    pipe = _wrapper_pipe(processes_sample=False)
    assert (
        OmniInference._generate_causal_multiview_batch(cast(OmniInference, pipe), sample, save_outputs=save_outputs)
        == []
    )
    # Latents are only kept for a pass that saves.
    generate.assert_called_once_with(pipe, sample, save_latents=save_outputs)
    save.assert_not_called()
    pipe._run_text_guardrail.assert_not_called()
    assert sync.call_count == 2 * int(save_outputs)


def _tokenize(captions: list[str], *, use_system_prompt: bool, system_prompt: str, is_video: bool) -> list[list[int]]:
    return [list((system_prompt + caption).encode()) for caption in captions]


@pytest.mark.parametrize("joint", [False, True])
@pytest.mark.parametrize("save_latents", [False, True])
def test_generate_batch_runs_the_causal_frontend_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, joint: bool, save_latents: bool
) -> None:
    """``generate_batch`` -> causal wrapper -> rolling frontend -> saver, with only media and the sampler stubbed.

    Checks what the wrapper owns on the real call path: the benchmark timer and its device
    synchronisation, the guardrail over the captions the schedule actually consumed, latent saving,
    and the saved layout.
    """
    size = 16
    monkeypatch.setattr(OmniSampleArgs, "vision_size", property(lambda self: (size, size)))
    config = _model_config(_SIM_AUTO_CONFIG)
    sample = _sample(tmp_path, config, joint=joint, save_latents=save_latents)
    assert sample.autoregressive and sample.output_dir is not None and sample.multiview is not None
    num_views, num_frames = len(_CAMERAS), sample.num_frames
    sweeps = 3  # 9 camera frames at 30 fps against 10 Hz LiDAR.

    # The partitioner each media loader received: this single-rank model has none, so every view loads here.
    loader_partitioners: dict[str, Any] = {}

    def load_controls(source: OmniSampleArgs, *, height: int, width: int, partitioner: Any) -> list[torch.Tensor]:
        loader_partitioners["control"] = partitioner
        return [torch.zeros(3, source.num_frames, height, width, dtype=torch.uint8) for _ in range(num_views)]

    def load_conditions(
        sample: OmniSampleArgs, model: Any, *, height: int, width: int, partitioner: Any
    ) -> list[torch.Tensor | None]:
        loader_partitioners["condition"] = partitioner
        return [None] * num_views

    def sample_chunks(
        model: Any, sample: OmniSampleArgs, data: dict[str, Any], *, save_latents: bool = False
    ) -> dict[str, torch.Tensor]:
        assert data["rolling_prompt_schedule"].num_frames == num_frames
        result = {"vision": torch.rand(1, 3, num_views * num_frames, size, size)}  # [1,3,V*F,H,W]
        if save_latents:
            result["vision_latent"] = torch.randn(1, 4, num_views * 3, 1, 1)  # [1,C,V*T,h,w]
        if joint:
            lidar = torch.stack(
                [torch.full((sweeps, 2, 8), 20.0), torch.full((sweeps, 2, 8), 0.5), torch.ones(sweeps, 2, 8)]
            )
            result["lidar"] = lidar[None]  # [1,3,F_lidar,Hl,Wl]
            result["lidar_control"] = torch.zeros(1, 3, sweeps, 2, 8)  # [1,C,F_lidar,Hl,Wl]
            if save_latents:
                result["lidar_latent"] = torch.randn(1, 4, sweeps, 1, 1)  # [1,C,T_lidar,h,w]
        return result

    monkeypatch.setattr(causal_entry, "_get_multiview_control_pixel_frames", load_controls)
    monkeypatch.setattr(causal_entry, "_get_multiview_condition_pixel_frames", load_conditions)
    monkeypatch.setattr(causal_entry, "get_multiview_sample_data", lambda *args, **kwargs: {})
    monkeypatch.setattr(causal_entry, "_finalize_data_batch", lambda **kwargs: kwargs["data_batch"])
    monkeypatch.setattr(
        causal_entry,
        "prepare_joint_batch",
        lambda *args: {"system_prompt": _SYSTEM_PROMPT_AV_JOINT_CAMERA_LIDAR_TRANSFER},
    )
    monkeypatch.setattr(causal_entry, "generate_joint_chunk" if joint else "generate_rgb_chunk", sample_chunks)
    monkeypatch.setattr(inference, "sync_distributed_errors", nullcontext)

    events: list[str] = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: events.append("synchronize"))
    model = SimpleNamespace(
        config=config,
        input_caption_key="ai_caption",
        tensor_kwargs={"device": "cpu"},
        parallel_dims=None,
        vlm_config=SimpleNamespace(use_system_prompt=False),
        llm_special_tokens={"eos_token_id": 2, "start_of_generation": 3},
        _tokenize_captions=_tokenize,
        tokenizer_lidar_gen=SimpleNamespace(
            range_projection=LidarRangeProjectionConfig(
                native_height=2, native_width=8, semantic_height=2, semantic_width=8, model_width=8
            )
        ),
    )
    # The setup of an exported checkpoint, as the causal frontend reads it.
    setup_args = SimpleNamespace(
        benchmark=True,
        warmup=0,
        keep_going=False,
        use_separate_pipeline_vision_decode_gpu=False,
        checkpoint_type=CheckpointType.HF,
        config_file_type=ConfigFileType.MODULE,
    )
    pipe = OmniInference(
        setup_args=cast(Any, setup_args),
        model=cast(Any, model),
        guardrails=None,
        _timer=TrainingTimer(),
    )
    pipe.per_view_captions = True
    timer = pipe._get_timer

    @contextmanager
    def recording_timer(name: str) -> Iterator[None]:
        with timer(name):
            events.append(f"enter {name}")
            yield
            events.append(f"exit {name}")

    guarded: list[tuple[str, str]] = []
    monkeypatch.setattr(pipe, "_get_timer", recording_timer)
    monkeypatch.setattr(pipe, "_run_text_guardrail", lambda name, prompt: guarded.append((name, prompt)))

    [output] = pipe.generate_batch([sample], {}, save_outputs=True)

    assert loader_partitioners == {"condition": None, "control": None}
    assert events == [
        "synchronize",
        "enter inference.encode_generate_decode",
        "enter inference.prepare_inputs",
        "exit inference.prepare_inputs",
        "enter inference.generate_decode",
        "exit inference.generate_decode",
        "synchronize",
        "exit inference.encode_generate_decode",
        "enter inference.save_outputs",
        "exit inference.save_outputs",
    ]
    timer_results = pipe.get_timer_results()
    assert timer_results is not None
    assert {key: len(values) for key, values in timer_results["all"].items()} == {
        "inference.encode_generate_decode": 1,
        "inference.prepare_inputs": 1,
        "inference.generate_decode": 1,
        "inference.save_outputs": 1,
    }

    # Every camera's caption of every consumed segment, as the schedule formatted it, in order.
    output_dir = sample.output_dir
    consumed = json.loads((output_dir / "caption_chunks.json").read_text())
    assert [prompt for chunk in consumed for prompt in chunk["prompt"]] == [prompt for _, prompt in guarded]
    assert {name for name, _ in guarded} == {str(output_dir)}
    expected_captions = [f"The {camera} view at frame {start}." for start in (0, 5) for camera in _CAMERAS]
    assert len(guarded) == len(expected_captions)
    for (_, prompt), caption in zip(guarded, expected_captions, strict=True):
        assert caption in prompt

    assert output.status == "success"
    files = {path.name for path in output.outputs[0].files}
    assert files == {path.name for path in output_dir.iterdir()} - {"sample_outputs.json"}
    assert len([name for name in files if name.startswith("vision_view")]) == num_views
    assert len([name for name in files if name.startswith("control_view")]) == num_views
    assert ("latents.pt" in files) is save_latents
    assert ({"lidar.safetensors", "lidar.pt"} <= files) is joint
    assert json.loads((output_dir / "sample_args.json").read_text())["save_latents"] is save_latents


def test_documented_input_glob_gives_every_sample_its_own_output_directory() -> None:
    """``-i "inputs/omni/*.json"`` (docs/inference.md) writes each sample to ``<output_dir>/<name>``.

    Nothing rejects two inputs that share a name: the second sample would overwrite the first
    one's files and leave the rest.
    """
    samples = OmniSampleOverrides.from_files(sorted((_REPO_DIR / "inputs" / "omni").glob("*.json")))
    names = [sample.name for sample in samples]
    assert len(names) == len(set(names))


@pytest.mark.parametrize("joint", [False, True])
def test_sim_auto_local_jsonl_builds_a_sample(tmp_path: Path, joint: bool) -> None:
    """Local bundle paths resolve against the JSONL, preserving camera and sampling semantics.

    Media files are empty placeholders: this exercises native argument parsing and input-path
    resolution, not decoding or GPU generation. No remote sample or download is required.
    """
    cameras = (
        "camera_cross_left_120fov",
        "camera_front_wide_120fov",
        "camera_front_tele_30fov",
        "camera_cross_right_120fov",
        "camera_rear_left_70fov",
        "camera_rear_tele_30fov",
        "camera_rear_right_70fov",
    )
    bundle = tmp_path / "sample_data"
    scene = bundle / "daytime_scene"
    for folder in ("rgb", "wsm", "captions", "lidar"):
        (scene / folder).mkdir(parents=True)
    views = []
    for camera in cameras:
        for folder in ("rgb", "wsm"):
            (scene / folder / f"{camera}.mp4").touch()
        _caption_file(scene / "captions" / f"{camera}.json", camera, [(0, 300)])
        views.append(
            {
                "camera_key": camera,
                "control_path": f"daytime_scene/wsm/{camera}.mp4",
                "vision_path": f"daytime_scene/rgb/{camera}.mp4",
                "caption_path": f"daytime_scene/captions/{camera}.json",
            }
        )
    raw: dict[str, Any] = {
        "name": "daytime_scene",
        "model_mode": "image2video",
        "resolution": "720",
        "aspect_ratio": "16,9",
        "fps": 30,
        "num_frames": 297,
        "num_video_frames_per_chunk": 297,
        "seed": 0,
        "guidance": 1,
        "control_guidance": 1,
        "normalize_cfg": False,
        "autoregressive": True,
        "wsm": {},
        "condition_frame_indexes_vision": [0],
        "num_conditional_frames": 1,
        "prompt": "",
        "multiview": {
            "load_caption_from_data": True,
            "use_first_chunk_only": True,
            "condition_video_as_image": True,
            "views": views,
        },
    }
    if joint:
        (scene / "lidar" / "control.tar").touch()
        raw["lidar"] = {"control_path": "daytime_scene/lidar/control.tar"}
    input_file = bundle / "input-sim-auto.jsonl"
    input_file.write_text(json.dumps(raw) + "\n")

    [overrides] = OmniSampleOverrides.from_files([input_file])
    overrides.output_dir = tmp_path / "run" / str(overrides.name)
    overrides.download(overrides.output_dir / "inputs")
    config = _model_config(_SIM_AUTO_CONFIG)
    sample = overrides.build_sample(model_config=config)
    assert sample.multiview is not None

    # Every asset resolves inside the local bundle, independently of the process working directory.
    for view in sample.multiview.views:
        assert view.control_path == (scene / "wsm" / f"{view.camera_key}.mp4").resolve()
        assert view.vision_path == (scene / "rgb" / f"{view.camera_key}.mp4").resolve()
        assert view.caption_path == (scene / "captions" / f"{view.camera_key}.json").resolve()

    # The seven physical views retain their identities even outside canonical rig order.
    keys = sample.multiview.camera_keys
    assert [CAMERA_VIEW_MAPPING[key] for key in keys] == [5, 0, 6, 1, 4, 3, 2]
    assert sample.is_joint_cam_lidar_transfer is joint
    assert sample.is_multiview_transfer is not joint
    if joint:
        assert sample.lidar is not None
        assert sample.lidar.control_path == str((scene / "lidar" / "control.tar").resolve())

    # Per-camera captions over a 297-frame horizon, AR at the export's clock, i2v from frame 0
    # at 720p, and the distilled student's sampling settings.
    assert sample.multiview.load_caption_from_data and sample.multiview.use_first_chunk_only
    assert all(view.caption_path is not None for view in sample.multiview.views)
    assert sample.autoregressive
    assert sample.num_frames == 297
    assert sample.fps == config.diffusion_expert_config.base_fps
    assert sample.name == "daytime_scene"
    assert sample.vision_size == (1280, 720)
    assert sample.condition_frame_indexes_vision == [0]
    assert (sample.guidance, sample.control_guidance, sample.normalize_cfg, sample.seed) == (1.0, 1.0, False, 0)
    assert sample.save_latents is False
