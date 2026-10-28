# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Tests for the causal multiview frontend: caption schedule, rolling sampler, decode and outputs.

Most cases are ported from the reference implementation's causal simulation tests and run against
this repo's own batch builders and caption loader. The model is a stub: a byte tokenizer that keeps
the system prompt visible in every token sequence, and fake samplers that return shaped tensors.
"""

from __future__ import annotations

import io
import json
import os
import tarfile
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from safetensors.torch import load_file

from cosmos_framework.data.generator.augmentors.duration_fps_text_timestamps import DurationFPSTextTimeStamps
from cosmos_framework.data.generator.augmentors.resolution_text_info import ResolutionTextInfo
from cosmos_framework.data.generator.augmentors.text_tokenizer import (
    _SYSTEM_PROMPT_AV_JOINT_CAMERA_LIDAR_TRANSFER,
    _SYSTEM_PROMPT_AV_MULTIVIEW_TRANSFER,
)
from cosmos_framework.inference import inference
from cosmos_framework.inference import multiview_causal as entry
from cosmos_framework.inference.args import NegativeMetadataMode, OmniSampleArgs, OmniSampleOverrides
from cosmos_framework.inference.common.args import CheckpointType, ConfigFileType
from cosmos_framework.inference.inference import (
    OmniInference,
    _format_prompt_with_template,
    _format_sample_prompt,
    _get_per_view_prompt_sample_data,
)
from cosmos_framework.model.generator.mot.replica_partition import WorkUnit, plan_partition
from cosmos_framework.model.generator.omni_mot_causal_model import OmniMoTCausalModel
from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel
from cosmos_framework.model.generator.tokenizers.lidar.range_projection import LidarRangeProjectionConfig
from cosmos_framework.model.generator.utils.rig_view_embedding import vision_view_ids
from cosmos_framework.tools.visualize.video import save_img_or_video
from cosmos_framework.utils.generator.multiview import safe_multiview_camera_name

pytestmark = [pytest.mark.L0, pytest.mark.CPU]

_CAMERAS = ("camera_front_wide_120fov", "camera_cross_left_120fov")
_EMPHASIS = "Follow the"


def _model_config(*, num_view_embeddings: int = 0) -> SimpleNamespace:
    """The causal multiview config fields sample building reads, as a Sim-Auto export records them."""
    return SimpleNamespace(
        action_gen=False,
        sound_gen=False,
        resolution="480",
        video_temporal_causal=True,
        teacher_forcing_kv_implementation="multiview_maskless_kv",
        tokenizer=SimpleNamespace(temporal_compression_factor=4),
        vlm_config=SimpleNamespace(model_name="Qwen/Qwen3-VL-8B-Instruct"),
        fixed_step_sampler_config=None,
        lidar_tokenizer=object(),
        lidar_fps=10.0,
        diffusion_expert_config=SimpleNamespace(num_view_embeddings=num_view_embeddings, base_fps=30),
    )


def _caption_file(path: Path, spans: list[tuple[int, int]]) -> str:
    path.write_text(
        json.dumps(
            {
                "caption_structured": {
                    f"chunk_{start}_{end}": {"caption": f"A car at {start}.", "start_frame": start, "end_frame": end}
                    for start, end in spans
                }
            }
        )
    )
    return str(path)


def _sample(tmp_path: Path, *, fps: int = 30, **overrides: Any) -> OmniSampleArgs:
    control = tmp_path / "control.mp4"
    control.touch()
    caption = _caption_file(tmp_path / "caption.json", [(0, 33)])
    return OmniSampleOverrides(
        name="sample",
        output_dir=tmp_path / "out",
        prompt="A car.",
        num_frames=33,
        fps=fps,
        wsm={},
        multiview={
            "load_caption_from_data": True,
            "views": [{"camera_key": key, "control_path": str(control), "caption_path": caption} for key in _CAMERAS],
        },
        **overrides,
    ).build_sample(model_config=_model_config())


def _joint(tmp_path: Path) -> dict[str, Any]:
    lidar_path = tmp_path / "control.pt"
    lidar_path.touch()
    return {"lidar": {"control_path": str(lidar_path)}}


def _tokenize(captions: list[str], *, use_system_prompt: bool, system_prompt: str, is_video: bool) -> list[list[int]]:
    assert not use_system_prompt and not is_video
    assert system_prompt in (_SYSTEM_PROMPT_AV_MULTIVIEW_TRANSFER, _SYSTEM_PROMPT_AV_JOINT_CAMERA_LIDAR_TRANSFER)
    return [list((system_prompt + caption).encode()) for caption in captions]


def _host() -> OmniMoTCausalModel:
    return cast(
        OmniMoTCausalModel,
        SimpleNamespace(
            input_caption_key="ai_caption",
            tensor_kwargs={"device": "cpu"},
            parallel_dims=None,
            vlm_config=SimpleNamespace(use_system_prompt=False),
            llm_special_tokens={"eos_token_id": 2, "start_of_generation": 3},
            config=SimpleNamespace(
                diffusion_expert_config=SimpleNamespace(num_view_embeddings=0),
                # The rolling profile and KV options of the Sim-Auto export.
                rolling_kv_cache_chunks=2,
                rolling_kv_sink_chunks=1,
                rolling_prompt_policy="segmented",
                rolling_text_sink_tokens=4,
                rolling_prompt_context_before=0,
                rolling_prompt_context_after=0,
                rolling_media_text_extent=None,
                kv_cache_dtype=None,
                kv_cache_inference_size=None,
                attention_sink_size=0,
                lidar_fps=10.0,
            ),
            _tokenize_captions=_tokenize,
        ),
    )


def _refuse_media(monkeypatch: pytest.MonkeyPatch, message: str) -> list[MagicMock]:
    loaders = [MagicMock(side_effect=AssertionError(message)) for _ in range(3)]
    for name, loader in zip(
        ("_get_multiview_condition_pixel_frames", "_get_multiview_control_pixel_frames", "load_lidar_control"),
        loaders,
        strict=True,
    ):
        monkeypatch.setattr(entry, name, loader)
    return loaders


def test_native_caption_boundaries_and_requested_short_tail_are_preserved(tmp_path: Path) -> None:
    sample = _sample(tmp_path)
    _caption_file(tmp_path / "caption.json", [(0, 7), (7, 20), (20, 40)])
    assert sample.multiview is not None
    chunks = entry.load_caption_chunks(sample.multiview.views, first_only=False)
    schedule, metadata = entry.prepare_rolling_prompt_schedule(
        sample, _host(), chunks, per_view_captions=True, system_prompt=_SYSTEM_PROMPT_AV_MULTIVIEW_TRANSFER
    )
    assert schedule.num_frames == sample.num_frames == 33
    assert [(segment.start_frame, segment.end_frame) for segment in schedule.views[0]] == [(0, 7), (7, 20), (20, 33)]
    assert [row["frame_range"] for row in metadata] == [[0, 6], [7, 19], [20, 32]]
    initial_sink = [doc.token_ids for doc in schedule.for_times((0.0,)).documents if doc.is_sink]
    for frame, expected_start in ((6, 0), (7, 7), (19, 7), (20, 20), (32, 20)):
        pack = schedule.for_times((frame / 30,))
        assert [doc.start_frame for doc in pack.documents if not doc.is_sink] == [expected_start] * 2
        assert [doc.token_ids for doc in pack.documents if doc.is_sink] == initial_sink
    with pytest.raises(ValueError, match="horizon"):
        schedule.for_times((33 / 30,))


def test_fixed10_first_caption_retains_301_frames_and_native_terminal_padding(tmp_path: Path) -> None:
    sample = _sample(tmp_path).model_copy(update={"num_frames": 301})
    _caption_file(tmp_path / "caption.json", [(0, 300), (300, 600)])
    assert sample.multiview is not None
    chunks = entry.load_caption_chunks(sample.multiview.views, first_only=True)
    schedule, metadata = entry.prepare_rolling_prompt_schedule(
        sample, _host(), chunks, per_view_captions=True, system_prompt=_SYSTEM_PROMPT_AV_MULTIVIEW_TRANSFER
    )
    assert len(schedule.views[0]) == 1 and schedule.views[0][0].end_frame == 301
    assert metadata[0]["source_frame_range"] == [0, 299]
    assert metadata[0]["frame_range"] == [0, 300]
    assert all(doc.start_frame == 0 for doc in schedule.for_times((10.0,)).documents)


@pytest.mark.parametrize("spans", [[(0, 7), (8, 40)], [(0, 8), (7, 40)]])
def test_caption_gaps_or_overlaps_cannot_shift_controls(tmp_path: Path, spans: list[tuple[int, int]]) -> None:
    sample = _sample(tmp_path)
    _caption_file(tmp_path / "caption.json", spans)
    assert sample.multiview is not None
    chunks = entry.load_caption_chunks(sample.multiview.views, first_only=False)
    with pytest.raises(ValueError, match="contiguous"):
        entry.prepare_rolling_prompt_schedule(
            sample, _host(), chunks, per_view_captions=True, system_prompt=_SYSTEM_PROMPT_AV_MULTIVIEW_TRANSFER
        )


@pytest.mark.parametrize("joint", [False, True])
@pytest.mark.parametrize("span", [(100, 140), (3, 36)], ids=["later_window", "offset_window"])
def test_rolling_source_window_rejects_nonzero_start_before_loading_media(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, joint: bool, span: tuple[int, int]
) -> None:
    sample = _sample(tmp_path, **(_joint(tmp_path) if joint else {}))
    assert sample.is_joint_cam_lidar_transfer is joint
    _caption_file(tmp_path / "caption.json", [span])
    loaders = _refuse_media(monkeypatch, "Media decoding preceded the caption window check.")
    with pytest.raises(ValueError, match="source window starting at frame zero"):
        entry.generate_rolling_sample(_host(), sample, per_view_captions=True)
    assert not any(loader.called for loader in loaders)


@pytest.mark.parametrize("joint", [False, True])
@pytest.mark.parametrize(
    "spans", [[(0, 7), (7, 20)], [(0, 10), (50, 60)]], ids=["short_tail", "later_span_outside_horizon"]
)
def test_undercovered_segmented_captions_are_rejected_before_loading_media(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, joint: bool, spans: list[tuple[int, int]]
) -> None:
    sample = _sample(tmp_path, **(_joint(tmp_path) if joint else {}))
    assert sample.multiview is not None
    sample.multiview.use_first_chunk_only = False
    _caption_file(tmp_path / "caption.json", spans)
    loaders = _refuse_media(monkeypatch, "Media decoding preceded caption coverage validation.")
    with pytest.raises(ValueError, match="caption spans must cover the requested.*horizon"):
        entry.generate_rolling_sample(_host(), sample, per_view_captions=True)
    assert not any(loader.called for loader in loaders)


@pytest.mark.parametrize("joint", [False, True])
def test_shared_inference_runs_one_continuous_sampler_without_media_wrap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, joint: bool
) -> None:
    sample = _sample(tmp_path, **(_joint(tmp_path) if joint else {})).model_copy(update={"seed": 7})
    assert sample.multiview is not None
    sample.multiview.use_first_chunk_only = False
    _caption_file(tmp_path / "caption.json", [(0, 7), (7, 20), (20, 40)])
    controls = [
        torch.arange(33).reshape(1, 33, 1, 1).expand(3, -1, -1, -1) + view * 100 for view in range(2)
    ]  # [3,33,1,1]
    loader = MagicMock(return_value=controls)
    monkeypatch.setattr(entry, "_get_multiview_control_pixel_frames", loader)
    monkeypatch.setattr(entry, "_get_multiview_condition_pixel_frames", lambda *args, **kwargs: [None, None])
    monkeypatch.setattr(
        entry,
        "get_multiview_sample_data",
        lambda *args, **kwargs: {"system_prompt": _SYSTEM_PROMPT_AV_MULTIVIEW_TRANSFER},
    )
    monkeypatch.setattr(
        entry, "prepare_joint_batch", lambda *args: {"system_prompt": _SYSTEM_PROMPT_AV_JOINT_CAMERA_LIDAR_TRANSFER}
    )
    monkeypatch.setattr(entry, "_finalize_data_batch", lambda **kwargs: kwargs["data_batch"])
    generated = torch.cat(controls, dim=1)[None].float()  # [1,3,66,1,1]
    sampler = MagicMock(return_value={"vision": generated})
    monkeypatch.setattr(entry, "generate_joint_chunk" if joint else "generate_rgb_chunk", sampler)
    host = _host()
    tokenizer = MagicMock(side_effect=_tokenize)
    monkeypatch.setattr(host, "_tokenize_captions", tokenizer)
    result = entry.generate_rolling_sample(host, sample, per_view_captions=True)
    assert sampler.call_count == loader.call_count == 1
    assert loader.call_args.args[0].num_frames == 33
    assert sampler.call_args.args[1].num_frames == 33 and sampler.call_args.args[1].seed == 7
    assert sampler.call_args.args[2]["rolling_prompt_schedule"].num_frames == 33
    system_prompt = _SYSTEM_PROMPT_AV_JOINT_CAMERA_LIDAR_TRANSFER if joint else _SYSTEM_PROMPT_AV_MULTIVIEW_TRANSFER
    assert tokenizer.call_count == 3
    assert all(call.kwargs["system_prompt"] == system_prompt for call in tokenizer.call_args_list)
    schedule = sampler.call_args.args[2]["rolling_prompt_schedule"]
    for view, segments in enumerate(schedule.views):
        for segment, caption in zip(segments, result.chunks, strict=True):
            assert segment.token_ids == tuple((system_prompt + caption["prompt"][view]).encode())
    assert len(result.chunks) == 3 and {chunk["seed"] for chunk in result.chunks} == {7}
    for found, wanted in zip(result.vision, controls, strict=True):
        torch.testing.assert_close(found, wanted.float(), rtol=0, atol=0)
    # The controls kept for saving are the decoded source window itself, on the source clock.
    for found, wanted in zip(result.controls, controls, strict=True):
        assert found.data_ptr() == wanted.data_ptr()
        torch.testing.assert_close(found, wanted, rtol=0, atol=0)
    assert result.rolling_prompt["control_repeat"] is False
    assert result.rolling_prompt["prompt_repeat"] is False
    assert result.rolling_prompt["policy"] == "segmented"
    assert result.latents == [] and result.lidar is None


@pytest.mark.parametrize("joint", [False, True])
def test_media_preparation_is_timed_apart_from_generation_and_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, joint: bool
) -> None:
    sample = _sample(tmp_path, **(_joint(tmp_path) if joint else {}))
    events: list[str] = []

    @contextmanager
    def timer(name: str):
        events.append(f"enter {name}")
        yield
        events.append(f"exit {name}")

    controls = [torch.zeros(3, 33, 1, 1, dtype=torch.uint8) for _ in range(2)]  # list[[3,F,H,W]]

    def load_controls(*args: Any, **kwargs: Any) -> list[torch.Tensor]:
        events.append("decode controls")
        return controls

    def sample_and_decode(*args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        events.append("sample and decode")
        return {"vision": torch.zeros(1, 3, 66, 1, 1)}  # [1,3,V*F,H,W]

    monkeypatch.setattr(entry, "_get_multiview_control_pixel_frames", load_controls)
    monkeypatch.setattr(entry, "_get_multiview_condition_pixel_frames", lambda *args, **kwargs: [None, None])
    monkeypatch.setattr(
        entry,
        "get_multiview_sample_data",
        lambda *args, **kwargs: {"system_prompt": _SYSTEM_PROMPT_AV_MULTIVIEW_TRANSFER},
    )
    monkeypatch.setattr(
        entry, "prepare_joint_batch", lambda *args: {"system_prompt": _SYSTEM_PROMPT_AV_JOINT_CAMERA_LIDAR_TRANSFER}
    )
    monkeypatch.setattr(entry, "_finalize_data_batch", lambda **kwargs: kwargs["data_batch"])
    monkeypatch.setattr(entry, "generate_joint_chunk" if joint else "generate_rgb_chunk", sample_and_decode)
    entry.generate_rolling_sample(_host(), sample, per_view_captions=True, timer=timer)
    assert events == [
        "enter inference.prepare_inputs",
        "decode controls",
        "exit inference.prepare_inputs",
        "enter inference.generate_decode",
        "sample and decode",
        "exit inference.generate_decode",
    ]


def _stub_rolling_inputs(monkeypatch: pytest.MonkeyPatch, *, joint: bool, events: list[str]) -> dict[str, list[Any]]:
    """Stub the camera media loads, the batch builders and the sampler around the rolling preparation.

    Returns the ``partitioner`` each media load received, by load.
    """
    partitioners: dict[str, list[Any]] = {"conditions": [], "controls": []}

    def load_conditions(*args: Any, partitioner: Any = None, **kwargs: Any) -> list[None]:
        events.append("load conditions")
        partitioners["conditions"].append(partitioner)
        return [None for _ in _CAMERAS]

    def load_controls(*args: Any, partitioner: Any = None, **kwargs: Any) -> list[torch.Tensor]:
        events.append("load controls")
        partitioners["controls"].append(partitioner)
        return [torch.zeros(3, 33, 1, 1, dtype=torch.uint8) for _ in _CAMERAS]  # list[[3,F,H,W]]

    monkeypatch.setattr(entry, "_get_multiview_condition_pixel_frames", load_conditions)
    monkeypatch.setattr(entry, "_get_multiview_control_pixel_frames", load_controls)
    monkeypatch.setattr(
        entry,
        "get_multiview_sample_data",
        lambda *args, **kwargs: {"system_prompt": _SYSTEM_PROMPT_AV_MULTIVIEW_TRANSFER},
    )
    monkeypatch.setattr(
        entry, "prepare_joint_batch", lambda *args: {"system_prompt": _SYSTEM_PROMPT_AV_JOINT_CAMERA_LIDAR_TRANSFER}
    )
    monkeypatch.setattr(entry, "_finalize_data_batch", lambda **kwargs: kwargs["data_batch"])
    monkeypatch.setattr(
        entry,
        "generate_joint_chunk" if joint else "generate_rgb_chunk",
        lambda *args, **kwargs: {"vision": torch.zeros(1, 3, 66, 1, 1)},  # [1,3,V*F,H,W]
    )
    return partitioners


@pytest.mark.parametrize("joint", [False, True])
@pytest.mark.parametrize("in_replica", [False, True])
def test_camera_media_loads_receive_the_replica_partitioner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, joint: bool, in_replica: bool
) -> None:
    """Inside a CP/CFG replica both camera media loads split their views over the replica's ranks."""
    sample = _sample(tmp_path, **(_joint(tmp_path) if joint else {}))
    partitioners = _stub_rolling_inputs(monkeypatch, joint=joint, events=[])
    host = _host()
    replica_partitioner = object()
    if in_replica:
        monkeypatch.setattr(host, "replica_partitioner", replica_partitioner, raising=False)
    entry.generate_rolling_sample(host, sample, per_view_captions=True)
    expected = replica_partitioner if in_replica else None
    assert len(partitioners["conditions"]) == len(partitioners["controls"]) == 1
    assert partitioners["conditions"][0] is expected
    assert partitioners["controls"][0] is expected


@pytest.mark.parametrize("joint", [False, True])
def test_caption_errors_reach_the_replica_before_the_partitioned_media_loads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, joint: bool
) -> None:
    """Preparation raises errors on the replica in two steps, and the media loads are in the second.

    The loads run the replica partitioner's own collectives. Under one shared error barrier, a rank
    that failed while reading its captions would wait in that barrier while its peers waited in the
    partitioner, and the job would hang.
    """
    sample = _sample(tmp_path, **(_joint(tmp_path) if joint else {}))
    events: list[str] = []

    @contextmanager
    def timer(name: str) -> Iterator[None]:
        events.append(f"enter {name}")
        try:
            yield
        finally:
            events.append(f"exit {name}")

    @contextmanager
    def sync_errors(process_groups: Sequence[Any]) -> Iterator[None]:
        assert process_groups == ()
        events.append("enter error barrier")
        try:
            yield
        except Exception as error:
            events.append(f"exit error barrier with {type(error).__name__}")
            raise
        events.append("exit error barrier")

    monkeypatch.setattr(entry, "_sync_inference_replica_errors", sync_errors)
    _stub_rolling_inputs(monkeypatch, joint=joint, events=events)
    entry.generate_rolling_sample(_host(), sample, per_view_captions=True, timer=timer)
    assert events == [
        "enter inference.prepare_inputs",
        "enter error barrier",
        "exit error barrier",
        "enter error barrier",
        "load conditions",
        "load controls",
        "exit error barrier",
        "exit inference.prepare_inputs",
        "enter inference.generate_decode",
        "exit inference.generate_decode",
    ]

    # A caption window error is raised through the first barrier; no rank reaches a media load.
    events.clear()
    _caption_file(tmp_path / "caption.json", [(3, 36)])
    with pytest.raises(ValueError, match="source window starting at frame zero"):
        entry.generate_rolling_sample(_host(), sample, per_view_captions=True, timer=timer)
    assert events == [
        "enter inference.prepare_inputs",
        "enter error barrier",
        "exit error barrier with ValueError",
        "exit inference.prepare_inputs",
    ]


def test_rgb_route_pins_the_reference_system_prompt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(OmniSampleArgs, "vision_size", property(lambda self: (2, 2)))
    sample = _sample(tmp_path)
    controls = [torch.zeros(3, 33, 2, 2, dtype=torch.uint8) for _ in range(2)]  # list[[3,F,H,W]]
    monkeypatch.setattr(entry, "_get_multiview_control_pixel_frames", lambda *args, **kwargs: controls)
    monkeypatch.setattr(entry, "_get_multiview_condition_pixel_frames", lambda *args, **kwargs: [None, None])
    host = _host()
    monkeypatch.setattr(host, "input_video_key", "video", raising=False)
    monkeypatch.setattr(host, "input_image_key", "images", raising=False)
    # Once the export's caption task is visible on the config, the one-shot builder follows it.
    host.config.per_view_caption_task = "av_joint_camera_lidar_transfer"
    built = entry.get_multiview_sample_data(
        sample, host, device="cpu", condition_frames=[None, None], control_frames=controls, view_prompts=["a", "b"]
    )
    assert built["system_prompt"] == _SYSTEM_PROMPT_AV_JOINT_CAMERA_LIDAR_TRANSFER
    captured: dict[str, Any] = {}

    def capture(model: Any, chunk_sample: OmniSampleArgs, data: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        captured.update(data)
        return {"vision": torch.zeros(1, 3, 66, 2, 2)}  # [1,3,V*F,H,W]

    monkeypatch.setattr(entry, "generate_rgb_chunk", capture)
    entry.generate_rolling_sample(host, sample, per_view_captions=True)
    # The rolling RGB route keeps sending what the reference sends, whatever the builder chose.
    assert captured["system_prompt"] == _SYSTEM_PROMPT_AV_MULTIVIEW_TRANSFER
    schedule = captured["rolling_prompt_schedule"]
    assert all(
        bytes(segment.token_ids).decode().startswith(_SYSTEM_PROMPT_AV_MULTIVIEW_TRANSFER)
        for segments in schedule.views
        for segment in segments
    )


def test_rgb_per_view_negative_prompt_carries_no_duration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The per-camera negative is formatted as the reference formats it for an autoregressive sample."""
    monkeypatch.setattr(OmniSampleArgs, "vision_size", property(lambda self: (2, 2)))
    sample = _sample(tmp_path, guidance=3.0, per_view_negative_prompt="A blurry car.")
    controls = [torch.zeros(3, 33, 2, 2, dtype=torch.uint8) for _ in range(2)]  # list[[3,F,H,W]]
    monkeypatch.setattr(entry, "_get_multiview_control_pixel_frames", lambda *args, **kwargs: controls)
    monkeypatch.setattr(entry, "_get_multiview_condition_pixel_frames", lambda *args, **kwargs: [None, None])
    host = _host()
    monkeypatch.setattr(host, "input_video_key", "video", raising=False)
    monkeypatch.setattr(host, "input_image_key", "images", raising=False)
    captured: dict[str, Any] = {}

    def capture(model: Any, chunk_sample: OmniSampleArgs, data: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        captured.update(data)
        return {"vision": torch.zeros(1, 3, 66, 2, 2)}  # [1,3,V*F,H,W]

    monkeypatch.setattr(entry, "generate_rgb_chunk", capture)
    entry.generate_rolling_sample(host, sample, per_view_captions=True)

    def negative(duration_template: str | None) -> str:
        return _format_sample_prompt(
            "A blurry car.", sample, h=2, w=2, duration_template=duration_template, truncate_duration=True
        )

    assert sample.duration_template is not None and negative(sample.duration_template) != negative(None)
    assert captured["neg_" + host.input_caption_key] == [negative(None)]


def test_unseeded_sample_warns_that_its_seed_follows_the_output_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(OmniSampleArgs, "vision_size", property(lambda self: (2, 2)))
    controls = [torch.zeros(3, 33, 2, 2, dtype=torch.uint8) for _ in range(2)]  # list[[3,F,H,W]]
    monkeypatch.setattr(entry, "_get_multiview_control_pixel_frames", lambda *args, **kwargs: controls)
    monkeypatch.setattr(entry, "_get_multiview_condition_pixel_frames", lambda *args, **kwargs: [None, None])
    monkeypatch.setattr(entry, "generate_rgb_chunk", lambda *args, **kwargs: {"vision": torch.zeros(1, 3, 66, 2, 2)})
    host = _host()
    monkeypatch.setattr(host, "input_video_key", "video", raising=False)
    monkeypatch.setattr(host, "input_image_key", "images", raising=False)
    warnings: list[str] = []
    monkeypatch.setattr(entry.log, "warning", lambda message, *args, **kwargs: warnings.append(message))

    sample = _sample(tmp_path)
    assert sample.seed is None
    result = entry.generate_rolling_sample(host, sample, per_view_captions=True)
    # The seed the rollout used, recorded with its captions, is the one the warning names.
    [seed] = {chunk["seed"] for chunk in result.chunks}
    assert seed == entry._fallback_seed(sample)
    assert [message for message in warnings if "sets no seed" in message] == [
        f"Sample '{sample.name}' sets no seed; using seed={seed}, derived from its name and output directory. "
        "Set seed to reproduce this sample under another output directory or runtime."
    ]
    # Moving the outputs changes the noise, which is what the warning is about.
    moved = sample.model_copy(update={"output_dir": tmp_path / "elsewhere"})
    assert entry._fallback_seed(moved) != seed

    warnings.clear()
    entry.generate_rolling_sample(host, sample.model_copy(update={"seed": 7}), per_view_captions=True)
    assert not [message for message in warnings if "sets no seed" in message]


@pytest.mark.parametrize("joint", [False, True])
@pytest.mark.parametrize("per_view_captions", [False, True])
@pytest.mark.parametrize("metadata_mode", [NegativeMetadataMode.SAME, NegativeMetadataMode.INVERSE])
def test_rolling_shared_negative_prompt_reaches_cfg_tokens(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    joint: bool,
    per_view_captions: bool,
    metadata_mode: NegativeMetadataMode,
) -> None:
    # Exercise the real RGB/joint batch builders and sampler dispatch with tiny
    # media; stop after native CFG tokenization, before model denoising.
    monkeypatch.setattr(OmniSampleArgs, "vision_size", property(lambda self: (2, 2)))
    sample = _sample(
        tmp_path,
        **(_joint(tmp_path) if joint else {}),
        guidance=3.0,
        negative_prompt="A blurry car with distorted wheels.",
        negative_metadata_mode=metadata_mode,
    )
    controls = [torch.zeros(3, 33, 2, 2, dtype=torch.uint8) for _ in range(2)]  # list[[3,F,H,W]]
    lidar_control = torch.zeros(3, 11, 2, 2)  # [Cl,F_lidar,Hl,Wl]
    monkeypatch.setattr(entry, "_get_multiview_control_pixel_frames", lambda *args, **kwargs: controls)
    monkeypatch.setattr(entry, "_get_multiview_condition_pixel_frames", lambda *args, **kwargs: [None, None])
    monkeypatch.setattr(entry, "load_lidar_control", lambda *args, **kwargs: (None, lidar_control, None))
    host = _host()
    host.config.fixed_step_sampler_config = None
    monkeypatch.setattr(host, "input_video_key", "video", raising=False)
    monkeypatch.setattr(host, "input_image_key", "images", raising=False)
    monkeypatch.setattr(host, "tokenizer_vision_gen", SimpleNamespace(use_cached_decoder=nullcontext), raising=False)
    observed: dict[str, Any] = {}

    def capture_cfg_tokens(data: dict[str, Any], **settings: Any) -> None:
        observed.update(settings)
        observed["system_prompt"] = data["system_prompt"]
        _, observed["negative_tokens"] = OmniMoTModel._get_inference_text_tokens(
            host,
            data,
            has_negative_prompt=settings["has_negative_prompt"],
            caption_groups=data[host.input_caption_key],
        )
        raise RuntimeError("CFG tokenization reached")

    monkeypatch.setattr(
        host,
        "generate_joint_samples_from_batch_autoregressive" if joint else "iter_samples_from_batch_autoregressive",
        capture_cfg_tokens,
        raising=False,
    )
    with pytest.raises(RuntimeError, match="CFG tokenization reached"):
        entry.generate_rolling_sample(host, sample, per_view_captions=per_view_captions)

    # The reference formats an autoregressive sample's shared negative without its duration sentence;
    # this repo's shared helper, given the sample as is, adds one.
    same = metadata_mode == NegativeMetadataMode.SAME
    duration_template = sample.duration_template if same else sample.inverse_duration_template
    assert duration_template and sample.negative_prompt
    duration_text = duration_template.format(duration=sample.num_frames / sample.fps, fps=sample.fps)
    native_prompt = entry._get_prompt_sample_data(sample, host, h=2, w=2, device="cpu")
    assert duration_text in native_prompt["neg_" + host.input_caption_key][0]
    reference_negative = (
        _format_prompt_with_template(
            sample.negative_prompt,
            fps=sample.fps,
            num_frames=sample.num_frames,
            duration_template=None,
            resolution_template=sample.resolution_template if same else sample.inverse_resolution_template,
            h=2,
            w=2,
            force_duration_template=not same,
        )
        .lstrip(".")
        .strip()
    )
    expected_negative = "" if per_view_captions else reference_negative
    system_prompt = _SYSTEM_PROMPT_AV_JOINT_CAMERA_LIDAR_TRANSFER if joint else _SYSTEM_PROMPT_AV_MULTIVIEW_TRANSFER
    assert observed["negative_tokens"] == _tokenize(
        [expected_negative] * 2, use_system_prompt=False, system_prompt=system_prompt, is_video=False
    )
    assert observed["has_negative_prompt"] is not per_view_captions
    assert observed["guidance"] == sample.guidance == 3.0
    assert observed["sampler_mode"] == "rf"
    assert observed["system_prompt"] == system_prompt


@pytest.mark.parametrize("joint", [False, True])
def test_schedule_and_batch_carry_no_control_emphasis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, joint: bool
) -> None:
    monkeypatch.setattr(OmniSampleArgs, "vision_size", property(lambda self: (2, 2)))
    sample = _sample(tmp_path, **(_joint(tmp_path) if joint else {}))
    assert sample.multiview is not None
    sample.multiview.use_first_chunk_only = False
    _caption_file(tmp_path / "caption.json", [(0, 7), (7, 20), (20, 40)])
    controls = [torch.zeros(3, 33, 2, 2, dtype=torch.uint8) for _ in range(2)]  # list[[3,F,H,W]]
    lidar_control = torch.zeros(3, 11, 2, 2)  # [Cl,F_lidar,Hl,Wl]
    monkeypatch.setattr(entry, "_get_multiview_control_pixel_frames", lambda *args, **kwargs: controls)
    monkeypatch.setattr(entry, "_get_multiview_condition_pixel_frames", lambda *args, **kwargs: [None, None])
    monkeypatch.setattr(entry, "load_lidar_control", lambda *args, **kwargs: (None, lidar_control, None))
    host = _host()
    monkeypatch.setattr(host, "input_video_key", "video", raising=False)
    monkeypatch.setattr(host, "input_image_key", "images", raising=False)

    # The real builders do append the control directive when they format captions themselves, so
    # its absence below is the rolling path replacing their captions, not a builder that never adds it.
    assert sample.emphasize_control_in_prompt
    if joint:
        built = entry.prepare_joint_batch(host, sample, controls, [None, None], None)
    else:
        built = entry.get_multiview_sample_data(
            sample, host, device="cpu", condition_frames=[None, None], control_frames=controls, view_prompts=["a", "b"]
        )
    assert _EMPHASIS in json.dumps(built[host.input_caption_key])

    captured: dict[str, Any] = {}

    def capture(model: Any, chunk_sample: OmniSampleArgs, data: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        captured.update(data)
        return {"vision": torch.zeros(1, 3, 66, 2, 2)}  # [1,3,V*F,H,W]

    monkeypatch.setattr(entry, "generate_joint_chunk" if joint else "generate_rgb_chunk", capture)
    result = entry.generate_rolling_sample(host, sample, per_view_captions=True)
    assert all(_EMPHASIS not in caption for caption in captured[host.input_caption_key][0])
    schedule = captured["rolling_prompt_schedule"]
    documents = [bytes(segment.token_ids).decode() for segments in schedule.views for segment in segments]
    assert len(documents) == 6
    assert all(document.startswith(captured["system_prompt"]) for document in documents)
    assert all(_EMPHASIS not in document for document in documents)
    assert all(_EMPHASIS not in prompt for chunk in result.chunks for prompt in chunk["prompt"])


@pytest.mark.parametrize("structured", [False, True])
@pytest.mark.parametrize(("num_frames", "seconds"), [(33, 1), (201, 6), (60, 2)])
def test_per_view_metadata_matches_training_without_control_emphasis(
    tmp_path: Path, structured: bool, num_frames: int, seconds: int
) -> None:
    sample = _sample(tmp_path).model_copy(
        update={
            "num_frames": num_frames,
            "duration_template": "The video is {duration:.1f} seconds long and is of {fps:.0f} FPS.",
            "resolution_template": "This video is {height}x{width}.",
        }
    )
    model = cast(OmniMoTCausalModel, SimpleNamespace(input_caption_key="ai_caption"))
    raw = json.dumps({"caption": "A car.", "duration": "old", "fps": 1}) if structured else "A car."
    result = entry.per_view_prompt_data(sample, model, [raw, raw])["ai_caption"]
    assert len(result) == 1 and len(result[0]) == 2
    width, height = sample.vision_size
    training: dict[str, Any] = {
        "ai_caption": json.loads(raw) if structured else raw,
        "conditioning_fps": float(sample.fps),
        "num_video_frames_per_view": num_frames,
        "video": None,
        "image_size": (height, width),
        "aspect_ratio": sample.aspect_ratio,
    }
    DurationFPSTextTimeStamps(args={"num_frames_key": "num_video_frames_per_view"})(training)
    ResolutionTextInfo(args={"video_template": sample.resolution_template})(training)
    if structured:
        assert json.loads(result[0][0]) == {
            "caption": "A car.",
            "duration": f"{seconds}s",
            "fps": 30.0,
            "resolution": {"H": height, "W": width},
            "aspect_ratio": sample.aspect_ratio,
        }
        assert json.loads(result[0][0]) == training["ai_caption"]
    else:
        assert result[0][0] == (
            f"A car. The video is {seconds:.1f} seconds long and is of 30 FPS. This video is {height}x{width}."
        )
        assert result[0][0] == training["ai_caption"]
    # The one-shot per-view builder formats the same captions identically.
    public = _get_per_view_prompt_sample_data(
        sample, model, [raw, raw], h=height, w=width, duration_template=sample.duration_template
    )["ai_caption"]
    assert public == result
    assert result[0][0] == result[0][1]
    assert _EMPHASIS not in result[0][0]


def test_disabled_caption_metadata_preserves_structured_body(tmp_path: Path) -> None:
    sample = _sample(tmp_path).model_copy(update={"duration_template": None, "resolution_template": None})
    model = cast(OmniMoTCausalModel, SimpleNamespace(input_caption_key="ai_caption"))
    result = entry.per_view_prompt_data(
        sample, model, ['{"caption":"A car.","duration":"99s","fps":1,"resolution":{"H":2,"W":3}}']
    )
    assert json.loads(result["ai_caption"][0][0]) == {"caption": "A car.", "resolution": {"H": 2, "W": 3}}


def test_ar_chunk_decode_round_trip_restores_camera_major_layout() -> None:
    num_views, channels = 3, 2
    chunk_lengths = [1, 3, 2]  # first frame, a full chunk, a short tail
    streams = torch.randn(num_views, channels, sum(chunk_lengths), 2, 2)  # [V,C,T,H,W], one stream per camera
    starts = [sum(chunk_lengths[:index]) for index in range(len(chunk_lengths))]
    prepared = []
    for start, length in zip(starts, chunk_lengths, strict=True):
        # The sampler yields each chunk camera-major: every camera's frames for this chunk in turn.
        chunk = torch.cat([streams[view, :, start : start + length] for view in range(num_views)], dim=1)[None]
        view_batch = entry._prepare_ar_vision_chunk_for_decode(chunk, sample_n_views=num_views)  # [V,C,T_chunk,H,W]
        torch.testing.assert_close(view_batch, streams[:, :, start : start + length], rtol=0, atol=0)
        prepared.append(view_batch)
    assembled = entry._assemble_decoded_ar_vision_chunks(prepared, sample_n_views=num_views)  # [1,C,V*T,H,W]
    expected = torch.cat([streams[view] for view in range(num_views)], dim=1)[None]  # [1,C,V*T,H,W]
    torch.testing.assert_close(assembled, expected, rtol=0, atol=0)

    with pytest.raises(ValueError, match="not divisible"):
        entry._prepare_ar_vision_chunk_for_decode(torch.zeros(1, channels, 4, 1, 1), sample_n_views=num_views)
    with pytest.raises(ValueError, match="batch_size=1"):
        entry._prepare_ar_vision_chunk_for_decode(torch.zeros(2, channels, 3, 1, 1), sample_n_views=num_views)
    with pytest.raises(ValueError, match="rank 5"):
        entry._prepare_ar_vision_chunk_for_decode(torch.zeros(channels, 3, 1, 1), sample_n_views=num_views)
    with pytest.raises(RuntimeError, match="no decoded"):
        entry._assemble_decoded_ar_vision_chunks([], sample_n_views=num_views)
    with pytest.raises(ValueError, match="one decoded stream per view"):
        entry._assemble_decoded_ar_vision_chunks([torch.zeros(2, channels, 1, 1, 1)], sample_n_views=num_views)


@pytest.mark.parametrize("student", [False, True])
def test_rgb_stream_preserves_sampler_and_decodes_each_camera(tmp_path: Path, student: bool) -> None:
    sample = _sample(tmp_path).model_copy(update={"guidance": 7.0, "seed": 19})
    # Two cameras, each with two logical chunks. Values identify camera ownership.
    chunks = [
        torch.tensor([0.1, 0.2]).reshape(1, 1, 2, 1, 1),
        torch.tensor([0.3, 0.4]).reshape(1, 1, 2, 1, 1),
    ]  # list[[1,1,V*T,1,1]]
    cached_decode = MagicMock(side_effect=nullcontext)
    model = MagicMock(spec=OmniMoTCausalModel)
    model.config = SimpleNamespace(fixed_step_sampler_config=object() if student else None)
    model.input_caption_key = "ai_caption"
    model.tokenizer_vision_gen = SimpleNamespace(use_cached_decoder=cached_decode, get_output_range=lambda: "[0,1]")
    model.decode.side_effect = lambda latent: latent.expand(-1, 3, -1, -1, -1)  # [V,3,T,1,1]
    model.iter_samples_from_batch_autoregressive.return_value = iter({"vision": chunk} for chunk in chunks)
    result = entry.generate_rgb_chunk(model, sample, {"ai_caption": [["front", "left"]]}, save_latents=True)
    expected = torch.tensor([0.1, 0.3, 0.2, 0.4]).reshape(1, 1, 4, 1, 1)  # [1,1,V*T,1,1]
    torch.testing.assert_close(result["vision"], expected.expand(-1, 3, -1, -1, -1))
    torch.testing.assert_close(result["vision_latent"], expected)
    assert model.decode.call_count == 2
    assert all(call.args[0].shape == (2, 1, 1, 1, 1) for call in model.decode.call_args_list)
    assert cached_decode.call_count == 1
    options = model.iter_samples_from_batch_autoregressive.call_args.kwargs
    assert options["mode"] == "video_transfer"
    assert options["guidance"] == (1.0 if student else 7.0)
    assert options["sampler_mode"] == ("distilled" if student else "rf")
    assert options["seed"] == 19


@pytest.mark.parametrize("student", [False, True])
def test_joint_generation_keeps_both_modalities_and_per_camera_decode(tmp_path: Path, student: bool) -> None:
    sample = _sample(tmp_path).model_copy(update={"num_frames": 2})
    model = MagicMock(spec=OmniMoTCausalModel)
    model.config = SimpleNamespace(fixed_step_sampler_config=object() if student else None)
    model.input_caption_key = "ai_caption"
    model.tokenizer_vision_gen = SimpleNamespace(get_output_range=lambda: "[0,1]")
    model.tensor_kwargs_fp32 = {"device": "cpu", "dtype": torch.float32}
    vision = torch.tensor([0.1, 0.2, 0.3, 0.4]).reshape(1, 1, 4, 1, 1)  # [1,1,V*T,1,1]
    lidar = torch.arange(6.0).reshape(1, 2, 3, 1, 1)  # [1,2,Tl,1,1]

    def sample_joint(data: dict[str, Any], **kwargs: Any) -> dict[str, list[torch.Tensor]]:
        # Real sampling rewrites each LiDAR clip from [C,T,H,W] to [1,C,T,H,W].
        # Exercise that mutation so an extra batch dimension cannot reach the result.
        OmniMoTCausalModel._collect_lidar_stream(model, data, batch_size=1)
        return {"vision": [vision], "lidar": [lidar]}

    model.generate_joint_samples_from_batch_autoregressive.side_effect = sample_joint
    model.decode.side_effect = lambda latent: latent.expand(-1, 3, -1, -1, -1)  # [1,3,T,1,1]
    model.decode_lidar.side_effect = lambda latent: latent * 2  # [1,2,Tl,1,1]
    result = entry.generate_joint_chunk(
        model, sample, {"ai_caption": [["front", "left"]], "lidar": [lidar[0]]}, save_latents=True
    )
    torch.testing.assert_close(result["vision"], vision.expand(-1, 3, -1, -1, -1))
    torch.testing.assert_close(result["lidar"], lidar * 2)
    torch.testing.assert_close(result["lidar_control"], lidar)
    torch.testing.assert_close(result["vision_latent"], vision)
    torch.testing.assert_close(result["lidar_latent"], lidar)
    assert model.decode.call_count == 2
    assert all(call.args[0].shape == (1, 1, 2, 1, 1) for call in model.decode.call_args_list)
    options = model.generate_joint_samples_from_batch_autoregressive.call_args.kwargs
    assert options["guidance"] == sample.guidance
    assert options["sampler_mode"] == ("distilled" if student else "rf")


class _FakeReplicaPartitioner:
    """Rank ``group_rank`` of a replica: runs its own round-robin units and receives the others' results."""

    def __init__(self, group_rank: int, group_size: int, received: Any) -> None:
        self.group_rank = group_rank
        self.group_size = group_size
        self.received = received  # index -> the result the owning rank broadcasts
        self.runs: list[tuple[int, int | None, str]] = []

    def run(
        self, units: Sequence[WorkUnit], *, collect_on: int | None = None, on_collected: Any = None, label: str = ""
    ) -> list[None]:
        self.runs.append((len(units), collect_on, label))
        for index, owner in enumerate(plan_partition(len(units), self.group_size)):
            on_collected(index, units[index].run() if owner == self.group_rank else self.received(index))
        return [None for _ in units]


@pytest.mark.parametrize("group_rank", [None, 0, 1, 2, 3], ids=["no_replica", "rank0", "rank1", "rank2", "rank3"])
def test_joint_decode_splits_cameras_and_lidar_over_the_replica(group_rank: int | None) -> None:
    """Each rank of a 4-rank replica decodes its round-robin share of 7 cameras + LiDAR, and gets every clip."""
    num_views, frames, group_size = 7, 2, 4
    # [1,1,V*T,1,1]: every latent frame of view v holds v.
    vision = torch.arange(num_views, dtype=torch.float32).repeat_interleave(frames).reshape(1, 1, -1, 1, 1)
    lidar = torch.arange(6.0).reshape(1, 2, 3, 1, 1)  # [1,2,Tl,1,1]
    decoded: list[str] = []

    def decode(latent: torch.Tensor) -> torch.Tensor:  # [1,1,T,1,1] -> [1,3,T,1,1]
        decoded.append(f"camera {int(latent[0, 0, 0, 0, 0])}")
        return latent.expand(-1, 3, -1, -1, -1)

    def decode_lidar(latent: torch.Tensor) -> torch.Tensor:  # [1,2,Tl,1,1]
        decoded.append("lidar")
        return latent * 2

    def received(index: int) -> torch.Tensor:  # What the owning rank decodes.
        return lidar * 2 if index == num_views else torch.full((1, 3, frames, 1, 1), float(index))

    model = SimpleNamespace(
        decode=decode, decode_lidar=decode_lidar, tokenizer_vision_gen=SimpleNamespace(get_output_range=lambda: "[0,1]")
    )
    partitioner = None
    if group_rank is not None:
        partitioner = _FakeReplicaPartitioner(group_rank, group_size, received)
        model.replica_partitioner = partitioner

    video, decoded_lidar = entry._decode_joint_outputs(
        cast(Any, model), vision, lidar, sample_n_views=num_views, output_camera_frames=frames
    )  # [1,3,V*F,1,1], [1,2,Tl,1,1]

    if partitioner is None:
        # One rank decodes everything, in the order it always has: the cameras, then LiDAR.
        assert decoded == [f"camera {view}" for view in range(num_views)] + ["lidar"]
    else:
        # Round-robin over the cameras and then LiDAR, so the LiDAR unit (index 7) is rank 3's.
        assert decoded == [f"camera {view}" for view in range(group_rank, num_views, group_size)] + (
            ["lidar"] if group_rank == 3 else []
        )
        assert partitioner.runs == [(num_views + 1, None, "joint decode")]
    torch.testing.assert_close(video, vision.expand(-1, 3, -1, -1, -1), rtol=0, atol=0)
    assert video.device.type == decoded_lidar.device.type == "cpu"
    torch.testing.assert_close(decoded_lidar, lidar * 2, rtol=0, atol=0)


@pytest.mark.parametrize("num_view_embeddings", [0, 12])
def test_joint_entry_keeps_controls_and_physical_camera_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, num_view_embeddings: int
) -> None:
    # Keep the real tensor loader and joint batch builder; reduce only spatial size.
    monkeypatch.setattr(OmniSampleArgs, "vision_size", property(lambda self: (2, 2)))
    control = torch.arange(132, dtype=torch.float32).reshape(3, 11, 2, 2)  # [Cl,F_lidar,Hl,Wl]
    control_path = tmp_path / "lidar_control.pt"
    torch.save(control, control_path)
    sample = _sample(tmp_path, lidar={"control_path": str(control_path)})
    assert sample.multiview is not None
    # Use a camera subset in a different order from its physical training IDs.
    sample.multiview.views.reverse()
    projection = LidarRangeProjectionConfig(
        native_height=2, native_width=2, semantic_height=2, semantic_width=2, model_width=2
    )
    model = cast(
        OmniMoTCausalModel,
        SimpleNamespace(
            config=_model_config(num_view_embeddings=num_view_embeddings),
            tokenizer_lidar_gen=SimpleNamespace(range_projection=projection, is_causal=True),
            input_caption_key="ai_caption",
            tensor_kwargs={"device": "cpu"},
        ),
    )
    controls = [torch.zeros(3, 33, 2, 2, dtype=torch.uint8) for _ in range(2)]  # list[[3,F,H,W]]
    batch = entry.prepare_joint_batch(model, sample, controls, [None, None], ["left", "front"])
    if num_view_embeddings:
        physical_ids = vision_view_ids(
            batch,
            batch_size=1,
            item_counts=batch["num_vision_items_per_sample"],
            views_per_item=[2, 2],
            num_embeddings=num_view_embeddings,
        )  # list[[V]]
        assert [ids.tolist() for ids in physical_ids] == [[5, 0], [5, 0]]
    else:
        assert "view_indices_selection" not in batch
    torch.testing.assert_close(batch["lidar"][0], control, atol=0, rtol=0)
    torch.testing.assert_close(batch["lidar"][1], torch.zeros_like(control), atol=0, rtol=0)
    assert batch["sequence_plan"][0].condition_frame_indexes_lidar == []
    assert batch["system_prompt"] == _SYSTEM_PROMPT_AV_JOINT_CAMERA_LIDAR_TRANSFER
    assert batch["ai_caption"][0][0].startswith("left")
    assert batch["ai_caption"][0][1].startswith("front")
    assert "neg_ai_caption" not in batch


@pytest.mark.parametrize("prefix_kind,num_view_embeddings", [("none", 0), ("nested", 0), ("nested", 12)])
def test_joint_entry_preserves_lidar_prefix_and_zeros_future_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prefix_kind: str, num_view_embeddings: int
) -> None:
    # Keep the real tensor loader and joint batch builder; reduce only spatial size.
    monkeypatch.setattr(OmniSampleArgs, "vision_size", property(lambda self: (2, 2)))
    control = torch.arange(132, dtype=torch.float32).reshape(3, 11, 2, 2)  # [Cl,F_lidar,Hl,Wl]
    target = torch.full_like(control, 999.0)  # [Cl,F_lidar,Hl,Wl]
    target[:, 0] = torch.tensor([20.0, 0.75, 1.0]).reshape(3, 1, 1)  # [Cl,Hl,Wl]
    control_path, target_path = tmp_path / "lidar_control.pt", tmp_path / "lidar_target.pt"
    torch.save(control, control_path)
    torch.save(target, target_path)
    prefix_frames = int(prefix_kind != "none")
    lidar_options = {"control_path": str(control_path)}
    if prefix_kind == "nested":
        lidar_options["condition_path"] = str(target_path)
    sample = _sample(tmp_path, lidar=lidar_options)
    assert sample.multiview is not None
    # Use a camera subset in a different order from its physical training IDs.
    sample.multiview.views.reverse()
    projection = LidarRangeProjectionConfig(
        native_height=2, native_width=2, semantic_height=2, semantic_width=2, model_width=2
    )
    model = cast(
        OmniMoTCausalModel,
        SimpleNamespace(
            config=_model_config(num_view_embeddings=num_view_embeddings),
            tokenizer_lidar_gen=SimpleNamespace(
                range_projection=projection, is_causal=True, temporal_compression_factor=1
            ),
            input_caption_key="ai_caption",
            tensor_kwargs={"device": "cpu"},
        ),
    )
    controls = [torch.zeros(3, 33, 2, 2) for _ in range(2)]  # list[[3,F,H,W]]
    batch = entry.prepare_joint_batch(model, sample, controls, [None, None], ["left", "front"])
    if num_view_embeddings:
        physical_ids = vision_view_ids(
            batch,
            batch_size=1,
            item_counts=batch["num_vision_items_per_sample"],
            views_per_item=[2, 2],
            num_embeddings=num_view_embeddings,
        )  # list[[V]]
        assert [ids.tolist() for ids in physical_ids] == [[5, 0], [5, 0]]
    else:
        assert "view_indices_selection" not in batch
    torch.testing.assert_close(batch["lidar"][0], control, atol=0, rtol=0)
    expected = torch.zeros_like(target)  # [Cl,F_lidar,Hl,Wl]
    expected[:, :prefix_frames] = target[:, :prefix_frames]  # [Cl,P,Hl,Wl]
    torch.testing.assert_close(batch["lidar"][1], expected, atol=0, rtol=0)
    assert batch["sequence_plan"][0].condition_frame_indexes_lidar == list(range(prefix_frames))
    assert batch["ai_caption"][0][0].startswith("left")
    assert batch["ai_caption"][0][1].startswith("front")


def test_causal_measured_archive_prefix_keeps_intensity_and_zero_future(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(OmniSampleArgs, "vision_size", property(lambda self: (2, 2)))
    path = tmp_path / "measured.tar"
    with tarfile.open(path, "w") as archive:
        for frame_id in range(0, 33, 3):
            arrays = {"row": [0], "col": [0], "range": [20.0 + frame_id], "intensity": [0.25]}
            for kind, values in arrays.items():
                payload = io.BytesIO()
                np.savez(payload, np.asarray(values))
                member = tarfile.TarInfo(f"clip.{frame_id:06d}.lidar_{kind}.npz")
                member.size = len(payload.getvalue())
                archive.addfile(member, io.BytesIO(payload.getvalue()))
    sample = _sample(
        tmp_path, lidar={"control_path": str(path), "condition_path": str(path), "num_conditional_sweeps": 1}
    )
    projection = LidarRangeProjectionConfig(semantic_width=1800, model_width=1808)
    model = cast(
        OmniMoTCausalModel,
        SimpleNamespace(
            config=_model_config(),
            tokenizer_lidar_gen=SimpleNamespace(
                range_projection=projection, is_causal=True, temporal_compression_factor=1
            ),
            input_caption_key="ai_caption",
            tensor_kwargs={"device": "cpu"},
        ),
    )
    controls = [torch.zeros(3, 33, 2, 2) for _ in range(2)]  # list[[3,F,H,W]]
    batch = entry.prepare_joint_batch(model, sample, controls, [None, None], ["front", "left"])
    target = batch["lidar"][1]  # [3,F_lidar,Hl,Wl]
    torch.testing.assert_close(target[:, 0, 0, 4], torch.tensor([20.0, 0.25, 1.0]))
    assert torch.count_nonzero(target[:, 1:]) == 0
    assert batch["sequence_plan"][0].condition_frame_indexes_lidar == [0]


def test_joint_entry_rejects_unknown_rig_cameras(tmp_path: Path) -> None:
    sample = _sample(tmp_path, **_joint(tmp_path))
    assert sample.multiview is not None
    sample.multiview.views[0].camera_key = "unknown_camera"
    model = cast(OmniMoTCausalModel, SimpleNamespace(config=_model_config(num_view_embeddings=12)))
    with pytest.raises(ValueError, match="supported camera keys.*unknown_camera"):
        entry.prepare_joint_batch(model, sample, [], [], None)


@pytest.mark.parametrize("bad_ids", [list(range(1, 34, 3)), [0, 3, 6, 9, 12, 16, 18, 21, 24, 27, 30]])
def test_joint_entry_rejects_lidar_clock_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad_ids: list[int]
) -> None:
    sample = _sample(tmp_path, **_joint(tmp_path))
    control = torch.zeros(3, 11, 2, 2)  # [Cl,F_lidar,Hl,Wl]
    ids = torch.tensor(bad_ids)  # [F_lidar]
    monkeypatch.setattr(entry, "load_lidar_control", lambda *args, **kwargs: ("control.pt", control, ids))
    model = cast(OmniMoTCausalModel, SimpleNamespace(config=_model_config()))
    with pytest.raises(ValueError, match="source IDs"):
        entry.prepare_joint_batch(model, sample, [], [], None)


@pytest.mark.parametrize("fps", [30, 24])
def test_joint_entry_places_sweeps_on_the_sample_camera_clock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fps: int
) -> None:
    monkeypatch.setattr(OmniSampleArgs, "vision_size", property(lambda self: (2, 2)))
    # Validation pins joint samples to 30 fps; a programmatic caller can still hand over another clock.
    sample = _sample(tmp_path, **_joint(tmp_path)).model_copy(update={"fps": fps})
    requested: list[int] = []

    def load_mads_archive(model: Any, lidar: Any, *, max_frames: int) -> tuple[str, torch.Tensor, torch.Tensor]:
        # A rangemap archive records every sweep's source frame on the 30 fps camera clock.
        requested.append(max_frames)
        return "clip.tar", torch.zeros(3, max_frames, 2, 2), torch.arange(max_frames) * 3

    monkeypatch.setattr(entry, "load_lidar_control", load_mads_archive)
    model = cast(
        OmniMoTCausalModel,
        SimpleNamespace(config=_model_config(), input_caption_key="ai_caption", tensor_kwargs={"device": "cpu"}),
    )
    controls = [torch.zeros(3, 33, 2, 2, dtype=torch.uint8) for _ in range(2)]  # list[[3,F,H,W]]
    expected_sweeps = round(sample.num_frames * 10 / fps)
    if fps == 30:
        batch = entry.prepare_joint_batch(model, sample, controls, [None, None], ["front", "left"])
        assert batch["lidar"][0].shape[1] == expected_sweeps
    else:
        with pytest.raises(ValueError, match="source IDs must align"):
            entry.prepare_joint_batch(model, sample, controls, [None, None], ["front", "left"])
    assert requested == [expected_sweeps]


def test_per_view_checkpoint_without_caption_files_is_refused_before_media_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sample = _sample(tmp_path)
    assert sample.multiview is not None
    sample.multiview.load_caption_from_data = False
    loaders = _refuse_media(monkeypatch, "media load reached")
    with pytest.raises(ValueError, match="load_caption_from_data=True"):
        entry.generate_rolling_sample(_host(), sample, per_view_captions=True)
    assert not any(loader.called for loader in loaders)
    # A merged-caption checkpoint still runs from the sample prompt, as the reference does.
    with pytest.raises(AssertionError, match="media load reached"):
        entry.generate_rolling_sample(_host(), sample, per_view_captions=False)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"rolling_kv_cache_chunks": None}, "--experiment-overrides"),
        ({"rolling_prompt_policy": None}, "--experiment-overrides"),
        ({"kv_cache_inference_size": 8}, "framewise"),
        ({"attention_sink_size": 1}, "framewise"),
    ],
)
def test_checkpoint_without_rolling_profile_is_refused_before_media_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, overrides: dict[str, Any], match: str
) -> None:
    sample = _sample(tmp_path)
    host = _host()
    for key, value in overrides.items():
        setattr(host.config, key, value)
    loaders = _refuse_media(monkeypatch, "media load reached")
    with pytest.raises(ValueError, match=match):
        entry.generate_rolling_sample(host, sample, per_view_captions=True)
    assert not any(loader.called for loader in loaders)


def test_rolling_profile_check_reads_the_exported_config_mapping() -> None:
    # The rolling fields of an exported config.json's model.config, with and without the rolling overrides.
    rolling = {
        "rolling_kv_cache_chunks": 2,
        "rolling_kv_sink_chunks": 1,
        "rolling_prompt_policy": "segmented",
        "kv_cache_dtype": None,
        "kv_cache_inference_size": None,
        "attention_sink_size": 0,
    }
    entry.require_segmented_rolling_config(rolling)
    plain = rolling | {"rolling_kv_cache_chunks": None, "rolling_prompt_policy": None}
    with pytest.raises(ValueError, match="Re-export") as refusal:
        entry.require_segmented_rolling_config(plain)
    for field in ("rolling_kv_cache_chunks", "rolling_kv_sink_chunks", "rolling_prompt_policy"):
        assert f"model.config.{field}=" in str(refusal.value)


@pytest.mark.parametrize(
    ("setup", "refusal"),
    [
        ({"use_separate_pipeline_vision_decode_gpu": True}, "separate VAE decode GPU"),
        # The training-experiment route: nothing on it records the caption layout.
        ({"checkpoint_type": CheckpointType.DCP, "config_file_type": ConfigFileType.MODULE}, "training experiment"),
        ({"checkpoint_type": CheckpointType.DCP, "config_file_type": ConfigFileType.JSON}, None),
        ({}, None),
    ],
)
def test_pipeline_entry_refuses_from_the_shared_setup_and_forwards_pipeline_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, setup: dict[str, Any], refusal: str | None
) -> None:
    sample = _sample(tmp_path)
    rolling = MagicMock(return_value="result")
    monkeypatch.setattr(entry, "generate_rolling_sample", rolling)
    exported_hf = {
        "use_separate_pipeline_vision_decode_gpu": False,
        "checkpoint_type": CheckpointType.HF,
        "config_file_type": ConfigFileType.MODULE,
    }
    # No decode stream: it exists on the decode-owning rank only, and every rank must refuse alike.
    pipe = SimpleNamespace(
        setup_args=SimpleNamespace(**(exported_hf | setup)),
        model=object(),
        vae_decode_stream=None,
        per_view_captions=True,
        _get_timer=MagicMock(),
    )
    if refusal is not None:
        with pytest.raises(ValueError, match=refusal):
            entry.generate_causal_multiview_sample(cast(OmniInference, pipe), sample, save_latents=False)
        rolling.assert_not_called()
        return
    assert entry.generate_causal_multiview_sample(cast(OmniInference, pipe), sample, save_latents=True) == "result"
    rolling.assert_called_once_with(
        pipe.model, sample, per_view_captions=True, save_latents=True, timer=pipe._get_timer
    )


def _pipe(model: Any) -> OmniInference:
    return OmniInference(setup_args=cast(Any, SimpleNamespace()), model=model, guardrails=None, _timer=None)


def _schedulable_cpus(monkeypatch: pytest.MonkeyPatch, cpu_count: int) -> None:
    """Let this rank schedule on ``cpu_count`` CPUs: what the encoder budget and the saver's concurrency read."""
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(cpu_count)), raising=False)


def _result(
    *, num_frames: int, size: int, lidar: torch.Tensor | None, save_latents: bool
) -> entry.CausalMultiviewResult:
    return entry.CausalMultiviewResult(
        vision=[torch.rand(3, num_frames, size, size) for _ in _CAMERAS],  # list[[3,F,H,W]]
        # One frame short of the horizon: the first-caption terminal padding case.
        controls=[torch.randint(0, 255, (3, num_frames - 1, size, size), dtype=torch.uint8) for _ in _CAMERAS],
        lidar=lidar,
        lidar_control=None if lidar is None else torch.zeros_like(lidar),
        chunks=[
            {"frame_range": [0, num_frames - 1], "source_frame_range": [0, 299], "prompt": ["front", "left"], "seed": 7}
        ],
        rolling_prompt={"policy": "segmented", "num_frames": num_frames},
        latents=[{"vision_latent": torch.randn(1, 4, 6, 2, 2)}] if save_latents else [],
        save_latents=save_latents,
    )


@pytest.mark.parametrize("fail", [False, True])
def test_save_publishes_complete_receipt_or_preserves_existing_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail: bool
) -> None:
    monkeypatch.setattr(OmniSampleArgs, "vision_size", property(lambda self: (2, 2)))
    sample = _sample(tmp_path)
    assert sample.output_dir is not None and sample.multiview is not None
    sample.output_dir.mkdir()
    receipt = sample.output_dir / "sample_outputs.json"
    receipt.write_text("previous receipt")
    previous = sample.output_dir / "sample_args.json"
    previous.write_text("previous arguments")
    pipe = _pipe(SimpleNamespace())

    def save(
        staged: OmniSampleArgs,
        view_videos: list[torch.Tensor],
        *,
        prefix: str,
        fps: int,
        encoder_threads: int,
        max_workers: int,
    ) -> Any:
        assert staged.output_dir is not None and staged.multiview is not None
        files = []
        for view in staged.multiview.views:
            path = staged.output_dir / f"{prefix}_{view.camera_key}.mp4"
            path.write_bytes(b"video")
            files.append(path)
        if fail and prefix == "control":
            raise RuntimeError("save failed")
        return files, {}

    monkeypatch.setattr(pipe, "_save_multiview_view_videos", save)
    result = _result(num_frames=sample.num_frames, size=2, lidar=None, save_latents=True)
    if fail:
        with pytest.raises(RuntimeError, match="save failed"):
            entry.save_causal_multiview_outputs(pipe, sample, result)
        assert receipt.read_text() == "previous receipt"
        assert previous.read_text() == "previous arguments"
        assert sorted(path.name for path in sample.output_dir.iterdir()) == ["sample_args.json", "sample_outputs.json"]
    else:
        output = entry.save_causal_multiview_outputs(pipe, sample, result)
        assert output.status == "success"
        assert all(path.exists() for path in output.outputs[0].files)
        saved = json.loads(receipt.read_text())
        assert saved["outputs"][0]["content"]["multiview_cameras"] == sample.multiview.camera_keys
        assert len(saved["outputs"][0]["content"]["vision_files_by_camera"]) == 2
        assert (sample.output_dir / "latents.pt").is_file()
        assert (sample.output_dir / "rolling_prompt.json").is_file()
        assert previous.read_text() == sample.model_dump_json(indent=2)
        assert not [path for path in sample.output_dir.iterdir() if path.name.startswith(".causal-")]


@pytest.mark.parametrize("joint", [False, True])
@pytest.mark.parametrize("save_latents", [False, True])
def test_saved_layout_matches_reference_plus_control_views(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, joint: bool, save_latents: bool
) -> None:
    size, num_frames = 16, 9
    monkeypatch.setattr(OmniSampleArgs, "vision_size", property(lambda self: (size, size)))
    sample = _sample(tmp_path, **(_joint(tmp_path) if joint else {})).model_copy(update={"num_frames": num_frames})
    assert sample.output_dir is not None and sample.multiview is not None
    projection = LidarRangeProjectionConfig(
        native_height=2, native_width=8, semantic_height=2, semantic_width=8, model_width=8
    )
    pipe = _pipe(
        SimpleNamespace(
            config=SimpleNamespace(lidar_fps=10.0), tokenizer_lidar_gen=SimpleNamespace(range_projection=projection)
        )
    )
    sweeps = 3
    lidar = None
    if joint:
        lidar = torch.stack(
            [torch.full((sweeps, 2, 8), 20.0), torch.full((sweeps, 2, 8), 0.5), torch.ones(sweeps, 2, 8)]
        )[None]  # [1,3,F_lidar,Hl,Wl]
    result = _result(num_frames=num_frames, size=size, lidar=lidar, save_latents=save_latents)

    guarded: list[str] = []

    def guardrail(name: str, video: torch.Tensor) -> torch.Tensor:  # [3,F,H,W]
        guarded.append(name)
        return video.flip(-1)  # Visibly different, so the written vision must be the guardrail output.

    written_views: dict[str, list[torch.Tensor]] = {}
    save_views = pipe._save_multiview_view_videos

    def record(
        staged: OmniSampleArgs,
        view_videos: list[torch.Tensor],
        *,
        prefix: str,
        fps: int,
        encoder_threads: int,
        max_workers: int,
    ) -> Any:
        written_views[prefix] = view_videos
        assert fps == sample.fps
        return save_views(
            staged, view_videos, prefix=prefix, fps=fps, encoder_threads=encoder_threads, max_workers=max_workers
        )

    monkeypatch.setattr(pipe, "_run_video_guardrail", guardrail)
    monkeypatch.setattr(pipe, "_save_multiview_view_videos", record)
    output = entry.save_causal_multiview_outputs(pipe, sample, result)

    output_dir = sample.output_dir
    camera_files = {
        f"{prefix}_{safe_multiview_camera_name(camera, index)}.mp4"
        for prefix in ("vision", "control")
        for index, camera in enumerate(_CAMERAS)
    }
    expected = camera_files | {"sample_args.json", "caption_chunks.json", "rolling_prompt.json"}
    if joint:
        expected |= {"lidar.safetensors", "lidar.pt"}
    if save_latents:
        expected.add("latents.pt")
    assert {path.name for path in output_dir.iterdir()} == expected | {"sample_outputs.json"}
    assert {path.name for path in output.outputs[0].files} == expected
    assert all(path.parent == output_dir for path in output.outputs[0].files)

    assert guarded == [f"{output_dir}/{camera}" for camera in _CAMERAS]
    for found, generated in zip(written_views["vision"], result.vision, strict=True):
        torch.testing.assert_close(found, generated.flip(-1), rtol=0, atol=0)
    for found, control in zip(written_views["control"], result.controls, strict=True):
        assert found.dtype == torch.uint8 and found.shape[1] == num_frames
        torch.testing.assert_close(found[:, : control.shape[1]], control, rtol=0, atol=0)
        torch.testing.assert_close(found[:, -1], control[:, -1], rtol=0, atol=0)

    assert json.loads((output_dir / "caption_chunks.json").read_text()) == result.chunks
    assert json.loads((output_dir / "rolling_prompt.json").read_text()) == result.rolling_prompt
    assert (output_dir / "sample_args.json").read_text() == sample.model_dump_json(indent=2)
    content = json.loads((output_dir / "sample_outputs.json").read_text())["outputs"][0]["content"]
    assert content["vision_files_by_camera"] == {
        camera: [str(output_dir / f"vision_{safe_multiview_camera_name(camera, index)}.mp4")]
        for index, camera in enumerate(_CAMERAS)
    }
    assert content["caption_chunks"] == result.chunks
    if save_latents:
        saved_latents = torch.load(output_dir / "latents.pt", weights_only=True)
        assert isinstance(saved_latents, list) and len(saved_latents) == 1
        torch.testing.assert_close(saved_latents[0]["vision_latent"], result.latents[0]["vision_latent"])
    if joint:
        assert lidar is not None
        torch.testing.assert_close(torch.load(output_dir / "lidar.pt", weights_only=True), lidar)
        saved = load_file(output_dir / "lidar.safetensors")
        # 10 Hz sweeps on the 30 fps camera clock.
        assert saved["frame_ids"].dtype == torch.int64 and saved["frame_ids"].tolist() == [0, 3, 6]
        assert saved["range_m"].shape == (sweeps, 2, 8)
        assert content["lidar_file"] == str(output_dir / "lidar.safetensors")
        assert content["lidar_range_projection"] == json.loads(json.dumps(projection.to_dict()))
    else:
        assert "lidar_file" not in content


@pytest.mark.parametrize(
    ("cpu_count", "num_jobs", "budget"),
    # The reference's own cases, and seven cameras on a rank with 72 schedulable CPUs.
    [(8, 8, (4, 2)), (32, 4, (4, 4)), (72, 7, (7, 4))],
)
def test_multiview_save_thread_budget_matches_the_reference(
    cpu_count: int, num_jobs: int, budget: tuple[int, int]
) -> None:
    assert inference._multiview_save_thread_budget(num_jobs, cpu_count=cpu_count) == budget


@pytest.mark.parametrize(("cpu_count", "encoder_threads"), [(72, 4), (8, 4), (4, 2)])
def test_camera_videos_are_byte_identical_to_the_reference_encode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cpu_count: int, encoder_threads: int
) -> None:
    """x264's bitstream depends on its thread count, so the same pixels must be encoded with the reference's count."""
    size, num_frames = 32, 9
    monkeypatch.setattr(OmniSampleArgs, "vision_size", property(lambda self: (size, size)))
    _schedulable_cpus(monkeypatch, cpu_count)
    sample = _sample(tmp_path).model_copy(update={"num_frames": num_frames})
    assert sample.output_dir is not None
    result = _result(num_frames=num_frames, size=size, lidar=None, save_latents=False)
    entry.save_causal_multiview_outputs(_pipe(SimpleNamespace()), sample, result)

    reference_dir = tmp_path / "reference"
    reference_dir.mkdir()
    for index, (camera, view) in enumerate(zip(_CAMERAS, result.vision, strict=True)):
        name = f"vision_{safe_multiview_camera_name(camera, index)}"
        # How the reference writes each camera: the same quality and fps, with an explicit thread count.
        save_img_or_video(
            view,
            str(reference_dir / name),
            fps=sample.fps,
            quality=sample.video_save_quality,
            ffmpeg_params=["-threads", str(encoder_threads)],
        )
        assert (sample.output_dir / f"{name}.mp4").read_bytes() == (reference_dir / f"{name}.mp4").read_bytes()


def _camera_file_stems(prefix: str) -> list[str]:
    return [f"{prefix}_{safe_multiview_camera_name(camera, index)}" for index, camera in enumerate(_CAMERAS)]


def test_vision_and_control_cameras_encode_concurrently(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With CPUs for every encode, the vision and control files of every camera are written at once."""
    monkeypatch.setattr(OmniSampleArgs, "vision_size", property(lambda self: (2, 2)))
    _schedulable_cpus(monkeypatch, 72)
    sample = _sample(tmp_path)
    assert sample.output_dir is not None
    # Every writer waits for all the others, so a saver that encodes in turn breaks the barrier.
    all_in_flight = threading.Barrier(2 * len(_CAMERAS), timeout=5)

    def write(video: torch.Tensor, path_without_extension: str, **kwargs: Any) -> None:  # video: [3,F,H,W]
        all_in_flight.wait()
        Path(f"{path_without_extension}.mp4").write_bytes(b"video")

    monkeypatch.setattr(inference, "save_img_or_video", write)
    result = _result(num_frames=sample.num_frames, size=2, lidar=None, save_latents=False)
    output = entry.save_causal_multiview_outputs(_pipe(SimpleNamespace()), sample, result)
    assert output.status == "success"
    assert sorted(path.name for path in sample.output_dir.glob("*.mp4")) == sorted(
        f"{stem}.mp4" for prefix in ("vision", "control") for stem in _camera_file_stems(prefix)
    )


@pytest.mark.parametrize(
    ("cpu_count", "encoder_threads", "workers_per_set"),
    # 8 CPUs: the budget for two cameras gives 4 threads; a budget for all four encodes would give 2.
    [(4, 2, 1), (8, 4, 1), (72, 4, 2)],
)
def test_concurrent_camera_encodes_keep_the_thread_count_of_one_camera_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cpu_count: int, encoder_threads: int, workers_per_set: int
) -> None:
    """x264's bytes follow its thread count, so encoding vision and control side by side must not change it."""
    monkeypatch.setattr(OmniSampleArgs, "vision_size", property(lambda self: (2, 2)))
    _schedulable_cpus(monkeypatch, cpu_count)
    sample = _sample(tmp_path)
    pipe = _pipe(SimpleNamespace())
    encodes: list[tuple[str, list[str] | None]] = []

    def write(
        video: torch.Tensor, path_without_extension: str, *, ffmpeg_params: list[str] | None, **kwargs: Any
    ) -> None:  # video: [3,F,H,W]
        encodes.append((Path(path_without_extension).name, ffmpeg_params))
        Path(f"{path_without_extension}.mp4").write_bytes(b"video")

    workers: dict[str, int | None] = {}
    save_views = pipe._save_multiview_view_videos

    def record(staged: OmniSampleArgs, view_videos: list[torch.Tensor], **kwargs: Any) -> Any:
        workers[kwargs["prefix"]] = kwargs.get("max_workers")
        return save_views(staged, view_videos, **kwargs)

    monkeypatch.setattr(inference, "save_img_or_video", write)
    monkeypatch.setattr(pipe, "_save_multiview_view_videos", record)
    entry.save_causal_multiview_outputs(
        pipe, sample, _result(num_frames=sample.num_frames, size=2, lidar=None, save_latents=False)
    )
    assert sorted(name for name, _ in encodes) == sorted(_camera_file_stems("vision") + _camera_file_stems("control"))
    assert all(params == ["-threads", str(encoder_threads)] for _, params in encodes)
    assert workers == {"vision": workers_per_set, "control": workers_per_set}
    if cpu_count == 8:
        assert inference._multiview_save_thread_budget(2 * len(_CAMERAS), cpu_count=cpu_count)[1] != encoder_threads


def test_concurrent_view_saves_return_files_in_camera_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sample = _sample(tmp_path)
    assert sample.output_dir is not None
    sample.output_dir.mkdir()
    stems = _camera_file_stems("vision")
    last_camera_written = threading.Event()
    finished: list[str] = []

    def write(video: torch.Tensor, path_without_extension: str, **kwargs: Any) -> None:  # video: [3,F,H,W]
        stem = Path(path_without_extension).name
        if stem == stems[0]:
            # The first camera finishes only after the last one: the reverse of camera order.
            assert last_camera_written.wait(timeout=5), "the cameras were not encoded concurrently"
        Path(f"{path_without_extension}.mp4").write_bytes(b"video")
        finished.append(stem)
        if stem == stems[-1]:
            last_camera_written.set()

    monkeypatch.setattr(inference, "save_img_or_video", write)
    views = [torch.rand(3, 3, 2, 2) for _ in _CAMERAS]  # list[[3,F,H,W]]
    files, files_by_camera = _pipe(SimpleNamespace())._save_multiview_view_videos(
        sample, views, prefix="vision", fps=sample.fps, encoder_threads=4, max_workers=len(_CAMERAS)
    )
    assert finished == stems[::-1]
    assert files == [sample.output_dir / f"{stem}.mp4" for stem in stems]
    assert files_by_camera == {camera: [str(path)] for camera, path in zip(_CAMERAS, files, strict=True)}


def test_sweep_ids_follow_the_camera_clock() -> None:
    assert entry.joint_lidar_sweep_ids(4, camera_fps=30.0, lidar_fps=10.0).tolist() == [0, 3, 6, 9]
    # A non-integral ratio still rounds each sweep onto the camera clock.
    assert entry.joint_lidar_sweep_ids(4, camera_fps=25.0, lidar_fps=10.0).tolist() == [0, 2, 5, 8]
    with pytest.raises(ValueError, match="positive"):
        entry.joint_lidar_sweep_ids(0, camera_fps=30.0, lidar_fps=10.0)
