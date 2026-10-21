# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Keep precomputed controls on their source clock and requested model canvas."""

import json
from collections.abc import Iterator
from contextlib import nullcontext
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import av
import numpy as np
import pytest
import torch
from PIL import Image

from cosmos_framework.inference import transfer, transfer_streaming, vision
from cosmos_framework.inference.args import OmniSampleArgs, TransferHintKey
from cosmos_framework.inference.inference import _format_json_prompt_with_template
from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel


class _Sample(SimpleNamespace):
    def model_copy(self, *, update: dict[str, Any]) -> "_Sample":
        return _Sample(**(vars(self) | update))


def _write_control(path: Path, *, fps: Fraction = Fraction(24), num_frames: int = 25) -> None:
    with av.open(str(path), "w") as container:
        stream = container.add_stream("ffv1", rate=fps)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "bgr0"
        for index in range(num_frames):
            pixels = np.full((48, 64, 3), index * 7 % 256, dtype=np.uint8)  # [H,W,C]
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _run_transfer(
    monkeypatch: pytest.MonkeyPatch,
    source: Path,
    *,
    streaming: bool,
    hint: TransferHintKey = TransferHintKey.DEPTH,
    aspect_ratio: str | None = "16,9",
    num_frames: int = 25,
    has_rgb: bool = False,
) -> tuple[transfer.TransferGenerationOutput, dict[str, Any]]:
    """Use actual decoding, resize, prompt metadata and AR wrappers around a CPU model stub."""
    sample = _Sample(
        transfer_hints={hint: SimpleNamespace(control_path=str(source), weight=1.0)},
        transfer_resize_mode="center_crop",
        resolution="480",
        vision_path=str(source) if has_rgb else None,
        aspect_ratio=aspect_ratio,
        max_frames=num_frames,
        num_video_frames_per_chunk=25,
        num_conditional_frames=1,
        num_first_chunk_conditional_frames=0,
        fps=30,
        autoregressive=True,
        emphasize_control_in_prompt=False,
        seed=42,
        guidance=1.0,
        control_guidance=1.0,
        share_vision_temporal_positions=True,
        num_steps=4,
        shift=5.0,
        normalize_cfg=False,
        show_control_condition=False,
        show_input=False,
    )
    observed: dict[str, Any] = {}

    def format_prompt(args: Any, model: Any, *, h: int, w: int, device: str) -> dict[str, list[str]]:
        # Avoid the text embedding/GPU boundary, retaining the production metadata formatter.
        prompt = _format_json_prompt_with_template(
            {"temporal_caption": "An object moves."},
            fps=args.fps,
            num_frames=args.num_frames,
            aspect_ratio=args.aspect_ratio,
            h=h,
            w=w,
            include_temporal_metadata=True,
        )
        observed["prompt"] = json.loads(prompt)
        observed["hw"] = (h, w)
        return {"caption": [prompt]}

    def build_batch(**kwargs: Any) -> dict[str, Any]:
        # The real batch builder creates CUDA tensors; expose its input clock on CPU.
        observed["batch_fps"] = kwargs["fps"]
        return {"fps": torch.tensor([kwargs["fps"]]), "caption": [kwargs["prompt"]]}

    def iter_ar(**kwargs: Any) -> Iterator[dict[str, torch.Tensor]]:
        observed["model_fps"] = float(kwargs["data_batch"]["fps"][0])
        height, width = observed["hw"]
        for count in [1] + [4] * ((num_frames - 1) // 4):
            decoded = torch.zeros(1, 3, count, height, width)  # [B,C,T,H,W]
            kwargs["on_clean_vision_chunk"](decoded)
            yield {"vision": decoded}

    monkeypatch.setattr(transfer, "_get_prompt_sample_data", format_prompt)
    monkeypatch.setattr(transfer_streaming, "_get_prompt_sample_data", format_prompt)
    monkeypatch.setattr(transfer, "build_transfer_batch", build_batch)
    model = SimpleNamespace(
        config=SimpleNamespace(
            tokenizer=SimpleNamespace(temporal_compression_factor=4),
            teacher_forcing_frames_per_chunk=1,
            causal_training_strategy="none",
            video_temporal_causal=True,
            fixed_step_sampler_config={"t_list": [1.0, 15 / 16, 5 / 6, 5 / 8], "sample_type": "sde"},
        ),
        input_caption_key="caption",
        fixed_step_sampler=Mock(),
        eval=Mock(),
        tokenizer_vision_gen=SimpleNamespace(use_cached_encoder=nullcontext, use_cached_decoder=nullcontext),
        iter_samples_from_batch_autoregressive=iter_ar,
        iter_samples_from_batch_autoregressive_streaming_transfer=iter_ar,
        decode=lambda latent: latent,
    )
    args, model_arg = cast(OmniSampleArgs, sample), cast(OmniMoTModel, model)
    if streaming:
        output = transfer_streaming.generate_transfer_batch(sample_args_list=[args], model=model_arg)[0]
    else:
        output = transfer.generate_transfer_sample(args, model_arg)
    return output, observed


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize(
    "hint", [TransferHintKey.EDGE, TransferHintKey.BLUR, TransferHintKey.DEPTH, TransferHintKey.SEG]
)
def test_precomputed_control_clock_reaches_prompt_model_and_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, streaming: bool, hint: TransferHintKey
) -> None:
    path = tmp_path / "control.mkv"
    _write_control(path)
    output, observed = _run_transfer(monkeypatch, path, streaming=streaming, hint=hint)
    assert output.fps == observed["model_fps"] == observed["prompt"]["fps"] == 24.0
    assert observed["prompt"]["duration"] == "1s"  # 25 source frames at 24 FPS, not 30 FPS.
    assert output.output_video.shape == (1, 3, 25, 480, 832)
    assert output.control_videos[hint].shape == output.output_video.shape
    assert output.original_hw == (48, 64)
    if not streaming:
        assert observed["batch_fps"] == 24.0


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("aspect_ratio,expected_hw", [("16,9", (480, 832)), (None, (544, 736))])
def test_control_only_explicit_aspect_or_source_detection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    streaming: bool,
    aspect_ratio: str | None,
    expected_hw: tuple[int, int],
) -> None:
    path = tmp_path / "control.mkv"
    _write_control(path, num_frames=5)
    output, observed = _run_transfer(monkeypatch, path, streaming=streaming, aspect_ratio=aspect_ratio, num_frames=5)
    assert output.output_video.shape[-2:] == expected_hw
    assert observed["prompt"]["resolution"] == {"H": expected_hw[0], "W": expected_hw[1]}


@pytest.mark.parametrize("streaming", [False, True])
def test_image_control_keeps_request_fps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, streaming: bool) -> None:
    path = tmp_path / "control.png"
    Image.fromarray(np.zeros((48, 64, 3), dtype=np.uint8)).save(path)
    output, observed = _run_transfer(monkeypatch, path, streaming=streaming, num_frames=1)
    assert output.fps == observed["model_fps"] == 30.0
    assert output.output_video.shape == (1, 3, 1, 480, 832)


@pytest.mark.parametrize("streaming", [False, True])
def test_source_rgb_clock_still_takes_precedence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, streaming: bool
) -> None:
    path = tmp_path / "source.mkv"
    _write_control(path)
    output, observed = _run_transfer(monkeypatch, path, streaming=streaming, has_rgb=True)
    assert output.fps == observed["model_fps"] == 24.0
    assert observed["prompt"]["duration"] == "1s"


@pytest.mark.parametrize("fps", [Fraction(24), Fraction(30000, 1001)])
def test_metadata_probe_matches_decoded_video_clock(tmp_path: Path, fps: Fraction) -> None:
    path = tmp_path / "control.mkv"
    _write_control(path, fps=fps, num_frames=5)
    pixels, decoded_fps = vision.read_media_frames(path, max_frames=5, video_decoder="pyav")
    shape, probed_fps = vision.probe_media_hw_fps(path, video_decoder="pyav")
    assert shape == tuple(pixels.shape[-2:]) == (48, 64)
    assert probed_fps == decoded_fps == float(fps)
