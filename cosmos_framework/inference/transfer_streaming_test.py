# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from collections.abc import Generator, Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from cosmos_framework.inference import inference
from cosmos_framework.inference import transfer as full_transfer
from cosmos_framework.inference import transfer_streaming as transfer
from cosmos_framework.inference.args import TransferHintKey
from cosmos_framework.inference.transfer_streaming import (
    _streaming_control_pixel_ranges,
    _StreamingTransferInput,
    generate_transfer_batch,
)


def test_streaming_control_pixel_ranges_use_singleton_prime_and_fixed_chunks() -> None:
    assert _streaming_control_pixel_ranges(9, 4, 4) == [(0, 1), (1, 17), (17, 33)]
    assert _streaming_control_pixel_ranges(4, 1, 4) == [(0, 1), (1, 5), (5, 9), (9, 13)]


@pytest.mark.parametrize("streaming", [False, True])
def test_output_callback_failure_closes_retained_ar_iterator(monkeypatch: pytest.MonkeyPatch, streaming: bool) -> None:
    """A disconnected output consumer must release the model iterator before the next request."""
    args = SimpleNamespace(
        show_control_condition=False,
        show_input=False,
        num_steps=4,
        shift=5.0,
        normalize_cfg=False,
        guidance=1.0,
        control_guidance=1.0,
        num_first_chunk_conditional_frames=0,
        share_vision_temporal_positions=True,
    )
    prepared = _StreamingTransferInput(
        sample_args=args,
        hint_key=TransferHintKey.DEPTH,
        control_frames=torch.zeros(3, 5, 2, 2, dtype=torch.uint8),
        input_frames=None,
        prompt="prompt",
        negative_prompt=None,
        total_frames=5,
        height=2,
        width=2,
        output_fps=30.0,
        original_hw=(2, 2),
        seed=42,
    )
    retained = []
    closed = []

    def iter_ar(**kwargs):
        def chunks():
            try:
                chunk = torch.zeros(1, 3, 1, 2, 2)
                kwargs["on_clean_vision_chunk"](chunk)
                yield {"vision": chunk}
            finally:
                closed.append(True)

        iterator = chunks()
        retained.append(iterator)
        return iterator

    def disconnected(chunk):
        raise RuntimeError("output consumer disconnected")

    model = SimpleNamespace(
        config=SimpleNamespace(
            causal_training_strategy="teacher_forcing",
            simulation_mode=None,
            video_temporal_causal=True,
            fixed_step_sampler_config={},
            tokenizer=SimpleNamespace(temporal_compression_factor=4),
            teacher_forcing_frames_per_chunk=1,
        ),
        input_caption_key="caption",
        tokenizer_vision_gen=SimpleNamespace(use_cached_encoder=nullcontext, use_cached_decoder=nullcontext),
        iter_samples_from_batch_autoregressive=iter_ar,
        iter_samples_from_batch_autoregressive_streaming_transfer=iter_ar,
        decode=lambda x: x,
        eval=Mock(),
    )
    monkeypatch.setattr(full_transfer, "build_transfer_batch", Mock(return_value={}))
    monkeypatch.setattr(transfer, "_prepare_streaming_transfer_input", Mock(return_value=prepared))
    with pytest.raises(RuntimeError, match="output consumer disconnected"):
        if streaming:
            generate_transfer_batch(sample_args_list=[args], model=model, on_output_video_chunk=disconnected)
        else:
            full_transfer._generate_causal_transfer_sample(
                sample_args=args,
                model=model,
                per_hint_frames={TransferHintKey.DEPTH: prepared.control_frames},
                input_frames=None,
                prompt="prompt",
                negative_prompt=None,
                total_frames=5,
                height=2,
                width=2,
                output_fps=30.0,
                original_hw=(2, 2),
                temporal_compression_factor=4,
                seed=42,
                on_output_video_chunk=disconnected,
            )
    assert closed == [True]
    assert retained[0].gi_frame is None


def test_generate_transfer_batch_pads_partial_final_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    """Streaming keeps fixed later chunks while trimming padded decoded frames."""

    monkeypatch.setattr(torch.Tensor, "cuda", lambda tensor: tensor)
    sample_args = SimpleNamespace(
        show_control_condition=False,
        show_input=False,
        num_steps=4,
        shift=5.0,
        normalize_cfg=False,
        guidance=1.0,
    )
    prepared = _StreamingTransferInput(
        sample_args=sample_args,
        hint_key=TransferHintKey.DEPTH,
        control_frames=torch.zeros(3, 18, 2, 2, dtype=torch.uint8),  # [C,T,H,W]
        input_frames=None,
        prompt="prompt",
        negative_prompt=None,
        total_frames=18,
        height=2,
        width=2,
        output_fps=30.0,
        original_hw=(2, 2),
        seed=10,
    )
    monkeypatch.setattr(transfer, "_prepare_streaming_transfer_input", Mock(return_value=prepared))
    encoded_pixel_lengths: list[int] = []
    requested_latent_frames: list[int] = []

    @contextmanager
    def cached_scope() -> Generator[None, None, None]:
        yield

    def encode(pixels: torch.Tensor) -> torch.Tensor:  # pixels: [B,3,T,H,W]
        encoded_pixel_lengths.append(int(pixels.shape[2]))
        latent_t = 1 if pixels.shape[2] == 1 else pixels.shape[2] // 4
        return torch.zeros(1, 4, latent_t, 1, 1)  # [B,C,T,H,W]

    def decode(latent: torch.Tensor) -> torch.Tensor:  # latent: [B,C,T,H,W]
        pixel_t = 1 if len(encoded_pixel_lengths) == 1 else 4 * latent.shape[2]
        return torch.zeros(1, 3, pixel_t, 2, 2)  # [B,3,T,H,W]

    def iter_ar(**kwargs: object) -> Iterator[dict[str, torch.Tensor]]:
        assert kwargs["data_batch"]["_preserve_transfer_numerics"] is True
        requested_latent_frames.append(int(kwargs["num_frames"]))
        callback = kwargs["on_clean_vision_chunk"]
        assert callable(callback)
        for control_latent in kwargs["control_latent_chunks"]:
            callback(control_latent)
            yield {"vision": control_latent}

    model = SimpleNamespace(
        config=SimpleNamespace(
            tokenizer=SimpleNamespace(temporal_compression_factor=4),
            teacher_forcing_frames_per_chunk=4,
            fixed_step_sampler_config={},
        ),
        input_caption_key="caption",
        tokenizer_vision_gen=SimpleNamespace(
            use_cached_encoder=cached_scope,
            use_cached_decoder=cached_scope,
        ),
        iter_samples_from_batch_autoregressive_streaming_transfer=iter_ar,
        encode=encode,
        decode=decode,
        eval=Mock(),
    )

    outputs = generate_transfer_batch(sample_args_list=[Mock()], model=model)

    assert encoded_pixel_lengths == [1, 16, 16]
    assert requested_latent_frames == [9]
    assert outputs[0].output_video.shape == (1, 3, 18, 2, 2)
    assert outputs[0].control_videos[TransferHintKey.DEPTH].shape == (1, 3, 18, 2, 2)


def test_generate_transfer_batch_streams_batched_condition_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two prompts share three causal calls while condition encoding stays chunkwise."""

    monkeypatch.setattr(torch.Tensor, "cuda", lambda tensor: tensor)
    controls = [torch.full((3, 9, 2, 2), sample_idx, dtype=torch.uint8) for sample_idx in range(2)]  # list of [3,T,H,W]
    sample_args = [
        SimpleNamespace(
            show_control_condition=False,
            show_input=False,
            num_steps=4,
            shift=5.0,
            normalize_cfg=False,
            guidance=1.0,
        )
        for _ in range(2)
    ]
    prepared = [
        _StreamingTransferInput(
            sample_args=sample_args[sample_idx],
            hint_key=TransferHintKey.DEPTH,
            control_frames=controls[sample_idx],
            input_frames=None,
            prompt=f"prompt {sample_idx}",
            negative_prompt=None,
            total_frames=9,
            height=2,
            width=2,
            output_fps=30.0,
            original_hw=(2, 2),
            seed=10 + sample_idx,
        )
        for sample_idx in range(2)
    ]
    prepare = Mock(side_effect=prepared)
    monkeypatch.setattr(transfer, "_prepare_streaming_transfer_input", prepare)

    events: list[str] = []
    encoder_active = False
    decoder_active = False
    encoded_pixel_lengths: list[int] = []
    ar_calls: list[dict[str, object]] = []

    @contextmanager
    def cached_encoder() -> Generator[None, None, None]:
        nonlocal encoder_active
        encoder_active = True
        try:
            yield
        finally:
            encoder_active = False

    @contextmanager
    def cached_decoder() -> Generator[None, None, None]:
        nonlocal decoder_active
        decoder_active = True
        try:
            yield
        finally:
            decoder_active = False

    def encode(pixels: torch.Tensor) -> torch.Tensor:  # pixels: [B,3,T,H,W]
        assert encoder_active
        encoded_pixel_lengths.append(int(pixels.shape[2]))
        events.append(f"encode-{pixels.shape[2]}")
        latent_t = 1 if pixels.shape[2] == 1 else pixels.shape[2] // 4
        return torch.zeros(2, 4, latent_t, 1, 1)  # [B,C,T,H,W]

    def decode(latent: torch.Tensor) -> torch.Tensor:  # latent: [B,C,T,H,W]
        assert decoder_active
        events.append("decode")
        pixel_t = 1 if len(encoded_pixel_lengths) == 1 else 4 * latent.shape[2]
        return torch.zeros(2, 3, pixel_t, 2, 2)  # [B,3,T,H,W]

    def iter_ar(**kwargs: object) -> Iterator[dict[str, torch.Tensor]]:
        assert kwargs["data_batch"]["_preserve_transfer_numerics"] is True
        ar_calls.append(kwargs)
        callback = kwargs["on_clean_vision_chunk"]
        assert callable(callback)
        for control_latent in kwargs["control_latent_chunks"]:
            events.append("ar")
            callback(control_latent)
            yield {"vision": control_latent}

    model = SimpleNamespace(
        config=SimpleNamespace(
            tokenizer=SimpleNamespace(temporal_compression_factor=4),
            teacher_forcing_frames_per_chunk=1,
            fixed_step_sampler_config={},
        ),
        input_caption_key="caption",
        tokenizer_vision_gen=SimpleNamespace(
            use_cached_encoder=cached_encoder,
            use_cached_decoder=cached_decoder,
        ),
        iter_samples_from_batch_autoregressive_streaming_transfer=iter_ar,
        encode=encode,
        decode=decode,
        eval=Mock(),
    )
    streamed_chunks: list[torch.Tensor] = []
    resolved_frame_count = Mock()

    outputs = generate_transfer_batch(
        sample_args_list=[Mock(), Mock()],
        model=model,
        on_output_video_chunk=streamed_chunks.append,
        on_total_frames_resolved=resolved_frame_count,
    )

    resolved_frame_count.assert_called_once_with(9)
    assert encoded_pixel_lengths == [1, 4, 4]
    assert events == ["encode-1", "ar", "decode", "encode-4", "ar", "decode", "encode-4", "ar", "decode"]
    assert len(ar_calls) == 1
    assert ar_calls[0]["data_batch"]["caption"] == ["prompt 0", "prompt 1"]
    assert ar_calls[0]["num_frames"] == 3
    assert ar_calls[0]["seeds"] == [10, 11]
    assert [tuple(chunk.shape) for chunk in streamed_chunks] == [(2, 3, 1, 2, 2), (2, 3, 4, 2, 2), (2, 3, 4, 2, 2)]
    assert [tuple(output.output_video.shape) for output in outputs] == [(1, 3, 9, 2, 2)] * 2
    assert [tuple(output.control_videos[TransferHintKey.DEPTH].shape) for output in outputs] == [
        (1, 3, 9, 2, 2),
        (1, 3, 9, 2, 2),
    ]
    assert not encoder_active
    assert not decoder_active


def test_streaming_controls_use_validated_pyav_decoder(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every RGB/control read uses the decoder matching native Transfer inputs."""
    sample = SimpleNamespace(
        resolution="480",
        transfer_hints={TransferHintKey.DEPTH: SimpleNamespace(control_path="depth.mp4")},
        vision_path="rgb.mp4",
        aspect_ratio=None,
        max_frames=9,
        fps=30,
    )
    frames = torch.zeros(3, 9, 2, 2, dtype=torch.uint8)  # [3,T,H,W]
    reads = Mock(return_value=(frames, 30.0, "1,1", (2, 2)))
    controls = Mock(return_value=frames)
    guard = Mock()
    monkeypatch.setattr(transfer, "read_and_resize_media", reads)
    monkeypatch.setattr(transfer, "load_transfer_control_frames", controls)
    monkeypatch.setattr(transfer, "_require_causal_transfer_pyav", guard)

    media = transfer._load_streaming_transfer_media(sample)

    guard.assert_called_once()
    assert reads.call_args.args == (Path("rgb.mp4"),)
    assert reads.call_args.kwargs["video_decoder"] == "pyav"
    assert controls.call_args.kwargs["video_decoder"] == "pyav"
    assert media.total_frames == 9
    assert media.output_fps == 30.0


def test_streaming_hf_student_keeps_negative_prompt_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """An exported causal student need not retain the distillation training class."""
    sample = Mock(
        autoregressive=True,
        num_first_chunk_conditional_frames=0,
        control_guidance=1.0,
        share_vision_temporal_positions=True,
        emphasize_control_in_prompt=False,
        seed=42,
    )
    config = SimpleNamespace(
        video_temporal_causal=True,
        fixed_step_sampler_config={"sample_type": "sde"},
        teacher_forcing_frames_per_chunk=1,
        tokenizer=SimpleNamespace(temporal_compression_factor=4),
    )
    model = SimpleNamespace(config=config, input_caption_key="caption")
    media = transfer._StreamingTransferMedia(
        hint_key=TransferHintKey.DEPTH,
        control_frames=torch.zeros(3, 9, 2, 2, dtype=torch.uint8),  # [3,T,H,W]
        input_frames=None,
        total_frames=9,
        height=2,
        width=2,
        output_fps=30.0,
        original_hw=(2, 2),
    )
    format_prompt = Mock(return_value={"caption": ["positive metadata"], "neg_caption": ["negative metadata"]})
    monkeypatch.setattr(transfer, "_load_streaming_transfer_media", Mock(return_value=media))
    monkeypatch.setattr(transfer, "_get_prompt_sample_data", format_prompt)

    prepared = transfer._prepare_streaming_transfer_input(sample, model)

    sample.model_copy.assert_called_once_with(update={"num_frames": 9, "fps": 30, "autoregressive": False})
    assert prepared.prompt == "positive metadata"
    assert prepared.negative_prompt == "negative metadata"
    assert prepared.seed == 42


@pytest.mark.parametrize("stream_controls", [False, True])
def test_public_transfer_routes_condition_streaming(monkeypatch: pytest.MonkeyPatch, stream_controls: bool) -> None:
    """The public setup flag selects incremental condition encoding without retaining warmup media."""
    incremental = Mock(return_value=[])
    full = Mock()
    monkeypatch.setattr(transfer, "generate_transfer_batch", incremental)
    monkeypatch.setattr(full_transfer, "generate_transfer_sample", full)
    monkeypatch.setattr(inference, "sync_distributed_errors", nullcontext)
    sample = Mock()
    runner = SimpleNamespace(
        setup_args=SimpleNamespace(condition_stream_encode=stream_controls),
        model=Mock(),
        should_process_sample=lambda _sample: False,
    )

    outputs = inference.OmniInference._generate_transfer_batch(runner, sample, save_outputs=False)

    assert outputs == []
    if stream_controls:
        incremental.assert_called_once_with(sample_args_list=[sample], model=runner.model, materialize_outputs=False)
        full.assert_not_called()
    else:
        full.assert_called_once_with(sample_args=sample, model=runner.model)
        incremental.assert_not_called()
