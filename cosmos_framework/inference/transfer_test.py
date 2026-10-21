# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Causal Transfer inference contracts mirrored from the native runtime."""

from collections.abc import Generator, Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch

from cosmos_framework.data.generator.sequence_packing import SequencePlan
from cosmos_framework.inference.args import TransferHintKey
from cosmos_framework.inference.transfer import (
    _align_ar_frames_to_chunk_partition,
    _generate_causal_transfer_sample,
    _require_causal_transfer_pyav,
    _resolve_transfer_seed,
    build_control_cfg_postprocess,
    build_transfer_batch,
)
from cosmos_framework.model.generator.mot.diffusion_cache import DiffusionCache, _velocity_pathways
from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel, VelocityPostprocess
from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean


@pytest.fixture(autouse=True)
def _supported_causal_transfer_pyav(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep Transfer tests independent of the PyAV installed by the test host."""
    from cosmos_framework.inference import transfer

    monkeypatch.setattr(transfer.av, "__version__", "18.0.0")


@pytest.mark.parametrize(
    ("version", "supported"),
    (("18.0.0", True), ("18.1.0", True), ("17.1.0", False), ("19.0.0", False), ("unknown", False)),
)
def test_causal_transfer_requires_pyav_18(
    monkeypatch: pytest.MonkeyPatch,
    version: str,
    supported: bool,
) -> None:
    from cosmos_framework.inference import transfer

    monkeypatch.setattr(transfer.av, "__version__", version)
    if supported:
        _require_causal_transfer_pyav()
    else:
        with pytest.raises(RuntimeError, match=r"requires PyAV >=18,<19.*Python >=3\.11"):
            _require_causal_transfer_pyav()


@pytest.mark.parametrize("autoregressive", [False, True])
@pytest.mark.parametrize("has_vision", [False, True])
def test_causal_transfer_selects_pyav_for_both_input_and_control(
    monkeypatch: pytest.MonkeyPatch, autoregressive: bool, has_vision: bool
) -> None:
    from cosmos_framework.inference import transfer

    frames = torch.zeros(3, 9, 2, 2, dtype=torch.uint8)  # [C,T,H,W]
    read_media = Mock(return_value=(frames, 30.0, "1,1", (2, 2)))
    probe_hw = Mock(return_value=((2, 2), 30.0))
    load_controls = Mock(side_effect=RuntimeError("stop after media selection"))
    monkeypatch.setattr(transfer, "read_and_resize_media", read_media)
    monkeypatch.setattr(transfer, "probe_media_hw_fps", probe_hw)
    monkeypatch.setattr(transfer, "load_transfer_control_frames", load_controls)
    sample = SimpleNamespace(
        transfer_hints={TransferHintKey.DEPTH: SimpleNamespace(control_path=Path("depth.mp4"))},
        resolution="480",
        max_frames=9,
        num_video_frames_per_chunk=9,
        num_conditional_frames=0,
        num_first_chunk_conditional_frames=0,
        vision_path=Path("rgb.mp4") if has_vision else None,
        aspect_ratio="1,1",
        autoregressive=autoregressive,
    )

    with pytest.raises(RuntimeError, match="stop after media selection"):
        transfer.generate_transfer_sample(sample, SimpleNamespace())

    expected_decoder = "pyav" if autoregressive else "torchcodec"
    if has_vision:
        assert read_media.call_args.args == (Path("rgb.mp4"),)
        assert read_media.call_args.kwargs["video_decoder"] == expected_decoder
        probe_hw.assert_not_called()
    else:
        # Without RGB, probe control geometry and timing without decoding the full clip twice.
        read_media.assert_not_called()
        assert probe_hw.call_args.args == (Path("depth.mp4"),)
        assert probe_hw.call_args.kwargs["video_decoder"] == expected_decoder
    assert load_controls.call_args.kwargs["video_decoder"] == expected_decoder


@pytest.mark.parametrize("decoder", ["torchcodec", "pyav"])
def test_precomputed_control_forwards_decoder_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, decoder: str
) -> None:
    from cosmos_framework.inference import transfer

    path = tmp_path / "depth.mp4"
    path.touch()
    frames = torch.zeros(3, 9, 2, 2, dtype=torch.uint8)  # [C,T,H,W]
    read_media = Mock(return_value=(frames, 30.0, "1,1", (2, 2)))
    monkeypatch.setattr(transfer, "read_and_resize_media", read_media)

    actual = transfer.load_transfer_control_frames(  # [C,T,H,W]
        hint_key=TransferHintKey.DEPTH,
        transfer=SimpleNamespace(control_path=path),
        resolution="480",
        aspect_ratio="1,1",
        max_frames=9,
        video_decoder=decoder,
    )

    assert actual is frames
    assert read_media.call_args.kwargs["video_decoder"] == decoder


def test_streaming_transfer_seed_is_stable_per_sample_identity() -> None:
    """Seedless samples resolve identically on independently prepared ranks."""
    rank_0_sample = SimpleNamespace(seed=None, name="sample-7", output_dir=Path("/outputs/run"))
    rank_1_sample = SimpleNamespace(seed=None, name="sample-7", output_dir=Path("/outputs/run"))
    other_sample = SimpleNamespace(seed=None, name="sample-8", output_dir=Path("/outputs/run"))

    rank_0_seed = _resolve_transfer_seed(rank_0_sample)
    rank_1_seed = _resolve_transfer_seed(rank_1_sample)
    other_seed = _resolve_transfer_seed(other_sample)

    assert rank_0_seed == rank_1_seed == 962391162
    assert other_seed != rank_0_seed
    assert 0 <= rank_0_seed <= 0x7FFFFFFF


def test_streaming_transfer_seed_preserves_explicit_value() -> None:
    sample_args = SimpleNamespace(seed=73, name="sample-7", output_dir=Path("/outputs/run"))

    assert _resolve_transfer_seed(sample_args) == 73


@pytest.mark.parametrize(
    (
        "causal_training_strategy",
        "simulation_mode",
        "video_temporal_causal",
        "hint_key",
        "distilled",
        "expected_guidance",
    ),
    [
        ("teacher_forcing", None, True, TransferHintKey.DEPTH, False, 4.0),
        ("teacher_forcing_dcm", None, True, TransferHintKey.EDGE, True, 1.0),
        ("none", "self_forcing", True, TransferHintKey.SEG, True, 1.0),
        ("none", None, True, TransferHintKey.BLUR, True, 1.0),
    ],
)
def test_causal_transfer_uses_ar_iterator(
    monkeypatch: pytest.MonkeyPatch,
    causal_training_strategy: str,
    simulation_mode: str | None,
    video_temporal_causal: bool,
    hint_key: TransferHintKey,
    distilled: bool,
    expected_guidance: float,
) -> None:
    """The wrapper builds one control/RGB batch and decodes RF or distilled chunks."""
    from cosmos_framework.inference import transfer

    built_batch = {"video": "aligned-control-rgb"}
    build_batch = Mock(return_value=built_batch)
    monkeypatch.setattr(transfer, "build_transfer_batch", build_batch)

    ar_calls: list[dict[str, object]] = []
    decoder_active = False
    refreshed_chunk_count = 0

    def iter_ar(**kwargs: object) -> Iterator[dict[str, torch.Tensor]]:
        nonlocal refreshed_chunk_count
        ar_calls.append(kwargs)
        callback = kwargs["on_clean_vision_chunk"]
        assert callable(callback)
        first_chunk = torch.zeros(1, 3, 1, 2, 2)  # [B,C,T,H,W]
        second_chunk = torch.ones(1, 3, 5, 2, 2)  # [B,C,T,H,W]
        callback(first_chunk)
        refreshed_chunk_count += 1
        yield {"vision": first_chunk}
        callback(second_chunk)
        refreshed_chunk_count += 1
        yield {"vision": second_chunk}

    @contextmanager
    def cached_decoder() -> Generator[None, None, None]:
        nonlocal decoder_active
        decoder_active = True
        try:
            yield
        finally:
            decoder_active = False

    def decode(latent: torch.Tensor) -> torch.Tensor:  # latent: [B,C,T,H,W]
        assert decoder_active
        return latent

    model = SimpleNamespace(
        config=SimpleNamespace(
            causal_training_strategy=causal_training_strategy,
            simulation_mode=simulation_mode,
            video_temporal_causal=video_temporal_causal,
            fixed_step_sampler_config={} if distilled else None,
        ),
        input_caption_key="caption",
        tokenizer_vision_gen=SimpleNamespace(use_cached_decoder=cached_decoder),
        iter_samples_from_batch_autoregressive=iter_ar,
        decode=decode,
    )
    sample_args = SimpleNamespace(
        control_guidance=1.0,
        guidance=4.0,
        normalize_cfg=False,
        num_first_chunk_conditional_frames=0,
        num_steps=50,
        share_vision_temporal_positions=True,
        shift=5.0,
        show_control_condition=False,
        show_input=False,
    )
    control_frames = torch.zeros(3, 5, 2, 2, dtype=torch.uint8)  # [C,T,H,W]
    streamed_chunks: list[torch.Tensor] = []

    def collect_streamed_chunk(chunk: torch.Tensor) -> None:  # chunk: [B,C,T,H,W]
        assert refreshed_chunk_count == len(streamed_chunks) + 1
        streamed_chunks.append(chunk)

    output = transfer._generate_causal_transfer_sample(
        sample_args=sample_args,
        model=model,
        per_hint_frames={hint_key: control_frames},
        input_frames=None,
        prompt="a prompt",
        negative_prompt="avoid blur",
        total_frames=5,
        height=2,
        width=2,
        output_fps=30.0,
        original_hw=(2, 2),
        temporal_compression_factor=4,
        seed=7,
        on_output_video_chunk=collect_streamed_chunk,
    )

    assert len(ar_calls) == 1
    assert ar_calls[0]["data_batch"] is built_batch
    assert built_batch["_preserve_transfer_numerics"] is True
    assert ar_calls[0]["mode"] == "video_transfer"
    assert ar_calls[0]["guidance"] == expected_guidance
    assert ar_calls[0]["num_steps"] == 50
    assert ar_calls[0]["seed"] == 7
    assert [chunk.shape for chunk in streamed_chunks] == [(1, 3, 1, 2, 2), (1, 3, 4, 2, 2)]
    assert output.output_video.shape == (1, 3, 5, 2, 2)
    streamed_output = torch.cat(streamed_chunks, dim=2)  # [B,C,T,H,W]
    assert torch.equal(streamed_output, output.output_video)
    assert output.control_videos[hint_key].shape == (1, 3, 5, 2, 2)
    build_batch.assert_called_once()
    build_kwargs = build_batch.call_args.kwargs
    assert build_kwargs["negative_prompt"] == "avoid blur"
    assert build_kwargs["share_vision_temporal_positions"] is True
    assert build_kwargs["control_weights"] == [1.0]


@pytest.mark.parametrize("total_frames", [1, 2, 4, 5, 599, 600, 601])
@pytest.mark.parametrize(
    ("show_control_condition", "show_input"),
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_causal_transfer_pads_encoder_input_without_extending_outputs(
    monkeypatch: pytest.MonkeyPatch,
    total_frames: int,
    show_control_condition: bool,
    show_input: bool,
) -> None:
    """The real batch must satisfy the VAE boundary while retaining every source/output frame."""
    # Exercise build_transfer_batch itself on CPU, replacing only its device transfer.
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *args, **kwargs: self)
    encoded_frames = 1 + ((total_frames - 1 + 3) // 4) * 4
    frame_ids = torch.arange(total_frames)
    source_control = (
        torch.stack([frame_ids % 256, frame_ids // 256, 255 - frame_ids % 256])
        .to(torch.uint8)[:, :, None, None]
        .expand(3, total_frames, 2, 2)
        .clone()
    )  # [C,T,H,W]
    source_input = 255 - source_control  # [C,T,H,W]
    expected_control = source_control.float() / 127.5 - 1.0  # [C,T,H,W]
    # The frontend already supplies a padded processing window; the causal wrapper
    # must not remove necessary padding when it limits media to total_frames.
    prepared_control = torch.cat(
        [source_control, source_control[:, -1:].expand(-1, encoded_frames - total_frames, -1, -1)], dim=1
    )  # [C,T_encoder,H,W]
    original_prepared_control = prepared_control.clone()
    generated = torch.linspace(-0.75, 0.75, encoded_frames).reshape(1, 1, encoded_frames, 1, 1)
    generated = generated.expand(1, 3, encoded_frames, 2, 2).clone()  # [B,C,T_encoder,H,W]
    checked_batches: list[dict[str, Any]] = []

    def iter_ar(
        *, data_batch: dict[str, Any], on_clean_vision_chunk: Any, **kwargs: Any
    ) -> Iterator[dict[str, torch.Tensor]]:
        # This strict encoder boundary reproduces Wan's actual precondition. The
        # model callback is a fixture; real checkpoint/GPU inference is a separate test.
        control, target = data_batch["video"]
        for item in (control, target):
            assert item.shape[2] == 1 or (item.shape[2] - 1) % 4 == 0
            assert item.shape == (1, 3, encoded_frames, 2, 2)
            assert item.shape[2] == data_batch["num_frames"].item()
        torch.testing.assert_close(control[0, :, :total_frames], expected_control.to(torch.bfloat16))
        assert torch.count_nonzero(target).item() == 0
        if total_frames == 600:
            # 600 source frames need exactly one repeated tail frame, not a trim
            # to 597 or a replacement of the original final source frame.
            torch.testing.assert_close(control[:, :, 600], control[:, :, 599])
        assert data_batch["sequence_plan"][0].share_vision_temporal_positions
        checked_batches.append(data_batch)
        for start, end in [(0, 1), *[(start, start + 4) for start in range(1, encoded_frames, 4)]]:
            decoded_chunk = generated[:, :, start:end]  # [B,C,1_or_4,H,W]
            on_clean_vision_chunk(decoded_chunk)
            yield {"vision": decoded_chunk}

    model = SimpleNamespace(
        config=SimpleNamespace(
            causal_training_strategy="teacher_forcing",
            simulation_mode=None,
            video_temporal_causal=True,
            fixed_step_sampler_config={},
        ),
        input_caption_key="caption",
        tokenizer_vision_gen=SimpleNamespace(use_cached_decoder=nullcontext),
        iter_samples_from_batch_autoregressive=iter_ar,
        decode=lambda latent: latent,
    )
    sample_args = SimpleNamespace(
        control_guidance=1.0,
        guidance=1.0,
        normalize_cfg=False,
        num_first_chunk_conditional_frames=0,
        num_steps=4,
        share_vision_temporal_positions=True,
        transfer_resize_mode="direct_resize",
        shift=5.0,
        show_control_condition=show_control_condition,
        show_input=show_input,
    )
    streamed_chunks: list[torch.Tensor] = []
    output = _generate_causal_transfer_sample(
        sample_args=sample_args,
        model=model,
        per_hint_frames={TransferHintKey.DEPTH: prepared_control},
        input_frames=source_input if show_input else None,
        prompt="Keep all source frames.",
        negative_prompt=None,
        total_frames=total_frames,
        height=2,
        width=2,
        output_fps=30.0,
        original_hw=(2, 2),
        temporal_compression_factor=4,
        seed=7,
        on_output_video_chunk=streamed_chunks.append,
    )
    assert len(checked_batches) == 1
    expected_panels = [generated[:, :, :total_frames]]
    if show_control_condition:
        expected_panels.insert(0, expected_control.unsqueeze(0))
    if show_input:
        expected_panels.insert(0, (source_input.float() / 127.5 - 1.0).unsqueeze(0))
    expected_output = torch.cat(expected_panels, dim=-1)  # [B,C,T,H,W_panels]
    assert torch.equal(output.output_video, expected_output)
    assert torch.equal(torch.cat(streamed_chunks, dim=2), expected_output)
    assert sum(chunk.shape[2] for chunk in streamed_chunks) == total_frames
    assert torch.equal(output.output_video[:, :, -1], expected_output[:, :, total_frames - 1])
    assert torch.equal(output.control_videos[TransferHintKey.DEPTH], expected_control.unsqueeze(0))
    assert torch.equal(prepared_control, original_prepared_control)
    if total_frames == 600:
        assert streamed_chunks[-1].shape[2] == 3


@pytest.mark.parametrize(
    ("total_frames", "expected"),
    (
        (25, 17),
        (201, 193),
        (18, 17),
        (17, 17),
        (33, 33),
        (14, 14),
    ),
)
def test_align_ar_frames_trims_to_chunk_partition(total_frames: int, expected: int) -> None:
    assert _align_ar_frames_to_chunk_partition(total_frames, 4, 4) == expected


def test_generate_transfer_reports_post_alignment_frame_count(monkeypatch: pytest.MonkeyPatch) -> None:
    from cosmos_framework.inference import transfer

    input_frames = torch.zeros(3, 25, 2, 2, dtype=torch.uint8)  # [C,T,H,W]
    control_frames = torch.zeros(3, 25, 2, 2, dtype=torch.uint8)  # [C,T,H,W]
    generated_output = SimpleNamespace()
    resolved_frame_count = Mock()

    def assert_resolved_before_generation(**_: object) -> SimpleNamespace:
        resolved_frame_count.assert_called_once_with(17)
        return generated_output

    generate_causal = Mock(side_effect=assert_resolved_before_generation)
    monkeypatch.setattr(
        transfer,
        "read_and_resize_media",
        Mock(return_value=(input_frames, 24.0, "16,9", (2, 2))),
    )
    monkeypatch.setattr(transfer, "load_transfer_control_frames", Mock(return_value=control_frames))
    monkeypatch.setattr(transfer, "_get_prompt_sample_data", Mock(return_value={"caption": ["prompt"]}))
    monkeypatch.setattr(transfer, "_generate_causal_transfer_sample", generate_causal)
    sample_args = SimpleNamespace(
        transfer_hints={TransferHintKey.DEPTH: SimpleNamespace(weight=1.0)},
        resolution="480",
        vision_path=Path("input.mp4"),
        num_frames=999,
        max_frames=25,
        num_video_frames_per_chunk=17,
        num_conditional_frames=0,
        num_first_chunk_conditional_frames=0,
        aspect_ratio=None,
        fps=24,
        autoregressive=True,
        emphasize_control_in_prompt=False,
        seed=7,
        model_copy=Mock(return_value=SimpleNamespace()),
    )
    model = SimpleNamespace(
        config=SimpleNamespace(
            tokenizer=SimpleNamespace(temporal_compression_factor=4),
            teacher_forcing_frames_per_chunk=4,
            fixed_step_sampler_config={},
        ),
        input_caption_key="caption",
        fixed_step_sampler=Mock(),
        eval=Mock(),
    )

    output = transfer.generate_transfer_sample(
        sample_args,
        model,
        on_total_frames_resolved=resolved_frame_count,
    )

    assert output is generated_output
    resolved_frame_count.assert_called_once_with(17)
    assert generate_causal.call_args.kwargs["total_frames"] == 17
    sample_args.model_copy.assert_called_once_with(update={"num_frames": 17, "fps": 24, "autoregressive": False})


@pytest.mark.parametrize("total_frames", (1, 5, 17, 25, 201))
def test_align_ar_frames_framewise_chunk_is_noop(total_frames: int) -> None:
    assert _align_ar_frames_to_chunk_partition(total_frames, 4, 1) == total_frames


@pytest.mark.parametrize("total_frames", (1, 9, 13))
def test_align_ar_frames_rejects_input_shorter_than_one_chunk(total_frames: int) -> None:
    with pytest.raises(ValueError, match="at least one full latent chunk"):
        _align_ar_frames_to_chunk_partition(total_frames, 4, 4)


def test_align_ar_frames_rejects_nonpositive_frame_count() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        _align_ar_frames_to_chunk_partition(0, 4, 4)


@pytest.mark.parametrize(
    ("fixed_step_sampler_config", "negative_prompt", "expected_sampler_mode", "expected_guidance"),
    [
        (object(), None, "distilled", 1.0),
        (None, "avoid artifacts", "rf", 7.0),
    ],
)
def test_causal_transfer_configures_ar_cfg(
    monkeypatch: pytest.MonkeyPatch,
    fixed_step_sampler_config: object | None,
    negative_prompt: str | None,
    expected_sampler_mode: str,
    expected_guidance: float,
) -> None:
    """The wrapper preserves negative prompts and selects distilled or RF sampling."""
    from cosmos_framework.inference import transfer

    built_batch: dict[str, object] = {"video": "aligned-control-target"}
    if negative_prompt is not None:
        built_batch["neg_caption"] = [negative_prompt]
    build_batch = Mock(return_value=built_batch)
    monkeypatch.setattr(transfer, "build_transfer_batch", build_batch)
    ar_calls: list[dict[str, object]] = []

    def iter_ar(**kwargs: object) -> Iterator[dict[str, torch.Tensor]]:
        ar_calls.append(kwargs)
        callback = kwargs["on_clean_vision_chunk"]
        assert callable(callback)
        first_chunk = torch.zeros(1, 3, 1, 2, 2)  # [B,C,T,H,W]
        second_chunk = torch.ones(1, 3, 4, 2, 2)  # [B,C,T,H,W]
        callback(first_chunk)
        callback(second_chunk)
        yield {"vision": first_chunk}
        yield {"vision": second_chunk}

    @contextmanager
    def cached_decoder() -> Generator[None, None, None]:
        yield

    def decode(latent: torch.Tensor) -> torch.Tensor:  # latent: [B,C,T,H,W], returns [B,C,T,H,W]
        return latent

    model = SimpleNamespace(
        config=SimpleNamespace(
            causal_training_strategy="none",
            simulation_mode="self_forcing",
            video_temporal_causal=True,
            fixed_step_sampler_config=fixed_step_sampler_config,
        ),
        input_caption_key="caption",
        tokenizer_vision_gen=SimpleNamespace(use_cached_decoder=cached_decoder),
        iter_samples_from_batch_autoregressive=iter_ar,
        decode=decode,
    )
    sample_args = SimpleNamespace(
        control_guidance=1.0,
        guidance=7.0,
        normalize_cfg=False,
        num_first_chunk_conditional_frames=0,
        num_steps=4,
        share_vision_temporal_positions=1,
        shift=3.0,
        show_control_condition=False,
        show_input=False,
    )
    control_frames = torch.zeros(3, 5, 2, 2, dtype=torch.uint8)  # [C,T,H,W]

    output = _generate_causal_transfer_sample(
        sample_args=sample_args,
        model=model,
        per_hint_frames={TransferHintKey.DEPTH: control_frames},
        input_frames=None,
        prompt="a prompt",
        negative_prompt=negative_prompt,
        total_frames=5,
        height=2,
        width=2,
        output_fps=24.0,
        original_hw=(2, 2),
        temporal_compression_factor=4,
        seed=7,
    )

    assert ar_calls[0]["data_batch"] is built_batch
    assert built_batch["_preserve_transfer_numerics"] is True
    assert ar_calls[0]["mode"] == "video_transfer"
    assert ar_calls[0]["sampler_mode"] == expected_sampler_mode
    assert ar_calls[0]["guidance"] == expected_guidance
    assert ar_calls[0]["has_negative_prompt"] == (negative_prompt is not None)
    assert output.output_video.shape == (1, 3, 5, 2, 2)
    assert output.control_videos[TransferHintKey.DEPTH].shape == (1, 3, 5, 2, 2)
    build_batch.assert_called_once()
    assert build_batch.call_args.kwargs["negative_prompt"] == negative_prompt


def test_causal_transfer_checks_ar_capability_before_building_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    """An incompatible model fails before constructing the transfer batch."""
    from cosmos_framework.inference import transfer

    build_batch = Mock()
    monkeypatch.setattr(transfer, "build_transfer_batch", build_batch)
    model = SimpleNamespace(
        config=SimpleNamespace(
            causal_training_strategy="teacher_forcing",
            simulation_mode=None,
            video_temporal_causal=True,
        ),
        input_caption_key="caption",
    )
    sample_args = SimpleNamespace(
        control_guidance=1.0,
        num_first_chunk_conditional_frames=0,
        share_vision_temporal_positions=True,
    )
    control_frames = torch.zeros(3, 5, 2, 2, dtype=torch.uint8)  # [C,T,H,W]

    with pytest.raises(TypeError, match="does not implement causal autoregressive inference"):
        _generate_causal_transfer_sample(
            sample_args=sample_args,
            model=model,
            per_hint_frames={TransferHintKey.DEPTH: control_frames},
            input_frames=None,
            prompt="a prompt",
            negative_prompt=None,
            total_frames=5,
            height=2,
            width=2,
            output_fps=24.0,
            original_hw=(2, 2),
            temporal_compression_factor=4,
            seed=7,
        )

    build_batch.assert_not_called()


def test_transfer_batch_carries_explicit_single_view_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.Tensor, "cuda", lambda tensor: tensor)
    control_video = torch.zeros((3, 5, 4, 6), dtype=torch.bfloat16)  # [C,T,H,W]
    target_video = torch.zeros((3, 5, 4, 6), dtype=torch.bfloat16)  # [C,T,H,W]

    batch = build_transfer_batch(
        control_videos=[control_video],
        target_video=target_video,
        num_frames=5,
        height=4,
        width=6,
        fps=10.0,
        num_conditional_frames=1,
        temporal_compression_factor=4,
        prompt_key="caption",
        prompt="test",
        negative_prompt=None,
        share_vision_temporal_positions=True,
    )

    assert batch["num_vision_items_per_sample"] == [2]
    assert batch["num_views_per_vision_item"] == [1, 1]
    assert batch["is_preprocessed"] is True
    assert "enable_per_camera_vae_encoding" not in batch

    metadata = OmniMoTModel._get_explicit_single_view_metadata(
        batch,
        num_vision_items_per_sample=[2],
        batch_size=1,
    )
    assert metadata == [1, 1]


@pytest.mark.parametrize("cache_enabled", [False, True])
@pytest.mark.parametrize("interval", [None, [0.25, 0.75]])
def test_control_cfg_declares_actual_cfg_branches(cache_enabled: bool, interval: list[float] | None) -> None:
    """Declare every executed branch, including when caching is off, for FSDP alignment."""
    control = torch.zeros(2, 3)
    target = torch.ones(4, 3)
    data = GenerationDataClean(
        batch_size=1,
        is_image_batch=False,
        x0_tokens_vision=[control, target],
        num_vision_items_per_sample=[2],
    )
    no_control = torch.zeros(target.numel())
    model = SimpleNamespace(_get_velocity=Mock(return_value=[no_control]))
    if cache_enabled:
        model._diffusion_cache = Mock()
    builder = build_control_cfg_postprocess(control_guidance=2.0, control_guidance_interval=interval)
    assert builder is not None
    postprocess = builder(
        model=model,
        cond_tokens=[[7]],
        sequence_plans=[SequencePlan(has_text=True, has_vision=True)],
        gen_data_clean=data,
    )
    assert isinstance(postprocess, VelocityPostprocess)
    noise = torch.zeros(control.numel() + target.numel())
    full_velocity = torch.ones_like(noise)
    for value in (0.9, 0.75, 0.5, 0.25, 0.1):
        timestep = torch.tensor([[value]])
        before = model._get_velocity.call_count
        pathways = postprocess.cfg_branches(timestep)
        assert model._get_velocity.call_count == before
        postprocess([full_velocity], [noise], timestep, 1.0)
        assert model._get_velocity.call_count - before == len(pathways)
        active = interval is None or 0.25 < value < 0.75
        assert pathways == (("cond_no_control",) if active else ())
    if cache_enabled:
        assert model._diffusion_cache.mock_calls == []


def test_unit_control_guidance_uses_single_branch() -> None:
    assert build_control_cfg_postprocess(control_guidance=1.0) is None


@pytest.mark.parametrize("text_guidance", [1.0, 3.0])
@pytest.mark.parametrize("control_guidance", [0.5, 1.5, 2.0])
def test_control_cfg_increment_is_independent_of_text_guidance(text_guidance: float, control_guidance: float) -> None:
    control = torch.zeros(2, 3)
    target = torch.ones(4, 3)
    data = GenerationDataClean(
        batch_size=1,
        is_image_batch=False,
        x0_tokens_vision=[control, target],
        num_vision_items_per_sample=[2],
    )
    full = torch.arange(control.numel() + target.numel(), dtype=torch.float32) + 10.0
    no_control = torch.arange(target.numel(), dtype=torch.float32) + 1.0
    unconditional = torch.full_like(full, -2.0)
    noise = torch.zeros_like(full)
    timestep = torch.tensor([[0.5]])
    model = SimpleNamespace(_get_velocity=Mock(return_value=[no_control]))
    builder = build_control_cfg_postprocess(control_guidance=control_guidance)
    assert builder is not None
    postprocess = builder(
        model=model,
        cond_tokens=[[7]],
        sequence_plans=[SequencePlan(has_text=True, has_vision=True)],
        gen_data_clean=data,
    )
    assert postprocess is not None

    adjusted = postprocess([full], [noise], timestep, text_guidance)[0]
    actual = unconditional + text_guidance * (adjusted - unconditional)
    expected = unconditional + text_guidance * (full - unconditional)
    expected[control.numel() :] += (control_guidance - 1.0) * (full[control.numel() :] - no_control)
    torch.testing.assert_close(actual, expected)

    with pytest.raises(ValueError, match="text guidance scale 0.0"):
        postprocess([full], [noise], timestep, 0.0)


class _CacheLanguageModel(torch.nn.Module):
    """Distinct, constant residuals make cross-branch cache reuse observable."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, pack: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        self.calls += 1
        return {**pack, "full_only_seq": pack["full_only_seq"] + pack["delta"]}, {}


class _CacheModel:
    def __init__(self) -> None:
        self.net = SimpleNamespace(language_model=_CacheLanguageModel())

    def generate_samples_from_batch(self, **kwargs: Any) -> None:
        pass

    def denoise(self, *, data_batch_packed: Any) -> torch.Tensor:
        output, _ = self.net.language_model(data_batch_packed.pack)
        return output["full_only_seq"].flatten()

    def _get_velocity(
        self,
        *,
        noise_x: list[torch.Tensor],
        timestep: torch.Tensor,
        gen_data_clean: GenerationDataClean,
        text_tokens: list[list[int]],
        **kwargs: Any,
    ) -> list[torch.Tensor]:
        assert gen_data_clean.x0_tokens_vision is not None
        latents = gen_data_clean.x0_tokens_vision
        conditioned = len(latents) > 1
        delta = 9.0 if text_tokens == [[0]] else (2.0 if conditioned else 5.0)
        batch = SimpleNamespace(
            vision=SimpleNamespace(
                tokens=latents,
                token_shapes=[tuple(latent.shape[-3:]) for latent in latents],
                timesteps=timestep.flatten(),
                condition_mask=[
                    torch.full((1, 1, 1), conditioned and index == 0, dtype=torch.bool) for index in range(len(latents))
                ],
            ),
            pack={
                "causal_seq": torch.zeros(1, 1),
                "full_only_seq": noise_x[0].reshape(-1, 1),
                "delta": delta,
            },
        )
        return [self.denoise(data_batch_packed=batch)]


@pytest.mark.parametrize("text_cfg", [False, True])
def test_control_cfg_cache_reuses_separate_branches_and_refreshes_at_interval_boundaries(text_cfg: bool) -> None:
    """Exercise the Transfer callback through real SeaCache hooks and sampler branch registration."""
    control = torch.ones(2, 1, 2, 2)
    target = torch.zeros_like(control)
    data = GenerationDataClean(
        batch_size=1,
        is_image_batch=False,
        x0_tokens_vision=[control, target],
        num_vision_items_per_sample=[2],
    )
    plans = [SequencePlan(has_text=True, has_vision=True)]
    model = _CacheModel()
    cache = DiffusionCache(
        num_steps=6,
        config={"ret_steps": 0, "cutoff_from_end": 0, "max_consecutive_cached": 0},
    )
    cache.install(SimpleNamespace(model=model))
    builder = build_control_cfg_postprocess(control_guidance=2.0, control_guidance_interval=[0.25, 0.75])
    assert builder is not None
    postprocess = builder(model=model, cond_tokens=[[7]], sequence_plans=plans, gen_data_clean=data)
    assert isinstance(postprocess, VelocityPostprocess)

    for step, value in enumerate((0.9, 0.8, 0.6, 0.4, 0.2, 0.1)):
        timestep = torch.tensor([[value]])
        # Fully conditioned control changes must not influence SEA's target indicator.
        control.fill_(float(100 * (step + 1)))
        noise = [torch.cat([control.flatten(), target.flatten()])]
        branches = postprocess.cfg_branches(timestep)
        pathways = _velocity_pathways(text_cfg, cfg_branches=branches)
        cache.begin_step(step, pathways)
        before = model.net.language_model.calls
        full = model._get_velocity(
            noise_x=noise,
            timestep=timestep,
            gen_data_clean=data,
            text_tokens=[[7]],
        )
        text_guidance = 3.0 if text_cfg else 1.0
        mixed = postprocess(full, noise, timestep, text_guidance)
        expected = 2.0 + (2.0 - 5.0) / text_guidance if branches else 2.0
        torch.testing.assert_close(mixed[0][control.numel() :], torch.full((target.numel(),), expected))
        if text_cfg:
            negative = model._get_velocity(
                noise_x=noise,
                timestep=timestep,
                gen_data_clean=data,
                text_tokens=[[0]],
            )
            torch.testing.assert_close(negative[0][control.numel() :], torch.full((target.numel(),), 9.0))
            prediction = negative[0] + text_guidance * (mixed[0] - negative[0])
            # Text-CFG baseline is 9 + 3 * (2 - 9) = -12; control adds 2 - 5 = -3.
            torch.testing.assert_close(
                prediction[control.numel() :], torch.full((target.numel(),), -15.0 if branches else -12.0)
            )
        # First step and both interval boundaries refresh every branch; the next step reuses each one.
        assert model.net.language_model.calls - before == (len(pathways) if step % 2 == 0 else 0)
        assert set(cache._pathways) == set(pathways)
        assert not cache._pending_pathways
