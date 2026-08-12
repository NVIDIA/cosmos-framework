# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Tests for standalone Parakeet raw-audio preprocessing and feature extraction."""

import os

import numpy as np
import pytest
import torch
from transformers import ParakeetEncoderConfig

from cosmos_framework.model.generator.reasoner.parakeet.configuration_parakeet import ParakeetAudioConfig
from cosmos_framework.model.generator.reasoner.parakeet.parakeet import ParakeetAudioModel
from cosmos_framework.model.generator.utils.safetensors_loader import load_vlm_model
from cosmos_framework.data.generator.processors.audio_utils import (
    AUDIO_END_TOKEN,
    AUDIO_PAD_TOKEN,
    AUDIO_START_TOKEN,
    add_audio_special_tokens,
    build_audio_timeline_text,
    expand_audio_placeholders_in_text,
    get_audio_only_timestamps,
    get_audio_segment_token_lengths,
    get_qwen_video_timestamps,
    splice_audio_segments_after_video_chunks,
)
from cosmos_framework.data.generator.processors.parakeet_audio_processor import ParakeetAudioProcessor


class _FakeFeatureExtractor:
    feature_size: int = 128
    hop_length: int = 160
    sampling_rate: int = 16_000

    def __call__(
        self,
        raw_speech: list[np.ndarray],
        *,
        sampling_rate: int,
        return_tensors: str,
        return_attention_mask: bool,
        padding: str,
    ) -> dict[str, torch.Tensor]:
        assert sampling_rate == self.sampling_rate
        assert return_tensors == "pt"
        assert return_attention_mask
        assert padding == "longest"
        feature_lengths = torch.tensor([waveform.shape[0] // self.hop_length for waveform in raw_speech])
        max_frames = int(feature_lengths.max()) + 1
        frame_indexes = torch.arange(max_frames)
        return {
            "input_features": torch.zeros(len(raw_speech), max_frames, self.feature_size),
            "attention_mask": frame_indexes.unsqueeze(0) < feature_lengths.unsqueeze(1),
        }


class _FakeTokenizer:
    unk_token_id: int = 0

    def __init__(self, vocabulary: dict[str, int] | None = None) -> None:
        self.vocabulary = vocabulary or {
            AUDIO_START_TOKEN: 41,
            AUDIO_PAD_TOKEN: 73,
            AUDIO_END_TOKEN: 109,
        }

    def get_vocab(self) -> dict[str, int]:
        return self.vocabulary.copy()

    def convert_tokens_to_ids(self, tokens: str) -> int:
        return self.vocabulary.get(tokens, self.unk_token_id)

    def add_tokens(self, new_tokens: list[str]) -> int:
        for token in new_tokens:
            self.vocabulary[token] = max(self.vocabulary.values(), default=-1) + 1
        return len(new_tokens)


@pytest.mark.L0
@pytest.mark.CPU
def test_processor_returns_qwen_style_audio_fields_and_exact_lengths() -> None:
    processor = ParakeetAudioProcessor(feature_extractor=_FakeFeatureExtractor())
    audios = [
        torch.linspace(-0.25, 0.25, 16_000),
        np.linspace(-0.5, 0.5, 8_001, dtype=np.float32),
    ]

    outputs = processor(audios, sampling_rate=16_000)

    assert set(outputs) == {"audio_features", "audio_feature_lengths", "audio_token_lengths"}
    assert outputs["audio_features"].shape == (2, 101, 128)
    assert outputs["audio_features"].dtype == torch.float32
    assert torch.equal(outputs["audio_feature_lengths"], torch.tensor([100, 50]))
    assert torch.equal(outputs["audio_token_lengths"], torch.tensor([13, 7]))


@pytest.mark.L0
@pytest.mark.CPU
def test_processor_uses_transformers_feature_extractor_without_network() -> None:
    processor = ParakeetAudioProcessor()

    outputs = processor(np.zeros(320, dtype=np.float32), sampling_rate=16_000)

    assert outputs["audio_features"].shape == (1, 3, 128)
    assert torch.equal(outputs["audio_feature_lengths"], torch.tensor([2]))
    assert torch.equal(outputs["audio_token_lengths"], torch.tensor([1]))


@pytest.mark.L0
@pytest.mark.CPU
def test_raw_waveform_processor_outputs_feed_tiny_audio_model() -> None:
    processor = ParakeetAudioProcessor()
    encoder_config = ParakeetEncoderConfig(
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=32,
        hidden_act="silu",
        attention_bias=False,
        conv_kernel_size=3,
        convolution_bias=False,
        subsampling_factor=8,
        subsampling_conv_channels=4,
        num_mel_bins=processor.num_mel_bins,
        subsampling_conv_kernel_size=3,
        subsampling_conv_stride=2,
        dropout=0.0,
        dropout_positions=0.0,
        layerdrop=0.0,
        activation_dropout=0.0,
        attention_dropout=0.0,
        max_position_embeddings=128,
        scale_input=False,
    )
    model = ParakeetAudioModel(
        ParakeetAudioConfig(
            encoder_config=encoder_config,
            projection_hidden_size=32,
            out_hidden_size=24,
        )
    ).eval()
    processor_outputs = processor(
        [np.zeros(1_280, dtype=np.float32), torch.zeros(1_600)],
        sampling_rate=16_000,
    )

    with torch.no_grad():
        audio_embeddings, output_lengths = model(**processor_outputs)

    assert audio_embeddings.shape == (2, 2, 24)
    assert torch.equal(processor_outputs["audio_feature_lengths"], torch.tensor([8, 10]))
    assert torch.equal(processor_outputs["audio_token_lengths"], torch.tensor([2, 2]))
    assert [
        len(processor.get_token_timestamps(int(feature_length)))
        for feature_length in processor_outputs["audio_feature_lengths"]
    ] == processor_outputs["audio_token_lengths"].tolist()
    assert torch.equal(
        output_lengths.to(dtype=processor_outputs["audio_token_lengths"].dtype),
        processor_outputs["audio_token_lengths"],
    )


@pytest.mark.L0
@pytest.mark.CPU
def test_processor_rejects_invalid_waveforms() -> None:
    processor = ParakeetAudioProcessor(feature_extractor=_FakeFeatureExtractor())
    invalid_inputs = [
        (torch.zeros(8_000), 8_000, ValueError, "does not resample"),
        (np.zeros((2, 320), dtype=np.float32), 16_000, ValueError, "one-dimensional mono waveform"),
        (np.empty(0, dtype=np.float32), 16_000, ValueError, "at least one sample"),
        (np.zeros(319, dtype=np.float32), 16_000, ValueError, "at least 320 samples"),
        (torch.tensor([0.0, float("nan")]), 16_000, ValueError, "NaN or infinite"),
        (np.ones(320, dtype=np.int16), 16_000, TypeError, "floating-point dtype"),
    ]

    for audio, sampling_rate, error_type, match in invalid_inputs:
        with pytest.raises(error_type, match=match):
            processor(audio, sampling_rate=sampling_rate)


@pytest.mark.L0
@pytest.mark.CPU
def test_add_audio_special_tokens_registers_dedicated_tokens_idempotently() -> None:
    tokenizer = _FakeTokenizer({"ordinary": 7})

    registered_tokens = add_audio_special_tokens(tokenizer)
    repeated_tokens = add_audio_special_tokens(tokenizer)

    assert registered_tokens.tokens == (AUDIO_START_TOKEN, AUDIO_PAD_TOKEN, AUDIO_END_TOKEN)
    assert registered_tokens.token_ids == (8, 9, 10)
    assert repeated_tokens == registered_tokens
    assert tokenizer.get_vocab() == {
        "ordinary": 7,
        AUDIO_START_TOKEN: 8,
        AUDIO_PAD_TOKEN: 9,
        AUDIO_END_TOKEN: 10,
    }


@pytest.mark.L0
@pytest.mark.CPU
def test_add_audio_special_tokens_reuses_explicit_reserved_slots_without_expanding_vocab() -> None:
    reserved_tokens = ("<RESERVED_AUDIO_START>", "<RESERVED_AUDIO_PAD>", "<RESERVED_AUDIO_END>")
    reserved_token_ids = (101, 203, 307)
    tokenizer = _FakeTokenizer(
        {
            "ordinary": 7,
            reserved_tokens[0]: reserved_token_ids[0],
            reserved_tokens[1]: reserved_token_ids[1],
            reserved_tokens[2]: reserved_token_ids[2],
        }
    )
    vocabulary_before = tokenizer.get_vocab()

    registered_tokens = add_audio_special_tokens(tokenizer, reserved_tokens=reserved_tokens)

    assert registered_tokens.tokens == reserved_tokens
    assert registered_tokens.token_ids == reserved_token_ids
    assert tokenizer.get_vocab() == vocabulary_before


@pytest.mark.L0
@pytest.mark.CPU
def test_expand_audio_placeholders_in_text_preserves_clip_order() -> None:
    text = f"first={AUDIO_PAD_TOKEN}; second={AUDIO_PAD_TOKEN}"

    expanded = expand_audio_placeholders_in_text(
        text,
        torch.tensor([1, 3]),
        audio_timestamps=[[0.125], [0.125]],
    )

    assert expanded == (
        f"first={AUDIO_START_TOKEN}<0.1 seconds>{AUDIO_PAD_TOKEN}{AUDIO_END_TOKEN}; "
        f"second={AUDIO_START_TOKEN}<0.1 seconds>{AUDIO_PAD_TOKEN * 3}{AUDIO_END_TOKEN}"
    )


@pytest.mark.L0
@pytest.mark.CPU
def test_expand_audio_placeholders_in_text_requires_one_marker_per_clip() -> None:
    with pytest.raises(ValueError, match="placeholder count"):
        expand_audio_placeholders_in_text("no marker", [2], audio_timestamps=[[0.125]])


@pytest.mark.L0
@pytest.mark.CPU
def test_audio_timeline_uses_qwen_timestamps_and_keeps_the_tail() -> None:
    timestamps = get_qwen_video_timestamps(num_frames=8, fps=4.0, temporal_patch_size=2)

    timeline = build_audio_timeline_text(26, timestamps)

    assert timestamps == [0.125, 0.625, 1.125, 1.625]
    assert timeline == (
        f"{AUDIO_START_TOKEN}<0.1 seconds>{AUDIO_PAD_TOKEN * 5}"
        f"<0.6 seconds>{AUDIO_PAD_TOKEN * 6}"
        f"<1.1 seconds>{AUDIO_PAD_TOKEN * 7}"
        f"<1.6 seconds>{AUDIO_PAD_TOKEN * 8}{AUDIO_END_TOKEN}"
    )
    assert timeline.count(AUDIO_PAD_TOKEN) == 26

    segment_lengths = [5, 6, 7, 8]
    cursor = 0
    for timestamp_index, segment_length in enumerate(segment_lengths):
        for token_index in range(cursor, cursor + segment_length):
            token_time = token_index * 0.08
            closest_timestamp_index = min(
                range(len(timestamps)),
                key=lambda index: abs(token_time - timestamps[index]),
            )
            assert closest_timestamp_index == timestamp_index
        cursor += segment_length

    assert get_qwen_video_timestamps(num_frames=3, fps=4.0, temporal_patch_size=2) == [0.125, 0.5]
    assert get_qwen_video_timestamps(num_frames=4, fps=4.0, temporal_patch_size=1) == [0.0, 0.25, 0.5, 0.75]
    assert get_audio_only_timestamps(num_audio_tokens=10, temporal_patch_size=2) == [0.125, 0.5]
    assert build_audio_timeline_text(4, [0.16]) == (
        f"{AUDIO_START_TOKEN}<0.2 seconds>{AUDIO_PAD_TOKEN * 4}{AUDIO_END_TOKEN}"
    )

    parakeet_token_timestamps = ParakeetAudioProcessor.get_token_timestamps(320)
    assert len(parakeet_token_timestamps) == 41
    assert get_audio_segment_token_lengths(
        41,
        get_qwen_video_timestamps(num_frames=20, fps=6.25, temporal_patch_size=2),
        audio_token_timestamps=parakeet_token_timestamps,
    ) == [4, 4, 4, 4, 4, 4, 4, 4, 4, 5]


@pytest.mark.L0
@pytest.mark.CPU
def test_audio_segments_splice_after_their_ordered_video_chunks() -> None:
    timestamp_id = 10
    vision_start_id = 11
    video_pad_id = 12
    image_pad_id = 13
    vision_end_id = 14
    audio_token_ids = (20, 21, 22)
    native_input_ids = torch.tensor(
        [
            1,
            vision_start_id,
            image_pad_id,
            vision_end_id,
            timestamp_id,
            vision_start_id,
            video_pad_id,
            vision_end_id,
            timestamp_id,
            vision_start_id,
            video_pad_id,
            vision_end_id,
            timestamp_id,
            vision_start_id,
            video_pad_id,
            vision_end_id,
            2,
        ]
    )

    output = splice_audio_segments_after_video_chunks(
        native_input_ids,
        [1, 2],
        [None, [2, 0]],
        audio_token_ids=audio_token_ids,
        video_pad_token_id=video_pad_id,
        vision_end_token_id=vision_end_id,
    )

    assert output.tolist() == [
        1,
        vision_start_id,
        image_pad_id,
        vision_end_id,
        timestamp_id,
        vision_start_id,
        video_pad_id,
        vision_end_id,
        timestamp_id,
        vision_start_id,
        video_pad_id,
        vision_end_id,
        audio_token_ids[0],
        audio_token_ids[1],
        audio_token_ids[1],
        audio_token_ids[2],
        timestamp_id,
        vision_start_id,
        video_pad_id,
        vision_end_id,
        audio_token_ids[0],
        audio_token_ids[2],
        2,
    ]
    restored = output.tolist()
    while audio_token_ids[0] in restored:
        start = restored.index(audio_token_ids[0])
        end = restored.index(audio_token_ids[2], start)
        del restored[start : end + 1]
    assert restored == native_input_ids.tolist()
    assert get_audio_segment_token_lengths(26, [0.125, 0.625, 1.125, 1.625]) == [5, 6, 7, 8]
    assert get_audio_segment_token_lengths(4, []) == [4]
    assert build_audio_timeline_text(4, []) == f"{AUDIO_START_TOKEN}{AUDIO_PAD_TOKEN * 4}{AUDIO_END_TOKEN}"

    with pytest.raises(ValueError):
        splice_audio_segments_after_video_chunks(
            native_input_ids,
            [1, 2],
            [None, [2]],
            audio_token_ids=audio_token_ids,
            video_pad_token_id=video_pad_id,
            vision_end_token_id=vision_end_id,
        )


@pytest.mark.L1
@pytest.mark.GPU
def test_real_encoder_checkpoint_extracts_features_from_raw_audio() -> None:
    """Smoke-test the complete standalone frontend against the real artifact."""
    checkpoint_path = os.environ.get("PARAKEET_ENCODER_TEST_CHECKPOINT")
    if checkpoint_path is None:
        pytest.skip("PARAKEET_ENCODER_TEST_CHECKPOINT is not configured")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the real Parakeet encoder smoke test")

    credentials_path = os.environ.get("PARAKEET_ENCODER_TEST_CREDENTIALS", "")
    processor = ParakeetAudioProcessor()
    processor_outputs = processor(torch.linspace(-0.1, 0.1, 16_000), sampling_rate=16_000)
    model = ParakeetAudioModel(
        ParakeetAudioConfig(
            projection_hidden_size=64,
            out_hidden_size=32,
        )
    ).to(device="cuda", dtype=torch.bfloat16)
    loaded_names = load_vlm_model(
        model=model.encoder,
        checkpoint_path=checkpoint_path,
        credential_path=credentials_path or None,
        parallel_dims=None,
    )
    model.eval()
    cuda_inputs = {name: tensor.cuda() for name, tensor in processor_outputs.items()}
    cuda_inputs["audio_features"] = cuda_inputs["audio_features"].to(torch.bfloat16)

    with torch.no_grad():
        features, output_lengths = model(**cuda_inputs)

    assert len(loaded_names) == len(model.encoder.state_dict())
    assert features.shape == (1, int(output_lengths.max()), 32)
    assert torch.equal(output_lengths, cuda_inputs["audio_token_lengths"].to(output_lengths.dtype))
    assert torch.isfinite(features).all()
