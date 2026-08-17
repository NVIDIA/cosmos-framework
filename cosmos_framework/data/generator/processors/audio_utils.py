# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Shared audio-understanding processor utilities."""

import math
from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import torch

AUDIO_START_TOKEN: str = "<audio_start>"
AUDIO_PAD_TOKEN: str = "<audio_pad>"
AUDIO_END_TOKEN: str = "<audio_end>"
DEFAULT_REASONER_VIDEO_FPS: float = 4.0
EDGE_REASONER_MODEL_NAME: str = "nvidia/Cosmos3-Edge-Reasoner"
EDGE_RESERVED_AUDIO_TOKENS: tuple[str, str, str] = (
    "<SPECIAL_23>",
    "<SPECIAL_24>",
    "<SPECIAL_25>",
)
EDGE_RESERVED_AUDIO_TOKEN_IDS: tuple[int, int, int] = (23, 24, 25)
_DEFAULT_AUDIO_TOKEN_STRIDE_SECONDS: float = 0.08


@dataclass(frozen=True, slots=True)
class AudioSpecialTokens:
    """Effective tokenizer strings and IDs selected for audio placeholders."""

    tokens: tuple[str, str, str]
    token_ids: tuple[int, int, int]


class _Tokenizer(Protocol):
    """Tokenizer operations required to resolve audio special tokens."""

    unk_token_id: int | None

    def convert_tokens_to_ids(self, tokens: str) -> int | None: ...

    def get_vocab(self) -> dict[str, int]: ...

    def add_tokens(self, new_tokens: list[str]) -> int: ...


def _resolve_token_id(tokenizer: _Tokenizer, token: str) -> int:
    """Resolve one registered special token without relying on a fixed ID."""
    vocabulary = tokenizer.get_vocab()
    if token not in vocabulary:
        raise ValueError(f"Tokenizer does not contain the required audio token {token!r}")

    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None or isinstance(token_id, bool) or not isinstance(token_id, int):
        raise ValueError(f"Tokenizer did not resolve {token!r} to a single integer token ID")
    if tokenizer.unk_token_id is not None and token_id == tokenizer.unk_token_id:
        raise ValueError(f"Tokenizer resolved the required audio token {token!r} to its unknown token")
    if vocabulary[token] != token_id:
        raise ValueError(f"Tokenizer returned inconsistent IDs for the required audio token {token!r}")
    return token_id


def add_audio_special_tokens(
    tokenizer: _Tokenizer,
    *,
    audio_start_token: str = AUDIO_START_TOKEN,
    audio_pad_token: str = AUDIO_PAD_TOKEN,
    audio_end_token: str = AUDIO_END_TOKEN,
    reserved_tokens: tuple[str, str, str] | None = None,
) -> AudioSpecialTokens:
    """Register dedicated audio tokens using the existing tokenizer style.

    This only extends the tokenizer's token-to-ID table. It deliberately does
    not resize model embeddings; callers must verify that the returned IDs fit
    inside the reasoner's preallocated ``vocab_size``. Callers may explicitly
    provide three existing ``reserved_tokens`` for a tokenizer that has no
    spare IDs. Target-specific setup must select reserved entries explicitly
    and use the returned effective strings and IDs for prompt preprocessing.
    """
    selected_tokens = (audio_start_token, audio_pad_token, audio_end_token)
    if any(not isinstance(token, str) or not token for token in selected_tokens):
        raise ValueError("Audio start, pad, and end tokens must be non-empty strings")
    if len(set(selected_tokens)) != 3:
        raise ValueError("Audio start, pad, and end token strings must be distinct")

    vocabulary = tokenizer.get_vocab()
    default_tokens = (AUDIO_START_TOKEN, AUDIO_PAD_TOKEN, AUDIO_END_TOKEN)
    if reserved_tokens is not None:
        if len(reserved_tokens) != 3 or any(not isinstance(token, str) or not token for token in reserved_tokens):
            raise ValueError("Reserved audio tokens must contain three non-empty strings")
        if len(set(reserved_tokens)) != 3:
            raise ValueError("Reserved audio token strings must be distinct")
        if (
            selected_tokens == default_tokens
            and not any(token in vocabulary for token in selected_tokens)
            and all(token in vocabulary for token in reserved_tokens)
        ):
            selected_tokens = reserved_tokens

    missing_tokens = [token for token in selected_tokens if token not in vocabulary]
    if missing_tokens:
        num_added = tokenizer.add_tokens(missing_tokens)
        if num_added != len(missing_tokens):
            raise RuntimeError(
                "Tokenizer did not register every missing audio token: "
                f"requested={len(missing_tokens)}, added={num_added}"
            )

    token_ids = (
        _resolve_token_id(tokenizer, selected_tokens[0]),
        _resolve_token_id(tokenizer, selected_tokens[1]),
        _resolve_token_id(tokenizer, selected_tokens[2]),
    )
    if len(set(token_ids)) != 3:
        raise ValueError(f"Audio start, pad, and end tokens must have distinct token IDs, got {token_ids}")
    return AudioSpecialTokens(tokens=selected_tokens, token_ids=token_ids)


def add_reasoner_audio_special_tokens(
    tokenizer: _Tokenizer,
    *,
    model_name_or_path: str,
    audio_start_token: str = AUDIO_START_TOKEN,
    audio_pad_token: str = AUDIO_PAD_TOKEN,
    audio_end_token: str = AUDIO_END_TOKEN,
) -> AudioSpecialTokens:
    """Resolve the shared Nano/Super versus Edge audio-token policy.

    Qwen-based Reasoners have unused rows in their preallocated embedding
    tables, so their tokenizers can append the dedicated ``<audio_*>`` strings.
    Edge's tokenizer fills its 131072-row table and therefore reuses three
    existing reserved tokens. Both the data processor and the live Reasoner
    model call this helper so placeholder strings and IDs cannot drift.
    """
    if not isinstance(model_name_or_path, str):
        raise TypeError(f"model_name_or_path must be a string, got {type(model_name_or_path).__name__}")
    reserved_tokens = EDGE_RESERVED_AUDIO_TOKENS if EDGE_REASONER_MODEL_NAME in model_name_or_path else None

    special_tokens = add_audio_special_tokens(
        tokenizer,
        audio_start_token=audio_start_token,
        audio_pad_token=audio_pad_token,
        audio_end_token=audio_end_token,
        reserved_tokens=reserved_tokens,
    )
    if (
        reserved_tokens == EDGE_RESERVED_AUDIO_TOKENS
        and special_tokens.tokens == EDGE_RESERVED_AUDIO_TOKENS
        and special_tokens.token_ids != EDGE_RESERVED_AUDIO_TOKEN_IDS
    ):
        raise ValueError(
            "Edge reserved audio tokens no longer resolve to the expected IDs: "
            f"{special_tokens.token_ids} != {EDGE_RESERVED_AUDIO_TOKEN_IDS}"
        )
    return special_tokens


def _normalize_audio_token_lengths(audio_token_lengths: Sequence[int] | torch.Tensor) -> list[int]:
    """Return validated per-clip encoder output lengths."""
    if isinstance(audio_token_lengths, torch.Tensor):
        if audio_token_lengths.ndim != 1:
            raise ValueError(
                f"audio_token_lengths must be one-dimensional, got shape {tuple(audio_token_lengths.shape)}"
            )
        if (
            audio_token_lengths.dtype == torch.bool
            or audio_token_lengths.dtype.is_floating_point
            or audio_token_lengths.dtype.is_complex
        ):
            raise TypeError(f"audio_token_lengths must use an integer dtype, got {audio_token_lengths.dtype}")
        lengths = audio_token_lengths.tolist()
    else:
        lengths = list(audio_token_lengths)

    for length in lengths:
        if isinstance(length, bool) or not isinstance(length, int):
            raise TypeError(f"Every audio token length must be an integer, got {type(length).__name__}")
        if length < 1:
            raise ValueError(f"Every audio clip must produce at least one token, got {length}")
    return lengths


def expand_audio_placeholders_in_text(
    text: str,
    audio_token_lengths: Sequence[int] | torch.Tensor,
    *,
    audio_timestamps: Sequence[Sequence[float]],
    audio_token_timestamps: Sequence[Sequence[float]] | None = None,
    audio_start_token: str = AUDIO_START_TOKEN,
    audio_pad_token: str = AUDIO_PAD_TOKEN,
    audio_end_token: str = AUDIO_END_TOKEN,
) -> str:
    """Expand one configured audio placeholder per clip inside prompt text.

    Callers first place one ``audio_pad_token`` in the user message for each
    audio clip, then this helper replaces those markers in order with one
    continuous dynamic audio block whose pad runs are prefixed by the supplied
    timestamp text.
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be a string, got {type(text).__name__}")
    lengths = _normalize_audio_token_lengths(audio_token_lengths)
    selected_tokens = (audio_start_token, audio_pad_token, audio_end_token)
    if any(not isinstance(token, str) or not token for token in selected_tokens):
        raise ValueError("Audio start, pad, and end tokens must be non-empty strings")
    if len(set(selected_tokens)) != 3:
        raise ValueError("Audio start, pad, and end token strings must be distinct")
    if len(audio_timestamps) != len(lengths):
        raise ValueError(
            "Audio timestamp groups must match the number of clips: "
            f"timestamps={len(audio_timestamps)}, clips={len(lengths)}"
        )
    text_parts = text.split(audio_pad_token)
    num_placeholders = len(text_parts) - 1
    if num_placeholders != len(lengths):
        raise ValueError(
            "Prompt audio placeholder count must match the number of clips: "
            f"placeholders={num_placeholders}, clips={len(lengths)}"
        )

    token_timestamp_groups: Sequence[Sequence[float] | None]
    if audio_token_timestamps is None:
        token_timestamp_groups = [None] * len(lengths)
    else:
        token_timestamp_groups = audio_token_timestamps

    expanded_parts = [text_parts[0]]
    for num_audio_tokens, timestamps, token_timestamps, trailing_text in zip(
        lengths,
        audio_timestamps,
        token_timestamp_groups,
        text_parts[1:],
        strict=True,
    ):
        expanded_parts.extend(
            (
                build_audio_timeline_text(
                    num_audio_tokens,
                    timestamps,
                    audio_start_token=audio_start_token,
                    audio_pad_token=audio_pad_token,
                    audio_end_token=audio_end_token,
                    audio_token_timestamps=token_timestamps,
                ),
                trailing_text,
            )
        )
    return "".join(expanded_parts)


def get_qwen_video_timestamps(*, num_frames: int, fps: float, temporal_patch_size: int) -> list[float]:
    """Mirror Transformers Qwen3-VL's ``_calculate_timestamps`` formula."""
    if isinstance(num_frames, bool) or not isinstance(num_frames, int) or num_frames < 1:
        raise ValueError(f"num_frames must be a positive integer, got {num_frames!r}")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(float(fps)) or fps <= 0:
        raise ValueError(f"fps must be a positive finite number, got {fps!r}")
    if isinstance(temporal_patch_size, bool) or not isinstance(temporal_patch_size, int) or temporal_patch_size < 1:
        raise ValueError(f"temporal_patch_size must be a positive integer, got {temporal_patch_size!r}")

    frame_indices = list(range(num_frames))
    remainder = len(frame_indices) % temporal_patch_size
    if remainder:
        frame_indices.extend([frame_indices[-1]] * (temporal_patch_size - remainder))
    frame_timestamps = [frame_index / float(fps) for frame_index in frame_indices]
    return [
        (frame_timestamps[index] + frame_timestamps[index + temporal_patch_size - 1]) / 2
        for index in range(0, len(frame_timestamps), temporal_patch_size)
    ]


def get_audio_only_timestamps(
    *,
    num_audio_tokens: int,
    temporal_patch_size: int,
    fps: float = DEFAULT_REASONER_VIDEO_FPS,
    audio_token_timestamps: Sequence[float] | None = None,
) -> list[float]:
    """Build the virtual Qwen video clock used by an audio-only prompt."""
    if isinstance(num_audio_tokens, bool) or not isinstance(num_audio_tokens, int) or num_audio_tokens < 1:
        raise ValueError(f"num_audio_tokens must be a positive integer, got {num_audio_tokens!r}")

    if audio_token_timestamps is None:
        last_audio_token_time = (num_audio_tokens - 1) * _DEFAULT_AUDIO_TOKEN_STRIDE_SECONDS
    else:
        last_audio_token_time = float(audio_token_timestamps[-1])
    num_frames = max(temporal_patch_size, math.ceil(last_audio_token_time * fps))
    return get_qwen_video_timestamps(
        num_frames=num_frames,
        fps=fps,
        temporal_patch_size=temporal_patch_size,
    )


def get_audio_segment_token_lengths(
    num_audio_tokens: int,
    timestamps: Sequence[float],
    *,
    audio_token_timestamps: Sequence[float] | None = None,
) -> list[int]:
    """Assign audio tokens to their nearest shared timestamp anchor."""
    if isinstance(num_audio_tokens, bool) or not isinstance(num_audio_tokens, int) or num_audio_tokens < 1:
        raise ValueError(f"num_audio_tokens must be a positive integer, got {num_audio_tokens!r}")

    default_token_timestamps = [
        token_index * _DEFAULT_AUDIO_TOKEN_STRIDE_SECONDS for token_index in range(num_audio_tokens)
    ]
    token_timestamps = default_token_timestamps if audio_token_timestamps is None else list(audio_token_timestamps)
    uses_default_token_grid = token_timestamps == default_token_timestamps
    normalized_timestamps: list[float] = []
    previous_timestamp = -math.inf
    for timestamp in timestamps:
        timestamp = float(timestamp)
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError(f"Audio timestamps must be finite and non-negative, got {timestamp!r}")
        if timestamp <= previous_timestamp:
            raise ValueError("Audio timestamps must be strictly increasing")
        normalized_timestamps.append(timestamp)
        previous_timestamp = timestamp

    if not normalized_timestamps:
        return [num_audio_tokens]

    segment_lengths: list[int] = []
    cursor = 0
    for timestamp_index, timestamp in enumerate(normalized_timestamps):
        if timestamp_index + 1 == len(normalized_timestamps):
            boundary = num_audio_tokens
        else:
            next_timestamp = normalized_timestamps[timestamp_index + 1]
            midpoint = (timestamp + next_timestamp) / 2
            if uses_default_token_grid:
                boundary = min(
                    num_audio_tokens,
                    math.floor(midpoint / _DEFAULT_AUDIO_TOKEN_STRIDE_SECONDS) + 1,
                )
            else:
                boundary = bisect_right(token_timestamps, midpoint)
        segment_lengths.append(boundary - cursor)
        cursor = boundary
    return segment_lengths


def build_audio_timeline_text(
    num_audio_tokens: int,
    timestamps: Sequence[float],
    *,
    audio_start_token: str = AUDIO_START_TOKEN,
    audio_pad_token: str = AUDIO_PAD_TOKEN,
    audio_end_token: str = AUDIO_END_TOKEN,
    audio_token_timestamps: Sequence[float] | None = None,
) -> str:
    """Arrange shared video timestamp text inside one continuous audio block.

    Each audio token is assigned to its nearest timestamp using adjacent
    timestamp midpoints as boundaries, then emitted after that timestamp
    marker to match Qwen video layout. The final timestamp owns the remaining
    tail tokens. Callers may provide backend-native token start times; the
    default preserves the existing 80-ms output stride.
    """
    segment_lengths = get_audio_segment_token_lengths(
        num_audio_tokens,
        timestamps,
        audio_token_timestamps=audio_token_timestamps,
    )
    parts = [audio_start_token]
    if not timestamps:
        parts.append(audio_pad_token * num_audio_tokens)
    else:
        for timestamp, segment_length in zip(timestamps, segment_lengths, strict=True):
            parts.append(f"<{float(timestamp):.1f} seconds>")
            parts.append(audio_pad_token * segment_length)

    parts.append(audio_end_token)
    return "".join(parts)


def splice_audio_segments_after_video_chunks(
    input_ids: torch.Tensor,
    video_chunk_counts: Sequence[int] | torch.Tensor,
    audio_segment_lengths_by_video: Sequence[Sequence[int] | None],
    *,
    audio_token_ids: tuple[int, int, int],
    video_pad_token_id: int,
    vision_end_token_id: int,
) -> torch.Tensor:
    """Insert ordered audio segments after their native VLM video chunks."""
    chunk_end_positions = (
        torch.nonzero(
            (input_ids[1:] == vision_end_token_id) & (input_ids[:-1] == video_pad_token_id),
            as_tuple=False,
        ).flatten()
        + 1
    )
    chunk_counts = (
        [int(count) for count in video_chunk_counts.tolist()]
        if isinstance(video_chunk_counts, torch.Tensor)
        else list(video_chunk_counts)
    )
    chunk_ends_by_video = torch.split(chunk_end_positions, chunk_counts)
    audio_start_token_id, audio_pad_token_id, audio_end_token_id = audio_token_ids
    insertions: list[tuple[int, torch.Tensor]] = []
    for chunk_ends, segment_lengths in zip(
        chunk_ends_by_video,
        audio_segment_lengths_by_video,
        strict=True,
    ):
        if segment_lengths is None:
            continue
        for chunk_end, segment_length in zip(chunk_ends.tolist(), segment_lengths, strict=True):
            audio_span = input_ids.new_tensor(
                [audio_start_token_id, *([audio_pad_token_id] * segment_length), audio_end_token_id]
            )
            insertions.append((chunk_end, audio_span))

    if not insertions:
        return input_ids

    parts: list[torch.Tensor] = []
    cursor = 0
    for chunk_end, audio_span in insertions:
        parts.extend((input_ids[cursor : chunk_end + 1], audio_span))
        cursor = chunk_end + 1
    parts.append(input_ids[cursor:])
    return torch.cat(parts)


def collate_audio_processor_outputs(
    audio_features: Sequence[torch.Tensor | None],
    audio_feature_lengths: Sequence[torch.Tensor | None],
    audio_token_lengths: Sequence[torch.Tensor | None],
    *,
    num_samples: int,
) -> dict[str, torch.Tensor]:
    """Collate per-sample processor outputs into flat Reasoner side inputs.

    Clips are flattened in text row-major order and padded only along time.
    The returned per-clip lengths preserve valid mel frames and projected token
    counts, while ``audio_sample_token_lengths`` keeps the packed-sample
    boundaries needed to validate audio placeholders.
    """
    if not (len(audio_features) == len(audio_feature_lengths) == len(audio_token_lengths) == num_samples):
        raise ValueError("Collated audio processor fields must match the number of samples")

    feature_clips: list[torch.Tensor] = []
    feature_length_groups: list[torch.Tensor] = []
    token_length_groups: list[torch.Tensor] = []
    sample_token_lengths: list[torch.Tensor | None] = []
    for sample_index, (features, feature_lengths, token_lengths) in enumerate(
        zip(audio_features, audio_feature_lengths, audio_token_lengths, strict=True)
    ):
        sample_values = (features, feature_lengths, token_lengths)
        if all(value is None for value in sample_values):
            sample_token_lengths.append(None)
            continue
        if any(value is None for value in sample_values):
            raise ValueError(
                "audio_features, audio_feature_lengths, and audio_token_lengths must be present together; "
                f"sample {sample_index} is incomplete"
            )
        assert features is not None and feature_lengths is not None and token_lengths is not None
        if features.ndim != 3:
            raise ValueError(
                f"audio_features for sample {sample_index} must have shape [clips, time, mel], "
                f"got {tuple(features.shape)}"
            )
        num_clips = features.shape[0]
        if feature_lengths.shape != (num_clips,) or token_lengths.shape != (num_clips,):
            raise ValueError(f"Audio processor fields for sample {sample_index} must describe {num_clips} clips")
        feature_clips.extend(features.unbind(0))
        feature_length_groups.append(feature_lengths)
        token_length_groups.append(token_lengths)
        sample_token_lengths.append(token_lengths.sum())

    flattened_feature_lengths = torch.cat(feature_length_groups)
    flattened_token_lengths = torch.cat(token_length_groups)
    reference = flattened_token_lengths
    return {
        "audio_features": torch.nn.utils.rnn.pad_sequence(feature_clips, batch_first=True),
        "audio_feature_lengths": flattened_feature_lengths,
        "audio_token_lengths": flattened_token_lengths,
        "audio_sample_token_lengths": torch.stack(
            [
                reference.new_zeros(())
                if sample_token_length is None
                else sample_token_length.to(device=reference.device, dtype=reference.dtype)
                for sample_token_length in sample_token_lengths
            ]
        ),
    }
