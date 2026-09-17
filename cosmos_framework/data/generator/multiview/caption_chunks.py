# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MultiviewCaptionChunk:
    """One caption chunk, optionally kept as the same captions one per camera.

    ``view_prompts`` is used by a checkpoint that tokenizes each view's caption separately.
    It is ``None`` on a chunk read from a single caption file, which describes one camera.
    """

    index: int
    frame_start: int
    frame_end: int
    prompt: str
    view_prompts: list[str] | None = None


# Style-keyed caption files predate the single-caption schema. They are still read, in this fixed
# order, so older MADS bundles keep working; no argument selects a style any more.
LEGACY_CAPTION_KEYS: tuple[str, ...] = ("long", "av_long", "medium", "av_medium", "short", "av_short")


def chunks_from_structured_captions(payload: Mapping[str, Any]) -> list[MultiviewCaptionChunk]:
    """Read the ``caption_structured`` chunk map that MADS Tier-1 bundles and Lance rows carry.

    Shape is ``{"chunk_<start>_<end>": {"caption": str, "start_frame": int, "end_frame": int}}``
    with ``end_frame`` exclusive. Chunks are sorted by frame span rather than trusted in key
    order, matching the training-side ``_build_caption_chunk_map``.
    """
    chunk_map = payload.get("caption_structured")
    if isinstance(chunk_map, str):
        # Lance rows carry the map JSON-encoded inside a string column.
        try:
            chunk_map = json.loads(chunk_map)
        except json.JSONDecodeError as error:
            raise ValueError("caption_structured must be valid JSON when stored as a string.") from error
    if not isinstance(chunk_map, Mapping) or not chunk_map:
        raise ValueError("Multiview caption JSON must contain a non-empty caption_structured object.")

    entries: list[tuple[int, int, str, str]] = []
    for chunk_id, entry in chunk_map.items():
        if not isinstance(entry, Mapping):
            raise ValueError(f"Caption chunk {chunk_id!r} must be an object, got {type(entry).__name__}.")
        frame_start = entry.get("start_frame")
        frame_end = entry.get("end_frame")
        if (
            not isinstance(frame_start, int)
            or not isinstance(frame_end, int)
            or frame_start < 0
            or frame_end <= frame_start
        ):
            raise ValueError(
                f"Caption chunk {chunk_id!r} must have integer 0 <= start_frame < end_frame, "
                f"got start_frame={frame_start!r}, end_frame={frame_end!r}."
            )
        prompt = entry.get("caption")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Caption chunk {chunk_id!r} must contain a non-empty caption string.")
        entries.append((frame_start, frame_end, str(chunk_id), prompt.strip()))

    entries.sort(key=lambda entry: (entry[0], entry[1], entry[2]))
    return [
        MultiviewCaptionChunk(index=index, frame_start=frame_start, frame_end=frame_end, prompt=prompt)
        for index, (frame_start, frame_end, _chunk_id, prompt) in enumerate(entries)
    ]


def chunks_from_style_keyed_captions(payload: list[Any]) -> list[MultiviewCaptionChunk]:
    """Read the legacy schema: ``[{"frame_range": [start, end], "captions": {<style>: text}}]``.

    ``frame_range`` is end-inclusive, unlike ``caption_structured``'s exclusive ``end_frame``,
    hence the ``+ 1`` below.
    """
    chunks: list[MultiviewCaptionChunk] = []
    for index, entry in enumerate(payload):
        if not isinstance(entry, Mapping):
            raise ValueError(f"Caption chunk {index} must be an object, got {type(entry).__name__}.")
        frame_range = entry.get("frame_range")
        if (
            not isinstance(frame_range, list)
            or len(frame_range) != 2
            or not all(isinstance(value, int) for value in frame_range)
        ):
            raise ValueError(f"Caption chunk {index} must contain integer frame_range=[start, end].")
        frame_start, frame_end_inclusive = frame_range
        if frame_start < 0 or frame_end_inclusive < frame_start:
            raise ValueError(f"Caption chunk {index} has invalid frame_range={frame_range}.")

        captions = entry.get("captions")
        if not isinstance(captions, Mapping):
            raise ValueError(f"Caption chunk {index} must contain a captions object.")
        prompt = next(
            (
                captions[key].strip()
                for key in LEGACY_CAPTION_KEYS
                if isinstance(captions.get(key), str) and captions[key].strip()
            ),
            None,
        )
        if prompt is None:
            available_keys = ", ".join(sorted(str(key) for key in captions)) or "none"
            raise ValueError(
                f"Caption chunk {index} has no non-empty caption under any of "
                f"{', '.join(LEGACY_CAPTION_KEYS)} (available keys: {available_keys})."
            )
        chunks.append(
            MultiviewCaptionChunk(
                index=index,
                frame_start=frame_start,
                frame_end=frame_end_inclusive + 1,
                prompt=prompt,
            )
        )
    return chunks


def parse_multiview_caption_chunks(payload: Any, *, source: str = "caption payload") -> list[MultiviewCaptionChunk]:
    """Parse a MADS caption payload in either the structured or legacy style-keyed schema."""
    if isinstance(payload, Mapping):
        chunks = chunks_from_structured_captions(payload)
    elif isinstance(payload, list):
        chunks = chunks_from_style_keyed_captions(payload)
    else:
        raise ValueError(
            "Expected multiview caption JSON to contain a caption_structured object or a list of "
            f"chunks, got {type(payload).__name__}."
        )

    if not chunks:
        raise ValueError(f"Multiview caption data is empty: {source}")
    return chunks


def load_multiview_caption_chunks(caption_path: Path) -> list[MultiviewCaptionChunk]:
    """Read a local MADS caption file in either supported schema."""
    return parse_multiview_caption_chunks(json.loads(caption_path.read_text()), source=str(caption_path))


def caption_chunk_frame_count(caption_chunks: list[MultiviewCaptionChunk], *, tokenizer: Any) -> int:
    """Frames to generate per chunk, derived from the span the captions describe.

    The generation loop shares one frame count across all chunks (it also splits the model output
    by it), so the chunks have to agree on their span. The span is then snapped through the
    tokenizer's own pixel/latent conversion, because only certain frame counts are representable;
    passing a raw span would either fail to encode or silently pad.

    ``tokenizer`` is duck-typed on ``get_latent_num_frames``/``get_pixel_num_frames`` so this stays
    a leaf module. It is the generation vision tokenizer in both runtimes, whose classes differ.
    """
    spans = sorted({chunk.frame_end - chunk.frame_start for chunk in caption_chunks})
    if len(spans) > 1:
        raise ValueError(
            f"Caption chunks cover differing frame counts {spans}; one count has to serve every "
            "chunk. Set multiview.num_video_frames_per_chunk to choose it explicitly."
        )
    return int(tokenizer.get_pixel_num_frames(tokenizer.get_latent_num_frames(spans[0])))
