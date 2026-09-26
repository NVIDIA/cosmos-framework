# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cosmos_framework.data.generator.multiview.camera_attributes import MADS_CAMERA_ATTRIBUTES
from cosmos_framework.data.generator.multiview.caption_format import (
    DEFAULT_CAPTION_PREFIXES,
    first_caption_paragraph,
    format_multiview_caption,
    format_separate_view_captions,
    format_view_caption,
)


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


def label_view_captions(view_captions: Sequence[str], *, camera_keys: Sequence[str]) -> list[str | dict[str, Any]]:
    """Label each caption as the legacy merged-caption training dataloader does.

    The returned camera-ordered entries are merged by ``format_multiview_caption`` for a model
    reading one prompt.
    """
    unlabeled = [key for key in camera_keys if key not in DEFAULT_CAPTION_PREFIXES]
    if unlabeled:
        raise ValueError(
            f"No caption prefix is defined for camera(s) {', '.join(unlabeled)}; captions can only be "
            f"assembled for {', '.join(sorted(DEFAULT_CAPTION_PREFIXES))}."
        )
    return [
        format_view_caption(
            caption=caption,
            camera_name=camera_key,
            view_index=view_index,
            add_view_prefix=True,
            camera_prefixes=DEFAULT_CAPTION_PREFIXES,
        )
        for view_index, (camera_key, caption) in enumerate(zip(camera_keys, view_captions, strict=True))
    ]


def view_caption_texts(labeled_captions: Sequence[str | dict[str, Any]]) -> list[str]:
    """One caption string per camera, matching how the training tokenizer takes a caption payload.

    ``TextTokenizerTransform`` JSON-serializes a dict caption and passes a string through, so doing
    the same here keeps the per-view text identical to training's.
    """
    return [caption if isinstance(caption, str) else json.dumps(caption) for caption in labeled_captions]


def load_multiview_caption_chunks_per_view(
    views: Sequence[Any],
    *,
    first_only: bool = False,
    first_caption_paragraph_only: bool = False,
) -> list[MultiviewCaptionChunk]:
    """Read every camera's caption file and assemble one prompt per chunk.

    Each chunk carries both forms: ``prompt`` is the merged text a model reading one prompt gets,
    and ``view_prompts`` is one caption per camera for a checkpoint trained with
    ``separate_view_text_tokenization``. Which one is packed is decided at generation time from the
    checkpoint's own recorded layout, not here.

    ``first_caption_paragraph_only`` matches short-window LiDAR training. It trims
    the raw text before adding camera headers; the default preserves full captions.

    Chunkwise rollout generates all cameras together and slices the model output by one shared
    frame count, so the cameras have to agree on the chunk boundaries: a caption file written
    against a different frame span would silently describe frames other than the generated ones.

    ``views`` is duck-typed on ``camera_key`` and ``caption_path`` -- the two attributes this reads
    -- so each runtime passes its own camera-args type unchanged. Same reason ``tokenizer`` is
    duck-typed in :func:`caption_chunk_frame_count`: naming either concrete type here would tie this
    leaf to one repo's arg schema.
    """
    per_view_chunks: list[list[MultiviewCaptionChunk]] = []
    for view in views:
        if view.caption_path is None:
            raise ValueError(
                f"multiview.load_caption_from_data=True needs a caption for camera {view.camera_key!r}. "
                "Set its caption_path, multiview.caption_path, or multiview.caption_root."
            )
        loaded = load_multiview_caption_chunks(Path(view.caption_path))
        per_view_chunks.append(loaded[:1] if first_only else loaded)

    if not per_view_chunks:
        raise ValueError("Caption chunking requires at least one camera.")

    chunk_counts = {view.camera_key: len(chunks) for view, chunks in zip(views, per_view_chunks, strict=True)}
    if len(set(chunk_counts.values())) > 1:
        raise ValueError(f"Cameras disagree on their caption chunk count: {chunk_counts}.")

    camera_keys = [view.camera_key for view in views]
    chunks: list[MultiviewCaptionChunk] = []
    for index, view_chunks in enumerate(zip(*per_view_chunks, strict=True)):
        spans = {(chunk.frame_start, chunk.frame_end) for chunk in view_chunks}
        if len(spans) > 1:
            raise ValueError(
                f"Caption chunk {index} covers a different frame span per camera: "
                f"{ {key: (chunk.frame_start, chunk.frame_end) for key, chunk in zip(camera_keys, view_chunks, strict=True)} }."
            )
        frame_start, frame_end = spans.pop()
        # Preserve the legacy one-camera prefixes for a model reading one merged prompt. A model
        # trained with separate_view_text_tokenization instead receives the same selected-rig and
        # current-camera headers as the training dataloader. Which one is packed is decided at
        # generation time from the checkpoint's own training config.
        view_captions = [chunk.prompt for chunk in view_chunks]
        if first_caption_paragraph_only:
            view_captions = [first_caption_paragraph(caption) for caption in view_captions]
        labeled = label_view_captions(view_captions, camera_keys=camera_keys)
        separate_view_captions = format_separate_view_captions(
            view_captions,
            camera_names=camera_keys,
            add_camera_rig_prefix=True,
            camera_attributes=MADS_CAMERA_ATTRIBUTES,
        )
        merged = format_multiview_caption(labeled, use_explicit_view_format=False, use_two_pass_format=False)
        chunks.append(
            MultiviewCaptionChunk(
                index=index,
                frame_start=frame_start,
                frame_end=frame_end,
                prompt=merged if isinstance(merged, str) else json.dumps(merged),
                view_prompts=view_caption_texts(separate_view_captions),
            )
        )
    return chunks
