# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Qwen3-VL shaping for rows that carry a VIDEO as an ordered list of frames.

``Qwen3VLDataPacker`` shapes text and image rows exactly. On a video row two of its contracts break
against the vLLM engine; this subclass repairs both and leaves the image packer untouched:

1. **Policy prompt ids.** vLLM tokenizes the rendered chat template with the video placeholder as
   three special tokens (``<|vision_start|><|video_pad|><|vision_end|>``) and then splices, at the
   TOKEN level, one group per temporal patch: ``encode("<t seconds>")`` + ``<|vision_start|>`` +
   ``<|video_pad|>`` x tokens_per_frame + ``<|vision_end|>`` (``Qwen3VLMultiModalProcessor``,
   ``get_video_repl``). The rollout stores those ids as ``policy_info["prompt_ids"]``. The HF
   processor instead writes the timestamps INTO THE TEXT and tokenizes the whole string once, so
   the last text character and the ``<`` of the first timestamp BPE-merge (``"result."`` +
   ``"<1.3 seconds>"`` -> ``".<"``): one id fewer, and ``get_policy_input`` refuses every such row
   with ``stored rollout prompt_ids differ from the Qwen3-VL processor prefix``. Here the policy
   ids of a video row are built the engine's way (render, tokenize with ``add_special_tokens=False``,
   splice), while the MEDIA tensors (``pixel_values_videos``, ``video_grid_thw``) still come from the
   HF processor byte for byte. m-RoPE reads the ``<|vision_start|>`` / ``<|video_pad|>`` positions
   and the grid; the spliced ids keep the group count, the pads per group and the grid, so the
   vision positions are the engine's positions.
2. **Rollout request.** The base request carries ``multi_modal_data["video"] = [(list[str],
   metadata)]`` and no ``multi_modal_uuids["video"]``: at TP > 1 ``vllm_request_media_identity``
   refuses it before the engine (``video media requires ordered content UUIDs``), and vLLM's HF
   video processor treats every ``str`` as a video FILE. Here every frame is decoded like an image,
   stacked into one ``uint8 [T,H,W,3]`` array, and named by one content UUID over the ordered frame
   digests plus the canonical metadata (fps / total_num_frames / frames_indices): the same frames
   at another fps expand to different timestamp tokens, so the metadata is part of the identity.

The HF processor stays as a cross-check of the splice, and that cross-check is transformers-version
agnostic: 4.x ``Qwen3VLProcessor`` replaces the WHOLE placeholder triple by the timestamped groups
(the "triple" text form: N groups -> N ``<|vision_start|>``); 5.x replaces only ``<|video_pad|>``
and keeps the template's outer marker pair (the "outer" form: N + 1). Either is accepted when the
HF ids carry the same video pads, the matched form's marker counts and the same decoded text;
anything else (a missing pad, a changed timestamp, a stray marker) is refused with the forms named.
The form that matched is recorded in ``last_hf_text_form`` for diagnostics.

Guards refuse identically on the request side and the policy side, so a row the rollout generates
on is a row the trainer can shape: a frame count that is not a multiple of the video processor's
``temporal_patch_size`` (HF would repeat the last frame, vLLM would pad ``frames_indices``, and 4.x
pads the caller's list in place), a literal placeholder or a stray vision marker/pad in the text, a
video block without a positive ``fps``, a video handle that is not a list of frames, frames of
unequal size, and a pretokenized request that carries a video (vLLM re-expands the placeholder, so
already expanded ids cannot be resubmitted).

A ``{"type": "video_frames", ...}`` block is the same row: the base media extraction already collects it
as a video, so the packer's copy of the messages carries it as ``type: "video"`` before either side reads
it (the HF metadata reader knows only ``video``; the ObsBundle's own block is never rewritten). The
policy side hands the HF processor the SAME frames the rollout request carried (every frame decoded,
EXIF-transposed and converted to RGB, ``_materialize_vllm_image``), never the row's original handles:
HF takes a list of frames as given, so an EXIF-oriented frame would otherwise reach the model as other
pixels than the engine sampled from, under the same grid.

Rows without a video take the base packer's request and policy input byte for byte. Select this
packer for a run through the existing knob
``data_packer.factory=cosmos_framework.posttrain.data.packer.qwen3_vl_video.make_packer``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from PIL import Image

from cosmos_framework.posttrain.data.packer.qwen3_vl import Qwen3VLDataPacker
from cosmos_framework.posttrain.schema import ObsBundle
from cosmos_framework.posttrain.utils.identity import canonical_json_sha256, qualified_type

VIDEO_UUID_SEED = "qwen3-vl-video-v1"
VIDEO_POLICY_IDS_SEMANTICS = (
    "qwen3-vl-video:server-splice;guards=even-frames,placeholder-triples,no-stray-markers;"
    "hf-crosscheck=text-form-agnostic(triple|outer);policy-frames=rollout-snapshots;video_frames-alias"
)
HF_TEXT_FORMS: tuple[str, ...] = ("triple", "outer")
_VIDEO_BLOCK_TYPES = frozenset({"video", "video_frames"})


def calculate_timestamps(indices: list[int] | torch.Tensor, video_fps: float, temporal_patch_size: int) -> list[float]:
    """The engine's per-group timestamps: pad the frame indices to a multiple of ``temporal_patch_size``
    by repeating the last one, convert to seconds, and take the mean of each group's first and last
    frame (vLLM ``Qwen3VLProcessingInfo._calculate_timestamps``; equal to the HF processor's)."""
    if not isinstance(indices, list):
        indices = indices.tolist()
    indices = [int(index) for index in indices]
    if len(indices) % temporal_patch_size != 0:
        indices = indices + [indices[-1]] * (temporal_patch_size - len(indices) % temporal_patch_size)
    timestamps = [index / video_fps for index in indices]
    return [
        (timestamps[i] + timestamps[i + temporal_patch_size - 1]) / 2
        for i in range(0, len(timestamps), temporal_patch_size)
    ]


def timestamp_token_ids(tokenizer: Any, timestamps: list[float]) -> list[list[int]]:
    """Every group's ``<t seconds>`` encoded ALONE, the way ``get_video_repl`` encodes it."""
    return [
        [int(token_id) for token_id in tokenizer.encode(f"<{timestamp:.1f} seconds>", add_special_tokens=False)]
        for timestamp in timestamps
    ]


def video_replacement_ids(
    *,
    timestamp_ids: list[list[int]],
    tokens_per_frame: int,
    vision_start_token_id: int,
    vision_end_token_id: int,
    video_token_id: int,
) -> list[int]:
    """The engine's replacement for one video placeholder: timestamp + vision_start + pads + vision_end per group."""
    replacement: list[int] = []
    for group_ids in timestamp_ids:
        replacement.extend(group_ids)
        replacement.append(vision_start_token_id)
        replacement.extend([video_token_id] * tokens_per_frame)
        replacement.append(vision_end_token_id)
    return replacement


class Qwen3VLVideoDataPacker(Qwen3VLDataPacker):
    """Qwen3-VL shaping whose video rows carry the engine's spliced prompt ids and a content-named request."""

    def __init__(self, *, processor: Any | None = None, model_config: Any | None = None) -> None:
        super().__init__(processor=processor, model_config=model_config)
        self.last_hf_text_form: str | None = None

    # ------------------------------------------------------------------ knobs read from the live processor / config
    def _video_processor_knobs(self) -> tuple[int, int]:
        """``(temporal_patch_size, merge_size)`` of the live HF video processor, guarded."""
        video_processor = getattr(self.processor, "video_processor", None)
        if video_processor is None:
            raise TypeError("Qwen3-VL video packer requires a processor with a video_processor for a video row")
        temporal_patch_size = getattr(video_processor, "temporal_patch_size", None)
        merge_size = getattr(video_processor, "merge_size", None)
        for name, value in (("temporal_patch_size", temporal_patch_size), ("merge_size", merge_size)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"Qwen3-VL video_processor.{name} must be a positive int, got {value!r}")
        if int(merge_size) != int(self.spatial_merge_size):
            raise ValueError(
                f"Qwen3-VL video_processor.merge_size {merge_size} != config spatial_merge_size "
                f"{self.spatial_merge_size}; the pad count per frame would disagree with the model"
            )
        return int(temporal_patch_size), int(merge_size)

    def _marker_ids(self) -> tuple[int, int, int, int]:
        """``(vision_start, vision_end, image_pad, video_pad)`` ids; every one must exist for the token-level splice."""
        ids = (self.vision_start_token_id, self.vision_end_token_id, self.image_token_id, self.video_token_id)
        if any(token_id is None for token_id in ids):
            raise TypeError(
                "Qwen3-VL video packer requires vision_start / vision_end / image_pad / video_pad token ids for a "
                f"video row, got {ids}"
            )
        return tuple(int(token_id) for token_id in ids)  # type: ignore[return-value]

    # ------------------------------------------------------------------ the messages both sides read
    def _messages_and_media(self, item: ObsBundle) -> tuple[list[dict[str, Any]], list[Any], list[Any]]:
        """The base extraction, with every ``video_frames`` block of the packer's COPY renamed ``video``: the base
        collects the alias as a video handle, but the HF metadata reader (``maybe_parse_video_content``) and the
        chat template know only ``video``, so an alias row died on both sides with "derived no video metadata"."""
        messages, images, videos = super()._messages_and_media(item)
        for message in messages:
            for block in message["content"]:
                if isinstance(block, dict) and block.get("type") == "video_frames":
                    block["type"] = "video"
        return messages, images, videos

    # ------------------------------------------------------------------ the one metadata reader both sides use
    @staticmethod
    def _video_blocks(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            block
            for message in messages
            for block in message["content"]
            if isinstance(block, Mapping) and block.get("type") in _VIDEO_BLOCK_TYPES
        ]

    def _video_metadata(self, messages: list[dict[str, Any]], videos: list[Any]) -> list[dict[str, Any]]:
        """Per-video metadata (list handle, positive fps, one index per frame, an even frame count); returns COPIES.

        Called by the rollout request before the render and by the policy side before the HF call, so every
        metadata defect is refused on both sides with one message. The returned rows own their ``frames_indices``
        lists: transformers 4.x ``Qwen3VLProcessor._calculate_timestamps`` pads the list it is given IN PLACE and
        the base ``_processor_kwargs`` hands HF the message block's list, so nothing downstream of the HF call may
        read the block's list.
        """
        blocks = self._video_blocks(messages)
        if len(blocks) != len(videos):
            raise ValueError(f"Qwen3-VL found {len(blocks)} video message block(s) for {len(videos)} handle(s)")
        for index, (block, handle) in enumerate(zip(blocks, videos, strict=True)):
            location = f"vLLM video[{index}]"
            if not isinstance(handle, (list, tuple)) or len(handle) == 0:
                raise ValueError(
                    f"{location} handle must be a non-empty list of frames (local paths / bytes / PIL / ndarray), "
                    f"got {qualified_type(handle)}: a video file path is not decodable here and vLLM rejects a str "
                    "video handle; sample the frames at build time"
                )
            fps = block.get("fps")
            if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not fps > 0:
                raise ValueError(
                    f"{location} block requires a positive numeric fps (the <t seconds> timestamp tokens are "
                    f"frames_indices / fps), got {fps!r}"
                )
        hf_kwargs = self._processor_kwargs(messages)
        raw_metadata = hf_kwargs.get("videos_kwargs", {}).get("video_metadata")
        if raw_metadata is None:
            raise ValueError("Qwen3-VL derived no video metadata for a row that carries video handles")
        metadata = raw_metadata if isinstance(raw_metadata, list) else [raw_metadata]
        if len(metadata) != len(videos):
            raise ValueError(f"Qwen3-VL video metadata has {len(metadata)} row(s) for {len(videos)} handle(s)")
        temporal_patch_size, _merge_size = self._video_processor_knobs()
        copied: list[dict[str, Any]] = []
        for index, (row, handle) in enumerate(zip(metadata, videos, strict=True)):
            if len(row["frames_indices"]) != len(handle):
                raise ValueError(
                    f"vLLM video[{index}] has {len(handle)} frame(s) but {len(row['frames_indices'])} frames_indices"
                )
            if len(handle) % temporal_patch_size:
                raise ValueError(
                    f"vLLM video[{index}] has {len(handle)} frame(s), not a multiple of the video processor's "
                    f"temporal_patch_size {temporal_patch_size}: the HF processor would repeat the last frame, vLLM would "
                    "pad frames_indices, and transformers 4.x pads the caller's frames_indices list in place; the "
                    "Qwen3-VL video packer refuses an odd frame count on the rollout-request side and on the policy side "
                    f"alike; sample a multiple of {temporal_patch_size} frames at build time"
                )
            copied.append({**row, "frames_indices": list(row["frames_indices"])})
        return copied

    # ------------------------------------------------------------------ the rollout request
    @staticmethod
    def _frame_array(payload: Any, *, location: str) -> np.ndarray:
        """One decoded frame as a ``uint8`` ``[H,W,3]`` array."""
        if isinstance(payload, Image.Image):
            array = np.asarray(payload.convert("RGB"), dtype=np.uint8)
        elif isinstance(payload, torch.Tensor):
            array = payload.numpy()
        elif isinstance(payload, np.ndarray):
            array = payload
        else:
            raise TypeError(f"{location} decoded to {qualified_type(payload)}, expected PIL/ndarray/tensor")
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError(f"{location} must be one [H,W,3] RGB frame, got shape {tuple(array.shape)}")
        if array.dtype != np.uint8:
            raise ValueError(f"{location} must be uint8 like a decoded JPEG, got {array.dtype}")
        return np.ascontiguousarray(array)

    def _materialize_vllm_videos(
        self,
        videos: list[Any],
        metadata: list[dict[str, Any]],
    ) -> tuple[list[np.ndarray], tuple[str, ...]]:
        """Stack every frame list into ``[T,H,W,3]`` uint8 and derive one ordered content UUID per video."""
        payloads: list[np.ndarray] = []
        video_uuids: list[str] = []
        for index, (handle, row) in enumerate(zip(videos, metadata, strict=True)):
            location = f"vLLM video[{index}]"
            frames: list[np.ndarray] = []
            digests: list[str] = []
            for frame_index, frame in enumerate(handle):
                payload, digest = self._materialize_vllm_image(frame, location=f"{location} frame[{frame_index}]")
                frames.append(self._frame_array(payload, location=f"{location} frame[{frame_index}]"))
                digests.append(digest)
            shapes = {frame.shape for frame in frames}
            if len(shapes) != 1:
                raise ValueError(f"{location} frames must share one HxWxC shape, got {sorted(shapes)}")
            video = np.stack(frames, axis=0)  # [T,H,W,3]
            content_sha256 = canonical_json_sha256(
                {"frames": digests, "metadata": row},
                context=f"{location} content identity",
            )
            cache_seed = f"{VIDEO_UUID_SEED}\0{index}\0{content_sha256}".encode("utf-8")
            payloads.append(video)
            video_uuids.append(hashlib.sha256(cache_seed).hexdigest())
        return payloads, tuple(video_uuids)

    @staticmethod
    def _bind_video_payloads(messages: list[dict[str, Any]], videos: list[list[Image.Image]]) -> None:
        """Replace every video block's handle with the normalized frames the rollout request carried."""
        video_index = 0
        for message in messages:
            for block in message["content"]:
                if not isinstance(block, Mapping) or block.get("type") not in _VIDEO_BLOCK_TYPES:
                    continue
                if video_index >= len(videos):
                    raise ValueError("Qwen3-VL message/video cardinality changed while binding the policy frames")
                block["video"] = videos[video_index]
                video_index += 1
        if video_index != len(videos):
            raise ValueError("Qwen3-VL message/video cardinality changed while binding the policy frames")

    def _request(self, item: ObsBundle, *, prompt_token_ids: list[int] | None = None) -> dict[str, Any]:
        messages, images, videos = self._messages_and_media(item)
        if not videos:
            # Text and image rows: the base request, byte for byte.
            return super()._request(item, prompt_token_ids=prompt_token_ids)
        if prompt_token_ids is not None:
            raise ValueError(
                "Qwen3-VL video packer refuses a pretokenized request with video: vLLM re-expands "
                "<|vision_start|><|video_pad|><|vision_end|> into timestamped groups, so an already expanded id "
                "prompt cannot be resubmitted; use the string-prompt rollout path"
            )
        materialized_images, image_uuids = self._materialize_vllm_images(images)
        self._bind_image_payloads(messages, materialized_images)
        metadata = self._video_metadata(messages, videos)
        materialized_videos, video_uuids = self._materialize_vllm_videos(videos, metadata)
        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            chat_template=self._resolved_chat_template(),
        )
        # ``_vllm_media`` re-derives the same metadata from ``messages`` (the video blocks still hold the frame
        # handles) and zips it with the materialised arrays -> ``[(ndarray, metadata)]``.
        multi_modal_data, vllm_kwargs = self._vllm_media(messages, materialized_images, materialized_videos)
        request: dict[str, Any] = {"prompt": prompt}
        if multi_modal_data:
            request["multi_modal_data"] = multi_modal_data
        multi_modal_uuids: dict[str, list[str]] = {}
        if materialized_images:
            multi_modal_uuids["image"] = list(image_uuids)
        multi_modal_uuids["video"] = list(video_uuids)
        request["multi_modal_uuids"] = multi_modal_uuids
        if vllm_kwargs:
            request["mm_processor_kwargs"] = vllm_kwargs
        # The same placeholder guard the policy side runs, on the string the engine will tokenize.
        self._refuse_placeholder_defects(
            self._unexpanded_prompt_ids(prompt), n_images=len(materialized_images), n_videos=len(materialized_videos)
        )
        return request

    # ------------------------------------------------------------------ the render's unexpanded ids (both sides)
    def _unexpanded_prompt_ids(self, rendered: str) -> list[int]:
        """Tokenize a rendered prompt the vLLM frontend's way: ``add_special_tokens=False``, the 3-token placeholders stay."""
        if not isinstance(rendered, str) or not rendered:
            raise TypeError("Qwen3-VL chat template must render a non-empty string")
        raw = self.tokenizer(rendered, add_special_tokens=False)["input_ids"]
        raw = raw.tolist() if hasattr(raw, "tolist") else list(raw)
        if raw and isinstance(raw[0], list):
            if len(raw) != 1:
                raise ValueError(f"Qwen3-VL tokenizer returned batch size {len(raw)} for one rendered prompt")
            raw = raw[0]
        prompt_ids = [int(token_id) for token_id in raw]
        if not prompt_ids:
            raise ValueError("Qwen3-VL tokenizer produced an empty prompt")
        return prompt_ids

    def _server_prompt_ids(self, messages: list[dict[str, Any]]) -> list[int]:
        rendered = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            chat_template=self._resolved_chat_template(),
        )
        return self._unexpanded_prompt_ids(rendered)

    def _count_placeholder_triples(self, prompt_ids: list[int]) -> tuple[int, int]:
        """``(image, video)`` counts of ``[vision_start, pad, vision_end]`` triples in the unexpanded ids."""
        vision_start, vision_end, image_pad, video_pad = self._marker_ids()
        images = 0
        videos = 0
        for index in range(len(prompt_ids) - 2):
            if prompt_ids[index] == vision_start and prompt_ids[index + 2] == vision_end:
                images += int(prompt_ids[index + 1] == image_pad)
                videos += int(prompt_ids[index + 1] == video_pad)
        return images, videos

    def _refuse_placeholder_defects(self, prompt_ids: list[int], *, n_images: int, n_videos: int) -> None:
        """Placeholder triples == media handles, and no vision marker or pad outside a triple (both sides)."""
        vision_start, vision_end, image_pad, video_pad = self._marker_ids()
        image_triples, video_triples = self._count_placeholder_triples(prompt_ids)
        if video_triples != n_videos or image_triples != n_images:
            raise ValueError(
                f"Qwen3-VL rendered prompt carries {video_triples} video / {image_triples} image placeholder triple(s) "
                f"for {n_videos} video / {n_images} image handle(s); a literal placeholder inside the text is refused on "
                "the rollout-request side and on the policy side alike"
            )
        triples = image_triples + video_triples
        stray = {
            "<|video_pad|>": prompt_ids.count(video_pad) - video_triples,
            "<|image_pad|>": prompt_ids.count(image_pad) - image_triples,
            "<|vision_start|>": prompt_ids.count(vision_start) - triples,
            "<|vision_end|>": prompt_ids.count(vision_end) - triples,
        }
        if any(stray.values()):
            raise ValueError(
                "Qwen3-VL rendered prompt carries special token(s) outside a placeholder triple: "
                + ", ".join(f"{count} {name}" for name, count in stray.items() if count)
                + "; a stray vision marker or pad inside the text is refused on the rollout-request side and on the "
                "policy side alike"
            )

    # ------------------------------------------------------------------ the engine's token-level splice
    def _splice_media_placeholders(
        self,
        prompt_ids: list[int],
        *,
        image_widths: list[int],
        video_replacements: list[list[int]],
    ) -> list[int]:
        """Replace every ``[vision_start, pad, vision_end]`` triple, in order, by its engine replacement."""
        vision_start, vision_end, image_pad, video_pad = self._marker_ids()
        spliced: list[int] = []
        image_index = 0
        video_index = 0
        index = 0
        length = len(prompt_ids)
        while index < length:
            if (
                index + 2 < length
                and prompt_ids[index] == vision_start
                and prompt_ids[index + 2] == vision_end
                and prompt_ids[index + 1] in (image_pad, video_pad)
            ):
                if prompt_ids[index + 1] == image_pad:
                    if image_index >= len(image_widths):
                        raise ValueError(
                            f"Qwen3-VL rendered prompt has more image placeholders than image grids ({len(image_widths)})"
                        )
                    spliced.append(vision_start)
                    spliced.extend([image_pad] * image_widths[image_index])
                    spliced.append(vision_end)
                    image_index += 1
                else:
                    if video_index >= len(video_replacements):
                        raise ValueError(
                            "Qwen3-VL rendered prompt has more video placeholders than video handles "
                            f"({len(video_replacements)}); a literal placeholder inside the text is refused"
                        )
                    spliced.extend(video_replacements[video_index])
                    video_index += 1
                index += 3
                continue
            spliced.append(prompt_ids[index])
            index += 1
        if image_index != len(image_widths) or video_index != len(video_replacements):
            raise ValueError(
                "Qwen3-VL placeholder/media cardinality mismatch in the rendered prompt: "
                f"image placeholders={image_index} for {len(image_widths)} grid(s), "
                f"video placeholders={video_index} for {len(video_replacements)} handle(s)"
            )
        return spliced

    def _video_replacements(
        self,
        metadata: list[dict[str, Any]],
        video_grid: torch.Tensor,
    ) -> tuple[list[list[int]], list[list[str]]]:
        """Per video: the engine's replacement ids and the ``<t seconds>`` strings (for diagnostics)."""
        vision_start, vision_end, _image_pad, video_pad = self._marker_ids()
        temporal_patch_size, merge_size = self._video_processor_knobs()
        if video_grid.ndim != 2 or video_grid.shape[1] != 3 or int(video_grid.shape[0]) != len(metadata):
            raise ValueError(
                f"Qwen3-VL video_grid_thw {tuple(video_grid.shape)} does not match {len(metadata)} video metadata row(s)"
            )
        replacements: list[list[int]] = []
        strings: list[list[str]] = []
        for index, (row, grid_row) in enumerate(zip(metadata, video_grid.tolist(), strict=True)):
            grid_t, grid_h, grid_w = (int(value) for value in grid_row)
            if grid_t < 1 or grid_h < 1 or grid_w < 1:
                raise ValueError(f"Qwen3-VL video[{index}] grid must be positive [t,h,w], got {grid_row}")
            if (grid_h * grid_w) % (merge_size**2):
                raise ValueError(
                    f"Qwen3-VL video[{index}] grid h*w {grid_h * grid_w} is not divisible by merge area {merge_size**2}"
                )
            tokens_per_frame = grid_h * grid_w // (merge_size**2)
            fps = row.get("fps")
            if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not fps > 0:
                raise ValueError(f"Qwen3-VL video[{index}] metadata requires a positive numeric fps, got {fps!r}")
            frames_indices = row.get("frames_indices")
            if not isinstance(frames_indices, (list, tuple)) or not frames_indices:
                raise ValueError(f"Qwen3-VL video[{index}] metadata requires non-empty frames_indices")
            timestamps = calculate_timestamps(list(frames_indices), float(fps), temporal_patch_size)
            if len(timestamps) != grid_t:
                raise ValueError(
                    f"Qwen3-VL video[{index}] has {len(timestamps)} timestamp group(s) from {len(frames_indices)} "
                    f"frames_indices / temporal_patch_size {temporal_patch_size} but grid t={grid_t}"
                )
            group_ids = timestamp_token_ids(self.tokenizer, timestamps)
            for group_index, ids in enumerate(group_ids):
                if not ids or any(token_id in (vision_start, vision_end, video_pad) for token_id in ids):
                    raise ValueError(
                        f"Qwen3-VL video[{index}] timestamp group {group_index} encoded to {ids}; expected plain text ids"
                    )
            replacements.append(
                video_replacement_ids(
                    timestamp_ids=group_ids,
                    tokens_per_frame=tokens_per_frame,
                    vision_start_token_id=vision_start,
                    vision_end_token_id=vision_end,
                    video_token_id=video_pad,
                )
            )
            strings.append([f"<{timestamp:.1f} seconds>" for timestamp in timestamps])
        return replacements, strings

    # ------------------------------------------------------------------ the policy side
    def _hf_inputs(
        self, messages: list[dict[str, Any]], images: list[Any], videos: list[Any]
    ) -> tuple[list[int], list[bool], dict[str, torch.Tensor]]:
        """The base HF path (media tensors + its own ids): one seam, so a test can emulate the other library form."""
        return Qwen3VLDataPacker._processor_inputs_from_messages(self, messages, images, videos)

    def _processor_inputs_from_messages(
        self,
        messages: list[dict[str, Any]],
        images: list[Any],
        videos: list[Any],
    ) -> tuple[list[int], list[bool], dict[str, torch.Tensor]]:
        if not videos:
            # Text and image rows: the base HF path, byte for byte.
            return super()._processor_inputs_from_messages(messages, images, videos)
        # 1. the video metadata BEFORE the HF call: the even-frame rule fires here, the rows are copies
        metadata = self._video_metadata(messages, videos)
        # 2. the rollout request's rendering, tokenized with the 3-token placeholders; placeholder defects refused
        unexpanded_ids = self._server_prompt_ids(messages)
        self._refuse_placeholder_defects(unexpanded_ids, n_images=len(images), n_videos=len(videos))
        # 3. media tensors (and the HF ids, kept only as a cross-check) from the base HF path, on the frames the ROLLOUT
        #    saw: every frame decoded, EXIF-transposed and RGB (the request's own arrays), bound into the blocks. HF takes
        #    a list of frames as given, so the row's original handles would give the policy other pixels than the
        #    engine sampled from (an EXIF-oriented JPEG frame: same grid, rotated tensor).
        materialized_videos, _video_uuids = self._materialize_vllm_videos(videos, metadata)
        bound_videos = [[Image.fromarray(frame) for frame in video] for video in materialized_videos]
        self._bind_video_payloads(messages, bound_videos)
        hf_prompt_ids, _hf_attention, model_inputs = self._hf_inputs(messages, images, bound_videos)
        video_grid = model_inputs.get("video_grid_thw")
        if video_grid is None:
            raise ValueError("Qwen3-VL HF processor returned no video_grid_thw for a video row")
        image_widths: list[int] = []
        if images:
            image_grid = model_inputs.get("image_grid_thw")
            if image_grid is None:
                raise ValueError("Qwen3-VL HF processor returned no image_grid_thw for the row's images")
            widths, _raw = self._grid_marker_widths(image_grid, modality="image")
            image_widths = list(widths)
        # 4. the engine's replacement per video from the metadata read in step 1 (never from the block after HF)
        replacements, _timestamp_strings = self._video_replacements(metadata, video_grid)
        spliced = self._splice_media_placeholders(
            unexpanded_ids, image_widths=image_widths, video_replacements=replacements
        )
        # 5. the text-form-agnostic cross-check against the HF path (fail closed), then the base media validation
        self.last_hf_text_form = self._crosscheck_hf(
            hf_prompt_ids, spliced, unexpanded_ids, image_widths=image_widths, video_replacements=replacements
        )
        self._validate_processed_media(prompt_ids=spliced, images=images, videos=videos, model_inputs=model_inputs)
        return spliced, [True] * len(spliced), model_inputs

    def _crosscheck_hf(
        self,
        hf_prompt_ids: list[int],
        spliced: list[int],
        unexpanded_ids: list[int],
        *,
        image_widths: list[int],
        video_replacements: list[list[int]],
    ) -> str:
        """Return the HF text form ('triple' | 'outer') the HF ids match exactly; raise ValueError otherwise."""
        vision_start, vision_end, _image_pad, video_pad = self._marker_ids()
        n_hf_pads = hf_prompt_ids.count(video_pad)
        if spliced.count(video_pad) != n_hf_pads:
            raise ValueError(
                f"Qwen3-VL spliced prompt has {spliced.count(video_pad)} video pads, the HF path {n_hf_pads}"
            )
        forms: dict[str, list[int]] = {
            "triple": spliced,
            "outer": self._splice_media_placeholders(
                unexpanded_ids,
                image_widths=image_widths,
                video_replacements=[[vision_start, *replacement, vision_end] for replacement in video_replacements],
            ),
        }
        decoded_hf = self.tokenizer.decode(hf_prompt_ids, skip_special_tokens=False)
        n_hf_vs, n_hf_ve = hf_prompt_ids.count(vision_start), hf_prompt_ids.count(vision_end)
        counts = {name: (ids.count(vision_start), ids.count(vision_end)) for name, ids in forms.items()}
        for name in HF_TEXT_FORMS:
            if (n_hf_vs, n_hf_ve) != counts[name]:
                continue
            if self.tokenizer.decode(forms[name], skip_special_tokens=False) == decoded_hf:
                return name
        decoded_triple = self.tokenizer.decode(forms["triple"], skip_special_tokens=False)
        first = next(
            (i for i, (a, b) in enumerate(zip(decoded_triple, decoded_hf, strict=False)) if a != b),
            None if decoded_triple == decoded_hf else min(len(decoded_triple), len(decoded_hf)),
        )
        raise ValueError(
            f"Qwen3-VL HF path ({n_hf_vs} vision_start / {n_hf_ve} vision_end / {n_hf_pads} video pads) matches neither "
            f"text form: triple form {counts['triple']} (transformers 4.x replaces the whole placeholder triple) nor "
            f"outer form {counts['outer']} (transformers 5.x replaces only <|video_pad|>); first differing char vs the "
            f"triple form {first}; the render, the timestamps or the pads disagree with the HF processor"
        )

    def validation_semantics_identity(self) -> dict[str, Any]:
        identity = super().validation_semantics_identity()
        identity["video_policy_ids"] = VIDEO_POLICY_IDS_SEMANTICS
        return identity


def make_packer() -> Qwen3VLVideoDataPacker:
    """Zero-argument factory for ``data_packer.factory`` (``init_data_packer_from_factory`` calls ``setup``)."""
    return Qwen3VLVideoDataPacker()
