# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""videophy2 RawItemProcessor extracted 1:1 from VideoPhy2DataPacker."""

from __future__ import annotations

import io
from typing import Any

from cosmos_framework.data.generator.dataflow.base import RawItemProcessor
from cosmos_framework.utils.reasoner.constant import IGNORE_INDEX, PROCESSOR_KEYS_TO_ADD

_MAX_VIDEO_FRAMES = 32
_TARGET_VIDEO_FPS = 2.0


def _decode_video_to_pil_frames(video_bytes: bytes) -> tuple[list, float]:
    from torchcodec.decoders import VideoDecoder
    from PIL import Image
    import numpy as np

    decoder = VideoDecoder(video_bytes)
    total_frames = decoder.metadata.num_frames or 0
    source_fps = float(decoder.metadata.average_fps or 0.0) or 30.0

    if total_frames <= 0:
        raise ValueError("video has zero frames")

    stride = max(1, int(round(source_fps / _TARGET_VIDEO_FPS)))
    indices = list(range(0, total_frames, stride))
    if len(indices) > _MAX_VIDEO_FRAMES:
        step = len(indices) / _MAX_VIDEO_FRAMES
        indices = [indices[int(i * step)] for i in range(_MAX_VIDEO_FRAMES)]

    frames_tensor = decoder.get_frames_at(indices=indices).data
    frames_np = frames_tensor.permute(0, 2, 3, 1).contiguous().cpu().numpy().astype(np.uint8)
    frames = [Image.fromarray(f) for f in frames_np]

    effective_fps = source_fps / stride if stride > 0 else source_fps
    return frames, float(effective_fps)


class VideoPhy2Processor(RawItemProcessor):
    """LocalSFT {"texts","media"} record -> VLM training tensors."""

    def __init__(self, processor: Any, ignore_index: int = IGNORE_INDEX) -> None:
        self._processor = processor
        self._ignore_index = ignore_index
        # Resolve pad token id once; VLMCollator uses it to right-pad input_ids.
        tok = getattr(processor, "tokenizer", processor)
        pad_id = getattr(tok, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(tok, "eos_token_id", None)
        if pad_id is None:
            raise ValueError(
                "VideoPhy2Processor: tokenizer exposes neither pad_token_id nor "
                "eos_token_id; cannot determine a padding id for VLMCollator. "
                "Configure the tokenizer's pad/eos token."
            )
        self._pad_token_id = int(pad_id)

    def _materialize_media_in_conversation(
        self,
        conversation: list,
        media_bytes_by_key: dict,
    ) -> list:
        # Resolve "video": "<key>" / "image": "<key>" references against
        # data_dict["media"] (bytes); decode each unique key once.
        decoded_cache: dict[str, tuple[list, float]] = {}
        new_messages: list[dict] = []
        for message in conversation:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if isinstance(content, str):
                new_messages.append({"role": message.get("role", "user"), "content": content})
                continue
            if not isinstance(content, list):
                continue
            new_content: list[dict] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                kind = item.get("type")
                if kind == "video":
                    key = item.get("video")
                    if not isinstance(key, str):
                        new_content.append(item)
                        continue
                    if key not in media_bytes_by_key:
                        raise KeyError(
                            f"conversation references video key {key!r} not present in "
                            f"sample['media'] (keys: {list(media_bytes_by_key)})"
                        )
                    if key not in decoded_cache:
                        decoded_cache[key] = _decode_video_to_pil_frames(media_bytes_by_key[key])
                    frames, fps = decoded_cache[key]
                    new_content.append({"type": "video", "video": frames, "fps": fps})
                elif kind == "image":
                    key = item.get("image")
                    if not isinstance(key, str):
                        new_content.append(item)
                        continue
                    if key not in media_bytes_by_key:
                        raise KeyError(
                            f"conversation references image key {key!r} not present in "
                            f"sample['media'] (keys: {list(media_bytes_by_key)})"
                        )
                    from PIL import Image

                    img = Image.open(io.BytesIO(media_bytes_by_key[key])).convert("RGB")
                    new_content.append({"type": "image", "image": img})
                else:
                    new_content.append(item)
            new_messages.append({"role": message.get("role", "user"), "content": new_content})
        return new_messages

    def process(self, item: dict) -> dict:
        conversation = item.get("texts")
        if not isinstance(conversation, list):
            raise TypeError(
                f"LocalSFTDataset sample expected 'texts' to be a list, got {type(conversation).__name__}"
            )
        media_bytes_by_key = item.get("media") or {}
        messages = self._materialize_media_in_conversation(conversation, media_bytes_by_key)
        inputs = self._processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        input_ids = inputs["input_ids"]
        token_mask = self._processor.add_assistant_tokens_mask(input_ids)
        labels = input_ids.clone()
        labels[~token_mask] = self._ignore_index
        result: dict = {
            "input_ids": input_ids,
            "labels": labels,
            "token_mask": token_mask,
            "pad_token_id": self._pad_token_id,
            "ignore_index": self._ignore_index,
        }
        for key in PROCESSOR_KEYS_TO_ADD:
            if key in inputs and inputs[key] is not None:
                result[key] = inputs[key]
        return result
