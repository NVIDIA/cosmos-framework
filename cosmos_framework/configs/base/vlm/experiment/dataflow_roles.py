# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""VLM dataflow roles (RawItemProcessor + BatchCollator) extracted 1:1 from
VLMDataPacker (llava_ov_vlm.py). Behavior-preserving."""

from __future__ import annotations

from typing import Any

import torch

from cosmos_framework.data.vfm.dataflow.base import BatchCollator, RawItemProcessor
from cosmos_framework.utils.vlm.constant import IGNORE_INDEX, PROCESSOR_KEYS_TO_ADD


class VLMProcessor(RawItemProcessor):
    """ShareGPT image+conversation record -> VLM training tensors."""

    def __init__(self, processor: Any, ignore_index: int = IGNORE_INDEX) -> None:
        self._processor = processor
        self._ignore_index = ignore_index

    @staticmethod
    def _decode_image(image: Any) -> Any:
        """Decode a HuggingFace streaming image to PIL.

        In streaming mode HuggingFace delivers images as
        ``{"bytes": bytes, "path": str}`` dicts rather than decoded PIL Images.
        """
        if isinstance(image, dict):
            import io

            from PIL import Image

            raw = image.get("bytes")
            if raw:
                return Image.open(io.BytesIO(raw)).convert("RGB")
            path = image.get("path")
            if path:
                return Image.open(path).convert("RGB")
            return None
        return image

    def _sharegpt_to_openai(self, item: dict) -> list[dict]:
        """Convert ShareGPT conversation to OpenAI message format.

        LLaVA-OneVision-Data records use ``from``/``value`` pairs where the
        human turn may contain a ``<image>`` placeholder.  We strip the
        placeholder and attach the PIL image as a separate content block.
        """
        conversations = item.get("conversations", [])
        image = self._decode_image(item.get("image"))  # PIL.Image or None
        messages: list[dict] = []
        image_inserted = False

        for turn in conversations:
            role = "user" if turn["from"] == "human" else "assistant"
            text = turn["value"].replace("<image>", "").strip()

            if role == "user" and not image_inserted and image is not None:
                content: Any = [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text},
                ]
                image_inserted = True
            else:
                content = text

            messages.append({"role": role, "content": content})

        return messages

    def process(self, item: dict) -> dict:
        messages = self._sharegpt_to_openai(item)
        inputs = self._processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False
        )
        input_ids = inputs["input_ids"]
        token_mask = self._processor.add_assistant_tokens_mask(input_ids)
        labels = input_ids.clone()
        labels[~token_mask] = self._ignore_index
        result: dict = {"input_ids": input_ids, "labels": labels}
        for key in PROCESSOR_KEYS_TO_ADD:
            if key in inputs and inputs[key] is not None:
                result[key] = inputs[key]
        return result


class VLMCollator(BatchCollator):
    """max_batch_size=1 collation: batch-dim sequence tensors, keep vision tensors
    flat, stamp resume meta (zeros — streaming source has no position)."""

    def collate(self, samples: list[dict]) -> dict:
        assert len(samples) == 1, f"VLMCollator expects max_batch_size=1, got {len(samples)}"
        s = samples[0]
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        batch: dict = {
            "input_ids": s["input_ids"].unsqueeze(0),
            "labels": s["labels"].unsqueeze(0),
            "sample_worker_id": torch.tensor([worker_id]),
            "sample_epoch": torch.tensor([0]),
            "sample_index": torch.tensor([0]),
        }
        if "attention_mask" in s and s["attention_mask"] is not None:
            batch["attention_mask"] = s["attention_mask"].unsqueeze(0)
        for key in (
            "pixel_values", "pixel_values_videos", "image_grid_thw",
            "video_grid_thw", "second_per_grid_ts",
        ):
            if key in s and s[key] is not None:
                batch[key] = s[key]
        return batch
