# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""VLM dataflow roles (RawItemProcessor + BatchCollator) extracted 1:1 from
VLMDataPacker (llava_ov_vlm.py). Behavior-preserving."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data._utils.collate import default_collate

from cosmos_framework.data.generator.dataflow.base import BatchCollator, RawItemProcessor
from cosmos_framework.utils.reasoner.constant import IGNORE_INDEX, PROCESSOR_KEYS_TO_ADD


class VLMProcessor(RawItemProcessor):
    """ShareGPT image+conversation record -> VLM training tensors."""

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
                "VLMProcessor: tokenizer exposes neither pad_token_id nor "
                "eos_token_id; cannot determine a padding id for VLMCollator. "
                "Configure the tokenizer's pad/eos token."
            )
        self._pad_token_id = int(pad_id)

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


class VLMCollator(BatchCollator):
    """Pad-and-stack collation for any batch size: right-pads sequence tensors to
    a multiple of 16, flat-concatenates vision tensors on dim 0, and stamps resume
    meta (zeros — streaming source has no position)."""

    def collate(self, samples: list[dict]) -> dict:
        # Parity with i4 custom_collate: skip if already collated.
        if samples and samples[0].get("collated"):
            return samples[0]

        # All four sequence tensors must be present and 1-D on every sample
        # before padding/stacking (matches i4 custom_collate). A missing key
        # here would otherwise fall through to default_collate as a ragged list.
        for key in ("input_ids", "token_mask", "attention_mask", "labels"):
            assert all(key in s and s[key].ndim == 1 for s in samples), (
                f"VLMCollator: {key} must be present and 1-D on every sample"
            )

        # Right-pad target length, rounded up to a multiple of 16 (FP8 support).
        max_seq_length = max(s["input_ids"].shape[0] for s in samples)
        max_seq_length = (max_seq_length + 15) // 16 * 16

        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        batch_size = len(samples)

        regular: dict = {}
        special: dict = {}

        def _pad_stack(key: str, fill, dtype) -> torch.Tensor:
            rows = []
            for s in samples:
                t = s[key]
                pad = torch.full((max_seq_length - t.shape[0],), fill, dtype=dtype)
                rows.append(torch.cat([t, pad]))
            return torch.stack(rows, dim=0)

        # input_ids: pad with each sample's pad_token_id.
        regular["input_ids"] = torch.stack(
            [
                torch.cat([
                    s["input_ids"],
                    torch.full((max_seq_length - s["input_ids"].shape[0],),
                               s["pad_token_id"], dtype=torch.long),
                ])
                for s in samples
            ],
            dim=0,
        )

        # token_mask / attention_mask: pad with False (guaranteed present by the
        # assertion above).
        for key in ("token_mask", "attention_mask"):
            regular[key] = _pad_stack(key, False, torch.bool)

        # labels: pad with each sample's ignore_index.
        regular["labels"] = torch.stack(
            [
                torch.cat([
                    s["labels"],
                    torch.full((max_seq_length - s["labels"].shape[0],),
                               s["ignore_index"], dtype=torch.long),
                ])
                for s in samples
            ],
            dim=0,
        )

        # raw_image / raw_video: keep per-sample, per-item boundaries (parity).
        if any("raw_image" in s for s in samples):
            ri: list = []
            for s in samples:
                img = s.get("raw_image", [])
                if isinstance(img, torch.Tensor):
                    if img.ndim == 3:
                        img = img[:, None]
                    img = [img[:, i:i + 1] for i in range(img.shape[1])]
                ri.append(img)
            regular["raw_image"] = ri
        if any("raw_video" in s for s in samples):
            rv: list = []
            for s in samples:
                vid = s.get("raw_video", [])
                if isinstance(vid, torch.Tensor):
                    vid = [vid]
                rv.append(vid)
            regular["raw_video"] = rv

        # Vision tensors: flat-concatenate on dim 0 (Qwen3-VL addresses them via
        # placeholder tokens in input_ids, not by batch position).
        vision_cat_keys = (
            "image_grid_thw", "video_grid_thw", "second_per_grid_ts",
            "pixel_values", "pixel_values_videos", "image_sizes",
        )
        all_keys = {k for s in samples for k in s}
        for key in all_keys:
            if key in regular:
                continue
            if key in vision_cat_keys:
                special[key] = torch.cat([s[key] for s in samples if key in s], dim=0)
            else:
                regular[key] = default_collate([s[key] for s in samples])

        batch = {**regular, **special, "collated": True}
        # Resume meta (streaming source has no position -> zeros), length-B.
        batch["sample_worker_id"] = torch.tensor([worker_id] * batch_size)
        batch["sample_epoch"] = torch.tensor([0] * batch_size)
        batch["sample_index"] = torch.tensor([0] * batch_size)
        return batch
