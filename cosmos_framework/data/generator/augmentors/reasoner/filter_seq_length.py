# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Augmentations to remove keys from the output data_dict"""

from collections.abc import Mapping
from typing import Any, List, Optional

from cosmos_framework.data.imaginaire.webdataset.augmentors.augmentor import Augmentor
from cosmos_framework.utils import log
from cosmos_framework.data.generator.processors.audio_utils import (
    AUDIO_END_TOKEN,
    AUDIO_PAD_TOKEN,
    AUDIO_START_TOKEN,
    add_reasoner_audio_special_tokens,
)
from cosmos_framework.data.generator.processors.qwen3vl_processor import Qwen3VLProcessor


class FilterSeqLength(Augmentor):
    """
    Check the sequence length of the input data_dict and filter out the samples that are too long (TODO: Instead of removing them, we can truncate the input ids, but need to make sure the image tokens are not truncated)
    """

    def __init__(
        self,
        input_keys: List = ["input_ids"],
        output_keys: Optional[list] = ["input_ids"],
        max_token_length: int = 24000,
        processor: Qwen3VLProcessor = None,
        sound_und: bool = False,
        audio_start_token: str = AUDIO_START_TOKEN,
        audio_pad_token: str = AUDIO_PAD_TOKEN,
        audio_end_token: str = AUDIO_END_TOKEN,
        drop_over_max_length: bool = False,
        complete_sample_max_length_by_source: Mapping[str, int] | None = None,
    ) -> None:
        """Optionally retain complete supervised examples for exact source URL roots.

        Sources in the map use their own limit, drop overlength examples instead
        of truncating, and require at least one next-token supervision target.
        All other sources retain the existing length and truncation policy.
        """
        self.complete_sample_max_length_by_source: dict[str, int] = dict(complete_sample_max_length_by_source or {})
        if any(
            not isinstance(root, str) or not root or type(limit) is not int or limit < 2
            for root, limit in self.complete_sample_max_length_by_source.items()
        ):
            raise ValueError("Complete-sample length policies require non-empty source roots and integer limits >= 2.")
        self.max_token_length = max_token_length
        self.processor = processor
        self.drop_over_max_length: bool = drop_over_max_length
        self._strict_drop_count: int = 0
        self.audio_pad_token_id: int | None = None
        if sound_und:
            self.audio_pad_token_id = add_reasoner_audio_special_tokens(
                processor.tokenizer,
                model_name_or_path=processor.name,
                audio_start_token=audio_start_token,
                audio_pad_token=audio_pad_token,
                audio_end_token=audio_end_token,
            ).token_ids[1]

    def __call__(self, data_dict: dict[str, Any]) -> dict[str, Any] | None:
        input_ids = data_dict["input_ids"]
        if self.complete_sample_max_length_by_source:
            limit = self.complete_sample_max_length_by_source.get(str(data_dict["__url__"].root))
            if limit is not None:
                # Prefix truncation can remove the answer from a long SFT example.
                if input_ids.shape[-1] > limit:
                    return None
                labels = data_dict["labels"]
                if not bool((labels[1:] != int(data_dict.get("ignore_index", -100))).any()):
                    return None
                return data_dict
        if input_ids.shape[-1] > self.max_token_length and not self.drop_over_max_length:
            # check if there is pixel values or pixel value videos in the remaining tokens, if not truncate the input ids
            input_ids_extra = input_ids[self.max_token_length :]
            has_video_tokens = sum(input_ids_extra == self.processor.video_token_id) > 0
            has_image_tokens = sum(input_ids_extra == self.processor.image_token_id) > 0
            has_audio_tokens = (
                self.audio_pad_token_id is not None and sum(input_ids_extra == self.audio_pad_token_id) > 0
            )
            if not has_video_tokens and not has_image_tokens and not has_audio_tokens:
                log.debug(
                    f"Truncating input_ids from {input_ids.shape[-1]} to {self.max_token_length} because there are no video, image, or audio tokens in the remaining tokens | __url__: path={data_dict['__url__'].path} root={data_dict['__url__'].root} | __key__: {data_dict['__key__']} | dialog_str: {data_dict.get('dialog_str', '')}"
                )
                for key in ("input_ids", "token_mask", "attention_mask", "labels", "mm_token_type_ids"):
                    if key in data_dict:
                        data_dict[key] = data_dict[key][: self.max_token_length]
                return data_dict

        if input_ids.shape[-1] > self.max_token_length:
            msg = f"Input ids length {input_ids.shape[-1]} is greater than max token length {self.max_token_length} | __url__: path={data_dict['__url__'].path} root={data_dict['__url__'].root} | __key__: {data_dict['__key__']} | dialog_str: {data_dict.get('dialog_str', '')}"
            if "pixel_values" in data_dict:
                msg += f" | pixel_values: {data_dict['pixel_values'].shape}"
            if "pixel_values_videos" in data_dict:
                msg += f" | pixel_values_videos: {data_dict['pixel_values_videos'].shape}"
            if self.drop_over_max_length:
                self._strict_drop_count += 1
                if self._strict_drop_count <= 5 or self._strict_drop_count % 1000 == 0:
                    log.warning(
                        f"Strictly dropping over-length sample (drop_count={self._strict_drop_count}): {msg}",
                        rank0_only=False,
                    )
            else:
                log.critical(msg, rank0_only=False)
            return None
        return data_dict
