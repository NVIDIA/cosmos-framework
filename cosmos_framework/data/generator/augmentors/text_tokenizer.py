# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Augmentor for tokenizing input text

import json
import random
from collections.abc import Callable
from typing import Optional

import torch

from cosmos_framework.data.imaginaire.webdataset.augmentors.augmentor import Augmentor
from cosmos_framework.utils.lazy_config import instantiate as lazy_instantiate
from cosmos_framework.utils.generator.data_utils import read_positive_int_metadata

_MAX_NUM_TOKENS = 4096


def _tokenize_captions_separately(
    data_dict: dict,
    *,
    input_key: str,
    token_output_key: str,
    length_output_key: str,
    cfg_dropout_rate: float,
    tokenize: Callable[[str], list[int]],
) -> dict:
    """Tokenize one ordered caption payload per view while preserving ragged boundaries."""
    input_captions = data_dict.get(input_key)
    if not isinstance(input_captions, list) or not input_captions:
        raise ValueError(f"Separate text tokenization requires a non-empty caption list at {input_key!r}")
    sample_n_views = read_positive_int_metadata(data_dict, "sample_n_views", expected_count=1)
    if sample_n_views is None:
        raise ValueError("Separate text tokenization requires sample_n_views metadata")
    if len(input_captions) != sample_n_views[0]:
        raise ValueError(
            f"Separate text tokenization requires one caption per view: "
            f"captions={len(input_captions)}, sample_n_views={sample_n_views[0]}"
        )

    caption_texts: list[str] = []
    for caption in input_captions:
        if isinstance(caption, dict):
            caption_texts.append(json.dumps(caption))
        elif isinstance(caption, str):
            caption_texts.append(caption)
        else:
            raise TypeError(f"Separate text tokenization does not support caption type {type(caption).__name__}")

    # CFG remains a sample-level decision: either every view is conditioned or none is.
    if cfg_dropout_rate > 0 and random.random() < cfg_dropout_rate:
        caption_texts = [""] * len(caption_texts)

    data_dict[input_key] = caption_texts
    token_tensors = [torch.tensor(tokenize(caption), dtype=torch.long) for caption in caption_texts]  # [V][N_v]
    token_lengths = [int(tokens.shape[0]) for tokens in token_tensors]
    data_dict[token_output_key] = token_tensors
    data_dict[length_output_key] = token_lengths
    return data_dict


class TextTokenizerTransform(Augmentor):
    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)

        tokenizer_config = self.args["tokenizer_config"]
        self.cfg_dropout_rate = self.args["cfg_dropout_rate"]
        self.use_system_prompt = self.args.get("use_system_prompt", False)
        self.tokenize_separately: bool = self.args.get("tokenize_separately", False)

        if self.tokenize_separately and (self.output_keys is None or len(self.output_keys) < 2):
            raise ValueError("Separate text tokenization requires token and length output keys")

        self._processor = lazy_instantiate(tokenizer_config)

    def __call__(self, data_dict: dict) -> dict:
        if self.tokenize_separately:
            assert self.output_keys is not None
            return _tokenize_captions_separately(
                data_dict,
                input_key=self.input_keys[0],
                token_output_key=self.output_keys[0],
                length_output_key=self.output_keys[1],
                cfg_dropout_rate=self.cfg_dropout_rate,
                tokenize=lambda caption: self._processor.tokenize_text(
                    caption,
                    is_video=False,
                    use_system_prompt=self.use_system_prompt,
                )[:_MAX_NUM_TOKENS],
            )

        input_caption = data_dict[self.input_keys[0]]

        if isinstance(input_caption, dict):
            # Encode dict into a json string. This json string is then passed to the transformer tokenizer.
            input_caption = json.dumps(input_caption)
            data_dict[self.input_keys[0]] = input_caption

        if self.cfg_dropout_rate > 0:
            # If CFG is used, randomly dropout the input caption
            # We dropout the input caption by replacing it with an empty string
            if random.random() < self.cfg_dropout_rate:
                input_caption = ""
                data_dict[self.input_keys[0]] = input_caption

        text_ids = self._processor.tokenize_text(
            input_caption,
            is_video=False,
            use_system_prompt=self.use_system_prompt,
        )
        text_ids = text_ids[:_MAX_NUM_TOKENS]  # truncate the text ids to the maximum number of tokens
        # This will take care of wierd edge cases where we generate extremely long captions
        data_dict[self.output_keys[0]] = torch.tensor(text_ids)  # [N_tokens]
        return data_dict


_SYSTEM_PROMPT_IMAGE_EDITING = "You are a helpful assistant who will edit images based on the user's instructions."
_SYSTEM_PROMPT_VIDEO_EDITING = "You are a helpful assistant who will edit videos based on the user's instructions."

_SYSTEM_PROMPT_TRANSFER = "You are a helpful assistant that generates images or videos following the user's instructions and control signals (edge maps, blur, depth, or segmentation)."

_SYSTEM_PROMPTS = {
    "editing": _SYSTEM_PROMPT_IMAGE_EDITING,
    "video_editing": _SYSTEM_PROMPT_VIDEO_EDITING,
    "transfer": _SYSTEM_PROMPT_TRANSFER,
}


class TextTokenizerTransformForEditing(Augmentor):
    """Tokenizer augmentor for interleaved tasks: image editing or transfer (control-conditioned generation).

    Uses a task-specific system prompt. Pass args["task"] = "editing" (default) or "transfer".
    """

    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)

        tokenizer_config = self.args["tokenizer_config"]
        self.cfg_dropout_rate = self.args.get("cfg_dropout_rate", 0.0)
        self.tokenize_separately: bool = self.args.get("tokenize_separately", False)
        task = self.args.get("task", "editing")
        self._system_prompt = _SYSTEM_PROMPTS.get(task, _SYSTEM_PROMPTS["editing"])

        if self.tokenize_separately and (self.output_keys is None or len(self.output_keys) < 2):
            raise ValueError("Separate text tokenization requires token and length output keys")

        self._processor = lazy_instantiate(tokenizer_config)

    def __call__(self, data_dict: dict) -> dict | None:
        if self.tokenize_separately:
            assert self.output_keys is not None
            return _tokenize_captions_separately(
                data_dict,
                input_key=self.input_keys[0],
                token_output_key=self.output_keys[0],
                length_output_key=self.output_keys[1],
                cfg_dropout_rate=self.cfg_dropout_rate,
                tokenize=lambda caption: self._processor.tokenize_text(caption, system_prompt=self._system_prompt),
            )

        input_caption = data_dict.get(self.input_keys[0], "")
        if isinstance(input_caption, dict):
            input_caption = json.dumps(input_caption)
            data_dict[self.input_keys[0]] = input_caption
        if self.cfg_dropout_rate > 0 and random.random() < self.cfg_dropout_rate:
            input_caption = ""
            data_dict[self.input_keys[0]] = input_caption
        text_ids = self._processor.tokenize_text(input_caption, system_prompt=self._system_prompt)
        data_dict[self.output_keys[0]] = torch.tensor(text_ids)  # [N_tokens]
        return data_dict


class TextTokenizerTransformForTransfer(TextTokenizerTransformForEditing):
    """Tokenizer augmentor for transfer (control-conditioned) generation. Uses transfer system prompt."""

    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        args = dict(args) if args else {}
        args["task"] = "transfer"
        super().__init__(input_keys, output_keys, args)
