# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Visual-Text Transformations or Augmentations."""

import random
import re
from typing import Any, Literal

from cosmos_framework.data.imaginaire.webdataset.augmentors.augmentor import Augmentor

_THINK_RE = re.compile(r"<think>.*?</think>\s*|<think>.*\Z", re.DOTALL)


class PromptFormat(Augmentor):
    def __init__(
        self,
        input_keys: list[str] = ["texts"],
        text_chat_order: Literal["text_end", "text_start", "random"] = "text_end",
        strip_thinking_prob: float = 0.0,
    ) -> None:
        """
        Args:
            input_keys: List of input keys.
            text_chat_order: Order of text items in user messages.
            strip_thinking_prob: Per-sample probability of dropping assistant thinking traces.
        """
        if not 0.0 <= strip_thinking_prob <= 1.0:
            raise ValueError(f"strip_thinking_prob must be in [0, 1], got {strip_thinking_prob}")
        self.input_keys = input_keys
        self.text_chat_order = text_chat_order
        self.strip_thinking_prob = strip_thinking_prob

    def __call__(self, data_dict: dict[str, Any]) -> dict[str, Any] | None:
        conversation_key = self.input_keys[0]

        # retrive conversations from dict
        try:
            list_of_conversation = data_dict[conversation_key]
        except KeyError:
            url = data_dict["__url__"].root + "/" + data_dict["__url__"].path
            print(f"KeyError: {conversation_key} not found in data_dict for url: {url}")
            return None

        # check if this is list of list of dict or list of dict

        if isinstance(list_of_conversation[0], list):
            selected_conversation = random.sample(list_of_conversation, 1)[0]
        elif isinstance(list_of_conversation[0], dict):
            selected_conversation = list_of_conversation
        else:
            raise ValueError(
                f"list_of_conversation is not a list of list of dict or list of dict: {list_of_conversation}"
            )

        # Now it should be list of dict
        assert isinstance(selected_conversation, list) and isinstance(selected_conversation[0], dict), (
            f"selected_conversation is not a list of dict: {selected_conversation}"
        )
        # Normalize all string content to list format
        for message in selected_conversation:
            if "content" in message and isinstance(message["content"], str):
                message["content"] = [{"type": "text", "text": message["content"]}]
            if "reasoning_content" in message and isinstance(message["reasoning_content"], str):
                message["reasoning_content"] = [{"type": "text", "text": message["reasoning_content"]}]

        is_thinking_stripped = False
        if random.random() < self.strip_thinking_prob:
            for message in selected_conversation:
                if message.get("role") != "assistant":
                    continue
                if message.pop("reasoning_content", None):
                    is_thinking_stripped = True
                content = message.get("content", [])
                for item in content:
                    if not isinstance(item, dict) or item.get("type") != "text":
                        continue
                    text = item.get("text")
                    if not isinstance(text, str):
                        continue
                    stripped_text = _THINK_RE.sub("", text).lstrip()
                    if stripped_text != text:
                        is_thinking_stripped = True
                    item["text"] = stripped_text
                has_text = any(
                    isinstance(item, dict)
                    and item.get("type") == "text"
                    and isinstance(item.get("text"), str)
                    and item["text"].strip()
                    for item in content
                )
                has_media = any(
                    isinstance(item, dict) and item.get("type") in ("image", "video", "audio") for item in content
                )
                if not has_text and not has_media:
                    return None
        else:
            # Merge reasoning_content into assistant message content
            for message in selected_conversation:
                if message.get("role") == "assistant" and message.get("reasoning_content"):
                    # Wrap reasoning items in <think>...</think> tags
                    reasoning_items = message["reasoning_content"]
                    think_start = [{"type": "text", "text": "<think>\n"}]
                    think_end = [{"type": "text", "text": "\n</think>\n\n"}]
                    message["content"] = think_start + reasoning_items + think_end + message["content"]
                    del message["reasoning_content"]

        data_dict["is_thinking_stripped"] = is_thinking_stripped
        data_dict["conversation"] = selected_conversation

        del data_dict[conversation_key]

        # # enforce chat order
        # self._enforce_text_chat_order(selected_conversation)

        return data_dict

    def _enforce_text_chat_order(self, conversation: list[dict[str, Any]]) -> None:
        """
        Reorder text content within user messages based on text_chat_order setting.
        NOTE (maxzhaoshuol): this does NOT work for interleaved data!!!!!!

        Args:
            conversation: List of message dictionaries
        """
        for message in conversation:
            if message.get("role") == "user" and "content" in message:
                content = message["content"]
                if isinstance(content, list):
                    # Separate text items from non-text items
                    text_items = [item for item in content if item.get("type") == "text"]
                    non_text_items = [item for item in content if item.get("type") != "text"]

                    if text_items:
                        # Reorder based on text_chat_order
                        if self.text_chat_order == "text_start":
                            # Put text items at the beginning
                            message["content"] = text_items + non_text_items
                        elif self.text_chat_order == "text_end":
                            # Put text items at the end
                            message["content"] = non_text_items + text_items
                        elif self.text_chat_order == "random":
                            print("random")
                            # Randomly put text items at beginning or end
                            if random.random() < 0.5:
                                message["content"] = text_items + non_text_items
                            else:
                                message["content"] = non_text_items + text_items
