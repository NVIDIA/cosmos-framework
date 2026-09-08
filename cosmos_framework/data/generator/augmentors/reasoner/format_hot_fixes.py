# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from collections.abc import Callable

from cosmos_framework.data.imaginaire.webdataset.augmentors.augmentor import Augmentor

# ---------------------------------------------------------------------------
# Pass-through default and content applicator
# ---------------------------------------------------------------------------


def _apply_to_content(content: str | list[object], fn: Callable[[str], str]) -> str | list[object]:
    if isinstance(content, str):
        return fn(content)
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                item["text"] = fn(item["text"])
    return content


def _text_content_has_any_term(content: object, terms: tuple[str, ...]) -> bool:
    if isinstance(content, str):
        lowered_content = content.lower()
        return any(term in lowered_content for term in terms)
    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict) or item.get("type") != "text":
                continue
            text = item.get("text")
            if isinstance(text, str) and _text_content_has_any_term(text, terms):
                return True
    return False


# ---------------------------------------------------------------------------
# HotfixBase — subclass and override methods to define per-dataset fixes
# ---------------------------------------------------------------------------


class HotfixBase:
    pattern: str = ""

    def fix_system_prompt(self, text: str) -> str:
        return text

    def fix_user_prompt(self, text: str) -> str:
        return text

    def fix_assistant_prompt(self, text: str) -> str:
        return text

    def should_drop_content(self, content: object) -> bool:
        return False


# ---------------------------------------------------------------------------
# Dataset-specific hotfixes
# ---------------------------------------------------------------------------


class _NemotronCascadeInstructionFollowing(HotfixBase):
    pattern = "nemotron_cascade_2_sft/instruction_following"
    blocked_terms: tuple[str, ...] = ("nvidia", "language model", "nemotron")

    def should_drop_content(self, content: object) -> bool:
        return _text_content_has_any_term(content, self.blocked_terms)


# ---------------------------------------------------------------------------
# Global registry — add dataset-specific hotfix instances here
# ---------------------------------------------------------------------------

HOTFIXES: list[HotfixBase] = [
    _NemotronCascadeInstructionFollowing(),
]

# ---------------------------------------------------------------------------
# Augmentor
# ---------------------------------------------------------------------------


class FormatHotFixes(Augmentor):
    def __init__(
        self,
        conversation_key: str = "conversation",
        hotfixes: list[HotfixBase] | None = None,
    ) -> None:
        self.conversation_key = conversation_key
        self.hotfixes = hotfixes if hotfixes is not None else HOTFIXES

    def _url_str(self, data_dict: dict[str, object]) -> str:
        url = data_dict.get("__url__")
        if url is None:
            return ""
        return url.root if hasattr(url, "root") else str(url)

    def _matching_hotfix(self, url_str: str) -> HotfixBase | None:
        return next((hf for hf in self.hotfixes if hf.pattern in url_str), None)

    def __call__(self, data_dict: dict[str, object]) -> dict[str, object] | None:
        url_str = self._url_str(data_dict)
        hf = self._matching_hotfix(url_str)
        if hf is None:
            return data_dict

        conversation = data_dict.get(self.conversation_key)
        if not isinstance(conversation, list):
            return data_dict

        for message in conversation:
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            if role == "system":
                fix_prompt = hf.fix_system_prompt
            elif role == "user":
                fix_prompt = hf.fix_user_prompt
            elif role == "assistant":
                fix_prompt = hf.fix_assistant_prompt
            else:
                continue

            if hf.should_drop_content(message["content"]):
                return None
            message["content"] = _apply_to_content(message["content"], fix_prompt)

        return data_dict
