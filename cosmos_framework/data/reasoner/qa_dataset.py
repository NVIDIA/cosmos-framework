# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Dependency-free conversation records for native video-reasoning annotations.

The structural dataset returns chat messages through ``__len__`` and
``__getitem__``. Framework's map-style training adapter adds sampling and
interleaves answer/reasoning targets without changing the source records.
"""

import bisect
import json
import os
from typing import Any, Literal, Optional, Union

# Byte-identical to the chat templates shipped with `nvidia/Cosmos-Reason2-8B`
# and `Qwen/Qwen3-VL-8B-Instruct`. Used to override the Qwen3-VL-Thinking
# processor so the assistant content emitted here is rendered verbatim.
# Qwen3-VL-Thinking's stock template auto-injects an empty
# `<think>\n\n</think>\n\n` shell into the last assistant turn, which collides
# with assistant-token loss masking. Keep the Instruct template so explicitly
# supplied reasoning targets are rendered without an injected empty shell.
QWEN3VL_INSTRUCT_CHAT_TEMPLATE = "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0].role == 'system' %}\n        {%- if messages[0].content is string %}\n            {{- messages[0].content }}\n        {%- else %}\n            {%- for content in messages[0].content %}\n                {%- if 'text' in content %}\n                    {{- content.text }}\n                {%- endif %}\n            {%- endfor %}\n        {%- endif %}\n        {{- '\\n\\n' }}\n    {%- endif %}\n    {{- \"# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0].role == 'system' %}\n        {{- '<|im_start|>system\\n' }}\n        {%- if messages[0].content is string %}\n            {{- messages[0].content }}\n        {%- else %}\n            {%- for content in messages[0].content %}\n                {%- if 'text' in content %}\n                    {{- content.text }}\n                {%- endif %}\n            {%- endfor %}\n        {%- endif %}\n        {{- '<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- set image_count = namespace(value=0) %}\n{%- set video_count = namespace(value=0) %}\n{%- for message in messages %}\n    {%- if message.role == \"user\" %}\n        {{- '<|im_start|>' + message.role + '\\n' }}\n        {%- if message.content is string %}\n            {{- message.content }}\n        {%- else %}\n            {%- for content in message.content %}\n                {%- if content.type == 'image' or 'image' in content or 'image_url' in content %}\n                    {%- set image_count.value = image_count.value + 1 %}\n                    {%- if add_vision_id %}Picture {{ image_count.value }}: {% endif -%}\n                    <|vision_start|><|image_pad|><|vision_end|>\n                {%- elif content.type == 'video' or 'video' in content %}\n                    {%- set video_count.value = video_count.value + 1 %}\n                    {%- if add_vision_id %}Video {{ video_count.value }}: {% endif -%}\n                    <|vision_start|><|video_pad|><|vision_end|>\n                {%- elif 'text' in content %}\n                    {{- content.text }}\n                {%- endif %}\n            {%- endfor %}\n        {%- endif %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {{- '<|im_start|>' + message.role + '\\n' }}\n        {%- if message.content is string %}\n            {{- message.content }}\n        {%- else %}\n            {%- for content_item in message.content %}\n                {%- if 'text' in content_item %}\n                    {{- content_item.text }}\n                {%- endif %}\n            {%- endfor %}\n        {%- endif %}\n        {%- if message.tool_calls %}\n            {%- for tool_call in message.tool_calls %}\n                {%- if (loop.first and message.content) or (not loop.first) %}\n                    {{- '\\n' }}\n                {%- endif %}\n                {%- if tool_call.function %}\n                    {%- set tool_call = tool_call.function %}\n                {%- endif %}\n                {{- '<tool_call>\\n{\"name\": \"' }}\n                {{- tool_call.name }}\n                {{- '\", \"arguments\": ' }}\n                {%- if tool_call.arguments is string %}\n                    {{- tool_call.arguments }}\n                {%- else %}\n                    {{- tool_call.arguments | tojson }}\n                {%- endif %}\n                {{- '}\\n</tool_call>' }}\n            {%- endfor %}\n        {%- endif %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if loop.first or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {%- if message.content is string %}\n            {{- message.content }}\n        {%- else %}\n            {%- for content in message.content %}\n                {%- if content.type == 'image' or 'image' in content or 'image_url' in content %}\n                    {%- set image_count.value = image_count.value + 1 %}\n                    {%- if add_vision_id %}Picture {{ image_count.value }}: {% endif -%}\n                    <|vision_start|><|image_pad|><|vision_end|>\n                {%- elif content.type == 'video' or 'video' in content %}\n                    {%- set video_count.value = video_count.value + 1 %}\n                    {%- if add_vision_id %}Video {{ video_count.value }}: {% endif -%}\n                    <|vision_start|><|video_pad|><|vision_end|>\n                {%- elif 'text' in content %}\n                    {{- content.text }}\n                {%- endif %}\n            {%- endfor %}\n        {%- endif %}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n{%- endif %}\n"


ResponseMode = Literal["think", "answer", "hybrid"]
"""Strategy for building the assistant response string from each item.

- ``"answer"`` -> ``{answer}``.
- ``"think"`` -> ``<think>\\n{reasoning}\\n</think>\\n\\n{answer}``; falls back
  to ``{answer}`` when the item has no ``reasoning``.
- ``"hybrid"`` -> dataset length is doubled. Indices in the first half emit
  the answer-only form, indices in the second half emit the think form. The
  mapping is deterministic and stateless.
"""


def apply_chat_template_override(processor: Any) -> None:
    """Replace ``processor.chat_template`` with :data:`QWEN3VL_INSTRUCT_CHAT_TEMPLATE`.

    Idempotent. Also patches ``processor.tokenizer.chat_template`` when present
    so direct tokenizer-level renderings (e.g. ``tokenizer.apply_chat_template``)
    use the same template.
    """
    processor.chat_template = QWEN3VL_INSTRUCT_CHAT_TEMPLATE
    inner = getattr(processor, "tokenizer", None)
    if inner is not None:
        inner.chat_template = QWEN3VL_INSTRUCT_CHAT_TEMPLATE


def build_response(answer: str, reasoning: str, mode: ResponseMode) -> str:
    """Render the assistant string from ``answer`` / ``reasoning`` under the given resolved ``mode``.

    ``mode`` must be ``"think"`` or ``"answer"``; ``"hybrid"`` is routed by
    :meth:`ReasoningConversationDataset.__getitem__` and never
    reaches this function.
    """
    if mode not in {"answer", "think"}:
        raise ValueError(f"Unknown response mode at conversation level: {mode!r}")
    if mode == "answer" or not reasoning:
        return answer
    if mode == "think":
        return f"<think>\n{reasoning}\n</think>\n\n{answer}"
    raise ValueError(f"Unknown response mode at conversation level: {mode!r}")


class ReasoningConversationDataset:
    """Concatenated conversation dataset over one or more cosmos-video-reasoning-v1.0 files.

    This class is **not** a ``torch.utils.data.Dataset``; downstream training
    code wraps it (typically via composition inside its own ``Dataset``
    subclass) or indexes it directly.

    Contract:

    - ``__len__()`` is the total item count across all annotation files.
    - ``__getitem__(i)`` returns a chat-style conversation list::

          [
              {"role": "system",    "content": <str>},                 # optional
              {"role": "user",      "content": [
                  {"type": "video"|"image", "video"|"image": <str>, **vision_kwargs},
                  {"type": "text",  "text": <str>},
              ]},
              {"role": "assistant", "content": <str>},
          ]

    Args:
        annotation_paths: Paths to JSON files conforming to the
            cosmos-video-reasoning-v1.0 schema. Items are concatenated in given order.
        media_roots: Per-file media base path. ``None`` (default) honors each
            file's own ``media_root`` field. A single ``str`` overrides every
            file's ``media_root``. A ``list[str]`` overrides per-file and must
            match ``annotation_paths`` length.
        system_prompt: Optional system prompt prepended to each conversation.
        vision_kwargs: Per-message vision options merged into each media
            content dict (e.g. ``{"fps": 1, "max_pixels": 81920}``).
        response_mode: See :data:`ResponseMode`.
    """

    def __init__(
        self,
        annotation_paths: list[str],
        media_roots: Optional[Union[str, list[str]]] = None,
        system_prompt: str = "",
        vision_kwargs: Optional[dict] = None,
        response_mode: ResponseMode = "answer",
    ) -> None:
        if response_mode not in {"answer", "think", "hybrid"}:
            raise ValueError(f"Unknown response mode: {response_mode!r}")
        annotation_paths = [annotation_paths] if isinstance(annotation_paths, str) else list(annotation_paths)
        if not annotation_paths:
            raise ValueError("annotation_paths must be a non-empty list of paths")
        resolved_roots: list[Optional[str]]
        if media_roots is None or isinstance(media_roots, str):
            resolved_roots = [media_roots] * len(annotation_paths)
        elif isinstance(media_roots, list):
            if len(media_roots) != len(annotation_paths):
                raise ValueError("media_roots list length must match annotation_paths length")
            resolved_roots = list(media_roots)
        else:
            raise ValueError(f"Invalid media_roots: {media_roots!r}")

        self.cumulative_lengths: list[int] = [0]
        self.annotations: list[dict] = []
        for annotation_path, media_root in zip(annotation_paths, resolved_roots, strict=True):
            with open(annotation_path) as f:
                annotation = json.load(f)
            if media_root is None:
                media_root = annotation.get("media_root") or ""
                annotation["media_root"] = os.path.join(os.path.dirname(os.path.abspath(annotation_path)), media_root)
            else:
                annotation["media_root"] = media_root
            self.annotations.append(annotation)
            self.cumulative_lengths.append(self.cumulative_lengths[-1] + len(annotation["items"]))

        self.system_prompt = system_prompt
        self.vision_kwargs = vision_kwargs or {}
        self.response_mode: ResponseMode = response_mode

    @property
    def _raw_length(self) -> int:
        """Total raw item count across all annotation files."""
        return self.cumulative_lengths[-1]

    def __len__(self) -> int:
        # Hybrid doubles the dataset: first half answer-only, second half think.
        return self._raw_length * 2 if self.response_mode == "hybrid" else self._raw_length

    def __getitem__(self, index: int) -> list[dict]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        if self.response_mode == "hybrid":
            raw = self._raw_length
            if index < raw:
                return self._build_conversation(index, "answer")
            return self._build_conversation(index - raw, "think")
        return self._build_conversation(index, self.response_mode)

    def _build_conversation(self, raw_index: int, mode: ResponseMode) -> list[dict]:
        """Build a conversation for the ``raw_index``-th item across all
        annotation files, using the given response ``mode``.

        ``mode`` is the *resolved* mode for this single conversation — it must
        be ``"think"`` or ``"answer"``. ``"hybrid"`` is routed by
        :meth:`__getitem__` and never reaches this method.
        """
        dataset_idx = bisect.bisect_right(self.cumulative_lengths, raw_index) - 1
        annotation = self.annotations[dataset_idx]
        idx_in_annotation = raw_index - self.cumulative_lengths[dataset_idx]
        item = annotation["items"][idx_in_annotation]
        media_root = annotation.get("media_root") or ""

        user_content: list[dict] = []
        if item.get("video_id"):
            user_content.append(
                {
                    "type": "video",
                    "video": os.path.join(media_root, item["video_id"]),
                    **self.vision_kwargs,
                }
            )
        elif item.get("image_id"):
            raise ValueError("Image is not supported yet")
        user_content.append({"type": "text", "text": item["question"]})

        conversation: list[dict] = []
        if self.system_prompt:
            conversation.append({"role": "system", "content": self.system_prompt})
        conversation.append({"role": "user", "content": user_content})

        answer = item.get("answer", "")
        reasoning = item.get("reasoning", "")
        conversation.append(
            {
                "role": "assistant",
                "content": build_response(answer, reasoning, mode),
            }
        )
        return conversation
