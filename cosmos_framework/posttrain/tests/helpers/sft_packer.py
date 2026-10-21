# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Offline HF fixtures: real byte-level tokenizer and real Qwen3-VL media expansion.

This intentionally small test template exercises ChatML structure, not a vendored model template.
Public-model metadata integration can separately check the checkpoint's official template.
"""

from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import Qwen2TokenizerFast, Qwen2VLImageProcessorFast, Qwen3VLProcessor, Qwen3VLVideoProcessor

TOKENS = [
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|image_pad|>",
    "<|video_pad|>",
    "<think>",
    "</think>",
]

CHAT_TEMPLATE = """
{%- macro content(value) -%}
{%- if value is string -%}{{ value }}
{%- else -%}
{%- for part in value -%}
{%- if part.type == 'text' -%}{{ part.text }}
{%- elif part.type == 'image' -%}{{ '<|vision_start|><|image_pad|><|vision_end|>' }}
{%- elif part.type == 'video' -%}{{ '<|vision_start|><|video_pad|><|vision_end|>' }}
{%- endif -%}
{%- endfor -%}
{%- endif -%}
{%- endmacro -%}
{%- if tools and messages[0].role != 'system' -%}
{{ '<|im_start|>system\n' }}{{ tools | tojson }}{{ '<|im_end|>\n' }}
{%- endif -%}
{%- for message in messages -%}
{%- if message.role == 'tool' -%}
{%- if loop.first or loop.previtem.role != 'tool' -%}{{ '<|im_start|>user\n' }}{%- endif -%}
{{ '<tool_response>' }}{{ content(message.content) }}{{ '</tool_response>' }}
{%- if loop.last or loop.nextitem.role != 'tool' -%}{{ '<|im_end|>\n' }}{%- endif -%}
{%- else -%}
{{ '<|im_start|>' + message.role + '\n' }}
{%- if message.role == 'assistant' and message.reasoning_content is defined and preserve_thinking -%}
{{ '<think>\n' + message.reasoning_content + '\n</think>\n\n' }}
{%- endif -%}
{{ content(message.content) }}
{%- if message.tool_calls -%}{{ '<tool_call>' }}{{ message.tool_calls | tojson }}{{ '</tool_call>' }}{%- endif -%}
{{ '<|im_end|>\n' }}
{%- endif -%}
{%- endfor -%}
"""


def local_tokenizer(*, merge_newlines: bool = False) -> Qwen2TokenizerFast:
    """A complete byte alphabet avoids unknowns without downloads or training randomness."""
    vocab = {value: i for i, value in enumerate(TOKENS + sorted(pre_tokenizers.ByteLevel.alphabet()))}
    merges = []
    if merge_newlines:
        vocab["ĊĊ"] = len(vocab)
        merges = [("Ċ", "Ċ")]
    backend = Tokenizer(models.BPE(vocab=vocab, merges=merges))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=not merge_newlines)
    backend.decoder = decoders.ByteLevel()
    tokenizer = Qwen2TokenizerFast(
        tokenizer_object=backend,
        pad_token=TOKENS[0],
        eos_token="<|im_end|>",
        additional_special_tokens=TOKENS[1:],
        chat_template=CHAT_TEMPLATE,
    )
    return tokenizer


def official_qwen3_vl_chat_template() -> str:
    """The Qwen3-VL processor template (chat_template.json of the public checkpoints).

    It renders reasoning_content as <think>\n...\n</think>\n on EVERY assistant turn, tool calls
    as <tool_call>\n{...}\n</tool_call> and tool replies inside a user block; the tokenizer-level
    template of the same checkpoints renders no reasoning at all, which the packer's boundary probe
    refuses. Saved verbatim so the masking contract is pinned against the template that trains.
    """
    return (Path(__file__).resolve().parents[1] / "fixtures" / "qwen3_vl_chat_template.jinja").read_text()


def official_qwen3_vl_tokenizer_chat_template() -> str:
    """The Qwen3-VL TOKENIZER template (``tokenizer_config.json``'s ``chat_template`` of the base of record, 5,292 chars,
    saved verbatim beside the processor template): what a tokenizer-only loader renders -- vLLM's chat path for a request
    that carries a tools array (the lab's dsh format record, review rounds 146 / 147). It renders no ``reasoning_content``
    on any assistant turn and keeps a ``<think></think>`` content prefix the processor template strips; content lists and
    tool calls render as the processor template renders them. The served prefix of a model call is this template's
    rendering of the history with ``add_generation_prompt=True``."""
    return (Path(__file__).resolve().parents[1] / "fixtures" / "qwen3_vl_tokenizer_chat_template.jinja").read_text()


def local_processor(tokenizer: Qwen2TokenizerFast, *, chat_template: str = CHAT_TEMPLATE) -> Qwen3VLProcessor:
    """Tiny patch geometry keeps real image/video preprocessing comfortably within L0."""
    image = Qwen2VLImageProcessorFast(
        patch_size=2,
        temporal_patch_size=2,
        merge_size=2,
        min_pixels=16,
        max_pixels=256,
    )
    video = Qwen3VLVideoProcessor(
        patch_size=2,
        temporal_patch_size=2,
        merge_size=2,
        size={"shortest_edge": 16, "longest_edge": 256},
        min_frames=2,
        max_frames=8,
        do_sample_frames=False,
    )
    return Qwen3VLProcessor(image, tokenizer, video, chat_template=chat_template)
