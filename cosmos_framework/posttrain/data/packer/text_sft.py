# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Explicit text-template spans, sharing SFT's rendering, offsets and collation.

No prefix token-length subtraction or generic assistant-mask assumption. Each
render is traced by the existing source-boundary validator; tokenization must
round-trip exactly before the same offset-based loss-mask implementation runs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from cosmos_framework.posttrain.data.packer.sft import QwenSFTDataPacker, _blocks, masking_contract

_LLAMA = re.compile(r"<\|start_header_id\|>(system|user|assistant)<\|end_header_id\|>\n\n(.*?)<\|eot_id\|>", re.DOTALL)
_GEMMA = re.compile(r"<start_of_turn>(user|model)\n(.*?)<end_of_turn>", re.DOTALL)
_RESERVED_TEXT = (
    "[INST]",
    "[/INST]",
    "<s>",
    "</s>",
    "<bos>",
    "<eos>",
    "<start_of_turn>",
    "<end_of_turn>",
    "<|begin_of_text|>",
    "<|end_of_text|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|eot_id|>",
    "[AVAILABLE_TOOLS]",
    "[/AVAILABLE_TOOLS]",
    "[TOOL_CALLS]",
    "[TOOL_RESULTS]",
    "[/TOOL_RESULTS]",
)


@dataclass(frozen=True)
class TextBlock:
    role: str
    body: str
    begin: int
    body_begin: int
    finish: int

    def __getitem__(self, group):
        return {1: self.role, 2: self.body}[group]

    def start(self, group=0):
        return self.body_begin if group == 2 else self.begin

    def end(self):
        return self.finish


def parse_text_blocks(text: str, format: str, bos: str | None):
    if format == "chatml":
        return _blocks(text)
    offset = len(bos) if bos and text.startswith(bos) else 0
    blocks = []
    if format == "mistral_inst":
        # This bounded profile supports alternating user/assistant conversations.
        # System prompts coalesced into an instruction and tool syntax are refused.
        position = offset
        while position < len(text):
            if not text.startswith("[INST]", position):
                raise ValueError("Mistral SFT requires [INST] user boundaries")
            split = text.find("[/INST]", position + 6)
            end = text.find("</s>", split + 7) if split >= 0 else -1
            if split < 0 or end < 0:
                raise ValueError("Mistral SFT requires a complete instruction and assistant EOS")
            blocks.append(TextBlock("user", text[position + 6 : split], position, position + 6, split + 7))
            blocks.append(TextBlock("assistant", text[split + 7 : end], split + 7, split + 7, end + 4))
            position = end + 4
        if not blocks:
            raise ValueError("Mistral SFT conversation is empty")
        return blocks
    pattern = {"llama3_headers": _LLAMA, "gemma_turns": _GEMMA}[format]
    end = offset
    for match in pattern.finditer(text, offset):
        if text[end : match.start()] not in ("", "\n"):
            raise ValueError(f"Unsupported {format} SFT text outside message boundaries")
        role = "assistant" if match[1] == "model" else match[1]
        blocks.append(TextBlock(role, match[2], match.start(), match.start(2), match.end()))
        end = match.end()
    if not blocks or text[end:] not in ("", "\n"):
        raise ValueError(f"Incomplete {format} SFT conversation")
    return blocks


class TextSFTDataPacker(QwenSFTDataPacker):
    def __init__(self, tokenizer, **kwargs):
        super().__init__(tokenizer, **kwargs)
        if self.processor is not None or self.processor_kwargs:
            raise ValueError("Text-template SFT accepts text inputs only")
        rendered = self._render(
            [{"role": "user", "content": "SFT_USER_CHECK"}, {"role": "assistant", "content": "SFT_ASSISTANT_CHECK"}],
            None,
        )
        matches = []
        for candidate in ("chatml", "llama3_headers", "gemma_turns", "mistral_inst"):
            try:
                blocks = parse_text_blocks(rendered, candidate, tokenizer.bos_token)
                if [b[1] for b in blocks] in (["user", "assistant"], ["system", "user", "assistant"]):
                    matches.append(candidate)
            except ValueError:
                pass
        if len(matches) != 1:
            raise ValueError(
                "Unsupported/ambiguous text SFT template; supported: ChatML, Llama3 headers, Gemma turns, Mistral [INST]"
            )
        self.text_format = matches[0]

    def _conversation(self, sample):
        messages = super()._conversation(sample)

        def reject(value):
            if isinstance(value, str) and any(marker in value for marker in _RESERVED_TEXT):
                raise ValueError("SFT source contains reserved text-template delimiters")
            if isinstance(value, dict):
                for key, item in value.items():
                    reject(key)
                    reject(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    reject(item)

        reject(sample.messages)
        reject(sample.tools)
        if sample.tools or any(m.get("role") == "tool" or m.get("tool_calls") for m in sample.messages):
            raise ValueError(
                "This text SFT profile supports user/assistant text and optional system messages; tools need their own template qualification"
            )
        if self.text_format == "mistral_inst" and any(m.get("role") == "system" for m in sample.messages):
            raise ValueError(
                "Mistral [INST] SFT supports alternating user/assistant rows; coalesced system prompts are unsupported"
            )
        return messages

    def _parse_blocks(self, text):
        return parse_text_blocks(text, self.text_format, self.tokenizer.bos_token)

    def _row_masking_contract(self, supervise, opener):
        # Reuse validation of per-row supervision/opener values, with a distinct
        # identity so exact resume cannot confuse different template protocols.
        masking_contract(supervise, opener)
        return f"hf_text_{self.text_format}_{supervise}_{opener}_bodies_and_end_v1"

    def validation_semantics_identity(self):
        result = super().validation_semantics_identity()
        result["masking"] = f"hf_text_{self.text_format}_all_assistant_bodies_and_end_v1"
        result["text_template_format"] = self.text_format
        return result
