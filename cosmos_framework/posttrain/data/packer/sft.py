# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""CPU-only, full-conversation SFT for Qwen ChatML text and Qwen3-VL processors.

The official template owns reasoning/tool rendering. HF assistant masks are deliberately not
used: their character spans can precede image/video expansion. Instead, decode the FINAL ids,
require an exact fast-tokenizer round trip, and use its offsets to mask validated ChatML bodies.
No prefix re-render lengths, truncation or model weights. Every assistant turn is supervised unless the row's
``SFTTarget.supervise`` is ``"last_assistant"``: then the last assistant turn only, the earlier ones context at -100.
"""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch
from packaging.version import Version
from transformers import AutoProcessor, AutoTokenizer, PretrainedConfig
from transformers import __version__ as transformers_version

from cosmos_framework.posttrain.data.packer.base import DataPacker
from cosmos_framework.posttrain.schema import (
    SFT_OPENER_PREFILLED,
    SFT_OPENER_SELF,
    SFT_SUPERVISE_ALL,
    SFT_SUPERVISE_LAST,
    ObsBundle,
    SFTTarget,
)

_IGNORE = -100
_CHAT = re.compile(r"<\|im_start\|>(system|user|assistant)\n(.*?)<\|im_end\|>", re.DOTALL)
_VIDEO_GROUP = r"(?:<[0-9]+(?:\.[0-9]+)? seconds>)?<\|vision_start\|>(?:<\|video_pad\|>)+<\|vision_end\|>"
_MEDIA = re.compile(
    r"<\|vision_start\|>(?:<\|image_pad\|>)+<\|vision_end\|>"
    # Transformers >= 5.13 keeps the template's outer vision markers around the timestamped frame groups;
    # 5.8.1 replaces the whole placeholder with the groups. Both are one media span.
    rf"|<\|vision_start\|>(?:{_VIDEO_GROUP})+<\|vision_end\|>"
    rf"|{_VIDEO_GROUP}"
)
_RESERVED = (
    "<|im_start|>",
    "<|im_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|image_pad|>",
    "<|video_pad|>",
    "<|placeholder|>",
    "<|endoftext|>",
)
_TEXT_TYPES = {"qwen2", "qwen2_moe", "qwen3", "qwen3_moe"}
# Families whose checkpoints ship ``Qwen3VLProcessor``: dense Qwen3.5/3.8 (trained through the HF
# backend) and Qwen3-VL / Qwen3-VL-MoE (trained through the native policy seat, ``model/qwen3_vl.py``;
# the HF backend still refuses MoE VLMs at ``model/hf.py::_is_moe_vlm``). Other VLM families need
# their own qualification before they are listed here.
_VLM_TYPES = {"qwen3_5", "qwen3_vl", "qwen3_vl_moe"}
_MEDIA_KEYS = {"pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"}
# The masking contract string per ``SFTTarget.supervise`` value. The all-assistant string is also the packer-level
# ``validation_semantics_identity()["masking"]`` (unchanged: it names the rule for rows without the field, so no existing
# run's resume identity moves); a row's own string travels on ``SFTPolicyInput.masking``, and the field itself is part of
# the row, hence of the source fingerprint.
SUPERVISE_CONTRACTS = {
    SFT_SUPERVISE_ALL: "qwen_chatml_all_assistant_bodies_and_end_v2",
    SFT_SUPERVISE_LAST: "qwen_chatml_last_assistant_body_and_end_v1",
}
# The per-row opener rule's suffix (row rule R4): ``opener="prefilled"`` masks the first ``<think>`` token of
# every supervised assistant block of that row, the rule the run-level ``mask_think_opener`` applies to every row; the
# run-level knob stays in the packer identity, the row's rule in its contract string.
OPENER_CONTRACT_SUFFIXES = {SFT_OPENER_SELF: "", SFT_OPENER_PREFILLED: "_opener_prefilled_v1"}


@dataclass
class SFTPolicyInput:
    """One unpadded sequence; labels are UNSHIFTED and ignored positions are -100.

    ``masking`` is the row's contract string (``SUPERVISE_CONTRACTS[sample.supervise]``), a record, not a collate input."""

    input_ids: list[int]
    labels: list[int]
    mm_kwargs: dict[str, Any] | None = None
    masking: str = SUPERVISE_CONTRACTS[SFT_SUPERVISE_ALL]


def supervised_assistant_blocks(assistant_blocks: int, supervise: str) -> list[int]:
    """Indices, in stream order among the ASSISTANT blocks only, of the blocks that carry loss under ``supervise``: every
    index under ``all_assistant``, the last index alone under ``last_assistant``. The packer paints exactly these blocks;
    a row validator that counts assistant tokens independently of the labels (per block: body fragments tokenized on
    their own plus the block's ``<|im_end|>``) must count over the same indices, or its per-block comparison is wrong
    for a ``last_assistant`` row."""
    if supervise not in SUPERVISE_CONTRACTS:
        raise ValueError(f"SFT supervise must be one of {sorted(SUPERVISE_CONTRACTS)}, got {supervise!r}")
    if not isinstance(assistant_blocks, int) or assistant_blocks < 1:
        raise ValueError("SFT conversation has no assistant target")
    return list(range(assistant_blocks)) if supervise == SFT_SUPERVISE_ALL else [assistant_blocks - 1]


def masking_contract(supervise: str, opener: str) -> str:
    """The row's masking contract string: the supervision rule's string plus the opener rule's suffix."""
    if supervise not in SUPERVISE_CONTRACTS:
        raise ValueError(f"SFT supervise must be one of {sorted(SUPERVISE_CONTRACTS)}, got {supervise!r}")
    if opener not in OPENER_CONTRACT_SUFFIXES:
        raise ValueError(f"SFT opener must be one of {sorted(OPENER_CONTRACT_SUFFIXES)}, got {opener!r}")
    return SUPERVISE_CONTRACTS[supervise] + OPENER_CONTRACT_SUFFIXES[opener]


def _reject_delimiters(value: Any) -> None:
    """Media payloads remain opaque; recursively check textual message/tool metadata."""
    if isinstance(value, str):
        if any(token in value for token in _RESERVED):
            raise ValueError("SFT source contains reserved chat/media delimiters")
    elif isinstance(value, dict):
        for key, item in value.items():
            _reject_delimiters(key)
            _reject_delimiters(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_delimiters(item)
    elif isinstance(value, torch.Tensor) and value.device.type != "cpu":
        raise ValueError("SFT media inputs must be on CPU")


def _template_variables(template: str) -> frozenset[str]:
    """The Jinja variables of a chat template, as Transformers 5 classifies render kwargs.

    ``ProcessorMixin.apply_chat_template`` (``processing_utils.py``) treats a kwarg outside this set as a processor
    kwarg, not a template variable: it is dropped from the render and a warning is logged on EVERY call. The packer
    asks the same cached helper the classifier uses, so the two cannot disagree. Imported here, not at module import:
    the packer module is imported by config preflight before any template is rendered.
    """
    from transformers.utils.chat_template_utils import _get_template_variables

    return frozenset(_get_template_variables(template))


def _blocks(text: str) -> list[re.Match[str]]:
    blocks = list(_CHAT.finditer(text))
    end = 0
    for block in blocks:
        if text[end : block.start()] not in ("", "\n"):
            raise ValueError("Unsupported SFT template: expected Qwen ChatML message boundaries")
        if "<|im_start|>" in block[2] or "<|im_end|>" in block[2]:
            raise ValueError("Ambiguous nested ChatML message boundary")
        end = block.end()
    if not blocks or text[end:] not in ("", "\n"):
        raise ValueError("Unsupported SFT template: incomplete or non-ChatML conversation")
    return blocks


class QwenSFTDataPacker(DataPacker):
    """SFT-only packer; its builder selects a tokenizer or Qwen3VLProcessor from metadata.

    ``processor_kwargs`` are HF's nested ``images_kwargs`` / ``videos_kwargs`` (plural),
    passed to the processor call, not its loader or Jinja renderer. Media must use HF content
    parts: ``type=text/image/video`` with ``text`` or one of ``image/video/url/path``.
    Video sampling, sizing and metadata are owned by the HF video processor. Unsupported
    side-input layouts and non-ChatML templates fail rather than guessing collation/masks.

    Assistant content that is ``None``, ``""``, whitespace-only text or a list of empty/whitespace text
    parts is rendered as NO content part (a tool-call-only turn renders ``<|im_start|>assistant\n<tool_call>``
    and, with reasoning, ``</think>\n<tool_call>``), matching how the Qwen3-VL template renders an
    empty-content history turn at serving time. The collated ``attention_mask`` is ``torch.int64`` 0/1 (``main``'s
    collator contract; the native eager mask path casts a 2-D mask to bool itself).
    """

    def __init__(
        self,
        tokenizer: Any,
        *,
        max_length: int,
        processor: Any = None,
        processor_kwargs: dict[str, Any] | None = None,
        mask_think_opener: bool = False,
    ) -> None:
        if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length <= 0:
            raise ValueError("SFT max_length must be a positive integer")
        if not getattr(tokenizer, "is_fast", False):
            raise ValueError("SFT requires a fast Qwen tokenizer with exact offset mappings")
        self.tokenizer: Any = tokenizer
        self.processor: Any = processor
        self.max_length: int = max_length
        self.processor_kwargs: dict[str, Any] = deepcopy(processor_kwargs or {})
        # The ``<think>`` opener rule:
        # when on, the token that opens an assistant body receives no loss, and only that token. On the lab's serving
        # path of record the server prefills NO opener: Codex ``--thinking high`` sends ``reasoning.effort`` AND a tools
        # array on every call, and with tools present vLLM 0.24 appends ``<|im_start|>assistant\n`` and nothing more
        # (the effort field adds 0 input tokens; ``/tokenize`` with tools + ``enable_thinking`` ends in ``assistant\n``;
        # the fixture ``tests/fixtures/served_prefix_with_tools.json``). The student therefore emits ``<think>`` itself
        # as the first token of every turn, then the reasoning or the closer of an empty block (``\n\n</think>\n\n``,
        # the exporter's rendering of a turn without reasoning); all of it is the student's and stays supervised under
        # the default. A ``<think>`` prefill exists only for requests without tools (a present effort ->
        # ``enable_thinking`` -> the template appends ``<think>``; 10 input tokens vs 9), not a path of record. The
        # Nemo-Chat processor rule (``cosmos3/processors/qwen3vl_nemo_chat_processor.py::add_assistant_tokens_mask``)
        # also unsupervises the empty block because its serve supplies the whole block; the path of record supplies
        # neither. Off by default because the opener's loss is the only training signal for that token there: ``True``
        # would remove the loss from the first token of every turn (what a ``True``-trained student emits there is
        # UNMEASURED with tools present; the RISK is C-04 (the lab's live probe record of the served generation prompt on
        # the A1a deployment, 2026-10-02, outside the repo; primary pointer vLLM 0.24 ``reasoning/basic_parsers.py``
        # 173-181: ``extract_reasoning`` returns all-reasoning only when ``</think>`` is absent) section 3's measured
        # trigger, an output that carries no ``</think>`` is returned as reasoning in full, tool-call turns unaffected,
        # while the closer stays supervised), and fits only a serve that prefills the opener, where that loss is one deterministic token
        # per turn (the run declares that serve through the typed field ``rl.serve_contract=prefills_opener``, the only
        # contract under which ``SFTConfig`` accepts the mask).
        self.mask_think_opener: bool = bool(mask_think_opener)
        vocab = tokenizer.get_vocab()
        self._think_id: int | None = vocab.get("<think>") if "</think>" in vocab else None
        if self.mask_think_opener and self._think_id is None:
            raise ValueError("SFT mask_think_opener requires a tokenizer whose vocabulary has <think> and </think>")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        if tokenizer.pad_token_id is None:
            raise ValueError("SFT tokenizer needs a pad or eos token")
        for group, kwargs in self.processor_kwargs.items():
            if group not in {"images_kwargs", "videos_kwargs"} or not isinstance(kwargs, dict):
                raise ValueError("SFT processor_kwargs supports HF images_kwargs and videos_kwargs mappings only")
            if {"min_pixels", "max_pixels"}.intersection(kwargs):
                raise ValueError(
                    "SFT media budgets use size.shortest_edge/longest_edge; legacy min/max_pixels may be ignored"
                )
            if processor is None:
                raise ValueError("Media processor_kwargs require a VLM processor")
            component = getattr(processor, "image_processor" if group == "images_kwargs" else "video_processor", None)
            valid = getattr(getattr(component, "valid_kwargs", None), "__annotations__", {})
            if not valid:
                raise ValueError(f"Cannot validate {group} for this processor; use a supported fast HF processor")
            unknown = set(kwargs) - set(valid)
            if unknown:
                raise ValueError(f"Unsupported {group}: {sorted(unknown)}")
            if torch.device(kwargs.get("device") or "cpu").type != "cpu":
                raise ValueError("SFT media processing must run on CPU")
            if "return_tensors" in kwargs or "return_metadata" in kwargs:
                raise ValueError("SFT owns return_tensors and video metadata output handling")

    def validation_semantics_identity(self) -> dict[str, Any]:
        """Bind the tokenization, media processing and masking contract to resume.

        ``masking`` names the rule for rows without ``SFTTarget.supervise`` (every assistant block). A ``last_assistant``
        row is identified by its own field (in the row, hence in the source fingerprint) and carries its contract string
        on ``SFTPolicyInput.masking``; this packer-level identity does not move for it.
        """
        from cosmos_framework.posttrain.roles.rollout.validation import _tokenizer_semantics_identity
        from cosmos_framework.posttrain.utils.identity import canonical_json_sha256

        media = {}
        if self.processor is not None:
            for name in ("image_processor", "video_processor"):
                component = getattr(self.processor, name, None)
                if component is not None:
                    media[name] = canonical_json_sha256(component.to_dict(), context=f"SFT {name}")
        return {
            "masking": "qwen_chatml_all_assistant_bodies_and_end_v2",
            # Present only when on, so the identity of every existing run is unchanged.
            **({"mask_think_opener": True} if self.mask_think_opener else {}),
            "transformers_version": transformers_version,
            "preserve_thinking": "preserve_thinking" in self._template_kwargs(),
            "max_length": self.max_length,
            "processor_kwargs": deepcopy(self.processor_kwargs),
            "tokenizer": _tokenizer_semantics_identity(self.tokenizer),
            "processor_template": getattr(self.processor, "chat_template", None),
            "media": media,
        }

    def _chat_template_source(self) -> str | None:
        renderer = self.processor if self.processor is not None else self.tokenizer
        template = getattr(renderer, "chat_template", None)
        if isinstance(template, dict):
            template = template.get("default")
        return template if isinstance(template, str) else None

    def _template_kwargs(self) -> dict[str, Any]:
        """``preserve_thinking=True`` only for a template that declares that variable.

        The toy test template reads it to decide whether ``reasoning_content`` is rendered. The official Qwen3-VL
        template has no such variable: it renders ``reasoning_content`` for every assistant turn and passes inline
        ``<think>`` text through as content, so the kwarg would be a no-op that Transformers 5 reclassifies as a
        processor kwarg and warns about on every render (32 lines per step in a 2 x 8-rank job).
        """
        template = self._chat_template_source()
        if template is not None and "preserve_thinking" in _template_variables(template):
            return {"preserve_thinking": True}
        return {}

    def _render(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> str:
        renderer = self.processor if self.processor is not None else self.tokenizer
        return renderer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=False,
            **self._template_kwargs(),
        )

    def _conversation(self, sample: SFTTarget) -> list[dict[str, Any]]:
        if not isinstance(sample, SFTTarget) or not isinstance(sample.messages, list) or not sample.messages:
            raise ValueError("sft_process_sample requires SFTTarget with non-empty messages")
        if sample.supervise not in SUPERVISE_CONTRACTS:
            raise ValueError(f"SFT supervise must be one of {sorted(SUPERVISE_CONTRACTS)}, got {sample.supervise!r}")
        if sample.opener not in OPENER_CONTRACT_SUFFIXES:
            raise ValueError(f"SFT opener must be one of {sorted(OPENER_CONTRACT_SUFFIXES)}, got {sample.opener!r}")
        if sample.opener == SFT_OPENER_PREFILLED and self._think_id is None:
            raise ValueError("SFT opener=prefilled requires a tokenizer whose vocabulary has <think> and </think>")
        if sample.tools is not None and (
            not isinstance(sample.tools, list) or any(not isinstance(tool, dict) for tool in sample.tools)
        ):
            raise ValueError("SFT tools must be a list of tool definitions")
        _reject_delimiters(sample.messages)
        _reject_delimiters(sample.tools)
        messages: list[dict[str, Any]] = []
        for index, message in enumerate(sample.messages):
            if not isinstance(message, dict) or message.get("role") not in {"system", "user", "assistant", "tool"}:
                raise ValueError(f"Unsupported SFT message role at index {index}")
            if message["role"] == "system" and index != 0:
                raise ValueError("SFT system message must be first")
            if message.get("reasoning_content") is not None and not isinstance(message["reasoning_content"], str):
                raise ValueError("SFT reasoning_content must be a string")
            calls = message.get("tool_calls")
            if calls is not None:
                if not isinstance(calls, list):
                    raise ValueError("SFT tool_calls must be a list")
                for call in calls:
                    function = call.get("function", call) if isinstance(call, dict) else None
                    if not isinstance(function, dict) or not isinstance(function.get("name"), str):
                        raise ValueError("SFT tool calls require a function name")
            content = message.get("content")
            if content is None:
                content = ""
            if isinstance(content, str):
                # An empty string becomes an EMPTY parts list: the Qwen3-VL processor template tests
                # ``message.content`` for truthiness before a tool call (``(loop.first and message.content)``)
                # and a one-element list would make it emit "\n", which the tokenizer merges with the
                # header's newline into one "\n\n" token straddling header and body (refused at the
                # label step). Serving renders ``content: ""`` with no newline; training must match.
                content = [{"type": "text", "text": content}] if content else []
            if not isinstance(content, list):
                raise ValueError("SFT message content must be text or a list of HF content parts")
            parts = []
            for part in content:
                if not isinstance(part, dict) or part.get("type") not in {"text", "image", "video"}:
                    raise ValueError("SFT supports text, image and video content parts only")
                kind = part["type"]
                if kind == "text":
                    if not isinstance(part.get("text"), str):
                        raise ValueError("SFT text content needs a string 'text'")
                elif self.processor is None:
                    raise ValueError("Image/video SFT requires a VLM processor")
                elif not any(key in part for key in (kind, "url", "path")):
                    raise ValueError(f"SFT {kind} content needs '{kind}', 'url' or 'path'")
                parts.append(dict(part))
            if message["role"] == "assistant" and all(
                part["type"] == "text" and not part["text"].strip() for part in parts
            ):
                # Whitespace-only text and empty text parts of an assistant turn carry no target; the
                # processor template would count them as content and separate them from a tool call with
                # a newline that the tokenizer merges into the header's ("\n\n", "\n   \n"), straddling the
                # ignored header and the supervised body. No content part renders as serving does.
                parts = []
            # Text-only Qwen tokenizers expect strings, whereas VLM templates accept content lists.
            normalized = parts if self.processor is not None else "".join(part["text"] for part in parts)
            messages.append({**message, "content": normalized})
        if not any(message["role"] == "assistant" for message in messages):
            raise ValueError("SFT conversation has no assistant target")
        if sample.supervise == SFT_SUPERVISE_LAST:
            self._refuse_history_reasoning(messages)
        return messages

    @staticmethod
    def _refuse_history_reasoning(messages: list[dict[str, Any]]) -> None:
        """``supervise="last_assistant"``: every history assistant turn (every assistant turn but the last) must carry no
        reasoning. The processor template renders a non-empty ``reasoning_content`` as a think block (250 vs 236 ids in
        a probe) while the default serve's prefix has none, and it strips an exact ``<think></think>``
        content prefix the tokenizer template keeps; a content opening with any ``<think>`` block never appears in a
        served history (the serve's reasoning parser removed the think text from what the harness replays). The row's
        prefix would otherwise not be the served prefix. Refused, not warned: a warning in a training
        log stops no launch. ``all_assistant`` rows are untouched."""
        last = max(index for index, message in enumerate(messages) if message["role"] == "assistant")
        for index, message in enumerate(messages[:last]):
            if message["role"] != "assistant":
                continue
            reasoning = message.get("reasoning_content")
            content = message["content"]
            text = content if isinstance(content, str) else "".join(p["text"] for p in content if p["type"] == "text")
            if isinstance(reasoning, str) and reasoning.strip():
                why = "reasoning_content"
            elif text.lstrip().startswith("<think>"):
                why = "content opening with <think>"
            else:
                continue
            raise ValueError(
                f"SFT supervise=last_assistant: history assistant turn {index} carries reasoning ({why}); the default "
                "serve renders no reasoning in the history, so the row's prefix would not be the served prefix (review "
                "R151-04); strip it from the history turns or supervise all_assistant"
            )

    def _validate_boundaries(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        rendered: str,
    ) -> list[str]:
        """Trace each source through the OFFICIAL template, including coalesced tool replies.

        The probe is render-only, never tokenized/processed. Markers are added to copies of text,
        reasoning and tool-call names; media are untouched. Thus a template that drops, duplicates,
        reorders or changes the role of a source message cannot silently supervise it.
        """
        probes = []
        auxiliary: list[tuple[str, int]] = []
        markers = [f"SFT_SOURCE_BOUNDARY_{i}_CHECK" for i in range(len(messages))]
        if any(marker in rendered for marker in markers):
            raise ValueError("SFT source collides with boundary validation markers")
        for index, (message, marker) in enumerate(zip(messages, markers, strict=True)):
            content = message["content"]
            if message["role"] == "assistant":
                # Probe inside inline thinking too: some Qwen templates strip earlier
                # <think> bodies even when asked to keep reasoning.
                parts = [{"type": "text", "text": content}] if isinstance(content, str) else [dict(p) for p in content]
                for part in parts:
                    if part["type"] != "text":
                        continue
                    fragments = part["text"].split("</think>")
                    for fragment_index in range(len(fragments) - 1):
                        reasoning_marker = f"SFT_INLINE_REASONING_{index}_{len(auxiliary)}_CHECK"
                        fragments[fragment_index] += reasoning_marker
                        auxiliary.append((reasoning_marker, index))
                    part["text"] = "</think>".join(fragments)
                content = parts[0]["text"] if isinstance(content, str) else parts
            content = content + marker if isinstance(content, str) else [*content, {"type": "text", "text": marker}]
            traced = {**message, "content": content}
            if message.get("reasoning_content"):
                reasoning_marker = f"SFT_REASONING_{index}_CHECK"
                traced["reasoning_content"] = message["reasoning_content"] + reasoning_marker
                auxiliary.append((reasoning_marker, index))
            if message.get("tool_calls"):
                traced["tool_calls"] = deepcopy(message["tool_calls"])
                for call_index, call in enumerate(traced["tool_calls"]):
                    function = call.get("function", call)
                    call_marker = f"SFT_TOOL_CALL_{index}_{call_index}_CHECK"
                    function["name"] += call_marker
                    auxiliary.append((call_marker, index))
            probes.append(traced)
        probe = self._render(probes, tools)
        probe_blocks = _blocks(probe)
        roles = [block[1] for block in _blocks(rendered)]
        if roles != [block[1] for block in probe_blocks]:
            raise ValueError("Unsupported SFT template: source and rendered message boundaries differ")
        seen = []
        for block_index, block in enumerate(probe_blocks):
            indices = sorted(
                (i for i, marker in enumerate(markers) if marker in block[2]),
                key=lambda i: block[2].index(markers[i]),
            )
            if not indices:
                if block_index != 0 or block[1] != "system":
                    raise ValueError("Unsupported SFT template: generated non-system message")
                continue
            source_roles = [messages[i]["role"] for i in indices]
            expected = "user" if source_roles[0] == "tool" else source_roles[0]
            if block[1] != expected or (len(indices) > 1 and set(source_roles) != {"tool"}):
                raise ValueError("Unsupported SFT template: source role/boundary mismatch")
            seen.extend(indices)
        if seen != list(range(len(messages))) or any(probe.count(marker) != 1 for marker in markers):
            raise ValueError("Unsupported SFT template: source messages were dropped, duplicated or reordered")
        for marker, index in auxiliary:
            if (
                marker in rendered
                or probe.count(marker) != 1
                or not any(
                    marker in block[2] and markers[index] in block[2] and block[1] == "assistant"
                    for block in probe_blocks
                )
            ):
                raise ValueError("Unsupported SFT template: supplied reasoning/tool calls were dropped or moved")
        return roles

    def render_conversation(self, sample: SFTTarget) -> str:
        """The template's text for a validated conversation, before media expansion and tokenization."""
        return self._render(self._conversation(sample), sample.tools)

    def sft_process_sample(self, sample: SFTTarget) -> SFTPolicyInput:
        messages = self._conversation(sample)
        rendered = self._render(messages, sample.tools)
        roles = self._validate_boundaries(messages, sample.tools, rendered)
        if self.processor is None:
            processed = dict(self.tokenizer(rendered, add_special_tokens=False, return_tensors="pt"))
        else:
            # Render separately so media kwargs cannot be swallowed as template variables. HF owns
            # media loading and expansion; never use RL's image-only normalization/removal helpers.
            media: dict[str, list[Any]] = {"images": [], "videos": []}
            for message in messages:
                for part in message["content"]:
                    kind = part["type"]
                    if kind in {"image", "video"}:
                        key = next(key for key in (kind, "url", "path") if key in part)
                        media["images" if kind == "image" else "videos"].append(part[key])
            for kind, key in (("image", "images"), ("video", "videos")):
                # The template owns which parts become tokens (the Qwen3-VL template joins only the text parts
                # of an assistant turn); the processor expands exactly the placeholders it is given, so a media
                # part without a placeholder would be encoded into pixel_values with no tokens to carry it.
                placeholders = rendered.count(f"<|{kind}_pad|>")
                if placeholders != len(media[key]):
                    raise ValueError(
                        f"SFT template rendered {placeholders} {kind} placeholder(s) for {len(media[key])} {kind} "
                        "part(s); media the template does not render (for example inside an assistant turn) cannot be "
                        "aligned with the processor's outputs"
                    )
            processed = dict(
                self.processor(
                    text=[rendered],
                    **{key: value for key, value in media.items() if value},
                    **deepcopy(self.processor_kwargs),
                    **(
                        {"return_mm_token_type_ids": True}
                        if "mm_token_type_ids" in self.processor.model_input_names
                        else {}
                    ),
                    add_special_tokens=False,
                    padding=False,
                    truncation=False,
                    return_tensors="pt",
                )
            )
        ids_tensor = processed.pop("input_ids")  # [1,L]
        if ids_tensor.ndim != 2 or ids_tensor.shape[0] != 1:
            raise ValueError("SFT processor must produce exactly one sequence")
        ids = ids_tensor[0].tolist()
        if len(ids) > self.max_length:
            raise ValueError(
                f"SFT sequence length {len(ids)} exceeds max_length={self.max_length}; truncation is forbidden"
            )
        processed.pop("attention_mask", None)
        # The token offsets are measured AFTER expansion, and accepted only when every token id
        # round-trips exactly. This is not HF's pre-expansion assistant-mask character mapping.
        expanded = self.tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        encoded = self.tokenizer(expanded, add_special_tokens=False, return_offsets_mapping=True)
        if encoded["input_ids"] != ids:
            raise ValueError("SFT tokenizer cannot exactly round-trip the expanded token stream")
        blocks = _blocks(expanded)
        if [block[1] for block in blocks] != roles:
            raise ValueError("SFT processor changed source message boundaries")
        # The row's rule: every assistant block, or under ``supervise="last_assistant"`` the last one alone, every earlier
        # assistant block left at -100 like the observations around it (the history of a prefix row).
        assistant_blocks = [block for block in blocks if block[1] == "assistant"]
        supervised = supervised_assistant_blocks(len(assistant_blocks), sample.supervise)
        train = bytearray(len(expanded))
        for index in supervised:
            block = assistant_blocks[index]
            train[block.start(2) : block.end()] = b"\x01" * (block.end() - block.start(2))
        for media_span in _MEDIA.finditer(expanded):
            train[media_span.start() : media_span.end()] = b"\x00" * (media_span.end() - media_span.start())
        stripped = _MEDIA.sub("", expanded)
        if any(token in stripped for token in _RESERVED[2:6]):
            raise ValueError("Unsupported SFT media expansion: unbalanced vision markers")
        labels = []
        for token, (start, end) in zip(ids, encoded["offset_mapping"], strict=True):
            span = train[start:end]
            if any(span) and not all(span):
                raise ValueError("Ambiguous SFT token overlaps assistant body and ignored header/media")
            labels.append(token if end > start and all(span) else _IGNORE)
        # One rule for the run-level knob and the row's own ``opener`` field (row rule R4: the prefilled slice shares the row
        # files with the self rows): the first <think> token of every supervised assistant block carries no loss; applied
        # once, so a row that sets both is not masked twice. A prefilled ROW must have had exactly one opener removed in
        # every supervised block, or it is refused: its contract string is recorded only when the masking
        # happened. The run-level knob on a bare block masks nothing and refuses nothing (the knob is the run's variant,
        # recorded in the packer identity, not the row's contract).
        if self.mask_think_opener or sample.opener == SFT_OPENER_PREFILLED:
            unmasked = labels
            labels = self._mask_think_openers(ids, labels)
            if sample.opener == SFT_OPENER_PREFILLED:
                self._require_one_opener_per_supervised_block(
                    unmasked, labels, encoded["offset_mapping"], assistant_blocks, supervised
                )
        if not any(label != _IGNORE for label in labels):
            raise ValueError("SFT conversation contains no supervised assistant tokens")
        for key, value in processed.items():
            if key not in _MEDIA_KEYS | {"mm_token_type_ids"}:
                raise ValueError(f"Unsupported SFT processor side input {key!r}")
            if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
                raise ValueError(f"SFT processor side input {key!r} must be a CPU tensor")
        return SFTPolicyInput(ids, labels, processed or None, masking=masking_contract(sample.supervise, sample.opener))

    @staticmethod
    def _require_one_opener_per_supervised_block(
        before: list[int],
        after: list[int],
        offsets: Any,
        assistant_blocks: list[re.Match[str]],
        supervised: list[int],
    ) -> None:
        """``opener="prefilled"``: the opener mask must have removed exactly one id in EVERY
        supervised assistant block of the row. A supervised block without a ``<think>`` opener (a bare turn) would otherwise
        be packed as a self row under a ``_opener_prefilled_v1`` contract string. The removed positions are assigned to
        blocks by their character offsets in the expanded stream; the refusal names the bare blocks (indices among the
        row's assistant turns) and the per-block counts."""
        removed = [pos for pos, (a, b) in enumerate(zip(before, after, strict=True)) if a != _IGNORE and b == _IGNORE]
        counts = {index: 0 for index in supervised}
        for pos in removed:
            start = offsets[pos][0]
            for index in supervised:
                block = assistant_blocks[index]
                if block.start(2) <= start < block.end():
                    counts[index] += 1
                    break
        bare = [index for index in supervised if counts[index] != 1]
        if bare:
            raise ValueError(
                f"SFT opener=prefilled: supervised assistant block(s) {bare} (indices among the row's assistant turns) do "
                f"not carry exactly one <think> opener to mask (masked per supervised block {supervised}: "
                f"{[counts[i] for i in supervised]}); a prefilled row needs a reasoning block in every supervised assistant "
                "turn (review R149-01)"
            )

    def _mask_think_openers(self, ids: list[int], labels: list[int]) -> list[int]:
        """Unsupervise the ``<think>`` token that opens an assistant body (the first supervised token after an ignored
        one) and nothing else (applied for the run-level ``mask_think_opener`` or for a row whose ``opener`` is
        ``prefilled``, row rule R4; one rule, applied once): the tokens after it -- reasoning text, a second ``<think>``, or
        the closer of an empty block -- stay supervised. This fits only a serve that prefills the opener (a request without tools): on the
        serving path of record (tools present) vLLM 0.24 prefills nothing after ``<|im_start|>assistant\n``, the
        student must emit the opener itself, and the default keeps it supervised. (the earlier rule
        also masked an empty ``<think>`` + newline + ``</think>`` block token for token as Nemo-Chat does; its newline
        was ``encode("\n")``, so under the exporter's ``<think>\n\n</think>\n\n`` rendering, where ``\n\n`` is ONE
        token, it masked only the opener, while under a single-newline rendering it masked the closer the student has
        to learn.) A ``<think>`` inside a body (inline reasoning text) stays supervised."""
        assert self._think_id is not None
        # The boundary is read from the ORIGINAL labels, never from ``out``: a body that begins with two consecutive
        # openers (``<think><think>``) masks its first token only. Reading ``out`` made the first masked opener an
        # ignored neighbour of the second, which then satisfied the boundary test and was masked too (review thread
        # 19bf3c650f; probe: labels [-100, T, T, 7] -> [-100, -100, -100, 7] before, [-100, -100, T, 7] now).
        out = list(labels)
        for pos, token in enumerate(ids):
            if token == self._think_id and labels[pos] != _IGNORE and (pos == 0 or labels[pos - 1] == _IGNORE):
                out[pos] = _IGNORE
        return out

    def sft_compute_max_len(self, processed_samples: list[SFTPolicyInput]) -> int:
        if not processed_samples:
            raise ValueError("SFT batch must not be empty")
        length = max(len(sample.input_ids) for sample in processed_samples)
        if length > self.max_length:
            raise ValueError("SFT batch exceeds max_length; truncation is forbidden")
        return length

    def sft_collate_fn(
        self,
        sub_batch: list[SFTPolicyInput],
        computed_max_len: int,
        ignore_label_id: int = _IGNORE,
    ) -> dict[str, Any]:
        length = self.sft_compute_max_len(sub_batch)
        if not length or not length <= computed_max_len <= self.max_length:
            raise ValueError("SFT computed_max_len must cover the batch without exceeding max_length")
        rows, labels, masks = [], [], []
        for sample in sub_batch:
            if len(sample.labels) != len(sample.input_ids):
                raise ValueError("SFT labels and input_ids must have identical lengths")
            pad = computed_max_len - len(sample.input_ids)
            rows.append(sample.input_ids + [self.pad_token_id] * pad)
            labels.append([ignore_label_id if x == _IGNORE else x for x in sample.labels] + [ignore_label_id] * pad)
            masks.append([1] * len(sample.input_ids) + [0] * pad)
        batch: dict[str, Any] = {
            "input_ids": torch.tensor(rows, dtype=torch.long),  # [B,L]
            "labels": torch.tensor(labels, dtype=torch.long),  # [B,L], unshifted
            # int64 0/1, main's collator contract (tests/test_model_qwen3_vl.py::test_native_causal_mask_accepts_sft_padding_masks
            # pins it): the native eager mask path (model/qwen3_vl.py::_qwen3_vl_causal_mask) casts a 2-D mask to bool
            # before eager_mask / torch.where, HF models and the SFT objective's 0/1 check accept it. (This branch emitted
            # torch.bool for that path before main grew the cast; the two met at the rebase.)
            "attention_mask": torch.tensor(masks, dtype=torch.long),  # [B,L]
        }
        side_inputs = [sample.mm_kwargs or {} for sample in sub_batch]
        for key in sorted(set().union(*(values.keys() for values in side_inputs))):
            if key not in _MEDIA_KEYS | {"mm_token_type_ids"}:
                raise ValueError(f"Unsupported SFT processor side input {key!r}")
            values = [sample[key] for sample in side_inputs if key in sample]
            if any(not isinstance(value, torch.Tensor) or value.device.type != "cpu" for value in values):
                raise ValueError(f"SFT side input {key!r} must contain CPU tensors")
            if key == "mm_token_type_ids":
                type_rows = []
                for sample, side in zip(sub_batch, side_inputs, strict=True):
                    value = side.get(key, torch.zeros((1, len(sample.input_ids)), dtype=torch.long))  # [1,L]
                    if value.shape != (1, len(sample.input_ids)):
                        raise ValueError("SFT mm_token_type_ids must have shape [1, sequence_length]")
                    type_rows.append(value[0].tolist() + [0] * (computed_max_len - len(sample.input_ids)))
                batch[key] = torch.tensor(type_rows, dtype=torch.long)  # [B,L]
            else:
                batch[key] = torch.cat(values, dim=0)  # [sum(patches or media), ...], sample/media order
        return batch

    def get_rollout_input(self, item: ObsBundle) -> Any:
        raise NotImplementedError("QwenSFTDataPacker is SFT-only; use the RL packer for rollout")

    def get_policy_input(
        self,
        sample: ObsBundle,
        rollout_output: Any,
        n_ignore_prefix_tokens: int = 0,
        *,
        prompt_ids: list[int] | None = None,
    ) -> Any:
        raise NotImplementedError("QwenSFTDataPacker is SFT-only; use the RL packer for policy scoring")

    def policy_compute_max_len(self, processed_samples: list[Any]) -> int:
        raise NotImplementedError("QwenSFTDataPacker is SFT-only")

    def policy_collate_fn(self, processed_samples: list[Any], computed_max_len: int) -> dict[str, Any]:
        raise NotImplementedError("QwenSFTDataPacker is SFT-only")


def build_sft_packer(
    name_or_path: str,
    *,
    max_length: int,
    processor_kwargs: dict[str, Any] | None = None,
    trust_remote_code: bool = False,
    mask_think_opener: bool = False,
) -> DataPacker:
    """Load public/local metadata only; supported Qwen families are explicit, not generic HF.

    Read the raw config so loading a Qwen3.5 processor does not require importing its model class
    (the model runtime separately requires a Transformers version supporting that architecture).
    """
    if Version(transformers_version) < Version("5.8.1"):
        raise RuntimeError(
            "SFT requires Transformers >=5.8.1; sync posttrain with --no-group transformers4 --group sft"
        )
    if not name_or_path:
        raise ValueError("build_sft_packer requires name_or_path")
    config, _ = PretrainedConfig.get_config_dict(name_or_path)
    model_type = config.get("model_type")
    if model_type in _VLM_TYPES:
        processor = AutoProcessor.from_pretrained(name_or_path, trust_remote_code=trust_remote_code)
        if type(processor).__name__ != "Qwen3VLProcessor":
            raise ValueError("SFT VLM support requires Qwen3VLProcessor")
        tokenizer = processor.tokenizer
    elif model_type in _TEXT_TYPES:
        processor = None
        tokenizer = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=trust_remote_code, use_fast=True)
    else:
        raise ValueError(
            f"Unsupported SFT model_type {model_type!r}; only Qwen text models, dense Qwen3.5/3.8 and "
            "Qwen3-VL / Qwen3-VL-MoE are supported"
        )
    return QwenSFTDataPacker(
        tokenizer,
        processor=processor,
        max_length=max_length,
        processor_kwargs=processor_kwargs,
        mask_think_opener=mask_think_opener,
    )
