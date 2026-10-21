# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Pre-launch SFT row validator: every source row through the real packer on CPU, before any trainer exists.

``--validate-manifest`` checks wiring and file availability only. A row the packer refuses (over-long, reserved
delimiters, a template that drops reasoning), a row whose supervised tokens disagree with its assistant turns, or
a conversation whose tool results carry the wrong role surfaces today at step 0 inside ``algorithm/sft.py``
(``packer.sft_process_sample`` in the objective), after the model was built on GPUs. This script runs the same
``build_sft_dataset`` -> ``SFTTarget`` -> ``QwenSFTDataPacker.sft_process_sample`` path over EVERY row and reports,
per row: total tokens; loss tokens (``labels != -100``); an INDEPENDENT assistant-token count (every rendered
assistant block the row's ``supervise`` field supervises: its text fragments and its ``<|im_end|>`` tokenized on
their own, never through offsets or labels, one token fewer per block when the row's ``opener`` is ``prefilled``),
compared PER ASSISTANT BLOCK with the loss tokens the labels carry between supervised ``<|im_end|>`` tokens; image
and video counts with their visual tokens (``*_grid_thw`` / merge^2) against the ``<|image_pad|>`` and
``<|video_pad|>`` tokens in the stream, EACH MODALITY ON ITS OWN (the model scatters image and video features
separately, so an image surplus must never cancel a video deficit in one combined sum); the row's length against
``rl.max_length``; and the source role grammar (tool results follow tool calls one-to-one, by id only when every
call AND every result of the turn carry string ids; the conversation ends with an assistant turn; consecutive
assistant turns are accepted because the normalizer and the packer accept them and supervise each as its own
block). It exits non-zero on any failed row, naming the row and every reason, and writes one JSONL line per row
plus one summary line per split.

Source availability is NOT a configuration error here: a selected split whose source cannot be opened or imported
is reported as that split's load refusal (FAIL line, gate line, ``load_error``), and an unavailable split never
prevents the other split from being inspected. The trainer's ``--validate-manifest`` preflight keeps refusing the
whole configuration in that case.

Terms used by this script, its output and its README:

- **row number**: the 1-based position of a conversation in its source (``row 2`` is the second conversation);
  ``index`` is the 0-based dataset index the training cursor uses.
- **loss tokens**: the count of ``labels != -100`` in the packer's output for the row; **loss tokens per block**:
  that count split at every supervised ``<|im_end|>`` token, one entry per assistant turn in stream order.
- **independent assistant-token count**: per ``<|im_start|>assistant\\n...<|im_end|>`` block of the template's
  rendered text, the body's text fragments (media placeholders removed) tokenized on their own plus the block's
  ``<|im_end|>``; computed without reading offsets, labels or the expanded stream; summed over the blocks the row's
  ``supervise`` field supervises, one token fewer per block under ``opener: prefilled`` (next term).
- **supervise / opener**: the row's own fields, read exactly as the packer reads them (``SFTTarget``): ``supervise``
  is ``all_assistant`` (the default: every assistant block carries loss) or ``last_assistant`` (the last assistant
  block alone; every earlier one stays at -100 like the observations around it, so **loss tokens per block** has one
  entry and the independent count covers that block only, while ``assistant_blocks`` and ``body_tokens`` still
  describe every assistant block of the row); ``opener`` is ``self`` (the default: the ``<think>`` that opens a
  supervised body is the student's token) or ``prefilled`` (the serve prefills that token, the packer leaves it at
  -100 and the independent count subtracts one per supervised block). A ``prefilled`` row with a supervised turn
  that renders no ``<think>`` opener is refused by the packer (``_require_one_opener_per_supervised_block``) and the
  refusal is the row's reason; a row whose ``meta.supervise`` or ``meta.opener`` disagrees with its field is refused
  by the source loader (``normalize_row``) and surfaces as the row's refusal for a factory source or as the split's
  load refusal for a JSONL source; an unknown value is the row's reason, not a crash. The report carries
  ``supervise``, ``opener`` and ``supervised_blocks`` per row. The run-level knob ``rl.mask_think_opener`` (the
  recipe's acknowledged variant) applies the same rule to every row of the run: the validator builds its packer with
  the run's value, and the independent count takes one token off every supervised block that OPENS with ``<think>``
  when the knob or the row's ``prefilled`` asks for it, once, never twice; a block that opens with anything else loses
  nothing under the knob (the packer masks nothing there and refuses nothing), and under ``prefilled`` the packer has
  refused such a block already.
- **straddle**: one token whose character span covers both a supervised assistant-body character and an ignored
  header or media character (for example the ``\\n\\n`` the tokenizer forms from the header's newline and a body
  that starts with a newline); the packer refuses such a row, and the validator reports that refusal as the row's
  reason.
- **FAIL line**: ``FAIL <split> row <n>: <reason>; <reason>`` on stdout for every refused row, or
  ``FAIL <split>: <reason>`` for a split whose source is refused as a whole.
- **gate line**: the one-line verdict per split on stdout, exactly
  ``<split>: rows <N> | loss_tokens == assistant_tokens on <K>/<N> | longest <L> <= <max_length> | failed <F>``
  (``>`` instead of ``<=`` when the longest row is over-long; ``| rows % dp_world(<W>) = <R>`` appended when
  ``--dp-world`` is given); for a split whose source was refused at load the line is
  ``<split>: rows 0 | load refused: <reason> | failed 1`` and never reads as a pass.
- **summary line**: the last JSONL line of a split, ``{"summary": {...}}`` with ``split``, ``rows``, ``failed``,
  ``failed_rows``, ``unmeasured_rows``, ``matched_rows``, ``longest_tokens``, ``max_length``, ``over_long_rows``,
  ``total_tokens``, ``loss_tokens``, ``loss_share``, ``images``, ``videos``, ``visual_tokens``, ``held_out_census``
  (the validation split's census against the training split, ``null`` elsewhere), ``census_error`` (the setup's
  refusal of the pair, or the read-failure sentence whose shape is ``CENSUS_READ_ERROR`` -- typed once, at that
  constant, and quoted by the README's ``census_error`` term -- when the census failed while it asked a source for a
  row, for that row's media, for its length or for its row list, whatever ``Exception`` the call raised, named by the
  side and the call the census asked for last and by the row and its item while a row was being read; both readers read
  a row through ``_read_row``, so one defect carries one class and one message in both; the split's rows are still
  inspected and its summary written, see ``held_out_census_entry``), ``dp_world``,
  ``dp_remainder``, ``load_error`` (the refusal text when the source could not be loaded, else null) and ``ok``.
  A split whose source is refused at load still gets its summary line (``rows 0``, ``load_error`` set,
  ``failed 1``, ``ok false``).
- **redaction**: every reason that quotes an exception keeps only the last two components of each absolute path
  (``/data/items/item_7/image.png`` -> ``item_7/image.png``, so the item directory still discriminates), is
  bounded in length as a whole sentence before it is printed or written (``_REASON_LIMIT`` characters, prefix
  included; an item id quoted inside a census reason is cut at ``_ITEM_LIMIT``, a quarter of that, so the class and
  the message after it survive the bound), and names the exception class unless it is a plain ``ValueError`` or
  ``TypeError`` (the loader's and the packer's own sentences; a corrupt record's ``JSONDecodeError`` is named); the
  JSONL report carries counts and reasons only, never row content and never a path with its tree.

The packer is built exactly as the trainer builds it (``build_sft_packer`` on the same artifact with the run's
``processor_kwargs``, ``trust_remote_code`` and ``mask_think_opener``, as ``entrypoints.sft.build_trainer`` passes them)
except for the length cap: the validator measures every row and applies ``rl.max_length`` itself, so the longest row
is reported even when it is the row that would have killed the run.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, TextIO

_IGNORE = -100
_ROLES = frozenset({"system", "user", "assistant", "tool"})
_MEDIA_KINDS = ("image", "video")
_PATH = re.compile(r"(?<![\w.~])/(?:[^\s/'\"`,;:()\[\]{}<>]+/)*([^\s/'\"`,;:()\[\]{}<>]+/[^\s/'\"`,;:()\[\]{}<>]+)")
_REASON_LIMIT = 400
_ITEM_LIMIT = _REASON_LIMIT // 4  # an item id quoted in a census reason; the class and the message must fit after it
MEASURE_MAX_LENGTH = 2**31 - 1
"""The packer cap used for measuring: rows are compared against ``rl.max_length`` by this script, not refused."""

DEFAULT_CONFIG = "cosmos_framework/posttrain/configs/sft_qwen3p8_27b_multimodal.py"
REPORT_NAME = "validate_sft_rows"
REPORT_SUFFIX = ".jsonl"
REPORT_GLOB = f"{REPORT_NAME}*{REPORT_SUFFIX}"


@dataclass
class RowReport:
    """One source row's measurements; ``reasons`` is empty when the row may train."""

    split: str
    row: int
    index: int
    row_id: str | None = None
    supervise: str | None = None
    opener: str | None = None
    total_tokens: int | None = None
    loss_tokens: int | None = None
    loss_tokens_per_block: list[int] | None = None
    mask_matches: bool | None = None
    assistant_tokens: int | None = None
    assistant_blocks: int | None = None
    supervised_blocks: int | None = None
    body_tokens: list[int] | None = None
    images: int = 0
    videos: int = 0
    image_grids: int | None = None
    video_grids: int | None = None
    visual_tokens: int | None = None
    image_pad_tokens: int | None = None
    video_pad_tokens: int | None = None
    reasons: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.reasons


def redact(text: Any, limit: int = _REASON_LIMIT) -> str:
    """Keep the last two components of every absolute path (``/a/b/item_7/x.png`` -> ``item_7/x.png``) and bound the length
    at ``limit`` characters (``...`` closes a cut text)."""
    reduced = _PATH.sub(r"\1", str(text))
    if len(reduced) > limit:
        reduced = reduced[: limit - 3] + "..."
    return reduced


class SourceReadError(Exception):
    """A ``StopIteration``, or a subclass of it, that a source raised while a reader asked it for a row, re-raised by
    ``_read_row`` with its detail already formed: ``<class>: <message>`` with the raised class's own name (a subclass by
    its name, as ``_refusal`` names every other class), which ``_refusal`` prints as it is. A ``StopIteration`` that
    escapes a generator frame becomes ``RuntimeError: generator raised StopIteration`` (PEP 479), and the census walks a
    factory source through a generator expression inside the setup's ``source_overlap`` while the per-row reader reads
    the same row in a plain loop, so the two readers filed one defect under two classes (and a subclass under the base
    name, until ``_read_row`` named the raised class); every other class crosses the frame unchanged and is named by
    ``_refusal`` itself."""


def _read_row(dataset: Any, index: int) -> Any:
    """``dataset[index]`` for BOTH readers, ``validate_dataset`` and the census's ``_CensusReads``: the one place a source row
    is read, so one defect is filed under one class and one message by both."""
    try:
        return dataset[index]
    except StopIteration as error:
        raise SourceReadError(f"{type(error).__name__}: {error}") from error


def _refusal(prefix: str, error: BaseException) -> str:
    """``<prefix>: <detail>``, the whole sentence redacted and bounded (``redact``: the prefix may quote an item id, the
    detail a path): a plain ``ValueError`` or ``TypeError`` is the loader's or the packer's own sentence and stands alone, and
    so does a ``SourceReadError``, whose detail names the source's class already; every other class is named, a subclass
    included, so a corrupt record's ``JSONDecodeError`` or ``UnicodeDecodeError`` is told apart from the loader's refusal of
    a row."""
    stands_alone = type(error) in (ValueError, TypeError, SourceReadError)
    detail = str(error) if stands_alone else f"{type(error).__name__}: {error}"
    return redact(f"{prefix}: {detail}")


def _call_ids(calls: Any) -> list[str] | None:
    """The call ids of one assistant turn when every call carries a string id, else None (count-only rule)."""
    if not isinstance(calls, list) or not calls:
        return None
    ids = [call.get("id") if isinstance(call, dict) else None for call in calls]
    return ids if all(isinstance(identifier, str) and identifier for identifier in ids) else None


def role_grammar_failures(messages: Any, supervise: str | None = None) -> list[str]:
    """Structural defects the template cannot see because it renders whatever roles it is given.

    V7 (2026-10-04, the P layout's contract, approved with both arms on 2026-10-03): under ``supervise == "last_assistant"`` a
    conversation whose LAST message is the supervised assistant turn carrying ``tool_calls`` is a complete row: the dangling
    call IS the completion the student must emit, and the serving stack stops there (dsh sends the call to its tool and the
    next row carries the result). Every other dangling-call shape still fails: a call left unanswered in an EARLIER turn (a
    later user or assistant turn arrives while results are owed), a whole-episode row (``all_assistant``) that ends in a
    call, a row that ends on a tool result or a user turn. ``supervise`` is the row's own field as the packer reads it
    (``SFTTarget``); callers that validate messages alone pass nothing and keep the whole-episode rule. The exemption TRUSTS the
    exporter's ``supervise`` field: a row labelled ``last_assistant`` whose final call is not the supervised completion is caught by
    the exporter's episode gate and by the pool census (LAST-TURN-NOT-ASSISTANT), not here.

    A tool result carrying role ``assistant`` becomes a supervised block (the model would learn to emit the
    observation); a tool call without its result teaches an action the episode never observed; a conversation
    that ends on a user or tool turn supervises nothing after its last observation. Results are matched to calls
    by COUNT; the id rule (unknown, duplicate and unanswered ids) applies to a turn only when every call of that
    turn carries a string ``id`` AND every result of that turn carries a string ``tool_call_id``; the Qwen
    template renders no ids, so a count-correct turn without ids on either side is a correct row.
    Consecutive user turns are allowed (harness exports carry an environment turn before the task turn), and so
    are consecutive assistant turns: ``normalize_row`` and ``QwenSFTDataPacker`` accept them and supervise each
    as its own block (``[2, 2]`` loss tokens for two one-token answers), so a rule against them would refuse rows
    the training path trains. A tool turn mislabelled as assistant is still caught by the pending-result rule.
    """
    if not isinstance(messages, list) or not messages:
        return ["messages must be a non-empty list"]
    failures: list[str] = []
    pending = 0  # tool results still owed by the previous assistant turn
    call_ids: list[str] | None = None  # the turn's call ids when every call carries one
    results: list[tuple[int, Any]] = []  # (message index, tool_call_id) of the turn's results
    previous: str | None = None

    def close_turn(index: int) -> None:
        nonlocal pending, call_ids, results
        ids_on_results = results and all(isinstance(identifier, str) and identifier for _, identifier in results)
        if pending == 0 and call_ids is not None and ids_on_results:
            seen: list[str] = []
            for message_index, identifier in results:
                if identifier not in call_ids:
                    failures.append(f"message {message_index}: tool result for unknown call id {identifier!r}")
                elif identifier in seen:
                    failures.append(f"message {message_index}: duplicate tool result for call id {identifier!r}")
                else:
                    seen.append(identifier)
            for identifier in call_ids:
                if identifier not in seen:
                    failures.append(f"message {index}: tool call id {identifier!r} has no result")
        pending, call_ids, results = 0, None, []

    for index, message in enumerate(messages):
        role = message.get("role") if isinstance(message, dict) else None
        if role not in _ROLES:
            failures.append(f"message {index}: unsupported role {role!r}")
            previous = None
            continue
        if role == "system" and index != 0:
            failures.append(f"message {index}: system message is not first")
        if role == "tool":
            if pending > 0:
                pending -= 1
            else:
                failures.append(f"message {index}: tool result without a preceding tool call")
            results.append((index, message.get("tool_call_id")))
        else:
            if pending > 0:
                hint = " (a tool turn mislabelled as assistant?)" if role == "assistant" else ""
                failures.append(
                    f"message {index}: {role} turn where {pending} tool result(s) were still expected{hint}"
                )
            close_turn(index)
            if role == "assistant":
                calls = message.get("tool_calls")
                if calls is not None and not isinstance(calls, list):
                    failures.append(f"message {index}: tool_calls is not a list ({type(calls).__name__})")
                    calls = None
                pending = len(calls) if isinstance(calls, list) else 0
                call_ids = _call_ids(calls)
        previous = role
    last = len(messages) - 1
    if pending > 0:
        if supervise == "last_assistant" and previous == "assistant":
            # V7: the supervised final turn's own calls are the completion (the P layout); nothing is owed inside the row
            pass
        else:
            failures.append(f"message {last}: conversation ends with {pending} tool call(s) and no result")
    else:
        close_turn(last)
        if previous != "assistant":
            failures.append(f"message {last}: conversation does not end with an assistant turn")
    if not any(isinstance(message, dict) and message.get("role") == "assistant" for message in messages):
        failures.append("no assistant turn to supervise")
    return failures


def media_counts(messages: Any) -> tuple[int, int]:
    """Image and video content parts across every message, in source order."""
    counts = {kind: 0 for kind in _MEDIA_KINDS}
    for message in messages if isinstance(messages, list) else []:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for part in content:
            kind = part.get("type") if isinstance(part, dict) else None
            if kind in counts:
                counts[kind] += 1
    return counts["image"], counts["video"]


@dataclass(frozen=True)
class AssistantBlockCount:
    body_tokens: int
    end_tokens: int
    opens_with_think: bool = False
    """The body begins with ``<think>``: the token the packer leaves at -100 under the run's ``mask_think_opener`` or
    the row's ``opener: prefilled`` (its first supervised token; a ``<think>`` later in the body stays supervised)."""


def assistant_block_counts(encode: Callable[[str], Sequence[int]], rendered: str) -> list[AssistantBlockCount]:
    """Count supervised tokens the way the mask is DEFINED, not the way the packer computes them.

    The packer paints character spans of the final expanded stream and reads labels through token offsets.
    This count never looks at offsets, labels or the expanded stream: it takes the template's rendered text,
    finds every ``<|im_start|>assistant\\n...<|im_end|>`` block, removes the media placeholders inside the body
    (the packer ignores media wherever they sit) and tokenizes each remaining text fragment on its own, then adds
    the block's ``<|im_end|>``. Fragment boundaries fall on special tokens (the header, vision markers,
    ``<|im_end|>``), which are pre-tokenization boundaries, so a standalone fragment tokenizes exactly as it does
    in the stream; a body whose first token would straddle the header is a row the packer refuses anyway. Each
    count records whether the body opens with ``<think>`` (``opens_with_think``), the one token the opener rules
    take off the block.
    """
    from cosmos_framework.posttrain.data.packer.sft import _CHAT, _MEDIA

    end_tokens = len(encode("<|im_end|>"))
    counts = []
    for block in _CHAT.finditer(rendered):
        if block[1] != "assistant":
            continue
        body_tokens = sum(len(encode(fragment)) for fragment in _MEDIA.split(block[2]) if fragment)
        counts.append(
            AssistantBlockCount(
                body_tokens=body_tokens, end_tokens=end_tokens, opens_with_think=block[2].startswith("<think>")
            )
        )
    return counts


def loss_tokens_per_block(labels: Sequence[int], end_id: int | None) -> list[int]:
    """Supervised label counts split at every supervised ``<|im_end|>``; a trailing open run is its own entry."""
    blocks: list[int] = []
    current = 0
    for label in labels:
        if label == _IGNORE:
            continue
        current += 1
        if end_id is not None and label == end_id:
            blocks.append(current)
            current = 0
    if current:
        blocks.append(current)
    return blocks


def _grid_tokens(grid: Any, component: Any, name: str, reasons: list[str]) -> tuple[int, int | None]:
    """Media count and visual tokens from a ``[N, 3]`` ``grid_thw`` tensor (``(0, 0)`` when the packer emitted none).

    ``merge_size`` comes from the processor; when it cannot be read the count is returned with ``None`` tokens
    and a reason, and the caller skips the pad comparison for that modality.
    """
    if grid is None:
        return 0, 0
    merge = getattr(component, "merge_size", None)
    if not isinstance(merge, int) or merge <= 0:
        reasons.append(f"cannot read merge_size of the {name} processor; visual tokens not verified")
        return int(grid.shape[0]), None
    tokens = 0
    for thw in grid.tolist():
        patches = 1
        for value in thw:
            patches *= int(value)
        if patches % (merge * merge):
            reasons.append(f"{name}_grid_thw {thw} is not divisible by merge_size^2={merge * merge}")
        tokens += patches // (merge * merge)
    return int(grid.shape[0]), tokens


def _token_id(tokenizer: Any, token: str) -> int | None:
    identifier = tokenizer.convert_tokens_to_ids(token)
    if not isinstance(identifier, int) or identifier < 0 or identifier == getattr(tokenizer, "unk_token_id", None):
        return None
    return identifier


def inspect_row(
    packer: Any,
    row: Any,
    *,
    split: str,
    index: int,
    max_length: int,
    allow_empty_assistant: bool = False,
) -> RowReport:
    """Measure one normalized source row; every defect becomes a reason, none raises."""
    from cosmos_framework.posttrain.data.packer.sft import supervised_assistant_blocks
    from cosmos_framework.posttrain.schema import SFTTarget

    report = RowReport(split=split, row=index + 1, index=index)
    if not isinstance(row, dict):
        report.reasons.append("row is not a mapping")
        return report
    row_id = row.get("id")
    if isinstance(row_id, (str, int)):
        report.row_id = str(row_id)
    messages = row.get("messages")
    # V7: the grammar reads the row's supervise mode (a last_assistant row may end on its supervised turn's own tool calls)
    report.reasons.extend(role_grammar_failures(messages, supervise=row.get("supervise", "all_assistant")))
    report.images, report.videos = media_counts(messages)
    # The row's own rules, read exactly as the packer reads them; ``SFTTarget`` refuses an unknown value, which is then
    # this row's reason (a JSONL or factory source has refused it already in ``normalize_row``).
    supervise, opener = row.get("supervise", "all_assistant"), row.get("opener", "self")
    try:
        target = SFTTarget(messages=messages, tools=row.get("tools"), supervise=supervise, opener=opener)
        report.supervise, report.opener = target.supervise, target.opener
        rendered = packer.render_conversation(target)
        sample = packer.sft_process_sample(target)
    except Exception as error:  # noqa: BLE001 -- a template or processor crash is this row's defect, not the validator's
        report.reasons.append(_refusal("packer refused the row", error))
        return report

    ids, labels = sample.input_ids, sample.labels
    report.total_tokens = len(ids)
    report.loss_tokens = sum(1 for label in labels if label != _IGNORE)
    if len(labels) != len(ids):
        report.reasons.append(f"labels ({len(labels)}) and input_ids ({len(ids)}) differ in length")

    def encode(text: str) -> Sequence[int]:
        return packer.tokenizer(text, add_special_tokens=False)["input_ids"]

    blocks = assistant_block_counts(encode, rendered)
    report.assistant_blocks = len(blocks)
    report.body_tokens = [block.body_tokens for block in blocks]
    # The blocks the row's rule supervises (every one, or the last alone), counted as the mask is defined: the body's
    # fragments plus ``<|im_end|>``, minus the ``<think>`` that opens the body where the packer leaves it at -100: under
    # the run's ``mask_think_opener`` or the row's ``opener: prefilled``, one rule applied once (a block that opens with
    # anything else loses nothing; under ``prefilled`` the packer has refused such a block already).
    keep = supervised_assistant_blocks(len(blocks), target.supervise) if blocks else []
    report.supervised_blocks = len(keep)
    masks_opener = bool(packer.mask_think_opener) or target.opener == "prefilled"
    independent = [
        blocks[i].body_tokens + blocks[i].end_tokens - (1 if masks_opener and blocks[i].opens_with_think else 0)
        for i in keep
    ]
    report.assistant_tokens = sum(independent)
    report.loss_tokens_per_block = loss_tokens_per_block(labels, _token_id(packer.tokenizer, "<|im_end|>"))
    if report.total_tokens > max_length:
        report.reasons.append(f"{report.total_tokens} tokens exceed rl.max_length={max_length}")
    if report.loss_tokens == 0:
        report.reasons.append("zero loss tokens")
    report.mask_matches = report.loss_tokens == report.assistant_tokens and report.loss_tokens_per_block == independent
    if report.loss_tokens != report.assistant_tokens:
        report.reasons.append(
            f"loss tokens {report.loss_tokens} != independent assistant tokens {report.assistant_tokens}; "
            f"per assistant block {report.loss_tokens_per_block} vs {independent}"
        )
    elif report.loss_tokens_per_block != independent:
        report.reasons.append(
            f"loss tokens per assistant block {report.loss_tokens_per_block} != independent {independent} "
            f"(totals agree: {report.loss_tokens})"
        )
    empty = [position for position in keep if blocks[position].body_tokens == 0]
    if empty and not allow_empty_assistant:
        report.reasons.append(
            f"empty assistant body in assistant block(s) {empty}: only <|im_end|> would be supervised"
        )

    mm_kwargs = sample.mm_kwargs or {}
    processor = packer.processor
    report.image_grids, image_tokens = _grid_tokens(
        mm_kwargs.get("image_grid_thw"), getattr(processor, "image_processor", None), "image", report.reasons
    )
    report.video_grids, video_tokens = _grid_tokens(
        mm_kwargs.get("video_grid_thw"), getattr(processor, "video_processor", None), "video", report.reasons
    )
    report.visual_tokens = (image_tokens or 0) + (video_tokens or 0)
    if report.image_grids != report.images:
        report.reasons.append(f"{report.images} image part(s) in the source but {report.image_grids} image grid(s)")
    if report.video_grids != report.videos:
        report.reasons.append(f"{report.videos} video part(s) in the source but {report.video_grids} video grid(s)")
    image_pad, video_pad = _token_id(packer.tokenizer, "<|image_pad|>"), _token_id(packer.tokenizer, "<|video_pad|>")
    report.image_pad_tokens = ids.count(image_pad) if image_pad is not None else 0
    report.video_pad_tokens = ids.count(video_pad) if video_pad is not None else 0
    # Each modality against its own pad token: the model forward scatters image and video features separately,
    # so one combined sum would let an image surplus cancel a video deficit and pass a row that dies in training.
    for kind, tokens, pads in (
        ("image", image_tokens, report.image_pad_tokens),
        ("video", video_tokens, report.video_pad_tokens),
    ):
        if tokens is not None and tokens != pads:
            report.reasons.append(
                f"{kind} tokens from {kind}_grid_thw ({tokens}) != <|{kind}_pad|> tokens in the stream ({pads})"
            )
    return report


def summarize(
    reports: list[RowReport],
    *,
    split: str,
    max_length: int,
    dp_world: int | None = None,
    load_error: str | None = None,
    census: dict[str, Any] | None = None,
    census_error: str | None = None,
) -> dict[str, Any]:
    rows = len(reports)
    failed = [report.row for report in reports if not report.ok]
    lengths = [report.total_tokens for report in reports if report.total_tokens is not None]
    summary: dict[str, Any] = {
        "split": split,
        "rows": rows,
        "failed": len(failed) + (1 if load_error is not None else 0),
        "failed_rows": failed[:100],
        "unmeasured_rows": sum(1 for report in reports if report.total_tokens is None),
        "matched_rows": sum(1 for report in reports if report.mask_matches is True),
        "longest_tokens": max(lengths, default=0),
        "max_length": max_length,
        "over_long_rows": sum(1 for length in lengths if length > max_length),
        "total_tokens": sum(lengths),
        "loss_tokens": sum(report.loss_tokens or 0 for report in reports),
        "images": sum(report.images for report in reports),
        "videos": sum(report.videos for report in reports),
        "visual_tokens": sum(report.visual_tokens or 0 for report in reports),
        "dp_world": dp_world,
        "dp_remainder": (rows % dp_world) if dp_world else 0,
        "load_error": load_error,
        # The setup's held-out census of this (validation) split against the training split (``held_out_census_entry``:
        # the setup's ``source_overlap`` then ``refuse_source_overlap``; review threads b9b8f14577 and 7bb34b5b76): the
        # census dict when it passed, the refusal text when the setup would refuse the pair or the census could not read
        # a source row; both ``None`` for the training split and when the training split was not loaded in the same run.
        "held_out_census": census,
        "census_error": census_error,
    }
    summary["failed"] += 1 if census_error is not None else 0
    summary["loss_share"] = (
        round(summary["loss_tokens"] / summary["total_tokens"], 4) if summary["total_tokens"] else 0.0
    )
    summary["ok"] = (
        not failed and not summary["dp_remainder"] and rows > 0 and load_error is None and census_error is None
    )
    return summary


def gate_line(summary: dict[str, Any]) -> str:
    if summary.get("load_error") is not None:
        return f"{summary['split']}: rows 0 | load refused: {summary['load_error']} | failed {summary['failed']}"
    longest, cap = summary["longest_tokens"], summary["max_length"]
    comparison = "<=" if longest <= cap else ">"
    line = (
        f"{summary['split']}: rows {summary['rows']} | loss_tokens == assistant_tokens on "
        f"{summary['matched_rows']}/{summary['rows']} | longest {longest} {comparison} {cap} | failed {summary['failed']}"
    )
    if summary["dp_world"]:
        line += f" | rows % dp_world({summary['dp_world']}) = {summary['dp_remainder']}"
    if summary.get("census_error") is not None:
        line += f" | held-out census refused: {summary['census_error']}"
    elif summary.get("held_out_census") is not None:
        census = summary["held_out_census"]
        line += f" | held-out census: shared rows {census['shared_rows']}, shared items {census['shared_items']}"
    return line


def validate_dataset(
    packer: Any,
    dataset: Any,
    *,
    max_length: int,
    split: str = "train",
    allow_empty_assistant: bool = False,
    dp_world: int | None = None,
    census: dict[str, Any] | None = None,
    census_error: str | None = None,
) -> tuple[list[RowReport], dict[str, Any]]:
    """Every row of a map-style source (``len`` + integer ``__getitem__`` returning normalized rows).

    A source that refuses a row on read (a factory's ``normalize_row`` for a missing media file, an unsupported
    role, a bad part) is reported as that row's reason, never raised: the split keeps its FAIL lines and summary. The
    row is read through ``_read_row``, the census's reader too, so the two file one defect under one class and message.
    """
    if dp_world is not None and (not isinstance(dp_world, int) or isinstance(dp_world, bool) or dp_world < 1):
        raise ValueError("dp_world must be a positive integer")
    reports = []
    for index in range(len(dataset)):
        try:
            row = _read_row(dataset, index)
        except Exception as error:  # noqa: BLE001 -- the source's refusal is this row's defect
            reports.append(
                RowReport(split=split, row=index + 1, index=index, reasons=[_refusal("source refused the row", error)])
            )
            continue
        reports.append(
            inspect_row(
                packer,
                row,
                split=split,
                index=index,
                max_length=max_length,
                allow_empty_assistant=allow_empty_assistant,
            )
        )
    return reports, summarize(
        reports, split=split, max_length=max_length, dp_world=dp_world, census=census, census_error=census_error
    )


def write_report(handle: TextIO, reports: list[RowReport], summary: dict[str, Any]) -> None:
    """One JSON line per row (counts and redacted reasons only), then the split's summary line."""
    for report in reports:
        handle.write(json.dumps(asdict(report), sort_keys=True) + "\n")
    handle.write(json.dumps({"summary": summary}, sort_keys=True) + "\n")


def default_report_path(directory: str) -> str:
    """Create and return a fresh ``validate_sft_rows_<timestamp>[_<n>].jsonl`` under ``directory``.

    The file is created with ``O_CREAT | O_EXCL`` (and closed at once), so two mints in the same second, even
    from two processes, never return the same path; the caller then opens it for writing.
    """
    os.makedirs(directory, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    counter = 0
    while True:
        suffix = f"_{counter}" if counter else ""
        candidate = os.path.join(directory, f"{REPORT_NAME}_{stamp}{suffix}{REPORT_SUFFIX}")
        try:
            os.close(os.open(candidate, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644))
        except FileExistsError:
            counter += 1
            continue
        return candidate


CENSUS_READ_ERROR = (
    "held-out census could not read <side> source {row <n>[ (item <id>)] | (<call>)}: <class>: <message>"
)
"""The validation split's ``census_error`` when the held-out census failed while it asked a source for something: the one place
where the sentence's shape is typed. ``<side>`` is ``training`` or ``validation``. ``row <n>[ (item <id>)]`` is the 1-based row
the census was READING (the factory's ``__getitem__``, the loader's ``normalize_row``) or whose media it was HASHING right after
(``sha256_file``), with the row's ``meta.item`` once the source returned the row. ``(<call>)`` names the source call that failed
while no row was being read: ``(len())`` for ``__len__`` (the ``range(len(...))`` that starts a side's walk; the row counts of
the census dict after both walks), ``(.rows)`` and ``(.media_hashes)`` for the attribute reads (a JSONL source's in-memory rows,
which the census walks without asking for an index, and its media digests, which seed the census's digest cache). The side and
the call are the ones the census asked for LAST (``_CensusReads`` stamps every call on a clock the two sides share), never a
guess from the walk order. The README's ``census_error`` term quotes it (a test holds the two equal), the docstrings name it,
and the code prints its prefix through ``_CensusReads.where()`` and ``_refusal`` (the class named unless it is a plain
``ValueError`` or ``TypeError``, the item id and every path redacted, the whole sentence bounded)."""
_CENSUS_READ_PREFIX = CENSUS_READ_ERROR.split(" <side>", 1)[0]


class _CensusReads:
    """The census's view of one source: hands out the source's rows, its length and its ``rows`` / ``media_hashes`` attributes
    (a JSONL source's in-memory list, which the census walks directly, and its media digests, which seed the census's digest
    cache), and stamps every one of these calls on a clock the two sides share, so a failure inside ``source_overlap`` is named
    by the side and the call the census asked for LAST: the row (number and item id) while a row was being read or its media
    hashed right after, else the call itself (``len()``, ``.rows``, ``.media_hashes``). ``__len__`` and the attribute reads clear
    the row, because no row is in flight while they run (when the side and the row were taken from the last
    ``__getitem__`` alone, a validation ``__len__`` that failed when the validation walk started was reported as the
    training source's last row, and one that failed in the census dict as the validation row read last)."""

    def __init__(self, dataset: Any, side: str, clock: Iterator[int]) -> None:
        self.dataset = dataset
        self.side = side
        # shared by the two sides: ``next(clock)`` stamps every call, so the side the census asked LAST is known
        self.clock = clock
        self.stamp = -1  # the clock value of the last call the census made on this source; -1 before the first
        self.call: str | None = None  # that call: "row", "len()", ".rows" or ".media_hashes"
        self.index: int | None = None  # 0-based index of the row being read, while ``call`` is "row"
        self.row: Any = None  # that row once the source returned it; None while the read itself failed

    def _touch(self, call: str, index: int | None = None) -> None:
        self.stamp, self.call, self.index, self.row = next(self.clock), call, index, None

    def __len__(self) -> int:
        self._touch("len()")
        return len(self.dataset)

    def __getitem__(self, index: int) -> Any:
        self._touch("row", index)
        self.row = _read_row(self.dataset, index)
        return self.row

    def __getattr__(self, name: str) -> Any:
        if name in {"rows", "media_hashes"}:
            self._touch(f".{name}")
            return getattr(self.dataset, name)
        raise AttributeError(name)

    def where(self) -> str:
        """``validation source row 3 (item held/G)`` while a row was being read (the item once the source returned it), else
        ``validation source (len())``, the call that failed; the item id is redacted like a path and cut at ``_ITEM_LIMIT`` so
        that the class and the message after it survive the reason's bound."""
        if self.index is None:
            return f"{self.side} source ({self.call})"
        meta = self.row.get("meta") if isinstance(self.row, dict) else None
        item = meta.get("item") if isinstance(meta, dict) else None
        suffix = f" (item {redact(item, limit=_ITEM_LIMIT)})" if isinstance(item, str) and item else ""
        return f"{self.side} source row {self.index + 1}{suffix}"


def held_out_census_entry(training: Any, validation: Any) -> tuple[dict[str, Any] | None, str | None]:
    """The setup's held-out guard as the validation split's report entry: ``(census, None)`` when the pair passes,
    ``(None, census_error)`` otherwise.

    ``entrypoints/sft.py::held_out_census`` runs two shared helpers in order and refuses the launch when either fails:
    ``source_overlap`` (the census: it reads every row of both sources and hashes their media by file bytes) and
    ``refuse_source_overlap`` (the disjointness rules on the census dict). The validator runs the same two helpers in the
    same order, never a copy of either, and reports instead of refusing, so the validation split still gets its row
    inspection, its FAIL lines, its gate line and its summary:

    - a source that fails while the census reads one of its rows, hashes its media, counts it (``len()``) or hands over its
      ``rows`` or ``media_hashes``, whatever ``Exception`` the call raised, is the sentence ``CENSUS_READ_ERROR`` through
      ``_refusal`` (the side and the call the census asked for last, the row and its item while a row was being read; class
      named, item id and paths redacted, the whole bounded); the census is then unavailable, which is a failure of the split,
      as it is a refusal of the launch in the setup;
    - a pair the rules refuse keeps the setup's own sentence, redacted and bounded;
    - a failure before the census asked either source for anything is the census's own, not a source's, and propagates like
      a defect of the validator's code (not reachable with the setup's ``source_overlap``, whose first statement reads both
      sides' ``media_hashes``);
    - a ``BaseException`` outside ``Exception`` (``KeyboardInterrupt``, ``SystemExit``) propagates, as everywhere.
    """
    from cosmos_framework.posttrain.data.datasets.sft import refuse_source_overlap, source_overlap

    clock = itertools.count()
    sides = (_CensusReads(training, "training", clock), _CensusReads(validation, "validation", clock))
    try:
        census = source_overlap(*sides)
    except Exception as error:  # noqa: BLE001 -- the call's failure is that source's defect, as in validate_dataset
        # The same boundary as ``validate_dataset`` around the same ``_read_row`` of the same row: a malformed record is refused
        # by whatever class the factory's own code happens to raise (a ``KeyError`` for a record without ``messages``, an
        # ``AttributeError`` for a string content part, an ``IndexError`` for a ``__len__`` that over-counts, a vendor
        # decoder's own ``Exception`` subclass, a ``RuntimeError``), and the per-row reader files every one of them as
        # ``source refused the row``; a narrower tuple here made the verdict depend on that class. The source at fault
        # is the one the census asked for LAST, whatever the call: ``source_overlap`` reads both sides' ``media_hashes``
        # first, walks the training rows, then the validation rows, and counts both at the end, so the walk order alone
        # named the wrong side or a row that had read fine (finding R4-01).
        failed = max(sides, key=lambda side: side.stamp)
        if failed.call is None:
            raise  # neither source was asked for anything yet: the census's own failure, not a source's
        return None, _refusal(f"{_CENSUS_READ_PREFIX} {failed.where()}", error)
    try:
        refuse_source_overlap(census)
    except ValueError as error:
        return None, redact(error)
    return census, None


def validate_sources(
    packer: Any,
    sources: Sequence[tuple[str, Any, int | None]],
    *,
    max_length: int,
    out: str,
    allow_empty_assistant: bool = False,
    force: bool = False,
    stream: TextIO | None = None,
) -> int:
    """Validate ``(split name, SFTDatasetConfig, dp_world)`` sources into one report; 0 only when every split is ok.

    Exit 2 when ``out`` exists and ``force`` is false (an explicit report path is never overwritten silently).
    A split refused at load, an empty split, or a held-out source identical to the training source each get a
    FAIL line and a summary line with ``load_error`` set. The validation split is then put through the SAME held-out
    contract the setup enforces (``data/datasets/sft.py::source_overlap`` then ``refuse_source_overlap``, as
    ``entrypoints/sft.py::held_out_census`` runs them: no shared row by content hash with media by file bytes, no shared
    item id, and item ids on the validation rows when the training rows carry them; review thread b9b8f14577): a
    refusal, or a census that could not read a source row (``held_out_census_entry``; review thread 7bb34b5b76), is a
    FAIL line and ``census_error`` in the summary and the split's rows are still inspected; a pass puts the census in
    the summary as ``held_out_census``. The census needs the training split loaded in the same run (``--split all``).
    """
    from cosmos_framework.posttrain.data.datasets.sft import build_sft_dataset

    stream = stream or sys.stdout
    if os.path.exists(out) and not force:
        print(f"validate_sft_rows: refusing to overwrite the existing report {out} (pass --force)", file=stream)
        return 2
    status = 0
    fingerprints: dict[str, str] = {}
    train_dataset: Any = None
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        for name, source, world in sources:
            refusal: str | None = None
            dataset = None
            try:
                dataset = build_sft_dataset(source)
                if len(dataset) == 0:
                    refusal = "source has no rows"
            except Exception as error:  # noqa: BLE001 -- the source's refusal is the split's defect
                refusal = _refusal("source refused", error)[len("source refused: ") :]
            if refusal is None and dataset is not None:
                fingerprint = getattr(dataset, "cursor_fingerprint", lambda: None)()
                twin = next((other for other, value in fingerprints.items() if value == fingerprint), None)
                if fingerprint is not None and twin is not None:
                    refusal = f"source fingerprint equals the {twin} source (the SFT setup refuses identical sources)"
                elif fingerprint is not None:
                    fingerprints[name] = fingerprint
            if refusal is not None:
                print(f"FAIL {name}: {refusal}", file=stream)
                summary = summarize([], split=name, max_length=max_length, dp_world=world, load_error=refusal)
                write_report(handle, [], summary)
                print(gate_line(summary), file=stream)
                status = 1
                continue
            census: dict[str, Any] | None = None
            census_error: str | None = None
            if name == "train":
                train_dataset = dataset
            elif name == "validation":
                if train_dataset is None:
                    print(
                        "validation: held-out census not run (the train split was not loaded in this run; "
                        "use --split all to apply the setup's disjointness contract)",
                        file=stream,
                    )
                else:
                    census, census_error = held_out_census_entry(train_dataset, dataset)
                    if census_error is not None:
                        print(f"FAIL {name}: {census_error}", file=stream)
            reports, summary = validate_dataset(
                packer,
                dataset,
                max_length=max_length,
                split=name,
                allow_empty_assistant=allow_empty_assistant,
                dp_world=world,
                census=census,
                census_error=census_error,
            )
            write_report(handle, reports, summary)
            for report in reports:
                if not report.ok:
                    print(f"FAIL {name} row {report.row}: " + "; ".join(report.reasons), file=stream)
            if summary["dp_remainder"]:
                print(
                    f"FAIL {name}: {summary['rows']} rows are not divisible by dp_world={summary['dp_world']} "
                    "(the SFT source refuses this at setup; no rows are dropped)",
                    file=stream,
                )
            print(gate_line(summary), file=stream)
            if name == "train" and world is None:
                print(
                    "train: dp_world not checked (run under torchrun or pass --dp-world to "
                    "scripts/validate_sft_rows.py; the setup refuses a row count the DP world does not divide)",
                    file=stream,
                )
            if not summary["ok"]:
                status = 1
    print(f"validate_sft_rows: report {out}", file=stream)
    return status


def validate_from_config(
    config: Any,
    *,
    out: str = "",
    report_dir: str = "",
    split: str = "all",
    allow_empty_assistant: bool = False,
    dp_world: int | None = None,
    force: bool = False,
    stream: TextIO | None = None,
) -> int:
    """Run the configured sources through the trainer's packer; 0 when every row may train, 1 otherwise.

    ``out`` names the report; when empty a fresh timestamped report is minted under ``report_dir`` (the job's
    output directory when that is empty too). Only the STRUCTURAL configuration checks run here
    (``validate_config_structure``: backend, grid, model path, source wiring); each selected source's availability
    is checked by ``validate_sources`` when it loads the source, so a missing or unreadable source is that split's
    load refusal (FAIL line, gate line, ``load_error``, redacted path) and an unavailable split never prevents the
    other split from being inspected.
    """
    from cosmos_framework.posttrain.data.packer.sft import build_sft_packer
    from cosmos_framework.posttrain.entrypoints.sft import sft_policy_source, validate_config_structure
    from cosmos_framework.posttrain.utils.entry import out_dir

    stream = stream or sys.stdout
    rl = validate_config_structure(config)
    sources: list[tuple[str, Any, int | None]] = []
    if split in ("train", "all"):
        sources.append(("train", rl.data, dp_world))
    if split in ("validation", "all") and rl.validation.enabled:
        sources.append(("validation", rl.validation.data, None))
    if not sources:
        print(
            f"validate_sft_rows: no source to validate for split={split!r} (validation enabled: {rl.validation.enabled})",
            file=stream,
        )
        return 1
    # The same artifact the trainer's packer is built from, through the trainer's own resolver
    # (``entrypoints.sft.sft_policy_source``: model_hf.name_or_path on the HF seat, the registered native source on the
    # i4 seat; review thread 037abda8b0), with the same knobs ``build_trainer`` passes (processor_kwargs,
    # trust_remote_code, the run-level mask_think_opener, which ``SFTConfig`` has already judged by ``rl.serve_contract``);
    # only the length cap differs.
    name_or_path, trust_remote_code = sft_policy_source(config)
    packer = build_sft_packer(
        name_or_path,
        max_length=MEASURE_MAX_LENGTH,
        processor_kwargs=rl.processor_kwargs,
        trust_remote_code=trust_remote_code,
        mask_think_opener=rl.mask_think_opener,
    )
    minted = not out
    if minted:
        out = default_report_path(report_dir or out_dir(config))  # created empty here, hence force below
    return validate_sources(
        packer,
        sources,
        max_length=rl.max_length,
        out=out,
        allow_empty_assistant=allow_empty_assistant,
        force=force or minted,
        stream=stream,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run every SFT source row through the real packer on CPU and refuse what would die at step 0.",
        epilog="Overrides follow `--` as an OmegaConf dotlist, exactly as for the entrypoint.",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--out",
        default="",
        help=f"JSONL report path (default: a fresh <job out_dir>/{REPORT_NAME}_<timestamp>{REPORT_SUFFIX})",
    )
    parser.add_argument("--force", action="store_true", help="overwrite an existing --out report")
    parser.add_argument("--split", choices=("train", "validation", "all"), default="all")
    parser.add_argument(
        "--dp-world",
        type=int,
        default=None,
        help="refuse when the training row count is not divisible by this DP world size",
    )
    parser.add_argument(
        "--allow-empty-assistant",
        action="store_true",
        help="accept assistant turns whose body is empty (only <|im_end|> supervised)",
    )
    args, rest = parser.parse_known_args(argv)
    if "--" in rest:
        unknown = [item for item in rest[: rest.index("--")] if "=" not in item]
        rest = rest[rest.index("--") + 1 :]
    else:
        unknown = [item for item in rest if "=" not in item]
    if unknown:
        parser.error(f"unknown argument(s) {unknown}; overrides belong after `--`")
    opts = [item for item in rest if "=" in item]

    from cosmos_framework.utils.config import load_config

    config = load_config(args.config, ["--", *opts])
    return validate_from_config(
        config,
        out=args.out,
        split=args.split,
        allow_empty_assistant=args.allow_empty_assistant,
        dp_world=args.dp_world,
        force=args.force,
    )


if __name__ == "__main__":
    raise SystemExit(main())
