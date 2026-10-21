# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Pre-launch SFT row validator: role grammar, independent assistant-token counts, media tokens, the entry flag.

The grammar and CLI tests are pure Python. The packer-backed tests build ``QwenSFTDataPacker`` on the local test
tokenizer/processor directly and need Transformers >= 5.4.0 (``_PACKER_TRANSFORMERS_FLOOR``: the packer's render imports
``transformers.utils.chat_template_utils._get_template_variables``, first exported at the v5.4.0 tag; below it they SKIP with
that reason instead of failing on the ImportError, R149T-06); only ``build_sft_packer`` carries the >= 5.8.1 gate, which the
entry-flag test bypasses with an equivalent local packer when the running Transformers is older. The
behavioural fail-on-parent node for the entry flag (no module-level import of the validator) lives in
``test_entrypoints_sft_validate_rows.py``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

from cosmos_framework.posttrain.scripts import validate_sft_rows as validator
from cosmos_framework.posttrain.utils.entry import cli, cli_flags

pytestmark = [pytest.mark.L0, pytest.mark.CPU]

_RECIPE = "cosmos_framework/posttrain/configs/sft_qwen3p8_27b_multimodal.py"
_RESERVED_DELIMITER = "<|im_end|>"


def _call(name: str, **arguments: Any) -> dict[str, Any]:
    return {"function": {"name": name, "arguments": arguments}}


def _assistant(content: str = "", calls: int = 0, **extra: Any) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content, **extra}
    if calls:
        message["tool_calls"] = [_call(f"tool{i}", x=i) for i in range(calls)]
    return message


def _agentic_row() -> dict[str, Any]:
    """One harness-shaped episode: reasoning, a single call, a parallel pair, a final answer."""
    return {
        "id": "episode-1",
        "messages": [
            {"role": "system", "content": "Inspect the board."},
            {"role": "user", "content": "Which option?"},
            {
                "role": "assistant",
                "content": "Looking.",
                "reasoning_content": "Zoom first.",
                "tool_calls": [_call("zoom", box=[0, 0, 8, 8])],
            },
            {"role": "tool", "content": "zoomed"},
            {"role": "assistant", "content": "", "tool_calls": [_call("measure", a=1), _call("measure", a=2)]},
            {"role": "tool", "content": "3 mm"},
            {"role": "tool", "content": "4 mm"},
            {"role": "assistant", "content": "The answer is B."},
        ],
        "tools": [
            {"type": "function", "function": {"name": "zoom"}},
            {"type": "function", "function": {"name": "measure"}},
        ],
    }


# ----------------------------------------------------------------------------- role grammar (pure Python)


def test_grammar_accepts_harness_shapes() -> None:
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "environment"},
        {"role": "user", "content": "task"},
        _assistant("a", calls=1),
        {"role": "tool", "content": "r"},
        _assistant("", calls=2),
        {"role": "tool", "content": "r1"},
        {"role": "tool", "content": "r2"},
        _assistant("b", calls=1),
        {"role": "tool", "content": "r"},
        _assistant("final"),
    ]
    assert validator.role_grammar_failures(messages) == []
    assert validator.role_grammar_failures(_agentic_row()["messages"]) == []


def test_grammar_names_a_tool_turn_mislabelled_as_assistant() -> None:
    messages = [{"role": "user", "content": "q"}, _assistant("call", calls=1), _assistant("found"), _assistant("done")]
    assert validator.role_grammar_failures(messages) == [
        "message 2: assistant turn where 1 tool result(s) were still expected (a tool turn mislabelled as assistant?)"
    ], "the pending-result rule names the mislabelled turn; back-to-back assistant turns are not a defect by themselves"


def test_grammar_accepts_consecutive_assistant_turns() -> None:
    """Review thread 56678e57c8: ``normalize_row`` and the packer accept back-to-back assistant turns (each its own block)."""
    user = {"role": "user", "content": "q"}
    assert validator.role_grammar_failures([user, _assistant("a"), _assistant("b")]) == []
    with_calls = [user, _assistant("call", calls=1), {"role": "tool", "content": "r"}, _assistant("a"), _assistant("b")]
    assert validator.role_grammar_failures(with_calls) == []
    assert validator.role_grammar_failures([{"role": "system", "content": "s"}, _assistant("a"), _assistant("b")]) == []


@pytest.mark.parametrize(
    ("messages", "expected"),
    [
        (
            [{"role": "user", "content": "q"}, _assistant("a"), {"role": "tool", "content": "r"}, _assistant("b")],
            "message 2: tool result without a preceding tool call",
        ),
        (
            [
                {"role": "user", "content": "q"},
                _assistant("a", calls=2),
                {"role": "tool", "content": "r"},
                _assistant("b"),
            ],
            "message 3: assistant turn where 1 tool result(s) were still expected (a tool turn mislabelled as assistant?)",
        ),
        (
            [
                {"role": "user", "content": "q"},
                _assistant("a", calls=1),
                {"role": "user", "content": "q2"},
                _assistant("b"),
            ],
            "message 2: user turn where 1 tool result(s) were still expected",
        ),
        (
            [{"role": "user", "content": "q"}, _assistant("a", calls=1), {"role": "tool", "content": "r"}],
            "message 2: conversation does not end with an assistant turn",
        ),
        (
            [{"role": "user", "content": "q"}, _assistant("a", calls=1)],
            "message 1: conversation ends with 1 tool call(s) and no result",
        ),
        (
            [{"role": "user", "content": "q"}, {"role": "system", "content": "s"}, _assistant("a")],
            "message 1: system message is not first",
        ),
        (
            [{"role": "user", "content": "q"}, {"role": "developer", "content": "d"}, _assistant("a")],
            "message 1: unsupported role 'developer'",
        ),
        ([{"role": "user", "content": "q"}], "no assistant turn to supervise"),
        ([], "messages must be a non-empty list"),
        ("not a list", "messages must be a non-empty list"),
    ],
)
def test_grammar_defects_are_named(messages: Any, expected: str) -> None:
    assert expected in validator.role_grammar_failures(messages)


def test_media_counts_span_every_message() -> None:
    messages = [
        {"role": "user", "content": [{"type": "image", "image": "a.png"}, {"type": "text", "text": "q"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "x"}, {"type": "video", "video": "v.mp4"}]},
        {"role": "tool", "content": [{"type": "image", "image": "b.png"}]},
        {"role": "assistant", "content": "plain string content is not media"},
    ]
    assert validator.media_counts(messages) == (2, 1)
    assert validator.media_counts("junk") == (0, 0)


# ----------------------------------------------------------------------------- CLI plumbing (pure Python)


def test_cli_flags_adds_entry_flags_and_cli_stays_compatible(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["sft", "--validate-rows", "--", "rl.max_length=8"])
    config, opts, flags = cli_flags("sft", _RECIPE, extra_flags={"--validate-rows": "rows"})
    assert (config, opts) == (_RECIPE, ["rl.max_length=8"])
    assert flags == {"validate_manifest": False, "validate_rows": True}
    assert cli("sft", _RECIPE) == (_RECIPE, ["rl.max_length=8"], False)
    monkeypatch.setattr(sys, "argv", ["sft", "--validate-manifest", "--", "a=b"])
    assert cli_flags("sft", _RECIPE, extra_flags={"--validate-rows": "rows"})[2] == {
        "validate_manifest": True,
        "validate_rows": False,
    }


def test_script_main_refuses_unknown_flags_before_loading_anything() -> None:
    with pytest.raises(SystemExit) as exc:
        validator.main(["--allow-empty-assistent", "--", "rl.max_length=8"])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        validator.main(["rl.max_length=8", "--bogus"])
    assert exc.value.code == 2


# ----------------------------------------------------------------------------- packer-backed


# The packer's render imports ``transformers.utils.chat_template_utils._get_template_variables`` (``data/packer/sft.py``,
# ``_template_variables``). The symbol first appears at the v5.4.0 tag (absent through v5.3.0, no v5.3.x tag: a bisect of
# src/transformers/utils/chat_template_utils.py over the release tags, round 149t fix R149T-06). Below this floor the
# packer-backed tests used to FAIL on "packer refused the row: ImportError: cannot import name '_get_template_variables'"
# (31 of them under 4.57.1) instead of skipping: the old floor "4.57" was the tokenizer classes' floor, not the render's.
_PACKER_TRANSFORMERS_FLOOR = "5.4.0"


def _packer_module() -> Any:
    """The packer module, or a SKIP whose reason names the floor and the symbol behind it (R149T-06)."""
    transformers = pytest.importorskip("transformers")
    from packaging.version import Version

    if Version(transformers.__version__) < Version(_PACKER_TRANSFORMERS_FLOOR):
        pytest.skip(
            f"Transformers {transformers.__version__} < {_PACKER_TRANSFORMERS_FLOOR}: QwenSFTDataPacker's render imports "
            "transformers.utils.chat_template_utils._get_template_variables, first exported at the v5.4.0 tag"
        )
    from cosmos_framework.posttrain.data.packer import sft as packer_module

    return packer_module


@pytest.fixture
def text_packer() -> Any:
    module = _packer_module()
    from cosmos_framework.posttrain.tests.helpers.sft_packer import local_tokenizer

    return module.QwenSFTDataPacker(local_tokenizer(), max_length=validator.MEASURE_MAX_LENGTH)


@pytest.fixture
def vlm_packer() -> Any:
    module = _packer_module()
    from cosmos_framework.posttrain.tests.helpers.sft_packer import local_processor, local_tokenizer

    tokenizer = local_tokenizer()
    return module.QwenSFTDataPacker(
        tokenizer, processor=local_processor(tokenizer), max_length=validator.MEASURE_MAX_LENGTH
    )


def _image() -> Any:
    import numpy as np
    from PIL import Image

    return Image.fromarray(np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3))


def _video() -> Any:
    import numpy as np

    return np.arange(4 * 8 * 8 * 3, dtype=np.uint8).reshape(4, 8, 8, 3)


def _supervised_text(packer: Any, row: dict[str, Any]) -> str:
    from cosmos_framework.posttrain.schema import SFTTarget

    target = SFTTarget(
        messages=row["messages"],
        tools=row.get("tools"),
        supervise=row.get("supervise", "all_assistant"),
        opener=row.get("opener", "self"),
    )
    sample = packer.sft_process_sample(target)
    return packer.tokenizer.decode([token for token in sample.labels if token != -100])


def test_agentic_row_loss_tokens_equal_the_independent_count(text_packer: Any) -> None:
    reports, summary = validator.validate_dataset(text_packer, [_agentic_row()], max_length=4096)
    (report,) = reports
    assert report.ok, report.reasons
    assert report.row == 1 and report.index == 0 and report.row_id == "episode-1"
    assert report.assistant_blocks == 3 and all(count > 0 for count in report.body_tokens)
    assert report.loss_tokens == report.assistant_tokens > 0
    assert report.mask_matches is True
    assert report.loss_tokens_per_block == [body + 1 for body in report.body_tokens]
    assert report.total_tokens > report.loss_tokens
    assert (report.images, report.videos, report.visual_tokens) == (0, 0, 0)
    assert summary["matched_rows"] == 1 and summary["ok"] and summary["longest_tokens"] == report.total_tokens
    assert validator.gate_line(summary) == (
        f"train: rows 1 | loss_tokens == assistant_tokens on 1/1 | longest {report.total_tokens} <= 4096 | failed 0"
    )
    supervised = _supervised_text(text_packer, _agentic_row())
    assert supervised.count("<|im_end|>") == 3 and "zoomed" not in supervised and "Inspect" not in supervised


def _p_layout_row(supervise: str | None = "last_assistant") -> dict[str, Any]:
    """The shape of a real P-layout row (V7, 2026-10-04).

    Terms: *dsh* = the external coding harness the lab's SFT teacher (Kimi-K3) runs under; its episodes are exported as SFT rows.
    *The P layout* = "prefix rows": one row per supervised assistant turn of an episode, the earlier turns kept as the prefix
    (loss 0) and ONLY the last assistant turn supervised (``supervise: "last_assistant"``); when that turn is a tool call the
    row ends on the call, because the serving stack stops there and the tool result opens the next row. *DSH_P_v4* = the lab's
    fourth P-layout pool of dsh episodes on the AOI training items (146,457 rows; the exporter's class filter v4). The real row
    S_type_mcq_v0/00835be2aabec1ba@..._hA_s01_aoi_train_dev_HOWTO_min/dsh_xtf_th/r1/call1 has four messages: system, the task
    turn, the runtime-context turn, the supervised assistant turn = one tool call with empty content. The exemption the row
    exercises trusts the exporter's ``supervise`` field: the consumers that catch a wrongly-labelled row are the exporter's
    episode gate and the pool census (LAST-TURN-NOT-ASSISTANT: 0 rows in DSH_P_v4). The P arms' integration pin carries V7
    (b821fb850fe); V7b and V7c change tests, docstrings and gate scripts only (whitespace and words), never the rule."""
    row: dict[str, Any] = {
        "id": "S_type_mcq_v0/00835be2aabec1ba@root/dsh_xtf_th/r1/call1",
        "messages": [
            {"role": "system", "content": "You are an AI agent powered by DeepSeek Harness."},
            {"role": "user", "content": [{"type": "text", "text": "Solve the task in TASK.md in this directory."}]},
            {"role": "user", "content": [{"type": "text", "text": "Current runtime context."}]},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "Let me start by reading the TASK.md file.",
                "tool_calls": [_call("read", file_path="TASK.md")],
            },
        ],
    }
    if supervise is not None:
        row["supervise"] = supervise
    return row


def test_v7_last_assistant_row_may_end_on_its_own_tool_call() -> None:
    """V7: under supervise last_assistant the dangling call of the LAST assistant turn is the completion; the same messages under
    the whole-episode rule (no supervise, or all_assistant) still fail exactly as on 92f8cab6175. The first assertion calls the
    rule WITHOUT the keyword, so on the parent 92f8cab6175 it is reached and passes (the mechanism-level check); the keyword
    calls that follow raise TypeError there (the parent's one-argument signature), which is the FAIL-on-old of this test."""
    messages = _p_layout_row()["messages"]
    dangling = "message 3: conversation ends with 1 tool call(s) and no result"
    assert validator.role_grammar_failures(messages) == [dangling]  # reachable on the parent: the old rule's own reason
    assert validator.role_grammar_failures(messages, supervise="last_assistant") == []
    assert validator.role_grammar_failures(messages, supervise="all_assistant") == [dangling]
    # two parallel calls on the supervised turn are one completion too
    two = messages[:-1] + [_assistant("", calls=2)]
    assert validator.role_grammar_failures(two, supervise="last_assistant") == []
    assert validator.role_grammar_failures(two) == ["message 3: conversation ends with 2 tool call(s) and no result"]


def test_v7_exemption_covers_only_the_final_supervised_turn() -> None:
    """A last_assistant row with a call left unanswered in an EARLIER turn, one that ends on a tool result, and a whole-episode
    row that ends in a call all still fail: the exemption is the final supervised turn's own calls, nothing else. On the parent
    92f8cab6175 this test fails at the first keyword call (TypeError: the one-argument signature), not at the mechanism; the
    mechanism-level FAIL-on-old lives in test_v7_last_assistant_row_may_end_on_its_own_tool_call (its first assertion) and in
    test_v7_last_assistant_tool_call_row_validates_with_the_mask_unchanged (the parent refuses the row with the dangling reason)."""
    user = {"role": "user", "content": "q"}
    earlier = [user, _assistant("a", calls=1), {"role": "user", "content": "q2"}, _assistant("b")]
    assert validator.role_grammar_failures(earlier, supervise="last_assistant") == [
        "message 2: user turn where 1 tool result(s) were still expected"
    ]
    ends_on_result = [user, _assistant("a", calls=1), {"role": "tool", "content": "r"}]
    assert validator.role_grammar_failures(ends_on_result, supervise="last_assistant") == [
        "message 2: conversation does not end with an assistant turn"
    ]
    episode = _agentic_row()["messages"][:-1] + [_assistant("", calls=1)]
    assert validator.role_grammar_failures(episode, supervise="all_assistant") == [
        "message 7: conversation ends with 1 tool call(s) and no result"
    ]
    assert validator.role_grammar_failures(episode) == [
        "message 7: conversation ends with 1 tool call(s) and no result"
    ]


def test_v7_last_assistant_tool_call_row_validates_with_the_mask_unchanged(text_packer: Any) -> None:
    """The real P-layout shape through the validator: no reason, and the loss tokens are the supervised turn's own tokens (one
    block); the same row under all_assistant carries the grammar reason. On the parent 92f8cab6175 this test fails at the
    MECHANISM: validate_dataset refuses the last_assistant row with "conversation ends with 1 tool call(s) and no result"."""
    reports, summary = validator.validate_dataset(text_packer, [_p_layout_row()], max_length=4096)
    (report,) = reports
    assert report.ok, report.reasons
    assert report.supervise == "last_assistant" and report.assistant_blocks == 1
    assert report.loss_tokens == report.assistant_tokens > 0 and report.mask_matches is True
    assert summary["ok"] and summary["failed"] == 0 and summary["matched_rows"] == 1
    old_rule, _ = validator.validate_dataset(text_packer, [_p_layout_row("all_assistant")], max_length=4096)
    assert "message 3: conversation ends with 1 tool call(s) and no result" in old_rule[0].reasons


def test_v7_malformed_final_turn_call_entry_is_still_refused_by_the_packer(text_packer: Any) -> None:
    """A last_assistant row whose final (supervised) turn carries ``tool_calls: [1, 2]``, entries that are not dicts, is still
    refused on the row-level entry point, and the leg that refuses it is the PACKER, not the grammar: ``validate_dataset`` ->
    ``inspect_row`` -> ``SFTTarget`` (checks only ``supervise``/``opener``) -> ``QwenSFTDataPacker.render_conversation`` ->
    ``_conversation`` raises ``ValueError("SFT tool calls require a function name")``, which becomes the row's one reason. The
    grammar never inspected call entries on any turn (the count rule), and under V7 its final-turn dangling reason no longer
    fires for a last_assistant row, so the packer is the only guard left for this shape (round 149t, R149T-05: the review's
    grammar-shape table called this message list "shape (m)").
    FAIL-on-old at the mechanism: on the parent 92f8cab6175 the same row carries TWO reasons, the whole-episode rule's
    "message 3: conversation ends with 2 tool call(s) and no result" before the packer's, so the exact-list assertion fails."""
    row = _p_layout_row()
    row["messages"][-1]["tool_calls"] = [1, 2]
    (report,), summary = validator.validate_dataset(text_packer, [row], max_length=4096)
    assert report.reasons == ["packer refused the row: SFT tool calls require a function name"]
    assert not report.ok and report.total_tokens is None and report.loss_tokens is None
    assert summary["failed"] == 1 and summary["unmeasured_rows"] == 1 and not summary["ok"]
    # the grammar alone on the same messages: silent under last_assistant (the exemption), the count rule otherwise
    assert validator.role_grammar_failures(row["messages"], supervise="last_assistant") == []
    assert validator.role_grammar_failures(row["messages"]) == [
        "message 3: conversation ends with 2 tool call(s) and no result"
    ]


@pytest.mark.parametrize("defect", ["leak", "drop"])
def test_independent_count_catches_a_wrong_mask(text_packer: Any, monkeypatch: pytest.MonkeyPatch, defect: str) -> None:
    """A packer that supervises a header token, or loses an assistant token, is caught without reading its labels."""
    original = text_packer.sft_process_sample

    def wrong(sample: Any) -> Any:
        out = original(sample)
        if defect == "leak":
            position = out.labels.index(-100)
            out.labels[position] = out.input_ids[position]
        else:
            position = max(i for i, label in enumerate(out.labels) if label != -100)
            out.labels[position] = -100
        return out

    monkeypatch.setattr(text_packer, "sft_process_sample", wrong)
    (report,), summary = validator.validate_dataset(text_packer, [_agentic_row()], max_length=4096)
    delta = 1 if defect == "leak" else -1
    assert report.loss_tokens == report.assistant_tokens + delta
    assert report.mask_matches is False
    assert len(report.reasons) == 1 and report.reasons[0].startswith(
        f"loss tokens {report.loss_tokens} != independent assistant tokens {report.assistant_tokens}; per assistant block "
    )
    assert summary["failed"] == 1 and summary["matched_rows"] == 0 and not summary["ok"]


def test_zero_loss_row_is_refused(text_packer: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    original = text_packer.sft_process_sample

    def unsupervised(sample: Any) -> Any:
        out = original(sample)
        out.labels = [-100] * len(out.labels)
        return out

    monkeypatch.setattr(text_packer, "sft_process_sample", unsupervised)
    (report,), _ = validator.validate_dataset(text_packer, [_agentic_row()], max_length=4096)
    assert report.loss_tokens == 0
    assert report.reasons[0] == "zero loss tokens"
    assert report.reasons[1].startswith("loss tokens 0 != independent assistant tokens")


def test_empty_assistant_body_supervises_only_im_end_and_is_refused(text_packer: Any) -> None:
    """The packer labels ``<|im_end|>`` even for an empty body (one loss token, not zero); the validator refuses the turn."""
    row = {"messages": [{"role": "user", "content": "q"}, _assistant("")]}
    (report,), summary = validator.validate_dataset(text_packer, [row], max_length=4096)
    assert report.loss_tokens == report.assistant_tokens == 1
    assert report.body_tokens == [0]
    assert report.reasons == ["empty assistant body in assistant block(s) [0]: only <|im_end|> would be supervised"]
    assert not summary["ok"]
    (report,), summary = validator.validate_dataset(text_packer, [row], max_length=4096, allow_empty_assistant=True)
    assert report.ok and summary["ok"]


def test_overlong_row_is_named_with_its_length_and_still_measured(text_packer: Any) -> None:
    short = {"messages": [{"role": "user", "content": "q"}, _assistant("ok")]}
    long = {"messages": [{"role": "user", "content": "q"}, _assistant("word " * 200)]}
    probe, _ = validator.validate_dataset(text_packer, [short, long], max_length=validator.MEASURE_MAX_LENGTH)
    cap = probe[0].total_tokens
    assert probe[1].total_tokens > cap
    reports, summary = validator.validate_dataset(text_packer, [short, long], max_length=cap)
    assert reports[0].ok
    assert reports[1].row == 2
    assert reports[1].reasons == [f"{probe[1].total_tokens} tokens exceed rl.max_length={cap}"]
    assert summary["longest_tokens"] == probe[1].total_tokens and summary["over_long_rows"] == 1
    assert summary["failed_rows"] == [2] and not summary["ok"]
    assert validator.gate_line(summary).endswith(f"longest {probe[1].total_tokens} > {cap} | failed 1")


def test_mislabelled_tool_turn_is_refused_by_grammar_while_the_counts_agree(text_packer: Any) -> None:
    """Both the packer and the independent count see an assistant block; only the source grammar knows it is a result."""
    row = {
        "messages": [
            {"role": "user", "content": "q"},
            _assistant("call", calls=1),
            _assistant("zoomed result"),
            _assistant("done"),
        ]
    }
    (report,), _ = validator.validate_dataset(text_packer, [row], max_length=4096)
    assert report.loss_tokens == report.assistant_tokens
    assert report.assistant_blocks == 3
    assert report.reasons == [
        "message 2: assistant turn where 1 tool result(s) were still expected (a tool turn mislabelled as assistant?)"
    ]
    assert "zoomed result" in _supervised_text(text_packer, row), (
        "the mislabelled observation IS supervised by the packer"
    )


def test_consecutive_assistant_turns_pass_and_their_loss_tokens_equal_the_packers(text_packer: Any) -> None:
    """Review thread 56678e57c8: user, assistant ``a``, assistant ``b`` packs as two supervised blocks ([2, 2]).

    The validator's independent count agrees with the packer's labels, and no grammar reason is raised: the row
    trains, so the validator may not refuse it.
    """
    from cosmos_framework.posttrain.schema import SFTTarget

    row = {"messages": [{"role": "user", "content": "q"}, _assistant("a"), _assistant("b")]}
    sample = text_packer.sft_process_sample(SFTTarget(messages=row["messages"]))
    packed_loss_tokens = sum(1 for label in sample.labels if label != -100)
    (report,), summary = validator.validate_dataset(text_packer, [row], max_length=4096)
    assert report.ok, report.reasons
    assert report.assistant_blocks == 2 and report.body_tokens == [1, 1]
    assert report.loss_tokens_per_block == [2, 2]
    assert report.loss_tokens == report.assistant_tokens == packed_loss_tokens == 4
    assert report.mask_matches is True
    assert summary["ok"] and validator.gate_line(summary).endswith("| failed 0")
    assert _supervised_text(text_packer, row) == "a<|im_end|>b<|im_end|>"


def test_packer_refusal_becomes_a_row_reason(text_packer: Any) -> None:
    row = {"messages": [{"role": "user", "content": "q"}, _assistant("answer " + _RESERVED_DELIMITER)]}
    (report,), summary = validator.validate_dataset(text_packer, [row], max_length=4096)
    assert report.total_tokens is None and report.loss_tokens is None
    assert report.reasons == ["packer refused the row: SFT source contains reserved chat/media delimiters"]
    assert summary["unmeasured_rows"] == 1 and summary["longest_tokens"] == 0 and not summary["ok"]


def test_image_tokens_are_counted_from_grids_and_excluded_from_the_assistant_count(vlm_packer: Any) -> None:
    row = {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Describe."}, {"type": "image", "image": _image()}]},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Here:"},
                    {"type": "image", "image": _image()},
                    {"type": "text", "text": "Done."},
                ],
            },
            {"role": "user", "content": "And now?"},
            _assistant("Second answer."),
        ]
    }
    (report,), summary = validator.validate_dataset(vlm_packer, [row], max_length=4096)
    assert report.ok, report.reasons
    assert (report.images, report.videos, report.image_grids, report.video_grids) == (2, 0, 2, 0)
    assert report.image_pad_tokens > 2, "the fixture must exercise expansion, not single markers"
    assert report.visual_tokens == report.image_pad_tokens and report.video_pad_tokens == 0
    assert report.loss_tokens == report.assistant_tokens
    assert _supervised_text(vlm_packer, row) == "Here:Done.<|im_end|>Second answer.<|im_end|>"
    assert summary["visual_tokens"] == report.visual_tokens and summary["images"] == 2


def test_video_tokens_are_counted_from_grids(vlm_packer: Any) -> None:
    """Skipped where this Transformers' video expansion is refused by the packer itself (its own suite fails there too)."""
    from cosmos_framework.posttrain.schema import SFTTarget

    row = {
        "messages": [{"role": "user", "content": [{"type": "video", "video": _video()}]}, _assistant("Moving right.")]
    }
    try:
        vlm_packer.sft_process_sample(SFTTarget(messages=row["messages"]))
    except ValueError as error:
        if "media expansion" in str(error):
            pytest.skip(f"the packer refuses this Transformers' video expansion: {error}")
        raise
    (report,), _ = validator.validate_dataset(vlm_packer, [row], max_length=4096)
    assert report.ok, report.reasons
    assert (report.videos, report.video_grids) == (1, 1)
    assert report.video_pad_tokens > 1 and report.visual_tokens == report.video_pad_tokens
    assert report.loss_tokens == report.assistant_tokens


def test_visual_token_disagreement_is_detected(vlm_packer: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    original = vlm_packer.sft_process_sample

    def lost_grid(sample: Any) -> Any:
        out = original(sample)
        out.mm_kwargs["image_grid_thw"] = out.mm_kwargs["image_grid_thw"][:0]
        return out

    monkeypatch.setattr(vlm_packer, "sft_process_sample", lost_grid)
    row = {"messages": [{"role": "user", "content": [{"type": "image", "image": _image()}]}, _assistant("a")]}
    (report,), _ = validator.validate_dataset(vlm_packer, [row], max_length=4096)
    assert report.reasons == [
        "1 image part(s) in the source but 0 image grid(s)",
        f"image tokens from image_grid_thw (0) != <|image_pad|> tokens in the stream ({report.image_pad_tokens})",
    ]


def test_image_and_video_mismatches_do_not_cancel(vlm_packer: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Review thread 6850f73605: +k image tokens and -k video tokens keep the combined sum equal to the combined pads.

    The model forward scatters each modality separately, so such a row dies in training; the validator compares
    image grids with image pads and video grids with video pads independently and names both.
    """
    from cosmos_framework.posttrain.schema import SFTTarget

    row = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": _image()},
                    {"type": "text", "text": "and"},
                    {"type": "video", "video": _video()},
                ],
            },
            _assistant("Moving right."),
        ]
    }
    try:
        honest = vlm_packer.sft_process_sample(SFTTarget(messages=row["messages"]))
    except ValueError as error:
        if "media expansion" in str(error):
            pytest.skip(f"the packer refuses this Transformers' video expansion: {error}")
        raise
    merge = vlm_packer.processor.image_processor.merge_size**2

    def tokens(grid: Any) -> int:
        return sum(int(t) * int(h) * int(w) for t, h, w in grid.tolist()) // merge

    before = (tokens(honest.mm_kwargs["image_grid_thw"]), tokens(honest.mm_kwargs["video_grid_thw"]))
    assert before == (4, 8), "the fixture geometry the cancellation below is built on"
    original = vlm_packer.sft_process_sample

    def cancelling(sample: Any) -> Any:
        out = original(sample)
        image, video = out.mm_kwargs["image_grid_thw"].clone(), out.mm_kwargs["video_grid_thw"].clone()
        image[0, 2] *= 2  # (1, 4, 4) -> (1, 4, 8): 4 -> 8 image tokens
        video[0, 2] //= 2  # (2, 4, 4) -> (2, 4, 2): 8 -> 4 video tokens
        assert tokens(image) + tokens(video) == sum(before), "the fixture must cancel in the combined sum"
        out.mm_kwargs["image_grid_thw"], out.mm_kwargs["video_grid_thw"] = image, video
        return out

    monkeypatch.setattr(vlm_packer, "sft_process_sample", cancelling)
    (report,), summary = validator.validate_dataset(vlm_packer, [row], max_length=4096)
    assert (report.images, report.videos, report.image_grids, report.video_grids) == (1, 1, 1, 1)
    assert (report.image_pad_tokens, report.video_pad_tokens) == before
    assert report.visual_tokens == report.image_pad_tokens + report.video_pad_tokens, (
        "the combined sum still agrees: a check on the sum is blind to this row"
    )
    assert report.reasons == [
        "image tokens from image_grid_thw (8) != <|image_pad|> tokens in the stream (4)",
        "video tokens from video_grid_thw (4) != <|video_pad|> tokens in the stream (8)",
    ]
    assert summary["failed"] == 1 and not summary["ok"]


def test_dp_world_divisibility_is_a_summary_failure(text_packer: Any) -> None:
    rows = [_agentic_row() for _ in range(3)]
    _, summary = validator.validate_dataset(text_packer, rows, max_length=4096, dp_world=2)
    assert summary["dp_remainder"] == 1 and summary["failed"] == 0 and not summary["ok"]
    assert validator.gate_line(summary).endswith("| rows % dp_world(2) = 1")
    _, summary = validator.validate_dataset(text_packer, rows, max_length=4096, dp_world=3)
    assert summary["ok"]
    with pytest.raises(ValueError, match="dp_world"):
        validator.validate_dataset(text_packer, rows, max_length=4096, dp_world=0)


def test_report_is_one_line_per_row_plus_summary_and_carries_no_content(tmp_path: Path, text_packer: Any) -> None:
    rows = [_agentic_row(), {"messages": [{"role": "user", "content": "q"}, _assistant("")]}]
    reports, summary = validator.validate_dataset(text_packer, rows, max_length=4096)
    path = tmp_path / "report.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        validator.write_report(handle, reports, summary)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert [json.loads(line)["row"] for line in lines[:2]] == [1, 2]
    assert json.loads(lines[2])["summary"]["failed_rows"] == [2]
    assert "Which option" not in path.read_text(encoding="utf-8") and "Inspect" not in path.read_text(encoding="utf-8")


# ----------------------------------------------------------------------------- the entry flag


def _vlm_model_dir(path: Path) -> None:
    from cosmos_framework.posttrain.tests.helpers.sft_packer import local_processor, local_tokenizer

    local_processor(local_tokenizer()).save_pretrained(path)
    (path / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))


def _builder_for_this_transformers(monkeypatch: pytest.MonkeyPatch) -> str:
    """``build_sft_packer`` refuses Transformers < 5.8.1 at line one; below it the test builds the same local packer."""
    import transformers
    from packaging.version import Version

    if Version(transformers.__version__) >= Version("5.8.1"):
        return "real build_sft_packer"
    from cosmos_framework.posttrain.data.packer import sft as packer_module
    from cosmos_framework.posttrain.tests.helpers.sft_packer import local_processor, local_tokenizer

    def local_builder(
        name_or_path: str,
        *,
        max_length: int,
        processor_kwargs: Any = None,
        trust_remote_code: bool = False,
        mask_think_opener: bool = False,
    ) -> Any:
        assert Path(name_or_path, "config.json").is_file(), name_or_path
        tokenizer = local_tokenizer()
        return packer_module.QwenSFTDataPacker(
            tokenizer,
            processor=local_processor(tokenizer),
            max_length=max_length,
            processor_kwargs=processor_kwargs,
            mask_think_opener=mask_think_opener,
        )

    monkeypatch.setattr(packer_module, "build_sft_packer", local_builder)
    return f"local packer (Transformers {transformers.__version__} < 5.8.1)"


def test_validate_rows_flag_names_the_overlong_row_before_any_trainer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """FAIL on the parent: ``--validate-rows`` is unknown there, the flag is dropped and ``run()`` is entered.

    On the parent the same two-row JSONL passes ``--validate-manifest`` and the second row kills the run at step 0
    inside ``algorithm/sft.py`` (``packer.sft_process_sample``), after the model was built.
    """
    _packer_module()
    from cosmos_framework.posttrain.entrypoints import sft as entry

    builder = _builder_for_this_transformers(monkeypatch)
    model = tmp_path / "model"
    model.mkdir()
    _vlm_model_dir(model)
    rows = tmp_path / "rows.jsonl"
    rows.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {"messages": [{"role": "user", "content": "hi"}, _assistant("ok")]},
                {"messages": [{"role": "user", "content": "hi"}, _assistant("word " * 200)]},
            )
        )
        + "\n"
    )
    monkeypatch.setenv("IMAGINAIRE_OUTPUT_ROOT", str(tmp_path / "outputs"))

    def never(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a trainer or run was entered before the rows were validated")

    monkeypatch.setattr(entry, "build_trainer", never)
    monkeypatch.setattr(entry, "run", never)
    overrides = [
        f"model_hf.name_or_path={model}",
        f"rl.data.path={rows}",
        "rl.validation.enabled=false",
        "rl.processor_kwargs={images_kwargs: {size: {shortest_edge: 16, longest_edge: 64}}}",
    ]

    monkeypatch.setattr(
        sys, "argv", ["sft", "--config", _RECIPE, "--validate-manifest", "--", *overrides, "rl.max_length=256"]
    )
    with pytest.raises(SystemExit) as exc:
        entry.main()
    assert exc.value.code == 0, "the manifest preflight does not look at rows: the premise of this validator"
    assert "bind_check: ok" in capsys.readouterr().out

    monkeypatch.setattr(
        sys, "argv", ["sft", "--config", _RECIPE, "--validate-rows", "--", *overrides, "rl.max_length=256"]
    )
    with pytest.raises(SystemExit) as exc:
        entry.main()
    out = capsys.readouterr().out
    assert exc.value.code == 1, out
    failing = [line for line in out.splitlines() if line.startswith("FAIL")]
    assert (
        len(failing) == 1 and failing[0].startswith("FAIL train row 2: ") and "exceed rl.max_length=256" in failing[0]
    ), out
    assert "train: rows 2 | loss_tokens == assistant_tokens on 2/2 | longest " in out and "> 256 | failed 1" in out, out
    reports = sorted((tmp_path / "outputs").rglob(validator.REPORT_GLOB))
    assert len(reports) == 1 and len(reports[0].read_text().splitlines()) == 3, (builder, reports)

    monkeypatch.setattr(
        sys, "argv", ["sft", "--config", _RECIPE, "--validate-rows", "--", *overrides, "rl.max_length=4096"]
    )
    with pytest.raises(SystemExit) as exc:
        entry.main()
    out = capsys.readouterr().out
    assert exc.value.code == 0, out
    assert "| failed 0" in out and "FAIL" not in out, out
    later = sorted((tmp_path / "outputs").rglob(validator.REPORT_GLOB))
    assert len(later) == 2 and reports[0] in later, "a second invocation mints a fresh report, never overwrites"
    print(f"entry flag exercised through: {builder}")


def test_missing_split_is_that_splits_load_refusal_and_never_blocks_the_other(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Review thread f7b0fdcd21: the shipped recipe, a readable training JSONL and a missing enabled validation JSONL.

    On the parent ``validate_from_config`` raised ``ValueError`` from the manifest-style preflight before any row was
    inspected or any report created, printing the full path. Now the structural checks run first, and each selected
    source's availability is that split's load refusal: ``split="train"`` inspects the rows and writes its report,
    ``split="validation"`` gets a FAIL line, a load-refused gate line and ``load_error`` (path redacted), and
    ``split="all"`` does both. The trainer's own ``validate_config`` still refuses the configuration.
    """
    _packer_module()
    from cosmos_framework.utils.config import load_config
    from cosmos_framework.posttrain.entrypoints import sft as entry

    _builder_for_this_transformers(monkeypatch)
    model = tmp_path / "model"
    model.mkdir()
    _vlm_model_dir(model)
    train = _stub_jsonl(tmp_path / "train.jsonl", [_agentic_row(), _agentic_row()])
    missing = tmp_path / "validation.jsonl"
    assert not missing.exists()
    config = load_config(
        _RECIPE,
        [
            "--",
            f"model_hf.name_or_path={model}",
            f"rl.data.path={train}",
            "rl.validation.enabled=true",
            f"rl.validation.data.path={missing}",
            "rl.processor_kwargs={images_kwargs: {size: {shortest_edge: 16, longest_edge: 64}}}",
            "rl.max_length=4096",
        ],
    )
    out = tmp_path / "train_only.jsonl"
    status = validator.validate_from_config(config, split="train", out=str(out))  # raised ValueError on the parent
    printed = capsys.readouterr().out
    assert status == 0, printed
    assert "FAIL" not in printed and "train: rows 2 | loss_tokens == assistant_tokens on 2/2 | longest " in printed
    assert "| failed 0" in printed and len(out.read_text(encoding="utf-8").splitlines()) == 3

    out = tmp_path / "validation_only.jsonl"
    status = validator.validate_from_config(config, split="validation", out=str(out))
    printed = capsys.readouterr().out
    assert status == 1, printed
    assert (
        "FAIL validation: " in printed and "validation: rows 0 | load refused: " in printed and "| failed 1" in printed
    )
    refusal = [line for line in printed.splitlines() if line.startswith(("FAIL validation: ", "validation: rows 0"))]
    assert len(refusal) == 2 and all(str(tmp_path) not in line for line in refusal), (
        "the refusal is a FAIL line and a gate line with a redacted path (only the report path is printed in full)"
    )
    assert "Traceback" not in printed and f"{tmp_path.name}/validation.jsonl" in refusal[0]
    (line,) = out.read_text(encoding="utf-8").splitlines()
    summary = json.loads(line)["summary"]
    assert summary["rows"] == 0 and summary["failed"] == 1 and summary["ok"] is False
    assert "validation.jsonl" in summary["load_error"] and str(tmp_path) not in line

    out = tmp_path / "all.jsonl"
    status = validator.validate_from_config(config, split="all", out=str(out))
    printed = capsys.readouterr().out
    assert status == 1, printed
    assert "train: rows 2 | loss_tokens == assistant_tokens on 2/2 |" in printed and "| failed 0" in printed
    assert "validation: rows 0 | load refused: " in printed
    lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 4 and lines[2]["summary"]["split"] == "train" and lines[2]["summary"]["ok"] is True
    assert lines[3]["summary"]["split"] == "validation" and lines[3]["summary"]["load_error"] is not None

    with pytest.raises(ValueError, match="is not readable"):
        entry.validate_config(config)  # the trainer's preflight keeps refusing the whole configuration
    assert entry.validate_config_structure(config).validation.enabled is True


def test_student_template_rows_pass_the_validator() -> None:
    """Opt-in: a real checkpoint's tokenizer and template on a harness-shaped row with tools.

    With a Qwen3-VL processor directory the packer renders through the processor's template, as the trainer does
    for a VLM; a text-only tokenizer directory renders through the tokenizer's.
    """
    path = os.environ.get("SFT_VALIDATOR_TOKENIZER")
    if not path:
        pytest.skip("set SFT_VALIDATOR_TOKENIZER=<local HF tokenizer/processor dir> to run the student-template check")
    module = _packer_module()
    from transformers import AutoProcessor, AutoTokenizer

    processor = None
    if Path(path, "preprocessor_config.json").is_file():
        processor = AutoProcessor.from_pretrained(path)
        if type(processor).__name__ != "Qwen3VLProcessor":
            processor = None
    tokenizer = processor.tokenizer if processor is not None else AutoTokenizer.from_pretrained(path, use_fast=True)
    packer = module.QwenSFTDataPacker(tokenizer, processor=processor, max_length=validator.MEASURE_MAX_LENGTH)
    (report,), summary = validator.validate_dataset(packer, [_agentic_row()], max_length=16384)
    template = "processor" if processor is not None else "tokenizer"
    print(
        f"student {template} template: {json.dumps(summary)} body_tokens={report.body_tokens} reasons={report.reasons}"
    )
    assert report.ok, report.reasons
    assert report.assistant_blocks == 3 and report.loss_tokens == report.assistant_tokens
    supervised = _supervised_text(packer, _agentic_row())
    assert "<tool_call>" in supervised and "<think>" in supervised and "<tool_response>" not in supervised


# ----------------------------------------------------------------------------- fix round (review 117)


def _stub_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


class _RawRows:
    """A map-style factory product: raw rows the loader normalizes on read (``FactorySFTDataset.__getitem__``)."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]

    def cursor_fingerprint(self) -> str:
        return "raw-rows-" + str(len(self.rows))


def refusing_factory() -> _RawRows:
    """Row 0 references a media file that does not exist; row 1 is a clean text row."""
    return _RawRows(
        [
            {
                "messages": [
                    {"role": "user", "content": [{"type": "image", "image": "/data/pool/img_0.png"}]},
                    _assistant("a"),
                ]
            },
            {"messages": [{"role": "user", "content": "q"}, _assistant("fine")]},
        ]
    )


def empty_factory() -> _RawRows:
    return _RawRows([])


_FACTORY = f"{__name__}.refusing_factory"
_EMPTY_FACTORY = f"{__name__}.empty_factory"


def test_redaction_keeps_the_last_two_path_components_and_bounds_length() -> None:
    text = "/path/to/shared-storage missing image file /path/to/shared-storage"
    assert validator.redact(text) == "y/rows.jsonl:2: missing image file media/missing_0042.png"
    assert (
        validator.redact("Got /nonexistent/dir/secret_item_0042.png. Failed") == "Got dir/secret_item_0042.png. Failed"
    )
    a = validator.redact("missing image file /path/to/shared-storage")
    b = validator.redact("missing image file /path/to/shared-storage")
    assert a == "missing image file item_0007/image.png" and a != b, "the item directory still discriminates"
    assert validator.redact("~/rel/x.png and ./rel/y.png and ratio 3/4") == "~/rel/x.png and ./rel/y.png and ratio 3/4"
    assert validator.redact("no path here") == "no path here"
    assert len(validator.redact("x" * 1000)) == 400 and validator.redact("x" * 1000).endswith("...")


def test_compensating_leak_and_drop_across_blocks_is_caught(text_packer: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Totals agree (one token leaked into block 1, one dropped from the last block); the per-block check names it."""
    original = text_packer.sft_process_sample
    end_id = text_packer.tokenizer.convert_tokens_to_ids("<|im_end|>")

    def compensating(sample: Any) -> Any:
        out = original(sample)
        leak = out.labels.index(-100)
        out.labels[leak] = out.input_ids[leak]
        drop = max(i for i, label in enumerate(out.labels) if label != -100 and label != end_id)
        out.labels[drop] = -100
        return out

    monkeypatch.setattr(text_packer, "sft_process_sample", compensating)
    (report,), summary = validator.validate_dataset(text_packer, [_agentic_row()], max_length=4096)
    independent = [body + 1 for body in report.body_tokens]
    assert report.loss_tokens == report.assistant_tokens, "the defect is invisible to the totals"
    assert report.loss_tokens_per_block != independent
    assert (
        report.loss_tokens_per_block[0] == independent[0] + 1
        and report.loss_tokens_per_block[-1] == independent[-1] - 1
    )
    assert report.mask_matches is False
    assert report.reasons == [
        f"loss tokens per assistant block {report.loss_tokens_per_block} != independent {independent} "
        f"(totals agree: {report.loss_tokens})"
    ]
    assert summary["matched_rows"] == 0 and summary["failed"] == 1


def test_source_refusing_a_row_is_a_fail_line_not_a_crash(
    text_packer: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A factory source normalizes on read; its refusal of row 1 is that row's reason, and the split keeps its summary."""
    from cosmos_framework.posttrain.config.sft import SFTDatasetConfig
    from cosmos_framework.posttrain.data.datasets.sft import build_sft_dataset

    source = SFTDatasetConfig(factory=_FACTORY)
    dataset = build_sft_dataset(source)
    with pytest.raises(ValueError, match="missing image file"):
        dataset[0]  # the loader's own refusal on read
    reports, summary = validator.validate_dataset(text_packer, dataset, max_length=4096)
    assert reports[0].reasons == ["source refused the row: SFT factory row 0: missing image file pool/img_0.png"]
    assert reports[0].total_tokens is None and reports[1].ok
    assert summary["failed"] == 1 and summary["failed_rows"] == [1] and summary["unmeasured_rows"] == 1

    out = tmp_path / "report.jsonl"
    status = validator.validate_sources(text_packer, [("train", source, None)], max_length=4096, out=str(out))
    printed = capsys.readouterr().out
    assert status == 1
    assert "FAIL train row 1: source refused the row: SFT factory row 0: missing image file pool/img_0.png" in printed
    assert "train: rows 2 | loss_tokens == assistant_tokens on 1/2 | longest" in printed and "| failed 1" in printed
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3 and json.loads(lines[2])["summary"]["failed_rows"] == [1]
    assert "/data/pool" not in out.read_text(encoding="utf-8")


def test_paths_in_packer_refusals_are_redacted(text_packer: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    message = (
        "Incorrect image source. Must be a valid URL starting with `http://` or `https://`, a valid path to an image "
        "file, or a base64 encoded string. Got /nonexistent/dir/secret_item_0042.png. Failed with Incorrect padding"
    )

    def refuse(sample: Any) -> Any:
        raise ValueError(message)

    monkeypatch.setattr(text_packer, "sft_process_sample", refuse)
    (report,), _ = validator.validate_dataset(text_packer, [_agentic_row()], max_length=4096)
    assert report.reasons == ["packer refused the row: " + message.replace("/nonexistent/dir/", "dir/")]
    assert "/nonexistent" not in json.dumps(validator.asdict(report))


def test_source_refused_at_load_writes_the_summary_line(
    text_packer: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from cosmos_framework.posttrain.config.sft import SFTDatasetConfig

    rows = _stub_jsonl(
        tmp_path / "rows.jsonl",
        [
            {"messages": [{"role": "user", "content": "q"}, _assistant("a")]},
            {
                "messages": [
                    {"role": "user", "content": [{"type": "image", "image": "media/missing_0042.png"}]},
                    _assistant("b"),
                ]
            },
        ],
    )
    out = tmp_path / "report.jsonl"
    status = validator.validate_sources(
        text_packer, [("train", SFTDatasetConfig(path=str(rows)), None)], max_length=4096, out=str(out)
    )
    printed = capsys.readouterr().out
    assert status == 1
    reason = f"{tmp_path.name}/rows.jsonl:2: missing image file media/missing_0042.png"
    assert f"FAIL train: {reason}" in printed
    assert f"train: rows 0 | load refused: {reason} | failed 1" in printed, (
        "a load-refused gate line never reads as a pass"
    )
    assert "failed 0" not in printed
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    summary = json.loads(lines[0])["summary"]
    assert summary["rows"] == 0 and summary["ok"] is False and summary["failed"] == 1
    assert summary["load_error"] == reason
    assert str(tmp_path) not in out.read_text(encoding="utf-8")


def test_empty_factory_split_prints_a_reason(
    text_packer: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from cosmos_framework.posttrain.config.sft import SFTDatasetConfig

    out = tmp_path / "report.jsonl"
    status = validator.validate_sources(
        text_packer, [("train", SFTDatasetConfig(factory=_EMPTY_FACTORY), None)], max_length=4096, out=str(out)
    )
    printed = capsys.readouterr().out
    assert status == 1
    assert "FAIL train: " in printed and "returned an empty dataset" in printed
    summary = json.loads(out.read_text(encoding="utf-8").splitlines()[0])["summary"]
    assert summary["rows"] == 0 and "returned an empty dataset" in summary["load_error"] and summary["ok"] is False


def test_validation_source_equal_to_training_source_is_refused(
    text_packer: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from cosmos_framework.posttrain.config.sft import SFTDatasetConfig

    rows = _stub_jsonl(tmp_path / "rows.jsonl", [_agentic_row(), _agentic_row()])
    out = tmp_path / "report.jsonl"
    status = validator.validate_sources(
        text_packer,
        [
            ("train", SFTDatasetConfig(path=str(rows)), None),
            ("validation", SFTDatasetConfig(path=str(rows), shuffle=False), None),
        ],
        max_length=4096,
        out=str(out),
    )
    printed = capsys.readouterr().out
    assert status == 1
    assert "FAIL validation: source fingerprint equals the train source" in printed
    lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 4 and lines[2]["summary"]["split"] == "train" and lines[2]["summary"]["ok"] is True
    assert lines[3]["summary"]["split"] == "validation" and lines[3]["summary"]["load_error"].startswith(
        "source fingerprint equals"
    )


def _held_out_row(question: str, item: str | None) -> dict[str, Any]:
    row: dict[str, Any] = {
        "messages": [{"role": "user", "content": question}, {"role": "assistant", "content": question[::-1]}]
    }
    if item is not None:
        row["meta"] = {"item": item, "pool": "TRAIN"}
    return row


def test_validation_split_is_held_to_the_setups_disjointness_census(
    text_packer: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """FIXTURE: on the parent the validator refused identical whole-source fingerprints only, so the three shapes below
    returned 0 while ``make_setup`` refused them. The validation split now runs the setup's own contract,
    ``data/datasets/sft.py::require_disjoint_datasets``: training rows A, B with a validation row A (one shared row), a
    validation row with B's item and other text (one shared item), and a validation row without ``meta.item`` while the
    training rows carry one, each exit 1 with the setup's message as ``census_error``; a disjoint pair exits 0 and its
    summary carries the census."""
    from cosmos_framework.posttrain.config.sft import SFTDatasetConfig

    train = _stub_jsonl(tmp_path / "train.jsonl", [_held_out_row("alpha", "item/A"), _held_out_row("beta", "item/B")])

    def run(name: str, rows: list[dict[str, Any]]) -> tuple[int, str, dict[str, Any]]:
        validation = _stub_jsonl(tmp_path / f"{name}.jsonl", rows)
        out = tmp_path / f"{name}_report.jsonl"
        status = validator.validate_sources(
            text_packer,
            [
                ("train", SFTDatasetConfig(path=str(train)), None),
                ("validation", SFTDatasetConfig(path=str(validation), shuffle=False), None),
            ],
            max_length=4096,
            out=str(out),
        )
        printed = capsys.readouterr().out
        summaries = [
            json.loads(line)["summary"] for line in out.read_text(encoding="utf-8").splitlines() if "summary" in line
        ]
        return status, printed, summaries[-1]

    status, printed, summary = run("shared_row", [_held_out_row("alpha", "item/A")])
    assert status == 1 and "FAIL validation: SFT validation source is not held out" in printed, printed
    assert "1 shared row" in summary["census_error"] and summary["ok"] is False and summary["rows"] == 1
    assert "held-out census refused" in printed
    status, printed, summary = run("shared_item", [_held_out_row("other words", "item/B")])
    assert status == 1 and "1 shared item" in summary["census_error"], printed
    status, printed, summary = run("no_item", [_held_out_row("gamma", None)])
    assert status == 1 and "not verifiably held out" in summary["census_error"], printed
    assert "carry no meta.item" in printed
    status, printed, summary = run("disjoint", [_held_out_row("gamma", "held/G"), _held_out_row("delta", "held/D")])
    assert status == 0, printed
    assert summary["census_error"] is None and summary["held_out_census"]["shared_rows"] == 0
    assert summary["held_out_census"]["shared_items"] == 0 and summary["held_out_census"]["validation_items"] == 2
    assert "held-out census: shared rows 0, shared items 0" in printed


# ----------------------------------------------------------------------------- the held-out census boundary


class _FactoryBug(RuntimeError):
    """The factory's own code failing on a row: that row's read failure, exactly as the per-row reader files it."""


class _VendorError(Exception):
    """A third-party decoder's own class, neither ``OSError`` nor ``ValueError``: a read failure like every other."""


class _NoImagePart(StopIteration):
    """A vendor's own ``StopIteration`` subclass (a decoder signalling a missing part by it): named by its own name, as
    every other class a factory raises is."""


class _Interrupt(KeyboardInterrupt):
    """Outside ``Exception``: the one kind of failure the census boundary lets propagate, like every boundary of the tree."""


_BUGS: dict[str, type[BaseException]] = {
    "runtime": _FactoryBug,
    "vendor": _VendorError,
    "interrupt": _Interrupt,
    "stop": StopIteration,
    "stop_sub": _NoImagePart,
}


class _BackedRows:
    """A lazy map-style factory product over per-row files: row ``i`` is read and decoded from ``<root>/<i>.json`` on
    ``__getitem__``, and every image part's file is opened first (a factory that reads media geometry, indexing
    ``row["messages"]`` and each part as a mapping, so a malformed record fails by whatever class that code raises); the
    loader's ``FactorySFTDataset.__getitem__`` normalizes what comes back. ``bug_row`` names a row on which the factory's
    own code raises ``_BUGS[bug]`` instead of reading."""

    def __init__(self, root: str, count: int, bug_row: int = -1, bug: str = "runtime") -> None:
        self.root, self.count, self.bug_row, self.bug = Path(root), count, bug_row, bug

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index == self.bug_row:
            raise _BUGS[self.bug](f"the factory's own code failed on row {index}")
        row = json.loads((self.root / f"{index}.json").read_text(encoding="utf-8"))
        for message in row["messages"]:
            for part in message["content"] if isinstance(message.get("content"), list) else []:
                if part.get("type") == "image":
                    with open(part["image"], "rb") as handle:
                        handle.read(1)
        return row

    def cursor_fingerprint(self) -> str:
        return f"backed-rows-{self.root.name}-{self.count}"


def backed_rows_factory(root: str, count: int, bug_row: int = -1, bug: str = "runtime") -> _BackedRows:
    return _BackedRows(root, count, bug_row, bug)


_BACKED_FACTORY = f"{__name__}.backed_rows_factory"
_TRAIN_ROWS = [_held_out_row("alpha", "item/A"), _held_out_row("beta", "item/B")]
_HELD_ROWS = [_held_out_row("gamma", "held/G"), _held_out_row("delta", "held/D")]


class _OverCountingRows(_RawRows):
    """``__len__`` says one row more than the list holds (a factory whose count and records disagree): the census and the
    per-row reader both ask for the last index, and the list itself raises ``IndexError``."""

    def __len__(self) -> int:
        return len(self.rows) + 1


def over_counting_factory() -> _OverCountingRows:
    return _OverCountingRows([_held_out_row("gamma", "held/G"), _held_out_row("delta", "held/D")])


_OVER_COUNTING_FACTORY = f"{__name__}.over_counting_factory"


class _CountedLenRows(_RawRows):
    """``__len__`` counts its calls and raises ``_FactoryBug`` on exactly one of them (a count read from a remote index, a
    length recomputed on every call): the source loads (the loader's and the validator's empty checks pass) and fails later,
    while the census counts it and no row is being read."""

    def __init__(self, rows: list[dict[str, Any]], fail_on_call: int) -> None:
        super().__init__(rows)
        self.fail_on_call, self.len_calls = fail_on_call, 0

    def __len__(self) -> int:
        self.len_calls += 1
        if self.len_calls == self.fail_on_call:
            raise _FactoryBug(f"the factory's __len__ failed on call {self.len_calls}")
        return len(self.rows)

    def cursor_fingerprint(self) -> str:
        return f"counted-len-{len(self.rows)}-{self.fail_on_call}"


def counted_len_factory(rows: list[dict[str, Any]], fail_on_call: int) -> _CountedLenRows:
    return _CountedLenRows(rows, fail_on_call)


_COUNTED_LEN_FACTORY = f"{__name__}.counted_len_factory"


class _ListSource:
    """A JSONL-shaped source for ``held_out_census_entry`` alone: ``rows`` (the in-memory list the census walks directly) and
    ``media_hashes`` (its digest cache) are attribute reads, and ``broken`` names the one that fails like an unreadable
    sidecar file."""

    def __init__(self, rows: list[dict[str, Any]], broken: str | None = None) -> None:
        self.held, self.broken = rows, broken

    def __len__(self) -> int:
        return len(self.held)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.held[index]

    def __getattr__(self, name: str) -> Any:
        if name == self.broken:
            raise PermissionError(13, "Permission denied", f"/data/pool/{name}.json")
        if name == "rows":
            return self.held
        if name == "media_hashes":
            return {}
        raise AttributeError(name)


def _backed_source(root: Path, rows: list[Any], **factory_kwargs: Any) -> Any:
    """Write ``<root>/<i>.json`` per row (a ``str`` is written verbatim: a corrupt record) and configure the factory."""
    from cosmos_framework.posttrain.config.sft import SFTDatasetConfig

    root.mkdir(parents=True)
    for index, row in enumerate(rows):
        (root / f"{index}.json").write_text(row if isinstance(row, str) else json.dumps(row), encoding="utf-8")
    kwargs = {"root": str(root), "count": len(rows), **factory_kwargs}
    return SFTDatasetConfig(factory=_BACKED_FACTORY, factory_kwargs=kwargs, shuffle=False)


def _jsonl_source(path: Path, rows: list[dict[str, Any]], *, shuffle: bool = True) -> Any:
    from cosmos_framework.posttrain.config.sft import SFTDatasetConfig

    return SFTDatasetConfig(path=str(_stub_jsonl(path, rows)), shuffle=shuffle)


def _image_row(path: Path, item: str) -> dict[str, Any]:
    """A row with one image part at ``path`` (written as a real 8x8 PNG) and ``meta.item``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    _image().save(path)
    return {
        "messages": [{"role": "user", "content": [{"type": "image", "image": str(path)}]}, _assistant("seen")],
        "meta": {"item": item},
    }


def _run_pair(
    packer: Any, tmp_path: Path, train: Any, validation: Any, capsys: pytest.CaptureFixture[str]
) -> tuple[int, str, list[dict[str, Any]], Path]:
    """``validate_sources`` over a training and a validation source into ``<tmp_path>/report.jsonl``: the status, the
    printed output, the report's parsed lines and its path."""
    out = tmp_path / "report.jsonl"
    status = validator.validate_sources(
        packer, [("train", train, None), ("validation", validation, None)], max_length=4096, out=str(out), force=True
    )
    printed = capsys.readouterr().out
    lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    return status, printed, lines, out


def test_census_read_failure_is_the_validation_splits_failure_and_its_rows_are_still_inspected(
    text_packer: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """FIXTURE: on the parent the census boundary caught ``ValueError`` alone, so the ``FileNotFoundError`` below
    escaped ``validate_sources`` before the validation rows were inspected: the report held the training rows and
    summary only, no validation FAIL, gate or summary line, and no report path line. A lazy factory whose backing record
    of validation row 2 is missing: the census cannot read that row, which is the validation split's failure
    (``census_error`` names the side, the row, the class and the redacted path; ``failed`` counts it; ``ok`` false; the
    gate line carries it), the split's rows are still inspected (row 1 measures, row 2 is the per-row refusal with the
    same class), both summaries are written, the report path line is printed and the status is 1."""
    validation = _backed_source(tmp_path / "backing", _HELD_ROWS)
    (tmp_path / "backing" / "1.json").unlink()
    train = _jsonl_source(tmp_path / "train.jsonl", _TRAIN_ROWS)
    status, printed, lines, out = _run_pair(text_packer, tmp_path, train, validation, capsys)
    detail = "FileNotFoundError: [Errno 2] No such file or directory: 'backing/1.json'"
    reason = f"held-out census could not read validation source row 2: {detail}"
    assert status == 1 and f"FAIL validation: {reason}" in printed, printed
    assert f"FAIL validation row 2: source refused the row: {detail}" in printed
    assert "validation: rows 2 | loss_tokens == assistant_tokens on 1/2 | longest " in printed
    assert f"| failed 2 | held-out census refused: {reason}" in printed
    assert f"validate_sft_rows: report {out}" in printed and "Traceback" not in printed
    assert [line["summary"]["split"] for line in lines if "summary" in line] == ["train", "validation"]
    assert len(lines) == 6 and lines[2]["summary"]["ok"] is True
    summary = lines[5]["summary"]
    assert (summary["rows"], summary["failed"], summary["failed_rows"], summary["ok"]) == (2, 2, [2], False)
    assert summary["census_error"] == reason and summary["held_out_census"] is None
    assert lines[3]["reasons"] == [] and lines[3]["loss_tokens"] > 0
    assert lines[4]["reasons"] == [f"source refused the row: {detail}"]
    assert str(tmp_path) not in printed.replace(str(out), "") and str(tmp_path) not in out.read_text(encoding="utf-8")


def test_census_media_os_error_names_the_row_and_the_rows_are_still_inspected(
    text_packer: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Greptile thread 7bb34b5b76 (FIXTURE: on the parent this ``IsADirectoryError`` escaped ``validate_sources``). The
    factory opens each image part's file while the census reads validation row 2, and that path is a directory, not a
    file: an ``OSError`` from the media is the validation split's failure, named with the row, the class and the redacted
    path; the rows are still inspected (row 2 is the same refusal per row) and both summaries are written."""
    media = tmp_path / "media" / "img_1"
    media.mkdir(parents=True)
    row = {
        **_held_out_row("delta", "held/D"),
        "messages": _image_row(tmp_path / "media" / "img_0.png", "x")["messages"],
    }
    row["messages"][0]["content"][0]["image"] = str(media)
    validation = _backed_source(tmp_path / "backing", [_held_out_row("gamma", "held/G"), row])
    train = _jsonl_source(tmp_path / "train.jsonl", _TRAIN_ROWS)
    status, printed, lines, out = _run_pair(text_packer, tmp_path, train, validation, capsys)
    detail = "IsADirectoryError: [Errno 21] Is a directory: 'media/img_1'"
    reason = f"held-out census could not read validation source row 2: {detail}"
    assert status == 1 and f"FAIL validation: {reason}" in printed, printed
    assert f"FAIL validation row 2: source refused the row: {detail}" in printed
    assert (
        f"| failed 2 | held-out census refused: {reason}" in printed and f"validate_sft_rows: report {out}" in printed
    )
    assert len(lines) == 6 and lines[2]["summary"]["split"] == "train" and lines[2]["summary"]["ok"] is True
    summary = lines[5]["summary"]
    assert summary["census_error"] == reason and (summary["failed"], summary["failed_rows"], summary["ok"]) == (
        2,
        [2],
        False,
    )


def test_census_corrupt_backing_row_names_the_decoder_error_and_the_row(
    text_packer: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Review thread b9b8f14577 (P2) (FIXTURE: on the parent the decoder's ``JSONDecodeError`` was caught as a bare
    ``ValueError`` and ``census_error`` read ``Expecting property name enclosed in double quotes: line 1 column 2 (char 1)``:
    no class, no side, no row). The backing record of validation row 2 is not JSON: ``census_error`` names the decoder's
    class, the side and the row; the rows are still inspected and row 2 carries the same refusal per row."""
    validation = _backed_source(tmp_path / "backing", [_held_out_row("gamma", "held/G"), "{corrupt"])
    train = _jsonl_source(tmp_path / "train.jsonl", _TRAIN_ROWS)
    status, printed, lines, out = _run_pair(text_packer, tmp_path, train, validation, capsys)
    detail = "JSONDecodeError: Expecting property name enclosed in double quotes: line 1 column 2 (char 1)"
    reason = f"held-out census could not read validation source row 2: {detail}"
    assert status == 1 and f"FAIL validation: {reason}" in printed, printed
    assert f"FAIL validation row 2: source refused the row: {detail}" in printed
    assert (
        f"| failed 2 | held-out census refused: {reason}" in printed and f"validate_sft_rows: report {out}" in printed
    )
    assert len(lines) == 6 and lines[5]["summary"]["census_error"] == reason and lines[5]["summary"]["failed"] == 2


def test_census_passes_through_a_lazy_factory_whose_rows_read(
    text_packer: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The positive control of the census boundary (unchanged by the fix: it passes on the parent too): the same lazy
    factory with both backing records present and items disjoint from the training rows passes the census, the summary
    carries it, nothing fails and the status is 0."""
    validation = _backed_source(tmp_path / "backing", _HELD_ROWS)
    train = _jsonl_source(tmp_path / "train.jsonl", _TRAIN_ROWS)
    status, printed, lines, out = _run_pair(text_packer, tmp_path, train, validation, capsys)
    assert status == 0, printed
    summary = lines[5]["summary"]
    assert summary["census_error"] is None and summary["ok"] is True and summary["failed"] == 0
    assert summary["held_out_census"]["validation_items"] == 2 and summary["held_out_census"]["shared_rows"] == 0
    assert "held-out census: shared rows 0, shared items 0" in printed and "FAIL" not in printed and len(lines) == 6


_READ_ERROR_SHAPES: dict[str, tuple[Any, str]] = {
    "KeyError": ({"meta": {"item": "held/D"}}, "KeyError: 'messages'"),
    "AttributeError": (
        {"messages": [{"role": "user", "content": ["text"]}, _assistant("x")], "meta": {"item": "held/D"}},
        "AttributeError: 'str' object has no attribute 'get'",
    ),
    "IndexError": (None, "IndexError: list index out of range"),
    "VendorError": ("vendor", "_VendorError: the factory's own code failed on row 1"),
    "RuntimeError": ("runtime", "_FactoryBug: the factory's own code failed on row 1"),
    "StopIteration": ("stop", "StopIteration: the factory's own code failed on row 1"),
    "StopIterationSubclass": ("stop_sub", "_NoImagePart: the factory's own code failed on row 1"),
}


@pytest.mark.parametrize("shape", sorted(_READ_ERROR_SHAPES), ids=sorted(_READ_ERROR_SHAPES))
def test_census_read_error_of_any_class_is_the_rows_failure_like_the_per_row_reader(
    text_packer: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str], shape: str
) -> None:
    """FIXTURE: on the parent the boundary caught ``OSError``, ``ValueError``, ``TypeError`` and ``EOFError`` only, so
    every class below escaped ``validate_sources``: the report held the training rows and summary alone, no validation
    FAIL, gate or summary line, no report path line, a traceback. The census reads the SAME row through the SAME
    ``dataset[index]`` as the per-row reader, whose boundary is ``except Exception``, and the verdict on a malformed
    record must not depend on which class the factory's own code happens to raise: a record without ``messages``
    (``KeyError``), a string content part (``AttributeError``), a ``__len__`` that over-counts (``IndexError`` from the
    list), a vendor decoder's own ``Exception`` subclass, a ``RuntimeError`` of the factory's code, a ``StopIteration``
    of it (finding R4-02 of review round 2; FIXTURE: on 544e3c9b38a1 the census walked the factory through
    ``source_overlap``'s generator expression, which turned the ``StopIteration`` into
    ``RuntimeError: generator raised StopIteration`` -- PEP 479 -- while the per-row reader named ``StopIteration``;
    both now read the row through ``_read_row``, which names it for both), and a SUBCLASS of ``StopIteration`` a
    vendor's code raises (finding R5-01 of review round 3; FIXTURE: on b01a38cde21e ``_read_row`` hardcoded the base
    name, so ``_NoImagePart`` was filed as ``StopIteration: ...`` by both readers while ``_refusal`` names every other
    subclass by its own name; ``_read_row`` now names the class that was raised). Each is the validation split's
    ``census_error`` with the class and the message the per-row reader files for that row; the rows are still inspected,
    both summaries and the report path line are written and the status is 1."""
    from cosmos_framework.posttrain.config.sft import SFTDatasetConfig
    from cosmos_framework.posttrain.data.datasets.sft import build_sft_dataset

    record, detail = _READ_ERROR_SHAPES[shape]
    if record is None:
        validation, failing_row = SFTDatasetConfig(factory=_OVER_COUNTING_FACTORY, shuffle=False), 3
    elif isinstance(record, str):
        validation, failing_row = _backed_source(tmp_path / "backing", _HELD_ROWS, bug_row=1, bug=record), 2
    else:
        validation, failing_row = _backed_source(tmp_path / "backing", [_HELD_ROWS[0], record]), 2
    reports, _ = validator.validate_dataset(
        text_packer, build_sft_dataset(validation), max_length=4096, split="validation"
    )
    assert reports[failing_row - 1].reasons == [f"source refused the row: {detail}"], "the per-row reader's verdict"
    train = _jsonl_source(tmp_path / "train.jsonl", _TRAIN_ROWS)
    status, printed, lines, out = _run_pair(text_packer, tmp_path, train, validation, capsys)
    reason = f"held-out census could not read validation source row {failing_row}: {detail}"
    assert status == 1 and f"FAIL validation: {reason}" in printed, printed
    assert f"FAIL validation row {failing_row}: source refused the row: {detail}" in printed
    assert f"| failed 2 | held-out census refused: {reason}" in printed
    assert f"validate_sft_rows: report {out}" in printed and "Traceback" not in printed
    assert [line["summary"]["split"] for line in lines if "summary" in line] == ["train", "validation"]
    summary = lines[-1]["summary"]
    assert (summary["rows"], summary["failed"], summary["failed_rows"]) == (failing_row, 2, [failing_row])
    assert summary["ok"] is False and summary["census_error"] == reason and summary["held_out_census"] is None
    assert len(lines) == 3 + failing_row + 1 and lines[2]["summary"]["ok"] is True


def test_census_boundary_lets_an_interrupt_propagate(
    text_packer: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The boundary is ``except Exception``, like the per-row reader's (a control that holds on the parent too): a
    ``KeyboardInterrupt`` raised while the census reads validation row 2 is not a row's failure and propagates out of
    ``validate_sources``, the report holding the training rows and summary only."""
    validation = _backed_source(tmp_path / "backing", _HELD_ROWS, bug_row=1, bug="interrupt")
    train = _jsonl_source(tmp_path / "train.jsonl", _TRAIN_ROWS)
    out = tmp_path / "report.jsonl"
    with pytest.raises(_Interrupt, match="failed on row 1"):
        validator.validate_sources(
            text_packer, [("train", train, None), ("validation", validation, None)], max_length=4096, out=str(out)
        )
    printed = capsys.readouterr().out
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3 and json.loads(lines[2])["summary"]["split"] == "train"
    assert "FAIL validation" not in printed and "validate_sft_rows: report" not in printed


def test_census_read_refusal_is_redacted_like_the_row_reason(
    text_packer: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Review thread b9b8f14577 (P2), second item (FIXTURE: on the parent ``census_error``, the FAIL line and the gate line
    carried the loader's refusal with its full path ``/data/pool/img_0.png`` while the row reason redacted it). The
    ``refusing_factory`` as the validation source: the loader refuses row 1 while the census reads it (a missing media
    file is a ``ValueError`` of ``normalize_row``), and the census entry is formatted by the same bounded, redacted helper
    as the row reason: ``pool/img_0.png``, never ``/data/pool``."""
    from cosmos_framework.posttrain.config.sft import SFTDatasetConfig

    train = _jsonl_source(tmp_path / "train.jsonl", _TRAIN_ROWS)
    validation = SFTDatasetConfig(factory=_FACTORY, shuffle=False)
    status, printed, lines, out = _run_pair(text_packer, tmp_path, train, validation, capsys)
    detail = "SFT factory row 0: missing image file pool/img_0.png"
    reason = f"held-out census could not read validation source row 1: {detail}"
    assert status == 1 and f"FAIL validation: {reason}" in printed, printed
    assert f"FAIL validation row 1: source refused the row: {detail}" in printed
    assert f"| failed 2 | held-out census refused: {reason}" in printed
    assert "/data/pool" not in printed and "/data/pool" not in out.read_text(encoding="utf-8")
    assert lines[5]["summary"]["census_error"] == reason and lines[5]["summary"]["failed_rows"] == [1]


def test_census_hash_failure_names_the_side_the_row_and_its_item(
    text_packer: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Greptile thread 7bb34b5b76 ("reads or hashes its rows"; FIXTURE: on the parent the ``PermissionError`` escaped
    ``validate_sources``). The census hashes every media file by bytes (``data/datasets/sft.py::sha256_file`` through
    ``media_digest_for``); here the hash raises ``PermissionError`` for the one image file, which exists and was read by
    the factory and the loader (so the row itself is known, item included). The entry names the side the census was
    walking -- the validation source when its row carries the image; the training source when the training factory's
    row does, since the census walks the training rows first and never asked the validation source -- the 1-based row,
    the item, the class and the redacted path; both summaries are written and the status is 1."""
    from cosmos_framework.posttrain.data.datasets import sft as datasets

    def unreadable(path: str) -> str:
        raise PermissionError(13, "Permission denied", path)

    monkeypatch.setattr(datasets, "sha256_file", unreadable)
    detail = "PermissionError: [Errno 13] Permission denied: 'media/img_0.png'"
    image = _image_row(tmp_path / "media" / "img_0.png", "held/G")
    validation = _backed_source(tmp_path / "held", [image, _held_out_row("delta", "held/D")])
    train = _jsonl_source(tmp_path / "train.jsonl", _TRAIN_ROWS)
    status, printed, lines, out = _run_pair(text_packer, tmp_path, train, validation, capsys)
    reason = f"held-out census could not read validation source row 1 (item held/G): {detail}"
    assert status == 1 and f"FAIL validation: {reason}" in printed, printed
    assert "FAIL validation row 1: packer refused the row: Image/video SFT requires a VLM processor" in printed
    assert len(lines) == 6 and lines[5]["summary"]["census_error"] == reason and lines[5]["summary"]["failed"] == 2

    training = _backed_source(tmp_path / "train_backing", [{**image, "meta": {"item": "item/A"}}, _TRAIN_ROWS[1]])
    held = _jsonl_source(tmp_path / "held.jsonl", _HELD_ROWS, shuffle=False)
    status, printed, lines, out = _run_pair(text_packer, tmp_path, training, held, capsys)
    reason = f"held-out census could not read training source row 1 (item item/A): {detail}"
    assert status == 1 and f"FAIL validation: {reason}" in printed, printed
    assert lines[2]["summary"]["split"] == "train" and lines[2]["summary"]["failed_rows"] == [1], "the packer's refusal"
    summary = lines[5]["summary"]
    assert summary["census_error"] == reason and (summary["failed"], summary["failed_rows"], summary["ok"]) == (
        1,
        [],
        False,
    )


def test_census_read_refusal_redacts_and_bounds_the_item_id_in_its_prefix(
    text_packer: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """FIXTURE: on the parent the item id was interpolated raw into the prefix, which ``_refusal`` neither redacted nor
    bounded: with this item ``census_error`` was 754 characters and carried ``/path/to/shared-storage`` into the FAIL
    line, the gate line and the JSONL. A path-shaped 600-character item id on the row whose media the census cannot
    hash: the whole sentence is redacted and bounded -- the item keeps its last two path components and is cut at
    ``_ITEM_LIMIT`` so that the class and the message survive, the sentence stays within ``_REASON_LIMIT``, and no
    absolute path of two or more components is printed or written anywhere but the report path line."""
    from cosmos_framework.posttrain.data.datasets import sft as datasets

    def unreadable(path: str) -> str:
        raise PermissionError(13, "Permission denied", path)

    monkeypatch.setattr(datasets, "sha256_file", unreadable)
    item = "/path/to/shared-storage" + "x" * 600
    validation = _backed_source(tmp_path / "held", [_image_row(tmp_path / "media" / "img_0.png", item), _HELD_ROWS[1]])
    train = _jsonl_source(tmp_path / "train.jsonl", _TRAIN_ROWS)
    status, printed, lines, out = _run_pair(text_packer, tmp_path, train, validation, capsys)
    text = out.read_text(encoding="utf-8")
    assert status == 1 and "/path/to/shared-storage" not in printed and "/path/to/shared-storage" not in text, printed
    assert validator._PATH.search(printed.replace(str(out), "")) is None and validator._PATH.search(text) is None
    census_error = lines[5]["summary"]["census_error"]
    assert len(census_error) <= validator._REASON_LIMIT and "x" * 200 not in printed + text, len(census_error)
    shown = "item_7/" + "x" * (validator._ITEM_LIMIT - len("item_7/...")) + "..."
    detail = "PermissionError: [Errno 13] Permission denied: 'media/img_0.png'"
    reason = f"held-out census could not read validation source row 1 (item {shown}): {detail}"
    assert census_error == reason and f"FAIL validation: {reason}" in printed, printed
    assert f"| failed 2 | held-out census refused: {reason}" in printed


def test_census_error_shape_is_one_place_of_record() -> None:
    """FIXTURE: on the parent the shape was typed in the module docstring, the ``held_out_census_entry`` docstring and
    the README beside the code, and no constant existed; later the shape still had no ``(<call>)`` alternative. The
    read-failure sentence's shape is typed once in the code, ``CENSUS_READ_ERROR``, and once in the README's
    ``census_error`` term, held equal here; the two docstrings name the constant; the code prints its prefix."""
    shape = validator.CENSUS_READ_ERROR
    source = Path(validator.__file__).read_text(encoding="utf-8")
    readme = Path(validator.__file__).with_name("README.md").read_text(encoding="utf-8")
    assert source.count(shape) == 1 and readme.count(shape) == 1, (source.count(shape), readme.count(shape))
    assert f"`{shape}`" in readme and "`CENSUS_READ_ERROR`" in readme
    assert "``CENSUS_READ_ERROR``" in validator.__doc__
    assert "``CENSUS_READ_ERROR``" in validator.held_out_census_entry.__doc__
    assert (
        shape.startswith(f"{validator._CENSUS_READ_PREFIX} <side> source {{row <n>[ (item <id>)] | (<call>)}}: <class>")
        and "<" not in validator._CENSUS_READ_PREFIX
    )


_LEN_CALLS: dict[str, tuple[str, int]] = {
    # case: (the side whose factory fails, the ``__len__`` call that fails, counted over the product's life). Validation: 1 the
    # loader's empty check, 2 the validator's, 3 the census starting the validation walk, 4 the census dict, 5 the per-row
    # reader. Training: 1 and 2 as above, 3 the per-row reader (the train split is inspected before the validation split
    # loads), 4 the census starting the training walk, 5 the census dict.
    "validation-walk": ("validation", 3),
    "validation-count": ("validation", 4),
    "training-walk": ("training", 4),
    "training-count": ("training", 5),
}


@pytest.mark.parametrize("case", list(_LEN_CALLS), ids=list(_LEN_CALLS))
def test_census_len_failure_names_the_side_and_the_call_not_a_row(
    text_packer: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str], case: str
) -> None:
    """FIXTURE: on the parent the side and the row came from the last ``__getitem__`` alone, so these four sentences
    read ``training source row 2 (item item/B)``, ``validation source row 2 (item held/D)``, ``training source`` and
    ``validation source row 2 (item held/D)``: twice the wrong side, twice a row that had read fine. A factory whose
    ``__len__`` passes at load and fails on exactly one of the census's two ``len()`` calls on its side -- the
    ``range(len(...))`` that starts the side's walk, the row count of the census dict after both walks -- while no row
    is being read: the sentence names that side and ``(len())``, no row and no item; the rows of both splits are still
    inspected and pass, ``failed`` counts the census alone, and the status is 1."""
    from cosmos_framework.posttrain.config.sft import SFTDatasetConfig

    side, call = _LEN_CALLS[case]
    rows = _HELD_ROWS if side == "validation" else _TRAIN_ROWS
    failing = SFTDatasetConfig(
        factory=_COUNTED_LEN_FACTORY, factory_kwargs={"rows": rows, "fail_on_call": call}, shuffle=False
    )
    if side == "validation":
        train, validation = _backed_source(tmp_path / "train_backing", _TRAIN_ROWS), failing
    else:
        train, validation = failing, _jsonl_source(tmp_path / "held.jsonl", _HELD_ROWS, shuffle=False)
    status, printed, lines, out = _run_pair(text_packer, tmp_path, train, validation, capsys)
    reason = f"held-out census could not read {side} source (len()): _FactoryBug: the factory's __len__ failed on call {call}"
    assert status == 1 and f"FAIL validation: {reason}" in printed, printed
    assert "FAIL train row" not in printed and "FAIL validation row" not in printed and "Traceback" not in printed
    assert (
        f"| failed 1 | held-out census refused: {reason}" in printed and f"validate_sft_rows: report {out}" in printed
    )
    assert [line["summary"]["split"] for line in lines if "summary" in line] == ["train", "validation"]
    assert len(lines) == 6 and lines[2]["summary"]["ok"] is True and lines[2]["summary"]["failed_rows"] == []
    summary = lines[5]["summary"]
    assert (summary["rows"], summary["failed"], summary["failed_rows"], summary["ok"]) == (2, 1, [], False)
    assert summary["census_error"] == reason and summary["held_out_census"] is None


def test_census_attribute_read_failure_names_the_side_and_the_attribute() -> None:
    """``source_overlap`` asks each side for ``media_hashes`` (its digest cache) before any row and for ``rows`` (a JSONL
    source's in-memory list) when that side's walk starts (FIXTURE: on the parent 544e3c9b38a1 both sentences below read
    ``training source``, no call named, the wrong side for the first). A source whose attribute read fails is named by its
    side and that call -- the validation one for ``media_hashes``, asked right after the training one's; the training one for
    ``rows`` -- never by a row. The same sources with nothing broken pass the census."""
    census, error = validator.held_out_census_entry(
        _ListSource(_TRAIN_ROWS), _ListSource(_HELD_ROWS, broken="media_hashes")
    )
    detail = "PermissionError: [Errno 13] Permission denied: 'pool/media_hashes.json'"
    assert (census, error) == (None, f"held-out census could not read validation source (.media_hashes): {detail}")
    census, error = validator.held_out_census_entry(_ListSource(_TRAIN_ROWS, broken="rows"), _ListSource(_HELD_ROWS))
    detail = "PermissionError: [Errno 13] Permission denied: 'pool/rows.json'"
    assert (census, error) == (None, f"held-out census could not read training source (.rows): {detail}")
    census, error = validator.held_out_census_entry(_ListSource(_TRAIN_ROWS), _ListSource(_HELD_ROWS))
    assert error is None and (census["shared_rows"], census["shared_items"], census["validation_items"]) == (0, 0, 2)


def test_census_failure_before_any_source_call_is_not_a_sources_and_propagates(
    text_packer: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The edge of the last-call rule, a control (FIXTURE: on the parent 544e3c9b38a1 both halves were filed as ``held-out
    census could not read training source: ...``, a source that had not been asked). ``source_overlap`` is stood in for by a
    function that fails before it asks either source for anything -- the setup's own helper cannot, its first statement reads
    both sides' ``media_hashes`` -- and that failure is the census's own, not a source's: it propagates out of
    ``validate_sources`` like a defect of the validator's code, nothing is filed. The same stand-in failing after one call on
    the validation source is filed against that source and that call."""
    from cosmos_framework.posttrain.data.datasets import sft as datasets

    def fails_at_once(training: Any, validation: Any) -> dict[str, Any]:
        raise RuntimeError("the census's own code failed")

    def fails_after_one_call(training: Any, validation: Any) -> dict[str, Any]:
        len(validation)
        raise RuntimeError("the census's own code failed")

    train = _jsonl_source(tmp_path / "train.jsonl", _TRAIN_ROWS)
    validation = _jsonl_source(tmp_path / "held.jsonl", _HELD_ROWS, shuffle=False)
    monkeypatch.setattr(datasets, "source_overlap", fails_at_once)
    with pytest.raises(RuntimeError, match="the census's own code failed"):
        validator.validate_sources(
            text_packer,
            [("train", train, None), ("validation", validation, None)],
            max_length=4096,
            out=str(tmp_path / "first.jsonl"),
        )
    assert "FAIL validation" not in capsys.readouterr().out
    monkeypatch.setattr(datasets, "source_overlap", fails_after_one_call)
    status, printed, lines, _ = _run_pair(text_packer, tmp_path, train, validation, capsys)
    reason = "held-out census could not read validation source (len()): RuntimeError: the census's own code failed"
    assert status == 1 and f"FAIL validation: {reason}" in printed, printed
    assert lines[5]["summary"]["census_error"] == reason


def test_default_report_path_is_fresh_and_an_explicit_out_is_not_overwritten(
    text_packer: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from cosmos_framework.posttrain.config.sft import SFTDatasetConfig

    first = validator.default_report_path(str(tmp_path))
    second = validator.default_report_path(str(tmp_path))  # same second, nothing written in between
    assert Path(first).name.startswith("validate_sft_rows_") and first.endswith(".jsonl")
    assert second != first and Path(first).exists() and Path(second).exists(), "the mint is exclusive (O_EXCL)"
    assert Path(first).stat().st_size == 0 and Path(second).stat().st_size == 0
    Path(first).write_text("taken\n")

    rows = _stub_jsonl(tmp_path / "rows.jsonl", [_agentic_row()])
    source = [("train", SFTDatasetConfig(path=str(rows)), None)]
    assert validator.validate_sources(text_packer, source, max_length=4096, out=first) == 2
    assert "refusing to overwrite" in capsys.readouterr().out and Path(first).read_text() == "taken\n"
    assert validator.validate_sources(text_packer, source, max_length=4096, out=first, force=True) == 0
    assert len(Path(first).read_text().splitlines()) == 2


def test_tool_call_ids_are_matched_when_present() -> None:
    def call(identifier: str) -> dict[str, Any]:
        return {"id": identifier, "function": {"name": "zoom", "arguments": {}}}

    def result(identifier: str) -> dict[str, Any]:
        return {"role": "tool", "tool_call_id": identifier, "content": "r"}

    user = {"role": "user", "content": "q"}
    exact = [
        user,
        {"role": "assistant", "content": "", "tool_calls": [call("a"), call("b")]},
        result("b"),
        result("a"),
        _assistant("done"),
    ]
    assert validator.role_grammar_failures(exact) == []
    unknown = [user, {"role": "assistant", "content": "", "tool_calls": [call("a")]}, result("z"), _assistant("done")]
    assert validator.role_grammar_failures(unknown) == [
        "message 2: tool result for unknown call id 'z'",
        "message 3: tool call id 'a' has no result",
    ]
    duplicate = [
        user,
        {"role": "assistant", "content": "", "tool_calls": [call("a"), call("b")]},
        result("a"),
        result("a"),
        _assistant("done"),
    ]
    assert validator.role_grammar_failures(duplicate) == [
        "message 3: duplicate tool result for call id 'a'",
        "message 4: tool call id 'b' has no result",
    ]
    count_only = [
        user,
        {"role": "assistant", "content": "", "tool_calls": [call("a"), {"function": {"name": "zoom"}}]},
        result("x"),
        result("y"),
        _assistant("done"),
    ]
    assert validator.role_grammar_failures(count_only) == [], "without ids on every call only the count rule applies"


def test_id_rule_needs_string_ids_on_every_call_and_every_result() -> None:
    """Calls with ids and results without ``tool_call_id`` (the Qwen template renders no ids) are a correct row."""

    def call(identifier: str) -> dict[str, Any]:
        return {"id": identifier, "function": {"name": "zoom", "arguments": {}}}

    user = {"role": "user", "content": "q"}
    plain = {"role": "tool", "content": "r"}
    two_calls = {"role": "assistant", "content": "", "tool_calls": [call("c1"), call("c2")]}
    assert validator.role_grammar_failures([user, two_calls, plain, plain, _assistant("done")]) == []
    one_without = {"role": "tool", "tool_call_id": "c1", "content": "r"}
    assert validator.role_grammar_failures([user, two_calls, one_without, plain, _assistant("done")]) == []
    integer_id = {"role": "tool", "tool_call_id": 7, "content": "r"}
    one_call = {"role": "assistant", "content": "", "tool_calls": [call("c1")]}
    assert validator.role_grammar_failures([user, one_call, integer_id, _assistant("done")]) == []
    dangling = [
        user,
        two_calls,
        {"role": "tool", "tool_call_id": "c1", "content": "r"},
        {"role": "tool", "tool_call_id": "zz", "content": "r"},
        _assistant("done"),
    ]
    assert validator.role_grammar_failures(dangling) == [
        "message 3: tool result for unknown call id 'zz'",
        "message 4: tool call id 'c2' has no result",
    ]


def test_tool_calls_not_a_list_is_reported_on_its_own_message() -> None:
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "x", "tool_calls": "zoom(1)"},
        {"role": "tool", "content": "r"},
        _assistant("done"),
    ]
    failures = validator.role_grammar_failures(messages)
    assert failures[0] == "message 1: tool_calls is not a list (str)"
    assert failures[1] == "message 2: tool result without a preceding tool call"


def test_coined_terms_are_defined_where_they_appear() -> None:
    readme = Path(validator.__file__).with_name("README.md").read_text(encoding="utf-8")
    for term in ("**gate line**", "**summary line**", "**straddle**", "**FAIL line**"):
        assert term in validator.__doc__, term
        assert term in readme, term
    shape = "rows <N> | loss_tokens == assistant_tokens on <K>/<N> | longest <L> <= <max_length> | failed <F>"
    assert shape in validator.__doc__ and shape in readme


def test_native_seat_config_builds_the_packer_from_sft_policy_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """FIXTURE: ``entrypoints/sft.py`` once kept ``packer_source`` with a ``globals().get("sft_policy_source")``
    fallback, a second resolution path beside the seat's resolver. The validator builds its packer through the ONE
    resolver the trainer uses, ``entrypoints.sft.sft_policy_source``: on the Qwen3-VL native seat
    ``model_hf.name_or_path`` is empty and the artifact is the native spec's ``model_id``; on the HF seat an empty
    ``model_hf.name_or_path`` is refused by that same function. The entrypoint defines no ``packer_source`` and contains
    no ``globals()``."""
    _packer_module()
    from cosmos_framework.utils.config import load_config
    from cosmos_framework.posttrain.entrypoints import sft as entry

    assert not hasattr(entry, "packer_source") and "globals()" not in Path(entry.__file__).read_text()
    _builder_for_this_transformers(monkeypatch)
    model = tmp_path / "model"
    model.mkdir()
    _vlm_model_dir(model)
    train = _stub_jsonl(tmp_path / "train.jsonl", [_agentic_row()])
    config = load_config(
        _RECIPE,
        [
            "--",
            "model_hf.name_or_path=''",
            f"rl.data.path={train}",
            "rl.validation.enabled=false",
            "rl.processor_kwargs={images_kwargs: {size: {shortest_edge: 16, longest_edge: 64}}}",
            "rl.max_length=4096",
        ],
    )
    assert config.model_hf.name_or_path == ""
    with pytest.raises(ValueError, match="model_hf.name_or_path"):  # the HF seat's refusal, from sft_policy_source
        validator.validate_from_config(config, split="train", out=str(tmp_path / "never.jsonl"))

    seen: list[Any] = []

    def native_seat(cfg: Any) -> tuple[str, bool]:
        seen.append(cfg)
        return str(model), False

    monkeypatch.setattr(entry, "sft_policy_source", native_seat)  # the seat's resolver, as the native spec resolves it
    out = tmp_path / "native.jsonl"
    status = validator.validate_from_config(config, split="train", out=str(out))
    printed = capsys.readouterr().out
    assert status == 0, printed
    assert seen and "train: rows 1 | loss_tokens == assistant_tokens on 1/1" in printed
    assert len(out.read_text(encoding="utf-8").splitlines()) == 2  # one row line and the summary


# ----------------------------------------------------------------------------- the row's supervise / opener fields


_LOOKUP_TOOLS = [{"function": {"name": "lookup"}}]
_LOOKUP_CALL = {"function": {"name": "lookup", "arguments": {"query": "value"}}}
_TOOL_CALL_TEXT = '<tool_call>[{"function": {"name": "lookup", "arguments": {"query": "value"}}}]</tool_call>'


def _three_turn(*, history_reasoning: str | None, final_reasoning: str | None) -> list[dict[str, Any]]:
    """The seat's three-turn row (``test_data_packer_sft.py``): a tool call, its result, the answer ``done``. The test
    tokenizer spends one token per ASCII byte and one per special token, and its template renders a reasoning turn as
    ``<think>\n`` + text + ``\n</think>\n\n`` before the body, so turn 1 costs 91 loss tokens bare (``<tool_call>`` 11 +
    67 + ``</tool_call>`` 12 + ``<|im_end|>``) and 102 with ``Look.``; turn 2 costs 5 bare (``done`` + ``<|im_end|>``) and
    16 with ``Sure.``."""
    first: dict[str, Any] = {"role": "assistant", "content": "", "tool_calls": [_LOOKUP_CALL]}
    last: dict[str, Any] = {"role": "assistant", "content": "done"}
    if history_reasoning is not None:
        first["reasoning_content"] = history_reasoning
    if final_reasoning is not None:
        last["reasoning_content"] = final_reasoning
    return [{"role": "user", "content": "Use a tool."}, first, {"role": "tool", "content": "found"}, last]


def test_last_assistant_row_is_counted_over_its_last_block_only(text_packer: Any) -> None:
    """A prefix row (``supervise: last_assistant``, history reasoning-free): the packer leaves the history block at -100,
    and the independent count covers the last block alone, so the row validates with 16 loss tokens = [16]; the same
    conversation without the field, or with ``all_assistant`` explicit, validates with 107 = [91, 16]. A history turn
    that carries reasoning under ``last_assistant`` is the packer's refusal in the row's reasons (review R151-04). On the
    rebased parent the validator built ``SFTTarget`` without the field, so the row was measured as 107 = [91, 16]
    (``assert 107 == 16``) and the reasoning-history row packed."""
    messages = _three_turn(history_reasoning=None, final_reasoning="Sure.")
    row = {"messages": messages, "tools": _LOOKUP_TOOLS, "supervise": "last_assistant"}
    (report,), summary = validator.validate_dataset(text_packer, [row], max_length=4096)
    assert report.loss_tokens == 16, (report.loss_tokens, report.loss_tokens_per_block, report.reasons)
    assert report.ok, report.reasons
    assert report.loss_tokens_per_block == [16] and report.assistant_tokens == 16 and report.mask_matches is True
    assert (report.supervise, report.opener) == ("last_assistant", "self")
    assert report.assistant_blocks == 2 and report.supervised_blocks == 1
    assert report.body_tokens == [90, 15], "every assistant block of the row is still described"
    assert summary["matched_rows"] == 1 and summary["ok"]
    assert _supervised_text(text_packer, row) == "<think>\nSure.\n</think>\n\ndone<|im_end|>"
    for explicit in ({}, {"supervise": "all_assistant"}):
        (whole,), _ = validator.validate_dataset(
            text_packer, [{"messages": messages, "tools": _LOOKUP_TOOLS, **explicit}], max_length=4096
        )
        assert whole.ok and whole.loss_tokens == whole.assistant_tokens == 107
        assert whole.loss_tokens_per_block == [91, 16] and whole.supervised_blocks == 2
        assert (whole.supervise, whole.opener) == ("all_assistant", "self")
    history_reasoning = {
        "messages": _three_turn(history_reasoning="Look.", final_reasoning="Sure."),
        "tools": _LOOKUP_TOOLS,
        "supervise": "last_assistant",
    }
    (refused,), _ = validator.validate_dataset(text_packer, [history_reasoning], max_length=4096)
    assert refused.total_tokens is None and len(refused.reasons) == 1
    assert refused.reasons[0].startswith(
        "packer refused the row: SFT supervise=last_assistant: history assistant turn 1 carries reasoning"
    ), refused.reasons


def test_prefilled_row_is_counted_one_token_fewer_per_supervised_block(text_packer: Any) -> None:
    """``opener: prefilled`` (row rule R4): the serve prefills the ``<think>`` opener of every supervised turn, the packer
    leaves that token at -100, and the independent count subtracts one per supervised block: 116 = [101, 15] under
    ``all_assistant`` (118 = [102, 16] as a self row), 15 = [15] under ``last_assistant`` (16 as a self row); the
    supervised text begins with the newline after the opener. ``body_tokens`` keeps the rendered bodies' own counts. On
    the rebased parent the field was dropped and the row measured as a self row (``assert 118 == 116``)."""
    messages = _three_turn(history_reasoning="Look.", final_reasoning="Sure.")
    row = {"messages": messages, "tools": _LOOKUP_TOOLS, "opener": "prefilled"}
    (report,), summary = validator.validate_dataset(text_packer, [row], max_length=4096)
    assert report.loss_tokens == 116, (report.loss_tokens, report.loss_tokens_per_block, report.reasons)
    assert report.ok, report.reasons
    assert report.loss_tokens_per_block == [101, 15] and report.assistant_tokens == 116 and report.mask_matches is True
    assert (report.supervise, report.opener, report.supervised_blocks) == ("all_assistant", "prefilled", 2)
    assert report.body_tokens == [101, 15], "the rendered bodies' own counts, the opener included"
    assert summary["matched_rows"] == 1 and summary["ok"]
    assert _supervised_text(text_packer, row) == (
        f"\nLook.\n</think>\n\n{_TOOL_CALL_TEXT}<|im_end|>\nSure.\n</think>\n\ndone<|im_end|>"
    )
    (as_self,), _ = validator.validate_dataset(text_packer, [{**row, "opener": "self"}], max_length=4096)
    assert as_self.ok and as_self.loss_tokens == as_self.assistant_tokens == 118
    assert as_self.loss_tokens_per_block == [102, 16] and as_self.body_tokens == [101, 15]
    assert _supervised_text(text_packer, {**row, "opener": "self"}).startswith(
        "<think>\nLook.\n</think>\n\n<tool_call>"
    )
    prefix = {
        "messages": _three_turn(history_reasoning=None, final_reasoning="Sure."),
        "tools": _LOOKUP_TOOLS,
        "supervise": "last_assistant",
        "opener": "prefilled",
    }
    (last,), _ = validator.validate_dataset(text_packer, [prefix], max_length=4096)
    assert last.ok and last.loss_tokens == last.assistant_tokens == 15 and last.loss_tokens_per_block == [15]
    assert (last.supervise, last.opener, last.assistant_blocks, last.supervised_blocks) == (
        "last_assistant",
        "prefilled",
        2,
        1,
    )
    assert _supervised_text(text_packer, prefix) == "\nSure.\n</think>\n\ndone<|im_end|>"


def test_prefilled_row_with_a_bare_supervised_turn_is_the_packers_refusal(text_packer: Any) -> None:
    """A ``prefilled`` row whose supervised turn renders no ``<think>`` opener is refused by the packer (review R149-01;
    ``_require_one_opener_per_supervised_block``), and the validator reports that refusal, verbatim, as the row's reason
    (the row is unmeasured, the split fails). B1: no reasoning turn at all under ``all_assistant`` (both blocks bare);
    B2: ``last_assistant`` with a bare last turn; B3: reasoning on turn 1 and a bare turn 2 under ``all_assistant``. On
    the rebased parent the field was dropped, so the packer saw self rows and every one of them validated."""
    refusal = (
        "packer refused the row: SFT opener=prefilled: supervised assistant block(s) {bare} (indices among the row's "
        "assistant turns) do not carry exactly one <think> opener to mask (masked per supervised block {supervised}: "
        "{counts}); a prefilled row needs a reasoning block in every supervised assistant turn (review R149-01)"
    )
    rows = [
        {
            "messages": _three_turn(history_reasoning=None, final_reasoning=None),
            "tools": _LOOKUP_TOOLS,
            "opener": "prefilled",
        },
        {
            "messages": _three_turn(history_reasoning=None, final_reasoning=None),
            "tools": _LOOKUP_TOOLS,
            "supervise": "last_assistant",
            "opener": "prefilled",
        },
        {
            "messages": _three_turn(history_reasoning="Look.", final_reasoning=None),
            "tools": _LOOKUP_TOOLS,
            "opener": "prefilled",
        },
    ]
    reports, summary = validator.validate_dataset(text_packer, rows, max_length=4096)
    assert [report.reasons for report in reports] == [
        [refusal.format(bare="[0, 1]", supervised="[0, 1]", counts="[0, 0]")],
        [refusal.format(bare="[1]", supervised="[1]", counts="[0]")],
        [refusal.format(bare="[1]", supervised="[0, 1]", counts="[1, 0]")],
    ]
    assert all(report.total_tokens is None for report in reports)
    assert [(report.supervise, report.opener) for report in reports] == [
        ("all_assistant", "prefilled"),
        ("last_assistant", "prefilled"),
        ("all_assistant", "prefilled"),
    ]
    assert summary["failed"] == 3 and summary["unmeasured_rows"] == 3 and not summary["ok"]
    for row in rows:  # the same conversations as self rows validate
        (as_self,), _ = validator.validate_dataset(text_packer, [{**row, "opener": "self"}], max_length=4096)
        assert as_self.ok, as_self.reasons


def prefix_rows_factory() -> _RawRows:
    """Row 0: ``meta.supervise`` disagrees with the field; row 1: a prefilled row with a bare supervised turn; row 2: a
    prefix row whose ``meta`` agrees (``last_assistant``, ``prefilled``, reasoning on the last turn only)."""
    history_free = _three_turn(history_reasoning=None, final_reasoning="Sure.")
    return _RawRows(
        [
            {
                "messages": history_free,
                "tools": _LOOKUP_TOOLS,
                "supervise": "last_assistant",
                "meta": {"supervise": "all_assistant"},
            },
            {
                "messages": _three_turn(history_reasoning=None, final_reasoning=None),
                "tools": _LOOKUP_TOOLS,
                "opener": "prefilled",
                "meta": {"opener": "prefilled"},
            },
            {
                "messages": history_free,
                "tools": _LOOKUP_TOOLS,
                "supervise": "last_assistant",
                "opener": "prefilled",
                "meta": {"supervise": "last_assistant", "opener": "prefilled"},
            },
        ]
    )


_PREFIX_FACTORY = f"{__name__}.prefix_rows_factory"


def test_meta_disagreement_is_the_loaders_refusal_and_an_agreeing_prefix_row_counts_its_last_block(
    text_packer: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Through the real source loader: a row whose ``meta.supervise`` disagrees with its ``supervise`` field, and a
    prefilled row with a bare supervised turn, are ``normalize_row``'s refusals -- per row for a factory source
    (``source refused the row``), for the whole split for a JSONL source (every JSONL row is normalized at load) -- and a
    prefix row whose ``meta`` agrees validates with its last block alone, one opener fewer (15 = [15]). On the rebased
    parent the loader refused the same two rows and the agreeing row measured 107 (``assert 107 == 15``)."""
    from cosmos_framework.posttrain.config.sft import SFTDatasetConfig
    from cosmos_framework.posttrain.data.datasets.sft import build_sft_dataset

    source = SFTDatasetConfig(factory=_PREFIX_FACTORY)
    reports, summary = validator.validate_dataset(text_packer, build_sft_dataset(source), max_length=4096)
    assert reports[2].loss_tokens == 15, (reports[2].loss_tokens, reports[2].loss_tokens_per_block, reports[2].reasons)
    assert reports[0].reasons == [
        "source refused the row: SFT factory row 0: meta.supervise 'all_assistant' (the exporter's record) disagrees "
        "with the row's supervise field 'last_assistant', which is what the packer reads; write the same value in both"
    ]
    assert reports[1].reasons == [
        "source refused the row: SFT factory row 1: opener=prefilled requires a non-empty reasoning_content on every "
        "supervised assistant turn (all of them under all_assistant, the last under last_assistant); message(s) [1, 3] "
        "carry none (review R149-01)"
    ]
    assert reports[2].ok and reports[2].loss_tokens_per_block == [15] and reports[2].assistant_tokens == 15
    assert (reports[2].supervise, reports[2].opener, reports[2].supervised_blocks) == ("last_assistant", "prefilled", 1)
    assert summary["failed"] == 2 and summary["failed_rows"] == [1, 2] and summary["unmeasured_rows"] == 2

    out = tmp_path / "report.jsonl"
    status = validator.validate_sources(text_packer, [("train", source, None)], max_length=4096, out=str(out))
    printed = capsys.readouterr().out
    assert status == 1
    assert "FAIL train row 1: source refused the row: SFT factory row 0: meta.supervise 'all_assistant'" in printed
    assert "FAIL train row 2: source refused the row: SFT factory row 1: opener=prefilled requires" in printed
    assert "train: rows 3 | loss_tokens == assistant_tokens on 1/3 | longest" in printed and "| failed 2" in printed
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4
    third = json.loads(lines[2])
    assert (third["supervise"], third["opener"], third["loss_tokens"], third["supervised_blocks"]) == (
        "last_assistant",
        "prefilled",
        15,
        1,
    )

    jsonl = _stub_jsonl(tmp_path / "rows.jsonl", [prefix_rows_factory()[0]])
    status = validator.validate_sources(
        text_packer, [("train", SFTDatasetConfig(path=str(jsonl)), None)], max_length=4096, out=str(out), force=True
    )
    printed = capsys.readouterr().out
    assert status == 1
    assert (
        "FAIL train: " in printed and "rows.jsonl:1: meta.supervise 'all_assistant' (the exporter's record)" in printed
    )
    assert "train: rows 0 | load refused: " in printed
    fail_line = next(line for line in printed.splitlines() if line.startswith("FAIL train: "))
    assert str(tmp_path) not in fail_line, "the JSONL path is redacted to its last two components"


def test_unknown_supervise_or_opener_value_is_the_rows_reason_not_a_crash(text_packer: Any) -> None:
    """``SFTTarget`` refuses an unknown value (the loader refuses it first for a JSONL or factory source); a raw row that
    carries one is a FAIL row with the refusal, never an exception out of the validator. On the rebased parent the fields
    were not read, so both rows validated."""
    rows = [
        {
            "messages": _three_turn(history_reasoning=None, final_reasoning=None),
            "tools": _LOOKUP_TOOLS,
            "supervise": "last",
        },
        {
            "messages": _three_turn(history_reasoning=None, final_reasoning=None),
            "tools": _LOOKUP_TOOLS,
            "opener": "Prefilled",
        },
    ]
    reports, summary = validator.validate_dataset(text_packer, rows, max_length=4096)
    assert [report.reasons for report in reports] == [
        ["packer refused the row: SFTTarget.supervise must be one of ['all_assistant', 'last_assistant'], got 'last'"],
        ["packer refused the row: SFTTarget.opener must be one of ['self', 'prefilled'], got 'Prefilled'"],
    ]
    assert [(report.supervise, report.opener, report.total_tokens) for report in reports] == [(None, None, None)] * 2
    assert summary["failed"] == 2 and not summary["ok"]


# ----------------------------------------------------------------------------- the run-level opener mask (the knob)


def _knob_packer(packer: Any) -> Any:
    """The trainer's packer under the run-level ``rl.mask_think_opener=True`` (the recipe's acknowledged variant)."""
    module = _packer_module()
    return module.QwenSFTDataPacker(packer.tokenizer, max_length=validator.MEASURE_MAX_LENGTH, mask_think_opener=True)


def test_run_level_opener_mask_is_counted_like_the_rows_prefilled_rule(text_packer: Any) -> None:
    """The run-level knob and the row's ``opener: prefilled`` are one rule the packer applies once (T6c/T6d): the ``<think>``
    that opens a supervised body is left at -100. Under a knob packer a self row with reasoning on both turns validates
    with 116 = [101, 15] (118 = [102, 16] without the knob) and a ``last_assistant`` row with 15 = [15]: the validator reads
    the packer's knob and takes the opener off every supervised block that opens with it. On 401d0564c97 the validator did
    not read the knob, so the row was a mismatch: 'loss tokens 116 != independent assistant tokens 118; per assistant block
    [101, 15] vs [102, 16]'."""
    knob = _knob_packer(text_packer)
    assert knob.mask_think_opener is True and text_packer.mask_think_opener is False
    both = {"messages": _three_turn(history_reasoning="Look.", final_reasoning="Sure."), "tools": _LOOKUP_TOOLS}
    (report,), summary = validator.validate_dataset(knob, [both], max_length=4096)
    assert report.ok, report.reasons
    assert report.loss_tokens == report.assistant_tokens == 116 and report.loss_tokens_per_block == [101, 15]
    assert (report.supervise, report.opener, report.body_tokens) == ("all_assistant", "self", [101, 15])
    assert summary["ok"] and summary["matched_rows"] == 1
    prefix = {
        "messages": _three_turn(history_reasoning=None, final_reasoning="Sure."),
        "tools": _LOOKUP_TOOLS,
        "supervise": "last_assistant",
    }
    (last,), _ = validator.validate_dataset(knob, [prefix], max_length=4096)
    assert last.ok, last.reasons
    assert last.loss_tokens == last.assistant_tokens == 15 and last.loss_tokens_per_block == [15]
    (plain,), _ = validator.validate_dataset(text_packer, [both], max_length=4096)
    assert plain.ok and plain.loss_tokens == plain.assistant_tokens == 118, (
        "without the knob the opener is the student's"
    )
    assert plain.loss_tokens_per_block == [102, 16]


def test_run_level_opener_mask_leaves_a_bare_block_whole_and_subtracts_a_prefilled_opener_once(
    text_packer: Any,
) -> None:
    """T6d's rule under the knob: a block that opens with anything but ``<think>`` loses nothing (the packer masks nothing
    there and refuses nothing). A bare self row validates with 96 = [91, 5], nothing off; the mixed row (turn 1 reasoning,
    turn 2 bare) with 106 = [101, 5], per block; a ``prefilled`` row under the knob with 116 = [101, 15], the opener
    subtracted once, never twice. On 401d0564c97 the mixed row was a mismatch ('loss tokens 106 != independent assistant
    tokens 107; per assistant block [101, 5] vs [102, 5]'); the bare and the prefilled rows pass there too and pin the
    per-block rule against a count that would take one off every supervised block or stack the knob on ``prefilled``."""
    knob = _knob_packer(text_packer)
    bare = {"messages": _three_turn(history_reasoning=None, final_reasoning=None), "tools": _LOOKUP_TOOLS}
    (nothing_off,), _ = validator.validate_dataset(knob, [bare], max_length=4096)
    assert nothing_off.ok, nothing_off.reasons
    assert nothing_off.loss_tokens == nothing_off.assistant_tokens == 96
    assert nothing_off.loss_tokens_per_block == [91, 5] and nothing_off.body_tokens == [90, 4]
    mixed = {"messages": _three_turn(history_reasoning="Look.", final_reasoning=None), "tools": _LOOKUP_TOOLS}
    (per_block,), _ = validator.validate_dataset(knob, [mixed], max_length=4096)
    assert per_block.ok, per_block.reasons
    assert per_block.loss_tokens == per_block.assistant_tokens == 106 and per_block.loss_tokens_per_block == [101, 5]
    prefilled = {
        "messages": _three_turn(history_reasoning="Look.", final_reasoning="Sure."),
        "tools": _LOOKUP_TOOLS,
        "opener": "prefilled",
    }
    (once,), _ = validator.validate_dataset(knob, [prefilled], max_length=4096)
    assert once.ok, once.reasons
    assert once.loss_tokens == once.assistant_tokens == 116 and once.loss_tokens_per_block == [101, 15]
    assert (once.supervise, once.opener) == ("all_assistant", "prefilled")
    assert _supervised_text(knob, prefilled).startswith("\nLook.\n</think>\n\n<tool_call>")
    (self_row,), _ = validator.validate_dataset(text_packer, [prefilled], max_length=4096)
    assert self_row.ok and self_row.loss_tokens == 116, "the row's prefilled alone: the same labels, the same count"


def test_validate_from_config_builds_the_packer_with_the_runs_opener_mask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``validate_from_config`` builds the packer with ``mask_think_opener=rl.mask_think_opener`` exactly as
    ``entrypoints.sft.build_trainer`` does, so the counts it reports are the counts that train: under
    ``rl.mask_think_opener=true`` (declared with the typed field ``rl.serve_contract=prefills_opener``, which the seat's
    ``SFTConfig`` requires where the config is built) a self row with reasoning on both turns reports 116 = [101, 15]
    and a bare row 96 = [91, 5]; without the knob the same rows report 118 and 96. On 401d0564c97 the validator built
    its packer without the knob and reported 118 under it (the opener counted as the student's where training leaves it
    at -100)."""
    _packer_module()
    from cosmos_framework.utils.config import load_config

    _builder_for_this_transformers(monkeypatch)
    model = tmp_path / "model"
    model.mkdir()
    _vlm_model_dir(model)
    rows = [
        {"messages": _three_turn(history_reasoning="Look.", final_reasoning="Sure."), "tools": _LOOKUP_TOOLS},
        {"messages": _three_turn(history_reasoning=None, final_reasoning=None), "tools": _LOOKUP_TOOLS},
    ]
    train = _stub_jsonl(tmp_path / "train.jsonl", rows)
    overrides = [
        "--",
        f"model_hf.name_or_path={model}",
        f"rl.data.path={train}",
        "rl.validation.enabled=false",
        "rl.processor_kwargs={images_kwargs: {size: {shortest_edge: 16, longest_edge: 64}}}",
        "rl.max_length=4096",
    ]
    with pytest.raises(ValueError, match="rl.mask_think_opener=True is refused"):  # refused where the config is built
        validator.validate_from_config(
            load_config(_RECIPE, [*overrides, "rl.mask_think_opener=true"]),
            split="train",
            out=str(tmp_path / "never.jsonl"),
        )
    out = tmp_path / "knob.jsonl"
    status = validator.validate_from_config(
        load_config(_RECIPE, [*overrides, "rl.mask_think_opener=true", "rl.serve_contract=prefills_opener"]),
        split="train",
        out=str(out),
    )
    printed = capsys.readouterr().out
    assert status == 0, printed
    lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 3 and lines[2]["summary"]["ok"]
    assert [(line["loss_tokens"], line["loss_tokens_per_block"], line["assistant_tokens"]) for line in lines[:2]] == [
        (116, [101, 15], 116),
        (96, [91, 5], 96),
    ], printed
    assert "train: rows 2 | loss_tokens == assistant_tokens on 2/2" in printed
    plain = tmp_path / "plain.jsonl"
    status = validator.validate_from_config(load_config(_RECIPE, overrides), split="train", out=str(plain))
    capsys.readouterr()
    assert status == 0
    lines = [json.loads(line) for line in plain.read_text(encoding="utf-8").splitlines()]
    assert [line["loss_tokens"] for line in lines[:2]] == [118, 96], "without the knob the opener is the student's"
