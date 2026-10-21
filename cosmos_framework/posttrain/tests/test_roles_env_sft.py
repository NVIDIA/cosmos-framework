# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Offline SFT source identity, coverage, and cursor tests without media decoding."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from cosmos_framework.posttrain.config.sft import SFTDatasetConfig
from cosmos_framework.posttrain.data.datasets.sft import JSONLSFTDataset, normalize_row
from cosmos_framework.posttrain.roles.env.sft import SFTDatasetEnv, build_sft_env

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def _write_rows(path: Path, count: int = 4) -> Path:
    rows = [
        {"messages": [{"role": "user", "content": str(i)}, {"role": "assistant", "content": str(i + 1)}]}
        for i in range(count)
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows))
    return path


def test_rank_shards_and_exact_resume(tmp_path: Path) -> None:
    config = SFTDatasetConfig(path=str(_write_rows(tmp_path / "data.jsonl")))
    first = build_sft_env(config, 7, rank=0, world_size=2)
    second = build_sft_env(config, 7, rank=1, world_size=2)
    a, b = first.get_samples(2), second.get_samples(2)
    assert set(t.id for t in a).isdisjoint(t.id for t in b)
    state = first.state_dict()
    expected = first.get_samples(5)
    restored = build_sft_env(config, 7, rank=0, world_size=2)
    assert restored.load_state_dict(state)
    assert restored.get_samples(5) == expected
    changed = build_sft_env(config, 8, rank=0, world_size=2)
    with pytest.raises(ValueError, match="seed"):
        changed.load_state_dict(state)


def test_validation_tail_counts_each_sample_once_and_preserves_cursor(tmp_path: Path) -> None:
    dataset = JSONLSFTDataset(str(_write_rows(tmp_path / "data.jsonl", 3)))
    physical: list[Any] = []
    for rank in range(2):
        env = SFTDatasetEnv(dataset, rank=rank, world_size=2, training=False)
        state = env.state_dict()
        assert env.validation_batch_count(2) == 1
        physical.extend(env.get_validation_batch(0, 2))
        assert env.state_dict() == state
    logical = [item for item in physical if item.transitions[0].algo_extra["validation_weight"]]
    assert len(logical) == 3
    assert len(set(item.id for item in logical)) == 3
    with pytest.raises(ValueError, match="divisible"):
        SFTDatasetEnv(dataset, rank=0, world_size=2)


def test_jsonl_media_identity_and_relative_paths(tmp_path: Path) -> None:
    image = tmp_path / "example.png"
    image.write_bytes(b"media-content-a")
    row = {
        "messages": [
            {"role": "user", "content": [{"type": "image", "image": image.name}]},
            {"role": "assistant", "content": "red"},
        ]
    }
    path = tmp_path / "data.jsonl"
    path.write_text(json.dumps(row))
    first = JSONLSFTDataset(str(path))
    assert first[0]["messages"][0]["content"][0]["image"] == str(image)
    assert row["messages"][0]["content"][0]["image"] == image.name
    image.write_bytes(b"media-content-b")
    second = JSONLSFTDataset(str(path))
    assert first.cursor_fingerprint() != second.cursor_fingerprint()
    env = SFTDatasetEnv(first)
    state = env.state_dict()
    with pytest.raises(ValueError, match="dataset_fingerprint"):
        SFTDatasetEnv(second).load_state_dict(state)


@pytest.mark.parametrize("row", [{}, {"messages": []}, {"messages": [{"role": "user", "content": "hello"}]}])
def test_reject_missing_supervision(row: dict[str, Any], tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        normalize_row(row, tmp_path, label="row-1")


@pytest.mark.parametrize("null_content", [False, True])
def test_tool_only_assistant_source_normalizes_content(tmp_path: Path, null_content: bool) -> None:
    assistant: dict[str, Any] = {
        "role": "assistant",
        "tool_calls": [{"function": {"name": "lookup", "arguments": {"query": "value"}}}],
    }
    if null_content:
        assistant["content"] = None
    row = {"messages": [{"role": "user", "content": "Use a tool."}, assistant]}
    original = json.dumps(row)
    normalized = normalize_row(row, tmp_path, label="tool-only")
    assert normalized["messages"][1] == {**assistant, "content": ""}
    assert json.dumps(row) == original
    path = tmp_path / "data.jsonl"
    path.write_text(original)
    env = build_sft_env(SFTDatasetConfig(path=str(path)), 7)
    target = env.get_samples(1)[0].transitions[0].targets
    assert target.messages == normalized["messages"]


def test_bad_json_reports_line(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text("\nnot json")
    with pytest.raises(ValueError, match="bad.jsonl:2"):
        JSONLSFTDataset(str(path))


def test_missing_media_reports_sample(tmp_path: Path) -> None:
    row = {
        "messages": [
            {"role": "user", "content": [{"type": "video", "video": "missing.mp4"}]},
            {"role": "assistant", "content": "left"},
        ]
    }
    with pytest.raises(ValueError, match="sample-7: missing video"):
        normalize_row(row, tmp_path, label="sample-7")


def test_supervise_field_is_validated_by_the_loader_and_reaches_the_target(tmp_path: Path) -> None:
    """``supervise`` (fix round T6): a row's value is validated by ``normalize_row`` and never added when absent (the
    fingerprint of an existing file does not change), carried by the JSONL source and set on the ``SFTTarget`` the env
    produces; any other value, ``null`` included, is refused with the row's label. On the parent commit the value passes
    the loader unchecked and never reaches a target (no field)."""
    last = {
        "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}],
        "supervise": "last_assistant",
    }
    plain = {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "b"}]}
    assert normalize_row(last, tmp_path, label="row-1")["supervise"] == "last_assistant"
    assert "supervise" not in normalize_row(plain, tmp_path, label="row-2")
    for bad in ("last", "LAST_ASSISTANT", None, 1, ["last_assistant"]):
        with pytest.raises(ValueError, match="row-3: supervise must be one of"):
            normalize_row({**plain, "supervise": bad}, tmp_path, label="row-3")
    path = tmp_path / "data.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in (last, plain)))
    dataset = JSONLSFTDataset(str(path))
    assert dataset[0]["supervise"] == "last_assistant" and "supervise" not in dataset[1]
    env = SFTDatasetEnv(dataset, shuffle=False)
    targets = [trajectory.transitions[0].targets for trajectory in env.get_samples(2)]
    assert {target.messages[1]["content"]: target.supervise for target in targets} == {
        "a": "last_assistant",
        "b": "all_assistant",
    }
    bad_path = tmp_path / "bad.jsonl"
    bad_path.write_text(json.dumps({**plain, "supervise": "last"}))
    with pytest.raises(ValueError, match="bad.jsonl:1: supervise must be one of"):
        JSONLSFTDataset(str(bad_path))


def test_opener_field_is_validated_by_the_loader_and_reaches_the_target(tmp_path: Path) -> None:
    """``opener`` (fix round T6c, row rule R4): a row's value is validated by ``normalize_row`` and never added when absent,
    carried by the JSONL source and set on the ``SFTTarget`` the env produces; any other value is refused with the row's
    label; the exporter's ``meta.opener`` record must agree with the field the packer reads (a ``prefilled`` in ``meta``
    without the field is refused, a ``self`` in ``meta`` without the field is the default). On the parent commit the value
    passes the loader unchecked and never reaches a target (no field). Fix round T6d: the prefilled row's supervised
    turn carries reasoning (a bare prefilled turn is refused, review R149-01)."""
    # the prefilled row's supervised turn carries reasoning: a prefilled row without one is refused (review R149-01)
    messages = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a", "reasoning_content": "r"}]
    prefilled = {"messages": messages, "opener": "prefilled", "meta": {"opener": "prefilled"}}
    plain = {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "b"}]}
    assert normalize_row(prefilled, tmp_path, label="row-1")["opener"] == "prefilled"
    assert "opener" not in normalize_row(plain, tmp_path, label="row-2")
    for bad in ("prefill", "PREFILLED", None, 1):
        with pytest.raises(ValueError, match="row-3: opener must be one of"):
            normalize_row({**plain, "opener": bad}, tmp_path, label="row-3")
    with pytest.raises(ValueError, match=r"row-3b: opener=prefilled requires .*message\(s\) \[1\] carry none"):
        normalize_row({**plain, "opener": "prefilled"}, tmp_path, label="row-3b")  # a bare turn (review R149-01)
    with pytest.raises(ValueError, match="row-4: meta.opener 'prefilled' \\(the exporter's record\\) disagrees"):
        normalize_row({**plain, "meta": {"opener": "prefilled"}}, tmp_path, label="row-4")
    with pytest.raises(ValueError, match="row-5: meta.opener 'self'"):
        normalize_row({**plain, "opener": "prefilled", "meta": {"opener": "self"}}, tmp_path, label="row-5")
    assert "opener" not in normalize_row({**plain, "meta": {"opener": "self"}}, tmp_path, label="row-6")
    path = tmp_path / "data.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in (prefilled, plain)))
    dataset = JSONLSFTDataset(str(path))
    env = SFTDatasetEnv(dataset, shuffle=False)
    targets = [trajectory.transitions[0].targets for trajectory in env.get_samples(2)]
    assert {target.messages[1]["content"]: (target.opener, target.supervise) for target in targets} == {
        "a": ("prefilled", "all_assistant"),
        "b": ("self", "all_assistant"),
    }


def test_opener_prefilled_rows_require_reasoning_on_every_supervised_turn_and_meta_supervise_must_agree(
    tmp_path: Path,
) -> None:
    """Review R149-01 / R149-05 (fix round T6d): the loader refuses a ``prefilled`` row whose supervised assistant turns
    (all of them under ``all_assistant``, the last under ``last_assistant``) lack a non-empty ``reasoning_content.strip()``
    (B1: no reasoning at all; B2: a bare last turn under ``last_assistant``; B3: mixed; whitespace-only reasoning counts
    as none), names the bare messages with the row's label, and accepts the rows with reasoning where it is required; a
    ``meta.supervise`` that disagrees with the row's ``supervise`` field is refused, an agreeing or defaulted one accepted.
    On cd2f6fddac67 every one of these rows passes the loader (no rule)."""
    call = {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "lookup", "arguments": {"q": 1}}}]}
    reasoned_call = {**call, "reasoning_content": "Look."}
    tool = {"role": "tool", "content": "found"}
    user = {"role": "user", "content": "Use a tool."}
    b1 = {"messages": [user, call, tool, {"role": "assistant", "content": "done"}], "opener": "prefilled"}
    b2 = {**b1, "supervise": "last_assistant"}
    b3 = {"messages": [user, reasoned_call, tool, {"role": "assistant", "content": "done"}], "opener": "prefilled"}
    with pytest.raises(ValueError, match=r"row-1: opener=prefilled requires .*message\(s\) \[1, 3\] carry none"):
        normalize_row(b1, tmp_path, label="row-1")
    with pytest.raises(ValueError, match=r"row-2: opener=prefilled requires .*message\(s\) \[3\] carry none"):
        normalize_row(b2, tmp_path, label="row-2")
    with pytest.raises(ValueError, match=r"row-3: opener=prefilled requires .*message\(s\) \[3\] carry none"):
        normalize_row(b3, tmp_path, label="row-3")
    whitespace = {
        **b3,
        "messages": [user, reasoned_call, tool, {"role": "assistant", "content": "done", "reasoning_content": " \n"}],
    }
    with pytest.raises(ValueError, match=r"row-4: opener=prefilled requires .*message\(s\) \[3\] carry none"):
        normalize_row(whitespace, tmp_path, label="row-4")
    full = {
        **b3,
        "messages": [user, reasoned_call, tool, {"role": "assistant", "content": "done", "reasoning_content": "Sure."}],
    }
    assert normalize_row(full, tmp_path, label="row-5")["opener"] == "prefilled"
    last_only = {
        "messages": [user, call, tool, {"role": "assistant", "content": "done", "reasoning_content": "Sure."}],
        "opener": "prefilled",
        "supervise": "last_assistant",
    }
    assert normalize_row(last_only, tmp_path, label="row-6")["supervise"] == "last_assistant"
    assert "opener" not in normalize_row({"messages": b1["messages"]}, tmp_path, label="row-7"), (
        "self rows need no reasoning"
    )
    with pytest.raises(ValueError, match=r"row-8: meta.supervise 'last_assistant' \(the exporter's record\) disagrees"):
        normalize_row({"messages": b1["messages"], "meta": {"supervise": "last_assistant"}}, tmp_path, label="row-8")
    with pytest.raises(ValueError, match=r"row-9: meta.supervise 'all_assistant'"):
        normalize_row(
            {"messages": b1["messages"], "supervise": "last_assistant", "meta": {"supervise": "all_assistant"}},
            tmp_path,
            label="row-9",
        )
    assert normalize_row(
        {"messages": b1["messages"], "meta": {"supervise": "all_assistant"}}, tmp_path, label="row-10"
    )["meta"] == {"supervise": "all_assistant"}
    assert (
        normalize_row(
            {**last_only, "meta": {"supervise": "last_assistant", "opener": "prefilled"}}, tmp_path, label="row-11"
        )["opener"]
        == "prefilled"
    )
