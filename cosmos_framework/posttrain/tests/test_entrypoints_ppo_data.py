# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""The PPO data role (`entrypoints/ppo.py::build_data`) on a JSONL pool of conversation-list prompts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cosmos_framework.posttrain.tests.helpers.conversation_rows import CONVERSATION_ROWS, write_conversation_jsonl

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def _rl(path: str, prompt_column: str = "messages", response_column: str = "reward_spec"):
    from cosmos_framework.posttrain.config.rl import PPOConfig, PPODatasetConfig

    data = PPODatasetConfig(path=path, prompt_column=prompt_column, response_column=response_column, shuffle=False)
    return PPOConfig(data=data, seed=42, prompts_per_iteration=2)


def test_build_data_serves_jsonl_conversation_rows_as_written(tmp_path: Path) -> None:
    """A JSONL pool whose prompt column is a conversation list (the image or video inside the content
    parts) trains through PPO's data role: `build_data` -> `JsonlRows` -> `project_prompt_items` ->
    `DatasetEnv`. Two properties, both read on the served observations:

    1. the prompt reaches the observation as the list the file wrote (the GRPO loader's shape), with the
       reference beside it — the projection no longer refuses a non-string prompt;
    2. the rows are served AS WRITTEN: Hugging Face ``load_dataset("json")`` infers one struct type over
       every content block and null-fills the keys a block lacks (a text block gains ``image: None``, which
       the Qwen chat template renders as an image placeholder), so the JSONL branch reads the file with
       `json.loads` per line instead.
    """
    from cosmos_framework.posttrain.entrypoints.ppo import build_data

    path = write_conversation_jsonl(tmp_path, CONVERSATION_ROWS)
    env = build_data(_rl(path))

    observations = env.get_samples(len(CONVERSATION_ROWS))

    assert [obs.representations["prompt"] for obs in observations] == [row["messages"] for row in CONVERSATION_ROWS]
    assert [obs.representations["reference"] for obs in observations] == [
        row["reward_spec"] for row in CONVERSATION_ROWS
    ]
    assert all(obs.modality_tags == ("text",) for obs in observations)
    assert len({obs.id for obs in observations}) == len(CONVERSATION_ROWS)
    # As written: every content block carries exactly the keys the file wrote, none null-filled.
    blocks = [
        block
        for obs in observations
        for turn in obs.representations["prompt"]
        if isinstance(turn["content"], list)
        for block in turn["content"]
    ]
    # The fifth row's turn content is a plain string (3,282 of the pool's 18,270 rows): served as that string.
    assert observations[4].representations["prompt"][0]["content"] == CONVERSATION_ROWS[4]["messages"][0]["content"]
    assert isinstance(observations[4].representations["prompt"][0]["content"], str)
    assert [sorted(block) for block in blocks if block["type"] == "text"] == [["text", "type"]] * 5
    assert [sorted(block) for block in blocks if block["type"] == "image"] == [["image", "type"]] * 3
    assert [sorted(block) for block in blocks if block["type"] == "video"] == [
        ["fps", "frames_indices", "total_num_frames", "type", "video"]
    ]
    # The source carries the content fingerprint exact resume needs: snapshot, restore, same next rows.
    state = env.state_dict()
    restored = build_data(_rl(path))
    assert restored.load_state_dict(state)
    assert [obs.representations["prompt"] for obs in restored.get_samples(2)] == [
        obs.representations["prompt"] for obs in env.get_samples(2)
    ]


def test_build_data_serves_string_prompt_rows_with_the_reference_as_written(tmp_path: Path) -> None:
    """The pre-existing PPO JSONL shape -- a flat string prompt column -- beside a nested reference column whose
    key sets differ between rows. Hugging Face ``load_dataset("json")``, the reader this branch replaces, infers
    one struct type over the column and null-fills the missing keys, so the reward route received
    ``{"ground_truth": "4", "answer_type": "numeric", "k": None, "constraint": None}`` for a row that wrote two
    keys. `JsonlRows` serves the reference as written; the prompt strings were never affected."""
    from cosmos_framework.posttrain.entrypoints.ppo import build_data

    rows = [
        {
            "question": "What is 2 + 2?",
            "reward_spec": {"verifier": "mm_exact_match", "payload": {"ground_truth": "4", "answer_type": "numeric"}},
        },
        {
            "question": "Write one sentence in lowercase.",
            "reward_spec": {"verifier": "open_instruct_if", "payload": {"k": 1, "constraint": "lowercase"}},
        },
    ]
    path = write_conversation_jsonl(tmp_path, rows, name="strings.jsonl")
    observations = build_data(_rl(path, prompt_column="question")).get_samples(2)

    assert [obs.representations["prompt"] for obs in observations] == [row["question"] for row in rows]
    assert [obs.representations["reference"] for obs in observations] == [row["reward_spec"] for row in rows]
    assert [sorted(obs.representations["reference"]["payload"]) for obs in observations] == [
        ["answer_type", "ground_truth"],
        ["constraint", "k"],
    ]


def test_build_data_serves_a_pool_mixing_string_and_list_content_turns(tmp_path: Path) -> None:
    """The real pool mixes turns whose ``content`` is a list of parts (14,988 of its 18,270 rows) with turns
    whose ``content`` is a plain string (3,282 text-only instruction-following rows). Arrow cannot type that
    column (``Column(/messages/[]/content) changed from array to string``), so Hugging Face
    ``load_dataset("json")`` refused the whole file at construction; `JsonlRows` serves both rows as written."""
    from cosmos_framework.posttrain.entrypoints.ppo import build_data

    rows = [CONVERSATION_ROWS[0], CONVERSATION_ROWS[4]]
    observations = build_data(_rl(write_conversation_jsonl(tmp_path, rows, name="mixed.jsonl"))).get_samples(2)

    assert [obs.representations["prompt"] for obs in observations] == [row["messages"] for row in rows]
    assert isinstance(observations[0].representations["prompt"][0]["content"], list)
    assert isinstance(observations[1].representations["prompt"][0]["content"], str)
    assert [obs.representations["reference"] for obs in observations] == [row["reward_spec"] for row in rows]


def test_build_data_reads_jsonl_gzip_and_json_array_paths_alike(tmp_path: Path) -> None:
    """`rl.data.path` keeps the shapes the Hugging Face reader accepted: one JSON object per line, the same
    gzip-compressed, or one top-level JSON array. The three files serve the same prompts and references, so a
    configuration that pointed PPO at a ``.json`` array or a ``.jsonl.gz`` before this change still trains."""
    import gzip

    from cosmos_framework.posttrain.entrypoints.ppo import build_data

    rows = [{"question": "1 + 1 =", "answer": "2"}, {"question": "2 + 2 =", "answer": "4"}]
    jsonl = Path(write_conversation_jsonl(tmp_path, rows, name="rows.jsonl"))
    (tmp_path / "rows.jsonl.gz").write_bytes(gzip.compress(jsonl.read_bytes()))
    (tmp_path / "rows.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    served = {}
    for name in ("rows.jsonl", "rows.jsonl.gz", "rows.json"):
        rl = _rl(str(tmp_path / name), prompt_column="question", response_column="answer")
        observations = build_data(rl).get_samples(2)
        served[name] = [(obs.representations["prompt"], obs.representations["reference"]) for obs in observations]
    assert served["rows.jsonl"] == [("1 + 1 =", "2"), ("2 + 2 =", "4")]
    assert served["rows.jsonl.gz"] == served["rows.json"] == served["rows.jsonl"]
