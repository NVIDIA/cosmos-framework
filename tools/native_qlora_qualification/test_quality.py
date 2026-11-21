# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
import copy

import pytest

from tools.native_qlora_qualification.common import digest_json, inventory_groups
from tools.native_qlora_qualification.quality import analyze, cluster_bootstrap


def fixtures():
    rows, inventory = [], []
    for i in range(6):
        identifier = f"20230929_{i // 2}_SN22_T1_Camera{i % 2 + 1}_0_0"
        inputs = {"input_ids": {"sha256": digest_json(identifier)}}
        rows.append(
            {
                "id": identifier,
                "source_line": i + 1,
                "raw_row_sha256": digest_json(identifier),
                "row_identity": digest_json(identifier),
                "gold": "B",
                "choices": ["A", "B"],
                "media_sha256": digest_json("video" + identifier),
                "input_sha256": digest_json(inputs),
                "inputs": inputs,
                "candidate_token_ids": {"A": 1, "B": 2},
                "answer_token_position": 3,
                "prediction_logit_position": 2,
                "native_supervised_tokens": i % 2 + 1,
                "sequence_length": 5,
                "native_sft_ce": (i + 1) / 10,
                "answer_nll": 0.5,
                "restricted_choice_nll": 0.4,
                "candidate_logits": {"A": 1.0, "B": 2.0},
                "restricted_choice_prediction": "B",
                "restricted_choice_correct": True,
                "strict_first_token_id": 2,
                "strict_first_token_correct": True,
                "valid_first_letter": True,
            }
        )
        inventory.append({"id": identifier, "split": "validation", "sha256": rows[-1]["media_sha256"]})
    arms = {name: copy.deepcopy(rows) for name in ("frozen_nf4", "trained_nf4", "frozen_bf16", "trained_bf16")}
    return arms, inventory


def test_recording_cluster_bootstrap_is_deterministic_paired_and_token_weighted():
    arms, inventory = fixtures()
    for row in arms["trained_nf4"]:
        row["native_sft_ce"] *= 1.05
    result = analyze(arms, inventory, repetitions=100)
    assert result == analyze(arms, list(reversed(inventory)), repetitions=100)
    assert result["trained_paired"]["relative_token_nll_increase"] == pytest.approx(0.05)
    ci = result["recording_cluster_uncertainty"]
    assert ci["clusters"] == 3
    assert ci["relative_nll_interval_95"] == pytest.approx([0.05, 0.05])
    assert result["release_qualified"] is False


@pytest.mark.parametrize("damage", ["mask", "order", "media", "tokens", "nan", "correctness"])
def test_incompatible_or_corrupt_scores_fail(damage):
    arms, inventory = fixtures()
    row = arms["trained_nf4"][0]
    if damage == "mask":
        row["inputs"]["labels"] = "changed"
    elif damage == "order":
        arms["trained_nf4"].reverse()
    elif damage == "media":
        inventory[0]["sha256"] = "f" * 64
    elif damage == "tokens":
        row["native_supervised_tokens"] = 1.0
    elif damage == "nan":
        row["native_sft_ce"] = float("nan")
    else:
        row["restricted_choice_correct"] = False
    with pytest.raises(ValueError):
        analyze(arms, inventory, repetitions=10)


def test_one_cluster_does_not_produce_uncertainty():
    arms, inventory = fixtures()
    groups = {row["id"]: "same" for row in arms["trained_nf4"]}
    result = cluster_bootstrap(arms["trained_nf4"], arms["trained_bf16"], groups)
    assert result["estimable"] is False
    assert "relative_nll_upper_one_sided_95" not in result


def test_cluster_membership_follows_content_links_across_records():
    arms, inventory = fixtures()
    inventory[2]["sha256"] = inventory[0]["sha256"]
    groups = inventory_groups(inventory)
    assert len(set(groups.values())) == 2
