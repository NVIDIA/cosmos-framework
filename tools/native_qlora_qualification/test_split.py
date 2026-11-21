# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
import copy
import json
import sys

import pytest

from tools.native_qlora_qualification.common import digest_file, digest_json, inventory_groups, recording_key
from tools.native_qlora_qualification.split import plan_partitions, select_sealed_rows, write_partition


def example(record, *, split="train", trial=1, camera="Camera1", media=None):
    identifier = f"20230929_{record}_SN22_T{trial}_{camera}_0_0"
    return {"id": identifier}, {"id": identifier, "split": split, "sha256": media or digest_json(identifier)}


def test_recording_family_joins_trial_camera_and_normal_events():
    a, x = example(1)
    b, y = example(1, trial=2, camera="192.168.0.13")
    assert recording_key(a["id"]) == recording_key(b["id"])
    assert recording_key("20231013_101813_normal_192.168.0.17_4_event_0_3") == "20231013_101813"
    assert inventory_groups([x, y])[a["id"]] == inventory_groups([x, y])[b["id"]]


@pytest.mark.parametrize("value", ["unknown", "20230230_1_SN1_T1_Camera1_0_0", "20230929_1_SN22_T1_Camera1_0"])
def test_unknown_grouping_fails(value):
    with pytest.raises(ValueError):
        recording_key(value)


def test_transitive_media_links_join_recordings():
    rows = [example(1), example(1, trial=2, media="a" * 64), example(2, media="a" * 64)]
    groups = inventory_groups([x[1] for x in rows])
    assert len(set(groups.values())) == 1


def test_partition_keeps_order_removes_related_training_and_labels_exposure():
    train = [example(i) for i in range(1, 8)]
    validation = [
        example(1, split="validation", trial=2),
        example(8, split="validation"),
        example(9, split="validation"),
    ]
    exposed = {validation[1][0]["id"]}
    partition, summary, groups = plan_partitions(
        [x[0] for x in train],
        [x[0] for x in validation],
        [x[1] for x in train + validation],
        exposed,
        development_groups=2,
    )
    assert summary["original_train_validation_shared_groups"] == 1
    assert summary["development_historically_training_exposed"] is True
    assert summary["untouched_independent_clusters"] == 1
    assert summary["untouched_cluster_uncertainty_estimable"] is False
    assert summary["release_qualified"] is False
    assert len(partition["development"]) == 2
    assert train[0][0] in partition["excluded"]
    assert validation[0][0] in partition["excluded"]
    assert partition["untouched_diagnostic"] == [validation[2][0]]
    assert partition["train"] == [x[0] for x in train if x[0] in partition["train"]]
    again = plan_partitions(
        [x[0] for x in train],
        [x[0] for x in validation],
        list(reversed([x[1] for x in train + validation])),
        exposed,
        development_groups=2,
    )
    assert again[0] == partition


def test_unknown_exposure_and_duplicate_inventory_fail():
    row, item = example(1)
    with pytest.raises(ValueError, match="duplicate"):
        inventory_groups([item, item])
    with pytest.raises(ValueError, match="exposure"):
        plan_partitions([row], [], [item], {"unknown"})


def test_sealed_all_rows_refuses_changes_and_final_promotion(tmp_path):
    rows = [example(i)[0] for i in range(1, 4)]
    groups = {row["id"]: digest_json(row["id"]) for row in rows}
    write_partition(tmp_path, "development", rows, groups, groups, provenance={})
    data, manifest = tmp_path / "development.jsonl", tmp_path / "development.manifest.json"
    selected, identity = select_sealed_rows(data, manifest, digest_file(manifest))
    assert [row for _, row in selected] == rows
    assert identity["selected_rows"] == 3
    with pytest.raises(ValueError, match="SHA256"):
        select_sealed_rows(data, manifest, "f" * 64)
    original = json.loads(manifest.read_text())
    damaged = copy.deepcopy(original)
    damaged["rows"].reverse()
    manifest.write_text(json.dumps(damaged))
    with pytest.raises(ValueError, match="ordered"):
        select_sealed_rows(data, manifest, digest_file(manifest))
    original["role"] = "final"
    manifest.write_text(json.dumps(original))
    with pytest.raises(ValueError, match="untouched"):
        select_sealed_rows(data, manifest, digest_file(manifest))


@pytest.mark.parametrize("damage", [None, "raw_hash", "duplicate"])
def test_historical_selection_preserves_order_and_rejects_damage_before_writing(tmp_path, monkeypatch, damage):
    from tools.native_qlora_qualification import split

    train = [example(i) for i in range(1, 5)]
    validation = [example(i, split="validation") for i in range(5, 8)]
    monkeypatch.setattr(
        split,
        "verify_conversion",
        lambda _: ([row for row, _ in train], [row for row, _ in validation], [item for _, item in train + validation]),
    )
    source = tmp_path / "source"
    source.mkdir()
    (source / "manifest.json").write_text("{}")
    # Preserve the original evaluator selection order, which need not be source order.
    chosen = [validation[2][0], validation[0][0]]
    scores = [{"id": row["id"], "raw_row_sha256": digest_json(row)} for row in chosen]
    if damage == "raw_hash":
        scores[0]["raw_row_sha256"] = "0" * 64
    elif damage == "duplicate":
        scores.append(scores[0])
    previous = tmp_path / "scores.jsonl"
    previous.write_text("".join(json.dumps(row) + "\n" for row in scores))
    output = tmp_path / "partition"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "split",
            "--data-root",
            str(source),
            "--regression-scores",
            str(previous),
            "--exposed-validation-prefix",
            "0",
            "--development-groups",
            "1",
            "--output",
            str(output),
        ],
    )
    if damage is not None:
        with pytest.raises(ValueError, match="historical regression"):
            split.main()
        assert not output.exists()
        return
    split.main()
    manifest = output / "regression_selection.manifest.json"
    selected, identity = select_sealed_rows(output / "regression_selection.jsonl", manifest, digest_file(manifest))
    assert [row for _, row in selected] == chosen
    assert identity["split_role"] == "regression"
    assert json.loads(manifest.read_text())["provenance"]["regression_scores_sha256"] == digest_file(previous)
