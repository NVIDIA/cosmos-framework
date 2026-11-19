# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Donor eligibility and physical invariants for the workflow entry point."""

import json

import numpy as np
import pytest
from failuregen import workflow
from failuregen.yam_training.contract import EEF_CONTRACT, file_sha256
from failuregen.yam_training.failure_test import transplant
from failuregen.yam_training.prepare_eef import poses


def source():
    manifest = {"episodes": [dict(episode=2, split="train", valid_starts=list(range(100)))]}
    annotation = dict(
        reviewed=True,
        phase_reviewed=True,
        segments=[dict(label="pickup_slip", start=2.0, end=2.4)],
        phase_segments=[dict(phase="lift_extract", start=0.8, end=3.0)],
    )
    return manifest, {"episode_000002": annotation}


def test_only_reviewed_training_donors_are_eligible():
    manifest, annotations = source()
    manifest["episodes"].append(dict(episode=99, split="test", valid_starts=[]))
    families, skipped = workflow.build_recipes(manifest, annotations)
    assert len(families) == 1 and families[0]["donor_episode"] == 2
    assert not skipped
    annotations["episode_000002"]["phase_reviewed"] = False
    with pytest.raises(ValueError, match="Review failure and phase"):
        workflow.build_recipes(manifest, annotations)


@pytest.mark.parametrize("event_start", [1.0, 1.19])
def test_insufficient_causal_lead_in_is_excluded(event_start):
    manifest, annotations = source()
    annotations["episode_000002"]["segments"][0]["start"] = event_start
    families, skipped = workflow.build_recipes(manifest, annotations)
    assert not families and len(skipped) == 1


def test_unknown_is_not_a_failure_and_incomplete_windows_are_excluded():
    manifest, annotations = source()
    ann = annotations["episode_000002"]
    ann["segments"].append(dict(label="unknown", start=4.0, end=5.0))
    manifest["episodes"][0]["valid_starts"] = []
    families, skipped = workflow.build_recipes(manifest, annotations)
    assert not families and len(skipped) == 1


def test_repeated_events_have_distinct_recipe_ids():
    manifest, annotations = source()
    annotations["episode_000002"]["segments"].append(dict(label="pickup_slip", start=2.8, end=3.0))
    families, _ = workflow.build_recipes(manifest, annotations)
    assert len({f["id"] for f in families}) == 2


def test_eef_transfer_preserves_prefix_other_arm_and_valid_rotations():
    host = np.zeros((16, 20), dtype=np.float32)
    for offset in (0, 10):
        host[:, offset + 3] = 1
        host[:, offset + 7] = 1
        host[:, offset + 9] = 0.25
    donor = host.copy()
    donor[:, 0] = 0.1
    donor[:, 3:9] = [0, 1, 0, -1, 0, 0]  # 90-degree rotation
    donor[:, 9] = 0.8
    result = transplant(host, donor, arms=("left",))
    np.testing.assert_array_equal(result[:4], host[:4])
    np.testing.assert_array_equal(result[:, 10:], host[:, 10:])
    np.testing.assert_allclose(result[5:, :10], donor[:11, :10])
    rotation = poses(result)[:, 0, :3, :3]
    np.testing.assert_allclose(np.linalg.det(rotation), 1, atol=1e-6)
    assert result[4, 9] == pytest.approx(0.525)  # absolute gripper blend


def test_plan_reuses_only_identical_inputs(tmp_path, monkeypatch):
    prepared, dataset, output = (tmp_path / name for name in ("prepared", "dataset", "selection"))
    prepared.mkdir()
    dataset.mkdir()
    manifest, annotations = source()
    annotation_path = dataset / "failuregen_annotations.json"
    annotation_path.write_text(json.dumps(dict(episodes=annotations)))
    states = prepared / "episode.npz"
    np.savez(states, observations_eef_20d=np.zeros((2, 20)))
    manifest["episodes"][0].update(states=states.name, states_sha256=file_sha256(states))
    manifest.update(
        contract=EEF_CONTRACT,
        annotations=annotation_path.name,
        annotations_sha256=file_sha256(annotation_path),
        stats_sha256="test-stats",
    )
    (prepared / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(workflow, "load_prepared", lambda _: (manifest, {}))
    monkeypatch.setattr(workflow, "select_windows", lambda *a, **k: ([dict(id="window")], [dict(id="donor")]))
    result = workflow.plan(prepared, dataset, output)
    assert result["total_clips"] == 4
    assert workflow.plan(prepared, dataset, output) == result
    with pytest.raises(FileExistsError, match="Selection changed"):
        workflow.plan(prepared, dataset, output, seeds=(0, 1))
    original = (output / "selection.json").read_text()
    annotation_path.write_text("{}")
    with pytest.raises(ValueError, match="Annotations differ"):
        workflow.plan(prepared, dataset, output)
    assert (output / "selection.json").read_text() == original
