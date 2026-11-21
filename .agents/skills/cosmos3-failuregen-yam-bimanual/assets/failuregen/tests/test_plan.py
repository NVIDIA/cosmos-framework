# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Manifest invariants: every injection has a control, and a donor that does not resolve raises."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from failuregen import plan
from failuregen.donors import Donor, load_labels, write_labels
from failuregen.rollout import render


def _injection(**patch) -> dict:
    run = {
        "name": "pickup__src134t18_c0__ep7",
        "recipe": "pickup_slip",
        "mode": "transplant",
        "collection": "X",
        "host_episode": 7,
        "donor": "134:0",
        "target_phase": "lift_extract",
        "arm": "left",
        "inject_at_s": 6.4,
        "cycle": 0,
        "dims": "all",
        "blend_frames": 4,
        "jaw_amplitude": 1.0,
        "pre_chunks": 2,
        "post_chunks": 2,
        "caption": "c",
    }
    return run | patch


def test_every_distinct_injection_point_gets_one_control():
    runs = [
        _injection(),
        _injection(name="other_donor", donor="112:0"),  # same point, one control suffices
        _injection(name="later", inject_at_s=9.0),
        _injection(name="other_host", host_episode=9),
    ]
    controls = plan._controls(runs, "X")
    assert len(controls) == 3
    assert all(c["mode"] == "control" and c["donor"] is None for c in controls)
    for c, r in zip(controls, [runs[0], runs[2], runs[3]]):
        assert (c["host_episode"], c["target_phase"], c["inject_at_s"], c["arm"]) == (
            r["host_episode"],
            r["target_phase"],
            r["inject_at_s"],
            r["arm"],
        )
        assert c["pre_chunks"] == r["pre_chunks"] and c["post_chunks"] == r["post_chunks"]


def test_a_manifest_naming_an_unknown_donor_fails_to_load(tmp_path):
    """The failure mode this prevents cost 64 runs of a production batch, which rendered from the
    host's own opening seconds and looked like weak failures rather than a bug."""
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"n_runs": 1, "runs": [_injection()]}))
    write_labels([], manifest.with_name("manifest_donors.json"))
    with pytest.raises(ValueError, match="missing donors"):
        plan.load(manifest)


def test_a_manifest_loads_when_its_donors_are_present(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"n_runs": 1, "runs": [_injection()]}))
    donor = Donor(
        collection="X",
        episode=134,
        event_index=0,
        t_start=18.76,
        t_end=20.14,
        phase="pick_grasp",
        arm="left",
        overshoot=0.56,
        outcome="",
        label="drop",
    )
    write_labels([donor], manifest.with_name("manifest_donors.json"))
    runs, donors = plan.load(manifest)
    assert len(runs) == 1 and donors["134:0"]["t_start"] == 18.76


def test_rendering_a_transplant_without_a_donor_raises_before_touching_the_dataset(profile):
    with pytest.raises(ValueError, match="donor did not resolve"):
        render(profile, model=None, run=_injection(), out_dir=Path("/nonexistent"), donor=None)


def test_hand_written_donor_files_load_and_keep_their_extra_fields(tmp_path):
    """Curated windows are how behaviours the annotation vocabulary lacks get in."""
    path = tmp_path / "curated.json"
    path.write_text(
        json.dumps(
            {
                "events": [
                    {
                        "source": "X",
                        "episode_index": 107,
                        "event_index": 72,
                        "t_start": 9.5,
                        "t_end": 18.0,
                        "phase": "any",
                        "arm": "left",
                        "outcome": "groping",
                        "prompt": "The gripper closes on nothing and the arm keeps hunting for the object.",
                        "note": "hand-marked from the wrist view",
                    }
                ]
            }
        )
    )
    donors = load_labels(path)
    assert len(donors) == 1
    d = donors[0]
    assert d.id == "107:72" and d.outcome == "groping" and d.caption.startswith("The gripper")
    assert d.extra["note"] == "hand-marked from the wrist view"


def test_a_donor_file_missing_a_window_is_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"events": [{"source": "X", "episode_index": 1, "arm": "left"}]}))
    with pytest.raises(ValueError, match="t_start"):
        load_labels(path)


def test_recipes_come_from_config_including_unknown_keys(tmp_path):
    recipe = plan.Recipe.from_dict(
        {
            "name": "partial",
            "target_phases": ["lift_extract"],
            "caption": "c",
            "jaw_amplitude": 0.5,
            "exclude_donor_phases": ["handover"],
            "comment": "not a field, carried through to the runs",
        }
    )
    assert recipe.jaw_amplitude == 0.5
    assert recipe.exclude_donor_phases == ("handover",)
    assert recipe.extra["comment"].startswith("not a field")
    assert recipe.dims == "all"  # the mechanism that works is the default
