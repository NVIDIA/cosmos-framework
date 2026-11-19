# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Export invariants: labels that mean what they say, and a selection that stays balanced.

Both failure modes here are silent. A label off by one frame still trains; a selection that
quietly drops half the hosts still produces a dataset. Neither raises, and both would only show
up as a model that does not work.
"""

from __future__ import annotations

import numpy as np
import pytest
from failuregen import export
from failuregen.export import TYPE_INDEX, Episode


def _record(
    *,
    mode="transplant",
    recipe="pickup_slip",
    host=7,
    t_inject=6.4,
    failure_time_s=6.4,
    donor_ep=134,
    donor_ev=0,
    arm="left",
) -> dict:
    donor = (
        None
        if mode == "control"
        else {
            "source": "X",
            "episode_index": donor_ep,
            "event_index": donor_ev,
            "t_start": 18.76,
            "phase": "pick_grasp",
            "arm": arm,
        }
    )
    return {
        "name": "run",
        "recipe": recipe if mode != "control" else "control",
        "mode": mode,
        "arm": arm,
        "target_phase": "lift_extract",
        "dims": "all",
        "blend_frames": 4,
        "jaw_amplitude": 1.0,
        "pre_chunks": 2,
        "post_chunks": 2,
        "fps": 5,
        "failure_time_s": failure_time_s,
        "caption": "the reel slips",
        "host": {"source": "X", "episode_index": host, "t_start": 0.0, "t_inject": t_inject},
        "donor": donor,
    }


def _episode(name="run", **kw) -> Episode:
    from pathlib import Path

    record = _record(**kw)
    return Episode(
        name=name,
        run_dir=Path("/nonexistent") / name,
        record=record,
        is_failure=record["mode"] != "control",
        ratio=2.0 if record["mode"] != "control" else None,
    )


def test_the_label_flips_exactly_at_the_injection_frame():
    ep = _episode(failure_time_s=6.4)  # 6.4 s at 5 fps is frame 32
    out = export.labels(ep, n_frames=65, fps=5)
    active = out["annotation.failure.active"]
    assert active[31] == 0 and active[32] == 1
    assert active[:32].sum() == 0 and active[32:].all()


def test_time_to_onset_counts_down_and_goes_negative_after():
    ep = _episode(failure_time_s=6.4)
    tto = export.labels(ep, n_frames=65, fps=5)["annotation.failure.time_to_onset"]
    assert tto[0] == pytest.approx(6.4)
    assert tto[32] == pytest.approx(0.0)
    assert tto[42] == pytest.approx(-2.0)
    assert np.all(np.diff(tto) < 0)


def test_a_control_is_negative_throughout_and_carries_no_horizon():
    """NaN rather than zero: a horizon loss must not read 'no failure' as 'failing right now'."""
    out = export.labels(_episode(mode="control"), n_frames=65, fps=5)
    assert out["annotation.failure.active"].sum() == 0
    assert np.isnan(out["annotation.failure.time_to_onset"]).all()
    assert np.isnan(out["annotation.failure.severity_ratio"]).all()
    assert (out["annotation.failure.type_index"] == TYPE_INDEX["control"]).all()


def test_the_family_reaches_the_type_index():
    out = export.labels(_episode(recipe="handover_loss"), n_frames=10, fps=5)
    assert (out["annotation.failure.type_index"] == TYPE_INDEX["handover_loss"]).all()


def test_provenance_records_the_donor_and_the_control_pairing():
    ep = _episode()
    entry = export.provenance(ep, index=3, checkpoint="iter_000003000")
    assert entry["episode_index"] == 3 and entry["is_failure"]
    assert entry["donor"]["episode_index"] == 134 and entry["host"]["episode_index"] == 7
    assert entry["onset_frame"] == 32 and entry["checkpoint"] == "iter_000003000"
    assert entry["transplant"] == {"dims": "all", "blend_frames": 4, "jaw_amplitude": 1.0}

    control = _episode(mode="control")
    control.control_of = "run"
    got = export.provenance(control, index=4, checkpoint="c")
    assert got["failure_type"] == "control" and got["control_of"] == "run"
    assert got["caption"] is None  # the caption names the failure, so a control has none


def _real_episode(profile, n_frames=120) -> Episode:
    states = np.zeros((n_frames, 14), dtype=np.float32)
    states[:, 6] = 0.3
    states[:, 13] = 0.988
    states[:, 0:6] = np.linspace(0, 1, n_frames)[:, None]
    return Episode(
        name="real_success__X_ep4",
        run_dir=None,
        is_failure=False,
        source="real",
        collection="X",
        episode_index=4,
        states=states,
        verified=True,
        record={
            "name": "real_success__X_ep4",
            "recipe": "success",
            "mode": "real",
            "arm": None,
            "target_phase": None,
            "fps": 5,
            "failure_time_s": None,
            "host": {"source": "X", "episode_index": 4, "t_start": 0.0, "t_inject": None},
            "donor": None,
        },
    )


def test_a_real_episode_is_negative_throughout_and_marked_as_real(profile):
    ep = _real_episode(profile)
    out = export.labels(ep, n_frames=120, fps=5)
    assert out["annotation.failure.active"].sum() == 0
    assert np.isnan(out["annotation.failure.time_to_onset"]).all()
    assert (out["annotation.failure.type_index"] == TYPE_INDEX["control"]).all()
    assert (out["annotation.is_generated"] == 0).all()
    assert ep.family == "success"


def test_a_generated_clip_is_marked_as_generated():
    out = export.labels(_episode(), n_frames=65, fps=5)
    assert (out["annotation.is_generated"] == 1).all()
    control = export.labels(_episode(mode="control"), n_frames=65, fps=5)
    assert (control["annotation.is_generated"] == 1).all()  # a control is rendered, not recorded


def test_real_actions_use_the_same_definition_as_generated_ones(profile):
    """One feature, one meaning. A real episode also records commanded positions, and writing
    those into `action` would make the column mean two things in one dataset."""
    states = _real_episode(profile).states
    actions = export._actions_from(profile, states)
    joints = profile.robot.joint_columns
    np.testing.assert_allclose(actions[:-1, joints], states[1:, joints] - states[:-1, joints], atol=1e-6)
    np.testing.assert_allclose(actions[:, profile.robot.jaw_columns], states[:, profile.robot.jaw_columns])
    assert not actions[-1, joints].any()  # nothing follows the last frame


def test_provenance_distinguishes_the_two_sources(profile):
    """Both are negatives, but recorded footage is not a control and must not be labelled one."""
    real = export.provenance(_real_episode(profile), index=9, checkpoint="iter_x")
    assert real["source"] == "real" and real["checkpoint"] is None and real["source_dir"] is None
    assert real["is_failure"] is False and real["failure_type"] == "success"

    control = export.provenance(_episode(mode="control"), index=1, checkpoint="iter_x")
    assert control["source"] == "generated" and control["failure_type"] == "control"

    generated = export.provenance(_episode(), index=0, checkpoint="iter_x")
    assert generated["source"] == "generated" and generated["checkpoint"] == "iter_x"
    assert generated["failure_type"] == "pickup_slip"


def test_a_short_clip_labels_from_its_own_onset():
    """Injections late in an episode leave only one post chunk, giving 49-frame clips."""
    ep = _episode(failure_time_s=6.4)
    out = export.labels(ep, n_frames=49, fps=5)
    assert len(out["annotation.failure.active"]) == 49
    assert out["annotation.failure.active"][:32].sum() == 0
    assert out["annotation.failure.active"][32:].all()


def test_the_feature_spec_does_not_depend_on_the_source(profile):
    feats = export.features(profile, joint_names=None)
    for key in ("observation.state", "action", "annotation.failure.active", "annotation.is_generated"):
        assert key in feats
    for key, spec in feats.items():
        if spec["dtype"] == "video":
            assert len(spec["shape"]) == 3


def test_camera_shapes_follow_the_canvas_geometry(profile):
    shapes = export._camera_shapes(profile, canvas=(640, 560))
    assert shapes["front"] == (427, 560, 3)
    assert shapes["wrist_left"] == (213, 280, 3) and shapes["wrist_right"] == (213, 280, 3)
    assert sum(s[0] for s in [shapes["front"], shapes["wrist_left"]]) == 640
