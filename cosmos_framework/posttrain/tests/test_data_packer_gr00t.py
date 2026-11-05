# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""L0 tests for the GR00T sample contract with a fake collator: which processor fields are kept or dropped,
that the derived mask equals the processor's own construction at full prefix, and that the payload survives
the metadata/bulk codec. The real processor is exercised at T1 of the parity gate."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cosmos_framework.posttrain.comm.bulk import BulkStore
from cosmos_framework.posttrain.comm.codec import materialize_payload, pack_payload
from cosmos_framework.posttrain.comm.transfer.movers import LocalStoreMover
from cosmos_framework.posttrain.model.gr00t.packer import (
    GR00T_SAMPLE_KEY,
    action_mask_for,
    collate_transitions,
    samples_from_trajectories,
    transition_from_processed,
)
from cosmos_framework.posttrain.schema import Status, Trajectory

HORIZON, WIDTH = 40, 132
ACTION_DIM, PREFIX = 17, 16


class _FakeCollator:
    """Isaac's collator minus the tokenizer: stacks the non-`vlm_content` fields, returns `{"inputs": batch}`."""

    def __init__(self) -> None:
        self.seen: list[list[dict]] = []

    def __call__(self, features: list[dict]) -> dict:
        self.seen.append(features)
        batch: dict = {}
        for key in set().union(*(f.keys() for f in features)):
            values = [f[key] for f in features]
            if key == "vlm_content":
                batch["text_list"] = [v["text"] for v in values]
            else:
                batch[key] = torch.from_numpy(np.stack(values))
        return {"inputs": batch}


def _processed(index: int = 0) -> dict:
    """A stand-in for one `Gr00tN1d7Processor` output, with the real dtypes and shapes."""
    frames = [torch.full((3, 8, 8), index + 1, dtype=torch.uint8) for _ in range(2)]
    action = torch.zeros(HORIZON, WIDTH, dtype=torch.float32)
    action[:, :ACTION_DIM] = float(index + 1)
    return {
        "state": torch.full((1, WIDTH), float(index), dtype=torch.float32),
        "action": action,
        "action_mask": torch.ones(HORIZON, WIDTH, dtype=torch.float32),
        "embodiment_id": index,
        "vlm_content": {
            "text": f"pick up cube {index}",
            "images": frames,
            # Re-references the SAME tensors as `images`; must not survive into the Transition.
            "conversation": [{"role": "user", "content": [{"image": f} for f in frames]}],
        },
    }


def _trajectory(*samples: dict) -> Trajectory:
    return Trajectory(
        transitions=[transition_from_processed(s, step_idx=i) for i, s in enumerate(samples)],
        id="t0",
        status=Status.COMPLETED,
    )


@pytest.mark.L0
@pytest.mark.CPU
def test_transition_keeps_the_sample_half_and_the_action_and_drops_conversation_and_mask() -> None:
    sample = _processed()
    transition = transition_from_processed(sample, step_idx=3)

    payload = transition.obs.representations[GR00T_SAMPLE_KEY]
    assert sorted(payload) == ["embodiment_id", "state", "vlm_content"]
    assert sorted(payload["vlm_content"]) == ["images", "text"]
    assert torch.equal(payload["state"], sample["state"])
    assert payload["embodiment_id"] == sample["embodiment_id"]
    # Frames stay the processor's uint8 tensors (the eval image transform already ran).
    assert all(f.dtype == torch.uint8 and tuple(f.shape) == (3, 8, 8) for f in payload["vlm_content"]["images"])
    assert torch.equal(transition.action, sample["action"])
    assert transition.step_idx == 3


@pytest.mark.L0
@pytest.mark.CPU
def test_samples_from_trajectories_preserves_producer_order() -> None:
    trajectories = [_trajectory(_processed(0), _processed(1)), _trajectory(_processed(2))]
    assert [int(s["embodiment_id"]) for s in samples_from_trajectories(trajectories)] == [0, 1, 2]


@pytest.mark.L0
@pytest.mark.CPU
def test_full_prefix_mask_equals_the_processor_mask() -> None:
    """At full horizon the derived mask IS the processor's construction (`processing_gr00t_n1d7.py:630-640`)."""
    action = torch.zeros(HORIZON, WIDTH)
    processor_mask = torch.ones_like(action)
    processor_mask[HORIZON:] = 0
    processor_mask[:, ACTION_DIM:] = 0
    assert torch.equal(action_mask_for(action, action_dim=ACTION_DIM, training_prefix=HORIZON), processor_mask)


@pytest.mark.L0
@pytest.mark.CPU
def test_short_prefix_narrows_only_the_horizon() -> None:
    mask = action_mask_for(torch.zeros(HORIZON, WIDTH), action_dim=ACTION_DIM, training_prefix=PREFIX)
    assert mask[:PREFIX, :ACTION_DIM].all()
    assert not mask[PREFIX:].any() and not mask[:, ACTION_DIM:].any()


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    ("action_dim", "prefix", "match"),
    [(ACTION_DIM, HORIZON + 1, "training_prefix"), (WIDTH + 1, PREFIX, "action_dim")],
)
def test_mask_refuses_shapes_the_recipe_and_checkpoint_disagree_on(action_dim, prefix, match) -> None:
    with pytest.raises(ValueError, match=match):
        action_mask_for(torch.zeros(HORIZON, WIDTH), action_dim=action_dim, training_prefix=prefix)


@pytest.mark.L0
@pytest.mark.CPU
def test_collate_hands_the_collator_the_official_per_sample_shape_with_derived_masks() -> None:
    collator = _FakeCollator()
    batch = collate_transitions(
        [_trajectory(_processed(0), _processed(1))], collator, action_dim=ACTION_DIM, training_prefix=PREFIX
    )

    (features,) = collator.seen
    assert sorted(features[0]) == ["action", "action_mask", "embodiment_id", "state", "vlm_content"]
    # The inner dict, not the BatchFeature wrapper: it is what `forward(inputs=...)` receives.
    assert tuple(batch["action"].shape) == (2, HORIZON, WIDTH)
    assert tuple(batch["state"].shape) == (2, 1, WIDTH)
    assert batch["text_list"] == ["pick up cube 0", "pick up cube 1"]
    assert batch["action_mask"].sum() == 2 * PREFIX * ACTION_DIM


@pytest.mark.L0
@pytest.mark.CPU
def test_payload_survives_the_metadata_bulk_codec_and_still_collates() -> None:
    """`threshold=0` is the production setting: every tensor leaf, including each nested frame, is externalized."""
    original = _trajectory(_processed(0), _processed(1))
    packed = pack_payload([original], BulkStore(), threshold=0)
    (restored,) = materialize_payload(packed, LocalStoreMover(cross_process=True))

    for before, after in zip(original.transitions, restored.transitions):
        lhs = before.obs.representations[GR00T_SAMPLE_KEY]
        rhs = after.obs.representations[GR00T_SAMPLE_KEY]
        assert torch.equal(lhs["state"], rhs["state"]) and rhs["state"].dtype == torch.float32
        assert lhs["embodiment_id"] == rhs["embodiment_id"]
        assert rhs["vlm_content"]["text"] == lhs["vlm_content"]["text"]
        for frame_before, frame_after in zip(lhs["vlm_content"]["images"], rhs["vlm_content"]["images"]):
            assert frame_after.dtype == torch.uint8 and torch.equal(frame_before, frame_after)
        assert torch.equal(before.action, after.action) and after.action.dtype == torch.float32

    batch = collate_transitions([restored], _FakeCollator(), action_dim=ACTION_DIM, training_prefix=PREFIX)
    assert tuple(batch["action"].shape) == (2, HORIZON, WIDTH)
