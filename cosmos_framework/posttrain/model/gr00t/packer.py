# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""GR00T-N1.7 batch assembly: what a `Transition` carries and how Isaac's own collator rebuilds model inputs from it.

Free functions over `Trajectory` (the batch-assembly seam, DESIGN.md v3; `data/packer/preference.py` is the other
instance): the `DataPacker` contract is token-shaped
(rollout prompt ids, `policy_collate_fn(processed, max_len)`) and a VLA does not fit it without distortion.

The per-sample unit is the processor's output `{state, embodiment_id, vlm_content{text, images}}` plus the
normalized `[horizon, width]` action -- not the model's input tensors: `input_ids`/`pixel_values`/... are
collator products that depend on the batch (left-padding), and `Gr00tN1d7DataCollator` raises on them.
`vlm_content["conversation"]` is dropped because it re-references the `images` tensors and `comm/codec.py`
would externalize every frame twice. `action_mask` is derived from the recipe (`training_prefix`, `action_dim`)
rather than stored, so a recipe change cannot disagree with recorded data.
"""

from __future__ import annotations

from typing import Any

import torch

from cosmos_framework.posttrain.schema.trajectory import ObsBundle, Trajectory, Transition

GR00T_SAMPLE_KEY = "gr00t_sample"

_SAMPLE_FIELDS = ("state", "embodiment_id", "vlm_content")


def transition_from_processed(sample: dict, *, step_idx: int = 0) -> Transition:
    """Wrap one processor output (training mode, so `action` is present) as a `Transition`."""
    vlm_content = sample["vlm_content"]
    payload = {
        "state": sample["state"],
        "embodiment_id": sample["embodiment_id"],
        "vlm_content": {"text": vlm_content["text"], "images": list(vlm_content["images"])},
    }
    return Transition(
        obs=ObsBundle(representations={GR00T_SAMPLE_KEY: payload}, modality_tags=("video", "state", "language")),
        action=sample["action"],
        step_idx=step_idx,
    )


def samples_from_trajectories(trajectories: list[Trajectory]) -> list[dict]:
    """Flatten to the per-sample dicts the collator consumes, in producer order."""
    return [
        {**{key: t.obs.representations[GR00T_SAMPLE_KEY][key] for key in _SAMPLE_FIELDS}, "action": t.action}
        for trajectory in trajectories
        for t in trajectory.transitions
    ]


def action_mask_for(action: Any, *, action_dim: int, training_prefix: int) -> torch.Tensor:
    """Ones over `[:training_prefix, :action_dim]`, zeros over the padding: the processor's own mask
    narrowed to the supervised prefix. Slicing past the tensor would silently shorten the mask, so a
    recipe/checkpoint disagreement about the action shape is raised here."""
    horizon, width = action.shape[-2], action.shape[-1]
    if not 1 <= training_prefix <= horizon:
        raise ValueError(f"training_prefix={training_prefix} does not fit the action horizon {horizon}")
    if not 1 <= action_dim <= width:
        raise ValueError(f"action_dim={action_dim} does not fit the action width {width}")
    mask = torch.zeros_like(action)
    mask[..., :training_prefix, :action_dim] = 1.0
    return mask


def collate_transitions(
    trajectories: list[Trajectory], collator: Any, *, action_dim: int, training_prefix: int
) -> dict:
    """Rebuild one model input batch with Isaac's own collator (tokenization, patchifying, left-padding).

    Returns the inner `inputs` dict: `Gr00tN1d7DataCollator` returns `BatchFeature(data={"inputs": batch})`
    and HF's trainer calls `model(**collated)`, so this is the value the official forward receives.
    """
    features = [
        {
            **sample,
            "action_mask": action_mask_for(sample["action"], action_dim=action_dim, training_prefix=training_prefix),
        }
        for sample in samples_from_trajectories(trajectories)
    ]
    return collator(features)["inputs"]
