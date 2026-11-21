# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Observation-only inputs for the selected native empty-reasoning Alpamayo policy.

This preparation does not need future trajectories, future orientations, labels or
supervision masks. It shares the loaded policy's immutable processor and preserves
the selected camera/history/prompt contract; it does not run a language rollout.
"""

from __future__ import annotations


def prepare_observation(policy, sample: dict) -> dict:
    """Prepare one four-camera observation for ``sample_trajectories(policy, inputs)``.

    ``sample`` contains ``media`` and ``observation`` with camera calibration,
    timestamps and 16 vehicle-at-t0 history poses. Model loading, evaluation mode,
    sampling seed and task-output handling remain the caller's responsibility.
    """
    import numpy as np
    import torch
    from alpamayo_r1.helper import create_message

    from cosmos_framework.posttrain.data.flow_media import open_flow_image
    from cosmos_framework.posttrain.model.alpamayo.sft import PHYSICALAI_PROFILE, physicalai_frames

    if policy.packer.config.observation_profile != PHYSICALAI_PROFILE:
        raise ValueError("observation preparation requires the native PhysicalAI four-camera Alpamayo profile")
    if (
        not isinstance(sample, dict)
        or not isinstance(sample.get("media"), dict)
        or not isinstance(sample.get("observation"), dict)
    ):
        raise ValueError("Alpamayo observation requires media and observation mappings")
    # Resolve/validate camera order, temporal bounds, vehicle-frame units and
    # calibration using exactly the training input contract, before image IO.
    frames = physicalai_frames(sample)
    observation = sample["observation"]
    if any(name not in observation for name in ("ego_history_xyz", "ego_history_rot")):
        raise ValueError("Alpamayo observation requires 16 history positions and rotations")
    history = {
        name: torch.tensor(observation[name], dtype=torch.float32).unsqueeze(0)
        for name in ("ego_history_xyz", "ego_history_rot")
    }
    if history["ego_history_xyz"].shape != (1, 16, 3) or history["ego_history_rot"].shape != (1, 16, 3, 3):
        raise ValueError("Alpamayo observation requires 16 history vehicle-frame poses at dt=0.1")
    if not all(torch.isfinite(value).all() for value in history.values()):
        raise ValueError("Alpamayo observation history must be finite")
    if not torch.allclose(
        history["ego_history_xyz"][:, -1], torch.zeros(1, 3), atol=1e-4, rtol=0
    ) or not torch.allclose(history["ego_history_rot"][:, -1], torch.eye(3).unsqueeze(0), atol=1e-4, rtol=0):
        raise ValueError("Alpamayo history must end at the vehicle-at-t0 origin and orientation")
    images = []
    for path in frames:
        with open_flow_image(path) as image:
            images.append(torch.from_numpy(np.asarray(image.convert("RGB")).copy()).permute(2, 0, 1))
    messages = create_message(torch.stack(images))
    messages[-1]["content"][0]["text"] = "<|cot_start|><|cot_end|><|traj_future_start|>"
    tokenized = policy.packer.processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    return {"tokenized": dict(tokenized), "history": {name: value.unsqueeze(1) for name, value in history.items()}}
