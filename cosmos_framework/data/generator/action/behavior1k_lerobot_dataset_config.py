# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1


"""Native local-bundle Bimanual semantics; no remote index access."""

import json
from pathlib import Path

import torch

BEHAVIOR1K_BENCHMARK_SOURCE_DATASET: str = "behavior_1k_demos_subtask__behavior-1k-2026-challenge-demos"


def load_behavior1k_holdout_episode_indices(path: str | Path) -> frozenset[int]:
    """Load and validate the source-episode identities held out for the B1K benchmark."""
    payload: object = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise TypeError("BEHAVIOR-1K holdout manifest must be a JSON object.")
    if payload.get("schema_version") != 1 or payload.get("holdout_scope") != "source_episode":
        raise ValueError("BEHAVIOR-1K holdout manifest must use source-episode schema version 1.")
    if payload.get("canonical_holdout_key_fields") != ["source_dataset", "episode_index"]:
        raise ValueError("BEHAVIOR-1K holdout manifest must identify source episodes.")
    if payload.get("source_dataset") != BEHAVIOR1K_BENCHMARK_SOURCE_DATASET:
        raise ValueError(
            f"BEHAVIOR-1K holdout source_dataset={payload.get('source_dataset')!r} does not match "
            f"{BEHAVIOR1K_BENCHMARK_SOURCE_DATASET!r}."
        )

    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise TypeError("BEHAVIOR-1K holdout manifest episodes must be a non-empty array.")
    episode_indices: set[int] = set()
    for position, episode in enumerate(episodes):
        if not isinstance(episode, dict):
            raise TypeError(f"BEHAVIOR-1K holdout episodes[{position}] must be a JSON object.")
        episode_index = episode.get("episode_index")
        if isinstance(episode_index, bool) or not isinstance(episode_index, int) or episode_index < 0:
            raise TypeError(f"BEHAVIOR-1K holdout episodes[{position}].episode_index must be non-negative.")
        if episode_index in episode_indices:
            raise ValueError(f"Duplicate BEHAVIOR-1K held-out episode_index: {episode_index}.")
        episode_indices.add(episode_index)

    if payload.get("held_out_episode_count") != len(episode_indices):
        raise ValueError("BEHAVIOR-1K held_out_episode_count does not match its episode identities.")
    if payload.get("benchmark_sample_count") != len(episodes):
        raise ValueError("BEHAVIOR-1K benchmark_sample_count does not match its episode records.")
    return frozenset(episode_indices)


STATE_DIM = 61


GRIPPER_QMAX = 0.05


def state61_to_action23(state: torch.Tensor, base_vel_state_scale: float = 0.75) -> torch.Tensor:
    """Map the 61-dim R1Pro ``observation.state`` into the 23-dim action layout.

    Used by ``use_state`` to build the initial-state action frame. Verified
    against OmniGibson ``PROPRIOCEPTION_INDICES["R1Pro"]`` + the ``r1pro_behavior``
    controllers and reconstruction on real data (position dims corr ~1.000):

        action[0:3]   base (velocity)  <- state[0:3]    base_qvel   (direct)
        action[3:7]   trunk (position) <- state[53:57]  trunk_qpos  (direct)
        action[7:14]  left arm         <- state[3:10]   arm_left_qpos
        action[14]    left gripper     <- state[24:26]  gripper_left_qpos  (2->1)
        action[15:22] right arm        <- state[28:35]  arm_right_qpos
        action[22]    right gripper    <- state[49:51]  gripper_right_qpos (2->1)

    Arms/trunk are absolute-position joint commands (same units as ``*_qpos``), so
    they copy directly. Each gripper has two finger joints; OmniGibson's smooth
    controller drives them symmetrically and its own state->command inverse is the
    finger mean, which we rescale from ``[0, GRIPPER_QMAX]`` to the action's
    ``[-1, 1]``. Accepts ``[..., 61]`` and returns ``[..., 23]``.

    ``base_vel_state_scale`` divides the base ``vx,vy`` (dims 0 and 1, i.e. ``[..., 0:2]``) by the given
    factor; ``rz`` is left untouched. The R1Pro state records base linear velocity
    at ~0.75x the action command (empirically state/action median 0.745/0.751),
    and with ``use_state`` the state is fed through the SAME action encoder as the
    actions, so dividing by 0.75 aligns the state frame with the action frames.
    Defaults to ``0.75``, the measured factor: the vxvy ablation arm scored highest of
    the eight arms tested (0.812 vs 0.763 baseline at the official 1.5x budget). ``1.0``
    is deprecated -- it is the original unscaled behaviour, kept only so checkpoints
    trained before the fix stay servable, and it should be removed once they are retired.
    Set it identically for training and serving: a legacy checkpoint trained at ``1.0``
    MUST be served with ``base_vel_state_scale=1.0``, or the state frame arrives in a
    different numeric range than it was trained on.
    """
    assert base_vel_state_scale > 0.0, f"base_vel_state_scale must be > 0, got {base_vel_state_scale}"
    a = torch.zeros((*state.shape[:-1], 23), dtype=state.dtype, device=state.device)
    a[..., 0:3] = state[..., 0:3]  # base_qvel (x, y, rz velocity)
    if base_vel_state_scale != 1.0:
        # Reverse the state-frame scaling on vx,vy so the state matches the action's
        # numeric range through the shared encoder. rz (dim 2) is already 1:1.
        a[..., 0:2] = a[..., 0:2] / base_vel_state_scale
    a[..., 3:7] = state[..., 53:57]  # trunk_qpos
    a[..., 7:14] = state[..., 3:10]  # arm_left_qpos
    a[..., 14] = (2.0 * state[..., 24:26].mean(-1) / GRIPPER_QMAX - 1.0).clamp(-1.0, 1.0)  # gripper_left
    a[..., 15:22] = state[..., 28:35]  # arm_right_qpos
    a[..., 22] = (2.0 * state[..., 49:51].mean(-1) / GRIPPER_QMAX - 1.0).clamp(-1.0, 1.0)  # gripper_right
    return a


def _cap(text: str) -> str:
    """Capitalize the first character, preserving inner caps (e.g. "Halloween")."""
    return text[0].upper() + text[1:] if text else text


def _join_and(items: list[str]) -> str:
    """Join clauses with commas and a final "and" (Oxford-style)."""
    if len(items) <= 1:
        return items[0] if items else ""
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + ", and " + items[-1]


def render_custom_prompt(entry: dict, task_name: str) -> str:
    """Render a ``behavior1k_task_prompts.json`` entry into the "custom" caption.

    Consecutive flat subtasks form one comma-joined clause; each ``for_each`` loop
    is its own "for each X, ..." clause; clauses are separated by "; ".
    """
    clauses: list[str] = []
    flat: list[str] = []

    def flush() -> None:
        if flat:
            clauses.append(_join_and(flat))
            flat.clear()

    for st in entry.get("subtasks", []):
        if isinstance(st, dict):  # for_each loop block
            flush()
            try:
                clauses.append(f"for each {st['for_each']}, {_join_and(st['steps'])}")
            except KeyError as exc:
                raise ValueError(
                    f"malformed for_each block in prompt_map for task {task_name!r}: missing key {exc}"
                ) from exc
        else:
            flat.append(st)
    flush()

    instructions = "; ".join(clauses)
    # Instructions-only caption: the subtask steps carry the task, so drop the
    # "<task>, following the instructions:" preamble (fall back to the task
    # name only when a task has no steps).
    return f"{_cap(instructions)}." if instructions else f"{_cap(entry['task'])}."
