# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Shared-prefix loss filtering (cosmos-rl `min_filter_prefix_tokens`).

Guards the semantics an earlier port got wrong: the config value is a THRESHOLD, and the number of
tokens masked is COMPUTED per completion as its maximal shared-prefix length — applied only where
that prefix set holds different rewards.
"""

from __future__ import annotations

import pytest

from cosmos_framework.posttrain.algorithm.annotators.prefix_filter import (
    assign_shared_prefix_ignore,
    find_maximal_prefix_groups,
)
from cosmos_framework.posttrain.algorithm.reward import RewardContractError
from cosmos_framework.posttrain.schema import ObsBundle, Status, Trajectory, Transition


def _traj(gid: str, ids: list[int], reward: float, tid: str) -> Trajectory:
    tr = Transition(obs=ObsBundle(), action=ids, done=True)
    tr.reward = {"final": float(reward)}
    return Trajectory(transitions=[tr], id=tid, group_id=gid, status=Status.COMPLETED)


def _n(t: Trajectory) -> int:
    return int(t.transitions[0].algo_extra["n_ignore_prefix_tokens"])


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    "seqs,n,want",
    [
        ([[1, 2, 3, 4], [1, 2, 9, 9], [7, 7, 7]], 1, {(1, 2): [0, 1]}),
        ([[1, 2, 3], [1, 2, 3], [1, 2, 3]], 1, {(1, 2, 3): [0, 1, 2]}),
        ([[1, 2], [3, 4]], 1, {}),
        ([[1, 2, 3], [1, 9, 9]], 2, {}),  # threshold blocks a length-1 shared prefix
        ([[1, 2, 3, 4], [1, 2, 3, 5], [1, 2, 9], [1, 2, 9]], 1, {(1, 2, 3): [0, 1], (1, 2, 9): [2, 3]}),
        ([[1, 2, 3], [1, 2, 4]], None, {}),  # None disables (cosmos-rl default)
        ([[1, 2, 3], [1, 2, 4]], 0, {}),
        ([[1, 2, 3]], 1, {}),  # a single sequence shares nothing
    ],
)
def test_find_maximal_prefix_groups(seqs, n, want):
    """MAXIMAL means reported at the deepest still-shared point, not every intermediate depth —
    verified against cosmos-rl's own implementation (8/8) in the parity harness."""
    got = {k: sorted(v) for k, v in find_maximal_prefix_groups(seqs, n).items()}
    assert got == {k: sorted(v) for k, v in want.items()}


@pytest.mark.L0
@pytest.mark.CPU
def test_masks_only_same_prefix_different_reward():
    """The de-biasing case: two completions share '#### ' but earn different rewards, so the shared
    prefix cannot explain the gap and is masked FOR THOSE TWO ONLY. A completion sharing nothing
    keeps every token — the bug this replaces masked the first token of EVERY completion, deleting
    the '####' marker the GSM8K verifier keys on."""
    EOS = 151645
    g = [
        _traj("g", [820, 220, 18, 20, EOS], 0.0, "a"),  # "#### 35" wrong
        _traj("g", [820, 220, 22, 17, EOS], 1.0, "b"),  # "#### 72" right
        _traj("g", [23, 20, EOS], 0.0, "c"),  # "85" — no shared prefix
    ]
    assign_shared_prefix_ignore(g, min_filter_prefix_tokens=1, eos_token_id=EOS)
    assert (_n(g[0]), _n(g[1]), _n(g[2])) == (2, 2, 0)


@pytest.mark.L0
@pytest.mark.CPU
def test_never_masks_the_whole_completion():
    """cosmos-rl guard: if the shared prefix IS the entire completion, masking it would erase the
    sample from the loss, so that trajectory is skipped."""
    g = [_traj("g", [1, 2], 0.0, "a"), _traj("g", [1, 2, 3, 4], 1.0, "b")]
    assign_shared_prefix_ignore(g, 1, None)
    assert _n(g[0]) == 0  # prefix (1,2) == the whole completion -> skipped
    assert _n(g[1]) == 2  # the longer one is safely masked


@pytest.mark.L0
@pytest.mark.CPU
def test_trailing_eos_trimmed_from_prefix():
    """A shared prefix ending in EOS gets one token trimmed (cosmos-rl parity) — otherwise the stop
    token itself would be masked as though it were shared content."""
    EOS = 7
    g = [_traj("g", [1, 2, EOS], 0.0, "a"), _traj("g", [1, 2, EOS], 1.0, "b")]
    assign_shared_prefix_ignore(g, 1, eos_token_id=EOS)
    assert (_n(g[0]), _n(g[1])) == (2, 2)  # (1,2,EOS) -> (1,2)


@pytest.mark.L0
@pytest.mark.CPU
def test_prefix_filter_uses_the_strict_terminal_final_reward_contract() -> None:
    group = [_traj("g", [1, 2, 3], 0.0, "a"), _traj("g", [1, 2, 4], 1.0, "b")]
    group[0].transitions[-1].reward = {"task": 0.0}

    with pytest.raises(RewardContractError, match=r"reward\['final'\]"):
        assign_shared_prefix_ignore(group, 1)
