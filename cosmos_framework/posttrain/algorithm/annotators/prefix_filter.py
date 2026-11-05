# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Shared-prefix loss filtering for GRPO groups (cosmos-rl `min_filter_prefix_tokens`).

WHY THIS EXISTS. Within one GRPO group, several completions often begin identically — the model
commits to the same opening tokens and only diverges later. When those same-prefix completions end
up with DIFFERENT rewards, the shared prefix demonstrably did not cause the difference, yet the
group-relative advantage pushes its tokens up in one completion and down in another. That is pure
noise on tokens the outcome does not depend on, so cosmos-rl masks the shared prefix out of the
loss for exactly those completions.

PORTED FROM `cosmos_rl/utils/util.py::{build_trie,find_maximal_prefix_groups}` +
`cosmos_rl/reward/local_calculator.py:238-268`, semantics preserved including both guards (strip a
trailing EOS from the prefix; skip when the prefix IS the whole completion, which would erase the
sample entirely).

NOTE ON THE NAME. cosmos-rl's config knob `min_filter_prefix_tokens` is a THRESHOLD — the minimum
prefix length worth considering — NOT the number of tokens to skip. The number skipped is COMPUTED
per completion as the length of its maximal shared prefix. An earlier version of this port read the
TOML's `min_filter_prefix_tokens = 1` as a constant "ignore the first 1 token of every completion",
which silently deleted the first real content token from every loss mask — including the `####`
marker the GSM8K verifier keys on.
"""

from __future__ import annotations

from typing import Any

from cosmos_framework.posttrain.algorithm.reward.contract import _terminal_reward


class _TrieNode:
    __slots__ = ("children", "idxs")

    def __init__(self) -> None:
        self.children: dict[int, _TrieNode] = {}
        self.idxs: list[int] = []


def _build_trie(seqs: list[list[int]]) -> _TrieNode:
    root = _TrieNode()
    for i, s in enumerate(seqs):
        node = root
        for x in s:
            node.idxs.append(i)
            node = node.children.setdefault(x, _TrieNode())
        node.idxs.append(i)
    return root


def find_maximal_prefix_groups(seqs: list[list[int]], n: int | None) -> dict[tuple[int, ...], list[int]]:
    """Maximal shared prefixes of length >= `n`, mapped to the sequence indices sharing them.

    "Maximal" means no LONGER prefix is still shared by more than one sequence — so each group is
    reported at the deepest point its members still agree, not at every intermediate depth.

    Args:
        seqs: token-id sequences (one per completion in a group).
        n: minimum prefix length to report; `None` or `< 1` disables the feature (returns `{}`),
            matching cosmos-rl's `min_filter_prefix_tokens=None` default.

    Returns:
        `{prefix_tuple: [sequence indices]}`, only for prefixes shared by >= 2 sequences.
    """
    if n is None or n < 1:
        return {}
    root = _build_trie(seqs)
    result: dict[tuple[int, ...], list[int]] = {}
    stack: list[tuple[_TrieNode, tuple[int, ...], bool]] = [(root, (), False)]
    while stack:
        node, prefix, visited = stack.pop()
        if not visited:
            stack.append((node, prefix, True))  # post-order: decide after children
            for x, child in node.children.items():
                stack.append((child, prefix + (x,), False))
        elif len(prefix) >= n and len(node.idxs) > 1:
            # Maximal only if no child is itself shared by >1 sequence.
            if not any(len(c.idxs) > 1 for c in node.children.values()):
                result[prefix] = list(node.idxs)
    return result


def assign_shared_prefix_ignore(
    trajectories: list, min_filter_prefix_tokens: int | None, eos_token_id: int | None = None
) -> dict[str, int]:
    """Write a per-trajectory `n_ignore_prefix_tokens` onto `transition.algo_extra`.

    For each `group_id`: find maximal shared prefixes among the group's completions; where a
    shared-prefix set holds MORE THAN ONE distinct reward, mask that prefix out of the loss for its
    members (the prefix cannot explain the reward spread, so its advantage is noise).

    Groups whose rewards are all equal are skipped entirely — they carry zero advantage anyway.

    WHERE THIS MUST RUN. Over a WHOLE group, at reward time — same requirement as the advantage
    estimator, for the same reason, and it is easy to miss because this one degrades quietly rather
    than raising. The maximal shared prefix of a SUBSET of a group is at least as long as the
    group's, so a rank holding half a group masks at least as many tokens as intended, and two ranks
    holding different halves mask differently. The loss then depends on how dispatch happened to
    split the batch. `shared_prefix_ignore` is the reward-stage annotator; see
    `algorithm/annotators/advantage.py` for the argument in full.

    Args:
        trajectories: the batch. Each is expected to hold one transition (v0 flat form).
        min_filter_prefix_tokens: the THRESHOLD (cosmos-rl's knob). `None`/`<1` disables — every
            trajectory gets 0, which is the default and the pre-existing behaviour.
        eos_token_id: if the shared prefix ends with EOS it is trimmed by one, mirroring cosmos-rl —
            a prefix that includes the stop token would otherwise mask the whole completion.

    Returns:
        `{trajectory_id: n_ignore}` for the trajectories that got a NON-ZERO value (diagnostics /
        logging); every trajectory's `algo_extra["n_ignore_prefix_tokens"]` is set either way.
    """
    assigned: dict[str, int] = {}
    groups: dict[Any, list] = {}
    for t in trajectories:
        for tr in t.transitions:
            tr.algo_extra["n_ignore_prefix_tokens"] = 0
        groups.setdefault(t.group_id, []).append(t)
    if min_filter_prefix_tokens is None or min_filter_prefix_tokens < 1:
        return assigned

    for members in groups.values():
        rewards = [_terminal_reward(m) for m in members]
        if len(set(rewards)) <= 1:
            continue  # a tied group has zero advantage; nothing to de-bias
        seqs = [_completion_ids(m) for m in members]
        for prefix, idxs in find_maximal_prefix_groups(seqs, min_filter_prefix_tokens).items():
            if len({rewards[i] for i in idxs}) <= 1:
                continue  # same prefix AND same reward -> the prefix is not the confound
            pref = prefix[:-1] if (eos_token_id is not None and prefix and prefix[-1] == eos_token_id) else prefix
            n_ign = len(pref)
            if n_ign < 1:
                continue
            for i in idxs:
                # Never mask the ENTIRE completion — that would drop the sample from the loss.
                if n_ign >= len(seqs[i]):
                    continue
                for tr in members[i].transitions:
                    tr.algo_extra["n_ignore_prefix_tokens"] = n_ign
                assigned[str(members[i].id)] = n_ign
    return assigned


def _completion_ids(traj) -> list[int]:
    """The completion token ids of a trajectory's first transition, as a plain list."""
    if not traj.transitions:
        return []
    a = traj.transitions[0].action
    if a is None:
        return []
    if hasattr(a, "tolist"):
        return [int(x) for x in a.tolist()]
    if isinstance(a, str):
        return []  # a decoded-string producer has no ids to compare; prefix filtering is a no-op
    return [int(x) for x in a]


def shared_prefix_ignore(min_filter_prefix_tokens: int | None, eos_token_id: int | None = None):
    """Bind `assign_shared_prefix_ignore` into a reward-stage annotator.

    Same shape as `algorithm/annotators/advantage.py::group_relative` — `(list[Trajectory]) -> None`, run once
    per reward batch — so both group-relative annotations reach the batch by the same route, before
    dispatch can split a group. Returns a no-op when the filter is disabled rather than `None`, so
    the caller never has to branch.

    Args:
        min_filter_prefix_tokens: the THRESHOLD; `None`/`<1` disables (the default).
        eos_token_id: trims a trailing EOS off the shared prefix, mirroring cosmos-rl.

    Returns:
        A callable that writes `algo_extra["n_ignore_prefix_tokens"]` on every transition it sees.
    """

    def annotate(trajectories: list) -> None:
        assign_shared_prefix_ignore(trajectories, min_filter_prefix_tokens, eos_token_id)

    return annotate


__all__ = ["find_maximal_prefix_groups", "assign_shared_prefix_ignore", "shared_prefix_ignore"]
