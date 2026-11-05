# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Reward-stage annotators: they rewrite trajectories BEFORE dispatch, over a whole group.

Named for the word already in the code -- `RewardWorker(annotators=...)`,
`config/build.py::build_reward_annotators` -- so someone reading the worker and asking "where do
these live" gets the answer from the path.

WHY THEY ARE NOT OBJECTIVES, and why that distinction earns a directory. An objective takes
`(model, batch)` and returns `(loss, metrics)`; it sees whatever slice dispatch handed this rank. An
annotator takes a batch of trajectories at REWARD time and writes onto them, and it is the only
stage that provably holds a whole group: one rollout call produces a prompt's G completions and the
reward worker sees exactly that. Anything group-relative must therefore be computed here, not in the
objective -- which is the entire thesis of the MR that created this package.

Where a new file goes:
    takes (model, batch) -> (loss, metrics)   ->  an objective, at the package root
    rewrites trajectories at reward time      ->  here
    neither                                   ->  it probably does not belong in algorithm/

Expected to grow: `advantage.py` names GAE as the next estimator in, which would turn GRPO into PPO
with the surrogate unchanged.
"""
