# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""A test-only trainer that publishes on the driver's outer clock, the way `roles/trainer/ppo.py::PPOPolicyTrainer` does.

`control/loop/weight_sync.py::weight_sync` checks `policy.iteration()` against the version it publishes, while
`comm/weight_transfer.py::push_stream_to_peer` stamps `policy.publication_version()`. A trainer that runs several
optimizer steps per publication must therefore answer `iteration()` with the publication clock, or the barrier
rejects the very stream it just received. PPO's production trainer owns that clock; this wrapper gives the GR00T
weight-sync driver the same contract for its `--update-weights-interval N` acceptance without adding a production
trainer. It changes no arithmetic: `_iter` still counts optimizer steps and is exposed as `optimizer_step`.
"""

from __future__ import annotations

from cosmos_framework.posttrain.roles.trainer.llm import LLMTrainer


class PublicationClockTrainer(LLMTrainer):
    """`LLMTrainer` whose published version is set by the driver, decoupled from the optimizer step count."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Own the publication clock from the start: the base's lockstep fallback would report optimizer steps.
        self.set_publication_version(0)

    @property
    def optimizer_step(self) -> int:
        """The durable optimizer counter, kept visible so the driver can assert `optimizer_step == N * k`."""
        return self._iter

    def iteration(self) -> int:
        """The outer publication version, which is what `weight_sync` publishes and checks."""
        return self.publication_version()
