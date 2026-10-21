# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Local (reference) trainer body + adapter contract (DESIGN §6, §8.2 #6).

Adapter contract: the inner may wrap an existing trainer (delegate train_step) or
absorb its loss as an objective plugin. The framework shell handles: receive batch
from a channel + pull bulk, inject the mesh, expose update_weights/save/load/
compute_logprob, and participate in weight-transfer.

v0: `LocalTrainer` wraps a single shared model + optimizer; no Ray, no mesh
injection, no real weight-transfer. SFT = dataset Env + targets + CE objective is
the degenerate config and the first end-to-end MVP loop.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from cosmos_framework.posttrain.schema import Trajectory
from cosmos_framework.posttrain.schema.checkpoint import CheckpointComponentPlan


class LocalTrainer:
    """v0 trainer body sharing a model instance with LocalRollout.

    Discipline (DESIGN §7): trainer/rollout are separate WGs with a weight-transfer
    edge between them; they merely *happen* to point at the same `model` in v0, and
    `push_weights()` is a no-op. Do not let "same model" leak past the edge.
    """

    def __init__(self, model: Any, optimizer: Any = None, objective: str = "grpo") -> None:
        self.model = model
        self.opt = optimizer
        # Preserve the descriptive alias used by the local example while satisfying the shared
        # structural trainer contract through ``opt``.
        self.optimizer = optimizer
        self.objective = objective  # TODO(#2): pluggable objective (PPO/GRPO/DPO/CE + KL + staleness)
        self._iter = 0

    def train(self, batch: list[Trajectory]) -> dict:
        """One training step. v0: compute objective loss, backward, optimizer step.

        TODO(#2): advantage (GAE / GRPO group-norm) on metadata before this; loss in
        algorithm; staleness correction (TIS) once async.
        TODO(#6): real FSDP/Megatron mesh; micro-batch / grad-accum.
        """
        raise NotImplementedError("v0 TODO: model.train(); fwd→objective loss→backward→optim.step")

    def offload(self) -> float:
        """No device state of its own to release (the model is shared with the rollout)."""
        return 0.0

    def onload(self) -> float:
        return 0.0

    def push_weights(self, target: Any = None) -> None:
        """Weight-transfer edge (DESIGN §8.2 #4). v0: NO-OP (shared instance).
        TODO(v1+): reshard trainer mesh → rollout mesh over the weight plane; stamp
        weight_version onto produced samples."""
        return None

    def iteration(self) -> int:
        """Return the local placeholder's optimizer clock."""
        return self._iter

    def publication_version(self) -> int:
        """Return the local placeholder's lockstep weight-publication clock."""
        return self._iter

    # framework-shell hooks (kept so the contract is real)
    def compute_logprob(self, batch: list[Trajectory]) -> Any:  # for ref/teacher (§1.4 hook)
        raise NotImplementedError("TODO")

    def save(
        self,
        path: str,
        *,
        publish: bool = True,
        update_latest: bool = True,
        components: Iterable[str] | CheckpointComponentPlan | None = None,
    ) -> None: ...  # TODO(#5): DCP / safetensors (FT solution from cosmos-rl)

    def load(self, path: str) -> int: ...

    def finalize_checkpoints(self) -> None:
        """The local placeholder owns no asynchronous checkpoint work."""
        return None

    def set_loaded_checkpoint_provenance(self, provenance: dict[str, int | str]) -> None:
        del provenance

    def loaded_checkpoint_provenance(self) -> dict[str, int | str] | None:
        return None
