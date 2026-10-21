# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Local (reference) rollout body (DESIGN §1.2 body=(a) local compute).

`LocalRollout` wraps a shared model instance and generates completions for a batch
of prompts, returning Trajectories.

KEY DISCIPLINE (DESIGN §7): when the trainer and rollout share ONE model instance the weight
edge is a no-op (rollout already sees the trainer's step); otherwise it receives pushed weights
through the producer-driven edge (`apply_bucket` + rollout-wide version prepare/commit,
design 2026-07-27 §2).
Either way the rollout uses *its own* model reference and never reaches into the trainer, so a
separate inference engine + real resharding drops in later (#4) without changing the runner.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cosmos_framework.posttrain.roles.rollout.base import (
    ShardSpec,
    WeightReceiverBase,
)
from cosmos_framework.posttrain.roles.rollout.validation import normalize_validation_semantics_identity
from cosmos_framework.posttrain.schema import ObsBundle, Status, Trajectory, Transition

if TYPE_CHECKING:
    pass


class LocalRollout(WeightReceiverBase):
    def __init__(self, model: Any, sampling: dict | None = None):
        self.model = model  # v0: shared with LocalTrainer (see discipline note)
        self.sampling = sampling or {}
        self._weight_version = 0

    def validation_semantics_identity(self) -> dict[str, Any]:
        """Require toy/custom local rollouts to declare their otherwise opaque generation path."""
        capability = getattr(self.model, "validation_semantics_identity", None)
        if not callable(capability):
            raise TypeError(
                f"local rollout model {type(self.model).__module__}.{type(self.model).__qualname__} "
                "must implement validation_semantics_identity() before held-out validation"
            )
        model_identity = normalize_validation_semantics_identity(
            capability(),
            context="local rollout model.validation_semantics_identity()",
        )
        return normalize_validation_semantics_identity(
            {
                "version": 1,
                "backend": "local",
                "backend_class": f"{type(self.model).__module__}.{type(self.model).__qualname__}",
                "model": model_identity,
                "sampling": normalize_validation_semantics_identity(
                    dict(self.sampling),
                    context="local rollout sampling",
                ),
            },
            context="local rollout validation semantics identity",
        )

    # --- weight-transfer edge target (DESIGN §8.2 #4) ---
    def apply_weights(self, source: "Any | None" = None) -> None:
        """v0: NO-OP. Shared instance → rollout already sees updated weights after
        the trainer's optimizer step. The edge exists so v1+ can do real resharding
        from a separate trainer mesh into a separate inference engine.
        PEER-ONLY: source is the dp-aligned policy peer → consumer-driven per-bucket pull
        stream; None (or no backend) keeps the v0 stamp-only contract."""
        from cosmos_framework.posttrain.comm.weight_transfer import pull_stream_from_peer

        return pull_stream_from_peer(self, source)  # stamps on the degenerate edges itself

    def prepare_recv(self) -> dict[str, ShardSpec]:
        """v1: no rollout-side sharding — receive each param whole, dim 0 (mirrors HFRollout)."""
        return {k: ShardSpec(0, (tuple(v.shape)[0] if v.dim() else 0), 0) for k, v in self.model.state_dict().items()}

    # --- streaming weight sync (per-bucket); set_bucket_plan + check_plan_coverage on WeightReceiverBase ---
    def apply_bucket(self, payload, i: int) -> None:
        # Producer-driven landing: fetch (IPC import) + load bucket i in place. PURE LOAD — the version
        # readiness/publication protocol is controller-owned, not per-bucket here.
        be = getattr(self, "_weight_backend", None)
        if be is not None:
            be.apply_bucket(self.model, i, payload)

    # --- generation ---
    def generate(self, prompts: list[ObsBundle], **knobs: Any) -> list[Trajectory]:
        """sim → policy is degenerate here: source is a prompt batch (1-step Env). Accepts the
        shared `**knobs` surface (roles/rollout/base.py); v0 ignores group/seed (one completion
        per prompt, no group sampling) — a real inference engine (#6) will honor them."""
        # TODO: set eval()/no_grad on the shared model; real batched generate via the
        # inference engine (SGLang/vLLM, #6). v0: a placeholder single-turn generate.
        trajs: list[Trajectory] = []
        for i, obs in enumerate(prompts):
            completion, rollout_logprobs = self._generate_one(obs)
            tr = Transition(
                obs=obs,
                action=completion,
                # `rollout_logprobs`, matching HFRollout and VLLMRollout. This said `logprob` until
                # per-token pi_old capture landed and renamed the key everywhere it was live; this
                # body was missed because the line is unreachable (see `_generate_one`). Leaving the
                # old name here would have handed the eventual implementer a contract the clipped
                # objective does not read -- `make_clipped_objective` looks up `rollout_logprobs`
                # and nothing else.
                policy_info={
                    "rollout_logprobs": rollout_logprobs,
                    "completion_token_count": len(completion),
                    "weight_version": self._weight_version,
                },
                done=True,
            )
            trajs.append(Trajectory(transitions=[tr], id=f"r{i}", status=Status.COMPLETED))
        return trajs

    def _generate_one(self, obs: ObsBundle):
        """Generate one completion. NOT IMPLEMENTED — `generate()` above is unreachable until it is.

        Contract for whoever implements it, stated because getting it wrong is silent-ish: return
        `(completion, rollout_logprobs)` where `rollout_logprobs` is PER COMPLETION TOKEN — one
        logprob per token of `completion`, in order — not a single sequence logprob. The clipped
        objective indexes it positionally against the loss mask, so a scalar under
        that name fails with a length mismatch rather than anything that names the real mistake.
        """
        raise NotImplementedError("v0 TODO: call self.model.generate(...) for the VLM")
