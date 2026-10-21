# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Independent dense critic role for synchronous PPO."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from cosmos_framework.posttrain.algorithm.ppo.batch import PPOUpdatePlan
from cosmos_framework.posttrain.algorithm.ppo.objective import VALUE_WINDOW_WEIGHTED, make_ppo_value_objective
from cosmos_framework.posttrain.algorithm.ppo.records import PPORoleUpdateReceipt
from cosmos_framework.posttrain.algorithm.ppo.views import (
    CriticScore,
    PPOCriticTargetReceipt,
    PPOEpisode,
    attach_critic_score,
    global_batch_digest,
    processor_inputs_for_role,
    project_rollout_for_role,
    training_episode_to_trajectory,
    training_target_digest,
    trajectory_digest,
    trajectory_to_episode,
    validate_training_target_overlay,
)
from cosmos_framework.posttrain.model.ppo import (
    episode_model_inputs,
    prepare_deterministic_ppo_forward,
)
from cosmos_framework.posttrain.model.value import action_state_values, model_device
from cosmos_framework.posttrain.roles.trainer.ppo_runtime import (
    PPOTrainableRole,
    run_cached_ppo_minibatch,
    run_ppo_batch_update,
)
from cosmos_framework.posttrain.schema import Trajectory
from cosmos_framework.posttrain.utils.dist import run_rank_converged


class PPOCriticRole(PPOTrainableRole):
    """Own the critic model, optimizer, scheduler, predictions, and value updates.

    Like the policy, this is the shared ``LLMTrainer`` engine with the PPO value `Objective`
    injected; the role adds only the prediction/target lifecycle the critic owns and the same engine
    seams (deterministic update forward, converged window opening, post-step finite check,
    optimizer-step schedule) under the same `fail_stop` non-finite policy.
    """

    value_clip: float
    value_coef: float

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        scheduler: Any = None,
        max_grad_norm: float = 0.0,
        force_finite_grads: bool = True,
        value_clip: float = 0.2,
        value_coef: float = 1.0,
        checkpoint_identity: Mapping[str, Any] | None = None,
        grad_accum: int = 1,
        parallel_dims: Any = None,
        checkpoint_manager: Any = None,
    ) -> None:
        if not 0.0 <= float(max_grad_norm) < float("inf"):
            raise ValueError(f"max_grad_norm must be finite and non-negative, got {max_grad_norm}")
        objective = make_ppo_value_objective(value_clip=value_clip, value_coef=value_coef)
        if checkpoint_identity is not None:
            objective.identity["ppo_recipe"] = dict(checkpoint_identity)
        super().__init__(
            model,
            objective,
            optimizer,
            lr_scheduler=scheduler,
            max_grad_norm=max_grad_norm,
            force_finite_grads=force_finite_grads,
            nonfinite_update="fail_stop",
            require_gradients=True,
            grad_accum=grad_accum,
            checkpoint_manager=checkpoint_manager,
        )
        self.value_clip = float(objective.identity["value_clip"])
        self.value_coef = float(objective.identity["value_coef"])
        # See `roles/trainer/ppo.py::PPOPolicyTrainer.__init__`: the live model-DP factorization is
        # what lets this role expand a controller update plan into its own share of the work.
        self._initialize_ppo_role("critic", parallel_dims, digest=trajectory_digest)
        # The critic's ingress is a model-DP SHARD, so its cache alone cannot say where that shard
        # sits in the global episode order — and without that it cannot reproduce a schedule the
        # other ranks agree with. `install_targets` already all-gathers the global order for its
        # receipt; retaining it here is what makes the critic's update schedule derivable at all.
        self._installed_receipt: PPOCriticTargetReceipt | None = None

    # ``_restore_optimizer_step`` is the shared engine's: it rewinds the durable clocks AND clears
    # the accumulation window and any partial-update poison, which is what makes checkpoint restore
    # the documented recovery path for this role.

    def predict(
        self,
        trajectories: list[Trajectory],
    ) -> tuple[Trajectory, ...]:
        """Return Trajectory value overlays, including the final bootstrap state."""
        if not trajectories:
            raise ValueError("critic predict received an empty episode batch")
        if any(not isinstance(item, Trajectory) for item in trajectories):
            raise TypeError("critic predict accepts only Trajectory values")
        rollout_trajectories = tuple(trajectories)
        batch_id = global_batch_digest(rollout_trajectories)
        cached_trajectories = tuple(project_rollout_for_role(rollout_trajectories, "critic", batch_id=batch_id))
        episodes = tuple(trajectory_to_episode(item) for item in rollout_trajectories)
        inputs_by_episode = {item.id: processor_inputs_for_role(item, "critic") for item in rollout_trajectories}
        self._ppo_cache.install(batch_id, cached_trajectories)
        try:
            prepare_deterministic_ppo_forward(self.model)
            device = model_device(self.model)
            prepared_inputs = run_rank_converged(
                lambda: tuple(episode_model_inputs(episode, device, inputs_by_episode) for episode in episodes),
                "critic input preparation",
            )
            results: list[CriticScore] = []
            with torch.no_grad():
                for episode_index, (episode, inputs) in enumerate(zip(episodes, prepared_inputs, strict=True)):
                    sequence_values = self.model(**inputs)  # [1,P+T]
                    results.append(
                        run_rank_converged(
                            lambda episode=episode, sequence_values=sequence_values: self._predict_episode(
                                episode,
                                sequence_values,
                            ),
                            f"critic prediction episode {episode_index}",
                        )
                    )
            return tuple(
                attach_critic_score(trajectory, result)
                for trajectory, result in zip(rollout_trajectories, results, strict=True)
            )
        except Exception:
            self._ppo_cache.abort(batch_id)
            raise

    def install_targets(self, targets: list[Trajectory]) -> PPOCriticTargetReceipt:
        """Install policy targets over cached critic inputs without exposing either to the controller."""
        if not targets:
            raise ValueError("critic target installation received an empty batch")
        batch_id = global_batch_digest(targets)
        target_by_id: dict[str, Trajectory] = {}
        for target in targets:
            if not isinstance(target, Trajectory) or not target.id or target.id in target_by_id:
                raise ValueError("critic target installation received invalid or duplicate trajectory IDs")
            target_by_id[target.id] = target
        cached_ids = tuple(self._ppo_cache.digest_map(batch_id))
        if set(target_by_id) != set(cached_ids):
            raise ValueError(
                "critic target IDs do not match cached prediction inputs: "
                f"missing={sorted(set(cached_ids) - set(target_by_id))!r}, "
                f"extra={sorted(set(target_by_id) - set(cached_ids))!r}"
            )
        bases = self._ppo_cache.select(batch_id, cached_ids)
        installed: list[Trajectory] = []
        target_digests: list[str] = []
        for base in bases:
            item = validate_training_target_overlay(base, target_by_id[base.id], batch_id=batch_id)
            installed.append(training_episode_to_trajectory(base, item))
            target_digests.append(training_target_digest(target_by_id[base.id]))
        self._ppo_cache.replace(batch_id, installed, preserve_identity=False)
        local = (cached_ids, tuple(target_digests))
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            gathered: list[tuple[tuple[str, ...], tuple[str, ...]] | None] = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, local)
            if any(shard is None for shard in gathered):
                raise RuntimeError("critic target receipt all-gather returned an incomplete shard set")
            shards = [shard for shard in gathered if shard is not None]
            episode_ids = tuple(episode_id for ids, _digests in shards for episode_id in ids)
            global_target_digests = tuple(digest for _ids, digests in shards for digest in digests)
        else:
            episode_ids, global_target_digests = local
        receipt = PPOCriticTargetReceipt(
            batch_id=batch_id,
            episode_ids=episode_ids,
            target_digests=global_target_digests,
        )
        self._installed_receipt = receipt
        return receipt

    def _global_episode_ids(self, batch_id: str) -> tuple[str, ...]:
        """Return the batch's complete episode order, as target installation recorded it."""
        receipt = self._installed_receipt
        if receipt is None:
            raise RuntimeError("critic has no installed PPO target receipt")
        if receipt.batch_id != batch_id:
            raise RuntimeError(
                f"critic installed receipt is stale: requested {batch_id!r}, active is {receipt.batch_id!r}"
            )
        return receipt.episode_ids

    def train_batch(
        self,
        batch_id: str,
        plan: PPOUpdatePlan,
    ) -> PPORoleUpdateReceipt:
        """Run the critic's COMPLETE update window over one prepared batch.

        Same ownership as the policy (`roles/trainer/ppo.py::PPOPolicyTrainer.train_batch`): the
        role expands the plan against its own cache and its own model-DP coordinate, and the
        controller sees one receipt.
        """

        def update(microbatches: list[list[str]]) -> dict[str, float | int]:
            return self.update_cached_minibatch(
                batch_id,
                microbatches,
                value_clip=self.value_clip,
                value_coef=self.value_coef,
            )

        return run_ppo_batch_update(
            role="critic",
            batch_id=batch_id,
            plan=plan,
            global_episode_ids=self._global_episode_ids(batch_id),
            cached_episode_ids=self._ppo_cache.episode_ids(batch_id),
            parallel_dims=self._parallel_dims,
            optimizer_step=lambda: self.optimizer_step,
            update=update,
        )

    def clear_batch(self, batch_id: str) -> None:
        """Strictly clear a successfully consumed critic cache."""
        self._ppo_cache.clear(batch_id)
        self._installed_receipt = None

    def abort_batch(self, batch_id: str) -> None:
        """Retry-safe force cleanup for a partially installed critic cache."""
        self._ppo_cache.abort(batch_id)
        if self._installed_receipt is not None and self._installed_receipt.batch_id == batch_id:
            self._installed_receipt = None

    def update_cached_minibatch(
        self,
        batch_id: str,
        microbatches: list[list[str]],
        *,
        value_clip: float,
        value_coef: float = 1.0,
    ) -> dict[str, float | int]:
        """Perform one critic optimizer step from ID-only microbatches."""
        return run_cached_ppo_minibatch(
            self,
            self._ppo_cache,
            batch_id,
            microbatches,
            role="critic",
            requested_coefficients={"value_clip": value_clip, "value_coef": value_coef},
            bound_coefficients={"value_clip": self.value_clip, "value_coef": self.value_coef},
            weighted=VALUE_WINDOW_WEIGHTED,
        )

    def _predict_episode(
        self,
        episode: PPOEpisode,
        sequence_values: torch.Tensor,  # [1,P+T]
    ) -> CriticScore:
        """Postprocess one completed rank-local model forward under convergence."""
        values = action_state_values(
            sequence_values,
            len(episode.prompt_token_ids),
            len(episode.action_token_ids),
        )[0]  # [T+1]
        if not bool(torch.isfinite(values).all().item()):
            raise ValueError(f"critic produced non-finite values for episode {episode.episode_id!r}")
        return CriticScore(
            episode_id=episode.episode_id,
            policy_version=episode.policy_version,
            rendered_prompt=episode.rendered_prompt,
            prompt_token_ids=episode.prompt_token_ids,
            action_token_ids=episode.action_token_ids,
            action_mask=episode.action_mask,
            media=episode.media,
            values=tuple(float(value) for value in values.cpu().tolist()),
        )


__all__ = ["PPOCriticRole"]
