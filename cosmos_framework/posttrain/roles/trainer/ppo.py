# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""PPO policy trainer body."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch

from cosmos_framework.posttrain.algorithm.ppo.batch import PPOUpdatePlan, prepare_ppo_training_batch
from cosmos_framework.posttrain.algorithm.ppo.objective import POLICY_WINDOW_WEIGHTED, make_ppo_policy_objective
from cosmos_framework.posttrain.algorithm.ppo.records import PPORoleUpdateReceipt
from cosmos_framework.posttrain.algorithm.ppo.views import (
    PPOPreparedBatchReceipt,
    PPOTrainingEpisode,
    global_batch_digest,
    prepared_batch_receipt,
    processor_inputs_for_role,
    project_rollout_for_role,
    training_episode_to_trajectory,
    training_target_to_trajectory,
    trajectory_digest,
    trajectory_ppo_scores,
    trajectory_to_training_episode,
)
from cosmos_framework.posttrain.model.ppo import (
    episode_model_inputs,
    prepare_deterministic_ppo_forward,
)
from cosmos_framework.posttrain.model.value import (
    extract_causal_logits,
    forward_causal_model,
    model_device,
    response_action_logits,
    sampled_action_logprobs,
)
from cosmos_framework.posttrain.roles.trainer.ppo_runtime import (
    PPOTrainableRole,
    run_cached_ppo_minibatch,
    run_ppo_batch_update,
    validate_ppo_metrics,
)
from cosmos_framework.posttrain.schema import Trajectory
from cosmos_framework.posttrain.utils.dist import collective_device, raise_if_any_rank_failed, run_rank_converged


def _gather_variable_length_1d(local: torch.Tensor) -> torch.Tensor:
    """Gather a variable-length 1D tensor in rank order over the model world."""
    if local.ndim != 1:
        raise ValueError(f"on-policy delta gather requires a 1D tensor, got shape {tuple(local.shape)!r}")
    import torch.distributed as dist

    if not dist.is_initialized() or dist.get_world_size() == 1:
        return local
    length = torch.tensor(local.numel(), dtype=torch.int64, device=local.device)  # []
    lengths = [torch.zeros_like(length) for _ in range(dist.get_world_size())]
    dist.all_gather(lengths, length)
    counts = [int(value.item()) for value in lengths]
    max_count = max(counts)
    padded = torch.zeros(max_count, dtype=local.dtype, device=local.device)  # [N_max]
    padded[: local.numel()] = local
    gathered = [torch.empty_like(padded) for _ in counts]
    dist.all_gather(gathered, padded)
    return torch.cat([value[:count] for value, count in zip(gathered, counts, strict=True)])


def _global_on_policy_delta_percentiles(local_delta: torch.Tensor) -> torch.Tensor:
    """Return exact p95/p99 over all variable-length model-DP delta shards."""
    global_delta = _gather_variable_length_1d(local_delta)
    if global_delta.numel() == 0 or not bool(torch.isfinite(global_delta).all()):
        raise ValueError("global policy on-policy deltas are empty or non-finite")
    levels = torch.tensor((0.95, 0.99), dtype=torch.float64, device=global_delta.device)  # [2]
    return torch.quantile(global_delta.double(), levels)  # [2]


class PPOPolicyTrainer(PPOTrainableRole):
    """Own policy optimization only; behavior and reference tensors stay fixed.

    This is the shared ``LLMTrainer`` engine with the PPO policy `Objective` injected, not a second
    optimizer body: the engine owns the accumulation window, finite checks, clipping, ``opt.step``,
    the scheduler, and the durable optimizer counter. PPO contributes only engine configuration and
    hook bodies -- a deterministic update-mode forward, a rank-converged window opening, a post-step
    finite check, and the optimizer-step scheduler clock -- plus `fail_stop` non-finite handling,
    because an on-policy update whose counter is a controller contract cannot silently skip.

    The loss coefficients are bound ONCE, at construction, because the objective's exact-resume
    identity is derived from them; `update_cached_minibatch` still accepts them per call (the
    controller's current wire shape) and rejects any value that disagrees with the bound objective.
    """

    temperature: float
    policy_clip: float
    entropy_coef: float
    jsd_coef: float
    jsd_beta: float
    gamma: float
    gae_lambda: float
    reference_kl_coef: float

    def __init__(
        self,
        model: Any,
        optimizer: torch.optim.Optimizer,
        *,
        scheduler: Any = None,
        temperature: float = 1.0,
        max_grad_norm: float = 0.0,
        force_finite_grads: bool = True,
        policy_clip: float = 0.2,
        entropy_coef: float = 0.0,
        jsd_coef: float = 0.0,
        jsd_beta: float = 0.5,
        gamma: float = 1.0,
        gae_lambda: float = 0.95,
        reference_kl_coef: float = 0.01,
        checkpoint_identity: Mapping[str, Any] | None = None,
        grad_accum: int = 1,
        parallel_dims: Any = None,
        checkpoint_manager: Any = None,
    ) -> None:
        if not 0.01 <= float(temperature) < float("inf"):
            raise ValueError(f"temperature must be finite and >= 0.01 to match vLLM, got {temperature}")
        if not 0.0 <= float(max_grad_norm) < float("inf"):
            raise ValueError(f"max_grad_norm must be finite and non-negative, got {max_grad_norm}")
        objective = make_ppo_policy_objective(
            policy_clip=policy_clip,
            entropy_coef=entropy_coef,
            jsd_coef=jsd_coef,
            jsd_beta=jsd_beta,
            temperature=temperature,
        )
        for name, value in (("gamma", gamma), ("gae_lambda", gae_lambda)):
            if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1], got {value!r}")
        if not math.isfinite(float(reference_kl_coef)) or float(reference_kl_coef) < 0.0:
            raise ValueError(f"reference_kl_coef must be finite and non-negative, got {reference_kl_coef!r}")
        objective.identity.update(
            {
                "gamma": float(gamma),
                "gae_lambda": float(gae_lambda),
                "reference_kl_coef": float(reference_kl_coef),
            }
        )
        if checkpoint_identity is not None:
            objective.identity["ppo_recipe"] = dict(checkpoint_identity)
        super().__init__(
            model,
            objective,
            optimizer,
            lr_scheduler=scheduler,
            max_grad_norm=max_grad_norm,
            force_finite_grads=force_finite_grads,
            # A rejected window must reach the controller: its optimizer counter is a durable
            # cross-role contract with no way to express "this step did not happen".
            nonfinite_update="fail_stop",
            # Every policy rank holds trajectories, so an empty gradient inventory anywhere is a
            # placement fault rather than the usual vacuously finite result.
            require_gradients=True,
            grad_accum=grad_accum,
            checkpoint_manager=checkpoint_manager,
        )
        self.temperature = float(temperature)
        self.policy_clip = float(objective.identity["policy_clip"])
        self.entropy_coef = float(objective.identity["entropy_coef"])
        self.jsd_coef = float(objective.identity["jsd_coef"])
        self.jsd_beta = float(objective.identity["jsd_beta"])
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.reference_kl_coef = float(reference_kl_coef)
        # Take ownership of the publication clock immediately: PPO publishes on the OUTER policy
        # version, so the base class's optimizer-lockstep fallback would report minibatch steps.
        self.set_publication_version(0)
        # The role's LIVE model-DP factorization, which is what makes it able to derive its own
        # share of an update plan (`roles/trainer/ppo_runtime.py::model_dp_coordinate`). None is legitimate:
        # a single-device build, or a torchrun/spawn world where the default process group IS the
        # model world.
        self._initialize_ppo_role("policy", parallel_dims, digest=trajectory_digest)
        self._prepared_receipt: PPOPreparedBatchReceipt | None = None

    @property
    def policy_version(self) -> int:
        """Outer policy version, the weight-publication clock this role owns."""
        return self.publication_version()

    # ``_restore_optimizer_step`` is the shared engine's: it rewinds the durable clocks AND clears
    # the accumulation window and any partial-update poison, which is what makes checkpoint restore
    # the documented recovery path for this role.

    @staticmethod
    def _reduce_advantage_stats(stats: torch.Tensor) -> torch.Tensor:
        """Sum advantage moments across the complete policy model-DP world."""
        import torch.distributed as dist

        reduced = stats.clone()
        if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
            return reduced
        # Batch preparation builds these moments in float64 on CPU, while the policy's process
        # group is the accelerator group. Reducing the CPU tensor directly raises "No backend type
        # associated with device type cpu" unless a gloo group also happens to exist, so stage the
        # pair on the collective's own device and return it in the caller's device and dtype.
        staged = reduced.to(device=collective_device())
        dist.all_reduce(staged, op=dist.ReduceOp.SUM)
        return staged.to(device=reduced.device, dtype=reduced.dtype)

    def prepare_batch(
        self,
        trajectories: list[Trajectory],
    ) -> tuple[Trajectory, ...]:
        """Compute PPO targets from trajectories enriched by reward, reference, and critic roles."""
        # The ingress hands this rank one model-DP shard, so the identity must name the COMPLETE
        # batch: the controller compares whole receipts and then feeds one rank's id back to every
        # rank as its own cache key.
        batch_id = global_batch_digest(trajectories)
        policy_rollout = project_rollout_for_role(trajectories, "policy", batch_id=batch_id)
        joined = tuple(
            (base, *trajectory_ppo_scores(source)) for base, source in zip(policy_rollout, trajectories, strict=True)
        )
        episodes = tuple(item[1] for item in joined)
        reference_scores = tuple(item[2] for item in joined)
        critic_scores = tuple(item[3] for item in joined)
        training = prepare_ppo_training_batch(
            episodes,
            reference_scores,
            critic_scores,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            reference_kl_coef=self.reference_kl_coef,
            reduce_stats=self._reduce_advantage_stats,
        )
        cached = tuple(
            training_episode_to_trajectory(base, item)
            for (base, _episode, _reference, _critic), item in zip(joined, training, strict=True)
        )
        self._ppo_cache.install(batch_id, cached)
        try:
            self._prepared_receipt = prepared_batch_receipt(batch_id, training)
            targets = tuple(
                training_target_to_trajectory(base, item)
                for (base, _episode, _reference, _critic), item in zip(joined, training, strict=True)
            )
            return targets
        except Exception:
            self._ppo_cache.abort(batch_id)
            self._prepared_receipt = None
            raise

    def prepared_batch_receipt(self, batch_id: str | None = None) -> PPOPreparedBatchReceipt:
        """Return the tiny global receipt created by ``prepare_batch``."""
        receipt = self._prepared_receipt
        if receipt is None:
            raise RuntimeError("policy has no prepared PPO batch receipt")
        if batch_id is not None and receipt.batch_id != batch_id:
            raise RuntimeError(
                f"policy prepared receipt is stale: requested {batch_id!r}, active is {receipt.batch_id!r}"
            )
        # Also proves the trajectory cache still owns the same live batch.
        self._ppo_cache.digest_map(receipt.batch_id)
        return receipt

    def clear_batch(self, batch_id: str) -> None:
        """Strictly clear a successfully consumed policy cache."""
        self._ppo_cache.clear(batch_id)
        self._prepared_receipt = None

    def abort_batch(self, batch_id: str) -> None:
        """Retry-safe force cleanup for a partially installed policy cache."""
        self._ppo_cache.abort(batch_id)
        if self._prepared_receipt is not None and self._prepared_receipt.batch_id == batch_id:
            self._prepared_receipt = None

    def _cached_training(
        self,
        batch_id: str,
        episode_ids: tuple[str, ...] | list[str],
    ) -> tuple[list[PPOTrainingEpisode], dict[str, Mapping[str, Any]]]:
        trajectories = self._ppo_cache.select(batch_id, episode_ids)
        training = [trajectory_to_training_episode(trajectory) for trajectory in trajectories]
        inputs = {trajectory.id: processor_inputs_for_role(trajectory, "policy") for trajectory in trajectories}
        return training, inputs

    def score_cached_on_policy_metrics(
        self,
        batch_id: str,
    ) -> dict[str, float]:
        """Recompute this rank's OWN cached shard against its immutable behaviour logprobs.

        The controller names only the batch. Which episodes this rank scores is a consequence of
        which episodes its cache holds, so there is no second, controller-computed partition that
        could disagree with the first one; the metrics are reduced over the complete model world
        inside `score_on_policy_metrics`. This is ordinary algorithm-correctness evidence, not
        acceptance evidence: no threshold, hash, or report schema is involved.
        """
        episode_ids = self._ppo_cache.episode_ids(batch_id)
        training, inputs = self._cached_training(batch_id, episode_ids)
        return self.score_on_policy_metrics(training, policy_clip=self.policy_clip, inputs_by_episode=inputs)

    def _global_episode_ids(self, batch_id: str) -> tuple[str, ...]:
        """Return the batch's complete episode order, as the prepared receipt recorded it."""
        receipt = self.prepared_batch_receipt(batch_id=batch_id)
        return receipt.episode_ids

    def train_batch(
        self,
        batch_id: str,
        plan: PPOUpdatePlan,
    ) -> PPORoleUpdateReceipt:
        """Run the policy's COMPLETE update window over one prepared batch.

        The controller calls this once per outer iteration and reads one receipt. Everything the
        old controller-side scheduler did — epochs, global minibatches, the model-DP split, and the
        per-minibatch optimizer counter — happens here, where the cache and the live
        ``ParallelDims`` that determine it actually live.
        """

        def update(microbatches: list[list[str]]) -> dict[str, float | int]:
            return self.update_cached_minibatch(
                batch_id,
                microbatches,
                policy_clip=self.policy_clip,
                entropy_coef=self.entropy_coef,
                jsd_coef=self.jsd_coef,
                jsd_beta=self.jsd_beta,
            )

        return run_ppo_batch_update(
            role="policy",
            batch_id=batch_id,
            plan=plan,
            global_episode_ids=self._global_episode_ids(batch_id),
            cached_episode_ids=self._ppo_cache.episode_ids(batch_id),
            parallel_dims=self._parallel_dims,
            optimizer_step=lambda: self.optimizer_step,
            update=update,
        )

    def update_cached_minibatch(
        self,
        batch_id: str,
        microbatches: list[list[str]],
        *,
        policy_clip: float,
        entropy_coef: float = 0.0,
        jsd_coef: float = 0.0,
        jsd_beta: float = 0.5,
    ) -> dict[str, float | int]:
        """Perform one policy optimizer step from ID-only microbatches."""
        return run_cached_ppo_minibatch(
            self,
            self._ppo_cache,
            batch_id,
            microbatches,
            role="policy",
            requested_coefficients={
                "policy_clip": policy_clip,
                "entropy_coef": entropy_coef,
                "jsd_coef": jsd_coef,
                "jsd_beta": jsd_beta,
            },
            bound_coefficients={
                "policy_clip": self.policy_clip,
                "entropy_coef": self.entropy_coef,
                "jsd_coef": self.jsd_coef,
                "jsd_beta": self.jsd_beta,
            },
            weighted=POLICY_WINDOW_WEIGHTED,
        )

    def score_on_policy_metrics(
        self,
        batch: list[PPOTrainingEpisode],
        *,
        policy_clip: float = 0.2,
        inputs_by_episode: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> dict[str, float]:
        """Compare the pre-update HF policy with the immutable vLLM behaviour scores.

        One forward over the batch, reduced over the whole model world, returning the plain
        scalars the entrypoint reports before the update.
        """
        if not batch:
            raise ValueError("policy on-policy scoring received an empty batch")
        if not math.isfinite(policy_clip) or not 0.0 < policy_clip < 1.0:
            raise ValueError(f"policy_clip must be finite and in (0, 1), got {policy_clip}")
        prepare_deterministic_ppo_forward(self.model)
        device = model_device(self.model)
        prepared_inputs = run_rank_converged(
            lambda: tuple(episode_model_inputs(item.scored.episode, device, inputs_by_episode) for item in batch),
            "policy on-policy input preparation",
        )
        log_ratios: list[torch.Tensor] = []
        clip_masks: list[torch.Tensor] = []
        reference_behavior_log_differences: list[torch.Tensor] = []
        policy_reference_log_differences: list[torch.Tensor] = []
        log_lower_ratio = math.log(1.0 - policy_clip)
        log_upper_ratio = math.log(1.0 + policy_clip)
        # ``inference_mode`` can leave model-created cache tensors permanently
        # marked as inference-only, which poisons the following PPO backward.
        with torch.no_grad():
            for episode_index, (item, inputs) in enumerate(zip(batch, prepared_inputs, strict=True)):
                output = forward_causal_model(self.model, inputs)
                (
                    valid_log_ratio,
                    outside_clip,
                    valid_reference_behavior_difference,
                    valid_policy_reference_difference,
                ) = run_rank_converged(
                    lambda item=item, output=output: self._on_policy_episode_terms(
                        item,
                        output,
                        device=device,
                        log_lower_ratio=log_lower_ratio,
                        log_upper_ratio=log_upper_ratio,
                    ),
                    f"policy on-policy post-forward episode {episode_index}",
                )
                log_ratios.append(valid_log_ratio.double())  # [M]
                clip_masks.append(outside_clip)  # [M]
                reference_behavior_log_differences.append(valid_reference_behavior_difference.double())  # [M]
                policy_reference_log_differences.append(valid_policy_reference_difference.double())  # [M]
        log_ratio = torch.cat(log_ratios).double()  # [N]
        outside_clip = torch.cat(clip_masks)  # [N]
        reference_behavior_delta = torch.cat(reference_behavior_log_differences).double().abs()  # [N]
        policy_reference_delta = torch.cat(policy_reference_log_differences).double().abs()  # [N]
        delta = log_ratio.abs()  # [N]
        ratio = torch.exp(log_ratio)  # [N]
        multiplicative = torch.exp(delta)  # [N]
        delta_is_finite = torch.isfinite(delta).all()  # []
        ratio_is_finite = torch.isfinite(ratio).all()  # []
        multiplicative_is_finite = torch.isfinite(multiplicative).all()  # []
        reference_behavior_is_finite = torch.isfinite(reference_behavior_delta).all()  # []
        policy_reference_is_finite = torch.isfinite(policy_reference_delta).all()  # []
        validity_error: ValueError | None = None
        if (
            delta.numel() == 0
            or not bool(delta_is_finite)
            or not bool(ratio_is_finite)
            or not bool(multiplicative_is_finite)
            or not bool(reference_behavior_is_finite)
            or not bool(policy_reference_is_finite)
        ):
            validity_error = ValueError("policy on-policy statistics are empty or non-finite")
        raise_if_any_rank_failed(validity_error, "policy on-policy local statistics")
        percentiles = _global_on_policy_delta_percentiles(delta)  # [2]
        global_max_log_weight = log_ratio.max().reshape(1)  # [1]

        import torch.distributed as dist

        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(global_max_log_weight, op=dist.ReduceOp.MAX)  # [1]
        scaled_weights = torch.exp(log_ratio - global_max_log_weight)  # [N]
        total = torch.stack(
            (
                delta.sum(),
                multiplicative.sum(),
                ratio.sum(),
                scaled_weights.sum(),
                scaled_weights.square().sum(),
                outside_clip.sum().double(),
                reference_behavior_delta.sum(),
                policy_reference_delta.sum(),
                torch.tensor(float(delta.numel()), dtype=torch.float64, device=device),
            )
        )  # [9]
        maxima = torch.stack(  # [4]
            (
                delta.max(),
                ratio.max(),
                reference_behavior_delta.max(),
                policy_reference_delta.max(),
            )
        )
        minimum = ratio.min().reshape(1)  # [1]

        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(total, op=dist.ReduceOp.SUM)  # [9]
            dist.all_reduce(maxima, op=dist.ReduceOp.MAX)  # [4]
            dist.all_reduce(minimum, op=dist.ReduceOp.MIN)  # [1]
        count = float(total[8].item())
        ratio_sum = float(total[2].item())
        scaled_sum = float(total[3].item())
        importance_ess = min(1.0, scaled_sum**2 / (count * float(total[4].item())))
        metrics = {
            "behavior_logprob_delta_mean": float(total[0].item()) / count,
            "behavior_logprob_delta_max": float(maxima[0].item()),
            "behavior_logprob_multiplicative_mean": float(total[1].item()) / count,
            "behavior_logprob_delta_p95": float(percentiles[0].item()),
            "behavior_logprob_delta_p99": float(percentiles[1].item()),
            "initial_ratio_mean": ratio_sum / count,
            "initial_ratio_min": float(minimum.item()),
            "initial_ratio_max": float(maxima[1].item()),
            "initial_clip_fraction": float(total[5].item()) / count,
            "initial_importance_ess": importance_ess,
            "reference_behavior_logprob_delta_mean": float(total[6].item()) / count,
            "reference_behavior_logprob_delta_max": float(maxima[2].item()),
            "policy_reference_logprob_delta_mean": float(total[7].item()) / count,
            "policy_reference_logprob_delta_max": float(maxima[3].item()),
        }
        return validate_ppo_metrics(metrics)

    def _on_policy_episode_terms(
        self,
        item: PPOTrainingEpisode,
        output: Any,
        *,
        device: torch.device,
        log_lower_ratio: float,
        log_upper_ratio: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Postprocess one completed on-policy forward before the next aligned forward."""
        episode = item.scored.episode
        logits = extract_causal_logits(output)  # [1,P+T,V]
        action_logits = response_action_logits(
            logits,
            len(episode.prompt_token_ids),
            len(episode.action_token_ids),
        )[0]  # [T,V]
        action_ids = torch.tensor(episode.action_token_ids, dtype=torch.long, device=device)  # [T]
        current = sampled_action_logprobs(
            action_logits,
            action_ids,
            temperature=self.temperature,
        )  # [T]
        old = torch.tensor(episode.old_logprobs, dtype=torch.float32, device=device)  # [T]
        reference = torch.tensor(  # [T]
            item.scored.reference.action_logprobs,
            dtype=torch.float32,
            device=device,
        )
        mask = torch.tensor(episode.action_mask, dtype=torch.bool, device=device)  # [T]
        difference = current - old  # [T]
        valid_log_ratio = difference[mask]  # [M]
        valid_reference_behavior_difference = (reference - old)[mask]  # [M]
        valid_policy_reference_difference = (current - reference)[mask]  # [M]
        outside_clip = torch.logical_or(  # [M]
            valid_log_ratio < log_lower_ratio,
            valid_log_ratio > log_upper_ratio,
        )
        return (
            valid_log_ratio,
            outside_clip,
            valid_reference_behavior_difference,
            valid_policy_reference_difference,
        )

    def set_policy_version(self, policy_version: int) -> None:
        """Stamp one completed outer update without conflating minibatch steps."""
        if isinstance(policy_version, bool) or not isinstance(policy_version, int) or policy_version < 0:
            raise ValueError(f"policy_version must be a non-negative integer, got {policy_version!r}")
        if policy_version < self.policy_version:
            raise ValueError(f"policy_version cannot move backward from {self.policy_version} to {policy_version}")
        self.set_publication_version(policy_version)

    def iteration(self) -> int:
        """Return the outer policy version used by shared weight synchronization."""
        return self.policy_version


__all__ = ["PPOPolicyTrainer"]
