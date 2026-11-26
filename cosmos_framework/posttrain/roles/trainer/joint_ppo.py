# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Model-independent joint PPO role over the shared optimizer engine and batch lifecycle."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, replace
from typing import Any

import torch
import torch.distributed as dist

from cosmos_framework.posttrain.algorithm.joint_ppo.objective import WEIGHT_KEY, WEIGHTED, make_joint_objective
from cosmos_framework.posttrain.algorithm.joint_ppo.records import (
    STATS_KEY,
    TARGET_KEY,
    JointPPOTargetDiagnostic,
    JointPPOTargets,
    JointUpdateReceipt,
    PreparedJointBatch,
    validate_decision,
)
from cosmos_framework.posttrain.algorithm.joint_ppo.targets import fixed_episode_targets, target_diagnostic_metrics
from cosmos_framework.posttrain.algorithm.ppo.batch import PPOUpdatePlan, prepare_sequence_targets
from cosmos_framework.posttrain.algorithm.ppo.losses import normalize_advantages
from cosmos_framework.posttrain.algorithm.ppo.records import PPOStepSource, PPOTargetStep
from cosmos_framework.posttrain.algorithm.ppo.wire import canonical_digest
from cosmos_framework.posttrain.config.rl import JointPPOConfig
from cosmos_framework.posttrain.model.protocol import BatchedDecisionPPOPolicy, DecisionPPOPolicy
from cosmos_framework.posttrain.roles.trainer.ppo_runtime import (
    PPOTrainableRole,
    run_cached_ppo_minibatch,
    run_ppo_batch_update,
)
from cosmos_framework.posttrain.schema import FINAL_REWARD_KEY, Trajectory
from cosmos_framework.posttrain.utils.dist import rank_max_count, run_rank_converged
from cosmos_framework.posttrain.utils.identity import normalize_json_dict
from cosmos_framework.posttrain.utils.rank_state import gather_rank_states
from cosmos_framework.utils.quantized_optimizer_names import is_local_master_optimizer


class JointPPOTrainer(PPOTrainableRole):
    def __init__(
        self,
        model: DecisionPPOPolicy,
        recipe: JointPPOConfig,
        *,
        parallel_dims: Any = None,
        checkpoint_manager: Any = None,
        source_identity: dict | None = None,
    ) -> None:
        from cosmos_framework.posttrain.checkpoint import CheckpointManager

        self.recipe = recipe
        from cosmos_framework.posttrain.config.build import OptimizerParameterGroup, build_lr_scheduler, build_optimizer

        optimizer = build_optimizer(
            model,
            recipe.optimizer,
            parameter_groups=(
                OptimizerParameterGroup("actor", model.ppo_parameters.actor, recipe.actor_lr),
                OptimizerParameterGroup("value", model.ppo_parameters.value, recipe.value_lr),
            ),
        )
        scheduler = build_lr_scheduler(optimizer, recipe.optimizer)
        objective = make_joint_objective(
            policy_clip=recipe.policy_clip,
            value_clip=recipe.value_clip,
            dual_clip=recipe.dual_clip,
            huber_delta=recipe.huber_delta,
            loss_reduction=recipe.loss_reduction,
            microbatch=recipe.microbatch,
            identity=recipe.identity(),
        )
        objective.identity["model"] = model.ppo_identity()
        objective.identity["score_batching"] = (
            "model-owned-fixed-microbatch-v1"
            if isinstance(model, BatchedDecisionPPOPolicy)
            else "recorded-invocation-selected-row"
        )
        if source_identity is not None:
            objective.identity["model_source"] = source_identity
        objective.identity["learner_ranks"] = dist.get_world_size() if dist.is_initialized() else 1
        if checkpoint_manager is None:
            checkpoint_manager = CheckpointManager(checkpoint_format="dcp", keep_optim_last=0, keep_model_last=0)
        super().__init__(
            model,
            objective,
            optimizer,
            lr_scheduler=scheduler,
            max_grad_norm=recipe.optimizer.grad_clip_norm,
            nonfinite_update="fail_stop",
            require_gradients=True,
            checkpoint_manager=checkpoint_manager,
            transfer_dtype=model.precision.transfer,
            validation_normalizer=lambda identity: normalize_json_dict(
                identity, context="joint PPO validation identity"
            ),
        )
        self.set_publication_version(0)
        self._initialize_ppo_role("policy", parallel_dims, digest=canonical_digest)
        self._global_ids: tuple[str, ...] = ()
        self.model.ppo_padding = None
        self._frozen_model_state = self.model.ppo_frozen_state()
        # FP32-master runs snapshot only the trainable PPO parameters before each update (see
        # ``train_batch``). The owner's names are their canonical state-dict names, fixed at
        # construction, so the set is built once rather than per update.
        self._update_snapshot_names = (
            _update_snapshot_names(optimizer, self.model.net) if is_local_master_optimizer(optimizer) else None
        )
        # Pinned host buffers for that snapshot, one per name, allocated on first use and kept across
        # updates: one pinned host copy of this rank's local trainable shards. Deliberate: it replaces
        # the per-update device clone that ran out of device memory at update 20 of the GR00T run.
        self._host_snapshot_buffers: dict[str, torch.Tensor] = {}

    def iteration(self) -> int:
        return self.publication_version()

    def progress(self) -> dict:
        return {"optimizer_step": self.optimizer_step, "publication_version": self.publication_version()}

    def ppo_identity(self) -> dict:
        return self.model.ppo_identity()

    def evaluation_state(self) -> dict:
        """Small attestations of learner state around external rollout evaluation."""
        from torch.distributed.tensor import DTensor

        from cosmos_framework.posttrain.checkpoint import collect_rng_state

        def digest(value: Any) -> str:
            if isinstance(value, DTensor):
                value = value.to_local()
            if isinstance(value, dict):
                return canonical_digest({key: digest(item) for key, item in value.items()})
            return canonical_digest(value)

        # Paged optimizer state is a CPU tensor over CUDA managed memory: hashing it on the host
        # would migrate every page there. The shared helper stages it through its CUDA view into
        # pinned memory (one wait per device below); the bytes, and so the digest, are the same.
        # Other state is hashed where it is, uncopied.
        from cosmos_framework.utils.quantized_optimizer import copy_state_tensor

        paged_devices: set[torch.device] = set()
        optimizer_state = [
            {
                key: copy_state_tensor(item, ordinary=None, paged_devices=paged_devices)
                if isinstance(item, torch.Tensor)
                else item
                for key, item in state.items()
            }
            for state in self.opt.state.values()
        ]
        for device in paged_devices:
            torch.cuda.synchronize(device)

        with self._parameter_state() as parameters:
            state = {
                "optimizer_step": self.optimizer_step,
                "publication_version": self.publication_version(),
                "parameters": digest(parameters),
                "optimizer": canonical_digest([digest(state) for state in optimizer_state]),
                "rng": digest(collect_rng_state()),
                "scheduler": digest(self.lr_scheduler.state_dict() if self.lr_scheduler is not None else {}),
            }
            if is_local_master_optimizer(self.opt):
                # An FP32 master aliases its parameter, already in ``parameters``: mark it, not rehash it.
                state["optimizer_masters"] = canonical_digest(
                    [
                        "fp32-parameter-alias" if live.dtype == torch.float32 else digest(master.detach())
                        for live, master in zip(self.opt.live, self.opt.masters)
                    ]
                )
                # Persist allocated state/precision accounting with the native evaluation receipt
                # (actor log forwarding can omit INFO records): the byte/element counters and a
                # digest of the optimizer identity, not the per-parameter layouts themselves.
                # Gradient staging is transient, not evaluation state.
                state["optimizer_storage"] = {
                    key: value
                    for key, value in self.opt.state_report().items()
                    if type(value) is int and key not in ("gradient_staging_bytes", "fp32_gradient_alias_bytes")
                }
                state["optimizer_storage"]["identity_digest"] = digest(self.opt.identity)
            current = self.model.ppo_frozen_state()

            def check_frozen_model() -> None:
                if current != self._frozen_model_state:
                    raise RuntimeError("frozen model parameters or buffers changed")

            run_rank_converged(check_frozen_model, "frozen model verification")
            state.update(current)
            return state

    def prepare_ppo_targets(self, trajectories: list[Trajectory]) -> PreparedJointBatch:
        version = self.publication_version()

        def inputs() -> list[Trajectory]:
            samples = list(trajectories)
            if not samples:
                raise ValueError("joint PPO requires a nonempty actor shard")
            for sample in samples:
                self.model.validate_ppo_record(validate_decision(sample, version))
            return samples

        samples = run_rank_converged(inputs, "joint PPO source validation")
        groups = gather_rank_states(
            [(sample.id, canonical_digest(sample)) for sample in samples], context="joint PPO sample identities"
        ).values()
        identities = [entry for group in groups for entry in group]
        batch_id = canonical_digest(identities)

        def targets() -> tuple[list[Trajectory], list[JointPPOTargetDiagnostic]]:
            if len(set(key for key, _ in identities)) != len(identities):
                raise ValueError("joint PPO window contains duplicate sample identities")
            steps = []
            for sample in samples:
                step = sample.transitions[0]
                stats = step.algo_extra[STATS_KEY]
                steps.append(
                    PPOTargetStep(
                        source=PPOStepSource(
                            step.source.sample_id,
                            step.source.episode_id,
                            step.source.fragment_id,
                            step.step_idx,
                            sample.session,
                        ),
                        rewards=(step.reward[FINAL_REWARD_KEY],),
                        values=(stats.old_value, stats.bootstrap_value),
                        mask=(True,),
                        terminated=(
                            any(step.feedback.control_terminated)
                            if self.recipe.loss_reduction == "fixed_episode_mean"
                            and step.feedback.control_terminated is not None
                            else step.done and not step.truncated
                        ),
                        truncated=step.truncated,
                        window_end=step.collection_end,
                        policy_version=version,
                    )
                )
            if self.recipe.loss_reduction == "fixed_episode_mean":
                computed = fixed_episode_targets(samples, gamma=self.recipe.gamma, gae_lambda=self.recipe.gae_lambda)
            else:
                sequence = prepare_sequence_targets(steps, gamma=self.recipe.gamma, gae_lambda=self.recipe.gae_lambda)
                computed = tuple(JointPPOTargets(item.advantages[0], item.returns[0]) for item in sequence)
            raw = torch.tensor([item.advantage for item in computed], dtype=torch.float32)
            mask = torch.tensor([item.loss_mask for item in computed], dtype=torch.bool)
            advantages = normalize_advantages(
                raw,
                mask,
                correction=1,
                eps=1e-5,
                eps_inside_sqrt=False,
                statistics_dtype=torch.float32 if self.recipe.loss_reduction == "fixed_episode_mean" else torch.float64,
            )
            diagnostics = [
                JointPPOTargetDiagnostic(
                    step.source.sample_id,
                    float(step.values[0]),
                    float(step.values[1]),
                    float(step.rewards[0]),
                    target.returns,
                    float(raw_value),
                    float(advantage),
                    step.terminated,
                    step.truncated,
                    step.window_end,
                    sample.transitions[0].feedback.executed_steps,
                    target.loss_mask,
                    target.loss_weight,
                )
                for sample, step, target, raw_value, advantage in zip(
                    samples, steps, computed, raw, advantages, strict=True
                )
            ]
            prepared = [
                replace(
                    sample,
                    transitions=[
                        replace(
                            sample.transitions[0],
                            advantage=float(advantage),
                            algo_extra={
                                **sample.transitions[0].algo_extra,
                                TARGET_KEY: replace(target, advantage=float(advantage)),
                            },
                        )
                    ],
                )
                for sample, advantage, target in zip(samples, advantages, computed, strict=True)
            ]
            return prepared, diagnostics

        prepared, diagnostics = run_rank_converged(targets, "joint PPO target preparation")
        self._global_ids = tuple(key for key, _ in identities)
        try:
            run_rank_converged(lambda: self._ppo_cache.install(batch_id, prepared), "joint PPO cache install")
            self.model.ppo_padding = prepared[0].transitions[0]
            drift = (
                self._on_policy(prepared)
                if self.recipe.verify_behavior_logprobs
                else {"max_logprob_delta": None, "initial_clip_fraction": None}
            )
            global_diagnostics = [
                record
                for shard in gather_rank_states(diagnostics, context="joint PPO target diagnostics").values()
                for record in shard
            ]
            return {
                "batch_id": batch_id,
                "decisions": len(identities),
                "policy_version": version,
                "behavior_verified": self.recipe.verify_behavior_logprobs,
                **drift,
                "target_diagnostics": [
                    {
                        key: value
                        for key, value in asdict(record).items()
                        if self.recipe.loss_reduction == "fixed_episode_mean" or key not in ("loss_mask", "loss_weight")
                    }
                    for record in global_diagnostics
                ],
                "target_metrics": target_diagnostic_metrics(global_diagnostics),
            }
        except BaseException:
            self.abort_batch(batch_id)
            raise

    def _on_policy(self, samples: list[Trajectory]) -> dict:
        device = next(self.model.net.parameters()).device
        delta = torch.zeros((), device=device)
        clip = torch.zeros((), device=device)
        with torch.no_grad():
            maximum = rank_max_count(len(samples), device)
            if isinstance(self.model, BatchedDecisionPPOPolicy):
                for start in range(0, maximum, self.recipe.microbatch):
                    selected = samples[start : start + self.recipe.microbatch]
                    steps = [sample.transitions[0] for sample in selected]
                    steps += [self.model.ppo_padding] * (self.recipe.microbatch - len(steps))
                    score = self.model.score_ppo_decisions(steps)

                    def compare() -> tuple[torch.Tensor, torch.Tensor]:
                        batch_delta, batch_clip = delta, clip
                        for index, sample in enumerate(selected):
                            old = sample.transitions[0].algo_extra[STATS_KEY].old_logprob
                            logprob = score.logprobs[index : index + 1]
                            batch_delta = torch.maximum(batch_delta, (logprob - old).abs().max())
                            batch_clip = torch.maximum(
                                batch_clip, ((logprob - old).exp().sub(1).abs() > self.recipe.policy_clip).float().max()
                            )
                        return batch_delta, batch_clip

                    delta, clip = run_rank_converged(compare, "joint PPO on-policy comparison")
            else:
                for index in range(maximum):
                    step = samples[min(index, len(samples) - 1)].transitions[0]
                    score = self.model.score_ppo_decision(step)
                    if index < len(samples):
                        logprob = score.logprobs
                        difference = (logprob - step.algo_extra[STATS_KEY].old_logprob).abs().max()
                        delta = torch.maximum(delta, difference)
                        clip = torch.maximum(
                            clip,
                            (
                                (logprob - step.algo_extra[STATS_KEY].old_logprob).exp().sub(1).abs()
                                > self.recipe.policy_clip
                            )
                            .float()
                            .max(),
                        )
        if dist.is_initialized():
            dist.all_reduce(delta, op=dist.ReduceOp.MAX)
            dist.all_reduce(clip, op=dist.ReduceOp.MAX)
        if not bool(torch.isfinite(delta)) or float(delta) > self.model.ppo_logprob_tolerance or float(clip) != 0.0:
            raise ValueError(f"joint PPO on-policy mismatch: max_logprob_delta={float(delta)}, clipped={float(clip)}")
        return {"max_logprob_delta": float(delta), "initial_clip_fraction": float(clip)}

    def train_batch(self, batch_id: str) -> JointUpdateReceipt:
        from cosmos_framework.posttrain.utils.log import info0

        ids = self._ppo_cache.episode_ids(batch_id)
        plan = PPOUpdatePlan(
            1,
            self.recipe.decisions // self.recipe.global_minibatch,
            self.recipe.microbatch,
            self.recipe.seed,
            rank_seeded_torch_shuffle=True,
        )

        optimizer_metrics: list[dict] = []

        def update(microbatches: list[list[str]]) -> dict:
            metrics = run_cached_ppo_minibatch(
                self,
                self._ppo_cache,
                batch_id,
                microbatches,
                role="policy",
                requested_coefficients={},
                bound_coefficients={},
                weighted=WEIGHTED,
                weight_key=WEIGHT_KEY,
            )
            optimizer_metrics.append(dict(metrics))
            return metrics

        # Allocator receipts are opt-in diagnostics (rl.cuda_allocator_receipts). Otherwise no update
        # resets the process-wide CUDA peak counters or gathers receipts across ranks.
        master = is_local_master_optimizer(self.opt)
        allocator_receipts = self.recipe.cuda_allocator_receipts
        pre_update_allocator = _cuda_allocator_window(self.model.net, reset=True) if allocator_receipts else None
        # The pre-update snapshot only proves both groups changed and reports their update norms.
        # The default path keeps the historical device clone of the whole state dict. The opt-in
        # FP32-master path already holds masters and moments next to the model, and a device clone
        # spanning forward/backward OOMed there, so it stages the snapshot in pinned host memory
        # asynchronously instead, and only for the trainable PPO parameters: a frozen backbone
        # cannot change, so copying it to the host every update would be pure transfer cost.
        with self._parameter_state(clone=True, host=master, names=self._update_snapshot_names) as before:
            result = run_ppo_batch_update(
                role="policy",
                batch_id=batch_id,
                plan=plan,
                global_episode_ids=self._global_ids,
                cached_episode_ids=ids,
                parallel_dims=self._parallel_dims,
                optimizer_step=lambda: self.optimizer_step,
                update=update,
            )
            info0(f"Joint PPO applied optimizer_step={self.optimizer_step}")
            with self._parameter_state() as after:
                # Squared storage deltas per parameter group, summed across shards before the root;
                # ranks need not own equal tensor sizes.
                changes = torch.zeros(2, device=next(self.model.net.parameters()).device, dtype=torch.float64)
                for name in before:
                    group = int(name in self.model.ppo_parameters.value)
                    # A host snapshot comes back one tensor at a time (pinned, stream-ordered, no
                    # host sync); the reduction stays on device until the single decision below.
                    previous = before[name].to(after[name].device, non_blocking=True)
                    changes[group] += (after[name].float() - previous.float()).square().double().sum()
                    del previous
                if dist.is_initialized():
                    dist.all_reduce(changes)
                if not bool(torch.isfinite(changes).all()) or not bool((changes > 0).all()):
                    raise RuntimeError(
                        f"joint PPO did not update both parameter groups: squared changes={changes.tolist()}"
                    )
                receipt = {
                    "optimizer_step": self.optimizer_step,
                    "metrics": result.metrics,
                    "optimizer_metrics": optimizer_metrics,
                    "learning_rates": {group["name"]: float(group["lr"]) for group in self.opt.param_groups},
                    "parameter_update_l2": dict(zip(("actor", "value"), changes.sqrt().tolist(), strict=True)),
                }
                if allocator_receipts:
                    receipt["cuda_allocator"] = _gather_allocator_receipt(
                        pre_update_allocator,
                        _cuda_allocator_window(self.model.net, reset=False),
                        {"optimizer_backend_step_ms": _backend_step_milliseconds(self.opt)},
                    )
                return receipt

    @contextmanager
    def _parameter_state(
        self, *, clone: bool = False, host: bool = False, names: frozenset[str] | None = None
    ) -> Iterator[dict[str, torch.Tensor]]:
        """Read public local shards at an update boundary, never forward-time FSDP parameters.

        Only a cloned pre-update state may span later forwards. ``host=True`` stages that
        clone in pinned host memory with stream-ordered asynchronous copies, for runs whose
        device cannot also hold a full copy of every local shard through forward/backward;
        the copy is enqueued before any later kernel that changes the parameters, so it needs
        no host synchronization. Its host buffers persist across updates, one per name, allocated
        on first use; each update's copies are stream-ordered after the previous update's reads
        of them. ``names`` keeps only those state-dict entries (e.g. the
        trainable parameters). Both raw state-dict bindings and detached views are released
        on exit, even with a retained traceback.
        Failed readers release completed traceback locals while retaining the exception chain.
        """
        from traceback import clear_frames

        from torch.distributed.checkpoint.state_dict import get_model_state_dict
        from torch.distributed.tensor import DTensor

        state: dict[str, torch.Tensor] = {}
        local: dict[str, torch.Tensor] = {}
        value = None
        try:
            state = get_model_state_dict(self.model.net)
            for name, value in state.items():
                if names is not None and name not in names:
                    continue
                value = (value.to_local() if isinstance(value, DTensor) else value).detach()
                if not clone:
                    local[name] = value
                elif host:
                    buffers = self._host_snapshot_buffers
                    buffer = buffers.get(name)
                    if buffer is None or buffer.shape != value.shape or buffer.dtype != value.dtype:
                        buffer = buffers[name] = torch.empty(
                            value.shape, dtype=value.dtype, device="cpu", pin_memory=value.is_cuda
                        )
                    local[name] = buffer.copy_(value, non_blocking=value.is_cuda)
                else:
                    local[name] = value.clone()
            # The cloned branch must not carry borrowed bindings through the update.
            value = None
            state.clear()
            yield local
        except BaseException as error:
            # Hashing/state-dict helpers can retain a tensor argument in an ended
            # frame. Keep exception locations and chains, but release their locals.
            pending, seen = [error], set()
            while pending:
                current = pending.pop()
                if id(current) in seen:
                    continue
                seen.add(id(current))
                clear_frames(current.__traceback__)
                pending.extend(cause for cause in (current.__cause__, current.__context__) if cause is not None)
            raise
        finally:
            value = None
            local.clear()
            state.clear()

    def clear_batch(self, batch_id: str) -> None:
        self._ppo_cache.clear(batch_id)
        self._global_ids, self.model.ppo_padding = (), None

    def abort_batch(self, batch_id: str) -> None:
        self.abort_update_window()
        self._ppo_cache.abort(batch_id)
        self._global_ids, self.model.ppo_padding = (), None


def _update_snapshot_names(optimizer: Any, net: torch.nn.Module) -> frozenset[str]:
    """The master owner's names, checked once against ``net``'s state-dict keys.

    The pre-update snapshot selects state-dict entries by these names. A name that is not a key
    would silently leave its parameter out, and the update check would then blame the update.
    """
    from torch.distributed.checkpoint.state_dict import get_model_state_dict

    names = frozenset(optimizer.names)
    missing = sorted(names - set(get_model_state_dict(net)))
    if missing:
        raise ValueError(
            f"FP32-master optimizer parameter names {missing} are not state-dict keys of the PPO model's net; "
            "the optimizer must be built over that same net"
        )
    return names


_ALLOCATOR_FIELDS = (
    "allocated_bytes.all.current",
    "reserved_bytes.all.current",
    "allocated_bytes.all.peak",
    "reserved_bytes.all.peak",
    "inactive_split_bytes.all.current",
    "inactive_split_bytes.all.peak",
    "num_alloc_retries",
    "num_ooms",
)


def _cuda_allocator_window(net: torch.nn.Module, *, reset: bool) -> dict[str, int] | None:
    """This rank's caching-allocator counters; optionally start a new peak window.

    ``inactive_split_bytes`` is cached memory inside partially used segments, i.e. the
    fragmentation that a later large request cannot use. Managed (paged) optimizer
    state is outside the caching allocator and therefore excluded.
    """
    parameter = next(net.parameters(), None)
    if parameter is None or not parameter.is_cuda:
        return None
    device = parameter.device
    stats = torch.cuda.memory_stats(device)
    receipt = {key: int(stats.get(key, 0)) for key in _ALLOCATOR_FIELDS}
    receipt["device_free_bytes"], receipt["device_total_bytes"] = (int(v) for v in torch.cuda.mem_get_info(device))
    if reset:
        torch.cuda.reset_peak_memory_stats(device)
    return receipt


def _backend_step_milliseconds(optimizer: Any) -> float | None:
    """CUDA-event time of the last FP32-master backend step on this rank, if recorded."""
    events = getattr(optimizer, "last_backend_step_events", None)
    if events is None:
        return None
    events[1].synchronize()
    return float(events[0].elapsed_time(events[1]))


def _gather_allocator_receipt(before: dict | None, after: dict | None, extra: dict | None = None) -> list[dict] | None:
    """Rank-ordered allocator receipts, identical on every rank so update results still agree.

    ``before`` peaks cover the phases since the previous update (collection, weight sync,
    target preparation); ``after`` peaks cover only this update's forward/backward/step.
    """
    local = None if after is None else {"before_update": before, "after_update": after, **(extra or {})}
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return None if local is None else [local]
    gathered = gather_rank_states(local, context="joint PPO allocator receipt")
    values = [gathered[rank] for rank in sorted(gathered)]
    return None if all(value is None for value in values) else values
