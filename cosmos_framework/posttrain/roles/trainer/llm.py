# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Objective-driven trainer for causal language models."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING, Any, Literal

import torch

from cosmos_framework.posttrain.algorithm.primitives.objective import (
    OBJECTIVE_WEIGHT_SUM,
    Objective,
    Reduction,
    normalization_scale,
)
from cosmos_framework.posttrain.algorithm.primitives.objective import backward as reduce_backward
from cosmos_framework.posttrain.checkpoint import CheckpointManager, rank0_snapshot_is_immutable_cpu
from cosmos_framework.posttrain.checkpoint.consolidated import fill_sharded_state_from_consolidated
from cosmos_framework.posttrain.control.validation import ValidationTrigger
from cosmos_framework.posttrain.control.validation.rollout import normalize_rollout_validation_identity
from cosmos_framework.posttrain.model.protocol import (
    STRICT_FORWARD_NOISE_TOLERANCE,
    ForwardNoisePolicy,
    ForwardNoiseTolerance,
    ForwardReplayPolicy,
)
from cosmos_framework.posttrain.roles.rollout.validation import validation_semantics_equal
from cosmos_framework.posttrain.roles.trainer.base import BaseTrainer
from cosmos_framework.posttrain.roles.trainer.finite import (
    all_ranks_gradients_finite,
    all_ranks_tensor_finite,
    all_ranks_tensor_true,
)
from cosmos_framework.posttrain.roles.trainer.validation.publication import PublicationJournal
from cosmos_framework.posttrain.roles.trainer.validation.state import ValidationState
from cosmos_framework.posttrain.schema import Trajectory
from cosmos_framework.posttrain.schema.checkpoint import CheckpointComponentPlan
from cosmos_framework.posttrain.schema.metrics import SUMMED_TRAIN_COUNT_KEYS
from cosmos_framework.posttrain.schema.routing import FORWARD_REPLAY_BATCH_KEY
from cosmos_framework.posttrain.utils.dist import (
    collective_device,
    is_dtensor_sharded,
    raise_if_any_rank_failed,
    rank_any,
    rank_or_0,
    rank_sum_count,
    run_rank_converged,
)
from cosmos_framework.posttrain.utils.host_memory import log_host_memory
from cosmos_framework.posttrain.utils.identity import json_identity_equal, normalize_json_dict
from cosmos_framework.posttrain.utils.log import info0
from cosmos_framework.posttrain.utils.profile import ProfileRecord, live_cuda_tensor_summary
from cosmos_framework.posttrain.utils.residency import HostOffload, make_offload

if TYPE_CHECKING:
    from cosmos_framework.posttrain.checkpoint.store import CheckpointStore
    from cosmos_framework.posttrain.comm.weight_transfer import WeightPayload, WeightWitnessSpec


def concat_batches(batches: list) -> list:
    """Concatenate fan-in trajectory batches without changing producer order."""
    return [t for b in batches for t in b]


def probe_drift_counts(recorded: torch.Tensor, probe: torch.Tensor, per_token_nats: float) -> tuple[int, int, float]:
    """How much of a chunk's π_old the probe forward failed to reproduce, as COUNTS.

    Returned as counts rather than a local fraction because this rank's tensors hold only its DP
    shard of the probe chunk, which can be a single short trajectory (measured: 11 tokens, so one
    flipped token was worth 9 points of a 5-point gate). The caller sums the counts over the DP mesh
    and gates on the fraction over ALL compared tokens. The worst move is returned for the message.

    Args:
        recorded: per-token π_old from the eval() ahead pass, ``[N]``.
        probe: the same tokens re-scored by the probe forward at unchanged weights, ``[N]``.
        per_token_nats: a token counts as moved past this many nats.

    Returns:
        ``(n_moved, n_compared, worst_move)``.
    """
    delta = (recorded.float() - probe.float()).abs()  # [N]
    return int((delta > per_token_nats).sum()), int(delta.numel()), float(delta.max())


# What needs a deterministic forward and what to do instead; the screen below prefixes its refusal
# with one of these. The KL text is passed by `entrypoints/grpo.py::build_trainer` when it builds the
# frozen reference, which scores under eval() while the policy trains under train() at ANY n_mini.
AHEAD_PASS_DETERMINISM_REQUIREMENT = (
    "the π_old ahead pass (rl.mini_batch_per_replica < batch_per_replica, i.e. n_mini > 1) "
    "requires a DETERMINISTIC forward: π_old is recorded under eval() and scored under "
    "train(), so live dropout makes the ratio noise rather than policy drift and the "
    "epsilon-clip fires on nothing; run with n_mini == 1 instead, where π_old comes from the same "
    "forward and any stochasticity cancels."
)
KL_REFERENCE_DETERMINISM_REQUIREMENT = (
    "the frozen KL reference (rl.kl_beta > 0 with rl.ref.build_ref=true) requires a DETERMINISTIC "
    "forward: the reference scores under eval() and the policy under train(), so live dropout makes "
    "the log-ratio logπ_ref − logπ_θ a dropout mask rather than policy drift — non-zero at step 0 "
    "even at identical weights — and the anchor pulls against that noise every step, at any n_mini; "
    "set rl.kl_beta=0 instead."
)


def raise_if_forward_is_stochastic(net: Any, *, requirement: str = AHEAD_PASS_DETERMINISM_REQUIREMENT) -> None:
    """Refuse a two-forward comparison on a model whose forward is not deterministic given its weights.

    Used by the π_old ahead pass (`record_old_logprobs`, the default ``requirement``) and by
    `entrypoints/grpo.py::build_trainer` before it builds the frozen KL reference
    (``KL_REFERENCE_DETERMINISM_REQUIREMENT``): both score one forward under eval() and another
    under train() and treat the difference as policy change.

    WHY. The ahead pass records π_old in `eval()` and training scores `cur` in `train()` — TWO
    forwards. The ratio `exp(cur - π_old)` is only the intended importance ratio if those agree when
    the weights have not moved. With dropout live they do not: eval disables it, train samples a
    mask, so the very first mini-batch gets a non-unit ratio built from noise, the clip fires on
    nothing, and the surrogate is corrupted. At `n_mini == 1` this cannot happen — π_old is
    `cur.detach()` from the SAME forward, so any stochasticity cancels exactly. The ahead pass is
    what introduces the requirement.

    cosmos-rl has the identical eval/train split (`set_model_eval` for OLD_LOGP_COMPUTE,
    `set_model_train` for TRAIN) and no such check, so mirroring it is not evidence of safety. That
    much IS verifiable from its trainer source. What is NOT — and what an earlier revision of this
    docstring asserted anyway — is any claim about every model cosmos-rl supports shipping zero
    dropout. Only one model was actually checked: `Qwen/Qwen2.5-3B-Instruct` reports
    `attention_dropout: 0.0`. Treat the rest as unknown.

    THIS IS A SCREEN, NOT A PROOF. It rejects the stochasticity we know how to name; it cannot
    certify that an arbitrary model's forward is deterministic given its weights. `LLMTrainer` is
    model-agnostic, so stochastic depth, custom sampling layers, or a functional dropout under some
    other config key would pass it and silently turn the first ratio into mode noise. If that becomes
    a real exposure the right answer is an explicit per-model capability that certifies eval/train
    logprob equivalence — not a longer list of attribute names here.

    Checks BOTH forms, because catching one is worse than catching neither — it reads as coverage:
      - `nn.Dropout` modules with p > 0 (GPT-2: resid/embd/attn all default 0.1). Anything realised
        as a module is caught here, whatever it is named.
      - a NARROW set of config attributes applied FUNCTIONALLY, with no module for the scan above to
        find — Qwen2/Llama-family `F.dropout(..., training=self.training)`.

    The config list is curated rather than a `*dropout*` glob on purpose: a glob flags
    `summary_first_dropout`, which GPT2Config carries for the classification head and a causal LM
    never executes, so a fully deterministic model would be refused. A guard that blocks valid
    configs gets disabled, and then it protects nothing.

    Args:
        net: the policy module (`HFModel.net` or a raw module).
        requirement: which mechanism needs the deterministic forward, why, and the alternative;
            prefixes the refusal.

    Raises:
        ValueError: the forward is stochastic, naming what to set to 0.0.
    """
    import torch.nn as nn

    active: list[str] = []
    for name, mod in net.named_modules():
        if isinstance(mod, nn.Dropout) and float(getattr(mod, "p", 0.0)) > 0.0:
            active.append(f"{name or '<root>'} ({type(mod).__name__}, p={mod.p})")
    # Applied via F.dropout in the decoder forward, so they have no module to find above.
    _FUNCTIONAL_DROPOUT_ATTRS = ("attention_dropout", "hidden_dropout", "activation_dropout", "dropout")
    cfg = getattr(net, "config", None)
    for attr in _FUNCTIONAL_DROPOUT_ATTRS:
        val = getattr(cfg, attr, 0.0) if cfg is not None else 0.0
        if isinstance(val, (int, float)) and val > 0:
            active.append(f"config.{attr}={val}")
    if active:
        raise ValueError(
            f"{requirement} Active stochasticity: {', '.join(active[:6])}"
            f"{' ...' if len(active) > 6 else ''}. Set these to 0.0."
        )


# Name the ahead-pass call sites and their tests were written against.
_raise_if_forward_is_stochastic = raise_if_forward_is_stochastic


class AuxiliaryModelResidency:
    """Trainer-owned registry of auxiliary networks whose device residency follows the policy's.

    The trainer is the engine: it decides when GPU state leaves the card (`offload`) and comes back
    (`onload`). Some algorithms need a second network on the trainer ranks — GRPO's frozen KL
    reference is one — and that network must leave the card with the policy under colocation or its
    policy-sized footprint stays resident while the rollout engine holds the card. WHICH network,
    and why, is the algorithm's business; the objective consumes the scores it produces and never
    hands the trainer a module. So the entry that builds such a network registers it here
    (`LLMTrainer(auxiliary_models=...)`), and `offload` moves the policy plus every registered
    module, in registration order. Nothing here is checkpointed or published: the registered
    networks are not part of the policy.
    """

    __slots__ = ("_modules", "_labels")

    def __init__(self, models: Iterable[Any] = ()) -> None:
        self._modules: list[torch.nn.Module] = []
        self._labels: list[str] = []
        for model in models:
            self.register(model)

    def register(self, model: Any, *, label: str = "auxiliary model") -> torch.nn.Module:
        """Register a policy adapter (with `.net`) or raw module; returns the module that will move.

        Raises:
            TypeError: `model` exposes no `torch.nn.Module`.
            ValueError: the same module is already registered (moving it twice is a wiring bug).
        """
        net = getattr(model, "net", model)
        if not isinstance(net, torch.nn.Module):
            raise TypeError(f"{label} must wrap a torch.nn.Module, got {type(net).__name__}")
        if any(net is registered for registered in self._modules):
            raise ValueError(f"{label} is already registered for residency")
        self._modules.append(net)
        self._labels.append(str(label))
        return net

    def modules(self) -> list[torch.nn.Module]:
        """The registered modules, in registration order."""
        return list(self._modules)

    def labels(self) -> list[str]:
        return list(self._labels)

    def __len__(self) -> int:
        return len(self._modules)


class LLMTrainer(BaseTrainer):
    """Model-agnostic causal-LM trainer body: `train(batch)` = objective → backward → step."""

    validation_state: ValidationState
    validation_publication: PublicationJournal
    skip_zero_grad: bool

    def __init__(
        self,
        model: Any,
        objective: Objective,
        optimizer: torch.optim.Optimizer,
        *,
        grad_accum: int = 1,
        n_mini: int = 1,
        callbacks: Any = None,
        checkpoint_format: str = "consolidated",
        save_async: bool = False,
        keep_optim_last: int | None = None,
        keep_model_last: int = 0,
        upload: dict[str, Any] | None = None,
        checkpoint_store: Mapping[str, Any] | CheckpointStore | None = None,
        checkpoint_manager: CheckpointManager | None = None,
        max_grad_norm: float = 0.0,
        lr_scheduler: Any = None,
        force_finite_grads: bool = True,
        nonfinite_update: str = "skip",
        require_gradients: bool = False,
        transfer_dtype: str | None = None,
        skip_zero_grad: bool = False,
        forward_replay: str | None = None,
        auxiliary_models: Iterable[Any] = (),
        validation_normalizer: Callable[[dict], dict] = normalize_rollout_validation_identity,
        validation_state: ValidationState | None = None,
        objective_evaluator: Callable[[Any, Objective, list[Trajectory]], dict[str, Any]] | None = None,
    ) -> None:
        from cosmos_framework.posttrain.callbacks.base import CallbackList

        self.model = model
        self.objective = objective
        # Networks other than the policy that live on the trainer ranks and must share the policy's
        # device residency (`offload`/`onload`) — e.g. the frozen KL reference GRPO's entry builds.
        # Injected by the entry: the trainer is the engine and never learns about them through the
        # objective, which only consumes what they compute (see `AuxiliaryModelResidency`).
        self.residency = AuxiliaryModelResidency(auxiliary_models)
        # Forward-routing replay (`model/protocol.py::ForwardReplayPolicy`). A non-None value names
        # the source of the recorded routing every policy batch carries (assembly passes
        # `rl.routing_replay`); None is natural dispatch. The model pins from the batch it prepares,
        # so the trainer never pins: after every forward it checks that the pinned state matches this
        # request (`_replay_pinned`) and picks the probe tolerance from it. The capability check is
        # here so a trainer asked to replay on a model that cannot fails at construction, not mid-run.
        self.forward_replay = None if forward_replay is None else str(forward_replay)
        if self.forward_replay is not None and not isinstance(model, ForwardReplayPolicy):
            raise TypeError(
                f"forward_replay requires a model implementing ForwardReplayPolicy, got {type(model).__name__}; "
                "rl.routing_replay applies to sparse-MoE policies only"
            )
        identity = dict(objective.identity)
        if self.forward_replay is not None:
            # Pinned versus natural dispatch changes every gradient: the source and WHAT is pinned
            # (block count, top-k; model-owned) ride in the exact-resume identity next to the loss
            # terms. Absent when natural, so every earlier checkpoint stays resumable.
            identity["forward_replay"] = {"source": self.forward_replay, **model.replay_identity()}
        self.objective_identity = normalize_json_dict(identity, context="LLM objective exact-resume identity")
        super().__init__(
            optimizer,
            grad_accum=grad_accum,
            force_finite_grads=force_finite_grads,
            nonfinite_update=nonfinite_update,
            require_gradients=require_gradients,
            checkpoint_format=checkpoint_format,
            save_async=save_async,
            keep_optim_last=keep_optim_last,
            keep_model_last=keep_model_last,
            upload=upload,
            checkpoint_store=checkpoint_store,
            checkpoint_manager=checkpoint_manager,
        )
        # BOOKKEEPING ONLY -- the trainer never uses this; it is checkpointed so a resume can prove
        # its batching shape is unchanged. The driver converts the durable OPTIMIZER-step clock into
        # its COLLECTED-BATCH clock by dividing by n_mini, so resuming under a different n_mini would
        # land on a silently wrong batch (smaller restarts too far along, larger re-runs). The count
        # alone can't reveal that, so the shape travels with it (checked in `load_state_dict`).
        self.n_mini = max(1, int(n_mini))
        self._forward_determinism_checked = False  # π_old ahead pass precondition, checked once
        # Consumed optimizer attempts since the last LR-schedule tick. The schedule advances on the
        # COLLECTED-BATCH clock (once per n_mini attempts), matching cosmos-rl, so a batch's n_mini
        # sequential updates all run at one LR. Zero-gradient skips still consume an attempt.
        self._sched_accum = 0
        self.callbacks = callbacks if isinstance(callbacks, CallbackList) else CallbackList(callbacks)
        # Gradient clipping is DTensor-aware; finite checks reject the whole window.
        self.max_grad_norm = float(max_grad_norm)
        # This changes optimizer semantics because AdamW normally steps on materialized all-zero
        # gradients. Keep it opt-in; AlpaGym enables it to match the cosmos-rl baseline.
        self.skip_zero_grad = bool(skip_zero_grad)
        # The optional schedule advances once per COLLECTED BATCH (n_mini optimizer steps); see the
        # accumulate-then-step block and `self._sched_accum`.
        self.lr_scheduler = lr_scheduler
        # Wire dtype for producer-driven weight sync: `comm/weight_transfer.py` reads
        # `getattr(producer, "transfer_dtype", None)` off this trainer to declare the manifest dtype
        # for floating parameters. Fed from `resolve_precision(...).transfer` at construction.
        self.transfer_dtype = transfer_dtype
        # Rollout/reward or recipe-bound objective validation keeps its transaction marker beside the model version.
        self.validation_state = (
            validation_state
            if validation_state is not None
            else ValidationState(
                normalizer=validation_normalizer,
                identity_equal=validation_semantics_equal,
            )
        )
        self.objective_evaluator: Callable[[Any, Objective, list[Trajectory]], dict[str, Any]] | None = (
            objective_evaluator
        )
        self.validation_publication = PublicationJournal(self.validation_state)
        # The colocated hand-off is deliberately split in two. Update-only state moves first while
        # policy shards remain on GPU as the direct IPC weight-sync source; the policy moves only
        # after publication. This changes ordering, not transfer volume.
        self._update_offload = HostOffload()
        self._offload = make_offload()
        self._parked_update_mib = 0.0
        # The optimizer iteration is the durable, idempotent weight-stream version.

    def _net(self) -> torch.nn.Module:
        """Return the raw model or a model body's trainable ``.net``."""
        return getattr(self.model, "net", self.model)

    def _prepare_update_model(self) -> None:
        """Apply the engine's model-mode policy immediately before an update forward."""
        self._net().train()

    def _validate_post_optimizer_step(self) -> None:
        """Validate an optimizer mutation before clocks and callbacks commit it."""

    def _advance_scheduler_after_applied_update(self) -> None:
        """Advance the default collected-batch scheduler after a validated mutation."""
        self._advance_batch_clock()

    def _scheduler_clock_name(self) -> Literal["batch", "optimizer_step"]:
        """Name the durable clock used by the configured learning-rate scheduler."""
        return "batch"

    def training_metric_weight_map(self) -> dict[str, str]:
        """Expose the objective's public training metric aggregation contract."""
        return self.objective.metric_weight_map("train")

    def training_metric_max_keys(self) -> frozenset[str]:
        """Expose the public training metrics the objective declares as MAX-reduced."""
        metric_max_keys = getattr(self.objective, "metric_max_keys", None)
        return frozenset(metric_max_keys("train")) if callable(metric_max_keys) else frozenset()

    def needs_old_logprobs(self) -> bool:
        """Whether THIS trainer's objective can consume a frozen π_old — the capability half of the
        ahead-pass gate.

        The config half (`GRPOConfig.needs_cached_old_logprobs`) says the batch drives several
        optimizer steps and the loss is not reading rollout logprobs. It is NECESSARY, not
        sufficient, and it is INHERITED: `OPDConfig` subclasses `GRPOConfig`, and OPD's DIRECT
        objective forms no importance ratio at all, so it carries no cache and the pass would abort
        the run on its first chunk. Gating on the objective itself is the only thing that cannot be
        wrong by inheritance.

        Returns:
            True when the objective carries an `OldLogprobCache` to record into.
        """
        return getattr(self.objective, "old_logp_cache", None) is not None

    def begin_replay_window(self) -> dict[str, Any]:
        """Close the π_old record window and rewind it for training. Called ONCE per collected batch,
        on every policy cell, after the ahead pass and before the first `train`.

        Why a dedicated call rather than a lazy transition inside `train`. The transition is a
        per-BATCH event, but `train` is a per-CHUNK call, so doing it there forces a rule that is
        wrong on one side or the other: validate only when the cache is in "record" and a restarted
        rank (cache "off") silently skips the collective its peers entered; validate on every call
        and the second chunk sees "replay" and aborts the batch. Both of those shipped. A separate
        once-per-batch call has neither problem — it runs on all cells, exactly once, at a point
        where every rank agrees what the state should be.

        Returns:
            `{"old_logp/window": <records frozen>}`.

        Raises:
            RuntimeError: this cell holds no recorded window when one was expected, or holds MORE
                records than the driver will walk chunks. A window holding FEWER is legal — an empty
                DP shard records nothing for that chunk and skips it identically in both passes.
                Converged across ranks first, so every rank raises together instead of one exiting
                into its peers' collectives.
        """
        cache = getattr(self.objective, "old_logp_cache", None)
        if cache is None or self.n_mini <= 1:
            return {"old_logp/window": 0}
        error: str | None = None
        if cache.mode != "record":
            error = (
                f"expected a recorded π_old window (n_mini={self.n_mini}) but the cache is "
                f"'{cache.mode}': this cell missed the ahead pass, so it would train against "
                "cur.detach() while its peers score against a frozen reference"
            )
        elif cache.recorded > self.n_mini * self.grad_accum:
            # UPPER BOUND ONLY, never equality. A rank whose DP shard is empty for some chunk returns
            # from the objective before reaching π_old, recording nothing for it — `select_chunk`
            # permits empty shards and the cache is built for them (both passes skip the same chunks
            # deterministically, so order-keyed replay still lines up). Demanding exactly
            # n_mini * n_micro records therefore fails every distributed run whose chunks are smaller
            # than the DP mesh. The count is rank-LOCAL; only this bound is rank-invariant, and it
            # still catches a driver that walked the ahead loop more times than it will train.
            error = (
                f"π_old window holds {cache.recorded} records but the driver walks only "
                f"{self.n_mini * self.grad_accum} chunks (n_mini={self.n_mini}, "
                f"n_micro={self.grad_accum}): the ahead pass ran more times than training will"
            )
        else:
            cache.begin_replay()
        raise_if_any_rank_failed(error, "π_old replay-window validation")
        return {"old_logp/window": cache.recorded}

    def compute_old_logprobs(self, batch: list[Trajectory]) -> dict[str, Any]:
        """Ahead pass: record π_old for ONE chunk, at the policy that collected the batch.

        Runs the REAL objective under `eval()` + `no_grad()` rather than a second forward written
        here, so the recorded π_old cannot drift from the one training scores — reimplementing the
        shaping/forward is exactly how a parallel path goes stale. The loss it computes is
        discarded; only `OldLogprobCache` retains anything.

        The driver calls this for every `(mini, micro)` BEFORE the first `train()`, and only when
        `rl.n_mini > 1` — at `n_mini == 1` π_old is exactly `cur.detach()` and this pass would be
        pure cost. See `algorithm/grpo.py::OldLogprobCache`.

        COLLECTIVE PARITY: this adds `n_mini * n_micro` forwards, and the objective's own
        all-reduces, on EVERY rank. `n_mini`/`n_micro` are config-uniform and the driver loop is
        identical everywhere, so the added collectives stay symmetric — the invariant the objective's
        empty-shard probe path exists to protect.

        Args:
            batch: this chunk's trajectories, materialized by the same `select_chunk` selection
                `train` receives, so chunk N here is chunk N there.

        Returns:
            `{"old_logp/recorded": <chunks recorded so far this window>}`.

        Raises:
            RuntimeError: the objective carries no cache (an objective with no ratio, so nothing
                to freeze) — the driver should not have called this.
        """
        cache = getattr(self.objective, "old_logp_cache", None)
        if cache is None:
            raise RuntimeError(
                f"{type(self.objective).__name__} carries no old_logp_cache, so π_old cannot be "
                "frozen; do not run the ahead pass for this objective."
            )
        net = self._net()
        first = not self._forward_determinism_checked
        if first:
            # Cheap NAMED screen: catches the common cases with a clear message before we spend a
            # forward. It fails OPEN by construction (see its docstring), so it is not the gate.
            _raise_if_forward_is_stochastic(net)
        cache.begin_record()
        was_training = net.training
        net.eval()
        try:
            with torch.no_grad():
                self.objective(self.model, batch)
            # The probe re-runs the objective on the SAME batch, so the model pins it exactly as it
            # pinned this forward (and as it will pin the train forward of this chunk): the probe
            # scores the forward configuration the run trains with, by construction.
            pinned = self._replay_pinned(active=bool(batch))
            if first:
                # EMPIRICAL fail-closed check, once per run, on the first chunk only (a one-shot
                # screen, not a per-batch monitor). The named screen above cannot certify an
                # arbitrary model — LLMTrainer accepts any — so verify the invariant itself: a
                # train()-mode forward must reproduce the eval()-mode π_old at UNCHANGED weights. Any
                # stochasticity breaks that, named or not (dropout, stochastic depth, sampling
                # layers, a functional dropout under some config key nobody listed).
                #
                # Probe capture is a scratch slot, never appended to the window, so the recorded
                # chunk is untouched. Costs ONE extra forward per run, and it runs on every rank, so
                # collective parity holds.
                recorded = cache.last_recorded
                cache.begin_probe()
                net.train()
                with torch.no_grad():
                    self.objective(self.model, batch)
                probe = cache.probe
                cache.mode = "record"  # restore the window; probe never entered it
                self._gate_determinism_probe(recorded, probe, replay_pinned=pinned)
        finally:
            net.train(was_training)
        return {"old_logp/recorded": cache.recorded}

    def _replay_pinned(self, *, active: bool) -> bool:
        """Mesh-wide verdict on the forwards just prepared, and a refusal when it contradicts
        `forward_replay`.

        The state is the model's (it pins from the batch it prepares); the trainer only holds the
        run to what `rl.routing_replay` promised, in both directions: a replay identity must not
        train through natural dispatch, and a natural identity must not train through a pin. Each
        rank is in one of three states: EMPTY (``active`` is False: its DP shard held no trajectory,
        so the objective ran only the placeholder forward for collective parity and prepared no batch
        of its own), active-natural, or active-pinned. The two active states are reduced separately,
        `any` over ranks, because one OR cannot enforce the identity: a pinned rank would mask an
        active rank that ran natural. Empty ranks decide nothing. Every rank sees the same reduced
        flags, so all raise together, before backward.

        Args:
            active: this rank's chunk held at least one trajectory.

        Returns:
            Whether the active forwards were pinned; False when every rank was empty.

        Raises:
            RuntimeError: an active rank's pinned state disagrees with `forward_replay`.
        """
        if not isinstance(self.model, ForwardReplayPolicy):
            return False
        local_pinned = active and bool(self.model.replay_pinned())
        any_pinned = rank_any(local_pinned)
        any_natural = rank_any(active and not local_pinned)
        if any_pinned and self.forward_replay is None:
            raise RuntimeError(
                f"the policy batch carried recorded routing ({FORWARD_REPLAY_BATCH_KEY}) and the model pinned "
                "its dispatch to it, but rl.routing_replay is off: the gradients would not be the natural-dispatch "
                "gradients this run's identity promises"
            )
        if any_natural and self.forward_replay is not None:
            raise RuntimeError(
                f"rl.routing_replay={self.forward_replay!r} but an active rank's policy batch carried no recorded "
                f"routing ({FORWARD_REPLAY_BATCH_KEY}), so its forward ran through NATURAL dispatch under a replay "
                "identity (this rank: "
                f"{'pinned' if local_pinned else 'natural' if active else 'empty shard'})"
            )
        return any_pinned

    def _replay_metrics(self) -> dict[str, int]:
        """Rank-local replay telemetry for the forward just run, under a replay identity only: the
        token rows this rank dispatched through the pin (summed mesh-wide as a count) and the blocks
        per row; see `ForwardReplayPolicy.replay_pinned_extent` for what a row is."""
        if self.forward_replay is None:
            return {}
        tokens, blocks = self.model.replay_pinned_extent()
        return {"train/routing_replay_pinned_tokens": int(tokens), "train/routing_replay_pinned_blocks": int(blocks)}

    def _forward_noise_tolerance(self, *, replay_pinned: bool) -> ForwardNoiseTolerance:
        """The probe gate: declared by the model when it can (`ForwardNoisePolicy`), else strict."""
        if isinstance(self.model, ForwardNoisePolicy):
            return self.model.forward_noise_tolerance(replay_pinned=replay_pinned)
        return STRICT_FORWARD_NOISE_TOLERANCE

    def _gate_determinism_probe(self, recorded: Any, probe: Any, *, replay_pinned: bool) -> None:
        """Compare the probe against the recorded π_old under the model-declared tolerance.

        Gates on the FRACTION of tokens whose logprob moved past the per-token bound, summed over
        the DP mesh: a max over N draws from forward noise grows with N, so a max-abs gate fails a
        long batch on a model a short batch passed, and one rank's shard can be a handful of tokens.
        The strict tolerance (moved fraction 0) reproduces the max-abs gate exactly.

        Raises:
            RuntimeError: converged across ranks, so no rank exits into its peers' collectives.
        """
        tolerance = self._forward_noise_tolerance(replay_pinned=replay_pinned)
        # VACUOUS COMPARISON IS NOT A PASS. An empty shard returns before `old_for` is ever
        # reached, so this rank has nothing to compare. Marking the model "checked" on that
        # would retire the guard for the whole run on the strength of a comparison that never
        # happened -- the rank that DOES hold data would then never be probed.
        compared = recorded is not None and probe is not None and bool(recorded.numel())
        moved, total, worst = probe_drift_counts(recorded, probe, tolerance.per_token_nats) if compared else (0, 0, 0.0)
        # Reduced UNCONDITIONALLY (a rank that compared nothing contributes 0/0) so the collectives
        # stay symmetric across ranks.
        device = collective_device() if torch.distributed.is_initialized() else torch.device("cpu")
        moved_mesh = rank_sum_count(moved, device)
        total_mesh = rank_sum_count(total, device)
        fraction = moved_mesh / total_mesh if total_mesh else 0.0
        info0(
            f"π_old determinism probe: {100.0 * fraction:.3f}% of {total_mesh} tokens moved by more than "
            f"{tolerance.per_token_nats} nats (gate {100.0 * tolerance.moved_fraction:.1f}%); worst move on this "
            f"rank {worst:.4f}; forward routing {'PINNED' if replay_pinned else 'NATURAL'}"
        )
        local_error: str | None = None
        if total_mesh and fraction > tolerance.moved_fraction:
            # `fraction` is mesh-wide, so every rank enters here together; only rank 0 carries the
            # diagnosis because `raise_if_any_rank_failed` concatenates every rank's text.
            if rank_or_0() == 0:
                local_error = (
                    f"a train()-mode recompute moved {100.0 * fraction:.2f}% of the logprobs (over "
                    f"{total_mesh} tokens) by more than {tolerance.per_token_nats} nats at UNCHANGED weights "
                    f"(worst move on this rank {worst:.4f}, gate {100.0 * tolerance.moved_fraction:.1f}%), so "
                    "the recorded π_old does not describe the policy training scores: the importance ratio "
                    "would be forward noise and the epsilon-clip would fire on nothing. Disable the model's "
                    "stochastic layers, or run with rl.mini_batch_per_replica unset (n_mini == 1), where "
                    "π_old comes from the same forward."
                )
                if replay_pinned:
                    local_error += (
                        " The probe forward was PINNED to the recorded routing, so this is not routing churn."
                    )
                elif isinstance(self.model, ForwardReplayPolicy):
                    local_error += (
                        " This policy can replay its forward routing (ForwardReplayPolicy) and ran with it "
                        "NATURAL: suspect expert-routing churn before dropout -- a nondeterministic expert "
                        "kernel jitter can change experts for near-tied router scores across repeated forwards. "
                        "rl.routing_replay pins the dispatch."
                    )
            else:
                local_error = (
                    f"mesh-wide verdict, see rank 0 (this rank: {moved}/{total} tokens moved, worst {worst:.4f})"
                )
        # CONVERGE THE VERDICT ACROSS RANKS: raising locally would abort one rank while its peers
        # walk into the next forward's FSDP all-gathers and block forever, a hang instead of a report.
        raise_if_any_rank_failed(local_error, "π_old ahead-pass determinism probe")
        # Only retire the guard once SOME rank actually compared something. All-reduce MAX so
        # every rank agrees; a run where no rank held data re-probes on the next batch.
        self._forward_determinism_checked = rank_any(compared)

    def train(self, batch: list[Trajectory]) -> dict[str, Any]:
        """Objective → backward → (deferred normalize) → clip → step, one call per micro-batch.

        Args:
            batch: this chunk's trajectories (may be empty — see `algorithm/grpo.py`'s
                empty-shard path, which still reports the objective weight sum).

        Returns:
            Metrics for this call: `"discarded"` for a chunk consumed by the poisoned-window
            countdown (§5.2), else the usual `"accumulating"` / `"applied"` / `"skipped"` outcome
            metrics from `BaseTrainer._outcome_metrics` / `_skip_nonfinite`.

        Failure semantics:
            A typed partial-update failure poisons future calls until checkpoint restore. Earlier
            raw exceptions are not a retry-safe receipt: attempt, accumulation, or replay-cache
            state may already have advanced before the optimizer mutation boundary.
        """
        self._raise_if_partial_update_poisoned()
        if self._skip_remaining > 0:
            # This chunk belongs to a mini-batch already abandoned by a non-finite skip. Consuming
            # it into the fresh window would step the optimizer on a MIXTURE of two mini-batches
            # under a divisor matching neither (§5.2).
            self._skip_remaining -= 1
            # The pi_old window is keyed by append order, and this return happens BEFORE the
            # objective runs -- so `old_for` never fires for this chunk while the ahead pass already
            # recorded one for it. Without advancing the read pointer every later chunk would replay
            # a reference belonging to different trajectories, silently (the cache under-consumes, so
            # the exhaustion guard cannot fire) and for the rest of the collected batch.
            cache = getattr(self.objective, "old_logp_cache", None)
            if cache is not None:
                cache.skip_chunk()
            return self._outcome_metrics("discarded")
        net = self._net()
        self._prepare_update_model()
        # The π_old record -> replay transition is NOT done here. It happens ONCE per collected batch
        # in `begin_replay_window()`, which the driver calls on EVERY cell after the ahead pass. Doing
        # it here meant the first train() flipped the mode and the second saw "replay" and treated it
        # as a missed ahead pass — aborting every n_mini > 1 batch on its second chunk.
        # By this point the window is already frozen and rewound; train() only consumes it via the
        # objective's `old_for`. No ahead pass ⇒ cache mode stays "off" ⇒ the objective falls back to
        # `cur.detach()`, which is exact at n_mini == 1 and is the pre-change behaviour everywhere else.
        self._begin_update_attempt()
        loss, metrics = self.objective(self.model, batch)
        self._replay_pinned(active=bool(batch))
        metrics.update(self._replay_metrics())
        # Consensus precedes FSDP backward so a bad rank cannot strand its peers.
        if not all_ranks_tensor_finite(loss):
            return self._skip_nonfinite(metrics, "nonfinite_loss")
        # The reduction owns normalize+backprop (TOKEN_SUM: as-is at grad_accum==1, else the
        # chunk SUM recovered via the objective's own divisor; MEAN: /grad_accum).
        denom = metrics.get(OBJECTIVE_WEIGHT_SUM)
        reduce_backward(loss, self.objective.reduction, self.grad_accum, denom)
        self._accum += 1
        self._denom_accum += float(denom or 0.0)
        if self._accum == self.grad_accum:
            # Reject the whole window rather than corrupt optimizer moments.
            params = [parameter for parameter in net.parameters() if parameter.grad is not None]
            if self.grad_accum > 1 and self.objective.reduction is Reduction.TOKEN_SUM:
                # ONE division by the window's divisor: Σ_j DPmean(n_j) == DPmean(Σ_j n_j) (the mean
                # is linear over the chunk index), so no extra collective. Only an exactly empty
                # window substitutes one; positive fractional DP means remain unchanged (§3.5).
                scale = normalization_scale(self._denom_accum)
                for parameter in params:
                    parameter.grad.mul_(scale)
            self._denom_at_last_step = self._denom_accum  # test seam; also useful telemetry
            self._denom_accum = 0.0
            if self.force_finite_grads and not all_ranks_gradients_finite(
                params, require_gradients=self.require_gradients
            ):
                return self._skip_nonfinite(metrics, "nonfinite_gradient")
            zero_gradient = False
            if self.max_grad_norm > 0 or self.skip_zero_grad:
                max_norm = self.max_grad_norm if self.max_grad_norm > 0 else float("inf")
                total_norm = torch.nn.utils.clip_grad_norm_(params, max_norm)  # []
                if not all_ranks_tensor_finite(total_norm):
                    return self._skip_nonfinite(metrics, "nonfinite_gradient")
                to_local = getattr(total_norm, "to_local", None)
                local_norm = to_local() if callable(to_local) else total_norm  # []
                detached_norm = local_norm.detach()  # []
                metrics["train/grad_norm"] = float(detached_norm)
                if self.max_grad_norm > 0:
                    # The clip record, from the norm the clip measured, by ONE rule:
                    # ``train/grad_clip_coef`` = ``min(1, max_grad_norm / (norm + 1e-6))``, the factor
                    # ``clip_grad_norm_`` applied to every gradient, and ``train/grad_clipped`` = 1.0 exactly when that
                    # factor is below 1.0. So a norm inside the 1e-6 band below ``max_grad_norm`` counts as clipped, as
                    # its gradients were scaled.
                    clip_coef = min(1.0, self.max_grad_norm / (float(detached_norm) + 1e-6))
                    metrics["train/grad_clip_coef"] = clip_coef
                    metrics["train/grad_clipped"] = 1.0 if clip_coef < 1.0 else 0.0
                if self.skip_zero_grad:
                    zero_gradient = all_ranks_tensor_true(detached_norm == 0)  # []
            update_learning_rate = float(self.opt.param_groups[0]["lr"])
            # A zero gradient is NO SIGNAL, not a signal of zero: stepping on it still applies
            # weight decay and still advances Adam's moments, so the weights move on a batch that
            # said nothing. GRPO makes this common rather than exotic -- a group whose rewards tie
            # has an advantage of exactly 0 for every member, and with a group of 6 near-identical
            # episodes that is most of them. Measured against the cosmos-rl baseline (which skips
            # the same way): stepping regardless drove the ratio from ~0.93 after a step's 66
            # updates to 0.03, with clip_fraction 0.8 against its 0.2.
            if zero_gradient:
                self._accum = 0
                self._denom_accum = 0.0
                self._zero_grad_skips_total += 1
                self.opt.zero_grad(set_to_none=True)
                self._advance_batch_clock()
                metrics = self._reduce_metrics(metrics)
                metrics["train/learning_rate"] = update_learning_rate
                metrics.update(self._outcome_metrics("skipped"))
                return metrics
            self._run_partial_update_stage(stage="optimizer_step", action=self.opt.step)
            self._run_partial_update_stage(
                stage="post_optimizer_step",
                action=self._validate_post_optimizer_step,
            )
            # The default hook keeps the COLLECTED-BATCH clock used by every existing LLM path.
            self._run_partial_update_stage(
                stage="scheduler",
                action=self._advance_scheduler_after_applied_update,
            )
            metrics = self._run_partial_update_stage(
                stage="metric_reduction",
                action=lambda: self._reduce_metrics(metrics),
            )  # global-batch mean under data-parallel FSDP
            self._commit_update()
            metrics["train/learning_rate"] = update_learning_rate
            metrics.update(self._outcome_metrics("applied"))
            self._run_partial_update_stage(
                stage="callback",
                action=lambda: self.callbacks.on_train_step(self, metrics),
            )
        else:
            metrics = self._reduce_metrics(metrics)
            metrics.update(self._outcome_metrics("accumulating"))
        return metrics

    def _advance_batch_clock(self) -> None:
        """Advance the scheduler after one consumed mini-batch attempt."""
        self._sched_accum += 1
        if self._sched_accum >= self.n_mini:
            self._sched_accum = 0
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

    def _reduce_metrics(self, metrics: dict[str, Any]) -> dict[str, Any]:
        """Reduce public objective metrics without depending on algorithm-private names."""
        import torch.distributed as dist

        def _prepare() -> tuple[
            dict[str, Any],
            dict[str, tuple[str, float]],
            tuple[str, ...],
            list[int],
            tuple[str, ...],
        ]:
            public_metrics = self.objective.adapt_metrics(metrics, "train")
            weighted_mean_by = self.training_metric_weight_map()
            # Maxima have ONE declaration point, the objective adapter's `max_reduced` (GRPO's
            # `train/kl_max`, a float, and `train/response_len_max`, an int); they ride the one MAX
            # collective below, packed in float64 so an integer extremum comes back an int.
            declared_max_keys = set(self.training_metric_max_keys())
            max_keys = tuple(key for key in sorted(declared_max_keys) if key in public_metrics)
            for key in max_keys:
                value = public_metrics[key]
                if isinstance(value, torch.Tensor):
                    if value.dtype == torch.bool or value.is_complex():
                        raise TypeError(f"max-reduced metric {key!r} must be a real numeric tensor, got {value!r}")
                elif isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise TypeError(f"max-reduced metric {key!r} must be numeric, got {value!r}")
                if key in weighted_mean_by:
                    raise ValueError(f"metric {key!r} cannot be both max-reduced and weighted")
            if any(not isinstance(key, str) or not key for key in weighted_mean_by):
                raise ValueError("weighted public metric keys must be non-empty strings")
            if any(not isinstance(key, str) or not key for key in weighted_mean_by.values()):
                raise ValueError("public metric weight keys must be non-empty strings")

            def _weight_value(metric_key: str, weight_key: str) -> float:
                if weight_key not in public_metrics:
                    raise ValueError(f"weighted metric {metric_key!r} requires public weight {weight_key!r}")
                raw_weight = public_metrics[weight_key]
                if isinstance(raw_weight, torch.Tensor):
                    if raw_weight.numel() != 1:
                        raise TypeError(f"metric weight {weight_key!r} must be a numeric scalar, got {raw_weight!r}")
                    raw_weight = raw_weight.detach().item()
                if isinstance(raw_weight, bool) or not isinstance(raw_weight, (int, float)):
                    raise TypeError(f"metric weight {weight_key!r} must be a numeric scalar, got {raw_weight!r}")
                weight = float(raw_weight)
                if not math.isfinite(weight) or weight < 0.0:
                    raise ValueError(f"metric weight {weight_key!r} must be finite and non-negative, got {weight!r}")
                return weight

            def _metric_is_zero(value: int | float | torch.Tensor) -> bool:
                if isinstance(value, torch.Tensor):
                    nonzero = torch.count_nonzero(value.detach())  # []
                    return bool(nonzero.item() == 0)
                return float(value) == 0.0

            active_weights: dict[str, tuple[str, float]] = {}
            for metric_key, weight_key in weighted_mean_by.items():
                if metric_key not in public_metrics:
                    continue
                value = public_metrics[metric_key]
                if isinstance(value, torch.Tensor):
                    if value.dtype == torch.bool or value.is_complex():
                        raise TypeError(f"weighted metric {metric_key!r} must be a real numeric tensor, got {value!r}")
                elif isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise TypeError(f"weighted metric {metric_key!r} must be numeric, got {value!r}")
                weight = _weight_value(metric_key, weight_key)
                if weight == 0.0 and not _metric_is_zero(value):
                    raise ValueError(f"weighted metric {metric_key!r} must be zero when {weight_key!r} is zero")
                active_weights[metric_key] = (weight_key, weight)

            def _integer_scalars(keys: tuple[str, ...], kind: str) -> list[int]:
                values: list[int] = []
                for key in keys:
                    value = public_metrics[key]
                    if isinstance(value, torch.Tensor) and value.numel() == 1:
                        value = value.detach().item()
                    if isinstance(value, bool) or not isinstance(value, int):
                        raise TypeError(f"objective {kind} metric {key!r} must be an integer scalar, got {value!r}")
                    values.append(value)
                return values

            count_keys = tuple(key for key in SUMMED_TRAIN_COUNT_KEYS if key in public_metrics)
            count_values = _integer_scalars(count_keys, "count")
            return public_metrics, active_weights, count_keys, count_values, max_keys

        public_metrics, active_weights, count_keys, count_values, max_keys = run_rank_converged(
            _prepare,
            "objective public metric reduction preparation",
        )
        if not (dist.is_initialized() and dist.get_world_size() > 1):
            return public_metrics
        world_size = dist.get_world_size()
        device = collective_device()
        out: dict[str, Any] = dict(public_metrics)  # pass-through keys (non-numeric labels etc.) kept as-is
        if count_values:
            counts = torch.tensor(count_values, dtype=torch.int64, device=device)  # [C]
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
            for index, key in enumerate(count_keys):
                out[key] = int(counts[index].item())

        def _flatten(v: Any, *, dtype: torch.dtype = torch.float32) -> torch.Tensor | None:
            if isinstance(v, torch.Tensor):
                return v.detach().reshape(-1).to(device=device, dtype=dtype)  # [N_metric]
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return torch.tensor([float(v)], dtype=dtype, device=device)  # [1]
            return None

        if max_keys:
            # Declared maxima ride their own MAX collective: a per-rank maximum summed or averaged
            # would report a number no token ever had (a mean of per-rank maxima understates the
            # step's longest completion). Every rank carries the same key set (the objective metric
            # contract is config-invariant), so the pack order is aligned. Packed in float64 so the
            # integer extrema (token counts) reduce exactly and come back as ints -- the same
            # exactness the int64 count pack gives the sums -- while float maxima keep their value.
            max_flats: list[torch.Tensor] = []
            for k in max_keys:
                flat = _flatten(public_metrics[k], dtype=torch.float64)
                if flat is None:  # unreachable: _prepare rejected non-numeric maxima on every rank
                    raise TypeError(f"max-reduced metric {k!r} must be numeric, got {public_metrics[k]!r}")
                max_flats.append(flat)
            packed_max = torch.cat(max_flats)  # [N_max]
            dist.all_reduce(packed_max, op=dist.ReduceOp.MAX)
            for k, chunk in zip(max_keys, packed_max.split([flat.numel() for flat in max_flats]), strict=True):
                v = public_metrics[k]
                if isinstance(v, torch.Tensor):
                    out[k] = chunk.reshape(v.shape).to(v.dtype)
                elif isinstance(v, int):
                    out[k] = int(chunk.item())
                else:
                    out[k] = float(chunk.item())

        flats: list[torch.Tensor] = []  # each numeric flattened to 1-D, concatenated into one tensor
        specs: list[tuple[str, tuple[int, ...], bool]] = []
        for k in sorted(public_metrics):
            if k in count_keys or k in max_keys:
                continue
            v = public_metrics[k]
            flat = _flatten(v)
            if flat is None:
                continue
            if k in active_weights:
                flat = flat * active_weights[k][1]  # [N_metric]
            flats.append(flat)
            specs.append((k, tuple(v.shape) if isinstance(v, torch.Tensor) else (), not isinstance(v, torch.Tensor)))
        if not flats:
            return out
        packed = torch.cat(flats)  # [N_packed]
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        chunks = packed.split([flat.numel() for flat in flats])  # each [N_metric]
        reduced = {key: chunk for (key, _shape, _is_scalar), chunk in zip(specs, chunks, strict=True)}
        for k, shape, is_scalar in specs:
            chunk = reduced[k]
            if k in active_weights:
                weight_key, _local_weight = active_weights[k]
                if weight_key in count_keys:
                    total_weight = float(out[weight_key])
                else:
                    weight_chunk = reduced.get(weight_key)
                    if weight_chunk is None or weight_chunk.numel() != 1:
                        raise TypeError(f"metric weight {weight_key!r} must be a reduced numeric scalar")
                    total_weight = float(weight_chunk.item())
                chunk = chunk * (0.0 if total_weight == 0.0 else 1.0 / total_weight)  # [N_metric]
            else:
                chunk = chunk / world_size  # [N_metric]
            out[k] = float(chunk.item()) if is_scalar else chunk.reshape(shape)  # [*metric] when tensor
        return out

    def prepare_weight_sync(self) -> float:
        """Park update-only GPU state while retaining policy weights for direct weight sync.

        The optimizer, frozen reference and other auxiliary models are not weight-sync sources.
        Moving them before vLLM remaps its weights removes the post-update peak without changing
        the bytes transferred over a full hand-off cycle.

        Returns:
            MiB newly moved off the device.
        """
        if self._offload.offloaded:
            raise RuntimeError("prepare_weight_sync requires the policy to be resident; call onload() first")
        if self._update_offload.offloaded:
            return 0.0
        if self._accum != 0:
            raise RuntimeError(
                f"prepare_weight_sync requires a completed accumulation window, got _accum={self._accum}"
            )
        profile = ProfileRecord("trainer_prepare_weight_sync")
        # A completed window's gradients are dead and otherwise keep per-layer reduce-scatter
        # buffers alive. The next window starts by materializing fresh gradients.
        self.opt.zero_grad(set_to_none=True)
        moved = self._update_offload.offload(self.residency.modules(), optimizer=self.opt)
        self._parked_update_mib = moved
        profile.checkpoint("offload_update_state")
        live = live_cuda_tensor_summary()
        profile.emit(**({"live_after": live} if live is not None else {}))
        return moved

    def offload(self) -> float:
        """Release the complete trainer after publication so rollout can use the card.

        Returns:
            MiB of total trainer state parked on the host by this hand-off.
        """
        if self._offload.offloaded:
            return 0.0
        profile = ProfileRecord("trainer_offload")
        if self._accum != 0:
            # NFT yields the shared card between accumulation rows. Preserve the historical
            # mid-window contract by moving policy, live gradients, auxiliaries and optimizer
            # together; selective pre-sync parking remains restricted to a committed boundary.
            moved = self._offload.offload([self._net(), *self.residency.modules()], optimizer=self.opt)
            profile.checkpoint("offload_open_accumulation")
            live = live_cuda_tensor_summary()
            profile.emit(**({"live_after": live} if live is not None else {}))
            return moved
        self.prepare_weight_sync()
        policy_mib = self._offload.offload([self._net()])
        moved = self._parked_update_mib + policy_mib
        profile.checkpoint("offload_policy")
        live = live_cuda_tensor_summary()
        profile.emit(**({"live_after": live} if live is not None else {}))
        return moved

    def onload(self) -> float:
        """Restore policy first, then its update-only state. Returns total MiB moved."""
        policy_mib = self._offload.onload()
        update_mib = self._update_offload.onload()
        self._parked_update_mib = 0.0
        return policy_mib + update_mib

    def push_weights(self, target=None) -> None:
        """Stream buckets to the routed rollout after every policy rank enters the same gather."""
        # Shared-model edges already expose current weights and transfer no bytes.
        be = getattr(self, "_weight_backend", None)
        if be is None:
            return None
        from cosmos_framework.posttrain.comm.transfer.movers import SharedMover

        if isinstance(be.mover, SharedMover):
            return None
        # The training iteration is the immutable stream version, including on retry.
        from cosmos_framework.posttrain.comm.weight_transfer import push_stream_to_peer

        push_stream_to_peer(self, target, be)
        return None

    def weight_push_timing(self) -> dict[str, Any] | None:
        """Return this rank's last producer-side stream timing for opt-in scale diagnostics."""
        import socket
        from dataclasses import asdict

        import torch.distributed as dist

        backend = getattr(self, "_weight_backend", None)
        timing = None if backend is None else backend.last_push_timing
        if timing is None:
            return None
        from cosmos_framework.posttrain.comm.transfer.flat import flat_nbytes_for_manifest

        plan = backend.plan()
        return {
            **asdict(timing),
            "rank": int(dist.get_rank()) if dist.is_initialized() else 0,
            "host": socket.gethostname(),
            "bucket_meta": tuple(
                {
                    "staged_bytes": flat_nbytes_for_manifest(bucket),
                    "names": len(bucket),
                    "first": bucket[0][0],
                    "last": bucket[-1][0],
                }
                for bucket in plan
            ),
        }

    def shared_weight_identity(self) -> str | None:
        """Return an identity token only for byte-free shared-model edges."""
        backend = getattr(self, "_weight_backend", None)
        if backend is not None:
            from cosmos_framework.posttrain.comm.transfer.movers import SharedMover

            if not isinstance(backend.mover, SharedMover):
                return None
        from cosmos_framework.posttrain.comm.weight_transfer import shared_model_identity

        return shared_model_identity(self._net())

    # --- streaming weight sync (per-bucket); additive, see comm.weight_transfer ---
    def set_bucket_plan(self, plan) -> None:
        be = getattr(self, "_weight_backend", None)
        if be is not None:
            be.set_bucket_plan(plan)

    def release_weight_buffer(self) -> None:
        """Release transport staging state after every bucket has been consumed."""
        be = getattr(self, "_weight_backend", None)
        if be is not None:
            be.release_staged()

    def bucket_count(self) -> int:
        return getattr(self, "_weight_backend").bucket_count()

    def get_bucket_plan(self, bucket_bytes: int | None = None) -> list:
        """Build and cache the deterministic global-shape bucket plan."""
        be = getattr(self, "_weight_backend", None)
        if be is None:
            raise ValueError("no weight backend bound (bind with weight_sync=True before get_bucket_plan)")
        # The backend cap also sizes fixed transport buffers such as NIXL's staging allocation.
        return be.ensure_plan(self.model, bucket_bytes)

    def get_weight_destination_names(self) -> tuple[str, ...] | None:
        """Return mapper-proven live rollout destinations for pre-mutation coverage."""
        be = getattr(self, "_weight_backend", None)
        if be is None:
            raise ValueError("no weight backend bound (bind with weight_sync=True before reading destinations)")
        be.ensure_plan(self.model)
        return be.destination_names()

    def get_weight_witness_spec(self) -> "WeightWitnessSpec | None":
        """Return the mapper's optional inline packed/TP witness metadata."""
        from cosmos_framework.posttrain.comm.weight_transfer import WeightWitnessSpec

        be = getattr(self, "_weight_backend", None)
        if be is None:
            raise ValueError("no weight backend bound (bind with weight_sync=True before reading witness metadata)")
        spec = be.weight_witness_spec(self.model)
        if spec is not None and not isinstance(spec, WeightWitnessSpec):
            raise TypeError(f"weight witness metadata must be WeightWitnessSpec or None, got {spec!r}")
        return spec

    def push_bucket(self, i: int) -> "WeightPayload | None":
        """Stage one version-stamped bucket; staging owns the required full-tensor collective."""
        be = getattr(self, "_weight_backend", None)
        if be is None:
            return None
        if not be.has_plan():  # lazy plan (peer-pull path needs no driver Phase-0)
            self.get_bucket_plan()
        return be.stage_bucket(self.model, i, version=self.publication_version())

    def evaluate_objective(self, batch: list[Trajectory]) -> dict[str, Any]:
        """Evaluate one held-out batch through the recipe-bound, state-preserving evaluator."""
        self._require_checkpoint_boundary()
        if self.objective_evaluator is None:
            raise RuntimeError("no held-out objective evaluator was configured")
        metrics = self.objective_evaluator(self.model, self.objective, batch)
        return self.objective.adapt_metrics(metrics, "val")

    def configure_validation(self, identity: dict[str, Any]) -> None:
        """Bind the live validation identity and restore a compatible pending transaction."""
        restored_extra = self.validation_state.configure(
            identity,
            resumed_from_checkpoint=self._resumed_from_checkpoint,
            iteration=self._iter,
            extra_fields_by_version={
                1: frozenset(),
                2: frozenset({"publication"}),
            },
        )
        if restored_extra is not None:
            self.validation_publication.restore(
                restored_extra.get("publication"),
                iteration=self._iter,
            )

    def _require_validation_enabled(self) -> None:
        self.validation_state.require_enabled("model validation is not enabled/configured on this trainer")

    def pending_validation_step(self) -> int | None:
        """Model step whose held-out rollout/reward pass must finish before training continues."""
        self._require_validation_enabled()
        return self.validation_state.progress.pending_step

    def pending_validation_metrics(self) -> dict[str, int | float | bool | str]:
        """Canonical train frame retained for an interrupted validation boundary."""
        self._require_validation_enabled()
        return dict(self.validation_state.progress.pending_metrics)

    def pending_validation_reasons(self) -> frozenset[ValidationTrigger]:
        """Original trigger reasons retained for an interrupted validation boundary."""
        self._require_validation_enabled()
        return self.validation_state.progress.pending_reasons

    def pending_validation_publication_step(self) -> int | None:
        """Completed model-validation step whose checkpoint publication must be retried."""
        return self.validation_publication.pending_step

    def mark_validation_pending(
        self,
        model_step: int,
        trigger_reasons: frozenset[ValidationTrigger],
        metrics: dict[str, int | float | bool | str] | None = None,
    ) -> None:
        """Mark validation pending before its first data read or an emergency save."""
        self._require_validation_enabled()
        self._require_checkpoint_boundary()
        if model_step != self._iter:
            raise ValueError(f"validation model_step={model_step} does not match trainer iteration {self._iter}")
        self.validation_publication.assert_can_begin()
        self.validation_state.begin(model_step, trigger_reasons, metrics=metrics)

    def complete_validation(self, model_step: int) -> None:
        """Move a successful pass into the recoverable checkpoint-publication phase."""
        self._require_validation_enabled()
        self.validation_publication.complete(model_step)

    def complete_ephemeral_validation(self, model_step: int) -> None:
        """Complete an in-memory initial phase without entering publication."""
        self._require_validation_enabled()
        publication_step = self.validation_publication.pending_step
        if publication_step is not None:
            raise RuntimeError(f"ephemeral validation unexpectedly left step {publication_step} awaiting publication")
        if model_step != self._iter:
            raise ValueError(f"validation model_step={model_step} does not match trainer iteration {self._iter}")
        self.validation_state.progress.complete(model_step)

    def _require_checkpoint_boundary(self) -> None:
        """Reject checkpoints while a poisoned mini-batch still has chunks to discard."""
        super()._require_checkpoint_boundary()
        from cosmos_framework.posttrain.utils.dist import raise_if_any_rank_failed

        error = (
            RuntimeError(
                f"checkpoint save requires the poisoned mini-batch to finish; skip_remaining={self._skip_remaining}"
            )
            if self._skip_remaining
            else None
        )
        raise_if_any_rank_failed(error, "LLM trainer checkpoint boundary")

    # Checkpoint extraction is engine-specific; layout and publication are shared.
    def _trainer_state(self) -> dict[str, Any]:
        from cosmos_framework.posttrain.checkpoint import collect_rng_state

        rng_state = collect_rng_state()

        def _state_with_rng() -> dict[str, Any]:
            state: dict[str, Any] = {
                **self._update_state_dict(),
                "rng": rng_state,
                "objective_identity": dict(self.objective_identity),
            }
            if self.validation_state.restored_state is not None and not self.validation_state.configured:
                raise RuntimeError("configure validation before saving a restored LLMTrainer checkpoint")
            if self.validation_state.configured:
                self.validation_publication.validate_checkpoint_state()
                state["validation"] = self.validation_state.state_dict(
                    version=2,
                    extra={"publication": self.validation_publication.checkpoint_record()},
                )
            if self.lr_scheduler is not None:
                # Persist scheduler progress so resume does not replay warmup.
                state["lr_scheduler"] = self.lr_scheduler.state_dict()
                # STAMP the clock the schedule was stepped on. A batch-clock schedule advances on the
                # COLLECTED-BATCH clock; a checkpoint written before this change (or its absence on
                # load) means the legacy per-optimizer-step clock, which resume converts. Engines
                # whose schedule rides the optimizer-step clock stamp that name instead. See
                # `_restore_trainer_state_local`.
                state["lr_scheduler_clock"] = self._scheduler_clock_name()
            return state

        return run_rank_converged(_state_with_rng, "LLM trainer state serialization")

    def save(
        self,
        path: str,
        publish: bool = True,
        update_latest: bool = True,
        components: Iterable[str] | CheckpointComponentPlan | None = None,
        data_state: dict[str, Any] | None = None,
    ) -> None:
        """Save selected logical state in consolidated or DCP form.

        NEITHER format undoes a producer's sharding pad, and that is correct: what is written is
        read back by resume into an identically padded model, so trimming here would break the very
        round trip the checkpoint exists for. The weight-sync export DOES trim
        (`comm/weight_transfer.py::WeightMapper`) because the rollout never padded. Two consequences
        worth naming: a checkpoint is not interchangeable with the base model's shapes, and a
        cross-world-size DCP resume does NOT hold for a padded parameter -- the pad width is the
        shard count, so a different world wants a different padding."""
        self._require_checkpoint_boundary()
        # State-dict extraction gathers over the device mesh, so offloaded shards must come back first.
        self.onload()
        component_plan = self.checkpoint_manager.component_plan(components)
        model_selected = component_plan.includes("model")
        optimizer_selected = component_plan.includes("optimizer")
        training_state_selected = component_plan.includes("training_state")
        needs_net = model_selected or optimizer_selected
        net = run_rank_converged(self._net, "LLM checkpoint network extraction") if needs_net else None

        if self.checkpoint_format == "dcp":
            from torch.distributed.checkpoint.state_dict import get_model_state_dict, get_optimizer_state_dict

            model_state = (
                run_rank_converged(  # [*sharded parameter] values
                    lambda: get_model_state_dict(net),
                    "LLM DCP model state extraction",
                )
                if model_selected
                else None
            )
            with self._preserved_pristine_optimizer() as guard:
                optim_state = (
                    run_rank_converged(  # [*sharded optimizer-state] values
                        lambda: get_optimizer_state_dict(net, self.opt),
                        "LLM DCP optimizer state extraction",
                    )
                    if optimizer_selected
                    else None
                )
            # Write what a never-stepped optimizer actually holds. The extraction had to run to
            # shape `param_groups` the way DCP expects, but the state it produced is an artifact
            # of the probe step, not state this run ever computed.
            optim_state = guard.sanitized(optim_state)
            tail_components = {"trainer": self._trainer_state(), "data": data_state} if training_state_selected else {}
            live_tail_state = (
                run_rank_converged(
                    lambda: (
                        self.opt.param_groups,
                        self.lr_scheduler.state_dict() if self.lr_scheduler is not None else {},
                    ),
                    "LLM DCP async-tail live state preparation",
                )
                if training_state_selected
                else ()
            )
            location = self._save_dcp_components(
                path,
                {"model": model_state, "optim": optim_state},
                tail_components,
                live_tail_state,
                component_plan=component_plan,
                publish=publish,
                update_latest=update_latest,
            )
        else:
            sharded = net is not None and is_dtensor_sharded(net)
            sharded_opt_sd: dict[str, Any] | None = None
            if sharded:
                from torch.distributed.checkpoint.state_dict import (
                    StateDictOptions,
                    get_model_state_dict,
                    get_optimizer_state_dict,
                )

                # `full_state_dict=True, cpu_offload=True` lands the WHOLE gathered model in rank 0's
                # host memory. Record the host footprint first: on a colocated node the trainer's
                # offload backup and the rollout engine already hold most of it.
                if model_selected:
                    self._log_host_memory("gathering the consolidated model on rank 0", component_plan)
                opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
                net_sd = get_model_state_dict(net, options=opts) if model_selected else None  # [*parameter] values
                # The optimizer is NEVER gathered: two fp32 moments per parameter is about 4x the bf16
                # model (roughly 244 GB for a 30 B-parameter policy), and rank 0 would hold it next to
                # the gathered model. With that gather a one-node colocated run reached 857 GB of a
                # 900 GB node at its emergency save and Ray killed the actors, so a terminated run left
                # no resumable checkpoint. The optimizer goes out as per-rank DCP shards instead; the
                # model artifact and its consumers are unchanged (see `load` for the mixed layout).
                # Same materializing extraction as the DCP branch, so the same rollback applies. The
                # unsharded branch below reads `self.opt.state_dict()`, which does not initialize and
                # already reports an empty state for a never-stepped optimizer.
                with self._preserved_pristine_optimizer() as guard:
                    sharded_opt_sd = (
                        run_rank_converged(  # [*sharded optimizer-state] values
                            lambda: get_optimizer_state_dict(net, self.opt),
                            "LLM consolidated checkpoint sharded optimizer state extraction",
                        )
                        if optimizer_selected
                        else None
                    )
                sharded_opt_sd = guard.sanitized(sharded_opt_sd)
                opt_sd = None
            else:
                net_sd = net.state_dict() if model_selected and net is not None else None  # [*parameter] values
                opt_sd = self.opt.state_dict() if optimizer_selected else None  # [*optimizer-state] values
            components = {
                "model": {"net": net_sd} if net_sd is not None else None,
                "optim": opt_sd,
                "trainer": self._trainer_state() if training_state_selected else None,
                "data": data_state if training_state_selected else None,
            }
            async_safe_snapshot = bool(
                sharded
                and rank0_snapshot_is_immutable_cpu(
                    components,
                    (
                        tuple(net.parameters()) if model_selected else (),  # [*parameter] values
                        tuple(net.buffers()) if model_selected else (),  # [*buffer] values
                        self.opt.state if optimizer_selected else {},
                        self.opt.param_groups if optimizer_selected else (),
                        (
                            self.lr_scheduler.state_dict()
                            if training_state_selected and self.lr_scheduler is not None
                            else {}
                        ),
                    ),
                )
            )
            location = self.checkpoint_manager.save_consolidated(
                path,
                components,
                # Plain state_dict tensors may alias live parameters.  The FSDP full-state path above
                # is an immutable CPU-offloaded snapshot and is the only safe asynchronous source.
                async_safe_snapshot=async_safe_snapshot,
                component_plan=component_plan,
                publish=publish,
                update_latest=update_latest,
                dcp_components={"optim": sharded_opt_sd} if sharded_opt_sd is not None else None,
            )
        if publish:
            self.callbacks.on_checkpoint(self, location)

    @staticmethod
    def _log_host_memory(event: str, component_plan: CheckpointComponentPlan) -> None:
        """Log this rank's host-memory reading before a host-side gather (every rank).

        The reading is the shared one (`utils/host_memory.py`: RSS, the job cgroup's usage against its
        limit, the node's meminfo), the same line the checkpoint manager logs at every save and at its
        finalize, so the gather, the write and the drain read as one series.
        """
        log_host_memory(f"checkpoint save: {event} (components={list(component_plan.names)})")

    def save_validation_progress(self, path: str) -> None:
        """Publish a completed held-out pass and clear its durable recovery marker."""
        self._require_checkpoint_boundary()
        self._require_validation_enabled()
        self.validation_publication.publish(
            path,
            manager=self.checkpoint_manager,
            publication_identity={},
            trainer_state=self._trainer_state,
            on_checkpoint=lambda: self.callbacks.on_checkpoint(
                self,
                self.checkpoint_manager.iteration_location(path),
            ),
        )

    def load(self, path: str) -> int:
        """Restore model, optimizer, and exact trainer state."""
        import os

        if self.checkpoint_manager.has_dcp_component(path, "model"):
            resumed = self._load_dcp(path)
            self._resumed_from_checkpoint = True
            return resumed

        optimizer_sharded = False
        if os.path.isfile(path):  # legacy single-file checkpoint
            ckpt = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
            model_sd, opt_sd = ckpt["model"], ckpt.get("opt")
            trainer_state = {key: ckpt[key] for key in ("iter", "optimizer_step") if key in ckpt}
        else:
            model_sd = (self.checkpoint_manager.load_component(path, "model") or {}).get("net")
            # A consolidated checkpoint written by a DTensor-sharded trainer carries its optimizer as
            # per-rank DCP shards next to the consolidated model and control files (see `save`).
            # Older checkpoints hold `optim.pt`; both are exact-resume checkpoints.
            optimizer_sharded = self.checkpoint_manager.has_dcp_component(path, "optim")
            opt_sd = None if optimizer_sharded else self.checkpoint_manager.load_component(path, "optim")
            trainer_state = self.checkpoint_manager.load_component(path, "trainer")
        if model_sd is None:
            raise FileNotFoundError(f"no model state found in checkpoint {path!r}")
        if opt_sd is None and not optimizer_sharded:
            raise RuntimeError(f"training resume requires optimizer state in checkpoint {path!r}")
        if trainer_state is None:
            raise RuntimeError(f"training resume requires trainer state in checkpoint {path!r}")
        self._validate_objective_identity(trainer_state)

        net = self._net()
        pristine = self._checkpoint_recorded_pristine_optimizer(trainer_state)
        if is_dtensor_sharded(net):
            # The consolidated files are mmap-backed (`io.load_component`); every rank copies only
            # its own chunk of each tensor out of them into its live shards, the way the DCP reader
            # does. The whole dicts never go through the `full_state_dict` setters: those move every
            # consolidated tensor whole to the device and scatter it from rank 0 on EVERY rank, so a
            # one-node resume of a 30 B-parameter policy (58 GB model.pt + 223 GB optim.pt) reached
            # 855 GB of a 900 GB job cgroup in setup and Ray killed the trainer on every retry.
            self._restore_sharded_model_from_consolidated(net, model_sd, path)
            if optimizer_sharded:
                self._restore_sharded_optimizer(path, trainer_state)
            elif pristine:
                # `set_optimizer_state_dict` initializes the optimizer before applying, which is
                # exactly the step this checkpoint recorded as not having happened. The group
                # options still have to land, and they are already in hand here.
                self.reset_optimizer_to_pristine()
                self._apply_saved_param_groups(opt_sd)
            else:
                self._restore_sharded_optimizer_from_consolidated(net, opt_sd, path)
        else:
            net.load_state_dict(model_sd)
            if optimizer_sharded:
                # DCP reshards the saved shards onto this unsharded target.
                self._restore_sharded_optimizer(path, trainer_state)
            else:
                # `load_state_dict` applies whatever the file holds, including `param_groups`, and
                # does not materialize moments. The reset keeps a pristine checkpoint pristine even
                # when it is loaded into a trainer that has already stepped, or was written before
                # the emptied-state save.
                self.opt.load_state_dict(opt_sd)
                if pristine:
                    self.reset_optimizer_to_pristine()
                    self._apply_saved_param_groups(opt_sd)
        resumed = self._restore_trainer_state(trainer_state)
        self._resumed_from_checkpoint = True
        return resumed

    def _load_dcp(self, path: str) -> int:
        """Restore live DCP state with topology resharding."""
        from torch.distributed.checkpoint.state_dict import get_model_state_dict, set_model_state_dict

        net = run_rank_converged(self._net, "LLM DCP restore network extraction")
        trainer_state = self.checkpoint_manager.load_component(path, "trainer")
        if trainer_state is None:
            raise RuntimeError(f"training resume requires trainer state in DCP checkpoint {path!r}")
        run_rank_converged(
            lambda: self._validate_objective_identity(trainer_state),
            "LLM DCP objective identity validation",
        )
        model_state = run_rank_converged(  # [*sharded parameter] values
            lambda: get_model_state_dict(net),
            "LLM DCP model target extraction",
        )
        if not self.checkpoint_manager.load_dcp_component(path, "model", model_state):
            raise FileNotFoundError(f"no model component in DCP checkpoint {path!r}")
        run_rank_converged(
            lambda: set_model_state_dict(net, model_state),  # [*sharded parameter] module values
            "LLM DCP model state application",
        )
        self._restore_sharded_optimizer(path, trainer_state)
        return self._restore_trainer_state(trainer_state)

    def _restore_sharded_optimizer(self, path: str, trainer_state: Mapping[str, Any]) -> None:
        """Load the DCP ``optim`` component of ``path`` into the live optimizer, resharding as needed.

        Shared by the all-DCP checkpoint and the consolidated checkpoint of a sharded trainer, whose
        optimizer is the one component written as shards (see `save`).
        """
        from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict, set_optimizer_state_dict

        net = run_rank_converged(self._net, "LLM sharded optimizer restore network extraction")
        if self._checkpoint_recorded_pristine_optimizer(trainer_state):
            # Resuming inside a frozen-actor stretch: there are no moments on disk to restore, but
            # `param_groups` still carries the learning rate and the other options. Shaping a load
            # target requires the same materializing extraction the save does, so it is rolled back
            # the same way; `state` is emptied to match what a pristine save wrote. The branch is
            # rank-uniform because the decision comes from the checkpoint, not from local state.
            # Make the optimizer pristine FIRST, so the target extracted from it carries the same
            # group schema the pristine save wrote. Doing it afterwards is too late: DCP's planner
            # rejects a target field the checkpoint omits before any cleanup could run.
            run_rank_converged(self.reset_optimizer_to_pristine, "LLM DCP pristine optimizer reset")
            with self._preserved_pristine_optimizer() as guard:
                # The guard rolls back what this extraction creates even if the load raises,
                # sanitizing the target exactly as the save sanitized what it wrote.
                opt_state = run_rank_converged(  # [*sharded optimizer-state] values
                    lambda: guard.sanitized(get_optimizer_state_dict(net, self.opt)),
                    "LLM DCP pristine optimizer group extraction",
                )
                if not self.checkpoint_manager.load_dcp_component(path, "optim", opt_state):
                    raise RuntimeError(f"training resume requires optimizer state in DCP checkpoint {path!r}")
            run_rank_converged(
                lambda: self._apply_saved_param_groups(opt_state),
                "LLM DCP pristine optimizer group application",
            )
            return
        opt_state = run_rank_converged(  # [*sharded optimizer-state] values
            lambda: get_optimizer_state_dict(net, self.opt),
            "LLM DCP optimizer target extraction",
        )
        if not self.checkpoint_manager.load_dcp_component(path, "optim", opt_state):
            raise RuntimeError(f"training resume requires optimizer state in DCP checkpoint {path!r}")
        run_rank_converged(
            lambda: set_optimizer_state_dict(  # [*sharded optimizer-state] optimizer values
                net,
                self.opt,
                optim_state_dict=opt_state,
            ),
            "LLM DCP optimizer state application",
        )

    def _restore_sharded_model_from_consolidated(
        self, net: torch.nn.Module, model_sd: Mapping[str, Any], path: str
    ) -> int:
        """Fill the live sharded model from a consolidated ``model.pt``; returns bytes this rank copied.

        Same shape as `_load_dcp`'s model half: extract the sharded target, fill it in place, apply
        it with the plain setter. Strict, like ``load_state_dict(strict=True)`` was on this path.
        """
        from torch.distributed.checkpoint.state_dict import get_model_state_dict, set_model_state_dict

        model_state = run_rank_converged(  # [*sharded parameter] values
            lambda: get_model_state_dict(net),
            "LLM consolidated model target extraction",
        )
        copied = run_rank_converged(
            lambda: fill_sharded_state_from_consolidated(
                model_state,
                model_sd,
                context=f"model state of consolidated checkpoint {path!r}",
                strict=True,
            ),
            "LLM consolidated model shard extraction",
        )
        run_rank_converged(
            lambda: set_model_state_dict(net, model_state),  # [*sharded parameter] module values
            "LLM consolidated model state application",
        )
        self._log_consolidated_restore("model", copied, path)
        return copied

    def _restore_sharded_optimizer_from_consolidated(
        self,
        net: torch.nn.Module,
        opt_sd: Mapping[str, Any],
        path: str,
    ) -> int:
        """Fill the live sharded optimizer from a consolidated ``optim.pt``; returns bytes this rank copied.

        Same shape as `_restore_sharded_optimizer`'s stepped branch with the DCP read replaced by
        the per-rank chunk copy. Extras in the file are ignored as ``set_optimizer_state_dict``
        ignores them; a live moment the file lacks is an error. The pristine checkpoint is handled
        by the caller (no moments to fill).
        """
        from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict, set_optimizer_state_dict

        opt_state = run_rank_converged(  # [*sharded optimizer-state] values
            lambda: get_optimizer_state_dict(net, self.opt),
            "LLM consolidated optimizer target extraction",
        )
        copied = run_rank_converged(
            lambda: fill_sharded_state_from_consolidated(
                opt_state,
                opt_sd,
                context=f"optimizer state of consolidated checkpoint {path!r}",
                strict=False,
            ),
            "LLM consolidated optimizer shard extraction",
        )
        run_rank_converged(
            lambda: set_optimizer_state_dict(  # [*sharded optimizer-state] optimizer values
                net,
                self.opt,
                optim_state_dict=opt_state,
            ),
            "LLM consolidated optimizer state application",
        )
        self._log_consolidated_restore("optimizer", copied, path)
        return copied

    @staticmethod
    def _log_consolidated_restore(component: str, copied: int, path: str) -> None:
        """Log what this rank materialized out of a consolidated file, with the shared host-memory reading beside it."""
        log_host_memory(
            f"checkpoint resume: copied {copied / 2**30:.2f} GiB of this rank's {component} shards out of the "
            f"consolidated files of {path}"
        )

    @staticmethod
    def _checkpoint_recorded_pristine_optimizer(trainer_state: Mapping[str, Any] | None) -> bool:
        """Whether this checkpoint states its optimizer had never stepped.

        Absent is NOT pristine: checkpoints written before the marker existed say nothing, and the
        only safe reading of silence is the behavior they were written under.
        """
        return bool((trainer_state or {}).get("optimizer_pristine", False))

    def _restore_trainer_state(self, trainer_state: dict[str, Any]) -> int:
        """Apply trainer control state and converge every rank-local restore failure."""
        return run_rank_converged(
            lambda: self._restore_trainer_state_local(trainer_state),
            "LLM trainer state restore",
        )

    def _restore_trainer_state_local(self, trainer_state: dict[str, Any]) -> int:
        """Apply rank-local trainer state; the public wrapper owns failure consensus."""
        self._validate_objective_identity(trainer_state)
        update_state = self._parse_update_state(trainer_state)
        # The first local revision wrote the counter but no explicit behavior marker. Treat field
        # presence as enabled so those checkpoints remain resumable under the matching rule.
        saved_skip_zero_grad = bool(trainer_state.get("skip_zero_grad", "zero_grad_skips_total" in trainer_state))
        if saved_skip_zero_grad != self.skip_zero_grad:
            raise RuntimeError(
                f"checkpoint skip_zero_grad={saved_skip_zero_grad} differs from this run's "
                f"skip_zero_grad={self.skip_zero_grad}: this setting changes whether AdamW mutates "
                "weights and optimizer moments on an all-zero gradient. Resume with the original "
                "setting, or start a fresh run."
            )
        # Refuse a resume whose batching shape changed: the driver's optimizer→batch-step conversion
        # divides by n_mini, so a changed n_mini lands on a silently wrong batch (see `n_mini` store).
        saved_n_mini = int(trainer_state.get("n_mini", 1))
        if saved_n_mini != self.n_mini:
            raise RuntimeError(
                f"checkpoint was written with n_mini={saved_n_mini} but this run has "
                f"n_mini={self.n_mini}: the loop resumes by dividing the durable optimizer-step "
                f"count by n_mini, so a changed value silently resumes at the wrong collected "
                f"batch (skipping data, or re-running it). Resume with the original "
                f"rl.mini_batch_per_replica, or start a fresh run."
            )
        resumed = self._apply_update_state(update_state, rng_state=trainer_state.get("rng"))
        self._denom_accum = 0.0
        self._skip_remaining = 0
        scheduler_clock = self._scheduler_clock_name()
        scheduler_attempts = update_state.optimizer_step
        if scheduler_clock == "batch":
            scheduler_attempts += update_state.zero_grad_skips_total
            # Resume mid-batch on the collected-batch clock: `_sched_accum` is how many of this batch's
            # n_mini scheduler-advancing attempts already consumed before the checkpoint boundary.
            # Zero-gradient skips advance this clock, while rejected non-finite attempts do not.
            self._sched_accum = scheduler_attempts % self.n_mini
        else:
            self._sched_accum = 0
        # Restore or reconstruct scheduler progress after update clocks.
        sched_sd = trainer_state.get("lr_scheduler")
        if self.lr_scheduler is not None and sched_sd is not None:
            # LEGACY conversion: a checkpoint written before the collected-batch clock stepped the
            # schedule once per OPTIMIZER step, so its last_epoch counted n_mini ticks per batch.
            # Divide by n_mini to land on the collected-batch position this run advances on. New
            # checkpoints carry lr_scheduler_clock == "batch" and load verbatim.
            saved_scheduler_clock = trainer_state.get("lr_scheduler_clock")
            if scheduler_clock == "batch":
                if saved_scheduler_clock not in (None, "batch"):
                    raise RuntimeError(
                        f"checkpoint scheduler clock {saved_scheduler_clock!r} cannot restore into "
                        "a batch-clock scheduler"
                    )
                if saved_scheduler_clock is None and self.n_mini > 1:
                    sched_sd = dict(sched_sd)
                    if "last_epoch" in sched_sd:
                        sched_sd["last_epoch"] = int(sched_sd["last_epoch"]) // self.n_mini
            elif saved_scheduler_clock not in (None, "optimizer_step"):
                raise RuntimeError(
                    f"checkpoint scheduler clock {saved_scheduler_clock!r} cannot restore into "
                    "an optimizer-step scheduler"
                )
            self.lr_scheduler.load_state_dict(sched_sd)
        elif self.lr_scheduler is not None and scheduler_attempts > 0:
            # Legacy checkpoints without scheduler state reconstruct progress from the optimizer
            # iteration, now on the collected-batch clock (one tick per n_mini optimizer steps).
            scheduler_ticks = scheduler_attempts // self.n_mini if scheduler_clock == "batch" else scheduler_attempts
            for _ in range(scheduler_ticks):
                self.lr_scheduler.step()
        self.validation_state.defer_restore(trainer_state.get("validation"))
        return resumed

    def _validate_objective_identity(self, trainer_state: dict[str, Any]) -> None:
        """Fail before continuation when the checkpoint used different loss semantics."""
        saved = trainer_state.get("objective_identity")
        if saved is None:
            if self.objective_identity:
                raise RuntimeError(
                    "checkpoint predates the objective exact-resume identity; its loss semantics "
                    "cannot be verified. Resume with the original code/config, or start a fresh run."
                )
            return
        restored = normalize_json_dict(saved, context="checkpoint LLM objective exact-resume identity")
        if not json_identity_equal(restored, self.objective_identity, context="LLM objective exact-resume identity"):
            raise RuntimeError(
                "checkpoint objective exact-resume identity differs from the configured objective: "
                f"saved={restored!r}, configured={self.objective_identity!r}"
            )
