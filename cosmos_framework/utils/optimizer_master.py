# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Explicit FP32 update ownership for native BF16 parameters and FSDP2 shards.

The model owns its Parameters throughout. The optimizer owns one ordinary FP32
master per *local* shard; gradients arrive after native reduction and clipping.
No forward casts, collectives over parameters, or extra clipping happen here.
"""

from __future__ import annotations

import logging
import math
import numbers
from collections import defaultdict
from copy import deepcopy
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Shard

from cosmos_framework.utils.quantized_optimizer import (
    NAMES,
    _snapshot,
    build_quantized_optimizer,
    named_parameter_groups,
    state_log_summary,
)


def local_tensor(parameter):
    """The currently owned FSDP shard, never a cached gathered forward parameter."""
    return parameter.to_local() if isinstance(parameter, DTensor) else parameter


def _real_scalar(value) -> float | None:
    """``value`` as a plain float if it is a real numeric scalar, else None.

    Group learning rates set by a scheduler may be Python numbers, NumPy scalars or 0-dim tensors
    (generator schedulers produce the latter two); booleans and complex values are not rates.
    """
    if isinstance(value, torch.Tensor):
        return None if value.ndim or value.dtype == torch.bool or value.is_complex() else float(value)
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return None
    return float(value)


def parameter_layout(parameter):
    if not isinstance(parameter, torch.nn.Parameter) or not parameter.requires_grad:
        raise ValueError("master updates require final trainable Parameters")
    if parameter.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("master updates support BF16/FP32 storage; FP16 unscaling is not implemented")
    local = local_tensor(parameter)
    if not local.is_contiguous():
        raise ValueError("master updates require contiguous local shards")
    layout = {"shape": list(parameter.shape), "local_shape": list(local.shape), "dtype": str(parameter.dtype)}
    if isinstance(parameter, DTensor):
        mesh = parameter.device_mesh
        if mesh.ndim != 1 or parameter.placements != (Shard(0),):
            raise ValueError("master updates require a one-dimensional FSDP2 Shard(0) mesh; no TP/HSDP/Partial")
        if not dist.is_initialized() or mesh.mesh.flatten().tolist() != list(range(dist.get_world_size())):
            raise ValueError("master updates require the complete learner process group in rank order")
        layout.update(mesh=mesh.mesh.tolist(), placements=["Shard(0)"])
    elif dist.is_initialized() and dist.get_world_size() > 1:
        raise ValueError("distributed master updates require native FSDP2 parameters, not replicated tensors")
    return layout


class MasterWeightOptimizer(torch.optim.Optimizer):
    """Model-facing optimizer with tracked local masters and a single AdamW backend.

    Native schedulers and trainers see live model parameters. The backend sees
    ordinary local FP32 Parameters only. Checkpoints use the rank-local schema,
    never torch's generic optimizer/DTensor conversion or dtype casting.
    """

    cosmos_local_master_optimizer = True
    # What a rank-consistently nonfinite update does. ``"raise"`` (posttrain: its trainers already
    # reject such windows, so reaching here is a bug) or ``"skip"`` (generator: legacy optimizers
    # survive a NaN step, so this owner skips it on every rank without touching state).
    nonfinite_policy = "raise"

    def __init__(self, net, params, config):
        self.config = deepcopy(config)
        groups, selected, names = named_parameter_groups(net, params, context="master optimizer")
        layouts = [parameter_layout(p) for p in selected]
        if len({local_tensor(p).device for p in selected}) != 1:
            raise ValueError("master optimizer requires one local device")
        if any(isinstance(p, DTensor) != isinstance(selected[0], DTensor) for p in selected):
            raise ValueError("mixed sharded/unsharded trainable ownership is unsupported")
        # CPU is useful for control-flow/numerical unit tests of the native AdamW
        # bridge. Quantized construction still performs the real CUDA preflight.
        self.live = tuple(selected)
        self.names = tuple(names[id(p)] for p in selected)
        defaults = dict(lr=config.lr, betas=tuple(config.betas), eps=config.eps, weight_decay=config.weight_decay)
        super().__init__(groups, defaults)
        # Exact-resume identity: everything that decides the saved layout or its meaning. The torch
        # version does not (masters and torch AdamW moments are plain FP32 tensors), so it is
        # recorded as informational metadata in ``state_dict`` instead. The bitsandbytes version
        # stays in the backend identity: its quantization maps and block layout are its own format.
        self.identity = {
            "format": "cosmos_local_master_v2",
            "optimizer": config.name,
            "implementation": config.impl,
            "update_policy": "fp32_master",
            "world_size": dist.get_world_size() if dist.is_initialized() else 1,
            "rank": dist.get_rank() if dist.is_initialized() else 0,
            "parameters": [dict(name=n, **layout) for n, layout in zip(self.names, layouts)],
            "groups": [
                dict(
                    names=g["param_names"],
                    name=g.get("name"),
                    initial_lr=g["lr"],
                    betas=g["betas"],
                    eps=g["eps"],
                    weight_decay=g["weight_decay"],
                )
                for g in self.param_groups
            ],
        }
        # Already-FP32 storage is itself the master. Wrap its owned local storage
        # without another dense copy; BF16 parameters need one retained FP32 copy (one conversion).
        self.masters = tuple(
            torch.nn.Parameter(
                local_tensor(p).detach()
                if p.dtype == torch.float32
                else local_tensor(p).detach().to(torch.float32, copy=True)
            )
            for p in self.live
        )
        self.backend = self._backend(self.masters)
        self.cosmos_quantized_optimizer = config.name in NAMES
        # Facade-state bookkeeping: the backend only ever adds per-parameter state during a step
        # (reset/load rebuild it and refresh explicitly), so a length change is the whole signal.
        self._mapped_states = -1
        self._reported_states = -1
        # Cheap per-step invariants; the full layout was validated once above.
        self._local_signature = tuple(
            (tuple(local_tensor(p).shape), p.dtype, isinstance(p, DTensor)) for p in self.live
        )
        # Trainers may load model weights after building the optimizer (warm start, model-only
        # or EMA loads). The first step after binding verifies masters against the live model.
        self._verify_masters = True
        # Set by a trainer that has already made the rank-consistent finite decision for the
        # coming step (``note_gradients_checked_finite``); consumed by that step.
        self._gradients_checked = False
        self.last_backend_step_events = None
        self.skipped_nonfinite_steps = 0
        self._refresh_state()

    def _backend(self, masters):
        mapping = {id(p): m for p, m in zip(self.live, masters)}
        groups = [
            dict(g, params=[mapping[id(p)] for p in g["params"]], lr=identity["initial_lr"])
            for g, identity in zip(self.param_groups, self.identity["groups"])
        ]
        if self.config.name in NAMES:
            net = SimpleNamespace(named_parameters=lambda: iter(zip(self.names, masters)))
            return build_quantized_optimizer(net, groups, self.config, _local_owner=self.identity)
        kernel = {"fused": True} if self.config.impl == "fused" else {}
        if self.config.impl in ("foreach", "for-loop"):
            kernel["foreach"] = self.config.impl == "foreach"
        return torch.optim.AdamW(
            groups,
            lr=self.config.lr,
            betas=tuple(self.config.betas),
            eps=self.config.eps,
            weight_decay=self.config.weight_decay,
            **kernel,
        )

    def _refresh_state(self):
        """Point the model-facing ``state`` at the backend's per-master state dicts (shared, not copied)."""
        self.state = defaultdict(
            dict, {p: self.backend.state[m] for p, m in zip(self.live, self.masters) if m in self.backend.state}
        )
        self._mapped_states = len(self.backend.state)

    def sync_groups(self):
        """Capture the scheduler-owned groups, including changes after a step."""
        for live, backend in zip(self.param_groups, self.backend.param_groups):
            backend.update({k: v for k, v in live.items() if k != "params"})

    def note_gradients_checked_finite(self) -> None:
        """The caller has collectively verified that every rank's gradients for the next step are finite.

        Posttrain trainers run ``all_ranks_gradients_finite`` (``force_finite_grads``, required for the
        eight-bit names) before stepping. The next step then stages locally without repeating that
        collective and its host synchronization.
        """
        self._gradients_checked = True

    def _stage_gradients(self):
        """This rank's FP32 gradient views and the gradients present on this rank."""
        local = []
        for p, master, expected in zip(self.live, self.masters, self._local_signature):
            if (tuple(local_tensor(p).shape), p.dtype, isinstance(p, DTensor)) != expected:
                raise ValueError("live master parameter layout changed after construction")
            if p.dtype == torch.float32 and local_tensor(p).data_ptr() != master.data_ptr():
                raise ValueError("FP32 model shard storage changed after optimizer binding")
            grad = p.grad
            if grad is not None:
                if isinstance(grad, DTensor) and (grad.placements != p.placements or grad.device_mesh != p.device_mesh):
                    raise ValueError("gradient ownership does not match model shard")
                grad = local_tensor(grad)
                if grad.is_sparse:
                    raise ValueError("sparse master gradient rejected before update")
                grad = grad if grad.numel() else None
            local.append(grad)
        present = [grad for grad in local if grad is not None]
        return [None if grad is None else grad.float().contiguous() for grad in local], present

    def _finite_flag(self, present):
        """One fused max-norm scan for the whole update, as a device flag (no host sync here).

        The backend skips its own scan because this owner checks every rank.
        """
        if not present:
            return torch.ones((), dtype=torch.bool, device=self.masters[0].device if self.masters else "cpu")
        return torch.isfinite(torch.stack(torch._foreach_norm(present, float("inf")))).all()

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError("master optimizer closures are unsupported")
        # A step that does not run the backend must not report the previous step's timing.
        self.last_backend_step_events = None
        prechecked, self._gradients_checked = self._gradients_checked, False
        if self._verify_masters or (self.nonfinite_policy == "raise" and not prechecked):
            # First step after binding (or a posttrain step no trainer has checked): stage every
            # gradient and converge any failure before any rank mutates state.
            from cosmos_framework.posttrain.utils.dist import run_rank_converged

            def stage():
                self._adopt_rebound_fp32_storage()
                gradients, present = self._stage_gradients()
                finite = prechecked or bool(self._finite_flag(present))
                if not finite and self.nonfinite_policy == "raise":
                    raise ValueError("nonfinite master gradient rejected before update")
                resync = finite and self._verify_masters and self._masters_stale()
                if resync and self.backend.state:
                    raise ValueError(
                        "model weights changed under an initialized FP32-master optimizer; its masters would "
                        "overwrite them. Load optimizer state with the model, or rebuild the optimizer."
                    )
                return gradients, finite, resync

            gradients, finite, resync = run_rank_converged(stage, "master optimizer gradient staging")
            if self.nonfinite_policy == "skip":
                finite = self._all_ranks_finite(torch.tensor(finite))
        elif prechecked:
            # Steady state, after a trainer already decided finiteness on every rank: no collective
            # or host sync here. That trainer converges this step's failures (``LLMTrainer`` runs it
            # under ``run_rank_converged``), so a local staging error may raise directly.
            gradients, _ = self._stage_gradients()
            finite, resync = True, False
        else:
            # Steady state, ``skip`` policy: ONE collective and ONE host sync decide the step. A local
            # staging failure (a layout invariant) rides in the same all-reduce, so every rank raises
            # it together instead of leaving its peers in the collective.
            try:
                gradients, present = self._stage_gradients()
            except ValueError as error:
                gradients, present, staging_error = None, [], error
            else:
                staging_error = None
            finite = self._all_ranks_finite(self._finite_flag(present), staging_error=staging_error)
            resync = False
        if not finite:
            # ``nonfinite_policy == "skip"``: no master, moment, counter or model change on any rank.
            self.skipped_nonfinite_steps += 1
            logging.getLogger(__name__).warning(
                "Skipping FP32-master optimizer step: nonfinite gradient on at least one rank (%d skipped so far)",
                self.skipped_nonfinite_steps,
            )
            return None
        if resync:
            # No moments exist yet, so the live model (e.g. a warm-start load after construction)
            # is the only state: adopt it rather than writing stale masters back over it.
            self.resync_masters_from_model()
        self._verify_masters = False
        for master, grad in zip(self.masters, gradients):
            master.grad = grad
        self.sync_groups()
        timing = None
        if self.masters and self.masters[0].is_cuda:
            # Device-time of the backend call only (moment/master arithmetic plus any
            # managed-memory prefetch or fault service it triggers). No synchronization
            # here; the events are read at the next update receipt.
            timing = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            timing[0].record()
        result = self.backend.step()
        if timing is not None:
            timing[1].record()
        self.last_backend_step_events = timing
        for p, m in zip(self.live, self.masters):
            if p.dtype == torch.float32:
                continue  # The master aliases this shard's storage; the backend already updated it.
            local_tensor(p).copy_(m)
            # FP32 staging is consumed by this step and is not recovery state. Holding it until
            # the next window's zero_grad keeps a full FP32 gradient copy (2x the BF16 trainables)
            # alive through weight sync, rollout and target preparation. FP32 parameters alias
            # their own gradient, which the trainer still owns, so only true copies go.
            m.grad = None
        if len(self.backend.state) != self._mapped_states:
            self._refresh_state()  # only when the backend allocated state for another parameter
        if len(self.state) != self._reported_states:
            logging.getLogger("posttrain").info(
                "Native master optimizer state: %s", state_log_summary(self.state_report())
            )
            self._reported_states = len(self.state)
        return result

    @staticmethod
    def _all_ranks_finite(flag: torch.Tensor, *, staging_error: ValueError | None = None) -> bool:
        """Rank-consistent AND of a local finite flag: one collective, one host sync.

        ``staging_error`` is this rank's gradient staging failure, if any. It rides in the same MIN
        all-reduce (status 0: some rank failed staging, 1: some rank is nonfinite, 2: finite), so
        every rank raises it together, after the collective; the failing rank chains its own cause.
        """
        from cosmos_framework.posttrain.utils.dist import all_ranks_tensor_min

        local = torch.zeros_like(flag, dtype=torch.int32) if staging_error is not None else flag.to(torch.int32) + 1
        status = all_ranks_tensor_min(local)
        if status == 0:
            raise ValueError(
                "FP32-master gradient staging failed on at least one rank: a parameter or gradient layout "
                "changed after optimizer binding (the failing rank's error names it)"
            ) from staging_error
        return status == 2

    def _masters_stale(self) -> bool:
        """Whether any BF16-stored shard differs from its master (FP32 shards alias theirs)."""
        stale = [
            (local_tensor(p) != m.to(p.dtype)).any()
            for p, m in zip(self.live, self.masters)
            if p.dtype != torch.float32 and m.numel()
        ]
        return bool(torch.stack(stale).any()) if stale else False

    @torch.no_grad()
    def resync_masters_from_model(self) -> None:
        """Re-seed masters from the live model after a model-only load; moments are untouched."""
        for p, m in zip(self.live, self.masters):
            if p.dtype != torch.float32:
                m.copy_(local_tensor(p))
        self._verify_masters = False

    def _settle_pending_verification(self) -> None:
        """Before ``state_dict`` writes the masters: account for a model load since binding.

        Construction, ``commit_state_dict`` and ``reset_state`` leave a verification pending for
        the first step, because the model may still be loaded afterwards (warm start, model-only
        load). A save in between (e.g. an iteration-0 checkpoint) must not write the pre-load
        masters. Without moments the live model is the only state, so the masters are re-seeded
        from it. With moments, masters that disagree with the model are refused, as the first
        step would refuse them. Local and collective-free: callers may call this per rank.
        """
        if not self._verify_masters:
            return
        if not self.backend.state:
            self._adopt_rebound_fp32_storage()
            self.resync_masters_from_model()
            # Still pending: a model load may follow this save, and the first step must then adopt
            # it instead of writing these masters over it.
            self._verify_masters = True
        elif self._masters_stale():
            raise ValueError(
                "model weights changed under an initialized FP32-master optimizer; its saved masters would "
                "not match the saved model. Load optimizer state with the model, or rebuild the optimizer."
            )

    def _adopt_rebound_fp32_storage(self) -> None:
        """While verification is pending and no moments exist, alias FP32 storage the model rebound.

        A model load after binding may rebind the same Parameters' FP32 storage (``.data``
        replacement, ``module._apply`` or ``load_state_dict(assign=True)`` with swapped tensors, an
        FSDP2 re-pad). Without moments the live model is the only state, so
        the masters wrap the new storage, with no copy, as ``reset_state`` does. With moments,
        ``_stage_gradients`` keeps refusing the rebinding.
        """
        if self._verify_masters and not self.backend.state and self._realias_fp32_masters():
            self.backend = self._backend(self.masters)  # the backend must own the new master objects

    def zero_grad(self, set_to_none=True):
        super().zero_grad(set_to_none=set_to_none)
        self.backend.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        self._settle_pending_verification()
        self.sync_groups()
        backend = self.backend.state_dict()
        # The eight-bit backend's state_dict is already an independent CPU snapshot;
        # only torch AdamW returns live references that need one here.
        backend = backend if self.config.name in NAMES else _snapshot(backend)
        for group in backend["param_groups"]:
            # Scheduler-owned rates may be NumPy scalars or 0-dim tensors; the payload stores floats.
            lr = _real_scalar(group.get("lr"))
            if lr is not None:
                group["lr"] = lr
        return {
            "identity": deepcopy(self.identity),
            # Informational only: never compared on load.
            "versions": {"torch": str(torch.__version__)},
            "masters": _snapshot(self.masters),
            "backend": backend,
        }

    def stage_state_dict(self, saved):
        """Validate a host-resident payload against this optimizer; allocates nothing on the device.

        Returns the validated payload for ``commit_state_dict``. Staging never builds a second
        set of device masters or moments next to the live ones. The eight-bit backend checks its
        own moment schema in ``commit_state_dict``, before it replaces anything.
        """
        if set(saved) - {"versions"} != {"identity", "masters", "backend"} or saved["identity"] != self.identity:
            raise ValueError("master optimizer identity/topology mismatch; exact resume requires the same layout")
        values = saved["masters"]
        if len(values) != len(self.masters):
            raise ValueError("missing master weights")
        for value, master in zip(values, self.masters):
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != master.shape
                or value.dtype != torch.float32
                or not torch.isfinite(value).all()
            ):
                raise ValueError("invalid master weight shape/dtype/value")
        saved_groups = saved["backend"].get("param_groups", [])
        if len(saved_groups) != len(self.param_groups):
            raise ValueError("master optimizer group count mismatch")
        for old, live in zip(saved_groups, self.param_groups):
            for key in ("param_names", "name", "betas", "eps", "weight_decay"):
                if old.get(key) != live.get(key):
                    raise ValueError(f"master optimizer group {key} mismatch")
            lr = _real_scalar(old.get("lr"))
            if lr is None or not math.isfinite(lr) or lr < 0:
                raise ValueError("invalid master optimizer learning rate")
            old["lr"] = lr  # e.g. a NumPy or 0-dim tensor rate from an older payload, as a plain float
        if self.config.name not in NAMES:
            # AdamW.load_state_dict casts moments. Reject schema/dtype mismatches
            # first rather than silently repairing a damaged full-precision control.
            ids = [i for g in saved_groups for i in g["params"]]
            if len(ids) != len(self.masters) or len(set(ids)) != len(ids) or set(saved["backend"]["state"]) - set(ids):
                raise ValueError("invalid master AdamW parameter mapping")
            for i, m in zip(ids, self.masters):
                state = saved["backend"]["state"].get(i, {})
                if state and set(state) != {"step", "exp_avg", "exp_avg_sq"}:
                    raise ValueError("invalid master AdamW state schema")
                for key in ("exp_avg", "exp_avg_sq"):
                    if state and (state[key].shape != m.shape or state[key].dtype != torch.float32):
                        raise ValueError("invalid master AdamW moment shape/dtype")
        return saved

    def commit_state_dict(self, saved):
        """Apply a staged payload in place: backend state first (it validates before replacing), then masters.

        Device peak: the live masters are overwritten in place, never duplicated. The backend
        allocates its restored moments before releasing the previous ones, so a restore into an
        optimizer that already has moments briefly holds both; a fresh optimizer (the usual resume)
        has none. Callers that need rollback keep ``state_dict()`` (host) or use ``reset_state``.
        """
        if self._realias_fp32_masters():
            self.backend = self._backend(self.masters)  # the backend must own the new master objects
        self.backend.load_state_dict(saved["backend"])
        with torch.no_grad():
            # FP32 masters alias the model shard; callers apply this only after every rank has
            # validated that the checkpoint's model and masters agree.
            for master, value in zip(self.masters, saved["masters"]):
                master.copy_(value)
        self._verify_masters = True
        for live, restored in zip(self.param_groups, self.backend.param_groups):
            live.update({k: v for k, v in restored.items() if k != "params"})
        self._refresh_state()

    def _realias_fp32_masters(self) -> bool:
        """Alias FP32 masters to the model's current storage again; whether any was rebound.

        The model may rebind FP32 storage after binding (e.g. ``.data`` replacement). Wrap the
        final storage again, with no second FP32 allocation; the caller rebuilds the backend over
        the new master objects.
        """
        if not any(
            p.dtype == torch.float32 and local_tensor(p).data_ptr() != m.data_ptr()
            for p, m in zip(self.live, self.masters)
        ):
            return False
        self.masters = tuple(
            torch.nn.Parameter(local_tensor(p).detach()) if p.dtype == torch.float32 else m
            for p, m in zip(self.live, self.masters)
        )
        return True

    def reset_state(self):
        """Return to the freshly built state: no moments, masters re-seeded from the live model.

        Scheduler-owned group values stay as they are; the next step copies them to the backend.
        """
        self._realias_fp32_masters()
        self.backend = self._backend(self.masters)
        self.resync_masters_from_model()
        # Deliberately pending again, exactly as after construction: "fresh" keeps the contract
        # that a model load may still follow (e.g. a pristine reset ahead of a model-only load),
        # and the one staging check on the next step (or ``state_dict``) then re-seeds the masters
        # instead of writing stale ones over the loaded model. Resets are rare (pristine saves,
        # fresh starts, rollback), so this single check costs nothing per step.
        self._verify_masters = True
        self._refresh_state()

    def load_state_dict(self, saved):
        # The native checkpoint owner coordinates model and optimizer restore.
        # BF16 model values are restored there, never reconstructed by an update.
        self.commit_state_dict(self.stage_state_dict(saved))

    def state_report(self):
        report = (
            self.backend.state_report()
            if self.config.name in NAMES
            else {
                "moment_bytes": sum(
                    v.numel() * v.element_size()
                    for s in self.backend.state.values()
                    for k, v in s.items()
                    if k in ("exp_avg", "exp_avg_sq")
                ),
                "metadata_bytes": sum(
                    value.numel() * value.element_size()
                    for state in self.backend.state.values()
                    for key, value in state.items()
                    if key not in ("exp_avg", "exp_avg_sq") and isinstance(value, torch.Tensor)
                ),
            }
        )
        return dict(
            report,
            master_identity=self.identity,
            master_bytes=sum(
                m.numel() * m.element_size() for p, m in zip(self.live, self.masters) if p.dtype != torch.float32
            ),
            fp32_model_alias_bytes=sum(
                m.numel() * m.element_size() for p, m in zip(self.live, self.masters) if p.dtype == torch.float32
            ),
            gradient_staging_bytes=sum(
                m.grad.numel() * m.grad.element_size()
                for p, m in zip(self.live, self.masters)
                if m.grad is not None and p.dtype != torch.float32
            ),
            fp32_gradient_alias_bytes=sum(
                m.grad.numel() * m.grad.element_size()
                for p, m in zip(self.live, self.masters)
                if m.grad is not None and p.dtype == torch.float32
            ),
        )
