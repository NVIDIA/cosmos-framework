# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Detect gradient-norm spikes and rewind the model past them.

A constant learning rate with no decay leaves LoRA SFT marginally stable: most steps
sit at a gradient norm well under the clip threshold, then a single unlucky batch
produces a norm one or two orders of magnitude larger. Clipping bounds the *magnitude*
of that step but not its *direction*, and AdamW folds the anomalous direction into its
first and second moments. Because AdamW's update is scale-invariant, the poisoned
moments keep steering the model for many steps after the offending batch is gone, and
the run either degrades or diverges outright.

Detection uses the gradient norm rather than the loss because the loss is a lagging
indicator. Measured on Cosmos3-Nano LoRA SFT, at the spike the gradient norm separates
from its running baseline by 10.4x while the loss still reads a healthy 0.2742 and
separates by only 2.8x -- the gradient norm fires roughly 100 steps before the loss
makes the problem visible.

Recovery rewinds rather than skips. Skipping the update leaves the poisoned moments in
place, and once the loss is elevated every subsequent norm looks like a spike, so a
skip-based guard freezes a diverged run instead of rescuing it (observed: 170 of 180
steps skipped in a single epoch). This callback keeps a short ring of parameter and
optimizer-moment snapshots and restores the oldest one, which discards the moments as
well as the weights, then backs the learning rate off so the retry is less likely to
trip the same edge.

MEMORY: the ring holds ``rollback_depth`` copies of every trainable parameter plus its
two AdamW moments. That is roughly 1.2GB for a rank-64 LoRA adapter, but it scales with
the *trainable* parameter count, so full fine-tuning of an 8B model would need hundreds
of gigabytes. Hence ``enabled`` defaults to False; turn it on for PEFT. Paged eight-bit
moments are the exception: their snapshots are staged in pinned host memory, not VRAM.
"""

from collections import deque
from copy import deepcopy
from typing import Any

import torch
import torch.distributed as dist

from cosmos_framework.utils import log
from cosmos_framework.utils.callback import Callback
from cosmos_framework.utils.quantized_optimizer import copy_state_tensor, device_view
from cosmos_framework.utils.quantized_optimizer_names import is_local_master_optimizer, is_quantized_optimizer

try:  # ImaginaireModel is only needed for type hints.
    from cosmos_framework.model._base import ImaginaireModel
except Exception:  # pragma: no cover - typing convenience only
    ImaginaireModel = Any  # type: ignore[assignment,misc]


def _masters_match(saved: dict[int, list[torch.Tensor | None]], opts: list[torch.optim.Optimizer]) -> bool:
    """Whether every snapshotted FP32 master still has a live counterpart of the same shape and dtype."""
    for opt_index, saved_masters in saved.items():
        if opt_index >= len(opts) or not is_local_master_optimizer(opts[opt_index]):
            return False
        live = opts[opt_index].masters
        if len(live) != len(saved_masters) or any(
            copy is not None and (copy.shape != master.shape or copy.dtype != master.dtype)
            for master, copy in zip(live, saved_masters)
        ):
            return False
    return True


def _real_optimizers(optimizer: Any) -> list[torch.optim.Optimizer]:
    """Normalise the trainer's optimizer argument to a list of real optimizers.

    The reasoner path hands over an ``OptimizersContainer``, which owns one optimizer per
    model part and exposes neither ``param_groups`` nor ``state``. It is iterable, so
    iterate; a plain ``torch.optim.Optimizer`` is wrapped in a single-element list.
    """
    if hasattr(optimizer, "param_groups"):
        return [optimizer]
    try:
        return list(iter(optimizer))
    except TypeError:
        return []


def _real_schedulers(scheduler: Any) -> list[Any]:
    """Normalise the trainer's scheduler argument to a list of real LR schedulers.

    ``SchedulersContainer`` holds one ``LambdaLR`` per optimizer under ``.schedulers``.
    """
    if hasattr(scheduler, "base_lrs"):
        return [scheduler]
    inner = getattr(scheduler, "schedulers", None)
    if inner:
        return list(inner)
    return []


class LossSpikeRollback(Callback):
    """Rewind past gradient-norm spikes instead of training through them.

    Hooks used, and why each:

    ``on_before_optimizer_step``
        Reads the complete accumulated gradient once per optimizer update. Register
        this callback before GradClip so detection sees the unclipped global norm.
        Native DTensor norm reduction preserves shard/replica semantics; a final
        rank-wide consensus keeps every rollback and learning-rate decision aligned.

    ``on_before_zero_grad``
        Runs immediately AFTER the optimizer step, so restoring here undoes the damaging
        update itself rather than trying to pre-empt it. Cancelling the step from
        ``on_before_optimizer_step`` is not possible -- the trainer calls ``_optimizer_step``
        unconditionally -- and merely zeroing the gradients would not help, since AdamW still
        moves the weights from momentum alone.
    """

    def __init__(
        self,
        enabled: bool = False,
        grad_norm_factor: float = 10.0,
        window: int = 50,
        min_observations: int = 12,
        rollback_depth: int = 4,
        max_consecutive: int = 8,
        lr_backoff: float = 0.5,
        lr_recovery: float = 1.02,
        lr_min_scale: float = 0.1,
        lr_ceiling_decay: float = 0.8,
        baseline_inflation_cap: float = 4.0,
        backoff_cooldown: int = 50,
        recovery_health_factor: float = 2.0,
    ) -> None:
        super().__init__()
        self.enabled = enabled
        self.grad_norm_factor = grad_norm_factor
        self.window = window
        self.min_observations = min_observations
        self.rollback_depth = max(1, rollback_depth)
        self.max_consecutive = max_consecutive
        self.lr_backoff = lr_backoff
        self.lr_recovery = lr_recovery
        self.lr_min_scale = lr_min_scale
        self.lr_ceiling_decay = lr_ceiling_decay
        self.baseline_inflation_cap = baseline_inflation_cap
        self.backoff_cooldown = backoff_cooldown
        self.recovery_health_factor = recovery_health_factor

        self._norms: deque[float] = deque(maxlen=window)
        self._ring: deque[dict[str, Any]] = deque(maxlen=self.rollback_depth)
        self._pending_reason: str | None = None
        self._consecutive = 0
        self._lr_scale = 1.0
        # Ceiling that recovery walks back toward. Ratchets down on every rollback so a run
        # that keeps tripping settles at a lower rate instead of climbing back to a rate it
        # has already demonstrated it cannot hold.
        self._lr_ceiling = 1.0
        self._base_lrs: list[list[float]] | None = None
        # Lowest median seen since arming: the run's demonstrated healthy scale.
        self._healthy_baseline: float | None = None
        # Iteration of the last learning-rate backoff, so a burst of spikes counts as one
        # episode rather than as one escalation per rollback.
        self._last_backoff: int | None = None
        self.rollbacks = 0

    # ---------------------------------------------------------------- detection

    def _baseline(self) -> float | None:
        """Median of the recent window, or None until enough steps have been seen.

        The median rather than the mean because a spike that is already inside the window
        would drag a mean upward and mask the next one.
        """
        if len(self._norms) < self.min_observations:
            return None
        ordered = sorted(self._norms)
        mid = len(ordered) // 2
        median = ordered[mid] if len(ordered) % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])

        # Anchor the baseline to the healthiest scale this run has demonstrated. Without
        # this the baseline chases a deteriorating run upward: once the window fills with
        # elevated norms, the relative test silently demands an ever larger spike to trip.
        # Observed in practice -- a baseline that drifted from ~0.4 to 9.03 needed a norm
        # above 90 to fire, so the guard went quiet exactly when it was needed most.
        if self._healthy_baseline is None or median < self._healthy_baseline:
            self._healthy_baseline = median
        return min(median, self.baseline_inflation_cap * self._healthy_baseline)

    @torch.no_grad()
    def on_before_optimizer_step(
        self,
        model: "ImaginaireModel",
        optimizer: Any,
        scheduler: Any,
        grad_scaler: Any,
        iteration: int = 0,
    ) -> None:
        if not self.enabled:
            return
        from cosmos_framework.callbacks.grad_clip import _group_params_by_mesh, _total_norm_by_mesh

        parameters = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
        if not parameters:
            raise RuntimeError("Enabled loss-spike guard requires gradients at the optimizer-step boundary")
        norm, _ = _total_norm_by_mesh(_group_params_by_mesh(parameters), foreach=True)
        # Do not mutate/unscale gradients here; the optimizer owns that operation.
        # Removing the current AMP scale keeps the historical baseline comparable.
        norm = norm / (grad_scaler.get_scale() if grad_scaler is not None else 1.0)
        norm = torch.nan_to_num(norm, nan=float("inf"), posinf=float("inf"), neginf=float("inf"))
        if dist.is_available() and dist.is_initialized():
            # Norms are already reduced across DTensor shard dimensions. MAX is
            # consensus across replicas/stages, not a second sum of those shards.
            dist.all_reduce(norm, op=dist.ReduceOp.MAX)
        grad_norm = float(norm.item())

        if not (grad_norm == grad_norm) or grad_norm in (float("inf"), float("-inf")):
            self._pending_reason = f"non-finite gradient norm ({grad_norm})"
            return

        baseline = self._baseline()
        if baseline is not None and baseline > 0.0 and grad_norm > self.grad_norm_factor * baseline:
            self._pending_reason = (
                f"gradient norm {grad_norm:.4f} exceeds {self.grad_norm_factor:.1f}x "
                f"the median of the last {len(self._norms)} steps ({baseline:.4f})"
            )
            return

        self._pending_reason = None
        self._norms.append(grad_norm)

    def _is_healthy(self) -> bool:
        """Is the current gradient-norm scale back near the run's demonstrated healthy one?"""
        if self._healthy_baseline is None or len(self._norms) < self.min_observations:
            return True  # not enough evidence to withhold recovery
        ordered = sorted(self._norms)
        mid = len(ordered) // 2
        median = ordered[mid] if len(ordered) % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])
        return median <= self.recovery_health_factor * self._healthy_baseline

    # ------------------------------------------------------------ snapshot/undo

    @torch.no_grad()
    def _capture(self, model: "ImaginaireModel", optimizer: torch.optim.Optimizer) -> None:
        params = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}
        moments: dict[tuple[int, int], dict[str, Any]] = {}
        masters: dict[int, list[torch.Tensor | None]] = {}
        for opt_index, opt in enumerate(_real_optimizers(optimizer)):
            if is_local_master_optimizer(opt):
                # The FP32-master owner writes its masters back into the model on every step, so
                # a rewind that skipped them would be undone by the next update. An FP32 master
                # aliases its parameter's storage, already captured in ``params``: no second copy.
                masters[opt_index] = [
                    None if live.dtype == torch.float32 else master.detach().clone()
                    for live, master in zip(opt.live, opt.masters)
                ]
            # bitsandbytes' quantization maps (``qmap1``/``qmap2``) are constant lookup tables shared
            # by every parameter; the optimizer never writes them, so they are neither copied nor
            # restored.
            constant = ("qmap",) if is_quantized_optimizer(opt) else ()
            for index, parameter in enumerate(p for group in opt.param_groups for p in group["params"]):
                state = opt.state.get(parameter)
                if not state:
                    continue
                # Non-tensor entries matter too: bitsandbytes keeps ``step`` as a Python int, and
                # restoring old moments with an advanced step would change bias correction. Tensors
                # are cloned where they live, except paged moments: a device clone would put their
                # bytes back in VRAM, which paging exists to avoid, so they are staged in pinned host
                # memory, stream-ordered after the step that wrote them.
                moments[(opt_index, index)] = {
                    key: copy_state_tensor(value, ordinary="device")
                    if isinstance(value, torch.Tensor)
                    else deepcopy(value)
                    for key, value in state.items()
                    if not key.startswith(constant)
                }
        self._ring.append({"params": params, "moments": moments, "masters": masters})

    @torch.no_grad()
    def _restore(self, model: "ImaginaireModel", optimizer: torch.optim.Optimizer) -> bool:
        if not self._ring:
            return False
        snapshot = self._ring[0]
        opts = _real_optimizers(optimizer)
        if not _masters_match(snapshot.get("masters", {}), opts):
            # Checked before anything is written, so a mismatch never leaves a half-applied rewind.
            log.warning("LossSpikeRollback: snapshot FP32 masters do not match the current optimizer; not rolling back")
            return False
        named = dict(model.named_parameters())
        for name, saved in snapshot["params"].items():
            target = named.get(name)
            if target is not None:
                target.detach().copy_(saved)
        flats = [[p for group in opt.param_groups for p in group["params"]] for opt in opts]
        paged_devices: set[torch.device] = set()
        for (opt_index, index), saved_state in snapshot["moments"].items():
            if opt_index >= len(opts) or index >= len(flats[opt_index]):
                continue
            state = opts[opt_index].state.get(flats[opt_index][index])
            if not state:
                continue
            for key, saved in saved_state.items():
                if key not in state:
                    continue
                if isinstance(state[key], torch.Tensor):
                    target = device_view(state[key])
                    if target is not state[key]:
                        paged_devices.add(target.device)
                    # A pinned host snapshot of paged state goes back through the CUDA view, on the
                    # current stream, so the next optimizer step is ordered after it.
                    target.copy_(saved, non_blocking=True)
                else:
                    state[key] = deepcopy(saved)
        for opt_index, saved_masters in snapshot.get("masters", {}).items():
            for master, saved in zip(opts[opt_index].masters, saved_masters):
                if saved is not None:  # None: aliases a parameter restored above
                    master.copy_(saved)
        for device in paged_devices:
            # Managed pages are also reachable from the host (checkpoint snapshots, host-side
            # checks): finish the restore before anything can read them outside stream order.
            torch.cuda.synchronize(device)
        # Everything newer than the restored point came after the poisoned step, so it is
        # not a safe place to rewind to a second time. The gradient-norm history is NOT
        # cleared alongside it: those samples were all taken before the spike (the spiking
        # value is never appended), they describe the state we just rewound to, and
        # discarding them forces the baseline to be rebuilt from post-rollback steps that
        # may themselves be unhealthy.
        self._ring.clear()
        return True

    def _scale_lr(self, optimizer: torch.optim.Optimizer, scheduler: Any, factor: float) -> None:
        """Rescale both the next update's rate and the scheduler's base rates.

        Writing ``param_group["lr"]`` directly would not survive: ``LambdaLR`` recomputes
        ``lr = base_lrs[i] * lr_lambda(step)`` on every ``scheduler.step()``, so a direct
        write is discarded on the very next iteration. ``base_lrs`` is the input to that
        product and therefore the durable place to apply the backoff. The trainer
        has already stepped the scheduler before this hook, so the current rates
        must also change before the next optimizer update.
        """
        schedulers = _real_schedulers(scheduler)
        previous_scale = self._lr_scale
        if self._base_lrs is None:
            if schedulers:
                self._base_lrs = [list(sched.base_lrs) for sched in schedulers]
            else:
                self._base_lrs = [[group["lr"] for group in opt.param_groups] for opt in _real_optimizers(optimizer)]
        if factor < 1.0:
            # Back off RELATIVE TO THE CEILING, not to the current scale. Compounding from
            # the current scale makes a cluster of spikes multiply out: four rollbacks
            # inside thirteen steps drove 0.5**4 straight into the floor, and the run then
            # spent its remaining 274 steps at a tenth of the intended rate -- rescued from
            # divergence but undertrained, which cost roughly ten points of accuracy.
            # The ceiling carries the persistent penalty; this is the temporary dip.
            self._lr_ceiling = max(self.lr_min_scale, self._lr_ceiling * self.lr_ceiling_decay)
            self._lr_scale = max(self.lr_min_scale, self._lr_ceiling * self.lr_backoff)
        else:
            self._lr_scale = max(self.lr_min_scale, min(self._lr_ceiling, self._lr_scale * factor))

        if schedulers:
            for sched, bases in zip(schedulers, self._base_lrs):
                sched.base_lrs = [base * self._lr_scale for base in bases]
            # Preserve the scheduler's current warmup/decay multiplier. Changing
            # only base_lrs takes effect after the next optimizer update, too late
            # to protect that update. The ratio also handles gradual recovery and
            # repeated backoffs without applying the cumulative scale twice.
            if self._lr_scale != previous_scale:
                for opt in _real_optimizers(optimizer):
                    for group in opt.param_groups:
                        group["lr"] = group["lr"] * self._lr_scale / previous_scale
            # Keep logging and checkpoint state aligned with the next update's
            # rate without advancing the schedule or its counters.
            for sched in schedulers:
                sched._last_lr = [group["lr"] for group in sched.optimizer.param_groups]
            return
        for opt, bases in zip(_real_optimizers(optimizer), self._base_lrs):
            for group, base in zip(opt.param_groups, bases):
                group["lr"] = base * self._lr_scale

    def on_before_zero_grad(
        self,
        model: "ImaginaireModel",
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        iteration: int = 0,
    ) -> None:
        if not self.enabled:
            return

        if self._pending_reason is None:
            self._consecutive = 0
            # Recover only once the run is demonstrably healthy again, not merely because
            # this particular step did not spike. A degraded run looks clean between its
            # spikes, so a recovery gated on "no spike this step" walks the rate back up
            # while the model is still sick -- observed climbing to 0.627 of base while the
            # median loss sat at 0.69, five times its healthy value, for 280 steps.
            if self._lr_scale < self._lr_ceiling and self._is_healthy():
                self._scale_lr(optimizer, scheduler, self.lr_recovery)
            self._capture(model, optimizer)
            return

        reason = self._pending_reason
        self._pending_reason = None
        self._consecutive += 1
        if self._consecutive > self.max_consecutive:
            log.warning(
                f"loss-spike guard: {self._consecutive} consecutive spikes at iteration {iteration}; "
                "standing down so a genuinely diverged run is not frozen mid-epoch"
            )
            return

        if self._restore(model, optimizer):
            self.rollbacks += 1
            # Rewinding repeatedly is cheap and safe; cutting the rate repeatedly is not.
            # Only the first rollback of an episode moves the rate.
            fresh_episode = self._last_backoff is None or (iteration - self._last_backoff) >= self.backoff_cooldown
            if fresh_episode:
                self._last_backoff = iteration
                self._scale_lr(optimizer, scheduler, self.lr_backoff)
            log.warning(
                f"loss-spike guard: rolled back at iteration {iteration} ({reason}); "
                f"learning rate scaled to {self._lr_scale:.3f} of base [rollback #{self.rollbacks}]"
            )
        else:
            log.warning(
                f"loss-spike guard: spike at iteration {iteration} ({reason}) but no snapshot is "
                "available yet; letting the step stand"
            )
