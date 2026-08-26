# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
of gigabytes. Hence ``enabled`` defaults to False; turn it on for PEFT.
"""

from collections import deque
from typing import Any

import torch

from cosmos_framework.utils.callback import Callback
from cosmos_framework.utils import log

try:  # ImaginaireModel is only needed for type hints.
    from cosmos_framework.model import ImaginaireModel
except Exception:  # pragma: no cover - typing convenience only
    ImaginaireModel = Any  # type: ignore[assignment,misc]


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

    ``on_after_backward``
        Reads the gradient norm BEFORE :class:`~cosmos_framework.callbacks.grad_clip.GradClip`
        rescales it. Clipping compresses every spike down to ``clip_norm``, so a norm read
        after clipping carries no signal at all. This hook is the last point at which the
        true magnitude is still visible, regardless of the order callbacks are registered in.

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

        self._norms: deque[float] = deque(maxlen=window)
        self._ring: deque[dict[str, Any]] = deque(maxlen=self.rollback_depth)
        self._pending_reason: str | None = None
        self._consecutive = 0
        self._lr_scale = 1.0
        self._base_lrs: list[float] | None = None
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
        if len(ordered) % 2:
            return ordered[mid]
        return 0.5 * (ordered[mid - 1] + ordered[mid])

    @torch.no_grad()
    def on_after_backward(self, model: "ImaginaireModel", iteration: int = 0) -> None:
        if not self.enabled:
            return
        total_sq = 0.0
        for parameter in model.parameters():
            if parameter.requires_grad and parameter.grad is not None:
                grad = parameter.grad
                # DTensor under FSDP: reduce to the local shard's contribution. The
                # baseline is a ratio against this same quantity, so a consistently
                # partial norm still detects a spike.
                grad = grad.to_local() if hasattr(grad, "to_local") else grad
                total_sq += float(grad.detach().float().pow(2).sum())
        grad_norm = total_sq**0.5

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

    # ------------------------------------------------------------ snapshot/undo

    @torch.no_grad()
    def _capture(self, model: "ImaginaireModel", optimizer: torch.optim.Optimizer) -> None:
        params = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}
        moments: dict[tuple[int, int], dict[str, torch.Tensor]] = {}
        for opt_index, opt in enumerate(_real_optimizers(optimizer)):
            for index, parameter in enumerate(p for group in opt.param_groups for p in group["params"]):
                state = opt.state.get(parameter)
                if not state:
                    continue
                moments[(opt_index, index)] = {
                    key: value.detach().clone() for key, value in state.items() if isinstance(value, torch.Tensor)
                }
        self._ring.append({"params": params, "moments": moments})

    @torch.no_grad()
    def _restore(self, model: "ImaginaireModel", optimizer: torch.optim.Optimizer) -> bool:
        if not self._ring:
            return False
        snapshot = self._ring[0]
        named = dict(model.named_parameters())
        for name, saved in snapshot["params"].items():
            target = named.get(name)
            if target is not None:
                target.detach().copy_(saved)
        opts = _real_optimizers(optimizer)
        flats = [[p for group in opt.param_groups for p in group["params"]] for opt in opts]
        for (opt_index, index), saved_state in snapshot["moments"].items():
            if opt_index >= len(opts) or index >= len(flats[opt_index]):
                continue
            state = opts[opt_index].state.get(flats[opt_index][index])
            if not state:
                continue
            for key, saved in saved_state.items():
                if key in state and isinstance(state[key], torch.Tensor):
                    state[key].copy_(saved)
        # Everything newer than the restored point came after the poisoned step, so it is
        # not a safe place to rewind to a second time.
        self._ring.clear()
        return True

    def _scale_lr(self, optimizer: torch.optim.Optimizer, scheduler: Any, factor: float) -> None:
        """Back the learning rate off by rescaling the scheduler's ``base_lrs``.

        Writing ``param_group["lr"]`` directly would not survive: ``LambdaLR`` recomputes
        ``lr = base_lrs[i] * lr_lambda(step)`` on every ``scheduler.step()``, so a direct
        write is discarded on the very next iteration. ``base_lrs`` is the input to that
        product and therefore the only durable place to apply the backoff.
        """
        schedulers = _real_schedulers(scheduler)
        if self._base_lrs is None:
            if schedulers:
                self._base_lrs = [list(sched.base_lrs) for sched in schedulers]
            else:
                self._base_lrs = [
                    [group["lr"] for group in opt.param_groups] for opt in _real_optimizers(optimizer)
                ]
        self._lr_scale = max(self.lr_min_scale, min(1.0, self._lr_scale * factor))

        if schedulers:
            for sched, bases in zip(schedulers, self._base_lrs):
                sched.base_lrs = [base * self._lr_scale for base in bases]
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
            if self._lr_scale < 1.0:
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
            self._scale_lr(optimizer, scheduler, self.lr_backoff)
            self._norms.clear()
            log.warning(
                f"loss-spike guard: rolled back at iteration {iteration} ({reason}); "
                f"learning rate scaled to {self._lr_scale:.3f} of base [rollback #{self.rollbacks}]"
            )
        else:
            log.warning(
                f"loss-spike guard: spike at iteration {iteration} ({reason}) but no snapshot is "
                "available yet; letting the step stand"
            )
