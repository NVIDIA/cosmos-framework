# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Callback that saves extra checkpoints on a wall-clock schedule.

Motivation
----------

``checkpoint.save_iter`` bounds checkpoint spacing in *iterations*, which
translates to a wildly varying amount of wall-clock time depending on model
size, resolution and node count.  This callback adds a second, time-based
trigger so that no more than ``interval_minutes`` of training is ever lost,
independent of how slow an iteration happens to be.  It supplements rather
than replaces ``save_iter``: whichever comes first wins.

Rank agreement
--------------

DCP saves are collective, so every rank must reach the same conclusion on
every step.  Each rank's own clock drifts, so comparing per-rank elapsed time
would eventually let ranks disagree and deadlock the job.  Rank 0 therefore
owns the timer and broadcasts its decision.

That collective is throttled to at most every ``check_every_n`` iterations,
but the stride shrinks as the deadline approaches: rank 0 divides the time
still owed by its measured step time and broadcasts the result, so a job at
minutes per iteration ends up checking every step rather than sailing hours
past the interval.  What remains is a one-iteration overshoot, because a
checkpoint can only be taken at a step boundary.

What the interval bounds
-----------------------

The clock starts when a save *starts*, not when it lands, so the interval bounds
how often a save is *initiated*: at most ``interval_minutes`` plus the iteration
in flight.  Durability lags that by however long the background write takes, and
that lag adds to work lost in a crash -- a save begun at minute 30 and still
writing when the job dies leaves the previous checkpoint as the resume point.
The lag is not charged against the cadence, though: ``on_save_checkpoint_success``
deliberately does not touch the timer, so a slow write cannot push the next save
further out.

Waiting for durability instead would mean blocking the training step on the write,
which is what asynchronous checkpointing exists to avoid.  What is guaranteed is
that writes never overlap: ``save()`` opens by waiting out the previous async
write, so a save triggered while one is still in flight -- during a filesystem
outage, say -- stalls the step rather than starting a second concurrent write, and
a previous write that failed ends the job there.

Retention
---------

Extra checkpoints would otherwise pile up in the experiments bucket, so this
callback prunes its own saves down to the last ``keep_last``.  It only ever
deletes checkpoints it created (identified by a marker object it writes) and
never a ``save_iter`` milestone, so tooling that assumes checkpoints persist
-- periodic generation, evaluations pinned to an iteration, retroactive
promotion to the permanent bucket -- is unaffected.

Deletion is asynchronous out of necessity, not as an optimization.
``on_save_checkpoint_success`` is dispatched from
``_wait_for_previous_async_checkpoint()``, which runs on the main thread at
the top of the *next* ``save()``, so any blocking work there stalls the
training step.  Removing a checkpoint means thousands of individual object
deletions, so it is handed to a background thread on rank 0.

Because that thread is a daemon, an interpreter exit (notably the
``sys.exit(0)`` in the SIGUSR1 preemption path) can abandon a deletion
midway.  A ``.deleting`` marker is therefore written before the first object
is removed, and any checkpoint carrying it is re-queued on the next startup,
which makes deletion idempotent and resumable rather than leaving a partial
directory behind forever.
"""

from __future__ import annotations

import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import torch

from cosmos_framework.model._base import ImaginaireModel
from cosmos_framework.utils import distributed, log
from cosmos_framework.utils.callback import Callback
from cosmos_framework.utils.easy_io import easy_io

#: Set by ``get_executor_gcp()`` so that submitting to the enforced clusters turns
#: this on without every experiment config having to opt in.
INTERVAL_MINUTES_ENV_VAR = "COSMOS3_CHECKPOINT_WALL_CLOCK_MINUTES"

#: Written inside each checkpoint this callback creates, so it can recognize its
#: own saves after a restart and never delete anyone else's.
SAVED_MARKER = "wall_clock_checkpoint.json"

#: Written before deletion starts so an interrupted deletion can be resumed.
DELETING_MARKER = "wall_clock_deleting.json"


class WallClockCheckpoint(Callback):
    """Save a checkpoint whenever ``interval_minutes`` of wall-clock time elapses.

    Args:
        interval_minutes: Minutes between wall-clock saves.  ``None`` reads
            ``COSMOS3_CHECKPOINT_WALL_CLOCK_MINUTES`` from the environment;
            a missing or non-positive value disables the callback entirely.
        keep_last: How many wall-clock checkpoints to retain.  Older ones are
            deleted; ``save_iter`` milestones are never touched.  Non-positive
            disables pruning.
        check_every_n: Upper bound on how many optimizer steps may pass between
            timer evaluations, which bounds the cost of the rank-agreement
            collective.  The stride falls to 1 near the deadline, so this only
            takes effect while a save is still far off.
        min_iterations: Minimum iterations since the last checkpoint of any
            kind before a wall-clock save is allowed.
        delete_concurrency: Parallel object deletions per checkpoint removal.
    """

    def __init__(
        self,
        interval_minutes: float | None = None,
        keep_last: int = 3,
        check_every_n: int = 10,
        min_iterations: int = 1,
        delete_concurrency: int = 8,
    ) -> None:
        super().__init__()
        self._interval_seconds = self._resolve_interval_minutes(interval_minutes) * 60.0
        self._keep_last = keep_last
        self._check_every_n = max(1, check_every_n)
        self._min_iterations = max(1, min_iterations)
        self._delete_concurrency = max(1, delete_concurrency)

        self._last_checkpoint_time = time.monotonic()
        self._last_checkpoint_iteration = 0
        # Next iteration at which the timer may be evaluated. Broadcast from rank 0
        # rather than derived locally, so every rank reaches the collective together.
        self._next_check_iteration = 0
        # Rank 0's smoothed step duration, used to convert time remaining into a stride.
        self._previous_step_end: float | None = None
        self._step_seconds: float | None = None
        # Iterations this callback asked for but whose save has not been confirmed yet.
        self._triggered_iterations: set[int] = set()
        # Confirmed wall-clock checkpoints, ascending. Only maintained on rank 0,
        # which is the only rank that prunes.
        self._wall_clock_iterations: list[int] = []

        # Captured from on_before_optimizer_step so we can call checkpointer.save().
        self._optimizer: torch.optim.Optimizer | None = None
        self._scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self._grad_scaler: torch.amp.GradScaler | None = None

        self._delete_queue: queue.Queue[int] = queue.Queue()
        self._delete_thread: threading.Thread | None = None

    @staticmethod
    def _resolve_interval_minutes(interval_minutes: float | None) -> float:
        if interval_minutes is not None:
            return max(0.0, interval_minutes)
        raw = os.environ.get(INTERVAL_MINUTES_ENV_VAR, "")
        if not raw:
            return 0.0
        try:
            return max(0.0, float(raw))
        except ValueError:
            log.warning(
                f"[WallClockCheckpoint] Ignoring unparseable {INTERVAL_MINUTES_ENV_VAR}={raw!r}; "
                "wall-clock checkpointing is disabled."
            )
            return 0.0

    @property
    def enabled(self) -> bool:
        return self._interval_seconds > 0.0

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def on_train_start(self, model: ImaginaireModel, iteration: int = 0) -> None:
        del model
        self._last_checkpoint_time = time.monotonic()
        self._last_checkpoint_iteration = iteration
        self._previous_step_end = None
        if not self.enabled:
            log.info(f"[WallClockCheckpoint] Disabled; set {INTERVAL_MINUTES_ENV_VAR} or interval_minutes to enable.")
            return
        log.info(
            f"[WallClockCheckpoint] Enabled: saving at least every {self._interval_seconds / 60:.1f} minutes, "
            f"retaining the last {self._keep_last} wall-clock checkpoints."
        )
        if distributed.is_rank0():
            self._recover_tracked_checkpoints()

    def on_before_optimizer_step(
        self,
        model: ImaginaireModel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        grad_scaler: torch.amp.GradScaler,
        iteration: int = 0,
    ) -> None:
        del model, iteration
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._grad_scaler = grad_scaler

    def on_save_checkpoint_start(self, model: ImaginaireModel, iteration: int = 0) -> None:
        del model
        # Any save resets the clock, so the guarantee is "at most interval_minutes
        # since the last checkpoint" regardless of which trigger produced it. This
        # runs before on_training_step_end for periodic saves, which is what keeps
        # us from saving twice on the same iteration.
        self._last_checkpoint_time = time.monotonic()
        self._last_checkpoint_iteration = iteration

    def on_save_checkpoint_success(self, iteration: int = 0, elapsed_time: float = 0) -> None:
        del elapsed_time
        if iteration not in self._triggered_iterations:
            return
        self._triggered_iterations.discard(iteration)
        if not distributed.is_rank0():
            return
        if self._mark_saved(iteration):
            self._wall_clock_iterations.append(iteration)
            self._prune()

    def on_training_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        del data_batch, output_batch, loss
        if not self.enabled:
            return
        self._observe_step_duration()
        if iteration < self._next_check_iteration:
            return
        # Evaluated before the collective rather than inside it: every rank sees the
        # same iteration and the same _last_checkpoint_iteration, so this cannot let
        # ranks disagree, and there is no point paying for a broadcast whose answer
        # we would then discard.
        if iteration - self._last_checkpoint_iteration < self._min_iterations:
            self._next_check_iteration = iteration + 1
            return
        elapsed = time.monotonic() - self._last_checkpoint_time
        should_save, stride = self._agree_on_save(elapsed >= self._interval_seconds, self._next_stride(elapsed))
        self._next_check_iteration = iteration + stride
        if not should_save:
            return
        assert self._optimizer is not None, (
            "[WallClockCheckpoint] Optimizer reference not set — on_before_optimizer_step was never called"
        )
        log.info(
            f"[WallClockCheckpoint] {elapsed / 60:.1f} minutes since the last checkpoint; "
            f"saving at iteration {iteration}."
        )
        self._triggered_iterations.add(iteration)
        # Deliberately no finalize() here: training continues and the async DCP
        # write should stay overlapped with it.
        self.trainer.checkpointer.save(model, self._optimizer, self._scheduler, self._grad_scaler, iteration=iteration)

    def _agree_on_save(self, should_save: bool, stride: int) -> tuple[bool, int]:
        """Replace every rank's answer with rank 0's, so the collective save stays in step.

        The stride travels in the same broadcast because it is derived from rank 0's
        step-time estimate, and a rank gating on its own estimate would arrive at this
        collective on a different iteration than everyone else.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        decision = torch.tensor([1 if should_save else 0, stride], dtype=torch.int64, device=device)
        distributed.broadcast(decision, src=0)
        return bool(decision[0].item()), int(decision[1].item())

    def _next_stride(self, elapsed: float) -> int:
        """How many iterations may pass before the timer is evaluated again.

        A flat ``check_every_n`` stride costs a slow job up to that many iterations of
        overshoot -- at minutes per iteration, hours past the interval it was supposed
        to enforce. Dividing the time still owed by the measured step time collapses
        the stride to 1 as the deadline nears, which is as tight as this can get: a
        checkpoint is only possible at a step boundary.

        Until a step has been timed, or once the interval has already elapsed, check
        every iteration. A step time that changes abruptly is absorbed at the next
        evaluation, so the worst case reverts to ``check_every_n`` iterations.
        """
        remaining = self._interval_seconds - elapsed
        if remaining <= 0.0 or self._step_seconds is None or self._step_seconds <= 0.0:
            return 1
        return max(1, min(self._check_every_n, int(remaining / self._step_seconds)))

    def _observe_step_duration(self) -> None:
        """Smooth the observed step time, which is what makes the stride adaptive."""
        now = time.monotonic()
        previous, self._previous_step_end = self._previous_step_end, now
        if previous is None:
            return
        sample = now - previous
        # Smoothed because step time is noisy -- data stalls, resolution changes -- and
        # one anomalously fast step should not stretch the stride past the deadline.
        self._step_seconds = sample if self._step_seconds is None else 0.8 * self._step_seconds + 0.2 * sample

    # ------------------------------------------------------------------
    # Retention
    # ------------------------------------------------------------------

    def _checkpoint_dirname(self, iteration: int) -> str:
        return os.path.join(self.trainer.checkpointer.save_dirname, f"iter_{iteration:09}")

    @property
    def _backend_key(self) -> str | None:
        return self.trainer.checkpointer.save_s3_backend_key

    def _mark_saved(self, iteration: int) -> bool:
        """Tag a checkpoint as ours. Returns False if the tag could not be written."""
        path = os.path.join(self._checkpoint_dirname(iteration), SAVED_MARKER)
        try:
            easy_io.dump({"iteration": iteration, "saved_at": time.time()}, path, backend_key=self._backend_key)
        except Exception as error:  # noqa: BLE001 - a missing tag only costs us pruning
            log.warning(f"[WallClockCheckpoint] Could not write {path}: {error}. This checkpoint will not be pruned.")
            return False
        return True

    def _prune(self) -> None:
        if self._keep_last <= 0:
            return
        while len(self._wall_clock_iterations) > self._keep_last:
            self._enqueue_delete(self._wall_clock_iterations.pop(0))

    def _enqueue_delete(self, iteration: int) -> None:
        if self._delete_thread is None:
            self._delete_thread = threading.Thread(
                target=self._delete_worker,
                name="wall-clock-ckpt-delete",
                daemon=True,
            )
            self._delete_thread.start()
        self._delete_queue.put(iteration)
        log.info(
            f"[WallClockCheckpoint] Queued iteration {iteration} for deletion "
            f"(queue depth {self._delete_queue.qsize()})."
        )

    def _delete_worker(self) -> None:
        while True:
            iteration = self._delete_queue.get()
            try:
                self._delete_checkpoint(iteration)
            except Exception as error:  # noqa: BLE001 - deletion must never take down training
                log.error(f"[WallClockCheckpoint] Failed to delete checkpoint at iteration {iteration}: {error}")
            finally:
                self._delete_queue.task_done()

    def _delete_checkpoint(self, iteration: int) -> None:
        dirname = self._checkpoint_dirname(iteration)
        if not self._safe_to_delete(iteration, dirname):
            return
        start_time = time.monotonic()
        deleting_marker = os.path.join(dirname, DELETING_MARKER)
        # Record the intent before removing anything, and tear it down last, so that
        # an interrupted deletion always leaves behind the one marker that lets the
        # next startup recognize the directory and finish the job.
        easy_io.dump({"iteration": iteration}, deleting_marker, backend_key=self._backend_key)
        paths = [
            os.path.join(dirname, relative_path)
            for relative_path in easy_io.list_dir_or_file(
                dirname, list_dir=False, list_file=True, recursive=True, backend_key=self._backend_key
            )
        ]
        paths = [path for path in paths if path != deleting_marker]
        with ThreadPoolExecutor(max_workers=self._delete_concurrency) as pool:
            list(pool.map(self._delete_object, paths))
        swept = self._sweep_subdirectories(dirname)
        self._delete_object(deleting_marker)
        # Nothing but an empty directory is left; drop it so that anything scanning for
        # iter_* directories does not mistake the husk for a checkpoint.
        self._rmtree(dirname)
        log.info(
            f"[WallClockCheckpoint] Deleted wall-clock checkpoint {dirname} "
            f"({len(paths) + 1} objects, {swept} subtrees in {time.monotonic() - start_time:.1f}s)."
        )

    def _sweep_subdirectories(self, dirname: str) -> int:
        """Remove the checkpoint's subtrees wholesale, returning how many were swept.

        The listing above leaves two kinds of residue on a filesystem backend: it omits
        dotfiles, so DCP's per-directory ``.metadata`` survives it, and it never reports
        the directories themselves. An object store has neither, so its subtrees are
        already gone and this is a no-op. Sweeping before the deleting marker comes down
        preserves the invariant that payload never outlives its marker.
        """
        swept = 0
        for name in self._list_subdirectories(dirname):
            if self._rmtree(os.path.join(dirname, name)):
                swept += 1
        return swept

    def _list_subdirectories(self, dirname: str) -> list[str]:
        try:
            return [name.rstrip("/") for name in easy_io.list_dir(dirname, backend_key=self._backend_key)]
        except Exception as error:  # noqa: BLE001 - an unlistable directory only costs us the sweep
            log.warning(f"[WallClockCheckpoint] Could not list subdirectories of {dirname}: {error}")
            return []

    def _rmtree(self, path: str) -> bool:
        try:
            easy_io.rmtree(path, backend_key=self._backend_key)
        except Exception as error:  # noqa: BLE001 - keep deleting the rest of the checkpoint
            log.warning(f"[WallClockCheckpoint] Could not remove {path}: {error}")
            return False
        return True

    def _delete_object(self, path: str) -> None:
        try:
            easy_io.remove(path, backend_key=self._backend_key)
        except Exception as error:  # noqa: BLE001 - keep deleting the rest of the checkpoint
            log.warning(f"[WallClockCheckpoint] Could not remove {path}: {error}")

    def _safe_to_delete(self, iteration: int, dirname: str) -> bool:
        """Refuse anything that is not unambiguously an expendable wall-clock save."""
        save_iter = self.config.checkpoint.save_iter
        if save_iter > 0 and iteration % save_iter == 0:
            log.warning(f"[WallClockCheckpoint] Refusing to delete iteration {iteration}: it is a save_iter milestone.")
            return False
        latest = self.trainer.checkpointer._read_latest_checkpoint_file()
        if latest is not None and latest.strip() == f"iter_{iteration:09}":
            log.info(f"[WallClockCheckpoint] Skipping iteration {iteration}: it is the checkpoint to resume from.")
            return False
        # Either marker proves the checkpoint is ours; a resumed deletion may have
        # already removed the saved marker along with the rest of the payload.
        saved, deleting = self._read_markers(iteration)
        if not (saved or deleting):
            log.warning(f"[WallClockCheckpoint] Refusing to delete {dirname}: not tagged as a wall-clock checkpoint.")
            return False
        return True

    def _recover_tracked_checkpoints(self) -> None:
        """Rediscover our own checkpoints after a restart, and finish interrupted deletions.

        Without this, wall-clock checkpoints orphaned by a preemption would never be
        pruned and would accumulate one restart at a time.
        """
        save_dirname = self.trainer.checkpointer.save_dirname
        try:
            # list_dir rather than list_dir_or_file: the object-store backend rejects
            # the directories-only, non-recursive combination outright.
            names = [name.rstrip("/") for name in easy_io.list_dir(save_dirname, backend_key=self._backend_key)]
            iterations = sorted(
                int(name[len("iter_") :])
                for name in names
                if name.startswith("iter_") and name[len("iter_") :].isdigit()
            )
        except Exception as error:  # noqa: BLE001 - a failed scan only costs us pruning
            log.warning(f"[WallClockCheckpoint] Could not list {save_dirname}: {error}. Starting with no history.")
            return
        if not iterations:
            return

        with ThreadPoolExecutor(max_workers=self._delete_concurrency) as pool:
            states = list(pool.map(self._read_markers, iterations))

        interrupted = [iteration for iteration, (_, deleting) in zip(iterations, states) if deleting]
        self._wall_clock_iterations = [
            iteration for iteration, (saved, deleting) in zip(iterations, states) if saved and not deleting
        ]
        log.info(
            f"[WallClockCheckpoint] Recovered {len(self._wall_clock_iterations)} wall-clock checkpoints "
            f"and {len(interrupted)} interrupted deletions from {save_dirname}."
        )
        for iteration in interrupted:
            self._enqueue_delete(iteration)
        self._prune()

    def _read_markers(self, iteration: int) -> tuple[bool, bool]:
        """Return whether the checkpoint is ours and whether its deletion was interrupted."""
        dirname = self._checkpoint_dirname(iteration)
        try:
            saved = easy_io.exists(os.path.join(dirname, SAVED_MARKER), backend_key=self._backend_key)
            deleting = easy_io.exists(os.path.join(dirname, DELETING_MARKER), backend_key=self._backend_key)
        except Exception as error:  # noqa: BLE001 - treat an unreadable checkpoint as not ours
            log.warning(f"[WallClockCheckpoint] Could not inspect {dirname}: {error}")
            return False, False
        return saved, deleting
