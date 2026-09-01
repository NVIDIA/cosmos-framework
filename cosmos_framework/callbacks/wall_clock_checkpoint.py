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
past the interval.  What remains is normally a one-iteration overshoot, because
a checkpoint can only be taken at a step boundary.  With context parallelism,
the save may wait a few more steps until the trainer reaches the next CP
data-window boundary, so dataloader sampler state is checkpointed safely.

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

Because that thread is a daemon, an interpreter exit can abandon a deletion
midway.  A ``.deleting`` marker is therefore written before the first object
is removed, and any checkpoint carrying it is re-queued on the next startup,
which makes deletion idempotent and resumable rather than leaving a partial
directory behind forever.

Completeness
------------

Retention would otherwise delete on the strength of a *reported* success:
``on_save_checkpoint_success`` fires when the background writer says it is
done, and nothing else in the checkpoint path ever reads a checkpoint back to
confirm it is loadable.  Had the object store silently truncated a write the
writer believed it completed, pruning would discard a good checkpoint in
favour of a bad one.

So before an expired checkpoint is removed, the checkpoint that displaced it
is checked two ways.  First, each component must have its ``.metadata``, which
DCP writes only once that component's shards are all in.  Second, every shard
must be at least as long as that metadata says it is: ``storage_data`` records
a file, an offset and a length for every shard written, so the high-water mark
of ``offset + length`` is the last byte the writer claims to have put in each
file, and a ``HEAD`` gives the byte count actually stored.  A failed check
defers the deletion rather than cancelling it: the retained window grows by one
instead of advancing over a checkpoint that may not load, and the next prune
retries.

A ``.metadata`` that will not parse defers too.  It was written by the same DCP
runtime minutes earlier, so it cannot plausibly be a format from the future,
and a corrupt or truncated one is exactly the damage being looked for.

The second check is what covers a shard whose write stopped partway.  The first
cannot see that, because ``.metadata`` is a separate, later object -- its
presence says the writer got to the end of the component, not that every byte
before it arrived.

Neither check reads shard contents, so neither can see corruption *within* a
shard: bytes that are present and the right length but wrong.  Detecting that
would mean reading every byte back on every save, and it would buy little here.
A digest computed while writing cannot catch a source buffer that was already
wrong, TLS covers the wire, and the object store checksums its own bytes at
rest.  Truncation is the failure mode left over, and it is the one this sees.

The cost is one ``HEAD`` per shard plus the metadata read itself, on the rank-0
deletion thread, so the training step pays nothing for it.  The metadata read is
the only part that is not negligible: ``storage_data`` carries an entry per
shard, so on the largest configurations that object is big enough to be worth
watching.
"""

from __future__ import annotations

import bisect
import os
import pickle
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import torch

from cosmos_framework.model._base import ImaginaireModel
from cosmos_framework.utils import distributed, log
from cosmos_framework.utils.callback import Callback
from cosmos_framework.utils.easy_io import easy_io

#: Set by ``get_executor_gcp()`` for every cluster it serves, so that submitting turns
#: this on without every experiment config having to opt in.
INTERVAL_MINUTES_ENV_VAR = "COSMOS3_CHECKPOINT_WALL_CLOCK_MINUTES"

#: Written inside each checkpoint this callback creates, so it can recognize its
#: own saves after a restart and never delete anyone else's.
SAVED_MARKER = "wall_clock_checkpoint.json"

#: Written before deletion starts so an interrupted deletion can be resumed.
DELETING_MARKER = "wall_clock_deleting.json"

#: The state a resume needs, one DCP component per subdirectory.  ``dataloader/`` is
#: left out: it is optional and written as pickles rather than by DCP, so it has no
#: ``.metadata`` and its absence says nothing about whether the save finished.
DCP_COMPONENTS = ("model", "optim", "scheduler", "trainer")

#: Written by DCP once a component's shards are all in, which makes it the closest
#: thing a checkpoint has to a completion marker.
DCP_METADATA = ".metadata"

#: How many short shards to name individually before summarizing the remainder, so a
#: component that lost all of them reports the scale rather than flooding the log.
SHORT_SHARD_REPORT_LIMIT = 5


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

        # Each entry pairs a checkpoint to remove with the one that displaced it, whose
        # completeness the removal is conditional on. ``None`` waives the condition.
        self._delete_queue: queue.Queue[tuple[int, int | None]] = queue.Queue()
        self._delete_thread: threading.Thread | None = None
        # Deletions the completeness probe declined, waiting to be taken back into the
        # history above. Routed through a queue rather than written back directly so that
        # the history stays owned by the one thread that prunes it.
        self._deferred_deletions: queue.Queue[int] = queue.Queue()

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
        if self._cp_data_window_active():
            self._next_check_iteration = iteration + 1
            log.info(
                f"[WallClockCheckpoint] Deferring checkpoint at iteration {iteration}: "
                "context-parallel data window is active."
            )
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

    def _cp_data_window_active(self) -> bool:
        """Return whether saving now would land inside an unfinished CP data window."""
        cp_data_window = getattr(self.trainer, "_cp_data_window", None)
        if cp_data_window is None:
            return False
        return bool(getattr(cp_data_window, "active", False)) or int(getattr(cp_data_window, "offset", 0)) != 0

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
        self._readmit_deferred()
        while len(self._wall_clock_iterations) > self._keep_last:
            expired = self._wall_clock_iterations.pop(0)
            # The newest checkpoint is what pushed this one out of the window, and it is
            # unaffected by popping from the front, so it survives a multi-step catch-up.
            self._enqueue_delete(expired, displaced_by=self._wall_clock_iterations[-1])

    def _readmit_deferred(self) -> None:
        """Take back the deletions the completeness probe declined, so this prune retries them.

        Each one is re-inserted in order and then immediately reconsidered against the
        checkpoint that has since arrived, which is how retention catches up as soon as a
        save turns up complete. Until then the window simply holds one more checkpoint.
        """
        while True:
            try:
                iteration = self._deferred_deletions.get_nowait()
            except queue.Empty:
                return
            bisect.insort(self._wall_clock_iterations, iteration)

    def _enqueue_delete(self, iteration: int, displaced_by: int | None = None) -> None:
        if self._delete_thread is None:
            self._delete_thread = threading.Thread(
                target=self._delete_worker,
                name="wall-clock-ckpt-delete",
                daemon=True,
            )
            self._delete_thread.start()
        self._delete_queue.put((iteration, displaced_by))
        log.info(
            f"[WallClockCheckpoint] Queued iteration {iteration} for deletion "
            f"(queue depth {self._delete_queue.qsize()})."
        )

    def _delete_worker(self) -> None:
        while True:
            iteration, displaced_by = self._delete_queue.get()
            try:
                self._delete_checkpoint(iteration, displaced_by)
            except Exception as error:  # noqa: BLE001 - deletion must never take down training
                log.error(f"[WallClockCheckpoint] Failed to delete checkpoint at iteration {iteration}: {error}")
            finally:
                self._delete_queue.task_done()

    def _delete_checkpoint(self, iteration: int, displaced_by: int | None = None) -> None:
        dirname = self._checkpoint_dirname(iteration)
        if not self._safe_to_delete(iteration, dirname):
            return
        # Probed here rather than at enqueue time because this is the deletion thread:
        # the same reason the deletion itself is not done on the training step.
        if displaced_by is not None and not self._verify_complete(displaced_by):
            log.error(
                f"[WallClockCheckpoint] Keeping {dirname}: iteration {displaced_by}, which displaced it, "
                "does not look complete. Retention holds one extra checkpoint rather than advancing "
                "over one that may not load, and will retry after the next save."
            )
            self._deferred_deletions.put(iteration)
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

    def _verify_complete(self, iteration: int) -> bool:
        """Report whether a checkpoint carries its metadata and shards of the stated length.

        This is the difference between a save that was *reported* successful and one that
        left something loadable behind. It stops short of being an integrity check: a
        shard of the right length holding the wrong bytes still passes, since proving
        otherwise means reading the checkpoint back. What it catches is truncation, at
        either granularity -- a component that never reached its metadata, or a shard that
        stopped partway -- which is the shape of failure that would have us delete the
        better of two checkpoints.

        A probe that cannot be answered counts as incomplete, so a blip in the object
        store costs a retained checkpoint rather than the one we were about to keep.
        """
        dirname = self._checkpoint_dirname(iteration)
        for component in DCP_COMPONENTS:
            component_dirname = os.path.join(dirname, component)
            path = os.path.join(component_dirname, DCP_METADATA)
            try:
                if not easy_io.exists(path, backend_key=self._backend_key):
                    log.error(f"[WallClockCheckpoint] {path} is missing; iteration {iteration} may not load.")
                    return False
            except Exception as error:  # noqa: BLE001 - an unanswerable probe is not a pass
                log.warning(f"[WallClockCheckpoint] Could not probe {path}: {error}")
                return False
            if not self._verify_shard_sizes(component_dirname):
                log.error(f"[WallClockCheckpoint] {component_dirname} is short; iteration {iteration} may not load.")
                return False
        return True

    def _verify_shard_sizes(self, component_dirname: str) -> bool:
        """Check every shard is at least as long as this component's metadata says.

        Sizes come from ``HEAD`` requests, so nothing is downloaded but the metadata. Only
        short files count as damage: a longer one is not a truncation, and padding a writer
        chose to add is none of our business.

        Metadata this cannot interpret is damage rather than an exemption, which is the
        distinction between the two early returns below; see ``_expected_shard_sizes``.
        """
        metadata_path = os.path.join(component_dirname, DCP_METADATA)
        try:
            payload = easy_io.get(metadata_path, backend_key=self._backend_key)
        except Exception as error:  # noqa: BLE001 - transient, and the next prune retries
            log.warning(f"[WallClockCheckpoint] Could not read {metadata_path}: {error}")
            return False
        expected = self._expected_shard_sizes(payload, metadata_path)
        if expected is None:
            return False
        if not expected:
            return True
        relative_paths = sorted(expected)
        # ``map`` submits every request before the first result can be read, so there is no
        # early exit to be had inside the pool: leaving the block early would still await
        # the requests already in flight. The sizes are therefore gathered inside the pool
        # and judged outside it, which also lets a component that lost many shards be
        # reported as the single systemic failure it is rather than as its first casualty.
        with ThreadPoolExecutor(max_workers=self._delete_concurrency) as pool:
            paths = (os.path.join(component_dirname, name) for name in relative_paths)
            sizes = list(pool.map(self._object_size, paths))
        short: list[tuple[str, int]] = []
        for relative_path, size in zip(relative_paths, sizes):
            if size is None:
                return False
            if size < expected[relative_path]:
                short.append((relative_path, size))
        if not short:
            return True
        for relative_path, size in short[:SHORT_SHARD_REPORT_LIMIT]:
            log.error(
                f"[WallClockCheckpoint] {os.path.join(component_dirname, relative_path)} holds {size} bytes "
                f"but its metadata accounts for {expected[relative_path]}; the write was cut short."
            )
        if len(short) > SHORT_SHARD_REPORT_LIMIT:
            log.error(
                f"[WallClockCheckpoint] ...and {len(short) - SHORT_SHARD_REPORT_LIMIT} further short shards under "
                f"{component_dirname}, of {len(relative_paths)} checked."
            )
        return False

    @staticmethod
    def _expected_shard_sizes(payload: bytes, metadata_path: str) -> dict[str, int] | None:
        """The byte length DCP's metadata implies for each shard file it wrote.

        ``None`` means the payload could not be interpreted, which counts as damage. The
        metadata was written by the same DCP runtime minutes earlier, so a payload that
        will not parse is evidence the save did not land rather than evidence of a format
        from the future -- and a corrupt or truncated ``.metadata`` is exactly the damage
        this is looking for, so reading it as "nothing to check" would clear the successor
        precisely when it is least trustworthy.

        An empty mapping is the separate, benign case: metadata that parsed and recorded
        no shards, which leaves nothing for the size check to compare.

        Each layout is therefore recognized explicitly -- a ``str`` path with a
        non-negative ``int`` offset and length -- and anything else defers. If a future
        format needs supporting, teach this function that format rather than widening what
        counts as acceptable.
        """
        try:
            # Plain pickle rather than a reader: this mirrors how DCP's own
            # FileSystemReader.read_metadata loads the file, without needing a
            # CheckpointLoadSource built to point back at the save location.
            metadata = pickle.loads(payload)
        except Exception as error:  # noqa: BLE001 - a payload that will not parse is damage
            log.error(f"[WallClockCheckpoint] Could not unpickle {metadata_path}: {error}.")
            return None
        storage_data = getattr(metadata, "storage_data", None)
        if not isinstance(storage_data, dict):
            log.error(
                f"[WallClockCheckpoint] {metadata_path} carries no storage_data mapping "
                f"(found {type(storage_data).__name__})."
            )
            return None
        sizes: dict[str, int] = {}
        for info in storage_data.values():
            relative_path = getattr(info, "relative_path", None)
            offset = getattr(info, "offset", None)
            length = getattr(info, "length", None)
            if (
                not isinstance(relative_path, str)
                or not isinstance(offset, int)
                or not isinstance(length, int)
                # A negative byte range is not something DCP writes, and left alone it would
                # be worse than useless: the high-water mark below floors at zero, so one
                # negative entry would quietly reduce a shard's expected length to nothing
                # and pass any file at all.
                or offset < 0
                or length < 0
            ):
                log.error(
                    f"[WallClockCheckpoint] {metadata_path} holds a storage_data entry this cannot read: {info!r}."
                )
                return None
            sizes[relative_path] = max(sizes.get(relative_path, 0), offset + length)
        return sizes

    def _object_size(self, path: str) -> int | None:
        try:
            return easy_io.size(path, backend_key=self._backend_key)
        except Exception as error:  # noqa: BLE001 - an unanswerable probe is not a pass
            log.warning(f"[WallClockCheckpoint] Could not size {path}: {error}")
            return None

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
            # Unconditional: this checkpoint has already been partly dismantled, so there
            # is nothing left to preserve by holding it back.
            self._enqueue_delete(iteration, displaced_by=None)
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
