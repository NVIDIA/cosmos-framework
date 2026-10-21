# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""@L0 @CPU: the controllers' lifecycle observers (`control/lifecycle.py`) and OneLogger as one of them.

Checked without the `one_logger` wheel, against a recording stand-in for imaginaire's
`one_logger_utils` (and, for the timing contract, against the real `OneLoggerUtils` wrapper over a
fake `one_logger.core`). Pinned: the generic observer list never lets an observer fail the loop and
its bounded asynchronous dispatcher never blocks the controller (overflow drops and reports, start/end
pairs are admitted and dropped together so no end is ever orphaned, bounded idempotent shutdown, a
submit racing shutdown is queued ahead of the stop or counted as dropped, after a timed-out shutdown
events are dropped while the worker lives and inline once it exits, sync mode); `on_app_start`,
`on_train_end` and `on_app_end` are gates: stamped at the call, placed as non-droppable markers behind
every earlier event and delivered by the worker in emission order while the caller waits its bound, so
a full queue never loses a boundary (the reviewer's probe: park the first callback, fill the queue, call
`on_app_end`, release after 50 ms: app_end is delivered), a parked observer is never entered from a
second thread (the gates wait their bound, the controller returns, and they arrive in order once the
observer returns, OneLogger torn down last), and the terminal timestamps are the controller's clock at
the call, not the clock after the drain; every
queued event carries the controller's emission clocks and OneLogger's timers read them, so a slow
observer ahead in the queue never shortens a 200 ms workload to microseconds; the OneLogger config
follows the trainer's schema with imaginaire's timer semantics (one train step per optimizer update,
one train batch per forward/backward, per-replica sample counts); the real `grpo.loop` at n_mini>1 /
n_micro>1 emits the events in order and count; the checkpoint coordinators emit save events only for
actual saves (deduplicated cadence, validation, emergency paths); preemption and hook failures leave
no interval open and still tear OneLogger down; a raise in the loop's setup never leaves the train
interval open and a raise inside a micro call closes its batch and step from the emitter's own
`finally`; `opd.run` reports through the same observers with the teacher's GPUs in the world size; a
backend that fails to build is logged and left out, `grpo.run` / `opd.run` proceed with the empty
observer list, and both close the app (OneLogger finish/destroy) even when the checkpoint drain
raises, without masking that failure.
"""

from __future__ import annotations

import inspect
import sys
import threading
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from cosmos_framework.posttrain.callbacks.one_logger import (
    OneLoggerLifecycleObserver,
    build_one_logger_observer,
    controller_one_logger_config,
    job_environment,
    one_logger_enabled,
)
from cosmos_framework.posttrain.control import dispatch as dispatch_module
from cosmos_framework.posttrain.control import lifecycle as lifecycle_module
from cosmos_framework.posttrain.control.dispatch import DEFAULT_QUEUE_MAXSIZE, BoundedEventDispatcher, LifecycleEvent
from cosmos_framework.posttrain.control.lifecycle import (
    ControllerLifecycleObserver,
    ControllerLifecycleObservers,
    build_controller_observers,
)
from cosmos_framework.posttrain.tests.helpers.entry_loop import loop_config, mock_loop_env

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


# --------------------------------------------------------------------------- stand-ins
class _RecordingOneLogger:
    """Stand-in for imaginaire's `OneLoggerUtils`: records every hook, optionally raising in one."""

    def __init__(self, fail_on: str | None = None) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.fail_on = fail_on

    def __getattr__(self, name: str):
        if not name.startswith("on_") and name != "finish":
            raise AttributeError(name)

        def hook(**kwargs):
            if name == self.fail_on:
                raise RuntimeError(f"boom in {name}")
            self.calls.append((name, kwargs))

        return hook

    @property
    def names(self) -> list[str]:
        return [name for name, _ in self.calls]


def _fake_utils(recorder: _RecordingOneLogger) -> SimpleNamespace:
    """Stand-in for `cosmos_framework.utils.one_logger.one_logger_utils` (the global-object functions)."""
    state: dict[str, Any] = {"ol": None}

    def initialize_one_logger(config):
        assert state["ol"] is None
        state["config"] = config
        state["ol"] = recorder

    return SimpleNamespace(
        initialize_one_logger=initialize_one_logger,
        one_logger_is_initialized=lambda: state["ol"] is not None,
        get_one_logger=lambda: state["ol"],
        destroy_one_logger=lambda: state.update(ol=None),
        state=state,
    )


class _RecordingObserver(ControllerLifecycleObserver):
    """A generic observer that records the loop's events as plain tuples (and the delivering thread per event)."""

    def __init__(self) -> None:
        self.events: list[tuple] = []
        self.threads: list[str] = []  # the thread each recorded event was delivered on, in event order

    def observe(self, event: Any) -> None:
        self.threads.append(threading.current_thread().name)
        super().observe(event)

    def on_app_start(self) -> None:
        self.events.append(("app_start",))

    def on_train_start(self, *, train_iterations_start: int, train_samples_start: int) -> None:
        self.events.append(("train_start", train_iterations_start, train_samples_start))

    def on_train_step_start(self) -> None:
        self.events.append(("step_start",))

    def on_train_step_end(self) -> None:
        self.events.append(("step_end",))

    def on_train_batch_start(self) -> None:
        self.events.append(("batch_start",))

    def on_train_batch_end(self) -> None:
        self.events.append(("batch_end",))

    def on_validation_start(self) -> None:
        self.events.append(("validation_start",))

    def on_validation_end(self) -> None:
        self.events.append(("validation_end",))

    def on_checkpoint_save_start(self, model_step: int) -> None:
        self.events.append(("save_start", model_step))

    def on_checkpoint_save_end(self, model_step: int) -> None:
        self.events.append(("save_end", model_step))

    def on_checkpoint_save_success(self, model_step: int) -> None:
        self.events.append(("save_success", model_step))

    def on_train_end(self) -> None:
        self.events.append(("train_end",))

    def on_app_end(self) -> None:
        self.events.append(("app_end",))

    def names(self) -> list[str]:
        return [event[0] for event in self.events]


class _ParkingObserver(_RecordingObserver):
    """Records like `_RecordingObserver`, but the `park_on` hook blocks the delivering thread until released:
    `entered` is set when the worker is inside that hook, `release` lets it record and continue."""

    def __init__(self, park_on: str) -> None:
        super().__init__()
        self.park_on = park_on
        self.entered = threading.Event()
        self.release = threading.Event()

    def __getattribute__(self, name: str):
        attr = super().__getattribute__(name)
        if name.startswith("on_") and name == super().__getattribute__("park_on"):

            def hook(*args, **kwargs):
                self.entered.set()
                assert self.release.wait(10.0), "the test never released the parked observer"
                return attr(*args, **kwargs)

            return hook
        return attr


class _SerialObserver(_RecordingObserver):
    """Records like `_RecordingObserver` and watches how many threads are inside its hooks at once: `max_inside` is
    the peak and `overlaps` the (event, thread) pairs that entered while another hook was still running. Every hook
    holds the observer's own `hook_lock` (an observer guarding its state, the reviewer's probe), so a second thread
    entering it blocks until the first leaves. `park_on` parks that hook until `release` (`entered` is set when the
    worker is inside it, as in `_ParkingObserver`); `sleep_s` makes every hook that slow instead."""

    def __init__(self, *, park_on: str | None = None, sleep_s: float = 0.0) -> None:
        super().__init__()
        self.park_on, self.sleep_s = park_on, sleep_s
        self.entered, self.release = threading.Event(), threading.Event()
        self.hook_lock = threading.Lock()
        self.max_inside = 0
        self.overlaps: list[tuple[str, str]] = []
        self._inside, self._guard = 0, threading.Lock()

    def observe(self, event: Any) -> None:
        with self._guard:
            self._inside += 1
            self.max_inside = max(self.max_inside, self._inside)
            if self._inside > 1:
                self.overlaps.append((event.name, threading.current_thread().name))
        try:
            with self.hook_lock:
                if event.name == self.park_on:
                    self.entered.set()
                    assert self.release.wait(10.0), "the test never released the parked observer"
                elif self.sleep_s:
                    time.sleep(self.sleep_s)
                super().observe(event)
        finally:
            with self._guard:
                self._inside -= 1


class _RaisingObserver(ControllerLifecycleObserver):
    def __getattribute__(self, name: str):
        if name.startswith("on_"):

            def hook(*_args, **_kwargs):
                raise RuntimeError(f"observer failure in {name}")

            return hook
        return super().__getattribute__(name)


def _config(**overrides):
    kwargs = dict(
        run_name="cosmos_reason2/carla_pivot/grpo_run",
        controller="grpo",
        world_size=4,
        topology="POLICY_1x4_ROLLOUT_1x4_colocated",
        batch_per_replica=32,
        n_mini=2,
        n_micro=4,
        max_steps=300,
        policy_dp_world=4,
        seq_length=3072,
        environ={"SLURM_JOB_ID": "12345"},
        gpu="NVIDIA-GB300",
        app_start_time_ms=1_700_000_000_000,
    )
    kwargs.update(overrides)
    return controller_one_logger_config(**kwargs)


def _one_logger(rec: _RecordingOneLogger, **config_overrides) -> tuple[OneLoggerLifecycleObserver, SimpleNamespace]:
    fake = _fake_utils(rec)
    return OneLoggerLifecycleObserver(fake, _config(**config_overrides)), fake


_STEP = ["step_start", "batch_start", "batch_end", "batch_start", "batch_end", "step_end"]  # n_micro=2
_ONE_LOGGER_STEP = [
    "on_train_step_start",
    "on_train_batch_start",
    "on_train_batch_end",
    "on_train_batch_start",
    "on_train_batch_end",
    "on_train_step_end",
]


# --------------------------------------------------------------------------- config schema
def test_enable_flag_parses_like_the_slurm_executor_sets_it():
    assert one_logger_enabled({"ENABLE_ONELOGGER": "TRUE"}) and one_logger_enabled({"ENABLE_ONELOGGER": " true "})
    assert not one_logger_enabled({"ENABLE_ONELOGGER": "FALSE"}) and not one_logger_enabled({})
    assert job_environment({"SLURM_JOB_ID": "1"}) == "slurm" and job_environment({"NGC_JOB_ID": "1"}) == "ngc"


def test_config_matches_the_trainer_schema_with_imaginaire_timer_semantics():
    """One train iteration = one optimizer update (max_steps * n_mini); one train batch = one
    forward/backward, so global_batch_size is the trajectories one forward/backward consumes."""
    c = _config()
    assert c["enable_for_current_rank"] is True and c["one_logger_project"] == "imaginaire4"
    assert c["one_logger_run_name"] == c["app_tag_run_name"] == "cosmos_reason2/carla_pivot/grpo_run"
    assert c["world_size"] == 4
    assert c["train_iterations_target"] == 600  # 300 collected batches x n_mini=2 optimizer updates
    assert c["global_batch_size"] == 4  # 32 trajectories / (n_mini=2 x n_micro=4) per forward/backward
    assert c["batch_size"] == c["micro_batch_size"] == 1  # the DP-rank share of it (policy_dp_world=4)
    assert c["train_samples_target"] == 9600  # 300 x 32; consistent with 600 x 4 x 4 batch ends
    assert (
        c["seq_length"] == 3072 and c["save_checkpoint_strategy"] == "sync" and c["is_log_throughput_enabled"] is False
    )
    assert c["app_start_time"] == 1_700_000_000_000 and c["one_logger_async"] is True and c["barrier"] is None
    assert c["app_tag"] == (
        "cosmos_reason2/carla_pivot/grpo_run/grpo/BS_32_nMINI_2_nMICRO_4_nGPUs_4_POLICY_1x4_ROLLOUT_1x4_colocated"
        "/ENV_slurm_GPU_NVIDIA-GB300"
    )
    assert "seq_length" not in _config(seq_length=None)
    assert _config(environ={"ONE_LOGGER_ASYNC": "false", "SLURM_JOB_ID": "1"})["one_logger_async"] is False
    with pytest.raises(ValueError, match="not a multiple"):
        _config(batch_per_replica=30)


def test_sample_counts_are_per_replica_whatever_the_rollout_data_mesh():
    """rl.batch_per_replica is already the replica-global trajectory batch: four rollout data replicas
    (a 4x4 rollout mesh) do not multiply the samples, the global batch, or the target."""
    narrow = _config(batch_per_replica=64, n_mini=1, n_micro=1, policy_dp_world=1)
    wide = _config(
        batch_per_replica=64,
        n_mini=1,
        n_micro=1,
        policy_dp_world=1,
        world_size=48,
        topology="POLICY_1x32_ROLLOUT_4x4_disaggregated",
    )
    assert narrow["train_samples_target"] == wide["train_samples_target"] == 300 * 64
    assert narrow["global_batch_size"] == wide["global_batch_size"] == 64
    assert wide["world_size"] == 48 and "ROLLOUT_4x4" in wide["app_tag"]


# --------------------------------------------------------------------------- the generic observer list
def test_observer_list_fans_out_and_never_lets_one_observer_fail_the_loop():
    rec = _RecordingObserver()
    observers = ControllerLifecycleObservers([_RaisingObserver(), rec])
    observers.on_app_start()
    observers.on_train_start(train_iterations_start=3, train_samples_start=12)
    observers.on_train_step_start()
    observers.on_train_batch_start()
    observers.on_train_batch_end()
    observers.on_train_step_end()
    observers.on_checkpoint_save_start(4)
    observers.on_checkpoint_save_end(4)
    observers.on_checkpoint_save_success(4)
    observers.on_train_end()
    observers.on_app_end()
    assert rec.events == [
        ("app_start",),
        ("train_start", 3, 12),
        ("step_start",),
        ("batch_start",),
        ("batch_end",),
        ("step_end",),
        ("save_start", 4),
        ("save_end", 4),
        ("save_success", 4),
        ("train_end",),
        ("app_end",),
    ]
    # coerce: None is the silent empty list, a list passes through as itself, one observer is wrapped.
    assert len(ControllerLifecycleObservers.coerce(None)) == 0
    assert ControllerLifecycleObservers.coerce(observers) is observers
    assert list(ControllerLifecycleObservers.coerce(rec)) == [rec]
    assert list(ControllerLifecycleObservers.coerce([rec, rec])) == [rec, rec]
    ControllerLifecycleObserver().on_train_step_start()  # the null implementation is silent


# --------------------------------------------------------------------------- the bounded asynchronous dispatcher
def test_async_dispatch_never_blocks_the_controller_thread_and_keeps_the_order():
    """The emits return while the observer is parked inside an earlier event (the worker holds it, the
    controller thread does not wait); once released, every observer sees the events in emission order."""
    parked = _ParkingObserver("on_train_step_start")
    other = _RecordingObserver()
    observers = ControllerLifecycleObservers([parked, other])
    assert observers.dispatch == "async" and not observers.worker_alive, "the worker starts on the first event"

    t0 = time.monotonic()
    observers.on_train_start(train_iterations_start=1, train_samples_start=2)
    observers.on_train_step_start()
    assert parked.entered.wait(5.0), "the worker delivered the step start"
    for _ in range(3):
        observers.on_train_batch_start()
        observers.on_train_batch_end()
    observers.on_checkpoint_save_start(7)
    elapsed = time.monotonic() - t0
    assert elapsed < 0.2, f"nine emits took {elapsed:.3f} s with an observer parked: the controller waited"
    assert observers.worker_alive
    # Parked: the later events are queued, not delivered, to either observer (list order within an event).
    assert parked.events == [("train_start", 1, 2)] and other.events == [("train_start", 1, 2)]

    parked.release.set()
    assert observers.flush(timeout=5.0)
    expected = [("train_start", 1, 2), ("step_start",), *[("batch_start",), ("batch_end",)] * 3, ("save_start", 7)]
    assert parked.events == expected == other.events
    assert observers.dropped == 0
    assert observers.shutdown(timeout=5.0) and not observers.worker_alive


def test_async_dispatch_drops_on_overflow_and_reports_the_count(monkeypatch):
    """With the worker parked inside the first event, the queue takes DEFAULT_QUEUE_MAXSIZE more events (512
    complete batch start/end pairs: a start is admitted only with room for its end) and drops the rest without
    blocking: the drops are counted, logged on the first and the tenth, and the total is reported once at
    shutdown."""
    warnings: list[str] = []
    monkeypatch.setattr(dispatch_module.logger, "warning", warnings.append)
    parked = _ParkingObserver("on_train_start")
    observers = ControllerLifecycleObservers([parked])
    observers.on_train_start(train_iterations_start=0, train_samples_start=0)
    assert parked.entered.wait(5.0)

    t0 = time.monotonic()
    for _ in range(DEFAULT_QUEUE_MAXSIZE // 2):
        observers.on_train_batch_start()  # fills the queue behind the parked event, pair by pair
        observers.on_train_batch_end()
    assert observers.dropped == 0 and warnings == []
    for step in range(12):
        observers.on_checkpoint_save_success(step)  # the full queue drops these (no start of theirs was queued)
    assert time.monotonic() - t0 < 2.0, "1036 emits against a full queue returned without blocking"
    assert observers.dropped == 12 and observers.dropped_pairs == 0
    assert len(warnings) == 2, "the first drop and the tenth, not one line per drop"
    assert "dropped on_checkpoint_save_success" in warnings[0] and "; 1 dropped so far" in warnings[0]
    assert "; 10 dropped so far" in warnings[1]

    parked.release.set()
    assert observers.flush(timeout=10.0)
    names = parked.names()
    assert names.count("batch_start") == DEFAULT_QUEUE_MAXSIZE // 2 == names.count("batch_end")
    assert "save_success" not in names
    assert observers.shutdown(timeout=5.0)
    assert len(warnings) == 3 and "12 events were dropped over the run" in warnings[2]
    assert observers.shutdown(timeout=5.0) and len(warnings) == 3, "a repeated shutdown reports nothing twice"


def test_train_end_and_app_end_drain_the_queue_before_returning():
    """`on_train_end` and `on_app_end` are gates: the caller waits for the queue ahead of them, so a slow observer
    has recorded every earlier event when the call returns, and the gate itself is delivered by the worker right
    behind them (never dropped, never overtaken, never concurrent with a hook); app_end also stops the worker."""

    class _Slow(_RecordingObserver):
        def on_train_batch_end(self) -> None:
            time.sleep(0.02)
            super().on_train_batch_end()

    slow = _Slow()
    observers = ControllerLifecycleObservers([slow])
    observers.on_train_start(train_iterations_start=0, train_samples_start=0)
    for _ in range(5):
        observers.on_train_batch_start()
        observers.on_train_batch_end()
    observers.on_train_end()
    assert slow.names() == ["train_start", *["batch_start", "batch_end"] * 5, "train_end"]
    assert observers.worker_alive, "train_end drains but keeps the worker for the checkpoint drain"
    assert slow.threads == ["lifecycle-observers"] * 12, (
        "the interval events and the gate behind them: all on the worker"
    )

    observers.on_checkpoint_save_start(1)
    observers.on_checkpoint_save_end(1)
    observers.on_app_end()
    assert slow.names()[-3:] == ["save_start", "save_end", "app_end"]
    assert slow.threads[-3:] == ["lifecycle-observers"] * 3
    assert not observers.worker_alive and observers.dropped == 0


def test_shutdown_is_bounded_and_idempotent_and_later_events_deliver_inline(monkeypatch):
    """A shutdown against a parked worker returns False after its timeout (never hangs the controller),
    can be repeated, succeeds once the worker drains, and afterwards events are delivered inline."""
    warnings: list[str] = []
    monkeypatch.setattr(dispatch_module.logger, "warning", warnings.append)
    parked = _ParkingObserver("on_train_start")
    observers = ControllerLifecycleObservers([parked])
    observers.on_train_start(train_iterations_start=0, train_samples_start=0)
    assert parked.entered.wait(5.0)
    observers.on_train_step_start()

    t0 = time.monotonic()
    assert observers.shutdown(timeout=0.2) is False
    elapsed = time.monotonic() - t0
    assert 0.1 <= elapsed < 5.0, f"bounded by the timeout, got {elapsed:.3f} s"
    assert observers.worker_alive and any("worker still running" in w for w in warnings)
    assert observers.shutdown(timeout=0.1) is False, "repeatable while the worker is parked"

    parked.release.set()
    assert observers.shutdown(timeout=5.0) is True and not observers.worker_alive
    assert parked.names() == ["train_start", "step_start"], "the queued events were delivered before the stop"
    observers.on_train_step_end()  # after shutdown: inline, on this thread, no new worker
    assert parked.names()[-1] == "step_end" and not observers.worker_alive
    assert observers.flush() and observers.shutdown(), "idempotent once stopped"


def _parking_deliver(delivered: list[str], entered: threading.Event, release: threading.Event):
    """A `deliver` for `BoundedEventDispatcher` that parks the delivering thread inside the event "park"."""

    def deliver(event: Any) -> None:
        if event.name == "park":
            entered.set()
            assert release.wait(10.0), "the test never released the parked deliver"
        delivered.append(event.name)

    return deliver


def test_a_submit_racing_shutdown_is_queued_ahead_of_the_stop_or_counted_as_dropped(monkeypatch):
    """The stopped check and the enqueue in `submit` are one critical section with the flag set in `shutdown`:
    (a) a submit that passed the check is queued before a concurrent shutdown can set the flag or send its
    stop request, so it is delivered; (b) a submit that arrives once the flag is set is counted as dropped
    while the worker still runs. Both orderings are forced, not timed: in (a) the emitting thread is held
    inside its enqueue while shutdown starts; in (b) the queue is pre-filled so shutdown blocks between
    setting the flag and enqueueing the stop request. Nothing acknowledged is ever lost."""
    warnings: list[str] = []
    monkeypatch.setattr(dispatch_module.logger, "warning", warnings.append)
    delivered: list[str] = []
    entered, release = threading.Event(), threading.Event()

    # (a) delivered: the emitter holds the critical section, shutdown cannot begin until its enqueue is done.
    d = BoundedEventDispatcher(_parking_deliver(delivered, entered, release), maxsize=4, name="racing")
    assert d.submit("park") and entered.wait(5.0)
    inside, go = threading.Event(), threading.Event()
    lock_held: list[bool] = []
    real_put_nowait = d._queue.put_nowait

    def held_put_nowait(item):
        lock_held.append(d._lock.locked())
        inside.set()
        assert go.wait(10.0), "the test never released the held emitter"
        real_put_nowait(item)

    monkeypatch.setattr(d._queue, "put_nowait", held_put_nowait)
    acked: list[bool] = []
    emitter = threading.Thread(target=lambda: acked.append(d.submit("racing")))
    emitter.start()
    assert inside.wait(5.0)
    assert lock_held == [True], "the enqueue runs inside the dispatcher's critical section"
    results: list[bool] = []
    stopper = threading.Thread(target=lambda: results.append(d.shutdown(timeout=10.0)))
    stopper.start()
    assert not d.stopped, "shutdown cannot set its flag while a submit is between its check and its enqueue"
    go.set()
    emitter.join(5.0)
    assert acked == [True]
    release.set()
    stopper.join(15.0)
    assert results == [True] and not d.worker_alive
    assert delivered == ["park", "racing"], "the racing event was queued ahead of the stop request and delivered"
    assert d.dropped == 0 and warnings == []

    # (b) counted: the flag is set, the stop request waits for room in the full queue, a submit now is dropped.
    delivered.clear()
    entered.clear()
    release.clear()
    d = BoundedEventDispatcher(_parking_deliver(delivered, entered, release), maxsize=4, name="racing")
    assert d.submit("park") and entered.wait(5.0)
    assert all(d.submit(f"queued-{i}") for i in range(4)), "maxsize events sit behind the parked one"
    stop_requested = threading.Event()
    real_put = d._queue.put

    def signalling_put(item, *args, **kwargs):
        if item is dispatch_module._STOP:
            stop_requested.set()  # shutdown set its flag and now waits for room for its stop request
        return real_put(item, *args, **kwargs)

    monkeypatch.setattr(d._queue, "put", signalling_put)
    results.clear()
    stopper = threading.Thread(target=lambda: results.append(d.shutdown(timeout=10.0)))
    stopper.start()
    assert stop_requested.wait(5.0) and d.stopped and d.worker_alive
    assert d.submit("late") is False and d.dropped == 1, "counted as dropped, not acknowledged"
    assert d.submit("later") is False and d.dropped == 2
    assert len(warnings) == 1 and "dropped late: it arrived after shutdown" in warnings[0], "logged once"
    release.set()
    stopper.join(15.0)
    assert results == [True] and not d.worker_alive
    assert delivered == ["park", *[f"queued-{i}" for i in range(4)]], "every acknowledged event, in order"


def test_after_a_timed_out_shutdown_events_are_dropped_while_the_worker_lives_and_inline_once_it_exits(monkeypatch):
    """After a bounded shutdown returned False with the worker parked inside an observer, a later event is
    dropped and counted (one warning), not delivered inline on the caller's thread ahead of the events still
    queued; once the worker has drained and exited, events deliver inline on the caller's thread."""
    warnings: list[str] = []
    monkeypatch.setattr(dispatch_module.logger, "warning", warnings.append)
    order: list[tuple[str, str]] = []
    entered, release = threading.Event(), threading.Event()

    def deliver(event: Any) -> None:
        if event.name == "park":
            entered.set()
            assert release.wait(10.0), "the test never released the parked deliver"
        order.append((threading.current_thread().name, event.name))

    d = BoundedEventDispatcher(deliver, maxsize=8, name="bus")
    assert d.submit("park") and d.submit("queued-1") and d.submit("queued-2")
    assert entered.wait(5.0)
    assert d.shutdown(timeout=0.2) is False and d.stopped and d.worker_alive

    assert d.submit("late-1") is False and d.submit("late-2") is False
    assert d.dropped == 2 and order == [], "nothing was delivered on this thread while the worker holds the queue"
    late = [w for w in warnings if "arrived after shutdown" in w]
    assert len(late) == 1 and "dropped late-1" in late[0], "the late drop is logged once"

    release.set()
    assert d.shutdown(timeout=5.0) is True and not d.worker_alive
    assert order == [("bus", "park"), ("bus", "queued-1"), ("bus", "queued-2")], "FIFO, all on the worker"
    assert d.submit("after") is True, "the worker is gone: inline on the caller's thread"
    assert order[-1] == (threading.current_thread().name, "after") and d.dropped == 2


def test_sync_dispatch_is_deterministic_and_starts_no_worker():
    rec = _RecordingObserver()
    observers = ControllerLifecycleObservers([rec, _RaisingObserver()], dispatch="sync")
    observers.on_train_start(train_iterations_start=0, train_samples_start=0)
    observers.on_train_step_start()
    assert rec.names() == ["train_start", "step_start"], "delivered inline, before the emit returns"
    assert observers.dispatch == "sync" and not observers.worker_alive and observers.dropped == 0
    assert observers.flush() and observers.shutdown()
    observers.on_train_end()
    observers.on_app_end()
    assert rec.names()[-2:] == ["train_end", "app_end"]
    # coerce honours the mode for the list it builds and leaves an existing list's mode alone.
    assert ControllerLifecycleObservers.coerce([rec], dispatch="sync").dispatch == "sync"
    assert ControllerLifecycleObservers.coerce(rec).dispatch == "async"
    assert ControllerLifecycleObservers.coerce(observers).dispatch == "sync"
    assert ControllerLifecycleObservers.coerce(None, dispatch="sync").dispatch == "sync"
    with pytest.raises(ValueError, match="dispatch must be"):
        ControllerLifecycleObservers([rec], dispatch="later")  # type: ignore[arg-type]


def test_app_start_is_the_synchronous_gate():
    """`on_app_start` runs on the caller's thread (DESIGN §2.6 gate; §11.2 precedent): the backend's
    singleton exists when the call returns, and nothing was queued for it."""
    rec = _RecordingObserver()
    observers = ControllerLifecycleObservers([rec])
    observers.on_app_start()
    assert rec.events == [("app_start",)] and not observers.worker_alive
    observers.on_train_start(train_iterations_start=0, train_samples_start=0)
    assert observers.worker_alive, "the first asynchronous event started the worker"
    assert observers.flush() and rec.names() == ["app_start", "train_start"]
    assert observers.shutdown()


@pytest.mark.parametrize("dispatch", ["async", "sync"])
def test_terminal_cleanup_before_training_bounds_a_blocked_finish(
    monkeypatch: pytest.MonkeyPatch, dispatch: str
) -> None:
    """The sync case is the negative control for the former no-worker inline path."""
    entered, release, returned = threading.Event(), threading.Event(), threading.Event()

    class ParkedFinish(_RecordingOneLogger):
        def finish(self) -> None:
            entered.set()
            assert release.wait(5.0)
            self.calls.append(("finish", {}))

    recorder = ParkedFinish()
    observer, _ = _one_logger(recorder)
    bus = ControllerLifecycleObservers([observer], dispatch=dispatch)
    monkeypatch.setattr(bus, "DRAIN_TIMEOUT_S", 0.05)
    bus.on_app_start()
    assert not bus.worker_alive

    def close() -> None:
        bus.on_app_end()
        returned.set()

    caller = threading.Thread(target=close)
    caller.start()
    try:
        assert entered.wait(2.0)
        assert returned.wait(0.5) is (dispatch == "async")
        assert recorder.names.count("finish") == 0, "bounded return does not mean SDK cleanup completed"
    finally:
        release.set()
        caller.join(3.0)
        assert bus.shutdown(3.0)
    assert returned.is_set() and recorder.names.count("finish") == 1


def test_validation_overflow_preserves_start_end_pairs() -> None:
    parked = _ParkingObserver("on_train_start")
    bus = ControllerLifecycleObservers([parked])
    bus.on_train_start(train_iterations_start=0, train_samples_start=0)
    try:
        assert parked.entered.wait(2.0)
        while bus.dropped == 0:
            bus.on_checkpoint_save_success(0)
        bus.on_validation_start()
        parked.release.set()
        assert bus.flush(5.0)
        bus.on_validation_end()
        assert bus.flush(5.0)
        assert "validation_start" not in parked.names() and "validation_end" not in parked.names()
        assert bus.dropped_pairs == 1
        bus.on_validation_start()
        bus.on_validation_end()
        assert bus.flush(5.0)
        assert parked.names()[-2:] == ["validation_start", "validation_end"]
    finally:
        parked.release.set()
        assert bus.shutdown(5.0)


def test_checkpoint_location_payload_keeps_legacy_observer_signatures() -> None:
    recorded = []
    legacy = SimpleNamespace(on_checkpoint_save_success=recorded.append)
    bus = ControllerLifecycleObservers([legacy], dispatch="sync")
    bus.on_checkpoint_save_success(7, location="s3://actual-store/iter_000007")
    assert recorded == [7]


def test_app_end_is_delivered_after_a_full_queue_when_the_parked_observer_recovers(monkeypatch):
    """The reviewer's probe: the first callback is parked, the queue is filled, `on_app_end()` is called and the
    callback is released 50 ms later. app_end must reach the observer: the gate takes no room in the queue, so the
    full queue cannot drop it; the worker delivers it behind every queued event and is then stopped."""
    warnings: list[str] = []
    monkeypatch.setattr(dispatch_module.logger, "warning", warnings.append)
    monkeypatch.setattr(lifecycle_module.logger, "warning", warnings.append)
    parked = _ParkingObserver("on_train_start")
    observers = ControllerLifecycleObservers([parked])
    observers.on_train_start(train_iterations_start=0, train_samples_start=0)
    assert parked.entered.wait(5.0)
    while observers.dropped == 0:  # fill the DEFAULT_QUEUE_MAXSIZE queue behind the parked event
        observers.on_train_batch_start()
        observers.on_train_batch_end()

    releaser = threading.Timer(0.05, parked.release.set)
    releaser.start()
    t0 = time.monotonic()
    observers.on_app_end()
    elapsed = time.monotonic() - t0
    releaser.join(5.0)
    names = parked.names()
    assert names[-1] == "app_end", f"app_end was delivered after the drained queue, got {names[-2:]}"
    assert elapsed < 10.0, f"the gate waited for the drain, not for a timeout ({elapsed:.2f} s)"
    assert parked.threads[-1] == "lifecycle-observers", "delivered by the worker, behind the queue"
    assert not observers.worker_alive
    # The fill stopped at the 513th batch start, dropped together with its end: no orphan, in order.
    assert observers.dropped == 2 and observers.dropped_pairs == 1
    assert names.count("batch_start") == names.count("batch_end") == DEFAULT_QUEUE_MAXSIZE // 2
    assert names[1:-1] == ["batch_start", "batch_end"] * (DEFAULT_QUEUE_MAXSIZE // 2), "no orphaned end, in order"
    assert any("2 events were dropped over the run (1 start/end groups dropped together" in w for w in warnings)
    assert not any("could not be delivered within" in w for w in warnings), "the gate was delivered within its bound"


def test_a_gate_never_enters_an_observer_the_worker_is_inside_and_arrives_in_order_after_its_bound(monkeypatch):
    """The probe reproduced at the previous head: `on_train_start` parked (the hook holds the observer's own lock),
    `step_start` / `step_end` queued, the bound forced to 0.2 s, then `on_train_end()` and `on_app_end()`. Before,
    both gates were forced onto the controller thread once the drain timed out: a second thread entered the observer
    while the worker sat inside `on_train_start` (and the lock kept the controller blocked for as long as the observer
    was parked), and recovery delivered [train_end, app_end, train_start, step_start, step_end], OneLogger torn down
    before its train start arrived. Now the gates are non-droppable markers behind the queue: the controller waits its
    bound (twice, plus the bounded join) and returns without entering the observer, nothing terminal is delivered
    while the hook runs, and once the observer returns the worker delivers every event in emission order with the
    two gates last, all on the worker thread."""
    warnings: list[str] = []
    monkeypatch.setattr(lifecycle_module.logger, "warning", warnings.append)
    monkeypatch.setattr(dispatch_module.logger, "warning", warnings.append)
    monkeypatch.setattr(ControllerLifecycleObservers, "DRAIN_TIMEOUT_S", 0.2)
    rec = _RecordingOneLogger()
    ol, fake = _one_logger(rec)
    parked = _SerialObserver(park_on="on_train_start")
    observers = ControllerLifecycleObservers([parked, ol])
    observers.on_app_start()
    assert ol.initialized and rec.names == ["on_app_start"] and parked.names() == ["app_start"]
    observers.on_train_start(train_iterations_start=0, train_samples_start=0)
    assert parked.entered.wait(5.0)
    observers.on_train_step_start()
    observers.on_train_step_end()

    controller = threading.Thread(
        target=lambda: (observers.on_train_end(), observers.on_app_end()), name="controller", daemon=True
    )
    t0 = time.monotonic()
    controller.start()
    controller.join(3.0)
    elapsed = time.monotonic() - t0
    try:
        assert not controller.is_alive(), "the controller thread is blocked inside the parked observer's lock"
        assert 0.5 <= elapsed < 3.0, f"two bounded gate waits and a bounded join, got {elapsed:.2f} s"
        assert parked.max_inside == 1 and parked.overlaps == [], (
            f"a second thread entered the observer: {parked.overlaps}"
        )
        assert parked.names() == ["app_start"] and rec.names == ["on_app_start"], "nothing terminal ran under the hook"
        assert ol.initialized, "OneLogger is not torn down while an earlier hook is still running"
        assert observers.worker_alive
        bounded = [w for w in warnings if "could not be delivered within 0.2 s" in w]
        assert len(bounded) == 2 and "on_train_end could not" in bounded[0] and "on_app_end could not" in bounded[1]
        assert all("parked inside on_train_start with 2 events queued ahead of it" in w for w in bounded), warnings
    finally:
        parked.release.set()

    assert observers.shutdown(timeout=5.0) and not observers.worker_alive, "the worker delivered the rest and stopped"
    assert parked.names() == ["app_start", "train_start", "step_start", "step_end", "train_end", "app_end"], (
        "emission order, the gates last"
    )
    assert parked.threads[1:] == ["lifecycle-observers"] * 5, "every queued event and both gates on the worker"
    assert rec.names == [
        "on_app_start",
        "on_train_start",
        "on_train_step_start",
        "on_train_step_end",
        "on_train_end",
        "on_app_end",
        "finish",
    ], "OneLogger saw its train start before its train end and its teardown"
    assert not ol.initialized and fake.state["ol"] is None
    assert observers.dropped == 0 and parked.max_inside == 1


def test_terminal_events_stay_serial_and_last_behind_a_slow_observer_when_every_bound_runs_out(monkeypatch):
    """A healthy but slow observer (30 ms per hook) with 21 events queued and the bound forced to 0.1 s: `on_train_end`
    and `on_app_end` both run out of their bound and return without entering the observer (about 0.3 s for the two
    waits and the join, not the ~0.7 s the queue needs); a late event is dropped and counted (the stop request sits
    behind the gates); the worker keeps delivering in emission order, reaches the two gates last and exits at the
    stop. Before, both gates were forced onto the controller thread mid-stream (two threads inside the observer) and
    the queued events kept arriving after the app end."""
    monkeypatch.setattr(ControllerLifecycleObservers, "DRAIN_TIMEOUT_S", 0.1)
    slow = _SerialObserver(sleep_s=0.03)
    observers = ControllerLifecycleObservers([slow])
    observers.on_train_start(train_iterations_start=0, train_samples_start=0)
    for _ in range(10):
        observers.on_train_batch_start()
        observers.on_train_batch_end()
    t0 = time.monotonic()
    observers.on_train_end()
    observers.on_app_end()
    elapsed = time.monotonic() - t0
    assert 0.25 <= elapsed < 0.6, f"three bounds of 0.1 s, not the queue's drain time, got {elapsed:.2f} s"
    assert observers.worker_alive and "train_end" not in slow.names(), "the worker is still on the queued events"
    observers.on_train_step_start()  # after a bounded shutdown while the worker lives: dropped and counted, not inline
    assert observers.dropped == 1
    deadline = time.monotonic() + 5.0
    while observers.worker_alive and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not observers.worker_alive, "the worker delivered the rest, the two gates, and stopped"
    assert slow.names() == ["train_start", *["batch_start", "batch_end"] * 10, "train_end", "app_end"]
    assert slow.max_inside == 1 and slow.overlaps == [], f"a second thread entered the observer: {slow.overlaps}"
    assert set(slow.threads) == {"lifecycle-observers"}


def test_dispatcher_gates_are_delivered_in_fifo_order_by_the_worker_within_a_bounded_wait(monkeypatch):
    """`deliver_gate` on a 2-slot dispatcher starts a worker even before ordinary events;
    with the worker parked it is queued behind every event without taking capacity, the caller waits its bound
    and gets False while the gate stays queued (the warning names the parked hook and the events ahead); after a
    bounded shutdown with the worker still parked a further gate is dropped and counted (it cannot go behind the stop
    request); once released, the worker delivers the queued events and the gate in FIFO order and stops; with the
    worker gone a gate runs inline again."""
    warnings: list[str] = []
    monkeypatch.setattr(dispatch_module.logger, "warning", warnings.append)
    order: list[tuple[str, str]] = []
    entered, release = threading.Event(), threading.Event()

    def deliver(event: Any) -> None:
        if event.name == "park":
            entered.set()
            assert release.wait(10.0), "the test never released the parked deliver"
        order.append((threading.current_thread().name, event.name))

    me = threading.current_thread().name
    d = BoundedEventDispatcher(deliver, maxsize=2, name="bus")
    assert d.deliver_gate(LifecycleEvent.now("gate-0"), 1.0) is True and order == [("bus", "gate-0")]
    assert d.worker_alive, "early terminal cleanup needs a worker to bound its wait"
    assert d.submit("park") and d.submit("queued") and d.submit("overflow") is False and d.dropped == 1
    assert entered.wait(5.0) and d.in_flight == "park"
    t0 = time.monotonic()
    assert d.deliver_gate(LifecycleEvent.now("gate-1"), 0.2) is False
    assert 0.15 <= time.monotonic() - t0 < 2.0, "bounded by the wait"
    assert d.queued == 1 and d.dropped == 1, "the gate took no capacity and was not dropped"
    assert order == [("bus", "gate-0")], "nothing was delivered on this thread while the worker holds a hook"
    assert [
        w
        for w in warnings
        if "gate-1 could not be delivered within 0.2 s" in w and "inside park with 1 events queued" in w
    ], warnings
    assert d.shutdown(timeout=0.2) is False and d.stopped and d.worker_alive
    assert d.deliver_gate(LifecycleEvent.now("gate-2"), 0.2) is False and d.dropped == 2
    assert any(
        "dropped gate-2: it arrived after shutdown while the worker is still running (inside park)" in w
        for w in warnings
    )
    release.set()
    assert d.shutdown(timeout=5.0) is True and not d.worker_alive
    assert order == [("bus", "gate-0"), ("bus", "park"), ("bus", "queued"), ("bus", "gate-1")], (
        "FIFO: the gate after the events queued ahead of it, delivered by the worker"
    )
    assert d.deliver_gate(LifecycleEvent.now("gate-3"), 0.2) is True and order[-1] == (me, "gate-3")
    assert d.in_flight is None


def test_a_gate_raised_on_the_worker_from_inside_a_hook_is_delivered_once_inline_and_never_queued(monkeypatch):
    """The reentrant case: ``deliver`` (running on the worker) raises a gate from inside a hook. Before, the
    ``alive and not stopped`` branch queued a `_Gate` first and the worker-thread branch then delivered the same
    item inline; when the hook returned, `_run()` dequeued the marker and delivered the terminal event a second
    time. Now the reentrant check comes before anything is queued: the gate is delivered inline, exactly once, on
    the worker; nothing is left on the queue; the events queued behind the hook follow in emission order."""
    warnings: list[str] = []
    monkeypatch.setattr(dispatch_module.logger, "warning", warnings.append)
    order: list[tuple[str, str]] = []
    results: list[bool] = []

    def deliver(event: Any) -> None:
        if event.name == "trigger":  # this thread IS the worker: the gate must run here, once, and never be queued
            results.append(d.deliver_gate(LifecycleEvent.now("gate"), 1.0))
        order.append((threading.current_thread().name, event.name))

    d = BoundedEventDispatcher(deliver, maxsize=4, name="bus")
    assert d.submit("trigger") and d.submit("after")
    assert d.flush(5.0) and d.shutdown(5.0) and not d.worker_alive
    assert results == [True], "the reentrant gate reports delivery"
    assert order == [("bus", "gate"), ("bus", "trigger"), ("bus", "after")], (
        "the gate once, inline on the worker, then the queued events in order (a queued copy would appear again)"
    )
    assert d.queued == 0 and d.dropped == 0 and d.in_flight is None
    assert not warnings, warnings


def test_an_observer_ending_the_app_from_inside_its_own_hook_tears_down_exactly_once(monkeypatch):
    """The reentrant case on the bus: an observer calls `on_app_end()` from inside one of its hooks, i.e. on the
    dispatcher's worker. The terminal event must reach every observer exactly once: before, the gate was queued AND
    delivered inline, so app_end arrived twice and a backend's finish/destroy cleanup ran twice. OneLogger's own
    teardown is idempotent; the counting observer beside it is not, and it is what shows the double delivery."""
    warnings: list[str] = []
    monkeypatch.setattr(dispatch_module.logger, "warning", warnings.append)
    monkeypatch.setattr(lifecycle_module.logger, "warning", warnings.append)
    rec = _RecordingOneLogger()
    ol, fake = _one_logger(rec)

    class _ClosingObserver(_RecordingObserver):
        """Ends the app from inside its `on_train_step_end` hook (on the worker) and counts its own teardown."""

        def __init__(self) -> None:
            super().__init__()
            self.bus: ControllerLifecycleObservers | None = None
            self.finish_calls = 0
            self.destroy_calls = 0
            self.hook_done = threading.Event()  # set once the hook that raised the gate has returned

        def on_train_step_end(self) -> None:
            super().on_train_step_end()
            assert self.bus is not None
            try:
                self.bus.on_app_end()  # a gate raised reentrantly on the worker thread
            finally:
                self.hook_done.set()

        def on_app_end(self) -> None:
            super().on_app_end()
            self.finish_calls += 1
            self.destroy_calls += 1

    closing = _ClosingObserver()
    # OneLogger first: it sees each event before the closer, delivered after it, raises the gate from its own hook.
    observers = ControllerLifecycleObservers([ol, closing])
    closing.bus = observers
    observers.on_app_start()
    observers.on_train_start(train_iterations_start=0, train_samples_start=0)
    observers.on_train_step_start()
    observers.on_train_step_end()
    assert closing.hook_done.wait(5.0), "the worker never reached the hook that ends the app"
    # The reentrant `on_app_end` could not join the worker it runs on; the controller's shutdown stops it. The stop
    # request lands behind anything the hook left on the queue, so a queued copy of the gate (the old behaviour)
    # would be delivered before the worker exits and show up below as a second app_end.
    assert observers.shutdown(timeout=5.0) and not observers.worker_alive

    assert closing.names() == ["app_start", "train_start", "step_start", "step_end", "app_end"], (
        "app_end exactly once, after the hook that raised it"
    )
    assert (closing.finish_calls, closing.destroy_calls) == (1, 1), "finish/destroy cleanup ran once"
    assert closing.threads[1:] == ["lifecycle-observers"] * 4, "the reentrant gate was delivered on the worker"
    assert rec.names == [
        "on_app_start",
        "on_train_start",
        "on_train_step_start",
        "on_train_step_end",
        "on_app_end",
        "finish",
    ], "OneLogger saw one app end and one finish"
    assert not ol.initialized and fake.state["ol"] is None
    assert observers.dropped == 0
    assert [w for w in warnings if "shutdown called from the worker thread" in w], warnings
    assert not any("could not be delivered" in w or "dropped" in w for w in warnings), warnings


def test_dispatcher_admits_a_start_only_with_room_for_its_followers_and_drops_groups_together(monkeypatch):
    """Pair-preserving overflow on a 3-slot queue with the worker parked inside the first event: a start is
    admitted only with room for itself and its followers, an admitted start's follower always finds its
    reserved slot, a start that does not fit is dropped together with its follower (one group, counted once,
    logged once), and a follower whose start was never seen is an ordinary event."""
    warnings: list[str] = []
    monkeypatch.setattr(dispatch_module.logger, "warning", warnings.append)
    delivered: list[str] = []
    entered, release = threading.Event(), threading.Event()
    d = BoundedEventDispatcher(
        _parking_deliver(delivered, entered, release), maxsize=3, name="pairs", pairs={"s": ("e",), "t": ("u",)}
    )
    assert d.submit("park") and entered.wait(5.0), "taken by the worker: the queue is empty again"
    assert d.submit("s"), "room for it and its e"
    assert d.queued == 1
    assert d.submit("p"), "1 queued + 1 reserved + itself = 3"
    assert d.submit("t") is False, "no room for t and its u"
    assert d.dropped == 1 and d.dropped_pairs == 1
    assert len(warnings) == 1 and "dropped t together with its u (no room for the group)" in warnings[0]
    assert d.submit("u") is False and d.dropped == 2 and d.dropped_pairs == 1, "its follower drops with it"
    assert len(warnings) == 1, "the group is logged once, at its start"
    assert d.submit("e"), "the follower of an admitted start always has its reserved slot"
    assert d.queued == 3 == d.maxsize, "the bound holds exactly"
    assert d.submit("q") is False and d.dropped == 3, "a plain event finds no room"
    assert d.submit("u") is False and d.dropped == 4, "a follower without a start is an ordinary event (queue full)"

    release.set()
    assert d.flush(timeout=5.0)
    assert delivered == ["park", "s", "p", "e"], "every admitted event in order, no orphan"
    assert d.submit("t") and d.submit("u"), "with room again a group is admitted whole"
    assert d.shutdown(timeout=5.0)
    assert delivered == ["park", "s", "p", "e", "t", "u"]
    assert "4 events were dropped over the run (1 start/end groups dropped together" in warnings[-1]


def test_a_dropped_start_drops_its_end_on_the_bus_even_once_there_is_room_again(monkeypatch):
    """The overflow that split pairs before: the queue is full when a step starts (the start is dropped) and has
    drained by the time the step ends. The end is dropped with its start instead of arriving as an orphan the
    observer would pair with nothing; the next step's pair is delivered whole; train_end is the gate."""
    warnings: list[str] = []
    monkeypatch.setattr(dispatch_module.logger, "warning", warnings.append)
    parked = _ParkingObserver("on_train_start")
    observers = ControllerLifecycleObservers([parked])
    observers.on_train_start(train_iterations_start=0, train_samples_start=0)
    assert parked.entered.wait(5.0)
    while observers.dropped == 0:  # plain events (a success whose save was never seen) fill the queue exactly
        observers.on_checkpoint_save_success(0)
    assert observers.dropped == 1

    observers.on_train_step_start()  # no room for the group: dropped together with its end
    assert observers.dropped == 2
    parked.release.set()
    assert observers.flush(timeout=10.0), "the queue drained meanwhile"
    observers.on_train_step_end()  # room now, but its start is gone: dropped with it, not delivered as an orphan
    assert observers.flush(timeout=5.0)
    names = parked.names()
    assert "step_end" not in names and "step_start" not in names, f"no orphaned end, got {names[-2:]}"
    assert observers.dropped == 3 and observers.dropped_pairs == 1, "the end is counted with its start's group"

    observers.on_train_step_start()
    observers.on_train_step_end()
    observers.on_train_end()
    assert parked.names()[-3:] == ["step_start", "step_end", "train_end"], "the next pair is delivered whole"
    assert parked.threads[-1] == "lifecycle-observers", "train_end is the gate, delivered by the worker behind the pair"
    assert observers.shutdown(timeout=5.0) and observers.dropped == 3


def test_queued_events_carry_the_controller_emission_clocks_to_observe():
    """Every queued event is stamped with the emitting thread's wall and monotonic clocks, and `observe(event)`
    hands them to the observer before its hook runs: two events emitted 100 ms apart on the controller thread
    and delivered back to back (the worker was parked ahead of them) still show the 100 ms between their
    `emitted_monotonic` stamps, and `emitted_wall` is the controller's wall clock at emission."""

    class _Stamped(_RecordingObserver):
        def __init__(self) -> None:
            super().__init__()
            self.stamps: dict[str, tuple[float, float, float]] = {}

        def observe(self, event: Any) -> None:
            self.stamps[event.name] = (event.emitted_wall, event.emitted_monotonic, time.monotonic())
            super().observe(event)

    parked = _ParkingObserver("on_train_start")
    stamped = _Stamped()
    observers = ControllerLifecycleObservers([parked, stamped])
    observers.on_train_start(train_iterations_start=0, train_samples_start=0)
    assert parked.entered.wait(5.0)
    wall0, mono0 = time.time(), time.monotonic()
    observers.on_train_step_start()
    time.sleep(0.1)
    observers.on_train_step_end()
    mono1 = time.monotonic()
    parked.release.set()
    assert observers.flush(timeout=5.0)

    start, end = stamped.stamps["on_train_step_start"], stamped.stamps["on_train_step_end"]
    assert end[1] - start[1] >= 0.09, "the emission stamps keep the controller's 100 ms"
    assert end[2] - start[2] < 0.05, "while the deliveries happened back to back"
    assert mono0 <= start[1] <= end[1] <= mono1 and abs(start[0] - wall0) < 0.05
    assert stamped.stamps["on_train_start"][1] < start[1]
    assert observers.shutdown(timeout=5.0)


class _FakeOneLoggerCore:
    """Stand-in for `one_logger.core.OneLogger` (1.5.5: `log_metrics`, `log_app_tag`, `store_*`, `finish`), so the
    real imaginaire `OneLoggerUtils` wrapper can be driven without the wheel and its metrics inspected."""

    def __init__(self, config: dict) -> None:
        self.config = dict(config)
        self.store: dict[str, Any] = {}
        self.logged: list[dict[str, Any]] = []
        self.app_tags: list[str] = []
        self.finished = False

    def store_set(self, key: str, value: Any, use_deepcopy: bool = True, override: bool = True) -> None:
        self.store[key] = value

    def store_get(self, key: str, use_deepcopy: bool = True) -> Any:
        return self.store[key]

    def store_pop(self, key: str) -> Any:
        return self.store.pop(key)

    def store_has_key(self, key: str) -> bool:
        return key in self.store

    def log_metrics(self, metrics: dict, step: Any = None, summary: Any = None, commit: bool = True) -> None:
        self.logged.append(dict(metrics))

    def log_app_tag(self, app_tag: str) -> None:
        self.app_tags.append(app_tag)

    def finish(self) -> None:
        self.finished = True

    def last(self, key: str) -> Any:
        """The most recently logged value of ``key``."""
        return [m[key] for m in self.logged if key in m][-1]


def _real_one_logger_utils(monkeypatch: pytest.MonkeyPatch) -> Any:
    """imaginaire's real `one_logger_utils` module (the `OneLoggerUtils` wrapper and its global-object
    functions) over a fake `one_logger.core`, the way the wrapper sees the wheel."""
    package, core = ModuleType("one_logger"), ModuleType("one_logger.core")
    core.OneLogger = _FakeOneLoggerCore  # type: ignore[attr-defined]
    package.core = core  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "one_logger", package)
    monkeypatch.setitem(sys.modules, "one_logger.core", core)
    from cosmos_framework.utils.one_logger import one_logger_utils

    assert not one_logger_utils.one_logger_is_initialized(), "no earlier test left the global wrapper behind"
    return one_logger_utils


def test_one_logger_timers_read_the_emission_clock_behind_a_slow_observer(monkeypatch):
    """The reviewer's probe on the real wrapper: a slow observer ahead of OneLogger in the list delays every
    OneLogger delivery, so with the wrapper's own `time.time()` clock a 200 ms batch is reported as the
    microseconds between two back-to-back deliveries. With the emission clock the wrapper's `train_batch`,
    `train_iterations` and `save_checkpoint` metrics measure the controller's intervals, its `*_finish_time`
    stamps are the controller's wall clock, `app_finish_time` goes through the wrapper's explicit-time kwarg,
    and the timer installed on the wrapper is the emission-clock one."""
    utils = _real_one_logger_utils(monkeypatch)

    class _Slow(_RecordingObserver):
        def on_train_step_start(self) -> None:
            time.sleep(0.25)  # everything the controller emits meanwhile queues up behind this hook
            super().on_train_step_start()

    ol = OneLoggerLifecycleObserver(
        utils, _config(batch_per_replica=2, n_mini=1, n_micro=1, max_steps=1, world_size=1, policy_dp_world=1)
    )
    observers = ControllerLifecycleObservers([_Slow(), ol])
    observers.on_app_start()
    wrapper = utils.get_one_logger()
    core = wrapper.one_logger
    assert isinstance(core, _FakeOneLoggerCore) and ol.active and core.last("app_start_time") == 1_700_000_000_000

    observers.on_train_start(train_iterations_start=0, train_samples_start=0)
    observers.on_train_step_start()
    observers.on_train_batch_start()
    time.sleep(0.2)  # the workload: one forward/backward
    observers.on_train_batch_end()
    observers.on_train_step_end()
    step_end_wall_ms = time.time() * 1000
    observers.on_checkpoint_save_start(1)
    time.sleep(0.1)  # the checkpoint write
    observers.on_checkpoint_save_end(1)
    observers.on_checkpoint_save_success(1)
    observers.on_train_end()

    batch_ms = core.last("train_batch_time_msecs_avg")
    step_ms = core.last("train_iterations_time_msecs_avg")
    save_s = core.last("save_checkpoint_sync_time_total")
    assert 150 <= batch_ms < 1000, f"train_batch measured the controller's 200 ms, got {batch_ms:.3f} ms"
    assert 150 <= step_ms < 1000, f"train_iterations measured the controller's 200 ms, got {step_ms:.3f} ms"
    assert 0.08 <= save_s < 1.0, f"save_checkpoint measured the controller's 100 ms, got {save_s:.4f} s"
    assert core.last("train_batch_time_total") == pytest.approx(batch_ms / 1000, rel=1e-6)
    assert abs(core.last("last_logged_train_iterations_finish_time") - step_end_wall_ms) < 50
    assert (
        core.last("train_iterations") == 1 and core.last("train_batch") == 1 and core.last("save_checkpoint_count") == 1
    )

    from cosmos_framework.posttrain.callbacks.one_logger import EmissionClockTimer

    assert isinstance(wrapper.timer, EmissionClockTimer) and ol.timer is wrapper.timer
    assert wrapper.timer.active_timers() == ["app_run"], "only the app interval is still open"

    app_end_wall_ms = time.time() * 1000
    observers.on_app_end()
    assert abs(core.last("app_finish_time") - app_end_wall_ms) < 50, (
        "the explicit-time kwarg carries the controller's clock"
    )
    assert core.finished and not utils.one_logger_is_initialized() and not ol.initialized and ol.timer is None


def test_validation_timer_uses_emission_clock_with_delayed_delivery_and_negative_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    durations = []
    for emission_clock in (True, False):
        utils = _real_one_logger_utils(monkeypatch)
        config = _config()
        config["is_validation_iterations_enabled"] = True
        observer = OneLoggerLifecycleObserver(utils, config)
        if not emission_clock:
            # Controlled regression: bypass only the adapter's emission-clock binding.
            observer.observe = lambda event: ControllerLifecycleObserver.observe(observer, event)
        parked = _ParkingObserver("on_validation_start")
        bus = ControllerLifecycleObservers([parked, observer])
        bus.on_app_start()
        core = utils.get_one_logger().one_logger
        try:
            bus.on_train_start(train_iterations_start=0, train_samples_start=0)
            bus.on_validation_start()
            assert parked.entered.wait(2.0)
            time.sleep(0.15)
            bus.on_validation_end()
            parked.release.set()
            bus.on_train_end()
            assert core.store["validation_iterations"] == 1
            durations.append(core.last("validation_iterations_time_total"))
        finally:
            parked.release.set()
            bus.on_app_end()
    assert durations[0] >= 0.12
    assert durations[1] < 0.05, "the negative control must expose collapsed delivery-time measurements"


def test_emission_clock_timer_matches_the_wrapper_timer_interface():
    """`EmissionClockTimer` behaves like the wrapper's `_NamedTimer` (start/stop/is_active/active_timers/reset/get,
    the same data keys, a double start or a stop without a start raise), with durations from the monotonic
    stamps and timestamps from the wall stamps it is given."""
    from cosmos_framework.posttrain.callbacks.one_logger import EmissionClockTimer

    clock = [(1_000.0, 50.0)]
    timer = EmissionClockTimer(lambda: clock[0])
    assert timer.get("x") == {} and timer["x"] == {} and not timer.is_active("x") and timer.active_timers() == []
    timer.start("x")
    assert timer.is_active("x") and timer.active_timers() == ["x"] and timer["x"]["start"] == 1_000.0
    with pytest.raises(RuntimeError, match="already active"):
        timer.start("x")
    clock[0] = (1_000.5, 50.2)  # the wall clock jumped 500 ms; the monotonic clock advanced 200 ms
    timer.stop("x")
    data = timer["x"]
    assert data["finish"] == 1_000.5 and data["latest"] == pytest.approx(0.2) and data["count"] == 1
    assert data["min"] == data["max"] == data["total"] == data["avg"] == pytest.approx(0.2)
    assert "is_active" not in data and "_start_monotonic" not in data
    with pytest.raises(RuntimeError, match="not active"):
        timer.stop("x")
    clock[0] = (1_001.0, 50.3)
    timer.start("x")
    clock[0] = (1_001.0, 50.7)
    timer.stop("x")
    data = timer["x"]
    assert data["count"] == 2 and data["latest"] == pytest.approx(0.4) and data["total"] == pytest.approx(0.6)
    assert data["min"] == pytest.approx(0.2) and data["max"] == pytest.approx(0.4) and data["avg"] == pytest.approx(0.3)
    timer.reset("x")
    assert timer["x"] == {}
    timer.start("y")
    timer.reset()
    assert timer.timers == {} and timer.active_timers() == []


def test_terminal_timestamps_are_the_emission_clock_despite_a_delayed_drain(monkeypatch):
    """The follow-up on the real wrapper: the gate event used to be stamped after the drain, so a 300 ms drain in
    front of `on_train_end` shifted `app_train_loop_finish_time` by 300 ms although the loop had already ended when
    the controller called it; `app_finish_time` took the same path. Both gates are now stamped when the controller
    calls them and delivered with that clock: the wrapper's finish times equal the controller's clock at the call
    within tolerance while each drain still takes its 300 ms."""
    utils = _real_one_logger_utils(monkeypatch)

    class _SlowAhead(_RecordingObserver):
        def on_train_step_end(self) -> None:
            time.sleep(0.3)  # the drain in front of on_train_end
            super().on_train_step_end()

        def on_checkpoint_save_success(self, model_step: int) -> None:
            time.sleep(0.3)  # the drain in front of on_app_end
            super().on_checkpoint_save_success(model_step)

    ol = OneLoggerLifecycleObserver(
        utils, _config(batch_per_replica=2, n_mini=1, n_micro=1, max_steps=1, world_size=1, policy_dp_world=1)
    )
    observers = ControllerLifecycleObservers([_SlowAhead(), ol])
    observers.on_app_start()
    core = utils.get_one_logger().one_logger
    observers.on_train_start(train_iterations_start=0, train_samples_start=0)
    observers.on_train_step_start()
    observers.on_train_batch_start()
    observers.on_train_batch_end()
    observers.on_train_step_end()
    train_end_wall_ms = time.time() * 1000
    t0 = time.monotonic()
    observers.on_train_end()
    drain = time.monotonic() - t0
    assert drain >= 0.25, f"the gate really waited behind the slow observer ({drain:.3f} s)"
    shift = core.last("app_train_loop_finish_time") - train_end_wall_ms
    assert abs(shift) < 50, f"app_train_loop_finish_time is the emission clock; it is shifted by {shift:.0f} ms"

    observers.on_checkpoint_save_start(1)
    observers.on_checkpoint_save_end(1)
    observers.on_checkpoint_save_success(1)
    app_end_wall_ms = time.time() * 1000
    t0 = time.monotonic()
    observers.on_app_end()
    drain = time.monotonic() - t0
    assert drain >= 0.25, f"the gate really waited behind the slow observer ({drain:.3f} s)"
    shift = core.last("app_finish_time") - app_end_wall_ms
    assert abs(shift) < 50, f"app_finish_time is the emission clock; it is shifted by {shift:.0f} ms"
    assert core.finished and not utils.one_logger_is_initialized() and not ol.initialized


# --------------------------------------------------------------------------- OneLogger mapping
def test_one_logger_hook_order_with_mini_and_micro_windows():
    rec = _RecordingOneLogger()
    ol, fake = _one_logger(rec, batch_per_replica=8, n_mini=2, n_micro=2, max_steps=10, policy_dp_world=1)
    assert not ol.active and not ol.initialized
    ol.on_app_start()
    assert ol.active and ol.initialized
    ol.on_train_start(train_iterations_start=20, train_samples_start=80)
    for _mini in range(2):
        ol.on_train_step_start()
        for _micro in range(2):
            ol.on_train_batch_start()
            ol.on_train_batch_end()
        ol.on_train_step_end()
    ol.on_checkpoint_save_start(22)
    ol.on_checkpoint_save_end(22)
    ol.on_checkpoint_save_success(22)
    ol.on_train_end()
    ol.on_app_end()
    assert not ol.active and not ol.initialized and fake.state["ol"] is None
    assert rec.names == [
        "on_app_start",
        "on_train_start",
        *_ONE_LOGGER_STEP,
        *_ONE_LOGGER_STEP,
        "on_save_checkpoint_start",
        "on_save_checkpoint_end",
        "on_save_checkpoint_success",
        "on_train_end",
        "on_app_end",
        "finish",
    ]
    by_name = dict(rec.calls)
    assert by_name["on_app_start"] == {"set_barrier": False, "app_start_time": 1_700_000_000_000}
    assert by_name["on_train_start"] == {"set_barrier": False, "train_iterations_start": 20, "train_samples_start": 80}
    assert by_name["on_save_checkpoint_success"] == {"global_step": 22}
    assert fake.state["config"]["train_iterations_target"] == 20 and fake.state["config"]["global_batch_size"] == 2
    # app_end is idempotent.
    ol.on_app_end()
    assert rec.names.count("finish") == 1


def test_preemption_between_step_boundaries_closes_the_open_intervals_at_train_end():
    """A break out of the loop (validation-triggered preemption, a raise inside a step) reaches
    on_train_end with a batch and a step open; OneLogger must not be left with running timers."""
    rec = _RecordingOneLogger()
    ol, _ = _one_logger(rec)
    ol.on_app_start()
    ol.on_train_start(train_iterations_start=0, train_samples_start=0)
    ol.on_train_step_start()
    ol.on_train_batch_start()
    ol.on_train_end()
    assert rec.names[-3:] == ["on_train_batch_end", "on_train_step_end", "on_train_end"]
    # After a cleanly closed step nothing extra is emitted, and a stray end without a start is ignored.
    rec.calls.clear()
    ol.on_train_step_start()
    ol.on_train_batch_start()
    ol.on_train_batch_end()
    ol.on_train_step_end()
    ol.on_train_batch_end()
    ol.on_train_end()
    assert rec.names == [
        "on_train_step_start",
        "on_train_batch_start",
        "on_train_batch_end",
        "on_train_step_end",
        "on_train_end",
    ]


@pytest.mark.parametrize("fail_on", ["on_train_step_end", "on_save_checkpoint_start", "on_app_start"])
def test_a_raising_hook_disables_reporting_but_app_end_still_flushes_and_releases(fail_on: str):
    rec = _RecordingOneLogger(fail_on=fail_on)
    ol, fake = _one_logger(rec)
    ol.on_app_start()
    ol.on_train_start(train_iterations_start=0, train_samples_start=0)
    ol.on_train_step_start()
    ol.on_train_batch_start()
    ol.on_train_batch_end()
    ol.on_train_step_end()
    ol.on_checkpoint_save_start(1)  # raises inside OneLogger for one parametrization; must be swallowed
    assert not ol.active
    before = list(rec.calls)
    ol.on_checkpoint_save_end(1)
    ol.on_checkpoint_save_success(1)
    ol.on_train_end()
    assert rec.calls == before, "after the first failure every hook is a no-op"
    assert ol.initialized, "the global OneLogger object still owes its teardown"
    ol.on_app_end()
    assert rec.names[-2:] == ["on_app_end", "finish"], "buffered metrics are flushed after a hook failure"
    assert fake.state["ol"] is None and not ol.initialized, "the global singleton is released"


def test_a_failing_teardown_still_releases_the_global_object():
    rec = _RecordingOneLogger(fail_on="on_app_end")
    ol, fake = _one_logger(rec)
    ol.on_app_start()
    ol.on_app_end()
    assert rec.names == ["on_app_start", "finish"] and fake.state["ol"] is None


def test_builder_is_none_unless_asked_for_and_reads_the_run(monkeypatch):
    rl = SimpleNamespace(batch_per_replica=32, n_mini=2, n_micro=4, max_steps=300)
    config = SimpleNamespace(
        job=SimpleNamespace(project="cosmos_reason2", group="carla_pivot", name="grpo_run"),
        rollout=SimpleNamespace(max_tokens=3072),
    )
    common = dict(controller="grpo", world_size=8, topology="POLICY_1x4_ROLLOUT_1x4_disaggregated", policy_dp_world=4)
    assert build_one_logger_observer(config, rl, environ={}, **common) is None

    rec = _RecordingOneLogger()
    fake = _fake_utils(rec)
    monkeypatch.setattr("cosmos_framework.posttrain.callbacks.one_logger.load_one_logger_utils", lambda: fake)
    monkeypatch.setattr("cosmos_framework.posttrain.callbacks.one_logger.gpu_name", lambda: "NVIDIA-GB300")
    on = build_one_logger_observer(
        config, rl, environ={"ENABLE_ONELOGGER": "true", "SLURM_JOB_ID": "1"}, app_start_time_ms=5, **common
    )
    assert isinstance(on, OneLoggerLifecycleObserver)
    sent = on.config
    assert sent["one_logger_run_name"] == "cosmos_reason2/carla_pivot/grpo_run"
    assert sent["train_iterations_target"] == 600 and sent["train_samples_target"] == 9600
    assert sent["world_size"] == 8 and sent["global_batch_size"] == 4 and sent["batch_size"] == 1
    assert sent["seq_length"] == 3072 and sent["app_start_time"] == 5
    assert sent["app_tag"].startswith("cosmos_reason2/carla_pivot/grpo_run/grpo/BS_32_nMINI_2_nMICRO_4_nGPUs_8_")

    # Missing wheel: the builder reports and returns None instead of raising.
    monkeypatch.setattr("cosmos_framework.posttrain.callbacks.one_logger.load_one_logger_utils", lambda: None)
    assert build_one_logger_observer(config, rl, environ={"ENABLE_ONELOGGER": "true"}, **common) is None

    # The generic assembly adds OneLogger only when the run asks for it.
    monkeypatch.delenv("ENABLE_ONELOGGER", raising=False)
    assert len(build_controller_observers(config, rl, **common)) == 0


def test_controller_observers_build_is_best_effort(monkeypatch):
    """A backend whose build raises is logged and left out: OneLogger's own run-shape check (a real
    ValueError from `controller_one_logger_config`) and any other failure of the build both leave the
    entry with the empty list instead of aborting the run."""
    warnings: list[str] = []
    monkeypatch.setattr(lifecycle_module.logger, "warning", warnings.append)
    rec = _RecordingOneLogger()
    fake = _fake_utils(rec)
    monkeypatch.setenv("ENABLE_ONELOGGER", "true")
    monkeypatch.setattr("cosmos_framework.posttrain.callbacks.one_logger.load_one_logger_utils", lambda: fake)
    monkeypatch.setattr("cosmos_framework.posttrain.callbacks.one_logger.gpu_name", lambda: "none")
    config = SimpleNamespace(
        job=SimpleNamespace(project="p", group="g", name="n"), rollout=SimpleNamespace(max_tokens=0)
    )
    common = dict(controller="grpo", world_size=1, topology="POLICY_1x1_ROLLOUT_1x1_colocated")
    bad_shape = SimpleNamespace(batch_per_replica=30, n_mini=2, n_micro=4, max_steps=3)  # 30 % (2*4) != 0

    with pytest.raises(ValueError, match="not a multiple"):
        build_one_logger_observer(config, bad_shape, **common)  # the builder itself raises ...
    observers = build_controller_observers(config, bad_shape, **common)  # ... and the assembly absorbs it
    assert isinstance(observers, ControllerLifecycleObservers) and len(observers) == 0
    assert len(warnings) == 1 and "ValueError" in warnings[0] and "not a multiple" in warnings[0]
    assert fake.state["ol"] is None, "OneLogger was never initialized"

    def _broken_build(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("injected build failure")

    monkeypatch.setattr("cosmos_framework.posttrain.callbacks.one_logger.build_one_logger_observer", _broken_build)
    good_shape = SimpleNamespace(batch_per_replica=32, n_mini=2, n_micro=4, max_steps=3)
    assert len(build_controller_observers(config, good_shape, **common)) == 0
    assert len(warnings) == 2 and "RuntimeError: injected build failure" in warnings[1]


# --------------------------------------------------------------------------- the real grpo.loop
class _LoopPolicy:
    """The fake policy the GRPO loop tests drive: optimizer steps on the last micro call of a mini."""

    def __init__(self, *, n_micro: int, iteration: int = 0, fail_save: bool = False) -> None:
        self._it, self._acc, self.grad_accum = iteration, 0, n_micro
        self.saved: list[str] = []
        self.fail_save = fail_save

    def iteration(self) -> int:
        return self._it

    def needs_old_logprobs(self) -> bool:
        return False  # the ahead-pass capability probe the driver runs at n_mini > 1

    def train(self, batch, *, mini=0, micro=0, n_mini=1, n_micro=1):
        del batch, mini, micro, n_mini, n_micro
        self._acc += 1
        if self._acc == self.grad_accum:
            self._acc, self._it = 0, self._it + 1
        return {"train/loss_avg": 0.5}

    def save(self, path: str, **_kwargs: Any) -> None:
        if self.fail_save:
            raise RuntimeError("injected save failure")
        self.saved.append(path)


def _loop_units(policy: Any) -> dict[str, Any]:
    from cosmos_framework.posttrain.utils.batch_stats import RewardBatchStats

    class _Rollout:
        def generate(self, prompts, group):
            del prompts, group
            return []

        @staticmethod
        def prepare_weight_version(_model_step: int, _producer_identity: str | None) -> None:
            return None

        @staticmethod
        def commit_weight_version(_model_step: int) -> None:
            return None

    report = RewardBatchStats(
        trajectories=2,
        reward_sum=1.0,
        reward_m2=0.5,
        reward_min=0.0,
        reward_max=1.0,
        groups=1,
        informative_groups=1,
        tied_groups=0,
        all_zero_groups=0,
        all_one_groups=0,
    )
    policy.push_weights = lambda rollout: None
    policy.shared_weight_identity = lambda: None
    return {
        "data": SimpleNamespace(get_samples=lambda k: []),
        "rollout": _Rollout(),
        "rm": SimpleNamespace(score_with_report=lambda t: t, batch_stats=lambda: report),
        "replay": SimpleNamespace(put_samples=lambda s: None, get_samples=lambda k: []),
        "policy": policy,
    }


def test_grpo_loop_takes_observers_by_keyword_with_a_silent_default():
    from cosmos_framework.posttrain.entrypoints import grpo

    params = inspect.signature(grpo.loop).parameters
    assert "observers" in params and params["observers"].default is None
    assert params["observers"].kind is inspect.Parameter.KEYWORD_ONLY
    assert "one_logger" not in params, "the loop knows lifecycle observers, not a telemetry vendor"


def test_grpo_loop_emits_one_step_per_optimizer_update_and_one_batch_per_micro_call(monkeypatch, tmp_path: Path):
    """n_mini=2, n_micro=2, max_steps=2, save_every=1: 4 train steps of 2 train batches each, in mini/micro
    order, and the coordinator's save events after each collected batch (model steps 2 and 4)."""
    from cosmos_framework.posttrain.entrypoints import grpo

    mock_loop_env(monkeypatch)
    generic = _RecordingObserver()
    rec = _RecordingOneLogger()
    ol, fake = _one_logger(rec, batch_per_replica=4, n_mini=2, n_micro=2, max_steps=2, world_size=1, policy_dp_world=1)
    ol.on_app_start()
    policy = _LoopPolicy(n_micro=2)
    config = loop_config(max_steps=2, prompts_per_step=2, group_size=2, n_mini=2, n_micro=2, save_every=1)
    summary = grpo.loop(_loop_units(policy), config, str(tmp_path), observers=[generic, ol])
    ol.on_app_end()

    assert summary["steps"] == 2 and policy.iteration() == 4
    assert policy.saved == [
        str(tmp_path / "checkpoints" / "iter_000002"),
        str(tmp_path / "checkpoints" / "iter_000004"),
    ]
    expected = [("train_start", 0, 0)]
    for model_step in (2, 4):
        expected += [(name,) for name in _STEP] * 2
        expected += [("save_start", model_step), ("save_end", model_step), ("save_success", model_step)]
    expected.append(("train_end",))
    assert generic.events == expected
    assert generic.names().count("step_start") == 4 == generic.names().count("step_end")  # max_steps x n_mini
    assert generic.names().count("batch_start") == 8 == generic.names().count("batch_end")  # x n_micro

    save_triple = ["on_save_checkpoint_start", "on_save_checkpoint_end", "on_save_checkpoint_success"]
    assert rec.names == [
        "on_app_start",
        "on_train_start",
        *_ONE_LOGGER_STEP,
        *_ONE_LOGGER_STEP,
        *save_triple,
        *_ONE_LOGGER_STEP,
        *_ONE_LOGGER_STEP,
        *save_triple,
        "on_train_end",
        "on_app_end",
        "finish",
    ]
    assert [kw["global_step"] for name, kw in rec.calls if name == "on_save_checkpoint_success"] == [2, 4]
    assert fake.state["ol"] is None


def test_grpo_loop_resume_reports_the_optimizer_step_and_per_replica_samples(monkeypatch, tmp_path: Path):
    """Restored optimizer step 2 at n_mini=2 is collected batch 1: train_iterations_start is the optimizer
    step and train_samples_start one collected batch (4 trajectories), NOT multiplied by the two rollout
    data replicas the grid declares."""
    from cosmos_framework.posttrain.entrypoints import grpo

    mock_loop_env(monkeypatch)
    generic = _RecordingObserver()
    config = loop_config(max_steps=2, prompts_per_step=2, group_size=2, n_mini=2, n_micro=1, save_every=0)
    config.grid.rollout_shape = lambda: (2, 1)  # two rollout/data replicas share the 4-trajectory batch
    summary = grpo.loop(_loop_units(_LoopPolicy(n_micro=1, iteration=2)), config, str(tmp_path), observers=generic)
    assert summary["start"] == 1 and summary["start_model_step"] == 2
    assert generic.events[0] == ("train_start", 2, 4)
    assert generic.names().count("step_start") == 2  # only the remaining collected batch: n_mini updates


@pytest.mark.parametrize("save_every", [0, 1])
def test_grpo_loop_emergency_save_is_reported_once_even_when_the_cadence_already_saved(
    monkeypatch, tmp_path: Path, save_every: int
):
    """Preemption after the first batch: `ensure_resumable` is an actual save only when no cadence save
    committed that step (save_every=0); with save_every=1 it deduplicates and emits nothing."""
    from cosmos_framework.posttrain.entrypoints import grpo

    mock_loop_env(monkeypatch, stops=["signal"])
    generic = _RecordingObserver()
    config = loop_config(max_steps=3, prompts_per_step=1, group_size=2, n_mini=1, n_micro=1, save_every=save_every)
    summary = grpo.loop(_loop_units(_LoopPolicy(n_micro=1)), config, str(tmp_path), observers=generic)
    assert summary["preempted"] == "signal"
    saves = [event for event in generic.events if event[0].startswith("save_")]
    assert saves == [("save_start", 1), ("save_end", 1), ("save_success", 1)]
    assert generic.events[-1] == ("train_end",)


@pytest.mark.parametrize("raise_in", ["setup", "resume"])
def test_grpo_loop_never_leaves_the_train_interval_open_when_setup_raises(monkeypatch, tmp_path: Path, raise_in: str):
    """on_train_start is emitted inside the region whose `finally` emits on_train_end, so the two always
    pair: a raise in the setup before it (the checkpoint lifecycle here) opens no interval, and a raise
    in the first protected step (the resume / initial-validation boundary) gets exactly one train_end.
    The exception itself still propagates; the observers never swallow a loop failure."""
    from cosmos_framework.posttrain.control.validation import boundary
    from cosmos_framework.posttrain.entrypoints import grpo

    mock_loop_env(monkeypatch)

    def _raise(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError(f"injected {raise_in} failure")

    if raise_in == "setup":
        monkeypatch.setattr(grpo.CheckpointLifecycle, "from_config", _raise)
    else:
        monkeypatch.setattr(boundary.ValidationBoundaryCoordinator, "resume_or_initial", _raise)
    generic = _RecordingObserver()
    rec = _RecordingOneLogger()
    ol, fake = _one_logger(rec, batch_per_replica=2, n_mini=1, n_micro=1, max_steps=2, world_size=1, policy_dp_world=1)
    ol.on_app_start()
    config = loop_config(max_steps=2, prompts_per_step=1, group_size=2, save_every=0)
    with pytest.raises(RuntimeError, match=f"injected {raise_in} failure"):
        grpo.loop(_loop_units(_LoopPolicy(n_micro=1)), config, str(tmp_path), observers=[generic, ol])
    ol.on_app_end()

    opened = 1 if raise_in == "resume" else 0
    assert generic.events == [("train_start", 0, 0), ("train_end",)][: 2 * opened]
    assert generic.names().count("train_start") == generic.names().count("train_end") == opened
    assert rec.names == ["on_app_start", *["on_train_start", "on_train_end"][: 2 * opened], "on_app_end", "finish"]
    assert not ol.initialized and fake.state["ol"] is None


@pytest.mark.parametrize("raise_in", ["train", "skipped_update"])
def test_grpo_loop_closes_the_batch_and_step_intervals_when_a_micro_call_raises(
    monkeypatch, tmp_path: Path, raise_in: str
):
    """Every `mini` window and every `micro` call is bracketed by a start and a `finally` end, so a
    raise inside the train call (a failed worker) or in the skipped-update handling right after it
    (the loop's own RuntimeError) reaches `on_train_end` with nothing open: the emitter itself closes
    the batch and the step (batch_end, step_end, train_end), the counts balance, and the exception
    still propagates. The second micro call of the first optimizer window fails at n_mini=2, n_micro=2."""
    from cosmos_framework.posttrain.entrypoints import grpo

    mock_loop_env(monkeypatch)

    class _FailingPolicy(_LoopPolicy):
        def train(self, batch, *, mini=0, micro=0, n_mini=1, n_micro=1):
            if (mini, micro) == (0, 1):
                if raise_in == "train":
                    raise RuntimeError("injected train failure")
                return {"train/loss_avg": 0.5, "optim/update_outcome": "skipped", "optim/skip_reason": "nonfinite"}
            return super().train(batch, mini=mini, micro=micro, n_mini=n_mini, n_micro=n_micro)

    generic = _RecordingObserver()
    rec = _RecordingOneLogger()
    ol, fake = _one_logger(rec, batch_per_replica=4, n_mini=2, n_micro=2, max_steps=2, world_size=1, policy_dp_world=1)
    ol.on_app_start()
    config = loop_config(max_steps=2, prompts_per_step=2, group_size=2, n_mini=2, n_micro=2, save_every=0)
    match = "injected train failure" if raise_in == "train" else "rejected non-finite GRPO update"
    with pytest.raises(RuntimeError, match=match):
        grpo.loop(_loop_units(_FailingPolicy(n_micro=2)), config, str(tmp_path), observers=[generic, ol])
    ol.on_app_end()

    assert generic.events == [
        ("train_start", 0, 0),
        ("step_start",),
        ("batch_start",),
        ("batch_end",),
        ("batch_start",),
        ("batch_end",),  # the failing call's `finally`
        ("step_end",),  # the window's `finally`
        ("train_end",),  # the loop's `finally`
    ]
    names = generic.names()
    assert names.count("step_start") == names.count("step_end") == 1
    assert names.count("batch_start") == names.count("batch_end") == 2
    # OneLogger sees the same closed intervals once: the emitter closed them, the adapter's own
    # on_train_end compensation found nothing open and added nothing.
    assert rec.names == [
        "on_app_start",
        "on_train_start",
        *_ONE_LOGGER_STEP,
        "on_train_end",
        "on_app_end",
        "finish",
    ]
    assert not ol.initialized and fake.state["ol"] is None


# --------------------------------------------------------------------------- the checkpoint boundary
def test_coordinator_emits_save_events_only_for_actual_saves(tmp_path: Path):
    """Deduplicated cadence saves, the validation transaction's `save_full(publish=False)`, and the
    emergency `ensure_resumable` upgrade: only a write that happens is reported, and a failed write
    gets its end without a success."""
    from cosmos_framework.posttrain.checkpoint import CheckpointManager
    from cosmos_framework.posttrain.control.checkpointing import CheckpointCoordinator

    class _Policy:
        def __init__(self) -> None:
            self.fail = False
            self.saves = 0

        def save(self, _checkpoint: str, *_args: Any, **_kwargs: Any) -> None:
            if self.fail:
                raise RuntimeError("injected save failure")
            self.saves += 1

    rec = _RecordingObserver()
    policy = _Policy()
    with pytest.warns(UserWarning, match="partial checkpoint artifact"):
        manager = CheckpointManager(checkpoint_components=["model"])  # cadence saves are partial
    coordinator = CheckpointCoordinator(out_dir=str(tmp_path), policy=policy, manager=manager, observer=rec)

    assert coordinator.save_configured_if_needed(1)
    assert rec.events == [("save_start", 1), ("save_end", 1), ("save_success", 1)]
    assert not coordinator.save_configured_if_needed(1), "already committed, e.g. by a coincident validation"
    assert len(rec.events) == 3, "a deduplicated request reports nothing"
    assert coordinator.ensure_resumable(1), "the emergency path upgrades the partial save: an actual write"
    assert rec.events[3:] == [("save_start", 1), ("save_end", 1), ("save_success", 1)]
    assert not coordinator.ensure_resumable(1) and len(rec.events) == 6

    coordinator.save_full(2, publish=False)  # what ValidationTransaction.run/defer call
    assert rec.events[6:] == [("save_start", 2), ("save_end", 2), ("save_success", 2)]
    assert not coordinator.save_configured_if_needed(2) and not coordinator.ensure_resumable(2)
    assert len(rec.events) == 9 and policy.saves == 3

    policy.fail = True
    with pytest.raises(RuntimeError, match="injected save failure"):
        coordinator.save(3)
    assert rec.events[9:] == [("save_start", 3), ("save_end", 3)], "no success for a save that raised"
    policy.fail = False
    assert coordinator.save_configured_if_needed(3), "the failed write did not advance the committed step"
    assert rec.events[11:] == [("save_start", 3), ("save_end", 3), ("save_success", 3)]


def test_coordinator_with_a_full_configured_plan_reports_each_step_once(tmp_path: Path):
    from cosmos_framework.posttrain.checkpoint import CheckpointManager
    from cosmos_framework.posttrain.control.checkpointing import CheckpointCoordinator, CheckpointLifecycle

    rec = _RecordingObserver()
    policy = SimpleNamespace(save=lambda _checkpoint, *_args, **_kwargs: None)
    lifecycle = CheckpointLifecycle(manager=CheckpointManager(), checkpoint_root=str(tmp_path), resume_checkpoint=None)
    coordinator = lifecycle.coordinator(str(tmp_path), policy, initial_step=0, observer=rec)
    assert isinstance(coordinator, CheckpointCoordinator)
    assert coordinator.save_configured_if_needed(1)
    assert not coordinator.ensure_resumable(1), "a full cadence save already made the step resumable"
    assert rec.events == [("save_start", 1), ("save_end", 1), ("save_success", 1)]
    # A raising observer never fails the save.
    failing = lifecycle.coordinator(str(tmp_path / "other"), policy, observer=_RaisingObserver())
    assert failing.save_configured_if_needed(1)


def test_multi_role_coordinator_emits_save_events_only_for_a_new_boundary(tmp_path: Path):
    """PPO's boundary: a written boundary is one save; adopting an already-committed boundary emits
    nothing; a failed role write gets end without success and publishes nothing."""
    import os

    import torch

    from cosmos_framework.posttrain.checkpoint import CheckpointManager
    from cosmos_framework.posttrain.control.checkpointing import MultiRoleCheckpointCoordinator
    from cosmos_framework.posttrain.schema.checkpoint import FULL_CHECKPOINT_COMPONENTS
    from cosmos_framework.posttrain.workers.worker_group import WorkerGroup

    class _Role:
        def __init__(self, *, fail: bool = False) -> None:
            self.fail = fail

        def save(self, checkpoint: str, *, publish: bool, update_latest: bool, components, data_state=None) -> None:
            assert not publish and not update_latest and components == FULL_CHECKPOINT_COMPONENTS
            if self.fail:
                raise RuntimeError("injected role save failure")
            os.makedirs(checkpoint)
            for name in ("model", "optim"):
                torch.save({"name": name}, os.path.join(checkpoint, f"{name}.pt"))
            torch.save({"name": "trainer", "iter": 1, "optimizer_step": 1}, os.path.join(checkpoint, "trainer.pt"))

    def _coordinator(rec: _RecordingObserver, *, fail_critic: bool = False) -> MultiRoleCheckpointCoordinator:
        roles = {
            "policy": WorkerGroup("policy", "trainer", _Role()),
            "critic": WorkerGroup("critic", "critic", _Role(fail=fail_critic)),
        }
        return MultiRoleCheckpointCoordinator(
            checkpoint_root=str(tmp_path / "checkpoints"),
            roles=roles,
            manager=CheckpointManager(checkpoint_format="consolidated", keep_optim_last=1),
            observer=rec,
        )

    rec = _RecordingObserver()
    coordinator = _coordinator(rec)
    assert coordinator.save_configured_if_needed(1, {"critic_warmup_iterations": 0})
    assert rec.events == [("save_start", 1), ("save_end", 1), ("save_success", 1)]
    assert not coordinator.save_configured_if_needed(1, {"critic_warmup_iterations": 0}) and len(rec.events) == 3

    adopting = _RecordingObserver()
    _coordinator(adopting).save(1, {"critic_warmup_iterations": 0})  # a retry adopting the committed boundary
    assert adopting.events == [], "adopting an immutable committed boundary is not a save"

    failing = _RecordingObserver()
    with pytest.raises(RuntimeError, match="injected role save failure"):
        _coordinator(failing, fail_critic=True).save(2)
    assert failing.events == [("save_start", 2), ("save_end", 2)]
    assert not os.path.exists(tmp_path / "checkpoints" / "iter_000002")


# --------------------------------------------------------------------------- resolved placement + OPD wiring
def test_placement_gpu_count_counts_every_gpu_device_group_once():
    from cosmos_framework.posttrain.control.placement import ActorGroup, DeviceGroup, placement_gpu_count
    from cosmos_framework.posttrain.workers.worker_group import WorkerGroup

    def _actor(name: str, *members: WorkerGroup) -> ActorGroup:
        return ActorGroup(members=members, setup=lambda: {}, name=name)

    def _groups(*, rollout_ranks: int, policy_ranks: int, colocated: bool, device: str = "gpu") -> list[DeviceGroup]:
        experience = _actor(
            "experience",
            WorkerGroup("data", "env"),
            WorkerGroup("rollout", "rollout", ranks=rollout_ranks),
            WorkerGroup("rm", "reward"),
        )
        teacher = _actor("teacher", WorkerGroup("teacher", "teacher"))
        policy = _actor("policy", WorkerGroup("policy", "trainer", ranks=policy_ranks))
        if colocated:
            return [DeviceGroup((experience, policy), device=device, name="shared"), DeviceGroup((teacher,), device)]
        return [DeviceGroup((experience,), device), DeviceGroup((teacher,), device), DeviceGroup((policy,), device)]

    # Colocated policy+experience share two cards; the OPD teacher adds its own: 3, not policy+rollout=4.
    assert placement_gpu_count(_groups(rollout_ranks=2, policy_ranks=2, colocated=True)) == 3
    # Disaggregated: 1 experience + 1 teacher + 4 policy.
    assert placement_gpu_count(_groups(rollout_ranks=1, policy_ranks=4, colocated=False)) == 6
    # The local/torchrun placement has no GPU device groups.
    assert placement_gpu_count(_groups(rollout_ranks=1, policy_ranks=1, colocated=True, device="cpu")) == 0
    with pytest.raises(TypeError):
        placement_gpu_count(["not a device group"])  # type: ignore[list-item]


def _stub_entry_plumbing(
    monkeypatch: pytest.MonkeyPatch, entry: Any, rec: _RecordingOneLogger, *, finalize_units: Any = None
) -> dict[str, Any]:
    """The stubs both `run()` harnesses share: the placement is not bound, the loop records what it was
    handed, the checkpoint drain records whether the app was still open (or is ``finalize_units`` when given).
    Returns the capture dict."""
    from cosmos_framework.posttrain.control.checkpointing import CheckpointLifecycle

    captured: dict[str, Any] = {}
    monkeypatch.setattr("cosmos_framework.posttrain.callbacks.one_logger.gpu_name", lambda: "none")
    monkeypatch.setattr(
        "cosmos_framework.posttrain.control.validation.rollout.validate_rollout_reward_sources",
        lambda _rl, _catalog: None,
    )
    monkeypatch.setattr(
        "cosmos_framework.posttrain.control.placement.bind_device_groups",
        lambda device_groups, **kwargs: captured.update(device_groups=device_groups),
    )
    monkeypatch.setattr(
        "cosmos_framework.posttrain.utils.entry.pin_run_attempt",
        lambda _config, opts, out_dir: SimpleNamespace(opts=opts, out_dir=out_dir),
    )
    monkeypatch.setattr(entry, "dump_resolved", lambda *_args, **_kwargs: None)

    def _loop(_wgs, _config, _out_dir, **kwargs):
        captured["loop_kwargs"] = kwargs
        captured["app_started_before_loop"] = rec.names == ["on_app_start"]
        return {}

    monkeypatch.setattr(entry, "loop", _loop)
    monkeypatch.setattr(
        CheckpointLifecycle,
        "finalize_units",
        finalize_units
        if finalize_units is not None
        else lambda wgs, teardown: captured.update(finalized_before_app_end=("on_app_end" not in rec.names)),
    )
    return captured


def _run_opd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, rec: _RecordingOneLogger, *, finalize_units: Any = None
) -> dict[str, Any]:
    """The real `opd.run` over a disaggregated placement (experience 1 GPU, teacher 1 GPU, policy 4 GPUs)."""
    from cosmos_framework.posttrain.config.model import HFModelConfig
    from cosmos_framework.posttrain.config.rl import OPDConfig, RolloutConfig
    from cosmos_framework.posttrain.entrypoints import opd

    captured = _stub_entry_plumbing(monkeypatch, opd, rec, finalize_units=finalize_units)
    rollout = RolloutConfig(tensor_parallel_size=1)
    config = SimpleNamespace(
        model_backend="hf",
        model_hf=HFModelConfig(name_or_path=rollout.name_or_path),
        rl=OPDConfig(
            weight_sync_transport="nixl",
            weight_sync_bucket_bytes=64 * 1024 * 1024,
            weight_sync_probe_bytes=8 * 1024 * 1024,
            weight_sync_min_gbps=7.5,
        ),
        rollout=rollout,
        grid=SimpleNamespace(
            backend="ray",
            gpu_placement="disaggregated",
            shapes={"policy": (1, 4), "rollout": (1, 1)},
            num_replicas=1,
            policy_shape=lambda: (1, 4),
            rollout_shape=lambda: (1, 1),
        ),
        job=SimpleNamespace(project="p", group="g", name="n"),
    )
    opd.run(config, "unused.py", [], str(tmp_path))
    captured["config"] = config
    return captured


def _run_grpo(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, rec: _RecordingOneLogger, *, finalize_units: Any = None
) -> dict[str, Any]:
    """The real `grpo.run` over the default colocated one-GPU placement."""
    from cosmos_framework.posttrain.config.model import HFModelConfig
    from cosmos_framework.posttrain.config.rl import GridConfig, GRPOConfig, RolloutConfig
    from cosmos_framework.posttrain.entrypoints import grpo

    captured = _stub_entry_plumbing(monkeypatch, grpo, rec, finalize_units=finalize_units)
    monkeypatch.setattr(grpo, "make_setup", lambda *_args, **_kwargs: lambda: {})
    monkeypatch.setattr("cosmos_framework.utils.lazy_config.instantiate", lambda value: value)
    config = SimpleNamespace(
        grid=GridConfig(backend="ray", ranks=1, num_replicas=1, gpu_placement="colocated"),
        rollout=RolloutConfig(tensor_parallel_size=1, name_or_path=""),
        model_backend="hf",
        model_hf=HFModelConfig(),
        rl=GRPOConfig(),
        job=SimpleNamespace(project="p", group="g", name="n"),
    )
    grpo.run(config, "unused.py", [], str(tmp_path), build_objective_fn=grpo.build_objective)
    captured["config"] = config
    return captured


@pytest.mark.parametrize("enabled", [True, False])
def test_opd_run_reports_through_the_lifecycle_observers_with_teacher_gpus_in_world_size(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, enabled: bool
) -> None:
    """opd.run builds the same observers grpo.run does, hands them to the shared loop, opens the app
    before the loop and closes it after the checkpoint drain. The world size is the bound placement's
    GPU count (experience 1 + teacher 1 + policy 4), which role shapes alone could not see."""
    rec = _RecordingOneLogger()
    fake = _fake_utils(rec)
    if enabled:
        monkeypatch.setenv("ENABLE_ONELOGGER", "true")
    else:
        monkeypatch.delenv("ENABLE_ONELOGGER", raising=False)
    monkeypatch.setattr("cosmos_framework.posttrain.callbacks.one_logger.load_one_logger_utils", lambda: fake)

    captured = _run_opd(monkeypatch, tmp_path, rec)

    config = captured["config"]
    observers = captured["loop_kwargs"]["observers"]
    assert isinstance(observers, ControllerLifecycleObservers)
    assert len(captured["device_groups"]) == 3, "experience, teacher, and policy device groups were bound"
    if not enabled:
        assert len(observers) == 0 and rec.calls == []
        return
    (one_logger,) = list(observers)
    assert isinstance(one_logger, OneLoggerLifecycleObserver)
    sent = one_logger.config
    assert sent["world_size"] == 6, "1 experience GPU + 1 teacher GPU + 4 policy GPUs from the bound placement"
    assert sent["train_iterations_target"] == config.rl.max_steps * config.rl.n_mini
    assert sent["train_samples_target"] == config.rl.max_steps * config.rl.batch_per_replica
    assert sent["batch_size"] == sent["global_batch_size"] // 4, "the per-DP-rank share over POLICY_RANKS=4"
    assert "/opd/" in sent["app_tag"] and "_TEACHER_1x1_disaggregated/" in sent["app_tag"]
    assert captured["app_started_before_loop"] and captured["finalized_before_app_end"]
    assert rec.names == ["on_app_start", "on_app_end", "finish"] and fake.state["ol"] is None


@pytest.mark.parametrize("run_entry", [_run_grpo, _run_opd], ids=["grpo", "opd"])
def test_run_proceeds_with_no_observers_when_the_telemetry_build_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, run_entry: Any
) -> None:
    """The observer build never aborts an entry: with OneLogger asked for and its config raising
    (injected; the run-shape checks are unreachable past GRPOConfig/OPDConfig validation), `run()` logs
    one warning, hands the loop the empty list and still goes through the loop and the checkpoint drain."""
    warnings: list[str] = []
    monkeypatch.setattr(lifecycle_module.logger, "warning", warnings.append)
    rec = _RecordingOneLogger()
    fake = _fake_utils(rec)
    monkeypatch.setenv("ENABLE_ONELOGGER", "true")
    monkeypatch.setattr("cosmos_framework.posttrain.callbacks.one_logger.load_one_logger_utils", lambda: fake)

    def _broken_config(**_kwargs: Any) -> None:
        raise ValueError("injected OneLogger config failure")

    monkeypatch.setattr("cosmos_framework.posttrain.callbacks.one_logger.controller_one_logger_config", _broken_config)

    captured = run_entry(monkeypatch, tmp_path, rec)

    observers = captured["loop_kwargs"]["observers"]
    assert isinstance(observers, ControllerLifecycleObservers) and len(observers) == 0
    assert captured["finalized_before_app_end"] is True, "the run went through the loop and the checkpoint drain"
    assert rec.calls == [] and fake.state["ol"] is None, "OneLogger was never initialized"
    assert len(warnings) == 1 and "ValueError: injected OneLogger config failure" in warnings[0]


@pytest.mark.parametrize("run_entry", [_run_grpo, _run_opd], ids=["grpo", "opd"])
def test_run_closes_the_app_when_the_checkpoint_drain_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, run_entry: Any
) -> None:
    """The entry's `finally` nests the checkpoint drain in a try/finally around `on_app_end()`: with the loop
    succeeded and the real finalization helper draining a trainer role whose asynchronous checkpoint write
    failed, the failure propagates out of `run()` unchanged and the observers' app end (OneLogger's
    on_app_end / finish / destroy) still runs, after the drain and the process-group teardown."""
    from cosmos_framework.posttrain.control.checkpointing import finalize_checkpoints_before_teardown

    rec = _RecordingOneLogger()
    fake = _fake_utils(rec)
    monkeypatch.setenv("ENABLE_ONELOGGER", "true")
    monkeypatch.setattr("cosmos_framework.posttrain.callbacks.one_logger.load_one_logger_utils", lambda: fake)

    class _FailedWrite:
        """A checkpoint handle whose one cell reports the asynchronous write's failure."""

        @staticmethod
        def as_completed():
            yield "policy-cell-0", RuntimeError("injected checkpoint failure")

    class _Trainer:
        def __init__(self) -> None:
            self.role = "trainer"

        @staticmethod
        def finalize_checkpoints() -> _FailedWrite:
            return _FailedWrite()

    order: list[str] = []

    def _finalize(wgs: Any, teardown: Any) -> None:
        order.append("drain")
        finalize_checkpoints_before_teardown({"policy": _Trainer()}, lambda: order.append("teardown"))

    with pytest.raises(RuntimeError, match="injected checkpoint failure"):
        run_entry(monkeypatch, tmp_path, rec, finalize_units=_finalize)

    assert order == ["drain", "teardown"], "the real helper drained the role and still ran the teardown"
    assert rec.names == ["on_app_start", "on_app_end", "finish"], "the app was closed after the failing drain"
    assert fake.state["ol"] is None, "the global OneLogger object was released"
