# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Controller lifecycle observers (DESIGN §2.6): the generic event seam the online controllers emit on.

The controller loops (`entrypoints/grpo.py` and the entries that reuse it) and the checkpoint
coordinators (`control/checkpointing.py`) emit lifecycle EVENTS; they never know which telemetry
backend listens. `ControllerLifecycleObserver` is the observer protocol and, unmodified, its null
implementation. `ControllerLifecycleObservers` is the optional list of observers an entry hands the
loop; it fans every event out best effort and off the emitting thread (a bounded asynchronous
dispatcher, `control/dispatch.py`), so an observer can never fail or stall training (§2.6: observers
are off-hot-path and must not block; observability never gates the loop). OneLogger is one
implementation (`callbacks/one_logger.py`); `build_controller_observers` assembles the implementations a
run has asked for. GRPO/OPD construction is best effort; Joint PPO retains fail-fast admission
and synchronous initialization before creating model roles.

Two kinds of event, §2.6's two kinds of callback:

- The interval events (train step/batch, checkpoint save) are *observer* events: queued, delivered on
  the worker in emission order, dropped when the queue is full. Every queued event carries the clocks
  of the controller thread that emitted it (`LifecycleEvent.emitted_wall` / ``emitted_monotonic``), so
  an observer that measures intervals reads the controller's timeline whatever the delivery delay.
  Overflow keeps start/end pairs together (``LIFECYCLE_PAIRS``): a start is queued only with room for
  its end (and success), and a start that had to be dropped drops its end with it, so no observer ever
  sees an orphaned end or an interval left open by a dropped end.
- ``on_app_start`` is a synchronous admission gate before the worker starts. The terminal
  ``on_train_end`` and ``on_app_end`` gates wait boundedly for delivery. They carry the caller's clocks the
  moment they are called, take no room in the queue and are never dropped by a full queue: placed
  as non-droppable markers behind whatever is queued and delivered by the worker in emission order
  (starting the worker for early cleanup), so the boundaries that close intervals, flush buffered telemetry
  and tear the backends down reach every observer, in order, and never while the worker is still
  inside one of its hooks.

Event vocabulary, in imaginaire's trainer semantics so cross-job dashboards compare like with like:

- ``on_app_start`` / ``on_app_end``: the controller process, from the entry's resolved placement to
  its checkpoint drain. ``on_app_end`` is the cleanup point and is always called by the entry, also
  when the checkpoint drain itself fails.
- ``on_train_start`` / ``on_train_end``: the loop. Restored counters use the run's declared unit:
  optimizer updates for GRPO/OPD and successful publications for Joint PPO.
- ``on_train_step_start`` / ``on_train_step_end``: one declared training iteration: a GRPO/OPD
  optimizer update or a successfully published Joint PPO window.
- ``on_train_batch_start`` / ``on_train_batch_end``: one actual trainer call: a GRPO/OPD micro call
  or the Joint PPO prepared-window update. Only successful contexts emit their end event.
- ``on_checkpoint_save_start`` / ``on_checkpoint_save_end`` / ``on_checkpoint_save_success``: one
  actual checkpoint write at the coordinator's transaction boundary. ``end`` closes the interval
  whether or not the write succeeded; ``success`` follows only a committed write. A request the
  coordinator deduplicates (the step was already committed) emits nothing.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Literal

from cosmos_framework.posttrain.control.dispatch import BoundedEventDispatcher, LifecycleEvent
from cosmos_framework.posttrain.utils.log import logger

if TYPE_CHECKING:
    from cosmos_framework.posttrain.control.validation.lifecycle import ValidationReport, ValidationTrigger
    from cosmos_framework.posttrain.control.validation.transaction import ValidationEvaluator

# Start events and the followers that close them, for the dispatcher's pair-preserving overflow: a start
# is queued only with room for its followers, a dropped start drops its followers. The train interval
# is not listed because its end is a gate (never queued).
LIFECYCLE_PAIRS: dict[str, tuple[str, ...]] = {
    "on_train_step_start": ("on_train_step_end",),
    "on_train_batch_start": ("on_train_batch_end",),
    "on_validation_start": ("on_validation_end",),
    "on_checkpoint_save_start": ("on_checkpoint_save_end", "on_checkpoint_save_success"),
}

# The boundaries delivered as synchronous gates after a drain, never queued and never dropped.
GATE_EVENTS: tuple[str, ...] = ("on_app_start", "on_train_end", "on_app_end")


class ControllerLifecycleObserver:
    """No-op observer base: override only the event points you need. Unmodified, it is the null object."""

    fail_fast_start: bool = False

    @property
    def metadata(self) -> dict[str, object]:
        """Reporter metadata established by successful observer initialization."""
        return {}

    def observe(self, event: LifecycleEvent) -> None:
        """Deliver ``event`` to the matching hook. Override to read the controller's emission clocks
        (``event.emitted_wall`` / ``event.emitted_monotonic``) before the hook runs; the hooks themselves
        keep their signatures."""
        if event.name == "on_checkpoint_save_success" and "location" in event.kwargs:
            self.on_checkpoint_committed(*event.args, **event.kwargs)
        else:
            getattr(self, event.name)(*event.args, **event.kwargs)

    def on_app_start(self) -> None: ...

    def on_train_start(self, *, train_iterations_start: int, train_samples_start: int) -> None: ...

    def on_train_step_start(self) -> None: ...

    def on_train_step_end(self) -> None: ...

    def on_train_batch_start(self) -> None: ...

    def on_train_batch_end(self) -> None: ...

    def on_validation_start(self) -> None: ...

    def on_validation_end(self) -> None: ...

    def on_checkpoint_save_start(self, model_step: int) -> None: ...

    def on_checkpoint_save_end(self, model_step: int) -> None: ...

    def on_checkpoint_save_success(self, model_step: int) -> None: ...

    def on_checkpoint_committed(self, model_step: int, *, location: str) -> None:
        """A successful write's immutable location; legacy observers retain their hook."""
        self.on_checkpoint_save_success(model_step)

    def on_train_end(self) -> None: ...

    def on_app_end(self) -> None: ...


def deliver_event(observer: Any, event: LifecycleEvent) -> None:
    """Hand ``event`` to ``observer``: through its ``observe`` when it has one, else straight to the hook."""
    observe = getattr(observer, "observe", None)
    if callable(observe):
        observe(event)
    else:  # a duck-typed observer without the base class
        if event.name == "on_checkpoint_save_success" and "location" in event.kwargs:
            committed = getattr(observer, "on_checkpoint_committed", None)
            if committed is not None:
                committed(*event.args, **event.kwargs)
            else:
                observer.on_checkpoint_save_success(*event.args)
        else:
            getattr(observer, event.name)(*event.args, **event.kwargs)


class ControllerLifecycleObservers(ControllerLifecycleObserver):
    """The observer list a loop emits on: a bounded asynchronous fan-out no observer can fail or stall.

    Dispatch (``dispatch="async"``, the default): every interval event goes onto a bounded queue
    (`control/dispatch.py`, 1024 events) that one lazily created daemon worker thread drains in
    emission order, so the emitting controller thread never blocks and never waits for an observer
    (§2.6: observers are async, off-hot-path, must not block). Each queued event is stamped with the
    emitting thread's clocks (`LifecycleEvent`), which :meth:`ControllerLifecycleObserver.observe`
    hands to an observer that measures intervals. An event the full queue cannot take is dropped and
    counted (a warning on the first drop and at every power of ten), never queued behind a slow
    observer; start/end pairs are dropped together (``LIFECYCLE_PAIRS``), never split.

    Gates: ``on_app_start``, ``on_train_end`` and ``on_app_end`` are the §2.6 *gate* kind (sync,
    on-path, rare). Each is stamped with the caller's clocks at the call (its emission time, like every
    other event, so OneLogger's finish times exclude the wait) and handed to the dispatcher as a
    non-droppable marker behind everything queued (:meth:`BoundedEventDispatcher.deliver_gate`); the
    caller waits, bounded by ``DRAIN_TIMEOUT_S`` for terminal gates, for delivery in emission order. A
    gate therefore can neither be dropped by a full queue nor overtaken by queued events, and it never
    enters an observer while the worker is still inside one of its hooks: the intervals observers close
    at ``on_train_end`` are closed when the call returns, and ``on_app_end`` reaches the backends'
    teardown (OneLogger's finish/destroy) after every earlier event and before :meth:`shutdown` stops
    the worker and logs the drop total. ``on_app_start`` initialises the backends' singletons
    (OneLogger's global object) once, before the loop and before any worker exists, so it runs inline
    on the caller's thread and the entry and the tests read the result right after the call; the
    precedent is §11.2: "The GRPO reward-boundary validator in §1.5 is deliberately synchronous and
    fail-closed; only its already-computed scalar report feeds the telemetry sinks. It is not an
    off-hot-path observer." When the wait runs out (an observer parked inside a hook for longer than
    the bound), the caller returns with a warning naming the hook and the events queued ahead, and the
    gate stays queued: it is delivered, in order, when the observer returns, never forced onto the
    caller's thread while a hook is still running. ``on_app_end`` then bounds :meth:`shutdown` the
    same way, so a hung observer holds the controller for at most two bounds plus the join, never
    indefinitely.

    ``dispatch="sync"`` calls the observers inline, for deterministic unit tests and the checkpoint
    coordinators, which never see the train or app end and must not own a worker; :meth:`coerce`
    builds the list it wraps with the mode it is given and leaves an existing list's mode alone.

    A raising observer is logged and skipped for that event only; the other observers still see the
    event, and the observer itself keeps receiving later events (an implementation that wants to turn
    itself off after a failure does so inside its own hooks).
    """

    DRAIN_TIMEOUT_S = 30.0

    def __init__(
        self,
        observers: Iterable[ControllerLifecycleObserver] | None = None,
        *,
        dispatch: Literal["async", "sync"] = "async",
    ) -> None:
        if dispatch not in ("async", "sync"):
            raise ValueError(f"lifecycle observers: dispatch must be 'async' or 'sync', got {dispatch!r}")
        self._observers: list[ControllerLifecycleObserver] = list(observers or [])
        self._dispatch = dispatch
        self._dispatcher = BoundedEventDispatcher(self._deliver, pairs=LIFECYCLE_PAIRS) if dispatch == "async" else None

    @classmethod
    def coerce(cls, value: Any, *, dispatch: Literal["async", "sync"] = "async") -> ControllerLifecycleObservers:
        """The list for ``value``: the list itself (its own mode kept), one observer wrapped, an iterable
        collected, ``None`` empty; ``dispatch`` applies to the list built here."""
        if value is None:
            return cls(dispatch=dispatch)
        if isinstance(value, cls):
            return value
        if isinstance(value, ControllerLifecycleObserver):
            return cls([value], dispatch=dispatch)
        return cls(value, dispatch=dispatch)

    def add(self, observer: ControllerLifecycleObserver) -> None:
        self._observers.append(observer)

    def __len__(self) -> int:
        return len(self._observers)

    def __iter__(self) -> Iterator[ControllerLifecycleObserver]:
        return iter(self._observers)

    @property
    def metadata(self) -> dict[str, object]:
        return {key: value for observer in self._observers for key, value in getattr(observer, "metadata", {}).items()}

    @property
    def dispatch(self) -> str:
        """``"async"`` (bounded queue + worker thread) or ``"sync"`` (inline)."""
        return self._dispatch

    @property
    def dropped(self) -> int:
        """Events the full queue dropped so far, pairs counted per event (always 0 in sync mode)."""
        return 0 if self._dispatcher is None else self._dispatcher.dropped

    @property
    def dropped_pairs(self) -> int:
        """Start/end groups the full queue dropped together (always 0 in sync mode)."""
        return 0 if self._dispatcher is None else self._dispatcher.dropped_pairs

    @property
    def worker_alive(self) -> bool:
        """True while the dispatcher's worker thread exists (started lazily, stopped by :meth:`shutdown`)."""
        return self._dispatcher is not None and self._dispatcher.worker_alive

    def flush(self, timeout: float | None = DRAIN_TIMEOUT_S) -> bool:
        """Drain barrier: return once every event emitted so far was delivered (False on ``timeout``)."""
        return True if self._dispatcher is None else self._dispatcher.flush(timeout)

    def shutdown(self, timeout: float | None = DRAIN_TIMEOUT_S) -> bool:
        """Flush, stop the worker and join it, bounded by ``timeout``. Idempotent. Later events deliver inline once
        the worker has exited and are dropped (counted) while it still runs, so the queued events are never overtaken."""
        return True if self._dispatcher is None else self._dispatcher.shutdown(timeout)

    def _deliver(self, event: LifecycleEvent) -> None:
        """Hand ``event`` to every observer, guarding each: on the worker in async mode, inline otherwise."""
        for observer in tuple(self._observers):
            try:
                deliver_event(observer, event)
            except Exception as e:  # observers must never break the control loop (§2.6)
                if event.name == "on_app_start" and getattr(observer, "fail_fast_start", False):
                    raise
                logger.warning(f"lifecycle observer {type(observer).__name__}.{event.name} raised (ignored): {e}")

    def _fan(self, event: str, *args: Any, **kwargs: Any) -> None:
        if self._dispatcher is None or not self._observers:  # sync mode, or nobody listens: no thread
            self._deliver(LifecycleEvent.now(event, args, kwargs))
        else:
            self._dispatcher.submit(event, args, kwargs)

    def _gate(self, event: str) -> None:
        """Deliver terminal events behind earlier events, waiting at most ``DRAIN_TIMEOUT_S``: stamped now
        (the emission clock the observers' finish times read), never dropped by a full queue, never overtaken,
        never run concurrently with a hook the worker is still inside. Initial app-start is the synchronous
        admission gate; its SDK initialization has no dispatcher timeout."""
        item = LifecycleEvent.now(event)  # the emission boundary: stamped before any wait, like every other event
        if (
            self._dispatcher is None
            or not self._observers
            or (event == "on_app_start" and not self._dispatcher.worker_alive)
        ):  # app initialization stays synchronous and may fail before models are created
            self._deliver(item)
            return
        self._dispatcher.deliver_gate(item, self.DRAIN_TIMEOUT_S)  # warns when the bound ran out

    def on_app_start(self) -> None:
        """The §2.6 gate: the caller blocks until it is delivered (inline before any worker exists; behind the queue
        otherwise)."""
        self._gate("on_app_start")

    def on_train_start(self, *, train_iterations_start: int, train_samples_start: int) -> None:
        self._fan(
            "on_train_start",
            train_iterations_start=int(train_iterations_start),
            train_samples_start=int(train_samples_start),
        )

    def on_train_step_start(self) -> None:
        self._fan("on_train_step_start")

    def on_train_step_end(self) -> None:
        self._fan("on_train_step_end")

    def on_train_batch_start(self) -> None:
        self._fan("on_train_batch_start")

    def on_train_batch_end(self) -> None:
        self._fan("on_train_batch_end")

    def on_validation_start(self) -> None:
        self._fan("on_validation_start")

    def on_validation_end(self) -> None:
        self._fan("on_validation_end")

    @contextmanager
    def train_step(self) -> Iterator[None]:
        self.on_train_step_start()
        yield
        self.on_train_step_end()

    @contextmanager
    def train_batch(self) -> Iterator[None]:
        self.on_train_batch_start()
        yield
        self.on_train_batch_end()

    def evaluate(self, evaluator: ValidationEvaluator) -> ValidationEvaluator:
        def evaluate(model_step: int, reasons: frozenset[ValidationTrigger]) -> ValidationReport:
            self.on_validation_start()
            report = evaluator(model_step, reasons)
            self.on_validation_end()
            return report

        return evaluate

    def on_checkpoint_save_start(self, model_step: int) -> None:
        self._fan("on_checkpoint_save_start", int(model_step))

    def on_checkpoint_save_end(self, model_step: int) -> None:
        self._fan("on_checkpoint_save_end", int(model_step))

    def on_checkpoint_save_success(self, model_step: int, *, location: str | None = None) -> None:
        if location is None:
            self._fan("on_checkpoint_save_success", int(model_step))
        else:
            self._fan("on_checkpoint_save_success", int(model_step), location=location)

    def on_train_end(self) -> None:
        """Gate: the train end is delivered behind every event emitted before it, stamped at this call; the caller
        waits (bounded) for it, so the intervals observers close here are closed on return whatever the queue held,
        unless an observer is parked past the bound (then when it returns, still in order)."""
        self._gate("on_train_end")

    def on_app_end(self) -> None:
        """Gate: the app end is delivered behind every earlier event (the backends' teardown runs after them, never
        concurrently with one), the caller waits (bounded), then the worker is stopped (bounded); the drop total
        is logged."""
        self._gate("on_app_end")
        self.shutdown(self.DRAIN_TIMEOUT_S)


def build_controller_observers(
    config: Any,
    rl: Any,
    *,
    controller: str,
    world_size: int,
    topology: str,
    policy_dp_world: int = 1,
    app_start_time_ms: int | None = None,
) -> ControllerLifecycleObservers:
    """The observers one controller run reports through: OneLogger when the run asks for it, else none.

    ``world_size`` is the GPU count of the RESOLVED placement (every device group the entry bound,
    teacher GPUs included), not something inferred from role shapes; ``topology`` is the entry's
    label for that placement. Adding a telemetry backend means adding a line here, not touching the
    loop.

    GRPO/OPD construction is best effort: failed optional backends are logged and left out.
    Joint PPO retains fail-fast configuration/credential admission and startup before model loading;
    its runtime events and terminal cleanup still use the shared bounded dispatcher.
    """
    observers = ControllerLifecycleObservers()
    if controller == "joint_ppo":
        from cosmos_framework.posttrain.callbacks.joint_ppo_one_logger import JointPPOOneLogger

        one_logger = JointPPOOneLogger.create(config, rl, world_size)
        if one_logger is not None:
            observers.add(one_logger)
        return observers
    try:
        from cosmos_framework.posttrain.callbacks.one_logger import build_one_logger_observer

        one_logger = build_one_logger_observer(
            config,
            rl,
            controller=controller,
            world_size=world_size,
            topology=topology,
            policy_dp_world=policy_dp_world,
            app_start_time_ms=app_start_time_ms,
        )
    except Exception as e:  # telemetry is best effort: a backend that fails to build never aborts the run
        logger.warning(
            f"lifecycle observers: OneLogger not built, the run reports to no observer ({type(e).__name__}: {e})"
        )
        return observers
    if one_logger is not None:
        observers.add(one_logger)
    return observers
