# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Bounded asynchronous event dispatch for the control plane's observer bus (DESIGN §2.6, §11.2).

`BoundedEventDispatcher` moves `LifecycleEvent`s from the emitting thread to ONE lazily created daemon
worker thread through a bounded FIFO. The emitter never blocks and never waits on a listener (§2.6:
observers are async and off the hot path; §11.2: "All observers, off-hot-path, must not block").
Delivery order is emission order, so a listener sees the sequence a synchronous fan-out would have
given it, just later.

Every event is stamped at the controller boundary: :meth:`BoundedEventDispatcher.submit` records the
emitting thread's wall clock and monotonic clock in the queued `LifecycleEvent`, so a listener that
measures intervals (OneLogger's timers) reads WHEN THE CONTROLLER EMITTED, not when the worker got
round to delivering. Without the stamps a slow listener ahead in the queue collapses a 200 ms
workload into the microseconds between two deliveries.

Overflow is explicit, not a stall, and it preserves start/end pairs: an event the full queue cannot
take is dropped and counted, and the drop is logged on its first occurrence and at every power of ten
after that. A start event (``pairs`` maps it to the followers that close it) is admitted only when
the queue has room for it AND for its followers, and once admitted its followers are always taken
(the room was reserved); a start that had to be dropped drops its followers with it, so a listener
never sees an end without its start or a start without its end, and the groups dropped together are
counted (:attr:`BoundedEventDispatcher.dropped_pairs`) and logged as groups.

Terminal boundary events (the bus's gates, `control/lifecycle.py`) go through :meth:`deliver_gate`:
a non-droppable marker placed behind everything queued and delivered by the same worker, in FIFO
order, while the caller waits a bounded time for it. The marker takes no capacity and is never
dropped, and because only the worker delivers it, a gate can never run concurrently with a hook the
worker is still inside, nor overtake the events queued ahead of it. When the wait runs out (an
observer parked inside a hook), the caller returns with a warning and the gate stays queued: it is
delivered, in order, when the worker gets to it, and never forced onto the caller's thread. A gate
raised from inside a hook, on the worker itself, is delivered inline exactly once and never queued.

`flush()` is the drain barrier (a marker whose arrival the caller awaits); `shutdown()` flushes, stops
the worker with a bounded join and reports the drop total; an event submitted after it is delivered
inline once the worker has exited and dropped (counted) while the worker still runs, so the FIFO is
never overtaken. The worker is a daemon so this dispatcher thread does not hold the process open at
exit, unlike the checkpoint writer in `checkpoint/worker.py`, whose non-daemon thread guards data.
This does not terminate or reap resources a backend owns, such as an SDK subprocess.
The dispatcher runs on the single controller process and converges nothing across ranks.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from cosmos_framework.posttrain.utils.log import logger

# Events the queue holds before it drops: about two loop steps of a b32 n_mini=4 n_micro=8 run at one
# event per lifecycle call, i.e. an observer that is seconds behind, not one that is merely slow.
DEFAULT_QUEUE_MAXSIZE = 1024

_STOP = object()

# Admission verdicts (identity-compared).
_ADMIT = object()
_DROP = object()
_DROP_START = object()  # a start with no room for its group: dropped together with its followers
_DROP_WITH_START = object()  # the follower of a dropped start: dropped as part of that group


@dataclass(frozen=True)
class LifecycleEvent:
    """One observer event with the clocks of the thread that emitted it.

    ``emitted_wall`` is ``time.time()`` (seconds since the epoch, what OneLogger's timestamps are made
    of) and ``emitted_monotonic`` is ``time.monotonic()`` (what durations are made of), both read on the
    emitting thread before the event is queued. A listener delivered late still sees the controller's
    timeline.
    """

    name: str
    args: tuple[Any, ...] = ()
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    emitted_wall: float = 0.0
    emitted_monotonic: float = 0.0

    @classmethod
    def now(cls, name: str, args: tuple[Any, ...] = (), kwargs: Mapping[str, Any] | None = None) -> LifecycleEvent:
        """The event ``name`` stamped with the calling thread's clocks, i.e. at the emission boundary."""
        return cls(name, tuple(args), dict(kwargs or {}), time.time(), time.monotonic())


Deliver = Callable[[LifecycleEvent], None]


@dataclass
class _PairGroup:
    """A start event the dispatcher has seen and the followers it still owes (admitted) or still drops."""

    accepted: bool
    remaining: list[str]


@dataclass(frozen=True)
class _Gate:
    """A terminal event queued behind everything submitted before it; ``done`` is set once the worker delivered it."""

    event: LifecycleEvent
    done: threading.Event = field(default_factory=threading.Event)


class BoundedEventDispatcher:
    """One daemon worker draining a bounded FIFO of `LifecycleEvent`s into ``deliver(event)``.

    ``deliver`` runs on the worker thread and is expected to guard its own listeners; whatever it
    still raises is logged and the worker moves on. The worker starts on the first submit or terminal gate;
    :meth:`shutdown` stops it. After shutdown, :meth:`submit` delivers inline on the caller's thread
    only once the worker has exited (a stray event after teardown is still seen and nothing starts a
    thread at exit); while the worker still runs, i.e. a bounded shutdown timed out with it parked
    inside a listener, the event is dropped and counted instead (one warning), because delivering it
    inline would overtake the events still queued and run two threads into the same listeners.

    Capacity and pairs: ``maxsize`` bounds the events queued plus the slots reserved for the followers
    of admitted start events (``pairs``: start name -> the follower names that close it, in order).
    A start is admitted only with room for itself and its followers, so its followers are never
    dropped; a start that does not fit is dropped together with its followers (counted once in
    :attr:`dropped_pairs`, every event in :attr:`dropped`). A follower whose start the dispatcher
    never saw is an ordinary event. A start arriving while its previous group is still open (a save
    that ended without a success) closes that group and releases its reservation.

    Gates: :meth:`deliver_gate` queues a terminal event as a marker behind everything submitted so far
    (no capacity taken, never dropped) and waits, bounded, for the worker to deliver it in FIFO order.
    The worker is the only thread that ever runs a listener while it lives, so a gate cannot enter a
    listener the worker is still inside and cannot overtake the queued events; a wait that runs out
    leaves the gate queued (delivered when the worker gets to it) and returns False with a warning
    naming the hook the worker is parked in (:attr:`in_flight`) and the events queued ahead. A gate
    raised on the worker thread itself (a listener calling a gate from inside its own hook) is
    delivered inline, once, and never queued: the reentrant check comes before the marker is put on
    the queue, so the terminal event is not delivered a second time when the hook returns. With no
    worker yet the gate starts one, including cleanup after failed setup; after shutdown it joins the
    worker within the bound and is delivered inline once the worker is gone, or dropped and counted
    while the worker still runs (inline delivery would run two threads into one listener).

    Thread safety: the admission decision and the enqueue in :meth:`submit` are one critical section
    with the flag set in :meth:`shutdown`, so a submit racing a shutdown from another thread is either
    queued ahead of the stop request (and delivered) or counted as dropped, never acknowledged and
    lost. :meth:`flush`, :meth:`deliver_gate` and :meth:`shutdown` are the controller's boundaries
    (`on_train_end`, `on_app_end`) and are called from one thread in practice; a flush racing a
    shutdown from another thread loses nothing but can wait out its timeout and return False (its
    barrier landed behind the stop request).
    """

    def __init__(
        self,
        deliver: Deliver,
        *,
        maxsize: int = DEFAULT_QUEUE_MAXSIZE,
        name: str = "lifecycle-observers",
        pairs: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        if maxsize < 1:
            raise ValueError(f"{name}: queue maxsize must be >= 1, got {maxsize}")
        self._deliver = deliver
        self._name = name
        self._maxsize = maxsize
        self._pairs: dict[str, tuple[str, ...]] = {start: tuple(ends) for start, ends in (pairs or {}).items()}
        self._follower_of: dict[str, str] = {}
        for start, ends in self._pairs.items():
            if not ends:
                raise ValueError(f"{name}: pair {start!r} names no follower")
            for end in ends:
                if end in self._pairs or end in self._follower_of:
                    raise ValueError(f"{name}: {end!r} is named by more than one pair")
                self._follower_of[end] = start
        # Unbounded underneath: the bound is kept by the counters below, so a reserved follower always
        # finds its slot and the flush barrier / stop request never wait for room.
        self._queue: queue.Queue[Any] = queue.Queue()
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None
        self._stopped = False
        self._stop_sent = False
        self._queued = 0  # events in the queue (barriers and the stop request do not count)
        self._reserved = 0  # slots held for the followers of admitted starts
        self._groups: dict[str, _PairGroup] = {}
        self._dropped = 0
        self._dropped_pairs = 0
        self._late_drop_logged = False
        self._in_flight: str | None = None  # the event the worker is delivering right now

    @property
    def maxsize(self) -> int:
        return self._maxsize

    @property
    def queued(self) -> int:
        """Events waiting in the queue (an event being delivered has already left it)."""
        return self._queued

    @property
    def dropped(self) -> int:
        """Events dropped: no room in the queue (a dropped start counts its followers too as they arrive), or
        they arrived after shutdown while the worker still ran."""
        return self._dropped

    @property
    def dropped_pairs(self) -> int:
        """Start events dropped together with their followers because the queue had no room for the group."""
        return self._dropped_pairs

    @property
    def stopped(self) -> bool:
        """True once :meth:`shutdown` ran; later submits deliver inline once the worker has exited, else drop."""
        return self._stopped

    @property
    def worker_alive(self) -> bool:
        worker = self._worker
        return worker is not None and worker.is_alive()

    @property
    def in_flight(self) -> str | None:
        """The name of the event the worker is delivering at this moment, None between deliveries."""
        return self._in_flight

    def submit(self, event: str, args: tuple[Any, ...] = (), kwargs: dict[str, Any] | None = None) -> bool:
        """Stamp ``event`` with this thread's clocks and enqueue it without blocking.

        False when the event was dropped and counted: the queue had no room for it (for a start: for it
        and its followers; for the follower of a dropped start: its start is already gone), or shutdown has
        begun and the worker still runs (the first such drop is logged).
        """
        item = LifecycleEvent.now(event, args, kwargs)  # the controller boundary: stamped before any wait
        verdict = _DROP
        with self._lock:  # one critical section with shutdown() setting the flag: nothing lands behind its stop request
            stopped = self._stopped
            if not stopped:
                verdict = self._admit_locked(event)
                if verdict is _ADMIT:
                    self._start_worker_locked()
                    self._queue.put_nowait(item)  # unbounded underneath: never blocks
                    self._queued += 1
                    return True
            worker_alive = self._worker is not None and self._worker.is_alive()
        if not stopped:
            self._record_drop(event, verdict)
            return False
        if worker_alive:  # a bounded shutdown timed out: inline delivery would overtake the queued events
            self._record_late_drop(event)
            return False
        self._deliver_guarded(item)  # stopped and the worker is gone: a stray event is still seen
        return True

    def flush(self, timeout: float | None = 30.0) -> bool:
        """Block until everything submitted so far is delivered; False when ``timeout`` elapsed first.

        The barrier is a marker behind the queued events; waiting for it ends when the worker has
        finished every event ahead of it.
        """
        with self._lock:
            worker, stopped = self._worker, self._stopped
        if worker is None or not worker.is_alive():
            return True
        if worker is threading.current_thread():  # an observer flushing from inside its own hook
            return False
        if stopped:  # the stop request already sits behind every event: the worker's exit is the barrier
            worker.join(timeout)
            return not worker.is_alive()
        marker = threading.Event()
        self._queue.put(marker)
        if marker.wait(timeout):
            return True
        logger.warning(f"{self._name}: flush timed out after {timeout} s with {self._queued} events queued")
        return False

    def deliver_gate(self, item: LifecycleEvent, timeout: float | None = 30.0) -> bool:
        """Deliver the terminal ``item`` behind everything submitted so far and wait for it, bounded by ``timeout``.

        True when it was delivered: by the worker, started here if needed, in FIFO order; inline on
        this thread after the worker exited following shutdown; or inline, exactly once
        and without being queued, when this thread IS the worker (a listener raising a gate from inside its
        own hook: serial by construction, and a queued copy would be delivered again after the hook returns).
        False when it was not delivered within ``timeout``: the worker is parked inside a hook, so the gate
        stays queued and is delivered, in order, when the worker gets to it (a warning names the hook and the
        events ahead); or shutdown has begun and the worker still runs, in which case it is dropped and
        counted (a gate cannot be queued behind the stop request, and inline delivery would run two threads
        into one listener). A gate is never counted against the capacity, never delivered concurrently with a
        listener still running, and never delivered twice.
        """
        with self._lock:
            # Terminal cleanup must also be bounded when setup failed before any
            # asynchronous event. App initialization is delivered by the bus itself.
            if self._worker is None and not self._stopped:
                self._start_worker_locked()
            worker, stopped = self._worker, self._stopped
            alive = worker is not None and worker.is_alive()
            # Decided BEFORE anything is queued: a listener raising a gate from inside its own hook IS the
            # worker, so the gate is delivered inline (serial by construction) and never also queued, which
            # would deliver it a second time once the hook returns and the worker reaches the marker.
            reentrant = worker is threading.current_thread()
            gate = None
            if alive and not stopped and not reentrant:
                gate = _Gate(item)
                self._queue.put(gate)  # unbounded underneath: never blocks, takes no capacity
        if reentrant or not alive:  # this thread is the worker, or it exited after shutdown
            self._deliver_guarded(item)
            return True
        if gate is None:  # shutdown has begun: the stop request is (or is about to be) behind the queue
            worker.join(timeout)
            if worker.is_alive():
                with self._lock:
                    self._dropped += 1
                    in_flight = self._in_flight
                logger.warning(
                    f"{self._name}: dropped {item.name}: it arrived after shutdown while the worker is still "
                    f"running (inside {in_flight or 'a hook'}); delivering it inline would run two threads into "
                    "the same listener"
                )
                return False
            self._deliver_guarded(item)
            return True
        if gate.done.wait(timeout):
            return True
        with self._lock:
            in_flight, queued = self._in_flight, self._queued
        logger.warning(
            f"{self._name}: {item.name} could not be delivered within {timeout} s: the worker is parked inside "
            f"{in_flight or 'a hook'} with {queued} events queued ahead of it; the gate stays queued and is "
            "delivered, in order, when the worker gets to it (it is never run concurrently with a hook)"
        )
        return False

    def shutdown(self, timeout: float | None = 30.0) -> bool:
        """Drain, stop the worker and join it, bounded by ``timeout``; True when the worker is gone.

        Idempotent: a repeated call sends no second stop request and reports the drop total only
        once, but does join a worker that outlived an earlier bounded call.
        """
        with self._lock:
            first = not self._stopped
            self._stopped = True
            worker = self._worker
        stopped = True
        if worker is threading.current_thread():  # an observer shutting the bus down from inside its own hook
            logger.warning(f"{self._name}: shutdown called from the worker thread; it cannot join itself")
            return False
        if worker is not None and worker.is_alive():
            if not self._stop_sent:
                self._queue.put(_STOP)
                self._stop_sent = True
            worker.join(timeout)
            stopped = not worker.is_alive()
            if not stopped:
                logger.warning(
                    f"{self._name}: worker still running after {timeout} s; "
                    "it is a daemon and does not hold the process open"
                )
        if first and self._dropped:
            logger.warning(
                f"{self._name}: {self._dropped} events were dropped over the run "
                f"({self._dropped_pairs} start/end groups dropped together; queue of {self._maxsize}; "
                "see the drop warnings above for the reasons)"
            )
        return stopped

    def _admit_locked(self, event: str) -> object:
        """The capacity verdict for ``event`` under the lock: admit, drop, or drop as part of a dropped group."""
        start = self._follower_of.get(event)
        if start is not None:
            group = self._groups.get(start)
            if group is not None and event in group.remaining:
                group.remaining.remove(event)
                if not group.remaining:
                    del self._groups[start]
                if group.accepted:
                    self._reserved -= 1  # the slot was reserved when its start was admitted
                    return _ADMIT
                return _DROP_WITH_START
            # a follower whose start this dispatcher never saw: an ordinary event
        followers = self._pairs.get(event)
        if followers is None:
            return _ADMIT if self._queued + self._reserved < self._maxsize else _DROP
        previous = self._groups.pop(event, None)
        if previous is not None and previous.accepted:  # the earlier group never closed: give its room back
            self._reserved -= len(previous.remaining)
        if self._queued + self._reserved + len(followers) < self._maxsize:  # room for it and its followers
            self._groups[event] = _PairGroup(accepted=True, remaining=list(followers))
            self._reserved += len(followers)
            return _ADMIT
        self._groups[event] = _PairGroup(accepted=False, remaining=list(followers))
        return _DROP_START

    def _start_worker_locked(self) -> None:
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(target=self._run, name=self._name, daemon=True)
            self._worker.start()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is _STOP:
                return
            if isinstance(item, threading.Event):  # a flush barrier
                item.set()
                continue
            if isinstance(item, _Gate):  # a terminal event: delivered here, in its FIFO place, then acknowledged
                self._deliver_tracked(item.event)
                item.done.set()
                continue
            with self._lock:
                self._queued -= 1
            self._deliver_tracked(item)

    def _deliver_tracked(self, item: LifecycleEvent) -> None:
        """The worker's delivery: records the event as in flight for the gate warnings."""
        with self._lock:
            self._in_flight = item.name
        try:
            self._deliver_guarded(item)
        finally:
            with self._lock:
                self._in_flight = None

    def _deliver_guarded(self, item: LifecycleEvent) -> None:
        try:
            self._deliver(item)
        except Exception as e:  # deliver guards its listeners; this keeps the worker alive whatever slips through
            logger.warning(f"{self._name}: delivering {item.name} raised (ignored): {e}")

    def _record_late_drop(self, event: str) -> None:
        with self._lock:
            self._dropped += 1
            announce = not self._late_drop_logged
            self._late_drop_logged = True
        if announce:
            logger.warning(
                f"{self._name}: dropped {event}: it arrived after shutdown while the worker is still running; "
                "delivering it inline would overtake the events still queued, so it and any later event are "
                "dropped and counted until the worker exits"
            )

    def _record_drop(self, event: str, verdict: object) -> None:
        with self._lock:
            self._dropped += 1
            if verdict is _DROP_START:
                self._dropped_pairs += 1
            count = self._dropped
        if verdict is _DROP_WITH_START:  # counted; the warning for its start already named it
            return
        if str(count).rstrip("0") == "1":  # 1, 10, 100, ...: the first drop and every power of ten after it
            what = event
            if verdict is _DROP_START:
                what = f"{event} together with its {', '.join(self._pairs[event])} (no room for the group)"
            logger.warning(
                f"{self._name}: dropped {what} (queue of {self._maxsize} full; {count} dropped so far); "
                "the observers are behind the controller, which does not wait for them"
            )
