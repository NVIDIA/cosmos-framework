# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""The sim<->gateway transport: a shared buffer for arrays, a unix socket for everything else.

This is the wire the container split runs on. The simulator lives in its own process (its own
venv, its own Python, speaking no Ray); the gateway is the Ray actor that holds the sim role and
forwards ``describe``/``act`` to the policy. Between them: **arrays go through POSIX shared
memory, and control goes through an ``AF_UNIX`` socket as length-prefixed JSON.**

**Why shmem and not just the socket, measured rather than argued.** Both were benchmarked with
the real 3,917,128 B observation on a cluster node (``scripts/obs_transport_bench.py``, and
SPEC §9.7 for the table). At the median they are close -- 899 µs for a socket against 628 µs for
the buffer. Under a saturated cgroup they are not: **29,413 µs against 2,884 µs, a factor of
~10**, and against the layout this module actually uses (sim copies in, gateway reads in place)
it is ~48. The reason is that the dominant cost is a scheduler **wakeup**, and streaming 3.9 MB
through kernel socket buffers takes many where a doorbell takes one. A design chosen on the
median would have shipped the tail.

**What is deliberately NOT here.**

*No pickle.* Measured at 6,015 µs against 911 µs for the same dict -- ``pickle.dumps`` alone is
2,044 µs because it builds a 3.9 MB intermediate. Writing an array into the buffer is a
``np.copyto``. (``embodied/sim.py``'s ``assert_wire_safe`` already constrains the payload to
primitives, which is what makes a codec this small sufficient.)

*No ring buffer, no double buffering, no locks.* The flow is **synchronous**: the sim blocks on
``act`` until the gateway answers, so exactly one message is in flight and the buffer has exactly
one writer at a time. Anything more would be machinery for a concurrency that the protocol
forbids.

*No zero-copy binding.* The sim copies its arrays INTO the segment rather than rendering directly
into it: binding would save 359 µs -> 72 µs, i.e. **0.0055% of episode wall time**.

**The one hazard this module cannot remove.** :func:`read_arrays` returns numpy **views onto the
segment** -- that is the whole point, it is what makes the gateway side free -- and those views
are invalidated by the next :func:`write_arrays` into the same region. Retaining one across a
round trip yields plausible, wrong data rather than an error. On the production path this is
safe for a specific reason worth stating rather than assuming: the gateway hands the views
straight to ``PeerClient.act``, whose ``ray.get(host.invoke.remote(...))`` serialises its
arguments **eagerly**, so the copy is taken inside the call. Any other consumer must copy. Set
``B1K_WIRE_POISON=1`` to overwrite each region after it is read, which converts a stale-view bug
from silently plausible into obviously garbage; it costs about one memcpy and is off by default.
"""

from __future__ import annotations

import json
import mmap
import os
import socket
import struct
import threading
import time
import uuid
import zlib
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

#: Bumped when a frame kind, a field, or the segment layout changes incompatibly. Checked in the
#: ``hello`` handshake; a mismatch is refused there rather than discovered as a decode error.
#:
#: 2 added the ``fault`` frame (gateway -> sim). Both halves always ship from one tree -- the
#: gateway spawns the child with this repo on its ``PYTHONPATH`` -- so the check cannot fire on
#: a vocabulary change. Bump anyway: the version means "the frame vocabulary changed".
#:
#: 5 added ``session_id`` and ``generation`` to ``hello``. See :func:`session_identity`.
#:
#: 6 added ``visible_devices`` to ``hello`` -- the child's OWN ``CUDA_VISIBLE_DEVICES``, read
#: before it builds anything. Required rather than optional, because the value it replaces is
#: the gateway's INTENT, and a field that is sometimes absent would silently fall back to that
#: intent on exactly the peers whose honesty is in question.
#:
#: 7 added ``pgid`` to ``hello`` -- the simulator's OWN process group. It is what lets a teardown
#: reach the simulator's tree without assuming that the process the gateway spawned IS the
#: simulator, which stops being true the moment a supervisor sits between them. See
#: :func:`group_alive`.
#: 9 adds decision contexts on ``act`` frames and ``feedback`` callbacks for chunk collection windows.
#: 10 preserves per-control ending facts separately from physical episode completion.
WIRE_VERSION = 10

#: Frame header: one unsigned big-endian length. The payload is always UTF-8 JSON -- arrays never
#: travel here, only their descriptors, so a frame is a few hundred bytes and the encoding choice
#: costs nothing measurable.
_HEADER = struct.Struct("!I")

#: Refuse a control frame larger than this. A JSON frame is descriptors and scalars; anything at
#: this size means an array leaked into the control path, which is exactly the regression that
#: would quietly undo the measurement above.
MAX_FRAME_BYTES = 4 << 20

#: Floor for the observation region, applied when the declared capabilities under-describe the
#: payload -- which they do by design whenever a camera's resolution was not explicitly
#: requested. Derived from the measured observation (3,917,128 B) plus the same slack, so the
#: fallback covers the real embodiment rather than a guess. Costs ~6 MB of /dev/shm per gateway.
#:
#: The RGBA number: ``the external simulator backend::detach_frames`` copies the frame out WITHOUT
#: narrowing, so the fourth plane crosses. The floor, not the arithmetic, decides the b1k
#: segment: 3 channels at 720/480 is 4,406,766 B with slack, under this 5,875,692.
MIN_OBS_BYTES = int(3_917_128 * 1.5)

#: Extra room over the size implied by the declared capabilities. Covers keys the embodiment did
#: not declare and any rounding; an observation that still does not fit RAISES with both numbers
#: rather than truncating.
SEGMENT_SLACK = 1.5

#: Poison a region after reading it, to turn a retained-view bug loud. Off by default: it costs
#: roughly one memcpy of the region.
_POISON = os.environ.get("B1K_WIRE_POISON") == "1"

#: Verify a digest over every payload that crosses the segment. **ON by default**, opt OUT with
#: ``B1K_WIRE_VERIFY=0``.
#:
#: Default-on because an opt-in integrity check is the same thing with extra steps: nobody
#: turns it on until after the corrupt run.
#:
#: The cost is bounded by how rarely the payload crosses, not by its size: the observation
#: crosses ONCE PER 32-STEP CHUNK, not per step. See :meth:`Segment.digest`.
_VERIFY = os.environ.get("B1K_WIRE_VERIFY", "1") != "0"

#: Where POSIX shared memory is mounted. Overridable because a container can mount it elsewhere,
#: and because a test may want a plain tmpfs.
SHM_DIR = os.environ.get("B1K_SHM_DIR", "/dev/shm")

#: Separator between a lane index and an observation key inside one ``act``/``feedback`` frame. A
#: unit separator rather than ``/`` or ``:`` because real keys contain both -- ``robot_r1::robot_r1:
#: zed_link:Camera:0::rgb`` would split on either and silently address the wrong lane. Both halves
#: of the transport read it from here.
LANE_SEP = "\x1f"

#: Prefix for every segment this module creates, so a crash-time sweep can find them. Segments
#: are named rather than anonymous specifically to make that sweep possible: an orphan holds real
#: memory until the pod dies, and on a long campaign that is discovered as an unrelated
#: allocation failure hours later.
SEGMENT_PREFIX = "b1k-wire-"


class WireError(RuntimeError):
    """The peer violated the protocol, or the transport failed.

    Deliberately NOT a subclass of ``SimProtocolError``: this module is imported by both halves
    of the seam, including the sim-side process that must not import the framework. The gateway
    translates it at the boundary, where ``classify_error`` can see it.
    """


class PeerGone(WireError):
    """The peer's socket reached a CLEAN EOF: it closed, or it died. Distinct from "it failed".

    The distinction exists so a caller can decide whether reconnecting is even meaningful. A
    peer that sends an ``error`` frame is ALIVE and has reported something about the work; a
    peer at EOF is not there any more, and the only way to continue is to talk to a different
    one. Retrying the first case would kill a healthy simulator to escape a unit-level fault.

    Raised ONLY where the evidence is unambiguous -- ``recv_frame`` returning ``None``. It is
    deliberately NOT raised on a failed SEND, because ``sim_gateway._child_failure`` is reached
    there even when the child is perfectly healthy: the pre-send size cap refuses the frame
    locally, the child never saw anything, and it is still blocked in its own ``recv``. That
    path already has a test, and calling it "gone" would turn an oversized frame into a
    simulator restart.

    A subclass of :class:`WireError` so every existing handler keeps working.
    """


class WireTimeout(WireError):
    """The peer went SILENT past a deadline. Distinct from "the transport broke".

    Needed because :func:`_recv_exact` converts every ``OSError`` into a :class:`WireError`, and
    ``socket.timeout`` is an ``OSError`` -- so a caller that set a deadline and tried to catch
    ``socket.timeout`` around a receive catches nothing, and its own error message never runs.

    A subclass of :class:`WireError` so every existing handler keeps working -- a silent peer IS
    a failed exchange; this only lets a caller that cares say something better about it.
    """


#: The supervisor's two handles on a simulator's identity, read by the CHILD and reported in
#: ``hello``. Both are absent today, which is the correct reading of "no supervisor yet".
SESSION_ID_ENV = "B1K_SIM_SESSION_ID"
GENERATION_ENV = "B1K_SIM_GENERATION"


def session_identity() -> tuple[str, int]:
    """``(session_id, generation)`` for this simulator process.

    **Why the CHILD owns this and not the gateway.** Under the sim-primary target the simulator
    container is the LONG-LIVED host while the Ray side comes and goes; identity has to
    originate with the party that outlives the other, or its semantics change when that flips.

    ``session_id`` is stable across simulator RESTARTS within one container: a supervisor sets
    it once and every child it spawns reports the same one. ``generation`` is what distinguishes
    those children -- it increments per restart, which is exactly ``sim_decouple.md`` §10's
    *"every backend restart increments generation"*.

    With no supervisor -- today -- the id is minted per process and the generation is 0. That is
    not a placeholder: one process IS the whole session when nothing restarts it.

    **This generation is the TRANSPORT's, and it must never gate unit ownership.** The claim
    plane (``campaign/dirlease.py``) has its own generation, and the two answer different
    questions: this one asks *"am I talking to the same simulator?"*, the claim plane asks
    *"who owns this work unit?"*. Two counters that both arbitrate ownership is precisely the
    coupling the sim-primary argument objects to in the framework-primary design, and it has
    already produced one false double-claim report in this subsystem. Keep them apart.
    """
    session_id = os.environ.get(SESSION_ID_ENV, "").strip()
    if not session_id:
        session_id = f"sim-{os.getpid()}-{uuid.uuid4().hex[:12]}"

    raw = os.environ.get(GENERATION_ENV, "").strip()
    if not raw:
        return session_id, 0
    try:
        generation = int(raw)
    except ValueError as exc:
        # REFUSED, not defaulted to 0. A supervisor that sets this to something unparseable has
        # lost its counter, and a silent 0 would make every restart look like a first start --
        # which is the one reading that disables the fencing this field exists for.
        raise WireError(
            f"{GENERATION_ENV}={raw!r} is not an integer. It is set by the supervisor and a "
            f"restart counter that cannot be read is not a counter."
        ) from exc
    if generation < 0:
        raise WireError(f"{GENERATION_ENV}={generation} is negative; generations only increase.")
    return session_id, generation


def refuse_identity_replay(
    previous: tuple[str, int] | None,
    current: tuple[str, int],
    *,
    where: str,
    expect_restart: bool = True,
) -> None:
    """An identity must never REPEAT. Within one session, the generation must strictly advance.

    **The first draft of this was stronger and wrong**, and the existing subprocess suite caught
    it within one run: it refused any change of ``session_id``, on the theory that a different
    session means "a different simulator, not a restart". That is right for a RECONNECT and wrong
    for the case that actually happens -- a gateway spawning a fresh child. With no supervisor
    the id is minted per process, so a deliberate respawn legitimately reports a new session, and
    the strong rule turned an ordinary restart into a hard failure.

    So the invariant is the one that holds in every case rather than in the one that was imagined:

    ==============================  ==================  ==========
    case                            identity            verdict
    ==============================  ==================  ==========
    fresh spawn, no supervisor      new id, gen 0       allowed
    supervisor restart              same id, gen + 1    allowed
    whole container replaced        new id, gen 0       allowed -- it IS a new session
    supervisor's counter stuck      same id, same gen   **REFUSED**
    ==============================  ==================  ==========

    The refused row is the one worth having. Per ``sim_decouple.md`` §10 a restart invalidates
    simulator state and the episode replays; a restart that fails to advance its generation is
    indistinguishable from no restart at all, so anything fencing on the pair would wave through
    exactly the transition it exists to catch.

    Note what this deliberately does NOT do: it does not refuse a new session, so it is not a
    guard against resuming in-flight work across a restart. There is no work-resumption path to
    guard yet, and writing one now would be a rule with no caller.

    **``expect_restart`` -- the table above has FOUR rows and sim-primary needs a fifth.** Every
    row assumes the handshake follows a spawn, because that is the only way a handshake could
    happen when this was written: the gateway starts the child, so a second ``hello`` means a
    second child. Under sim-primary the simulator container is the long-lived host and the Ray
    side attaches to a process that is ALREADY RUNNING. A reattach to that same live simulator
    truthfully reports the same id and the same generation -- nothing restarted -- and the
    four-row rule REFUSES it, on a message about a stuck restart counter when no restart was
    ever claimed.

    So the caller says which question it is asking:

    * ``expect_restart=True`` (default, and what a gateway that spawns its own child passes):
      *"I watched my child die and accepted a replacement."* A generation that did not advance
      is the fault, exactly as the four rows say.
    * ``expect_restart=False``: *"I lost a connection to a process I do not own, and took the
      next one."* An unchanged generation is then the CONFIRMATION -- the same simulator is
      still there and its state is intact -- rather than evidence of a stuck counter.

    **Attach mode does NOT also refuse a CHANGED session id**, tempting as that is. The gateway
    holds a listener that a RESTARTED SIMULATOR CONTAINER reconnects to, and a supervisor mints
    its session id once per container, so a replaced container legitimately arrives with a new
    id -- the row the table already calls *"allowed -- it IS a new session"*. The stranger such
    a rule would catch is close to unreachable anyway: ``sim_gateway.py::_socket_path`` is
    per-PID, so two gateways in one IPC directory do not share a path.
    """
    if previous is None:
        return
    old_id, old_gen = previous
    new_id, new_gen = current
    if new_id != old_id:
        return
    if new_gen < old_gen or (new_gen == old_gen and expect_restart):
        raise WireError(
            f"{where}: simulator session {new_id!r} reports generation {new_gen} again, which "
            f"does not advance past {old_gen}. Either the supervisor's restart counter is stuck "
            f"or this is the same process being handshaken twice -- and a restart that cannot be "
            f"told from no restart is the case generation fencing exists for."
        )


class PolicyFault(Exception):
    """The FRAMEWORK side raised while answering a callback. Relayed, not absorbed.

    This exists to make ``gateway`` fault-equivalent to ``direct``. In ``direct`` the policy is
    called from inside ``driver.run_unit``'s per-unit ``try``, so a transient policy fault (a CUDA
    OOM on the 8B checkpoint, a Ray blip) costs ONE unit and the collector tolerates
    ``max_consecutive_errors`` of them. In ``gateway`` the driver runs in the child while
    ``policy.act`` is called in the gateway process -- *outside* that handler -- so the same fault
    unwound out of the Ray actor and killed the whole cell, which on today's topology is the whole
    pod. Sending it across as a frame puts it back inside the handler that already exists.

    **Not a subclass of any ``sim.SimError``, and that is the point.** ``classify_error`` maps
    ``SimProtocolError`` -> ``"protocol"``, which means "the code on the far side is wrong"; a
    policy OOM is not that. In ``direct`` this same fault arrives as a ``RayTaskError`` and buckets
    as ``"unclassified"``, so anything more specific here would make the two modes disagree in the
    error table for identical failures -- equivalence broken by the fix meant to create it.
    """


#: Gateway -> sim. Carries a framework-side exception into the child's per-unit handler.
FAULT = "fault"


def fault_payload(where: str, exc: BaseException) -> dict:
    """The ``fault`` frame body. ``repr`` rather than ``str`` to match ``UnitResult.error``.

    The traceback does NOT travel. It is printed on the gateway's stdout, which IS the pod log,
    and the frame is on the control path where :data:`MAX_FRAME_BYTES` is the budget.
    """
    return {"where": where, "error": repr(exc), "class": type(exc).__name__}


def policy_fault_from(payload: Mapping[str, Any], expected: str) -> PolicyFault:
    """Rebuild the peer's fault on this side, saying plainly that it is not ours."""
    return PolicyFault(
        f"the framework raised while answering {payload.get('where', '?')!r} "
        f"(expected {expected!r}): {payload.get('error')}"
    )


@dataclass(frozen=True)
class ArrayDescriptor:
    """Where one array sits in the segment, and how to interpret it.

    Rides in the JSON control frame. ``dtype`` is a numpy dtype string (``"uint8"``,
    ``"float32"``), ``shape`` a list of ints, ``offset`` a byte offset into the segment.
    """

    key: str
    dtype: str
    shape: tuple[int, ...]
    offset: int
    nbytes: int

    def as_wire(self) -> dict:
        return {
            "key": self.key,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "offset": self.offset,
            "nbytes": self.nbytes,
        }

    @staticmethod
    def from_wire(payload: Mapping[str, Any]) -> ArrayDescriptor:
        try:
            return ArrayDescriptor(
                key=str(payload["key"]),
                dtype=str(payload["dtype"]),
                shape=tuple(int(n) for n in payload["shape"]),
                offset=int(payload["offset"]),
                nbytes=int(payload["nbytes"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise WireError(f"malformed array descriptor {payload!r}: {exc}") from exc


def segment_bytes(camera_shapes: Mapping[str, tuple[int, int]], proprio_width: int, *, channels: int = 3) -> int:
    """Size the segment from the capabilities handshake, with a floor that is not optional.

        ``SimCapabilities`` carries ``camera_shapes`` and ``proprio_width``, which is what that
        handshake was for -- so in the normal case the segment needs no negotiation round and no
        configured constant that drifts from the embodiment.

        **But ``camera_shapes`` is legitimately EMPTY.** ``the external simulator backend::capabilities`` omits a
        camera whose resolution was never explicitly requested, deliberately: absence there means
        "unrequested", not "zero", and the alternative -- naming the wrapper's default -- is how
        "512/256" became a fact about this system without ever having been one. Sizing the segment
        off that directly gives a few hundred bytes, and the first observation overflows several
        minutes into a unit. So the declared size is a LOWER BOUND on what to trust, and
        :data:`MIN_OBS_BYTES` is the floor.

    **The default is 3 and an ``ObsBundle`` carries 4** -- that is deliberate, and it is the one
        thing to understand before changing it. The renderer emits RGBA (measured 2026-08-27,
        3,917,128 B) and ``the external simulator backend::detach_frames`` does not narrow it, so the fourth
        plane does reach a segment.

        A segment sized at 3 admits it anyway because :data:`SEGMENT_SLACK` is 1.5 and 4/3 is 1.33.
        **That is a 0.17 margin, not a design.** It held across the C8 reversal without anyone
        touching this line, which is the useful fact; it would stop holding at 5 channels, or if
        the slack were ever trimmed. Raising the default to 4 is safe and costs a quarter more
        /dev/shm per gateway -- pages that are virtual until touched. It is left at 3 because
        nothing has measured a reason to spend them, and an under-size RAISES with both numbers
        rather than truncating.
    """
    total = sum(int(h) * int(w) * int(channels) for h, w in camera_shapes.values())
    total += int(proprio_width) * 4
    return max(int(total * SEGMENT_SLACK), MIN_OBS_BYTES)


class Segment:
    """One POSIX shared-memory segment, split into a region per direction.

    Two regions, not one, and not a single bump allocator over the whole thing: the sim writes
    observations while the gateway writes actions, and a shared cursor would make each side's
    correctness depend on the other's message ordering. With a region each, a bug on one side
    cannot corrupt the other's data -- it can only overflow its own region, which raises.

    **Built on ``mmap`` and a file under ``/dev/shm`` rather than on
    ``multiprocessing.shared_memory``**, for two reasons that both bite in production. That class
    registers every segment with a ``resource_tracker`` -- a whole extra process, which warns
    about "leaked shared_memory objects" on any teardown it does not like and would run inside
    the simulator's container for no purpose. And importing it pulls ``__mp_main__`` into the
    module graph, which breaks ``embodied/``'s import rule: the core must load in the Isaac venv,
    the policy venv and the trainer venv alike, and the gate that enforces that caught this.
    """

    #: Actions are tiny -- [32, 23] float32 = 2,944 B measured. 64 KiB is room for a chunk width
    #: or action dimension an order of magnitude larger without a second thought.
    ACT_REGION = 64 << 10

    def __init__(self, path: str, buf: mmap.mmap, obs_capacity: int, *, owner: bool):
        self._path = path
        self._buf = buf
        self._obs_capacity = obs_capacity
        self._owner = owner
        self._closed = False
        self._unlinked = False
        #: Crossings this segment has integrity-checked. PUBLIC and deliberately not
        #: underscore-prefixed: it is read from outside (the gateway prints it per unit) and
        #: it exists to give "zero digest mismatches" a denominator. See :meth:`verify`.
        self.verified = 0

    @classmethod
    def create(cls, obs_capacity: int, *, name: str | None = None) -> Segment:
        """Create a segment. ``O_EXCL`` on purpose -- a name collision must not silently share.

        **The suffix is a uuid4, and it used to be ``id(object())``, which is not unique.**
        CPython hands the address of a just-freed object straight back, so a throwaway object
        allocated twice in one process very often lands at the same address and the "unique"
        suffix repeats verbatim. Measured, not reasoned: the gateway's reattach path created a
        second segment and got ``b1k-wire-788136-72ed91fc2390`` both times.

        It stayed invisible because of a second thing, not because it was harmless. ``O_EXCL``
        is exactly the guard that should have caught it -- but ``sim_gateway._handshake`` calls
        ``unlink_now()`` as soon as the child has mapped the segment, so the path is free again
        and the create succeeds against a NEW inode under an OLD name. The old mapping stays
        alive and valid, and from then on the name no longer identifies the segment: two live
        regions, one string, and ``sweep_orphans`` reasoning about the name.

        One segment per process was the only reason this never fired.
        """
        size = obs_capacity + cls.ACT_REGION
        path = os.path.join(SHM_DIR, name or f"{SEGMENT_PREFIX}{os.getpid()}-{uuid.uuid4().hex[:12]}")
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        try:
            os.ftruncate(fd, size)
            buf = mmap.mmap(fd, size)
        except BaseException:
            os.close(fd)
            os.unlink(path)
            raise
        os.close(fd)  # the mapping keeps the memory alive; the fd is not needed
        return cls(path, buf, obs_capacity, owner=True)

    @classmethod
    def attach(cls, name: str, obs_capacity: int) -> Segment:
        path = name if os.path.isabs(name) else os.path.join(SHM_DIR, name)
        size = obs_capacity + cls.ACT_REGION
        fd = os.open(path, os.O_RDWR)
        try:
            actual = os.fstat(fd).st_size
            if actual < size:
                raise WireError(
                    f"segment {path!r} is {actual} B but this side expects at least {size} B; "
                    f"the two halves disagree about the embodiment"
                )
            buf = mmap.mmap(fd, size)
        finally:
            os.close(fd)
        return cls(path, buf, obs_capacity, owner=False)

    @property
    def name(self) -> str:
        """The path, which is what the peer needs to attach."""
        return self._path

    @property
    def obs_capacity(self) -> int:
        return self._obs_capacity

    def _region(self, which: str) -> memoryview:
        if self._closed:
            raise WireError("segment is closed")
        view = memoryview(self._buf)
        if which == "obs":
            return view[: self._obs_capacity]
        if which == "act":
            return view[self._obs_capacity :]
        raise WireError(f"unknown region {which!r}")

    def write_arrays(self, which: str, arrays: Mapping[str, np.ndarray]) -> list[ArrayDescriptor]:
        """Copy ``arrays`` into ``which`` region and return their descriptors.

        The one copy this design pays. Keys are written in the mapping's own order, packed with
        no padding: every consumer addresses by the returned offset, so alignment is not load
        bearing and a gap would only shrink the usable region.
        """
        region = self._region(which)
        out: list[ArrayDescriptor] = []
        offset = 0
        for key, value in arrays.items():
            array = np.ascontiguousarray(value)
            if array.dtype == object:
                raise WireError(
                    f"{key!r} has object dtype, which is a box holding arbitrary Python rather "
                    f"than an array of bytes; it cannot cross this seam"
                )
            nbytes = int(array.nbytes)
            end = offset + nbytes
            if end > len(region):
                raise WireError(
                    f"{which} region overflow writing {key!r}: needs {end} B, region is "
                    f"{len(region)} B. Raise SEGMENT_SLACK or check the declared camera_shapes "
                    f"against what the backend actually renders."
                )
            region[offset:end] = memoryview(array).cast("B")
            out.append(
                ArrayDescriptor(
                    key=key,
                    dtype=str(array.dtype),
                    shape=tuple(int(n) for n in array.shape),
                    offset=offset,
                    nbytes=nbytes,
                )
            )
            offset = end
        region.release()
        return out

    def read_arrays(self, which: str, descriptors: Iterable[ArrayDescriptor]) -> dict[str, np.ndarray]:
        """Build numpy **views** onto ``which`` region. No copy -- see the module docstring.

        The views are read-only. That does not stop a caller retaining one past the next write
        (nothing can), but it does stop the other failure in the same family: a consumer mutating
        what it believes is its own array and silently editing the peer's buffer.
        """
        region = self._region(which)
        out: dict[str, np.ndarray] = {}
        for desc in descriptors:
            end = desc.offset + desc.nbytes
            if desc.offset < 0 or end > len(region):
                size = len(region)  # read BEFORE release: a released memoryview has no len()
                region.release()
                raise WireError(
                    f"descriptor for {desc.key!r} spans [{desc.offset}, {end}) which is outside "
                    f"the {size} B {which} region"
                )
            array = np.frombuffer(region, dtype=np.dtype(desc.dtype), count=_count(desc), offset=desc.offset)
            array = array.reshape(desc.shape)
            array.flags.writeable = False
            out[desc.key] = array
        return out

    def unlink_now(self) -> None:
        """Drop the NAME while both sides keep the mapping. Idempotent.

        The standard POSIX shared-memory idiom, and the reason it matters here is teardown that
        never runs. A gateway killed with the actor -- which is what happens on a normal driver
        exit, since nothing calls ``close()`` on a role body -- skips every ``finally``, and a
        named segment then holds 5.9 MB until something removes the name. Measured: two survived
        a clean CPU smoke.

        Once both peers have mapped it the name has no readers left: there are exactly two
        parties and the attach happens once, at startup. Unlinking there makes the kernel free
        the memory when the last mapping goes, whether that is a clean close, a SIGKILL, or the
        pod ending -- so the leak stops being possible rather than being swept up afterwards.

        :func:`sweep_orphans` stays, because a child that dies BEFORE it attaches leaves a name
        this never reached.
        """
        if self._unlinked or not self._owner:
            return
        self._unlinked = True
        try:
            os.unlink(self._path)
        except FileNotFoundError:
            pass

    def digest(self, which: str, nbytes: int) -> str:
        """CRC-32 over the first ``nbytes`` of ``which``, as hex.

        **A CHECKSUM, NOT A SECURITY PROPERTY, and the distinction is the reason it is cheap.**
        The threat is ACCIDENTAL corruption -- a view retained across the next write, a segment
        whose name prefix a sibling gateway reused -- between two processes this pod spawned
        itself. There is no adversary on a UNIX socket you are both ends of, so collision
        resistance buys nothing here and costs 7.5x.

        Measured on the real 3,917,128 B observation, this host: crc32 0.70 ms vs blake2b-128
        5.28 ms (5.6 GB/s vs 0.74). Over a 15,803-step episode that is 0.025% of wall instead of
        0.188% -- against a pre-registered isolation budget of 0.008-0.194%, so the strong hash
        would have doubled the thing it is measuring.

        The integrity check for the one failure this transport can produce SILENTLY. Everything
        else it can do -- a short read, a bad descriptor, a closed segment, an overflow -- raises
        on its own. A payload that arrives with the wrong BYTES does not: it is an observation
        that still looks like an observation, and it reaches the policy.

        Digested by REGION SPAN rather than per array, because the corruption modes that motivate
        it (a stale view from a peer that wrote again, a segment reused by a sibling gateway with
        the same name prefix) damage the buffer, not one key. One hash of the packed span sees
        all of them and costs one pass instead of N.

        `nbytes` comes from the descriptors the peer sent, so a peer that lies about the span
        cannot make the check pass by shortening it -- the descriptors are covered by the same
        frame the digest travels in, and a mismatch in either direction fails.
        """
        region = self._region(which)
        try:
            return format(zlib.crc32(region[:nbytes]) & 0xFFFFFFFF, "08x")
        finally:
            region.release()

    def verify(self, which: str, nbytes: int, expected: str) -> None:
        """Raise if ``which``'s first ``nbytes`` do not hash to ``expected``.

        **Poisons the region before raising**, which is what finally gives :meth:`poison` a
        caller. The ordering matters: a caller holding a view is exactly the situation where a
        retained view reads plausible-but-wrong data, so the region is made obviously garbage
        before the exception unwinds past anyone who might still read it.
        """
        if not _VERIFY:
            return
        # COUNTED: `_VERIFY` off, a `nbytes` of 0, and a path that never reaches this method all
        # produce the identical silence a clean run does; `self.verified` is the denominator.
        self.verified += 1
        actual = self.digest(which, nbytes)
        if actual != expected:
            self.poison(which)
            raise WireError(
                f"{which} payload failed its integrity check: peer sent digest {expected}, "
                f"{nbytes} B in the segment hash to {actual}. The bytes that crossed are not the "
                f"bytes that were written -- do NOT treat this as a transient fault."
            )

    def poison(self, which: str) -> None:
        """Overwrite a region so a retained view reads garbage instead of plausible data.

        Opt-in (``B1K_WIRE_POISON=1``). A stale view is the one bug this transport can produce
        that does NOT raise, and a wrong observation that looks like an observation is worth more
        debugging time than the memcpy costs.
        """
        if _POISON:
            region = self._region(which)
            region[:] = b"\xa5" * len(region)
            region.release()

    def close(self) -> None:
        """Unmap; unlink too if this side created it. Safe to call twice.

        **Unlinks FIRST, and never raises when views are outstanding.** A live numpy view is an
        exported pointer into the mapping, and ``mmap.close()`` refuses while one exists. Raising
        would turn a caller who held an array a moment too long into a failed teardown, on the
        path that runs during error handling. Unlinking is unaffected by live mappings, so doing
        it first means ``/dev/shm`` is reclaimed even in that case; the mapping itself goes when
        the last view is collected.
        """
        if self._closed:
            return
        self._closed = True
        self.unlink_now()
        try:
            self._buf.close()
        except BufferError:
            print(
                f"[wire] segment {self._path} closed while views were still held: the mapping "
                f"survives until they are collected, and any array still pointing into it now "
                f"aliases a buffer nobody owns. Drop views before close().",
                flush=True,
            )


#: How often the orphan watchdog looks. Long enough to be free, short enough that a leaked
#: simulator does not hold a card for most of a unit.
PARENT_POLL_S = 5.0

#: Exit status when the watchdog wins the race. Distinct from 1 so a reader of the logs can tell
#: "the simulator failed" from "the simulator was orphaned and cleaned itself up".
EXIT_ORPHANED = 3


def exit_when_orphaned(poll_s: float = PARENT_POLL_S) -> None:
    """Exit if the process on the other end of the wire disappears, even mid-silence.

    The socket already covers the ordinary cases: a peer that exits closes its end, so the next
    ``recv_frame`` returns ``None`` and the next ``send_frame`` raises. What it does not cover is
    a peer SIGKILLed while this process is busy elsewhere -- booting Isaac, encoding a package,
    waiting on a lease -- because none of those touch the socket, and that window is minutes
    long. A simulator that outlives it holds a GPU until something else reaps it.

    Reparenting is the signal: when the parent dies this process is adopted by init, so
    ``getppid()`` changes exactly once and never spuriously.

    Deliberately NOT ``PR_SET_PDEATHSIG``, which is the reflex here and is wrong: it fires on the
    death of the parent THREAD, and the gateway is a Ray actor whose calls are not promised to
    stay on one thread. It would kill a healthy simulator mid-episode, and the failure would look
    like the simulator crashing rather than like a watchdog misfiring.

    Started as a daemon thread so it never keeps a finished process alive.
    """
    original = os.getppid()

    def poll() -> None:
        while os.getppid() == original:
            time.sleep(poll_s)
        print(f"[wire] parent (pid {original}) is gone; exiting rather than holding resources.", flush=True)
        os._exit(EXIT_ORPHANED)

    threading.Thread(target=poll, name="orphan-watchdog", daemon=True).start()


def sweep_orphans(prefix: str = SEGMENT_PREFIX) -> list[str]:
    """Unlink segments whose creating process is gone. Returns what it removed.

    A hard death (segfault, OOM-kill, ``SIGKILL``) skips every ``finally``, and the segment holds
    real memory until something removes the name. Called at gateway startup, so a container that
    outlives several units does not accumulate them.

    **Scoped by PID, not by prefix.** A pod runs one gateway per rank, and ``ranks_per_pod > 1``
    means live siblings own segments with the same prefix -- an unconditional sweep would unlink
    theirs. The name carries the creator's PID for exactly this check.

    The liveness test is deliberately biased. PID space wraps in about four hours, so a recycled
    PID can make a dead creator look alive: that direction leaks one segment until the container
    exits, which is harmless. The other direction is also survivable here and worth stating --
    unlinking removes only the NAME, and both peers already hold their mapping, so a live pair is
    unaffected and only a subsequent ``attach`` would fail. Attaches happen once, at startup.
    """
    removed = []
    try:
        names = os.listdir(SHM_DIR)
    except OSError:
        return removed
    for name in names:
        if not name.startswith(prefix):
            continue
        rest = name[len(prefix) :]
        pid_text = rest.split("-", 1)[0]
        if not pid_text.isdigit():
            continue
        pid = int(pid_text)
        if pid == os.getpid() or os.path.exists(f"/proc/{pid}"):
            continue
        try:
            os.unlink(os.path.join(SHM_DIR, name))
            removed.append(name)
        except OSError:
            pass
    return removed


def observation_payload(segment: Segment, observations: Sequence[Mapping[str, Any]]) -> dict:
    """Stage a flat observation batch; descriptors and their digest travel together."""
    arrays, scalars = {}, []
    for lane, observation in enumerate(observations):
        rest = {}
        for key, value in observation.items():
            if isinstance(value, np.ndarray):
                arrays[f"{lane}{LANE_SEP}{key}"] = value
            else:
                rest[key] = value
        scalars.append(rest)
    descriptors = segment.write_arrays("obs", arrays)
    return {
        "descs": [d.as_wire() for d in descriptors],
        "scalars": scalars,
        "digest": segment.digest("obs", span_of(descriptors)),
    }


def read_observations(segment: Segment, frame: Mapping[str, Any]) -> list[dict]:
    """Verify before constructing views; consumers must copy before the next crossing."""
    descriptors = [ArrayDescriptor.from_wire(d) for d in frame["descs"]]
    segment.verify("obs", span_of(descriptors), frame["digest"])
    batch = [dict(value) for value in frame.get("scalars", [])]
    for key, array in segment.read_arrays("obs", descriptors).items():
        lane, _, name = key.partition(LANE_SEP)
        batch[int(lane)][name] = array
    return batch


def span_of(descs: Iterable[ArrayDescriptor]) -> int:
    """The packed length a descriptor list covers, for :meth:`Segment.digest`.

    ``max(offset + nbytes)`` rather than ``sum(nbytes)``: the two agree for the contiguous
    packing :meth:`Segment.write_arrays` produces, but the max is what the digest must cover if
    a future writer ever leaves a gap, and a sum would then hash the wrong span while looking
    correct on every test that uses the current writer.
    """
    return max((d.offset + d.nbytes for d in descs), default=0)


def _count(desc: ArrayDescriptor) -> int:
    itemsize = np.dtype(desc.dtype).itemsize
    if desc.nbytes % itemsize:
        raise WireError(f"{desc.key!r}: {desc.nbytes} B is not a whole number of {desc.dtype} items")
    return desc.nbytes // itemsize


def _json_default(value: Any) -> Any:
    """Encode what ``sim.assert_wire_safe`` PERMITS but ``json`` cannot.

    These two contracts have to agree or the seam is not transport-neutral. ``assert_wire_safe``
    is the declared payload contract for this boundary and it explicitly allows numpy scalars and
    ndarrays -- and a b1k session really does put them in ``EpisodeResult.metrics``, since the
    numbers come out of OmniGibson. On the ``direct`` path those cross Ray, which pickles them
    without comment. On ``gateway`` they reach a bare ``json.dumps``.

    So without this, a session whose metrics hold an ``np.float32`` works in ``direct`` and
    raises ``TypeError`` in ``gateway`` -- at the END of a unit, after all the physics has been
    paid for. That is the same class of divergence as the policy-fault relay: the two modes must
    fail, and succeed, on the same inputs.

    ``.item()`` for scalars is exact and size-neutral. ``.tolist()`` for arrays is deliberate and
    NOT a licence to put arrays here: :data:`MAX_FRAME_BYTES` is still the backstop, and its
    message already says what an array on the control path means.
    """
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(
        f"{type(value).__name__} cannot cross the control path. It is also refused by "
        f"`sim.assert_wire_safe`, so this is a payload that should never have been built, "
        f"not a gap in the encoder."
    )


def send_frame(sock: socket.socket, kind: str, payload: Mapping[str, Any] | None = None) -> None:
    """Send one control frame: ``[u32 length][utf-8 JSON]``."""
    blob = json.dumps({"kind": kind, **(payload or {})}, separators=(",", ":"), default=_json_default).encode()
    if len(blob) > MAX_FRAME_BYTES:
        raise WireError(
            f"control frame for {kind!r} is {len(blob)} B, over the {MAX_FRAME_BYTES} B cap. "
            f"Control frames carry descriptors and scalars; an array in here would bypass the "
            f"shared buffer and undo the transport's whole reason to exist."
        )
    try:
        sock.sendall(_HEADER.pack(len(blob)) + blob)
    except OSError as exc:
        raise WireError(f"sending {kind!r}: {exc}") from exc


def group_alive(pgid: int) -> bool:
    """Is anything left in this process group? Signal 0, delivered to nobody.

    ONE implementation, used by both halves of a teardown: ``sim_supervisor`` polls its
    simulator's group, and ``sim_gateway`` polls the group a ``hello`` reported. Two copies
    would be two chances to get the ``PermissionError`` branch wrong, and that is the branch
    that matters -- a group we are not ALLOWED to signal is a group that EXISTS, and reading the
    refusal as "empty" would report a clean teardown over a live simulator holding a card.

    ``ProcessLookupError`` is the ordinary race rather than a fault: a group drains between the
    check and the signal all the time.
    """
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def bind_listener(path: str, *, backlog: int = 1) -> socket.socket:
    """An ``AF_UNIX`` listener at ``path``, replacing whatever stale file is there.

    Shared by both halves because either one can be the party that binds, and that is the whole
    question this function exists to make symmetric. When the GATEWAY binds, the rendezvous dies
    with the Ray actor; when the SIMULATOR binds, it survives the Ray side and a later gateway
    can find the same live simulator. The second is what "the simulator container is the host"
    actually means at the socket layer.

    The unlink is unconditional and is not a race with a live peer: a bound-but-unlinked socket
    keeps working for everyone already connected, and a peer connecting later is meant to reach
    the NEW listener. What it does prevent is ``EADDRINUSE`` against the file a crashed process
    left behind, which is otherwise the normal state after any hard death.
    """
    if len(path.encode()) > 107:
        raise WireError(f"unix socket path is {len(path.encode())} B, over the 107 B sun_path limit: {path!r}")
    if os.path.exists(path):
        os.unlink(path)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(path)
    sock.listen(backlog)
    return sock


def connect_with_retry(path: str, *, timeout_s: float, poll_s: float = 0.1) -> socket.socket:
    """Connect to ``path``, WAITING for it to appear. Raises :class:`WireTimeout` if it does not.

    The retry is the point, not a convenience. Whoever connects has no way to be ordered after
    whoever binds -- under a supervisor the listener disappears and comes back on every restart,
    and on a cold start the two processes are launched by the same script microseconds apart.
    A single ``connect`` turns that ordinary interleaving into ``FileNotFoundError``, which reads
    like a configuration error and is not one.

    Two errnos retry and everything else raises. ``ENOENT`` is "not bound yet"; ``ECONNREFUSED``
    is "the file is there but nobody is accepting", which is exactly the window between a crash
    and the supervisor's rebind. A ``PermissionError`` or an ``ENOTSOCK`` is a real
    misconfiguration and waiting on it would only postpone the message by ``timeout_s``.
    """
    deadline = time.monotonic() + timeout_s
    last: OSError | None = None
    while True:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.connect(path)
        except (FileNotFoundError, ConnectionRefusedError) as exc:
            sock.close()
            last = exc
        except OSError:
            sock.close()
            raise
        else:
            return sock
        if time.monotonic() >= deadline:
            raise WireTimeout(
                f"nothing was listening on {path!r} after {timeout_s:.0f}s (last: {last!r}). "
                f"The peer that OWNS this socket never bound it, or bound a different path."
            )
        time.sleep(poll_s)


def recv_frame(sock: socket.socket) -> dict | None:
    """Receive one control frame. ``None`` at a clean EOF (the peer closed).

    ``None`` rather than an exception because a clean close is how a unit ends, and a transport
    that raised on its own success would push every caller into catching the same error for two
    opposite reasons.
    """
    header = _recv_exact(sock, _HEADER.size)
    if header is None:
        return None
    (length,) = _HEADER.unpack(header)
    if length > MAX_FRAME_BYTES:
        raise WireError(f"peer announced a {length} B frame, over the {MAX_FRAME_BYTES} B cap")
    body = _recv_exact(sock, length)
    if body is None:
        raise WireError(f"peer closed mid-frame: announced {length} B, got none")
    try:
        frame = json.loads(body)
    except json.JSONDecodeError as exc:
        raise WireError(f"peer sent {length} B that is not JSON: {exc}") from exc
    if not isinstance(frame, dict) or "kind" not in frame:
        raise WireError(f"frame is not a tagged object: {frame!r}")
    return frame


def _recv_exact(sock: socket.socket, n: int) -> bytes | None:
    """Exactly ``n`` bytes, or ``None`` at EOF before the first byte.

    A ``SOCK_STREAM`` does not preserve message boundaries, so a bare ``recv(n)`` short-reads
    under load and only under load -- the worst way to find out that a frame decoder was written
    against a datagram mental model.
    """
    chunks: list[bytes] = []
    got = 0
    while got < n:
        try:
            piece = sock.recv(min(n - got, 1 << 20))
        except socket.timeout as exc:
            # BEFORE the OSError branch: socket.timeout IS an OSError (it is TimeoutError from
            # 3.10), so the order here is what makes the distinction reachable at all.
            raise WireTimeout(f"receiving: the peer sent nothing for {sock.gettimeout()}s") from exc
        except OSError as exc:
            raise WireError(f"receiving: {exc}") from exc
        if not piece:
            if got == 0:
                return None
            raise WireError(f"peer closed after {got} of {n} B")
        chunks.append(piece)
        got += len(piece)
    return b"".join(chunks)
