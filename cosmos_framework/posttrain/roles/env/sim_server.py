# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""The simulator-side half of the container split: the process that speaks no Ray.

Run as ``python -m cosmos_framework.posttrain.roles.env.sim_server --socket <path>`` in the
simulator's own venv, spawned by ``roles/env/sim_gateway.py``. It holds everything that touches
the engine -- ``CollectorBody``, the driver loop, the sink, the source -- and reaches the policy
only through :class:`_RemotePolicy`, which forwards ``describe``/``act`` back up the socket.

**This module is what makes the version assertions in ``docker/Dockerfile.behavior1k`` removable
-- they are still there.** Nothing here imports Ray, which removes the *reason* for the
pin (``embodied/plugin_surface.py`` carries the argument); actually deleting it also needs the
sim raylet gone from ``utils/launch_multinode_eval.sh``. The only thing crossing the
process boundary is a unix socket and a shared buffer, which is what the design thread meant by
*"an interaction over shmem and a unix socket should not depend on the python or cuda versions;
the envs can be isolated"*.

**Import discipline is load-bearing here, not stylistic.** Everything this module reaches must be
importable without Ray, torch, or the native policy stack -- that set is declared in
``embodied/plugin_surface.py`` and enforced by ``tests/test_import_discipline.py``. A stray
framework import would still WORK on a single-image deployment and would only fail once the
images actually diverge, which is exactly the coupling being removed.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import traceback
from typing import Any, Mapping, Sequence

import numpy as np

from cosmos_framework.posttrain.embodied.rollout import RolloutInfo, rollout_info_from
from cosmos_framework.posttrain.embodied.wire import (
    FAULT,
    WIRE_VERSION,
    ArrayDescriptor,
    Segment,
    WireError,
    bind_listener,
    exit_when_orphaned,
    policy_fault_from,
    recv_frame,
    send_frame,
    session_identity,
    span_of,
)

#: Fault injection, DISARMED unless ``B1K_SIM_FAULT_AFTER_ACTS`` is set to a positive int.
#:
#: **Why this lives in production code rather than in a test double.** The claim it exists to
#: test -- that a supervised restart under ``sim_attach`` saves an Isaac boot -- needs a REAL
#: simulator dying at a REAL moment. The CPU fake has no Isaac to boot, and nothing on a pod can
#: reach in and kill one: this CLI has no `lep job exec`. So the simulator kills itself on request.
#:
#: Three properties, each one a scar from somewhere else in this subsystem:
#:
#: * **It dies at an `act`, not at startup.** By then Isaac has booted and the scene is built,
#:   so the restart is paying for exactly the thing under measurement. A fault at startup would
#:   measure a cheap restart and be reported as a dear one.
#: * **It dies ONCE across incarnations**, claimed with ``O_EXCL`` on a marker file so that "am
#:   I the first?" and "claim first" are one syscall. In-process state cannot do this: the next
#:   incarnation is a different process, which is the entire thing under test. Without it
#:   generation 1 dies too, the budget is spent, and a working restart reads as a broken one.
#: * **``os._exit``, not an exception.** No error frame, no clean ``close``, no ``finally`` --
#:   so the gateway sees ``PeerGone`` rather than a unit-level fault, which is the only path
#:   that reaches ``reattach()``. It also leaves the process group behind, which is what makes
#:   the supervisor's ``terminate_tree`` do real work rather than reap an already-empty group.
FAULT_AFTER_ACTS_ENV = "B1K_SIM_FAULT_AFTER_ACTS"
FAULT_MARKER_ENV = "B1K_SIM_FAULT_MARKER"

#: Counted per PROCESS, not per session: a simulator in host mode serves many gateways, and the
#: question the injector answers is "how far into this incarnation's work did it get".
_acts_seen = 0


def _fault_after_acts_if_armed() -> None:
    """Kill this process hard, once, after N ``act`` calls. No-op unless armed."""
    global _acts_seen
    after = int(os.environ.get(FAULT_AFTER_ACTS_ENV, "0") or "0")
    if after <= 0:
        return
    _acts_seen += 1
    if _acts_seen < after:
        return
    marker = os.environ.get(FAULT_MARKER_ENV, "")
    if marker:
        try:
            os.close(os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600))
        except FileExistsError:
            # Someone already died for this campaign. Say so -- an injector that goes quiet
            # after arming is indistinguishable from one that never armed.
            print(
                f"[sim] fault injector: {_acts_seen} act(s) seen, but {marker!r} already "
                f"exists, so this incarnation lives (armed check, not silence)",
                flush=True,
            )
            return
    print(
        f"[sim] fault injector: FIRING at act #{_acts_seen} (>= {after}); exiting 9 without a "
        f"clean close, so the gateway sees PeerGone",
        flush=True,
    )
    os._exit(9)


class _RemotePolicy:
    """A ``Rollout`` whose ``act`` and ``describe`` cross the socket.

    Those two are the entire policy surface the driver touches -- measured, not assumed: they are
    the only names read on the policy handle anywhere in ``embodied/`` or ``roles/env/``. A third
    would need an entry here and a branch in the gateway's service loop, which is the intended
    friction.
    """

    def __init__(self, sock: socket.socket, segment: Segment) -> None:
        self._sock = sock
        self._seg = segment
        self._info: RolloutInfo | None = None

    def describe(self) -> RolloutInfo:
        """Cached after the first call: ``validate_rollout`` and the driver both ask, and the
        answer is static by contract."""
        if self._info is None:
            send_frame(self._sock, "describe")
            self._info = rollout_info_from(_expect(self._sock, "describe_reply")["info"])
        return self._info

    def act(self, obs_batch: Sequence[Mapping[str, Any]], *, context: dict | None = None) -> list[np.ndarray]:
        """One batched forward, through the shared buffer.

        Arrays go into the segment; anything else (``task_name``) rides in the JSON frame. That
        split is not a special case for one key -- it is the rule that keeps the control path
        small enough to stay JSON, and ``embodied/sim.py``'s ``assert_wire_safe`` already
        guarantees the values are primitives.
        """
        _fault_after_acts_if_armed()
        from cosmos_framework.posttrain.embodied.wire import observation_payload

        send_frame(
            self._sock,
            "act",
            {
                # The digest travels in the SAME frame as the descriptors it covers, so a peer
                # cannot satisfy the check by disagreeing about the span.
                **observation_payload(self._seg, obs_batch),
                **({"context": context} if context is not None else {}),
            },
        )
        reply = _expect(self._sock, "act_reply")
        act_descs = [ArrayDescriptor.from_wire(d) for d in reply["descs"]]
        # Verified BEFORE the views are built: the point is that nothing downstream ever sees
        # bytes that failed. The act payload is small (2,944 B per chunk measured), so this
        # direction costs nothing measurable at all.
        self._seg.verify("act", span_of(act_descs), reply["digest"])
        chunks = self._seg.read_arrays("act", act_descs)

        # COPIED out of the segment, deliberately. The driver holds a chunk across the whole
        # 32-step window it covers, and the act region is overwritten by the NEXT call -- so a
        # view here would silently become the following chunk partway through being consumed.
        # 2,944 B per chunk measured, so the copy is free; the bug it prevents is not.
        return [np.array(chunks[str(i)]) for i in range(len(obs_batch))]

    def feedback(self, observations: Sequence[Mapping[str, Any]], *, context: dict) -> None:
        from cosmos_framework.posttrain.embodied.wire import observation_payload

        send_frame(self._sock, "feedback", {**observation_payload(self._seg, observations), "context": context})
        _expect(self._sock, "feedback_reply")


def _expect(sock: socket.socket, kind: str) -> dict:
    frame = recv_frame(sock)
    if frame is None:
        raise WireError(f"expected {kind!r}, the gateway closed the connection instead")
    # A fault is the framework's exception, relayed. Raised as PolicyFault rather than WireError
    # because nothing here is broken: this socket is healthy and stays in step -- the next frame
    # this process sends is simply the next request. Distinguishing them is what lets
    # `driver.run_unit` fail ONE unit and carry on, which is the whole point of the frame.
    if frame["kind"] == FAULT:
        raise policy_fault_from(frame, kind)
    if frame["kind"] != kind:
        raise WireError(f"expected {kind!r}, got {frame['kind']!r}")
    return frame


#: How one session ended. Three outcomes, because a HOST has to tell two of them apart that a
#: child never had to: ``close`` means *shut down*, EOF means *my client went away*.
SESSION_CLOSED = "closed"
SESSION_PEER_LEFT = "peer_left"
SESSION_FAILED = "failed"


def serve(sock: socket.socket) -> int:
    """One session, as a process exit code. The connect-mode entry point.

    Kept so the existing shape -- one connection is the process's whole life -- keeps its exit
    codes exactly. ``_child_exit_code`` and the gateway's hard-death tests read them.
    """
    return 1 if serve_session(sock, session_identity()) == SESSION_FAILED else 0


def serve_session(sock: socket.socket, identity: tuple[str, int]) -> str:
    """The child's whole life: announce, configure, then answer commands until told to stop.

    Errors are reported as an ``error`` frame rather than only as a traceback on stdout. A
    simulator that dies silently looks to the gateway like a clean exit, and the difference
    matters: one retires a unit, the other hands it back.
    """
    from cosmos_framework.posttrain.roles.env.collector import CollectorBody
    from cosmos_framework.posttrain.roles.env.sim_role import capabilities_as_wire

    segment: Segment | None = None
    body: Any = None
    try:
        # The handshake is five frames, and the shape is forced by one fact: the gateway cannot
        # size the shared segment until it knows the embodiment, and the embodiment is a property
        # of the body the gateway asked for. So the body is built FIRST and the segment second.
        #
        # The alternative -- build a throwaway simulator just to read `capabilities()` -- is
        # cheap (nothing boots until `open()`), but it would be constructed with different
        # options than the real one and keyed off an env var nobody sets. Two simulators that
        # disagree about the embodiment is precisely the failure the handshake exists to catch.
        # The CHILD mints the identity, not the gateway -- see `wire.session_identity`.
        sid, gen = identity
        # `visible_devices` is THIS process's own reading, taken before anything is built. The
        # gateway has only ever known what it INTENDED to hand a child; `sim_role.py`'s S-04
        # check says so in as many words -- it "cannot see a child that ignored the variable".
        # Now it can, and in attach mode (where the gateway spawns nothing and therefore intends
        # nothing) this is the only reading that exists at all.
        #
        # Read raw and sent raw: no normalising, no defaulting to "0". `_one_device` on the far
        # side maps "" and "?" to "no answer", which is a THIRD outcome the check already
        # handles, and inventing a device here would convert an unanswerable question into a
        # confident wrong answer.
        send_frame(
            sock,
            "hello",
            {
                "wire_version": WIRE_VERSION,
                "session_id": sid,
                "generation": gen,
                "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                # THIS process's group, so a teardown can reach Isaac's helper processes without
                # assuming the process the gateway spawned IS the simulator. That assumption
                # holds only while the gateway spawns `sim_server` directly; put a supervisor
                # between them and the gateway's `killpg` reaches the supervisor's group alone,
                # leaving the simulator's tree on the card. Reported rather than derived, because
                # the gateway cannot see through whatever it started.
                "pgid": os.getpgid(0),
            },
        )

        cfg = _expect(sock, "configure")
        body_cfg = dict(cfg["body"])
        # Registrations must happen in THIS interpreter. The direct path preloads in the same
        # process that builds the body, so a component registered there is visible to it; here
        # the two are different processes, and a gateway-side import registers into the wrong
        # one. The symptom without this is an unknown sim/sink key, which reads as a config
        # typo rather than as a module that was imported on the far side of a socket.
        for module in body_cfg.pop("preload", ()):
            __import__(module)
        body = CollectorBody(**body_cfg)
        # FRAMEWORK ISOLATION, reported AFTER the body is built and therefore after the backend
        # has actually loaded. `tests/test_import_discipline.py`'s G-s walks this module's import
        # closure statically and proves OUR code never reaches Ray -- but it says so about the
        # eager path only, and it says nothing at all about what the ENGINE drags in, because
        # backends arrive through `embodied/registry.py` by name and the registry imports them
        # lazily. This line closes that gap, from inside the process that would suffer for it.
        print(
            f"[sim] framework isolation @body: ray "
            f"{'PRESENT -- the sim venv is still coupled to the policy venv' if 'ray' in sys.modules else 'absent'}"
            f" in sys.modules after the body is built (armed check, not silence)",
            flush=True,
        )
        send_frame(sock, "capabilities", {"capabilities": capabilities_as_wire(body.capabilities())})

        attach = _expect(sock, "attach")
        segment = Segment.attach(attach["segment"], int(attach["obs_capacity"]))
        send_frame(sock, "ready")

        policy = _RemotePolicy(sock, segment)
        while True:
            frame = recv_frame(sock)
            if frame is None:
                # The gateway VANISHED. Distinct from `close`, and in listen mode the difference
                # decides whether this process ends: a host waits for the next client, a child
                # exits. Note what does NOT survive either way -- the `finally` below closes the
                # body, because `configure` is per-session and the next gateway may ask for a
                # different one. What survives is the PROCESS: the interpreter, the imports and
                # Isaac itself, which is where the ~294 s of bind time actually goes.
                return SESSION_PEER_LEFT
            if frame["kind"] == "close":
                return SESSION_CLOSED
            kind = frame["kind"]
            if kind == "probe":
                reply = body.probe()
                send_frame(sock, "probe_reply", {"host": reply["host"], "gpu": reply["gpu"]})
            elif kind == "run":
                method = getattr(body, frame["method"])
                result = method(frame["units"], policy=policy, **frame.get("kwargs", {}))
                send_frame(sock, "result", {"result": result})
            else:
                raise WireError(f"unexpected frame {kind!r} from the gateway")
    except BaseException as exc:  # noqa: BLE001 -- the gateway must learn WHY, not just that
        traceback.print_exc()
        try:
            send_frame(sock, "error", {"error": f"{type(exc).__name__}: {exc}"})
        except Exception:  # noqa: BLE001 -- the socket may be what failed
            pass
        return SESSION_FAILED
    finally:
        # THE SECOND HALF OF THE ISOLATION CLAIM, and the reason there are two.
        #
        # The `@body` receipt above fires after `CollectorBody(**body_cfg)` -- which is BEFORE
        # `sim.open()`. Isaac has been imported by then but no scene has been built, so a lazy
        # `import ray` anywhere in the engine's scene path would not have run yet and the early
        # receipt would report `absent` while being blind to it. Measured on a card
        # (`simiso-p00-bx4b`) the early one said `absent`; that was a true answer to a smaller
        # question than the Dockerfile pin actually asks.
        #
        # This one fires at SESSION END, after every scene build, every step and every sink
        # write this process will ever do, so between the two the whole lifetime is covered.
        print(
            f"[sim] framework isolation @session-end: ray "
            f"{'PRESENT -- the sim venv is still coupled to the policy venv' if 'ray' in sys.modules else 'absent'}"
            f" in sys.modules after a full session (armed check, not silence)",
            flush=True,
        )
        if body is not None:
            try:
                body.close()
            except Exception:  # noqa: BLE001 -- teardown must not mask the original failure
                traceback.print_exc()
        if segment is not None:
            segment.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", required=True, help="the AF_UNIX rendezvous path")
    parser.add_argument(
        "--listen",
        action="store_true",
        help=(
            "OWN the socket: bind it and wait for a gateway, instead of connecting to one the "
            "gateway bound. This is what makes the simulator the host -- the rendezvous then "
            "outlives the Ray side, so a restarted gateway finds the SAME live simulator."
        ),
    )
    args = parser.parse_args(argv)

    # ARM RECEIPT, printed in BOTH directions and before anything can fail. A production corpus
    # needs to be able to prove no injector was live in the process that produced it, and
    # "disarmed" and "this build has no injector" must not look the same in a log.
    _after = int(os.environ.get(FAULT_AFTER_ACTS_ENV, "0") or "0")
    if _after > 0:
        print(
            f"[sim] fault injector ARMED: will exit 9 at act #{_after}, once, "
            f"marker={os.environ.get(FAULT_MARKER_ENV, '')!r}",
            flush=True,
        )
    else:
        print("[sim] fault injector: disarmed (armed check, not silence)", flush=True)

    # THE EXECUTION-ENVIRONMENT RECEIPT. This process is the SLOW SIDE of the ~1.8x attach
    # regression (SPEC 0.43), and 94-99% of the extra wall clock is inside its own stepping
    # loop -- not boot, not bind, not scene load, and not the policy, whose diffusion sampler
    # matches across arms to 0.1-0.3%. The per-step code path is byte-identical in both modes
    # (`self._attach` is read at eight sites in `sim_gateway.py`, none of them in the hot loop),
    # so the cause cannot be code. It has to be what this process INHERITED.

    #
    # The leading hypothesis: Ray's `set_omp_num_threads_if_unset()` sets OMP_NUM_THREADS=1 for
    # an actor with no `num_cpus`, and `workers/grid.py` never passes `num_cpus`. So under SPAWN
    # this process descends from that actor and inherits 1; under ATTACH it descends from the
    # pod's bash and inherits nothing. Arithmetic: at a measured 8.42x slowdown for small-op
    # torch at 64 threads vs 1, ~14-16% of the step being OpenMP-parallel accounts for the whole
    # 2.0x. THE AFFINITY MASK IS WHAT DECIDES IT -- at a 15-CPU mask the model needs f=225% and
    # the hypothesis dies outright. That is why the mask is printed here and not only the env.
    #
    # `cwd` is the distant second hypothesis: `_spawn` passes an explicit cwd, `sim_supervisor`
    # passes none, so the attach child inherits bash's. Free to print, so printed.
    # READ torch's thread count; NEVER IMPORT torch TO GET IT. `test_import_discipline`'s
    # sim-side closure forbids a runtime `import torch` in this module, and the reason is the
    # whole decouple: the sim venv has its own torch and must not inherit the policy venv's.
    # The first version of this block did the import inside a try/except, which does not help --
    # the walk is an AST walk, so it sees the `import` statement whatever guards it at runtime.
    #
    # `sys.modules` is also the HONEST instrument: importing torch to measure would change the
    # environment being measured.
    _t = sys.modules.get("torch")
    if _t is None:
        _torch_threads = "<not-loaded>"
    else:
        try:
            _torch_threads = _t.get_num_threads()
        except Exception as _e:  # loaded but broken -- say which, do not swallow
            _torch_threads = f"<{type(_e).__name__}>"
    print(
        f"[sim] exec env: OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')!r} "
        f"MKL_NUM_THREADS={os.environ.get('MKL_NUM_THREADS')!r} "
        f"affinity={len(os.sched_getaffinity(0))} cpu_count={os.cpu_count()} "
        f"torch_threads={_torch_threads} cwd={os.getcwd()!r} "
        f"ppid={os.getppid()} (armed check, not silence)",
        flush=True,
    )

    exit_when_orphaned()
    if not args.listen:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(args.socket)
        try:
            return serve(sock)
        finally:
            sock.close()

    # ── host mode ──
    # Bound BEFORE anything expensive, so a gateway launched alongside us has something to
    # connect to while we import. `connect_with_retry` covers the other ordering; this covers
    # the long tail, where the import IS the delay -- a cold NFS venv has been measured in the
    # tens of seconds, and that is a window in which the path simply must exist.
    listener = bind_listener(args.socket)
    # ONCE, OUTSIDE THE LOOP, and this is the line that makes the fence work. `session_id`
    # answers "am I talking to the same simulator?", so it must be scoped to the SIMULATOR --
    # the process -- not to a connection. With no supervisor it is minted from a uuid, so
    # calling it per session gave one process a different identity on every accept, and a
    # gateway reconnecting to the very same live simulator was told it had found a new one.
    # Measured: the same pid reported sim-847814-cba73a944919 and sim-847814-ce2da2d02c1f.
    identity = session_identity()
    print(
        f"[sim] listening on {args.socket!r} as session={identity[0]}/gen{identity[1]}; waiting for a gateway",
        flush=True,
    )
    try:
        while True:
            sock, _ = listener.accept()
            try:
                outcome = serve_session(sock, identity)
            finally:
                sock.close()
            if outcome == SESSION_PEER_LEFT:
                print("[sim] the gateway went away; waiting for another", flush=True)
                continue
            # A FAILED session ends the process rather than waiting for another client, and that
            # is deliberate. Recovery from a broken simulator is a RESTART -- which a supervisor
            # performs, incrementing the generation, which is the value the gateway's identity
            # fence needs in order to know that simulator state was invalidated. Surviving the
            # failure in place would keep the generation still and make a broken incarnation
            # indistinguishable from a healthy one.
            return 1 if outcome == SESSION_FAILED else 0
    finally:
        listener.close()


if __name__ == "__main__":
    sys.exit(main())
