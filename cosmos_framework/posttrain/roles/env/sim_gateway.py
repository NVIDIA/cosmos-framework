# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""The framework-side half of the container split: a ``SimRole`` that forwards, and nothing else.

``SimGatewayBody`` is a drop-in for ``CollectorBody`` as the ``"collector"`` body. It satisfies
the same ``SimRole`` surface, so every other role -- and ``bind_device_groups`` -- sees exactly
what it saw before. Behind it, the simulator runs as a plain subprocess in its own venv, speaking
no Ray at all.

**Where the seam is, and why it is not one layer down.** The split is at ``SimRole``, not at
``Simulator``. Splitting at ``Simulator`` would put ``to_record_view``'s frame on the wire on
EVERY step -- 2,937,600 B x 32 per chunk, ~94 MB -- instead of the observation once per chunk.
That is 32x the traffic and it voids every number the transport was chosen on. So the driver, the
sink and the source all stay on the sim side; what crosses is the two calls the driver makes on
the policy handle, ``describe`` and ``act``, measured as the only two names touched on that
handle anywhere in ``embodied/``.

**The direction of the calls is inverted, and that is the whole design.** The framework calls IN
(``run_from_source``), and then the simulator calls back OUT (``act``) many times before
answering. So :meth:`run_from_source` is not a forward-and-return; it hands the command over and
then *services* the child until a result arrives. One connection, strictly alternating, because
the flow is synchronous -- see ``embodied/wire.py``.

**What this body must get right that a same-venv body never had to.** The child's environment is
NOT the gateway's. The policy/sim pair binds through one STRICT_PACK bundle at
``1.0 / len(actor_groups)`` GPU each (``control/placement/colocate.py``), so Ray hands this
actor a REAL device -- but it is the GATEWAY's card, not the child's.

The variable is SET rather than inherited, and the reason survives any rebinding: the
child's card is an explicit INPUT, not a consequence of how the gateway happens to be bound --
and that binding has already changed once without this file being touched. **Nothing in this
tree asserts either binding**, so it is a property this body DEFENDS rather than one it may
assume. Reading Ray's own per-actor value instead would collapse the two independent
derivations that ``refuse_cross_card_gateway`` below exists to police; it is not folded in
because a wrong card still produces correct episodes, so it cannot be proven right by running it.

``LD_LIBRARY_PATH`` is the harder half: under one raylet the gateway is forked from the policy
venv's environment, whose CUDA libraries make Isaac Sim segfault during SimulationApp startup.
So both variables are SET rather than inherited, and :meth:`_child_env` is where that is decided.
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import tempfile
import time
import traceback
from dataclasses import asdict, is_dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from cosmos_framework.posttrain.control.capabilities_builtin import SIM_IMPL_KEY
from cosmos_framework.posttrain.embodied.rollout import rollout_info_as_wire
from cosmos_framework.posttrain.embodied.wire import (
    FAULT,
    WIRE_VERSION,
    PeerGone,
    Segment,
    WireError,
    WireTimeout,
    connect_with_retry,
    fault_payload,
    group_alive,
    recv_frame,
    refuse_identity_replay,
    segment_bytes,
    send_frame,
    span_of,
    sweep_orphans,
)
from cosmos_framework.posttrain.roles.core.registry import register_env
from cosmos_framework.posttrain.roles.env.sim_role import capabilities_from

#: How long to wait for the child's ``hello``. Generous: the child imports omnigibson and builds
#: nothing, but a cold NFS venv has been measured in the tens of seconds.
DEFAULT_START_TIMEOUT_S = 300.0


#: How long :meth:`_child_failure` will wait for the child's own error frame before giving up and
#: reporting the exit code alone. Short on purpose: it runs while another failure is already
#: being reported, so it must never become the thing that hangs.
_FAILURE_PROBE_S = 5.0

#: Which forwarded methods may be RE-ISSUED against a fresh simulator after one dies.
#:
#: The distinction is not about the transport, it is about who owns unit ownership.
#: ``run_from_source`` is a CLAIM LOOP: the ``run`` frame carries the campaign PLAN, and the
#: child claims one unit at a time from ``dirlease`` -- so re-issuing the identical frame to a
#: new child is idempotent by construction, because the claim plane already knows which units
#: are retired. Nothing in this process has to account for anything.
#:
#: ``run_units`` is the PUSH shape: the units in the frame ARE the work, with no claim plane
#: behind them. Re-issuing it would re-run every unit the dead child had already finished, and
#: this process cannot subtract them without becoming a second, weaker copy of the accounting
#: that exists in `dirlease` -- the second-counter coupling the whole sim-primary argument
#: objects to. So it is not resumable, and it says so rather than silently doing the wrong thing.
#:
#: Cost that IS paid either way: each handover consumes one of the unit's ``MAX_GENERATIONS``
#: (``collector.py::run_from_source``), so a simulator that crashes repeatedly on one unit walks
#: it toward permanently stuck. That is an argument for a SMALL restart budget, not for none.
_RESUMABLE_METHODS = frozenset({"run_from_source"})

#: The module a production gateway runs in the child. Compared by identity rather than sniffed,
#: so the "am I in a test?" question has exactly one answer and it is visible at the call site.
PRODUCTION_SIM_MODULE = "cosmos_framework.posttrain.roles.env.sim_server"


def _units_as_wire(units: Sequence[Any]) -> list[dict]:
    """Units as plain dicts, because the frame is JSON and a ``WorkUnit`` is a dataclass.

    ``json.dumps`` on a dataclass raises ``TypeError: Object of type WorkUnit is not JSON
    serializable`` -- a bare message naming neither the field nor the caller. The child wants
    dicts anyway: ``collector.py``'s ``_as_unit`` accepts a ``Mapping`` and rebuilds the unit on
    the far side, which is the same shape ``dirlease`` hands it when a unit is read back from
    disk.

    Not a ``default=`` hook on the encoder: that would silently accept any dataclass, including
    ones whose fields are themselves unserialisable, and report the failure one level deeper
    than the mistake.
    """
    return [asdict(u) if is_dataclass(u) and not isinstance(u, type) else u for u in units]


class SimGatewayBody:
    """A ``SimRole`` that owns a simulator subprocess and forwards to it.

    Constructor mirrors ``CollectorBody`` so ``setup_sim`` can return either one without the
    caller knowing which -- that substitutability IS the A/B, and it is also the property the
    design thread asked for: if isolation ever costs too much, the forwarding interface is
    swapped for a direct call and the result is the same object.
    """

    def __init__(
        self,
        *,
        sim: str = "behavior1k",
        sink: str = "package",
        embodiment: str = "r1pro_b1k",
        sim_options: dict | None = None,
        sink_options: dict | None = None,
        force_render: bool = True,
        render_every: int = 1,
        sim_python: str | None = None,
        sim_gpu: str | None = None,
        sim_ld_library_path: str | None = None,
        sim_ipc_dir: str | None = None,
        start_timeout_s: float = DEFAULT_START_TIMEOUT_S,
        sim_module: str = PRODUCTION_SIM_MODULE,
        preload: Sequence[str] = (),
        attach: bool = False,
        restart_budget: int = 0,
        rendezvous: str = "listen",
        sim_socket_path: str | None = None,
    ) -> None:
        self._child_cfg = {
            "sim": sim,
            "sink": sink,
            "embodiment": embodiment,
            "sim_options": dict(sim_options or {}),
            "sink_options": dict(sink_options or {}),
            "force_render": bool(force_render),
            "render_every": int(render_every),
            # Imported by the CHILD before the body is built. The direct path does this in
            # `setup_sim`, in the same process as the body; here the body is built across a
            # process boundary, so a registration performed gateway-side registers into the
            # wrong interpreter and the child fails with an unknown sim/sink key.
            "preload": list(preload),
        }
        # SIM_PY, not a new name: the image already exports it (docker/Dockerfile.behavior1k:347),
        # asserts it is executable (:357), and the launcher already consumes it. B1K_SIM_PYTHON
        # was invented here and set nowhere -- `git grep` finds this line and nothing else.
        self._sim_python = sim_python or os.environ.get("SIM_PY", "")
        # No SIM_GPUS default: that name holds the pod's whole device LIST, where this needs
        # ONE CELL's single device. Reading it would hand every cell on the pod the same list.
        self._sim_gpu = sim_gpu
        self._sim_ld = (
            sim_ld_library_path if sim_ld_library_path is not None else os.environ.get("SIM_LD_LIBRARY_PATH", "")
        )
        self._ipc_dir = sim_ipc_dir or os.environ.get("B1K_SIM_IPC_DIR") or tempfile.gettempdir()
        self._start_timeout_s = float(start_timeout_s)
        #: The module run as ``python -m`` in the child. A parameter ONLY so tests can substitute
        #: a stub that speaks the protocol without Isaac -- the alternative is that the gateway's
        #: forwarding is exercised solely on a GPU pod, i.e. never on the path where a bug in it
        #: is cheap to find. Production never sets it.
        self._sim_module = sim_module
        #: ATTACH rather than spawn: bind the socket and wait for a simulator that already
        #: exists to connect. This is the sim-primary shape -- the simulator container is the
        #: long-lived host and the Ray side is what comes and goes -- and it is the mode in
        #: which `sim_supervisor` is the sim container's entrypoint rather than a child of ours.
        #:
        #: Deliberately a constructor parameter with NO environment fallback. `sim_transport`
        #: already has six default sites and only two of them may move together; adding a
        #: seventh switch readable from the environment invites the same split-brain, where two
        #: halves of one launch disagree and the symptom is a bind that hangs with no message.
        #: Whoever wires the container split adds the config field, at the layer that knows the
        #: suite.
        self._attach = bool(attach)
        #: How many times a resumable forwarded call may be re-issued against a fresh simulator.
        #: DEFAULT 0 -- this changes nothing until something asks for it. It counts re-issued
        #: CALLS, not simulator restarts; the process that performs the restart lands with the
        #: fleet layer, and raising one without the other is inert in both directions.
        self._restart_budget = max(0, int(restart_budget))
        #: WHO OWNS THE SOCKET, which is the same question as who can outlive whom.
        #:
        #: ``"listen"`` -- this gateway binds and the simulator connects to it. The rendezvous
        #: then dies with the Ray actor, and since the path is per-PID a restarted actor binds a
        #: DIFFERENT one: a simulator that survived has no way back. That is fine while the
        #: gateway spawns the simulator, which is every production run to date.
        #:
        #: ``"connect"`` -- the SIMULATOR binds (``sim_server --listen``) and this gateway
        #: connects, retrying until the path appears. The rendezvous now outlives the Ray side,
        #: so a restarted gateway reaches the SAME live simulator -- which is what "the simulator
        #: container is the host" means at the socket layer, and the case
        #: ``refuse_identity_replay(expect_restart=False)`` exists for.
        if rendezvous not in ("listen", "connect"):
            raise WireError(f"rendezvous must be 'listen' or 'connect', not {rendezvous!r}")
        self._rendezvous = rendezvous
        self._explicit_socket = sim_socket_path or os.environ.get("B1K_SIM_SOCKET") or ""
        if self._rendezvous == "connect" and not self._explicit_socket:
            raise WireError(
                "rendezvous='connect' needs sim_socket_path (or $B1K_SIM_SOCKET): the simulator "
                "owns the socket in this mode, so the path is a value both halves must be TOLD, "
                "not one either can derive."
            )

        self._proc: subprocess.Popen | None = None
        self._sock: socket.socket | None = None
        self._listener: socket.socket | None = None
        #: The bound path, kept because three steps need it and `_socket_path` is not free --
        #: it re-checks the 107 B `sun_path` limit and can raise. Set by `_listen`.
        self._sock_path = ""
        self._seg: Segment | None = None
        self._caps: dict | None = None
        #: ``(session_id, generation)`` from the child's ``hello``. Re-handshaking against a
        #: DIFFERENT session, or one that did not advance, is refused -- see `_handshake`.
        self._identity: tuple[str, int] | None = None
        #: The child's OWN ``CUDA_VISIBLE_DEVICES``, from ``hello`` (wire >= 6). ``None`` until
        #: a handshake completes. This is what :meth:`child_gpu` answers with -- see there for
        #: why the child's reading beats this process's intent even when we set it ourselves.
        self._child_devices: str | None = None
        #: The SIMULATOR's own process group, from ``hello`` (wire >= 7). Teardown does not need
        #: it while the process this gateway spawned IS the simulator -- signalling
        #: ``self._proc.pid``'s group reaches the same tree. It is reported by :meth:`probe` so
        #: the value the wire insists on is one somebody can actually read, and so the operator
        #: has the group to signal when a teardown has to be finished by hand.
        self._child_pgid: int | None = None
        self._closed = False
        self._refuse_incomplete_production_config()
        self._start()

    def _refuse_incomplete_production_config(self) -> None:
        """Refuse to spawn a real simulator without being told which interpreter and which card.

        Both used to have defaults, and both defaults were wrong in a way that surfaces late.
        ``sys.executable`` is the FRAMEWORK venv -- the child would import a torch and a
        numpy built for the other half of the image, which is the coupling this whole split
        exists to remove, and it would get all the way to an omnigibson import before saying so.
        ``CUDA_VISIBLE_DEVICES`` defaulted to ``"0"``, which is correct for exactly one cell per
        pod and silently puts every other cell's Isaac on rank 0's card.

        Scoped to the production child module: the test stub needs neither, and demanding them
        there would only teach the tests to pass values that are never read.
        """
        if self._sim_module != PRODUCTION_SIM_MODULE:
            return
        if self._attach:
            # Nothing is spawned, so neither value has anything to be right ABOUT. `sim_python`
            # is inert. `sim_gpu` is worse than inert: `child_gpu` feeds S-04, and returning a
            # device this process never assigned to anything would manufacture bind-time
            # evidence out of an intent nobody acted on -- a check reporting SAME-CARD from two
            # readings of the same wish. Refused rather than ignored, because a caller that
            # passes it is working from the spawn-mode mental model and should find out here.
            if self._sim_gpu:
                raise WireError(
                    f"sim_gpu={self._sim_gpu!r} was passed with attach=True. In attach mode this "
                    f"gateway spawns nothing, so it assigns no card, and reporting one to the "
                    f"S-04 bind check would compare the policy's real device against a wish. "
                    f"Leave it unset: S-04 still has evidence, and BETTER evidence than spawn "
                    f"mode's -- `child_gpu` answers from the handshake, and `hello` carries the "
                    f"card the simulator actually holds rather than the one someone intended."
                )
            return
        missing = [
            name
            for name, value in (("sim_python (or $SIM_PY)", self._sim_python), ("sim_gpu", self._sim_gpu))
            if not value
        ]
        if missing:
            raise WireError(
                f"refusing to spawn {PRODUCTION_SIM_MODULE} without {' and '.join(missing)}. "
                f"Guessing either one produces a simulator that runs -- in the wrong venv, or on "
                f"another cell's GPU -- rather than one that fails."
            )

    # ── lifecycle ───────────────────────────────────────────────────────────

    def _socket_path(self) -> str:
        """A per-process path under a directory the container can bind in.

        ``AF_UNIX`` paths are capped at 107 bytes by ``sun_path``, and the failure is a bare
        ``OSError`` from ``bind`` that names neither the limit nor the path -- so the length is
        checked here, where both are in hand.
        """
        # In `connect` mode the simulator owns the socket, so the path is whatever it was told
        # -- never this process's pid, which is the one value guaranteed to change when the Ray
        # actor restarts and is therefore the one value a durable rendezvous cannot contain.
        path = self._explicit_socket or os.path.join(self._ipc_dir, f"b1k-sim-{os.getpid()}.sock")
        if len(path.encode()) > 107:
            raise WireError(
                f"unix socket path is {len(path.encode())} B, over the 107 B sun_path limit: "
                f"{path!r}. Set B1K_SIM_IPC_DIR to something shorter."
            )
        return path

    def _child_env(self) -> dict[str, str]:
        """The child's environment, with two variables SET rather than inherited.

        ``CUDA_VISIBLE_DEVICES``: set from ``sim_gpu``, never passed through. Under the binding
        this was written for (``num_gpus=0``) inheriting gave the simulator no GPU at all, and
        Isaac's failure for that is neither immediate nor obviously about devices. That binding
        is gone -- the pair now takes a GPU fraction each -- so today this makes the child's card
        an explicit input rather than a rescue. See the module docstring for why that is still
        wanted and what the alternative would buy.

        ``LD_LIBRARY_PATH``: replaced, not extended. ``utils/launch_multinode_eval.sh:75-77``
        says the policy venv's CUDA libraries "make Isaac Sim segfault during SimulationApp
        startup", and it keeps them out by applying them per-command instead of exporting them.
        Under one raylet the gateway runs *inside* that environment, so the protection that used
        to come from a separate ``ray start`` invocation has to be re-established here.
        """
        env = dict(os.environ)
        if self._sim_gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = self._sim_gpu
        if self._sim_ld:
            env["LD_LIBRARY_PATH"] = self._sim_ld
        else:
            env.pop("LD_LIBRARY_PATH", None)
        env["PYTHONUNBUFFERED"] = "1"
        return env

    def _start(self) -> None:
        """Bind, spawn, accept -- three steps, split out because the third has to run twice.

        The third step runs twice under ``reattach()``: a simulator that comes back connects
        into the same listener, and a gateway that can only ``accept()`` once has nothing to
        answer it with.

        The order is not arbitrary and not the obvious one. The listener must be BOUND before
        the child is spawned: the child connects to a path, and a child that wins the race
        against its own parent dies on ``ENOENT`` -- an error that names a path and says nothing
        about a startup ordering.
        """
        # BEFORE anything binds or spawns, and in BOTH directions. This used to live in
        # `_listen`, which meant connect mode -- the direction whose whole point is that a
        # rendezvous outlives the process that used it -- never swept at all. The sweep collects
        # segments owned by processes that are already gone, so by definition no `close()` of
        # ours ever ran for those. See `embodied/wire.py::sweep_orphans`.
        # PRINTS UNCONDITIONALLY, including the zero, and that is the point rather than noise.
        # This sweep spent two days running in one rendezvous direction only, and nothing could
        # have told you: it spoke only when it found something, so "never ran" and "ran, found
        # nothing" produced byte-identical logs.
        orphans = sweep_orphans()
        if orphans:
            print(f"[gateway] reclaimed {len(orphans)} shared segment(s) from dead processes: {orphans}", flush=True)
        else:
            print("[gateway] orphan sweep: 0 shared segment(s) reclaimed (armed check, not silence)", flush=True)

        if self._rendezvous == "listen":
            self._listen()
        else:
            # Nothing to bind, but the path still has to be resolved and length-checked before a
            # child is told to bind it -- the 107 B sun_path failure is a bare OSError from the
            # CHILD otherwise, naming neither the limit nor the path.
            self._sock_path = self._socket_path()
        if not self._attach:
            self._spawn()
        try:
            self._rendezvous_and_handshake()
        except BaseException:
            # A CONSTRUCTOR THAT RAISES LEAVES NOBODY TO CALL `close()`. `_start` runs from
            # `__init__`, so every failure below this line escapes with no object bound to any
            # name -- and the child is already spawned, holding this cell's card. Ray reports an
            # actor that failed to construct; the simulator and Isaac's helpers keep the GPU.
            # That is the orphan §9.9 check 6 exists to catch, produced by us.
            #
            # HERE rather than at a third kill site: the listen path already kills on
            # `socket.timeout` and `_finish_handshake` on `WireTimeout`, but neither covers the
            # connect branch's `connect_with_retry`, nor a handshake that fails for any OTHER
            # reason (a wire-version mismatch, the identity fence, a malformed frame).
            #
            # `BaseException`, not `Exception`: a KeyboardInterrupt or a Ray cancellation
            # between spawn and handshake orphans the child exactly as effectively.
            # `_kill_child` is idempotent -- it re-checks the group -- so the paths that already
            # kill are not harmed by killing again, and it is a no-op in attach mode where the
            # simulator is not ours to reap.
            self._kill_child()
            raise

    def _listen(self) -> None:
        """Bind the socket the child connects back on."""
        self._sock_path = self._socket_path()
        if os.path.exists(self._sock_path):
            os.unlink(self._sock_path)
        self._listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._listener.bind(self._sock_path)
        self._listener.listen(1)
        self._listener.settimeout(self._start_timeout_s)

    def _spawn(self) -> None:
        """Start one simulator incarnation against the already-bound socket."""
        argv = [self._sim_python, "-u", "-m", self._sim_module, "--socket", self._sock_path]
        if self._rendezvous == "connect":
            # UNLINK BEFORE SPAWNING, and this is not tidiness. In connect mode the path is
            # CELL-scoped and therefore identical across runs on this pod, so a simulator
            # orphaned by a previous run can still be listening on it. Our `connect_with_retry`
            # would reach that orphan -- successfully -- and handshake with it: a fresh gateway
            # has no previous identity, so the fence has nothing to compare and waves it
            # through. The run then feeds a simulator holding another run's scene, on a card
            # this cell does not own, and nothing anywhere reports a problem.
            #
            # Unlinking first makes the NAME unreachable while leaving the orphan's existing
            # connections intact; the child then binds a fresh inode under it.
            # PRINTS UNCONDITIONALLY, including the zero, for the reason the orphan sweep in
            # `_start` prints its zero. RZ-0 read this check as OBSERVED rather than PASS on
            # exactly that ground -- a fresh pod has no stale socket, so silence was the only
            # outcome the run could produce and it confirmed nothing.
            if os.path.exists(self._sock_path):
                print(f"[gateway] stale rendezvous check: removing {self._sock_path!r}", flush=True)
                os.unlink(self._sock_path)
            else:
                print(
                    f"[gateway] stale rendezvous check: nothing at {self._sock_path!r} (armed check, not silence)",
                    flush=True,
                )
            # The SIMULATOR binds. Spawning in connect mode is not a contradiction: it is the
            # one-raylet shape where the simulator owns the socket but this process still starts
            # it, and it is how connect mode is testable without a second launcher.
            #
            argv.append("--listen")
        self._proc = subprocess.Popen(
            argv,
            env=self._child_env(),
            cwd=os.environ.get("IMAGINAIRE_ROOT", os.getcwd()),
            # Its own process group, so teardown can signal the whole tree. Isaac spawns
            # children of its own, and terminating only the child it was handed leaves those
            # holding the card -- which the next cell to draw this node sees as a GPU that is
            # busy for no visible reason.
            start_new_session=True,
        )

    def _rendezvous_and_handshake(self, *, expect_restart: bool = True) -> None:
        """Obtain one connection -- by accepting or by connecting -- and handshake on it.

        Idempotent with respect to the LISTENER, which is what makes a second call meaningful:
        it neither rebinds nor respawns, so a caller that has just watched a child die can call
        this again to pick up the replacement.

        A second call drops the previous socket AND the previous shared segment before taking a
        new one. The segment matters more than the socket: ``_handshake`` calls
        ``Segment.create`` every time, so a reconnect that only replaced the socket would leak
        one segment per restart -- and an unlinked segment held by a live process shows as 0 in
        ``ls /dev/shm`` while four processes hold it, which is the residue the teardown probe
        was built to find and would be found here as our own.
        """
        for stale in (self._sock, self._seg):
            if stale is not None:
                try:
                    stale.close()
                except OSError:
                    pass
        self._sock = None
        self._seg = None
        if self._rendezvous == "connect":
            # WireTimeout out of here already names the path and the last errno. Do not add a
            # "no child stdout to point at" caveat: `connect` is about WHO BINDS, `attach` is
            # about WHOSE CHILD IT IS. In connect+spawn, the production shape, the simulator IS
            # ours and its stdout IS this process's. `_start` reaps it on any failure out of
            # here; see the handler there.
            self._sock = connect_with_retry(self._sock_path, timeout_s=self._start_timeout_s)
            self._finish_handshake(expect_restart=expect_restart)
            return
        try:
            self._sock, _ = self._listener.accept()
        except socket.timeout as exc:
            self._kill_child()
            # Two modes, two different things to go and look at, and the spawn-mode message
            # would send an attach-mode operator to a stdout that has nothing in it: in attach
            # mode the simulator is not our child, so its output is somewhere else entirely.
            if self._attach:
                raise WireError(
                    f"no simulator connected to {self._sock_path!r} within "
                    f"{self._start_timeout_s:.0f}s. In attach mode this gateway spawns nothing "
                    f"and waits, so the simulator container is either not running, not sharing "
                    f"this IPC directory, or was told a different socket path."
                ) from exc
            raise WireError(
                f"the simulator subprocess did not connect within {self._start_timeout_s:.0f}s "
                f"(python={self._sim_python!r}, socket={self._sock_path!r}). Its stdout is this "
                f"process's stdout; look there for an import or venv failure."
            ) from exc

        # The deadline has to be re-applied to the ACCEPTED socket, and it is not inherited:
        # CPython forces a socket returned by `accept()` to blocking regardless of the
        # listener's timeout. Without this, `start_timeout_s` bounds only the connect -- and the
        # child connects BEFORE it does any work, so a stall during import, `CollectorBody`
        # construction or `capabilities()` hung the gateway with nothing bounding it.
        #
        # Scoped to the HANDSHAKE and cleared before the service loop. A blanket timeout would
        # be wrong there: the child is silent for the whole scene build (~5.5 min) and a unit may
        # legitimately run for `unit_deadline_s`, which the driver already enforces. A socket
        # timeout over that would be a second, shorter, less informative deadline.
        self._finish_handshake(expect_restart=expect_restart)

    def _finish_handshake(self, *, expect_restart: bool) -> None:
        """The deadline-scoped handshake, shared by both rendezvous directions."""
        self._sock.settimeout(self._start_timeout_s)
        try:
            self._handshake(expect_restart=expect_restart)
        except WireTimeout as exc:
            self._kill_child()
            raise WireError(
                f"the simulator connected but did not finish the handshake within "
                f"{self._start_timeout_s:.0f}s. It is running but stuck before it could report "
                f"capabilities -- an import, the body constructor, or a first CUDA call. Its "
                f"stdout is this process's stdout."
            ) from exc
        finally:
            self._sock.settimeout(None)

    def reattach(self) -> tuple[str, int]:
        """Get back to a simulator over the same rendezvous. Returns its identity.

        In ``listen`` mode that means accepting the next connection on the already-bound
        listener; in ``connect`` mode it means dialling the simulator's socket again, waiting
        for it to come back if a supervisor is in the middle of rebinding it.

        **Policy now, and only one.** :meth:`_run` calls this on ``PeerGone``, for
        ``run_from_source`` alone and only up to ``restart_budget`` (default 0). Deciding WHEN
        to reconnect is deciding what happens to the units that were in flight, and those
        belong to the claim plane (``campaign/dirlease.py``), not to the transport -- see
        :data:`_RESUMABLE_METHODS` for why only ``run_from_source`` may be re-issued.

        Mode decides who produces the replacement, and it decides ``expect_restart`` with it:

        * **spawn mode** -- this gateway owns the child, so it reaps the old process group and
          starts a new one. A restart definitely happened, so the strict rule applies.
        * **attach mode** -- the simulator is not ours. Something else (a supervisor inside the
          sim container, or the platform restarting that container) has already produced the
          peer that is about to connect, and we cannot assert that any restart occurred. The
          relaxed rule applies, which is the whole reason it exists.
        """
        if self._closed:
            raise WireError("reattach() after close(); this gateway is spent.")
        if self._rendezvous == "listen" and self._listener is None:
            raise WireError("reattach() before _listen(); there is no socket to accept on.")
        if not self._attach:
            # Reap FIRST. `_spawn` overwrites `self._proc`, so a respawn that skipped this would
            # drop the only handle to the previous process group -- and that group is where
            # Isaac's children are, holding the card the new one is about to ask for.
            self._kill_child()
            self._spawn()
        self._rendezvous_and_handshake(expect_restart=not self._attach)
        assert self._identity is not None  # set by a successful handshake
        return self._identity

    def _handshake(self, *, expect_restart: bool = True) -> None:
        """Five frames, under the deadline its caller sets. See :meth:`_start`.

        ``expect_restart`` is handed straight to the identity fence and means what it means
        there: ``True`` for a handshake that follows a spawn, ``False`` for a reattach to a
        simulator that was already running. See ``embodied/wire.py::refuse_identity_replay``.
        """
        hello = self._expect("hello")
        if int(hello.get("wire_version", -1)) != WIRE_VERSION:
            self._kill_child()
            raise WireError(
                f"simulator speaks wire version {hello.get('wire_version')}, this process speaks "
                f"{WIRE_VERSION}. The two halves are from different builds."
            )
        # Body first, segment second: the segment's size is a property of the embodiment, and the
        # embodiment is a property of the body this gateway asked the child to build.
        # Identity BEFORE `configure`: if this is a stranger or a rewound generation, we should
        # not have asked it to build a body first. Missing fields are refused rather than
        # defaulted -- a `.get(..., 0)` here would turn a malformed hello into a perfectly
        # plausible generation 0, i.e. into "first start", which is the one value that disables
        # the fence.
        sid = str(hello.get("session_id", "")).strip()
        if not sid:
            self._kill_child()
            raise WireError("simulator's hello carries no session_id; wire >= 5 requires one.")
        if "generation" not in hello:
            self._kill_child()
            raise WireError(f"simulator session {sid!r} sent no generation; wire >= 5 requires one.")
        try:
            gen = int(hello["generation"])
        except (TypeError, ValueError) as exc:
            self._kill_child()
            raise WireError(f"simulator session {sid!r} sent generation {hello['generation']!r}, not an int.") from exc
        if "visible_devices" not in hello:
            self._kill_child()
            raise WireError(
                f"simulator session {sid!r} sent no visible_devices; wire >= {WIRE_VERSION} requires "
                f"it. Absent is refused rather than defaulted to this gateway's intent, because "
                f"the intent is precisely the reading it exists to replace."
            )
        reported = str(hello["visible_devices"]).strip()
        # THE COMPARISON `sim_role.py` could never make. When this gateway spawned the child it
        # SET the variable, so intent and reading must agree -- and if they do not, the
        # environment did not survive the spawn, which is the exact hole S-04's docstring
        # records as invisible to it. Scoped to spawn mode: in attach mode there is no intent to
        # disagree with, and the child's reading is simply the answer.
        if not self._attach and self._sim_gpu is not None and reported != str(self._sim_gpu).strip():
            self._kill_child()
            raise WireError(
                f"simulator session {sid!r} was spawned with CUDA_VISIBLE_DEVICES="
                f"{self._sim_gpu!r} and reports {reported!r}. The environment did not survive "
                f"the spawn, so this cell's simulator is on a card nobody assigned it -- which "
                f"produces correct episodes whose timings no longer describe the same-card pair "
                f"every isolation-cost number assumes."
            )
        self._child_devices = reported
        if "pgid" not in hello:
            self._kill_child()
            raise WireError(
                f"simulator session {sid!r} sent no pgid; wire >= {WIRE_VERSION} requires it. It "
                f"is what teardown signals, and defaulting it to this gateway's child would be "
                f"correct only in the arrangement that does not need it."
            )
        try:
            self._child_pgid = int(hello["pgid"])
        except (TypeError, ValueError) as exc:
            self._kill_child()
            raise WireError(f"simulator session {sid!r} sent pgid {hello['pgid']!r}, not an int.") from exc
        try:
            refuse_identity_replay(self._identity, (sid, gen), where="gateway handshake", expect_restart=expect_restart)
        except WireError:
            self._kill_child()
            raise
        self._identity = (sid, gen)

        send_frame(self._sock, "configure", {"body": self._child_cfg})
        self._caps = dict(self._expect("capabilities")["capabilities"])
        caps = capabilities_from(self._caps)
        capacity = segment_bytes(caps.camera_shapes, caps.proprio_width)
        self._seg = Segment.create(capacity)
        send_frame(self._sock, "attach", {"segment": self._seg.name, "obs_capacity": capacity})
        self._expect("ready")
        # THE START RECEIPT, and it is the ONLY artifact SPEC §9.9 criteria 3 and 4 can be
        # judged from. Both criteria are about values that exist solely inside this process:
        # criterion 3 is "the child resolved the SIM venv's interpreter, not the framework's",
        # criterion 4 is "the child holds the card this cell was given". Until this line the
        # resolved interpreter reached stdout only inside the start-TIMEOUT error path, so a
        # SUCCESSFUL start printed neither -- and a criterion with no artifact is one that gets
        # marked satisfied because nothing contradicted it.
        #
        # Printed after `ready` rather than after `Popen`: before the handshake completes these
        # are what we ASKED for, and the interesting failure is exactly when the child resolves
        # something else. After `ready` the child has imported, built the body and mapped the
        # segment with these values.
        # `rendezvous` and `socket` are on this line for the reason the transport gate had to
        # learn: an arm's MODE was readable only as an ABSENCE. Nothing else in a successful
        # run names the direction -- `[sim] listening` is printed by the simulator and only in
        # connect mode -- so a `listen` arm and a `connect` arm that silently fell back to
        # listen produce the SAME log, and a differential between them would report equivalence
        # having run one mode twice.
        print(
            f"[gateway] child READY pid={self._proc.pid if self._proc else '?'} "
            f"session={self._identity[0] if self._identity else '?'}"
            f"/gen{self._identity[1] if self._identity else '?'} "
            f"python={self._sim_python!r} CUDA_VISIBLE_DEVICES={self._sim_gpu!r} "
            f"backend={self._caps.get('backend')!r} segment={capacity}B "
            f"rendezvous={self._rendezvous!r} socket={self._sock_path!r}",
            flush=True,
        )
        # `ready` means the child has mapped it, so the NAME has no readers left and the mapping
        # is what both sides use from here. Dropping it now makes the kernel reclaim the memory
        # whenever the last mapping goes -- including when this actor is killed without ever
        # running `close()`, which is what a normal driver exit does.
        self._seg.unlink_now()

    def _kill_child(self) -> None:
        """Signal the child's process group until the GROUP is gone, escalating once.

        ``killpg`` rather than ``proc.terminate()`` because of ``start_new_session=True``: the
        child leads its own group, and Isaac's helper processes are in it. A ``terminate`` that
        reaped only the direct child would leave them holding the GPU.

        For the same reason both the liveness question and the stop condition are asked of the
        group -- :func:`embodied.wire.group_alive` on the pgid the child's ``hello`` reported --
        and never of the leader. A leader that has already exited is the ordinary case on every
        path that calls this, so a guard on the leader skips teardown exactly when there is
        something left to tear down. See the comments in the body for the specific paths.

        Skipped entirely in attach mode, and the reason is sharper than "not ours to reap".
        In attach mode the simulator's group is whatever the PLATFORM gave it, and we have no
        claim on the rest of it. Deleting that guard is not a leak, it is a signal fired into a
        group this process does not own: measured on the test suite, the mutant killed the
        pytest runner itself, because the attach fixture's simulator shares the runner's process
        group. On a pod the neighbour in that group is the Ray worker.
        """
        if self._attach:
            return
        # The GROUP is the unit of teardown, so the group is what liveness is asked about.
        # Keying on the LEADER (`self._proc.poll() is None`) inverted the method on exactly the
        # paths it exists for: a leader that has already exited is the normal case here, not the
        # exceptional one -- the planned restart (`collector.py`) and the orphan watchdog
        # (`wire.py`) both leave via `os._exit`, and a native `ERROR_DEVICE_LOST` or a segfault
        # takes the faulting process alone. In every one of those the helpers Isaac forked into
        # the session are still holding the card, and the old guard returned before signalling.
        #
        # `_child_pgid` is what the child's `hello` reported; `self._proc.pid` is the fallback
        # for a child that died before handshaking (under `start_new_session=True` the leader's
        # pid IS the pgid, so the two agree whenever both exist).
        pgid = self._child_pgid or (self._proc.pid if self._proc is not None else None)
        if pgid is None:
            return
        for sig, grace in ((signal.SIGTERM, 30), (signal.SIGKILL, 30)):
            # Re-asked before EVERY signal, which is also what makes this idempotent: a second
            # call after a successful teardown finds an empty group and returns without
            # signalling. Checking first also closes the pid-reuse hole the old `poll()` guard
            # was reaching for -- a pgid is not recycled while any member still holds it, so a
            # group that answers here is this child's group, not a stranger's.
            if not group_alive(pgid):
                break
            try:
                os.killpg(pgid, sig)
            except ProcessLookupError:
                break
            except PermissionError:
                # Not ours to signal, and no escalation changes that. Waiting out the grace
                # would stall teardown for a minute to reach the same place.
                break
            # Escalate on the GROUP draining, not on the leader being reaped. The old code
            # `break`-ed as soon as `self._proc.wait()` returned, so SIGKILL was never reached
            # even when a helper had ignored the SIGTERM.
            deadline = time.monotonic() + grace
            while time.monotonic() < deadline and self._group_pending(pgid):
                time.sleep(0.2)
        self._reap_leader()

    def _group_pending(self, pgid: int) -> bool:
        """Is anything OTHER than a zombie we still owe a ``wait`` left in the group?

        The reap is inside the liveness question because a leader we signalled becomes a zombie
        that is still a member of its own group, so a bare :func:`group_alive` answers True
        forever and every clean teardown burns both grace periods before returning. Measured:
        60.08s for the leader-alive case, against 0.25s once the leader is reaped here.
        """
        self._reap_leader(timeout=0)
        return group_alive(pgid)

    def _reap_leader(self, timeout: float = 5.0) -> None:
        """Collect the leader if it has exited, so its pid is released rather than held.

        ``poll`` for the zero-timeout probe and ``wait`` for the final call: ``poll`` reaps a
        process that has already exited and returns immediately for one that has not, which is
        what a liveness loop needs. Never fatal -- a child that outlived SIGKILL is not
        something one more ``wait`` will fix.
        """
        if self._proc is None:
            return
        if timeout <= 0:
            self._proc.poll()
            return
        try:
            self._proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            pass

    def close(self) -> None:
        """Ask the simulator to shut down, then make sure it did. Safe after a partial run.

        **The close frame is NOT gated on owning the process.** It used to be -- the condition
        read ``self._proc is not None and self._proc.poll() is None`` -- which was equivalent
        while every simulator was this gateway's child. In attach mode ``_proc`` is ``None`` for
        a simulator that is alive and connected, so that condition skipped the frame and the
        peer was never told anything: it would sit in ``recv`` until the socket closed under it.
        The two questions are separate now, so they are asked separately.

        **``close`` is shutdown, not detach.** It tells the simulator to exit. That is right
        for phase 1, where the whole allocation ends together and the
        card should be released. A "the Ray side is leaving, stay up for the next attachment"
        operation is a DIFFERENT frame and does not exist; do not read this one as it.

        In attach mode the simulator is not ours to reap, so the 120 s confirmation wait is
        skipped -- there is no process handle to wait on and no authority to signal one. The
        honest scope is "asked, and the socket is closed"; the platform owns the rest.
        """
        if self._closed:
            return
        self._closed = True
        try:
            if self._sock is not None and (self._proc is None or self._proc.poll() is None):
                try:
                    send_frame(self._sock, "close")
                    if self._proc is not None:
                        self._proc.wait(timeout=120)
                except (WireError, subprocess.TimeoutExpired):
                    pass
        finally:
            self._kill_child()
            for closeable in (self._sock, self._listener):
                if closeable is not None:
                    closeable.close()
            if self._seg is not None:
                self._seg.close()

    # ── the SimRole surface ─────────────────────────────────────────────────

    def probe(self) -> dict:
        """Forwarded, with the capabilities taken from the startup handshake.

        The handshake already carries them, so this does not pay a round trip to answer a
        question that was answered at spawn. ``host`` and ``gpu`` come from the CHILD, not from
        the gateway: they describe where the simulator is, and under a remote transport the two
        would genuinely differ.
        """
        self._require_live()
        send_frame(self._sock, "probe")
        reply = self._expect("probe_reply")
        sid, gen = self._identity if self._identity else ("", -1)
        return {
            "host": reply["host"],
            "gpu": reply["gpu"],
            "capabilities": self._caps,
            # From the HANDSHAKE, not from this round trip. Reported so the framework side can
            # see which simulator incarnation produced a result without asking the child again.
            "session_id": sid,
            "generation": gen,
            # The simulator's own group, from the same handshake. Teardown signals the group of
            # the process this gateway spawned, which is the same tree; this is here so the
            # value the wire requires is readable rather than merely validated.
            "child_pgid": self._child_pgid,
        }

    def child_gpu(self) -> str:
        """The device this gateway HANDED its child, answered without touching the child.

        Exists for one caller: the bind-time S-04 check in
        ``refuse_cross_card_gateway`` below. That check needs the card the
        simulator actually holds, and none of the three obvious sources has it --
        ``WorkerGroup.gpus()`` reads the ACTOR's ``CUDA_VISIBLE_DEVICES``, which is the
        GATEWAY's card, not what the CHILD holds -- and in attach mode this body assigns nothing
        at all, so no actor-side reading can answer. ``probe()`` would answer, but it forwards
        to the child and runs a real card probe. Episode-work-unit binding uses this cheap
        handshake lookup; decision-window capability binding separately calls ``probe()``.
        The driver cannot recompute
        ``REPLICA_INDEX % ranks_per_pod`` itself without becoming the second copy of the
        arithmetic whose correctness is the very thing under test.

        So it answers from the HANDSHAKE: ``hello`` carries the child's own
        ``CUDA_VISIBLE_DEVICES``, read in the child before it builds anything (wire 6).

        **The answer is the CHILD's reading, not this body's INTENT.** Intent cannot see a child
        that ignored the variable, and the second witness -- the card
        ``embodied/the external simulator backend::_first_visible_gpu_uuid`` stamps into every episode row --
        arrives after a scene is built, while this value is consumed at BIND. In attach mode the
        child's reading is the ONLY one that exists: there this gateway spawns nothing, assigns
        nothing, and therefore intends nothing.

        In spawn mode the two must agree -- :meth:`_handshake` refuses the bind if they do not --
        so which one is returned here cannot change a PASSING result. It changes the failing
        ones.

        Falls back to intent only before a handshake has completed. No caller is in that window
        today, and returning nothing there would look like an answer.
        """
        if self._child_devices is not None:
            return self._child_devices
        return self._sim_gpu or ""

    def run_units(self, units: Sequence[Any], *, policy: Any = None, **kw: Any) -> dict:
        return self._run("run_units", units, policy=policy, **kw)

    def run_window(self, units: Sequence[Any] = (), *, policy: Any, version: int, decisions: int) -> dict:
        if units:
            raise ValueError("window collection does not accept episode work units")
        return self._run("run_window", units, policy=policy, version=version, decisions=decisions)

    def run_evaluation(
        self,
        units: Sequence[Any] = (),
        *,
        policy: Any,
        cases: list[dict[str, int | str]],
        version: int,
        seed: int,
        decisions: int,
    ) -> dict:
        if units:
            raise ValueError("evaluation does not accept episode work units")
        return self._run(
            "run_evaluation", units, policy=policy, cases=cases, version=version, seed=seed, decisions=decisions
        )

    def run_from_source(self, units: Sequence[Any], *, policy: Any = None, **kw: Any) -> dict:
        return self._run("run_from_source", units, policy=policy, **kw)

    # ── the forwarding loop ─────────────────────────────────────────────────

    def _run(self, method: str, units: Sequence[Any], *, policy: Any, **kw: Any) -> dict:
        """Forward one call, re-issuing it against a fresh simulator if the peer disappears.

        Retries are gated on THREE things, and each one is load-bearing:

        1. **``PeerGone``, not ``WireError``.** Only a clean EOF says the simulator is not there
           any more. A peer that sends an ``error`` frame is alive and reporting about the work,
           and killing it to escape a unit-level fault would trade one bad unit for a scene
           rebuild -- roughly 5.5 minutes -- and another generation off that unit.
        2. **A resumable method.** See :data:`_RESUMABLE_METHODS`. ``run_units`` is refused
           because re-issuing it would re-run finished units.
        3. **``restart_budget``**, default 0. Nothing retries until something asks.

        **``wall_s`` is charged for the time the dead attempt spent.** The child restarts its own
        clock on every call, so re-issuing the original value would hand the pod a second full
        wall budget -- and ``wall_s`` is what stops a cell claiming a unit it cannot finish
        before the job is killed. The remaining wall is passed instead, and an exhausted one
        stops the retry rather than starting a run that has no time to do anything.
        """
        progress = kw.pop("progress", None)
        budget = self._restart_budget if method in _RESUMABLE_METHODS else 0
        wall_s = kw.get("wall_s")
        started = time.monotonic()
        reattaches = 0
        while True:
            try:
                out = self._run_once(method, units, policy=policy, progress=progress, kw=kw)
            except PeerGone as exc:
                if reattaches >= budget:
                    raise
                if wall_s:
                    left = float(wall_s) - (time.monotonic() - started)
                    if left <= 0:
                        raise
                    kw["wall_s"] = left
                reattaches += 1
                print(
                    f"[gateway] {method}: {exc} -- reattaching ({reattaches}/{budget} of the restart budget)",
                    flush=True,
                )
                self.reattach()
                continue
            if reattaches:
                # Stamped so nobody reads this cell's counters as a complete tally. They are not:
                # they cover only the LAST attempt, because the earlier ones died without
                # returning. The corpus on disk and the claim plane's `.done` records are the
                # truth for what actually ran; these stats are one cell's view of one attempt.
                out = dict(out)
                out["gateway_reattaches"] = reattaches
            return out

    def _run_once(self, method: str, units: Sequence[Any], *, policy: Any, progress: Any, kw: dict) -> dict:
        """One attempt. Everything that decides whether there is another one is in :meth:`_run`."""
        self._require_live()
        try:
            send_frame(self._sock, "run", {"method": method, "units": _units_as_wire(units), "kwargs": kw})
        except WireError as exc:
            # A broken pipe here means the child died between the liveness check and the send --
            # and it may well have said WHY first. Reporting the transport error would discard
            # the child's own message in favour of "[Errno 32] Broken pipe", on exactly the path
            # where the cause is otherwise unrecoverable.
            raise self._child_failure(f"sending {method!r}") from exc
        before = self._seg.verified if self._seg is not None else 0
        out = self._service(policy, progress)
        # THE ARM RECEIPT for the crc32 crossing check, emitted once per forwarded call. The
        # verifier only speaks on FAILURE, so without this "zero digest mismatches" is a claim
        # no observation could have contradicted -- `B1K_WIRE_VERIFY=0`, a payload that never
        # crossed, or a run that died before its first `act` all look exactly like a clean one.
        # A count of 0 here is a RED flag, not a quiet success.
        if self._seg is not None:
            print(
                f"[gateway] {method} done: crossings_verified={self._seg.verified - before} "
                f"(cumulative {self._seg.verified})",
                flush=True,
            )
        return out

    def _child_exit_code(self) -> int | None:
        """The child's exit status, waited for rather than polled.

        ``poll()`` races the child's exit: the gateway notices the closed socket before the
        kernel has reaped the process, so it reports ``exit code None`` -- a useless diagnostic
        on the one path where the diagnostic is all there is. Measured: the hard-death test read
        ``None`` every time until this waited.
        """
        if self._proc is None:
            return None
        try:
            return self._proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            return None

    def _child_failure(self, context: str) -> WireError:
        """Build the most informative error available about a child that has stopped talking.

        Prefers the child's own ``error`` frame if one is still buffered, because it names the
        Python exception; falls back to the exit code. Never raises from inside itself -- it runs
        while another failure is being reported.
        """
        detail = ""
        # BOUNDED, and the bound is the whole point. This runs while another failure is being
        # reported, and its caller reaches it whenever `send_frame` raised -- including when the
        # frame was refused by the pre-send size cap, in which case the child is perfectly
        # HEALTHY and blocked in its own `recv`. An unbounded read against that child is a
        # two-sided deadlock: each side waiting for a frame the other will never send. Measured
        # relevance: the broadcast `run` frame is 2,102,029 B at 20 tasks x 300 instances,
        # 50.1% of MAX_FRAME_BYTES, so the cap is reached at roughly 600 instances per task.
        try:
            self._sock.settimeout(_FAILURE_PROBE_S)
            pending = recv_frame(self._sock)
            if pending is not None and pending.get("kind") == "error":
                detail = f": {pending.get('error')}"
        except (WireError, OSError):
            pass
        finally:
            try:
                self._sock.settimeout(None)
            except OSError:
                pass
        code = self._child_exit_code()
        return WireError(f"the simulator failed while {context} (exit code {code}){detail}")

    def _service(self, policy: Any, progress: Callable[[dict], None] | None) -> dict:
        """Answer the child's callbacks until it returns a result.

        This is the inversion: the framework asked the child to run, and now the child drives.
        The loop is flat and synchronous on purpose -- there is exactly one outstanding request
        at any time, which is what lets the segment have one writer per region and no locks.
        """
        while True:
            frame = recv_frame(self._sock)
            if frame is None:
                # PeerGone, not WireError: a clean EOF is the one unambiguous "it is not there
                # any more", and it is what `_run` gates its reattach on. An `error` frame --
                # handled below -- comes from a peer that is ALIVE and reporting about the work,
                # and restarting a simulator to escape a unit-level fault would be a much worse
                # trade than the one it looks like.
                raise PeerGone(
                    f"the simulator exited without returning a result (exit code "
                    f"{self._child_exit_code()}). Its leases are not lost -- they age out and "
                    f"another cell claims them -- but this unit produced nothing."
                )
            kind = frame["kind"]
            if kind == "act":
                self._answer_act(frame, policy)
            elif kind == "feedback":
                self._answer_feedback(frame, policy)
            elif kind == "describe":
                try:
                    info = rollout_info_as_wire(policy.describe())
                except Exception as exc:  # noqa: BLE001 -- see _forward_fault
                    self._forward_fault("describe", exc)
                else:
                    send_frame(self._sock, "describe_reply", {"info": info})
            elif kind == "progress":
                if progress is not None:
                    progress(frame["event"])
            elif kind == "result":
                return frame["result"]
            elif kind == "error":
                raise WireError(f"simulator failed: {frame.get('error')}")
            else:
                raise WireError(f"unexpected frame {kind!r} from the simulator")

    def _answer_act(self, frame: Mapping[str, Any], policy: Any) -> None:
        """One policy call, with the observation read as VIEWS onto the shared segment.

        Zero copy on this side is the entire reason the transport was built this way, and it is
        safe here for a specific reason rather than by luck: ``policy.act`` is a ``PeerClient``
        call whose ``ray.get(host.invoke.remote(...))`` serialises its arguments eagerly, so the
        copy is taken inside the call and no view outlives it. A consumer that deferred would
        read whatever the next observation overwrote.
        """
        # BEFORE read_arrays, and OUTSIDE the fault boundary below: a payload whose bytes are
        # wrong is the TRANSPORT failing, not the framework, and relaying it as a unit fault
        # would let a corrupt segment produce `max_consecutive_errors` quiet unit deaths with
        # the cause several units back. This is the one failure the transport can produce that
        # would otherwise reach the policy looking like a real observation.
        from cosmos_framework.posttrain.embodied.wire import read_observations

        batch = read_observations(self._seg, frame)

        # ── the fault boundary ──
        # Inside: the framework call and the contract it must satisfy. A failure here is a UNIT
        # fault -- exactly what `driver.run_unit` catches in `direct` -- so it is relayed rather
        # than raised, and the service loop continues.
        #
        # Outside, deliberately: the segment reads above and the writes below. Those are the
        # TRANSPORT. A seam that cannot move bytes fails every subsequent call identically, so
        # relaying it would turn one loud death into `max_consecutive_errors` quiet ones and then
        # a death anyway, with the cause three units further back.
        #
        # `ValueError`, not `WireError`, for the chunk-count check: `driver.py` raises exactly
        # that for the same mismatch on the direct path, inside the same handler.
        try:
            actions = policy.act(batch, context=frame["context"]) if "context" in frame else policy.act(batch)
            if len(actions) != len(batch):
                raise ValueError(f"policy returned {len(actions)} chunks for {len(batch)} observations")
            out = {str(i): np.ascontiguousarray(a) for i, a in enumerate(actions)}
        except Exception as exc:  # noqa: BLE001 -- see _forward_fault
            self._forward_fault("act", exc)
            return

        reply = self._seg.write_arrays("act", out)
        send_frame(
            self._sock,
            "act_reply",
            {"descs": [d.as_wire() for d in reply], "digest": self._seg.digest("act", span_of(reply))},
        )

    def _answer_feedback(self, frame: Mapping[str, Any], policy: Any) -> None:
        from cosmos_framework.posttrain.embodied.wire import read_observations

        observations = read_observations(self._seg, frame)
        try:
            policy.feedback(observations, context=frame["context"])
        except Exception as exc:  # noqa: BLE001 -- relay the same failure boundary as action callbacks
            self._forward_fault("feedback", exc)
        else:
            send_frame(self._sock, "feedback_reply")

    def _forward_fault(self, where: str, exc: BaseException) -> None:
        """Hand a framework-side exception to the child instead of unwinding around it.

        The child re-raises it inside ``driver.run_unit``'s per-unit ``try``, which records a
        ``UnitResult`` with ``error_class="unclassified"`` -- the same bucket the same fault gets
        in ``direct``, where it arrives as a ``RayTaskError`` -- hands the unit back, and takes
        the next one. ``max_consecutive_errors`` still ends the cell if the policy is genuinely
        dead rather than blipping; that decision stays where it already is.

        The traceback is printed HERE because this process's stdout is the pod log and the child
        only receives the ``repr``. If the send itself fails the transport is gone, so the
        exception propagates and the cell dies -- which is correct, and self-selecting: a fault
        that cannot be delivered cannot be handled.

        Callers catch ``Exception``, not ``BaseException``, deliberately. Every fault this is for
        is one -- ``ray.exceptions.RayError`` and all of its subclasses are ``Exception``
        (measured against ray 2.56.1, and asserted in the tests so the assumption cannot rot).
        A ``KeyboardInterrupt`` or ``SystemExit`` is the operator or the runtime ending this
        process, which is not a unit fault and must not be relayed as one.
        """
        traceback.print_exc()
        send_frame(self._sock, FAULT, fault_payload(where, exc))

    # ── internals ───────────────────────────────────────────────────────────

    def _require_live(self) -> None:
        if self._closed or self._sock is None:
            raise WireError("this gateway is closed")
        if self._proc is not None and self._proc.poll() is not None:
            raise WireError(f"the simulator subprocess is gone (exit code {self._proc.returncode})")

    def _expect(self, kind: str) -> dict:
        frame = recv_frame(self._sock)
        if frame is None:
            raise PeerGone(f"expected {kind!r}, the simulator closed the connection instead")
        if frame["kind"] == "error":
            raise WireError(f"expected {kind!r}, the simulator reported: {frame.get('error')}")
        if frame["kind"] != kind:
            raise WireError(f"expected {kind!r}, got {frame['kind']!r}")
        return frame


# ── S-04: the same-card refusal, and the builder this body answers to ────────
#
# HERE, not beside the `SimRole` contract in `sim_role.py`: a placement policy is not a
# contract. The framework seam says nothing about cards, and this refusal is a property of ONE
# implementation -- a gateway whose simulator process and policy actor can land on different
# cards, which an in-process `CollectorBody` cannot do.


class CardPinningError(RuntimeError):
    """S-04 violated: a gateway's simulator holds a card its policy actor does not."""


def _one_device(raw: Any) -> str | None:
    """Normalise a ``CUDA_VISIBLE_DEVICES`` reading, or ``None`` when it says nothing.

    ``workers/ray.py::gpu`` returns the literal ``"?"`` when the variable is unset, and a
    0-GPU actor gets ``""``. Both are *absence of an answer*, not an answer of "no card" --
    and folding them into a comparison would let a pair of unreadable values compare equal
    and report a pass. They are mapped to ``None`` so the caller has to handle them as a
    third outcome.
    """
    if raw is None:
        return None
    text = str(raw).strip()
    if not text or text == "?":
        return None
    return ",".join(part.strip() for part in text.split(",") if part.strip()) or None


def _cells_of(handle: Any) -> dict:
    """``{(r, k): value}`` from whichever Handle flavour the bound backend returned.

    ``GridBackend.submit`` -- the eval path -- returns a ``CollectiveHandle`` and ``.all()`` is
    the right call. A WorkerGroup with NO backend bound returns a plain ``Handle`` wrapping the
    single in-process body's return value, and ``Handle`` has no ``.all()``: reaching for it
    there is an ``AttributeError`` raised from inside a placement check, which is a confusing
    way to fail and would take the bind down for a reason that has nothing to do with cards.

    Both shapes are handled because only one of them is reachable from a test.
    """
    if hasattr(handle, "all"):
        return handle.all()
    return {(0, 0): handle.result()}


def refuse_cross_card_gateway(policy_wg: Any, sim_wg: Any, n_ranks: int, *, tag: str) -> dict:
    """S-04, checked at bind: each gateway's child must hold ITS policy actor's card.

    The owner's ruling is "do not put it on a different card, or a different machine, from the
    sim" -- the gateway and the simulator share one card so the observation crosses through
    shared memory rather than a network. ``SPEC_sim_role_decouple.md`` §9.9 criterion 4 has
    carried it as **EXPRESSIBLE
    AND UNVERIFIED** since 2026-08-28 for a specific reason: at ``ranks_per_pod=1`` the
    gateway's ``sim_gpu = REPLICA_INDEX % ranks_per_pod`` is always ``"0"`` and the policy
    actor's only card is index 0, so the two agree **by construction, not by measurement**.
    Every gateway slice run to date -- tier 0, the 20 of tier 1, the 12 of tier 2 -- is that
    shape, and so none of them is evidence.

    **What could actually be wrong.** The gateway computes its child's device from the
    replica index; Ray computes the policy actor's device from its own scheduler. Two
    independent derivations of one quantity, never compared. At ``ranks_per_pod >= 2`` Ray is
    free to hand replica 0's policy actor GPU 1, and then cell 0's simulator runs on GPU 0
    while the policy it feeds runs on GPU 1 -- still one host, so **shmem still works and the
    run still produces correct episodes**. Nothing fails. What breaks is the argument: §2.2's
    isolation cost, and the leader's decision that rests on it, are measurements of a
    same-card pair.

    Both readings live in one namespace, which is what makes them comparable. The gateway
    branch of ``utils/launch_multinode_eval.sh`` starts ONE raylet with
    ``CUDA_VISIBLE_DEVICES=0..R-1`` and ``--num-gpus=R``, so a Ray-assigned actor device and
    a pod-local child device index both name the same physical card.

    Raises:
        CardPinningError: naming both devices, on the first definite mismatch.

    Returns:
        ``{"same": int, "cross": int, "unverifiable": int}`` -- returned rather than only
        printed so a test can assert on the census instead of on captured stdout.

    A receipt is printed for **every** cell, including the ones this cannot resolve: a check
    that prints only its failures is indistinguishable, in a log, from one that never ran.
    """
    policy_cards = policy_wg.gpus() or {}
    child_cards = _cells_of(sim_wg.child_gpu())

    census = {"same": 0, "cross": 0, "unverifiable": 0}
    crossed: list[str] = []
    for r in range(n_ranks):
        pol = _one_device(policy_cards.get((r, 0)))
        sim = _one_device(child_cards.get((r, 0)))
        if pol is None or sim is None:
            status, key = "UNVERIFIABLE", "unverifiable"
        elif pol == sim:
            status, key = "SAME-CARD", "same"
        else:
            status, key = "CROSS-CARD", "cross"
            crossed.append(f"cell {r}: policy on {pol}, sim child on {sim}")
        census[key] += 1
        print(
            f"[{tag}]   card {r}: policy CUDA_VISIBLE_DEVICES={pol if pol is not None else '<unreadable>'} "
            f"sim child={sim if sim is not None else '<unreadable>'} -> {status}",
            flush=True,
        )
    print(
        f"[{tag}] S-04 same-card: {census['same']}/{n_ranks} "
        f"(cross-card {census['cross']}, unverifiable {census['unverifiable']})",
        flush=True,
    )

    if crossed:
        raise CardPinningError(
            "S-04 violated: the gateway put a simulator on a card its policy actor does not "
            "hold -- " + "; ".join(crossed) + ". This does not break the run, which is why it "
            "is refused here: it would produce correct episodes whose timings silently no "
            "longer describe the same-card pair every isolation-cost number assumes."
        )
    return census


@register_env(SIM_IMPL_KEY)
def build_sim_gateway_body(config: Any) -> SimGatewayBody:
    """Construct this body from a kwargs mapping, under the manifest's own ``impl_key``.

    ONE NAME, THREE REGISTRIES. ``SIM_IMPL_KEY`` already resolves a ``CapabilityDescriptor``
    in ``control/capabilities_builtin.py`` and a ``RoleDecl`` in
    ``control/recipe.py::sim_manifest``. Until this decorator, it resolved nothing in
    ``roles/core/registry.py``, so the framework could validate a contract for a role whose
    body it had no way to build -- and a near-miss spelling between the three would be found
    by a campaign rather than by ``bind_check``.

    NO CALLER YET, AND SAID PLAINLY. Nothing in this tree calls ``build_env``; the entrypoint
    still constructs the body directly, exactly as ``sim_server.py`` does
    ``CollectorBody(**body_cfg)``. What this buys today is that the three registries agree,
    which ``tests/test_control_sim_manifest_keys.py`` asserts. The mapping is
    kwargs-shaped rather than a typed config object for the same reason: that is how the
    bodies are actually built here, and a builder that took a shape no caller has would be
    the declaration-without-a-referent this exists to remove.
    """
    return SimGatewayBody(**dict(config))


class _ZeroPolicy:
    """The smallest thing the forwarding loop will service: ``describe`` plus ``act``.

    Those two names are the WHOLE policy handle the simulator reaches back for -- measured, not
    assumed, as the only two touched on that handle anywhere in ``embodied/``. A real run puts a
    Ray ``ActorHandle`` here and this process calls it; the demo below puts this, so the second
    half of the two-process example needs no checkpoint, no GPU and no Ray.
    """

    def __init__(self, action_dim: int, horizon: int = 1, embodiment: str = "example") -> None:
        self._dim, self._horizon, self._emb = action_dim, horizon, embodiment

    def act(self, obs_batch: Sequence[Any]) -> list:
        return [np.zeros((self._horizon, self._dim), dtype=np.float32) for _ in obs_batch]

    def describe(self) -> Any:
        from cosmos_framework.posttrain.embodied.rollout import RolloutInfo

        return RolloutInfo(name="zero", action_dim=self._dim, horizon=self._horizon, embodiment=self._emb)


def main(argv: Sequence[str] | None = None) -> int:
    """The FRAMEWORK half of the two-process example: connect to a simulator, run one unit.

    Deliberately NOT an entrypoint, and the counterpart of ``embodied/sims/example.py::_main``
    one layer up. That one proves a backend against the real loop in ONE process; this one
    proves the same backend across the container split, with the simulator in its own
    interpreter -- which is the only thing the gateway adds and therefore the only thing worth
    a second command.

    Fixed to ``attach`` + ``rendezvous='connect'``: the simulator owns the socket
    (``sim_server --listen``) and this process dials it. That is the sim-primary shape, it is
    the half a simulation team can run without our venv, and it is the mode in which neither
    ``sim_python`` nor ``sim_gpu`` means anything here -- see
    :meth:`SimGatewayBody._refuse_incomplete_production_config`.
    """
    import argparse

    from cosmos_framework.posttrain.embodied.unit import EpisodeKey, WorkUnit

    ap = argparse.ArgumentParser(description=main.__doc__.split("\n")[0])
    ap.add_argument("--socket", required=True, help="the path the simulator bound with --listen")
    ap.add_argument("--sim", default="example", help="backend registry name, resolved in the CHILD")
    ap.add_argument("--sink", default="null")
    ap.add_argument("--embodiment", default="example")
    ap.add_argument("--task", default="demo")
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument(
        "--preload",
        action="append",
        default=[],
        help="module the CHILD imports before building; repeatable. Passed through, never imported here",
    )
    ap.add_argument("--action-dim", type=int, default=None, help="default: read from the embodiment if registered")
    ap.add_argument("--start-timeout-s", type=float, default=DEFAULT_START_TIMEOUT_S)
    args = ap.parse_args(argv)

    # The embodiment is resolved in the CHILD; this side needs only the action width, to shape
    # the zero action. Looked up when the map happens to be registered here too, and otherwise
    # asked for -- a framework process has no business importing a simulator's backend to learn
    # a number, which is the same reason `--preload` is passed through rather than imported.
    action_dim = args.action_dim
    if action_dim is None:
        from cosmos_framework.posttrain.embodied.embodiment import get_embodiment

        try:
            action_dim = get_embodiment(args.embodiment).action_dim
        except Exception as exc:
            print(f"[gwdemo] embodiment {args.embodiment!r} is not registered here ({exc}); pass --action-dim")
            return 2

    body = SimGatewayBody(
        sim=args.sim,
        sink=args.sink,
        embodiment=args.embodiment,
        force_render=False,
        attach=True,
        rendezvous="connect",
        sim_socket_path=args.socket,
        preload=args.preload,
        start_timeout_s=args.start_timeout_s,
    )
    try:
        # `probe()`, not a capabilities call: the handshake already carried the child's
        # declaration, so this reports what the gateway HOLDS rather than asking again.
        print(f"[gwdemo] attached: {body.probe()}")
        unit = WorkUnit.make(
            suite=args.sim,
            # num_envs=1, PINNED and not a flag. Round 4 measured scene tiling corrupting
            # physical particle systems -- V=2 scored 0/316 where V=1 scored 5/40 -- and the
            # standing ruling is that V>1 is refused rather than defaulted against. b1k
            # declares `max_lanes=1`, so `validate_sim_role` refuses a run that asks for more;
            # exposing a `--lanes` knob here would only let a reader discover that by hitting
            # it. A backend whose engine genuinely tiles raises its own ceiling first.
            scenario={"task": args.task, "max_steps": args.max_steps, "num_envs": 1},
            episodes=tuple(
                EpisodeKey(suite=args.sim, task=args.task, instance=i, seed=i) for i in range(args.episodes)
            ),
            est_minutes=0.1,
        )
        out = body.run_units([unit], policy=_ZeroPolicy(action_dim, embodiment=args.embodiment))
    finally:
        # The simulator is NOT ours to reap in attach mode -- `close()` drops the connection and
        # leaves it listening, which is what lets the next `--listen`-side run reuse it.
        body.close()

    for res in out.get("results", []):
        print(f"[gwdemo] unit={res['unit_id']} wall={res['wall_s']:.2f}s error={res['error_class']}")
        for ep in res["episodes"]:
            k = ep["key"]
            print(f"[gwdemo]   {k['task']}/i{k['instance']}  {ep['outcome']:<9} steps={ep['steps']}")
        if res["error"]:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
