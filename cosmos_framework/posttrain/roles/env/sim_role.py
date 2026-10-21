# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""What the framework may call on a simulator body, and the bind-time check for it.

Separate from ``base.py`` on purpose: a simulator holds ``n_lanes`` concurrent episodes, is
stepped by the caller in a batch, and owns truncation, so it cannot satisfy ``base.Env``'s
1-step ``reset``/``step`` contract -- as ``base.Env``'s own docstring concedes.

**The three layers, and which one this is.**

    embodied/sim.py       Simulator / SimSession   -- the ENGINE seam (one scene, n lanes)
    roles/env/sim_role.py SimRole                  <- HERE: the FRAMEWORK seam (one actor)
    roles/env/base.py     Env                      -- the 1-step dataset seam

``SimRole`` is what a Ray handle answers to; ``Simulator`` is what runs behind it. A
container-backed backend changes the second and leaves this one untouched, which is the
property that makes the target topology reachable without re-opening the framework side.

**WHAT THIS MODULE HOLDS, AND WHAT IT DOES NOT.** The contract and its capability codec, and
nothing else. A placement policy is not a contract: the same-card refusal (S-04) is a
property of ONE implementation -- the gateway, whose simulator and policy actor can land on
different cards -- so it lives beside ``roles/env/sim_gateway.py``, its only caller, not here where
a reader would take it for something every ``SimRole`` must satisfy.
"""

from __future__ import annotations

from dataclasses import fields
from typing import Any, Mapping, Protocol, Sequence

from cosmos_framework.posttrain.embodied.sim import PROTOCOL_VERSION, SimCapabilities, SimProtocolError
from cosmos_framework.posttrain.embodied.window import DecisionRollout, EvaluationWindowReport, WindowReport


class SimRole(Protocol):
    """The framework-facing surface of a simulator body.

    **NOT ``@runtime_checkable``**, for the reason ``embodied/rollout.py`` gives about
    ``Rollout``: both ``PeerClient.__getattr__`` and ``WorkerGroup.__getattr__`` manufacture
    any attribute name, so ``isinstance(handle, SimRole)`` is unconditionally True -- even
    against a handle whose remote actor was never constructed. Validate by CALLING
    :meth:`probe`; see :func:`validate_sim_role`.

    **ROBOLAB IS EXPLICITLY OUT OF THIS CONTRACT until phase 3.**
    ``the external simulator environment body`` hands its whole loop to ``robolab.eval.runner.run_evaluation``
    through a ``client_factory``; that package is not in this repo, not vendored, not
    installed, and has no stub, so nobody here can say whether it could satisfy this.
    """

    def probe(self) -> dict[str, Any]:
        """Second-line card check **and** the capability handshake.

        Returns ``{"host": str, "gpu": str | None, "capabilities": dict}``. Called at bind
        time, before any scene build.

        The capabilities ride as a **plain dict**, not a :class:`SimCapabilities`. This
        return value crosses an actor boundary, and ``embodied/sim.py``'s declared wire rule
        forbids custom classes on exactly that hop -- shipping the dataclass here would be
        this module violating the contract it exists to police. :func:`capabilities_from`
        rebuilds it on the far side.
        """

    def run_units(self, units: Sequence[Any], *, policy: Any, **kw: Any) -> dict[str, Any]:
        """PUSH: run the units this call names. Returns a result dict."""

    def run_from_source(self, units: Sequence[Any], *, policy: Any, source: str, **kw: Any) -> dict[str, Any]:
        """PULL: claim work from ``source`` until it is exhausted or a budget runs out."""

    def run_window(
        self, units: Sequence[Any] = (), *, policy: DecisionRollout, version: int, decisions: int
    ) -> WindowReport:
        """Drive a bound decision peer; requires the decision_chunks capability."""

    def run_evaluation(
        self,
        units: Sequence[Any] = (),
        *,
        policy: DecisionRollout,
        cases: list[dict[str, int | str]],
        version: int,
        seed: int,
        decisions: int,
    ) -> EvaluationWindowReport:
        """Evaluate independent sessions; requires fixed_case_evaluation."""

    def close(self) -> None:
        """Release the sink and the simulator. Must be safe after a partial run."""


def capabilities_as_wire(caps: SimCapabilities) -> dict[str, Any]:
    """``SimCapabilities`` -> the plain dict :meth:`SimRole.probe` returns.

    ``camera_shapes`` values become lists rather than tuples, deliberately: this payload is
    written into provenance and has to survive JSON as well as pickle.
    :func:`capabilities_from` normalises back.
    """
    return {
        "protocol_version": caps.protocol_version,
        "backend": caps.backend,
        "backend_version": caps.backend_version,
        "embodiment": caps.embodiment,
        "max_lanes": caps.max_lanes,
        "camera_shapes": {k: [int(v[0]), int(v[1])] for k, v in caps.camera_shapes.items()},
        "proprio_width": caps.proprio_width,
        "extras": list(caps.extras),
        "supports_render_skip": caps.supports_render_skip,
    }


def _need_str(payload: Mapping[str, Any], key: str) -> str:
    v = payload[key]
    if type(v) is not str:
        raise SimProtocolError(f"capability {key!r} must be a str, got {type(v).__name__} ({v!r})")
    return v


def _need_int(payload: Mapping[str, Any], key: str, *, minimum: int) -> int:
    v = payload[key]
    # `bool` is a subclass of `int` and `True` would silently become 1. `float` is refused
    # rather than truncated: `protocol_version=1.9` becoming 1 is a version skew reported as
    # agreement, which is the one thing this payload exists to make impossible.
    if type(v) is not int:
        raise SimProtocolError(
            f"capability {key!r} must be an int, got {type(v).__name__} ({v!r}). "
            f"A float would be truncated and a bool would read as 0/1; both would turn a "
            f"malformed payload into a plausible one."
        )
    if v < minimum:
        raise SimProtocolError(f"capability {key!r} is {v}, which is below the minimum {minimum}")
    return v


def _need_bool(payload: Mapping[str, Any], key: str) -> bool:
    v = payload[key]
    if type(v) is not bool:
        raise SimProtocolError(
            f"capability {key!r} must be a bool, got {type(v).__name__} ({v!r}). "
            f"Coercing would make the string 'false' mean True."
        )
    return v


def capabilities_from(payload: Mapping[str, Any]) -> SimCapabilities:
    """The inverse. Raises :class:`SimProtocolError` on a payload that is not one.

    **Validated, never coerced**, and that is the whole point of the function. This payload
    arrives from another PROCESS -- since the container split, potentially another team's
    simulator built against this contract -- so it is the seam where a foreign
    implementation's mistake has to become a legible protocol error rather than a plausible
    capability. Every coercion here had a silent-success mode:

    ``bool("false")`` is ``True``; ``int(1.9)`` is ``1``, a version skew reported as
    agreement; ``bool`` passes an ``int`` check because it subclasses one; ``tuple("head")``
    is ``('h','e','a','d')``, four extras nobody declared; and a malformed ``camera_shapes``
    value raised ``TypeError``/``IndexError`` out of a function whose contract is
    :class:`SimProtocolError`.

    Ranges are checked with the types, because a negative ``max_lanes`` or ``proprio_width``
    is not a capability either.
    """
    missing = sorted({f.name for f in fields(SimCapabilities)} - set(payload))
    if missing:
        raise SimProtocolError(
            f"capability payload is missing {missing}; a backend that does not answer the "
            f"whole contract cannot be checked against it, and defaulting the gaps would "
            f"turn the mismatch this exists to catch into a silent one"
        )

    shapes_raw = payload["camera_shapes"]
    if not isinstance(shapes_raw, Mapping):
        raise SimProtocolError(f"capability 'camera_shapes' must be a mapping, got {type(shapes_raw).__name__}")
    camera_shapes: dict[str, tuple[int, int]] = {}
    for name, shape in shapes_raw.items():
        if type(name) is not str:
            raise SimProtocolError(f"camera name {name!r} is {type(name).__name__}, not a str")
        # Length and element type BOTH, and before any indexing: `shape[0]` on a bare int is
        # a TypeError that escapes this function's declared error type, and on a 1-tuple an
        # IndexError. Either way the caller sees a traceback from numpy-adjacent code instead
        # of "your simulator's capability payload is malformed".
        if isinstance(shape, (str, bytes)) or not isinstance(shape, Sequence) or len(shape) != 2:
            raise SimProtocolError(f"camera {name!r} shape must be a 2-sequence (height, width), got {shape!r}")
        h, w = shape
        if type(h) is not int or type(w) is not int or h <= 0 or w <= 0:
            raise SimProtocolError(f"camera {name!r} shape {shape!r} must be two positive ints (height, width)")
        camera_shapes[name] = (h, w)

    extras_raw = payload["extras"]
    # A str IS a Sequence, and `tuple("head")` yields four one-character extras that then
    # satisfy a subset check nobody meant to pass.
    if isinstance(extras_raw, (str, bytes)) or not isinstance(extras_raw, Sequence):
        raise SimProtocolError(
            f"capability 'extras' must be a sequence of str, got {type(extras_raw).__name__} "
            f"({extras_raw!r}). A bare string would become one extra per character."
        )
    extras = tuple(extras_raw)
    if bad_extras := [e for e in extras if type(e) is not str]:
        raise SimProtocolError(f"capability 'extras' holds non-str entries {bad_extras!r}")

    return SimCapabilities(
        protocol_version=_need_int(payload, "protocol_version", minimum=0),
        backend=_need_str(payload, "backend"),
        backend_version=_need_str(payload, "backend_version"),
        embodiment=_need_str(payload, "embodiment"),
        max_lanes=_need_int(payload, "max_lanes", minimum=0),
        camera_shapes=camera_shapes,
        proprio_width=_need_int(payload, "proprio_width", minimum=0),
        extras=extras,
        supports_render_skip=_need_bool(payload, "supports_render_skip"),
    )


def validate_sim_role(role: Any, *, expect: SimCapabilities) -> SimCapabilities:
    """Bind-time check, by CALLING ``probe()``. A mismatch RAISES.

    The value of doing it here is entirely about cost: every mismatch below is otherwise
    discovered after an Isaac boot and a scene build, i.e. minutes into a unit that has
    already been claimed off the shared lease plane -- and two of them are not discovered at
    all, they just produce well-formed data that means something else.

    **Each field is compared with its own semantics, and one is deliberately not compared:**

    ``protocol_version``  EQUAL. The two sides disagree about the contract itself.
    ``backend``           EQUAL. A different engine is a different experiment.
    ``embodiment``        EQUAL. The silent one: same ``action_dim``, different joints.
    ``max_lanes``         role >= expect. The role's is a CEILING, expect's is a demand --
                          and for b1k the ceiling is 1, carrying round 4's measured tiling
                          refusal (V=2 scored 0/316 where V=1 scored 5/40). This turns that
                          refusal from a scene-build-time raise into a bind-time one.
    ``proprio_width``     role >= expect. Same direction and the same reason
                          ``to_record_view`` refuses a short vector rather than padding it.
    ``camera_shapes``     Every camera EXPECT names must exist and match. Absence is a
                          mismatch, not a pass: a run that names ``head`` at 720 is stating a
                          requirement, and a backend without a head camera cannot meet it --
                          it would bind, boot, and produce episodes missing the stream the
                          policy reads. The default configuration stays bindable because it
                          names NO cameras, which is the case the earlier "absence is
                          unrequested" rule was really about; a backend declaring extra
                          cameras nobody asked for is still free.
    ``extras``            expect's must be a SUBSET. Extra capabilities are free; a missing
                          one is not.
    ``supports_render_skip``  Only when expect demands it.
    ``backend_version``   **NEVER COMPARED.** It is best-effort provenance -- an image
                          digest when one is known and ``""`` when not -- so a check on it
                          would fail every bind that could not read it, which is most of
                          them. Stated here because a reader who assumes it is checked would
                          also assume a version skew had been ruled out.
    """
    reply = role.probe()
    if not isinstance(reply, Mapping) or "capabilities" not in reply:
        raise SimProtocolError(
            f"sim role probe() returned {type(reply).__name__} without a 'capabilities' key. "
            f"probe() is the handshake site: a body that answers the call but not the "
            f"contract is a body nothing can be checked against."
        )
    caps = capabilities_from(reply["capabilities"])

    bad: list[str] = []
    if caps.protocol_version != expect.protocol_version:
        bad.append(
            f"protocol_version {caps.protocol_version} != {expect.protocol_version} "
            f"(this process is built for {PROTOCOL_VERSION})"
        )
    if caps.backend != expect.backend:
        bad.append(f"backend {caps.backend!r} != {expect.backend!r}")
    if caps.embodiment != expect.embodiment:
        bad.append(
            f"embodiment {caps.embodiment!r} != {expect.embodiment!r} -- an action_dim check "
            f"cannot see this, and the result is well-formed actions on the wrong joints"
        )
    if caps.max_lanes < expect.max_lanes:
        bad.append(f"max_lanes {caps.max_lanes} < the {expect.max_lanes} this run wants")
    if caps.proprio_width < expect.proprio_width:
        bad.append(f"proprio_width {caps.proprio_width} < the {expect.proprio_width} declared")
    for name, want in expect.camera_shapes.items():
        got = caps.camera_shapes.get(name)
        if got is None:
            bad.append(
                f"camera {name!r} is required by this run at {tuple(want)} and the backend declares no such camera"
            )
        elif tuple(got) != tuple(want):
            bad.append(f"camera {name!r} renders {tuple(got)}, run wants {tuple(want)}")
    if missing_extras := sorted(set(expect.extras) - set(caps.extras)):
        bad.append(f"missing extras {missing_extras}")
    if expect.supports_render_skip and not caps.supports_render_skip:
        bad.append("render-skip demanded, backend cannot")

    if bad:
        raise SimProtocolError(
            f"sim role on {reply.get('host')!r} (gpu {reply.get('gpu')!r}) does not satisfy this "
            f"run: " + "; ".join(bad) + ". Refused at bind time rather than after a scene build."
        )
    return caps


def require_sim_capability(role: SimRole, capability: str) -> SimCapabilities:
    """Bind an optional operation using the same explicit capability handshake."""
    caps = capabilities_from(role.probe()["capabilities"])
    if capability not in caps.extras:
        raise SimProtocolError(f"simulator {caps.backend!r} does not provide {capability}")
    return caps
