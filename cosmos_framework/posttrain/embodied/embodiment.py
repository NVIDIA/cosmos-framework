# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Observation keys, declared once.

These three ``::``-flattened camera keys are declared in two other places, and nothing
compares either declaration with the map below. ``roles/env/the external simulator backend:42-44`` carries
them under a comment reading "If these ever diverge the flatten test catches it" -- no such
test exists. Outside this package,
``cosmos3/evaluation/action/websocket_policy_server/serve_policy_the external simulator backend:262-264``
declares ``_HEAD_KEY`` / ``_LWRIST_KEY`` / ``_RWRIST_KEY`` as its own. Divergence is silent
until a policy call raises
``KeyError`` on the head camera, after the scene has loaded and an 8B checkpoint is resident.

This module makes divergence impossible rather than test-detected: there is one literal,
and both sides read it.

Two module-level functions rather than a Protocol: the key lists across simulators are
**duplicated, not divergent**, so a plugin point here would be abstraction without a second
instance. When a second embodiment genuinely disagrees, add a second :class:`EmbodimentMap`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from cosmos_framework.posttrain.schema.trajectory import ObsBundle


@dataclass(frozen=True)
class EmbodimentMap:
    """What a robot's observation looks like, in the flattened key space the policy uses.

    ``proprio_key`` is a single key: OmniGibson emits one ``robot_r1::proprio`` vector, and
    ``state61`` is its first ``state_dim`` entries (§5.2 of the round-4 spec, normative about
    the package contract). There is no separate ``observation/joint_position`` key on this
    path.
    """

    name: str
    cameras: Mapping[str, str]  # logical name -> raw (flattened) sim obs key
    proprio_key: str
    action_dim: int
    state_dim: int
    #: Raw obs key under which the session publishes per-frame camera extrinsics in the
    #: robot-base frame, ``[n_cameras, 7]`` as ``(xyz, quat)``. Extrinsics are not part of
    #: OmniGibson's observation dict, so the SimSession must inject them; the core stays
    #: sim-agnostic by reading one declared key. ``None`` -> the sink records no extrinsics.
    extrinsics_key: str | None = None
    #: Order of rows in the extrinsics array, for the record. v1 packages use
    #: ``[left_realsense, right_realsense, zed]`` and the sealed converter depends on it.
    extrinsics_order: tuple[str, ...] = ()


R1PRO_B1K = EmbodimentMap(
    name="r1pro_b1k",
    cameras={
        "head": "robot_r1::robot_r1:zed_link:Camera:0::rgb",
        "left_wrist": "robot_r1::robot_r1:left_realsense_link:Camera:0::rgb",
        "right_wrist": "robot_r1::robot_r1:right_realsense_link:Camera:0::rgb",
    },
    proprio_key="robot_r1::proprio",
    action_dim=23,
    state_dim=61,  # == the registered ACT2 spec `behavior_1k_demos`
    extrinsics_key="robot2cam",
    extrinsics_order=("left_realsense", "right_realsense", "zed"),
)

#: Embodiments known by name. **Use :func:`register_embodiment` to add one.**
MAPS: dict[str, EmbodimentMap] = {R1PRO_B1K.name: R1PRO_B1K}


def register_embodiment(emb: EmbodimentMap) -> EmbodimentMap:
    """Make `emb` resolvable by name, and return it so this can be used at module scope.

    Backends call this at import time, next to their `Simulator`. The alternative -- editing
    the ``MAPS`` literal above -- makes every new embodiment a change to a CORE module, which
    is the coupling `embodied/` exists to remove: a backend is supposed to be reachable
    through the registry without the core knowing its name.

    **Re-registering the same name with a DIFFERENT map raises.** Two backends silently
    sharing one name is the failure this cannot afford to be quiet about: the loser's camera
    and proprio keys are simply wrong for the engine that ends up using them, and the symptom
    is a `MissingObsKey` from inside a file the author did not write. Re-registering an
    IDENTICAL map is a no-op, so a module imported twice is fine.
    """
    prev = MAPS.get(emb.name)
    if prev is not None and prev != emb:
        raise ValueError(
            f"embodiment {emb.name!r} is already registered with a different map. "
            f"Names are global across backends; pick another."
        )
    MAPS[emb.name] = emb
    return emb


def get_embodiment(name: str) -> EmbodimentMap:
    """Resolve `name`, or raise naming what IS registered.

    A ``.get()`` with a fallback was the original shape here and it was worse than an error:
    an unknown name silently became the example's map, so a backend author who named their
    embodiment and never registered it got the WRONG camera and proprio keys, no warning, and
    a failure later that named an embodiment they had not asked for.
    """
    try:
        return MAPS[name]
    except KeyError:
        raise KeyError(
            f"no embodiment named {name!r}. Registered: {sorted(MAPS)}. "
            f"Call embodiment.register_embodiment(...) from your backend module."
        ) from None


class MissingObsKey(KeyError):
    """A declared observation key is absent from the raw observation.

    Its own class so a caller can tell "the simulator did not give us what the embodiment
    declares" apart from any other ``KeyError`` raised inside a policy adapter.
    """


def _require(raw: Mapping[str, Any], key: str, emb: EmbodimentMap, what: str) -> Any:
    try:
        return raw[key]
    except KeyError:
        near = sorted(k for k in raw if key.split("::")[0] in k)[:6]
        raise MissingObsKey(
            f"embodiment {emb.name!r} declares {what} at {key!r}, absent from the raw "
            f"observation. Present keys under the same root: {near or '(none)'}. "
            f"This is the divergence embodiment.py exists to prevent -- fix the key here, "
            f"in one place, not at the call site."
        ) from None


@dataclass(frozen=True)
class ArraySpec:
    """What ONE declared observation array is.

    The content of the observation type. :class:`~schema.trajectory.ObsBundle` is the box
    and its ``representations`` is a bare ``dict[str, Any]`` with no required keys and no
    ``__post_init__`` -- so the box types nothing on its own, and reusing it would be
    ``dict`` with a nicer name if this did not exist.

    It is also the unit a shared-memory slot is sized from, which is why it carries dtype
    and channels even though nothing asserts them yet.
    """

    #: ``None`` = NEVER MEASURED. Not "any dtype is acceptable".
    #:
    #: These were all ``None`` until ``describe_obs`` ran on a pod (2026-08-27, campaign
    #: ``round5-c5-smoke``) and they are filled in from that run, not from reading code.
    dtype: str | None
    #: Number of array dimensions. Asserted by :func:`check_obs` only where a mismatch is
    #: ALREADY fatal downstream -- see that function.
    rank: int
    #: Trailing-axis width for an image. ``None`` = not applicable to this key.
    #:
    #: **The ENGINE emits 4, MEASURED on 2026-08-27** -- the cameras are RGBA. The number
    #: came from a pod printing it, and the raw payload reconciles to the byte
    #: (`720^2*4 + 2*480^2*4` + 61 float32 + a 3x7 block = 3,917,128).
    #:
    #: This declares the observation AT THE BOUNDARY: ``the external simulator backend::detach_frames``
    #: copies frames out of the engine's buffer WITHOUT narrowing, and every reader narrows
    #: for itself. ``describe_obs`` reports the measured width once per unit into the pod
    #: log, so a declaration that stops matching the engine is visible. Still NOT asserted
    #: by ``check_obs``: see there for why only the already-fatal properties are.
    channels: int | None = None
    #: Whether the raw observation must contain this key. Extrinsics are the one declared
    #: key that legitimately goes missing -- ``_camera_extrinsics`` returns ``None`` rather
    #: than raising when the sensors cannot be resolved, and ``to_record_view`` already
    #: treats absence as "write zeros". Encoding that here rather than special-casing it
    #: inside ``check_obs`` keeps the whole contract readable in one place.
    required: bool = True


def obs_spec(emb: EmbodimentMap) -> dict[str, ArraySpec]:
    """The observation this embodiment DECLARES, keyed by raw flattened key.

    Derived from :class:`EmbodimentMap` rather than written out, so the key space still
    lives in exactly one literal.
    """
    # dtype MEASURED 2026-08-27 by `describe_obs` on campaign `round5-c5-smoke` (see
    # ArraySpec). uint8 is not inferred from the encoder's appetite -- the raw payload
    # measured 3,917,128 B and reconciles exactly at one byte per channel, at 4 channels.
    spec = {key: ArraySpec(dtype="uint8", rank=3, channels=4) for key in emb.cameras.values()}
    spec[emb.proprio_key] = ArraySpec(dtype="float32", rank=1)
    if emb.extrinsics_key is not None:
        # dtype IS known here, unlike everywhere else in this function: the extrinsics are
        # not an engine output at all. The session builds them itself and ends with an
        # explicit ``.astype(np.float32)``, so this is a fact about our own code.
        spec[emb.extrinsics_key] = ArraySpec(dtype="float32", rank=2, required=False)
    return spec


def to_sim_obs(raw: Mapping[str, Any]) -> ObsBundle:
    """Box a simulator's flattened observation. **ADOPTS ``raw`` without copying.**

    ``flatten_obs`` builds a fresh dict every step and hands over ownership, so adopting is
    correct and copying would duplicate ~2.9 MB of camera views per lane per step for
    nothing. This is the same consume-before-return convention ``EpisodeWriter.append``
    already runs on, for the same reason: this pipeline has exactly one thing big enough to
    matter and it is these arrays.

    A non-``dict`` Mapping IS copied, because ``ObsBundle.representations`` is typed
    ``dict`` and a caller that later mutates through the original would otherwise be
    mutating the bundle.
    """
    return ObsBundle(representations=raw if type(raw) is dict else dict(raw))


def camera(obs: ObsBundle, emb: EmbodimentMap, logical: str) -> np.ndarray:
    """Read one camera by its LOGICAL name (``"head"``, ``"left_wrist"``, ...).

    This is how the ``::``-flattened OmniGibson key space stays out of every call site
    while also staying out of the type. The raw key is written in one place --
    :class:`EmbodimentMap` -- and a reader never types it.

    It is also why a dedicated observation class was not needed: ``camera(obs, emb, "head")``
    is as greppable as ``obs.head``, and unlike a named field it does not have to change when
    a second robot arrives with different cameras.
    """
    try:
        key = emb.cameras[logical]
    except KeyError:
        raise MissingObsKey(
            f"embodiment {emb.name!r} has no camera named {logical!r}; it declares "
            f"{sorted(emb.cameras)}. Logical names are the embodiment's vocabulary, not "
            f"the engine's -- adding a camera means adding it to the EmbodimentMap."
        ) from None
    return np.asarray(_require(obs.representations, key, emb, f"camera {logical!r}"))


def proprio(obs: ObsBundle, emb: EmbodimentMap) -> np.ndarray:
    """Read the proprio vector, as float32 and flattened.

    Same reason as :func:`camera`. The cast and the flatten are here because every consumer
    in the tree applies them -- keeping the accessor faithful to the raw array would mean
    every caller repeating the same two operations, which is how they come to disagree.

    Returns the WHOLE vector. ``state61`` is its first ``emb.state_dim`` entries and is
    ``to_record_view``'s business, not this function's.
    """
    return np.asarray(_require(obs.representations, emb.proprio_key, emb, "proprio"), dtype=np.float32).reshape(-1)


def check_obs(obs: ObsBundle, emb: EmbodimentMap) -> None:
    """Raise if a required declared key is absent, or if a camera has the wrong rank.

    **MANDATORY, and called from a declared site** -- ``driver.run_unit`` calls it once per
    unit, on the first live lane's observation. Making it optional is how
    ``ObsBundle`` + :class:`ArraySpec` decay into ``dict[str, Any]`` with a nicer name.

    **It asserts ONLY the two properties that are already fatal downstream**, just later
    and less legibly:

    - a missing key crashes inside ``Behavior1KPolicy._adapt_obs``, after a scene load AND
      an 8B checkpoint load, on the far side of the wire;
    - a camera of the wrong rank crashes inside the hevc encoder, mid-episode, after the
      episode has already cost minutes of physics.

    Rank is checked for CAMERAS ONLY, and the asymmetry is deliberate rather than an
    oversight: the encoder is the thing a wrong rank is fatal to, and it only ever sees
    cameras. ``to_record_view`` reshapes proprio with ``reshape(-1)``, so a rank-2 proprio
    works today -- asserting rank 1 there would ADD a failure mode rather than move an
    existing one earlier.

    dtype and channels are RECORDED by ``describe_obs``, not asserted.
    """
    raw = obs.representations
    # LOGICAL names, not raw keys. A reader who has to map
    # `robot_r1::robot_r1:zed_link:Camera:0::rgb` back to "the head camera" by eye is doing
    # the work this module exists to do for them -- and they are doing it at the moment a
    # unit has just failed, which is the worst time to ask.
    label = {key: f"camera {name!r}" for name, key in emb.cameras.items()}
    label[emb.proprio_key] = "proprio"
    if emb.extrinsics_key is not None:
        label[emb.extrinsics_key] = "extrinsics"

    for key, spec in obs_spec(emb).items():
        if key not in raw:
            if spec.required:
                _require(raw, key, emb, label.get(key, f"key {key!r}"))
            continue
        if key not in emb.cameras.values():
            continue
        ndim = np.asarray(raw[key]).ndim
        if ndim != spec.rank:
            raise ValueError(
                f"embodiment {emb.name!r} declares {label[key]} ({key!r}) with rank "
                f"{spec.rank}, and the simulator produced rank {ndim}. Left alone this "
                f"reaches the hevc encoder mid-episode, after the physics has been paid for."
            )


def _array_facts(value: Any) -> dict:
    """(shape, dtype, nbytes) for one raw observation value.

    Anything that is not array-like is reported as ``kind`` and no measurement, rather than
    coerced. A reporting tool that turned an engine handle into ``shape: []`` would answer the
    question it was built to ask with a number it invented.
    """
    # Type-checked BEFORE asarray, not after. `np.asarray(engine_handle)` does not raise --
    # it succeeds and returns a 0-d object array -- so a try/except here never fires and the
    # report would read `object-array[]`, which names neither the value nor its type. The
    # whole purpose of this function is to say what is actually in the observation.
    if not isinstance(value, (np.ndarray, np.generic, list, tuple)):
        return {"kind": type(value).__name__}
    try:
        arr = np.asarray(value)
    except Exception:  # noqa: BLE001 -- describing an observation must never cost a unit
        return {"kind": type(value).__name__}
    if arr.dtype == object:
        return {"kind": f"object-array{list(arr.shape)}"}
    return {"shape": [int(n) for n in arr.shape], "dtype": str(arr.dtype), "nbytes": int(arr.nbytes)}


def describe_obs(obs: ObsBundle, emb: EmbodimentMap) -> dict:
    """Per-key ``(shape, dtype, nbytes)`` for the DECLARED and UNDECLARED sets, plus totals.

    The measurement behind :class:`ArraySpec`'s dtype and channel declarations: per-key dtype
    and shape, the size of the undeclared set, and the camera channel count, read off a real
    observation rather than off the code. ``driver.run_unit`` calls it once per unit, so a
    declaration that stops matching the engine shows up in the pod log.

    Wire-safe by construction: strings, ints and lists only, so it can be printed, put on the
    heartbeat, and later written into provenance without a second encoding decision.
    """
    raw = obs.representations
    declared_spec = obs_spec(emb)
    logical_of = {key: name for name, key in emb.cameras.items()}
    logical_of[emb.proprio_key] = "proprio"
    if emb.extrinsics_key is not None:
        logical_of[emb.extrinsics_key] = "extrinsics"

    declared: dict[str, Any] = {}
    undeclared: dict[str, Any] = {}
    for key in sorted(raw):
        facts = _array_facts(raw[key])
        if key in declared_spec:
            declared[key] = {"logical": logical_of.get(key, ""), **facts}
        else:
            undeclared[key] = facts

    def _sum(group: Mapping[str, Any]) -> int:
        return int(sum(int(v.get("nbytes", 0)) for v in group.values()))

    # THE question, hoisted out of the per-key detail so it cannot be missed in a log line.
    # ``None`` when the camera is absent or not rank-3 -- an honest "not measured" rather than
    # a zero that would read as an answer.
    channels: dict[str, int | None] = {}
    for name, key in emb.cameras.items():
        shape = declared.get(key, {}).get("shape") or []
        channels[name] = int(shape[-1]) if len(shape) == 3 else None

    return {
        "embodiment": emb.name,
        "declared": declared,
        "undeclared": undeclared,
        "declared_bytes": _sum(declared),
        "undeclared_bytes": _sum(undeclared),
        "total_bytes": _sum(declared) + _sum(undeclared),
        "camera_channels": channels,
        "n_declared": len(declared),
        "n_undeclared": len(undeclared),
    }


def to_policy_obs(obs: ObsBundle, emb: EmbodimentMap, *, task: str) -> dict:
    """The wire payload the policy expects: flattened obs plus ``task_name``.

    **NOT byte-identical to what ``roles/env/the external simulator backend::_obs_for_policy`` sends**, and
    that path is still in the tree and still live. The difference is exactly one: this
    payload carries ``robot2cam``, injected by the session, and the eval role never produces
    it. Both wires carry 4-channel cameras; ``serve_policy_behavior1k._to_hwc_uint8`` narrows
    for the model.

    Why that does not move a score: ``serve_policy_behavior1k._adapt_obs`` names exactly four keys
    (head, left wrist, right wrist, proprio) and raises if any is missing, and
    ``_build_data_batch`` reads the three composed views. ``robot2cam`` is named by neither.
    That is what lets C7 switch loops without an A/B on the observation itself -- but it is a
    claim about TODAY's server, so a change to ``_adapt_obs``'s key list invalidates it.

    Whether ``robot2cam`` is present is decided by the SINK, not the embodiment: computing it
    costs a measured 2.383 ms/step (see :func:`sink.sink_wants_extrinsics`), so the gate is
    ``sink_wants_extrinsics`` -> ``CollectorBody`` -> ``Behavior1KSimulator(emit_extrinsics=)``.
    COLLECTION still computes it; the eval path does not.

    The keys are checked here rather than left to fail inside the policy. The failure this
    prevents costs a scene load and an 8B checkpoint load before it surfaces.

    **Proprio is checked too, not just the cameras.** ``Behavior1KPolicy._adapt_obs`` guards
    all FOUR raw keys in one condition -- the three cameras *and* the proprio key -- then
    indexes them immediately, so omitting proprio here just moves that failure across the
    wire.

    **The return type is deliberately UNCHANGED** -- still the flat dict the policy expects,
    plus ``task_name``. The sim->policy hop is not the boundary being decoupled here; it is
    already two actors. Typing it would open both WSS servers and ``roles/rollout/
    cosmos3.py`` -- the exact surface phase 0's score comparison has to run through
    untouched.
    """
    raw = obs.representations
    for logical, key in emb.cameras.items():
        _require(raw, key, emb, f"camera {logical!r}")
    _require(raw, emb.proprio_key, emb, "proprio")
    out = dict(raw)
    out["task_name"] = task
    return out


def to_record_view(obs: ObsBundle, emb: EmbodimentMap) -> dict:
    """The subset the sink persists.

    Returns ``{"head": HxWx3 u8, "left_wrist": ..., "right_wrist": ..., "state61": (61,)
    f32, "raw_proprio": (D,) f32}``, plus ``"robot2cam": (n_cameras, 7) f32`` when the
    session publishes extrinsics.

    ``D`` is whatever OmniGibson emits and is deliberately not asserted: RLinf's v1
    packages carried a 256-wide ``raw_proprio`` from a different observation entirely, and
    claiming parity with it would be false. The width is recorded in ``episode.json``
    instead (spec §5.3).

    Frames are returned as views, not copies. The writer contract
    (:meth:`sink.EpisodeWriter.append`) is that a frame passed to ``append`` is consumed
    before return, so copying here would double the peak footprint of the one thing in this
    pipeline big enough to matter.
    """
    raw = obs.representations
    view: dict = {}
    for logical, key in emb.cameras.items():
        frame = np.asarray(_require(raw, key, emb, f"camera {logical!r}"))
        view[logical] = frame[..., :3] if frame.ndim == 3 and frame.shape[-1] == 4 else frame

    proprio = np.asarray(_require(raw, emb.proprio_key, emb, "proprio"), dtype=np.float32).reshape(-1)
    if proprio.size < emb.state_dim:
        raise ValueError(
            f"embodiment {emb.name!r} declares state_dim={emb.state_dim} but "
            f"{emb.proprio_key!r} is only {proprio.size} wide. Truncating would silently "
            f"produce a short state vector that trains badly rather than failing."
        )
    view["state61"] = proprio[: emb.state_dim]
    view["raw_proprio"] = proprio

    if emb.extrinsics_key is not None and emb.extrinsics_key in raw:
        view["robot2cam"] = np.asarray(raw[emb.extrinsics_key], dtype=np.float32)
    return view
