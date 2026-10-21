# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""The generation seam.

Named ``rollout``, not ``policy``: DESIGN 4.0 reserves ``policy`` for the trainer/learner
role and ``rollout`` for the generation role, precisely to kill the actor/policy ambiguity.
This module describes the SERVING surface, so it is a rollout.

(``the external native policy rollout body`` is the body. ``the external policy-evaluation entrypoint`` labels its WG
"policy"; that predates this file and is deliberately not renamed.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class RolloutInfo:
    name: str
    action_dim: int
    #: Rows the model EMITS per chunk -- the first axis of what ``act()`` returns.
    #:
    #: **MEASURED 2026-08-27: 32**, for the round-4/5 8B checkpoint. Recovered from six real
    #: episodes rather than from config: a chunk boundary is a discontinuity in the action
    #: stream, and grouping ``|Delta action|`` by position modulo k concentrates it at one
    #: residue for **k=32** (ratio 2.7-3.4 over the median) and at none for k=31 (1.1-1.4,
    #: i.e. flat). Campaign ``round5-c5-smoke``.
    #:
    #: It is NOT guaranteed to equal the chunk length in general -- ``roles/rollout/
    #: cosmos3.py:325`` slices ``[history_length:]`` before returning, so a checkpoint whose
    #: config sets ``history_length > 0`` emits fewer rows than it generates. On this one it
    #: does not.
    #:
    #: ``Cosmos3Policy`` measures it on the first inference, so ``describe()`` reports the
    #: real width even where a config makes the two differ.
    horizon: int
    #: Stamped into every ``Transition.policy_info`` -> RL provenance. Bumped by the
    #: trainer on each weight push; 0 for a frozen checkpoint.
    weight_version: int = 0
    embodiment: str = ""  # key into embodied.embodiment.MAPS


def rollout_info_as_wire(info: RolloutInfo) -> dict:
    """``RolloutInfo`` -> a plain dict, for the sim<->gateway socket.

    Same reason ``roles/env/sim_role.py`` does this for ``SimCapabilities``: the far side of
    that socket is a process that does not import this module, so a dataclass cannot cross it.
    Every field is already a primitive, which is the invariant that keeps this three lines.
    """
    return {
        "name": info.name,
        "action_dim": int(info.action_dim),
        "horizon": int(info.horizon),
        "weight_version": int(info.weight_version),
        "embodiment": info.embodiment,
    }


def rollout_info_from(payload: Mapping[str, Any]) -> RolloutInfo:
    """The inverse. A missing field raises rather than defaulting.

    Defaulting ``horizon`` here would hand the driver a plausible chunk width for a policy that
    never declared one, and the loop would slice against it without complaint.
    """
    missing = sorted({"name", "action_dim", "horizon"} - set(payload))
    if missing:
        raise ValueError(f"rollout info payload is missing {missing}")
    return RolloutInfo(
        name=str(payload["name"]),
        action_dim=int(payload["action_dim"]),
        horizon=int(payload["horizon"]),
        weight_version=int(payload.get("weight_version", 0)),
        embodiment=str(payload.get("embodiment", "")),
    )


class Rollout(Protocol):
    """The serving surface.

    Do NOT use ``@runtime_checkable`` / ``isinstance`` against this. ``PeerClient.
    __getattr__`` (``workers/ray.py``) forwards ANY non-dunder name, so
    ``isinstance(peer, Rollout)`` is unconditionally True and provides no protection across
    exactly the seam that matters -- a cross-venv handle to a policy that may not have been
    built. Validate by CALLING :meth:`describe` at bind time instead; see
    :func:`validate_rollout`.
    """

    def act(self, obs_batch: Sequence[Mapping[str, Any]]) -> list[Any]:
        """One batched forward. Returns one action chunk per obs: ndarray ``[H, action_dim]``."""

    def describe(self) -> RolloutInfo:
        """Static description of what is being served."""


def validate_rollout(
    rollout: Any,
    *,
    expect_action_dim: int | None = None,
    expect_embodiment: str | None = None,
) -> RolloutInfo:
    """Bind-time check that ``rollout`` is really serving, and serving the right thing.

    Calling ``describe()`` is the cheapest operation that actually crosses the wire, and so
    the cheapest thing that can fail honestly -- see :class:`Rollout` for why the
    ``isinstance`` check a reader would reach for first is worthless at this seam.

    ``expect_action_dim`` catches the failure that is otherwise silent: serving a policy
    trained for another embodiment produces well-formed actions of the wrong width, or the
    right width and the wrong meaning, and the robot simply flails. Passing the
    embodiment's declared ``action_dim`` turns that into an error at bind time.

    ``expect_embodiment`` closes the half of that hole a width check cannot reach. Two
    embodiments with the SAME ``action_dim`` pass every check above -- 23 numbers arrive, 23
    numbers are applied, and they mean different joints. Optional rather than
    required because a rollout may legitimately not know (a bare ``RolloutInfo`` in a test,
    an OSS body), and a check that forced every caller to answer would be answered with a
    guess.
    """
    info = rollout.describe()
    if not isinstance(info, RolloutInfo):
        raise TypeError(f"rollout.describe() returned {type(info).__name__}, expected RolloutInfo")
    if info.horizon < 1:
        raise ValueError(f"rollout {info.name!r} declares horizon {info.horizon}; must be >= 1")
    if expect_action_dim is not None and info.action_dim != expect_action_dim:
        raise ValueError(
            f"rollout {info.name!r} serves action_dim {info.action_dim}, but this embodiment "
            f"declares {expect_action_dim}. Serving another embodiment's weights produces "
            f"well-formed actions and a flailing robot, with no error anywhere downstream."
        )
    if expect_embodiment is not None and info.embodiment != expect_embodiment:
        raise ValueError(
            f"rollout {info.name!r} serves embodiment {info.embodiment or '(undeclared)'!r}, "
            f"but this role runs {expect_embodiment!r}. An action_dim check cannot catch this: "
            f"two embodiments of the same width produce well-formed actions that mean "
            f"different joints, and every gate downstream reads them as valid data."
        )
    return info
