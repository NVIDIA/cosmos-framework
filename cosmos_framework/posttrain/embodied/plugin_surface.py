# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""What ships into the simulator container, declared in code rather than in prose.

The set existed already -- as three prose descriptions and one hard-coded 8-tuple inside a
test. That is enough to be wrong in two directions at once: the tuple can fall behind the
directory (a new module under ``embodied/`` escapes gate G4 in silence), and the prose can
fall behind the tuple (a reader plans an image around a list nothing checks).

**This module is the declaration; ``tests/test_import_discipline.py`` is the enforcement.**
Keeping them apart matters: the gates need something to check *against*, and a gate that
declared its own expectation would be a test agreeing with itself.

Stdlib only, like the rest of the core -- this file is inside the surface it describes.
"""

from __future__ import annotations

from pathlib import Path

#: Files under ``embodied/`` that are not part of the core module set.
_NOT_CORE = frozenset({"__init__.py"})

#: Subpackages under ``embodied/`` that are EXEMPT from the core import rule -- they are
#: where the heavy imports live, reached only through the registry's lazy ``__import__``.
SUBPACKAGES = ("sims", "rollouts", "sinks", "sources")


def _modules_in(pkg_dir: Path) -> tuple[str, ...]:
    return tuple(sorted(p.stem for p in pkg_dir.glob("*.py") if p.name not in _NOT_CORE))


def core_modules() -> tuple[str, ...]:
    """Every module directly under ``embodied/``. **A WALK, NOT A LITERAL.**

    Walked at CALL time rather than import time so importing this module stays free. The
    core's whole claim is that it costs nothing to import.
    """
    return _modules_in(Path(__file__).resolve().parent)


def sim_side_roots() -> tuple[str, ...]:
    """The processes the SIM CONTAINER starts, as ``posttrain.`` module paths.

    A LITERAL, and deliberately -- unlike :func:`core_modules` above. That one walks a directory
    because "everything under ``embodied/``" is the rule itself. This is a different kind of
    fact: ``roles/env/`` holds modules for BOTH venvs -- ``sim_gateway`` runs in the policy venv,
    ``collector`` and ``sim_role`` are imported by the sim side but are not entrypoints -- so a
    walk of that directory would assert the wrong thing in both directions.

    **What this list is FOR.** The sim venv's Ray and its python-version pin to the policy venv
    exist for one reason: under ``sim_transport: direct`` a sim raylet has to join the GCS, and
    Ray compares python at patch level. Under ``gateway`` there is no sim raylet, and the gate
    built on this list is what turns "nothing here should need Ray" from a design intention into
    a measured, enforced property -- which is the precondition for ever deleting the pin from
    ``docker/Dockerfile.behavior1k``.
    """
    return ("cosmos_framework.posttrain.roles.env.sim_server",)


#: The shared data contract. Zero-dependency by construction and imported by both halves of
#: the seam, which is what makes it safe to ship into an image that has neither torch nor Ray.
SCHEMA = ("metrics", "trajectory")

#: Backend modules the sim container needs, as ``embodied.`` submodule paths.
#:
#: ``policies.cosmos3`` is DELIBERATELY EXCLUDED and that exclusion is the point of the
#: exercise: it is the only thing in ``embodied/`` that drags in the i4 policy stack, and it
#: runs in the FRAMEWORK venv, on the other side of the seam. A surface that included it
#: would put torch in the Isaac image and there would be nothing left to decouple.
BACKENDS = ("sims.behavior1k", "sinks.null")

#: Sim-side modules that live OUTSIDE ``embodied/``, as ``posttrain.`` module paths.
#:
#: A LITERAL, not a walk of ``roles/env/`` -- that directory holds modules for BOTH venvs, so a
#: walk would assert the wrong thing in both directions (see :func:`sim_side_roots`). Entries are
#: appended by the commit that adds the module.
LOOP = (
    "roles.env.collector",
    "roles.env.sim_role",
    "roles.env.sim_server",
)

#: The FRAMEWORK-side half of the container split, listed so it is visibly NOT in ``LOOP``.
#:
#: ``sim_gateway`` is the one module in this pair that must never ship into the sim image: it
#: spawns the simulator and talks to the policy handle, so it belongs to the side that has Ray.
#: Naming it here rather than leaving it unmentioned is the point -- the two files sit next to
#: each other with nearly the same name, and "which one goes in the image" is exactly the
#: question a reader will get wrong.
GATEWAY = ("roles.env.sim_gateway",)


#: Framework subtrees the loop must never pull in, as ``posttrain.`` prefixes.
#:
#: **A DENY-LIST, AND THAT IS DELIBERATE.** An allowlist enumerating what the loop imports
#: today would name ``utils.dist``, ``utils.log``, ``workers.worker_group``, ``roles.core``
#: and ``config.rl`` as "the sim-container surface" -- freezing exactly the coupling the
#: decoupling exists to remove, and turning every step toward the target topology into a test
#: edit. The deny-list instead names what must never arrive, and lets the rest shrink.
FRAMEWORK_DENIED = (
    "model",
    "algorithm",
    "checkpoint",
    "comm",
    "data",
    "control",
    "roles.rollout",
    "roles.trainer",
)

#: Third-party packages that must not be needed to import the loop. Each is absent from the
#: Isaac venv, the framework venv, or both.
ENGINE_ABSENT = ("ray", "torch", "pydantic", "yaml", "omnigibson", "av")
