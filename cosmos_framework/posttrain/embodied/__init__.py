# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Engine-agnostic embodied rollout core.

Everything simulator-specific sits behind :class:`sim.Simulator` / :class:`sim.SimSession`.
This commit lands that seam and the work-unit vocabulary it speaks (:mod:`unit`); the rollout
loop, the policy and sink abstractions and the backends arrive on top of it.

**Why this is a top-level package and not a subtree of ``roles/``.** Every module here --
excluding the backend subpackages added later -- imports **stdlib, numpy and ``schema`` only**.
That is what lets the same core load in the Isaac venv, the policy venv and the trainer venv
alike, and the Isaac venv deliberately carries neither Ray nor torch. ``roles/`` cannot hold
it, and for a reason that is structural rather than incidental: ``roles/`` is the framework's
ROLE TAXONOMY, and this is a shared library that two different venvs load. The acute symptom
was that ``roles/env/__init__.py`` eagerly re-exported ``wfm`` -- which imports torch at module
scope -- so importing anything under ``roles.env`` dragged the training stack in. That barrel
is lazy as of this commit and a gate keeps it lazy, but the fix is one someone can undo in a
line, whereas a directory boundary is not. Both raylets already
export ``PYTHONPATH="${IMAGINAIRE_ROOT}"`` and ``posttrain/__init__.py`` is already PEP-562
lazy, so this needs no packaging work -- only the discipline, which
``tests/test_import_discipline.py`` enforces by simulating the missing modules rather than by
convention.

This package is deliberately NOT re-exporting its submodules here: an eager
``from . import sim`` would drag numpy into a process that only wanted
:class:`unit.EpisodeKey`, and the point of the layer is that it costs nothing to import.
"""
