# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Import a dependency that has no Windows wheel without failing at import time there.

``pyarrow`` has no ``win_arm64`` wheel, so a module-scope ``import pyarrow`` makes every module
that transitively reaches it unimportable on Windows on Arm (RTX Spark), including the
reasoner config graph. On Windows :func:`optional_import` returns the real module where it is
installed and a stand-in otherwise. The stand-in raises ``AttributeError`` on first attribute
access, naming the module and chained to the original ``ImportError``, so code paths that need
the dependency fail loudly where they use it instead of at import, and every other path keeps
working. ``AttributeError`` is what the attribute protocol expects: ``hasattr`` and
three-argument ``getattr`` on the stand-in answer instead of raising.

On every other platform :func:`optional_import` is a plain import: a missing or broken install
raises ``ImportError`` at import time, as ``import pyarrow`` always did.
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType
from typing import Any


class MissingModule:
    """Stand-in for a module that failed to import; any attribute access raises."""

    def __init__(self, name: str, error: ImportError):
        self._name = name
        self._error = error

    def __getattr__(self, attr: str) -> Any:
        raise AttributeError(
            f"{self._name} is required for this code path but could not be imported ({self._error}). "
            "Install it, or avoid the feature that needs it on this platform."
        ) from self._error

    def __bool__(self) -> bool:
        return False

    def __repr__(self) -> str:
        return f"<MissingModule {self._name}>"


def optional_import(name: str) -> ModuleType | MissingModule:
    """Return module ``name``; on Windows only, a :class:`MissingModule` if it does not import."""
    try:
        return importlib.import_module(name)
    except ImportError as error:  # ModuleNotFoundError and failed native extensions alike
        if sys.platform != "win32":
            raise
        return MissingModule(name, error)
