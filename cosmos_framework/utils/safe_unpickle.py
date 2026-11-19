# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Restricted unpickling for checkpoint state that must stay plain data.

A pickle can only call code by naming a global (``module.name``) that the
unpickler resolves through :meth:`pickle.Unpickler.find_class`. Restricting
that lookup to an explicit allow-list keeps loading to data reconstruction:
with an empty allow-list only builtin containers and scalars (dict, list,
tuple, str, int, float, bool, None) can be produced.
"""

import io
import pickle
from typing import Any


class ForbiddenPickleGlobalError(pickle.UnpicklingError):
    """Raised when a pickle references a global that is not on the allow-list."""


class _AllowListUnpickler(pickle.Unpickler):
    def __init__(self, file: io.BytesIO, allowed_globals: frozenset[tuple[str, str]]) -> None:
        super().__init__(file)
        self._allowed_globals = allowed_globals

    def find_class(self, module: str, name: str) -> Any:
        # Check the raw names before ``super().find_class`` applies any protocol-2 import remapping.
        if (module, name) not in self._allowed_globals:
            raise ForbiddenPickleGlobalError(f"forbidden global {module}.{name}")
        return super().find_class(module, name)


def safe_loads(data: bytes, allowed_globals: frozenset[tuple[str, str]] = frozenset()) -> Any:
    """Unpickle ``data``, resolving only the ``(module, name)`` globals in ``allowed_globals``.

    Raises:
        ForbiddenPickleGlobalError: ``data`` references a global outside ``allowed_globals``.
        pickle.UnpicklingError: ``data`` is not a valid pickle.
    """
    return _AllowListUnpickler(io.BytesIO(data), allowed_globals).load()
