# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Construction scope for inspecting model architecture during checkpoint conversion."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_architecture_only: ContextVar[bool] = ContextVar("architecture_only_model_construction", default=False)


def is_architecture_only_model_construction() -> bool:
    """Whether construction may skip execution-backend availability checks."""
    return _architecture_only.get()


@contextmanager
def architecture_only_model_construction() -> Iterator[None]:
    """Skip runtime backend checks for architecture inspection, without changing model config.

    Callers must separately arrange empty-weight construction and must not execute
    the resulting model. Normal training and inference must keep backend validation.
    """
    token = _architecture_only.set(True)
    try:
        yield
    finally:
        _architecture_only.reset(token)
