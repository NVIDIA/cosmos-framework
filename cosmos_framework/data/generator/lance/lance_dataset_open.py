# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Retry transient object-store failures while opening Lance datasets."""

from __future__ import annotations

import random
import re
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

from cosmos_framework.utils import log

if TYPE_CHECKING:
    import lance

_T = TypeVar("_T")
_MAX_IO_ATTEMPTS = 5
_RETRYABLE_HTTP_STATUS = re.compile(
    r"\b(?:http(?:\s+status)?|status(?:\s+code)?)\s*[:=]?\s*(?:429|503)\b",
    flags=re.IGNORECASE,
)
_RETRYABLE_IO_PHRASES = (
    "http error: error sending request",
    "request body error",
    "response body error",
    "timed out",
    "deadline exceeded",
    "timeout error",
    "request timeout",
    "operation timeout",
    "connection timeout",
)


def _is_retryable_lance_io_error(error: Exception) -> bool:
    """Return whether ``error`` is an explicitly transient Lance I/O failure."""
    message = str(error)
    if "lanceerror(io)" not in message.lower():
        return False
    if _RETRYABLE_HTTP_STATUS.search(message) is not None:
        return True
    normalized_message = message.lower()
    return any(phrase in normalized_message for phrase in _RETRYABLE_IO_PHRASES)


def _retry_delay_seconds(retry_number: int) -> float:
    """Return jittered exponential delay for retry numbers one through four."""
    lower_bound = float(2 ** (retry_number - 1))
    return random.SystemRandom().uniform(lower_bound, lower_bound * 2)


def run_lance_io_with_retry(operation: Callable[[], _T], *, description: str) -> _T:
    """Run one Lance I/O operation with bounded retries for known transient failures."""
    for attempt in range(1, _MAX_IO_ATTEMPTS + 1):
        try:
            return operation()
        except Exception as error:
            if attempt == _MAX_IO_ATTEMPTS or not _is_retryable_lance_io_error(error):
                raise
            delay_seconds = _retry_delay_seconds(attempt)
            log.warning(
                f"[Lance] Transient {description} failure on attempt "
                f"{attempt}/{_MAX_IO_ATTEMPTS}; retrying in {delay_seconds:.1f}s: {error}",
                rank0_only=False,
            )
            time.sleep(delay_seconds)

    raise AssertionError("Lance I/O retry loop exited unexpectedly.")


def open_lance_dataset_with_retry(
    uri: str,
    *,
    storage_options: dict[str, str],
    version: int | None = None,
    metadata_cache_size_bytes: int | None = None,
) -> lance.LanceDataset:
    """Open a Lance dataset, dispersing retries for transient object-store errors."""
    import lance

    cache_options = (
        {} if metadata_cache_size_bytes is None else {"metadata_cache_size_bytes": metadata_cache_size_bytes}
    )
    return run_lance_io_with_retry(
        lambda: lance.dataset(uri, storage_options=storage_options, version=version, **cache_options),
        description=f"dataset-open for {uri!r}",
    )
