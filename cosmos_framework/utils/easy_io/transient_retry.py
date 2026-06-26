# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Shared retry helpers for transient object-store / transport errors.

This module is the single source of truth for what counts as a *transient*
(retryable) error when talking to an object store, and a small synchronous
``retry_on_transient_error`` wrapper around individual operations (e.g. an
``easy_io`` ``get``/``exists`` call).

It lives under ``easy_io`` -- the lowest-level storage layer -- so any caller
(datasets, ``ObjectStore``, etc.) can depend on it without creating a reverse
dependency on the higher-level dataset code.
"""

from __future__ import annotations

import time
from http.client import IncompleteRead
from typing import Callable, TypeVar

from botocore.exceptions import (
    ConnectionClosedError,
    EndpointConnectionError,
    ResponseStreamingError,
)
from botocore.exceptions import (
    ReadTimeoutError as BotocoreReadTimeoutError,
)
from multistorageclient.types import RetryableError
from urllib3.exceptions import ProtocolError as URLLib3ProtocolError
from urllib3.exceptions import ReadTimeoutError as URLLib3ReadTimeoutError
from urllib3.exceptions import SSLError as URLLib3SSLError

from cosmos_framework.utils import log

__all__ = [
    "RETRYABLE_EXCEPTIONS",
    "is_retryable_exception",
    "retry_on_transient_error",
]

T = TypeVar("T")

# Exceptions that indicate a *transient* transport failure and are worth
# retrying. These are connection/timeout/SSL/protocol errors -- never logical
# errors (e.g. a missing key, bad credentials, or a programming bug).
RETRYABLE_EXCEPTIONS = (
    # built-in
    IOError,
    # http
    IncompleteRead,
    # urllib3
    URLLib3ReadTimeoutError,
    URLLib3ProtocolError,
    URLLib3SSLError,
    # AWS SDK for Python (boto)
    BotocoreReadTimeoutError,
    ConnectionClosedError,
    EndpointConnectionError,
    ResponseStreamingError,
    # NVIDIA Multi-Storage Client (MSC)
    RetryableError,
)

# Default retry policy. Tuned for transient SSL / connection resets seen on
# long-running training jobs: a handful of attempts with exponential backoff
# is enough to ride out a brief blip without masking a real outage for long.
DEFAULT_MAX_RETRIES = 5
DEFAULT_BASE_DELAY_S = 0.5


def is_retryable_exception(exc: BaseException) -> bool:
    """Return True if ``exc`` -- or any exception in its cause/context chain -- is retryable.

    Object-store clients frequently re-wrap a transient transport error (e.g. an
    ``SSLError``) inside an opaque, non-retryable exception. Walking the
    ``__cause__``/``__context__`` chain lets us still recognise it.
    """
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        if isinstance(exc, RETRYABLE_EXCEPTIONS):
            return True
        seen.add(id(exc))
        exc = exc.__cause__ or exc.__context__
    return False


def retry_on_transient_error(
    func: Callable[[], T],
    operation: str,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY_S,
) -> T:
    """Call ``func`` and retry it on transient transport errors with exponential backoff.

    Args:
        func: Zero-argument callable performing a single object-store operation.
        operation: Human-readable label for logging (e.g. ``"load_object(foo.pt)"``).
        max_retries: Total number of attempts (must be >= 1).
        base_delay: Base delay in seconds; attempt ``i`` (0-indexed) sleeps
            ``base_delay * 2**i`` before the next try.

    Returns:
        Whatever ``func`` returns on the first successful attempt.

    Raises:
        The last exception if all attempts fail, or immediately if a raised
        exception is not transient (see :func:`is_retryable_exception`).
    """
    assert max_retries >= 1, f"max_retries must be >= 1, got {max_retries}"

    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            # Re-raise anything that is not a transient transport error, unchanged.
            if not is_retryable_exception(e):
                raise
            # Out of retries: re-raise the last (transient) exception.
            if attempt == max_retries - 1:
                log.warning(
                    f"[{operation}] {type(e).__name__}: {e} -- giving up after {max_retries} attempts",
                    rank0_only=False,
                )
                raise
            delay = base_delay * 2**attempt
            log.warning(
                f"[{operation}] {type(e).__name__}: {e} -- retry {attempt + 1}/{max_retries} in {delay:.1f}s",
                rank0_only=False,
            )
            time.sleep(delay)

    # Unreachable: the loop either returns or raises on the final attempt.
    raise AssertionError("retry_on_transient_error exited its loop without returning or raising")
