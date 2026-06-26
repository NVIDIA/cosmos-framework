# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from unittest import mock

import pytest
from urllib3.exceptions import SSLError as URLLib3SSLError

from cosmos_framework.utils.easy_io.transient_retry import (
    is_retryable_exception,
    retry_on_transient_error,
)


@pytest.mark.L0
@pytest.mark.CPU
def test_is_retryable_exception_direct():
    # A transient transport error is retryable.
    assert is_retryable_exception(URLLib3SSLError("ssl handshake failed"))
    assert is_retryable_exception(IOError("connection reset"))
    # A logical/programming error is not.
    assert not is_retryable_exception(ValueError("bad value"))
    assert not is_retryable_exception(KeyError("missing"))


@pytest.mark.L0
@pytest.mark.CPU
def test_is_retryable_exception_walks_cause_chain():
    # An opaque wrapper that re-raises `from` a transient error is still retryable.
    try:
        try:
            raise URLLib3SSLError("ssl handshake failed")
        except URLLib3SSLError as inner:
            raise RuntimeError("opaque client error") from inner
    except RuntimeError as wrapped:
        assert is_retryable_exception(wrapped)

    # ...and via the implicit __context__ chain (raise during handling).
    try:
        try:
            raise URLLib3SSLError("ssl handshake failed")
        except URLLib3SSLError:
            raise RuntimeError("opaque client error")
    except RuntimeError as wrapped:
        assert is_retryable_exception(wrapped)


@pytest.mark.L0
@pytest.mark.CPU
def test_retry_returns_on_success_without_retrying():
    func = mock.Mock(return_value="ok")
    with mock.patch("cosmos_framework.utils.easy_io.transient_retry.time.sleep") as sleep:
        result = retry_on_transient_error(func, operation="op")
    assert result == "ok"
    assert func.call_count == 1
    sleep.assert_not_called()


@pytest.mark.L0
@pytest.mark.CPU
def test_retry_then_success():
    # Fail twice with a transient error, then succeed.
    func = mock.Mock(
        side_effect=[
            URLLib3SSLError("ssl"),
            URLLib3SSLError("ssl"),
            "ok",
        ]
    )
    with mock.patch("cosmos_framework.utils.easy_io.transient_retry.time.sleep") as sleep:
        result = retry_on_transient_error(func, operation="op", max_retries=5, base_delay=0.5)
    assert result == "ok"
    assert func.call_count == 3
    # Exponential backoff between the two failures: 0.5 * 2**0, then 0.5 * 2**1.
    assert [c.args[0] for c in sleep.call_args_list] == [0.5, 1.0]


@pytest.mark.L0
@pytest.mark.CPU
def test_non_retryable_reraised_immediately():
    func = mock.Mock(side_effect=ValueError("logical error"))
    with mock.patch("cosmos_framework.utils.easy_io.transient_retry.time.sleep") as sleep:
        with pytest.raises(ValueError, match="logical error"):
            retry_on_transient_error(func, operation="op", max_retries=5)
    # No retry for a non-transient error.
    assert func.call_count == 1
    sleep.assert_not_called()


@pytest.mark.L0
@pytest.mark.CPU
def test_retry_exhausted_reraises_last_exception():
    func = mock.Mock(side_effect=URLLib3SSLError("persistent ssl failure"))
    with mock.patch("cosmos_framework.utils.easy_io.transient_retry.time.sleep") as sleep:
        with pytest.raises(URLLib3SSLError, match="persistent ssl failure"):
            retry_on_transient_error(func, operation="op", max_retries=3, base_delay=0.5)
    assert func.call_count == 3
    # Slept before attempts 2 and 3, but not after the final failure.
    assert [c.args[0] for c in sleep.call_args_list] == [0.5, 1.0]
