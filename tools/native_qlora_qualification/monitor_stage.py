# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""Serialize stage publication with a complete host/device observation.

This prevents new samples from spanning stage transitions. It does not change
how memory receipts are classified, recover failed reads, or repair old rows.
The prospective v11 collector may acquire host/device observations for 13.5
seconds; publishers allow 15 seconds and record the actual wait. Historical
receipts are unchanged. Blocking syscalls can still overrun; an observed sample
hold overrun fails before publication rather than claiming complete coverage.
"""

from __future__ import annotations

import fcntl
import json
import os
import stat
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

PROTOCOL = "cosmos.native-qlora-stage-lock.v1"
LEGACY_LOCK_TIMEOUT_SECONDS = 5.0
LOCK_TIMEOUT_SECONDS = 15.0


def _paths(marker: Path) -> tuple[Path, Path, bytes]:
    marker = Path(marker).absolute()
    if marker.is_symlink():
        raise ValueError("stage marker must not be a symlink")
    lock = marker.with_name(marker.name + ".lock")
    expected = (json.dumps({"protocol": PROTOCOL, "marker": str(marker.resolve())}, sort_keys=True) + "\n").encode()
    return marker, lock, expected


@contextmanager
def _locked(marker: Path, *, exclusive: bool, timeout: float):
    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or not 0 < timeout <= LOCK_TIMEOUT_SECONDS:
        raise ValueError("stage lock timeout must be positive and at most fifteen seconds")
    marker, lock, expected = _paths(marker)
    fd = os.open(lock, os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC)
    started = time.monotonic()
    try:
        while True:
            try:
                fcntl.flock(fd, (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() - started >= timeout:
                    raise TimeoutError("stage publication/sample lock timed out") from None
                time.sleep(min(0.01, timeout / 10))
        acquired = time.monotonic()
        if acquired - started > timeout:
            raise TimeoutError("stage publication/sample lock timed out")
        info = os.fstat(fd)
        current = lock.stat(follow_symlinks=False)
        if (
            not stat.S_ISREG(info.st_mode)
            or (info.st_dev, info.st_ino) != (current.st_dev, current.st_ino)
            or os.read(fd, len(expected) + 1) != expected
        ):
            raise ValueError("stage lock identity or protocol differs")
        receipt = {
            "protocol": PROTOCOL,
            "mode": "publication" if exclusive else "sample",
            "wait_seconds": acquired - started,
            "timeout_seconds": timeout,
            "acquired_monotonic_seconds": acquired,
        }
        yield receipt
        current = lock.stat(follow_symlinks=False)
        if (info.st_dev, info.st_ino) != (current.st_dev, current.st_ino):
            raise ValueError("stage lock was replaced while held")
        if not exclusive:
            finished = time.monotonic()
            receipt.update(finished_monotonic_seconds=finished, hold_seconds=finished - acquired)
            if finished - acquired > LOCK_TIMEOUT_SECONDS:
                raise TimeoutError("stage sample hold exceeded fifteen-second deadline")
    finally:
        # Closing also releases a held flock on exceptions or process exit.
        os.close(fd)


def publish_stage(marker: Path, name: str, *, initialize: bool = False, timeout: float = LOCK_TIMEOUT_SECONDS) -> dict:
    """Atomically publish one label, waiting for any in-flight sample to finish."""
    if not isinstance(name, str) or not name or name.strip() != name or "\n" in name or "\r" in name:
        raise ValueError("stage must be one nonempty line")
    marker, lock, expected = _paths(marker)
    if initialize:
        if marker.exists():
            raise FileExistsError(marker)
        with lock.open("xb") as stream:
            stream.write(expected)
    with _locked(marker, exclusive=True, timeout=timeout) as receipt:
        if not initialize and (not marker.is_file() or marker.is_symlink()):
            raise ValueError("existing coordinated stage marker is missing or unsafe")
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", dir=marker.parent, prefix=marker.name + ".", delete=False
            ) as stream:
                temporary = Path(stream.name)
                stream.write(name + "\n")
            temporary.replace(marker)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return {**receipt, "stage": name, "published_monotonic_seconds": time.monotonic()}


@contextmanager
def sample_stage(marker: Path, *, timeout: float = LOCK_TIMEOUT_SECONDS):
    """Hold a stable stage through host/NVML reads and the final stage check.

    The caller publishes its record only after this context exits successfully.
    """
    marker = Path(marker)
    with _locked(marker, exclusive=False, timeout=timeout) as receipt:
        before = marker.read_bytes()
        if not before.strip() or not before.endswith(b"\n") or len(before.splitlines()) != 1:
            raise ValueError("stage marker is empty or malformed")
        yield receipt
        if marker.is_symlink() or marker.read_bytes() != before:
            raise ValueError("uncoordinated stage marker changed during sample")
