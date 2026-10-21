# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Default-on, low-frequency diagnostics for stalled Lance sample production."""

import os
import threading
import time
from _thread import LockType
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

from torch.utils.data import get_worker_info

from cosmos_framework.utils import log

_WARNING_SECONDS = 60.0
_POLL_SECONDS = 10.0


def _identity() -> str:
    worker = get_worker_info()
    return f"rank={os.environ.get('RANK', '?')} worker={worker.id if worker is not None else 'main'} pid={os.getpid()}"


def _safe_url(url: str) -> str:
    parts = urlsplit(url)
    # Never include signed query parameters or embedded credentials.
    return urlunsplit((parts.scheme, parts.netloc.rsplit("@", 1)[-1], parts.path, "", ""))


@dataclass
class _Operation:
    stage: str
    label: str
    started: float
    warned: bool = False


class _Watchdog:
    def __init__(self) -> None:
        self.pid: int = os.getpid()
        self.lock: LockType = threading.Lock()
        self.active: dict[int, _Operation] = {}
        self.last_warning: float = float("-inf")

    def report(self) -> None:
        now = time.monotonic()
        with self.lock:
            if now - self.last_warning < _WARNING_SECONDS:
                return
            slow = [op for op in self.active.values() if now - op.started >= _WARNING_SECONDS]
            if not slow:
                return
            self.last_warning = now
            # Prefer artifact URLs over their enclosing fetch/materialization stages.
            slow.sort(key=lambda op: (op.stage != "download", op.started))
            for op in slow[:5]:
                op.warned = True
            details = "; ".join(f"{op.label} elapsed_s={now - op.started:.1f}" for op in slow[:5])
            log.warning(
                f"[LanceProgress] {len(slow)} operations still running (showing up to 5): {details}",
                rank0_only=False,
            )

    def run(self) -> None:
        while True:
            time.sleep(_POLL_SECONDS)
            self.report()


_watchdog: _Watchdog | None = None
_watchdog_lock = threading.Lock()


@contextmanager
def watch_operation(stage: str, *, source: str, sample_id: str = "", url: str = "") -> Iterator[None]:
    """Report a blocked call before it returns; keep strings only, never sample tensors."""
    global _watchdog
    with _watchdog_lock:
        if _watchdog is None or _watchdog.pid != os.getpid():
            _watchdog = _Watchdog()
            threading.Thread(target=_watchdog.run, name="lance-progress", daemon=True).start()
        watchdog = _watchdog
    op = _Operation(
        stage=stage,
        label=f"{_identity()} source={source} stage={stage} sample={sample_id} url={_safe_url(url)}",
        started=time.monotonic(),
    )
    with watchdog.lock:
        watchdog.active[id(op)] = op
    completed = False
    try:
        yield
        completed = True
    finally:
        with watchdog.lock:
            watchdog.active.pop(id(op))
        if op.warned:
            log.warning(
                f"[LanceProgress] Slow operation {'completed' if completed else 'failed'}: "
                f"{op.label} elapsed_s={time.monotonic() - op.started:.1f}",
                rank0_only=False,
            )


class RejectionProgress:
    """Summarize rejected attempts, excluding time a mixed stream sits idle."""

    def __init__(self, source: str) -> None:
        self.source: str = source
        self.reasons: Counter[str] = Counter()
        self.rejected: int = 0
        self.active_seconds: float = 0.0
        self.last_warning: float = float("-inf")
        self.lock: LockType = threading.Lock()

    def record(self, reason: str | Counter[str] | None, elapsed: float, sample_id: str) -> None:
        with self.lock:
            if reason is None:
                self.reasons.clear()
                self.rejected = 0
                self.active_seconds = 0.0
                return
            self.rejected += 1
            self.reasons.update([reason] if isinstance(reason, str) else reason)
            self.active_seconds += elapsed
            now = time.monotonic()
            if self.active_seconds < _WARNING_SECONDS or now - self.last_warning < _WARNING_SECONDS:
                return
            self.last_warning = now
            log.warning(
                f"[LanceProgress] Repeated rejection: {_identity()} source={self.source} "
                f"rejected={self.rejected} reasons={dict(self.reasons)} "
                f"active_s={self.active_seconds:.1f} last_sample={sample_id}",
                rank0_only=False,
            )
