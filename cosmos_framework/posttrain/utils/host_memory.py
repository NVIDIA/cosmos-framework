# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Host-memory headroom of this process and its job cgroup: the ONE reader behind every host-memory log line.

A colocated online job runs every actor inside one Slurm job cgroup (``ConstrainRAMSpace=yes``):
the number that decides whether the kernel kills an actor is the cgroup's usage against its limit,
which counts anonymous memory AND the page cache, not this process's resident set alone. This module
reads both from ``/proc`` and ``/sys/fs/cgroup`` (v2, then the v1 memory controller) and falls back
to ``/proc/meminfo`` when no cgroup limit is readable. Everything is best-effort: a missing file
yields ``None`` for that field and never raises, so a save path can log it unconditionally.

Every host-memory diagnostic in posttrain goes through :func:`host_memory` / :func:`log_host_memory`
and is formatted by :meth:`HostMemory.format`: the checkpoint manager's save and finalize lines, the
LLM trainer's pre-gather line and its consolidated-resume lines. Keeping one reader, one return type
and one formatting path is what makes those series comparable; no other module reads ``/proc`` for
memory figures.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from cosmos_framework.posttrain.utils.log import logger

_GIB = float(2**30)


@dataclass(frozen=True)
class HostMemory:
    """One reading; every field is GiB, ``None`` when its source was unreadable or unlimited."""

    rss_gib: float | None
    cgroup_used_gib: float | None
    cgroup_limit_gib: float | None
    cgroup_source: str | None
    mem_total_gib: float | None
    mem_available_gib: float | None

    def format(self) -> str:
        """Compact ``key=value`` line for a log record."""
        parts = [f"rss={_fmt(self.rss_gib)}GiB"]
        if self.cgroup_used_gib is not None:
            limit = f"/{_fmt(self.cgroup_limit_gib)}GiB" if self.cgroup_limit_gib is not None else "GiB/unlimited"
            pct = f" ({100.0 * self.cgroup_used_gib / self.cgroup_limit_gib:.1f}%)" if self.cgroup_limit_gib else ""
            parts.append(f"cgroup({self.cgroup_source})={_fmt(self.cgroup_used_gib)}{limit}{pct}")
        if self.mem_total_gib is not None:
            parts.append(f"node_available={_fmt(self.mem_available_gib)}/{_fmt(self.mem_total_gib)}GiB")
        return " ".join(parts)


def _fmt(value: float | None) -> str:
    return "?" if value is None else f"{value:.1f}"


def _read_text(path: str) -> str | None:
    try:
        with open(path, encoding="ascii", errors="replace") as handle:
            return handle.read()
    except OSError:
        return None


def _read_bytes_value(path: str) -> float | None:
    """A single integer byte count in GiB; ``None`` when unreadable, non-numeric, or unlimited."""
    text = _read_text(path)
    if text is None:
        return None
    token = text.strip().split()[0] if text.strip() else ""
    if token in ("", "max"):
        return None
    try:
        value = int(token)
    except ValueError:
        return None
    # cgroup v1 reports "no limit" as a very large number (PAGE_COUNTER_MAX * PAGE_SIZE).
    if value >= 2**60:
        return None
    return value / _GIB


def _meminfo_kib(text: str, key: str) -> float | None:
    for line in text.splitlines():
        name, _, rest = line.partition(":")
        if name.strip() == key:
            try:
                return int(rest.split()[0]) * 1024 / _GIB
            except (IndexError, ValueError):
                return None
    return None


def _cgroup_paths(proc_root: str) -> tuple[str | None, str | None]:
    """``(v2 path, v1 memory-controller path)`` of this process from ``/proc/self/cgroup``."""
    text = _read_text(os.path.join(proc_root, "self", "cgroup"))
    if text is None:
        return None, None
    v2_path: str | None = None
    v1_path: str | None = None
    for line in text.splitlines():
        fields = line.split(":", 2)
        if len(fields) != 3:
            continue
        hierarchy, controllers, path = fields
        if hierarchy == "0" and controllers == "":
            v2_path = path
        elif "memory" in controllers.split(","):
            v1_path = path
    return v2_path, v1_path


def _cgroup_memory(proc_root: str, cgroup_root: str) -> tuple[float | None, float | None, str | None]:
    """``(used, limit, source)`` for the innermost cgroup that exposes memory accounting."""
    v2_path, v1_path = _cgroup_paths(proc_root)
    if v2_path is not None:
        base = os.path.join(cgroup_root, v2_path.lstrip("/"))
        used = _read_bytes_value(os.path.join(base, "memory.current"))
        if used is not None:
            return used, _read_bytes_value(os.path.join(base, "memory.max")), "v2"
    if v1_path is not None:
        base = os.path.join(cgroup_root, "memory", v1_path.lstrip("/"))
        used = _read_bytes_value(os.path.join(base, "memory.usage_in_bytes"))
        if used is not None:
            return used, _read_bytes_value(os.path.join(base, "memory.limit_in_bytes")), "v1"
    return None, None, None


def host_memory(proc_root: str = "/proc", cgroup_root: str = "/sys/fs/cgroup") -> HostMemory:
    """Read this process's RSS, its cgroup's memory usage/limit, and the node's meminfo.

    ``proc_root`` and ``cgroup_root`` exist so a test can point the reader at a fixture tree; the
    defaults are the live kernel interfaces. Never raises: unreadable sources become ``None``.
    """
    status = _read_text(os.path.join(proc_root, "self", "status"))
    rss = _meminfo_kib(status, "VmRSS") if status is not None else None
    used, limit, source = _cgroup_memory(proc_root, cgroup_root)
    meminfo = _read_text(os.path.join(proc_root, "meminfo"))
    total = _meminfo_kib(meminfo, "MemTotal") if meminfo is not None else None
    available = _meminfo_kib(meminfo, "MemAvailable") if meminfo is not None else None
    return HostMemory(
        rss_gib=rss,
        cgroup_used_gib=used,
        cgroup_limit_gib=limit,
        cgroup_source=source,
        mem_total_gib=total,
        mem_available_gib=available,
    )


def log_host_memory(context: str, *, level: int = logging.INFO) -> HostMemory:
    """Log one ``host memory`` reading tagged with ``context`` and return it."""
    reading = host_memory()
    logger.log(level, f"host memory [{context}] {reading.format()}")
    return reading
