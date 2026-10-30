# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""CPU tests for the host/cgroup memory reader logged at checkpoint saves and the final drain.

The fixtures replay the colocated-job shape the reader exists for: a Slurm step cgroup whose
``memory.max`` is 900 GiB and whose usage reads what one production run peaked at, 923,320,576 KB.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from cosmos_framework.posttrain.utils import host_memory as host_memory_module
from cosmos_framework.posttrain.utils import residency
from cosmos_framework.posttrain.utils.host_memory import HostMemory, host_memory, log_host_memory

GIB = 2**30
STEP_CGROUP = "/system.slice/slurmstepd.scope/job_461919/step_0/user/task_0"
STATUS = "Name:\tpython\nVmPeak:\t  3145728 kB\nVmRSS:\t  2097152 kB\nThreads:\t12\n"
MEMINFO = "MemTotal:       964689920 kB\nMemFree:        10485760 kB\nMemAvailable:   41943040 kB\n"
OBSERVED_PEAK_BYTES = 923_320_576 * 1024
LIMIT_900_GIB = 900 * GIB
V1_UNLIMITED = 9223372036854771712  # PAGE_COUNTER_MAX * PAGE_SIZE, what cgroup v1 reports for "no limit"


def _proc(root: Path, *, cgroup: str, status: str | None = STATUS, meminfo: str | None = MEMINFO) -> str:
    proc = root / "proc"
    (proc / "self").mkdir(parents=True)
    if status is not None:
        (proc / "self" / "status").write_text(status, encoding="ascii")
    (proc / "self" / "cgroup").write_text(cgroup, encoding="ascii")
    if meminfo is not None:
        (proc / "meminfo").write_text(meminfo, encoding="ascii")
    return str(proc)


def _cgroup_v2(root: Path, path: str, *, current: int, maximum: str) -> str:
    base = root / "cgroup" / path.lstrip("/")
    base.mkdir(parents=True, exist_ok=True)
    (base / "memory.current").write_text(f"{current}\n", encoding="ascii")
    (base / "memory.max").write_text(f"{maximum}\n", encoding="ascii")
    return str(root / "cgroup")


def _cgroup_v1(root: Path, path: str, *, usage: int, limit: int) -> str:
    base = root / "cgroup" / "memory" / path.lstrip("/")
    base.mkdir(parents=True, exist_ok=True)
    (base / "memory.usage_in_bytes").write_text(f"{usage}\n", encoding="ascii")
    (base / "memory.limit_in_bytes").write_text(f"{limit}\n", encoding="ascii")
    return str(root / "cgroup")


@pytest.mark.L0
@pytest.mark.CPU
def test_host_memory_reads_cgroup_v2_usage_and_limit(tmp_path: Path) -> None:
    proc = _proc(tmp_path, cgroup=f"0::{STEP_CGROUP}\n")
    cgroup = _cgroup_v2(tmp_path, STEP_CGROUP, current=OBSERVED_PEAK_BYTES, maximum=str(LIMIT_900_GIB))

    reading = host_memory(proc_root=proc, cgroup_root=cgroup)

    assert reading.rss_gib == 2.0
    assert reading.cgroup_source == "v2"
    assert reading.cgroup_used_gib == pytest.approx(OBSERVED_PEAK_BYTES / GIB)
    assert reading.cgroup_limit_gib == 900.0
    assert reading.mem_total_gib == pytest.approx(920.0, abs=1e-6)
    assert reading.mem_available_gib == 40.0
    line = reading.format()
    assert line == "rss=2.0GiB cgroup(v2)=880.5/900.0GiB (97.8%) node_available=40.0/920.0GiB"


@pytest.mark.L0
@pytest.mark.CPU
def test_host_memory_treats_unlimited_v2_and_v1_limits_as_none(tmp_path: Path) -> None:
    v2_root = tmp_path / "v2"
    proc = _proc(v2_root, cgroup=f"0::{STEP_CGROUP}\n")
    cgroup = _cgroup_v2(v2_root, STEP_CGROUP, current=4 * GIB, maximum="max")
    unlimited_v2 = host_memory(proc_root=proc, cgroup_root=cgroup)
    assert (unlimited_v2.cgroup_used_gib, unlimited_v2.cgroup_limit_gib, unlimited_v2.cgroup_source) == (
        4.0,
        None,
        "v2",
    )
    assert "cgroup(v2)=4.0GiB/unlimited" in unlimited_v2.format() and "%" not in unlimited_v2.format()

    v1_root = tmp_path / "v1"
    proc = _proc(v1_root, cgroup=f"12:memory:{STEP_CGROUP}\n11:cpu,cpuacct:/\n")
    cgroup = _cgroup_v1(v1_root, STEP_CGROUP, usage=6 * GIB, limit=V1_UNLIMITED)
    unlimited_v1 = host_memory(proc_root=proc, cgroup_root=cgroup)
    assert (unlimited_v1.cgroup_used_gib, unlimited_v1.cgroup_limit_gib, unlimited_v1.cgroup_source) == (
        6.0,
        None,
        "v1",
    )


@pytest.mark.L0
@pytest.mark.CPU
def test_host_memory_falls_back_from_v2_to_v1_to_meminfo(tmp_path: Path) -> None:
    # Hybrid hierarchy: the v2 path exists but exposes no memory controller, v1 does.
    hybrid = tmp_path / "hybrid"
    proc = _proc(hybrid, cgroup=f"12:memory:{STEP_CGROUP}\n0::/user.slice/session.scope\n")
    (hybrid / "cgroup" / "user.slice" / "session.scope").mkdir(parents=True)
    cgroup = _cgroup_v1(hybrid, STEP_CGROUP, usage=OBSERVED_PEAK_BYTES, limit=LIMIT_900_GIB)
    reading = host_memory(proc_root=proc, cgroup_root=cgroup)
    assert reading.cgroup_source == "v1"
    assert reading.cgroup_used_gib == pytest.approx(OBSERVED_PEAK_BYTES / GIB) and reading.cgroup_limit_gib == 900.0

    # No readable cgroup at all: the node-wide meminfo figures are what remains.
    bare = tmp_path / "bare"
    proc = _proc(bare, cgroup=f"0::{STEP_CGROUP}\n")
    reading = host_memory(proc_root=proc, cgroup_root=str(bare / "no-cgroup-mount"))
    assert (reading.cgroup_used_gib, reading.cgroup_limit_gib, reading.cgroup_source) == (None, None, None)
    assert reading.rss_gib == 2.0 and reading.mem_available_gib == 40.0
    assert reading.format() == "rss=2.0GiB node_available=40.0/920.0GiB"


@pytest.mark.L0
@pytest.mark.CPU
def test_host_memory_never_raises_on_missing_or_malformed_sources(tmp_path: Path) -> None:
    empty = host_memory(proc_root=str(tmp_path / "absent"), cgroup_root=str(tmp_path / "absent"))
    assert empty == HostMemory(None, None, None, None, None, None)
    assert empty.format() == "rss=?GiB"

    garbled = tmp_path / "garbled"
    proc = _proc(garbled, cgroup="not a cgroup line\n0::/x\n", status="VmRSS: lots kB\n", meminfo="MemTotal: ?\n")
    cgroup = _cgroup_v2(garbled, "/x", current=0, maximum="")
    (garbled / "cgroup" / "x" / "memory.current").write_text("banana\n", encoding="ascii")
    reading = host_memory(proc_root=proc, cgroup_root=cgroup)
    assert reading == HostMemory(None, None, None, None, None, None)


@pytest.mark.L0
@pytest.mark.CPU
def test_log_host_memory_emits_one_tagged_record(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    fixed = HostMemory(2.0, 880.6, 900.0, "v2", 920.0, 40.0)
    monkeypatch.setattr(host_memory_module, "host_memory", lambda: fixed)
    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    # The posttrain logger does not propagate, so capture on it directly at the level the loop logs at.
    handler = _Capture(level=logging.DEBUG)
    host_memory_module.logger.addHandler(handler)
    try:
        with caplog.at_level(logging.INFO, logger=host_memory_module.logger.name):
            assert log_host_memory("final checkpoint drain model_step=300") is fixed
    finally:
        host_memory_module.logger.removeHandler(handler)
    assert [(record.levelno, record.getMessage()) for record in records] == [
        (logging.INFO, f"host memory [final checkpoint drain model_step=300] {fixed.format()}"),
    ]

    # The live kernel interfaces are readable on the CI host; nothing here may raise.
    live = host_memory()
    assert live.rss_gib is None or live.rss_gib > 0.0


@pytest.mark.L0
@pytest.mark.CPU
def test_host_memory_is_the_single_reader_and_residency_exposes_no_second_one() -> None:
    """One host-memory mechanism: the checkpoint save/finalize lines (checkpoint/manager.py), the trainer's
    pre-gather line and its consolidated-resume lines (roles/trainer/llm.py) all read through
    `utils.host_memory` and are formatted by `HostMemory.format`. Before, `utils.residency.host_memory_gib()`
    was a second, independent `/proc` reader with its own return type and its own formatting, and the two
    could drift. Guarded here at the source: no other posttrain module reads the kernel memory interfaces."""
    assert not hasattr(residency, "host_memory_gib"), "the second host-memory reader is gone"
    package = Path(host_memory_module.__file__).resolve().parents[1]
    assert package.name == "posttrain"
    # Quoted literals: a Python reader of the kernel memory interfaces in this process. (A launcher's embedded
    # remote shell snippet that awks /proc/meminfo on another node is not one and does not match.)
    kernel_memory_literals = tuple(
        f'"{token}"' for token in ("/proc/meminfo", "/proc/self/status", "VmRSS", "MemAvailable", "memory.current")
    )
    readers = sorted(
        str(source.relative_to(package))
        for source in package.rglob("*.py")
        if "tests" not in source.relative_to(package).parts
        and source.name != "host_memory.py"
        and any(literal in source.read_text(encoding="utf-8", errors="replace") for literal in kernel_memory_literals)
    )
    assert readers == [], f"host memory is read outside utils/host_memory.py: {readers}"

    # The two trainer lines and the manager's lines are the same component: same reader, same formatter.
    from cosmos_framework.posttrain.checkpoint import manager as manager_module
    from cosmos_framework.posttrain.roles.trainer import llm as llm_module

    assert manager_module.log_host_memory is host_memory_module.log_host_memory
    assert llm_module.log_host_memory is host_memory_module.log_host_memory
