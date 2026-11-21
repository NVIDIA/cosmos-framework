# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""Real process coordination tests; stub NVML tests scheduling, never GPU evidence."""

import json
import select
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tools.native_qlora_qualification import monitor_stage as stage

MODULE_ROOT = Path(stage.__file__).resolve().parents[2]


@pytest.fixture
def marker(tmp_path):
    value = tmp_path / "current-stage.txt"
    stage.publish_stage(value, "training", initialize=True)
    return value


def start(code, *args):
    return subprocess.Popen(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-c",
            f"import sys;sys.path.insert(0,{str(MODULE_ROOT)!r});" + code,
            *map(str, args),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def ready(child):
    assert select.select([child.stdout], [], [], 5)[0], "child failed to reach its bounded barrier"
    assert child.stdout.readline() == b"ready\n"


def finish(child):
    stdout, stderr = child.communicate(timeout=5)
    assert child.returncode == 0, (stdout, stderr)
    return stdout


def cleanup(child):
    if child.poll() is None:
        child.kill()
    child.communicate(timeout=5)


def test_atomic_publication_and_protocol(marker):
    proof = stage.publish_stage(marker, "portable_publication")
    assert proof["protocol"] == stage.PROTOCOL and proof["mode"] == "publication"
    assert proof["published_monotonic_seconds"] >= proof["acquired_monotonic_seconds"]
    assert marker.read_bytes() == b"portable_publication\n"
    with stage.sample_stage(marker) as sample:
        assert sample["protocol"] == stage.PROTOCOL and sample["mode"] == "sample"
    with pytest.raises(FileExistsError):
        stage.publish_stage(marker, "initialization", initialize=True)


def test_writer_waits_for_whole_sample(marker):
    child = start(
        "from pathlib import Path;from tools.native_qlora_qualification.monitor_stage import publish_stage;"
        "print('ready',flush=True);publish_stage(Path(sys.argv[1]),'portable_publication');print('published',flush=True)",
        marker,
    )
    try:
        with stage.sample_stage(marker):
            ready(child)
            assert not select.select([child.stdout], [], [], 0.1)[0]
            assert marker.read_text() == "training\n"
        assert finish(child) == b"published\n"
        assert marker.read_text() == "portable_publication\n"
    finally:
        cleanup(child)


@pytest.mark.parametrize("exclusive", [False, True])
def test_process_death_releases_lock(marker, exclusive):
    code = (
        "from pathlib import Path;from tools.native_qlora_qualification.monitor_stage import _locked;"
        f"\nwith _locked(Path(sys.argv[1]),exclusive={exclusive!r},timeout=1):"
        "\n print('ready',flush=True);sys.stdin.read(1)"
    )
    child = start(code, marker)
    try:
        ready(child)
        child.kill()
        child.wait(timeout=5)
        assert stage.publish_stage(marker, "after_exit", timeout=0.5)["stage"] == "after_exit"
        with stage.sample_stage(marker, timeout=0.5):
            assert marker.read_text() == "after_exit\n"
    finally:
        cleanup(child)


def test_timeout_never_changes_stage(marker):
    child = start(
        "from pathlib import Path;from tools.native_qlora_qualification.monitor_stage import sample_stage;"
        "\nwith sample_stage(Path(sys.argv[1])):\n print('ready',flush=True);sys.stdin.read(1)",
        marker,
    )
    try:
        ready(child)
        before = time.monotonic()
        with pytest.raises(TimeoutError):
            stage.publish_stage(marker, "forbidden", timeout=0.05)
        assert time.monotonic() - before < 1
        assert marker.read_text() == "training\n"
    finally:
        cleanup(child)


@pytest.mark.parametrize("timeout", [0, -1, 16, True, float("nan"), float("inf")])
def test_invalid_timeouts_fail(marker, timeout):
    with pytest.raises(ValueError):
        with stage.sample_stage(marker, timeout=timeout):
            pytest.fail("invalid timeout entered collection")


def test_missing_marker_and_uncoordinated_legacy_marker_fail(tmp_path, marker):
    marker.unlink()
    with pytest.raises(FileNotFoundError):
        with stage.sample_stage(marker):
            pytest.fail("missing marker admitted")
    with pytest.raises(ValueError):
        stage.publish_stage(marker, "next")
    legacy = tmp_path / "legacy-stage.txt"
    legacy.write_text("training\n")
    with pytest.raises(FileNotFoundError):
        with stage.sample_stage(legacy):
            pytest.fail("uncoordinated legacy marker admitted")


@pytest.mark.parametrize("malformation", ["protocol", "wrong_marker", "symlink", "replaced", "empty"])
def test_unsafe_or_replaced_lock_fails(marker, malformation):
    lock = marker.with_name(marker.name + ".lock")
    if malformation == "protocol":
        value = json.loads(lock.read_text())
        value["protocol"] = "other"
        lock.write_text(json.dumps(value))
    elif malformation == "wrong_marker":
        value = json.loads(lock.read_text())
        value["marker"] += "_other"
        lock.write_text(json.dumps(value))
    elif malformation == "symlink":
        target = lock.with_suffix(".real")
        lock.rename(target)
        lock.symlink_to(target)
    elif malformation == "empty":
        marker.write_text("")
    with pytest.raises((ValueError, OSError)):
        with stage.sample_stage(marker):
            if malformation == "replaced":
                old = lock.read_bytes()
                lock.unlink()
                lock.write_bytes(old)
            else:
                pytest.fail("unsafe marker/lock admitted")


def test_uncoordinated_write_fails_before_caller_publication(marker):
    published = []
    with pytest.raises(ValueError):
        with stage.sample_stage(marker):
            marker.write_text("foreign_writer\n")
        published.append("must not publish")
    assert published == []


def test_actual_host_collection_and_stub_nvml_share_one_stage(marker, tmp_path):
    root = start("payload=bytearray(1024*1024);print('ready',flush=True);sys.stdin.read(1)")
    monitor = writer = None
    try:
        ready(root)
        output, entered, release = (tmp_path / name for name in ("samples.jsonl", "nvml-entered", "nvml-release"))
        monitor_file = Path(stage.__file__).with_name("monitor.py")
        # Only device calls are stubbed. HostSampler reads a real, live owned PID;
        # the pause represents an NVML call occurring after host collection.
        code = """
import os,runpy,signal,types,time
from pathlib import Path
root,out,marker,entered,release,monitor=sys.argv[1:]
sys.path.insert(0,str(Path(monitor).parent))
n=types.ModuleType('pynvml');n.NVMLError=RuntimeError
n.nvmlInit=n.nvmlShutdown=lambda:None
n.nvmlDeviceGetHandleByUUID=lambda value:0
n.nvmlDeviceGetUUID=lambda _: 'GPU-12345678-1234-1234-1234-123456789abc'
def memory(_):
 Path(entered).write_text('ready')
 deadline=time.monotonic()+4
 while not Path(release).exists():
  assert time.monotonic()<deadline;time.sleep(.005)
 return types.SimpleNamespace(used=1,total=100)
n.nvmlDeviceGetMemoryInfo=memory
n.nvmlDeviceGetComputeRunningProcesses=lambda _:[types.SimpleNamespace(pid=int(root),usedGpuMemory=1)]
n.nvmlDeviceGetUtilizationRates=lambda _:types.SimpleNamespace(gpu=0,memory=0)
n.NVML_CLOCK_SM=0;n.NVML_CLOCK_MEM=1;n.NVML_TEMPERATURE_GPU=0
n.nvmlDeviceGetClockInfo=lambda *_:1;n.nvmlDeviceGetPowerUsage=lambda _:1
def temperature(*_):signal.raise_signal(signal.SIGTERM);return 1
n.nvmlDeviceGetTemperature=temperature
sys.modules['pynvml']=n;os.environ['COSMOS_QLORA_GPU_UUID']='GPU-12345678-1234-1234-1234-123456789abc'
sys.argv=[monitor,root,out,marker];runpy.run_path(monitor,run_name='__main__')
"""
        monitor = start(code, root.pid, output, marker, entered, release, monitor_file)
        deadline = time.monotonic() + 5
        while not entered.exists():
            assert monitor.poll() is None, monitor.stderr.read()
            assert time.monotonic() < deadline
            time.sleep(0.005)
        writer = start(
            "from pathlib import Path;from tools.native_qlora_qualification.monitor_stage import publish_stage;"
            "print('ready',flush=True);publish_stage(Path(sys.argv[1]),'portable_publication');print('published',flush=True)",
            marker,
        )
        ready(writer)
        assert not select.select([writer.stdout], [], [], 0.1)[0]
        assert marker.read_text() == "training\n"
        release.write_text("release")
        finish(monitor)
        assert finish(writer) == b"published\n"
        rows = [json.loads(line) for line in output.read_text().splitlines()]
        assert len(rows) == 1
        row = rows[0]
        assert row["stage"] == "training" and row["host_pss_complete"] is True
        assert row["stage_serialization"]["protocol"] == stage.PROTOCOL
        assert row["host_collection"]["stage_publish"]["value"] == "training"
        assert row["selected_host_pss_sum_bytes"] > 0
        assert marker.read_text() == "portable_publication\n"
    finally:
        for child in (writer, monitor, root):
            if child is not None:
                cleanup(child)
