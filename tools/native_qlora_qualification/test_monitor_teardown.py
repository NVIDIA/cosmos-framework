# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""Prospective v9 delayed-exit evidence; retained v8 gaps stay incomplete."""

import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tools.native_qlora_qualification import monitor_host as host


def test_actual_retained_v8_gap_and_later_exit_keep_original_decisions():
    fixture = json.loads((Path(__file__).parent / "fixtures/host_monitor_v8_exit.json").read_text())
    original = copy.deepcopy(fixture)
    assert fixture["source_job"] == 38515136 and fixture["sample_indices"] == [472, 474]
    failed, later = fixture["samples"]
    assert failed["host_collection"]["max_collection_seconds"] == 0.75
    assert host.validate_sample(failed)["host_collection_errors"] == ["live_or_unknown_memory_unreadable"]
    assert not host.validate_sample(failed)["host_pss_complete"]
    assert host.validate_sample(later)["host_pss_complete"]
    assert fixture == original
    for sample in fixture["samples"]:
        sample["host_collection"]["protocol"] = sample["host_process_memory_method"] = host.TEARDOWN_METHOD
        with pytest.raises(ValueError, match="bounds"):
            host.validate_sample(sample)


def _child():
    return subprocess.Popen(
        [sys.executable, "-c", "import sys,time; print('ready',flush=True); sys.stdin.read(1); time.sleep(0.9)"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )


@pytest.mark.parametrize("exit_process", [True, False])
def test_v9_real_pidfd_delayed_exit_or_live_timeout(tmp_path, monkeypatch, request, exit_process):
    """Inject only unreadable proc memory; actual PIDFD proof comes from a real child."""
    process = _child()
    sampler = host.HostSampler(process.pid, tmp_path / "stage", method=host.TEARDOWN_METHOD)
    read, armed, released = host.read_proc, False, False
    try:
        assert process.stdout.readline() == b"ready\n"
        assert sampler.collect()["host_pss_complete"]  # Real memory/identity before the fault fixture.

        def unreadable_owned_memory(pid, name):
            nonlocal released
            result = read(pid, name)
            if armed and pid == process.pid and name in ("status", "smaps_rollup"):
                result.pop("value", None)
                result.pop("error", None)
                if name == "status":
                    result["value"] = {}
                else:
                    result["error"] = {"type": "ProcessLookupError", "errno": 3}
                    if exit_process and not released:
                        process.stdin.write(b"x")
                        process.stdin.flush()
                        released = True
            return result

        monkeypatch.setattr(host, "read_proc", unreadable_owned_memory)
        armed = True
        start = time.monotonic()
        sample = sampler.collect()
        elapsed = time.monotonic() - start
        receipt = sample["host_collection"]
        assert receipt["protocol"] == host.TEARDOWN_METHOD and receipt["max_collection_seconds"] == 4.0
        first = receipt["attempts"][0]["processes"][str(process.pid)]
        assert 350 < first["poll_after"]["timeout_milliseconds"] <= 3500
        if exit_process:
            assert elapsed >= 0.8 and elapsed < 4.0
            assert host._released(first["poll_after"]) is True
            assert len(receipt["attempts"]) == 2
            assert receipt["attempts"][1]["begin_monotonic_seconds"] >= first["poll_after"]["end_monotonic_seconds"]
            assert host.validate_sample(sample)["host_pss_complete"]
            assert sample["process_host_memory"][str(process.pid)]["address_space_released"] is True
            # A complete post-exit observation never provides a peak for the failed interval.
            assert "VmRSS" not in host.classify_process(first, owner_is_identity=False)[0]
        else:
            assert 3.4 <= elapsed < 4.0
            assert process.poll() is None and host._released(first["poll_after"]) is False
            assert len(receipt["attempts"]) == 1
            assert not host.validate_sample(sample)["host_pss_complete"]
            assert "address_space_released" not in sample["process_host_memory"][str(process.pid)]
        request.node.user_properties.append(("v9_wait_elapsed_seconds", elapsed))
        request.node.user_properties.append(("v9_delayed_exit_receipt", json.dumps(sample, sort_keys=True)))
    finally:
        sampler.close()
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
        process.stdin.close()
        process.stdout.close()


def test_v9_budget_and_deadline_are_not_user_relaxable(tmp_path):
    process = _child()
    sampler = host.HostSampler(process.pid, tmp_path / "stage", method=host.TEARDOWN_METHOD)
    try:
        assert process.stdout.readline() == b"ready\n"
        sample = sampler.collect()
        assert host.validate_sample(sample)["host_pss_complete"]
        altered = copy.deepcopy(sample)
        altered["host_collection"]["max_collection_seconds"] = 5
        with pytest.raises(ValueError, match="bounds"):
            host.validate_sample(altered)
        late = copy.deepcopy(sample)
        publication = late["host_collection"]["stage_publish"]
        start = late["host_collection"]["attempts"][0]["begin_monotonic_seconds"]
        publication["begin_monotonic_seconds"] = start + 4.01
        publication["end_monotonic_seconds"] = start + 4.02
        derived = host.summarize_collection(late["host_collection"])
        assert not derived["host_pss_complete"] and "collection_deadline_exceeded" in derived["host_collection_errors"]
    finally:
        sampler.close()
        process.kill()
        process.wait(timeout=5)
        process.stdin.close()
        process.stdout.close()


@pytest.mark.skipif(os.environ.get("COSMOS_RUN_LARGE_EXIT_TEST") != "1", reason="explicit 8 GiB CPU integration")
@pytest.mark.parametrize("method", [host.METHOD, host.TEARDOWN_METHOD])
def test_actual_large_mmap_teardown_recollects_after_bound_pidfd_exit(tmp_path, request, method):
    """No memory-read mocks: exercise actual Linux exit_mmap before process-wide POLLIN."""
    script = """import mmap,os,sys
n=8*1024**3
m=mmap.mmap(-1,n)
chunk=b"x"*(8*1024**2)
for offset in range(0,n,len(chunk)): m[offset:offset+len(chunk)]=chunk
print('ready',flush=True)
sys.stdin.read(1)
os._exit(0)
"""
    process = subprocess.Popen([sys.executable, "-c", script], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    sampler = host.HostSampler(process.pid, tmp_path / "stage", method=method)
    try:
        assert process.stdout.readline() == b"ready\n"
        initial = sampler.collect()
        assert host.validate_sample(initial)["host_pss_complete"]
        assert initial["process_host_memory"][str(process.pid)]["VmRSS"] >= 8 * 1024**3
        process.stdin.write(b"x")
        process.stdin.flush()
        begin = time.monotonic()
        # Actual empty memory/ESRCH is the trigger, never an inference of exit.
        deadline = begin + 5
        while True:
            result = host.read_proc(process.pid, "status")
            if result.get("value") == {}:
                break
            assert time.monotonic() < deadline, "no observable large-mm teardown interval"
        sample = sampler.collect()
        elapsed = time.monotonic() - begin
        records = [a["processes"][str(process.pid)] for a in sample["host_collection"]["attempts"]]
        request.node.user_properties.append(
            (
                "actual_8g_observation",
                json.dumps({"initial": initial, "sample": sample, "elapsed": elapsed}, sort_keys=True),
            )
        )
        assert records[0]["rollup"].get("error", {}).get("errno") == 3
        assert records[0]["exit_wait"]["initial_poll"]["events"] == []
        wait = records[0]["poll_after"]
        actual_wait = wait["end_monotonic_seconds"] - wait["begin_monotonic_seconds"]
        if method == host.TEARDOWN_METHOD:
            assert len(records) == 2 and host._released(records[0]["poll_after"]) is True
            assert actual_wait > 0.35
            assert host.validate_sample(sample)["host_pss_complete"] and elapsed < 4.0
            assert sample["process_host_memory"][str(process.pid)]["address_space_released"] is True
        else:
            assert len(records) == 1 and host._released(records[0]["poll_after"]) is False
            assert not host.validate_sample(sample)["host_pss_complete"]
            assert "address_space_released" not in sample["process_host_memory"][str(process.pid)]
        request.node.user_properties.append(("actual_8g_initial_receipt", json.dumps(initial, sort_keys=True)))
        process.wait(timeout=5)
        assert process.returncode == 0
        request.node.user_properties.append(("actual_8g_exit_wait_seconds", actual_wait))
        request.node.user_properties.append(("actual_8g_collection_seconds", elapsed))
        request.node.user_properties.append(("actual_8g_receipt", json.dumps(sample, sort_keys=True)))
    finally:
        sampler.close()
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
        process.stdin.close()
        process.stdout.close()
