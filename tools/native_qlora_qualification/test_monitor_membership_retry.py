# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""Prospective late-exit recollection, never retrospective repair of v9 evidence."""

import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tools.native_qlora_qualification import monitor_host as host
from tools.native_qlora_qualification import monitor_stage


def test_actual_v9_deadline_failure_stays_incomplete_and_cannot_be_relabelled():
    fixture = json.loads((Path(__file__).parent / "fixtures/host_monitor_v9_deadline.json").read_text())
    original = copy.deepcopy(fixture)
    assert fixture["source_job"] == 38536188 and fixture["sample_indices"] == [819, 820]
    failed, later = fixture["samples"]
    assert host.validate_sample(failed)["host_collection_errors"] == [
        "collection_deadline_exceeded",
        "membership_transition_retry_exhausted",
    ]
    assert not host.validate_sample(failed)["host_pss_complete"]
    assert host.validate_sample(later)["host_pss_complete"]
    assert fixture == original
    for sample in fixture["samples"]:
        sample["host_collection"]["protocol"] = sample["host_process_memory_method"] = host.MEMBERSHIP_RETRY_METHOD
        with pytest.raises(ValueError, match="bounds"):
            host.validate_sample(sample)


def test_budgets_and_stage_lock_keep_all_historical_policies():
    for method in (
        host.LEGACY_METHOD,
        host.EXIT_OWNER_METHOD,
        host.PROCESS_IDENTITY_METHOD,
        host.LIVE_THREAD_METHOD,
        host.THREAD_EXIT_METHOD,
        host.THREAD_WAIT_METHOD,
        host.REAPED_THREAD_METHOD,
        host.METHOD,
    ):
        assert host._collection_seconds(method) == 0.75
        assert host._exit_wait_milliseconds(method) == 350
    assert host._collection_seconds(host.TEARDOWN_METHOD) == 4.0
    assert host._collection_seconds(host.MEMBERSHIP_RETRY_METHOD) == 4.5
    for method in (host.TEARDOWN_METHOD, host.MEMBERSHIP_RETRY_METHOD):
        assert host._exit_wait_milliseconds(method) == 3500
    assert monitor_stage.LEGACY_LOCK_TIMEOUT_SECONDS == 5.0
    assert monitor_stage.LOCK_TIMEOUT_SECONDS == 15.0


def child(script=None):
    return subprocess.Popen(
        [sys.executable, "-c", script or "import sys;print('ready',flush=True);sys.stdin.read(1)"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )


def cleanup(process, sampler):
    sampler.close()
    if process.poll() is None:
        process.kill()
    process.wait(timeout=5)
    process.stdin.close()
    process.stdout.close()


@pytest.mark.parametrize(
    "method,delay,complete",
    [
        (host.TEARDOWN_METHOD, 4.02, False),
        (host.MEMBERSHIP_RETRY_METHOD, 4.02, True),
        (host.MEMBERSHIP_RETRY_METHOD, 4.52, False),
    ],
)
def test_real_exit_after_delayed_read_needs_new_in_budget_membership(
    tmp_path, monkeypatch, request, method, delay, complete
):
    """Real child/PIDFD and memory reads; only the first read's elapsed latency is injected."""
    process = child()
    sampler = host.HostSampler(process.pid, tmp_path / "stage", method=method)
    read = host.read_proc
    armed = delayed = False
    try:
        assert process.stdout.readline() == b"ready\n"
        assert host.validate_sample(sampler.collect())["host_pss_complete"]

        def slow_read(pid, name):
            nonlocal delayed
            result = read(pid, name)
            if armed and not delayed and pid == process.pid and name == "smaps_rollup":
                delayed = True
                process.stdin.write(b"x")
                process.stdin.flush()
                # Match the measured race: a read begins with a live process,
                # returns its actual memory values late, and then PIDFD proves exit.
                time.sleep(max(0, delay - (time.monotonic() - result["begin_monotonic_seconds"])))
                result.update(end_monotonic_seconds=time.monotonic(), end_unix_seconds=time.time())
            return result

        monkeypatch.setattr(host, "read_proc", slow_read)
        armed = True
        begin = time.monotonic()
        sample = sampler.collect()
        elapsed = time.monotonic() - begin
        derived = host.validate_sample(sample)
        assert derived["host_pss_complete"] is complete
        attempts = sample["host_collection"]["attempts"]
        first = attempts[0]["processes"][str(process.pid)]
        assert host._released(first["poll_before"]) is False and host._released(first["poll_after"]) is True
        if complete:
            assert len(attempts) == 2
            assert attempts[1]["begin_monotonic_seconds"] >= first["poll_after"]["end_monotonic_seconds"]
            assert sample["process_host_memory"][str(process.pid)]["address_space_released"] is True
            assert elapsed < 4.5
            # Removing the new actual read must never turn the old interval complete.
            partial = copy.deepcopy(sample["host_collection"])
            partial["attempts"] = partial["attempts"][:1]
            partial["final_attempt"] = 0
            assert not host.summarize_collection(partial)["host_pss_complete"]
        else:
            assert len(attempts) == 1
            assert "collection_deadline_exceeded" in derived["host_collection_errors"]
        request.node.user_properties.append(
            (
                "late_exit_acquisition",
                json.dumps(
                    {
                        "elapsed_seconds": elapsed,
                        "read_delay_injected_seconds": delay,
                        "actual_pidfd_and_memory_reads": True,
                        "sample": sample,
                    },
                    sort_keys=True,
                ),
            )
        )
    finally:
        cleanup(process, sampler)


def test_v10_live_unreadable_process_is_not_zero_or_complete(tmp_path, monkeypatch):
    process = child()
    sampler = host.HostSampler(process.pid, tmp_path / "stage", method=host.MEMBERSHIP_RETRY_METHOD)
    read = host.read_proc
    try:
        assert process.stdout.readline() == b"ready\n"
        assert sampler.collect()["host_pss_complete"]

        def unreadable(pid, name):
            value = read(pid, name)
            if pid == process.pid and name in ("status", "smaps_rollup"):
                value.pop("value", None)
                if name == "status":
                    value["value"] = {}
                else:
                    value["error"] = {"errno": 3, "type": "ProcessLookupError"}
            return value

        monkeypatch.setattr(host, "read_proc", unreadable)
        sample = sampler.collect()
        assert process.poll() is None and not host.validate_sample(sample)["host_pss_complete"]
        first = sample["host_collection"]["attempts"][0]["processes"][str(process.pid)]
        assert first["poll_after"]["timeout_milliseconds"] <= 3500
        assert host._released(first["poll_after"]) is False
        assert "address_space_released" not in sample["process_host_memory"][str(process.pid)]
    finally:
        cleanup(process, sampler)


@pytest.mark.skipif(os.environ.get("COSMOS_RUN_LARGE_EXIT_TEST") != "1", reason="explicit8GiB real CPU teardown")
def test_v10_real_large_mmap_initial_peak_and_fresh_exit_observation(tmp_path, request):
    """No read mocks or latency injection: actual Linux large-mm teardown."""
    script = """import mmap,os,sys
n=8*1024**3
m=mmap.mmap(-1,n)
chunk=b'x'*(8*1024**2)
for offset in range(0,n,len(chunk)):m[offset:offset+len(chunk)]=chunk
print('ready',flush=True)
sys.stdin.read(1)
os._exit(0)
"""
    process = child(script)
    sampler = host.HostSampler(process.pid, tmp_path / "stage", method=host.MEMBERSHIP_RETRY_METHOD)
    try:
        assert process.stdout.readline() == b"ready\n"
        initial = sampler.collect()
        assert host.validate_sample(initial)["host_pss_complete"]
        assert initial["process_host_memory"][str(process.pid)]["VmRSS"] >= 8 * 1024**3
        process.stdin.write(b"x")
        process.stdin.flush()
        begin = time.monotonic()
        while host.read_proc(process.pid, "status").get("value") != {}:
            assert time.monotonic() - begin < 5, "no observable teardown interval"
        sample = sampler.collect()
        elapsed = time.monotonic() - begin
        assert host.validate_sample(sample)["host_pss_complete"] and elapsed < 4.5
        records = [a["processes"][str(process.pid)] for a in sample["host_collection"]["attempts"]]
        assert len(records) == 2 and host._released(records[0]["poll_after"]) is True
        assert records[0]["rollup"].get("error", {}).get("errno") == 3
        assert records[0]["exit_wait"]["initial_poll"]["events"] == []
        assert records[0]["poll_after"]["timeout_milliseconds"] <= 3500
        assert sample["process_host_memory"][str(process.pid)]["address_space_released"] is True
        request.node.user_properties.append(
            (
                "actual8GiB_v10",
                json.dumps(
                    {
                        "initial": initial,
                        "sample": sample,
                        "elapsed_seconds": elapsed,
                        "read_mocks": False,
                        "injected_delay": False,
                    },
                    sort_keys=True,
                ),
            )
        )
        process.wait(timeout=5)
        assert process.returncode == 0
    finally:
        cleanup(process, sampler)
