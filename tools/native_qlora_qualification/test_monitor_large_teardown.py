# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""Prospective long teardown acquisition; old gaps remain incomplete."""

import copy
import json
import threading
import time
from pathlib import Path

import pytest

from tools.native_qlora_qualification import monitor_host as host
from tools.native_qlora_qualification import monitor_stage as stage
from tools.native_qlora_qualification.test_monitor_membership_retry import child, cleanup
from tools.native_qlora_qualification.test_monitor_stage import cleanup as cleanup_writer
from tools.native_qlora_qualification.test_monitor_stage import finish, ready, start


def test_retained_v10_failures_and_fresh_release_keep_exact_decisions():
    fixture = json.loads((Path(__file__).parent / "fixtures/host_monitor_v10_large_exit.json").read_text())
    original = copy.deepcopy(fixture)
    assert [case["source_job"] for case in fixture["cases"]] == [38540661, 38543002]
    for case in fixture["cases"]:
        failed, later = case["samples"]
        assert host.validate_sample(failed)["host_collection_errors"] == ["live_or_unknown_memory_unreadable"]
        assert not host.validate_sample(failed)["host_pss_complete"]
        assert host.validate_sample(later)["host_pss_complete"]
        assert failed["host_collection"]["max_collection_seconds"] == 4.5
        for old in (failed, later):
            altered = copy.deepcopy(old)
            altered["host_process_memory_method"] = host.LARGE_TEARDOWN_METHOD
            altered["host_collection"]["protocol"] = host.LARGE_TEARDOWN_METHOD
            with pytest.raises(ValueError, match="bounds"):
                host.validate_sample(altered)
    assert fixture == original


def test_method_budgets_and_wait_reserve_are_fixed(monkeypatch):
    assert host._collection_seconds(host.LARGE_TEARDOWN_METHOD) == 13.5
    assert host._exit_wait_milliseconds(host.LARGE_TEARDOWN_METHOD) == 12000
    assert host._collection_seconds(host.MEMBERSHIP_RETRY_METHOD) == 4.5
    assert host._exit_wait_milliseconds(host.MEMBERSHIP_RETRY_METHOD) == 3500
    assert stage.LOCK_TIMEOUT_SECONDS == 15.0
    monkeypatch.setattr(host.time, "monotonic", lambda: 100)
    assert host._exit_wait_timeout(113.5, method=host.LARGE_TEARDOWN_METHOD) == 12000
    assert host._exit_wait_timeout(104.5, method=host.MEMBERSHIP_RETRY_METHOD) == 3500
    assert host._exit_wait_timeout(100.5, method=host.LARGE_TEARDOWN_METHOD) == 249
    assert host._exit_wait_timeout(100.25, method=host.LARGE_TEARDOWN_METHOD) <= 0


@pytest.mark.parametrize("exit_process", [True, False])
def test_real_pidfd_exit_or_live_unreadable_timeout(tmp_path, monkeypatch, exit_process):
    """Only proc unreadability is injected; child/PIDFD/retry reads are real."""
    process = child()
    sampler = host.HostSampler(process.pid, tmp_path / "stage", method=host.LARGE_TEARDOWN_METHOD)
    read = host.read_proc
    timer = None
    armed = False
    try:
        assert process.stdout.readline() == b"ready\n"
        initial = sampler.collect()
        assert host.validate_sample(initial)["host_pss_complete"]

        def unreadable(pid, name):
            nonlocal timer
            result = read(pid, name)
            if armed and pid == process.pid and name in ("status", "smaps_rollup"):
                result.pop("value", None)
                result.pop("error", None)
                if name == "status":
                    result["value"] = {}
                else:
                    result["error"] = {"errno": 3, "type": "ProcessLookupError"}
                    if exit_process and timer is None:
                        timer = threading.Timer(4.0, lambda: (process.stdin.write(b"x"), process.stdin.flush()))
                        timer.start()
            return result

        monkeypatch.setattr(host, "read_proc", unreadable)
        armed = True
        begun = time.monotonic()
        sample = sampler.collect()
        elapsed = time.monotonic() - begun
        assert host.validate_sample(sample)["host_pss_complete"] is exit_process
        attempts = sample["host_collection"]["attempts"]
        first = attempts[0]["processes"][str(process.pid)]
        assert first["poll_after"]["timeout_milliseconds"] == 12000
        if exit_process:
            assert 3.8 < elapsed < 13.5
            assert len(attempts) == 2 and host._released(first["poll_after"]) is True
            assert attempts[1]["begin_monotonic_seconds"] >= first["poll_after"]["end_monotonic_seconds"]
            assert sample["process_host_memory"][str(process.pid)]["address_space_released"]
            partial = copy.deepcopy(sample["host_collection"])
            partial["attempts"] = partial["attempts"][:1]
            partial["final_attempt"] = 0
            assert not host.summarize_collection(partial)["host_pss_complete"]
            too_long = copy.deepcopy(sample)
            too_long["host_collection"]["attempts"][0]["processes"][str(process.pid)]["poll_after"][
                "timeout_milliseconds"
            ] = 12001
            with pytest.raises(ValueError, match="exit wait"):
                host.validate_sample(too_long)
        else:
            assert elapsed >= 11.9 and process.poll() is None
            assert len(attempts) == 1 and host._released(first["poll_after"]) is False
            assert "address_space_released" not in sample["process_host_memory"][str(process.pid)]
    finally:
        if timer is not None:
            timer.cancel()
            timer.join(timeout=5)
        cleanup(process, sampler)


def test_v11_overrun_and_permission_error_remain_incomplete(tmp_path, monkeypatch):
    process = child()
    sampler = host.HostSampler(process.pid, tmp_path / "stage", method=host.LARGE_TEARDOWN_METHOD)
    read = host.read_proc
    try:
        assert process.stdout.readline() == b"ready\n"
        sample = sampler.collect()
        late = copy.deepcopy(sample["host_collection"])
        begin = late["attempts"][0]["begin_monotonic_seconds"]
        late["stage_publish"].update(begin_monotonic_seconds=begin + 13.51, end_monotonic_seconds=begin + 13.52)
        result = host.summarize_collection(late)
        assert not result["host_pss_complete"] and "collection_deadline_exceeded" in result["host_collection_errors"]

        def denied(pid, name):
            result = read(pid, name)
            if pid == process.pid and name == "smaps_rollup":
                result.pop("value", None)
                result["error"] = {"errno": 13, "type": "PermissionError"}
            return result

        monkeypatch.setattr(host, "read_proc", denied)
        denied_sample = sampler.collect()
        assert not host.validate_sample(denied_sample)["host_pss_complete"]
        item = denied_sample["host_collection"]["attempts"][0]["processes"][str(process.pid)]
        assert (
            "exit_wait" not in item
            and "address_space_released" not in denied_sample["process_host_memory"][str(process.pid)]
        )
    finally:
        cleanup(process, sampler)


def test_publication_waits_past_legacy_five_seconds_without_changing_stage(tmp_path):
    marker = tmp_path / "stage"
    stage.publish_stage(marker, "old", initialize=True)
    writer = None
    try:
        with stage.sample_stage(marker) as proof:
            writer = start(
                "from pathlib import Path;from tools.native_qlora_qualification.monitor_stage import publish_stage;"
                "print('ready',flush=True);r=publish_stage(Path(sys.argv[1]),'next');"
                "import json;print(json.dumps(r),flush=True)",
                marker,
            )
            ready(writer)
            time.sleep(5.1)
            assert writer.poll() is None and marker.read_text() == "old\n"
            assert proof["timeout_seconds"] == 15
        published = json.loads(finish(writer))
        assert published["timeout_seconds"] == 15 and published["wait_seconds"] >= 5
        assert marker.read_text() == "next\n"
    finally:
        if writer is not None:
            cleanup_writer(writer)


def test_actual_hold_overrun_refuses_publication_and_releases_lock(tmp_path, monkeypatch):
    marker = tmp_path / "stage"
    stage.publish_stage(marker, "old", initialize=True)
    published = []
    with monkeypatch.context() as patch:
        with pytest.raises(TimeoutError, match="sample hold"):
            with stage.sample_stage(marker) as receipt:
                # Exercise the actual hold check after the final same-inode
                # check; no fifteen-second sleep is necessary for this boundary.
                patch.setattr(stage.time, "monotonic", lambda: receipt["acquired_monotonic_seconds"] + 15.01)
            published.append("must not publish a skipped overrun as complete")
    assert published == [] and marker.read_text() == "old\n"
    assert stage.publish_stage(marker, "after_failure", timeout=0.5)["stage"] == "after_failure"
