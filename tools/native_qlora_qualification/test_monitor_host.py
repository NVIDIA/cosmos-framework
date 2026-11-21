# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""Real Linux child lifetimes plus hostile receipt tests; no models or CUDA."""

import copy
import ctypes
import errno
import json
import os
import select
import shutil
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import pytest

from tools.native_qlora_qualification import monitor_host as host


class _CapabilityHeader(ctypes.Structure):
    _fields_ = [("version", ctypes.c_uint32), ("pid", ctypes.c_int)]


class _CapabilityWord(ctypes.Structure):
    _fields_ = [("effective", ctypes.c_uint32), ("permitted", ctypes.c_uint32), ("inheritable", ctypes.c_uint32)]


def _capabilities(values=None):
    """Linux v3 capget/capset ABI, pid=0 means this sampling thread only."""
    libc = ctypes.CDLL(None, use_errno=True)
    header = _CapabilityHeader(0x20080522, 0)
    words = (_CapabilityWord * 2)()
    call = libc.capget if values is None else libc.capset
    call.argtypes = [ctypes.POINTER(_CapabilityHeader), ctypes.POINTER(_CapabilityWord)]
    call.restype = ctypes.c_int
    if values is not None:
        for index, value in enumerate(values):
            for key in ("effective", "permitted", "inheritable"):
                setattr(words[index], key, value[key])
    if call(ctypes.byref(header), words) != 0:
        code = ctypes.get_errno()
        raise OSError(code, "current-thread capability operation failed")
    return [{key: getattr(word, key) for key in ("effective", "permitted", "inheritable")} for word in words]


def _capability_status():
    return {
        key: value.strip()
        for line in Path("/proc/thread-self/status").read_text().splitlines()
        for key, _, value in (line.partition(":"),)
        if key.startswith("Cap")
    }


@contextmanager
def without_ptrace_capability(request):
    """Permission fixture only; restore the complete original structure even on failure.

    Containers may grant CAP_SYS_PTRACE, which overrides nondumpable protection.
    Clear only that effective bit, retaining permitted/inheritable/bounding/ambient
    capabilities. No training or monitor production policy uses this context.
    """
    initial = _capabilities()
    initial_status = _capability_status()
    expected = copy.deepcopy(initial)
    expected[0]["effective"] &= ~(1 << 19)  # Linux CAP_SYS_PTRACE.
    evidence = {"initial": initial, "initial_proc_status": initial_status, "drop_expected": expected}
    try:
        _capabilities(expected)
        evidence["dropped"] = _capabilities()
        evidence["dropped_proc_status"] = _capability_status()
        assert evidence["dropped"] == expected
        assert all(
            evidence["dropped_proc_status"][key] == initial_status[key] for key in initial_status if key != "CapEff"
        )
        yield evidence
    finally:
        try:
            _capabilities(initial)
            evidence["restored"] = _capabilities()
            evidence["restored_proc_status"] = _capability_status()
            assert evidence["restored"] == initial
            assert evidence["restored_proc_status"] == initial_status
        finally:
            request.node.user_properties.append(("ptrace_capability_fixture", json.dumps(evidence, sort_keys=True)))


@pytest.fixture
def child():
    children = []

    def start(*, nondumpable=False):
        code = "import ctypes,sys; data=bytearray(4*1024*1024); "
        if nondumpable:
            code += "libc=ctypes.CDLL(None); assert libc.prctl(4,0,0,0,0)==0; assert libc.prctl(3,0,0,0,0)==0; "
        code += "print('ready',flush=True); sys.stdin.read(1)"
        process = subprocess.Popen(
            [sys.executable, "-I", "-S", "-c", code], stdin=subprocess.PIPE, stdout=subprocess.PIPE
        )
        children.append(process)
        assert process.stdout.readline() == b"ready\n"
        return process

    yield start
    for process in children:
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=5)
        process.stdin.close()
        process.stdout.close()


@pytest.fixture
def sampler(tmp_path, child):
    process = child()
    marker = tmp_path / "stage"
    marker.write_text("training")
    value = host.HostSampler(process.pid, marker)
    yield value, process
    value.close()


def test_real_live_child_receipt_and_descriptor_cleanup(sampler):
    value, process = sampler
    before = len(list(Path("/proc/self/fd").iterdir()))
    sample = value.collect()
    assert host.validate_sample(sample)["host_pss_complete"]
    assert sample["selected_host_pss_sum_bytes"] > 4 * 1024 * 1024
    assert set(sample["process_host_memory"]) == {str(process.pid)}
    item = sample["host_collection"]["attempts"][0]["processes"][str(process.pid)]
    assert item["pin"]["identity"]["start_ticks"] > 0
    assert item["stat_before"]["value"]["state"] in "RS"
    assert item["poll_before"]["events"] == []
    assert len(list(Path("/proc/self/fd").iterdir())) == before + 1
    value.close()
    assert len(list(Path("/proc/self/fd").iterdir())) == before


def test_real_exit_between_status_and_rollup_retries_and_preserves_failure(sampler, monkeypatch):
    value, process = sampler
    read = host.read_proc
    fired = False

    def exit_between(pid, name):
        nonlocal fired
        result = read(pid, name)
        if pid == process.pid and name == "status" and not fired:
            fired = True
            process.terminate()
            process.wait(timeout=5)
        return result

    monkeypatch.setattr(host, "read_proc", exit_between)
    sample = value.collect()
    attempts = sample["host_collection"]["attempts"]
    assert len(attempts) == 2
    first = attempts[0]["processes"][str(process.pid)]
    assert first["rollup"]["error"]["errno"] in (2, 3)
    assert first["poll_before"]["events"] == []
    assert first["poll_after"]["events"][0] & select.POLLIN
    assert host.validate_sample(sample)["host_pss_complete"]
    assert sample["selected_host_pss_sum_bytes"] == 0
    assert sample["process_host_memory"][str(process.pid)]["address_space_released"] is True


def test_actual_live_nondumpable_child_is_incomplete(child, tmp_path, request):
    process = child(nondumpable=True)
    value = host.HostSampler(process.pid, tmp_path / "stage")
    before = host.read_proc(process.pid, "smaps_rollup")
    try:
        with without_ptrace_capability(request) as evidence:
            evidence.update(child_pr_get_dumpable=0, before_read=before)
            sample = value.collect()
            item = sample["host_collection"]["attempts"][0]["processes"][str(process.pid)]
            evidence["dropped_read"] = item["rollup"]
            assert item["rollup"]["error"]["errno"] == 13
            assert item["poll_after"]["events"] == []
            assert not host.validate_sample(sample)["host_pss_complete"]
            assert "live_or_unknown_memory_unreadable" in sample["host_collection_errors"]
            assert len(sample["host_collection"]["attempts"]) == 1
        restored = host.read_proc(process.pid, "smaps_rollup")
        request.node.user_properties.append(("restored_read", json.dumps(restored, sort_keys=True)))
        assert ("error" in restored) == ("error" in before)
        if "error" in before:
            assert restored["error"]["errno"] == before["error"]["errno"]
    finally:
        value.close()


def test_pid_identity_mismatch_not_released_even_with_readable_pidfd(sampler):
    value, process = sampler
    sample = value.collect()
    receipt = sample["host_collection"]
    item = receipt["attempts"][0]["processes"][str(process.pid)]
    item["stat_after"]["value"]["start_ticks"] += 1
    item["poll_before"]["events"] = [select.POLLIN]
    item["poll_after"]["events"] = [select.POLLIN]
    result = host.summarize_collection(receipt)
    assert not result["host_pss_complete"]
    assert "process_identity_changed" in result["host_collection_errors"]
    assert "address_space_released" not in result["process_host_memory"][str(process.pid)]


def test_stage_race_during_collection_remains_incomplete(sampler, monkeypatch):
    value, process = sampler
    read = host.read_proc

    def change_stage(pid, name):
        result = read(pid, name)
        if pid == process.pid and name == "status":
            value.stage_file.write_text("portable_publication")
        return result

    monkeypatch.setattr(host, "read_proc", change_stage)
    sample = value.collect()
    assert sample["stage"] == "training"
    assert not host.validate_sample(sample)["host_pss_complete"]
    assert "stage_changed_or_unreadable" in sample["host_collection_errors"]
    assert len(sample["host_collection"]["attempts"]) == 1


def test_stage_race_during_device_sampling_remains_incomplete(sampler):
    value, _ = sampler
    sample = value.collect()
    assert sample["host_pss_complete"]
    value.stage_file.write_text("next_stage")
    host.finish_sample(sample, value.stage_file)
    assert sample["stage"] == "training"
    assert not host.validate_sample(sample)["host_pss_complete"]


def test_repeated_proven_transitions_exhaust_retry_without_hiding_attempts(sampler, monkeypatch):
    value, _ = sampler
    original = value._process

    def transitioning(pid):
        record = original(pid)
        record["poll_before"]["events"] = []
        record["poll_after"]["events"] = [select.POLLIN]
        return record

    monkeypatch.setattr(value, "_process", transitioning)
    poll = host._poll

    def final_released(fd):
        result = poll(fd)
        result["events"] = [select.POLLIN]
        return result

    monkeypatch.setattr(host, "_poll", final_released)
    sample = value.collect()
    assert len(sample["host_collection"]["attempts"]) == host.MAX_ATTEMPTS
    assert not host.validate_sample(sample)["host_pss_complete"]
    assert sample["host_collection_errors"] == ["membership_transition_retry_exhausted"]


def test_actual_exit_after_memory_before_final_discovery_retains_owner_change_and_peak(sampler, monkeypatch, request):
    value, process = sampler
    discover = value._discover
    calls = 0

    def exit_before_final_discovery():
        nonlocal calls
        calls += 1
        if calls == 2:
            process.terminate()
            poller = select.poll()
            poller.register(value.tracked[process.pid]["fd"], select.POLLIN)
            assert any(mask & select.POLLIN for _, mask in poller.poll(5000)), "child did not fully exit"
            # Do not reap: the real zombie's proc inode owner must be observed.
        return discover()

    monkeypatch.setattr(value, "_discover", exit_before_final_discovery)
    sample = value.collect()
    attempts = sample["host_collection"]["attempts"]
    assert len(attempts) == 2
    first, last = attempts
    key = str(process.pid)
    item = first["processes"][key]
    before = first["discovery_before"][key]["value"]
    after = first["discovery_after"][key]["value"]
    assert all(before[name] == after[name] for name in ("pid", "start_ticks", "ppid"))
    assert before["uid"] != after["uid"], "qualification requires an actual proc inode owner transition"
    assert after["state"] == "Z"
    assert item["poll_after"]["events"] == []
    assert first["post_discovery_polls"][key]["events"][0] & select.POLLIN
    assert item["rollup"]["value"]["Pss"] > 4 * 1024 * 1024
    assert host.validate_sample(sample)["host_pss_complete"]
    assert sample["selected_host_pss_sum_bytes"] == 0
    assert sample["host_observed_complete_attempt_pss_peak_bytes"] >= item["rollup"]["value"]["Pss"]
    assert last["processes"][key]["stat_before"]["value"]["uid"] == after["uid"]
    request.node.user_properties.append(
        (
            "actual_exit_inode_owner_transition",
            json.dumps(
                {
                    "before": before,
                    "after": after,
                    "pre_exit_pss_bytes": item["rollup"]["value"]["Pss"],
                    "post_discovery_events": first["post_discovery_polls"][key]["events"],
                    "retained_attempts": len(attempts),
                    "final_pss_bytes": sample["selected_host_pss_sum_bytes"],
                    "observed_attempt_peak_bytes": sample["host_observed_complete_attempt_pss_peak_bytes"],
                },
                sort_keys=True,
            ),
        )
    )


def test_actual_live_group_owner_change_preserves_valid_memory_without_exit_proof(tmp_path, monkeypatch, request):
    """The leader exits after a real read; a live thread still owns the shared mm.

    This makes proc ownership change on the qualified user-namespace target,
    without requiring an unmapped UID or granting memory-read capabilities.
    The existing unreadable-leader test separately requires incompleteness when
    the actual memory read occurs after the leader's exit instead.
    """
    compiler = shutil.which("cc")
    assert compiler, "real owner-transition qualification requires a C compiler"
    source = tmp_path / "owner_transition.c"
    source.write_text("""
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdatomic.h>
static atomic_int leader_leaving;
static void *worker(void *unused) {
  char *p = malloc(8 * 1024 * 1024);
  if (!p) _exit(11);
  for (int i = 0; i < 8 * 1024 * 1024; i += 4096) p[i] = 1;
  puts("ready"); fflush(stdout);
  while (!atomic_load(&leader_leaving)) usleep(1000);
  char c; if (read(0, &c, 1) != 1) _exit(12);
  free(p); return NULL;
}
int main(void) {
  pthread_t thread;
  if (pthread_create(&thread, NULL, worker, NULL)) return 13;
  char c; if (read(0, &c, 1) != 1) return 14;
  atomic_store(&leader_leaving, 1);
  pthread_exit(NULL);
}
""")
    binary = tmp_path / "owner_transition"
    subprocess.run([compiler, "-pthread", "-O0", str(source), "-o", str(binary)], check=True, timeout=15)
    process = subprocess.Popen([str(binary)], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    stage = tmp_path / "stage"
    stage.write_text("training")
    sampler = host.HostSampler(process.pid, stage)
    read = host.read_proc
    fired = False
    try:
        assert process.stdout.readline() == b"ready\n"

        def leave_leader_after_actual_memory(pid, name):
            nonlocal fired
            result = read(pid, name)
            if pid == process.pid and name == "smaps_rollup" and not fired:
                assert result["value"]["Pss"] > 4 * 1024 * 1024
                fired = True
                process.stdin.write(b"x")
                process.stdin.flush()
                deadline = time.monotonic() + 5
                while read(pid, "stat").get("value", {}).get("state") != "Z":
                    assert time.monotonic() < deadline, "leader did not exit while worker remained alive"
                    time.sleep(0.001)
            return result

        monkeypatch.setattr(host, "read_proc", leave_leader_after_actual_memory)
        sample = sampler.collect()
        key = str(process.pid)
        attempts = sample["host_collection"]["attempts"]
        assert len(attempts) == 1
        item = attempts[0]["processes"][key]
        before, after = item["stat_before"]["value"], item["stat_after"]["value"]
        assert before["uid"] != after["uid"], "qualification requires actual mutable proc inode ownership"
        assert all(before[name] == after[name] for name in ("pid", "start_ticks", "ppid"))
        assert after["state"] == "Z"
        assert item["poll_before"]["events"] == item["poll_after"]["events"] == []
        assert attempts[0]["post_discovery_polls"][key]["events"] == []
        assert host.validate_sample(sample)["host_pss_complete"]
        assert sample["selected_host_pss_sum_bytes"] == item["rollup"]["value"]["Pss"] > 4 * 1024 * 1024
        assert "address_space_released" not in sample["process_host_memory"][key]
        request.node.user_properties.append(
            (
                "actual_live_group_inode_owner_transition",
                json.dumps(
                    {
                        "before": before,
                        "after": after,
                        "memory_read": item["rollup"],
                        "poll_before": item["poll_before"],
                        "poll_after": item["poll_after"],
                        "post_discovery_poll": attempts[0]["post_discovery_polls"][key],
                        "final_pss_bytes": sample["selected_host_pss_sum_bytes"],
                        "address_space_released": False,
                        "complete": sample["host_pss_complete"],
                    },
                    sort_keys=True,
                ),
            )
        )
        process.stdin.write(b"y")
        process.stdin.flush()
        process.wait(timeout=5)
        assert process.returncode == 0
    finally:
        sampler.close()
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
        process.stdin.close()
        process.stdout.close()


@pytest.mark.parametrize("state", ["R", "S", "Z"])
def test_v3_owner_only_is_metadata_and_never_zero_memory(sampler, state):
    value, process = sampler
    sample = value.collect()
    receipt = sample["host_collection"]
    attempt = receipt["attempts"][0]
    key = str(process.pid)
    for read in (attempt["processes"][key]["stat_after"], attempt["discovery_after"][key]):
        read["value"].update(uid=read["value"]["uid"] + 1, state=state)
    sample.update(host.summarize_collection(receipt))
    assert host.validate_sample(sample)["host_pss_complete"]
    assert sample["selected_host_pss_sum_bytes"] > 4 * 1024 * 1024
    assert "address_space_released" not in sample["process_host_memory"][key]


@pytest.mark.parametrize("field", ["pid", "start_ticks"])
def test_v3_owner_change_cannot_conceal_bound_identity_change(sampler, field):
    value, process = sampler
    sample = value.collect()
    attempt = sample["host_collection"]["attempts"][0]
    key = str(process.pid)
    for read in (attempt["processes"][key]["stat_after"], attempt["discovery_after"][key]):
        read["value"]["uid"] += 1
        read["value"][field] += 1
    result = host.summarize_collection(sample["host_collection"])
    assert not result["host_pss_complete"]
    assert "process_identity_changed" in result["host_collection_errors"]
    assert "discovery_identity_changed" in result["host_collection_errors"]


@pytest.mark.parametrize("failed_initial_pin", [False, True])
def test_actual_pinned_descendant_remains_owned_after_reparenting(tmp_path, request, monkeypatch, failed_initial_pin):
    compiler = shutil.which("cc")
    assert compiler, "real owned-reparent qualification requires a C compiler"
    source = tmp_path / "reparent.c"
    source.write_text("""
#include <sys/prctl.h>
#include <sys/wait.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
int main(void) {
  int parent_go[2], child_go[2];
  if (prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) || pipe(parent_go) || pipe(child_go)) return 11;
  pid_t parent = fork(); if (parent < 0) return 12;
  if (!parent) {
    pid_t child = fork(); if (child < 0) _exit(13);
    if (!child) {
      char *p = malloc(4 * 1024 * 1024); if (!p) _exit(14);
      for (int i = 0; i < 4 * 1024 * 1024; i += 4096) p[i] = 1;
      printf("child %d\\n", getpid()); fflush(stdout);
      char c; if (read(child_go[0], &c, 1) != 1) _exit(15);
      free(p); _exit(0);
    }
    char c; if (read(parent_go[0], &c, 1) != 1) _exit(16);
    _exit(0);
  }
  printf("parent %d\\n", parent); fflush(stdout);
  char c; int status;
  if (read(0, &c, 1) != 1 || write(parent_go[1], "x", 1) != 1) return 17;
  if (waitpid(parent, &status, 0) != parent || status != 0) return 18;
  puts("reparented"); fflush(stdout);
  if (read(0, &c, 1) != 1 || write(child_go[1], "x", 1) != 1) return 19;
  if (waitpid(-1, &status, 0) <= 0 || status != 0) return 20;
  return 0;
}
""")
    binary = tmp_path / "reparent"
    subprocess.run([compiler, "-O0", str(source), "-o", str(binary)], check=True, timeout=15)
    process = subprocess.Popen([str(binary)], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    stage = tmp_path / "stage"
    stage.write_text("training")
    sampler = host.HostSampler(process.pid, stage)
    try:
        members = dict(process.stdout.readline().decode().split() for _ in range(2))
        parent, child = int(members["parent"]), int(members["child"])
        pin, read = sampler._pin, host.read_proc

        def unavailable_before_pin(pid, discovery):
            if pid != child:
                return pin(pid, discovery)

            def missing_once(target, name):
                result = read(target, name)
                if target == child and name == "stat":
                    result.pop("value", None)
                    result["error"] = {"type": "FileNotFoundError", "errno": errno.ENOENT}
                return result

            with monkeypatch.context() as patch:
                patch.setattr(host, "read_proc", missing_once)
                return pin(pid, discovery)

        with monkeypatch.context() as patch:
            if failed_initial_pin:
                patch.setattr(sampler, "_pin", unavailable_before_pin)
            before = sampler.collect()
        assert host.validate_sample(before)["host_pss_complete"] is not failed_initial_pin
        initial_pin = before["host_collection"]["attempts"][0]["processes"][str(child)]["pin"]
        original = initial_pin["discovery"]["value"] if failed_initial_pin else initial_pin["identity"]
        assert original["ppid"] == parent
        process.stdin.write(b"p")
        process.stdin.flush()
        assert process.stdout.readline() == b"reparented\n"
        after = sampler.collect()
        item = after["host_collection"]["attempts"][0]["processes"][str(child)]
        if failed_initial_pin:
            proof = after["host_collection"]["unbound_recoveries"][str(child)]
            assert proof["result"] == "rebound" and proof["pidfd_open"]["value"] == {"opened": True}
            assert item["pin"]["discovery"] == initial_pin["discovery"]
            assert host._identity(item["pin"]["identity"], owner_is_identity=False) == host._identity(
                original, owner_is_identity=False
            )
            request.node.user_properties.append(
                (
                    "actual_unbound_live_orphan_rebinding",
                    json.dumps(
                        {
                            "failed": before,
                            "recovered": after,
                            "initial_stat_enoent_injected": True,
                            "actual_reparent_and_pidfd_open": True,
                            "previous_interval_repaired": False,
                        },
                        sort_keys=True,
                    ),
                )
            )
        else:
            assert item["pin"]["identity"] == original
            assert "unbound_recoveries" not in after["host_collection"]
        assert item["stat_before"]["value"]["ppid"] == process.pid != parent
        assert item["poll_before"]["events"] == item["poll_after"]["events"] == []
        assert host.validate_sample(after)["host_pss_complete"]
        assert after["process_host_memory"][str(child)]["Pss_rollup"] > 3 * 1024 * 1024
        if not failed_initial_pin:
            request.node.user_properties.append(
                (
                    "actual_owned_descendant_reparenting",
                    json.dumps(
                        {"pin": original, "observed": item["stat_before"], "complete": after["host_pss_complete"]},
                        sort_keys=True,
                    ),
                )
            )
        process.stdin.write(b"c")
        process.stdin.flush()
        process.wait(timeout=5)
        assert process.returncode == 0
    finally:
        # Release both private fixture gates even if a receipt assertion fails;
        # the isolated subreaper waits for its own child, leaving no orphan.
        if process.poll() is None:
            try:
                process.stdin.write(b"pc")
                process.stdin.flush()
                process.wait(timeout=5)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                for item in sampler.tracked.values():
                    if item["fd"] is not None:
                        try:
                            sampler.backend.send_signal(item["fd"], 9)
                        except ProcessLookupError:
                            pass
                process.wait(timeout=5)
        sampler.close()
        process.stdin.close()
        process.stdout.close()


def test_unrelated_new_member_cannot_use_owner_metadata_to_gain_admission(sampler):
    value, process = sampler
    sample = value.collect()
    receipt = sample["host_collection"]
    attempt = receipt["attempts"][0]
    stranger = process.pid + 10000000
    attempt["selected_after"].append(stranger)
    read = copy.deepcopy(attempt["discovery_after"][str(process.pid)])
    read["value"].update(pid=stranger, ppid=stranger, uid=read["value"]["uid"] + 1)
    attempt["discovery_after"][str(stranger)] = read
    result = host.summarize_collection(receipt)
    assert not result["host_pss_complete"]
    assert "unproven_new_descendant" in result["host_collection_errors"]


@pytest.mark.parametrize("method", [host.LEGACY_METHOD, host.EXIT_OWNER_METHOD])
def test_v3_receipt_cannot_be_relabelled_as_prior_method(sampler, method):
    value, _ = sampler
    sample = value.collect()
    sample["host_process_memory_method"] = sample["host_collection"]["protocol"] = method
    with pytest.raises(ValueError, match="v3 identity semantics"):
        host.validate_sample(sample)


def test_v2_owner_only_change_retains_original_incomplete_classification(sampler):
    value, process = sampler
    sample = value.collect()
    receipt = sample["host_collection"]
    sample["host_process_memory_method"] = receipt["protocol"] = host.EXIT_OWNER_METHOD
    del receipt["identity_semantics"]
    receipt["attempts"][0]["discovery_after"][str(process.pid)]["value"]["uid"] += 1
    sample.update(host.summarize_collection(receipt))
    assert not host.validate_sample(sample)["host_pss_complete"]
    assert sample["host_collection_errors"] == ["discovery_identity_changed"]


@pytest.mark.parametrize("damage", ["live_owner", "birth", "parent", "no_exit", "poll_error"])
def test_final_discovery_owner_exception_is_strictly_exit_bound(sampler, damage):
    value, process = sampler
    sample = value.collect()
    # This is the immutable v2 classifier, including its owner-as-identity rule.
    sample["host_process_memory_method"] = sample["host_collection"]["protocol"] = host.EXIT_OWNER_METHOD
    del sample["host_collection"]["identity_semantics"]
    attempt = sample["host_collection"]["attempts"][0]
    key = str(process.pid)
    changed = attempt["discovery_after"][key]["value"]
    changed["uid"] += 1
    changed["state"] = "R" if damage == "live_owner" else "Z"
    if damage == "birth":
        changed["start_ticks"] += 1
    if damage == "parent":
        changed["ppid"] += 1
    attempt["post_discovery_polls"][key]["events"] = (
        [] if damage == "no_exit" else [select.POLLERR] if damage == "poll_error" else [select.POLLIN]
    )
    result = host.summarize_collection(sample["host_collection"])
    assert not result["host_pss_complete"]
    assert "discovery_identity_changed" in result["host_collection_errors"]


@pytest.mark.parametrize("damage", ["missing", "order", "downgrade", "semantics", "scope"])
def test_final_discovery_proof_receipt_cannot_be_omitted_or_relabeled(sampler, damage):
    value, process = sampler
    sample = value.collect()
    receipt = sample["host_collection"]
    attempt = receipt["attempts"][0]
    key = str(process.pid)
    if damage == "missing":
        del attempt["post_discovery_polls"]
    elif damage == "order":
        attempt["post_discovery_polls"][key]["begin_monotonic_seconds"] = 0
    elif damage == "downgrade":
        sample["host_process_memory_method"] = receipt["protocol"] = host.LEGACY_METHOD
    elif damage == "semantics":
        receipt["uid_semantics"] = "credential_uid"
    else:
        attempt["post_discovery_polls"][str(process.pid + 10000000)] = copy.deepcopy(
            attempt["post_discovery_polls"][key]
        )
    with pytest.raises(ValueError):
        host.validate_sample(sample)


def test_legacy_v1_owner_mismatch_remains_incomplete_without_new_proof(sampler):
    from tools.native_qlora_qualification.performance import sampled_memory

    value, process = sampler
    sample = value.collect()
    receipt = sample["host_collection"]
    sample["host_process_memory_method"] = receipt["protocol"] = host.LEGACY_METHOD
    del receipt["uid_semantics"]
    del receipt["identity_semantics"]
    for attempt in receipt["attempts"]:
        del attempt["post_discovery_polls"]
    changed = receipt["attempts"][0]["discovery_after"][str(process.pid)]["value"]
    changed["uid"] += 1
    changed["state"] = "Z"
    sample.update(host.summarize_collection(receipt))
    assert host.validate_sample(sample)["host_pss_complete"] is False
    assert sample["host_collection_errors"] == ["discovery_identity_changed"]
    sample.update(
        unix_seconds=time.time(),
        device_uuid="gpu",
        device_memory_total_bytes=100,
        device_memory_used_bytes=10,
        device_compute_processes=[],
    )
    assert sampled_memory([sample], expected_uuid="gpu", device_bytes=100)["host_pss_complete"] is False


@pytest.mark.parametrize("mutation", ["complete", "total", "omit", "pid", "poll_error", "retry", "stage", "timestamp"])
def test_receipt_validator_refuses_tampered_totals_or_process_proof(sampler, mutation):
    value, process = sampler
    sample = value.collect()
    attempt = sample["host_collection"]["attempts"][0]
    item = attempt["processes"][str(process.pid)]
    if mutation == "complete":
        sample["host_pss_complete"] = False
    elif mutation == "total":
        sample["selected_host_pss_sum_bytes"] += 1
    elif mutation == "omit":
        attempt["processes"].clear()
    elif mutation == "pid":
        item["pin"]["identity"]["pid"] += 1
    elif mutation == "poll_error":
        item["poll_after"]["events"] = [select.POLLERR]
    elif mutation == "retry":
        sample["host_collection"]["attempts"].append(copy.deepcopy(attempt))
        sample["host_collection"]["final_attempt"] = 1
    elif mutation == "stage":
        sample["stage"] = "other"
    else:
        item["status"]["end_monotonic_seconds"] = 0
    with pytest.raises(ValueError):
        host.validate_sample(sample)


def test_unknown_missing_process_stays_selected_and_incomplete(tmp_path):
    # Reap a real PID first: no pin may be fabricated for the absent process.
    process = subprocess.Popen([sys.executable, "-I", "-S", "-c", "pass"])
    process.wait(timeout=5)
    value = host.HostSampler(process.pid, tmp_path / "stage")
    try:
        sample = value.collect()
        assert set(sample["process_host_memory"]) == {str(process.pid)}
        assert not host.validate_sample(sample)["host_pss_complete"]
        assert "unbound_process_identity" in sample["host_collection_errors"]
    finally:
        value.close()


def test_thread_group_leader_zombie_with_live_thread_is_not_released(tmp_path):
    compiler = shutil.which("cc")
    assert compiler, "allocated Linux validation requires a C compiler for the actual pthread-exit case"
    source = tmp_path / "leader_exit.c"
    source.write_text("""#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
static void *worker(void *unused) {
  char *p = malloc(8 * 1024 * 1024);
  for (int i = 0; i < 8 * 1024 * 1024; i += 4096) p[i] = 1;
  puts("ready"); fflush(stdout);
  char c; read(0, &c, 1); free(p); return NULL;
}
int main(void) { pthread_t thread; pthread_create(&thread, NULL, worker, NULL); pthread_exit(NULL); }
""")
    binary = tmp_path / "leader_exit"
    subprocess.run([compiler, "-pthread", "-O0", str(source), "-o", str(binary)], check=True, timeout=15)
    process = subprocess.Popen([str(binary)], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    value = host.HostSampler(process.pid, tmp_path / "stage", method=host.PROCESS_IDENTITY_METHOD)
    try:
        assert process.stdout.readline() == b"ready\n"
        deadline = time.monotonic() + 5
        while host.read_proc(process.pid, "stat").get("value", {}).get("state") != "Z":
            assert time.monotonic() < deadline
            time.sleep(0.01)
        sample = value.collect()
        item = sample["host_collection"]["attempts"][0]["processes"][str(process.pid)]
        assert item["stat_before"]["value"]["state"] == "Z"
        assert item["poll_after"]["events"] == []
        assert item["exit_wait"]["initial_poll"]["events"] == []
        assert 0 < item["poll_after"]["timeout_milliseconds"] <= host.MAX_EXIT_WAIT_MILLISECONDS
        assert not host.validate_sample(sample)["host_pss_complete"]
        assert "address_space_released" not in sample["process_host_memory"][str(process.pid)]
        process.stdin.write(b"x")
        process.stdin.flush()
        process.wait(timeout=5)
        ended = value.collect()
        assert host.validate_sample(ended)["host_pss_complete"]
        assert ended["process_host_memory"][str(process.pid)]["address_space_released"]
    finally:
        value.close()
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
        process.stdin.close()
        process.stdout.close()


def test_new_receipt_performance_validation_preserves_legacy_semantics(sampler):
    from tools.native_qlora_qualification.performance import sampled_memory

    value, _ = sampler
    sample = value.collect()
    sample.update(
        unix_seconds=time.time(),
        device_uuid="gpu",
        device_memory_total_bytes=100,
        device_memory_used_bytes=10,
        device_compute_processes=[],
    )
    assert sampled_memory([sample], expected_uuid="gpu", device_bytes=100)["host_pss_complete"]
    corrupt = copy.deepcopy(sample)
    corrupt["selected_host_pss_sum_bytes"] += 1
    result = sampled_memory([corrupt], expected_uuid="gpu", device_bytes=100)
    assert not result["host_pss_complete"]
    assert "invalid_host_collection_receipt" in result["errors"]
    legacy = {
        key: value for key, value in sample.items() if key not in ("host_collection", "host_process_memory_method")
    }
    legacy["host_pss_complete"] = False
    assert not sampled_memory([legacy], expected_uuid="gpu", device_bytes=100)["host_pss_complete"]


def test_discovery_to_pin_identity_reuse_fails_closed(sampler, monkeypatch):
    value, process = sampler
    original = value._discover

    def changed_discovery():
        selected = original()
        value.discovery[str(process.pid)]["value"]["start_ticks"] += 1
        return selected

    monkeypatch.setattr(value, "_discover", changed_discovery)
    sample = value.collect()
    assert not host.validate_sample(sample)["host_pss_complete"]
    assert "unbound_process_identity" in sample["host_collection_errors"]
    assert len(sample["host_collection"]["attempts"]) == 1
    assert all("exit_wait" not in item for item in sample["host_collection"]["attempts"][0]["processes"].values())


def test_actual_live_eacces_followed_by_exit_cannot_be_hidden_by_retry(child, tmp_path, monkeypatch, request):
    process = child(nondumpable=True)
    value = host.HostSampler(process.pid, tmp_path / "stage")
    read = host.read_proc
    before = read(process.pid, "smaps_rollup")
    try:
        with without_ptrace_capability(request) as evidence:
            evidence.update(child_pr_get_dumpable=0, before_read=before)

            def error_then_exit(pid, name):
                result = read(pid, name)
                if pid == process.pid and name == "smaps_rollup":
                    evidence["dropped_read"] = result
                    assert result["error"]["errno"] == 13
                    process.terminate()
                    process.wait(timeout=5)
                return result

            monkeypatch.setattr(host, "read_proc", error_then_exit)
            sample = value.collect()
            evidence["after_exit_read"] = sample["host_collection"]["attempts"][0]["processes"][str(process.pid)][
                "stat_after"
            ]
            assert not host.validate_sample(sample)["host_pss_complete"]
            assert len(sample["host_collection"]["attempts"]) == 1
            assert "unexplained_read_error_before_exit" in sample["host_collection_errors"]
            assert "exit_wait" not in sample["host_collection"]["attempts"][0]["processes"][str(process.pid)]
    finally:
        value.close()


def test_capability_fixture_restores_exact_state_after_assertion_failure(request):
    original = _capabilities()
    original_status = _capability_status()
    with pytest.raises(AssertionError, match="deliberate fixture body failure"):
        with without_ptrace_capability(request):
            assert False, "deliberate fixture body failure"
    assert _capabilities() == original
    assert _capability_status() == original_status


def test_valid_pre_exit_attempt_peak_is_not_lost_when_final_retry_is_zero(sampler, monkeypatch):
    from tools.native_qlora_qualification.performance import sampled_memory

    value, process = sampler
    read = host.read_proc
    fired = False

    def read_then_exit(pid, name):
        nonlocal fired
        result = read(pid, name)
        if pid == process.pid and name == "smaps_rollup" and not fired:
            assert result["value"]["Pss"] > 4 * 1024 * 1024
            fired = True
            process.terminate()
            process.wait(timeout=5)
        return result

    monkeypatch.setattr(host, "read_proc", read_then_exit)
    sample = value.collect()
    assert host.validate_sample(sample)["host_pss_complete"]
    assert sample["selected_host_pss_sum_bytes"] == 0
    assert sample["host_observed_complete_attempt_pss_peak_bytes"] > 4 * 1024 * 1024
    sample.update(
        unix_seconds=time.time(),
        device_uuid="gpu",
        device_memory_total_bytes=100,
        device_memory_used_bytes=10,
        device_compute_processes=[],
    )
    result = sampled_memory([sample], expected_uuid="gpu", device_bytes=100)
    assert result["host_pss_sum_peak_bytes"] == sample["host_observed_complete_attempt_pss_peak_bytes"]


@pytest.mark.parametrize("field", ["stage_publish", "discovery_after", "rollup"])
def test_missing_receipt_parts_fail_closed(sampler, field):
    value, process = sampler
    sample = value.collect()
    receipt = sample["host_collection"]
    if field == "stage_publish":
        del receipt[field]
    elif field == "discovery_after":
        receipt["attempts"][0][field][str(process.pid)]["value"]["start_ticks"] += 1
    else:
        receipt["attempts"][0]["processes"][str(process.pid)][field] = None
    with pytest.raises(ValueError):
        host.validate_sample(sample)


def test_new_membership_cycle_is_not_a_proven_descendant(sampler):
    value, process = sampler
    sample = value.collect()
    receipt = sample["host_collection"]
    attempt = receipt["attempts"][0]
    unexpected = max(attempt["selected_after"]) + 10000000
    attempt["selected_after"].append(unexpected)
    read = copy.deepcopy(attempt["discovery_after"][str(process.pid)])
    read["value"].update(pid=unexpected, ppid=unexpected)
    attempt["discovery_after"][str(unexpected)] = read
    result = host.summarize_collection(receipt)
    assert not result["host_pss_complete"]
    assert "unproven_new_descendant" in result["host_collection_errors"]


def test_retry_start_after_bound_is_rejected(sampler):
    value, _ = sampler
    sample = value.collect()
    receipt = sample["host_collection"]
    attempt = copy.deepcopy(receipt["attempts"][0])
    attempt["begin_monotonic_seconds"] += host.MAX_COLLECTION_SECONDS + 1
    receipt["attempts"].append(attempt)
    receipt["final_attempt"] = 1
    # Check the bound independently; prior transition proof is also required.
    for record in receipt["attempts"][0]["processes"].values():
        record["poll_after"]["events"] = [select.POLLIN]
    for poll in receipt["attempts"][0]["post_discovery_polls"].values():
        poll["events"] = [select.POLLIN]
    with pytest.raises(ValueError, match="bounded collection interval"):
        host.summarize_collection(receipt)


def test_actual_delayed_last_thread_exit_waits_then_recollects(tmp_path, monkeypatch, request):
    compiler = shutil.which("cc")
    assert compiler, "actual delayed whole-process exit qualification requires a C compiler"
    source = tmp_path / "delayed_exit.c"
    source.write_text("""#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
static void *worker(void *unused) {
  char *p = malloc(8 * 1024 * 1024);
  if (!p) _exit(11);
  for (int i = 0; i < 8 * 1024 * 1024; i += 4096) p[i] = 1;
  puts("ready"); fflush(stdout);
  char c; if (read(0, &c, 1) != 1) _exit(12);
  usleep(100000); free(p); return NULL;
}
int main(void) { pthread_t t; if (pthread_create(&t, NULL, worker, NULL)) return 13; pthread_exit(NULL); }
""")
    binary = tmp_path / "delayed_exit"
    subprocess.run([compiler, "-pthread", "-O0", str(source), "-o", str(binary)], check=True, timeout=15)
    process = subprocess.Popen([str(binary)], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    sampler = host.HostSampler(process.pid, tmp_path / "stage", method=host.PROCESS_IDENTITY_METHOD)
    read, fired = host.read_proc, False
    try:
        assert process.stdout.readline() == b"ready\n"
        deadline = time.monotonic() + 5
        while read(process.pid, "stat").get("value", {}).get("state") != "Z":
            assert time.monotonic() < deadline
            time.sleep(0.001)

        def release_after_failed_memory(pid, name):
            nonlocal fired
            result = read(pid, name)
            if pid == process.pid and name == "smaps_rollup" and not fired:
                assert result["error"]["errno"] == 3
                fired = True
                process.stdin.write(b"x")
                process.stdin.flush()
            return result

        monkeypatch.setattr(host, "read_proc", release_after_failed_memory)
        sample = sampler.collect()
        attempts, key = sample["host_collection"]["attempts"], str(process.pid)
        assert len(attempts) == 2
        first, second = (attempt["processes"][key] for attempt in attempts)
        assert first["rollup"]["error"]["errno"] == 3
        assert first["poll_before"]["events"] == first["exit_wait"]["initial_poll"]["events"] == []
        assert host._released(first["poll_after"]) is True
        assert 0 < first["poll_after"]["timeout_milliseconds"] <= host.MAX_EXIT_WAIT_MILLISECONDS
        assert first["poll_after"]["end_monotonic_seconds"] - first["poll_after"]["begin_monotonic_seconds"] >= 0.05
        assert attempts[1]["begin_monotonic_seconds"] <= attempts[0]["begin_monotonic_seconds"] + 0.75
        assert host._released(second["poll_before"]) is host._released(second["poll_after"]) is True
        assert host.validate_sample(sample)["host_pss_complete"]
        assert sample["process_host_memory"][key]["address_space_released"] is True
        assert sample["selected_host_pss_sum_bytes"] == 0
        request.node.user_properties.append(("actual_delayed_group_exit_wait", json.dumps(sample, sort_keys=True)))
        process.wait(timeout=5)
        assert process.returncode == 0
    finally:
        sampler.close()
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
        process.stdin.close()
        process.stdout.close()


@pytest.mark.parametrize("damage", ["permission", "unknown_error", "birth", "missing_identity"])
def test_exit_wait_refuses_non_exit_compatible_reads(sampler, damage):
    value, process = sampler
    record = value.collect()["host_collection"]["attempts"][0]["processes"][str(process.pid)]
    record["rollup"].pop("value")
    record["rollup"]["error"] = {"type": "ProcessLookupError", "errno": 3}
    assert host._exit_wait_eligible(record)
    if damage == "permission":
        record["rollup"]["error"]["errno"] = 13
    elif damage == "unknown_error":
        record["status"].pop("value")
        record["status"]["error"] = {"type": "OSError", "errno": 5}
    elif damage == "birth":
        record["stat_after"]["value"]["start_ticks"] += 1
    else:
        record["pin"].pop("identity")
    assert not host._exit_wait_eligible(record)


@pytest.mark.parametrize("damage", ["initial_poll", "duration", "deadline", "reserve", "permission", "method"])
def test_exit_wait_receipt_cannot_invent_a_budget_or_eligibility(sampler, damage):
    value, process = sampler
    sample = value.collect()
    attempt = sample["host_collection"]["attempts"][0]
    record = attempt["processes"][str(process.pid)]
    record["rollup"].pop("value")
    record["rollup"]["error"] = {"type": "ProcessLookupError", "errno": 3}
    initial = copy.deepcopy(record["poll_after"])
    final = copy.deepcopy(initial)
    final["begin_monotonic_seconds"] = initial["end_monotonic_seconds"]
    final["timeout_milliseconds"] = 1
    record["poll_after"] = final
    record["exit_wait"] = {
        "initial_poll": initial,
        "deadline_monotonic_seconds": attempt["begin_monotonic_seconds"] + 0.75,
        "retry_reserve_seconds": 0.25,
    }
    method = host.METHOD
    if damage == "initial_poll":
        initial["events"] = [select.POLLIN]
    elif damage == "duration":
        final["timeout_milliseconds"] = 1000
    elif damage == "deadline":
        record["exit_wait"]["deadline_monotonic_seconds"] += 1
    elif damage == "reserve":
        record["exit_wait"]["retry_reserve_seconds"] = 0
    elif damage == "permission":
        record["rollup"]["error"]["errno"] = 13
    else:
        method = host.EXIT_OWNER_METHOD
    with pytest.raises(ValueError, match="exit.wait|exit wait"):
        host._validate_exit_wait(record, method=method, collection_start=attempt["begin_monotonic_seconds"])


@pytest.fixture(scope="module")
def live_thread_binary(tmp_path_factory):
    root = tmp_path_factory.mktemp("actual_live_thread_mm")
    source, binary = root / "leader.c", root / "leader"
    source.write_text(r"""
#define _GNU_SOURCE
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
static atomic_int ready, leaving;
static pid_t tid;
static void *worker(void *unused) {
  (void)unused; tid=(pid_t)syscall(SYS_gettid); atomic_store(&ready,1);
  while(!atomic_load(&leaving)) usleep(1000);
  char c; if(read(0,&c,1)!=1) _exit(11);
  if(c=='E') { execl("/bin/sleep","sleep","5",NULL); _exit(12); }
  return NULL;
}
int main(void) {
  size_t n=16UL*1024*1024;
  volatile char *p=mmap(NULL,n,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
  if(p==MAP_FAILED)return 13;
  for(size_t i=0;i<n;i+=4096)p[i]=1;
  pthread_t t;if(pthread_create(&t,NULL,worker,NULL))return 14;
  while(!atomic_load(&ready))usleep(1000);
  printf("%d %d\n",getpid(),tid);fflush(stdout);
  char c;if(read(0,&c,1)!=1)return 15;
  atomic_store(&leaving,1);pthread_exit(NULL);
}
""")
    compiler = shutil.which("cc")
    assert compiler, "actual live-thread memory validation requires a C compiler"
    subprocess.run([compiler, "-pthread", "-O0", str(source), "-o", str(binary)], check=True, timeout=15)
    return binary


@pytest.fixture
def live_thread(live_thread_binary, tmp_path):
    process = subprocess.Popen([str(live_thread_binary)], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    sampler = host.HostSampler(process.pid, tmp_path / "stage")
    try:
        pid, tid = map(int, process.stdout.readline().split())
        assert pid == process.pid and tid != pid
        # Bind the process before its leader exits; retain that original pin.
        assert sampler.collect()["host_pss_complete"]
        process.stdin.write(b"X")
        process.stdin.flush()
        deadline = time.monotonic() + 3
        while host.read_proc(pid, "stat").get("value", {}).get("state") != "Z":
            assert time.monotonic() < deadline
            time.sleep(0.001)
        yield sampler, process, tid
    finally:
        sampler.close()
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
        process.stdin.close()
        process.stdout.close()


def test_actual_live_thread_mm_reads_shared_address_space_once(live_thread, request):
    sampler, process, tid = live_thread
    descriptors = len(os.listdir("/proc/self/fd"))
    sample = sampler.collect()
    assert host.validate_sample(sample)["host_pss_complete"]
    assert len(os.listdir("/proc/self/fd")) == descriptors
    assert sample["host_collection"]["protocol"] == host.METHOD
    assert len(sample["host_collection"]["attempts"]) == 1
    item = sample["host_collection"]["attempts"][0]["processes"][str(process.pid)]
    assert item["rollup"]["error"]["errno"] == 3
    assert item["poll_before"]["events"] == item["poll_after"]["events"] == []
    proof = item["thread_mm"]
    assert proof["poll_after"]["events"] == []
    assert proof["status_before"]["value"]["Tgid"] == process.pid
    assert proof["status_before"]["value"]["Pid"] == tid
    assert sample["selected_host_pss_sum_bytes"] == proof["rollup"]["value"]["Pss"] >= 16 * 1024**2
    assert set(sample["process_host_memory"]) == {str(process.pid)}
    assert sample["process_host_memory"][str(process.pid)]["memory_tid"] == tid
    assert "address_space_released" not in sample["process_host_memory"][str(process.pid)]
    # The same retained leader read is still incomplete under the old method.
    old = copy.deepcopy(sample["host_collection"])
    old["protocol"] = host.PROCESS_IDENTITY_METHOD
    del old["attempts"][0]["processes"][str(process.pid)]["thread_mm"]
    assert not host.summarize_collection(old)["host_pss_complete"]
    request.node.user_properties.append(("actual_live_thread_shared_mm", json.dumps(sample, sort_keys=True)))


@pytest.mark.parametrize(
    "damage",
    [
        "tgid",
        "tid",
        "birth",
        "zombie",
        "directory",
        "pidfd",
        "leader_exec",
        "permission",
        "rollup_missing",
        "deadline",
        "order",
        "stage",
        "initial_permission",
        "initial_live",
        "missing_directory",
        "duplicate_tasks",
        "too_many_tasks",
    ],
)
def test_live_thread_mm_rejects_changed_or_incomplete_proof(live_thread, damage):
    sampler, process, tid = live_thread
    sample = sampler.collect()
    assert sample["host_pss_complete"]
    receipt = sample["host_collection"]
    attempt = receipt["attempts"][0]
    record = attempt["processes"][str(process.pid)]
    proof = record["thread_mm"]
    if damage == "tgid":
        proof["status_after"]["value"]["Tgid"] += 1
    elif damage == "tid":
        proof["stat_after"]["value"]["pid"] += 1
    elif damage == "birth":
        proof["stat_after"]["value"]["start_ticks"] += 1
    elif damage == "zombie":
        proof["stat_after"]["value"]["state"] = "Z"
    elif damage == "directory":
        proof["directory_after"]["value"]["inode"] += 1
    elif damage == "pidfd":
        proof["poll_after"]["events"] = [select.POLLIN]
    elif damage == "leader_exec":
        proof["leader_after"]["value"]["state"] = "R"
    elif damage == "permission":
        proof["rollup"].pop("value")
        proof["rollup"]["error"] = {"errno": 13, "type": "PermissionError"}
    elif damage == "rollup_missing":
        del proof["rollup"]["value"]["Pss"]
    elif damage == "deadline":
        proof["deadline_monotonic_seconds"] += 1
    elif damage == "order":
        proof["rollup"]["begin_monotonic_seconds"] = 0
    elif damage == "stage":
        attempt["stage_after"]["value"] = "other"
    elif damage == "initial_permission":
        record["rollup"]["error"]["errno"] = 13
    elif damage == "initial_live":
        record["stat_before"]["value"]["state"] = "S"
    elif damage == "missing_directory":
        del proof["directory"]
    elif damage == "duplicate_tasks":
        proof["directory"]["value"]["task_ids"].append(tid)
    else:
        proof["directory"]["value"]["task_ids"] = list(range(1, host.MAX_TASK_ENTRIES + 2))
    result = host.summarize_collection(receipt)
    assert not result["host_pss_complete"]
    assert (
        "stage_changed_or_unreadable" in result["host_collection_errors"]
        if damage == "stage"
        else "unproven_live_thread_memory" in result["host_collection_errors"]
    )


@pytest.mark.parametrize("method", [host.LEGACY_METHOD, host.EXIT_OWNER_METHOD, host.PROCESS_IDENTITY_METHOD])
def test_live_thread_proof_cannot_be_relabelled_as_old_method(live_thread, method):
    sampler, _, _ = live_thread
    receipt = sampler.collect()["host_collection"]
    receipt["protocol"] = method
    with pytest.raises(ValueError):
        host.summarize_collection(receipt)


@pytest.mark.parametrize("action", [b"Q", b"E"])
def test_actual_thread_exit_or_exec_during_read_fails_closed(live_thread, monkeypatch, action):
    sampler, process, _ = live_thread
    original = host._thread_read
    fired = False

    def read(fd, name):
        nonlocal fired
        result = original(fd, name)
        if name == "smaps_rollup" and not fired:
            fired = True
            assert result["value"]["Pss"] >= 16 * 1024**2
            process.stdin.write(action)
            process.stdin.flush()
            deadline = time.monotonic() + 3
            if action == b"Q":
                process.wait(timeout=3)
            else:
                while host.read_proc(process.pid, "stat").get("value", {}).get("state") == "Z":
                    assert time.monotonic() < deadline
                    time.sleep(0.001)
        return result

    monkeypatch.setattr(host, "_thread_read", read)
    sample = sampler.collect()
    assert fired
    assert not host.validate_sample(sample)["host_pss_complete"]
    assert "unproven_live_thread_memory" in sample["host_collection_errors"]


def test_live_thread_failed_directory_open_closes_and_retains_error(live_thread, monkeypatch):
    sampler, _, _ = live_thread
    original = os.open

    def opened(path, *args, **kwargs):
        if "/task/" in str(path):
            raise PermissionError(13, "fixture")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(host.os, "open", opened)
    descriptors = len(os.listdir("/proc/self/fd"))
    sample = sampler.collect()
    assert not host.validate_sample(sample)["host_pss_complete"]
    assert len(os.listdir("/proc/self/fd")) == descriptors


def _controlled_thread_exit(
    live_thread, monkeypatch, *, error=3, mismatch=False, release=True, reap_before_discovery=False
):
    """Inject a bounded ESRCH read; use the real held pidfd for final release."""
    sampler, process, _ = live_thread
    read, discover = host._thread_read, sampler._discover
    discoveries = 0

    def failed_rollup(fd, name):
        result = read(fd, name)
        if name == "smaps_rollup":
            assert result["value"]["Pss"] >= 16 * 1024**2
            result.pop("value")
            result["error"] = {"errno": error, "type": "ProcessLookupError" if error == 3 else "PermissionError"}
        elif name == "status":
            result["value"].pop("VmRSS", None)
            if mismatch:
                result["value"]["Tgid"] += 1
        return result

    def after_read():
        nonlocal discoveries
        discoveries += 1
        if not reap_before_discovery:
            result = discover()
        if discoveries == 2 and release:
            process.stdin.write(b"Q")
            process.stdin.flush()
            process.wait(timeout=3)
            assert process.returncode == 0
        if reap_before_discovery:
            result = discover()
        return result

    monkeypatch.setattr(host, "_thread_read", failed_rollup)
    monkeypatch.setattr(sampler, "_discover", after_read)
    return sampler.collect(), process.pid


def test_actual_pidfd_exit_after_esrch_thread_read_requires_new_collection(live_thread, monkeypatch, request):
    sample, pid = _controlled_thread_exit(live_thread, monkeypatch)
    receipt = sample["host_collection"]
    assert receipt["protocol"] == host.METHOD
    assert len(receipt["attempts"]) == 2
    first, second = receipt["attempts"]
    failed = first["processes"][str(pid)]
    assert failed["thread_mm"]["rollup"]["error"]["errno"] == 3
    assert host._released(failed["thread_mm"]["poll_after"]) is False
    assert host._released(first["post_discovery_polls"][str(pid)]) is True
    assert second["begin_monotonic_seconds"] <= first["begin_monotonic_seconds"] + host.MAX_COLLECTION_SECONDS
    assert host.validate_sample(sample)["host_pss_complete"]
    assert sample["process_host_memory"][str(pid)]["address_space_released"] is True
    assert sample["selected_host_pss_sum_bytes"] == 0
    # The failed attempt itself never becomes a memory observation.
    single = copy.deepcopy(receipt)
    single["attempts"] = single["attempts"][:1]
    single["final_attempt"] = 0
    result = host.summarize_collection(single)
    assert not result["host_pss_complete"]
    assert result["process_host_memory"][str(pid)] == {}
    assert result["host_observed_complete_attempt_pss_peak_bytes"] is None
    # v4 remains incomplete and cannot accept the new retry provenance.
    single["protocol"] = host.LIVE_THREAD_METHOD
    old = host.summarize_collection(single)
    assert not old["host_pss_complete"] and "unproven_live_thread_memory" in old["host_collection_errors"]
    receipt_old = copy.deepcopy(receipt)
    receipt_old["protocol"] = host.LIVE_THREAD_METHOD
    with pytest.raises(ValueError, match="retry lacks a proven membership transition"):
        host.summarize_collection(receipt_old)
    request.node.user_properties.append(("actual_pidfd_thread_exit_recollection", json.dumps(sample, sort_keys=True)))


@pytest.mark.parametrize("damage", ["permission", "mismatch", "no_release"])
def test_thread_exit_recollection_refuses_unproven_reads(live_thread, monkeypatch, damage):
    sample, _ = _controlled_thread_exit(
        live_thread,
        monkeypatch,
        error=13 if damage == "permission" else 3,
        mismatch=damage == "mismatch",
        release=damage != "no_release",
    )
    assert len(sample["host_collection"]["attempts"]) == 1
    assert not host.validate_sample(sample)["host_pss_complete"]
    assert "unproven_live_thread_memory" in sample["host_collection_errors"]


@pytest.mark.parametrize(
    "damage", ["late_release", "pidfd_unknown", "birth", "directory", "thread_poll", "rollup_unknown"]
)
def test_thread_exit_retry_receipt_cannot_hide_tampering(live_thread, monkeypatch, damage):
    sample, pid = _controlled_thread_exit(live_thread, monkeypatch)
    assert sample["host_pss_complete"]
    receipt = sample["host_collection"]
    first = receipt["attempts"][0]
    proof = first["processes"][str(pid)]["thread_mm"]
    if damage == "late_release":
        final = first["post_discovery_polls"][str(pid)]
        final["end_monotonic_seconds"] = first["begin_monotonic_seconds"] + host.MAX_COLLECTION_SECONDS
        with pytest.raises(ValueError, match="in-budget"):
            host._thread_mm_memory(
                first["processes"][str(pid)],
                collection_start=first["begin_monotonic_seconds"],
                discovery_after=first["discovery_after"][str(pid)],
                released_poll=final,
            )
    elif damage == "pidfd_unknown":
        first["post_discovery_polls"][str(pid)]["events"] = [select.POLLERR]
    elif damage == "birth":
        proof["stat_after"]["value"]["start_ticks"] += 1
    elif damage == "directory":
        proof["directory_after"]["value"]["inode"] += 1
    elif damage == "thread_poll":
        proof["poll_after"]["events"] = [select.POLLIN]
    else:
        proof["rollup"]["error"]["errno"] = 5
    with pytest.raises(ValueError):
        host.summarize_collection(receipt)


def _delayed_thread_exit(live_thread, monkeypatch, *, release=True, damage=None, method=host.METHOD):
    """Only the ESRCH read is injected; the wait observes an actual delayed pidfd release."""
    sampler, process, _ = live_thread
    sampler.method = method
    poll, thread_mm = host._poll, host._thread_mm
    timers, waits = [], []
    proof_done = False

    def observed_thread(*args):
        nonlocal proof_done
        proof = thread_mm(*args)
        proof_done = True
        if damage == "birth":
            proof["stat_after"]["value"]["start_ticks"] += 1
        elif damage == "directory":
            proof["directory_after"]["value"]["inode"] += 1
        elif damage == "leader":
            proof["leader_after"]["value"]["start_ticks"] += 1
        elif damage == "deadline":
            # Consume the original budget; never invent a later deadline.
            time.sleep(host.MAX_COLLECTION_SECONDS - host.EXIT_WAIT_RETRY_RESERVE_SECONDS + 0.02)
        return proof

    def stop():
        process.stdin.write(b"Q")
        process.stdin.flush()

    def observed_poll(fd, *, timeout_milliseconds=0):
        if timeout_milliseconds:
            assert proof_done and fd == sampler.tracked[process.pid]["fd"]
            assert host._released(poll(fd)) is False
            waits.append(timeout_milliseconds)
            if release:
                timer = threading.Timer(0.025, stop)
                timers.append(timer)
                timer.start()
        result = poll(fd, timeout_milliseconds=timeout_milliseconds)
        if damage == "unknown_poll" and proof_done:
            result["events"] = [select.POLLERR]
        return result

    monkeypatch.setattr(host, "_thread_mm", observed_thread)
    monkeypatch.setattr(host, "_poll", observed_poll)
    try:
        sample, pid = _controlled_thread_exit(
            live_thread,
            monkeypatch,
            error=13 if damage == "permission" else 3,
            mismatch=damage == "tgid",
            release=False,
        )
    finally:
        for timer in timers:
            timer.join(timeout=3)
            assert not timer.is_alive()
    if waits and release:
        assert process.wait(timeout=3) == 0
    return sample, pid, waits


def test_actual_thread_exit_wait_recollects_only_after_same_pidfd_release(live_thread, monkeypatch, request):
    sample, pid, waits = _delayed_thread_exit(live_thread, monkeypatch)
    assert len(waits) == 1 and 0 < waits[0] <= host.MAX_EXIT_WAIT_MILLISECONDS
    receipt = sample["host_collection"]
    assert receipt["protocol"] == host.METHOD
    assert len(receipt["attempts"]) == 2
    first, second = receipt["attempts"]
    wait = first["thread_exit_waits"][str(pid)]
    assert host._released(wait["initial_poll"]) is False
    assert host._released(first["post_discovery_polls"][str(pid)]) is True
    assert first["processes"][str(pid)]["thread_mm"]["rollup"]["error"]["errno"] == 3
    final = first["post_discovery_polls"][str(pid)]
    assert final["end_monotonic_seconds"] + host.EXIT_WAIT_RETRY_RESERVE_SECONDS <= wait["deadline_monotonic_seconds"]
    assert final["end_monotonic_seconds"] <= second["begin_monotonic_seconds"] <= wait["deadline_monotonic_seconds"]
    assert host.validate_sample(sample)["host_pss_complete"]
    assert sample["process_host_memory"][str(pid)]["address_space_released"] is True
    first_only = copy.deepcopy(receipt)
    first_only["attempts"] = [first]
    first_only["final_attempt"] = 0
    derived = host.summarize_collection(first_only)
    assert not derived["host_pss_complete"]
    assert derived["process_host_memory"][str(pid)] == {}
    assert derived["host_observed_complete_attempt_pss_peak_bytes"] is None
    for old_method in (host.LIVE_THREAD_METHOD, host.THREAD_EXIT_METHOD):
        first_only["protocol"] = old_method
        with pytest.raises(ValueError, match="older method"):
            host.summarize_collection(first_only)
    request.node.user_properties.append(("actual_delayed_thread_pidfd_wait", json.dumps(sample, sort_keys=True)))


@pytest.mark.parametrize("method", [host.THREAD_EXIT_METHOD, host.METHOD])
def test_thread_exit_wait_without_release_preserves_missing_memory(live_thread, monkeypatch, method):
    sample, pid, waits = _delayed_thread_exit(live_thread, monkeypatch, release=False, method=method)
    assert len(waits) == (1 if method == host.METHOD else 0)
    assert len(sample["host_collection"]["attempts"]) == 1
    assert not host.validate_sample(sample)["host_pss_complete"]
    assert sample["process_host_memory"][str(pid)] == {"rollup_unavailable": True}
    assert "unproven_live_thread_memory" in sample["host_collection_errors"]


@pytest.mark.parametrize("damage", ["permission", "tgid", "birth", "directory", "leader", "deadline", "unknown_poll"])
def test_thread_exit_wait_requires_complete_identity_and_original_budget(live_thread, monkeypatch, damage):
    sample, _, waits = _delayed_thread_exit(live_thread, monkeypatch, damage=damage)
    assert waits == []
    assert len(sample["host_collection"]["attempts"]) == 1
    assert not host.validate_sample(sample)["host_pss_complete"]


@pytest.mark.parametrize(
    "damage", ["missing_wait", "initial_ready", "unknown_final", "extended_deadline", "reserve", "timeout", "late"]
)
def test_thread_exit_wait_receipt_cannot_invent_readiness_or_time(live_thread, monkeypatch, damage):
    sample, pid, _ = _delayed_thread_exit(live_thread, monkeypatch)
    first = sample["host_collection"]["attempts"][0]
    wait = first["thread_exit_waits"][str(pid)]
    final = first["post_discovery_polls"][str(pid)]
    if damage == "missing_wait":
        del first["thread_exit_waits"]
    elif damage == "initial_ready":
        wait["initial_poll"]["events"] = [select.POLLIN]
    elif damage == "unknown_final":
        final["events"] = [select.POLLERR]
    elif damage == "extended_deadline":
        wait["deadline_monotonic_seconds"] += 1
    elif damage == "reserve":
        wait["retry_reserve_seconds"] = 0
    elif damage == "timeout":
        final["timeout_milliseconds"] = host.MAX_EXIT_WAIT_MILLISECONDS + 1
    else:
        final["end_monotonic_seconds"] = wait["deadline_monotonic_seconds"]
    with pytest.raises(ValueError):
        host.summarize_collection(sample["host_collection"])


def test_actual_reaped_thread_group_requires_whole_recollection(live_thread, monkeypatch, request):
    """Control ESRCH; actually exit/reap after the verified leader read, before discovery."""
    sample, pid = _controlled_thread_exit(live_thread, monkeypatch, reap_before_discovery=True)
    receipt = sample["host_collection"]
    assert receipt["protocol"] == host.METHOD
    assert len(receipt["attempts"]) == 2
    first, second = receipt["attempts"]
    record = first["processes"][str(pid)]
    proof = record["thread_mm"]
    discovery, final = first["discovery_after"][str(pid)], first["post_discovery_polls"][str(pid)]
    assert proof["leader_after"]["value"]["state"] == "Z"
    assert proof["rollup"]["error"]["errno"] == errno.ESRCH
    assert discovery["error"] == {"type": "FileNotFoundError", "errno": errno.ENOENT}
    assert host._released(proof["poll_after"]) is False and host._released(final) is True
    assert final["end_monotonic_seconds"] + host.EXIT_WAIT_RETRY_RESERVE_SECONDS <= proof["deadline_monotonic_seconds"]
    assert final["end_monotonic_seconds"] <= second["begin_monotonic_seconds"] <= proof["deadline_monotonic_seconds"]
    assert host.validate_sample(sample)["host_pss_complete"]
    assert sample["process_host_memory"][str(pid)]["address_space_released"] is True
    failed = copy.deepcopy(receipt)
    failed["attempts"] = [first]
    failed["final_attempt"] = 0
    derived = host.summarize_collection(failed)
    assert not derived["host_pss_complete"] and derived["process_host_memory"][str(pid)] == {}
    assert derived["host_observed_complete_attempt_pss_peak_bytes"] is None
    for method in (host.LIVE_THREAD_METHOD, host.THREAD_EXIT_METHOD, host.THREAD_WAIT_METHOD):
        failed["protocol"] = method
        old = host.summarize_collection(failed)
        assert not old["host_pss_complete"] and "unproven_live_thread_memory" in old["host_collection_errors"]
        relabeled = copy.deepcopy(receipt)
        relabeled["protocol"] = method
        with pytest.raises(ValueError, match="retry lacks a proven membership transition"):
            host.summarize_collection(relabeled)
    request.node.user_properties.append(("actual_reaped_thread_group_recollection", json.dumps(sample, sort_keys=True)))


@pytest.mark.parametrize("method", [host.THREAD_EXIT_METHOD, host.THREAD_WAIT_METHOD])
def test_old_collectors_keep_reaped_thread_discovery_incomplete(live_thread, monkeypatch, method):
    live_thread[0].method = method
    sample, _ = _controlled_thread_exit(live_thread, monkeypatch, reap_before_discovery=True)
    assert len(sample["host_collection"]["attempts"]) == 1
    assert not host.validate_sample(sample)["host_pss_complete"]
    assert "unproven_live_thread_memory" in sample["host_collection_errors"]


@pytest.mark.parametrize(
    "damage",
    [
        "discovery_permission",
        "discovery_esrch",
        "discovery_missing",
        "discovery_type",
        "discovery_identity",
        "leader_missing",
        "leader_birth",
        "leader_exec",
        "thread_birth",
        "tgid",
        "directory",
        "rollup_permission",
        "no_release",
        "unknown_poll",
        "late_release",
        "reversed_reads",
        "outside_deadline",
    ],
)
def test_reaped_thread_retry_requires_exact_identity_release_and_deadline(live_thread, monkeypatch, damage):
    sample, pid = _controlled_thread_exit(live_thread, monkeypatch, reap_before_discovery=True)
    assert host.validate_sample(sample)["host_pss_complete"]
    receipt = sample["host_collection"]
    first = receipt["attempts"][0]
    record = first["processes"][str(pid)]
    proof = record["thread_mm"]
    discovery, final = first["discovery_after"][str(pid)], first["post_discovery_polls"][str(pid)]
    if damage == "discovery_permission":
        discovery["error"] = {"type": "PermissionError", "errno": errno.EACCES}
    elif damage == "discovery_esrch":
        discovery["error"] = {"type": "ProcessLookupError", "errno": errno.ESRCH}
    elif damage == "discovery_missing":
        del first["discovery_after"][str(pid)]
    elif damage == "discovery_type":
        discovery["error"]["type"] = "UnknownError"
    elif damage == "discovery_identity":
        discovery.pop("error")
        discovery["value"] = copy.deepcopy(proof["leader_after"]["value"])
        discovery["value"]["start_ticks"] += 1
    elif damage == "leader_missing":
        proof["leader_after"].pop("value")
        proof["leader_after"]["error"] = {"type": "FileNotFoundError", "errno": errno.ENOENT}
    elif damage == "leader_birth":
        proof["leader_after"]["value"]["start_ticks"] += 1
    elif damage == "leader_exec":
        proof["leader_after"]["value"]["state"] = "R"
    elif damage == "thread_birth":
        proof["stat_after"]["value"]["start_ticks"] += 1
    elif damage == "tgid":
        proof["status_after"]["value"]["Tgid"] += 1
    elif damage == "directory":
        proof["directory_after"]["value"]["inode"] += 1
    elif damage == "rollup_permission":
        proof["rollup"]["error"] = {"type": "PermissionError", "errno": errno.EACCES}
    elif damage == "no_release":
        final["events"] = []
    elif damage == "unknown_poll":
        final["events"] = [select.POLLERR]
    elif damage == "late_release":
        final["end_monotonic_seconds"] = proof["deadline_monotonic_seconds"]
    elif damage == "reversed_reads":
        discovery["begin_monotonic_seconds"] = proof["leader_after"]["begin_monotonic_seconds"] - 0.001
    else:
        proof["deadline_monotonic_seconds"] += 1
    with pytest.raises(ValueError):
        host.summarize_collection(receipt)


@contextmanager
def _vanished_before_pin(tmp_path, monkeypatch, *, method=host.METHOD, count=1):
    """A real parent reaps its child between discovery and the first pin read."""
    code = """import subprocess,sys
p=subprocess.Popen([sys.executable,'-I','-S','-c','import time; time.sleep(60)']); children=[p]
children.extend(
 subprocess.Popen([sys.executable,'-I','-S','-c','import time; time.sleep(60)'])
 for _ in range(int(sys.argv[1])-1))
print(','.join(str(p.pid) for p in children),flush=True)
try:
 sys.stdin.buffer.read(1)
 for p in children: p.terminate(); p.wait(timeout=5)
 print('reaped',flush=True)
 while sys.stdin.buffer.read(1)==b'n':
  c=subprocess.Popen([sys.executable,'-I','-S','-c','import time; time.sleep(60)'])
  children.append(c); print(c.pid,flush=True)
finally:
 for p in children:
  if p.poll() is None: p.kill(); p.wait(timeout=5)
"""
    parent = subprocess.Popen(
        [sys.executable, "-I", "-S", "-c", code, str(count)], stdin=subprocess.PIPE, stdout=subprocess.PIPE
    )
    targets = [int(p) for p in parent.stdout.readline().split(b",")]
    target = targets[0]
    marker = tmp_path / "stage"
    marker.write_text("training")
    sampler = host.HostSampler(parent.pid, marker, method=method)
    pin = sampler._pin
    fired = False

    def reap_before_pin(pid, discovery):
        nonlocal fired
        if pid in targets and not fired:
            assert discovery["value"]["pid"] == pid
            fired = True
            parent.stdin.write(b"x")
            parent.stdin.flush()
            assert parent.stdout.readline() == b"reaped\n"
        pin(pid, discovery)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(sampler, "_pin", reap_before_pin)
            failed = sampler.collect()
        assert fired and not host.validate_sample(failed)["host_pss_complete"]
        assert sampler.tracked[target]["fd"] is None
        assert "identity" not in sampler.tracked[target]["pin"]
        assert sampler.tracked[target]["pin"]["before"]["error"]["errno"] == errno.ENOENT
        assert "Pss_rollup" not in failed["process_host_memory"][str(target)]
        sampler._fixture_parent = parent
        sampler._fixture_targets = targets
        yield sampler, failed, target
    finally:
        sampler.close()
        if parent.poll() is None:
            parent.stdin.write(b"xx")
            parent.stdin.flush()
        parent.wait(timeout=5)
        parent.stdin.close()
        parent.stdout.close()


def test_actual_unbound_disappearance_recovers_only_in_new_collection(tmp_path, monkeypatch, request):
    with _vanished_before_pin(tmp_path, monkeypatch) as (sampler, failed, pid):
        unchanged = json.dumps(failed, sort_keys=True)
        recovered = sampler.collect()
        assert host.validate_sample(recovered)["host_pss_complete"]
        proof = recovered["host_collection"]["unbound_recoveries"][str(pid)]
        assert proof["result"] == "absent"
        assert proof["pidfd_open"]["flags"] == 0
        assert proof["pidfd_open"]["error"] == {"type": "ProcessLookupError", "errno": errno.ESRCH}
        assert pid not in sampler.tracked and str(pid) not in recovered["process_host_memory"]
        assert "address_space_released" not in proof
        assert json.dumps(failed, sort_keys=True) == unchanged and not host.validate_sample(failed)["host_pss_complete"]
        for _ in range(3):
            later = sampler.collect()
            assert host.validate_sample(later)["host_pss_complete"]
            assert "unbound_recoveries" not in later["host_collection"]
        request.node.user_properties.append(
            (
                "actual_unbound_disappearance_recollection",
                json.dumps(
                    {
                        "failed": failed,
                        "recovered": recovered,
                        "actual_child_reaped_before_pin": True,
                        "proc_or_syscall_errors_injected": False,
                        "previous_interval_repaired": False,
                    },
                    sort_keys=True,
                ),
            )
        )


@pytest.mark.parametrize(
    "method",
    [
        host.PROCESS_IDENTITY_METHOD,
        host.LIVE_THREAD_METHOD,
        host.THREAD_EXIT_METHOD,
        host.THREAD_WAIT_METHOD,
        host.REAPED_THREAD_METHOD,
    ],
)
def test_historical_unbound_disappearance_remains_incomplete(tmp_path, monkeypatch, method):
    with _vanished_before_pin(tmp_path, monkeypatch, method=method) as (sampler, failed, pid):
        later = sampler.collect()
        assert not host.validate_sample(later)["host_pss_complete"]
        assert pid in sampler.tracked and "unbound_recoveries" not in later["host_collection"]
        assert "unbound_process_identity" in later["host_collection_errors"]


@pytest.mark.parametrize("code", [errno.EACCES, errno.EPERM, errno.EINVAL, errno.EMFILE, errno.ENOSYS])
def test_unbound_recovery_non_esrch_open_errors_never_retire(tmp_path, monkeypatch, code):
    with _vanished_before_pin(tmp_path, monkeypatch) as (sampler, failed, pid):
        original = sampler.backend.open

        def error(target):
            if target == pid:
                raise OSError(code, "controlled unavailable pidfd")
            return original(target)

        monkeypatch.setattr(sampler.backend, "open", error)
        sample = sampler.collect()
        assert not host.validate_sample(sample)["host_pss_complete"] and pid in sampler.tracked
        assert sample["host_collection"]["unbound_recoveries"][str(pid)]["result"] == "unresolved"


@pytest.mark.parametrize("mutation", ["nspid", "self_pid", "mount", "own_namespace", "proc_root", "unreadable"])
def test_unbound_namespace_ambiguity_never_retires(tmp_path, monkeypatch, mutation):
    with _vanished_before_pin(tmp_path, monkeypatch) as (sampler, failed, pid):
        read = host._read_pid_namespace
        calls = 0

        def changed():
            nonlocal calls
            value = read()
            calls += 1
            if mutation == "nspid":
                value["value"]["nspid"].append(1)
            elif mutation == "self_pid":
                value["value"]["proc_self"] += 1
            elif mutation == "mount" and calls == 2:
                value["value"]["mount_namespace"][1] += 1
            elif mutation == "own_namespace" and calls == 2:
                value["value"]["pid_namespace"][1] += 1
            elif mutation == "proc_root" and calls == 2:
                value["value"]["proc_root"][1] += 1
            elif mutation == "unreadable":
                value.pop("value")
                value["error"] = {"type": "PermissionError", "errno": errno.EACCES}
            return value

        monkeypatch.setattr(host, "_read_pid_namespace", changed)
        sample = sampler.collect()
        assert not host.validate_sample(sample)["host_pss_complete"] and pid in sampler.tracked


@pytest.mark.parametrize(
    "mutation",
    [
        "legacy",
        "flags",
        "open_errno",
        "open_pid",
        "proc_errno",
        "namespace",
        "prior_time",
        "order",
        "deadline",
        "omitted_proof",
        "forged_identity",
    ],
)
def test_unbound_recovery_receipt_tampering_rejected(tmp_path, monkeypatch, mutation):
    with _vanished_before_pin(tmp_path, monkeypatch) as (sampler, failed, pid):
        sample = sampler.collect()
        assert host.validate_sample(sample)["host_pss_complete"]
        receipt = sample["host_collection"]
        proof = receipt["unbound_recoveries"][str(pid)]
        if mutation == "legacy":
            sample["host_process_memory_method"] = receipt["protocol"] = host.REAPED_THREAD_METHOD
        elif mutation == "flags":
            proof["pidfd_open"]["flags"] = 1
        elif mutation == "open_errno":
            proof["pidfd_open"]["error"]["errno"] = errno.EINVAL
        elif mutation == "open_pid":
            proof["pidfd_open"]["pid"] += 1
        elif mutation == "proc_errno":
            proof["stat_after"]["error"]["errno"] = errno.EACCES
        elif mutation == "namespace":
            proof["namespace_after"]["value"]["nspid"].append(1)
        elif mutation == "prior_time":
            proof["failed_collection_end_monotonic_seconds"] = receipt["collection_begin_monotonic_seconds"] + 1
        elif mutation == "order":
            proof["pidfd_open"]["begin_monotonic_seconds"] = 0
        elif mutation == "deadline":
            receipt["collection_begin_monotonic_seconds"] -= 1
        elif mutation == "omitted_proof":
            del receipt["unbound_recoveries"]
        elif mutation == "forged_identity":
            proof["failed_pin"]["identity"] = proof["failed_pin"]["discovery"]["value"]
        with pytest.raises(ValueError):
            host.validate_sample(sample)


def test_slow_unbound_recovery_is_incomplete_without_collector_crash(tmp_path, monkeypatch):
    with _vanished_before_pin(tmp_path, monkeypatch) as (sampler, failed, pid):
        read = host._read_pid_namespace
        delayed = False

        def slow():
            nonlocal delayed
            if not delayed:
                delayed = True
                time.sleep(host.MAX_COLLECTION_SECONDS + 0.02)
            return read()

        with monkeypatch.context() as patch:
            patch.setattr(host, "_read_pid_namespace", slow)
            sample = sampler.collect()
        assert not host.validate_sample(sample)["host_pss_complete"] and pid in sampler.tracked
        assert "unbound_recovery_collection_deadline_exceeded" in sample["host_collection_errors"]
        assert sample["host_collection"]["unbound_recoveries"][str(pid)]["result"] == "unresolved"
        assert host.validate_sample(sampler.collect())["host_pss_complete"]


def _recover_then_new_descendant(sampler, monkeypatch):
    original = sampler._process
    spawned = None

    def start_after_first_read(pid):
        nonlocal spawned
        result = original(pid)
        if spawned is None:
            parent = sampler._fixture_parent
            parent.stdin.write(b"n")
            parent.stdin.flush()
            spawned = int(parent.stdout.readline())
        return result

    with monkeypatch.context() as patch:
        patch.setattr(sampler, "_process", start_after_first_read)
        sample = sampler.collect()
    return sample, spawned


def test_unbound_prefix_allows_fresh_membership_transition_retry(tmp_path, monkeypatch):
    with _vanished_before_pin(tmp_path, monkeypatch) as (sampler, failed, pid):
        sample, new_pid = _recover_then_new_descendant(sampler, monkeypatch)
        assert host.validate_sample(sample)["host_pss_complete"]
        assert len(sample["host_collection"]["attempts"]) == 2
        assert sample["host_collection"]["unbound_recoveries"][str(pid)]["result"] == "absent"
        assert sample["process_host_memory"][str(new_pid)]["Pss_rollup"] > 0
        assert sampler.tracked[new_pid]["pin"]["identity"]["ppid"] == sampler.root_pid


def test_simulated_pid_reappearance_is_retained_incomplete_not_decoder_crash(tmp_path, monkeypatch):
    with _vanished_before_pin(tmp_path, monkeypatch) as (sampler, failed, old_pid):
        sample, new_pid = _recover_then_new_descendant(sampler, monkeypatch)
        assert host.validate_sample(sample)["host_pss_complete"]
        receipt = sample["host_collection"]
        # Model immediate PID reuse without attempting to wrap the machine's
        # real PID allocator. Fresh descendant reads/ancestry remain actual.
        proof = receipt["unbound_recoveries"].pop(str(old_pid))
        proof["failed_pin"]["discovery"]["value"]["pid"] = new_pid
        proof["pidfd_open"]["pid"] = new_pid
        receipt["unbound_recoveries"][str(new_pid)] = proof
        # The fresh complete tree observation contains the new process. This
        # models reuse before that observation, not a retry after a hard error.
        receipt["attempts"] = [receipt["attempts"][-1]]
        receipt["final_attempt"] = 0
        derived = host.summarize_collection(receipt)
        assert not derived["host_pss_complete"] and "unbound_absent_pid_reappeared" in derived["host_collection_errors"]
        assert str(new_pid) in derived["process_host_memory"]
        sample.update(derived)
        assert not host.validate_sample(sample)["host_pss_complete"]


def test_multiple_unbound_recoveries_survive_sorted_json_roundtrip(tmp_path, monkeypatch):
    with _vanished_before_pin(tmp_path, monkeypatch, count=2) as (sampler, failed, pid):
        assert all(sampler.tracked[p]["fd"] is None for p in sampler._fixture_targets)
        # Exercise the opposite of JSON's numeric-string key order deliberately.
        sampler.tracked = dict(sorted(sampler.tracked.items(), reverse=True))
        sample = sampler.collect()
        assert len(sample["host_collection"]["unbound_recoveries"]) == 2
        decoded = json.loads(json.dumps(sample, sort_keys=True))
        assert host.validate_sample(decoded)["host_pss_complete"]


@pytest.mark.parametrize("mutation", ["post_open_failure", "changed_birth", "initial_permission"])
def test_unqualified_failed_pins_are_not_recovered(tmp_path, monkeypatch, mutation):
    with _vanished_before_pin(tmp_path, monkeypatch) as (sampler, failed, pid):
        pin = sampler.tracked[pid]["pin"]
        if mutation == "post_open_failure":
            pin["after"] = copy.deepcopy(pin["before"])
        elif mutation == "initial_permission":
            pin["error"] = {"type": "PermissionError", "errno": errno.EACCES}
        else:
            pin["before"].pop("error")
            pin["before"]["value"] = dict(
                pin["discovery"]["value"], start_ticks=pin["discovery"]["value"]["start_ticks"] + 1
            )
        sample = sampler.collect()
        assert not host.validate_sample(sample)["host_pss_complete"]
        assert pid in sampler.tracked and "unbound_recoveries" not in sample["host_collection"]


def test_live_replacement_birth_closes_temporary_pidfd_and_remains_incomplete(tmp_path, monkeypatch):
    with _vanished_before_pin(tmp_path, monkeypatch) as (sampler, failed, pid):
        read, opened, temporary = host.read_proc, sampler.backend.open, []

        def changed_birth(target, name):
            if target == pid and name == "stat":
                result = read(sampler.root_pid, "stat")
                result["value"]["pid"] = pid
                result["value"]["start_ticks"] = sampler.tracked[pid]["pin"]["discovery"]["value"]["start_ticks"] + 1
                return result
            return read(target, name)

        def current_replacement(target):
            fd = opened(sampler.root_pid if target == pid else target)
            if target == pid:
                temporary.append(fd)
            return fd

        # Controlled PID-reuse analogue: real live-parent descriptor and reads,
        # mapped to the disappeared PID with its different birth identifier.
        with monkeypatch.context() as patch:
            patch.setattr(host, "read_proc", changed_birth)
            patch.setattr(sampler.backend, "open", current_replacement)
            sample = sampler.collect()
        assert not host.validate_sample(sample)["host_pss_complete"] and pid in sampler.tracked
        assert sample["host_collection"]["unbound_recoveries"][str(pid)]["result"] == "unresolved"
        assert len(temporary) == 1
        with pytest.raises(OSError) as error:
            os.fstat(temporary[0])
        assert error.value.errno == errno.EBADF


def test_v9_live_thread_still_measures_shared_mm_instead_of_inferring_exit(live_thread):
    sampler, process, tid = live_thread
    sampler.method = host.TEARDOWN_METHOD
    sample = sampler.collect()
    assert host.validate_sample(sample)["host_pss_complete"]
    memory = sample["process_host_memory"][str(process.pid)]
    assert memory["memory_tid"] == tid and memory["Pss_rollup"] >= 16 * 1024**2
    assert "address_space_released" not in memory
    receipt = sample["host_collection"]
    proof = receipt["attempts"][0]["processes"][str(process.pid)]["thread_mm"]
    assert proof["deadline_monotonic_seconds"] == receipt["attempts"][0]["begin_monotonic_seconds"] + 4.0


def test_v9_thread_exit_requires_same_pidfd_release_then_new_observation(live_thread, monkeypatch):
    sample, pid, waits = _delayed_thread_exit(live_thread, monkeypatch, method=host.TEARDOWN_METHOD)
    assert len(waits) == 1 and 350 < waits[0] <= 3500
    assert host.validate_sample(sample)["host_pss_complete"]
    first, second = sample["host_collection"]["attempts"]
    assert host._released(first["post_discovery_polls"][str(pid)]) is True
    assert second["begin_monotonic_seconds"] >= first["post_discovery_polls"][str(pid)]["end_monotonic_seconds"]
    first_only = copy.deepcopy(sample["host_collection"])
    first_only["attempts"], first_only["final_attempt"] = [first], 0
    assert not host.summarize_collection(first_only)["host_pss_complete"]
    malformed = copy.deepcopy(sample["host_collection"])
    malformed["attempts"][0]["post_discovery_polls"][str(pid)]["timeout_milliseconds"] = 3501
    with pytest.raises(ValueError, match="in-budget"):
        host.summarize_collection(malformed)
