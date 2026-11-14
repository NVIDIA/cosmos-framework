# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""Auditable, non-atomic Linux process-tree memory samples, without Torch/NVML.

Only flags=0 process pidfd POLLIN proves that the entire thread group exited.
A zombie leader alone does not prove the shared address space was released:
https://github.com/torvalds/linux/blob/v5.15/kernel/fork.c#L1748-L1765
All reads and failed attempts remain in the receipt. A retry is permitted only
for a proven exit or newly observed descendant. Unreadable live processes,
identity changes and stage races are incomplete, never interpreted as zero.
V3 binds PID/start to the process pidfd; proc inode ownership is retained
only as mutable metadata, never used to infer identity or released memory:
https://github.com/torvalds/linux/blob/v5.15/fs/proc/base.c#L1709-L1763
Ancestry admits new descendants. Already pinned descendants remain owned if
their parent exits and they are reparented; PPID is not process-birth identity.
"""

from __future__ import annotations

import errno
import os
import select
import time
from pathlib import Path
from typing import Any

try:
    from .processes import _PidfdBackend
except ImportError:  # Direct monitor.py invocation from the sealed tool directory.
    from processes import _PidfdBackend

LEGACY_METHOD = "proc-identity-state-read-attempts-v1"
EXIT_OWNER_METHOD = "proc-identity-state-read-attempts-v2"
PROCESS_IDENTITY_METHOD = "proc-identity-state-read-attempts-v3"
LIVE_THREAD_METHOD = "proc-identity-state-read-attempts-v4"
THREAD_EXIT_METHOD = "proc-identity-state-read-attempts-v5"
THREAD_WAIT_METHOD = "proc-identity-state-read-attempts-v6"
REAPED_THREAD_METHOD = "proc-identity-state-read-attempts-v7"
METHOD = "proc-identity-state-read-attempts-v8"
TEARDOWN_METHOD = "proc-identity-state-read-attempts-v9"
MEMBERSHIP_RETRY_METHOD = "proc-identity-state-read-attempts-v10"
IDENTITY_SEMANTICS = "pid_start_pidfd_owned_ancestry"
THREAD_MM_PROTOCOL = "proc-live-thread-shared-mm-v1"
MAX_TASK_ENTRIES = 4096
MAX_ATTEMPTS = 3
MAX_COLLECTION_SECONDS = 0.75
MAX_EXIT_WAIT_MILLISECONDS = 350
EXIT_WAIT_RETRY_RESERVE_SECONDS = 0.25
TEARDOWN_COLLECTION_SECONDS = 4.0
MEMBERSHIP_RETRY_COLLECTION_SECONDS = 4.5
TEARDOWN_EXIT_WAIT_MILLISECONDS = 3500


def _collection_seconds(method: str) -> float:
    """Versioned acquisition budgets never upgrade historical observations."""
    if method == MEMBERSHIP_RETRY_METHOD:
        return MEMBERSHIP_RETRY_COLLECTION_SECONDS
    return TEARDOWN_COLLECTION_SECONDS if method == TEARDOWN_METHOD else MAX_COLLECTION_SECONDS


def _exit_wait_milliseconds(method: str) -> int:
    return (
        TEARDOWN_EXIT_WAIT_MILLISECONDS
        if method in (TEARDOWN_METHOD, MEMBERSHIP_RETRY_METHOD)
        else MAX_EXIT_WAIT_MILLISECONDS
    )


def _error(error: Exception) -> dict:
    return {"type": type(error).__name__, "errno": getattr(error, "errno", None)}


def read_proc(pid: int, name: str) -> dict:
    result: dict[str, Any] = {"begin_unix_seconds": time.time(), "begin_monotonic_seconds": time.monotonic()}
    try:
        path = Path("/proc") / str(pid) / name
        text = path.read_text()
        if name == "stat":
            tail = text.rsplit(")", 1)[1].split()
            value = {
                "pid": pid,
                "state": tail[0],
                "ppid": int(tail[1]),
                "start_ticks": int(tail[19]),
                # Historical wire name: this is proc inode ownership, not a
                # credential UID or immutable process identity. Linux
                # task_dump_owner() derives it from credentials/dumpability/mm;
                # it can change while the same process remains alive. Also,
                # this stat() is later than the text read, not an atomic snapshot.
                "uid": path.stat().st_uid,
            }
        else:
            fields = {"VmRSS", "VmHWM"} if name == "status" else {"Pss", "Rss"}
            value = {}
            for line in text.splitlines():
                key, _, raw = line.partition(":")
                if key in fields:
                    parts = raw.split()
                    if len(parts) != 2 or parts[1] != "kB" or int(parts[0]) < 0:
                        raise ValueError("invalid proc memory field")
                    value[key] = int(parts[0]) * 1024
        result["value"] = value
    except (OSError, ValueError, IndexError) as error:
        result["error"] = _error(error)
    result.update(end_unix_seconds=time.time(), end_monotonic_seconds=time.monotonic())
    return result


def _identity(value: dict, *, owner_is_identity: bool = True) -> tuple:
    # Keep the historical tuple exclusively for decoding v1/v2 evidence.
    fields = ("pid", "start_ticks", "uid") if owner_is_identity else ("pid", "start_ticks")
    return tuple(value.get(k) for k in fields)


def _valid_identity(value: Any, *, owner_is_identity: bool = True) -> bool:
    fields = ("pid", "start_ticks", "uid") if owner_is_identity else ("pid", "start_ticks", "ppid", "uid")
    return isinstance(value, dict) and all(type(value.get(k)) is int and value[k] >= 0 for k in fields)


def _released_owner_transition(value: Any, expected: Any, released: bool) -> bool:
    """An owner-only change is admissible solely for this proven exited task.

    A zombie leader is insufficient: `released` comes from the identity-bound
    process pidfd after final discovery, which waits for the entire group.
    Live owner changes, reparenting and changed birth identities remain errors.
    https://github.com/torvalds/linux/blob/v5.15/fs/proc/base.c#L1846-L1870
    """
    return (
        released
        and _valid_identity(value)
        and _valid_identity(expected)
        and value.get("state") in ("Z", "X")
        and all(value.get(key) == expected.get(key) for key in ("pid", "start_ticks", "ppid"))
        and type(value.get("ppid")) is int
        and value["uid"] != expected["uid"]
    )


def _read_stage(path: Path) -> dict:
    result = {"begin_unix_seconds": time.time(), "begin_monotonic_seconds": time.monotonic()}
    try:
        result["value"] = path.read_text().strip()
    except FileNotFoundError:
        result["value"] = "startup"
    except OSError as error:
        result["error"] = _error(error)
    result.update(end_unix_seconds=time.time(), end_monotonic_seconds=time.monotonic())
    return result


def _poll(fd: int | None, *, timeout_milliseconds: int = 0) -> dict:
    result = {"begin_unix_seconds": time.time(), "begin_monotonic_seconds": time.monotonic()}
    if timeout_milliseconds:
        result["timeout_milliseconds"] = timeout_milliseconds
    try:
        if fd is None:
            raise OSError(errno.EBADF, "no identity-bound pidfd")
        poller = select.poll()
        poller.register(fd, select.POLLIN)
        result["events"] = [mask for _, mask in poller.poll(timeout_milliseconds)]
    except OSError as error:
        result["error"] = _error(error)
    result.update(end_unix_seconds=time.time(), end_monotonic_seconds=time.monotonic())
    return result


def _released(poll: dict) -> bool | None:
    masks = poll.get("events")
    if "error" in poll or not isinstance(masks, list) or any(type(mask) is not int for mask in masks):
        return None
    if any(mask & ~(select.POLLIN | select.POLLHUP) or not mask & select.POLLIN for mask in masks):
        return None
    return any(mask & select.POLLIN for mask in masks)


def _receipt_valid(read: Any) -> bool:
    if not isinstance(read, dict):
        return False
    stamps = [
        read.get(key)
        for key in ("begin_monotonic_seconds", "end_monotonic_seconds", "begin_unix_seconds", "end_unix_seconds")
    ]
    return (
        all(type(value) in (int, float) and 0 <= value < float("inf") for value in stamps)
        and stamps[0] <= stamps[1]
        and (("error" in read) != ("value" in read or "events" in read))
    )


def _enoent(read: dict) -> bool:
    return _receipt_valid(read) and read.get("error") == {"type": "FileNotFoundError", "errno": errno.ENOENT}


def _unbound_pin(pin: dict) -> bool:
    """Only a failed initial binding, never a valid pin or changed birth."""
    discovery, before = pin.get("discovery", {}), pin.get("before", {})
    return (
        pin.get("error") == {"type": "ProcessLookupError", "errno": errno.ESRCH}
        and pin.get("flags") == 0
        and "identity" not in pin
        and "after" not in pin
        and _receipt_valid(discovery)
        and _valid_identity(discovery.get("value"), owner_is_identity=False)
        and _receipt_valid(before)
        and discovery["end_monotonic_seconds"] <= before["begin_monotonic_seconds"]
        and (
            _enoent(before)
            or (
                _valid_identity(before.get("value"), owner_is_identity=False)
                and _identity(before["value"], owner_is_identity=False)
                == _identity(discovery["value"], owner_is_identity=False)
            )
        )
    )


def _read_pid_namespace() -> dict:
    """NSpid starts at the procfs mount's namespace, ends at the task's.

    A singleton matching getpid proves the required namespace alignment without
    access to /proc/1/ns/pid. Retain own namespace/mount and proc-root identities.
    https://github.com/torvalds/linux/blob/v5.15/fs/proc/array.c#L192-L198
    """
    result = {"begin_unix_seconds": time.time(), "begin_monotonic_seconds": time.monotonic()}
    try:
        fields = dict(line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines())
        value = {
            "pid": os.getpid(),
            "proc_self": int(os.readlink("/proc/self")),
            "status_pid": int(fields["Pid"]),
            "status_tgid": int(fields["Tgid"]),
            "nspid": [int(n) for n in fields["NSpid"].split()],
        }
        for name, path in (
            ("pid_namespace", "/proc/thread-self/ns/pid"),
            ("mount_namespace", "/proc/self/ns/mnt"),
            ("proc_root", "/proc"),
        ):
            stat = os.stat(path)
            value[name] = [stat.st_dev, stat.st_ino]
        result["value"] = value
    except (OSError, ValueError, KeyError) as error:
        result["error"] = _error(error)
    result.update(end_unix_seconds=time.time(), end_monotonic_seconds=time.monotonic())
    return result


def _namespace_aligned(value: dict) -> bool:
    pid = value.get("pid")
    return (
        type(pid) is int
        and pid > 0
        and value.get("proc_self") == value.get("status_pid") == value.get("status_tgid") == pid
        and value.get("nspid") == [pid]
        and all(
            isinstance(value.get(k), list) and len(value[k]) == 2 and all(type(n) is int and n >= 0 for n in value[k])
            for k in ("pid_namespace", "mount_namespace", "proc_root")
        )
    )


def _unbound_recovery_result(proof: dict) -> str:
    """Current namespace absence is NOT a historical same-pidfd exit proof.

    Linux5.15 returns ESRCH only when find_get_pid in the caller's active PID
    namespace fails; EINVAL/permission/API errors do not establish absence.
    https://github.com/torvalds/linux/blob/v5.15/kernel/pid.c#L556-L573
    """
    failed = proof["failed_pin"]
    names = ("namespace_before", "stat_before", "pidfd_open", "stat_after", "namespace_after")
    reads = [proof[n] for n in names]
    if (
        not _unbound_pin(failed)
        or not all(_receipt_valid(r) for r in reads)
        or any(a["end_monotonic_seconds"] > b["begin_monotonic_seconds"] for a, b in zip(reads, reads[1:]))
    ):
        return "unresolved"
    namespace = reads[0].get("value", {})
    if not _namespace_aligned(namespace) or namespace != reads[-1].get("value"):
        return "unresolved"
    before, opened, after = reads[1:4]
    if opened.get("pid") != failed["discovery"]["value"]["pid"] or opened.get("flags") != 0:
        return "unresolved"
    if opened.get("backend") not in ("python_pidfd", "linux_x86_64_pidfd_syscalls"):
        return "unresolved"
    if (
        _enoent(before)
        and _enoent(after)
        and opened.get("error") == {"type": "ProcessLookupError", "errno": errno.ESRCH}
    ):
        return "absent"
    expected = _identity(failed["discovery"]["value"], owner_is_identity=False)
    if opened.get("value") == {"opened": True} and all(
        _valid_identity(r.get("value"), owner_is_identity=False)
        and _identity(r["value"], owner_is_identity=False) == expected
        for r in (before, after)
    ):
        return "rebound"
    return "unresolved"


def _validate_unbound_recoveries(receipt: dict) -> tuple[float | None, set[int]]:
    proofs = receipt.get("unbound_recoveries")
    if proofs is None:
        if "collection_begin_monotonic_seconds" in receipt:
            raise ValueError("unbound recovery start lacks proof")
        return None, set()
    start = receipt.get("collection_begin_monotonic_seconds")
    if (
        receipt["protocol"] not in (METHOD, TEARDOWN_METHOD, MEMBERSHIP_RETRY_METHOD)
        or not isinstance(proofs, dict)
        or not proofs
        or type(start) not in (float, int)
    ):
        raise ValueError("invalid unbound recovery protocol")
    first = receipt["attempts"][0]
    preceding = start
    absent = set()
    # JSON key sorting must not change the chronological recovery order.
    for pid, proof in sorted(proofs.items(), key=lambda item: item[1]["namespace_before"]["begin_monotonic_seconds"]):
        if not pid.isdecimal() or int(pid) in (receipt["root_pid"], receipt["excluded_monitor_pid"]):
            raise ValueError("unbound recovery cannot retire the scope root or monitor")
        failed = proof["failed_pin"]
        reported = proof["failed_collection_end_monotonic_seconds"]
        if (
            not _unbound_pin(failed)
            or failed["discovery"]["value"]["pid"] != int(pid)
            or not failed["before"]["end_monotonic_seconds"] <= reported < start
        ):
            raise ValueError("unbound recovery lacks a prior failed binding")
        begin, end = proof["namespace_before"], proof["namespace_after"]
        if (
            not preceding
            <= begin["begin_monotonic_seconds"]
            <= end["end_monotonic_seconds"]
            <= first["begin_monotonic_seconds"]
        ):
            raise ValueError("unbound recovery outside new collection")
        decided = proof["decision_monotonic_seconds"]
        if not end["end_monotonic_seconds"] <= decided <= first["begin_monotonic_seconds"]:
            raise ValueError("unbound recovery decision outside proof interval")
        preceding = decided
        result = _unbound_recovery_result(proof)
        if decided + EXIT_WAIT_RETRY_RESERVE_SECONDS > start + _collection_seconds(receipt["protocol"]):
            result = "unresolved"
        if proof["result"] != result:
            raise ValueError("unbound recovery result disagrees with actual reads")
        if result == "absent":
            absent.add(int(pid))
        elif result == "rebound":
            pin = first["processes"].get(pid, {}).get("pin", {})
            expected = {
                "discovery": failed["discovery"],
                "before": proof["stat_before"],
                "after": proof["stat_after"],
                "identity": proof["stat_before"]["value"],
                "flags": 0,
                "backend": proof["pidfd_open"]["backend"],
            }
            if pin != expected:
                raise ValueError("fresh pin does not preserve the originally discovered birth and ancestry")
        elif first["processes"].get(pid, {}).get("pin") != failed:
            raise ValueError("unresolved task silently omitted or rebound")
    return start, absent


def classify_process(
    record: dict, *, released_after_discovery: bool = False, owner_is_identity: bool = True
) -> tuple[dict, list[str], bool]:
    """Recompute memory, hard errors and proven exit transition from raw reads."""
    errors: list[str] = []
    pin = record.get("pin", {})
    expected = pin.get("identity")
    if not _valid_identity(expected, owner_is_identity=owner_is_identity) or pin.get("flags") != 0 or pin.get("error"):
        errors.append("unbound_process_identity")
    for key in ("discovery", "before", "after"):
        read = pin.get(key, {})
        if (
            not _receipt_valid(read)
            or not _valid_identity(read.get("value"), owner_is_identity=owner_is_identity)
            or _identity(read["value"], owner_is_identity=owner_is_identity)
            != _identity(expected or {}, owner_is_identity=owner_is_identity)
        ):
            errors.append("invalid_pidfd_binding")
    if all(_receipt_valid(pin.get(key)) for key in ("discovery", "before", "after")):
        if any(
            pin[left]["end_monotonic_seconds"] > pin[right]["begin_monotonic_seconds"]
            for left, right in (("discovery", "before"), ("before", "after"))
        ):
            errors.append("invalid_pidfd_binding_order")
    reads = {
        name: record.get(name, {})
        for name in ("stat_before", "status", "rollup", "stat_after", "poll_before", "poll_after")
    }
    if not all(_receipt_valid(read) for read in reads.values()):
        errors.append("invalid_read_receipt")
    else:
        order = ("stat_before", "poll_before", "status", "rollup", "stat_after", "poll_after")
        if any(
            reads[left]["end_monotonic_seconds"] > reads[right]["begin_monotonic_seconds"]
            for left, right in zip(order, order[1:])
        ):
            errors.append("invalid_read_order")
    for name in ("stat_before", "stat_after"):
        value = reads[name].get("value")
        if (
            value is not None
            and (
                not _valid_identity(value, owner_is_identity=owner_is_identity)
                or _identity(value, owner_is_identity=owner_is_identity)
                != _identity(expected or {}, owner_is_identity=owner_is_identity)
            )
            and not (owner_is_identity and _released_owner_transition(value, expected, released_after_discovery))
        ):
            errors.append("process_identity_changed")
    before, after = _released(reads["poll_before"]), _released(reads["poll_after"])
    if before is None or after is None or (before is True and after is not True):
        errors.append("invalid_pidfd_exit_proof")
    transition = before is False and after is True
    if transition:
        for name in ("stat_before", "status", "rollup", "stat_after"):
            error = reads[name].get("error")
            if error is not None and error.get("errno") not in (errno.ENOENT, errno.ESRCH):
                errors.append("unexplained_read_error_before_exit")
    status, rollup = reads["status"].get("value", {}), reads["rollup"].get("value", {})
    memory = {key: status[key] for key in ("VmRSS", "VmHWM") if type(status.get(key)) is int and status[key] >= 0}
    memory.update(
        {key + "_rollup": rollup[key] for key in ("Pss", "Rss") if type(rollup.get(key)) is int and rollup[key] >= 0}
    )
    if before is True and after is True and not errors:
        # pidfd POLLIN requires the last thread to exit; never infer this from Z.
        memory.update(VmRSS=0, Pss_rollup=0, Rss_rollup=0, address_space_released=True)
    elif not transition:
        for name in ("stat_before", "stat_after"):
            if "error" in reads[name]:
                errors.append("live_or_unknown_identity_unreadable")
        if "VmRSS" not in memory or "Pss_rollup" not in memory or "Rss_rollup" not in memory:
            errors.append("live_or_unknown_memory_unreadable")
    if errors or transition:
        memory["rollup_unavailable"] = "Pss_rollup" not in memory
    return memory, sorted(set(errors)), transition


def _exit_wait_eligible(record: dict) -> bool:
    """Wait only for exit-compatible missing memory on a valid, still-owned pidfd."""
    _, errors, transition = classify_process(record, owner_is_identity=False)
    if (
        transition
        or "live_or_unknown_memory_unreadable" not in errors
        or set(errors) - {"live_or_unknown_memory_unreadable", "live_or_unknown_identity_unreadable"}
        or _released(record["poll_before"]) is not False
        or _released(record["poll_after"]) is not False
    ):
        return False
    # Permission failures and malformed/unknown reads must remain hard failures,
    # even if the same process happens to exit shortly afterward.
    return all(
        read.get("error", {}).get("errno") in (errno.ENOENT, errno.ESRCH) if "error" in read else True
        for read in (record[name] for name in ("stat_before", "status", "rollup", "stat_after"))
    )


def _thread_mm_eligible(record: dict) -> bool:
    """A narrowly identified dead leader, not a generic unreadable process."""
    _, errors, transition = classify_process(record, owner_is_identity=False)
    return (
        not transition
        and errors == ["live_or_unknown_memory_unreadable"]
        and "exit_wait" not in record
        and all(record[name].get("value", {}).get("state") == "Z" for name in ("stat_before", "stat_after"))
        and "error" not in record["status"]
        and record["rollup"].get("error", {}).get("errno") == errno.ESRCH
        and _released(record["poll_before"]) is False
        and _released(record["poll_after"]) is False
    )


def _timed_read(function) -> dict:
    result = {"begin_unix_seconds": time.time(), "begin_monotonic_seconds": time.monotonic()}
    try:
        result["value"] = function()
    except (OSError, ValueError, IndexError) as error:
        result["error"] = _error(error)
    result.update(end_unix_seconds=time.time(), end_monotonic_seconds=time.monotonic())
    return result


def _thread_read(directory_fd: int, name: str) -> dict:
    """Read through one held proc task directory, never a re-resolved TID path."""

    def read():
        fd = os.open(name, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW, dir_fd=directory_fd)
        with os.fdopen(fd) as stream:
            text = stream.read()
        if name == "stat":
            tail = text.rsplit(")", 1)[1].split()
            return {"pid": int(text.split("(", 1)[0]), "state": tail[0], "start_ticks": int(tail[19])}
        value = {}
        for line in text.splitlines():
            key, _, raw = line.partition(":")
            if name == "status" and key in ("Pid", "Tgid"):
                value[key] = int(raw.strip())
            elif name == "status" and key == "State":
                value[key] = raw.split()[0]
            elif key in (("VmRSS", "VmHWM") if name == "status" else ("Rss", "Pss")):
                parts = raw.split()
                if len(parts) != 2 or parts[1] != "kB" or int(parts[0]) < 0:
                    raise ValueError("invalid thread memory field")
                value[key] = int(parts[0]) * 1024
        return value

    return _timed_read(read)


def _thread_mm(pid: int, pidfd: int, deadline: float) -> dict:
    """One actual shared-mm observation, counted once for the owned TGID.

    Linux5.15 CLONE_THREAD requires CLONE_VM; both proc paths use the same
    smaps_rollup operations. A nonleader exec changes its TID via de_thread;
    the held directory and before/after identities make that a failed read.
    https://github.com/torvalds/linux/blob/v5.15/kernel/fork.c#L1836-L1849
    https://github.com/torvalds/linux/blob/v5.15/fs/proc/base.c#L3325-L3365
    https://github.com/torvalds/linux/blob/v5.15/fs/exec.c#L1035-L1083
    This is non-atomic sampled memory, not an address-space snapshot.
    """
    proof = {"protocol": THREAD_MM_PROTOCOL, "deadline_monotonic_seconds": deadline}
    directory_fd = None

    def open_task():
        nonlocal directory_fd
        if time.monotonic() >= deadline:
            raise TimeoutError("thread observation deadline exhausted")
        # One candidate only: a disappearing/dead first member fails closed;
        # no scan for a more favorable memory result or summing shared mm.
        with os.scandir(f"/proc/{pid}/task") as entries:
            tids = []
            for entry in entries:
                if not entry.name.isdigit() or len(tids) >= MAX_TASK_ENTRIES:
                    raise ValueError("invalid or excessive task inventory")
                tids.append(int(entry.name))
        tids.sort()
        if pid not in tids or len(tids) < 2:
            raise ValueError("no retained thread-group member")
        tid = next(value for value in tids if value != pid)
        directory_fd = os.open(f"/proc/{pid}/task/{tid}", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW)
        stat = os.fstat(directory_fd)
        return {"task_ids": tids, "tid": tid, "device": stat.st_dev, "inode": stat.st_ino}

    proof["directory"] = _timed_read(open_task)
    try:
        if "value" in proof["directory"]:
            for key, name in (
                ("stat_before", "stat"),
                ("status_before", "status"),
                ("rollup", "smaps_rollup"),
                ("status_after", "status"),
                ("stat_after", "stat"),
            ):
                proof[key] = _thread_read(directory_fd, name)

            def directory_identity():
                stat = os.fstat(directory_fd)
                return {"device": stat.st_dev, "inode": stat.st_ino}

            proof["directory_after"] = _timed_read(directory_identity)
    finally:
        if directory_fd is not None:
            os.close(directory_fd)
    proof["leader_after"] = read_proc(pid, "stat")
    proof["poll_after"] = _poll(pidfd)
    return proof


def _thread_mm_identity(
    record: dict,
    *,
    collection_start: float,
    discovery_after: dict,
    allow_failed_rollup: bool = False,
    allow_discovery_enoent: bool = False,
    collection_seconds: float = MAX_COLLECTION_SECONDS,
) -> dict:
    """Validate the held thread/leader proof, optionally allowing only an ESRCH rollup."""
    proof = record["thread_mm"]
    order = (
        "directory",
        "stat_before",
        "status_before",
        "rollup",
        "status_after",
        "stat_after",
        "directory_after",
        "leader_after",
        "poll_after",
    )
    if (
        not _thread_mm_eligible(record)
        or not isinstance(proof, dict)
        or set(proof) != {"protocol", "deadline_monotonic_seconds", *order}
        or proof["protocol"] != THREAD_MM_PROTOCOL
        or proof["deadline_monotonic_seconds"] != collection_start + collection_seconds
        or not all(
            _receipt_valid(proof[key])
            and (
                "error" not in proof[key]
                or (allow_failed_rollup and key == "rollup" and proof[key]["error"].get("errno") == errno.ESRCH)
            )
            for key in order
        )
    ):
        raise ValueError("unproven live-thread memory read")
    reads = [record["poll_after"], *(proof[key] for key in order), discovery_after]
    if (
        any(a["end_monotonic_seconds"] > b["begin_monotonic_seconds"] for a, b in zip(reads, reads[1:]))
        or proof["poll_after"]["end_monotonic_seconds"] > proof["deadline_monotonic_seconds"]
        or _released(proof["poll_after"]) is not False
    ):
        raise ValueError("invalid live-thread read interval")
    expected = record["pin"]["identity"]
    for read in (proof["leader_after"], discovery_after):
        if (
            read is discovery_after
            and allow_failed_rollup
            and allow_discovery_enoent
            and _receipt_valid(read)
            and read.get("error") == {"type": "FileNotFoundError", "errno": errno.ENOENT}
        ):
            # Only a v7 caller with an actual same-pidfd release may accept
            # disappearance AFTER our own identity-valid leader read. This
            # supplies no memory; the entire attempt must be recollected.
            continue
        value = read.get("value", {})
        if (
            not _valid_identity(value, owner_is_identity=False)
            or _identity(value, owner_is_identity=False) != _identity(expected, owner_is_identity=False)
            or value.get("state") != "Z"
        ):
            raise ValueError("thread-group leader changed during memory read")
    directory = proof["directory"]["value"]
    tid, tids = directory.get("tid"), directory.get("task_ids")
    if (
        type(tid) is not int
        or tid == expected["pid"]
        or not isinstance(tids, list)
        or not 2 <= len(tids) <= MAX_TASK_ENTRIES
        or any(type(value) is not int or value <= 0 for value in tids)
        or tids != sorted(set(tids))
        or expected["pid"] not in tids
        or tid != next(value for value in tids if value != expected["pid"])
        or any(type(directory.get(k)) is not int or directory[k] < 0 for k in ("device", "inode"))
        or proof["directory_after"]["value"] != {k: directory[k] for k in ("device", "inode")}
    ):
        raise ValueError("invalid held task directory")
    before, after = (proof[key]["value"] for key in ("stat_before", "stat_after"))
    if (
        any(
            type(value.get("pid")) is not int
            or value["pid"] != tid
            or type(value.get("start_ticks")) is not int
            or value["start_ticks"] <= 0
            or value.get("state") not in ("R", "S", "D", "I")
            for value in (before, after)
        )
        or before["start_ticks"] != after["start_ticks"]
    ):
        raise ValueError("live thread identity changed")
    for key in ("status_before", "status_after"):
        value = proof[key]["value"]
        if (
            type(value.get("Pid")) is not int
            or value["Pid"] != tid
            or type(value.get("Tgid")) is not int
            or value["Tgid"] != expected["pid"]
            or value.get("State") not in ("R", "S", "D", "I")
            or (
                not (allow_failed_rollup and "VmRSS" not in value)
                and (type(value.get("VmRSS")) is not int or value["VmRSS"] < 0)
            )
        ):
            raise ValueError("live thread group or memory status changed")
    return proof


def _thread_mm_memory(
    record: dict,
    *,
    collection_start: float,
    discovery_after: dict,
    released_poll: dict | None = None,
    allow_discovery_enoent: bool = False,
    collection_seconds: float = MAX_COLLECTION_SECONDS,
) -> dict:
    """Read shared-mm values, or prove an exit transition with NO memory."""
    if allow_discovery_enoent and (
        released_poll is None
        or not _receipt_valid(released_poll)
        or _released(released_poll) is not True
        or not _receipt_valid(discovery_after)
        or discovery_after["end_monotonic_seconds"] > released_poll["begin_monotonic_seconds"]
        or released_poll["end_monotonic_seconds"] + EXIT_WAIT_RETRY_RESERVE_SECONDS
        > collection_start + collection_seconds
    ):
        raise ValueError("discovery disappearance lacks an in-budget same-pidfd release")
    proof = _thread_mm_identity(
        record,
        collection_start=collection_start,
        discovery_after=discovery_after,
        allow_failed_rollup=released_poll is not None,
        allow_discovery_enoent=allow_discovery_enoent,
        collection_seconds=collection_seconds,
    )
    if released_poll is not None:
        if (
            not _receipt_valid(released_poll)
            or _released(released_poll) is not True
            or discovery_after["end_monotonic_seconds"] > released_poll["begin_monotonic_seconds"]
            or released_poll["end_monotonic_seconds"] + EXIT_WAIT_RETRY_RESERVE_SECONDS
            > proof["deadline_monotonic_seconds"]
            or proof["rollup"].get("error", {}).get("errno") != errno.ESRCH
        ):
            raise ValueError("thread read lacks an in-budget same-pidfd exit transition")
        return {}  # No PSS/RSS/zero inferred for this failed read interval.
    rollup = proof["rollup"]["value"]
    if any(type(rollup.get(k)) is not int or rollup[k] < 0 for k in ("Pss", "Rss")):
        raise ValueError("unreadable live-thread rollup")
    status = proof["status_before"]["value"]
    return {
        "VmRSS": status["VmRSS"],
        "Pss_rollup": rollup["Pss"],
        "Rss_rollup": rollup["Rss"],
        "memory_source": "verified_live_thread_shared_mm",
        "memory_tid": proof["directory"]["value"]["tid"],
    }


def _thread_exit_wait_eligible(
    record: dict, *, collection_start: float, discovery_after: dict, collection_seconds: float = MAX_COLLECTION_SECONDS
) -> bool:
    """A wait needs complete identity evidence; it does not prove exit or memory."""
    try:
        proof = _thread_mm_identity(
            record,
            collection_start=collection_start,
            discovery_after=discovery_after,
            allow_failed_rollup=True,
            collection_seconds=collection_seconds,
        )
        return proof["rollup"].get("error", {}).get("errno") == errno.ESRCH
    except (ValueError, KeyError, TypeError, StopIteration):
        return False


def _exit_wait_timeout(deadline: float, *, method: str = METHOD) -> int:
    remaining = deadline - time.monotonic() - EXIT_WAIT_RETRY_RESERVE_SECONDS
    return min(_exit_wait_milliseconds(method), int(remaining * 1000) - 1)


def _validate_exit_wait(record: dict, *, method: str, collection_start: float) -> None:
    """Optional v3 acquisition detail; historical reads retain identical semantics."""
    wait = record.get("exit_wait")
    if wait is None:
        if "timeout_milliseconds" in record.get("poll_after", {}):
            raise ValueError("timed pidfd poll lacks its initial read and budget")
        return
    if (
        method
        not in (
            PROCESS_IDENTITY_METHOD,
            LIVE_THREAD_METHOD,
            THREAD_EXIT_METHOD,
            THREAD_WAIT_METHOD,
            REAPED_THREAD_METHOD,
            METHOD,
            TEARDOWN_METHOD,
            MEMBERSHIP_RETRY_METHOD,
        )
        or not isinstance(wait, dict)
        or set(wait) != {"initial_poll", "deadline_monotonic_seconds", "retry_reserve_seconds"}
        or wait["retry_reserve_seconds"] != EXIT_WAIT_RETRY_RESERVE_SECONDS
        or wait["deadline_monotonic_seconds"] != collection_start + _collection_seconds(method)
    ):
        raise ValueError("invalid exit-wait bounds or method")
    initial, final = wait["initial_poll"], record.get("poll_after", {})
    timeout = final.get("timeout_milliseconds")
    if (
        not _receipt_valid(initial)
        or not _receipt_valid(final)
        or type(timeout) is not int
        or not 0 < timeout <= _exit_wait_milliseconds(method)
        or record["stat_after"]["end_monotonic_seconds"] > initial["begin_monotonic_seconds"]
        or initial["end_monotonic_seconds"] > final["begin_monotonic_seconds"]
        or final["begin_monotonic_seconds"] + timeout / 1000
        > wait["deadline_monotonic_seconds"] - EXIT_WAIT_RETRY_RESERVE_SECONDS
        or not _exit_wait_eligible({**record, "poll_after": initial})
    ):
        raise ValueError("exit wait was not justified by owned exit-compatible reads")


def _validate_thread_exit_waits(attempt: dict, *, method: str, collection_start: float) -> None:
    waits = attempt.get("thread_exit_waits", {})
    if "thread_exit_waits" in attempt and method not in (
        THREAD_WAIT_METHOD,
        REAPED_THREAD_METHOD,
        METHOD,
        TEARDOWN_METHOD,
        MEMBERSHIP_RETRY_METHOD,
    ):
        raise ValueError("thread exit wait cannot be relabeled as an older method")
    if not isinstance(waits, dict) or not set(waits) <= set(attempt["processes"]):
        raise ValueError("invalid thread exit wait ownership")
    if method not in (THREAD_WAIT_METHOD, REAPED_THREAD_METHOD, METHOD, TEARDOWN_METHOD, MEMBERSHIP_RETRY_METHOD):
        return
    for pid, final in attempt["post_discovery_polls"].items():
        wait = waits.get(pid)
        if wait is None:
            if "timeout_milliseconds" in final:
                raise ValueError("timed thread exit poll lacks its initial read and budget")
            continue
        if (
            not isinstance(wait, dict)
            or set(wait) != {"initial_poll", "deadline_monotonic_seconds", "retry_reserve_seconds"}
            or wait["deadline_monotonic_seconds"] != collection_start + _collection_seconds(method)
            or wait["retry_reserve_seconds"] != EXIT_WAIT_RETRY_RESERVE_SECONDS
        ):
            raise ValueError("invalid thread exit wait bounds")
        initial, timeout = wait["initial_poll"], final.get("timeout_milliseconds")
        discovery = attempt["discovery_after"][pid]
        if (
            not _receipt_valid(initial)
            or _released(initial) is not False
            or discovery["end_monotonic_seconds"] > initial["begin_monotonic_seconds"]
            or initial["end_monotonic_seconds"] > final["begin_monotonic_seconds"]
            or type(timeout) is not int
            or not 0 < timeout <= _exit_wait_milliseconds(method)
            or final["begin_monotonic_seconds"] + timeout / 1000
            > wait["deadline_monotonic_seconds"] - EXIT_WAIT_RETRY_RESERVE_SECONDS
            or not _thread_exit_wait_eligible(
                attempt["processes"][pid],
                collection_start=collection_start,
                discovery_after=discovery,
                collection_seconds=_collection_seconds(method),
            )
        ):
            raise ValueError("thread exit wait lacks an in-budget owned thread failure")


def summarize_collection(receipt: dict) -> dict:
    """Fail closed and independently derive the final attempt's legacy fields."""
    method = receipt.get("protocol")
    if method not in (
        LEGACY_METHOD,
        EXIT_OWNER_METHOD,
        PROCESS_IDENTITY_METHOD,
        LIVE_THREAD_METHOD,
        THREAD_EXIT_METHOD,
        THREAD_WAIT_METHOD,
        REAPED_THREAD_METHOD,
        METHOD,
        TEARDOWN_METHOD,
        MEMBERSHIP_RETRY_METHOD,
    ):
        raise ValueError("unsupported host collection protocol")
    owner_is_identity = method not in (
        PROCESS_IDENTITY_METHOD,
        LIVE_THREAD_METHOD,
        THREAD_EXIT_METHOD,
        THREAD_WAIT_METHOD,
        REAPED_THREAD_METHOD,
        METHOD,
        TEARDOWN_METHOD,
        MEMBERSHIP_RETRY_METHOD,
    )
    if not owner_is_identity:
        if receipt.get("identity_semantics") != IDENTITY_SEMANTICS:
            raise ValueError("missing pidfd-bound process identity semantics")
    elif "identity_semantics" in receipt:
        raise ValueError("v3 identity semantics cannot be relabeled as v1/v2")
    final_polls_required = method in (
        EXIT_OWNER_METHOD,
        PROCESS_IDENTITY_METHOD,
        LIVE_THREAD_METHOD,
        THREAD_EXIT_METHOD,
        THREAD_WAIT_METHOD,
        REAPED_THREAD_METHOD,
        METHOD,
        TEARDOWN_METHOD,
        MEMBERSHIP_RETRY_METHOD,
    )
    if final_polls_required and receipt.get("uid_semantics") != "proc_inode_owner":
        raise ValueError("missing proc inode ownership semantics")
    attempts = receipt.get("attempts")
    if (
        not isinstance(attempts, list)
        or not 1 <= len(attempts) <= MAX_ATTEMPTS
        or receipt.get("final_attempt") != len(attempts) - 1
    ):
        raise ValueError("invalid host collection attempts")
    if receipt.get("max_attempts") != MAX_ATTEMPTS or receipt.get("max_collection_seconds") != _collection_seconds(
        method
    ):
        raise ValueError("unsupported collection bounds")
    recovery_start, absent_unbound = _validate_unbound_recoveries(receipt)
    hard_errors: list[str] = []
    first_stage = None
    first_start = recovery_start
    prior_after: set[str] = set()
    prior_discovery: dict = {}
    observed_rss, observed_pss = [], []
    final_memory: dict = {}
    final_transition = False
    for index, attempt in enumerate(attempts):
        started = attempt.get("begin_monotonic_seconds")
        if type(started) not in (int, float) or not 0 <= started < float("inf"):
            raise ValueError("invalid attempt start")
        if first_start is None:
            first_start = started
        elif not first_start <= started <= first_start + _collection_seconds(method):
            if recovery_start is not None and index == 0 and started > first_start:
                hard_errors.append("unbound_recovery_collection_deadline_exceeded")
            else:
                raise ValueError("retry outside bounded collection interval")
        records = attempt.get("processes")
        before, after = attempt.get("selected_before"), attempt.get("selected_after")
        if not isinstance(records, dict) or not isinstance(before, list) or not isinstance(after, list) or not before:
            raise ValueError("missing selected process receipts")
        if (
            any(type(pid) is not int or pid <= 0 for pid in before + after)
            or len(set(before)) != len(before)
            or len(set(after)) != len(after)
        ):
            raise ValueError("invalid selected process identities")
        if (
            set(records) != {str(pid) for pid in before}
            or not prior_after <= set(records)
            or not set(before) <= set(after)
        ):
            raise ValueError("selected process silently omitted")
        if receipt.get("root_pid") not in before or receipt.get("excluded_monitor_pid") in before:
            raise ValueError("invalid observed tree scope")
        if absent_unbound.intersection(before + after):
            # A legitimately reused PID stays in the fresh evidence, but this
            # ambiguous observation is incomplete rather than a collector crash.
            hard_errors.append("unbound_absent_pid_reappeared")
        for suffix, selected in (("before", before), ("after", after)):
            discovery = attempt.get("discovery_" + suffix)
            if (
                not isinstance(discovery, dict)
                or set(discovery) != {str(pid) for pid in selected}
                or not all(_receipt_valid(read) for read in discovery.values())
            ):
                raise ValueError("missing discovery receipts")
        final_polls = attempt.get("post_discovery_polls", {})
        if final_polls_required:
            if not isinstance(final_polls, dict) or set(final_polls) != set(records):
                raise ValueError("missing post-discovery pidfd receipts")
            for pid, poll in final_polls.items():
                if not _receipt_valid(poll):
                    raise ValueError("invalid post-discovery pidfd receipt")
                preceding = [attempt["discovery_after"][pid], records[pid].get("poll_after", {})]
                if any(
                    not _receipt_valid(read) or read["end_monotonic_seconds"] > poll["begin_monotonic_seconds"]
                    for read in preceding
                ):
                    raise ValueError("post-discovery pidfd read precedes discovery")
                if (
                    not _receipt_valid(attempt.get("stage_after"))
                    or poll["end_monotonic_seconds"] > attempt["stage_after"]["begin_monotonic_seconds"]
                ):
                    raise ValueError("post-discovery pidfd read follows stage boundary")
        elif final_polls or "uid_semantics" in receipt:
            raise ValueError("v2 exit proof cannot be relabeled as v1")
        _validate_thread_exit_waits(attempt, method=method, collection_start=first_start)
        for pid in set(after) - set(before):
            ancestor, visited = pid, set()
            while ancestor not in before:
                value = attempt["discovery_after"].get(str(ancestor), {}).get("value", {})
                if (
                    ancestor in visited
                    or not _valid_identity(value, owner_is_identity=owner_is_identity)
                    or value.get("ppid") not in after
                ):
                    hard_errors.append("unproven_new_descendant")
                    break
                visited.add(ancestor)
                ancestor = value["ppid"]
        stages = [attempt.get(key, {}) for key in ("stage_before", "stage_after")]
        if not all(_receipt_valid(stage) and isinstance(stage.get("value"), str) for stage in stages) or stages[0].get(
            "value"
        ) != stages[1].get("value"):
            hard_errors.append("stage_changed_or_unreadable")
        if first_stage is None:
            first_stage = stages[0].get("value")
        elif stages[0].get("value") != first_stage:
            hard_errors.append("stage_changed_or_unreadable")
        memory, transitioned, process_errors = {}, False, []
        for pid, record in records.items():
            _validate_exit_wait(record, method=method, collection_start=first_start)
            if record.get("pin", {}).get("identity", {}).get("pid") != int(pid):
                hard_errors.append("process_key_identity_mismatch")
            expected = record.get("pin", {}).get("identity", {})
            final_released = _released(final_polls[pid]) if final_polls_required else False
            previous_released = _released(record.get("poll_after", {}))
            if final_polls_required and (
                final_released is None or (previous_released is True and final_released is not True)
            ):
                hard_errors.append("invalid_post_discovery_exit_proof")
            for discovery in (attempt["discovery_before"], attempt["discovery_after"], prior_discovery):
                observed = discovery.get(pid, {}).get("value")
                if (
                    observed is not None
                    and (
                        (not owner_is_identity and not _valid_identity(observed, owner_is_identity=False))
                        or _identity(observed, owner_is_identity=owner_is_identity)
                        != _identity(expected, owner_is_identity=owner_is_identity)
                    )
                    and not (
                        owner_is_identity and _released_owner_transition(observed, expected, final_released is True)
                    )
                ):
                    hard_errors.append("discovery_identity_changed")
            values, errors, changed = classify_process(
                record, released_after_discovery=final_released is True, owner_is_identity=owner_is_identity
            )
            if "thread_mm" in record:
                if method not in (
                    LIVE_THREAD_METHOD,
                    THREAD_EXIT_METHOD,
                    THREAD_WAIT_METHOD,
                    REAPED_THREAD_METHOD,
                    METHOD,
                    TEARDOWN_METHOD,
                    MEMBERSHIP_RETRY_METHOD,
                ):
                    raise ValueError("live-thread evidence cannot be relabeled as an older method")
                try:
                    values = _thread_mm_memory(
                        record,
                        collection_start=first_start,
                        discovery_after=attempt["discovery_after"][pid],
                        collection_seconds=_collection_seconds(method),
                    )
                    errors = []
                except (ValueError, KeyError, TypeError, StopIteration):
                    try:
                        if method not in (
                            THREAD_EXIT_METHOD,
                            THREAD_WAIT_METHOD,
                            REAPED_THREAD_METHOD,
                            METHOD,
                            TEARDOWN_METHOD,
                            MEMBERSHIP_RETRY_METHOD,
                        ):
                            raise ValueError("historical live-thread method retains failed read")
                        values = _thread_mm_memory(
                            record,
                            collection_start=first_start,
                            discovery_after=attempt["discovery_after"][pid],
                            released_poll=final_polls[pid],
                            allow_discovery_enoent=method
                            in (REAPED_THREAD_METHOD, METHOD, TEARDOWN_METHOD, MEMBERSHIP_RETRY_METHOD),
                            collection_seconds=_collection_seconds(method),
                        )
                        errors, changed = [], True
                    except (ValueError, KeyError, TypeError, StopIteration):
                        errors.append("unproven_live_thread_memory")
            memory[pid] = values
            hard_errors.extend(errors)
            process_errors.extend(errors)
            transitioned |= changed
            transitioned |= final_polls_required and previous_released is False and final_released is True
        if not process_errors:
            if all("VmRSS" in value for value in memory.values()):
                observed_rss.append(sum(value["VmRSS"] for value in memory.values()))
            if all("Pss_rollup" in value for value in memory.values()):
                observed_pss.append(sum(value["Pss_rollup"] for value in memory.values()))
        transitioned |= set(before) != set(after)
        if index < len(attempts) - 1 and (hard_errors or not transitioned):
            raise ValueError("retry lacks a proven membership transition")
        prior_after = {str(pid) for pid in after}
        prior_discovery = attempt["discovery_after"]
        final_memory, final_transition = memory, transitioned
    published = receipt.get("stage_publish")
    if published is not None and (not _receipt_valid(published) or published.get("value") != first_stage):
        hard_errors.append("stage_changed_or_unreadable")
    if (
        recovery_start is not None
        and published is not None
        and (
            not _receipt_valid(published)
            or published["end_monotonic_seconds"] > recovery_start + _collection_seconds(method)
        )
    ):
        hard_errors.append("unbound_recovery_collection_deadline_exceeded")
    if (
        method in (TEARDOWN_METHOD, MEMBERSHIP_RETRY_METHOD)
        and published is not None
        and _receipt_valid(published)
        and published["end_monotonic_seconds"] > first_start + _collection_seconds(method)
    ):
        hard_errors.append("collection_deadline_exceeded")
    complete = (
        not hard_errors
        and not final_transition
        and all("Pss_rollup" in value and "VmRSS" in value for value in final_memory.values())
    )
    return {
        "process_host_memory": final_memory,
        "selected_host_rss_sum_bytes": sum(value.get("VmRSS", 0) for value in final_memory.values()),
        "selected_host_pss_sum_bytes": sum(value.get("Pss_rollup", 0) for value in final_memory.values()),
        "host_pss_complete": complete,
        "host_observed_attempt_rss_peak_bytes": max(observed_rss, default=None),
        "host_observed_complete_attempt_pss_peak_bytes": max(observed_pss, default=None),
        "host_collection_errors": sorted(
            set(hard_errors + (["membership_transition_retry_exhausted"] if final_transition else []))
        ),
    }


def _validate_sample(sample: dict) -> dict:
    """Validate each version without trusting producer totals or upgrading old evidence."""
    if sample.get("host_process_memory_method") not in (
        LEGACY_METHOD,
        EXIT_OWNER_METHOD,
        PROCESS_IDENTITY_METHOD,
        LIVE_THREAD_METHOD,
        THREAD_EXIT_METHOD,
        THREAD_WAIT_METHOD,
        REAPED_THREAD_METHOD,
        METHOD,
        TEARDOWN_METHOD,
        MEMBERSHIP_RETRY_METHOD,
    ):
        raise ValueError("unsupported host memory method")
    receipt = sample.get("host_collection", {})
    if receipt.get("protocol") != sample["host_process_memory_method"]:
        raise ValueError("host method differs from collection protocol")
    if not _receipt_valid(receipt.get("stage_publish")):
        raise ValueError("missing final stage read receipt")
    derived = summarize_collection(receipt)
    if any(sample.get(key) != value for key, value in derived.items()):
        raise ValueError("host sample disagrees with retained read attempts")
    stages = [attempt["stage_before"].get("value") for attempt in sample["host_collection"]["attempts"]]
    if sample.get("stage") != stages[0]:
        raise ValueError("host sample stage mismatch")
    return derived


def validate_sample(sample: dict) -> dict:
    """Validate externally read JSON, converting malformed schema to ValueError."""
    try:
        return _validate_sample(sample)
    except (KeyError, TypeError, AttributeError, IndexError, OverflowError) as error:
        raise ValueError("malformed host collection receipt") from error


def finish_sample(sample: dict, stage_file: Path) -> None:
    """Recheck stage after device telemetry without concealing a stage race."""
    sample["host_collection"]["stage_publish"] = _read_stage(stage_file)
    sample.update(summarize_collection(sample["host_collection"]))
    validate_sample(sample)


class HostSampler:
    """Observe descendants; retain identity-bound orphaned members until exit.

    Discovery is a /proc snapshot, not a proof that no short-lived process fell
    between samples. RSS/status and PSS/rollup reads are not simultaneous.
    V9 allows 3.5 seconds for an owned process exit within a four-second
    collection. This bounds requested waits, not OS scheduling or /proc I/O; an
    overrun stays incomplete. V10 retains the same 3.5-second wait and allows
    4.5 seconds for acquisition, leaving time for a fresh membership/read pass
    after a slow exit-time rollup. The stage-lock deadline remains five seconds.
    V1–V8 retain their original 350 ms / 750 ms limits; v9 stays four seconds.
    A longer acquisition holds the coordinated stage lock and may delay the
    next stage; both timing arms must include that cost. A successful retry
    observes the later state, never reconstructs a peak in the unreadable interval.
    """

    def __init__(self, root_pid: int, stage_file: Path, *, exclude_pid: int | None = None, method: str = METHOD):
        if method not in (
            PROCESS_IDENTITY_METHOD,
            LIVE_THREAD_METHOD,
            THREAD_EXIT_METHOD,
            THREAD_WAIT_METHOD,
            REAPED_THREAD_METHOD,
            METHOD,
            TEARDOWN_METHOD,
            MEMBERSHIP_RETRY_METHOD,
        ):
            raise ValueError("unsupported host acquisition method")
        self.method = method
        self.root_pid, self.stage_file = root_pid, stage_file
        self.exclude_pid = os.getpid() if exclude_pid is None else exclude_pid
        self.backend = _PidfdBackend()
        self.tracked: dict[int, dict] = {}

    def close(self):
        for item in self.tracked.values():
            if item["fd"] is not None:
                os.close(item["fd"])
        self.tracked.clear()

    def _discover(self) -> set[int]:
        parents, reads = {}, {}
        for path in Path("/proc").iterdir():
            if path.name.isdigit():
                read = read_proc(int(path.name), "stat")
                reads[str(int(path.name))] = read
                if "value" in read:
                    parents[int(path.name)] = read["value"]["ppid"]
        selected = set(self.tracked) | {self.root_pid}
        while True:
            more = {pid for pid, parent in parents.items() if parent in selected and pid != self.exclude_pid}
            if more <= selected:
                selected -= {self.exclude_pid}
                self.discovery = {str(pid): reads.get(str(pid)) or read_proc(pid, "stat") for pid in selected}
                return selected
            selected |= more

    def _pin(self, pid: int, discovery: dict):
        before = read_proc(pid, "stat")
        pin = {"discovery": discovery, "before": before, "flags": 0, "backend": self.backend.backend}
        fd = None
        try:
            if (
                "value" not in before
                or "value" not in discovery
                or not _valid_identity(before["value"], owner_is_identity=False)
                or not _valid_identity(discovery["value"], owner_is_identity=False)
                or _identity(before["value"], owner_is_identity=False)
                != _identity(discovery["value"], owner_is_identity=False)
            ):
                raise ProcessLookupError(errno.ESRCH, "identity unavailable before pin")
            fd = self.backend.open(pid)
            after = read_proc(pid, "stat")
            pin["after"] = after
            if (
                "value" not in after
                or not _valid_identity(after["value"], owner_is_identity=False)
                or _identity(before["value"], owner_is_identity=False)
                != _identity(after["value"], owner_is_identity=False)
            ):
                raise ProcessLookupError(errno.ESRCH, "identity changed while pinning")
            pin["identity"] = before["value"]
        except OSError as error:
            pin["error"] = _error(error)
            if fd is not None:
                os.close(fd)
                fd = None
        self.tracked[pid] = {"pin": pin, "fd": fd}

    def _process(self, pid: int) -> dict:
        item = self.tracked[pid]
        record = {"pin": item["pin"], "stat_before": read_proc(pid, "stat"), "poll_before": _poll(item["fd"])}
        record["status"] = read_proc(pid, "status")
        record["rollup"] = read_proc(pid, "smaps_rollup")
        record["stat_after"] = read_proc(pid, "stat")
        record["poll_after"] = _poll(item["fd"])
        return record

    def _recover_unbound(self, started: float) -> dict:
        """Revisit only failed bindings already published in an earlier sample.

        Successful pins, including reparented live descendants, are untouched.
        A failed interval is never retried into a complete interval here: these
        proofs precede an entirely new sample and contain no memory estimate.
        """
        proofs = {}
        for pid, item in list(self.tracked.items()):
            if (
                pid == self.root_pid
                or item["fd"] is not None
                or not _unbound_pin(item["pin"])
                or "failed_sample_end" not in item
                or time.monotonic() + EXIT_WAIT_RETRY_RESERVE_SECONDS >= started + _collection_seconds(self.method)
            ):
                continue
            proof = {
                "failed_pin": item["pin"],
                "failed_collection_end_monotonic_seconds": item["failed_sample_end"],
                "namespace_before": _read_pid_namespace(),
                "stat_before": read_proc(pid, "stat"),
            }
            opened = {
                "pid": pid,
                "flags": 0,
                "backend": self.backend.backend,
                "begin_unix_seconds": time.time(),
                "begin_monotonic_seconds": time.monotonic(),
            }
            fd = None
            try:
                fd = self.backend.open(pid)
                opened["value"] = {"opened": True}
            except OSError as error:
                opened["error"] = _error(error)
            opened.update(end_unix_seconds=time.time(), end_monotonic_seconds=time.monotonic())
            proof.update(pidfd_open=opened, stat_after=read_proc(pid, "stat"), namespace_after=_read_pid_namespace())
            result = _unbound_recovery_result(proof)
            proof["decision_monotonic_seconds"] = time.monotonic()
            if proof["decision_monotonic_seconds"] + EXIT_WAIT_RETRY_RESERVE_SECONDS > started + _collection_seconds(
                self.method
            ):
                result = "unresolved"
            proof["result"] = result
            proofs[str(pid)] = proof
            if result == "absent":
                del self.tracked[pid]
            elif result == "rebound":
                pin = {
                    "discovery": item["pin"]["discovery"],
                    "before": proof["stat_before"],
                    "after": proof["stat_after"],
                    "identity": proof["stat_before"]["value"],
                    "flags": 0,
                    "backend": self.backend.backend,
                }
                self.tracked[pid] = {"pin": pin, "fd": fd}
                fd = None
            if fd is not None:
                os.close(fd)
        return proofs

    def _wait_for_exit(self, pid: int, record: dict, deadline: float) -> None:
        if not _exit_wait_eligible(record):
            return
        # Reserve time for a new complete collection. Scheduler delay may still
        # exceed a requested poll timeout; only an in-budget retry is accepted.
        timeout = _exit_wait_timeout(deadline, method=self.method)
        if timeout <= 0:
            return
        initial = record["poll_after"]
        record["exit_wait"] = {
            "initial_poll": initial,
            "deadline_monotonic_seconds": deadline,
            "retry_reserve_seconds": EXIT_WAIT_RETRY_RESERVE_SECONDS,
        }
        record["poll_after"] = _poll(self.tracked[pid]["fd"], timeout_milliseconds=timeout)

    def collect(self) -> dict:
        started = time.monotonic()
        receipt = {
            "protocol": self.method,
            "uid_semantics": "proc_inode_owner",
            "identity_semantics": IDENTITY_SEMANTICS,
            "root_pid": self.root_pid,
            "excluded_monitor_pid": self.exclude_pid,
            "attempts": [],
            "max_attempts": MAX_ATTEMPTS,
            "max_collection_seconds": _collection_seconds(self.method),
        }
        if self.method in (METHOD, TEARDOWN_METHOD, MEMBERSHIP_RETRY_METHOD):
            recoveries = self._recover_unbound(started)
            if recoveries:
                receipt.update(unbound_recoveries=recoveries, collection_begin_monotonic_seconds=started)
        for _ in range(MAX_ATTEMPTS):
            attempt_started = time.monotonic()
            if not receipt["attempts"] and "unbound_recoveries" not in receipt:
                started = attempt_started
            stage = _read_stage(self.stage_file)
            selected = self._discover()
            discovery_before = self.discovery
            for pid in selected - self.tracked.keys():
                self._pin(pid, discovery_before[str(pid)])
            records = {}
            for pid in sorted(selected):
                record = self._process(pid)
                if self.method in (
                    LIVE_THREAD_METHOD,
                    THREAD_EXIT_METHOD,
                    THREAD_WAIT_METHOD,
                    REAPED_THREAD_METHOD,
                    METHOD,
                    TEARDOWN_METHOD,
                    MEMBERSHIP_RETRY_METHOD,
                ) and _thread_mm_eligible(record):
                    record["thread_mm"] = _thread_mm(
                        pid, self.tracked[pid]["fd"], started + _collection_seconds(self.method)
                    )
                elif len(receipt["attempts"]) < MAX_ATTEMPTS - 1:
                    self._wait_for_exit(pid, record, started + _collection_seconds(self.method))
                records[str(pid)] = record
            after = self._discover()
            discovery_after = self.discovery
            for pid in after - self.tracked.keys():
                self._pin(pid, discovery_after[str(pid)])
            final_polls, thread_waits = {}, {}
            for pid in sorted(selected):
                initial = _poll(self.tracked[pid]["fd"])
                final_polls[str(pid)] = initial
                if (
                    self.method
                    in (THREAD_WAIT_METHOD, REAPED_THREAD_METHOD, METHOD, TEARDOWN_METHOD, MEMBERSHIP_RETRY_METHOD)
                    and len(receipt["attempts"]) < MAX_ATTEMPTS - 1
                    and _released(initial) is False
                    and _thread_exit_wait_eligible(
                        records[str(pid)],
                        collection_start=started,
                        discovery_after=discovery_after[str(pid)],
                        collection_seconds=_collection_seconds(self.method),
                    )
                ):
                    timeout = _exit_wait_timeout(started + _collection_seconds(self.method), method=self.method)
                    if timeout > 0:
                        thread_waits[str(pid)] = {
                            "initial_poll": initial,
                            "deadline_monotonic_seconds": started + _collection_seconds(self.method),
                            "retry_reserve_seconds": EXIT_WAIT_RETRY_RESERVE_SECONDS,
                        }
                        final_polls[str(pid)] = _poll(self.tracked[pid]["fd"], timeout_milliseconds=timeout)
            attempt = {
                "begin_monotonic_seconds": attempt_started,
                "discovery_before": discovery_before,
                "discovery_after": discovery_after,
                "stage_before": stage,
                "stage_after": _read_stage(self.stage_file),
                "selected_before": sorted(selected),
                "selected_after": sorted(after),
                "processes": records,
                "post_discovery_polls": final_polls,
            }
            if thread_waits:
                attempt["thread_exit_waits"] = thread_waits
            receipt["attempts"].append(attempt)
            receipt["final_attempt"] = len(receipt["attempts"]) - 1
            derived = summarize_collection(receipt)
            if (
                derived["host_pss_complete"]
                or derived["host_collection_errors"] != ["membership_transition_retry_exhausted"]
                or time.monotonic() - started >= _collection_seconds(self.method)
            ):
                break
        receipt["stage_publish"] = _read_stage(self.stage_file)
        derived = summarize_collection(receipt)
        record = {
            "host_process_memory_method": self.method,
            "host_collection": receipt,
            "stage": receipt["attempts"][0]["stage_before"].get("value", "unreadable"),
        }
        record.update(derived)
        if self.method in (METHOD, TEARDOWN_METHOD, MEMBERSHIP_RETRY_METHOD):
            for item in self.tracked.values():
                if item["fd"] is None:
                    item.setdefault("failed_sample_end", time.monotonic())
        # Keep a released root as scope evidence; retire other members only after
        # an explicit released record, never merely on a failed /proc read.
        for pid in list(self.tracked):
            if pid != self.root_pid and derived["process_host_memory"].get(str(pid), {}).get("address_space_released"):
                fd = self.tracked.pop(pid)["fd"]
                if fd is not None:
                    os.close(fd)
        return record
