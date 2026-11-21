# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""CPU-only diagnostic trace accounting; never a throughput/quality selector.

Chrome CPU annotation durations overlap their operators and GPU kernels. Keep
inclusive annotation spans separate from non-overlapping interval unions and
from summed GPU operation durations (which can overlap on different streams).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from bisect import bisect_left, bisect_right
from collections import defaultdict
from pathlib import Path

PHASE_PREFIXES = ("sft.", "trainer.", "family.")
MODULE_PREFIXES = ("sft.module.", "family.module.", "trainer.optimizer", "trainer.backward")


def union(intervals: list[tuple[float, float]]) -> float:
    total, end = 0.0, -math.inf
    for start, stop in sorted(intervals):
        total += max(0.0, stop - max(start, end))
        end = max(end, stop)
    return total


def interval(event: dict) -> tuple[float, float]:
    values = [event.get("ts"), event.get("dur")]
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in values):
        raise ValueError("trace duration/timestamp must be finite")
    if values[1] < 0:
        raise ValueError("negative trace duration")
    return values[0], values[0] + values[1]


def phase_totals(events: list[dict]) -> dict:
    """Exclusive means subtracting nested *named phase* wall spans on one thread.

    It is not CPU execution time: scheduling and waits remain within these spans.
    Overlapping peer spans are unioned and never subtracted twice.
    """
    phases = [e for e in events if e.get("cat") == "user_annotation" and e.get("name", "").startswith(PHASE_PREFIXES)]
    groups = defaultdict(list)
    for event in phases:
        groups[event["name"]].append(event)
    threads = defaultdict(list)
    for event in phases:
        threads[(event.get("pid"), event.get("tid"))].append(event)
    starts = {}
    for key, entries in threads.items():
        entries.sort(key=lambda e: e["ts"])
        starts[key] = [e["ts"] for e in entries]
    result = {}
    for name, group in sorted(groups.items()):
        inclusive = exclusive = 0.0
        for event in group:
            start, stop = interval(event)
            inclusive += stop - start
            nested = []
            key = (event.get("pid"), event.get("tid"))
            lo, hi = bisect_left(starts[key], start), bisect_right(starts[key], stop)
            for child in threads[key][lo:hi]:
                a, b = interval(child)
                if (
                    child is not event
                    and child.get("pid") == event.get("pid")
                    and child.get("tid") == event.get("tid")
                    and start <= a <= b <= stop
                    and (a, b) != (start, stop)
                ):
                    nested.append((a, b))
            exclusive += stop - start - union(nested)
        result[name] = {"calls": len(group), "inclusive_wall_us": inclusive, "exclusive_named_phase_wall_us": exclusive}
    return result


def _innermost_recorded_name(active: list[dict], event: dict) -> str | None:
    """Require one innermost, fully containing record on the caller's CPU lane."""
    start, stop = interval(event)
    parents = [e for e in active if interval(e)[0] <= start <= stop <= interval(e)[1]]
    if not parents:
        return None
    shortest = min(e["dur"] for e in parents)
    nearest = [e for e in parents if e["dur"] == shortest]
    if len(nearest) != 1:
        return None
    a, b = interval(nearest[0])
    if any(not (interval(e)[0] <= a <= b <= interval(e)[1]) for e in parents):
        return None  # Crossing peer records do not establish nested ancestry.
    return nearest[0]["name"]


def summarize(trace: dict) -> dict:
    raw = trace.get("traceEvents")
    if not isinstance(raw, list):
        raise ValueError("Chrome traceEvents list required")
    events = [e for e in raw if e.get("ph") == "X"]
    for event in events:
        interval(event)
    annotations = [e for e in events if e.get("cat") == "user_annotation"]
    # External IDs may be reused across processes. Never silently select one.
    external = defaultdict(list)
    for event in events:
        if event.get("cat") == "cpu_op" and "External id" in event.get("args", {}):
            external[str(event["args"]["External id"])].append(event)
    # Resolve enclosing named module scopes in a per-thread sweep. This avoids
    # an O(kernel_count * annotation_count) scan on full native traces.
    lanes = defaultdict(list)
    for event in annotations + [e for e in events if e.get("cat") == "cpu_op"]:
        lanes[(event.get("pid"), event.get("tid"))].append(event)
    enclosing_by_event = {}
    recorded_context = {}
    for lane in lanes.values():
        active = []
        named, autograd = [], []
        for event in sorted(lane, key=lambda e: (e["ts"], -e["dur"])):
            start, stop = interval(event)
            active = [e for e in active if interval(e)[1] >= start]
            named = [e for e in named if interval(e)[1] >= start]
            autograd = [e for e in autograd if interval(e)[1] >= start]
            if event.get("cat") == "user_annotation":
                if event.get("name", "").startswith(MODULE_PREFIXES):
                    active.append(event)
                if event.get("name", "").startswith(PHASE_PREFIXES):
                    named.append(event)
            else:
                parents = [e for e in active if interval(e)[1] >= stop]
                if parents:
                    enclosing_by_event[id(event)] = min(parents, key=lambda e: e["dur"])["name"]
                recorded_context[id(event)] = (
                    _innermost_recorded_name(named, event),
                    _innermost_recorded_name(autograd, event),
                )
                if event.get("name", "").startswith("autograd::engine::evaluate_function:"):
                    autograd.append(event)
    device, classes = defaultdict(list), defaultdict(list)
    kernel_contexts = defaultdict(list)
    unresolved = 0
    for event in events:
        category = event.get("cat", "")
        if category not in {"kernel", "gpu_memcpy", "gpu_memset"}:
            continue
        name, args = event.get("name", ""), event.get("args", {})
        key = str(args.get("device", event.get("pid", "unknown")))
        device[key].append(interval(event))
        if category == "gpu_memcpy":
            label = "transfers"
        elif category == "gpu_memset":
            label = "device_memset"
        elif "dequantize" in name.lower():
            label = "dequantization"
        else:
            candidates = external.get(str(args.get("External id")), [])
            enclosing = enclosing_by_event.get(id(candidates[0])) if len(candidates) == 1 else None
            if enclosing:
                label = enclosing
            else:
                label = "unattributed_kernel"
                unresolved += 1
        classes[label].append(interval(event))
        if category == "kernel":
            candidates = external.get(str(args.get("External id")), [])
            cpu_op = named_range = autograd_ancestor = None
            if "External id" not in args:
                correlation = "missing_kernel_external_id"
            elif not candidates:
                correlation = "unmatched_external_id"
            elif len(candidates) != 1:
                correlation = "ambiguous_external_id"
            else:
                correlation = "unique_external_id"
                cpu_op = candidates[0]["name"]
                named_range, autograd_ancestor = recorded_context[id(candidates[0])]
            kernel_contexts[(label, correlation, cpu_op, named_range, autograd_ancestor)].append(interval(event))
    return {
        "protocol": "cosmos.native-qlora-trace-accounting.v1",
        "diagnostic_only": True,
        "end_to_end_speed_evidence": False,
        "cpu_phases": phase_totals(events),
        "device_observation_union_us": {k: union(v) for k, v in sorted(device.items())},
        "device_operations": {
            k: {"count": len(v), "summed_duration_us": sum(b - a for a, b in v), "wall_union_us": union(v)}
            for k, v in sorted(classes.items())
        },
        "unattributed_kernel_count": unresolved,
        "kernel_attribution_diagnostics": {
            "protocol": "cosmos.native-qlora-kernel-attribution-diagnostics.v1",
            "groups": [
                {
                    "original_device_label": key[0],
                    "cpu_correlation": key[1],
                    "cpu_op": key[2],
                    "same_thread_named_range": key[3],
                    "same_thread_autograd_ancestor": key[4],
                    "kernel_count": len(values),
                    "summed_duration_us": sum(b - a for a, b in values),
                    "wall_union_us": union(values),
                }
                for key, values in sorted(kernel_contexts.items(), key=lambda item: tuple(x or "" for x in item[0]))
            ],
            "scope": "An alternative grouping of the same kernel events, not additional device or CPU wall time.",
            "limitations": [
                "Original v1 labels, buckets and totals are unchanged; no missing module identity is inferred.",
                "CPU correlation requires one unique recorded External id. Ambiguous or missing links stay unresolved.",
                "Named ranges and autograd ancestry require unambiguous full containment on the same CPU process/thread.",
                "Python ContextVar phase labels need not propagate to autograd workers: a recorded .forward module range "
                "can occur inside an explicit autograd engine event. Its literal label is retained, not a pure-forward claim.",
                "Absent ancestry means no such enclosing record, not proof of forward execution. No cross-thread "
                "ancestry or critical-path causality is inferred from overlapping timestamps.",
                "Group unions may overlap across streams; do not add this breakdown to the original device totals.",
            ],
        },
        "limitations": [
            "CPU ranges are wall time, including waits; inclusive parent/child values must not be added.",
            "Native family preparation, evaluation and checkpoint annotations are included when recorded; an update-only trace does not establish unrecorded workflow costs.",
            "Device class unions can overlap each other; summed operation durations can exceed elapsed time across streams.",
            "GPU attribution requires one unique CPU external ID and its enclosing module range; unmatched kernels remain unattributed.",
            "Traced measurements cannot establish end-to-end speed or quality; retain cold and untraced complete runs separately.",
        ],
    }


def worker_spans(rows: list[dict]) -> dict:
    events = []
    for row in rows:
        if row.get("protocol") != "cosmos.sft-cpu-span.v1" or row.get("inclusive_wall_ns") != row.get(
            "end_monotonic_ns", 0
        ) - row.get("begin_monotonic_ns", 0):
            raise ValueError("CPU span protocol/duration differs")
        events.append(
            {
                "ph": "X",
                "cat": "user_annotation",
                "pid": row["pid"],
                "tid": row["thread"],
                "name": row["name"],
                "ts": row["begin_monotonic_ns"] / 1000,
                "dur": row["inclusive_wall_ns"] / 1000,
            }
        )
    for event in events:
        interval(event)
    return phase_totals(events)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--cpu-spans", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.trace.read_bytes()
    result = summarize(json.loads(gzip.decompress(raw) if args.trace.suffix == ".gz" else raw))
    files = {str(args.trace): hashlib.sha256(raw).hexdigest()}
    spans = []
    for path in args.cpu_spans:
        content = path.read_bytes()
        files[str(path)] = hashlib.sha256(content).hexdigest()
        spans.extend(json.loads(line) for line in content.splitlines())
    result["worker_and_parent_cpu_spans"] = worker_spans(spans)
    result["source_files_sha256"] = files
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
