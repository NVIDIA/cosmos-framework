# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""Trace accounting never adds annotations to their GPU kernel durations."""

import pytest

from tools.native_qlora_qualification.profile_trace import summarize, union, worker_spans


def event(name, start, duration, category="user_annotation", **args):
    return {"ph": "X", "name": name, "cat": category, "pid": 1, "tid": 1, "ts": start, "dur": duration, "args": args}


def test_nested_annotations_and_overlapping_device_streams_not_double_counted():
    events = [
        event("trainer.backward", 0, 100),
        event("sft.module.adapter_a.backward_recompute", 10, 40),
        event("aten::mm", 15, 10, "cpu_op", **{"External id": 7}),
        event("gemm", 1000, 30, "kernel", **{"External id": 7, "device": 0, "stream": 1}),
        event("kDequantizeBlockwise", 1010, 40, "kernel", device=0, stream=2),
        event("Memcpy HtoD", 1040, 20, "gpu_memcpy", device=0, stream=3),
    ]
    result = summarize({"traceEvents": events})
    assert result["cpu_phases"]["trainer.backward"]["inclusive_wall_us"] == 100
    assert result["cpu_phases"]["trainer.backward"]["exclusive_named_phase_wall_us"] == 60
    assert result["device_observation_union_us"] == {"0": 60}
    assert sum(r["summed_duration_us"] for r in result["device_operations"].values()) == 90
    assert result["device_operations"]["sft.module.adapter_a.backward_recompute"]["count"] == 1
    assert result["device_operations"]["dequantization"]["summed_duration_us"] == 40
    assert not result["end_to_end_speed_evidence"]


def test_native_family_head_preparation_and_workflow_ranges_keep_separate_accounting():
    records = [
        event("family.native_update.21", 0, 100),
        event("family.packer.forward", 0, 10),
        event("family.processor.forward", 1, 6),
        event("family.module.head.forward", 20, 50),
        event("family.module.head_model.forward", 25, 30),
        event("aten::mm", 30, 5, "cpu_op", **{"External id": 7}),
        event("gemm", 200, 20, "kernel", **{"External id": 7, "device": 0}),
        event("family.evaluation", 300, 200),
        event("family.evaluation.callback", 320, 100),
        event("family.checkpoint", 600, 50),
    ]
    result = summarize({"traceEvents": records})
    phases = result["cpu_phases"]
    assert phases["family.native_update.21"]["exclusive_named_phase_wall_us"] == 40
    assert phases["family.packer.forward"]["exclusive_named_phase_wall_us"] == 4
    assert phases["family.module.head.forward"]["exclusive_named_phase_wall_us"] == 20
    assert phases["family.evaluation"]["exclusive_named_phase_wall_us"] == 100
    assert phases["family.checkpoint"]["inclusive_wall_us"] == 50
    assert result["device_observation_union_us"] == {"0": 20}
    assert result["device_operations"]["family.module.head_model.forward"]["summed_duration_us"] == 20
    [group] = result["kernel_attribution_diagnostics"]["groups"]
    assert group["same_thread_named_range"] == "family.module.head_model.forward"
    assert not result["end_to_end_speed_evidence"]


def test_family_cpu_spans_are_included_without_conflating_separate_workers():
    rows = [
        {
            "protocol": "cosmos.sft-cpu-span.v1",
            "name": "family.processor.forward",
            "pid": pid,
            "thread": 1,
            "begin_monotonic_ns": 0,
            "end_monotonic_ns": stop,
            "inclusive_wall_ns": stop,
        }
        for pid, stop in ((1, 100000), (2, 80000))
    ]
    result = worker_spans(rows)["family.processor.forward"]
    assert result == {"calls": 2, "inclusive_wall_us": 180, "exclusive_named_phase_wall_us": 180}


def test_ambiguous_external_id_retains_unattributed_kernel():
    result = summarize(
        {
            "traceEvents": [
                event("trainer.optimizer", 0, 10),
                event("a", 1, 2, "cpu_op", **{"External id": 1}),
                event("b", 4, 2, "cpu_op", **{"External id": 1}),
                event("gemm", 12, 2, "kernel", **{"External id": 1}),
            ]
        }
    )
    assert result["unattributed_kernel_count"] == 1
    assert "trainer.optimizer" not in result["device_operations"]


def test_peer_child_union_and_different_cpu_threads():
    a = event("sft.parent", 0, 100)
    b = event("sft.child", 10, 30)
    c = event("sft.peer", 20, 30)
    other = event("sft.other_thread", 0, 90)
    other["tid"] = 2
    report = summarize({"traceEvents": [a, b, c, other]})
    assert report["cpu_phases"]["sft.parent"]["exclusive_named_phase_wall_us"] == 60
    assert union([(2, 3), (0, 5), (4, 8)]) == 8


@pytest.mark.parametrize("duration", [float("nan"), float("inf"), -1, True, None])
def test_invalid_complete_event_duration_rejected(duration):
    with pytest.raises(ValueError):
        summarize({"traceEvents": [event("sft.invalid", 0, duration)]})


def test_worker_spans_preserve_nested_decode_and_process_identity():
    def row(name, pid, start, stop):
        return {
            "protocol": "cosmos.sft-cpu-span.v1",
            "name": name,
            "pid": pid,
            "thread": 1,
            "begin_monotonic_ns": start,
            "end_monotonic_ns": stop,
            "inclusive_wall_ns": stop - start,
        }

    result = worker_spans(
        [
            row("sft.processor", 1, 0, 100000),
            row("sft.video_decode_and_sample", 1, 10000, 50000),
            row("sft.prefetch_wait", 2, 0, 90000),
        ]
    )
    assert result["sft.processor"]["exclusive_named_phase_wall_us"] == 60
    assert result["sft.prefetch_wait"]["exclusive_named_phase_wall_us"] == 90
    bad = row("sft.processor", 1, 0, 10)
    bad["inclusive_wall_ns"] = 9
    with pytest.raises(ValueError, match="duration"):
        worker_spans([bad])


def on_lane(record, pid=1, tid=2):
    return {**record, "pid": pid, "tid": tid}


def test_autograd_worker_evidence_keeps_literal_forward_label_and_v1_unknown_bucket():
    engine = "autograd::engine::evaluate_function: MmBackward0"
    module = "sft.module.adapter_a.forward"
    records = [
        event("trainer.backward", 0, 100),
        on_lane(event(engine, 5, 80, "cpu_op", **{"External id": 10})),
        on_lane(event(module, 10, 20)),
        on_lane(event("aten::mm", 12, 4, "cpu_op", **{"External id": 11})),
        on_lane(event("aten::fill_", 40, 4, "cpu_op", **{"External id": 12})),
        event("gemm", 200, 8, "kernel", **{"External id": 11, "device": 0}),
        event("fill", 210, 3, "kernel", **{"External id": 12, "device": 0}),
    ]
    result = summarize({"traceEvents": records})
    assert result["device_operations"] == {
        module: {"count": 1, "summed_duration_us": 8, "wall_union_us": 8},
        "unattributed_kernel": {"count": 1, "summed_duration_us": 3, "wall_union_us": 3},
    }
    assert result["unattributed_kernel_count"] == 1
    assert result["device_observation_union_us"] == {"0": 11}
    groups = {g["cpu_op"]: g for g in result["kernel_attribution_diagnostics"]["groups"]}
    assert groups["aten::mm"]["same_thread_named_range"] == module
    assert groups["aten::mm"]["same_thread_autograd_ancestor"] == engine
    assert groups["aten::fill_"]["same_thread_named_range"] is None
    assert groups["aten::fill_"]["same_thread_autograd_ancestor"] == engine
    assert "trainer.backward" not in result["device_operations"]  # No inheritance across CPU threads.


@pytest.mark.parametrize(
    ("cpu_records", "kernel_args", "status"),
    [
        ([], {}, "missing_kernel_external_id"),
        ([], {"External id": 99}, "unmatched_external_id"),
        (
            [
                event("aten::mm", 1, 2, "cpu_op", **{"External id": 7}),
                on_lane(event("aten::mm", 1, 2, "cpu_op", **{"External id": 7}), pid=2),
            ],
            {"External id": 7},
            "ambiguous_external_id",
        ),
    ],
)
def test_supplementary_correlation_does_not_invent_missing_or_ambiguous_links(cpu_records, kernel_args, status):
    result = summarize({"traceEvents": [*cpu_records, event("gemm", 20, 2, "kernel", **kernel_args)]})
    [group] = result["kernel_attribution_diagnostics"]["groups"]
    assert group["original_device_label"] == "unattributed_kernel"
    assert group["cpu_correlation"] == status
    assert group["cpu_op"] is group["same_thread_named_range"] is group["same_thread_autograd_ancestor"] is None


@pytest.mark.parametrize("other_lane", [{"tid": 2}, {"pid": 2}])
def test_overlapping_backward_range_on_another_lane_is_not_ancestry(other_lane):
    ancestor = {**event("autograd::engine::evaluate_function: MmBackward0", 0, 100, "cpu_op"), **other_lane}
    result = summarize(
        {
            "traceEvents": [
                ancestor,
                event("aten::mm", 20, 5, "cpu_op", **{"External id": 7}),
                event("gemm", 200, 3, "kernel", **{"External id": 7}),
            ]
        }
    )
    [group] = result["kernel_attribution_diagnostics"]["groups"]
    assert group["cpu_op"] == "aten::mm"
    assert group["same_thread_autograd_ancestor"] is None


@pytest.mark.parametrize("parents", [[(0, 20), (10, 30)], [(0, 40), (0, 40)], [(10, 3)]])
def test_crossing_tied_or_partially_containing_engine_records_remain_unresolved(parents):
    records = [
        event(f"autograd::engine::evaluate_function: Parent{i}", a, b, "cpu_op") for i, (a, b) in enumerate(parents)
    ]
    records += [
        event("aten::mm", 12, 4, "cpu_op", **{"External id": 7}),
        event("gemm", 200, 3, "kernel", **{"External id": 7}),
    ]
    [group] = summarize({"traceEvents": records})["kernel_attribution_diagnostics"]["groups"]
    assert group["same_thread_autograd_ancestor"] is None


def test_nested_engine_evidence_and_overlapping_kernels_keep_existing_totals():
    outer = "autograd::engine::evaluate_function: OuterBackward0"
    inner = "autograd::engine::evaluate_function: MatMul4BitBackward"
    result = summarize(
        {
            "traceEvents": [
                event(outer, 0, 100, "cpu_op"),
                event(inner, 10, 40, "cpu_op"),
                event("aten::mm", 15, 5, "cpu_op", **{"External id": 7}),
                event("gemm", 200, 10, "kernel", **{"External id": 7, "device": 0, "stream": 1}),
                event("gemm", 205, 10, "kernel", **{"External id": 7, "device": 0, "stream": 2}),
                event("kDequantizeBlockwise", 209, 5, "kernel", device=0),
                event("Memcpy", 214, 5, "gpu_memcpy", device=0),
            ]
        }
    )
    assert result["device_observation_union_us"] == {"0": 19}
    assert result["unattributed_kernel_count"] == 2
    assert result["device_operations"]["unattributed_kernel"] == {
        "count": 2,
        "summed_duration_us": 20,
        "wall_union_us": 15,
    }
    groups = result["kernel_attribution_diagnostics"]["groups"]
    assert sum(g["kernel_count"] for g in groups) == 3  # Copies are outside the kernel-only supplement.
    [linked] = [g for g in groups if g["cpu_correlation"] == "unique_external_id"]
    assert linked["same_thread_autograd_ancestor"] == inner
    assert linked["kernel_count"] == 2 and linked["wall_union_us"] == 15
    [unlinked] = [g for g in groups if g["cpu_correlation"] == "missing_kernel_external_id"]
    assert unlinked["original_device_label"] == "dequantization"
