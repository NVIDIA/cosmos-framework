# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""The pure decision functions of the weight-sync bench drivers, without a cluster or an engine.

`tests/gr00t_weight_sync_cli.py` and `tests/vllm_weight_sync_cli.py` decide "significant delta", "digests agree" and
"witness landed" from dictionaries the mesh returns; those decisions live in `tests/helpers/weight_sync_bench.py` and
are pinned here with a positive and a negative case each, so a bench that could not fail is caught before it runs on
GPUs.
"""

from __future__ import annotations

import pytest
import torch

from cosmos_framework.posttrain.tests.helpers import weight_sync_bench as bench

_DIGEST = {"shape": [2], "dtype": "torch.bfloat16", "sha256": "ab" * 32}


@pytest.mark.L0
@pytest.mark.CPU
def test_arm_order_alternates_the_starting_arm() -> None:
    assert bench.arm_order(3) == ["on", "off", "off", "on", "on", "off"]
    assert bench.arm_order(0) == []
    order = bench.arm_order(10)
    assert order.count("on") == order.count("off") == 10
    assert order[::2].count("on") == 5, "each arm goes first in half of the rounds"


@pytest.mark.L0
@pytest.mark.CPU
def test_mann_whitney_u_is_exact_for_small_samples() -> None:
    total = sum(bench._u_count(10, 10, k) for k in range(101))
    assert total == 184756, "C(20, 10) arrangements of the ranks"
    # The tabulated two-sided 5 % critical value for n1 = n2 = 10 is U <= 23.
    below_23 = sum(bench._u_count(10, 10, k) for k in range(24))
    assert 2 * below_23 / total <= 0.05 < 2 * (below_23 + bench._u_count(10, 10, 24)) / total
    u, p = bench.mann_whitney_u(list(range(10, 20)), list(range(10)))
    assert u == 0.0 and p == pytest.approx(2 / total)
    _, p_interleaved = bench.mann_whitney_u(list(range(0, 20, 2)), list(range(1, 20, 2)))
    assert p_interleaved > 0.5
    with pytest.raises(ValueError, match="non-empty"):
        bench.mann_whitney_u([], [1.0])


@pytest.mark.L0
@pytest.mark.CPU
def test_summarize_bench_flags_disjoint_ranges_and_the_u_test() -> None:
    on = [4.5, 4.6, 4.55, 4.7, 4.5, 4.62, 4.58, 4.49, 4.66, 4.53]
    off = [x - 4.0 for x in on]
    summary = bench.summarize_bench({"on": on, "off": off})
    assert (summary["on"]["n"], summary["on"]["min"], summary["on"]["max"]) == (10, 4.49, 4.7)
    assert summary["on"]["median"] == pytest.approx(4.565) and summary["on"]["samples"] == on
    assert summary["delta_s"] == pytest.approx(4.0)
    assert summary["ranges_disjoint"] and summary["significant"] and summary["mann_whitney_p"] < 0.05
    assert summary["delta_over_max_range"] == pytest.approx(4.0 / (4.7 - 4.49))
    # Controlled perturbation: one "off" sample lifted into the "on" range must flip the range verdict.
    lifted = list(off)
    lifted[3] = 4.6
    perturbed = bench.summarize_bench({"on": on, "off": lifted})
    assert not perturbed["ranges_disjoint"] and not perturbed["significant"]
    # Overlapping arms with a small shift: neither the range criterion nor the U test call it.
    shifted = bench.summarize_bench({"on": [x + 0.02 for x in off], "off": off})
    assert not shifted["significant"] and shifted["mann_whitney_p"] > 0.05
    # Disjoint but under twice the spread: not significant by the primary criterion.
    narrow = bench.summarize_bench({"on": [1.0, 1.4], "off": [0.5, 0.9]})
    assert narrow["ranges_disjoint"] and not narrow["significant"]
    with pytest.raises(ValueError, match="both arms"):
        bench.summarize_bench({"on": on})
    with pytest.raises(ValueError, match="both arms"):
        bench.summarize_bench({"on": on, "off": []})


@pytest.mark.L0
@pytest.mark.CPU
def test_check_bench_digests_two_way_and_three_way() -> None:
    trainer = {"a": dict(_DIGEST), "b": dict(_DIGEST)}
    good = {(0, 0): {"a": dict(_DIGEST), "b": dict(_DIGEST)}, (1, 0): {"a": dict(_DIGEST), "b": dict(_DIGEST)}}
    assert bench.check_bench_digests(trainer, good) == []
    assert bench.check_bench_digests(trainer, good, body_by_lane=good) == []
    corrupt = {(0, 0): {"a": dict(_DIGEST), "b": {**_DIGEST, "sha256": "0" * 64}}}
    failures = bench.check_bench_digests(trainer, corrupt)
    assert len(failures) == 1 and failures[0].startswith("lane (0, 0) worker: b:")
    missing = {(0, 0): {"a": dict(_DIGEST)}, (1, 0): {"a": dict(_DIGEST), "b": dict(_DIGEST)}}
    body_failures = bench.check_bench_digests(trainer, good, body_by_lane=missing)
    assert len(body_failures) == 1 and "lane (0, 0) body: name sets differ" in body_failures[0]


@pytest.mark.L0
@pytest.mark.CPU
def test_check_witness_values_is_bitwise() -> None:
    expected = {"w": torch.tensor([1.0, 2.0], dtype=torch.bfloat16)}
    same = {(0, 0): {"w": torch.tensor([1.0, 2.0], dtype=torch.bfloat16)}}
    assert bench.check_witness_values(expected, same) == []
    one_ulp = {(0, 0): {"w": torch.tensor([1.0, 2.015625], dtype=torch.bfloat16)}}
    assert bench.check_witness_values(expected, one_ulp) == ["lane (0, 0) w: values differ from the trainer's"]
    wrong_dtype = {(0, 0): {"w": torch.tensor([1.0, 2.0])}}
    assert "torch.float32" in bench.check_witness_values(expected, wrong_dtype)[0]
    assert "witness names" in bench.check_witness_values(expected, {(0, 0): {}})[0]


@pytest.mark.L0
@pytest.mark.CPU
def test_medians_identical_and_bucket_nbytes() -> None:
    assert bench.medians({"a": [3.0, None, 1.0, 2.0], "b": [None]}) == {"a": 2.0, "b": None}
    assert bench.bucket_nbytes([("x", (2, 3), "torch.bfloat16"), ("y", (4,), "torch.float32")]) == 12 + 16
    assert bench.identical({(0, 0): 1, (0, 1): 1}, "x") == 1
    with pytest.raises(RuntimeError, match="disagrees"):
        bench.identical({(0, 0): 1, (0, 1): 2}, "x")


def _push(total: float, *, deliver: tuple[float, ...] = (1.0, 1.0), ready: float = 0.5) -> dict:
    return {
        "version": 3,
        "plan_s": 0.1,
        "coverage_s": 0.2,
        "materialize_s": (0.3, 0.3),
        "gather_gpu_s": (0.25, None),
        "deliver_s": deliver,
        "ready_s": ready,
        "total_s": total,
    }


def _lane(fetch: float, sink: float) -> list[dict]:
    return [
        {
            "bucket": i,
            "fetch_s": fetch,
            "sink_s": sink,
            "copy_s": 0.02,
            "sync_s": 0.03,
            "compare_s": 0.05,
            "digest_s": sink - 0.1,
        }
        for i in range(2)
    ]


@pytest.mark.L0
@pytest.mark.CPU
def test_stage_rows_follow_the_critical_rank_and_its_paired_lane() -> None:
    assert bench.paired_lane((0, 3)) == (3, 0)
    push = {(0, 0): _push(3.0), (0, 1): _push(3.4, deliver=(1.2, 1.2))}  # rank 1 is the critical path
    lanes = {(0, 0): _lane(0.1, 0.6), (1, 0): _lane(0.2, 0.9)}
    verify = {
        (0, 0): {"rpc_s": 0.4, "worker_s": 0.35, "digest_s": 0.3, "verified": 3},
        (1, 0): {"rpc_s": 0.45, "worker_s": 0.4, "digest_s": 0.35, "verified": 3},
    }
    rows, totals = bench.stage_rows(4.0, push, lanes, verify)
    assert rows["push_total_s"] == 3.4 and rows["controller_residual_s"] == pytest.approx(0.6)
    assert rows["plan_coverage_s"] == pytest.approx(0.3) and rows["materialize_sum_s"] == pytest.approx(0.6)
    assert rows["gather_gpu_sum_s"] is None, "one incomplete CUDA span makes the sum unknown, not smaller"
    assert rows["deliver_sum_s"] == pytest.approx(2.4)
    assert (rows["lane_fetch_sum_s"], rows["lane_sink_sum_s"]) == (pytest.approx(0.4), pytest.approx(1.8))
    assert rows["deliver_overhead_sum_s"] == pytest.approx(2.4 - 0.4 - 1.8)
    assert rows["copy_sum_s"] == pytest.approx(0.04) and rows["sync_sum_s"] == pytest.approx(0.06)
    assert rows["compare_sum_s"] == pytest.approx(0.1) and rows["digest_sum_s"] == pytest.approx(1.6)
    assert (rows["finish_rpc_s"], rows["finish_worker_s"], rows["finish_digest_s"]) == (0.45, 0.4, 0.35), (
        "the finish record of the paired lane"
    )
    assert rows["producer_other_s"] == pytest.approx(3.4 - 0.3 - 0.6 - 2.4 - 0.5)
    assert totals["ranks"]["(0, 1)"]["deliver_sum_s"] == pytest.approx(2.4)
    assert totals["lanes"]["(0, 0)"] == {"fetch_sum_s": pytest.approx(0.2), "sink_sum_s": pytest.approx(1.2)}
    # The verification-off arm records a finish with no digest share; the plain-vLLM path records no finish at all.
    off, _ = bench.stage_rows(
        3.5, push, lanes, {(1, 0): {"rpc_s": 0.01, "worker_s": 0.005, "digest_s": None, "verified": None}}
    )
    assert (off["finish_rpc_s"], off["finish_worker_s"], off["finish_digest_s"]) == (0.01, 0.005, 0.0)
    plain, _ = bench.stage_rows(3.5, {(0, 0): _push(3.0)}, {(0, 0): [{"bucket": 0, "fetch_s": 0.1, "sink_s": 0.2}] * 2})
    assert plain["copy_sum_s"] is None and plain["compare_sum_s"] is None and plain["digest_sum_s"] is None
    assert (plain["finish_rpc_s"], plain["finish_worker_s"], plain["finish_digest_s"]) == (0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="producer timing"):
        bench.stage_rows(1.0, {(0, 0): None}, {})


@pytest.mark.L0
@pytest.mark.CPU
def test_columns_and_lane_spread_compare_cells_without_summing_them() -> None:
    assert bench.columns([{"a": 1.0, "b": 2.0}, {"a": 3.0}]) == {"a": [1.0, 3.0], "b": [2.0, None]}
    samples = [
        {"ranks": {"(0, 0)": {"total_s": 3.0}, "(0, 1)": {"total_s": 3.4}}, "lanes": {"(0, 0)": {"sink_sum_s": 1.0}}},
        {"ranks": {"(0, 0)": {"total_s": 3.2}, "(0, 1)": {"total_s": 3.6}}, "lanes": {"(0, 0)": {"sink_sum_s": None}}},
    ]
    spread = bench.lane_spread(samples)
    ranks = spread["ranks"]["total_s"]
    assert ranks["per_cell"] == {"(0, 0)": 3.1, "(0, 1)": 3.5}
    assert (ranks["min"], ranks["median"], ranks["max"]) == (3.1, 3.3, 3.5) and ranks["spread"] == pytest.approx(0.4)
    assert spread["lanes"]["sink_sum_s"]["per_cell"] == {"(0, 0)": 1.0}, "None samples are skipped, not zeroed"


@pytest.mark.L0
@pytest.mark.CPU
def test_vllm_baseline_parse_args_defaults_and_guards() -> None:
    from cosmos_framework.posttrain.tests import vllm_weight_sync_cli as cli

    args = cli.parse_args(["--out-root", "x"])
    assert (args.bench_syncs, args.bench_warmup, args.gpu_memory_utilization, args.max_model_len) == (10, 3, 0.25, 1024)
    assert args.witness == cli.DEFAULT_WITNESSES and not args.do_assert
    assert cli.parse_args(["--out-root", "x", "--witness", "model.norm.weight"]).witness == ("model.norm.weight",)
    with pytest.raises(SystemExit):
        cli.parse_args(["--out-root", "x", "--bench-syncs", "0"])
    with pytest.raises(SystemExit):
        cli.parse_args(["--out-root", "x", "--gpu-memory-utilization", "1.5"])
    with pytest.raises(SystemExit):
        cli.parse_args([])  # --out-root is required: the report is the deliverable
