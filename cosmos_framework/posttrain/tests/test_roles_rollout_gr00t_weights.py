# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""The GR00T weight-sync driver's decision functions and the probes' fault hooks, without a cluster or an engine.

`tests/gr00t_weight_sync_cli.py` decides PASS/FAIL from dictionaries the mesh returns; those decisions are pure
functions and are pinned here with a positive and a negative case each, so a driver that could not fail is caught
before it runs on GPUs. The probes' one-shot fault arming is exercised on a stub whose base records calls.
"""

from __future__ import annotations

import pytest

from cosmos_framework.posttrain.roles.rollout.vllm_omni import BucketSinkTiming
from cosmos_framework.posttrain.tests import gr00t_weight_sync_cli as cli
from cosmos_framework.posttrain.tests.helpers import gr00t_weight_sync_probes as probes

_DIGEST = {"shape": [2, 3], "dtype": "torch.bfloat16", "sha256": "ab" * 32}


@pytest.mark.L0
@pytest.mark.CPU
def test_precision_mode_vocabulary_is_the_model_module_preset_table() -> None:
    from cosmos_framework.posttrain.model.gr00t.policy import BF16_PRECISION, OFFICIAL_PRECISION, PRECISION_PRESETS

    assert PRECISION_PRESETS == {"autocast": OFFICIAL_PRECISION, "fsdp": BF16_PRECISION}
    parser_default = cli.parse_args(["--out-root", "x"]).precision_mode
    assert parser_default == "fsdp", "the driver defaults to the all-bf16 preset; `autocast` is the gated one"
    assert set(PRECISION_PRESETS) == {"autocast", "fsdp"}
    for preset in PRECISION_PRESETS.values():
        assert preset.transfer == "bfloat16", "both presets put bf16 on the wire; the engine holds bf16"


@pytest.mark.L0
@pytest.mark.CPU
def test_parse_args_defaults_and_fault_spec() -> None:
    args = cli.parse_args(["--out-root", "x"])
    assert (args.placement, args.num_train_steps, args.update_weights_interval, args.batch, args.seed) == (
        "colocated",
        2,
        1,
        1,
        1234,
    )
    assert (args.publications, args.weight_verification) == (2, "first-update")
    assert args.fault is None and args.fault_lane == -1 and args.sdpa_mode == "math" and not args.do_assert
    assert (args.bench_syncs, args.bench_warmup) == (0, 3), "the bench is off unless --bench-syncs asks for it"
    assert (
        cli.parse_args(["--out-root", "x", "--bench-syncs", "10", "--weight-verification", "always"]).bench_syncs == 10
    )
    with pytest.raises(SystemExit):
        cli.parse_args(["--out-root", "x", "--bench-syncs", "10"])  # the bench IS the A/B, it needs `always`
    with pytest.raises(SystemExit):
        cli.parse_args(
            ["--out-root", "x", "--bench-syncs", "10", "--fault", "bucket:3", "--weight-verification", "always"]
        )  # the bench and a scripted fault are different runs
    with pytest.raises(SystemExit):
        cli.parse_args(["--out-root", "x", "--bench-warmup", "-1"])
    assert cli.parse_args(["--out-root", "x", "--fault", "bucket:3"]).fault == ("bucket", "3")
    assert cli.parse_args(["--out-root", "x", "--fault", "tamper:action_head.a.W"]).fault == (
        "tamper",
        "action_head.a.W",
    )
    with pytest.raises(SystemExit):
        cli.parse_args(["--out-root", "x", "--num-train-steps", "0"])
    with pytest.raises(SystemExit):
        cli.parse_args([])  # --out-root is required: the report is the deliverable


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("spec", ["bucket", "bucket:", "bucket:x", "explode:1", "digest:"])
def test_parse_fault_rejects_malformed_specs(spec: str) -> None:
    with pytest.raises(ValueError, match="--fault"):
        probes.parse_fault(spec)


@pytest.mark.L0
@pytest.mark.CPU
def test_mesh_width_follows_the_placement() -> None:
    assert cli.mesh_width(2, 4, "colocated") == 8
    assert cli.mesh_width(2, 4, "disaggregated") == 4
    with pytest.raises(ValueError, match="split evenly"):
        cli.mesh_width(1, 3, "disaggregated")


def _colocated_two_nodes() -> dict:
    policy = {(0, 0): "nA", (0, 1): "nA", (0, 2): "nB", (0, 3): "nB"}
    rollout = {(0, 0): "nA", (1, 0): "nA", (2, 0): "nB", (3, 0): "nB"}
    pg = {(0, 0): "g0", (0, 1): "g1", (0, 2): "g2", (0, 3): "g3"}
    rg = {(0, 0): "g0", (1, 0): "g1", (2, 0): "g2", (3, 0): "g3"}
    return {"policy_nodes": policy, "rollout_nodes": rollout, "policy_gpus": pg, "rollout_gpus": rg}


@pytest.mark.L0
@pytest.mark.CPU
def test_check_topology_colocated_requires_same_node_same_gpu_and_ipc() -> None:
    good = _colocated_two_nodes()
    assert cli.check_topology("colocated", 2, **good, movers={"IpcMover"}) == []
    # a pair split across nodes
    bad = _colocated_two_nodes()
    bad["rollout_nodes"][(3, 0)] = "nA"
    assert any("different nodes" in f for f in cli.check_topology("colocated", 2, **bad, movers={"IpcMover"}))
    # a pair on different cards of the same node
    bad = _colocated_two_nodes()
    bad["rollout_gpus"][(1, 0)] = "g9"
    assert any("different GPUs" in f for f in cli.check_topology("colocated", 2, **bad, movers={"IpcMover"}))
    # the node count the run claims must be what the cells landed on
    assert any("expected 1 node" in f for f in cli.check_topology("colocated", 1, **good, movers={"IpcMover"}))
    # a Copy fallback would also pass value equality, so the mover class is part of the proof
    assert any("IpcMover" in f for f in cli.check_topology("colocated", 2, **good, movers={"CopyMover"}))


@pytest.mark.L0
@pytest.mark.CPU
def test_check_topology_disaggregated_requires_disjoint_nodes_and_nixl() -> None:
    policy = {(0, 0): "nA", (0, 1): "nA"}
    rollout = {(0, 0): "nB", (1, 0): "nB"}
    pg, rg = {(0, 0): "g0", (0, 1): "g1"}, {(0, 0): "g2", (1, 0): "g3"}
    ok = cli.check_topology("disaggregated", 2, policy, rollout, pg, rg, {"NixlWeightMover"})
    assert ok == []
    # bind has no node anti-affinity: both sides landing on one node must be reported, not assumed away
    same_node = cli.check_topology(
        "disaggregated", 1, policy, {(0, 0): "nA", (1, 0): "nA"}, pg, rg, {"NixlWeightMover"}
    )
    assert any("share node" in f for f in same_node)
    assert any(
        "NixlWeightMover" in f for f in cli.check_topology("disaggregated", 2, policy, rollout, pg, rg, {"IpcMover"})
    )


@pytest.mark.L0
@pytest.mark.CPU
def test_check_digest_chain_needs_all_three_to_agree() -> None:
    a = {"action_head.a": _DIGEST, "action_head.b": {**_DIGEST, "sha256": "cd" * 32}}
    assert cli.check_digest_chain(a, dict(a), dict(a)) == []
    worker = {**a, "action_head.b": {**_DIGEST, "sha256": "ef" * 32}}
    assert any("action_head.b" in f for f in cli.check_digest_chain(a, a, worker))
    assert any("name sets differ" in f for f in cli.check_digest_chain(a, a, {"action_head.a": _DIGEST}))
    # Unverified publication: the body must hold NOTHING (an expectation would mean the switch did not take),
    # while the trainer's snapshot and the worker's live bytes are still read independently and must agree.
    assert cli.check_digest_chain(a, {}, dict(a), verified=False) == []
    assert any("expected none" in f for f in cli.check_digest_chain(a, dict(a), dict(a), verified=False))
    assert any("action_head.b" in f for f in cli.check_digest_chain(a, {}, worker, verified=False))
    assert any(
        "name sets differ" in f for f in cli.check_digest_chain(a, {}, {"action_head.a": _DIGEST}, verified=False)
    )


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.L0
@pytest.mark.CPU
def test_verification_policy_maps_publications_to_the_production_flag() -> None:
    """`first-update` is the steady-state shape: prove v0 and v1, then stop paying for the checks."""
    assert [cli.verification_for("first-update", v) for v in range(4)] == [True, True, False, False]
    assert [cli.verification_for("never", v) for v in range(3)] == [False, False, False]
    assert [cli.verification_for("always", v) for v in range(3)] == [True, True, True]
    with pytest.raises(ValueError, match="unknown weight-verification policy"):
        cli.verification_for("sometimes", 0)
    # The faults are placed where they can be caught: digest/tamper need the checks, bucket needs them off.
    assert cli.FAULT_VERSION == {"digest": 1, "tamper": 1, "bucket": 2}
    assert cli.verification_for("first-update", cli.FAULT_VERSION["digest"]) is True
    assert cli.verification_for("first-update", cli.FAULT_VERSION["bucket"]) is False


@pytest.mark.L0
@pytest.mark.CPU
def test_parse_args_rejects_training_shapes_the_policy_cannot_express() -> None:
    base = ["--out-root", "x", "--num-train-steps", "4", "--update-weights-interval", "2"]
    assert cli.parse_args(base).publications == 2
    with pytest.raises(SystemExit):  # 3 steps do not divide into 2-step publications
        cli.parse_args(["--out-root", "x", "--num-train-steps", "3", "--update-weights-interval", "2"])
    with pytest.raises(SystemExit):  # one publication cannot show `first-update` switching off
        cli.parse_args(["--out-root", "x", "--num-train-steps", "2", "--update-weights-interval", "2"])
    assert (
        cli.parse_args(
            [
                "--out-root",
                "x",
                "--num-train-steps",
                "2",
                "--update-weights-interval",
                "2",
                "--weight-verification",
                "never",
            ]
        ).publications
        == 1
    )
    with pytest.raises(SystemExit):  # a digest fault lands on v1, which `never` does not verify
        cli.parse_args([*base, "--weight-verification", "never", "--fault", "digest:action_head.a.W"])
    with pytest.raises(SystemExit):  # a bucket fault lands on v2, which `always` still verifies
        cli.parse_args([*base, "--weight-verification", "always", "--fault", "bucket:3"])
    assert cli.parse_args([*base, "--fault", "bucket:3"]).fault == ("bucket", "3")
    with pytest.raises(SystemExit):  # v2 is never published by a single-publication run
        cli.parse_args(
            ["--out-root", "x", "--num-train-steps", "1", "--weight-verification", "never", "--fault", "bucket:3"]
        )


@pytest.mark.L0
@pytest.mark.CPU
def test_check_clocks_ties_the_three_clocks_and_the_optimizer_counter() -> None:
    clocks = {(0, 0): {"iteration": 2, "publication_version": 2, "optimizer_step": 8}}
    assert (
        cli.check_clocks(
            2, publications_done=2, update_weights_interval=4, clocks=clocks, versions={(0, 0): 2, (1, 0): 2}
        )
        == []
    )
    assert cli.check_clocks(2, 2, 4, clocks, {(0, 0): 2, (1, 0): 1})  # a lane behind
    assert cli.check_clocks(2, 2, 3, clocks, {(0, 0): 2})  # 8 steps is not 2 x 3
    stale = {(0, 0): {"iteration": 1, "publication_version": 2, "optimizer_step": 8}}
    assert cli.check_clocks(2, 2, 4, stale, {(0, 0): 2})


@pytest.mark.L0
@pytest.mark.CPU
def test_identical_rejects_a_disagreeing_cell() -> None:
    assert cli.identical({(0, 0): 1, (0, 1): 1}, "x") == 1
    with pytest.raises(RuntimeError, match="disagrees"):
        cli.identical({(0, 0): 1, (0, 1): 2}, "x")


class _PlanBackend:
    """The bound backend as the probe reads it: the plan (one 2-float tensor per bucket) and the consumer records."""

    def __init__(self) -> None:
        self.apply_timings: list = []

    def plan(self) -> list[list]:
        return [[("x", (2,), "torch.float32")], [("x", (2,), "torch.float32")]]


class _RecordingBase:
    """Stands in for `Gr00tPolicyBody` under the probe mixin: records what the mixin lets through, and keeps the
    body's sync-cost records the way `OmniPolicyBody` does (reset at bucket 0, one `BucketSinkTiming` per sink)."""

    def __init__(self) -> None:
        self.sunk: list[dict] = []
        self.ready: list[int] = []
        self._expected_digests = {"action_head.a": dict(_DIGEST)}
        self._weight_backend = _PlanBackend()
        self.bucket_timings: list[BucketSinkTiming] = []
        self.verify_timing = None
        self._stream_open = False

    def _coverage_reset(self, i: int) -> None:
        if i == 0:
            self.bucket_timings = []

    def _engine_sink(self, state: dict) -> set[str]:
        self.sunk.append(state)
        self.bucket_timings.append(BucketSinkTiming(copy_s=0.25, sync_s=0.0, compare_s=0.0, digest_s=0.0))
        return set(state)

    def mark_weight_version_ready(self, version: int) -> None:
        self.ready.append(version)


class _Stub(probes.RolloutProbe, _RecordingBase):
    pass


@pytest.mark.L0
@pytest.mark.CPU
def test_bucket_fault_is_armed_on_one_lane_and_fires_once() -> None:
    lane0, lane1 = _Stub(), _Stub()
    lane0.replica_index, lane1.replica_index = 0, 1
    assert [lane0.arm_fault("bucket", "1", lane=1), lane1.arm_fault("bucket", "1", lane=1)] == [False, True]
    for lane in (lane0, lane1):
        lane._coverage_reset(0)
        lane._engine_sink({"x": __import__("torch").zeros(2)})
    lane0._coverage_reset(1)
    lane0._engine_sink({"x": __import__("torch").zeros(2)})
    lane1._coverage_reset(1)
    with pytest.raises(RuntimeError, match="injected transport fault at bucket 1 on lane 1"):
        lane1._engine_sink({"x": __import__("torch").zeros(2)})
    assert lane1._fault is None, "one-shot: the recovery sync must not hit the same fault"
    lane1._engine_sink({"x": __import__("torch").zeros(2)})
    assert len(lane0.sunk) == 2 and len(lane1.sunk) == 2
    rows = lane1.sink_timings()
    assert [(r["bucket"], r["names"], r["bytes"], r["copy_s"]) for r in rows] == [(0, 1, 8, 0.25), (1, 1, 8, 0.25)]
    assert lane1.verify_record() is None
    lane1._coverage_reset(0)
    assert lane1.sink_timings() == [], "bucket 0 starts the next stream's records"


@pytest.mark.L0
@pytest.mark.CPU
def test_set_verify_weight_values_refuses_mid_stream() -> None:
    lane = _Stub()
    lane._stream_open = True
    with pytest.raises(RuntimeError, match="stream is open"):
        lane.set_verify_weight_values(False)
    lane._stream_open = False
    assert lane.set_verify_weight_values(False) is False and lane.verify_weight_values is False
    assert lane.set_verify_weight_values(True) is True


@pytest.mark.L0
@pytest.mark.CPU
def test_digest_fault_corrupts_one_expectation_before_readiness_then_disarms() -> None:
    lane = _Stub()
    lane.replica_index = 0
    lane.arm_fault("digest", "action_head.a", lane=0)
    lane.mark_weight_version_ready(3)
    assert lane._expected_digests["action_head.a"]["sha256"] == "0" * 64
    assert lane.ready == [3] and lane._fault is None
    with pytest.raises(KeyError, match="not a live parameter"):
        lane.arm_fault("digest", "action_head.nope", lane=0)
        lane.mark_weight_version_ready(4)
