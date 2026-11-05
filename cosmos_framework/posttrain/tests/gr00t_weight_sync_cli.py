# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Stage 3's acceptance driver: train the GR00T-N1.7 head, hot-update the vLLM-Omni engines, prove what they serve.

One launcher for all six combinations -- three topologies (single-node colocated, multi-node colocated, multi-node
disaggregated over NIXL) times two precision presets (`autocast`, `fsdp`) -- so the acceptance criteria cannot
drift between them. The topology comes from the Ray cluster it is started in (`utils/mesh.py::detect_mesh`) and
`--placement`; the presets are `model/gr00t/policy.py::PRECISION_PRESETS`. It is a test launcher like
`tests/gr00t_parity_cli.py`, not an entrypoint: it manages the training loop and the syncs itself.

Training runs `--num-train-steps` optimizer steps and publishes every `--update-weights-interval` of them, so
version k is the k-th publication and carries k x interval steps. `--weight-verification` decides which
publications run the production byte-level checks (`verify_weight_values`, one bool over the byte-wise compare,
the body's wire digests and the worker's SHA256 re-read): `first-update` verifies v0 and v1 and then switches
off, which is what a steady-state loop does; `never` verifies none; `always` is the old A/B baseline. The switch
only ever moves between streams.

Every published version k (0, then one per publication, then the negative-control publications) is accepted by
the same checks, and any missing item is a FAIL:
  - digest chain   verified: trainer publication snapshot == body's recorded expectation == every worker's live
                   parameters; unverified: the body must hold NO expectation and the trainer's snapshot must
                   still equal every worker's live parameters, read outside the timed sync
  - switch         every lane reports the `verify_weight_values` the policy asked for at that version
  - clocks         rollout.weight_version() == policy.iteration() == policy.publication_version() == k,
                   optimizer_step == k x update-weights-interval
  - gate           `tests/helpers/gr00t_parity_gate.py::run_gate` against a reference loaded from the exported
                   checkpoint, through the HOT engines (all cells serve the same action), with diagnostic D
  - hot == cold    the hot workers' live digests equal a separate engine's cold load of the export
  - stability      worker pid unchanged across a normal sync; the body's parameter mapping is of that worker
  - data E         |master - wire(master)| on the trainer (zero for `fsdp`, reported for `autocast`)
Fault scenarios (`--fault`) are injected INTO a publication, not appended after it: `digest` and `tamper` into
v1, where the byte-level checks are the only thing that can catch them, and `bucket` into v2, where they are off
and the transport failure must still block and recover. The injected failure must raise, leave no version
committed, block inference on every lane (weight_version=None included), and the recovery (`rebuild_engine` on
the invalidated lane, then a full sync of the SAME version, with no further optimizer step and the same switch)
must change that lane's worker pid and restore serving. `--assert` turns FAIL into a
non-zero exit; the JSON report under `--out-root` is the deliverable either way.

Run inside the Ray cluster (see `operator-provided multinode documentation` for the launcher) with `GR00T_DIT_SDPA_MODE=math`; the
driver itself hosts the reference/serving/native models of the gate on its own GPU 0.
"""

from __future__ import annotations

import argparse
import copy
import functools
import json
import os
import sys
import time
from typing import Any

import numpy as np

from cosmos_framework.posttrain.tests.helpers.gr00t_weight_sync_probes import (
    SDPA_MODE_ENV,
    parse_fault,
    setup_policy,
    setup_rollout,
)
from cosmos_framework.posttrain.tests.helpers.weight_sync_bench import (  # noqa: F401 -- mesh_width/identical re-exported
    ARM_OFF,
    ARM_ON,
    arm_order,
    bind_pair,
    check_bench_digests,
    columns,
    identical,
    lane_spread,
    medians,
    mesh_width,
    revisions,
    stage_rows,
    summarize_bench,
)

#: Which head parameter the negative control perturbs: a bias in the last decoder layer moves every action.
PERTURB_PARAMETER = "action_head.action_decoder.layer2.b"

#: How `--weight-verification` maps a publication to the production `verify_weight_values` flag. That flag stays a
#: plain bool over ALL byte-level checks; only this driver decides when to flip it, and only between streams.
#: `first-update` is what a steady-state loop does -- prove the plane on the first publications, then stop paying
#: two SHA256 passes per sync -- and `always` is the old A/B baseline.
VERIFICATION_POLICIES = ("first-update", "never", "always")

#: Which publication each fault is injected into. `digest` and `tamper` are visible only to the byte-level checks,
#: so they go into v1, which `first-update` still verifies; `bucket` is a transport failure that must block and
#: recover with the checks OFF, so it goes into v2, which no policy in this matrix verifies.
FAULT_VERSION = {"digest": 1, "tamper": 1, "bucket": 2}


def verification_for(policy: str, publication: int) -> bool:
    """Whether publication `publication` (0 = the initial sync) runs the byte-level checks under `policy`."""

    if policy == "always":
        return True
    if policy == "never":
        return False
    if policy == "first-update":
        return publication <= 1
    raise ValueError(f"unknown weight-verification policy {policy!r}, expected one of {VERIFICATION_POLICIES}")


# ── pure checks (unit-tested) ─────────────────────────────────────────────────────────────────────────────────


def check_topology(
    placement: str,
    nodes: int,
    policy_nodes: dict[Any, str],
    rollout_nodes: dict[Any, str],
    policy_gpus: dict[Any, str],
    rollout_gpus: dict[Any, str],
    movers: set[str],
) -> list[str]:
    """Return the ways the bound mesh fails to be the topology the run claims (empty == proven)."""
    failures: list[str] = []
    seen_nodes = set(policy_nodes.values()) | set(rollout_nodes.values())
    if len(seen_nodes) != nodes:
        failures.append(f"expected {nodes} node(s), the cells landed on {len(seen_nodes)}: {sorted(seen_nodes)}")
    if placement == "colocated":
        for p_cell, r_cell in zip(sorted(policy_nodes), sorted(rollout_nodes), strict=True):
            if policy_nodes[p_cell] != rollout_nodes[r_cell]:
                failures.append(f"cells {p_cell}->{r_cell} are on different nodes; IPC needs the same node")
            if policy_gpus[p_cell] != rollout_gpus[r_cell]:
                failures.append(f"cells {p_cell}->{r_cell} hold different GPUs; IPC needs the same card")
        if movers != {"IpcMover"}:
            failures.append(f"colocated placement must bind IpcMover on every lane, got {sorted(movers)}")
    else:
        shared = set(policy_nodes.values()) & set(rollout_nodes.values())
        if shared:
            failures.append(f"disaggregated sides share node(s) {sorted(shared)}; the NIXL edge must cross nodes")
        if set(policy_gpus.values()) & set(rollout_gpus.values()):
            failures.append("disaggregated sides share a GPU")
        if movers != {"NixlWeightMover"}:
            failures.append(f"disaggregated placement must bind NixlWeightMover on every lane, got {sorted(movers)}")
    return failures


def check_digest_chain(
    trainer: dict[str, dict[str, Any]],
    body: dict[str, dict[str, Any]],
    worker: dict[str, dict[str, Any]],
    *,
    verified: bool = True,
) -> list[str]:
    """The digest sets must be one set: same names, same shape/dtype/sha256 everywhere.

    With `verified=False` the publication ran without the byte-level checks, so the body recorded no expectation:
    holding one would mean the switch did not take. The trainer's snapshot and the worker's live parameters are
    still read independently and must still agree -- that comparison happens outside the timed sync, so it proves
    the bytes arrived without putting a digest pass back into the publication.
    """
    failures: list[str] = []
    if not verified:
        if body:
            failures.append(f"body holds {len(body)} digests for an unverified publication, expected none")
        if set(trainer) != set(worker):
            failures.append(
                f"digest name sets differ: trainer {len(trainer)}, worker {len(worker)}; "
                f"e.g. {sorted(set(trainer) ^ set(worker))[:5]}"
            )
        failures.extend(
            f"{name}: trainer {trainer[name]} worker {worker[name]}"
            for name in sorted(set(trainer) & set(worker))
            if trainer[name] != worker[name]
        )
        return failures
    if set(trainer) != set(body) or set(body) != set(worker):
        failures.append(
            f"digest name sets differ: trainer {len(trainer)}, body {len(body)}, worker {len(worker)}; "
            f"e.g. {sorted((set(trainer) ^ set(body)) | (set(body) ^ set(worker)))[:5]}"
        )
    for name in sorted(set(trainer) & set(body) & set(worker)):
        if not (trainer[name] == body[name] == worker[name]):
            failures.append(f"{name}: trainer {trainer[name]} body {body[name]} worker {worker[name]}")
    return failures


def check_clocks(
    version: int,
    publications_done: int,
    update_weights_interval: int,
    clocks: dict[Any, dict[str, int]],
    versions: dict[Any, int],
) -> list[str]:
    """Every clock reads the publication version, and the optimizer counter reads the steps behind it."""
    failures: list[str] = []
    for cell, clock in clocks.items():
        if clock["iteration"] != version or clock["publication_version"] != version:
            failures.append(f"policy cell {cell}: iteration/publication {clock} != published version {version}")
        if clock["optimizer_step"] != publications_done * update_weights_interval:
            failures.append(
                f"policy cell {cell}: optimizer_step {clock['optimizer_step']} != "
                f"{publications_done} publications x {update_weights_interval} steps"
            )
    for cell, held in versions.items():
        if held != version:
            failures.append(f"rollout cell {cell} holds weight_version {held}, expected {version}")
    return failures


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", default=os.environ.get("GR00T_CKPT", "nvidia/GR00T-N1.7-DROID"))
    parser.add_argument("--placement", choices=("colocated", "disaggregated"), default="colocated")
    parser.add_argument("--precision-mode", choices=("autocast", "fsdp"), default="fsdp")
    parser.add_argument("--num-train-steps", type=int, default=2, help="optimizer steps in total, after version 0")
    parser.add_argument(
        "--update-weights-interval", type=int, default=1, help="optimizer steps between weight publications"
    )
    parser.add_argument(
        "--weight-verification",
        choices=VERIFICATION_POLICIES,
        default="first-update",
        help=(
            "which publications run the byte-level checks: first-update (v0 and the first trained publication), "
            "never, or always (the old A/B baseline)"
        ),
    )
    parser.add_argument("--train-batch", type=int, default=1, help="trajectories per policy rank per step")
    parser.add_argument("--batch", type=int, default=1, help="observations per gate request")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--fault", default=None, help="bucket:<i> | digest:<name> | tamper:<name>, one scenario")
    parser.add_argument("--fault-lane", type=int, default=-1, help="rollout replica to inject on (-1: the last)")
    parser.add_argument("--no-perturb", action="store_true", help="skip the controlled-perturbation publications")
    parser.add_argument("--sdpa-mode", default="math", help=f"value of {SDPA_MODE_ENV} on every side")
    parser.add_argument("--nixl-probe-bytes", type=int, default=None)
    parser.add_argument("--nixl-min-gbps", type=float, default=None)
    parser.add_argument(
        "--bench-syncs", type=int, default=0, help="measured syncs per arm of the verification on/off bench (0: off)"
    )
    parser.add_argument("--bench-warmup", type=int, default=3, help="warmup syncs per arm before the measured ones")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--assert", dest="do_assert", action="store_true", help="non-zero exit unless OVERALL PASS")
    args = parser.parse_args(argv)
    args.fault = parse_fault(args.fault)
    if args.num_train_steps < 1 or args.update_weights_interval < 1 or args.train_batch < 1 or args.batch < 1:
        parser.error("--num-train-steps, --update-weights-interval, --train-batch and --batch must be >= 1")
    if args.num_train_steps % args.update_weights_interval:
        parser.error(
            f"--num-train-steps {args.num_train_steps} is not a multiple of --update-weights-interval "
            f"{args.update_weights_interval}: the last publication would carry a partial interval"
        )
    args.publications = args.num_train_steps // args.update_weights_interval
    if args.weight_verification == "first-update" and args.publications < 2:
        parser.error(
            "--weight-verification first-update needs at least two trained publications to show the switch off; "
            f"--num-train-steps {args.num_train_steps} / --update-weights-interval {args.update_weights_interval} "
            "gives one"
        )
    if args.fault is not None:
        kind = args.fault[0]
        if kind in ("digest", "tamper") and not verification_for(args.weight_verification, 1):
            parser.error(
                f"--fault {kind} is caught by the byte-level checks, which --weight-verification "
                f"{args.weight_verification} turns off at the publication it is injected into (v1)"
            )
        if kind == "bucket" and verification_for(args.weight_verification, FAULT_VERSION["bucket"]):
            parser.error(
                "--fault bucket exercises the blocking and recovery of an UNVERIFIED publication; "
                f"--weight-verification {args.weight_verification} still verifies v{FAULT_VERSION['bucket']}"
            )
        if args.publications < FAULT_VERSION[kind]:
            parser.error(f"--fault {kind} is injected at v{FAULT_VERSION[kind]}, which this run never publishes")
    if args.bench_syncs < 0 or args.bench_warmup < 0:
        parser.error("--bench-syncs and --bench-warmup must be >= 0")
    if args.bench_syncs > 0 and args.fault is not None:
        parser.error(
            "--bench-syncs runs half of its syncs with verification off, where --fault scenarios are meaningless"
        )
    if args.bench_syncs > 0 and args.weight_verification != "always":
        parser.error(
            "--bench-syncs IS the on/off A/B: it drives verify_weight_values per arm, so it needs "
            "--weight-verification always for the publications that precede it"
        )
    return args


# ── the run ───────────────────────────────────────────────────────────────────────────────────────────────────
class _MeshBody:
    """What the gate needs from the rollout WG: one action per request, proven identical on every cell."""

    def __init__(self, rollout: Any) -> None:
        self._wg = rollout
        self.pids: dict[Any, int] = {}

    def infer_batch(self, obs_list: list[dict], *, seed: int | None, weight_version: int | None) -> list[dict]:
        per_cell = self._wg.infer_batch(obs_list, seed=seed, weight_version=weight_version).all()
        cells = sorted(per_cell)
        first = per_cell[cells[0]]
        for cell in cells[1:]:
            for a, b in zip(first, per_cell[cell], strict=True):
                if not np.array_equal(a["action"], b["action"]):
                    raise RuntimeError(f"rollout cell {cell} serves a different action than cell {cells[0]}")
        return first

    def describe_pipeline(self) -> dict[str, Any]:
        per_cell = self._wg.describe_pipeline().all()
        self.pids = {cell: info["worker_pid"] for cell, info in per_cell.items()}
        return per_cell[sorted(per_cell)[0]]


class Run:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.out_root = os.path.abspath(args.out_root)
        os.makedirs(self.out_root, exist_ok=True)
        self.failures: list[str] = []
        self.report: dict[str, Any] = {
            "command": " ".join(sys.argv),
            "args": {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(args).items()},
            "revisions": revisions(),
            "versions": [],
            "fault": None,
            "perturbation": None,
        }
        self.publications_done = 0
        self.verifying: bool | None = None
        self.previous_states: dict[Any, dict[str, Any]] | None = None
        self.actions_by_version: dict[int, np.ndarray] = {}
        self.digests_by_version: dict[int, dict[str, dict[str, Any]]] = {}

    def fail(self, message: str) -> None:
        print(f"FAIL: {message}")
        self.failures.append(message)

    # -- setup --------------------------------------------------------------------------------------------------
    def bind(self) -> None:
        args = self.args
        bound = bind_pair(
            functools.partial(
                setup_rollout, checkpoint=args.checkpoint, precision_mode=args.precision_mode, sdpa_mode=args.sdpa_mode
            ),
            functools.partial(
                setup_policy,
                checkpoint=args.checkpoint,
                precision_mode=args.precision_mode,
                sdpa_mode=args.sdpa_mode,
                out_root=self.out_root,
            ),
            placement=args.placement,
            name="gr00t",
            nixl_probe_bytes=args.nixl_probe_bytes,
            nixl_min_gbps=args.nixl_min_gbps,
        )
        self.rollout, self.policy = bound.rollout, bound.policy
        self.nodes, self.width = bound.nodes, bound.width
        self.report["mesh"] = {
            "nodes": bound.nodes,
            "gpus_per_node": bound.gpus_per_node,
            "width": bound.width,
            "placement": args.placement,
        }
        self.report["bind_seconds"] = bound.bind_seconds

    def prove_topology(self) -> None:
        import ray

        policy_nodes, rollout_nodes = self.policy.node_id().all(), self.rollout.node_id().all()
        policy_ids = {c: ray.get(a.gpu_identity.remote()) for c, a in self.policy._backend.actors.items()}
        rollout_ids = {c: ray.get(a.gpu_identity.remote()) for c, a in self.rollout._backend.actors.items()}
        states = self.rollout.probe_state().all()
        envs = self.policy.trainer_env().all()
        movers = {s["mover_kind"] for s in states.values()} | {e["mover_kind"] for e in envs.values()}
        failures = check_topology(
            self.args.placement,
            self.nodes,
            policy_nodes,
            rollout_nodes,
            {c: i["cuda_uuid"] for c, i in policy_ids.items()},
            {c: i["cuda_uuid"] for c, i in rollout_ids.items()},
            movers,
        )
        sdpa = {
            "trainer": sorted({e["sdpa_mode"] for e in envs.values()}),
            "body": sorted({s["body_sdpa_mode"] for s in states.values()}),
            "worker": sorted({s["worker_sdpa_mode"] for s in states.values()}),
        }
        # Isaac honours the override (the trainer actors); the vLLM-Omni port has no reader for it, so on the body
        # and the worker this proves the environment propagated, not the kernel used.
        for side, modes in sdpa.items():
            if modes != [self.args.sdpa_mode]:
                failures.append(f"{SDPA_MODE_ENV} on the {side} side is {modes}, expected [{self.args.sdpa_mode!r}]")
        precisions = {json.dumps(e["precision"], sort_keys=True) for e in envs.values()}
        transfer = {e["transfer_dtype"] for e in envs.values()} | {s["transfer_dtype"] for s in states.values()}
        if len(precisions) != 1 or len(transfer) != 1:
            failures.append(f"precision/transfer dtype disagree across cells: {precisions} / {transfer}")
        print(
            f"topology: nodes={sorted(set(policy_nodes.values()) | set(rollout_nodes.values()))} movers={sorted(movers)} failures={len(failures)}"
        )
        self.report["topology"] = {
            "policy_nodes": {str(c): n for c, n in policy_nodes.items()},
            "rollout_nodes": {str(c): n for c, n in rollout_nodes.items()},
            "policy_gpus": {str(c): i for c, i in policy_ids.items()},
            "rollout_gpus": {str(c): i for c, i in rollout_ids.items()},
            "movers": sorted(movers),
            "sdpa_mode": sdpa,
            "precision": [json.loads(p) for p in precisions],
            "failures": failures,
        }
        for failure in failures:
            self.fail(f"topology: {failure}")

    # -- data -----------------------------------------------------------------------------------------------------
    def prepare_data(self) -> None:
        from cosmos_framework.posttrain.configs.gr00t_n1d7_droid import make_model_config
        from cosmos_framework.posttrain.model.gr00t.packer import action_mask_for, transition_from_processed
        from cosmos_framework.posttrain.model.gr00t.policy import load_processor
        from cosmos_framework.posttrain.schema import Status, Trajectory
        from cosmos_framework.posttrain.tests.helpers.gr00t_synthetic import build_observations, synthetic_samples

        args = self.args
        self.config = make_model_config(checkpoint_path=args.checkpoint)
        processor = load_processor(args.checkpoint)
        per_step = self.width * args.train_batch
        total = args.num_train_steps * per_step
        samples = synthetic_samples(processor, self.config, total, seed=args.seed)
        if self.config.training_prefix != self.config.model_action_horizon:
            for sample in samples:
                sample["action_mask"] = action_mask_for(
                    sample["action"], action_dim=self.config.action_dim, training_prefix=self.config.training_prefix
                )
        self.step_batches = [
            [
                Trajectory(
                    transitions=[transition_from_processed(copy.deepcopy(s))],
                    id=f"step{step}-{i}",
                    status=Status.COMPLETED,
                )
                for i, s in enumerate(samples[step * per_step : (step + 1) * per_step])
            ]
            for step in range(args.num_train_steps)
        ]
        self.obs_list = build_observations(self.config, args.batch)

    # -- one publication ----------------------------------------------------------------------------------------
    def sync(self, version: int) -> dict[str, Any]:
        from cosmos_framework.posttrain.control.loop.weight_sync import weight_sync

        started = time.perf_counter()
        weight_sync(self.rollout, self.policy, validation_step=version)
        seconds = time.perf_counter() - started
        print(f"[v{version}] weight_sync committed in {seconds:.1f}s")
        return {"seconds": seconds}

    def accept(self, version: int, kind: str, *, expect_pid_change: set[int] = frozenset()) -> dict[str, Any]:
        """Every acceptance item for one published version; failures are recorded, never raised."""
        from cosmos_framework.posttrain.configs.gr00t_n1d7_droid import make_model_config
        from cosmos_framework.posttrain.model.gr00t.policy import PRECISION_PRESETS, Gr00tN1d7TrainPolicy
        from cosmos_framework.posttrain.scripts.export_gr00t_checkpoint import export_checkpoint, read_head_state
        from cosmos_framework.posttrain.tests.helpers.gr00t_parity_gate import (
            evaluate_gate,
            isaac_auto_classes,
            render,
            run_gate,
        )

        entry: dict[str, Any] = {"version": version, "kind": kind, "failures": []}
        failures: list[str] = entry["failures"]

        # digest chain: trainer snapshot -> body expectation -> worker live bytes, on every lane
        trainer_digests = identical(self.policy.publication_digests().all(), "trainer publication digests")
        body_digests = self.rollout.expected_digests().all()
        worker_digests = self.rollout.worker_digests().all()
        verified = bool(self.verifying)
        for cell in sorted(body_digests):
            failures.extend(
                f"lane {cell} {f}"
                for f in check_digest_chain(
                    trainer_digests, body_digests[cell], worker_digests[cell], verified=verified
                )
            )
        entry["verified"] = verified
        entry["digest_names"] = len(trainer_digests)
        self.digests_by_version[version] = trainer_digests
        print(
            f"[v{version}] digest chain over {len(trainer_digests)} names on {len(body_digests)} lanes: {len(failures)} failure(s)"
        )

        # clocks
        clocks = self.policy.clock().all()
        held = self.rollout.weight_version().all()
        failures.extend(check_clocks(version, self.publications_done, self.args.update_weights_interval, clocks, held))
        entry["clock"] = identical(clocks, "policy clock")

        # pid / buffer stability, memory
        states = self.rollout.probe_state().all()
        entry["lanes"] = {str(c): s for c, s in states.items()}
        if self.previous_states is not None:
            for cell, state in states.items():
                before = self.previous_states[cell]
                lane = state["replica_index"]
                if (state["worker_pid"] != before["worker_pid"]) != (lane in expect_pid_change):
                    failures.append(
                        f"lane {cell}: worker pid {before['worker_pid']} -> {state['worker_pid']}, "
                        f"expected {'a change' if lane in expect_pid_change else 'no change'}"
                    )
        self.previous_states = states
        for cell, state in states.items():
            if state["stream_open"] or state["engine_invalidated"] or state["pending"] is not None:
                failures.append(f"lane {cell} is not cleanly committed: {state}")
            if bool(state["verify_weight_values"]) != verified:
                failures.append(
                    f"lane {cell}: verify_weight_values={state['verify_weight_values']}, the policy "
                    f"{self.args.weight_verification} wants {verified} at v{version}"
                )
            if state["mapping_pid"] != state["worker_pid"]:
                failures.append(
                    f"lane {cell}: the body maps the parameters of pid {state['mapping_pid']}, the worker is "
                    f"{state['worker_pid']}"
                )
        entry["sink_timings"] = {str(c): t for c, t in self.rollout.sink_timings().all().items()}
        entry["plan"] = identical(self.policy.plan_summary().all(), "bucket plan")
        if entry["plan"]["dtypes"] != [f"torch.{PRECISION_PRESETS[self.args.precision_mode].transfer}"]:
            failures.append(f"plan dtypes {entry['plan']['dtypes']} are not the preset's transfer dtype")

        # data E (the full head is gathered to rank 0; the other ranks answer None)
        gaps = [g for g in self.policy.master_to_wire_gap().all().values() if g is not None]
        if len(gaps) != 1:
            failures.append(f"data E: expected exactly one rank to hold the full head, got {len(gaps)}")
        entry["master_to_wire_gap"] = gaps[0] if gaps else None
        if self.args.precision_mode == "fsdp" and gaps and gaps[0]["max"] != 0.0:
            failures.append(f"fsdp preset has a master->wire gap: {gaps[0]}")

        # checkpoint + export
        iter_dir = os.path.join(self.out_root, "ckpt", f"iter_{version:06d}")
        export_dir = os.path.join(self.out_root, "export", f"iter_{version:06d}")
        self.policy.save(iter_dir).all()
        # bf16: what the engines hold and serve, and the base checkpoint's own storage layout, so the reference and
        # the in-process port load it exactly as they load the base.
        export_checkpoint(base=self.args.checkpoint, posttrain_ckpt=iter_dir, out=export_dir, dtype="bf16")
        entry["checkpoint"], entry["export"] = iter_dir, export_dir
        print(f"[v{version}] checkpoint {iter_dir} exported to {export_dir}")

        # gate against a reference loaded from the export, through the hot engines, plus diagnostic D
        config = make_model_config(checkpoint_path=export_dir)
        native_dir = os.path.join(self.out_root, "cfg", f"native_{version:06d}")
        os.makedirs(native_dir, exist_ok=True)
        # Diagnostic D's native copy: the base model under the preset, plus the head straight from the DCP checkpoint
        # (the trainer's master values, not a re-export), unsharded so `get_action` sees whole parameters.
        with isaac_auto_classes():
            native = Gr00tN1d7TrainPolicy(
                self.config,
                device="cuda:0",
                save_cfg_dir=native_dir,
                precision=PRECISION_PRESETS[self.args.precision_mode],
            )
            native.net.load_state_dict(read_head_state(iter_dir, native.net.state_dict()), strict=True)
        mesh_body = _MeshBody(self.rollout)
        try:
            gate = run_gate(
                checkpoint=export_dir,
                config=config,
                batch=self.args.batch,
                seed=self.args.seed,
                body=mesh_body,
                weight_version=version,
                native_policy=native,
            )
        finally:
            del native
            _free_cuda()
        print(render(gate))
        gate_path = os.path.join(self.out_root, f"gate_v{version:06d}.json")
        with open(gate_path, "w") as handle:
            json.dump(gate, handle, indent=2)
        ok, over, not_exact = evaluate_gate(gate)
        if not ok:
            failures.append(
                f"gate: {[d['name'] for d in over + not_exact]} processor_identical={gate.get('processor_identical')}"
            )
        if "diagnostic" not in gate or not gate["diagnostic"]["rows"]:
            failures.append("gate: diagnostic D missing")
        entry["gate"] = {"path": gate_path, "ok": ok, "diagnostic": gate.get("diagnostic")}
        engine_row = next(d for d in gate["diffs"] if d["name"] == "absolute_action_via_engine")
        entry["engine_row"] = engine_row
        actions = self.rollout.infer_batch(self.obs_list, seed=self.args.seed, weight_version=version).all()
        self.actions_by_version[version] = np.stack([a["action"] for a in actions[sorted(actions)[0]]])
        for cell, pid in mesh_body.pids.items():
            if pid != states[cell]["worker_pid"]:
                failures.append(
                    f"lane {cell}: the gate was served by pid {pid}, the body reports {states[cell]['worker_pid']}"
                )

        # hot == cold
        cold = self.cold_digests(export_dir)
        for cell in sorted(worker_digests):
            if worker_digests[cell] != cold:
                differing = sorted(n for n in cold if worker_digests[cell].get(n) != cold[n])
                failures.append(
                    f"lane {cell}: hot live digests differ from a cold load of the export at {differing[:5]}"
                )
        entry["hot_equals_cold"] = all(worker_digests[c] == cold for c in worker_digests)
        print(f"[v{version}] gate ok={ok} hot==cold={entry['hot_equals_cold']} failures={len(failures)}")

        for failure in failures:
            self.fail(f"version {version} ({kind}): {failure}")
        self.report["versions"].append(entry)
        return entry

    def cold_digests(self, export_dir: str) -> dict[str, dict[str, Any]]:
        """A separate engine loads the export from disk; its live digests are the cold truth."""
        from cosmos_framework.posttrain.configs.gr00t_n1d7_droid import make_model_config
        from cosmos_framework.posttrain.model.gr00t.serving.body import Gr00tPolicyBody

        body = Gr00tPolicyBody(make_model_config(checkpoint_path=export_dir))
        try:
            return body._worker_rpc("digest_live_parameters", timeout=body.WEIGHT_RPC_TIMEOUT_S)
        finally:
            body.close()

    # -- scenarios ------------------------------------------------------------------------------------------------
    def set_verification(self, version: int) -> bool:
        """Put every lane on the switch this publication is supposed to run under, before its stream opens.

        The production knob refuses to change mid-stream, so this is the only place the policy is applied; the
        value it returns is what `accept` then requires each lane to report.
        """
        wanted = verification_for(self.args.weight_verification, version)
        if self.verifying is not wanted:
            got = identical(self.rollout.set_verify_weight_values(wanted).all(), "verify_weight_values")
            if got != wanted:
                self.fail(f"v{version}: could not set verify_weight_values={wanted}, lanes report {got}")
        self.verifying = wanted
        return wanted

    def train_steps(self) -> list[dict[str, Any]]:
        """One publication's worth of optimizer steps. Counted once, so a failed publication is not retrained."""
        metrics = []
        for _ in range(self.args.update_weights_interval):
            batch = self.step_batches.pop(0)
            # Per rank: each rank trains its own DP shard, so the reported loss legitimately differs.
            metrics.append({str(c): m for c, m in self.policy.train_sharded(batch).all().items()})
        self.publications_done += 1
        return metrics

    def train_publication(self, version: int) -> None:
        metrics = self.train_steps()
        self.set_verification(version)
        self.policy.set_publication_version(version).all()
        timing = self.sync(version)
        entry = self.accept(version, "train")
        entry["train_metrics"], entry["sync"] = metrics, timing

    def fault_publication(self, version: int) -> None:
        """The same publication as `train_publication`, with the fault armed: it must fail, block, and recover.

        The optimizer steps run ONCE: the recovery re-publishes this version from the same trained weights and
        under the same verification switch, which is what a real retry does.
        """
        metrics = self.train_steps()
        self.set_verification(version)
        entry = self.fault_scenario(version)
        entry["train_metrics"] = metrics

    def fault_scenario(self, version: int) -> dict[str, Any]:
        kind, arg = self.args.fault
        lane = self.args.fault_lane if self.args.fault_lane >= 0 else self.width - 1
        armed = self.rollout.arm_fault(kind, arg, lane).all()
        record: dict[str, Any] = {
            "kind": kind,
            "arg": arg,
            "lane": lane,
            "armed_on": [str(c) for c, a in armed.items() if a],
        }
        if len(record["armed_on"]) != 1:
            self.fail(f"fault: armed on {record['armed_on']}, expected exactly one lane")
        before = self.rollout.probe_state().all()
        self.policy.set_publication_version(version).all()
        try:
            self.sync(version)
        except Exception as error:  # noqa: BLE001 -- the injected failure surfaces through Ray as its own type
            record["error"] = {"type": type(error).__name__, "message": str(error)[-2000:]}
        else:
            self.fail("fault: the injected failure did not make weight_sync raise")
        message = record.get("error", {}).get("message", "")
        expected_text = {
            "bucket": "injected transport fault",
            "digest": "do not hold the staged bytes",
            "tamper": "do not hold the staged bytes",
        }[kind]
        if expected_text not in message:
            self.fail(f"fault: error does not name the injected cause ({expected_text!r}): {message[-300:]}")
        after = self.rollout.probe_state().all()
        record["after"] = {str(c): s for c, s in after.items()}
        for cell, state in after.items():
            # No lane commits: the prepare/commit barrier never ran (prepared is None everywhere). A lane that
            # finished its own stream before the injected one failed keeps its readiness (`pending == version`);
            # that is the framework's contract, and the recovery sync re-marks it. The injected lane itself never
            # became ready.
            if state["weight_version"] != version - 1 or state["prepared"] is not None:
                self.fail(f"fault: lane {cell} moved off the last committed version: {state}")
            if state["pending"] not in (None, version):
                self.fail(f"fault: lane {cell} is ready for {state['pending']}, expected None or {version}")
            if state["replica_index"] == lane and state["pending"] is not None:
                self.fail(f"fault: the injected lane {cell} was marked ready despite the fault: {state}")
            if not state["stream_open"]:
                self.fail(f"fault: lane {cell} closed its stream without a commit")
            if (state["replica_index"] == lane) != state["engine_invalidated"]:
                self.fail(f"fault: lane {cell} invalidated={state['engine_invalidated']}, injected lane is {lane}")
        refusals = self.rollout.refusal(self.obs_list).all()
        record["refusals"] = {str(c): r for c, r in refusals.items()}
        for cell, refusal in refusals.items():
            if not refusal or "refuses to serve" not in refusal:
                self.fail(f"fault: lane {cell} served an unversioned request while blocked: {refusal!r}")
        # recovery: rebuild the invalidated lane only, then a full sync of the trainer's CURRENT version
        rebuilt = self.rollout.rebuild_if_invalidated().all()
        record["rebuilt"] = [str(c) for c, r in rebuilt.items() if r]
        if len(record["rebuilt"]) != 1:
            self.fail(f"fault: rebuilt {record['rebuilt']}, expected exactly the injected lane")
        rebuilt_state = self.rollout.probe_state().all()
        for cell, state in rebuilt_state.items():
            if state["replica_index"] == lane and state["worker_pid"] == before[cell]["worker_pid"]:
                self.fail(f"fault: lane {cell} kept worker pid {state['worker_pid']} across rebuild_engine")
        record["sync"] = self.sync(version)
        record["verified"] = bool(self.verifying)
        self.report["fault"] = record
        entry = self.accept(version, "fault-recovery", expect_pid_change={lane})
        entry["fault"] = record
        return entry

    def perturbation_scenario(self, version: int) -> None:
        """Negative control: a >= 2-ULP nudge to one parameter must show up everywhere; no change must show nowhere."""
        base_version = version - 1
        perturbed = identical(self.policy.perturb_parameter(PERTURB_PARAMETER).all(), "perturbation")
        self.set_verification(version)
        self.policy.set_publication_version(version).all()
        timing = self.sync(version)
        entry = self.accept(version, "perturbation")
        entry["sync"] = timing
        changed = sorted(
            n
            for n in self.digests_by_version[version]
            if self.digests_by_version[version][n] != self.digests_by_version[base_version][n]
        )
        record: dict[str, Any] = {**perturbed, "digests_changed": changed}
        if changed != [PERTURB_PARAMETER]:
            self.fail(f"perturbation: digests changed at {changed[:5]}, expected exactly [{PERTURB_PARAMETER}]")
        if perturbed["changed_after_cast_fraction"] != 1.0:
            self.fail(
                f"perturbation: only {perturbed['changed_after_cast_fraction']:.3f} of the elements survive the bf16 cast"
            )
        if np.array_equal(self.actions_by_version[version], self.actions_by_version[base_version]):
            self.fail("perturbation: the served action did not change")
        # no change: a publication with nothing new must reproduce digests and actions exactly
        self.set_verification(version + 1)
        self.policy.set_publication_version(version + 1).all()
        timing = self.sync(version + 1)
        entry = self.accept(version + 1, "no-change")
        entry["sync"] = timing
        same_digests = self.digests_by_version[version + 1] == self.digests_by_version[version]
        same_actions = np.array_equal(self.actions_by_version[version + 1], self.actions_by_version[version])
        record["no_change"] = {"same_digests": same_digests, "same_actions": same_actions}
        if not (same_digests and same_actions):
            self.fail(f"no-change publication drifted: digests same={same_digests}, actions same={same_actions}")
        self.report["perturbation"] = record

    # -- bench: the full-value verification on/off, on frozen weights ----------------------------------------------
    def bench_scenario(self, first_version: int) -> None:
        """Interleaved publications with the full-value verification on and off, timed per stage, on frozen weights.

        Two controls come first and double as warmup: with verification OFF, one perturbed parameter per bucket must
        land on every worker (the off arm still loads every bucket); with verification ON, one perturbed parameter
        must pass the three-way digest chain again. Then `--bench-warmup` and `--bench-syncs` interleaved on/off
        pairs, the starting arm alternating per round. Every publication is verified OUTSIDE the timed bracket
        (`bench_sync`), and the digest breakdown runs once at the end on a scratch tensor the size of the largest bucket.
        """
        args = self.args
        base = first_version - 1  # the last accepted version: its digests, actions and lane states are the reference
        reference: dict[str, Any] = {
            "digests": self.digests_by_version[base],
            "actions": self.actions_by_version[base],
            "states": self.previous_states,
        }
        record: dict[str, Any] = {
            "warmup": args.bench_warmup,
            "measured": args.bench_syncs,
            "controls": {},
            "samples": [],
        }
        version = first_version

        # One perturbable parameter per bucket (the plan may also carry buffers): the off arm must move all of them.
        parameters = set(identical(self.policy.parameter_names().all(), "parameter names"))
        picks = [
            next(name for name in names if name in parameters)
            for names in identical(self.policy.plan_names().all(), "plan names")
        ]
        perturbed = [identical(self.policy.perturb_parameter(name).all(), f"perturbation of {name}") for name in picks]
        record["controls"][ARM_OFF] = self.bench_sync(
            version, ARM_OFF, reference, expect_changed=picks, perturbed=perturbed
        )
        version += 1
        perturbed = [identical(self.policy.perturb_parameter(PERTURB_PARAMETER).all(), "perturbation")]
        record["controls"][ARM_ON] = self.bench_sync(
            version, ARM_ON, reference, expect_changed=[PERTURB_PARAMETER], perturbed=perturbed
        )
        version += 1

        for index, arm in enumerate(arm_order(args.bench_warmup + args.bench_syncs)):
            sample = self.bench_sync(version, arm, reference, expect_changed=[])
            sample["warmup"] = index < 2 * args.bench_warmup
            record["samples"].append(sample)
            version += 1
        measured = [s for s in record["samples"] if not s["warmup"]]
        record["summary"] = summarize_bench(
            {arm: [s["sync_s"] for s in measured if s["arm"] == arm] for arm in (ARM_ON, ARM_OFF)}
        )
        record["stages"] = {
            arm: medians(columns([s["rows"] for s in measured if s["arm"] == arm])) for arm in (ARM_ON, ARM_OFF)
        }
        record["lane_spread"] = {
            arm: lane_spread([s["totals"] for s in measured if s["arm"] == arm]) for arm in (ARM_ON, ARM_OFF)
        }
        largest = max(row["bytes"] for rows in record["samples"][-1]["lanes"].values() for row in rows)
        record["digest_breakdown"] = {
            str(c): rows for c, rows in self.rollout.digest_breakdown(largest, repeats=3).all().items()
        }
        summary = record["summary"]
        print(
            f"[bench] on {summary[ARM_ON]['median']:.3f}s off {summary[ARM_OFF]['median']:.3f}s "
            f"delta {summary['delta_s']:.3f}s disjoint={summary['ranges_disjoint']} significant={summary['significant']}"
        )
        self.report["bench"] = record

    def bench_sync(
        self,
        version: int,
        arm: str,
        reference: dict[str, Any],
        *,
        expect_changed: list[str],
        perturbed: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """One bench publication: set the arm, sync (the only timed call), drain the stage records, verify outside timing.

        Verification, in a fixed order so both arms see the same pre-sync state next time: trainer publication
        digests == every worker's live digests (== the body's expectation when on; the body must hold none when off),
        exactly `expect_changed` digests changed since the previous publication, the served action changed iff
        something was perturbed, clocks, every lane cleanly committed with its worker pid unchanged and its mapping
        still that worker's (`mapping_pid == worker_pid`).
        """
        flag = arm == ARM_ON
        failures: list[str] = []
        if identical(self.rollout.set_verify_weight_values(flag).all(), "verify_weight_values") != flag:
            failures.append(f"could not set verify_weight_values={flag}")
        self.verifying = flag  # the bench overrides the policy per arm; keep `accept`'s expectation in step
        self.policy.set_publication_version(version).all()
        sync_s = self.sync(version)["seconds"]
        push = self.policy.push_timings().all()
        lanes = self.rollout.sink_timings().all()
        verify = self.rollout.verify_record().all()
        rows, totals = stage_rows(sync_s, push, lanes, verify)
        sample: dict[str, Any] = {
            "version": version,
            "arm": arm,
            "sync_s": sync_s,
            "push": {str(c): t for c, t in push.items()},
            "lanes": {str(c): r for c, r in lanes.items()},
            "verify": {str(c): v for c, v in verify.items()},
            "rows": rows,
            "totals": totals,
        }

        worker = self.rollout.worker_digests().all()
        body = self.rollout.expected_digests().all()
        trainer = identical(self.policy.publication_digests().all(), "trainer publication digests")
        failures.extend(check_bench_digests(trainer, worker, body_by_lane=body if flag else None))
        for cell in sorted(body, key=str):
            if not flag and body[cell]:
                failures.append(f"lane {cell}: {len(body[cell])} body digests recorded with verification off")
            record = verify[cell]
            if record is None or (record["verified"] is None) == flag:
                failures.append(f"lane {cell}: finish record {record} with verification {'on' if flag else 'off'}")
        changed = sorted(name for name in trainer if trainer[name] != reference["digests"].get(name))
        if changed != sorted(expect_changed):
            failures.append(f"{len(changed)} digests changed ({changed[:5]}), expected {sorted(expect_changed)[:5]}")
        reference["digests"] = self.digests_by_version[version] = trainer
        try:
            served = _MeshBody(self.rollout).infer_batch(self.obs_list, seed=self.args.seed, weight_version=version)
        except RuntimeError as error:
            failures.append(str(error))
        else:
            actions = np.stack([a["action"] for a in served])
            same = np.array_equal(actions, reference["actions"])
            if expect_changed and same:
                failures.append("the served action did not change after the perturbation")
            if not expect_changed and not same:
                failures.append("the served action drifted on frozen weights")
            reference["actions"] = self.actions_by_version[version] = actions
        failures.extend(
            check_clocks(
                version,
                self.publications_done,
                self.args.update_weights_interval,
                self.policy.clock().all(),
                self.rollout.weight_version().all(),
            )
        )
        states = self.rollout.probe_state().all()
        for cell, state in states.items():
            before = reference["states"][cell]
            if state["worker_pid"] != before["worker_pid"] or state["mapping_pid"] != state["worker_pid"]:
                failures.append(
                    f"lane {cell}: worker pid {before['worker_pid']} -> {state['worker_pid']}, "
                    f"mapping pid {state['mapping_pid']}"
                )
            if state["stream_open"] or state["engine_invalidated"] or state["pending"] is not None:
                failures.append(f"lane {cell} is not cleanly committed: {state}")
            if state["verify_weight_values"] != flag:
                failures.append(f"lane {cell}: verify_weight_values={state['verify_weight_values']}, expected {flag}")
        if perturbed is not None:
            sample["perturbations"] = perturbed
            failures.extend(
                f"perturbation {p['name']}: only {p['changed_after_cast_fraction']:.3f} of the elements survive the cast"
                for p in perturbed
                if p["changed_after_cast_fraction"] != 1.0
            )
        sample["failures"] = failures
        for failure in failures:
            self.fail(f"bench v{version} ({arm}): {failure}")
        print(f"[v{version}] bench {arm}: sync {sync_s:.3f}s, {len(failures)} failure(s)")
        return sample

    # -- main -----------------------------------------------------------------------------------------------------
    def execute(self) -> dict[str, Any]:
        self.bind()
        try:
            self.prove_topology()
            self.prepare_data()
            self.set_verification(0)
            timing = self.sync(0)
            entry = self.accept(0, "initial")
            entry["sync"] = timing
            fault_at = FAULT_VERSION[self.args.fault[0]] if self.args.fault is not None else None
            version = 0
            for publication in range(1, self.args.publications + 1):
                version = publication
                if publication == fault_at:
                    self.fault_publication(version)
                else:
                    self.train_publication(version)
            if self.args.bench_syncs > 0:
                self.bench_scenario(version + 1)
            elif not self.args.no_perturb:
                self.perturbation_scenario(version + 1)
                version += 2
        finally:
            try:
                rollout = getattr(self, "rollout", None)
                if rollout is not None and rollout._backend is not None:
                    rollout.close().all()
            finally:
                import ray

                ray.shutdown()
        self.report["failures"] = list(self.failures)
        self.report["overall"] = "PASS" if not self.failures else "FAIL"
        return self.report


def _free_cuda() -> None:
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    os.environ[SDPA_MODE_ENV] = args.sdpa_mode  # the driver's own gate models draw through Isaac's DiT too
    report = Run(args).execute()
    path = os.path.join(os.path.abspath(args.out_root), "weight_sync_report.json")
    with open(path, "w") as handle:
        json.dump(report, handle, indent=2, default=str)
    print(f"\nreport: {path}")
    for failure in report["failures"]:
        print(f"FAIL {failure}")
    print("\nOVERALL:", report["overall"])
    if not args.do_assert:
        return 0
    return 0 if report["overall"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
