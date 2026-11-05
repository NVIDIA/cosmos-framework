# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Test-only role bodies for `tests/gr00t_weight_sync_cli.py`: the production bodies plus introspection and fault hooks.

Same pattern as `tests/test_control_grpo_topologies.py::_WeightProbe`: a body ships into its Ray actor, so the
probes must be ON the body class (a driver-side monkeypatch never reaches the actor process). Everything here reads
or perturbs state the production classes already own; nothing changes the weight plane's behaviour unless a fault
is armed, and a fault fires once.

The setup functions are module-level so `functools.partial` can pickle them into `ActorGroup(setup=...)`.
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import asdict
from typing import Any

import torch

from cosmos_framework.posttrain.config.build import REPLICA_INDEX_ENV
from cosmos_framework.posttrain.model.gr00t.serving.body import Gr00tPolicyBody
from cosmos_framework.posttrain.tests.helpers.gr00t_publication_clock import PublicationClockTrainer
from cosmos_framework.posttrain.tests.helpers.weight_sync_bench import TrainerBenchProbe, bucket_nbytes

SDPA_MODE_ENV = "GR00T_DIT_SDPA_MODE"
#: The tamper value: no trained weight is ever this, so a tampered tensor is guaranteed to differ from its digest.
TAMPER_VALUE = -12345.0
#: Names of the fault kinds `--fault` accepts.
FAULT_KINDS = ("bucket", "digest", "tamper")


def parse_fault(spec: str | None) -> tuple[str, str] | None:
    """``"bucket:3"`` / ``"digest:NAME"`` / ``"tamper:NAME"`` -> ``(kind, arg)``; None passes through."""
    if spec is None:
        return None
    kind, sep, arg = spec.partition(":")
    if not sep or kind not in FAULT_KINDS or not arg:
        raise ValueError(
            f"--fault {spec!r}: expected one of {FAULT_KINDS} followed by ':' and a bucket index or a name"
        )
    if kind == "bucket" and not arg.isdigit():
        raise ValueError(f"--fault {spec!r}: bucket faults take a bucket index")
    return kind, arg


class RolloutProbe:
    """Mixin over `Gr00tPolicyBody`: state read-back, per-bucket timings, one-shot fault injection."""

    replica_index: int = -1
    _fault: tuple[str, str] | None = None
    _bucket_index: int = -1

    # -- state ---------------------------------------------------------------------------------------------------
    def probe_state(self) -> dict[str, Any]:
        backend = getattr(self, "_weight_backend", None)
        used = total = None
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            used = total - free
        return {
            "replica_index": self.replica_index,
            "weight_version": self.weight_version(),
            "pending": self._pending_weight_version,
            "prepared": self._prepared_weight_version,
            "stream_open": self._stream_open,
            "engine_invalidated": self.engine_invalidated,
            "worker_pid": self.worker_device.get("pid"),
            "worker_sdpa_mode": self.worker_device.get("sdpa_mode"),
            "body_sdpa_mode": os.environ.get(SDPA_MODE_ENV),
            "mapping_pid": (self._mapping or {}).get("pid"),
            "update_epoch": self._update_epoch,
            "mover_kind": type(getattr(backend, "mover", backend)).__name__,
            "transfer_dtype": self.transfer_dtype,
            "body_max_memory_allocated": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
            "device_used_bytes": used,
            "device_total_bytes": total,
            "verify_weight_values": bool(getattr(self, "verify_weight_values", False)),
            "armed_fault": self._fault,
        }

    def refusal(self, obs_list: list[dict]) -> str | None:
        """The error message an unversioned inference request gets, or None if it was served."""
        try:
            self.infer_batch(obs_list, weight_version=None)
        except RuntimeError as error:
            return str(error)
        return None

    def expected_digests(self) -> dict[str, dict[str, Any]]:
        return dict(self._expected_digests)

    def worker_digests(self) -> dict[str, dict[str, Any]]:
        return self._worker_rpc("digest_live_parameters", timeout=self.WEIGHT_RPC_TIMEOUT_S)

    def set_verify_weight_values(self, flag: bool) -> bool:
        """Flip the full-value verification between streams -- never inside one: bucket 0 clears the expectation
        the end-of-stream check needs, so a mid-stream flip would fail that check."""
        if self._stream_open:
            raise RuntimeError("verify_weight_values cannot change while a weight stream is open")
        self.verify_weight_values = bool(flag)
        return self.verify_weight_values

    def sink_timings(self) -> list[dict[str, Any]]:
        """Per-bucket rows of the last stream: the body's `BucketSinkTiming` (copy/sync/compare/digest), the backend's
        fetch/sink wall clock, and the bucket's names and raw bytes from the plan."""
        be = self._weight_backend
        plan = be.plan()
        applied = {t.bucket: t for t in getattr(be, "apply_timings", [])}
        rows: list[dict[str, Any]] = []
        for i, timing in enumerate(self.bucket_timings):
            row: dict[str, Any] = {
                "bucket": i,
                "names": len(plan[i]),
                "bytes": bucket_nbytes(plan[i]),
                **asdict(timing),
            }
            if i in applied:
                row["fetch_s"] = applied[i].fetch_s
                row["sink_s"] = applied[i].sink_s
            rows.append(row)
        return rows

    def verify_record(self) -> dict[str, Any] | None:
        """The body's `VerifyTiming` of the last stream (the `finish_weight_update` round trip; `verified` is None when
        verification was off); None before the first stream."""
        timing = self.verify_timing
        return None if timing is None else asdict(timing)

    def digest_breakdown(self, nbytes: int, repeats: int = 3) -> list[dict[str, float]]:
        """OUTSIDE timing: the three steps of `tensor_digest_metadata` over ``nbytes`` of scratch bytes on this GPU
        -- `.cpu()`, `.numpy().tobytes()`, `sha256` -- each timed separately, ``repeats`` times (their throughput does
        not depend on the content). Every repeat pays fresh host allocations, as every in-sync digest does.
        Contiguous bytes only: the per-tensor call overhead of a real bucket (one D2H sync per tensor) is not
        covered."""
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        source = torch.empty(int(nbytes), dtype=torch.uint8, device=device)
        n = source.numel()
        rows: list[dict[str, float]] = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            host = source.cpu()
            t1 = time.perf_counter()
            raw = host.numpy().tobytes()
            t2 = time.perf_counter()
            hashlib.sha256(raw).hexdigest()
            t3 = time.perf_counter()
            rows.append(
                {
                    "bytes": float(n),
                    "d2h_s": t1 - t0,
                    "tobytes_s": t2 - t1,
                    "sha256_s": t3 - t2,
                    "d2h_gbps": n / max(t1 - t0, 1e-12) / 1e9,
                    "tobytes_gbps": n / max(t2 - t1, 1e-12) / 1e9,
                    "sha256_gbps": n / max(t3 - t2, 1e-12) / 1e9,
                }
            )
            del host, raw
        return rows

    def rebuild_if_invalidated(self) -> bool:
        """Recover this lane only if a failed stream invalidated it (a mirrored call touches nothing else)."""
        if not self.engine_invalidated:
            return False
        self.rebuild_engine()
        return True

    # -- faults ----------------------------------------------------------------------------------------------------
    def arm_fault(self, kind: str, arg: str, lane: int) -> bool:
        """Arm a one-shot fault on the lane whose replica index is ``lane``; every other lane disarms."""
        self._fault = (kind, arg) if lane == self.replica_index else None
        return self._fault is not None

    def _coverage_reset(self, i: int) -> None:
        self._bucket_index = i
        super()._coverage_reset(i)

    def _engine_sink(self, state: dict) -> set[str]:
        if self._fault is not None and self._fault[0] == "bucket" and int(self._fault[1]) == self._bucket_index:
            self._fault = None
            raise RuntimeError(f"injected transport fault at bucket {self._bucket_index} on lane {self.replica_index}")
        return super()._engine_sink(state)

    def mark_weight_version_ready(self, version: int) -> None:
        fault, self._fault = self._fault, None
        if fault is not None and fault[0] == "digest":
            name = fault[1]
            if name not in self._expected_digests:
                raise KeyError(f"injected digest fault names {name!r}, which is not a live parameter")
            self._expected_digests[name] = {**self._expected_digests[name], "sha256": "0" * 64}
        elif fault is not None and fault[0] == "tamper":
            # Overwrite one live tensor through the mapping AFTER its bucket's witness ran: no digest update. Only
            # the worker's independent re-read (`finish_weight_update` with verification on) can see it.
            name = fault[1]
            live = self._live_params or {}
            if name not in live:
                raise KeyError(f"injected tamper fault names {name!r}, which is not a live parameter")
            with torch.no_grad():
                live[name].fill_(TAMPER_VALUE)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        elif fault is not None:
            self._fault = fault  # a bucket fault past its bucket: leave it armed and report it
        super().mark_weight_version_ready(version)


class TrainerProbe(TrainerBenchProbe):
    """GR00T additions over `TrainerBenchProbe`: DP-sharded training from a driver batch, the clocks, the precision
    environment and the master-to-wire gap."""

    dp_coords: tuple[int, int] = (0, 1)

    def train_sharded(self, batch: list) -> dict[str, Any]:
        """Train on this rank's contiguous DP shard of ``batch`` (the registry's `select_model_dp`)."""
        from cosmos_framework.posttrain.workers.dispatch import select_model_dp

        rank, world = self.dp_coords
        return self.train(select_model_dp(list(batch), dp_rank=rank, dp_world=world))

    def clock(self) -> dict[str, int]:
        return {
            "iteration": self.iteration(),
            "publication_version": self.publication_version(),
            "optimizer_step": self.optimizer_step,
        }

    def trainer_env(self) -> dict[str, Any]:
        import dataclasses

        backend = getattr(self, "_weight_backend", None)
        return {
            "sdpa_mode": os.environ.get(SDPA_MODE_ENV),
            "dp_coords": list(self.dp_coords),
            "precision": dataclasses.asdict(self.model.precision),
            "transfer_dtype": self.transfer_dtype,
            "mover_kind": type(getattr(backend, "mover", backend)).__name__,
            "max_memory_allocated": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
        }

    def _full_head_state(self) -> dict[str, torch.Tensor]:
        from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

        return dict(
            get_model_state_dict(self.model.net, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
        )

    def master_to_wire_gap(self) -> dict[str, Any] | None:
        """Data E: ``|master - wire(master)|`` over the head, where wire is the preset's transfer dtype.

        The full state is gathered to rank 0 only (`cpu_offload=True`); every other rank answers None.
        """
        wire = getattr(torch, self.transfer_dtype)
        head = self._full_head_state()
        if not head:
            return None
        diffs = []
        for tensor in head.values():
            master = tensor.detach().float()
            diffs.append((master - master.to(wire).float()).abs().flatten())
        gap = torch.cat(diffs)
        return {
            "master_dtype": str(next(iter(head.values())).dtype),
            "transfer_dtype": self.transfer_dtype,
            "count": int(gap.numel()),
            "max": float(gap.max()),
            "p99": float(torch.quantile(gap[:: max(1, gap.numel() // 2_000_000)], 0.99)),
            "nonzero_fraction": float((gap > 0).float().mean()),
        }


class ProbeRollout(RolloutProbe, Gr00tPolicyBody):
    """`Gr00tPolicyBody` + probes."""


class ProbeTrainer(TrainerProbe, PublicationClockTrainer):
    """The driver-clocked trainer + probes (`optimizer_step` stays the durable optimizer counter)."""


def setup_rollout(*, checkpoint: str, precision_mode: str, sdpa_mode: str) -> dict[str, Any]:
    """Actor setup for the rollout WG: one hot engine per cell, wire dtype from the SAME preset the trainer uses."""
    os.environ[SDPA_MODE_ENV] = sdpa_mode
    from cosmos_framework.posttrain.configs.gr00t_n1d7_droid import make_model_config
    from cosmos_framework.posttrain.model.gr00t.policy import PRECISION_PRESETS

    preset = PRECISION_PRESETS[precision_mode]
    body = ProbeRollout(
        make_model_config(checkpoint_path=checkpoint), transfer_dtype=preset.transfer, verify_weight_values=True
    )
    body.replica_index = int(os.environ[REPLICA_INDEX_ENV])
    return {"rollout": body}


def setup_policy(*, checkpoint: str, precision_mode: str, sdpa_mode: str, out_root: str) -> dict[str, Any]:
    """Actor setup for the policy WG: the FSDP trainer over the injected mesh, `fail_stop`, DCP checkpoints."""
    os.environ[SDPA_MODE_ENV] = sdpa_mode
    import torch.distributed as dist

    from cosmos_framework.posttrain.config.build import build_optimizer
    from cosmos_framework.posttrain.configs.gr00t_n1d7_droid import (
        make_model_config,
        make_objective,
        make_optimizer_config,
    )
    from cosmos_framework.posttrain.model.gr00t.policy import PRECISION_PRESETS, Gr00tN1d7TrainPolicy
    from cosmos_framework.posttrain.workers.ray import current_pdims

    preset = PRECISION_PRESETS[precision_mode]
    config = make_model_config(checkpoint_path=checkpoint)
    rank, world = (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)
    cfg_dir = os.path.join(out_root, "cfg", f"policy_rank{rank}")  # Isaac writes final_model_config.json per build
    os.makedirs(cfg_dir, exist_ok=True)
    policy = Gr00tN1d7TrainPolicy(
        config, device="cuda", parallel_dims=current_pdims(), save_cfg_dir=cfg_dir, precision=preset
    )
    optimizer_config = make_optimizer_config()
    trainer = ProbeTrainer(
        policy,
        make_objective(config, policy.gr00t.collator),
        build_optimizer(policy, optimizer_config),
        grad_accum=1,
        n_mini=1,
        max_grad_norm=optimizer_config.grad_clip_norm,
        nonfinite_update="fail_stop",
        checkpoint_format="dcp",
        transfer_dtype=policy.precision.transfer,
    )
    trainer.dp_coords = (rank, world)
    return {"policy": trainer}
