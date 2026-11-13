# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Versioned finite SFT validation progress; this module does not execute objectives."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

PROTOCOL = "posttrain.sft-validation-continuation.v1"


@dataclass(frozen=True)
class ValidationPause:
    """A committed partial pass, deliberately distinct from ValidationReport."""

    model_step: int
    next_batch: int
    batch_count: int
    reason: str

    def __post_init__(self) -> None:
        if type(self.model_step) is not int or self.model_step < 0:
            raise ValueError("invalid validation pause model step")
        if (
            type(self.next_batch) is not int
            or type(self.batch_count) is not int
            or not 0 < self.next_batch < self.batch_count
            or not isinstance(self.reason, str)
            or not self.reason
        ):
            raise ValueError("invalid partial validation boundary")


def identity_digest(identity: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def continuation_stop_reason(should_stop: Callable[[], str | None] | None) -> str | None:
    """Choose one reason when only some ranks observe the termination signal.

    TerminationWatch converges the Boolean decision, but intentionally retains a
    local explanation (the signaled rank and its peers can return different text).
    The continuation journal must carry one identical, deterministic explanation.
    """
    from cosmos_framework.posttrain.utils.dist import run_rank_converged
    from cosmos_framework.posttrain.utils.rank_state import gather_rank_states

    def poll() -> str | None:
        reason = should_stop() if should_stop is not None else None
        if reason is not None and (not isinstance(reason, str) or not reason):
            raise ValueError("validation continuation stop reason must be a non-empty string or None")
        return reason

    local = run_rank_converged(poll, "SFT validation continuation stop polling")
    reasons = gather_rank_states(local, context="SFT validation continuation stop reasons")
    return next((reason for reason in reasons.values() if reason is not None), None)


def topology_identity(model: Any) -> dict[str, Any]:
    """Bind actual parameter mesh placement, not a guessed launcher world size."""
    import torch.distributed as dist

    net = getattr(model, "net", model)
    layouts = {}
    devices = set()
    for parameter in net.parameters():
        devices.add(parameter.device.type)
        mesh = getattr(parameter, "device_mesh", None)
        if mesh is None:
            continue
        row = {
            "device_type": mesh.device_type,
            "ranks": mesh.mesh.tolist(),
            "names": list(mesh.mesh_dim_names or ()),
            "placements": [str(placement) for placement in parameter.placements],
        }
        layouts[identity_digest(row)] = row
    return {
        "world_size": dist.get_world_size() if dist.is_initialized() else 1,
        "parameter_devices": sorted(devices),
        "layouts": [layouts[key] for key in sorted(layouts)],
    }


def expected_batch(layout: dict[str, Any], rank: int, batch: int) -> list[dict[str, Any]]:
    """The existing finite SFT index mapping, including zero-weight tail slots."""
    start = (batch * layout["world_size"] + rank) * layout["batch_size"]
    return [
        {
            "id": f"{layout['source_fingerprint']}:{index if index < layout['rows'] else 0}",
            "weight": float(index < layout["rows"]),
        }
        for index in range(start, start + layout["batch_size"])
    ]


def validate_progress(
    state: Any,
    identity: dict[str, Any],
    *,
    iteration: int,
    pending_step: int | None,
) -> None:
    """Reject malformed or incompatible progress before RNG or model mutation."""
    if state is None:
        return
    fields = {"protocol", "identity_sha256", "model_step", "next_batch", "layout", "ranks", "rng"}
    if not isinstance(state, dict) or set(state) != fields:
        raise ValueError("invalid SFT validation continuation schema")
    if state["protocol"] != PROTOCOL or state["identity_sha256"] != identity_digest(identity):
        raise ValueError("validation continuation identity differs")
    if identity.get("continuation", {}).get("protocol") != PROTOCOL:
        raise ValueError("validation continuation is not enabled for this checkpoint")
    step, following, layout = state["model_step"], state["next_batch"], state["layout"]
    if type(step) is not int or not 0 <= step <= iteration or type(following) is not int:
        raise ValueError("invalid validation continuation clock")
    if not isinstance(layout, dict) or set(layout) != {
        "rows",
        "batch_size",
        "world_size",
        "batch_count",
        "source_fingerprint",
    }:
        raise ValueError("invalid validation continuation layout")
    if any(
        type(layout[key]) is not int or layout[key] < 1 for key in ("rows", "batch_size", "world_size", "batch_count")
    ):
        raise ValueError("invalid validation continuation dimensions")
    if (
        layout["source_fingerprint"] != identity["source_fingerprint"]
        or layout["batch_size"] != identity["batch_size"]
        or layout["world_size"] != identity["continuation"]["topology"]["world_size"]
        or layout["batch_count"] != math.ceil(layout["rows"] / (layout["batch_size"] * layout["world_size"]))
        or not 0 < following <= layout["batch_count"]
    ):
        raise ValueError("validation continuation layout/identity differs")
    if following < layout["batch_count"] and pending_step != step:
        raise ValueError("incomplete validation continuation requires its pending model step")
    if pending_step is not None and pending_step != step:
        raise ValueError("validation continuation and pending model step differ")
    if not isinstance(state["ranks"], dict) or set(state["ranks"]) != set(range(layout["world_size"])):
        raise ValueError("validation continuation requires every rank")
    if any(type(rank) is not int for rank in state["ranks"]):
        raise ValueError("validation continuation rank keys must be integers")
    for rank, record in state["ranks"].items():
        if not isinstance(record, dict) or set(record) != {
            "numerator",
            "tokens",
            "sequences",
            "duration",
            "batches",
            "batch_metrics",
            "source_state",
        }:
            raise ValueError("invalid validation continuation rank schema")
        for key in ("numerator", "tokens", "sequences", "duration"):
            value = record[key]
            if type(value) is not float or not math.isfinite(value) or key != "numerator" and value < 0:
                raise ValueError(f"invalid validation continuation {key}")
        if not isinstance(record["source_state"], dict):
            raise ValueError("invalid validation continuation source state")
        batches = record["batches"]
        metrics = record["batch_metrics"]
        if (
            not isinstance(batches, list)
            or len(batches) != following
            or not isinstance(metrics, list)
            or len(metrics) != following
        ):
            raise ValueError("validation continuation has missing/duplicate batch indices")
        for index, batch in enumerate(batches):
            expected = expected_batch(layout, rank, index)
            if batch != expected or any(type(row.get("weight")) is not float for row in batch):
                raise ValueError("validation continuation batch IDs/weights differ")
        sums = {"numerator": 0.0, "tokens": 0.0, "sequences": 0.0}
        for row in metrics:
            if not isinstance(row, dict) or set(row) != set(sums):
                raise ValueError("invalid validation batch metric record")
            for key in sums:
                value = row[key]
                if type(value) is not float or not math.isfinite(value) or key != "numerator" and value < 0:
                    raise ValueError("invalid validation batch metric value")
                sums[key] += value
        if any(record[key] != value for key, value in sums.items()):
            raise ValueError("validation continuation totals differ from its ordered batch receipts")
    # This validation uses private generators, so malformed states cannot change
    # the live training RNG before the surrounding isolation context is entered.
    import torch

    from cosmos_framework.posttrain.checkpoint.rng import validate_exact_rng_state

    device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_initialized() else torch.device("cpu")
    validate_exact_rng_state(state["rng"], device=device)


def validate_checkpoint_progress(trainer_state: dict[str, Any], identity: dict[str, Any], iteration: int) -> None:
    """Check a new-mode checkpoint before model/optimizer/RNG restoration."""
    from cosmos_framework.posttrain.roles.trainer.validation.publication import PublicationJournal
    from cosmos_framework.posttrain.roles.trainer.validation.state import ValidationState

    recorded = trainer_state.get("validation")
    enabled = "continuation" in identity
    if enabled and (
        not isinstance(recorded, dict) or type(recorded.get("version")) is not int or recorded["version"] != 3
    ):
        raise ValueError("validation continuation requires its version3 checkpoint; use a fresh run for a new mode")
    if not enabled and isinstance(recorded, dict) and recorded.get("version") == 3:
        raise ValueError("a validation continuation checkpoint cannot resume with continuation disabled")
    state = ValidationState()
    state.defer_restore(trainer_state.get("validation"))
    extra = state.configure(
        identity,
        resumed_from_checkpoint=True,
        iteration=iteration,
        extra_fields_by_version={
            1: frozenset(),
            2: frozenset({"publication"}),
            3: frozenset({"publication", "continuation"}),
        },
    )
    if extra is not None:
        PublicationJournal(state).restore(extra.get("publication"), iteration=iteration)
        validate_progress(
            extra.get("continuation"), identity, iteration=iteration, pending_step=state.progress.pending_step
        )


def preflight_source_progress(
    progress: dict[str, Any] | None,
    source: Any,
    *,
    batch_size: int,
    max_samples: int | None,
) -> None:
    """Validate the source-owned cursor before loading checkpoint model tensors.

    Dataset-local state belongs to the existing source, so its strict loader is
    the authority. Trial restoration is isolated from live source and RNG state.
    No consumed example is read, and no second cursor implementation is introduced.
    """
    from cosmos_framework.posttrain.roles.rollout.validation import distributed_validation_rng
    from cosmos_framework.posttrain.utils.dist import rank_identical, run_rank_converged
    from cosmos_framework.posttrain.utils.entry import resolve
    from cosmos_framework.posttrain.utils.rank_state import select_rank_state

    rank_identical(progress is not None, "SFT validation continuation presence")
    if progress is None:
        return

    def select():
        if progress["layout"] != resolve(source.validation_layout(batch_size, max_samples)):
            raise ValueError("validation continuation finite population differs")
        return select_rank_state(progress["ranks"], on_world_size_mismatch="raise")["source_state"]

    restored = run_rank_converged(select, "SFT validation continuation source admission")
    before = run_rank_converged(lambda: resolve(source.state_dict()), "SFT validation source preflight snapshot")
    with distributed_validation_rng("SFT validation source preflight"):
        try:
            run_rank_converged(
                lambda: resolve(source.load_state_dict(restored)), "SFT validation continuation source preflight"
            )
        finally:
            run_rank_converged(
                lambda: resolve(source.load_state_dict(before)), "SFT validation source preflight restoration"
            )
