# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""CLI, run-attempt, config-dump, and distributed-result helpers for entrypoints."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cosmos_framework.posttrain.control.manifest import Manifest


def cli(description: str, default_config: str) -> tuple[str, list[str], bool]:
    """Parse config, manifest-preflight mode, and the `--`-separated OmegaConf dotlist."""
    config, opts, flags = cli_flags(description, default_config)
    return config, opts, flags["validate_manifest"]


def cli_flags(
    description: str, default_config: str, *, extra_flags: dict[str, str] | None = None
) -> tuple[str, list[str], dict[str, bool]]:
    """`cli` plus entrypoint-specific boolean flags, returned by argparse destination.

    ``extra_flags`` maps ``"--flag-name"`` to its help text; the result holds ``validate_manifest`` and one
    ``flag_name`` key per extra flag.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=default_config)
    parser.add_argument(
        "--validate-manifest",
        dest="validate_manifest",
        action="store_true",
        help="GPU-lessly check the recipe manifest (wiring, fields, transport) and exit 0/1 without running.",
    )
    parser.add_argument(
        "--validate",
        dest="validate_manifest",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    for flag, help_text in (extra_flags or {}).items():
        parser.add_argument(flag, action="store_true", help=help_text)
    args, rest = parser.parse_known_args()
    if "--" in rest:
        rest = rest[rest.index("--") + 1 :]
    flags = {name: bool(value) for name, value in vars(args).items() if name != "config"}
    return args.config, [a for a in rest if "=" in a], flags


def out_dir(config: Any) -> str:
    """Job output dir. With `rl.run_ts` resolved (each entry's `run()` does this before setup), the layout
    descends one more level into the run-attempt dir: <root>/<project>/<group>/<name>/<run_ts>/. Before
    resolution (or for legacy flat jobs) it is the name dir."""
    root = os.environ.get("IMAGINAIRE_OUTPUT_ROOT", "/tmp/imaginaire-output")
    d = os.path.join(root, config.job.project, config.job.group, config.job.name)
    ts = str(getattr(config.rl, "run_ts", "") or "")
    return os.path.join(d, ts) if ts else d


def dump_resolved(config: Any, out: str, *, provenance: bool = False) -> None:
    """Rank 0 writes ``<out>/config_resolved.yaml``; with ``provenance`` it also stamps the attempt.

    The stamp is ``<out>/provenance.<attempt_ts>.json`` plus the same mapping under the YAML's
    top-level ``provenance`` key (see ``utils/provenance.py``); it is best effort and never raises.
    """
    from cosmos_framework.utils.lazy_config import LazyConfig

    if int(os.environ.get("RANK", 0)) == 0:
        os.makedirs(out, exist_ok=True)
        resolved = os.path.join(out, "config_resolved.yaml")
        LazyConfig.save_yaml(config, resolved)
        if provenance:
            from cosmos_framework.posttrain.utils.provenance import stamp_provenance

            stamp_provenance(out, resolved_config=resolved)


@dataclass(frozen=True)
class RunAttempt:
    """Driver-resolved run directory, worker overrides, and concrete resume source."""

    out_dir: str
    opts: tuple[str, ...]
    resume_checkpoint: str | None


def _artifact_remedies(run_dir: str, committed: list[str]) -> str:
    """The operator decisions every refusal over committed model-only artifacts spells out.

    Nothing in the framework continues a run from a model-only artifact, so the remedies are listed because each
    is a different decision the operator has to make.
    """
    from cosmos_framework.posttrain.checkpoint.artifact import CHECKPOINT_ARTIFACTS_DIR

    artifact = os.path.join(run_dir, CHECKPOINT_ARTIFACTS_DIR, committed[-1])
    return (
        "Model-only artifacts are not exact-resume sources. Either "
        f"continue from the artifact as a NEW run under a new job.name (rl.init_from={artifact}, its pool "
        f"advanced by the exact consumed-source count for {run_dir} "
        "before restart, with fresh optimizer state), or pass rl.resume=off "
        "explicitly to mint a fresh attempt directory beside this one and start from step 0 on purpose. Save a "
        "full component plan when exact resume across requeues is required."
    )


def _cold_restart_refusal(requested_resume: str, run_dir: str, committed: list[str]) -> str:
    """Explain why a requested resume that found nothing resumable must not fall back to a cold start here.

    `rl.checkpoint_components=[model]` saves hold no `training_state`, so `resume=auto` finds no checkpoint and
    the old behaviour silently restarted at step 0 inside the same attempt directory; the run then died at its
    first save on the committed artifact it collided with, and a requeue loop repeated that for hours.
    """
    newest = committed[-1]
    return (
        f"rl.resume={requested_resume!r} found no resumable checkpoint in run attempt {run_dir!r}, but that "
        f"attempt already holds {len(committed)} committed model-only checkpoint artifact(s) under "
        f"checkpoint_artifacts/ (newest {newest}); refusing to restart from step 0 inside it, which would "
        f"collide with {newest} at the first save. " + _artifact_remedies(run_dir, committed)
    )


def _stale_resume_refusal(
    resume_checkpoint: str, run_dir: str, committed: list[str], iteration: int, established: str
) -> str:
    """Explain why a full checkpoint older than the newest committed artifact must not be resumed here.

    A full save that is sparser than the artifact schedule (or one that failed after a later artifact landed)
    leaves `resume=auto` a complete `checkpoints/iter_*` behind the newest `checkpoint_artifacts/iter_*`.
    Resuming it inside the same attempt directory replays the steps up to that artifact and dies on it at the
    save there: the collision the cold-restart refusal prevents, one save later.
    """
    from cosmos_framework.posttrain.checkpoint.manifest import iteration_number

    newest = committed[-1]
    return (
        f"rl.resume selected the full checkpoint {resume_checkpoint!r} (iteration {iteration}, from {established}), "
        f"but run attempt {run_dir!r} already holds a newer committed model-only checkpoint artifact under "
        f"checkpoint_artifacts/ (newest {newest}, iteration {iteration_number(newest)}; {len(committed)} committed "
        f"in total); refusing to resume behind it, which would replay the steps up to {newest} and collide with "
        "it at the save there. " + _artifact_remedies(run_dir, committed)
    )


def _unknown_iteration_refusal(resume_checkpoint: str, run_dir: str, committed: list[str], reason: str) -> str:
    """Explain why a resume source whose iteration cannot be established must not proceed over committed artifacts.

    Without the iteration nothing proves the checkpoint is at or beyond the newest artifact, and a legacy
    checkpoint behind it replays into exactly the collision the two refusals above prevent. While committed
    artifacts exist a warning is not enough; the operator has to say what the checkpoint is.
    """
    newest = committed[-1]
    return (
        f"rl.resume={resume_checkpoint!r} has no iter_* segment and its iteration could not be established from "
        f"its trainer state ({reason}), while run attempt {run_dir!r} already holds {len(committed)} committed "
        f"model-only checkpoint artifact(s) under checkpoint_artifacts/ (newest {newest}); refusing to resume a "
        f"checkpoint that cannot be proven to be at or beyond {newest}, since an older one would replay the steps "
        "up to it and collide with it at the save there. " + _artifact_remedies(run_dir, committed)
    )


def _resume_iteration(resume_checkpoint: str) -> tuple[int | None, str]:
    """The iteration a resume source is at and how that was established; ``(None, why not)`` when it cannot be.

    The ``iter_*`` segment is authoritative for a canonical checkpoint location. A legacy path without one is
    asked for its committed trainer state instead (``optimizer_step``/``iter`` in ``trainer.pt``, or at the top
    level of a single-file checkpoint), validated the way the trainer validates it on restore. Rank 0 reads and
    the caller broadcasts the answer, so every rank refuses or proceeds identically.
    """
    from cosmos_framework.posttrain.checkpoint.io import checkpoint_trainer_iteration
    from cosmos_framework.posttrain.checkpoint.manifest import iteration_name, iteration_number

    try:
        return iteration_number(iteration_name(resume_checkpoint)), "its iter_* segment"
    except ValueError:
        pass
    try:
        return checkpoint_trainer_iteration(resume_checkpoint), "its trainer state"
    except ValueError as error:
        return None, str(error)


def pin_run_attempt(config: Any, opts: list[str], name_dir: str) -> RunAttempt:
    """Resolve run timestamp and resume source once, then pin both for every worker.

    Committed ``checkpoint_artifacts/iter_*`` in the adopted attempt directory bound both outcomes of a requested
    resume (anything but ``off``). One that resolves to no checkpoint falls back to a fresh start only when there
    are none (see `_cold_restart_refusal`). A resumable checkpoint is honoured only when its iteration is at or
    after the newest committed artifact (see `_stale_resume_refusal`); that iteration comes from the ``iter_*``
    segment or, for a legacy path without one, from the checkpoint's validated trainer state, and a source whose
    iteration cannot be established is refused as well (see `_unknown_iteration_refusal`). Without committed
    artifacts none of this applies and every resume proceeds as before.
    """
    from cosmos_framework.posttrain.checkpoint import (
        committed_artifact_iterations,
        resolve_external_checkpoint_successor,
        resolve_run_ts,
    )
    from cosmos_framework.posttrain.control.checkpointing import CheckpointLifecycle
    from cosmos_framework.posttrain.utils.dist import (
        broadcast_from_rank0,
        raise_if_any_rank_failed,
        rank_identical,
        run_rank_converged,
    )

    opts_eff = list(opts)
    requested_resume = str(config.rl.resume)
    run_ts = str(config.rl.run_ts)
    if not run_ts:
        run_ts = str(broadcast_from_rank0(lambda: resolve_run_ts(name_dir, requested_resume)))
    if run_ts:
        config.rl.run_ts = run_ts
        opts_eff.append(f"rl.run_ts={run_ts}")
        name_dir = out_dir(config)

    checkpoint_lifecycle = CheckpointLifecycle.from_config(config, config.rl, run_out_dir=name_dir)
    resume_checkpoint = str(checkpoint_lifecycle.resolve_resume(str(config.rl.resume)) or "")
    directory_resume = run_rank_converged(
        lambda: bool(resume_checkpoint and os.path.isdir(resume_checkpoint)),
        "external checkpoint directory selection",
    )
    directory_resume = rank_identical(directory_resume, "external checkpoint directory selection")
    if directory_resume:
        successor_target = os.path.join(
            name_dir,
            "checkpoints",
            os.path.basename(os.path.abspath(resume_checkpoint)),
        )
        needs_successor = run_rank_converged(
            lambda: os.path.realpath(resume_checkpoint) != os.path.realpath(successor_target),
            "external checkpoint successor selection",
        )
        needs_successor = rank_identical(needs_successor, "external checkpoint successor selection")
        if needs_successor:
            import torch.distributed as dist

            successor: str | None = None
            successor_error: BaseException | None = None
            if not dist.is_initialized() or dist.get_rank() == 0:
                try:
                    successor = (
                        resolve_external_checkpoint_successor(resume_checkpoint, successor_target) or resume_checkpoint
                    )
                except BaseException as error:  # noqa: BLE001 -- converge before broadcasting the decision
                    successor_error = error
            raise_if_any_rank_failed(successor_error, "external validation-journal successor resolution")
            resume_checkpoint = str(broadcast_from_rank0(lambda: successor))
    # One filesystem observation, one decision: rank 0 scans the adopted attempt directory for committed
    # model-only artifacts and every rank draws the same conclusion, the way the run timestamp itself is minted
    # above. Both arms consult it: a full checkpoint behind the newest artifact and a cold start over any artifact
    # would each collide with a committed save inside this attempt.
    committed: list[str] = []
    if requested_resume not in ("", "off"):
        committed = list(
            broadcast_from_rank0(
                lambda: committed_artifact_iterations(name_dir),
                context="committed checkpoint artifact scan",
            )
        )
    if resume_checkpoint:
        if committed:
            from cosmos_framework.posttrain.checkpoint.manifest import iteration_number

            # Same discipline as the artifact scan: rank 0 reads the checkpoint's own record of where it is
            # (a legacy path means opening its trainer state) and every rank draws the same conclusion.
            latest_full, established = broadcast_from_rank0(
                lambda: _resume_iteration(resume_checkpoint),
                context="resume checkpoint iteration",
            )
            if latest_full is None:
                raise RuntimeError(_unknown_iteration_refusal(resume_checkpoint, name_dir, committed, established))
            if latest_full < iteration_number(committed[-1]):
                raise RuntimeError(
                    _stale_resume_refusal(resume_checkpoint, name_dir, committed, latest_full, established)
                )
        config.rl.resume = resume_checkpoint
        opts_eff.append(f"rl.resume={resume_checkpoint}")
    else:
        if committed:
            raise RuntimeError(_cold_restart_refusal(requested_resume, name_dir, committed))
        config.rl.resume = "off"
        opts_eff.append("rl.resume=off")
    return RunAttempt(
        out_dir=name_dir,
        opts=tuple(opts_eff),
        resume_checkpoint=resume_checkpoint or None,
    )


def render_manifest_validation(manifest: Manifest) -> tuple[str, bool]:
    """Render the schema-versioned manifest and bind report without owning entrypoint exit policy."""
    from cosmos_framework.posttrain.control.bind import bind_check
    from cosmos_framework.posttrain.control.manifest import manifest_document

    report = bind_check(manifest)
    output = "\n".join((json.dumps({"manifest": manifest_document(manifest)}), report.format()))
    return output, report.ok


def resolve(x: Any) -> Any:
    """A WG method returns a Handle (grid) → .result(); a plain body returns the value (single).

    CAUTION at replicas > 1: on a `CollectiveHandle` this returns replica-0 rank-0 ONLY (its own
    docstring says "debug convenience ... hot path never calls this"). Safe for a value already
    reduced across replicas — e.g. `policy.train(...)`, which `Communicator.all_reduce` means — but
    NOT for per-replica data. Feeding such a result into another WorkerGroup silently trains every
    replica on replica 0's batch. Use `replica_lead_values` + `Scatter` for that.
    """
    return x.result() if hasattr(x, "result") else x


def step_resolver(backend: str) -> Callable[[Any], Any]:
    """The per-step intermediate handler for a loop running on `backend`. Handle flow, not cosmetics.

    Every RL loop in this tree opens with the same line, and it is load-bearing in BOTH directions:

    - ``"local"`` → **resolve eagerly**. Bodies live in this process, but a WorkerGroup method still
      returns a `Handle`; passing that `Handle` on as an argument would NOT auto-resolve, so the next
      role would receive the wrapper instead of its data.
    - ``"ray"`` → **pass the handle through untouched**. The controller then forwards only refs,
      never payloads. Resolving here would pull each intermediate back to the driver and re-ship it
      — and worse, in a colocated FSDP mesh it can DEADLOCK: a peer actor `ray.get()`s a payload
      owned by an actor already blocked inside the `train()` collective.

    So neither branch is an optimization of the other; picking the wrong one is a correctness bug in
    one direction and a hang in the other. Extracted because three loops
    (the GRPO entry, the shared-model FSDP fixture, the parity harness) had written it out
    identically, which is how one of them eventually gets it backwards.

    NOTE this is the SINGLE-value resolver. At `replicas > 1` a per-replica batch must go through
    `replica_lead_values` + `Scatter` instead — see `resolve`'s caution.

    Args:
        backend: ``"local"`` or ``"ray"``. Anything else raises rather than silently defaulting: the
            two behaviors are not interchangeable, so guessing is worse than failing.

    Returns:
        A callable taking one intermediate and returning either its resolved value (local) or the
        intermediate unchanged (ray).

    Raises:
        ValueError: on an unrecognized backend.
    """
    if backend == "local":
        return resolve
    if backend == "ray":
        return lambda x: x
    raise ValueError(f"backend={backend!r}: expected 'local' or 'ray' (the handle flow differs; there is no default)")


def resolve_all(x: Any) -> list:
    """Settle EVERY cell of a mesh result, then return values or one aggregate failure.

    `resolve` awaits replica-0 rank-0 ONLY. When the point of the call IS the await — a write like
    `put_samples`, or a checkpoint `save` — that means a failure on any other replica is neither
    raised nor noticed: the write silently did not happen, and the next `get_samples` under-fills so
    the trainer steps on a short batch without complaint. Awaiting one cell to catch failures is
    exactly as good as awaiting none, once there is more than one cell.

    `CollectiveHandle.all()` is fail-fast: one early failure can return while another already-submitted
    state mutation is still running. Control barriers instead drain `as_completed()` so the caller never
    begins a later phase concurrently with a lagging cell. Failures are raised only after every cell
    settled; successful values retain deterministic coordinate order.

    Args:
        x: a `CollectiveHandle`, a `Handle`, or an already-concrete value.

    Returns:
        Every cell's value (one entry per cell), or `[x]` for a non-handle.
    """
    as_completed = getattr(x, "as_completed", None)
    if callable(as_completed):
        values: dict[Any, Any] = {}
        failures: list[tuple[Any, Exception]] = []
        for coordinate, value in as_completed():
            if isinstance(value, Exception):
                failures.append((coordinate, value))
            else:
                values[coordinate] = value
        if failures:
            details = "; ".join(f"cell {coordinate}: {type(error).__name__}: {error}" for coordinate, error in failures)
            first_error = failures[0][1]
            raise RuntimeError(f"mesh operation failed on {len(failures)} cell(s): {details}") from first_error
        return [values[coordinate] for coordinate in sorted(values)]
    if hasattr(x, "all"):
        return list(x.all().values())
    if hasattr(x, "result"):
        return [x.result()]
    return [x]


def resolve_identical(x: Any, *, operation: str) -> Any:
    """Resolve every cell and reject an empty or rank-divergent result."""
    values = resolve_all(x)
    if not values:
        raise RuntimeError(f"{operation} returned no rank values")
    first = values[0]
    if any(value != first for value in values[1:]):
        raise RuntimeError(f"{operation} returned inconsistent values across ranks: {values!r}")
    return first


def replica_lead_values(x: Any) -> list:
    """Concrete per-replica values behind a mesh result: every replica's rank-0 cell.

    The counterpart to `resolve` for data that is PER-REPLICA. `resolve` collapses a
    `CollectiveHandle` to replica 0, so using it on a scored batch discards every other replica's
    rollouts and — because a plain list is not a handle — makes the next WorkerGroup edge BROADCAST
    replica 0's data instead of routing it. Each replica then trains on identical samples while the
    run looks healthy. That bug has now been found twice, in two different loops, which is why this
    lives beside `resolve` rather than in either caller.

    Rank-0 only: ranks within a replica are SPMD mirrors that each return the full output, so
    including them would double-count every trajectory.

    Args:
        x: a `CollectiveHandle` (one entry per replica), a `Handle`, or an already-concrete value.

    Returns:
        One value per replica, in replica order. Pair with `Scatter` to route them back.
    """
    as_completed = getattr(x, "as_completed", None)
    if callable(as_completed):
        values: dict[Any, Any] = {}
        failures: list[tuple[Any, Exception]] = []
        for coordinate, value in as_completed():
            if isinstance(value, Exception):
                failures.append((coordinate, value))
            else:
                values[coordinate] = value
        if failures:
            details = "; ".join(f"cell {coordinate}: {type(error).__name__}: {error}" for coordinate, error in failures)
            first_error = failures[0][1]
            raise RuntimeError(f"mesh operation failed on {len(failures)} cell(s): {details}") from first_error
        return [value for (replica, rank), value in sorted(values.items()) if rank == 0]
    if hasattr(x, "all"):  # legacy/fake CollectiveHandle without the all-settled API
        return [value for (replica, rank), value in sorted(x.all().items()) if rank == 0]
    if hasattr(x, "result"):
        return [x.result()]
    return [x]
