# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Console and W&B transports over one canonical metric frame.

cosmos-rl exposes the two reporting channels as callbacks inside the trainer
(`IterationLoggerCallback` for the console, `BaseWandBCallback` for W&B, both rank-0-
gated). We keep the same two channels and the same W&B conventions (resolved W&B
project plus `job.group/name`, config attached to the run, stable online run id +
`resume="allow"`), but the
reporters live on the CONTROL PLANE (the entry loop), not in the trainer body: on the Ray
backend the trainer callbacks run inside K worker processes — initializing W&B there
would mean K runs or a shared-run-id dance. The loop already receives the all-reduced
metrics every step and runs at exactly one point per job (the driver; rank 0 under
torchrun via the factory's gate), so it is the single writer.

Usage (the entry loop, where ``step`` is the public committed model step):
    reporters = make_reporters(config, out_dir)      # [] on non-rank-0
    for r in reporters: r.start()
    ... for each step without validation: [r.report(metrics, step=step) for r in reporters]
    ... for a train+validation step:
        [r.report({**train_metrics, **val_metrics}, step=step) for r in reporters]
    finally: [r.finish() for r in reporters]

Channel selection: the console reporter is always on; W&B is added unless
`job.wandb_mode == "disabled"` (the default; "offline" is passed through
to `wandb.init` for air-gapped runs) or wandb is not importable.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import math
import os
import uuid
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from cosmos_framework.utils.launch_defaults import get_wandb_entity, get_wandb_project
from cosmos_framework.posttrain.schema.metrics import MetricScalar, validate_public_metric_keys
from cosmos_framework.posttrain.utils.batch_stats import RewardSamplePreview
from cosmos_framework.posttrain.utils.dist import raise_if_any_rank_failed
from cosmos_framework.posttrain.utils.log import info0, logger
from cosmos_framework.posttrain.utils.scalar import as_report_scalar, as_scalar

# The headline metrics echoed on the console line (full dict still goes to W&B).
_CONSOLE_KEYS = (
    "train/loss_avg",
    "train/reward_mean",
    "train/reward_std",
    "train/dpo_accuracy",
    "val/loss_avg",
    "val/reward_avg",
    "val/reward_std",
)


@dataclasses.dataclass(frozen=True)
class WandbDestination:
    """Resolved public run coordinates; credentials never belong to serialized config."""

    entity: str
    project: str
    group: str
    name: str
    mode: str


def resolve_wandb_destination(job: Any) -> WandbDestination:
    """Share environment-over-config precedence between submission and the reporter."""
    return WandbDestination(
        entity=get_wandb_entity(),
        project=get_wandb_project(str(job.project)),
        group=os.environ.get("WANDB_RUN_GROUP") or str(job.group),
        name=os.environ.get("WANDB_NAME") or str(job.name),
        mode=os.environ.get("WANDB_MODE") or str(job.wandb_mode),
    )


class Reporter:
    """One staged report channel with exactly one commit per logical model step.

    ``report`` is the single-payload convenience path.  A step that also runs validation uses
    ``stage`` repeatedly and ``commit`` once, so W&B receives one history row containing both the
    train and validation namespaces.  Payloads are already-finalized schema-v2 metrics: Reporter
    validates canonical keys but never renames or aliases them.
    """

    _pending_step: int | None
    _pending_metrics: dict[str, Any]
    _last_committed_step: int | None

    def __init__(self) -> None:
        self._pending_step = None
        self._pending_metrics = {}
        self._last_committed_step = None

    def start(self) -> None: ...

    @staticmethod
    def _validate_step(step: int) -> None:
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError(f"report step must be a non-negative integer, got {step!r}")

    @property
    def last_committed_step(self) -> int | None:
        """The latest step emitted by this reporter in the current process."""
        return self._last_committed_step

    def stage(self, metrics: dict[str, Any], step: int) -> None:
        """Stage one disjoint canonical payload for ``step`` without emitting it."""
        self._validate_step(step)
        if not metrics:
            raise ValueError(f"cannot stage an empty metric payload at step {step}")
        validate_metric_frame(metrics)
        if self._last_committed_step is not None and step <= self._last_committed_step:
            raise ValueError(
                f"report step {step} is not newer than committed step {self._last_committed_step}; "
                "one logical step may be committed only once"
            )
        if self._pending_step is None:
            self._pending_step = step
        elif self._pending_step != step:
            raise ValueError(
                f"cannot stage step {step} while step {self._pending_step} is pending; commit the pending frame first"
            )
        duplicates = set(self._pending_metrics).intersection(metrics)
        if duplicates:
            raise ValueError(f"step {step} staged duplicate metric keys: {sorted(duplicates)!r}")
        self._pending_metrics.update(metrics)

    def commit(self, step: int, metrics: dict[str, Any] | None = None) -> None:
        """Optionally stage one final payload, then emit exactly one row for ``step``."""
        self._validate_step(step)
        if metrics is not None:
            self.stage(metrics, step)
        if self._pending_step is None:
            raise ValueError(f"cannot commit step {step}: no metrics are staged")
        if self._pending_step != step:
            raise ValueError(f"cannot commit step {step}: pending metrics belong to step {self._pending_step}")
        payload = dict(self._pending_metrics)
        # Consume the logical frame before touching a non-transactional transport. If ``_emit``
        # raises after a partial external side effect, finish/recovery must not retry an ambiguous
        # row. Validation completion already follows the same durable-checkpoint-before-report order.
        self._pending_metrics.clear()
        self._pending_step = None
        self._last_committed_step = step
        self._emit(payload, step)

    def report(self, metrics: dict[str, Any], step: int) -> None:
        """Stage and immediately commit one complete canonical metric frame."""
        self.commit(step, metrics)

    def _emit(self, metrics: dict[str, Any], step: int) -> None: ...

    def finish(self) -> None:
        # Staging remains a general API for callers that assemble a frame in one process. Validation
        # coordinators build their complete frame before touching Reporter, so no retryable held-out
        # phase can arrive here as a partial staged row.
        if self._pending_step is not None:
            logger.warning(f"reporter finishing with staged step {self._pending_step}; committing partial metric frame")
            self.commit(self._pending_step)


def validate_metric_frame(metrics: dict[str, Any]) -> None:
    """Validate one finalized public frame before any reporting transport is attempted."""
    validate_public_metric_keys(metrics)
    for key, value in metrics.items():
        if not isinstance(value, (int, float, str, bool)):
            raise TypeError(f"public metric {key!r} must be a finalized scalar, got {type(value).__name__}")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"public metric {key!r} must be finite, got {value!r}")


def reportable_metrics(metrics: dict[str, Any]) -> dict[str, MetricScalar]:
    """Detach scalar metrics into a checkpoint- and transport-safe frame."""
    frame: dict[str, MetricScalar] = {}
    for key, value in metrics.items():
        scalar = as_report_scalar(value)
        if scalar is not None:
            frame[key] = scalar
        elif isinstance(value, str):
            frame[key] = value
    return frame


def merge_training_metric_window(
    reports: list[dict[str, Any]],
    *,
    sum_keys: frozenset[str] = frozenset(),
    max_keys: frozenset[str] = frozenset(),
    weighted_mean_by: Mapping[str, str] = MappingProxyType({}),
    last_keys: frozenset[str] = frozenset(
        {
            "train/learning_rate",
            "train/grad_norm",
            "train/grad_clipped",
            "train/grad_clip_coef",
            "optim/update_outcome",
            "optim/attempt_id",
            "optim/nonfinite_skips_total",
            "optim/nonfinite_skips_consecutive",
            "optim/skip_reason",
        }
    ),
) -> dict[str, Any]:
    """Merge an optimizer window: average values, sum counts/weights, max extrema, keep step signals."""
    if not reports:
        raise ValueError("cannot merge an empty training metric window")
    weighted_mean_by = dict(weighted_mean_by)
    if any(not isinstance(key, str) or not key for key in weighted_mean_by):
        raise ValueError("weighted metric keys must be non-empty strings")
    if any(not isinstance(key, str) or not key for key in weighted_mean_by.values()):
        raise ValueError("metric weight keys must be non-empty strings")
    weight_keys = frozenset(weighted_mean_by.values())
    effective_sum_keys = sum_keys | weight_keys
    conflicting_max_keys = max_keys & (effective_sum_keys | frozenset(weighted_mean_by))
    if conflicting_max_keys:
        raise ValueError(
            "max_keys cannot also be summed or weighted (they must be disjoint from sum_keys, the weighted "
            f"metrics and their weights): {sorted(conflicting_max_keys)!r}"
        )
    merged = dict(reports[-1])
    numeric_keys = {
        key
        for report in reports
        for key, value in report.items()
        if key.startswith("train/") and (isinstance(value, (int, float)) or hasattr(value, "item"))
    }
    for key in numeric_keys:
        raw_values = [report[key] for report in reports if key in report]
        values = [float(value) for value in raw_values]
        integer_valued = all(isinstance(value, int) and not isinstance(value, bool) for value in raw_values)
        if key in max_keys:
            largest = max(values)
            merged[key] = int(largest) if integer_valued else largest
        elif key in effective_sum_keys:
            total = sum(values)
            merged[key] = int(total) if integer_valued else total
        elif key in weighted_mean_by:
            weight_key = weighted_mean_by[key]
            contributors = [report for report in reports if key in report]
            has_weights = [weight_key in report for report in contributors]
            if any(has_weights) and not all(has_weights):
                raise ValueError(f"metric {key!r} requires {weight_key!r} in every contributing report")
            if not any(has_weights):
                merged[key] = sum(values) / len(values)
            else:
                weights = [float(report[weight_key]) for report in contributors]
                if any(not math.isfinite(weight) or weight < 0.0 for weight in weights):
                    raise ValueError(f"metric weights in {weight_key!r} must be finite and non-negative")
                if any(weight == 0.0 and value != 0.0 for value, weight in zip(values, weights, strict=True)):
                    raise ValueError(f"metric {key!r} must be zero whenever {weight_key!r} is zero")
                total_weight = sum(weights)
                if total_weight == 0.0:
                    if any(value != 0.0 for value in values):
                        raise ValueError(f"metric {key!r} must be zero when {weight_key!r} sums to zero")
                    merged[key] = 0.0
                else:
                    merged[key] = (
                        sum(value * weight for value, weight in zip(values, weights, strict=True)) / total_weight
                    )
        elif key not in last_keys:
            merged[key] = sum(values) / len(values)
    return merged


def log_sample_previews(previews: tuple[RewardSamplePreview, ...], step: int) -> None:
    """Log representative scored completions under a stable comparison prefix."""
    for preview in previews:
        info0(
            f"[sample] step={step} reward={preview.reward:.1f} gold_tail={preview.reference_tail!r} "
            f"completion={preview.completion!r}"
        )


def emit_best_effort(
    reporters: list[Reporter],
    metrics: dict[str, MetricScalar],
    step: int,
    *,
    failed_reporters: set[int],
) -> None:
    """Attempt each transport once; warn and trip that reporter's circuit breaker on failure."""
    validate_metric_frame(metrics)
    for reporter in reporters:
        if id(reporter) in failed_reporters:
            continue
        try:
            reporter.report(metrics, step=step)
        except Exception as error:  # noqa: BLE001 -- transports are non-transactional and best effort
            failed_reporters.add(id(reporter))
            logger.warning(
                "metric reporter %s failed at step %d (%s); disabling it for this run",
                type(reporter).__name__,
                step,
                error,
            )


def emit_metric_frame(
    reporters: list[Reporter],
    metrics: dict[str, MetricScalar],
    step: int,
    *,
    failed_reporters: set[int],
) -> None:
    """Converge schema failures before attempting rank-zero best-effort transports."""
    schema_error: BaseException | None = None
    try:
        validate_metric_frame(metrics)
    except BaseException as error:  # noqa: BLE001 -- every local rank must reject the same bad frame
        schema_error = error
    raise_if_any_rank_failed(schema_error, "public metric frame validation")
    emit_best_effort(reporters, metrics, step, failed_reporters=failed_reporters)


def start_best_effort(reporters: list[Reporter], *, failed_reporters: set[int]) -> None:
    """Start every transport independently; one broken sink cannot abort training."""
    for reporter in reporters:
        try:
            reporter.start()
        except Exception as error:  # noqa: BLE001 -- reporting transport is explicitly noncritical
            failed_reporters.add(id(reporter))
            logger.warning(
                "metric reporter %s failed to start (%s); disabling it for this run",
                type(reporter).__name__,
                error,
            )


def finish_best_effort(reporters: list[Reporter], *, failed_reporters: set[int]) -> None:
    """Finish healthy transports without replacing a training/validation outcome."""
    for reporter in reporters:
        if id(reporter) in failed_reporters:
            logger.warning("skipping failed metric reporter finish to preserve at-most-once emission")
            continue
        try:
            reporter.finish()
        except Exception as error:  # noqa: BLE001 -- shutdown transport cannot mask training state
            failed_reporters.add(id(reporter))
            logger.warning(
                "metric reporter %s failed to finish (%s); continuing shutdown",
                type(reporter).__name__,
                error,
            )


class ConsoleReporter(Reporter):
    """Console channel: one INFO line per step with the headline scalars.

    `total_steps` renders progress as `step  246/500` (1-based completed count, matching
    the iter_* checkpoint numbering). It is a CONSTRUCTOR-OR-LATER seam: fixed-length
    algorithms (DPO/SFT) pass it up front via `make_reporters(..., total_steps=rl.max_steps)`;
    algorithms that only learn the total at runtime (GRPO: total depends on the live
    replica count probed each round) construct with None and assign `total_steps` once
    known — until then lines fall back to the bare step number."""

    total_steps: int | None
    headline_keys: tuple[str, ...]

    def __init__(
        self,
        total_steps: int | None = None,
        headline_keys: tuple[str, ...] = _CONSOLE_KEYS,
    ) -> None:
        super().__init__()
        self.total_steps = total_steps
        self.headline_keys = headline_keys

    def _emit(self, metrics: dict[str, Any], step: int) -> None:
        # The caller supplies its committed step clock. Publication versions may be separate
        # metrics; the transport never translates between those clocks.
        parts = [f"step {step:4d}/{self.total_steps}" if self.total_steps else f"step {step:5d}"]
        for k in self.headline_keys:
            if k in metrics:
                value = as_scalar(metrics[k], allow_bool=True)
                if value is not None:
                    # Keep the complete canonical stage/key label; train and val loss can coexist.
                    # A magnitude below 1e-3 (a learning rate of 1e-5) prints as +0.0000 at four decimals; show it in
                    # scientific notation instead (review R121B-01). Zero stays +0.0000.
                    parts.append(f"{k} {value:+.4f}" if value == 0 or abs(value) >= 1e-3 else f"{k} {value:+.3e}")
        logger.info("  ".join(parts))


class JsonlReporter(Reporter):
    """File channel: one JSON object per committed logical step appended to ``path``.

    Row = ``{"step": n, "time": <iso>, **frame}``; a step that also validates lands as ONE row because the staging
    protocol merges its train and validation payloads before the commit. Bounded by construction (one line per step,
    no per-iteration files), it is the run directory's record of what the console line only summarises: with W&B off,
    ``train/grad_norm``, ``train/learning_rate``, ``train/valid_token_count`` and the step time were computed every step
    and recorded nowhere (review R125-B-03), so whether the clip was active on a run could not be read after the fact.

    Attempts: the run directory is reused across relaunches, and a relaunch resumes from the last checkpoint, which may
    be older than the last committed row (a crash past the last save). ``keep_through`` is the step the attempt resumes
    at: ``start`` keeps the rows with ``step <= keep_through`` and drops the rest, which the attempt re-emits, so the
    file holds one row per step; ``-1`` (a fresh attempt) drops every row. The kept rows' clip record seeds
    ``prior_clip_counts`` = (clipped updates, counted updates) so ``train/grad_clip_fraction`` stays the share of the
    run's updates rather than the attempt's (review R134-05: the append-only file held step 3 twice after a relaunch
    from the step-2 checkpoint and the fraction restarted).
    """

    path: str
    keep_through: int
    prior_clip_counts: tuple[int, int]

    def __init__(self, path: str, *, keep_through: int = -1) -> None:
        super().__init__()
        if not isinstance(path, str) or not path:
            raise ValueError("JsonlReporter requires a non-empty file path")
        if isinstance(keep_through, bool) or not isinstance(keep_through, int) or keep_through < -1:
            raise ValueError(f"JsonlReporter keep_through must be an integer >= -1, got {keep_through!r}")
        self.path = path
        self.keep_through = keep_through
        self.prior_clip_counts = (0, 0)

    def start(self) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.path)) or ".", exist_ok=True)
        if not os.path.exists(self.path):
            return
        kept: list[str] = []
        clipped = counted = 0
        with open(self.path, encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)  # a corrupt line fails the start: the channel is marked failed, the run goes on
                step = row.get("step") if isinstance(row, dict) else None
                if isinstance(step, bool) or not isinstance(step, int):
                    raise ValueError(f"{self.path}:{number}: metrics row without an integer step")
                if step > self.keep_through:
                    continue
                kept.append(line if line.endswith("\n") else line + "\n")
                flag = row.get("train/grad_clipped")
                if isinstance(flag, (int, float)) and not isinstance(flag, bool):
                    counted += 1
                    clipped += int(flag > 0)
        staged = f"{self.path}.tmp"
        with open(staged, "w", encoding="utf-8") as handle:
            handle.writelines(kept)
        os.replace(staged, self.path)
        self.prior_clip_counts = (clipped, counted)

    def _emit(self, metrics: dict[str, Any], step: int) -> None:
        row: dict[str, Any] = {"step": step, "time": datetime.datetime.now().astimezone().isoformat(timespec="seconds")}
        for key in sorted(metrics):
            value = as_scalar(metrics[key], allow_bool=True)
            row[key] = metrics[key] if value is None else value
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")


class WandbReporter(Reporter):
    """W&B channel. Init follows cosmos-rl's `init_wandb`: project from a non-empty
    `WANDB_PROJECT`, otherwise by resolving `config.job.project` (`cosmos3_vfm` maps to
    `cosmos3_generator`); group/name from a non-empty `WANDB_RUN_GROUP`/`WANDB_NAME`,
    otherwise from `config.job`. All four W&B coordinates take the same
    env-overrides-config precedence, so a launcher can place a run in a study cohort
    without editing the recipe -- before this, only project and mode honored the env and
    a run launched with `WANDB_RUN_GROUP` set still landed under the recipe's group.
    The run id is generated once and persisted to `<out_dir>/wandb_run_id.txt`
    with `resume="allow"` — a preempted/requeued job re-reads the file and continues the
    SAME curve. (The id is random, NOT derived from the out_dir: a deterministic id
    resurrects deleted-run tombstones on the server, and `wandb.init` then hangs until its
    timeout — observed 2026-07-06 after deleting a bad run and relaunching into the same
    out_dir.) `WANDB_MODE` env overrides `job.wandb_mode` ("online"/"offline"/"disabled").
    Failures degrade to a warning, never kill training."""

    _config: Any
    _out_dir: str
    _run: Any | None
    _run_metadata: dict[str, object] | None

    def __init__(self, config: Any, out_dir: str, *, run_metadata: Mapping[str, object] | None = None) -> None:
        super().__init__()
        self._config = config
        self._out_dir = out_dir
        self._run = None
        self._run_metadata = dict(run_metadata) if run_metadata is not None else None

    def _persistent_run_id(self) -> str:
        # i4 convention first: when the out_dir IS a run-attempt dir (<name>/<RUN_TS>/),
        # the directory name is the run identity — same string as the eval $RUN_TS, no
        # state file needed, and a fresh attempt is a fresh id (tombstone-proof).
        ts = os.path.basename(os.path.normpath(self._out_dir))
        if len(ts) == 14 and ts.isdigit():
            return ts
        # Legacy flat layout: random id persisted once (curve continuity across windows).
        path = os.path.join(self._out_dir, "wandb_run_id.txt")
        if os.path.exists(path):
            with open(path) as f:
                rid = f.read().strip()
            if rid:
                return rid
        rid = uuid.uuid4().hex[:16]
        os.makedirs(self._out_dir, exist_ok=True)
        with open(path, "w") as f:
            f.write(rid + "\n")
        return rid

    def start(self) -> None:
        try:
            import wandb

            destination = resolve_wandb_destination(self._config.job)
            mode = destination.mode
            # init_timeout: the SDK default (90s) is too tight for batch-node egress —
            # a transient slow path to api.wandb.ai cost a full window of metrics on
            # 2026-07-06 (init failed, breaker degraded to console-only). One retry on
            # top since the observed failures ARE transient (same init succeeds in ~2s).
            # console_multipart: without it each resumed window re-uploads `output.log`
            # and OVERWRITES the previous session's console capture server-side (metrics
            # are unaffected — separate append-only channel); multipart uploads one
            # timestamped log file per session so the full history survives resume.
            last_err: Exception | None = None
            for attempt in (1, 2):
                try:
                    self._run = wandb.init(
                        entity=destination.entity,
                        project=destination.project,
                        group=destination.group,
                        name=destination.name,
                        mode=mode,
                        dir=self._out_dir,
                        id=self._persistent_run_id(),
                        resume="allow",
                        config=self._safe_config(),
                        settings=wandb.Settings(init_timeout=300, console_multipart=True),
                    )
                    break
                except Exception as e:  # noqa: BLE001
                    last_err = e
                    logger.warning(f"wandb init attempt {attempt} failed ({e})")
            if self._run is None:
                raise last_err if last_err else RuntimeError("wandb init failed")
            logger.info(
                f"wandb initialized: project={destination.project} group={destination.group} "
                f"name={destination.name} mode={mode}"
            )
        except Exception as e:  # noqa: BLE001 — reporting must never kill training
            logger.warning(f"wandb init failed ({e}); continuing without wandb")
            self._run = None

    def _safe_config(self) -> dict[str, Any]:
        """RL/grid/job sections as plain containers (the inherited model tree holds
        callables that don't survive wandb's serialization — the full resolved config is
        already on disk as config_resolved.yaml). Sections are OmegaConf nodes on the
        hydra path but plain dataclasses when built programmatically — handle both."""
        out: dict[str, Any] = {}
        for k in ("rl", "grid", "job"):
            if not hasattr(self._config, k):
                continue
            section = getattr(self._config, k)
            try:
                from omegaconf import OmegaConf

                if OmegaConf.is_config(section):
                    out[k] = OmegaConf.to_container(section, resolve=True)
                elif dataclasses.is_dataclass(section) and not isinstance(section, type):
                    out[k] = dataclasses.asdict(section)
                else:
                    out[k] = dict(section) if isinstance(section, dict) else str(section)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"could not serialize config section {k!r} for wandb ({e}); skipped")
        if self._run_metadata is not None:
            out["metadata"] = dict(self._run_metadata)
        return out

    def _emit(self, metrics: dict[str, Any], step: int) -> None:
        # Same never-kill-training discipline as start()/finish(), with a CIRCUIT BREAKER:
        # a failing SDK (e.g. network gone in online mode) would otherwise warn on every
        # remaining step — warn once, drop the run, degrade to console-only.
        if self._run is None:
            return
        try:
            # Preserve schema types at the transport boundary: counts remain integers and status
            # flags remain booleans rather than becoming floating-point measurements.
            scalars = {}
            for k, v in metrics.items():
                value = as_report_scalar(v)
                if value is None:
                    continue
                # Canonical schema-v2 keys pass through byte-for-byte; aliases are forbidden at
                # ``stage`` rather than silently creating a second historical series here.
                scalars[k] = value
            self._run.log(scalars, step=step)
        except Exception as e:  # noqa: BLE001 — reporting must never kill training
            logger.warning(f"wandb log failed at step {step} ({e}); disabling wandb for this run")
            self._run = None

    def finish(self) -> None:
        # Called from the loop's `finally`: a raise here (e.g. an online-mode network
        # flush) would REPLACE the original training exception — degrade instead, same
        # discipline as start()/report().
        try:
            super().finish()
        except Exception as e:  # noqa: BLE001 — reporting must never mask the real failure
            logger.warning(f"wandb pending metric flush failed ({e}); continuing shutdown")
        if self._run is None:
            return
        try:
            self._run.finish()
        except Exception as e:  # noqa: BLE001 — reporting must never mask the real failure
            logger.warning(f"wandb finish failed ({e}); run may need a manual `wandb sync`")
        finally:
            self._run = None


def prior_clip_counts(reporters: list[Reporter]) -> tuple[int, int]:
    """(clipped updates, counted updates) the started ``JsonlReporter`` channels kept from earlier attempts; (0, 0)
    without one (other ranks, a fresh attempt, a failed start)."""
    clipped = counted = 0
    for reporter in reporters:
        if isinstance(reporter, JsonlReporter):
            clipped += reporter.prior_clip_counts[0]
            counted += reporter.prior_clip_counts[1]
    return clipped, counted


def make_reporters(
    config: Any,
    out_dir: str,
    total_steps: int | None = None,
    headline_keys: tuple[str, ...] = _CONSOLE_KEYS,
    *,
    run_metadata: Mapping[str, object] | None = None,
    jsonl_path: str | None = None,
    jsonl_keep_through: int = -1,
) -> list[Reporter]:
    """The reporter set for THIS process: empty off rank 0 (under torchrun every rank runs the loop;
    on the Ray grid only the driver does), else console + W&B (unless the mode resolves to "disabled"
    — `WANDB_MODE` env wins over `job.wandb_mode` — or wandb is unavailable) + a `JsonlReporter` when
    `jsonl_path` is given (`jsonl_keep_through` = the step the attempt resumes at; -1 = a fresh file).
    `total_steps` feeds the console progress fraction; `headline_keys` selects canonical console
    columns. W&B and the JSONL file receive the same keys unchanged."""
    if int(os.environ.get("RANK", 0)) != 0:
        return []
    reporters: list[Reporter] = [ConsoleReporter(total_steps=total_steps, headline_keys=headline_keys)]
    if jsonl_path:
        reporters.append(JsonlReporter(jsonl_path, keep_through=jsonl_keep_through))
    mode = os.environ.get("WANDB_MODE") or str(getattr(config.job, "wandb_mode", "disabled"))
    if mode != "disabled":
        try:
            import wandb  # noqa: F401

            reporters.append(WandbReporter(config, out_dir, run_metadata=run_metadata))
        except ImportError:
            logger.warning("job.wandb_mode=%s but wandb is not installed; console only", mode)
    return reporters
