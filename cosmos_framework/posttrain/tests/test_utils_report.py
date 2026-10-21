# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""CPU tests for the reporter abstraction (utils.report) + the package logger.

Covers channel selection (rank gate, wandb_mode), the console line, scalar filtering,
and a REAL offline wandb roundtrip (no network/login needed in offline mode)."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from cosmos_framework.posttrain.utils.log import logger as posttrain_logger
from cosmos_framework.posttrain.utils.profile import ProfileRecord
from cosmos_framework.posttrain.utils.report import (
    ConsoleReporter,
    Reporter,
    WandbReporter,
    emit_best_effort,
    finish_best_effort,
    make_reporters,
    merge_training_metric_window,
    start_best_effort,
)


def _cfg(wandb_mode: str = "disabled") -> SimpleNamespace:
    return SimpleNamespace(
        job=SimpleNamespace(
            project="cosmos3_vfm",
            group="posttrain_dpo",
            name="unit",
            wandb_mode=wandb_mode,
        )
    )


@pytest.fixture()
def capture_posttrain_log() -> Iterator[list[logging.LogRecord]]:
    """The posttrain logger doesn't propagate to root (so caplog misses it) — attach a
    temporary capturing handler instead."""
    records: list[logging.LogRecord] = []

    class _Cap(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    h = _Cap(level=logging.DEBUG)
    posttrain_logger.addHandler(h)
    yield records
    posttrain_logger.removeHandler(h)


@pytest.mark.L0
@pytest.mark.CPU
def test_module_level_logger_configured_once() -> None:
    """cosmos-rl style: ONE module-level logger, import-and-use. Re-import yields the
    same object; the idempotence guard means our handler is attached exactly once."""
    import importlib

    from cosmos_framework.posttrain.utils import log as ptlog

    ours = [h for h in posttrain_logger.handlers if any(type(f).__name__ == "_RankFilter" for f in h.filters)]
    assert len(ours) == 1  # exactly one posttrain handler (pytest may add its own)
    assert importlib.reload(ptlog).logger is posttrain_logger  # reload doesn't stack a second handler
    ours = [h for h in posttrain_logger.handlers if any(type(f).__name__ == "_RankFilter" for f in h.filters)]
    assert len(ours) == 1
    assert posttrain_logger.propagate is False


@pytest.mark.L0
@pytest.mark.CPU
def test_make_reporters_rank_gate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RANK", "1")
    assert make_reporters(_cfg(), str(tmp_path)) == []  # non-rank0: silent
    monkeypatch.setenv("RANK", "0")
    reporters = make_reporters(_cfg(wandb_mode="disabled"), str(tmp_path))
    assert len(reporters) == 1 and isinstance(reporters[0], ConsoleReporter)  # console always on


@pytest.mark.L0
@pytest.mark.CPU
def test_console_reporter_line(capture_posttrain_log: list[logging.LogRecord]) -> None:
    # The caller supplies the public model step; Reporter does not add one.
    ConsoleReporter(total_steps=500).report(
        {"train/loss_avg": 0.5, "train/dpo_accuracy": 1.0, "optim/update_outcome": "applied"}, step=3
    )
    line = capture_posttrain_log[-1].getMessage()
    # Labels keep the complete stage namespace, so train and val losses remain unambiguous.
    assert "step    3/500" in line
    assert "train/loss_avg +0.5000" in line
    assert "train/dpo_accuracy +1.0000" in line
    assert "optim/update_outcome" not in line  # console shows headline keys only


@pytest.mark.L0
@pytest.mark.CPU
def test_console_reporter_dynamic_total_seam(capture_posttrain_log: list[logging.LogRecord]) -> None:
    """GRPO seam: total unknown at construction → bare step; assign once probed → fraction."""
    r = ConsoleReporter()
    r.report({"val/reward_avg": 0.5}, step=0)
    assert "step     0" in capture_posttrain_log[-1].getMessage()
    r.total_steps = 42  # e.g. computed from the live replica count
    r.report({"train/loss_avg": 0.5}, step=42)
    assert "step   42/42" in capture_posttrain_log[-1].getMessage()


@pytest.mark.L0
@pytest.mark.CPU
def test_console_reporter_prints_sub_milli_magnitudes_in_scientific_notation(
    capture_posttrain_log: list[logging.LogRecord],
) -> None:
    """A learning rate of 1e-5 printed as `+0.0000` at four decimals (review R121B-01): a magnitude below 1e-3
    prints in scientific notation; zero and everything from 1e-3 up keep the fixed four-decimal format."""
    keys = ("train/loss_avg", "train/learning_rate", "train/grad_norm", "train/valid_token_count", "val/loss_avg")
    ConsoleReporter(total_steps=1401, headline_keys=keys).report(
        {
            "train/loss_avg": 0.81234,
            "train/learning_rate": 1e-5,
            "train/grad_norm": -0.0004,
            "train/valid_token_count": 176703.0,
            "val/loss_avg": 0.0,
        },
        step=1,
    )
    line = capture_posttrain_log[-1].getMessage()
    assert "train/learning_rate +1.000e-05" in line, line
    assert "train/grad_norm -4.000e-04" in line, line
    assert "train/loss_avg +0.8123" in line and "train/valid_token_count +176703.0000" in line, line
    assert "val/loss_avg +0.0000" in line, line  # zero keeps the fixed format
    ConsoleReporter(headline_keys=("train/loss_avg", "val/loss_avg")).report(
        {"train/loss_avg": 0.001, "val/loss_avg": 0.00099}, step=2
    )
    line = capture_posttrain_log[-1].getMessage()
    assert "train/loss_avg +0.0010" in line and "val/loss_avg +9.900e-04" in line, line


@pytest.mark.L0
@pytest.mark.CPU
def test_training_metric_window_preserves_weighted_and_empty_reductions() -> None:
    merged = merge_training_metric_window(
        [
            {
                "train/loss_avg": 1.0,
                "train/weight_sum": 10.5,
                "train/opd_divergence": 4.0,
                "train/valid_sequence_count": 3,
                "train/valid_token_count": 11,
                "train/learning_rate": 0.2,
            },
            {
                "train/loss_avg": 2.0,
                "train/weight_sum": 30,
                "train/opd_divergence": 1.0,
                "train/valid_sequence_count": 5,
                "train/valid_token_count": 13,
                "train/learning_rate": 0.1,
                "train/grad_norm": 7.0,
            },
        ],
        sum_keys=frozenset({"train/valid_sequence_count"}),
        weighted_mean_by={
            "train/loss_avg": "train/weight_sum",
            "train/opd_divergence": "train/valid_token_count",
        },
    )

    assert merged["train/loss_avg"] == pytest.approx(70.5 / 40.5)
    assert merged["train/opd_divergence"] == pytest.approx(57.0 / 24.0)
    assert merged["train/weight_sum"] == 40.5
    assert merged["train/valid_sequence_count"] == 8
    assert merged["train/valid_token_count"] == 24
    assert merged["train/learning_rate"] == 0.1
    assert merged["train/grad_norm"] == 7.0

    zero = merge_training_metric_window(
        [{"train/loss_avg": 0.0, "train/weight_sum": 0.0}],
        weighted_mean_by={"train/loss_avg": "train/weight_sum"},
    )
    assert zero["train/loss_avg"] == 0.0
    assert zero["train/weight_sum"] == 0.0

    unweighted = merge_training_metric_window(
        [{"train/loss_avg": 1.0}, {"train/loss_avg": 3.0}],
        weighted_mean_by={"train/loss_avg": "train/weight_sum"},
    )
    assert unweighted["train/loss_avg"] == 2.0

    with pytest.raises(ValueError, match="must be zero whenever 'train/weight_sum' is zero"):
        merge_training_metric_window(
            [{"train/loss_avg": 1.0, "train/weight_sum": 0.0}],
            weighted_mean_by={"train/loss_avg": "train/weight_sum"},
        )
    with pytest.raises(ValueError, match="requires 'train/weight_sum' in every contributing report"):
        merge_training_metric_window(
            [{"train/loss_avg": 1.0, "train/weight_sum": 1.0}, {"train/loss_avg": 2.0}],
            weighted_mean_by={"train/loss_avg": "train/weight_sum"},
        )


@pytest.mark.L0
@pytest.mark.CPU
def test_info0_rank_gate(
    monkeypatch: pytest.MonkeyPatch,
    capture_posttrain_log: list[logging.LogRecord],
) -> None:
    """Collective-semantics events: INFO on rank 0, demoted to DEBUG elsewhere (still
    emitted — visible under POSTTRAIN_LOG_LEVEL=DEBUG when debugging a hung collective)."""
    import logging

    from cosmos_framework.posttrain.utils.log import info0, logger

    monkeypatch.setenv("RANK", "0")
    info0("saved checkpoint x")
    assert capture_posttrain_log[-1].levelno == logging.INFO
    # Non-rank-0 at the default INFO level: fully silent (the whole point — no x32 spam).
    n = len(capture_posttrain_log)
    monkeypatch.setenv("RANK", "7")
    info0("saved checkpoint x")
    assert len(capture_posttrain_log) == n
    # With DEBUG enabled the per-rank line reappears (hung-collective debugging path).
    old_level = logger.level
    try:
        logger.setLevel(logging.DEBUG)
        info0("saved checkpoint x")
        assert capture_posttrain_log[-1].levelno == logging.DEBUG
    finally:
        logger.setLevel(old_level)


@pytest.mark.L0
@pytest.mark.CPU
def test_wandb_reporter_offline_roundtrip(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Real wandb in OFFLINE mode: init → log → finish, run files land under out_dir.
    Also proves non-scalar metrics are filtered rather than crashing the log call."""
    wandb = pytest.importorskip("wandb")
    del wandb
    monkeypatch.setenv("RANK", "0")
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    reporters = make_reporters(_cfg(wandb_mode="offline"), str(tmp_path))
    assert any(isinstance(r, WandbReporter) for r in reporters)
    wb = next(r for r in reporters if isinstance(r, WandbReporter))
    wb.start()
    assert wb._run is not None, "offline wandb init should succeed without login"
    assert wb._run.project == "cosmos3_generator"
    wb.report({"val/loss_avg": 0.25, "val/note": "not-a-scalar"}, step=0)
    wb.report({"train/loss_avg": 0.20}, step=1)
    wb.finish()
    assert wb._run is None
    assert list(tmp_path.glob("wandb/offline-run-*")), "offline run dir not created"
    # The run id must have been persisted for cross-window curve continuity.
    assert (tmp_path / "wandb_run_id.txt").exists()


@pytest.mark.L0
@pytest.mark.CPU
def test_wandb_reporter_stages_train_and_val_into_one_canonical_row(tmp_path: Path) -> None:
    """A validation step is one W&B commit, with finalized primitives passed through unchanged."""

    class _RecordingRun:
        def __init__(self) -> None:
            self.calls: list[tuple[dict[str, int | float | bool], int]] = []

        def log(self, metrics: dict[str, int | float | bool], *, step: int) -> None:
            self.calls.append((metrics, step))

    run = _RecordingRun()
    wb = WandbReporter(_cfg(wandb_mode="online"), str(tmp_path))
    wb._run = run

    wb.stage(
        {
            "train/loss_avg": 0.25,
            "train/reward_mean": 0.5,
            "train/valid_token_count": 12,
        },
        step=7,
    )
    wb.stage({"val/reward_avg": 0.75, "val/rollout_count": 8}, step=7)
    assert run.calls == []
    wb.commit(step=7)

    assert run.calls == [
        (
            {
                "train/loss_avg": 0.25,
                "train/reward_mean": 0.5,
                "train/valid_token_count": 12,
                "val/reward_avg": 0.75,
                "val/rollout_count": 8,
            },
            7,
        )
    ]
    assert type(run.calls[0][0]["train/valid_token_count"]) is int
    assert type(run.calls[0][0]["val/rollout_count"]) is int
    assert wb.last_committed_step == 7
    with pytest.raises(ValueError, match="only once"):
        wb.report({"val/reward_avg": 0.8}, step=7)


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    "key",
    [
        "grpo_loss",
        "dpo_accuracy",
        "GRPO/grpo_loss",
        "DPO/dpo_loss",
        "optim/lr",
        "optim/grad_norm",
        "timer/step_s",
    ],
)
def test_reporter_rejects_legacy_metric_keys(key: str) -> None:
    reporter = ConsoleReporter()
    with pytest.raises(ValueError, match="legacy metric key"):
        reporter.report({key: 1.0}, step=1)


@pytest.mark.L0
@pytest.mark.CPU
def test_reporter_rejects_duplicate_or_cross_step_staging() -> None:
    reporter = ConsoleReporter()
    reporter.stage({"train/loss_avg": 0.5}, step=1)
    with pytest.raises(ValueError, match="duplicate metric keys"):
        reporter.stage({"train/loss_avg": 0.4}, step=1)
    with pytest.raises(ValueError, match="step 1 is pending"):
        reporter.stage({"val/loss_avg": 0.3}, step=2)


@pytest.mark.L0
@pytest.mark.CPU
def test_reporter_transport_failure_consumes_ambiguous_frame() -> None:
    """An emit that may have partially succeeded is at-most-once, never flushed again at finish."""

    class _FailingReporter(ConsoleReporter):
        attempts: int = 0

        def _emit(self, metrics: dict[str, Any], step: int) -> None:
            del metrics, step
            self.attempts += 1
            raise RuntimeError("transport failed after an ambiguous side effect")

    reporter = _FailingReporter()
    with pytest.raises(RuntimeError, match="ambiguous side effect"):
        reporter.report({"val/loss_avg": 0.3}, step=2)

    assert reporter.last_committed_step == 2
    reporter.finish()
    assert reporter.attempts == 1


@pytest.mark.L0
@pytest.mark.CPU
def test_emit_best_effort_disables_only_the_failed_transport(
    capture_posttrain_log: list[logging.LogRecord],
) -> None:
    """One broken sink is attempted once while healthy sinks keep receiving later model steps."""

    class _RecordingReporter(Reporter):
        def __init__(self, *, fail: bool) -> None:
            super().__init__()
            self.fail = fail
            self.rows: list[tuple[int, dict[str, Any]]] = []
            self.attempts = 0

        def _emit(self, metrics: dict[str, Any], step: int) -> None:
            self.attempts += 1
            if self.fail:
                raise RuntimeError("sink offline")
            self.rows.append((step, metrics))

    failed = _RecordingReporter(fail=True)
    healthy = _RecordingReporter(fail=False)
    disabled: set[int] = set()

    emit_best_effort(
        [failed, healthy],
        {"train/loss_avg": 0.5},
        1,
        failed_reporters=disabled,
    )
    emit_best_effort(
        [failed, healthy],
        {"train/loss_avg": 0.25},
        2,
        failed_reporters=disabled,
    )

    assert failed.attempts == 1
    assert disabled == {id(failed)}
    assert healthy.rows == [
        (1, {"train/loss_avg": 0.5}),
        (2, {"train/loss_avg": 0.25}),
    ]
    assert sum("disabling it for this run" in record.getMessage() for record in capture_posttrain_log) == 1


@pytest.mark.L0
@pytest.mark.CPU
def test_emit_best_effort_rejects_invalid_schema_before_any_transport() -> None:
    """Schema violations remain fatal framework errors and cannot partially reach a sink."""

    class _RecordingReporter(Reporter):
        def __init__(self) -> None:
            super().__init__()
            self.attempts = 0

        def _emit(self, metrics: dict[str, Any], step: int) -> None:
            del metrics, step
            self.attempts += 1

    reporter = _RecordingReporter()
    disabled: set[int] = set()

    with pytest.raises(ValueError, match="legacy metric key"):
        emit_best_effort(
            [reporter],
            {"dpo_loss": 0.5},
            1,
            failed_reporters=disabled,
        )

    assert reporter.attempts == 0
    assert disabled == set()


@pytest.mark.L0
@pytest.mark.CPU
def test_reporter_start_and_finish_failures_are_isolated(
    capture_posttrain_log: list[logging.LogRecord],
) -> None:
    """Lifecycle transport failures obey the same best-effort circuit breaker as report()."""

    class _LifecycleReporter(Reporter):
        def __init__(self, *, fail_start: bool = False, fail_finish: bool = False) -> None:
            super().__init__()
            self.fail_start = fail_start
            self.fail_finish = fail_finish
            self.starts = 0
            self.finishes = 0

        def start(self) -> None:
            self.starts += 1
            if self.fail_start:
                raise RuntimeError("start transport failed")

        def _emit(self, metrics: dict[str, Any], step: int) -> None:
            del metrics, step

        def finish(self) -> None:
            self.finishes += 1
            if self.fail_finish:
                raise RuntimeError("finish transport failed")

    broken_start = _LifecycleReporter(fail_start=True)
    broken_finish = _LifecycleReporter(fail_finish=True)
    healthy = _LifecycleReporter()
    reporters = [broken_start, broken_finish, healthy]
    disabled: set[int] = set()

    start_best_effort(reporters, failed_reporters=disabled)
    finish_best_effort(reporters, failed_reporters=disabled)

    assert disabled == {id(broken_start), id(broken_finish)}
    assert broken_start.starts == 1 and broken_start.finishes == 0
    assert broken_finish.starts == 1 and broken_finish.finishes == 1
    assert healthy.starts == 1 and healthy.finishes == 1
    messages = [record.getMessage() for record in capture_posttrain_log]
    assert any("failed to start" in message for message in messages)
    assert any("failed to finish" in message for message in messages)


@pytest.mark.L0
@pytest.mark.CPU
def test_wandb_reporter_maps_generator_project_at_init(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The legacy storage project maps only at the W&B initialization boundary."""
    import wandb as wandb_mod

    init_kwargs: dict[str, object] = {}
    fake_run = SimpleNamespace()

    def _fake_wandb_init(**kwargs: object) -> SimpleNamespace:
        init_kwargs.update(kwargs)
        return fake_run

    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    monkeypatch.setattr(wandb_mod, "init", _fake_wandb_init)

    wb = WandbReporter(_cfg(wandb_mode="online"), str(tmp_path))
    wb.start()

    assert wb._run is fake_run
    assert init_kwargs["project"] == "cosmos3_generator"
    assert wb._config.job.project == "cosmos3_vfm"


@pytest.mark.L0
@pytest.mark.CPU
def test_wandb_reporter_env_overrides_group_and_name(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """All four W&B coordinates take env-over-config precedence.

    `WANDB_PROJECT`/`WANDB_MODE` always did; group/name did not, so a run launched into a
    study cohort with `WANDB_RUN_GROUP` set still landed under the recipe's own group and
    silently fell out of the comparison cohort.
    """
    import wandb as wandb_mod

    init_kwargs: dict[str, object] = {}

    def _fake_wandb_init(**kwargs: object) -> SimpleNamespace:
        init_kwargs.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setenv("WANDB_RUN_GROUP", "reasoner_grpo_parity_r1")
    monkeypatch.setenv("WANDB_NAME", "posttrain_seed42")
    monkeypatch.setattr(wandb_mod, "init", _fake_wandb_init)

    WandbReporter(_cfg(wandb_mode="online"), str(tmp_path)).start()

    assert init_kwargs["group"] == "reasoner_grpo_parity_r1"
    assert init_kwargs["name"] == "posttrain_seed42"


@pytest.mark.L0
@pytest.mark.CPU
def test_wandb_reporter_falls_back_to_config_group_and_name(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """With no env override the recipe's own `job.group`/`job.name` still win — the
    override must not become a required launch variable."""
    import wandb as wandb_mod

    init_kwargs: dict[str, object] = {}

    def _fake_wandb_init(**kwargs: object) -> SimpleNamespace:
        init_kwargs.update(kwargs)
        return SimpleNamespace()

    monkeypatch.delenv("WANDB_RUN_GROUP", raising=False)
    monkeypatch.delenv("WANDB_NAME", raising=False)
    monkeypatch.setattr(wandb_mod, "init", _fake_wandb_init)

    WandbReporter(_cfg(wandb_mode="online"), str(tmp_path)).start()

    assert init_kwargs["group"] == "posttrain_dpo"
    assert init_kwargs["name"] == "unit"


@pytest.mark.L0
@pytest.mark.CPU
def test_wandb_run_id_persistent_and_dir_unique(tmp_path: Path) -> None:
    """Same out_dir ⇒ same id across restarts (curve continuity); the id is random per
    dir, NOT a hash of the path (a deterministic id resurrects deleted-run tombstones
    and wandb.init hangs — the 2026-07-06 relaunch failure)."""
    a = WandbReporter(_cfg(), str(tmp_path / "a"))
    id1 = a._persistent_run_id()
    id2 = WandbReporter(_cfg(), str(tmp_path / "a"))._persistent_run_id()
    assert id1 == id2 and len(id1) == 16
    assert (tmp_path / "a" / "wandb_run_id.txt").read_text().strip() == id1
    assert WandbReporter(_cfg(), str(tmp_path / "b"))._persistent_run_id() != id1


@pytest.mark.L0
@pytest.mark.CPU
def test_wandb_mode_env_overrides_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """WANDB_MODE=disabled must win over job.wandb_mode=online (smoke-run hygiene:
    an accidental online sync from a test node pollutes the real run's group)."""
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WANDB_MODE", "disabled")
    reporters = make_reporters(_cfg(wandb_mode="online"), str(tmp_path))
    assert not any(isinstance(r, WandbReporter) for r in reporters)


@pytest.mark.L0
@pytest.mark.CPU
def test_wandb_reporter_start_failure_degrades(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capture_posttrain_log: list[logging.LogRecord],
) -> None:
    """A broken wandb must degrade to a warning, never kill training."""
    wb = WandbReporter(_cfg(wandb_mode="online"), str(tmp_path))
    import wandb as wandb_mod

    def boom(**kw: object) -> None:
        raise RuntimeError("no api key")

    monkeypatch.setattr(wandb_mod, "init", boom)
    wb.start()  # must not raise
    assert wb._run is None
    wb.report({"val/loss_avg": 1.0}, step=0)  # no-op, must not raise
    wb.finish()
    assert any("wandb init failed" in r.getMessage() for r in capture_posttrain_log)


@pytest.mark.L0
@pytest.mark.CPU
def test_wandb_reporter_finish_failure_degrades(
    tmp_path: Path,
    capture_posttrain_log: list[logging.LogRecord],
) -> None:
    """finish() runs in the loop's `finally` — a raise there would mask the real training
    exception. It must degrade to a warning and still drop the run handle."""

    class _BoomRun:
        def finish(self) -> None:
            raise RuntimeError("network flush failed")

    wb = WandbReporter(_cfg(wandb_mode="online"), str(tmp_path))
    wb._run = _BoomRun()
    wb.finish()  # must not raise
    assert wb._run is None
    assert any("wandb finish failed" in r.getMessage() for r in capture_posttrain_log)


@pytest.mark.L0
@pytest.mark.CPU
def test_wandb_reporter_log_failure_circuit_breaks(
    tmp_path: Path,
    capture_posttrain_log: list[logging.LogRecord],
) -> None:
    """A failing run.log warns ONCE, drops the run (no per-step warning flood), and
    subsequent report() calls are silent no-ops — training never sees the exception."""

    class _BoomRun:
        def log(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("sdk broke")

    wb = WandbReporter(_cfg(wandb_mode="online"), str(tmp_path))
    wb._run = _BoomRun()
    wb.report({"val/loss_avg": 1.0}, step=0)  # must not raise
    assert wb._run is None  # circuit broken
    wb.report({"train/loss_avg": 1.0}, step=1)  # silent no-op
    warns = [r for r in capture_posttrain_log if "wandb log failed" in r.getMessage()]
    assert len(warns) == 1  # exactly one warning, no flood


@pytest.mark.L0
@pytest.mark.CPU
def test_wandb_run_id_from_run_ts_dir(tmp_path: Path) -> None:
    """out_dir = <name>/<RUN_TS>/ ⇒ the wandb id IS the directory name (i4 convention,
    no state file); non-ts dirs keep the legacy persisted-random-id mechanism."""
    ts_dir = tmp_path / "exp" / "20260707103000"
    ts_dir.mkdir(parents=True)
    assert WandbReporter(_cfg(), str(ts_dir))._persistent_run_id() == "20260707103000"
    assert not (ts_dir / "wandb_run_id.txt").exists()


@pytest.mark.L0
@pytest.mark.CPU
def test_profile_record_formats_phases_and_checkpoints() -> None:
    ticks = iter((0.0, 1.0, 3.0, 5.0, 10.0))
    with (
        patch("cosmos_framework.posttrain.utils.profile.gpu_memory_used_mib", return_value=None),
        patch("cosmos_framework.posttrain.utils.profile.torch_memory_mib", return_value=(None, None)),
    ):
        record = ProfileRecord("step", enabled=True, clock=lambda: next(ticks))
        with record.phase("body"):
            pass
        record.checkpoint("experience")
        record.set_phase("release")
    with patch("cosmos_framework.posttrain.utils.profile.logger") as mock_logger:
        message = record.emit(step=1)
    assert message == (
        "benchmark profile component=step step=1 body_s=2.000000 experience_s=5.000000 "
        "release_s=0.000000 total_s=10.000000"
    )
    mock_logger.info.assert_called_once_with(message)

    def fail_clock() -> float:
        raise AssertionError("disabled profiling must not read the clock")

    assert ProfileRecord("disabled", enabled=False, clock=fail_clock).emit() is None


@pytest.mark.L0
@pytest.mark.CPU
def test_profile_record_samples_device_memory_at_start_and_each_checkpoint() -> None:
    ticks = iter((0.0, 1.0, 2.0, 3.0))
    samples = iter((1000.0, 1200.5, 300.25))
    with (
        patch("cosmos_framework.posttrain.utils.profile.gpu_memory_used_mib", side_effect=lambda: next(samples)),
        patch("cosmos_framework.posttrain.utils.profile.torch_memory_mib", return_value=(None, None)),
    ):
        record = ProfileRecord("grpo_step", enabled=True, clock=lambda: next(ticks))
        record.checkpoint("experience")  # engine awake, trainer offloaded
        record.checkpoint("offload_rollout")  # engine asleep, trainer resident
    with patch("cosmos_framework.posttrain.utils.profile.logger"):
        message = record.emit()
    assert message == (
        "benchmark profile component=grpo_step experience_s=1.000000 offload_rollout_s=1.000000 "
        "start_mib=1000.0 experience_mib=1200.5 offload_rollout_mib=300.2 total_s=3.000000"
    )

    def fail_sampler() -> float:
        raise AssertionError("disabled profiling must not query NVML")

    with patch("cosmos_framework.posttrain.utils.profile.gpu_memory_used_mib", side_effect=fail_sampler):
        ProfileRecord("disabled", enabled=False).checkpoint("x")


@pytest.mark.L0
@pytest.mark.CPU
def test_profile_record_progress_logs_each_phase_before_the_record_is_emitted() -> None:
    """A setup cancelled before ``emit`` still leaves a line per finished phase; plain records stay quiet."""
    ticks = iter((0.0, 2.0, 7.5))
    samples = iter((100.0, 4096.0, 8192.5))
    with (
        patch("cosmos_framework.posttrain.utils.profile.gpu_memory_used_mib", side_effect=lambda: next(samples)),
        patch("cosmos_framework.posttrain.utils.profile.torch_memory_mib", return_value=(None, None)),
        patch("cosmos_framework.posttrain.utils.profile.logger") as mock_logger,
    ):
        record = ProfileRecord("grpo_setup", enabled=True, progress=True, clock=lambda: next(ticks))
        record.checkpoint("rollout")
        record.checkpoint("policy")
    assert [call.args[0] for call in mock_logger.info.call_args_list] == [
        "benchmark phase component=grpo_setup phase=rollout phase_s=2.000 elapsed_s=2.000 used_mib=4096.0",
        "benchmark phase component=grpo_setup phase=policy phase_s=5.500 elapsed_s=7.500 used_mib=8192.5",
    ]

    quiet_ticks = iter((0.0, 1.0))
    with (
        patch("cosmos_framework.posttrain.utils.profile.gpu_memory_used_mib", return_value=None),
        patch("cosmos_framework.posttrain.utils.profile.torch_memory_mib", return_value=(None, None)),
        patch("cosmos_framework.posttrain.utils.profile.logger") as quiet_logger,
    ):
        ProfileRecord("grpo_step", enabled=True, clock=lambda: next(quiet_ticks)).checkpoint("experience")
    quiet_logger.info.assert_not_called()


@pytest.mark.L0
@pytest.mark.CPU
def test_training_metric_window_takes_the_max_for_max_keys() -> None:
    """A per-chunk maximum merges as the window MAX (GRPO's `train/kl_max`), beside the sums and
    weighted means; a key cannot be both a maximum and a sum or weighted mean."""
    merged = merge_training_metric_window(
        [
            {"train/kl_max": 0.5, "train/kl_mean": 0.1, "train/valid_token_count": 4},
            {"train/kl_max": 173.0, "train/kl_mean": 0.3, "train/valid_token_count": 12},
            {"train/kl_max": 2.0, "train/kl_mean": 0.2, "train/valid_token_count": 4},
        ],
        sum_keys=frozenset({"train/valid_token_count"}),
        max_keys=frozenset({"train/kl_max"}),
        weighted_mean_by={"train/kl_mean": "train/valid_token_count"},
    )
    assert merged["train/kl_max"] == 173.0  # not the mean 58.5, not the sum 175.5
    assert merged["train/kl_mean"] == pytest.approx((0.4 + 3.6 + 0.8) / 20.0)
    assert merged["train/valid_token_count"] == 20
    # A max key absent from some chunks takes the max over the chunks that reported it.
    partial = merge_training_metric_window(
        [{"train/loss_avg": 1.0}, {"train/loss_avg": 2.0, "train/kl_max": 7.0}],
        max_keys=frozenset({"train/kl_max"}),
    )
    assert partial["train/kl_max"] == 7.0 and partial["train/loss_avg"] == 1.5
    # Default (no max keys) is unchanged: the same key would be averaged.
    assert merge_training_metric_window([{"train/kl_max": 0.5}, {"train/kl_max": 173.0}])["train/kl_max"] == 86.75
    for conflicting in (
        {"sum_keys": frozenset({"train/kl_max"})},
        {"weighted_mean_by": {"train/kl_max": "train/valid_token_count"}},
        {"weighted_mean_by": {"train/kl_mean": "train/kl_max"}},
    ):
        with pytest.raises(ValueError, match="max_keys cannot also be summed or weighted"):
            merge_training_metric_window(
                [{"train/kl_max": 1.0, "train/kl_mean": 1.0, "train/valid_token_count": 1}],
                max_keys=frozenset({"train/kl_max"}),
                **conflicting,
            )


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("mode", ["online", "offline"])
def test_wandb_destination_and_safe_config_never_serialize_credentials(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: str
) -> None:
    import json
    from dataclasses import asdict

    from cosmos_framework.posttrain.utils.report import resolve_wandb_destination

    marker = "test-only-credential-marker"
    monkeypatch.setenv("WANDB_API_KEY", marker)
    monkeypatch.setenv("WANDB_MODE", mode)
    monkeypatch.setenv("WANDB_PROJECT", "cosmos3_action")
    monkeypatch.setenv("WANDB_ENTITY", "your-wandb-entity")
    job = SimpleNamespace(project="cosmos3_action", group="test_group", name="test_run", wandb_mode="offline")
    destination = resolve_wandb_destination(job)
    assert destination.mode == mode  # the environment overrides the recipe's mode
    config = SimpleNamespace(job=job, rl=SimpleNamespace(), grid=SimpleNamespace())
    serialized = json.dumps(WandbReporter(config, str(tmp_path))._safe_config())
    assert marker not in json.dumps(asdict(destination)) + serialized


@pytest.mark.L0
@pytest.mark.CPU
def test_jsonl_reporter_appends_one_row_per_committed_step(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Review R125-B-03: a step's train and validation payloads land as ONE row (stage, stage, commit); every public key
    is kept; ``make_reporters`` adds the channel on rank 0 when a path is given and never off rank 0."""
    import json

    from cosmos_framework.posttrain.utils.report import JsonlReporter

    path = tmp_path / "run" / "metrics.jsonl"
    reporter = JsonlReporter(str(path))
    reporter.start()
    reporter.stage({"train/loss_avg": 1.5, "train/grad_norm": 7.25, "train/learning_rate": 1e-5}, step=1)
    reporter.commit(1, {"val/loss_avg": 0.5, "optim/skip_reason": "none"})
    reporter.report({"train/loss_avg": 1.25, "train/grad_clipped": 1.0}, step=2)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["step"] for row in rows] == [1, 2]
    assert (
        rows[0]["train/grad_norm"] == 7.25 and rows[0]["val/loss_avg"] == 0.5 and rows[0]["optim/skip_reason"] == "none"
    )
    assert rows[1] == {"step": 2, "time": rows[1]["time"], "train/loss_avg": 1.25, "train/grad_clipped": 1.0}
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.setenv("WANDB_MODE", "disabled")
    kinds = [type(r).__name__ for r in make_reporters(_cfg(), str(tmp_path), jsonl_path=str(path))]
    assert kinds == ["ConsoleReporter", "JsonlReporter"]
    assert [type(r).__name__ for r in make_reporters(_cfg(), str(tmp_path))] == ["ConsoleReporter"]
    monkeypatch.setenv("RANK", "1")
    assert make_reporters(_cfg(), str(tmp_path), jsonl_path=str(path)) == []
    with pytest.raises(ValueError, match="path"):
        JsonlReporter("")


@pytest.mark.L0
@pytest.mark.CPU
def test_jsonl_reporter_keeps_only_the_rows_the_attempt_will_not_re_emit(tmp_path: Path) -> None:
    """Review R134-05: ``keep_through`` is the step the attempt resumes at. ``start`` keeps the rows with ``step <=
    keep_through`` (the relaunch re-emits the later ones, so the file stays one row per step), seeds ``prior_clip_counts``
    from the kept clip record, and ``-1`` (a fresh attempt) empties the file; a row without an integer step fails the
    start. 22bfce78f40's reporter only created the directory and appended."""
    import json

    from cosmos_framework.posttrain.utils.report import JsonlReporter, prior_clip_counts

    path = tmp_path / "metrics.jsonl"
    rows = [
        {"step": 0, "val/loss_avg": 1.0},
        {"step": 1, "train/grad_clipped": 1.0},
        {"step": 2, "train/grad_clipped": 0.0},
        {"step": 3, "train/grad_clipped": 1.0},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    resumed = JsonlReporter(str(path), keep_through=2)
    resumed.start()
    assert [json.loads(line)["step"] for line in path.read_text().splitlines()] == [0, 1, 2]
    assert resumed.prior_clip_counts == (1, 2) and prior_clip_counts([resumed]) == (1, 2)
    resumed.report({"train/grad_clipped": 0.0}, step=3)
    assert [json.loads(line)["step"] for line in path.read_text().splitlines()] == [0, 1, 2, 3]
    fresh = JsonlReporter(str(path))
    fresh.start()
    assert path.read_text() == "" and fresh.prior_clip_counts == (0, 0) and prior_clip_counts([]) == (0, 0)
    with pytest.raises(ValueError, match="keep_through"):
        JsonlReporter(str(path), keep_through=-2)
    path.write_text('{"no_step": 1}\n')
    with pytest.raises(ValueError, match="integer step"):
        JsonlReporter(str(path), keep_through=5).start()
