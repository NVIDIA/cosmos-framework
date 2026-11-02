# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""@L0 @CPU: the DPO entry's loop contract, launch validation, and the derived launch values that fail silently.

The loop-invariant pins (mirroring grpo's) cover the six behaviors whose failure is silent or first
surfaces as an expensive cluster crash: one-loop-step == one-optimizer-step with a fresh pair pull per
micro-batch; a non-finite loss aborts before it is reported or checkpointed; a resumed loop runs only the
remaining steps; preemption emergency-saves and never double-saves an already-checkpointed step; and a run
that completes off the save cadence still saves its final state. `dpo.run()` also rejects local+per_role,
and the two silent-failure launch derivations are pinned: the FSDP shard degree pinned into worker opts
(mis-sized mesh ⇒ an unsharded single-device model built per worker, discovered only as a cluster-launch
OOM/crash) and the minted run_ts pinned exactly once (ray workers reloading with run_ts="" checkpoint into
the legacy flat dir ⇒ resume split-brain). Real training is GPU/parity-tested.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from cosmos_framework.posttrain.config.rl import PreferenceEnvConfig
from cosmos_framework.posttrain.schema.checkpoint import FULL_CHECKPOINT_COMPONENTS
from cosmos_framework.posttrain.tests.helpers.entry_loop import emulate_supported_atomic_noreplace
from cosmos_framework.posttrain.tests.helpers.entry_loop import loop_config as _config
from cosmos_framework.posttrain.tests.helpers.entry_loop import mock_loop_env as _mock_env
from cosmos_framework.posttrain.workers.worker_group import WorkerGroup

_SHARD = "model.config.parallelism.data_parallel_shard_degree"


def _stop_at_bind(monkeypatch: pytest.MonkeyPatch, captured: dict[str, Any]) -> None:
    """Run dpo.run() up to bind, capturing the opts handed to make_setup: no NCCL, run_ts fixed to a
    known value (so the pin has a target), no config dump, no real bind/loop/teardown."""
    from cosmos_framework.posttrain.entrypoints import dpo

    monkeypatch.setattr(
        dpo, "make_setup", lambda config_path, opts, roles: captured.update(opts=list(opts)) or (lambda: {})
    )

    def _capture_bind(device_groups: list[Any], **kwargs: Any) -> None:
        captured["device_groups"] = device_groups
        captured["bind_kwargs"] = kwargs

    monkeypatch.setattr("cosmos_framework.posttrain.control.placement.bind_device_groups", _capture_bind)
    monkeypatch.setattr("cosmos_framework.posttrain.utils.dist.init_local_process_group", lambda: None)
    monkeypatch.setattr("cosmos_framework.posttrain.utils.dist.destroy_process_group_if_initialized", lambda: None)
    monkeypatch.setattr("cosmos_framework.posttrain.checkpoint.resolve_run_ts", lambda *a, **k: "20260101000000")
    monkeypatch.setattr(dpo, "dump_resolved", lambda *a, **k: None)
    monkeypatch.setattr(dpo, "loop", lambda *a, **k: None)
    finished = SimpleNamespace(result=lambda: None)

    def _declare_wgs(_ranks: int) -> dict[str, WorkerGroup]:
        data = WorkerGroup("data", "env")
        policy = WorkerGroup("policy", "trainer")
        policy.finalize_checkpoints = lambda: finished
        return {"data": data, "policy": policy}

    monkeypatch.setattr(
        dpo,
        "declare_wgs",
        _declare_wgs,
    )


def _ray_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        grid=SimpleNamespace(backend="ray", grouping="per_role", ranks=2),
        rl=SimpleNamespace(run_ts="", resume="off", deterministic=False),
        job=SimpleNamespace(project="p", group="g", name="n"),
    )


@pytest.mark.L0
@pytest.mark.CPU
def test_dpo_local_per_role_is_rejected() -> None:
    from cosmos_framework.posttrain.entrypoints import dpo

    cfg = SimpleNamespace(grid=SimpleNamespace(backend="local", grouping="per_role"))
    with pytest.raises(ValueError, match="requires grid.backend=ray"):
        dpo.run(cfg, "cfg", [], "/tmp/out")


@pytest.mark.L0
@pytest.mark.CPU
def test_dpo_ray_derives_fsdp_shard_degree(monkeypatch: pytest.MonkeyPatch) -> None:
    """ray sizes its FSDP mesh from grid.ranks — the loop must pin data_parallel_shard_degree into the
    worker opts (else the reloaded worker builds an unsharded single-device model), and must NOT clobber a
    shard degree the user already passed."""
    from cosmos_framework.posttrain.entrypoints import dpo

    captured: dict = {}
    _stop_at_bind(monkeypatch, captured)
    config = _ray_cfg()
    config.rl.deterministic = True
    determinism_calls: list[None] = []
    monkeypatch.setattr(
        "cosmos_framework.posttrain.utils.determinism.enable_determinism",
        lambda: determinism_calls.append(None),
    )
    dpo.run(config, "cfg", [], "/tmp/out")
    assert determinism_calls == [None]
    assert f"{_SHARD}=2" in captured["opts"]  # derived from grid.ranks=2

    captured2: dict = {}
    _stop_at_bind(monkeypatch, captured2)
    dpo.run(_ray_cfg(), "cfg", [f"{_SHARD}=4"], "/tmp/out")  # user already set it
    assert f"{_SHARD}=4" in captured2["opts"] and f"{_SHARD}=2" not in captured2["opts"]
    assert sum(o.startswith(f"{_SHARD}=") for o in captured2["opts"]) == 1  # not duplicated


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    ("backend", "grouping", "expected_device", "expected_actor_groups"),
    [
        ("ray", "per_role", "gpu", [["data"], ["policy"]]),
        ("ray", "single", "gpu", [["data", "policy"]]),
        ("local", "single", "cpu", [["data", "policy"]]),
    ],
)
def test_dpo_run_lowers_grouping_to_explicit_device_groups(
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    grouping: str,
    expected_device: str,
    expected_actor_groups: list[list[str]],
) -> None:
    captured: dict[str, Any] = {}
    _stop_at_bind(monkeypatch, captured)
    config = _ray_cfg()
    config.grid.backend = backend
    config.grid.grouping = grouping

    from cosmos_framework.posttrain.entrypoints import dpo

    dpo.run(config, "cfg", [], "/tmp/out")

    device_groups = captured["device_groups"]
    assert [device_group.device for device_group in device_groups] == [expected_device] * len(device_groups)
    actual_actor_groups = [
        [worker.name for worker in actor_group.members]
        for device_group in device_groups
        for actor_group in device_group.actor_groups
    ]
    assert actual_actor_groups == expected_actor_groups
    assert len(device_groups) == (2 if backend == "ray" and grouping == "per_role" else 1)


@pytest.mark.L0
@pytest.mark.CPU
def test_dpo_pins_run_ts_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """The minted run_ts is pinned into the worker opts exactly once, so ray workers reload the SAME
    attempt dir as the driver (a missing pin ⇒ workers checkpoint into the legacy flat dir ⇒ resume
    split-brain)."""
    from cosmos_framework.posttrain.entrypoints import dpo

    captured: dict = {}
    _stop_at_bind(monkeypatch, captured)
    dpo.run(_ray_cfg(), "cfg", [], "/tmp/out")
    assert captured["opts"].count("rl.run_ts=20260101000000") == 1


@pytest.mark.L0
@pytest.mark.CPU
def test_dpo_driver_preflights_custom_dataset_factory_before_binding(monkeypatch: pytest.MonkeyPatch) -> None:
    """A DPO factory import typo must fail before placement just like an online source typo."""
    from cosmos_framework.posttrain.entrypoints import dpo

    captured: dict[str, Any] = {}
    _stop_at_bind(monkeypatch, captured)
    config = _ray_cfg()
    config.rl.data = PreferenceEnvConfig(factory="my_project.data.build_preferences")
    seen: list[str] = []
    monkeypatch.setattr(
        "cosmos_framework.posttrain.data.datasets.factory.validate_dataset_factory",
        lambda factory_path: seen.append(factory_path),
    )

    dpo.run(config, "cfg", [], "/tmp/out")

    assert seen == ["my_project.data.build_preferences"]


@pytest.mark.L0
@pytest.mark.CPU
def test_dpo_driver_pins_external_journal_successor_before_worker_setup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Every fresh worker loads the promoted journal successor, never the older source generation."""
    from cosmos_framework.posttrain.entrypoints import dpo

    captured: dict[str, Any] = {}
    _stop_at_bind(monkeypatch, captured)
    config = _ray_cfg()
    source_path = tmp_path / "archive" / "previous" / "checkpoints" / "iter_000001"
    source_path.mkdir(parents=True)
    source = str(source_path)
    config.rl.resume = source
    calls: list[tuple[str, str]] = []

    monkeypatch.setattr(
        "cosmos_framework.posttrain.checkpoint.resolve_resume",
        lambda _resume, _root: source,
    )

    def _resolve_successor(resume: str, target: str) -> str:
        calls.append((resume, target))
        return target

    monkeypatch.setattr(
        "cosmos_framework.posttrain.checkpoint.resolve_external_checkpoint_successor",
        _resolve_successor,
    )

    dpo.run(config, "cfg", [], "/tmp/out")

    assert len(calls) == 1 and calls[0][0] == source
    successor = calls[0][1]
    assert successor.endswith("/checkpoints/iter_000001")
    assert config.rl.resume == successor
    assert captured["opts"].count(f"rl.resume={successor}") == 1
    assert f"rl.resume={source}" not in captured["opts"]


@pytest.mark.L0
@pytest.mark.CPU
def test_dpo_run_propagates_checkpoint_finalization_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed final async write must fail the job after teardown, never return success."""
    from cosmos_framework.posttrain.entrypoints import dpo

    captured: dict[str, Any] = {}
    _stop_at_bind(monkeypatch, captured)

    def _fail_finalize() -> None:
        raise RuntimeError("async checkpoint disk full")

    failed = SimpleNamespace(result=_fail_finalize)

    def _declare_failed_wgs(_ranks: int) -> dict[str, WorkerGroup]:
        data = WorkerGroup("data", "env")
        policy = WorkerGroup("policy", "trainer")
        policy.finalize_checkpoints = lambda: failed
        return {"data": data, "policy": policy}

    monkeypatch.setattr(
        dpo,
        "declare_wgs",
        _declare_failed_wgs,
    )

    with pytest.raises(RuntimeError, match="async checkpoint disk full"):
        dpo.run(_ray_cfg(), "cfg", [], "/tmp/out")


# --------------------------------------------------------------------------- loop contract (invariant pins)
def _units(policy: Any, n: dict[str, int]) -> dict[str, Any]:
    """The DPO loop's two roles: a preference `data` env whose get_samples is counted (a fresh pull per
    micro-batch is the invariant) + the given policy. No rollout/rm/replay — DPO is offline, no weight_sync."""

    def _get_samples(k: int) -> list[Any]:
        del k
        n["get_samples"] = n.get("get_samples", 0) + 1
        return []

    # `state_dict` is the source-cursor hook the checkpoint path pulls and hands to `policy.save`.
    return {
        "data": SimpleNamespace(
            get_samples=_get_samples,
            state_dict=lambda: None,
            loaded_checkpoint_provenance=lambda: policy.loaded_checkpoint_provenance(),
        ),
        "policy": policy,
    }


@pytest.mark.L0
@pytest.mark.CPU
def test_dpo_loop_step_is_one_optimizer_step(monkeypatch: pytest.MonkeyPatch) -> None:
    """grad_accum=2 across 3 steps → 6 train micro-batches with a FRESH get_samples each, but 3 optimizer
    steps (policy.iteration()==3) — reporting the last micro-batch only would draw the curves from
    1/grad_accum of the data."""
    from cosmos_framework.posttrain.entrypoints import dpo

    _mock_env(monkeypatch)
    n = {"train": 0}

    class _Policy:
        def __init__(self) -> None:
            self._it, self._acc, self.grad_accum = 0, 0, 2

        def iteration(self) -> int:
            return self._it

        def train(self, batch: list[Any]) -> dict[str, float]:
            del batch
            n["train"] += 1
            self._acc += 1
            if self._acc == self.grad_accum:  # steps the optimizer + bumps iteration on the last micro-batch
                self._acc, self._it = 0, self._it + 1
            return {"train/loss_avg": 0.5}

        @staticmethod
        def save(path: str, data_state: Any = None, **kwargs: Any) -> None:
            del path, data_state, kwargs

    policy = _Policy()
    summary = dpo.loop(
        _units(policy, n), _config(max_steps=3, grad_accum=2, pairs_per_step=1, save_every=0), "/tmp/out"
    )
    assert n["train"] == 6 and n["get_samples"] == 6  # a fresh pair pull per micro-batch, not one batch sliced
    assert policy.iteration() == 3  # 3 optimizer steps, not 6
    assert summary["steps"] == 3


@pytest.mark.L0
@pytest.mark.CPU
def test_dpo_loop_nan_step_aborts_before_report_and_save(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-finite train/loss_avg raises BEFORE the step is reported or checkpointed — a NaN/inf must never be
    logged as a real data point or saved. Step 1 is NaN (grad_accum=1): the finite step 0 reports+saves, the
    NaN step does NEITHER, and the run aborts."""
    from cosmos_framework.posttrain.entrypoints import dpo

    _mock_env(monkeypatch)
    reported: list[int] = []
    monkeypatch.setattr(  # spy reporter over _mock_env's empty default — proves the NaN step never reports
        "cosmos_framework.posttrain.utils.report.make_reporters",
        lambda *a, **k: [
            SimpleNamespace(start=lambda: None, report=lambda m, step: reported.append(step), finish=lambda: None)
        ],
    )
    n = {"train": 0}
    saves: list[str] = []

    class _Policy:
        def __init__(self) -> None:
            self._iteration = 0

        def iteration(self) -> int:
            return self._iteration

        def train(self, batch: list[Any]) -> dict[str, float]:
            del batch
            n["train"] += 1
            loss = float("nan") if n["train"] == 2 else 0.5
            if math.isfinite(loss):
                self._iteration += 1
            return {"train/loss_avg": loss}  # NaN on the 2nd step

        def save(self, path: str, data_state: Any = None, **kwargs: Any) -> None:
            del data_state, kwargs
            saves.append(path)

    with pytest.raises(RuntimeError, match="non-finite"):
        dpo.loop(_units(_Policy(), n), _config(max_steps=3, grad_accum=1, pairs_per_step=1, save_every=1), "/tmp/out")
    assert n["train"] == 2  # step 0 (finite) + step 1 (NaN, raises)
    assert reported == [1] and saves == ["/tmp/out/checkpoints/iter_000001"]  # only model step 1


@pytest.mark.L0
@pytest.mark.CPU
def test_dpo_loop_resumes_from_iteration(monkeypatch: pytest.MonkeyPatch) -> None:
    """A resumed loop starts at policy.iteration() and runs only the remaining steps (never re-runs 0..N,
    never rewrites an iter_* at or below the resume point). iteration()==2, max_steps=3 → start==2, one step's
    train, and the only save is strictly above the resume point."""
    from cosmos_framework.posttrain.entrypoints import dpo

    _mock_env(monkeypatch)
    n = {"train": 0}
    saves: list[str] = []

    class _Policy:
        def __init__(self) -> None:
            self._iteration = 2

        def iteration(self) -> int:
            return self._iteration

        def train(self, batch: list[Any]) -> dict[str, float]:
            del batch
            n["train"] += 1
            self._iteration += 1
            return {"train/loss_avg": 0.5}

        def save(self, path: str, data_state: Any = None, **kwargs: Any) -> None:
            del data_state, kwargs
            saves.append(path)

    summary = dpo.loop(
        _units(_Policy(), n), _config(max_steps=3, grad_accum=1, pairs_per_step=1, save_every=1), "/tmp/out"
    )
    assert summary["start"] == 2 and n["train"] == 1  # only step 2 ran (grad_accum=1)
    assert saves == ["/tmp/out/checkpoints/iter_000003"]  # saved step 3 only; nothing at/below the resume point


@pytest.mark.L0
@pytest.mark.CPU
def test_dpo_loop_preemption_emergency_saves(monkeypatch: pytest.MonkeyPatch) -> None:
    """A termination signal at the step-2 boundary (save_every=0, nothing saved yet) triggers exactly one
    emergency save at iter_000003 (step+1, 1-based) and breaks the loop."""
    from cosmos_framework.posttrain.entrypoints import dpo

    _mock_env(monkeypatch, stops=[None, None, "signal"])  # fires after step 2
    n = {"train": 0}
    saves: list[str] = []

    class _Policy:
        def __init__(self) -> None:
            self._iteration = 0

        def iteration(self) -> int:
            return self._iteration

        def train(self, batch: list[Any]) -> dict[str, float]:
            del batch
            n["train"] += 1
            self._iteration += 1
            return {"train/loss_avg": 0.5}

        def save(self, path: str, data_state: Any = None, **kwargs: Any) -> None:
            del data_state, kwargs
            saves.append(path)

    summary = dpo.loop(
        _units(_Policy(), n), _config(max_steps=5, grad_accum=1, pairs_per_step=1, save_every=0), "/tmp/out"
    )
    assert len(saves) == 1 and saves[0].endswith("iter_000003")
    assert n["train"] == 3 and summary["preempted"] == "signal"  # broke at step 2 of 5


@pytest.mark.L0
@pytest.mark.CPU
def test_dpo_loop_preemption_no_double_save_when_already_checkpointed(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the step-boundary cadence already saved this step, preemption does NOT re-save it — it just
    exits (no torn/duplicate write)."""
    from cosmos_framework.posttrain.entrypoints import dpo

    _mock_env(monkeypatch, stops=[None, None, "signal"])
    n = {"train": 0}
    saves: list[str] = []

    class _Policy:
        def __init__(self) -> None:
            self._iteration = 0

        def iteration(self) -> int:
            return self._iteration

        def train(self, batch: list[Any]) -> dict[str, float]:
            del batch
            n["train"] += 1
            self._iteration += 1
            return {"train/loss_avg": 0.5}

        def save(self, path: str, data_state: Any = None, **kwargs: Any) -> None:
            del data_state, kwargs
            saves.append(path)

    summary = dpo.loop(
        _units(_Policy(), n), _config(max_steps=5, grad_accum=1, pairs_per_step=1, save_every=1), "/tmp/out"
    )
    assert saves[-1].endswith("iter_000003") and saves.count(saves[-1]) == 1  # cadence saved step 2 once, not twice
    assert n["train"] == 3 and summary["preempted"] == "signal"


def _dpo_preemption_plan_saves(
    monkeypatch: pytest.MonkeyPatch, **rl_overrides: object
) -> list[tuple[str, tuple[str, ...] | None]]:
    """Run the DPO loop to a termination trigger at the step-2 boundary (save_every=0) under a model-only
    cadence plan; return every policy save as (iter dir name, `components` override or None). None = the
    trainer's configured plan (the cadence call shape); a tuple = an explicit override."""
    from cosmos_framework.posttrain.entrypoints import dpo

    _mock_env(monkeypatch, stops=[None, None, "signal"])  # fires after step 2
    n = {"train": 0}
    saves: list[tuple[str, tuple[str, ...] | None]] = []

    class _Policy:
        def __init__(self) -> None:
            self._iteration = 0

        def iteration(self) -> int:
            return self._iteration

        def train(self, batch: list[Any]) -> dict[str, float]:
            del batch
            n["train"] += 1
            self._iteration += 1
            return {"train/loss_avg": 0.5}

        def save(self, path: str, data_state: Any = None, **kwargs: Any) -> None:
            del data_state
            saves.append((Path(path).name, kwargs.get("components")))

    with pytest.warns(UserWarning, match="partial checkpoint artifact"):  # the model-only cadence plan
        summary = dpo.loop(
            _units(_Policy(), n),
            _config(
                max_steps=5,
                grad_accum=1,
                pairs_per_step=1,
                save_every=0,
                checkpoint_components=["model"],
                **rl_overrides,
            ),
            "/tmp/out",
        )
    assert n["train"] == 3 and summary["preempted"] == "signal"  # broke at step 2 of 5
    return saves


@pytest.mark.L0
@pytest.mark.CPU
def test_dpo_loop_preemption_emergency_save_honours_the_run_plan_when_opted_in(monkeypatch: pytest.MonkeyPatch) -> None:
    """`rl.emergency_checkpoint_components="run"` with a model-only cadence plan: the termination trigger writes
    exactly ONE save through the cadence path (the configured plan, no `components` override) instead of the
    forced full plan."""
    saves = _dpo_preemption_plan_saves(monkeypatch, emergency_checkpoint_components="run")
    assert saves == [("iter_000003", None)], f"emergency save wrote {saves}; expected the run's own (model-only) plan"


@pytest.mark.L0
@pytest.mark.CPU
def test_dpo_loop_preemption_emergency_save_defaults_to_the_full_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without opting in, a model-only cadence plan still gets the forced full plan at a termination trigger."""
    saves = _dpo_preemption_plan_saves(monkeypatch)
    assert saves == [("iter_000003", FULL_CHECKPOINT_COMPONENTS)], f"emergency save wrote {saves}"


@pytest.mark.L0
@pytest.mark.CPU
def test_dpo_loop_run_to_completion_saves_final_off_cadence(monkeypatch: pytest.MonkeyPatch) -> None:
    """A run that completes with the last step OFF the save_every cadence still saves the end state —
    otherwise the final weights are lost. steps=3, save_every=2 → cadence saves iter_000002, then the
    completion path saves iter_000003 (the final step the cadence skipped)."""
    from cosmos_framework.posttrain.entrypoints import dpo

    _mock_env(monkeypatch)
    n = {"train": 0}
    saves: list[str] = []

    class _Policy:
        def __init__(self) -> None:
            self._iteration = 0

        def iteration(self) -> int:
            return self._iteration

        def train(self, batch: list[Any]) -> dict[str, float]:
            del batch
            n["train"] += 1
            self._iteration += 1
            return {"train/loss_avg": 0.5}

        def save(self, path: str, data_state: Any = None, **kwargs: Any) -> None:
            del data_state, kwargs
            saves.append(path)

    summary = dpo.loop(
        _units(_Policy(), n), _config(max_steps=3, grad_accum=1, pairs_per_step=1, save_every=2), "/tmp/out"
    )
    assert [s.split("/")[-1] for s in saves] == ["iter_000002", "iter_000003"]  # cadence@2 + final@3 (off-cadence)
    assert n["train"] == 3 and "preempted" not in summary  # clean completion, no preemption


# --------------------------------------------------------------------------- held-out validation lifecycle
class _RecordingReporter:
    """Small Reporter-compatible spy that preserves stage/commit row boundaries."""

    def __init__(self, rows: list[tuple[int, dict[str, Any]]]) -> None:
        self._rows = rows
        self._pending_step: int | None = None
        self._pending_metrics: dict[str, Any] = {}

    def start(self) -> None:
        pass

    def stage(self, metrics: dict[str, Any], step: int) -> None:
        if self._pending_step is None:
            self._pending_step = step
        else:
            assert self._pending_step == step
        assert not set(self._pending_metrics).intersection(metrics)
        self._pending_metrics.update(metrics)

    def commit(self, step: int, metrics: dict[str, Any] | None = None) -> None:
        if metrics is not None:
            self.stage(metrics, step)
        assert self._pending_step == step
        self._rows.append((step, dict(self._pending_metrics)))
        self._pending_step = None
        self._pending_metrics.clear()

    def report(self, metrics: dict[str, Any], step: int) -> None:
        self.commit(step, metrics)

    def finish(self) -> None:
        assert self._pending_step is None


class _ValidationPolicy:
    """Dependency-free trainer double for DPO validation control-plane tests."""

    def __init__(
        self,
        *,
        start: int = 0,
        fail_at: int | None = None,
        resumed: bool | None = None,
    ) -> None:
        self._iteration = start
        self._resumed = start > 0 if resumed is None else resumed
        self._pending_step: int | None = None
        self._pending_reasons: frozenset[Any] = frozenset()
        self._pending_metrics: dict[str, int | float | bool | str] = {}
        self._publication_pending_step: int | None = None
        self.fail_at = fail_at
        self.train_calls = 0
        self.mark_calls: list[tuple[int, frozenset[Any], dict[str, int | float | bool | str]]] = []
        self.validation_calls: list[tuple[int, frozenset[Any]]] = []
        self.complete_calls: list[int] = []
        self.ephemeral_complete_calls: list[int] = []
        self.saves: list[tuple[str, int | None, dict[str, int | float | bool | str], dict[str, Any]]] = []
        self.persisted: list[tuple[str, int | None]] = []
        self._loaded_provenance: dict[str, int | str] | None = None

    def iteration(self) -> int:
        return self._iteration

    def resumed_from_checkpoint(self) -> bool:
        return self._resumed

    def loaded_checkpoint_provenance(self) -> dict[str, int | str] | None:
        return None if self._loaded_provenance is None else dict(self._loaded_provenance)

    def train(self, batch: list[Any]) -> dict[str, float]:
        del batch
        self.train_calls += 1
        self._iteration += 1
        return {"train/loss_avg": 0.5, "train/dpo_accuracy": 0.75}

    def pending_validation_step(self) -> int | None:
        return self._pending_step

    def pending_validation_metrics(self) -> dict[str, int | float | bool | str]:
        return dict(self._pending_metrics)

    def pending_validation_publication_step(self) -> int | None:
        return self._publication_pending_step

    def mark_validation_pending(
        self,
        model_step: int,
        reasons: frozenset[Any],
        metrics: dict[str, int | float | bool | str] | None = None,
    ) -> None:
        frame = dict(metrics or {})
        self.mark_calls.append((model_step, reasons, frame))
        assert self._publication_pending_step is None
        if self._pending_step is None:
            self._pending_step = model_step
            self._pending_reasons = reasons
            self._pending_metrics = frame
            return
        assert self._pending_step == model_step
        assert self._pending_metrics == frame

    def validate(self, model_step: int, reasons: frozenset[Any]) -> Any:
        from cosmos_framework.posttrain.control.validation import ValidationReport

        self.validation_calls.append((model_step, reasons))
        if self.fail_at == model_step:
            raise RuntimeError(f"held-out validation exploded at step {model_step}")
        return ValidationReport(
            model_step=model_step,
            sample_count=2,
            metrics={
                "val/loss_avg": 0.25,
                "val/pair_count": 2,
                "val/batch_count": 1,
                "timer/validation_s": 0.01,
            },
            source_fingerprint="held-out-preferences-v1",
            trigger_reasons=reasons,
            duration_s=0.01,
        )

    def complete_validation(self, model_step: int) -> None:
        assert self._pending_step == model_step
        assert self._publication_pending_step is None
        self.complete_calls.append(model_step)
        self._pending_step = None
        self._pending_reasons = frozenset()
        self._pending_metrics = {}
        self._publication_pending_step = model_step

    def complete_ephemeral_validation(self, model_step: int) -> None:
        assert self._pending_step == model_step
        assert self._publication_pending_step is None
        assert model_step == self._iteration
        self._pending_step = None
        self._pending_reasons = frozenset()
        self._pending_metrics = {}
        self.ephemeral_complete_calls.append(model_step)

    def save(self, path: str, data_state: Any = None, **kwargs: Any) -> None:
        del data_state
        self.saves.append((path, self._pending_step, dict(self._pending_metrics), dict(kwargs)))
        if kwargs.get("update_latest") is False:
            checkpoint = Path(path)
            checkpoint.mkdir(parents=True)
            for component in ("model.pt", "optim.pt", "trainer.pt", "data.pt"):
                (checkpoint / component).write_bytes(b"staged fake checkpoint\n")

    def save_validation_progress(self, path: str) -> None:
        assert self._publication_pending_step is not None
        self.persisted.append((path, self._pending_step))
        self._publication_pending_step = None


def _validation_config(
    *,
    steps: int,
    before_training: bool = True,
    every_steps: int = 2,
    resume: str = "off",
    save_checkpoint: bool = False,
) -> SimpleNamespace:
    validation = SimpleNamespace(
        enabled=True,
        before_training=before_training,
        every_steps=every_steps,
        save_checkpoint=save_checkpoint,
    )
    return _config(
        max_steps=steps,
        grad_accum=1,
        pairs_per_step=1,
        save_every=0,
        resume=resume,
        validation=validation,
    )


def _trigger_values(reasons: frozenset[Any]) -> set[str]:
    return {str(reason.value) for reason in reasons}


def _write_fake_external_checkpoint(path: Path) -> None:
    path.mkdir(parents=True)
    for component in ("model.pt", "optim.pt", "trainer.pt", "data.pt"):
        (path / component).write_bytes(b"external fake checkpoint\n")


@pytest.mark.L0
@pytest.mark.CPU
def test_dpo_validation_runs_before_periodic_and_final_without_training_on_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With saves enabled, step zero is ephemeral while periodic/final passes are durable."""
    from cosmos_framework.posttrain.entrypoints import dpo

    _mock_env(monkeypatch)
    rows: list[tuple[int, dict[str, Any]]] = []
    policy = _ValidationPolicy()
    persisted_before_report: list[int] = []

    class _Reporter(_RecordingReporter):
        def report(self, metrics: dict[str, Any], step: int) -> None:
            if "val/loss_avg" in metrics:
                assert policy.pending_validation_step() is None
                if step == 0:
                    assert policy.persisted == []
                else:
                    assert policy.persisted[-1][1] is None
                persisted_before_report.append(step)
            super().report(metrics, step)

    monkeypatch.setattr(
        "cosmos_framework.posttrain.utils.report.make_reporters",
        lambda *_args, **_kwargs: [_Reporter(rows)],
    )
    n: dict[str, int] = {}

    dpo.loop(_units(policy, n), _validation_config(steps=3, save_checkpoint=True), "/tmp/out")

    assert [(step, _trigger_values(reasons)) for step, reasons in policy.validation_calls] == [
        (0, {"before_training"}),
        (2, {"periodic"}),
        (3, {"final"}),
    ]
    assert policy.train_calls == 3 and n["get_samples"] == 3
    assert [(step, _trigger_values(reasons)) for step, reasons, _metrics in policy.mark_calls] == [
        (0, {"before_training"}),
        (2, {"periodic"}),
        (3, {"final"}),
    ]
    assert policy.mark_calls[0][2] == {}
    assert all("train/loss_avg" in metrics for _step, _reasons, metrics in policy.mark_calls[1:])
    assert policy.complete_calls == [2, 3]
    assert policy.ephemeral_complete_calls == [0]
    assert [Path(path).name for path, *_rest in policy.saves] == ["iter_000002", "iter_000003"]
    assert [step for step, _metrics in rows] == [0, 1, 2, 3]
    assert "train/loss_avg" in dict(rows)[2] and "val/loss_avg" in dict(rows)[2]
    assert all(not key.startswith("dpo_") for _step, metrics in rows for key in metrics)
    assert persisted_before_report == [0, 2, 3]


@pytest.mark.L0
@pytest.mark.CPU
def test_dpo_validation_does_not_force_checkpoint_when_save_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Periodic and final validation remain in memory unless explicitly configured to save."""
    from cosmos_framework.posttrain.entrypoints import dpo

    _mock_env(monkeypatch)
    policy = _ValidationPolicy()

    dpo.loop(
        _units(policy, {}),
        _validation_config(steps=2, before_training=False, every_steps=1),
        "/tmp/out",
    )

    assert [(step, _trigger_values(reasons)) for step, reasons in policy.validation_calls] == [
        (1, {"periodic"}),
        (2, {"periodic", "final"}),
    ]
    assert policy.saves == []
    assert policy.persisted == []
    assert policy.ephemeral_complete_calls == [1, 2]


@pytest.mark.L0
@pytest.mark.CPU
def test_dpo_preemption_before_ephemeral_initial_validation_does_not_save(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fresh retry can rerun initial validation from source weights, so step zero stays checkpoint-free."""
    from cosmos_framework.posttrain.entrypoints import dpo

    _mock_env(monkeypatch, stops=["signal"])
    policy = _ValidationPolicy()

    summary = dpo.loop(_units(policy, {}), _validation_config(steps=1), "/tmp/out")

    assert summary["preempted"] == "signal"
    assert policy.train_calls == 0
    assert policy.validation_calls == []
    assert policy.mark_calls == []
    assert policy.saves == []
    assert policy.complete_calls == []
    assert policy.ephemeral_complete_calls == []
    assert policy.persisted == []


@pytest.mark.L0
@pytest.mark.CPU
def test_dpo_zero_step_final_validation_is_durable_when_checkpoint_save_is_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """At max_steps=0, an enabled save makes the final step use the resumable transaction."""
    from cosmos_framework.posttrain.control.validation import ValidationTrigger
    from cosmos_framework.posttrain.entrypoints import dpo

    _mock_env(monkeypatch)
    policy = _ValidationPolicy()

    dpo.loop(_units(policy, {}), _validation_config(steps=0, save_checkpoint=True), "/tmp/out")

    expected_reasons = frozenset({ValidationTrigger.BEFORE_TRAINING, ValidationTrigger.FINAL})
    assert policy.validation_calls == [(0, expected_reasons)]
    assert policy.mark_calls == [(0, expected_reasons, {})]
    assert [(Path(path).name, pending, options) for path, pending, _metrics, options in policy.saves] == [
        (
            "iter_000000",
            0,
            {"publish": False, "components": ("model", "optimizer", "training_state")},
        )
    ]
    assert policy.complete_calls == [0]
    assert policy.ephemeral_complete_calls == []
    assert policy.persisted == [("/tmp/out/checkpoints/iter_000000", None)]


@pytest.mark.L0
@pytest.mark.CPU
def test_dpo_pending_resume_validates_once_clears_and_persists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pending checkpoint retries once; completion is durable and a second resume does no work."""
    from cosmos_framework.posttrain.entrypoints import dpo

    _mock_env(monkeypatch, stops=["signal"])
    rows: list[tuple[int, dict[str, Any]]] = []
    monkeypatch.setattr(
        "cosmos_framework.posttrain.utils.report.make_reporters",
        lambda *_args, **_kwargs: [_RecordingReporter(rows)],
    )
    policy = _ValidationPolicy()
    config = _validation_config(steps=1, before_training=False, every_steps=1, save_checkpoint=True)

    first = dpo.loop(_units(policy, {}), config, "/tmp/out")
    assert first["preempted"] == "signal" and policy.pending_validation_step() == 1

    config.rl.resume = "/tmp/out/checkpoints/iter_000001"
    second = dpo.loop(_units(policy, {}), config, "/tmp/out")
    assert second["start"] == 1 and policy.pending_validation_step() is None
    assert [(step, _trigger_values(reasons)) for step, reasons in policy.validation_calls] == [(1, {"pending"})]
    assert policy.persisted == [("/tmp/out/checkpoints/iter_000001", None)]
    assert len(rows) == 1 and rows[0][0] == 1
    resumed_metrics = rows[0][1]
    assert resumed_metrics["train/loss_avg"] == 0.5
    assert resumed_metrics["train/dpo_accuracy"] == 0.75
    assert resumed_metrics["timer/e2e_step_s"] >= 0.0
    assert resumed_metrics["val/loss_avg"] == 0.25
    assert resumed_metrics["val/pair_count"] == 2
    assert resumed_metrics["val/batch_count"] == 1
    assert type(resumed_metrics["val/pair_count"]) is int
    assert type(resumed_metrics["val/batch_count"]) is int
    assert resumed_metrics["timer/validation_s"] == 0.01

    third = dpo.loop(_units(policy, {}), config, "/tmp/out")
    assert third["start"] == 1
    assert len(policy.validation_calls) == 1 and len(policy.persisted) == 1


@pytest.mark.L0
@pytest.mark.CPU
def test_dpo_publication_pending_resume_republishes_without_revalidating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A crash-window archive failure retries only publication at the restored model step."""
    from cosmos_framework.posttrain.entrypoints import dpo

    _mock_env(monkeypatch)
    rows: list[tuple[int, dict[str, Any]]] = []
    monkeypatch.setattr(
        "cosmos_framework.posttrain.utils.report.make_reporters",
        lambda *_args, **_kwargs: [_RecordingReporter(rows)],
    )

    class _FailFirstPublicationPolicy(_ValidationPolicy):
        def save_validation_progress(self, path: str) -> None:
            assert self._publication_pending_step is not None
            self.persisted.append((path, self._pending_step))
            if len(self.persisted) == 1:
                raise OSError("archive unavailable")
            self._publication_pending_step = None

    policy = _FailFirstPublicationPolicy()
    config = _validation_config(steps=1, before_training=False, every_steps=1, save_checkpoint=True)

    with pytest.raises(OSError, match="archive unavailable"):
        dpo.loop(_units(policy, {}), config, "/tmp/out")

    assert policy.pending_validation_step() is None
    assert policy.pending_validation_publication_step() == 1
    assert len(policy.validation_calls) == 1
    assert rows == []

    config.rl.resume = "/tmp/out/checkpoints/iter_000001"
    # Model a genuinely fresh process: only the durable publication journal is restored. No
    # in-memory validation result frame or objective-call history crosses the process boundary.
    resumed_policy = _FailFirstPublicationPolicy(start=1, resumed=True)
    resumed_policy._publication_pending_step = 1
    resumed_policy.persisted.append(("/tmp/out/checkpoints/iter_000001", None))
    resumed = dpo.loop(_units(resumed_policy, {}), config, "/tmp/out")

    assert resumed["start"] == 1
    assert resumed_policy.pending_validation_publication_step() is None
    assert resumed_policy.validation_calls == []
    assert resumed_policy.train_calls == 0
    assert len(policy.validation_calls) == 1
    assert resumed_policy.persisted == [
        ("/tmp/out/checkpoints/iter_000001", None),
        ("/tmp/out/checkpoints/iter_000001", None),
    ]
    assert rows == []


@pytest.mark.L0
@pytest.mark.CPU
def test_dpo_external_pending_resume_copies_into_current_attempt_before_completion(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    """An external pending successor stays full even when ordinary saves default to partial."""
    from cosmos_framework.posttrain.control.validation import ValidationTrigger
    from cosmos_framework.posttrain.entrypoints import dpo

    _mock_env(monkeypatch)
    emulate_supported_atomic_noreplace(monkeypatch)
    out_dir = str(tmp_path / "new-run")
    policy = _ValidationPolicy(start=1)
    policy.mark_validation_pending(1, frozenset({ValidationTrigger.PERIODIC}), {"train/loss_avg": 0.5})
    external_path = tmp_path / "old-run" / "checkpoints" / "iter_000001"
    _write_fake_external_checkpoint(external_path)
    from cosmos_framework.posttrain.checkpoint import checkpoint_provenance

    policy._loaded_provenance = checkpoint_provenance(str(external_path))
    external = str(external_path)
    config = _validation_config(
        steps=1,
        before_training=False,
        every_steps=1,
        resume=external,
        save_checkpoint=True,
    )
    config.rl.checkpoint_components = ["model"]
    units = _units(policy, {})
    units["data"].loaded_checkpoint_provenance = lambda: {
        **policy._loaded_provenance,
        "signature": "different-data-revision",
    }

    with pytest.warns(UserWarning, match="partial checkpoint artifact"):
        with pytest.raises(RuntimeError, match="data cursor and policy loaded different external checkpoint revisions"):
            dpo.loop(units, config, out_dir)

    units["data"].loaded_checkpoint_provenance = lambda: policy.loaded_checkpoint_provenance()
    with pytest.warns(UserWarning, match="partial checkpoint artifact"):
        dpo.loop(units, config, out_dir)

    current = tmp_path / "new-run" / "checkpoints" / "iter_000001"
    assert len(policy.saves) == 1
    staged, pending_step, pending_metrics, save_options = policy.saves[0]
    assert Path(staged).parent == current.parent and ".external-copy-" in Path(staged).name
    assert pending_step == 1 and pending_metrics == {"train/loss_avg": 0.5}
    assert save_options == {
        "publish": False,
        "update_latest": False,
        "components": ("model", "optimizer", "training_state"),
    }
    assert current.is_dir()
    assert policy.persisted == [(str(current), None)]
    assert policy.pending_validation_step() is None


@pytest.mark.L0
@pytest.mark.CPU
def test_dpo_validation_surfaces_nonroot_cell_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A late non-root failure cannot be hidden behind CollectiveHandle.result()."""
    from cosmos_framework.posttrain.entrypoints import dpo

    _mock_env(monkeypatch)
    rows: list[tuple[int, dict[str, Any]]] = []
    monkeypatch.setattr(
        "cosmos_framework.posttrain.utils.report.make_reporters",
        lambda *_args, **_kwargs: [_RecordingReporter(rows)],
    )

    class _NonRootFailure:
        def __init__(self, root_value: Any) -> None:
            self._root_value = root_value

        def result(self) -> Any:
            return self._root_value

        def all(self) -> dict[tuple[int, int], Any]:
            raise RuntimeError("validation failed on replica 0 rank 1")

    class _Policy(_ValidationPolicy):
        def validate(self, model_step: int, reasons: frozenset[Any]) -> Any:
            return _NonRootFailure(super().validate(model_step, reasons))

    policy = _Policy()
    with pytest.raises(RuntimeError, match="replica 0 rank 1"):
        dpo.loop(
            _units(policy, {}),
            _validation_config(steps=1, before_training=False, every_steps=1, save_checkpoint=True),
            "/tmp/out",
        )

    assert policy.complete_calls == [] and policy.pending_validation_step() == 1
    assert rows == []


@pytest.mark.L0
@pytest.mark.CPU
def test_dpo_forwards_explicit_gpu_topology_to_shared_binder(monkeypatch: pytest.MonkeyPatch) -> None:
    from cosmos_framework.posttrain.entrypoints import dpo

    captured: dict[str, Any] = {}
    _stop_at_bind(monkeypatch, captured)
    cfg = _ray_cfg()
    cfg.grid.gpu_node_domains = {"worker-1": "rack-a"}
    dpo.run(cfg, "cfg", [], "/tmp/out")
    assert captured["bind_kwargs"]["gpu_node_domains"] == cfg.grid.gpu_node_domains


@pytest.mark.L0
@pytest.mark.CPU
def test_dpo_loop_emergency_run_plan_after_off_cadence_validation_still_writes_the_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Reproduction: `rl.emergency_checkpoint_components="run"`, run
    components `[model]`, validation components `[training_state]`, `save_every=0` (every validation is off-cadence), and
    the termination signal is seen at the step-2 boundary right after the periodic validation committed its
    training-state artifact for step 2. The parent logged "already checkpointed, exiting" and wrote no weights. Now the
    step's model is written: the training-state artifact of step 2 is committed and cannot be re-committed, so the
    emergency save is the full plan under `checkpoints/iter_000002`. The real config refuses this combination
    (`RunConfig.__post_init__`: under "run" a saving validation plan must include the model, so the validation artifact
    stands in for the emergency save); this loop drives a raw namespace config, so the coordinator's fallback is what is
    exercised here: the weights are written, never skipped."""
    from cosmos_framework.posttrain.entrypoints import dpo

    # should_stop is polled once by the loop at every step boundary and once more by the validation boundary BEFORE a
    # due validation (a signal seen there defers the validation and takes the full-plan path): step 1 -> None; step 2
    # boundary check -> None, the validation runs and commits its artifact; the loop's poll after it -> "signal".
    _mock_env(monkeypatch, stops=[None, None, "signal"])
    policy = _ValidationPolicy()
    validation = SimpleNamespace(
        enabled=True,
        before_training=False,
        every_steps=2,
        save_checkpoint=True,
        checkpoint_components=["training_state"],
    )
    with pytest.warns(UserWarning, match="partial checkpoint artifact"):
        summary = dpo.loop(
            _units(policy, {}),
            _config(
                max_steps=5,
                grad_accum=1,
                pairs_per_step=1,
                save_every=0,
                checkpoint_components=["model"],
                emergency_checkpoint_components="run",
                validation=validation,
            ),
            str(tmp_path),  # the partial validation save writes fake files: keep them out of the shared /tmp/out
        )
    assert summary["preempted"] == "signal" and policy.train_calls == 2
    assert [(step, _trigger_values(reasons)) for step, reasons in policy.validation_calls] == [(2, {"periodic"})]
    saves = [(Path(path).name, kwargs.get("components")) for path, _pending, _metrics, kwargs in policy.saves]
    # The validation's training-state artifact, then the emergency save carrying the model.
    assert saves[0] == ("iter_000002", ("training_state",))
    assert len(saves) == 2, f"the parent wrote only the validation artifact: {saves}"
    assert saves[1][0] == "iter_000002" and "model" in saves[1][1]
    assert saves[1] == ("iter_000002", FULL_CHECKPOINT_COMPONENTS)
