# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""GR00T entry lifecycle with real local collection, optimization and checkpoint state."""

import gc
import json
import logging
import signal
import sys
import threading
import time
import weakref
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from cosmos_framework.posttrain.config.rl import JointPPOConfig
from cosmos_framework.posttrain.configs.gr00t_n1d7_libero_ppo import make_config
from cosmos_framework.posttrain.embodied.sims.libero.protocol import fixed_manifest
from cosmos_framework.posttrain.entrypoints import joint_ppo as gr00t_ppo
from cosmos_framework.posttrain.roles.env.ppo_window import PPOWindowCollector
from cosmos_framework.posttrain.roles.trainer.joint_ppo import JointPPOTrainer
from cosmos_framework.posttrain.tests.helpers.ppo_continuous import EvaluationIntegratorGateway, GaussianPolicy
from cosmos_framework.posttrain.utils.entry import RunAttempt, resolve_all
from cosmos_framework.posttrain.utils.log import logger
from cosmos_framework.posttrain.utils.report import ConsoleReporter, make_reporters
from cosmos_framework.posttrain.workers.worker_group import WorkerGroup

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


@pytest.fixture
def local_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import ray

    from cosmos_framework.posttrain.control.checkpointing import CheckpointLifecycle
    from cosmos_framework.posttrain.control.loop import weight_sync
    from cosmos_framework.posttrain.utils import entry, mesh, report

    events, allocated = [], []
    config = make_config()
    config.model.checkpoint_path, config.environment.config.sim_python, config.job.wandb_mode = (
        "gaussian",
        "python",
        "disabled",
    )
    config.rl = JointPPOConfig(
        num_envs=2, window_decisions=2, global_minibatch=2, microbatch=1, max_steps=4, save_every=0
    )
    config.rl.validation.enabled = False
    for split in ("train", "eval"):
        path = tmp_path / f"{split}.json"
        fixed_manifest(split).write(path)
        setattr(config.environment.config, f"{split}_manifest", str(path))
    attempt = {"output": tmp_path / "first", "resume": None}

    class Gateway(EvaluationIntegratorGateway):
        def __init__(self, name: str) -> None:
            super().__init__()
            self.name = name
            self.stream = (tmp_path / f"{name}-resource").open("w")

        def close(self) -> None:
            events.append(self.name)
            self.stream.close()
            super().close()

    class Collector(PPOWindowCollector):
        def close(self) -> None:
            events.append("collector")
            super().close()

    class Rollout(GaussianPolicy):
        def __init__(self) -> None:
            super().__init__()
            self.stream = (tmp_path / "rollout-resource").open("w")

        def close(self) -> None:
            events.append("rollout")
            self.stream.close()

    class RecordingReporter(ConsoleReporter):
        def finish(self) -> None:
            events.append("reporters")
            super().finish()

    def assemble(config, recipe, output, resume, width, *, evaluation=None):
        learner = JointPPOTrainer(GaussianPolicy(), recipe)
        rollout, gateway = Rollout(), Gateway("sim")
        evaluation_enabled = recipe.validation.enabled if evaluation is None else evaluation
        evaluation_gateway = Gateway("eval_sim") if evaluation_enabled else None
        from cosmos_framework.utils.lazy_config import instantiate
        from cosmos_framework.posttrain.control.buffer import ExperienceBuffer
        from cosmos_framework.posttrain.roles.reward.worker import RewardWorker

        environment = instantiate(config.environment)
        collector = Collector(recipe, 0, 2, environment=environment)
        reward = RewardWorker([instantiate(config.reward)], training_group_size=1)
        replay = ExperienceBuffer(exact_batch_size=recipe.decisions)
        if resume:
            learner.load(resume)
            state = learner.checkpoint_manager.load_component(resume, "data")
            collector.load_state_dict(state["collectors"]["0"])
        if recipe.validation.enabled:
            learner.configure_validation(gr00t_ppo.validation_identity(config))
        allocated.append((learner, collector, rollout, gateway, evaluation_gateway))
        units = {
            "collector": WorkerGroup("collector", "env", body=collector),
            "rollout": WorkerGroup("rollout", "rollout", body=rollout),
            "sim": WorkerGroup("sim", "env", body=gateway),
            "policy": WorkerGroup("policy", "trainer", body=learner),
            "rm": WorkerGroup("rm", "reward", body=reward),
            "replay": WorkerGroup("replay", "buffer", body=replay),
        }
        if evaluation_gateway is not None:
            units["eval_sim"] = WorkerGroup("eval_sim", "env", body=evaluation_gateway)
        resolve_all(units["collector"].bind_rollout(units["rollout"]))
        return units

    def publish(rollout, policy, *, validation_step):
        rollout.body.net.load_state_dict(policy.body.model.net.state_dict())
        rollout.body.version = validation_step

    finalize = CheckpointLifecycle.finalize_units
    shutdown = ray.shutdown

    def finalize_units(units):
        events.append("checkpoint")
        finalize(units)

    def shutdown_ray():
        events.append("ray")
        shutdown()

    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setattr(mesh, "detect_mesh", lambda: (1, 2))
    monkeypatch.setattr(entry, "pin_run_attempt", lambda *_: RunAttempt(str(attempt["output"]), (), attempt["resume"]))
    monkeypatch.setattr(gr00t_ppo, "assemble", assemble)
    monkeypatch.setattr(weight_sync, "weight_sync", publish)
    monkeypatch.setattr(report, "make_reporters", lambda *_args, **_kwargs: [RecordingReporter()])
    monkeypatch.setattr(CheckpointLifecycle, "finalize_units", finalize_units)
    monkeypatch.setattr(ray, "shutdown", shutdown_ray)
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGUSR1, signal.SIGTERM)}
    try:
        yield config, attempt, events, allocated
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        for _, _, rollout, gateway, evaluation_gateway in allocated:
            # Close even when a failed assertion interrupts the test itself.
            rollout.stream.close()
            gateway.stream.close()
            if evaluation_gateway is not None:
                evaluation_gateway.stream.close()


def test_window_counter_is_per_process_and_preserves_durable_window_rows(
    local_run, one_logger_sink, monkeypatch
) -> None:
    from cosmos_framework.posttrain.control.checkpointing import CheckpointLifecycle

    config, attempt, events, allocated = local_run
    coordinators = []
    retained_telemetry = []
    build_coordinator = CheckpointLifecycle.coordinator

    def capture(*args, **kwargs):
        coordinator = build_coordinator(*args, **kwargs)
        coordinators.append(weakref.ref(coordinator))
        retained_telemetry.append(kwargs["observer"])
        return coordinator

    monkeypatch.setattr(CheckpointLifecycle, "coordinator", capture)
    first = gr00t_ppo.run(config)
    # Discard unrelated unrooted frames while deliberately keeping telemetry alive:
    # a strong checkpoint callback would still retain its coordinator through this root.
    gc.collect()
    assert len(retained_telemetry) == len(coordinators) == 1 and coordinators[0]() is None
    assert first["windows"] == 2 and first["optimizer_step"] == 4 and first["publication_version"] == 2
    counters = ("train_iterations_start", "train_iterations_end", "train_samples_start", "train_samples_end")
    assert tuple(one_logger_sink[0].store[key] for key in counters) == (
        0,
        first["publication_version"],
        0,
        first["windows"] * config.rl.decisions,
    )
    rows = [json.loads(line) for line in (attempt["output"] / "windows.jsonl").read_text().splitlines()]
    assert [row["optimizer_step"] for row in rows] == [2, 4]
    checkpoint_metrics = [frame for frame in one_logger_sink[0].metrics if "joint_ppo_checkpoint_path" in frame]
    assert checkpoint_metrics == [
        {
            "joint_ppo_checkpoint_path": str(attempt["output"] / "checkpoints/iter_000004"),
            "joint_ppo_checkpoint_optimizer_step": first["optimizer_step"],
        }
    ]
    assert events[-6:] == ["checkpoint", "reporters", "sim", "collector", "rollout", "ray"]
    attempt["resume"] = str(attempt["output"] / "checkpoints/iter_000004")
    attempt["output"] = attempt["output"].parent / "restored"
    config.rl.max_steps = 6
    second = gr00t_ppo.run(config)
    assert second["windows"] == 1 and second["optimizer_step"] == 6 and second["publication_version"] == 3
    assert len(one_logger_sink) == 2 and all(sink.finishes == 1 for sink in one_logger_sink)
    assert tuple(one_logger_sink[1].store[key] for key in counters) == (
        first["publication_version"],
        second["publication_version"],
        first["publication_version"] * config.rl.decisions,
        second["publication_version"] * config.rl.decisions,
    )
    assert all(rollout.stream.closed and gateway.stream.closed for _, _, rollout, gateway, _ in allocated)


@pytest.mark.parametrize("failure", ["counter", "loop", "collector_close"])
def test_training_cleanup_covers_early_and_late_failures(
    local_run, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    config, _, events, allocated = local_run
    assemble = gr00t_ppo.assemble

    def inject(*args, **kwargs):
        units = assemble(*args, **kwargs)
        collector, policy = units["collector"], units["policy"]
        if failure == "counter":
            policy.body.progress = lambda: {"optimizer_step": 1, "publication_version": 0}
        elif failure == "loop":

            def fail_window(version):
                raise RuntimeError("injected collection failure")

            collector.body.begin_window = fail_window
        elif failure == "collector_close":
            close = collector.body.close

            def fail_close():
                close()
                raise RuntimeError("injected close failure")

            collector.body.close = fail_close
        return units

    monkeypatch.setattr(gr00t_ppo, "assemble", inject)
    with pytest.raises((RuntimeError, ValueError)) as retained:
        gr00t_ppo.run(config)
    assert retained.value.__traceback__ is not None
    expected = ["checkpoint", "sim", "collector", "rollout", "ray"]
    if failure in ("loop", "collector_close"):
        expected.insert(1, "reporters")
    assert events[-len(expected) :] == expected
    assert all(rollout.stream.closed and gateway.stream.closed for _, _, rollout, gateway, _ in allocated)


def test_assessment_cleanup_and_complete_reentry_leave_training_untouched(local_run) -> None:
    from cosmos_framework.posttrain.embodied.sims.libero.protocol import LiberoCase, LiberoTrial, LiberoTrialManifest
    from cosmos_framework.posttrain.tests.libero_assessment_cli import run_assessment

    config, attempt, events, allocated = local_run
    directory = attempt["output"].parent
    manifest = directory / "trials.json"
    LiberoTrialManifest("libero_spatial", (LiberoTrial(LiberoCase(3, 1), 7),)).write(manifest)
    result = run_assessment(config, manifest=str(manifest), output_dir=str(directory / "assessment"))
    assert result["complete"] and result["completed_trials"] == 1
    assert result["optimizer_step"] == 0 and result["publication_version"] == 0
    assert events[-7:] == ["reporters", "checkpoint", "eval_sim", "sim", "collector", "rollout", "ray"]
    events.clear()
    assert run_assessment(config, manifest=str(manifest), output_dir=str(directory / "assessment")) == result
    assert events == ["checkpoint", "eval_sim", "sim", "collector", "rollout", "ray"]
    assert allocated[-1][-1].evaluated == []


def test_formal_window_validation_runs_before_and_after_training_without_analysis(
    local_run, one_logger_sink, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cosmos_framework.posttrain.utils import mesh

    config, attempt, _, allocated = local_run
    # The local Gaussian fixture has one real collection replica and no GR00T hardware constraint.
    monkeypatch.setattr(mesh, "detect_mesh", lambda: (1, 1))
    config.recipe_check = None
    config.rl.validation.enabled = True
    config.rl.validation.save_checkpoint = False
    result = gr00t_ppo.run(config)
    directory = attempt["output"] / "evaluation"
    reports = [json.loads(path.read_text()) for path in sorted(directory.glob("step_*.json"))]
    assert [report["optimizer_step"] for report in reports] == [0, result["optimizer_step"]]
    assert all(report["training_state_unchanged"] for report in reports)
    assert all(report["metrics"]["val/episode_count"] == len(fixed_manifest("eval").cases) for report in reports)
    assert len(allocated[-1][-1].evaluated) == sum(report["metrics"]["val/episode_count"] for report in reports)
    assert not (directory / "baseline-check.json").exists()
    assert len(one_logger_sink) == 1 and one_logger_sink[0].finishes == 1
    assert one_logger_sink[0].store["validation_iterations"] == len(reports)
    assert any(frame.get("validation_iterations_time_total", 0) > 0 for frame in one_logger_sink[0].metrics)


def test_online_reporting_failure_does_not_block_training(
    local_run, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import wandb

    from cosmos_framework.posttrain.utils import report

    config, _, _, allocated = local_run
    monkeypatch.setenv("WANDB_MODE", "online")
    monkeypatch.setenv("WANDB_PROJECT", "cosmos3_action")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setattr(logger, "propagate", True)
    caplog.set_level(logging.INFO, logger="posttrain")
    attempts = []

    def fail_init(**kwargs):
        attempts.append(kwargs)
        raise wandb.errors.CommError("injected reporting authentication failure")

    monkeypatch.setattr(wandb, "init", fail_init)
    monkeypatch.setattr(report, "make_reporters", make_reporters)
    result = gr00t_ppo.run(config)
    assert result["windows"] == 2 and result["optimizer_step"] == config.rl.max_steps
    assert allocated and attempts
    assert all(attempt["mode"] == "online" for attempt in attempts)
    messages = [record.getMessage() for record in caplog.records]
    assert any("continuing without wandb" in message for message in messages)
    assert any("train/joint_loss" in message for message in messages)


@pytest.fixture
def one_logger_sink(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Replace only the SDK transport; the adapter and i4 counters/timers remain real."""
    sinks = []

    def create_client(config):
        # Independent transports represent separate attempts; this does not test
        # reopening the external SDK's singleton within one production process.
        sink = SimpleNamespace(store={}, metrics=[], fail=False, send_failures=0, finishes=0)

        def log_metrics(metrics):
            if sink.fail:
                sink.send_failures += 1
                raise RuntimeError("injected telemetry send failure")
            sink.metrics.append(dict(metrics))

        def finish():
            sink.finishes += 1
            if sink.fail:
                raise RuntimeError("injected telemetry finish failure")

        sink.store_set, sink.store_get = sink.store.__setitem__, sink.store.__getitem__
        sink.store_has_key, sink.store_pop = sink.store.__contains__, sink.store.pop
        sink.log_metrics, sink.finish = log_metrics, finish
        sink.log_app_tag = lambda _tag: None
        sinks.append(sink)
        return sink

    package, core = ModuleType("one_logger"), ModuleType("one_logger.core")
    package.core, core.OneLogger = core, create_client
    monkeypatch.setitem(sys.modules, "one_logger", package)
    monkeypatch.setitem(sys.modules, "one_logger.core", core)
    netrc = tmp_path / "telemetry.netrc"
    netrc.write_text("machine api.wandb.ai login fixture password fixture-only\n")
    netrc.chmod(0o600)
    monkeypatch.setenv("NETRC", str(netrc))
    monkeypatch.setenv("ENABLE_ONELOGGER", "true")
    monkeypatch.setenv("ONE_LOGGER_ASYNC", "true")
    monkeypatch.delenv("SLURM_CLUSTER_NAME", raising=False)
    return sinks


@pytest.mark.parametrize("hook", ["__init__", "on_app_start"])
def test_telemetry_startup_failure_closes_client_before_model_setup(local_run, one_logger_sink, monkeypatch, hook):
    from cosmos_framework.utils.one_logger.one_logger_utils import OneLoggerUtils

    config, _, _, allocated = local_run
    failure = RuntimeError(f"injected telemetry {hook} failure after client creation")
    original = getattr(OneLoggerUtils, hook)

    def fail(*args, **kwargs):
        original(*args, **kwargs)
        raise failure

    with monkeypatch.context() as injection:
        injection.setattr(OneLoggerUtils, hook, fail)
        with pytest.raises(RuntimeError) as retained:
            gr00t_ppo.run(config)
    assert retained.value is failure
    assert not allocated
    assert len(one_logger_sink) == 1 and one_logger_sink[0].finishes == 1
    assert gr00t_ppo.run(config)["optimizer_step"] == config.rl.max_steps
    assert len(one_logger_sink) == 2 and all(sink.finishes == 1 for sink in one_logger_sink)


def test_telemetry_ends_app_when_model_setup_fails_before_training(local_run, one_logger_sink, monkeypatch):
    config, _, _, allocated = local_run
    failure = RuntimeError("injected model setup failure")

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(gr00t_ppo, "assemble", fail)
    with pytest.raises(RuntimeError) as retained:
        gr00t_ppo.run(config)
    assert retained.value is failure
    assert not allocated
    assert len(one_logger_sink) == 1 and one_logger_sink[0].finishes == 1
    assert any("app_finish_time" in frame for frame in one_logger_sink[0].metrics)
    assert not any("app_train_loop_finish_time" in frame for frame in one_logger_sink[0].metrics)


def test_checkpoint_failure_preserves_error_without_telemetry_success(local_run, one_logger_sink, monkeypatch):
    config, attempt, events, allocated = local_run
    config.rl.max_steps = 2
    failure = RuntimeError("injected checkpoint failure after trained window")
    assemble = gr00t_ppo.assemble

    def inject(*args, **kwargs):
        units = assemble(*args, **kwargs)
        learner = units["policy"].body

        def fail_save(*_args, **_kwargs):
            assert learner.progress() == {"optimizer_step": 2, "publication_version": 1}
            raise failure

        learner.save = fail_save
        return units

    monkeypatch.setattr(gr00t_ppo, "assemble", inject)
    with pytest.raises(RuntimeError) as retained:
        gr00t_ppo.run(config)
    assert retained.value is failure
    rows = [json.loads(line) for line in (attempt["output"] / "windows.jsonl").read_text().splitlines()]
    assert [row["optimizer_step"] for row in rows] == [2]
    assert any(frame.get("save_checkpoint_count") == 1 for frame in one_logger_sink[0].metrics)
    assert any(frame.get("save_checkpoint_sync_count") == 1 for frame in one_logger_sink[0].metrics)
    assert not any("joint_ppo_checkpoint_path" in frame for frame in one_logger_sink[0].metrics)
    assert not any("train_iterations_productive_end" in frame for frame in one_logger_sink[0].metrics)
    assert one_logger_sink[0].finishes == 1
    assert events[-6:] == ["checkpoint", "reporters", "sim", "collector", "rollout", "ray"]
    assert all(rollout.stream.closed and gateway.stream.closed for _, _, rollout, gateway, _ in allocated)


def test_training_error_survives_telemetry_send_and_finish_failures(local_run, one_logger_sink, monkeypatch):
    config, attempt, events, allocated = local_run
    failure = RuntimeError("injected second-window training failure")
    assemble = gr00t_ppo.assemble

    def inject(*args, **kwargs):
        units = assemble(*args, **kwargs)
        learner = units["policy"].body
        train_batch = learner.train_batch

        def fail_after_first_window(batch_id):
            if learner.optimizer_step == 2:
                one_logger_sink[0].fail = True
                raise failure
            return train_batch(batch_id)

        learner.train_batch = fail_after_first_window
        return units

    monkeypatch.setattr(gr00t_ppo, "assemble", inject)
    with pytest.raises(RuntimeError) as retained:
        gr00t_ppo.run(config)
    assert retained.value is failure
    rows = [json.loads(line) for line in (attempt["output"] / "windows.jsonl").read_text().splitlines()]
    assert [row["optimizer_step"] for row in rows] == [2]
    assert one_logger_sink[0].store["train_iterations"] == 1
    assert one_logger_sink[0].send_failures == one_logger_sink[0].finishes == 1
    assert events[-6:] == ["checkpoint", "reporters", "sim", "collector", "rollout", "ray"]
    assert all(rollout.stream.closed and gateway.stream.closed for _, _, rollout, gateway, _ in allocated)


@pytest.mark.parametrize("stage", ["trainer", "publication", "validation"])
def test_joint_failure_does_not_complete_telemetry_intervals(
    local_run, one_logger_sink, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    from cosmos_framework.posttrain.control.loop import weight_sync
    from cosmos_framework.posttrain.control.validation.rollout import WindowEvaluator

    config, _, _, _ = local_run
    failure = RuntimeError(f"injected {stage} failure")
    assemble = gr00t_ppo.assemble

    def fail(*_args, **_kwargs):
        raise failure

    if stage == "trainer":

        def inject(*args, **kwargs):
            units = assemble(*args, **kwargs)
            monkeypatch.setattr(units["policy"].body, "train_batch", fail)
            return units

        monkeypatch.setattr(gr00t_ppo, "assemble", inject)
    elif stage == "publication":
        publish = weight_sync.weight_sync

        def fail_publication(*args, validation_step, **kwargs):
            if validation_step:
                raise failure
            return publish(*args, validation_step=validation_step, **kwargs)

        monkeypatch.setattr(weight_sync, "weight_sync", fail_publication)
    else:
        config.rl.validation.enabled = config.rl.validation.before_training = True
        monkeypatch.setattr(WindowEvaluator, "__call__", fail)

    with pytest.raises(RuntimeError) as retained:
        gr00t_ppo.run(config)
    assert retained.value is failure
    sink = one_logger_sink[0]
    assert sink.store["train_iterations_end"] == 0
    assert sink.store["train_samples_end"] == (config.rl.decisions if stage == "publication" else 0)
    if stage == "validation":
        assert sink.store["validation_iterations"] == 0
        assert not any("validation_iterations_time_total" in frame for frame in sink.metrics)
    assert sink.finishes == 1


def test_checkpoint_location_is_captured_before_later_save_and_delayed_delivery(
    local_run, one_logger_sink, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cosmos_framework.posttrain.control.checkpointing import CheckpointLifecycle
    from cosmos_framework.posttrain.control.lifecycle import ControllerLifecycleObserver

    config, attempt, _, _ = local_run
    config.rl.save_every = 2
    entered, release = threading.Event(), threading.Event()

    class Delayed(ControllerLifecycleObserver):
        def on_checkpoint_save_start(self, model_step: int) -> None:
            if model_step == 2:
                entered.set()
                assert release.wait(5.0)

    create = CheckpointLifecycle.coordinator

    def capture(*args, **kwargs):
        kwargs["observer"].add(Delayed())
        coordinator = create(*args, **kwargs)
        save = coordinator.save

        def release_after_second_save(model_step, **options):
            save(model_step, **options)
            if model_step == 2:
                assert entered.wait(2.0)
            else:
                release.set()

        monkeypatch.setattr(coordinator, "save", release_after_second_save)
        return coordinator

    monkeypatch.setattr(CheckpointLifecycle, "coordinator", capture)
    try:
        gr00t_ppo.run(config)
    finally:
        release.set()
    records = [frame for frame in one_logger_sink[0].metrics if "joint_ppo_checkpoint_path" in frame]
    assert records == [
        {
            "joint_ppo_checkpoint_path": str(attempt["output"] / f"checkpoints/iter_{step:06d}"),
            "joint_ppo_checkpoint_optimizer_step": step,
        }
        for step in (2, 4)
    ]


def test_partial_initialization_error_survives_bounded_pending_sdk_finish(
    local_run, one_logger_sink, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cosmos_framework.utils.one_logger.one_logger_utils import OneLoggerUtils
    from cosmos_framework.posttrain.control import lifecycle

    config, _, _, allocated = local_run
    failure = RuntimeError("injected partial initialization with stalled SDK finish")
    entered, release = threading.Event(), threading.Event()
    failure_started = []
    observers = []
    create = lifecycle.build_controller_observers
    initialize = OneLoggerUtils.__init__

    def capture(*args, **kwargs):
        result = create(*args, **kwargs)
        observers.append(result)
        return result

    def fail_after_client(self, sdk_config):
        initialize(self, sdk_config)
        finish = self.one_logger.finish

        def blocked_finish():
            entered.set()
            assert release.wait(5.0)
            finish()

        self.one_logger.finish = blocked_finish
        failure_started.append(time.monotonic())
        raise failure

    monkeypatch.setattr(lifecycle, "build_controller_observers", capture)
    monkeypatch.setattr(lifecycle.ControllerLifecycleObservers, "DRAIN_TIMEOUT_S", 0.05)
    monkeypatch.setattr(OneLoggerUtils, "__init__", fail_after_client)
    try:
        with pytest.raises(RuntimeError) as retained:
            gr00t_ppo.run(config)
        assert retained.value is failure
        assert time.monotonic() - failure_started[0] < 1.0
        assert entered.is_set() and not allocated
        assert one_logger_sink[0].finishes == 0, "SDK finish remains pending after the controller returned"
    finally:
        release.set()
        assert all(observer.shutdown(3.0) for observer in observers)
    assert one_logger_sink[0].finishes == 1
