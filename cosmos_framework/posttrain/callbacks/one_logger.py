# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""OneLogger as one `ControllerLifecycleObserver` implementation of the bus in `control/lifecycle.py`.

Lives in `callbacks/` with the other observability implementations (DESIGN §9, Appendix A: callbacks
hang on control's event bus; `control/` owns the bus, not its backends). It is imported lazily by
`build_controller_observers` at the composition boundary and is not part of the package's eager
`callbacks/__init__` exports, so an absent `one_logger` wheel or torch is touched only by a run that
asked for OneLogger.

The online controllers (`entrypoints/grpo.py` and the entries that reuse its loop) drive training
from one Python process over Ray or torchrun worker groups; they never run
`cosmos_framework.trainer.ImaginaireTrainer`, so the `OneLoggerCallback` that reports every imaginaire
training job to the goodput dashboards never sees them. `OneLoggerLifecycleObserver` maps the
controllers' generic lifecycle events onto the repository's
`cosmos_framework.utils.one_logger.one_logger_utils`, so a GRPO or OPD job is recorded like any other job.
The loop and the checkpoint coordinator emit only generic events and never import this module.
Joint PPO specializes this observer's counter policy and owns a private wrapper; its initialization
remains fail-fast while interval delivery and terminal waits use the same observer bus.

Contract:

- Reporting is on when ``ENABLE_ONELOGGER=true`` (the ``SlurmExecutor`` default) and the
  ``one_logger`` package imports. The controller is the single reporting process
  (``enable_for_current_rank=True``); no barrier is used.
- imaginaire's timer semantics: one ``train_iterations`` interval is ONE optimizer update (a
  ``mini`` window of the GRPO loop; ``rl.max_steps * rl.n_mini`` per run) and one ``train_batch``
  interval is ONE forward/backward call (a ``micro`` call). ``global_batch_size`` is therefore the
  trajectories one forward/backward consumes across the policy's data-parallel world,
  ``rl.batch_per_replica / (n_mini * n_micro)``; ``rl.batch_per_replica`` is already the replica-global
  trajectory batch, so the rollout data replicas never multiply it.
- The clocks are the controller's. The bus delivers events asynchronously, so a hook runs on the
  observer worker some time after the loop emitted it, and a slow observer ahead in the queue can
  deliver a batch's start and end back to back. Every event carries the controller thread's wall and
  monotonic clocks from the moment of emission (`LifecycleEvent`); :meth:`OneLoggerLifecycleObserver.observe`
  makes them the current clock for the hook it runs, and the wrapper's timers read that clock
  (:class:`EmissionClockTimer`, installed on the wrapper at ``on_app_start``), so every interval
  OneLogger derives (``train_batch`` / ``train_iterations`` / ``save_checkpoint`` latest, min, max,
  total, avg, and the ``*_start_time`` / ``*_finish_time`` stamps) measures the controller's timeline.
  The two boundaries the wrapper takes explicit times for, ``app_start_time`` and ``app_finish_time``,
  are passed as such. Neither the installed ``one_logger`` package (1.5.5: ``log_metrics``,
  ``log_app_tag``, ``store_*``, ``finish``) nor the wrapper's train/batch/save hooks accept an explicit
  time, so the timer is the one seam where the emission clock can enter without re-deriving the
  wrapper's metrics.
- Saves are reported as ``sync``: save start/end bracket the coordinator's checkpoint write and
  success follows a committed write; the object-store publish behind it is asynchronous and is not
  what OneLogger measures. A save that raised gets its end and no success.
- Every OneLogger call is best effort. The first failure is logged as a warning and turns the hooks
  into no-ops for the rest of the run; OneLogger never fails or stalls training (the same stance
  imaginaire's callback took in 62c44d846f8). ``on_app_end`` still flushes and releases the global
  OneLogger object after such a failure, and the bus delivers it as a non-droppable gate behind every
  earlier event, so it runs whatever the queue held, in order, and never while another hook of this
  observer is still running. The gates carry their emission stamps like every other event, so
  ``app_train_loop_finish_time`` and ``app_finish_time`` are the controller's clock at the call, not
  the time the queue took to drain.
- ``on_train_end`` closes a batch or step interval a break or raise left open, so a preempted or
  failed step never leaves OneLogger's timers running.
"""

from __future__ import annotations

import os
import platform
import time
from collections.abc import Callable, Mapping
from typing import Any

from cosmos_framework.posttrain.control.dispatch import LifecycleEvent
from cosmos_framework.posttrain.control.lifecycle import ControllerLifecycleObserver
from cosmos_framework.posttrain.utils.log import logger

ONE_LOGGER_PROJECT = "imaginaire4"
APP_TAG_RUN_VERSION = "0.0.0"

Clock = Callable[[], tuple[float, float]]  # (wall seconds since the epoch, monotonic seconds)


def one_logger_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """True when ``ENABLE_ONELOGGER`` asks for OneLogger (case-insensitive ``true``)."""
    env = os.environ if environ is None else environ
    return str(env.get("ENABLE_ONELOGGER", "false")).strip().lower() == "true"


def load_one_logger_utils() -> Any | None:
    """Return imaginaire's ``one_logger_utils`` module, or None when the ``one_logger`` wheel is absent.

    The wheel comes from the hwinf-mlwfo index and is not in every training container; a missing
    package is reported once and the run continues without OneLogger data.
    """
    try:
        import one_logger  # noqa: F401  # the wheel from the hwinf-mlwfo index; absent in some containers

        from cosmos_framework.utils.one_logger import one_logger_utils
    except Exception as e:  # ImportError, or whatever the wheel raises while importing
        logger.warning(
            f"OneLogger requested (ENABLE_ONELOGGER=true) but unavailable: {type(e).__name__}: {e}. "
            "This run reports no OneLogger data."
        )
        return None
    return one_logger_utils


def job_environment(environ: Mapping[str, str] | None = None) -> str:
    """``slurm`` / ``ngc`` / the node name, the way imaginaire's initializer classifies the job."""
    env = os.environ if environ is None else environ
    if env.get("SLURM_JOB_ID"):
        return "slurm"
    if env.get("NGC_JOB_ID"):
        return "ngc"
    return platform.node()


def gpu_name() -> str:
    """The controller's GPU model with spaces dashed, or ``none`` when the controller has no GPU."""
    try:
        import torch

        if torch.cuda.is_available():
            return str(torch.cuda.get_device_name()).replace(" ", "-")
    except Exception:
        pass
    return "none"


def controller_one_logger_config(
    *,
    run_name: str,
    controller: str,
    world_size: int,
    topology: str,
    batch_per_replica: int,
    n_mini: int,
    n_micro: int,
    max_steps: int,
    policy_dp_world: int = 1,
    seq_length: int | None = None,
    environ: Mapping[str, str] | None = None,
    gpu: str | None = None,
    app_start_time_ms: int | None = None,
) -> dict[str, Any]:
    """The OneLogger config for one controller run, in the schema imaginaire's trainer sends.

    ``batch_per_replica`` is the replica-global trajectory batch one loop step collects; ``n_mini``
    optimizer updates of ``n_micro`` forward/backward calls consume it. ``world_size`` is the GPU
    count of the resolved placement and ``topology`` the entry's label for it.
    """
    env = os.environ if environ is None else environ
    batch_per_replica, n_mini, n_micro = int(batch_per_replica), int(n_mini), int(n_micro)
    max_steps, world_size, policy_dp_world = int(max_steps), int(world_size), int(policy_dp_world)
    if min(batch_per_replica, n_mini, n_micro, policy_dp_world) < 1 or max_steps < 0 or world_size < 1:
        raise ValueError(
            f"invalid OneLogger run shape: batch_per_replica={batch_per_replica} n_mini={n_mini} "
            f"n_micro={n_micro} max_steps={max_steps} world_size={world_size} policy_dp_world={policy_dp_world}"
        )
    calls_per_batch = n_mini * n_micro
    if batch_per_replica % calls_per_batch:
        raise ValueError(
            f"rl.batch_per_replica={batch_per_replica} is not a multiple of n_mini*n_micro={calls_per_batch}"
        )
    # One forward/backward call consumes this many trajectories across the policy's DP world
    # (imaginaire's `global_batch_size`); each DP rank sees its share (`batch_size`/`micro_batch_size`).
    global_batch_size = batch_per_replica // calls_per_batch
    per_rank_batch = max(1, global_batch_size // policy_dp_world)
    train_iterations_target = max_steps * n_mini
    job_size = f"BS_{batch_per_replica}_nMINI_{n_mini}_nMICRO_{n_micro}_nGPUs_{world_size}_{topology}"
    device = gpu if gpu is not None else gpu_name()
    config: dict[str, Any] = dict(
        enable_for_current_rank=True,
        one_logger_project=ONE_LOGGER_PROJECT,
        one_logger_run_name=run_name,
        one_logger_async=str(env.get("ONE_LOGGER_ASYNC", "true")).strip().lower() == "true",
        log_every_n_train_iterations=1,
        barrier=None,
        app_tag=f"{run_name}/{controller}/{job_size}/ENV_{job_environment(env)}_GPU_{device}",
        app_tag_run_name=run_name,
        app_tag_run_version=APP_TAG_RUN_VERSION,
        app_start_time=int(app_start_time_ms if app_start_time_ms is not None else round(time.time() * 1000)),
        world_size=world_size,
        micro_batch_size=per_rank_batch,
        batch_size=per_rank_batch,
        global_batch_size=global_batch_size,
        train_iterations_target=train_iterations_target,
        # Every optimizer update consumes global_batch_size * n_micro trajectories, so the run's
        # target is the collected batch times the loop steps, independent of the rollout data mesh.
        train_samples_target=max_steps * batch_per_replica,
        is_baseline_run=False,
        is_train_iterations_enabled=True,
        is_validation_iterations_enabled=False,
        is_test_iterations_enabled=False,
        is_save_checkpoint_enabled=True,
        save_checkpoint_strategy="sync",
        is_log_throughput_enabled=False,
    )
    if seq_length:
        config["seq_length"] = int(seq_length)
    return config


def _current_clock() -> tuple[float, float]:
    return time.time(), time.monotonic()


class EmissionClockTimer:
    """The wrapper's named-timer interface (``cosmos_framework.utils.one_logger.one_logger_utils._NamedTimer``:
    ``start`` / ``stop`` / ``is_active`` / ``active_timers`` / ``reset`` / ``get`` / ``[]`` over a
    ``timers`` dict of ``start``, ``finish``, ``latest``, ``min``, ``max``, ``total``, ``avg``, ``count``),
    reading its clock from ``now()`` instead of ``time.time()``.

    ``now()`` returns the controller's ``(wall, monotonic)`` for the event being delivered
    (:meth:`OneLoggerLifecycleObserver.observe`) and the current clocks otherwise. Timestamps
    (``start`` / ``finish``) are wall seconds, as the wrapper multiplies them by 1000 for its
    ``*_time`` metrics; durations (``latest`` and what derives from it) come from the monotonic clock,
    so a wall-clock step never shows up as a negative or inflated interval. ``set_barrier`` is accepted
    for signature parity and ignored: the controller configures no barrier.
    """

    def __init__(self, now: Clock) -> None:
        self._now = now
        self.timers: dict[str, dict[str, Any]] = {}

    def __getitem__(self, name: str) -> dict[str, Any]:
        return self.get(name)

    def reset(self, name: str | None = None) -> None:
        if name is None:
            self.timers = {}
        else:
            self.timers[name] = {}

    def start(self, name: str, set_barrier: bool = False) -> None:
        data = self.timers.get(name, {})
        if "is_active" in data:
            raise RuntimeError(f"Cannot start timer = '{name}' since it is already active")
        wall, monotonic = self._now()
        data["start"] = wall
        data["_start_monotonic"] = monotonic
        data["is_active"] = True
        data["count"] = data.get("count", 0) + 1
        self.timers[name] = data

    def stop(self, name: str, set_barrier: bool = False) -> None:
        data = self.timers.get(name)
        if data is None or "is_active" not in data:
            raise RuntimeError(f"Cannot end timer = '{name}' since it is not active")
        wall, monotonic = self._now()
        data["finish"] = wall
        data.pop("is_active")
        data["latest"] = max(0.0, monotonic - data.pop("_start_monotonic"))
        if data.get("min", float("inf")) > data["latest"]:
            data["min"] = data["latest"]
        if data.get("max", 0) < data["latest"]:
            data["max"] = data["latest"]
        data["total"] = data.get("total", 0) + data["latest"]
        data["avg"] = data["total"] / data["count"]
        self.timers[name] = data

    def is_active(self, name: str = "") -> bool:
        return "is_active" in self.timers.get(name, {})

    def active_timers(self) -> list[str]:
        return [name for name, data in self.timers.items() if "is_active" in data]

    def get(self, name: str) -> dict[str, Any]:
        if name not in self.timers:
            self.timers[name] = {}
        return self.timers[name]


class OneLoggerLifecycleObserver(ControllerLifecycleObserver):
    """Shared OneLogger lifecycle; only an explicitly required startup may raise.

    Construct through :func:`build_one_logger_observer`. ``utils`` is imaginaire's
    ``one_logger_utils`` module (or a stand-in exposing the same functions). The hooks read the
    clock :meth:`observe` sets from the event's emission stamps; called directly (sync dispatch, the
    tests) they read the current clocks. The base retains the optimizer-step observer's existing
    completion/failure policy; Joint PPO specializes it without duplicating the lifecycle.
    """

    count_open_intervals_on_train_end: bool = True
    emit_app_end_after_failure: bool = True

    def __init__(self, utils: Any, config: Mapping[str, Any]) -> None:
        self._utils = utils
        self._config = dict(config)
        self._one_logger: Any | None = None
        # Hooks report between a successful on_app_start and the first failure; the global object
        # lives (and is torn down by on_app_end) independently of that.
        self._hooks_enabled = False
        self._step_open = False
        self._batch_open = False
        self._validation_open = False
        self._event_clock: tuple[float, float] | None = None  # the emission clocks of the event being delivered
        self._timer: EmissionClockTimer | None = None

    @property
    def config(self) -> dict[str, Any]:
        return dict(self._config)

    @property
    def active(self) -> bool:
        """True while the hooks report: after a successful :meth:`on_app_start`, until a failure or :meth:`on_app_end`."""
        return self._hooks_enabled

    @property
    def initialized(self) -> bool:
        """True while an attached wrapper awaits teardown; this is not an SDK-process liveness check."""
        return self._one_logger is not None

    @property
    def timer(self) -> EmissionClockTimer | None:
        """The emission-clock timer installed on the wrapper at :meth:`on_app_start`, None before it or when the
        wrapper (a stand-in) has no timer to replace."""
        return self._timer

    def now(self) -> tuple[float, float]:
        """``(wall, monotonic)``: the emission clocks of the event being delivered, else the current clocks."""
        clock = self._event_clock
        return _current_clock() if clock is None else clock

    def observe(self, event: LifecycleEvent) -> None:
        """Run the hook for ``event`` with the controller's emission clocks as the current clock."""
        self._event_clock = (event.emitted_wall, event.emitted_monotonic)
        try:
            super().observe(event)
        finally:
            self._event_clock = None

    def on_app_start(self) -> None:
        """Initialize and install emission clocks. Optional failures disable hooks; required startup re-raises."""
        try:
            self._initialize_one_logger()
            self._install_timer(self._one_logger)
            self._hooks_enabled = True
            self._one_logger.on_app_start(set_barrier=False, app_start_time=self._config["app_start_time"])
        except Exception as e:
            self._fail("on_app_start", e)
            if self.fail_fast_start:
                raise

    def _initialize_one_logger(self) -> None:
        if not self._utils.one_logger_is_initialized():
            self._utils.initialize_one_logger(self._config)
        self._one_logger = self._utils.get_one_logger()

    def _release_one_logger(self) -> None:
        self._utils.destroy_one_logger()

    def on_train_start(self, *, train_iterations_start: int, train_samples_start: int) -> None:
        self._call(
            "on_train_start",
            lambda ol: ol.on_train_start(
                set_barrier=False,
                train_iterations_start=int(train_iterations_start),
                train_samples_start=int(train_samples_start),
            ),
        )

    def on_train_step_start(self) -> None:
        if self._step_open:
            return
        self._step_open = True
        self._call("on_train_step_start", lambda ol: ol.on_train_step_start(set_barrier=False))

    def on_train_step_end(self) -> None:
        if not self._step_open:
            return
        self._step_open = False
        self._call("on_train_step_end", lambda ol: ol.on_train_step_end(set_barrier=False))

    def on_train_batch_start(self) -> None:
        if self._batch_open:
            return
        self._batch_open = True
        self._call("on_train_batch_start", lambda ol: ol.on_train_batch_start(set_barrier=False))

    def on_train_batch_end(self) -> None:
        if not self._batch_open:
            return
        self._batch_open = False
        self._call("on_train_batch_end", lambda ol: ol.on_train_batch_end(set_barrier=False))

    def on_validation_start(self) -> None:
        self._validation_open = True
        self._call("on_validation_start", lambda ol: ol.on_validation_start(set_barrier=False))
        self._call("on_validation_batch_start", lambda ol: ol.on_validation_batch_start(set_barrier=False))

    def on_validation_end(self) -> None:
        if not self._validation_open:
            return
        self._validation_open = False
        self._call("on_validation_batch_end", lambda ol: ol.on_validation_batch_end(set_barrier=False))
        self._call("on_validation_end", lambda ol: ol.on_validation_end(set_barrier=False))

    def _checkpoint_iteration(self, model_step: int) -> int:
        return model_step

    def on_checkpoint_save_start(self, model_step: int) -> None:
        self._call(
            "on_save_checkpoint_start",
            lambda ol: ol.on_save_checkpoint_start(global_step=self._checkpoint_iteration(model_step)),
        )

    def on_checkpoint_save_end(self, model_step: int) -> None:
        self._call(
            "on_save_checkpoint_end",
            lambda ol: ol.on_save_checkpoint_end(global_step=self._checkpoint_iteration(model_step)),
        )

    def on_checkpoint_save_success(self, model_step: int) -> None:
        self._call(
            "on_save_checkpoint_success",
            lambda ol: ol.on_save_checkpoint_success(global_step=self._checkpoint_iteration(model_step)),
        )

    def log_metrics(self, metrics: Mapping[str, int | float | bool | str | None]) -> None:
        """Send additional scalar metadata through the SDK's public transport API."""
        self._call("log_metrics", lambda ol: ol.one_logger.log_metrics(dict(metrics)))

    def _close_incomplete_intervals(self) -> None:
        # An abandoned interval is not a successful batch/update/validation. Stop
        # its clock without invoking SDK end hooks, which would increment counters.
        if self._timer is not None:
            for name in ("train_batch", "train_iterations", "validation_iterations", "validation_loop"):
                if self._timer.is_active(name):
                    self._timer.stop(name)
        self._step_open = self._batch_open = self._validation_open = False

    def on_train_end(self) -> None:
        """Close open intervals according to the observer's completion policy, then end the train loop."""
        if self.count_open_intervals_on_train_end:
            self.on_train_batch_end()
            self.on_train_step_end()
        self._close_incomplete_intervals()
        self._call("on_train_end", lambda ol: ol.on_train_end(set_barrier=False))

    def on_app_end(self) -> None:
        """Emit app end with the controller's finish time and attempt wrapper teardown once.

        Finish is attempted even after a hook failure disabled reporting. The bus bounds the caller's
        wait, not the SDK's subprocess cleanup; a blocked finish can outlive this controller boundary.
        """
        one_logger = self._one_logger
        if one_logger is None:
            return
        self._one_logger = None
        self._close_incomplete_intervals()
        finish_ms = int(round(self.now()[0] * 1000))  # the wrapper's explicit-time kwarg for this boundary
        try:
            calls = [("finish", one_logger.finish)]
            if self._hooks_enabled or self.emit_app_end_after_failure:
                calls.insert(
                    0, ("on_app_end", lambda: one_logger.on_app_end(set_barrier=False, app_finish_time=finish_ms))
                )
            for name, fn in calls:
                try:
                    fn()
                except Exception as e:
                    self._warn(name, e)
        finally:
            self._hooks_enabled = False
            self._timer = None
            try:
                self._release_one_logger()
            except Exception as e:
                self._warn("destroy_one_logger", e)

    def _install_timer(self, one_logger: Any) -> None:
        """Put the wrapper's named timer on the emission clock; a stand-in without a timer is left alone."""
        current = getattr(one_logger, "timer", None)
        if current is None:
            return
        active = getattr(current, "active_timers", None)
        if callable(active) and active():  # not our object: another component is timing on it
            logger.warning(
                f"OneLogger timers {active()} are already running on the shared object; "
                "keeping its own clock, so the controller's intervals may include observer delivery delay"
            )
            return
        self._timer = EmissionClockTimer(self.now)
        one_logger.timer = self._timer

    def _call(self, name: str, fn: Callable[[Any], None]) -> None:
        if not self._hooks_enabled:
            return
        try:
            fn(self._one_logger)
        except Exception as e:
            self._fail(name, e)

    def _fail(self, name: str, e: Exception) -> None:
        self._hooks_enabled = False
        logger.warning(
            f"OneLogger {name} failed ({type(e).__name__}: {e}); OneLogger reporting is off for the rest of this run"
        )

    @staticmethod
    def _warn(name: str, e: Exception) -> None:
        logger.warning(f"OneLogger {name} failed during teardown ({type(e).__name__}: {e})")


def build_one_logger_observer(
    config: Any,
    rl: Any,
    *,
    controller: str,
    world_size: int,
    topology: str,
    policy_dp_world: int = 1,
    app_start_time_ms: int | None = None,
    environ: Mapping[str, str] | None = None,
) -> OneLoggerLifecycleObserver | None:
    """The OneLogger observer for this run, or None when the run does not report to OneLogger.

    None when ``ENABLE_ONELOGGER`` is not ``true`` or the ``one_logger`` package is missing.
    """
    if not one_logger_enabled(environ):
        return None
    utils = load_one_logger_utils()
    if utils is None:
        return None
    job = getattr(config, "job", None)
    run_name = "/".join(
        str(getattr(job, key, default))
        for key, default in (("project", "posttrain"), ("group", "default"), ("name", "run"))
    )
    rollout = getattr(config, "rollout", None)
    seq_length = int(getattr(rollout, "max_tokens", 0) or 0) or None
    one_logger_config = controller_one_logger_config(
        run_name=run_name,
        controller=controller,
        world_size=world_size,
        topology=topology,
        batch_per_replica=int(rl.batch_per_replica),
        n_mini=int(rl.n_mini),
        n_micro=int(rl.n_micro),
        max_steps=int(rl.max_steps),
        policy_dp_world=policy_dp_world,
        seq_length=seq_length,
        environ=environ,
        app_start_time_ms=app_start_time_ms,
    )
    logger.info(f"OneLogger on for this run: {one_logger_config['app_tag']}")
    return OneLoggerLifecycleObserver(utils, one_logger_config)
