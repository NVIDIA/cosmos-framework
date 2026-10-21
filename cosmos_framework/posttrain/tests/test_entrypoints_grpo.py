# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""@L0 @CPU: the GRPO entry's loop contract, resume wiring, and launch derivations.

Every test pins a behavior whose failure is SILENT or first surfaces as an expensive cluster crash:
the BATCH boundary (not the optimizer step) driving generate/weight_sync, and resume
converting between the two clocks; a NaN step never streams weights into
the running engine; a resumed loop runs only the remaining steps; `build_trainer` resumes from the latest
checkpoint (dinghaoy's requeue-restarts-from-base bug); preemption saves before exit; run_ts pinned into
worker opts. The real vLLM rollout + FSDP trainer are GPU-tested; these use stub roles / fakes.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from cosmos_framework.posttrain.config.model import DataPackerConfig, HFModelConfig
from cosmos_framework.posttrain.config.rl import (
    CheckpointUploadConfig,
    DatasetConfig,
    GRPOConfig,
    GRPOValidationConfig,
    RolloutConfig,
)
from cosmos_framework.posttrain.control.checkpointing import CheckpointLifecycle
from cosmos_framework.posttrain.schema import Status, Trajectory, Transition
from cosmos_framework.posttrain.schema.checkpoint import FULL_CHECKPOINT_COMPONENTS
from cosmos_framework.posttrain.tests.helpers.entry_loop import emulate_supported_atomic_noreplace
from cosmos_framework.posttrain.tests.helpers.entry_loop import loop_config as _config
from cosmos_framework.posttrain.tests.helpers.entry_loop import mock_loop_env as _mock_env
from cosmos_framework.posttrain.workers.worker_group import Scatter

_EMPTY_PROVIDER_IDENTITY_HASH = hashlib.sha256(b"{}").hexdigest()


@pytest.mark.L0
@pytest.mark.CPU
def test_online_preflight_validates_configured_data_packer_factories(monkeypatch: pytest.MonkeyPatch) -> None:
    """A bad dotted path must fail on the driver before any GPU actor is placed."""
    from cosmos_framework.posttrain.entrypoints import grpo

    data_packer = DataPackerConfig(factory="tests.custom.build_packer")
    selection = SimpleNamespace(backend="hf", model_config=SimpleNamespace(), model_id="model")
    seen: list[DataPackerConfig] = []
    monkeypatch.setattr("cosmos_framework.posttrain.config.build.select_policy_backend", lambda _config: selection)
    monkeypatch.setattr(
        "cosmos_framework.posttrain.config.build.validate_online_policy_backend",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "cosmos_framework.posttrain.data.packer.validate_data_packer_factories",
        lambda config: seen.append(config),
    )

    grpo._validate_online_backend(
        SimpleNamespace(rollout=RolloutConfig(), data_packer=data_packer),
        GRPOConfig(),
    )

    assert seen == [data_packer]


@pytest.mark.L0
@pytest.mark.CPU
def test_build_rollout_uses_student_data_packer_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rollout shaping must use the same model-owned factory as policy shaping."""
    from cosmos_framework.posttrain.entrypoints import grpo

    data_packer = DataPackerConfig(factory="tests.custom.build_packer")
    rollout_config = RolloutConfig(name_or_path="rollout/model", tensor_parallel_size=1)
    selected = object()
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr("cosmos_framework.utils.lazy_config.instantiate", lambda value: value)
    monkeypatch.setattr(
        "cosmos_framework.posttrain.data.packer.init_data_packer_from_factory",
        lambda name, factory, **_kwargs: (calls.append((name, factory)), selected)[1],
    )
    monkeypatch.setattr(
        "cosmos_framework.posttrain.config.build.build_vllm_rollout",
        lambda _config,
        *,
        packer,
        trust_remote_code,
        return_routed_experts,
        enable_sleep_mode,
        topk_logprobs=0: SimpleNamespace(
            packer=packer,
            trust_remote_code=trust_remote_code,
            return_routed_experts=return_routed_experts,
            enable_sleep_mode=enable_sleep_mode,
            topk_logprobs=topk_logprobs,
        ),
    )
    monkeypatch.setattr(
        "cosmos_framework.posttrain.utils.dist.run_rank_converged",
        lambda operation, _context: operation(),
    )

    rollout = grpo.build_rollout(
        SimpleNamespace(
            model_backend="hf",
            model_hf=HFModelConfig(),
            rollout=rollout_config,
            rl=GRPOConfig(routing_replay="rollout"),
            data_packer=data_packer,
        )
    )

    assert calls == [("rollout/model", data_packer.factory)]
    assert rollout.packer is selected
    assert rollout.trust_remote_code is False
    assert rollout.return_routed_experts is True  # rl.routing_replay reaches the engine builder
    assert rollout.enable_sleep_mode is True  # colocated default with rl.colocated_offload on


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    ("gpu_placement", "colocated_offload", "expected"),
    [
        ("colocated", True, True),  # default: the loop will sleep the engine every step
        ("colocated", False, False),  # rl.colocated_offload=False: engine stays resident, no sleep mode
        ("disaggregated", True, True),  # disjoint pools never sleep the engine
        (
            "disaggregated",
            False,
            True,
        ),  # the knob is inert off a shared card; the engine keeps main's sleep-mode allocator (never slept)
    ],
)
def test_build_rollout_enables_engine_sleep_mode_only_when_the_loop_will_sleep_it(
    monkeypatch: pytest.MonkeyPatch, gpu_placement: str, colocated_offload: bool, expected: bool
) -> None:
    """`enable_sleep_mode` swaps vLLM's allocator; it is on exactly when the hand-off cycle will run."""
    from cosmos_framework.posttrain.config.rl import GridConfig
    from cosmos_framework.posttrain.entrypoints import grpo

    monkeypatch.setattr("cosmos_framework.utils.lazy_config.instantiate", lambda value: value)
    monkeypatch.setattr(
        "cosmos_framework.posttrain.data.packer.init_data_packer_from_factory",
        lambda _name, _factory, **_kwargs: object(),
    )
    monkeypatch.setattr(
        "cosmos_framework.posttrain.config.build.build_vllm_rollout",
        lambda _config, *, enable_sleep_mode, **_kwargs: SimpleNamespace(enable_sleep_mode=enable_sleep_mode),
    )
    monkeypatch.setattr(
        "cosmos_framework.posttrain.utils.dist.run_rank_converged",
        lambda operation, _context: operation(),
    )

    rollout = grpo.build_rollout(
        SimpleNamespace(
            model_backend="hf",
            model_hf=HFModelConfig(),
            rollout=RolloutConfig(name_or_path="rollout/model", tensor_parallel_size=1),
            rl=GRPOConfig(colocated_offload=colocated_offload),
            data_packer=DataPackerConfig(factory="tests.custom.build_packer"),
            grid=GridConfig(gpu_placement=gpu_placement),
        )
    )

    assert rollout.enable_sleep_mode is expected


@pytest.mark.L0
@pytest.mark.CPU
def test_reward_packer_uses_student_factory_only_for_prefix_filtering(monkeypatch: pytest.MonkeyPatch) -> None:
    """Optional reward tokenization must share the student's model-owned factory."""
    from cosmos_framework.posttrain.entrypoints import grpo

    data_packer = DataPackerConfig(factory="tests.custom.build_packer")
    config = SimpleNamespace(
        model_backend="hf",
        model_hf=HFModelConfig(),
        rollout=RolloutConfig(name_or_path="model"),
        data_packer=data_packer,
    )
    selected = object()
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "cosmos_framework.posttrain.data.packer.init_data_packer_from_factory",
        lambda name, factory, **_kwargs: (calls.append((name, factory)), selected)[1],
    )
    monkeypatch.setattr(
        "cosmos_framework.posttrain.config.build.build_reward_annotators",
        lambda _rl, _name, packer=None: (packer,),
    )

    enabled = grpo._build_reward_annotators(config, GRPOConfig(min_filter_prefix_tokens=1), "model")
    disabled = grpo._build_reward_annotators(config, GRPOConfig(min_filter_prefix_tokens=None), "model")

    assert calls == [("model", data_packer.factory)]
    assert enabled == (selected,)
    assert disabled == (None,)


@pytest.mark.L0
@pytest.mark.CPU
def test_online_i4_rejects_legacy_model_before_instantiate(monkeypatch: pytest.MonkeyPatch) -> None:
    """A legacy i4 tree must fail on the driver instead of constructing a large model there."""
    from cosmos_framework.posttrain.config import build as build_module
    from cosmos_framework.posttrain.entrypoints import grpo

    legacy = SimpleNamespace(
        model_backend="i4",
        model=SimpleNamespace(config=SimpleNamespace(parallelism=object())),
    )
    monkeypatch.setattr(
        "cosmos_framework.utils.lazy_config.instantiate",
        lambda _value: pytest.fail("legacy online preflight must not instantiate config.model"),
    )

    with pytest.raises(TypeError, match="NativePolicyBuildSpec"):
        grpo._validate_online_backend(legacy, GRPOConfig())
    with pytest.raises(TypeError, match="NativePolicyBuildSpec"):
        build_module.build_native_policy(legacy)


@pytest.mark.L0
@pytest.mark.CPU
def test_native_spec_declares_whether_grpo_requires_collated_forward() -> None:
    from cosmos_framework.posttrain.config.model import NativePolicyBuildSpec, Qwen3VLModelConfig
    from cosmos_framework.posttrain.config.rl import RolloutConfig
    from cosmos_framework.posttrain.entrypoints import grpo

    model_id = "Qwen/Qwen3-VL-8B-Instruct"

    def _config(*, requires_collated: bool) -> SimpleNamespace:
        return SimpleNamespace(
            model_backend="i4",
            model=NativePolicyBuildSpec(
                builder="cosmos_framework.posttrain.model.qwen3_vl.build_qwen3_vl_native_policy",
                model_id=model_id,
                model_config=Qwen3VLModelConfig(name_or_path=model_id),
                requires_collated_policy_forward=requires_collated,
            ),
            rollout=RolloutConfig(name_or_path=model_id),
        )

    grpo._validate_online_backend(
        _config(requires_collated=False),
        GRPOConfig(grpo_surrogate="legacy_pg"),
    )
    with pytest.raises(ValueError, match="requires_collated_policy_forward=True"):
        grpo._validate_online_backend(
            _config(requires_collated=True),
            GRPOConfig(grpo_surrogate="legacy_pg"),
        )
    grpo._validate_online_backend(
        _config(requires_collated=True),
        GRPOConfig(grpo_surrogate="clipped"),
    )


def _record_reports(
    monkeypatch: pytest.MonkeyPatch,
    sink: Callable[[dict[str, Any], int], None],
) -> None:
    def report(metrics: dict[str, Any], step: int) -> None:
        sink(metrics, step)

    reporter = SimpleNamespace(start=lambda: None, report=report, finish=lambda: None)
    monkeypatch.setattr(
        "cosmos_framework.posttrain.utils.report.make_reporters",
        lambda *_args, **_kwargs: [reporter],
    )


@pytest.mark.L0
@pytest.mark.CPU
def test_setup_assembles_the_selected_source_and_reward(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Exercise the public setup seam so source/reward selection cannot regress to hard-coded defaults."""
    from cosmos_framework.posttrain.entrypoints import grpo

    prompt_path = tmp_path / "prompts.jsonl"
    prompt_path.write_text(json.dumps({"question": "Capital of France?", "answer": "Paris"}) + "\n", encoding="utf-8")
    configured = SimpleNamespace(
        rl=GRPOConfig(
            data=DatasetConfig(path=str(prompt_path), prompt_column="question", response_column="answer"),
            reward="exact_match",
            # `setup` wires the advantage estimator onto the RewardWorker, and the estimator checks
            # that each group arrives with its full G members -- in the real loop it does
            # (`rm.score(trajs)` receives a whole rollout batch), so a short group there means the
            # rollout or the reward edge dropped samples. Declaring the size this test actually
            # builds keeps that check live rather than disabling it. (G >= 2 is a config invariant:
            # a one-member group has no variance to normalize over.)
            group_size=2,
        ),
        # `setup` builds the reward stage from the SAME config the objective uses, so a fake
        # config missing model_hf is an incomplete fake, not an over-reach by the code under
        # test: the shared-prefix annotator needs that model's tokenizer for its EOS.
        model_hf=SimpleNamespace(name_or_path="gpt2"),
    )
    config_module = ModuleType("cosmos_framework.utils.config")
    lazy_config_module = ModuleType("cosmos_framework.utils.lazy_config")
    config_module.load_config = lambda *_args, **_kwargs: configured
    lazy_config_module.instantiate = lambda value: value
    monkeypatch.setitem(sys.modules, "cosmos_framework.utils.config", config_module)
    monkeypatch.setitem(sys.modules, "cosmos_framework.utils.lazy_config", lazy_config_module)

    units = grpo.make_setup(
        "unused.py",
        [],
        ["data", "rm"],
        build_objective_fn=grpo.build_objective,
    )()
    obs = units["data"].reset()
    group = [
        Trajectory(
            transitions=[Transition(obs=obs, policy_info={"completion_text": text}, done=True)],
            id=f"configured_{k}",
            group_id=obs.id,
            status=Status.COMPLETED,
        )
        for k, text in enumerate((" paris ", " berlin "))
    ]
    scored = units["rm"].score(group)
    assert [t.transitions[-1].reward["final"] for t in scored] == [1.0, 0.0]

    validation_path = tmp_path / "validation.jsonl"
    validation_path.write_text(
        json.dumps({"question": "Capital of France?", "alternative": "Lyon"}) + "\n",
        encoding="utf-8",
    )
    overlap = GRPOConfig(
        data=DatasetConfig(path=str(prompt_path), prompt_column="question", response_column="answer"),
        validation=GRPOValidationConfig(
            enabled=True,
            data=DatasetConfig(
                path=str(validation_path),
                prompt_column="question",
                response_column="alternative",
            ),
        ),
    )
    with pytest.raises(ValueError, match="overlap semantically.*1 canonical prompt"):
        grpo.build_env(overlap)

    validation_path.write_text('{ "answer": "Paris", "question": "Capital of France?" }\n', encoding="utf-8")
    overlap.validation.data.response_column = "answer"
    with pytest.raises(ValueError, match="same content fingerprint"):
        grpo.build_env(overlap)


# --------------------------------------------------------------------------- loop spy scaffolding
def _units(policy, n):
    """Inert experience-side roles + the given policy; `n` is a shared counter for generate calls and
    weight syncs. The sync is producer-push — `weight_sync(rollout, policy)` drives
    `policy.push_weights(rollout)` — so the counter is attached to the policy here, leaving each test's
    `_Policy` focused on its train/iteration invariants."""

    class _Rollout:
        def generate(self, prompts, group):  # vLLM signature (prompts, group) — no seed
            n["generate"] = n.get("generate", 0) + 1
            return []

        @staticmethod
        def prepare_weight_version(_model_step: int, _producer_identity: str | None) -> None:
            return None

        @staticmethod
        def commit_weight_version(_model_step: int) -> None:
            return None

    def _push_weights(rollout):
        n["sync"] = n.get("sync", 0) + 1
        return None

    from cosmos_framework.posttrain.utils.batch_stats import RewardBatchStats

    reward_report = RewardBatchStats(
        trajectories=2,
        reward_sum=1.0,
        reward_m2=0.5,
        reward_min=0.0,
        reward_max=1.0,
        groups=1,
        informative_groups=1,
        tied_groups=0,
        all_zero_groups=0,
        all_one_groups=0,
    )

    policy.push_weights = _push_weights
    policy.shared_weight_identity = lambda: None
    return {
        "data": SimpleNamespace(get_samples=lambda k: []),
        "rollout": _Rollout(),
        "rm": SimpleNamespace(score_with_report=lambda t: t, batch_stats=lambda: reward_report),
        "replay": SimpleNamespace(put_samples=lambda s: None, get_samples=lambda k: []),
        "policy": policy,
    }


# --------------------------------------------------------------------------- the BATCH is the loop unit
@pytest.mark.L0
@pytest.mark.CPU
def test_grpo_loop_batch_boundary_not_optimizer_step_drives_generate_and_sync(monkeypatch):
    """One collected batch per loop step, consumed as `n_mini` optimizer steps.

    "One loop step == one optimizer step" NO LONGER HOLDS (it is now `steps x n_mini`) — this test
    pins what actually keys off the batch boundary instead: `generate` fires once per loop step and
    `weight_sync` once per boundary, so splitting the batch multiplies neither."""
    from cosmos_framework.posttrain.entrypoints import grpo

    _mock_env(monkeypatch)
    n = {"train": 0}
    reported: list[dict[str, Any]] = []
    reported_steps: list[int] = []
    _record_reports(
        monkeypatch,
        lambda metrics, step: (reported.append(metrics), reported_steps.append(step)),
    )

    class _Policy:
        def __init__(self, n_micro=1):
            self._it, self._acc, self.grad_accum = 0, 0, n_micro
            self.calls: list[tuple[int, int]] = []

        def iteration(self):
            return self._it

        def needs_old_logprobs(self):
            """Ahead-pass capability probe (`entrypoints/grpo.py:461`).

            The driver asks every policy cell whether it can record pi_old before the training
            passes, and only when `rl.needs_cached_old_logprobs` says the objective consults it.
            These fixtures drive the plain train loop, so they answer False -- a fake that omits the
            method fails with AttributeError from inside the driver, which reads as a production bug.
            """
            return False

        def train(self, batch, *, mini=0, micro=0, n_mini=1, n_micro=1):
            # The driver passes INDICES ONLY. In production the dispatch layer CONSUMES these and
            # the body is invoked as `train(batch)`; this fake stands in for the whole WorkerGroup,
            # so it receives them and records the order (spec §3.4).
            del batch, n_mini, n_micro
            self.calls.append((mini, micro))
            n["train"] += 1
            self._acc += 1
            if self._acc == self.grad_accum:  # optimizer steps on the LAST micro call of each mini-batch
                self._acc, self._it = 0, self._it + 1
            return {"train/loss_avg": 0.5}

        def save(self, path):
            pass

    # ONE collected batch per loop step, consumed as n_mini optimizer steps. The batch boundary --
    # not the optimizer step -- is what generate/sync/report key off, which is why splitting the
    # batch must not multiply either of them.
    policy = _Policy(n_micro=1)
    units = _units(policy, n)
    registry_calls = 0

    def _registry() -> dict[str, object]:
        nonlocal registry_calls
        registry_calls += 1
        return {"rollout_r0_k0": object()}

    units["rollout"].connection_registry = _registry
    config = _config(max_steps=3, prompts_per_step=1, group_size=2, n_mini=2, n_micro=1, seed=0, save_every=0)
    config.rl.data_pool = SimpleNamespace(data_plane="ray_nixl_pool")
    barriers: list[dict[str, object]] = []
    monkeypatch.setattr(
        grpo,
        "release_pool_step_barrier",
        lambda _payload, owners: barriers.append(owners) or 0,
    )
    summary = grpo.loop(
        units,
        config,
        "/dev/shm/posttrain_grpo_loop_test",
    )
    assert registry_calls == 1
    assert len(barriers) == 3
    assert all(barrier is barriers[0] for barrier in barriers)
    assert n["generate"] == 3  # ONE rollout per loop step -- NOT one per optimizer step
    assert n["train"] == 6  # 3 steps x n_mini=2
    assert policy.iteration() == 6  # loop step is no longer one optimizer step: it is steps x n_mini
    assert n["sync"] == 4  # 1 post-resume + 3 BATCH boundaries (NOT 6 -- sync is outside the mini loop)
    assert policy.calls == [(0, 0), (1, 0)] * 3  # mini outer, micro inner, restarting each step
    assert reported_steps == [2, 4, 6]  # public report steps are committed optimizer model versions
    assert summary["steps"] == 3
    assert reported[-1]["train/grpo_group_count"] == 1
    assert reported[-1]["train/grpo_informative_group_count"] == 1
    assert (
        reported[-1]["train/grpo_informative_group_count"] + reported[-1]["train/grpo_tied_group_count"]
        == reported[-1]["train/grpo_group_count"]
    )


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    ("gpu_placement", "colocated_offload", "expect_cycle"),
    [
        ("colocated", None, True),  # knob absent from the config: main's behaviour, cycle on
        ("colocated", True, True),
        ("colocated", False, False),  # rl.colocated_offload=False: no sleep/wake, no offload/onload
        ("disaggregated", True, False),  # disjoint pools: nothing to hand off
        ("disaggregated", False, False),  # inert off a shared card
    ],
)
def test_grpo_loop_hand_off_cycle_follows_rl_colocated_offload(
    monkeypatch, gpu_placement: str, colocated_offload: bool | None, expect_cycle: bool
):
    """The per-step residency hand-offs run only on a shared card with `rl.colocated_offload` on.

    With the knob off the engine was built without sleep mode and must never be put to sleep (its
    awake guards would otherwise fire on the next generate), and the trainer must never be moved;
    generate, train and weight sync are unchanged either way."""
    from cosmos_framework.posttrain.entrypoints import grpo

    _mock_env(monkeypatch)
    n = {"train": 0}
    events: list[tuple] = []

    class _Policy:
        def __init__(self):
            self._it = 0

        def iteration(self):
            return self._it

        def needs_old_logprobs(self):
            return False

        def train(self, batch, *, mini=0, micro=0, n_mini=1, n_micro=1):
            del batch, mini, micro, n_mini, n_micro
            n["train"] += 1
            self._it += 1
            return {"train/loss_avg": 0.5}

        def save(self, path):
            pass

        def offload(self):
            events.append(("policy.offload",))
            return 36000.5

        def onload(self):
            events.append(("policy.onload",))
            return 17800.0

        def prepare_weight_sync(self):
            events.append(("policy.prepare_weight_sync",))
            return 24000.25

    policy = _Policy()
    units = _units(policy, n)
    units["rollout"].sleep = lambda: events.append(("rollout.sleep",))
    units["rollout"].wake_up = lambda tags=None: events.append(("rollout.wake_up", tags))
    config = _config(max_steps=1, prompts_per_step=1, group_size=2, seed=0, save_every=0)
    config.grid.gpu_placement = gpu_placement
    if colocated_offload is not None:
        config.rl.colocated_offload = colocated_offload

    summary = grpo.loop(units, config, "/dev/shm/posttrain_grpo_loop_test")

    assert summary["steps"] == 1
    assert n["generate"] == 1 and n["train"] == 1 and n["sync"] == 2  # initial sync + one boundary
    if expect_cycle:
        assert events == [
            # Initial sync: park update-only state, retain policy shards for IPC, then yield policy.
            ("policy.prepare_weight_sync",),
            ("rollout.wake_up", ("weights",)),
            ("policy.offload",),
            ("rollout.wake_up", None),
            # Step: engine sleeps, full trainer returns for update, update-only state parks before sync.
            ("rollout.sleep",),
            ("policy.onload",),
            ("policy.prepare_weight_sync",),
            ("rollout.wake_up", ("weights",)),
            ("policy.offload",),
            ("rollout.wake_up", None),
            # after checkpoint/validation work the trainer is parked again before the next draw
            ("policy.offload",),
            # Normal teardown restores the trainer, so the rollout yields the shared card first.
            ("rollout.sleep",),
        ]
    else:
        assert events == []


@pytest.mark.L0
@pytest.mark.CPU
def test_grpo_colocated_checkpoints_get_an_exclusive_trainer_memory_window(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A scheduled save runs inside the update's own window: the engine is still asleep and the
    trainer resident, and the step's one publication follows. Saving after publication would sleep
    the engine again, and at sleep level 2 that discards the published weights, so every saved step
    would publish twice."""
    from cosmos_framework.posttrain.entrypoints import grpo

    _mock_env(monkeypatch)
    n = {"train": 0}
    events: list[tuple] = []

    class _Policy:
        def __init__(self) -> None:
            self._iteration = 0

        def iteration(self) -> int:
            return self._iteration

        def needs_old_logprobs(self) -> bool:
            return False

        def train(self, _batch: list[Any], **_control: Any) -> dict[str, float]:
            n["train"] += 1
            self._iteration += 1
            return {"train/loss_avg": 0.5}

        def save(self, _path: str) -> None:
            events.append(("policy.save",))

        def offload(self) -> float:
            events.append(("policy.offload",))
            return 36_000.5

        def onload(self) -> float:
            events.append(("policy.onload",))
            return 17_800.0

        def prepare_weight_sync(self) -> float:
            events.append(("policy.prepare_weight_sync",))
            return 24_000.25

    policy = _Policy()
    units = _units(policy, n)
    units["rollout"].sleep = lambda: events.append(("rollout.sleep",))
    units["rollout"].wake_up = lambda tags=None: events.append(("rollout.wake_up", tags))

    def _push_weights(_rollout: Any) -> None:
        n["sync"] = n.get("sync", 0) + 1
        events.append(("weight_sync",))

    policy.push_weights = _push_weights
    config = _config(max_steps=2, prompts_per_step=1, group_size=2, seed=0, save_every=1)
    config.rollout = RolloutConfig(sleep_level=2)

    grpo.loop(units, config, str(tmp_path))

    save_indices = [index for index, event in enumerate(events) if event == ("policy.save",)]
    assert len(save_indices) == 2
    for index in save_indices:
        assert events[index - 2 : index + 6] == [
            ("rollout.sleep",),
            ("policy.onload",),
            ("policy.save",),
            ("policy.prepare_weight_sync",),
            ("rollout.wake_up", ("weights",)),
            ("weight_sync",),
            ("policy.offload",),
            ("rollout.wake_up", None),
        ]
    # The initial publication plus exactly one per update, although every update saved.
    assert events.count(("weight_sync",)) == 3


@pytest.mark.L0
@pytest.mark.CPU
def test_grpo_validating_boundary_keeps_its_save_after_publication(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validation needs the new weights published and merges its checkpoint plan with the scheduled
    one, so a validating boundary saves after publication; other scheduled saves run before it."""
    from cosmos_framework.posttrain.entrypoints import grpo

    _mock_env(monkeypatch)
    _record_reports(monkeypatch, lambda _metrics, _step: None)
    events: list[tuple[str, int]] = []

    class _Policy:
        def __init__(self) -> None:
            self._iteration = 0

        def iteration(self) -> int:
            return self._iteration

        def needs_old_logprobs(self) -> bool:
            return False

        def train(self, _batch: list[Any], **_control: Any) -> dict[str, float]:
            self._iteration += 1
            return {"train/loss_avg": 0.5}

        def save(self, path: str, publish: bool = True, **_kwargs: Any) -> None:
            del publish
            events.append(("save", int(path.rsplit("_", 1)[-1])))

    policy = _Policy()
    units = _validation_units(policy, events)
    policy.push_weights = lambda _rollout: events.append(("publish", policy.iteration()))
    units["rollout"].sleep = lambda: None
    units["rollout"].wake_up = lambda tags=None: None
    config = _validation_config(max_steps=2, before_training=False, every_steps=2, save_every=1)
    config.grid.gpu_placement = "colocated"

    grpo.loop(units, config, "/tmp/out")

    assert events[:5] == [
        ("publish", 0),  # initial publication
        ("save", 1),  # step 1 does not validate: saved inside the update, before publishing
        ("publish", 1),
        ("publish", 2),  # step 2 validates: publish, then the merged save, then the evaluator
        ("save", 2),
    ]
    assert events[5:] and set(events[5:]) == {("validation", 2)}  # one entry per validation batch


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    ("resume", "gpu_placement", "colocated_offload", "expect_parked"),
    [
        pytest.param("/checkpoints/iter_000000002", "colocated", True, True, id="colocated-resume"),
        pytest.param(
            "s3://bucket/run/checkpoints/iter_000000002", "colocated", True, True, id="colocated-object-store-resume"
        ),
        pytest.param("off", "colocated", True, False, id="fresh-start"),
        pytest.param("/checkpoints/iter_000000002", "colocated", False, False, id="resident-engine"),
        pytest.param("/checkpoints/iter_000000002", "disaggregated", True, False, id="disjoint-gpus"),
    ],
)
def test_setup_parks_the_engine_before_a_colocated_trainer_restores_its_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    resume: str,
    gpu_placement: str,
    colocated_offload: bool,
    expect_parked: bool,
) -> None:
    """A resumed trainer restores its optimizer state onto the card the rollout engine shares.

    Bind sets up the experience ActorGroup before the policy one, so an engine left awake holds
    `rollout.gpu_memory_utilization` of the card while that restore runs, and the restore OOMs. The
    parked engine's base weights are never sampled: the loop's first hand-off wakes it and publishes
    the restored policy (`test_grpo_loop_hand_off_cycle_follows_rl_colocated_offload`). An engine
    built without sleep mode (`rl.colocated_offload=False`) or on its own GPUs is left alone.
    """
    from cosmos_framework.posttrain.entrypoints import grpo

    events: list[str] = []

    class _Engine:
        def sleep(self) -> None:
            events.append("rollout.sleep")

    def _build_rollout(_config: Any) -> _Engine:
        events.append("rollout.build")
        return _Engine()

    def _build_trainer(*_args: Any, **_kwargs: Any) -> object:
        events.append("policy.build")
        return object()

    configured = SimpleNamespace(
        rl=GRPOConfig(resume=resume, colocated_offload=colocated_offload),
        grid=SimpleNamespace(gpu_placement=gpu_placement),
    )
    config_module = ModuleType("cosmos_framework.utils.config")
    lazy_config_module = ModuleType("cosmos_framework.utils.lazy_config")
    config_module.load_config = lambda *_args, **_kwargs: configured
    lazy_config_module.instantiate = lambda value: value
    monkeypatch.setitem(sys.modules, "cosmos_framework.utils.config", config_module)
    monkeypatch.setitem(sys.modules, "cosmos_framework.utils.lazy_config", lazy_config_module)
    monkeypatch.setattr(grpo, "build_rollout", _build_rollout)
    monkeypatch.setattr(grpo, "build_trainer", _build_trainer)

    # One setup per ActorGroup, experience before policy, as bind_device_groups runs them.
    grpo.make_setup("unused.py", [], ["rollout"], build_objective_fn=grpo.build_objective)()
    grpo.make_setup("unused.py", [], ["policy"], build_objective_fn=grpo.build_objective)()

    parked = ["rollout.sleep"] if expect_parked else []
    assert events == ["rollout.build", *parked, "policy.build"]


@pytest.mark.L0
@pytest.mark.CPU
def test_grpo_loop_micro_chunks_accumulate_into_one_optimizer_step(monkeypatch):
    """`n_micro` chunks per mini-batch ACCUMULATE: the optimizer steps once per MINI, not per chunk,
    and `micro` is the inner loop."""
    from cosmos_framework.posttrain.entrypoints import grpo

    _mock_env(monkeypatch)
    n = {"train": 0}
    monkeypatch.setattr(
        "cosmos_framework.posttrain.utils.report.make_reporters",
        lambda *_args, **_kwargs: [],
    )

    class _Policy:
        def __init__(self):
            self._it, self._acc, self.grad_accum = 0, 0, 2  # grad_accum IS n_micro
            self.calls: list[tuple[int, int]] = []

        def iteration(self):
            return self._it

        def needs_old_logprobs(self):
            """Ahead-pass capability probe (`entrypoints/grpo.py:461`).

            The driver asks every policy cell whether it can record pi_old before the training
            passes, and only when `rl.needs_cached_old_logprobs` says the objective consults it.
            These fixtures drive the plain train loop, so they answer False -- a fake that omits the
            method fails with AttributeError from inside the driver, which reads as a production bug.
            """
            return False

        def train(self, batch, *, mini=0, micro=0, n_mini=1, n_micro=1):
            del batch, n_mini, n_micro
            self.calls.append((mini, micro))
            n["train"] += 1
            self._acc += 1
            if self._acc == self.grad_accum:
                self._acc, self._it = 0, self._it + 1
            return {"train/loss_avg": 0.5}

        def save(self, path):
            pass

    policy = _Policy()
    grpo.loop(
        _units(policy, n),
        _config(max_steps=2, prompts_per_step=2, group_size=2, n_mini=2, n_micro=2, seed=0, save_every=0),
        "/tmp/out",
    )
    assert n["train"] == 8  # 2 steps x n_mini=2 x n_micro=2
    assert policy.iteration() == 4  # 2 x n_mini optimizer steps — the micro chunks do NOT step
    assert n["generate"] == 2  # still ONE rollout per loop step
    assert policy.calls[:4] == [(0, 0), (0, 1), (1, 0), (1, 1)]  # micro is the INNER loop


# --------------------------------------------------------------------------- NaN abort + resume loop-start
@pytest.mark.L0
@pytest.mark.CPU
def test_grpo_loop_nan_step_aborts_before_weight_sync(monkeypatch):
    """The loop's invariant backstop rejects an impossible applied-NaN result before weight sync."""
    from cosmos_framework.posttrain.entrypoints import grpo

    _mock_env(monkeypatch)
    n = {"train": 0}

    class _Policy:
        def __init__(self) -> None:
            self._iteration = 0

        def iteration(self) -> int:
            return self._iteration

        def needs_old_logprobs(self) -> bool:
            """Ahead-pass capability probe; see the note on the other policy fakes."""
            return False

        def train(self, batch, **_control):  # **_control: the driver's mini/micro indices (§3.4)
            n["train"] += 1
            self._iteration += 1
            return {"train/loss_avg": float("nan") if n["train"] == 2 else 0.5}  # NaN on the 2nd step

        def save(self, path):
            pass

    with pytest.raises(RuntimeError, match="non-finite"):
        grpo.loop(
            _units(_Policy(), n),
            _config(max_steps=3, grad_accum=1, prompts_per_step=1, group_size=2, seed=0, save_every=0),
            "/tmp/out",
        )
    assert n["train"] == 2  # step 0 (finite) + step 1 (NaN, raises)
    assert n["sync"] == 2  # post-resume + step-0 boundary; the NaN step did NOT sync


@pytest.mark.L0
@pytest.mark.CPU
def test_grpo_loop_rejected_update_aborts_before_weight_sync_or_checkpoint(monkeypatch):
    """A trainer-owned skipped outcome is reported as a failure but never published or saved."""
    from cosmos_framework.posttrain.entrypoints import grpo

    _mock_env(monkeypatch)
    n = {"train": 0}
    saves: list[str] = []
    reported: list[dict[str, Any]] = []
    _record_reports(monkeypatch, lambda metrics, _step: reported.append(metrics))

    class _Policy:
        def needs_old_logprobs(self) -> bool:
            """Ahead-pass capability probe; see the note on the other policy fakes."""
            return False

        def __init__(self) -> None:
            self._iteration = 0

        def iteration(self) -> int:
            return self._iteration

        def train(self, batch: list[Any], **_control: Any) -> dict[str, Any]:  # **_control: mini/micro (§3.4)
            del batch
            n["train"] += 1
            if n["train"] == 2:
                return {
                    "train/loss_avg": float("nan"),
                    "optim/update_outcome": "skipped",
                    "optim/skip_reason": "nonfinite_loss",
                    "optim/attempt_id": 2,
                    "optim/nonfinite_skips_total": 1,
                }
            self._iteration += 1
            return {"train/loss_avg": 0.5, "optim/update_outcome": "applied"}

        def save(self, path: str) -> None:
            saves.append(path)

    with pytest.raises(RuntimeError, match="rejected non-finite GRPO update"):
        grpo.loop(
            _units(_Policy(), n),
            _config(max_steps=3, grad_accum=1, prompts_per_step=1, group_size=2, seed=0, save_every=1),
            "/tmp/out",
        )

    assert n["train"] == 2
    assert n["sync"] == 2  # bootstrap + applied step 0; the rejected step is never published
    assert len(saves) == 1 and saves[0].endswith("iter_000001")
    assert len(reported) == 1
    assert reported[0]["optim/update_outcome"] == "applied"
    assert reported[0]["train/reward_mean"] == 0.5
    assert reported[0]["train/grpo_informative_group_count"] == 1


@pytest.mark.L0
@pytest.mark.CPU
def test_grpo_loop_resumes_from_iteration(monkeypatch):
    """A resumed loop runs only the remaining steps (never re-runs 0..N). At n_mini=1 the two clocks
    coincide, so optimizer step 2 IS batch step 2: max_steps=3 → summary.start==2, one step's worth of
    train. `test_grpo_loop_resume_converts_optimizer_steps_to_batch_steps` covers n_mini>1, where
    they diverge."""
    from cosmos_framework.posttrain.entrypoints import grpo

    _mock_env(monkeypatch)
    n = {"train": 0}

    class _Policy:
        def __init__(self) -> None:
            self._iteration = 2

        def iteration(self) -> int:
            return self._iteration

        def needs_old_logprobs(self) -> bool:
            """Ahead-pass capability probe; see the note on the other policy fakes."""
            return False

        def train(self, batch, **_control):  # **_control: the driver's mini/micro indices (§3.4)
            n["train"] += 1
            self._iteration += 1
            return {"train/loss_avg": 0.5}

        def save(self, path):
            pass

    summary = grpo.loop(
        _units(_Policy(), n),
        _config(max_steps=3, grad_accum=1, prompts_per_step=1, group_size=2, seed=0, save_every=0),
        "/tmp/out",
    )
    assert summary["start"] == 2 and n["train"] == 1  # only step 2 ran (grad_accum=1)


@pytest.mark.L0
@pytest.mark.CPU
def test_grpo_resume_rejects_iteration_beyond_configured_run(monkeypatch: pytest.MonkeyPatch) -> None:
    from cosmos_framework.posttrain.entrypoints import grpo

    _mock_env(monkeypatch)

    class _Policy:
        def needs_old_logprobs(self) -> bool:
            """Ahead-pass capability probe; see the note on the other policy fakes."""
            return False

        @staticmethod
        def iteration() -> int:
            return 2

    with pytest.raises(RuntimeError, match="restored batch step 2 exceeds configured total steps 1"):
        grpo.loop(
            _units(_Policy(), {}),
            _config(max_steps=1, grad_accum=1, prompts_per_step=1, group_size=2, seed=0, save_every=0),
            "/tmp/out",
        )


def _validation_units(policy: Any, events: list[tuple[str, int]]) -> dict[str, Any]:
    """Model-free data/rollout/rm roles that expose the live validation loop contract."""
    from cosmos_framework.posttrain.control.validation import ValidationProgress
    from cosmos_framework.posttrain.utils.batch_stats import ValidationRewardStats

    class _OpaquePromptHandle:
        """Stand-in proving the controller forwards validation data without resolving it."""

        def __init__(self, payload: list[int]) -> None:
            self.payload: list[int] = payload

        @staticmethod
        def as_completed() -> Any:
            raise AssertionError("driver must not inspect validation prompt handles")

        @staticmethod
        def all() -> Any:
            raise AssertionError("driver must not inspect validation prompt handles")

        @staticmethod
        def result() -> Any:
            raise AssertionError("driver must not inspect validation prompt handles")

    class _Data:
        def __init__(self) -> None:
            self.position = 0

        @staticmethod
        def get_samples(_count: int) -> list[Any]:
            return []

        def begin_validation(self, batch_size: int) -> int:
            assert batch_size == 2
            self.position = 0
            return 3

        @staticmethod
        def validation_source_fingerprint() -> str:
            return "held-out-v1"

        def get_validation_samples(self, count: int, *, expected_count: int | Scatter) -> _OpaquePromptHandle:
            stop = min(self.position + count, 3)
            samples = list(range(self.position, stop))
            self.position = stop
            expected = expected_count.seq[0] if isinstance(expected_count, Scatter) else expected_count
            assert len(samples) == expected
            return _OpaquePromptHandle(samples)

    class _Rollout:
        def __init__(self) -> None:
            self.weight_version = -1
            self.prepared_weight_version: int | None = None
            self.discarded_batches = 0

        def prepare_weight_version(self, model_step: int, producer_identity: str | None) -> None:
            assert producer_identity == "shared-validation-model"
            self.prepared_weight_version = model_step

        def commit_weight_version(self, model_step: int) -> None:
            assert self.prepared_weight_version == model_step
            self.weight_version = model_step
            self.prepared_weight_version = None

        def generate(self, prompts: list[int] | _OpaquePromptHandle, group: int = 2, **knobs: Any) -> list[int]:
            if "validation" in knobs:
                context = knobs["validation"]
                assert self.weight_version == context.model_step
                assert isinstance(prompts, _OpaquePromptHandle)
                events.append(("validation", context.model_step))
                return [sample for sample in prompts.payload for _ in range(context.n_generation)]
            return []

        def discard_payload(self, _payload: list[int]) -> None:
            self.discarded_batches += 1

        @staticmethod
        def validation_semantics_identity() -> dict[str, Any]:
            return {"version": 1, "backend": "fake", "artifact": "validation-rollout-v1"}

    class _Reward:
        def __init__(self) -> None:
            self.count = 0
            self.model_step = 0
            self.n_generation = 1
            self.sample_ids: tuple[str, ...] = ()

        def score_with_report(self, trajectories: list[Any]) -> list[Any]:
            return trajectories

        @staticmethod
        def batch_stats() -> Any:
            from cosmos_framework.posttrain.utils.batch_stats import RewardBatchStats

            return RewardBatchStats(2, 1.0, 0.5, 0.0, 1.0, 1, 1, 0, 0, 0)

        def score_validation_with_report(
            self,
            trajectories: list[int],
            *,
            model_step: int,
            n_generation: int,
        ) -> None:
            self.count = len(trajectories)
            self.model_step = model_step
            self.n_generation = n_generation
            logical_indices = tuple(dict.fromkeys(trajectories))
            self.sample_ids = tuple(f"validation:held-out-v1:0:{index}" for index in logical_indices)
            return None

        def validation_batch_stats(self) -> ValidationRewardStats:
            from cosmos_framework.posttrain.schema.metrics import Moments

            return ValidationRewardStats(
                reward_moments=Moments.from_values([1.0] * self.count),
                model_step=self.model_step,
                n_generation=self.n_generation,
                sample_ids=self.sample_ids,
                trajectory_ids=tuple(
                    f"{sample_id}_k{generation_index}"
                    for sample_id in self.sample_ids
                    for generation_index in range(self.n_generation)
                ),
            )

    def _push_weights(_rollout: Any) -> None:
        return None

    progress = ValidationProgress()
    policy.push_weights = _push_weights
    policy.shared_weight_identity = lambda: "shared-validation-model"
    if not hasattr(policy, "configure_validation"):
        policy.configure_validation = lambda _identity: None
    if not hasattr(policy, "resumed_from_checkpoint"):
        policy.resumed_from_checkpoint = lambda: int(policy.iteration()) > 0
    if not hasattr(policy, "pending_validation_step"):
        policy.pending_validation_step = lambda: progress.pending_step
    if not hasattr(policy, "pending_validation_metrics"):
        policy.pending_validation_metrics = lambda: dict(progress.pending_metrics)
    if not hasattr(policy, "pending_validation_reasons"):
        policy.pending_validation_reasons = lambda: progress.pending_reasons
    if not hasattr(policy, "mark_validation_pending"):
        policy.mark_validation_pending = lambda step, reasons, metrics: progress.begin(
            step,
            reasons,
            metrics=metrics,
        )
    if not hasattr(policy, "complete_validation"):
        policy.complete_validation = progress.complete
    if not hasattr(policy, "complete_ephemeral_validation"):

        def _complete_ephemeral_validation(_model_step: int) -> None:
            progress.complete(_model_step)

        policy.complete_ephemeral_validation = _complete_ephemeral_validation
    if not hasattr(policy, "save_validation_progress"):
        policy.save_validation_progress = lambda _path: None

    return {
        "data": _Data(),
        "rollout": _Rollout(),
        "rm": _Reward(),
        "replay": SimpleNamespace(put_samples=lambda _samples: None, get_samples=lambda _count: []),
        "policy": policy,
    }


def _validation_config(
    *,
    max_steps: int,
    before_training: bool = True,
    every_steps: int = 2,
    save_every: int = 0,
    n_mini: int = 1,
    save_checkpoint: bool = False,
) -> Any:
    from cosmos_framework.posttrain.config.rl import GRPOValidationConfig, ValidationSamplingConfig

    validation = GRPOValidationConfig(
        enabled=True,
        before_training=before_training,
        every_steps=every_steps,
        save_checkpoint=save_checkpoint,
        batch_size=2,
        sampling=ValidationSamplingConfig(n_generation=1),
    )
    return _config(
        max_steps=max_steps,
        grad_accum=1,
        prompts_per_step=1,
        group_size=2,
        seed=0,
        reward="exact_match",
        save_every=save_every,
        n_mini=n_mini,
        validation=validation,
    )


@pytest.mark.L0
@pytest.mark.CPU
def test_grpo_validation_rejects_different_nonlead_rollout_rank_semantics() -> None:
    from cosmos_framework.posttrain.roles.rollout.validation import resolve_rollout_validation_semantics
    from cosmos_framework.posttrain.workers.worker_group import CollectiveHandle

    identities = CollectiveHandle(
        {
            (0, 0): {"version": 1, "backend": "vllm", "tokenizer": "artifact-a"},
            (0, 1): {"version": 1, "backend": "vllm", "tokenizer": "artifact-b"},
        },
        (1, 2),
        getter=lambda value: value,
    )
    rollout = SimpleNamespace(validation_semantics_identity=lambda: identities)

    with pytest.raises(RuntimeError, match="differ across cells"):
        resolve_rollout_validation_semantics(rollout)


def _rollout_identity_failure_worker(rank: int, world_size: int, rendezvous: str) -> None:
    import torch.distributed as dist

    from cosmos_framework.posttrain.roles.rollout.validation import resolve_rollout_validation_semantics

    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=15),
    )

    class _Rollout:
        @staticmethod
        def validation_semantics_identity() -> dict[str, Any]:
            if dist.get_rank() == 1:
                raise RuntimeError("rank-local tokenizer identity failed")
            return {"version": 1, "backend": "fake"}

    try:
        with pytest.raises(RuntimeError, match="identity preparation failed on one or more ranks"):
            resolve_rollout_validation_semantics(_Rollout())
    finally:
        dist.destroy_process_group()


@pytest.mark.L0
@pytest.mark.CPU
def test_grpo_rollout_identity_rank_local_failure_converges_before_gather(tmp_path: Path) -> None:
    import torch

    rendezvous = str(tmp_path / "rollout-identity-failure")
    torch.multiprocessing.spawn(
        _rollout_identity_failure_worker,
        args=(2, rendezvous),
        nprocs=2,
        join=True,
    )


@pytest.mark.L0
@pytest.mark.CPU
def test_streamed_weight_confirmation_does_not_route_handle_across_asymmetric_meshes() -> None:
    """Real streams use their version stamp; a policy-shaped None handle must not cross meshes."""
    from cosmos_framework.posttrain.control.loop.weight_sync import confirm_weight_version
    from cosmos_framework.posttrain.workers.worker_group import CollectiveHandle

    identity = CollectiveHandle(
        {(replica, rank): None for replica in range(2) for rank in range(4)},
        (2, 4),
        getter=lambda value: value,
    )

    class _Policy:
        def needs_old_logprobs(self) -> bool:
            """Ahead-pass capability probe; see the note on the other policy fakes."""
            return False

        @staticmethod
        def shared_weight_identity() -> Any:
            return identity

    class _Rollout:
        replicas = 4
        ranks = 2

        def __init__(self) -> None:
            self.prepared: tuple[int, str | None] | None = None
            self.committed: int | None = None

        def prepare_weight_version(self, step: int, producer_identity: str | None) -> None:
            self.prepared = (step, producer_identity)

        def commit_weight_version(self, step: int) -> None:
            self.committed = step

    rollout = _Rollout()
    confirm_weight_version(rollout, _Policy(), 7)
    assert rollout.prepared == (7, None)
    assert rollout.committed == 7

    shared_identity = CollectiveHandle(
        {(replica, rank): f"shared-{replica}-{rank}" for replica in range(2) for rank in range(4)},
        (2, 4),
        getter=lambda value: value,
    )
    shared_policy = SimpleNamespace(shared_weight_identity=lambda: shared_identity)
    asymmetric_rollout = SimpleNamespace(
        replicas=4,
        ranks=2,
        prepare_weight_version=lambda _step, _identity: None,
        commit_weight_version=lambda _step: None,
    )
    with pytest.raises(RuntimeError, match="matching policy/rollout meshes"):
        confirm_weight_version(asymmetric_rollout, shared_policy, 7)


@pytest.mark.L0
@pytest.mark.CPU
def test_grpo_validation_resolves_optional_sampling_sentinels_backend_independently() -> None:
    from cosmos_framework.posttrain.config.rl import ValidationSamplingConfig
    from cosmos_framework.posttrain.control.validation.rollout import (
        normalize_rollout_validation_identity,
        rollout_validation_checkpoint_identity,
    )
    from cosmos_framework.posttrain.roles.rollout.validation import (
        build_validation_context,
        resolved_validation_sampling,
    )

    validation = _validation_config(max_steps=1).rl.validation
    validation.sampling = ValidationSamplingConfig(
        temperature=0.5,
        top_p=None,
        top_k=None,
        max_tokens=None,
    )

    resolved = resolved_validation_sampling(validation, effective_max_tokens=48)
    context = build_validation_context(validation, 3, effective_max_tokens=48)

    assert resolved == {
        "temperature": 0.5,
        "top_p": None,
        "top_k": None,
        "repetition_penalty": 1.0,
        "n_generation": 1,
        "max_tokens": 48,
    }
    assert (context.top_p, context.top_k, context.max_tokens) == (None, None, 48)

    identity_args = {
        "training_reward": "exact_match",
        "reward_provider_identity": {
            "name": "exact_match",
            "revision": "builtin-v1",
            "catalog": "builtin",
            "identity_hash": _EMPTY_PROVIDER_IDENTITY_HASH,
        },
        "rollout_semantics": {"backend": "fake", "artifact": "v1"},
        "source_fingerprint": "held-out-v1",
    }
    first = rollout_validation_checkpoint_identity(validation, effective_max_tokens=48, **identity_args)
    second = rollout_validation_checkpoint_identity(validation, effective_max_tokens=49, **identity_args)
    assert normalize_rollout_validation_identity(first) == first
    assert first["sampling"]["max_tokens"] == 48 and first != second
    changed_provider = rollout_validation_checkpoint_identity(
        validation,
        effective_max_tokens=48,
        **{
            **identity_args,
            "reward_provider_identity": {
                "name": "exact_match",
                "revision": "custom-v2",
                "catalog": "custom.catalog",
                "identity_hash": "1" * 64,
            },
        },
    )
    assert first != changed_provider


class _DurableValidationPolicy:
    """Model-free implementation of the LLMTrainer validation transaction surface."""

    def needs_old_logprobs(self) -> bool:
        """Ahead-pass capability probe; see the note on the other policy fakes."""
        return False

    def __init__(self, *, iteration: int = 0, resumed: bool = False) -> None:
        from cosmos_framework.posttrain.control.validation import ValidationProgress

        self._iteration = iteration
        self._resumed = resumed
        self.progress = ValidationProgress()
        self.saves: list[tuple[str, bool, dict[str, Any]]] = []
        self.save_options: list[dict[str, bool]] = []
        self.completion_rewrites: list[str] = []
        self.transaction_events: list[str] = []
        self.configured_identity: dict[str, Any] | None = None
        self._loaded_provenance: dict[str, int | str] | None = None
        self._publication_step: int | None = None

    def configure_validation(self, identity: dict[str, Any]) -> None:
        if self.configured_identity is not None and identity != self.configured_identity:
            raise RuntimeError("validation identity changed")
        self.configured_identity = dict(identity)

    def resumed_from_checkpoint(self) -> bool:
        return self._resumed

    def loaded_checkpoint_provenance(self) -> dict[str, int | str] | None:
        return None if self._loaded_provenance is None else dict(self._loaded_provenance)

    def iteration(self) -> int:
        return self._iteration

    def train(self, _batch: list[Any], **_control: Any) -> dict[str, float]:
        self._iteration += 1
        return {"train/loss_avg": 0.5}

    def pending_validation_step(self) -> int | None:
        return self.progress.pending_step

    def pending_validation_metrics(self) -> dict[str, int | float | bool | str]:
        return dict(self.progress.pending_metrics)

    def pending_validation_reasons(self) -> frozenset[Any]:
        return self.progress.pending_reasons

    def pending_validation_publication_step(self) -> int | None:
        return self._publication_step

    def mark_validation_pending(
        self,
        model_step: int,
        reasons: Any,
        metrics: dict[str, int | float | bool | str],
    ) -> None:
        if self.progress.pending_step is None:
            self.progress.begin(model_step, reasons, metrics=metrics)
        elif self.progress.pending_step != model_step or self.progress.pending_metrics != metrics:
            raise RuntimeError("incompatible pending validation retry")

    def complete_validation(self, model_step: int) -> None:
        self.progress.complete(model_step)
        self.transaction_events.append("complete")

    def complete_ephemeral_validation(self, model_step: int) -> None:
        self.progress.complete(model_step)
        self.transaction_events.append("complete_ephemeral")

    def save(self, path: str, publish: bool = True, update_latest: bool = True, **_kwargs: Any) -> None:
        self.saves.append((path, publish, self.progress.state_dict()))
        self.save_options.append({"publish": publish, "update_latest": update_latest})
        if not update_latest:
            checkpoint = Path(path)
            checkpoint.mkdir(parents=True)
            for component in ("model.pt", "optim.pt", "trainer.pt"):
                (checkpoint / component).write_bytes(b"staged fake checkpoint\n")

    def save_validation_progress(self, path: str) -> None:
        assert self.progress.pending_step is None
        self._publication_step = None
        self.completion_rewrites.append(path)
        self.transaction_events.append("persist")


def _write_fake_external_checkpoint(path: Path) -> None:
    path.mkdir(parents=True)
    for component in ("model.pt", "optim.pt", "trainer.pt"):
        (path / component).write_bytes(b"external fake checkpoint\n")


@pytest.mark.L0
@pytest.mark.CPU
def test_grpo_validation_runs_before_periodic_and_final_without_training_on_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """Validation follows optimizer/model steps while collection follows batch steps."""
    from cosmos_framework.posttrain.entrypoints import grpo

    _mock_env(monkeypatch)
    reports: list[tuple[int, dict[str, Any]]] = []
    _record_reports(monkeypatch, lambda metrics, step: reports.append((step, metrics)))
    events: list[tuple[str, int]] = []

    class _Policy:
        def __init__(self) -> None:
            self.trains = 0
            self._iteration = 0
            self.saves: list[tuple[str, bool]] = []
            self.completion_rewrites: list[str] = []

        def iteration(self) -> int:
            return self._iteration

        def needs_old_logprobs(self) -> bool:
            """Ahead-pass capability probe; see the note on the other policy fakes."""
            return False

        def train(self, _batch: list[Any], **_control: Any) -> dict[str, float]:
            self.trains += 1
            self._iteration += 1
            return {"train/loss_avg": 0.5}

        def save(self, path: str, publish: bool = True, **_kwargs: Any) -> None:
            self.saves.append((path, publish))

        def save_validation_progress(self, path: str) -> None:
            self.completion_rewrites.append(path)

    policy = _Policy()
    units = _validation_units(policy, events)
    summary = grpo.loop(
        units,
        _validation_config(max_steps=3, every_steps=4, n_mini=2, save_checkpoint=True),
        "/tmp/out",
    )

    assert summary["validation_steps"] == [0, 4, 6]
    assert policy.trains == 6
    assert {step for step, metrics in reports if "val/reward_avg" in metrics} == {0, 4, 6}
    assert all(metrics["val/rollout_count"] == 3 for _, metrics in reports if "val/reward_avg" in metrics)
    assert all(type(metrics["val/rollout_count"]) is int for _, metrics in reports if "val/reward_avg" in metrics)
    assert [step for step, _metrics in reports] == [0, 2, 4, 6]
    assert "train/loss_avg" in dict(reports)[4] and "val/reward_avg" in dict(reports)[4]
    assert {step for kind, step in events if kind == "validation"} == {0, 4, 6}
    assert [(path.rsplit("_", 1)[-1], publish) for path, publish in policy.saves] == [
        ("000004", False),
        ("000006", False),
    ]
    assert [path.rsplit("_", 1)[-1] for path in policy.completion_rewrites] == ["000004", "000006"]
    assert units["rollout"].discarded_batches == 6


@pytest.mark.L0
@pytest.mark.CPU
def test_grpo_checkpoint_cadence_uses_batch_clock_and_model_step_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    from cosmos_framework.posttrain.entrypoints import grpo

    class _Policy:
        def __init__(self) -> None:
            self._iteration = 0
            self.saved: list[str] = []

        def iteration(self) -> int:
            return self._iteration

        def needs_old_logprobs(self) -> bool:
            return False

        def train(self, _batch: list[Any], **_control: Any) -> dict[str, float]:
            self._iteration += 1
            return {"train/loss_avg": 0.5}

        def save(self, path: str) -> None:
            self.saved.append(path)

    _mock_env(monkeypatch)
    policy = _Policy()
    summary = grpo.loop(
        _units(policy, {}),
        _config(max_steps=3, prompts_per_step=1, group_size=2, n_mini=2, save_every=2),
        "/tmp/out",
    )

    assert summary["start_model_step"] == 0
    assert policy.saved == ["/tmp/out/checkpoints/iter_000004", "/tmp/out/checkpoints/iter_000006"]


def _pool_validation_setup(
    monkeypatch: pytest.MonkeyPatch, policy: Any, **config_kwargs: Any
) -> tuple[Any, dict[str, Any], Any, list[tuple[Any, dict]]]:
    """Validation loop with the Ray/NIXL pool enabled; records every release barrier the driver issues."""
    from cosmos_framework.posttrain.entrypoints import grpo

    _mock_env(monkeypatch)
    _record_reports(monkeypatch, lambda _metrics, _step: None)
    units = _validation_units(policy, [])
    owners = {"rollout_r0_k0": object()}
    units["rollout"].connection_registry = lambda: owners
    config = _validation_config(**config_kwargs)
    config.rl.data_pool = SimpleNamespace(data_plane="ray_nixl_pool")
    barriers: list[tuple[Any, dict]] = []
    monkeypatch.setattr(
        grpo, "release_pool_step_barrier", lambda payload, owners: barriers.append((payload, owners)) or 0
    )
    return grpo, units, config, barriers


class _PoolValidationPolicy:
    def __init__(self) -> None:
        self._iteration = 0

    def iteration(self) -> int:
        return self._iteration

    def needs_old_logprobs(self) -> bool:
        return False

    def train(self, _batch: list[Any], **_control: Any) -> dict[str, float]:
        self._iteration += 1
        return {"train/loss_avg": 0.5}

    def save(self, path: str, publish: bool = True) -> None:
        del path, publish


@pytest.mark.L0
@pytest.mark.CPU
def test_grpo_validation_batches_release_pool_leases(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every validation rollout batch releases producer leases, not only the training step barrier."""
    grpo, units, config, barriers = _pool_validation_setup(
        monkeypatch, _PoolValidationPolicy(), max_steps=3, every_steps=4, n_mini=2
    )
    summary = grpo.loop(units, config, "/tmp/out")

    assert summary["validation_steps"] == [0, 4, 6]
    assert units["rollout"].discarded_batches == 6
    # resolve_all wraps one cell's value: the fake replay yields [] per training step, the fake
    # rollout yields the non-empty sample list per validation batch.
    training_barriers = [payload for payload, _owners in barriers if payload == [[]]]
    validation_barriers = [payload for payload, _owners in barriers if payload != [[]]]
    assert len(training_barriers) == 3
    assert len(validation_barriers) == 6  # one barrier per validation batch
    assert all(len(payload) == 1 and payload[0] for payload in validation_barriers)
    assert all(owners is barriers[0][1] for _payload, owners in barriers)


@pytest.mark.L0
@pytest.mark.CPU
def test_grpo_validation_releases_pool_leases_when_scoring_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed validation batch still returns its producer pages before the failure propagates."""
    grpo, units, config, barriers = _pool_validation_setup(
        monkeypatch, _PoolValidationPolicy(), max_steps=1, every_steps=1
    )

    def _fail(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("validation reward failed")

    units["rm"].score_validation_with_report = _fail
    with pytest.raises(RuntimeError, match="validation reward failed"):
        grpo.loop(units, config, "/tmp/out")

    assert units["rollout"].discarded_batches == 1
    assert len(barriers) == 1 and barriers[0][0] != [[]]


@pytest.mark.L0
@pytest.mark.CPU
def test_grpo_validation_surfaces_nonlead_rollout_rank_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rollout rank failures converge before reward and remain visible during payload cleanup."""
    from cosmos_framework.posttrain.control.validation import rollout as rollout_validation
    from cosmos_framework.posttrain.entrypoints import grpo

    _mock_env(monkeypatch)
    stage_policy = _DurableValidationPolicy()
    stage_units = _validation_units(stage_policy, [])
    stage_contexts: list[str] = []
    run_validation_stage = rollout_validation.run_validation_stage

    def _fail_rollout_stage(stage: Any, context: str) -> Any:
        result = run_validation_stage(stage, context)
        stage_contexts.append(context)
        if context == "rollout/reward validation rollout batch 0":
            raise RuntimeError("rollout validation failed on rank 1")
        return result

    monkeypatch.setattr(rollout_validation, "run_validation_stage", _fail_rollout_stage)
    with pytest.raises(RuntimeError, match="rollout validation failed on rank 1"):
        grpo.loop(stage_units, _validation_config(max_steps=0), "/tmp/stage-out")
    assert "rollout/reward validation rollout batch 0" in stage_contexts
    assert stage_units["rm"].count == 0
    assert stage_policy.pending_validation_step() == 0
    assert stage_policy.completion_rewrites == []
    monkeypatch.setattr(rollout_validation, "run_validation_stage", run_validation_stage)

    policy = _DurableValidationPolicy()
    units = _validation_units(policy, [])
    generate = units["rollout"].generate

    class _NonLeadFailure(list[int]):
        pass

    class _DiscardHandle:
        def as_completed(self) -> Any:
            yield (0, 0), None
            yield (0, 1), RuntimeError("validation rollout failed on replica 0 rank 1")

    def _generate_with_late_failure(prompts: list[int], group: int = 2, **knobs: Any) -> Any:
        return _NonLeadFailure(generate(prompts, group=group, **knobs))

    units["rollout"].generate = _generate_with_late_failure
    units["rollout"].discard_payload = lambda _trajectories: _DiscardHandle()
    reports: list[tuple[int, dict[str, Any]]] = []
    _record_reports(monkeypatch, lambda metrics, step: reports.append((step, metrics)))

    with pytest.raises(RuntimeError, match="replica 0 rank 1"):
        grpo.loop(units, _validation_config(max_steps=0), "/tmp/out")

    assert policy.pending_validation_step() == 0
    assert policy.completion_rewrites == []
    assert reports == []


@pytest.mark.L0
@pytest.mark.CPU
def test_grpo_remote_rank_validation_failure_keeps_local_pass_pending(monkeypatch: pytest.MonkeyPatch) -> None:
    """Peer failures and metric disagreement both stop validation before commit."""
    from cosmos_framework.posttrain.control.validation import ValidationReport, ValidationTrigger
    from cosmos_framework.posttrain.control.validation import rollout as rollout_validation
    from cosmos_framework.posttrain.entrypoints import grpo

    _mock_env(monkeypatch)
    policy = _DurableValidationPolicy()
    consensus_calls: list[tuple[BaseException | str | None, str]] = []

    def _remote_rank_failure(local_error: BaseException | str | None, context: str) -> None:
        consensus_calls.append((local_error, context))
        assert local_error is None
        if context == "rollout/reward held-out validation":
            raise RuntimeError("rollout/reward held-out validation failed on rank 1: reward exploded")

    monkeypatch.setattr(
        "cosmos_framework.posttrain.control.validation.rollout.raise_if_any_rank_failed",
        _remote_rank_failure,
    )

    with pytest.raises(RuntimeError, match="rank 1: reward exploded"):
        grpo.loop(_validation_units(policy, []), _validation_config(max_steps=0), "/tmp/out")

    assert consensus_calls[-1] == (None, "rollout/reward held-out validation")
    assert policy.pending_validation_step() == 0
    assert policy.completion_rewrites == []

    reports = {
        rank: ValidationReport(
            model_step=2,
            sample_count=3,
            metrics={"val/reward_avg": reward},
            source_fingerprint="held-out-v1",
            trigger_reasons=frozenset({ValidationTrigger.PERIODIC}),
            duration_s=duration,
        )
        for rank, reward, duration in ((0, 0.25, 0.1), (1, 0.75, 0.2))
    }

    def _gather(_report: Any, *, context: str) -> dict[int, Any]:
        assert context == "rollout/reward validation report preparation"
        return reports

    monkeypatch.setattr(rollout_validation, "gather_rank_states", _gather)
    with pytest.raises(RuntimeError, match="metric 'val/reward_avg' differs on rank 1"):
        rollout_validation.converge_validation_report(reports[0])


@pytest.mark.L0
@pytest.mark.CPU
def test_grpo_signal_before_final_validation_saves_unpublished_pending_then_resumes_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resume retries a periodic model step, then finishes the remaining collected batch."""
    from cosmos_framework.posttrain.control.validation import ValidationTrigger
    from cosmos_framework.posttrain.entrypoints import grpo

    events: list[tuple[str, int]] = []
    policy = _DurableValidationPolicy()
    config = _validation_config(
        max_steps=3,
        before_training=False,
        every_steps=4,
        n_mini=2,
        save_checkpoint=True,
    )

    _mock_env(monkeypatch, stops=[None, "signal"])
    summary = grpo.loop(_validation_units(policy, events), config, "/tmp/out")
    assert summary["preempted"] == "signal"
    assert events == []
    assert policy.iteration() == 4
    assert policy.pending_validation_step() == 4
    assert policy.pending_validation_metrics()["train/loss_avg"] == 0.5
    assert policy.saves[-1][1] is False

    policy._resumed = True
    reports: list[tuple[int, dict[str, Any]]] = []
    retry_reasons: list[frozenset[Any]] = []

    _mock_env(monkeypatch)
    from cosmos_framework.posttrain.control.validation.rollout import RolloutRewardValidationRunner

    run_validation = RolloutRewardValidationRunner.run

    def _capture_retry(self: Any, model_step: int, reasons: frozenset[Any]) -> Any:
        retry_reasons.append(reasons)
        return run_validation(self, model_step, reasons)

    def _record_report(metrics: dict[str, Any], step: int) -> None:
        reports.append((step, metrics))
        policy.transaction_events.append("report")

    monkeypatch.setattr(RolloutRewardValidationRunner, "run", _capture_retry)
    _record_reports(monkeypatch, _record_report)
    resumed = grpo.loop(_validation_units(policy, events), config, "/tmp/out")
    assert resumed["validation_steps"] == [4, 6]
    assert policy.iteration() == 6
    assert policy.pending_validation_step() is None
    assert policy.completion_rewrites[-1].endswith("iter_000006")
    assert [step for step, _metrics in reports] == [4, 6]
    assert "train/loss_avg" in reports[0][1] and "val/reward_avg" in reports[0][1]
    assert type(reports[0][1]["val/rollout_count"]) is int
    assert retry_reasons == [
        frozenset({ValidationTrigger.PERIODIC, ValidationTrigger.PENDING}),
        frozenset({ValidationTrigger.FINAL}),
    ]
    assert policy.transaction_events == ["complete", "persist", "report"] * 2

    event_count = len(events)
    report_count = len(reports)
    _mock_env(monkeypatch)
    grpo.loop(_validation_units(policy, events), config, "/tmp/out")
    assert len(events) == event_count
    assert len(reports) == report_count


@pytest.mark.L0
@pytest.mark.CPU
def test_grpo_resume_retries_pending_publication_without_rerunning_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The online loop opts in to the shared post-evaluation publication retry boundary."""
    from cosmos_framework.posttrain.entrypoints import grpo

    events: list[tuple[str, int]] = []
    policy = _DurableValidationPolicy(iteration=1, resumed=True)
    policy._publication_step = 1
    _mock_env(monkeypatch)

    summary = grpo.loop(
        _validation_units(policy, events),
        _validation_config(max_steps=1, before_training=False, every_steps=0),
        "/tmp/out",
    )

    assert summary["start_model_step"] == 1
    assert summary["validation_steps"] == []
    assert events == []
    assert policy._publication_step is None
    assert policy.completion_rewrites == ["/tmp/out/checkpoints/iter_000001"]


@pytest.mark.L0
@pytest.mark.CPU
def test_grpo_external_pending_resume_copies_to_current_target_before_clear(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An external resume source is immutable; completion is persisted in this run's checkpoint."""
    from cosmos_framework.posttrain.control.validation import ValidationTrigger
    from cosmos_framework.posttrain.entrypoints import grpo

    out_dir = str(tmp_path / "new-run")
    policy = _DurableValidationPolicy(iteration=1, resumed=True)
    policy.progress.begin(1, frozenset({ValidationTrigger.FINAL}), metrics={"train/loss_avg": 0.5})
    config = _validation_config(max_steps=1, before_training=False, every_steps=0)
    external = tmp_path / "old-run" / "checkpoints" / "iter_000001"
    _write_fake_external_checkpoint(external)
    from cosmos_framework.posttrain.checkpoint import checkpoint_provenance

    policy._loaded_provenance = checkpoint_provenance(str(external))
    config.rl.resume = str(external)

    _mock_env(monkeypatch)
    emulate_supported_atomic_noreplace(monkeypatch)
    summary = grpo.loop(_validation_units(policy, []), config, out_dir)

    current = tmp_path / "new-run" / "checkpoints" / "iter_000001"
    assert summary["validation_steps"] == [1]
    assert len(policy.saves) == 1
    staged, publish, _state = policy.saves[0]
    assert Path(staged).parent == current.parent and ".external-copy-" in Path(staged).name
    assert publish is False
    assert policy.save_options == [{"publish": False, "update_latest": False}]
    assert current.is_dir()
    assert policy.completion_rewrites == [str(current)]
    assert policy.pending_validation_step() is None


# --------------------------------------------------------------------------- preemption save (survives-requeue)
@pytest.mark.L0
@pytest.mark.CPU
def test_grpo_loop_preemption_emergency_saves(monkeypatch):
    """A termination signal at the step-2 boundary (save_every=0, nothing saved yet) triggers exactly one
    emergency save at iter_000003 (step+1, 1-based) and breaks the loop."""
    from cosmos_framework.posttrain.entrypoints import grpo

    _mock_env(monkeypatch, stops=[None, None, "signal"])  # fires after step 2
    n = {"train": 0}
    saves: list[str] = []

    class _Policy:
        def __init__(self) -> None:
            self._iteration = 0

        def iteration(self) -> int:
            return self._iteration

        def needs_old_logprobs(self) -> bool:
            """Ahead-pass capability probe; see the note on the other policy fakes."""
            return False

        def train(self, batch, **_control):  # **_control: the driver's mini/micro indices (§3.4)
            n["train"] += 1
            self._iteration += 1
            return {"train/loss_avg": 0.5}

        def save(self, path: str, **_kwargs: Any) -> None:
            saves.append(path)

    summary = grpo.loop(
        _units(_Policy(), n),
        _config(max_steps=5, grad_accum=1, prompts_per_step=1, group_size=2, seed=0, save_every=0),
        "/tmp/out",
    )
    assert len(saves) == 1 and saves[0].endswith("iter_000003")
    assert n["train"] == 3 and summary["preempted"] == "signal"  # broke at step 2 of 5


@pytest.mark.L0
@pytest.mark.CPU
def test_grpo_loop_preemption_no_double_save_when_already_checkpointed(monkeypatch):
    """When the step-boundary cadence already saved this step, preemption does NOT re-save it — it just
    exits (no torn/duplicate write)."""
    from cosmos_framework.posttrain.entrypoints import grpo

    _mock_env(monkeypatch, stops=[None, None, "signal"])
    n = {"train": 0}
    saves: list[str] = []

    class _Policy:
        def __init__(self) -> None:
            self._iteration = 0

        def iteration(self) -> int:
            return self._iteration

        def needs_old_logprobs(self) -> bool:
            """Ahead-pass capability probe; see the note on the other policy fakes."""
            return False

        def train(self, batch, **_control):  # **_control: the driver's mini/micro indices (§3.4)
            n["train"] += 1
            self._iteration += 1
            return {"train/loss_avg": 0.5}

        def save(self, path: str, **_kwargs: Any) -> None:
            saves.append(path)

    summary = grpo.loop(
        _units(_Policy(), n),
        _config(max_steps=5, grad_accum=1, prompts_per_step=1, group_size=2, seed=0, save_every=1),
        "/tmp/out",
    )
    assert saves[-1].endswith("iter_000003") and saves.count(saves[-1]) == 1  # cadence saved step 2 once, not twice
    assert n["train"] == 3 and summary["preempted"] == "signal"


def _preemption_plan_saves(monkeypatch, **rl_overrides) -> list[tuple[str, tuple[str, ...] | None]]:
    """Run the loop to a termination trigger at the step-2 boundary (save_every=0, nothing saved yet) under a
    model-only cadence plan and return every policy save as (iter dir name, `components` override or None).
    None = the trainer's configured plan, i.e. the cadence call shape; a tuple = an explicit override."""
    from cosmos_framework.posttrain.entrypoints import grpo

    _mock_env(monkeypatch, stops=[None, None, "signal"])  # fires after step 2
    n = {"train": 0}
    saves: list[tuple[str, tuple[str, ...] | None]] = []

    class _Policy:
        def __init__(self) -> None:
            self._iteration = 0

        def iteration(self) -> int:
            return self._iteration

        def needs_old_logprobs(self) -> bool:
            """Ahead-pass capability probe; see the note on the other policy fakes."""
            return False

        def train(self, batch, **_control):  # **_control: the driver's mini/micro indices (§3.4)
            n["train"] += 1
            self._iteration += 1
            return {"train/loss_avg": 0.5}

        def save(self, path: str, **kwargs: Any) -> None:
            saves.append((Path(path).name, kwargs.get("components")))

    with pytest.warns(UserWarning, match="partial checkpoint artifact"):  # the model-only cadence plan
        summary = grpo.loop(
            _units(_Policy(), n),
            _config(
                max_steps=5,
                grad_accum=1,
                prompts_per_step=1,
                group_size=2,
                seed=0,
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
def test_grpo_loop_preemption_emergency_save_honours_the_run_plan_when_opted_in(monkeypatch):
    """`rl.emergency_checkpoint_components="run"` with a model-only `rl.checkpoint_components`: the termination
    trigger writes exactly ONE save through the cadence path — the trainer is asked for its configured plan
    (no `components` override), so the artifact is what the next cadence boundary would have written — and
    NOT the forced full plan, which for a TiB-scale optimizer state cannot finish inside the grace."""
    saves = _preemption_plan_saves(monkeypatch, emergency_checkpoint_components="run")
    assert saves == [("iter_000003", None)], f"emergency save wrote {saves}; expected the run's own (model-only) plan"


@pytest.mark.L0
@pytest.mark.CPU
def test_grpo_loop_preemption_emergency_save_defaults_to_the_full_plan(monkeypatch):
    """Without opting in, a model-only cadence plan still gets the forced full plan at a termination trigger
    (the behaviour before the knob existed): the trainer is asked for `[model, optimizer, training_state]`."""
    saves = _preemption_plan_saves(monkeypatch)
    assert saves == [("iter_000003", FULL_CHECKPOINT_COMPONENTS)], f"emergency save wrote {saves}"


# --------------------------------------------------------------------------- resume wiring (build_trainer)
def _mock_build_trainer_deps(monkeypatch, tmp_path, loaded):
    """Fakes for build_trainer's heavy parts (model/optimizer/objective/mesh) with a FakeTrainer that
    records the driver-pinned resume passed through the lifecycle. instantiate → identity."""
    import sys

    class _FakeTrainer:
        def __init__(self, *a, **k):
            loaded["kwargs"] = k  # capture ctor kwargs (grad-hygiene wiring assertion)
            self.checkpoint_manager = k["checkpoint_manager"]

        def load(self, ckpt):
            loaded["ckpt"] = ckpt
            return 3

        def set_loaded_checkpoint_provenance(self, provenance: dict[str, int | str]) -> None:
            loaded["provenance"] = provenance

    monkeypatch.setattr("cosmos_framework.posttrain.roles.trainer.LLMTrainer", _FakeTrainer)
    monkeypatch.setattr(
        "cosmos_framework.posttrain.model.hf.build_hf_model", lambda *a, **k: SimpleNamespace(net=object())
    )
    monkeypatch.setattr("cosmos_framework.posttrain.config.build.build_optimizer", lambda *a, **k: object())
    # Stub the config-owned dispatch rather than one surrogate factory: the helper stays valid when
    # GRPOConfig's default surrogate changes, while the public entry builder remains under test.
    monkeypatch.setattr(
        "cosmos_framework.posttrain.data.packer.init_data_packer_from_factory",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        "cosmos_framework.posttrain.config.build.build_objective",
        lambda rl, name_or_path, pdims=None, packer=None: object(),
    )
    # workers/ray.py does `import ray` at module top — absent in the CPU unit-test image. Inject a stub
    # SUBMODULE into sys.modules (the `workers` package is ray-free by design) so build_trainer's
    # `from ...workers.ray import current_pdims` resolves without importing the real module. A plain
    # setattr would import the real workers.ray first and blow up (ModuleNotFoundError: No module 'ray').
    monkeypatch.setitem(
        sys.modules, "cosmos_framework.posttrain.workers.ray", SimpleNamespace(current_pdims=lambda: None)
    )
    monkeypatch.setattr("cosmos_framework.utils.lazy_config.instantiate", lambda x: x)
    monkeypatch.setenv("IMAGINAIRE_OUTPUT_ROOT", str(tmp_path))


def _resume_config(model_name=""):
    from cosmos_framework.posttrain.config.model import HFModelConfig

    return SimpleNamespace(
        job=SimpleNamespace(project="p", group="g", name="n"),
        rl=SimpleNamespace(run_ts="20260101000000"),  # deterministic attempt dir
        model_backend="hf",
        model_hf=HFModelConfig(name_or_path=model_name),
        rollout=SimpleNamespace(name_or_path="base/model", max_tokens=16),
        data_packer=DataPackerConfig(),
    )


@pytest.mark.L0
@pytest.mark.CPU
def test_grpo_objective_builder_passes_the_rollout_bound_only_to_fixed_response_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cosmos_framework.posttrain.entrypoints import grpo

    captured: dict[str, Any] = {}

    def _build(
        rl: Any,
        name_or_path: str,
        pdims: Any = None,
        packer: Any = None,
        *,
        max_response_length: int | None = None,
    ) -> object:
        captured.update(
            rl=rl,
            name_or_path=name_or_path,
            pdims=pdims,
            packer=packer,
            max_response_length=max_response_length,
        )
        return object()

    monkeypatch.setattr("cosmos_framework.posttrain.config.build.build_objective", _build)
    rl = GRPOConfig(loss_type="seq-mean-token-sum-fixed")
    packer = object()
    grpo.build_objective(
        rl,
        SimpleNamespace(name_or_path="model", max_response_length=37),
        packer,
    )

    assert captured == {
        "rl": rl,
        "name_or_path": "model",
        "pdims": None,
        "packer": packer,
        "max_response_length": 37,
    }


@pytest.mark.L0
@pytest.mark.CPU
def test_build_trainer_passes_student_factory_packer_to_objective(monkeypatch, tmp_path) -> None:
    """The objective must use a fresh packer built from the shared student factory."""
    from cosmos_framework.posttrain.entrypoints import grpo

    loaded: dict[str, Any] = {}
    calls: list[tuple[str, str]] = []
    selected = object()
    config = _resume_config(model_name="policy/model")
    _mock_build_trainer_deps(monkeypatch, tmp_path, loaded)
    monkeypatch.setattr(
        "cosmos_framework.posttrain.data.packer.init_data_packer_from_factory",
        lambda name, factory, **_kwargs: (
            calls.append((name, factory)),
            selected,
        )[1],
    )

    objective = object()

    def _objective_builder(rl: Any, model_config: Any, packer: Any) -> Any:
        assert model_config.name_or_path == "policy/model"
        assert packer is selected
        return objective

    grpo.build_trainer(
        config,
        GRPOConfig(resume="off"),
        build_objective_fn=_objective_builder,
    )

    assert calls == [("policy/model", config.data_packer.factory)]


@pytest.mark.L0
@pytest.mark.CPU
def test_build_trainer_converges_policy_packer_setup_across_ranks(monkeypatch, tmp_path) -> None:
    """A rank-local policy packer failure must converge before later collective setup."""
    from cosmos_framework.posttrain.entrypoints import grpo

    loaded: dict[str, Any] = {}
    contexts: list[str] = []
    _mock_build_trainer_deps(monkeypatch, tmp_path, loaded)
    monkeypatch.setattr(
        "cosmos_framework.posttrain.utils.dist.run_rank_converged",
        lambda operation, context: (contexts.append(context), operation())[1],
    )

    grpo.build_trainer(
        _resume_config(model_name="policy/model"),
        GRPOConfig(resume="off"),
        build_objective_fn=grpo.build_objective,
    )

    assert "GRPO policy DataPacker setup" in contexts


@pytest.mark.L0
@pytest.mark.CPU
def test_injected_i4_builder_resolves_transfer_precision(monkeypatch, tmp_path) -> None:
    """The config-selected native builder must deliver the trainer's transfer precision."""
    from cosmos_framework.posttrain.config.model import NativePolicyBuildSpec, Qwen3VLModelConfig
    from cosmos_framework.posttrain.entrypoints import grpo

    loaded: dict[str, Any] = {}
    _mock_build_trainer_deps(monkeypatch, tmp_path, loaded)
    config = _resume_config(model_name="")
    config.model_backend = "i4"
    model_config = Qwen3VLModelConfig(name_or_path="native/model", dtype="bfloat16")
    config.model = NativePolicyBuildSpec(
        builder="tests.fake.build_policy",
        model_id="native/model",
        model_config=model_config,
    )
    monkeypatch.setattr(
        "cosmos_framework.posttrain.config.build.build_native_policy",
        lambda _config: (SimpleNamespace(net=object()), model_config),
    )

    grpo.build_trainer(
        config,
        GRPOConfig(resume="off"),
        build_objective_fn=grpo.build_objective,
    )

    assert loaded["kwargs"]["transfer_dtype"] == "bfloat16"


@pytest.mark.L0
@pytest.mark.CPU
def test_grpo_build_trainer_consumes_the_driver_pinned_checkpoint(monkeypatch, tmp_path):
    """The worker consumes one concrete resume selected by the driver lifecycle."""
    from cosmos_framework.posttrain.entrypoints import grpo

    loaded: dict = {}
    _mock_build_trainer_deps(monkeypatch, tmp_path, loaded)
    config = _resume_config(model_name="")  # unset → defaults to rollout's
    # A REAL GRPOConfig, not a hand-rolled SimpleNamespace: the fake duplicated the field
    # list and broke every time a knob was threaded through build_trainer.
    rl = GRPOConfig(
        resume="off",
        grad_accum=1,
        checkpoint_format="consolidated",
        save_async=False,
        keep_optim_last=2,
        keep_model_last=3,
        upload=CheckpointUploadConfig(
            enabled=True,
            bucket="checkpoint-bucket",
            prefix="posttrain/publication",
            credentials="credentials/<credential>.secret",
        ),
    )
    rl.optimizer.grad_clip_norm = 1.5
    rl.optimizer.force_finite_grads = False

    ckpt = tmp_path / "p" / "g" / "n" / "20260101000000" / "checkpoints" / "iter_000003"
    ckpt.mkdir(parents=True)
    # Stable local provenance requires a complete component set; FakeTrainer.load only records the path.
    (ckpt / "model.pt").touch()
    (ckpt / "optim.pt").touch()
    (ckpt / "trainer.pt").touch()
    rl.resume = str(ckpt)
    grpo.build_trainer(config, rl, build_objective_fn=grpo.build_objective)
    assert loaded["ckpt"] == str(ckpt)
    assert loaded["provenance"]["source"] == str(ckpt)
    assert config.model_hf.name_or_path == "base/model"  # same-weights default applied
    # non-default values prove the knobs are wired from rl.optimizer, not the ctor defaults
    assert loaded["kwargs"]["max_grad_norm"] == 1.5
    assert loaded["kwargs"]["force_finite_grads"] is False
    manager = loaded["kwargs"]["checkpoint_manager"]
    assert manager.save_async is False
    assert manager.keep_optim_last == 2
    assert manager.keep_model_last == 3
    assert manager.upload == {
        "bucket": "checkpoint-bucket",
        "prefix": "posttrain/publication",
        "credentials": "credentials/<credential>.secret",
    }
    assert "checkpoint_store" not in loaded["kwargs"]
    assert "upload" not in loaded["kwargs"]


@pytest.mark.L0
@pytest.mark.CPU
def test_checkpoint_lifecycle_finalization_drains_every_cell_before_teardown() -> None:
    """One failed cell must not prevent the remaining cells from finalizing or PG teardown."""
    from cosmos_framework.posttrain.control.checkpointing import CheckpointLifecycle

    events: list[str] = []

    class _FinalizationHandle:
        def as_completed(self) -> Any:
            events.append("cell-0")
            yield (0, 0), RuntimeError("async checkpoint disk full")
            events.append("cell-1")
            yield (0, 1), None

    policy = SimpleNamespace(finalize_checkpoints=lambda: _FinalizationHandle())

    with pytest.raises(RuntimeError, match="async checkpoint disk full"):
        CheckpointLifecycle.finalize_units(
            {"policy": policy},
            lambda: events.append("teardown"),
        )

    assert events == ["cell-0", "cell-1", "teardown"]


@pytest.mark.L0
@pytest.mark.CPU
def test_grpo_checkpoint_finalization_drains_baseexception_outcome_before_teardown() -> None:
    """A cancellation outcome is propagated only after every cell and PG teardown settle."""
    events: list[str] = []

    class _FinalizationHandle:
        def as_completed(self) -> Any:
            events.append("cell-0")
            yield (0, 0), KeyboardInterrupt("checkpoint writer interrupted")
            events.append("cell-1")
            yield (0, 1), None

    policy = SimpleNamespace(finalize_checkpoints=lambda: _FinalizationHandle())

    with pytest.raises(KeyboardInterrupt, match="checkpoint writer interrupted"):
        CheckpointLifecycle.finalize_units(
            {"policy": policy},
            lambda: events.append("teardown"),
        )

    assert events == ["cell-0", "cell-1", "teardown"]


@pytest.mark.L0
@pytest.mark.CPU
def test_grpo_checkpoint_finalization_preserves_an_active_run_failure() -> None:
    """Cleanup BaseExceptions are logged but cannot replace the active training exception."""
    events: list[str] = []

    class _FinalizationHandle:
        def as_completed(self) -> Any:
            events.append("finalize")
            raise KeyboardInterrupt("checkpoint writer interrupted")

    def _interrupt_teardown() -> None:
        events.append("teardown")
        raise SystemExit("teardown interrupted")

    policy = SimpleNamespace(finalize_checkpoints=lambda: _FinalizationHandle())

    def _fail_during_run() -> None:
        try:
            raise ValueError("training failed")
        finally:
            CheckpointLifecycle.finalize_units(
                {"policy": policy},
                _interrupt_teardown,
            )

    with pytest.raises(ValueError, match="training failed"):
        _fail_during_run()
    assert events == ["finalize", "teardown"]


@pytest.mark.L0
@pytest.mark.CPU
def test_grpo_rollout_cleanup_preserves_an_active_run_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    from cosmos_framework.posttrain.entrypoints import grpo

    def _fail_sleep(_rollout: object) -> None:
        raise SystemExit("sleep failed")

    monkeypatch.setattr(grpo, "sleep_rollout", _fail_sleep)

    def _fail_during_run() -> None:
        try:
            raise ValueError("training failed")
        finally:
            grpo._sleep_rollout_before_trainer_finalization(object())

    with pytest.raises(ValueError, match="training failed"):
        _fail_during_run()


@pytest.mark.L0
@pytest.mark.CPU
def test_grpo_rollout_cleanup_skips_an_unbound_worker_group() -> None:
    from cosmos_framework.posttrain.entrypoints import grpo
    from cosmos_framework.posttrain.workers.worker_group import WorkerGroup

    rollout = WorkerGroup("rollout", "rollout")
    grpo._sleep_rollout_before_trainer_finalization(rollout)


@pytest.mark.L0
@pytest.mark.CPU
def test_grpo_checkpoint_finalization_propagates_drain_baseexception_after_teardown() -> None:
    """A BaseException raised by the drain is deferred until process-group teardown finishes."""
    events: list[str] = []

    class _FinalizationHandle:
        def as_completed(self) -> Any:
            events.append("drain")
            raise SystemExit("checkpoint drain interrupted")

    policy = SimpleNamespace(finalize_checkpoints=lambda: _FinalizationHandle())

    with pytest.raises(SystemExit, match="checkpoint drain interrupted"):
        CheckpointLifecycle.finalize_units(
            {"policy": policy},
            lambda: events.append("teardown"),
        )

    assert events == ["drain", "teardown"]


@pytest.mark.L0
@pytest.mark.CPU
def test_grpo_checkpoint_finalization_propagates_teardown_baseexception() -> None:
    """A teardown cancellation remains visible when checkpoint finalization succeeded."""
    events: list[str] = []
    policy = SimpleNamespace(finalize_checkpoints=lambda: None)

    def _interrupt_teardown() -> None:
        events.append("teardown")
        raise KeyboardInterrupt("process-group teardown interrupted")

    with pytest.raises(KeyboardInterrupt, match="process-group teardown interrupted"):
        CheckpointLifecycle.finalize_units({"policy": policy}, _interrupt_teardown)

    assert events == ["teardown"]


@pytest.mark.L0
@pytest.mark.CPU
def test_grpo_build_trainer_consumes_pinned_fresh_start(monkeypatch, tmp_path):
    """A driver-pinned fresh start skips loading and preserves the explicit model name."""
    from cosmos_framework.posttrain.entrypoints import grpo

    loaded: dict = {}
    _mock_build_trainer_deps(monkeypatch, tmp_path, loaded)
    config = _resume_config(model_name="explicit/model")
    # A REAL GRPOConfig, not a hand-rolled SimpleNamespace: the fake duplicated the field
    # list and broke every time a knob was threaded through build_trainer.
    rl = GRPOConfig(resume="off", grad_accum=1, checkpoint_format="consolidated")
    rl.optimizer.grad_clip_norm = 0.0
    rl.optimizer.force_finite_grads = True

    grpo.build_trainer(config, rl, build_objective_fn=grpo.build_objective)  # no checkpoints/ dir created
    assert "ckpt" not in loaded  # fresh start, load never called
    assert config.model_hf.name_or_path == "explicit/model"  # explicit name_or_path not overridden


def _capture_trainer_checkpoint(monkeypatch, tmp_path, declared: str) -> str:
    """Run build_trainer with `model_hf.name_or_path=declared` and return the path the model builder
    actually received — the value that decides which weights the trainer loads."""
    from cosmos_framework.posttrain.entrypoints import grpo

    seen: dict = {}
    _mock_build_trainer_deps(monkeypatch, tmp_path, {})

    def _capture(cfg, pdims, precision=None):
        # `precision` mirrors `build_hf_model(hf_cfg, pdims=None, precision=None)`: the entrypoint
        # now resolves it once and threads it in, so a stand-in with the old two-argument signature
        # fails with "unexpected keyword argument 'precision'". This test only asserts which
        # checkpoint the trainer loads, so the value is accepted and ignored.
        seen["name_or_path"] = cfg.name_or_path
        return SimpleNamespace(net=object())

    monkeypatch.setattr("cosmos_framework.posttrain.model.hf.build_hf_model", _capture)
    rl = GRPOConfig(resume="off", grad_accum=1, checkpoint_format="consolidated")
    grpo.build_trainer(
        _resume_config(model_name=declared),
        rl,
        build_objective_fn=grpo.build_objective,
    )
    return seen["name_or_path"]


@pytest.mark.L0
@pytest.mark.CPU
def test_grpo_trainer_checkpoint_defaults_to_rollout(monkeypatch, tmp_path):
    """An unset `model_hf.name_or_path` takes the rollout's, so trainer and engine load the SAME weights.
    Losing this fallback fails silently: the engine boots on `rollout.name_or_path` while the trainer
    loads whatever `build_hf_model` defaults to, and every weight_sync then streams one model's weights
    into an engine initialized from another — visible only as a bad reward curve days later."""
    assert _capture_trainer_checkpoint(monkeypatch, tmp_path, declared="") == "base/model"


@pytest.mark.L0
@pytest.mark.CPU
def test_grpo_trainer_checkpoint_explicit_wins(monkeypatch, tmp_path):
    """The fallback only fills an EMPTY name_or_path — an explicitly configured trainer checkpoint is
    never silently replaced by the rollout's."""
    assert _capture_trainer_checkpoint(monkeypatch, tmp_path, declared="explicit/model") == "explicit/model"


# --------------------------------------------------------------------------- run_ts pin (worker split-brain guard)
@pytest.mark.L0
@pytest.mark.CPU
def test_grpo_pins_run_ts_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """The minted run_ts is pinned into the worker opts exactly once, so ray workers reload the SAME
    attempt dir as the driver (a missing pin ⇒ workers checkpoint into the legacy flat dir ⇒ resume
    split-brain)."""
    from cosmos_framework.posttrain.config.model import HFModelConfig
    from cosmos_framework.posttrain.config.rl import GridConfig, RolloutConfig
    from cosmos_framework.posttrain.entrypoints import grpo

    captured: dict = {"objective_builders": []}
    monkeypatch.setattr(
        grpo,
        "make_setup",
        lambda config_path, opts, roles, **kwargs: (
            captured.update(opts=list(opts))
            or captured["objective_builders"].append(kwargs["build_objective_fn"])
            or (lambda: {})
        ),
    )

    def _capture_bind(device_groups: list[Any], **kwargs: Any) -> None:
        captured["device_groups"] = device_groups
        captured["bind_kwargs"] = kwargs

    monkeypatch.setattr(
        "cosmos_framework.posttrain.control.placement.bind_device_groups",
        _capture_bind,
    )
    monkeypatch.setattr("cosmos_framework.posttrain.utils.dist.init_local_process_group", lambda: None)
    monkeypatch.setattr("cosmos_framework.posttrain.utils.dist.destroy_process_group_if_initialized", lambda: None)
    monkeypatch.setattr("cosmos_framework.posttrain.checkpoint.resolve_run_ts", lambda *a, **k: "20260101000000")
    monkeypatch.setattr(grpo, "dump_resolved", lambda *a, **k: captured.update(dump_resolved=(a, k)))
    monkeypatch.setattr(grpo, "loop", lambda *a, **k: None)
    # This run-ts fixture leaves WorkerGroup bodies unbound; separate run-level tests cover
    # rollout cleanup.
    monkeypatch.setattr(grpo, "sleep_rollout", lambda _rollout: None)
    monkeypatch.setattr(CheckpointLifecycle, "finalize_units", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("cosmos_framework.utils.lazy_config.instantiate", lambda x: x)

    # A real GRPOConfig, not a stub: run() bind-checks the recipe it derives from rl before binding,
    # so the config has to carry whatever that derivation reads.
    cfg = SimpleNamespace(
        # num_replicas is load-bearing: run() reads it to declare every WorkerGroup's replica count.
        grid=GridConfig(backend="ray", ranks=1, num_replicas=1),
        rollout=RolloutConfig(tensor_parallel_size=1, name_or_path=""),
        model_backend="hf",
        model_hf=HFModelConfig(),
        # GRPOConfig (main) rather than a SimpleNamespace, so `run()` reads real fields. There is no
        # longer a startup gate coupling prompts_per_step to the rank count (`validate_group_dispatch`
        # was deleted with the whole-group requirement), so these values are free and do not constrain
        # what this test actually pins (run_ts stamped exactly once).
        rl=GRPOConfig(run_ts="", resume="off", batch_per_replica=8, group_size=8),
        job=SimpleNamespace(project="p", group="g", name="n"),
    )
    grpo.run(cfg, "cfg", [], "/tmp/out", build_objective_fn=grpo.build_objective)
    assert captured["opts"].count("rl.run_ts=20260101000000") == 1
    # The resolved-config dump of the pinned attempt dir also carries the attempt's provenance stamp.
    dump_args, dump_kwargs = captured["dump_resolved"]
    assert dump_args[1].endswith("/path/to/resource") and dump_kwargs == {"provenance": True}
    (device_group,) = captured["device_groups"]
    assert device_group.device == "gpu"
    assert [[worker.name for worker in actor_group.members] for actor_group in device_group.actor_groups] == [
        ["data", "rollout", "rm", "replay"],
        ["policy"],
    ]
    assert captured["bind_kwargs"]["weight_sync"] is True
    assert "num_gpus_each" not in captured["bind_kwargs"]
    assert captured["objective_builders"] == [grpo.build_objective, grpo.build_objective]

    cfg.grid = GridConfig(backend="ray", ranks=1, num_replicas=2)
    with pytest.raises(ValueError, match="no longer implicitly replicates policy"):
        grpo.run(cfg, "cfg", [], "/tmp/out", build_objective_fn=grpo.build_objective)

    cfg.grid = GridConfig(backend="ray", shapes={"policy": [1, 2], "rollout": [2, 1]})
    cfg.rollout.tensor_parallel_size = 2
    with pytest.raises(ValueError, match="ROLLOUT_RANKS"):
        grpo.run(cfg, "cfg", [], "/tmp/out", build_objective_fn=grpo.build_objective)
    cfg.grid.backend = "local"
    with pytest.raises(ValueError, match="grid.shapes requires"):
        grpo.run(cfg, "cfg", [], "/tmp/out", build_objective_fn=grpo.build_objective)


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    ("gpu_placement", "transport", "expected_group_names", "expected_actor_groups", "relationship", "loop_fails"),
    [
        ("colocated", None, ["grpo_shared_gpus"], [["grpo_experience", "grpo_policy"]], "same_gpu", False),
        (
            "disaggregated",
            "nixl",
            ["grpo_experience_gpus", "grpo_policy_gpus"],
            [["grpo_experience"], ["grpo_policy"]],
            "cross_device",
            False,
        ),
        ("colocated", None, ["grpo_shared_gpus"], [["grpo_experience", "grpo_policy"]], "same_gpu", True),
    ],
)
def test_grpo_run_lowers_gpu_placement_and_threads_weight_transport(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    gpu_placement: str,
    transport: str | None,
    expected_group_names: list[str],
    expected_actor_groups: list[list[str]],
    relationship: str,
    loop_fails: bool,
) -> None:
    from cosmos_framework.posttrain.config.model import HFModelConfig
    from cosmos_framework.posttrain.config.rl import GridConfig, RolloutConfig
    from cosmos_framework.posttrain.entrypoints import grpo

    captured: dict[str, Any] = {}
    finalizations: list[tuple[dict[str, Any], Any]] = []
    cleanup_rollouts: list[Any] = []

    def _capture_bind(device_groups: list[Any], **kwargs: Any) -> None:
        captured["device_groups"] = device_groups
        captured["kwargs"] = kwargs

    monkeypatch.setattr("cosmos_framework.posttrain.control.placement.bind_device_groups", _capture_bind)
    monkeypatch.setattr(
        "cosmos_framework.posttrain.control.validation.rollout.validate_rollout_reward_sources",
        lambda _rl, _catalog: None,
    )
    monkeypatch.setattr(
        "cosmos_framework.posttrain.utils.entry.pin_run_attempt",
        lambda _config, opts, out_dir: SimpleNamespace(opts=opts, out_dir=out_dir),
    )
    monkeypatch.setattr(grpo, "make_setup", lambda *_args, **_kwargs: lambda: {})
    monkeypatch.setattr(grpo, "dump_resolved", lambda *_args, **_kwargs: None)

    def _loop(*_args: Any, **_kwargs: Any) -> None:
        if loop_fails:
            raise RuntimeError("injected loop failure")

    monkeypatch.setattr(grpo, "loop", _loop)
    monkeypatch.setattr(grpo, "_sleep_rollout_before_trainer_finalization", cleanup_rollouts.append)
    monkeypatch.setattr(
        CheckpointLifecycle,
        "finalize_units",
        lambda wgs, teardown: finalizations.append((wgs, teardown)),
    )
    monkeypatch.setattr("cosmos_framework.utils.lazy_config.instantiate", lambda value: value)

    grid = GridConfig(backend="ray", ranks=1, num_replicas=1, gpu_placement=gpu_placement)
    config = SimpleNamespace(
        grid=grid,
        rollout=RolloutConfig(tensor_parallel_size=1, name_or_path=""),
        model_backend="hf",
        model_hf=HFModelConfig(),
        rl=GRPOConfig(weight_sync_transport=transport),
    )

    if loop_fails:
        with pytest.raises(RuntimeError, match="injected loop failure"):
            grpo.run(
                config,
                "unused.py",
                [],
                str(tmp_path),
                build_objective_fn=grpo.build_objective,
            )
    else:
        grpo.run(
            config,
            "unused.py",
            [],
            str(tmp_path),
            build_objective_fn=grpo.build_objective,
        )

    device_groups = captured["device_groups"]
    assert [device_group.name for device_group in device_groups] == expected_group_names
    assert [[actor_group.name for actor_group in device_group.actor_groups] for device_group in device_groups] == (
        expected_actor_groups
    )
    assert all(device_group.device == "gpu" for device_group in device_groups)
    assert captured["kwargs"]["weight_sync_transport"] == transport
    assert captured["kwargs"]["weight_sync_r2r_broadcast"] is False
    assert captured["kwargs"]["weight_sync_bucket_bytes"] is None
    assert captured["kwargs"]["weight_sync_probe_bytes"] == 256 * 1024 * 1024
    assert captured["kwargs"]["weight_sync_min_gbps"] == 20.0
    assert len(finalizations) == 1
    finalized_wgs, teardown = finalizations[0]
    assert "policy" in finalized_wgs
    assert teardown is None
    assert cleanup_rollouts == ([finalized_wgs["rollout"]] if gpu_placement == "colocated" else [])
    weight_edge = next(
        edge
        for edge in grpo.manifest(config.rl, weight_relationship=relationship).edges
        if edge.plane.value == "weight"
    )
    assert weight_edge.relationship == relationship


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("custom_all_reduce", [False, True], ids=["nccl-all-reduce", "custom-all-reduce"])
def test_grpo_run_exposes_peer_gpus_only_to_the_engine_group_under_custom_all_reduce(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    custom_all_reduce: bool,
) -> None:
    """vLLM's custom all-reduce needs its TP peers' GPUs visible; the colocated trainer keeps one GPU."""
    from cosmos_framework.posttrain.config.model import HFModelConfig
    from cosmos_framework.posttrain.config.rl import GridConfig, RolloutConfig
    from cosmos_framework.posttrain.entrypoints import grpo

    captured: dict[str, Any] = {}

    def _capture_bind(device_groups: list[Any], **_kwargs: Any) -> None:
        captured["device_groups"] = device_groups

    monkeypatch.setattr("cosmos_framework.posttrain.control.placement.bind_device_groups", _capture_bind)
    monkeypatch.setattr(
        "cosmos_framework.posttrain.control.validation.rollout.validate_rollout_reward_sources",
        lambda _rl, _catalog: None,
    )
    monkeypatch.setattr(
        "cosmos_framework.posttrain.utils.entry.pin_run_attempt",
        lambda _config, opts, out_dir: SimpleNamespace(opts=opts, out_dir=out_dir),
    )
    monkeypatch.setattr(grpo, "make_setup", lambda *_args, **_kwargs: lambda: {})
    monkeypatch.setattr(grpo, "dump_resolved", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(grpo, "loop", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(grpo, "_sleep_rollout_before_trainer_finalization", lambda _rollout: None)
    monkeypatch.setattr(CheckpointLifecycle, "finalize_units", lambda _wgs, _teardown: None)
    monkeypatch.setattr("cosmos_framework.utils.lazy_config.instantiate", lambda value: value)

    config = SimpleNamespace(
        grid=GridConfig(backend="ray", shapes={"policy": [1, 2], "rollout": [1, 2]}, gpu_placement="colocated"),
        rollout=RolloutConfig(tensor_parallel_size=2, custom_all_reduce=custom_all_reduce, name_or_path=""),
        model_backend="hf",
        model_hf=HFModelConfig(),
        rl=GRPOConfig(batch_per_replica=8, group_size=8),
    )
    grpo.run(config, "unused.py", [], str(tmp_path), build_objective_fn=grpo.build_objective)

    actor_groups = {
        actor_group.name: actor_group
        for device_group in captured["device_groups"]
        for actor_group in device_group.actor_groups
    }
    assert actor_groups["grpo_experience"].expose_peer_gpus is custom_all_reduce
    assert actor_groups["grpo_experience"].env_vars is None
    assert actor_groups["grpo_policy"].expose_peer_gpus is False


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    ("backend", "gpu_placement", "transport", "match"),
    [
        ("local", "disaggregated", None, "requires grid.backend='ray'"),
        ("ray", "disaggregated", None, "requires rl.weight_sync_transport='nixl'"),
        ("ray", "colocated", "nixl", "cannot serve resolved relationship='same_gpu'"),
    ],
)
def test_grpo_rejects_incompatible_gpu_placement_and_weight_transport(
    backend: str,
    gpu_placement: str,
    transport: str | None,
    match: str,
) -> None:
    from cosmos_framework.posttrain.config.rl import GridConfig
    from cosmos_framework.posttrain.control.placement import (
        resolve_online_weight_placement,
        validate_online_weight_transport,
    )

    grid = GridConfig(backend=backend, gpu_placement=gpu_placement)
    if backend == "local":
        with pytest.raises(ValueError, match=match):
            resolve_online_weight_placement(grid.backend, grid.gpu_placement)
        return
    _backend, _placement, relationship = resolve_online_weight_placement(grid.backend, grid.gpu_placement)
    with pytest.raises(ValueError, match=match):
        validate_online_weight_transport(relationship, transport)


@pytest.mark.L0
@pytest.mark.CPU
def test_setup_wires_the_advantage_estimator_from_the_configured_algorithm_knobs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`rl.unbiased_advantage` / `rl.group_size` must actually REACH the estimator.

    Successor to a whole-group-guard test. The guard itself is gone -- the advantage is computed at
    reward time now, so the objective no longer needs the batch to hold whole groups -- but the
    regression it was written for is not: a configured algorithm knob silently dropped on one of two
    routes, so the run trains with a different estimator than the config declares and nothing says
    so. That knob now lives on the reward stage, so this checks it there.

    Asserts BEHAVIOUR (the advantage actually written) rather than plumbing: `unbiased=True` is
    mean-centring only, so a 2-member group with rewards 0/1 gets +-0.5, whereas the default
    std-normalized form gets ~+-1.0. Reading the number cannot be fooled by a knob that is passed
    along and then ignored.
    """
    from cosmos_framework.posttrain.entrypoints import grpo

    prompt_path = tmp_path / "prompts.jsonl"
    prompt_path.write_text(json.dumps({"question": "Capital of France?", "answer": "Paris"}) + "\n", encoding="utf-8")

    def _score(unbiased: bool) -> list[float]:
        configured = SimpleNamespace(
            rl=GRPOConfig(
                data=DatasetConfig(path=str(prompt_path), prompt_column="question", response_column="answer"),
                reward="exact_match",
                group_size=2,
                unbiased_advantage=unbiased,
            ),
            # `setup` builds the reward stage from the SAME config the objective uses, so a fake
            # config missing model_hf is an incomplete fake, not an over-reach by the code under
            # test: the shared-prefix annotator needs that model's tokenizer for its EOS.
            model_hf=SimpleNamespace(name_or_path="gpt2"),
        )
        config_module = ModuleType("cosmos_framework.utils.config")
        lazy_config_module = ModuleType("cosmos_framework.utils.lazy_config")
        config_module.load_config = lambda *_args, **_kwargs: configured
        lazy_config_module.instantiate = lambda value: value
        monkeypatch.setitem(sys.modules, "cosmos_framework.utils.config", config_module)
        monkeypatch.setitem(sys.modules, "cosmos_framework.utils.lazy_config", lazy_config_module)

        units = grpo.make_setup(
            "unused.py",
            [],
            ["data", "rm"],
            build_objective_fn=grpo.build_objective,
        )()
        obs = units["data"].reset()
        # One right, one wrong -> rewards 1.0 / 0.0 within the same group.
        group = [
            Trajectory(
                transitions=[Transition(obs=obs, policy_info={"completion_text": text}, done=True)],
                id=f"g_{k}",
                group_id=obs.id,
                status=Status.COMPLETED,
            )
            for k, text in enumerate((" paris ", " berlin "))
        ]
        return [t.transitions[0].advantage for t in units["rm"].score(group)]

    biased = _score(unbiased=False)
    assert biased == pytest.approx([1.0, -1.0], abs=1e-5), (
        f"default (std-normalized) estimator did not reach the reward stage: advantages {biased}"
    )
    unbiased = _score(unbiased=True)
    assert unbiased == pytest.approx([0.5, -0.5]), (
        f"rl.unbiased_advantage=True was dropped on the way to the estimator: advantages {unbiased}"
    )


@pytest.mark.L0
@pytest.mark.CPU
def test_build_objective_path_does_not_require_ray(monkeypatch):
    """The whole trainer-setup path must work with Ray absent — `grid.backend=local` uses no actors.

    Regression: `workers/ray.py` imports `ray` at module scope, so importing `current_pdims`
    unconditionally made the Ray-FREE path depend on Ray to discover it is not using Ray. CPU CI
    runners need not have it installed.

    Covers `build_trainer` too, not just `build_objective`: guarding the callee while its caller
    imported `current_pdims` two lines earlier left the path exactly as broken.

    Absence is SIMULATED rather than assumed — this box has Ray, so an environment-dependent test
    would pass here and fail nowhere useful.
    """
    import builtins
    import inspect

    import cosmos_framework.posttrain.entrypoints.grpo as _grpo

    # Source check FIRST: the monkeypatch below breaks any further import, including this one.
    assert "workers.ray import" not in inspect.getsource(_grpo.build_trainer), (
        "build_trainer must resolve pdims via config.build.resolve_pdims(), not a direct workers.ray import"
    )

    real_import = builtins.__import__

    def _no_ray(name, *a, **kw):
        if name.startswith("cosmos_framework.posttrain.workers.ray") or name == "ray":
            raise ImportError("simulated: ray not installed")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", _no_ray)
    from cosmos_framework.posttrain.config.build import resolve_pdims

    assert resolve_pdims() is None, "no Ray, no torchrun world -> single device"


@pytest.mark.L0
@pytest.mark.CPU
def test_items_fn_requires_a_dataset_and_is_actually_applied():
    """items_fn is a MAP over rows of a framework-loaded dataset -- it does not bring its own source.

    The original defect this guards: `prompt_source` keyed the dataset branch off `name` only and
    `build_env` returned ArithmeticEnv before ever looking at `items_fn`, so a source configured by
    items_fn passed --validate and then trained on synthetic addition. Under the map contract that
    is structurally impossible -- items_fn without a name is a config error -- so this pins BOTH
    halves: the config rejects it, and when a name IS present the map really reaches the rows.
    """
    import pytest

    from cosmos_framework.posttrain.config.rl import DatasetConfig, GRPOConfig
    from cosmos_framework.posttrain.entrypoints.grpo import prompt_source
    from cosmos_framework.posttrain.roles.env.dataset import _ColumnMapped

    with pytest.raises(ValueError, match="needs rl.data.name"):
        DatasetConfig(items_fn="pkg.mod.fn")

    rl = GRPOConfig(
        data=DatasetConfig(
            name="org/data",
            subset="main",
            prompt_column="q",
            response_column="gold",
            items_fn="pkg.mod.fn",
        )
    )
    source, ground_truth = prompt_source(rl)
    assert source == "hf", f"a named source is the dataset branch, got {source!r}"
    assert ground_truth is True, "ground truth is response_column -- knowable without probing items_fn"

    # and the map is genuinely applied to the drawn row, not accepted and dropped
    view = _ColumnMapped([{"q": "2+2?", "gold": "4"}], "q", "gold", row_map=lambda r: {**r, "q": f"MAPPED:{r['q']}"})
    assert view[0] == {"prompt": "MAPPED:2+2?", "reference": "4"}


def _items_without_answer(cfg):
    """Test items_fn: prompt-only rows. Module-level so `load_callable` can import it by path."""
    return [{"text": "q0"}, {"text": "q1"}]


def _items_with_answer(cfg):
    """Test items_fn: rows carrying ground truth."""
    return [{"text": "q0", "answer": "#### 0"}, {"text": "q1", "answer": "#### 1"}]


@pytest.mark.L0
@pytest.mark.CPU
def test_ground_truth_is_derived_from_response_column_not_from_items_fn():
    """`items_fn` can no longer lie about ground truth, so the driver-side row probe is gone.

    It used to be TAKEN AT ITS WORD -- prompt_source reported an items_fn source as verifiable,
    which let bind_check accept a verifiable reward, and a prompt-only items_fn then died on the
    first batch with RewardContractError while holding GPUs. A probe was added to draw one row and
    check. Under the map contract items_fn never owns `reference`: it maps a row to a row and the
    framework applies response_column, so verifiability is a static property of the config and the
    probe (which executed user code on the driver during --validate) is unnecessary.
    """
    from cosmos_framework.posttrain.config.rl import DatasetConfig, GRPOConfig
    from cosmos_framework.posttrain.entrypoints.grpo import prompt_source

    verifiable = GRPOConfig(
        data=DatasetConfig(
            name="org/d", subset="main", prompt_column="q", response_column="gold", items_fn="pkg.mod.fn"
        )
    )
    assert prompt_source(verifiable)[1] is True
    # NOTE: response_column is required alongside `name`, so a prompt-only HF source is rejected by
    # DatasetConfig itself -- the case the probe existed to catch cannot be configured at all now.
    import pytest

    with pytest.raises(ValueError, match="needs rl.data.response_column"):
        DatasetConfig(name="org/d", subset="main", prompt_column="q", items_fn="pkg.mod.fn")


@pytest.mark.L0
@pytest.mark.CPU
def test_build_env_wires_items_fn_through_to_the_drawn_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The GRPO entrypoint's own dataset wiring, which nothing else covered.

    Regression: `build_env` called `hf_items(rl.data, builder=...)` after that parameter was renamed
    to `row_map`, i.e. a TypeError on every items_fn run through the production entrypoint. The full
    L0 suite stayed green because no test called `build_env` on the HF branch, and the equivalence
    gate drove a runner that composed `hf_items` itself.

    Patches ONLY `load_dataset`, so the real `hf_items` and the real `_ColumnMapped` run -- a mocked
    `hf_items` could not catch a signature mismatch, which is the defect being pinned.
    """
    import datasets

    from cosmos_framework.posttrain.config.rl import DatasetConfig, GRPOConfig
    from cosmos_framework.posttrain.entrypoints.grpo import build_env

    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: [{"q": "2+2?", "gold": "4", "level": 3}])
    rl = GRPOConfig(
        data=DatasetConfig(
            name="org/data",
            subset="main",
            prompt_column="q",
            response_column="gold",
            items_fn="cosmos_framework.posttrain.tests.test_entrypoints_grpo._mark_row",
        )
    )
    obs = build_env(rl).reset()
    assert obs.representations["prompt"] == "MAPPED:2+2?", "items_fn must reach the drawn row"
    assert obs.representations["reference"] == "4", "response_column still supplies ground truth"
    assert obs.representations["level"] == 3, "columns the map does not touch must pass through"


def _mark_row(row: dict) -> dict:
    """items_fn target for the test above: a row map that rewrites the prompt column in place."""
    return {**row, "q": f"MAPPED:{row['q']}"}


@pytest.mark.L0
@pytest.mark.CPU
def test_sample_log_reads_the_canonical_reference_key():
    """The sample log prints the gold tail; it read the pre-rename key and would log '' forever.

    A wrong key here does not fail anything -- it silently degrades the one line an operator uses to
    eyeball whether the reward is scoring what they think, which is exactly the failure mode this
    branch keeps arguing against.
    """
    import inspect

    from cosmos_framework.posttrain.utils.report import log_sample_previews

    src = inspect.getsource(log_sample_previews)
    assert "answer_tail" not in src, "sample log still reads the pre-rename preview field"
    assert "reference_tail" in src


@pytest.mark.L0
@pytest.mark.CPU
def test_build_env_hf_branch_works_without_an_items_fn(monkeypatch):
    """The plain `rl.data.name` path -- no items_fn -- through the entrypoint's own wiring.

    The `hf_items(...)` call sits OUTSIDE the `if rl.data.items_fn` block, so the parameter-rename
    break hit EVERY source resolving to "hf", not just recipes supplying a map. After this MR that
    branch is the only way a name-bearing recipe reaches its dataset, so it is pinned on its own
    rather than only via the items_fn test above.
    """
    import datasets

    from cosmos_framework.posttrain.config.rl import DatasetConfig, GRPOConfig
    from cosmos_framework.posttrain.entrypoints.grpo import build_env

    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: [{"q": "2+2?", "gold": "4"}])
    rl = GRPOConfig(data=DatasetConfig(name="org/data", subset="main", prompt_column="q", response_column="gold"))
    obs = build_env(rl).reset()
    assert obs.representations["prompt"] == "2+2?", "the raw column is the prompt when no map is set"
    assert obs.representations["reference"] == "4"


@pytest.mark.L0
@pytest.mark.CPU
def test_grid_build_rejects_micro_batches_that_do_not_divide_by_dp_world() -> None:
    """The `select_chunk` precondition (§5.1: `chunk_len % dp_world == 0`) must be checked at
    grid-build time, not discovered by a rank whose share silently rounds to zero.

    Cannot live in `GRPOConfig.__post_init__` -- `dp_world` is `ParallelDims`-derived and
    `ParallelDims` does not exist yet at config load, only once `entrypoints/grpo.py::run` has
    resolved the policy WG's rank width.
    """
    from cosmos_framework.posttrain.control.loop.batching import validate_batch_divisibility

    for micro, dp_world in ((8, 3), (6, 4)):
        with pytest.raises(ValueError, match="dp_world"):
            validate_batch_divisibility(
                batch_size=micro,
                mini_batch_size=None,
                micro_batch_size=micro,
                dp_world=dp_world,
            )

    with pytest.raises(ValueError, match="mini_batch_size"):
        validate_batch_divisibility(
            batch_size=32,
            mini_batch_size=8,
            micro_batch_size=None,
            dp_world=16,
        )


@pytest.mark.L0
@pytest.mark.CPU
def test_grpo_loop_rolls_chunk_metrics_up_by_token_weight_not_plain_mean(monkeypatch):
    """Chunk token means use their public weight sums for an exact batch reduction.

    Chunks are deliberately UNEQUAL: with equal divisors the two reductions agree and the test
    would pass against the broken implementation.
    """
    from cosmos_framework.posttrain.entrypoints import grpo

    _mock_env(monkeypatch)
    n = {"train": 0}
    reported: list[dict[str, Any]] = []

    class _Reporter:
        def start(self) -> None:
            pass

        def report(self, metrics: dict[str, Any], step: int) -> None:
            del step
            reported.append(metrics)

        def finish(self) -> None:
            pass

    monkeypatch.setattr(
        "cosmos_framework.posttrain.utils.report.make_reporters",
        lambda *_args, **_kwargs: [_Reporter()],
    )
    # chunk sums 10 and 60 over 10 and 30 tokens -> batch loss (10+60)/(10+30) = 1.75.
    # A plain mean would report (1.0+2.0)/2 = 1.5.
    chunks = [(1.0, 10.0, 10.0, 1), (2.0, 30.0, 20.0, 3)]

    class _Policy:
        def __init__(self):
            self._it, self._acc, self.grad_accum, self.metric_map_calls = 0, 0, 2, 0

        def iteration(self):
            return self._it

        def needs_old_logprobs(self):
            """Ahead-pass capability probe (`entrypoints/grpo.py:461`).

            The driver asks every policy cell whether it can record pi_old before the training
            passes, and only when `rl.needs_cached_old_logprobs` says the objective consults it.
            These fixtures drive the plain train loop, so they answer False -- a fake that omits the
            method fails with AttributeError from inside the driver, which reads as a production bug.
            """
            return False

        def training_metric_weight_map(self) -> dict[str, str]:
            self.metric_map_calls += 1
            return {
                "train/loss_avg": "train/weight_sum",
                "train/diagnostic_avg": "train/valid_sequence_count",
            }

        def train(self, batch, *, mini=0, micro=0, n_mini=1, n_micro=1):
            del batch, mini, n_mini, n_micro
            loss, denom, diagnostic, sequences = chunks[micro]
            n["train"] += 1
            self._acc += 1
            if self._acc == self.grad_accum:
                self._acc, self._it = 0, self._it + 1
            return {
                "train/loss_avg": loss,
                "train/weight_sum": denom,
                "train/diagnostic_avg": diagnostic,
                "train/valid_sequence_count": sequences,
            }

        def save(self, path):
            pass

    policy = _Policy()
    grpo.loop(
        _units(policy, n),
        _config(max_steps=1, prompts_per_step=2, group_size=2, n_mini=1, n_micro=2, seed=0, save_every=0),
        "/tmp/out",
    )
    assert reported[-1]["train/loss_avg"] == pytest.approx(1.75)  # token-weighted, NOT 1.5
    assert reported[-1]["train/diagnostic_avg"] == pytest.approx(17.5)
    assert reported[-1]["train/weight_sum"] == pytest.approx(40.0)
    assert reported[-1]["train/valid_sequence_count"] == 4
    assert policy.metric_map_calls == 1


@pytest.mark.L0
@pytest.mark.CPU
def test_grpo_loop_resume_converts_optimizer_steps_to_batch_steps(monkeypatch):
    """TWO CLOCKS. `policy.iteration()` counts OPTIMIZER steps; this loop counts COLLECTED BATCHES,
    and now takes `n_mini` optimizer steps per batch. Feeding the optimizer count straight in makes a
    resumed mini-batched run start n_mini x too far along and silently skip most of its remaining
    batches — no error, just a short run. Restoring optimizer step 4 at n_mini=2 must resume at
    BATCH step 2, so a 3-batch run has exactly 1 batch left."""
    from cosmos_framework.posttrain.entrypoints import grpo

    _mock_env(monkeypatch)
    n = {"train": 0}
    monkeypatch.setattr("cosmos_framework.posttrain.utils.report.make_reporters", lambda *_a, **_k: [])

    class _Policy:
        def __init__(self):
            self._it, self._acc, self.grad_accum = 4, 0, 1  # restored: 4 optimizer steps done

        def iteration(self):
            return self._it

        def needs_old_logprobs(self):
            """Ahead-pass capability probe (`entrypoints/grpo.py:461`).

            The driver asks every policy cell whether it can record pi_old before the training
            passes, and only when `rl.needs_cached_old_logprobs` says the objective consults it.
            These fixtures drive the plain train loop, so they answer False -- a fake that omits the
            method fails with AttributeError from inside the driver, which reads as a production bug.
            """
            return False

        def train(self, batch, **_control):
            n["train"] += 1
            self._it += 1
            return {"train/loss_avg": 0.5}

        def save(self, path):
            pass

    summary = grpo.loop(
        _units(_Policy(), n),
        _config(max_steps=3, prompts_per_step=2, group_size=2, n_mini=2, n_micro=1, seed=0, save_every=0),
        "/tmp/out",
    )
    assert summary["start"] == 2  # batch step 2, NOT optimizer step 4
    assert n["train"] == 2  # one remaining batch x n_mini=2 optimizer steps


@pytest.mark.L0
@pytest.mark.CPU
def test_grpo_loop_resume_rejects_a_mid_batch_checkpoint(monkeypatch):
    """A batch boundary is a whole number of mini-batches, so a remainder means the checkpoint was
    written mid-batch and the batch step it maps to is ambiguous. Fail loud rather than round."""
    from cosmos_framework.posttrain.entrypoints import grpo

    _mock_env(monkeypatch)
    n = {"train": 0}
    monkeypatch.setattr("cosmos_framework.posttrain.utils.report.make_reporters", lambda *_a, **_k: [])

    class _Policy:
        def __init__(self):
            self._it, self.grad_accum = 3, 1  # 3 is NOT a multiple of n_mini=2

        def iteration(self):
            return self._it

        def needs_old_logprobs(self):
            """Ahead-pass capability probe (`entrypoints/grpo.py:461`).

            The driver asks every policy cell whether it can record pi_old before the training
            passes, and only when `rl.needs_cached_old_logprobs` says the objective consults it.
            These fixtures drive the plain train loop, so they answer False -- a fake that omits the
            method fails with AttributeError from inside the driver, which reads as a production bug.
            """
            return False

        def train(self, batch, **_control):
            return {"train/loss_avg": 0.5}

        def save(self, path):
            pass

    with pytest.raises(RuntimeError, match="not a multiple of rl.n_mini"):
        grpo.loop(
            _units(_Policy(), n),
            _config(max_steps=3, prompts_per_step=2, group_size=2, n_mini=2, n_micro=1, seed=0, save_every=0),
            "/tmp/out",
        )


# --------------------------------------------------------------------------- §8.3 generalized sizing


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    "batch,group,r_data,expected",
    [
        (16, 8, 2, 1),
        (32, 8, 4, 1),
        (32, 8, 2, 2),
    ],
)
def test_prompts_per_data_cell_asymmetric(batch: int, group: int, r_data: int, expected: int) -> None:
    """The sole policy execution replica's batch is divided across rollout/data replicas."""
    from cosmos_framework.posttrain.control.loop.batching import prompts_per_data_cell

    assert (
        prompts_per_data_cell(
            policy_batch_size=batch,
            rollouts_per_prompt=group,
            data_replicas=r_data,
        )
        == expected
    )


@pytest.mark.L0
@pytest.mark.CPU
def test_prompts_per_data_cell_rejects_nonpositive_inputs() -> None:
    from cosmos_framework.posttrain.control.loop.batching import prompts_per_data_cell

    with pytest.raises(ValueError, match="must be positive"):
        prompts_per_data_cell(
            policy_batch_size=0,
            rollouts_per_prompt=8,
            data_replicas=2,
        )


@pytest.mark.L0
@pytest.mark.CPU
def test_prompts_per_data_cell_rejects_non_dividing_data_cells():
    """total_prompts must divide by R_data (each data cell draws a whole number)."""
    from cosmos_framework.posttrain.control.loop.batching import prompts_per_data_cell

    with pytest.raises(ValueError, match="data_replicas|data cell|prompts"):
        prompts_per_data_cell(
            policy_batch_size=8,
            rollouts_per_prompt=8,
            data_replicas=3,
        )


@pytest.mark.L0
@pytest.mark.CPU
def test_grpo_final_drain_joins_publishers_once_after_the_final_save(monkeypatch: pytest.MonkeyPatch) -> None:
    """The in-loop drain runs exactly once, at the last step, between that step's save and the trailing
    offload, and this controller loop takes no host-memory reading of its own.

    canM7 (job 461919) lost its four policy actors to the cgroup OOM killer inside this drain. The
    readings that explain the next incident are the worker's (CheckpointManager samples at every save
    and at its finalize); a reading from this process would be a different RSS and, under a multi-node
    or disaggregated placement, a different cgroup from the one the kernel enforced."""
    from cosmos_framework.posttrain.entrypoints import grpo
    from cosmos_framework.posttrain.utils import host_memory

    _mock_env(monkeypatch)
    events: list[str] = []
    monkeypatch.setattr(
        host_memory,
        "log_host_memory",
        lambda context, **_kwargs: events.append(f"host_memory:{context}"),
    )

    class _Policy:
        def __init__(self) -> None:
            self._it = 0

        def iteration(self) -> int:
            return self._it

        def needs_old_logprobs(self) -> bool:
            return False

        def train(self, batch: Any, **_control: Any) -> dict[str, float]:
            del batch
            self._it += 1
            return {"train/loss_avg": 0.5}

        def save(self, path: str) -> None:
            events.append(f"save:{Path(path).name}")

        def offload(self) -> float:
            events.append("offload")
            return 1.0

        def onload(self) -> float:
            events.append("onload")
            return 1.0

        def finalize_checkpoints(self) -> None:
            events.append("finalize_checkpoints")

    policy = _Policy()
    config = _config(max_steps=2, prompts_per_step=1, group_size=2, seed=0, save_every=1)
    config.grid.gpu_placement = "colocated"

    summary = grpo.loop(_units(policy, {}), config, "/tmp/out")

    assert summary["steps"] == 2
    assert events == [
        "offload",  # after the initial sync: both bodies were built resident
        # step 1: the update saves while it still holds the card, then publication hands it back
        "onload",
        "save:iter_000001",
        "offload",
        "offload",  # generic trailing handoff is idempotent in the real trainer
        # step 2 (final): the same, then the loop sleeps rollout again for the publisher drain
        "onload",
        "save:iter_000002",
        "offload",
        "finalize_checkpoints",
        "offload",
    ]
    assert not [event for event in events if event.startswith("host_memory:")], (
        "the controller process took a host-memory reading; that series belongs to the checkpoint worker"
    )
