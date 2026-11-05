# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""CPU acceptance tests for the colocated online-NFT transaction."""

from __future__ import annotations

import inspect
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.posttrain.algorithm.annotators.advantage import distributed_group_relative
from cosmos_framework.posttrain.algorithm.nft import NFTObjectiveConfig, make_objective, optimality_weight_std
from cosmos_framework.posttrain.algorithm.reward import RewardCombiner
from cosmos_framework.posttrain.checkpoint import load_component
from cosmos_framework.posttrain.config.build import build_wfm_lr_scheduler
from cosmos_framework.posttrain.config.rl import NFTConfig, RefConfig, VideoRewardConfig, WFMLRSchedulerConfig
from cosmos_framework.posttrain.control import ExperienceBuffer
from cosmos_framework.posttrain.control.loop.weight_sync import weight_sync
from cosmos_framework.posttrain.entrypoints import nft
from cosmos_framework.posttrain.entrypoints.nft import (
    RejectedNFTUpdate,
    build_shared_bodies,
    online_nft_step,
    restore_wfm_source,
    validate_nft_dataloader,
    validate_nft_runtime_config,
)
from cosmos_framework.posttrain.roles.env import WFMDataEnv
from cosmos_framework.posttrain.roles.reward import ProviderMetricReport, RewardWorker
from cosmos_framework.posttrain.roles.reward.video import (
    VideoRewardError,
    VideoRewardRequest,
    VideoRewardResult,
    build_video_reward_provider,
)
from cosmos_framework.posttrain.roles.rollout import WFMOnlineRollout
from cosmos_framework.posttrain.roles.trainer import DiffusionTrainer
from cosmos_framework.posttrain.schema import (
    WFM_CONDITIONING_KEY,
    ObsBundle,
    Trajectory,
    Transition,
    WFMConditioningInput,
)
from cosmos_framework.posttrain.tests.helpers.distributed import init_gloo_process_group
from cosmos_framework.posttrain.tests.helpers.toy_model import ToyVideoFlowModel
from cosmos_framework.posttrain.utils.rank_state import state_for_current_rank


class _Loader:
    def __init__(self) -> None:
        self.start = 0

    def set_start_iteration(self, iteration: int) -> None:
        self.start = iteration

    def __iter__(self) -> Iterator[dict[str, object]]:
        index = self.start
        while True:
            yield {
                "video": [torch.zeros(1, 2)],  # one sample [C,T]
                "ai_caption": [f"prompt-{index}"],
                "x": torch.tensor([[0.2 + index, -0.4]]),  # [B=1,D]
                "fps": torch.tensor([16.0]),  # [B]
            }
            index += 1


@pytest.mark.L0
@pytest.mark.CPU
def test_online_nft_step_is_topology_and_callsite_dispatch_agnostic() -> None:
    source = inspect.getsource(online_nft_step)
    for forbidden in ("world_size_or_1", "RankAligned", "execution_world", "local_group", ".ranks"):
        assert forbidden not in source


class _RewardClient:
    def __init__(self, *, timeout: bool = False, fail_on_call: int | None = None) -> None:
        self.timeout = timeout
        self.fail_on_call = fail_on_call
        self.calls = 0
        self.seeds: list[int] = []
        self.poll_attempts = 1
        self.transport_retries = 0

    def score(self, request: VideoRewardRequest, *, deadline_monotonic: float) -> VideoRewardResult:
        del deadline_monotonic
        self.calls += 1
        self.seeds.append(request.seed)
        if self.timeout or self.calls == self.fail_on_call:
            raise VideoRewardError("timeout", "test timeout")
        return VideoRewardResult(
            score=float(request.seed % 10_000),
            service_id=f"service-{request.sample_id}",
        )


def _units(
    seed: int = 0,
    *,
    timeout: bool = False,
    fail_on_reward_call: int | None = None,
    grad_accum: int = 1,
) -> tuple[dict[str, object], _RewardClient]:
    model = ToyVideoFlowModel(dim=2, seed=seed)
    optimizer = torch.optim.AdamW(model.net.parameters(), lr=1e-3)
    scheduler = build_wfm_lr_scheduler(
        optimizer,
        WFMLRSchedulerConfig(warm_up_steps=0, cycle_length=10, f_start=1.0, f_max=1.0, f_min=1.0),
    )
    trainer = DiffusionTrainer(
        model,
        make_objective(NFTObjectiveConfig()),
        optimizer,
        scheduler=scheduler,
        update_ema_after_iter=False,
        grad_accum=grad_accum,
        update_old_every_iter=2,
        update_ref_every_iter=0,
    )
    source = WFMDataEnv(_Loader(), grad_accum=grad_accum)
    rollout = WFMOnlineRollout(
        model,
        sampler=object(),
        num_steps=2,
        shift=5.0,
        guidance=3.0,
        group_size=4,
    )
    client = _RewardClient(timeout=timeout, fail_on_call=fail_on_reward_call)
    provider = build_video_reward_provider(
        VideoRewardConfig(provider_revision="test-dance-grpo-v1"),
        client,
        model.video_reward_decoder(),
        run_id="test-run",
    )
    reward = RewardWorker(
        [provider],
        RewardCombiner(),
        annotators=[distributed_group_relative(eps=1e-4, group_size=4, correction=1)],
        training_group_size=4,
    )
    return {
        "data": source,
        "rollout": rollout,
        "rm": reward,
        "replay": ExperienceBuffer(exact_batch_size=4),
        "policy": trainer,
    }, client


@pytest.mark.L0
@pytest.mark.CPU
def test_online_nft_step_applies_update_and_commits_source() -> None:
    units, _client = _units()
    trainer = units["policy"]
    assert isinstance(trainer, DiffusionTrainer)
    before = [parameter.detach().clone() for parameter in trainer.model.net.parameters()]
    metrics = online_nft_step(units, NFTConfig(max_steps=1), step=0)
    assert metrics["optim/update_outcome"] == "applied"
    assert metrics["train/reward_provider/dance_grpo/sample_count"] == 4
    assert metrics["train/reward_provider/dance_grpo/raw_score"] == pytest.approx(1.5)
    assert metrics["rollout/reward_group_mean"] == pytest.approx(1.5)
    assert metrics["rollout/reward_group_spread"] == pytest.approx(3.0)
    assert metrics["rollout/reward_nonzero_fraction"] == pytest.approx(0.75)
    assert metrics["rollout/reward_group_size"] == 4
    assert metrics["rollout/reward_group_informative"] == 1
    assert metrics["rollout/reward_group_all_zero"] == 0
    assert metrics["train/optimality_std"] == pytest.approx(
        optimality_weight_std(metrics["train/optimality_mean"], metrics["train/optimality_second_moment"])
    )
    assert metrics["train/optimality_std"] > 0.0
    assert trainer.iteration() == 1
    assert metrics["train/learning_rate"] > 1e-9
    # The toy adapter exposes no maintained EMA; the entrypoint follows the model's
    # configured EMA capability instead of treating any stale attribute as enabled.
    assert trainer.model.ema_iterations == []
    source = units["data"]
    assert isinstance(source, WFMDataEnv)
    assert source.state_dict() == {
        "version": 1,
        "resume_kind": "optimizer_window_rows",
        "committed_rows": 1,
        "optimizer_step": 1,
        "grad_accum": 1,
        "source_fingerprint": "",
    }
    assert any(not torch.equal(old, new) for old, new in zip(before, trainer.model.net.parameters()))


@pytest.mark.L0
@pytest.mark.CPU
def test_online_nft_optimizer_step_seed_scope_reuses_seed_across_accumulation_rows() -> None:
    units, client = _units(grad_accum=4)
    rl = NFTConfig(max_steps=1, grad_accum=4, rollout_seed_clock="optimizer_step")

    online_nft_step(units, rl, step=0)

    assert client.seeds == [0, 1, 2, 3] * 4


@pytest.mark.L0
@pytest.mark.CPU
def test_reward_timeout_prevents_replay_update_and_cursor_commit() -> None:
    units, client = _units(timeout=True)
    trainer = units["policy"]
    source = units["data"]
    replay = units["replay"]
    assert isinstance(trainer, DiffusionTrainer)
    assert isinstance(source, WFMDataEnv)
    assert isinstance(replay, ExperienceBuffer)
    before = [parameter.detach().clone() for parameter in trainer.model.net.parameters()]
    with pytest.raises(VideoRewardError, match="test timeout"):
        online_nft_step(units, NFTConfig(max_steps=1), step=0)
    assert trainer.iteration() == 0
    assert source.state_dict()["committed_rows"] == 0
    assert len(replay) == 0
    assert all(torch.equal(old, new) for old, new in zip(before, trainer.model.net.parameters()))


@pytest.mark.L0
@pytest.mark.CPU
def test_online_nft_accumulation_commits_one_window_and_refreshes_old_on_optimizer_cadence() -> None:
    units, _client = _units(grad_accum=2)
    trainer = units["policy"]
    source = units["data"]
    rollout = units["rollout"]
    assert isinstance(trainer, DiffusionTrainer)
    assert isinstance(source, WFMDataEnv)
    assert isinstance(rollout, WFMOnlineRollout)

    first = online_nft_step(units, NFTConfig(max_steps=2, grad_accum=2), step=0)
    assert first["optim/update_outcome"] == "applied"
    assert first["rollout/source_window_rows"] == 2
    assert first["rollout/source_committed_rows"] == 2
    assert first["rollout/reward_trajectory_count"] == 8
    assert first["train/reward_provider/dance_grpo/sample_count"] == 8
    assert first["optim/old_policy_refreshed"] == 0.0
    assert trainer.iteration() == 1
    assert rollout._weight_version == 1
    assert source.state_dict()["optimizer_step"] == 1
    assert any(
        not torch.equal(old, policy)
        for old, policy in zip(trainer.model.net_old.parameters(), trainer.model.net.parameters(), strict=True)
    )

    second = online_nft_step(units, NFTConfig(max_steps=2, grad_accum=2), step=1)
    assert second["optim/old_policy_refreshed"] == 1.0
    assert trainer.iteration() == 2
    assert rollout._weight_version == 2
    assert source.state_dict()["committed_rows"] == 4
    assert all(
        torch.equal(old, policy)
        for old, policy in zip(trainer.model.net_old.parameters(), trainer.model.net.parameters(), strict=True)
    )


@pytest.mark.L0
@pytest.mark.CPU
def test_online_nft_late_microbatch_failure_aborts_gradient_and_source_window(tmp_path: Path) -> None:
    units, client = _units(grad_accum=3, fail_on_reward_call=9)
    trainer = units["policy"]
    source = units["data"]
    replay = units["replay"]
    assert isinstance(trainer, DiffusionTrainer)
    assert isinstance(source, WFMDataEnv)
    assert isinstance(replay, ExperienceBuffer)
    before = [parameter.detach().clone() for parameter in trainer.model.net.parameters()]

    with pytest.raises(VideoRewardError, match="test timeout"):
        online_nft_step(units, NFTConfig(max_steps=1, grad_accum=3), step=0)

    assert client.calls == 9
    assert trainer.iteration() == 0
    assert trainer._accum == 0
    assert trainer._attempt_id == 0
    assert all(parameter.grad is None for parameter in trainer.model.net.parameters())
    assert source.state_dict()["committed_rows"] == 0
    assert len(replay) == 0
    assert all(torch.equal(old, new) for old, new in zip(before, trainer.model.net.parameters(), strict=True))

    retried = online_nft_step(units, NFTConfig(max_steps=1, grad_accum=3), step=0)
    assert retried["optim/update_outcome"] == "applied"
    assert retried["optim/attempt_id"] == 1
    assert trainer.iteration() == 1
    checkpoint = tmp_path / "iter_000001"
    trainer.save(str(checkpoint))
    restored_units, _restored_client = _units(grad_accum=3)
    restored = restored_units["policy"]
    assert isinstance(restored, DiffusionTrainer)
    assert restored.load(str(checkpoint)) == 1
    assert restored._attempt_id == 1


@pytest.mark.L0
@pytest.mark.CPU
def test_nft_loop_does_not_publish_rejected_attempt_as_a_model_step(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    units, _client = _units()
    source = units["data"]
    assert isinstance(source, WFMDataEnv)

    class _RejectedPolicy:
        def iteration(self) -> int:
            return 0

        def train(self, _batch: list[Trajectory]) -> dict[str, object]:
            return {
                "optim/update_outcome": "skipped",
                "optim/skip_reason": "nonfinite_gradient",
                "optim/nonfinite_skips_total": 1,
            }

    class _Reporter:
        def start(self) -> None:
            return None

        def report(self, metrics: dict[str, object], step: int) -> None:
            reported.append((step, dict(metrics)))

        def finish(self) -> None:
            return None

    reported: list[tuple[int, dict[str, object]]] = []
    reporter_options: dict[str, object] = {}

    def make_reporters(
        _config: object,
        _out_dir: str,
        total_steps: int | None = None,
        headline_keys: tuple[str, ...] = (),
    ) -> list[_Reporter]:
        reporter_options.update(total_steps=total_steps, headline_keys=headline_keys)
        return [_Reporter()]

    units["policy"] = _RejectedPolicy()
    monkeypatch.setattr("cosmos_framework.utils.lazy_config.instantiate", lambda value: value)
    monkeypatch.setattr("cosmos_framework.posttrain.utils.report.make_reporters", make_reporters)
    monkeypatch.setattr(
        "cosmos_framework.posttrain.control.termination.TerminationWatch",
        lambda: SimpleNamespace(should_stop=lambda: None),
    )
    config = SimpleNamespace(rl=NFTConfig(max_steps=1, save_every=0))

    with pytest.raises(RejectedNFTUpdate, match="nonfinite_gradient"):
        nft.loop(units, config, str(tmp_path))

    assert source.state_dict()["committed_rows"] == 0
    assert reporter_options["total_steps"] == 1
    assert reporter_options["headline_keys"] == (
        "train/loss_avg",
        "train/policy_loss",
        "train/kl_loss",
        "rollout/reward_group_mean",
    )
    assert reported == []


@pytest.mark.L0
@pytest.mark.CPU
def test_nft_loop_publishes_primitive_metrics_at_committed_model_step(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    units, _client = _units()
    reported: list[tuple[int, dict[str, object]]] = []

    class _Reporter:
        def start(self) -> None:
            return None

        def report(self, metrics: dict[str, object], step: int) -> None:
            reported.append((step, dict(metrics)))

        def finish(self) -> None:
            return None

    monkeypatch.setattr("cosmos_framework.utils.lazy_config.instantiate", lambda value: value)
    monkeypatch.setattr(
        "cosmos_framework.posttrain.utils.report.make_reporters",
        lambda *_args, **_kwargs: [_Reporter()],
    )
    monkeypatch.setattr(
        "cosmos_framework.posttrain.control.termination.TerminationWatch",
        lambda: SimpleNamespace(should_stop=lambda: None),
    )

    summary = nft.loop(units, SimpleNamespace(rl=NFTConfig(max_steps=1, save_every=0)), str(tmp_path))

    assert summary["steps"] == 1
    assert len(reported) == 1
    model_step, metrics = reported[0]
    assert model_step == 1
    assert all(isinstance(value, (int, float, str, bool)) for value in metrics.values())
    assert all(key.split("/", 1)[0] in {"train", "rollout", "val", "optim", "timer"} for key in metrics)
    assert metrics["train/reward_provider/dance_grpo/sample_count"] == 4


@pytest.mark.L0
@pytest.mark.CPU
def test_identical_metric_frame_scalarizes_before_rank_gather(monkeypatch: pytest.MonkeyPatch) -> None:
    gathered_inputs: list[list[dict[str, object]]] = []

    def gather_rank_states(state: list[dict[str, object]], *, context: str) -> dict[int, object]:
        assert context == "online NFT trainer metric-frame gathering"
        gathered_inputs.append(state)
        return {0: state}

    monkeypatch.setattr(
        "cosmos_framework.posttrain.utils.rank_state.gather_rank_states",
        gather_rank_states,
    )

    frame = nft._identical_metric_frame(
        {"train/loss_avg": torch.tensor(1.5)},
        operation="online NFT trainer",
    )

    assert frame == {"train/loss_avg": 1.5}
    assert gathered_inputs == [[{"train/loss_avg": 1.5}]]
    assert all(not isinstance(value, torch.Tensor) for value in gathered_inputs[0][0].values())


@pytest.mark.L0
@pytest.mark.CPU
def test_nft_loop_checkpoint_lifecycle_covers_cadence_preemption_and_final_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class _LoopPolicy:
        def __init__(self) -> None:
            self._iteration = 0
            self.saved: list[tuple[str, int]] = []

        def iteration(self) -> int:
            return self._iteration

        def advance(self) -> None:
            self._iteration += 1

        def save(self, path: str, data_state: dict[str, int], **_kwargs: object) -> None:
            self.saved.append((Path(path).name, data_state["committed_rows"]))

    class _LoopData:
        def __init__(self, policy: _LoopPolicy) -> None:
            self.policy = policy

        def state_dict(self) -> dict[str, int]:
            return {"committed_rows": self.policy.iteration()}

    monkeypatch.setattr("cosmos_framework.utils.lazy_config.instantiate", lambda value: value)
    monkeypatch.setattr("cosmos_framework.posttrain.utils.report.make_reporters", lambda *_args, **_kwargs: [])

    def run_case(
        *, max_steps: int, save_every: int, stop_reasons: list[str | None]
    ) -> tuple[dict[str, object], list[tuple[str, int]]]:
        policy = _LoopPolicy()
        units: dict[str, object] = {"policy": policy, "data": _LoopData(policy)}
        reasons = iter(stop_reasons)

        def applied_step(_units: dict[str, object], _rl: NFTConfig, *, step: int) -> dict[str, float | str]:
            assert step == policy.iteration()
            policy.advance()
            return {"optim/update_outcome": "applied", "train/loss_avg": float(policy.iteration())}

        monkeypatch.setattr(nft, "online_nft_step", applied_step)
        monkeypatch.setattr(
            "cosmos_framework.posttrain.control.termination.TerminationWatch",
            lambda: SimpleNamespace(should_stop=lambda: next(reasons)),
        )
        summary = nft.loop(
            units,
            SimpleNamespace(rl=NFTConfig(max_steps=max_steps, save_every=save_every)),
            str(tmp_path),
        )
        return summary, policy.saved

    cadence_summary, cadence_saves = run_case(max_steps=2, save_every=1, stop_reasons=[None, None])
    assert cadence_summary["final_loss"] == 2.0
    assert cadence_saves == [("iter_000001", 1), ("iter_000002", 2)]

    preempted, preempted_saves = run_case(max_steps=3, save_every=10, stop_reasons=["signal"])
    assert preempted["preempted"] == "signal" and preempted_saves == [("iter_000001", 1)]

    final_summary, final_saves = run_case(max_steps=2, save_every=3, stop_reasons=[None, None])
    assert "preempted" not in final_summary
    assert final_saves == [("iter_000002", 2)]


@pytest.mark.L0
@pytest.mark.CPU
def test_nft_loop_disables_failed_reporter_without_aborting_update(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    units, _client = _units()
    healthy_rows: list[tuple[int, dict[str, object]]] = []

    class _FailedReporter:
        def start(self) -> None:
            return None

        def report(self, _metrics: dict[str, object], _step: int) -> None:
            raise OSError("W&B unavailable")

        def finish(self) -> None:
            raise AssertionError("failed reporters must not be retried during shutdown")

    class _HealthyReporter:
        def start(self) -> None:
            return None

        def report(self, metrics: dict[str, object], step: int) -> None:
            healthy_rows.append((step, dict(metrics)))

        def finish(self) -> None:
            return None

    monkeypatch.setattr("cosmos_framework.utils.lazy_config.instantiate", lambda value: value)
    monkeypatch.setattr(
        "cosmos_framework.posttrain.utils.report.make_reporters",
        lambda *_args, **_kwargs: [_FailedReporter(), _HealthyReporter()],
    )
    monkeypatch.setattr(
        "cosmos_framework.posttrain.control.termination.TerminationWatch",
        lambda: SimpleNamespace(should_stop=lambda: None),
    )

    summary = nft.loop(units, SimpleNamespace(rl=NFTConfig(max_steps=1, save_every=0)), str(tmp_path))

    assert summary["final_loss"] is not None
    assert [step for step, _metrics in healthy_rows] == [1]


@pytest.mark.L0
@pytest.mark.CPU
def test_nft_loop_rejects_checkpoint_beyond_configured_total(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    units = {"policy": SimpleNamespace(iteration=lambda: 2)}
    monkeypatch.setattr("cosmos_framework.utils.lazy_config.instantiate", lambda value: value)

    with pytest.raises(RuntimeError, match="restored trainer iteration 2 exceeds configured total steps 1"):
        nft.loop(units, SimpleNamespace(rl=NFTConfig(max_steps=1)), str(tmp_path))


@pytest.mark.L0
@pytest.mark.CPU
def test_online_nft_combined_trainer_and_source_continuation(tmp_path: Path) -> None:
    units, _client = _units(seed=7)
    online_nft_step(units, NFTConfig(max_steps=2), step=0)
    trainer = units["policy"]
    source = units["data"]
    assert isinstance(trainer, DiffusionTrainer)
    assert isinstance(source, WFMDataEnv)
    checkpoint = str(tmp_path / "iter_000001")
    trainer.save(checkpoint, source.state_dict())

    restored, _restored_client = _units(seed=7)
    restored_trainer = restored["policy"]
    restored_source = restored["data"]
    assert isinstance(restored_trainer, DiffusionTrainer)
    assert isinstance(restored_source, WFMDataEnv)
    assert restored_trainer.load(checkpoint) == 1
    data_state = state_for_current_rank(load_component(checkpoint, "data"), on_world_size_mismatch="raise")
    assert data_state is not None
    assert restored_source.load_state_dict(data_state)
    assert restored_source.get_samples(1)[0].id == "wfm_source_00000001"
    weight_sync(restored["rollout"], restored_trainer)
    metrics = online_nft_step(restored, NFTConfig(max_steps=2), step=1)
    assert metrics["optim/update_outcome"] == "applied"
    assert restored_trainer.iteration() == 2
    assert restored_source.state_dict()["committed_rows"] == 2


@pytest.mark.L0
@pytest.mark.CPU
def test_build_shared_bodies_restores_source_then_publishes_restored_weight_version(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = SimpleNamespace(
        dataloader_train={"_target_": "test.wfm_loader", "inventory": "fixture-v1"},
        job=SimpleNamespace(project="project", group="group", name="nft-resume-test"),
    )
    rl = NFTConfig(
        max_steps=2,
        save_async=False,
        reward=VideoRewardConfig(provider_revision="dance-grpo@git:0123456789abcdef"),
    )
    config.rl = rl

    def instantiate(value: object) -> object:
        return _Loader() if value is config.dataloader_train else value

    def build_backbone(_config: object, _world_size: int) -> ToyVideoFlowModel:
        model = ToyVideoFlowModel(dim=2, seed=11)
        model.net_ema = None
        return model

    monkeypatch.setattr("cosmos_framework.utils.lazy_config.instantiate", instantiate)
    monkeypatch.setattr(
        "cosmos_framework.posttrain.config.build.build_backbone",
        build_backbone,
    )
    monkeypatch.setattr(
        "cosmos_framework.posttrain.config.build.build_optimizer",
        lambda model, _config: torch.optim.AdamW(model.net.parameters(), lr=1e-3),
    )
    monkeypatch.setattr(
        "cosmos_framework.posttrain.config.build.build_wfm_lr_scheduler",
        lambda _optimizer, _config: None,
    )
    monkeypatch.setattr(
        "cosmos_framework.posttrain.model.cosmos3.Cosmos3FlowPolicy",
        lambda model, *, build_ref, build_old: model,
    )
    monkeypatch.setattr(
        "cosmos_framework.posttrain.model.unipc.build_unipc_sampler",
        lambda _model: object(),
    )
    units = build_shared_bodies(config, rl, reward_client=_RewardClient())
    assert units["data"].state_dict()["version"] == 2
    assert units["data"].state_dict()["seed_contract"] == {
        "version": 1,
        "base_seed": rl.seed,
        "clock": "source_row",
        "stride": 10_000,
        "group_size": rl.group_size,
    }
    online_nft_step(units, rl, step=0)
    checkpoint = str(tmp_path / "iter_000001")
    units["policy"].save(checkpoint, units["data"].state_dict())

    restored = build_shared_bodies(
        config,
        rl,
        resume_checkpoint=checkpoint,
        reward_client=_RewardClient(),
    )
    assert restored["policy"].iteration() == 1
    assert restored["rollout"]._weight_version == 1
    assert restored["data"].state_dict()["committed_rows"] == 1
    metrics = online_nft_step(restored, rl, step=1)
    assert metrics["optim/update_outcome"] == "applied"
    assert restored["policy"].iteration() == 2
    assert restored["rollout"]._weight_version == 2
    assert restored["data"].state_dict()["committed_rows"] == 2


@pytest.mark.L0
@pytest.mark.CPU
def test_build_shared_bodies_defers_only_initialized_frozen_reference_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = SimpleNamespace(
        dataloader_train={"_target_": "test.wfm_loader"},
        job=SimpleNamespace(project="project", group="group", name="nft-reference-identity-test"),
    )
    captured: list[bool] = []
    captured_policy_loss_scales: list[float] = []

    class _CapturedTrainer(RuntimeError):
        pass

    def capture_trainer(*_args: object, **kwargs: object) -> None:
        captured.append(bool(kwargs["defer_frozen_reference_identity"]))
        raise _CapturedTrainer

    def capture_objective(config: NFTObjectiveConfig) -> object:
        captured_policy_loss_scales.append(config.policy_loss_scale)
        return object()

    def build_backbone(_config: object, _world_size: int) -> ToyVideoFlowModel:
        model = ToyVideoFlowModel(dim=2, seed=13)
        model.net_ema = None
        return model

    monkeypatch.setattr(
        "cosmos_framework.posttrain.config.build.build_backbone",
        build_backbone,
    )
    monkeypatch.setattr(
        "cosmos_framework.posttrain.config.build.build_optimizer",
        lambda model, _config: torch.optim.AdamW(model.net.parameters(), lr=1e-3),
    )
    monkeypatch.setattr(
        "cosmos_framework.posttrain.config.build.build_wfm_lr_scheduler",
        lambda _optimizer, _config: None,
    )
    monkeypatch.setattr(
        "cosmos_framework.posttrain.model.cosmos3.Cosmos3FlowPolicy",
        lambda model, *, build_ref, build_old: model,
    )
    monkeypatch.setattr("cosmos_framework.posttrain.roles.trainer.DiffusionTrainer", capture_trainer)
    monkeypatch.setattr("cosmos_framework.posttrain.algorithm.nft.make_objective", capture_objective)
    cases = (
        NFTConfig(init_from="s3://base", max_steps=1, policy_loss_scale=7.0),
        NFTConfig(init_from="", max_steps=1),
        NFTConfig(
            init_from="s3://base",
            max_steps=1,
            kl_beta=0.0,
            ref=RefConfig(build_ref=False, update_every=0),
        ),
    )
    for rl in cases:
        config.rl = rl
        with pytest.raises(_CapturedTrainer):
            build_shared_bodies(config, rl, reward_client=_RewardClient())

    assert captured == [True, False, False]
    assert captured_policy_loss_scales == [7.0, 5.0, 5.0]


@pytest.mark.L0
@pytest.mark.CPU
def test_nft_preflight_checks_real_recipe_and_environment_without_model_allocation() -> None:
    config = SimpleNamespace(
        grid=SimpleNamespace(
            backend="ray",
            grouping="single",
            gpu_placement="colocated",
            num_replicas=1,
            ranks=4,
            gpus_per_role={},
            shapes={},
        ),
        dataloader_train=SimpleNamespace(max_sequence_length=None, max_samples_per_batch=1),
        model=SimpleNamespace(
            config=SimpleNamespace(
                parallelism=SimpleNamespace(
                    data_parallel_replicate_degree=1,
                    data_parallel_shard_degree=4,
                ),
                compile=SimpleNamespace(enabled=False),
                diffusion_expert_config=SimpleNamespace(max_vae_latent_side_after_patchify=45),
                ema=SimpleNamespace(enabled=True, rate=0.1, iteration_shift=0),
                rectified_flow_inference_config=SimpleNamespace(scheduler_type="unipc", shift=1.0),
                rectified_flow_training_config=SimpleNamespace(shift=1),
            )
        ),
    )
    rl = NFTConfig(
        max_steps=3,
        init_from="s3://checkpoint",
        reward=VideoRewardConfig(provider_revision="dance-grpo@git:0123456789abcdef"),
    )
    environment = {
        "REMOTE_REWARD_ENQUEUE_URL": "https://reward/enqueue",
        "REMOTE_REWARD_FETCH_URL": "https://reward/fetch",
        "REMOTE_REWARD_TOKEN": "secret",
    }
    assert validate_nft_runtime_config(config, rl, environ=environment) == []

    unpinned_errors = validate_nft_runtime_config(
        config,
        NFTConfig(max_steps=3, init_from="s3://checkpoint"),
        environ=environment,
    )
    assert any("immutable deployed scorer contract" in error for error in unpinned_errors)

    broken = dict(environment)
    broken.pop("REMOTE_REWARD_TOKEN")
    errors = validate_nft_runtime_config(config, rl, environ=broken)
    assert any("REMOTE_REWARD_TOKEN" in error for error in errors)

    oversized_group_errors = validate_nft_runtime_config(config, NFTConfig(group_size=8), environ=environment)
    assert any("role topology" in error for error in oversized_group_errors)

    overlapping_seed_errors = validate_nft_runtime_config(
        config,
        NFTConfig(group_size=nft._WFM_SEED_STRIDE + 1),
        environ=environment,
    )
    assert any("exceeds the seed stride" in error for error in overlapping_seed_errors)

    first_row = int(rl.seed)
    last_seed_in_first_row = first_row + nft._WFM_SEED_STRIDE - 1
    first_seed_in_second_row = first_row + nft._WFM_SEED_STRIDE
    assert last_seed_in_first_row < first_seed_in_second_row

    parallelism = config.model.config.parallelism
    parallelism.data_parallel_replicate_degree = 2
    parallelism.data_parallel_shard_degree = 2
    assert validate_nft_runtime_config(config, rl, environ=environment) == []
    undersized_group_errors = validate_nft_runtime_config(
        config,
        NFTConfig(
            max_steps=3,
            group_size=2,
            init_from="s3://checkpoint",
            reward=VideoRewardConfig(provider_revision="dance-grpo@git:0123456789abcdef"),
        ),
        environ=environment,
    )
    assert any("role topology" in error for error in undersized_group_errors)
    parallelism.data_parallel_replicate_degree = 1
    parallelism.data_parallel_shard_degree = 4

    config.dataloader_train.max_sequence_length = 8192
    config.dataloader_train.max_samples_per_batch = None
    errors = validate_nft_runtime_config(config, rl, environ=environment)
    assert any("max_sequence_length must be null" in error for error in errors)
    assert any("max_samples_per_batch must be 1" in error for error in errors)
    config.dataloader_train.max_sequence_length = None
    config.dataloader_train.max_samples_per_batch = 1

    config.model.config.rectified_flow_training_config.shift = {"256": 3}
    config.model.config.compile.enabled = True
    config.model.config.diffusion_expert_config.max_vae_latent_side_after_patchify = 20
    config.model.config.ema.rate = 0.2
    errors = validate_nft_runtime_config(config, rl, environ=environment)
    assert any("rectified_flow_training_config.shift" in error for error in errors)
    assert any("compile.enabled" in error for error in errors)
    assert any("max_vae_latent_side_after_patchify" in error for error in errors)
    assert any("power EMA" in error for error in errors)

    config.model.config.rectified_flow_training_config.shift = 1.0
    errors = validate_nft_runtime_config(config, rl, environ=environment)
    assert any("float scalars are not accepted by OmniMoT" in error for error in errors)


@pytest.mark.L0
@pytest.mark.CPU
def test_nft_preflight_prints_fail_when_manifest_bind_fails(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source_smokes: list[object] = []
    monkeypatch.setattr(nft, "render_manifest_validation", lambda _manifest: ("manifest bind failed", False))
    monkeypatch.setattr(nft, "validate_nft_runtime_config", lambda _config, _rl: [])
    monkeypatch.setattr(nft, "validate_nft_dataloader", lambda config: source_smokes.append(config))

    with pytest.raises(SystemExit) as error_info:
        nft._validate_and_exit(SimpleNamespace(rl=NFTConfig()))

    assert error_info.value.code == 1
    assert source_smokes == []
    output = capsys.readouterr().out
    assert "manifest bind failed" in output
    assert "NFT preflight: FAIL" in output
    assert "NFT preflight: PASS" not in output


@pytest.mark.L0
@pytest.mark.CPU
def test_nft_run_binds_one_ray_gpu_actor_group_without_driver_process_group(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from cosmos_framework.posttrain.control import bind as bind_module
    from cosmos_framework.posttrain.control import placement as placement_module
    from cosmos_framework.posttrain.utils import entry as entry_utils

    config = SimpleNamespace(
        grid=SimpleNamespace(
            backend="ray",
            grouping="single",
            gpu_placement="colocated",
            num_replicas=1,
            ranks=4,
            shapes={},
            gpus_per_role={},
        ),
        rl=NFTConfig(group_size=4),
    )
    report = SimpleNamespace(format=lambda: "ok", raise_for_status=lambda: None)
    captured: dict[str, object] = {}
    original_declare = nft.declare_wgs

    def declare(*args: object, **kwargs: object) -> dict[str, object]:
        wgs = original_declare(*args, **kwargs)
        wgs["policy"].body = SimpleNamespace(finalize_checkpoints=lambda: None)
        return wgs

    def bind(groups: list[object], **kwargs: object) -> None:
        captured["groups"] = groups
        captured["kwargs"] = kwargs

    monkeypatch.setattr(bind_module, "bind_check", lambda _manifest: report)
    monkeypatch.setattr(placement_module, "bind_device_groups", bind)
    monkeypatch.setattr(
        entry_utils,
        "pin_run_attempt",
        lambda *_args: SimpleNamespace(out_dir=str(tmp_path), opts=[], resume_checkpoint=None),
    )
    monkeypatch.setattr("cosmos_framework.utils.lazy_config.instantiate", lambda value: value)
    monkeypatch.setattr(
        "cosmos_framework.posttrain.config.build.resolve_policy_parallelism",
        lambda _config, _ranks: (1, 4),
    )
    monkeypatch.setattr(nft, "validate_nft_runtime_config", lambda _config, _rl: [])
    monkeypatch.setattr(nft, "dump_resolved", lambda *_args: None)
    monkeypatch.setattr(nft, "declare_wgs", declare)
    monkeypatch.setattr(nft, "loop", lambda *_args: captured.setdefault("loop", True))

    nft.run(config, "config.py", [], str(tmp_path))

    groups = captured["groups"]
    assert isinstance(groups, list) and len(groups) == 1
    assert groups[0].device == "gpu"
    assert len(groups[0].actor_groups) == 1
    assert len(groups[0].actor_groups[0].members) == 5
    assert captured["kwargs"] == {"backend": "ray", "threshold": 0, "weight_sync": True, "gpu_node_domains": None}
    assert captured["loop"] is True


@pytest.mark.L0
@pytest.mark.CPU
def test_nft_dataloader_preflight_instantiates_and_prepares_one_t2v_row() -> None:
    config = SimpleNamespace(dataloader_train="lazy-loader")
    instantiated: list[object] = []

    class _ClosableLoader(_Loader):
        closed = False

        def close(self) -> None:
            self.closed = True

    loader = _ClosableLoader()

    def instantiate(value: object) -> _ClosableLoader:
        instantiated.append(value)
        return loader

    validate_nft_dataloader(config, instantiate_fn=instantiate)
    assert instantiated == ["lazy-loader"]
    assert loader.closed

    class _PackedLoader(_Loader):
        closed = False

        def __iter__(self) -> Iterator[dict[str, object]]:
            yield {
                "video": [torch.zeros(1, 2), torch.zeros(1, 2)],
                "ai_caption": ["prompt-a", "prompt-b"],
            }

        def close(self) -> None:
            self.closed = True

    packed_loader = _PackedLoader()
    with pytest.raises(ValueError, match="exactly one video source row"):
        validate_nft_dataloader(config, instantiate_fn=lambda _value: packed_loader)
    assert packed_loader.closed

    with pytest.raises(RuntimeError, match="produced no T2V sample"):
        validate_nft_dataloader(config, instantiate_fn=lambda _value: iter(()))

    class _T2ILoader:
        def __init__(self, *, batched_image_size: bool) -> None:
            self.batched_image_size = batched_image_size
            self.closed = False

        def __iter__(self) -> Iterator[dict[str, object]]:
            image_size = torch.tensor([4.0, 4.0, 4.0, 4.0])  # [4]
            yield {
                "images": [[torch.zeros((3, 4, 4), dtype=torch.uint8)]],  # one sample with one image [C,H,W]
                "ai_caption": "prompt-a",
                "image_size": [image_size] if self.batched_image_size else image_size,
                "conditioning_fps": torch.tensor([24.0], dtype=torch.bfloat16),  # [B=1]
            }

        def close(self) -> None:
            self.closed = True

    image_loader = _T2ILoader(batched_image_size=True)
    validate_nft_dataloader(config, modality="image", instantiate_fn=lambda _value: image_loader)
    assert image_loader.closed

    malformed_image_loader = _T2ILoader(batched_image_size=False)
    with pytest.raises(ValueError, match=r"leading singleton batch axis.*\(4,\)"):
        validate_nft_dataloader(config, modality="image", instantiate_fn=lambda _value: malformed_image_loader)
    assert malformed_image_loader.closed


@pytest.mark.L0
@pytest.mark.CPU
def test_wfm_restore_rejects_trainer_source_boundary_mismatch() -> None:
    source = WFMDataEnv(_Loader())
    trainer = SimpleNamespace(iteration=lambda: 3)
    with pytest.raises(RuntimeError, match="trainer_step=3, source_step=2"):
        restore_wfm_source(
            source,
            trainer,
            {
                "version": 1,
                "resume_kind": "optimizer_window_rows",
                "committed_rows": 2,
                "optimizer_step": 2,
                "grad_accum": 1,
                "source_fingerprint": "",
            },
        )


def _restore_failure_worker(rank: int, world_size: int, rendezvous_path: str) -> None:
    import torch.distributed as dist

    init_gloo_process_group(rank, world_size, rendezvous_path)
    try:
        source = WFMDataEnv(_Loader())
        trainer = SimpleNamespace(iteration=lambda: 3)
        committed_rows = 2 if rank == 1 else 3
        state = {
            "version": 1,
            "resume_kind": "optimizer_window_rows",
            "committed_rows": committed_rows,
            "optimizer_step": committed_rows,
            "grad_accum": 1,
            "source_fingerprint": "",
        }
        with pytest.raises(RuntimeError, match="online NFT source restore failed on one or more ranks"):
            restore_wfm_source(source, trainer, state)
    finally:
        dist.destroy_process_group()


@pytest.mark.L0
@pytest.mark.CPU
def test_rank_local_source_restore_failure_converges(tmp_path: Path) -> None:
    import torch.multiprocessing as mp

    world_size = 2
    rendezvous_path = str(tmp_path / "source-restore-failure-rendezvous")
    mp.spawn(_restore_failure_worker, args=(world_size, rendezvous_path), nprocs=world_size, join=True)


class _FaultSource:
    grad_accum = 1

    def __init__(self, *, fail_stage: bool = False) -> None:
        self.staged = False
        self.committed_rows = 0
        self.fail_stage = fail_stage

    def get_samples(self, _count: int) -> list[ObsBundle]:
        conditioning = WFMConditioningInput(
            id="source",
            template_identity="test-template",
            model_batch={"x": torch.zeros(1, 2)},
            prompt="prompt",
        )
        return [
            ObsBundle(
                representations={WFM_CONDITIONING_KEY: conditioning},
                modality_tags=("text", "video"),
                id=conditioning.id,
            )
        ]

    def stage(self) -> None:
        if self.fail_stage:
            raise ValueError("rank-local source stage failure")
        self.staged = True

    def commit_window(self, expected_rows: int) -> None:
        if expected_rows != 1 or not self.staged:
            raise RuntimeError("invalid fake source commit")
        self.committed_rows += 1
        self.staged = False

    def abort_window(self) -> None:
        self.staged = False

    def state_dict(self) -> dict[str, int | str]:
        return {
            "version": 1,
            "resume_kind": "optimizer_window_rows",
            "committed_rows": self.committed_rows,
            "optimizer_step": self.committed_rows,
            "grad_accum": 1,
            "source_fingerprint": "fake-source-v1",
        }


class _FaultPolicy:
    def iteration(self) -> int:
        return 0

    def train(self, _batch: list[Trajectory]) -> dict[str, str]:
        raise AssertionError("a failed distributed phase must not reach the trainer")

    def abort_accumulation(self) -> None:
        return None


class _AppliedPolicy:
    def __init__(self) -> None:
        self._iteration = 0

    def iteration(self) -> int:
        return self._iteration

    def train(self, _batch: list[Trajectory]) -> dict[str, float | str]:
        self._iteration += 1
        return {
            "optim/update_outcome": "applied",
            "train/optimality_mean": 0.5,
            "train/optimality_second_moment": 0.25,
        }

    def abort_accumulation(self) -> None:
        return None

    def push_weights(self, _target: object) -> None:
        return None

    def shared_weight_identity(self) -> None:
        return None


class _DistributedMetricPolicy:
    def __init__(self, *, omit_second_moment: bool = False) -> None:
        self.omit_second_moment = omit_second_moment
        self._iteration = 0

    def iteration(self) -> int:
        return self._iteration

    def train(self, _batch: list[Trajectory]) -> dict[str, float | str]:
        import torch.distributed as dist

        optimality = 0.25 + 0.5 * dist.get_rank()
        moments = torch.tensor([optimality, optimality * optimality], dtype=torch.float32)  # [E[r],E[r^2]]
        dist.all_reduce(moments, op=dist.ReduceOp.AVG)
        metrics: dict[str, float | str] = {
            "optim/update_outcome": "applied",
            "train/optimality_mean": float(moments[0].item()),
            "train/optimality_second_moment": float(moments[1].item()),
        }
        if self.omit_second_moment:
            del metrics["train/optimality_second_moment"]
        self._iteration += 1
        return metrics

    def abort_accumulation(self) -> None:
        return None

    def push_weights(self, _target: object) -> None:
        return None

    def shared_weight_identity(self) -> None:
        return None


class _FaultRollout:
    def __init__(self, fail: bool) -> None:
        self.fail = fail
        self.weight_version = 0

    def generate(self, conditioning: list[ObsBundle], **_kwargs: object) -> list[Trajectory]:
        if self.fail:
            raise ValueError("rank-local rollout failure")
        source = conditioning[0]
        return [
            Trajectory(
                id="sample",
                group_id=source.id,
                transitions=[Transition(obs=ObsBundle(id=source.id), action=torch.zeros(1, 2))],
            )
        ]

    def prepare_weight_version(self, version: int, producer_identity: str | None) -> None:
        del producer_identity
        if version < self.weight_version:
            raise RuntimeError("weight version regressed")

    def commit_weight_version(self, version: int) -> None:
        self.weight_version = version


class _PassReward:
    def __init__(self) -> None:
        self._reported = False

    def score_with_provider_report(self, trajectories: list[Trajectory]) -> list[Trajectory]:
        self._reported = True
        return trajectories

    def provider_report(self) -> ProviderMetricReport:
        if not self._reported:
            raise RuntimeError("fake reward report requested before scoring")
        self._reported = False
        return ProviderMetricReport({}, ())

    def annotation_metrics(self) -> dict[str, int | float]:
        return {
            "rollout/reward_group_mean": 0.5,
            "rollout/reward_group_spread": 1.0,
            "rollout/reward_nonzero_fraction": 1.0,
            "rollout/reward_group_size": 2,
            "rollout/reward_group_informative": 1,
            "rollout/reward_group_all_zero": 0,
            "rollout/advantage_group_std": 1.0,
            "rollout/advantage_abs_max": 1.0,
        }


class _FaultReplay:
    def __init__(self, fail: bool) -> None:
        self.fail = fail
        self.items: list[Trajectory] = []

    def put_samples(self, trajectories: list[Trajectory]) -> None:
        if self.fail:
            raise ValueError("rank-local replay failure")
        self.items.extend(trajectories)

    def get_exact_samples(self, count: int | None = None) -> list[Trajectory]:
        count = 1 if count is None else count
        return self.items[:count]


class _CommitFaultSource(_FaultSource):
    def __init__(self, fail: bool) -> None:
        super().__init__()
        self.fail = fail

    def commit_window(self, expected_rows: int) -> None:
        if self.fail:
            raise ValueError("rank-local source commit failure")
        super().commit_window(expected_rows)


class _TrackingSource(_FaultSource):
    def __init__(self) -> None:
        super().__init__()
        self.committed = False

    def commit_window(self, expected_rows: int) -> None:
        super().commit_window(expected_rows)
        self.committed = True


def _phase_failure_worker(rank: int, world_size: int, rendezvous_path: str, phase: str) -> None:
    import torch.distributed as dist

    init_gloo_process_group(rank, world_size, rendezvous_path)
    try:
        units = {
            "data": _FaultSource(fail_stage=phase == "stage" and rank == 1),
            "rollout": _FaultRollout(fail=phase == "rollout" and rank == 1),
            "rm": _PassReward(),
            "replay": _FaultReplay(fail=phase == "replay" and rank == 1),
            "policy": _AppliedPolicy() if phase == "stage" else _FaultPolicy(),
        }
        operation = "source-window stage" if phase == "stage" else phase
        with pytest.raises(RuntimeError, match=f"online NFT {operation}.*failed on one or more ranks"):
            online_nft_step(units, NFTConfig(max_steps=1, group_size=2), step=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("phase", ["rollout", "replay assembly", "source-window stage"])
def test_rank_local_online_phase_failure_converges(tmp_path: Path, phase: str) -> None:
    import torch.multiprocessing as mp

    worker_phase = {"replay assembly": "replay", "source-window stage": "stage"}.get(phase, phase)
    rendezvous_path = str(tmp_path / f"{worker_phase}-failure-rendezvous")
    mp.spawn(_phase_failure_worker, args=(2, rendezvous_path, worker_phase), nprocs=2, join=True)


def _commit_failure_worker(rank: int, world_size: int, rendezvous_path: str) -> None:
    import torch.distributed as dist

    init_gloo_process_group(rank, world_size, rendezvous_path)
    try:
        units = {
            "data": _CommitFaultSource(fail=rank == 1),
            "rollout": _FaultRollout(fail=False),
            "rm": _PassReward(),
            "replay": _FaultReplay(fail=False),
            "policy": _AppliedPolicy(),
        }
        with pytest.raises(RuntimeError, match="online NFT source-window commit.*failed on one or more ranks"):
            online_nft_step(units, NFTConfig(max_steps=1, group_size=2), step=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.L0
@pytest.mark.CPU
def test_rank_local_source_commit_failure_is_fatal_on_every_rank(tmp_path: Path) -> None:
    import torch.multiprocessing as mp

    rendezvous_path = str(tmp_path / "source-commit-failure-rendezvous")
    mp.spawn(_commit_failure_worker, args=(2, rendezvous_path), nprocs=2, join=True)


def _optimality_telemetry_worker(rank: int, world_size: int, rendezvous_path: str) -> None:
    import torch.distributed as dist

    init_gloo_process_group(rank, world_size, rendezvous_path)
    try:
        source = _TrackingSource()
        units = {
            "data": source,
            "rollout": _FaultRollout(fail=False),
            "rm": _PassReward(),
            "replay": _FaultReplay(fail=False),
            "policy": _DistributedMetricPolicy(),
        }
        metrics = online_nft_step(units, NFTConfig(max_steps=1, group_size=2), step=0)
        assert metrics["train/optimality_std"] == pytest.approx(0.25)
        assert source.committed

        malformed_source = _TrackingSource()
        malformed_units = {
            "data": malformed_source,
            "rollout": _FaultRollout(fail=False),
            "rm": _PassReward(),
            "replay": _FaultReplay(fail=False),
            "policy": _DistributedMetricPolicy(omit_second_moment=rank == 1),
        }
        with pytest.raises(RuntimeError, match="online NFT trainer returned inconsistent metric frames"):
            online_nft_step(malformed_units, NFTConfig(max_steps=1, group_size=2), step=0)
        assert not malformed_source.committed
    finally:
        dist.destroy_process_group()


@pytest.mark.L0
@pytest.mark.CPU
def test_optimality_dispersion_uses_global_moments_and_converges_failures(tmp_path: Path) -> None:
    import torch.multiprocessing as mp

    rendezvous_path = str(tmp_path / "optimality-telemetry-rendezvous")
    mp.spawn(_optimality_telemetry_worker, args=(2, rendezvous_path), nprocs=2, join=True)
