# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""@L0 @CPU: `entrypoints/opd.py` — the OPD topology, and the teacher hook in the reused GRPO loop.

Two things are worth pinning here, and neither needs a GPU.

**The topology.** `bind_check` is the GPU-less pre-flight, so the OPD manifest has to pass it before
anything is placed — an unregistered capability or a mis-declared edge is the failure that would
otherwise surface after an engine and a trainer have loaded.

**The hook.** `entrypoints/opd.py` reuses `grpo.loop` rather than forking it, which is only safe if
the teacher is called in the right place and a teacher-less run is unchanged. Both are asserted
against the real `grpo.loop` with inert stand-in roles — the same shape the GRPO loop tests use.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from cosmos_framework.posttrain.config.model import DataPackerConfig, HFModelConfig
from cosmos_framework.posttrain.config.rl import (
    DataPoolConfig,
    DatasetConfig,
    GRPOValidationConfig,
    OPDConfig,
    RolloutConfig,
)
from cosmos_framework.posttrain.control.bind import bind_check
from cosmos_framework.posttrain.control.checkpointing import CheckpointLifecycle
from cosmos_framework.posttrain.control.manifest import Plane
from cosmos_framework.posttrain.entrypoints import opd


@pytest.mark.L0
@pytest.mark.CPU
def test_build_teacher_uses_student_factory_and_threads_teacher_model_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """The teacher actor uses the student codec while retaining separate teacher-model config."""
    data_packer = DataPackerConfig(factory="tests.custom.build_packer")
    selected = object()
    calls: list[tuple[str, str, bool]] = []
    captured: dict[str, Any] = {}
    monkeypatch.setattr("cosmos_framework.utils.lazy_config.instantiate", lambda value: value)
    monkeypatch.setattr("cosmos_framework.posttrain.config.build.effective_model_id", lambda _config: "student/model")
    monkeypatch.setattr(
        "cosmos_framework.posttrain.data.packer.init_data_packer_from_factory",
        lambda name, factory, **kwargs: (
            calls.append((name, factory, bool(kwargs["trust_remote_code"]))),
            selected,
        )[1],
    )
    monkeypatch.setattr(
        "cosmos_framework.posttrain.config.build.build_vllm_teacher",
        lambda teacher, packer, packer_config: captured.update(
            teacher=teacher,
            packer=packer,
            packer_config=packer_config,
        )
        or object(),
    )
    monkeypatch.setattr(
        "cosmos_framework.posttrain.utils.dist.run_rank_converged",
        lambda operation, _context: operation(),
    )
    config = SimpleNamespace(
        model_backend="hf",
        model_hf=HFModelConfig(name_or_path="student/model", trust_remote_code=True),
        rl=OPDConfig(),
        data_packer=data_packer,
    )

    opd.build_teacher(config)

    assert calls == [("student/model", data_packer.factory, True)]
    assert captured["packer"] is selected
    assert captured["packer_config"] is data_packer


@pytest.mark.L0
@pytest.mark.CPU
def test_opd_objective_consumes_supplied_policy_packer(monkeypatch: pytest.MonkeyPatch) -> None:
    """The shared trainer must not let OPD rebuild and bypass its selected policy packer."""
    selected = object()
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        "cosmos_framework.posttrain.config.build.build_opd_objective",
        lambda rl, packer, pdims=None: captured.update(rl=rl, packer=packer, pdims=pdims) or object(),
    )
    monkeypatch.setattr("cosmos_framework.posttrain.config.build.resolve_pdims", lambda: "pdims")
    monkeypatch.setattr(
        "cosmos_framework.posttrain.data.packer.init_data_packer",
        lambda *_args, **_kwargs: pytest.fail("OPD objective must consume the shared policy packer"),
    )
    rl = OPDConfig()

    opd.build_objective(rl, SimpleNamespace(name_or_path="policy/model"), selected)

    assert captured == {"rl": rl, "packer": selected, "pdims": "pdims"}


@pytest.mark.L0
@pytest.mark.CPU
def test_opd_validate_preflights_custom_data_packer_factories(monkeypatch: pytest.MonkeyPatch) -> None:
    """OPD validation must reject packer import errors before role placement."""
    data_packer = DataPackerConfig(factory="missing.module.build_packer")
    seen: list[DataPackerConfig] = []
    monkeypatch.setattr(
        "cosmos_framework.posttrain.data.packer.validate_data_packer_factories",
        lambda config: seen.append(config),
    )
    monkeypatch.setattr(opd, "render_manifest_validation", lambda _manifest: ("bind_check: ok", True))
    monkeypatch.setattr(opd, "manifest", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        "cosmos_framework.posttrain.config.build.validate_opd_runtime",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "cosmos_framework.posttrain.control.validation.rollout.validate_rollout_reward_sources",
        lambda *_args, **_kwargs: None,
    )

    with pytest.raises(SystemExit) as error:
        opd._validate_and_exit(
            OPDConfig(),
            RolloutConfig(),
            SimpleNamespace(backend="local", gpu_placement="colocated"),
            data_packer_config=data_packer,
        )

    assert error.value.code == 0
    assert seen == [data_packer]


@pytest.mark.L0
@pytest.mark.CPU
def test_opd_run_preflights_custom_data_packer_factories(monkeypatch: pytest.MonkeyPatch) -> None:
    """A normal OPD launch must reject packer import errors before model or placement setup."""
    data_packer = DataPackerConfig(factory="missing.module.build_packer")
    seen: list[DataPackerConfig] = []

    def _reject(config: DataPackerConfig) -> None:
        seen.append(config)
        raise ValueError("invalid custom packer")

    monkeypatch.setattr(
        "cosmos_framework.posttrain.data.packer.validate_data_packer_factories",
        _reject,
    )

    with pytest.raises(ValueError, match="invalid custom packer"):
        opd.run(SimpleNamespace(rl=OPDConfig(), data_packer=data_packer), "unused.py", [], "/tmp/out")

    assert seen == [data_packer]


@pytest.mark.L0
@pytest.mark.CPU
def test_manifest_binds_clean_and_declares_the_teacher_data_topology() -> None:
    """The manifest is internally valid and keeps the frozen teacher off the weight plane."""
    manifest = opd.manifest(OPDConfig(), weight_relationship="cross_device")
    report = bind_check(manifest)

    assert not report.errors, report.format()
    assert not report.obligations, report.format()
    assert [role.name for role in manifest.roles] == [
        "data",
        "rollout",
        "rm",
        "teacher",
        "replay",
        "policy",
    ]
    edges = {(edge.src, edge.dst): edge for edge in manifest.edges}

    assert edges[("teacher", "replay")].plane is Plane.DATA
    assert edges[("rm", "teacher")].plane is Plane.DATA
    assert ("policy", "teacher") not in edges
    assert edges[("policy", "rollout")].plane is Plane.WEIGHT  # the student's engine still syncs
    assert edges[("policy", "rollout")].relationship == "cross_device"
    assert ("rm", "replay") not in edges  # the teacher sits BETWEEN them now


@pytest.mark.L0
@pytest.mark.CPU
def test_teacher_worker_group_has_one_independently_ranked_replica() -> None:
    """Teacher TP is independent of rollout/policy shape and never creates teacher replicas."""
    assert OPDConfig().teacher.tensor_parallel_size == 1
    wgs = opd.declare_wgs(ranks=4, replicas=2, teacher_ranks=8)

    assert wgs["teacher"].ranks == 8
    assert wgs["teacher"].needs_bulk is True
    assert wgs["teacher"].replicas == 1
    assert wgs["rollout"].ranks == 4 and wgs["policy"].ranks == 4  # only these span the mesh
    assert wgs["rollout"].replicas == 2
    assert wgs["policy"].replicas == 1

    asymmetric = opd.declare_wgs(
        ranks=1,
        shapes={"policy": (1, 4), "rollout": (4, 1)},
        policy_parallelism=(2, 2),
        teacher_ranks=8,
    )
    assert (asymmetric["policy"].replicas, asymmetric["policy"].ranks) == (1, 4)
    assert asymmetric["policy"].model_parallelism == (2, 2)
    assert (asymmetric["rollout"].replicas, asymmetric["rollout"].ranks) == (4, 1)
    assert (asymmetric["teacher"].replicas, asymmetric["teacher"].ranks) == (1, 8)

    text_only = opd.declare_wgs(ranks=2, teacher_ranks=2, teacher_needs_bulk=False)
    assert text_only["teacher"].needs_bulk is False
    multimodal = opd.declare_wgs(
        ranks=2,
        teacher_ranks=2,
        teacher_needs_bulk=opd._teacher_needs_bulk("cross_token_multimodal"),
    )
    assert multimodal["teacher"].needs_bulk is True
    assert opd._teacher_needs_bulk("cross_token_text") is False
    with pytest.raises(ValueError, match="teacher.input_mode must be"):
        opd._teacher_needs_bulk("cross_token_video")


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    (
        "rl_kwargs",
        "rollout_ranks",
        "gpu_placement",
        "shapes",
        "policy_shape",
        "rollout_shape",
        "expected_device_groups",
        "expected_weight_sync",
    ),
    [
        (
            {"data_pool": DataPoolConfig(data_plane="ray_nixl_pool")},
            2,
            "colocated",
            {},
            (1, 2),
            (1, 2),
            [["experience", "policy"], ["teacher"]],
            {
                "weight_sync_transport": None,
                "weight_sync_bucket_bytes": None,
                "weight_sync_probe_bytes": 256 * 1024 * 1024,
                "weight_sync_min_gbps": 20.0,
            },
        ),
        (
            {
                "weight_sync_transport": "nixl",
                "weight_sync_bucket_bytes": 64 * 1024 * 1024,
                "weight_sync_probe_bytes": 8 * 1024 * 1024,
                "weight_sync_min_gbps": 7.5,
            },
            1,
            "disaggregated",
            {"policy": (1, 4), "rollout": (1, 1)},
            (1, 4),
            (1, 1),
            [["experience"], ["teacher"], ["policy"]],
            {
                "weight_sync_transport": "nixl",
                "weight_sync_bucket_bytes": 64 * 1024 * 1024,
                "weight_sync_probe_bytes": 8 * 1024 * 1024,
                "weight_sync_min_gbps": 7.5,
            },
        ),
    ],
)
def test_run_places_ranked_teacher_and_threads_weight_sync_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    rl_kwargs: dict[str, Any],
    rollout_ranks: int,
    gpu_placement: str,
    shapes: dict[str, tuple[int, int]],
    policy_shape: tuple[int, int],
    rollout_shape: tuple[int, int],
    expected_device_groups: list[list[str]],
    expected_weight_sync: dict[str, Any],
) -> None:
    """OPD threads independent data- and weight-plane settings into its Ray actors."""
    captured: dict[str, Any] = {}
    finalizations: list[tuple[dict[str, Any], Any]] = []

    def _capture_bind(device_groups: list[Any], **kwargs: Any) -> None:
        captured["device_groups"] = [
            {
                "device": device_group.device,
                "actor_groups": [
                    [worker.name for worker in actor_group.members] for actor_group in device_group.actor_groups
                ],
            }
            for device_group in device_groups
        ]
        captured["kwargs"] = kwargs

    monkeypatch.setattr(
        "cosmos_framework.posttrain.control.validation.rollout.validate_rollout_reward_sources",
        lambda _rl, _catalog: None,
    )
    monkeypatch.setattr("cosmos_framework.posttrain.control.placement.bind_device_groups", _capture_bind)
    monkeypatch.setattr(
        "cosmos_framework.posttrain.utils.entry.pin_run_attempt",
        lambda _config, opts, out_dir: SimpleNamespace(opts=opts, out_dir=out_dir),
    )
    monkeypatch.setattr(opd, "dump_resolved", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(opd, "loop", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        CheckpointLifecycle,
        "finalize_units",
        lambda wgs, teardown: finalizations.append((wgs, teardown)),
    )
    rollout = RolloutConfig(tensor_parallel_size=rollout_ranks)
    config = SimpleNamespace(
        model_backend="hf",
        model_hf=HFModelConfig(name_or_path=rollout.name_or_path),
        rl=OPDConfig(**rl_kwargs),
        rollout=rollout,
        grid=SimpleNamespace(
            backend="ray",
            gpu_placement=gpu_placement,
            shapes=shapes,
            num_replicas=1,
            policy_shape=lambda: policy_shape,
            rollout_shape=lambda: rollout_shape,
        ),
    )

    opd.run(config, "unused.py", [], str(tmp_path))

    actor_group_labels = {
        ("data", "rollout", "rm", "replay"): "experience",
        ("teacher",): "teacher",
        ("policy",): "policy",
    }
    assert [
        [actor_group_labels[tuple(members)] for members in device_group["actor_groups"]]
        for device_group in captured["device_groups"]
    ] == expected_device_groups
    assert all(device_group["device"] == "gpu" for device_group in captured["device_groups"])
    assert "num_gpus_each" not in captured["kwargs"]
    for key, expected in expected_weight_sync.items():
        assert captured["kwargs"][key] == expected
    assert len(finalizations) == 1
    finalized_wgs, teardown = finalizations[0]
    assert "policy" in finalized_wgs
    assert teardown is None  # Ray actor process groups are finalized inside their policy cells.
    assert captured["kwargs"]["data_pool_config"] == opd._data_pool_config(config.rl)


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    "gpu_placement,rl_kwargs,match",
    [
        ("colocated", {"weight_sync_transport": "nixl"}, "cannot serve resolved relationship='same_gpu'"),
        ("disaggregated", {}, "requires rl.weight_sync_transport='nixl'"),
        ("colocated", {"weight_sync_r2r_broadcast": True}, "has no replica fan-out"),
    ],
)
def test_run_rejects_sync_option_that_disagrees_with_placement(
    monkeypatch: pytest.MonkeyPatch,
    gpu_placement: str,
    rl_kwargs: dict[str, Any],
    match: str,
) -> None:
    """Reject sync options that cannot serve the resolved OPD placement."""
    placement_called = False

    def _placement(*_args: Any, **_kwargs: Any) -> None:
        nonlocal placement_called
        placement_called = True

    monkeypatch.setattr("cosmos_framework.posttrain.control.placement.bind_device_groups", _placement)
    rollout = RolloutConfig(tensor_parallel_size=1)
    config = SimpleNamespace(
        model_backend="hf",
        model_hf=HFModelConfig(name_or_path=rollout.name_or_path),
        rl=OPDConfig(**rl_kwargs),
        rollout=rollout,
        grid=SimpleNamespace(
            backend="ray",
            gpu_placement=gpu_placement,
            shapes={},
            num_replicas=1,
            policy_shape=lambda: (1, 1),
            rollout_shape=lambda: (1, 1),
        ),
    )

    with pytest.raises(ValueError, match=match):
        opd.run(config, "unused.py", [], "/tmp/out")
    assert placement_called is False


@pytest.mark.L0
@pytest.mark.CPU
def test_local_backend_rejects_ranked_teacher_before_process_group_or_placement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ranked vLLM uses Ray's per-replica rendezvous; local remains the TP1 test path."""
    initialized = False
    placement_called = False

    def _init() -> None:
        nonlocal initialized
        initialized = True

    def _placement(*_args: Any, **_kwargs: Any) -> None:
        nonlocal placement_called
        placement_called = True

    monkeypatch.setattr(
        "cosmos_framework.posttrain.control.validation.rollout.validate_rollout_reward_sources",
        lambda _rl, _catalog: None,
    )
    monkeypatch.setattr("cosmos_framework.posttrain.utils.dist.init_local_process_group", _init)
    monkeypatch.setattr("cosmos_framework.posttrain.control.placement.bind_device_groups", _placement)
    rl = OPDConfig()
    rl.teacher.tensor_parallel_size = 2
    rollout = RolloutConfig(tensor_parallel_size=1)
    config = SimpleNamespace(
        model_backend="hf",
        model_hf=HFModelConfig(name_or_path=rollout.name_or_path),
        rl=rl,
        rollout=rollout,
        grid=SimpleNamespace(backend="local", shapes={}, num_replicas=1),
    )

    with pytest.raises(ValueError, match="Ranked vLLM groups require Ray"):
        opd.run(config, "unused.py", [], "/tmp/out")
    assert initialized is False
    assert placement_called is False


@pytest.mark.L0
@pytest.mark.CPU
def test_setup_wires_a_validation_specific_reward_without_a_teacher(monkeypatch: pytest.MonkeyPatch) -> None:
    """Held-out rollout/reward evaluation may select another reward, but never builds a teacher."""
    configured = SimpleNamespace(
        rl=OPDConfig(
            reward="arithmetic",
            validation=GRPOValidationConfig(
                enabled=True,
                data=DatasetConfig(path="held-out.jsonl", prompt_column="prompt"),
                reward="nonempty",
            ),
        ),
        model_hf=SimpleNamespace(name_or_path="unused"),
        rollout=SimpleNamespace(name_or_path="unused"),
    )
    config_module = ModuleType("cosmos_framework.utils.config")
    lazy_config_module = ModuleType("cosmos_framework.utils.lazy_config")
    config_module.load_config = lambda *_args, **_kwargs: configured
    lazy_config_module.instantiate = lambda value: value
    monkeypatch.setitem(sys.modules, "cosmos_framework.utils.config", config_module)
    monkeypatch.setitem(sys.modules, "cosmos_framework.utils.lazy_config", lazy_config_module)

    units = opd.make_setup("unused.py", [], ["rm"])()

    assert "teacher" not in units
    assert units["rm"].providers[0].provider_name == "arithmetic"
    assert units["rm"].validation_providers[0].provider_name == "nonempty"


@pytest.mark.L0
@pytest.mark.CPU
def test_setup_selects_the_opd_entry_objective(monkeypatch: pytest.MonkeyPatch) -> None:
    """OPD keeps its existing objective while using the shared trainer builder seam."""
    configured = SimpleNamespace(rl=OPDConfig())
    config_module = ModuleType("cosmos_framework.utils.config")
    lazy_config_module = ModuleType("cosmos_framework.utils.lazy_config")
    config_module.load_config = lambda *_args, **_kwargs: configured
    lazy_config_module.instantiate = lambda value: value
    monkeypatch.setitem(sys.modules, "cosmos_framework.utils.config", config_module)
    monkeypatch.setitem(sys.modules, "cosmos_framework.utils.lazy_config", lazy_config_module)

    captured: dict[str, Any] = {}

    def _build_trainer(config: Any, rl: Any, *, build_objective_fn: Any) -> object:
        captured.update(config=config, rl=rl, build_objective_fn=build_objective_fn)
        return object()

    monkeypatch.setattr(opd, "build_trainer", _build_trainer)
    units = opd.make_setup("unused.py", [], ["policy"])()

    assert units["policy"] is not None
    assert captured == {
        "config": configured,
        "rl": configured.rl,
        "build_objective_fn": opd.build_objective,
    }


# --------------------------------------------------------------------------- the reused loop


def _config(**kw: Any) -> Any:
    """The loop's config, through the shared helper GRPO's own loop tests use.

    Same scaffolding on purpose: OPD reuses `grpo.loop`, so a divergent fake config here would let
    the two drift and this file would stop testing the loop that actually runs."""
    from cosmos_framework.posttrain.tests.helpers.entry_loop import loop_config

    cfg = loop_config(max_steps=kw.get("max_steps", 1), prompts_per_step=1, group_size=2, save_every=0, seed=0)
    cfg.grid.num_replicas = 1
    cfg.rollout = SimpleNamespace()
    return cfg


def _units(policy: Any, calls: list[str], teacher: Any | None = None) -> dict[str, Any]:
    """Inert experience roles; `calls` records what the loop invoked, in order."""

    from cosmos_framework.posttrain.utils.batch_stats import RewardBatchStats

    class _Rollout:
        def generate(self, prompts: list[Any], group: int) -> list[Any]:
            del prompts, group
            calls.append("generate")
            return []

    # The reward role reports its own batch statistics now (main's reward-boundary reporting), so the
    # stand-in has to answer both halves: `score_with_report` returns the scored handle the teacher
    # and replay route on, `batch_stats` the small report the loop resolves as its step barrier.
    # Non-zero `trajectories`: merging rejects an empty report (`utils/batch_stats.py:100`), which is
    # the right guard and makes an all-zero stand-in invalid rather than merely uninteresting.
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

    policy.push_weights = lambda rollout: calls.append("sync")
    units = {
        "data": SimpleNamespace(get_samples=lambda k: []),
        "rollout": _Rollout(),
        "rm": SimpleNamespace(
            score_with_report=lambda t: (calls.append("score"), t)[1],
            batch_stats=lambda: reward_report,
        ),
        "replay": SimpleNamespace(put_samples=lambda s: calls.append("put"), get_samples=lambda k: []),
        "policy": policy,
    }
    if teacher is not None:
        units["teacher"] = teacher
    return units


class _Policy:
    def __init__(self) -> None:
        self._it, self.grad_accum = 0, 1

    def iteration(self) -> int:
        return self._it

    def train(
        self,
        batch: list[Any],
        *,
        mini: int = 0,
        micro: int = 0,
        n_mini: int = 1,
        n_micro: int = 1,
    ) -> dict[str, float]:
        # The driver passes INDICES ONLY; in production the dispatch layer consumes them and the
        # body sees `train(batch)`. This fake stands in for the whole WorkerGroup, so it takes them.
        del batch, mini, micro, n_mini, n_micro
        self._it += 1
        return {"train/loss_avg": 0.5}

    def save(self, path: str, publish: bool = True, update_latest: bool = True) -> None:
        del path, publish, update_latest


@pytest.mark.L0
@pytest.mark.CPU
def test_the_loop_calls_the_teacher_between_reward_and_replay(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Placement is the whole correctness argument for the hook.

    AFTER the reward stage, because the teacher scores completions the reward has already annotated
    and the group must still be intact. BEFORE the replay put, because the signal has to ride the
    trajectory through dispatch — a teacher called on the trainer side would see whatever slice
    dispatch happened to deliver, not a group."""
    from cosmos_framework.posttrain.entrypoints import grpo
    from cosmos_framework.posttrain.tests.helpers.entry_loop import mock_loop_env
    from cosmos_framework.posttrain.tests.test_entrypoints_grpo import _validation_config, _validation_units

    mock_loop_env(monkeypatch)
    calls: list[str] = []
    validation_events: list[tuple[str, int]] = []
    teacher = SimpleNamespace(
        score=lambda t: (calls.append("teacher"), t)[1],
        batch_stats=lambda: (
            calls.append("teacher_stats"),
            {"train/opd_teacher_scored_sequences": 2},
        )[1],
    )
    units = _validation_units(_Policy(), validation_events)
    training_score = units["rm"].score_with_report
    replay_put = units["replay"].put_samples
    units["rm"].score_with_report = lambda trajectories: (calls.append("score"), training_score(trajectories))[1]
    units["replay"].put_samples = lambda trajectories: (calls.append("put"), replay_put(trajectories))[1]
    units["teacher"] = teacher

    summary = grpo.loop(
        units,
        _validation_config(max_steps=1, before_training=True, every_steps=0),
        str(tmp_path),
    )

    assert summary["validation_steps"] == [0, 1]
    assert calls.count("teacher") == 1, "teacher must run once for training, never for either held-out pass"
    assert calls.count("teacher_stats") == 1
    assert calls.index("score") < calls.index("teacher") < calls.index("teacher_stats") < calls.index("put"), calls
    assert validation_events == [
        ("validation", 0),
        ("validation", 0),
        ("validation", 1),
        ("validation", 1),
    ]


@pytest.mark.L0
@pytest.mark.CPU
def test_a_teacherless_run_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """The hook must be inert when absent — every existing GRPO run goes through this same code."""
    from cosmos_framework.posttrain.entrypoints import grpo

    _mock_env(monkeypatch)
    calls: list[str] = []

    grpo.loop(_units(_Policy(), calls), _config(max_steps=1), "/tmp/out")

    assert "teacher" not in calls
    assert calls.index("score") < calls.index("put")


def _mock_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the loop's side effects: reporters, weight sync, termination watch."""
    from cosmos_framework.posttrain.control.loop import weight_sync as ws_mod

    monkeypatch.setattr(
        ws_mod,
        "weight_sync",
        lambda rollout, policy, **_kwargs: policy.push_weights(rollout),
        raising=False,
    )
    monkeypatch.setattr(
        "cosmos_framework.posttrain.utils.report.make_reporters",
        lambda *a, **k: [],
        raising=False,
    )
