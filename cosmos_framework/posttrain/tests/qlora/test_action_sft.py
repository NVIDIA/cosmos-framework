# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Native action SFT startup order, evaluation and checkpoint lifecycle on CPU."""

from __future__ import annotations

import copy
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from cosmos_framework.posttrain.algorithm.primitives.objective import OBJECTIVE_WEIGHT_SUM, Objective, Reduction
from cosmos_framework.posttrain.checkpoint.rng import collect_rng_state, restore_rng_state
from cosmos_framework.posttrain.entrypoints.action_common import ActionSource, evaluate, preserved_evaluation
from cosmos_framework.posttrain.roles.env.cursor import IndexedCursor
from cosmos_framework.posttrain.roles.trainer.llm import LLMTrainer
from cosmos_framework.posttrain.tests.qlora.action_helpers import assert_state_equal

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


@pytest.fixture
def runtime_startup(monkeypatch, tmp_path):
    """Run the native startup and real runtime validator with CUDA launch plumbing replaced."""
    from cosmos_framework.posttrain.control.checkpointing import CheckpointLifecycle
    from cosmos_framework.posttrain.entrypoints import action_common, action_sft
    from cosmos_framework.posttrain.model.quantization import family
    from cosmos_framework.posttrain.utils import determinism, entry, qlora_metrics
    from cosmos_framework.utils.config import load_config

    calls = []
    initialized = False

    class ReachedData(Exception):
        pass

    def initialize(backend):
        nonlocal initialized
        assert backend == "nccl" and calls == ["device"]
        initialized = True
        calls.append("process_group")

    def prepare(_config):
        calls.append("data")
        raise ReachedData

    def lookup(package):
        assert initialized and calls == ["device", "process_group"]
        calls.append(package)
        return "0.49.2"

    def forbidden_factory(*_args, **_kwargs):
        pytest.fail("no model or data factory may run before runtime admission")

    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: calls.append("device"))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda including_emulation=True: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (8, 0))
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda device: SimpleNamespace(name="A100", total_memory=80_000_000_000)
    )
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: initialized)
    monkeypatch.setattr(torch.distributed, "init_process_group", initialize)
    monkeypatch.setattr(torch.distributed, "destroy_process_group", lambda: calls.append("destroy_process_group"))
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(determinism, "enable_determinism", lambda **_kwargs: None)
    monkeypatch.setattr(entry, "out_dir", lambda _config: str(tmp_path / "run"))
    monkeypatch.setattr(entry, "dump_resolved", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        entry,
        "pin_run_attempt",
        lambda *_args: SimpleNamespace(out_dir=str(tmp_path / "run"), resume_checkpoint=None),
    )
    monkeypatch.setattr(CheckpointLifecycle, "from_config", lambda *_args, **_kwargs: SimpleNamespace(manager=None))
    monkeypatch.setattr(action_sft, "prepare_sources", prepare)
    monkeypatch.setattr(action_common, "load_factory", forbidden_factory)
    monkeypatch.setattr(qlora_metrics, "record_qlora_phase", lambda phase, **_kwargs: calls.append(phase))
    monkeypatch.setattr(family, "version", lookup)

    def run(recipe, mode=None):
        path = f"cosmos_framework/posttrain/configs/{recipe}.py"
        opts = ["--"] if mode is None else ["--", f"model.backbone_adaptation.mode={mode}"]
        config = load_config(path, opts)
        action_sft.run(config, config_path=path, opts=[])

    return SimpleNamespace(run=run, calls=calls, reached_data=ReachedData, family=family)


@pytest.mark.parametrize("recipe", ["sft_alpamayo_qlora", "gr00t_n1d7_libero_qlora"])
@pytest.mark.parametrize("mode", ["nf4_lora", "nf4_frozen"])
@pytest.mark.parametrize("installed", [None, "0.48.0"])
def test_native_startup_rejects_missing_or_wrong_nf4_dependency_before_inputs(
    runtime_startup, monkeypatch, recipe, mode, installed
):
    from importlib.metadata import PackageNotFoundError

    def lookup(package):
        assert package == "bitsandbytes"
        assert runtime_startup.calls == ["device", "process_group"]
        if installed is None:
            raise PackageNotFoundError(package)
        return installed

    monkeypatch.setattr(runtime_startup.family, "version", lookup)
    with pytest.raises(ValueError, match="bitsandbytes") as error:
        runtime_startup.run(recipe, mode)
    assert runtime_startup.calls == ["device", "process_group", "destroy_process_group"]
    if installed is None:
        assert "docker/posttrain/native_action/README.md" in str(error.value)
        assert isinstance(error.value.__cause__, PackageNotFoundError)


@pytest.mark.parametrize("recipe", ["sft_alpamayo_qlora", "gr00t_n1d7_libero_qlora"])
@pytest.mark.parametrize("mode", ["nf4_lora", "nf4_frozen", "bf16_lora", "bf16_frozen"])
@pytest.mark.parametrize("capability", [(7, 0), (8, 0)])
def test_native_startup_rejects_missing_bf16_hardware_before_inputs(
    runtime_startup, monkeypatch, recipe, mode, capability
):
    # (7, 0) is a V100: torch reports emulated BF16 there unless including_emulation=False.
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: capability)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda including_emulation=True: including_emulation)
    monkeypatch.setattr(runtime_startup.family, "version", lambda _package: pytest.fail("hardware must fail first"))
    with pytest.raises(ValueError, match="compute capability 8.0\\+ and BF16 kernels"):
        runtime_startup.run(recipe, mode)
    assert runtime_startup.calls == ["device", "process_group", "destroy_process_group"]


@pytest.mark.parametrize("recipe", ["sft_alpamayo_qlora", "gr00t_n1d7_libero_qlora"])
@pytest.mark.parametrize("mode", ["nf4_lora", "nf4_frozen", "bf16_lora", "bf16_frozen"])
def test_native_startup_checks_only_selected_runtime_before_inputs(runtime_startup, monkeypatch, recipe, mode):
    if mode.startswith("bf16_"):
        monkeypatch.setattr(runtime_startup.family, "version", lambda _package: pytest.fail("BF16 must not query bnb"))
    with pytest.raises(runtime_startup.reached_data):
        runtime_startup.run(recipe, mode)
    assert runtime_startup.calls == ["device", "process_group"] + (
        ["bitsandbytes"] if mode.startswith("nf4_") else []
    ) + ["data_preparation_begin", "data", "destroy_process_group"]


@pytest.mark.parametrize("recipe", ["sft_alpamayo_qlora", "gr00t_n1d7_libero_qlora"])
def test_resume_checks_the_saved_adapter_before_inputs_or_model(runtime_startup, monkeypatch, tmp_path, recipe):
    from cosmos_framework.posttrain.checkpoint import adapter
    from cosmos_framework.posttrain.utils import entry

    class AdapterChecked(Exception):
        pass

    seen = {}

    def read(path, *, expected_identity=None, require_resume=False, manager=None):
        seen.update(path=path, expected=expected_identity, require_resume=require_resume)
        raise AdapterChecked

    monkeypatch.setattr(
        entry,
        "pin_run_attempt",
        lambda *_args: SimpleNamespace(out_dir=str(tmp_path / "run"), resume_checkpoint=str(tmp_path / "ckpt")),
    )
    monkeypatch.setattr(adapter, "read_adapter_metadata", read)
    with pytest.raises(AdapterChecked):
        runtime_startup.run(recipe, "nf4_lora")
    assert seen == {
        "path": str(tmp_path / "ckpt"),
        "expected": {"adapter": {"mode": "nf4_lora", "rank": 32, "alpha": 64.0}},
        "require_resume": True,
    }
    assert "data" not in runtime_startup.calls


def test_invalid_explicit_backbone_mode_never_falls_back_to_dense(monkeypatch):
    from cosmos_framework.posttrain.entrypoints.action_sft import preflight_backbone_runtime
    from cosmos_framework.posttrain.model.quantization import family

    monkeypatch.setattr(
        family, "require_family_runtime", lambda *_a, **_k: pytest.fail("invalid mode must not select a runtime")
    )
    config = SimpleNamespace(model=SimpleNamespace(backbone_adaptation=SimpleNamespace(mode="nf4_typo")))
    with pytest.raises(ValueError, match="backbone mode"):
        preflight_backbone_runtime(config, torch.device("cuda", 0))


def test_runtime_preflight_does_not_relabel_unrelated_missing_packages(monkeypatch):
    from importlib.metadata import PackageNotFoundError

    from cosmos_framework.posttrain.entrypoints.action_sft import preflight_backbone_runtime
    from cosmos_framework.posttrain.model.quantization import family

    error = PackageNotFoundError("unrelated-dependency")

    def fail(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(family, "require_family_runtime", fail)
    config = SimpleNamespace(model=SimpleNamespace(backbone_adaptation=SimpleNamespace(mode="nf4_lora")))
    with pytest.raises(PackageNotFoundError) as result:
        preflight_backbone_runtime(config, torch.device("cuda", 0))
    assert result.value is error


@pytest.mark.parametrize("mode", ["bf16_frozen", "nf4_frozen"])
def test_explicit_base_scoring_requires_frozen_policy_and_never_substitutes_missing_artifact(mode):
    from cosmos_framework.posttrain.entrypoints.action_common import validate_policy_source as validate_source

    native = SimpleNamespace(
        policy_factory="fixture.build_policy", model=SimpleNamespace(backbone_adaptation=SimpleNamespace(mode=mode))
    )
    validate_source(native, checkpoint=None, dense=False, base=True)
    validate_source(native, checkpoint="/committed", dense=False, base=False)
    validate_source(native, checkpoint="/dense", dense=True, base=False)
    validate_source(native, checkpoint="/unmerged", dense=False, base=False, unmerged=True)
    for checkpoint, dense, base in [("/artifact", True, False), (None, False, True), (None, False, False)]:
        with pytest.raises(ValueError, match="exactly one"):
            validate_source(native, checkpoint=checkpoint, dense=dense, base=base, unmerged=True)
    for checkpoint, dense, base in [(None, False, False), ("/committed", False, True), (None, True, True)]:
        with pytest.raises(ValueError, match="exactly one"):
            validate_source(native, checkpoint=checkpoint, dense=dense, base=base)
    native.model.backbone_adaptation.mode = mode.replace("frozen", "lora")
    with pytest.raises(ValueError, match="LoRA needs a checkpoint"):
        validate_source(native, checkpoint=None, dense=False, base=True)


@pytest.mark.parametrize("recipe", ["sft_alpamayo_qlora", "gr00t_n1d7_libero_qlora"])
def test_actual_native_recipe_training_identity_is_finite_json_and_binds_optimizer(recipe, monkeypatch):
    """CLI construction retains ListConfig leaves inside instantiated dataclasses."""
    import json

    from cosmos_framework.posttrain.entrypoints.action_common import validate_config
    from cosmos_framework.posttrain.entrypoints.action_sft import training_identity
    from cosmos_framework.utils.config import load_config

    native = load_config(f"cosmos_framework/posttrain/configs/{recipe}.py", ["--", "rl.optimizer.lr=0.0002"])
    rl = validate_config(native)
    objective = SimpleNamespace(identity={"format": "native-task-fixture"})
    identity = training_identity(native, rl, objective)
    encoded = json.dumps(identity, sort_keys=True, allow_nan=False)
    assert json.loads(encoded) == identity
    assert identity["optimizer"]["betas"] == [0.9, 0.999]
    assert identity["optimizer"]["lr"] == 0.0002
    assert identity["optimizer"]["train_keys"] == []
    assert identity["objective"] == objective.identity
    assert identity["factories"]["policy_factory"] == native.policy_factory
    assert identity["adapter_seed"] == native.model.backbone_adaptation.seed
    # Evaluation cadence/seed/sample count do not change the trajectory; nor does run length
    # unless it is the horizon of a cosine schedule without cycle_steps.
    for name, value in (("evaluation_every", 7), ("evaluation_seed", 5), ("evaluation_max_samples", 3)):
        setattr(rl, name, value)
    rl.max_steps += 1
    assert training_identity(native, rl, objective) == identity
    rl.lr_schedule.kind = "cosine"
    cosine = training_identity(native, rl, objective)
    assert cosine["lr_schedule"]["resolved_cycle_steps"] == rl.max_steps
    rl.max_steps += 1
    assert training_identity(native, rl, objective) != cosine
    rl.lr_schedule.cycle_steps = 50
    cycle = training_identity(native, rl, objective)
    rl.max_steps += 1
    assert training_identity(native, rl, objective) == cycle
    rl.optimizer.lr = 0.0001
    assert training_identity(native, rl, objective) != cycle


def test_build_provenance_is_optional_and_recorded_from_the_environment(monkeypatch):
    from cosmos_framework.posttrain.entrypoints.action_sft import training_provenance

    monkeypatch.delenv("COSMOS_TRAINING_CODE_REVISION", raising=False)
    monkeypatch.delenv("COSMOS_TRAINING_WHEEL_SHA256", raising=False)
    assert training_provenance() == {}
    monkeypatch.setenv("COSMOS_TRAINING_CODE_REVISION", "1" * 40)
    monkeypatch.setenv("COSMOS_TRAINING_WHEEL_SHA256", "2" * 64)
    assert training_provenance() == {"code_revision": "1" * 40, "wheel_sha256": "2" * 64}


class Dataset:
    def __init__(self, *, identity=None, count=5):
        self.identity = identity if identity is not None else {"format": "fixture", "sha256": "a" * 64}
        self.items = [SimpleNamespace(id=f"row-{index}", value=index + 1) for index in range(count)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def config(*, batch_size=2, limit=0):
    return SimpleNamespace(
        rl=SimpleNamespace(micro_batch_size=batch_size, evaluation_seed=971, evaluation_max_samples=limit)
    )


def policy():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(311)
        net = nn.Sequential(nn.Linear(2, 4), nn.Dropout(0.35), nn.Linear(4, 1))
    net[0].eval()
    return SimpleNamespace(net=net)


@pytest.mark.parametrize("fail", [False, True])
def test_evaluation_restores_mixed_modes_rng_and_autograd_even_on_failure(fail):
    model = policy()
    modes = [module.training for module in model.net.modules()]
    original = collect_rng_state()
    initial_parameters = copy.deepcopy(model.net.state_dict())
    for parameter in model.net.parameters():
        parameter.grad = torch.full_like(parameter, 0.125)
    initial_gradients = [parameter.grad.clone() for parameter in model.net.parameters()]

    def run():
        with preserved_evaluation(model, 17):
            assert not torch.is_grad_enabled()
            assert not any(module.training for module in model.net.modules())
            # Consume every supported stream and change one nested mode. The context
            # must restore the actual entry modes, rather than calling net.train().
            random.random()
            np.random.randn(5)
            model.net(torch.randn(3, 2))
            model.net[1].train()
            if fail:
                raise RuntimeError("injected task evaluator failure")

    if fail:
        with pytest.raises(RuntimeError, match="injected task evaluator"):
            run()
    else:
        run()
    assert torch.is_grad_enabled()
    assert [module.training for module in model.net.modules()] == modes
    assert_state_equal(collect_rng_state(), original)
    assert_state_equal(model.net.state_dict(), initial_parameters)
    assert_state_equal([parameter.grad for parameter in model.net.parameters()], initial_gradients)


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_evaluate_uses_native_denominator_not_equal_batch_or_example_weight(batch_size):
    source = ActionSource(Dataset())
    model = policy()

    def objective(model, batch):
        weights = [float(item.value) for item in batch]
        numerator = sum(item.value**2 for item in batch)
        denominator = sum(weights)
        return torch.tensor(numerator / denominator, dtype=torch.float64), {OBJECTIVE_WEIGHT_SUM: denominator}

    def callback(model, batch, config):
        return {"task_error": {"sum": sum(2 * item.value for item in batch), "count": 2 * len(batch)}}

    metrics, rows = evaluate(model, objective, source, config(batch_size=batch_size), callback)
    assert metrics == pytest.approx({"loss": 55 / 15, "task_error": 3.0})
    assert [item for row in rows for item in row["ids"]] == [f"row-{index}" for index in range(5)]


def test_evaluate_limit_and_unweighted_objective_preserve_explicit_row_order():
    source = ActionSource(Dataset())
    metrics, rows = evaluate(
        policy(),
        lambda model, batch: (torch.tensor(sum(item.value for item in batch) / len(batch)), {}),
        source,
        config(batch_size=2, limit=3),
    )
    assert metrics["loss"] == 2.0
    assert [row["ids"] for row in rows] == [["row-0", "row-1"], ["row-2"]]


@pytest.mark.parametrize(
    "extra,match",
    [
        ({"loss": {"sum": 0, "count": 1}}, "must not replace"),
        ({"mse": {"mean": 1}}, "explicit sum and count"),
        ({"mse": {"sum": 1, "count": 0}}, "nonfinite or empty"),
        ({"mse": {"sum": float("nan"), "count": 1}}, "nonfinite or empty"),
        ({"mse": {"sum": 1, "count": float("inf")}}, "nonfinite or empty"),
    ],
)
def test_invalid_task_metrics_fail_and_restore_training_state(extra, match):
    model = policy()
    before_rng = collect_rng_state()
    before_modes = [module.training for module in model.net.modules()]
    with pytest.raises(ValueError, match=match):
        evaluate(
            model,
            lambda model, batch: (torch.rand(()), {}),
            ActionSource(Dataset()),
            config(),
            lambda *_: extra,
        )
    assert_state_equal(collect_rng_state(), before_rng)
    assert [module.training for module in model.net.modules()] == before_modes


def test_source_binds_identity_and_cursor_resumes_exactly_across_epoch_boundary():
    dataset = Dataset()
    source = ActionSource(dataset)
    source_identity = copy.deepcopy(source.identity)
    dataset.identity["sha256"] = "b" * 64
    assert source.identity == source_identity
    cursor = IndexedCursor(source, shuffle=True, seed=19, cursor_resume="strict")
    cursor.advance(7)
    state = cursor.state_dict()
    expected = [cursor.next_index() for _ in range(9)]
    restored = IndexedCursor(ActionSource(Dataset()), shuffle=True, seed=19, cursor_resume="strict")
    restored.load_state_dict(state)
    assert restored.consumed == 7
    assert [restored.next_index() for _ in range(9)] == expected
    changed = IndexedCursor(ActionSource(dataset), shuffle=True, seed=19, cursor_resume="strict")
    with pytest.raises(ValueError, match="fingerprint"):
        changed.load_state_dict(state)


@pytest.mark.parametrize("identity,count", [({}, 5), ({"bad": float("nan")}, 5), ({"format": "fixture"}, 0)])
def test_source_rejects_empty_or_nonfinite_identity_and_empty_data(identity, count):
    with pytest.raises(ValueError):
        ActionSource(Dataset(identity=identity, count=count))


@pytest.mark.parametrize("fail", [False, True])
def test_native_trainer_next_updates_are_identical_with_inserted_evaluation(fail):
    """Use the actual trainer/AdamW/scheduler and stochastic objective, not mocked steps."""
    data = ActionSource(Dataset(count=7))

    def run(insert_evaluation):
        random.seed(51)
        np.random.seed(51)
        torch.manual_seed(51)
        model = policy()
        optimizer = torch.optim.AdamW(model.net.parameters(), lr=0.01, weight_decay=0.02)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

        def loss_fn(model, batch):
            x = torch.tensor([[item.value, -item.value] for item in batch], dtype=torch.float32)
            prediction = model.net(x + torch.rand_like(x))
            target = random.random() + float(np.random.uniform())
            return (prediction - target).square().mean(), {}

        objective = Objective(loss_fn, reduction=Reduction.MEAN, identity={"task": "cpu-lifecycle-fixture"})
        trainer = LLMTrainer(
            model, objective, optimizer, lr_scheduler=scheduler, require_gradients=True, nonfinite_update="fail_stop"
        )
        cursor = IndexedCursor(data, shuffle=True, seed=42, cursor_resume="strict")
        updates = []
        for step in range(4):
            batch = [data[cursor.next_index()] for _ in range(2)]
            updates.append(trainer.train(batch))
            if insert_evaluation and step == 1:
                before_cursor = cursor.state_dict()
                before_trainer = copy.deepcopy(trainer._trainer_state())

                def callback(*_):
                    if fail:
                        raise RuntimeError("evaluation failed after native loss")
                    return {}

                if fail:
                    with pytest.raises(RuntimeError, match="evaluation failed"):
                        evaluate(model, objective, data, config(), callback)
                else:
                    evaluate(model, objective, data, config(), callback)
                assert cursor.state_dict() == before_cursor
                assert_state_equal(trainer._trainer_state(), before_trainer)
        assert trainer.iteration() == 4
        return {
            "parameters": copy.deepcopy(model.net.state_dict()),
            "optimizer": copy.deepcopy(optimizer.state_dict()),
            "scheduler": copy.deepcopy(scheduler.state_dict()),
            "trainer": copy.deepcopy(trainer._trainer_state()),
            "cursor": cursor.state_dict(),
            "rng": collect_rng_state(),
            "updates": updates,
        }

    original_rng = collect_rng_state()
    try:
        control = run(False)
        evaluated = run(True)
        assert_state_equal(evaluated, control)
    finally:
        restore_rng_state(original_rng)


@pytest.mark.parametrize("head_lr_multiplier", [None, 0.1])
@pytest.mark.parametrize("completed_updates", [0, 3])
def test_native_adapter_checkpoint_restores_head_optimizer_rng_cursor_and_next_update(
    tmp_path, completed_updates, head_lr_multiplier
):
    """Exercise real DCP/codec/trainer at pristine and trained completed-update boundaries."""
    from cosmos_framework.posttrain.checkpoint.adapter import read_adapter_metadata
    from cosmos_framework.posttrain.checkpoint.manager import CheckpointManager
    from cosmos_framework.posttrain.model.quantization.family import FamilyBackboneConfig, install_family_backbone
    from cosmos_framework.posttrain.utils.rank_state import collect_rank_state, state_for_current_rank

    class TaskNet(nn.Module):
        def __init__(self):
            super().__init__()
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(916)
                self.backbone = nn.Linear(2, 4, bias=False, dtype=torch.bfloat16)
                self.head = nn.Linear(4, 1, dtype=torch.float32)

        def forward(self, inputs):
            return self.head(self.backbone(inputs.to(torch.bfloat16)).float())

    def build():
        net = TaskNet()
        family = install_family_backbone(
            net,
            config=FamilyBackboneConfig(mode="bf16_lora", rank=2, alpha=4, seed=39),
            targets={"backbone": (4, 2)},
            auxiliary_parameters=("head.weight", "head.bias"),
            source_identity={"repository": "fixture/action-dcp", "revision": "a" * 40},
            task_identity={"objective": "fixture-regression", "action_dim": 1},
            device=torch.device("cpu"),
        )
        family.codec.bind_training_identity({"seed": 19, "effective_batch": 2})
        model = SimpleNamespace(net=net)
        if head_lr_multiplier is None:
            optimizer = torch.optim.AdamW(family.codec.parameters.values(), lr=0.01)
        else:
            from cosmos_framework.posttrain.config.build import build_optimizer
            from cosmos_framework.posttrain.config.rl import OptimizerConfig

            optimizer = build_optimizer(
                net,
                OptimizerConfig(
                    name="adamw",
                    impl="fused",
                    lr=0.01,
                    weight_decay=0.0,
                    lr_multipliers={"head.": head_lr_multiplier},
                ),
            )
            assert [g["lr_multiplier"] for g in optimizer.param_groups] == [1.0, head_lr_multiplier]
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

        def objective(model, batch):
            inputs = torch.tensor([[row.value, -row.value] for row in batch], dtype=torch.float32)
            target = torch.rand(len(batch), 1)
            return (model.net(inputs) - target).square().mean(), {}

        trainer = LLMTrainer(
            model,
            Objective(objective, reduction=Reduction.MEAN, identity={"objective": "fixture-action"}),
            optimizer,
            lr_scheduler=scheduler,
            checkpoint_manager=CheckpointManager(checkpoint_format="dcp", keep_optim_last=0),
            require_gradients=True,
            nonfinite_update="fail_stop",
        )
        source = ActionSource(Dataset(count=7))
        cursor = IndexedCursor(source, shuffle=True, seed=19, cursor_resume="strict")
        return model, family, trainer, source, cursor

    def update(trainer, source, cursor):
        metrics = trainer.train([source[cursor.next_index()] for _ in range(2)])
        assert metrics["optim/update_outcome"] == "applied"

    saved_rng = collect_rng_state()
    try:
        torch.manual_seed(19)
        model, family, trainer, source, cursor = build()
        for _ in range(completed_updates):
            update(trainer, source, cursor)
        history = [{"step": completed_updates, "metrics": {"loss": 0.25}}]
        data_state = collect_rank_state(
            {"format": "action-sft-state-v1", "cursor": cursor.state_dict(), "evaluation": history}
        )
        path = str(tmp_path / "checkpoints" / f"iter_{completed_updates:06d}")
        before_save = copy.deepcopy(trainer._trainer_state())
        before_optimizer = copy.deepcopy(trainer.opt.state_dict())
        trainer.save(path, data_state=data_state)
        trainer.finalize_checkpoints()
        metadata = read_adapter_metadata(path, require_resume=True, expected_identity=family.codec.identity)
        assert set(metadata["parameters"]) == set(family.codec.parameters)
        assert {"head.weight", "head.bias"} <= set(metadata["parameters"])
        assert_state_equal(trainer._trainer_state(), before_save)
        assert_state_equal(trainer.opt.state_dict(), before_optimizer)
        update(trainer, source, cursor)
        uninterrupted = {
            "model": copy.deepcopy(model.net.state_dict()),
            "optimizer": copy.deepcopy(trainer.opt.state_dict()),
            "trainer": copy.deepcopy(trainer._trainer_state()),
            "cursor": cursor.state_dict(),
        }
        restored, restored_family, resumed, resumed_source, resumed_cursor = build()
        assert resumed.load(path) == completed_updates
        recovered = state_for_current_rank(resumed.checkpoint_manager.load_component(path, "data"))
        assert recovered["evaluation"] == history
        resumed_cursor.load_state_dict(recovered["cursor"])
        assert resumed_cursor.consumed == completed_updates * 2
        update(resumed, resumed_source, resumed_cursor)
        restored_family.validate_frozen_base()
        restored_family.assert_optimizer(resumed.opt)
        assert_state_equal(
            {
                "model": restored.net.state_dict(),
                "optimizer": resumed.opt.state_dict(),
                "trainer": resumed._trainer_state(),
                "cursor": resumed_cursor.state_dict(),
            },
            uninterrupted,
        )
    finally:
        restore_rng_state(saved_rng)


def test_input_admission_does_not_require_model_and_rejects_overlapping_groups(monkeypatch):
    from cosmos_framework.posttrain.entrypoints import action_common, action_sft

    train = Dataset(count=3)
    validation = Dataset(count=2)
    train.groups = {"train-recording"}
    validation.groups = {"validation-recording"}
    calls = []

    def data_factory(config, split, policy):
        assert policy is None
        calls.append(split)
        return train if split == "train" else validation

    def factory(path):
        assert path == "fixture.data", "model factory must not be called during input admission"
        return data_factory

    monkeypatch.setattr(action_common, "load_factory", factory)
    config = SimpleNamespace(data_factory="fixture.data", policy_factory="forbidden.model")
    left, right = action_sft.prepare_sources(config)
    assert len(left) == 3 and len(right) == 2 and calls == ["train", "validation"]
    validation.groups = train.groups
    with pytest.raises(ValueError, match="overlap in groups"):
        action_sft.prepare_sources(config)


def test_missing_native_inputs_fail_during_cpu_admission(monkeypatch):
    from cosmos_framework.posttrain.entrypoints import action_common, action_sft

    def missing(config, split, policy):
        assert policy is None
        raise FileNotFoundError("missing mounted media")

    monkeypatch.setattr(action_common, "load_factory", lambda path: missing)
    with pytest.raises(FileNotFoundError, match="missing mounted media"):
        action_sft.prepare_sources(SimpleNamespace(data_factory="fixture.data"))


@pytest.mark.parametrize("recipe,enabled", [("gr00t_n1d7_libero_qlora", True), ("sft_alpamayo_qlora", False)])
def test_entrypoint_tf32_follows_the_family_recipe(recipe, enabled):
    from cosmos_framework.posttrain.entrypoints.action_common import configure_tf32
    from cosmos_framework.utils.config import load_config

    config = load_config(f"cosmos_framework/posttrain/configs/{recipe}.py", ["--"])
    previous = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        configure_tf32(config)
        assert torch.backends.cuda.matmul.allow_tf32 is enabled
        assert torch.backends.cudnn.allow_tf32 is enabled
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous


@pytest.mark.parametrize(("forced", "expected"), [(True, "math"), (False, "default")])
def test_gr00t_training_contract_binds_the_effective_dit_sdpa_backend(monkeypatch, forced, expected):
    import sys
    import types

    from cosmos_framework.posttrain.model.gr00t import action_sft

    # Upstream's DiT switches attention backends from GR00T_DIT_SDPA_MODE on every forward.
    dit = types.ModuleType("gr00t.model.modules.dit")
    dit._should_force_math_sdpa = lambda: forced
    monkeypatch.setitem(sys.modules, "gr00t.model.modules.dit", dit)
    policy = SimpleNamespace(packer=lambda batch: batch, config=SimpleNamespace(training_prefix=10, action_dim=7))
    objective = action_sft.build_objective(policy, None)
    assert objective.identity == {"algorithm": "flow_bc", "training_prefix": 10, "action_dim": 7, "dit_sdpa": expected}
