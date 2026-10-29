# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Pre-allocation SFT resume contracts; these CPU tests do not qualify quantized kernels."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch

from cosmos_framework.posttrain.algorithm.primitives.objective import Objective, Reduction
from cosmos_framework.posttrain.config.build import resolve_precision
from cosmos_framework.posttrain.configs.sft_qwen3_vl_8b_qlora import make_config
from cosmos_framework.posttrain.entrypoints import sft
from cosmos_framework.posttrain.tests.test_checkpoint_adapter import TinyAdapterModel, _save, _trainer


@pytest.fixture(autouse=True)
def _installed_build(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COSMOS_TRAINING_CODE_REVISION", "a" * 40)
    monkeypatch.setenv("COSMOS_TRAINING_WHEEL_SHA256", "b" * 64)


@pytest.fixture(autouse=True)
def _portable_processor_identity(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Exercise entrypoint wiring; actual snapshot proofs live in test_adapter_identity."""
    from cosmos_framework.posttrain.utils import adapter_identity

    def identity(packer, *, source_path, source_identity):
        return {
            **copy.deepcopy(packer.validation_semantics_identity()),
            "adapter_processor_identity": {"protocol": "controlled-portable-processor"},
        }

    helper = Mock(side_effect=identity)
    monkeypatch.setattr(adapter_identity, "adapter_processor_identity", helper)
    return helper


def _packer(config: Any) -> SimpleNamespace:
    return SimpleNamespace(
        validation_semantics_identity=lambda: {
            "template": "controlled-template",
            "max_length": config.rl.max_length,
            "processor_kwargs": copy.deepcopy(config.rl.processor_kwargs),
            "mask_think_opener": config.rl.mask_think_opener,
        }
    )


def _objective_identity(config: Any, *, source: str = "immutable-training-data") -> dict[str, Any]:
    return {
        "kind": "controlled-sft",
        "training_source": source,
        "micro_batch_size": config.rl.micro_batch_size,
        "grad_accum": config.rl.grad_accum,
        "dp_world": 1,
        "precision": asdict(resolve_precision(config)),
    }


def _checkpoint(config: Any, tmp_path: Path) -> SimpleNamespace:
    from cosmos_framework.utils.lazy_config import instantiate

    processor, training = sft._adapter_run_contract(config, _packer(config), _objective_identity(config))
    trainer = _trainer()
    cfg = instantiate(config.model.model_config)
    trainer._adapter_checkpoint_codec().identity = {
        "base": dict(cfg.source_identity),
        "adapter": asdict(cfg.adapter),
        "quantization": asdict(cfg.quantized_base) if cfg.quantized_base is not None else None,
        "processor_contract": processor,
        "training_contract": training,
    }
    path = _save(trainer, tmp_path)
    return SimpleNamespace(manager=trainer.checkpoint_manager, resume_checkpoint=str(path))


def test_adapter_contract_is_finite_json_and_binds_processor_separately() -> None:
    config = make_config()
    objective = _objective_identity(config)
    processor, training = sft._adapter_run_contract(config, _packer(config), objective)
    assert json.loads(json.dumps([processor, training], allow_nan=False)) == [processor, training]
    assert processor["processor_kwargs"]["videos_kwargs"]["num_frames"] == 8
    assert training["objective"] == objective
    assert training["max_steps"] == config.rl.max_steps
    assert training["seed"] == config.rl.seed
    assert training["deterministic"] is True
    assert training["determinism_mode"] == "strict"
    assert training["code_revision"] == "a" * 40
    assert training["wheel_sha256"] == "b" * 64
    assert "processor_kwargs" not in training


def test_adapter_contract_forwards_verified_source_and_preserves_helper_result(
    _portable_processor_identity: Mock,
) -> None:
    config = make_config()
    packer = _packer(config)
    portable = {"proof": "verified-content-identity", "settings": {"video_frames": 8}}
    _portable_processor_identity.side_effect = None
    _portable_processor_identity.return_value = portable

    processor, _ = sft._adapter_run_contract(config, packer, _objective_identity(config))

    _portable_processor_identity.assert_called_once_with(
        packer,
        source_path=config.model.model_id,
        source_identity=config.model.model_config.source_identity,
    )
    assert processor is portable


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("COSMOS_TRAINING_CODE_REVISION", ""),
        ("COSMOS_TRAINING_CODE_REVISION", "main"),
        ("COSMOS_TRAINING_WHEEL_SHA256", ""),
        ("COSMOS_TRAINING_WHEEL_SHA256", "b" * 63),
    ],
)
def test_adapter_contract_requires_immutable_installed_build_receipt(
    name: str, value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(name, value)
    config = make_config()
    with pytest.raises(ValueError, match="installed build receipt"):
        sft._adapter_run_contract(config, _packer(config), _objective_identity(config))


def test_dense_path_does_not_require_adapter_receipt_or_processor_contract(
    monkeypatch: pytest.MonkeyPatch, _portable_processor_identity: Mock
) -> None:
    monkeypatch.delenv("COSMOS_TRAINING_CODE_REVISION")
    monkeypatch.delenv("COSMOS_TRAINING_WHEEL_SHA256")
    config = make_config()
    config.model.model_config.adapter = None
    config.model.model_config.quantized_base = None
    assert sft._adapter_run_contract(config, object(), {}) is None
    _portable_processor_identity.assert_not_called()


@pytest.mark.parametrize("quantized", [False, True])
def test_native_adapter_refuses_determinism_opt_out_before_source_setup(monkeypatch, quantized):
    config = make_config()
    if not quantized:
        config.model.model_config.quantized_base = None
    config.rl.deterministic = False
    source = Mock(side_effect=AssertionError("invalid execution policy must fail before source setup"))
    monkeypatch.setattr(sft, "sft_policy_source", source)
    with pytest.raises(ValueError, match="native adapter SFT requires rl.deterministic=True"):
        sft.validate_config_structure(config)
    source.assert_not_called()
    with pytest.raises(ValueError, match="strict deterministic algorithms"):
        sft._adapter_run_contract(config, _packer(config), _objective_identity(config))


@pytest.mark.parametrize("quantized", [False, True])
def test_native_adapter_driver_enables_strict_before_cuda_process_group(monkeypatch, quantized):
    from cosmos_framework.posttrain.utils import determinism, dist

    config = make_config()
    config.rl.validation.enabled = False
    if not quantized:
        config.model.model_config.quantized_base = None
    events = []
    monkeypatch.setattr(sft, "validate_config", lambda cfg: cfg.rl)
    monkeypatch.setattr(determinism, "enable_determinism", lambda **kwargs: events.append(("determinism", kwargs)))

    class StopBeforeCuda(RuntimeError):
        pass

    def initialize():
        events.append(("process_group", {}))
        raise StopBeforeCuda

    monkeypatch.setattr(dist, "init_local_process_group", initialize)
    with pytest.raises(StopBeforeCuda):
        sft.run(config, "recipe.py", [], "/unused-output")
    assert events == [("determinism", {"strict": True}), ("process_group", {})]


@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("resumed", [False, True])
def test_native_adapter_worker_enables_strict_before_construction_and_resume(monkeypatch, quantized, resumed):
    from cosmos_framework.posttrain.control.checkpointing import CheckpointLifecycle
    from cosmos_framework.posttrain.roles.env import sft as data_env
    from cosmos_framework.posttrain.utils import determinism
    from cosmos_framework.utils import config as config_loader

    config = make_config()
    config.rl.validation.enabled = False
    if not quantized:
        config.model.model_config.quantized_base = None
    events = []
    monkeypatch.setattr(config_loader, "load_config", lambda *args: config)
    monkeypatch.setattr(sft, "validate_config", lambda cfg: cfg.rl)
    monkeypatch.setattr(determinism, "enable_determinism", lambda **kwargs: events.append(("determinism", kwargs)))

    def checkpoint(*args):
        events.append(("checkpoint", {}))
        return SimpleNamespace(resume_checkpoint="committed-checkpoint" if resumed else None)

    def dataset(*args, **kwargs):
        events.append(("data", {}))
        return SimpleNamespace(source_fingerprint="controlled-training-source")

    def trainer(cfg, rl, data, validation, state):
        events.append(("model", {}))
        if state.resume_checkpoint:
            events.append(("restore", {}))
        return "controlled-trainer"

    monkeypatch.setattr(CheckpointLifecycle, "from_config", checkpoint)
    monkeypatch.setattr(data_env, "build_sft_env", dataset)
    monkeypatch.setattr(sft, "build_trainer", trainer)
    assert sft.make_setup("recipe.py", [])()["policy"] == "controlled-trainer"
    assert events == [
        ("determinism", {"strict": True}),
        ("checkpoint", {}),
        ("data", {}),
        ("model", {}),
        *([("restore", {})] if resumed else []),
    ]


@pytest.mark.parametrize("enabled", [False, True])
def test_general_sft_retains_default_determinism_enforcement(monkeypatch, enabled):
    from cosmos_framework.posttrain.utils import determinism

    config = make_config()
    config.model.model_config.adapter = None
    config.model.model_config.quantized_base = None
    config.rl.deterministic = enabled
    helper = Mock()
    monkeypatch.setattr(determinism, "enable_determinism", helper)
    sft._enable_sft_determinism(config, adapter_only=True)
    helper.assert_not_called()
    sft._enable_sft_determinism(config)
    if enabled:
        helper.assert_called_once_with()
    else:
        helper.assert_not_called()


def test_resume_rejects_a_different_determinism_mode_before_loading(tmp_path):
    config = make_config()
    checkpoint = _checkpoint(config, tmp_path)
    processor, training = sft._adapter_run_contract(config, _packer(config), _objective_identity(config))
    training["determinism_mode"] = "warn_only"
    with pytest.raises(ValueError, match="determinism_mode.*differs"):
        sft._preflight_adapter_resume(config, checkpoint, processor, training)


def test_adapter_contract_survives_typed_dotlist_round_trip() -> None:
    from cosmos_framework.utils.config_helper import override

    config = make_config()
    restored = override(config, ["--", f"rl.max_steps={config.rl.max_steps}"])
    assert sft._adapter_run_contract(config, _packer(config), _objective_identity(config)) == sft._adapter_run_contract(
        restored, _packer(restored), _objective_identity(restored)
    )


def test_cli_loaded_config_preflights_resume_without_dataclass_assumptions(tmp_path: Path) -> None:
    from cosmos_framework.utils.config import load_config

    recipe = "cosmos_framework/posttrain/configs/sft_qwen3_vl_8b_qlora.py"
    options = ["--", "rl.max_steps=3", "rl.lr_schedule.cycle_steps=3", "rl.optimizer.warmup_steps=1"]
    config = load_config(recipe, options)
    checkpoint = _checkpoint(config, tmp_path)
    restored = load_config(recipe, options)
    processor, training = sft._adapter_run_contract(restored, _packer(restored), _objective_identity(restored))
    sft._preflight_adapter_resume(restored, checkpoint, processor, training)


def test_structured_dict_model_config_preflights_without_asdict_type_error(tmp_path: Path) -> None:
    from omegaconf import OmegaConf

    config = make_config()
    checkpoint = _checkpoint(config, tmp_path)
    config.model.model_config = OmegaConf.structured(config.model.model_config)
    processor, training = sft._adapter_run_contract(config, _packer(config), _objective_identity(config))
    sft._preflight_adapter_resume(config, checkpoint, processor, training)


@pytest.mark.parametrize("defect", ["format", "async", "store", "init_from", "optimizer_only", "validation_partial"])
def test_fresh_adapter_build_rejects_checkpoint_policy_before_allocation(monkeypatch, defect: str) -> None:
    from cosmos_framework.posttrain.checkpoint import CheckpointManager
    from cosmos_framework.posttrain.config import build

    config = make_config()
    if defect == "format":
        config.rl.checkpoint_format = "consolidated"
    elif defect == "async":
        config.rl.save_async = True
    elif defect == "store":
        config.rl.checkpoint_store.backend = "s3"
    elif defect == "init_from":
        config.rl.init_from = "/another/base"
    elif defect == "optimizer_only":
        config.rl.checkpoint_components = ["optimizer"]
    else:
        config.rl.validation.save_checkpoint = True
        config.rl.validation.checkpoint_components = ["training_state"]
    events = []
    _patch_construction(monkeypatch, config, events)

    def forbidden_model(*args, **kwargs):
        events.append("backbone")
        raise AssertionError("unsupported checkpoint policy must fail before model allocation")

    monkeypatch.setattr(build, "build_backbone", forbidden_model)
    checkpoint = SimpleNamespace(manager=CheckpointManager(checkpoint_format="dcp"), resume_checkpoint=None)
    with pytest.raises(ValueError, match="native adapter"):
        sft.build_trainer(
            config, config.rl, SimpleNamespace(source_fingerprint="immutable-training-data"), None, checkpoint
        )
    assert events == []


def test_preflight_accepts_same_identity_and_skips_fresh_checkpoint_io(tmp_path: Path, monkeypatch) -> None:
    config = make_config()
    checkpoint = _checkpoint(config, tmp_path)
    processor, training = sft._adapter_run_contract(config, _packer(config), _objective_identity(config))
    sft._preflight_adapter_resume(config, checkpoint, processor, training)
    checkpoint.resume_checkpoint = None

    def unexpected_read(*args, **kwargs):
        raise AssertionError("fresh training must not read a checkpoint")

    monkeypatch.setattr("cosmos_framework.posttrain.checkpoint.adapter.read_adapter_metadata", unexpected_read)
    sft._preflight_adapter_resume(config, checkpoint, processor, training)


@pytest.mark.parametrize(
    "change",
    ["base", "rank", "quantization", "frames", "template", "lr", "schedule", "horizon", "source", "batch", "seed"],
)
def test_preflight_rejects_resume_identity_changes(tmp_path: Path, change: str) -> None:
    config = make_config()
    checkpoint = _checkpoint(config, tmp_path)
    if change == "base":
        config.model.model_config.source_identity["revision"] = "c" * 40
    elif change == "rank":
        config.model.model_config.adapter.rank *= 2
    elif change == "quantization":
        config.model.model_config.quantized_base = None
    elif change == "frames":
        config.rl.processor_kwargs["videos_kwargs"]["num_frames"] = 4
    elif change == "lr":
        config.rl.optimizer.lr *= 2
    elif change == "schedule":
        config.rl.lr_schedule.f_min = 0.2
    elif change == "horizon":
        config.rl.max_steps += 1
    elif change == "batch":
        config.rl.grad_accum *= 2
    elif change == "seed":
        config.rl.seed += 1
    objective = _objective_identity(
        config, source="other-training-data" if change == "source" else "immutable-training-data"
    )
    processor, training = sft._adapter_run_contract(config, _packer(config), objective)
    if change == "template":
        processor["template"] = "other-template"
    with pytest.raises(ValueError, match="differs"):
        sft._preflight_adapter_resume(config, checkpoint, processor, training)


def _patch_construction(monkeypatch: pytest.MonkeyPatch, config: Any, events: list[str]) -> None:
    from cosmos_framework.posttrain.algorithm import sft as sft_algorithm
    from cosmos_framework.posttrain.config import build
    from cosmos_framework.posttrain.data.packer import sft as packer_module
    from cosmos_framework.utils import misc

    monkeypatch.setattr(sft, "sft_policy_source", lambda _config: ("/verified/snapshot", False))
    monkeypatch.setattr(packer_module, "build_sft_packer", lambda *args, **kwargs: _packer(config))

    def objective(*args, **kwargs):
        result = Objective(lambda *_: None, reduction=Reduction.TOKEN_SUM)
        result.identity = {"kind": "controlled-sft"}
        return result

    monkeypatch.setattr(sft_algorithm, "make_objective", objective)
    monkeypatch.setattr(misc, "set_random_seed", lambda *, seed, by_rank: events.append(f"seed:{by_rank}"))
    monkeypatch.setattr(build, "build_sft_lr_scheduler", lambda *args: None)


@pytest.mark.parametrize("defect", ["wrong_base", "missing_metadata", "missing_data", "missing_optimizer"])
def test_build_trainer_resume_refuses_before_backbone_allocation(tmp_path: Path, monkeypatch, defect: str) -> None:
    from cosmos_framework.posttrain.config import build

    config = make_config()
    checkpoint = _checkpoint(config, tmp_path)
    if defect == "wrong_base":
        config.model.model_config.source_identity["revision"] = "c" * 40
    else:
        component = {
            "missing_metadata": "model_metadata.pt",
            "missing_data": "data.pt",
            "missing_optimizer": "optim/.metadata",
        }[defect]
        (Path(checkpoint.resume_checkpoint) / component).unlink()
    events = []
    _patch_construction(monkeypatch, config, events)

    def forbidden_model(*args, **kwargs):
        events.append("backbone")
        raise AssertionError("incompatible checkpoint must fail before model allocation")

    monkeypatch.setattr(build, "build_backbone", forbidden_model)
    with pytest.raises((ValueError, RuntimeError)):
        sft.build_trainer(
            config, config.rl, SimpleNamespace(source_fingerprint="immutable-training-data"), None, checkpoint
        )
    assert events == []


def test_build_trainer_binds_identities_before_trainer_and_authoritative_resume(
    tmp_path: Path, monkeypatch, _portable_processor_identity: Mock
) -> None:
    from cosmos_framework.posttrain.config import build
    from cosmos_framework.posttrain.roles.trainer import llm

    config = make_config()
    checkpoint = _checkpoint(config, tmp_path)
    events = []
    _patch_construction(monkeypatch, config, events)
    identity_helper = _portable_processor_identity.side_effect
    expected_processor = None

    def processor_identity(packer, *, source_path, source_identity):
        nonlocal expected_processor
        events.append("processor_identity")
        assert source_path == config.model.model_id
        assert source_identity is config.model.model_config.source_identity
        expected_processor = identity_helper(packer, source_path=source_path, source_identity=source_identity)
        return expected_processor

    _portable_processor_identity.side_effect = processor_identity
    original_preflight = sft._preflight_adapter_resume

    def preflight(config, checkpoint, processor, training):
        events.append("preflight")
        assert processor is expected_processor
        return original_preflight(config, checkpoint, processor, training)

    monkeypatch.setattr(sft, "_preflight_adapter_resume", preflight)
    net = TinyAdapterModel()
    codec = net._cosmos_adapter_checkpoint_codec
    codec.identity.pop("processor_contract")
    codec.identity.pop("training_contract")
    for name in ("processor", "training"):
        original = getattr(codec, f"bind_{name}_identity")

        def bind(value, original=original, name=name):
            events.append(f"bind:{name}")
            original(value)

        monkeypatch.setattr(codec, f"bind_{name}_identity", bind)

    def backbone(_config):
        events.append("backbone")
        return SimpleNamespace(net=net)

    def optimizer(model, _config):
        events.append("optimizer")
        return torch.optim.AdamW(model._cosmos_adapter_parameters.values(), lr=0.01)

    original_trainer = llm.LLMTrainer

    def trainer(*args, **kwargs):
        events.append("trainer")
        return original_trainer(*args, **kwargs)

    def restore(policy, component):
        assert component == "data"
        assert policy._adapter_checkpoint_codec().identity["processor_contract"] == expected_processor
        assert "training_contract" in policy._adapter_checkpoint_codec().identity
        events.append("restore")
        return 0, {"cursor": 0}

    monkeypatch.setattr(build, "build_backbone", backbone)
    monkeypatch.setattr(build, "build_optimizer", optimizer)
    monkeypatch.setattr(llm, "LLMTrainer", trainer)
    checkpoint.load_trainer_and_component = restore
    data = SimpleNamespace(
        source_fingerprint="immutable-training-data",
        load_state_dict=lambda state: events.append("cursor"),
    )
    sft.build_trainer(config, config.rl, data, None, checkpoint)
    assert events == [
        "processor_identity",
        "preflight",
        "seed:False",
        "backbone",
        "optimizer",
        "bind:processor",
        "bind:training",
        "trainer",
        "seed:True",
        "restore",
        "cursor",
    ]


def test_processor_identity_failure_refuses_before_backbone_allocation(
    monkeypatch: pytest.MonkeyPatch, _portable_processor_identity: Mock
) -> None:
    from cosmos_framework.posttrain.checkpoint import CheckpointManager
    from cosmos_framework.posttrain.config import build

    config = make_config()
    events = []
    _patch_construction(monkeypatch, config, events)
    _portable_processor_identity.side_effect = ValueError("processor source identity proof failed")
    backbone = Mock(side_effect=AssertionError("processor identity must be verified before allocation"))
    monkeypatch.setattr(build, "build_backbone", backbone)
    checkpoint = SimpleNamespace(manager=CheckpointManager(checkpoint_format="dcp"), resume_checkpoint=None)

    with pytest.raises(ValueError, match="processor source identity proof failed"):
        sft.build_trainer(
            config, config.rl, SimpleNamespace(source_fingerprint="immutable-training-data"), None, checkpoint
        )

    _portable_processor_identity.assert_called_once()
    backbone.assert_not_called()
    assert events == []


def test_native_entrypoint_resume_restores_real_cursor_optimizer_and_rng(tmp_path: Path, monkeypatch) -> None:
    from cosmos_framework.posttrain.algorithm import sft as sft_algorithm
    from cosmos_framework.posttrain.checkpoint import CheckpointManager
    from cosmos_framework.posttrain.config import build
    from cosmos_framework.posttrain.control.checkpointing import CheckpointLifecycle
    from cosmos_framework.posttrain.roles.env.sft import build_sft_env
    from cosmos_framework.posttrain.tests.test_checkpoint_adapter import _objective, _parameters
    from cosmos_framework.utils import misc

    config = make_config()
    config.rl.grad_accum = 1
    config.rl.max_steps = 4
    config.rl.validation.enabled = False
    data_path = tmp_path / "train.jsonl"
    data_path.write_text(
        "\n".join(
            json.dumps(
                {"messages": [{"role": "user", "content": str(index)}, {"role": "assistant", "content": "answer"}]}
            )
            for index in range(6)
        )
    )
    config.rl.data.path = str(data_path)
    seed_function = misc.set_random_seed
    _patch_construction(monkeypatch, config, [])
    monkeypatch.setattr(misc, "set_random_seed", seed_function)

    def objective(*args, **kwargs):
        result = Objective(lambda model, batch: _objective(model.net, batch), reduction=Reduction.MEAN)
        result.identity = {"kind": "controlled-sft"}
        return result

    def backbone(_config):
        net = TinyAdapterModel()
        cfg = config.model.model_config
        net._cosmos_adapter_checkpoint_codec.identity = {
            "base": dict(cfg.source_identity),
            "adapter": asdict(cfg.adapter),
            "quantization": asdict(cfg.quantized_base),
        }
        return SimpleNamespace(net=net)

    monkeypatch.setattr(sft_algorithm, "make_objective", objective)
    monkeypatch.setattr(build, "build_backbone", backbone)
    monkeypatch.setattr(
        build, "build_optimizer", lambda net, _: torch.optim.AdamW(net._cosmos_adapter_parameters.values(), lr=0.01)
    )
    root = tmp_path / "checkpoints"
    initial = CheckpointLifecycle(CheckpointManager(checkpoint_format="dcp"), str(root), None)
    data = build_sft_env(config.rl.data, config.rl.seed)
    trainer = sft.build_trainer(config, config.rl, data, None, initial)
    for _ in range(2):
        trainer.train(data.get_samples(1))
    checkpoint = str(root / "iter_000002")
    trainer.save(checkpoint, data_state=data.state_dict())
    expected_batch = data.get_samples(1)
    trainer.train(expected_batch)
    expected = _parameters(trainer)

    continuation = CheckpointLifecycle(CheckpointManager(checkpoint_format="dcp"), str(root), checkpoint)
    restored_data = build_sft_env(config.rl.data, config.rl.seed)
    restored = sft.build_trainer(config, config.rl, restored_data, None, continuation)
    assert restored._iter == 2
    assert restored.loaded_checkpoint_provenance() == restored_data.loaded_checkpoint_provenance()
    actual_batch = restored_data.get_samples(1)
    assert actual_batch[0].transitions[0].targets.messages == expected_batch[0].transitions[0].targets.messages
    restored.train(actual_batch)
    for name, parameter in _parameters(restored).items():
        torch.testing.assert_close(parameter, expected[name], rtol=0, atol=0)
