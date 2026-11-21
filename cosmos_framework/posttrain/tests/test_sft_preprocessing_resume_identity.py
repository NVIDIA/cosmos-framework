# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""Entrypoint-bound preprocessing identity and real CPU DCP admission."""

import copy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from cosmos_framework.posttrain.config.model import native_adapter_config_identity, quantization_config_identity
from cosmos_framework.posttrain.configs.sft_qwen3_vl_8b_qlora import make_config
from cosmos_framework.posttrain.entrypoints import sft
from cosmos_framework.posttrain.tests.test_checkpoint_adapter import _save, _trainer
from cosmos_framework.posttrain.tests.test_entrypoints_sft_qlora import _patch_construction
from cosmos_framework.utils.lazy_config import instantiate

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


class _BeforeBackbone(RuntimeError):
    pass


@pytest.fixture
def wiring(monkeypatch):
    from cosmos_framework.posttrain.config import build
    from cosmos_framework.posttrain.data.packer import sft_preprocess
    from cosmos_framework.posttrain.utils import adapter_identity

    monkeypatch.setenv("COSMOS_TRAINING_CODE_REVISION", "a" * 40)
    monkeypatch.setenv("COSMOS_TRAINING_WHEEL_SHA256", "b" * 64)
    monkeypatch.setattr(
        adapter_identity,
        "adapter_processor_identity",
        lambda packer, **kwargs: copy.deepcopy(packer.validation_semantics_identity()),
    )
    preprocessing = Mock(side_effect=lambda packer, **kwargs: packer)
    backbone = Mock(side_effect=_BeforeBackbone)
    monkeypatch.setattr(sft_preprocess, "build_sft_preprocessing", preprocessing)
    monkeypatch.setattr(build, "build_backbone", backbone)
    contracts = []
    real_preflight = sft._preflight_adapter_resume

    def preflight(config, checkpoint, processor, training):
        contracts.append((copy.deepcopy(processor), copy.deepcopy(training)))
        return real_preflight(config, checkpoint, processor, training)

    monkeypatch.setattr(sft, "_preflight_adapter_resume", preflight)

    def invoke(config, checkpoint=None):
        _patch_construction(monkeypatch, config, [])
        data = SimpleNamespace(source_fingerprint="immutable-training-data", dataset=SimpleNamespace(media_hashes={}))
        return sft.build_trainer(config, config.rl, data, None, checkpoint or SimpleNamespace(resume_checkpoint=None))

    return SimpleNamespace(invoke=invoke, contracts=contracts, preprocessing=preprocessing, backbone=backbone)


def _config():
    config = make_config()
    config.rl.preprocessing_cache_bytes = 1024
    config.rl.preprocessing_prefetch = True
    return config


def test_default_and_explicit_one_worker_keep_original_entrypoint_training_identity(wiring):
    config = _config()
    with pytest.raises(_BeforeBackbone):
        wiring.invoke(config)
    original = wiring.contracts[-1]
    assert wiring.preprocessing.call_args.kwargs["workers"] == 1
    assert original[1]["objective"]["preprocessing"] == {
        "protocol": "cosmos.sft-preprocessing.v1",
        "cache_bytes": 1024,
        "prefetch": True,
        "window_examples": config.rl.micro_batch_size * config.rl.grad_accum,
    }
    config.rl.preprocessing_workers = 1
    with pytest.raises(_BeforeBackbone):
        wiring.invoke(config)
    assert wiring.contracts[-1] == original


@pytest.mark.parametrize("saved_workers,new_workers", [(1, 2), (2, 1)])
def test_real_preprocessing_checkpoint_accepts_same_workers_and_refuses_changed_workers_before_backbone(
    tmp_path, wiring, saved_workers, new_workers
):
    config = _config()
    config.rl.preprocessing_workers = saved_workers
    with pytest.raises(_BeforeBackbone):
        wiring.invoke(config)
    processor, training = wiring.contracts[-1]
    assert wiring.preprocessing.call_args.kwargs["workers"] == saved_workers
    if saved_workers == 2:
        preprocessing = training["objective"]["preprocessing"]
        assert preprocessing["protocol"] == "cosmos.sft-preprocessing.v3"
        assert preprocessing["workers"] == 2
        assert preprocessing["dispatch"] == "window_least_assigned_sticky_lru_v1"
        assert preprocessing["media_owner_capacity"] == 1024

    trainer = _trainer()
    model_config = instantiate(config.model.model_config)
    trainer._adapter_checkpoint_codec().identity = {
        "base": dict(model_config.source_identity),
        "adapter": native_adapter_config_identity(model_config.adapter),
        "quantization": quantization_config_identity(model_config.quantized_base),
        "processor_contract": processor,
        "training_contract": training,
    }
    checkpoint = SimpleNamespace(manager=trainer.checkpoint_manager, resume_checkpoint=str(_save(trainer, tmp_path)))
    wiring.backbone.reset_mock()
    with pytest.raises(_BeforeBackbone):
        wiring.invoke(config, checkpoint)
    wiring.backbone.assert_called_once()
    assert wiring.contracts[-1] == (processor, training)

    config.rl.preprocessing_workers = new_workers
    wiring.backbone.reset_mock()
    with pytest.raises(ValueError, match="training_contract.*preprocessing"):
        wiring.invoke(config, checkpoint)
    wiring.backbone.assert_not_called()
    assert wiring.preprocessing.call_args.kwargs["workers"] == new_workers
    assert wiring.contracts[-1][0] == processor
    assert wiring.contracts[-1][1]["objective"]["preprocessing"] != training["objective"]["preprocessing"]


def test_legacy_hash_routing_checkpoint_is_rejected_before_backbone(wiring, tmp_path):
    config = _config()
    config.rl.preprocessing_workers = 2
    with pytest.raises(_BeforeBackbone):
        wiring.invoke(config)
    processor, training = copy.deepcopy(wiring.contracts[-1])
    legacy = training["objective"]["preprocessing"]
    legacy.update(protocol="cosmos.sft-preprocessing.v2", dispatch="media_sha256_or_round_robin_v1")
    legacy.pop("media_owner_capacity")
    trainer = _trainer()
    model_config = instantiate(config.model.model_config)
    trainer._adapter_checkpoint_codec().identity = {
        "base": dict(model_config.source_identity),
        "adapter": native_adapter_config_identity(model_config.adapter),
        "quantization": quantization_config_identity(model_config.quantized_base),
        "processor_contract": processor,
        "training_contract": training,
    }
    checkpoint = SimpleNamespace(manager=trainer.checkpoint_manager, resume_checkpoint=str(_save(trainer, tmp_path)))
    wiring.backbone.reset_mock()
    with pytest.raises(ValueError, match="training_contract.*preprocessing"):
        wiring.invoke(config, checkpoint)
    wiring.backbone.assert_not_called()
