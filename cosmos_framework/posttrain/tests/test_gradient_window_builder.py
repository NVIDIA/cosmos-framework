# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Public construction order with small CPU stand-ins for weights and FSDP.

These tests exercise the real native builder and SFT assembly, with explicitly
mocked expensive construction, media and distributed registration boundaries.
They do not qualify a model checkpoint or a distributed backend.
"""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import transformers

from cosmos_framework.posttrain.config.adapters import AdapterComponentConfig, AdapterConfig
from cosmos_framework.posttrain.config.rl import OptimizerConfig
from cosmos_framework.posttrain.config.sft import SFTConfig
from cosmos_framework.posttrain.model.pi05 import sft as pi05
from cosmos_framework.posttrain.roles.trainer.gradient_window import PROFILE

pytestmark = [
    pytest.mark.L0,
    pytest.mark.CPU,
    pytest.mark.skipif(transformers.__version__ != "4.57.6", reason="native Pi0.5 runtime binds Transformers 4.57.6"),
]


@pytest.fixture
def native(monkeypatch):
    from cosmos_framework.posttrain.model import adapters, parallelize
    from cosmos_framework.posttrain.model.pi05 import modeling, processing, source
    from cosmos_framework.posttrain.roles.trainer import gradient_window

    events = []

    class TinyNative(torch.nn.Module):
        def __init__(self, config, *, device):
            super().__init__()
            events.append("construct")
            self.weight = torch.nn.Parameter(torch.ones(2, device=device))
            self.base = torch.nn.Parameter(torch.zeros(2, device=device), requires_grad=False)

        def apply_reference_precision(self):
            events.append("reference_precision")

    def install_adapters(net, cfg, **kwargs):
        events.append(("adapters", kwargs["resume"]))
        net._posttrain_adapter_identity = {"fixture": "native-pi05"}

    def shard(net, plan, pdims):
        events.append(("shard", plan.reduce_dtype, plan.param_dtype, pdims.dp_shard, pdims.dp_replicate))
        net._fixture_sharded = True

    def trainability(net, optimizer=None):
        if optimizer is not None:
            assert [p for group in optimizer.param_groups for p in group["params"]] == [net.weight]
        events.append("trainability")

    def install(net, optimizer, *, microbatches):
        assert net._fixture_sharded
        assert optimizer.param_groups[0]["params"][0] is net.weight
        events.append(("policy", microbatches))
        return SimpleNamespace(fixture=True)

    monkeypatch.setattr(modeling, "Pi05Model", TinyNative)
    monkeypatch.setattr(pi05, "_model_config", lambda model_id, config: config)
    monkeypatch.setattr(source.Pi05Source, "resolve", lambda config: object())
    monkeypatch.setattr(source, "load_model_weights", lambda net, src: events.append("load"))
    monkeypatch.setattr(pi05, "source_identity", lambda net, src: {})
    monkeypatch.setattr(pi05, "validate_prefix_targets", lambda net, cfg: events.append("targets"))
    monkeypatch.setattr(processing, "Pi05Processor", lambda config: object())
    monkeypatch.setattr(adapters, "prepare_adapter_model", install_adapters)
    monkeypatch.setattr(adapters, "assert_trainability", trainability)
    monkeypatch.setattr(adapters, "validate_adapter_parallelism", lambda pdims: None)
    monkeypatch.setattr(parallelize, "parallelize", shard)
    monkeypatch.setattr(gradient_window, "install_fsdp", install)
    return events


def dimensions(shards=4, replicas=1):
    return SimpleNamespace(world_size=shards * replicas, dp_shard=shards, dp_replicate=replicas)


def rl_config(enabled=True):
    return SFTConfig(
        task="flow_bc",
        adapters=AdapterConfig(components=[AdapterComponentConfig(component="action", targets=["action_in_proj"])]),
        gradient_policy=PROFILE if enabled else None,
        optimizer=OptimizerConfig(grad_clip_norm=1),
        grad_accum=2,
    )


@pytest.mark.parametrize("layout", [(4, 1), (2, 2)])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("resume", [False, True])
def test_native_reduction_is_declared_before_registration_and_final_optimizer(native, layout, enabled, resume):
    rl = rl_config(enabled)
    net = pi05.build_pi05_sft_policy(
        "fixture",
        pi05.Pi05SFTModelConfig("fixture", "fixture", "none"),
        dimensions(*layout),
        adapters=rl.adapters,
        gradient_policy=rl.gradient_policy,
        adapter_resume=resume,
        device="cpu",
    )
    optimizer = torch.optim.AdamW([net.net.weight], lr=1e-3)
    native.append("optimizer")
    if enabled:
        assert pi05.build_gradient_window(net, optimizer, rl).fixture
        assert net.precision.reduce == "float64" and net.precision.param is None
    else:
        assert net.precision.reduce is None
    assert native[:5] == ["construct", "load", "reference_precision", "targets", ("adapters", resume)]
    assert native[5] == ("shard", torch.float64 if enabled else torch.float32, None, *layout)
    assert native[6:8] == ["trainability", "optimizer"]
    assert native[8:] == (["trainability", ("policy", 2)] if enabled else [])


@pytest.mark.parametrize("layout", [(1, 1), (2, 1), (1, 4), (8, 1)])
def test_builder_rejects_unsupported_topology_before_construction(native, layout):
    with pytest.raises(ValueError, match="FSDP4 or HSDP2x2"):
        pi05.build_pi05_sft_policy(
            "fixture",
            pi05.Pi05SFTModelConfig("fixture", "fixture", "none"),
            dimensions(*layout),
            adapters=rl_config().adapters,
            gradient_policy=PROFILE,
            device="cpu",
        )
    assert not native


def test_factory_refuses_a_default_reducer_and_another_policy_class(native):
    rl = rl_config()
    policy = pi05.build_pi05_sft_policy(
        "fixture",
        pi05.Pi05SFTModelConfig("fixture", "fixture", "none"),
        dimensions(),
        adapters=rl.adapters,
        device="cpu",
    )
    optimizer = torch.optim.AdamW([policy.net.weight], lr=1e-3)
    with pytest.raises(ValueError, match="registered FP64"):
        pi05.build_gradient_window(policy, optimizer, rl)
    foreign = SimpleNamespace(net=policy.net, precision=replace(policy.precision, reduce="float64"))
    with pytest.raises(ValueError, match="native Pi0.5"):
        pi05.build_gradient_window(foreign, optimizer, rl)
    assert not any(isinstance(e, tuple) and e[0] == "policy" for e in native)


@pytest.mark.parametrize("enabled", [False, True])
def test_public_sft_constructs_final_optimizer_then_policy_then_trainer(monkeypatch, enabled):
    from cosmos_framework.posttrain.algorithm.flow_bc import make_objective
    from cosmos_framework.posttrain.config import build
    from cosmos_framework.posttrain.entrypoints.sft import build_trainer
    from cosmos_framework.posttrain.model import adapters, sft_source
    from cosmos_framework.posttrain.model.pi05.policy import NATIVE_PRECISION
    from cosmos_framework.posttrain.roles.trainer import llm

    events, rl = [], rl_config(enabled)
    net = torch.nn.Linear(1, 1, bias=False)
    net._posttrain_adapter_identity = {"fixture": True}
    policy = SimpleNamespace(net=net, precision=replace(NATIVE_PRECISION, reduce="float64" if enabled else None))
    packer = SimpleNamespace(validation_semantics_identity=lambda: {"fixture": "media"})
    monkeypatch.setattr(sft_source, "build_flow_sft_packer", lambda spec, cfg: packer)
    monkeypatch.setattr(build, "select_policy_backend", lambda config: SimpleNamespace(native_spec=object()))
    monkeypatch.setattr(build, "resolve_precision", lambda config: NATIVE_PRECISION)
    monkeypatch.setattr("cosmos_framework.posttrain.entrypoints.sft.adapter_execution_identity", lambda config: {})

    def backbone(config, *, native_build_kwargs):
        assert ("gradient_policy" in native_build_kwargs) == enabled
        assert native_build_kwargs.get("gradient_policy") == rl.gradient_policy
        events.append("backbone")
        return policy

    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)

    def final_optimizer(model, config, *, parameter_groups):
        assert model is net and parameter_groups == "groups"
        events.append("optimizer")
        return optimizer

    window = object()

    def install(actual, opt, cfg):
        assert actual is policy and opt is optimizer and cfg is rl
        assert events[-1] == "trainability"
        events.append("policy")
        return window

    def trainer(actual, objective, opt, **kwargs):
        assert actual is policy and opt is optimizer
        assert ("gradient_window" in kwargs) == enabled
        if enabled:
            assert kwargs["gradient_window"] is window and objective.identity["precision"]["reduce"] == "float64"
        events.append("trainer")
        return SimpleNamespace(configure_validation=lambda identity: events.append("validation"))

    assert make_objective  # The entrypoint creates the actual native objective.
    monkeypatch.setattr(build, "build_backbone", backbone)
    monkeypatch.setattr(build, "build_optimizer", final_optimizer)
    monkeypatch.setattr(build, "build_sft_lr_scheduler", lambda *a: None)
    monkeypatch.setattr(adapters, "optimizer_groups", lambda *a: "groups")
    monkeypatch.setattr(adapters, "assert_trainability", lambda *a: events.append("trainability"))
    monkeypatch.setattr(pi05, "build_gradient_window", install)
    monkeypatch.setattr(llm, "LLMTrainer", trainer)
    build_trainer(
        SimpleNamespace(),
        rl,
        SimpleNamespace(source_fingerprint={"fixture": "data"}),
        None,
        SimpleNamespace(resume_checkpoint=None, manager=object()),
    )
    assert events == [
        "backbone",
        "optimizer",
        "trainability",
        *(["policy"] if enabled else []),
        "trainer",
        "validation",
    ]
