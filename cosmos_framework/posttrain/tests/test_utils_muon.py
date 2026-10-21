# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Posttrain wiring tests for the shared MuonWithAuxAdamW optimizer."""

from __future__ import annotations

import math
import os
import sys
import types
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn

from cosmos_framework.posttrain.config.build import (
    MUON_EXPERT_PARAM_KEYWORDS,
    MUON_ORTHOGONALIZE_SKIP_PATTERNS,
    build_optimizer,
    resolve_precision,
)
from cosmos_framework.posttrain.config.rl import OPTIMIZER_NAMES, OptimizerConfig


class _StackedExperts(nn.Module):
    """Native Qwen3-VL-MoE parameter layout: expert axis first."""

    def __init__(self, num_experts: int = 4, hidden_size: int = 6, intermediate_size: int = 3) -> None:
        super().__init__()
        self.gate_up_proj = nn.Parameter(  # [E,H,2I]
            torch.randn(num_experts, hidden_size, 2 * intermediate_size, dtype=torch.float32)
        )
        self.down_proj = nn.Parameter(  # [E,I,H]
            torch.randn(num_experts, intermediate_size, hidden_size, dtype=torch.float32)
        )
        self.unsupported_rank4 = nn.Parameter(  # [E,2,I,H]
            torch.randn(num_experts, 2, intermediate_size, hidden_size, dtype=torch.float32)
        )


class _RoutingModel(nn.Module):
    """Every parameter family that the posttrain Muon boundary must classify."""

    def __init__(self, *, num_experts: int = 4) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(16, 6)
        self.hidden = nn.Linear(6, 6)
        self.gate_proj = nn.Linear(6, 8, bias=False)
        self.up_proj = nn.Linear(6, 8, bias=False)
        self.down_proj = nn.Linear(8, 6, bias=False)
        self.gate_up_proj = nn.Linear(6, 16, bias=False)
        self.qkv_proj = nn.Linear(6, 18, bias=False)
        self.gate = nn.Linear(6, num_experts, bias=False)
        self.router = nn.Linear(6, num_experts, bias=False)
        self.norm = nn.LayerNorm(6)
        self.conv = nn.Conv2d(1, 2, kernel_size=3)
        self.lm_head = nn.Linear(6, 16, bias=False)
        self.experts = _StackedExperts(num_experts=num_experts)


@pytest.mark.L0
@pytest.mark.CPU
def test_shared_classifier_routes_only_supported_muon_parameters() -> None:
    from cosmos_framework.utils.generator.aux_optimizer_utils import split_orthogonalizable_params

    model = _RoutingModel()
    optimizer_param_ids = {id(parameter) for parameter in model.parameters()}

    muon_params, adam_params, names = split_orthogonalizable_params(
        model,
        optimizer_param_ids,
        expert_param_keywords=MUON_EXPERT_PARAM_KEYWORDS,
        orthogonalize_skip_patterns=MUON_ORTHOGONALIZE_SKIP_PATTERNS,
    )

    assert {names[parameter] for parameter in muon_params} == {
        "hidden.weight",
        "gate_proj.weight",
        "up_proj.weight",
        "down_proj.weight",
        "experts.gate_up_proj",
        "experts.down_proj",
    }
    assert {names[parameter] for parameter in adam_params} == {
        "embed_tokens.weight",
        "hidden.bias",
        "gate_up_proj.weight",
        "qkv_proj.weight",
        "gate.weight",
        "router.weight",
        "norm.weight",
        "norm.bias",
        "conv.weight",
        "conv.bias",
        "lm_head.weight",
        "experts.unsupported_rank4",
    }


@pytest.mark.L0
@pytest.mark.CPU
def test_muon_config_rejects_lossy_or_nonfinite_values() -> None:
    invalid = [
        ("muon_momentum", True),
        ("muon_momentum", float("nan")),
        ("muon_momentum", -0.1),
        ("muon_momentum", 1.0),
        ("muon_ns_steps", True),
        ("muon_ns_steps", 1.5),
        ("muon_ns_steps", 0),
        ("muon_rms_match", True),
        ("muon_rms_match", float("nan")),
        ("muon_rms_match", float("inf")),
        ("muon_rms_match", 0.0),
    ]
    for field, value in invalid:
        with pytest.raises(ValueError, match="muon_"):
            OptimizerConfig(name="muon", **{field: value})


@pytest.mark.L0
@pytest.mark.CPU
def test_config_and_ppo_validator_accept_muon_and_keep_other_optimizers_unchanged() -> None:
    from cosmos_framework.posttrain.config.config_ppo import _validate_optimizer

    assert "muon" in OPTIMIZER_NAMES
    cfg = OptimizerConfig(name="muon", lr=2e-6, eps=1e-6, grad_clip_norm=1.0)
    assert (cfg.muon_momentum, cfg.muon_ns_steps, cfg.muon_rms_match) == (0.95, 5, 0.2)
    _validate_optimizer(cfg, "policy")
    _validate_optimizer(OptimizerConfig(name="fusedadam"), "critic")
    with pytest.raises(ValueError, match="must be one of"):
        _validate_optimizer(SimpleNamespace(name="sgd", lr=1e-6, eps=1e-8, grad_clip_norm=0.0), "policy")
    with pytest.raises(ValueError, match="must be one of"):
        OptimizerConfig(name="sgd")
    assert OptimizerConfig(name="adamw", muon_momentum=5.0).muon_momentum == 5.0
    with pytest.raises(ValueError, match="applies only to"):
        OptimizerConfig(name="muon", impl="fused")


@pytest.mark.L0
@pytest.mark.CPU
def test_muon_requires_float32_storage_during_precision_preflight() -> None:
    rl = SimpleNamespace(
        optimizer=OptimizerConfig(name="muon"),
        critic_optimizer=OptimizerConfig(),
        param_dtype="bfloat16",
        master_dtype="",
        reduce_dtype="",
        transfer_dtype="",
        mixed_precision_mode="",
    )
    root = SimpleNamespace(
        rl=rl,
        model_backend="hf",
        model_config=SimpleNamespace(dtype="bfloat16", fsdp_reduce_dtype="float32"),
    )

    with pytest.raises(ValueError, match="requires float32 parameter storage"):
        resolve_precision(root)

    rl.master_dtype = "float32"
    precision = resolve_precision(root)
    assert precision.master == "float32" and precision.param == "bfloat16"


@pytest.mark.L0
@pytest.mark.CPU
def test_build_optimizer_reuses_shared_muon_and_carries_posttrain_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    module_name = "cosmos_framework.utils.generator.muon_with_aux_adamw"
    fake_module = types.ModuleType(module_name)

    class _FakeMuon:
        def __init__(self, params: Any, **kwargs: Any) -> None:
            captured["params"] = list(params)
            captured["kwargs"] = kwargs
            self.muon_params: list[nn.Parameter] = []
            self.stacked_muon_params: list[nn.Parameter] = []

        def categorize_params(self, model: nn.Module) -> None:
            captured["model"] = model
            self.muon_params = [model.hidden.weight]
            self.stacked_muon_params = [model.experts.gate_up_proj, model.experts.down_proj]

    fake_module.MuonWithAuxAdamW = _FakeMuon
    monkeypatch.setitem(sys.modules, module_name, fake_module)

    model = _RoutingModel()
    cfg = OptimizerConfig(
        name="muon",
        lr=3e-6,
        weight_decay=0.01,
        muon_momentum=0.9,
        muon_ns_steps=6,
        muon_rms_match=0.25,
    )

    optimizer = build_optimizer(model, cfg)

    assert isinstance(optimizer, _FakeMuon)
    assert captured["model"] is model
    assert {id(parameter) for parameter in captured["params"]} == {id(parameter) for parameter in model.parameters()}
    assert captured["kwargs"] == {
        "lr": 3e-6,
        "weight_decay": 0.01,
        "muon_momentum": 0.9,
        "muon_lr_scale": 0.25,
        "ns_steps": 6,
        "adam_betas": (0.9, 0.999),
        "eps": 1e-8,
        "use_distributed": True,
        "capturable": True,
        "expert_param_keywords": MUON_EXPERT_PARAM_KEYWORDS,
        "orthogonalize_skip_patterns": MUON_ORTHOGONALIZE_SKIP_PATTERNS,
        "split_expert_gate_up": True,
        "batch_split_expert_ns": True,
        "max_moe_expert_ns_matrices": 0,
    }


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    "kwargs",
    [
        {"muon_momentum": float("nan")},
        {"ns_steps": 1.5},
        {"muon_lr_scale": float("inf")},
        {"max_moe_expert_ns_matrices": 1.5},
    ],
)
def test_shared_muon_constructor_rejects_invalid_values_before_cuda_allocation(kwargs: dict[str, Any]) -> None:
    pytest.importorskip("transformer_engine")
    from cosmos_framework.utils.generator.muon_with_aux_adamw import MuonWithAuxAdamW

    parameter = nn.Parameter(torch.ones(2, 2, dtype=torch.float32))  # [O,I]
    with pytest.raises(ValueError):
        MuonWithAuxAdamW([parameter], use_distributed=False, capturable=False, **kwargs)


class _CheckpointModel(nn.Module):
    def __init__(self, num_experts: int) -> None:
        super().__init__()
        self.hidden = nn.Linear(6, 6)
        self.router = nn.Linear(6, num_experts, bias=False)
        self.norm = nn.LayerNorm(6)
        self.experts = _StackedExperts(num_experts=num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B,6], returns []
        hidden = self.norm(self.hidden(x))  # [B,6]
        routed = self.router(hidden)  # [B,E]
        expert_regularizer = (  # []
            self.experts.gate_up_proj.square().mean()
            + self.experts.down_proj.square().mean()
            + self.experts.unsupported_rank4.square().mean()
        )
        return hidden.square().mean() + routed.square().mean() + 0.01 * expert_regularizer  # []


@pytest.mark.L1
@pytest.mark.GPU
def test_muon_round_trips_through_llm_trainer_dcp_and_matches_the_next_step(tmp_path: Any) -> None:
    import torch.distributed as dist
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    from cosmos_framework.posttrain.algorithm.primitives.objective import Objective, Reduction
    from cosmos_framework.posttrain.roles.trainer import LLMTrainer

    if not dist.is_initialized():
        dist.init_process_group("nccl")
    world_size = dist.get_world_size()
    if world_size < 2:
        pytest.skip("needs at least two GPUs")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp_shard",))

    def make_trainer(seed: int) -> tuple[LLMTrainer, nn.Module]:
        torch.manual_seed(seed)
        model = _CheckpointModel(num_experts=2 * world_size).cuda()
        fully_shard(model, mesh=mesh)
        optimizer = build_optimizer(
            model,
            OptimizerConfig(name="muon", lr=1e-3, weight_decay=0.01),
        )

        def objective(net: nn.Module, batch: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
            loss = net(batch)  # []
            return loss, {"loss": float(loss.detach())}

        trainer = LLMTrainer(
            model,
            Objective(objective, reduction=Reduction.MEAN),
            optimizer,
            checkpoint_format="dcp",
        )
        return trainer, model

    trainer, model = make_trainer(seed=17)
    first_batch = torch.arange(24, device="cuda", dtype=torch.float32).reshape(4, 6) / 24.0  # [4,6]
    second_batch = torch.flip(first_batch, dims=(0,))  # [4,6]
    trainer.train([first_batch])

    checkpoint_holder = [str(tmp_path / "iter_000001")] if rank == 0 else [None]
    dist.broadcast_object_list(checkpoint_holder, src=0)
    checkpoint = checkpoint_holder[0]
    assert checkpoint is not None
    trainer.save(checkpoint)
    trainer.finalize_checkpoints()

    resumed, resumed_model = make_trainer(seed=99)
    assert resumed.load(checkpoint) == 1

    trainer.train([second_batch])
    resumed.train([second_batch])

    options = StateDictOptions(full_state_dict=True, cpu_offload=True)
    expected = get_model_state_dict(model, options=options)
    actual = get_model_state_dict(resumed_model, options=options)
    assert expected.keys() == actual.keys()
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], msg=lambda message: f"{name}: {message}")

    assert trainer.iteration() == resumed.iteration() == 2
    assert all(math.isfinite(float(value.float().norm())) for value in actual.values())
    assert os.path.exists(os.path.join(checkpoint, "optim", ".metadata"))
