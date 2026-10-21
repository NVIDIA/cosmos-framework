# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""CPU contract tests for Cosmos3 online-NFT adapter details."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.posttrain.model import cosmos3 as cosmos3_module
from cosmos_framework.posttrain.model.cosmos3 import (
    _COSMOS3_RMSNORM_TYPES,
    _REFERENCE_SEMANTIC_CONFIG_FIELDS,
    Cosmos3FlowPolicy,
    _Cosmos3FlowCondition,
    _enable_fp32_rmsnorms,
    _fp32_rmsnorm_forward,
    _rebatch_uniform,
)
from cosmos_framework.posttrain.tests.helpers.distributed import init_gloo_process_group


def _flow_condition(*, is_image: bool = False) -> _Cosmos3FlowCondition:
    return _Cosmos3FlowCondition(
        gen_data_clean=SimpleNamespace(fps_vision=None),
        sequence_plans=[],
        text_tokens=[],
        is_image_batch=is_image,
        resolutions=None,
    )


class _RecognizedRMSNorm(torch.nn.Module):
    variance_epsilon: float
    weight: torch.nn.Parameter

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.25, -0.75, 0.5], dtype=torch.bfloat16))  # [D]
        self.variance_epsilon = 1e-6

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:  # [B,D] -> [B,D]
        return hidden_states  # [B,D]


# Give the test double one exact allowlisted runtime identity without importing the heavy model package.
_RecognizedRMSNorm.__name__ = "Qwen3VLTextRMSNorm"
_RecognizedRMSNorm.__module__ = "cosmos_framework.model.generator.reasoner.qwen3_vl.qwen3_vl"


def _recognized_rmsnorm() -> _RecognizedRMSNorm:
    return _RecognizedRMSNorm()


@pytest.mark.L0
@pytest.mark.CPU
def test_fp32_rmsnorm_registry_matches_real_cosmos3_types() -> None:
    from cosmos_framework.model.generator.reasoner.nemotron_3_dense_vl.nemotron_3_dense_vl import (
        Nemotron3DenseVLRMSNorm,
    )
    from cosmos_framework.model.generator.reasoner.qwen3_vl.qwen3_vl import Qwen3VLTextRMSNorm

    expected_types = {
        f"{module_type.__module__}.{module_type.__name__}"
        for module_type in (Nemotron3DenseVLRMSNorm, Qwen3VLTextRMSNorm)
    }
    # Importing the MoE module loads Triton kernels, which the required CPU gate deliberately omits.
    # Pin its real source-qualified identity here. Exact equality makes a future LayerTypes variant fail closed.
    expected_types.add("cosmos_framework.model.generator.reasoner.qwen3_vl_moe.qwen3_vl_moe.Qwen3VLMoeTextRMSNorm")
    assert _COSMOS3_RMSNORM_TYPES == expected_types


def _fp32_rmsnorm_dtensor_worker(
    rank: int,
    world_size: int,
    rendezvous_path: str,
    sharded: bool,
) -> None:
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Replicate, Shard, distribute_tensor

    init_gloo_process_group(rank, world_size, rendezvous_path)
    try:
        mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("dp",))
        norm = _recognized_rmsnorm()
        full_weight = torch.tensor([1.25, -0.75, 0.5, 2.0], dtype=torch.bfloat16)  # [D]
        placement = Shard(0) if sharded else Replicate()
        norm.weight = torch.nn.Parameter(distribute_tensor(full_weight, mesh, [placement]))  # DTensor [D]
        _enable_fp32_rmsnorms(norm)
        hidden_states = torch.tensor([[2.0, -3.0, 4.0, 1.0]], dtype=torch.bfloat16)  # [B,D]
        if sharded:
            with pytest.raises(RuntimeError, match="DTensor weight must be replicated.*Shard"):
                norm(hidden_states)
        else:
            variance = hidden_states.float().pow(2).mean(-1, keepdim=True)  # [B,1]
            normalized = hidden_states.float() * torch.rsqrt(variance + norm.variance_epsilon)  # [B,D]
            expected = (full_weight.float() * normalized).to(torch.bfloat16)  # [B,D]
            torch.testing.assert_close(norm(hidden_states), expected, rtol=0.0, atol=0.0)
    finally:
        dist.destroy_process_group()


@pytest.mark.L0
@pytest.mark.CPU
def test_fp32_rmsnorm_accepts_replicated_dtensor_weight(tmp_path: Path) -> None:
    import torch.multiprocessing as mp

    rendezvous_path = str(tmp_path / "replicated-rmsnorm-rendezvous")
    mp.spawn(
        _fp32_rmsnorm_dtensor_worker,
        args=(2, rendezvous_path, False),
        nprocs=2,
        join=True,
    )


@pytest.mark.L0
@pytest.mark.CPU
def test_fp32_rmsnorm_rejects_sharded_dtensor_weight(tmp_path: Path) -> None:
    import torch.multiprocessing as mp

    rendezvous_path = str(tmp_path / "sharded-rmsnorm-rendezvous")
    mp.spawn(
        _fp32_rmsnorm_dtensor_worker,
        args=(2, rendezvous_path, True),
        nprocs=2,
        join=True,
    )


@pytest.mark.L0
@pytest.mark.CPU
def test_fp32_rmsnorm_override_matches_oracle_and_is_checkpoint_neutral() -> None:
    norm = _recognized_rmsnorm()
    root = torch.nn.Sequential(norm, torch.nn.LayerNorm(3, dtype=torch.bfloat16))
    state_keys = tuple(root.state_dict())

    expected_inventory = {"0": f"{type(norm).__module__}.{type(norm).__name__}"}
    assert _enable_fp32_rmsnorms(root) == expected_inventory
    assert _enable_fp32_rmsnorms(root) == expected_inventory
    assert getattr(norm.forward, "__func__", None) is _fp32_rmsnorm_forward
    assert tuple(root.state_dict()) == state_keys

    actual_input = torch.tensor([[2.0, -3.0, 4.0]], dtype=torch.bfloat16, requires_grad=True)  # [B,D]
    oracle_input = actual_input.detach().clone().requires_grad_(True)  # [B,D]
    oracle_weight = norm.weight.detach().clone().requires_grad_(True)  # [D]
    variance = oracle_input.float().pow(2).mean(-1, keepdim=True)  # [B,1]
    normalized = oracle_input.float() * torch.rsqrt(variance + norm.variance_epsilon)  # [B,D]
    expected = (oracle_weight.float() * normalized).to(torch.bfloat16)  # [B,D]

    actual = norm(actual_input)  # [B,D]
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    upstream = torch.tensor([[0.5, -1.0, 1.5]], dtype=torch.bfloat16)  # [B,D]
    actual.backward(upstream)
    expected.backward(upstream)
    torch.testing.assert_close(actual_input.grad, oracle_input.grad, rtol=0.0, atol=0.0)
    torch.testing.assert_close(norm.weight.grad, oracle_weight.grad, rtol=0.0, atol=0.0)

    restored = torch.nn.Sequential(_recognized_rmsnorm(), torch.nn.LayerNorm(3, dtype=torch.bfloat16))
    assert _enable_fp32_rmsnorms(restored) == expected_inventory
    restored.load_state_dict(root.state_dict())
    torch.testing.assert_close(restored(actual_input.detach()), root(actual_input.detach()))


@pytest.mark.L0
@pytest.mark.CPU
def test_fp32_rmsnorm_contract_covers_policy_reference_and_ema() -> None:
    class _Omni:
        def __init__(self) -> None:
            self.precision: torch.dtype = torch.bfloat16
            self.net = torch.nn.Sequential(_recognized_rmsnorm())
            self.net_ema = deepcopy(self.net)
            self.net_ema_worker = SimpleNamespace(update_average=lambda *_args, **_kwargs: None)
            self.ema_beta = lambda _iteration: 0.999
            self.config = SimpleNamespace(ema=SimpleNamespace(enabled=True))

    policy = Cosmos3FlowPolicy(_Omni(), build_ref=True, build_old=True)

    example_norm = _recognized_rmsnorm()
    expected_inventory = {"0": f"{type(example_norm).__module__}.{type(example_norm).__name__}"}
    assert policy._fp32_rmsnorm_inventories == {
        "net": expected_inventory,
        "net_old": expected_inventory,
        "net_ref": expected_inventory,
        "net_ema": expected_inventory,
    }
    assert policy.net_old is not None
    assert policy.net_ref is not None
    for network in (policy.net, policy.net_old, policy.net_ref, policy.net_ema):
        assert network is not None
        norm = network[0]
        assert getattr(norm.forward, "__func__", None) is _fp32_rmsnorm_forward
    assert not policy.net_ref.training
    assert not policy.net_old.training
    assert all(not parameter.requires_grad for parameter in policy.net_old.parameters())
    assert all(not parameter.requires_grad for parameter in policy.net_ref.parameters())


@pytest.mark.L0
@pytest.mark.CPU
def test_cosmos3_policy_keeps_moving_old_and_frozen_reference_independent() -> None:
    class _StatefulNet(torch.nn.Module):
        linear: torch.nn.Linear
        running_scale: torch.Tensor
        runtime_cache: torch.Tensor

        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(2, 2)
            self.register_buffer("running_scale", torch.tensor([2.0, 3.0]))  # [D]
            self.register_buffer("runtime_cache", torch.tensor([-1.0, -2.0]), persistent=False)  # [D]

    class _Omni:
        def __init__(self) -> None:
            self.net = _StatefulNet()
            self.precision = torch.float32
            self.config = SimpleNamespace(ema=SimpleNamespace(enabled=False))

    policy = Cosmos3FlowPolicy(_Omni(), build_ref=True, build_old=True)
    assert policy.net_old is not None
    assert policy.net_ref is not None
    frozen_ref = deepcopy(policy.net_ref.state_dict())
    old_runtime_cache = policy.net_old.runtime_cache.detach().clone()  # [D]
    ref_runtime_cache = policy.net_ref.runtime_cache.detach().clone()  # [D]

    with torch.no_grad():
        for parameter in policy.net.parameters():
            parameter.add_(1.0)
        policy.net.running_scale.add_(4.0)  # [D]
        policy.net.runtime_cache.add_(8.0)  # [D]
    policy.refresh_old()

    for name, value in policy.net.state_dict().items():
        torch.testing.assert_close(policy.net_old.state_dict()[name], value)
        torch.testing.assert_close(policy.net_ref.state_dict()[name], frozen_ref[name])
    torch.testing.assert_close(policy.net_old.runtime_cache, old_runtime_cache)
    torch.testing.assert_close(policy.net_ref.runtime_cache, ref_runtime_cache)


@pytest.mark.L0
@pytest.mark.CPU
def test_cosmos3_old_refresh_casts_fp32_master_state_to_bf16_snapshot() -> None:
    policy_net = torch.nn.Module()
    policy_net.register_parameter(
        "weight",
        torch.nn.Parameter(torch.tensor([1.125, -2.25], dtype=torch.float32)),  # [D]
    )
    policy_net.register_buffer("running", torch.tensor([3.5], dtype=torch.float32))  # [1]
    old_net = torch.nn.Module()
    old_net.register_parameter(
        "weight",
        torch.nn.Parameter(torch.zeros(2, dtype=torch.bfloat16)),  # [D]
    )
    old_net.register_buffer("running", torch.zeros(1, dtype=torch.bfloat16))  # [1]
    old_net.requires_grad_(False)
    policy = Cosmos3FlowPolicy.__new__(Cosmos3FlowPolicy)
    policy.net = policy_net
    policy.net_old = old_net

    policy.refresh_old()

    assert old_net.weight.dtype is torch.bfloat16
    assert old_net.running.dtype is torch.bfloat16
    assert torch.equal(old_net.weight, policy_net.weight.to(torch.bfloat16))
    assert torch.equal(old_net.running, policy_net.running.to(torch.bfloat16))

    integer_policy = torch.nn.Module()
    integer_policy.register_buffer("count", torch.tensor([7], dtype=torch.int64))  # [1]
    integer_old = torch.nn.Module()
    integer_old.register_buffer("count", torch.tensor([0], dtype=torch.int32))  # [1]
    policy.net = integer_policy
    policy.net_old = integer_old
    with pytest.raises(RuntimeError, match="non-floating dtype differs for buffer count"):
        policy.refresh_old()


@pytest.mark.L0
@pytest.mark.CPU
def test_cosmos3_dpo_style_construction_builds_only_frozen_reference() -> None:
    class _Omni:
        def __init__(self) -> None:
            self.net = torch.nn.Linear(2, 2)
            self.net_ema = deepcopy(self.net)
            self.precision = torch.float32
            self.config = SimpleNamespace(ema=SimpleNamespace(enabled=False))

    policy = Cosmos3FlowPolicy(_Omni(), build_ref=True)

    assert policy.net_old is None
    assert policy.net_ref is not None
    assert policy.net_ema is None
    assert not policy.net_ref.training
    assert all(not parameter.requires_grad for parameter in policy.net_ref.parameters())


@pytest.mark.L0
@pytest.mark.CPU
def test_cosmos3_reference_identity_binds_fp32_rmsnorm_contract() -> None:
    class _Omni:
        def __init__(self) -> None:
            self.net = torch.nn.Sequential(_recognized_rmsnorm())
            self.precision = torch.bfloat16
            config_values = {field: None for field in _REFERENCE_SEMANTIC_CONFIG_FIELDS}
            self.config = SimpleNamespace(**config_values, ema=SimpleNamespace(enabled=False))

    policy = Cosmos3FlowPolicy(_Omni(), build_ref=True)
    fp32_identity = policy.reference_semantics_identity()
    policy._fp32_rmsnorm_inventories = {}
    native_identity = policy.reference_semantics_identity()

    assert fp32_identity["contract_version"] == 2
    assert fp32_identity["rmsnorm_contract"] == "recognized-fp32-v1"
    assert native_identity["rmsnorm_contract"] == "native-v1"
    assert fp32_identity["rmsnorm_inventory_sha256"] != native_identity["rmsnorm_inventory_sha256"]


@pytest.mark.L0
@pytest.mark.CPU
def test_cosmos3_policy_exposes_only_configured_ema() -> None:
    class _Omni:
        def __init__(self, *, enabled: bool, with_network: bool = True, with_updater: bool = True) -> None:
            self.net = torch.nn.Linear(2, 2)
            self.net_ema = deepcopy(self.net) if with_network else None
            self.config = SimpleNamespace(ema=SimpleNamespace(enabled=enabled))
            if with_updater:
                self.net_ema_worker = SimpleNamespace(update_average=lambda *_args, **_kwargs: None)
                self.ema_beta = lambda _iteration: 0.999

    disabled = _Omni(enabled=False)
    assert Cosmos3FlowPolicy(disabled, build_ref=False).net_ema is None

    enabled = _Omni(enabled=True)
    assert Cosmos3FlowPolicy(enabled, build_ref=False).net_ema is enabled.net_ema

    with pytest.raises(RuntimeError, match="has no net_ema"):
        Cosmos3FlowPolicy(_Omni(enabled=True, with_network=False), build_ref=False)
    with pytest.raises(RuntimeError, match="updater is unavailable"):
        Cosmos3FlowPolicy(_Omni(enabled=True, with_updater=False), build_ref=False)


@pytest.mark.L0
@pytest.mark.CPU
def test_bf16_cosmos3_policy_rejects_missing_rmsnorm_contract() -> None:
    class _Omni:
        def __init__(self) -> None:
            self.precision: torch.dtype = torch.bfloat16
            self.net = torch.nn.Linear(2, 2, dtype=torch.bfloat16)
            self.net_ema = None

    with pytest.raises(RuntimeError, match="found no recognized RMSNorm"):
        Cosmos3FlowPolicy(_Omni(), build_ref=False)


@pytest.mark.L0
@pytest.mark.CPU
def test_bf16_cosmos3_policy_rejects_partially_supported_rmsnorm_inventory() -> None:
    class FutureRMSNorm(torch.nn.Module):
        variance_epsilon: float = 1e-6
        weight: torch.nn.Parameter

        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(3, dtype=torch.bfloat16))  # [D]

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:  # [B,D] -> [B,D]
            return hidden_states  # [B,D]

    root = torch.nn.Sequential(_recognized_rmsnorm(), FutureRMSNorm())
    with pytest.raises(RuntimeError, match="unsupported RMSNorm.*FutureRMSNorm"):
        _enable_fp32_rmsnorms(root)


@pytest.mark.L0
@pytest.mark.CPU
def test_nft_fresh_rng_draws_epsilon_before_sigma() -> None:
    class _Omni:
        def _get_train_noise_level_vision(self, *, batch_size: int, **_kwargs: object) -> tuple[None, torch.Tensor]:
            return None, torch.rand(batch_size, 1)

    policy = Cosmos3FlowPolicy.__new__(Cosmos3FlowPolicy)
    policy.omnimot = _Omni()
    policy.net = torch.nn.Linear(2, 2)
    shape = torch.Size((2, 1, 1, 1, 2))

    torch.manual_seed(17)
    expected_epsilon = torch.randn(shape)
    expected_sigma = torch.rand(2, 1)
    torch.manual_seed(17)
    sigma, epsilon = policy.draw_nft_sigma_epsilon(shape, _flow_condition())

    torch.testing.assert_close(epsilon, expected_epsilon)
    torch.testing.assert_close(sigma, expected_sigma)


@pytest.mark.L0
@pytest.mark.CPU
def test_cosmos3_online_adapter_rebatches_uniform_predictions() -> None:
    samples = [torch.zeros(1, 2, 3), torch.zeros(1, 2, 3)]
    predictions = [torch.ones(1, 1, 2, 3), 2.0 * torch.ones(1, 1, 2, 3)]

    actual = _rebatch_uniform(predictions, samples)

    assert actual.shape == (2, 2, 3)
    torch.testing.assert_close(actual[0], torch.ones(2, 3))
    torch.testing.assert_close(actual[1], 2.0 * torch.ones(2, 3))


@pytest.mark.L0
@pytest.mark.CPU
def test_cosmos3_online_adapter_returns_independent_velocity_roles(monkeypatch: pytest.MonkeyPatch) -> None:
    forward_calls: list[tuple[float, bool]] = []
    convergence_contexts: list[str] = []

    def converge(action: Callable[[], object], context: str) -> object:
        convergence_contexts.append(context)
        return action()

    monkeypatch.setattr(cosmos3_module, "run_rank_converged", converge)

    class _Packed:
        def __init__(self) -> None:
            self.vision = SimpleNamespace(tokens=None)

        def to_cuda(self) -> None:
            pass

    class _RectifiedFlow:
        noise_scheduler = SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000))

        @staticmethod
        def get_interpolation(
            epsilon_items: list[torch.Tensor],
            x0_items: list[torch.Tensor],
            effective_t_items: list[torch.Tensor],
        ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
            x_t_items: list[torch.Tensor] = []
            targets: list[torch.Tensor] = []
            for epsilon, clean, effective_t in zip(epsilon_items, x0_items, effective_t_items, strict=True):
                fraction = effective_t.unsqueeze(0).unsqueeze(0)  # [1,1,T,1,1]
                x_t_items.append((1.0 - fraction) * clean + fraction * epsilon)  # [1,C,T,H,W]
                targets.append(clean - epsilon)  # [1,C,T,H,W]
            return x_t_items, targets

    class _Scale(torch.nn.Module):
        def __init__(self, value: float) -> None:
            super().__init__()
            self.value = torch.nn.Parameter(torch.tensor(value))  # []

        def forward(self, tokens: torch.Tensor) -> torch.Tensor:  # [1,C,T,H,W] -> [1,C,T,H,W]
            return tokens * self.value  # [1,C,T,H,W]

    class _Omni:
        rectified_flow_video = _RectifiedFlow()
        rectified_flow_image = _RectifiedFlow()
        tensor_kwargs: dict[str, object] = {"device": torch.device("cpu"), "dtype": torch.float32}

        @staticmethod
        def denoise(*, net: torch.nn.Module, data_batch_packed: _Packed) -> dict[str, list[torch.Tensor]]:
            assert isinstance(data_batch_packed.vision.tokens, list)
            assert isinstance(net, _Scale)
            forward_calls.append((float(net.value.item()), torch.is_grad_enabled()))
            return {"preds_vision": [net(tokens) for tokens in data_batch_packed.vision.tokens]}

    policy = Cosmos3FlowPolicy.__new__(Cosmos3FlowPolicy)
    policy.omnimot = _Omni()
    policy.net = _Scale(2.0)
    policy.net_old = _Scale(3.0)
    policy.net_old.requires_grad_(False)
    policy.net_ref = _Scale(4.0)
    policy.net_ref.requires_grad_(False)
    policy._pack = lambda _cond, _timesteps: _Packed()
    x0 = torch.arange(4, dtype=torch.float32).reshape(2, 1, 2, 1, 1)  # [B,C,T,H,W]
    epsilon = torch.full_like(x0, 10.0)  # [B,C,T,H,W]
    sigma = torch.tensor([[0.25], [0.75]])  # [B,1]
    condition_mask = [torch.tensor([[[1.0]], [[0.0]]]), torch.tensor([[[1.0]], [[0.0]]])]  # B * [T,1,1]

    predictions = policy.nft_velocity_predictions(
        x0=x0,
        cond=_flow_condition(),
        sigma=sigma,
        epsilon=epsilon,
        condition_mask=condition_mask,
    )

    assert predictions.x_t.shape == x0.shape
    assert predictions.t_effective.shape == (2, 1, 2, 1, 1)
    torch.testing.assert_close(predictions.t_effective[:, :, 0], torch.zeros(2, 1, 1, 1))
    torch.testing.assert_close(predictions.x_t[:, :, 0], x0[:, :, 0])
    torch.testing.assert_close(predictions.v_policy, 2.0 * predictions.x_t)
    torch.testing.assert_close(predictions.v_old, 3.0 * predictions.x_t)
    assert predictions.v_ref is not None
    torch.testing.assert_close(predictions.v_ref, 4.0 * predictions.x_t)
    assert forward_calls == [(2.0, True), (3.0, False), (4.0, False)]
    assert predictions.v_policy.requires_grad
    assert not predictions.v_old.requires_grad
    assert not predictions.v_ref.requires_grad
    predictions.v_policy.sum().backward()
    assert policy.net.value.grad is not None
    assert convergence_contexts == ["Cosmos3 NFT forward preparation", "Cosmos3 NFT forward finalization"]

    policy.net_ref = None
    predictions_without_ref = policy.nft_velocity_predictions(
        x0=x0,
        cond=_flow_condition(),
        sigma=sigma,
        epsilon=epsilon,
        condition_mask=condition_mask,
    )
    assert predictions_without_ref.v_ref is None
    assert convergence_contexts == [
        "Cosmos3 NFT forward preparation",
        "Cosmos3 NFT forward finalization",
        "Cosmos3 NFT forward preparation",
        "Cosmos3 NFT forward finalization",
    ]


def _asymmetric_nft_velocity_worker(rank: int, world_size: int, rendezvous_path: str) -> None:
    import torch.distributed as dist

    init_gloo_process_group(rank, world_size, rendezvous_path)
    try:

        class _Packed:
            def __init__(self) -> None:
                self.vision = SimpleNamespace(tokens=None)

            def to_cuda(self) -> None:
                pass

        class _RectifiedFlow:
            noise_scheduler = SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000))

            @staticmethod
            def get_interpolation(
                epsilon_items: list[torch.Tensor],
                x0_items: list[torch.Tensor],
                effective_t_items: list[torch.Tensor],
            ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
                del effective_t_items
                return x0_items, [clean - epsilon for clean, epsilon in zip(x0_items, epsilon_items, strict=True)]

        class _TaggedIdentity(torch.nn.Module):
            def __init__(self, role: str) -> None:
                super().__init__()
                self.role = role

            def forward(self, tokens: torch.Tensor) -> torch.Tensor:  # [1,C,T,H,W] -> [1,C,T,H,W]
                return tokens  # [1,C,T,H,W]

        class _Omni:
            rectified_flow_video = _RectifiedFlow()
            tensor_kwargs: dict[str, object] = {"device": torch.device("cpu"), "dtype": torch.float32}

            def __init__(self) -> None:
                self.forward_roles: list[str] = []
                self.invalidate_old_output = False

            def denoise(self, *, net: torch.nn.Module, data_batch_packed: _Packed) -> object:
                assert isinstance(net, _TaggedIdentity)
                assert isinstance(data_batch_packed.vision.tokens, list)
                self.forward_roles.append(net.role)
                if self.invalidate_old_output and rank == 1 and net.role == "old":
                    return {"preds_vision": "rank-local invalid container"}
                return {"preds_vision": [net(tokens) for tokens in data_batch_packed.vision.tokens]}

        omni = _Omni()
        policy = Cosmos3FlowPolicy.__new__(Cosmos3FlowPolicy)
        policy.omnimot = omni
        policy.net = _TaggedIdentity("policy")
        policy.net_old = _TaggedIdentity("old")
        policy.net_ref = _TaggedIdentity("reference")
        x0 = torch.ones(1, 1, 2, 1, 1)  # [B,C,T,H,W]
        sigma = torch.full((1, 1), 0.5)  # [B,1]
        epsilon = torch.zeros_like(x0)  # [B,C,T,H,W]

        def asymmetric_pack(_cond: object, _timesteps: torch.Tensor) -> _Packed:
            if rank == 1:
                raise ValueError("rank-local pack failure")
            return _Packed()

        policy._pack = asymmetric_pack
        with pytest.raises(RuntimeError, match="NFT forward preparation.*rank 1.*rank-local pack failure"):
            policy.nft_velocity_predictions(
                x0=x0,
                cond=_flow_condition(),
                sigma=sigma,
                epsilon=epsilon,
                condition_mask=None,
            )
        assert omni.forward_roles == []

        policy._pack = lambda _cond, _timesteps: _Packed()
        omni.invalidate_old_output = True
        with pytest.raises(RuntimeError, match="NFT forward finalization"):
            policy.nft_velocity_predictions(
                x0=x0,
                cond=_flow_condition(),
                sigma=sigma,
                epsilon=epsilon,
                condition_mask=None,
            )
        assert omni.forward_roles == ["policy", "old", "reference"]
    finally:
        dist.destroy_process_group()


@pytest.mark.L0
@pytest.mark.CPU
def test_cosmos3_online_adapter_converges_asymmetric_nft_stages(tmp_path: Path) -> None:
    import torch.multiprocessing as mp

    world_size = 2
    rendezvous_path = str(tmp_path / "asymmetric-nft-velocity-rendezvous")
    mp.spawn(_asymmetric_nft_velocity_worker, args=(world_size, rendezvous_path), nprocs=world_size, join=True)


@pytest.mark.L0
@pytest.mark.CPU
def test_cosmos3_online_adapter_rejects_partial_reference_refresh() -> None:
    policy_net = torch.nn.Module()
    policy_net.register_parameter("weight", torch.nn.Parameter(torch.full((2,), 7.0)))  # [D]
    policy_net.register_buffer("running", torch.full((2,), 8.0))  # [D]
    reference_net = torch.nn.Module()
    reference_net.register_parameter("weight", torch.nn.Parameter(torch.full((2,), -1.0)))  # [D]
    reference_net.register_buffer("running", torch.full((3,), -2.0))  # [D_other]
    original_weight = reference_net.weight.detach().clone()  # [D]
    original_running = reference_net.running.detach().clone()  # [D_other]
    policy = Cosmos3FlowPolicy.__new__(Cosmos3FlowPolicy)
    policy.net = policy_net
    policy.net_ref = reference_net

    with pytest.raises(RuntimeError, match="global shape differs for buffer running"):
        policy.refresh_ref()
    torch.testing.assert_close(reference_net.weight, original_weight)
    torch.testing.assert_close(reference_net.running, original_running)


def _mixed_reference_refresh_worker(rank: int, world_size: int, rendezvous_path: str) -> None:
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor, Shard, distribute_tensor

    init_gloo_process_group(rank, world_size, rendezvous_path)
    try:
        mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("dp",))
        expected_plain_to_shard = torch.arange(15, dtype=torch.float32).reshape(5, 3)  # [M,N]
        expected_shard_to_plain = expected_plain_to_shard + 100.0  # [M,N]
        expected_reshard = expected_plain_to_shard + 200.0  # [M,N]
        expected_same_layout = expected_plain_to_shard + 300.0  # [M,N]
        expected_plain = expected_plain_to_shard + 400.0  # [M,N]
        expected_buffer_plain_to_shard = expected_plain_to_shard + 500.0  # [M,N]
        expected_buffer_shard_to_plain = expected_plain_to_shard + 600.0  # [M,N]
        expected_buffer_reshard = expected_plain_to_shard + 700.0  # [M,N]
        policy_runtime_cache = torch.full_like(expected_plain_to_shard, 8.0)  # [M,N]
        reference_runtime_cache = torch.full_like(expected_plain_to_shard, -8.0)  # [M,N]

        policy_net = torch.nn.Module()
        policy_net.register_parameter("plain_to_shard", torch.nn.Parameter(expected_plain_to_shard.clone()))
        policy_net.register_parameter(
            "shard_to_plain",
            torch.nn.Parameter(distribute_tensor(expected_shard_to_plain.clone(), mesh, [Shard(0)])),
        )
        policy_net.register_parameter(
            "reshard",
            torch.nn.Parameter(distribute_tensor(expected_reshard.clone(), mesh, [Shard(0)])),
        )
        policy_net.register_parameter(
            "same_layout",
            torch.nn.Parameter(distribute_tensor(expected_same_layout.clone(), mesh, [Shard(0)])),
        )
        policy_net.register_parameter("plain", torch.nn.Parameter(expected_plain.clone()))
        policy_net.register_buffer("buffer_plain_to_shard", expected_buffer_plain_to_shard.clone())  # [M,N]
        policy_net.register_buffer(
            "buffer_shard_to_plain",
            distribute_tensor(expected_buffer_shard_to_plain.clone(), mesh, [Shard(0)]),  # DTensor [M,N]
        )
        policy_net.register_buffer(
            "buffer_reshard",
            distribute_tensor(expected_buffer_reshard.clone(), mesh, [Shard(0)]),  # DTensor [M,N]
        )
        policy_net.register_buffer("runtime_cache", policy_runtime_cache, persistent=False)  # [M,N]

        reference_net = torch.nn.Module()
        reference_net.register_parameter(
            "plain_to_shard",
            torch.nn.Parameter(distribute_tensor(torch.zeros_like(expected_plain_to_shard), mesh, [Shard(0)])),
        )
        reference_net.register_parameter(
            "shard_to_plain",
            torch.nn.Parameter(torch.zeros_like(expected_shard_to_plain)),
        )
        reference_net.register_parameter(
            "reshard",
            torch.nn.Parameter(distribute_tensor(torch.zeros_like(expected_reshard), mesh, [Shard(1)])),
        )
        reference_net.register_parameter(
            "same_layout",
            torch.nn.Parameter(distribute_tensor(torch.zeros_like(expected_same_layout), mesh, [Shard(0)])),
        )
        reference_net.register_parameter("plain", torch.nn.Parameter(torch.zeros_like(expected_plain)))
        reference_net.register_buffer(
            "buffer_plain_to_shard",
            distribute_tensor(torch.zeros_like(expected_buffer_plain_to_shard), mesh, [Shard(0)]),  # DTensor [M,N]
        )
        reference_net.register_buffer(
            "buffer_shard_to_plain",
            torch.zeros_like(expected_buffer_shard_to_plain),
        )
        reference_net.register_buffer(
            "buffer_reshard",
            distribute_tensor(torch.zeros_like(expected_buffer_reshard), mesh, [Shard(1)]),  # DTensor [M,N]
        )
        reference_net.register_buffer("runtime_cache", reference_runtime_cache, persistent=False)  # [M,N]

        class _Omni:
            def __init__(self) -> None:
                self.net: torch.nn.Module = policy_net
                self.precision: torch.dtype = torch.float32

            def build_net(self, dtype: torch.dtype) -> torch.nn.Module:
                assert dtype == self.precision
                return reference_net

        # Construction and periodic refresh must share the same validated copy
        # primitive. The former zip-based initial copy fails this mixed layout.
        policy = Cosmos3FlowPolicy(_Omni(), build_ref=True)

        def assert_reference_values() -> None:
            torch.testing.assert_close(reference_net.plain_to_shard.full_tensor(), expected_plain_to_shard)
            torch.testing.assert_close(reference_net.shard_to_plain, expected_shard_to_plain)
            torch.testing.assert_close(reference_net.reshard.full_tensor(), expected_reshard)
            torch.testing.assert_close(reference_net.same_layout.full_tensor(), expected_same_layout)
            torch.testing.assert_close(reference_net.plain, expected_plain)
            torch.testing.assert_close(
                reference_net.buffer_plain_to_shard.full_tensor(), expected_buffer_plain_to_shard
            )
            torch.testing.assert_close(reference_net.buffer_shard_to_plain, expected_buffer_shard_to_plain)
            torch.testing.assert_close(reference_net.buffer_reshard.full_tensor(), expected_buffer_reshard)
            torch.testing.assert_close(reference_net.runtime_cache, reference_runtime_cache)

        assert_reference_values()
        assert not policy.net_ref.training
        assert all(not parameter.requires_grad for parameter in policy.net_ref.parameters())
        with torch.no_grad():
            for parameter in policy.net_ref.parameters():
                local = parameter.to_local() if isinstance(parameter, DTensor) else parameter  # [*local or global]
                local.zero_()  # [*local or global]
            for name, buffer in policy.net_ref.named_buffers():
                if name == "runtime_cache":
                    continue
                local = buffer.to_local() if isinstance(buffer, DTensor) else buffer  # [*local or global]
                local.zero_()  # [*local or global]
            policy.net.runtime_cache.fill_(9.0)  # [M,N]
        policy.refresh_ref()
        assert_reference_values()

        mismatched_policy = torch.nn.Module()
        mismatched_policy.register_parameter("weight", torch.nn.Parameter(expected_plain.clone()))
        distributed_target = distribute_tensor(torch.zeros_like(expected_plain), mesh, [Shard(0)])  # DTensor [M,N]
        mismatched_reference = torch.nn.Module()
        target = distributed_target if rank == 1 else torch.zeros_like(expected_plain)  # [M,N] or DTensor [M,N]
        mismatched_reference.register_parameter("weight", torch.nn.Parameter(target))
        mismatched_reference.requires_grad_(False)
        mismatched = Cosmos3FlowPolicy.__new__(Cosmos3FlowPolicy)
        mismatched.net = mismatched_policy
        mismatched.net_ref = mismatched_reference
        with pytest.raises(RuntimeError, match="copy plans differ across ranks"):
            mismatched.refresh_ref()
    finally:
        dist.destroy_process_group()


@pytest.mark.L0
@pytest.mark.CPU
def test_cosmos3_online_adapter_refreshes_mixed_distributed_layouts(tmp_path: Path) -> None:
    import torch.multiprocessing as mp

    world_size = 2
    rendezvous_path = str(tmp_path / "mixed-reference-refresh-rendezvous")
    mp.spawn(_mixed_reference_refresh_worker, args=(world_size, rendezvous_path), nprocs=world_size, join=True)


@pytest.mark.L0
@pytest.mark.CPU
def test_cosmos3_online_adapter_delegates_final_latent_sampling() -> None:
    expected = [torch.ones(1, 2), torch.zeros(1, 2)]
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    class _Omni:
        def generate_samples_from_batch(self, *args: object, **kwargs: object) -> dict[str, object]:
            calls.append((args, kwargs))
            return {"vision": expected}

    policy = Cosmos3FlowPolicy.__new__(Cosmos3FlowPolicy)
    policy.omnimot = _Omni()
    policy.net = torch.nn.Linear(2, 2)
    model_batch = {"prompt": ["a", "a"]}
    sampler = object()
    seeds = [1, 2]

    actual = policy.sample_final_latents(
        model_batch,
        sampler=sampler,
        seeds=seeds,
        num_steps=4,
        shift=5.0,
        guidance=3.0,
    )

    assert actual is expected
    assert calls == [
        (
            (model_batch,),
            {
                "net": policy.net,
                "sampler": sampler,
                "guidance": 3.0,
                "seed": seeds,
                "n_sample": len(seeds),
                "num_steps": 4,
                "shift": 5.0,
            },
        )
    ]


@pytest.mark.L0
@pytest.mark.CPU
def test_cosmos3_online_adapter_decodes_one_uniform_latent_batch() -> None:
    calls: list[torch.Tensor] = []
    grad_modes: list[bool] = []

    class _Omni:
        @staticmethod
        def decode(latents: torch.Tensor) -> torch.Tensor:  # [B,C,T,H,W] -> [B,C,T,H,W]
            calls.append(latents)
            grad_modes.append(torch.is_grad_enabled())
            return latents * 2.0  # [B,C,T,H,W]

    policy = Cosmos3FlowPolicy.__new__(Cosmos3FlowPolicy)
    policy.omnimot = _Omni()
    first = torch.ones(1, 3, 2, 2, 2, requires_grad=True)  # [1,C,T,H,W]
    second = torch.zeros(1, 3, 2, 2, 2)  # [1,C,T,H,W]

    decoded = policy.decode_final_latents([first, second])  # [B,C,T,H,W]

    assert len(calls) == 1
    assert grad_modes == [False]
    assert decoded.grad_fn is None
    expected = 2.0 * torch.cat([first, second], dim=0)  # [B,C,T,H,W]
    torch.testing.assert_close(decoded, expected)


@pytest.mark.L0
@pytest.mark.CPU
def test_cosmos3_online_adapter_uses_model_ema_schedule() -> None:
    calls: list[tuple[torch.nn.Module, torch.nn.Module, float]] = []
    buffer_sync_calls: list[bool] = []

    class _Worker:
        def update_average(
            self,
            source: torch.nn.Module,
            target: torch.nn.Module,
            *,
            beta: float,
        ) -> None:
            calls.append((source, target, beta))

    class _Omni:
        net_ema_worker: _Worker = _Worker()

        @staticmethod
        def ema_beta(iteration: int) -> float:
            return 0.25 + iteration

        @staticmethod
        def sync_ema_buffers() -> None:
            buffer_sync_calls.append(True)

    policy = Cosmos3FlowPolicy.__new__(Cosmos3FlowPolicy)
    policy.omnimot = _Omni()
    policy.net = torch.nn.Linear(2, 2)
    policy.net_ema = torch.nn.Linear(2, 2)

    policy.update_ema(3)

    assert calls == [(policy.net, policy.net_ema, 3.25)]
    assert buffer_sync_calls == [True]
