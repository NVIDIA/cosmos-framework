# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""@L0 @CPU: model-owned policy parallelism normalization and mesh resolution."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from cosmos_framework.posttrain.config.build import (  # noqa: E402
    build_backbone,
    native_policy_build_spec,
    normalize_model_backend,
    resolve_pdims,
    resolve_policy_parallelism,
)
from cosmos_framework.posttrain.config.model import (  # noqa: E402
    HFModelConfig,
    NativePolicyBuildSpec,
    NativePolicyParallelismConfig,
    Qwen3VLModelConfig,
)
from cosmos_framework.posttrain.model.parallel_dims import ParallelDims  # noqa: E402


@pytest.fixture
def world(monkeypatch: pytest.MonkeyPatch):
    """Stub a live process group and skip real device-mesh construction."""
    import torch.distributed as dist

    def _set(size: int) -> None:
        monkeypatch.setattr(dist, "is_initialized", lambda: True)
        monkeypatch.setattr(dist, "get_world_size", lambda: size)

    monkeypatch.setattr(ParallelDims, "build_device_mesh", lambda self, device: None)
    return _set


@pytest.mark.L0
@pytest.mark.CPU
def test_hf_policy_parallelism_is_inferred_inside_policy_ranks() -> None:
    cfg = SimpleNamespace(model_backend="hf", model_hf=HFModelConfig())
    assert resolve_policy_parallelism(cfg, 8) == (1, 8)

    cfg.model_hf.parallelism.dp_replicate = 2
    assert resolve_policy_parallelism(cfg, 8) == (2, 4)

    cfg.model_hf.parallelism.dp_shard = 2
    with pytest.raises(ValueError, match="POLICY_RANKS"):
        resolve_policy_parallelism(cfg, 8)


@pytest.mark.L0
@pytest.mark.CPU
def test_i4_policy_parallelism_uses_native_model_config() -> None:
    parallelism = SimpleNamespace(
        data_parallel_replicate_degree=4,
        data_parallel_shard_degree=2,
        context_parallel_shard_degree=1,
        cfg_parallel_shard_degree=1,
    )
    cfg = SimpleNamespace(
        model_backend="i4",
        model=SimpleNamespace(config=SimpleNamespace(parallelism=parallelism)),
    )
    assert resolve_policy_parallelism(cfg, 8) == (4, 2)
    assert parallelism.data_parallel_replicate_degree == 4
    assert parallelism.data_parallel_shard_degree == 2

    parallelism.data_parallel_replicate_degree = 2
    parallelism.data_parallel_shard_degree = -1
    assert resolve_policy_parallelism(cfg, 8) == (2, 4)
    assert parallelism.data_parallel_shard_degree == 4

    parallelism.context_parallel_shard_degree = 2
    with pytest.raises(ValueError, match="context_parallel"):
        resolve_policy_parallelism(cfg, 8)


@pytest.mark.L0
@pytest.mark.CPU
def test_native_policy_build_spec_owns_generic_i4_parallelism() -> None:
    spec = NativePolicyBuildSpec(
        builder="tests.fake.build_policy",
        model_id="native/model",
        model_config=SimpleNamespace(name_or_path="native/model"),
        parallelism=NativePolicyParallelismConfig(dp_replicate=2, dp_shard=-1),
    )
    cfg = SimpleNamespace(model_backend="i4", model=spec)

    assert resolve_policy_parallelism(cfg, 8) == (2, 4)
    # Inference is repeatable and does not rewrite the public spec's sentinel.
    assert spec.parallelism.dp_shard == -1


@pytest.mark.L0
@pytest.mark.CPU
def test_native_policy_spec_rejects_a_split_policy_identity() -> None:
    spec = NativePolicyBuildSpec(
        builder="tests.fake.build_policy",
        model_id="native/model-a",
        model_config=SimpleNamespace(name_or_path="native/model-b"),
    )

    with pytest.raises(ValueError, match="must identify the same policy artifact"):
        native_policy_build_spec(SimpleNamespace(model_backend="i4", model=spec))


@pytest.mark.L0
@pytest.mark.CPU
def test_legacy_qwen_backend_normalizes_to_generic_i4_spec() -> None:
    model_config = Qwen3VLModelConfig(name_or_path="native/model")
    cfg = SimpleNamespace(model_backend="qwen3_vl", model_qwen3_vl=model_config)

    spec = native_policy_build_spec(cfg)
    assert normalize_model_backend(cfg.model_backend) == "i4"
    assert spec.model_config is model_config
    assert spec.requires_collated_policy_forward is True
    assert resolve_policy_parallelism(cfg, 8) == (1, 8)


@pytest.mark.L0
@pytest.mark.CPU
def test_build_backbone_dispatches_native_spec_without_a_family_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    fixture = ModuleType("posttrain_native_builder_fixture")

    def _build(model_id: str, model_config: object, pdims: object) -> object:
        captured.update(model_id=model_id, model_config=model_config, pdims=pdims)
        return "native-policy"

    fixture.build = _build
    monkeypatch.setitem(sys.modules, fixture.__name__, fixture)
    # The generic builder config does not need to mimic an HF config; tokenizer identity lives on
    # NativePolicyBuildSpec.model_id.
    model_config = SimpleNamespace(native_width=7)
    config = SimpleNamespace(
        model_backend="i4",
        model=NativePolicyBuildSpec(
            builder=f"{fixture.__name__}.build",
            model_id="native/model",
            model_config=model_config,
            parallelism=NativePolicyParallelismConfig(dp_replicate=2, dp_shard=2),
        ),
    )
    pdims = SimpleNamespace(world_size=4, dp_replicate=2, dp_shard=2)

    assert build_backbone(config, pdims=pdims) == "native-policy"
    assert captured == {"model_id": "native/model", "model_config": model_config, "pdims": pdims}


@pytest.mark.L0
@pytest.mark.CPU
def test_explicit_hsdp_factorization_builds_over_live_policy_world(world) -> None:
    world(4)
    pdims = resolve_pdims(parallelism=(2, 2))
    assert (pdims.world_size, pdims.dp_replicate, pdims.dp_shard) == (4, 2, 2)


@pytest.mark.L0
@pytest.mark.CPU
def test_pure_fsdp_fallback_remains_for_unconfigured_callers(world) -> None:
    world(4)
    pdims = resolve_pdims()
    assert (pdims.dp_replicate, pdims.dp_shard) == (1, 4)


@pytest.mark.L0
@pytest.mark.CPU
def test_explicit_factor_must_equal_live_policy_world(world) -> None:
    world(4)
    with pytest.raises(ValueError, match="live policy world size"):
        resolve_pdims(parallelism=(2, 3))


@pytest.mark.L0
@pytest.mark.CPU
def test_injected_mesh_must_match_model_config(world, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("ray")
    from cosmos_framework.posttrain.workers import ray as ray_workers

    world(4)
    sentinel = ParallelDims(world_size=4, dp_replicate=2, dp_shard=2)
    monkeypatch.setattr(ray_workers, "_ACTOR_PDIMS", sentinel)
    assert resolve_pdims(parallelism=(2, 2)) is sentinel
    with pytest.raises(ValueError, match="disagrees"):
        resolve_pdims(parallelism=(1, 4))
