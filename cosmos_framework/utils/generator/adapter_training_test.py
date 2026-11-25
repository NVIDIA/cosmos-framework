# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from cosmos_framework.utils.generator import adapter_training
from cosmos_framework.utils.generator.adapter_artifacts import write_adapter_artifact
from cosmos_framework.utils.generator.adapter_config import (
    AdapterComponentConfig,
    AdapterConfig,
    AdapterSetup,
    adapter_method_settings,
)
from cosmos_framework.utils.generator.adapter_training import (
    adapter_optimizer_groups,
    check_full_base_fits,
    full_base_bytes,
    prepare_adapter_model,
    validate_adapter_build,
    validate_adapter_dtypes,
)
from cosmos_framework.utils.generator.adapters import AdapterLinear, adapter_state_spec

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


class _Text(nn.Module):
    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.config = SimpleNamespace(model_type="qwen3")
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(nn.Module() for _ in range(2))
        for layer in self.model.layers:
            layer.q_proj = nn.Linear(8, 8, bias=False)
            layer.v_proj = nn.Linear(8, 8, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.model.layers:
            x = layer.v_proj(layer.q_proj(x))
        return x


def _config() -> AdapterConfig:
    return AdapterConfig(
        components=[
            AdapterComponentConfig(targets=["model.layers.*.q_proj"], method="dora", weight_decay=0.1, rank=2),
            AdapterComponentConfig(targets=["model.layers.*.v_proj"], method="pissa", lr_multiplier=0.5, rank=2),
        ]
    )


def _adapter_state(net: nn.Module) -> dict[str, torch.Tensor]:
    return {n: t.clone() for n, t in net.state_dict().items() if n in adapter_state_spec(net.adapter_identity)}


def test_setup_and_method_names() -> None:
    assert adapter_method_settings("rsdora") == {"method": "dora", "use_rslora": True}
    assert adapter_method_settings("pissa") == {"method": "pissa", "use_rslora": False}
    with pytest.raises(ValueError, match="unknown adapter method"):
        adapter_method_settings("vera")
    with pytest.raises(ValueError, match="not both"):
        AdapterSetup(_config(), from_checkpoint=True, warm_start="adapter")


def test_build_checks() -> None:
    validate_adapter_build(SimpleNamespace(model_type="qwen3"), dtype="bfloat16", reduce_dtype="float32")
    with pytest.raises(ValueError, match="no adapter binding"):
        validate_adapter_build(SimpleNamespace(model_type="llama"), dtype="bfloat16", reduce_dtype="float32")
    with pytest.raises(ValueError, match="quantized"):
        validate_adapter_build(
            SimpleNamespace(model_type="qwen3", quantization_config={}), dtype="bfloat16", reduce_dtype="float32"
        )
    with pytest.raises(ValueError, match="reduce dtype to float32"):
        validate_adapter_build(SimpleNamespace(model_type="qwen3"), dtype="bfloat16", reduce_dtype="bfloat16")
    with pytest.raises(ValueError, match=r"storage \(float32\) must equal compute \(bfloat16\)"):
        validate_adapter_build(
            SimpleNamespace(model_type="qwen3"), dtype="float32", reduce_dtype="float32", compute_dtype="bfloat16"
        )


def test_dtype_checks() -> None:
    common = dict(storage_dtype="bfloat16", compute_dtype=None, reduce_dtype="float32")
    validate_adapter_dtypes("qwen3_vl", **common)
    validate_adapter_dtypes("qwen3", **{**common, "compute_dtype": "bfloat16"})
    with pytest.raises(ValueError, match="float32 or bfloat16"):
        validate_adapter_dtypes("qwen3", **{**common, "storage_dtype": "float16"})
    with pytest.raises(ValueError, match="no adapter binding"):
        validate_adapter_dtypes(None, **common)


class _Tied(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(tie_word_embeddings=True)
        self.embed = nn.Embedding(100, 8)
        self.head = nn.Linear(8, 100, bias=False)
        self.register_buffer("scale", torch.ones(4))

    def get_input_embeddings(self) -> nn.Module:
        return self.embed

    def get_output_embeddings(self) -> nn.Module:
        return self.head


def test_full_base_size_counts_tied_and_shared_tensors_once() -> None:
    with torch.device("meta"):
        tied = _Tied().to(torch.bfloat16)
    assert full_base_bytes(tied) == (100 * 8 + 4) * 2  # the untied head is tied after materialization
    already_tied = _Tied()
    already_tied.head.weight = already_tied.embed.weight
    assert full_base_bytes(already_tied) == (100 * 8 + 4) * 4
    shared = _Text()
    shared.model.layers[1].v_proj.weight = shared.model.layers[0].v_proj.weight
    assert full_base_bytes(shared) == 3 * 8 * 8 * 4
    check_full_base_fits(900, device_memory=1000)
    with pytest.raises(ValueError, match="does not fit in 90% of the"):
        check_full_base_fits(901, device_memory=1000)


def test_prepare_initializes_reproducibly_restores_and_warm_starts(tmp_path: Path) -> None:
    torch.manual_seed(100)
    fresh = _Text()
    identity = prepare_adapter_model(fresh, AdapterSetup(_config()), base_dtype="float32", architecture="qwen3")
    assert fresh.adapter_identity is identity and len(identity["base_fingerprint"]) == 64
    torch.manual_seed(100)
    again = _Text()
    prepare_adapter_model(again, AdapterSetup(_config()), base_dtype="float32", architecture="qwen3")
    for name, tensor in _adapter_state(fresh).items():
        torch.testing.assert_close(_adapter_state(again)[name], tensor, atol=0, rtol=0)
    x = torch.randn(3, 8)
    torch.testing.assert_close(fresh(x), _Text()(x), atol=1e-5, rtol=1e-5)  # initial function preserved

    restored = _Text()
    prepare_adapter_model(
        restored, AdapterSetup(_config(), from_checkpoint=True), base_dtype="float32", architecture="qwen3"
    )
    assert all(not m.lora_B.weight.any() for m in restored.modules() if isinstance(m, AdapterLinear))

    write_adapter_artifact(str(tmp_path / "adapter"), identity, _adapter_state(fresh))
    warm = _Text()
    prepare_adapter_model(
        warm, AdapterSetup(_config(), warm_start=str(tmp_path / "adapter")), base_dtype="bfloat16", architecture="qwen3"
    )
    for name, tensor in _adapter_state(fresh).items():
        torch.testing.assert_close(_adapter_state(warm)[name], tensor, atol=0, rtol=0)
    with pytest.raises(ValueError, match="different base"):
        prepare_adapter_model(
            _Text(seed=1),
            AdapterSetup(_config(), warm_start=str(tmp_path / "adapter")),
            base_dtype="float32",
            architecture="qwen3",
        )


def test_optimizer_groups_put_the_base_rate_first() -> None:
    net = _Text()
    prepare_adapter_model(net, AdapterSetup(_config()), base_dtype="float32", architecture="qwen3")
    groups = adapter_optimizer_groups(net, lr=1e-3, weight_decay=0.01)
    summary = [(group.lr, group.weight_decay, len(group.parameter_names)) for group in groups]
    assert summary[2] == (5e-4, 0.01, 4)  # PiSSA factors: inherit the optimizer's decay, half the rate, last
    assert sorted(summary[:2]) == [(1e-3, 0.0, 2), (1e-3, 0.1, 4)]  # DoRA magnitudes (no decay), DoRA factors
    assert [name for name, *_ in groups] == ["adapter_0", "adapter_1", "adapter_2"]


def test_single_process_initialization_failure_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(self: AdapterLinear, generator: torch.Generator | None = None) -> None:
        raise RuntimeError("svd failed")

    monkeypatch.setattr(adapter_training.AdapterLinear, "initialize", fail)
    with pytest.raises(RuntimeError, match="svd failed"):
        prepare_adapter_model(_Text(), AdapterSetup(_config()), base_dtype="float32", architecture="qwen3")


def test_initialization_leaves_the_global_rng_untouched() -> None:
    torch.manual_seed(7)
    net = _Text()  # its own construction draws from the RNG; take the state after it
    before = torch.get_rng_state()
    prepare_adapter_model(net, AdapterSetup(_config()), base_dtype="float32", architecture="qwen3")
    assert torch.equal(torch.get_rng_state(), before)


def test_owner_assignment_balances_cost_deterministically() -> None:
    from cosmos_framework.utils.generator.adapter_training import _assign_owners

    sizes = {"a": (64, 64), "b": (8, 8), "c": (32, 32), "d": (8, 8), "e": (16, 16)}
    modules = [
        (
            name,
            AdapterLinear(nn.Linear(n, m, bias=False), AdapterComponentConfig(targets=["x"], rank=2), initialize=False),
        )
        for name, (m, n) in sizes.items()
    ]
    plan = _assign_owners(modules, 2)
    assert [[name for name, _ in owned] for owned in plan] == [["a"], ["b", "c", "d", "e"]]
    assert _assign_owners(list(reversed(modules)), 2) == plan
    assert [len(owned) for owned in _assign_owners(modules[:1], 4)] == [1, 0, 0, 0]
