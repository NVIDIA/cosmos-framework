# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import pytest

from cosmos_framework.configs.base.defaults.quantization import QuantizationConfig
from cosmos_framework.utils.generator.mixed_precision import use_w8a16_step


def test_config_defaults_disable_mixed_precision() -> None:
    config = QuantizationConfig()
    assert config.mixed_precision_first_steps == 0
    assert config.mixed_precision_last_steps == 0
    assert config.mixed_precision_reasoner_policy == "high_precision"
    assert config.mixed_precision_w8a16_cache == "gpu_block"
    assert not config.mixed_precision_enabled


def test_config_enabled_when_any_width_positive() -> None:
    assert QuantizationConfig(mixed_precision_first_steps=1).mixed_precision_enabled
    assert QuantizationConfig(mixed_precision_last_steps=2).mixed_precision_enabled


def test_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        QuantizationConfig(mixed_precision_first_steps=-1)
    with pytest.raises(ValueError):
        QuantizationConfig(mixed_precision_reasoner_policy="fast")
    with pytest.raises(ValueError):
        QuantizationConfig(mixed_precision_w8a16_cache="huge")


def test_schedule_selects_first_and_last_steps() -> None:
    selected = [i for i in range(10) if use_w8a16_step(2, 4, i, 10)]
    assert selected == [0, 1, 6, 7, 8, 9]


def test_schedule_overlap_selects_every_step() -> None:
    assert all(use_w8a16_step(3, 3, i, 4) for i in range(4))


def test_schedule_single_step_is_base_precision() -> None:
    # FSDP alignment pads slow ranks with dummy num_steps=1 samples; keep them cheap.
    assert not use_w8a16_step(3, 3, 0, 1)


def test_schedule_validates_bounds() -> None:
    with pytest.raises(ValueError):
        use_w8a16_step(1, 1, 0, 0)
    with pytest.raises(IndexError):
        use_w8a16_step(1, 1, 5, 5)
    with pytest.raises(IndexError):
        use_w8a16_step(1, 1, -1, 5)


import torch
from torch import nn

from cosmos_framework.utils.generator.mixed_precision import (
    install_mixed_precision_runtime,
    _unwrap_float8,
    _dequantize_w8a16_weight,
)
from cosmos_framework.utils.generator.quantization import _ModelOptFloat8Linear


def _make_fp8_linear(
    out_features: int = 8, in_features: int = 4, device: str = "cpu"
) -> _ModelOptFloat8Linear:
    """Build a _ModelOptFloat8Linear with a real PrototypeFloat8Tensor weight on ``device``.

    Built directly on the target device: moving a constructed PrototypeFloat8Tensor
    with ``.to()`` is not guaranteed to be supported by the subclass.
    """
    from torchao.float8.inference import Float8MMConfig
    from torchao.prototype.quantization.float8_static_quant.prototype_float8_tensor import (
        PrototypeFloat8Tensor,
    )
    from torchao.quantization import PerTensor
    from torchao.quantization.quantize_.workflows import QuantizeTensorToFloat8Kwargs

    module = _ModelOptFloat8Linear(in_features, out_features, bias=False, dtype=torch.bfloat16, device=device)
    module._modelopt_high_precision_dtype = torch.bfloat16
    qdata = torch.randn(out_features, in_features, device=device).clamp(-1, 1).to(torch.float8_e4m3fn)
    mm_config = Float8MMConfig(use_fast_accum=True)
    module.weight = nn.Parameter(
        PrototypeFloat8Tensor(
            qdata,
            torch.full((1, 1), 0.5, dtype=torch.float32, device=device),
            act_quant_scale=torch.full((1, 1), 1.0, dtype=torch.float32, device=device),
            block_size=[out_features, in_features],
            mm_config=mm_config,
            act_quant_kwargs=QuantizeTensorToFloat8Kwargs(
                float8_dtype=torch.float8_e4m3fn, granularity=PerTensor(), mm_config=mm_config
            ),
            dtype=torch.bfloat16,
        ),
        requires_grad=False,
    )
    return module


class _TinyMoTNet(nn.Module):
    """Minimal net shape: one reasoner-path and one generation-path FP8 linear."""

    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.up_proj = _make_fp8_linear()
        self.mlp_moe_gen = nn.Module()
        self.mlp_moe_gen.up_proj = _make_fp8_linear()


def _enabled_config(**overrides) -> QuantizationConfig:
    values = dict(mixed_precision_first_steps=1, mixed_precision_last_steps=1, mixed_precision_w8a16_cache="none")
    values.update(overrides)
    return QuantizationConfig(**values)


def test_install_returns_none_when_disabled() -> None:
    net = _TinyMoTNet()
    assert install_mixed_precision_runtime(net, QuantizationConfig()) is None
    assert getattr(net, "_mixed_precision_runtime", None) is None


def test_install_classifies_paths_and_tags_modules() -> None:
    net = _TinyMoTNet()
    runtime = install_mixed_precision_runtime(net, _enabled_config())
    assert runtime is not None
    assert net._mixed_precision_runtime is runtime
    assert runtime.installed_counts == {"reasoner": 1, "generation": 1}
    assert net.mlp.up_proj._mixed_precision_path == "reasoner"
    assert net.mlp_moe_gen.up_proj._mixed_precision_path == "generation"
    assert net.mlp.up_proj._mixed_precision_runtime is runtime


def test_install_requires_both_paths() -> None:
    net = _TinyMoTNet()
    net.mlp.up_proj = nn.Linear(4, 8)  # no reasoner-path FP8 linear left
    with pytest.raises(ValueError, match="reasoner"):
        install_mixed_precision_runtime(net, _enabled_config())


def test_step_state_and_trace() -> None:
    net = _TinyMoTNet()
    runtime = install_mixed_precision_runtime(net, _enabled_config())
    runtime.set_step(0, 4)
    assert runtime.use_high_precision("generation")
    runtime.set_step(1, 4)
    assert not runtime.use_high_precision("generation")
    # reasoner policy is static, independent of the step
    assert runtime.use_high_precision("reasoner")
    runtime.set_base_precision()
    assert not runtime.use_high_precision("generation")
    runtime.reset()
    assert runtime.last_trace == ("W8A16", "W8A8")
    assert not runtime.use_high_precision("generation")


def test_reasoner_base_policy() -> None:
    net = _TinyMoTNet()
    runtime = install_mixed_precision_runtime(
        net, _enabled_config(mixed_precision_reasoner_policy="base_precision")
    )
    assert not runtime.use_high_precision("reasoner")


def test_w8a16_forward_matches_manual_dequant() -> None:
    net = _TinyMoTNet()
    runtime = install_mixed_precision_runtime(net, _enabled_config())
    module = net.mlp_moe_gen.up_proj
    inputs = torch.randn(3, 5, 4, dtype=torch.bfloat16)
    runtime.set_step(0, 4)  # W8A16
    output = module(inputs)
    weight = module.weight
    expected = torch.nn.functional.linear(
        inputs.reshape(-1, 4), weight.qdata.to(torch.bfloat16) * weight.scale.to(torch.bfloat16)
    ).reshape(3, 5, 8)
    assert output.shape == (3, 5, 8)
    torch.testing.assert_close(output, expected)


def test_w8a16_forward_zero_rows() -> None:
    net = _TinyMoTNet()
    runtime = install_mixed_precision_runtime(net, _enabled_config())
    runtime.set_step(0, 4)
    output = net.mlp_moe_gen.up_proj(torch.empty(0, 4, dtype=torch.bfloat16))
    assert output.shape == (0, 8)


def test_install_rejects_smoothquant_layer() -> None:
    net = _TinyMoTNet()
    net.mlp_moe_gen.up_proj.pre_quant_scale = torch.ones(4)
    with pytest.raises(ValueError, match="pre_quant_scale"):
        install_mixed_precision_runtime(net, _enabled_config())


def test_dequantize_w8a16_weight_unwraps_float8() -> None:
    module = _make_fp8_linear()
    dequantized = _dequantize_w8a16_weight(_unwrap_float8(module.weight))
    expected = module.weight.qdata.to(torch.bfloat16) * module.weight.scale.to(torch.bfloat16)
    torch.testing.assert_close(dequantized, expected)


def test_generation_cache_covers_generation_path_only() -> None:
    net = _TinyMoTNet()
    runtime = install_mixed_precision_runtime(
        net, _enabled_config(mixed_precision_w8a16_cache="generation")
    )
    gen = net.mlp_moe_gen.up_proj
    reasoner = net.mlp.up_proj
    assert isinstance(gen._w8a16_weight_cache, torch.Tensor)
    assert gen._w8a16_weight_cache.dtype == torch.bfloat16
    assert getattr(reasoner, "_w8a16_weight_cache", None) is None
    assert runtime.cached_bytes == gen._w8a16_weight_cache.numel() * 2
    # cache must not leak into state_dict
    assert not any("_w8a16_weight_cache" in key for key in net.state_dict())


def test_all_cache_covers_both_paths_and_matches_dequant() -> None:
    net = _TinyMoTNet()
    install_mixed_precision_runtime(net, _enabled_config(mixed_precision_w8a16_cache="all"))
    for module in (net.mlp.up_proj, net.mlp_moe_gen.up_proj):
        weight = module.weight
        torch.testing.assert_close(
            module._w8a16_weight_cache,
            weight.qdata.to(torch.bfloat16) * weight.scale.to(torch.bfloat16),
        )


def test_cached_forward_matches_uncached_forward() -> None:
    torch.manual_seed(0)
    net_cached = _TinyMoTNet()
    torch.manual_seed(0)
    net_plain = _TinyMoTNet()
    runtime_cached = install_mixed_precision_runtime(
        net_cached, _enabled_config(mixed_precision_w8a16_cache="all")
    )
    runtime_plain = install_mixed_precision_runtime(net_plain, _enabled_config())
    runtime_cached.set_step(0, 4)
    runtime_plain.set_step(0, 4)
    inputs = torch.randn(2, 4, dtype=torch.bfloat16)
    torch.testing.assert_close(net_cached.mlp_moe_gen.up_proj(inputs), net_plain.mlp_moe_gen.up_proj(inputs))
