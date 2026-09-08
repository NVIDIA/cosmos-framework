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
    assert config.mixed_precision_w8a16_cache == "none"
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
    out_features: int = 8,
    in_features: int = 4,
    device: str = "cpu",
    high_precision_dtype: torch.dtype = torch.bfloat16,
) -> _ModelOptFloat8Linear:
    """Build a _ModelOptFloat8Linear with a real PrototypeFloat8Tensor weight on ``device``.

    Built directly on the target device: moving a constructed PrototypeFloat8Tensor
    with ``.to()`` is not guaranteed to be supported by the subclass.

    ``high_precision_dtype`` controls the PrototypeFloat8Tensor subclass's own
    ``.dtype`` (i.e. what the checkpoint declares itself as, e.g. float32 on
    the real Cosmos3-Nano checkpoint) -- independent of the activation dtype
    the mixed-precision runtime is configured with.
    """
    from torchao.float8.inference import Float8MMConfig
    from torchao.prototype.quantization.float8_static_quant.prototype_float8_tensor import (
        PrototypeFloat8Tensor,
    )
    from torchao.quantization import PerTensor
    from torchao.quantization.quantize_.workflows import QuantizeTensorToFloat8Kwargs

    module = _ModelOptFloat8Linear(
        in_features, out_features, bias=False, dtype=high_precision_dtype, device=device
    )
    module._modelopt_high_precision_dtype = high_precision_dtype
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
            dtype=high_precision_dtype,
        ),
        requires_grad=False,
    )
    return module


class _TinyMoTNet(nn.Module):
    """Minimal net shape: one reasoner-path and one generation-path FP8 linear."""

    def __init__(self, high_precision_dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.up_proj = _make_fp8_linear(high_precision_dtype=high_precision_dtype)
        self.mlp_moe_gen = nn.Module()
        self.mlp_moe_gen.up_proj = _make_fp8_linear(high_precision_dtype=high_precision_dtype)


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
    dequantized = _dequantize_w8a16_weight(_unwrap_float8(module.weight), torch.bfloat16)
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


class _TinyLayeredNet(nn.Module):
    """N decoder-layer-like blocks, each with reasoner and generation linears."""

    def __init__(
        self,
        device: str = "cpu",
        num_layers: int = 2,
        out_features: int = 16,
        in_features: int = 16,
        high_precision_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                _TinyMoTLayer(
                    device,
                    out_features=out_features,
                    in_features=in_features,
                    high_precision_dtype=high_precision_dtype,
                )
                for _ in range(num_layers)
            ]
        )


class _TinyMoTLayer(nn.Module):
    def __init__(
        self,
        device: str = "cpu",
        out_features: int = 16,
        in_features: int = 16,
        high_precision_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        # 16/16 default: real (non-mixed-precision) W8A8 forward dispatches to
        # torch._scaled_mm, which requires both GEMM operands' reduction/output
        # dims to be multiples of 16 on this hardware — unlike the W8A16 path
        # (dense F.linear over a dequantized weight), which tolerates any shape.
        self.mlp = nn.Module()
        self.mlp.up_proj = _make_fp8_linear(
            out_features=out_features, in_features=in_features, device=device, high_precision_dtype=high_precision_dtype
        )
        self.mlp_moe_gen = nn.Module()
        self.mlp_moe_gen.up_proj = _make_fp8_linear(
            out_features=out_features, in_features=in_features, device=device, high_precision_dtype=high_precision_dtype
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.mlp_moe_gen.up_proj(inputs)


def _cuda_layered_net() -> "_TinyLayeredNet":
    # Built directly on cuda — no .to() move of the FP8 tensor subclass.
    return _TinyLayeredNet(device="cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="block providers need CUDA streams")
@pytest.mark.parametrize("cache_mode", ["gpu_block", "cpu_block"])
def test_block_provider_stages_and_matches_dequant(cache_mode: str) -> None:
    net = _cuda_layered_net()
    runtime = install_mixed_precision_runtime(
        net,
        _enabled_config(mixed_precision_w8a16_cache=cache_mode),
        blocks=list(net.layers),
    )
    runtime.set_step(0, 4)  # W8A16 -> preload_first
    inputs = torch.randn(2, 16, dtype=torch.bfloat16, device="cuda")
    for layer in net.layers:
        output = layer(inputs)
        weight = layer.mlp_moe_gen.up_proj.weight
        expected = torch.nn.functional.linear(
            inputs, weight.qdata.to(torch.bfloat16) * weight.scale.to(torch.bfloat16)
        )
        torch.testing.assert_close(output, expected)
    runtime.reset()
    assert all(layer.mlp_moe_gen.up_proj._mixed_precision_staged_weight is None for layer in net.layers)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="block providers need CUDA streams")
def test_block_provider_idle_on_w8a8_steps() -> None:
    net = _cuda_layered_net()
    runtime = install_mixed_precision_runtime(
        net,
        _enabled_config(mixed_precision_w8a16_cache="gpu_block"),
        blocks=list(net.layers),
    )
    runtime.set_step(1, 4)  # middle step -> W8A8
    _ = net.layers[0](torch.randn(2, 16, dtype=torch.bfloat16, device="cuda"))
    assert net.layers[0].mlp_moe_gen.up_proj._mixed_precision_staged_weight is None
    runtime.reset()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="block providers need CUDA streams")
def test_block_provider_reusable_across_requests() -> None:
    net = _cuda_layered_net()
    runtime = install_mixed_precision_runtime(
        net,
        _enabled_config(mixed_precision_w8a16_cache="gpu_block"),
        blocks=list(net.layers),
    )
    inputs = torch.randn(2, 16, dtype=torch.bfloat16, device="cuda")
    for _ in range(2):  # two full "requests" with mixed schedules
        for step in range(4):
            runtime.set_step(step, 4)
            for layer in net.layers:
                output = layer(inputs)
                if runtime.use_high_precision("generation"):
                    weight = layer.mlp_moe_gen.up_proj.weight
                    expected = torch.nn.functional.linear(
                        inputs, weight.qdata.to(torch.bfloat16) * weight.scale.to(torch.bfloat16)
                    )
                    torch.testing.assert_close(output, expected)
        runtime.reset()
    assert runtime.last_trace == ("W8A16", "W8A8", "W8A8", "W8A16")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="block providers need CUDA streams")
def test_block_provider_wraparound_across_four_blocks() -> None:
    # 4 blocks over 2 slots (block_index % 2) means blocks 0/2 share slot 0 and
    # blocks 1/3 share slot 1 within a single W8A16 step -- this exercises
    # mid-request slot wraparound, not just cross-request reuse. Larger (256,
    # 256) linears and 20 repeated passes create real staging-stream work and
    # contention around the free/ready events, instead of finishing so fast
    # that a sync bug could go unnoticed.
    net = _TinyLayeredNet(device="cuda", num_layers=4, out_features=256, in_features=256)
    runtime = install_mixed_precision_runtime(
        net,
        _enabled_config(mixed_precision_w8a16_cache="gpu_block"),
        blocks=list(net.layers),
    )
    runtime.set_step(0, 4)  # W8A16 -> preload_first
    inputs = torch.randn(2, 256, dtype=torch.bfloat16, device="cuda")
    for _ in range(20):
        for layer in net.layers:
            output = layer(inputs)
            weight = layer.mlp_moe_gen.up_proj.weight
            expected = torch.nn.functional.linear(
                inputs, weight.qdata.to(torch.bfloat16) * weight.scale.to(torch.bfloat16)
            )
            torch.testing.assert_close(output, expected)
    runtime.reset()
    assert all(layer.mlp_moe_gen.up_proj._mixed_precision_staged_weight is None for layer in net.layers)


def test_quantization_args_round_trip_mixed_precision_fields() -> None:
    from cosmos_framework.inference.common.args import QuantizationOverrides

    overrides = QuantizationOverrides(
        mixed_precision_first_steps=2,
        mixed_precision_last_steps=3,
        mixed_precision_reasoner_policy="base_precision",
        mixed_precision_w8a16_cache="none",
    )
    args = overrides.build_quantization()
    assert args.mixed_precision_first_steps == 2
    assert args.mixed_precision_last_steps == 3
    assert args.mixed_precision_reasoner_policy == "base_precision"
    assert args.mixed_precision_w8a16_cache == "none"


def test_validate_mixed_precision_load_rejects_non_fp8_checkpoint() -> None:
    from cosmos_framework.inference.model import _validate_mixed_precision_load

    with pytest.raises(ValueError, match="ModelOpt FP8"):
        _validate_mixed_precision_load(
            _enabled_config(), modelopt_checkpoint=False, use_cuda_graphs=False
        )


def test_validate_mixed_precision_load_rejects_cuda_graphs() -> None:
    from cosmos_framework.inference.model import _validate_mixed_precision_load

    with pytest.raises(ValueError, match="cuda_graphs"):
        _validate_mixed_precision_load(
            _enabled_config(), modelopt_checkpoint=True, use_cuda_graphs=True
        )


def test_validate_mixed_precision_load_noop_when_disabled() -> None:
    from cosmos_framework.inference.model import _validate_mixed_precision_load

    _validate_mixed_precision_load(QuantizationConfig(), modelopt_checkpoint=False, use_cuda_graphs=True)


# --- Regression coverage for the real-checkpoint W8A16 dtype bug -----------
#
# On the real Cosmos3-Nano FP8 checkpoint, the PrototypeFloat8Tensor
# subclass's own ``.dtype`` is float32 (not the activation dtype). The first
# W8A16 forward used to dequantize into ``weight.dtype`` and crashed with
# "expected mat1 and mat2 to have the same dtype, but got: c10::BFloat16 !=
# float" against bf16 activations. These tests build fp32-subclass-dtype FP8
# linears (mirroring the real checkpoint) and assert the runtime always
# resolves dense W8A16 weights in its configured activation dtype instead.


def test_w8a16_forward_matches_manual_dequant_fp32_subclass_dtype() -> None:
    net = _TinyMoTNet(high_precision_dtype=torch.float32)
    runtime = install_mixed_precision_runtime(net, _enabled_config())
    module = net.mlp_moe_gen.up_proj
    inputs = torch.randn(3, 5, 4, dtype=torch.bfloat16)
    runtime.set_step(0, 4)  # W8A16
    output = module(inputs)
    weight = module.weight
    expected = torch.nn.functional.linear(
        inputs.reshape(-1, 4), weight.qdata.to(torch.bfloat16) * weight.scale.to(torch.bfloat16)
    ).reshape(3, 5, 8)
    assert output.dtype == torch.bfloat16
    assert output.shape == (3, 5, 8)
    torch.testing.assert_close(output, expected)


def test_all_cache_bf16_regardless_of_fp32_subclass_dtype() -> None:
    net = _TinyMoTNet(high_precision_dtype=torch.float32)
    runtime = install_mixed_precision_runtime(net, _enabled_config(mixed_precision_w8a16_cache="all"))
    for module in (net.mlp.up_proj, net.mlp_moe_gen.up_proj):
        weight = module.weight
        assert module._w8a16_weight_cache.dtype == torch.bfloat16
        torch.testing.assert_close(
            module._w8a16_weight_cache,
            weight.qdata.to(torch.bfloat16) * weight.scale.to(torch.bfloat16),
        )
    runtime.set_step(0, 4)  # W8A16
    inputs = torch.randn(2, 4, dtype=torch.bfloat16)
    output = net.mlp_moe_gen.up_proj(inputs)
    weight = net.mlp_moe_gen.up_proj.weight
    expected = torch.nn.functional.linear(inputs, weight.qdata.to(torch.bfloat16) * weight.scale.to(torch.bfloat16))
    assert output.dtype == torch.bfloat16
    torch.testing.assert_close(output, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="block providers need CUDA streams")
def test_gpu_block_provider_bf16_slots_regardless_of_fp32_subclass_dtype() -> None:
    net = _TinyLayeredNet(device="cuda", high_precision_dtype=torch.float32)
    runtime = install_mixed_precision_runtime(
        net,
        _enabled_config(mixed_precision_w8a16_cache="gpu_block"),
        blocks=list(net.layers),
    )
    assert runtime._block_provider._slots[0].dtype == torch.bfloat16
    assert runtime._block_provider._slots[1].dtype == torch.bfloat16
    runtime.set_step(0, 4)  # W8A16 -> preload_first
    inputs = torch.randn(2, 16, dtype=torch.bfloat16, device="cuda")
    for layer in net.layers:
        output = layer(inputs)
        weight = layer.mlp_moe_gen.up_proj.weight
        expected = torch.nn.functional.linear(
            inputs, weight.qdata.to(torch.bfloat16) * weight.scale.to(torch.bfloat16)
        )
        assert output.dtype == torch.bfloat16
        torch.testing.assert_close(output, expected)
    runtime.reset()


class _ReasonerOnlyLayer(nn.Module):
    """A decoder-layer-like block with only a reasoner-path FP8 linear."""

    def __init__(self, device: str = "cpu", features: int = 16) -> None:
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.up_proj = _make_fp8_linear(out_features=features, in_features=features, device=device)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.mlp.up_proj(inputs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="block providers need CUDA streams")
@pytest.mark.parametrize("cache_mode", ["gpu_block", "cpu_block"])
def test_block_provider_tolerates_generation_free_leading_block(cache_mode: str) -> None:
    """Block 0 with no generation-path linears must not break initialize();
    later blocks' staged weights must still resolve correctly."""
    net = nn.Module()
    net.layers = nn.ModuleList([_ReasonerOnlyLayer(device="cuda"), _TinyMoTLayer(device="cuda")])
    runtime = install_mixed_precision_runtime(
        net,
        _enabled_config(mixed_precision_w8a16_cache=cache_mode),
        blocks=list(net.layers),
    )
    runtime.set_step(0, 4)  # W8A16 -> preload_first stages the empty block 0
    inputs = torch.randn(2, 16, dtype=torch.bfloat16, device="cuda")
    _ = net.layers[0](inputs)
    output = net.layers[1](inputs)
    weight = net.layers[1].mlp_moe_gen.up_proj.weight
    expected = torch.nn.functional.linear(
        inputs, weight.qdata.to(torch.bfloat16) * weight.scale.to(torch.bfloat16)
    )
    torch.testing.assert_close(output, expected)
    runtime.reset()
