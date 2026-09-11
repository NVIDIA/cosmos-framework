# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json
from pathlib import Path

import pytest
import safetensors.torch
import torch
from torch import nn

from cosmos_framework.configs.base.defaults.quantization import QuantizationConfig
from cosmos_framework.utils.generator.quantization import (
    QdqSimLinear,
    apply_modelopt_fp8_checkpoint_inplace,
    apply_quantization_inplace,
    fake_quant_fp8,
    fake_quant_int8,
    is_modelopt_fp8_checkpoint,
)


class TinyLinearModel(nn.Module):
    selected: nn.Linear
    unselected: nn.Linear

    def __init__(self, device: torch.device | str = "cuda") -> None:
        super().__init__()
        self.selected = nn.Linear(16, 16, bias=False, device=device, dtype=torch.bfloat16)
        self.unselected = nn.Linear(16, 16, bias=False, device=device, dtype=torch.bfloat16)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # inputs: [B,D], returns: [B,D]
        return self.selected(inputs) + self.unselected(inputs)  # [B,D]


def _write_modelopt_fp8_checkpoint(
    checkpoint_path: Path,
    quantized_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    input_scale: torch.Tensor,
    additional_module: str | None = None,
    additional_module_has_scales: bool = True,
) -> dict[str, str]:
    """Write a minimal ModelOpt-style FP8 export with weights and scales in separate shards."""
    checkpoint_path.mkdir()
    (checkpoint_path / "hf_quant_config.json").write_text(
        json.dumps({"quant_method": "modelopt", "quant_algo": "FP8"}),
        encoding="utf-8",
    )
    weight_shard_name = "model-00001-of-00002.safetensors"
    scale_shard_name = "model-00002-of-00002.safetensors"
    module_names = ["selected"]
    if additional_module is not None:
        module_names.append(additional_module)
    scale_module_names = module_names if additional_module_has_scales else ["selected"]
    weight_tensors = {f"{module_name}.weight": quantized_weight.clone() for module_name in module_names}
    scale_tensors = {
        key: value
        for module_name in scale_module_names
        for key, value in (
            (f"{module_name}.input_scale", input_scale.clone()),
            (f"{module_name}.weight_scale", weight_scale.clone()),
        )
    }
    safetensors.torch.save_file(weight_tensors, checkpoint_path / weight_shard_name)
    safetensors.torch.save_file(scale_tensors, checkpoint_path / scale_shard_name)
    weight_map = {
        **{f"{module_name}.weight": weight_shard_name for module_name in module_names},
        **{
            f"{module_name}.{scale_name}": scale_shard_name
            for module_name in scale_module_names
            for scale_name in ("input_scale", "weight_scale")
        },
    }
    (checkpoint_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}),
        encoding="utf-8",
    )
    return weight_map


def _identity_key_mapper(source_key: str, shard_path: str) -> str:
    del shard_path
    return source_key


def _ignore_extra_key_mapper(source_key: str, shard_path: str) -> str | None:
    del shard_path
    return None if source_key.startswith("ignored.") else source_key


def test_is_modelopt_fp8_checkpoint_without_config(tmp_path: Path) -> None:
    assert not is_modelopt_fp8_checkpoint(tmp_path)


def test_is_modelopt_fp8_checkpoint_rejects_malformed_config(tmp_path: Path) -> None:
    (tmp_path / "hf_quant_config.json").write_text("{", encoding="utf-8")

    with pytest.raises(ValueError, match="Unable to read a valid checkpoint quantization config"):
        is_modelopt_fp8_checkpoint(tmp_path)


@pytest.mark.parametrize(
    "quant_config",
    [
        pytest.param([], id="non-object"),
        pytest.param({"quant_method": "other", "quant_algo": "FP8"}, id="unsupported-method"),
        pytest.param({"quant_method": "modelopt", "quant_algo": "OTHER"}, id="unsupported-algorithm"),
    ],
)
def test_is_modelopt_fp8_checkpoint_rejects_unsupported_config(tmp_path: Path, quant_config: object) -> None:
    (tmp_path / "hf_quant_config.json").write_text(json.dumps(quant_config), encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported checkpoint quantization configuration"):
        is_modelopt_fp8_checkpoint(tmp_path)


def test_apply_modelopt_fp8_checkpoint_preserves_exported_tensors(tmp_path: Path) -> None:
    model = TinyLinearModel(device="cpu").eval()
    original_selected = model.selected
    quantized_weight = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).reshape(16, 16)
    weight_scale = torch.tensor(0.125, dtype=torch.float32)
    input_scale = torch.tensor(0.25, dtype=torch.float32)
    weight_map = _write_modelopt_fp8_checkpoint(tmp_path / "checkpoint", quantized_weight, weight_scale, input_scale)

    converted = apply_modelopt_fp8_checkpoint_inplace(
        model,
        tmp_path / "checkpoint",
        key_mapper=_identity_key_mapper,
        weight_map=weight_map,
    )

    assert is_modelopt_fp8_checkpoint(tmp_path / "checkpoint")
    assert converted == ["selected"]
    assert isinstance(model.selected, nn.Linear)
    assert model.selected is not original_selected
    assert type(original_selected) is nn.Linear
    assert type(model.selected.weight).__name__ == "PrototypeFloat8Tensor"
    assert torch.equal(model.selected.weight.qdata.view(torch.uint8), quantized_weight.view(torch.uint8))
    assert torch.equal(model.selected.weight.scale, weight_scale.reshape(1, 1))
    assert torch.equal(model.selected.weight.act_quant_scale, input_scale.reshape(1, 1))
    assert type(model.unselected.weight) is nn.Parameter


def test_apply_modelopt_fp8_checkpoint_skips_ignored_keys(tmp_path: Path) -> None:
    model = TinyLinearModel(device="cpu").eval()
    quantized_weight = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).reshape(16, 16)
    weight_scale = torch.tensor(0.125, dtype=torch.float32)
    input_scale = torch.tensor(0.25, dtype=torch.float32)
    weight_map = _write_modelopt_fp8_checkpoint(
        tmp_path / "checkpoint",
        quantized_weight,
        weight_scale,
        input_scale,
        additional_module="ignored",
        additional_module_has_scales=False,
    )

    converted = apply_modelopt_fp8_checkpoint_inplace(
        model,
        tmp_path / "checkpoint",
        key_mapper=_ignore_extra_key_mapper,
        weight_map=weight_map,
    )

    assert converted == ["selected"]


def test_apply_modelopt_fp8_checkpoint_skips_absent_target(tmp_path: Path) -> None:
    model = TinyLinearModel(device="cpu").eval()
    quantized_weight = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).reshape(16, 16)
    weight_scale = torch.tensor(0.125, dtype=torch.float32)
    input_scale = torch.tensor(0.25, dtype=torch.float32)
    weight_map = _write_modelopt_fp8_checkpoint(
        tmp_path / "checkpoint",
        quantized_weight,
        weight_scale,
        input_scale,
        additional_module="missing",
        additional_module_has_scales=False,
    )

    converted = apply_modelopt_fp8_checkpoint_inplace(
        model,
        tmp_path / "checkpoint",
        key_mapper=_identity_key_mapper,
        weight_map=weight_map,
    )

    assert converted == ["selected"]


@pytest.mark.gpus(1)
def test_apply_modelopt_fp8_checkpoint_uses_torchao_linear_dispatch(tmp_path: Path) -> None:
    if torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("requires an Ada or newer GPU")
    pytest.importorskip("torchao")

    model = TinyLinearModel().eval()
    original_weight = model.selected.weight.detach().cpu()
    weight_scale = original_weight.abs().amax().float() / torch.finfo(torch.float8_e4m3fn).max
    quantized_weight = (original_weight / weight_scale).to(torch.float8_e4m3fn)
    input_scale = torch.tensor(0.025, dtype=torch.float32)
    weight_map = _write_modelopt_fp8_checkpoint(tmp_path / "checkpoint", quantized_weight, weight_scale, input_scale)
    apply_modelopt_fp8_checkpoint_inplace(
        model,
        tmp_path / "checkpoint",
        key_mapper=_identity_key_mapper,
        weight_map=weight_map,
    )

    inputs = torch.randn((6, 16), device="cuda", dtype=torch.bfloat16)
    output = model.selected(inputs)
    reasoning_inputs = torch.randn((2, 3, 16), device="cuda", dtype=torch.bfloat16)
    reasoning_output = model.selected(reasoning_inputs)
    output_after_reasoning = model.selected(inputs)
    compiled_model = torch.compile(model.selected, dynamic=True)
    compiled_output = compiled_model(inputs)
    compiled_reasoning_output = compiled_model(reasoning_inputs)
    empty_output = model.selected(inputs[:0])

    assert output.shape == (6, 16)
    assert torch.isfinite(output).all()
    assert reasoning_output.shape == (2, 3, 16)
    assert torch.isfinite(reasoning_output).all()
    assert output_after_reasoning.shape == (6, 16)
    assert torch.isfinite(output_after_reasoning).all()
    assert compiled_output.shape == (6, 16)
    assert torch.isfinite(compiled_output).all()
    assert compiled_reasoning_output.shape == (2, 3, 16)
    assert torch.isfinite(compiled_reasoning_output).all()
    assert empty_output.shape == (0, 16)
    assert model.selected.weight.act_quant_scale.shape == (1, 1)


# --------------------------------------------------------------------------
# int8_sim / fp8_sim quantize-dequantize simulation (torchao-free, CPU-testable)
# --------------------------------------------------------------------------


class TinyMixedModel(nn.Module):
    """Two linears plus a non-linear module, all on CPU, for selection tests."""

    def __init__(self, dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        self.selected = nn.Linear(32, 16, bias=True, dtype=dtype)
        self.unselected = nn.Linear(32, 16, bias=False, dtype=dtype)
        self.norm = nn.LayerNorm(32, dtype=dtype)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # inputs: [B,32], returns: [B,16]
        return self.selected(self.norm(inputs)) + self.unselected(inputs)  # [B,16]


def _reference_int8_per_row(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    values32 = values.float()
    scale = values32.abs().amax(dim=-1, keepdim=True) / 127.0
    quantized = torch.round(values32 / scale).clamp(-127, 127)
    return quantized, scale


def test_fake_quant_int8_per_row_is_symmetric_per_row_absmax() -> None:
    torch.manual_seed(0)
    values = (torch.randn(8, 64) * torch.tensor([[0.01], [1.0], [100.0], [3.0], [0.5], [7.0], [1e-3], [42.0]])).to(
        torch.bfloat16
    )
    dequantized = fake_quant_int8(values, per_row=True)
    assert dequantized.dtype == values.dtype
    quantized, scale = _reference_int8_per_row(values)
    # Every output lies on its row's integer grid, within bf16 output rounding.
    codes = dequantized.float() / scale
    assert torch.allclose(codes, torch.round(codes), atol=0.51)
    assert codes.abs().max() <= 127.0 + 0.51
    # The row absmax is a grid point: it round-trips to +-127 * scale exactly (up to bf16).
    row_absmax_idx = values.float().abs().argmax(dim=-1)
    row_absmax = values.float().gather(1, row_absmax_idx[:, None])
    assert torch.allclose(dequantized.float().gather(1, row_absmax_idx[:, None]), row_absmax, rtol=2**-7)
    # Q/DQ error is bounded by half a step of the row's own scale (plus bf16 output rounding).
    error = (dequantized.float() - values.float()).abs()
    assert bool((error <= scale / 2 + values.float().abs() * 2**-7 + 1e-9).all())
    torch.testing.assert_close(dequantized.float(), (quantized * scale).to(values.dtype).float())


def test_fake_quant_int8_per_row_scales_rows_independently() -> None:
    small = torch.full((1, 16), 1e-3)
    large = torch.full((1, 16), 100.0)
    small[0, 0] = 2e-3
    values = torch.cat([small, large])
    per_row = fake_quant_int8(values, per_row=True)
    per_tensor = fake_quant_int8(values, per_row=False)
    # Per-row: the small row keeps its own 127-level grid, so 1e-3 vs 2e-3 stay distinct.
    assert per_row[0, 0] > per_row[0, 1] > 0
    assert torch.allclose(per_row[0], small[0], rtol=1e-2)
    # Per-tensor: one scale from the 100.0 row (step ~0.79) flushes the small row to zero.
    assert torch.all(per_tensor[0] == 0)


def test_fake_quant_fp8_snaps_to_e4m3_grid() -> None:
    torch.manual_seed(1)
    values = (torch.randn(4, 32) * 5).to(torch.bfloat16)
    for per_row in (True, False):
        dequantized = fake_quant_fp8(values, per_row=per_row)
        assert dequantized.dtype == values.dtype
        values32 = values.float()
        amax = values32.abs().amax(dim=-1, keepdim=True) if per_row else values32.abs().amax()
        scale = amax / 448.0
        expected = (values32 / scale).to(torch.float8_e4m3fn).float() * scale
        torch.testing.assert_close(dequantized.float(), expected.to(values.dtype).float())
        assert torch.isfinite(dequantized.float()).all()


def test_apply_quantization_inplace_int8_sim_swaps_only_selected_linears() -> None:
    torch.manual_seed(2)
    model = TinyMixedModel()
    original_selected_weight = model.selected.weight.detach().clone()
    original_bias = model.selected.bias
    inputs = torch.randn(5, 32).to(torch.bfloat16)
    expected_unselected = model.unselected(inputs)
    normed = model.norm(inputs)

    matched = apply_quantization_inplace(model, QuantizationConfig(method="int8_sim", include_regex=["^selected$"]))

    assert matched == ["selected"]
    assert isinstance(model.selected, QdqSimLinear)
    assert type(model.unselected) is nn.Linear
    assert model.selected.qdq_method == "int8_sim" and model.selected.qdq_per_row
    # Same parameter objects, weight rewritten to its per-output-channel Q/DQ image.
    assert model.selected.bias is original_bias
    torch.testing.assert_close(model.selected.weight.float(), fake_quant_int8(original_selected_weight).float())
    assert not torch.equal(model.selected.weight, original_selected_weight)
    # Forward = dense bf16 GEMM over per-token fake-quantized activations.
    expected_selected = nn.functional.linear(fake_quant_int8(normed), model.selected.weight, model.selected.bias)
    torch.testing.assert_close(model(inputs), expected_selected + expected_unselected)


def test_apply_quantization_inplace_fp8_sim_honors_granularity() -> None:
    torch.manual_seed(3)
    for granularity, per_row in (("per_row", True), ("per_tensor", False)):
        model = TinyMixedModel()
        original_weight = model.selected.weight.detach().clone()
        apply_quantization_inplace(
            model, QuantizationConfig(method="fp8_sim", fp8_granularity=granularity, include_regex=["^selected$"])
        )
        assert isinstance(model.selected, QdqSimLinear)
        assert model.selected.qdq_method == "fp8_sim" and model.selected.qdq_per_row is per_row
        torch.testing.assert_close(
            model.selected.weight.float(), fake_quant_fp8(original_weight, per_row=per_row).float()
        )


def test_apply_quantization_inplace_int8_sim_and_fp8_sim_quantize_identical_module_sets(tmp_path: Path) -> None:
    dumps: dict[str, list[str]] = {}
    matched: dict[str, list[str]] = {}
    for method in ("int8_sim", "fp8_sim"):
        model = TinyMixedModel()
        dump_path = tmp_path / f"{method}.txt"
        matched[method] = apply_quantization_inplace(
            model,
            QuantizationConfig(
                method=method,
                include_regex=["selected", "unselected"],
                exclude_regex=["norm"],
                matched_fqns_dump_path=str(dump_path),
            ),
        )
        dumps[method] = dump_path.read_text(encoding="utf-8").splitlines()
    assert matched["int8_sim"] == matched["fp8_sim"] == ["selected", "unselected"]
    assert dumps["int8_sim"] == dumps["fp8_sim"] == ["selected", "unselected"]


def test_apply_quantization_inplace_target_fqns_pin_exact_set() -> None:
    model = TinyMixedModel()
    # Regexes are ignored once target_fqns is set.
    matched = apply_quantization_inplace(
        model, QuantizationConfig(method="int8_sim", include_regex=[".*"], target_fqns=["unselected"])
    )
    assert matched == ["unselected"]
    assert type(model.selected) is nn.Linear
    assert isinstance(model.unselected, QdqSimLinear)


def test_apply_quantization_inplace_target_fqns_accept_vfm_relative_alias() -> None:
    class Outer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = TinyMixedModel()

    outer = Outer()
    matched = apply_quantization_inplace(
        outer, QuantizationConfig(method="int8_sim", target_fqns=["selected", "net.unselected"])
    )
    assert matched == ["net.selected", "net.unselected"]


@pytest.mark.parametrize(
    ("target_fqns", "message"),
    [
        (["selected", "does_not_exist"], "not found in the model"),
        (["selected", "norm"], "not nn.Linear"),
    ],
)
def test_apply_quantization_inplace_target_fqns_reject_missing_or_non_linear(
    target_fqns: list[str], message: str
) -> None:
    model = TinyMixedModel()
    with pytest.raises(ValueError, match=message):
        apply_quantization_inplace(model, QuantizationConfig(method="int8_sim", target_fqns=target_fqns))
    # Validation happens before any swap, so the model is untouched.
    assert type(model.selected) is nn.Linear


def test_apply_quantization_inplace_rejects_double_quantization() -> None:
    model = TinyMixedModel()
    config = QuantizationConfig(method="int8_sim", include_regex=["^selected$"])
    apply_quantization_inplace(model, config)
    with pytest.raises(ValueError, match="already quantized"):
        apply_quantization_inplace(model, config)


def test_qdq_sim_linear_rejects_forward_before_weight_finalized() -> None:
    linear = QdqSimLinear(8, 4, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="not been fake-quantized"):
        linear(torch.zeros(2, 8, dtype=torch.bfloat16))


def test_fake_quant_int8_group_size_scales_each_k_block_independently() -> None:
    torch.manual_seed(4)
    # Two rows x 128 columns; block 1 of row 0 is 1000x larger than block 0.
    values = torch.randn(2, 128)
    values[0, :64] *= 1e-3
    values[0, 64:] *= 1.0
    grouped = fake_quant_int8(values, group_size=64)
    per_row = fake_quant_int8(values, per_row=True)
    # Group-wise: block 0 keeps its own grid (relative error ~< 1/127).
    small_block = values[0, :64]
    assert torch.allclose(grouped[0, :64], small_block, atol=small_block.abs().max() / 127 / 2 + 1e-9)
    # Per-row: block 0 is crushed by block 1's scale (error ~ step of the big block).
    assert (per_row[0, :64] - small_block).abs().max() > 10 * (grouped[0, :64] - small_block).abs().max()
    # Every block's absmax is a grid point and every value lies on its block grid.
    blocks = values.reshape(2, 2, 64)
    scale = blocks.abs().amax(dim=-1, keepdim=True) / 127.0
    codes = grouped.reshape(2, 2, 64) / scale
    assert torch.allclose(codes, torch.round(codes), atol=1e-4)
    assert codes.abs().max() <= 127.0 + 1e-4


def test_fake_quant_group_size_must_divide_k() -> None:
    with pytest.raises(ValueError, match="divide the input dimension"):
        fake_quant_int8(torch.randn(2, 100), group_size=64)
    with pytest.raises(ValueError, match="divide the input dimension"):
        fake_quant_fp8(torch.randn(2, 100), group_size=64)


def test_apply_quantization_inplace_int8_sim_group_size_blocks_weight_and_activation_along_k() -> None:
    torch.manual_seed(5)
    model = TinyMixedModel()  # in_features=32 -> 2 blocks of 16
    original_weight = model.selected.weight.detach().clone()
    apply_quantization_inplace(
        model, QuantizationConfig(method="int8_sim", qdq_group_size=16, include_regex=["^selected$"])
    )
    assert isinstance(model.selected, QdqSimLinear)
    assert model.selected.qdq_group_size == 16
    assert "group16" in repr(model.selected)
    # Weight: one scale per (output channel, 16-wide K block).
    torch.testing.assert_close(model.selected.weight.float(), fake_quant_int8(original_weight, group_size=16).float())
    # Activation: one scale per (token, 16-wide K block), applied per call.
    inputs = torch.randn(3, 32).to(torch.bfloat16)
    normed = model.norm(inputs)
    expected = nn.functional.linear(fake_quant_int8(normed, group_size=16), model.selected.weight, model.selected.bias)
    torch.testing.assert_close(model.selected(normed), expected)


def test_apply_quantization_inplace_group_size_rejects_per_tensor_fp8() -> None:
    model = TinyMixedModel()
    with pytest.raises(ValueError, match="requires fp8_granularity='per_row'"):
        apply_quantization_inplace(
            model,
            QuantizationConfig(
                method="fp8_sim", fp8_granularity="per_tensor", qdq_group_size=16, include_regex=["^selected$"]
            ),
        )
