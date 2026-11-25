# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""CPU tests for the Transfer-Auto FP8 recipe on a tiny synthetic safetensors checkpoint."""

import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from cosmos_framework.quantization import transfer_auto as ta

DEPTH = 2
HIDDEN = 16


def _projection_keys(depth: int) -> list[str]:
    keys = []
    for i in range(depth):
        for leaf in ("to_q", "to_k", "to_v", "to_out", "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out"):
            keys.append(f"layers.{i}.self_attn.{leaf}.weight")
        for mlp in ("mlp", "mlp_moe_gen"):
            for role in ("gate", "up", "down"):
                keys.append(f"layers.{i}.{mlp}.{role}_proj.weight")
    return keys


@pytest.fixture
def source(tmp_path: Path) -> Path:
    """Two-layer BF16 checkpoint in two shards, plus untouched tensors and an auxiliary file."""
    torch.manual_seed(0)
    root = tmp_path / "source"
    (root / "transformer").mkdir(parents=True)
    keys = _projection_keys(DEPTH)
    others = ["embed_tokens.weight", "proj_in.weight", "layers.0.input_layernorm.weight"]
    tensors = {k: (torch.randn(HIDDEN, HIDDEN) * (1 + n % 3)).to(torch.bfloat16) for n, k in enumerate(keys)}
    tensors.update({k: torch.randn(HIDDEN, HIDDEN).to(torch.bfloat16) for k in others})
    names = sorted(tensors)
    half = len(names) // 2
    weight_map = {}
    for n, part in enumerate((names[:half], names[half:]), start=1):
        shard = f"diffusion_pytorch_model-0000{n}-of-00002.safetensors"
        save_file({k: tensors[k] for k in part}, root / "transformer" / shard, metadata={"format": "pt"})
        weight_map.update({k: shard for k in part})
    (root / "transformer/diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map})
    )
    (root / "transformer/config.json").write_text(json.dumps({"num_hidden_layers": DEPTH, "hidden_size": HIDDEN}))
    (root / "model_index.json").write_text('{"_class_name": "Tiny"}')
    return root


@pytest.fixture
def amax(source: Path) -> dict[str, float]:
    return {key: 1.0 + n for n, key in enumerate(_projection_keys(DEPTH))}


def _read(directory: Path) -> dict[str, torch.Tensor]:
    out = {}
    for shard in sorted(directory.glob("*.safetensors")):
        with safe_open(shard, framework="pt") as handle:
            out.update({k: handle.get_tensor(k) for k in handle.keys()})
    return out


def test_projection_map_full_model():
    keys = _projection_keys(36)
    mapping = ta.projection_map(keys, 36, "both")
    assert len(mapping) == 504
    assert set(mapping) == set(keys)
    assert mapping["layers.3.self_attn.to_out.weight"] == "language_model.layers.3.self_attn.to_out"
    assert mapping["layers.3.self_attn.to_q.weight"] == "language_model.layers.3.self_attn.to_q"
    assert mapping["layers.3.self_attn.add_k_proj.weight"] == "gen_layers.3.cross_attention.to_k"
    assert mapping["layers.3.self_attn.to_add_out.weight"] == "gen_layers.3.cross_attention.to_out"
    assert mapping["layers.3.mlp.gate_proj.weight"] == "language_model.layers.3.mlp.gate_proj"
    assert mapping["layers.3.mlp_moe_gen.down_proj.weight"] == "gen_layers.3.mlp.down_proj"
    assert len(ta.projection_map(keys, 36, "gen")) == 252


def test_projection_map_rejects_missing_key_and_bad_arguments():
    keys = _projection_keys(2)
    with pytest.raises(ValueError, match="missing"):
        ta.projection_map(keys[:-1], 2, "both")
    with pytest.raises(ValueError, match="Invalid"):
        ta.projection_map(keys, 0, "both")
    with pytest.raises(ValueError, match="Invalid"):
        ta.projection_map(keys, 2, "all")


def test_export_dtypes_scales_and_unchanged_tensors(source, amax, tmp_path):
    out = tmp_path / "fp8"
    metadata = ta.export_transfer_auto_fp8(source, out, amax, device="cpu")
    src, dst = _read(source / "transformer"), _read(out / "transformer")
    selected = set(_projection_keys(DEPTH))

    assert metadata["quantized_count"] == len(selected)
    assert set(dst) == set(src) | {
        p.removesuffix(".weight") + s for p in selected for s in (".weight_scale", ".input_scale")
    }
    for key, original in src.items():
        if key not in selected:
            assert dst[key].dtype == original.dtype
            assert torch.equal(dst[key], original)
            continue
        prefix = key.removesuffix(".weight")
        assert dst[key].dtype == torch.float8_e4m3fn
        assert dst[key].shape == original.shape
        expected_scale = original.float().abs().max() / 448.0
        assert dst[prefix + ".weight_scale"].dtype == torch.float32
        assert dst[prefix + ".weight_scale"].shape == ()
        assert torch.equal(dst[prefix + ".weight_scale"], expected_scale)
        assert dst[prefix + ".input_scale"].dtype == torch.float32
        assert dst[prefix + ".input_scale"].shape == ()
        assert dst[prefix + ".input_scale"].item() == pytest.approx(amax[key] / 448.0, rel=1e-7)
        reference = (original.float() / expected_scale).clamp(-448, 448).to(torch.float8_e4m3fn)
        assert torch.equal(dst[key].view(torch.uint8), reference.view(torch.uint8))
    # Dequantized weights stay close to the source (E4M3 has a 3-bit mantissa).
    key = "layers.0.self_attn.to_q.weight"
    dequant = dst[key].float() * dst[key.removesuffix(".weight") + ".weight_scale"]
    assert (dequant - src[key].float()).abs().max() <= 0.07 * src[key].float().abs().max()


def test_export_metadata_configs_and_layout(source, amax, tmp_path):
    out = tmp_path / "fp8"
    ta.export_transfer_auto_fp8(source, out, amax, device="cpu")
    # Hard-linked auxiliary files must not be written through into the source.
    assert "quantization_config" not in json.loads((source / "transformer/config.json").read_text())
    config = json.loads((out / "transformer/config.json").read_text())
    quant = config["quantization_config"]
    policy = quant["runtime"]["diffusion_step_policy"]
    assert policy["first_steps"] == {"count": 3, "mode": "a16"}
    assert policy["last_steps"] == {"count": 3, "mode": "a16"}
    assert policy["reasoner"] == "a16"
    assert quant["quant_algo"] == "FP8" and quant["ignore"] == list(ta.IGNORE_PATTERNS)
    assert len(quant["quantized_layers"]) == len(_projection_keys(DEPTH))
    assert "language_model.layers.0.self_attn.to_q" in quant["quantized_layers"]
    assert "gen_layers.1.cross_attention.to_out" in quant["quantized_layers"]

    root_quant = json.loads((out / "hf_quant_config.json").read_text())
    assert "runtime" not in root_quant
    assert {k: v for k, v in quant.items() if k != "runtime"} == root_quant

    transformer_index = json.loads((out / "transformer/diffusion_pytorch_model.safetensors.index.json").read_text())
    root_index = json.loads((out / "model.safetensors.index.json").read_text())
    tensors = _read(out / "transformer")
    assert set(transformer_index["weight_map"]) == set(tensors)
    assert root_index["weight_map"] == {k: f"transformer/{v}" for k, v in transformer_index["weight_map"].items()}
    assert transformer_index["metadata"]["total_size"] == sum(t.numel() * t.element_size() for t in tensors.values())

    metadata = json.loads((out / "quantization_metadata.json").read_text())
    assert metadata["quantized_count"] == len(_projection_keys(DEPTH))
    assert metadata["activation_amax"] == amax
    assert metadata["candidate_changes"]["weight_scales"] == {}
    assert {row["source_weight"] for row in metadata["tensor_coverage"]} == set(_projection_keys(DEPTH))
    assert json.loads((out / "model_index.json").read_text()) == {"_class_name": "Tiny"}
    assert not out.with_name("fp8.partial").exists()


def test_export_applies_weight_scale_map(source, amax, tmp_path):
    key = "layers.1.mlp.up_proj.weight"
    src = _read(source / "transformer")[key]
    scale = float(src.float().abs().max() / 448.0) * 0.8
    out = tmp_path / "fp8"
    metadata = ta.export_transfer_auto_fp8(source, out, amax, weight_scales={key: scale}, device="cpu")
    dst = _read(out / "transformer")
    prefix = key.removesuffix(".weight")
    assert dst[prefix + ".weight_scale"].item() == pytest.approx(scale, rel=1e-7)
    expected = (src.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    assert torch.equal(dst[key].view(torch.uint8), expected.view(torch.uint8))
    other = "layers.1.mlp.down_proj"
    assert dst[other + ".weight_scale"].item() == pytest.approx(
        float(_read(source / "transformer")[other + ".weight"].float().abs().max() / 448.0), rel=1e-7
    )
    assert metadata["candidate_changes"]["weight_scales"] == {key: scale}


def test_export_accepts_json_paths_and_merges_amax(source, amax, tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps({"activation_amax": amax}))
    b.write_text(json.dumps({"activation_amax": {k: 100.0 - v for k, v in amax.items()}}))
    merged = ta.load_activation_amax([a, b])
    assert merged == {k: max(v, 100.0 - v) for k, v in amax.items()}
    out = tmp_path / "fp8"
    ta.export_transfer_auto_fp8(source, out, [a, b], device="cpu")
    key = "layers.0.self_attn.to_q.weight"
    assert _read(out / "transformer")["layers.0.self_attn.to_q.input_scale"].item() == pytest.approx(
        merged[key] / 448.0, rel=1e-7
    )


def test_merge_rejects_mismatched_keys(tmp_path, amax):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps({"activation_amax": amax}))
    b.write_text(json.dumps({"activation_amax": {"layers.0.self_attn.to_q.weight": 1.0}}))
    with pytest.raises(ValueError, match="different projection keys"):
        ta.load_activation_amax([a, b])


def test_export_input_validation(source, amax, tmp_path):
    out = tmp_path / "fp8"
    ta.export_transfer_auto_fp8(source, out, amax, device="cpu")
    with pytest.raises(FileExistsError):
        ta.export_transfer_auto_fp8(source, out, amax, device="cpu")
    with pytest.raises(ValueError, match="pristine"):
        ta.export_transfer_auto_fp8(out, tmp_path / "again", amax, device="cpu")
    missing = {k: v for k, v in amax.items() if k != "layers.0.mlp.up_proj.weight"}
    with pytest.raises(ValueError, match="activation amax"):
        ta.export_transfer_auto_fp8(source, tmp_path / "bad", missing, device="cpu")
    with pytest.raises(ValueError, match="non-quantized"):
        ta.export_transfer_auto_fp8(
            source, tmp_path / "bad2", amax, weight_scales={"embed_tokens.weight": 1.0}, device="cpu"
        )
    with pytest.raises(ValueError, match="overlap"):
        ta.export_transfer_auto_fp8(source, source / "inside", amax, device="cpu")


def test_optimize_weight_scale_prefers_smaller_scale_for_outlier_weight():
    torch.manual_seed(0)
    weight = torch.randn(32, 64) * 1e-3
    weight[0, 0] = 200.0  # one outlier pushes the small weights into the E4M3 subnormal range
    x = torch.randn(128, 64)
    x[:, 0] = 0.0  # inputs never excite the outlier column, so clipping it costs nothing
    ws = float(weight.abs().max() / 448.0)
    best = ta.optimize_weight_scale(x, weight, input_scale=float(x.abs().max() / 448.0), weight_scale=ws)
    assert best["multiplier"] < 1.0
    assert best["scale"] == pytest.approx(ws * best["multiplier"])
    assert best["best_mse"] < best["baseline_mse"]


def test_optimize_weight_scale_keeps_control_on_tie_and_validates():
    weight = torch.full((4, 8), 448.0)  # exactly representable at multiplier 1; any change is worse or equal
    x = torch.ones(3, 8)
    best = ta.optimize_weight_scale(x, weight, input_scale=1.0 / 448.0, weight_scale=1.0)
    assert best["multiplier"] == 1.0 and best["scale"] == 1.0
    with pytest.raises(ValueError):
        ta.optimize_weight_scale(x, weight, input_scale=0.0, weight_scale=1.0)
    with pytest.raises(ValueError):
        ta.optimize_weight_scale(x[:, :4], weight, input_scale=1.0, weight_scale=1.0)
    with pytest.raises(ValueError):
        ta.optimize_weight_scale(x * float("nan"), weight, input_scale=1.0, weight_scale=1.0)


def test_search_weight_scales_writes_map_and_export_uses_it(source, amax, tmp_path):
    torch.manual_seed(1)
    # Plant an outlier so that the search changes the scale of exactly this projection.
    outlier_key = "layers.0.mlp.gate_proj.weight"
    shard = next(p for p in (source / "transformer").glob("*.safetensors") if outlier_key in _read_keys(p))
    tensors = _read_file(shard)
    tensors[outlier_key] = (torch.randn(HIDDEN, HIDDEN) * 1e-3).to(torch.bfloat16)
    tensors[outlier_key][0, 0] = 400.0
    save_file(tensors, shard, metadata={"format": "pt"})

    def rows() -> torch.Tensor:
        value = torch.randn(64, HIDDEN)
        value[:, 0] = 0.0  # the outlier column is never excited, so clipping it costs nothing
        return value

    captures = {f"step0:{key}": rows() for key in _projection_keys(DEPTH)}
    second = {f"step1:{key}": rows() for key in _projection_keys(DEPTH)}
    paths = [tmp_path / "cap0.pt", tmp_path / "cap1.pt"]
    torch.save(captures, paths[0])
    torch.save(second, paths[1])
    amax_file = tmp_path / "amax.json"
    amax_file.write_text(json.dumps({"activation_amax": amax}))

    result = ta.search_weight_scales(source, amax_file, paths, tmp_path / "scales.json", device="cpu")
    assert json.loads((tmp_path / "scales.json").read_text()) == result
    assert outlier_key in result
    original = _read(source / "transformer")[outlier_key].float().abs().max() / 448.0
    assert result[outlier_key] < float(original)
    assert set(result) <= set(_projection_keys(DEPTH))

    out = tmp_path / "fp8"
    ta.export_transfer_auto_fp8(source, out, amax_file, weight_scales=tmp_path / "scales.json", device="cpu")
    dst = _read(out / "transformer")
    assert dst[outlier_key.removesuffix(".weight") + ".weight_scale"].item() == pytest.approx(
        result[outlier_key], rel=1e-7
    )


def test_search_requires_rows_for_every_projection(source, amax, tmp_path):
    only_one = {"layers.0.mlp.gate_proj.weight": torch.randn(8, HIDDEN)}
    torch.save(only_one, tmp_path / "cap.pt")
    with pytest.raises(ValueError, match="No captured rows"):
        ta.search_weight_scales(source, amax, [tmp_path / "cap.pt"], tmp_path / "out.json", device="cpu")


def _read_keys(path: Path) -> set[str]:
    with safe_open(path, framework="pt") as handle:
        return set(handle.keys())


def _read_file(path: Path) -> dict[str, torch.Tensor]:
    with safe_open(path, framework="pt") as handle:
        return {k: handle.get_tensor(k) for k in handle.keys()}
