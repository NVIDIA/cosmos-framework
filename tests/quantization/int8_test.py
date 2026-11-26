# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""CPU tests for the Transfer-Auto INT8 W8A8 export, on a tiny synthetic checkpoint.

Run without the repository's training/inference conftest in a CPU environment:
pytest --noconftest -c /dev/null tests/quantization/int8_test.py
"""

import importlib
import json
import sys
import types
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

NUM_LAYERS = 2
HIDDEN = 8


@pytest.fixture()
def int8(monkeypatch):
    """Import ``quantization/int8.py`` without running the package ``__init__``."""
    name = "cosmos_int8_pkg"
    package = types.ModuleType(name)
    package.__path__ = [str(Path(__file__).resolve().parents[2] / "cosmos_framework/quantization")]
    monkeypatch.setitem(sys.modules, name, package)
    module = importlib.import_module(f"{name}.int8")
    yield module
    for key in list(sys.modules):
        if key.startswith(f"{name}."):
            monkeypatch.delitem(sys.modules, key)


def _make_source(root: Path, int8) -> dict[str, torch.Tensor]:
    """Write a 2-layer BF16 checkpoint in two shards; return the full state dict."""
    generator = torch.Generator().manual_seed(0)
    state: dict[str, torch.Tensor] = {}
    for prefix in int8.list_transfer_auto_linears(NUM_LAYERS):
        state[f"{prefix}.weight"] = torch.randn(6, HIDDEN, generator=generator).to(torch.bfloat16)
    state["layers.0.self_attn.to_q.weight"][2] = 0  # all-zero row
    state["layers.1.mlp.up_proj.weight"][0] = 100.0 * torch.randn(HIDDEN, generator=generator)  # outlier row
    for i in range(NUM_LAYERS):
        state[f"layers.{i}.input_layernorm.weight"] = torch.randn(HIDDEN, generator=generator).to(torch.bfloat16)
    state["proj_in.weight"] = torch.randn(HIDDEN, 3, generator=generator).to(torch.bfloat16)
    state["proj_in.bias"] = torch.randn(HIDDEN, generator=generator).to(torch.bfloat16)
    state["embed_tokens.weight"] = torch.randn(5, HIDDEN, generator=generator).to(torch.bfloat16)

    transformer = root / "transformer"
    transformer.mkdir(parents=True)
    keys = list(state)
    halves = [keys[: len(keys) // 2], keys[len(keys) // 2 :]]
    weight_map = {}
    for n, half in enumerate(halves, start=1):
        name = f"diffusion_pytorch_model-0000{n}-of-00002.safetensors"
        save_file({k: state[k] for k in half}, str(transformer / name), metadata={"format": "pt"})
        weight_map.update(dict.fromkeys(half, name))
    (transformer / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 0}, "weight_map": weight_map})
    )
    (transformer / "config.json").write_text(json.dumps({"num_hidden_layers": NUM_LAYERS, "hidden_size": HIDDEN}))
    (root / "model_index.json").write_text("{}")
    (root / "vae").mkdir()
    (root / "vae/w.bin").write_bytes(b"vae")
    return state


def _read_all(directory: Path) -> dict[str, torch.Tensor]:
    out = {}
    for shard in sorted(directory.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt") as f:
            for key in f.keys():  # noqa: SIM118
                out[key] = f.get_tensor(key)
    return out


def test_linear_list_has_504_for_36_layers(int8):
    linears = int8.list_transfer_auto_linears(36)
    assert len(linears) == 504
    assert sum(info["tower"] == "und" for info in linears.values()) == 252
    modules = {info["module"] for info in linears.values()}
    assert len(modules) == 504
    assert "language_model.layers.35.self_attn.to_out" in modules
    assert "gen_layers.0.cross_attention.to_q" in modules
    assert "gen_layers.35.mlp.down_proj" in modules


def test_quantize_weight_math(int8):
    weight = torch.tensor(
        [
            [127.0, -63.5, 0.5, 1.5],  # scale 1.0: ties round half to even (-64, 0, 2)
            [0.0, 0.0, 0.0, 0.0],  # all-zero row: scale 1.0, q 0
            [-2.0, 1.0, 0.0, 2.0],  # scale 2/127
        ],
        dtype=torch.float32,
    )
    q, scale = int8.quantize_weight_int8(weight)
    assert q.dtype == torch.int8 and q.shape == (3, 4)
    assert scale.dtype == torch.float32 and scale.shape == (3, 1)
    assert scale[:, 0].tolist() == [1.0, 1.0, pytest.approx(2.0 / 127)]
    assert q[0].tolist() == [127, -64, 0, 2]
    assert q[1].tolist() == [0, 0, 0, 0]
    assert q[2].tolist() == [-127, 64, 0, 127]  # 1/(2/127) = 63.5 -> 64 (half to even)


def test_quantize_weight_matches_vllm_kernel_tie_rounding(int8):
    # 0.0242919921875 is exactly amax / 2. In fp32, w * (127 / amax) = 63.499996, so the
    # vLLM kernel gives 63; w / scale = 63.5 would round half to even to 64.
    amax = 0.048583984375
    weight = torch.tensor([[amax, 0.0242919921875, -0.0242919921875]], dtype=torch.bfloat16)
    q, _ = int8.quantize_weight_int8(weight)
    assert q[0].tolist() == [127, 63, -63]


def test_quantize_weight_stays_in_range_and_dequantizes_close(int8):
    weight = torch.randn(64, 128, generator=torch.Generator().manual_seed(1)).to(torch.bfloat16)
    q, scale = int8.quantize_weight_int8(weight)
    assert q.min() >= -127 and q.max() <= 127
    assert (q.abs().amax(dim=1) == 127).all()
    error = (q.float() * scale - weight.float()).abs()
    assert (error <= scale / 2 + 1e-6).all()


def test_export_both_towers(int8, tmp_path):
    source, output = tmp_path / "src", tmp_path / "out"
    state = _make_source(source, int8)
    recipe = int8.export_transfer_auto_int8(source, output)

    assert recipe["quantized_linears"] == 14 * NUM_LAYERS
    exported = _read_all(output / "transformer")
    linears = int8.list_transfer_auto_linears(NUM_LAYERS)
    assert len(exported) == len(state) + len(linears)
    for prefix in linears:
        q, scale = int8.quantize_weight_int8(state[f"{prefix}.weight"])
        assert exported[f"{prefix}.weight"].dtype == torch.int8
        assert torch.equal(exported[f"{prefix}.weight"], q)
        assert exported[f"{prefix}.weight_scale"].dtype == torch.float32
        assert exported[f"{prefix}.weight_scale"].shape == (6, 1)
        assert torch.equal(exported[f"{prefix}.weight_scale"], scale)
        assert f"{prefix}.input_scale" not in exported
    # Other tensors are byte-identical.
    for key, tensor in state.items():
        if key.removesuffix(".weight") not in linears:
            assert exported[key].dtype == tensor.dtype
            assert torch.equal(exported[key], tensor)
    # The all-zero row has scale 1.0.
    assert exported["layers.0.self_attn.to_q.weight_scale"][2].item() == 1.0
    assert exported["layers.0.self_attn.to_q.weight"][2].abs().sum().item() == 0


def test_export_writes_config_index_and_links(int8, tmp_path):
    source, output = tmp_path / "src", tmp_path / "out"
    _make_source(source, int8)
    int8.export_transfer_auto_int8(source, output)

    config = json.loads((output / "transformer/config.json").read_text())
    assert config["hidden_size"] == HIDDEN
    quant = config["quantization_config"]
    assert quant.pop("runtime")["diffusion_step_policy"]["reasoner"] == "a16"  # default: MAPS policy
    assert quant == {
        "quant_method": "int8",
        "is_checkpoint_int8_serialized": True,
        "activation_scheme": "dynamic",
        "ignored_layers": [],
    }
    index = json.loads((output / "transformer/diffusion_pytorch_model.safetensors.index.json").read_text())
    assert set(index["weight_map"]) == set(_read_all(output / "transformer"))
    assert "layers.1.mlp_moe_gen.down_proj.weight_scale" in index["weight_map"]
    root_index = json.loads((output / "model.safetensors.index.json").read_text())
    assert set(root_index["weight_map"]) == set(index["weight_map"])
    assert (output / "vae/w.bin").read_bytes() == b"vae"
    assert (output / "model_index.json").is_symlink()
    metadata = json.loads((output / "quantization_metadata.json").read_text())
    assert metadata["recipe"]["format"] == "int8_w8a8"
    assert metadata["recipe"]["quantized_linears"] == 14 * NUM_LAYERS
    # The source is not modified.
    assert not (source / "transformer/config.json").read_text().count("quantization_config")


@pytest.mark.parametrize("scope, tower_kept_bf16", [("und", "gen"), ("gen", "und")])
def test_export_single_tower_lists_ignored_layers(int8, tmp_path, scope, tower_kept_bf16):
    source, output = tmp_path / "src", tmp_path / "out"
    state = _make_source(source, int8)
    recipe = int8.export_transfer_auto_int8(source, output, scope=scope)

    linears = int8.list_transfer_auto_linears(NUM_LAYERS)
    kept = {p: i for p, i in linears.items() if i["tower"] == tower_kept_bf16}
    assert recipe["quantized_linears"] == 7 * NUM_LAYERS
    config = json.loads((output / "transformer/config.json").read_text())
    assert sorted(config["quantization_config"]["ignored_layers"]) == sorted(i["module"] for i in kept.values())
    exported = _read_all(output / "transformer")
    for prefix in kept:
        assert exported[f"{prefix}.weight"].dtype == torch.bfloat16
        assert torch.equal(exported[f"{prefix}.weight"], state[f"{prefix}.weight"])
        assert f"{prefix}.weight_scale" not in exported


def test_export_rejects_bad_input(int8, tmp_path):
    source = tmp_path / "src"
    _make_source(source, int8)
    with pytest.raises(ValueError, match="scope"):
        int8.export_transfer_auto_int8(source, tmp_path / "o1", scope="all")
    with pytest.raises(FileNotFoundError):
        int8.export_transfer_auto_int8(tmp_path / "missing", tmp_path / "o2")
    # A source that already has a quantization_config is refused.
    config_path = source / "transformer/config.json"
    config = json.loads(config_path.read_text())
    config["quantization_config"] = {"quant_method": "modelopt"}
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="BF16"):
        int8.export_transfer_auto_int8(source, tmp_path / "o3")


def test_export_rejects_missing_linear(int8, tmp_path):
    source = tmp_path / "src"
    _make_source(source, int8)
    config_path = source / "transformer/config.json"
    config_path.write_text(json.dumps({"num_hidden_layers": NUM_LAYERS + 1}))
    with pytest.raises(ValueError, match="missing"):
        int8.export_transfer_auto_int8(source, tmp_path / "out")


def _amax_map(int8, state, value=None):
    return {
        f"{p}.weight": (value if value is not None else 1.0 + i)
        for i, p in enumerate(int8.list_transfer_auto_linears(NUM_LAYERS))
    }


def test_static_input_scales(int8, tmp_path):
    source, output = tmp_path / "src", tmp_path / "out"
    state = _make_source(source, int8)
    amax = _amax_map(int8, state)
    recipe = int8.export_transfer_auto_int8(source, output, activation_amax=amax)

    exported = _read_all(output / "transformer")
    for prefix in int8.list_transfer_auto_linears(NUM_LAYERS):
        scale = exported[f"{prefix}.input_scale"]
        assert scale.dtype == torch.float32 and scale.shape == ()
        assert scale.item() == pytest.approx(amax[f"{prefix}.weight"] / 127)
        assert f"{prefix}.pre_quant_scale" not in exported
        # Weights are the same as in the dyn variant.
        q, w_scale = int8.quantize_weight_int8(state[f"{prefix}.weight"])
        assert torch.equal(exported[f"{prefix}.weight"], q)
        assert torch.equal(exported[f"{prefix}.weight_scale"], w_scale)
    config = json.loads((output / "transformer/config.json").read_text())
    assert config["quantization_config"]["activation_scheme"] == "static"
    assert "smoothquant" not in config["quantization_config"]
    assert recipe["variant"] == "static"
    metadata = json.loads((output / "quantization_metadata.json").read_text())
    assert metadata["recipe"]["variant"] == "static"


def test_static_needs_every_amax(int8, tmp_path):
    source = tmp_path / "src"
    state = _make_source(source, int8)
    amax = _amax_map(int8, state)
    amax.pop("layers.0.self_attn.to_q.weight")
    with pytest.raises(ValueError, match="activation_amax"):
        int8.export_transfer_auto_int8(source, tmp_path / "out", activation_amax=amax)
    with pytest.raises(ValueError, match="static_activations"):
        int8.export_transfer_auto_int8(source, tmp_path / "out2", static_activations=True)


def test_load_activation_amax(int8, tmp_path):
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"activation_amax": {"a.weight": 2.5}, "other": 1}))
    assert int8.load_activation_amax(path) == {"a.weight": 2.5}
    path.write_text(json.dumps({"activation_amax": {"a.weight": float("inf")}}))
    with pytest.raises(ValueError, match="Invalid"):
        int8.load_activation_amax(path)


def _channel_amax(int8, state):
    gen = torch.Generator().manual_seed(5)
    out = {}
    for prefix in int8.list_transfer_auto_linears(NUM_LAYERS):
        ch = 0.1 + torch.rand(HIDDEN, generator=gen)
        ch[1] = 50.0  # outlier channel
        out[f"{prefix}.weight"] = ch
    return out


def test_smoothquant_equivalence_with_outliers(int8):
    gen = torch.Generator().manual_seed(3)
    k, n = 64, 32
    x = torch.randn(256, k, generator=gen)
    x[:, 5] *= 60.0  # outlier channels
    x[:, 17] *= 30.0
    w = torch.randn(n, k, generator=gen) * 0.05
    ch = x.abs().amax(dim=0)
    s = int8.smoothquant_scale(ch, w, 0.5)
    assert s.dtype == torch.float32 and s.shape == (k,)
    q, w_scale = int8.quantize_weight_int8(w * s)
    pre = 1.0 / s
    ref = x @ w.T
    out = (x * pre) @ (q.float() * w_scale).T
    rel = (out - ref).norm() / ref.norm()
    assert rel < 0.02
    # The smoothing without quantization is exact up to FP32 rounding.
    assert torch.allclose((x * pre) @ (w * s).T, ref, rtol=1e-4, atol=1e-4)
    # Smoothing flattens the activation channel range.
    assert (ch / s).amax() / (ch / s).median() < ch.amax() / ch.median()
    # Plain per-channel INT8 of W is not worse by more than the smoothed one on the weight side only.
    q0, s0 = int8.quantize_weight_int8(w)
    assert ((x @ (q0.float() * s0).T - ref).norm() / ref.norm()) < 0.02


def test_smoothquant_scale_formula_and_clamp(int8):
    ch = torch.tensor([4.0, 0.0, 0.0, 1e9, 1e-14])
    w = torch.tensor([[1.0, 1.0, 0.0, 1e-9, 1.0], [0.25, 0.5, 0.0, 1e-9, 1.0]])
    s = int8.smoothquant_scale(ch, w, 0.5)
    assert s[0].item() == pytest.approx(2.0 / 1.0**0.5)  # sqrt(4) / sqrt(1)
    assert s[1].item() == pytest.approx(1e-5)  # zero activation: clamped low
    assert s[2].item() == 1.0  # zero activation and zero weight
    assert s[3].item() == pytest.approx(1e5)  # clamped high
    assert s[4].item() == pytest.approx(1e-5)
    assert int8.smoothed_activation_amax(ch[:1], s[:1]) == pytest.approx(2.0)
    with pytest.raises(ValueError, match="alpha"):
        int8.smoothquant_scale(ch, w, 1.5)
    with pytest.raises(ValueError, match="channel_amax"):
        int8.smoothquant_scale(ch[:3], w, 0.5)


@pytest.mark.parametrize("static", [False, True])
def test_export_smoothquant(int8, tmp_path, static):
    source, output = tmp_path / "src", tmp_path / "out"
    state = _make_source(source, int8)
    ch = _channel_amax(int8, state)
    recipe = int8.export_transfer_auto_int8(source, output, smoothquant=(ch, 0.5), static_activations=static)

    exported = _read_all(output / "transformer")
    for prefix in int8.list_transfer_auto_linears(NUM_LAYERS):
        w = state[f"{prefix}.weight"]
        s = int8.smoothquant_scale(ch[f"{prefix}.weight"], w, 0.5)
        pre = exported[f"{prefix}.pre_quant_scale"]
        assert pre.dtype == torch.float32 and pre.shape == (HIDDEN,)
        assert torch.allclose(pre, 1.0 / s)
        q, w_scale = int8.quantize_weight_int8(w.float() * s)
        assert torch.equal(exported[f"{prefix}.weight"], q)
        assert torch.equal(exported[f"{prefix}.weight_scale"], w_scale)
        if static:
            expected = float((ch[f"{prefix}.weight"] / s).amax()) / 127
            assert exported[f"{prefix}.input_scale"].item() == pytest.approx(expected)
        else:
            assert f"{prefix}.input_scale" not in exported
    config = json.loads((output / "transformer/config.json").read_text())
    assert config["quantization_config"]["smoothquant"] == {"alpha": 0.5}
    assert config["quantization_config"]["activation_scheme"] == ("static" if static else "dynamic")
    assert recipe["variant"] == ("sq-static" if static else "sq-dyn")
    # Other tensors are untouched.
    for key, tensor in state.items():
        if key.removesuffix(".weight") not in int8.list_transfer_auto_linears(NUM_LAYERS):
            assert torch.equal(exported[key], tensor)


def test_dyn_export_is_unchanged_by_the_variant_options(int8, tmp_path):
    source = tmp_path / "src"
    state = _make_source(source, int8)
    int8.export_transfer_auto_int8(source, tmp_path / "a", mixed_precision_steps=0, reasoner="native")
    int8.export_transfer_auto_int8(
        source, tmp_path / "b", activation_amax=None, smoothquant=None, mixed_precision_steps=0, reasoner="native"
    )
    a, b = _read_all(tmp_path / "a/transformer"), _read_all(tmp_path / "b/transformer")
    assert a.keys() == b.keys()
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert not any(k.endswith(("input_scale", "pre_quant_scale")) for k in a)
    assert len(a) == len(state) + len(int8.list_transfer_auto_linears(NUM_LAYERS))
    config = json.loads((tmp_path / "a/transformer/config.json").read_text())
    assert config["quantization_config"] == {
        "quant_method": "int8",
        "is_checkpoint_int8_serialized": True,
        "activation_scheme": "dynamic",
        "ignored_layers": [],
    }
    assert json.loads((tmp_path / "a/quantization_metadata.json").read_text())["recipe"]["variant"] == "dyn"


POLICY_KEYS = {
    "schema_version", "type", "index_space", "scope", "default_mode",
    "first_steps", "last_steps", "overlap", "reasoner",
}  # fmt: skip


def _policy(int8, tmp_path, name, **kwargs):
    source = tmp_path / "src"
    if not source.exists():
        _make_source(source, int8)
    int8.export_transfer_auto_int8(source, tmp_path / name, **kwargs)
    config = json.loads((tmp_path / name / "transformer/config.json").read_text())
    return config["quantization_config"]


def test_default_export_writes_the_maps_policy(int8, tmp_path):
    quant = _policy(int8, tmp_path, "out")
    assert quant["runtime"] == {
        "diffusion_step_policy": {
            "schema_version": 1,
            "type": "first_last_n",
            "index_space": "denoising_loop_iteration",
            "scope": ["transformer"],
            "default_mode": "native",
            "first_steps": {"count": 3, "mode": "a16"},
            "last_steps": {"count": 3, "mode": "a16"},
            "overlap": "a16",
            "reasoner": "a16",
        }
    }
    assert set(quant["runtime"]["diffusion_step_policy"]) == POLICY_KEYS
    metadata = json.loads((tmp_path / "out/quantization_metadata.json").read_text())
    assert metadata["recipe"]["quantization_config"]["runtime"] == quant["runtime"]


def test_custom_policy_steps_and_reasoner(int8, tmp_path):
    policy = _policy(int8, tmp_path, "out", mixed_precision_steps=5, reasoner="native")["runtime"][
        "diffusion_step_policy"
    ]
    assert policy["first_steps"] == {"count": 5, "mode": "a16"}
    assert policy["last_steps"] == {"count": 5, "mode": "a16"}
    assert policy["reasoner"] == "native"


def test_zero_steps_with_a16_reasoner_keeps_the_policy(int8, tmp_path):
    policy = _policy(int8, tmp_path, "out", mixed_precision_steps=0)["runtime"]["diffusion_step_policy"]
    assert policy["first_steps"]["count"] == 0 and policy["reasoner"] == "a16"


def test_no_policy_for_zero_steps_and_native_reasoner(int8, tmp_path):
    assert "runtime" not in _policy(int8, tmp_path, "out", mixed_precision_steps=0, reasoner="native")


def test_policy_arguments_are_validated(int8, tmp_path):
    source = tmp_path / "src"
    _make_source(source, int8)
    with pytest.raises(ValueError, match="mixed_precision_steps"):
        int8.export_transfer_auto_int8(source, tmp_path / "o", mixed_precision_steps=-1)
    with pytest.raises(ValueError, match="reasoner"):
        int8.export_transfer_auto_int8(source, tmp_path / "o", reasoner="fp8")
