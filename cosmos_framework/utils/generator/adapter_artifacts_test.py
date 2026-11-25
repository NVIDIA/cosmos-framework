# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file

from cosmos_framework.utils.generator.adapter_artifacts import (
    MANIFEST,
    WEIGHTS,
    adapter_identity,
    compatibility_differences,
    export_merged_adapter,
    load_adapter_artifact,
    read_adapter_artifact,
    write_adapter_artifact,
)
from cosmos_framework.utils.generator.adapter_config import AdapterComponentConfig, AdapterConfig
from cosmos_framework.utils.generator.adapters import (
    adapter_state_spec,
    install_adapters,
    merged_target_weight,
    model_base_fingerprint,
)

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


class _Text(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(model_type="qwen3")
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(nn.Module() for _ in range(2))
        for layer in self.model.layers:
            layer.q_proj = nn.Linear(8, 8, bias=False)
            layer.v_proj = nn.Linear(8, 6, bias=False)
        self.lm_head = nn.Linear(8, 10, bias=False)


def _adapted(
    method: str = "dora", base_dtype: torch.dtype = torch.float32, *, base_seed: int = 0, **settings
) -> nn.Module:
    torch.manual_seed(base_seed)
    net = _Text().to(base_dtype)
    install_adapters(
        net,
        AdapterConfig(
            components=[
                AdapterComponentConfig(targets=["model.layers.*.q_proj"], method=method, rank=2, alpha=4.0, **settings),
                AdapterComponentConfig(targets=["model.layers.1.v_proj"], method="pissa", rank=2, alpha=4.0),
            ]
        ),
    )
    generator = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for parameter in net.parameters():
            if parameter.requires_grad:
                parameter.add_(0.1 * torch.randn(parameter.shape, generator=generator))
    net.adapter_identity = adapter_identity(
        net.adapter_manifest,
        base_dtype=str(base_dtype).removeprefix("torch."),
        base_fingerprint=model_base_fingerprint(net, net.adapter_manifest),
    )
    return net


def _artifact(net: nn.Module, path: Path) -> Path:
    state = net.state_dict()
    write_adapter_artifact(
        str(path), net.adapter_identity, {n: state[n] for n in adapter_state_spec(net.adapter_identity)}
    )
    return path


def _snapshot(net: nn.Module, path: Path, **config) -> Path:
    path.mkdir()
    base = {n: t for n, t in net.state_dict().items() if not any(k in n for k in ("lora_", "initial_"))}
    save_file({n: t.contiguous() for n, t in base.items()}, str(path / "model.safetensors"))
    (path / "config.json").write_text(json.dumps({"model_type": "qwen3", **config}))
    (path / "tokenizer.json").write_text("{}")
    (path / "extra").mkdir()
    (path / "extra" / "notes.txt").write_text("kept")
    (path / "extra" / "legacy.bin").write_bytes(b"dropped")
    (path / "pytorch_model.bin.index.json").write_text("{}")
    return path


def test_artifact_round_trip_and_warm_start_load(tmp_path: Path) -> None:
    net = _adapted()
    _artifact(net, tmp_path / "adapter")
    assert {p.name for p in (tmp_path / "adapter").iterdir()} == {MANIFEST, WEIGHTS}
    with pytest.raises(FileExistsError):
        _artifact(net, tmp_path / "adapter")
    identity, state = read_adapter_artifact(str(tmp_path / "adapter"))
    assert identity == net.adapter_identity and identity["format"] == "cosmos-adapter-v1"

    # Training-only settings and the base storage dtype may differ for a warm start.
    other = _adapted(base_dtype=torch.bfloat16, dropout=0.1, lr_multiplier=2.0)
    load_adapter_artifact(other, str(tmp_path / "adapter"))
    for name, tensor in state.items():
        torch.testing.assert_close(other.state_dict()[name], tensor, atol=0, rtol=0)
    with pytest.raises(ValueError, match=r"model\.layers\.0\.q_proj\.method: 'dora' \(saved\) != 'lora'"):
        load_adapter_artifact(_adapted("lora"), str(tmp_path / "adapter"))
    assert compatibility_differences(identity, {**identity, "base_dtype": "bfloat16"}) == []
    # The fingerprint ignores the storage dtype (bfloat16 copy above) but not a different base.
    with pytest.raises(ValueError, match="trained on a different base"):
        load_adapter_artifact(_adapted(base_seed=1), str(tmp_path / "adapter"))
    # A different target set is reported as such, not as a different base.
    fewer = _adapted()
    fewer.adapter_identity = {**fewer.adapter_identity, "targets": fewer.adapter_identity["targets"][:1]}
    with pytest.raises(ValueError, match="incompatible with the configured adapters: targets: unexpected"):
        load_adapter_artifact(fewer, str(tmp_path / "adapter"))


def test_malformed_and_tampered_artifacts_are_refused(tmp_path: Path) -> None:
    net = _adapted()
    state = {n: t for n, t in net.state_dict().items() if n in adapter_state_spec(net.adapter_identity)}
    with pytest.raises(ValueError, match="finite float32"):
        write_adapter_artifact(
            str(tmp_path / "bad"), net.adapter_identity, {**state, next(iter(state)): torch.ones(2, 8)}
        )
    broken = {**net.adapter_identity, "targets": [{**net.adapter_identity["targets"][0], "rank": "2"}]}
    with pytest.raises(ValueError, match=r"targets\[0\] has invalid fields: \['rank'\]"):
        write_adapter_artifact(str(tmp_path / "bad"), broken, state)
    for alpha in (True, float("inf"), float("nan"), 4):
        bad_alpha = {**net.adapter_identity, "targets": [{**net.adapter_identity["targets"][0], "alpha": alpha}]}
        with pytest.raises(ValueError, match=r"invalid fields: \['alpha'\]"):
            write_adapter_artifact(str(tmp_path / "bad"), bad_alpha, state)
    first = net.adapter_identity["targets"][0]
    duplicate = {**net.adapter_identity, "targets": [first, first]}
    with pytest.raises(ValueError, match="more than once"):
        write_adapter_artifact(str(tmp_path / "bad"), duplicate, state)
    path = _artifact(net, tmp_path / "adapter")
    (path / WEIGHTS).write_bytes((path / WEIGHTS).read_bytes()[:-1] + b" ")
    with pytest.raises(ValueError, match="digest"):
        read_adapter_artifact(str(path))
    (path / MANIFEST).write_text("{not json")
    with pytest.raises(ValueError, match="not valid JSON"):
        read_adapter_artifact(str(path))


@pytest.mark.parametrize("base_dtype", [torch.float32, torch.bfloat16])
def test_merged_export(base_dtype: torch.dtype, tmp_path: Path) -> None:
    net = _adapted(base_dtype=base_dtype)
    adapter = _artifact(net, tmp_path / "adapter")
    base = _snapshot(net, tmp_path / "base", text_config={"dtype": "bfloat16"})
    export_merged_adapter(str(adapter), str(base), str(tmp_path / "merged"), dtype="float32")
    merged = load_file(str(tmp_path / "merged" / "model.safetensors"))
    state = net.state_dict()
    for row in net.adapter_identity["targets"]:
        name = f"{row['name']}.weight"
        torch.testing.assert_close(merged[name], merged_target_weight(row, state[name], state), atol=0, rtol=0)
    assert torch.equal(merged["lm_head.weight"], state["lm_head.weight"].float())
    config = json.loads((tmp_path / "merged" / "config.json").read_text())
    assert config["torch_dtype"] == "float32" and config["text_config"]["dtype"] == "float32"
    assert (tmp_path / "merged" / "tokenizer.json").is_file() and (
        tmp_path / "merged" / "extra" / "notes.txt"
    ).is_file()
    assert not (tmp_path / "merged" / "extra" / "legacy.bin").exists()
    assert not (tmp_path / "merged" / "pytorch_model.bin.index.json").exists()

    export_merged_adapter(str(adapter), str(base), str(tmp_path / "merged_default"))  # default: training dtype
    assert load_file(str(tmp_path / "merged_default" / "model.safetensors"))["lm_head.weight"].dtype == base_dtype


def test_merged_export_refuses_mismatched_bases(tmp_path: Path) -> None:
    net = _adapted()
    adapter = _artifact(net, tmp_path / "adapter")
    with pytest.raises(ValueError, match="model_type 'llama' differs"):
        export_merged_adapter(
            str(adapter), str(_snapshot(net, tmp_path / "llama", model_type="llama")), str(tmp_path / "a")
        )
    reshaped = _Text()
    reshaped.model.layers[0].q_proj = nn.Linear(8, 4, bias=False)
    reshaped.model.layers[1].q_proj = nn.Linear(8, 4, bias=False)
    with pytest.raises(ValueError, match=r"model\.layers\.0\.q_proj\.weight has shape \[4, 8\]"):
        export_merged_adapter(str(adapter), str(_snapshot(reshaped, tmp_path / "reshaped")), str(tmp_path / "b"))
    assert not (tmp_path / "b").exists()


def test_merged_export_reproduces_the_trained_function(tmp_path: Path) -> None:
    """A float32 snapshot, adapters trained on its bfloat16 copy: every tensor is rounded to bfloat16 first."""
    trained = _adapted(base_dtype=torch.bfloat16)
    snapshot = _snapshot(_adapted(base_dtype=torch.float32), tmp_path / "base")
    export_merged_adapter(
        str(_artifact(trained, tmp_path / "adapter")), str(snapshot), str(tmp_path / "merged"), dtype="float32"
    )
    merged, state = load_file(str(tmp_path / "merged" / "model.safetensors")), trained.state_dict()
    assert torch.equal(merged["lm_head.weight"], state["lm_head.weight"].float())
    for row in trained.adapter_identity["targets"]:
        name = f"{row['name']}.weight"
        torch.testing.assert_close(merged[name], merged_target_weight(row, state[name], state), atol=0, rtol=0)


def test_merged_export_checks_the_base_fingerprint(tmp_path: Path) -> None:
    adapter = _artifact(_adapted(), tmp_path / "adapter")
    other = _snapshot(_adapted(base_seed=1), tmp_path / "other")
    before = sorted(path.name for path in tmp_path.iterdir())
    with pytest.raises(ValueError, match="does not match the weights the adapter was trained on"):
        export_merged_adapter(str(adapter), str(other), str(tmp_path / "refused"))
    assert sorted(path.name for path in tmp_path.iterdir()) == before  # nothing written, not even staging
    export_merged_adapter(str(adapter), str(other), str(tmp_path / "forced"), allow_base_mismatch=True)
    assert (tmp_path / "forced" / "model.safetensors").is_file()


def test_merged_export_reports_targets_missing_from_the_shards(tmp_path: Path) -> None:
    net = _adapted()
    adapter = _artifact(net, tmp_path / "adapter")
    base = _snapshot(net, tmp_path / "base")
    state = load_file(str(base / "model.safetensors"))
    del state["model.layers.0.q_proj.weight"]
    save_file(state, str(base / "model.safetensors"))
    # The index still lists the removed target weight.
    weight_map = {name: "model.safetensors" for name in [*state, "model.layers.0.q_proj.weight"]}
    (base / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    with pytest.raises(
        ValueError,
        match=r"shards lack adapter target weights listed in its index: \['model\.layers\.0\.q_proj\.weight'\]",
    ):
        export_merged_adapter(str(adapter), str(base), str(tmp_path / "merged"))
