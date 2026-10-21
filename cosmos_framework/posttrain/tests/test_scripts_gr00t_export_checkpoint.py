# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""L0 tests for the exporter's read half, against real `LLMTrainer.save` output in both checkpoint formats.
The write half needs the official model and lives in the L1/GPU parity test."""

from __future__ import annotations

import pathlib

import pytest
import torch
import torch.nn as nn

from cosmos_framework.posttrain.algorithm.primitives.objective import Objective, Reduction
from cosmos_framework.posttrain.model.gr00t.policy import Gr00tHeads
from cosmos_framework.posttrain.roles.trainer.llm import LLMTrainer
from cosmos_framework.posttrain.scripts.export_gr00t_checkpoint import (
    PROCESSOR_FILES,
    export_checkpoint,
    read_head_state,
    serving_config,
)


class _ToyActionHead(nn.Module):
    def __init__(self, scale: float = 1.0) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 4)
        self.norm = nn.LayerNorm(4)
        with torch.no_grad():
            self.proj.weight.mul_(scale)


def _saved_checkpoint(tmp_path, checkpoint_format: str) -> tuple[str, dict[str, torch.Tensor]]:
    """Train one step and save, returning the iter dir and the state that was written."""
    net = Gr00tHeads(_ToyActionHead(scale=3.0))
    objective = Objective(
        fn=lambda model, batch: (sum(p.sum() for p in model.parameters()), {}), reduction=Reduction.MEAN
    )
    trainer = LLMTrainer(net, objective, torch.optim.SGD(net.parameters(), lr=0.1), checkpoint_format=checkpoint_format)
    trainer.train([])
    iter_dir = tmp_path / "checkpoints" / "iter_000001"
    trainer.save(str(iter_dir))
    return str(iter_dir), {k: v.detach().clone() for k, v in net.state_dict().items()}


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("checkpoint_format", ["consolidated", "dcp"])
def test_read_head_state_round_trips_both_checkpoint_formats(tmp_path, checkpoint_format) -> None:
    iter_dir, saved = _saved_checkpoint(tmp_path, checkpoint_format)
    # A DIFFERENTLY initialized model supplies the template, as the exporter's rebuild does.
    template = Gr00tHeads(_ToyActionHead(scale=1.0)).state_dict()
    assert not torch.equal(template["action_head.proj.weight"], saved["action_head.proj.weight"])

    state = read_head_state(iter_dir, template)
    assert sorted(state) == sorted(saved)
    assert all(torch.equal(state[key], value) for key, value in saved.items())


@pytest.mark.L0
@pytest.mark.CPU
def test_dcp_read_leaves_the_template_untouched(tmp_path) -> None:
    """DCP loads IN PLACE, so a template passed straight through would be silently mutated."""
    iter_dir, _ = _saved_checkpoint(tmp_path, "dcp")
    template = Gr00tHeads(_ToyActionHead(scale=1.0)).state_dict()
    before = {k: v.detach().clone() for k, v in template.items()}
    read_head_state(iter_dir, template)
    assert all(torch.equal(template[key], value) for key, value in before.items())


@pytest.mark.L0
@pytest.mark.CPU
def test_read_head_state_reports_a_directory_with_no_model_component(tmp_path) -> None:
    empty = tmp_path / "iter_000000"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="no model component"):
        read_head_state(str(empty), Gr00tHeads(_ToyActionHead()).state_dict())


@pytest.mark.L0
@pytest.mark.CPU
def test_serving_config_is_the_base_contract_with_only_the_storage_dtype_following_the_export() -> None:
    """The exported config must be the base's (`load_bf16`, `tune_*` as served), never the training factory's."""
    base = {"model_type": "Gr00tN1d7", "dtype": "bfloat16", "load_bf16": True, "tune_llm": True, "tune_visual": True}
    assert serving_config(base, "bf16") == base
    assert serving_config(base, "fp32") == {**base, "dtype": "float32"}
    assert serving_config({"torch_dtype": "bfloat16", "load_bf16": True}, "fp32") == {
        "torch_dtype": "float32",
        "load_bf16": True,
    }
    assert base["dtype"] == "bfloat16", "the base config is copied, not mutated"


def _fake_base(tmp_path) -> tuple[str, dict[str, torch.Tensor]]:
    """A two-shard 'base checkpoint' in the DROID layout: a backbone shard (lm_head included) and the toy head, bf16."""
    import json

    from safetensors.torch import save_file

    base = tmp_path / "base"
    base.mkdir()
    backbone = {
        "backbone.model.embed.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3).to(torch.bfloat16),
        "backbone.model.lm_head.weight": torch.full((2, 3), 0.5, dtype=torch.bfloat16),
    }
    head = {k: v.to(torch.bfloat16) for k, v in Gr00tHeads(_ToyActionHead(scale=1.0)).state_dict().items()}
    save_file(backbone, str(base / "model-00001-of-00002.safetensors"), metadata={"format": "pt"})
    save_file(head, str(base / "model-00002-of-00002.safetensors"), metadata={"format": "pt"})
    weight_map = {k: "model-00001-of-00002.safetensors" for k in backbone}
    weight_map.update({k: "model-00002-of-00002.safetensors" for k in head})
    (base / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 0}, "weight_map": weight_map})
    )
    (base / "config.json").write_text(json.dumps({"model_type": "Gr00tN1d7", "dtype": "bfloat16", "load_bf16": True}))
    for name in PROCESSOR_FILES:
        (base / name).write_text(json.dumps({"file": name, "value": [1.5, 2.5]}))
    return str(base), {**backbone, **head}


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("dtype, head_dtype", [("bf16", torch.bfloat16), ("fp32", torch.float32)])
def test_export_is_the_base_with_only_the_head_replaced(tmp_path, dtype, head_dtype) -> None:
    """Every base tensor survives byte for byte (lm_head included), the head is the trained one in `--dtype`, the
    config and processor files are the base's; nothing goes through a model factory."""
    import json

    from safetensors.torch import load_file

    base, base_tensors = _fake_base(tmp_path)
    iter_dir, trained = _saved_checkpoint(tmp_path, "dcp")
    out = tmp_path / "export"
    export_checkpoint(base=base, posttrain_ckpt=iter_dir, out=str(out), dtype=dtype)

    exported = {}
    for shard in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"):
        exported.update(load_file(str(out / shard)))
    assert set(exported) == set(base_tensors), "the export must carry exactly the base's tensor names"
    for key in ("backbone.model.embed.weight", "backbone.model.lm_head.weight"):
        assert torch.equal(exported[key], base_tensors[key]) and exported[key].dtype == torch.bfloat16
    for key, value in trained.items():
        assert exported[key].dtype == head_dtype
        assert torch.equal(exported[key], value.to(head_dtype)), key
        assert not torch.equal(exported[key], base_tensors[key].to(head_dtype)), (
            "the trained head differs from the base"
        )
    index = json.loads((out / "model.safetensors.index.json").read_text())
    assert (
        index["weight_map"]
        == json.loads((pathlib.Path(base) / "model.safetensors.index.json").read_text())["weight_map"]
    )
    assert index["metadata"]["total_size"] == sum(t.numel() * t.element_size() for t in exported.values())
    config = json.loads((out / "config.json").read_text())
    assert config == {
        "model_type": "Gr00tN1d7",
        "dtype": "bfloat16" if dtype == "bf16" else "float32",
        "load_bf16": True,
    }
    for name in PROCESSOR_FILES:
        assert (out / name).read_bytes() == (pathlib.Path(base) / name).read_bytes()


@pytest.mark.L0
@pytest.mark.CPU
def test_export_refuses_a_head_that_does_not_match_the_base(tmp_path) -> None:
    base, _ = _fake_base(tmp_path)
    net = Gr00tHeads(nn.Linear(4, 4))  # different parameter names than the base's head
    objective = Objective(
        fn=lambda model, batch: (sum(p.sum() for p in model.parameters()), {}), reduction=Reduction.MEAN
    )
    trainer = LLMTrainer(net, objective, torch.optim.SGD(net.parameters(), lr=0.1))
    trainer.train([])
    iter_dir = tmp_path / "other" / "iter_000001"
    trainer.save(str(iter_dir))
    with pytest.raises(RuntimeError, match="does not match the base"):
        export_checkpoint(base=base, posttrain_ckpt=str(iter_dir), out=str(tmp_path / "export"), dtype="bf16")


@pytest.mark.L0
@pytest.mark.CPU
def test_dcp_read_rejects_extra_source_parameters(tmp_path) -> None:
    from cosmos_framework.posttrain.checkpoint.io import save_component_dcp

    source = {"action_head.a": torch.ones(2), "action_head.extra": torch.ones(1)}
    save_component_dcp(str(tmp_path), "model", source)
    template = {"action_head.a": torch.zeros(2)}
    with pytest.raises(RuntimeError, match="extra.*action_head.extra"):
        read_head_state(str(tmp_path), template)
    assert torch.equal(template["action_head.a"], torch.zeros(2))


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("invalid", ["missing", "shape", "non_tensor"])
def test_dcp_read_validates_source_inventory_before_loading(tmp_path, invalid) -> None:
    from cosmos_framework.posttrain.checkpoint.io import save_component_dcp

    source = {"action_head.a": torch.ones(2), "action_head.b": torch.ones(3)}
    if invalid == "missing":
        del source["action_head.b"]
    elif invalid == "shape":
        source["action_head.b"] = torch.ones(4)
    else:
        source["action_head.b"] = "not a tensor"
    save_component_dcp(str(tmp_path), "model", source)
    template = {"action_head.a": torch.zeros(2), "action_head.b": torch.zeros(3)}
    with pytest.raises(RuntimeError, match="DCP head does not match.*action_head.b"):
        read_head_state(str(tmp_path), template)
    assert all(torch.count_nonzero(value) == 0 for value in template.values())


@pytest.mark.L0
@pytest.mark.CPU
def test_dcp_read_allows_the_intended_template_dtype_conversion(tmp_path) -> None:
    from cosmos_framework.posttrain.checkpoint.io import save_component_dcp

    source = {"action_head.a": torch.arange(8, dtype=torch.bfloat16)}
    save_component_dcp(str(tmp_path), "model", source)
    loaded = read_head_state(str(tmp_path), {"action_head.a": torch.empty(8, dtype=torch.float32)})
    assert loaded["action_head.a"].dtype == torch.float32
    assert torch.equal(loaded["action_head.a"], source["action_head.a"].float())
