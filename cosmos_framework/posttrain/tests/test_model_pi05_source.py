# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Strict source inventory and per-parameter publication for mixed-precision policies."""

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from cosmos_framework.posttrain.comm.weight_transfer import model_manifest
from cosmos_framework.posttrain.model.pi05.config import Pi05ModelConfig
from cosmos_framework.posttrain.model.pi05.source import Pi05Source, load_model_weights

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def _model() -> torch.nn.Module:
    model = torch.nn.Module()
    model.embed = torch.nn.Embedding(5, 3, dtype=torch.bfloat16)
    model.lm_head = torch.nn.Linear(3, 5, bias=False, dtype=torch.bfloat16)
    model.lm_head.weight = model.embed.weight
    model.value_head = torch.nn.Linear(3, 1)
    return model


def _source(directory: Path, tensors: dict[str, torch.Tensor]) -> Pi05Source:
    save_file(tensors, str(directory / "model.safetensors"))
    (directory / "tokenizer.model").write_bytes(b"test tokenizer bytes")
    (directory / "norm_stats.json").write_text("{}")
    return Pi05Source.resolve(
        Pi05ModelConfig(str(directory), str(directory / "tokenizer.model"), str(directory / "norm_stats.json"))
    )


def test_tied_source_alias_loads_both_names_and_keeps_native_wire_dtypes(tmp_path: Path) -> None:
    source_model = _model()
    source = _source(
        tmp_path, {name: tensor for name, tensor in source_model.state_dict().items() if name != "embed.weight"}
    )
    destination = _model()
    load_model_weights(destination, source)
    for name, tensor in source_model.state_dict().items():
        torch.testing.assert_close(destination.state_dict()[name], tensor, rtol=0, atol=0)
    assert destination.embed.weight is destination.lm_head.weight
    dtypes = {name: dtype for name, _, dtype in model_manifest(destination, transfer_dtype=None)}
    assert dtypes["embed.weight"] == "torch.bfloat16"
    assert dtypes["value_head.weight"] == "torch.float32"


@pytest.mark.parametrize("corruption", ["extra", "partial_value", "missing_actor", "wrong_shape", "alias_conflict"])
def test_bad_source_inventory_changes_no_destination_parameter(tmp_path: Path, corruption: str) -> None:
    source_model = _model()
    weights = {name: value.clone() for name, value in source_model.state_dict().items() if name != "embed.weight"}
    if corruption == "extra":
        weights["unknown.weight"] = torch.ones(2)
    elif corruption == "partial_value":
        weights.pop("value_head.bias")
    elif corruption == "missing_actor":
        weights.pop("lm_head.weight")
    elif corruption == "wrong_shape":
        weights["value_head.weight"] = torch.ones(2, 3)
    else:
        weights["embed.weight"] = weights["lm_head.weight"] + 1
    source = _source(tmp_path, weights)
    destination = _model()
    before = {name: value.clone() for name, value in destination.state_dict().items()}
    with pytest.raises(ValueError):
        load_model_weights(destination, source)
    for name, value in destination.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_source_identity_binds_normalization_and_tokenizer_bytes(tmp_path: Path) -> None:
    source = _source(tmp_path, {"lm_head.weight": _model().lm_head.weight.detach()})
    initial = source.identity()
    source.normalization.write_text('{"changed":true}')
    changed_stats = source.identity()
    assert changed_stats != initial
    source.tokenizer.write_bytes(b"other tokenizer bytes")
    assert source.identity() != changed_stats


def test_source_identity_revalidates_inventory_after_resolution(tmp_path: Path) -> None:
    source = _source(tmp_path, {"lm_head.weight": _model().lm_head.weight.detach()})
    source.identity()
    save_file({"extra.weight": torch.ones(2)}, str(tmp_path / "extra.safetensors"))
    with pytest.raises(ValueError, match="exactly one model.safetensors"):
        source.identity()
