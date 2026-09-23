# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
import torch
import torch.distributed.checkpoint as dcp
from torch import nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    set_model_state_dict,
)

from cosmos_framework.checkpoint.reasoner_only import (
    ReasonerCheckpointSource,
    load_reasoner_only_dcp,
)


class _TinyReasoner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(3, 4)
        self.norm = nn.LayerNorm(4)
        self.register_buffer("persistent_scale", torch.ones(1))


class _FullMoTLanguageModel(_TinyReasoner):
    """Training shape: Reasoner state plus source-only Generator/visual state."""

    def __init__(self) -> None:
        super().__init__()
        self.proj_moe_gen = nn.Linear(3, 4)
        self.norm_moe_gen = nn.LayerNorm(4)
        self.visual = nn.Linear(6, 7)


class _TrainingPath(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = _FullMoTLanguageModel()
        self.generator = nn.Linear(5, 6)
        self.vae = nn.Linear(7, 8)


class _TrainingCheckpointModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = _TrainingPath()
        self.net_ema = _TrainingPath()


def _fill_module(module: nn.Module, value: float) -> None:
    with torch.no_grad():
        for tensor in module.state_dict().values():
            if tensor.is_floating_point():
                tensor.fill_(value)


def _save_training_checkpoint(path: Path, model: nn.Module) -> None:
    state = get_model_state_dict(model, options=StateDictOptions(strict=True))
    dcp.save(state_dict=state, checkpoint_id=path)


def _full_dcp_load(path: Path, model: nn.Module) -> None:
    state = get_model_state_dict(model, options=StateDictOptions(strict=True))
    dcp.load(state_dict=state, checkpoint_id=path)
    incompatible = set_model_state_dict(
        model,
        model_state_dict=state,
        options=StateDictOptions(strict=True),
    )
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []


@pytest.mark.level(0)
@pytest.mark.gpus(0)
@pytest.mark.parametrize(
    ("source", "expected_prefix", "expected_value"),
    [
        ("regular", "net.language_model.", 1.0),
        ("ema", "net_ema.language_model.", 2.0),
    ],
)
def test_reasoner_only_dcp_load_matches_full_model_dcp_load(
    tmp_path: Path,
    source: ReasonerCheckpointSource,
    expected_prefix: str,
    expected_value: float,
) -> None:
    source_model = _TrainingCheckpointModel()
    _fill_module(source_model.net, 1.0)
    _fill_module(source_model.net_ema, 2.0)
    checkpoint_path = tmp_path / "model"
    _save_training_checkpoint(checkpoint_path, source_model)

    full_target = _TrainingCheckpointModel()
    _fill_module(full_target, -1.0)
    _full_dcp_load(checkpoint_path, full_target)
    full_reasoner = full_target.net.language_model if source == "regular" else full_target.net_ema.language_model

    reasoner_only_target = _TinyReasoner()
    _fill_module(reasoner_only_target, -2.0)
    result = load_reasoner_only_dcp(
        reasoner_only_target,
        tmp_path,
        source=source,
    )

    assert result.checkpoint_path == checkpoint_path
    assert result.source == source
    assert result.checkpoint_prefix == expected_prefix
    assert result.num_state_leaves == len(reasoner_only_target.state_dict())
    for name, actual in reasoner_only_target.state_dict().items():
        torch.testing.assert_close(actual, full_reasoner.state_dict()[name], rtol=0, atol=0)
        if actual.is_floating_point():
            torch.testing.assert_close(actual, torch.full_like(actual, expected_value), rtol=0, atol=0)


@pytest.mark.level(0)
@pytest.mark.gpus(0)
def test_reasoner_only_dcp_load_rejects_missing_and_unexpected_selected_fqns_before_mutation(
    tmp_path: Path,
) -> None:
    source = _TinyReasoner()
    source_state = source.state_dict()
    checkpoint_state = {
        f"net.language_model.{name}": tensor.clone() for name, tensor in source_state.items() if name != "norm.bias"
    }
    checkpoint_state["net.language_model.not_in_target"] = torch.ones(1)
    checkpoint_state["net.generator.unrelated"] = torch.full((2,), 99.0)
    checkpoint_path = tmp_path / "model"
    dcp.save(state_dict=checkpoint_state, checkpoint_id=checkpoint_path)

    target = _TinyReasoner()
    _fill_module(target, -3.0)
    before = {name: tensor.clone() for name, tensor in target.state_dict().items()}

    with pytest.raises(ValueError, match="Reasoner checkpoint/target FQN mismatch") as error:
        load_reasoner_only_dcp(target, checkpoint_path, source="regular")

    assert "net.language_model.norm.bias" in str(error.value)
    assert "net.language_model.not_in_target" in str(error.value)
    for name, actual in target.state_dict().items():
        torch.testing.assert_close(actual, before[name], rtol=0, atol=0)


@pytest.mark.level(0)
@pytest.mark.gpus(0)
def test_reasoner_only_dcp_load_rejects_target_that_still_contains_generator_tower(tmp_path: Path) -> None:
    source_model = _TrainingCheckpointModel()
    checkpoint_path = tmp_path / "model"
    _save_training_checkpoint(checkpoint_path, source_model)

    with pytest.raises(ValueError, match="include_gen_pathway=False"):
        load_reasoner_only_dcp(_FullMoTLanguageModel(), checkpoint_path, source="regular")


@pytest.mark.level(0)
@pytest.mark.gpus(0)
def test_reasoner_only_dcp_load_never_falls_back_between_regular_and_ema(tmp_path: Path) -> None:
    ema_only = {
        f"net_ema.language_model.{name}": tensor.clone() for name, tensor in _TinyReasoner().state_dict().items()
    }
    checkpoint_path = tmp_path / "model"
    dcp.save(state_dict=ema_only, checkpoint_id=checkpoint_path)

    with pytest.raises(KeyError, match="net.language_model"):
        load_reasoner_only_dcp(_TinyReasoner(), checkpoint_path, source="regular")

    result = load_reasoner_only_dcp(_TinyReasoner(), checkpoint_path, source="ema")
    assert result.source == "ema"


@pytest.mark.level(0)
@pytest.mark.gpus(0)
def test_reasoner_only_dcp_load_rejects_invalid_source_and_non_dcp_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="regular.*ema"):
        load_reasoner_only_dcp(
            _TinyReasoner(),
            tmp_path,
            source=cast(ReasonerCheckpointSource, "automatic"),
        )

    with pytest.raises(FileNotFoundError, match="Could not find DCP metadata"):
        load_reasoner_only_dcp(_TinyReasoner(), tmp_path, source="regular")

    with pytest.raises(ValueError, match="local DCP paths only"):
        load_reasoner_only_dcp(_TinyReasoner(), "s3://bucket/model", source="regular")


@pytest.mark.level(0)
@pytest.mark.gpus(0)
def test_reasoner_only_dcp_load_rejects_meta_target(tmp_path: Path) -> None:
    source_model = _TrainingCheckpointModel()
    checkpoint_path = tmp_path / "model"
    _save_training_checkpoint(checkpoint_path, source_model)

    with torch.device("meta"):
        target = _TinyReasoner()
    with pytest.raises(ValueError, match="must be materialized"):
        load_reasoner_only_dcp(target, checkpoint_path, source="regular")


@pytest.mark.level(0)
@pytest.mark.gpus(0)
def test_reasoner_only_dcp_load_casts_fp32_master_weights_into_bf16_compute_target(tmp_path: Path) -> None:
    source_model = _TrainingCheckpointModel()
    _fill_module(source_model.net, 1.25)
    checkpoint_path = tmp_path / "model"
    _save_training_checkpoint(checkpoint_path, source_model)

    target = _TinyReasoner().to(dtype=torch.bfloat16)
    result = load_reasoner_only_dcp(target, checkpoint_path, source="regular")

    assert result.num_state_leaves == len(target.state_dict())
    for tensor in target.state_dict().values():
        if tensor.is_floating_point():
            assert tensor.dtype == torch.bfloat16
            torch.testing.assert_close(tensor, torch.full_like(tensor, 1.25), rtol=0, atol=0)
