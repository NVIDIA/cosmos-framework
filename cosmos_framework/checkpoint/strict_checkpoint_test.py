# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Exercise exact regular model initialization through native DCP storage."""

from pathlib import Path

import pytest
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_model_state_dict

from cosmos_framework.checkpoint.strict_checkpoint import load_regular_checkpoint

pytestmark = [pytest.mark.level(0), pytest.mark.gpus(0)]


@pytest.mark.parametrize("compiled_source", [False, True])
@pytest.mark.parametrize("compiled_target", [False, True])
def test_regular_restore_ignores_ema_and_normalizes_compile_names(
    tmp_path: Path, compiled_source: bool, compiled_target: bool
) -> None:
    source: torch.nn.Module = torch.nn.Linear(3, 2)
    target: torch.nn.Module = torch.nn.Linear(3, 2)
    if compiled_source:
        source = torch.compile(source, backend="eager")
    if compiled_target:
        target = torch.compile(target, backend="eager")
    regular = get_model_state_dict(source)  # weight: [2,3], bias: [2]
    ema = {key: torch.full_like(value, 99.0) for key, value in regular.items()}  # weight: [2,3], bias: [2]
    path = tmp_path / "model"
    dcp.save({"net": regular, "ema_net": ema}, checkpoint_id=path)

    load_regular_checkpoint(target, str(path))

    restored = get_model_state_dict(target)  # weight: [2,3], bias: [2]
    assert restored.keys() == regular.keys()
    for key in regular:
        torch.testing.assert_close(restored[key], regular[key], rtol=0, atol=0)
        assert not torch.equal(restored[key], ema[key])


@pytest.mark.parametrize("corruption", ["missing", "unexpected", "shape", "ema_only"])
def test_invalid_regular_checkpoint_fails_before_changing_model(tmp_path: Path, corruption: str) -> None:
    source = torch.nn.Linear(3, 2)
    target = torch.nn.Linear(3, 2)
    regular = get_model_state_dict(source)  # weight: [2,3], bias: [2]
    before = {key: value.clone() for key, value in target.state_dict().items()}  # weight: [2,3], bias: [2]
    if corruption == "missing":
        regular.pop("bias")
    elif corruption == "unexpected":
        regular["extra"] = torch.zeros(1)  # [1]
    elif corruption == "shape":
        regular["weight"] = torch.zeros(3, 3)  # [3,3]
    path = tmp_path / "invalid"
    dcp.save({"ema_net" if corruption == "ema_only" else "net": regular}, checkpoint_id=path)

    with pytest.raises(ValueError, match="exact regular checkpoint"):
        load_regular_checkpoint(target, str(path))

    for key, value in target.state_dict().items():  # weight: [2,3], bias: [2]
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_explicit_ema_prefix_is_rejected() -> None:
    with pytest.raises(ValueError, match="exclusively from regular net weights"):
        load_regular_checkpoint(torch.nn.Linear(3, 2), "unused", prefix="ema_net")
