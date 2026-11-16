# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Test the fresh-EMA callback policy with small CPU modules, not DCP/FSDP."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.callbacks.initialize_ema import InitializeEmaFromRegular

pytestmark = [pytest.mark.level(0), pytest.mark.gpus(0)]


class _Worker:
    def __init__(self) -> None:
        self.copies: int = 0

    def copy_to(self, src_model: torch.nn.Module, tgt_model: torch.nn.Module) -> None:
        self.copies += 1
        tgt_model.load_state_dict(src_model.state_dict(), strict=True)


def _model() -> SimpleNamespace:
    net = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
    ema = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
    worker = _Worker()
    syncs: list[bool] = []
    return SimpleNamespace(
        config=SimpleNamespace(ema=SimpleNamespace(enabled=True)),
        net=net,
        net_ema=ema,
        net_ema_worker=worker,
        syncs=syncs,
        sync_ema_buffers=lambda: syncs.append(True),
    )


def test_fresh_load_copies_after_loading_regular_weights() -> None:
    model = _model()
    with torch.no_grad():
        for parameter in model.net.parameters():
            parameter.fill_(0.125)
    InitializeEmaFromRegular("/checkpoints/base/").on_load_checkpoint_end(
        model, iteration=0, checkpoint_path="/checkpoints/base"
    )
    for name, value in model.net.state_dict().items():
        torch.testing.assert_close(model.net_ema.state_dict()[name], value, rtol=0, atol=0)
    assert model.net_ema_worker.copies == 1
    assert model.syncs == [True]


@pytest.mark.parametrize(
    "path",
    [None, "/checkpoints/other", "/checkpoints/base/checkpoints/iter_bad", "/checkpoints/base-prefix"],
)
def test_undeclared_source_cannot_initialize_ema(path: str | None) -> None:
    model = _model()
    with pytest.raises(RuntimeError, match="declared base checkpoint"):
        InitializeEmaFromRegular("/checkpoints/base").on_load_checkpoint_end(model, checkpoint_path=path)
    assert model.net_ema_worker.copies == 0


def test_a_full_state_resume_preserves_ema() -> None:
    model = _model()
    InitializeEmaFromRegular("/checkpoints/base").on_load_checkpoint_end(
        model, iteration=8, checkpoint_path="/outputs/stage0/checkpoints/iter_000000008"
    )
    assert model.net_ema_worker.copies == 0
    assert model.syncs == []


def test_same_job_iteration_zero_checkpoint_preserves_ema() -> None:
    model = _model()
    callback = InitializeEmaFromRegular("/checkpoints/base")
    callback.trainer = SimpleNamespace(checkpointer=SimpleNamespace(save_dirname="/outputs/stage0/checkpoints"))
    callback.on_load_checkpoint_end(
        model,
        iteration=0,
        checkpoint_path="/outputs/stage0/checkpoints/iter_000000000",
    )
    assert model.net_ema_worker.copies == 0
    assert model.syncs == []


def test_explicit_resume_configuration_disables_fresh_copy() -> None:
    model = _model()
    InitializeEmaFromRegular("/checkpoints/base", enabled=False).on_load_checkpoint_end(
        model, iteration=8, checkpoint_path="/outputs/stage0/checkpoints/iter_000000008"
    )
    assert model.net_ema_worker.copies == 0
    assert model.syncs == []


def test_disabled_ema_is_not_a_valid_stage0_warm_start() -> None:
    model = _model()
    model.config.ema.enabled = False
    with pytest.raises(RuntimeError, match="keep EMA enabled"):
        InitializeEmaFromRegular("/checkpoints/base").on_load_checkpoint_end(model, checkpoint_path="/checkpoints/base")
