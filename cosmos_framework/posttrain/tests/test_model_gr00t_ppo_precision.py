# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""BF16 PPO precision boundary and native value-head initialization."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from cosmos_framework.posttrain.model.gr00t.ppo import Gr00tPPOHeads

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def _heads() -> Gr00tPPOHeads:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1234)
        head = nn.Linear(2, 2, dtype=torch.bfloat16)
        head.config = SimpleNamespace(
            action_horizon=40, max_action_dim=132, backbone_embedding_dim=2, input_embedding_dim=2
        )
        head.num_inference_timesteps = 4
        return Gr00tPPOHeads(head, score_horizon=8)


def test_non_bf16_ppo_precision_is_rejected_before_loading(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from cosmos_framework.posttrain.configs.gr00t_n1d7_libero_ppo import make_model_config
    from cosmos_framework.posttrain.model.gr00t import policy as gr00t

    def forbidden(*args, **kwargs):
        pytest.fail("invalid PPO precision reached model loading")

    monkeypatch.setattr(gr00t, "build_official_gr00t_model", forbidden)
    with pytest.raises(ValueError, match="GR00T PPO requires BF16_PRECISION"):
        gr00t.Gr00tN1d7TrainPolicy(
            make_model_config("unloaded"),
            device="cpu",
            save_cfg_dir=tmp_path,
            ppo=True,
            precision=gr00t.OFFICIAL_PRECISION,
        )


def test_native_value_initialization_stays_bf16() -> None:
    first, second = _heads(), _heads()
    assert any(name.startswith("action_head.value_head.") for name, _ in first.named_parameters())
    for name, parameter in first.named_parameters():
        assert parameter.dtype == torch.bfloat16
        torch.testing.assert_close(parameter, second.state_dict()[name], rtol=0, atol=0)
