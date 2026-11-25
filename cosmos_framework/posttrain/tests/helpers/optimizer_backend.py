# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Real native factory/entry routing, with CPU backend doubles only at the CUDA boundary."""

import pytest
import torch


@pytest.fixture
def backend_capture(monkeypatch):
    from cosmos_framework.utils import quantized_optimizer

    calls = []

    def build(net, groups, config, *, _local_owner=None):
        assert _local_owner is not None
        assert all(p.dtype == torch.float32 for g in groups for p in g["params"])
        calls.append((net, groups, config, _local_owner))
        return torch.optim.AdamW(
            groups, lr=config.lr, betas=tuple(config.betas), eps=config.eps, weight_decay=config.weight_decay
        )

    monkeypatch.setattr(quantized_optimizer, "build_quantized_optimizer", build)
    # optimizer_master imports the factory directly; patch its binding too.
    monkeypatch.setattr("cosmos_framework.utils.optimizer_master.build_quantized_optimizer", build)
    monkeypatch.setattr(quantized_optimizer, "require_backend", lambda **kwargs: None)
    return calls
