# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from cosmos_framework.utils import distributed


def test_init_uses_gloo_backend_for_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    init_process_group = Mock()
    monkeypatch.setenv("COSMOS_DEVICE", "cpu")
    monkeypatch.setenv("TORCH_NCCL_BLOCKING_WAIT", "0")
    monkeypatch.setenv("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    monkeypatch.setattr(distributed, "INTERNAL", False)
    monkeypatch.setattr(distributed.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(distributed.dist, "is_available", lambda: True)
    monkeypatch.setattr(distributed.dist, "init_process_group", init_process_group)
    monkeypatch.setattr(distributed.pynvml, "nvmlInit", lambda: None)
    monkeypatch.setattr(
        distributed,
        "Device",
        lambda _local_rank: SimpleNamespace(get_cpu_affinity=lambda: [0]),
    )
    monkeypatch.setattr(distributed.os, "sched_setaffinity", lambda _pid, _affinity: None)
    monkeypatch.setattr(distributed.torch.cuda, "set_device", lambda _local_rank: None)

    distributed.init()

    assert init_process_group.call_args.kwargs["backend"] == "gloo"
