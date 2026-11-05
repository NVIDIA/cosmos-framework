# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""`utils/mesh.py::detect_mesh` — the reader half of the launcher's NNODES/GPUS_PER_NODE contract."""

from __future__ import annotations

import pytest


@pytest.mark.L0
@pytest.mark.CPU
def test_detect_mesh_reads_env_portably(monkeypatch):
    from cosmos_framework.posttrain.utils import mesh as mn

    # replicas = nodes, ranks = GPUs/node — portable across cluster node sizes.
    monkeypatch.setenv("NNODES", "2")
    monkeypatch.setenv("GPUS_PER_NODE", "4")
    assert mn.detect_mesh() == (2, 4)  # 2x 4-GPU nodes ⇒ 8 bundles
    monkeypatch.setenv("GPUS_PER_NODE", "8")
    assert mn.detect_mesh() == (2, 8)  # 2x 8-GPU nodes ⇒ 16 bundles


@pytest.mark.L0
@pytest.mark.CPU
def test_detect_mesh_standalone_fallback_single_node(monkeypatch):
    from cosmos_framework.posttrain.utils import mesh as mn

    monkeypatch.delenv("NNODES", raising=False)
    monkeypatch.setenv("GPUS_PER_NODE", "4")  # avoid depending on the test box's GPU count
    nodes, ranks = mn.detect_mesh()
    assert nodes == 1 and ranks == 4  # no NNODES ⇒ single node


@pytest.mark.L0
@pytest.mark.CPU
def test_detect_mesh_torch_fallback(monkeypatch):
    import torch

    from cosmos_framework.posttrain.utils import mesh as mn

    monkeypatch.delenv("NNODES", raising=False)
    monkeypatch.delenv("GPUS_PER_NODE", raising=False)
    # Patch torch.cuda.device_count inside a context that is undone at the END OF THE
    # TEST BODY, not at fixture teardown: the repo-root conftest's pytest_runtest_teardown
    # hook calls torch.cuda.get_device_properties(0) BEFORE fixtures are finalized — with
    # the function-scoped monkeypatch still active (0 devices) that hook raises, which
    # aborts the remaining teardown hooks, so the patch is NEVER undone and every later
    # test in the session errors at setup ("Invalid device id").
    with monkeypatch.context() as m:
        m.setattr(torch.cuda, "device_count", lambda: 4)
        assert mn.detect_mesh() == (1, 4)  # no env ⇒ locally-visible GPUs, single node
    with monkeypatch.context() as m:
        m.setattr(torch.cuda, "device_count", lambda: 0)
        assert mn.detect_mesh() == (1, 1)  # 0 GPUs ⇒ `or 1` fallback


@pytest.mark.L0
@pytest.mark.CPU
def test_detect_mesh_rejects_zero(monkeypatch):
    from cosmos_framework.posttrain.utils import mesh as mn

    monkeypatch.setenv("NNODES", "2")
    monkeypatch.setenv("GPUS_PER_NODE", "0")
    with pytest.raises(ValueError):
        mn.detect_mesh()
