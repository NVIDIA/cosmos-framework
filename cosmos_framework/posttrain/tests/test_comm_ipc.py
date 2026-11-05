# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""CUDA-IPC cross-process round-trip (GPU only; the real validation of IpcMover).

A second OS process is mandatory: a process cannot open its own CUDA IPC handle (raises
"invalid device context"). The child is a plain ``python -c`` that imports only
``comm.transfer.ipc`` (via PYTHONPATH) — importing this pytest module in a spawned child is
fragile, so we avoid it. The producer keeps the source tensor alive across the call.
"""

from __future__ import annotations

import os
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


@pytest.mark.L0
@pytest.mark.GPU
def test_ipc_cross_process_roundtrip(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("GPU required for CUDA IPC")

    from cosmos_framework.posttrain.comm.transfer.ipc import tensor_to_ipc

    t = torch.arange(8, dtype=torch.float32, device="cuda")
    handle_path = tmp_path / "handle.pkl"
    result_path = tmp_path / "result.pkl"
    with open(handle_path, "wb") as f:
        pickle.dump(tensor_to_ipc(t), f)

    child = (
        "import pickle, torch;"
        "from cosmos_framework.posttrain.comm.transfer.ipc import tensor_from_ipc;"
        f"h = pickle.load(open(r'{handle_path}', 'rb'));"
        f"pickle.dump(tensor_from_ipc(h).cpu(), open(r'{result_path}', 'wb'))"
    )
    # Anchor PYTHONPATH on this file's repo root (comm→posttrain→cosmos3→projects→root), not cwd,
    # so the child import works regardless of where pytest is invoked from.
    repo_root = Path(__file__).resolve().parents[4]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
    # check=True surfaces the child's stderr on failure.
    subprocess.run([sys.executable, "-c", child], check=True, env=env, timeout=120)

    with open(result_path, "rb") as f:
        got = pickle.load(f)
    assert torch.allclose(got, t.cpu())  # same memory, read from another process


@pytest.mark.L0
@pytest.mark.GPU
def test_ipc_backend_cross_process_syncs_model(tmp_path):
    """WeightSync(IpcMover) end-to-end: export handles in this process, apply them into a fresh model
    in a child process, and confirm the child's params now equal the source (a real IPC weight sync)."""
    if not torch.cuda.is_available():
        pytest.skip("GPU required for CUDA IPC")

    from cosmos_framework.posttrain.comm.weight_transfer import make_weight_backend

    src = torch.nn.Linear(16, 16).cuda()
    be = make_weight_backend("ipc")  # WeightSync(IpcMover) — one flat IPC staging allocation
    payload = be.export(src, version=5)
    src_sum = float(sum(p.detach().double().sum().item() for p in src.parameters()))

    payload_path = tmp_path / "payload.pkl"
    result_path = tmp_path / "result.pkl"
    with open(payload_path, "wb") as f:
        pickle.dump(payload, f)

    child = (
        "import pickle, torch, torch.nn as nn;"
        "from cosmos_framework.posttrain.comm.weight_transfer import make_weight_backend;"
        f"payload = pickle.load(open(r'{payload_path}', 'rb'));"
        "dst = nn.Linear(16, 16).cuda();"  # different random init
        "make_weight_backend('ipc').apply(dst, payload);"
        "s = float(sum(p.detach().double().sum().item() for p in dst.parameters()));"
        f"pickle.dump((s, payload.version), open(r'{result_path}', 'wb'))"
    )
    repo_root = Path(__file__).resolve().parents[4]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run([sys.executable, "-c", child], check=True, env=env, timeout=120)

    with open(result_path, "rb") as f:
        dst_sum, version = pickle.load(f)
    assert version == 5
    assert abs(dst_sum - src_sum) < 1e-4  # child's model synced to src via CUDA IPC
