# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for torchrun ranks and offline launcher-template math."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nccl_allreduce_probe as probe  # noqa: E402


def test_global_size_is_nodes_times_gpus():
    cfg = probe.rendezvous_config({"NNODES": "2", "NPROC_PER_NODE": "8"})
    assert cfg["global_world_size"] == 16  # 2 nodes * 8 gpus, NOT 2
    assert cfg["node_count"] == 2 and cfg["gpus_per_node"] == 8


def test_global_rank_math():
    # node 1, local gpu 3, 8 gpus/node -> global rank 11
    cfg = probe.rendezvous_config({"NNODES": "4", "NPROC_PER_NODE": "8", "NODE_RANK": "1", "LOCAL_RANK": "3"})
    assert cfg["global_rank"] == 11
    assert cfg["global_world_size"] == 32


def test_rank_zero_on_first_node_first_gpu():
    cfg = probe.rendezvous_config({"NNODES": "2", "NPROC_PER_NODE": "8", "NODE_RANK": "0", "LOCAL_RANK": "0"})
    assert cfg["global_rank"] == 0


def test_last_rank():
    cfg = probe.rendezvous_config({"NNODES": "2", "NPROC_PER_NODE": "8", "NODE_RANK": "1", "LOCAL_RANK": "7"})
    assert cfg["global_rank"] == cfg["global_world_size"] - 1  # 15


def test_torchrun_ranks_need_no_custom_aliases():
    cfg = probe.rendezvous_config(
        {
            "WORLD_SIZE": "16",  # torchrun's global process count
            "RANK": "11",
            "LOCAL_RANK": "3",
            "LOCAL_WORLD_SIZE": "8",
        }
    )
    assert cfg["node_count"] == 2
    assert cfg["global_world_size"] == 16
    assert cfg["global_rank"] == 11


def test_torchrun_values_take_precedence_over_launcher_defaults():
    cfg = probe.rendezvous_config(
        {
            "WORLD_SIZE": "8",
            "RANK": "5",
            "LOCAL_WORLD_SIZE": "4",
            "LOCAL_RANK": "1",
            "NNODES": "1",
            "NPROC_PER_NODE": "1",
        }
    )
    assert cfg["global_world_size"] == 8
    assert cfg["global_rank"] == 5
    assert cfg["node_count"] == 2


@pytest.mark.parametrize(
    "env",
    [
        {"WORLD_SIZE": "0"},
        {"LOCAL_WORLD_SIZE": "0"},
        {"WORLD_SIZE": "3", "LOCAL_WORLD_SIZE": "2"},
        {"RANK": "1"},
        {"LOCAL_RANK": "-1"},
    ],
)
def test_invalid_rendezvous_is_rejected(env):
    with pytest.raises(ValueError):
        probe.rendezvous_config(env)


def test_master_addr_port_passthrough():
    cfg = probe.rendezvous_config({"MASTER_ADDR": "cosmos3-0.cosmos3", "MASTER_PORT": "29500"})
    assert cfg["master_addr"] == "cosmos3-0.cosmos3" and cfg["master_port"] == "29500"


def test_defaults_single_process():
    cfg = probe.rendezvous_config({})
    assert cfg["global_world_size"] == 1 and cfg["global_rank"] == 0


def test_cli_dry_run_no_torch(monkeypatch, capsys):
    monkeypatch.setenv("NNODES", "2")
    monkeypatch.setenv("NPROC_PER_NODE", "4")
    monkeypatch.setenv("NODE_RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "2")
    assert probe.main(["--dry-run"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["global_world_size"] == 8 and out["global_rank"] == 6
