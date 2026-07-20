# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Pins the actionable k_norm_und_for_gen hint on CustomLoadPlanner's strict
missing-key failure (a flag-ON Edge model warm-starting from a pre-f7f180c2
base DCP). The hint depends on torch's create_default_local_load_plan raising
RuntimeError with the fqn in the message — these tests catch a torch upgrade
or refactor silently dropping the guidance."""

import pytest
import torch
import torch.distributed.checkpoint as dcp

from cosmos_framework.checkpoint.dcp import CustomLoadPlanner

_K_NORM_FQN = "net.language_model.model.layers.0.self_attn.k_norm_und_for_gen.weight"


def _checkpoint_metadata(state_dict, tmp_path):
    """Real Metadata for a checkpoint containing exactly `state_dict`."""
    dcp.save(state_dict, checkpoint_id=str(tmp_path / "ckpt"))
    return dcp.FileSystemReader(str(tmp_path / "ckpt")).read_metadata()


@pytest.mark.L0
@pytest.mark.CPU
def test_missing_k_norm_key_raises_actionable_reconvert_hint(tmp_path):
    metadata = _checkpoint_metadata({"net.other.weight": torch.zeros(2)}, tmp_path)
    model_state_dict = {
        "net.other.weight": torch.zeros(2),
        _K_NORM_FQN: torch.zeros(4),  # flag-ON model requests it; checkpoint lacks it
    }
    planner = CustomLoadPlanner()
    planner.set_up_planner(state_dict=model_state_dict, metadata=metadata, is_coordinator=True)
    with pytest.raises(RuntimeError, match="re-run convert_model_to_dcp"):
        planner.create_local_plan()


@pytest.mark.L0
@pytest.mark.CPU
def test_missing_non_k_norm_key_keeps_bare_error(tmp_path):
    metadata = _checkpoint_metadata({"net.other.weight": torch.zeros(2)}, tmp_path)
    model_state_dict = {
        "net.other.weight": torch.zeros(2),
        "net.unrelated.weight": torch.zeros(4),
    }
    planner = CustomLoadPlanner()
    planner.set_up_planner(state_dict=model_state_dict, metadata=metadata, is_coordinator=True)
    with pytest.raises(RuntimeError) as exc_info:
        planner.create_local_plan()
    assert "re-run convert_model_to_dcp" not in str(exc_info.value)
