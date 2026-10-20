# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Load a local Edge FP8 export through Cosmos Framework's inference loader."""

import os
from pathlib import Path

import pytest
import torch


@pytest.mark.level(2)
@pytest.mark.gpus(1)
def test_edge_fp8_checkpoint_loads_into_framework() -> None:
    checkpoint = os.environ.get("COSMOS3_EDGE_FP8_CHECKPOINT")
    if not checkpoint:
        pytest.skip("Set COSMOS3_EDGE_FP8_CHECKPOINT to a freshly exported Edge FP8 checkpoint")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required to load the full Edge checkpoint")

    from cosmos_framework.inference.args import OmniSetupOverrides
    from cosmos_framework.inference.common.args import CheckpointOverrides
    from cosmos_framework.inference.model import Cosmos3OmniConfig, Cosmos3OmniModel
    from cosmos_framework.utils.generator.quantization import collect_modelopt_fp8_params, is_modelopt_fp8_checkpoint

    checkpoint_path = Path(checkpoint)
    assert is_modelopt_fp8_checkpoint(checkpoint_path)
    checkpoint_args = CheckpointOverrides(checkpoint_path=str(checkpoint_path)).build_checkpoint(
        checkpoints=OmniSetupOverrides.CHECKPOINTS
    )
    config = Cosmos3OmniConfig(model=checkpoint_args.load_model_config_dict())
    model = Cosmos3OmniModel.from_pretrained_dcp(checkpoint_path, config=config)
    fp8_weights = collect_modelopt_fp8_params(model.model.net)
    assert fp8_weights, "The framework loaded no FP8 linears"
    assert all(not weight.is_meta for weight in fp8_weights)
