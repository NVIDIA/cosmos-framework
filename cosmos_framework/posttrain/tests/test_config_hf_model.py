# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
import pytest

from cosmos_framework.posttrain.config.model import HFModelConfig


@pytest.mark.L0
@pytest.mark.CPU
def test_hf_model_config_new_fields_defaults() -> None:
    c = HFModelConfig()
    assert c.grad_ckpt is False
    assert c.fsdp_reduce_dtype == "float32"  # fp32 grad reduction over bf16 params (reference recipe)
    assert c.parallelism.dp_replicate == 1
    assert c.parallelism.dp_shard == -1
    # export_dir/export_every live on RunConfig (config/rl.py), NOT on the model config. Both sides
    # of the merge said "CheckpointConfig" — stale; that class did not survive the landed refactor.
    assert not hasattr(c, "export_dir") and not hasattr(c, "export_every")
    # existing fields intact
    assert c.dtype == "bfloat16" and c.auto_class == "AutoModelForCausalLM"
