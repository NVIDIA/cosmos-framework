# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
import dataclasses

import pytest

torch = pytest.importorskip("torch")
from cosmos_framework.posttrain.model.specs import LoadPlan, ShardPlan


@pytest.mark.L0
@pytest.mark.CPU
def test_shard_plan_fields_and_frozen():
    import torch.nn as nn

    p = ShardPlan(block_types=(nn.Linear,), param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    assert p.block_types == (nn.Linear,)
    assert p.param_dtype is torch.bfloat16 and p.reduce_dtype is torch.float32
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.param_dtype = torch.float16  # type: ignore[misc]


@pytest.mark.L0
@pytest.mark.CPU
def test_load_plan_defaults_and_rename():
    lp = LoadPlan()
    assert lp.rename is None and lp.skip_patterns == ()
    lp2 = LoadPlan(rename=lambda k: k.replace("a.", "b."), skip_patterns=(r"vision\.",))
    assert lp2.rename("a.x") == "b.x" and lp2.skip_patterns == (r"vision\.",)
