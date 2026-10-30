# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Model-owned base autocast and FP32 adapter accumulation are independent."""

import pytest
import torch

from cosmos_framework.posttrain.config.adapters import AdapterComponentConfig
from cosmos_framework.posttrain.model.adapters import PosttrainAdapterLinear


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("method,rs", [("lora", False), ("lora", True), ("dora", False), ("pissa", False)])
def test_fp32_inputs_autocast_base_with_independent_adapter_precision(dtype, method, rs):
    torch.manual_seed(7)
    base = torch.nn.Linear(8, 8, dtype=dtype)
    cfg = AdapterComponentConfig(targets=["action_in_proj"], component="action", rank=2, method=method, use_rslora=rs)
    net = PosttrainAdapterLinear(base, cfg)
    x = torch.randn(3, 8)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        expected = base(x)
        actual = net(x)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    types = []
    hook = net.lora_B.register_forward_hook(lambda _, __, result: types.append(result.dtype))
    optimizer = torch.optim.AdamW(
        [net.lora_A.weight, net.lora_B.weight] + ([net.magnitude] if method == "dora" else []), lr=0.01
    )
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss = net(x).float().square().mean()
        loss.backward()
        optimizer.step()
        assert torch.isfinite(loss)
        assert net.lora_A.weight.grad is not None and net.lora_B.weight.grad is not None
    hook.remove()
    assert types == [torch.float32] * 3
