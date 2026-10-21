# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""FusedAdam (fp32 master + fp32 moments) — the bf16-update-underflow regression.

Root cause this guards (wallace 8B DPO repro, 2026-07-06): plain `torch.optim.AdamW`
on a bf16 net keeps bf16 moments and applies bf16 updates; at posttrain lr the relative
update (lr/|w| ~ 2.5e-4) is below bf16 resolution (2^-8 ~ 3.9e-3), so params never move.
`utils/fused_adam.FusedAdam` accumulates in fp32 masters and must keep learning."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from cosmos_framework.posttrain.config.build import build_optimizer
from cosmos_framework.posttrain.config.rl import OptimizerConfig


@pytest.mark.L0
@pytest.mark.CPU
def test_unknown_optimizer_name_fails_loud():
    net = torch.nn.Linear(4, 4)
    with pytest.raises(ValueError, match="adamw.*fusedadam|fusedadam.*adamw"):
        build_optimizer(net, OptimizerConfig(name="sgd", train_keys=[]))


@pytest.mark.L0
@pytest.mark.GPU
def test_bf16_underflow_adamw_vs_fusedadam():
    """Constant grad, lr far below bf16 resolution of the param: AdamW-bf16 must stall
    (that IS the bug), FusedAdam must keep moving via its fp32 master accumulation."""
    pytest.importorskip("apex")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    lr, steps = 5e-6, 100

    def run(opt_name: str) -> tuple[float, float]:
        p = torch.nn.Parameter(torch.full((256,), 0.02, dtype=torch.bfloat16, device="cuda"))
        start = float(p.detach().float().mean())  # bf16-rounded 0.02, the true anchor
        opt = build_optimizer(
            torch.nn.ParameterList([p]),
            OptimizerConfig(name=opt_name, lr=lr, weight_decay=0.0, betas=[0.9, 0.99], train_keys=[]),
        )
        for _ in range(steps):
            p.grad = torch.ones_like(p)
            opt.step()
            opt.zero_grad()
        return start, float(p.detach().float().mean())

    adamw_start, adamw_end = run("adamw")
    fused_start, fused_end = run("fusedadam")
    # AdamW-bf16: every update rounds away — the param is bit-identical to its start.
    assert adamw_end == pytest.approx(adamw_start, abs=1e-9), "bf16 AdamW unexpectedly moved (env change?)"
    # FusedAdam: ~lr per step accumulated in fp32 → clearly visible drop after `steps`.
    assert fused_start - fused_end > lr * steps * 0.5, f"FusedAdam barely moved: {fused_start} -> {fused_end}"


@pytest.mark.L0
@pytest.mark.GPU
def test_fusedadam_state_is_fp32_and_checkpoint_shaped():
    """Moments must be fp32 tensors shaped like the param (DCP-friendly), and the
    state_dict round-trip must preserve them."""
    pytest.importorskip("apex")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    p = torch.nn.Parameter(torch.randn(8, 8, dtype=torch.bfloat16, device="cuda"))
    opt = build_optimizer(
        torch.nn.ParameterList([p]),
        OptimizerConfig(name="fusedadam", lr=1e-4, train_keys=[]),
    )
    p.grad = torch.randn_like(p)
    opt.step()
    st = opt.state[p]
    assert st["exp_avg"].dtype == torch.float32 and st["exp_avg"].shape == p.shape
    assert st["exp_avg_sq"].dtype == torch.float32

    sd = opt.state_dict()
    opt2 = build_optimizer(
        torch.nn.ParameterList([p]),
        OptimizerConfig(name="fusedadam", lr=1e-4, train_keys=[]),
    )
    opt2.load_state_dict(sd)
    st2 = opt2.state[p]
    assert torch.equal(st2["exp_avg"], st["exp_avg"])
    # Masters rebuild lazily after load (reference semantics) — next step must not crash.
    p.grad = torch.randn_like(p)
    opt2.step()
