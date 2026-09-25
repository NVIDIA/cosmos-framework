# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1


"""Caption loss and recurrent precision regression tests without model downloads."""

import inspect
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict, set_state_dict

from cosmos_framework.model.generator.algorithm.loss import cross_entropy as ce
from cosmos_framework.model.generator.reasoner.qwen35_caption import install_qwen35_fp32_decay, qwen35_fp32_decay_modules

pytestmark = [pytest.mark.L1, pytest.mark.CPU]


class _DecayLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.A_log = nn.Parameter(torch.tensor([0.31415927, 0.92718281]))
        self.in_proj_a = nn.Linear(2, 2, bias=False)
        self.conv1d = nn.Conv1d(2, 2, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return (self.in_proj_a(inputs) + self.conv1d(inputs.T.unsqueeze(0)).squeeze(0).T).float() * self.A_log.exp()


@pytest.mark.parametrize("full_state", [False, True])
def test_fp32_decay_keys_and_optimizer_checkpoint(full_state: bool) -> None:
    torch.manual_seed(19)
    model = _DecayLayer()
    original = {k: v.clone() for k, v in model.state_dict().items()}
    assert install_qwen35_fp32_decay(model) == 1
    assert install_qwen35_fp32_decay(model) == 0
    assert set(model.state_dict()) == set(original)
    assert set(dict(model.named_parameters())) == set(original)
    assert model.A_log.dtype == torch.float32
    model.load_state_dict(original, strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    model(torch.ones(3, 2)).square().sum().backward()
    optimizer.step()
    expected = model(torch.ones(3, 2)).detach()
    options = StateDictOptions(full_state_dict=full_state)
    state, optim_state = get_state_dict(model, optimizer, options=options)
    assert set(optim_state["state"]) == set(original)
    replica = _DecayLayer()
    install_qwen35_fp32_decay(replica)
    replica_optimizer = torch.optim.AdamW(replica.parameters(), lr=0.01)
    set_state_dict(replica, replica_optimizer, model_state_dict=state, optim_state_dict=optim_state, options=options)
    torch.testing.assert_close(replica(torch.ones(3, 2)), expected)
    assert qwen35_fp32_decay_modules(replica)[0].A_log.dtype == torch.float32
    assert replica_optimizer.state_dict()["state"]


@pytest.mark.parametrize("exponent", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("all_ignored", [False, True])
def test_fused_weighted_formula_and_gradients(
    monkeypatch: pytest.MonkeyPatch, exponent: float, all_ignored: bool
) -> None:
    def scalar_ce(hidden: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor, ignore_index: int) -> torch.Tensor:
        return F.cross_entropy(F.linear(hidden, weight).float(), labels, ignore_index=ignore_index, reduction="sum")

    monkeypatch.setattr(ce, "_fused_linear_ce_sum", scalar_ce)
    torch.manual_seed(31)
    hidden = torch.randn(3, 9, 7, requires_grad=True)
    weight = torch.randn(17, 7, requires_grad=True)
    labels = torch.randint(0, 17, (3, 9))
    labels[0] = -100
    labels[1, :5] = -100
    if all_ignored:
        labels[:] = -100
    fused, stats = ce.fused_weighted_cross_entropy_loss(hidden, labels, weight, exponent=exponent)
    reference, reference_stats = ce.weighted_cross_entropy_loss(
        F.linear(hidden, weight), labels, exponent, return_stats=True
    )
    torch.testing.assert_close(fused, reference)
    for actual, expected in zip(
        torch.autograd.grad(fused, (hidden, weight), retain_graph=True),
        torch.autograd.grad(reference, (hidden, weight)),
        strict=True,
    ):
        torch.testing.assert_close(actual, expected)
    for key in stats.__dataclass_fields__:
        torch.testing.assert_close(getattr(stats, key).float(), getattr(reference_stats, key).float())


@pytest.mark.GPU
@pytest.mark.parametrize("exponent", [0.0, 0.5, 1.0])
def test_liger_scalar_ce_gpu_gradients(exponent: float) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Liger")
    torch.manual_seed(55)
    hidden = torch.randn(2, 33, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    labels = torch.randint(0, 128, (2, 33), device="cuda")
    labels[0, :15] = -100
    fused, _ = ce.fused_weighted_cross_entropy_loss(hidden, labels, weight, exponent=exponent)
    reference = ce.weighted_cross_entropy_loss(F.linear(hidden, weight), labels, exponent)
    torch.testing.assert_close(fused, reference, atol=0.02, rtol=0.01)
    for actual, expected in zip(
        torch.autograd.grad(fused, (hidden, weight), retain_graph=True),
        torch.autograd.grad(reference, (hidden, weight)),
        strict=True,
    ):
        torch.testing.assert_close(actual, expected, atol=0.005, rtol=0.05)


def test_caption_forward_preserves_input_signature() -> None:
    from cosmos_framework.model.generator.reasoner.qwen35_caption import configure_qwen35_caption_model

    class Model(nn.Module):
        config = SimpleNamespace(model_type="qwen3_5")

        def forward(
            self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor | None = None
        ) -> torch.Tensor:
            return input_ids

    model = Model()
    original_signature = inspect.signature(model.forward)
    configure_qwen35_caption_model(model)
    assert inspect.signature(model.forward) == original_signature
