# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""Supervised-position projection preserves causal context, masking and updates."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from cosmos_framework.posttrain.algorithm.primitives.objective import OBJECTIVE_WEIGHT_SUM
from cosmos_framework.posttrain.algorithm.primitives.policy_forward import (
    policy_logits,
    selected_sft_forward,
    selected_sft_logits,
)
from cosmos_framework.posttrain.algorithm.sft import make_objective
from cosmos_framework.posttrain.model.protocol import SelectedSFTPolicyOutput
from cosmos_framework.posttrain.model.qwen3_vl import Qwen3VLPolicy
from cosmos_framework.posttrain.roles.trainer.llm import LLMTrainer
from cosmos_framework.posttrain.tests.helpers.sft_objective import MemoryPacker
from cosmos_framework.utils.generator.lora import LoraInjectedLinear

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


class _CausalBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(16, 8).requires_grad_(False)
        self.projection = LoraInjectedLinear(nn.Linear(8, 8, bias=False).requires_grad_(False), rank=2, alpha=2)
        self.projection.reset_adapter_parameters(generator=torch.Generator().manual_seed(10))
        self.last_batch = None
        self.last_projected = None

    def forward(self, input_ids, attention_mask, *, use_cache=False, return_dict=True, **media):
        assert not use_cache and return_dict
        self.last_batch = {"input_ids": input_ids, "attention_mask": attention_mask, **media}
        embedded = self.embedding(input_ids)
        if media:
            embedded = embedded + torch.stack([value["features"] for value in media["media"]])[:, None, :]
        projected = self.projection(embedded)
        if projected.requires_grad:
            projected.retain_grad()
        self.last_projected = projected
        # A genuine causal dependency: every prediction consumes all preceding context.
        return SimpleNamespace(last_hidden_state=(projected * attention_mask[:, :, None]).cumsum(dim=1))


class _RecordingHead(nn.Linear):
    def __init__(self):
        super().__init__(8, 16, bias=False)
        self.shapes = []

    def forward(self, hidden):
        self.shapes.append(tuple(hidden.shape))
        return super().forward(hidden)


class _CausalNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _CausalBackbone()
        self.lm_head = _RecordingHead().requires_grad_(False)
        self.config = SimpleNamespace(tie_word_embeddings=False)

    def forward(self, **batch):
        hidden = self.model(**batch).last_hidden_state
        return SimpleNamespace(logits=self.lm_head(hidden))


def _policies():
    torch.manual_seed(123)
    net = _CausalNet()
    return Qwen3VLPolicy(copy.deepcopy(net), "qwen3_vl"), Qwen3VLPolicy(net, "qwen3_vl", sft_logits_mode="supervised")


def _rows(packer):
    return [
        packer.add([1, 2, 3], [-100, -100, 3], weight=1, mm_kwargs={"media": {"features": torch.arange(8) / 8}}),
        packer.add([4, 5, 6, 7, 8], [-100, 5, -100, 7, 8], weight=1, mm_kwargs={"media": {"features": torch.ones(8)}}),
        packer.add([9, 10, 11, 12], [-100, 10, 11, 12], weight=0, mm_kwargs={"media": {"features": -torch.ones(8)}}),
    ]


@pytest.mark.parametrize("ce_exponent", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("padding", [0, 3])
def test_selected_head_matches_full_loss_gradients_and_multiple_adapter_updates(ce_exponent, padding):
    full, selected = _policies()
    packer = MemoryPacker(padding=padding)
    rows = _rows(packer)
    objective = make_objective(packer, ce_exponent=ce_exponent)
    optimizers = [
        torch.optim.SGD([p for p in policy.net.parameters() if p.requires_grad], lr=0.02) for policy in (full, selected)
    ]
    frozen = {name: p.detach().clone() for name, p in selected.net.named_parameters() if not p.requires_grad}
    initial = {name: p.detach().clone() for name, p in selected.net.named_parameters() if p.requires_grad}
    for step in range(3):
        results = []
        for policy, optimizer in zip((full, selected), optimizers, strict=True):
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = objective(policy, rows)
            loss.backward()
            results.append((loss.detach(), metrics))
        torch.testing.assert_close(results[0][0], results[1][0])
        assert results[0][1].keys() == results[1][1].keys()
        for key in results[0][1]:
            torch.testing.assert_close(torch.as_tensor(results[0][1][key]), torch.as_tensor(results[1][1][key]))
        assert results[1][1]["sft_valid_tokens"] == 4
        assert results[1][1]["sft_valid_sequences"] == 2
        assert selected.net.lm_head.shapes[-1] == (4, 8)
        assert full.net.lm_head.shapes[-1] == (3, 5 + padding, 8)
        for a, b in zip(full.net.parameters(), selected.net.parameters(), strict=True):
            if a.requires_grad:
                assert a.grad is not None and b.grad is not None
                torch.testing.assert_close(a.grad, b.grad, atol=2e-7, rtol=2e-5)
            else:
                assert a.grad is None and b.grad is None
        # Unsupervised prompt positions still affect later supervised predictions.
        assert selected.net.model.last_projected.grad[0, :2].abs().sum() > 0
        assert selected.net.model.last_projected.grad[2].count_nonzero() == 0
        if step == 0:
            assert selected.net.model.projection.lora_A.weight.grad.count_nonzero() == 0
        for optimizer in optimizers:
            optimizer.step()
    for name, parameter in selected.net.named_parameters():
        if parameter.requires_grad:
            assert not torch.equal(parameter, initial[name])
        else:
            assert torch.equal(parameter, frozen[name])


def test_selected_head_retains_full_logit_api_and_real_media_conditioning():
    full, selected = _policies()
    ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "media": [{"features": torch.ones(8)}] * 2}
    valid = torch.tensor([[False, True], [True, False]])
    projected = selected.forward_sft_batch(batch, valid).logits
    ordinary = policy_logits(selected.forward_policy_batch(batch), ids)
    torch.testing.assert_close(projected, ordinary[:, :-1][valid])
    torch.testing.assert_close(ordinary, full.forward_policy_batch(batch).logits)
    assert selected.net.model.last_batch["media"] is batch["media"]
    text_only = selected.forward_policy_batch({key: value for key, value in batch.items() if key != "media"}).logits
    assert not torch.equal(ordinary, text_only)


def test_selected_head_uses_both_attention_positions_and_excludes_dummy_rows(monkeypatch):
    _, selected = _policies()
    packer = MemoryPacker()
    real = packer.add([0, 1, 2, 0], [0, 1, 2, 0], weight=1)
    dummy = packer.add([3, 4, 5, 6], [3, 4, 5, 6], weight=0)
    collate = packer.sft_collate_fn

    def padded(*args):
        batch = collate(*args)
        batch["attention_mask"][0] = torch.tensor([0, 1, 1, 0])
        return batch

    monkeypatch.setattr(packer, "sft_collate_fn", padded)
    loss, metrics = make_objective(packer)(selected, [real, dummy])
    assert metrics["sft_valid_tokens"] == 1
    assert selected.net.lm_head.shapes[-1] == (1, 8)
    loss.backward()
    assert selected.net.model.last_projected.grad[1].count_nonzero() == 0
    assert selected.net.model.last_projected.grad[0, [0, 2, 3]].count_nonzero() == 0


def test_zero_weight_local_lane_preserves_adapter_backward_graph(monkeypatch):
    from cosmos_framework.posttrain.algorithm import sft

    _, selected = _policies()
    packer = MemoryPacker()
    dummy = packer.add([1, 2, 3], [-100, 2, 3], weight=0)
    monkeypatch.setattr(sft, "rank_sum_count", lambda *_: 2)
    monkeypatch.setattr(sft, "world_size_or_1", lambda: 2)
    loss, metrics = make_objective(packer)(selected, [dummy])
    assert loss.item() == 0
    assert metrics["sft_valid_tokens"] == 0 and metrics[OBJECTIVE_WEIGHT_SUM] == 1
    assert selected.net.lm_head.shapes[-1] == (0, 8)
    loss.backward()
    for parameter in selected.net.parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None and parameter.grad.count_nonzero() == 0


def test_selected_accumulation_matches_full_global_token_mean():
    full, selected = _policies()
    packer = MemoryPacker(padding=1)
    rows = _rows(packer)[:2]
    objective = make_objective(packer)
    optimizer = torch.optim.SGD([p for p in selected.net.parameters() if p.requires_grad], lr=0.1)
    control_optimizer = torch.optim.SGD([p for p in full.net.parameters() if p.requires_grad], lr=0.1)
    loss, metrics = objective(full, rows)
    loss.backward()
    control_optimizer.step()
    trainer = LLMTrainer(selected, objective, optimizer, grad_accum=2)
    for row in rows:
        trainer.train([row])
    assert trainer.iteration() == 1
    assert trainer._denom_at_last_step == metrics[OBJECTIVE_WEIGHT_SUM]
    for actual, reference in zip(selected.net.parameters(), full.net.parameters(), strict=True):
        torch.testing.assert_close(actual, reference)


@pytest.mark.parametrize("valid", [torch.ones(1, 3, dtype=torch.bool), torch.ones(1, 2)])
def test_selected_invalid_masks_fail_before_forward(valid):
    _, selected = _policies()
    with pytest.raises(ValueError, match=r"bool \[B,T-1\]"):
        selected.forward_sft_batch({"input_ids": torch.tensor([[1, 2, 3]])}, valid)
    assert selected.net.model.last_batch is None


def test_selected_refuses_trainable_head_and_invalid_capability():
    _, selected = _policies()
    selected.net.lm_head.weight.requires_grad_(True)
    with pytest.raises(ValueError, match="frozen output head"):
        selected.forward_sft_batch({"input_ids": torch.tensor([[1, 2, 3]])}, torch.ones(1, 2, dtype=torch.bool))
    assert selected.net.model.last_batch is None
    with pytest.raises(TypeError, match="SelectedSFTPolicy"):
        selected_sft_forward(SimpleNamespace(sft_logits_mode="supervised"))
    with pytest.raises(ValueError, match="unsupported SFT logits mode"):
        selected_sft_forward(SimpleNamespace(sft_logits_mode="unknown"))
    assert selected_sft_forward(SimpleNamespace()) is None


@pytest.mark.parametrize("shape", [(2, 3, 16), (2, 16), (3, 0)])
def test_selected_output_contract_rejects_wrong_projection(shape):
    with pytest.raises(ValueError, match=r"logits \[N,V\]"):
        selected_sft_logits(SelectedSFTPolicyOutput(torch.zeros(shape)), 3)
    with pytest.raises(ValueError, match=r"logits \[B,L,V\]"):
        policy_logits(SelectedSFTPolicyOutput(torch.zeros(3, 16)), torch.zeros(1, 4))


def test_actual_native_tiny_multimodal_backbone_preserves_loss_and_gradients():
    from cosmos_framework.posttrain.model.qwen3_vl import (
        Qwen3VLBuildHooks,
        build_qwen3_vl_policy,
        override_qwen3_vl_forward,
    )
    from cosmos_framework.posttrain.tests.test_model_qwen3_vl_gpu import _batch, _dense_config

    def prepare(net):
        net.requires_grad_(False)
        override_qwen3_vl_forward(net.model)
        attention = net.model.language_model.layers[0].self_attn
        attention.q_proj = LoraInjectedLinear(attention.q_proj, rank=2, alpha=2, adapter_dtype=torch.float32)
        attention.q_proj.reset_adapter_parameters(generator=torch.Generator().manual_seed(9))

    policy = build_qwen3_vl_policy(
        model_type="qwen3_vl",
        model_config=_dense_config(),
        dtype=torch.float32,
        initialize_on_meta=False,
        hooks=Qwen3VLBuildHooks(prepare_model=prepare, materialize=lambda _: None),
        sft_logits_mode="supervised",
    )
    batch = _batch()
    valid = torch.tensor([[False, False, True, True]])
    labels = torch.tensor([3, 4])
    full = policy.forward_policy_batch(batch).logits[:, :-1][valid]
    loss_full = torch.nn.functional.cross_entropy(full, labels)
    loss_full.backward()
    gradients = {name: p.grad.clone() for name, p in policy.net.named_parameters() if p.requires_grad}
    policy.net.zero_grad(set_to_none=True)
    selected = policy.forward_sft_batch(batch, valid).logits
    loss_selected = torch.nn.functional.cross_entropy(selected, labels)
    loss_selected.backward()
    torch.testing.assert_close(full, selected)
    torch.testing.assert_close(loss_full, loss_selected)
    for name, parameter in policy.net.named_parameters():
        if parameter.requires_grad:
            torch.testing.assert_close(parameter.grad, gradients[name], atol=1e-7, rtol=1e-5)
        else:
            assert parameter.grad is None
