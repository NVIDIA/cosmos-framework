# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""@L0 @CPU: routing replay on the vendored Qwen3-VL-MoE block and the posttrain MoE adapter.

Real router/top-k code; the expert GEMMs use the vendored ``naive`` implementation (no CPU kernel).
"""

from __future__ import annotations

import copy

import pytest
import torch

from cosmos_framework.model.generator.reasoner.qwen3_vl_moe.configuration_qwen3_vl_moe import (
    Qwen3VLMoeTextConfig,
)
from cosmos_framework.model.generator.reasoner.qwen3_vl_moe.moe import create_text_experts
from cosmos_framework.model.generator.reasoner.qwen3_vl_moe.qwen3_vl_moe import (
    Qwen3VLMoeTextSparseMoeBlock,
)
from cosmos_framework.posttrain.model.qwen3_vl_moe import Qwen3VLMoePolicy, override_qwen3_vl_moe_for_padded_batches
from cosmos_framework.posttrain.schema.routing import FORWARD_REPLAY_BATCH_KEY

pytestmark = [pytest.mark.L0, pytest.mark.CPU]

_HIDDEN = 8
_EXPERTS = 4
_TOP_K = 2
_B, _T = 1, 3


def _tiny_block(gate_rows: torch.Tensor | None = None, top_k: int = _TOP_K) -> Qwen3VLMoeTextSparseMoeBlock:
    config = Qwen3VLMoeTextConfig(
        hidden_size=_HIDDEN,
        moe_intermediate_size=4,
        num_experts=_EXPERTS,
        num_experts_per_tok=_TOP_K,
        hidden_act="silu",
    )
    torch.manual_seed(7)
    block = Qwen3VLMoeTextSparseMoeBlock(config, top_k=top_k)
    block.experts = create_text_experts(config, implementation_type="naive")
    block.experts.init_weights(torch.device("cpu"))
    if gate_rows is not None:
        with torch.no_grad():
            block.gate.weight.copy_(gate_rows)
    return block


def _tokens(n: int) -> torch.Tensor:
    torch.manual_seed(11)
    return torch.randn(n, _HIDDEN)  # [N,H]


def _reference(block: Qwen3VLMoeTextSparseMoeBlock, tokens: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """``tokens`` dispatched to ``indices`` [N,top_k], combined by this block's softmax scores renormalized over them."""

    def expert_out(expert: int, x: torch.Tensor) -> torch.Tensor:
        gate, up = (x @ block.experts.gate_up_proj[expert]).chunk(2, dim=-1)
        return (up * torch.nn.functional.silu(gate)) @ block.experts.down_proj[expert]  # [1,H]

    with torch.no_grad():
        scores = torch.softmax(tokens @ block.gate.weight.t(), dim=-1)  # [N,E]
        combine = scores.gather(1, indices)
        combine = combine / combine.sum(dim=-1, keepdim=True)  # [N,top_k]
        return torch.stack(
            [
                sum(combine[t, k] * expert_out(int(indices[t, k]), tokens[t : t + 1])[0] for k in range(_TOP_K))
                for t in range(tokens.shape[0])
            ]
        )  # [N,H]


def test_block_runs_natural_by_default_and_dispatches_the_pinned_experts() -> None:
    """The vendored block's default selection is the gate-natural top-k and knows no pin; the posttrain
    override installed at adaptation dispatches exactly the pinned experts and keeps the gate gradient."""
    block = _tiny_block()
    tokens = _tokens(2)
    with torch.no_grad():
        natural = torch.softmax(tokens @ block.gate.weight.t(), dim=-1).topk(_TOP_K, dim=-1).indices  # [N,top_k]
        out_default, _ = block(tokens)
    torch.testing.assert_close(out_default, _reference(block, tokens, natural), rtol=1e-5, atol=1e-5)
    assert "select_experts" not in vars(block) and not hasattr(block, "_posttrain_routing_replay_indices")

    forced = torch.tensor([[2, 3], [0, 3]], dtype=torch.long)  # [N,top_k]
    assert not torch.equal(forced, natural)
    net = torch.nn.Sequential(block)
    assert override_qwen3_vl_moe_for_padded_batches(net) == 1
    Qwen3VLMoePolicy(net).prepare_policy_batch(_batch(forced.view(1, 2, 1, _TOP_K), tokens=2))
    out = net(tokens)  # [N,H]
    torch.testing.assert_close(out, _reference(block, tokens, forced), rtol=1e-5, atol=1e-5)
    out.sum().backward()
    assert block.gate.weight.grad is not None and float(block.gate.weight.grad.abs().sum()) > 0.0


def _moe_policy() -> tuple[Qwen3VLMoePolicy, torch.nn.Sequential]:
    """Two adapted blocks (gates biased toward experts {0,1} for +v tokens) and an un-adapted copy as reference."""
    v = torch.ones(_HIDDEN)
    gate_rows = torch.stack([v, 0.5 * v, -0.5 * v, -v])  # [E,H]
    net = torch.nn.Sequential(_tiny_block(gate_rows), _tiny_block(gate_rows))
    reference = copy.deepcopy(net)
    assert override_qwen3_vl_moe_for_padded_batches(net) == 2
    return Qwen3VLMoePolicy(net), reference


def test_adapter_declares_the_load_balancing_capability_over_its_adapted_blocks() -> None:
    """``Qwen3VLMoePolicy`` implements ``LoadBalancingPolicy``.
    ``begin_lbl_collection`` switches the stash on over the two adapted blocks (returns 2); a natural forward fills it
    with the vendored block's own ``LBLMetadata`` (counts over tokens x top_k); ``collect_lbl_metadata`` pops both records
    and a second collect is empty."""
    policy, _ = _moe_policy()
    for name in ("begin_lbl_collection", "collect_lbl_metadata"):
        assert callable(getattr(policy, name, None)), f"the adapter declares no {name}"
    # Imported after the behavioural assertion so that a tree without the capability fails on it, not on this ImportError
    # (review round 1 of fix round 1, finding F3).
    from cosmos_framework.posttrain.model.protocol import LoadBalancingPolicy

    assert isinstance(policy, LoadBalancingPolicy)
    assert policy.collect_lbl_metadata() == []
    assert policy.begin_lbl_collection() == 2
    tokens = _tokens(2)
    with torch.no_grad():
        policy.net(tokens)
    records = policy.collect_lbl_metadata()
    assert len(records) == 2
    assert [int(record.num_tokens_per_expert.sum()) for record in records] == [2 * _TOP_K, 2 * _TOP_K]
    assert policy.collect_lbl_metadata() == [], "collecting pops the stash"


def _reference_forward(
    reference: torch.nn.Sequential, chunk: torch.Tensor, routing: torch.Tensor | None
) -> torch.Tensor:
    """The un-adapted blocks: their own natural forward, or the hand-written dispatch to ``routing``."""
    hidden = chunk.reshape(-1, _HIDDEN)  # [N,H]
    with torch.no_grad():
        for layer, block in enumerate(reference):
            if routing is None:
                hidden, _ = block(hidden)
            else:
                hidden = _reference(block, hidden, routing[:, :, layer, :].reshape(-1, _TOP_K).long())
    return hidden.reshape(chunk.shape)  # [B,T,H]


def _batch(routing: torch.Tensor | None, tokens: int = _T) -> dict[str, torch.Tensor]:
    batch = {"input_ids": torch.zeros((_B, tokens), dtype=torch.long)}
    if routing is not None:
        batch[FORWARD_REPLAY_BATCH_KEY] = routing
    return batch


def test_adapter_pins_from_the_batch_key_and_refuses_a_foreign_record() -> None:
    """The batch key pins every block for that batch only and is consumed; a record not describing the model or batch is refused."""
    policy, reference = _moe_policy()
    chunk = torch.ones(_B, _T, _HIDDEN) + 0.01 * _tokens(_T).unsqueeze(0)  # [B,T,H], along +v
    routing = torch.tensor([[2, 3], [3, 2], [2, 3]], dtype=torch.int16).view(_B, _T, 1, _TOP_K)
    routing = torch.cat([routing, routing.flip(-1)], dim=2)  # [B,T,L=2,top_k], against the gate's {0,1}

    prepared = policy.prepare_policy_batch(_batch(routing))
    assert FORWARD_REPLAY_BATCH_KEY not in prepared and policy.replay_pinned() is True
    assert policy.replay_pinned_extent() == (_B * _T, 2)
    with torch.no_grad():
        pinned = policy.net(chunk)
    torch.testing.assert_close(pinned, _reference_forward(reference, chunk, routing), rtol=1e-5, atol=1e-5)
    assert not torch.equal(pinned, _reference_forward(reference, chunk, None))

    policy.prepare_policy_batch(_batch(None))
    assert policy.replay_pinned() is False
    with torch.no_grad():
        assert torch.equal(policy.net(chunk), _reference_forward(reference, chunk, None))

    good = torch.arange(_TOP_K, dtype=torch.int16).expand(_B, _T, 2, _TOP_K)  # distinct in-range experts per row
    negative, too_large, duplicate = good.clone(), good.clone(), good.clone()
    negative[0, 0, 0, 0], too_large[0, 1, 1, 1], duplicate[0, 2, 0, :] = -1, _EXPERTS, 3
    for bad, pattern in (
        (good[:, :, :1, :], "sparse-MoE blocks"),
        (good[:, :, :, :1], "sparse-MoE blocks"),
        (good[:, :1], "sparse-MoE blocks"),
        (good.float(), "integer tensor"),
        (negative, "valid ids"),
        (too_large, "valid ids"),
        (duplicate, "repeats an expert"),
    ):
        with pytest.raises(ValueError, match=pattern):
            policy.prepare_policy_batch(_batch(bad))
        assert policy.replay_pinned() is False  # a refused record stages nothing
    with torch.no_grad():
        assert torch.equal(policy.net(chunk), _reference_forward(reference, chunk, None))

    mixed = torch.nn.Sequential(_tiny_block(), _tiny_block(top_k=1))  # blocks that no one record can describe
    override_qwen3_vl_moe_for_padded_batches(mixed)
    with pytest.raises(ValueError, match="disagree on"):
        Qwen3VLMoePolicy(mixed).prepare_policy_batch(_batch(good))
