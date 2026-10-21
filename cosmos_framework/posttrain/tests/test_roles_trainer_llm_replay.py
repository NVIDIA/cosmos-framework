# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""@L0 @CPU: `LLMTrainer` holds a run to `rl.routing_replay` and gates the probe with the model-declared tolerance.

Drives the REAL trainer over a real padded-batch-adapted sparse block whose router flips ONE token
between two tied experts at unchanged weights (wrapped in the real `Qwen3VLMoePolicy`) and over a
dense stub with one stochastic token, alone and over a 2-rank gloo DP mesh.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from cosmos_framework.model.generator.reasoner.qwen3_vl_moe.configuration_qwen3_vl_moe import (
    Qwen3VLMoeTextConfig,
)
from cosmos_framework.model.generator.reasoner.qwen3_vl_moe.moe import create_text_experts
from cosmos_framework.model.generator.reasoner.qwen3_vl_moe.qwen3_vl_moe import (
    Qwen3VLMoeTextSparseMoeBlock,
)
from cosmos_framework.posttrain.algorithm.grpo import OldLogprobCache
from cosmos_framework.posttrain.algorithm.primitives.objective import (
    OBJECTIVE_WEIGHT_SUM,
    Objective,
    Reduction,
)
from cosmos_framework.posttrain.algorithm.primitives.policy_forward import prepare_policy_capability
from cosmos_framework.posttrain.model.protocol import STRICT_FORWARD_NOISE_TOLERANCE, ForwardNoiseTolerance
from cosmos_framework.posttrain.model.qwen3_vl_moe import (
    PINNED_MOE_FORWARD_NOISE_TOLERANCE,
    Qwen3VLMoePolicy,
    override_qwen3_vl_moe_for_padded_batches,
)
from cosmos_framework.posttrain.roles.trainer.llm import LLMTrainer, probe_drift_counts
from cosmos_framework.posttrain.schema.routing import FORWARD_REPLAY_BATCH_KEY
from cosmos_framework.posttrain.tests.helpers.distributed import init_gloo_process_group

pytestmark = [pytest.mark.L0, pytest.mark.CPU]

_HIDDEN = 8
_EXPERTS = 4
_TOP_K = 2
_VOCAB = 16
_T = 17  # 16 next-token positions per row
_N_SCORED = 2 * (_T - 1)


class _FlakyGate(torch.nn.Linear):
    """Router that, on ONE row, boosts expert 0 on even calls and expert 2 on odd calls at UNCHANGED
    weights: over tied gate rows that flips the natural top-k of exactly that token between forwards."""

    def __init__(self, gate: torch.nn.Linear, row: int | None, boost: float) -> None:
        super().__init__(gate.in_features, gate.out_features, bias=False)
        with torch.no_grad():
            self.weight.copy_(gate.weight)
        self.row = row
        self.boost = boost
        self.calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [N,H] -> [N,E]
        logits = super().forward(x)
        if self.row is not None:
            logits = logits.clone()
            logits[self.row, 2 * (self.calls % 2)] += self.boost
            self.calls += 1
        return logits


class _FlakyMoE(torch.nn.Module):
    """Embedding -> ONE real padded-batch-adapted sparse block (experts {0,1,2} tied, 3 rejected) -> LM head."""

    def __init__(self, flaky_row: int | None = 0) -> None:
        super().__init__()
        config = Qwen3VLMoeTextConfig(
            hidden_size=_HIDDEN, moe_intermediate_size=4, num_experts=_EXPERTS, num_experts_per_tok=_TOP_K
        )
        torch.manual_seed(7)
        block = Qwen3VLMoeTextSparseMoeBlock(config)
        block.experts = create_text_experts(config, implementation_type="naive")
        v = torch.ones(_HIDDEN)
        with torch.no_grad():
            block.gate.weight.copy_(torch.stack([v, v, v, -v]))
            weights = torch.Generator().manual_seed(9)
            block.experts.gate_up_proj.normal_(0.0, 1.0, generator=weights)
            block.experts.down_proj.normal_(0.0, 1.0, generator=weights)
        block.gate = _FlakyGate(block.gate, row=flaky_row, boost=1.0)
        assert override_qwen3_vl_moe_for_padded_batches(block) == 1
        self.block = block
        self.embed = torch.nn.Embedding(_VOCAB, _HIDDEN)
        self.head = torch.nn.Linear(_HIDDEN, _VOCAB)

    def hidden(self, input_ids: torch.Tensor) -> torch.Tensor:  # [B,T] -> [B,T,H]
        return 1.0 + 0.1 * self.embed(input_ids)  # along +v: expert 3 always loses

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:  # [B,T] -> [B,T,V]
        return self.head(self.block(self.hidden(input_ids)))


class _OneNoisyTokenLM(torch.nn.Module):
    """Dense stub: in train() mode exactly ONE scored token's logits move by O(1) per call."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(3)
        self.embed = torch.nn.Embedding(_VOCAB, _HIDDEN)
        self.head = torch.nn.Linear(_HIDDEN, _VOCAB)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:  # [B,T] -> [B,T,V]
        logits = self.head(self.embed(input_ids))  # [B,T,V]
        if self.training:
            noise = torch.zeros_like(logits)
            noise[0, 0] = torch.randn(_VOCAB)
            logits = logits + noise
        return logits


def _ratio_objective(cache: OldLogprobCache) -> Objective:
    """A ratio objective that prepares the batch through the model capability and consults `old_for`.

    Like the real clipped objective, an EMPTY shard still runs one placeholder forward (a batch that
    carries no routing) for collective parity and contributes a zero-weight loss."""

    def fn(model: Any, batch: list[dict[str, torch.Tensor]]) -> tuple[torch.Tensor, dict[str, float]]:
        prepared = prepare_policy_capability(model, batch[0] if batch else {"input_ids": _input_ids(slice(0, 1))})
        net = getattr(model, "net", model)
        input_ids = prepared["input_ids"]  # [B,T]
        logp = torch.log_softmax(net(input_ids)[:, :-1].float(), dim=-1)  # [B,T-1,V]
        cur = logp.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1).reshape(-1)  # [N]
        old = cache.old_for(cur)  # records / probes / replays
        loss = -torch.exp(cur - old).sum()
        return (
            (loss, {OBJECTIVE_WEIGHT_SUM: float(cur.numel())}) if batch else (loss * 0.0, {OBJECTIVE_WEIGHT_SUM: 0.0})
        )

    return Objective(fn, reduction=Reduction.TOKEN_SUM, old_logp_cache=cache)


def _input_ids(rows: slice = slice(None)) -> torch.Tensor:
    return torch.randint(0, _VOCAB, (2, _T), generator=torch.Generator().manual_seed(23))[rows]  # [B,T]


def _chunk(model: _FlakyMoE | None = None, rows: slice = slice(None)) -> list[dict[str, torch.Tensor]]:
    """One trajectory batch; with a model, it carries that model's flip-free top-k as the recorded routing."""
    batch = {"input_ids": _input_ids(rows)}
    if model is not None:
        with torch.no_grad():
            logits = torch.nn.functional.linear(model.hidden(batch["input_ids"]), model.block.gate.weight)  # [B,T,E]
        batch[FORWARD_REPLAY_BATCH_KEY] = logits.topk(_TOP_K, dim=-1).indices.unsqueeze(2).to(torch.int16)  # [B,T,1,K]
    return [batch]


def _trainer(model: Any, *, forward_replay: str | None) -> tuple[LLMTrainer, OldLogprobCache]:
    cache = OldLogprobCache()
    params = getattr(model, "net", model).parameters()
    trainer = LLMTrainer(
        model, _ratio_objective(cache), torch.optim.SGD(params, lr=0.0), n_mini=2, forward_replay=forward_replay
    )
    return trainer, cache


def _mesh_worker(rank: int, world_size: int, rendezvous_path: str) -> None:
    """Rank 0 holds the flaky token (1 of 16 moved locally, over the 5% gate); rank 1 holds a clean shard."""
    import torch.distributed as dist

    init_gloo_process_group(rank, world_size, rendezvous_path)
    try:
        rows = slice(rank, rank + 1)
        model = _FlakyMoE(flaky_row=0 if rank == 0 else None)
        pinned, _ = _trainer(Qwen3VLMoePolicy(model), forward_replay="rollout")
        assert pinned.compute_old_logprobs(_chunk(model, rows)) == {"old_logp/recorded": 1}  # 1/32 mesh-wide passes
        natural, _ = _trainer(Qwen3VLMoePolicy(_FlakyMoE(flaky_row=0 if rank == 0 else None)), forward_replay=None)
        with pytest.raises(RuntimeError, match="determinism probe"):  # rank 1 moved 0/16 yet raises with the mesh
            natural.compute_old_logprobs(_chunk(rows=rows))

        # Three-state replay verdict: rank 0 pins; rank 1 ACTIVE without routing fails every rank before
        # backward (a mesh-wide OR would have passed on rank 0's pin); rank 1 EMPTY is exempt.
        model = _FlakyMoE(flaky_row=None)
        replay, _ = _trainer(Qwen3VLMoePolicy(model), forward_replay="rollout")
        with pytest.raises(RuntimeError, match="NATURAL dispatch"):
            replay.train(_chunk(model, rows) if rank == 0 else _chunk(rows=rows))
        assert all(parameter.grad is None for parameter in model.parameters())
        assert replay.train(_chunk(model, rows) if rank == 0 else [])["optim/update_outcome"] == "applied"
    finally:
        dist.destroy_process_group()


def test_pinned_routing_brackets_every_forward_and_relaxes_only_the_moe_probe(tmp_path: Path) -> None:
    """Dense and natural-MoE probes are strict; recorded routing pins ahead, probe and train forwards under the relaxed MoE gate, mesh-wide."""
    import torch.multiprocessing as mp

    torch.manual_seed(21)
    # Strict rows: each moves ONE of 32 tokens, inside the relaxed 5% fraction, and is still refused.
    for model, is_moe in ((_OneNoisyTokenLM(), False), (Qwen3VLMoePolicy(_FlakyMoE()), True)):
        trainer, cache = _trainer(model, forward_replay=None)
        with pytest.raises(RuntimeError, match="determinism probe") as excinfo:
            trainer.compute_old_logprobs(_chunk())
        assert ("rl.routing_replay" in str(excinfo.value)) is is_moe
        moved, total, _ = probe_drift_counts(
            cache.last_recorded, cache.probe, STRICT_FORWARD_NOISE_TOLERANCE.per_token_nats
        )
        assert (moved, total) == (1, _N_SCORED)
        assert moved / total <= PINNED_MOE_FORWARD_NOISE_TOLERANCE.moved_fraction

    # The trainer trusts the model-declared tolerance, so one that cannot gate is refused at construction.
    nan = float("nan")
    for nats, fraction in ((nan, 0.0), (0.05, nan), (float("inf"), 0.0), (-0.01, 0.0), (0.05, -0.1), (0.05, 1.5)):
        with pytest.raises(ValueError, match="ForwardNoiseTolerance"):
            ForwardNoiseTolerance(per_token_nats=nats, moved_fraction=fraction)

    # A replay run refuses a batch that carries no routing rather than training through natural dispatch.
    with pytest.raises(RuntimeError, match="carried no recorded routing"):
        _trainer(Qwen3VLMoePolicy(_FlakyMoE()), forward_replay="rollout")[0].compute_old_logprobs(_chunk())

    # Pinned: the flaky token still moves past the per-token bound (the combine weights come from this
    # forward), so the strict gate would refuse it; the relaxed MoE gate admits 1 of 32.
    model = _FlakyMoE()
    policy = Qwen3VLMoePolicy(model)
    trainer, cache = _trainer(policy, forward_replay="rollout")
    assert trainer.objective_identity["forward_replay"] == {"source": "rollout", **policy.replay_identity()}
    chunk = _chunk(model)
    assert trainer.compute_old_logprobs(chunk) == {"old_logp/recorded": 1}
    moved, total, _ = probe_drift_counts(
        cache.last_recorded, cache.probe, PINNED_MOE_FORWARD_NOISE_TOLERANCE.per_token_nats
    )
    assert (moved, total) == (1, _N_SCORED)
    assert 0 < moved / total <= PINNED_MOE_FORWARD_NOISE_TOLERANCE.moved_fraction

    trainer.begin_replay_window()
    metrics = trainer.train(chunk)
    assert metrics["optim/update_outcome"] == "applied"
    assert metrics["train/routing_replay_pinned_tokens"] == _input_ids().numel()
    assert metrics["train/routing_replay_pinned_blocks"] == 1
    assert policy.replay_pinned() is True

    # The gate sums moved/compared over the DP mesh and every rank takes the one verdict.
    world_size = 2
    mp.spawn(_mesh_worker, args=(world_size, str(tmp_path / "mesh-gate-rendezvous")), nprocs=world_size, join=True)
