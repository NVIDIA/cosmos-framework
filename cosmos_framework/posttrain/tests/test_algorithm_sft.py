# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""SFT causal alignment, masking, batched media forwarding and trainer numerics."""

from __future__ import annotations

import copy
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel

from cosmos_framework.posttrain.algorithm.primitives.objective import OBJECTIVE_WEIGHT_SUM, Reduction
from cosmos_framework.posttrain.algorithm.sft import make_objective
from cosmos_framework.posttrain.data.packer.sft import SFTPolicyInput
from cosmos_framework.posttrain.roles.trainer.llm import LLMTrainer
from cosmos_framework.posttrain.tests.helpers.sft_objective import MemoryPacker, Policy, sample_batch

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


@pytest.mark.parametrize("extra_padding", [0, 4])
def test_ce_matches_hf_with_one_shift(extra_padding: int) -> None:
    torch.manual_seed(11)
    net = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=16,
            n_positions=32,
            n_layer=1,
            n_head=2,
            n_embd=8,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
        )
    ).eval()
    packer = MemoryPacker(padding=extra_padding)
    trajectories = sample_batch(packer)
    samples = [packer.sft_process_sample(trajectory.transitions[0].targets) for trajectory in trajectories]
    batch = packer.sft_collate_fn(samples, packer.sft_compute_max_len(samples))
    expected = net(**batch).loss
    policy = Policy(net)
    objective = make_objective(packer)
    loss, metrics = objective(policy, trajectories)
    torch.testing.assert_close(loss, expected)
    assert policy.calls == 1
    assert objective.reduction is Reduction.TOKEN_SUM
    assert objective.old_logp_cache is None and objective.reference_scorer is None
    count = int((batch["labels"][:, 1:] != -100).sum())
    assert metrics[OBJECTIVE_WEIGHT_SUM] == count
    public = objective.adapt_metrics(metrics, "train")
    assert public["train/valid_token_count"] == count
    assert public["train/valid_sequence_count"] == len(trajectories)
    assert objective.metric_weight_map("train") == {"train/loss_avg": "train/valid_token_count"}


def test_masked_and_padding_logits_have_zero_gradients() -> None:
    packer = MemoryPacker(padding=3)
    trajectories = sample_batch(packer)
    policy = Policy()
    loss, _ = make_objective(packer)(policy, trajectories)
    loss.backward()
    assert policy.last_logits is not None and policy.last_logits.grad is not None
    samples = list(packer.samples.values())
    batch = packer.sft_collate_fn(samples, packer.sft_compute_max_len(samples))
    active = torch.zeros_like(batch["labels"], dtype=torch.bool)  # [B,L]
    active[:, :-1] = batch["labels"][:, 1:] != -100
    grads = policy.last_logits.grad  # [B,L,V]
    assert torch.count_nonzero(grads[~active]) == 0
    assert torch.all(grads[active].abs().sum(dim=-1) > 0)


def test_attention_excludes_left_and_right_padding(monkeypatch: pytest.MonkeyPatch) -> None:
    packer = MemoryPacker()
    row = packer.add([0, 1, 2, 0], [0, 1, 2, 0])
    collate = packer.sft_collate_fn

    def padded_batch(samples: list[SFTPolicyInput], max_len: int, ignore_label_id: int) -> dict[str, Any]:
        batch = collate(samples, max_len, ignore_label_id)
        batch["attention_mask"] = torch.tensor([[0, 1, 1, 0]])
        return batch

    monkeypatch.setattr(packer, "sft_collate_fn", padded_batch)
    policy = Policy()
    loss, metrics = make_objective(packer)(policy, [row])
    assert policy.last_logits is not None
    torch.testing.assert_close(loss, F.cross_entropy(policy.last_logits[:, 1], torch.tensor([2])))
    assert metrics["sft_valid_tokens"] == 1
    loss.backward()
    assert policy.last_logits.grad is not None
    assert torch.count_nonzero(policy.last_logits.grad[:, [0, 2, 3]]) == 0


def test_media_is_preserved_and_batched() -> None:
    packer = MemoryPacker()
    media = {"features": torch.arange(8, dtype=torch.float32)}
    trajectories = [packer.add([1, 2, 3], [-100, 2, 3], mm_kwargs={"media": media}) for _ in range(2)]
    policy = Policy()
    loss, _ = make_objective(packer)(policy, trajectories)
    assert policy.calls == 1
    assert policy.last_batch is not None and "labels" not in policy.last_batch
    assert policy.last_batch["media"] == [media, media]
    expected = policy.net(**policy.last_batch).logits[:, :-1]  # [B,L-1,V]
    torch.testing.assert_close(loss, F.cross_entropy(expected.flatten(0, 1), torch.tensor([2, 3, 2, 3])))
    with pytest.raises(TypeError, match="VLMPolicy"):
        make_objective(packer)(policy.net, trajectories)


def test_variable_length_accumulation_matches_single_global_token_mean() -> None:
    torch.manual_seed(13)
    packer = MemoryPacker()
    batch = sample_batch(packer)
    policy = Policy()
    expected = copy.deepcopy(policy)
    objective = make_objective(packer)
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    trainer = LLMTrainer(policy, objective, optimizer, grad_accum=len(batch))
    baseline_optimizer = torch.optim.SGD(expected.parameters(), lr=0.1)
    loss, metrics = objective(expected, batch)
    loss.backward()
    baseline_optimizer.step()
    before = [parameter.detach().clone() for parameter in policy.parameters()]
    for index, trajectory in enumerate(batch):
        result = trainer.train([trajectory])
        assert torch.isfinite(torch.as_tensor(result["train/loss_avg"]))
        if index == 0:
            for parameter, original in zip(policy.parameters(), before):
                torch.testing.assert_close(parameter, original)
    assert trainer.iteration() == 1
    assert trainer._denom_at_last_step == metrics[OBJECTIVE_WEIGHT_SUM]
    for actual, reference in zip(policy.parameters(), expected.parameters()):
        torch.testing.assert_close(actual, reference)


def test_validation_dummy_rows_do_not_contribute_loss_or_counts() -> None:
    packer = MemoryPacker()
    real = packer.add([1, 2, 3], [-100, 2, 3], weight=1.0)
    dummy = packer.add([4, 5, 6, 7], [-100, 5, 6, 7], weight=0.0)
    policy = Policy()
    objective = make_objective(packer)
    expected, _ = objective(policy, [real])
    loss, metrics = objective(policy, [real, dummy])
    torch.testing.assert_close(loss, expected)
    assert metrics["sft_valid_tokens"] == 2
    assert metrics["sft_valid_sequences"] == 1
    loss.backward()
    assert policy.last_logits is not None and policy.last_logits.grad is not None
    assert torch.count_nonzero(policy.last_logits.grad[1]) == 0


@pytest.mark.parametrize("kind", ["empty", "unsupervised", "dummy", "wrong_target", "multi_transition", "weight"])
def test_invalid_batches_fail_before_forward(kind: str) -> None:
    packer = MemoryPacker()
    row = packer.add([1, 2], [-100, 2])
    batch = [row]
    if kind == "empty":
        batch = []
    elif kind == "unsupervised":
        packer.samples[row.id].labels = [-100, -100]
    elif kind == "dummy":
        row.transitions[0].algo_extra["validation_weight"] = 0.0
    elif kind == "wrong_target":
        row.transitions[0].targets = {"messages": []}
    elif kind == "multi_transition":
        row.transitions.append(copy.deepcopy(row.transitions[0]))
    elif kind == "weight":
        row.transitions[0].algo_extra["validation_weight"] = 0.5
    policy = Policy()
    with pytest.raises((ValueError, TypeError)):
        make_objective(packer)(policy, batch)
    assert policy.calls == 0


def test_identity_binds_masking_and_snapshots_packer_settings() -> None:
    packer = MemoryPacker()
    objective = make_objective(packer)
    assert objective.identity == make_objective(MemoryPacker()).identity
    packer.masking = "all"
    assert objective.identity != make_objective(packer).identity


def _distributed_worker(rank: int, rendezvous: str) -> None:
    """Two CPU ranks exercise uneven tokens, dummy tails and converged preflight."""
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        torch.set_num_threads(1)
        torch.manual_seed(29)
        policy = Policy()
        baseline = copy.deepcopy(policy)
        policy.net = DistributedDataParallel(policy.net)
        packer = MemoryPacker()
        rows = sample_batch(packer)
        objective = make_objective(packer)
        trainer = LLMTrainer(policy, objective, torch.optim.SGD(policy.parameters(), lr=0.1), grad_accum=2)
        # Equal physical sizes, but different supervised lengths across ranks AND chunks.
        for micro in range(2):
            metrics = trainer.train([rows[(rank + micro) % 2]])
            assert metrics["train/valid_token_count"] == 5
            assert metrics["train/valid_sequence_count"] == 2
        samples = list(packer.samples.values())
        tensors = packer.sft_collate_fn(samples, packer.sft_compute_max_len(samples))
        logits = baseline.net(input_ids=tensors["input_ids"], attention_mask=tensors["attention_mask"]).logits
        loss = F.cross_entropy(logits[:, :-1].flatten(0, 1), tensors["labels"][:, 1:].flatten(), ignore_index=-100)
        optimizer = torch.optim.SGD(baseline.parameters(), lr=0.1)
        loss.backward()
        optimizer.step()
        for actual, expected in zip(policy.parameters(), baseline.parameters()):
            torch.testing.assert_close(actual, expected)
        tail = packer.add([1, 2, 3], [-100, 2, 3], weight=float(rank == 0))
        loss, metrics = objective(policy, [tail])
        assert torch.isfinite(loss)
        assert metrics["sft_valid_tokens"] == (2 if rank == 0 else 0)
        assert metrics[OBJECTIVE_WEIGHT_SUM] == 1.0
        loss.backward()  # A zero-weight local loss retains the complete DDP graph.
        if rank == 1:
            tail.transitions[0].targets = None
        calls = policy.calls
        with pytest.raises(RuntimeError, match="SFT objective preparation"):
            objective(policy, [tail])
        assert policy.calls == calls
        with pytest.raises(RuntimeError, match="nonempty physical batch"):
            objective(policy, [] if rank == 0 else [rows[1]])
        assert policy.calls == calls
        tail.transitions[0].targets = rows[0].transitions[0].targets
        tail.transitions[0].algo_extra["validation_weight"] = 0.0
        with pytest.raises(ValueError, match="globally supervised"):
            objective(policy, [tail])
        assert policy.calls == calls
    finally:
        dist.destroy_process_group()


def test_two_cpu_ranks_global_token_mean_and_preflight(tmp_path: Any) -> None:
    torch.multiprocessing.spawn(_distributed_worker, args=(f"file://{tmp_path / 'rendezvous'}",), nprocs=2, join=True)


def test_ce_exponent_weights_rows_like_the_backbone() -> None:
    """Knob 6: with ``ce_exponent=e`` every row contributes ``sum(CE)/n^e`` over ``sum_rows n^(1-e)``
    (the backbone's ``weighted_cross_entropy_loss``); ``e=0`` is the token mean the parent computed (unchanged bytes),
    ``e=0.5`` the square-root weighting, ``e=1`` the per-row mean. The parent accepted no exponent."""
    packer = MemoryPacker()
    short = packer.add([0, 1, 2, 3], [-100, 1, 2, 3])  # 3 supervised predictions
    long = packer.add([0, 4, 5, 6, 7, 8, 9, 10], [-100, 4, 5, 6, 7, 8, 9, 10])  # 7 supervised predictions
    trajectories = [short, long]
    torch.manual_seed(3)
    policy = Policy()
    with torch.no_grad():
        token_mean, metrics0 = make_objective(packer)(policy, trajectories)
        logits = policy.last_logits  # [B,L,V]
    samples = [packer.sft_process_sample(t.transitions[0].targets) for t in trajectories]
    batch = packer.sft_collate_fn(samples, packer.sft_compute_max_len(samples))
    labels = batch["labels"][:, 1:]  # [B,L-1]
    valid = labels != -100
    per_token = F.cross_entropy(
        logits[:, :-1].float().reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="none"
    ).reshape(labels.shape)
    row_sums = (per_token * valid).sum(dim=1)  # [B]
    counts = valid.sum(dim=1).float()  # [B]
    assert counts.tolist() == [3.0, 7.0]
    torch.testing.assert_close(token_mean, row_sums.sum() / counts.sum())
    for exponent in (0.5, 1.0):
        with torch.no_grad():
            loss, metrics = make_objective(packer, ce_exponent=exponent)(policy, trajectories)
        expected = (row_sums / counts.pow(exponent)).sum() / counts.pow(1 - exponent).sum()
        torch.testing.assert_close(loss, expected)
        assert metrics[OBJECTIVE_WEIGHT_SUM] == pytest.approx(float(counts.pow(1 - exponent).sum()))
        torch.testing.assert_close(metrics["sft_loss"], row_sums.sum() / counts.sum())  # loss_avg stays the token mean
        torch.testing.assert_close(metrics["sft_objective"], loss)
        public = make_objective(packer, ce_exponent=exponent).adapt_metrics(metrics, "train")
        assert "train/objective" in public
    assert "sft_objective" not in metrics0
    identity = make_objective(packer, ce_exponent=0.5).identity
    assert identity["reduction"] == "global_row_weighted_token_sum_exponent_0.5" and identity["ce_exponent"] == 0.5
    assert make_objective(packer).identity["reduction"] == "global_supervised_token_mean"
    assert "ce_exponent" not in make_objective(packer).identity
    with pytest.raises(ValueError, match="ce_exponent"):
        make_objective(packer, ce_exponent=2.0)


def test_identity_carries_the_serve_contract_only_when_it_is_not_the_default() -> None:
    """``rl.serve_contract`` enters the objective identity when it is not the default, so the checkpoint of a run that
    unsupervised the opener for a prefilling serve says so, while every existing run's identity (default contract) is
    unchanged; an unknown contract is refused."""
    packer = MemoryPacker()
    assert "serve_contract" not in make_objective(packer).identity
    assert make_objective(packer, serve_contract="tools_no_prefill").identity == make_objective(packer).identity
    assert make_objective(packer, serve_contract="prefills_opener").identity["serve_contract"] == "prefills_opener"
    with pytest.raises(ValueError, match="serve_contract"):
        make_objective(packer, serve_contract="bogus")


class _AdaptedBlock(torch.nn.Module):
    """A stand-in for a padded-batch-adapted sparse block: the flags the adapter sets and a stash the forward fills."""

    def __init__(self, metadata: Any) -> None:
        super().__init__()
        self._posttrain_padded_forward = True
        self._posttrain_collect_lbl = False
        self._posttrain_lbl_metadata = None
        self.metadata = metadata

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._posttrain_collect_lbl:
            self._posttrain_lbl_metadata = self.metadata
        return x


def _metadata(counts: list[int], probs: list[float], top_k: int = 2) -> Any:
    from types import SimpleNamespace

    return SimpleNamespace(
        num_tokens_per_expert=torch.tensor(counts, dtype=torch.int64),
        num_tokens=torch.tensor([sum(counts) // top_k], dtype=torch.int64),
        mean_router_prob_per_expert=torch.tensor(probs, dtype=torch.float32, requires_grad=True),
        top_k=torch.tensor([top_k], dtype=torch.int64),
    )


def test_sft_objective_reads_load_balancing_through_the_model_protocol() -> None:
    """FIXTURE: on 02fed8cbea ``algorithm/sft.py`` imported ``model.qwen3_vl_moe`` inside the objective and called its
    module functions over ``net``. The objective asks the policy for the ``LoadBalancingPolicy`` capability
    (``model/protocol.py``): a toy policy that implements it yields ``sft_lbl_loss`` equal to
    :func:`load_balancing_loss` over the metadata it hands back; a policy without it is refused with ``TypeError``
    naming the capability; the module imports nothing from ``cosmos_framework.posttrain.model`` but ``model.protocol``;
    ``Qwen3VLMoePolicy`` declares both methods."""
    import ast
    from pathlib import Path

    from cosmos_framework.posttrain.algorithm import sft as sft_module
    from cosmos_framework.posttrain.algorithm.sft import load_balancing_loss
    from cosmos_framework.posttrain.model.qwen3_vl_moe import Qwen3VLMoePolicy

    tree = ast.parse(Path(sft_module.__file__).read_text())
    imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    imported |= {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    offenders = {
        module
        for module in imported
        if module.startswith("cosmos_framework.posttrain.model")
        and module != "cosmos_framework.posttrain.model.protocol"
    }
    assert not offenders, offenders
    # Imported after the behavioural assertion so that a tree whose objective imports the MoE module fails on it, not on
    # this ImportError.
    from cosmos_framework.posttrain.model.protocol import LoadBalancingPolicy

    uniform = _metadata([4, 4, 4, 4], [0.25, 0.25, 0.25, 0.25])
    skewed = _metadata([16, 0, 0, 0], [1.0, 0.0, 0.0, 0.0])

    class _Balancing(Policy):
        def begin_lbl_collection(self) -> int:
            return 2

        def collect_lbl_metadata(self) -> list[Any]:
            return [uniform, skewed]

    packer = MemoryPacker()
    batch = sample_batch(packer)
    balancing = _Balancing()
    assert isinstance(balancing, LoadBalancingPolicy) and not isinstance(Policy(), LoadBalancingPolicy)
    _, metrics = make_objective(packer, moe_lbl_coeff=8e-3, moe_lbl_method="local")(balancing, batch)
    expected, load = load_balancing_loss([uniform, skewed], coeff=8e-3, method="local")
    torch.testing.assert_close(metrics["sft_lbl_loss"], expected.detach())
    torch.testing.assert_close(metrics["sft_moe_load_max_rel"], load)
    with pytest.raises(TypeError, match="LoadBalancingPolicy"):
        make_objective(packer, moe_lbl_coeff=8e-3)(Policy(), batch)
    for name in ("begin_lbl_collection", "collect_lbl_metadata"):
        assert callable(getattr(Qwen3VLMoePolicy, name)), name


def test_load_balancing_loss_matches_the_backbone_formula() -> None:
    """Knob 1: ``coeff * E * mean_layers(sum_e f_e P_e)`` -- 1.0 * coeff at uniform routing, ``E`` *
    coeff when one expert takes everything; the gradient reaches the router probabilities, not the counts; the load
    readout is ``E * max f_e`` (1.0 = balanced). Stashing only happens when switched on, and collecting pops."""
    from cosmos_framework.posttrain.algorithm.sft import load_balancing_loss
    from cosmos_framework.posttrain.model.qwen3_vl_moe import collect_lbl_metadata, set_lbl_collection

    uniform = _metadata([4, 4, 4, 4], [0.25, 0.25, 0.25, 0.25])
    loss, load = load_balancing_loss([uniform, uniform], coeff=8e-3, method="local")
    assert loss.item() == pytest.approx(8e-3) and load.item() == pytest.approx(1.0)
    skewed = _metadata([16, 0, 0, 0], [1.0, 0.0, 0.0, 0.0])
    loss, load = load_balancing_loss([skewed], coeff=8e-3, method="local")
    assert loss.item() == pytest.approx(8e-3 * 4) and load.item() == pytest.approx(4.0)
    loss.backward()
    assert skewed.mean_router_prob_per_expert.grad is not None and skewed.mean_router_prob_per_expert.grad[0] > 0
    net = torch.nn.Sequential(_AdaptedBlock(uniform), torch.nn.Identity(), _AdaptedBlock(skewed))
    assert collect_lbl_metadata(net) == []
    net(torch.zeros(1))
    assert collect_lbl_metadata(net) == [], "nothing is stashed until collection is switched on"
    assert set_lbl_collection(net, True) == 2
    net(torch.zeros(1))
    assert [m is uniform or m is skewed for m in collect_lbl_metadata(net)] == [True, True]
    assert collect_lbl_metadata(net) == [], "collecting pops the stash"
    assert set_lbl_collection(torch.nn.Linear(2, 2), True) == 0
    dense_packer = MemoryPacker()
    dense_batch = sample_batch(dense_packer)
    with pytest.raises(TypeError, match="LoadBalancingPolicy"):
        make_objective(dense_packer, moe_lbl_coeff=8e-3)(Policy(), dense_batch)  # a dense policy declares no capability
