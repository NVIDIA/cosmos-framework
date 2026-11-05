# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Contract of the clipped objective's train-vs-rollout logprob gap diagnostics."""

import pytest

torch = pytest.importorskip("torch")

from cosmos_framework.posttrain.algorithm.grpo import make_clipped_objective  # noqa: E402
from cosmos_framework.posttrain.config.rl import GRPOConfig  # noqa: E402
from cosmos_framework.posttrain.utils.report import merge_training_metric_window  # noqa: E402


def _tiny():
    """Random 2-layer GPT2 in eval mode plus the gpt2 packer."""
    from transformers import GPT2Config, GPT2LMHeadModel

    from cosmos_framework.posttrain.data.packer import init_data_packer

    torch.manual_seed(0)
    model = GPT2LMHeadModel(GPT2Config(n_layer=2, n_head=2, n_embd=32, vocab_size=50257, n_positions=64)).eval()
    return model, init_data_packer("gpt2")


def _reference_logprobs(model, packer, transition) -> torch.Tensor:  # returns [C]
    """Temperature-1 logprobs of the completion tokens from an unbatched forward."""
    from cosmos_framework.posttrain.data.packer.decoder_only_llm import shape_policy_input

    shaped = shape_policy_input(transition, packer, 0)
    ids = torch.tensor([shaped.input_ids], dtype=torch.long)  # [1,L]
    mask = torch.tensor(shaped.logprob_masks, dtype=torch.bool)  # [L]
    with torch.no_grad():
        logp = torch.log_softmax(model(input_ids=ids).logits[0, :-1], dim=-1)  # [L-1,V]
    return logp.gather(-1, ids[0, 1:].unsqueeze(-1)).squeeze(-1)[mask[:-1]]  # [C]


def _batch(model, packer, specs, *, offsets: list[list[float]] | None):
    """One reward-annotated group; `offsets[i]` (per completion token) is added to the recorded logprobs."""
    from cosmos_framework.posttrain.algorithm.annotators.advantage import group_relative
    from cosmos_framework.posttrain.schema import ObsBundle, Status, Trajectory, Transition

    tok = packer.tokenizer
    trajs = []
    for i, (prompt, completion, reward) in enumerate(specs):
        comp = torch.tensor(tok(completion, add_special_tokens=False).input_ids, dtype=torch.long)  # [C]
        tr = Transition(
            obs=ObsBundle(representations={"prompt": prompt}, id=f"p{i}"),
            action=comp,
            policy_info={"completion_text": completion},
            done=True,
        )
        if offsets is not None:
            reference = _reference_logprobs(model, packer, tr)  # [C]
            assert reference.numel() == len(offsets[i]) == comp.numel()
            tr.policy_info["rollout_logprobs"] = (reference + torch.tensor(offsets[i])).tolist()
        tr.reward = {"final": reward}
        trajs.append(Trajectory(transitions=[tr], id=f"g0_k{i}", group_id="g0", status=Status.COMPLETED))
    group_relative()(trajs)
    return trajs


@pytest.mark.L0
@pytest.mark.CPU
def test_gap_metrics_measure_recorded_rollout_logprobs_token_weighted(monkeypatch):
    """Injected gaps come back as mean / p99 / >0.5-nat fraction at T != 1, merge token-weighted, weigh zero without a record."""
    model, packer = _tiny()
    # A policy temperature away from one must not leak into the gap: it compares temperature-1 logprobs.
    objective = make_clipped_objective(
        packer,
        GRPOConfig(epsilon_low=0.2, epsilon_high=0.2, kl_beta=0.0, group_size=2, batch_per_replica=2, temperature=0.7),
    )
    specs_a, offsets_a = (
        [("3+4=", " 7 8 9", 1.0), ("5+5=", " 10 11 12 13", 0.0)],
        [[0.2, -0.2, 0.8], [0.1, 0.1, -0.8, 0.8]],
    )
    specs_b, offsets_b = [("3+4=", " 7", 1.0), ("3+4=", " 7 8", 0.0)], [[0.6], [0.9, 0.9]]
    recorded_a = _batch(model, packer, specs_a, offsets=offsets_a)
    _, metrics_a = objective.fn(model, recorded_a)
    gap_a = torch.tensor([abs(v) for row in offsets_a for v in row])  # [7]
    assert metrics_a["grpo_logprob_gap_tokens"] == gap_a.numel()
    assert metrics_a["grpo_logprob_gap_mean"] == pytest.approx(float(gap_a.mean()), abs=1e-4)
    assert metrics_a["grpo_logprob_gap_p99"] == pytest.approx(float(torch.quantile(gap_a, 0.99)), abs=1e-4)
    assert metrics_a["grpo_logprob_gap_frac"] == pytest.approx(float((gap_a > 0.5).float().mean()), abs=1e-6)

    # Two reports of unequal token counts merge to the global per-token mean, not a mean of means.
    _, metrics_b = objective.fn(model, _batch(model, packer, specs_b, offsets=offsets_b))
    reports = [objective.adapt_metrics(m, "train") for m in (metrics_a, metrics_b)]
    gaps = gap_a.tolist() + [abs(v) for row in offsets_b for v in row]
    merged = merge_training_metric_window(reports, weighted_mean_by=objective.metric_weight_map("train"))
    assert merged["train/logprob_gap_token_count"] == len(gaps)
    assert merged["train/logprob_gap_mean"] == pytest.approx(sum(gaps) / len(gaps), abs=1e-4)
    assert merged["train/logprob_gap_gt0p5_frac"] == pytest.approx(sum(g > 0.5 for g in gaps) / len(gaps), abs=1e-6)

    # No recorded rollout logprobs: zero values with a zero token count, so the merge ignores them.
    unrecorded = _batch(model, packer, specs_a, offsets=None)
    _, metrics_none = objective.fn(model, unrecorded)
    assert metrics_none["grpo_valid_tokens"] > 0 and metrics_none["grpo_logprob_gap_tokens"] == 0
    assert (
        metrics_none["grpo_logprob_gap_mean"],
        metrics_none["grpo_logprob_gap_p99"],
        metrics_none["grpo_logprob_gap_frac"],
    ) == (0.0, 0.0, 0.0)

    # Collective parity: an empty shard, a recorded one and an unrecorded one issue the same collectives.
    import torch.distributed as dist

    def run(shard):
        calls: list[tuple[str, torch.dtype]] = []

        def fake_all_reduce(tensor, op=None):
            calls.append((str(op), tensor.dtype))
            if tensor.dtype is not torch.int32:  # int32 is the failure flag: leave it at 0 == no failure
                tensor.fill_(1.0)

        monkeypatch.setattr(dist, "is_initialized", lambda: True)
        monkeypatch.setattr(dist, "get_world_size", lambda: 2)
        monkeypatch.setattr(dist, "get_backend", lambda *a, **k: "gloo")
        monkeypatch.setattr(dist, "all_reduce", fake_all_reduce)
        try:
            _, metrics = objective.fn(model, shard)
        finally:
            monkeypatch.undo()
        return tuple(calls), frozenset(metrics)

    assert len({run(shard) for shard in ([], recorded_a[:1], unrecorded[:1])}) == 1
