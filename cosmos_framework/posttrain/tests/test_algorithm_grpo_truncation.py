# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Completion truncation: the `finish_reason` stamp, its diagnostics, and DAPO overlong filtering.

Pins, on tiny CPU tensors and a transformers-free two-layer LM:

* the stamp -- the vLLM rollout copies `CompletionOutput.finish_reason` into
  `policy_info["finish_reason"]` on training and validation completions (None when the engine
  reports none) and carries it in the TP-lane output identity; `is_length_truncated` answers only
  to the `"length"` verdict;
* the diagnostics -- the clipped objective reports the completions it was handed, the fraction the
  engine cut at its token budget, their mean generated length and the longest one, over every
  completion of the chunk whether or not the mask drops it, with zeros of zero weight on an empty
  chunk; across chunks the optimizer-window merge yields Σ/Σ ratios, a summed count and a true
  maximum, and across ranks the trainer takes the MAX rather than a mean of maxima;
* the mask -- `rl.mask_truncated_completions=false` leaves loss, gradient and every pre-existing
  metric identical to a batch without stamps; `true` removes the cut completion from the surrogate,
  the KL term and their divisors exactly as an all-zero loss mask would, down to the empty-shard
  path when every completion was cut;
* the config -- the knob defaults off, must be a bool, parses through the structured schema, joins
  the objective identity only when on, and is refused by `legacy_pg` and by OPD.
"""

from __future__ import annotations

import copy
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from cosmos_framework.posttrain.algorithm.grpo import (  # noqa: E402
    _GRPO_PUBLIC_METRICS,
    _response_metrics,
    make_clipped_objective,
)
from cosmos_framework.posttrain.config.rl import GRPOConfig, OPDConfig, RefConfig  # noqa: E402
from cosmos_framework.posttrain.data.packer.decoder_only_llm import RLPolicyInput  # noqa: E402
from cosmos_framework.posttrain.schema import (  # noqa: E402
    FINISH_REASON_KEY,
    FINISH_REASON_LENGTH,
    FINISH_REASON_STOP,
    ObsBundle,
    Status,
    Trajectory,
    Transition,
    is_length_truncated,
)
from cosmos_framework.posttrain.schema.metrics import SUMMED_TRAIN_COUNT_KEYS  # noqa: E402
from cosmos_framework.posttrain.tests.helpers.distributed import (  # noqa: E402
    init_gloo_process_group,
    run_spawned_ranks,
)
from cosmos_framework.posttrain.utils.report import merge_training_metric_window  # noqa: E402

# --------------------------------------------------------------------------- fixtures (no transformers)

_COMPLETIONS: list[list[int]] = [[5, 6, 7], [8, 9], [10, 11, 12, 13]]  # lengths 3, 2, 4
_RESPONSE_KEYS = frozenset(
    {"grpo_response_count", "grpo_truncated_frac", "grpo_response_len_mean", "grpo_response_len_max"}
)


class _TinyLM(nn.Module):
    """Embedding + linear head; `forward(ids) -> SimpleNamespace(logits=[B,L,V])`, counting calls."""

    def __init__(self, vocab: int = 16, d: int = 8, seed: int = 0) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.emb = nn.Embedding(vocab, d)
        self.head = nn.Linear(d, vocab)
        self.calls: list[tuple[int, ...]] = []

    def forward(self, ids: torch.Tensor) -> Any:  # ids [B,L]
        self.calls.append(tuple(ids.shape))
        return SimpleNamespace(logits=self.head(torch.tanh(self.emb(ids))))  # [B,L,V]


class _IdsPacker:
    """Token-id packer following `DecoderOnlyLLMDataPacker`'s policy contract, without a tokenizer."""

    def get_policy_input(
        self,
        obs: Any,
        completion: Any,
        *,
        n_ignore_prefix_tokens: int = 0,
        prompt_ids: list[int] | None = None,
    ) -> RLPolicyInput:
        assert prompt_ids, "tests store prompt_ids on every transition"
        completion_ids = [int(t) for t in completion]
        return RLPolicyInput(
            input_ids=list(prompt_ids) + completion_ids,
            logprob_masks=[0] * (len(prompt_ids) - 1 + n_ignore_prefix_tokens)
            + [1] * (len(completion_ids) - n_ignore_prefix_tokens)
            + [0],
        )

    def policy_compute_max_len(self, processed: list[RLPolicyInput]) -> int:
        return max(len(x.input_ids) for x in processed)

    def policy_collate_fn(self, processed: list[RLPolicyInput], max_len: int) -> dict[str, Any]:
        ids = torch.tensor([x.input_ids + [0] * (max_len - len(x.input_ids)) for x in processed], dtype=torch.long)
        masks = torch.tensor(
            [x.logprob_masks + [0] * (max_len - len(x.logprob_masks)) for x in processed], dtype=torch.bool
        )
        return {"input_ids": ids, "logprob_masks": masks}  # [B,L], [B,L]


def _batch(
    rewards: list[float],
    *,
    finish_reasons: list[str | None] | None = None,
    stamp_counts: list[bool] | None = None,
) -> list[Trajectory]:
    """One shared group of distinct-length completions; rewards -> group-relative advantages.

    `finish_reasons` stamps `policy_info["finish_reason"]` per completion (None is stamped as None,
    the engine's "unreported"); omitted, the key is absent as on a pre-stamp producer.
    `stamp_counts` stamps `completion_token_count` where True, so the fallback path is reachable.
    """
    from cosmos_framework.posttrain.algorithm.annotators.advantage import group_relative

    completions = _COMPLETIONS[: len(rewards)]
    trajs = []
    for k, (reward, completion) in enumerate(zip(rewards, completions, strict=True)):
        policy_info: dict[str, Any] = {"prompt_ids": torch.tensor([1, 2, 3 + k])}
        if finish_reasons is not None:
            policy_info[FINISH_REASON_KEY] = finish_reasons[k]
        if stamp_counts is not None and stamp_counts[k]:
            policy_info["completion_token_count"] = len(completion)
        tr = Transition(
            obs=ObsBundle(),
            action=torch.tensor(completion, dtype=torch.long),
            policy_info=policy_info,
            reward={"final": reward},
            done=True,
        )
        trajs.append(Trajectory(transitions=[tr], id=f"g_k{k}", group_id="g", status=Status.COMPLETED))
    group_relative()(trajs)
    return trajs


def _cfg(kl_beta: float = 0.0, *, mask: bool = False) -> GRPOConfig:
    ref = RefConfig(build_ref=kl_beta > 0, update_every=0)
    return GRPOConfig(kl_beta=kl_beta, ref=ref, group_size=2, temperature=1.0, mask_truncated_completions=mask)


def _perturb(model: nn.Module, scale: float = 0.5, seed: int = 1) -> None:
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(scale * torch.randn(p.shape, generator=g))


def _grads(policy: nn.Module, loss: torch.Tensor) -> list[torch.Tensor]:
    policy.zero_grad()
    loss.backward()
    return [torch.zeros_like(p) if p.grad is None else p.grad.detach().clone() for p in policy.parameters()]


# --------------------------------------------------------------------------- the stamp


@pytest.mark.L0
@pytest.mark.CPU
def test_is_length_truncated_answers_only_to_the_length_verdict() -> None:
    assert FINISH_REASON_KEY == "finish_reason" and FINISH_REASON_LENGTH == "length" and FINISH_REASON_STOP == "stop"
    assert is_length_truncated({FINISH_REASON_KEY: FINISH_REASON_LENGTH})
    assert not is_length_truncated({FINISH_REASON_KEY: FINISH_REASON_STOP})
    assert not is_length_truncated({FINISH_REASON_KEY: "abort"})
    # Unreported (None) and absent (a producer from before the key) both mean "not known to be cut".
    assert not is_length_truncated({FINISH_REASON_KEY: None})
    assert not is_length_truncated({})


class _StringTokenizer:
    name_or_path = "fake-tokenizer"
    model_max_length = 128
    padding_side = "right"
    truncation_side = "right"
    chat_template = "{{ messages }}"
    eos_token_id = 2
    pad_token_id = 2

    @staticmethod
    def __call__(text: str, **_kwargs: object) -> object:
        return SimpleNamespace(input_ids=[len(text)])

    @staticmethod
    def get_vocab() -> dict[str, int]:
        return {"<eos>": 2, "token": 3}


class _Packer:
    tokenizer: _StringTokenizer = _StringTokenizer()

    @staticmethod
    def adds_special_tokens() -> bool:
        return False

    @staticmethod
    def get_rollout_input(obs: ObsBundle) -> str:
        return str(obs.representations["prompt"])


def _completion_output(token_ids: list[int], finish_reason: str | None, *, stamped: bool = True) -> Any:
    """A `CompletionOutput` stand-in; `stamped=False` omits the attribute altogether."""
    fields: dict[str, Any] = {
        "token_ids": token_ids,
        "text": "ok",
        "logprobs": [{token_id: SimpleNamespace(logprob=-0.5)} for token_id in token_ids],
    }
    if stamped:
        fields["finish_reason"] = finish_reason
    return SimpleNamespace(**fields)


@pytest.mark.L0
@pytest.mark.CPU
def test_vllm_rollout_stamps_finish_reason_on_training_and_validation_completions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cosmos_framework.posttrain.roles.rollout.validation import ValidationRolloutContext
    from cosmos_framework.posttrain.roles.rollout.vllm import VLLMRollout

    class _SamplingParams:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    fake_vllm = ModuleType("vllm")
    fake_vllm.SamplingParams = _SamplingParams
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setenv("POSTTRAIN_REPLICA_INDEX", "0")
    monkeypatch.setenv("POSTTRAIN_REPLICA_COUNT", "1")

    class _Engine:
        @staticmethod
        def generate(prompts: list[Any], params: Any) -> list[Any]:
            if isinstance(params, list):  # validation: one request per generation, no logprobs
                return [
                    SimpleNamespace(
                        prompt_token_ids=[1, 2],
                        outputs=[SimpleNamespace(token_ids=[7], text="ok", logprobs=None, finish_reason="length")],
                    )
                    for _ in prompts
                ]
            return [
                SimpleNamespace(
                    prompt_token_ids=[1, 2],
                    outputs=[
                        _completion_output([5, 6, 7], "stop"),
                        _completion_output([8, 9], "length"),
                        _completion_output([10], None, stamped=False),  # an engine that reports none
                    ],
                )
                for _ in prompts
            ]

    rollout = VLLMRollout(_Engine(), packer=_Packer(), sampling={"temperature": 1.0, "max_tokens": 3})
    obs = ObsBundle(representations={"prompt": "question"}, id="sample")

    trajectories = rollout.generate([obs], group=3)
    stamps = [t.transitions[0].policy_info[FINISH_REASON_KEY] for t in trajectories]
    assert stamps == ["stop", "length", None]
    assert [is_length_truncated(t.transitions[0].policy_info) for t in trajectories] == [False, True, False]
    # The other per-completion fields are untouched beside the new one.
    assert [t.transitions[0].policy_info["completion_token_count"] for t in trajectories] == [3, 2, 1]
    # Every TP lane must agree on the verdict before publication, like the rest of the record.
    rows = VLLMRollout._training_output_identity(trajectories)["trajectories"]
    assert [row["finish_reason"] for row in rows] == ["stop", "length", None]

    held_out = rollout.generate(
        [obs],
        validation=ValidationRolloutContext(model_step=0, n_generation=1, seed=11),
    )
    assert [t.transitions[0].policy_info[FINISH_REASON_KEY] for t in held_out] == ["length"]


# --------------------------------------------------------------------------- the diagnostics


@pytest.mark.L0
@pytest.mark.CPU
def test_objective_reports_truncation_diagnostics_without_changing_the_loss() -> None:
    policy = _TinyLM()
    objective = make_clipped_objective(_IdsPacker(), _cfg())
    plain = _batch([1.0, 0.0, 1.0])
    stamped = _batch(
        [1.0, 0.0, 1.0],
        finish_reasons=["stop", "length", None],
        stamp_counts=[True, True, False],  # the third falls back to the shaped completion width
    )

    loss_plain, metrics_plain = objective(policy, plain)
    loss_stamped, metrics_stamped = objective(policy, stamped)
    # Stamps are diagnostics only at the default: bit-identical loss, identical pre-existing metrics.
    assert torch.equal(loss_plain, loss_stamped)
    assert set(metrics_plain) == set(_GRPO_PUBLIC_METRICS) == set(metrics_stamped)
    assert {k: v for k, v in metrics_plain.items() if k not in _RESPONSE_KEYS} == {
        k: v for k, v in metrics_stamped.items() if k not in _RESPONSE_KEYS
    }
    assert metrics_stamped["grpo_n"] == 3 and metrics_stamped["grpo_valid_tokens"] == 9

    assert metrics_stamped["grpo_response_count"] == 3
    assert metrics_stamped["grpo_truncated_frac"] == pytest.approx(1 / 3)
    assert metrics_stamped["grpo_response_len_mean"] == pytest.approx((3 + 2 + 4) / 3)
    assert metrics_stamped["grpo_response_len_max"] == 4 and isinstance(metrics_stamped["grpo_response_len_max"], int)
    # No stamp anywhere: nothing is known to be cut, lengths still report.
    assert metrics_plain["grpo_truncated_frac"] == 0.0 and metrics_plain["grpo_response_count"] == 3
    assert metrics_plain["grpo_response_len_mean"] == pytest.approx(3.0) and metrics_plain["grpo_response_len_max"] == 4


@pytest.mark.L0
@pytest.mark.CPU
def test_truncation_diagnostics_merge_as_ratios_and_a_maximum_across_chunks() -> None:
    policy = _TinyLM()
    objective = make_clipped_objective(_IdsPacker(), _cfg())
    weight_map = objective.metric_weight_map("train")
    assert weight_map["train/rollout_truncated_frac"] == "train/response_count"
    assert weight_map["train/response_len_mean"] == "train/response_count"
    assert "train/response_count" in SUMMED_TRAIN_COUNT_KEYS
    # The adapter's `max_reduced` is the ONE declaration point for maxima (no schema-level twin to forget).
    assert objective.metric_max_keys("train") == frozenset(
        {"train/kl_max", "train/response_len_max", "train/sc_rho_max"}
    )

    chunk_a = _batch([1.0, 0.0, 1.0], finish_reasons=["stop", "length", "stop"])  # lengths 3, 2, 4; one cut
    chunk_b = _batch([1.0, 0.0], finish_reasons=["length", "length"])  # lengths 3, 2; both cut
    reports = [objective.adapt_metrics(objective(policy, chunk)[1], "train") for chunk in (chunk_a, chunk_b)]
    # An empty chunk (every completion empty) contributes zero weight and must not move the ratios.
    reports.append(
        {
            "train/response_count": 0,
            "train/rollout_truncated_frac": 0.0,
            "train/response_len_mean": 0.0,
            "train/response_len_max": 0,
        }
    )
    merged = merge_training_metric_window(
        reports,
        sum_keys=frozenset(SUMMED_TRAIN_COUNT_KEYS),
        max_keys=objective.metric_max_keys("train"),
        weighted_mean_by=weight_map,
    )
    assert merged["train/response_count"] == 5
    assert merged["train/rollout_truncated_frac"] == pytest.approx(3 / 5)  # Σcut/Σcompletions, not mean(1/3, 1, 0)
    assert merged["train/response_len_mean"] == pytest.approx(14 / 5)
    assert merged["train/response_len_max"] == 4 and isinstance(merged["train/response_len_max"], int)

    with pytest.raises(ValueError, match="disjoint"):
        merge_training_metric_window(
            reports,
            sum_keys=frozenset({"train/response_len_max"}),
            max_keys=frozenset({"train/response_len_max"}),
        )


def _reduce_metrics_worker(rank: int, world_size: int, rendezvous_path: str, output_queue: Any) -> None:
    """Two gloo ranks reduce one chunk's metrics each: the count sums, the ratios weight, the max maxes."""
    import torch.distributed as dist

    try:
        init_gloo_process_group(rank, world_size, rendezvous_path)
        from cosmos_framework.posttrain.roles.trainer import LLMTrainer

        policy = _TinyLM()
        objective = make_clipped_objective(_IdsPacker(), _cfg())
        trainer = LLMTrainer(policy, objective, torch.optim.SGD(policy.parameters(), lr=0.1))
        lengths, truncated = ([3, 2, 4], 1) if rank == 0 else ([9, 1], 2)
        metrics: dict[str, Any] = {key: 0.0 for key in _GRPO_PUBLIC_METRICS}
        metrics.update(
            {
                "grpo_n": len(lengths),
                "grpo_valid_tokens": sum(lengths),
                "grpo_logprob_gap_tokens": 0,
                "grpo_sc_tokens": 0,  # summed count (train/sc_token_count): an int like the other counts
                **_response_metrics(truncated, lengths),
            }
        )
        out = trainer._reduce_metrics(metrics)
        output_queue.put(
            (
                rank,
                {
                    "response_count": out["train/response_count"],
                    "truncated_frac": out["train/rollout_truncated_frac"],
                    "len_mean": out["train/response_len_mean"],
                    "len_max": out["train/response_len_max"],
                    "len_max_type": type(out["train/response_len_max"]).__name__,
                },
            )
        )
    except Exception as error:  # noqa: BLE001 -- surfaced by the harness as a failed assertion
        output_queue.put((rank, {"worker_error": repr(error)}))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.L0
@pytest.mark.CPU
def test_truncation_diagnostics_reduce_across_ranks_with_a_sum_ratios_and_a_max(tmp_path: Any) -> None:
    results = run_spawned_ranks(
        _reduce_metrics_worker,
        world_size=2,
        rendezvous_path=tmp_path / "rendezvous",
        timeout_seconds=240.0,
        context="truncation metric reduction",
    )
    for result in results.values():
        assert result["response_count"] == 5
        assert result["truncated_frac"] == pytest.approx(3 / 5)  # (1 + 2) / (3 + 2), not mean(1/3, 1)
        assert result["len_mean"] == pytest.approx(19 / 5)
        assert result["len_max"] == 9 and result["len_max_type"] == "int"  # a mean of maxima would say 6.5


# --------------------------------------------------------------------------- the mask


@pytest.mark.L0
@pytest.mark.CPU
def test_mask_truncated_completions_removes_the_cut_completion_from_surrogate_kl_and_divisors() -> None:
    policy = _TinyLM()
    reference = copy.deepcopy(policy)
    _perturb(policy)  # a live KL term, so the mask is checked against it too
    kl_beta = 0.1
    masked = make_clipped_objective(_IdsPacker(), _cfg(kl_beta, mask=True))
    masked.reference_scorer.attach(reference)
    unmasked = make_clipped_objective(_IdsPacker(), _cfg(kl_beta))
    unmasked.reference_scorer.attach(copy.deepcopy(reference))

    batch = _batch([1.0, 0.0, 1.0], finish_reasons=["stop", "length", "stop"])  # the 2-token middle one is cut
    kept = [batch[0], batch[2]]  # advantages were already stamped over the whole group

    loss_masked, m_masked = masked(policy, batch)
    grads_masked = _grads(policy, loss_masked)
    loss_kept, m_kept = unmasked(policy, kept)
    grads_kept = _grads(policy, loss_kept)

    # Masking the cut completion == never having handed it in: same loss, same gradient, same KL,
    # same token and sequence divisors.
    assert torch.allclose(loss_masked, loss_kept)
    assert all(torch.allclose(a, b) for a, b in zip(grads_masked, grads_kept, strict=True))
    for key in ("grpo_loss", "grpo_n", "grpo_valid_tokens", "grpo_kl_mean"):
        assert m_masked[key] == pytest.approx(m_kept[key]), key
    assert m_masked["grpo_n"] == 2 and m_masked["grpo_valid_tokens"] == 3 + 4
    assert m_masked["grpo_kl_mean"] > 1e-3, "the KL term must be live for this check to mean anything"
    # The diagnostics still describe the chunk that was handed in.
    assert m_masked["grpo_response_count"] == 3 and m_masked["grpo_truncated_frac"] == pytest.approx(1 / 3)
    assert m_kept["grpo_response_count"] == 2 and m_kept["grpo_truncated_frac"] == 0.0

    # And without the mask the same batch trains on all nine tokens, to a different loss.
    loss_all, m_all = unmasked(policy, batch)
    assert m_all["grpo_n"] == 3 and m_all["grpo_valid_tokens"] == 9
    assert not torch.allclose(loss_all, loss_masked)


@pytest.mark.L0
@pytest.mark.CPU
def test_mask_truncated_completions_all_cut_takes_the_empty_shard_path() -> None:
    """Every completion cut: nothing to score, so the chunk behaves as an all-empty shard -- zero loss,
    the probe forward on policy AND reference for FSDP parity, the full metric key set -- while the
    diagnostics still report what was handed in."""
    policy = _TinyLM()
    reference = copy.deepcopy(policy)
    objective = make_clipped_objective(_IdsPacker(), _cfg(0.1, mask=True))
    objective.reference_scorer.attach(reference)

    loss, metrics = objective(policy, _batch([1.0, 0.0], finish_reasons=["length", "length"]))  # lengths 3, 2
    assert float(loss.detach()) == 0.0 and metrics["grpo_n"] == 0 and metrics["grpo_valid_tokens"] == 0
    assert set(metrics) == set(_GRPO_PUBLIC_METRICS)
    assert metrics["grpo_response_count"] == 2 and metrics["grpo_truncated_frac"] == 1.0
    assert metrics["grpo_response_len_mean"] == pytest.approx(2.5) and metrics["grpo_response_len_max"] == 3
    assert len(policy.calls) == 1 and reference.calls == policy.calls


# --------------------------------------------------------------------------- the config


@pytest.mark.L0
@pytest.mark.CPU
def test_mask_truncated_completions_config_defaults_off_and_is_guarded() -> None:
    from omegaconf import OmegaConf

    from cosmos_framework.posttrain.config.build import build_objective

    assert GRPOConfig().mask_truncated_completions is False
    assert GRPOConfig(mask_truncated_completions=True).mask_truncated_completions is True
    with pytest.raises(TypeError, match="mask_truncated_completions must be a bool"):
        GRPOConfig(mask_truncated_completions=1)
    with pytest.raises(ValueError, match="does not consume clipped-surrogate settings.*mask_truncated_completions"):
        GRPOConfig(grpo_surrogate="legacy_pg", mask_truncated_completions=True)
    with pytest.raises(ValueError, match="does not consume mask_truncated_completions"):
        OPDConfig(mask_truncated_completions=True)
    assert OPDConfig().mask_truncated_completions is False

    # The dotted override parses through the structured schema into a valid config.
    schema = OmegaConf.structured(GRPOConfig)
    merged = OmegaConf.merge(schema, OmegaConf.from_dotlist(["mask_truncated_completions=true"]))
    resolved = OmegaConf.to_object(merged)
    assert isinstance(resolved, GRPOConfig) and resolved.mask_truncated_completions is True

    # Off is recorded by absence, like token-level importance sampling: every earlier clipped
    # checkpoint trained without the mask and must keep its exact resume.
    off = build_objective(GRPOConfig(), "unused", packer=object())
    on = build_objective(GRPOConfig(mask_truncated_completions=True), "unused", packer=object())
    assert "mask_truncated_completions" not in off.identity
    assert on.identity["mask_truncated_completions"] is True
    assert set(on.identity) == set(off.identity) | {"mask_truncated_completions"}
