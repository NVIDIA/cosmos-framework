# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Training-only choice smoothing: real loss/gradients, hard evaluation and DCP identity."""

import copy

import pytest
import torch
import torch.nn.functional as F

from cosmos_framework.posttrain.algorithm.primitives.objective import OBJECTIVE_WEIGHT_SUM
from cosmos_framework.posttrain.algorithm.sft import make_objective
from cosmos_framework.posttrain.config.model import native_adapter_config_identity, quantization_config_identity
from cosmos_framework.posttrain.config.sft import SFTConfig
from cosmos_framework.posttrain.data.packer.sft import SFTPolicyInput
from cosmos_framework.posttrain.roles.trainer.llm import LLMTrainer
from cosmos_framework.posttrain.roles.trainer.validation.sft import evaluate_sft_batch
from cosmos_framework.posttrain.roles.trainer.validation.wts import evaluate_row
from cosmos_framework.posttrain.schema import ObsBundle, SFTTarget, Trajectory, Transition
from cosmos_framework.posttrain.tests.helpers.sft_objective import MemoryPacker, Policy, sample_batch
from cosmos_framework.posttrain.tests.test_checkpoint_adapter import _save, _trainer
from cosmos_framework.posttrain.tests.test_sft_development_evaluation import VideoNet, VideoPacker
from cosmos_framework.posttrain.tests.test_sft_preprocessing_resume_identity import (
    _BeforeBackbone,
    _config,
    wiring,  # noqa: F401 - reuse the actual entrypoint/pre-backbone fixture
)
from cosmos_framework.utils.lazy_config import instantiate

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


@pytest.fixture
def real_objective_wiring(wiring, monkeypatch):
    from cosmos_framework.posttrain.algorithm import sft as algorithm
    from cosmos_framework.posttrain.tests import test_sft_preprocessing_resume_identity as fixture_module

    original = fixture_module._patch_construction

    def construct(*args):
        original(*args)
        # The older fixture deliberately stubs objective construction. Resume
        # coverage here must use the actual new objective and its identity.
        monkeypatch.setattr(algorithm, "make_objective", make_objective)

    monkeypatch.setattr(fixture_module, "_patch_construction", construct)
    return wiring


def _row(choices=4, answer="B", *, long=False):
    return {
        "id": "cpu-wts",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": ("Long question?" if long else "Question?")
                        + "\n"
                        + "\n".join(f"{letter}. choice" for letter in "ABCD"[:choices]),
                    },
                    {"type": "video", "video": "/cpu-fixture/video.mp4"},
                ],
            },
            {"role": "assistant", "content": answer},
        ],
    }


def _trajectory(row):
    return Trajectory(
        transitions=[Transition(obs=ObsBundle(), targets=SFTTarget(messages=copy.deepcopy(row["messages"])))]
    )


class VariablePacker(VideoPacker):
    def sft_process_sample(self, target):
        sample = super().sft_process_sample(target)
        if target.messages[0]["content"][0]["text"].startswith("Long"):
            return SFTPolicyInput([0, 0, *sample.input_ids], [-100, -100, *sample.labels], sample.mm_kwargs)
        return sample


def _policy():
    torch.manual_seed(81)
    policy = Policy(VideoNet())
    policy.net.dropout.p = 0.0
    return policy


@pytest.mark.parametrize("choices", [2, 3, 4])
def test_known_loss_and_gradient_full_vocabulary_denominator(choices):
    packer, policy = VariablePacker(), _policy()
    objective = make_objective(packer, wts_answer_smoothing=0.02)
    loss, metrics = objective(policy, [_trajectory(_row(choices))])
    logits = policy.last_logits.detach().clone().requires_grad_()
    logp = logits[0, 1].float().log_softmax(-1)
    answer = 0.98 * -logp[3] + 0.02 * -logp[2 : 2 + choices].mean()
    end = F.cross_entropy(logits[0, 2].float()[None], torch.tensor([6]))
    expected = (answer + end) / 2
    expected.backward()
    loss.backward()
    torch.testing.assert_close(loss, expected, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(policy.last_logits.grad, logits.grad, rtol=1e-6, atol=1e-7)
    hard = F.cross_entropy(logits[0, 1:3].float(), torch.tensor([3, 6]))
    torch.testing.assert_close(metrics["sft_loss"], hard)
    torch.testing.assert_close(metrics["sft_objective"], loss.detach())
    assert metrics[OBJECTIVE_WEIGHT_SUM] == 2
    assert torch.count_nonzero(policy.last_logits.grad[0, [0, 3]]) == 0
    # Non-choice vocabulary logits still receive the full-softmax gradient.
    assert torch.all(policy.last_logits.grad[0, 1, [0, 1, 6]] > 0)


def test_zero_epsilon_preserves_ordinary_arithmetic_identity_and_rng():
    packer = MemoryPacker()
    rows = sample_batch(packer)  # Ordinary non-WTS rows must remain supported.
    policy = Policy()
    baseline, explicit = make_objective(packer), make_objective(packer, wts_answer_smoothing=0)
    assert baseline.identity == explicit.identity and "wts_answer_smoothing" not in explicit.identity
    rng = torch.get_rng_state().clone()
    a, am = baseline(policy, rows)
    ag = torch.autograd.grad(a, tuple(policy.parameters()))
    b, bm = explicit(policy, rows)
    bg = torch.autograd.grad(b, tuple(policy.parameters()))
    assert torch.equal(a, b) and am.keys() == bm.keys()
    assert all(torch.equal(torch.as_tensor(am[k]), torch.as_tensor(bm[k])) for k in am)
    assert all(torch.equal(x, y) for x, y in zip(ag, bg, strict=True))
    assert torch.equal(rng, torch.get_rng_state())


def test_variable_length_accumulation_matches_one_batch_without_mask_or_count_changes():
    packer = VariablePacker()
    rows = [_trajectory(_row(2, "A")), _trajectory(_row(4, "D", long=True))]
    policy, expected = _policy(), _policy()
    objective = make_objective(packer, wts_answer_smoothing=0.02)
    trainer = LLMTrainer(policy, objective, torch.optim.SGD(policy.parameters(), lr=0.1), grad_accum=2)
    optimizer = torch.optim.SGD(expected.parameters(), lr=0.1)
    loss, metrics = objective(expected, rows)
    loss.backward()
    optimizer.step()
    original = [p.clone() for p in policy.parameters()]
    trainer.train(rows[:1])
    assert all(torch.equal(p, q) for p, q in zip(policy.parameters(), original, strict=True))
    trainer.train(rows[1:])
    assert trainer.iteration() == 1 and trainer._denom_at_last_step == metrics[OBJECTIVE_WEIGHT_SUM] == 4
    for p, q in zip(policy.parameters(), expected.parameters(), strict=True):
        torch.testing.assert_close(p, q, rtol=1e-6, atol=1e-7)


def test_hard_gold_evaluation_and_wts_scores_unchanged_with_candidate_training_objective():
    packer, policy = VariablePacker(), _policy()
    hard = make_objective(packer)
    candidate = make_objective(packer, wts_answer_smoothing=0.02)
    row = _row()
    rng = torch.get_rng_state().clone()
    hm = evaluate_sft_batch(policy, hard, [_trajectory(row)])
    cm = evaluate_sft_batch(policy, candidate, [_trajectory(row)])
    assert hm.keys() == cm.keys() and "sft_objective" not in cm
    assert all(torch.equal(torch.as_tensor(hm[k]), torch.as_tensor(cm[k])) for k in hm)
    assert evaluate_row(policy, packer, candidate, row) == evaluate_row(policy, packer, hard, row)
    assert policy.net.training and torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize("kind", ["one", "gap", "duplicate", "empty", "lowercase", "gold", "turns", "opener", "tools"])
def test_malformed_choice_or_answer_contract_refuses_before_forward(kind):
    row = _row()
    if kind in {"one", "gap", "duplicate", "empty", "lowercase"}:
        row["messages"][0]["content"][0]["text"] = {
            "one": "Q?\nA. one",
            "gap": "Q?\nA. one\nC. three",
            "duplicate": "Q?\nA. one\nA. two",
            "empty": "Q?\nA. one\nB.",
            "lowercase": "Q?\na. one\nb. two",
        }[kind]
    elif kind == "gold":
        row = _row(2, "C")
    elif kind == "turns":
        row["messages"] += copy.deepcopy(row["messages"])
    trajectory = _trajectory(row)
    if kind == "opener":
        trajectory.transitions[0].targets.opener = "prefilled"
    if kind == "tools":
        trajectory.transitions[0].targets.tools = [{"function": {"name": "tool"}}]
    policy = _policy()
    with pytest.raises(ValueError):
        make_objective(VariablePacker(), wts_answer_smoothing=0.02)(policy, [trajectory])
    assert policy.calls == 0


@pytest.mark.parametrize(
    "kind", ["wrong_gold_token", "missing_end", "wrong_end", "extra_label", "masked_end", "attention", "collation"]
)
def test_alignment_mask_and_truncation_refuse_before_forward(kind, monkeypatch):
    packer, policy = VariablePacker(), _policy()
    process, collate = packer.sft_process_sample, packer.sft_collate_fn

    def altered(target):
        sample = process(target)
        if kind == "wrong_gold_token":
            sample.labels[2] = 2
        elif kind == "missing_end":
            sample.input_ids.pop()
            sample.labels.pop()
        elif kind == "wrong_end":
            sample.input_ids[3] = 5
        elif kind == "extra_label":
            sample.labels[0] = 0
        elif kind == "masked_end":
            sample.labels[3] = -100
        return sample

    def collated(*args):
        batch = collate(*args)
        if kind == "attention":
            batch["attention_mask"][0, 3] = 0
        elif kind == "collation":
            batch["input_ids"][0, 2] = 2
        return batch

    monkeypatch.setattr(packer, "sft_process_sample", altered)
    monkeypatch.setattr(packer, "sft_collate_fn", collated)
    with pytest.raises(ValueError):
        make_objective(packer, wts_answer_smoothing=0.02)(policy, [_trajectory(_row())])
    assert policy.calls == 0


@pytest.mark.parametrize("epsilon", [True, -0.01, 1.0, float("nan"), float("inf"), "0.02"])
def test_typed_epsilon_rejects_invalid_values(epsilon):
    with pytest.raises(ValueError, match="wts_answer_smoothing"):
        SFTConfig(wts_answer_smoothing=epsilon)
    with pytest.raises(ValueError, match="wts_answer_smoothing"):
        make_objective(VariablePacker(), wts_answer_smoothing=epsilon)


@pytest.mark.parametrize("knob", [{"ce_exponent": 0.5}, {"moe_lbl_coeff": 0.01}])
def test_candidate_refuses_different_reduction_or_auxiliary_loss(knob):
    with pytest.raises(ValueError, match="ordinary supervised-token"):
        SFTConfig(wts_answer_smoothing=0.02, **knob)
    with pytest.raises(ValueError, match="ordinary supervised-token"):
        make_objective(VariablePacker(), wts_answer_smoothing=0.02, **knob)


@pytest.mark.parametrize("saved,new", [(0.0, 0.02), (0.02, 0.0), (0.02, 0.03)])
def test_real_dcp_entrypoint_refuses_changed_epsilon_before_backbone(tmp_path, real_objective_wiring, saved, new):
    wiring = real_objective_wiring
    config = _config()
    config.rl.wts_answer_smoothing = saved
    with pytest.raises(_BeforeBackbone):
        wiring.invoke(config)
    processor, training = copy.deepcopy(wiring.contracts[-1])
    if saved:
        assert training["objective"]["wts_answer_smoothing"]["epsilon"] == saved
    else:
        assert "wts_answer_smoothing" not in training["objective"]
    trainer = _trainer()
    model_config = instantiate(config.model.model_config)
    trainer._adapter_checkpoint_codec().identity = {
        "base": dict(model_config.source_identity),
        "adapter": native_adapter_config_identity(model_config.adapter),
        "quantization": quantization_config_identity(model_config.quantized_base),
        "processor_contract": processor,
        "training_contract": training,
    }
    from types import SimpleNamespace

    checkpoint = SimpleNamespace(manager=trainer.checkpoint_manager, resume_checkpoint=str(_save(trainer, tmp_path)))
    with pytest.raises(_BeforeBackbone):
        wiring.invoke(config, checkpoint)  # Exact same objective is admissible.
    config.rl.wts_answer_smoothing = new
    wiring.backbone.reset_mock()
    with pytest.raises(ValueError, match="training_contract.*wts_answer_smoothing"):
        wiring.invoke(config, checkpoint)
    wiring.backbone.assert_not_called()
