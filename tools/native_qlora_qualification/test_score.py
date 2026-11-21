# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""CPU arithmetic/lifecycle checks; no quantized-kernel qualification."""

from __future__ import annotations

import importlib.util
import json
import math
import os
from pathlib import Path
from typing import ClassVar

import pytest
import torch

SPEC = importlib.util.spec_from_file_location("native_qlora_score", Path(__file__).with_name("score.py"))
assert SPEC is not None and SPEC.loader is not None
heldout = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(heldout)


def test_untrained_residual_state_is_not_reported_as_zero_b_frozen_base():
    zero = {"adapter": {"initialization": "kaiming_A_zero_B"}}
    residual = {"format": "cosmos.adapter-model.v3", "adapter": {"initialization": "nf4_residual_svd"}}
    assert heldout.adapter_state_label(zero, reloaded=False) == "initial_zero_B_frozen_base"
    assert heldout.adapter_state_label(residual, reloaded=False) == "residual_initialized_untrained_adapter"
    assert heldout.adapter_state_label(residual, reloaded=True) == "reloaded_checkpoint"
    with pytest.raises(ValueError, match="explicit supported"):
        heldout.adapter_state_label({}, reloaded=False)
    with pytest.raises(ValueError, match="explicit supported"):
        heldout.adapter_state_label({"adapter": {"initialization": "nf4_residual_svd"}}, reloaded=False)


def row(identifier="item0", answer="C"):
    return {
        "id": identifier,
        "messages": [
            {
                "role": "system",
                "content": "Answer the multiple-choice question with its letter.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": "clip.mp4"},
                    {
                        "type": "text",
                        "text": " \n What is the age group? \n A: 0-9 \n B: 10s \n C: 20s \n D: 50s \n\nAnswer directly.",
                    },
                ],
            },
            {"role": "assistant", "content": answer},
        ],
    }


@pytest.mark.parametrize("marker", [":", "."])
@pytest.mark.parametrize("choices", ["AB", "ABC", "ABCD"])
def test_exact_ordered_option_lines(marker, choices):
    question = "Question?\n" + "\n".join(f" {letter}{marker} option {letter} " for letter in choices)
    assert heldout.parse_options(question) == tuple(choices)


@pytest.mark.parametrize(
    "question",
    [
        "Question?\nA: x",
        "Question?\nA: x\nC: x",
        "Question?\nA: x\nB: x\nC: x\nD: x\nE: x",
        "Question?\nA: x\nB: x\nB: x\nD: x",
        "Question?\nB: x\nA: x\nC: x\nD: x",
        "Question?\nA: x\nB. x\nC: x\nD: x",
        "Question? A: x B: x C: x D: x",
        "A: x\nB: x\nC: x\nD: x",
        "Question?\nA: \nB: x\nC: x\nD: x",
        "Question?\nA: x\nB: x\nC: x\nD: ",
        "Question?\nA: x\nB: x\nC:\n",
        "Question?\nA: x\nB: x\nC:x",
        "Question?\nA: x\nB: x\nc: x",
    ],
)
def test_rejects_ambiguous_or_incomplete_options(question):
    with pytest.raises(ValueError):
        heldout.parse_options(question)


@pytest.mark.parametrize("answer", [" C", "C\n", "The answer is C", "c", "E", "", None])
def test_requires_exact_single_letter_gold(answer):
    with pytest.raises(ValueError, match="exactly one uppercase"):
        heldout.validate_wts(row(answer=answer))


def choice_row(choices, *, answer=None, identifier="item0"):
    result = row(identifier, choices[-1] if answer is None else answer)
    result["messages"][-2]["content"][-1]["text"] = "Question?\n" + "\n".join(
        f"{letter}: option {letter}" for letter in choices
    )
    return result


@pytest.mark.parametrize("choices", ["AB", "ABC", "ABCD"])
def test_validation_returns_actual_choices_and_gold(choices):
    _, answer, actual, _ = heldout.validate_wts(choice_row(choices))
    assert actual == tuple(choices)
    assert answer == choices[-1]


@pytest.mark.parametrize("choices,answer", [("AB", "C"), ("AB", "D"), ("ABC", "D")])
def test_missing_actual_gold_is_rejected(choices, answer):
    with pytest.raises(ValueError, match="actual question choices"):
        heldout.validate_wts(choice_row(choices, answer=answer))


def test_selection_preserves_mixed_choice_rows_and_original_priority_rule(tmp_path):
    raw = [choice_row("ABCD"[: 2 + index % 3], identifier=f"row{index}") for index in range(30)]
    source = tmp_path / "mixed.jsonl"
    source.write_text("".join(json.dumps(item) + "\n" for item in raw))
    selected, metadata = heldout.select_rows(source, 12, 2026)
    ranked = sorted(
        (
            heldout.json_digest({"seed": 2026, "id": str(item["id"]), "row": item}),
            line,
            item,
        )
        for line, item in enumerate(raw, 1)
    )[:12]
    assert selected == sorted((line, item) for _, line, item in ranked)
    assert metadata["source_rows"] == 30
    assert metadata["selected_rows"] == 12
    assert {len(heldout.validate_wts(item)[2]) for _, item in selected} == {2, 3, 4}


def test_source_selection_is_deterministic_and_bounded(tmp_path):
    data = tmp_path / "validation.jsonl"
    data.write_text("".join(json.dumps(row(f"item{index}")) + "\n" for index in range(20)))
    first, metadata = heldout.select_rows(data, 4, 2026)
    assert first == heldout.select_rows(data, 4, 2026)[0]
    assert first != heldout.select_rows(data, 4, 2027)[0]
    assert len(first) == 4
    assert [line for line, _ in first] == sorted(line for line, _ in first)
    assert metadata["source_rows"] == 20
    assert metadata["source_sha256"] == heldout.file_digest(data)
    assert len(heldout.select_rows(data, 25, 2026)[0]) == 20


@pytest.mark.parametrize("extra", ["\n", json.dumps(row()) + "\n"])
def test_selection_never_drops_blank_or_duplicate_rows(tmp_path, extra):
    data = tmp_path / "validation.jsonl"
    data.write_text(json.dumps(row()) + "\n" + extra)
    with pytest.raises(ValueError):
        heldout.select_rows(data, 1, 2026)


class TinyTokenizer:
    """An exact toy header tokenization; deliberately not a claim about Qwen."""

    pieces: ClassVar[dict[int, str]] = {
        0: "question",
        1: "<|im_start|>assistant\n",
        2: "A",
        3: "B",
        4: "C",
        5: "D",
        6: "<|im_end|>",
    }

    def decode(self, ids, **_kwargs):
        return "".join(self.pieces[token] for token in ids)

    def __call__(self, text, **_kwargs):
        ids = []
        while text:
            matches = [(token, value) for token, value in self.pieces.items() if text.startswith(value)]
            assert len(matches) == 1
            token, value = matches[0]
            ids.append(token)
            text = text[len(value) :]
        return {"input_ids": ids}


def test_answer_alignment_identifies_exact_prefix_next_token():
    position, candidates = heldout.answer_alignment(TinyTokenizer(), [0, 1, 4, 6], [-100, -100, 4, 6], "C")
    assert position == 2
    assert candidates == dict(zip("ABCD", [2, 3, 4, 5], strict=True))
    with pytest.raises(ValueError, match="gold answer token"):
        heldout.answer_alignment(TinyTokenizer(), [0, 1, 4, 6], [-100, -100, 4, 6], "B")
    with pytest.raises(ValueError, match="assistant header"):
        heldout.answer_alignment(TinyTokenizer(), [0, 1, 4, 6], [-100, -100, -100, 6], "C")


@pytest.mark.parametrize("choices", ["AB", "ABC", "ABCD"])
def test_alignment_only_tokenizes_actual_choices(choices):
    gold = choices[-1]
    token = 2 + "ABCD".index(gold)
    position, candidates = heldout.answer_alignment(
        TinyTokenizer(),
        [0, 1, token, 6],
        [-100, -100, token, 6],
        gold,
        choices=tuple(choices),
    )
    assert position == 2
    assert candidates == dict(zip(choices, range(2, 2 + len(choices)), strict=True))
    if len(choices) < 4:
        with pytest.raises(ValueError, match="actual question choices"):
            heldout.answer_alignment(
                TinyTokenizer(),
                [0, 1, 5, 6],
                [-100, -100, 5, 6],
                "D",
                choices=tuple(choices),
            )


def test_answer_alignment_refuses_context_changed_candidate_tokenization():
    class BadTokenizer(TinyTokenizer):
        def __call__(self, text, **kwargs):
            result = super().__call__(text, **kwargs)
            if text.endswith("B"):
                result["input_ids"] = [0, 5]
            return result

    with pytest.raises(ValueError, match="context-stable"):
        heldout.answer_alignment(BadTokenizer(), [0, 1, 4, 6], [-100, -100, 4, 6], "C")


def test_score_uses_previous_logit_position_and_full_vocabulary_nll():
    logits = torch.zeros(1, 4, 7)
    logits[0, 1, 4] = 3.0
    logits[0, 2, 2] = 100.0  # An erroneous unshifted scorer would choose A.
    scores = heldout.score_answer(logits, 2, dict(zip("ABCD", [2, 3, 4, 5], strict=True)), "C")
    assert scores["prediction_logit_position"] == 1
    assert scores["restricted_choice_prediction"] == "C"
    assert scores["restricted_choice_correct"] and scores["strict_first_token_correct"]
    assert scores["valid_first_letter"]
    assert scores["answer_nll"] == pytest.approx(math.log(math.exp(3) + 6) - 3, abs=2e-7)
    assert scores["restricted_choice_nll"] == pytest.approx(math.log(math.exp(3) + 3) - 3, abs=2e-7)


def test_choice_correct_does_not_imply_strict_correct_or_valid_letter():
    logits = torch.zeros(1, 4, 7)
    logits[0, 1, 4], logits[0, 1, 0] = 3.0, 4.0
    scores = heldout.score_answer(logits, 2, dict(zip("ABCD", [2, 3, 4, 5], strict=True)), "C")
    assert scores["restricted_choice_correct"]
    assert not scores["strict_first_token_correct"]
    assert not scores["valid_first_letter"]
    assert scores["strict_first_token_letter"] is None


@pytest.mark.parametrize("choices", ["AB", "ABC", "ABCD"])
def test_scoring_normalizes_only_actual_choices(choices):
    candidates = dict(zip(choices, range(2, 2 + len(choices)), strict=True))
    logits = torch.zeros(1, 4, 7)
    logits[0, 1, candidates[choices[-1]]] = 3
    scores = heldout.score_answer(logits, 2, candidates, choices[-1])
    assert scores["restricted_choice_correct"] and scores["strict_first_token_correct"]
    assert tuple(scores["candidate_logits"]) == tuple(choices)
    assert scores["restricted_choice_nll"] == pytest.approx(math.log(math.exp(3) + len(choices) - 1) - 3, abs=2e-7)
    assert scores["answer_nll"] == pytest.approx(math.log(math.exp(3) + 6) - 3, abs=2e-7)
    if len(choices) < 4:
        logits[0, 1, 5] = 4  # D is an invalid answer for this question.
        scores = heldout.score_answer(logits, 2, candidates, choices[-1])
        assert scores["restricted_choice_correct"]
        assert not scores["valid_first_letter"] and not scores["strict_first_token_correct"]
        assert scores["strict_first_token_letter"] is None


@pytest.mark.parametrize(
    "candidates,gold",
    [
        ({"A": 2}, "A"),
        ({"A": 2, "C": 4}, "A"),
        ({"B": 3, "A": 2}, "A"),
        ({"A": 2, "B": 2}, "A"),
        ({"A": 2, "B": 3}, "C"),
    ],
)
def test_invalid_candidate_set_or_gold_is_rejected(candidates, gold):
    with pytest.raises(ValueError):
        heldout.score_answer(torch.zeros(1, 4, 7), 2, candidates, gold)


def test_nonfinite_answer_logits_are_refused():
    logits = torch.zeros(1, 4, 7)
    logits[0, 1, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        heldout.score_answer(logits, 2, dict(zip("ABCD", [2, 3, 4, 5], strict=True)), "C")


def test_aggregate_uses_token_weighting_and_exact_content_clusters():
    logits = torch.zeros(1, 4, 7)
    candidates = dict(zip("ABCD", [2, 3, 4, 5], strict=True))
    scores = heldout.score_answer(logits, 2, candidates, "C")
    rows = [
        {
            **scores,
            "native_supervised_tokens": 2,
            "native_sft_ce": 1,
            "media_sha256": "same",
        },
        {
            **scores,
            "native_supervised_tokens": 6,
            "native_sft_ce": 3,
            "media_sha256": "same",
        },
    ]
    report = heldout.aggregate(rows)
    assert report["token_weighted_ce"] == 2.5
    assert report["unique_exact_video_content_clusters"] == 1
    assert report["native_supervised_tokens"] == 8


@pytest.mark.parametrize("reload_adapter", [False, True])
@pytest.mark.parametrize("touch_source", [False, True])
def test_run_binds_verified_portable_processor_and_reports_same_contract(
    tmp_path, monkeypatch, reload_adapter, touch_source
):
    from types import SimpleNamespace

    from cosmos_framework.posttrain.algorithm import sft as algorithm
    from cosmos_framework.posttrain.checkpoint import adapter as checkpoint
    from cosmos_framework.posttrain.config import build
    from cosmos_framework.posttrain.configs.sft_qwen3_vl_8b_qlora import make_config
    from cosmos_framework.posttrain.data.packer import sft as packers
    from cosmos_framework.posttrain.entrypoints import sft as entrypoint
    from cosmos_framework.posttrain.model.quantization.loader import resolved_execution_policy
    from cosmos_framework.posttrain.utils import adapter_identity, determinism, dist
    from cosmos_framework.utils import config as config_loader
    from cosmos_framework.utils import misc

    for name, value in {
        "WORLD_SIZE": "1",
        "RANK": "0",
        "LOCAL_RANK": "0",
        "COSMOS_TRAINING_CODE_REVISION": "a" * 40,
        "COSMOS_TRAINING_WHEEL_SHA256": "b" * 64,
    }.items():
        monkeypatch.setenv(name, value)
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps(row()) + "\n")
    (tmp_path / "clip.mp4").write_bytes(b"controlled media identity; no decoder claim")
    config = make_config()
    model_source = str(tmp_path / "verified-model")
    base_identity = {"repository": "controlled-model", "revision": "c" * 40}
    portable = {
        "proof": "verified-content-identity",
        "processor_settings": {"frames": 8},
    }
    packer = SimpleNamespace(validation_semantics_identity=lambda: {"raw_local_path": model_source})
    events = []

    class Codec:
        def __init__(self):
            self.identity = {
                "base": base_identity,
                "adapter": {"initialization": "kaiming_A_zero_B"},
                "initial_adapters": {},
                "quantized_tensors": {},
            }
            self.parameters = {}

        def bind_processor_identity(self, value):
            assert value is portable
            self.identity["processor_contract"] = value
            events.append("bind")

        def assert_live(self, net, optimizer):
            assert net._cosmos_adapter_checkpoint_codec is self
            assert optimizer is sentinel_optimizer
            assert self.identity["processor_contract"] is portable
            events.append("live")

        def validate_frozen_base(self):
            events.append("frozen")

    codec = Codec()
    net = torch.nn.Linear(1, 1)
    net._cosmos_adapter_runtime_precision = {"fixture": "CPU-only lifecycle"}
    net._cosmos_adapter_checkpoint_codec = codec
    model = SimpleNamespace(net=net)
    sentinel_optimizer = object()

    def identity(actual_packer, *, source_path, source_identity):
        assert actual_packer is packer
        assert source_path == model_source
        assert source_identity is base_identity
        events.append("identity")
        return portable

    def load(path, actual_codec, *, manager):
        assert path == str(tmp_path / "adapter")
        assert actual_codec is codec and codec.identity["processor_contract"] is portable
        events.append("reload")

    scores = heldout.score_answer(torch.zeros(1, 4, 7), 2, dict(zip("ABCD", [2, 3, 4, 5], strict=True)), "C")
    scores.update(native_supervised_tokens=2, native_sft_ce=1.25)
    monkeypatch.setattr(config_loader, "load_config", lambda *args: config)
    monkeypatch.setattr(entrypoint, "validate_config", lambda cfg: cfg.rl)
    monkeypatch.setattr(entrypoint, "sft_policy_source", lambda cfg: (model_source, False))
    monkeypatch.setattr(packers, "build_sft_packer", lambda *args, **kwargs: packer)

    def backbone(cfg):
        assert cfg.model.model_config.adapter_compute_dtype == ("bfloat16" if reload_adapter else "float32")
        assert cfg.model.model_config.sft_logits_mode == ("supervised" if reload_adapter else "full")
        return model

    monkeypatch.setattr(build, "build_backbone", backbone)
    monkeypatch.setattr(build, "build_optimizer", lambda *args: sentinel_optimizer)
    monkeypatch.setattr(adapter_identity, "adapter_processor_identity", identity)
    optimized = make_config().model.model_config
    optimized.adapter_compute_dtype = "bfloat16"
    optimized.sft_logits_mode = "supervised"
    saved_identity = {"format": "cosmos.adapter-model.v2", "execution_policy": resolved_execution_policy(optimized)}
    monkeypatch.setattr(checkpoint, "read_adapter_metadata", lambda path: {"identity": saved_identity})
    monkeypatch.setattr(checkpoint, "load_adapter_weights", load)
    monkeypatch.setattr(
        algorithm,
        "make_objective",
        lambda *args, **kwargs: SimpleNamespace(identity={"kind": "fixture"}),
    )

    def enable_determinism(*, strict):
        assert strict is True
        events.append("strict_determinism")

    def initialize_process_group():
        assert events == ["strict_determinism"]

    monkeypatch.setattr(determinism, "enable_determinism", enable_determinism)
    monkeypatch.setattr(dist, "init_local_process_group", initialize_process_group)
    monkeypatch.setattr(dist, "destroy_process_group_if_initialized", lambda: None)
    monkeypatch.setattr(misc, "set_random_seed", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "CPU lifecycle fixture")
    monkeypatch.setattr(heldout.importlib.metadata, "version", lambda name: "fixture")

    def score(*args):
        if touch_source:
            stamp = source.stat()
            os.utime(source, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 1))
        return scores

    monkeypatch.setattr(heldout, "evaluate_row", score)

    output = tmp_path / "evaluation"
    args = SimpleNamespace(
        data=str(source),
        output=str(output),
        max_rows=1,
        seed=2026,
        config="controlled-recipe",
        override=[],
        arm="controlled-arm",
        checkpoint=str(tmp_path / "adapter") if reload_adapter else None,
    )
    if touch_source:
        with pytest.raises(RuntimeError, match="data source changed during evaluation"):
            heldout.run(args)
        assert not (output / "summary.json").exists()
        return
    result = heldout.run(args)
    assert result["complete"] is True
    assert events == [
        "strict_determinism",
        "identity",
        "bind",
        "live",
        *(["reload"] if reload_adapter else []),
        "frozen",
    ]
    assert json.loads((output / "summary.json").read_text())["processor_contract"] == portable
    assert json.loads((output / "model_identity.json").read_text())["processor_contract"] == portable


@pytest.mark.parametrize("change_media", [False, True])
@pytest.mark.parametrize("choices", ["AB", "ABC", "ABCD"])
def test_native_objective_observation_is_single_forward_and_restores_state(
    change_media,
    choices,
):
    from types import SimpleNamespace

    from cosmos_framework.posttrain.algorithm.sft import make_objective

    gold_id = 2 + "ABCD".index(choices[-1])

    class Packer:
        def __init__(self):
            self.tokenizer = TinyTokenizer()
            self.process_calls = 0
            self.collate_calls = 0

        def validation_semantics_identity(self):
            return {"masking": "test-single-letter"}

        def sft_process_sample(self, _target):
            self.process_calls += 1
            return SimpleNamespace(input_ids=[0, 1, gold_id, 6], labels=[-100, -100, gold_id, 6])

        def sft_compute_max_len(self, _samples):
            return 4

        def sft_collate_fn(self, samples, _length, _ignore=-100):
            self.collate_calls += 1
            return {
                "input_ids": torch.tensor([samples[0].input_ids]),
                "labels": torch.tensor([samples[0].labels]),
                "attention_mask": torch.ones(1, 4, dtype=torch.long),
                "pixel_values_videos": torch.ones(1, 2),
            }

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            values = torch.zeros(1, 4, 7)
            values[0, 1, gold_id] = 3
            self.values = torch.nn.Parameter(values)
            self.calls = 0

        def forward(self, input_ids, attention_mask, pixel_values_videos):
            self.calls += 1
            assert not self.training
            assert input_ids.shape == attention_mask.shape == (1, 4)
            # Real tiny media dependence without video decoding or GPU-kernel claims.
            return SimpleNamespace(logits=self.values + pixel_values_videos.mean() * torch.arange(7) * 0.01)

    class Policy:
        def __init__(self):
            self.net = Net()

        def prepare_policy_batch(self, batch):
            if change_media:
                batch["pixel_values_videos"] = torch.zeros_like(batch["pixel_values_videos"])
            return batch

        def forward_policy_batch(self, batch):
            return self.net(**batch)

    packer, model = Packer(), Policy()
    collate = packer.sft_collate_fn
    initial = model.net.values.detach().clone()
    objective = make_objective(packer)
    if change_media:
        with pytest.raises(RuntimeError, match="changed the observed pixel_values_videos"):
            heldout.evaluate_row(model, packer, objective, choice_row(choices))
    else:
        scores = heldout.evaluate_row(model, packer, objective, choice_row(choices))
        assert tuple(scores["candidate_token_ids"]) == tuple(choices)
        assert scores["restricted_choice_correct"]
        assert scores["strict_first_token_correct"]
        assert scores["native_supervised_tokens"] == 2
        assert scores["prediction_logit_position"] == 1
        assert scores["native_sft_ce"] > scores["answer_nll"]
        assert "pixel_values_videos" in scores["inputs"]
    assert model.net.calls == packer.process_calls == packer.collate_calls == 1
    assert packer.sft_collate_fn == collate
    assert model.net.training
    assert not model.net._forward_hooks
    assert torch.equal(model.net.values, initial)
    assert model.net.values.grad is None


@pytest.mark.parametrize("choices", ["AB", "ABC", "ABCD"])
@pytest.mark.parametrize("selected", [False, True])
@pytest.mark.parametrize("change_media", [False, True])
def test_native_selected_head_observes_one_complete_media_forward_and_correct_answer_row(
    choices, selected, change_media
):
    from types import SimpleNamespace

    from cosmos_framework.posttrain.algorithm.sft import make_objective
    from cosmos_framework.posttrain.model.qwen3_vl import Qwen3VLPolicy

    gold_id = 2 + "ABCD".index(choices[-1])

    class Packer:
        tokenizer = TinyTokenizer()
        calls = 0

        def validation_semantics_identity(self):
            return {"masking": "single-letter-private-fixture"}

        def sft_process_sample(self, _target):
            return SimpleNamespace(input_ids=[0, 1, gold_id, 6], labels=[-100, -100, gold_id, 6])

        def sft_compute_max_len(self, samples):
            return 4

        def sft_collate_fn(self, samples, _length, _ignore=-100):
            self.calls += 1
            return {
                "input_ids": torch.tensor([samples[0].input_ids]),
                "labels": torch.tensor([samples[0].labels]),
                "attention_mask": torch.ones(1, 4, dtype=torch.long),
                "pixel_values_videos": torch.ones(1, 2),
            }

    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            values = torch.zeros(1, 4, 7)
            values[0, 1, gold_id] = 3
            values[0, 2, 6] = 8  # End token deliberately differs from the answer prediction.
            self.values = torch.nn.Parameter(values)
            self.calls = 0

        def forward(self, input_ids, attention_mask, pixel_values_videos, *, use_cache=False, return_dict=True):
            self.calls += 1
            assert not self.training and not use_cache
            assert input_ids.shape == attention_mask.shape == (1, 4)
            return SimpleNamespace(last_hidden_state=self.values + pixel_values_videos.mean() * torch.arange(7) * 0.01)

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Backbone()
            self.lm_head = torch.nn.Linear(7, 7, bias=False).requires_grad_(False)
            self.lm_head.weight.copy_(torch.eye(7))

        def forward(self, **batch):
            return SimpleNamespace(logits=self.lm_head(self.model(**batch).last_hidden_state))

    model = Qwen3VLPolicy(Net(), "qwen3_vl", sft_logits_mode="supervised" if selected else "full")
    prepare = model.prepare_policy_batch

    def prepare_batch(batch):
        result = prepare(batch)
        if change_media:
            result["pixel_values_videos"] = torch.zeros_like(result["pixel_values_videos"])
        return result

    model.prepare_policy_batch = prepare_batch
    packer = Packer()
    collate = packer.sft_collate_fn
    before = {name: parameter.clone() for name, parameter in model.net.named_parameters()}
    if change_media:
        with pytest.raises(RuntimeError, match="changed the observed pixel_values_videos"):
            heldout.evaluate_row(model, packer, make_objective(packer), choice_row(choices))
    else:
        observed = heldout.evaluate_row(model, packer, make_objective(packer), choice_row(choices))
        assert observed["candidate_token_ids"] == dict(zip(choices, range(2, 2 + len(choices)), strict=True))
        assert observed["strict_first_token_correct"] and observed["restricted_choice_correct"]
        assert observed["answer_token_position"] == 2 and observed["prediction_logit_position"] == 1
        assert observed["native_supervised_tokens"] == 2
        assert model.net.model.calls == 1
    assert packer.calls == 1 and packer.sft_collate_fn == collate
    assert model.net.training
    assert (
        not model.net._forward_hooks and not model.net.model._forward_pre_hooks and not model.net.lm_head._forward_hooks
    )
    assert all(torch.equal(before[name], parameter) for name, parameter in model.net.named_parameters())


def test_all_rows_requires_sealed_role_and_selection_for_untouched(tmp_path):
    from argparse import Namespace

    from tools.native_qlora_qualification.common import digest_file
    from tools.native_qlora_qualification.split import write_partition

    records = [row(identifier=f"item{i}") for i in range(3)]
    groups = {item["id"]: heldout.json_digest(item["id"]) for item in records}
    write_partition(tmp_path, "development", records, groups, groups, provenance={})
    manifest = tmp_path / "development.manifest.json"
    args = Namespace(
        all_rows=True,
        split_manifest=str(manifest),
        split_manifest_sha256=digest_file(manifest),
        split="development",
        config="recipe.py",
        override=[],
    )
    selected, identity = heldout.select_scoring_rows(args, tmp_path / "development.jsonl")
    assert [value for _, value in selected] == records
    assert identity["selected_rows"] == 3
    args.split = "final"
    with pytest.raises(ValueError, match="role"):
        heldout.select_scoring_rows(args, tmp_path / "development.jsonl")
    identity["split_role"] = "untouched_diagnostic"
    with pytest.raises(ValueError, match="selection receipt"):
        heldout.validate_selection_receipt(args, identity)


def test_selection_receipt_binds_data_config_and_development(tmp_path):
    from argparse import Namespace

    args = Namespace(config="recipe.py", override=["rl.seed=42"], selection_receipt=str(tmp_path / "selection.json"))
    selection = {"split_role": "untouched_diagnostic", "split_manifest_sha256": "a" * 64}
    receipt = {
        "protocol": "cosmos.qlora-selection.v1",
        "split_manifest_sha256": "a" * 64,
        "development_manifest_sha256": "b" * 64,
        "locked_utc": "2026-01-01T00:00:00+00:00",
        "selected_scoring_config_sha256": [heldout.json_digest({"config": args.config, "overrides": args.override})],
        "development_evidence_sha256": ["c" * 64],
    }
    path = Path(args.selection_receipt)
    path.write_text(json.dumps(receipt))
    args.selection_receipt_sha256 = heldout.file_digest(path)
    heldout.validate_selection_receipt(args, selection)
    assert selection["selection_receipt_sha256"] == args.selection_receipt_sha256
    args.override = ["rl.seed=43"]
    with pytest.raises(ValueError, match="not selected"):
        heldout.validate_selection_receipt(args, selection)
    args.override = ["rl.seed=42"]
    receipt["split_manifest_sha256"] = "d" * 64
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="exact precommitted"):
        heldout.validate_selection_receipt(args, selection)
    args.selection_receipt_sha256 = heldout.file_digest(path)
    with pytest.raises(ValueError, match="different untouched"):
        heldout.validate_selection_receipt(args, selection)
