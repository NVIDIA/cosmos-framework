# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Full-pass native WTS scoring and its strict, deterministic recovery contract."""

from __future__ import annotations

import copy
import json
import random
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from cosmos_framework.posttrain.checkpoint.rng import collect_rng_state
from cosmos_framework.posttrain.config.sft import SFTConfig, SFTDatasetConfig, SFTValidationConfig, SFTWTSConfig
from cosmos_framework.posttrain.data.datasets.sft import sha256_file
from cosmos_framework.posttrain.data.packer.sft import QwenSFTDataPacker, SFTPolicyInput
from cosmos_framework.posttrain.model.checkpoint import AdapterCheckpointCodec
from cosmos_framework.posttrain.roles.env.sft import build_sft_env
from cosmos_framework.posttrain.roles.trainer.validation import development
from cosmos_framework.posttrain.roles.trainer.validation.development import DevelopmentEvaluation
from cosmos_framework.posttrain.roles.trainer.validation.wts import aggregate, evaluate_row, json_digest
from cosmos_framework.posttrain.tests.helpers.sft_objective import MemoryPacker, Policy
from cosmos_framework.posttrain.tests.test_entrypoints_sft_pause import _exact

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


class Tokenizer:
    pieces = {0: "question", 1: "<|im_start|>assistant\n", 2: "A", 3: "B", 4: "C", 5: "D", 6: "<|im_end|>"}

    def decode(self, ids, **_):
        return "".join(self.pieces[i] for i in ids)

    def __call__(self, text, **_):
        ids = []
        while text:
            for key, value in self.pieces.items():
                if text.startswith(value):
                    ids.append(key)
                    text = text[len(value) :]
                    break
            else:
                raise ValueError(text)
        return {"input_ids": ids}


class VideoPacker(MemoryPacker, QwenSFTDataPacker):
    tokenizer = Tokenizer()

    def __init__(self):
        MemoryPacker.__init__(self)
        self.max_length = 32
        self.processor_kwargs = {}
        self.mask_think_opener = False

    def sft_process_sample(self, sample):
        answer = 2 + "ABCD".index(sample.messages[-1]["content"])
        return SFTPolicyInput([0, 1, answer, 6], [-100, -100, answer, 6], {"pixel_values_videos": torch.ones(2, 3)})

    def sft_collate_fn(self, samples, computed_max_len, ignore_label_id=-100):
        batch = super().sft_collate_fn(samples, computed_max_len, ignore_label_id)
        batch["pixel_values_videos"] = torch.cat(batch["pixel_values_videos"])
        return batch


class VideoNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(7, 8)
        self.dropout = nn.Dropout(0.2)
        self.head = nn.Linear(8, 7)

    def forward(self, input_ids, attention_mask, pixel_values_videos):
        hidden = self.dropout(self.embedding(input_ids)) + pixel_values_videos.mean()
        return SimpleNamespace(logits=self.head(hidden))


def _settings(**changes):
    return SFTWTSConfig(
        **{
            "enabled": True,
            "data": SFTDatasetConfig(path="development.jsonl", shuffle=False),
            "source_sha256": "a" * 64,
            "manifest_path": "manifest.json",
            "manifest_sha256": "b" * 64,
            "expected_rows": 2,
            "steps": [2, 4],
            "min_correct": 2,
            "max_token_ce": 0.0,
            **changes,
        }
    )


def _fixture(tmp_path, monkeypatch, **changes):
    source = tmp_path / "development.jsonl"
    media = tmp_path / "video.mp4"
    media.write_bytes(b"immutable source video; toy packer does not decode media")
    raw = [
        {
            "id": str(i),
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Which?\nA. left\nB. right"},
                        {"type": "video", "video": str(media)},
                    ],
                },
                {"role": "assistant", "content": answer},
            ],
        }
        for i, answer in enumerate(("A", "B"))
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in raw))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "protocol": "cosmos.wts-group-split.v1",
                "role": "development",
                "data_sha256": sha256_file(str(source)),
                "group_count": 1,
                "rows": [
                    {
                        "id": row["id"],
                        "raw_row_sha256": json_digest(row),
                        "media_sha256": sha256_file(str(media)),
                        "group_sha256": "c" * 64,
                    }
                    for row in raw
                ],
            }
        )
    )
    train = tmp_path / "training.jsonl"
    train.write_text(
        json.dumps(
            {
                "id": "train",
                "messages": [
                    {"role": "user", "content": "Unrelated training input"},
                    {"role": "assistant", "content": "Unrelated training answer"},
                ],
            }
        )
        + "\n"
    )
    settings = _settings(
        data=SFTDatasetConfig(path=str(source), shuffle=False),
        source_sha256=sha256_file(str(source)),
        manifest_path=str(manifest),
        manifest_sha256=sha256_file(str(manifest)),
        **changes,
    )
    rl = SFTConfig(max_steps=4)
    training = build_sft_env(SFTDatasetConfig(path=str(train)), 17)
    snapshot = {"model": "toy immutable CPU model", "processor": "toy processor"}
    monkeypatch.setattr(
        "cosmos_framework.posttrain.utils.adapter_source.verify_adapter_snapshot", lambda _: dict(snapshot)
    )
    monkeypatch.setattr(
        "cosmos_framework.posttrain.utils.adapter_identity.adapter_processor_identity", lambda *a, **kw: {"toy": True}
    )
    (tmp_path / "model").mkdir()
    packer = VideoPacker()
    monkeypatch.setattr("cosmos_framework.posttrain.data.packer.sft.build_sft_packer", lambda *a, **kw: packer)
    evaluator = DevelopmentEvaluation(
        settings, rl, training, model_source=str(tmp_path / "model"), trust=False, output=str(tmp_path / "native")
    )
    torch.manual_seed(44)
    policy = Policy(VideoNet())
    policy.net._cosmos_adapter_parameters = dict(policy.net.named_parameters())
    policy.net._cosmos_adapter_checkpoint_codec = AdapterCheckpointCodec(policy.net, identity={"toy": True})
    optimizer = torch.optim.AdamW(policy.parameters(), lr=0.01)
    loss, _ = evaluator.objective(policy, [evaluator.source._trajectory(0)])
    loss.backward()
    optimizer.step()  # Leave populated optimizer moments AND existing gradients.
    policy.train()
    policy.net.dropout.eval()  # Mixed modes must survive a whole pass, including exceptions.
    return evaluator, policy, optimizer, snapshot


def _snapshot(evaluator, policy, optimizer):
    return (
        collect_rng_state(),
        evaluator.training.state_dict(),
        evaluator.source.state_dict(),
        [m.training for m in policy.modules()],
        [p.detach().clone() for p in policy.parameters()],
        [p.grad.clone() for p in policy.parameters()],
        copy.deepcopy(optimizer.state_dict()),
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("enabled", 1),
        ("stop_on_target", 0),
        ("expected_rows", True),
        ("cache_bytes", -1),
        ("min_correct", 3),
        ("max_token_ce", float("nan")),
        ("max_token_ce", float("inf")),
        ("max_token_ce", True),
        ("steps", [2, 2]),
        ("steps", [4, 2]),
        ("steps", [True, 4]),
        ("source_sha256", "bad"),
        ("manifest_sha256", "A" * 64),
        ("manifest_path", 4),
    ],
)
def test_typed_wts_contract_refuses_invalid_values(field, value):
    with pytest.raises((ValueError, TypeError)):
        _settings(**{field: value})


@pytest.mark.parametrize(
    "changes",
    [
        {"save_async": True},
        {"checkpoint_format": "consolidated"},
        {"deterministic": False},
        {"ce_exponent": 0.5},
        {"max_steps": 5},
    ],
)
def test_wts_refuses_incompatible_checkpoint_and_schedule_contract(changes):
    validation = SFTValidationConfig(
        enabled=True, every_steps=2, data=SFTDatasetConfig(path="validation.jsonl"), wts=_settings()
    )
    with pytest.raises(ValueError, match="validation.wts"):
        SFTConfig(
            **{"max_steps": 4, "save_async": False, "checkpoint_format": "dcp", "validation": validation, **changes}
        )


def test_full_pass_matches_shared_scorer_and_preserves_all_training_state(tmp_path, monkeypatch):
    evaluator, policy, optimizer, _ = _fixture(tmp_path, monkeypatch)
    expected = [evaluate_row(policy, evaluator.packer, evaluator.objective, row["row"]) for row in evaluator.selected]
    before = _snapshot(evaluator, policy, optimizer)
    metrics = evaluator.evaluate(policy, optimizer, 2)
    _exact(_snapshot(evaluator, policy, optimizer), before)
    directory = evaluator.output / evaluator.artifacts[2]["directory"]
    scores = [json.loads(line) for line in (directory / "scores.jsonl").read_text().splitlines()]
    for row, want in zip(scores, expected, strict=True):
        assert {key: row[key] for key in want} == want
    assert evaluator.history[0]["metrics"] == aggregate(scores)
    assert "preprocessing_cache" not in json.loads((directory / "summary.json").read_text())
    assert metrics["val/wts_rows"] == 2
    assert evaluator.history[0]["metrics"]["native_supervised_tokens"] == 4
    assert not (evaluator.output / "wts/target-result.json").exists(), "evaluation alone is not a committed target"
    assert evaluator.state_dict() == {"version": 1, "observations": evaluator.history}
    assert all("directory" not in row and "available_monotonic_seconds" not in row for row in evaluator.history)


@pytest.mark.parametrize("failure", ["forward", "source", "media", "processor", "cursor"])
def test_failed_pass_restores_rng_modes_and_cursors_without_committing_history(tmp_path, monkeypatch, failure):
    evaluator, policy, optimizer, snapshot = _fixture(tmp_path, monkeypatch)
    before = _snapshot(evaluator, policy, optimizer)
    forward = policy.forward_policy_batch
    calls = 0

    def failing(batch):
        nonlocal calls
        calls += 1
        if calls == 2:
            random.random()
            np.random.rand()
            torch.rand(7)
            if failure == "forward":
                evaluator.training.get_samples(1)
                evaluator.source.cursor.next_index()
                policy.net.dropout.train()
                raise RuntimeError("second row failed")
            if failure == "source":
                Path(evaluator.settings.data.path).write_text("changed")
            elif failure == "media":
                Path(next(iter(evaluator.media))).write_bytes(b"changed")
            elif failure == "processor":
                snapshot["processor"] = "changed"
            elif failure == "cursor":
                evaluator.training.get_samples(1)
        return forward(batch)

    monkeypatch.setattr(policy, "forward_policy_batch", failing)
    with pytest.raises((RuntimeError, ValueError)):
        evaluator.evaluate(policy, optimizer, 2)
    # The native cursor checks source content as well; source mutation is a hard failure,
    # not an instruction to accept a newly hashed training identity.
    _exact(_snapshot(evaluator, policy, optimizer), before)
    assert evaluator.history == [] and evaluator.artifacts == {}
    assert not (evaluator.output / "wts/target-result.json").exists()


def test_constructor_rejects_wrong_manifest_membership_and_training_overlap(tmp_path, monkeypatch):
    evaluator, _, _, _ = _fixture(tmp_path, monkeypatch)
    manifest = Path(evaluator.settings.manifest_path)
    payload = json.loads(manifest.read_text())
    payload["rows"].reverse()
    manifest.write_text(json.dumps(payload))
    settings = replace(evaluator.settings, manifest_sha256=sha256_file(str(manifest)))
    with pytest.raises(ValueError, match="sealed development"):
        DevelopmentEvaluation(
            settings,
            SFTConfig(max_steps=4),
            evaluator.training,
            model_source=evaluator.model_source,
            trust=False,
            output=str(tmp_path / "other"),
        )
    with pytest.raises(ValueError, match="not held out"):
        DevelopmentEvaluation(
            evaluator.settings,
            SFTConfig(max_steps=4),
            evaluator.source,
            model_source=evaluator.model_source,
            trust=False,
            output=str(tmp_path / "other"),
        )


def test_restore_requires_complete_finite_content_bound_scheduled_history(tmp_path, monkeypatch):
    evaluator, policy, optimizer, _ = _fixture(tmp_path, monkeypatch)
    evaluator.evaluate(policy, optimizer, 2)
    valid = evaluator.state_dict()
    evaluator.load_state_dict(valid, iteration=3)
    for field, value in [
        ("scores_sha256", ""),
        ("token_weighted_ce", float("nan")),
        ("token_weighted_ce", True),
        ("target_met", True),
        ("correct", 3),
    ]:
        invalid = copy.deepcopy(valid)
        invalid["observations"][0][field] = value
        with pytest.raises(RuntimeError):
            evaluator.load_state_dict(invalid, iteration=3)
    with pytest.raises(RuntimeError, match="prefix"):
        evaluator.load_state_dict({"version": 1, "observations": []}, iteration=3)
    evaluator.load_state_dict({"version": 1, "observations": []}, iteration=2, pending_step=2)
    evaluator.load_state_dict(valid, iteration=2)  # completed evaluation awaiting publication
    with pytest.raises(RuntimeError, match="prefix"):
        evaluator.load_state_dict(valid, iteration=4)


def test_screen_mode_records_first_pass_without_stopping_grid(tmp_path, monkeypatch):
    evaluator, policy, optimizer, _ = _fixture(
        tmp_path, monkeypatch, min_correct=0, max_token_ce=100.0, stop_on_target=False
    )
    evaluator.evaluate(policy, optimizer, 2)
    journal = evaluator.state_dict()
    result = evaluator.publish(step=2, checkpoint=str(tmp_path / "cp2"), started_at=0, resumed=False)
    assert result["first_passing_step"] == 2 and result["complete"] is False and result["stop_on_target"] is False
    evaluator.evaluate(policy, optimizer, 4)
    evaluator.load_state_dict(evaluator.state_dict(), iteration=4)
    result = evaluator.publish(step=4, checkpoint=str(tmp_path / "cp4"), started_at=0, resumed=False)
    assert result["first_passing_step"] == 2 and result["complete"] is True and result["horizon_censored"] is False
    evaluator.settings.stop_on_target = True
    with pytest.raises(RuntimeError, match="first-pass"):
        evaluator.load_state_dict(evaluator.state_dict(), iteration=4)
    evaluator.load_state_dict(journal, iteration=2)
    with pytest.raises(RuntimeError, match="first target pass"):
        evaluator.evaluate(policy, optimizer, 4)


def test_readback_and_ancestor_durability_precede_availability(tmp_path, monkeypatch):
    evaluator, policy, optimizer, _ = _fixture(tmp_path, monkeypatch)
    evaluator.evaluate(policy, optimizer, 2)
    events = []
    codec = policy.net._cosmos_adapter_checkpoint_codec
    metadata = codec.metadata(optimizer)
    validate_metadata = codec.validate_metadata

    def validate_with_live_optimizer(saved, live_optimizer, *, purpose):
        assert live_optimizer is optimizer
        events.append(("identity", {"purpose": purpose}))
        validate_metadata(saved, live_optimizer, purpose=purpose)

    monkeypatch.setattr(
        "cosmos_framework.posttrain.checkpoint.adapter.read_adapter_metadata",
        lambda *a, **kw: events.append(("read", kw)) or metadata,
    )
    monkeypatch.setattr(codec, "validate_metadata", validate_with_live_optimizer)
    manager = SimpleNamespace(
        load_component=lambda *a: {
            "validation": {"wts": evaluator.state_dict(), "progress": {"pending_step": None}, "publication": None}
        }
    )
    monkeypatch.setattr(development, "_fsync_directory", lambda path: events.append(("fsync", path)))
    evaluator.verify_checkpoint("cp", manager, policy, optimizer)
    synced = [value for name, value in events if name == "fsync"]
    leaf = evaluator.output / evaluator.artifacts[2]["directory"]
    assert synced == [leaf, leaf.parent, leaf.parent.parent, evaluator.output]
    assert events[0][0] == "read" and events[0][1]["require_resume"] is True
    assert events[1] == ("identity", {"purpose": "resume"})
    assert "available_monotonic_seconds" not in evaluator.artifacts[2]
    evaluator.publish(step=2, checkpoint="cp", started_at=0, resumed=False)
    assert evaluator.artifacts[2]["available_monotonic_seconds"] > 0
    bad = SimpleNamespace(
        load_component=lambda *a: {"validation": {"wts": evaluator.state_dict(), "progress": {"pending_step": 2}}}
    )
    with pytest.raises(RuntimeError, match="readback"):
        evaluator.verify_checkpoint("cp", bad, policy, optimizer)


@pytest.mark.parametrize("defect", ["missing", "membership", "options"])
def test_checkpoint_readback_requires_exact_live_optimizer_before_availability(tmp_path, monkeypatch, defect):
    evaluator, policy, optimizer, _ = _fixture(tmp_path, monkeypatch)
    evaluator.evaluate(policy, optimizer, 2)
    metadata = policy.net._cosmos_adapter_checkpoint_codec.metadata(optimizer)
    monkeypatch.setattr(
        "cosmos_framework.posttrain.checkpoint.adapter.read_adapter_metadata", lambda *a, **kw: metadata
    )
    if defect == "missing":
        optimizer = None
        message = "requires the live optimizer"
    elif defect == "membership":
        optimizer.param_groups[0]["params"].pop()
        message = "optimizer membership"
    else:
        optimizer.param_groups[0]["weight_decay"] += 0.1
        message = "optimizer identity differs"

    def cannot_load(*args):
        raise AssertionError("trainer state must not be read before optimizer validation")

    with pytest.raises(ValueError, match=message):
        evaluator.verify_checkpoint("cp", SimpleNamespace(load_component=cannot_load), policy, optimizer)
    assert "available_monotonic_seconds" not in evaluator.artifacts[2]
    assert not (evaluator.output / "wts/target-result.json").exists()


def test_evaluation_cache_is_separate_persistent_bounded_and_score_exact(tmp_path, monkeypatch):
    evaluator, policy, optimizer, _ = _fixture(tmp_path, monkeypatch, cache_bytes=100_000)
    pipeline = evaluator.pipeline
    assert pipeline is evaluator.packer and pipeline._executor is None
    assert pipeline._max_pending == 1 and pipeline._spec.cache_bytes == 100_000
    before = _snapshot(evaluator, policy, optimizer)
    before_calls = pipeline.preprocessing_stats["timing"]["processing_calls"]
    evaluator.evaluate(policy, optimizer, 2)
    first = copy.deepcopy(evaluator.history[0])
    first_directory = evaluator.output / evaluator.artifacts[2]["directory"]
    first_summary_bytes = (first_directory / "summary.json").read_bytes()
    first_cache = json.loads(first_summary_bytes)["preprocessing_cache"]
    first_stats = copy.deepcopy(pipeline.preprocessing_stats)
    assert first_cache["stats"] == first_stats
    assert first_cache["policy"] == {
        "cache_bytes": 100_000,
        "prefetch": False,
        "workers": 1,
        "max_pending": 1,
    }
    assert first_stats["timing"]["processing_calls"] == before_calls + len(evaluator.selected)
    misses = pipeline.preprocessing_stats["samples"]["misses"]
    evaluator.evaluate(policy, optimizer, 4)
    _exact(_snapshot(evaluator, policy, optimizer), before)
    assert evaluator.pipeline is pipeline
    assert evaluator.history[1] == {**first, "step": 4}
    assert pipeline.preprocessing_stats["samples"]["hits"] >= 2
    assert pipeline.preprocessing_stats["samples"]["misses"] == misses
    assert sum(pipeline.preprocessing_stats[name]["bytes"] for name in ("samples", "videos")) <= 100_000
    second_directory = evaluator.output / evaluator.artifacts[4]["directory"]
    second_cache = json.loads((second_directory / "summary.json").read_text())["preprocessing_cache"]
    assert second_cache["stats"] == pipeline.preprocessing_stats
    assert second_cache["stats"]["timing"]["processing_calls"] == before_calls + 2 * len(evaluator.selected)
    assert second_cache["stats"]["samples"]["hits"] > first_stats["samples"]["hits"]
    assert (first_directory / "summary.json").read_bytes() == first_summary_bytes
    assert (first_directory / "scores.jsonl").read_bytes() == (second_directory / "scores.jsonl").read_bytes()
    assert "preprocessing_cache" not in evaluator.state_dict()
    assert all("preprocessing_cache" not in row for row in evaluator.history)
    evaluator.close()
    assert pipeline._closed


@pytest.mark.parametrize(
    "field,value", [("role", "regression"), ("protocol", "wrong"), ("data_sha256", "0" * 64), ("group_count", 2)]
)
def test_native_manifest_cannot_mislabel_a_non_development_source(tmp_path, monkeypatch, field, value):
    evaluator, _, _, _ = _fixture(tmp_path, monkeypatch)
    path = Path(evaluator.settings.manifest_path)
    manifest = json.loads(path.read_text())
    manifest[field] = value
    path.write_text(json.dumps(manifest))
    settings = replace(evaluator.settings, manifest_sha256=sha256_file(str(path)))
    with pytest.raises(ValueError):
        DevelopmentEvaluation(
            settings,
            SFTConfig(max_steps=4),
            evaluator.training,
            model_source=evaluator.model_source,
            trust=False,
            output=str(tmp_path / "other"),
        )


def test_native_config_accepts_omegaconf_step_lists_without_weakening_values():
    from omegaconf import OmegaConf

    assert _settings(steps=OmegaConf.create([2, 4])).steps == [2, 4]
    for values in ([True, 4], [2, 2], [4, 2]):
        with pytest.raises(ValueError):
            _settings(steps=OmegaConf.create(values))


def test_disabled_validation_preserves_legacy_contract_and_enabled_stop_policy_is_identity(monkeypatch):
    from cosmos_framework.posttrain.entrypoints import sft
    from cosmos_framework.posttrain.utils.identity import canonical_config_value

    monkeypatch.setenv("COSMOS_TRAINING_CODE_REVISION", "a" * 40)
    monkeypatch.setenv("COSMOS_TRAINING_WHEEL_SHA256", "b" * 64)
    monkeypatch.setattr(sft, "_validate_determinism", lambda config: None)
    selection = SimpleNamespace(
        native_spec=SimpleNamespace(model_config=SimpleNamespace(adapter=object(), source_identity={"toy": True})),
        model_id="model",
    )
    monkeypatch.setattr("cosmos_framework.posttrain.config.build.select_policy_backend", lambda config: selection)
    monkeypatch.setattr(
        "cosmos_framework.posttrain.utils.adapter_identity.adapter_processor_identity",
        lambda *a, **kw: {"processor": "fixed"},
    )
    rl = SFTConfig(max_steps=4, save_async=False, checkpoint_format="dcp")
    _, disabled = sft._adapter_run_contract(SimpleNamespace(rl=rl), object(), {"objective": "unchanged"})
    legacy = canonical_config_value(rl.validation)
    legacy["fields"].pop("wts")
    assert disabled["validation"] == legacy
    rl.validation = SFTValidationConfig(
        enabled=True, every_steps=2, data=SFTDatasetConfig(path="ordinary.jsonl"), wts=_settings()
    )
    _, enabled = sft._adapter_run_contract(SimpleNamespace(rl=rl), object(), {"objective": "unchanged"})
    assert enabled["validation"]["fields"]["wts"] == canonical_config_value(rl.validation.wts)
    rl.validation.wts.stop_on_target = False
    _, screen = sft._adapter_run_contract(SimpleNamespace(rl=rl), object(), {"objective": "unchanged"})
    expected = copy.deepcopy(enabled)
    expected["validation"]["fields"]["wts"]["fields"]["stop_on_target"] = False
    assert screen == expected
    assert screen["objective"] == disabled["objective"]
