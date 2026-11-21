# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""Scorer integration with the actual CPU cache; no model or video-decoder qualification."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.posttrain.data.packer.sft import QwenSFTDataPacker, SFTPolicyInput
from cosmos_framework.posttrain.data.packer.sft_preprocess import SFTPreprocessingPipeline
from tools.native_qlora_qualification import score
from tools.native_qlora_qualification.test_score import TinyTokenizer, row


class VideoProcessor:
    def __init__(self):
        self.calls = 0

    def to_dict(self):
        return {"fixture": "CPU pixels without decoding"}

    def __call__(self, *, videos):
        assert len(videos) == 1
        self.calls += 1
        return {"pixel_values_videos": torch.ones(1, 2)}


class Packer(QwenSFTDataPacker):
    """Synthetic processor boundary, actual native facade and objective below."""

    def __init__(self):
        self.tokenizer = TinyTokenizer()
        self.processor = SimpleNamespace(video_processor=VideoProcessor())
        self.max_length, self.processor_kwargs, self.mask_think_opener = 64, {}, False
        self.calls = self.collations = 0

    def validation_semantics_identity(self):
        return {"masking": "single-letter CPU fixture", "video": self.processor.video_processor.to_dict()}

    def sft_process_sample(self, target):
        self.calls += 1
        gold = 2 + "ABCD".index(target.messages[-1]["content"])
        path = target.messages[-2]["content"][0]["video"]
        pixels = self.processor.video_processor(videos=[path])
        return SFTPolicyInput([0, 1, gold, 6], [-100, -100, gold, 6], pixels)

    def sft_compute_max_len(self, samples):
        return 4

    def sft_collate_fn(self, samples, _length, _ignore=-100):
        self.collations += 1
        return {
            "input_ids": torch.tensor([samples[0].input_ids]),
            "labels": torch.tensor([samples[0].labels]),
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
            **samples[0].mm_kwargs,
        }


@pytest.fixture
def execution(tmp_path, monkeypatch):
    from cosmos_framework.posttrain.config import build
    from cosmos_framework.posttrain.configs.sft_qwen3_vl_8b_qlora import make_config
    from cosmos_framework.posttrain.data.packer import sft as packers
    from cosmos_framework.posttrain.entrypoints import sft as entrypoint
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
    source, media = tmp_path / "rows.jsonl", tmp_path / "clip.mp4"
    source.write_text("".join(json.dumps(row(f"row{i}", answer)) + "\n" for i, answer in enumerate("CBC")))
    media.write_bytes(b"source-bound synthetic media")
    model_source = tmp_path / "processor"
    model_source.mkdir()
    state = SimpleNamespace(events=[], failure=None, pipelines=[], packers=[], nets=[])

    class Codec:
        def __init__(self):
            self.identity = {
                "base": {"fixture": "no actual model"},
                "adapter": {"initialization": "kaiming_A_zero_B"},
                "initial_adapters": {},
                "quantized_tensors": {},
            }
            self.parameters = {}

        def bind_processor_identity(self, identity):
            self.identity["processor_contract"] = identity

        def assert_live(self, *_args):
            pass

        def validate_frozen_base(self):
            pass

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.arange(28).reshape(1, 4, 7).float() / 10, requires_grad=False)
            self._cosmos_adapter_checkpoint_codec = Codec()
            self._cosmos_adapter_runtime_precision = {"fixture": "CPU only"}

        def forward(self, input_ids, attention_mask, pixel_values_videos):
            assert not self.training
            assert input_ids.shape == attention_mask.shape == (1, 4)
            if state.failure == "forward":
                raise RuntimeError("fixture forward failure")
            if state.failure == "source":
                source.write_text(source.read_text() + "\n")
            if state.failure == "media":
                media.write_bytes(b"changed source media")
            return SimpleNamespace(logits=self.logits + pixel_values_videos.mean() * torch.arange(7) * 0.01)

    class Policy:
        def __init__(self):
            self.net = Net()
            state.nets.append(self.net)

        def prepare_policy_batch(self, batch):
            if state.failure == "changed_forward_input":
                batch["pixel_values_videos"] = torch.zeros_like(batch["pixel_values_videos"])
            return batch

        def forward_policy_batch(self, batch):
            return self.net(**batch)

    def packer(*_args, **_kwargs):
        value = Packer()
        state.packers.append(value)
        return value

    def identity(value, *, source_path, source_identity):
        assert value is state.packers[-1]
        assert type(value.processor.video_processor) is VideoProcessor
        assert source_path == str(model_source)
        return value.validation_semantics_identity()

    native_close = SFTPreprocessingPipeline.close

    def close(value):
        state.events.append("close")
        state.pipelines.append(value)
        native_close(value)

    config = make_config()
    monkeypatch.setattr(config_loader, "load_config", lambda *_args: config)
    monkeypatch.setattr(entrypoint, "validate_config", lambda cfg: cfg.rl)
    monkeypatch.setattr(entrypoint, "sft_policy_source", lambda cfg: (str(model_source), False))
    monkeypatch.setattr(packers, "build_sft_packer", packer)
    monkeypatch.setattr(build, "build_backbone", lambda cfg: Policy())
    monkeypatch.setattr(build, "build_optimizer", lambda *_args: None)
    monkeypatch.setattr(adapter_identity, "adapter_processor_identity", identity)
    monkeypatch.setattr(determinism, "enable_determinism", lambda **_kwargs: None)
    monkeypatch.setattr(dist, "init_local_process_group", lambda: state.events.append("init"))
    monkeypatch.setattr(dist, "destroy_process_group_if_initialized", lambda: state.events.append("destroy"))
    monkeypatch.setattr(misc, "set_random_seed", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "CPU fixture")
    monkeypatch.setattr(score.importlib.metadata, "version", lambda name: "fixture")
    monkeypatch.setattr(SFTPreprocessingPipeline, "close", close)

    def run(name, cache_bytes=None):
        output = tmp_path / name
        args = SimpleNamespace(
            data=str(source),
            output=str(output),
            max_rows=3,
            seed=2026,
            config="fixture",
            override=[],
            arm="fixture",
            checkpoint=None,
        )
        if cache_bytes is not None:
            args.preprocessing_cache_bytes = cache_bytes
        score.run(args)
        return output, json.loads((output / "summary.json").read_text())

    state.run, state.root = run, tmp_path
    return state


def test_opt_in_native_cache_preserves_all_rows_and_identities_and_hits(execution):
    uncached, raw = execution.run("uncached")
    cached, actual = execution.run("cached", 100_000)
    for name in ("selected_rows.jsonl", "selection.json", "scores.jsonl", "model_identity.json"):
        assert (uncached / name).read_bytes() == (cached / name).read_bytes()
    assert "preprocessing_cache" not in raw
    assert {k: v for k, v in raw.items() if k != "elapsed_seconds"} == {
        k: v for k, v in actual.items() if k not in {"elapsed_seconds", "preprocessing_cache"}
    }
    report = actual["preprocessing_cache"]
    assert report["policy"] == {"cache_bytes": 100_000, "prefetch": False, "workers": 1, "max_pending": 1}
    assert report["stats"]["samples"]["hits"] == report["stats"]["videos"]["hits"] == 1
    assert report["stats"]["timing"]["processing_calls"] == 3
    assert execution.packers[0].calls == 3 and execution.packers[1].calls == 2
    assert execution.packers[0].processor.video_processor.calls == 3
    assert execution.packers[1].processor.video_processor.calls == 1
    assert all(p.collations == 3 for p in execution.packers)
    assert execution.events == ["init", "destroy", "init", "close", "destroy"]
    assert execution.pipelines[0]._closed and not execution.pipelines[0]._executors
    assert all(not net._forward_hooks for net in execution.nets)
    assert execution.pipelines[0].sft_collate_fn == execution.packers[1].sft_collate_fn


@pytest.mark.parametrize("failure", ["forward", "changed_forward_input", "media", "source"])
def test_cache_failure_restores_scorer_hooks_and_closes_before_destroy(execution, failure):
    execution.failure = failure
    with pytest.raises(RuntimeError, match="fixture forward|changed the observed|video changed|data source changed"):
        execution.run("failed", 100_000)
    assert execution.events == ["init", "close", "destroy"]
    pipeline = execution.pipelines[0]
    assert pipeline._closed
    assert pipeline.sft_collate_fn == execution.packers[0].sft_collate_fn
    assert not execution.nets[0]._forward_hooks
    assert not (execution.root / "failed" / "summary.json").exists()


@pytest.mark.parametrize("value", [-1, True, 1.5, "100", None])
def test_invalid_cache_bound_fails_before_any_source_or_model(value):
    with pytest.raises(ValueError, match="nonnegative integer"):
        score.run(SimpleNamespace(preprocessing_cache_bytes=value))


def test_explicit_zero_preserves_default_summary_schema(execution):
    _, report = execution.run("zero", 0)
    assert "preprocessing_cache" not in report and not execution.pipelines
    assert execution.events == ["init", "destroy"]


def test_cli_cache_option_defaults_off_and_accepts_explicit_bytes(tmp_path, monkeypatch):
    import sys

    observed = []
    monkeypatch.setattr(score, "run", lambda args: observed.append(args.preprocessing_cache_bytes) or {})
    command = ["score", "--config", "fixture", "--data", "unused", "--output", str(tmp_path / "out")]
    monkeypatch.setattr(sys, "argv", command)
    score.main()
    monkeypatch.setattr(sys, "argv", command + ["--preprocessing-cache-bytes", "1234"])
    score.main()
    assert observed == [0, 1234]
