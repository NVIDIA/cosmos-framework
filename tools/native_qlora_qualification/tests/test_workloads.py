# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
from __future__ import annotations

import copy
import importlib.util
import io
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

_DIRECTORY = Path(__file__).parents[1]
sys.path.insert(0, str(_DIRECTORY))
_SPEC = importlib.util.spec_from_file_location("workloads", _DIRECTORY / "workloads.py")
assert _SPEC and _SPEC.loader
workloads = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(workloads)


def original():
    return {
        "id": "fixture",
        "messages": [
            {"role": "system", "content": "Answer with a letter."},
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": "/fixture/video.mp4"},
                    {"type": "text", "text": "Original question A: first B: second"},
                ],
            },
            {"role": "assistant", "content": "B"},
        ],
    }


def test_long_target_preserves_media_question_and_original_gold():
    row = original()
    snapshot = copy.deepcopy(row)
    changed = workloads.long_row(row, 3, 2)
    assert row == snapshot
    assert changed["messages"][1]["content"][:2] == row["messages"][1]["content"]
    answer = changed["messages"][-1]["content"]
    assert answer.startswith("B\nRecord 2001:") and "Record 2003:" in answer
    assert "synthetic text, not a claim" in changed["messages"][1]["content"][-1]["text"]
    assert changed == workloads.long_row(row, 3, 2)
    assert changed != workloads.long_row(row, 3, 3)


def test_video_total_pixel_budgets_and_microbatch_padding():
    small = workloads.processor_options(8, 65_536)["videos_kwargs"]
    large = workloads.processor_options(16, 262_144)["videos_kwargs"]
    assert small["size"] == {"shortest_edge": 32768, "longest_edge": 524288}
    assert large["size"] == {"shortest_edge": 65536, "longest_edge": 4194304}
    assert large["fps"] is None
    assert workloads.padding_summary([100, 105, 200, 150], 2) == {
        "microbatch": 2,
        "unpadded_tokens": 555,
        "padded_tokens": 610,
        "padding_tokens": 55,
    }
    for frames in (1, 3, True):
        with pytest.raises(ValueError):
            workloads.processor_options(frames, 65536)
    with pytest.raises(ValueError):
        workloads.padding_summary([1, 2, 3], 2)


def test_fits_actual_complete_targets_and_refuses_decoder_failure():
    def process(row):
        records = row["messages"][-1]["content"].count("Record ")
        length = 200 + records * 41
        if length > 4096:
            raise ValueError(f"SFT sequence length {length} exceeds max_length=4096; truncation is forbidden")
        return SimpleNamespace(input_ids=[1] * length, labels=[-100] * 200 + [2] * (length - 200))

    row, sample, count = workloads.fit_long_row(original(), 0, process)
    assert 3840 <= len(sample.input_ids) <= 4096 and count == 95
    assert row["messages"][-1]["content"].count("Record ") == count

    def fail(row):
        raise ValueError("decoder failed")

    with pytest.raises(ValueError, match="decoder failed"):
        workloads.fit_long_row(original(), 0, fail)
    with pytest.raises(ValueError, match="lower bound"):
        workloads.fit_long_row(original(), 0, lambda row: SimpleNamespace(input_ids=[1] * 100, labels=[2] * 100))


def test_selected_rows_are_fixed_and_duplicates_fail(tmp_path):
    path = tmp_path / "source.jsonl"
    rows = [{**original(), "id": str(i)} for i in range(8)]
    media = tmp_path / "fixture.mp4"
    media.write_bytes(b"read-only metadata fixture; no decoder called")
    for row in rows:
        row["messages"][1]["content"][0]["video"] = str(media)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    first = workloads.select_rows(path, 4, 42)
    assert first == workloads.select_rows(path, 4, 42)
    assert [int(row["id"]) for row in first] == sorted(int(row["id"]) for row in first)
    path.write_text(path.read_text() + json.dumps(rows[0]) + "\n")
    with pytest.raises(ValueError, match="distinct"):
        workloads.select_rows(path, 4, 42)


def census_fixture(tmp_path, count=11):
    from cosmos_framework.posttrain.config.sft import SFTDatasetConfig
    from cosmos_framework.posttrain.roles.env.sft import build_sft_env

    media = tmp_path / "video.mp4"
    media.write_bytes(b"CPU contract fixture; never decoded")
    rows = [{**original(), "id": str(index)} for index in range(count)]
    for index, row in enumerate(rows):
        row["messages"] = copy.deepcopy(row["messages"])
        row["messages"][1]["content"][0]["video"] = str(media)
        row["messages"][1]["content"][1]["text"] = str(index)
    source = tmp_path / "train.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = tmp_path / "train.manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "protocol": "cosmos.wts-group-split.v1",
                "role": "train",
                "group_count": 1,
                "data_sha256": workloads.file_hash(source),
                "rows": [
                    {
                        "id": row["id"],
                        "raw_row_sha256": workloads.json_hash(row),
                        "media_sha256": workloads.file_hash(media),
                        "group_sha256": "a" * 64,
                    }
                    for row in rows
                ],
            }
        )
    )
    dataset_config = SFTDatasetConfig(path=str(source))
    return source, manifest, media, dataset_config, build_sft_env(dataset_config, 42)


@dataclass
class Sample:
    input_ids: list[int]
    labels: list[int]


class CensusPipeline:
    """Tiny tensor fixture around the real native source/cursor; no processor or media decoding."""

    def __init__(self):
        self.windows, self.finished, self.closed = [], 0, False
        self.processor = SimpleNamespace(
            video_processor=SimpleNamespace(patch_size=16, temporal_patch_size=2, merge_size=2)
        )
        self.on_close = lambda: None
        self._executor = SimpleNamespace(submit=lambda fn: SimpleNamespace(result=lambda timeout: fn()))

    def prepare_window(self, targets):
        self.windows.append(len(targets))

    def finish_window(self):
        self.finished += 1

    def sft_process_sample(self, target):
        length = 5 + int(target.messages[1]["content"][1]["text"]) % 7
        return Sample([1] * length, [-100] * (length - 2) + [2, 3])

    def sft_compute_max_len(self, samples):
        return len(samples[0].input_ids)

    def sft_collate_fn(self, samples, length, ignore):
        import torch

        return {
            "input_ids": torch.tensor([samples[0].input_ids]),
            "labels": torch.tensor([samples[0].labels]),
            "attention_mask": torch.ones((1, length), dtype=torch.long),
            "video_grid_thw": torch.tensor([[1, 2, 2]]),
            "pixel_values_videos": torch.zeros((4, 1536)),
        }

    def validation_semantics_identity(self):
        return {"fixture": "tiny CPU tensors"}

    def close(self):
        self.closed = True
        self.on_close()


def test_census_exact_native_cursor_all_4611_rows_and_three_row_tail(tmp_path):
    source, manifest, _, _, data = census_fixture(tmp_path, 4611)
    membership = workloads.census_membership(source, manifest, data.dataset, 4611)
    pipeline, stream = CensusPipeline(), io.StringIO()
    result = workloads.census_rows(data, pipeline, membership["rows"], stream)
    rows = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert len(rows) == len({row["source_index"] for row in rows}) == 4611
    assert pipeline.windows == [8] * 576 + [3] and pipeline.finished == 577
    assert result["cursor_before"]["position"] == 0
    assert result["cursor_after"]["position"] == 4611 and result["cursor_after"]["epoch"] == 1
    assert result["rng_before_sha256"] == result["rng_after_sha256"]
    assert all(row["supervised_tokens"] == 2 for row in rows)
    assert rows[0]["video_grid_thw"] == [1, 2, 2]
    assert rows[0]["tensors"]["pixel_values_videos"]["shape"] == [4, 1536]


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "reordered", "raw", "media", "group", "role"])
def test_census_manifest_requires_exact_complete_membership(tmp_path, mutation):
    source, manifest, _, _, data = census_fixture(tmp_path)
    document = json.loads(manifest.read_text())
    if mutation == "missing":
        document["rows"].pop()
    elif mutation == "duplicate":
        document["rows"][1] = document["rows"][0]
    elif mutation == "reordered":
        document["rows"].reverse()
    elif mutation in ("raw", "media"):
        document["rows"][0][mutation + "_row_sha256" if mutation == "raw" else "media_sha256"] = "b" * 64
    elif mutation == "group":
        document["group_count"] = 2
    else:
        document["role"] = "development"
    manifest.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        workloads.census_membership(source, manifest, data.dataset, 11)


@pytest.mark.parametrize("mutation", ["duplicate", "reordered", "foreign", "target", "rng"])
def test_census_refuses_cursor_target_and_rng_mutations(tmp_path, mutation):
    import random

    source, manifest, _, _, data = census_fixture(tmp_path)
    membership = workloads.census_membership(source, manifest, data.dataset, 11)
    get_samples, pipeline = data.get_samples, CensusPipeline()

    def changed_samples(count):
        rows = get_samples(count)
        if mutation == "duplicate":
            rows[1] = rows[0]
        elif mutation == "reordered":
            rows.reverse()
        elif mutation == "foreign":
            rows[0].id = "foreign:0"
        elif mutation == "target":
            rows[0].transitions[0].targets.messages[-1]["content"] = "changed"
        else:
            random.random()
        return rows

    data.get_samples = changed_samples
    with pytest.raises(ValueError):
        workloads.census_rows(data, pipeline, membership["rows"], io.StringIO())


def configure_census(tmp_path, monkeypatch):
    from cosmos_framework.posttrain.data.packer import sft, sft_preprocess
    from cosmos_framework.posttrain.utils import adapter_source

    source, manifest, media, _, _ = census_fixture(tmp_path)
    model = tmp_path / "base"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_vl",
                "text_config": {
                    "num_hidden_layers": 36,
                    "hidden_size": 4096,
                    "intermediate_size": 12288,
                },
            }
        )
    )
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
    ):
        (model / name).write_text("{}")
    (model / "model.safetensors").write_bytes(b"source-verification fixture; never tensor-loaded")
    (model / adapter_source.SOURCE_MANIFEST).write_text(json.dumps(adapter_source._identity(model)))
    monkeypatch.setenv("SFT_MODEL_PATH", str(model))
    monkeypatch.setenv("SFT_DATA_ROOT", str(tmp_path))
    receipt = tmp_path / "build.json"
    receipt.write_text("{}\n")
    pipeline = CensusPipeline()
    monkeypatch.setattr(sft, "build_sft_packer", lambda *args, **kwargs: pipeline)
    monkeypatch.setattr(sft_preprocess, "build_sft_preprocessing", lambda *args, **kwargs: pipeline)
    args = SimpleNamespace(
        config="cosmos_framework/posttrain/configs/sft_qwen3_vl_8b_qlora.py",
        source_manifest=str(manifest),
        build_receipt=str(receipt),
        output=str(tmp_path / "output"),
        override=[
            "rl.preprocessing_cache_bytes=3221225472",
            "rl.preprocessing_prefetch=true",
            "rl.preprocessing_workers=1",
        ],
        expected_rows=11,
    )
    return args, pipeline, source, manifest, media


def test_census_publishes_only_after_closed_source_and_rng_verification(tmp_path, monkeypatch):
    args, pipeline, _, _, _ = configure_census(tmp_path, monkeypatch)
    workloads.census(args)
    result = json.loads((Path(args.output) / "manifest.json").read_text())
    assert pipeline.closed and result["complete"] and result["source_before_after_unchanged"]
    assert result["rows"] == 11 and result["final_window_rows"] == 3
    assert result["worker_rng_before_sha256"] == result["worker_rng_after_sha256"]
    assert result["training_performed"] is result["model_constructed"] is result["long_gpu_training_qualified"] is False


def test_census_real_native_recipe_resolves_and_hashes_outside_checkout(tmp_path, monkeypatch):
    import importlib

    from cosmos_framework.utils.config_helper import get_config_module

    args, _, _, _, _ = configure_census(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)
    assert not Path(args.config).exists()
    workloads.census(args)
    result = json.loads((Path(args.output) / "manifest.json").read_text())
    module = importlib.import_module(get_config_module(args.config))
    assert result["source_files_sha256"]["config"] == workloads.file_hash(Path(module.__file__))
    assert result["configuration_overrides"] == args.override
    assert result["processor_source"] == str(tmp_path / "base")
    assert result["verified_adapter_source_identity_sha256"]


def test_census_worker_rng_change_cannot_publish_complete(tmp_path, monkeypatch):
    args, pipeline, _, _, _ = configure_census(tmp_path, monkeypatch)
    hashes = iter(["a" * 64, "b" * 64])
    pipeline._executor = SimpleNamespace(submit=lambda fn: SimpleNamespace(result=lambda timeout: next(hashes)))
    with pytest.raises(ValueError, match="worker RNG"):
        workloads.census(args)
    assert pipeline.closed and not (Path(args.output) / "manifest.json").exists()


@pytest.mark.parametrize("field", ["patch_size", "temporal_patch_size", "merge_size"])
def test_census_refuses_a_different_processor_geometry(tmp_path, monkeypatch, field):
    args, pipeline, _, _, _ = configure_census(tmp_path, monkeypatch)
    setattr(pipeline.processor.video_processor, field, 1)
    with pytest.raises(ValueError, match="fixed Qwen3-VL"):
        workloads.census(args)
    assert not pipeline.windows and not (Path(args.output) / "manifest.json").exists()


@pytest.mark.parametrize("change", ["source", "manifest", "media", "processor", "close"])
def test_census_changed_source_or_drain_failure_cannot_publish_complete(tmp_path, monkeypatch, change):
    args, pipeline, source, manifest, media = configure_census(tmp_path, monkeypatch)

    def mutate():
        if change == "close":
            raise RuntimeError("worker drain failed")
        path = {
            "source": source,
            "manifest": manifest,
            "media": media,
            "processor": tmp_path / "base" / "tokenizer.json",
        }[change]
        path.write_bytes(path.read_bytes() + b" ")

    pipeline.on_close = mutate
    with pytest.raises((ValueError, RuntimeError)):
        workloads.census(args)
    assert not (Path(args.output) / "manifest.json").exists()


def test_legacy_cli_and_explicit_census_count_dispatch_are_separate(monkeypatch):
    calls = []
    monkeypatch.setattr(workloads, "prepare", lambda args: calls.append(("prepare", vars(args))))
    monkeypatch.setattr(workloads, "census", lambda args: calls.append(("census", vars(args))))
    workloads.main(
        [
            "--source-train",
            "train",
            "--source-validation",
            "val",
            "--model-path",
            "model",
            "--build-receipt",
            "receipt",
            "--output",
            "prepared",
        ]
    )
    assert calls[0] == (
        "prepare",
        {
            "source_train": "train",
            "source_validation": "val",
            "model_path": "model",
            "build_receipt": "receipt",
            "output": "prepared",
            "rows": 32,
            "validation_rows": 8,
            "seed": 42,
        },
    )
    argv = [
        "census",
        "--config",
        "recipe",
        "--source-manifest",
        "manifest",
        "--build-receipt",
        "receipt",
        "--output",
        "census",
    ]
    with pytest.raises(SystemExit):
        workloads.main(argv)
    workloads.main([*argv, "--expected-rows", "4611", "--override", "rl.preprocessing_cache_bytes=3221225472"])
    assert calls[1][0] == "census" and calls[1][1]["expected_rows"] == 4611
