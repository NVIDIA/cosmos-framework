# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""Actual-media CPU qualification; no model, optimizer or CUDA construction.

Ordinary CI without fixtures skips. The allocated qualification stage must bind
all four COSMOS_PARALLEL_* variables below and require exactly one pass/no skips.
It must launch with eight distinct physical/logical CPUs and CUDA hidden.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import pytest

from tools.native_qlora_qualification import layer_sensitivity, score, worker_memory

_ENV = (
    "COSMOS_PARALLEL_PROCESSOR_PATH",
    "COSMOS_PARALLEL_TRAINING_DATA",
    "COSMOS_PARALLEL_TRAINING_MANIFEST",
    "COSMOS_PARALLEL_TRAINING_MANIFEST_SHA256",
)
_CACHE_BYTES = 1024**3


def _worker_state():
    import torch

    from cosmos_framework.posttrain.data.packer import sft_preprocess

    threads = {}
    for task in Path("/proc/self/task").iterdir():
        try:
            threads[task.name] = sorted(os.sched_getaffinity(int(task.name)))
        except ProcessLookupError:
            pass
    assert str(os.getpid()) in threads
    return {
        "pid": os.getpid(),
        "start_ticks": int(Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19]),
        "thread_affinities": threads,
        "rng_sha256": worker_memory.rng_hash(),
        "cuda_initialized": torch.cuda.is_initialized(),
        "cache": sft_preprocess._WORKER_CACHE.stats(),
        "processing_calls": sft_preprocess._WORKER_PROCESSING_CALLS,
    }


def _case(processor, rl, rows, media_hashes, workers):
    import torch

    from cosmos_framework.posttrain.data.packer.sft import build_sft_packer
    from cosmos_framework.posttrain.data.packer.sft_preprocess import build_sft_preprocessing

    started = time.monotonic()
    packer = build_sft_packer(
        processor,
        max_length=rl.max_length,
        processor_kwargs=rl.processor_kwargs,
        mask_think_opener=rl.mask_think_opener,
        trust_remote_code=False,
    )
    identity = packer.validation_semantics_identity()
    pipeline = build_sft_preprocessing(
        packer,
        name_or_path=processor,
        cache_bytes=_CACHE_BYTES,
        prefetch=True,
        media_hashes=media_hashes,
        max_pending=8,
        workers=workers,
    )
    parent_rng, parent_cpus = worker_memory.rng_hash(), sorted(os.sched_getaffinity(0))
    processes = []
    try:
        # Production submits each window without awaiting individual workers.
        # Start all workers before awaiting the diagnostic state RPCs as well.
        initial_futures = [pool.submit(_worker_state) for pool in pipeline._executors]
        initial = [future.result(timeout=180) for future in initial_futures]
        setup_seconds = time.monotonic() - started
        assert len({item["pid"] for item in initial}) == workers
        assert all(item["processing_calls"] == 0 for item in initial)
        records = []
        replay_start = time.monotonic()
        for offset in range(0, len(rows), 8):
            window = rows[offset : offset + 8]
            targets = [layer_sensitivity.target_for(row) for row in window]
            pipeline.prepare_window(targets)
            for row, target in zip(window, targets, strict=True):
                sample = pipeline.sft_process_sample(target)
                batch = pipeline.sft_collate_fn([sample], pipeline.sft_compute_max_len([sample]), -100)
                assert batch["pixel_values_videos"].numel() > 0
                assert all(value.device.type == "cpu" for value in batch.values())
                valid = (
                    (batch["labels"][:, 1:] != -100)
                    & batch["attention_mask"][:, 1:].bool()
                    & batch["attention_mask"][:, :-1].bool()
                )
                assert valid.any()
                records.append(
                    {
                        "id_sha256": hashlib.sha256(str(row["id"]).encode()).hexdigest(),
                        "sample_sha256": worker_memory.tree_hash(sample),
                        "batch_sha256": worker_memory.tree_hash(batch),
                        "input_ids_sha256": worker_memory.tree_hash(batch["input_ids"]),
                        "labels_sha256": worker_memory.tree_hash(batch["labels"]),
                        "supervised_mask_sha256": worker_memory.tree_hash(valid),
                        "sequence_length": len(sample.input_ids),
                        "supervised_tokens": int(valid.sum()),
                    }
                )
                del sample, batch, valid
            pipeline.finish_window()
            assert worker_memory.rng_hash() == parent_rng
            assert sorted(os.sched_getaffinity(0)) == parent_cpus
        replay_seconds = time.monotonic() - replay_start
        final = [pool.submit(_worker_state).result(timeout=30) for pool in pipeline._executors]
        expected_cpus = [parent_cpus] if workers == 1 else [list(value) for value in pipeline.worker_cpu_affinities]
        assert set.union(*(set(value) for value in expected_cpus)) == set(parent_cpus)
        if workers == 2:
            assert set(expected_cpus[0]).isdisjoint(expected_cpus[1])
        for before, after, cpus in zip(initial, final, expected_cpus, strict=True):
            assert (before["pid"], before["start_ticks"]) == (after["pid"], after["start_ticks"])
            assert before["rng_sha256"] == after["rng_sha256"]
            assert not before["cuda_initialized"] and not after["cuda_initialized"]
            assert after["processing_calls"] > 0
            assert all(value == cpus for value in after["thread_affinities"].values())
            assert all(value["bytes"] <= value["limit"] for value in after["cache"].values())
        assert sum(item["processing_calls"] for item in final) == len(rows)
        assert sum(value["limit"] for item in final for value in item["cache"].values()) == _CACHE_BYTES
        assert sum(value["bytes"] for item in final for value in item["cache"].values()) <= _CACHE_BYTES
        stats = pipeline.preprocessing_stats
        for kind in ("samples", "videos"):
            assert stats[kind] == {key: sum(item["cache"][kind][key] for item in final) for key in stats[kind]}
        assert stats["timing"]["processing_calls"] == stats["timing"]["wait_calls"] == len(rows)
        assert pipeline.validation_semantics_identity() == identity
        assert not torch.cuda.is_initialized()
    finally:
        processes = [p for pool in pipeline._executors for p in (pool._processes or {}).values()]
        close_start = time.monotonic()
        pipeline.close()
        close_seconds = time.monotonic() - close_start
        assert all(not process.is_alive() for process in processes)
    assert worker_memory.rng_hash() == parent_rng and sorted(os.sched_getaffinity(0)) == parent_cpus
    return {
        "workers": workers,
        "processor_identity": identity,
        "rows": records,
        "initial_worker_states": initial,
        "final_worker_states": final,
        "cache_and_processing_stats": stats,
        "setup_seconds": setup_seconds,
        "replay_seconds": replay_seconds,
        "close_seconds": close_seconds,
        "case_seconds": time.monotonic() - started,
        "all_children_closed": True,
        "parent_rng_unchanged": True,
        "parent_rng_scope": "After parent processor/pipeline construction through worker startup, replay and close.",
    }


def test_actual_training_media_one_two_and_cold_two_workers_match(record_property):
    values = [os.environ.get(name, "") for name in _ENV]
    if not any(values):
        pytest.skip("requires explicit local processor and sealed training64 fixtures")
    if not all(values):
        raise ValueError("all four COSMOS_PARALLEL_* fixture bindings are required")
    processor, data_name, manifest_name, manifest_sha = values
    assert Path(processor).is_dir()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert os.environ.get("HF_HUB_OFFLINE") == os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    topology = worker_memory.cpu_topology()
    worker_memory.validate_cpu_topology(topology, 8)
    assert topology["logical_cpu_count"] == 8
    data, manifest_path = Path(data_name), Path(manifest_name)
    raw_rows, manifest, _ = layer_sensitivity.training_rows(data, manifest_path, manifest_sha)

    import torch

    from cosmos_framework.posttrain.configs.sft_qwen3_vl_8b_qlora import make_config
    from cosmos_framework.posttrain.data.datasets.sft import normalize_row

    assert not torch.cuda.is_initialized()
    rl = make_config().rl
    assert rl.max_length == 4096 and rl.processor_kwargs["videos_kwargs"]["num_frames"] == 8
    rows, media_hashes = [], {}
    for (line, raw), expected in zip(raw_rows, manifest["rows"], strict=True):
        row = normalize_row(raw, data.parent, label=f"training64:{line}")
        _, _, _, video = score.validate_wts(row)
        digest = score.file_digest(Path(video))
        assert digest == expected["media_sha256"]
        media_hashes[video] = digest
        rows.append(row)
    assert len(rows) == 64
    cases = [_case(processor, rl, rows, media_hashes, workers) for workers in (1, 2, 2)]
    assert cases[0]["rows"] == cases[1]["rows"] == cases[2]["rows"]
    assert cases[0]["processor_identity"] == cases[1]["processor_identity"] == cases[2]["processor_identity"]
    assert set((v["pid"], v["start_ticks"]) for v in cases[1]["final_worker_states"]).isdisjoint(
        (v["pid"], v["start_ticks"]) for v in cases[2]["final_worker_states"]
    )
    layer_sensitivity.training_rows(data, manifest_path, manifest_sha)
    assert all(score.file_digest(Path(path)) == digest for path, digest in media_hashes.items())
    assert not torch.cuda.is_initialized()
    record_property(
        "parallel_preprocessing_actual_media",
        json.dumps(
            {
                "protocol": "cosmos.parallel-preprocessing-actual-media.v1",
                "manifest_sha256": manifest_sha,
                "data_sha256": manifest["data_sha256"],
                "recipe_max_length": rl.max_length,
                "recipe_processor_kwargs": rl.processor_kwargs,
                "recipe_mask_think_opener": rl.mask_think_opener,
                "topology": topology,
                "rows": 64,
                "window_rows": 8,
                "cache_bytes_total": _CACHE_BYTES,
                "case_worker_counts": [1, 2, 2],
                "cases": cases,
                "exact_inputs_masks_ids": True,
                "cold_reconstruction_exact": True,
                "timing_scope": "Sequential setup/replay/close observations; replay includes exact input hashing and collation, no speed comparison claim.",
                "model_constructed": False,
                "cuda_initialized": False,
                "training_or_quality_qualified": False,
            },
            sort_keys=True,
            allow_nan=False,
        ),
    )
