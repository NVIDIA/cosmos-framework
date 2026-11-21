# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
from __future__ import annotations

import json
import subprocess
import sys
from collections import UserDict
from pathlib import Path

import numpy as np
import pytest
import torch

from tools.native_qlora_qualification import worker_memory as memory

_PATH = Path(memory.__file__)


def test_policy_changes_only_blas_and_cpu_safety():
    before = {"OMP_NUM_THREADS": "8", "OPENBLAS_NUM_THREADS": "32", "OTHER": "keep"}
    inherited = memory.policy_environment(before, "inherited")
    changed = memory.policy_environment(before, "blas-one")
    assert inherited["OPENBLAS_NUM_THREADS"] == "32"
    assert changed["OPENBLAS_NUM_THREADS"] == changed["MKL_NUM_THREADS"] == "1"
    assert changed["OMP_NUM_THREADS"] == "8" and changed["OTHER"] == "keep"
    assert before == {"OMP_NUM_THREADS": "8", "OPENBLAS_NUM_THREADS": "32", "OTHER": "keep"}
    assert inherited["CUDA_VISIBLE_DEVICES"] == changed["CUDA_VISIBLE_DEVICES"] == ""
    with pytest.raises(ValueError):
        memory.policy_environment({}, "other")


def test_physical_cpu_preflight_counts_smt_once_and_refuses_unknown(tmp_path, monkeypatch):
    monkeypatch.setattr(memory.os, "sched_getaffinity", lambda pid: {0, 1, 2, 3})
    for cpu in range(4):
        root = tmp_path / f"cpu{cpu}" / "topology"
        root.mkdir(parents=True)
        (root / "physical_package_id").write_text("0")
        (root / "core_id").write_text(str(cpu // 2))
    topology = memory.cpu_topology(sysfs_root=tmp_path)
    assert topology["logical_cpu_count"] == 4 and topology["physical_cpu_count"] == 2
    memory.validate_cpu_topology(topology, 2)
    with pytest.raises(ValueError, match="exactly 8 physical"):
        memory.validate_cpu_topology(topology, 8)
    (tmp_path / "cpu3/topology/core_id").unlink()
    incomplete = memory.cpu_topology(sysfs_root=tmp_path)
    assert incomplete["physical_cpu_count"] is None and incomplete["unavailable_cpus"] == [3]
    with pytest.raises(ValueError, match="observed None"):
        memory.validate_cpu_topology(incomplete, 2)


def test_wrong_affinity_fails_before_heavy_imports_or_child_launch(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            str(_PATH),
            "run",
            "--config",
            "unused.py",
            "--model-path",
            "unused",
            "--data-root",
            "unused",
            "--build-receipt",
            "unused",
            "--output",
            str(tmp_path / "run"),
            "--thread-policy",
            "inherited",
            "--expected-physical-cpus",
            "999999",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode != 0 and "requires exactly 999999 physical cores" in result.stderr
    assert "ModuleNotFoundError" not in result.stderr
    assert (tmp_path / "run/cpu-topology.json").is_file()
    assert not (tmp_path / "run/child.log").exists()


def test_exact_fingerprints_preserve_types_shapes_and_bits():
    value = {"pixels": torch.arange(12).reshape(3, 4), "mask": np.array([1, 0], dtype=np.int64)}
    assert memory.tree_hash(value) == memory.tree_hash(UserDict(value))
    assert memory.tree_hash(torch.tensor(1)) != memory.tree_hash(torch.tensor([1]))
    assert memory.tree_hash(torch.tensor([0.0])) != memory.tree_hash(torch.tensor([-0.0]))
    assert memory.tree_hash([1]) != memory.tree_hash((1,))
    assert memory.tree_hash(value) != memory.tree_hash({**value, "pixels": value["pixels"].float()})
    with pytest.raises(ValueError, match="detached CPU"):
        memory.tree_hash(torch.ones(2, requires_grad=True))
    with pytest.raises(TypeError, match="object arrays"):
        memory.tree_hash(np.array([object()], dtype=object))
    before = memory.rng_hash()
    memory.tree_hash(value)
    assert memory.rng_hash() == before


def _report(path, policy):
    path.mkdir()
    (path / "rows.jsonl").write_text('{"sample":"same bytes"}\n')
    result = {
        "protocol": memory.PROTOCOL,
        "complete": True,
        "build_receipt_sha256": "a" * 64,
        "source_fingerprint": "source",
        "processor_identity_sha256": "processor",
        "row_output_sha256": memory.file_hash(path / "rows.jsonl"),
        "row_order_sha256": "rows",
        "data_state_sha256": "cursor",
        "rng_before_sha256": "rng",
        "rng_after_sha256": "rng",
        "worker_rng_unchanged": True,
        "rows": 1000,
        "seed": 42,
        "cache_bytes": 2**30,
        "window_rows": 8,
        "consumer_delay_seconds": 0,
        "config": "recipe",
        "override": [],
        "versions": {"torch": "fixture"},
        "end_diagnostic": {"cache_and_rng_unchanged": True},
        "thread_policy": policy,
        "repeat": 0,
        "peak_worker_pss_bytes": 100,
        "elapsed_seconds": 2,
    }
    memory.write_json(path / "result.json", result)
    return result


def test_compare_requires_exact_replay_bytes_and_control_state(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    _report(a, "inherited")
    changed = _report(b, "blas-one")
    assert memory.compare_reports([a, b])["semantic_and_tensor_bytes_exact"]
    changed["data_state_sha256"] = "different"
    (b / "result.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="data_state"):
        memory.compare_reports([a, b])


def test_compare_rejects_changed_row_receipt_and_incomplete_run(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    _report(a, "inherited")
    changed = _report(b, "blas-one")
    (b / "rows.jsonl").write_text("changed\n")
    with pytest.raises(ValueError, match="row records"):
        memory.compare_reports([a, b])
    changed["complete"] = False
    (b / "result.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="incomplete"):
        memory.compare_reports([a, b])


def test_help_has_no_heavy_import_requirement():
    result = subprocess.run([sys.executable, "-S", str(_PATH), "--help"], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0
    assert "compare" in result.stdout


def test_atomic_receipts_refuse_overwrite(tmp_path):
    path = tmp_path / "receipt.json"
    memory.write_json(path, {"valid": True})
    with pytest.raises(FileExistsError):
        memory.write_json(path, {"valid": False})
    assert json.loads(path.read_text()) == {"valid": True}


def test_snapshot_runs_in_actual_native_spawned_worker(tmp_path):
    from cosmos_framework.posttrain.data.packer.sft import build_sft_packer
    from cosmos_framework.posttrain.data.packer.sft_preprocess import build_sft_preprocessing
    from cosmos_framework.posttrain.schema import SFTTarget
    from cosmos_framework.posttrain.tests.helpers.sft_packer import local_tokenizer

    local_tokenizer().save_pretrained(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen2"}))
    packer = build_sft_packer(str(tmp_path), max_length=128)
    pipeline = build_sft_preprocessing(
        packer, name_or_path=str(tmp_path), cache_bytes=100_000, prefetch=True, media_hashes={}, max_pending=1
    )
    target = SFTTarget([{"role": "user", "content": "question"}, {"role": "assistant", "content": "A"}])
    try:
        before = pipeline._executor.submit(memory.worker_snapshot, "before").result(timeout=30)
        pipeline.prepare_window([target])
        actual = pipeline.sft_process_sample(target)
        pipeline.finish_window()
        after = pipeline._executor.submit(memory.worker_snapshot, "after").result(timeout=30)
        assert memory.tree_hash(actual) == memory.tree_hash(packer.sft_process_sample(target))
        assert before["rng_sha256"] == after["rng_sha256"]
        assert after["cuda_initialized"] is False and after["torch_threads"] == 1
        assert after["cache"]["samples"]["misses"] == 1
        assert after["memory"]["Pss"] > 0
    finally:
        pipeline.close()
