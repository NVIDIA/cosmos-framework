# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""Opt-in CPU periodic reclamation; no training or production-policy claims."""

import ast
import copy
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.native_qlora_qualification import worker_memory as memory

_PATH = Path(memory.__file__)


def test_maintenance_refuses_pending_or_active_window():
    for pending, active, executor in (([1], False, object()), ([], True, object()), ([], False, None)):
        with pytest.raises(RuntimeError, match="drained native window"):
            memory.drained_maintenance(
                SimpleNamespace(_pending=pending, _window_active=active, _executor=executor), 128, True
            )


def test_private_diagnostic_in_actual_native_worker(tmp_path):
    from cosmos_framework.posttrain.data.packer.sft import build_sft_packer
    from cosmos_framework.posttrain.data.packer.sft_preprocess import build_sft_preprocessing
    from cosmos_framework.posttrain.schema import SFTTarget
    from cosmos_framework.posttrain.tests.helpers.sft_packer import local_tokenizer

    local_tokenizer().save_pretrained(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen2"}))
    packer = build_sft_packer(str(tmp_path), max_length=128)
    pipeline = build_sft_preprocessing(
        packer, name_or_path=str(tmp_path), cache_bytes=100000, prefetch=True, media_hashes={}, max_pending=1
    )
    target = SFTTarget([{"role": "user", "content": "question"}, {"role": "assistant", "content": "A"}])
    try:
        installed = pipeline._executor.submit(memory.worker_install_processing_timer).result(timeout=30)
        with pytest.raises(RuntimeError, match="already installed"):
            pipeline._executor.submit(memory.worker_install_processing_timer).result(timeout=30)
        before = memory.rng_hash()
        pipeline.prepare_window([target])
        with pytest.raises(RuntimeError, match="drained"):
            memory.drained_maintenance(pipeline, 1, True)
        first = pipeline.sft_process_sample(target)
        pipeline.finish_window()
        sham = memory.drained_maintenance(pipeline, 1, False)
        trim = memory.drained_maintenance(pipeline, 1, True)
        assert sham["trim_seconds"] == 0 and sham["malloc_trim_return"] is None
        assert trim["trim_seconds"] >= 0 and trim["malloc_trim_return"] in (0, 1)
        assert sham["gc_called"] is trim["gc_called"] is False
        assert sham["cache_sha256"] == trim["cache_sha256"]
        assert sham["cache_and_rng_unchanged"] and trim["cache_and_rng_unchanged"]
        assert trim["before"]["rng_sha256"] == trim["after"]["rng_sha256"] == installed["rng_sha256"]
        pipeline.prepare_window([target])
        second = pipeline.sft_process_sample(target)
        pipeline.finish_window()
        assert (
            memory.tree_hash(first) == memory.tree_hash(second) == memory.tree_hash(packer.sft_process_sample(target))
        )
        counts = pipeline._executor.submit(memory.worker_processing_metrics).result(timeout=30)
        assert counts["calls"] == 2 and counts["seconds"] >= 0
        assert memory.rng_hash() == before
    finally:
        pipeline.close()


def report(path, interval):
    path.mkdir()
    (path / "rows.jsonl").write_text('{"all_tensors":"same bytes"}\n')
    records = [
        {
            "rows": i,
            "trim": bool(interval),
            "cache_and_rng_unchanged": True,
            "gc_called": False,
            "cache_sha256": "same-cache",
        }
        for i in range(128, 1001, 128)
    ]
    (path / "maintenance.jsonl").write_text("".join(json.dumps(x) + "\n" for x in records))
    r = {
        "protocol": memory.PERIODIC_PROTOCOL,
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
        "thread_policy": "inherited",
        "thread_environment": {"OMP_NUM_THREADS": "8"},
        "parent_affinity": list(range(8)),
        "cpu_topology": {"physical_cpu_count": 8},
        "repeat": 0,
        "end_diagnostic": {
            "cache_and_rng_unchanged": True,
            "cache_sha256": "same-cache",
            "before": {"cache": {"samples": {"hits": 0}}},
        },
        "peak_worker_pss_bytes": 100,
        "elapsed_seconds": 2,
        "trim_every_rows": interval,
        "maintenance_sha256": memory.file_hash(path / "maintenance.jsonl"),
        "processing_timing": {"calls": 1000, "seconds": 1},
        "periodic_maintenance": {"calls": 7, "trim_seconds": 0.1},
        "replay_seconds_before_end_diagnostic": 1.5,
    }
    memory.write_json(path / "result.json", r)
    memory.write_json(path / "exit.json", {"exit_code": 0})
    return r


def test_compare_exact_arms_and_affinity(tmp_path):
    a, b = tmp_path / "disabled", tmp_path / "periodic"
    report(a, 0)
    changed = report(b, 128)
    result = memory.compare_reports([a, b])
    assert result["passed"] and result["semantic_and_tensor_bytes_exact"]
    for field, value in [
        ("parent_affinity", [99]),
        ("row_order_sha256", "different"),
        ("data_state_sha256", "other"),
        ("rng_after_sha256", "changed"),
    ]:
        bad = copy.deepcopy(changed)
        bad[field] = value
        (b / "result.json").write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            memory.compare_reports([a, b])
    (b / "result.json").write_text(json.dumps(changed))
    records = [json.loads(x) for x in (b / "maintenance.jsonl").read_text().splitlines()]
    records[-1]["rows"] = 900
    (b / "maintenance.jsonl").write_text("".join(json.dumps(x) + "\n" for x in records))
    changed["maintenance_sha256"] = memory.file_hash(b / "maintenance.jsonl")
    (b / "result.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="exact drained128-row"):
        memory.compare_reports([a, b])


def test_compare_refuses_sham_gc_and_changed_tensor_receipts(tmp_path):
    a, b = tmp_path / "disabled", tmp_path / "periodic"
    report(a, 0)
    changed = report(b, 128)
    records = [json.loads(x) for x in (b / "maintenance.jsonl").read_text().splitlines()]
    records[0]["gc_called"] = True
    (b / "maintenance.jsonl").write_text("".join(json.dumps(x) + "\n" for x in records))
    changed["maintenance_sha256"] = memory.file_hash(b / "maintenance.jsonl")
    (b / "result.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="intervention"):
        memory.compare_reports([a, b])
    records[0]["gc_called"] = False
    (b / "maintenance.jsonl").write_text("".join(json.dumps(x) + "\n" for x in records))
    changed["maintenance_sha256"] = memory.file_hash(b / "maintenance.jsonl")
    (b / "result.json").write_text(json.dumps(changed))
    (b / "rows.jsonl").write_text("altered")
    with pytest.raises(ValueError, match="row records"):
        memory.compare_reports([a, b])


@pytest.mark.parametrize("exit_code", [None, 1, False])
def test_periodic_comparison_requires_clean_exit_after_result(tmp_path, exit_code):
    a, b = tmp_path / "sham", tmp_path / "trim"
    report(a, 0)
    report(b, 128)
    if exit_code is None:
        (a / "exit.json").unlink()
    else:
        (a / "exit.json").write_text(json.dumps({"exit_code": exit_code}))
    with pytest.raises(ValueError, match="clean child exit"):
        memory.compare_reports([a, b])


def test_mixed_protocols_and_unmatched_periodic_arms_fail(tmp_path):
    a, b = tmp_path / "sham", tmp_path / "trim"
    report(a, 0)
    changed = report(b, 0)
    with pytest.raises(ValueError, match="exactly one"):
        memory.compare_reports([a, b])
    changed["protocol"] = memory.PROTOCOL
    (b / "result.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="protocols must match"):
        memory.compare_reports([a, b])


def _arguments(tmp_path, *, policy="inherited", rows=128):
    return [
        str(_PATH),
        "run",
        "--config",
        "fixture",
        "--model-path",
        "fixture",
        "--data-root",
        "fixture",
        "--build-receipt",
        "fixture",
        "--output",
        str(tmp_path),
        "--thread-policy",
        policy,
        "--rows",
        str(rows),
    ]


@pytest.mark.parametrize("interval", [None, 0, 128])
def test_omitted_flag_preserves_legacy_path_and_periodic_requires_opt_in(tmp_path, monkeypatch, interval):
    calls = []

    def legacy(args):
        calls.append(("legacy", args.trim_every_rows, args.thread_policy))

    def periodic(args):
        calls.append(("periodic", args.trim_every_rows, args.thread_policy))
        (tmp_path / "result.json").write_text("{}")

    monkeypatch.setattr(memory, "_run_end_only_child", legacy)
    monkeypatch.setattr(memory, "_run_periodic_child", periodic)
    argv = _arguments(tmp_path, policy="blas-one" if interval is None else "inherited") + ["--child"]
    if interval is not None:
        argv += ["--trim-every-rows", str(interval)]
    monkeypatch.setattr(sys, "argv", argv)
    memory.main()
    if interval is None:
        assert calls == [("legacy", None, "blas-one")]
        assert not (tmp_path / "shutdown.jsonl").exists()
    else:
        assert calls == [("periodic", interval, "inherited")]
        phases = [json.loads(row)["phase"] for row in (tmp_path / "shutdown.jsonl").read_text().splitlines()]
        assert phases == ["leaving_main"]


@pytest.mark.parametrize("policy,rows", [("blas-one", 128), ("inherited", 8)])
def test_incompatible_periodic_controls_fail_before_heavy_imports(tmp_path, policy, rows):
    result = subprocess.run(
        [sys.executable, "-S", *_arguments(tmp_path / "run", policy=policy, rows=rows), "--trim-every-rows", "128"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode != 0 and "periodic" in result.stderr
    assert "ModuleNotFoundError" not in result.stderr
    assert not (tmp_path / "run").exists()


def test_stamps_require_result_and_preserve_rng_and_result(tmp_path):
    before = random.getstate()
    memory.shutdown_phase(tmp_path, "absent_result")
    assert not (tmp_path / "shutdown.jsonl").exists()
    original = b'{"elapsed_seconds":123.0,"complete":true}\n'
    (tmp_path / "result.json").write_bytes(original)
    for phase in ("before_pipeline_close", "after_pipeline_close", "leaving_main"):
        memory.shutdown_phase(tmp_path, phase)
    rows = [json.loads(line) for line in (tmp_path / "shutdown.jsonl").read_text().splitlines()]
    assert [row["phase"] for row in rows] == ["before_pipeline_close", "after_pipeline_close", "leaving_main"]
    assert [row["monotonic_seconds"] for row in rows] == sorted(row["monotonic_seconds"] for row in rows)
    assert (tmp_path / "result.json").read_bytes() == original
    assert random.getstate() == before


@pytest.mark.parametrize("close_fails", [False, True])
def test_actual_periodic_cleanup_order_and_failure_stamps(tmp_path, close_fails):
    (tmp_path / "result.json").write_text("{}")
    events = []

    def close():
        events.append("close")
        if close_fails:
            raise RuntimeError("shutdown failed")

    tree = ast.parse(_PATH.read_text())
    run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_run_measured_child")
    finally_body = next(node for node in run.body if isinstance(node, ast.Try)).finalbody
    scope = {
        "out": tmp_path,
        "shutdown_phase": memory.shutdown_phase,
        "pipeline": SimpleNamespace(close=close),
        "stop": SimpleNamespace(set=lambda: events.append("stop")),
        "watcher": SimpleNamespace(join=lambda timeout: events.append(("join", timeout)), is_alive=lambda: False),
    }
    code = compile(ast.Module(body=finally_body, type_ignores=[]), "<actual-finally-body>", "exec")
    if close_fails:
        with pytest.raises(RuntimeError, match="shutdown failed"):
            exec(code, scope)  # noqa: S102 - execute only the locally parsed cleanup AST
    else:
        exec(code, scope)  # noqa: S102 - execute only the locally parsed cleanup AST
    records = [json.loads(line) for line in (tmp_path / "shutdown.jsonl").read_text().splitlines()]
    if close_fails:
        assert [row["phase"] for row in records] == ["before_pipeline_close"]
        assert events == ["close"]
    else:
        assert [row["phase"] for row in records] == [
            "before_pipeline_close",
            "after_pipeline_close",
            "before_watcher_stop",
            "after_watcher_stop",
            "before_watcher_join",
            "after_watcher_join",
        ]
        assert records[-1]["watcher_alive"] is False
        assert events == ["close", "stop", ("join", 5)]


# Semantic source receipts from the qualified private v4 diagnostic and the
# previous public API. Function-name/protocol aliases are normalized only.
QUALIFIED_V4_SOURCE_SHA256 = "e48747d29c63086291b4598fe018a19d980fdb3c5cbd44fc043ae4ea2e351377"
LEGACY_PUBLIC_SOURCE_SHA256 = "221f9a564c24ce7fbd596a0f2b6639a402e77b678e5f55e9da70c65bc7d2eac5"
QUALIFIED_V4_FUNCTION_AST_SHA256 = {
    "worker_snapshot": "b6c6e39d04330d8e7b903d1a7e60a787bd8cb2e2c7235f5e847b5e4632b681eb",
    "tree_hash": "92ed4ceca0a4b6443f1d043b24a1ea7d55918ea191e7d522b004e55ca432f908",
    "rng_hash": "e1e4857a599b16468d4742601a7e683ea5de05aca6cb1637602f3e44450dad92",
    "process_memory": "692229e4ec9861c7e88a47f9d6315d3fbfa6a2bd8827e6ec23d58302d4feaded",
    "cpu_topology": "f7587896464def5cb7f5a1fe67057fd63fc4fd31a4282b0658d655ec9bf6819d",
    "policy_environment": "5d87e57e15e89b39608bbb23654992c0ea23a01f2bba1f61f2f888fc59b4bdb7",
    "validate_cpu_topology": "697ed7e014d90ea2db8e21b9cd80ebfb95dfd454bde2c983ddb9564c757539f2",
    "worker_install_processing_timer": "3b7568acfa061b0b374dc53db0e2527a1b2abfa52c72fc8741f80e596d23aea3",
    "worker_processing_metrics": "5c99dcdad5d32286a6c5312f0b7a7a12d0b2889c10d984b159f8148a9b3738b8",
    "worker_maintenance": "9a4383a8c57403a740ae1d97ed9aef864eb61aa64a90eb3e29c58bc72b9c954d",
    "drained_maintenance": "1d0c56bd30ef45ac33734e369566efb522ae0bf84f443657e7c79e5e605852fa",
    "shutdown_phase": "7d127ddff196dc36f9f4aa5583ac9181a6f14649a05fa37737cbbe031aa9aa32",
    # The replay loop is now shared with the opt-in arena observation mode.
    # Its old default schema/maintenance path is exercised end-to-end in
    # test_worker_memory_arena.test_shared_replay_preserves_periodic_pair_contract.
    "_compare_periodic_reports": "36f881582bfdb7400afc0e3228fb3da52faeb13f8275f3708c7374d13bbc9de2",
}
LEGACY_FUNCTION_AST_SHA256 = {
    "_run_end_only_child": "50de253230453131c806962c55d07189ce16c62d1db820397b1d26f883420a13",
    "_compare_thread_reports": "fd6a950be98caa83b418cbc72dc18cc710f4cde64172c4995c90eaaad35bef0a",
    "worker_end_diagnostic": "a864b885be449184c048c3f8db6428a7d2acc39a650eb6c6fcfc05f63c126d75",
}


def test_promoted_replay_matches_qualified_v4_and_preserves_legacy_semantics():
    functions = {node.name: node for node in ast.parse(_PATH.read_text()).body if isinstance(node, ast.FunctionDef)}

    class NormalizeAliases(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            node.name = {
                "_run_periodic_child": "run_child",
                "_run_end_only_child": "run_child",
                "_compare_periodic_reports": "compare_reports",
                "_compare_thread_reports": "compare_reports",
            }.get(node.name, node.name)
            return self.generic_visit(node)

        def visit_Name(self, node):
            if node.id == "PERIODIC_PROTOCOL":
                node.id = "PROTOCOL"
            return node

    for name, expected in {**QUALIFIED_V4_FUNCTION_AST_SHA256, **LEGACY_FUNCTION_AST_SHA256}.items():
        node = NormalizeAliases().visit(copy.deepcopy(functions[name]))
        dump = ast.dump(node, show_empty=True) if sys.version_info >= (3, 13) else ast.dump(node)
        assert hashlib.sha256(dump.encode()).hexdigest() == expected, name
