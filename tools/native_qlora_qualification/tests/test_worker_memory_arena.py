# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""Fresh libc policy, real spawned native worker, and unchanged legacy dispatch."""

import ast
import copy
import hashlib
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.native_qlora_qualification import worker_memory as memory


def test_allocator_policy_changes_one_variable_without_mutating_parent():
    inherited = {"OMP_NUM_THREADS": "8", "GLIBC_TUNABLES": "glibc.cpu.hwcaps=-AVX2", "OTHER": "keep"}
    original = copy.deepcopy(inherited)
    for maximum in (0, 2):
        result = memory.arena_environment(inherited, maximum)
        assert result == {**inherited, "MALLOC_ARENA_MAX": str(maximum)}
        assert memory.arena_environment(result, maximum, child=True) == result
    assert inherited == original


@pytest.mark.parametrize(
    "environment",
    [
        {"MALLOC_ARENA_MAX": "0"},
        {"MALLOC_ARENA_MAX": "2"},
        {"MALLOC_TRIM_THRESHOLD_": "1"},
        {"MALLOC_ARENA_TEST": "1"},
        {"MALLOC_CHECK_": "0"},
        {"GLIBC_TUNABLES": "glibc.cpu.hwcaps=-AVX2:glibc.malloc.arena_max=2"},
        {"GLIBC_TUNABLES": "glibc.malloc.tcache_count=0"},
        {"LD_PRELOAD": "/some/library.so"},
    ],
)
def test_conflicting_allocator_environment_is_rejected_not_cleared(environment):
    original = copy.deepcopy(environment)
    with pytest.raises(ValueError, match="conflicting inherited"):
        memory.arena_environment(environment, 2)
    assert environment == original


@pytest.mark.parametrize("maximum", [True, 1, -1, "2", None])
def test_invalid_policy_is_never_implicitly_default(maximum):
    with pytest.raises(ValueError):
        memory.arena_environment({}, maximum)


def test_child_requires_exact_preexec_environment():
    for environment in ({}, {"MALLOC_ARENA_MAX": "0"}, {"MALLOC_ARENA_MAX": "02"}):
        with pytest.raises(ValueError, match="fresh child"):
            memory.arena_environment(environment, 2, child=True)


@pytest.mark.parametrize("maximum", [0, 2])
def test_real_glibc_malloc_info_in_fresh_threaded_process(maximum):
    # ctypes calls release the GIL, so real malloc activity can contend across threads.
    script = r"""
import ctypes,importlib.util,json,os,sys,threading
spec=importlib.util.spec_from_file_location('worker_memory',sys.argv[1]);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
libc=ctypes.CDLL(None);libc.malloc.argtypes=[ctypes.c_size_t];libc.malloc.restype=ctypes.c_void_p;libc.free.argtypes=[ctypes.c_void_p];libc.free.restype=None
start=threading.Barrier(5);ready=threading.Barrier(5);release=threading.Event()
def allocate():
 start.wait();blocks=[libc.malloc(4096) for _ in range(1000)];assert all(blocks)
 ready.wait();release.wait()
 for p in blocks:libc.free(p)
threads=[threading.Thread(target=allocate) for _ in range(4)]
for t in threads:t.start()
start.wait();ready.wait();before_fds=len(os.listdir('/proc/self/fd'))
first=m.malloc_info_snapshot(int(sys.argv[2]))
for _ in range(10):m.malloc_info_snapshot(int(sys.argv[2]))
assert len(os.listdir('/proc/self/fd'))==before_fds
release.set()
for t in threads:t.join()
print(json.dumps({'first':first,'after':m.malloc_info_snapshot(int(sys.argv[2])),'models_imported':any(k.split('.')[0] in {'torch','transformers','cosmos_framework'} for k in sys.modules)}))
"""
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-c", script, memory.__file__, str(maximum)],
        env=memory.arena_environment(dict(os.environ), maximum),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    value = json.loads(result.stdout)
    assert value["models_imported"] is False
    for snapshot in (value["first"], value["after"]):
        assert snapshot["arena_count"] >= 1 and snapshot["arena_system_current_bytes"] > 0
        assert (
            snapshot["arena_system_current_bytes"]
            == snapshot["arena_free_bin_bytes"] + snapshot["arena_nonfree_bytes_including_overhead_and_tcache"]
        )
        assert snapshot["environment"]["MALLOC_ARENA_MAX"] == str(maximum)
        if maximum == 2:
            assert snapshot["arena_count"] <= 2


def xml(count=1):
    heaps = "".join(f'<heap nr="{i}"><system type="current" size="100"/></heap>' for i in range(count))
    return f'<malloc version="1">{heaps}<total type="fast" size="0"/><total type="rest" size="{40 * count}"/><total type="mmap" size="0"/><system type="current" size="{100 * count}"/></malloc>'


@pytest.mark.parametrize(
    "raw",
    [
        "",
        '<malloc version="2"/>',
        xml().replace('nr="0"', 'nr="1"'),
        xml().replace('size="40"', 'size="140"'),
        xml().replace('size="100"', 'size="-1"', 1),
    ],
)
def test_malloc_info_unknown_or_inconsistent_schema_fails(raw):
    with pytest.raises((ValueError, ET.ParseError)):
        memory.parse_malloc_info(raw)


def report(path, maximum):
    path.mkdir()
    (path / "rows.jsonl").write_text('{"all_tensors":"same"}\n' * 8)
    topology = {"affinity": [1], "physical_cpu_count": 1, "unavailable_cpus": []}

    def snapshot(label, cache):
        return {
            "label": label,
            "memory": {"Pss": 100, "Rss": 200},
            "allocator": {"available": True, "glibc": "fixture"},
            "cache": {"samples": {"bytes": cache, "limit": 1000, "hits": 0}},
            "cache_sha256": str(cache),
            "cache_and_rng_unchanged": True,
            "worker_identity": {"pid": maximum + 100, "start_ticks": 1000},
            "rng_sha256": "rng",
            "thread_environment": {"OMP_NUM_THREADS": "8"},
            "cpu_topology": topology,
            "malloc_info": {
                **memory.parse_malloc_info(xml()),
                "requested_arena_max": maximum,
                "glibc": "fixture",
                "environment": {"MALLOC_ARENA_MAX": str(maximum)},
            },
        }

    final = snapshot("rows_8", 8)
    final["rows"] = 8
    (path / "snapshots.jsonl").write_text(json.dumps(final) + "\n")
    value = {
        "protocol": memory.ARENA_PROTOCOL,
        "complete": True,
        "malloc_arena_max": maximum,
        "build_receipt_sha256": "build",
        "source_fingerprint": "data",
        "processor_identity_sha256": "processor",
        "row_output_sha256": memory.file_hash(path / "rows.jsonl"),
        "row_order_sha256": "order",
        "data_state_sha256": "cursor",
        "rng_before_sha256": "rng",
        "rng_after_sha256": "rng",
        "rows": 8,
        "seed": 42,
        "cache_bytes": 2**30,
        "window_rows": 8,
        "consumer_delay_seconds": 0,
        "config": "recipe",
        "override": [],
        "versions": {"torch": "fixture"},
        "thread_policy": "inherited",
        "thread_environment": {"OMP_NUM_THREADS": "8"},
        "parent_affinity": [1],
        "cpu_topology": topology,
        "repeat": 0,
        "expected_physical_cpus": 1,
        "periodic_trim_calls": 0,
        "gc_called": False,
        "worker_rng_unchanged": True,
        "processing_timing": {"calls": 8, "seconds": 1},
        "initial_worker_snapshot": snapshot("initial", 0),
        "snapshots_sha256": memory.file_hash(path / "snapshots.jsonl"),
        "end_diagnostic": {
            "cache_and_rng_unchanged": True,
            "gc_called": False,
            "trim": True,
            "outside_replay_timing": True,
            "cache_sha256": "8",
            "arena_before": snapshot("before", 8),
            "arena_after": snapshot("after", 8),
        },
        "replay_seconds_before_end_diagnostic": 2,
        "peak_worker_pss_bytes": 100,
        "elapsed_seconds": 3,
    }
    memory.write_json(path / "result.json", value)
    memory.write_json(path / "exit.json", {"exit_code": 0})
    memory.write_json(
        path / "launch.json",
        {
            "script_sha256": "source",
            "allocator_policy": {
                "inherited": {},
                "effective": {"MALLOC_ARENA_MAX": str(maximum)},
                "set_before_fresh_interpreter": True,
            },
        },
    )
    return value


def test_compare_requires_actual_matched_fresh_policies(tmp_path):
    a, b = tmp_path / "zero", tmp_path / "two"
    report(a, 0)
    report(b, 2)
    result = memory.compare_reports([a, b])
    assert result["passed"] and result["semantic_and_tensor_bytes_exact"]
    assert result["production_policy_qualified"] is False and result["end_to_end_training_speed_claim"] is False


@pytest.mark.parametrize(
    "fault",
    [
        "same_policy",
        "same_worker",
        "threads",
        "placement",
        "rows",
        "snapshot_hash",
        "arena_count",
        "raw_xml",
        "periodic_trim",
        "end_inside",
        "cache",
        "nonzero_exit",
        "missing_exit",
        "preexec",
        "environment",
    ],
)
def test_comparison_rejects_unmatched_or_incomplete_evidence(tmp_path, fault):
    a, b = tmp_path / "zero", tmp_path / "two"
    report(a, 0)
    value = report(b, 2)
    if fault == "same_policy":
        value["malloc_arena_max"] = 0
    elif fault == "same_worker":
        value["initial_worker_snapshot"]["worker_identity"]["pid"] = 100
    elif fault == "threads":
        value["thread_environment"]["OMP_NUM_THREADS"] = "1"
    elif fault == "placement":
        value["parent_affinity"] = [3]
    elif fault == "rows":
        (b / "rows.jsonl").write_text("changed")
    elif fault == "snapshot_hash":
        (b / "snapshots.jsonl").write_text("changed")
    elif fault == "arena_count":
        value["initial_worker_snapshot"]["malloc_info"].update(memory.parse_malloc_info(xml(3)))
    elif fault == "raw_xml":
        value["initial_worker_snapshot"]["malloc_info"]["arena_count"] = 2
    elif fault == "periodic_trim":
        value["periodic_trim_calls"] = 1
    elif fault == "end_inside":
        value["end_diagnostic"]["outside_replay_timing"] = False
    elif fault == "cache":
        value["end_diagnostic"]["arena_after"]["cache_sha256"] = "different"
    elif fault == "nonzero_exit":
        (b / "exit.json").write_text('{"exit_code":1}')
    elif fault == "missing_exit":
        (b / "exit.json").unlink()
    else:
        launch = json.loads((b / "launch.json").read_text())
        if fault == "preexec":
            launch["allocator_policy"]["set_before_fresh_interpreter"] = False
        else:
            launch["allocator_policy"]["inherited"] = {"MALLOC_TRIM_THRESHOLD_": "1"}
        (b / "launch.json").write_text(json.dumps(launch))
    (b / "result.json").write_text(json.dumps(value))
    with pytest.raises(ValueError):
        memory.compare_reports([a, b])


@pytest.mark.parametrize("maximum", [0, 2])
def test_allocator_snapshot_and_end_only_diagnostic_in_real_native_worker(tmp_path, monkeypatch, maximum):
    from cosmos_framework.posttrain.data.packer.sft import build_sft_packer
    from cosmos_framework.posttrain.data.packer.sft_preprocess import build_sft_preprocessing
    from cosmos_framework.posttrain.schema import SFTTarget
    from cosmos_framework.posttrain.tests.helpers.sft_packer import local_tokenizer

    monkeypatch.setenv("MALLOC_ARENA_MAX", str(maximum))
    local_tokenizer().save_pretrained(tmp_path)
    (tmp_path / "config.json").write_text('{"model_type":"qwen2"}')
    packer = build_sft_packer(str(tmp_path), max_length=128)
    pipeline = build_sft_preprocessing(
        packer, name_or_path=str(tmp_path), cache_bytes=100000, prefetch=True, media_hashes={}, max_pending=1
    )
    target = SFTTarget([{"role": "user", "content": "question"}, {"role": "assistant", "content": "A"}])
    try:
        initial = pipeline._executor.submit(memory.worker_arena_snapshot, "initial", maximum).result(timeout=30)
        pipeline.prepare_window([target])
        sample = pipeline.sft_process_sample(target)
        pipeline.finish_window()
        before = pipeline._executor.submit(memory.worker_arena_snapshot, "before", maximum).result(timeout=30)
        end = pipeline._executor.submit(memory.worker_arena_end_diagnostic, 1, maximum).result(timeout=30)
        assert before["cache_sha256"] == end["arena_before"]["cache_sha256"] == end["arena_after"]["cache_sha256"]
        assert before["worker_identity"] == initial["worker_identity"]
        assert end["gc_called"] is False and end["outside_replay_timing"] is True
        pipeline.prepare_window([target])
        again = pipeline.sft_process_sample(target)
        pipeline.finish_window()
        assert (
            memory.tree_hash(sample) == memory.tree_hash(again) == memory.tree_hash(packer.sft_process_sample(target))
        )
    finally:
        pipeline.close()


def test_shared_replay_retains_qualified_native_row_operations():
    functions = {
        node.name: node
        for node in ast.parse(Path(memory.__file__).read_text()).body
        if isinstance(node, ast.FunctionDef)
    }

    def native_loop(name):
        loop = next(
            node
            for node in ast.walk(functions[name])
            if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == "offset"
        )
        finish = next(
            i
            for i, node in enumerate(loop.body)
            if isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "finish_window"
        )
        node = ast.Module(body=loop.body[: finish + 1], type_ignores=[])
        return ast.dump(node, show_empty=True) if sys.version_info >= (3, 13) else ast.dump(node)

    # Hash of the qualified legacy periodic loop through finish_window, not
    # initialization/observations. Both modes now execute this one exact body.
    assert (
        hashlib.sha256(native_loop("_run_measured_child").encode()).hexdigest()
        == "074fa8a42a491f5c437432614a7a9d5df9bac68f91bc08a563d659b0f4b5496e"
    )


def arguments(output):
    return [
        memory.__file__,
        "run",
        "--config",
        "unused",
        "--model-path",
        "unused",
        "--data-root",
        "unused",
        "--build-receipt",
        "unused",
        "--output",
        str(output),
        "--thread-policy",
        "inherited",
        "--malloc-arena-max",
        "2",
    ]


def test_cli_sets_allocator_before_spawn_and_records_no_unrelated_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("UNRELATED_SECRET_FOR_TEST", "do not record")
    spawned = []

    def launch(argv, **kwargs):
        spawned.append((argv, kwargs))
        return SimpleNamespace(pid=999999, wait=lambda timeout=None: 0)

    monkeypatch.setattr(memory.subprocess, "Popen", launch)
    monkeypatch.setattr(memory.os, "killpg", lambda pid, sig: None)
    monkeypatch.setattr(sys, "argv", arguments(tmp_path / "run"))
    memory.main()
    assert len(spawned) == 1
    argv, options = spawned[0]
    assert argv[-1] == "--child" and options["start_new_session"] is True
    assert options["env"]["MALLOC_ARENA_MAX"] == "2" and options["env"]["UNRELATED_SECRET_FOR_TEST"] == "do not record"
    assert os.environ.get("MALLOC_ARENA_MAX") is None
    raw = (tmp_path / "run/launch.json").read_text()
    assert "UNRELATED_SECRET_FOR_TEST" not in raw and "do not record" not in raw
    policy = json.loads(raw)["allocator_policy"]
    assert policy["effective"] == {**policy["inherited"], "MALLOC_ARENA_MAX": "2"}


@pytest.mark.parametrize("extra", [["--thread-policy", "blas-one"], ["--trim-every-rows", "0"], ["--child"]])
def test_incompatible_cli_or_unbound_child_fails_before_native_imports(tmp_path, extra):
    result = subprocess.run(
        [sys.executable, "-S", "-B", *arguments(tmp_path / "run"), *extra],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode != 0 and "ModuleNotFoundError" not in result.stderr
    assert not (tmp_path / "run").exists()


def test_shared_replay_preserves_periodic_pair_contract(tmp_path, monkeypatch):
    """Run the whole shared loop with deterministic CPU tensors and isolated I/O.

    Native processor/RNG helpers have separate real spawned-worker tests. This
    fixture exercises default dispatch, maintenance ordering, old result schema
    and the unchanged historical paired-artifact validator after loop reuse.
    """
    import dataclasses
    import importlib.metadata
    import time

    import torch

    from cosmos_framework.posttrain.data.packer import sft, sft_preprocess
    from cosmos_framework.posttrain.roles.env import sft as env_sft
    from cosmos_framework.utils import config, lazy_config

    topology = {"physical_cpu_count": 8, "unavailable_cpus": []}
    allow_arena = False

    class Data:
        source_fingerprint = "fixed-data"
        dataset = SimpleNamespace(media_hashes={})

        def __init__(self):
            self.count = 0

        def get_samples(self, count):
            rows = [
                SimpleNamespace(id=str(i), transitions=[SimpleNamespace(targets="same")])
                for i in range(self.count, self.count + count)
            ]
            self.count += count
            return rows

        def state_dict(self):
            return {"count": self.count}

    @dataclasses.dataclass
    class Sample:
        input_ids: list

    class Pipeline:
        def __init__(self):
            self.count = 0
            self._window_active = False
            self._pending = []
            self._executor = self
            self.events = []

        def prepare_window(self, targets):
            self._window_active = True

        def sft_process_sample(self, target):
            self.count += 1
            return Sample([1, 2, 3])

        def sft_compute_max_len(self, samples):
            return 3

        def sft_collate_fn(self, *args):
            return {
                "labels": torch.tensor([[-100, 1, 2]]),
                "attention_mask": torch.ones((1, 3), dtype=torch.int64),
                "pixel_values_videos": torch.ones((1, 2)),
            }

        def finish_window(self):
            self._window_active = False

        def close(self):
            self.events.append(("close", self.count))

        def snapshot(self):
            return {
                "memory": {"Pss": 100, "Rss": 200},
                "cpu_topology": topology,
                "rng_sha256": memory.rng_hash(),
                "cache": {"count": self.count},
            }

        def submit(self, fn, *args):
            if fn is memory.worker_install_processing_timer:
                value = {"rng_sha256": memory.rng_hash()}
            elif fn is memory.worker_processing_metrics:
                value = {"calls": self.count, "seconds": 1}
            elif fn is memory.worker_snapshot:
                value = self.snapshot()
            elif fn is memory.worker_maintenance:
                assert not self._window_active and not self._pending
                rows, trim = args
                self.events.append(("maintenance", rows, trim))
                value = {
                    "rows": rows,
                    "trim": trim,
                    "cache_and_rng_unchanged": True,
                    "gc_called": False,
                    "cache_sha256": str(rows),
                    "before": self.snapshot(),
                    "after": self.snapshot(),
                    "trim_seconds": 0,
                    "verification_seconds": 0,
                }
            elif fn is memory.worker_arena_snapshot:
                assert allow_arena and not self._window_active and not self._pending
                self.events.append(("arena_snapshot", self.count))
                value = self.snapshot()
            elif fn is memory.worker_arena_end_diagnostic:
                assert allow_arena and not self._window_active and not self._pending
                self.events.append(("arena_end", self.count))
                time.sleep(0.02)
                value = {"before": self.snapshot(), "cache_and_rng_unchanged": True}
            else:
                raise AssertionError("default replay invoked arena observation")
            return SimpleNamespace(result=lambda timeout=None: value)

    pipelines = []

    def new_pipeline(*args, **kwargs):
        value = Pipeline()
        pipelines.append(value)
        return value

    monkeypatch.setattr(env_sft, "build_sft_env", lambda *args: Data())
    monkeypatch.setattr(
        sft,
        "build_sft_packer",
        lambda *args, **kwargs: SimpleNamespace(validation_semantics_identity=lambda: {"fixed": True}),
    )
    monkeypatch.setattr(sft_preprocess, "build_sft_preprocessing", new_pipeline)
    monkeypatch.setattr(config, "load_config", lambda *args: SimpleNamespace(rl=None))
    monkeypatch.setattr(
        lazy_config,
        "instantiate",
        lambda _: SimpleNamespace(seed=42, data=None, max_length=4096, processor_kwargs={}, mask_think_opener=False),
    )
    monkeypatch.setattr(memory, "cpu_topology", lambda: topology)
    monkeypatch.setattr(memory, "descendants", lambda pid: [])
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "fixture")
    receipt = tmp_path / "build.json"
    receipt.write_text("{}")
    paths = []
    for interval in (0, 128):
        output = tmp_path / str(interval)
        output.mkdir()
        paths.append(output)
        args = SimpleNamespace(
            expected_physical_cpus=8,
            output=str(output),
            config="fixed",
            override=[],
            seed=42,
            model_path="fixed",
            cache_bytes=2**30,
            max_rss_gib=24,
            rows=128,
            consumer_delay_seconds=0,
            trim_every_rows=interval,
            build_receipt=str(receipt),
            repeat=0,
            thread_policy="inherited",
        )
        memory.run_child(args)
        memory.write_json(output / "exit.json", {"exit_code": 0})
        value = json.loads((output / "result.json").read_text())
        assert value["protocol"] == memory.PERIODIC_PROTOCOL and value["trim_every_rows"] == interval
        assert not {"malloc_arena_max", "initial_worker_snapshot", "snapshots_sha256"} & set(value)
        assert pipelines[-1].events == [
            ("maintenance", 128, bool(interval)),
            ("maintenance", 128, True),
            ("close", 128),
        ]
        assert value["periodic_maintenance"]["calls"] == 1 and value["periodic_maintenance"][
            "actual_trim_calls"
        ] == bool(interval)
    assert memory.compare_reports(paths)["semantic_and_tensor_bytes_exact"] is True
    allow_arena = True
    output = tmp_path / "arena"
    output.mkdir()
    args = SimpleNamespace(**{**vars(args), "output": str(output), "malloc_arena_max": 2, "trim_every_rows": None})
    monkeypatch.setenv("MALLOC_ARENA_MAX", "2")
    memory.run_child(args)
    value = json.loads((output / "result.json").read_text())
    assert value["protocol"] == memory.ARENA_PROTOCOL and value["periodic_trim_calls"] == 0
    assert not {"trim_every_rows", "maintenance_sha256", "periodic_maintenance"} & set(value)
    assert (output / "maintenance.jsonl").read_bytes() == b""
    assert pipelines[-1].events == [
        ("arena_snapshot", 0),
        ("arena_snapshot", 8),
        ("arena_snapshot", 104),
        ("arena_snapshot", 128),
        ("arena_end", 128),
        ("close", 128),
    ]
    assert value["elapsed_seconds"] - value["replay_seconds_before_end_diagnostic"] >= 0.015
