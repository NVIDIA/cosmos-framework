# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""Bounded CPU replay of native preprocessing; never trains or changes a live job.

The launcher owns the sealed runtime, CPU allocation and container. Each ``run``
starts a fresh interpreter before importing NumPy/Torch. ``blas-one`` changes only
the two BLAS environment variables; the inherited OMP setting stays unchanged.
Compare runs before interpreting memory differences. End-only GC/trim is a
diagnostic in the disposable worker, never a production allocator policy.
Explicit ``--trim-every-rows 0|128`` instead selects matched periodic sham/trim
verification with inherited threads and no GC; omitting it preserves defaults.
``--malloc-arena-max 0|2`` selects a separate fresh-process allocation-policy
diagnostic with no periodic maintenance; neither intervention is production policy.
"""

from __future__ import annotations

import argparse
import ctypes
import dataclasses
import hashlib
import json
import math
import os
import platform
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PROTOCOL = "cosmos.native-preprocessing-memory.v1"
PERIODIC_PROTOCOL = "cosmos.native-preprocessing-memory-periodic-trim.v1"
ARENA_PROTOCOL = "cosmos.native-preprocessing-memory-arena.v1"
_WORKER_PROCESSING = {"calls": 0, "seconds": 0.0}
THREAD_KEYS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "TOKENIZERS_PARALLELISM")
MALL_FIELDS = (
    "arena",
    "ordblks",
    "smblks",
    "hblks",
    "hblkhd",
    "usmblks",
    "fsmblks",
    "uordblks",
    "fordblks",
    "keepcost",
)


class Mallinfo2(ctypes.Structure):
    _fields_ = [(key, ctypes.c_size_t) for key in MALL_FIELDS]


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def cpu_topology(*, sysfs_root: Path = Path("/sys/devices/system/cpu")) -> dict[str, Any]:
    """Inspect this process's effective affinity, not the scheduler's requested CPU count."""
    affinity = sorted(os.sched_getaffinity(0))
    records, missing = [], []
    for cpu in affinity:
        directory = sysfs_root / f"cpu{cpu}"
        try:
            socket = int((directory / "topology/physical_package_id").read_text())
            core = int((directory / "topology/core_id").read_text())
        except (OSError, ValueError):
            missing.append(cpu)
            continue
        nodes = sorted(int(path.name[4:]) for path in directory.glob("node[0-9]*"))
        records.append({"logical_cpu": cpu, "socket": socket, "core": core, "numa_nodes": nodes})
    models = set()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                models.add(line.partition(":")[2].strip())
    except OSError:
        pass
    return {
        "hostname": platform.node(),
        "architecture": platform.machine(),
        "models": sorted(models),
        "affinity": affinity,
        "logical_cpu_count": len(affinity),
        "physical_cpu_count": len({(row["socket"], row["core"]) for row in records}) if not missing else None,
        "cpu_records": records,
        "unavailable_cpus": missing,
    }


def validate_cpu_topology(topology: dict[str, Any], expected: int | None) -> None:
    if expected is None:
        return
    if type(expected) is not int or expected < 1:
        raise ValueError("expected physical CPUs must be a positive integer")
    if topology["unavailable_cpus"] or topology["physical_cpu_count"] != expected:
        raise ValueError(
            f"effective CPU affinity requires exactly {expected} physical cores; "
            f"observed {topology['physical_cpu_count']} physical / {topology['logical_cpu_count']} logical"
        )


def policy_environment(environ: dict[str, str], policy: str) -> dict[str, str]:
    result = dict(environ)
    if policy == "blas-one":
        result.update(OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    elif policy != "inherited":
        raise ValueError("unknown thread policy")
    result.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    return result


def arena_environment(environ: dict[str, str], maximum: int, *, child: bool = False) -> dict[str, str]:
    """Set the one intervention before exec; never clear conflicting inherited settings."""
    # glibc2.39: arena_max=0 uses its online-core limit; 2 caps process arenas.
    # https://sourceware.org/glibc/manual/2.39/html_node/Memory-Allocation-Tunables.html
    # This does not constrain thread count or establish how many arenas exist.
    if type(maximum) is not int or maximum not in (0, 2):
        raise ValueError("arena diagnostic requires explicit default0 or maximum2")
    conflicts = [key for key in environ if key.startswith("MALLOC_") and not (child and key == "MALLOC_ARENA_MAX")]
    if any(
        item.partition("=")[0].strip().startswith("glibc.malloc.")
        for item in environ.get("GLIBC_TUNABLES", "").split(":")
    ):
        conflicts.append("GLIBC_TUNABLES:glibc.malloc")
    if environ.get("LD_PRELOAD", "").strip():
        conflicts.append("LD_PRELOAD")
    if conflicts:
        raise ValueError("conflicting inherited allocator settings: " + ", ".join(sorted(conflicts)))
    if child and environ.get("MALLOC_ARENA_MAX") != str(maximum):
        raise ValueError("fresh child allocator environment does not match requested arena policy")
    return {**environ, "MALLOC_ARENA_MAX": str(maximum)}


def parse_malloc_info(raw: str) -> dict[str, Any]:
    """Retain raw versioned all-arena XML; nonfree includes overhead and tcache."""
    import xml.etree.ElementTree as ET

    root = ET.fromstring(raw)
    heaps = root.findall("heap")
    if root.tag != "malloc" or root.get("version") != "1" or not heaps:
        raise ValueError("unsupported or empty malloc_info XML")
    if [heap.get("nr") for heap in heaps] != [str(i) for i in range(len(heaps))]:
        raise ValueError("malloc_info arena indices are not unique and consecutive")

    def size(node: ET.Element, tag: str, kind: str) -> int:
        entries = [entry for entry in node.findall(tag) if entry.get("type") == kind]
        if len(entries) != 1 or not entries[0].get("size", "").isdigit():
            raise ValueError("malloc_info requires one nonnegative " + tag + ":" + kind)
        return int(entries[0].attrib["size"])

    current = size(root, "system", "current")
    free = size(root, "total", "fast") + size(root, "total", "rest")
    if current < free or sum(size(heap, "system", "current") for heap in heaps) != current:
        raise ValueError("malloc_info aggregate arena bytes disagree")
    return {
        "xml": raw,
        "xml_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "arena_count": len(heaps),
        "arena_system_current_bytes": current,
        "arena_free_bin_bytes": free,
        "arena_nonfree_bytes_including_overhead_and_tcache": current - free,
        "mmap_bytes": size(root, "total", "mmap"),
        "scope": "All arenas; free-bin bytes exclude per-thread tcache. Nonfree is not exact live tensor/application memory. XML traversal and observation allocate memory and are not a simultaneous RSS/PSS census.",
    }


def malloc_info_snapshot(maximum: int) -> dict[str, Any]:
    """Bounded private FILE* capture with exact ctypes signatures and owned descriptor."""
    import tempfile

    arena_environment(dict(os.environ), maximum, child=True)
    libc = ctypes.CDLL(None, use_errno=True)
    if any(not hasattr(libc, name) for name in ("gnu_get_libc_version", "malloc_info", "mallinfo2")):
        raise RuntimeError("arena diagnostic requires glibc malloc_info and mallinfo2")
    libc.gnu_get_libc_version.restype = ctypes.c_char_p
    libc.fdopen.argtypes, libc.fdopen.restype = [ctypes.c_int, ctypes.c_char_p], ctypes.c_void_p
    libc.malloc_info.argtypes, libc.malloc_info.restype = [ctypes.c_int, ctypes.c_void_p], ctypes.c_int
    libc.fflush.argtypes, libc.fflush.restype = [ctypes.c_void_p], ctypes.c_int
    libc.fclose.argtypes, libc.fclose.restype = [ctypes.c_void_p], ctypes.c_int
    with tempfile.TemporaryFile(mode="w+b") as temporary:
        descriptor = os.dup(temporary.fileno())
        stream = libc.fdopen(descriptor, b"w")
        if not stream:
            os.close(descriptor)
            raise OSError(ctypes.get_errno(), "fdopen for malloc_info failed")
        try:
            if libc.malloc_info(0, stream) != 0 or libc.fflush(stream) != 0:
                raise OSError(ctypes.get_errno(), "malloc_info capture failed")
        finally:
            if libc.fclose(stream) != 0:
                raise OSError(ctypes.get_errno(), "malloc_info stream close failed")
        temporary.seek(0)
        raw = temporary.read(4 * 1024**2 + 1)
    if len(raw) > 4 * 1024**2:
        raise ValueError("malloc_info exceeded bounded diagnostic size")
    result = parse_malloc_info(raw.decode("utf-8"))
    if maximum == 2 and result["arena_count"] > 2:
        raise RuntimeError("observed allocator arenas exceed the fresh-process maximum2")
    result.update(
        glibc=libc.gnu_get_libc_version().decode(),
        requested_arena_max=maximum,
        environment={
            key: os.environ.get(key) for key in ("MALLOC_ARENA_MAX", "GLIBC_TUNABLES", "LD_PRELOAD", "PYTHONMALLOC")
        },
        online_cpus=os.sysconf("SC_NPROCESSORS_ONLN"),
        pointer_bits=8 * ctypes.sizeof(ctypes.c_void_p),
    )
    return result


def worker_arena_snapshot(label: str, maximum: int) -> dict[str, Any]:
    from cosmos_framework.posttrain.data.packer import sft_preprocess

    cache = sft_preprocess._WORKER_CACHE
    if cache is None:
        raise RuntimeError("arena snapshot requires an initialized native worker")
    before = tree_hash((dict(cache.samples.entries), dict(cache.videos.entries)))
    rng = rng_hash()
    row = worker_snapshot(label)
    row["malloc_info"] = malloc_info_snapshot(maximum)
    if row["memory"].get("unavailable") or any(key not in row["memory"] for key in ("Pss", "Rss")):
        raise RuntimeError("arena diagnostic requires actual worker RSS/PSS")
    if before != tree_hash((dict(cache.samples.entries), dict(cache.videos.entries))) or rng != rng_hash():
        raise RuntimeError("allocator observation changed cache tensor bytes or RNG")
    row.update(
        cache_sha256=before,
        cache_and_rng_unchanged=True,
        worker_identity={
            "pid": os.getpid(),
            "start_ticks": int(Path("/proc/self/stat").read_text().rpartition(")")[2].split()[19]),
        },
    )
    return row


def worker_arena_end_diagnostic(rows: int, maximum: int) -> dict[str, Any]:
    before = worker_arena_snapshot("before_end_only_trim", maximum)
    result = worker_maintenance(rows, True)
    after = worker_arena_snapshot("after_end_only_trim", maximum)
    if before["cache_sha256"] != after["cache_sha256"] or before["cache"] != after["cache"]:
        raise RuntimeError("end-only allocator diagnostic changed live cache")
    result.update(arena_before=before, arena_after=after, outside_replay_timing=True)
    return result


def tree_hash(value: Any) -> str:
    """Hash exact tensor bytes, dtype/shape, metadata and sequence types in bounded chunks."""
    import numpy as np
    import torch

    digest = hashlib.sha256()

    def tag(item: Any) -> None:
        encoded = json.dumps(item, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)

    def visit(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            if item.device.type != "cpu" or item.requires_grad:
                raise ValueError("only detached CPU tensors may enter diagnostic fingerprints")
            tag(["tensor", str(item.dtype), list(item.shape)])
            flat = item.detach().contiguous().reshape(-1).view(torch.uint8).numpy()
            for offset in range(0, flat.size, 4 * 1024**2):
                digest.update(memoryview(flat[offset : offset + 4 * 1024**2]))
        elif isinstance(item, np.ndarray):
            if item.dtype.hasobject:
                raise TypeError("object arrays cannot be fingerprinted")
            tag(["ndarray", str(item.dtype), list(item.shape)])
            flat = np.ascontiguousarray(item).reshape(-1).view(np.uint8)
            for offset in range(0, flat.size, 4 * 1024**2):
                digest.update(memoryview(flat[offset : offset + 4 * 1024**2]))
        elif dataclasses.is_dataclass(item) and not isinstance(item, type):
            tag(["dataclass", type(item).__qualname__])
            for field in dataclasses.fields(item):
                tag(field.name)
                visit(getattr(item, field.name))
        elif isinstance(item, Mapping):
            tag(["mapping", len(item)])
            for key in sorted(item):
                tag(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            tag([type(item).__name__, len(item)])
            for child in item:
                visit(child)
        elif isinstance(item, np.generic):
            visit(item.item())
        elif item is None or isinstance(item, (str, bool, int, float)):
            tag([type(item).__name__, item])
        elif isinstance(item, bytes):
            tag(["bytes", len(item)])
            digest.update(item)
        else:
            raise TypeError(f"unsupported fingerprint value: {type(item).__name__}")

    visit(value)
    return digest.hexdigest()


def rng_hash() -> str:
    import random

    import numpy as np
    import torch

    return tree_hash((random.getstate(), np.random.get_state(), torch.get_rng_state()))


def process_memory(pid: int) -> dict[str, Any]:
    result: dict[str, Any] = {"pid": pid}
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            key, _, raw = line.partition(":")
            if key in {"VmRSS", "VmHWM"}:
                result[key] = int(raw.split()[0]) * 1024
            elif key == "Threads":
                result[key] = int(raw)
            elif key in {"Cpus_allowed_list", "Mems_allowed_list"}:
                result[key] = raw.strip()
        for line in Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines():
            key, _, raw = line.partition(":")
            if raw.strip().endswith("kB"):
                result[key] = int(raw.split()[0]) * 1024
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        result["unavailable"] = True
    return result


def descendants(pid: int) -> list[int]:
    found = [pid]
    try:
        for child in Path(f"/proc/{pid}/task/{pid}/children").read_text().split():
            found.extend(descendants(int(child)))
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        pass
    return found


def worker_snapshot(label: str) -> dict[str, Any]:
    import torch

    from cosmos_framework.posttrain.data.packer import sft_preprocess

    if torch.cuda.is_initialized() or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("diagnostic worker must remain CPU-only")
    cache = sft_preprocess._WORKER_CACHE
    if cache is None:
        raise RuntimeError("native preprocessing worker is not initialized")
    stats = cache.stats()
    if any(item["bytes"] > item["limit"] for item in stats.values()):
        raise RuntimeError("cache exceeded its retained-entry byte budget")
    libc = ctypes.CDLL(None)
    allocator: dict[str, Any] = {"available": False}
    if hasattr(libc, "mallinfo2") and hasattr(libc, "gnu_get_libc_version"):
        libc.gnu_get_libc_version.restype = ctypes.c_char_p
        libc.mallinfo2.argtypes, libc.mallinfo2.restype = [], Mallinfo2
        state = libc.mallinfo2()
        allocator = {
            "available": True,
            "glibc": libc.gnu_get_libc_version().decode(),
            "mallinfo2": {key: getattr(state, key) for key in MALL_FIELDS},
        }
    thread_names = {}
    for task in Path("/proc/self/task").iterdir():
        try:
            name = (task / "comm").read_text().strip()
            thread_names[name] = thread_names.get(name, 0) + 1
        except FileNotFoundError:
            pass
    maps = Path("/proc/self/maps").read_text().splitlines()
    libraries = sorted(
        {
            Path(line.split()[-1]).name
            for line in maps
            if any(word in line.lower() for word in ("blas", "mkl", "libgomp", "libiomp", "avcodec"))
        }
    )
    return {
        "label": label,
        "memory": process_memory(os.getpid()),
        "allocator": allocator,
        "cache": stats,
        "torch_threads": torch.get_num_threads(),
        "thread_names": thread_names,
        "native_libraries": libraries,
        "thread_environment": {key: os.environ.get(key) for key in THREAD_KEYS},
        "fd_count": len(list(Path("/proc/self/fd").iterdir())),
        "cuda_initialized": False,
        "rng_sha256": rng_hash(),
        "cpu_topology": cpu_topology(),
    }


def worker_end_diagnostic() -> dict[str, Any]:
    import gc

    from cosmos_framework.posttrain.data.packer import sft_preprocess

    cache = sft_preprocess._WORKER_CACHE
    assert cache is not None

    def live():
        return tree_hash((dict(cache.samples.entries), dict(cache.videos.entries)))

    original, rng = live(), rng_hash()
    before = worker_snapshot("before_gc")
    collected = gc.collect()
    after_gc = worker_snapshot("after_gc")
    libc = ctypes.CDLL(None)
    trim = None
    if hasattr(libc, "malloc_trim"):
        libc.malloc_trim.argtypes, libc.malloc_trim.restype = [ctypes.c_size_t], ctypes.c_int
        trim = libc.malloc_trim(0)
    after_trim = worker_snapshot("after_end_only_trim")
    if live() != original or rng_hash() != rng or not before["cache"] == after_gc["cache"] == after_trim["cache"]:
        raise RuntimeError("end diagnostic changed live cache contents, counters or RNG")
    return {
        "before": before,
        "after_gc": after_gc,
        "after_trim": after_trim,
        "gc_collected": collected,
        "malloc_trim_return": trim,
        "cache_sha256": original,
        "cache_and_rng_unchanged": True,
    }


def _compare_thread_reports(paths: list[Path]) -> dict[str, Any]:
    if len(paths) < 2:
        raise ValueError("comparison requires at least two completed replays")
    reports = [json.loads((path / "result.json").read_text()) for path in paths]
    required = (
        "build_receipt_sha256",
        "source_fingerprint",
        "processor_identity_sha256",
        "row_output_sha256",
        "row_order_sha256",
        "data_state_sha256",
        "rng_before_sha256",
        "rng_after_sha256",
        "rows",
        "seed",
        "cache_bytes",
        "window_rows",
        "consumer_delay_seconds",
        "config",
        "override",
        "versions",
    )
    for path, report in zip(paths, reports, strict=True):
        if report.get("protocol") != PROTOCOL or report.get("complete") is not True:
            raise ValueError("incomplete or incompatible diagnostic report")
        if report["rng_before_sha256"] != report["rng_after_sha256"]:
            raise ValueError("replay changed main RNG")
        if not report["worker_rng_unchanged"] or not report["end_diagnostic"]["cache_and_rng_unchanged"]:
            raise ValueError("worker RNG/cache check failed")
        for key in required:
            if report[key] != reports[0][key]:
                raise ValueError(f"replays differ in {key}")
        records = path / "rows.jsonl"
        if file_hash(records) != report["row_output_sha256"]:
            raise ValueError("row records differ from receipt")
    return {
        "protocol": PROTOCOL,
        "passed": True,
        "semantic_and_tensor_bytes_exact": True,
        "inputs": [
            {
                "result_sha256": file_hash(path / "result.json"),
                "policy": report["thread_policy"],
                "repeat": report["repeat"],
                "peak_worker_pss_bytes": report["peak_worker_pss_bytes"],
                "elapsed_seconds": report["elapsed_seconds"],
            }
            for path, report in zip(paths, reports, strict=True)
        ],
        "compared_fields": list(required),
        "causality_requires_matched_cpu_placement": True,
    }


def _run_end_only_child(args: argparse.Namespace) -> None:
    topology = cpu_topology()
    validate_cpu_topology(topology, args.expected_physical_cpus)
    import importlib.metadata
    import random

    import numpy as np
    import torch

    from cosmos_framework.posttrain.data.packer.sft import build_sft_packer
    from cosmos_framework.posttrain.data.packer.sft_preprocess import build_sft_preprocessing
    from cosmos_framework.posttrain.roles.env.sft import build_sft_env
    from cosmos_framework.utils.config import load_config
    from cosmos_framework.utils.lazy_config import instantiate

    if torch.cuda.is_initialized():
        raise RuntimeError("CPU replay inherited CUDA initialization")
    out = Path(args.output)
    config = load_config(args.config, ["--", *args.override])
    rl = instantiate(config.rl)
    if rl.seed != args.seed:
        raise ValueError("CLI seed must match the native recipe")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    data = build_sft_env(rl.data, args.seed)
    packer = build_sft_packer(
        args.model_path,
        max_length=rl.max_length,
        processor_kwargs=rl.processor_kwargs,
        mask_think_opener=rl.mask_think_opener,
        trust_remote_code=False,
    )
    pipeline = build_sft_preprocessing(
        packer,
        name_or_path=args.model_path,
        cache_bytes=args.cache_bytes,
        prefetch=True,
        media_hashes=data.dataset.media_hashes,
        max_pending=8,
    )
    stop, peaks = threading.Event(), {"Pss": 0, "Rss": 0}

    def monitor() -> None:
        with (out / "host-memory.jsonl").open("x") as stream:
            while not stop.is_set():
                rows = [process_memory(pid) for pid in descendants(os.getpid())]
                totals = {key: sum(row.get(key, 0) for row in rows) for key in peaks}
                for key in peaks:
                    peaks[key] = max(peaks[key], totals[key])
                stream.write(json.dumps({"unix_seconds": time.time(), "processes": rows, "sum": totals}) + "\n")
                stream.flush()
                if totals["Rss"] > args.max_rss_gib * 1024**3:
                    os._exit(97)  # Outer process terminates this entire owned process group.
                stop.wait(1)

    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    snapshots, order = [], hashlib.sha256()
    before, started, next_snapshot = rng_hash(), time.monotonic(), 0
    try:
        with (out / "rows.jsonl").open("x") as records:
            for offset in range(0, args.rows, 8):
                trajectories = data.get_samples(8)
                targets = [row.transitions[0].targets for row in trajectories]
                pipeline.prepare_window(targets)
                for trajectory, target in zip(trajectories, targets, strict=True):
                    sample = pipeline.sft_process_sample(target)
                    batch = pipeline.sft_collate_fn([sample], pipeline.sft_compute_max_len([sample]), -100)
                    if not batch.get("pixel_values_videos", torch.empty(0)).numel():
                        raise RuntimeError("WTS replay requires actual video payloads")
                    valid = (
                        (batch["labels"][:, 1:] != -100)
                        & batch["attention_mask"][:, 1:].bool()
                        & batch["attention_mask"][:, :-1].bool()
                    )
                    record = {
                        "row_identity_sha256": hashlib.sha256(trajectory.id.encode()).hexdigest(),
                        "sample_sha256": tree_hash(sample),
                        "batch_sha256": tree_hash(batch),
                        "sequence_length": len(sample.input_ids),
                        "supervised_tokens": int(valid.sum()),
                        "tensor_shapes": {key: list(value.shape) for key, value in batch.items()},
                    }
                    records.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
                    order.update((trajectory.id + "\n").encode())
                    del sample, batch, valid
                    if args.consumer_delay_seconds:
                        time.sleep(args.consumer_delay_seconds)
                pipeline.finish_window()
                if rng_hash() != before:
                    raise RuntimeError("CPU replay changed main RNG")
                done = offset + 8
                if done >= next_snapshot or done == args.rows:
                    row = pipeline._executor.submit(worker_snapshot, f"rows_{done}").result(timeout=60)
                    validate_cpu_topology(row["cpu_topology"], args.expected_physical_cpus)
                    row.update(rows=done, elapsed_seconds=time.monotonic() - started)
                    snapshots.append(row)
                    with (out / "snapshots.jsonl").open("a") as stream:
                        stream.write(json.dumps(row, sort_keys=True) + "\n")
                    print(json.dumps({"rows": done, "worker_pss_bytes": row["memory"].get("Pss")}), flush=True)
                    next_snapshot = (done // 100 + 1) * 100
                del trajectories, targets
        end = pipeline._executor.submit(worker_end_diagnostic).result(timeout=180)
        worker_rng = {row["rng_sha256"] for row in snapshots}
        if len(worker_rng) != 1 or end["before"]["rng_sha256"] not in worker_rng:
            raise RuntimeError("worker RNG changed across preprocessing windows")
        result = {
            "protocol": PROTOCOL,
            "complete": True,
            "training_performed": False,
            "cuda_initialized": False,
            "output": str(out.resolve()),
            "config": args.config,
            "override": args.override,
            "build_receipt_sha256": file_hash(Path(args.build_receipt)),
            "rows": args.rows,
            "seed": args.seed,
            "repeat": args.repeat,
            "thread_policy": args.thread_policy,
            "thread_environment": {key: os.environ.get(key) for key in THREAD_KEYS},
            "parent_affinity": sorted(os.sched_getaffinity(0)),
            "cpu_topology": topology,
            "expected_physical_cpus": args.expected_physical_cpus,
            "source_fingerprint": data.source_fingerprint,
            "processor_identity_sha256": json_hash(packer.validation_semantics_identity()),
            "row_order_sha256": order.hexdigest(),
            "row_output_sha256": file_hash(out / "rows.jsonl"),
            "data_state_sha256": tree_hash(data.state_dict()),
            "rng_before_sha256": before,
            "rng_after_sha256": rng_hash(),
            "worker_rng_unchanged": True,
            "cache_bytes": args.cache_bytes,
            "window_rows": 8,
            "consumer_delay_seconds": args.consumer_delay_seconds,
            "end_diagnostic": end,
            "peak_worker_pss_bytes": max(row["memory"].get("Pss", 0) for row in snapshots),
            "sampled_tree_peaks_bytes": peaks,
            "elapsed_seconds": time.monotonic() - started,
            "versions": {
                key: importlib.metadata.version(key) for key in ("torch", "numpy", "transformers", "torchcodec")
            },
            "limitations": [
                "No model or concurrent GPU training; CPU replay is a separate experiment.",
                "Boundary worker samples and 1Hz process-tree samples are lower bounds.",
                "Thread environment is an intervention, not proof of decoder or allocator attribution.",
                "RSS/PSS excludes unmapped filesystem page cache; HWM is lifetime, not phase-local.",
                "End-only trim is diagnostic; no production allocator change is qualified.",
            ],
        }
        write_json(out / "result.json", result)
    finally:
        pipeline.close()
        stop.set()
        watcher.join(timeout=5)


def worker_install_processing_timer() -> dict[str, Any]:
    """Instrument both arms identically, without changing processing or cache keys."""
    from cosmos_framework.posttrain.data.packer import sft_preprocess

    cache = sft_preprocess._WORKER_CACHE
    if cache is None or _WORKER_PROCESSING["calls"]:
        raise RuntimeError("install the private timer once before any native processing")
    if getattr(cache, "_private_memory_diagnostic_timed", False):
        raise RuntimeError("private worker timer already installed")
    original = cache.process

    def measured(target):
        began = time.perf_counter()
        result = original(target)
        _WORKER_PROCESSING["seconds"] += time.perf_counter() - began
        _WORKER_PROCESSING["calls"] += 1
        return result

    cache.process = measured
    cache._private_memory_diagnostic_timed = True
    return {"installed": True, "rng_sha256": rng_hash()}


def worker_processing_metrics() -> dict[str, Any]:
    return dict(_WORKER_PROCESSING)


def worker_maintenance(rows: int, trim: bool) -> dict[str, Any]:
    """Private worker-only trim or matched sham; no GC, eviction or RNG changes."""
    from cosmos_framework.posttrain.data.packer import sft_preprocess

    if type(rows) is not int or rows < 1 or type(trim) is not bool:
        raise ValueError("maintenance requires a positive completed-row count and explicit bool")
    cache = sft_preprocess._WORKER_CACHE
    if cache is None:
        raise RuntimeError("maintenance requires an initialized native worker cache")

    def live():
        return tree_hash((dict(cache.samples.entries), dict(cache.videos.entries)))

    verification_started = time.perf_counter()
    original, rng = live(), rng_hash()
    before = worker_snapshot("before_periodic_trim" if trim else "before_periodic_sham")
    verification_before = time.perf_counter() - verification_started
    trim_seconds, trim_return = 0.0, None
    if trim:
        libc = ctypes.CDLL(None)
        if not hasattr(libc, "malloc_trim") or not hasattr(libc, "gnu_get_libc_version"):
            raise RuntimeError("periodic trim diagnostic requires glibc malloc_trim")
        libc.malloc_trim.argtypes, libc.malloc_trim.restype = [ctypes.c_size_t], ctypes.c_int
        began = time.perf_counter()
        trim_return = libc.malloc_trim(0)
        trim_seconds = time.perf_counter() - began
    after = worker_snapshot("after_periodic_trim" if trim else "after_periodic_sham")
    verification_started = time.perf_counter()
    if live() != original or rng_hash() != rng or before["cache"] != after["cache"]:
        raise RuntimeError("maintenance changed live cache contents, counters or RNG")
    return {
        "rows": rows,
        "trim": trim,
        "before": before,
        "after": after,
        "trim_seconds": trim_seconds,
        "verification_seconds": verification_before + time.perf_counter() - verification_started,
        "malloc_trim_return": trim_return,
        "cache_sha256": original,
        "cache_and_rng_unchanged": True,
        "gc_called": False,
    }


def drained_maintenance(pipeline: Any, rows: int, trim: bool) -> dict[str, Any]:
    if pipeline._window_active or pipeline._pending or pipeline._executor is None:
        raise RuntimeError("maintenance requires a drained native window and its owned worker")
    return pipeline._executor.submit(worker_maintenance, rows, trim).result(timeout=180)


def shutdown_phase(out: Path, phase: str, **observations: Any) -> None:
    """Post-result diagnostics only; replay and result timing remain unchanged."""
    if not (out / "result.json").is_file():
        return
    record = {
        "phase": phase,
        "unix_seconds": time.time(),
        "monotonic_seconds": time.monotonic(),
        "pid": os.getpid(),
        **observations,
    }
    with (out / "shutdown.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")


def _compare_periodic_reports(paths: list[Path]) -> dict[str, Any]:
    if len(paths) < 2:
        raise ValueError("comparison requires at least two completed replays")
    reports = [json.loads((path / "result.json").read_text()) for path in paths]
    required = (
        "build_receipt_sha256",
        "source_fingerprint",
        "processor_identity_sha256",
        "row_output_sha256",
        "row_order_sha256",
        "data_state_sha256",
        "rng_before_sha256",
        "rng_after_sha256",
        "rows",
        "seed",
        "cache_bytes",
        "window_rows",
        "consumer_delay_seconds",
        "config",
        "override",
        "versions",
        "thread_policy",
        "thread_environment",
        "parent_affinity",
        "cpu_topology",
        "repeat",
    )
    if len(paths) != 2 or {report["trim_every_rows"] for report in reports} != {0, 128}:
        raise ValueError("comparison requires exactly one disabled and one128-row trim replay")
    baseline_cache_hashes = None
    for path, report in zip(paths, reports, strict=True):
        if report.get("protocol") != PERIODIC_PROTOCOL or report.get("complete") is not True:
            raise ValueError("incomplete or incompatible diagnostic report")
        if report["rng_before_sha256"] != report["rng_after_sha256"]:
            raise ValueError("replay changed main RNG")
        if not report["worker_rng_unchanged"] or not report["end_diagnostic"]["cache_and_rng_unchanged"]:
            raise ValueError("worker RNG/cache check failed")
        for key in required:
            if report[key] != reports[0][key]:
                raise ValueError(f"replays differ in {key}")
        if report["thread_policy"] != "inherited" or report["processing_timing"]["calls"] != report["rows"]:
            raise ValueError("paired trim replay changed thread policy or preprocessing call count")
        maintenance = [json.loads(line) for line in (path / "maintenance.jsonl").read_text().splitlines()]
        if file_hash(path / "maintenance.jsonl") != report["maintenance_sha256"]:
            raise ValueError("maintenance receipt differs")
        expected_boundaries = list(range(128, report["rows"] + 1, 128))
        if [item["rows"] for item in maintenance] != expected_boundaries:
            raise ValueError("maintenance did not occur at exact drained128-row boundaries")
        if any(
            item["trim"] != bool(report["trim_every_rows"]) or not item["cache_and_rng_unchanged"] or item["gc_called"]
            for item in maintenance
        ):
            raise ValueError("maintenance intervention or cache/RNG preservation differs")
        if report["periodic_maintenance"]["calls"] != len(expected_boundaries):
            raise ValueError("maintenance count differs")
        cache_hashes = [item["cache_sha256"] for item in maintenance] + [report["end_diagnostic"]["cache_sha256"]]
        if baseline_cache_hashes is None:
            baseline_cache_hashes = cache_hashes
        elif cache_hashes != baseline_cache_hashes:
            raise ValueError("live cache tensor bytes differ across paired maintenance boundaries")
        if report["end_diagnostic"]["before"]["cache"] != reports[0]["end_diagnostic"]["before"]["cache"]:
            raise ValueError("final native cache counters differ across arms")
        records = path / "rows.jsonl"
        if file_hash(records) != report["row_output_sha256"]:
            raise ValueError("row records differ from receipt")
    return {
        "protocol": PERIODIC_PROTOCOL,
        "passed": True,
        "semantic_and_tensor_bytes_exact": True,
        "inputs": [
            {
                "result_sha256": file_hash(path / "result.json"),
                "policy": report["thread_policy"],
                "trim_every_rows": report["trim_every_rows"],
                "processing_timing": report["processing_timing"],
                "periodic_maintenance": report["periodic_maintenance"],
                "replay_seconds_before_end_diagnostic": report["replay_seconds_before_end_diagnostic"],
                "repeat": report["repeat"],
                "peak_worker_pss_bytes": report["peak_worker_pss_bytes"],
                "elapsed_seconds": report["elapsed_seconds"],
            }
            for path, report in zip(paths, reports, strict=True)
        ],
        "compared_fields": list(required),
        "causality_requires_matched_cpu_placement": True,
    }


def _run_measured_child(args: argparse.Namespace) -> None:
    """One native replay loop; arena mode changes observations, never row processing."""
    arena_max = getattr(args, "malloc_arena_max", None)
    if arena_max is not None:
        arena_environment(dict(os.environ), arena_max, child=True)
    snapshot = worker_snapshot if arena_max is None else worker_arena_snapshot
    snapshot_args = () if arena_max is None else (arena_max,)
    topology = cpu_topology()
    validate_cpu_topology(topology, args.expected_physical_cpus)
    import importlib.metadata
    import random

    import numpy as np
    import torch

    from cosmos_framework.posttrain.data.packer.sft import build_sft_packer
    from cosmos_framework.posttrain.data.packer.sft_preprocess import build_sft_preprocessing
    from cosmos_framework.posttrain.roles.env.sft import build_sft_env
    from cosmos_framework.utils.config import load_config
    from cosmos_framework.utils.lazy_config import instantiate

    if torch.cuda.is_initialized():
        raise RuntimeError("CPU replay inherited CUDA initialization")
    out = Path(args.output)
    config = load_config(args.config, ["--", *args.override])
    rl = instantiate(config.rl)
    if rl.seed != args.seed:
        raise ValueError("CLI seed must match the native recipe")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    data = build_sft_env(rl.data, args.seed)
    packer = build_sft_packer(
        args.model_path,
        max_length=rl.max_length,
        processor_kwargs=rl.processor_kwargs,
        mask_think_opener=rl.mask_think_opener,
        trust_remote_code=False,
    )
    pipeline = build_sft_preprocessing(
        packer,
        name_or_path=args.model_path,
        cache_bytes=args.cache_bytes,
        prefetch=True,
        media_hashes=data.dataset.media_hashes,
        max_pending=8,
    )
    stop, peaks = threading.Event(), {"Pss": 0, "Rss": 0}

    def monitor() -> None:
        with (out / "host-memory.jsonl").open("x") as stream:
            while not stop.is_set():
                rows = [process_memory(pid) for pid in descendants(os.getpid())]
                totals = {key: sum(row.get(key, 0) for row in rows) for key in peaks}
                for key in peaks:
                    peaks[key] = max(peaks[key], totals[key])
                stream.write(json.dumps({"unix_seconds": time.time(), "processes": rows, "sum": totals}) + "\n")
                stream.flush()
                if totals["Rss"] > args.max_rss_gib * 1024**3:
                    os._exit(97)  # Outer process terminates this entire owned process group.
                stop.wait(1)

    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    snapshots, maintenance_records, order = [], [], hashlib.sha256()
    maintenance_wait_seconds = 0.0
    before, started, next_snapshot = rng_hash(), time.monotonic(), 0
    try:
        installed = pipeline._executor.submit(worker_install_processing_timer).result(timeout=180)
        initial = None
        if arena_max is not None:
            initial = pipeline._executor.submit(snapshot, "initial", *snapshot_args).result(timeout=60)
        with (out / "maintenance.jsonl").open("x"):
            pass
        with (out / "rows.jsonl").open("x") as records:
            for offset in range(0, args.rows, 8):
                trajectories = data.get_samples(8)
                targets = [row.transitions[0].targets for row in trajectories]
                pipeline.prepare_window(targets)
                for trajectory, target in zip(trajectories, targets, strict=True):
                    sample = pipeline.sft_process_sample(target)
                    batch = pipeline.sft_collate_fn([sample], pipeline.sft_compute_max_len([sample]), -100)
                    if not batch.get("pixel_values_videos", torch.empty(0)).numel():
                        raise RuntimeError("WTS replay requires actual video payloads")
                    valid = (
                        (batch["labels"][:, 1:] != -100)
                        & batch["attention_mask"][:, 1:].bool()
                        & batch["attention_mask"][:, :-1].bool()
                    )
                    record = {
                        "row_identity_sha256": hashlib.sha256(trajectory.id.encode()).hexdigest(),
                        "sample_sha256": tree_hash(sample),
                        "batch_sha256": tree_hash(batch),
                        "sequence_length": len(sample.input_ids),
                        "supervised_tokens": int(valid.sum()),
                        "tensor_shapes": {key: list(value.shape) for key, value in batch.items()},
                    }
                    records.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
                    order.update((trajectory.id + "\n").encode())
                    del sample, batch, valid
                    if args.consumer_delay_seconds:
                        time.sleep(args.consumer_delay_seconds)
                pipeline.finish_window()
                if rng_hash() != before:
                    raise RuntimeError("CPU replay changed main RNG")
                done = offset + 8
                if arena_max is None and done % 128 == 0:
                    began = time.perf_counter()
                    maintenance = drained_maintenance(pipeline, done, bool(args.trim_every_rows))
                    maintenance_wait_seconds += time.perf_counter() - began
                    maintenance_records.append(maintenance)
                    with (out / "maintenance.jsonl").open("a") as stream:
                        stream.write(json.dumps(maintenance, sort_keys=True) + "\n")
                    if rng_hash() != before:
                        raise RuntimeError("worker maintenance changed parent RNG")
                if done >= next_snapshot or done == args.rows:
                    row = pipeline._executor.submit(snapshot, f"rows_{done}", *snapshot_args).result(timeout=60)
                    validate_cpu_topology(row["cpu_topology"], args.expected_physical_cpus)
                    row.update(rows=done, elapsed_seconds=time.monotonic() - started)
                    snapshots.append(row)
                    with (out / "snapshots.jsonl").open("a") as stream:
                        stream.write(json.dumps(row, sort_keys=True) + "\n")
                    print(json.dumps({"rows": done, "worker_pss_bytes": row["memory"].get("Pss")}), flush=True)
                    next_snapshot = (done // 100 + 1) * 100
                del trajectories, targets
        replay_seconds = time.monotonic() - started
        processing = pipeline._executor.submit(worker_processing_metrics).result(timeout=60)
        if processing["calls"] != args.rows:
            raise RuntimeError("private worker timer missed native preprocessing calls")
        # Both arms receive the same no-GC end-only trim AFTER replay timing and
        # all1,000-row boundary snapshots; it is separate from the intervention.
        if arena_max is None:
            end = drained_maintenance(pipeline, args.rows, True)
        else:
            if pipeline._window_active or pipeline._pending:
                raise RuntimeError("arena end diagnostic requires drained replay")
            end = pipeline._executor.submit(worker_arena_end_diagnostic, args.rows, arena_max).result(timeout=180)
        worker_rng = {row["rng_sha256"] for row in snapshots}
        if (
            len(worker_rng) != 1
            or end["before"]["rng_sha256"] not in worker_rng
            or installed["rng_sha256"] not in worker_rng
        ):
            raise RuntimeError("worker RNG changed across preprocessing windows")
        result = {
            "protocol": PERIODIC_PROTOCOL,
            "complete": True,
            "training_performed": False,
            "cuda_initialized": False,
            "output": str(out.resolve()),
            "config": args.config,
            "override": args.override,
            "build_receipt_sha256": file_hash(Path(args.build_receipt)),
            "rows": args.rows,
            "seed": args.seed,
            "repeat": args.repeat,
            "thread_policy": args.thread_policy,
            "trim_every_rows": args.trim_every_rows,
            "maintenance_sha256": file_hash(out / "maintenance.jsonl"),
            "processing_timing": processing,
            "periodic_maintenance": {
                "calls": len(maintenance_records),
                "actual_trim_calls": sum(item["trim"] for item in maintenance_records),
                "trim_seconds": sum(item["trim_seconds"] for item in maintenance_records),
                "verification_seconds": sum(item["verification_seconds"] for item in maintenance_records),
                "parent_wait_seconds": maintenance_wait_seconds,
                "sham_uses_same_snapshots_and_cache_hashing": True,
                "gc_called": False,
            },
            "replay_seconds_before_end_diagnostic": replay_seconds,
            "timing_scope": "Worker process() timings exclude IPC and maintenance; they overlap parent execution and are not additive. Replay time includes startup, row hashing, snapshots and matched maintenance/sham verification; excludes end-only diagnostic.",
            "thread_environment": {key: os.environ.get(key) for key in THREAD_KEYS},
            "parent_affinity": sorted(os.sched_getaffinity(0)),
            "cpu_topology": topology,
            "expected_physical_cpus": args.expected_physical_cpus,
            "source_fingerprint": data.source_fingerprint,
            "processor_identity_sha256": json_hash(packer.validation_semantics_identity()),
            "row_order_sha256": order.hexdigest(),
            "row_output_sha256": file_hash(out / "rows.jsonl"),
            "data_state_sha256": tree_hash(data.state_dict()),
            "rng_before_sha256": before,
            "rng_after_sha256": rng_hash(),
            "worker_rng_unchanged": True,
            "cache_bytes": args.cache_bytes,
            "window_rows": 8,
            "consumer_delay_seconds": args.consumer_delay_seconds,
            "end_diagnostic": end,
            "peak_worker_pss_bytes": max(
                row["memory"].get("Pss", 0)
                for row in [
                    *snapshots,
                    *(item[side] for item in maintenance_records for side in ("before", "after")),
                    end["before"],
                ]
            ),
            "sampled_tree_peaks_bytes": peaks,
            "elapsed_seconds": time.monotonic() - started,
            "versions": {
                key: importlib.metadata.version(key) for key in ("torch", "numpy", "transformers", "torchcodec")
            },
            "limitations": [
                "No model or concurrent GPU training; CPU replay is a separate experiment.",
                "Boundary worker samples and 1Hz process-tree samples are lower bounds.",
                "Thread environment is inherited identically; only periodic malloc_trim differs from matched sham verification.",
                "RSS/PSS excludes unmapped filesystem page cache; HWM is lifetime, not phase-local.",
                "Periodic and end-only trim are private diagnostics; no production allocator change is qualified. No GC or explicit cache eviction is performed.",
            ],
        }
        if arena_max is not None:
            for key in ("trim_every_rows", "maintenance_sha256", "periodic_maintenance"):
                del result[key]
            result.update(
                protocol=ARENA_PROTOCOL,
                malloc_arena_max=arena_max,
                initial_worker_snapshot=initial,
                snapshots_sha256=file_hash(out / "snapshots.jsonl"),
                periodic_trim_calls=0,
                gc_called=False,
                allocation_policy_scope="Fresh replay interpreter and spawned worker; worker allocator/PSS is separate from process-tree lower bounds.",
                timing_scope="Worker process timings exclude IPC and observation and overlap parent execution. Replay time includes worker setup, row hashing and matched allocator/cache snapshots; excludes the common end-only diagnostic. Earlier parent/processor setup and full process exit belong to outer command time.",
                limitations=[
                    "CPU preprocessing diagnostic, no model/GPU training or production allocator qualification.",
                    "Only explicit MALLOC_ARENA_MAX differs; no periodic trim, explicit GC or cache eviction during replay.",
                    "Both arms receive the same no-GC end-only trim outside replay timing.",
                    "Worker/tree RSS/PSS samples are lower bounds and exclude unmapped filesystem page cache; HWM is lifetime.",
                    "malloc_info free bins exclude thread tcache; nonfree includes overhead and is not exact live tensor memory.",
                ],
            )
        write_json(out / "result.json", result)
    finally:
        shutdown_phase(out, "before_pipeline_close")
        pipeline.close()
        shutdown_phase(out, "after_pipeline_close")
        shutdown_phase(out, "before_watcher_stop")
        stop.set()
        shutdown_phase(out, "after_watcher_stop")
        shutdown_phase(out, "before_watcher_join")
        watcher.join(timeout=5)
        shutdown_phase(out, "after_watcher_join", watcher_alive=watcher.is_alive())


def _run_periodic_child(args: argparse.Namespace) -> None:
    _run_measured_child(args)


def _run_arena_child(args: argparse.Namespace) -> None:
    _run_measured_child(args)


def _compare_arena_reports(paths: list[Path]) -> dict[str, Any]:
    reports = [json.loads((path / "result.json").read_text()) for path in paths]
    if len(reports) != 2 or {report.get("malloc_arena_max") for report in reports} != {0, 2}:
        raise ValueError("arena comparison requires exactly explicit default0 and maximum2")
    required = (
        "build_receipt_sha256",
        "source_fingerprint",
        "processor_identity_sha256",
        "row_output_sha256",
        "row_order_sha256",
        "data_state_sha256",
        "rng_before_sha256",
        "rng_after_sha256",
        "rows",
        "seed",
        "cache_bytes",
        "window_rows",
        "consumer_delay_seconds",
        "config",
        "override",
        "versions",
        "thread_policy",
        "thread_environment",
        "parent_affinity",
        "cpu_topology",
        "repeat",
        "expected_physical_cpus",
    )
    cache_records = []
    origins = []
    launch_sources, launch_environments, allocator_versions = [], [], []
    for path, report in zip(paths, reports, strict=True):
        if report.get("protocol") != ARENA_PROTOCOL or report.get("complete") is not True:
            raise ValueError("incomplete or incompatible arena report")
        maximum = report["malloc_arena_max"]
        if type(maximum) is not int or maximum not in (0, 2):
            raise ValueError("arena policy must be an explicit integer")
        if (
            report["thread_policy"] != "inherited"
            or report["periodic_trim_calls"] != 0
            or report["gc_called"] is not False
        ):
            raise ValueError("arena replay changed threads or performed periodic maintenance")
        if report["rng_before_sha256"] != report["rng_after_sha256"] or report["worker_rng_unchanged"] is not True:
            raise ValueError("arena replay changed RNG")
        if report["processing_timing"]["calls"] != report["rows"]:
            raise ValueError("arena preprocessing call count differs")
        for key in required:
            if report[key] != reports[0][key]:
                raise ValueError(f"arena replays differ in {key}")
        if file_hash(path / "rows.jsonl") != report["row_output_sha256"]:
            raise ValueError("arena row records differ from receipt")
        if len((path / "rows.jsonl").read_text().splitlines()) != report["rows"]:
            raise ValueError("arena row count differs")
        if file_hash(path / "snapshots.jsonl") != report["snapshots_sha256"]:
            raise ValueError("arena snapshot receipt differs")
        snapshots = [json.loads(line) for line in (path / "snapshots.jsonl").read_text().splitlines()]
        expected, next_boundary = [], 0
        for rows in range(8, report["rows"] + 1, 8):
            if rows >= next_boundary or rows == report["rows"]:
                expected.append(rows)
                next_boundary = (rows // 100 + 1) * 100
        if [row["rows"] for row in snapshots] != expected:
            raise ValueError("arena snapshots differ from original drained observation cadence")
        end = report["end_diagnostic"]
        if (
            not end["cache_and_rng_unchanged"]
            or end["gc_called"]
            or end["trim"] is not True
            or end["outside_replay_timing"] is not True
        ):
            raise ValueError("arena end-only diagnostic scope differs")
        observed = [report["initial_worker_snapshot"], *snapshots, end["arena_before"], end["arena_after"]]
        identity = observed[0]["worker_identity"]
        if set(identity) != {"pid", "start_ticks"} or any(
            type(value) is not int or value < 1 for value in identity.values()
        ):
            raise ValueError("invalid fresh worker process identity")
        origins.append(identity)
        for row in observed:
            if row["worker_identity"] != identity or not row["cache_and_rng_unchanged"]:
                raise ValueError("arena worker identity or cache preservation differs")
            if row["memory"].get("unavailable") or any(
                type(row["memory"].get(key)) is not int or row["memory"][key] <= 0 for key in ("Pss", "Rss")
            ):
                raise ValueError("arena RSS/PSS evidence incomplete")
            if (
                row["rng_sha256"] != observed[0]["rng_sha256"]
                or row["thread_environment"] != report["thread_environment"]
            ):
                raise ValueError("arena worker RNG or thread environment changed")
            validate_cpu_topology(row["cpu_topology"], report["expected_physical_cpus"])
            if row["cpu_topology"] != report["cpu_topology"]:
                raise ValueError("arena worker placement differs")
            info = row["malloc_info"]
            decoded = parse_malloc_info(info["xml"])
            if any(info[key] != value for key, value in decoded.items()):
                raise ValueError("malloc_info summary differs from raw XML")
            if info["requested_arena_max"] != maximum or (maximum == 2 and info["arena_count"] > 2):
                raise ValueError("observed arena policy differs")
            arena_environment({k: v for k, v in info["environment"].items() if v is not None}, maximum, child=True)
            if not row["allocator"]["available"] or row["allocator"]["glibc"] != info["glibc"]:
                raise ValueError("allocator observation unavailable or inconsistent")
            allocator_versions.append(info["glibc"])
        cache_records.append([(row["cache_sha256"], row["cache"]) for row in observed])
        if (
            end["arena_before"]["cache_sha256"] != end["cache_sha256"]
            or end["arena_after"]["cache_sha256"] != end["cache_sha256"]
        ):
            raise ValueError("end-only trim changed live cache")
        launch = json.loads((path / "launch.json").read_text())
        policy = launch["allocator_policy"]
        inherited = policy["inherited"]
        if (
            arena_environment(inherited, maximum) != policy["effective"]
            or policy["set_before_fresh_interpreter"] is not True
        ):
            raise ValueError("allocator policy was not bound before fresh interpreter")
        for row in observed:
            if {key: value for key, value in row["malloc_info"]["environment"].items() if value is not None} != policy[
                "effective"
            ]:
                raise ValueError("worker allocator environment differs from its fresh launch")
        launch_sources.append(launch["script_sha256"])
        launch_environments.append(inherited)
    if origins[0] == origins[1]:
        raise ValueError("arena comparison requires distinct fresh native worker identities")
    if cache_records[0] != cache_records[1] or launch_sources[0] != launch_sources[1]:
        raise ValueError("paired cache tensor bytes/counters or source differ")
    if launch_environments[0] != launch_environments[1] or len(set(allocator_versions)) != 1:
        raise ValueError("paired inherited allocator environment or glibc differs")
    return {
        "protocol": ARENA_PROTOCOL,
        "passed": True,
        "semantic_and_tensor_bytes_exact": True,
        "production_policy_qualified": False,
        "end_to_end_training_speed_claim": False,
        "compared_fields": list(required),
        "causality_requires_matched_cpu_placement": True,
        "inputs": [
            {
                "result_sha256": file_hash(path / "result.json"),
                "malloc_arena_max": report["malloc_arena_max"],
                "processing_timing": report["processing_timing"],
                "replay_seconds_before_end_diagnostic": report["replay_seconds_before_end_diagnostic"],
                "peak_worker_pss_bytes": report["peak_worker_pss_bytes"],
                "elapsed_seconds": report["elapsed_seconds"],
                "repeat": report["repeat"],
            }
            for path, report in zip(paths, reports, strict=True)
        ],
    }


def compare_reports(paths: list[Path]) -> dict[str, Any]:
    """Keep historical comparisons separate from complete matched periodic pairs."""
    if len(paths) < 2:
        raise ValueError("comparison requires at least two completed replays")
    protocols = {json.loads((path / "result.json").read_text()).get("protocol") for path in paths}
    if protocols == {PROTOCOL}:
        return _compare_thread_reports(paths)
    if protocols not in ({PERIODIC_PROTOCOL}, {ARENA_PROTOCOL}):
        raise ValueError("diagnostic protocols must match; end-only and periodic replays cannot be mixed")
    for path in paths:
        try:
            status = json.loads((path / "exit.json").read_text())
        except (OSError, ValueError) as error:
            raise ValueError("periodic comparison requires a clean child exit receipt") from error
        if set(status) != {"exit_code"} or type(status["exit_code"]) is not int or status["exit_code"] != 0:
            raise ValueError("periodic comparison requires a clean child exit receipt")
    return _compare_arena_reports(paths) if protocols == {ARENA_PROTOCOL} else _compare_periodic_reports(paths)


def run_child(args: argparse.Namespace) -> None:
    """Omitting the periodic option retains the original thread/end-only replay."""
    if getattr(args, "malloc_arena_max", None) is not None:
        _run_arena_child(args)
    elif getattr(args, "trim_every_rows", None) is None:
        _run_end_only_child(args)
    else:
        _run_periodic_child(args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    for name in ("config", "model-path", "data-root", "build-receipt", "output"):
        run.add_argument("--" + name, required=True)
    run.add_argument("--thread-policy", choices=("inherited", "blas-one"), required=True)
    run.add_argument("--rows", type=int, default=1000)
    run.add_argument(
        "--malloc-arena-max",
        type=int,
        choices=(0, 2),
        default=None,
        help="Opt-in allocation-policy diagnostic: explicit default0 versus maximum2, fresh process and inherited threads; no periodic trim",
    )
    run.add_argument(
        "--trim-every-rows",
        type=int,
        choices=(0, 128),
        default=None,
        help="Opt-in periodic CPU diagnostic:0 is matched sham,128 trims every128 drained rows; omitted keeps end-only GC/trim",
    )
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--repeat", type=int, default=0)
    run.add_argument("--cache-bytes", type=int, default=2**30)
    run.add_argument("--consumer-delay-seconds", type=float, default=0.0)
    run.add_argument("--max-rss-gib", type=float, default=24.0)
    run.add_argument("--timeout-seconds", type=int, default=3300)
    run.add_argument(
        "--expected-physical-cpus", type=int, help="Refuse wrong effective affinity before importing Torch"
    )
    run.add_argument("--override", action="append", default=[])
    run.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    compare = sub.add_parser("compare")
    compare.add_argument("--run", type=Path, action="append", required=True)
    compare.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "compare":
        write_json(args.output, compare_reports(args.run))
        return
    if args.rows < 8 or args.rows % 8 or args.cache_bytes < 0 or args.repeat < 0 or args.timeout_seconds < 1:
        parser.error("rows must be a positive multiple of 8; cache/repeat nonnegative; timeout positive")
    if (
        not all(math.isfinite(value) and value >= 0 for value in (args.consumer_delay_seconds, args.max_rss_gib))
        or args.max_rss_gib == 0
    ):
        parser.error("delay must be finite/nonnegative and RSS bound positive")
    if args.trim_every_rows is not None:
        if args.thread_policy != "inherited":
            parser.error("periodic trim comparison must retain the inherited thread environment")
        if args.rows < 128:
            parser.error("periodic comparison requires at least128 rows and one drained maintenance boundary")
    allocator_policy = None
    if args.malloc_arena_max is not None:
        if args.trim_every_rows is not None or args.thread_policy != "inherited":
            parser.error("arena comparison requires inherited threads and no periodic trim option")
        inherited = dict(os.environ)
        try:
            effective = arena_environment(inherited, args.malloc_arena_max, child=args.child)
        except ValueError as error:
            parser.error(str(error))
        keys = {"MALLOC_ARENA_MAX", "GLIBC_TUNABLES", "LD_PRELOAD", "PYTHONMALLOC"}
        allocator_policy = {
            "inherited": {key: value for key, value in inherited.items() if key in keys},
            "effective": {key: value for key, value in effective.items() if key in keys},
            "set_before_fresh_interpreter": True,
        }
    if args.child:
        run_child(args)
        if args.trim_every_rows is not None or args.malloc_arena_max is not None:
            shutdown_phase(Path(args.output), "leaving_main")
        return
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    topology = cpu_topology()
    write_json(output / "cpu-topology.json", topology)
    validate_cpu_topology(topology, args.expected_physical_cpus)
    env = policy_environment(dict(os.environ), args.thread_policy)
    if args.malloc_arena_max is not None:
        env = arena_environment(env, args.malloc_arena_max)
    env.update(SFT_MODEL_PATH=args.model_path, SFT_DATA_ROOT=args.data_root, PYTHONHASHSEED=str(args.seed))
    write_json(
        output / "launch.json",
        {
            "argv": sys.argv,
            "script_sha256": file_hash(Path(__file__)),
            "inherited_thread_environment": {key: os.environ.get(key) for key in THREAD_KEYS},
            "effective_thread_environment": {key: env.get(key) for key in THREAD_KEYS},
            **({"allocator_policy": allocator_policy} if allocator_policy is not None else {}),
        },
    )
    with (output / "child.log").open("x") as log:
        child = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--child"],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = child.wait(timeout=args.timeout_seconds)
        finally:
            try:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=10)
            except ProcessLookupError:
                pass
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        write_json(output / "exit.json", {"exit_code": code})
    if code:
        raise SystemExit(code)


if __name__ == "__main__":
    main()
