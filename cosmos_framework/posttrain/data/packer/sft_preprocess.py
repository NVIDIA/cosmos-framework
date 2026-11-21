# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""Bounded deterministic CPU preprocessing over one optimizer-update window.

The driver owns data order and cursor advancement. Spawned CPU workers receive
only the already selected rows of the current update, never a dataset/cursor or
model. Cache contents are disposable: misses and fresh-process resume have the
same semantics as hits. Video entries retain processor metadata and timestamps.
"""

from __future__ import annotations

import copy
import multiprocessing
import os
import random
import sys
import time
from collections import OrderedDict, deque
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cosmos_framework.posttrain.data.datasets.sft import sha256_file
from cosmos_framework.posttrain.data.packer.sft import QwenSFTDataPacker, SFTPolicyInput, build_sft_packer
from cosmos_framework.posttrain.schema import SFTTarget
from cosmos_framework.posttrain.utils.identity import canonical_json_sha256, project_typed_json_value
from cosmos_framework.posttrain.utils.training_profile import profile_cpu_range

MEDIA_OWNER_CAPACITY = 1024


def _cpu_bytes(value: Any) -> int:
    """Conservative retained-size accounting; reject GPU/autograd-owned cache values."""
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu" or value.requires_grad:
            raise ValueError("SFT preprocessing caches only detached CPU tensors")
        # deepcopy preserves a view's entire backing storage, even for a tiny
        # slice. Conservatively count it for each view rather than underbound
        # retained host memory using only the visible tensor shape.
        return sys.getsizeof(value) + value.untyped_storage().nbytes()
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("object arrays are not valid preprocessing cache entries")
        return sys.getsizeof(value) + value.nbytes
    if isinstance(value, Mapping):
        return sys.getsizeof(value) + sum(_cpu_bytes(key) + _cpu_bytes(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return sys.getsizeof(value) + sum(map(_cpu_bytes, value))
    if is_dataclass(value) and not isinstance(value, type):
        return sys.getsizeof(value) + sum(_cpu_bytes(getattr(value, field.name)) for field in fields(value))
    if value is None or isinstance(value, (str, bytes, int, float, bool, np.generic)):
        return sys.getsizeof(value)
    raise TypeError(f"unsupported CPU preprocessing cache value {type(value).__name__}")


class _ByteLRU:
    def __init__(self, limit: int) -> None:
        self.limit, self.bytes, self.hits, self.misses = limit, 0, 0, 0
        self.entries: OrderedDict[str, tuple[Any, int]] = OrderedDict()

    def get(self, key: str) -> Any | None:
        if key not in self.entries:
            self.misses += 1
            return None
        self.hits += 1
        value, _ = self.entries[key]
        self.entries.move_to_end(key)
        return copy.deepcopy(value)

    def put(self, key: str, value: Any) -> None:
        size = _cpu_bytes(value)
        if size > self.limit:
            return
        if key in self.entries:
            self.bytes -= self.entries.pop(key)[1]
        while self.entries and self.bytes + size > self.limit:
            self.bytes -= self.entries.popitem(last=False)[1][1]
        self.entries[key] = (copy.deepcopy(value), size)
        self.bytes += size

    def stats(self) -> dict[str, int]:
        return {"bytes": self.bytes, "limit": self.limit, "hits": self.hits, "misses": self.misses}


class _MediaIdentity:
    def __init__(self, media_hashes: Mapping[str, str]) -> None:
        self.hashes = {str(Path(path).resolve()): digest for path, digest in media_hashes.items()}
        if any(len(value) != 64 or any(c not in "0123456789abcdef" for c in value) for value in self.hashes.values()):
            raise ValueError("preprocessing media identities must be SHA256 digests")
        self.verified: dict[str, tuple[int, int, int, int, int]] = {}

    def reference(self, value: Any, *, verify: bool) -> dict[str, str]:
        if not isinstance(value, str) or "://" in value:
            raise ValueError("cached preprocessing requires verified local media paths")
        path = Path(value).resolve()
        digest = self.hashes.get(str(path))
        if digest is None:
            raise ValueError(f"media is absent from the verified dataset manifest: {path.name}")
        stat = path.stat()
        signature = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        old = self.verified.get(str(path))
        if old is not None and signature != old:
            raise ValueError(f"verified preprocessing media changed: {path.name}")
        if verify and old is None:
            actual = sha256_file(str(path))
            after = path.stat()
            after_signature = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
            if actual != digest or after_signature != signature:
                raise ValueError(f"preprocessing media digest differs from dataset: {path.name}")
            self.verified[str(path)] = signature
        return {"sha256": digest}

    def target(self, target: SFTTarget, *, verify: bool) -> dict[str, Any]:
        if not isinstance(target, SFTTarget):
            raise TypeError("preprocessing requires SFTTarget rows")
        result = asdict(target)
        for message in result["messages"]:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict) or part.get("type") not in {"image", "video"}:
                    continue
                kind = part["type"]
                candidates = [key for key in (kind, "url", "path") if key in part]
                if len(candidates) != 1:
                    raise ValueError("cached media requires one unambiguous local file reference")
                key = candidates[0]
                part[key] = self.reference(part[key], verify=verify)
        return result


def _deterministic_call(fn, *args, **kwargs):
    """Reject stochastic preprocessing instead of changing its RNG semantics on cache hits."""
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    try:
        result = fn(*args, **kwargs)
        current_numpy = np.random.get_state()
        changed = (
            random.getstate() != python_state
            or current_numpy[0] != numpy_state[0]
            or not np.array_equal(current_numpy[1], numpy_state[1])
            or current_numpy[2:] != numpy_state[2:]
            or not torch.equal(torch.get_rng_state(), torch_state)
        )
        if changed:
            raise ValueError("cached/prefetched SFT preprocessing must not consume RNG state")
        return result
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)


class _CachedVideoProcessor:
    """Cache the complete processor call before decoding, preserving mutable metadata."""

    def __init__(self, processor: Any, cache: _ByteLRU, media: _MediaIdentity, identity: str) -> None:
        self._processor, self._cache, self._media, self._identity = processor, cache, media, identity

    def __getattr__(self, name: str) -> Any:
        return getattr(self._processor, name)

    def __call__(self, *args, **kwargs):
        if args or "videos" not in kwargs:
            raise ValueError("cached Qwen video preprocessing requires keyword videos")
        videos = kwargs["videos"]
        if not isinstance(videos, (list, tuple)) or not videos:
            raise ValueError("cached Qwen video preprocessing requires a nonempty local video list")
        values = {**kwargs, "videos": [self._media.reference(path, verify=True) for path in videos]}
        key = canonical_json_sha256(
            {"processor": self._identity, "call": project_typed_json_value(values, context="video cache settings")},
            context="video preprocessing cache key",
        )
        with profile_cpu_range("sft.video_cache_lookup_copy"):
            cached = self._cache.get(key)
        if cached is not None:
            return cached
        result = _deterministic_call(self._processor, **kwargs)
        self._cache.put(key, result)
        return result


class _PreprocessingCache:
    def __init__(self, packer: Any, *, cache_bytes: int, media_hashes: Mapping[str, str]) -> None:
        self.packer = packer
        self.identity = canonical_json_sha256(packer.validation_semantics_identity(), context="preprocessing identity")
        self.media = _MediaIdentity(media_hashes)
        self.samples = _ByteLRU(cache_bytes // 4)
        self.videos = _ByteLRU(cache_bytes - cache_bytes // 4)
        processor = getattr(packer, "processor", None)
        if processor is not None and getattr(processor, "video_processor", None) is not None:
            processor.video_processor = _CachedVideoProcessor(
                processor.video_processor, self.videos, self.media, self.identity
            )
        self._settings = self._runtime_settings()

    def _runtime_settings(self) -> str:
        # Tokenizer vocabulary bytes were bound by the initial processor
        # identity. These small mutable settings must also remain fixed while
        # serving hits; do not re-hash the entire vocabulary for every row.
        processor, tokenizer = getattr(self.packer, "processor", None), getattr(self.packer, "tokenizer", None)
        media = {}
        for name in ("image_processor", "video_processor"):
            component = getattr(processor, name, None)
            if callable(getattr(component, "to_dict", None)):
                media[name] = component.to_dict()
        return canonical_json_sha256(
            project_typed_json_value(
                {
                    "max_length": getattr(self.packer, "max_length", None),
                    "mask_think_opener": getattr(self.packer, "mask_think_opener", None),
                    "processor_kwargs": getattr(self.packer, "processor_kwargs", None),
                    "tokenizer_template": getattr(tokenizer, "chat_template", None),
                    "processor_template": getattr(processor, "chat_template", None),
                    "media": media,
                },
                context="preprocessing runtime settings",
            ),
            context="preprocessing runtime settings",
        )

    def key(self, target: SFTTarget, *, verify: bool = False) -> str:
        if self._runtime_settings() != self._settings:
            raise ValueError("preprocessing processor settings changed after its identity was bound")
        return canonical_json_sha256(
            {"processor": self.identity, "sample": self.media.target(target, verify=verify)}, context="SFT cache key"
        )

    def process(self, target: SFTTarget) -> SFTPolicyInput:
        key = self.key(target, verify=True)
        with profile_cpu_range("sft.sample_cache_lookup_copy"):
            cached = self.samples.get(key)
        if cached is not None:
            return cached
        result = _deterministic_call(self.packer.sft_process_sample, target)
        self.samples.put(key, result)
        return result

    def stats(self) -> dict[str, Any]:
        return {"samples": self.samples.stats(), "videos": self.videos.stats()}


@dataclass(frozen=True)
class _WorkerSpec:
    name_or_path: str
    source_cwd: str
    max_length: int
    processor_kwargs: dict[str, Any]
    mask_think_opener: bool
    identity: dict[str, Any]
    cache_bytes: int
    media_hashes: dict[str, str]
    cpu_affinity: tuple[int, ...] = ()


_WORKER_CACHE: _PreprocessingCache | None = None
_WORKER_PROCESSING_CALLS = 0
_WORKER_PROCESSING_SECONDS = 0.0


def _restrict_worker_threads(cpus: tuple[int, ...]) -> None:
    """Restrict existing import-time threads and their future descendants.

    FFmpeg may still create several threads. This bounds their usable CPUs, not
    their number or decoder heap size. The default one-worker path is unchanged.
    """
    if not cpus or not set(cpus) <= os.sched_getaffinity(0):
        raise ValueError("preprocessing worker CPU partition is outside its inherited affinity")
    tasks = Path("/proc/self/task")
    for _ in range(4):
        for task in tasks.iterdir():
            try:
                os.sched_setaffinity(int(task.name), cpus)
            except ProcessLookupError:
                pass
        try:
            if all(os.sched_getaffinity(int(task.name)) == set(cpus) for task in tasks.iterdir()):
                return
        except ProcessLookupError:
            pass
    raise RuntimeError("could not bind all preprocessing worker threads to their CPU partition")


def _initialize_worker(spec: _WorkerSpec) -> None:
    global _WORKER_CACHE, _WORKER_PROCESSING_CALLS, _WORKER_PROCESSING_SECONDS
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if spec.cpu_affinity:
        _restrict_worker_threads(spec.cpu_affinity)
    torch.set_num_threads(1)
    if torch.cuda.is_initialized():
        raise RuntimeError("spawned SFT preprocessing workers must never initialize CUDA")
    # Tokenizer identities include source spelling. Resolving a symlink or a
    # filesystem alias here changes the contract even when file bytes match.
    # The pool starts lazily, so retain the source directory for relative names
    # if the parent changes cwd after constructing the pipeline.
    os.chdir(spec.source_cwd)
    packer = build_sft_packer(
        spec.name_or_path,
        max_length=spec.max_length,
        processor_kwargs=spec.processor_kwargs,
        mask_think_opener=spec.mask_think_opener,
        trust_remote_code=False,
    )
    if packer.validation_semantics_identity() != spec.identity:
        raise ValueError("spawned preprocessing worker rebuilt a different processor contract")
    _WORKER_CACHE = _PreprocessingCache(packer, cache_bytes=spec.cache_bytes, media_hashes=spec.media_hashes)
    _WORKER_PROCESSING_CALLS, _WORKER_PROCESSING_SECONDS = 0, 0.0


def _process_worker(target: SFTTarget) -> tuple[SFTPolicyInput, dict[str, Any]]:
    global _WORKER_PROCESSING_CALLS, _WORKER_PROCESSING_SECONDS
    if _WORKER_CACHE is None:
        raise RuntimeError("preprocessing worker was not initialized")
    started = time.perf_counter()
    with profile_cpu_range("sft.worker_prepare"):
        result = _WORKER_CACHE.process(target)
    elapsed = time.perf_counter() - started
    if torch.cuda.is_initialized():
        raise RuntimeError("SFT preprocessing initialized CUDA in the CPU worker")
    _WORKER_PROCESSING_CALLS += 1
    _WORKER_PROCESSING_SECONDS += elapsed
    return result, {
        **_WORKER_CACHE.stats(),
        "timing": {"processing_calls": _WORKER_PROCESSING_CALLS, "processing_seconds": _WORKER_PROCESSING_SECONDS},
    }


class SFTPreprocessingPipeline:
    """A transparent packer facade with ordered, bounded intra-update prefetch."""

    def __init__(
        self, packer: QwenSFTDataPacker, spec: _WorkerSpec, *, prefetch: bool, max_pending: int, workers: int = 1
    ) -> None:
        self._packer, self._spec, self._max_pending = packer, spec, max_pending
        self._cache = _PreprocessingCache(
            packer, cache_bytes=0 if prefetch else spec.cache_bytes, media_hashes=spec.media_hashes
        )
        worker_specs = [spec]
        if workers != 1:
            if not sys.platform.startswith("linux") or not hasattr(os, "sched_getaffinity"):
                raise ValueError("parallel SFT preprocessing requires Linux CPU affinity support")
            cpus = sorted(os.sched_getaffinity(0))
            if len(cpus) < workers:
                raise ValueError("parallel SFT preprocessing requires at least one allowed CPU per worker")
            worker_specs = [
                replace(
                    spec,
                    cache_bytes=spec.cache_bytes // workers + (index < spec.cache_bytes % workers),
                    cpu_affinity=tuple(cpus[index::workers]),
                )
                for index in range(workers)
            ]
        self.worker_cpu_affinities = tuple(item.cpu_affinity for item in worker_specs)
        self._executors = [
            ProcessPoolExecutor(
                max_workers=1,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_initialize_worker,
                initargs=(item,),
            )
            for item in worker_specs
            if prefetch
        ]
        # Preserve the original single-worker diagnostic interface.
        self._executor = self._executors[0] if self._executors else None
        self._next_worker = 0
        # Ownership is disposable, bounded execution state, like the tensor
        # caches. Pin owners within one update even if the persistent LRU evicts
        # them; a later window may reassign an evicted key. Any resulting cache
        # miss changes execution cost only, never the prepared sample.
        self._media_owners: OrderedDict[str, int] = OrderedDict()
        self._window_owners: dict[str, int] = {}
        self._window_assignments = [0] * workers
        self._worker_stats = [
            {
                "samples": _ByteLRU(item.cache_bytes // 4).stats(),
                "videos": _ByteLRU(item.cache_bytes - item.cache_bytes // 4).stats(),
                "timing": {"processing_calls": 0, "processing_seconds": 0.0},
            }
            for item in worker_specs
        ]
        self._pending: deque = deque()
        self._window_active, self._closed, self._failed = False, False, False
        # Disposable execution counters, intentionally absent from cache keys,
        # processor identity and checkpoints. Worker processing includes cache
        # hits but excludes IPC; parent waiting is Future.result only. These
        # durations overlap and must never be added to infer training wall time.
        self._processing_calls, self._processing_seconds = 0, 0.0
        self._wait_calls, self._wait_seconds = 0, 0.0
        self.preprocessing_stats: dict[str, Any] = {
            **self._cache.stats(),
            "timing": {"processing_calls": 0, "processing_seconds": 0.0, "wait_calls": 0, "wait_seconds": 0.0},
        }

    def __getattr__(self, name: str) -> Any:
        return getattr(self._packer, name)

    def validation_semantics_identity(self) -> dict[str, Any]:
        return self._packer.validation_semantics_identity()

    @property
    def semantic_packer_class(self) -> str:
        """The cache changes execution only; masking and processor code remain native."""
        return f"{type(self._packer).__module__}.{type(self._packer).__qualname__}"

    def prepare_window(self, targets: list[SFTTarget]) -> None:
        if self._closed or self._failed or self._window_active:
            raise RuntimeError("cannot start a preprocessing window while closed or another window is active")
        if not targets or len(targets) > self._max_pending:
            raise ValueError("preprocessing window must fit its configured pending-row bound")
        if self._executor is None:
            raise ValueError("prepare_window requires preprocessing_prefetch=True")
        records = [(self._cache.key(target), copy.deepcopy(target)) for target in targets]
        self._clear_window_routing()
        self._window_active = True
        try:
            for key, target in records:
                worker, future = self._submit(target)
                self._pending.append((key, worker, future))
        except BaseException:
            self._failed = True
            self._clear_window_routing()
            raise

    def _clear_window_routing(self) -> None:
        self._window_owners.clear()
        self._window_assignments[:] = [0] * len(self._window_assignments)

    def _submit(self, target: SFTTarget):
        worker = self._next_worker % len(self._executors)
        if len(self._executors) > 1:
            # Related questions on identical media share a worker's video cache.
            # Balance unseen media by submitted row count within this update,
            # without moving already-owned media or changing consumption order.
            # Outside an update, new/no-media rows retain round-robin routing.
            if self._window_active:
                worker = min(range(len(self._executors)), key=lambda i: (self._window_assignments[i], i))
            content = self._cache.media.target(target, verify=False)
            media = [
                part[key]["sha256"]
                for message in content["messages"]
                if isinstance(message.get("content"), list)
                for part in message["content"]
                if isinstance(part, dict) and part.get("type") in {"image", "video"}
                for key in (part["type"], "url", "path")
                if key in part
            ]
            if media:
                key = canonical_json_sha256(media, context="preprocessing worker media")
                worker = self._window_owners.get(key, self._media_owners.get(key, worker))
                if self._window_active:
                    self._window_owners[key] = worker
                self._media_owners[key] = worker
                self._media_owners.move_to_end(key)
                while len(self._media_owners) > MEDIA_OWNER_CAPACITY:
                    self._media_owners.popitem(last=False)
            if self._window_active:
                self._window_assignments[worker] += 1
        future = self._executors[worker].submit(_process_worker, target)
        self._next_worker += 1
        return worker, future

    def sft_process_sample(self, target: SFTTarget) -> SFTPolicyInput:
        if self._closed or self._failed:
            raise RuntimeError("SFT preprocessing pipeline is closed or failed")
        key = self._cache.key(target)
        if self._window_active:
            if not self._pending or self._pending[0][0] != key:
                raise ValueError("SFT preprocessing must consume exactly the prepared row order")
            _, worker, future = self._pending.popleft()
            return self._consume_future(future, worker)
        if self._executor is not None:
            try:
                worker, future = self._submit(copy.deepcopy(target))
                return self._consume_future(future, worker)
            except BaseException:
                self._failed = True
                self._clear_window_routing()
                raise
        started = time.perf_counter()
        result = self._cache.process(target)
        self._processing_seconds += time.perf_counter() - started
        self._processing_calls += 1
        self.preprocessing_stats = {
            **self._cache.stats(),
            "timing": {
                "processing_calls": self._processing_calls,
                "processing_seconds": self._processing_seconds,
                "wait_calls": 0,
                "wait_seconds": 0.0,
            },
        }
        return result

    def _consume_future(self, future, worker: int) -> SFTPolicyInput:
        try:
            started = time.perf_counter()
            with profile_cpu_range("sft.prefetch_wait"):
                sample, stats = future.result(timeout=300)
            self._worker_stats[worker] = stats
            # A reply contains that worker's cumulative counters. Replace its
            # last observation and sum workers, never sum successive replies.
            self.preprocessing_stats = {
                kind: {name: sum(item[kind][name] for item in self._worker_stats) for name in stats[kind]}
                for kind in ("samples", "videos", "timing")
            }
            self._wait_seconds += time.perf_counter() - started
            self._wait_calls += 1
            self.preprocessing_stats["timing"].update(wait_calls=self._wait_calls, wait_seconds=self._wait_seconds)
            _cpu_bytes(sample)
            return sample
        except BaseException:
            self._failed = True
            self._clear_window_routing()
            raise

    def finish_window(self) -> None:
        if self._failed or not self._window_active or self._pending:
            raise RuntimeError("preprocessing window must be completely consumed before an optimizer boundary")
        self._window_active = False
        self._clear_window_routing()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._clear_window_routing()
        self._media_owners.clear()
        for _, _, future in self._pending:
            future.cancel()
        self._pending.clear()
        if self._executors:
            # Python 3.10--3.12 lack terminate_workers(). Capture the owned
            # spawned processes before shutdown clears this implementation
            # detail, and bound decoder failure cleanup instead of blocking exit.
            processes = [process for executor in self._executors for process in (executor._processes or {}).values()]
            for executor in self._executors:
                executor.shutdown(wait=False, cancel_futures=True)
            deadline = time.monotonic() + 5
            for process in processes:
                process.join(timeout=max(0, deadline - time.monotonic()))
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=2)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=2)


def build_sft_preprocessing(
    packer: QwenSFTDataPacker,
    *,
    name_or_path: str,
    cache_bytes: int,
    prefetch: bool,
    media_hashes: Mapping[str, str],
    max_pending: int,
    workers: int = 1,
) -> SFTPreprocessingPipeline:
    if not isinstance(packer, QwenSFTDataPacker):
        raise TypeError("bounded preprocessing currently supports the native Qwen SFT packer")
    if isinstance(cache_bytes, bool) or not isinstance(cache_bytes, int) or cache_bytes < 0:
        raise ValueError("preprocessing cache bytes must be a nonnegative integer")
    if (
        not isinstance(prefetch, bool)
        or isinstance(max_pending, bool)
        or not isinstance(max_pending, int)
        or max_pending < 1
    ):
        raise ValueError("preprocessing requires a bool prefetch flag and positive pending-row bound")
    if type(workers) is not int or workers not in (1, 2):
        raise ValueError("preprocessing workers supports only 1 or 2")
    if workers != 1 and (not prefetch or max_pending < workers):
        raise ValueError("two preprocessing workers require prefetch and a pending-row bound of at least two")
    if not Path(name_or_path).is_dir():
        raise ValueError("preprocessing workers require a prepared local processor snapshot")
    spec = _WorkerSpec(
        str(name_or_path),
        os.getcwd(),
        packer.max_length,
        copy.deepcopy(packer.processor_kwargs),
        packer.mask_think_opener,
        copy.deepcopy(packer.validation_semantics_identity()),
        cache_bytes,
        dict(media_hashes),
    )
    return SFTPreprocessingPipeline(packer, spec, prefetch=prefetch, max_pending=max_pending, workers=workers)
