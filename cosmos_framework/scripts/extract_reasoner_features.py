# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Build an immutable offline Reasoner K/V cache for Nano generator SFT.

Each torchrun rank owns a deterministic slice of the SFT metadata, loads only
the frozen Reasoner tower, and writes resumable rank-local safetensors shards.
Rank zero publishes the cache only after every rank has completed successfully.

Example::

    torchrun --nproc-per-node=8 -m cosmos_framework.scripts.extract_reasoner_features \\
      --sft-toml examples/toml/sft_config/vision_sft_nano.toml \\
      --checkpoint /shared/checkpoints/iter_000000100 \\
      --checkpoint-source regular \\
      --output /shared/caches/nano-sft-reasoner-kv \\
      --reasoner-fingerprint <reasoner-subtree-content-digest> \\
      --tokenizer-fingerprint <tokenizer-content-digest> \\
      --framing-fingerprint <dataset-and-framing-content-digest> \\
      -- model.config.compile.enabled=false

The current MVP supports local/shared-POSIX DCP checkpoints and cache output,
Nano's standard single-caption two-way-attention recipe, BF16, and CP=1.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import tempfile
import time
import traceback
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml
from cosmos_framework.data.generator.local_datasets.sft_dataset import (
    SFTDataset,
    get_sft_dataset,
    sft_metadata_sort_key,
)
from cosmos_framework.data.generator.local_datasets.sft_reasoner_documents import (
    iter_sft_reasoner_documents,
)
from cosmos_framework.data.generator.sequence_packing.modalities import add_special_tokens, compute_text_split_length
from cosmos_framework.model.generator.reasoner_feature_cache import (
    IncrementalReasonerFeatureCacheWriter,
    OfflineReasonerFeatureProvider,
    ReasonerFeatureCacheEntry,
    ReasonerFeatureCacheIdentity,
    build_reasoner_feature_request_from_text_tokens,
    finalize_incremental_reasoner_feature_cache,
)
from cosmos_framework.model.generator.reasoner_features import (
    ReasonerFeatureRequest,
)
from cosmos_framework.model.generator.reasoner_runtime import (
    ReasonerFeatureRuntime,
    ReasonerRuntimeSpec,
    prepare_reasoner_model_config,
)
from cosmos_framework.utils.lazy_config import instantiate as lazy_instantiate

_NULL_PROMPT_KEY = "__null__"
_RANK_STATS_FILE = "extraction.stats.json"
_RANK_FAILURE_FILE = "extraction.failure.json"


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    run_id: str


@dataclass
class ExtractionStats:
    rank: int
    assigned_documents: int = 0
    cache_hits: int = 0
    written_records: int = 0
    extracted_tokens: int = 0
    reasoner_construction_seconds: float = 0.0
    checkpoint_load_seconds: float = 0.0
    extraction_seconds: float = 0.0
    peak_allocated_bytes: int = 0
    peak_reserved_bytes: int = 0


def _target_matches_get_sft_dataset(target: object) -> bool:
    if target is get_sft_dataset:
        return True
    if isinstance(target, str):
        return target.rsplit(".", 1)[-1] == "get_sft_dataset"
    return getattr(target, "__name__", None) == "get_sft_dataset"


def _find_sft_dataset_config(root: object) -> object:
    """Find exactly one lazy ``get_sft_dataset`` node below a dataloader config."""

    matches: list[object] = []
    visited: set[int] = set()

    def visit(value: object) -> None:
        value_id = id(value)
        if value_id in visited:
            return
        visited.add(value_id)
        if isinstance(value, Mapping):
            target = value.get("_target_")
            if _target_matches_get_sft_dataset(target):
                matches.append(value)
                return
            for child in value.values():
                visit(child)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for child in value:
                visit(child)

    visit(root)
    if len(matches) != 1:
        raise ValueError(
            "Reasoner extraction requires exactly one get_sft_dataset node under dataloader_train; "
            f"found {len(matches)}"
        )
    return matches[0]


def _instantiate_sft_dataset(config: object) -> SFTDataset:
    dataloader_config = getattr(config, "dataloader_train")
    dataset_config = _find_sft_dataset_config(dataloader_config)
    dataset = lazy_instantiate(dataset_config)
    if not isinstance(dataset, SFTDataset):
        raise TypeError(f"Expected get_sft_dataset to return SFTDataset, got {type(dataset).__name__}")
    return dataset


def _validate_extraction_config(config: object) -> None:
    model = getattr(getattr(config, "model"), "config")
    if model.joint_attn_implementation != "two_way":
        raise ValueError("Reasoner extraction currently requires joint_attn_implementation='two_way'")
    if model.parallelism.context_parallel_shard_degree != 1:
        raise ValueError("Reasoner extraction currently requires context_parallel_shard_degree=1")
    if model.video_temporal_causal:
        raise ValueError("Reasoner extraction does not support video_temporal_causal training")
    if model.causal_training_strategy != "none":
        raise ValueError("Reasoner extraction currently requires causal_training_strategy='none'")
    if model.diffusion_expert_config.vision_temporal_position_mode != "latent_index":
        raise ValueError("Reasoner extraction currently supports vision_temporal_position_mode='latent_index' only")

    model_instance = model.vlm_config.model_instance
    if model_instance is None:
        raise ValueError("Reasoner extraction requires model.config.vlm_config.model_instance")
    target = model_instance.get("_target_")
    target_name = target if isinstance(target, str) else getattr(target, "__name__", repr(target))
    if "Qwen3VLTextForCausalLM" not in target_name:
        raise ValueError(f"Reasoner extraction MVP supports the Nano Qwen3VLTextForCausalLM target, got {target_name}")


# Backward-compatible private alias retained for tests and downstream scripts.
_prepare_reasoner_model_config = prepare_reasoner_model_config


def _special_tokens(dataset: SFTDataset) -> dict[str, int]:
    tokenizer, special_tokens = add_special_tokens(dataset.vlm_tokenizer)
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("SFT tokenizer must define eos_token_id")
    return {**special_tokens, "eos_token_id": int(eos_token_id)}


def _extraction_max_total_tokens(dataset: SFTDataset, special_tokens: Mapping[str, int]) -> int:
    """Derive the largest framed request from the dataset's caption limit."""

    max_caption_tokens = dataset.max_caption_tokens
    if not isinstance(max_caption_tokens, int) or isinstance(max_caption_tokens, bool) or max_caption_tokens <= 0:
        raise ValueError(f"SFT max_caption_tokens must be a positive integer, got {max_caption_tokens!r}")
    return compute_text_split_length(max_caption_tokens, dict(special_tokens), has_generation=True)


def _rank_dataset(dataset: SFTDataset, *, rank: int, world_size: int) -> SFTDataset:
    """Shallow-copy a dataset and assign it a deterministic video-level slice."""

    result = copy.copy(dataset)
    ordered = sorted(dataset.metadata, key=sft_metadata_sort_key)
    result.metadata = ordered[rank::world_size]
    return result


def _request_owner(request: ReasonerFeatureRequest, world_size: int) -> int:
    digest = hashlib.sha256(f"{request.sample_key}\0{request.fingerprint}".encode()).digest()
    return int.from_bytes(digest[:8], byteorder="big") % world_size


def _iter_rank_requests(
    dataset: SFTDataset,
    *,
    identity: ReasonerFeatureCacheIdentity,
    rank: int,
    world_size: int,
    use_float_positions: bool,
    max_documents_per_rank: int | None,
    special_tokens: Mapping[str, int] | None = None,
) -> Iterator[ReasonerFeatureRequest]:
    """Yield framed requests for one deterministic metadata slice.

    The standard CFG mode (drop the complete caption) gets one shared null
    record. Its content-based owner is deterministic, so no cross-rank duplicate
    is published. Other requests stay attached to their diagnostic sample key.
    """

    tokens = dict(special_tokens) if special_tokens is not None else _special_tokens(dataset)
    produced = 0
    if dataset.cfg_dropout_rate > 0 and not dataset.cfg_dropout_keep_metadata:
        null_text_ids, _ = dataset._tokenize_caption("")
        null_request = build_reasoner_feature_request_from_text_tokens(
            sample_key=_NULL_PROMPT_KEY,
            text_ids=null_text_ids,
            special_tokens=tokens,
            use_float_positions=use_float_positions,
            identity=identity,
        )
        if _request_owner(null_request, world_size) == rank:
            yield null_request
            produced += 1
            if max_documents_per_rank is not None and produced >= max_documents_per_rank:
                return

    rank_dataset = _rank_dataset(dataset, rank=rank, world_size=world_size)
    for document in iter_sft_reasoner_documents(rank_dataset):
        if document.caption == "" and dataset.cfg_dropout_rate > 0 and not dataset.cfg_dropout_keep_metadata:
            continue
        yield build_reasoner_feature_request_from_text_tokens(
            sample_key=document.sample_key,
            text_ids=document.text_token_ids,
            special_tokens=tokens,
            use_float_positions=use_float_positions,
            identity=identity,
        )
        produced += 1
        if max_documents_per_rank is not None and produced >= max_documents_per_rank:
            return


def _initialize_distributed(coordination_run_id: str | None = None) -> DistributedContext:
    """Resolve torchrun worker identity without creating a process group.

    Extraction workers are intentionally independent: each owns a complete
    Reasoner replica, a disjoint metadata slice, and rank-local staging files.
    Avoiding NCCL collectives means one worker can fail without stranding peers
    in a mismatched or hours-long collective; torchrun remains the fail-fast
    process supervisor.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("Reasoner feature extraction requires CUDA")
    topology_names = ("WORLD_SIZE", "RANK", "LOCAL_RANK", "LOCAL_WORLD_SIZE")
    present_topology = {name for name in topology_names if name in os.environ}
    if not present_topology:
        world_size, rank, local_rank, local_world_size = 1, 0, 0, 1
    elif present_topology != set(topology_names):
        missing = sorted(set(topology_names) - present_topology)
        raise ValueError(
            "Incomplete torchrun topology environment; either set none for a direct single-worker run "
            f"or set all of {topology_names}. Missing: {missing}"
        )
    else:
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    if world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError(f"Invalid torchrun rank topology: rank={rank}, world_size={world_size}")
    if local_world_size <= 0 or not 0 <= local_rank < local_world_size:
        raise ValueError(
            f"Invalid local torchrun topology: local_rank={local_rank}, local_world_size={local_world_size}"
        )
    if local_rank < 0 or local_rank >= torch.cuda.device_count():
        raise ValueError(f"LOCAL_RANK={local_rank} is outside the {torch.cuda.device_count()} visible CUDA devices")
    torch.cuda.set_device(local_rank)
    restart_count = os.environ.get("TORCHELASTIC_RESTART_COUNT", "0")
    if coordination_run_id is not None:
        # A restart is a new coordination attempt. Durable shards/completion
        # remain resumable, while stale failure/status files from the previous
        # attempt cannot poison the replacement workers.
        run_id = f"{coordination_run_id}:{restart_count}"
    else:
        if world_size > local_world_size:
            raise ValueError("Multi-node extraction requires an explicit --coordination-run-id shared by every node")
        elastic_run_id = os.environ.get("TORCHELASTIC_RUN_ID", "manual")
        # All workers launched by one local torchrun agent share a parent PID;
        # adding it prevents stale status from a later default-rdzv invocation.
        # A direct single-worker invocation uses its own PID for the same reason.
        launch_pid = os.getpid() if world_size == 1 else os.getppid()
        run_id = f"{elastic_run_id}:{restart_count}:{launch_pid}"
    return DistributedContext(rank, world_size, local_rank, torch.device("cuda", local_rank), run_id)


def _extract_rank(
    *,
    runtime: ReasonerFeatureRuntime,
    requests: Iterator[ReasonerFeatureRequest],
    writer: IncrementalReasonerFeatureCacheWriter,
    context: DistributedContext,
) -> ExtractionStats:
    stats = ExtractionStats(rank=context.rank)
    torch.cuda.synchronize(context.device)
    started = time.perf_counter()
    for request in requests:
        stats.assigned_documents += 1
        if writer.contains(request.sample_key, request.fingerprint):
            stats.cache_hits += 1
            continue
        features = runtime.execute([request])
        stats.extracted_tokens += request.token_ids.numel()
        if not writer.append(ReasonerFeatureCacheEntry(request.sample_key, features)):
            raise RuntimeError(f"Writer unexpectedly rejected newly extracted record {request.sample_key!r}")
        stats.written_records += 1
        del features
    # Commit resumable data first. The caller writes the current-run stats and
    # only then creates rank.complete as the last successful worker action.
    writer.flush()
    torch.cuda.synchronize(context.device)
    stats.extraction_seconds = time.perf_counter() - started
    stats.peak_allocated_bytes = torch.cuda.max_memory_allocated(context.device)
    stats.peak_reserved_bytes = torch.cuda.max_memory_reserved(context.device)
    return stats


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _rank_status_path(staging_root: Path, rank: int, name: str) -> Path:
    return staging_root / f"rank-{rank:05d}" / name


def _write_rank_stats(
    writer: IncrementalReasonerFeatureCacheWriter,
    stats: ExtractionStats,
    *,
    context: DistributedContext,
    publishable: bool,
) -> None:
    _atomic_write_json(
        writer.rank_dir / _RANK_STATS_FILE,
        {
            "run_id": context.run_id,
            "world_size": context.world_size,
            "publishable": publishable,
            "stats": asdict(stats),
        },
    )


def _commit_rank_result(
    writer: IncrementalReasonerFeatureCacheWriter,
    stats: ExtractionStats,
    *,
    context: DistributedContext,
    publishable: bool,
) -> None:
    """Atomically record stats, then optionally make the rank publishable."""
    _write_rank_stats(writer, stats, context=context, publishable=publishable)
    if publishable:
        # rank.complete is deliberately the final fallible worker commit. Rank
        # zero waits for both this marker and the current-run stats record.
        writer.finalize()


def _read_current_rank_stats(path: Path, *, context: DistributedContext) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    if not isinstance(payload, dict) or payload.get("run_id") != context.run_id:
        return None
    if payload.get("world_size") != context.world_size or payload.get("publishable") is not True:
        return None
    stats = payload.get("stats")
    if not isinstance(stats, dict):
        raise ValueError(f"Invalid extraction stats payload: {path}")
    return stats


def _wait_for_rank_completion(
    staging_root: Path,
    *,
    context: DistributedContext,
    timeout_seconds: float,
) -> list[dict[str, Any]]:
    """Wait through shared POSIX markers, never through a GPU collective."""
    deadline = time.monotonic() + timeout_seconds
    while True:
        failures: list[tuple[int, str]] = []
        stats: list[dict[str, Any]] = []
        missing: list[int] = []
        for rank in range(context.world_size):
            failure_path = _rank_status_path(staging_root, rank, _RANK_FAILURE_FILE)
            try:
                failure = json.loads(failure_path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                failure = None
            if isinstance(failure, dict) and failure.get("run_id") == context.run_id:
                failures.append((rank, str(failure.get("traceback", "unknown worker failure"))))
                continue

            completion_path = _rank_status_path(staging_root, rank, "rank.complete.json")
            rank_stats = _read_current_rank_stats(
                _rank_status_path(staging_root, rank, _RANK_STATS_FILE),
                context=context,
            )
            if not completion_path.is_file() or rank_stats is None:
                missing.append(rank)
            else:
                stats.append(rank_stats)

        if failures:
            details = "\n".join(f"--- rank {rank} ---\n{error}" for rank, error in failures)
            raise RuntimeError(f"Reasoner extraction worker failed:\n{details}")
        if not missing:
            return sorted(stats, key=lambda item: int(item["rank"]))
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "Timed out waiting for Reasoner extraction ranks to finish; "
                f"missing current-run completion/stats from ranks {missing}"
            )
        time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))


def _write_worker_failure(args: argparse.Namespace, context: DistributedContext, error: str) -> None:
    output = Path(args.output)
    staging_root = output.parent / f".{output.name}.reasoner-kv-staging"
    _atomic_write_json(
        _rank_status_path(staging_root, context.rank, _RANK_FAILURE_FILE),
        {"rank": context.rank, "run_id": context.run_id, "traceback": error},
    )


def _run(args: argparse.Namespace, context: DistributedContext) -> Path:
    config = load_experiment_from_toml(args.sft_toml, extra_overrides=args.overrides)
    _validate_extraction_config(config)

    dtype = torch.bfloat16
    identity = ReasonerFeatureCacheIdentity(
        reasoner=args.reasoner_fingerprint,
        tokenizer=args.tokenizer_fingerprint,
        framing=args.framing_fingerprint,
    )
    output = Path(args.output)
    if output.exists():
        provider = OfflineReasonerFeatureProvider(
            output,
            expected_identity=identity,
            expected_dtype=dtype,
            verify_checksums=args.verify_resume_checksums,
        )
        if context.rank == 0:
            provider.verify_all_shards()
        manifest = output / "manifest.json"
        if context.rank == 0:
            print(f"Reasoner feature cache is already complete: {manifest}", flush=True)
        return manifest

    dataset = _instantiate_sft_dataset(config)
    special_tokens = _special_tokens(dataset)
    max_total_tokens = _extraction_max_total_tokens(dataset, special_tokens)
    writer = IncrementalReasonerFeatureCacheWriter.resume(
        output,
        identity=identity,
        rank=context.rank,
        world_size=context.world_size,
        max_shard_bytes=args.max_shard_bytes,
        verify_checksums=args.verify_resume_checksums,
    )
    use_float_positions = bool(config.model.config.diffusion_expert_config.enable_fps_modulation)
    requests = _iter_rank_requests(
        dataset,
        identity=identity,
        rank=context.rank,
        world_size=context.world_size,
        use_float_positions=use_float_positions,
        max_documents_per_rank=args.max_documents_per_rank,
        special_tokens=special_tokens,
    )

    torch.cuda.reset_peak_memory_stats(context.device)
    runtime = ReasonerFeatureRuntime.load(
        config,
        ReasonerRuntimeSpec(
            checkpoint=args.checkpoint,
            checkpoint_source=args.checkpoint_source,
            device=context.device,
            dtype=dtype,
            identity=identity,
            max_requests=1,
            max_total_tokens=max_total_tokens,
        ),
    )
    publishable = args.max_documents_per_rank is None
    stats = _extract_rank(
        runtime=runtime,
        requests=requests,
        writer=writer,
        context=context,
    )
    if runtime.load_stats is None:
        raise RuntimeError("Loaded Reasoner runtime did not report startup statistics")
    stats.reasoner_construction_seconds = runtime.load_stats.construction_seconds
    stats.checkpoint_load_seconds = runtime.load_stats.checkpoint_load_seconds
    _commit_rank_result(writer, stats, context=context, publishable=publishable)

    if not publishable:
        summary = {
            "published": False,
            "reason": ("INCOMPLETE / NOT FOR TRAINING: --max-documents-per-rank is a resumable smoke run"),
            "staging_root": str(writer.staging_root),
            "rank": asdict(stats),
        }
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return writer.staging_root

    if context.rank != 0:
        return output / "manifest.json"

    all_stats = _wait_for_rank_completion(
        writer.staging_root,
        context=context,
        timeout_seconds=args.coordination_timeout_seconds,
    )

    manifest = finalize_incremental_reasoner_feature_cache(
        output,
        identity=identity,
        world_size=context.world_size,
        verify_checksums=args.verify_resume_checksums,
    )
    summary = {
        "manifest": str(manifest),
        "checkpoint": str(args.checkpoint),
        "checkpoint_source": args.checkpoint_source,
        "identity": asdict(identity),
        "world_size": context.world_size,
        "ranks": all_stats,
    }
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return manifest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sft-toml", required=True)
    parser.add_argument("--checkpoint", required=True, help="Local DCP model component or iteration directory")
    parser.add_argument("--checkpoint-source", choices=("regular", "ema"), default="regular")
    parser.add_argument("--output", required=True, help="New immutable cache root on shared POSIX storage")
    parser.add_argument("--reasoner-fingerprint", required=True)
    parser.add_argument("--tokenizer-fingerprint", required=True)
    parser.add_argument("--framing-fingerprint", required=True)
    parser.add_argument("--dtype", choices=("bfloat16",), default="bfloat16")
    parser.add_argument("--max-shard-bytes", type=int, default=512 * 1024**2)
    parser.add_argument(
        "--max-documents-per-rank",
        type=int,
        default=None,
        help="Resumable smoke-test limit; flushes staging shards but never publishes a cache",
    )
    parser.add_argument(
        "--coordination-timeout-seconds",
        type=float,
        default=24 * 60 * 60,
        help="Rank-0 timeout while polling shared-POSIX completion markers",
    )
    parser.add_argument(
        "--coordination-run-id",
        default=None,
        help="Unique shared job ID; required for multi-node extraction",
    )
    parser.add_argument(
        "--verify-resume-checksums",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="Hydra overrides applied after TOML; prefix the list with --",
    )
    args = parser.parse_args(argv)
    args.overrides = [item for item in args.overrides if item != "--"]
    if args.max_shard_bytes <= 0:
        parser.error("--max-shard-bytes must be positive")
    if args.max_documents_per_rank is not None and args.max_documents_per_rank <= 0:
        parser.error("--max-documents-per-rank must be positive")
    if args.coordination_timeout_seconds <= 0:
        parser.error("--coordination-timeout-seconds must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    context = _initialize_distributed(args.coordination_run_id)
    try:
        _run(args, context)
        return 0
    except BaseException:
        error = traceback.format_exc()
        try:
            _write_worker_failure(args, context, error)
        except BaseException:
            pass
        raise


if __name__ == "__main__":
    raise SystemExit(main())
