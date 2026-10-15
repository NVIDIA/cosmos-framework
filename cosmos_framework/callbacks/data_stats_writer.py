# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Checkpoint-coupled per-sample parquet logging for native i4 VLM training.

Data stats live next to the checkpoints: ``<job>/checkpoints`` and ``<job>/data_stats``. The save that
writes checkpoint ``checkpoints/iter_<N>`` first writes the rows recorded since the previous save to
``data_stats/iter_<N>``, one ``rank_<r>_of_<world>.parquet`` file per rank. A merge then combines them into
``merged.parquet`` and deletes them. The merged file only appears once it holds every rank's rows, so its
presence marks the save complete. Rank 0 runs that merge in the background as
``python -m cosmos_framework.callbacks.data_stats_writer SAVE_DIR``.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.distributed as dist

from cosmos_framework.model._base import ImaginaireModel
from cosmos_framework.utils import distributed, log
from cosmos_framework.utils.callback import Callback

DATA_STATS_DIRNAME = "data_stats"
MERGED_FILE = "merged.parquet"
DATA_STATS_LINEAGE = "lineage.json"

_SAVE_DIR_PATTERN = re.compile(r"^iter_(\d{9})$")


@dataclass(frozen=True)
class _Field:
    name: str
    dtype: pa.DataType
    description: str

    def arrow(self) -> pa.Field:
        return pa.field(self.name, self.dtype, metadata={b"description": self.description.encode()})


_FIELDS = (
    _Field("phase", pa.string(), 'Data split that produced the row: "train", "val", or "data_only".'),
    _Field("step", pa.int64(), "Completed optimizer step associated with the sample."),
    _Field("rank", pa.int32(), "Distributed process rank that consumed the sample."),
    _Field("sample_rank", pa.int32(), "Rank recorded by the dataset distributor."),
    _Field("worker_id", pa.int32(), "Dataloader worker that selected the sample."),
    _Field("epoch", pa.int32(), "WDS distributor epoch or Lance recipe epoch (legacy: per-source pass)."),
    _Field("sample_index", pa.int64(), "WDS epoch shard index or zero-based Lance worker-local schedule position."),
    _Field("dataset_name", pa.string(), "Recipe dataset/category name."),
    _Field("source_backend", pa.string(), "Source identity domain: webdataset or lance."),
    _Field("dataset_population", pa.int64(), "Lance grouped sample count; null for WebDataset."),
    _Field("sample_id", pa.string(), "Stable sample identity independent of GCS or Lustre storage."),
    _Field("source_occurrence", pa.int64(), "Worker-local Lance source cursor after this occurrence."),
    _Field("group_index", pa.int64(), "Source-local Lance grouped sample index; null for WebDataset or legacy runs."),
    _Field("batch_sample_index", pa.int32(), "Logical sample position within the delivered batch."),
    _Field("file_name", pa.string(), "Sample key with the leading dataset prefix removed."),
    _Field("url", pa.string(), "Source artifact URL/path; a TAR for WebDataset."),
    _Field("num_trainable_tokens", pa.int32(), "Valid next-token labels for this sample."),
    _Field("token_ce_sum", pa.float32(), "Cross entropy summed over valid labels in this sample."),
    _Field("mean_token_ce", pa.float32(), "token_ce_sum divided by num_trainable_tokens."),
    _Field("objective_numerator", pa.float32(), "Sample contribution to the configured CE numerator."),
    _Field("objective_weight", pa.float32(), "Sample contribution to the configured CE denominator."),
    _Field(
        "objective_contribution",
        pa.float32(),
        "objective_numerator divided by the global denominator for this forward pass.",
    ),
    _Field("is_thinking_stripped", pa.bool_(), "Whether a reasoning trace was removed from this sample."),
    _Field("system_prompt", pa.string(), "Effective system prompt passed to the chat template."),
    _Field("num_images", pa.int32(), "Image references in the selected raw conversation."),
    _Field("num_videos", pa.int32(), "Video references in the selected raw conversation."),
    _Field("raw_image_tokens", pa.int64(), "Estimated native-resolution tokens across retained images."),
    _Field("raw_video_tokens", pa.int64(), "Estimated native-resolution/native-FPS video tokens."),
    _Field("video_native_fps", pa.float32(), "Mean native FPS across videos; null when unavailable."),
    _Field("video_native_num_frames", pa.int32(), "Total native frames across videos; null without video."),
    _Field("video_sampled_num_frames", pa.int32(), "Frames retained by decoding across videos; null without video."),
    _Field(
        "video_pixels_per_frame",
        pa.int64(),
        "Mean native spatial pixels per video frame; null without video.",
    ),
    _Field("video_duration_sec", pa.float32(), "Mean native video duration in seconds; null without video."),
    _Field("max_image_token_length", pa.int32(), "Configured image-token budget for the sample."),
    _Field("max_video_token_length", pa.int32(), "Configured video-token budget before random augmentation."),
    _Field(
        "effective_max_video_token_length",
        pa.int32(),
        "Video-token budget after decoder randomization; null without video.",
    ),
    _Field("raw_image_pixels", pa.int64(), "Native H*W summed across retained images."),
    _Field("raw_video_pixels", pa.int64(), "Native H*W*frames summed across videos."),
    _Field("image_min_pixels", pa.int64(), "Effective minimum image-pixel budget; null without image."),
    _Field("image_max_pixels", pa.int64(), "Effective maximum image-pixel budget; null without image."),
    _Field("video_min_pixels", pa.int64(), "Effective minimum video-pixel budget; null without video."),
    _Field("video_max_pixels", pa.int64(), "Effective maximum video-pixel budget; null without video."),
    _Field("seq_image_tokens", pa.int64(), "Image placeholders before context-length filtering."),
    _Field("seq_video_tokens", pa.int64(), "Video placeholders before context-length filtering."),
    _Field("raw_text_tokens", pa.int64(), "All non-image/video tokens before context-length filtering."),
    _Field("context_length", pa.int32(), "Configured model context length."),
    _Field("dataset_weight", pa.float64(), "Effective normalized sampling share of the dataset."),
    _Field("dataset_num_urls", pa.int64(), "Dataset URL count after recipe expansion."),
    _Field("total_training_steps", pa.int64(), "Configured maximum optimizer iterations."),
)

DATA_STATS_SCHEMA = pa.schema(
    [field.arrow() for field in _FIELDS],
    metadata={b"schema_version": b"2"},
)


def data_stats_root(checkpoint_root: Path) -> Path:
    """Return ``<job>/data_stats`` for the checkpoint directory ``<job>/checkpoints``."""
    return checkpoint_root.parent / DATA_STATS_DIRNAME


def save_dir_name(iteration: int) -> str:
    """Name a save folder the same as the checkpoint folder it belongs to."""
    return f"iter_{iteration:09d}"


def list_save_dirs(data_stats_root: Path) -> list[tuple[int, Path]]:
    """Return ``(iteration, folder)`` for every save folder under ``data_stats_root``, oldest first."""
    if not data_stats_root.is_dir():
        return []
    entries = []
    for path in data_stats_root.iterdir():
        match = _SAVE_DIR_PATTERN.match(path.name)
        if match is not None and path.is_dir():
            entries.append((int(match.group(1)), path))
    return sorted(entries)


def rank_file(save_dir: Path, rank: int, world_size: int) -> Path:
    return save_dir / f"rank_{rank:05d}_of_{world_size:05d}.parquet"


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary_path, path)


def is_merged(save_dir: Path) -> bool:
    return (save_dir / MERGED_FILE).is_file()


def merge_save_dir(save_dir: Path) -> None:
    """Merge one save's rank files into ``merged.parquet``, then delete them. Safe to rerun."""
    rank_files = sorted(save_dir.glob("rank_*_of_*.parquet"))
    if not is_merged(save_dir):
        world_sizes = {int(path.stem.rsplit("_of_", 1)[1]) for path in rank_files}
        if world_sizes != {len(rank_files)}:
            raise FileNotFoundError(
                f"{save_dir} holds {len(rank_files)} rank files written for world sizes {sorted(world_sizes)}"
            )
        expected_rows = sum(pq.read_metadata(path).num_rows for path in rank_files)
        temporary_path = save_dir / f"{MERGED_FILE}.tmp"
        with pq.ParquetWriter(temporary_path, pq.read_schema(rank_files[0]), compression="zstd") as writer:
            for path in rank_files:
                writer.write_table(pq.read_table(path))
        # The rank files are deleted next, so the merged file must read back with every row first.
        merged_rows = pq.read_metadata(temporary_path).num_rows
        if merged_rows != expected_rows:
            raise RuntimeError(f"Merged {merged_rows} rows in {save_dir}, but its rank files hold {expected_rows}")
        os.replace(temporary_path, save_dir / MERGED_FILE)
    for path in rank_files:
        path.unlink()


def checkpoint_data_stats_complete(checkpoint_dir: str | Path) -> bool:
    """Whether the save folder of the same name as ``checkpoint_dir`` has been merged."""
    checkpoint_dir = Path(checkpoint_dir)
    try:
        return is_merged(data_stats_root(checkpoint_dir.parent) / checkpoint_dir.name)
    except OSError:
        return False


_MERGE_MODULE = "cosmos_framework.callbacks.data_stats_writer"
_LOSS_KEYS = (
    "per_sample_token_ce_sum",
    "per_sample_valid_token_count",
    "per_sample_objective_numerator",
    "per_sample_objective_weight",
    "global_objective_weight",
)
_BATCH_KEYS = (
    "__key__",
    "__url__",
    "dataset_name",
    "sample_rank",
    "sample_worker_id",
    "sample_epoch",
    "sample_index",
)
_PROVENANCE_KEYS = frozenset({"source_backend", "dataset_population", "sample_id", "source_occurrence", "group_index"})
# Every schema field that _record does not compute must come from the sample's data_stats dict.
_COMPUTED_FIELDS = (
    frozenset(
        {
            "phase",
            "step",
            "rank",
            "sample_rank",
            "worker_id",
            "epoch",
            "sample_index",
            "dataset_name",
            "file_name",
            "url",
            "num_trainable_tokens",
            "token_ce_sum",
            "mean_token_ce",
            "objective_numerator",
            "objective_weight",
            "objective_contribution",
            "context_length",
            "dataset_num_urls",
            "total_training_steps",
            "batch_sample_index",
        }
    )
    | _PROVENANCE_KEYS
)
_METADATA_KEYS = frozenset(DATA_STATS_SCHEMA.names) - _COMPUTED_FIELDS


def _check_sample(key: str, metadata: dict[str, Any], batch_values: dict[str, Any]) -> None:
    """Raise, naming the sample, if its data-stats fields are missing, unexpected, null, or implausible."""
    required_metadata = set(metadata) - _PROVENANCE_KEYS
    if required_metadata != _METADATA_KEYS:
        raise KeyError(
            f"data_stats of sample {key}: missing {sorted(_METADATA_KEYS - required_metadata)}, "
            f"unexpected {sorted(required_metadata - _METADATA_KEYS)}"
        )
    values = {**batch_values, "dataset_weight": metadata["dataset_weight"]}
    nulls = [name for name, value in values.items() if value is None]
    if nulls:
        raise ValueError(f"Sample {key} has no value for {nulls}")
    population_key = "dataset_population" if metadata.get("source_backend") == "lance" else "dataset_num_urls"
    population = metadata.get(population_key, values.get(population_key))
    if not 0 < values["dataset_weight"] <= 1 or population is None or population <= 0:
        raise ValueError(
            f"Sample {key} has dataset_weight={values['dataset_weight']} and "
            f"{population_key}={population}; expected a weight in (0, 1] and a positive count"
        )
    if metadata.get("source_backend") == "lance" and (
        not metadata.get("sample_id")
        or not isinstance(metadata.get("source_occurrence"), int)
        or metadata["source_occurrence"] <= 0
    ):
        raise ValueError(f"Sample {key} has no valid Lance identity/cursor")
    group_index = metadata.get("group_index")
    if group_index is not None and (
        metadata.get("source_backend") != "lance" or type(group_index) is not int or not 0 <= group_index < population
    ):
        raise ValueError(f"Sample {key} has invalid group_index={group_index}")


def _values(value: Any, size: int) -> list[Any]:
    if isinstance(value, torch.Tensor):
        result = value.detach().cpu().reshape(-1).tolist()
    elif isinstance(value, (list, tuple)):
        result = list(value)
    else:
        result = [value] * size
    if len(result) != size:
        raise ValueError(f"Expected {size} values, got {len(result)}")
    return result


class DataStatsWriterCallback(Callback):
    """Write one parquet row for every logical sample seen during train and validation.

    Rows stay in memory until the next checkpoint save. Before the model is saved, every rank writes its rows
    to ``<job>/data_stats/iter_<N>``, and rank 0 starts merging those files in a background process while
    training continues. The next save, the end of training, and resume each wait for that merge and stop the
    job if it cannot succeed, so rank files never accumulate unnoticed. Rows recorded after the final save,
    which can only come from the last validation, are not saved.
    """

    def __init__(
        self,
        enabled: bool = False,
        merge_in_background: bool = True,
        merge_timeout_seconds: float = 1800.0,
        flush_on_end: bool = False,
    ) -> None:
        super().__init__()
        self.enabled = enabled
        self.merge_in_background = merge_in_background
        self.merge_timeout_seconds = merge_timeout_seconds
        self.flush_on_end = flush_on_end
        self._last_recorded_iteration: int = 0
        # Rows recorded since the previous save, one small table per microbatch.
        self._tables: list[pa.Table] = []
        self._rank = 0
        self._world_size = 1
        self._ready = False
        # Iteration of the newest save on the current timeline.
        self._last_save: int | None = None
        self._merge_process: subprocess.Popen[bytes] | None = None

    @property
    def output_dir(self) -> Path:
        return data_stats_root(Path(self.trainer.checkpointer.save_dirname))

    def _ensure_ready(self) -> None:
        if self._ready:
            return
        if str(self.trainer.checkpointer.save_dirname).startswith("s3://"):
            raise NotImplementedError("Checkpoint-coupled data stats currently require local checkpoint storage")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._rank = distributed.get_rank()
        self._world_size = distributed.get_world_size()
        self._ready = True

    def on_load_checkpoint_start(self, model: ImaginaireModel) -> None:
        del model
        if self.enabled:
            self._ensure_ready()

    def on_train_start(self, model: ImaginaireModel, iteration: int = 0) -> None:
        del model, iteration
        if self.enabled:
            self._ensure_ready()

    def _record(
        self,
        phase: str,
        data_batch: dict[str, Any],
        output_batch: dict[str, torch.Tensor] | None,
        iteration: int,
    ) -> None:
        if not self.enabled:
            return
        if not self._ready:
            raise RuntimeError("DataStatsWriterCallback was not initialized")
        metadata = data_batch.get("data_stats")
        if not isinstance(metadata, list) or not all(isinstance(sample_metadata, dict) for sample_metadata in metadata):
            raise TypeError(
                "data_batch['data_stats'] must be a list of per-sample dictionaries; "
                "is the data pipeline built with data_stats_writer_enabled?"
            )
        if data_batch.get("true_packing") or "cu_seq_lens_q" in data_batch:
            raise NotImplementedError("Per-sample data-stats logging currently requires padded batches")
        missing = [key for key in _LOSS_KEYS if output_batch is not None and key not in output_batch]
        if missing:
            raise KeyError(f"Model output is missing data-stats loss fields: {missing}")
        missing = [key for key in _BATCH_KEYS if data_batch.get(key) is None]
        if missing:
            raise KeyError(f"data_batch is missing data-stats fields: {missing}")

        size = len(metadata)
        columns = {key: _values(data_batch[key], size) for key in _BATCH_KEYS}
        columns["dataset_num_urls"] = _values(data_batch.get("dataset_num_urls", [None] * size), size)
        keys = [str(key) for key in columns["__key__"]]
        for index, sample_metadata in enumerate(metadata):
            batch_values = {name: values[index] for name, values in columns.items()}
            if sample_metadata.get("source_backend") == "lance":
                batch_values.pop("dataset_num_urls")
            _check_sample(keys[index], sample_metadata, batch_values)
        if output_batch is None:
            labels = data_batch["labels"]
            ignore_indices = _values(data_batch["ignore_index"], size)
            valid_counts = [int((labels[index, 1:] != ignore_indices[index]).sum()) for index in range(size)]
            token_sums = numerators = weights = [None] * size
            global_weight = 0.0
        else:
            token_sums = _values(output_batch[_LOSS_KEYS[0]], size)
            valid_counts = _values(output_batch[_LOSS_KEYS[1]], size)
            numerators = _values(output_batch[_LOSS_KEYS[2]], size)
            weights = _values(output_batch[_LOSS_KEYS[3]], size)
            global_weight = float(output_batch[_LOSS_KEYS[4]].detach().cpu())

        rows: list[dict[str, Any]] = []
        context_length = int(self.config.model.config.policy.model_max_length)
        total_steps = int(self.config.trainer.max_iter)
        for index, sample_metadata in enumerate(metadata):
            token_count = int(valid_counts[index])
            token_sum = None if token_sums[index] is None else float(token_sums[index])
            numerator = None if numerators[index] is None else float(numerators[index])
            key = keys[index]
            rows.append(
                {
                    "phase": phase,
                    "step": int(iteration),
                    "rank": self._rank,
                    "sample_rank": int(columns["sample_rank"][index]),
                    "worker_id": int(columns["sample_worker_id"][index]),
                    "epoch": int(columns["sample_epoch"][index]),
                    "sample_index": int(columns["sample_index"][index]),
                    "dataset_name": str(columns["dataset_name"][index]),
                    "batch_sample_index": index,
                    "source_backend": sample_metadata.get("source_backend", "webdataset"),
                    "dataset_population": sample_metadata.get("dataset_population"),
                    "sample_id": sample_metadata.get("sample_id", f"{columns['__url__'][index]}:{key}"),
                    "source_occurrence": sample_metadata.get("source_occurrence"),
                    "group_index": sample_metadata.get("group_index"),
                    "file_name": key.split("__", 1)[1] if "__" in key else key,
                    "url": str(columns["__url__"][index]),
                    "num_trainable_tokens": token_count,
                    "token_ce_sum": token_sum,
                    "mean_token_ce": token_sum / token_count if token_count and token_sum is not None else None,
                    "objective_numerator": numerator,
                    "objective_weight": None if weights[index] is None else float(weights[index]),
                    "objective_contribution": numerator / global_weight
                    if global_weight > 0 and numerator is not None
                    else None,
                    "context_length": context_length,
                    "dataset_num_urls": columns["dataset_num_urls"][index],
                    "total_training_steps": total_steps,
                    # Includes dataset_weight: the batch-level tensor is bf16 here, but custom_collate keeps
                    # the exact value per sample.
                    **sample_metadata,
                }
            )
        try:
            self._tables.append(pa.Table.from_pylist(rows, schema=DATA_STATS_SCHEMA))
        except (pa.ArrowException, OverflowError) as error:
            raise ValueError(f"data_stats rows of samples {keys} do not match the schema: {error}") from error
        self._last_recorded_iteration = iteration

    def record_data_batch(self, data_batch: dict[str, Any], iteration: int) -> None:
        """Record delivered samples without a model forward pass or invented losses."""
        self._record("data_only", data_batch, None, iteration)

    def on_training_step_batch_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        del model, loss
        if self.enabled:
            # This hook runs before regular checkpointing and once per accumulation
            # microbatch. All samples represented by checkpoint N therefore carry step N.
            self._record("train", data_batch, output_batch, iteration + 1)

    def on_validation_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        del model, loss
        self._record("val", data_batch, output_batch, iteration)

    def _all_ranks(self, local: bool) -> bool:
        """Whether ``local`` holds on every rank."""
        if not dist.is_available() or not dist.is_initialized():
            return local
        device = (
            torch.device("cuda", torch.cuda.current_device()) if dist.get_backend() == "nccl" else torch.device("cpu")
        )
        flag = torch.tensor(int(local), device=device)  # []
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        return bool(flag.item())

    def _on_rank0(self, action: Callable[[], None], description: str) -> None:
        """Run filesystem work on rank 0 and fail every rank together if it fails."""
        error: Exception | None = None
        if distributed.is_rank0():
            try:
                action()
            except Exception as caught:  # noqa: BLE001 - every rank must reach the success collective
                error = caught
        if not self._all_ranks(error is None):
            raise RuntimeError(f"Data stats: could not {description}") from error

    def _save_dir(self, iteration: int) -> Path:
        return self.output_dir / save_dir_name(iteration)

    def _flush(self, iteration: int) -> Path:
        """Write every rank's rows since the previous save to a new save folder."""
        save_dir = self._save_dir(iteration)
        error: Exception | None = None
        try:
            save_dir.mkdir(parents=True, exist_ok=True)
            path = rank_file(save_dir, self._rank, self._world_size)
            temporary_path = path.with_name(f"{path.name}.tmp")
            table = pa.concat_tables(self._tables) if self._tables else DATA_STATS_SCHEMA.empty_table()
            pq.write_table(table, temporary_path, compression="zstd")
            os.replace(temporary_path, path)
        except Exception as caught:  # noqa: BLE001 - every rank must reach the success collective
            error = caught
        if not self._all_ranks(error is None):
            raise RuntimeError(f"Data-stats flush failed before checkpoint {iteration}") from error
        self._tables = []
        self._last_save = iteration
        return save_dir

    def _finish_save(self, iteration: int) -> None:
        """Wait for a save's background merge, and merge here if it did not succeed."""
        process, self._merge_process = self._merge_process, None
        if process is not None:
            try:
                process.wait(timeout=self.merge_timeout_seconds)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            if process.returncode != 0:
                log.warning(f"[DataStats] Background merge of {save_dir_name(iteration)} failed; retrying here.")
        merge_save_dir(self._save_dir(iteration))

    def _complete_last_save(self) -> None:
        last = self._last_save
        if last is not None:
            self._on_rank0(lambda: self._finish_save(last), f"complete {save_dir_name(last)}")

    def on_save_checkpoint_start(self, model: ImaginaireModel, iteration: int = 0) -> None:
        """Require the previous save to be merged, then write this save's rows before the model."""
        del model
        if not self.enabled:
            return
        self._ensure_ready()
        self._complete_last_save()
        if iteration == self._last_save:
            # No training happens between two saves at one iteration, so the model and its rows are unchanged
            # and the existing folder already describes this checkpoint.
            return
        save_dir = self._flush(iteration)
        if distributed.is_rank0() and self.merge_in_background:
            try:
                self._merge_process = subprocess.Popen([sys.executable, "-m", _MERGE_MODULE, str(save_dir)])
            except OSError as error:
                log.warning(f"[DataStats] Could not start the merge of {save_dir.name} ({error}); merging later.")

    def on_app_end(self) -> None:
        """Make sure the final save is merged before the job exits."""
        if not self.enabled:
            return
        self._complete_last_save()
        if self.flush_on_end:
            iteration = self._last_recorded_iteration
            if dist.is_available() and dist.is_initialized():
                iterations = [iteration] * self._world_size
                dist.all_gather_object(iterations, iteration)
                iteration = max(iterations)
            if iteration != self._last_save:
                self._flush(iteration)
                self._complete_last_save()
            return
        dropped = sum(table.num_rows for table in self._tables)
        if dropped:
            log.info(f"[DataStats] Not saving {dropped} rows recorded after the final checkpoint on rank 0.")

    def on_load_checkpoint_end(
        self,
        model: ImaginaireModel,
        iteration: int = 0,
        checkpoint_path: str | None = None,
    ) -> None:
        """Make the stats history match the checkpoint that training continues from."""
        del model
        if not self.enabled:
            return
        self._ensure_ready()
        if checkpoint_path is None:
            self._on_rank0(self._discard_history, "discard stale data stats before training from scratch")
            return
        checkpoint_root = Path(checkpoint_path).parent
        if checkpoint_root.resolve() == Path(self.trainer.checkpointer.save_dirname).resolve():
            self._on_rank0(lambda: self._resume(iteration), f"resume data stats at iteration {iteration}")
            self._last_save = iteration
            return
        parent_root = data_stats_root(checkpoint_root)
        if not (parent_root / save_dir_name(iteration)).is_dir():
            log.warning(f"[DataStats] {checkpoint_path} has no data stats; this job's history starts empty.")
            self._on_rank0(self._discard_history, "discard stale data stats before a warm start")
            return
        self._on_rank0(lambda: self._start_branch(iteration, parent_root), f"record lineage from {checkpoint_path}")

    def _discard_history(self) -> None:
        for _, save_dir in list_save_dirs(self.output_dir):
            shutil.rmtree(save_dir)
        (self.output_dir / DATA_STATS_LINEAGE).unlink(missing_ok=True)

    def _resume(self, iteration: int) -> None:
        """Drop saves newer than the checkpoint, then merge the ones it depends on that are still unmerged."""
        kept: list[Path] = []
        for save_iteration, save_dir in list_save_dirs(self.output_dir):
            if save_iteration > iteration:
                shutil.rmtree(save_dir)
            else:
                kept.append(save_dir)
        if not self._save_dir(iteration).is_dir():
            raise RuntimeError(f"Checkpoint iteration {iteration} has no data-stats save folder")
        for save_dir in kept:
            merge_save_dir(save_dir)

    def _start_branch(self, iteration: int, parent_root: Path) -> None:
        self._discard_history()
        write_json_atomic(
            self.output_dir / DATA_STATS_LINEAGE,
            {"branch_start_step": iteration, "parent_data_stats_root": str(parent_root)},
        )


def main(arguments: list[str] | None = None) -> int:
    """Merge one save folder; the exit code reports success to the training process."""
    arguments = sys.argv[1:] if arguments is None else arguments
    if len(arguments) != 1:
        print(f"usage: python -m {_MERGE_MODULE} SAVE_DIR", file=sys.stderr)
        return 2
    try:
        merge_save_dir(Path(arguments[0]))
    except Exception as error:  # noqa: BLE001 - the parent reads failures from the exit code
        print(f"[data_stats_writer] Merge of {arguments[0]} failed: {error!r}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
