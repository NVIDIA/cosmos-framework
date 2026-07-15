# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Record the image and video IDs consumed by training.

The callback is disabled by default. When enabled, every rank buffers the
``__key__`` and ``__url__`` values from batches that reached the training step.
At a configurable interval the buffers are gathered to rank 0, which appends
every consumed sample occurrence to one Lance table.

The resulting table can be inspected with the Streamlit viewer in
``tools/lance_sample_viewer``.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import torch
import torch.distributed as dist

from cosmos_framework.model._base import ImaginaireModel
from cosmos_framework.utils import log
from cosmos_framework.utils.callback import Callback

_REMOTE_URI_SCHEMES = {"gs", "s3"}
# Keep the training callback independent of the optional cosmos-sila package.
_LANCE_DATA_STORAGE_VERSION = "2.1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_list(value: Any, count: int, default: str = "") -> list[str]:
    """Normalize scalar or batched metadata to exactly ``count`` strings."""
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple)):
        values = [str(item) for item in value]
    elif value is None:
        values = []
    else:
        values = [str(value)]

    if len(values) == 1 and count > 1:
        values *= count
    if len(values) < count:
        values.extend([default] * (count - len(values)))
    return values[:count]


class SampledMediaRecorder(Callback):
    """Append consumed sample metadata to a Lance table from rank 0.

    Args:
        enabled: Enable recording. Disabled by default to avoid training overhead.
        output_uri: Destination ``.lance`` table. Local, ``s3://``, and ``gs://``
            URIs are supported.
        creds_path: Optional S3-compatible credential JSON used for remote Lance
            writes.
        flush_every_n_batches: Number of consumed microbatches buffered per rank
            between distributed gathers and table appends.
    """

    def __init__(
        self,
        enabled: bool = False,
        output_uri: str = "",
        creds_path: str | None = None,
        flush_every_n_batches: int = 100,
    ) -> None:
        super().__init__()
        if flush_every_n_batches < 1:
            raise ValueError(f"flush_every_n_batches must be >= 1, got {flush_every_n_batches}.")
        if enabled and not output_uri.endswith(".lance"):
            raise ValueError(f"output_uri must end with '.lance', got {output_uri!r}.")

        self.enabled: bool = enabled
        self.output_uri: str = output_uri
        self.creds_path: str | None = creds_path
        self.flush_every_n_batches: int = flush_every_n_batches

        self._pending: list[dict[str, Any]] = []
        self._batch_index: int = 0
        self._batches_since_flush: int = 0

    @staticmethod
    def _rank() -> int:
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
        return 0

    @staticmethod
    def _media_type(data_batch: dict[str, Any]) -> str:
        has_images = data_batch.get("images") is not None
        has_video = data_batch.get("video") is not None
        if has_images and not has_video:
            return "image"
        if has_video and not has_images:
            return "video"
        return "image_video" if has_images and has_video else "unknown"

    def _job_name(self) -> str:
        job_config = getattr(getattr(self, "config", None), "job", None)
        return str(getattr(job_config, "name", ""))

    def _extract_records(self, data_batch: dict[str, Any], iteration: int, rank: int) -> list[dict[str, Any]]:
        sample_ids_raw = data_batch.get("__key__")
        if sample_ids_raw is None:
            return []
        if isinstance(sample_ids_raw, str):
            sample_ids = [sample_ids_raw]
        elif isinstance(sample_ids_raw, (list, tuple)):
            sample_ids = [str(sample_id) for sample_id in sample_ids_raw]
        else:
            sample_ids = [str(sample_ids_raw)]
        if not sample_ids:
            return []

        count = len(sample_ids)
        media_urls = _as_list(data_batch.get("__url__"), count)
        dataset_names = _as_list(data_batch.get("dataset_name"), count)
        source_names = _as_list(data_batch.get("source_dataset_name"), count)
        recorded_at = _utc_now()
        media_type = self._media_type(data_batch)
        run_id = os.environ.get("SLURM_JOB_ID") or os.environ.get("WANDB_RUN_ID", "local")

        return [
            {
                "recorded_at": recorded_at,
                "run_id": run_id,
                "job_name": self._job_name(),
                "iteration": int(iteration),
                "batch_index": self._batch_index,
                "sample_index": sample_index,
                "rank": rank,
                "media_type": media_type,
                "dataset_name": dataset_names[sample_index],
                "source_dataset_name": source_names[sample_index],
                "sample_id": sample_id,
                "media_url": media_urls[sample_index],
            }
            for sample_index, sample_id in enumerate(sample_ids)
        ]

    @staticmethod
    def _table_schema() -> Any:
        import pyarrow as pa

        return pa.schema(
            [
                pa.field("recorded_at", pa.string(), nullable=False),
                pa.field("run_id", pa.string(), nullable=False),
                pa.field("job_name", pa.string(), nullable=False),
                pa.field("iteration", pa.int64(), nullable=False),
                pa.field("batch_index", pa.int64(), nullable=False),
                pa.field("sample_index", pa.int32(), nullable=False),
                pa.field("rank", pa.int32(), nullable=False),
                pa.field("media_type", pa.string(), nullable=False),
                pa.field("dataset_name", pa.string(), nullable=False),
                pa.field("source_dataset_name", pa.string(), nullable=False),
                pa.field("sample_id", pa.string(), nullable=False),
                pa.field("media_url", pa.string(), nullable=False),
            ]
        )

    def _load_credentials(self) -> dict[str, Any]:
        if self.creds_path is None:
            raise ValueError("creds_path is required for remote sampled-media output.")
        with Path(self.creds_path).open(encoding="utf-8") as handle:
            return json.load(handle)

    def _lance_storage_options(self) -> dict[str, str]:
        scheme = urlparse(self.output_uri).scheme
        if scheme not in _REMOTE_URI_SCHEMES:
            return {}

        raw = self._load_credentials()
        options: dict[str, str] = {
            "aws_access_key_id": str(raw["aws_access_key_id"]),
            "aws_secret_access_key": str(raw["aws_secret_access_key"]),
        }
        endpoint = raw.get("endpoint_url")
        if endpoint:
            options["aws_endpoint"] = str(endpoint)
            if "storage.googleapis.com" in str(endpoint):
                options["aws_virtual_hosted_style_request"] = "false"
        if raw.get("region_name"):
            options["aws_region"] = str(raw["region_name"])
        return options

    def _write_lance_records(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return

        import lance
        import pyarrow as pa

        schema = self._table_schema()
        table = pa.Table.from_pylist(records, schema=schema)
        storage_options = self._lance_storage_options()
        parsed = urlparse(self.output_uri)
        is_remote = parsed.scheme in _REMOTE_URI_SCHEMES
        if not is_remote:
            local_path = Path(parsed.path if parsed.scheme == "file" else self.output_uri)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            exists = local_path.exists()
        else:
            try:
                lance.dataset(self.output_uri, storage_options=storage_options)
            except (FileNotFoundError, OSError):
                exists = False
            else:
                exists = True

        if not exists:
            lance.write_dataset(
                table,
                self.output_uri,
                mode="create",
                data_storage_version=_LANCE_DATA_STORAGE_VERSION,
                enable_v2_manifest_paths=True,
                storage_options=storage_options,
            )
            return

        existing = lance.dataset(self.output_uri, storage_options=storage_options)
        if not existing.schema.equals(schema, check_metadata=False):
            raise ValueError(
                f"Sample recorder schema mismatch for {self.output_uri!r}: expected {schema}, found {existing.schema}."
            )
        lance.write_dataset(table, self.output_uri, mode="append", storage_options=storage_options)

    def _flush(self) -> None:
        if not self.enabled:
            return

        distributed = dist.is_available() and dist.is_initialized()
        rank = self._rank()
        if distributed:
            gathered: list[list[dict[str, Any]] | None] | None = (
                [None for _ in range(dist.get_world_size())] if rank == 0 else None
            )
            dist.gather_object(self._pending, object_gather_list=gathered, dst=0)
        else:
            gathered = [self._pending]

        error_message = ""
        if rank == 0:
            try:
                records = [record for rank_records in gathered or [] if rank_records for record in rank_records]
                self._write_lance_records(records)
                if records:
                    log.info(f"[SampledMediaRecorder] Appended {len(records):,} records to {self.output_uri!r}.")
            except Exception as error:
                error_message = f"SampledMediaRecorder failed to flush: {error}"

        if distributed:
            messages = [error_message]
            dist.broadcast_object_list(messages, src=0)
            error_message = messages[0]
        if error_message:
            raise RuntimeError(error_message)

        self._pending.clear()
        self._batches_since_flush = 0

    def on_training_step_batch_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, Any],
        output_batch: dict[str, Any],
        loss: torch.Tensor,  # []
        iteration: int = 0,
    ) -> None:
        del model, output_batch, loss
        if not self.enabled:
            return

        self._pending.extend(self._extract_records(data_batch, iteration, self._rank()))
        self._batch_index += 1
        self._batches_since_flush += 1
        if self._batches_since_flush >= self.flush_every_n_batches:
            self._flush()

    def on_train_end(self, model: ImaginaireModel, iteration: int = 0) -> None:
        del model, iteration
        if self.enabled:
            self._flush()
