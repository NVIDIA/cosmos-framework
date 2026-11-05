# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Population-indexed Transfer loading from immutable WebDataset tar members.

The offline index chooses one complete caption window and eligible horizon per
unique video. Workers range-read only RGB and the selected persisted control;
they do not scan shards or replace a scheduled horizon after sample rejection.
"""

import fcntl
import hashlib
import json
import math
import os
import random
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar
from urllib.parse import urlsplit

from botocore.exceptions import BotoCoreError, ClientError

from cosmos_framework.utils import log
from cosmos_framework.data.generator.indexed_transfer_length_loader import IndexedTransferDatasetBase
from cosmos_framework.data.generator.transfer_length_index_cache import (
    _build_index_s3_client,
    _ensure_private_cache_directory,
    _sha256_file,
)


def _source_draw_seed(draw_seed: int, purpose: str) -> int:
    """Keep video-slot, control, and caption/dropout decisions independent."""
    digest = hashlib.sha256(f"transfer-webdataset:{purpose}:{draw_seed}".encode()).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _verify_local_index(path: Path, expected_hash: str) -> None:
    """Hash a large shared index once per node while its file identity is unchanged."""
    if len(expected_hash) != 64 or any(character not in "0123456789abcdef" for character in expected_hash):
        raise ValueError("WebDataset length index requires a lowercase SHA-256 index_sha256.")
    cache = Path(tempfile.gettempdir()) / f"cosmos3-transfer-wds-verification-{os.getuid()}"
    _ensure_private_cache_directory(cache)
    identity = hashlib.sha256(f"{path}\0{expected_hash}".encode()).hexdigest()
    marker = cache / f"{identity}.json"
    lock_fd = os.open(cache / f"{identity}.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(lock_fd, "a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        metadata = path.stat()
        stamp = [metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns, metadata.st_ctime_ns]
        if marker.is_file() and json.loads(marker.read_text()) == stamp:
            return
        if _sha256_file(path) != expected_hash:
            raise ValueError("WebDataset length-index SHA-256 mismatch.")
        after = path.stat()
        if [after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns] != stamp:
            raise ValueError("WebDataset length index changed while its SHA-256 was being verified.")
        with tempfile.NamedTemporaryFile(mode="w", dir=cache, delete=False) as temporary:
            temporary.write(json.dumps(stamp))
        os.replace(temporary.name, marker)


class _IndexedTarSampleURL:
    """Keep the original shard aspect bucket available to shared augmentors."""

    def __init__(self, uri: str, sample_key: str, aspect_ratio: str) -> None:
        self.uri: str = uri
        self.sample_key: str = sample_key
        self.meta: SimpleNamespace = SimpleNamespace(opts={"aspect_ratio": aspect_ratio})

    def __str__(self) -> str:
        return f"{self.uri}#{self.sample_key}"


def _read_tar_member(client: Any, locator: dict[str, Any]) -> bytes:
    """Read one payload range, refusing a changed object or an incomplete range."""
    parsed = urlsplit(locator["uri"])
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError("Indexed WebDataset media must use an s3:// object URI.")
    start, length, object_size = int(locator["offset"]), int(locator["length"]), int(locator["object_size"])
    etag = locator["etag"]
    if start < 0 or length < 1 or start + length > object_size or not isinstance(etag, str) or not etag:
        raise ValueError("Indexed tar member has invalid bounds or no pinned ETag.")
    end = start + length - 1
    response = client.get_object(
        Bucket=parsed.netloc,
        Key=parsed.path.lstrip("/"),
        Range=f"bytes={start}-{end}",
        IfMatch=etag,
    )
    try:
        if str(response.get("ETag", "")).strip('"') != etag.strip('"'):
            raise ValueError("Indexed tar object ETag changed.")
        if response.get("ContentRange") != f"bytes {start}-{end}/{object_size}":
            raise ValueError("Object store did not return the exact indexed tar member range.")
        payload = response["Body"].read(length + 1)
        if len(payload) != length:
            raise ValueError("Object store returned an incomplete indexed tar member.")
        return payload
    finally:
        response["Body"].close()


def transfer_sample_from_payloads(
    entry: dict[str, Any],
    caption_key: str,
    dataset_name: str,
    modality: str,
    rgb_bytes: bytes,
    control_bytes: bytes | None = None,
) -> dict[str, Any]:
    """Build the same raw sample for remote ranges and local encoded assets."""
    rgb_locator = json.loads(entry["rgb_locator_json"])
    persisted = {
        name: json.loads(entry[key])
        for name, key in (("depth", "depth_locator_json"), ("seg", "seg_locator_json"))
        if entry[key] is not None
    }
    caption = {
        "caption_structured": json.dumps(
            {
                entry["caption_key"]: {
                    "start_frame": entry["caption_start"],
                    "end_frame": entry["caption_end"],
                    "caption": entry["caption_json"],
                }
            }
        )
    }
    sample: dict[str, Any] = {
        "video": rgb_bytes,
        "metas": {
            "width": entry["source_width"],
            "height": entry["source_height"],
            "framerate": entry["source_fps"],
            "nb_frames": entry["source_frames"],
            caption_key: caption,
        },
        "__key__": entry["sample_key"],
        "__url__": _IndexedTarSampleURL(rgb_locator["uri"], entry["sample_key"], entry["aspect_ratio"]),
        "source_dataset_name": dataset_name,
        "_selected_control_modality": modality,
    }
    if control_bytes is not None:
        sample["persisted_control"] = control_bytes
        sample["_persisted_control_meta"] = persisted[modality]["metadata"]
    return sample


class IndexedTransferWebDataset(IndexedTransferDatasetBase):
    """Read the corrected teacher's WDS source through a complete frozen index."""

    index_columns: ClassVar[set[str]] = IndexedTransferDatasetBase.index_columns | {
        "sample_key",
        "aspect_ratio",
        "caption_key",
        "caption_start",
        "caption_end",
        "caption_json",
        "source_frames",
        "source_fps",
        "source_width",
        "source_height",
        "rgb_locator_json",
        "depth_locator_json",
        "seg_locator_json",
    }

    def __init__(
        self,
        manifest_path: str | Path,
        creds_path: str | Path,
        tokenizer_config: Any,
        *,
        dataset_name: str,
        caption_key: str,
        index_creds_path: str | Path | None = None,
        index_cache_dir: str | Path | None = None,
        modality_weights: dict[str, float] | None = None,
        conditioning_config: dict[int, float] | None = None,
        resolution: str = "480",
        seed: int = 42,
        max_attempts: int = 100,
        min_num_frames: int = 9,
        cfg_dropout_rate: float = 0.1,
        dense_caption_probability: float = 0.07,
        resize_on_read: bool = True,
    ) -> None:
        super().__init__(
            manifest_path,
            tokenizer_config,
            index_creds_path=index_creds_path,
            index_cache_dir=index_cache_dir,
            caption_key=caption_key,
            conditioning_config=conditioning_config,
            resolution=resolution,
            seed=seed,
            max_attempts=max_attempts,
            min_num_frames=min_num_frames,
            cfg_dropout_rate=cfg_dropout_rate,
            dense_caption_probability=dense_caption_probability,
            resize_on_read=resize_on_read,
        )
        expected_source = {
            "source_kind": "webdataset",
            "dataset_name": dataset_name,
            "caption_key": caption_key,
            "resolution": resolution,
        }
        if any(self.manifest.get(key) != value for key, value in expected_source.items()):
            raise ValueError("WebDataset length index does not match the requested source, captions, or resolution.")
        coverage = self.manifest["coverage"]
        if int(coverage.get("expected_shards", 0)) < 1 or coverage.get("scanned_shards") != coverage["expected_shards"]:
            raise ValueError("The WebDataset length index must cover every source shard.")
        # The S3 materializer already verifies its cached SQLite file. For a
        # shared local index, avoid hashing multi-GB captions on every rank.
        if not str(manifest_path).startswith("s3://"):
            _verify_local_index(self.index_path, str(self.manifest.get("index_sha256", "")))
        self.dataset_name: str = dataset_name
        self.creds_path: str | Path = creds_path
        self.modality_weights: dict[str, float] = dict(
            modality_weights if modality_weights is not None else {"edge": 1.0, "blur": 1.0, "depth": 1.0, "seg": 1.0}
        )
        if (
            set(self.modality_weights) - {"edge", "blur", "depth", "seg"}
            or any(not math.isfinite(weight) or weight < 0 for weight in self.modality_weights.values())
            or sum(self.modality_weights.values()) <= 0
        ):
            raise ValueError("Transfer modality weights must select supported modalities with positive total weight.")
        self._s3_client: Any = None

    def _load_entry(self, entry: dict[str, Any], draw_seed: int) -> dict[str, Any] | None:
        rgb_locator = json.loads(entry["rgb_locator_json"])
        persisted = {
            modality: json.loads(entry[key])
            for modality, key in (("depth", "depth_locator_json"), ("seg", "seg_locator_json"))
            if entry[key] is not None
        }
        available = [
            modality
            for modality, weight in self.modality_weights.items()
            if weight > 0 and (modality in {"edge", "blur"} or modality in persisted)
        ]
        if not available:
            return None
        modality = random.Random(_source_draw_seed(draw_seed, "control")).choices(
            available, weights=[self.modality_weights[name] for name in available], k=1
        )[0]
        if self._s3_client is None:
            self._s3_client = _build_index_s3_client(self.creds_path)
        try:
            rgb_bytes = _read_tar_member(self._s3_client, rgb_locator)
            control_bytes = _read_tar_member(self._s3_client, persisted[modality]) if modality in persisted else None
        except (BotoCoreError, ClientError, OSError, ValueError) as error:
            log.warning(f"Skipping indexed WebDataset sample {entry['sample_key']}: {error}", rank0_only=False)
            return None
        return transfer_sample_from_payloads(
            entry, self.caption_key, self.dataset_name, modality, rgb_bytes, control_bytes
        )

    def _augment_sample(self, sample: dict[str, Any], bucket: int, draw_seed: int) -> dict[str, Any] | None:
        return super()._augment_sample(sample, bucket, _source_draw_seed(draw_seed, "augmentation"))
