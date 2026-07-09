# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""IterableDataset for manifest-indexed local SFT datasets.

Reads a JSON manifest (``meta.json``) listing per-sample media + conversation
files on local disk, decodes each sample lazily into the dict shape the VLM
augmentor chain expects, and applies the same augmentor sequence used by the
WebDataset path so downstream tokenization is unchanged.

Canonical on-disk layout (matches preprocessing scripts output):

    <data_root>/
      meta.json             # JSON array: [{id, media, conversation}, ...]
      media/<id>.mp4        # one media file per sample
      text/<id>.json        # one conversation JSON per sample

Each conversation JSON has shape::

    {"conversations": [
        {"role": "user", "content": [
            {"type": "video", "video": "<media_field_name>"},
            {"type": "text",  "text":  "<prompt>"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "<answer>"}]}]}

The string ``<media_field_name>`` (e.g. ``"video_0"``) is the key into
``data_dict["media"]`` that the loader populates with the media bytes — the
``BytesToMedia`` augmentor then decodes those bytes to PIL frames.
"""

from __future__ import annotations

import json
import os
import random
import time
from typing import Any, Iterable, Iterator, Optional

import torch
import torch.distributed as dist
import torch.utils.data

from cosmos_framework.utils.lazy_config import instantiate
from cosmos_framework.utils import log


def _wrap_augmentor_func_as_generator(func, data):
    """Inline copy of imaginaire's helper — yields outputs of `func(sample)` and
    skips any sample where `func` returns None (the "unhealthy sample" convention)."""
    for data_dict in data:
        data_dict_out = func(data_dict)
        if data_dict_out is None:
            continue
        yield data_dict_out


def _run_augmentor_chain(data, augmentations):
    """Inline copy of `imaginaire.datasets.webdataset.webdataset.Dataset.augmentor_fn`.

    Importing the real `Dataset.augmentor_fn` transitively loads
    `imaginaire.lazy_config`, which collides with `cosmos_framework.utils.lazy_config`
    at OmegaConf resolver registration time. We inline the few-line wrapper so the
    local SFT loader needs only the OSS lazy_config module.
    """
    def _stamp_pre_aug(upstream):
        for sample in upstream:
            sample["_pre_aug_time"] = time.monotonic()
            sample["_aug_step_last"] = sample["_pre_aug_time"]
            yield sample

    def _checkpoint(upstream, step_name):
        for sample in upstream:
            now = time.monotonic()
            last = sample.get("_aug_step_last", now)
            sample.setdefault("_aug_step_times", {})[step_name] = now - last
            sample["_aug_step_last"] = now
            yield sample

    data = _stamp_pre_aug(data)
    for aug_fn in augmentations:
        name = getattr(aug_fn, "__name__", None) or type(aug_fn).__name__
        if getattr(aug_fn, "is_generator", False):
            data = aug_fn(data)
        else:
            data = _wrap_augmentor_func_as_generator(aug_fn, data)
        data = _checkpoint(data, name)
    for sample in data:
        sample.pop("_aug_step_last", None)
        pre = sample.pop("_pre_aug_time", None)
        if pre is not None:
            sample["_aug_time"] = time.monotonic() - pre
        yield sample


class _UrlObj:
    """Lightweight stand-in carrying `.root` and `.path` accessors.

    Several augmentors in the chain peek at `__url__.root` (e.g.
    `TimeStamp.__call__`: ``url.root``, see
    `augmentors/timestamp.py:460`). After the chain runs we normalize
    `__url__` to a plain `os.path.join(root, path)` string — matching what
    the existing webdataset path does via `update_url` at
    `imaginaire/datasets/webdataset/utils/misc.py:82` — so the value
    survives multiprocessing pickling and `imaginaire.utils.misc.to`
    traversal that would otherwise mis-reconstruct a namedtuple.

    Implemented as a plain class (not a namedtuple, not a str subclass)
    so it pickles via the default `__reduce__` (no custom `__new__` to
    fight). The augmentor chain accesses attrs, never iterates it.
    """

    __slots__ = ("root", "path")

    def __init__(self, root: str, path: str) -> None:
        self.root = str(root)
        self.path = str(path)

    def __repr__(self) -> str:
        return f"_UrlObj(root={self.root!r}, path={self.path!r})"


def _make_url_obj(root: str, path: str) -> _UrlObj:
    return _UrlObj(root, path)


def _normalize_url(sample: dict) -> dict:
    """Replace the augmentor-chain `_UrlObj` with the joined string the
    collate / training process expects."""
    url = sample.get("__url__")
    if url is not None and not isinstance(url, str):
        sample["__url__"] = os.path.join(str(url.root), str(url.path))
    return sample


# Public re-export retained for tests/imports.
Url = _UrlObj


def _extract_media_key(conversation: list) -> Optional[str]:
    """Find the first video/image reference inside a conversation.

    The conversation schema is a list of message dicts; each user message's
    ``content`` is a list of typed blocks like ``{"type": "video", "video":
    "<key>"}`` or ``{"type": "image", "image": "<key>"}``. ``TokenizeData``
    looks the returned ``<key>`` up in ``data_dict["media"]`` to bind tokens
    to media — so the loader must place the media bytes under that exact key.
    Returns the first ``<key>`` found, or ``None`` if no media block is
    present (text-only sample).
    """
    if not isinstance(conversation, list):
        return None
    for message in conversation:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            kind = item.get("type")
            if kind in ("video", "image") and isinstance(item.get(kind), str):
                return item[kind]
    return None


def _augmentations_from_config(augmentor_config: Optional[dict]) -> list:
    """Instantiate augmentor LazyCalls in registration order."""
    if not augmentor_config:
        return []
    augmentations = []
    for key in augmentor_config:
        augmentations.append(instantiate(augmentor_config[key]))
    return augmentations


def _rank_world() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _worker_info() -> tuple[int, int]:
    info = torch.utils.data.get_worker_info()
    if info is None:
        return 0, 1
    return info.id, info.num_workers


class LocalSFTDataset(torch.utils.data.IterableDataset):
    """Manifest-indexed local-file dataset for VLM SFT.

    Args:
        manifest_path: Absolute filesystem path to ``meta.json``.
        data_root: Directory the manifest's relative ``media`` and
            ``conversation`` paths resolve against. Defaults to the directory
            containing ``manifest_path``.
        media_field_name: Key under which media bytes are stored in
            ``data_dict["media"]``. Must match the ``video``/``image`` field
            referenced inside each conversation JSON so the tokenizer can
            stitch tokens to media. Must contain ``"video"`` or ``".mp4"`` to
            route through ``BytesToMedia``'s video branch.
        augmentor_config: Dict of augmentor LazyCalls (e.g. the output of
            ``create_data_augmentor_config()``). Applied in iteration order.
        text_only: Skip media loading and emit no ``media`` key. The
            conversation must reference no media items.
        shuffle: Shuffle manifest entries per epoch.
        distributor_seed: Base RNG seed for per-epoch shuffles.
        is_infinite_loader: If True, restart from the next epoch when the
            current shard is exhausted. The joint loader expects this.
        subsample_config: Optional ``{manifest_name: {"train": frac, "val":
            frac}}`` to slice the manifest by ratio.
    """

    is_iterable_dataset = True

    def __init__(
        self,
        manifest_path: str,
        data_root: Optional[str] = None,
        media_field_name: str = "video_0",
        augmentor_config: Optional[dict] = None,
        text_only: bool = False,
        shuffle: bool = True,
        distributor_seed: int = 1993,
        is_infinite_loader: bool = True,
        subsample_config: Optional[dict] = None,
        split: str = "train",
        dataset_name: str = "local_sft",
    ) -> None:
        super().__init__()
        if not manifest_path:
            raise ValueError("manifest_path is required")
        self.manifest_path = str(manifest_path)
        self.data_root = str(data_root) if data_root else os.path.dirname(self.manifest_path)
        self.media_field_name = media_field_name
        self.augmentor_config = augmentor_config
        self.text_only = text_only
        self.shuffle = shuffle
        self.distributor_seed = int(distributor_seed)
        self.is_infinite_loader = bool(is_infinite_loader)
        self.subsample_config = subsample_config
        self.split = split
        self.dataset_name = dataset_name

        self._augmentations: Optional[list] = None
        self._entries: Optional[list[dict]] = None

        # Eagerly read the manifest length so JointDatasetDynamicBatchingWebLoader
        # has a usable `total_images`. We avoid loading bytes here.
        manifest = self._load_manifest()
        self.total_images: int = len(manifest)
        log.info(
            f"LocalSFTDataset({self.dataset_name}, split={self.split}): "
            f"manifest={self.manifest_path} entries={self.total_images} data_root={self.data_root}"
        )

    # ------------------------------------------------------------------ public
    def build_dataset(self):  # JointDatasetDynamicBatchingWebLoader contract
        return self

    def __len__(self) -> int:
        return self.total_images

    def __iter__(self) -> Iterator[dict[str, Any]]:
        if self._augmentations is None:
            self._augmentations = _augmentations_from_config(self.augmentor_config)
        epoch = 0
        while True:
            sample_iter = self._raw_sample_iter(epoch=epoch)
            if self._augmentations:
                sample_iter = _run_augmentor_chain(sample_iter, self._augmentations)
            for sample in sample_iter:
                # Match the wdinfo path's post-augmentor `update_url`
                # normalization so `__url__` is a plain string downstream.
                _normalize_url(sample)
                yield sample
            if not self.is_infinite_loader:
                return
            epoch += 1

    # ----------------------------------------------------------------- helpers
    def _load_manifest(self) -> list[dict]:
        if self._entries is not None:
            return self._entries
        with open(self.manifest_path, "r") as f:
            manifest = json.load(f)
        if not isinstance(manifest, list):
            raise ValueError(f"manifest at {self.manifest_path} must be a JSON array")
        if self.subsample_config and self.dataset_name in self.subsample_config:
            frac = self.subsample_config[self.dataset_name].get(self.split, 1.0)
            if frac < 1.0:
                n = max(1, int(len(manifest) * frac))
                # Deterministic slice — sort by id for stability, then take first n
                manifest = sorted(manifest, key=lambda e: e.get("id", ""))[:n]
        self._entries = manifest
        return manifest

    def _per_partition_indices(self, epoch: int) -> list[int]:
        """Two-level stride sharding so every rank sees at least some data.

        We slice once by world-size (DP shard) and again by worker count.
        The previous single-stride scheme `indices[rank*nw+wid :: world*nw]`
        starves later ranks when `manifest_len < world * nw` because the
        partition_ids beyond `manifest_len` get empty slices — FSDP then
        hangs (e.g. with 32 samples + 8 ranks + 8 workers, ranks 4-7 got
        zero samples).
        """
        manifest = self._load_manifest()
        total = len(manifest)
        rank, world = _rank_world()
        worker_id, num_workers = _worker_info()

        indices = list(range(total))
        if self.shuffle:
            rng = random.Random(self.distributor_seed + epoch)
            rng.shuffle(indices)
        per_rank = indices[rank::world]
        return per_rank[worker_id::num_workers]

    def _read_sample(self, entry: dict) -> Optional[dict[str, Any]]:
        sample_id = entry.get("id") or entry.get("key") or "<unknown>"
        conv_rel = entry.get("conversation")
        media_rel = entry.get("media")
        if conv_rel is None:
            log.warning(f"manifest entry {sample_id} missing 'conversation'; skipping")
            return None

        conv_path = os.path.join(self.data_root, conv_rel)
        try:
            with open(conv_path, "r") as f:
                conv_json = json.load(f)
        except FileNotFoundError:
            log.warning(f"conversation file missing for {sample_id}: {conv_path}; skipping")
            return None
        except json.JSONDecodeError as exc:
            log.warning(f"conversation JSON invalid for {sample_id}: {exc}; skipping")
            return None

        # Accept both {"conversations": [...]} (canonical layout) and a raw list.
        if isinstance(conv_json, dict) and "conversations" in conv_json:
            texts = conv_json["conversations"]
        elif isinstance(conv_json, list):
            texts = conv_json
        else:
            log.warning(
                f"conversation JSON for {sample_id} is not a list or "
                f"{{'conversations': [...]}}; got {type(conv_json).__name__}"
            )
            return None

        sample: dict[str, Any] = {
            "texts": texts,
            "__url__": _make_url_obj(self.data_root, str(media_rel or conv_rel)),
            "__key__": str(sample_id),
            "__source__": self.dataset_name,
        }

        if not self.text_only:
            if media_rel is None:
                log.warning(f"manifest entry {sample_id} missing 'media'; skipping")
                return None
            media_path = os.path.join(self.data_root, media_rel)
            try:
                with open(media_path, "rb") as f:
                    media_bytes = f.read()
            except FileNotFoundError:
                log.warning(f"media file missing for {sample_id}: {media_path}; skipping")
                return None
            # Resolve the media key from the conversation when possible — each
            # sample's conversation may reference a different name (e.g.
            # "video_0" for the first sample, "video_29" for the 30th).
            media_key = _extract_media_key(texts) or self.media_field_name
            sample["media"] = {media_key: media_bytes}

        return sample

    def _raw_sample_iter(self, epoch: int) -> Iterable[dict[str, Any]]:
        manifest = self._load_manifest()
        indices = self._per_partition_indices(epoch)
        for idx in indices:
            entry = manifest[idx]
            sample = self._read_sample(entry)
            if sample is not None:
                yield sample
