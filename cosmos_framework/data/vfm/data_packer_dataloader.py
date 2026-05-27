# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""
OSS-facing dataloader that wires any Python iterable + DataPacker into the
shared PackingIterableDataset engine.

Follows the same two-layer pattern as the internal path:
  private  _DataPackerIterableDataset   ↔  private  _JointIterableDataset
  public   DataPackerDataLoader         ↔  public   JointDatasetDynamicBatchingWebLoader

Data-parallel sharding
----------------------
When ``torch.distributed`` is initialized, ``DataPackerDataLoader`` automatically
shards ``data_source`` across ranks **and** DataLoader workers using round-robin
filtering — the same pattern as ``SFTDataset`` in
``projects/cosmos3/vfm/datasets/local_datasets/sft_dataset.py``.

Each ``(dp_rank, worker_id)`` pair sees every
``dp_world_size × num_workers``-th item, giving disjoint coverage.

Usage
-----
Pass a pre-built iterable directly::

    loader = DataPackerDataLoader(
        data_source=my_dataset,           # any Python iterable
        data_packer=MyDataPacker(...),
        max_tokens=16000,
        num_workers=4,
    )

Or load a HuggingFace / local dataset via ``load_data_source`` — compatible
with Hydra ``LazyCall`` so CLI overrides work without editing Python files::

    from cosmos_framework.utils.lazy_config import LazyCall as L
    from cosmos_framework.data.vfm.data_packer_dataloader import (
        DataPackerDataLoader,
        load_data_source,
    )

    dataloader_train = L(DataPackerDataLoader)(
        data_source=L(load_data_source)(
            name="liuhaotian/LLaVA-Instruct-150K",
            split=["train"],
        ),
        data_packer=L(MyDataPacker)(...),
        max_tokens=16000,
    )

    # CLI override (no Python file edit needed):
    # dataloader_train.data_source.name=my-org/my-dataset
    # dataloader_train.data_source.split=[train,validation]

    # FSDP + TP/PP (pass parallel_dims for correct DP rank):
    loader = DataPackerDataLoader(
        data_source=...,
        data_packer=...,
        max_tokens=16000,
        parallel_dims=parallel_dims,  # uses parallel_dims.dp_coord
    )
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
import torch.utils.data

from cosmos_framework.utils import log
from cosmos_framework.data.vfm.data_packer import DataPacker
from cosmos_framework.data.vfm.packing_iterable_dataset import PackingIterableDataset


def load_data_source(
    name: str,
    split: str | list[str] = "train",
    subset: str | None = None,
    revision: str | None = None,
) -> Any:
    """Load a HuggingFace or local dataset for use as ``data_source``.

    Designed to be used as a ``LazyCall`` in Hydra experiment configs so that
    dataset name and split can be overridden from the CLI without editing Python
    files (see module docstring for an example).

    Parameters
    ----------
    name:
        HuggingFace dataset name (e.g. ``"liuhaotian/LLaVA-Instruct-150K"``) or
        a local directory path to a dataset saved with ``dataset.save_to_disk()``.
        Local paths are detected via ``os.path.isdir`` and loaded with
        ``load_from_disk``; all other values go through ``load_dataset``.
    split:
        Split name or list of split names to load.  When a list is given the
        splits are concatenated into a single dataset.
    subset:
        HuggingFace dataset subset / config name (optional).
    revision:
        Git revision / commit hash of the dataset (optional).

    Returns
    -------
    datasets.Dataset
        A concatenated ``datasets.Dataset`` ready to be passed to
        ``DataPackerDataLoader`` as ``data_source``.

    Raises
    ------
    ImportError
        If the ``datasets`` package is not installed.
    """
    try:
        from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required by load_data_source. Install it with: pip install datasets"
        ) from exc

    import os

    if os.path.isdir(name):
        # Dataset saved with dataset.save_to_disk() — use load_from_disk.
        raw = load_from_disk(name)
    else:
        # HuggingFace Hub name or other format supported by load_dataset.
        raw = load_dataset(name, subset, revision=revision)

    if isinstance(raw, Dataset):
        # load_from_disk on a single Dataset (not DatasetDict) — return as-is.
        return raw

    # DatasetDict: select and concatenate requested splits.
    splits = [split] if isinstance(split, str) else split
    return concatenate_datasets([raw[s] for s in splits])


class _IterableWrapper(torch.utils.data.IterableDataset):
    """Wraps any Python iterable as a ``torch.utils.data.IterableDataset``
    with built-in data-parallel + multi-worker sharding.

    Sharding follows the same ``(dp_rank × num_workers)`` formula as
    ``SFTDataset`` — each ``(dp_rank, worker_id)`` pair receives every
    ``dp_world_size × num_workers``-th item starting at
    ``dp_rank * num_workers + worker_id``.

    .. warning::
        For ``num_workers=0``, worker-level sharding is skipped automatically.
    """

    def __init__(self, iterable: Any, dp_rank: int = 0, dp_world_size: int = 1):
        super().__init__()
        self._iterable = iterable
        self._dp_rank = dp_rank
        self._dp_world_size = dp_world_size

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
        else:
            num_workers, worker_id = 1, 0

        # Total independent streams = dp_world_size × num_workers.
        # Each (rank, worker) pair owns stream = rank * num_workers + worker_id.
        total_streams = self._dp_world_size * num_workers
        my_stream = self._dp_rank * num_workers + worker_id

        for i, item in enumerate(self._iterable):
            if i % total_streams == my_stream:
                yield item


class _ShuffledMapIterableDataset(torch.utils.data.IterableDataset):
    """Stateful, sharded wrapper for map-style ``torch.utils.data.Dataset``.

    Used for ALL map-style ``data_source`` inputs, regardless of ``shuffle``.
    Handles DP × worker sharding and stateful checkpoint/resume.

    - Shuffle (``shuffle=True``): per-epoch ``torch.randperm(n)`` seeded with
      ``base_seed + epoch``, giving a different but reproducible ordering every epoch.
    - No shuffle (``shuffle=False``): sequential iteration ``[0, 1, ..., n-1]``
      each epoch — deterministic and resumable at the exact position.
    - Sharding: ``stream_id = dp_rank * num_workers + worker_id``; each stream
      yields ``perm[stream_id :: total_streams]`` — disjoint, full coverage.
    - Resume: reads ``DP_STATE_WORKER_{worker_id}_EPOCH`` /
      ``DP_STATE_WORKER_{worker_id}_INDEX`` env vars set by
      ``DataLoaderStateCallback.load_state_dict`` before workers start.
      When a dataset ``name`` is provided (non-empty), env vars are namespaced
      as ``DP_STATE_{name}_WORKER_{worker_id}_EPOCH`` to avoid conflicts when
      multiple ``DataPackerDataLoader`` instances share the same process (e.g.
      inside ``JointDataPackerDataLoader``).

    The generator body is lazy: ``worker_info`` (and env vars) are read on the
    first ``next()`` call inside the worker process, not at construction time.
    Requires ``persistent_workers=True`` and ``fork`` start method (Linux/CUDA
    default) — both enforced / documented by ``DataPackerDataLoader``.
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        seed: int,
        dp_rank: int,
        dp_world_size: int,
        shuffle: bool = True,
        name: str = "",
    ) -> None:
        super().__init__()
        self._dataset = dataset
        self._seed = seed
        self._dp_rank = dp_rank
        self._dp_world_size = dp_world_size
        self._shuffle = shuffle
        self._name = name

    def __len__(self) -> int:
        return len(self._dataset)  # type: ignore[arg-type]

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0

        stream_id = self._dp_rank * num_workers + worker_id
        total_streams = self._dp_world_size * num_workers
        n = len(self._dataset)  # type: ignore[arg-type]

        # os.environ.pop: consume once so a hypothetical second __iter__ call
        # in the same worker process defaults to a fresh-start sentinel instead of
        # re-fast-forwarding.  -1 means "no items seen yet" → start = 0.
        # DataLoaderStateCallback always saves index ≥ 0, so -1 is unambiguous.
        # When self._name is non-empty, env vars are namespaced to avoid conflicts
        # between multiple DataPackerDataLoader instances in the same process.
        _pfx = f"DP_STATE_{self._name}_" if self._name else "DP_STATE_"
        resume_epoch = int(os.environ.pop(f"{_pfx}WORKER_{worker_id}_EPOCH", 0))
        resume_pos = int(os.environ.pop(f"{_pfx}WORKER_{worker_id}_INDEX", -1))

        epoch = resume_epoch
        while True:
            if self._shuffle:
                g = torch.Generator().manual_seed(self._seed + epoch)
                perm = torch.randperm(n, generator=g).tolist()
            else:
                perm = list(range(n))
            stream_slice = perm[stream_id::total_streams]

            # resume_pos is the last index successfully included in a training
            # batch, so start one past it.  On new epochs start from 0.
            start = (resume_pos + 1) if epoch == resume_epoch else 0
            for pos in range(start, len(stream_slice)):
                item = self._dataset[stream_slice[pos]]
                # Attach position metadata; _DataPackerIterableDataset strips
                # these before sft_process_sample and re-attaches after so they
                # survive through the pool to collate_batch.
                yield {"_dp_epoch": epoch, "_dp_stream_pos": pos, **item}

            epoch += 1


class _DataPackerIterableDataset(PackingIterableDataset):
    """Private: injects a DataPacker into the shared packing engine.

    Not registered in Hydra directly.  Use ``DataPackerDataLoader`` instead.
    """

    def __init__(
        self,
        data_source: Any,
        data_packer: DataPacker,
        max_tokens: int,
        pool_size: int,
        max_batch_size: int,
        long_threshold: int,
        batching_strategy: str,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        shuffle: bool = False,
        seed: int = 0,
        name: str = "",
        apply_long_sample_halving: bool = True,
    ):
        is_map = isinstance(data_source, torch.utils.data.Dataset) and not isinstance(
            data_source, torch.utils.data.IterableDataset
        )
        is_iterable = isinstance(data_source, torch.utils.data.IterableDataset)
        if not is_map and not is_iterable:
            raise TypeError(
                f"data_source must be a torch.utils.data.Dataset or "
                f"torch.utils.data.IterableDataset, got {type(data_source).__name__}"
            )

        if is_map:
            # All map-style datasets go through _ShuffledMapIterableDataset,
            # which handles sharding and position metadata regardless of shuffle.
            # This enables stateful checkpoint/resume even when shuffle=False.
            data_source = _ShuffledMapIterableDataset(
                dataset=data_source,
                seed=seed,
                dp_rank=dp_rank,
                dp_world_size=dp_world_size,
                shuffle=shuffle,
                name=name,
            )
            self._has_dp_meta = True
        else:
            # Iterable-style: wrap with _IterableWrapper for sharding only.
            # Stateful resume is not supported for IterableDataset sources.
            data_source = _IterableWrapper(data_source, dp_rank=dp_rank, dp_world_size=dp_world_size)
            self._has_dp_meta = False

        datasets_cfg = {"default": {"dataset": data_source, "ratio": 1.0}}
        super().__init__(
            datasets_cfg=datasets_cfg,
            max_tokens=max_tokens,
            pool_size=pool_size,
            max_batch_size=max_batch_size,
            long_threshold=long_threshold,
            batching_strategy=batching_strategy,
            apply_long_sample_halving=apply_long_sample_halving,
        )
        self._data_packer = data_packer

    def _get_next_sample(self) -> dict:
        raw_item = super()._get_next_sample()
        if self._has_dp_meta:
            # Strip _dp_* keys before sft_process_sample so the user's packer
            # receives a clean item, then re-attach so metadata survives the pool.
            dp_meta = {k: raw_item.pop(k) for k in list(raw_item) if k.startswith("_dp_")}
            processed = self._data_packer.sft_process_sample(raw_item)
            processed.update(dp_meta)
            return processed
        return self._data_packer.sft_process_sample(raw_item)

    def compute_sample_tokens(self, sample: dict) -> int:
        return self._data_packer.compute_num_tokens(sample)

    def collate_batch(self, samples: list) -> dict:
        max_len = max(self.compute_sample_tokens(s) for s in samples)

        if self._has_dp_meta and "_dp_epoch" in samples[0]:
            max_epoch = max(s["_dp_epoch"] for s in samples)
            max_pos = max(s["_dp_stream_pos"] for s in samples)
            clean = [{k: v for k, v in s.items() if not k.startswith("_dp_")} for s in samples]
            batch = self._data_packer.sft_collate_fn(clean, max_len)
            worker_info = torch.utils.data.get_worker_info()
            worker_id = worker_info.id if worker_info is not None else 0
            batch["sample_worker_id"] = torch.tensor([worker_id] * len(samples))
            batch["sample_epoch"] = torch.tensor([max_epoch] * len(samples))
            batch["sample_index"] = torch.tensor([max_pos] * len(samples))
        else:
            batch = self._data_packer.sft_collate_fn(samples, max_len)

        return batch


class DataPackerDataLoader(torch.utils.data.DataLoader):
    """Public OSS entry point for bringing any dataset into i4 training.

    Wraps ``_DataPackerIterableDataset`` in a standard
    ``torch.utils.data.DataLoader`` — no WebDataset dependency required.
    OSS users' data can be HuggingFace datasets, local files, generators,
    or any Python iterable.

    Data-parallel sharding is automatic when ``torch.distributed`` is
    initialized.  Each ``(dp_rank, worker_id)`` pair receives a disjoint
    subset of ``data_source``.

    Parameters
    ----------
    data_source:
        ``torch.utils.data.Dataset`` (map-style) or
        ``torch.utils.data.IterableDataset`` — HuggingFace datasets, custom
        datasets, or generators wrapped in an ``IterableDataset``.  Plain
        lists / generators are not accepted; wrap them in an ``IterableDataset``
        first.
    data_packer:
        A ``DataPacker`` subclass instance.  Provides sample-level transform
        (``sft_process_sample``), token counting (``compute_num_tokens``), and
        batch collation (``sft_collate_fn``).
    max_tokens:
        Token budget per batch.
    pool_size:
        Samples to buffer before bin-packing.
    max_batch_size:
        Hard cap on items per batch.
    long_threshold:
        Samples with token count >= this are emitted as singleton batches.
    batching_strategy:
        ``"prefer_closest"`` (default) or ``"prefer_first"``.
    shuffle:
        If ``True`` and ``data_source`` is a map-style ``Dataset``, shuffle
        samples with a per-epoch ``torch.randperm`` seeded by ``seed + epoch``.
        Enables stateful checkpoint/resume via ``DataLoaderStateCallback``
        (``distributor_type="data_packer"``).  Has no effect for
        ``IterableDataset`` inputs — a warning is logged in that case.
    seed:
        Base seed for the per-epoch shuffle permutation.  Epoch ``e`` uses
        ``seed + e`` as the generator seed.  Ignored when ``shuffle=False``.
    num_workers, prefetch_factor, persistent_workers, pin_memory:
        Forwarded to ``torch.utils.data.DataLoader``.  When ``shuffle=True``
        and ``num_workers > 0``, ``persistent_workers`` is automatically
        promoted to ``True`` (required for correct resume behaviour).
    parallel_dims:
        Optional ``ParallelDims`` instance (from cosmos-rl).  When provided,
        ``parallel_dims.dp_coord`` supplies the data-parallel rank and world
        size, which is correct for FSDP+TP/PP where the DP degree differs from
        the global world size.  When ``None`` (default), rank info is read from
        ``torch.distributed`` if initialized, else defaults to ``(0, 1)``.
    name:
        Optional identifier used to namespace resume env vars when multiple
        ``DataPackerDataLoader`` instances share the same process (e.g. inside
        ``JointDataPackerDataLoader``).  When non-empty, env vars become
        ``DP_STATE_{name}_WORKER_{id}_EPOCH/INDEX`` instead of the default
        ``DP_STATE_WORKER_{id}_EPOCH/INDEX``.  Must match the ``name`` passed
        to the corresponding ``DataLoaderStateCallback`` or
        ``JointDataLoaderStateCallback``.  Leave empty (default) for
        single-loader configurations.
    apply_long_sample_halving:
        When ``True`` (default), the inner ``PackingIterableDataset._max_tokens``
        halves the budget for any batch whose largest sample has >= 1000 tokens
        — a memory-safety heuristic.  Set ``False`` to use the literal
        ``max_tokens`` budget unconditionally; only do this when memory
        headroom at the un-halved budget has been validated for the recipe
        (large MoT + LoRA recipes can OOM at the literal budget — see
        ``packing_iterable_dataset.py::_max_tokens``).
    """

    def __init__(
        self,
        data_source: Any,
        data_packer: DataPacker,
        max_tokens: int,
        pool_size: int = 16,
        max_batch_size: int = 1,
        long_threshold: int = 6400,
        batching_strategy: str = "prefer_closest",
        shuffle: bool = False,
        seed: int = 0,
        num_workers: int = 0,
        prefetch_factor: int | None = None,
        persistent_workers: bool = False,
        pin_memory: bool = False,
        parallel_dims=None,
        name: str = "",
        apply_long_sample_halving: bool = True,
    ):
        is_map = isinstance(data_source, torch.utils.data.Dataset) and not isinstance(
            data_source, torch.utils.data.IterableDataset
        )
        is_iterable = isinstance(data_source, torch.utils.data.IterableDataset)
        if shuffle and is_iterable:
            log.warning(
                "DataPackerDataLoader: shuffle=True has no effect for IterableDataset "
                "data_source. Shuffle the dataset before passing it in.",
                rank0_only=True,
            )

        # Correctness requirement: map-style datasets use _ShuffledMapIterableDataset
        # which reads resume env vars on the first __iter__ call inside each worker.
        # With persistent_workers=False, workers re-spawn each iteration and
        # re-inherit the env vars, causing incorrect fast-forward on every epoch
        # boundary. Enforce persistent_workers=True for all map-style datasets.
        if is_map and num_workers > 0 and not persistent_workers:
            log.info(
                "DataPackerDataLoader: map-style data_source requires persistent_workers=True "
                "for correct stateful resume behaviour. Overriding persistent_workers to True.",
                rank0_only=True,
            )
            persistent_workers = True

        # Resolve data-parallel rank and world-size.
        # Priority: explicit parallel_dims > torch.distributed > single-GPU default.
        if parallel_dims is not None:
            dp_rank, dp_world_size = parallel_dims.dp_coord
        elif torch.distributed.is_initialized():
            dp_rank = torch.distributed.get_rank()
            dp_world_size = torch.distributed.get_world_size()

            # rank/world_size differ from the data-parallel rank/world_size.
            # Pass `parallel_dims` to use the correct DP coordinates; otherwise
            # data sharding will be incorrect (each logical DP group sees the
            # same shard as another group).
            if dp_world_size > 1:
                log.info(
                    "DataPackerDataLoader: using global rank for DP sharding. "
                    "For FSDP+TP/PP setups pass parallel_dims= to use the correct "
                    "DP rank/world_size.",
                    rank0_only=True,
                )
        else:
            dp_rank, dp_world_size = 0, 1

        dataset = _DataPackerIterableDataset(
            data_source=data_source,
            data_packer=data_packer,
            max_tokens=max_tokens,
            pool_size=pool_size,
            max_batch_size=max_batch_size,
            long_threshold=long_threshold,
            batching_strategy=batching_strategy,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            shuffle=shuffle,
            seed=seed,
            name=name,
            apply_long_sample_halving=apply_long_sample_halving,
        )
        loader_kwargs: dict = dict(
            num_workers=num_workers,
            persistent_workers=persistent_workers and num_workers > 0,
            pin_memory=pin_memory,
        )
        if num_workers > 0 and prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = prefetch_factor
        # batch_size=None disables PyTorch's automatic batching/collation.
        # _DataPackerIterableDataset.__iter__ already yields fully-collated batch dicts;
        # letting the DataLoader re-collate them adds spurious batch dimensions.
        super().__init__(dataset, batch_size=None, **loader_kwargs)


class JointDataPackerDataLoader:
    """Wraps multiple ``DataPackerDataLoader`` instances with ratio-based seeded selection.

    Mirrors the design of ``IterativeJointDataLoader``: one output batch = one
    inner loader, selected deterministically by ratio at each step.  Adds a
    ``"dataset_name"`` key to every yielded batch so downstream callbacks can
    route state updates to the correct inner loader.

    Parameters
    ----------
    dataloaders:
        ``{name: {"dataloader": DataPackerDataLoader, "ratio": int}}`` mapping.
        Entries with ``ratio <= 0`` are silently skipped.
    seed:
        Base seed for the per-step dataset selection.  Step ``i`` uses
        ``np.random.RandomState(seed + i)`` to pick the inner loader index,
        giving the same sequence on every rank (assuming synchronized
        ``set_start_iteration`` calls) and fully reproducible resume.

    Stateful checkpoint/resume
    --------------------------
    Pair with ``JointDataLoaderStateCallback`` (from
    ``cosmos_framework.callbacks.dataloader_state``).  That callback saves the outer
    ``global_id`` and each inner loader's per-worker ``(epoch, index)`` state
    in a single DCP checkpoint entry.  On resume:

    1. ``JointDataLoaderStateCallback.load_state_dict`` calls
       ``set_start_iteration(global_id)`` to restore the selection sequence.
    2. Each inner ``DataLoaderStateCallback.load_state_dict`` sets namespaced
       env vars so inner-loader workers fast-forward to the saved position.

    Each ``DataPackerDataLoader`` must be constructed with a unique ``name``
    that matches the key used in this ``dataloaders`` dict so env vars are
    namespaced correctly (see ``DataPackerDataLoader`` ``name`` parameter).
    """

    def __init__(
        self,
        dataloaders: dict[str, dict],
        seed: int = 42,
    ) -> None:
        entries = [
            (name, cfg["dataloader"], cfg["ratio"])
            for name, cfg in dataloaders.items()
            if cfg.get("ratio", 0) > 0
        ]
        if not entries:
            raise ValueError("JointDataPackerDataLoader: no dataloaders with ratio > 0")

        self._names: list[str] = [e[0] for e in entries]
        if "global_id" in self._names:
            raise ValueError(
                "JointDataPackerDataLoader: dataset name 'global_id' is reserved "
                "by the checkpoint state format; use a different name."
            )
        self._loaders: list[DataPackerDataLoader] = [e[1] for e in entries]
        ratios = np.array([e[2] for e in entries], dtype=float)
        self._probs: np.ndarray = ratios / ratios.sum()
        self._seed = seed
        self._global_id = 0
        # Iterators are created lazily on the first __iter__ call so that
        # DataLoaderStateCallback.load_state_dict can install resume env vars
        # before workers are spawned (for num_workers > 0, iter(DataLoader)
        # forks workers immediately; env vars must be set in the parent first).
        self._iterators: list | None = None

        total = ratios.sum()
        lines = [f"JointDataPackerDataLoader: {len(self._names)} streams"]
        for name, ratio in zip(self._names, ratios):
            lines.append(f"  {name}: ratio={ratio:.4g} ({ratio / total:.1%})")
        log.info("\n".join(lines))

    def set_start_iteration(self, iteration: int) -> None:
        """Restore deterministic selection sequence after checkpoint resume.

        Called by ``JointDataLoaderStateCallback.load_state_dict`` and by the
        trainer (if present) via ``hasattr`` guard.
        """
        self._global_id = iteration

    def __iter__(self):
        # Lazy init: create iterators here (not in __init__) so that
        # load_state_dict can set resume env vars before workers fork.
        if self._iterators is None:
            self._iterators = [iter(loader) for loader in self._loaders]
        while True:
            rng = np.random.RandomState(self._seed + self._global_id)
            idx = int(rng.choice(len(self._loaders), p=self._probs))
            try:
                batch = next(self._iterators[idx])
            except StopIteration:
                # Inner DataPackerDataLoaders are infinite; this guard handles
                # the unlikely case of a finite IterableDataset inner source.
                self._iterators[idx] = iter(self._loaders[idx])
                batch = next(self._iterators[idx])
            batch["dataset_name"] = self._names[idx]
            self._global_id += 1
            yield batch
