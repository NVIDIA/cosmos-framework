# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""
Distributed checkpoint (DCP) directory structure and storage backends.

The checkpointer saves model state in a sharded format across multiple processes:

self.save_dirname/
├── iter_000000005/                   # Checkpoint at iteration 5
│   ├── model/                        # Model state shards
│   │   ├── __0_0.distcp              # Shard 0 from rank 0
│   │   └── __1_0.distcp              # Shard 1 from rank 1
│   ├── optim/                        # Optimizer state shards
│   │   ├── __0_0.distcp              # Shard 0 from rank 0
│   │   └── __1_0.distcp              # Shard 1 from rank 1
│   ├── scheduler/                    # Learning rate scheduler state
│   │   ├── __0_0.distcp              # Shard 0 from rank 0
│   │   └── __1_0.distcp              # Shard 1 from rank 1
│   └── trainer/                      # Additional training state
│       ├── __0_0.distcp              # Shard 0 from rank 0
│       └── __1_0.distcp              # Shard 1 from rank 1
│   └── dataloader/                   # Optional per-rank dataloader state
│       ├── rank_0.pkl
│       └── rank_1.pkl
└── latest_checkpoint.txt             # Points to most recent checkpoint folder, e.g. iter_000000005

Storage backends:
- Local filesystem:
  self.save_dirname = "{config_job.path_local}/checkpoints"

- S3 object store:
  self.save_dirname = "s3://{bucket}/{config_job.path}/checkpoints"
  where bucket = self.config_checkpoint.save_to_object_store.bucket

The sharded format enables efficient distributed saving/loading by:
1. Parallelizing I/O across processes
2. Reducing memory usage per process
3. Supporting both local and cloud storage backends
"""

import dataclasses
import enum
import multiprocessing
import os
import re
import time
from multiprocessing import get_context
from typing import Any, Dict, Optional, Protocol, Tuple, Union, runtime_checkable

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch import nn
from torch.distributed.checkpoint.default_planner import create_default_local_load_plan
from torch.distributed.checkpoint.filesystem import FileSystemReader, FileSystemWriter
from torch.distributed.checkpoint.metadata import (
    STATE_DICT_TYPE,
    Metadata,
    StorageMeta,
)
from torch.distributed.checkpoint.planner import LoadPlan
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    set_model_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.tensor import DTensor, Replicate
from torch.nn.modules.module import _IncompatibleKeys

from cosmos_framework.checkpoint.base import AbstractCheckpointer
from cosmos_framework.checkpoint.s3_filesystem import S3StorageReader, S3StorageWriter
from cosmos_framework.model._base import ImaginaireModel
from cosmos_framework.utils import callback, distributed, log, misc
from cosmos_framework.utils.config import CheckpointConfig, JobConfig
from cosmos_framework.utils.easy_io import easy_io
from cosmos_framework.utils.generator.rand_state import get_rand_state_dict, set_rand_state_dict


class ModelWrapper(Stateful):
    """
    Wrapper for model state dict handling. Strips away the _orig_mod. prefix
    among other things from the state dict keys.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def state_dict(self) -> dict[str, Any]:
        return get_model_state_dict(self.model)

    def load_state_dict(self, state_dict: dict[str, Any]) -> _IncompatibleKeys:
        return set_model_state_dict(
            self.model,
            model_state_dict=state_dict,
            options=StateDictOptions(strict=False),
        )


@runtime_checkable
class _DataloaderStateHandler(Protocol):
    """Structural contract for callbacks that participate in dataloader-state checkpointing."""

    checkpoint_component: str

    def has_checkpoint_state(self) -> bool: ...
    def state_dict(self) -> dict[Any, Any]: ...
    def load_state_dict(self, state_dict: dict[Any, Any]) -> None: ...


class _DataloaderWrapper:
    """Adapter that surfaces a dataloader-state callback's checkpoint API.

    Walks the registered callbacks at construction time and binds to the
    first callback that:

    1. Declares ``checkpoint_component == "dataloader"``, AND
    2. Returns ``True`` from ``has_checkpoint_state()``.

    The bound callback's ``state_dict`` / ``load_state_dict`` methods are
    re-exposed via :meth:`state_dict` / :meth:`load_state_dict`.  Callers
    must gate those on :meth:`has_state` — invoking them when nothing was
    bound raises :class:`RuntimeError`.

    Note: only the first callback tagged ``checkpoint_component=="dataloader"``
    is considered; if it does not currently want its state checkpointed,
    no further callbacks are searched.  In practice there is at most one
    such callback (see ``DataLoaderStateCallback``).
    """

    def __init__(self, callbacks: callback.CallBackGroup | None) -> None:
        self._callback: _DataloaderStateHandler | None = None
        if callbacks is None:
            return
        for current_callback in callbacks._callbacks:
            if getattr(current_callback, "checkpoint_component", None) != "dataloader":
                continue
            if current_callback.has_checkpoint_state():
                self._callback = current_callback
            return

    def has_state(self) -> bool:
        return self._callback is not None

    def state_dict(self) -> dict[Any, Any]:
        if self._callback is None:
            raise RuntimeError("No dataloader state handler is registered, cannot save dataloader state.")
        return self._callback.state_dict()

    def load_state_dict(self, state_dict: dict[Any, Any]) -> None:
        if self._callback is None:
            raise RuntimeError("No dataloader state handler is registered, cannot load dataloader state.")
        self._callback.load_state_dict(state_dict)


class AsyncMode(str, enum.Enum):
    DISABLED = "disabled"
    ASYNC_WITH_PINNED_MEM = "async_with_pinned_mem"


class Terminate:
    pass


class SaveDone:
    def __init__(self, iteration: int, elapsed_time: float, succeeded: bool):
        self.iteration = iteration
        self.elapsed_time = elapsed_time
        self.succeeded = succeeded

    def __str__(self):
        return f"SaveDone(iteration={self.iteration}, elapsed_time={self.elapsed_time}, succeeded={self.succeeded})"


def save_checkpoint_in_background(
    receiver_queue: multiprocessing.Queue,
    sender_queue: multiprocessing.Queue,
    config_checkpoint: CheckpointConfig,
    config_job: JobConfig,
) -> None:
    """
    Handles model checkpoint saving in a separate background process using PyTorch's distributed functionality.
    This function runs in a dedicated process to avoid blocking the main training loop.

    Args:
        receiver_queue: Queue to receive state dictionaries and commands from the main process
        sender_queue: Queue to send completion signals back to the main process
        config_checkpoint: Configuration settings for checkpoint saving behavior
        config_job: Configuration settings for the training job

    Flow:
        1. Initializes distributed processing environment
        2. Continuously waits for state dictionaries to save
        3. Saves checkpoints asynchronously
        4. Signals completion back to main process
        5. Terminates when receiving a Terminate signal

    Raises:
        AssertionError: If received object is neither Terminate signal nor valid state dict tuple

    Note:
        - Uses a different port than the main process to avoid conflicts
        - Disables TorchElastic agent store for checkpoint operations
        - Automatically cleans up distributed process group on exit
    """
    # Configure distributed environment
    os.environ["MASTER_PORT"] = str(int(os.environ["MASTER_PORT"]) + 2)
    os.environ["TORCHELASTIC_USE_AGENT_STORE"] = "False"

    # Set up GPU device and distributed processing
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    if dist.is_initialized():
        dist.destroy_process_group()
    dist.init_process_group(backend="gloo")

    # Initialize checkpointing mechanism
    checkpoint_handler = DistributedCheckpointer(
        config_checkpoint=config_checkpoint,
        config_job=config_job,
        callbacks=None,
        disable_async=True,
    )

    while True:
        log.info(f"Checkpoint background process is ready for next task, waiting for new state_dict")
        received_data = receiver_queue.get()
        log.info(f"Checkpoint background process received new state_dict")

        if isinstance(received_data, Terminate):
            log.info(f"Checkpoint background process received termination signal, closing sender queue")
            break

        assert isinstance(received_data, tuple), "Received data must be a tuple of (state_dict, checkpoint_path)"
        state_dict, checkpoint_path = received_data

        # Save checkpoint and measure time taken.
        start_time = time.monotonic()
        iteration = state_dict["trainer"][0]["iteration"]
        succeeded = False

        try:
            log.info(f"Saving checkpoint to {checkpoint_path}")
            checkpoint_handler.save_state_dict_worker(state_dict, checkpoint_path)
            succeeded = True
        except Exception as e:
            log.error(f"Error saving checkpoint to {checkpoint_path}: {e}")
            # continue because if the thread exits, the main thread keeps on adding to the queue
        finally:
            elapsed_time = time.monotonic() - start_time
            log.info(
                f"Checkpoint save completed in background process. "
                f"Time taken: {elapsed_time:.2f} seconds, iteration: {iteration}, "
                f"status: {'SUCCESS' if succeeded else 'FAILURE'}"
            )
            sender_queue.put(SaveDone(iteration, elapsed_time, succeeded))

    log.info("Cleaning up: destroying distributed process group")
    dist.destroy_process_group()


def _replace_keys_with_ema_keys(state_dict: STATE_DICT_TYPE) -> STATE_DICT_TYPE:
    """
    Renames model parameters from "net." to "net_ema.".
    """
    if not all(k.startswith("net.") for k in state_dict.keys()):
        raise ValueError("State dict must start with net. keys when load_ema_to_reg is True")
    return {k.replace("net.", "net_ema."): v for k, v in state_dict.items()}


class CustomLoadPlanner(dcp.DefaultLoadPlanner):
    """
    CustomLoadPlanner that supports ignoring keys during checkpoint load.
    This is useful when the checkpoint is saved with a different component
    architecture, e.g. different RoPE embeddings than the current model.

    Setting ``keys_to_skip_loading``: do not construct this planner directly. Set the experiment's
    checkpoint config field ``checkpoint.keys_to_skip_loading`` (a ``list[str]``; see
    :class:`cosmos_framework.utils.config.CheckpointConfig`) and the checkpointer forwards it here. Each entry is
    matched as a *substring* against every flattened fully-qualified name in the model/optimizer state
    dict, and any leaf whose fqn contains one of the entries is skipped. The list is only honored on a
    warm start (loading from a *different* run via ``checkpoint.load_path``); when resuming the latest
    checkpoint of the same run it is forced to ``[]``, since there is nothing to skip. Examples:

    1. Skip a reshaped positional embedding (different sequence length) for reg + EMA copies.
       checkpoint.keys_to_skip_loading=["net.latent_pos_embed.seq", "net_ema.latent_pos_embed.seq"]
    2. Skip action-head layers when warm-starting an action model from a non-action checkpoint.
       checkpoint.keys_to_skip_loading=["action2llm", "llm2action", "action_modality_embed", "action_pos_embed"]

    When ``dedup=True`` it additionally elects a single reader per replicated leaf to kill
    redundant storage reads. Two replication patterns waste reads at large world sizes:

    1. Fully-replicated leaves (e.g. optimizer scalar ``step`` tensors, replicated buffers) are
       saved on global rank 0 but appear in *every* rank's state dict, so all ``world_size`` ranks
       read the same object — a single-object hotspot.
    2. HSDP shards are identical across the ``dp_replicate`` dim, so each shard file is read by
       ``dp_replicate`` ranks.

    With ``dedup=True``, ``create_local_plan`` drops, from the read plan it builds, the items this
    rank is *not* the designated reader for, so each leaf is fetched from storage exactly once.

    ``create_local_plan`` always builds the read plan by calling ``create_default_local_load_plan``
    directly on ``self._skip_keys_if_found(self.state_dict)`` rather than delegating to
    ``super().create_local_plan()``. This serves two ends:

    1. It drops ``keys_to_skip_loading`` *before* plan construction, so skipped keys never reach the
       base helper's strict missing-key validation or its *unconditional* size-mismatch check.
       Skipping therefore works with ``strict_resume=True``, with a skipped key absent from the
       checkpoint, and with a skipped key present-but-reshaped (e.g. different RoPE embeddings).
    2. It avoids the base ``create_local_plan``'s pre-2.4 "missing keys" fallback, which would
       re-derive ``self.state_dict`` from the full ``original_state_dict`` and reinstate any skipped
       key still present in the checkpoint. Dropping that fallback is acceptable -- this repo does
       not load pre-2.4 checkpoints.

    ``self.state_dict`` is never mutated (``_skip_keys_if_found`` returns a pruned *copy*, or the dict
    unchanged when nothing matches), so the loader still writes non-tensor leaves back into the caller's
    state dict in place.

    Dedup removes reads at a different point: ``create_local_plan`` filters the already-built plan's
    read items via ``_drop_non_reader_items``, looking each leaf up in the full ``self.state_dict``.

    - ``DTensor`` leaf -> reader is local rank 0 along the tensor's replicate mesh dim (one reader
      per replicate group; a fully-sharded tensor has none, so every rank reads its own shard);
    - non-``DTensor`` leaf (plain tensor / python scalar) -> fully replicated across the world, so
      the reader is global rank 0.

    Reading the replicate dim from each tensor's *own* mesh (rather than one global ``dp_replicate``
    group) generalizes to tensors on different meshes, e.g. MoE expert weights under expert
    parallelism. The dropped data is filled in afterwards by :func:`_broadcast_state_dict`.
    Because each rank then reads a disjoint subset, dedup requires ``no_dist=True`` (there is no
    global plan to coordinate).
    """

    def __init__(
        self,
        flatten_state_dict: bool = True,
        flatten_sharded_tensors: bool = True,
        allow_partial_load: bool = False,
        keys_to_skip_loading: list[str] | None = None,
        load_ema_to_reg: bool = False,
        dedup: bool = False,
        global_rank: int = 0,
    ) -> None:
        super().__init__(
            flatten_state_dict=flatten_state_dict,
            flatten_sharded_tensors=flatten_sharded_tensors,
            allow_partial_load=allow_partial_load,
        )

        # Default to [] without a mutable default argument (which would be shared across instances).
        self.keys_to_skip_loading = keys_to_skip_loading or []
        self.load_ema_to_reg = load_ema_to_reg
        # When set, prune non-reader leaves so each replicated leaf is read by exactly one rank.
        self.dedup = dedup
        self._global_rank = global_rank

        if len(self.keys_to_skip_loading) > 0:
            log.info(f"Skipping loading of keys that match the following patterns: {self.keys_to_skip_loading}")

    def set_up_planner(
        self,
        state_dict: STATE_DICT_TYPE,
        metadata: Metadata | None = None,
        is_coordinator: bool = False,
    ) -> None:
        if self.load_ema_to_reg:
            state_dict = _replace_keys_with_ema_keys(state_dict)

        super().set_up_planner(
            state_dict=state_dict,
            metadata=metadata,
            is_coordinator=is_coordinator,
        )

    def create_local_plan(self) -> LoadPlan:
        # Build the read plan from a pruned *copy* of self.state_dict so skipped keys never
        # reach create_default_local_load_plan, which would otherwise (a) raise a missing-key
        # error for a skipped key absent from the checkpoint under strict load
        # (allow_partial_load=False), and (b) raise an *unconditional* size-mismatch ValueError
        # for a skipped key that is present in the checkpoint but reshaped.
        if self.metadata is None:
            raise AssertionError("metadata must be set (via set_up_planner) before create_local_plan")

        plan = create_default_local_load_plan(
            state_dict=self._skip_keys_if_found(self.state_dict),
            metadata=self.metadata,
            strict=not self.allow_partial_load,
        )

        return self._drop_non_reader_items(plan)

    def _drop_non_reader_items(self, plan: LoadPlan) -> LoadPlan:
        """Under dedup load, drop the read items this rank is not the elected reader for.

        ``self.state_dict`` is the full flattened fqn -> leaf mapping at this point, so we look
        each item's leaf up by fqn to decide readership. The dropped reads are filled afterward by
        :func:`_broadcast_state_dict`. No-op unless ``self.dedup`` is set.
        """
        if not self.dedup:
            return plan

        kept_items = [
            item
            for item in plan.items
            if _is_assigned_reader(self.state_dict.get(item.dest_index.fqn), self._global_rank)
        ]
        dropped = len(plan.items) - len(kept_items)
        log.info(
            f"[DCP-LOAD-DEDUP] kept_read_items={len(kept_items)} dropped_read_items={dropped}",
            rank0_only=False,
        )
        return dataclasses.replace(plan, items=kept_items)

    def _skip_keys_if_found(self, state_dict: STATE_DICT_TYPE) -> STATE_DICT_TYPE:
        """Return a *copy* of ``state_dict`` without keys whose fqn contains any ``keys_to_skip_loading`` substring.

        ``create_local_plan`` feeds this pruned copy to ``create_default_local_load_plan`` so skipped
        keys are kept out of the read plan, the base planner's strict missing-key validation, and its
        unconditional size-mismatch check (skipping therefore works with ``strict_resume=True``, with a
        skipped key absent from the checkpoint, and with a skipped key present-but-reshaped). The caller
        never mutates ``self.state_dict`` with this result, so the loader still writes non-tensor leaves
        back into the caller's state dict in place. No-op when no skip patterns are configured.
        """
        if len(self.keys_to_skip_loading) == 0:
            return state_dict

        kept = {}
        for fqn, obj in state_dict.items():
            if any(skip_key in fqn for skip_key in self.keys_to_skip_loading):
                log.warning(f"Skipping loading of key: {fqn}")
                continue
            kept[fqn] = obj
        log.info(
            f"[DCP-LOAD-SKIP-KEYS] kept_keys={len(kept)} dropped_keys={len(state_dict) - len(kept)}",
            rank0_only=False,
        )
        return kept


def _is_assigned_reader(value: Any, global_rank: int) -> bool:
    """Whether this rank is the single elected reader for ``value`` under dedup load.

    Used by :meth:`CustomLoadPlanner.create_local_plan` to drop the read items a rank is not
    responsible for, and mirrored by :func:`_broadcast_state_dict` to fill those dropped leaves back
    in. The election rule depends on how the leaf is replicated:

    - ``DTensor`` -> reader iff this rank sits at local rank 0 along EVERY (size>1) replicate mesh
      dim (see :func:`_replicate_mesh_dims`). A fully-sharded tensor has no replicate dims, so the
      ``all(...)`` over an empty list is ``True`` and every rank reads its own (unique) shard. A
      fully-replicated leaf (e.g. an optimizer ``step`` with ``[Replicate(), Replicate()]``) elects
      exactly one reader instead of a whole ``dp_shard`` line.
    - any other leaf (plain replicated tensor or python scalar) -> fully replicated across the world,
      so global rank 0 is the sole reader.
    """
    if isinstance(value, DTensor):
        dims = _replicate_mesh_dims(value)
        return all(value.device_mesh.get_local_rank(dim) == 0 for dim in dims)
    return global_rank == 0


def _replicate_mesh_dims(value: DTensor) -> list[int]:
    """Mesh-dim indices (ascending) along which a ``DTensor``'s local shard is replicated.

    A ``DTensor``'s local shard is byte-identical across every mesh dim whose placement is
    ``Replicate()``; it differs along ``Shard``/``Partial`` dims. Reading those dims' data once and
    broadcasting along them is therefore lossless. Deriving this from the tensor's *own* mesh (not a
    single global ``dp_replicate`` group) is what lets the dedup generalize to tensors on other
    meshes — e.g. MoE expert weights on a future expert-parallel mesh whose replicate dim differs.

    Mesh dims of **size 1** are excluded: they carry no replication, and — critically — every rank
    is local rank 0 along a size-1 dim, so treating one as a replicate dim would make the
    single-reader election (see :func:`_is_assigned_reader`) pass on *every* rank
    and silently disable dedup along the real replicate dim. This is exactly the failure for a
    fully-replicated leaf (e.g. an optimizer ``step`` with ``[Replicate(), Replicate()]``) on the
    2-D ``(dp_replicate=1, dp_shard=N)`` FSDP mesh: without this filter the size-1 ``dp_replicate``
    dim is picked, so all ``N`` ranks read the single object in ``__0_0.distcp`` instead of just
    global rank 0.

    Returns multiple dims for a leaf replicated across several mesh dims (e.g. a fully-replicated
    tensor on an HSDP ``(dp_replicate>1, dp_shard)`` mesh); empty when the tensor is fully sharded
    (no replication, so every rank legitimately reads its own shard).
    """
    mesh = value.device_mesh
    return [i for i, placement in enumerate(value.placements) if isinstance(placement, Replicate) and mesh.size(i) > 1]


def _broadcast_tensor_leaf(value: torch.Tensor) -> None:
    """Broadcast a tensor leaf from its single reader to the ranks that share it.

    Handles the two tensor leaf kinds the dedup planner drops (see
    :func:`_is_assigned_reader`):

    - ``DTensor`` -> broadcast the local shard across the tensor's replicate mesh dims. The reader
      is the rank at local rank 0 along every (size>1) replicate dim; the local shard is
      byte-identical across those dims, so broadcasting from that rank is lossless. Groups are taken
      from the tensor's *own* mesh so this works regardless of which mesh the tensor lives on (e.g. a
      future expert-parallel mesh). Dims are broadcast in ascending order so each step's source
      already holds valid data: after broadcasting along dim ``d``, every rank that is local rank 0
      along the remaining (higher) replicate dims is populated, which is exactly the source set the
      next broadcast needs. A fully-sharded ``DTensor`` (no replicate dims) needs no broadcast — every
      rank already read its own shard.
    - plain (non-``DTensor``) tensor -> fully replicated across the world and read by global rank 0
      only, so broadcast it over the default (world) group with ``src=0``. This covers replicated
      ``param_groups`` tensors such as a capturable optimizer ``step``.
    """
    if not torch.is_tensor(value):
        raise ValueError(f"Unsupported type: {type(value)}")
    if isinstance(value, DTensor):
        dims = _replicate_mesh_dims(value)
        if not dims:
            # Fully sharded across its mesh: every rank already read its own shard.
            return
        local = value.to_local()
        for dim in dims:
            group = value.device_mesh.get_group(dim)
            dist.broadcast(local, group=group, group_src=0)
    else:
        dist.broadcast(value, src=0)


def _broadcast_state_dict(
    state_dict: STATE_DICT_TYPE,
    global_rank: int,
) -> None:
    """Broadcast every leaf from its single reader (see :class:`CustomLoadPlanner` dedup mode) to
    the ranks that share it, filling in the reads that the planner dropped.

    Must be called by ALL ranks in lockstep, *after* a dedup ``dcp.load`` and *before* the
    ``load_state_dict`` that consumes ``state_dict``. Each leaf's replication is a
    structural property (identical on every rank), so all ranks walk the (identically ordered) tree
    and issue the same sequence of collectives:

    - ``DTensor`` leaf -> broadcast along its mesh's replicate dims (src = local rank 0 there); a
      fully-sharded tensor needs no broadcast;
    - non-``DTensor`` tensor leaf (e.g. a replicated ``param_groups`` ``step`` tensor) -> replicated
      across the world, broadcast (src = global rank 0);
    - non-tensor leaf (python scalars, e.g. optimizer ``param_groups`` ``lr``/``betas``/``eps``/
      ``step``) -> a single world ``broadcast_object_list`` of the tensor-stripped skeleton, merged
      back in.

    Reader election here mirrors :func:`_is_assigned_reader` exactly so that the
    set of leaves broadcast is precisely the set the planner dropped: DTensors elect a per-mesh
    reader; every other leaf (plain tensor or scalar) is read by global rank 0 only.
    """

    # 1) Broadcast tensor leaves in place. Keys are sorted by repr() so the traversal order is
    #    identical across ranks regardless of key type (int param ids vs str fqns). DTensors go over
    #    their own mesh's replicate dims; plain (non-DTensor) tensors are world-replicated and read by
    #    global rank 0 only, so they broadcast from global src 0. Non-tensor leaves are skipped here
    #    and handled by the object broadcast in step 2.
    def _walk_tensors(obj: Any) -> None:
        if torch.is_tensor(obj):
            _broadcast_tensor_leaf(obj)
        elif isinstance(obj, dict):
            for k in sorted(obj.keys(), key=repr):
                _walk_tensors(obj[k])
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                _walk_tensors(v)
        # else: non-tensor scalar leaf -> handled in step 2.

    start_time_tensor_broadcast = time.monotonic()
    _walk_tensors(state_dict)
    end_time_tensor_broadcast = time.monotonic()

    # 2) Broadcast all non-tensor leaves as one tensor-stripped skeleton (world, src = rank 0).
    #    Tensors (already broadcast in step 1) are stripped to None so they are not pickled; every
    #    other value (python scalars, lists, None, ...) rides along in the object broadcast.
    def _strip(obj: Any) -> Any:
        if torch.is_tensor(obj):
            return None
        if isinstance(obj, dict):
            return {k: _strip(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(_strip(v) for v in obj)
        return obj

    start_time_object_broadcast = time.monotonic()
    box = [_strip(state_dict) if global_rank == 0 else None]
    dist.broadcast_object_list(box, src=0)
    skeleton = box[0]
    end_time_object_broadcast = time.monotonic()

    # 3) Rebuild: keep the (already-broadcast) tensors in place, take non-tensors from rank 0.
    def _rebuild(local_obj: Any, skeleton_obj: Any) -> Any:
        if torch.is_tensor(local_obj):
            return local_obj
        if isinstance(local_obj, dict):
            return {k: _rebuild(local_obj[k], skeleton_obj[k]) for k in local_obj}
        if isinstance(local_obj, list):
            return [_rebuild(lv, sv) for lv, sv in zip(local_obj, skeleton_obj)]
        if isinstance(local_obj, tuple):
            return tuple(_rebuild(lv, sv) for lv, sv in zip(local_obj, skeleton_obj))
        return skeleton_obj

    start_time_rebuild = time.monotonic()
    rebuilt = _rebuild(state_dict, skeleton)
    state_dict.clear()
    state_dict.update(rebuilt)
    end_time_rebuild = time.monotonic()

    log.info(
        "[DCP-LOAD-TIMING] broadcast: "
        f"tensors={end_time_tensor_broadcast - start_time_tensor_broadcast:.1f}s, "
        f"objects={end_time_object_broadcast - start_time_object_broadcast:.1f}s, "
        f"rebuild={end_time_rebuild - start_time_rebuild:.1f}s",
        rank0_only=False,
    )


def _copy_ema_weights_to_reg(state_dict: STATE_DICT_TYPE) -> None:
    """Copy EMA weights to regular weights."""
    for sd_key in list(state_dict.keys()):
        if sd_key.startswith("net."):
            key_ema = "net_ema." + sd_key.removeprefix("net.")
            assert key_ema in state_dict, (
                f"EMA key {key_ema} not found in state_dict. Ensure the model has net_ema submodule."
            )
            state_dict[sd_key] = state_dict[key_ema]


class CustomSavePlanner(dcp.DefaultSavePlanner):
    """
    Custom save planner that enables an override for cache_plans_key when
    caching of save plans is enabled. Caching of save plans reduces checkpointing
    time by reusing the same save plan across checkpoints. This reduces the
    checkpointing time by ~60% (benchmarked using the 235B-A22B Qwen3-VL model
    on 64 GB200 nodes).
    """

    def __init__(
        self,
        flatten_state_dict: bool = True,
        flatten_sharded_tensors: bool = True,
        dedup_save_to_lowest_rank: bool = False,
        save_reg_to_ema: bool = False,
        enable_plan_caching: bool = False,
        cache_plans_key: str | None = None,
    ) -> None:
        super().__init__(
            flatten_state_dict=flatten_state_dict,
            flatten_sharded_tensors=flatten_sharded_tensors,
            dedup_save_to_lowest_rank=dedup_save_to_lowest_rank,
            enable_plan_caching=enable_plan_caching,
        )
        if cache_plans_key is not None:
            self._cached_plans_key = cache_plans_key

        self.save_reg_to_ema = save_reg_to_ema

    def set_up_planner(
        self,
        state_dict: STATE_DICT_TYPE,
        storage_meta: StorageMeta | None = None,
        is_coordinator: bool = False,
    ) -> None:
        if self.save_reg_to_ema:
            state_dict = _replace_keys_with_ema_keys(state_dict)

        super().set_up_planner(
            state_dict=state_dict,
            storage_meta=storage_meta,
            is_coordinator=is_coordinator,
        )


class DistributedCheckpointer(AbstractCheckpointer):
    CHECKPOINT_KEYS = ["model", "optim", "scheduler", "trainer", "dataloader"]

    def __init__(
        self,
        config_checkpoint: CheckpointConfig,
        config_job: JobConfig,
        callbacks: Optional[callback.CallBackGroup] = None,
        disable_async: bool = False,
    ):
        super().__init__(config_checkpoint, config_job, callbacks)
        self.config_checkpoint = config_checkpoint
        if config_checkpoint.load_ema_to_reg and config_checkpoint.load_training_state:
            raise ValueError(
                "load_ema_to_reg=True requires load_training_state=False. "
                "Loading optimizer/EMA state after copying EMA->reg weights produces an "
                "inconsistent checkpoint: the optimizer moments would track the original "
                "reg-weight trajectory, not the EMA weights just copied in."
            )
        if config_checkpoint.dcp_async_mode_enabled and not disable_async:
            self.async_mode = AsyncMode.ASYNC_WITH_PINNED_MEM
        else:
            self.async_mode = AsyncMode.DISABLED

        if self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            ctx = get_context("spawn")
            self.mp_queue_send = ctx.Queue()
            self.mp_queue_recv = ctx.Queue()
            self.mp = ctx.Process(
                target=save_checkpoint_in_background,
                args=(
                    self.mp_queue_send,
                    self.mp_queue_recv,
                    config_checkpoint,
                    config_job,
                ),
                daemon=True,
            )
            self.mp.start()
            self.cpu_offload_state_dict = None
            self.staging_ckpt_file = None
            self.staging_stream = torch.cuda.Stream()
            self.checkpoint_in_progress = False

    def keys_to_resume_during_load(self) -> tuple[set[str], str | None, bool | None]:
        """
        Determines the keys to resume from the checkpoint and the checkpoint path.
        If the checkpoint is the latest checkpoint of the same model, then it is a
        normal resume. If the checkpoint is a different model's checkpoint, then it is
        a warm start.

        Args:
            None

        Returns:
            resume_keys: The keys to resume from the checkpoint.
            checkpoint_path: The path to the checkpoint. If the checkpoint is a different
            warm_start: Whether to warm start the training from a different model's checkpoint.
                If the checkpoint is a different model's checkpoint, then this is True.
                If the checkpoint is the latest checkpoint of the same model, then this is False.
        """
        latest_checkpoint_file = self._read_latest_checkpoint_file()

        resume_keys = []
        warm_start = None

        if latest_checkpoint_file is not None:
            # 1. Resume training from the latest checkpoint of the same model.
            warm_start = False
            checkpoint_path = os.path.join(self.load_dirname, latest_checkpoint_file)
            resume_keys.extend(self.CHECKPOINT_KEYS)

        else:
            if self.load_path and not str(self.load_path).endswith(".pt"):
                # 2. Warm Start: Resume training from a different model's checkpoint
                # specified by `load_path`.
                warm_start = True
                checkpoint_path = self.load_path

                if self.load_s3_backend_key:
                    checkpoint_path = f"s3://{self.config_checkpoint.load_from_object_store.bucket}/{checkpoint_path}"

                    # If the path doesn't end with specific checkpoint, read the latest
                    # checkpoint file to determine the most recent checkpoint iteration.
                    if not re.search(r"/checkpoints/iter_\d{9}/?$", checkpoint_path):
                        old_ckpt_path = checkpoint_path
                        latest_ckpt_path = os.path.join(checkpoint_path, "checkpoints/latest_checkpoint.txt")

                        # If the latest checkpoint file exists, use it to determine the
                        # checkpoint path. Otherwise, use the original path.
                        if easy_io.exists(latest_ckpt_path, backend_key=self.load_s3_backend_key):
                            checkpoint_file = easy_io.load(
                                latest_ckpt_path, backend_key=self.load_s3_backend_key
                            ).strip()
                            checkpoint_path = f"{checkpoint_path}/checkpoints/{checkpoint_file}"
                        else:
                            log.warning(
                                f"Latest checkpoint file {latest_ckpt_path} not found, load from {old_ckpt_path}"
                            )
                            checkpoint_path = old_ckpt_path

                if self.load_training_state:
                    resume_keys.extend(self.CHECKPOINT_KEYS)
                else:
                    resume_keys.append("model")
                    if self.only_load_scheduler_state:
                        resume_keys.append("scheduler")
            else:
                checkpoint_path = None

        if len(self.keys_not_to_resume) > 0:
            for key in self.keys_not_to_resume:
                assert key in self.CHECKPOINT_KEYS, f"Invalid key to resume: {key} not in {self.CHECKPOINT_KEYS}"
            resume_keys = [key for key in resume_keys if key not in self.keys_not_to_resume]

        return set(resume_keys), checkpoint_path, warm_start

    @misc.timer("checkpoint loading")
    def load(
        self,
        model: ImaginaireModel,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        grad_scaler: torch.amp.GradScaler | None = None,
    ) -> int:
        if self.callbacks is not None:
            self.callbacks.on_load_checkpoint_start(model)

        resume_keys, checkpoint_path, warm_start = self.keys_to_resume_during_load()
        resume_keys = sorted(resume_keys)
        log.critical(f"Resuming ckpt {checkpoint_path} with keys: {resume_keys}")

        iteration = 0
        global_rank = dist.get_rank() if dist.is_initialized() else 0

        if checkpoint_path is not None:
            self._check_checkpoint_exists(checkpoint_path)

            for key in resume_keys:
                dist.barrier()

                cur_key_ckpt_full_path = os.path.join(checkpoint_path, key)
                log.critical(f"Start loading checkpoint from {cur_key_ckpt_full_path}")

                storage_reader = self.get_storage_reader(cur_key_ckpt_full_path)
                strict_resume = self.config_checkpoint.strict_resume

                # Note that we only allow skipping loading of keys during warm start. If the checkpoint is
                # the latest checkpoint of the same model, then we don't need to skip any keys.
                keys_to_skip_loading = self.config_checkpoint.keys_to_skip_loading if warm_start else []

                # Dedup-load context: when enabled, replicated state is read by a single rank and
                # broadcast over the device mesh instead of being read redundantly by every rank. The
                # per-tensor replicate group is derived from each DTensor's own mesh (see
                # _broadcast_state_dict). Applied only to the "model" and "optim" components
                # (the bulk of the data); the per-rank-unique "dataloader" state and
                # "trainer" RNG and the tiny "scheduler" state stay on the normal read-everywhere path.
                if key in ("model", "optim"):
                    load_planner = CustomLoadPlanner(
                        allow_partial_load=not strict_resume,
                        keys_to_skip_loading=keys_to_skip_loading,
                        dedup=self.config_checkpoint.dcp_load_dedup,
                        global_rank=global_rank,
                    )
                else:
                    load_planner = dcp.DefaultLoadPlanner(allow_partial_load=not strict_resume)

                if key == "model":
                    log.info("- Loading the model...")
                    _model_wrapper = ModelWrapper(model)
                    _state_dict = _model_wrapper.state_dict()
                    dcp.load(
                        _state_dict,
                        storage_reader=storage_reader,
                        planner=load_planner,
                        no_dist=True,
                    )
                    if self.config_checkpoint.dcp_load_dedup:
                        # Fill in the reads the planner dropped by broadcasting from each leaf's
                        # single reader. Must run before the EMA copy and load_state_dict below.
                        _broadcast_state_dict(_state_dict, global_rank)
                    if self.config_checkpoint.load_ema_to_reg and warm_start:
                        # The model has both net.* and net_ema.* submodules, so _state_dict
                        # contains both sets of keys after dcp.load(). Copy EMA weights into
                        # regular model weights so we can warm-start from EMA (and reset EMA
                        # when load_training_state=False).
                        #
                        # This must only run on warm start. During regular auto-resume
                        # (warm_start=False) the full training state is always reloaded, so
                        # copying EMA->reg here would silently overwrite the resumed regular
                        # weights with the (lagging) EMA snapshot on every restart while the
                        # optimizer state still tracks the pre-copy trajectory -- corrupting
                        # training in a way that depends on how often the job is preempted.
                        _copy_ema_weights_to_reg(_state_dict)

                    results = _model_wrapper.load_state_dict(_state_dict)
                    if len(results.missing_keys) > 0:
                        raise ValueError(f"Missing keys (not found in checkpoint): {results.missing_keys}")
                    if len(results.unexpected_keys) > 0:
                        raise ValueError(
                            f"Unexpected keys (found in checkpoint but not in model): {results.unexpected_keys}"
                        )
                    # Warm start that skipped net_ema (e.g. loading an EMA-only HF export
                    # with no net_ema.* keys): the EMA shadow would otherwise keep its random
                    # build-time generation pathway (init_moe is skipped when a checkpoint is
                    # present). Seed net_ema from the freshly loaded net so the EMA starts equal
                    # to net ("EMA warm-starts from net") instead of from random weights.
                    if warm_start and any("net_ema" in skip_key for skip_key in keys_to_skip_loading):
                        ema_worker = getattr(model, "net_ema_worker", None)
                        if ema_worker is not None and getattr(model, "net_ema", None) is not None:
                            ema_worker.copy_to(src_model=model.net, tgt_model=model.net_ema)
                            log.info("Warm start: re-seeded net_ema from net (net_ema was skipped on load).")

                elif key == "optim":
                    log.info("- Loading the optimizer...")
                    _state_dict = optimizer.state_dict()
                    dcp.load(
                        _state_dict,
                        storage_reader=storage_reader,
                        planner=load_planner,
                        no_dist=True,
                    )
                    if self.config_checkpoint.dcp_load_dedup:
                        # Fill in the reads the planner dropped by broadcasting from each leaf's
                        # single reader. Must run before load_state_dict below.
                        _broadcast_state_dict(_state_dict, global_rank)
                    optimizer.load_state_dict(_state_dict)

                elif key == "scheduler":
                    log.info("- Loading the scheduler...")
                    _state_dict = scheduler.state_dict()
                    dcp.load(
                        _state_dict,
                        storage_reader=storage_reader,
                        planner=load_planner,
                        no_dist=True,
                    )
                    scheduler.load_state_dict(_state_dict)

                elif key == "trainer":
                    log.info("- Loading the trainer...")

                    # Use rank-specific key for RNG state to support correct per-rank restoration
                    rng_key = f"rng_state_{dist.get_rank()}"
                    current_rng_state = get_rand_state_dict()
                    _state_dict = {
                        "grad_scaler": grad_scaler.state_dict(),
                        "iteration": iteration,
                    }
                    # Check if rng_key exists in checkpoint metadata to avoid failure with strict_resume=True
                    metadata = storage_reader.read_metadata()
                    rng_key_exists = any(
                        k.startswith(f"{rng_key}.") or k == rng_key for k in metadata.state_dict_metadata.keys()
                    )
                    if rng_key_exists:
                        _state_dict[rng_key] = current_rng_state

                    dcp.load(
                        _state_dict,
                        storage_reader=storage_reader,
                        planner=load_planner,
                        no_dist=True,
                    )
                    grad_scaler.load_state_dict(_state_dict["grad_scaler"])
                    iteration = _state_dict["iteration"]
                    set_rand_state_dict(_state_dict.get(rng_key, current_rng_state))

                elif key == "dataloader":
                    if not easy_io.exists(cur_key_ckpt_full_path, backend_key=self.load_s3_backend_key):
                        log.info(
                            f"Checkpoint {cur_key_ckpt_full_path} does not exist, skip loading dataloader.",
                            rank0_only=False,
                        )
                        continue

                    rank = dist.get_rank()
                    dataloader_pkl_path = os.path.join(cur_key_ckpt_full_path, f"rank_{rank}.pkl")
                    if not easy_io.exists(dataloader_pkl_path, backend_key=self.load_s3_backend_key):
                        log.info(f"No dataloader checkpoint found at {dataloader_pkl_path}", rank0_only=False)
                        continue

                    log.info(f"- Loading the dataloader {cur_key_ckpt_full_path}...", rank0_only=False)
                    _state_dict = easy_io.load(
                        dataloader_pkl_path,
                        file_format="pkl",
                        backend_key=self.load_s3_backend_key,
                    )
                    dataloader_wrapper = _DataloaderWrapper(self.callbacks)
                    if dataloader_wrapper.has_state():
                        dataloader_wrapper.load_state_dict(_state_dict)

                else:
                    raise ValueError(f"Invalid key: {key}. not support to resume.")

            if self.callbacks is not None and resume_keys:
                # Note that this callback is never used in the codebase.
                self.callbacks.on_load_checkpoint(model, state_dict={})
            log.info(f"Loaded checkpoint from {checkpoint_path} in iteration {iteration}")

        else:
            log.info("Training from scratch.")

        torch.cuda.empty_cache()

        if self.callbacks is not None:
            self.callbacks.on_load_checkpoint_end(model, iteration=iteration, checkpoint_path=checkpoint_path)
        return iteration

    def _checkpoint_async_with_pinned_memory(
        self, checkpoint_file: str, state_dict: Dict[str, Tuple[Any, str]]
    ) -> None:
        assert self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM, "Async mode must be AsyncMode.ASYNC_WITH_PINNED_MEM"

        from torch.distributed._state_dict_utils import _copy_state_dict, _create_cpu_state_dict

        if self.cpu_offload_state_dict is None:
            log.info(f"Preparing the CPU memory for staging")
            self.cpu_offload_state_dict = _create_cpu_state_dict(state_dict, pin_memory=True, share_memory=True)

        log.info(f"Staging the state_dict in CPU memory")
        with torch.cuda.stream(self.staging_stream):
            self.cpu_offload_state_dict = _copy_state_dict(
                state_dict,
                self.cpu_offload_state_dict,
                non_blocking=True,
            )
            self.staging_ckpt_file = checkpoint_file

        self.staging_stream.synchronize()
        log.info(f"Staging the state_dict in CPU memory completed")

        self.mp_queue_send.put_nowait((self.cpu_offload_state_dict, self.staging_ckpt_file))
        self.checkpoint_in_progress = True
        log.info(f"Submitted checkpoint to background process")

    def _wait_for_previous_async_checkpoint(self) -> None:
        """
        Gets the results of previously submitted checkpoints.
        Pass them to callbacks if checkpoint succeeded.
        """
        assert self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM, "Async mode must be AsyncMode.ASYNC_WITH_PINNED_MEM"

        if not self.checkpoint_in_progress:
            return

        success = False
        try:
            log.info(f"Waiting for checkpoint save result")

            # Note that we set a timeout of 1 hour to avoid blocking the main process
            # indefinitely. Gloo and NCCL timeouts are ~30 minutes, so this timeout
            # should typically be sufficient.
            save_done: SaveDone = self.mp_queue_recv.get(timeout=3600)

            log.info(f"Received checkpoint save result: {save_done}")

            if self.callbacks is not None and save_done.succeeded:
                self.callbacks.on_save_checkpoint_success(
                    iteration=save_done.iteration, elapsed_time=save_done.elapsed_time
                )
            self.checkpoint_in_progress = False
            success = save_done.succeeded

        except Exception as e:
            log.error(f"Error waiting for checkpoint save result: {e}")

        if not success:
            # Terminate training execution upon a failed checkpoint save attempt.
            # A failure at this stage typically indicates a non-recoverable system error.
            # Continuing execution would result in subsequent persistent failures and
            # unnecessary waste of GPU resources.
            raise RuntimeError("Previous checkpoint save failed. Exiting...")

    def get_storage_writer(self, checkpoint_path: str) -> Union[S3StorageWriter, FileSystemWriter]:
        if self.save_to_object_store:
            return S3StorageWriter(
                credential_path=self.config_checkpoint.save_to_object_store.credentials,
                path=checkpoint_path,
                enable_gcs_patch_in_boto3=self.config_checkpoint.enable_gcs_patch_in_boto3,
            )
        return FileSystemWriter(path=checkpoint_path)

    def get_storage_reader(self, checkpoint_path: str) -> Union[S3StorageReader, FileSystemReader]:
        if self.load_from_object_store:
            return S3StorageReader(
                credential_path=self.config_checkpoint.load_from_object_store.credentials,
                path=checkpoint_path,
                enable_gcs_patch_in_boto3=self.config_checkpoint.enable_gcs_patch_in_boto3,
            )
        return FileSystemReader(checkpoint_path)

    def _save_as_pkl(self, obj: Any, output_dir: str) -> None:
        """Save per-rank Python checkpoint state such as no-replace dataloader progress."""
        rank = dist.get_rank()
        path = os.path.join(output_dir, f"rank_{rank}.pkl")
        easy_io.dump(
            obj,
            path,
            file_format="pkl",
            backend_key=self.save_s3_backend_key,
        )
        log.info(f"Saved state to {path}")

    def save_state_dict_worker(self, to_save_dict: Dict[str, Tuple[Any, str]], checkpoint_file: str) -> None:
        for key, (v, full_checkpoint_path) in to_save_dict.items():
            if key == "dataloader":
                self._save_as_pkl(v, full_checkpoint_path)
            else:
                storage_writer = self.get_storage_writer(full_checkpoint_path)
                # Note that it is ok to create a new CustomSavePlanner object
                # for each checkpoint save since the save plans are cached in a
                # class dictionary.
                save_planner = CustomSavePlanner(
                    dedup_save_to_lowest_rank=True,
                    enable_plan_caching=True,
                    cache_plans_key=f"custom_planner_{key}",
                )
                dcp.save(
                    v,
                    storage_writer=storage_writer,
                    planner=save_planner,
                )

        if distributed.is_rank0():
            log.info(f"Saving last checkpoint file {checkpoint_file}")
            self._write_latest_checkpoint_file(checkpoint_file)

        log.info(f"Saved checkpoint to {os.path.join(self.save_dirname, checkpoint_file)}")

    def save(
        self,
        model: ImaginaireModel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        grad_scaler: torch.amp.GradScaler,
        iteration: int,
    ) -> None:
        """Save network weights, optimizer parameters, scheduler parameters to a checkpoint.

        Args:
            model (ImaginaireModel): The PyTorch model.
            optimizer (torch.optim.Optimizer): The model optimizer.
            scheduler (torch.optim.lr_scheduler.LRScheduler): The optimization scheduler.
            grad_scaler (torch.amp.GradScaler): The gradient scaler (for mixed precision training).
            iteration (int): Current iteration number.
        """
        if self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            self._wait_for_previous_async_checkpoint()

        if self.callbacks is not None:
            self.callbacks.on_save_checkpoint_start(model, iteration)

        checkpoint_file = f"iter_{iteration:09}"

        # Use rank-specific key for RNG state to ensure each rank saves its own state
        rng_key = f"rng_state_{dist.get_rank()}"

        to_save_dict = {
            "model": ModelWrapper(model).state_dict(),
            "optim": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "trainer": {
                "grad_scaler": grad_scaler.state_dict(),
                "iteration": iteration,
                rng_key: get_rand_state_dict(),
            },
        }
        dataloader_wrapper = _DataloaderWrapper(self.callbacks)
        if dataloader_wrapper.has_state():
            to_save_dict["dataloader"] = dataloader_wrapper.state_dict()

        if self.callbacks is not None:
            self.callbacks.on_save_checkpoint(model, state_dict=to_save_dict)

        for k in to_save_dict.keys():
            output_dirname = os.path.join(self.save_dirname, f"iter_{iteration:09}/{k}")
            to_save_dict[k] = (to_save_dict[k], output_dirname)

        if self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            dataloader_entry = to_save_dict.pop("dataloader", None)
            if dataloader_entry is not None:
                dataloader_state, dataloader_save_dir = dataloader_entry
                self._save_as_pkl(dataloader_state, dataloader_save_dir)
            self._checkpoint_async_with_pinned_memory(checkpoint_file, to_save_dict)
        else:
            start_time = time.monotonic()
            self.save_state_dict_worker(to_save_dict, checkpoint_file)
            elapsed_time = time.monotonic() - start_time
            log.info(f"Checkpoint save completed: Time taken: {elapsed_time:.2f} seconds")

            if self.callbacks is not None:
                self.callbacks.on_save_checkpoint_success(iteration=iteration, elapsed_time=elapsed_time)

        # This measures exposed (synchronous) checkpoint time, on_save_checkpoint_success()
        # is instead called to measure the entire duration for asynchronous checkpoint for the async case too.
        if self.callbacks is not None:
            self.callbacks.on_save_checkpoint_end(model=None, iteration=iteration)

    def finalize(self) -> None:
        super().finalize()
        if self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            if self.mp and self.mp.is_alive():
                # Wait for the previous checkpoint to complete.
                self._wait_for_previous_async_checkpoint()

                self.mp_queue_send.put(Terminate())
                self.mp.join()
