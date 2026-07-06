# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import importlib
import os
import os.path as osp
import re
import time
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from enum import IntEnum
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp

try:
    from filelock import SoftReadWriteLock
except ImportError:  # Older filelock versions in some inference containers.
    try:
        from filelock import ReadWriteLock as SoftReadWriteLock
    except ImportError:
        from filelock import FileLock

        class SoftReadWriteLock:
            """Compatibility adapter for filelock versions without read/write locks."""

            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self._lock = FileLock(*args, **kwargs)

            def write_lock(self) -> FileLock:
                return self._lock


from torch.distributed.checkpoint.filesystem import FileSystemReader, FileSystemWriter

from cosmos_framework.checkpoint.s3_filesystem import S3StorageReader
from cosmos_framework.utils.lazy_config import instantiate
from cosmos_framework.utils import log, misc
from cosmos_framework.utils.config_helper import get_config_module, override
from cosmos_framework.utils.easy_io import easy_io
from cosmos_framework.checkpoint.dcp import CustomLoadPlanner, CustomSavePlanner, ModelWrapper
from cosmos_framework.model.generator.utils.safetensors_loader import load_vfm_model

###################################################
# below are the load_model function for inference #
###################################################
# Thus these load_model functions are designed with less dependency.


def checkpoint_path_to_cached_path(path: str, cache_rootdir: Optional[str] = None) -> str:
    if cache_rootdir is None:
        homedir = os.getenv("HOME") or ""
        cache_rootdir = osp.join(homedir, ".cache/imaginaire4/checkpoints/")

    if path.startswith("s3://"):
        return osp.join(cache_rootdir, path.removeprefix("s3://").split("/", maxsplit=1)[1])
    else:
        return path


def _is_safetensors_checkpoint(checkpoint_path: str, credential_path: str | None) -> bool:
    """Return True if ``checkpoint_path`` is a directory containing any ``*.safetensors`` shard.

    Probes the path (local or ``s3://`` / ``gs://``) via ``easy_io`` and
    short-circuits on the first ``.safetensors`` entry, so the cost is O(1)
    listing calls regardless of shard count.  Returns False for DCP
    checkpoints (which have ``.metadata`` + ``*.distcp`` files but no
    ``.safetensors``), for non-directory paths, and for any path where
    the listing call raises (treated as "definitely not safetensors" so
    the caller falls through to the legacy DCP path).
    """
    if checkpoint_path.startswith("s3://"):
        backend_args: dict[str, Any] | None = {
            "backend": "s3",
            "s3_credential_path": credential_path or "",
        }
    else:
        backend_args = None

    try:
        listing = easy_io.list_dir_or_file(
            checkpoint_path,
            list_dir=False,
            list_file=True,
            suffix="safetensors",
            recursive=False,
            backend_args=backend_args,
        )
        return next(iter(listing), None) is not None
    except Exception:
        return False


def _is_checkpoint_cache_ready(checkpoint_path: str) -> bool:
    """Return True when FileSystemWriter has published its completion metadata."""
    return osp.isfile(osp.join(checkpoint_path, ".metadata"))


class _CheckpointCacheAction(IntEnum):
    LOAD_CACHE = 0
    POPULATE_CACHE = 1
    ERROR = 2


def _is_distributed_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def _dist_backend_device() -> torch.device:
    backend = str(dist.get_backend()).lower()
    if backend.endswith("nccl"):
        return torch.device("cuda", torch.cuda.current_device())
    else:
        return torch.device("cpu")


def _get_rank0_action(action: _CheckpointCacheAction) -> _CheckpointCacheAction:
    if not _is_distributed_initialized():
        return action

    action_tensor = torch.tensor(int(action), dtype=torch.int64, device=_dist_backend_device())  # []
    dist.broadcast(action_tensor, src=0)  # []
    return _CheckpointCacheAction(int(action_tensor.item()))


def _resolve_checkpoint_cache_action(
    checkpoint_cache_path: str,
    cache_lock_path: str,
    exit_stack: ExitStack,
) -> _CheckpointCacheAction:
    assert (not _is_distributed_initialized()) or (dist.get_rank() == 0), (
        "Only rank 0 should resolve checkpoint cache action"
    )

    if _is_checkpoint_cache_ready(checkpoint_cache_path):
        return _CheckpointCacheAction.LOAD_CACHE

    cache_lock_parent = osp.dirname(cache_lock_path)
    if cache_lock_parent:
        os.makedirs(cache_lock_parent, exist_ok=True)

    try:
        cache_lock = SoftReadWriteLock(
            cache_lock_path,
            heartbeat_interval=60,
            stale_threshold=3 * 60,
            poll_interval=1,
        )
    except TypeError:
        cache_lock = SoftReadWriteLock(cache_lock_path)
    exit_stack.enter_context(cache_lock.write_lock())

    # Re-check after waiting on the inter-process lock.
    if _is_checkpoint_cache_ready(checkpoint_cache_path):
        return _CheckpointCacheAction.LOAD_CACHE

    return _CheckpointCacheAction.POPULATE_CACHE


@contextmanager
def _checkpoint_cache_group_lock(
    checkpoint_cache_path: str,
    cache_lock_path: str,
) -> Iterator[_CheckpointCacheAction]:
    """Coordinate checkpoint cache population across distributed ranks."""
    action = _CheckpointCacheAction.ERROR
    rank0_error: Exception | None = None

    with ExitStack() as exit_stack:
        if not _is_distributed_initialized() or dist.get_rank() == 0:
            try:
                action = _resolve_checkpoint_cache_action(checkpoint_cache_path, cache_lock_path, exit_stack)
            except Exception as error:
                rank0_error = error
                action = _CheckpointCacheAction.ERROR

        action = _get_rank0_action(action)
        if action == _CheckpointCacheAction.ERROR:
            if rank0_error is not None:
                raise rank0_error
            raise RuntimeError("Rank 0 failed to resolve checkpoint cache action.")

        yield action


def _reload_pretrained_reasoner_after_checkpoint_load(model: torch.nn.Module) -> None:
    """Re-seed the reasoner pathway after a DCP load, mirroring the LoadPretrained
    callback that runs during training (inference does not run training callbacks).

    The decision is delegated entirely to the model's own gate in
    ``load_pretrained_model_if_needed``: this is a no-op unless the model was built
    with ``exclude_reasoner_weights_from_checkpoint=True`` (and pretrained weights
    enabled), i.e. the case where the DCP checkpoint deliberately omits the reasoner
    tower so it must be re-seeded from the pretrained source. For a normal checkpoint
    that already contains the reasoner, the model's gate evaluates to False and
    nothing is reloaded.

    ``has_resumable_checkpoint=True`` / ``has_load_path=False`` is load-bearing: it
    re-seeds the reasoner from the pretrained source while skipping the
    understanding->generation copy (the generation pathway was already populated by
    the DCP load). Passing ``has_load_path=True`` would instead force a reasoner
    reload even for non-excluded checkpoints, clobbering any fine-tuned reasoner
    weights restored from the DCP.
    """
    load_pretrained_model_if_needed = getattr(model, "load_pretrained_model_if_needed")
    load_pretrained_model_if_needed(
        has_resumable_checkpoint=True,
        has_load_path=False,
    )


def _load_model(
    model: torch.nn.Module,
    checkpoint_path: str,
    credential_path: str | None,
    enable_gcs_patch_in_boto3: bool = False,
    load_ema_to_reg: bool = False,
    keys_to_skip_loading: list[str] | None = None,
) -> None:
    """
    Args:
        model: The model to load weights into
        checkpoint_path: Path to checkpoint (can be s3 or local path)
        credential_path: Path to S3 credentials (can be none if load local)
        enable_gcs_patch_in_boto3: Whether to enable GCS patch in boto3 for DCP loading from GCS
        load_ema_to_reg: Whether to load EMA weights into the regular (non-EMA) model parameters.
        keys_to_skip_loading: List of key substrings to skip when loading from checkpoint.
            Useful for loading pretrained checkpoints that are missing certain keys (e.g. action heads).
    """

    log.info(f"Loading model from {checkpoint_path}")
    start_time = time.time()

    state_dict = ModelWrapper(model).state_dict()
    if any(key.startswith("net_teacher.") for key in state_dict):
        log.info("Dropping net_teacher.* keys from inference load target; distillation checkpoints do not save them.")
        state_dict = {key: value for key, value in state_dict.items() if not key.startswith("net_teacher.")}

    if checkpoint_path.startswith("s3://"):
        storage_reader = S3StorageReader(
            credential_path=credential_path or "",
            path=checkpoint_path,
            enable_gcs_patch_in_boto3=enable_gcs_patch_in_boto3,
        )
    else:
        storage_reader = FileSystemReader(checkpoint_path)

    load_planner = CustomLoadPlanner(
        load_ema_to_reg=load_ema_to_reg,
        keys_to_skip_loading=keys_to_skip_loading or [],
    )

    dcp.load(
        state_dict=state_dict,
        storage_reader=storage_reader,
        planner=load_planner,
    )

    log.info(f"Successfully loaded model from {checkpoint_path}")
    log.info(f"Time taken to load model: {time.time() - start_time:.2f} seconds")


def _save_model(
    model: torch.nn.Module,
    checkpoint_path: str,
    save_reg_to_ema: bool = False,
) -> None:
    """
    Args:
        model: The model to load weights into
        checkpoint_path: Path to cached checkpoint (can be s3 or local path)
        save_reg_to_ema: Whether to save regular (non-EMA) model parameters to EMA model parameters.
    """

    log.info(f"Saving model to {checkpoint_path}")
    start_time = time.time()

    state_dict = ModelWrapper(model).state_dict()

    assert not checkpoint_path.startswith("s3://"), "Cached checkpoint path must be local path"
    storage_writer = FileSystemWriter(checkpoint_path)

    save_planner = CustomSavePlanner(save_reg_to_ema=save_reg_to_ema, dedup_save_to_lowest_rank=True)

    dcp.save(
        state_dict=state_dict,
        storage_writer=storage_writer,
        planner=save_planner,
    )

    log.info(f"Successfully saved model to {checkpoint_path}")
    log.info(f"Time taken to save model: {time.time() - start_time:.2f} seconds")


def load_model_from_checkpoint(
    experiment_name: str,
    checkpoint_path: Optional[str] = None,
    credential_path: Optional[str] = None,
    enable_gcs_patch_in_boto3: bool = False,
    config_file: str = "cosmos_framework/configs/base/config.py",
    load_ema_to_reg: bool = False,
    parallelism_config: dict[str, Any] = {},
    compile_config: dict[str, Any] = {},
    seed: int = 0,
    experiment_opts: list[str] = [],
    use_cache_checkpoint: bool = False,
    cache_checkpoint_rootdir: Optional[str] = None,
    keys_to_skip_loading: list[str] | None = None,
) -> tuple[torch.nn.Module, Any]:
    """
    Args:
        experiment_name: Experiment name.
        checkpoint_path: Path to the checkpoint (local path or s3 URI).  Two
            on-disk formats are auto-detected:
              * **DCP** (default): a directory tree containing ``.metadata`` +
                ``*.distcp`` shards.  Loaded via
                :func:`torch.distributed.checkpoint.load`.  The legacy
                "weights live under ``<base>/model/``" convention is honored
                here — if ``checkpoint_path`` does not already end with
                ``/model``, ``/model`` is appended.
              * **safetensors**: a directory containing one or more
                ``*.safetensors`` shards in the native Cosmos3 VFM state-dict
                layout.  Loaded via
                :func:`cosmos_framework.model.generator.utils.safetensors_loader.load_vfm_model`.
                No ``/model`` suffix is appended.
        credential_path: Path to credentials file (if required for remote storage). Optional.
        enable_gcs_patch_in_boto3: Whether to enable the boto3 patch for GCS S3-compatibility.
            Applies to the DCP path only; ignored for safetensors (``easy_io`` handles
            GCS routing internally).
        config_file: Path to the config file used to construct the experiment/model.
        load_ema_to_reg: If True, load EMA weights into the regular (non-EMA) model parameters.
            Only supported for DCP checkpoints — safetensors VFM checkpoints carry a single
            weight set with no EMA branch, so this raises ``ValueError`` when paired with a
            safetensors source.
        parallelism_config: Dictionary of parallelism configuration options. Keys are applied to
            ``config.model.config.parallelism``.
        compile_config: Dictionary of torch.compile configuration options. Keys are applied to
            ``config.model.config.compile``.
        seed: Random seed used for initialization (if applicable).
        experiment_opts: Extra experiment/config override options.
        use_cache_checkpoint: If True, locally save & read remote checkpoints to speed up repeated loads.
            Be aware, the default cache path is $HOME/.cache/imaginaire4/checkpoints/<same s3 path>.
            Applies to the DCP path only; for safetensors checkpoints a warning is logged
            and caching is skipped (the DCP write cache is not a meaningful round-trip for
            safetensors sources).
        cache_checkpoint_rootdir: Customizable root directory for checkpoint cache. Optional.
        keys_to_skip_loading: List of key substrings to skip when loading from checkpoint.
            Useful for loading pretrained checkpoints that are missing certain keys (e.g. action heads).
            For DCP this is forwarded as-is to ``CustomLoadPlanner`` (substring match).  For
            safetensors each substring is escaped and wrapped as ``.*<substring>.*`` so that
            ``load_vfm_model``'s ``re.fullmatch``-based ``skip_patterns`` reproduces the
            substring semantics one-for-one.

    Returns:
        The loaded model and config
    """

    # Ensure checkpoint_path is provided
    if checkpoint_path is None:
        raise ValueError("'checkpoint_path' must be provided.")

    # Detect checkpoint format BEFORE applying any DCP-specific path
    # rewriting (the legacy "/model" suffix only makes sense for DCP).
    # ``_is_safetensors_checkpoint`` is a cheap directory listing that
    # short-circuits on the first hit and tolerates I/O failures (returns
    # False) so a bad path naturally falls through to the DCP code path
    # and produces the same error message it always did.
    is_safetensors = _is_safetensors_checkpoint(checkpoint_path, credential_path)
    if not is_safetensors and not checkpoint_path.strip("/").endswith("model"):
        checkpoint_path = os.path.join(checkpoint_path, "model")

    config_module = get_config_module(config_file)
    config = importlib.import_module(config_module).make_config()
    config = override(config, ["--", f"experiment={experiment_name}"] + experiment_opts)

    if parallelism_config is not None:
        for key, value in parallelism_config.items():
            if hasattr(config.model.config.parallelism, key):
                setattr(config.model.config.parallelism, key, value)
            else:
                raise ValueError(f"Key {key} not found in config.model.config.parallelism")

    if compile_config is not None:
        for key, value in compile_config.items():
            if hasattr(config.model.config.compile, key):
                setattr(config.model.config.compile, key, value)
            else:
                raise ValueError(f"Key {key} not found in config.model.config.compile")

    # Disable activation checkpointing for inference.
    config.model.config.activation_checkpointing.mode = "none"

    # Disable EMA for inference.
    config.model.config.ema.enabled = False
    if hasattr(config.model.config, "load_teacher_weights"):
        log.info("Setting load_teacher_weights=False for inference to skip teacher checkpoint download.")
        config.model.config.load_teacher_weights = False
    if getattr(config.model.config, "student_load_from", None) is not None:
        log.info("Setting student_load_from=None for inference to skip train-time student warm-start download.")
        config.model.config.student_load_from = None

    if (
        config.model.config.exclude_reasoner_weights_from_checkpoint
        and not config.model.config.vlm_config.pretrained_weights.enabled
    ):
        log.info("Enabling pretrained reasoner weights because this checkpoint excludes the reasoner tower from DCP.")
        config.model.config.vlm_config.pretrained_weights.enabled = True

    config.validate()
    config.freeze()  # type: ignore

    misc.set_random_seed(seed=seed, by_rank=True)

    torch.backends.cudnn.deterministic = config.trainer.cudnn.deterministic
    torch.backends.cudnn.benchmark = config.trainer.cudnn.benchmark

    with misc.timer("instantiate model"):
        model = instantiate(config.model).cuda()  # type: ignore
        model.on_train_start()

    if is_safetensors:
        # Validate DCP-only flags up front so the caller gets a clear error
        # instead of a silent no-op.
        if load_ema_to_reg:
            raise ValueError(
                "load_ema_to_reg=True is not supported for safetensors checkpoints — "
                "safetensors VFM checkpoints contain a single weight set with no EMA branch."
            )
        if use_cache_checkpoint:
            log.warning(
                "use_cache_checkpoint is ignored for safetensors checkpoints "
                "(the local cache is a DCP-format reshard optimization and does not "
                "round-trip through safetensors)."
            )

        # Translate DCP-style substring skips into regex fullmatch patterns
        # so load_vfm_model's regex contract reproduces the substring semantics.
        skip_patterns: list[str] | None = (
            [f".*{re.escape(s)}.*" for s in keys_to_skip_loading] if keys_to_skip_loading else None
        )

        # load_vfm_model expects the Cosmos3VFMNetwork directly.  In
        # OmniMoTModel this lives at ``.net``; for tests/standalone callers
        # that pass a Cosmos3VFMNetwork in directly, fall back to the model
        # itself.  ``parallel_dims`` is populated by ``set_up_parallelism()``
        # (called from ``on_train_start()`` above) — None for single-process
        # / non-distributed runs, which load_vfm_model accepts.
        vfm_network = getattr(model, "net", model)
        parallel_dims = getattr(model, "parallel_dims", None)

        log.info(f"Loading safetensors VFM checkpoint from {checkpoint_path}")
        start_time = time.time()
        load_vfm_model(
            model=vfm_network,
            checkpoint_path=checkpoint_path,
            credential_path=credential_path,
            parallel_dims=parallel_dims,
            skip_patterns=skip_patterns,
        )
        log.info(
            f"Successfully loaded safetensors VFM checkpoint from {checkpoint_path}; "
            f"time taken: {time.time() - start_time:.2f} seconds"
        )
        return model, config

    # DCP path with optional local cache reshard.
    def load_model(checkpoint_load_path: str) -> None:
        _load_model(
            model,
            checkpoint_path=checkpoint_load_path,
            credential_path=credential_path,
            enable_gcs_patch_in_boto3=enable_gcs_patch_in_boto3,
            load_ema_to_reg=load_ema_to_reg,
            keys_to_skip_loading=keys_to_skip_loading,
        )

    checkpoint_cache_path = None
    if use_cache_checkpoint:
        checkpoint_cache_path = checkpoint_path_to_cached_path(checkpoint_path, cache_checkpoint_rootdir)

    if checkpoint_cache_path is None:
        load_model(checkpoint_path)
        _reload_pretrained_reasoner_after_checkpoint_load(model)
        return model, config

    cache_lock_path = f"{checkpoint_cache_path}.lock"
    cache_action = _CheckpointCacheAction.ERROR
    log.info(f"Acquiring checkpoint cache write lock: {cache_lock_path}")
    with _checkpoint_cache_group_lock(checkpoint_cache_path, cache_lock_path) as cache_action:
        if cache_action == _CheckpointCacheAction.POPULATE_CACHE:
            load_model(checkpoint_path)
            _save_model(
                model,
                checkpoint_path=checkpoint_cache_path,
                save_reg_to_ema=load_ema_to_reg,
            )

    if cache_action == _CheckpointCacheAction.LOAD_CACHE:
        load_model(checkpoint_cache_path)

    _reload_pretrained_reasoner_after_checkpoint_load(model)

    return model, config
