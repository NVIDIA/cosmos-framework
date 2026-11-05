# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Atomic copy transactions for loaded checkpoints owned by another run."""

from __future__ import annotations

import os
from dataclasses import dataclass

from cosmos_framework.posttrain.checkpoint.io import (
    component_kind,
    fsync_directory,
    is_complete_iter_dir,
    update_latest,
)
from cosmos_framework.posttrain.checkpoint.io import rename_directory_noreplace as _rename_directory_noreplace
from cosmos_framework.posttrain.checkpoint.provenance import (
    CheckpointProvenance,
    checkpoint_provenance,
    target_matches_external_copy,
    write_external_copy_provenance,
)


def _rank() -> int:
    import torch.distributed as dist

    return dist.get_rank() if dist.is_initialized() else 0


@dataclass(frozen=True)
class ExternalCheckpointCopyPlan:
    """Immutable source, destination, staging path, and source identity."""

    source_dir: str
    target_dir: str
    staging_dir: str | None
    provenance: CheckpointProvenance


def resolve_external_checkpoint_successor(source_dir: str, target_dir: str) -> str | None:
    """Publish and return a target proven to descend from ``source_dir``."""
    try:
        provenance = checkpoint_provenance(source_dir)
    except (FileNotFoundError, RuntimeError, OSError):
        return None
    if not target_matches_external_copy(target_dir, provenance):
        return None
    update_latest(target_dir)
    return target_dir


def _prepare_external_checkpoint_copy_rank0(
    source_dir: str,
    target_dir: str,
    expected_provenance: CheckpointProvenance | None,
) -> ExternalCheckpointCopyPlan:
    from uuid import uuid4

    source = os.path.realpath(os.path.expanduser(source_dir))
    target = os.path.abspath(os.path.expanduser(target_dir))
    if source == os.path.realpath(target):
        raise ValueError("external checkpoint copy source and target must be different directories")
    provenance = checkpoint_provenance(source)
    if expected_provenance is not None and provenance != expected_provenance:
        raise RuntimeError(f"external pending checkpoint changed after it was loaded and before copy: {source!r}")
    if os.path.lexists(target):
        if target_matches_external_copy(target, provenance):
            update_latest(target)
            return ExternalCheckpointCopyPlan(source, target, None, provenance)
        raise FileExistsError(
            f"refusing to overwrite pre-existing checkpoint target {target!r}: it is not a "
            "complete copy or authorized journal successor with matching external-source provenance"
        )

    root = os.path.dirname(target)
    os.makedirs(root, exist_ok=True)
    name = os.path.basename(target)
    staging = os.path.join(root, f".{name}.external-copy-{uuid4().hex}.staging")
    return ExternalCheckpointCopyPlan(source, target, staging, provenance)


def prepare_external_checkpoint_copy(
    source_dir: str,
    target_dir: str,
    *,
    expected_provenance: CheckpointProvenance | None = None,
) -> ExternalCheckpointCopyPlan:
    """Reserve same-filesystem staging, or reuse an exact prior promotion."""
    from cosmos_framework.posttrain.checkpoint.worker import join_pending_rank0
    from cosmos_framework.posttrain.utils.dist import broadcast_from_rank0, raise_if_any_rank_failed

    plan: ExternalCheckpointCopyPlan | None = None
    preparation_error: BaseException | None = None
    if _rank() == 0:
        try:
            join_pending_rank0()
            plan = _prepare_external_checkpoint_copy_rank0(source_dir, target_dir, expected_provenance)
        except BaseException as error:  # noqa: BLE001 -- every distributed rank must exit together
            preparation_error = error
    raise_if_any_rank_failed(preparation_error, "external pending-checkpoint copy preparation")
    shared = broadcast_from_rank0(lambda: plan)
    if not isinstance(shared, ExternalCheckpointCopyPlan):
        raise RuntimeError("external pending-checkpoint copy preparation returned no plan")
    return shared


def _promote_external_checkpoint_copy_rank0(plan: ExternalCheckpointCopyPlan) -> None:
    import errno
    import shutil

    current_provenance = checkpoint_provenance(plan.source_dir)
    if current_provenance != plan.provenance:
        raise RuntimeError(f"external pending checkpoint source changed while it was being copied: {plan.source_dir!r}")
    if plan.staging_dir is None:
        if not target_matches_external_copy(plan.target_dir, plan.provenance):
            raise RuntimeError(f"previously reusable external checkpoint target changed: {plan.target_dir!r}")
        update_latest(plan.target_dir)
        return

    staging = os.path.abspath(plan.staging_dir)
    target = os.path.abspath(plan.target_dir)
    if os.path.dirname(staging) != os.path.dirname(target):
        raise ValueError("external checkpoint staging and target directories must share one parent filesystem")
    if not is_complete_iter_dir(staging) or component_kind(staging, "optim") is None:
        raise RuntimeError(f"refusing to promote incomplete external checkpoint staging directory {staging!r}")
    write_external_copy_provenance(staging, plan.provenance)
    try:
        _rename_directory_noreplace(staging, target)
    except OSError as error:
        if error.errno not in (errno.EEXIST, errno.ENOTEMPTY):
            raise
        if not target_matches_external_copy(target, plan.provenance):
            raise FileExistsError(
                f"refusing to overwrite checkpoint target {target!r} during atomic promotion"
            ) from error
        # A concurrent process promoted the same immutable source first. Discard only this call's
        # uniquely named staging directory and reuse the authoritative identical target.
        shutil.rmtree(staging)
    fsync_directory(os.path.dirname(target))
    update_latest(target)


def promote_external_checkpoint_copy(plan: ExternalCheckpointCopyPlan) -> None:
    """Atomically promote a complete staged copy and publish its latest pointer."""
    from cosmos_framework.posttrain.checkpoint.worker import join_pending_rank0
    from cosmos_framework.posttrain.utils.dist import raise_if_any_rank_failed

    promotion_error: BaseException | None = None
    if _rank() == 0:
        try:
            join_pending_rank0()
            _promote_external_checkpoint_copy_rank0(plan)
        except BaseException as error:  # noqa: BLE001 -- every distributed rank must exit together
            promotion_error = error
    raise_if_any_rank_failed(promotion_error, "external pending-checkpoint atomic promotion")
