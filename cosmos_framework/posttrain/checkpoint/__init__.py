# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Unified checkpoint API, resolved lazily so metadata/conversion helpers stay torch-free."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "EXTERNAL_COPY_PROVENANCE_FILE",
    "ExternalCheckpointCopyPlan",
    "CheckpointManager",
    "ContinuationManifest",
    "ContinuationPlan",
    "HfPublication",
    "publish_hf_export",
    "CheckpointStore",
    "LocalCheckpointStore",
    "ObjectStoreCheckpointStore",
    "POSTTRAIN_MANIFEST",
    "ReferenceIdentity",
    "checkpoint_provenance",
    "committed_artifact_iterations",
    "build_checkpoint_store",
    "save_components",
    "rank0_snapshot_is_immutable_cpu",
    "wait_for_pending",
    "load_component",
    "component_kind",
    "save_component_dcp",
    "load_component_dcp",
    "fill_sharded_state_from_consolidated",
    "load_with_stable_provenance",
    "materialized_reference_sha256",
    "keep_newest_iterations",
    "latest_dir",
    "resolve_resume",
    "resolve_run_ts",
    "finalize",
    "prune_local_optimizer_state",
    "prepare_external_checkpoint_copy",
    "promote_external_checkpoint_copy",
    "apply_local_archive_retention",
    "replace_component_atomic",
    "resolve_external_checkpoint_successor",
    "collect_rng_state",
    "restore_rng_state",
]
_EXPORTS = {
    "EXTERNAL_COPY_PROVENANCE_FILE": ("provenance", "EXTERNAL_COPY_PROVENANCE_FILE"),
    "ExternalCheckpointCopyPlan": ("external_copy", "ExternalCheckpointCopyPlan"),
    "CheckpointManager": ("manager", "CheckpointManager"),
    "ContinuationManifest": ("continuation", "ContinuationManifest"),
    "ContinuationPlan": ("continuation", "ContinuationPlan"),
    "HfPublication": ("hf_export", "HfPublication"),
    "publish_hf_export": ("hf_export", "publish_hf_export"),
    "CheckpointStore": ("store", "CheckpointStore"),
    "LocalCheckpointStore": ("store", "LocalCheckpointStore"),
    "ObjectStoreCheckpointStore": ("store", "ObjectStoreCheckpointStore"),
    "POSTTRAIN_MANIFEST": ("manifest", "POSTTRAIN_MANIFEST"),
    "ReferenceIdentity": ("reference", "ReferenceIdentity"),
    "checkpoint_provenance": ("provenance", "checkpoint_provenance"),
    "committed_artifact_iterations": ("artifact", "committed_artifact_iterations"),
    "build_checkpoint_store": ("store", "build_checkpoint_store"),
    "save_components": ("manager", "save_components"),
    "rank0_snapshot_is_immutable_cpu": ("manager", "rank0_snapshot_is_immutable_cpu"),
    "wait_for_pending": ("manager", "wait_for_pending"),
    "load_component": ("io", "load_component"),
    "component_kind": ("io", "component_kind"),
    "save_component_dcp": ("io", "save_component_dcp"),
    "load_component_dcp": ("io", "load_component_dcp"),
    "fill_sharded_state_from_consolidated": ("consolidated", "fill_sharded_state_from_consolidated"),
    "load_with_stable_provenance": ("provenance", "load_with_stable_provenance"),
    "materialized_reference_sha256": ("reference", "materialized_reference_sha256"),
    "keep_newest_iterations": ("retention", "keep_newest_iterations"),
    "latest_dir": ("io", "latest_dir"),
    "resolve_resume": ("io", "resolve_resume"),
    "resolve_run_ts": ("io", "resolve_run_ts"),
    "finalize": ("manager", "finalize"),
    "prune_local_optimizer_state": ("retention", "prune_local_optimizer_state"),
    "prepare_external_checkpoint_copy": ("external_copy", "prepare_external_checkpoint_copy"),
    "promote_external_checkpoint_copy": ("external_copy", "promote_external_checkpoint_copy"),
    "apply_local_archive_retention": ("manager", "apply_local_archive_retention"),
    "replace_component_atomic": ("provenance", "replace_component_atomic"),
    "resolve_external_checkpoint_successor": ("external_copy", "resolve_external_checkpoint_successor"),
    "collect_rng_state": ("rng", "collect_rng_state"),
    "restore_rng_state": ("rng", "restore_rng_state"),
}


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(importlib.import_module(f"{__name__}.{module_name}"), attribute)
    globals()[name] = value
    return value
