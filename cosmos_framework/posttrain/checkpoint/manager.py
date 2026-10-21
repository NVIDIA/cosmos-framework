# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Checkpoint save orchestration."""

from __future__ import annotations

import logging
import os
import tempfile
import warnings
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from cosmos_framework.posttrain.checkpoint.archive.publisher import archive_checkpoint, publish_artifact
from cosmos_framework.posttrain.checkpoint.io import update_latest as _update_latest
from cosmos_framework.posttrain.checkpoint.io import write_component_files
from cosmos_framework.posttrain.checkpoint.manifest import iteration_name
from cosmos_framework.posttrain.checkpoint.provenance import load_with_stable_provenance as _load_with_stable_provenance
from cosmos_framework.posttrain.checkpoint.retention import (
    prune_local_iterations,
    prune_local_optimizer_state,
)
from cosmos_framework.posttrain.checkpoint.store import CheckpointStore, build_checkpoint_store
from cosmos_framework.posttrain.checkpoint.worker import (
    join_pending_rank0,
    run_rank0_converged,
    start_background_write,
    wait_for_pending,
)
from cosmos_framework.posttrain.schema.checkpoint import CheckpointComponent, CheckpointComponentPlan
from cosmos_framework.posttrain.utils.dist import (
    broadcast_from_rank0,
    raise_if_any_rank_failed,
    rank_identical,
    run_rank_converged,
)
from cosmos_framework.posttrain.utils.host_memory import log_host_memory
from cosmos_framework.posttrain.utils.log import logger

if TYPE_CHECKING:
    import torch


_PHYSICAL_COMPONENT_FAMILIES: dict[str, CheckpointComponent] = {
    "model": "model",
    "optim": "optimizer",
    "trainer": "training_state",
    "data": "training_state",
    "old": "training_state",
    "ref": "training_state",
}


def _iter_tensors(value: Any, seen: set[int] | None = None) -> Iterator["torch.Tensor"]:
    """Yield tensors from a nested state object without revisiting container cycles."""
    import torch

    if isinstance(value, torch.Tensor):
        yield value
        return
    if not isinstance(value, (Mapping, list, tuple)):
        return
    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        return
    seen.add(identity)
    items = value.values() if isinstance(value, Mapping) else value
    for item in items:
        yield from _iter_tensors(item, seen)


def _iter_mutable_objects(value: Any, seen: set[int] | None = None) -> Iterator[Any]:
    """Yield mutable containers/leaves while traversing immutable containers."""
    import torch

    if isinstance(value, torch.Tensor) or value is None or isinstance(value, (bool, int, float, complex, str, bytes)):
        return
    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        return
    seen.add(identity)
    if isinstance(value, Mapping):
        yield value
        for key, item in value.items():
            yield from _iter_mutable_objects(key, seen)
            yield from _iter_mutable_objects(item, seen)
        return
    if isinstance(value, (list, set, bytearray)):
        yield value
        for item in value:
            yield from _iter_mutable_objects(item, seen)
        return
    if isinstance(value, (tuple, frozenset)):
        for item in value:
            yield from _iter_mutable_objects(item, seen)
        return
    # NumPy arrays and custom state objects are mutable leaves. Distinct objects are accepted under
    # the producer's snapshot contract; sharing the exact object with live state fails closed below.
    yield value


def snapshot_is_immutable_cpu(snapshot: Any, live_state: Any) -> bool:
    """Prove a tensor-bearing snapshot is CPU-only and detached from live training objects.

    Tensor storage and mutable object identity are checked independently. Tensor-free snapshots fail
    closed because no positive tensor-snapshot evidence exists, even when their containers differ.
    """

    def _storage_pointer(tensor: "torch.Tensor") -> int | None:
        if tensor.device.type != "cpu" or tensor.numel() == 0:
            return None
        return tensor.untyped_storage().data_ptr()

    try:
        live_pointers = {
            pointer for tensor in _iter_tensors(live_state) if (pointer := _storage_pointer(tensor)) is not None
        }
    except (NotImplementedError, RuntimeError):
        return False
    live_object_ids = {id(value) for value in _iter_mutable_objects(live_state)}
    if any(id(value) in live_object_ids for value in _iter_mutable_objects(snapshot)):
        return False
    found_tensor = False
    for tensor in _iter_tensors(snapshot):
        found_tensor = True
        if tensor.device.type != "cpu":
            return False
        try:
            pointer = _storage_pointer(tensor)
        except (NotImplementedError, RuntimeError):
            return False
        if pointer is not None and pointer in live_pointers:
            return False
    return found_tensor


def rank0_snapshot_is_immutable_cpu(snapshot: Any, live_state: Any) -> bool:
    """Prove the rank-zero writer snapshot once and broadcast one branch decision to every rank."""
    return broadcast_from_rank0(
        lambda: snapshot_is_immutable_cpu(snapshot, live_state),
        context="checkpoint snapshot immutability proof",
    )


@dataclass
class ArtifactSnapshot:
    """One-shot carrier of the host state an artifact commit writes.

    The consolidated model-only save gathers the whole model onto rank 0's host and hands it to the
    background writer as a plain argument. A tuple argument would keep those tensors alive for the
    writer's entire job, i.e. through the upload that follows the write, exactly when the freshly
    written payload also fills the page cache. Taking the snapshot out of this carrier lets the
    commit drop the tensors the moment the payload files are on disk.
    """

    components: Mapping[str, dict[str, Any] | None] | None

    def take(self) -> Mapping[str, dict[str, Any] | None]:
        """Return the snapshot once and forget it; a second call is a programming error."""
        components = self.components
        if components is None:
            raise RuntimeError("checkpoint artifact snapshot was already consumed")
        self.components = None
        return components


def _log_host_memory(context: str) -> None:
    """One ``host memory`` line per checkpoint event, sampled in the process that owns the state.

    The controller loop is the wrong place to read this: under a multi-node or disaggregated
    placement its RSS and cgroup are not the policy worker's, which holds the gathered snapshot and
    the artifact page cache, so its series was not comparable with the writer's own readings after
    the write and the publish. Sampling here, in the shared manager, gives every algorithm's loop
    one worker-side series. ``info0`` semantics: INFO on rank 0, DEBUG on the other ranks.
    """
    level = logging.INFO if os.environ.get("RANK", "0") == "0" else logging.DEBUG
    log_host_memory(context, level=level)


def _commit_checkpoint_artifact(
    store: CheckpointStore,
    location: str,
    checkpoint_format: str,
    checkpoint_components: tuple[str, ...],
    dcp_component_names: list[str],
    snapshot: ArtifactSnapshot,
    upload: dict[str, Any] | None,
) -> None:
    """Commit one partial artifact and optionally archive only its declared payloads.

    The gathered host snapshot is released before the upload starts: once the payload files are
    written and bound into the manifest nothing reads the tensors again, and holding them through a
    model-sized upload is what pushed a colocated job over its cgroup limit at the final drain.
    """
    file_components = snapshot.take()
    store.commit_artifact(
        location,
        checkpoint_format,
        checkpoint_components,
        dcp_component_names,
        file_components,
    )
    del file_components
    log_host_memory(f"checkpoint artifact written {location}")
    if upload:
        try:
            publish_artifact(location, upload)
        except Exception as error:  # noqa: BLE001 -- ordinary training archive is best effort
            logger.warning(f"checkpoint artifact upload failed ({error}); keeping the local artifact")
        log_host_memory(f"checkpoint artifact publish finished {location}")


@dataclass
class CheckpointManager:
    """Own the checkpoint lifecycle policy shared by every engine trainer.

    Trainers still extract model-, optimizer-, and control-specific state.  This object owns the
    format and primary-store choice, logical component plan, DCP transaction, asynchronous write
    eligibility, archive/retention policy, and finalization so a new engine cannot accidentally grow
    a second checkpoint lifecycle. Full plans are resumable checkpoints; every partial plan is
    committed under the sibling ``checkpoint_artifacts`` namespace and never advances training latest.
    """

    checkpoint_format: str = "consolidated"
    save_async: bool = False
    keep_optim_last: int | None = None
    keep_model_last: int = 0
    upload: dict[str, Any] | None = None
    checkpoint_store: Mapping[str, Any] | CheckpointStore | None = None
    checkpoint_components: Iterable[str] | CheckpointComponentPlan | None = None
    _store: CheckpointStore = field(init=False, repr=False)
    _component_plan: CheckpointComponentPlan = field(init=False, repr=False)
    _warned_partial_plans: set[tuple[str, ...]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.save_async = bool(self.save_async)
        self.upload = dict(self.upload) if self.upload else None
        self._store = build_checkpoint_store(self.checkpoint_store)
        if self.keep_optim_last is None:
            self.keep_optim_last = 0 if self._store.is_object_store else 1
        self._warned_partial_plans = set()
        self._component_plan = CheckpointComponentPlan.from_value(self.checkpoint_components)
        self._validate_policy(self.checkpoint_format)
        self._warn_partial_plan(self._component_plan)

    def _validate_policy(self, checkpoint_format: str) -> None:
        """Validate the normalized cross-option policy owned by this manager."""
        if checkpoint_format not in ("consolidated", "dcp"):
            raise ValueError(f"checkpoint_format must be 'consolidated' or 'dcp', got {checkpoint_format!r}")
        for name in ("keep_optim_last", "keep_model_last"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
        if self._store.is_object_store:
            if checkpoint_format != "dcp":
                raise ValueError(
                    "checkpoint_store.backend='object_store' requires checkpoint_format='dcp'; "
                    "direct consolidated checkpoints are unsupported"
                )
            if self.upload is not None:
                raise ValueError(
                    "checkpoint_store.backend='object_store' is mutually exclusive with upload.enabled=true; "
                    "rl.upload archives a local-primary checkpoint"
                )
            if self.keep_optim_last > 0 or self.keep_model_last > 0:
                raise ValueError(
                    "checkpoint_store.backend='object_store' requires keep_optim_last=0 and keep_model_last=0; "
                    "remote retention is unsupported"
                )

    @property
    def store(self) -> CheckpointStore:
        """Primary storage backend selected for this run."""
        return self._store

    def set_checkpoint_format(self, checkpoint_format: str) -> None:
        """Change format for explicit test/export flows without leaking store policy."""
        self._validate_policy(checkpoint_format)
        self.checkpoint_format = checkpoint_format

    def component_plan(
        self,
        override: Iterable[str] | CheckpointComponentPlan | None = None,
    ) -> CheckpointComponentPlan:
        """Return the manager default or one validated per-save component plan."""
        plan = run_rank_converged(
            lambda: self._component_plan if override is None else CheckpointComponentPlan.from_value(override),
            "checkpoint component plan preparation",
        )
        names = rank_identical(plan.names, "checkpoint component plan")
        plan = CheckpointComponentPlan.from_value(names)
        self._warn_partial_plan(plan)
        return plan

    def _warn_partial_plan(self, plan: CheckpointComponentPlan) -> None:
        """Warn once on rank zero that one logical plan cannot support exact resume."""
        if plan.is_full or plan.names in self._warned_partial_plans:
            return
        self._warned_partial_plans.add(plan.names)

        import torch.distributed as dist

        rank = dist.get_rank() if dist.is_initialized() else int(os.environ.get("RANK", "0"))
        warning_error: BaseException | None = None
        if rank == 0:
            try:
                warnings.warn(
                    f"checkpoint components {list(plan.names)!r} select a partial checkpoint artifact; "
                    "partial artifacts are not exact-resume capable",
                    UserWarning,
                    stacklevel=3,
                )
            except BaseException as error:  # noqa: BLE001 -- warning filters must not split ranks
                warning_error = error
        raise_if_any_rank_failed(warning_error, "partial checkpoint warning")

    def save_location(self, iter_hint: str, component_plan: CheckpointComponentPlan) -> str:
        """Return the primary location used by one full checkpoint or partial artifact save."""
        if component_plan.is_full:
            return self.store.iteration_location(iter_hint)
        return self.store.artifact_location(iter_hint)

    @staticmethod
    def _selected_physical_components(
        components: Mapping[str, dict[str, Any] | None],
        component_plan: CheckpointComponentPlan,
    ) -> dict[str, dict[str, Any] | None]:
        """Map standardized physical component names into logical save families."""
        unknown = sorted(set(components) - set(_PHYSICAL_COMPONENT_FAMILIES))
        if unknown:
            raise ValueError(f"unknown physical checkpoint components: {unknown}")
        return {
            name: state
            for name, state in components.items()
            if component_plan.includes(_PHYSICAL_COMPONENT_FAMILIES[name])
        }

    def _write_dcp_components(
        self,
        location: str,
        selected: Mapping[str, dict[str, Any] | None],
        names: Iterable[str],
        *,
        context: str,
    ) -> None:
        """Write the selected DCP components of one save as collective, rank-converged shard writes.

        Shared by ``save_dcp`` and ``save_consolidated``: one store write per component in ``names``
        order, each under the operation label ``f"{context} component write ({name})"``. Pending-write
        joins, artifact collision checks, committed-checkpoint recovery and the manifest/latest
        publication stay with the callers, in their existing order.
        """
        for name in names:
            state = selected[name]
            if state is None:
                raise AssertionError(f"DCP component plan retained absent component {name!r}")
            run_rank_converged(
                lambda name=name, state=state: self.store.save_dcp_component(location, name, state),
                f"{context} component write ({name})",
            )

    def reader_for_initialization(self, path: str) -> CheckpointManager:
        """Return the manager that reads one model-initialization artifact.

        ``init_from`` object-store artifacts are localized before trainer copy mode. A direct-primary
        run therefore needs a local reader for that path, while exact resume continues through this
        manager's configured primary store.
        """
        if path.startswith("s3://"):
            if not self.store.is_object_store:
                raise ValueError("remote initialization artifacts require an object-store checkpoint reader")
            return self
        if self.store.is_object_store:
            return CheckpointManager(checkpoint_format=self.checkpoint_format)
        return self

    def _local_archive_retention_policy(self, publish: bool) -> tuple[int, int, dict[str, Any] | None]:
        if not publish:
            return 0, 0, None
        if self.keep_optim_last is None:
            raise AssertionError("checkpoint optimizer retention was not resolved")
        return self.keep_optim_last, self.keep_model_last, self.upload

    def save_consolidated(
        self,
        iter_dir: str,
        components: dict[str, dict | None],
        *,
        async_safe_snapshot: bool,
        publish: bool = True,
        update_latest: bool = True,
        component_plan: CheckpointComponentPlan | None = None,
        dcp_components: Mapping[str, dict[str, Any] | None] | None = None,
    ) -> str:
        """Save a full consolidated checkpoint or a commit-marked partial artifact.

        ``dcp_components`` are bulk components written as per-rank DCP shards (a collective write,
        no gather) BEFORE the rank-0 consolidated files, for state whose consolidated form does not
        fit one host: the optimizer of a DTensor-sharded network (two fp32 moments per parameter,
        about 244 GB for a 30 B-parameter model). The resulting iteration directory is the
        consolidated layout with ``<name>/`` in place of ``<name>.pt``; every reader that resolves a
        component through ``component_kind`` accepts it. A name may not be given in both mappings.
        """
        if self.checkpoint_format != "consolidated":
            raise RuntimeError("save_consolidated requires checkpoint_format='consolidated'")
        component_plan = self.component_plan(component_plan)
        if component_plan.is_full and publish and not update_latest:
            raise ValueError("a checkpoint cannot be published before its latest pointer is enabled")
        location = run_rank_converged(
            lambda: self.save_location(iter_dir, component_plan),
            "consolidated checkpoint location",
        )
        location = rank_identical(location, "consolidated checkpoint location")
        # The components arrive gathered; this reading is the post-gather footprint of the process that
        # holds it, against the job cgroup the kernel enforces.
        _log_host_memory(f"checkpoint save {location}")
        selected = run_rank_converged(
            lambda: self._selected_physical_components(components, component_plan),
            "consolidated checkpoint physical component preparation",
        )
        rank_identical(
            tuple(sorted((name, state is not None) for name, state in selected.items())),
            "consolidated checkpoint physical component plan",
        )
        selected_dcp = run_rank_converged(
            lambda: self._selected_physical_components(dict(dcp_components or {}), component_plan),
            "consolidated checkpoint DCP component preparation",
        )
        rank_identical(
            tuple(sorted((name, state is not None) for name, state in selected_dcp.items())),
            "consolidated checkpoint DCP component plan",
        )
        dcp_names = sorted(name for name, state in selected_dcp.items() if state is not None)
        duplicated = [name for name in dcp_names if selected.get(name) is not None]
        if duplicated:
            raise ValueError(f"checkpoint components {duplicated} were given both consolidated and DCP state")
        if not component_plan.is_full:
            run_rank0_converged(join_pending_rank0, "checkpoint artifact write preparation")
            run_rank0_converged(
                lambda: self.store.ensure_artifact_uncommitted(location),
                "consolidated checkpoint artifact collision check",
            )
            self._write_dcp_components(location, selected_dcp, dcp_names, context="consolidated checkpoint DCP")
            args = (
                self.store,
                location,
                self.checkpoint_format,
                component_plan.names,
                dcp_names,
                ArtifactSnapshot(selected),
                self.upload if publish else None,
            )
            # The carrier now holds this frame's only handle on the gathered tensors; the commit
            # releases them as soon as the payload files are written.
            del selected
            if publish and self.save_async and async_safe_snapshot:
                start_background_write(
                    _commit_checkpoint_artifact,
                    args,
                    context="consolidated checkpoint artifact async commit launch",
                )
            else:
                run_rank0_converged(
                    lambda: _commit_checkpoint_artifact(*args),
                    "consolidated checkpoint artifact commit",
                )
            return location
        # Shards first, then the rank-0 files and the latest pointer: an interrupted save leaves at
        # most a pointer-less directory that ``is_complete_iter_dir`` rejects.
        self._write_dcp_components(location, selected_dcp, dcp_names, context="consolidated checkpoint DCP")
        keep_optim_last, keep_model_last, upload = self._local_archive_retention_policy(publish)
        save_components(
            location,
            selected,
            async_write=bool(publish and self.save_async and async_safe_snapshot),
            keep_optim_last=keep_optim_last,
            keep_model_last=keep_model_last,
            upload=upload,
            update_latest=update_latest,
        )
        return location

    def save_dcp(
        self,
        iter_dir: str,
        dcp_components: Mapping[str, dict[str, Any] | None],
        control_components: Mapping[str, dict[str, Any] | None],
        *,
        async_safe_snapshot: bool,
        publish: bool = True,
        update_latest: bool = True,
        component_plan: CheckpointComponentPlan | None = None,
    ) -> str:
        """Save a full DCP checkpoint or a commit-marked partial artifact."""
        if self.checkpoint_format != "dcp":
            raise RuntimeError("save_dcp requires checkpoint_format='dcp'")
        component_plan = self.component_plan(component_plan)
        if component_plan.is_full and publish and not update_latest:
            raise ValueError("a checkpoint cannot be published before its latest pointer is enabled")
        selected_dcp = run_rank_converged(
            lambda: self._selected_physical_components(dcp_components, component_plan),
            "DCP checkpoint component preparation",
        )
        selected_control = run_rank_converged(
            lambda: self._selected_physical_components(control_components, component_plan),
            "DCP checkpoint control component preparation",
        )
        rank_identical(
            tuple(sorted((name, state is not None) for name, state in selected_dcp.items())),
            "DCP checkpoint component plan",
        )
        rank_identical(
            tuple(sorted((name, state is not None) for name, state in selected_control.items())),
            "DCP checkpoint control component plan",
        )
        names = sorted(name for name, state in selected_dcp.items() if state is not None)
        if component_plan.is_full and ("model" not in names or "optim" not in names):
            raise RuntimeError("DCP checkpoint requires model and optimizer components")
        control_names = sorted(name for name, state in selected_control.items() if state is not None)
        run_rank0_converged(join_pending_rank0, "DCP checkpoint write preparation")
        location = run_rank_converged(
            lambda: self.save_location(iter_dir, component_plan),
            "DCP checkpoint iteration location",
        )
        location = rank_identical(location, "DCP checkpoint iteration location")
        _log_host_memory(f"checkpoint save {location}")
        if not component_plan.is_full:
            run_rank0_converged(
                lambda: self.store.ensure_artifact_uncommitted(location),
                "DCP checkpoint artifact collision check",
            )
        elif self.store.is_object_store:
            already_committed = broadcast_from_rank0(
                lambda: self.store.finish_committed_iteration(
                    location,
                    names,
                    control_names,
                    update_latest=update_latest,
                ),
                context="object-store committed checkpoint recovery",
            )
            if already_committed:
                return location
        self._write_dcp_components(location, selected_dcp, names, context="DCP checkpoint")
        if not component_plan.is_full:
            args = (
                self.store,
                location,
                self.checkpoint_format,
                component_plan.names,
                names,
                ArtifactSnapshot(selected_control),
                self.upload if publish else None,
            )
            # Same ownership rule as the consolidated branch: the carrier is this frame's only handle
            # on the control snapshot, and the commit releases it once the payload files are written.
            del selected_control
            run_rank0_converged(
                lambda: _commit_checkpoint_artifact(*args),
                "DCP checkpoint artifact commit",
            )
            return location
        if not self.store.is_object_store:
            keep_optim_last, keep_model_last, upload = self._local_archive_retention_policy(publish)
            save_components(
                location,
                dict(selected_control),
                # DCP tensor writes are already complete. Only this immutable local tail can overlap
                # training, independently of the consolidated ``save_async`` setting.
                async_write=bool(publish and async_safe_snapshot),
                keep_optim_last=keep_optim_last,
                keep_model_last=keep_model_last,
                upload=upload,
                update_latest=update_latest,
            )
            return location
        run_rank0_converged(
            lambda: self.store.commit_iteration(
                location,
                names,
                selected_control,
                update_latest=update_latest,
            ),
            "object-store checkpoint commit",
        )
        return location

    def load_dcp_component(
        self,
        iter_dir: str,
        name: str,
        state: dict[str, Any],
        *,
        allow_partial: bool = False,
    ) -> bool:
        """Load one DCP component through the selected primary backend."""
        iter_dir, name, allow_partial = rank_identical(
            (iter_dir, name, allow_partial),
            "DCP checkpoint component load plan",
        )
        loaded = run_rank_converged(
            lambda: self.store.load_dcp_component(iter_dir, name, state, allow_partial=allow_partial),
            f"DCP checkpoint component load ({name})",
        )
        return rank_identical(loaded, f"DCP checkpoint component presence ({name})")

    def has_dcp_component(self, iter_dir: str, name: str) -> bool:
        """Return rank-identical DCP component presence through the selected primary store."""
        present = run_rank_converged(
            lambda: self.store.has_dcp_component(iter_dir, name),
            f"DCP checkpoint component presence ({name})",
        )
        return rank_identical(present, f"DCP checkpoint component presence ({name})")

    def dcp_tensor_inventory(self, iter_dir: str, name: str) -> dict[str, tuple[int, ...] | None]:
        """Return one rank-identical logical DCP tensor inventory."""
        iter_dir, name = rank_identical((iter_dir, name), "DCP tensor inventory plan")
        inventory = run_rank_converged(
            lambda: self.store.dcp_tensor_inventory(iter_dir, name),
            f"DCP tensor inventory ({name})",
        )
        return rank_identical(inventory, f"DCP tensor inventory ({name})")

    def load_component(self, iter_dir: str, name: str) -> dict[str, Any] | None:
        """Load one small control component through the selected primary backend."""
        iter_dir, name = rank_identical((iter_dir, name), "checkpoint control component load plan")
        component = run_rank_converged(
            lambda: self.store.load_component(iter_dir, name),
            f"checkpoint control component load ({name})",
        )
        rank_identical(component is not None, f"checkpoint control component presence ({name})")
        return component

    def resolve_resume(self, resume: str, ckpt_root: str) -> str | None:
        """Resolve exact resume through the selected primary backend."""
        return self.store.resolve_resume(resume, ckpt_root)

    def iteration_location(self, iter_hint: str) -> str:
        """Resolve a trainer iteration hint to the canonical primary-store location."""
        return self.store.iteration_location(iter_hint)

    def checkpoint_provenance(self, iter_dir: str) -> dict[str, int | str]:
        """Return the selected store's stable checkpoint identity."""
        return self.store.checkpoint_provenance(iter_dir)

    def load_with_stable_provenance(
        self,
        iter_dir: str,
        load: Callable[[], Any],
    ) -> tuple[Any, dict[str, int | str]]:
        """Load only when the selected checkpoint identity stays stable across the read."""
        return _load_with_stable_provenance(
            iter_dir,
            load,
            provenance=self.checkpoint_provenance,
        )

    def load_checkpoint(
        self,
        path: str,
        load: Callable[[], Any],
    ) -> tuple[Any, dict[str, int | str] | None]:
        """Load a checkpoint and return stable provenance when it is directory-shaped."""
        path = rank_identical(path, "checkpoint load path")
        has_provenance = run_rank_converged(
            lambda: self.store.is_object_store or os.path.isdir(path),
            "checkpoint provenance capability",
        )
        has_provenance = rank_identical(has_provenance, "checkpoint provenance capability")
        if has_provenance:
            return self.load_with_stable_provenance(path, load)
        return load(), None

    def replace_component(self, iter_dir: str, name: str, state: dict[str, Any]) -> None:
        """Atomically advance one control component through the selected primary backend."""
        if self.store.is_object_store:
            run_rank0_converged(
                lambda: self.store.replace_component(iter_dir, name, state),
                f"object-store checkpoint component replacement ({name})",
            )
            return
        self.store.replace_component(iter_dir, name, state)

    def publication_identity(self, iter_dir: str) -> dict[str, Any]:
        """Return the non-secret checkpoint generation and publication destination."""
        if self.store.is_object_store:
            return dict(self.store.publication_identity(iter_dir))
        provenance = self.checkpoint_provenance(iter_dir)
        if self.upload:
            from cosmos_framework.posttrain.utils.object_store import upload_destination_identity

            destination = upload_destination_identity(self.upload, os.path.basename(os.path.abspath(iter_dir)))
        else:
            destination = {"kind": "local"}
        return {"destination": destination, "checkpoint_generation": provenance["signature"]}

    def rebind_copied_publication_record(
        self,
        iter_dir: str,
        current_record: Mapping[str, Any],
        expected_record: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        """Rebind one copied direct-primary journal to its configured destination.

        A complete object-store checkpoint can be copied to a new root. Its durable validation
        journal still names the old URI, so continuing publication at the copied root must first
        write a fresh pending record. Local/archive destinations remain immutable across resume.
        """
        if not self.store.is_object_store:
            return None
        current_destination = current_record.get("destination")
        expected_destination = expected_record.get("destination")
        location = self.store.iteration_location(iter_dir)
        current_uri = current_destination.get("uri") if isinstance(current_destination, dict) else None
        if (
            current_record.get("archive_required") is not False
            or expected_record.get("archive_required") is not False
            or not isinstance(current_destination, dict)
            or current_destination.get("kind") != "object_store"
            or not isinstance(current_uri, str)
            or not current_uri.startswith("s3://")
            or expected_destination != {"kind": "object_store", "uri": location}
        ):
            return None
        try:
            canonical_current = self.store.iteration_location(current_uri)
            same_iteration = canonical_current == current_uri and iteration_name(current_uri) == iteration_name(
                location
            )
        except ValueError:
            return None
        return dict(expected_record) if same_iteration else None

    def publish_validation(
        self,
        iter_dir: str,
        *,
        completed_trainer_state: dict[str, Any],
        publication_record: dict[str, Any],
    ) -> None:
        """Strictly publish a validation-complete checkpoint through the shared policy."""
        if self.store.is_object_store:
            run_rank0_converged(
                lambda: self.store.publish_validation(
                    iter_dir,
                    completed_trainer_state=completed_trainer_state,
                    publication_record=publication_record,
                ),
                "object-store validation publication",
            )
            return
        apply_local_validation_archive_retention(
            iter_dir,
            completed_trainer_state=completed_trainer_state,
            publication_record=publication_record,
            keep_optim_last=self.keep_optim_last,
            keep_model_last=self.keep_model_last,
            upload=self.upload,
        )
        self.replace_component(iter_dir, "trainer", completed_trainer_state)

    def finalize(self) -> None:
        """Drain this process's shared background slot and propagate its failure.

        The drain is where a writer's leftover footprint overlaps the resident trainer (a colocated
        job lost its policy actors to the cgroup OOM killer here); the reading is taken in the
        process being joined, not in the controller that asked for the join.
        """
        _log_host_memory("checkpoint finalize")
        finalize()


def _write_files(
    iter_dir: str,
    components: dict[str, dict | None],
    keep_optim_last: int = 0,
    keep_model_last: int = 0,
    upload: dict[str, Any] | None = None,
    update_latest: bool = True,
) -> None:
    """Write consolidated components, publish their pointer, then apply archive policy."""
    if not write_component_files(iter_dir, components):
        return
    # Pointer last: interrupted component writes leave auto-resume on the preceding complete checkpoint.
    if update_latest:
        _update_latest(iter_dir)
    _apply_local_archive_retention(
        iter_dir,
        keep_optim_last,
        keep_model_last,
        upload,
        False,
        None,
        None,
    )


def save_components(
    iter_dir: str,
    components: dict[str, dict | None],
    async_write: bool = False,
    keep_optim_last: int = 0,
    keep_model_last: int = 0,
    upload: dict[str, Any] | None = None,
    update_latest: bool = True,
) -> None:
    """Save consolidated components on rank zero, optionally in the async slot."""
    if not update_latest and (keep_optim_last > 0 or keep_model_last > 0 or upload is not None):
        raise ValueError("checkpoint staging cannot run retention or upload before atomic promotion")

    run_rank0_converged(join_pending_rank0, "checkpoint write preparation")
    args = (iter_dir, components, keep_optim_last, keep_model_last, upload, update_latest)
    if async_write:
        start_background_write(_write_files, args, context="checkpoint async write launch")
        return
    run_rank0_converged(lambda: _write_files(*args), "checkpoint write")


def finalize() -> None:
    """Join any in-flight checkpoint worker before process exit."""
    wait_for_pending()


def _apply_local_archive_retention(
    iter_dir: str,
    keep_optim_last: int,
    keep_model_last: int,
    upload: dict[str, Any] | None,
    require_archive_success: bool,
    archive_overrides: dict[str, str] | None,
    archive_commit: dict[str, Any] | None,
    optimizer_subdirs: tuple[str, ...] = ("",),
) -> None:
    """Apply archive and retention policy to an already-complete local checkpoint."""
    root = os.path.dirname(os.path.abspath(iter_dir))
    if upload:
        try:
            archive_checkpoint(iter_dir, upload, overrides=archive_overrides, commit=archive_commit)
        except Exception as error:  # noqa: BLE001 -- training's best-effort policy is outside the publisher
            if require_archive_success:
                raise
            logger.warning(f"checkpoint upload failed ({error}); keeping the local copy")
            return
        # The marker is the archive gate for BOTH optimizer and whole-model retention.  A best-effort
        # upload failure keeps the complete local working set; deleting optimizer state first would
        # leave an older checkpoint neither resumable locally nor proven durable remotely.
        if not os.path.isfile(os.path.join(iter_dir, ".uploaded")):
            return
    prune_local_optimizer_state(
        root,
        keep_optim_last,
        require_uploaded=bool(upload),
        optimizer_subdirs=optimizer_subdirs,
    )
    if keep_model_last > 0:
        prune_local_iterations(root, keep_model_last, require_uploaded=bool(upload))


def apply_local_archive_retention(
    iter_dir: str,
    *,
    keep_optim_last: int = 0,
    keep_model_last: int = 0,
    upload: dict[str, Any] | None = None,
    async_archive_retention: bool = False,
    require_archive_success: bool = False,
    archive_overrides: dict[str, str] | None = None,
    archive_commit: dict[str, Any] | None = None,
    optimizer_subdirs: tuple[str, ...] = ("",),
) -> None:
    """Apply archive/retention to an existing local checkpoint, optionally in the async slot."""
    if require_archive_success and upload is None:
        raise ValueError("require_archive_success=True requires an upload configuration")
    if require_archive_success and async_archive_retention:
        raise ValueError("strict checkpoint archive must be synchronous")
    if (archive_overrides is not None or archive_commit is not None) and not require_archive_success:
        raise ValueError("transactional archive metadata requires a strict checkpoint archive")
    if (archive_overrides is None) != (archive_commit is None):
        raise ValueError("transactional archive requires both overrides and a commit manifest")
    if not optimizer_subdirs or any(
        not isinstance(name, str) or (name and (os.path.basename(name) != name or name in (".", "..")))
        for name in optimizer_subdirs
    ):
        raise ValueError("optimizer_subdirs must contain only canonical single-directory names or the empty root")

    from cosmos_framework.posttrain.checkpoint.worker import (
        join_pending_rank0,
        run_rank0_converged,
        start_background_write,
    )

    def prepare() -> None:
        join_pending_rank0()
        if not os.path.isdir(iter_dir):
            raise FileNotFoundError(f"cannot apply archive/retention to missing checkpoint {iter_dir!r}")

    run_rank0_converged(prepare, "local checkpoint archive/retention preparation")
    args = (
        iter_dir,
        keep_optim_last,
        keep_model_last,
        upload,
        require_archive_success,
        archive_overrides,
        archive_commit,
        optimizer_subdirs,
    )
    if async_archive_retention:
        start_background_write(
            _apply_local_archive_retention,
            args,
            context="checkpoint async archive/retention launch",
        )
        return
    run_rank0_converged(
        lambda: _apply_local_archive_retention(*args),
        "local checkpoint archive/retention",
    )


def apply_local_validation_archive_retention(
    iter_dir: str,
    *,
    completed_trainer_state: dict[str, Any],
    publication_record: dict[str, Any],
    keep_optim_last: int,
    keep_model_last: int,
    upload: dict[str, Any] | None,
) -> None:
    """Archive/retain a local validation successor with completed trainer state."""
    completed_path: str | None = None
    if upload:
        from cosmos_framework.posttrain.utils.dist import broadcast_from_rank0

        def _stage() -> str:
            import torch

            descriptor, path = tempfile.mkstemp(
                prefix=f".{os.path.basename(os.path.abspath(iter_dir))}.completed-trainer-",
                suffix=".pt",
                dir=os.path.dirname(os.path.abspath(iter_dir)),
            )
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    torch.save(completed_trainer_state, handle)
                    handle.flush()
                    os.fsync(handle.fileno())
            except BaseException:  # noqa: BLE001 -- failed staging must not leak checkpoint state
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass
                raise
            return path

        completed_path = broadcast_from_rank0(_stage, context="validation completed-trainer preparation")
    archive_retention_error: BaseException | None = None
    try:
        apply_local_archive_retention(
            iter_dir,
            keep_optim_last=keep_optim_last,
            keep_model_last=keep_model_last,
            upload=upload,
            async_archive_retention=False,
            require_archive_success=bool(upload),
            archive_overrides={"trainer.pt": completed_path} if completed_path else None,
            archive_commit=publication_record if upload else None,
        )
    except BaseException as error:  # noqa: BLE001 -- rank-zero cleanup must still run
        archive_retention_error = error
    cleanup_error: BaseException | None = None
    if completed_path is not None:
        from cosmos_framework.posttrain.checkpoint.worker import run_rank0_converged

        def _cleanup() -> None:
            try:
                os.remove(completed_path)
            except FileNotFoundError:
                pass

        try:
            run_rank0_converged(_cleanup, "validation completed-trainer cleanup")
        except BaseException as error:  # noqa: BLE001 -- preserve archive/retention failure precedence
            cleanup_error = error
    if archive_retention_error is not None:
        raise archive_retention_error
    if cleanup_error is not None:
        raise cleanup_error
