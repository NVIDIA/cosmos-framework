# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Control-plane checkpoint sequencing shared by offline and online loops."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from cosmos_framework.posttrain.checkpoint import CheckpointManager
from cosmos_framework.posttrain.control.lifecycle import ControllerLifecycleObserver, ControllerLifecycleObservers
from cosmos_framework.posttrain.schema.checkpoint import (
    EMERGENCY_CHECKPOINT_PLANS,
    FULL_CHECKPOINT_COMPONENTS,
    CheckpointComponentPlan,
)
from cosmos_framework.posttrain.utils.entry import resolve_all, resolve_identical
from cosmos_framework.posttrain.utils.log import info0


def _drain_checkpoint_finalization(role: Any, *, role_name: str) -> None:
    """Wait for every role cell and surface the first checkpoint-finalization failure."""
    from cosmos_framework.posttrain.utils.log import logger

    handle = role.finalize_checkpoints()
    if handle is None:
        return
    as_completed = getattr(handle, "as_completed", None)
    if callable(as_completed):
        failures: list[tuple[Any, BaseException]] = []
        for cell, outcome in as_completed():
            if isinstance(outcome, BaseException):
                failures.append((cell, outcome))
        if failures:
            for cell, error in failures[1:]:
                logger.error(f"checkpoint finalize also failed on {role_name} cell {cell}: {error}")
            raise failures[0][1]
        return
    all_results = getattr(handle, "all", None)
    if callable(all_results):
        all_results()
        return
    handle.result()


def finalize_checkpoints_before_teardown(
    units: Mapping[str, Any],
    teardown: Callable[[], None] | None,
) -> None:
    """Drain every trainer role before teardown without masking an active failure."""
    import sys

    from cosmos_framework.posttrain.utils.log import logger

    active_error = sys.exc_info()[1] is not None
    finalization_errors: list[tuple[str, BaseException]] = []
    teardown_error: BaseException | None = None
    for role_name, role in units.items():
        role_kind = getattr(role, "__dict__", {}).get("role")
        if role_kind is not None and role_kind not in ("trainer", "critic"):
            continue
        if not callable(getattr(role, "finalize_checkpoints", None)):
            continue
        try:
            _drain_checkpoint_finalization(role, role_name=role_name)
        except BaseException as error:  # noqa: BLE001 -- cancellation must not skip teardown or mask an active error
            finalization_errors.append((role_name, error))
            logger.error(f"{role_name} checkpoint finalize failed: {error}")
    if teardown is not None:
        try:
            teardown()
        except BaseException as error:  # noqa: BLE001 -- preserve the run/finalization failure when present
            teardown_error = error
            logger.error(f"process-group teardown failed: {error}")
    if active_error:
        return
    if finalization_errors:
        raise finalization_errors[0][1]
    if teardown_error is not None:
        raise teardown_error


def finalize_policy_checkpoints_before_teardown(
    units: Mapping[str, Any],
    teardown: Callable[[], None] | None,
) -> None:
    """Compatibility alias for the now role-generic finalization helper."""
    finalize_checkpoints_before_teardown(units, teardown)


def _localize_checkpoint(path: str) -> str:
    """Materialize an S3 checkpoint in the shared cache; pass local paths through."""
    if not path.startswith("s3://"):
        return path
    root = os.environ.get("POSTTRAIN_CKPT_CACHE") or os.path.join(
        os.environ.get("IMAGINAIRE_CACHE_DIR", os.path.expanduser("~/.cache/imaginaire")),
        "posttrain_ckpts",
    )
    local = os.path.join(root, path[len("s3://") :].rstrip("/"))

    def pull() -> str:
        from cosmos_framework.posttrain.utils.object_store import _parse_s3_url, download_prefix

        bucket, prefix = _parse_s3_url(path)
        download_prefix(bucket, prefix, local)
        return local

    from cosmos_framework.posttrain.utils.dist import broadcast_from_rank0

    return str(broadcast_from_rank0(pull, context="checkpoint download"))


def _checkpoint_object_store_prefix(config: Any, configured_prefix: str) -> str:
    """Resolve an explicit prefix or mirror the canonical local checkpoint layout."""
    prefix = configured_prefix
    if not prefix:
        from cosmos_framework.posttrain.utils.entry import out_dir

        root = os.environ.get("IMAGINAIRE_OUTPUT_ROOT", "/tmp/imaginaire-output")
        relative = os.path.relpath(os.path.join(out_dir(config), "checkpoints"), root)
        prefix = relative.replace(os.sep, "/")
    return prefix


def _checkpoint_store_spec(config: Any, rl: Any) -> dict[str, Any]:
    """Lower checkpoint-store config and expand only its run-relative default prefix."""
    store = getattr(rl, "checkpoint_store", None)
    backend = getattr(store, "backend", "local")
    if backend == "local":
        return {"backend": "local"}
    configured_prefix = getattr(store, "prefix", "")
    prefix = (
        _checkpoint_object_store_prefix(config, configured_prefix)
        if isinstance(configured_prefix, str)
        else configured_prefix
    )
    return {
        "backend": backend,
        "bucket": getattr(store, "bucket", ""),
        "prefix": prefix,
        "credentials": getattr(store, "credentials", ""),
    }


def _explicit_resume_checkpoint(resume: object) -> str | None:
    """Normalize one explicit local or S3 resume value for validation coordination."""
    value = str(resume)
    if value in ("", "off", "auto"):
        return None
    if value.startswith("s3://"):
        return value.rstrip("/")
    return os.path.realpath(os.path.expanduser(value))


def _checkpoint_upload_spec(config: Any, rl: Any) -> dict[str, str] | None:
    """Resolve the optional archive destination for this run's checkpoint layout."""
    upload = getattr(rl, "upload", None)
    if upload is None or not bool(getattr(upload, "enabled", False)):
        return None
    return {
        "bucket": str(upload.bucket),
        "prefix": _checkpoint_object_store_prefix(config, upload.prefix),
        "credentials": str(upload.credentials),
    }


@dataclass(frozen=True)
class CheckpointLifecycle:
    """Backend-neutral checkpoint assembly and resume lifecycle used by entrypoints.

    Algorithms still choose their safe save boundaries. This facade owns every storage-specific
    decision so entrypoints never branch on local versus direct object-store checkpoints.
    """

    manager: CheckpointManager
    checkpoint_root: str
    resume_checkpoint: str | None

    @classmethod
    def from_config(
        cls,
        config: Any,
        rl: Any,
        *,
        resume_checkpoint: str | None = None,
        run_out_dir: str | None = None,
    ) -> CheckpointLifecycle:
        """Build one manager and consume only an already-explicit worker resume selection."""
        from cosmos_framework.posttrain.utils.dist import rank_identical, run_rank_converged

        root = os.path.join(run_out_dir or _run_out_dir(config), "checkpoints")
        selected_resume = getattr(rl, "resume", "off") if resume_checkpoint is None else resume_checkpoint
        resume = run_rank_converged(
            lambda: _explicit_resume_checkpoint(selected_resume),
            "explicit checkpoint resume selection",
        )
        resume = rank_identical(resume, "explicit checkpoint resume selection")
        manager = CheckpointManager(
            checkpoint_format=str(getattr(rl, "checkpoint_format", "consolidated")),
            save_async=bool(getattr(rl, "save_async", False)),
            keep_optim_last=getattr(rl, "keep_optim_last", None),
            keep_model_last=int(getattr(rl, "keep_model_last", 0)),
            checkpoint_components=getattr(rl, "checkpoint_components", None),
            checkpoint_store=_checkpoint_store_spec(config, rl),
            upload=_checkpoint_upload_spec(config, rl),
        )
        return cls(manager=manager, checkpoint_root=root, resume_checkpoint=resume)

    @staticmethod
    def worker_resume_checkpoint(rl: Any) -> str | None:
        """The checkpoint a worker's trainer restores, read from its driver-pinned ``rl.resume``.

        ``from_config`` selects the same value. This reads it without a manager or a collective, so
        a role sharing the trainer's GPUs can prepare for the restore before the trainer exists.
        """
        return _explicit_resume_checkpoint(getattr(rl, "resume", "off"))

    def resolve_resume(self, resume: object) -> str | None:
        """Resolve ``auto``/``latest`` once on the driver before worker options are pinned."""
        from cosmos_framework.posttrain.utils.dist import rank_identical, run_rank_converged

        checkpoint = run_rank_converged(
            lambda: self.manager.resolve_resume(str(resume), self.checkpoint_root),
            "checkpoint resume resolution",
        )
        return rank_identical(checkpoint, "checkpoint resume selection")

    def localize_init_from(self, checkpoint: object) -> str:
        """Materialize a model-initialization artifact when the underlying loader needs a path."""
        return _localize_checkpoint(str(checkpoint))

    def load_trainer(self, trainer: Any, checkpoint: str | None = None) -> int | None:
        """Restore an exact trainer checkpoint and bind its stable generation when applicable."""
        selected = checkpoint or self.resume_checkpoint
        if selected is None:
            return None
        iteration, provenance = self.manager.load_checkpoint(selected, lambda: trainer.load(selected))
        if provenance is not None:
            trainer.set_loaded_checkpoint_provenance(provenance)
        return int(iteration)

    def load_trainer_and_component(
        self,
        trainer: Any,
        component: str,
        checkpoint: str | None = None,
    ) -> tuple[int, dict[str, Any] | None] | None:
        """Restore a trainer and one rank-aligned control component from the same generation."""
        selected = checkpoint or self.resume_checkpoint
        if selected is None:
            return None

        def load() -> tuple[int, dict[str, Any] | None]:
            return int(trainer.load(selected)), self.manager.load_component(selected, component)

        value, provenance = self.manager.load_checkpoint(selected, load)
        if provenance is not None:
            trainer.set_loaded_checkpoint_provenance(provenance)
        return value

    def restore_cursor(self, env: Any, cursor_resume: str, checkpoint: str | None = None) -> None:
        """Restore one data cursor through the same manager and selected checkpoint as the trainer."""
        selected = checkpoint or self.resume_checkpoint
        if selected is None:
            return
        from cosmos_framework.posttrain.roles.env.cursor import restore_env_cursor

        restore_env_cursor(
            env,
            selected,
            cursor_resume,
            checkpoint_manager=self.manager,
        )

    def coordinator(
        self,
        out_dir: str,
        policy: Any,
        *,
        data: Any | None = None,
        initial_step: int = 0,
        observer: ControllerLifecycleObserver | None = None,
        before_save: Callable[[int], None] | None = None,
        after_save: Callable[[int], None] | None = None,
    ) -> CheckpointCoordinator:
        """Build the loop-side coordinator without exposing its storage backend.

        ``observer`` receives the checkpoint lifecycle events of every ACTUAL save this coordinator
        performs (`control/lifecycle.py`); the loop never wraps individual save calls. Optional save
        hooks let a placement owner acquire and release an exclusive trainer-memory window around
        the actual write without teaching this backend-neutral coordinator about GPU roles.
        """
        return CheckpointCoordinator(
            out_dir=out_dir,
            policy=policy,
            manager=self.manager,
            data=data,
            resume_checkpoint=self.resume_checkpoint,
            initial_step=initial_step,
            observer=observer,
            before_save=before_save,
            after_save=after_save,
        )

    def multi_role_coordinator(
        self,
        roles: Mapping[str, Any],
        *,
        data: Any | None = None,
        state_role: str = "policy",
        initial_step: int = 0,
        observer: ControllerLifecycleObserver | None = None,
    ) -> MultiRoleCheckpointCoordinator:
        """Build one atomic boundary coordinator for several trainer roles."""
        return MultiRoleCheckpointCoordinator(
            checkpoint_root=self.checkpoint_root,
            roles=roles,
            manager=self.manager,
            data=data,
            state_role=state_role,
            initial_step=initial_step,
            observer=observer,
        )

    @staticmethod
    def finalize_units(units: Mapping[str, Any], teardown: Callable[[], None] | None = None) -> None:
        """Drain checkpoint work before optional process-group teardown."""
        finalize_checkpoints_before_teardown(units, teardown)


def _run_out_dir(config: Any) -> str:
    """Import lazily to keep this control module usable in lightweight setup workers."""
    from cosmos_framework.posttrain.utils.entry import out_dir

    return out_dir(config)


def save_policy_checkpoint(
    policy: Any,
    checkpoint: str,
    *,
    data: Any | None = None,
    components: Iterable[str] | CheckpointComponentPlan | None = None,
    publish: bool = True,
    update_latest: bool = True,
) -> None:
    """Save every policy cell, optionally with its rank-aligned data cursor."""
    component_plan = CheckpointComponentPlan.from_value(components) if components is not None else None
    arguments = [checkpoint]
    keywords: dict[str, Any] = {}
    if data is not None and (component_plan is None or component_plan.includes("training_state")):
        # Keep the mesh result unresolved so the dispatcher aligns each data rank to its policy rank.
        keywords["data_state"] = data.state_dict()
    if not publish:
        keywords["publish"] = False
    if not update_latest:
        keywords["update_latest"] = False
    if component_plan is not None:
        keywords["components"] = component_plan.names
    resolve_all(policy.save(*arguments, **keywords))


@dataclass
class CheckpointCoordinator:
    """Own checkpoint paths, save deduplication, and durable validation sequencing.

    Every ACTUAL save (cadence, validation transaction, emergency ``ensure_resumable`` upgrade) funnels
    through :meth:`save`, which is where the checkpoint lifecycle events are emitted to ``observer``
    (`control/lifecycle.py`): start before the write, end after it whether or not it raised, success
    after a committed write. A request deduplicated by ``_latest_saved_step`` /
    ``_latest_resumable_step`` emits nothing, so an observer never sees a save that did not happen.
    """

    out_dir: str
    policy: Any
    manager: CheckpointManager
    data: Any | None = None
    resume_checkpoint: str | None = None
    initial_step: int = 0
    observer: ControllerLifecycleObserver | None = None
    before_save: Callable[[int], None] | None = field(default=None, repr=False)
    after_save: Callable[[int], None] | None = field(default=None, repr=False)
    last_save_location: str | None = field(init=False, default=None)
    _latest_saved_step: int = field(init=False, repr=False)
    _latest_resumable_step: int = field(init=False, repr=False)
    # Components this coordinator has written per step: the emergency save under the "run" plan decides its skip by
    # component coverage of the step (a validation-only artifact is not "already checkpointed" for the weights).
    _saved_components: dict[int, frozenset[str]] = field(init=False, repr=False, default_factory=dict)
    _observer: ControllerLifecycleObservers = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.initial_step, bool) or not isinstance(self.initial_step, int) or self.initial_step < 0:
            raise ValueError(f"checkpoint initial_step must be >= 0, got {self.initial_step!r}")
        self._latest_saved_step = self.initial_step
        self._latest_resumable_step = self.initial_step
        # The fan-out list guards every observer call, so a raising observer never fails a save. A raw
        # observer handed to the coordinator is delivered inline: the coordinator sees no train/app end
        # and must not own a worker thread; the entry's list passes through with its own dispatch.
        self._observer = ControllerLifecycleObservers.coerce(self.observer, dispatch="sync")

    def path(self, model_step: int) -> str:
        """Return the canonical checkpoint directory for a committed model step."""
        if isinstance(model_step, bool) or not isinstance(model_step, int) or model_step < 0:
            raise ValueError(f"checkpoint model_step must be >= 0, got {model_step!r}")
        return os.path.join(self.out_dir, "checkpoints", f"iter_{model_step:06d}")

    def resolve_component_plan(
        self,
        *,
        include_configured: bool,
        additional: Iterable[str] | CheckpointComponentPlan | None = None,
    ) -> CheckpointComponentPlan | None:
        """Resolve the union of an optional configured plan and one additional request."""
        plans: list[CheckpointComponentPlan] = []
        if include_configured:
            plans.append(self.manager.component_plan())
        if additional is not None:
            plans.append(CheckpointComponentPlan.from_value(additional))
        if not plans:
            return None
        selected = {
            component for component in FULL_CHECKPOINT_COMPONENTS if any(plan.includes(component) for plan in plans)
        }
        return self.manager.component_plan(selected)

    def _save_path(
        self,
        checkpoint: str,
        *,
        components: Iterable[str] | CheckpointComponentPlan | None = None,
        publish: bool,
        update_latest: bool = True,
    ) -> CheckpointComponentPlan:
        """Save one explicit target and drain every policy cell."""
        component_plan = self.manager.component_plan(components)
        if component_plan.is_full:
            # A full save is the one that grows the trainer's host footprint (the gathered model on
            # rank 0); the trainer logs its RSS and the node's free memory when it starts the gather.
            info0(f"saving full checkpoint {checkpoint} components={list(component_plan.names)}")
        save_policy_checkpoint(
            self.policy,
            checkpoint,
            data=self.data if component_plan.includes("training_state") else None,
            # Configured cadence/final saves use the same manager already injected into every
            # trainer, so preserve their existing call shape. Runtime union/full overrides remain
            # explicit because they may differ from that configured default.
            components=component_plan if components is not None else None,
            publish=publish,
            update_latest=bool(update_latest and component_plan.is_full),
        )
        location = self.manager.save_location(checkpoint, component_plan)
        self.last_save_location = location
        info0(
            f"saved checkpoint {location} components={list(component_plan.names)} "
            f"resumable={component_plan.is_full} publish={publish}"
        )
        return component_plan

    def save(
        self,
        model_step: int,
        *,
        components: Iterable[str] | CheckpointComponentPlan | None = None,
        publish: bool = True,
        window_held: bool = False,
    ) -> None:
        """Save the canonical checkpoint for ``model_step``.

        ``window_held`` means the caller already holds the exclusive trainer-memory window and hands
        the card back itself, so neither save hook runs.
        """
        if self.before_save is not None and not window_held:
            self.before_save(model_step)
        self._observer.on_checkpoint_save_start(model_step)
        try:
            component_plan = self._save_path(
                self.path(model_step),
                components=components,
                publish=publish,
            )
        finally:
            self._observer.on_checkpoint_save_end(model_step)
        self._latest_saved_step = max(self._latest_saved_step, model_step)
        self._saved_components[model_step] = (
            self._saved_components.get(model_step, frozenset()) | component_plan.selected
        )
        if component_plan.is_full:
            self._latest_resumable_step = max(self._latest_resumable_step, model_step)
        self._observer.on_checkpoint_save_success(model_step, location=self.last_save_location)
        if self.after_save is not None and not window_held:
            self.after_save(model_step)

    def save_configured_if_needed(self, model_step: int, *, window_held: bool = False) -> bool:
        """Save the configured cadence/final plan unless this step was already saved."""
        if self._latest_saved_step >= model_step:
            return False
        self.save(model_step, window_held=window_held)
        return True

    def save_full(self, model_step: int, *, publish: bool = True) -> None:
        """Save every component required for exact resume."""
        self.save(model_step, components=FULL_CHECKPOINT_COMPONENTS, publish=publish)

    def ensure_resumable(self, model_step: int) -> bool:
        """Ensure ``model_step`` has a full checkpoint, upgrading a partial save if needed."""
        if self._latest_resumable_step >= model_step:
            return False
        self.save_full(model_step)
        return True

    def save_emergency(self, model_step: int, *, components: str = "full") -> bool:
        """Save the termination-triggered checkpoint for ``model_step``; True when a save happened.

        ``components`` is ``rl.emergency_checkpoint_components``: ``"full"`` is :meth:`ensure_resumable`
        (the full plan, upgrading a partial cadence save of this step); ``"run"`` writes the run's
        configured plan UNION the model — an emergency save always carries the weights — through the
        cadence path (same location, same commit marker) and skips only when every component of that
        plan is already on disk for THIS step (:meth:`_save_run_plan_with_model`). A partial ``"run"``
        artifact is not exact-resume capable; the config docstring states the trade.
        """
        if components == "full":
            return self.ensure_resumable(model_step)
        if components == "run":
            return self._save_run_plan_with_model(model_step)
        raise ValueError(
            f"emergency checkpoint components must be one of {list(EMERGENCY_CHECKPOINT_PLANS)}, got {components!r}"
        )

    def _save_run_plan_with_model(self, model_step: int) -> bool:
        """Write the run's plan plus the model for ``model_step`` unless this step already holds all of it.

        The cadence path dedups by step number alone, which let a validation-only artifact
        (``validation.checkpoint_components=[training_state]``) of the same step answer "already
        checkpointed" to an emergency save whose run plan was ``[model]``: no weights were written
        (review of e11b239d02). Coverage is decided per component here. A step that is already partially
        saved owns a committed artifact, and a committed artifact is never re-committed
        (``store.commit_artifact`` refuses a second manifest), so a missing model could only go into the
        step's full checkpoint: the save escalates to the full plan, with a log line. No accepted config
        reaches that branch — under "run" a saving validation plan must COVER the run plan plus the model
        (``RunConfig.__post_init__``; the model alone was not enough: run plan ``[model, training_state]``
        with validation plan ``[model]`` escalated here and added the optimizer), so the one artifact a
        step can hold before the emergency save already holds every component of this plan; the branch
        is the fallback for a coordinator driven outside that contract. A
        step with nothing saved gets the union plan as one artifact — through the cadence call shape (no
        override) when it IS the configured plan, so the usual model-only run writes exactly its cadence
        artifact.
        """
        if self._latest_resumable_step >= model_step:
            return False  # a full checkpoint of this step (or a later one) already exists
        plan = self.resolve_component_plan(include_configured=True, additional=("model",))
        if plan is None:  # pragma: no cover -- include_configured=True always yields a plan
            raise AssertionError("emergency component plan resolved to nothing")
        saved = self._saved_components.get(model_step, frozenset())
        if plan.selected <= saved:
            return False
        if saved:
            info0(
                f"emergency save: step {model_step} already holds a committed partial artifact "
                f"{sorted(saved)} without {sorted(plan.selected - saved)}; a committed artifact is never "
                "re-committed, so the emergency save writes the full plan"
            )
            self.save_full(model_step)
            return True
        configured = self.manager.component_plan()
        self.save(model_step, components=None if plan.selected == configured.selected else plan)
        return True

    def persist_validation_completion(self, model_step: int) -> None:
        """Publish validation completion and atomically clear its recovery journal."""
        checkpoint = self.path(model_step)
        resolve_all(self.policy.save_validation_progress(checkpoint))
        location = self.manager.iteration_location(checkpoint)
        info0(f"persisted validation completion in {location}")

    def _loaded_provenance(self) -> dict[str, int | str]:
        policy_provenance = resolve_identical(
            self.policy.loaded_checkpoint_provenance(),
            operation="policy.loaded_checkpoint_provenance",
        )
        if policy_provenance is None:
            raise RuntimeError("external pending resume lacks setup-time checkpoint provenance")
        if self.data is None:
            return policy_provenance
        data_provenance_fn = getattr(self.data, "loaded_checkpoint_provenance", None)
        if not callable(data_provenance_fn):
            raise RuntimeError("external pending resume lacks data checkpoint provenance")
        data_provenance = resolve_identical(
            data_provenance_fn(),
            operation="data.loaded_checkpoint_provenance",
        )
        if data_provenance != policy_provenance:
            raise RuntimeError(
                "DPO data cursor and policy loaded different external checkpoint revisions: "
                f"data={data_provenance!r} policy={policy_provenance!r}"
            )
        return policy_provenance

    def copy_external_pending(self, model_step: int) -> None:
        """Copy an immutable external pending checkpoint into the current run atomically."""
        source = self.resume_checkpoint
        if source is None:
            return
        store = self.manager.store
        if store.is_object_store:
            if not source.startswith("s3://"):
                raise RuntimeError(
                    f"object-store pending-checkpoint resume requires an explicit s3:// source; got {source!r}"
                )
            source_location = store.iteration_location(source)
            target_location = store.iteration_location(self.path(model_step))
            if source_location == target_location:
                return
            raise RuntimeError(
                "cross-prefix object-store resume with pending validation publication is unsupported; "
                f"refusing to mutate source {source_location!r} while the current run owns {target_location!r}"
            )
        target = os.path.realpath(self.path(model_step))
        if os.path.realpath(source) == target:
            return
        from cosmos_framework.posttrain.checkpoint import (
            checkpoint_provenance,
            prepare_external_checkpoint_copy,
            promote_external_checkpoint_copy,
        )

        provenance = self._loaded_provenance()
        if os.path.isdir(target) and checkpoint_provenance(target) == provenance:
            info0(f"reusing already-loaded external validation-journal successor {target}")
            return
        plan = prepare_external_checkpoint_copy(source, target, expected_provenance=provenance)
        if plan.staging_dir is None:
            info0(f"reusing previously promoted external pending checkpoint {target}")
            return
        # The staging write restores trainer state like any save, so it takes the same exclusive
        # trainer-memory window; a failed write leaves it acquired, as `save` does.
        if self.before_save is not None:
            self.before_save(model_step)
        self._save_path(
            plan.staging_dir,
            components=FULL_CHECKPOINT_COMPONENTS,
            publish=False,
            update_latest=False,
        )
        promote_external_checkpoint_copy(plan)
        info0(f"atomically promoted external pending checkpoint to {target}")
        if self.after_save is not None:
            self.after_save(model_step)


_MULTI_ROLE_SUCCESS = "_SUCCESS"
# The marker is JSON, so a slot value has to survive a JSON round trip EXACTLY: these are the types
# `json` reproduces without loss, which is what makes a read-back comparison meaningful.
_LOOP_STATE_VALUE_TYPES = (bool, int, float, str)


def _validated_loop_state(loop_state: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate one caller-owned loop-state mapping before it can reach a committed marker."""
    if loop_state is None:
        return {}
    if not isinstance(loop_state, Mapping):
        raise ValueError(f"checkpoint loop_state must be a mapping, got {loop_state!r}")
    validated: dict[str, Any] = {}
    for key, value in loop_state.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"checkpoint loop_state keys must be non-empty strings, got {key!r}")
        if not isinstance(value, _LOOP_STATE_VALUE_TYPES):
            raise ValueError(f"checkpoint loop_state value for {key!r} must be a JSON scalar, got {value!r}")
        validated[key] = value
    return validated


def _marker_loop_state(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Read one boundary's recorded loop state; absent and empty both mean "recorded nothing"."""
    return dict(payload.get("loop_state") or {})


@dataclass
class MultiRoleCheckpointCoordinator:
    """Atomically publish one resumable boundary spanning named trainer roles."""

    checkpoint_root: str
    roles: Mapping[str, Any]
    manager: CheckpointManager
    data: Any | None = None
    state_role: str = "policy"
    initial_step: int = 0
    observer: ControllerLifecycleObserver | None = None
    _latest_saved_step: int = field(init=False, repr=False)
    _committed_loop_state: dict[str, Any] = field(init=False, repr=False)
    _observer: ControllerLifecycleObservers = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.roles or any(not isinstance(name, str) or not name for name in self.roles):
            raise ValueError("multi-role checkpoints require named trainer roles")
        if any(os.path.basename(name) != name or name in (".", "..") for name in self.roles):
            raise ValueError("multi-role checkpoint names must be canonical single-directory names")
        if self.state_role not in self.roles:
            raise ValueError(f"checkpoint state_role {self.state_role!r} is not one of {sorted(self.roles)}")
        if type(self.initial_step) is not int or self.initial_step < 0:
            raise ValueError(f"checkpoint initial_step must be a non-negative integer, got {self.initial_step!r}")
        if self.manager.store.is_object_store:
            raise NotImplementedError("multi-role checkpoint boundaries currently require the shared local store")
        self.roles = dict(self.roles)
        self._latest_saved_step = self.initial_step
        self._committed_loop_state = {}
        # As in CheckpointCoordinator: inline for a raw observer, the entry's list keeps its own dispatch.
        self._observer = ControllerLifecycleObservers.coerce(self.observer, dispatch="sync")

    def path(self, model_step: int) -> str:
        """Return the canonical directory for one complete multi-role boundary."""
        if type(model_step) is not int or model_step < 0:
            raise ValueError(f"checkpoint model_step must be a non-negative integer, got {model_step!r}")
        return os.path.join(self.checkpoint_root, f"iter_{model_step:06d}")

    def committed_step(self) -> int:
        """Return the newest boundary this coordinator has committed or resumed from.

        The ``_SUCCESS`` marker records the model step its boundary was published under, and
        ``resolve_resume`` reads it back before any role loads. A loop whose durable step is NOT one
        of its roles' optimizer clocks -- PPO's outer iteration, which stands still for the actor
        during critic-only warmup -- recovers it from here rather than reconstructing it from a role
        counter that no longer moves in lockstep with it.
        """
        return self._latest_saved_step

    def committed_loop_state(self) -> dict[str, Any]:
        """Return the loop state carried by the boundary just committed or resumed from, else ``{}``.

        One OPAQUE run-level slot the caller owns and this coordinator never interprets. A loop can
        therefore persist a configured choice whose effect is invisible in every role counter -- a
        length measured in outer iterations, during which one role deliberately does not step, so its
        counter reads the same under every such length -- and compare it DIRECTLY on resume instead
        of inferring it from a counter that cannot express it. Giving the slot a schema here would
        bind this shared boundary to whichever algorithm defined that schema.
        """
        return dict(self._committed_loop_state)

    def role_path(self, boundary: str, role: str) -> str:
        """Return one named role's shared-manager checkpoint inside a boundary."""
        if role not in self.roles:
            raise KeyError(f"unknown checkpoint role {role!r}")
        return os.path.join(boundary, role)

    def _success_payload(
        self,
        model_step: int,
        role_steps: Mapping[str, int],
        loop_state: Mapping[str, Any],
    ) -> dict[str, Any]:
        # `version` stays 1 for `loop_state` exactly as it did for `role_steps`: both are ADDITIVE
        # optional fields whose absence the reader already handles, and whose presence a reader that
        # predates them ignores. Bumping it would make every boundary written here unreadable to the
        # previous binary -- `_read_success` rejects any version but its own -- so a rollback in the
        # middle of a run could not resume a checkpoint it had just written, buying nothing.
        return {
            "version": 1,
            "model_step": model_step,
            "roles": sorted(self.roles),
            "state_role": self.state_role,
            "role_steps": {role: role_steps[role] for role in sorted(self.roles)},
            "loop_state": dict(loop_state),
        }

    def _committed_role_steps(self, boundary: str) -> dict[str, int]:
        """Read back the optimizer counter each role just committed into ``boundary``.

        Recorded because roles in one boundary are NOT required to share an optimizer counter: an
        algorithm may deliberately hold one role still while another trains (PPO's critic-only
        warmup). Comparing the roles only against EACH OTHER can no longer express that, so the
        boundary records what each role actually wrote and restore checks every role against its own
        committed value -- which also rejects a boundary whose roles were all corrupted in agreement.
        """
        steps: dict[str, int] = {}
        for role in self.roles:
            role_path = self.role_path(boundary, role)
            state = self.manager.load_component(role_path, "trainer")
            if not isinstance(state, dict):
                raise RuntimeError(f"multi-role checkpoint role {role!r} committed no trainer state at {boundary!r}")
            step = state.get("optimizer_step", state.get("iter"))
            if type(step) is not int or step < 0:
                raise RuntimeError(
                    f"multi-role checkpoint role {role!r} committed an invalid optimizer step {step!r} at {boundary!r}"
                )
            steps[role] = step
        return steps

    def _read_success(self, boundary: str) -> dict[str, Any]:
        import json

        marker = os.path.join(boundary, _MULTI_ROLE_SUCCESS)
        try:
            with open(marker, encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"incomplete multi-role checkpoint boundary {boundary!r}") from error
        if not isinstance(payload, dict):
            raise RuntimeError(f"invalid multi-role checkpoint marker {marker!r}")
        expected_roles = sorted(self.roles)
        if payload.get("version") != 1 or payload.get("roles") != expected_roles:
            raise RuntimeError(f"multi-role checkpoint marker disagrees with configured roles at {boundary!r}")
        if payload.get("state_role") != self.state_role:
            raise RuntimeError(f"multi-role checkpoint marker disagrees with state role at {boundary!r}")
        model_step = payload.get("model_step")
        if type(model_step) is not int or model_step < 0:
            raise RuntimeError(f"multi-role checkpoint marker has an invalid model step at {boundary!r}")
        role_steps = payload.get("role_steps")
        # Absent only in boundaries written before per-role steps were recorded; those are validated
        # by the cross-role agreement rule in ``restore`` exactly as they always were.
        if role_steps is not None and (
            not isinstance(role_steps, dict)
            or sorted(role_steps) != expected_roles
            or any(type(step) is not int or step < 0 for step in role_steps.values())
        ):
            raise RuntimeError(f"multi-role checkpoint marker has invalid role steps at {boundary!r}")
        loop_state = payload.get("loop_state")
        # Absent in boundaries written before the loop-state slot existed, which recorded nothing --
        # indistinguishable, deliberately, from a caller that passes no loop state today. A reader
        # keys off ITS OWN entry being missing and decides what that means for itself.
        if loop_state is not None:
            try:
                _validated_loop_state(loop_state)
            except ValueError as error:
                raise RuntimeError(f"multi-role checkpoint marker has an invalid loop state at {boundary!r}") from error
        self._require_resumable_roles(boundary)
        return payload

    def _require_resumable_roles(self, boundary: str) -> None:
        """Require every configured role to have left a resumable save under this boundary."""
        from cosmos_framework.posttrain.checkpoint.io import component_kind, is_complete_iter_dir

        for role in self.roles:
            role_path = self.role_path(boundary, role)
            if not is_complete_iter_dir(role_path) or component_kind(role_path, "optim") is None:
                raise RuntimeError(f"multi-role checkpoint is missing resumable role {role!r} at {boundary!r}")

    def resolve_resume(self, resume: object) -> str | None:
        """Resolve only complete top-level boundaries; partial role saves remain invisible."""
        value = str(resume)
        if value in ("", "off"):
            return None
        if value == "auto":
            from cosmos_framework.posttrain.checkpoint.io import LATEST_FILE

            pointer = os.path.join(self.checkpoint_root, LATEST_FILE)
            candidate: str | None = None
            if os.path.isfile(pointer):
                with open(pointer, encoding="utf-8") as handle:
                    candidate = os.path.join(self.checkpoint_root, handle.read().strip())
            elif os.path.isdir(self.checkpoint_root):
                complete = sorted(
                    name
                    for name in os.listdir(self.checkpoint_root)
                    if name.startswith("iter_")
                    and os.path.isfile(os.path.join(self.checkpoint_root, name, _MULTI_ROLE_SUCCESS))
                )
                candidate = os.path.join(self.checkpoint_root, complete[-1]) if complete else None
            if candidate is None:
                return None
        else:
            candidate = os.path.realpath(os.path.expanduser(value))
        payload = self._read_success(candidate)
        self._latest_saved_step = max(self._latest_saved_step, int(payload["model_step"]))
        self._committed_loop_state = _marker_loop_state(payload)
        return candidate

    def _data_state(self) -> dict[str, Any] | None:
        if self.data is None:
            return None
        from cosmos_framework.posttrain.utils.entry import replica_lead_values

        states = replica_lead_values(self.data.state_dict())
        if not states or any(not isinstance(state, dict) for state in states):
            raise RuntimeError("checkpoint data role must return one state dictionary per replica")
        return {"version": 1, "replica_count": len(states), "states": {i: state for i, state in enumerate(states)}}

    def restore_data(self, boundary: str) -> None:
        """Restore each data replica from the state committed with ``boundary``."""
        if self.data is None:
            return
        component = self.manager.load_component(self.role_path(boundary, self.state_role), "data")
        if not isinstance(component, dict) or component.get("version") != 1:
            raise RuntimeError("multi-role checkpoint is missing its versioned data component")
        replica_count = component.get("replica_count")
        states = component.get("states")
        if type(replica_count) is not int or replica_count < 1 or not isinstance(states, dict):
            raise RuntimeError("multi-role checkpoint has an invalid data component")
        if set(states) != set(range(replica_count)) or any(not isinstance(state, dict) for state in states.values()):
            raise RuntimeError("multi-role checkpoint has invalid per-replica data states")
        from cosmos_framework.posttrain.utils.entry import resolve_all
        from cosmos_framework.posttrain.workers.worker_group import WorkerGroup, scatter

        ordered = [states[index] for index in range(replica_count)]
        # ``Scatter`` is a DISPATCH marker, so only a WorkerGroup edge unwraps it; a plain in-process
        # role body would be handed the marker object itself and reject it as a malformed state.
        # `_data_state` already accepts a plain body -- `replica_lead_values` collapses a concrete
        # value to exactly one replica -- so the restore side has to accept the same topology back.
        if not isinstance(self.data, WorkerGroup):
            if replica_count != 1:
                raise RuntimeError(
                    f"multi-role checkpoint holds {replica_count} data replicas, but the data role is "
                    "one in-process body"
                )
            resolve_all(self.data.load_state_dict(ordered[0]))
            return
        resolve_all(self.data.load_state_dict(scatter(ordered)))

    def restore(self, resume: object) -> int | None:
        """Restore every named role and the source from one committed boundary."""
        boundary = self.resolve_resume(resume)
        if boundary is None:
            return None
        committed = self._read_success(boundary).get("role_steps")
        steps: dict[str, int] = {}
        for role_name, role in self.roles.items():
            steps[role_name] = int(
                resolve_identical(
                    role.load(self.role_path(boundary, role_name)),
                    operation=f"{role_name} checkpoint restore",
                )
            )
        if committed is None:
            # Pre-``role_steps`` boundary: agreement across roles is the only invariant it recorded.
            if len(set(steps.values())) != 1:
                raise RuntimeError(f"multi-role checkpoint optimizer steps disagree: {steps!r}")
        elif steps != committed:
            raise RuntimeError(
                f"multi-role checkpoint roles restored optimizer steps {steps!r}, "
                f"but the boundary committed {committed!r}"
            )
        self.restore_data(boundary)
        return steps[self.state_role]

    def save(self, model_step: int, loop_state: Mapping[str, Any] | None = None) -> None:
        """Save every role through its shared manager, then commit one top-level boundary.

        ``loop_state`` is the caller's opaque run-level slot, committed with this boundary and
        handed back verbatim by ``committed_loop_state`` after a later resume.

        Commit-last, exactly like a single-role save: the roles write straight into the boundary
        and ``_SUCCESS`` is the commit point, published only once every role is resumable. Until
        that marker exists the boundary is invisible -- `resolve_resume` selects only marked
        directories -- and once it exists the boundary is immutable. There is deliberately no
        staging directory promoted by a top-level rename: that required
        ``renameat2(RENAME_NOREPLACE)``, which shared cluster filesystems (Lustre among them) do
        not implement, so it would have made a multi-role boundary unsavable exactly where
        multinode runs must store one.
        """
        from cosmos_framework.posttrain.checkpoint.io import update_latest

        state = _validated_loop_state(loop_state)
        target = self.path(model_step)
        marker = os.path.join(target, _MULTI_ROLE_SUCCESS)
        if os.path.isfile(marker):
            payload = self._read_success(target)
            if payload["model_step"] != model_step:
                raise RuntimeError(f"checkpoint boundary at {target!r} reports another model step")
            recorded = _marker_loop_state(payload)
            # A committed boundary is immutable, so adopting one may not silently re-label it: a
            # recorded slot that disagrees with what this run would commit means the two runs are
            # not the same run. Absence keeps its back-compatible meaning here too -- a boundary
            # that recorded nothing is unknown, not a disagreement -- so a boundary predating the
            # slot is adopted exactly as it was before, rather than failing every such retry.
            if recorded and recorded != state:
                raise RuntimeError(
                    f"checkpoint boundary at {target!r} recorded loop state {recorded!r}, "
                    f"but this run would commit {state!r}"
                )
            update_latest(target)
            self._latest_saved_step = max(self._latest_saved_step, model_step)
            self._committed_loop_state = state or recorded
            return
        # An unmarked directory is an uncommitted remnant of an interrupted attempt, never a
        # boundary any reader can have resumed from. Clearing it keeps a retry from inheriting a
        # half-written role. Only this path is an actual save, so only it is observed; adopting an
        # already-committed boundary above emits nothing.
        self._observer.on_checkpoint_save_start(model_step)
        try:
            self._write_boundary(target, marker, model_step, state)
        finally:
            self._observer.on_checkpoint_save_end(model_step)
        self._latest_saved_step = max(self._latest_saved_step, model_step)
        self._committed_loop_state = state
        info0(f"saved multi-role checkpoint {target} roles={sorted(self.roles)} resumable=True")
        self._observer.on_checkpoint_save_success(model_step, location=target)

    def _write_boundary(
        self,
        target: str,
        marker: str,
        model_step: int,
        state: dict[str, Any],
    ) -> None:
        """Write every role into ``target`` and commit the boundary marker; the observed save body."""
        import shutil

        from cosmos_framework.posttrain.checkpoint.io import atomic_write_json, fsync_directory, update_latest
        from cosmos_framework.posttrain.schema.checkpoint import FULL_CHECKPOINT_COMPONENTS
        from cosmos_framework.posttrain.utils.entry import resolve_all

        if os.path.lexists(target):
            shutil.rmtree(target)
        os.makedirs(target)
        try:
            data_state = self._data_state()
            for role_name, role in self.roles.items():
                kwargs: dict[str, Any] = {
                    "publish": False,
                    "update_latest": False,
                    "components": FULL_CHECKPOINT_COMPONENTS,
                }
                if role_name == self.state_role and data_state is not None:
                    kwargs["data_state"] = data_state
                resolve_all(role.save(self.role_path(target, role_name), **kwargs))
            # Verify BEFORE committing: a marker written over an incomplete boundary would publish
            # it.
            self._require_resumable_roles(target)
            role_steps = self._committed_role_steps(target)
        except BaseException:
            shutil.rmtree(target)
            raise

        # The marker write below is the commit point, so nothing from here on may enter the removal
        # path above: a failure in the pointer update or in retention must leave the committed
        # boundary intact rather than delete one a reader may already have resumed from.
        atomic_write_json(marker, self._success_payload(model_step, role_steps, state))
        fsync_directory(self.checkpoint_root)
        update_latest(target)
        from cosmos_framework.posttrain.checkpoint import apply_local_archive_retention

        apply_local_archive_retention(
            target,
            keep_optim_last=self.manager.keep_optim_last,
            keep_model_last=self.manager.keep_model_last,
            upload=self.manager.upload,
            async_archive_retention=False,
            optimizer_subdirs=tuple(sorted(self.roles)),
        )

    def save_configured_if_needed(self, model_step: int, loop_state: Mapping[str, Any] | None = None) -> bool:
        """Save one boundary, with the caller's loop state, unless this step was already committed."""
        if self._latest_saved_step >= model_step:
            return False
        self.save(model_step, loop_state)
        return True
