# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""L0 CPU tests for the backend-neutral checkpoint lifecycle facade."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from cosmos_framework.posttrain.checkpoint import CheckpointManager, LocalCheckpointStore, ObjectStoreCheckpointStore
from cosmos_framework.posttrain.config.rl import CheckpointStoreConfig, CheckpointUploadConfig
from cosmos_framework.posttrain.control.checkpointing import (
    CheckpointCoordinator,
    CheckpointLifecycle,
    MultiRoleCheckpointCoordinator,
)
from cosmos_framework.posttrain.schema.checkpoint import FULL_CHECKPOINT_COMPONENTS
from cosmos_framework.posttrain.workers.worker_group import WorkerGroup


def _config(rl: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(
        job=SimpleNamespace(project="project", group="group", name="name"),
        rl=rl,
    )


def _rl(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "run_ts": "20260901120000",
        "resume": "off",
        "checkpoint_format": "dcp",
        "save_async": True,
        "keep_optim_last": 0,
        "keep_model_last": 0,
        "checkpoint_store": CheckpointStoreConfig(),
        "upload": CheckpointUploadConfig(),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.L0
@pytest.mark.CPU
def test_lifecycle_builds_local_primary_and_archive_without_entrypoint_lowering(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("IMAGINAIRE_OUTPUT_ROOT", str(tmp_path))
    rl = _rl(
        upload=CheckpointUploadConfig(
            enabled=True,
            bucket="archive",
            credentials="credentials/<credential>.secret",
        )
    )

    lifecycle = CheckpointLifecycle.from_config(_config(rl), rl)

    assert isinstance(lifecycle.manager.store, LocalCheckpointStore)
    assert lifecycle.manager.upload == {
        "bucket": "archive",
        "credentials": "credentials/<credential>.secret",
        "prefix": "project/group/name/20260901120000/checkpoints",
    }


@pytest.mark.L0
@pytest.mark.CPU
def test_lifecycle_builds_direct_primary_with_the_canonical_default_prefix(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("IMAGINAIRE_OUTPUT_ROOT", str(tmp_path))
    rl = _rl(
        checkpoint_store=CheckpointStoreConfig(
            backend="object_store",
            bucket="primary",
            credentials="credentials/<credential>.secret",
        )
    )

    lifecycle = CheckpointLifecycle.from_config(_config(rl), rl)

    store = lifecycle.manager.store
    assert isinstance(store, ObjectStoreCheckpointStore)
    assert store.root_uri == "s3://primary/project/group/name/20260901120000/checkpoints"
    assert lifecycle.manager.upload is None


@pytest.mark.L0
@pytest.mark.CPU
def test_lifecycle_delegates_cross_option_policy_to_the_manager() -> None:
    rl = _rl(
        checkpoint_format="consolidated",
        checkpoint_store=CheckpointStoreConfig(backend="object_store"),
    )

    with pytest.raises(ValueError, match="requires checkpoint_format='dcp'"):
        CheckpointLifecycle.from_config(_config(rl), rl)


@pytest.mark.L0
@pytest.mark.CPU
def test_lifecycle_does_not_coerce_invalid_store_fields() -> None:
    rl = _rl(
        checkpoint_store=SimpleNamespace(
            backend="object_store",
            bucket=1,
            prefix="run/checkpoints",
            credentials="credentials/<credential>.secret",
        )
    )

    with pytest.raises(TypeError, match="bucket must be a string"):
        CheckpointLifecycle.from_config(_config(rl), rl)


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    "resume",
    [
        pytest.param("off", id="off"),
        pytest.param("auto", id="auto"),
        pytest.param("", id="empty"),
        pytest.param("/runs/name/checkpoints/iter_000000002", id="local-path"),
        pytest.param("s3://primary/run/checkpoints/iter_000000002/", id="object-store-uri"),
    ],
)
def test_worker_resume_checkpoint_is_the_checkpoint_the_trainer_restores(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    resume: str,
) -> None:
    """A role sharing the trainer's GPUs frees them for this restore, so both reads must agree."""
    monkeypatch.setenv("IMAGINAIRE_OUTPUT_ROOT", str(tmp_path))
    rl = _rl(resume=resume)

    selected = CheckpointLifecycle.worker_resume_checkpoint(rl)

    assert selected == CheckpointLifecycle.from_config(_config(rl), rl).resume_checkpoint
    assert (selected is None) == (resume in ("off", "auto", ""))


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("s3://primary/run/checkpoints/iter_000007/", None),
        ("s3://primary/other-run/checkpoints/iter_000007", "cross-prefix object-store resume"),
        ("/other-run/checkpoints/iter_000007", "requires an explicit s3:// source"),
    ],
)
def test_pending_validation_mutates_only_the_configured_direct_root(source: str, message: str | None) -> None:
    manager = CheckpointManager(
        checkpoint_format="dcp",
        checkpoint_store=ObjectStoreCheckpointStore(
            bucket="primary",
            prefix="run/checkpoints",
            credentials="unused-remote-credentials.json",
        ),
    )
    coordinator = CheckpointCoordinator(
        out_dir="/local/run",
        policy=SimpleNamespace(),
        manager=manager,
        resume_checkpoint=source,
    )

    if message is None:
        coordinator.copy_external_pending(7)
    else:
        with pytest.raises(RuntimeError, match=message):
            coordinator.copy_external_pending(7)


@pytest.mark.L0
@pytest.mark.CPU
def test_external_pending_copy_takes_the_exclusive_trainer_memory_window(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The staging write restores trainer state like any save: the colocated engine sleeps first and
    gets the card back (at sleep level 2, through a full republish) only after the copy is promoted."""
    from cosmos_framework.posttrain import checkpoint

    events: list[str] = []
    source = tmp_path / "old" / "checkpoints" / "iter_000007"
    staging = tmp_path / "new" / "checkpoints" / ".iter_000007.staging"
    monkeypatch.setattr(checkpoint, "checkpoint_provenance", lambda _target: None)
    monkeypatch.setattr(
        checkpoint,
        "prepare_external_checkpoint_copy",
        lambda _source, _target, *, expected_provenance: SimpleNamespace(staging_dir=str(staging)),
    )
    monkeypatch.setattr(checkpoint, "promote_external_checkpoint_copy", lambda _plan: events.append("promote"))

    class _Policy:
        @staticmethod
        def loaded_checkpoint_provenance() -> dict[str, str]:
            return {"source": str(source)}

        @staticmethod
        def save(_checkpoint: str, **_kwargs: object) -> None:
            events.append("save")

    coordinator = CheckpointCoordinator(
        out_dir=str(tmp_path / "new"),
        policy=_Policy(),
        manager=CheckpointManager(),
        resume_checkpoint=str(source),
        before_save=lambda step: events.append(f"before:{step}"),
        after_save=lambda step: events.append(f"after:{step}"),
    )

    coordinator.copy_external_pending(7)

    assert events == ["before:7", "save", "promote", "after:7"]


@pytest.mark.L0
@pytest.mark.CPU
def test_coordinator_unions_plans_and_upgrades_a_partial_save_for_recovery(tmp_path: Path) -> None:
    saved: list[tuple[tuple[str, ...] | None, bool]] = []

    class _Policy:
        @staticmethod
        def save(
            _checkpoint: str,
            *,
            components: tuple[str, ...] | None = None,
            data_state: dict[str, int] | None = None,
            **_kwargs: object,
        ) -> None:
            saved.append((components, data_state is not None))

    class _Data:
        @staticmethod
        def state_dict() -> dict[str, int]:
            return {"cursor": 1}

    with pytest.warns(UserWarning, match="partial checkpoint artifact"):
        manager = CheckpointManager(checkpoint_components=["model"])
    coordinator = CheckpointCoordinator(
        out_dir=str(tmp_path),
        policy=_Policy(),
        manager=manager,
        data=_Data(),
    )

    union = coordinator.resolve_component_plan(include_configured=True, additional=("optimizer", "training_state"))
    assert union is not None and union.is_full
    assert coordinator.last_save_location is None
    assert coordinator.save_configured_if_needed(1)
    assert coordinator.last_save_location == str(tmp_path / "checkpoint_artifacts" / "iter_000001")
    assert not coordinator.save_configured_if_needed(1)
    assert coordinator.ensure_resumable(1)
    assert coordinator.last_save_location == coordinator.path(1)
    assert not coordinator.ensure_resumable(1)
    assert saved == [(None, False), (FULL_CHECKPOINT_COMPONENTS, True)]


@pytest.mark.L0
@pytest.mark.CPU
def test_checkpoint_save_hooks_wrap_only_actual_successful_writes(tmp_path: Path) -> None:
    events: list[tuple[str, int]] = []

    class _Policy:
        @staticmethod
        def save(_checkpoint: str, **_kwargs: object) -> None:
            events.append(("save", 1))

    coordinator = CheckpointCoordinator(
        out_dir=str(tmp_path),
        policy=_Policy(),
        manager=CheckpointManager(),
        before_save=lambda step: events.append(("before", step)),
        after_save=lambda step: events.append(("after", step)),
    )

    assert coordinator.save_configured_if_needed(1)
    assert not coordinator.save_configured_if_needed(1)
    assert events == [("before", 1), ("save", 1), ("after", 1)]


@pytest.mark.L0
@pytest.mark.CPU
def test_checkpoint_save_in_a_held_window_skips_both_hooks(tmp_path: Path) -> None:
    """A caller that already holds the trainer-memory window keeps it: no acquire, no hand-back."""
    events: list[tuple[str, int]] = []

    class _Policy:
        @staticmethod
        def save(_checkpoint: str, **_kwargs: object) -> None:
            events.append(("save", 1))

    coordinator = CheckpointCoordinator(
        out_dir=str(tmp_path),
        policy=_Policy(),
        manager=CheckpointManager(),
        before_save=lambda step: events.append(("before", step)),
        after_save=lambda step: events.append(("after", step)),
    )

    assert coordinator.save_configured_if_needed(1, window_held=True)
    assert not coordinator.save_configured_if_needed(1)
    assert events == [("save", 1)]


@pytest.mark.L0
@pytest.mark.CPU
def test_checkpoint_save_failure_leaves_the_exclusive_window_acquired(tmp_path: Path) -> None:
    events: list[tuple[str, int]] = []

    class _Policy:
        @staticmethod
        def save(_checkpoint: str, **_kwargs: object) -> None:
            events.append(("save", 1))
            raise RuntimeError("injected save failure")

    coordinator = CheckpointCoordinator(
        out_dir=str(tmp_path),
        policy=_Policy(),
        manager=CheckpointManager(),
        before_save=lambda step: events.append(("before", step)),
        after_save=lambda step: events.append(("after", step)),
    )

    with pytest.raises(RuntimeError, match="injected save failure"):
        coordinator.save_configured_if_needed(1)
    assert events == [("before", 1), ("save", 1)]


@pytest.mark.L0
@pytest.mark.CPU
def test_coordinator_emergency_save_writes_the_run_plan_only_when_asked(tmp_path: Path) -> None:
    """`save_emergency` under "run" IS the cadence save (the configured plan, the artifact location, the
    same dedup of an already-saved step); under "full" (the default) it is `ensure_resumable`, which still
    upgrades a partial save of the same step; any other selector is refused before a writer is entered."""
    saved: list[tuple[tuple[str, ...] | None, bool]] = []

    class _Policy:
        @staticmethod
        def save(
            _checkpoint: str,
            *,
            components: tuple[str, ...] | None = None,
            data_state: dict[str, int] | None = None,
            **_kwargs: object,
        ) -> None:
            saved.append((components, data_state is not None))

    class _Data:
        @staticmethod
        def state_dict() -> dict[str, int]:
            return {"cursor": 1}

    with pytest.warns(UserWarning, match="partial checkpoint artifact"):
        manager = CheckpointManager(checkpoint_components=["model"])
    coordinator = CheckpointCoordinator(out_dir=str(tmp_path), policy=_Policy(), manager=manager, data=_Data())

    # "run": the model-only plan the run configured, through the cadence call shape (no override, no cursor).
    assert coordinator.save_emergency(1, components="run")
    assert coordinator.last_save_location == str(tmp_path / "checkpoint_artifacts" / "iter_000001")
    assert saved == [(None, False)]
    assert not coordinator.save_emergency(1, components="run")  # step 1 is saved: no second write
    assert not coordinator.save_configured_if_needed(1)  # and the cadence agrees
    # "full" after a partial save of the same step still upgrades it to the exact-resume plan.
    assert coordinator.save_emergency(1, components="full")
    assert coordinator.last_save_location == coordinator.path(1)
    assert saved[-1] == (FULL_CHECKPOINT_COMPONENTS, True)
    assert not coordinator.save_emergency(1)  # default = "full": already resumable
    # Default at a new step: the full plan, whatever the configured plan is.
    assert coordinator.save_emergency(2)
    assert saved[-1] == (FULL_CHECKPOINT_COMPONENTS, True)
    assert not coordinator.save_emergency(2, components="run")  # a full save of step 2 already exists
    with pytest.raises(ValueError, match="emergency checkpoint components must be one of"):
        coordinator.save_emergency(3, components="model")
    assert len(saved) == 3


@pytest.mark.L0
@pytest.mark.CPU
def test_coordinator_emergency_run_plan_always_covers_the_model(tmp_path: Path) -> None:
    """With `emergency_checkpoint_components=run`, run components `[model]` and
    validation components `[training_state]`, a signal during OFF-cadence validation left only the training-state
    artifact -- the step-only dedup of the cadence path answered "already checkpointed" and NO weights were written.
    The emergency plan under "run" is the run's plan UNION {model}, and the skip is decided by the component coverage
    of THIS step, never by the step number alone."""
    saved: list[tuple[tuple[str, ...] | None, bool]] = []

    class _Policy:
        @staticmethod
        def save(
            _checkpoint: str,
            *,
            components: tuple[str, ...] | None = None,
            data_state: dict[str, int] | None = None,
            **_kwargs: object,
        ) -> None:
            saved.append((components, data_state is not None))

    class _Data:
        @staticmethod
        def state_dict() -> dict[str, int]:
            return {"cursor": 1}

    with pytest.warns(UserWarning, match="partial checkpoint artifact"):
        manager = CheckpointManager(checkpoint_components=["model"])
    coordinator = CheckpointCoordinator(out_dir=str(tmp_path), policy=_Policy(), manager=manager, data=_Data())

    # The off-cadence validation committed its training-state artifact for step 1 (`run_partial_checkpoint`'s call).
    with pytest.warns(UserWarning, match="partial checkpoint artifact"):
        coordinator.save(1, components=("training_state",))
    assert saved == [(("training_state",), True)]
    assert not coordinator.save_configured_if_needed(1)  # the cadence path itself still dedups by step
    # Before the fix: False ("already checkpointed"), no model. The committed training-state artifact of step 1 can never be
    # re-committed (`store.commit_artifact` refuses a second manifest), so the one writable location left for the model
    # is the full checkpoint of the step: the emergency save escalates to the full plan under `checkpoints/iter_000001`.
    assert coordinator.save_emergency(1, components="run")
    assert saved[-1] == (FULL_CHECKPOINT_COMPONENTS, True)
    assert coordinator.last_save_location == coordinator.path(1)
    assert not coordinator.save_emergency(1, components="run")  # every component of the plan is on disk now
    assert not coordinator.save_emergency(1)  # and so is the full plan
    assert len(saved) == 2

    # A run plan WITHOUT the model still writes the model at a termination trigger: nothing of step 3 was saved, so the
    # union plan `[model, training_state]` lands as ONE artifact (explicit override, cursor included).
    saved.clear()
    with pytest.warns(UserWarning, match="partial checkpoint artifact"):
        state_manager = CheckpointManager(checkpoint_components=["training_state"])
    state_coordinator = CheckpointCoordinator(
        out_dir=str(tmp_path / "state_only"), policy=_Policy(), manager=state_manager, data=_Data()
    )
    with pytest.warns(UserWarning, match="partial checkpoint artifact"):
        assert state_coordinator.save_emergency(3, components="run")
    assert saved == [(("model", "training_state"), True)]
    assert state_coordinator.last_save_location == str(tmp_path / "state_only" / "checkpoint_artifacts" / "iter_000003")
    assert not state_coordinator.save_emergency(3, components="run")

    # The run's own model-only cadence save of a step still dedups the emergency save of that step (the cadence call
    # shape, no override): unchanged from before the fix.
    saved.clear()
    model_coordinator = CheckpointCoordinator(
        out_dir=str(tmp_path / "model_only"), policy=_Policy(), manager=manager, data=_Data()
    )
    assert model_coordinator.save_configured_if_needed(2)
    assert saved == [(None, False)]
    assert not model_coordinator.save_emergency(2, components="run")
    assert model_coordinator.save_emergency(
        4, components="run"
    )  # a new step: the run plan (model) through the cadence shape
    assert saved[-1] == (None, False)
    assert len(saved) == 2


class _CheckpointRole:
    def __init__(self, *, fail: bool = False, optimizer_step: int = 0) -> None:
        self.fail = fail
        self.optimizer_step = optimizer_step

    def save(
        self,
        checkpoint: str,
        *,
        publish: bool,
        update_latest: bool,
        components: tuple[str, ...],
        data_state: dict[str, object] | None = None,
    ) -> None:
        import os

        import torch

        assert not publish and not update_latest
        assert components == FULL_CHECKPOINT_COMPONENTS
        if self.fail:
            raise RuntimeError("injected role save failure")
        os.makedirs(checkpoint)
        for name in ("model", "optim"):
            torch.save({"name": name}, os.path.join(checkpoint, f"{name}.pt"))
        # Every real trainer's `trainer.pt` carries its durable optimizer clock -- resume refuses a
        # trainer state without one -- and the boundary marker records what each role committed.
        torch.save(
            {"name": "trainer", "iter": self.optimizer_step, "optimizer_step": self.optimizer_step},
            os.path.join(checkpoint, "trainer.pt"),
        )
        if data_state is not None:
            torch.save(data_state, os.path.join(checkpoint, "data.pt"))

    def load(self, checkpoint: str) -> int:
        del checkpoint
        return self.optimizer_step


class _CheckpointData:
    def __init__(self) -> None:
        self.cursor = 3

    def state_dict(self) -> dict[str, int]:
        return {"cursor": self.cursor}

    def load_state_dict(self, state: dict[str, int]) -> bool:
        self.cursor = state["cursor"]
        return True


def _multi_role_coordinator(
    tmp_path: Path,
    *,
    fail_critic: bool = False,
    policy_step: int = 0,
    critic_step: int = 0,
) -> tuple[MultiRoleCheckpointCoordinator, _CheckpointData]:
    data_body = _CheckpointData()
    roles = {
        "policy": WorkerGroup("policy", "trainer", _CheckpointRole(optimizer_step=policy_step)),
        "critic": WorkerGroup("critic", "critic", _CheckpointRole(fail=fail_critic, optimizer_step=critic_step)),
    }
    data = WorkerGroup("data", "env", data_body)
    coordinator = MultiRoleCheckpointCoordinator(
        checkpoint_root=str(tmp_path / "checkpoints"),
        roles=roles,
        manager=CheckpointManager(checkpoint_format="consolidated", keep_optim_last=1),
        data=data,
    )
    return coordinator, data_body


@pytest.mark.L0
@pytest.mark.CPU
def test_multi_role_checkpoint_publishes_policy_critic_and_source_as_one_boundary(tmp_path: Path) -> None:
    coordinator, data = _multi_role_coordinator(tmp_path)

    coordinator.save(4)
    data.cursor = 99
    boundary = coordinator.resolve_resume("auto")
    assert boundary is not None
    coordinator.restore_data(boundary)

    assert data.cursor == 3
    assert Path(boundary, "_SUCCESS").is_file()
    assert Path(boundary, "policy", "data.pt").is_file()
    assert Path(tmp_path, "checkpoints", "latest_checkpoint.txt").read_text().strip() == "iter_000004"


@pytest.mark.L0
@pytest.mark.CPU
def test_multi_role_checkpoint_failure_publishes_neither_boundary_nor_latest(tmp_path: Path) -> None:
    coordinator, _data = _multi_role_coordinator(tmp_path, fail_critic=True)

    with pytest.raises(RuntimeError, match="injected role save failure"):
        coordinator.save(4)

    root = tmp_path / "checkpoints"
    assert not (root / "iter_000004").exists()
    assert not (root / "latest_checkpoint.txt").exists()
    assert list(root.glob("*.staging")) == []


@pytest.mark.L0
@pytest.mark.CPU
def test_multi_role_boundary_records_and_restores_unequal_role_optimizer_steps(tmp_path: Path) -> None:
    """Roles in one boundary need not share a counter: PPO's critic-only warmup holds the actor
    still while the critic trains, so each role is checked against its OWN committed step."""
    coordinator, _data = _multi_role_coordinator(tmp_path, policy_step=0, critic_step=3)

    coordinator.save(3)

    marker = tmp_path / "checkpoints" / "iter_000003" / "_SUCCESS"
    assert json.loads(marker.read_text(encoding="utf-8"))["role_steps"] == {"critic": 3, "policy": 0}
    assert coordinator.restore("auto") == 0
    assert coordinator.committed_step() == 3


@pytest.mark.L0
@pytest.mark.CPU
def test_multi_role_restore_rejects_a_role_returning_another_step(tmp_path: Path) -> None:
    """Recording the committed steps rejects a drifted role even when the roles still agree."""
    coordinator, _data = _multi_role_coordinator(tmp_path, policy_step=2, critic_step=2)
    coordinator.save(2)

    coordinator.roles["policy"].body.optimizer_step = 5
    coordinator.roles["critic"].body.optimizer_step = 5

    with pytest.raises(RuntimeError, match="but the boundary committed"):
        coordinator.restore("auto")


@pytest.mark.L0
@pytest.mark.CPU
def test_multi_role_restore_keeps_the_agreement_rule_for_a_legacy_boundary(tmp_path: Path) -> None:
    """A boundary written before per-role steps existed keeps exactly its old validation."""
    coordinator, _data = _multi_role_coordinator(tmp_path, policy_step=1, critic_step=1)
    coordinator.save(1)
    marker = tmp_path / "checkpoints" / "iter_000001" / "_SUCCESS"
    payload = json.loads(marker.read_text(encoding="utf-8"))
    payload.pop("role_steps")
    marker.write_text(json.dumps(payload), encoding="utf-8")

    assert coordinator.restore("auto") == 1

    coordinator.roles["critic"].body.optimizer_step = 7
    with pytest.raises(RuntimeError, match="optimizer steps disagree"):
        coordinator.restore("auto")


@pytest.mark.L0
@pytest.mark.CPU
def test_multi_role_boundary_records_and_returns_an_opaque_loop_state(tmp_path: Path) -> None:
    """One run-level slot the CALLER owns: recorded verbatim, read back verbatim, never inspected.

    A loop whose configured choice is invisible in every role counter needs the boundary to carry
    that choice, but the coordinator is shared, so it keeps no schema for it.
    """
    coordinator, _data = _multi_role_coordinator(tmp_path, policy_step=0, critic_step=2)
    state = {"frozen_prefix_layers": 4, "sampler": "exact", "beta": 0.5}

    coordinator.save(2, state)

    payload = json.loads((tmp_path / "checkpoints" / "iter_000002" / "_SUCCESS").read_text(encoding="utf-8"))
    assert payload["loop_state"] == state
    # An additive optional field, so the marker version is unchanged: a reader that predates the
    # slot still accepts this boundary instead of rejecting every one written after the change.
    assert payload["version"] == 1

    reopened, _reopened_data = _multi_role_coordinator(tmp_path, policy_step=0, critic_step=2)
    assert reopened.committed_loop_state() == {}
    assert reopened.restore("auto") == 0
    assert reopened.committed_loop_state() == state


@pytest.mark.L0
@pytest.mark.CPU
def test_multi_role_boundary_reports_no_loop_state_for_a_legacy_marker(tmp_path: Path) -> None:
    """A boundary written before the slot existed restores and simply reports nothing recorded."""
    coordinator, _data = _multi_role_coordinator(tmp_path, policy_step=1, critic_step=1)
    coordinator.save(1, {"frozen_prefix_layers": 4})
    marker = tmp_path / "checkpoints" / "iter_000001" / "_SUCCESS"
    payload = json.loads(marker.read_text(encoding="utf-8"))
    payload.pop("loop_state")
    marker.write_text(json.dumps(payload), encoding="utf-8")

    reopened, _reopened_data = _multi_role_coordinator(tmp_path, policy_step=1, critic_step=1)

    assert reopened.restore("auto") == 1
    assert reopened.committed_loop_state() == {}


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    "state",
    [{"schedule": [1, 2]}, {"nested": {"a": 1}}, {4: "int-key"}, {"": 1}],
    ids=["list-value", "mapping-value", "non-string-key", "empty-key"],
)
def test_multi_role_save_rejects_a_loop_state_that_is_not_named_json_scalars(
    tmp_path: Path,
    state: dict[object, object],
) -> None:
    """Rejected BEFORE any role writes, so an unrepresentable slot cannot reach a committed marker."""
    coordinator, _data = _multi_role_coordinator(tmp_path)

    with pytest.raises(ValueError, match="loop_state"):
        coordinator.save(1, state)

    assert not (tmp_path / "checkpoints" / "iter_000001").exists()


@pytest.mark.L0
@pytest.mark.CPU
def test_multi_role_save_refuses_to_relabel_a_committed_boundary(tmp_path: Path) -> None:
    """A committed boundary is immutable, so adopting one may not silently re-label its loop state.

    Absence keeps its back-compatible meaning on this path too: a boundary that recorded nothing is
    unknown, not a disagreement, so it is adopted exactly as it was before the slot existed.
    """
    coordinator, _data = _multi_role_coordinator(tmp_path)
    coordinator.save(1, {"frozen_prefix_layers": 4})

    reopened, _reopened_data = _multi_role_coordinator(tmp_path)
    with pytest.raises(RuntimeError, match="recorded loop state"):
        reopened.save(1, {"frozen_prefix_layers": 9})

    marker = tmp_path / "checkpoints" / "iter_000001" / "_SUCCESS"
    payload = json.loads(marker.read_text(encoding="utf-8"))
    payload.pop("loop_state")
    marker.write_text(json.dumps(payload), encoding="utf-8")

    legacy, _legacy_data = _multi_role_coordinator(tmp_path)
    legacy.save(1, {"frozen_prefix_layers": 9})
    assert legacy.committed_step() == 1


@pytest.mark.L0
@pytest.mark.CPU
def test_multi_role_checkpoint_reuses_named_optimizer_retention(tmp_path: Path) -> None:
    coordinator, _data = _multi_role_coordinator(tmp_path)

    coordinator.save(1)
    coordinator.save(2)

    root = tmp_path / "checkpoints"
    for role in ("policy", "critic"):
        assert not (root / "iter_000001" / role / "optim.pt").exists()
        assert (root / "iter_000001" / role / "model.pt").is_file()
        assert (root / "iter_000002" / role / "optim.pt").is_file()
