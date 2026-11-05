# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""@L0 @CPU: durable GRPO validation state in the LLM trainer checkpoint."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from cosmos_framework.posttrain.algorithm.primitives.objective import Objective, Reduction
from cosmos_framework.posttrain.callbacks.base import Callback
from cosmos_framework.posttrain.control.validation import ValidationTrigger
from cosmos_framework.posttrain.roles.trainer.llm import LLMTrainer

_EMPTY_PROVIDER_IDENTITY_HASH = hashlib.sha256(b"{}").hexdigest()


def _objective(model: Any, _batch: list[Any]) -> tuple[torch.Tensor, dict[str, float]]:  # returns loss: []
    loss = next(model.parameters()).sum() * 0.0  # []
    return loss, {"grpo_loss": 0.0}


def _identity(source: str = "held-out-v1") -> dict[str, Any]:
    return {
        "enabled": True,
        "source_fingerprint": source,
        "rollout_semantics": {
            "version": 1,
            "backend": "fake",
            "tokenizer_sha256": "tokenizer-v1",
        },
        "sampling": {
            "temperature": 0.0,
            "top_p": None,
            "top_k": 1,
            "repetition_penalty": 1.0,
            "n_generation": 1,
            "max_tokens": 32,
        },
        "reward": "exact_match",
        "reward_provider": {
            "name": "exact_match",
            "revision": "builtin-v1",
            "catalog": "builtin",
            "identity_hash": _EMPTY_PROVIDER_IDENTITY_HASH,
        },
        "seed": 7,
        "batch_size": 2,
        "max_samples": 8,
        "schedule": {
            "before_training": True,
            "every_steps": 2,
            "train_total_steps": 10,
        },
    }


def _trainer(
    *,
    callbacks: list[Callback] | None = None,
    upload: dict[str, str] | None = None,
) -> LLMTrainer:
    model = torch.nn.Linear(2, 2)
    objective = Objective(_objective, reduction=Reduction.MEAN)
    return LLMTrainer(
        model,
        objective,
        torch.optim.SGD(model.parameters(), lr=0.1),
        callbacks=callbacks,
        upload=upload,
    )


@pytest.mark.L0
@pytest.mark.CPU
def test_llm_validation_pending_state_round_trips_and_checks_identity() -> None:
    trainer = _trainer()
    trainer.configure_validation(_identity())
    trainer.train([])
    trainer.train([])
    trainer.mark_validation_pending(
        2,
        frozenset({ValidationTrigger.PERIODIC}),
        {"train/loss_avg": 0.25},
    )
    state = trainer._trainer_state()
    assert state["validation"]["version"] == 2
    assert state["validation"]["publication"] is None

    restored = _trainer()
    assert restored._restore_trainer_state(state) == 2
    restored.configure_validation(_identity())
    assert restored.pending_validation_step() == 2
    assert restored.pending_validation_reasons() == frozenset({ValidationTrigger.PERIODIC})
    assert restored.pending_validation_metrics() == {"train/loss_avg": 0.25}

    incompatible_identities = []
    for field, value in (
        ("source_fingerprint", "different-source"),
        ("reward", "different-reward"),
        ("seed", 8),
        ("batch_size", 3),
        ("max_samples", 9),
    ):
        identity = _identity()
        identity[field] = value
        if field == "reward":
            identity["reward_provider"] = {**identity["reward_provider"], "name": value}
        incompatible_identities.append(identity)
    sampling_identity = _identity()
    sampling_identity["sampling"] = {**sampling_identity["sampling"], "temperature": 0.5}
    incompatible_identities.append(sampling_identity)
    for field, value in (("every_steps", 3), ("train_total_steps", 11)):
        schedule_identity = _identity()
        schedule_identity["schedule"] = {**schedule_identity["schedule"], field: value}
        incompatible_identities.append(schedule_identity)
    rollout_identity = _identity()
    rollout_identity["rollout_semantics"] = {
        **rollout_identity["rollout_semantics"],
        "tokenizer_sha256": "tokenizer-v2",
    }
    incompatible_identities.append(rollout_identity)
    provider_identity = _identity()
    provider_identity["reward_provider"] = {
        **provider_identity["reward_provider"],
        "revision": "custom-v2",
    }
    incompatible_identities.append(provider_identity)
    provider_semantics = _identity()
    provider_semantics["reward_provider"] = {
        **provider_semantics["reward_provider"],
        "identity_hash": "1" * 64,
    }
    incompatible_identities.append(provider_semantics)

    for identity in incompatible_identities:
        incompatible = _trainer()
        incompatible._restore_trainer_state(state)
        with pytest.raises(RuntimeError, match="validation identity differs"):
            incompatible.configure_validation(identity)


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    "reward_provider",
    (
        None,
        {"name": "exact_match", "revision": "builtin-v1"},
        {"name": "exact_match", "revision": "builtin-v1", "catalog": "builtin", "extra": "drift"},
        {"name": "exact_match", "revision": "", "catalog": "builtin"},
        {"name": "gsm8k", "revision": "builtin-v1", "catalog": "builtin"},
    ),
)
def test_llm_validation_reward_provider_identity_is_exact(reward_provider: object) -> None:
    identity = _identity()
    identity["reward_provider"] = reward_provider
    with pytest.raises(ValueError, match="reward_provider"):
        _trainer().configure_validation(identity)


@pytest.mark.L0
@pytest.mark.CPU
def test_llm_validation_legacy_identity_migrates_builtin_provider() -> None:
    trainer = _trainer()
    trainer.configure_validation(_identity())
    legacy_state = trainer._trainer_state()
    legacy_state["validation"]["identity"].pop("reward_provider")

    restored = _trainer()
    restored._restore_trainer_state(legacy_state)
    restored.configure_validation(_identity())
    assert restored._trainer_state()["validation"]["identity"]["reward_provider"] == {
        "name": "exact_match",
        "revision": "builtin-v1",
        "catalog": "builtin",
        "identity_hash": _EMPTY_PROVIDER_IDENTITY_HASH,
    }


@pytest.mark.L0
@pytest.mark.CPU
def test_llm_validation_disabled_identity_is_durable_and_cannot_change_on_resume() -> None:
    trainer = _trainer()
    trainer.configure_validation({"enabled": False})
    disabled_state = trainer._trainer_state()
    assert disabled_state["validation"] == {
        "version": 2,
        "identity": {"enabled": False},
        "progress": {"pending_step": None, "pending_reasons": [], "pending_metrics": {}},
        "publication": None,
    }

    enabled_from_disabled = _trainer()
    enabled_from_disabled._restore_trainer_state(disabled_state)
    with pytest.raises(RuntimeError, match="validation identity differs"):
        enabled_from_disabled.configure_validation(_identity())

    enabled = _trainer()
    enabled.configure_validation(_identity())
    enabled.mark_validation_pending(0, frozenset({ValidationTrigger.BEFORE_TRAINING}), {})
    restored = _trainer()
    restored._restore_trainer_state(enabled._trainer_state())
    with pytest.raises(RuntimeError, match="validation identity differs"):
        restored.configure_validation({"enabled": False})

    # A versioned disabled identity is compatible with the same disabled run on later saves.
    restored_disabled = _trainer()
    restored_disabled._restore_trainer_state(disabled_state)
    restored_disabled.configure_validation({"enabled": False})
    assert restored_disabled._trainer_state()["validation"] == disabled_state["validation"]


@pytest.mark.L0
@pytest.mark.CPU
def test_llm_ephemeral_validation_completion_creates_no_journal() -> None:
    trainer = _trainer()
    trainer.configure_validation(_identity())

    trainer.mark_validation_pending(0, frozenset({ValidationTrigger.BEFORE_TRAINING}), {})
    trainer.complete_ephemeral_validation(0)

    assert trainer.pending_validation_step() is None
    assert trainer.pending_validation_publication_step() is None
    validation = trainer._trainer_state()["validation"]
    assert validation["progress"] == {"pending_step": None, "pending_reasons": [], "pending_metrics": {}}
    assert validation["publication"] is None


@pytest.mark.L0
@pytest.mark.CPU
def test_legacy_checkpoint_cannot_enable_grpo_validation_on_exact_resume() -> None:
    trainer = _trainer()
    trainer._restore_trainer_state({"iter": 0, "accum": 0})
    trainer._resumed_from_checkpoint = True

    with pytest.raises(RuntimeError, match="predates the validation identity journal"):
        trainer.configure_validation(_identity())


@pytest.mark.L0
@pytest.mark.CPU
def test_pending_checkpoint_is_cleared_locally_before_callback(tmp_path: Any) -> None:
    published: list[str] = []

    class _PublishSpy(Callback):
        def on_checkpoint(self, _trainer: Any, path: str) -> None:
            published.append(path)

    trainer = _trainer(callbacks=[_PublishSpy()])
    trainer.configure_validation(_identity())
    trainer.mark_validation_pending(0, frozenset({ValidationTrigger.BEFORE_TRAINING}), {})
    checkpoint = tmp_path / "checkpoints" / "iter_000000"
    trainer.save(str(checkpoint), publish=False)
    assert published == []

    pending_state = torch.load(checkpoint / "trainer.pt", map_location="cpu", weights_only=False)
    assert pending_state["validation"]["progress"]["pending_step"] == 0

    trainer.complete_validation(0)
    assert trainer.pending_validation_step() is None
    assert trainer.pending_validation_publication_step() == 0
    trainer.save_validation_progress(str(checkpoint))
    assert published == [str(checkpoint)]
    completed_state = torch.load(checkpoint / "trainer.pt", map_location="cpu", weights_only=False)
    assert completed_state["validation"]["progress"]["pending_step"] is None
    assert completed_state["validation"]["publication"] is None
    assert trainer.pending_validation_publication_step() is None


@pytest.mark.L0
@pytest.mark.CPU
def test_llm_validation_v1_state_restores_without_a_publication_journal() -> None:
    """The pre-publication LLM schema remains resumable as evaluation-pending state."""
    trainer = _trainer()
    trainer.configure_validation(_identity())
    trainer.mark_validation_pending(0, frozenset({ValidationTrigger.BEFORE_TRAINING}), {})
    legacy_state = copy.deepcopy(trainer._trainer_state())
    legacy_state["validation"]["version"] = 1
    legacy_state["validation"].pop("publication")

    restored = _trainer()
    assert restored._restore_trainer_state(legacy_state) == 0
    restored.configure_validation(_identity())

    assert restored.pending_validation_step() == 0
    assert restored.pending_validation_reasons() == frozenset({ValidationTrigger.BEFORE_TRAINING})
    assert restored.pending_validation_publication_step() is None


@pytest.mark.L0
@pytest.mark.CPU
def test_llm_validation_upload_failure_resumes_publication_without_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Strict archive failure persists v2 publication state and resume retries only that phase."""
    from cosmos_framework.posttrain.checkpoint import load_component
    from cosmos_framework.posttrain.checkpoint import manager as checkpoint_manager

    callbacks: list[str] = []

    class _PublishSpy(Callback):
        def on_checkpoint(self, _trainer: Any, path: str) -> None:
            callbacks.append(path)

    upload = {"bucket": "archive", "prefix": "posttrain/checkpoints"}
    trainer = _trainer(callbacks=[_PublishSpy()], upload=upload)
    trainer.configure_validation(_identity())
    trainer.mark_validation_pending(0, frozenset({ValidationTrigger.FINAL}), {})
    checkpoint = tmp_path / "checkpoints" / "iter_000000"
    trainer.save(str(checkpoint), publish=False)
    trainer.complete_validation(0)

    monkeypatch.setattr(
        "cosmos_framework.posttrain.utils.object_store.upload_destination_identity",
        lambda current_upload, iter_name: {
            "kind": "object_store",
            "endpoint_authority": "storage.example.test",
            "bucket": str(current_upload["bucket"]),
            "prefix": f"{str(current_upload['prefix']).rstrip('/')}/{iter_name}",
        },
    )

    def _fail_upload(
        iter_dir: str,
        current_upload: dict[str, str],
        *,
        overrides: dict[str, str] | None = None,
        commit: dict[str, Any] | None = None,
    ) -> None:
        del iter_dir, current_upload
        assert overrides is not None and commit is not None
        raise OSError("archive unavailable")

    monkeypatch.setattr(checkpoint_manager, "archive_checkpoint", _fail_upload)
    with pytest.raises(OSError, match="archive unavailable"):
        trainer.save_validation_progress(str(checkpoint))

    failed_validation = load_component(str(checkpoint), "trainer")["validation"]
    assert failed_validation["version"] == 2
    assert failed_validation["progress"]["pending_step"] is None
    assert failed_validation["publication"]["step"] == 0
    assert callbacks == []

    restored_callbacks: list[str] = []

    class _RestoredPublishSpy(Callback):
        def on_checkpoint(self, _trainer: Any, path: str) -> None:
            restored_callbacks.append(path)

    restored = _trainer(callbacks=[_RestoredPublishSpy()], upload=upload)
    assert restored.load(str(checkpoint)) == 0
    restored.configure_validation(_identity())
    assert restored.pending_validation_step() is None
    assert restored.pending_validation_publication_step() == 0

    archived_states: list[dict[str, Any]] = []

    def _complete_upload(
        iter_dir: str,
        current_upload: dict[str, str],
        *,
        overrides: dict[str, str] | None = None,
        commit: dict[str, Any] | None = None,
    ) -> None:
        del current_upload
        assert overrides is not None and commit is not None
        archived_states.append(torch.load(overrides["trainer.pt"], map_location="cpu", weights_only=False))
        (Path(iter_dir) / ".uploaded").write_text("archive\n", encoding="utf-8")

    monkeypatch.setattr(checkpoint_manager, "archive_checkpoint", _complete_upload)
    restored.save_validation_progress(str(checkpoint))

    assert archived_states[0]["validation"]["publication"] is None
    completed_validation = load_component(str(checkpoint), "trainer")["validation"]
    assert completed_validation["publication"] is None
    assert restored.pending_validation_publication_step() is None
    assert restored_callbacks == [str(checkpoint)]


@pytest.mark.L0
@pytest.mark.CPU
def test_llm_validation_publication_rejects_destination_drift_on_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pending publication stays bound to its original bucket/prefix across resume."""
    from cosmos_framework.posttrain.checkpoint import manager as checkpoint_manager

    monkeypatch.setattr(
        "cosmos_framework.posttrain.utils.object_store.upload_destination_identity",
        lambda current_upload, iter_name: {
            "kind": "object_store",
            "endpoint_authority": "storage.example.test",
            "bucket": str(current_upload["bucket"]),
            "prefix": f"{str(current_upload['prefix']).rstrip('/')}/{iter_name}",
        },
    )

    def _fail_upload(
        iter_dir: str,
        current_upload: dict[str, str],
        *,
        overrides: dict[str, str] | None = None,
        commit: dict[str, Any] | None = None,
    ) -> None:
        del iter_dir, current_upload
        assert overrides is not None and commit is not None
        raise OSError("archive unavailable")

    monkeypatch.setattr(checkpoint_manager, "archive_checkpoint", _fail_upload)
    trainer = _trainer(upload={"bucket": "archive", "prefix": "posttrain/checkpoints"})
    trainer.configure_validation(_identity())
    trainer.mark_validation_pending(0, frozenset({ValidationTrigger.FINAL}), {})
    checkpoint = tmp_path / "checkpoints" / "iter_000000"
    trainer.save(str(checkpoint), publish=False)
    trainer.complete_validation(0)
    with pytest.raises(OSError, match="archive unavailable"):
        trainer.save_validation_progress(str(checkpoint))

    changed = _trainer(upload={"bucket": "archive", "prefix": "different/checkpoints"})
    assert changed.load(str(checkpoint)) == 0
    changed.configure_validation(_identity())
    with pytest.raises(RuntimeError, match="destination changed across resume"):
        changed.save_validation_progress(str(checkpoint))


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    ("component", "error"),
    [("optim", "requires optimizer state"), ("trainer", "requires trainer state")],
)
def test_llm_resume_requires_consolidated_training_state(
    tmp_path: Any,
    component: str,
    error: str,
) -> None:
    trainer = _trainer()
    checkpoint = tmp_path / component / "iter_000000"
    trainer.save(str(checkpoint))
    (checkpoint / f"{component}.pt").unlink()

    with pytest.raises(RuntimeError, match=error):
        _trainer().load(str(checkpoint))


@pytest.mark.L0
@pytest.mark.CPU
def test_llm_dcp_resume_requires_trainer_before_model_collectives(tmp_path: Any) -> None:
    checkpoint = tmp_path / "iter_000000"
    (checkpoint / "model").mkdir(parents=True)

    with pytest.raises(RuntimeError, match="requires trainer state in DCP checkpoint"):
        _trainer()._load_dcp(str(checkpoint))


@pytest.mark.L0
@pytest.mark.CPU
def test_llm_dcp_resume_requires_optimizer_component(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch.distributed.checkpoint.state_dict as state_dict

    checkpoint = tmp_path / "iter_000000"
    checkpoint.mkdir()
    torch.save({"iter": 0}, checkpoint / "trainer.pt")
    monkeypatch.setattr(state_dict, "get_model_state_dict", lambda _net: {})
    monkeypatch.setattr(state_dict, "set_model_state_dict", lambda _net, _state: None)
    monkeypatch.setattr(state_dict, "get_optimizer_state_dict", lambda _net, _opt: {})
    trainer = _trainer()
    monkeypatch.setattr(
        trainer.checkpoint_manager,
        "load_dcp_component",
        lambda _path, name, _state: name == "model",
    )

    with pytest.raises(RuntimeError, match="requires optimizer state in DCP checkpoint"):
        trainer._load_dcp(str(checkpoint))
