# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""CPU contracts for i4-compatible direct-primary DCP checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from cosmos_framework.posttrain.checkpoint import (
    POSTTRAIN_MANIFEST,
    CheckpointManager,
    ObjectStoreCheckpointStore,
    build_checkpoint_store,
)
from cosmos_framework.posttrain.checkpoint.artifact import CHECKPOINT_ARTIFACT_MANIFEST


class _MemoryEasyIOBackend:
    """Minimal EasyIO backend fake; DCP bulk IO is stubbed separately."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.events: list[tuple[str, str]] = []
        self.fail_put_suffix_once: str | None = None
        self.fail_put_uri_once: str | None = None
        self.put_then_fail_uri_once: str | None = None

    def put(self, payload: bytes, uri: str) -> None:
        if uri == self.fail_put_uri_once or (
            self.fail_put_suffix_once is not None and uri.endswith(self.fail_put_suffix_once)
        ):
            self.fail_put_suffix_once = None
            self.fail_put_uri_once = None
            raise OSError(f"injected put failure for {uri}")
        self.objects[uri] = bytes(payload)
        self.events.append(("put", uri))
        if uri == self.put_then_fail_uri_once:
            self.put_then_fail_uri_once = None
            raise OSError(f"injected ambiguous put success for {uri}")

    def get(self, uri: str) -> bytes:
        return self.objects[uri]

    def exists(self, uri: str) -> bool:
        return uri in self.objects

    def copy_prefix(self, source: str, target: str) -> None:
        copied = {
            target + uri.removeprefix(source): payload
            for uri, payload in self.objects.items()
            if uri.startswith(source + "/")
        }
        self.objects.update(copied)


def _metadata() -> Any:
    return SimpleNamespace(
        state_dict_metadata={
            "net.weight": SimpleNamespace(size=(2, 3)),
            "optimizer.step": SimpleNamespace(size=None),
        }
    )


def _manager(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    prefix: str = "project/group/run/checkpoints",
    backend: _MemoryEasyIOBackend | None = None,
) -> tuple[CheckpointManager, _MemoryEasyIOBackend]:
    credentials = tmp_path / "credentials.json"
    credentials.write_text("{}", encoding="utf-8")
    manager = CheckpointManager(
        checkpoint_format="dcp",
        checkpoint_store={
            "backend": "object_store",
            "bucket": "checkpoints",
            "prefix": prefix,
            "credentials": str(credentials),
        },
    )
    store = manager.store
    assert isinstance(store, ObjectStoreCheckpointStore)
    backend = backend or _MemoryEasyIOBackend()
    store.__dict__["_io"] = backend

    def _save_component(iter_location: str, name: str, state: dict[str, Any]) -> None:
        del state
        backend.put(name.encode(), f"{iter_location}/{name}/__0_0.distcp")
        backend.put(b"dcp metadata", f"{iter_location}/{name}/.metadata")

    monkeypatch.setattr(store, "save_dcp_component", _save_component)

    def _read_dcp_metadata(iter_location: str, name: str) -> Any:
        uri = f"{iter_location}/{name}/.metadata"
        if not backend.exists(uri):
            raise FileNotFoundError(f"DCP metadata is missing: {uri}")
        return _metadata()

    monkeypatch.setattr(store, "_read_dcp_metadata", _read_dcp_metadata)
    return manager, backend


def _save(manager: CheckpointManager, tmp_path: Path, iteration: int) -> str:
    local = tmp_path / "local" / "checkpoints" / f"iter_{iteration:06d}"
    return manager.save_dcp(
        str(local),
        {"model": {"weight": torch.ones(1)}, "optim": {"step": torch.ones(1)}},  # [1], [1]
        {"trainer": {"iter": iteration}},
        async_safe_snapshot=True,
    )


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    ("spec", "error", "message"),
    [
        ({"backend": "invalid"}, ValueError, "backend must be 'local' or 'object_store'"),
        ({"backend": "local", "bucket": "unused"}, ValueError, "local checkpoint_store accepts only"),
        ({"backend": "object_store", "unexpected": "value"}, ValueError, "unsupported fields"),
        ({"backend": "object_store", "bucket": 1}, TypeError, "bucket must be a string"),
        ({"backend": "object_store", "bucket": ""}, ValueError, "bucket must be one non-empty name"),
        (
            {"backend": "object_store", "bucket": "primary", "credentials": ""},
            ValueError,
            "credentials must be non-empty",
        ),
    ],
)
def test_store_builder_owns_backend_boundary_validation(
    spec: dict[str, Any],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        build_checkpoint_store(spec)


@pytest.mark.L0
@pytest.mark.CPU
def test_direct_save_commits_control_before_latest_and_uses_no_local_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, backend = _manager(tmp_path, monkeypatch)
    remote = _save(manager, tmp_path, 7)

    assert remote == "s3://checkpoints/project/group/run/checkpoints/iter_000007"
    assert not (tmp_path / "local").exists()
    assert manager.resolve_resume("auto", str(tmp_path)) == remote
    assert manager.load_component(remote, "trainer") == {"iter": 7}
    assert manager.has_dcp_component(remote, "model")
    assert manager.has_dcp_component(remote, "optim")
    puts = [uri for operation, uri in backend.events if operation == "put"]
    assert puts.index(f"{remote}/{POSTTRAIN_MANIFEST}") < puts.index(
        "s3://checkpoints/project/group/run/checkpoints/latest_checkpoint.txt"
    )


@pytest.mark.L0
@pytest.mark.CPU
def test_direct_model_only_dcp_commits_to_sibling_artifact_without_training_visibility(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, backend = _manager(
        tmp_path,
        monkeypatch,
        prefix="project/group/run/custom_checkpoint_root",
    )
    local = tmp_path / "local" / "checkpoints" / "iter_000007"
    with pytest.warns(UserWarning, match="partial checkpoint artifact"):
        plan = manager.component_plan(["model"])

    remote = manager.save_dcp(
        str(local),
        {"model": {"weight": torch.ones(1)}, "optim": {"step": torch.ones(1)}},  # [1], [1]
        {"trainer": {"iter": 7}},
        async_safe_snapshot=True,
        component_plan=plan,
    )

    assert remote == "s3://checkpoints/project/group/run/checkpoint_artifacts/iter_000007"
    assert backend.exists(f"{remote}/model/.metadata")
    assert not any(backend.exists(f"{remote}/{path}") for path in ("optim/.metadata", "trainer.pt"))
    assert backend.exists(f"{remote}/{CHECKPOINT_ARTIFACT_MANIFEST}")
    assert not backend.exists("s3://checkpoints/project/group/run/custom_checkpoint_root/latest_checkpoint.txt")
    manifest = json.loads(backend.get(f"{remote}/{CHECKPOINT_ARTIFACT_MANIFEST}"))
    assert manifest["checkpoint_components"] == ["model"]
    assert manifest["components"] == {"model": {"kind": "dcp", "path": "model"}}
    with pytest.raises(ValueError, match="not exact-resume capable"):
        manager.resolve_resume(remote, str(tmp_path))
    puts = [uri for operation, uri in backend.events if operation == "put"]
    assert puts[-1] == f"{remote}/{CHECKPOINT_ARTIFACT_MANIFEST}"

    calls: list[str] = []

    def _forbid_rewrite(_location: str, name: str, _state: dict[str, Any]) -> None:
        calls.append(name)
        raise AssertionError("committed artifact DCP writer was re-entered")

    monkeypatch.setattr(manager.store, "save_dcp_component", _forbid_rewrite)
    with pytest.raises(FileExistsError, match="without content identity"):
        manager.save_dcp(
            str(local),
            {"model": {"weight": torch.zeros(1)}, "optim": {"step": torch.ones(1)}},  # [1], [1]
            {"trainer": {"iter": 7}},
            async_safe_snapshot=True,
            component_plan=plan,
        )
    assert calls == []


@pytest.mark.L0
@pytest.mark.CPU
def test_direct_training_state_artifact_uses_verified_small_component_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, backend = _manager(tmp_path, monkeypatch)
    local = tmp_path / "local" / "checkpoints" / "iter_000008"
    with pytest.warns(UserWarning, match="not exact-resume capable"):
        plan = manager.component_plan(["training_state"])

    remote = manager.save_dcp(
        str(local),
        {"model": {"weight": torch.ones(1)}, "optim": {"step": torch.ones(1)}},  # [1], [1]
        {"trainer": {"iter": 8}},
        async_safe_snapshot=True,
        component_plan=plan,
    )

    manifest = json.loads(backend.get(f"{remote}/{CHECKPOINT_ARTIFACT_MANIFEST}"))
    descriptor = manifest["components"]["trainer"]
    assert descriptor["kind"] == "file"
    assert descriptor["path"].startswith(".posttrain-artifact/generations/")
    assert backend.exists(f"{remote}/{descriptor['path']}")
    assert not any(uri.startswith(f"{remote}/model/") for uri in backend.objects)
    assert not any(uri.startswith(f"{remote}/optim/") for uri in backend.objects)


@pytest.mark.L0
@pytest.mark.CPU
def test_partial_bulk_or_control_save_leaves_objects_but_does_not_advance_latest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, backend = _manager(tmp_path, monkeypatch)
    previous = _save(manager, tmp_path, 7)
    store = manager.store
    save_component = store.save_dcp_component

    def _fail_optim(iter_location: str, name: str, state: dict[str, Any]) -> None:
        if name == "optim":
            raise OSError("injected optimizer upload failure")
        save_component(iter_location, name, state)

    monkeypatch.setattr(store, "save_dcp_component", _fail_optim)
    with pytest.raises(OSError, match="optimizer upload failure"):
        _save(manager, tmp_path, 8)

    partial = "s3://checkpoints/project/group/run/checkpoints/iter_000008"
    assert backend.exists(f"{partial}/model/.metadata")
    assert not backend.exists(f"{partial}/{POSTTRAIN_MANIFEST}")
    assert manager.resolve_resume("auto", str(tmp_path)) == previous

    retry_components: list[str] = []

    def _record_retry(iter_location: str, name: str, state: dict[str, Any]) -> None:
        retry_components.append(name)
        save_component(iter_location, name, state)

    monkeypatch.setattr(store, "save_dcp_component", _record_retry)
    previous = _save(manager, tmp_path, 8)
    assert retry_components == ["model", "optim"]
    assert manager.resolve_resume("auto", str(tmp_path)) == previous

    monkeypatch.setattr(store, "save_dcp_component", save_component)
    backend.fail_put_suffix_once = "/trainer.pt"
    with pytest.raises(OSError, match="injected put failure"):
        _save(manager, tmp_path, 9)

    control_partial = "s3://checkpoints/project/group/run/checkpoints/iter_000009"
    assert backend.exists(f"{control_partial}/optim/.metadata")
    assert not backend.exists(f"{control_partial}/{POSTTRAIN_MANIFEST}")
    assert manager.resolve_resume("auto", str(tmp_path)) == previous


@pytest.mark.L0
@pytest.mark.CPU
def test_latest_failure_allows_same_iteration_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, backend = _manager(tmp_path, monkeypatch)
    previous = _save(manager, tmp_path, 8)
    latest = "s3://checkpoints/project/group/run/checkpoints/latest_checkpoint.txt"
    backend.fail_put_uri_once = latest

    with pytest.raises(OSError, match="injected put failure"):
        _save(manager, tmp_path, 9)

    remote = "s3://checkpoints/project/group/run/checkpoints/iter_000009"
    assert backend.exists(f"{remote}/{POSTTRAIN_MANIFEST}")
    assert manager.resolve_resume("auto", str(tmp_path)) == previous

    committed = {uri: payload for uri, payload in backend.objects.items() if uri.startswith(remote + "/")}
    bulk_calls: list[str] = []

    def _forbid_bulk_rewrite(iter_location: str, name: str, state: dict[str, Any]) -> None:
        del iter_location, state
        bulk_calls.append(name)
        raise AssertionError(f"committed bulk component was rewritten: {name}")

    monkeypatch.setattr(manager.store, "save_dcp_component", _forbid_bulk_rewrite)
    metadata_uri = f"{remote}/model/.metadata"
    metadata = backend.objects.pop(metadata_uri)
    with pytest.raises(FileNotFoundError, match="DCP metadata is missing"):
        _save(manager, tmp_path, 9)
    assert manager.resolve_resume("auto", str(tmp_path)) == previous
    assert bulk_calls == []
    backend.objects[metadata_uri] = metadata
    assert {uri: payload for uri, payload in backend.objects.items() if uri.startswith(remote + "/")} == committed

    backend.put_then_fail_uri_once = latest
    with pytest.raises(OSError, match="ambiguous put success"):
        _save(manager, tmp_path, 9)

    assert manager.resolve_resume("auto", str(tmp_path)) == remote
    assert {uri: payload for uri, payload in backend.objects.items() if uri.startswith(remote + "/")} == committed
    assert bulk_calls == []

    assert _save(manager, tmp_path, 9) == remote
    assert manager.resolve_resume("auto", str(tmp_path)) == remote
    assert bulk_calls == []


@pytest.mark.L0
@pytest.mark.CPU
def test_ambiguous_latest_success_retry_never_reenters_committed_dcp_writers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, backend = _manager(tmp_path, monkeypatch)
    latest = "s3://checkpoints/project/group/run/checkpoints/latest_checkpoint.txt"
    backend.put_then_fail_uri_once = latest

    with pytest.raises(OSError, match="ambiguous put success"):
        _save(manager, tmp_path, 9)

    remote = "s3://checkpoints/project/group/run/checkpoints/iter_000009"
    assert manager.resolve_resume("auto", str(tmp_path)) == remote
    committed = {uri: payload for uri, payload in backend.objects.items() if uri.startswith(remote + "/")}

    def _corrupt_if_called(iter_location: str, name: str, state: dict[str, Any]) -> None:
        del state
        backend.objects[f"{iter_location}/{name}/__0_0.distcp"] = b"corrupt"
        raise OSError("injected retry crash during bulk rewrite")

    monkeypatch.setattr(manager.store, "save_dcp_component", _corrupt_if_called)
    backend.events.clear()

    assert _save(manager, tmp_path, 9) == remote
    assert backend.events == []
    assert {uri: payload for uri, payload in backend.objects.items() if uri.startswith(remote + "/")} == committed

    local = tmp_path / "local" / "checkpoints" / "iter_000009"
    with pytest.raises(RuntimeError, match="component plan differs"):
        manager.save_dcp(
            str(local),
            {"model": {"weight": torch.ones(1)}, "optim": {"step": torch.ones(1)}},  # [1], [1]
            {"trainer": {"iter": 9}, "data": {"cursor": 1}},
            async_safe_snapshot=True,
        )
    assert {uri: payload for uri, payload in backend.objects.items() if uri.startswith(remote + "/")} == committed


@pytest.mark.L0
@pytest.mark.CPU
def test_complete_checkpoint_copy_is_an_explicit_resume_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, backend = _manager(tmp_path, monkeypatch)
    original = _save(manager, tmp_path, 11)
    copied = "s3://archive/copied/checkpoints/iter_000011"
    backend.copy_prefix(original, copied)

    assert manager.resolve_resume(copied, str(tmp_path)) == copied
    assert manager.load_component(copied, "trainer") == {"iter": 11}
    assert manager.has_dcp_component(copied, "model")


@pytest.mark.L0
@pytest.mark.CPU
def test_direct_tensor_inventory_comes_from_i4_dcp_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, _backend = _manager(tmp_path, monkeypatch)
    remote = _save(manager, tmp_path, 3)

    assert manager.dcp_tensor_inventory(remote, "model") == {
        "net.weight": (2, 3),
        "optimizer.step": None,
    }


@pytest.mark.L0
@pytest.mark.CPU
def test_direct_control_descriptor_rejects_same_size_payload_corruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, backend = _manager(tmp_path, monkeypatch)
    local = tmp_path / "local" / "checkpoints" / "iter_000012"
    manager.save_dcp(
        str(local),
        {"model": {"weight": torch.ones(1)}, "optim": {"step": torch.ones(1)}},  # [1], [1]
        {"trainer": {"iter": 12}, "data": {"position": 5}},
        async_safe_snapshot=True,
    )
    remote = manager.iteration_location(str(local))
    manifest = json.loads(backend.get(f"{remote}/{POSTTRAIN_MANIFEST}"))
    data_uri = f"{remote}/{manifest['control_components']['data']['path']}"
    corrupted = bytearray(backend.get(data_uri))
    corrupted[-1] ^= 1
    backend.objects[data_uri] = bytes(corrupted)

    with pytest.raises(RuntimeError, match="control component failed verification"):
        manager.load_component(remote, "data")


@pytest.mark.L0
@pytest.mark.CPU
def test_validation_publication_retry_advances_only_the_control_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, backend = _manager(tmp_path, monkeypatch)
    remote = _save(manager, tmp_path, 10)
    pending = {"iter": 10, "validation": {"publication": {"step": 10}}}
    completed = {"iter": 10, "validation": {"publication": None}}
    manager.replace_component(remote, "trainer", pending)
    publication_id = "a" * 64
    record = {
        "step": 10,
        "id": publication_id,
        "archive_required": False,
        "destination": manager.publication_identity(remote)["destination"],
        "manifest": f".validation-publication-v1-{publication_id}.json",
    }
    control_pointer = f"{remote}/{POSTTRAIN_MANIFEST}"
    visibility = f"{remote}/{record['manifest']}"
    bulk_before = {uri for uri in backend.objects if "/model/" in uri or "/optim/" in uri}
    backend.fail_put_uri_once = control_pointer

    with pytest.raises(OSError, match="injected put failure"):
        manager.publish_validation(
            remote,
            completed_trainer_state=completed,
            publication_record=record,
        )

    assert backend.exists(visibility)
    assert manager.load_component(remote, "trainer") == pending

    manager.publish_validation(
        remote,
        completed_trainer_state=completed,
        publication_record=record,
    )
    bulk_after = {uri for uri in backend.objects if "/model/" in uri or "/optim/" in uri}
    assert manager.load_component(remote, "trainer") == completed
    assert bulk_after == bulk_before

    committed_manifest = backend.get(control_pointer)
    bulk_calls: list[str] = []

    def _forbid_bulk_rewrite(iter_location: str, name: str, state: dict[str, Any]) -> None:
        del iter_location, state
        bulk_calls.append(name)
        raise AssertionError(f"validation-complete bulk component was rewritten: {name}")

    monkeypatch.setattr(manager.store, "save_dcp_component", _forbid_bulk_rewrite)
    assert _save(manager, tmp_path, 10) == remote
    assert manager.load_component(remote, "trainer") == completed
    assert backend.get(control_pointer) == committed_manifest
    assert bulk_calls == []


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("copied_visibility_commit", [False, True])
def test_copied_pending_publication_rebinds_to_the_configured_direct_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    copied_visibility_commit: bool,
) -> None:
    from cosmos_framework.posttrain.roles.trainer.validation.publication import PublicationJournal
    from cosmos_framework.posttrain.roles.trainer.validation.state import ValidationState

    source_manager, backend = _manager(tmp_path, monkeypatch)
    source = _save(source_manager, tmp_path, 10)
    publication_id = "a" * 64
    source_record = {
        "step": 10,
        "id": publication_id,
        "archive_required": False,
        "destination": source_manager.publication_identity(source)["destination"],
        "manifest": f".validation-publication-v1-{publication_id}.json",
    }
    pending = {"iter": 10, "validation": {"publication": source_record}}
    completed = {"iter": 10, "validation": {"publication": None}}
    source_manager.replace_component(source, "trainer", pending)
    if copied_visibility_commit:
        backend.fail_put_uri_once = f"{source}/{POSTTRAIN_MANIFEST}"
        with pytest.raises(OSError, match="injected put failure"):
            source_manager.publish_validation(
                source,
                completed_trainer_state=completed,
                publication_record=source_record,
            )

    copied = "s3://checkpoints/copied/checkpoints/iter_000010"
    backend.copy_prefix(source, copied)
    copied_manager, _ = _manager(
        tmp_path,
        monkeypatch,
        prefix="copied/checkpoints",
        backend=backend,
    )
    state = ValidationState({"enabled": True})
    journal = PublicationJournal(state)
    journal.restore(source_record, iteration=10)
    callbacks: list[str] = []

    def trainer_state() -> dict[str, Any]:
        return {"iter": 10, "validation": {"publication": journal.checkpoint_record()}}

    def fail_trainer_state() -> dict[str, Any]:
        raise OSError("injected trainer serialization failure")

    with pytest.raises(OSError, match="trainer serialization failure"):
        journal.publish(
            copied,
            manager=copied_manager,
            publication_identity={},
            trainer_state=fail_trainer_state,
            on_checkpoint=lambda: callbacks.append(copied),
        )
    assert journal.checkpoint_record() == source_record

    journal.publish(
        copied,
        manager=copied_manager,
        publication_identity={},
        trainer_state=trainer_state,
        on_checkpoint=lambda: callbacks.append(copied),
    )

    assert copied_manager.load_component(copied, "trainer") == completed
    manifest = json.loads(backend.get(f"{copied}/{POSTTRAIN_MANIFEST}"))
    rebound = manifest["validation_publication"]
    assert rebound["destination"] == {"kind": "object_store", "uri": copied}
    assert rebound["id"] != source_record["id"]
    assert backend.exists(f"{copied}/{rebound['manifest']}")
    assert callbacks == [copied]
