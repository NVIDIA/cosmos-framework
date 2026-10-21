# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""CPU tests for the unified checkpoint API shared by every engine trainer."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any

import pytest
import torch

from cosmos_framework.posttrain.checkpoint import (
    EXTERNAL_COPY_PROVENANCE_FILE,
    CheckpointManager,
    apply_local_archive_retention,
    latest_dir,
    load_component,
    load_with_stable_provenance,
    prepare_external_checkpoint_copy,
    promote_external_checkpoint_copy,
    rank0_snapshot_is_immutable_cpu,
    replace_component_atomic,
    resolve_resume,
    save_components,
)
from cosmos_framework.posttrain.checkpoint.archive.marker import ARCHIVE_RECORD, marker_covers, read_marker
from cosmos_framework.posttrain.checkpoint.artifact import (
    CHECKPOINT_ARTIFACT_MANIFEST,
    read_local_artifact_manifest,
)
from cosmos_framework.posttrain.schema.checkpoint import FULL_CHECKPOINT_COMPONENTS
from cosmos_framework.posttrain.tests.helpers.distributed import init_gloo_process_group
from cosmos_framework.posttrain.tests.helpers.object_store_fake import FakeS3Client
from cosmos_framework.posttrain.utils.object_store import file_sha256


def _write_credentials(path: Path, endpoint: str = "https://storage.example.test") -> str:
    """Write a non-secret credential JSON so destination identities can bind an endpoint authority."""
    path.write_text(
        json.dumps({"endpoint_url": endpoint, "aws_access_key_id": "test-key", "aws_secret_access_key": "test-key"}),
        encoding="utf-8",
    )
    return str(path)


def _save_complete_training_checkpoint(path: Path, *, value: int) -> None:
    save_components(
        str(path),
        {
            "model": {"weight": torch.tensor([value])},  # [1]
            "optim": {"step": value},
            "trainer": {"iter": value},
        },
    )


def _rank0_snapshot_proof_worker(rank: int, world_size: int, rendezvous_path: str) -> None:
    import torch.distributed as dist

    init_gloo_process_group(rank, world_size, rendezvous_path, timeout_seconds=10)
    try:
        live = torch.ones(2)  # [D]
        snapshot = {"weight": live.clone()} if rank == 0 else {}  # [D] on rank 0
        decision = rank0_snapshot_is_immutable_cpu(snapshot, {"weight": live})
        decisions: list[bool | None] = [None] * world_size
        dist.all_gather_object(decisions, decision)
        assert decisions == [True] * world_size
    finally:
        dist.destroy_process_group()


def _manager_load_failure_worker(rank: int, world_size: int, rendezvous_path: str) -> None:
    import torch.distributed as dist

    from cosmos_framework.posttrain.checkpoint import LocalCheckpointStore

    class _RankStore(LocalCheckpointStore):
        def load_dcp_component(
            self,
            iter_location: str,
            name: str,
            state: dict[str, Any],
            *,
            allow_partial: bool = False,
        ) -> bool:
            del iter_location, name, state, allow_partial
            if rank == 1:
                raise OSError("injected component preflight failure")
            return True

    init_gloo_process_group(rank, world_size, rendezvous_path, timeout_seconds=10)
    try:
        manager = CheckpointManager(checkpoint_format="dcp", checkpoint_store=_RankStore())
        error_text: str | None = None
        try:
            manager.load_dcp_component(
                "unused/iter_000001",
                "model",
                {"weight": torch.zeros(1)},  # [1]
            )
        except RuntimeError as error:
            error_text = str(error)
        errors: list[str | None] = [None] * world_size
        dist.all_gather_object(errors, error_text)
        assert all(error is not None and "injected component preflight failure" in error for error in errors)
    finally:
        dist.destroy_process_group()


def _manager_load_shape_mismatch_worker(rank: int, world_size: int, rendezvous_path: str) -> None:
    import torch.distributed as dist

    from cosmos_framework.posttrain.checkpoint import manager as manager_module

    init_gloo_process_group(rank, world_size, rendezvous_path, timeout_seconds=10)
    original_isdir = manager_module.os.path.isdir
    try:
        manager_module.os.path.isdir = lambda _path: rank == 0
        manager = CheckpointManager()
        error_text: str | None = None
        try:
            manager.load_checkpoint("unused/iter_000001", lambda: 1)
        except RuntimeError as error:
            error_text = str(error)
        errors: list[str | None] = [None] * world_size
        dist.all_gather_object(errors, error_text)
        assert all(error is not None and "provenance capability differs across ranks" in error for error in errors)
    finally:
        manager_module.os.path.isdir = original_isdir
        dist.destroy_process_group()


def _manager_component_plan_mismatch_worker(rank: int, world_size: int, rendezvous_path: str) -> None:
    import torch.distributed as dist

    init_gloo_process_group(rank, world_size, rendezvous_path, timeout_seconds=10)
    try:
        manager = CheckpointManager()
        checkpoint = f"{rendezvous_path}.checkpoint/iter_000001"

        def _converged_error(action: Any, expected: str) -> None:
            error_text: str | None = None
            try:
                action()
            except RuntimeError as error:
                error_text = str(error)
            errors: list[str | None] = [None] * world_size
            dist.all_gather_object(errors, error_text)
            assert all(error is not None and expected in error for error in errors)

        _converged_error(
            lambda: manager.component_plan(["model"] if rank == 0 else ["optimizer"]),
            "checkpoint component plan differs across ranks",
        )
        _converged_error(
            lambda: manager.component_plan(["model"] if rank == 0 else ["unknown"]),
            "checkpoint component plan preparation failed on one or more ranks",
        )
        _converged_error(
            lambda: manager.save_consolidated(
                checkpoint,
                {"model": {"weight": torch.ones(1)}} if rank == 0 else {"unknown": {"weight": torch.ones(1)}},
                async_safe_snapshot=False,
            ),
            "consolidated checkpoint physical component preparation failed on one or more ranks",
        )
        _converged_error(
            lambda: manager.save_consolidated(
                checkpoint,
                {"model": {"weight": torch.ones(1)}} if rank == 0 else {"trainer": {"iteration": 1}},
                async_safe_snapshot=False,
            ),
            "consolidated checkpoint physical component plan differs across ranks",
        )

        def _partial_warning_as_error() -> None:
            import warnings

            with warnings.catch_warnings():
                if rank == 0:
                    warnings.simplefilter("error", UserWarning)
                manager.component_plan(["model"])

        _converged_error(_partial_warning_as_error, "partial checkpoint warning failed on one or more ranks")
    finally:
        dist.destroy_process_group()


def _emulate_supported_atomic_noreplace(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a single-process stand-in for filesystems that implement RENAME_NOREPLACE."""
    from cosmos_framework.posttrain.checkpoint import external_copy

    def _rename_noreplace(source: str, target: str) -> None:
        # Production requires the atomic kernel primitive; this synchronous stand-in is test-only.
        if os.path.lexists(target):
            raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), target)
        os.rename(source, target)

    monkeypatch.setattr(external_copy, "_rename_directory_noreplace", _rename_noreplace)


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"checkpoint_format": "consolidated"}, "requires checkpoint_format='dcp'"),
        ({"upload": {"bucket": "archive"}}, "mutually exclusive with upload.enabled=true"),
        ({"keep_optim_last": 1}, "remote retention is unsupported"),
        ({"keep_model_last": 1}, "remote retention is unsupported"),
    ],
)
def test_manager_is_the_single_direct_checkpoint_policy_gate(
    override: dict[str, Any],
    message: str,
) -> None:
    values: dict[str, Any] = {
        "checkpoint_format": "dcp",
        "checkpoint_store": {
            "backend": "object_store",
            "bucket": "primary",
            "prefix": "run/checkpoints",
            "credentials": "credentials/<credential>.secret",
        },
    }
    values.update(override)

    with pytest.raises(ValueError, match=message):
        CheckpointManager(**values)


@pytest.mark.L0
@pytest.mark.CPU
def test_manager_resolves_component_and_optimizer_retention_defaults() -> None:
    local = CheckpointManager()
    direct = CheckpointManager(
        checkpoint_format="dcp",
        checkpoint_store={
            "backend": "object_store",
            "bucket": "primary",
            "prefix": "run/checkpoints",
            "credentials": "credentials/<credential>.secret",
        },
    )

    assert local.keep_optim_last == 1
    assert direct.keep_optim_last == 0
    assert local.component_plan().names == FULL_CHECKPOINT_COMPONENTS


@pytest.mark.L0
@pytest.mark.CPU
def test_partial_consolidated_save_commits_only_selected_artifact_without_latest(tmp_path: Path) -> None:
    with pytest.warns(UserWarning, match="partial checkpoint artifact"):
        manager = CheckpointManager(checkpoint_components=["model"])
    checkpoint = tmp_path / "run" / "checkpoints" / "iter_000004"
    plan = manager.component_plan()

    location = manager.save_consolidated(
        str(checkpoint),
        {
            "model": {"weight": torch.ones(2)},  # [2]
            "optim": {"step": 4},
            "trainer": {"iter": 4},
        },
        async_safe_snapshot=False,
        component_plan=plan,
    )

    artifact = tmp_path / "run" / "checkpoint_artifacts" / "iter_000004"
    assert location == str(artifact)
    assert {path.name for path in artifact.iterdir()} == {"model.pt", CHECKPOINT_ARTIFACT_MANIFEST}
    assert not (checkpoint.parent / "latest_checkpoint.txt").exists()
    assert not checkpoint.exists()
    with pytest.raises(ValueError, match="not exact-resume capable"):
        manager.resolve_resume(str(artifact), str(checkpoint.parent))
    with pytest.raises(FileExistsError, match="without content identity"):
        manager.save_consolidated(
            str(checkpoint),
            {"model": {"weight": torch.zeros(2)}},  # [2]
            async_safe_snapshot=False,
            component_plan=plan,
        )
    with pytest.raises(ValueError, match="cannot be published before its latest pointer"):
        manager.save_consolidated(
            str(checkpoint),
            {"model": {}, "optim": {}, "trainer": {}},
            async_safe_snapshot=False,
            update_latest=False,
            component_plan=manager.component_plan(FULL_CHECKPOINT_COMPONENTS),
        )


@pytest.mark.L0
@pytest.mark.CPU
def test_consolidated_save_writes_dcp_components_as_shards_beside_the_files(tmp_path: Path) -> None:
    """A consolidated save can carry a bulk component as DCP shards: the layout stays one complete
    iteration directory (``optim/`` in place of ``optim.pt``), an artifact records the shard kind,
    and one name may not arrive in both forms."""
    from cosmos_framework.posttrain.checkpoint import component_kind, load_component_dcp
    from cosmos_framework.posttrain.checkpoint.io import is_complete_iter_dir

    manager = CheckpointManager()
    root = tmp_path / "run" / "checkpoints"
    checkpoint = root / "iter_000002"
    shards = {"state": {"w": {"moment": torch.arange(2.0)}}, "param_groups": [{"lr": 0.1}]}  # moment: [2]

    location = manager.save_consolidated(
        str(checkpoint),
        {"model": {"net": {"w": torch.ones(2)}}, "optim": None, "trainer": {"iter": 2}},  # w: [2]
        async_safe_snapshot=False,
        dcp_components={"optim": shards},
    )
    manager.finalize()

    assert location == str(checkpoint)
    assert {path.name for path in checkpoint.iterdir()} == {"model.pt", "optim", "trainer.pt"}
    assert component_kind(str(checkpoint), "optim") == "dcp"
    assert is_complete_iter_dir(str(checkpoint))
    assert (root / "latest_checkpoint.txt").read_text().strip() == "iter_000002"
    assert manager.resolve_resume("auto", str(root)) == str(checkpoint)
    target = {"state": {"w": {"moment": torch.zeros(2)}}, "param_groups": [{"lr": 0.0}]}  # moment: [2]
    assert load_component_dcp(str(checkpoint), "optim", target)
    assert torch.equal(target["state"]["w"]["moment"], torch.arange(2.0))
    assert target["param_groups"] == [{"lr": 0.1}]

    with pytest.raises(ValueError, match="both consolidated and DCP state"):
        manager.save_consolidated(
            str(root / "iter_000003"),
            {"model": {"net": {"w": torch.ones(2)}}, "optim": {"step": 3}, "trainer": {"iter": 3}},  # w: [2]
            async_safe_snapshot=False,
            dcp_components={"optim": shards},
        )
    assert not (root / "iter_000003").exists()

    with pytest.warns(UserWarning, match="partial checkpoint artifact"):
        plan = manager.component_plan(("model", "optimizer"))
    location = manager.save_consolidated(
        str(root / "iter_000004"),
        {"model": {"net": {"w": torch.ones(2)}}, "optim": None, "trainer": {"iter": 4}},  # w: [2]
        async_safe_snapshot=False,
        component_plan=plan,
        dcp_components={"optim": shards},
    )
    artifact = tmp_path / "run" / "checkpoint_artifacts" / "iter_000004"
    assert location == str(artifact)
    manifest = read_local_artifact_manifest(str(artifact))
    assert manifest["checkpoint_components"] == ["model", "optimizer"]
    assert manifest["components"]["optim"] == {"kind": "dcp", "path": "optim"}
    assert manifest["components"]["model"]["kind"] == "file"
    assert "trainer" not in manifest["components"]
    assert not (root / "iter_000004").exists()
    assert (root / "latest_checkpoint.txt").read_text().strip() == "iter_000002"


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(("async_safe", "background"), [(True, True), (False, False)])
def test_partial_consolidated_async_requires_a_safe_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    async_safe: bool,
    background: bool,
) -> None:
    from cosmos_framework.posttrain.checkpoint import manager as manager_module

    original_commit = manager_module._commit_checkpoint_artifact
    caller = threading.get_ident()
    commit_threads: list[int] = []

    def _record_commit(*args: Any) -> None:
        commit_threads.append(threading.get_ident())
        original_commit(*args)

    monkeypatch.setattr(manager_module, "_commit_checkpoint_artifact", _record_commit)
    with pytest.warns(UserWarning, match="partial checkpoint artifact"):
        manager = CheckpointManager(checkpoint_components=["model"], save_async=True)
    location = manager.save_consolidated(
        str(tmp_path / "run" / "checkpoints" / f"iter_{int(async_safe):06d}"),
        {"model": {"weight": torch.ones(1)}},  # [1]
        async_safe_snapshot=async_safe,
        component_plan=manager.component_plan(),
    )
    manager.finalize()

    assert len(commit_threads) == 1
    assert (commit_threads[0] != caller) is background
    assert Path(location, CHECKPOINT_ARTIFACT_MANIFEST).is_file()


@pytest.mark.L0
@pytest.mark.CPU
def test_partial_consolidated_archive_uploads_payload_then_artifact_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cosmos_framework.posttrain.utils import object_store

    store = FakeS3Client()
    monkeypatch.setattr(object_store, "build_s3_client", lambda _credentials: store)
    monkeypatch.setattr(object_store, "_transfer_config", lambda: object())
    upload = {
        "bucket": "archive",
        "prefix": "project/run/checkpoints",
        "credentials": _write_credentials(tmp_path / "archive.json"),
    }
    with pytest.warns(UserWarning, match="partial checkpoint artifact"):
        manager = CheckpointManager(checkpoint_components=["model"], upload=upload)
    checkpoint = tmp_path / "run" / "checkpoints" / "iter_000005"
    artifact_dir = tmp_path / "run" / "checkpoint_artifacts" / checkpoint.name
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "optim.pt").write_bytes(b"stale partial-attempt payload")
    location = manager.save_consolidated(
        str(checkpoint),
        {"model": {"weight": torch.ones(1)}, "optim": {"step": 5}},  # [1]
        async_safe_snapshot=False,
        component_plan=manager.component_plan(),
    )
    model_sha256 = file_sha256(os.path.join(location, "model.pt"))
    manifest_sha256 = file_sha256(os.path.join(location, CHECKPOINT_ARTIFACT_MANIFEST))
    descriptor = read_local_artifact_manifest(location)["components"]["model"]
    assert descriptor == {"kind": "file", "path": "model.pt", "size": descriptor["size"], "sha256": model_sha256}

    prefix = "project/run/checkpoint_artifacts/iter_000005"
    assert [key for _bucket, key, _extra in store.uploads] == [
        f"{prefix}/{ARCHIVE_RECORD}",
        f"{prefix}/model.pt",
        f"{prefix}/{CHECKPOINT_ARTIFACT_MANIFEST}",
    ], "the inventory reserves the prefix, then the payload uploads, then the manifest commits"
    assert all(bucket == "archive" for bucket, _key, _extra in store.uploads)
    assert store.uploads[0][2]["IfNoneMatch"] == "*"
    assert store.uploads[-1][2]["IfNoneMatch"] == "*"
    assert store.objects[f"{prefix}/model.pt"][1] == {"sha256": model_sha256}
    assert store.objects[f"{prefix}/{CHECKPOINT_ARTIFACT_MANIFEST}"][1] == {"sha256": manifest_sha256}
    assert f"{prefix}/optim.pt" not in store.objects

    destination = {
        "kind": "object_store",
        "endpoint_authority": "storage.example.test",
        "bucket": "archive",
        "prefix": "project/run/checkpoint_artifacts/iter_000005",
    }
    marker = read_marker(location)
    assert marker is not None and not marker.legacy
    assert marker.destination == destination
    assert marker.manifest_sha256 == manifest_sha256
    payload_files = {
        "model.pt": {"size": descriptor["size"], "sha256": model_sha256},
        CHECKPOINT_ARTIFACT_MANIFEST: {
            "size": os.path.getsize(os.path.join(location, CHECKPOINT_ARTIFACT_MANIFEST)),
            "sha256": manifest_sha256,
        },
    }
    declaration = store.objects[f"{prefix}/{ARCHIVE_RECORD}"][0]
    assert json.loads(declaration)["files"] == payload_files
    assert json.loads(declaration)["manifest_sha256"] == manifest_sha256
    assert marker.files == {
        **payload_files,
        ARCHIVE_RECORD: {"size": len(declaration), "sha256": hashlib.sha256(declaration).hexdigest()},
    }
    assert all(f"{prefix}/{relative}" in store.heads for relative in marker.files)
    # The stale optimizer payload was never archived: the marker does not cover the directory until it is gone.
    assert not marker_covers(location, destination)
    Path(location, "optim.pt").unlink()
    assert marker_covers(location, destination)
    assert not marker_covers(location, {**destination, "prefix": "project/run/checkpoints/iter_000005"})


@pytest.mark.L0
@pytest.mark.CPU
def test_partial_consolidated_archive_failure_writes_no_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cosmos_framework.posttrain.utils import object_store

    store = FakeS3Client()

    def _corrupt_payload(key: str) -> None:
        if key.endswith("/model.pt"):
            store.corrupt_uploads = True

    store.before_put = _corrupt_payload
    monkeypatch.setattr(object_store, "build_s3_client", lambda _credentials: store)
    monkeypatch.setattr(object_store, "_transfer_config", lambda: object())
    with pytest.warns(UserWarning, match="partial checkpoint artifact"):
        manager = CheckpointManager(
            checkpoint_components=["model"],
            upload={
                "bucket": "archive",
                "prefix": "project/run/checkpoints",
                "credentials": _write_credentials(tmp_path / "c.json"),
            },
        )
    location = manager.save_consolidated(
        str(tmp_path / "run" / "checkpoints" / "iter_000006"),
        {"model": {"weight": torch.ones(1)}},  # [1]
        async_safe_snapshot=False,
        component_plan=manager.component_plan(),
    )

    assert read_marker(location) is None and not Path(location, ".uploaded").exists()
    assert Path(location, "model.pt").is_file(), "the local artifact is retained after a failed archive"
    prefix = "project/run/checkpoint_artifacts/iter_000006"
    assert f"{prefix}/{ARCHIVE_RECORD}" in store.objects
    assert f"{prefix}/model.pt" in store.objects
    assert f"{prefix}/{CHECKPOINT_ARTIFACT_MANIFEST}" not in store.objects, "a bad payload never earns a commit"


@pytest.mark.L0
@pytest.mark.CPU
def test_save_components_layout_and_roundtrip(tmp_path):
    root = tmp_path / "checkpoints"
    it = root / "iter_000010"
    save_components(str(it), {"model": {"w": torch.ones(2)}, "optim": {"lr": 0.1}, "ref": None})
    assert (it / "model.pt").is_file() and (it / "optim.pt").is_file()
    assert not (it / "ref.pt").exists()  # None/empty components are skipped
    assert (root / "latest_checkpoint.txt").read_text().strip() == "iter_000010"
    assert torch.equal(load_component(str(it), "model")["w"], torch.ones(2))
    assert load_component(str(it), "ref") is None  # absent component → None, not an error


@pytest.mark.L0
@pytest.mark.CPU
def test_external_pending_copy_promotes_atomically_and_is_reusable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cosmos_framework.posttrain.checkpoint.external_copy import resolve_external_checkpoint_successor
    from cosmos_framework.posttrain.checkpoint.provenance import target_matches_external_copy
    from cosmos_framework.posttrain.utils import object_store

    _emulate_supported_atomic_noreplace(monkeypatch)
    source = tmp_path / "old" / "checkpoints" / "iter_000001"
    target = tmp_path / "new" / "checkpoints" / "iter_000001"
    _save_complete_training_checkpoint(source, value=1)

    plan = prepare_external_checkpoint_copy(str(source), str(target))
    assert plan.staging_dir is not None and not target.exists()
    save_components(
        plan.staging_dir,
        {"model": {"weight": torch.ones(1)}, "optim": {"step": 1}, "trainer": {"iter": 1}},  # [1]
        update_latest=False,
    )
    assert not (target.parent / "latest_checkpoint.txt").exists()
    promote_external_checkpoint_copy(plan)

    assert target.is_dir() and not Path(plan.staging_dir).exists()
    assert (target.parent / "latest_checkpoint.txt").read_text().strip() == target.name
    marker = json.loads((target / EXTERNAL_COPY_PROVENANCE_FILE).read_text(encoding="utf-8"))
    assert {key: marker[key] for key in plan.provenance} == plan.provenance
    assert isinstance(marker["copy_signature"], str)
    assert marker["successor_signatures"] == []

    # A new process attempt with the same immutable source repairs/reaffirms latest without saving.
    (target / ".uploaded").write_text("archive\n", encoding="utf-8")
    (target.parent / "latest_checkpoint.txt").unlink()
    retry = prepare_external_checkpoint_copy(str(source), str(target))
    assert retry.staging_dir is None
    assert (target.parent / "latest_checkpoint.txt").read_text().strip() == target.name

    # The framework's atomic journal replacement extends the provenance marker before the rename,
    # so a fresh process can reuse the promoted successor instead of reloading the old pending source.
    replaced_trainer = {"iter": 1, "validation": {"publication": {"step": 1}}}
    replace_component_atomic(str(target), "trainer", replaced_trainer)
    assert load_component(str(target), "trainer") == replaced_trainer
    assert torch.equal(load_component(str(target), "model")["weight"], torch.ones(1))  # [1]
    assert not (target / ".trainer.pt.tmp").exists()
    successor_retry = prepare_external_checkpoint_copy(str(source), str(target))
    assert successor_retry.staging_dir is None

    # A marker fsync failure after the remote commit must preserve the copied successor's provenance,
    # so a fresh process can retry publication instead of refusing its own pending checkpoint.
    store = FakeS3Client()
    monkeypatch.setattr(object_store, "build_s3_client", lambda _credentials: store)
    monkeypatch.setattr(object_store, "_transfer_config", lambda: object())
    manager = CheckpointManager(
        keep_optim_last=0,
        upload={
            "bucket": "archive",
            "prefix": "project/run/checkpoints",
            "credentials": _write_credentials(tmp_path / "archive.json"),
        },
    )
    publication_id = "c" * 64
    publication = {
        "step": 1,
        "id": publication_id,
        "archive_required": True,
        "destination": manager.publication_identity(str(target))["destination"],
        "manifest": f".validation-publication-v1-{publication_id}.json",
    }
    completed_trainer = {"iter": 1, "validation": {"publication": None}}
    original_fsync = os.fsync

    def _fail_marker_fsync(descriptor: int) -> None:
        if os.readlink(f"/proc/self/fd/{descriptor}").endswith("/.uploaded.tmp"):
            raise OSError(errno.ENOSPC, "injected archive marker fsync failure")
        original_fsync(descriptor)

    with monkeypatch.context() as failure:
        failure.setattr(os, "fsync", _fail_marker_fsync)
        with pytest.raises(OSError, match="archive marker fsync failure"):
            manager.publish_validation(
                str(target), completed_trainer_state=completed_trainer, publication_record=publication
            )

    assert f"project/run/checkpoints/{target.name}/{publication['manifest']}" in store.objects
    assert (target / ".uploaded.tmp").exists(), "the interrupted marker write leaves crash residue"
    assert (target / ".uploaded").read_text(encoding="utf-8") == "archive\n", "the previous marker survives"
    assert load_component(str(target), "trainer") == replaced_trainer, "local completion awaits a successful archive"
    assert target_matches_external_copy(str(target), plan.provenance)
    assert resolve_external_checkpoint_successor(str(source), str(target)) is not None
    publication_retry = prepare_external_checkpoint_copy(str(source), str(target), expected_provenance=plan.provenance)
    assert publication_retry.staging_dir is None

    manager.publish_validation(str(target), completed_trainer_state=completed_trainer, publication_record=publication)
    assert not (target / ".uploaded.tmp").exists(), "successful retry atomically publishes the marker"
    assert load_component(str(target), "trainer") == completed_trainer
    assert read_marker(str(target)) is not None
    assert target_matches_external_copy(str(target), plan.provenance)
    assert prepare_external_checkpoint_copy(str(source), str(target)).staging_dir is None

    # An out-of-band trainer mutation is not an authorized journal transition and still fails closed.
    torch.save({"iter": 1, "validation": {"pending_step": None}}, target / "trainer.pt")
    with pytest.raises(FileExistsError, match="refusing to overwrite pre-existing checkpoint target"):
        prepare_external_checkpoint_copy(str(source), str(target))


@pytest.mark.L0
@pytest.mark.CPU
def test_external_pending_copy_write_or_promotion_failure_never_changes_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cosmos_framework.posttrain.checkpoint import external_copy

    source = tmp_path / "old" / "checkpoints" / "iter_000001"
    target = tmp_path / "new" / "checkpoints" / "iter_000001"
    _save_complete_training_checkpoint(source, value=1)
    plan = prepare_external_checkpoint_copy(str(source), str(target))
    assert plan.staging_dir is not None

    # A trainer crash can leave arbitrary partial staging, but never creates the authoritative target.
    partial = Path(plan.staging_dir)
    partial.mkdir(parents=True)
    (partial / "model.pt").write_bytes(b"partial")
    assert not target.exists()
    with pytest.raises(RuntimeError, match="incomplete external checkpoint staging"):
        promote_external_checkpoint_copy(plan)
    assert not target.exists()

    _save_complete_training_checkpoint(partial, value=1)
    (target.parent / "latest_checkpoint.txt").unlink()

    def _fail_rename(_source: str, _target: str) -> None:
        raise OSError(5, "injected I/O failure")

    monkeypatch.setattr(external_copy, "_rename_directory_noreplace", _fail_rename)
    with pytest.raises(OSError, match="injected I/O failure"):
        promote_external_checkpoint_copy(plan)
    assert not target.exists()
    assert partial.is_dir()
    assert not (target.parent / "latest_checkpoint.txt").exists()


@pytest.mark.L0
@pytest.mark.CPU
def test_external_pending_copy_race_never_replaces_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _emulate_supported_atomic_noreplace(monkeypatch)
    source = tmp_path / "old" / "checkpoints" / "iter_000001"
    _save_complete_training_checkpoint(source, value=1)

    raced_target = tmp_path / "raced" / "checkpoints" / "iter_000001"
    raced_plan = prepare_external_checkpoint_copy(str(source), str(raced_target))
    assert raced_plan.staging_dir is not None
    save_components(
        raced_plan.staging_dir,
        {"model": {"weight": torch.ones(1)}, "optim": {"step": 1}, "trainer": {"iter": 1}},  # [1]
        update_latest=False,
    )
    _save_complete_training_checkpoint(raced_target, value=99)
    with pytest.raises(FileExistsError, match="refusing to overwrite checkpoint target"):
        promote_external_checkpoint_copy(raced_plan)
    assert load_component(str(raced_target), "trainer") == {"iter": 99}
    assert Path(raced_plan.staging_dir).is_dir()


@pytest.mark.L0
@pytest.mark.CPU
def test_external_pending_copy_refuses_source_mutation_after_staging(tmp_path: Path) -> None:
    source = tmp_path / "old" / "checkpoints" / "iter_000001"
    target = tmp_path / "new" / "checkpoints" / "iter_000001"
    _save_complete_training_checkpoint(source, value=1)
    plan = prepare_external_checkpoint_copy(str(source), str(target))
    assert plan.staging_dir is not None
    save_components(
        plan.staging_dir,
        {"model": {"weight": torch.ones(1)}, "optim": {"step": 1}, "trainer": {"iter": 1}},  # [1]
        update_latest=False,
    )
    torch.save({"iter": 2}, source / "trainer.pt")

    with pytest.raises(RuntimeError, match="source changed while it was being copied"):
        promote_external_checkpoint_copy(plan)
    assert not target.exists()


@pytest.mark.L0
@pytest.mark.CPU
def test_stable_provenance_rejects_checkpoint_mutation_during_load(tmp_path: Path) -> None:
    source = tmp_path / "old" / "checkpoints" / "iter_000001"
    _save_complete_training_checkpoint(source, value=1)

    def _mutating_load() -> int:
        torch.save({"iter": 2}, source / "trainer.pt")
        return 1

    with pytest.raises(RuntimeError, match="checkpoint changed while it was being loaded"):
        load_with_stable_provenance(str(source), _mutating_load)


@pytest.mark.L0
@pytest.mark.CPU
def test_latest_dir_pointer_and_fallback(tmp_path):
    root = tmp_path / "checkpoints"
    assert latest_dir(str(root)) is None  # nothing there yet
    save_components(str(root / "iter_000005"), {"model": {"a": 1}, "trainer": {"iter": 5}})
    save_components(str(root / "iter_000010"), {"model": {"a": 2}, "trainer": {"iter": 10}})
    assert latest_dir(str(root)).endswith("iter_000010")
    # Pointer lost → fall back to the highest iter_* dir.
    (root / "latest_checkpoint.txt").unlink()
    assert latest_dir(str(root)).endswith("iter_000010")


@pytest.mark.L0
@pytest.mark.CPU
def test_latest_dir_fallback_ignores_higher_partial_dcp_directory(tmp_path: Path) -> None:
    """A DCP crash before trainer.pt lands must not shadow the prior complete checkpoint."""
    root = tmp_path / "checkpoints"
    complete = root / "iter_000010"
    save_components(str(complete), {"model": {"a": 1}, "trainer": {"iter": 10}})
    (root / "latest_checkpoint.txt").unlink()
    (root / "iter_000011" / "model").mkdir(parents=True)

    assert latest_dir(str(root)) == str(complete)


@pytest.mark.L0
@pytest.mark.CPU
def test_resolve_resume_semantics(tmp_path):
    root = tmp_path / "checkpoints"
    # "off"/"" → fresh; "auto" with no checkpoints → fresh (how requeued jobs first launch).
    assert resolve_resume("", str(root)) is None
    assert resolve_resume("off", str(root)) is None
    assert resolve_resume("auto", str(root)) is None
    save_components(str(root / "iter_000003"), {"model": {"a": 1}})
    assert resolve_resume("auto", str(root)).endswith("iter_000003")
    assert resolve_resume("off", str(root)) is None  # off forces fresh even with ckpts present
    # Explicit path must exist (dir or legacy file) — else fail loud.
    assert resolve_resume(str(root / "iter_000003"), str(root)).endswith("iter_000003")
    with pytest.raises(FileNotFoundError):
        resolve_resume(str(root / "iter_999999"), str(root))


@pytest.mark.L0
@pytest.mark.CPU
def test_async_write_and_finalize(tmp_path):
    """Async save returns immediately; finalize() guarantees the files + pointer are on
    disk; a second save joins the first (single write slot — ordering preserved)."""
    from cosmos_framework.posttrain.checkpoint import finalize

    root = tmp_path / "checkpoints"
    save_components(str(root / "iter_000001"), {"model": {"w": torch.ones(2)}}, async_write=True)
    save_components(str(root / "iter_000002"), {"model": {"w": torch.zeros(2)}}, async_write=True)
    finalize()
    assert (root / "iter_000001" / "model.pt").is_file()
    assert (root / "iter_000002" / "model.pt").is_file()
    assert (root / "latest_checkpoint.txt").read_text().strip() == "iter_000002"
    finalize()  # idempotent


@pytest.mark.L0
@pytest.mark.CPU
def test_rank0_snapshot_proof_broadcasts_one_async_decision(tmp_path: Path) -> None:
    """Non-writer ranks with empty full-state dicts follow rank 0 instead of selecting sync."""
    import torch.multiprocessing as mp

    mp.spawn(
        _rank0_snapshot_proof_worker,
        args=(2, str(tmp_path / "snapshot-proof-rendezvous")),
        nprocs=2,
        join=True,
    )


@pytest.mark.L0
@pytest.mark.CPU
def test_manager_component_failure_converges_across_ranks(tmp_path: Path) -> None:
    import torch.multiprocessing as mp

    mp.spawn(
        _manager_load_failure_worker,
        args=(2, str(tmp_path / "manager-load-failure-rendezvous")),
        nprocs=2,
        join=True,
    )


@pytest.mark.L0
@pytest.mark.CPU
def test_manager_checkpoint_shape_mismatch_converges_before_provenance_collectives(tmp_path: Path) -> None:
    import torch.multiprocessing as mp

    mp.spawn(
        _manager_load_shape_mismatch_worker,
        args=(2, str(tmp_path / "manager-load-shape-rendezvous")),
        nprocs=2,
        join=True,
    )


@pytest.mark.L0
@pytest.mark.CPU
def test_manager_component_plan_mismatch_converges_before_trainer_extraction(tmp_path: Path) -> None:
    import torch.multiprocessing as mp

    mp.spawn(
        _manager_component_plan_mismatch_worker,
        args=(2, str(tmp_path / "manager-component-plan-rendezvous")),
        nprocs=2,
        join=True,
    )


@pytest.mark.L0
@pytest.mark.CPU
def test_tensor_free_snapshot_fails_closed_and_forces_sync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mutable Python-only state cannot prove detachment, so it stays on the calling thread."""
    from cosmos_framework.posttrain.checkpoint import manager as manager_module

    components = {"trainer": {"progress": {"iteration": 1}}}
    async_safe = manager_module.snapshot_is_immutable_cpu(components, {"progress": components["trainer"]})
    writes: list[bool] = []

    def _capture_save(
        iter_dir: str,
        current_components: dict[str, dict | None],
        async_write: bool = False,
        keep_optim_last: int = 0,
        keep_model_last: int = 0,
        upload: dict[str, Any] | None = None,
        update_latest: bool = True,
    ) -> None:
        del iter_dir, current_components, keep_optim_last, keep_model_last, upload, update_latest
        writes.append(async_write)

    monkeypatch.setattr(manager_module, "save_components", _capture_save)
    CheckpointManager(checkpoint_format="consolidated", save_async=True).save_consolidated(
        str(tmp_path / "consolidated"),
        components,
        async_safe_snapshot=async_safe,
    )

    assert async_safe is False
    assert writes == [False]


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("shared", [{"epoch": 1}, [1, 2]])
def test_cpu_tensor_snapshot_with_shared_mutable_state_fails_closed(shared: Any) -> None:
    """One detached tensor cannot make a concurrently mutable Python object safe to serialize."""
    rng = torch.ones(1)  # [1]
    snapshot = {"trainer": {"rng": rng, "shared": shared}}
    live_state = {"shared": shared}

    assert not rank0_snapshot_is_immutable_cpu(snapshot, live_state)


@pytest.mark.L0
@pytest.mark.CPU
def test_dcp_safe_tail_is_async_even_when_consolidated_async_is_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """save_async gates consolidated snapshots, not a safe post-collective DCP tail."""
    from cosmos_framework.posttrain.checkpoint import manager as manager_module

    components = {"trainer": {"weight": torch.ones(1)}}  # [1]
    writes: list[bool] = []

    def _capture_save(
        iter_dir: str,
        current_components: dict[str, dict | None],
        async_write: bool = False,
        keep_optim_last: int = 0,
        keep_model_last: int = 0,
        upload: dict[str, Any] | None = None,
        update_latest: bool = True,
    ) -> None:
        del iter_dir, current_components, keep_optim_last, keep_model_last, upload, update_latest
        writes.append(async_write)

    monkeypatch.setattr(manager_module, "save_components", _capture_save)
    manager = CheckpointManager(checkpoint_format="dcp", save_async=False)
    monkeypatch.setattr(manager.store, "save_dcp_component", lambda *_args, **_kwargs: None)
    manager.save_dcp(
        str(tmp_path / "dcp"),
        {"model": {}, "optim": {}},
        components,
        async_safe_snapshot=True,
    )

    assert writes == [True]


@pytest.mark.L0
@pytest.mark.CPU
def test_live_scheduler_tensor_forces_consolidated_and_dcp_tail_sync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A CPU tensor shared with scheduler state is not an immutable background snapshot."""
    from cosmos_framework.posttrain.checkpoint import manager as manager_module

    live_learning_rate = torch.tensor(0.1)  # []
    components = {"trainer": {"scheduler": {"learning_rate": live_learning_rate}}}
    async_safe = rank0_snapshot_is_immutable_cpu(components, {"learning_rate": live_learning_rate})
    writes: list[bool] = []

    def _capture_save(
        iter_dir: str,
        current_components: dict[str, dict | None],
        async_write: bool = False,
        keep_optim_last: int = 0,
        keep_model_last: int = 0,
        upload: dict[str, Any] | None = None,
        update_latest: bool = True,
    ) -> None:
        del iter_dir, current_components, keep_optim_last, keep_model_last, upload, update_latest
        writes.append(async_write)

    monkeypatch.setattr(manager_module, "save_components", _capture_save)
    CheckpointManager(checkpoint_format="consolidated", save_async=True).save_consolidated(
        str(tmp_path / "consolidated"),
        components,
        async_safe_snapshot=async_safe,
    )
    manager = CheckpointManager(checkpoint_format="dcp")
    monkeypatch.setattr(manager.store, "save_dcp_component", lambda *_args, **_kwargs: None)
    manager.save_dcp(
        str(tmp_path / "dcp"),
        {"model": {}, "optim": {}},
        components,
        async_safe_snapshot=async_safe,
    )

    assert async_safe is False
    assert writes == [False, False]


@pytest.mark.L0
@pytest.mark.CPU
def test_prune_local_optimizer_state_retention(tmp_path):
    """keep_optim_last=N: optim dropped from older iters; model + trainer kept everywhere."""
    root = tmp_path / "checkpoints"
    for i in (1, 2, 3):
        save_components(
            str(root / f"iter_{i:06d}"),
            {"model": {"w": torch.ones(1)}, "optim": {"s": i}, "trainer": {"iter": i}},
            keep_optim_last=1,
        )
    assert not (root / "iter_000001" / "optim.pt").exists()
    assert not (root / "iter_000002" / "optim.pt").exists()
    assert (root / "iter_000003" / "optim.pt").is_file()  # newest kept
    for i in (1, 2, 3):  # weights + trainer state survive for EVERY iter
        assert (root / f"iter_{i:06d}" / "model.pt").is_file()
        assert (root / f"iter_{i:06d}" / "trainer.pt").is_file()


@pytest.mark.L0
@pytest.mark.CPU
def test_rng_state_roundtrip():
    """collect → perturb → restore reproduces the exact RNG streams (python/numpy/torch)."""
    import random

    import numpy as np

    from cosmos_framework.posttrain.checkpoint import collect_rng_state, restore_rng_state

    states = collect_rng_state()
    expect = (random.random(), np.random.rand(), torch.rand(2).clone())
    # Perturb all streams, then restore and replay.
    random.random(), np.random.rand(), torch.rand(5)
    assert restore_rng_state(states)
    got = (random.random(), np.random.rand(), torch.rand(2))
    assert got[0] == expect[0] and got[1] == expect[1] and torch.equal(got[2], expect[2])
    assert restore_rng_state(None) is False  # legacy checkpoint → graceful no-op
    assert restore_rng_state({0: states[0], 1: states[0]}) is False  # changed world → fresh RNG
    with pytest.raises(ValueError, match="at least one rank"):
        restore_rng_state({})  # corrupt inventory is not a legacy/missing checkpoint
    with pytest.raises(ValueError, match="RNG checkpoint states must be dictionaries"):
        restore_rng_state({0: None})  # type: ignore[dict-item]  # corrupt rank payload


@pytest.mark.L0
@pytest.mark.CPU
def test_rng_capture_failure_is_converged_before_rank_state_gather(monkeypatch: pytest.MonkeyPatch) -> None:
    """A rank-local CUDA RNG failure cannot strand peers in the following object gather."""
    from cosmos_framework.posttrain.checkpoint import collect_rng_state
    from cosmos_framework.posttrain.utils import rank_state

    gathered: list[object] = []

    def _fail_cuda_capture() -> torch.Tensor:
        raise RuntimeError("synthetic CUDA RNG capture failure")

    def _forbid_gather(state: object, *, context: str) -> dict[int, object]:
        gathered.append((state, context))
        raise AssertionError("rank-state gather must not start after local capture failure")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_rng_state", _fail_cuda_capture)
    monkeypatch.setattr(rank_state, "gather_rank_states", _forbid_gather)

    with pytest.raises(RuntimeError, match="synthetic CUDA RNG capture failure"):
        collect_rng_state()
    assert gathered == []


@pytest.mark.L0
@pytest.mark.CPU
def test_dcp_component_roundtrip(tmp_path):
    """Sharded (DCP) component save/load round-trips on a plain CPU module, and
    component_kind distinguishes the two formats."""
    from cosmos_framework.posttrain.checkpoint import component_kind, load_component_dcp, save_component_dcp

    it = tmp_path / "iter_000001"
    src = {"net.w": torch.arange(4.0), "net.b": torch.ones(2)}
    save_component_dcp(str(it), "model", src)
    assert component_kind(str(it), "model") == "dcp"
    save_components(str(it), {"trainer": {"iter": 1}})  # pointer + tiny file, mixed layout
    assert component_kind(str(it), "trainer") == "pt"

    target = {"net.w": torch.zeros(4), "net.b": torch.zeros(2)}
    assert load_component_dcp(str(it), "model", target)
    assert torch.equal(target["net.w"], torch.arange(4.0))
    assert torch.equal(target["net.b"], torch.ones(2))
    assert not load_component_dcp(str(it), "ref", {})  # absent dir → False, not an error


@pytest.mark.L0
@pytest.mark.CPU
def test_prune_local_optimizer_state_custom_keep_policy(tmp_path):
    """The retention seam: a custom keep_fn (stand-in for a future validation-score
    policy) fully controls which iters keep optimizer state."""
    from cosmos_framework.posttrain.checkpoint import prune_local_optimizer_state

    root = tmp_path / "checkpoints"
    for i in (1, 2, 3):
        save_components(str(root / f"iter_{i:06d}"), {"model": {"w": 1}, "optim": {"s": i}})
    # "Score-based" policy: keep the (pretend) best iter regardless of recency.
    prune_local_optimizer_state(str(root), keep_last=0, keep_fn=lambda names: {"iter_000001"})
    assert (root / "iter_000001" / "optim.pt").is_file()  # kept by policy, despite being oldest
    assert not (root / "iter_000002" / "optim.pt").exists()
    assert (root / "iter_000003" / "optim.pt").is_file()  # latest resume target is always protected


@pytest.mark.L0
@pytest.mark.CPU
def test_checkpoint_write_failures_are_raised_not_swallowed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Async and synchronous rank-0 write failures remain observable."""
    from cosmos_framework.posttrain.checkpoint import finalize, manager
    from cosmos_framework.posttrain.utils import dist as dist_utils

    root = tmp_path / "checkpoints"
    save_components(str(root / "iter_000001"), {"model": {"w": torch.ones(1)}})  # good baseline
    # A lambda is unpicklable → torch.save raises inside the write thread.
    save_components(str(root / "iter_000002"), {"model": {"bad": lambda: 1}}, async_write=True)
    with pytest.raises(RuntimeError, match="async checkpoint write failed"):
        finalize()
    assert (root / "latest_checkpoint.txt").read_text().strip() == "iter_000001"  # pointer not advanced
    finalize()  # error is consumed once; subsequent finalize is clean

    class _CheckpointAbort(BaseException):
        pass

    converged: list[tuple[BaseException | str | None, str]] = []
    original_raise_if_any_rank_failed = dist_utils.raise_if_any_rank_failed

    def _record_convergence(local_error: BaseException | str | None, context: str) -> None:
        converged.append((local_error, context))
        original_raise_if_any_rank_failed(local_error, context)

    def _abort_write(*_args: object, **_kwargs: object) -> None:
        raise _CheckpointAbort("interrupted synchronous write")

    monkeypatch.setattr(dist_utils, "raise_if_any_rank_failed", _record_convergence)
    monkeypatch.setattr(manager, "_write_files", _abort_write)
    with pytest.raises(_CheckpointAbort, match="interrupted synchronous write"):
        save_components(str(root / "iter_000002"), {"model": {"w": torch.zeros(1)}})
    assert isinstance(converged[-1][0], _CheckpointAbort)
    assert converged[-1][1] == "checkpoint write"


@pytest.mark.L0
@pytest.mark.CPU
def test_async_archive_retention_failure_is_rethrown_by_finalize(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unexpected archive/retention failures in the manager's async slot must surface at finalization."""
    from cosmos_framework.posttrain.checkpoint import finalize
    from cosmos_framework.posttrain.checkpoint import manager as checkpoint_manager

    checkpoint = tmp_path / "checkpoints" / "iter_000001"
    save_components(str(checkpoint), {"trainer": {"validation": {"pending_step": None}}})

    def _fail_archive_retention(*_args: Any) -> None:
        raise OSError("archive unavailable")

    # Publisher failures are best effort; an unexpected failure of the async operation itself must still surface.
    monkeypatch.setattr(checkpoint_manager, "_apply_local_archive_retention", _fail_archive_retention)
    apply_local_archive_retention(
        str(checkpoint),
        upload={"bucket": "archive", "prefix": "posttrain"},
        async_archive_retention=True,
    )

    with pytest.raises(RuntimeError, match="async checkpoint write failed: archive unavailable"):
        finalize()
    assert not (checkpoint / ".uploaded").exists()
    finalize()  # the failure is consumed exactly once


@pytest.mark.L0
@pytest.mark.CPU
def test_upload_destination_identity_allows_credential_rotation_and_binds_target(tmp_path: Path) -> None:
    """Secrets may rotate, while endpoint authority, exact bucket, and effective prefix stay bound."""
    from cosmos_framework.posttrain.utils.object_store import upload_destination_identity

    def _credentials(name: str, endpoint: str, secret: str) -> str:
        path = tmp_path / name
        path.write_text(
            json.dumps({"endpoint_url": endpoint, "aws_access_key_id": secret, "aws_secret_access_key": secret}),
            encoding="utf-8",
        )
        return str(path)

    first_credentials = _credentials("first.json", "https://storage.example.test", "first-key")
    second_credentials = _credentials("second.json", "https://storage.example.test.", "rotated-key")
    other_endpoint = _credentials("other.json", "https://other.example.test", "other-key")

    first = upload_destination_identity(
        {"bucket": "archive", "prefix": "/exact/prefix/", "credentials": first_credentials},
        "iter_000001",
    )
    rotated = upload_destination_identity(
        {"bucket": "archive", "prefix": "/exact/prefix/", "credentials": second_credentials},
        "iter_000001",
    )
    changed = upload_destination_identity(
        {"bucket": "archive", "prefix": "/exact/prefix/", "credentials": other_endpoint},
        "iter_000001",
    )

    assert first == rotated
    assert first == {
        "kind": "object_store",
        "endpoint_authority": "storage.example.test",
        "bucket": "archive",
        "prefix": "/exact/prefix/iter_000001",
    }
    assert changed != first


@pytest.mark.L0
@pytest.mark.CPU
def test_transactional_upload_forces_same_size_objects_and_commits_last(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale same-size remote object is overwritten; only the final manifest proves success."""
    from cosmos_framework.posttrain.utils import object_store

    source = tmp_path / "iter_000001"
    source.mkdir()
    (source / "model.pt").write_bytes(b"fresh-model")
    completed_trainer = tmp_path / "completed-trainer.pt"
    completed_trainer.write_bytes(b"clear-trainer")
    events: list[tuple[str, str]] = []
    remote: dict[str, tuple[bytes, dict[str, str]]] = {
        "checkpoints/iter_000001/model.pt": (b"stale-model", {}),
        "checkpoints/iter_000001/trainer.pt": (b"stale-traine", {}),
    }

    class _Body:
        def __init__(self, payload: bytes) -> None:
            self._payload = payload

        def read(self) -> bytes:
            return self._payload

    class _Client:
        def head_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
            del Bucket
            payload, metadata = remote[Key]
            return {"ContentLength": len(payload), "Metadata": metadata}

        def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
            del Bucket
            payload, metadata = remote[Key]
            return {"Body": _Body(payload), "Metadata": metadata}

        def upload_file(
            self,
            path: str,
            bucket: str,
            key: str,
            *,
            Config: Any,
            ExtraArgs: dict[str, dict[str, str]],
        ) -> None:
            del bucket, Config
            events.append(("upload", key))
            remote[key] = (Path(path).read_bytes(), dict(ExtraArgs["Metadata"]))

    monkeypatch.setattr(object_store, "build_s3_client", lambda _credentials: _Client())
    monkeypatch.setattr(object_store, "_transfer_config", lambda: object())
    commit = {
        "step": 1,
        "id": "a" * 64,
        "archive_required": True,
        "destination": {"kind": "object_store"},
        "manifest": ".validation-publication-v1-" + "a" * 64 + ".json",
    }

    def _upload() -> int:
        return object_store.upload_prefix_transactional(
            str(source),
            "bucket",
            "checkpoints/iter_000001",
            overrides={"trainer.pt": str(completed_trainer)},
            commit_manifest=commit,
            commit_path=str(commit["manifest"]),
            workers=1,
        )

    _upload()

    uploaded_keys = [key for event, key in events if event == "upload"]
    assert uploaded_keys[-1].endswith(str(commit["manifest"]))
    assert remote["checkpoints/iter_000001/model.pt"][0] == b"fresh-model"
    assert remote["checkpoints/iter_000001/trainer.pt"][0] == b"clear-trainer"

    # An exact commit is reusable only while every referenced object still has the committed
    # size/publication metadata (and the completed trainer retains its SHA-256 metadata).
    events.clear()
    assert _upload() == 0
    assert events == []

    expected_uploads = [
        "checkpoints/iter_000001/model.pt",
        "checkpoints/iter_000001/trainer.pt",
        f"checkpoints/iter_000001/{commit['manifest']}",
    ]
    for corruption in ("missing", "wrong-generation"):
        if corruption == "missing":
            remote.pop("checkpoints/iter_000001/model.pt")
        else:
            model_payload = remote["checkpoints/iter_000001/model.pt"][0]
            remote["checkpoints/iter_000001/model.pt"] = (model_payload, {"publication-id": corruption})
        events.clear()
        _upload()
        assert [key for event, key in events if event == "upload"] == expected_uploads


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("operation", ["save", "archive_retention"])
def test_async_thread_launch_failure_is_converged_and_leaves_no_slot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    """A rank-0 Thread.start failure is surfaced synchronously and cannot poison finalize."""
    from cosmos_framework.posttrain.checkpoint import finalize, worker
    from cosmos_framework.posttrain.utils import dist as dist_utils

    checkpoint = tmp_path / "checkpoints" / "iter_000001"
    if operation == "archive_retention":
        save_components(
            str(checkpoint),
            {"model": {"w": torch.ones(1)}, "trainer": {"iter": 1}},  # [1]
        )

    class _BrokenThread:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

        def start(self) -> None:
            raise RuntimeError("thread quota exhausted")

    converged: list[tuple[BaseException | str | None, str]] = []
    original_raise_if_any_rank_failed = dist_utils.raise_if_any_rank_failed

    def _record_convergence(local_error: BaseException | str | None, context: str) -> None:
        converged.append((local_error, context))
        original_raise_if_any_rank_failed(local_error, context)

    monkeypatch.setattr(dist_utils, "raise_if_any_rank_failed", _record_convergence)
    monkeypatch.setattr(worker.threading, "Thread", _BrokenThread)
    with pytest.raises(RuntimeError, match="thread quota exhausted"):
        if operation == "save":
            save_components(str(checkpoint), {"model": {"w": torch.ones(1)}}, async_write=True)  # [1]
        else:
            apply_local_archive_retention(str(checkpoint), async_archive_retention=True)

    assert worker._write_thread is None
    assert isinstance(converged[-1][0], RuntimeError)
    expected_context = (
        "checkpoint async write launch" if operation == "save" else "checkpoint async archive/retention launch"
    )
    assert converged[-1][1] == expected_context
    finalize()


@pytest.mark.L0
@pytest.mark.CPU
def test_dcp_component_waits_for_prior_async_archive_retention(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The shared DCP primitive drains older archive/retention before writing a component."""
    import torch.distributed.checkpoint as dcp

    from cosmos_framework.posttrain.checkpoint import save_component_dcp, worker

    events: list[str] = []
    monkeypatch.setattr(worker, "wait_for_pending", lambda: events.append("wait"))
    monkeypatch.setattr(dcp, "save", lambda *_args, **_kwargs: events.append("save"))

    state = {"w": torch.ones(1)}  # [1]
    save_component_dcp(str(tmp_path / "iter_000002"), "model", state)

    assert events == ["wait", "save"]


@pytest.mark.L0
@pytest.mark.CPU
def test_all_empty_components_leave_no_pointer(tmp_path):
    """All-None/empty components ⇒ nothing written AND no latest pointer — a phantom
    pointer at an empty dir would make resume=auto fail on a fresh-looking job."""
    root = tmp_path / "checkpoints"
    save_components(str(root / "iter_000001"), {"model": None, "ref": {}})
    assert not (root / "latest_checkpoint.txt").exists()
    assert latest_dir(str(root)) is None


# --------------------------------------------------------------------------- run_ts layout
@pytest.mark.L0
@pytest.mark.CPU
def test_resolve_run_ts_branches(tmp_path):
    from cosmos_framework.posttrain.checkpoint import resolve_run_ts

    name = tmp_path / "exp"
    # Fresh experiment (no dir at all): mint.
    assert resolve_run_ts(str(name), "auto", now="20260707000001") == "20260707000001"
    # resume=off never adopts — always a new sibling attempt (no overwrite semantics).
    (name / "20260707000001").mkdir(parents=True)
    assert resolve_run_ts(str(name), "off", now="20260707000002") == "20260707000002"
    # resume=auto adopts the LATEST attempt, even one with no checkpoints yet (a window
    # that crashed before its first save must not orphan its dir).
    (name / "20260707000002").mkdir()
    assert resolve_run_ts(str(name), "auto", now="20260707999999") == "20260707000002"
    # Explicit resume path INSIDE an attempt dir adopts THAT attempt, not the latest.
    ck = name / "20260707000001" / "checkpoints" / "iter_000000010"
    ck.mkdir(parents=True)
    assert resolve_run_ts(str(name), str(ck), now="x") == "20260707000001"
    # Legacy flat layout (<name>/checkpoints exists): stay in place, no ts level.
    legacy = tmp_path / "old_exp"
    (legacy / "checkpoints").mkdir(parents=True)
    assert resolve_run_ts(str(legacy), "auto", now="x") == ""


@pytest.mark.L0
@pytest.mark.CPU
def test_prune_local_iterations_keeps_newest_and_gates_on_upload(tmp_path):
    from cosmos_framework.posttrain.checkpoint.retention import prune_local_iterations

    root = tmp_path / "checkpoints"
    for i in (1, 2, 3, 4):
        save_components(str(root / f"iter_00000{i}"), {"model": {"w": torch.zeros(1)}})
    # No upload gate: oldest two entire dirs go, newest two stay.
    prune_local_iterations(str(root), 2)
    assert sorted(d.name for d in root.iterdir() if d.is_dir()) == ["iter_000003", "iter_000004"]
    # With require_uploaded: un-marked dirs are NEVER deleted (only copy protection)...
    save_components(str(root / "iter_000005"), {"model": {"w": torch.zeros(1)}})
    prune_local_iterations(str(root), 1, require_uploaded=True)
    assert sorted(d.name for d in root.iterdir() if d.is_dir()) == [
        "iter_000003",
        "iter_000004",
        "iter_000005",
    ]
    # ...and marked ones are.
    (root / "iter_000003" / ".uploaded").write_text("bucket\n")
    prune_local_iterations(str(root), 1, require_uploaded=True)
    assert sorted(d.name for d in root.iterdir() if d.is_dir()) == ["iter_000004", "iter_000005"]


@pytest.mark.L0
@pytest.mark.CPU
def test_save_with_upload_marks_and_prunes(tmp_path, monkeypatch):
    """End-to-end retention ride-along: save → (mock) upload marks the dir → older
    uploaded dirs get pruned, all inside the same write job."""
    from cosmos_framework.posttrain.checkpoint import manager

    pushed = []
    monkeypatch.setattr(
        "cosmos_framework.posttrain.utils.object_store.upload_prefix",
        lambda src, bucket, prefix, credentials=None: pushed.append((src, bucket, prefix)) or 1,
    )
    root = tmp_path / "checkpoints"
    up = {"bucket": "b", "prefix": "p/checkpoints", "credentials": _write_credentials(tmp_path / "c.json")}
    for i in (1, 2, 3):
        manager.save_components(
            str(root / f"iter_00000{i}"),
            {"model": {"w": torch.zeros(1)}},
            keep_model_last=1,
            upload=up,
        )
    assert len(pushed) == 3 and pushed[-1][2] == "p/checkpoints/iter_000003"
    dirs = sorted(d.name for d in root.iterdir() if d.is_dir())
    assert dirs == ["iter_000003"], f"older uploaded dirs should be pruned, got {dirs}"
    assert (root / "iter_000003" / ".uploaded").is_file()
    # The full-checkpoint marker is destination-bound with a size inventory; bulk bytes carry no digest.
    destination = {
        "kind": "object_store",
        "endpoint_authority": "storage.example.test",
        "bucket": "b",
        "prefix": "p/checkpoints/iter_000003",
    }
    marker = read_marker(str(root / "iter_000003"))
    assert marker is not None and not marker.legacy and marker.destination == destination
    assert marker.files == {"model.pt": {"size": (root / "iter_000003" / "model.pt").stat().st_size, "sha256": None}}
    assert marker.manifest_sha256 is None
    assert marker_covers(str(root / "iter_000003"), destination)
    assert not marker_covers(str(root / "iter_000003"), {**destination, "prefix": "p/checkpoints/iter_000002"})


@pytest.mark.L0
@pytest.mark.CPU
def test_failed_upload_gates_optimizer_retention(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Archive failure keeps every local optimizer copy until a later upload succeeds."""
    from cosmos_framework.posttrain.checkpoint import manager

    uploads = 0

    def _upload(src: str, bucket: str, prefix: str, credentials: str | None = None) -> int:
        nonlocal uploads
        del src, bucket, prefix, credentials
        uploads += 1
        if uploads == 1:
            raise OSError("archive unavailable")
        return 1

    monkeypatch.setattr("cosmos_framework.posttrain.utils.object_store.upload_prefix", _upload)
    root = tmp_path / "checkpoints"
    upload = {
        "bucket": "archive",
        "prefix": "posttrain/checkpoints",
        "credentials": _write_credentials(tmp_path / "secret.json"),
    }
    manager.save_components(
        str(root / "iter_000001"),
        {"model": {"w": torch.ones(1)}, "optim": {"step": 1}, "trainer": {"iter": 1}},
        keep_optim_last=1,
        upload=upload,
    )
    manager.save_components(
        str(root / "iter_000002"),
        {"model": {"w": torch.ones(1)}, "optim": {"step": 2}, "trainer": {"iter": 2}},
        keep_optim_last=1,
        upload=upload,
    )

    assert not (root / "iter_000001" / ".uploaded").exists()
    assert (root / "iter_000002" / ".uploaded").is_file()
    assert (root / "iter_000001" / "optim.pt").is_file()
    assert (root / "iter_000002" / "optim.pt").is_file()


@pytest.mark.L0
@pytest.mark.CPU
def test_transactional_archive_marker_records_override_digest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The strict archive marker binds the destination and carries a digest only for the hashed override."""
    from cosmos_framework.posttrain.checkpoint.archive.publisher import archive_checkpoint
    from cosmos_framework.posttrain.utils import object_store

    iter_dir = tmp_path / "checkpoints" / "iter_000001"
    _save_complete_training_checkpoint(iter_dir, value=1)
    completed_trainer = tmp_path / "completed-trainer.pt"
    torch.save({"iter": 1, "validation": {"publication": {"step": 1}}}, completed_trainer)
    client = FakeS3Client()
    monkeypatch.setattr(object_store, "build_s3_client", lambda _credentials: client)
    monkeypatch.setattr(object_store, "_transfer_config", lambda: object())
    commit = {
        "step": 1,
        "id": "b" * 64,
        "archive_required": True,
        "destination": {"kind": "object_store"},
        "manifest": ".validation-publication-v1-" + "b" * 64 + ".json",
    }
    upload = {
        "bucket": "archive",
        "prefix": "posttrain/checkpoints/",
        "credentials": _write_credentials(tmp_path / "s.json"),
    }

    archive_checkpoint(
        str(iter_dir),
        upload,
        overrides={"trainer.pt": str(completed_trainer)},
        commit=commit,
    )

    marker = read_marker(str(iter_dir))
    assert marker is not None and not marker.legacy
    assert marker.destination == {
        "kind": "object_store",
        "endpoint_authority": "storage.example.test",
        "bucket": "archive",
        "prefix": "posttrain/checkpoints/iter_000001",
    }
    assert marker.files["model.pt"] == {"size": (iter_dir / "model.pt").stat().st_size, "sha256": None}
    assert marker.files["optim.pt"] == {"size": (iter_dir / "optim.pt").stat().st_size, "sha256": None}
    assert marker.files["trainer.pt"] == {
        "size": completed_trainer.stat().st_size,
        "sha256": file_sha256(str(completed_trainer)),
    }, "the archived override, not the local pre-completion trainer, is what the marker describes"
    assert (
        client.objects["posttrain/checkpoints/iter_000001/trainer.pt"][1]["sha256"]
        == marker.files["trainer.pt"]["sha256"]
    )
    assert set(marker.files) == {"model.pt", "optim.pt", "trainer.pt"}, "the marker itself is not payload"


@pytest.mark.L0
@pytest.mark.CPU
def test_localize_ckpt_s3_and_local(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """rl.init_from s3:// form: pre-pulled once into the shared cache (idempotent
    download_prefix); local paths pass through untouched."""
    from cosmos_framework.posttrain.control.checkpointing import CheckpointLifecycle

    monkeypatch.setenv("POSTTRAIN_CKPT_CACHE", str(tmp_path / "cache"))
    calls: list[tuple[str, str, str]] = []
    monkeypatch.setattr(
        "cosmos_framework.posttrain.utils.object_store.download_prefix",
        lambda bucket, prefix, local, **kw: calls.append((bucket, prefix, local)) or 1,
    )
    lifecycle = CheckpointLifecycle(
        manager=CheckpointManager(),
        checkpoint_root=str(tmp_path / "checkpoints"),
        resume_checkpoint=None,
    )
    out = lifecycle.localize_init_from("s3://bkt/ckpts/iter_000000100/model")
    assert out == str(tmp_path / "cache" / "bkt" / "ckpts" / "iter_000000100" / "model")
    assert calls == [("bkt", "ckpts/iter_000000100/model", out)]
    assert lifecycle.localize_init_from("/path/to/shared-storage") == "/path/to/shared-storage"


@pytest.mark.L0
@pytest.mark.CPU
def test_load_component_dcp_allow_partial(tmp_path):
    """Optimizer DCP resume with lazily-created Adam state: the checkpoint holds state
    only for gradded params, the live template asks for every param — strict load fails
    with 'Missing key in checkpoint state_dict' (the 2026-07-07 32B resume failure);
    allow_partial leaves missing entries at their fresh init."""
    from cosmos_framework.posttrain.checkpoint import load_component_dcp, save_component_dcp

    saved = {"state.a.exp_avg": torch.ones(4)}  # only param `a` ever got grads
    save_component_dcp(str(tmp_path / "iter_000000001"), "optim", saved)
    template = {"state.a.exp_avg": torch.zeros(4), "state.b.exp_avg": torch.full((2,), 7.0)}
    from torch.distributed.checkpoint.api import CheckpointException

    # strict: the extra template key must fail loudly. NB CheckpointException derives
    # from BaseException, not Exception — a bare `except Exception` will NOT catch it.
    with pytest.raises((CheckpointException, RuntimeError)):
        load_component_dcp(str(tmp_path / "iter_000000001"), "optim", dict(template))
    ok = load_component_dcp(str(tmp_path / "iter_000000001"), "optim", template, allow_partial=True)
    assert ok
    assert torch.equal(template["state.a.exp_avg"], torch.ones(4))  # loaded
    assert torch.equal(template["state.b.exp_avg"], torch.full((2,), 7.0))  # left at init


@pytest.mark.L0
@pytest.mark.CPU
def test_partial_consolidated_commit_releases_the_snapshot_before_publishing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The gathered host tensors die once the payload is on disk, before the upload starts.

    A colocated job's final drain joined a writer that still held the whole gathered model while its
    62 GB payload also filled the page cache; the commit must not carry the tensors into the publish."""
    import weakref

    from cosmos_framework.posttrain.checkpoint import manager as manager_module
    from cosmos_framework.posttrain.checkpoint.store import build_checkpoint_store

    weight = torch.ones(4)  # [4]
    alive = weakref.ref(weight)
    snapshot = manager_module.ArtifactSnapshot({"model": {"weight": weight}})
    del weight
    published: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        manager_module,
        "publish_artifact",
        lambda location, upload: published.append((location, alive() is None)) or {},
    )
    logged: list[str] = []
    monkeypatch.setattr(manager_module, "log_host_memory", lambda context: logged.append(context))
    location = str(tmp_path / "run" / "checkpoint_artifacts" / "iter_000002")

    manager_module._commit_checkpoint_artifact(
        build_checkpoint_store(None),
        location,
        "consolidated",
        ("model",),
        [],
        snapshot,
        {"bucket": "archive", "prefix": "run/checkpoints"},
    )

    assert published == [(location, True)], "publish ran while the snapshot tensors were still alive"
    assert snapshot.components is None
    assert read_local_artifact_manifest(location)["components"]["model"]["path"] == "model.pt"
    assert logged == [
        f"checkpoint artifact written {location}",
        f"checkpoint artifact publish finished {location}",
    ]
    with pytest.raises(RuntimeError, match="already consumed"):
        snapshot.take()


@pytest.mark.L0
@pytest.mark.CPU
def test_partial_consolidated_save_hands_the_writer_a_snapshot_carrier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cosmos_framework.posttrain.checkpoint import manager as manager_module

    seen: list[tuple[Any, ...]] = []
    monkeypatch.setattr(manager_module, "_commit_checkpoint_artifact", lambda *args: seen.append(args))
    with pytest.warns(UserWarning, match="partial checkpoint artifact"):
        manager = CheckpointManager(checkpoint_components=["model"])
    components = {"model": {"weight": torch.ones(1)}}  # [1]
    manager.save_consolidated(
        str(tmp_path / "run" / "checkpoints" / "iter_000001"),
        components,
        async_safe_snapshot=False,
        component_plan=manager.component_plan(),
    )

    assert len(seen) == 1
    carrier = seen[0][5]
    assert isinstance(carrier, manager_module.ArtifactSnapshot)
    assert carrier.take() == components


@pytest.mark.L0
@pytest.mark.CPU
def test_partial_dcp_save_commits_the_artifact_end_to_end(tmp_path: Path) -> None:
    """A partial DCP save writes its shards, then commits the artifact through the same one-shot
    snapshot carrier as the consolidated branch, and the result is a readable model-only artifact.

    Before this, ``save_dcp`` still handed the commit a plain mapping while the commit called
    ``snapshot.take()``: every partial DCP save died with ``AttributeError`` after its shards were on
    disk, leaving an uncommitted artifact prefix behind."""
    from cosmos_framework.posttrain.checkpoint import component_kind, load_component_dcp

    with pytest.warns(UserWarning, match="partial checkpoint artifact"):
        manager = CheckpointManager(checkpoint_format="dcp", checkpoint_components=["model", "training_state"])
    plan = manager.component_plan()
    checkpoint = tmp_path / "run" / "checkpoints" / "iter_000003"

    location = manager.save_dcp(
        str(checkpoint),
        {"model": {"net.w": torch.arange(4.0)}, "optim": {"step": torch.ones(1)}},  # [4], [1]
        {"trainer": {"iter": 3}},
        async_safe_snapshot=False,
        component_plan=plan,
    )
    manager.finalize()

    artifact = tmp_path / "run" / "checkpoint_artifacts" / "iter_000003"
    assert location == str(artifact)
    manifest = read_local_artifact_manifest(str(artifact))
    assert manifest["checkpoint_format"] == "dcp"
    assert manifest["checkpoint_components"] == list(plan.names)
    assert manifest["components"]["model"] == {"kind": "dcp", "path": "model"}
    assert manifest["components"]["trainer"]["kind"] == "file"
    assert manifest["components"]["trainer"]["path"] == "trainer.pt"
    assert "optim" not in manifest["components"], "the optimizer is outside the plan"
    assert component_kind(str(artifact), "model") == "dcp"
    target = {"net.w": torch.zeros(4)}  # [4]
    assert load_component_dcp(str(artifact), "model", target)
    assert torch.equal(target["net.w"], torch.arange(4.0))
    assert torch.load(artifact / "trainer.pt", weights_only=False) == {"iter": 3}
    assert not checkpoint.exists()
    assert not (checkpoint.parent / "latest_checkpoint.txt").exists()
    with pytest.raises(FileExistsError, match="without content identity"):
        manager.save_dcp(
            str(checkpoint),
            {"model": {"net.w": torch.zeros(4)}},  # [4]
            {"trainer": {"iter": 3}},
            async_safe_snapshot=False,
            component_plan=plan,
        )


@pytest.mark.L0
@pytest.mark.CPU
def test_partial_dcp_save_hands_the_writer_a_snapshot_carrier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cosmos_framework.posttrain.checkpoint import manager as manager_module

    seen: list[tuple[Any, ...]] = []
    monkeypatch.setattr(manager_module, "_commit_checkpoint_artifact", lambda *args: seen.append(args))
    with pytest.warns(UserWarning, match="partial checkpoint artifact"):
        manager = CheckpointManager(checkpoint_format="dcp", checkpoint_components=["model", "training_state"])
    monkeypatch.setattr(manager.store, "save_dcp_component", lambda *_args, **_kwargs: None)
    control = {"trainer": {"iter": 5}}
    manager.save_dcp(
        str(tmp_path / "run" / "checkpoints" / "iter_000005"),
        {"model": {"net.w": torch.ones(1)}, "optim": {"step": torch.ones(1)}},  # [1], [1]
        control,
        async_safe_snapshot=False,
        component_plan=manager.component_plan(),
    )

    assert len(seen) == 1
    assert seen[0][4] == ["model"], "only the planned DCP component is declared"
    carrier = seen[0][5]
    assert isinstance(carrier, manager_module.ArtifactSnapshot)
    assert carrier.take() == control


@pytest.mark.L0
@pytest.mark.CPU
def test_manager_samples_host_memory_at_every_save_and_at_finalize(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The save/drain host-memory series belongs to the process that owns the checkpoint state, not
    to the controller loop: one reading per save once its location is fixed (the components arrive
    gathered), one when this process joins its writer, INFO on rank 0 and DEBUG on the other ranks,
    for every algorithm that saves through the manager."""
    import logging

    from cosmos_framework.posttrain.checkpoint import manager as manager_module

    logged: list[tuple[str, int | None]] = []
    monkeypatch.setattr(
        manager_module,
        "log_host_memory",
        lambda context, *, level=None: logged.append((context, level)),
    )
    monkeypatch.delenv("RANK", raising=False)
    root = tmp_path / "run" / "checkpoints"
    manager = CheckpointManager()
    manager.save_consolidated(
        str(root / "iter_000001"),
        {"model": {"w": torch.ones(1)}, "optim": {"step": 1}, "trainer": {"iter": 1}},  # [1]
        async_safe_snapshot=False,
    )
    dcp_manager = CheckpointManager(checkpoint_format="dcp")
    monkeypatch.setattr(dcp_manager.store, "save_dcp_component", lambda *_args, **_kwargs: None)
    dcp_manager.save_dcp(
        str(root / "iter_000002"),
        {"model": {}, "optim": {}},
        {"trainer": {"iter": 2}},
        async_safe_snapshot=False,
    )
    manager.finalize()

    assert logged == [
        (f"checkpoint save {root / 'iter_000001'}", logging.INFO),
        (f"checkpoint save {root / 'iter_000002'}", logging.INFO),
        ("checkpoint finalize", logging.INFO),
    ]
    monkeypatch.setenv("RANK", "1")
    manager_module._log_host_memory("checkpoint finalize")
    assert logged[-1] == ("checkpoint finalize", logging.DEBUG)
