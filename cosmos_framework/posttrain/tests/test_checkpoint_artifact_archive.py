# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""CPU known answers for the offline artifact archive (publish / verify / fetch / prune-verified).

Every case runs on the in-memory ``FakeS3Client``; nothing touches a bucket. The cases that were
wrong under the earlier size-only rule are called out: a same-size stranger skipped the upload and
was then "verified" for deletion, a corrupted store released the only good copy, and a marker
alone (never re-HEADed) released files another pruner had already removed remotely.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.distributed.checkpoint as dcp

pytest.importorskip("botocore")

from cosmos_framework.posttrain.checkpoint.archive.marker import (
    ARCHIVE_PROTOCOL,
    ARCHIVE_RECORD,
    MARKER_NAME,
    marker_covers,
    read_marker,
    write_marker,
)
from cosmos_framework.posttrain.checkpoint.archive.publisher import (
    LEGACY_MANIFEST_RECORD,
    artifact_destination,
    publish_artifact,
)
from cosmos_framework.posttrain.checkpoint.archive.reader import (
    STAGED_PROTOCOL,
    STAGED_RECORD,
    discard_staged_artifact,
    fetch_archived_artifact,
    verify_archived_artifact,
)
from cosmos_framework.posttrain.checkpoint.artifact import (
    CHECKPOINT_ARTIFACT_MANIFEST,
    build_artifact_manifest,
    commit_local_artifact,
    read_local_artifact_manifest,
)
from cosmos_framework.posttrain.checkpoint.io import atomic_write_json
from cosmos_framework.posttrain.checkpoint.retention import prune_verified_artifacts
from cosmos_framework.posttrain.checkpoint.tools.archive_cli import main
from cosmos_framework.posttrain.tests.helpers.object_store_fake import FakeS3Client
from cosmos_framework.posttrain.utils import object_store

WEIGHTS = b"merged" * 4096
STRANGER = b"STALE!" * 4096  # same size, different bytes
RUN = "proj/group/run/20260909000000"
ITER = "iter_000300"
DESTINATION = f"{RUN}/checkpoint_artifacts/{ITER}"
MODEL_KEY = f"{DESTINATION}/model.pt"
MANIFEST_KEY = f"{DESTINATION}/{CHECKPOINT_ARTIFACT_MANIFEST}"
ARCHIVE_KEY = f"{DESTINATION}/{ARCHIVE_RECORD}"
IDENTITY = {
    "kind": "object_store",
    "endpoint_authority": "storage.example.test",
    "bucket": "archive",
    "prefix": DESTINATION,
}


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _credentials(path: Path, endpoint: str = "https://storage.example.test") -> str:
    """Write a non-secret credential JSON so destination identities can bind an endpoint authority."""
    path.write_text(
        json.dumps({"endpoint_url": endpoint, "aws_access_key_id": "test-key", "aws_secret_access_key": "test-key"}),
        encoding="utf-8",
    )
    return str(path)


def _artifact(root: Path, *, body: bytes = WEIGHTS, name: str = ITER, digest: bool = True) -> Path:
    artifact_dir = root / "runs" / RUN / "checkpoint_artifacts" / name
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "model.pt").write_bytes(body)
    descriptor = {"kind": "file", "path": "model.pt", "size": len(body)}
    if digest:
        descriptor["sha256"] = _sha(body)
    manifest = build_artifact_manifest(str(artifact_dir), "consolidated", ("model",), {"model": descriptor})
    atomic_write_json(str(artifact_dir / CHECKPOINT_ARTIFACT_MANIFEST), manifest)
    return artifact_dir


def _upload(tmp_path: Path) -> dict[str, str]:
    return {"bucket": "archive", "prefix": f"{RUN}/checkpoints", "credentials": _credentials(tmp_path / "creds.json")}


def _manifest_bytes(artifact_dir: Path) -> bytes:
    return (artifact_dir / CHECKPOINT_ARTIFACT_MANIFEST).read_bytes()


def _publish(artifact_dir: str, upload: dict[str, str], *, delete_local: bool = False) -> dict[str, Any]:
    """Compose publication and optional verified retention, as the archive CLI does."""
    report = dict(publish_artifact(artifact_dir, upload))
    report["removed"] = []
    if delete_local:
        retained = prune_verified_artifacts([artifact_dir], credentials=upload["credentials"])
        if retained["kept"]:
            raise RuntimeError(str(retained["kept"]))
        report["removed"] = retained["removed"].get(artifact_dir, [])
    return report


def _plant_inventory(store: FakeS3Client, prefix: str, manifest: bytes, files: dict[str, bytes]) -> None:
    """Bind synthetic payloads to the native commit with an explicit archive declaration."""
    payloads = {**files, CHECKPOINT_ARTIFACT_MANIFEST: manifest}
    record = {
        "protocol": ARCHIVE_PROTOCOL,
        "manifest_sha256": _sha(manifest),
        "files": {relative: {"size": len(body), "sha256": _sha(body)} for relative, body in payloads.items()},
    }
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    store.plant(f"{prefix}/{ARCHIVE_RECORD}", encoded, {"sha256": _sha(encoded)})


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> FakeS3Client:
    client = FakeS3Client()
    monkeypatch.setattr(object_store, "build_s3_client", lambda _credentials=None: client)
    monkeypatch.setattr(object_store, "_transfer_config", lambda: object())
    return client


@pytest.mark.L0
@pytest.mark.CPU
def test_artifact_destination_matches_the_trainer_archive_rule(tmp_path: Path) -> None:
    upload = _upload(tmp_path)
    assert artifact_destination(upload, ITER) == (DESTINATION, IDENTITY)
    with pytest.raises(ValueError):
        artifact_destination({**upload, "prefix": ""}, ITER)
    with pytest.raises(ValueError):
        artifact_destination(upload, "nested/iter_000300")


@pytest.mark.L0
@pytest.mark.CPU
def test_publish_uploads_payloads_then_manifest_and_binds_the_marker(tmp_path: Path, store: FakeS3Client) -> None:
    artifact_dir = _artifact(tmp_path)
    upload = _upload(tmp_path)

    report = _publish(str(artifact_dir), upload)

    manifest_sha256 = _sha(_manifest_bytes(artifact_dir))
    assert report["status"] == "published" and report["removed"] == []
    assert (report["bucket"], report["prefix"], report["destination"]) == ("archive", DESTINATION, IDENTITY)
    assert [key for _bucket, key, _extra in store.uploads] == [ARCHIVE_KEY, MODEL_KEY, MANIFEST_KEY], (
        "reserve first, commit last"
    )
    assert store.objects[MODEL_KEY] == (WEIGHTS, {"sha256": _sha(WEIGHTS)})
    assert store.objects[MANIFEST_KEY] == (_manifest_bytes(artifact_dir), {"sha256": manifest_sha256})
    marker = read_marker(str(artifact_dir))
    assert marker is not None and not marker.legacy
    assert marker.destination == IDENTITY and marker.manifest_sha256 == manifest_sha256
    assert marker.files == {
        "model.pt": {"size": len(WEIGHTS), "sha256": _sha(WEIGHTS)},
        CHECKPOINT_ARTIFACT_MANIFEST: {"size": len(_manifest_bytes(artifact_dir)), "sha256": manifest_sha256},
        ARCHIVE_RECORD: {"size": len(store.objects[ARCHIVE_KEY][0]), "sha256": _sha(store.objects[ARCHIVE_KEY][0])},
    }
    assert report["files"] == marker.files and report["manifest_sha256"] == manifest_sha256
    assert marker_covers(str(artifact_dir), IDENTITY)
    assert not list(artifact_dir.glob(f"{MARKER_NAME}.tmp*")), "atomic marker write leaves no residue"
    assert (artifact_dir / "model.pt").read_bytes() == WEIGHTS, "no --delete-local: the payload stays"
    assert verify_archived_artifact(str(artifact_dir), credentials=upload["credentials"])


@pytest.mark.L0
@pytest.mark.CPU
def test_publish_reuploads_a_same_size_stranger_left_without_a_commit(tmp_path: Path, store: FakeS3Client) -> None:
    """OLD: a same-size object meant "already uploaded"; the stranger survived and the local copy was released."""
    artifact_dir = _artifact(tmp_path)
    store.plant(MODEL_KEY, STRANGER)

    _publish(str(artifact_dir), _upload(tmp_path))

    assert store.objects[MODEL_KEY] == (WEIGHTS, {"sha256": _sha(WEIGHTS)}), "the stranger is replaced"
    assert store.puts == 3

    # A stranger carrying forged digest metadata under an uncommitted prefix is replaced just the same.
    other = _artifact(tmp_path / "other")
    store.plant(MODEL_KEY, STRANGER, {"sha256": _sha(WEIGHTS)})
    store.objects.pop(MANIFEST_KEY)
    _publish(str(other), _upload(tmp_path / "other"))
    assert store.objects[MODEL_KEY][0] == WEIGHTS


@pytest.mark.L0
@pytest.mark.CPU
def test_publish_is_idempotent_and_adopts_an_identical_committed_archive(tmp_path: Path, store: FakeS3Client) -> None:
    artifact_dir = _artifact(tmp_path)
    upload = _upload(tmp_path)
    _publish(str(artifact_dir), upload)
    uploads = list(store.uploads)

    assert _publish(str(artifact_dir), upload)["status"] == "skipped"
    assert store.uploads == uploads, "a verified, marked archive is not re-uploaded"

    # Crash between the upload and the marker write: the retry adopts the identical committed archive.
    (artifact_dir / MARKER_NAME).unlink()
    assert _publish(str(artifact_dir), upload)["status"] == "adopted"
    assert store.uploads == uploads
    assert marker_covers(str(artifact_dir), IDENTITY)

    # A marker for another destination or an unverifiable remote does not qualify for the skip.
    write_marker(
        str(artifact_dir),
        destination={**IDENTITY, "prefix": "elsewhere/iter_000300"},
        files=read_marker(str(artifact_dir)).files,  # type: ignore[union-attr]
        manifest_sha256=_sha(_manifest_bytes(artifact_dir)),
    )
    assert _publish(str(artifact_dir), upload)["status"] == "adopted"
    assert read_marker(str(artifact_dir)).destination == IDENTITY  # type: ignore[union-attr]


@pytest.mark.L0
@pytest.mark.CPU
def test_publish_never_overwrites_a_committed_archive_that_differs_or_no_longer_verifies(
    tmp_path: Path, store: FakeS3Client
) -> None:
    artifact_dir = _artifact(tmp_path)
    upload = _upload(tmp_path)
    _publish(str(artifact_dir), upload)
    uploads = list(store.uploads)

    # Different content at the same iteration: the write-once commit is refused, local retained.
    different = _artifact(tmp_path / "different", body=STRANGER)
    with pytest.raises(FileExistsError, match="refusing to overwrite committed object-store artifact"):
        _publish(str(different), _upload(tmp_path / "different"), delete_local=True)
    assert store.uploads == uploads and (different / "model.pt").read_bytes() == STRANGER
    assert read_marker(str(different)) is None

    # Same commit, but the payload object was replaced: forged digest metadata on a same-size stranger,
    # or our digest claimed for other-sized bytes. Neither qualifies; the commit is not overwritten.
    (artifact_dir / MARKER_NAME).unlink()
    for body, metadata in (
        (STRANGER, {"sha256": _sha(STRANGER)}),
        (STRANGER, {}),
        (WEIGHTS[:-1], {"sha256": _sha(WEIGHTS)}),
    ):
        store.plant(MODEL_KEY, body, metadata)
        with pytest.raises(RuntimeError, match="does not verify"):
            _publish(str(artifact_dir), upload, delete_local=True)
        assert store.uploads == uploads
        assert (artifact_dir / "model.pt").read_bytes() == WEIGHTS and read_marker(str(artifact_dir)) is None


@pytest.mark.L0
@pytest.mark.CPU
def test_publish_corrupt_store_raises_runtime_error_and_retains_the_local_copy(
    tmp_path: Path, store: FakeS3Client
) -> None:
    """The gate between a corrupted upload and --delete-local is an explicit raise, never an ``assert``."""
    artifact_dir = _artifact(tmp_path)
    store.corrupt_uploads = True

    with pytest.raises(RuntimeError, match="SHA-256 verification"):
        _publish(str(artifact_dir), _upload(tmp_path), delete_local=True)

    assert (artifact_dir / "model.pt").read_bytes() == WEIGHTS, "local copy retained"
    assert read_marker(str(artifact_dir)) is None and not (artifact_dir / MARKER_NAME).exists(), "no marker"
    assert MANIFEST_KEY not in store.objects, "the commit was never written"
    assert MODEL_KEY not in store.objects, "a corrupt declaration is refused before any payload is written"


@pytest.mark.L0
@pytest.mark.CPU
def test_publish_rewrites_a_legacy_merge_manifest_and_binds_missing_digests(
    tmp_path: Path, store: FakeS3Client
) -> None:
    artifact_dir = _artifact(tmp_path)
    native = json.loads(_manifest_bytes(artifact_dir))
    del native["components"]["model"]["sha256"]
    legacy = {**native, "merge": {"mode": "tv", "lam": 1.0, "experts": ["a", "b"], "identity_digest": "0" * 64}}
    legacy_bytes = json.dumps(legacy, sort_keys=True).encode("utf-8")
    (artifact_dir / CHECKPOINT_ARTIFACT_MANIFEST).write_bytes(legacy_bytes)
    with pytest.raises(RuntimeError, match="unsupported posttrain checkpoint artifact manifest"):
        read_local_artifact_manifest(str(artifact_dir))

    report = _publish(str(artifact_dir), _upload(tmp_path))

    manifest = read_local_artifact_manifest(str(artifact_dir))
    assert manifest["components"]["model"] == {
        "kind": "file",
        "path": "model.pt",
        "size": len(WEIGHTS),
        "sha256": _sha(WEIGHTS),
    }
    assert (artifact_dir / LEGACY_MANIFEST_RECORD).read_bytes() == legacy_bytes, "the original manifest is preserved"
    assert store.objects[MANIFEST_KEY][0] == _manifest_bytes(artifact_dir), "the native manifest is what was archived"
    assert json.loads(store.objects[MANIFEST_KEY][0]) == manifest
    assert set(report["files"]) == {"model.pt", LEGACY_MANIFEST_RECORD, CHECKPOINT_ARTIFACT_MANIFEST, ARCHIVE_RECORD}, (
        "whole directory and remote declaration"
    )
    assert marker_covers(str(artifact_dir), IDENTITY)

    # A native manifest without digests is bound in place too (no legacy copy: nothing was dropped).
    plain = _artifact(tmp_path / "plain", name="iter_000301", digest=False)
    _publish(str(plain), _upload(tmp_path / "plain"))
    assert read_local_artifact_manifest(str(plain))["components"]["model"]["sha256"] == _sha(WEIGHTS)
    assert not (plain / LEGACY_MANIFEST_RECORD).exists()

    # A manifest that is not a native superset is never invented.
    broken = _artifact(tmp_path / "broken")
    (broken / CHECKPOINT_ARTIFACT_MANIFEST).write_text(
        json.dumps({"protocol": "x", "components": {}}), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="unsupported posttrain checkpoint artifact manifest"):
        _publish(str(broken), _upload(tmp_path / "broken"))
    bare = tmp_path / "bare" / "runs" / RUN / "checkpoint_artifacts" / ITER
    bare.mkdir(parents=True)
    (bare / "model.pt").write_bytes(WEIGHTS)
    with pytest.raises(FileNotFoundError):
        _publish(str(bare), _upload(tmp_path / "bare"))


@pytest.mark.L0
@pytest.mark.CPU
def test_publish_refuses_a_locally_corrupted_artifact(tmp_path: Path, store: FakeS3Client) -> None:
    artifact_dir = _artifact(tmp_path)
    (artifact_dir / "model.pt").write_bytes(STRANGER)

    with pytest.raises(RuntimeError, match="does not match its manifest digest"):
        _publish(str(artifact_dir), _upload(tmp_path))
    assert store.uploads == [] and read_marker(str(artifact_dir)) is None

    (artifact_dir / "model.pt").write_bytes(WEIGHTS + b"x")
    with pytest.raises(RuntimeError, match="missing or resized"):
        _publish(str(artifact_dir), _upload(tmp_path))
    assert store.uploads == []


@pytest.mark.L0
@pytest.mark.CPU
def test_verify_archived_artifact_known_answers(tmp_path: Path, store: FakeS3Client) -> None:
    artifact_dir = _artifact(tmp_path)
    upload = _upload(tmp_path)
    credentials = upload["credentials"]
    assert not verify_archived_artifact(str(artifact_dir), credentials=credentials), "no marker"
    _publish(str(artifact_dir), upload)
    assert verify_archived_artifact(str(artifact_dir), credentials=credentials)

    good = store.objects[MODEL_KEY]
    store.objects.pop(MODEL_KEY)
    assert not verify_archived_artifact(str(artifact_dir), credentials=credentials), "absent"
    store.plant(MODEL_KEY, STRANGER, {"sha256": _sha(STRANGER)})
    assert not verify_archived_artifact(str(artifact_dir), credentials=credentials), "different digest"
    store.plant(MODEL_KEY, STRANGER)
    assert not verify_archived_artifact(str(artifact_dir), credentials=credentials), "no digest metadata"
    store.plant(MODEL_KEY, WEIGHTS[:-1], {"sha256": _sha(WEIGHTS)})
    assert not verify_archived_artifact(str(artifact_dir), credentials=credentials), "other size, forged digest"
    store.plant(MODEL_KEY, *good)
    assert verify_archived_artifact(str(artifact_dir), credentials=credentials)

    other_endpoint = _credentials(tmp_path / "other.json", endpoint="https://other.example.test")
    assert not verify_archived_artifact(str(artifact_dir), credentials=other_endpoint), "another store's credentials"
    (artifact_dir / MARKER_NAME).write_text("archive\n", encoding="utf-8")
    assert not verify_archived_artifact(str(artifact_dir), credentials=credentials), "legacy marker never verifies"


@pytest.mark.L0
@pytest.mark.CPU
def test_prune_verified_keeps_on_every_doubt_and_removes_only_live_verified_payloads(
    tmp_path: Path, store: FakeS3Client
) -> None:
    upload = _upload(tmp_path)
    credentials = upload["credentials"]
    verified = _artifact(tmp_path)
    _publish(str(verified), upload)
    stranger = _artifact(tmp_path / "stranger", name="iter_000301")
    _publish(str(stranger), _upload(tmp_path / "stranger"))
    store.plant(f"{RUN}/checkpoint_artifacts/iter_000301/model.pt", STRANGER, {"sha256": _sha(STRANGER)})
    absent = _artifact(tmp_path / "absent", name="iter_000302")
    _publish(str(absent), _upload(tmp_path / "absent"))
    store.objects.pop(f"{RUN}/checkpoint_artifacts/iter_000302/model.pt")
    no_digest = _artifact(tmp_path / "nodigest", name="iter_000303")
    _publish(str(no_digest), _upload(tmp_path / "nodigest"))
    marker = read_marker(str(no_digest))
    assert marker is not None
    write_marker(
        str(no_digest),
        destination=marker.destination,
        files={**marker.files, "model.pt": {"size": len(WEIGHTS), "sha256": None}},
        manifest_sha256=marker.manifest_sha256,
    )
    legacy = _artifact(tmp_path / "legacy", name="iter_000304")
    _publish(str(legacy), _upload(tmp_path / "legacy"))
    (legacy / MARKER_NAME).write_text("archive\n", encoding="utf-8")
    unmarked = _artifact(tmp_path / "unmarked", name="iter_000305")
    modified = _artifact(tmp_path / "modified", name="iter_000306")
    _publish(str(modified), _upload(tmp_path / "modified"))
    (modified / "model.pt").write_bytes(STRANGER)
    outside = tmp_path / "outside" / "checkpoints" / ITER
    outside.mkdir(parents=True)
    (outside / "model.pt").write_bytes(WEIGHTS)
    atomic_write_json(str(outside / CHECKPOINT_ARTIFACT_MANIFEST), json.loads(_manifest_bytes(verified)))
    write_marker(str(outside), destination=IDENTITY, files=read_marker(str(verified)).files)  # type: ignore[union-attr]

    paths = [
        str(verified),
        str(stranger / "model.pt"),
        str(absent),
        str(no_digest),
        str(legacy),
        str(unmarked),
        str(modified),
        str(outside),
        str(tmp_path / "missing" / ITER),
    ]
    report = prune_verified_artifacts(paths, credentials=credentials)

    assert report["removed"] == {str(verified): ["model.pt"]} and report["freed_bytes"] == len(WEIGHTS)
    assert not (verified / "model.pt").exists()
    assert (verified / CHECKPOINT_ARTIFACT_MANIFEST).is_file() and (verified / MARKER_NAME).is_file()
    assert set(report["kept"]) == set(paths[1:])
    assert "different" in report["kept"][str(stranger / "model.pt")], "OLD released on size alone"
    assert "absent" in report["kept"][str(absent)]
    assert "no archived digest" in report["kept"][str(no_digest)]
    assert "legacy" in report["kept"][str(legacy)]
    assert "no usable archive marker" in report["kept"][str(unmarked)]
    assert "local bytes differ" in report["kept"][str(modified)]
    assert "checkpoint_artifacts/" in report["kept"][str(outside)]
    assert "not a committed local checkpoint artifact" in report["kept"][str(tmp_path / "missing" / ITER)]
    for kept in (stranger, absent, no_digest, legacy, unmarked, modified, outside):
        assert (kept / "model.pt").is_file(), kept
    # The manifest object counts too: a re-written commit (other bytes) keeps the local payload.
    assert prune_verified_artifacts([str(verified)], credentials=credentials)["kept"] == {
        str(verified): "no local payload to release"
    }

    # A store that claims our digest for other-sized bytes is corrupt: keep, and raise after the sweep.
    corrupt = _artifact(tmp_path / "corrupt", name="iter_000307")
    _publish(str(corrupt), _upload(tmp_path / "corrupt"))
    store.plant(f"{RUN}/checkpoint_artifacts/iter_000307/model.pt", WEIGHTS[:-1], {"sha256": _sha(WEIGHTS)})
    with pytest.raises(RuntimeError, match="corrupt object store"):
        prune_verified_artifacts([str(corrupt)], credentials=credentials)
    assert (corrupt / "model.pt").read_bytes() == WEIGHTS

    # Another store's credentials never release a copy bound to this one.
    other = _credentials(tmp_path / "other.json", endpoint="https://other.example.test")
    fresh = _artifact(tmp_path / "fresh", name="iter_000308")
    _publish(str(fresh), _upload(tmp_path / "fresh"))
    assert "another destination" in prune_verified_artifacts([str(fresh)], credentials=other)["kept"][str(fresh)]
    assert (fresh / "model.pt").is_file()


@pytest.mark.L0
@pytest.mark.CPU
def test_fetch_stages_manifest_first_verifies_payloads_and_records_ownership(
    tmp_path: Path, store: FakeS3Client
) -> None:
    artifact_dir = _artifact(tmp_path)
    upload = _upload(tmp_path)
    _publish(str(artifact_dir), upload, delete_local=True)
    store.heads.clear()
    stage = tmp_path / "stage"

    local = fetch_archived_artifact(f"s3://archive/{DESTINATION}", str(stage), credentials=upload["credentials"])

    assert local == str(stage / ITER)
    assert store.heads[:3] == [MANIFEST_KEY, ARCHIVE_KEY, MODEL_KEY], "commit and inventory precede any payload"
    assert (stage / ITER / "model.pt").read_bytes() == WEIGHTS
    assert (stage / ITER / CHECKPOINT_ARTIFACT_MANIFEST).read_bytes() == store.objects[MANIFEST_KEY][0]
    assert read_local_artifact_manifest(local)["components"]["model"]["sha256"] == _sha(WEIGHTS)
    record = json.loads((stage / ITER / STAGED_RECORD).read_text(encoding="utf-8"))
    manifest_sha256 = _sha(store.objects[MANIFEST_KEY][0])
    assert record == {
        "protocol": STAGED_PROTOCOL,
        "source": IDENTITY,
        "manifest_sha256": manifest_sha256,
        "files": {
            "model.pt": {"size": len(WEIGHTS), "sha256": _sha(WEIGHTS)},
            CHECKPOINT_ARTIFACT_MANIFEST: {"size": len(store.objects[MANIFEST_KEY][0]), "sha256": manifest_sha256},
        },
        "complete": True,
    }
    assert not list((stage / ITER).glob("*.part")) and not list((stage / ITER).glob("*.tmp"))

    # The own copy is reused: only the manifest is re-read, the payload is proven locally by digest.
    store.heads.clear()
    assert (
        fetch_archived_artifact(f"s3://archive/{DESTINATION}", str(stage), credentials=upload["credentials"]) == local
    )
    assert store.heads == [MANIFEST_KEY, ARCHIVE_KEY]
    # A damaged own copy is re-downloaded, not trusted.
    (stage / ITER / "model.pt").write_bytes(STRANGER)
    store.heads.clear()
    fetch_archived_artifact(f"s3://archive/{DESTINATION}", str(stage), credentials=upload["credentials"])
    assert store.heads == [MANIFEST_KEY, ARCHIVE_KEY, MODEL_KEY] and (stage / ITER / "model.pt").read_bytes() == WEIGHTS

    with pytest.raises(ValueError):
        fetch_archived_artifact(f"s3://archive/{RUN}/checkpoint_artifacts", str(stage))
    with pytest.raises(ValueError):
        fetch_archived_artifact(f"/archive/{DESTINATION}", str(stage))
    with pytest.raises(RuntimeError, match="no readable manifest"):
        fetch_archived_artifact(
            f"s3://archive/{RUN}/checkpoint_artifacts/iter_000999", str(stage), credentials=upload["credentials"]
        )


@pytest.mark.L0
@pytest.mark.CPU
def test_fetch_refuses_a_foreign_copy_and_replaces_only_its_own_stale_files(
    tmp_path: Path, store: FakeS3Client
) -> None:
    artifact_dir = _artifact(tmp_path)
    upload = _upload(tmp_path)
    _publish(str(artifact_dir), upload)
    uri = f"s3://archive/{DESTINATION}"
    stage = tmp_path / "stage"
    local = stage / ITER
    local.mkdir(parents=True)

    # No record: the copy was not proven staged by this tool and is never reused, replaced, or deleted.
    (local / "model.pt").write_bytes(WEIGHTS)
    with pytest.raises(RuntimeError, match="not stage"):
        fetch_archived_artifact(uri, str(stage), credentials=upload["credentials"])
    assert (local / "model.pt").read_bytes() == WEIGHTS and sorted(os.listdir(local)) == ["model.pt"]
    for garbage in ("{}", "[]", "not json", json.dumps({"protocol": "other", "bucket": "archive", "key": MODEL_KEY})):
        (local / STAGED_RECORD).write_text(garbage, encoding="utf-8")
        with pytest.raises(RuntimeError, match="not stage"):
            fetch_archived_artifact(uri, str(stage), credentials=upload["credentials"])
        assert (local / "model.pt").read_bytes() == WEIGHTS and (local / STAGED_RECORD).read_text() == garbage

    # This tool's record naming ANOTHER source is stale: its listed files go, a caller's files survive.
    (local / STAGED_RECORD).unlink()
    (local / "model.pt").unlink()
    fetch_archived_artifact(uri, str(stage), credentials=upload["credentials"])
    record = json.loads((local / STAGED_RECORD).read_text(encoding="utf-8"))
    (local / "sentinel.txt").write_text("caller-owned", encoding="utf-8")
    (local / "model.pt").write_bytes(STRANGER)
    (local / "model.pt.part").write_bytes(b"partial")
    stale = {**record, "source": {**IDENTITY, "prefix": f"{RUN}/checkpoint_artifacts/iter_000999"}}
    (local / STAGED_RECORD).write_text(json.dumps(stale), encoding="utf-8")
    fetch_archived_artifact(uri, str(stage), credentials=upload["credentials"])
    assert (local / "model.pt").read_bytes() == WEIGHTS and not (local / "model.pt.part").exists()
    assert (local / "sentinel.txt").read_text(encoding="utf-8") == "caller-owned"
    assert json.loads((local / STAGED_RECORD).read_text(encoding="utf-8")) == record

    # An interrupted fetch (claim written, payload missing) resumes under its own claim.
    (local / "model.pt").unlink()
    (local / STAGED_RECORD).write_text(json.dumps({**record, "complete": False}), encoding="utf-8")
    fetch_archived_artifact(uri, str(stage), credentials=upload["credentials"])
    assert (local / "model.pt").read_bytes() == WEIGHTS
    assert json.loads((local / STAGED_RECORD).read_text(encoding="utf-8"))["complete"] is True


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("suffix", ["", ".part"])
def test_fetch_source_replacement_refuses_foreign_payload_and_temporary_paths(
    tmp_path: Path, store: FakeS3Client, suffix: str
) -> None:
    original = _artifact(tmp_path / "original")
    upload = _upload(tmp_path)
    _publish(str(original), upload)
    stage = tmp_path / "stage"
    local = Path(fetch_archived_artifact(f"s3://archive/{DESTINATION}", str(stage), credentials=upload["credentials"]))

    successor = _artifact(tmp_path / "successor", body=STRANGER)
    optimizer = b"replacement optimizer"
    (successor / "optim.pt").write_bytes(optimizer)
    components = read_local_artifact_manifest(str(successor))["components"]
    components["optim"] = {"kind": "file", "path": "optim.pt", "size": len(optimizer), "sha256": _sha(optimizer)}
    atomic_write_json(
        str(successor / CHECKPOINT_ARTIFACT_MANIFEST),
        build_artifact_manifest(str(successor), "consolidated", ("model", "optimizer"), components),
    )
    second_upload = {**upload, "prefix": "another/run/checkpoints"}
    report = _publish(str(successor), second_upload)

    # The old record owns model.pt and its interrupted temporary, but never claimed this optimizer path.
    (local / "model.pt.part").write_bytes(b"owned interrupted download")
    foreign = local / ("optim.pt" + suffix)
    foreign.write_bytes(b"caller-owned optimizer")
    before = {path.name: path.read_bytes() for path in local.iterdir()}
    with pytest.raises(RuntimeError, match="not owned by the prior staging record"):
        fetch_archived_artifact(f"s3://archive/{report['prefix']}", str(stage), credentials=upload["credentials"])
    assert {path.name: path.read_bytes() for path in local.iterdir()} == before, "refusal precedes all staging mutation"

    foreign.unlink()
    fetch_archived_artifact(f"s3://archive/{report['prefix']}", str(stage), credentials=upload["credentials"])
    assert (local / "model.pt").read_bytes() == STRANGER
    assert (local / "optim.pt").read_bytes() == optimizer
    assert not (local / "model.pt.part").exists(), "the old owned temporary remains eligible for cleanup"
    record = json.loads((local / STAGED_RECORD).read_text(encoding="utf-8"))
    assert record["complete"] is True and record["source"]["prefix"] == report["prefix"]


@pytest.mark.L0
@pytest.mark.CPU
def test_fetch_closes_manifest_and_inventory_response_streams(
    tmp_path: Path, store: FakeS3Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_dir = _artifact(tmp_path)
    upload = _upload(tmp_path)
    _publish(str(artifact_dir), upload)
    original_get = store.get_object
    streams: list[Any] = []

    def track_streams(*, Bucket: str, Key: str, **kwargs: Any) -> dict[str, Any]:
        response = original_get(Bucket=Bucket, Key=Key, **kwargs)
        if Key in (MANIFEST_KEY, ARCHIVE_KEY):
            streams.append(response["Body"])
        return response

    monkeypatch.setattr(store, "get_object", track_streams)
    fetch_archived_artifact(f"s3://archive/{DESTINATION}", str(tmp_path / "stage"), credentials=upload["credentials"])
    assert len(streams) == 2 and all(stream.closed for stream in streams)


@pytest.mark.L0
@pytest.mark.CPU
def test_fetch_rejects_a_corrupt_payload_or_manifest_and_leaves_no_partial_copy(
    tmp_path: Path, store: FakeS3Client
) -> None:
    artifact_dir = _artifact(tmp_path)
    upload = _upload(tmp_path)
    _publish(str(artifact_dir), upload)
    uri = f"s3://archive/{DESTINATION}"
    stage = tmp_path / "stage"

    # Same size, echoed digest metadata: invisible to HEAD, caught by hashing the downloaded bytes.
    store.plant(MODEL_KEY, STRANGER, {"sha256": _sha(WEIGHTS)})
    with pytest.raises(RuntimeError, match="sha256"):
        fetch_archived_artifact(uri, str(stage), credentials=upload["credentials"])
    assert not (stage / ITER / "model.pt").exists() and not (stage / ITER / "model.pt.part").exists()
    assert not (stage / ITER / CHECKPOINT_ARTIFACT_MANIFEST).exists(), "no manifest before the payloads are proven"
    record = json.loads((stage / ITER / STAGED_RECORD).read_text(encoding="utf-8"))
    assert record["complete"] is False, "the claim stays so the retry resumes under its own record"

    store.plant(MODEL_KEY, WEIGHTS[:-1], {"sha256": _sha(WEIGHTS)})
    with pytest.raises(RuntimeError, match="bytes"):
        fetch_archived_artifact(uri, str(stage), credentials=upload["credentials"])
    store.plant(MODEL_KEY, WEIGHTS, {"sha256": _sha(WEIGHTS)})
    fetch_archived_artifact(uri, str(stage), credentials=upload["credentials"])
    assert (stage / ITER / "model.pt").read_bytes() == WEIGHTS

    # A remote manifest that is not a valid native manifest, or whose digest metadata lies, is refused.
    good_manifest = store.objects[MANIFEST_KEY]
    store.plant(MANIFEST_KEY, json.dumps({"protocol": "x"}).encode("utf-8"), {"sha256": "0" * 64})
    with pytest.raises(RuntimeError, match="unsupported posttrain checkpoint artifact manifest"):
        fetch_archived_artifact(uri, str(tmp_path / "stage2"), credentials=upload["credentials"])
    store.plant(MANIFEST_KEY, good_manifest[0], {"sha256": _sha(STRANGER)})
    with pytest.raises(RuntimeError, match="corrupt store"):
        fetch_archived_artifact(uri, str(tmp_path / "stage2"), credentials=upload["credentials"])
    assert not (tmp_path / "stage2").exists(), "nothing is created before the manifest is proven"


@pytest.mark.L0
@pytest.mark.CPU
def test_discard_staged_artifact_removes_only_its_own_files(tmp_path: Path, store: FakeS3Client) -> None:
    artifact_dir = _artifact(tmp_path)
    upload = _upload(tmp_path)
    _publish(str(artifact_dir), upload)
    stage = tmp_path / "stage"
    local = Path(fetch_archived_artifact(f"s3://archive/{DESTINATION}", str(stage), credentials=upload["credentials"]))
    (local / "sentinel.txt").write_text("caller-owned", encoding="utf-8")
    (local / "model.pt.part").write_bytes(b"partial")

    removed = discard_staged_artifact(str(local))

    assert sorted(removed) == sorted(
        [
            str(local / "model.pt"),
            str(local / "model.pt.part"),
            str(local / CHECKPOINT_ARTIFACT_MANIFEST),
            str(local / STAGED_RECORD),
        ]
    )
    assert os.listdir(local) == ["sentinel.txt"], "a caller's file survives and keeps the directory"
    with pytest.raises(RuntimeError, match="no .staged.json ownership record"):
        discard_staged_artifact(str(local))
    assert os.listdir(local) == ["sentinel.txt"]

    (local / "sentinel.txt").unlink()
    local.rmdir()
    local = Path(fetch_archived_artifact(f"s3://archive/{DESTINATION}", str(stage), credentials=upload["credentials"]))
    assert discard_staged_artifact(str(local))[-1] == str(local) + os.sep and not local.exists()

    foreign = tmp_path / "foreign" / ITER
    foreign.mkdir(parents=True)
    (foreign / "model.pt").write_bytes(WEIGHTS)
    with pytest.raises(RuntimeError, match="refusing to discard"):
        discard_staged_artifact(str(foreign))
    assert (foreign / "model.pt").is_file()


@pytest.mark.L0
@pytest.mark.CPU
def test_main_subcommands_exit_codes(tmp_path: Path, store: FakeS3Client, capsys: pytest.CaptureFixture[str]) -> None:
    artifact_dir = _artifact(tmp_path)
    credentials = _credentials(tmp_path / "creds.json")
    common = ["--credentials", credentials]

    assert (
        main(
            [
                "publish",
                "--artifact-dir",
                str(artifact_dir),
                "--upload-prefix",
                f"{RUN}/checkpoints",
                "--bucket",
                "archive",
                *common,
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "published" and report["prefix"] == DESTINATION
    assert main(["verify", "--artifact-dir", str(artifact_dir), *common]) == 0
    assert capsys.readouterr().out.strip() == "verified"

    assert (
        main(["fetch", "--artifact-uri", f"s3://archive/{DESTINATION}", "--dst", str(tmp_path / "stage"), *common]) == 0
    )
    assert capsys.readouterr().out.strip() == str(tmp_path / "stage" / ITER)

    assert main(["prune-verified", "--paths", f"{artifact_dir},{tmp_path / 'nowhere'}", *common]) == 0
    prune_report = json.loads(capsys.readouterr().out)
    assert (
        prune_report["removed"] == {str(artifact_dir): ["model.pt"]}
        and str(tmp_path / "nowhere") in prune_report["kept"]
    )

    store.objects.pop(MODEL_KEY)
    assert main(["verify", "--artifact-dir", str(artifact_dir), *common]) == 1
    assert capsys.readouterr().out.strip() == "not verified"

    different = _artifact(tmp_path / "different", body=STRANGER)
    argv = ["publish", "--artifact-dir", str(different), "--upload-prefix", f"{RUN}/checkpoints", "--bucket", "archive"]
    assert main([*argv, *common]) == 1
    assert "refusing to overwrite committed object-store artifact" in capsys.readouterr().err
    assert (
        main(["fetch", "--artifact-uri", "s3://archive/nope/iter_000001", "--dst", str(tmp_path / "s"), *common]) == 1
    )
    assert "no readable manifest" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["publish", "--artifact-dir", str(artifact_dir)])


@pytest.mark.L0
@pytest.mark.CPU
def test_fetch_refuses_non_canonical_shard_keys_before_writing_anything(tmp_path: Path, store: FakeS3Client) -> None:
    """A DCP inventory is store-controlled input: a key with a ``..`` segment must never resolve outside the stage."""
    prefix = f"{RUN}/checkpoint_artifacts/iter_000301"
    manifest = build_artifact_manifest(
        f"s3://archive/{prefix}", "dcp", ("model",), {"model": {"kind": "dcp", "path": "model"}}
    )
    store.plant(f"{prefix}/{CHECKPOINT_ARTIFACT_MANIFEST}", json.dumps(manifest).encode("utf-8"))
    store.plant(f"{prefix}/model/.metadata", b"meta")
    store.plant(f"{prefix}/model/../../../escape.bin", b"hostile")
    _plant_inventory(
        store,
        prefix,
        json.dumps(manifest).encode("utf-8"),
        {"model/.metadata": b"meta", "model/../../../escape.bin": b"hostile"},
    )
    caller_file = tmp_path / "escape.bin"
    caller_file.write_bytes(b"caller-owned")
    stage = tmp_path / "stage"

    with pytest.raises(RuntimeError, match="canonical"):
        fetch_archived_artifact(f"s3://archive/{prefix}", str(stage), credentials=_credentials(tmp_path / "c.json"))

    assert caller_file.read_bytes() == b"caller-owned", "a caller's file outside the stage is never replaced"
    assert not (tmp_path / "escape.bin.part").exists()
    assert not stage.exists(), "the inventory is validated before any local path is created"


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("component", ["payload", "manifest"])
def test_publish_refuses_remote_tampering_before_the_commit(
    tmp_path: Path, store: FakeS3Client, component: str
) -> None:
    """Out-of-band remote changes must fail publication without a covering marker or local deletion."""
    artifact_dir = _artifact(tmp_path)
    upload = _upload(tmp_path)
    key = MODEL_KEY if component == "payload" else MANIFEST_KEY
    body = STRANGER if component == "payload" else b'{"other": "writer"}'
    metadata = {"sha256": _sha(body)}

    def tamper_before_commit(put_key: str) -> None:
        if put_key == MANIFEST_KEY:
            store.plant(key, body, metadata)

    store.before_put = tamper_before_commit
    with pytest.raises(RuntimeError, match="another writer touched the prefix"):
        _publish(str(artifact_dir), upload, delete_local=True)

    assert store.objects[key] == (body, metadata), "a foreign manifest is never overwritten"
    assert MANIFEST_KEY in store.objects, "the committed prefix is rejected by the final verification"
    assert (artifact_dir / "model.pt").read_bytes() == WEIGHTS
    assert read_marker(str(artifact_dir)) is None and not (artifact_dir / MARKER_NAME).exists()


@pytest.mark.L0
@pytest.mark.CPU
def test_publish_commit_carries_the_precondition_and_refuses_clients_without_it(
    tmp_path: Path, store: FakeS3Client
) -> None:
    artifact_dir = _artifact(tmp_path)
    upload = _upload(tmp_path)

    _publish(str(artifact_dir), upload)

    manifest_sha256 = _sha(_manifest_bytes(artifact_dir))
    assert store.uploads[-1] == ("archive", MANIFEST_KEY, {"Metadata": {"sha256": manifest_sha256}, "IfNoneMatch": "*"})
    assert store.uploads[-2][2] == {"Metadata": {"sha256": _sha(WEIGHTS)}}, "payloads are ordinary uploads"

    # A botocore without the parameter cannot reserve the destination: no payload or marker may land.
    store.conditional_puts_supported = False
    other = _artifact(tmp_path / "other", name="iter_000301")
    other_prefix = f"{RUN}/checkpoint_artifacts/iter_000301"
    uploads = list(store.uploads)
    with pytest.raises(RuntimeError, match="conditional object creation is required"):
        _publish(str(other), _upload(tmp_path / "other"))
    assert store.uploads == uploads
    assert not any(key.startswith(other_prefix + "/") for key in store.objects)
    assert not (other / MARKER_NAME).exists()
    assert (other / "model.pt").read_bytes() == WEIGHTS


@pytest.mark.L0
@pytest.mark.CPU
def test_fetch_proves_dcp_shards_by_digest_and_refuses_an_archive_with_an_undigested_shard(
    tmp_path: Path, store: FakeS3Client
) -> None:
    """OLD: shards discovered from the listing carried no expected digest and were staged on size alone."""
    prefix = f"{RUN}/checkpoint_artifacts/iter_000301"
    uri = f"s3://archive/{prefix}"
    manifest = build_artifact_manifest(uri, "dcp", ("model",), {"model": {"kind": "dcp", "path": "model"}})
    store.plant(f"{prefix}/{CHECKPOINT_ARTIFACT_MANIFEST}", json.dumps(manifest).encode("utf-8"))
    meta, shard = b"dcp-metadata", b"shard-0" * 1024
    shard_key = f"{prefix}/model/__0_0.distcp"
    store.plant(f"{prefix}/model/.metadata", meta, {"sha256": _sha(meta)})
    store.plant(shard_key, shard, {"sha256": _sha(shard)})
    _plant_inventory(
        store,
        prefix,
        json.dumps(manifest).encode("utf-8"),
        {"model/.metadata": meta, "model/__0_0.distcp": shard},
    )
    credentials = _credentials(tmp_path / "creds.json")

    local = Path(fetch_archived_artifact(uri, str(tmp_path / "stage"), credentials=credentials))

    assert (local / "model" / ".metadata").read_bytes() == meta and (
        local / "model" / "__0_0.distcp"
    ).read_bytes() == shard
    record = json.loads((local / STAGED_RECORD).read_text(encoding="utf-8"))
    assert record["complete"] is True
    assert record["files"]["model/__0_0.distcp"] == {"size": len(shard), "sha256": _sha(shard)}
    assert record["files"]["model/.metadata"] == {"size": len(meta), "sha256": _sha(meta)}
    # The own copy is reused only while its bytes still match the recorded digest.
    (local / "model" / "__0_0.distcp").write_bytes(b"SHARD-0" * 1024)
    fetch_archived_artifact(uri, str(tmp_path / "stage"), credentials=credentials)
    assert (local / "model" / "__0_0.distcp").read_bytes() == shard

    # Same size, other bytes, metadata echoing the good digest: invisible to HEAD, caught by hashing the download.
    store.plant(shard_key, b"SHARD-0" * 1024, {"sha256": _sha(shard)})
    with pytest.raises(RuntimeError, match=r"__0_0\.distcp: downloaded bytes have sha256"):
        fetch_archived_artifact(uri, str(tmp_path / "stage2"), credentials=credentials)
    assert not (tmp_path / "stage2" / "iter_000301" / "model" / "__0_0.distcp").exists()
    assert not list((tmp_path / "stage2").rglob("*.part")), "no partial file survives"
    # The committed inventory, rather than mutable object metadata, defines the expected digest.
    store.plant(shard_key, shard, {"sha256": _sha(b"SHARD-0" * 1024)})
    fetch_archived_artifact(uri, str(tmp_path / "stage2"), credentials=credentials)

    # An archive without a shard digest in its declaration is unverifiable: nothing is staged.
    store.plant(shard_key, shard)
    declaration_key = f"{prefix}/{ARCHIVE_RECORD}"
    valid_declaration = store.objects[declaration_key]
    declaration = json.loads(valid_declaration[0])
    for invalid_digest in (None, "not-a-digest"):
        declaration["files"]["model/__0_0.distcp"]["sha256"] = invalid_digest
        store.plant(declaration_key, json.dumps(declaration).encode("utf-8"))
        with pytest.raises(RuntimeError, match="inventory"):
            fetch_archived_artifact(uri, str(tmp_path / "stage3"), credentials=credentials)
    assert not (tmp_path / "stage3").exists()

    # Restoring the declaration stages again, even without metadata on the payload object.
    store.plant(declaration_key, *valid_declaration)
    local = Path(fetch_archived_artifact(uri, str(tmp_path / "stage3"), credentials=credentials))
    assert json.loads((local / STAGED_RECORD).read_text(encoding="utf-8"))["files"]["model/__0_0.distcp"] == {
        "size": len(shard),
        "sha256": _sha(shard),
    }


@pytest.mark.L0
@pytest.mark.CPU
def test_fetch_refuses_a_legacy_file_artifact_before_creating_anything(tmp_path: Path, store: FakeS3Client) -> None:
    """A file component with neither a descriptor digest nor object metadata is unverifiable: no directory, no claim."""
    plain = _artifact(tmp_path / "plain", digest=False)
    store.plant(MODEL_KEY, WEIGHTS)
    store.plant(MANIFEST_KEY, _manifest_bytes(plain))
    credentials = _credentials(tmp_path / "creds.json")
    stage = tmp_path / "stage"

    with pytest.raises(RuntimeError, match="no sha256 metadata") as refusal:
        fetch_archived_artifact(f"s3://archive/{DESTINATION}", str(stage), credentials=credentials)
    assert f"s3://archive/{MODEL_KEY}" in str(refusal.value)
    assert not stage.exists(), "nothing is created when any object is unverifiable"

    # Object metadata supplies the digest when the legacy descriptor has none.
    store.plant(MODEL_KEY, WEIGHTS, {"sha256": _sha(WEIGHTS)})
    local = Path(fetch_archived_artifact(f"s3://archive/{DESTINATION}", str(stage), credentials=credentials))
    assert (local / "model.pt").read_bytes() == WEIGHTS
    assert json.loads((local / STAGED_RECORD).read_text())["files"]["model.pt"]["sha256"] == _sha(WEIGHTS)

    # A descriptor digest vouches on its own; the object then needs no metadata...
    digested = _artifact(tmp_path / "digested", name="iter_000301")
    other_prefix = f"{RUN}/checkpoint_artifacts/iter_000301"
    store.plant(f"{other_prefix}/model.pt", WEIGHTS)
    store.plant(f"{other_prefix}/{CHECKPOINT_ARTIFACT_MANIFEST}", _manifest_bytes(digested))
    local = Path(fetch_archived_artifact(f"s3://archive/{other_prefix}", str(stage), credentials=credentials))
    assert (local / "model.pt").read_bytes() == WEIGHTS
    record = json.loads((local / STAGED_RECORD).read_text(encoding="utf-8"))
    assert record["files"]["model.pt"]["sha256"] == _sha(WEIGHTS)
    # ...and same-size other bytes under that descriptor are refused after the download, never at the final path.
    store.plant(f"{other_prefix}/model.pt", STRANGER)
    with pytest.raises(RuntimeError, match="downloaded bytes have sha256"):
        fetch_archived_artifact(f"s3://archive/{other_prefix}", str(tmp_path / "stage2"), credentials=credentials)
    assert not (tmp_path / "stage2" / "iter_000301" / "model.pt").exists()
    assert not list((tmp_path / "stage2").rglob("*.part"))


@pytest.mark.L0
@pytest.mark.CPU
def test_fetch_requires_every_committed_dcp_shard(
    tmp_path: Path, store: FakeS3Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing remote shard must not disappear from the expected inventory or expose a complete stage."""
    artifact_dir = tmp_path / "checkpoint_artifacts" / ITER
    weights = torch.arange(8)  # [8]
    dcp.save({"weight": weights}, checkpoint_id=str(artifact_dir / "model"))
    commit_local_artifact(str(artifact_dir), "dcp", ("model",), ["model"], {})
    (artifact_dir / "notes.txt").write_text("diagnostic only", encoding="utf-8")
    upload = _upload(tmp_path)
    _publish(str(artifact_dir), upload)
    shard_key = next(key for key in store.objects if key.endswith(".distcp"))
    metadata_key = f"{DESTINATION}/model/.metadata"

    def unexpected_listing(**_kwargs: object) -> None:
        pytest.fail("fetch must use the committed DCP inventory, never a live listing")

    monkeypatch.setattr(store, "list_objects_v2", unexpected_listing)
    uri = f"s3://archive/{DESTINATION}"
    local = Path(fetch_archived_artifact(uri, str(tmp_path / "complete"), credentials=upload["credentials"]))
    loaded = {"weight": torch.empty_like(weights)}  # weight: [8]
    dcp.load(loaded, checkpoint_id=str(local / "model"))
    assert torch.equal(loaded["weight"], weights)
    assert not (local / "notes.txt").exists(), "fetch stages only the native physical components"

    for index, key in enumerate((metadata_key, shard_key)):
        archived = store.objects.pop(key)
        stage = tmp_path / f"missing_{index}"
        with pytest.raises(RuntimeError, match="cannot download required object"):
            fetch_archived_artifact(uri, str(stage), credentials=upload["credentials"])
        record = json.loads((stage / ITER / STAGED_RECORD).read_text(encoding="utf-8"))
        assert record["complete"] is False
        assert not (stage / ITER / CHECKPOINT_ARTIFACT_MANIFEST).exists()
        assert not list(stage.rglob("*.part"))
        store.plant(key, *archived)

    # Legacy DCP has no authoritative inventory even when every listed object carries metadata.
    store.objects.pop(ARCHIVE_KEY)
    stage = tmp_path / "no_inventory"
    with pytest.raises(RuntimeError, match="no committed shard inventory"):
        fetch_archived_artifact(uri, str(stage), credentials=upload["credentials"])
    assert not stage.exists()


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("changed_entry", ["missing_component", "component_size", "component_digest", "manifest_size"])
def test_fetch_refuses_a_declaration_that_disagrees_with_the_native_manifest(
    tmp_path: Path, store: FakeS3Client, changed_entry: str
) -> None:
    artifact_dir = _artifact(tmp_path)
    upload = _upload(tmp_path)
    _publish(str(artifact_dir), upload)
    declaration = json.loads(store.objects[ARCHIVE_KEY][0])
    if changed_entry == "missing_component":
        del declaration["files"]["model.pt"]
    elif changed_entry == "component_size":
        declaration["files"]["model.pt"]["size"] += 1
    elif changed_entry == "component_digest":
        declaration["files"]["model.pt"]["sha256"] = _sha(STRANGER)
    else:
        declaration["files"][CHECKPOINT_ARTIFACT_MANIFEST]["size"] += 1
    store.plant(ARCHIVE_KEY, json.dumps(declaration).encode("utf-8"))
    stage = tmp_path / "stage"
    with pytest.raises(RuntimeError, match="inventory"):
        fetch_archived_artifact(f"s3://archive/{DESTINATION}", str(stage), credentials=upload["credentials"])
    assert not stage.exists()


@pytest.mark.L0
@pytest.mark.CPU
def test_main_publish_delete_local_uses_verified_retention(
    tmp_path: Path, store: FakeS3Client, capsys: pytest.CaptureFixture[str]
) -> None:
    artifact_dir = _artifact(tmp_path)
    upload = _upload(tmp_path)
    assert (
        main(
            [
                "publish",
                "--artifact-dir",
                str(artifact_dir),
                "--upload-prefix",
                upload["prefix"],
                "--bucket",
                upload["bucket"],
                "--credentials",
                upload["credentials"],
                "--delete-local",
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["removed"] == ["model.pt"]
    assert not (artifact_dir / "model.pt").exists()
    assert (artifact_dir / CHECKPOINT_ARTIFACT_MANIFEST).is_file() and (artifact_dir / MARKER_NAME).is_file()
    assert marker_covers(str(artifact_dir), IDENTITY), "the inventory still covers what remains"
    assert verify_archived_artifact(str(artifact_dir), credentials=upload["credentials"])
    assert store.objects[MODEL_KEY][0] == WEIGHTS
    # A released payload cannot be re-published: the manifest still names it, the directory no longer holds it.
    with pytest.raises(RuntimeError, match="missing or resized"):
        _publish(str(artifact_dir), upload)


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("identical", [False, True])
def test_a_delayed_publisher_cannot_replace_another_inventory(
    tmp_path: Path, store: FakeS3Client, monkeypatch: pytest.MonkeyPatch, identical: bool
) -> None:
    """A delayed payload either shares the winner's exact bytes or excludes it before upload."""
    slow = _artifact(tmp_path / "slow")
    other = _artifact(tmp_path / "other", body=WEIGHTS if identical else STRANGER)
    upload = _upload(tmp_path)
    started = threading.Event()
    release = threading.Event()
    original_upload = store.upload_file

    def delayed_upload(Filename: str, Bucket: str, Key: str, **kwargs: Any) -> None:
        if Filename == str(slow / "model.pt"):
            started.set()
            if not release.wait(timeout=10):
                raise TimeoutError("test did not release the delayed payload")
        original_upload(Filename, Bucket, Key, **kwargs)

    monkeypatch.setattr(store, "upload_file", delayed_upload)
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(_publish, str(slow), upload, delete_local=True)
        try:
            assert started.wait(timeout=10), "the first writer must reserve the prefix before its payload pauses"
            if identical:
                report = _publish(str(other), upload, delete_local=True)
                assert report["status"] == "published" and not (other / "model.pt").exists()
            else:
                with pytest.raises(FileExistsError, match="reserved object-store artifact"):
                    _publish(str(other), upload, delete_local=True)
                assert MODEL_KEY not in store.objects, "the conflicting writer never reaches payload upload"
                assert (other / "model.pt").read_bytes() == STRANGER
        finally:
            release.set()
        report = pending.result(timeout=10)
    assert report["removed"] == ["model.pt"] and not (slow / "model.pt").exists()
    assert store.objects[MODEL_KEY][0] == WEIGHTS
    assert json.loads(store.objects[MANIFEST_KEY][0])["components"]["model"]["sha256"] == _sha(WEIGHTS)
    assert verify_archived_artifact(str(slow), credentials=upload["credentials"])


@pytest.mark.L0
@pytest.mark.CPU
def test_publication_retries_keep_the_inventory_reserved_after_a_failed_upload(
    tmp_path: Path, store: FakeS3Client
) -> None:
    artifact = _artifact(tmp_path)
    upload = _upload(tmp_path)

    def fail_payload(key: str) -> None:
        if key == MODEL_KEY:
            raise RuntimeError("interrupted payload upload")

    store.before_put = fail_payload
    with pytest.raises(RuntimeError, match="interrupted payload upload"):
        publish_artifact(str(artifact), upload)
    assert ARCHIVE_KEY in store.objects and MANIFEST_KEY not in store.objects
    store.before_put = None
    other = _artifact(tmp_path / "other", body=STRANGER)
    with pytest.raises(FileExistsError, match="reserved object-store artifact"):
        publish_artifact(str(other), upload)
    assert publish_artifact(str(artifact), upload)["status"] == "published"
    assert store.objects[MODEL_KEY][0] == WEIGHTS


@pytest.mark.L0
@pytest.mark.CPU
def test_publication_covers_recorded_merge_diagnostics_without_archiving_stale_components(
    tmp_path: Path, store: FakeS3Client
) -> None:
    artifact = _artifact(tmp_path)
    diagnostic = artifact / "diagnostics.json"
    diagnostic.write_text('{"mean_delta": 0.5}')
    (artifact / ".merge-complete.json").write_text(
        json.dumps({"artifacts": {"diag": {"path": str(diagnostic), "bytes": diagnostic.stat().st_size}}})
    )
    stale = artifact / "optim.pt"
    stale.write_bytes(b"stale optimizer")
    upload = _upload(tmp_path)
    report = publish_artifact(str(artifact), upload)
    assert "diagnostics.json" in report["files"] and ".merge-complete.json" in report["files"]
    assert "optim.pt" not in report["files"] and not marker_covers(str(artifact), IDENTITY)
    assert str(artifact) in prune_verified_artifacts([str(artifact)], credentials=upload["credentials"])["kept"]
    stale.unlink()
    assert prune_verified_artifacts([str(artifact)], credentials=upload["credentials"])["removed"] == {
        str(artifact): ["model.pt"]
    }
    assert diagnostic.is_file(), "retention removes payloads, preserving local diagnostics"


@pytest.mark.L0
@pytest.mark.CPU
def test_release_page_cache_advises_the_file_and_never_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``posix_fadvise(DONTNEED)`` over the whole file, fsync first when asked; every failure is swallowed."""
    import errno

    from cosmos_framework.posttrain.checkpoint import io as io_module

    path = tmp_path / "model.pt"
    path.write_bytes(WEIGHTS)
    # Real call against the host kernel: Linux advises, a platform without the call reports False.
    assert io_module.release_page_cache(str(path), sync=True) is (getattr(os, "posix_fadvise", None) is not None)

    calls: list[tuple[int, int, int, int]] = []

    def _fadvise(fd: int, offset: int, length: int, advice: int) -> None:
        calls.append((os.fstat(fd).st_size, offset, length, advice))

    monkeypatch.setattr(os, "posix_fadvise", _fadvise, raising=False)
    monkeypatch.setattr(os, "POSIX_FADV_DONTNEED", 4, raising=False)
    assert io_module.release_page_cache(str(path)) is True
    assert io_module.release_page_cache(str(path), sync=True) is True
    assert calls == [(len(WEIGHTS), 0, 0, 4)] * 2

    def _unsupported(*_args: object) -> None:
        raise OSError(errno.EINVAL, "advice not supported")

    monkeypatch.setattr(os, "posix_fadvise", _unsupported)
    assert io_module.release_page_cache(str(path)) is False
    assert io_module.release_page_cache(str(tmp_path / "missing.pt")) is False
    monkeypatch.delattr(os, "posix_fadvise")
    assert io_module.release_page_cache(str(path)) is False


@pytest.mark.L0
@pytest.mark.CPU
def test_commit_local_artifact_releases_every_written_payload_from_the_page_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cosmos_framework.posttrain.checkpoint import artifact as artifact_module

    released: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        artifact_module,
        "release_page_cache",
        lambda path, *, sync=False: released.append((os.path.basename(path), sync)) or True,
    )
    location = str(tmp_path / "run" / "checkpoint_artifacts" / ITER)

    commit_local_artifact(
        location,
        "consolidated",
        ("model", "optimizer"),
        [],
        {"model": {"weight": torch.ones(2)}, "optim": {"step": torch.zeros(1)}, "trainer": None},  # [2], [1]
    )

    assert released == [("model.pt", True), ("optim.pt", True)], "flush+drop once per payload file, after its hash"
    assert set(read_local_artifact_manifest(location)["components"]) == {"model", "optim"}


@pytest.mark.L0
@pytest.mark.CPU
def test_publish_artifact_releases_uploaded_payloads_after_verification(
    tmp_path: Path, store: FakeS3Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cosmos_framework.posttrain.checkpoint.archive import publisher as publisher_module

    artifact_dir = _artifact(tmp_path)
    released: list[str] = []

    def _release(path: str, **_kwargs: object) -> bool:
        released.append(os.path.relpath(path, str(artifact_dir)))
        assert read_marker(str(artifact_dir)) is None, "cache release must precede the marker, after verification"
        return True

    monkeypatch.setattr(publisher_module, "release_page_cache", _release)

    report = publish_artifact(str(artifact_dir), _upload(tmp_path))

    assert report["status"] == "published"
    assert released == sorted(["model.pt", CHECKPOINT_ARTIFACT_MANIFEST])
    assert [key for _bucket, key, _extra in store.uploads] == [ARCHIVE_KEY, MODEL_KEY, MANIFEST_KEY]
    assert read_marker(str(artifact_dir)) is not None


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("failure", ["reservation", "upload", "verification"])
def test_publish_artifact_releases_payloads_even_when_the_publication_fails(
    tmp_path: Path, store: FakeS3Client, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """A refused reservation, an interrupted upload or a failed remote verification keeps the local
    artifact (the manager treats archive failure as best effort), and the just-hashed payload must not
    then sit in the job cgroup's page cache through the final drain: the release runs from a ``finally``
    once the inventory is known, so no exception on the way to the marker can skip it."""
    from cosmos_framework.posttrain.checkpoint.archive import publisher as publisher_module

    artifact_dir = _artifact(tmp_path)
    released: list[str] = []

    def _release(path: str, **_kwargs: object) -> bool:
        released.append(os.path.relpath(path, str(artifact_dir)))
        return True

    monkeypatch.setattr(publisher_module, "release_page_cache", _release)
    if failure == "reservation":
        _plant_inventory(store, DESTINATION, _manifest_bytes(artifact_dir), {"model.pt": STRANGER})
        expected: tuple[type[Exception], str] = (FileExistsError, "reserved object-store artifact")
    elif failure == "upload":

        def fail_payload(key: str) -> None:
            if key == MODEL_KEY:
                raise RuntimeError("interrupted payload upload")

        store.before_put = fail_payload
        expected = (RuntimeError, "interrupted payload upload")
    else:
        store.corrupt_uploads = True
        expected = (RuntimeError, "SHA-256 verification")

    with pytest.raises(expected[0], match=expected[1]):
        publish_artifact(str(artifact_dir), _upload(tmp_path))

    assert released == sorted(["model.pt", CHECKPOINT_ARTIFACT_MANIFEST])
    assert read_marker(str(artifact_dir)) is None, "no marker: the publication did not complete"
    assert (artifact_dir / "model.pt").read_bytes() == WEIGHTS, "local artifact retained"
    assert MANIFEST_KEY not in store.objects, "the commit was never written"
