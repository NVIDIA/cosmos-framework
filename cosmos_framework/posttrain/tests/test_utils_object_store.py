# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""CPU tests for the GCS-safe object-store helpers (utils.object_store).

Network-free: covers the pure parts — the GCS-endpoint config (the three interop
workarounds), directory-marker filtering, url parsing, destination-key mapping, and the
digest-bound upload path on an in-memory client."""

from __future__ import annotations

import hashlib
import io
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError

import pytest

pytest.importorskip("botocore")

from cosmos_framework.posttrain.checkpoint.archive.marker import MARKER_NAME, write_marker
from cosmos_framework.posttrain.tests.helpers.object_store_fake import FakeS3Client
from cosmos_framework.posttrain.utils import object_store
from cosmos_framework.posttrain.utils.object_store import (
    _parse_s3_url,
    build_s3_config,
    file_sha256,
    list_prefix,
    put_object_write_once,
)


@pytest.mark.L0
@pytest.mark.CPU
def test_file_sha256_known_answer(tmp_path: Path) -> None:
    path = tmp_path / "zeros.bin"
    path.write_bytes(b"\0" * 10)

    assert file_sha256(str(path)) == "01d448afd928065458cf670b60f5a594d735af0172c8d67f22a81680132681ca"
    assert object_store._file_sha256 is file_sha256, "the private spelling stays an alias"
    (tmp_path / "empty.bin").write_bytes(b"")
    assert file_sha256(str(tmp_path / "empty.bin")) == hashlib.sha256(b"").hexdigest()


@pytest.mark.L0
@pytest.mark.CPU
def test_gcs_endpoint_gets_interop_config():
    cfg = build_s3_config("https://storage.googleapis.com")
    assert cfg.signature_version == "s3v4"
    assert cfg.s3["addressing_style"] == "path" and cfg.s3["payload_signing_enabled"] is True
    # boto3>=1.36 CRC32 default disabled (attribute exists on modern botocore)
    if hasattr(cfg, "request_checksum_calculation"):
        assert cfg.request_checksum_calculation == "when_required"


@pytest.mark.L0
@pytest.mark.CPU
def test_non_gcs_endpoint_untouched():
    cfg = build_s3_config("https://pdx.s8k.io")
    assert cfg.signature_version is None  # stock signing for real S3-compatible stores


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("endpoint", ["https://storage.googleapis.com", "https://s3.us-east-1.amazonaws.com"])
def test_conditional_creation_is_signed_for_the_actual_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, endpoint: str
) -> None:
    """Capture a real botocore request before network I/O, including its signed precondition.

    Both stores receive the plain ``If-None-Match: *`` header. GCS's S3-interoperability endpoint
    rejects any ``x-goog-*`` header sent beside AWS-signed ``x-amz-*`` headers with
    ``ExcessHeaderValues``; translating the precondition into ``x-goog-if-generation-match`` made
    every archive PutObject fail (regression: a84c51b114), so no ``x-goog-*`` header may appear.
    """
    credentials = tmp_path / "credentials.json"
    credentials.write_text(
        json.dumps(
            {
                "endpoint_url": endpoint,
                "region_name": "us-east-1",
                "aws_access_key_id": "test-key",
                "aws_secret_access_key": "test-secret",
            }
        )
    )
    client = object_store.build_s3_client(str(credentials))
    headers: dict[str, str] = {}

    class RequestCaptured(Exception):
        pass

    def capture(request: Any) -> None:
        headers.update(
            {
                name.lower(): value.decode() if isinstance(value, bytes) else value
                for name, value in request.headers.items()
            }
        )
        raise RequestCaptured

    monkeypatch.setattr(client._endpoint.http_session, "send", capture)
    with pytest.raises(RequestCaptured):
        client.put_object(Bucket="archive", Key="inventory.json", Body=b"{}", IfNoneMatch="*")
    assert headers["if-none-match"] == "*"
    assert "if-none-match" in headers["authorization"]
    assert not [name for name in headers if name.startswith("x-goog-")], headers


@pytest.mark.L0
@pytest.mark.CPU
def test_gcs_atomic_create_has_a_pinned_goog4_hmac_request() -> None:
    request = object_store._gcs_v4_request(
        {
            "endpoint_url": "https://storage.googleapis.com",
            "region_name": "us-east4",
            "aws_access_key_id": "test-key",
            "aws_secret_access_key": "test-secret",
        },
        "archive",
        "run/inventory name.json",
        b"{}",
        {"sha256": "abc", "publication-id": "p1"},
        now=datetime(2026, 9, 17, 1, 2, 3, tzinfo=timezone.utc),
    )
    headers = {name.lower(): value for name, value in request.header_items()}

    assert request.get_method() == "PUT" and request.data == b"{}"
    assert request.full_url == "https://storage.googleapis.com/archive/run/inventory%20name.json"
    assert headers["x-goog-if-generation-match"] == "0"
    assert headers["x-goog-content-sha256"] == hashlib.sha256(b"{}").hexdigest()
    assert headers["x-goog-meta-sha256"] == "abc"
    assert headers["x-goog-meta-publication-id"] == "p1"
    assert not [name for name in headers if name.startswith("x-amz-")]
    assert headers["authorization"] == (
        "GOOG4-HMAC-SHA256 Credential=test-key/20260917/us-east4/storage/goog4_request, "
        "SignedHeaders=host;x-goog-content-sha256;x-goog-date;x-goog-if-generation-match;"
        "x-goog-meta-publication-id;x-goog-meta-sha256, "
        "Signature=6a58b4a1d6d6d581ee2415d575c9c33088a5f9a85716f1f2a833638516f12d60"
    )
    assert object_store.response_metadata(
        {
            "Metadata": {"from-model": "kept"},
            "ResponseMetadata": {
                "HTTPHeaders": {
                    "x-goog-meta-sha256": "abc",
                    "x-amz-meta-publication-id": "p1",
                }
            },
        }
    ) == {"from-model": "kept", "sha256": "abc", "publication-id": "p1"}


@pytest.mark.L0
@pytest.mark.CPU
def test_gcs_atomic_create_linearizes_two_writers_after_both_heads_miss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    credentials = tmp_path / "credentials.json"
    credentials.write_text(
        json.dumps(
            {
                "endpoint_url": "https://storage.googleapis.com",
                "region_name": "us-east4",
                "aws_access_key_id": "test-key",
                "aws_secret_access_key": "test-secret",
            }
        ),
        encoding="utf-8",
    )
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_bytes(b'{"writer": 1}')
    second.write_bytes(b'{"writer": 2}')
    client = FakeS3Client()
    barrier = threading.Barrier(2)
    lock = threading.Lock()
    stored: list[bytes] = []

    class _Response:
        status = 200

        def getcode(self) -> int:
            return self.status

        def __enter__(self) -> _Response:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    def atomic_create(request: Any, timeout: int) -> _Response:
        assert timeout == 60
        barrier.wait(timeout=10)
        with lock:
            if stored:
                raise HTTPError(request.full_url, 412, "Precondition Failed", hdrs=None, fp=io.BytesIO())
            stored.append(bytes(request.data))
        return _Response()

    monkeypatch.setattr(object_store, "urlopen", atomic_create)
    key = "run/checkpoint_artifacts/iter_000001/inventory.json"

    def publish(path: Path) -> str:
        try:
            return put_object_write_once(
                client,
                "archive",
                key,
                str(path),
                credentials=str(credentials),
            )
        except FileExistsError:
            return "exists"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = sorted(executor.map(publish, (first, second)))

    assert outcomes == ["conditional", "exists"]
    assert stored in ([first.read_bytes()], [second.read_bytes()])
    assert client.uploads == [], "GCS commit creation never enters boto3's AWS4 PutObject path"


@pytest.mark.L0
@pytest.mark.CPU
def test_list_prefix_drops_directory_markers():
    class _Fake:
        def list_objects_v2(self, **kw):
            return {
                "Contents": [
                    {"Key": "p/model/", "Size": 0},  # directory marker — GCS rejects writing these
                    {"Key": "p/model/__0_0.distcp", "Size": 10},
                ],
                "IsTruncated": False,
            }

    assert list_prefix(_Fake(), "b", "p/") == [("p/model/__0_0.distcp", 10)]


@pytest.mark.L0
@pytest.mark.CPU
def test_parse_s3_url():
    assert _parse_s3_url("s3://bucket/a/b/c") == ("bucket", "a/b/c")
    with pytest.raises(ValueError):
        _parse_s3_url("/local/path")


@pytest.mark.L0
@pytest.mark.CPU
def test_upload_prefix_write_once_commit_orders_payloads_and_rejects_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "artifact"
    source.mkdir()
    for name in ("model.pt", "config.json", "manifest.json", "stale.pt"):
        (source / name).write_text(name, encoding="utf-8")
    client = FakeS3Client()
    monkeypatch.setattr(object_store, "build_s3_client", lambda _credentials: client)
    monkeypatch.setattr(object_store, "_transfer_config", lambda: object())

    object_store.upload_prefix(
        str(source),
        "archive",
        "run/checkpoint_artifacts/iter_000001",
        credentials="credentials/<credential>.secret",
        workers=2,
        relative_paths=["model.pt", "config.json", "manifest.json"],
        write_once_commit="manifest.json",
    )

    assert {key.rsplit("/", 1)[-1] for _, key, _ in client.uploads[:-1]} == {"model.pt", "config.json"}
    commit_key = client.uploads[-1][1]
    assert commit_key.endswith("/manifest.json")
    assert client.uploads[-1][2] == {"IfNoneMatch": "*"}
    assert client.objects[commit_key][0] == b"manifest.json"
    assert not any(key.endswith("stale.pt") for key in client.objects)

    first_uploads = list(client.uploads)
    with pytest.raises(FileExistsError, match="refusing to overwrite committed object-store artifact"):
        object_store.upload_prefix(
            str(source),
            "archive",
            "run/checkpoint_artifacts/iter_000001",
            relative_paths=["model.pt", "config.json", "manifest.json"],
            write_once_commit="manifest.json",
        )
    assert client.uploads == first_uploads


@pytest.mark.L0
@pytest.mark.CPU
def test_put_object_write_once_is_conditional_and_fails_closed(tmp_path: Path) -> None:
    """The commit is created only if still absent; a commit that lands in between is never overwritten."""
    commit = tmp_path / "manifest.json"
    commit.write_bytes(b'{"commit": 1}')
    digest = hashlib.sha256(commit.read_bytes()).hexdigest()
    client = FakeS3Client()
    key = "run/checkpoint_artifacts/iter_000001/manifest.json"

    assert put_object_write_once(client, "archive", key, str(commit), {"sha256": digest}) == "conditional"
    assert client.objects[key] == (b'{"commit": 1}', {"sha256": digest})
    assert client.uploads == [("archive", key, {"Metadata": {"sha256": digest}, "IfNoneMatch": "*"})]

    # An existing commit is refused by the last-look HEAD before any put is attempted.
    with pytest.raises(FileExistsError, match="refusing to overwrite committed object-store artifact"):
        put_object_write_once(client, "archive", key, str(commit))
    assert len(client.uploads) == 1

    # A commit that lands between that HEAD and the put is refused by the store's precondition (412).
    other_key = "run/checkpoint_artifacts/iter_000002/manifest.json"
    client.before_put = lambda put_key: client.plant(put_key, b"other writer")
    with pytest.raises(FileExistsError, match="refusing to overwrite committed object-store artifact"):
        put_object_write_once(client, "archive", other_key, str(commit))
    assert client.objects[other_key] == (b"other writer", {}), "the other writer's commit survives"
    assert len(client.uploads) == 1, "a refused put is not an upload"
    client.before_put = None

    # A botocore that rejects the parameter cannot reserve payload keys safely; nothing is written.
    client.conditional_puts_supported = False
    third_key = "run/checkpoint_artifacts/iter_000003/manifest.json"
    with pytest.raises(RuntimeError, match="conditional object creation is required"):
        put_object_write_once(client, "archive", third_key, str(commit))
    assert third_key not in client.objects
    assert len(client.uploads) == 1


@pytest.mark.L0
@pytest.mark.CPU
def test_upload_prefix_object_metadata_binds_digest_and_verifies_by_head(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "artifact"
    source.mkdir()
    weights = b"weights" * 1000
    (source / "model.pt").write_bytes(weights)
    (source / "notes.txt").write_bytes(b"no digest known")
    digest = hashlib.sha256(weights).hexdigest()
    client = FakeS3Client()
    monkeypatch.setattr(object_store, "build_s3_client", lambda _credentials: client)
    monkeypatch.setattr(object_store, "_transfer_config", lambda: object())
    metadata = {"model.pt": {"sha256": digest, "publication-id": "p1"}}
    prefix = "run/checkpoint_artifacts/iter_000001"

    assert object_store.upload_prefix(str(source), "archive", prefix, workers=1, object_metadata=metadata) == 2
    assert client.objects[f"{prefix}/model.pt"] == (weights, {"sha256": digest, "publication-id": "p1"})
    assert client.objects[f"{prefix}/notes.txt"][1] == {}
    extra_args = {key: extra for _bucket, key, extra in client.uploads}
    assert extra_args[f"{prefix}/model.pt"] == {"Metadata": {"sha256": digest, "publication-id": "p1"}}
    assert extra_args[f"{prefix}/notes.txt"] == {}, "files without metadata upload exactly as before"

    # Idempotent skip needs size AND digest; a same-size stranger is re-uploaded, not skipped.
    assert object_store.upload_prefix(str(source), "archive", prefix, workers=1, object_metadata=metadata) == 0
    client.plant(f"{prefix}/model.pt", b"WEIGHTS" * 1000)
    assert object_store.upload_prefix(str(source), "archive", prefix, workers=1, object_metadata=metadata) == 1
    assert client.objects[f"{prefix}/model.pt"][0] == weights

    # verify_sha256 HEADs after the upload: digest where known, size otherwise.
    client.objects.clear()
    client.heads.clear()
    assert (
        object_store.upload_prefix(
            str(source),
            "archive",
            prefix,
            workers=1,
            object_metadata=metadata,
            verify_sha256=True,
        )
        == 2
    )
    # One HEAD decides the skip (absent), one HEAD verifies the upload: two per file.
    assert sorted(client.heads) == [f"{prefix}/model.pt"] * 2 + [f"{prefix}/notes.txt"] * 2

    corrupt = FakeS3Client()
    corrupt.corrupt_uploads = True
    monkeypatch.setattr(object_store, "build_s3_client", lambda _credentials: corrupt)
    with pytest.raises(RuntimeError, match="SHA-256 verification"):
        object_store.upload_prefix(
            str(source),
            "archive",
            prefix,
            workers=1,
            relative_paths=["model.pt"],
            object_metadata=metadata,
            verify_sha256=True,
        )
    assert corrupt.objects[f"{prefix}/model.pt"][1]["sha256"] == digest, "forged metadata alone must not verify"


@pytest.mark.L0
@pytest.mark.CPU
def test_upload_prefix_rejects_metadata_for_unselected_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "model.pt").write_bytes(b"model")
    client = FakeS3Client()
    monkeypatch.setattr(object_store, "build_s3_client", lambda _credentials: client)
    monkeypatch.setattr(object_store, "_transfer_config", lambda: object())

    with pytest.raises(ValueError, match="object_metadata names files that are not uploaded"):
        object_store.upload_prefix(
            str(tmp_path),
            "archive",
            "run/artifact",
            relative_paths=["model.pt"],
            object_metadata={"optim.pt": {"sha256": "0" * 64}},
        )
    assert client.uploads == []


@pytest.mark.L0
@pytest.mark.CPU
def test_upload_prefix_transactional_skips_the_archive_marker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    iter_dir = tmp_path / "checkpoints" / "iter_000001"
    iter_dir.mkdir(parents=True)
    (iter_dir / "model.pt").write_bytes(b"model")
    trainer = tmp_path / "completed-trainer.pt"
    trainer.write_bytes(b"trainer")
    write_marker(
        str(iter_dir),
        destination={"kind": "object_store", "bucket": "archive", "prefix": "old/iter_000001"},
        files={"model.pt": {"size": 5, "sha256": None}},
    )
    client = FakeS3Client()
    monkeypatch.setattr(object_store, "build_s3_client", lambda _credentials: client)
    monkeypatch.setattr(object_store, "_transfer_config", lambda: object())
    commit = {
        "step": 1,
        "id": "a" * 64,
        "archive_required": True,
        "destination": {"kind": "object_store"},
        "manifest": ".validation-publication-v1-" + "a" * 64 + ".json",
    }

    object_store.upload_prefix_transactional(
        str(iter_dir),
        "archive",
        "checkpoints/iter_000001",
        overrides={"trainer.pt": str(trainer)},
        commit_manifest=commit,
        commit_path=str(commit["manifest"]),
        workers=1,
    )

    uploaded = {key.rsplit("/", 1)[-1] for _bucket, key, _extra in client.uploads}
    assert uploaded == {"model.pt", "trainer.pt", str(commit["manifest"])}
    assert MARKER_NAME not in uploaded


@pytest.mark.L0
@pytest.mark.CPU
def test_upload_prefix_transactional_digest_identity_single_put_and_publish_last(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The served-directory mode: an explicit inventory (the .trainer.bak copy is not published), one PUT per object,
    identity by content digest (a planted same-size stranger is replaced, a planted object with the right digest is
    kept), the index strictly after every shard and the commit last, an identical re-run writes nothing, and a changed
    file re-publishes only itself plus the commit."""
    export = tmp_path / "hf"
    export.mkdir()
    files = {
        "config.json": b'{"a": 1}',
        "model-00001-of-00002.safetensors": b"shard-one",
        "model-00002-of-00002.safetensors": b"shard-two",
        "model.safetensors.index.json": b'{"weight_map": {}}',
        "tokenizer.json": b"tok",
    }
    for name, body in files.items():
        (export / name).write_bytes(body)
    (export / "config.json.trainer.bak").write_bytes(b"trainer")
    client = FakeS3Client()
    monkeypatch.setattr(object_store, "build_s3_client", lambda _credentials: client)
    monkeypatch.setattr(
        object_store, "_transfer_config", lambda: pytest.fail("multipart config used in single_put mode")
    )
    monkeypatch.setattr(object_store, "_single_put_transfer_config", lambda: "SINGLE-PUT")
    configs: list[Any] = []
    original_upload = object_store._upload_file

    def recording_upload(client_, path, bucket, key, config, size, metadata, *, verify):  # noqa: ANN001, ANN202
        configs.append(config)
        return original_upload(client_, path, bucket, key, config, size, metadata, verify=verify)

    monkeypatch.setattr(object_store, "_upload_file", recording_upload)
    prefix = "rl_upsampler/oci-c35sr-x/model"
    stale = f"{prefix}/model-00002-of-00002.safetensors"
    client.plant(stale, b"SHARD-TWO", {"sha256": "0" * 64})  # same size, other bytes, wrong digest
    good = f"{prefix}/tokenizer.json"
    client.plant(good, b"tok", {"sha256": hashlib.sha256(b"tok").hexdigest(), "publication-id": "someone-else"})
    commit = {"id": "pub-1", "kind": "hf-export"}
    kwargs: dict[str, Any] = dict(
        credentials="c.json",
        overrides={},
        commit_manifest=commit,
        commit_path=".publication.json",
        workers=2,
        relative_paths=sorted(files),
        transfer="single_put",
        digest_all=True,
        publish_last=["model.safetensors.index.json"],
        invalidate_before_publish=True,
    )

    written = object_store.upload_prefix_transactional(str(export), "b", prefix, **kwargs)
    keys = [key for _bucket, key, _extra in client.uploads]
    assert written == 5 and len(keys) == 5, keys
    assert keys[-1] == f"{prefix}/.publication.json" and keys[-2] == f"{prefix}/model.safetensors.index.json"
    assert set(keys[:-2]) == {f"{prefix}/config.json", f"{prefix}/model-00001-of-00002.safetensors", stale}
    assert good not in keys, "an object already carrying the right digest is kept"
    assert not any(key.endswith(".trainer.bak") for key in keys)
    assert configs and all(config == "SINGLE-PUT" for config in configs)
    assert (
        client.objects[stale][0] == b"shard-two"
        and client.objects[stale][1]["sha256"] == hashlib.sha256(b"shard-two").hexdigest()
    )
    for name, body in files.items():
        assert client.objects[f"{prefix}/{name}"][1]["sha256"] == hashlib.sha256(body).hexdigest(), name
    manifest = json.loads(client.objects[f"{prefix}/.publication.json"][0])
    assert manifest["id"] == "pub-1" and set(manifest["files"]) == set(files)
    assert all("sha256" in entry for entry in manifest["files"].values())

    assert object_store.upload_prefix_transactional(str(export), "b", prefix, **kwargs) == 0
    assert len(client.uploads) == 5, "an identical re-run writes nothing"

    commit_key = f"{prefix}/.publication.json"
    index_key = f"{prefix}/model.safetensors.index.json"

    def assert_invalidated_before_put(key: str) -> None:
        if key == f"{prefix}/config.json":
            assert commit_key not in client.objects and index_key not in client.objects

    client.before_put = assert_invalidated_before_put
    (export / "config.json").write_bytes(b'{"a": 2}')  # same size, other bytes
    assert object_store.upload_prefix_transactional(str(export), "b", prefix, **kwargs) == 3
    assert [key for _b, key, _e in client.uploads[5:]] == [f"{prefix}/config.json", index_key, commit_key]
    assert client.deletes[-2:] == [("b", commit_key), ("b", index_key)]
    client.before_put = None
    with pytest.raises(ValueError):
        object_store.upload_prefix_transactional(str(export), "b", prefix, **{**kwargs, "transfer": "sftp"})
    with pytest.raises(ValueError):
        object_store.upload_prefix_transactional(str(export), "b", prefix, **{**kwargs, "publish_last": ["absent.bin"]})
    (export / ".publication.json").write_bytes(b"an earlier commit left beside the export")
    with pytest.raises(ValueError, match="reserved"):
        object_store.upload_prefix_transactional(
            str(export), "b", prefix, **{**kwargs, "relative_paths": [*sorted(files), ".publication.json"]}
        )


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    ("relative_paths", "write_once_commit", "error"),
    [
        (["../outside.pt"], None, ValueError),
        (["missing.pt"], None, FileNotFoundError),
        (["model.pt"], "manifest.json", ValueError),
    ],
)
def test_upload_prefix_rejects_invalid_selected_paths(
    tmp_path: Path,
    relative_paths: list[str],
    write_once_commit: str | None,
    error: type[Exception],
) -> None:
    (tmp_path / "model.pt").write_bytes(b"model")

    with pytest.raises(error):
        object_store.upload_prefix(
            str(tmp_path),
            "archive",
            "run/artifact",
            relative_paths=relative_paths,
            write_once_commit=write_once_commit,
        )
