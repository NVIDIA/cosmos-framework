# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Object-store access with the GCS S3-interop quirks handled (cosmos-rl parity).

The checkpoint buckets (`<internal-bucket-prefix>`) are Google Cloud Storage behind its S3-compatible
XML API, which implements a SUBSET of AWS S3. Anything that talks to them with stock boto3
hits, in order (all observed 2026-07-06 while preserving the wallace/32B DPO checkpoints):

1. `UploadPartCopy` (server-side multipart copy) → NotImplemented. Bucket-to-bucket copies
   of >5GB objects CANNOT be server-side; `copy_prefix` streams through local staging.
2. Streaming-signature uploads (`STREAMING-AWS4-HMAC-SHA256-PAYLOAD`, non-seekable body)
   → SignatureDoesNotMatch. Uploads must come from seekable files/buffers.
3. boto3 >= 1.36 default flexible checksums (`x-amz-checksum-crc32` on PutObject /
   UploadPart) → SignatureDoesNotMatch "Invalid argument".
4. Zero-byte "directory marker" objects (key ends with `/`) → PutObject rejected.

`build_s3_client()` centralizes 2+3 exactly like cosmos-rl's
`utils/wfm/io/object_store._build_s3_config_for_endpoint`; `copy_prefix`/`download_prefix`
handle 1+4 and are idempotent (same-size destination objects are skipped — safe to rerun
after any interruption).

`put_object_write_once` is the create-if-absent write for small commit objects (a manifest).
S3 uses one `PutObject` with `IfNoneMatch='*'`; GCS uses a separate XML API `PUT`, signed from
the same HMAC credentials as `GOOG4-HMAC-SHA256`, with `x-goog-if-generation-match: 0`. The
store therefore refuses an existing key atomically (412 / 409 → `FileExistsError`, nothing
written). Unsupported conditional creation fails closed.

torch-free on purpose: runs on slurm login nodes. Run BY FILE PATH there (a `-m` import
would pull the posttrain package `__init__` → torch):

    # pull a base checkpoint to lustre (e.g. for rl.init_from)
    python3 cosmos_framework/posttrain/utils/object_store.py download \
        s3://<checkpoint-bucket>/<...>/iter_000010000/model /path/to/shared-storage

    # preserve an expiring checkpoint (experiments bucket has 2-MONTH retention)
    python3 cosmos_framework/posttrain/utils/object_store.py copy \
        s3://<checkpoint-experiments-bucket>/<path> s3://<checkpoint-bucket>/<path>

Credentials default to `credentials/<credential>.secret` (repo-standard JSON with
aws_access_key_id / aws_secret_access_key / endpoint_url / region_name).
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import hmac
import json
import os
import re
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from typing import Any
from urllib.error import HTTPError
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen

try:
    from cosmos_framework.posttrain.utils.log import logger
except ModuleNotFoundError:  # run BY FILE PATH without PYTHONPATH: bootstrap the repo root
    import pathlib
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))
    from cosmos_framework.posttrain.utils.log import logger

_GCS_S3_ENDPOINT = "storage.googleapis.com"
DEFAULT_CREDENTIALS = "credentials/<credential>.secret"


def build_s3_config(endpoint_url: str | None):
    """botocore Config with the GCS-interop fixes when the endpoint is GCS (cosmos-rl
    parity): sigv4 + path addressing + full payload signing (no streaming signatures),
    checksums only when required (kills the boto3>=1.36 CRC32 default). The TypeError
    fallback keeps older botocore (no checksum options) working."""
    from botocore.config import Config

    config = Config(retries={"max_attempts": 20, "mode": "adaptive"}, connect_timeout=10, read_timeout=60)
    if endpoint_url and _GCS_S3_ENDPOINT in endpoint_url:
        try:
            gcs = Config(
                signature_version="s3v4",
                s3={"addressing_style": "path", "payload_signing_enabled": True},
                request_checksum_calculation="when_required",
                response_checksum_validation="when_required",
            )
        except TypeError:  # botocore < 1.36: no checksum knobs (and no CRC32 default either)
            gcs = Config(signature_version="s3v4", s3={"addressing_style": "path", "payload_signing_enabled": True})
        config = config.merge(gcs)
    return config


def _load_credentials(credentials: str) -> dict[str, Any]:
    with open(credentials, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"object-store credentials must be a JSON mapping: {credentials!r}")
    return payload


def build_s3_client(credentials: str = DEFAULT_CREDENTIALS) -> Any:
    """boto3 S3 client from the repo-standard credential JSON, GCS quirks pre-applied."""
    import boto3

    cred = _load_credentials(credentials)
    # No header rewriting for GCS: its S3-interoperability endpoint answers ``ExcessHeaderValues``
    # (HTTP 400 "Invalid argument") to any request that carries an ``x-goog-*`` header next to the
    # ``x-amz-*``/AWS-signed ones botocore always sends, so translating ``If-None-Match`` into
    # ``x-goog-if-generation-match`` made EVERY conditional PutObject fail (verified live against
    # <checkpoint-experiments-bucket> with botocore 1.42; the trainer logged it only as a
    # rank-0 warning inside a Ray actor). Bulk objects stay on this boto3 client; the two small
    # write-once objects use a separately GOOG4-HMAC-signed XML request below.
    return boto3.client("s3", config=build_s3_config(cred.get("endpoint_url")), **cred)


def _gcs_v4_request(
    credentials: Mapping[str, Any],
    bucket: str,
    key: str,
    body: bytes,
    metadata: Mapping[str, str],
    *,
    now: datetime | None = None,
) -> Request:
    """Build one GOOG4-HMAC-SHA256 XML PUT with an atomic generation-zero precondition."""
    endpoint = str(credentials.get("endpoint_url") or "")
    parsed = urlsplit(endpoint)
    if parsed.scheme != "https" or parsed.hostname != _GCS_S3_ENDPOINT or parsed.path not in ("", "/"):
        raise ValueError(f"GCS atomic creation requires the HTTPS XML endpoint, got {endpoint!r}")
    access_id = str(credentials.get("aws_access_key_id") or "")
    secret = str(credentials.get("aws_secret_access_key") or "")
    location = str(credentials.get("region_name") or "auto")
    if not access_id or not secret:
        raise ValueError("GCS atomic creation requires an HMAC access ID and secret")
    if not bucket or not key:
        raise ValueError("GCS atomic creation requires a bucket and object key")

    active = now or datetime.now(timezone.utc)
    if active.tzinfo is None:
        raise ValueError("GCS request signing time must be timezone-aware")
    active = active.astimezone(timezone.utc)
    datestamp = active.strftime("%Y%m%d")
    timestamp = active.strftime("%Y%m%dT%H%M%SZ")
    payload_hash = hashlib.sha256(body).hexdigest()
    canonical_uri = quote(f"/{bucket}/{key.lstrip('/')}", safe="/~")
    authority = parsed.netloc.lower()
    signed: dict[str, str] = {
        "host": authority,
        "x-goog-content-sha256": payload_hash,
        "x-goog-date": timestamp,
        "x-goog-if-generation-match": "0",
    }
    for name, value in metadata.items():
        normalized_name = str(name).strip().lower()
        if re.fullmatch(r"[a-z0-9][a-z0-9-]*", normalized_name) is None:
            raise ValueError(f"invalid GCS metadata name: {name!r}")
        signed[f"x-goog-meta-{normalized_name}"] = " ".join(str(value).strip().split())
    signed_headers = ";".join(sorted(signed))
    canonical_headers = "".join(f"{name}:{signed[name]}\n" for name in sorted(signed))
    canonical_request = "\n".join(("PUT", canonical_uri, "", canonical_headers, signed_headers, payload_hash))
    algorithm = "GOOG4-HMAC-SHA256"
    scope = f"{datestamp}/{location}/storage/goog4_request"
    string_to_sign = "\n".join((algorithm, timestamp, scope, hashlib.sha256(canonical_request.encode()).hexdigest()))

    def _sign(signing_key: bytes, value: str) -> bytes:
        return hmac.new(signing_key, value.encode(), hashlib.sha256).digest()

    date_key = _sign(f"GOOG4{secret}".encode(), datestamp)
    location_key = _sign(date_key, location)
    service_key = _sign(location_key, "storage")
    request_key = _sign(service_key, "goog4_request")
    signature = hmac.new(request_key, string_to_sign.encode(), hashlib.sha256).hexdigest()
    headers = {
        **signed,
        "Authorization": (
            f"{algorithm} Credential={access_id}/{scope}, SignedHeaders={signed_headers}, Signature={signature}"
        ),
        "Content-Type": "application/octet-stream",
    }
    return Request(
        f"{parsed.scheme}://{authority}{canonical_uri}",
        data=body,
        headers=headers,
        method="PUT",
    )


def _gcs_xml_put_if_absent(
    credentials: Mapping[str, Any],
    bucket: str,
    key: str,
    body: bytes,
    metadata: Mapping[str, str],
) -> str:
    """Atomically create one GCS XML object from HMAC credentials, or raise FileExistsError."""
    request = _gcs_v4_request(credentials, bucket, key, body, metadata)
    try:
        with urlopen(request, timeout=60) as response:
            status = int(getattr(response, "status", response.getcode()))
    except HTTPError as error:
        if error.code in (409, 412):
            raise FileExistsError(
                f"refusing to overwrite committed object-store artifact: s3://{bucket}/{key} "
                f"(GCS refused the generation-zero create: HTTP {error.code})"
            ) from error
        raise RuntimeError(f"GCS atomic object creation failed for s3://{bucket}/{key}: HTTP {error.code}") from error
    if not 200 <= status < 300:
        raise RuntimeError(f"GCS atomic object creation failed for s3://{bucket}/{key}: HTTP {status}")
    return "conditional"


def list_prefix(client: Any, bucket: str, prefix: str) -> list[tuple[str, int]]:
    """All (key, size) under a prefix, EXCLUDING zero-byte directory markers (keys ending
    in `/`) — GCS interop rejects writing them back and they carry no data."""
    keys, token = [], None
    while True:
        kwargs: dict[str, Any] = dict(Bucket=bucket, Prefix=prefix, MaxKeys=1000)
        if token:
            kwargs["ContinuationToken"] = token
        r = client.list_objects_v2(**kwargs)
        keys += [(o["Key"], o["Size"]) for o in r.get("Contents", []) if not o["Key"].endswith("/")]
        if not r.get("IsTruncated"):
            return keys
        token = r["NextContinuationToken"]


def _transfer_config():
    from boto3.s3.transfer import TransferConfig

    return TransferConfig(multipart_threshold=512 << 20, multipart_chunksize=512 << 20, max_concurrency=8)


def _single_put_transfer_config():
    """One ``PutObject`` per file, however large: for S3-compatible gateways that refuse multipart ``UploadPart``
    (the SigV2-signed evaluation gateway answers it with SignatureDoesNotMatch). Parallelism then comes from uploading
    several objects at once, never from within one object."""
    from boto3.s3.transfer import TransferConfig

    return TransferConfig(
        multipart_threshold=5 * 1024**4, multipart_chunksize=5 * 1024**4, max_concurrency=1, use_threads=False
    )


TRANSFER_MODES = ("multipart", "single_put")


def _selected_files(src_dir: str, relative_paths: Iterable[str]) -> list[tuple[str, str, int]]:
    """``(absolute path, canonical relative path, size)`` of an explicit inventory below ``src_dir``; a duplicate, an
    absolute or non-canonical path, or a missing file is refused."""
    selected = tuple(relative_paths)
    if len(selected) != len(set(selected)):
        raise ValueError("selected upload paths must not contain duplicates")
    source_root = os.path.realpath(src_dir)
    files: list[tuple[str, str, int]] = []
    for relative in selected:
        normalized = relative.replace(os.sep, "/")
        if (
            not normalized
            or normalized.startswith("/")
            or "\\" in normalized
            or any(segment in ("", ".", "..") for segment in normalized.split("/"))
        ):
            raise ValueError(f"selected upload path must be canonical and relative: {relative!r}")
        path = os.path.realpath(os.path.join(src_dir, *normalized.split("/")))
        if not path.startswith(source_root + os.sep) or not os.path.isfile(path):
            raise FileNotFoundError(f"selected upload file does not exist below source: {relative!r}")
        files.append((path, normalized, os.path.getsize(path)))
    return files


def _dst_exists(client: Any, bucket: str, key: str, size: int) -> bool:
    try:
        return client.head_object(Bucket=bucket, Key=key)["ContentLength"] == size
    except Exception:  # noqa: BLE001 — any head failure ⇒ treat as absent, re-copy
        return False


def _require_missing_object(client: Any, bucket: str, key: str) -> None:
    """Fail closed unless one object is authoritatively absent."""
    from botocore.exceptions import ClientError

    try:
        client.head_object(Bucket=bucket, Key=key)
    except ClientError as error:
        response = error.response
        code = str(response.get("Error", {}).get("Code", ""))
        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if code in ("404", "NoSuchKey", "NotFound") or status == 404:
            return
        raise
    raise FileExistsError(f"refusing to overwrite committed object-store artifact: s3://{bucket}/{key}")


def put_object_write_once(
    client: Any,
    bucket: str,
    key: str,
    path: str,
    metadata: Mapping[str, str] | None = None,
    *,
    credentials: str | None = None,
) -> str:
    """Create ``s3://bucket/key`` from one small local file only if no object exists there yet.

    The object is written with a single ``PutObject`` carrying ``IfNoneMatch='*'``, so the store
    itself refuses the write when the key already exists (S3 answers 412 ``PreconditionFailed``; a
    concurrent conditional writer surfaces as 409 ``ConditionalRequestConflict``); either raises
    ``FileExistsError`` and nothing is written. A last-look HEAD precedes the put for a clear
    existing-object error, but correctness comes from the store-side precondition. S3 uses
    botocore's ``IfNoneMatch='*'``. GCS uses a separately signed XML request with
    ``x-goog-if-generation-match: 0``; it must not share the AWS4 request whose mixed
    ``x-amz-*`` / ``x-goog-*`` headers GCS rejects as ``ExcessHeaderValues``. When the required
    conditional primitive is unavailable, the write is refused: HEAD followed by an unconditional
    PUT cannot provide this contract. Returns ``"conditional"``. The file is read whole into
    memory, so this is for commit objects, never payloads.
    """
    from botocore.exceptions import ClientError, ParamValidationError

    _require_missing_object(client, bucket, key)
    with open(path, "rb") as handle:
        body = handle.read()
    metadata_dict = {str(name): str(value) for name, value in (metadata or {}).items()}
    client_endpoint = str(getattr(getattr(client, "meta", None), "endpoint_url", "") or "")
    client_host = urlsplit(client_endpoint).hostname
    credential_payload = (
        _load_credentials(credentials)
        if credentials is not None and (client_host == _GCS_S3_ENDPOINT or os.path.isfile(credentials))
        else None
    )
    credential_endpoint = str((credential_payload or {}).get("endpoint_url") or "")
    endpoint_hosts = {urlsplit(endpoint).hostname for endpoint in (credential_endpoint, client_endpoint) if endpoint}
    if _GCS_S3_ENDPOINT in endpoint_hosts:
        if credential_payload is None:
            raise RuntimeError("GCS conditional object creation requires the credential file used by the client")
        return _gcs_xml_put_if_absent(credential_payload, bucket, key, body, metadata_dict)
    kwargs: dict[str, Any] = {"Bucket": bucket, "Key": key, "Body": body}
    if metadata_dict:
        kwargs["Metadata"] = metadata_dict
    try:
        client.put_object(**kwargs, IfNoneMatch="*")
        return "conditional"
    except ParamValidationError as error:
        raise RuntimeError(
            "conditional object creation is required; this botocore does not accept IfNoneMatch"
        ) from error
    except ClientError as error:
        response = error.response
        code = str(response.get("Error", {}).get("Code", ""))
        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if code in ("PreconditionFailed", "ConditionalRequestConflict") or status in (412, 409):
            raise FileExistsError(
                f"refusing to overwrite committed object-store artifact: s3://{bucket}/{key} "
                f"(the store refused the create-if-absent write: {code or status})"
            ) from error
        if code != "NotImplemented" and status != 501:
            raise
        raise RuntimeError(f"conditional object creation is required; store returned {code or status}") from error


def upload_destination_identity(upload: dict[str, Any], iter_name: str) -> dict[str, str]:
    """Return the non-secret semantic destination bound into a publication journal."""
    bucket = str(upload.get("bucket", ""))
    prefix = str(upload.get("prefix", "")).rstrip("/")
    if not bucket or not prefix or not iter_name or os.path.basename(iter_name) != iter_name:
        raise ValueError("checkpoint upload requires a bucket, prefix, and one-segment iteration name")
    credentials = str(upload.get("credentials") or DEFAULT_CREDENTIALS)
    with open(credentials, encoding="utf-8") as handle:
        credential_payload = json.load(handle)
    endpoint_url = str(credential_payload.get("endpoint_url") or "https://s3.amazonaws.com")
    parsed = urlsplit(endpoint_url if "://" in endpoint_url else f"https://{endpoint_url}")
    hostname = (parsed.hostname or "").lower().rstrip(".")
    if not hostname:
        raise ValueError(f"checkpoint upload credentials have no endpoint authority: {credentials!r}")
    authority = f"{hostname}:{parsed.port}" if parsed.port is not None else hostname
    return {
        "kind": "object_store",
        "endpoint_authority": authority,
        "bucket": bucket,
        "prefix": f"{prefix}/{iter_name}",
    }


def file_sha256(path: str) -> str:
    """Return the lowercase SHA-256 of one file, streamed in 8 MiB chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


_file_sha256 = file_sha256


def response_metadata(response: Mapping[str, Any]) -> dict[str, str]:
    """Normalize custom metadata from boto3 fields or raw S3/GCS XML response headers."""
    metadata = {str(name).lower(): str(value) for name, value in (response.get("Metadata") or {}).items()}
    headers = (response.get("ResponseMetadata") or {}).get("HTTPHeaders") or {}
    for name, value in headers.items():
        lowered = str(name).lower()
        for prefix in ("x-amz-meta-", "x-goog-meta-"):
            if lowered.startswith(prefix):
                metadata[lowered.removeprefix(prefix)] = str(value)
    return metadata


def _remote_sha256_matches(
    client: Any,
    bucket: str,
    key: str,
    size: int,
    digest: str,
    required_metadata: dict[str, str] | None = None,
) -> bool:
    try:
        head = client.head_object(Bucket=bucket, Key=key)
    except Exception:  # noqa: BLE001 -- any head failure requires an upload
        return False
    metadata = response_metadata(head)
    return (
        head.get("ContentLength") == size
        and metadata.get("sha256") == digest
        and all(metadata.get(name.lower()) == value for name, value in (required_metadata or {}).items())
    )


def _verify_uploaded_object(client: Any, bucket: str, key: str, size: int, metadata: Mapping[str, str]) -> None:
    """Apply the same size/digest/identity proof to every checkpoint archive upload."""
    digest = metadata.get("sha256")
    if digest is not None:
        required = {name: value for name, value in metadata.items() if name != "sha256"}
        if not _remote_sha256_matches(client, bucket, key, size, digest, required):
            raise RuntimeError(f"uploaded object failed SHA-256 verification: s3://{bucket}/{key}")
    else:
        head = client.head_object(Bucket=bucket, Key=key)
        actual_metadata = response_metadata(head)
        if head.get("ContentLength") != size or any(
            actual_metadata.get(name.lower()) != value for name, value in metadata.items()
        ):
            raise RuntimeError(f"uploaded object failed size verification: s3://{bucket}/{key}")


def _upload_file(
    client: Any, path: str, bucket: str, key: str, config: Any, size: int, metadata: Mapping[str, str], *, verify: bool
) -> None:
    """Transfer one seekable file, then verify it when the archive contract requires proof."""
    if metadata:
        client.upload_file(path, bucket, key, Config=config, ExtraArgs={"Metadata": dict(metadata)})
    else:
        client.upload_file(path, bucket, key, Config=config)
    if verify:
        _verify_uploaded_object(client, bucket, key, size, metadata)


def _remote_bytes_match(client: Any, bucket: str, key: str, payload: bytes, digest: str) -> bool:
    try:
        response = client.get_object(Bucket=bucket, Key=key)
        remote_payload = response["Body"].read()
    except Exception:  # noqa: BLE001 -- an absent/unreadable commit is not authoritative
        return False
    metadata = response_metadata(response)
    return remote_payload == payload and metadata.get("sha256") == digest


def _remote_inventory_matches(
    client: Any,
    bucket: str,
    dst_prefix: str,
    inventory: dict[str, dict[str, int | str]],
    publication_id: str,
    *,
    workers: int,
    identity: str = "publication",
) -> bool:
    """Confirm that every object named by an existing commit is still visible and attributable.

    ``identity`` is what makes an object attributable: ``"publication"`` requires the publication-id tag of this
    commit (plus the digest where the inventory knows one); ``"digest"`` requires the inventory's content digest on
    every object, whichever publication wrote it -- the same bytes are the same object.
    """

    def _matches(item: tuple[str, dict[str, int | str]]) -> bool:
        relative, entry = item
        key = dst_prefix.rstrip("/") + "/" + relative
        try:
            head = client.head_object(Bucket=bucket, Key=key)
        except Exception:  # noqa: BLE001 -- an incomplete committed prefix must be repaired
            return False
        metadata = response_metadata(head)
        if head.get("ContentLength") != entry["size"]:
            return False
        expected_sha256 = entry.get("sha256")
        if identity == "digest":
            return expected_sha256 is not None and metadata.get("sha256") == expected_sha256
        if metadata.get("publication-id") != publication_id:
            return False
        return expected_sha256 is None or metadata.get("sha256") == expected_sha256

    with concurrent.futures.ThreadPoolExecutor(workers) as executor:
        return all(executor.map(_matches, sorted(inventory.items())))


def upload_prefix_transactional(
    src_dir: str,
    bucket: str,
    dst_prefix: str,
    *,
    credentials: str = DEFAULT_CREDENTIALS,
    overrides: dict[str, str],
    commit_manifest: dict[str, Any],
    commit_path: str,
    workers: int = 8,
    relative_paths: Iterable[str] | None = None,
    transfer: str = "multipart",
    digest_all: bool = False,
    publish_last: Sequence[str] = (),
    invalidate_before_publish: bool = False,
) -> int:
    """Upload one immutable generation and write its commit object last.

    ``overrides`` maps a checkpoint-relative object name (notably ``trainer.pt``) to bytes outside
    ``src_dir``. An exact verified commit is the only reuse proof. Without one, every bulk object is
    force-uploaded (so a stale same-size object cannot survive), tagged with the publication id, and
    size-verified; the completed trainer is additionally SHA-256 verified before the commit is force-
    written last. Bulk model/optimizer bytes rely on the checkpoint-generation immutability contract,
    avoiding an extra hash pass over giant payloads while keeping partial prefixes non-authoritative.

    Four keyword options adapt the same contract to a served directory that is not a checkpoint
    generation (an HF export the evaluation service deploys from):

    * ``relative_paths`` restricts the inventory to an explicit, canonical list below ``src_dir``
      (files a caller keeps beside the export but must not publish); the default walks ``src_dir``.
    * ``transfer="single_put"`` uploads every object as one ``PutObject`` for gateways that refuse
      multipart ``UploadPart``; the default keeps the multipart transfer of the archive path.
    * ``digest_all`` gives every inventory entry its SHA-256, so identity becomes the content digest:
      objects carry it as metadata, every upload is digest-verified, an object already present with
      the same size and digest is kept (a same-size stranger is replaced), and a verified commit's
      inventory is attributable by digest whichever publication wrote it. Without it, identity is the
      publication id and bulk objects are force-uploaded, as before.
    * ``publish_last`` names inventory files uploaded strictly after every other bulk object, in the
      given order and before the commit (a safetensors index: readers find no complete-looking
      directory until every shard it names is there).
    * ``invalidate_before_publish`` deletes an existing commit and every ``publish_last`` object
      before replacing bulk objects. Mutable served prefixes then expose an explicit uncommitted,
      no-index window instead of an old commit/index beside a mixture of generations; immutable
      checkpoint-generation uploads leave it false.

    Returns the number of objects written (uploads plus the commit); 0 when the exact publication is
    already committed and its inventory verified.
    """
    normalized_overrides = {path.replace(os.sep, "/"): source for path, source in overrides.items()}
    if not commit_path or "/" in commit_path or os.path.basename(commit_path) != commit_path:
        raise ValueError(f"transaction commit path must be one filename, got {commit_path!r}")
    if commit_path in normalized_overrides:
        raise ValueError("transaction commit path cannot also be a bulk override")
    if transfer not in TRANSFER_MODES:
        raise ValueError(f"transfer must be one of {TRANSFER_MODES}, got {transfer!r}")
    reserved = (".uploaded", ".uploaded.tmp", commit_path)

    sources: dict[str, str] = {}
    if relative_paths is None:
        for root, _dirs, names in os.walk(src_dir):
            for name in names:
                path = os.path.join(root, name)
                relative = os.path.relpath(path, src_dir).replace(os.sep, "/")
                if relative in reserved:
                    continue
                sources[relative] = path
    else:
        for path, relative, _size in _selected_files(src_dir, relative_paths):
            if relative in reserved:
                raise ValueError(f"selected upload path {relative!r} is reserved for the transaction itself")
            sources[relative] = path
    sources.update(normalized_overrides)
    if not sources:
        raise ValueError("transactional checkpoint upload has no objects")
    last = [path.replace(os.sep, "/") for path in publish_last]
    if len(last) != len(set(last)) or any(relative not in sources for relative in last):
        raise ValueError(f"publish_last must name distinct uploaded files, got {list(publish_last)!r}")

    inventory: dict[str, dict[str, int | str]] = {
        relative: {"size": os.path.getsize(path)} for relative, path in sorted(sources.items())
    }
    for relative in sorted(sources) if digest_all else normalized_overrides:
        inventory[relative]["sha256"] = _file_sha256(sources[relative])
    identity = "digest" if digest_all else "publication"
    client = build_s3_client(credentials)
    cfg = _transfer_config() if transfer == "multipart" else _single_put_transfer_config()

    manifest_payload = {**commit_manifest, "files": inventory}
    manifest_bytes = (json.dumps(manifest_payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    commit_key = dst_prefix.rstrip("/") + "/" + commit_path
    publication_id = str(commit_manifest["id"])
    if _remote_bytes_match(
        client,
        bucket,
        commit_key,
        manifest_bytes,
        manifest_sha256,
    ) and _remote_inventory_matches(
        client,
        bucket,
        dst_prefix,
        inventory,
        publication_id,
        workers=workers,
        identity=identity,
    ):
        return 0

    if invalidate_before_publish:
        # Remove authority first, then every complete-looking index, before any bulk object changes.
        # Deletion failures propagate rather than publishing into a still-authoritative prefix.
        client.delete_object(Bucket=bucket, Key=commit_key)
        for relative in last:
            client.delete_object(Bucket=bucket, Key=dst_prefix.rstrip("/") + "/" + relative)

    def _upload_one(relative: str) -> str:
        path = sources[relative]
        entry = inventory[relative]
        key = dst_prefix.rstrip("/") + "/" + relative
        item_client = build_s3_client(credentials)
        metadata = {"publication-id": publication_id}
        if "sha256" in entry:
            metadata["sha256"] = str(entry["sha256"])
        if digest_all and _remote_sha256_matches(item_client, bucket, key, int(entry["size"]), str(entry["sha256"])):
            return "skip"  # the same bytes are already there: content identity, not size, is the proof
        _upload_file(
            item_client,
            path,
            bucket,
            key,
            cfg,
            int(entry["size"]),
            metadata,
            verify=True,
        )
        return "ok"

    bulk = sorted(relative for relative in sources if relative not in last)
    with concurrent.futures.ThreadPoolExecutor(workers) as executor:
        results = list(executor.map(_upload_one, bulk))
    # publish_last objects follow every bulk object and precede the commit, in the caller's order.
    results += [_upload_one(relative) for relative in last]

    descriptor, manifest_file = tempfile.mkstemp(prefix="posttrain-validation-publication-", suffix=".json")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(manifest_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        # Force the commit object even if an old/corrupt object has the same length. It is the
        # linearization point and therefore must be written strictly after the verified bulk set.
        client.upload_file(
            manifest_file,
            bucket,
            commit_key,
            Config=cfg,
            ExtraArgs={"Metadata": {"sha256": manifest_sha256}},
        )
        if not _remote_bytes_match(client, bucket, commit_key, manifest_bytes, manifest_sha256):
            raise RuntimeError(f"checkpoint publication commit failed SHA-256 verification: {commit_key}")
    finally:
        try:
            os.remove(manifest_file)
        except FileNotFoundError:
            pass
    return results.count("ok") + 1


def download_prefix(
    bucket: str, prefix: str, dst_dir: str, credentials: str = DEFAULT_CREDENTIALS, workers: int = 16
) -> int:
    """Threaded, idempotent download of everything under `prefix` into `dst_dir`
    (existing same-size files are skipped). Returns the number of objects downloaded."""
    client = build_s3_client(credentials)
    keys = list_prefix(client, bucket, prefix)
    logger.info(
        f"download s3://{bucket}/{prefix} -> {dst_dir}: {len(keys)} objects, {sum(s for _, s in keys) / 1e9:.1f} GB"
    )
    os.makedirs(dst_dir, exist_ok=True)
    cfg = _transfer_config()

    def dl(item: tuple[str, int]) -> str:
        key, size = item
        rel = key[len(prefix) :].lstrip("/") or os.path.basename(key)  # prefix may BE one object
        dst = os.path.join(dst_dir, rel)
        if os.path.exists(dst) and os.path.getsize(dst) == size:
            return "skip"
        os.makedirs(os.path.dirname(dst) or dst_dir, exist_ok=True)
        build_s3_client(credentials).download_file(bucket, key, dst, Config=cfg)
        return "ok"

    with concurrent.futures.ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(dl, keys))
    logger.info(f"download done: {res.count('ok')} fetched, {res.count('skip')} skipped / {len(keys)}")
    return res.count("ok")


def upload_prefix(
    src_dir: str,
    bucket: str,
    dst_prefix: str,
    credentials: str = DEFAULT_CREDENTIALS,
    workers: int = 8,
    *,
    relative_paths: Iterable[str] | None = None,
    write_once_commit: str | None = None,
    object_metadata: Mapping[str, Mapping[str, str]] | None = None,
    verify_sha256: bool = False,
) -> int:
    """Thread a local directory-tree upload to ``s3://bucket/dst_prefix``.

    Ordinary uploads skip same-size objects and are safe to rerun. Files remain seekable, avoiding
    the GCS streaming-signature limitation. ``relative_paths`` restricts the upload to an explicit
    manifest-backed inventory. ``write_once_commit`` names its commit file and instead requires that
    remote object to be absent, force-uploads every payload, then creates the commit last through
    ``put_object_write_once`` (a conditional create-if-absent put, never an overwrite): a commit that
    another writer landed in between raises ``FileExistsError`` and the payloads it force-uploaded
    are left as an uncommitted prefix. Callers that release local bytes must re-read the prefix after
    the commit; the payload uploads themselves are unconditional.

    ``object_metadata`` maps a relative path to the user metadata stored on its object. When it
    carries a ``sha256`` entry the digest becomes the object's identity: the idempotent skip then
    requires size AND digest metadata to match (a same-size stranger is re-uploaded), and
    ``verify_sha256`` HEADs every uploaded object afterwards, checking size plus digest metadata
    where one is known and size alone otherwise, raising ``RuntimeError`` on the first mismatch.
    This is an identity check over what the store reports, not a re-read of the bytes: multipart
    ETags are not content digests, so a store that echoes metadata while corrupting same-size bytes
    is only caught by a later content-level verification.
    """
    files: list[tuple[str, str, int]] = []
    if relative_paths is None:
        for root, _dirs, names in os.walk(src_dir):
            for name in names:
                path = os.path.join(root, name)
                files.append((path, os.path.relpath(path, src_dir).replace(os.sep, "/"), os.path.getsize(path)))
    else:
        files = _selected_files(src_dir, relative_paths)
    normalized_commit = write_once_commit.replace(os.sep, "/") if write_once_commit is not None else None
    commit_items = [item for item in files if item[1] == normalized_commit]
    if write_once_commit is not None:
        if len(commit_items) != 1:
            raise ValueError(f"write_once_commit must name exactly one selected upload file, got {write_once_commit!r}")
        commit_key = dst_prefix.rstrip("/") + "/" + commit_items[0][1]
        _require_missing_object(build_s3_client(credentials), bucket, commit_key)
    payload_files = [item for item in files if item[1] != normalized_commit]
    metadata_by_path: dict[str, dict[str, str]] = {
        relative.replace(os.sep, "/"): {str(name): str(value) for name, value in entries.items()}
        for relative, entries in (object_metadata or {}).items()
    }
    unknown = set(metadata_by_path) - {item[1] for item in files}
    if unknown:
        raise ValueError(f"object_metadata names files that are not uploaded: {sorted(unknown)!r}")
    logger.info(
        f"upload {src_dir} -> s3://{bucket}/{dst_prefix}: {len(files)} files, {sum(s for *_, s in files) / 1e9:.1f} GB"
    )
    cfg = _transfer_config()

    def up(item: tuple[str, str, int], *, force: bool = False, commit: bool = False) -> str:
        path, rel, size = item
        key = dst_prefix.rstrip("/") + "/" + rel
        cli = build_s3_client(credentials)
        metadata = metadata_by_path.get(rel, {})
        digest = metadata.get("sha256")
        if not force and not commit:
            if digest is not None and _remote_sha256_matches(cli, bucket, key, size, digest):
                return "skip"
            if digest is None and _dst_exists(cli, bucket, key, size):
                return "skip"
        if commit:
            # The commit is the linearization point: created only if still absent, never overwritten.
            put_object_write_once(cli, bucket, key, path, metadata, credentials=credentials)
        else:
            _upload_file(cli, path, bucket, key, cfg, size, metadata, verify=verify_sha256)
        if commit and verify_sha256:
            _verify_uploaded_object(cli, bucket, key, size, metadata)
        return "ok"

    with concurrent.futures.ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(lambda item: up(item, force=normalized_commit is not None), payload_files))
    if commit_items:
        res.append(up(commit_items[0], commit=True))
    logger.info(f"upload done: {res.count('ok')} uploaded, {res.count('skip')} skipped / {len(files)}")
    return res.count("ok")


def copy_prefix(
    src_bucket: str,
    src_prefix: str,
    dst_bucket: str,
    dst_prefix: str,
    credentials: str = DEFAULT_CREDENTIALS,
    workers: int = 3,
    stage_dir: str | None = None,
) -> int:
    """Idempotent bucket-to-bucket copy via LOCAL STAGING (GCS interop supports neither
    server-side multipart copy nor streaming-signature uploads — each object is staged in
    `stage_dir` [default $TMPDIR] then uploaded seekable). Returns objects copied."""
    client = build_s3_client(credentials)
    keys = list_prefix(client, src_bucket, src_prefix)
    logger.info(
        f"copy s3://{src_bucket}/{src_prefix} -> s3://{dst_bucket}/{dst_prefix}: "
        f"{len(keys)} objects, {sum(s for _, s in keys) / 1e9:.1f} GB"
    )
    stage = stage_dir or os.environ.get("TMPDIR", "/tmp")
    cfg = _transfer_config()

    def cp(item: tuple[str, int]) -> str:
        key, size = item
        rel = key[len(src_prefix) :].lstrip("/")
        dst_key = dst_prefix.rstrip("/") + ("/" + rel if rel else "")  # src prefix may BE one object
        cli = build_s3_client(credentials)
        if _dst_exists(cli, dst_bucket, dst_key, size):
            return "skip"
        with tempfile.NamedTemporaryFile(dir=stage) as tmp:
            cli.download_file(src_bucket, key, tmp.name, Config=cfg)
            cli.upload_file(tmp.name, dst_bucket, dst_key, Config=cfg)
        logger.info(f"copied {dst_key.rsplit('/', 1)[-1]} ({size / 1e9:.1f}GB)")
        return "ok"

    with concurrent.futures.ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(cp, keys))
    logger.info(f"copy done: {res.count('ok')} copied, {res.count('skip')} skipped / {len(keys)}")
    return res.count("ok")


def _parse_s3_url(url: str) -> tuple[str, str]:
    if not url.startswith("s3://"):
        raise ValueError(f"expected s3://bucket/prefix, got {url!r}")
    bucket, _, prefix = url[len("s3://") :].partition("/")
    return bucket, prefix


def main(argv: list[str] | None = None) -> None:
    import argparse

    p = argparse.ArgumentParser(description="GCS-safe object-store download / bucket-to-bucket copy.")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download", help="download s3://bucket/prefix to a local dir")
    d.add_argument("src"), d.add_argument("dst")
    u = sub.add_parser("upload", help="upload a local dir to s3://bucket/prefix")
    u.add_argument("src"), u.add_argument("dst")
    c = sub.add_parser("copy", help="copy s3://src/prefix to s3://dst/prefix via local staging")
    c.add_argument("src"), c.add_argument("dst")
    c.add_argument("--stage-dir", default=None, help="staging dir (default $TMPDIR)")
    for sp in (d, u, c):
        sp.add_argument("--credentials", default=DEFAULT_CREDENTIALS)
        sp.add_argument("--workers", type=int, default=None)
    a = p.parse_args(argv)
    if a.cmd == "download":
        sb, spfx = _parse_s3_url(a.src)
        download_prefix(sb, spfx, a.dst, credentials=a.credentials, workers=a.workers or 16)
    elif a.cmd == "upload":
        db, dpfx = _parse_s3_url(a.dst)
        upload_prefix(a.src, db, dpfx, credentials=a.credentials, workers=a.workers or 8)
    else:
        sb, spfx = _parse_s3_url(a.src)
        db, dpfx = _parse_s3_url(a.dst)
        copy_prefix(sb, spfx, db, dpfx, credentials=a.credentials, workers=a.workers or 3, stage_dir=a.stage_dir)


if __name__ == "__main__":
    main()
