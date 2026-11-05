# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Shared local-to-archive publication for trainers and offline checkpoint tools.

An immutable artifact inventory reserves canonical object keys before any payload write. All
publishers of that prefix must prove identical bytes, so a delayed publisher cannot overwrite a
winner's payload with another checkpoint. The native manifest remains the commit written last;
an inventory without that commit is an interrupted upload which only the same request may resume.
Full checkpoint archives retain their existing generation and validation-override contracts.
Error handling belongs to the caller: this service always raises when publication fails.
"""

from __future__ import annotations

import os
import posixpath
import tempfile
from collections.abc import Mapping
from typing import Any

from cosmos_framework.posttrain.checkpoint.archive.marker import (
    ARCHIVE_PROTOCOL,
    ARCHIVE_RECORD,
    payload_inventory,
    read_marker,
    write_marker,
)
from cosmos_framework.posttrain.checkpoint.archive.reader import (
    remote_inventory_status,
    remote_object_status,
    unverified_objects,
)
from cosmos_framework.posttrain.checkpoint.artifact import (
    CHECKPOINT_ARTIFACT_MANIFEST,
    build_artifact_manifest,
    validate_artifact_manifest,
)
from cosmos_framework.posttrain.checkpoint.io import atomic_write_bytes, atomic_write_json, release_page_cache
from cosmos_framework.posttrain.checkpoint.manifest import iteration_name, json_bytes, parse_json_object, sha256
from cosmos_framework.posttrain.utils import object_store
from cosmos_framework.posttrain.utils.log import logger

ARTIFACT_ROOT_NAME = "checkpoint_artifacts"
LEGACY_MANIFEST_RECORD = ".posttrain-checkpoint-artifact.legacy.json"
_NATIVE_MANIFEST_KEYS = frozenset(
    {"protocol", "iteration", "iteration_name", "checkpoint_format", "checkpoint_components", "components"}
)


def artifact_destination(upload: Mapping[str, Any], name: str) -> tuple[str, dict[str, str]]:
    """Return the object prefix and destination identity of one artifact under a run's upload prefix.

    ``upload`` carries the ``rl.upload.*`` fields (``bucket``, ``prefix``, optional ``credentials``).
    ``prefix`` is the run's ``<project>/<group>/<run>/<ts>/checkpoints``; the artifact lands beside it
    under ``<dirname>/checkpoint_artifacts/<name>``, the rule the trainer's archive path applies, so
    offline and training-time publication address the same objects and write comparable markers.
    """
    configured_root = str(upload.get("prefix") or "").strip("/")
    if not configured_root:
        raise ValueError("artifact publication requires the run's checkpoints upload prefix")
    artifact_root = posixpath.join(posixpath.dirname(configured_root), ARTIFACT_ROOT_NAME)
    identity = object_store.upload_destination_identity({**upload, "prefix": artifact_root}, name)
    return posixpath.join(artifact_root, name), identity


def _manifest_path(artifact_dir: str) -> str:
    return os.path.join(artifact_dir, CHECKPOINT_ARTIFACT_MANIFEST)


# ---------------------------------------------------------------- local manifest ----------------------------------------
def _native_manifest(artifact_dir: str) -> dict[str, Any]:
    """Return the local manifest in the native schema, rewriting a legacy merged-artifact manifest.

    Merged artifacts produced before the merge tool emitted the native schema carry the six native
    keys plus a ``merge`` block, which the validator rejects. Such a manifest is reduced to the
    native keys and rewritten in place (its original bytes are preserved beside it as
    ``LEGACY_MANIFEST_RECORD``), so the archived artifact is readable by every native reader. Any
    other validation failure is an error: this never invents a manifest for a bare payload.
    """
    path = _manifest_path(artifact_dir)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"not a committed checkpoint artifact (no {CHECKPOINT_ARTIFACT_MANIFEST}): {artifact_dir!r}"
        )
    with open(path, "rb") as handle:
        raw = handle.read()
    value = parse_json_object(raw, context=f"checkpoint artifact manifest at {artifact_dir!r}")
    try:
        return validate_artifact_manifest(value, artifact_location=artifact_dir)
    except RuntimeError:
        if not _NATIVE_MANIFEST_KEYS < set(value):
            raise
    components = value["components"]
    plan = value["checkpoint_components"]
    if not isinstance(components, dict) or not isinstance(plan, list):
        raise RuntimeError(f"unsupported posttrain checkpoint artifact manifest at {artifact_dir!r}")
    rebuilt = build_artifact_manifest(artifact_dir, value["checkpoint_format"], tuple(plan), components)
    legacy_path = os.path.join(artifact_dir, LEGACY_MANIFEST_RECORD)
    if not os.path.exists(legacy_path):
        atomic_write_bytes(legacy_path, raw)
    atomic_write_json(path, rebuilt)
    logger.info(f"rewrote legacy artifact manifest to the native schema at {artifact_dir} (original kept beside it)")
    return rebuilt


def _hashed_inventory(artifact_dir: str, manifest: dict[str, Any]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Hash every archived file, prove file components against their descriptors, bind missing digests.

    The archive covers declared physical components plus the merge/provenance records, so a stale
    optimizer left by another save is never published as a selected model-only artifact. A file
    component must exist at its descriptor's
    size; a descriptor digest must match the bytes on disk (a locally corrupted artifact is never
    published); a descriptor without a digest receives one and the manifest is rewritten in place.
    Returns the (possibly rewritten) manifest and ``{relative: {"size", "sha256"}}`` including the
    manifest itself.
    """
    selected = {LEGACY_MANIFEST_RECORD, ".merge-complete.json"}
    merge_record = os.path.join(artifact_dir, ".merge-complete.json")
    if os.path.isfile(merge_record):
        with open(merge_record, "rb") as handle:
            merge = parse_json_object(handle.read(), context=f"merge completion record at {artifact_dir!r}")
        diagnostic = (merge.get("artifacts") or {}).get("diag")
        if isinstance(diagnostic, dict) and isinstance(diagnostic.get("path"), str):
            path = os.path.realpath(diagnostic["path"])
            root = os.path.realpath(artifact_dir)
            if path.startswith(root + os.sep):
                selected.add(os.path.relpath(path, root).replace(os.sep, "/"))
    directories = tuple(
        str(descriptor["path"]) + "/" for descriptor in manifest["components"].values() if descriptor["kind"] == "dcp"
    )
    selected.update(
        str(descriptor["path"]) for descriptor in manifest["components"].values() if descriptor["kind"] == "file"
    )
    files: dict[str, dict[str, Any]] = {}
    for relative, size in sorted(payload_inventory(artifact_dir).items()):
        if relative not in selected and not relative.startswith(directories):
            continue
        files[relative] = {"size": size, "sha256": object_store.file_sha256(os.path.join(artifact_dir, relative))}
    components: dict[str, dict[str, Any]] = {}
    rewrite = False
    for name, descriptor in manifest["components"].items():
        descriptor = dict(descriptor)
        relative = str(descriptor["path"])
        if descriptor["kind"] == "file":
            entry = files.get(relative)
            if entry is None or entry["size"] != descriptor["size"]:
                raise RuntimeError(
                    f"artifact payload {relative!r} is missing or resized against its manifest at {artifact_dir!r}"
                )
            if descriptor.get("sha256") is None:
                descriptor["sha256"] = entry["sha256"]
                rewrite = True
            elif descriptor["sha256"] != entry["sha256"]:
                raise RuntimeError(
                    f"artifact payload {relative!r} does not match its manifest digest at {artifact_dir!r}"
                )
        elif not os.path.isdir(os.path.join(artifact_dir, relative)):
            raise RuntimeError(f"artifact DCP component {relative!r} is missing at {artifact_dir!r}")
        components[name] = descriptor
    if rewrite:
        manifest = build_artifact_manifest(
            artifact_dir,
            manifest["checkpoint_format"],
            tuple(manifest["checkpoint_components"]),
            components,
        )
        atomic_write_json(_manifest_path(artifact_dir), manifest)
    manifest_path = _manifest_path(artifact_dir)
    files[CHECKPOINT_ARTIFACT_MANIFEST] = {
        "size": os.path.getsize(manifest_path),
        "sha256": object_store.file_sha256(manifest_path),
    }
    return manifest, files


def _reserve_inventory(
    client: Any,
    bucket: str,
    prefix: str,
    files: Mapping[str, Mapping[str, Any]],
    manifest_sha256: str,
    *,
    credentials: str,
) -> dict[str, Any]:
    """Bind a prefix permanently to one inventory before any canonical payload key is written.

    A crashed upload is resumed by the same inventory, including concurrent identical retries.
    A different request must choose another iteration. The reservation is never deleted: removing
    it would let a delayed publisher overwrite bytes whose original owner already released locally.
    """
    encoded = json_bytes({"protocol": ARCHIVE_PROTOCOL, "manifest_sha256": manifest_sha256, "files": dict(files)})
    digest = sha256(encoded)
    key = f"{prefix}/{ARCHIVE_RECORD}"
    with tempfile.NamedTemporaryFile(prefix="posttrain-archive-inventory-", suffix=".json") as temporary:
        temporary.write(encoded)
        temporary.flush()
        try:
            object_store.put_object_write_once(
                client,
                bucket,
                key,
                temporary.name,
                {"sha256": digest},
                credentials=credentials,
            )
        except FileExistsError:
            # The store arbitrates the race; only byte-identical reservations may share payload keys.
            pass
    response = client.get_object(Bucket=bucket, Key=key)
    try:
        actual = response["Body"].read()
    finally:
        response["Body"].close()
    status = remote_object_status(client, bucket, key, len(encoded), digest)
    if status == "corrupt":
        raise RuntimeError(f"artifact archive inventory failed SHA-256 verification: s3://{bucket}/{key}")
    if actual != encoded:
        raise FileExistsError(f"refusing to overwrite reserved object-store artifact: s3://{bucket}/{prefix}")
    if status != "verified":
        raise RuntimeError(f"artifact archive inventory failed SHA-256 verification: s3://{bucket}/{key}")
    return {"size": len(encoded), "sha256": digest}


def publish_artifact(artifact_dir: str, upload: dict[str, Any]) -> dict[str, Any]:
    """Publish one native artifact through the shared trainer/offline integrity contract.

    Every file, including DCP shards, is hashed. A permanent inventory declaration reserves the
    destination before payload writes; a conflicting declaration or commit is refused. Identical
    concurrent publishers may write only the same payload bytes. The native manifest commits last,
    followed by live verification and a local receipt. This function never deletes local files.
    """
    artifact_dir = os.path.abspath(artifact_dir)
    name = iteration_name(artifact_dir)
    _manifest, payload_files = _hashed_inventory(artifact_dir, _native_manifest(artifact_dir))
    try:
        manifest_entry = payload_files[CHECKPOINT_ARTIFACT_MANIFEST]
        manifest_sha256 = str(manifest_entry["sha256"])
        destination, identity = artifact_destination(upload, name)
        bucket = str(upload["bucket"])
        credentials = str(upload.get("credentials") or object_store.DEFAULT_CREDENTIALS)
        client = object_store.build_s3_client(credentials)
        manifest_key = f"{destination}/{CHECKPOINT_ARTIFACT_MANIFEST}"
        manifest_status = remote_object_status(
            client, bucket, manifest_key, int(manifest_entry["size"]), manifest_sha256
        )
        if manifest_status not in ("absent", "verified"):
            raise FileExistsError(
                f"refusing to overwrite committed object-store artifact s3://{bucket}/{destination}: "
                f"its manifest is {manifest_status} from the local one"
            )
        if manifest_status == "verified":
            problems = unverified_objects(remote_inventory_status(client, bucket, destination, payload_files))
            if problems:
                raise RuntimeError(
                    f"committed archive s3://{bucket}/{destination} does not verify ({'; '.join(problems)})"
                )

        files = dict(payload_files)
        files[ARCHIVE_RECORD] = _reserve_inventory(
            client,
            bucket,
            destination,
            payload_files,
            manifest_sha256,
            credentials=credentials,
        )
        marker = read_marker(artifact_dir)
        if manifest_status == "verified":
            status = (
                "skipped"
                if marker is not None and marker.destination == identity and marker.files == files
                else "adopted"
            )
        else:
            payloads = [relative for relative in sorted(payload_files) if relative != CHECKPOINT_ARTIFACT_MANIFEST]
            try:
                object_store.upload_prefix(
                    artifact_dir,
                    bucket,
                    destination,
                    credentials=credentials,
                    relative_paths=[*payloads, CHECKPOINT_ARTIFACT_MANIFEST],
                    write_once_commit=CHECKPOINT_ARTIFACT_MANIFEST,
                    object_metadata={
                        relative: {"sha256": str(entry["sha256"])} for relative, entry in payload_files.items()
                    },
                    verify_sha256=True,
                )
                status = "published"
            except FileExistsError:
                # Another publisher of this same reserved inventory may have committed first. Re-read
                # every object below before adopting its commit; a merely present manifest proves nothing.
                status = "adopted"
        problems = unverified_objects(remote_inventory_status(client, bucket, destination, files))
        if problems:
            raise RuntimeError(
                f"archive s3://{bucket}/{destination} changed under this publication ({'; '.join(problems)}): "
                "another writer touched the prefix; no marker written, local artifact retained"
            )
    finally:
        # Hashing and uploading streamed every payload through the page cache. Whether the publication
        # succeeded or a reservation, upload or remote-verification step raised (the manager treats
        # that as best effort and keeps the local artifact), nothing here reads the bytes again:
        # release them from the job cgroup's accounting either way. Best effort, never fatal.
        for relative in sorted(payload_files):
            release_page_cache(os.path.join(artifact_dir, relative))
    write_marker(artifact_dir, destination=identity, files=files, manifest_sha256=manifest_sha256)
    logger.info(f"artifact {status}: {artifact_dir} -> s3://{bucket}/{destination} ({len(files)} objects)")
    return {
        "status": status,
        "bucket": bucket,
        "prefix": destination,
        "destination": identity,
        "manifest_sha256": manifest_sha256,
        "files": files,
        "removed": [],
    }


def archive_checkpoint(
    iter_dir: str,
    upload: dict[str, Any],
    *,
    overrides: dict[str, str] | None = None,
    commit: dict[str, Any] | None = None,
) -> None:
    """Archive a full checkpoint, optionally with a validation-complete trainer override.

    The full-checkpoint generation contract avoids hashing bulk model and optimizer state.
    Validation overrides carry digests and commit last through the existing transactional transport.
    The marker describes archived override bytes, not the pending local trainer state. Failure is
    always raised; the training caller owns the best-effort versus strict transaction policy.
    """
    if (overrides is None) != (commit is None):
        raise ValueError("transactional checkpoint upload requires both overrides and commit metadata")
    name = os.path.basename(os.path.abspath(iter_dir))
    destination = upload["prefix"].rstrip("/") + "/" + name
    identity = object_store.upload_destination_identity(upload, name)
    credentials = upload.get("credentials") or object_store.DEFAULT_CREDENTIALS
    if overrides is not None and commit is not None:
        manifest_name = commit.get("manifest")
        if not isinstance(manifest_name, str):
            raise ValueError("transactional checkpoint upload commit requires a manifest filename")
        # Commit last: the completed validation frame follows every canonical/override object.
        object_store.upload_prefix_transactional(
            iter_dir,
            upload["bucket"],
            destination,
            credentials=credentials,
            overrides=overrides,
            commit_manifest=commit,
            commit_path=manifest_name,
        )
    else:
        object_store.upload_prefix(iter_dir, upload["bucket"], destination, credentials=credentials)
    files: dict[str, dict[str, Any]] = {
        relative: {"size": size, "sha256": None} for relative, size in payload_inventory(iter_dir).items()
    }
    for relative, source in (overrides or {}).items():
        files[relative.replace(os.sep, "/")] = {
            "size": os.path.getsize(source),
            "sha256": object_store.file_sha256(source),
        }
    write_marker(iter_dir, destination=identity, files=files)
