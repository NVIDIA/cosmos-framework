# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Primary checkpoint stores built on local IO and the built-in S3 DCP support."""

from __future__ import annotations

import io
import os
import posixpath
import uuid
from abc import ABC, abstractmethod
from collections.abc import Mapping
from functools import cached_property
from typing import Any
from urllib.parse import urlsplit

from cosmos_framework.posttrain.checkpoint.artifact import (
    ARTIFACT_CONTROL_ROOT,
    CHECKPOINT_ARTIFACT_MANIFEST,
    build_artifact_manifest,
    commit_local_artifact,
    local_artifact_component_descriptor,
    local_artifact_location,
    read_local_artifact_manifest,
    validate_artifact_manifest,
)
from cosmos_framework.posttrain.checkpoint.io import (
    LATEST_FILE,
    component_kind,
    load_component,
    load_component_dcp,
    resolve_resume,
    save_component_dcp,
)
from cosmos_framework.posttrain.checkpoint.manifest import (
    CONTROL_ROOT,
    MANIFEST_PROTOCOL,
    POSTTRAIN_MANIFEST,
    VALIDATION_PUBLICATION_PROTOCOL,
    CommittedManifest,
    dcp_tensor_inventory,
    is_canonical_relative_path,
    iteration_name,
    iteration_number,
    json_bytes,
    parse_json_object,
    sha256,
    validate_manifest,
    validate_publication_record,
    validate_visibility_commit,
)
from cosmos_framework.posttrain.checkpoint.provenance import checkpoint_provenance, replace_component_atomic
from cosmos_framework.posttrain.utils.dist import rank_identical, run_rank_converged


def _torch_bytes(state: Mapping[str, Any]) -> bytes:
    """Serialize one small control component without creating a local file."""
    import torch

    stream = io.BytesIO()
    torch.save(dict(state), stream)
    return stream.getvalue()


def _load_torch_bytes(payload: bytes) -> dict[str, Any]:
    """Deserialize one trusted checkpoint control component."""
    import torch

    value = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)  # [*] control tensor values
    if not isinstance(value, dict):
        raise RuntimeError(f"checkpoint control component must be a dictionary, got {type(value).__name__}")
    return value


def _is_bucket_name(value: Any) -> bool:
    """Return whether an authority is one unambiguous S3 bucket name."""
    alphanumeric = "abcdefghijklmnopqrstuvwxyz0123456789"
    return (
        isinstance(value, str)
        and bool(value)
        and value[0] in alphanumeric
        and value[-1] in alphanumeric
        and ".." not in value
        and all(character in alphanumeric + ".-" for character in value)
    )


def _split_s3_uri(uri: str, *, allow_root: bool = False) -> tuple[str, str]:
    """Parse one canonical S3-compatible URI into bucket and key."""
    parsed = urlsplit(uri)
    if (
        parsed.scheme != "s3"
        or not _is_bucket_name(parsed.netloc)
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(f"invalid object-store checkpoint URI: {uri!r}")
    key = parsed.path.lstrip("/")
    if parsed.path.startswith("//") or "\\" in key or "%" in key:
        raise ValueError(f"invalid object-store checkpoint URI: {uri!r}")
    if key and any(segment in ("", ".", "..") for segment in key.split("/")):
        raise ValueError(f"invalid object-store checkpoint URI: {uri!r}")
    if not key and not allow_root:
        raise ValueError(f"invalid object-store checkpoint URI: {uri!r}")
    return parsed.netloc, key


def _component_name(name: str, *, kind: str) -> str:
    if not name or posixpath.basename(name) != name:
        raise ValueError(f"{kind} component name must be one path segment, got {name!r}")
    return name


class CheckpointStore(ABC):
    """Backend-neutral primary checkpoint store owned by :class:`CheckpointManager`."""

    backend: str

    @property
    def is_object_store(self) -> bool:
        """Return whether the durable primary copy lives in object storage."""
        return self.backend == "object_store"

    @abstractmethod
    def iteration_location(self, iter_hint: str) -> str:
        """Resolve a trainer iteration hint to this store's canonical location."""

    @abstractmethod
    def artifact_location(self, iter_hint: str) -> str:
        """Resolve an iteration hint to the sibling non-resumable artifact namespace."""

    @abstractmethod
    def save_dcp_component(self, iter_hint: str, name: str, state: dict[str, Any]) -> None:
        """Collectively save one DCP component."""

    @abstractmethod
    def load_dcp_component(
        self,
        iter_location: str,
        name: str,
        state: dict[str, Any],
        *,
        allow_partial: bool = False,
    ) -> bool:
        """Collectively load one DCP component when present."""

    @abstractmethod
    def has_dcp_component(self, iter_location: str, name: str) -> bool:
        """Return whether a committed checkpoint contains one DCP component."""

    @abstractmethod
    def dcp_tensor_inventory(self, iter_location: str, name: str) -> dict[str, tuple[int, ...] | None]:
        """Return logical DCP keys and tensor shapes for one component."""

    @abstractmethod
    def load_component(self, iter_location: str, name: str) -> dict[str, Any] | None:
        """Load one small control component when present."""

    @abstractmethod
    def resolve_resume(self, resume: str, ckpt_root_hint: str) -> str | None:
        """Resolve an auto or explicit exact-resume request."""

    @abstractmethod
    def checkpoint_provenance(self, iter_location: str) -> dict[str, int | str]:
        """Return a stable non-secret identity for one complete checkpoint."""

    @abstractmethod
    def replace_component(self, iter_location: str, name: str, state: dict[str, Any]) -> None:
        """Atomically advance one small control component."""

    def commit_iteration(
        self,
        iter_location: str,
        bulk_component_names: list[str],
        control_components: Mapping[str, dict[str, Any] | None],
        *,
        update_latest: bool,
    ) -> None:
        """Commit a complete direct-primary iteration and optionally advance latest."""
        raise NotImplementedError(f"{type(self).__name__} does not commit direct-primary iterations")

    def finish_committed_iteration(
        self,
        iter_location: str,
        bulk_component_names: list[str],
        control_component_names: list[str],
        *,
        update_latest: bool,
    ) -> bool:
        """Finish the latest-pointer update when an immutable iteration is already committed."""
        del iter_location, bulk_component_names, control_component_names, update_latest
        return False

    def ensure_artifact_uncommitted(self, artifact_location: str) -> None:
        """Fail when an artifact manifest already commits this iteration."""
        del artifact_location

    def commit_artifact(
        self,
        artifact_location: str,
        checkpoint_format: str,
        checkpoint_components: tuple[str, ...],
        dcp_component_names: list[str],
        file_components: Mapping[str, dict[str, Any] | None],
    ) -> None:
        """Write remaining artifact state and its independent commit marker last."""
        del artifact_location, checkpoint_format, checkpoint_components, dcp_component_names, file_components
        raise NotImplementedError(f"{type(self).__name__} does not commit checkpoint artifacts")

    def publication_identity(self, iter_location: str) -> dict[str, Any]:
        """Return a non-secret identity for validation publication."""
        raise NotImplementedError(f"{type(self).__name__} has no direct-primary publication identity")

    def publish_validation(
        self,
        iter_location: str,
        *,
        completed_trainer_state: dict[str, Any],
        publication_record: Mapping[str, Any],
    ) -> None:
        """Commit validation visibility and advance the completed control generation."""
        raise NotImplementedError(f"{type(self).__name__} does not publish direct-primary validation")


class LocalCheckpointStore(CheckpointStore):
    """Existing local checkpoint behavior behind the common store contract."""

    backend: str = "local"

    def iteration_location(self, iter_hint: str) -> str:
        return iter_hint

    def artifact_location(self, iter_hint: str) -> str:
        return local_artifact_location(iter_hint)

    def ensure_artifact_uncommitted(self, artifact_location: str) -> None:
        manifest_path = os.path.join(artifact_location, CHECKPOINT_ARTIFACT_MANIFEST)
        if not os.path.isfile(manifest_path):
            return
        read_local_artifact_manifest(artifact_location)
        raise FileExistsError(
            f"refusing to overwrite committed checkpoint artifact without content identity: {artifact_location!r}"
        )

    def commit_artifact(
        self,
        artifact_location: str,
        checkpoint_format: str,
        checkpoint_components: tuple[str, ...],
        dcp_component_names: list[str],
        file_components: Mapping[str, dict[str, Any] | None],
    ) -> None:
        commit_local_artifact(
            artifact_location,
            checkpoint_format,
            checkpoint_components,
            dcp_component_names,
            file_components,
        )

    def save_dcp_component(self, iter_hint: str, name: str, state: dict[str, Any]) -> None:
        save_component_dcp(iter_hint, name, state)

    def load_dcp_component(
        self,
        iter_location: str,
        name: str,
        state: dict[str, Any],
        *,
        allow_partial: bool = False,
    ) -> bool:
        is_artifact, descriptor = local_artifact_component_descriptor(iter_location, name)
        if is_artifact and (descriptor is None or descriptor["kind"] != "dcp"):
            return False
        return load_component_dcp(iter_location, name, state, allow_partial=allow_partial)

    def has_dcp_component(self, iter_location: str, name: str) -> bool:
        is_artifact, descriptor = local_artifact_component_descriptor(iter_location, name)
        if is_artifact:
            return descriptor is not None and descriptor["kind"] == "dcp"
        return component_kind(iter_location, name) == "dcp"

    def dcp_tensor_inventory(self, iter_location: str, name: str) -> dict[str, tuple[int, ...] | None]:
        from torch.distributed.checkpoint import FileSystemReader

        component_path = os.path.join(iter_location, name)
        if not self.has_dcp_component(iter_location, name):
            raise FileNotFoundError(f"DCP component {name!r} does not exist at {iter_location!r}")
        metadata_path = os.path.join(component_path, ".metadata")
        metadata = FileSystemReader(component_path).read_metadata()
        return dcp_tensor_inventory(metadata, metadata_path)

    def load_component(self, iter_location: str, name: str) -> dict[str, Any] | None:
        is_artifact, descriptor = local_artifact_component_descriptor(iter_location, name)
        if is_artifact and (descriptor is None or descriptor["kind"] != "file"):
            return None
        return load_component(iter_location, name)

    def resolve_resume(self, resume: str, ckpt_root_hint: str) -> str | None:
        resolved = resolve_resume(resume, ckpt_root_hint)
        if resolved is not None and os.path.isfile(os.path.join(resolved, CHECKPOINT_ARTIFACT_MANIFEST)):
            raise ValueError(
                f"checkpoint artifact {resolved!r} is not exact-resume capable; use it for model initialization or eval"
            )
        return resolved

    def checkpoint_provenance(self, iter_location: str) -> dict[str, int | str]:
        if os.path.isfile(os.path.join(iter_location, CHECKPOINT_ARTIFACT_MANIFEST)):
            raise ValueError(
                f"checkpoint artifact {iter_location!r} is not exact-resume capable; "
                "use it for model initialization or eval"
            )
        return checkpoint_provenance(iter_location)

    def replace_component(self, iter_location: str, name: str, state: dict[str, Any]) -> None:
        replace_component_atomic(iter_location, name, state)


class ObjectStoreCheckpointStore(CheckpointStore):
    """Direct-primary DCP using DCP readers/writers and EasyIO control objects.

    The protocol intentionally matches the upstream single-writer model: partial iteration prefixes may
    remain after a failed save, while the manifest and ``latest_checkpoint.txt`` are written last.
    """

    backend: str = "object_store"

    def __init__(self, *, bucket: str, prefix: str, credentials: str) -> None:
        if not _is_bucket_name(bucket):
            raise ValueError(f"object-store checkpoint bucket must be one non-empty name, got {bucket!r}")
        if not credentials:
            raise ValueError("object-store checkpoint credentials must be non-empty")
        if prefix.startswith("s3://"):
            raise ValueError("object-store checkpoint prefix must be bucket-relative, not an s3:// URI")
        normalized_prefix = prefix.strip("/")
        if normalized_prefix and (
            "\\" in normalized_prefix
            or "%" in normalized_prefix
            or "?" in normalized_prefix
            or "#" in normalized_prefix
            or any(segment in ("", ".", "..") for segment in normalized_prefix.split("/"))
        ):
            raise ValueError(f"object-store checkpoint prefix is not canonical: {prefix!r}")
        self.bucket = bucket
        self.prefix = normalized_prefix
        self.credentials = credentials
        self.root_uri = f"s3://{bucket}" + (f"/{normalized_prefix}" if normalized_prefix else "")
        root_parent = posixpath.dirname(normalized_prefix)
        artifact_prefix = posixpath.join(root_parent, "checkpoint_artifacts")
        self.artifact_root_uri = f"s3://{bucket}/{artifact_prefix}"

    @cached_property
    def _io(self) -> Any:
        """Reuse EasyIO for the small control plane."""
        from cosmos_framework.utils.easy_io import easy_io

        return easy_io.get_file_backend(
            backend_args={
                "backend": "s3",
                "s3_credential_path": self.credentials,
                "path_mapping": None,
            }
        )

    def _object_uri(self, iter_location: str, relative: str) -> str:
        if not is_canonical_relative_path(relative):
            raise ValueError(f"checkpoint-relative object path is invalid: {relative!r}")
        return self.iteration_location(iter_location).rstrip("/") + "/" + relative

    def _read(self, uri: str) -> bytes:
        payload = self._io.get(uri)
        return bytes(payload)

    def _read_optional(self, uri: str) -> bytes | None:
        return self._read(uri) if self._io.exists(uri) else None

    def _put(self, uri: str, payload: bytes) -> None:
        self._io.put(payload, uri)

    def _writable_iteration_location(self, iter_hint: str) -> str:
        """Resolve one mutation target and prove it is below the configured root."""
        name = iteration_name(iter_hint)
        expected = f"{self.root_uri}/{name}"
        location = self.iteration_location(iter_hint)
        if location != expected:
            raise ValueError(
                f"checkpoint mutation target {location!r} is outside configured store root {self.root_uri!r}"
            )
        return expected

    def _writable_artifact_location(self, iter_hint: str) -> str:
        """Resolve one artifact mutation target below its independent sibling root."""
        name = iteration_name(iter_hint)
        expected = f"{self.artifact_root_uri}/{name}"
        location = self.artifact_location(iter_hint)
        if location != expected:
            raise ValueError(
                f"checkpoint artifact mutation target {location!r} is outside configured artifact root "
                f"{self.artifact_root_uri!r}"
            )
        return expected

    def _manifest_uri(self, iter_location: str) -> str:
        return self._object_uri(iter_location, POSTTRAIN_MANIFEST)

    def _artifact_manifest_uri(self, artifact_location: str) -> str:
        return self._object_uri(artifact_location, CHECKPOINT_ARTIFACT_MANIFEST)

    def _read_manifest(self, iter_location: str) -> CommittedManifest:
        location = self.iteration_location(iter_location)
        encoded = self._read(self._manifest_uri(location))
        manifest = validate_manifest(
            parse_json_object(encoded, context=f"posttrain checkpoint manifest at {location!r}"),
            iter_location=location,
        )
        publication = manifest.get("validation_publication")
        if publication is not None:
            commit_uri = self._object_uri(location, str(publication["manifest"]))
            commit = parse_json_object(self._read(commit_uri), context="validation publication commit")
            validate_visibility_commit(commit, record=publication, active_manifest=manifest)
        return CommittedManifest(value=manifest, encoded=encoded, digest=sha256(encoded))

    def _write_manifest(self, iter_location: str, manifest: Mapping[str, Any]) -> None:
        location = self._writable_iteration_location(iter_location)
        validate_manifest(manifest, iter_location=location)
        payload = json_bytes(manifest)
        uri = self._manifest_uri(location)
        self._put(uri, payload)
        if self._read(uri) != payload:
            raise RuntimeError(f"object-store checkpoint manifest write verification failed: {location!r}")

    def _read_optional_artifact_manifest(self, artifact_location: str) -> dict[str, Any] | None:
        """Read one artifact commit marker when present."""
        location = (
            artifact_location.rstrip("/")
            if artifact_location.startswith("s3://")
            else self.artifact_location(artifact_location)
        )
        encoded = self._read_optional(self._artifact_manifest_uri(location))
        if encoded is None:
            return None
        return validate_artifact_manifest(
            parse_json_object(encoded, context=f"checkpoint artifact manifest at {location!r}"),
            artifact_location=location,
        )

    def _artifact_component_descriptor(
        self,
        artifact_location: str,
        name: str,
    ) -> tuple[bool, Mapping[str, Any] | None]:
        """Return whether a path is an artifact and its descriptor for ``name``."""
        manifest = self._read_optional_artifact_manifest(artifact_location)
        return manifest is not None, None if manifest is None else manifest["components"].get(name)

    def _write_artifact_manifest(self, artifact_location: str, manifest: Mapping[str, Any]) -> None:
        location = self._writable_artifact_location(artifact_location)
        validate_artifact_manifest(manifest, artifact_location=location)
        payload = json_bytes(manifest)
        uri = self._artifact_manifest_uri(location)
        self._put(uri, payload)
        if self._read(uri) != payload:
            raise RuntimeError(f"object-store checkpoint artifact manifest verification failed: {location!r}")

    def _new_reader(self, iter_location: str, name: str) -> Any:
        from cosmos_framework.checkpoint.s3_filesystem import S3StorageReader

        return S3StorageReader(
            credential_path=self.credentials,
            path=self._object_uri(iter_location, name),
            enable_gcs_patch_in_boto3=True,
        )

    def _read_dcp_metadata(self, iter_location: str, name: str) -> Any:
        reader = self._new_reader(iter_location, name)
        return reader.read_metadata()

    def _write_file_payload(
        self,
        location: str,
        relative: str,
        payload: bytes,
        *,
        kind: str,
    ) -> dict[str, int | str]:
        """Write and verify one append-only file component."""
        uri = self._object_uri(location, relative)
        if self._io.exists(uri):
            raise FileExistsError(f"{kind} already exists: {uri!r}")
        self._put(uri, payload)
        if self._read(uri) != payload:
            raise RuntimeError(f"{kind} write verification failed: {uri!r}")
        return {"path": relative, "size": len(payload), "sha256": sha256(payload)}

    def _write_control_payload(self, iter_location: str, relative: str, payload: bytes) -> dict[str, int | str]:
        location = self._writable_iteration_location(iter_location)
        return self._write_file_payload(location, relative, payload, kind="checkpoint control generation")

    def _read_control_payload(self, iter_location: str, descriptor: Mapping[str, Any], name: str) -> bytes:
        payload = self._read(self._object_uri(iter_location, str(descriptor["path"])))
        if len(payload) != descriptor["size"] or sha256(payload) != descriptor["sha256"]:
            raise RuntimeError(f"object-store checkpoint control component failed verification: {name!r}")
        return payload

    def _stage_control_successor(
        self,
        iter_location: str,
        manifest: Mapping[str, Any],
        name: str,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        """Write one append-only control generation without advancing the manifest pointer."""
        generation = uuid.uuid4().hex
        controls: dict[str, dict[str, int | str]] = {}
        current = manifest.get("control_components")
        if not isinstance(current, Mapping):
            raise RuntimeError(f"checkpoint manifest has no control component map: {iter_location!r}")
        for component, descriptor in current.items():
            if not isinstance(component, str) or not isinstance(descriptor, Mapping):
                raise RuntimeError(f"checkpoint manifest has an invalid control descriptor: {iter_location!r}")
            payload = (
                _torch_bytes(state)
                if component == name
                else self._read_control_payload(
                    iter_location,
                    descriptor,
                    component,
                )
            )
            relative = f"{CONTROL_ROOT}/{generation}/{component}.pt"
            controls[component] = self._write_control_payload(iter_location, relative, payload)
        if name not in controls:
            relative = f"{CONTROL_ROOT}/{generation}/{name}.pt"
            controls[name] = self._write_control_payload(iter_location, relative, _torch_bytes(state))
        successor = dict(manifest)
        successor["control_generation"] = generation
        successor["control_components"] = controls
        return successor

    def _write_latest(self, iter_location: str) -> None:
        location = self._writable_iteration_location(iter_location)
        root = posixpath.dirname(location)
        uri = f"{root}/{LATEST_FILE}"
        payload = (iteration_name(location) + "\n").encode("utf-8")
        if self._read_optional(uri) == payload:
            return
        self._put(uri, payload)
        if self._read(uri) != payload:
            raise RuntimeError(f"object-store latest pointer write verification failed: {uri!r}")

    def iteration_location(self, iter_hint: str) -> str:
        name = iteration_name(iter_hint)
        if iter_hint.startswith("s3://"):
            location = iter_hint.rstrip("/")
            _split_s3_uri(location)
            return location
        return f"{self.root_uri}/{name}"

    def artifact_location(self, iter_hint: str) -> str:
        name = iteration_name(iter_hint)
        if iter_hint.startswith("s3://"):
            location = iter_hint.rstrip("/")
            _split_s3_uri(location)
            if posixpath.dirname(location) == self.root_uri:
                return f"{self.artifact_root_uri}/{name}"
            return location
        return f"{self.artifact_root_uri}/{name}"

    def ensure_artifact_uncommitted(self, artifact_location: str) -> None:
        location = self._writable_artifact_location(artifact_location)
        if self._read_optional_artifact_manifest(location) is None:
            return
        raise FileExistsError(
            f"refusing to overwrite committed checkpoint artifact without content identity: {location!r}"
        )

    def commit_artifact(
        self,
        artifact_location: str,
        checkpoint_format: str,
        checkpoint_components: tuple[str, ...],
        dcp_component_names: list[str],
        file_components: Mapping[str, dict[str, Any] | None],
    ) -> None:
        """Write artifact controls, then expose the independent artifact manifest."""
        location = self._writable_artifact_location(artifact_location)
        if checkpoint_format != "dcp":
            raise ValueError("direct object-store checkpoint artifacts require DCP")
        if self._read_optional_artifact_manifest(location) is not None:
            raise FileExistsError(f"refusing to replace committed checkpoint artifact: {location!r}")
        components: dict[str, dict[str, Any]] = {}
        for name in sorted(dcp_component_names):
            name = _component_name(name, kind="DCP")
            self._read_dcp_metadata(location, name)
            components[name] = {"kind": "dcp", "path": name}
        generation = uuid.uuid4().hex
        for name, state in file_components.items():
            if state is None:
                continue
            name = _component_name(name, kind="checkpoint artifact")
            relative = f"{ARTIFACT_CONTROL_ROOT}/{generation}/{name}.pt"
            payload = _torch_bytes(state)
            components[name] = {
                "kind": "file",
                **self._write_file_payload(location, relative, payload, kind="checkpoint artifact component"),
            }
        if not components:
            raise RuntimeError("checkpoint artifact contains no physical components")
        manifest = build_artifact_manifest(
            location,
            checkpoint_format,
            checkpoint_components,
            components,
        )
        self._write_artifact_manifest(location, manifest)

    def finish_committed_iteration(
        self,
        iter_location: str,
        bulk_component_names: list[str],
        control_component_names: list[str],
        *,
        update_latest: bool,
    ) -> bool:
        """Treat a valid manifest as the immutable full-save commit marker.

        A retry after an ambiguous manifest/latest response must never re-enter DCP writers beneath
        a manifest that readers can already observe. Structural plan drift fails instead of
        replacing a committed iteration; validation changes use the control-successor APIs.
        """
        location = self._writable_iteration_location(iter_location)
        if self._read_optional(self._manifest_uri(location)) is None:
            return False
        manifest = self._read_manifest(location).value
        expected_bulk = sorted({_component_name(name, kind="DCP") for name in bulk_component_names})
        expected_controls = sorted(
            {_component_name(name, kind="checkpoint control") for name in control_component_names}
        )
        actual_controls = sorted(manifest["control_components"])
        if manifest["bulk_components"] != expected_bulk or actual_controls != expected_controls:
            raise RuntimeError(
                "committed object-store checkpoint component plan differs from same-iteration retry: "
                f"bulk={manifest['bulk_components']!r}/{expected_bulk!r}, "
                f"controls={actual_controls!r}/{expected_controls!r}"
            )
        for name in expected_bulk:
            self._read_dcp_metadata(location, name)
        for name, descriptor in manifest["control_components"].items():
            self._read_control_payload(location, descriptor, name)
        if update_latest:
            self._write_latest(location)
        return True

    def save_dcp_component(self, iter_hint: str, name: str, state: dict[str, Any]) -> None:
        name = _component_name(name, kind="DCP")

        def _writer() -> Any:
            from cosmos_framework.checkpoint.s3_filesystem import S3StorageWriter

            explicit = iter_hint.rstrip("/")
            location = (
                self._writable_artifact_location(explicit)
                if explicit.startswith(self.artifact_root_uri + "/")
                else self._writable_iteration_location(explicit)
            )
            return S3StorageWriter(
                credential_path=self.credentials,
                path=self._object_uri(location, name),
                enable_gcs_patch_in_boto3=True,
            )

        writer = run_rank_converged(_writer, f"object-store DCP save preflight ({name})")
        import torch.distributed.checkpoint as dcp

        dcp.save(state, storage_writer=writer)

    def load_dcp_component(
        self,
        iter_location: str,
        name: str,
        state: dict[str, Any],
        *,
        allow_partial: bool = False,
    ) -> bool:
        name = _component_name(name, kind="DCP")

        def _reader() -> tuple[bool, Any | None, Any | None]:
            is_artifact, descriptor = self._artifact_component_descriptor(iter_location, name)
            if is_artifact:
                if descriptor is None or descriptor["kind"] != "dcp":
                    return False, None, None
            elif name not in self._read_manifest(iter_location).value["bulk_components"]:
                return False, None, None
            reader = self._new_reader(iter_location, name)
            reader.read_metadata()
            if not allow_partial:
                return True, reader, None
            from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner

            return True, reader, DefaultLoadPlanner(allow_partial_load=True)

        present, reader, planner = run_rank_converged(_reader, f"object-store DCP load preflight ({name})")
        present = rank_identical(present, f"object-store DCP component presence ({name})")
        if not present:
            return False
        if reader is None:
            raise AssertionError("object-store DCP load preflight converged without a storage reader")
        import torch.distributed.checkpoint as dcp

        if planner is None:
            dcp.load(state, storage_reader=reader)  # [*] sharded state tensor values
        else:
            dcp.load(state, storage_reader=reader, planner=planner)  # [*] sharded state tensor values
        return True

    def has_dcp_component(self, iter_location: str, name: str) -> bool:
        name = _component_name(name, kind="DCP")
        is_artifact, descriptor = self._artifact_component_descriptor(iter_location, name)
        if is_artifact:
            return descriptor is not None and descriptor["kind"] == "dcp"
        return name in self._read_manifest(iter_location).value["bulk_components"]

    def dcp_tensor_inventory(self, iter_location: str, name: str) -> dict[str, tuple[int, ...] | None]:
        name = _component_name(name, kind="DCP")
        if not self.has_dcp_component(iter_location, name):
            raise FileNotFoundError(f"DCP component {name!r} does not exist at {iter_location!r}")
        metadata_path = self._object_uri(iter_location, f"{name}/.metadata")
        metadata = self._read_dcp_metadata(iter_location, name)
        return dcp_tensor_inventory(metadata, metadata_path)

    def commit_iteration(
        self,
        iter_location: str,
        bulk_component_names: list[str],
        control_components: Mapping[str, dict[str, Any] | None],
        *,
        update_latest: bool,
    ) -> None:
        """Write controls, then the control manifest, and finally latest."""
        location = self._writable_iteration_location(iter_location)
        if self._read_optional(self._manifest_uri(location)) is not None:
            raise FileExistsError(f"refusing to replace committed object-store checkpoint: {location!r}")
        bulk = sorted({_component_name(name, kind="DCP") for name in bulk_component_names})
        if not {"model", "optim"}.issubset(bulk) or control_components.get("trainer") is None:
            raise RuntimeError("object-store DCP checkpoint requires model, optim, and trainer components")
        for name in bulk:
            self._read_dcp_metadata(location, name)

        generation = uuid.uuid4().hex
        controls: dict[str, dict[str, int | str]] = {}
        for name, state in control_components.items():
            if state is None:
                continue
            name = _component_name(name, kind="checkpoint control")
            relative = f"{CONTROL_ROOT}/{generation}/{name}.pt"
            controls[name] = self._write_control_payload(location, relative, _torch_bytes(state))
        name = iteration_name(location)
        manifest: dict[str, Any] = {
            "protocol": MANIFEST_PROTOCOL,
            "iteration": iteration_number(name),
            "iteration_name": name,
            "checkpoint_format": "dcp",
            "bulk_components": bulk,
            "control_generation": generation,
            "control_components": controls,
            "validation_publication": None,
        }
        self._write_manifest(location, manifest)
        if update_latest:
            self._write_latest(location)

    def load_component(self, iter_location: str, name: str) -> dict[str, Any] | None:
        name = _component_name(name, kind="checkpoint control")
        is_artifact, descriptor = self._artifact_component_descriptor(iter_location, name)
        if is_artifact:
            if descriptor is not None and descriptor["kind"] != "file":
                return None
        else:
            manifest = self._read_manifest(iter_location).value
            descriptor = manifest["control_components"].get(name)
        if descriptor is None:
            return None
        return _load_torch_bytes(self._read_control_payload(iter_location, descriptor, name))

    def _checkpoint_root(self, ckpt_root_hint: str) -> str:
        root = ckpt_root_hint.rstrip("/") if ckpt_root_hint.startswith("s3://") else self.root_uri
        _split_s3_uri(root, allow_root=True)
        return root

    def resolve_resume(self, resume: str, ckpt_root_hint: str) -> str | None:
        if not resume or resume == "off":
            return None
        if resume == "auto":
            root = self._checkpoint_root(ckpt_root_hint)
            latest = self._read_optional(f"{root}/{LATEST_FILE}")
            if latest is None:
                return None
            name = latest.decode("utf-8").strip()
            if name != iteration_name(name):
                raise RuntimeError(f"object-store latest pointer is not one iteration name: {name!r}")
            iteration_number(name)
            location = f"{root}/{name}"
        else:
            if not resume.startswith("s3://"):
                raise FileNotFoundError(
                    f"rl.resume={resume!r} is not an object-store iter_* URI; direct-primary resume "
                    "requires 'auto', 'off', or s3://.../iter_*"
                )
            location = self.iteration_location(resume)
            if self._read_optional_artifact_manifest(location) is not None:
                raise ValueError(
                    f"checkpoint artifact {location!r} is not exact-resume capable; "
                    "use it for model initialization or eval"
                )
        self._read_manifest(location)
        return location

    def checkpoint_provenance(self, iter_location: str) -> dict[str, int | str]:
        location = self.iteration_location(iter_location)
        if self._read_optional_artifact_manifest(location) is not None:
            raise ValueError(
                f"checkpoint artifact {location!r} is not exact-resume capable; use it for model initialization or eval"
            )
        manifest = self._read_manifest(location)
        return {"version": 1, "source": location, "signature": manifest.digest}

    def publication_identity(self, iter_location: str) -> dict[str, Any]:
        """Identify the current direct-primary destination and control generation."""
        location = self.iteration_location(iter_location)
        provenance = self.checkpoint_provenance(location)
        return {
            "destination": {"kind": "object_store", "uri": location},
            "checkpoint_generation": provenance["signature"],
        }

    def replace_component(self, iter_location: str, name: str, state: dict[str, Any]) -> None:
        """Advance one append-only control generation and then its manifest pointer."""
        name = _component_name(name, kind="checkpoint control")
        location = self._writable_iteration_location(iter_location)
        manifest = self._read_manifest(location).value
        if manifest.get("validation_publication") is not None:
            raise RuntimeError("cannot replace a control component after validation publication is committed")
        self._write_manifest(location, self._stage_control_successor(location, manifest, name, state))

    def publish_validation(
        self,
        iter_location: str,
        *,
        completed_trainer_state: dict[str, Any],
        publication_record: Mapping[str, Any],
    ) -> None:
        """Commit visibility before exposing its validation-complete trainer generation."""
        location = self._writable_iteration_location(iter_location)
        record = validate_publication_record(publication_record, iteration_number(iteration_name(location)))
        expected_destination = {"kind": "object_store", "uri": location}
        if record["archive_required"] or record["destination"] != expected_destination:
            raise ValueError("validation publication destination does not match object-store primary")
        commit_uri = self._object_uri(location, str(record["manifest"]))
        encoded_commit = self._read_optional(commit_uri)
        if encoded_commit is None:
            active = self._read_manifest(location)
            successor = self._stage_control_successor(location, active.value, "trainer", completed_trainer_state)
            successor["validation_publication"] = record
            commit: dict[str, Any] = {
                "protocol": VALIDATION_PUBLICATION_PROTOCOL,
                "record": record,
                "base_manifest_sha256": active.digest,
                "successor_manifest": successor,
            }
            encoded_commit = json_bytes(commit)
            self._put(commit_uri, encoded_commit)
            if self._read(commit_uri) != encoded_commit:
                raise RuntimeError(f"validation publication commit write verification failed: {commit_uri!r}")
        else:
            commit = validate_visibility_commit(
                parse_json_object(encoded_commit, context="validation publication commit"),
                record=record,
            )
        base_digest = commit.get("base_manifest_sha256")
        successor = commit.get("successor_manifest")
        if not isinstance(base_digest, str) or not isinstance(successor, dict):
            raise RuntimeError(f"validation publication commit lacks manifest transition: {commit_uri!r}")
        current = self._read_manifest(location)
        successor_digest = sha256(json_bytes(successor))
        if current.digest == successor_digest:
            return
        if current.digest != base_digest:
            raise RuntimeError(
                "checkpoint manifest changed after validation publication was staged: "
                f"expected={base_digest}, actual={current.digest}"
            )
        self._write_manifest(location, successor)


def build_checkpoint_store(config: Mapping[str, Any] | CheckpointStore | None) -> CheckpointStore:
    """Build one validated primary store from ``rl.checkpoint_store``."""
    if isinstance(config, CheckpointStore):
        return config
    values = dict(config or {"backend": "local"})
    backend = values.get("backend", "local")
    if backend == "local":
        if set(values) - {"backend"}:
            raise ValueError("local checkpoint_store accepts only backend='local'")
        return LocalCheckpointStore()
    if backend != "object_store":
        raise ValueError(f"checkpoint_store.backend must be 'local' or 'object_store', got {backend!r}")
    allowed = {"backend", "bucket", "prefix", "credentials"}
    unexpected = sorted(set(values) - allowed)
    if unexpected:
        raise ValueError(f"object-store checkpoint_store has unsupported fields: {unexpected}")
    for name in ("bucket", "prefix", "credentials"):
        value = values.get(name, "")
        if not isinstance(value, str):
            raise TypeError(f"object-store checkpoint_store {name} must be a string, got {value!r}")
    return ObjectStoreCheckpointStore(
        bucket=values.get("bucket", ""),
        prefix=values.get("prefix", ""),
        credentials=values.get("credentials", ""),
    )
