# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""In-memory boto3-shaped S3 client for offline object-store tests."""

from __future__ import annotations

import hashlib
import io
from collections.abc import Callable
from typing import Any


class FakeS3Client:
    """Stand-in for ``boto3.client("s3")`` with the subset the checkpoint tooling calls.

    Single-part semantics like the GCS interop shim: ``ETag`` is the quoted MD5 of the body and user
    ``Metadata`` round-trips through ``head_object``/``get_object``. Keys planted with
    ``composite=True`` report a multipart-style ETag (``"<md5>-2"``) so the no-trustworthy-checksum
    branch can be exercised. ``corrupt_uploads`` models a store that loses bytes: every uploaded
    body lands one byte short (an empty body lands one byte long) while the caller's metadata,
    including its ``sha256``, is kept verbatim. The object therefore claims the caller's digest but
    fails the post-upload HEAD size check, which is what the digest-bound upload path must catch;
    same-size corruption with echoed metadata is invisible to HEAD by construction and is not what
    this flag models. Objects live in one namespace regardless of bucket; ``uploads`` records every
    ``(bucket, key, extra_args)`` that landed (``upload_file`` extra arguments verbatim,
    ``put_object`` as ``Metadata`` / ``IfNoneMatch`` where given) and ``heads`` every HEAD key so
    tests can assert what was attached and verified.

    Conditional writes are modelled the way S3 answers them: ``put_object(..., IfNoneMatch="*")``
    raises a 412 ``PreconditionFailed`` ``ClientError`` when the key exists and lands nothing.
    ``conditional_puts_supported=False`` instead raises botocore's ``ParamValidationError`` for the
    parameter, the shape an older botocore produces, so the degraded unconditional path can be
    exercised. ``before_put`` is a hook called with the key before every put or upload lands; a test
    installs a concurrent writer there (plant a payload or a commit) to model the race between two
    publishers of one iteration.
    """

    def __init__(self) -> None:
        self.objects: dict[str, tuple[bytes, dict[str, str]]] = {}
        self.composite: set[str] = set()
        self.corrupt_uploads = False
        self.conditional_puts_supported = True
        self.before_put: Callable[[str], None] | None = None
        self.puts = 0
        self.uploads: list[tuple[str, str, dict[str, Any]]] = []
        self.heads: list[str] = []
        self.deletes: list[tuple[str, str]] = []

    def plant(self, key: str, body: bytes, metadata: dict[str, str] | None = None, composite: bool = False) -> None:
        """Place an object directly in the store, bypassing the upload path."""
        self.objects[key] = (bytes(body), dict(metadata or {}))
        (self.composite.add if composite else self.composite.discard)(key)

    def _land(self, key: str, body: bytes, metadata: dict[str, str] | None) -> None:
        """Store one body; ``corrupt_uploads`` changes the length of what lands."""
        body = bytes(body)
        if self.corrupt_uploads:
            body = body[:-1] if body else b"\x00"
        self.objects[key] = (body, {str(name): str(value) for name, value in (metadata or {}).items()})
        self.composite.discard(key)
        self.puts += 1

    def _precondition_failed(self, key: str) -> Exception:
        from botocore.exceptions import ClientError

        return ClientError(
            {
                "Error": {
                    "Code": "PreconditionFailed",
                    "Message": f"At least one of the pre-conditions you specified did not hold: {key}",
                },
                "ResponseMetadata": {"HTTPStatusCode": 412},
            },
            "PutObject",
        )

    def put_object(
        self,
        *,
        Bucket: str,
        Key: str,
        Body: Any,
        Metadata: dict[str, str] | None = None,
        IfNoneMatch: str | None = None,
        **_: object,
    ) -> dict[str, Any]:
        """Store one object; ``IfNoneMatch="*"`` refuses an existing key with a 412 like S3 does."""
        if self.before_put is not None:
            self.before_put(Key)
        extra: dict[str, Any] = {}
        if Metadata:
            extra["Metadata"] = dict(Metadata)
        if IfNoneMatch is not None:
            if not self.conditional_puts_supported:
                from botocore.exceptions import ParamValidationError

                raise ParamValidationError(report='Unknown parameter in input: "IfNoneMatch", must be one of: ...')
            if IfNoneMatch != "*":
                raise NotImplementedError("FakeS3Client models only IfNoneMatch='*'")
            if Key in self.objects:
                raise self._precondition_failed(Key)
            extra["IfNoneMatch"] = IfNoneMatch
        body = Body.read() if hasattr(Body, "read") else Body
        self.uploads.append((Bucket, Key, extra))
        self._land(Key, body, Metadata)
        return {}

    def upload_file(
        self,
        Filename: str,
        Bucket: str,
        Key: str,
        ExtraArgs: dict[str, Any] | None = None,
        Callback: Any = None,
        Config: Any = None,
    ) -> None:
        """Upload one local file, recording the call and honouring ``ExtraArgs["Metadata"]``."""
        del Callback, Config
        if self.before_put is not None:
            self.before_put(Key)
        self.uploads.append((Bucket, Key, dict(ExtraArgs or {})))
        with open(Filename, "rb") as handle:
            self._land(Key, handle.read(), (ExtraArgs or {}).get("Metadata"))

    def _not_found(self, key: str, operation: str) -> Exception:
        # Imported lazily so this fixture imports without botocore; callers that reach a 404 need it,
        # exactly like utils.object_store._require_missing_object does.
        from botocore.exceptions import ClientError

        return ClientError(
            {
                "Error": {"Code": "404", "Message": f"Not Found: {key}"},
                "ResponseMetadata": {"HTTPStatusCode": 404},
            },
            operation,
        )

    def head_object(self, *, Bucket: str, Key: str, **_: object) -> dict[str, Any]:
        """Return size, ETag, and user metadata, or raise a 404 ``ClientError``."""
        del Bucket
        self.heads.append(Key)
        if Key not in self.objects:
            raise self._not_found(Key, "HeadObject")
        body, metadata = self.objects[Key]
        md5 = hashlib.md5(body).hexdigest()
        return {
            "ContentLength": len(body),
            "ETag": f'"{md5}-2"' if Key in self.composite else f'"{md5}"',
            "Metadata": dict(metadata),
        }

    def get_object(self, *, Bucket: str, Key: str, **_: object) -> dict[str, Any]:
        """Return the head fields plus a readable ``Body``."""
        head = self.head_object(Bucket=Bucket, Key=Key)
        return {**head, "Body": io.BytesIO(self.objects[Key][0])}

    def download_file(self, Bucket: str, Key: str, Filename: str, **_: object) -> None:
        """Write one object's bytes to a local file."""
        body = self.get_object(Bucket=Bucket, Key=Key)["Body"].read()
        with open(Filename, "wb") as handle:
            handle.write(body)

    def delete_object(self, *, Bucket: str, Key: str, **_: object) -> dict[str, Any]:
        """Remove one object; deleting an absent key is not an error, as on S3."""
        self.deletes.append((Bucket, Key))
        self.objects.pop(Key, None)
        self.composite.discard(Key)
        return {}

    def list_objects_v2(
        self,
        *,
        Bucket: str,
        Prefix: str = "",
        MaxKeys: int = 1000,
        ContinuationToken: str | None = None,
        **_: object,
    ) -> dict[str, Any]:
        """List objects under ``Prefix`` in key order with S3-style continuation tokens."""
        del Bucket
        keys = sorted(key for key in self.objects if key.startswith(Prefix))
        start = int(ContinuationToken) if ContinuationToken else 0
        page = keys[start : start + MaxKeys]
        response: dict[str, Any] = {
            "Contents": [{"Key": key, "Size": len(self.objects[key][0])} for key in page],
            "KeyCount": len(page),
            "IsTruncated": start + MaxKeys < len(keys),
        }
        if response["IsTruncated"]:
            response["NextContinuationToken"] = str(start + MaxKeys)
        return response
