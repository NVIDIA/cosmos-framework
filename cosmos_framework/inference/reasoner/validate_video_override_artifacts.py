# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate a fingerprinted video-override artifact before Cosmos training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cosmos_framework.inference.reasoner.video_override_artifacts import _media_paths, _sha256, _stable_sha

_NON_CORE_KEYS = {
    "artifact_fingerprint",
    "created_at",
    "integration_commit",
    "complete",
}


def _identity_values(path: Path) -> set[str]:
    values = {str(path.expanduser().absolute())}
    if path.exists():
        values.add(str(path.expanduser().resolve(strict=True)))
    return values


def validate_artifact(
    *,
    override_map: Path,
    manifest_path: Path,
    artifact_fingerprint: str,
    dataset_fingerprint: str,
    model_fingerprint: str,
    processor_fingerprint: str,
    integration_commit: str,
    required_covered_annotations: list[Path],
    verify_file_hashes: bool = True,
) -> dict[str, Any]:
    """Validate self-consistency, provenance, files, and forced coverage."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    overrides = json.loads(override_map.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(overrides, dict):
        raise ValueError("override manifest and map must contain JSON objects")
    if manifest.get("schema_version") != 2:
        raise ValueError("override manifest schema_version must be 2")
    if manifest.get("complete") is not True:
        raise ValueError("override manifest is not complete")
    if manifest.get("overrides") != overrides:
        raise ValueError("override map does not match the map embedded in the manifest")

    core = {key: value for key, value in manifest.items() if key not in _NON_CORE_KEYS}
    computed_fingerprint = _stable_sha(core)
    if manifest.get("artifact_fingerprint") != computed_fingerprint:
        raise ValueError("override manifest artifact fingerprint is invalid")
    if computed_fingerprint != artifact_fingerprint:
        raise ValueError("override artifact fingerprint does not match the requested training input")
    for field, expected in (
        ("dataset_fingerprint", dataset_fingerprint),
        ("model_fingerprint", model_fingerprint),
        ("processor_fingerprint", processor_fingerprint),
    ):
        if manifest.get(field) != expected:
            raise ValueError(f"override manifest {field} does not match the current plan")
    if manifest.get("integration_commit") != integration_commit:
        raise ValueError("override manifest integration_commit does not match the current image")

    annotation_records = manifest.get("annotations")
    records = manifest.get("records")
    if not isinstance(annotation_records, list) or not isinstance(records, list):
        raise ValueError("override manifest annotations and records must be arrays")
    covered_source_identities: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("override manifest record must be an object")
        source = Path(str(record.get("source", "")))
        output = Path(str(record.get("output", "")))
        if not source.is_file() or not output.is_file():
            raise ValueError(f"override record source/output is inaccessible: {source} -> {output}")
        covered_source_identities.update(_identity_values(source))
        source_realpath = record.get("source_realpath")
        if isinstance(source_realpath, str):
            covered_source_identities.add(source_realpath)
        if overrides.get(str(source)) != str(output):
            raise ValueError(f"override map has no matching logical source entry: {source}")
        if verify_file_hashes:
            if _sha256(source) != record.get("source_sha256"):
                raise ValueError(f"override source checksum mismatch: {source}")
            if _sha256(output) != record.get("output_sha256"):
                raise ValueError(f"override output checksum mismatch: {output}")

    for requested in required_covered_annotations:
        requested_identities = _identity_values(requested)
        matches = [
            record
            for record in annotation_records
            if requested_identities & _identity_values(Path(str(record.get("path", ""))))
        ]
        if len(matches) != 1:
            raise ValueError(f"required covered annotation has {len(matches)} manifest entries: {requested}")
        annotation_record = matches[0]
        if annotation_record.get("force_all") is not True:
            raise ValueError(f"required annotation was not prepared with force_all: {requested}")
        if annotation_record.get("sha256") != _sha256(requested):
            raise ValueError(f"required annotation checksum mismatch: {requested}")
        media_root_value = annotation_record.get("media_root")
        media_root = Path(media_root_value) if isinstance(media_root_value, str) else None
        annotation_media = _media_paths(requested.resolve(strict=True), media_root)
        missing = [
            str(path)
            for path in sorted(annotation_media, key=str)
            if not (_identity_values(path) & covered_source_identities)
        ]
        if missing:
            raise ValueError(f"required annotation has {len(missing)} media files without overrides: {missing[:3]}")

    return {
        "ok": True,
        "artifact_fingerprint": computed_fingerprint,
        "override_count": len(records),
        "required_covered_annotations": len(required_covered_annotations),
        "file_hashes_verified": verify_file_hashes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--override-map", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifact-fingerprint", required=True)
    parser.add_argument("--dataset-fingerprint", required=True)
    parser.add_argument("--model-fingerprint", required=True)
    parser.add_argument("--processor-fingerprint", required=True)
    parser.add_argument("--integration-commit", required=True)
    parser.add_argument("--require-covered-annotation", action="append", type=Path, default=[])
    parser.add_argument("--skip-file-hashes", action="store_true")
    args = parser.parse_args()
    result = validate_artifact(
        override_map=args.override_map.expanduser().resolve(strict=True),
        manifest_path=args.manifest.expanduser().resolve(strict=True),
        artifact_fingerprint=args.artifact_fingerprint,
        dataset_fingerprint=args.dataset_fingerprint,
        model_fingerprint=args.model_fingerprint,
        processor_fingerprint=args.processor_fingerprint,
        integration_commit=args.integration_commit,
        required_covered_annotations=args.require_covered_annotation,
        verify_file_hashes=not args.skip_file_hashes,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
