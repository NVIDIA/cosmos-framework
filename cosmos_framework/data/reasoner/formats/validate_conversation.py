# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validator for the ``cosmos-reason-v1.0`` annotation format.

Owns the CLI surface for the ``cosmos-reason-v1.0`` subcommand. Cosmos-Reason
has no notion of raw type or contextual annotations, so the surface is much
slimmer than metropolis: just structure / schema / cross-reference toggles.
"""

import argparse
from pathlib import Path
from typing import Any, ClassVar, Optional

from cosmos_framework.data.reasoner.formats.conversation import find_datasets
from cosmos_framework.data.reasoner.formats.utils import FormatError, read_json_object
from cosmos_framework.data.reasoner.formats.validation import BaseValidator
from cosmos_framework.data.reasoner.formats.validation_result import ValidationResult


class CosmosReasonV1_0Validator(BaseValidator):
    """Validator for the ``cosmos-reason-v1.0`` format.

    Validates the three-component dataset structure:
    - meta.json  — dataset index (version, metadata, samples[])
    - media/     — video and image files
    - text/      — one conversation JSON file per training sample
    """

    format: ClassVar[str] = "cosmos-reason-v1.0"

    # ------------------------------------------------------------------
    # CLI plumbing
    # ------------------------------------------------------------------
    @classmethod
    def register_subparser(cls, subparsers: "argparse._SubParsersAction") -> None:
        parser = subparsers.add_parser(
            cls.format,
            help=f"Validate a {cls.format} dataset",
        )
        parser.add_argument(
            "--path",
            type=Path,
            required=True,
            help="Path to dataset root (containing meta.json, media/, text/)",
        )
        parser.add_argument(
            "--strict",
            action="store_true",
            help="Strict mode: treat warnings as errors",
        )

    # ------------------------------------------------------------------
    # CLI execution loop
    # ------------------------------------------------------------------
    def run(self) -> int:
        from cosmos_framework.data.reasoner.formats import __version__

        assert self.args is not None, "run() requires args; pass them at construction"
        args = self.args

        print(f"🔍 Cosmos dataset Validator v{__version__}")
        print(f"Format: {self.format}\n")
        print(f"Target: {args.path}\n")

        datasets = find_datasets(args.path)
        if not datasets:
            print(f"❌ No cosmos-reason datasets (containing meta.json) found under: {args.path}")
            return 1

        is_lenient = not args.strict
        all_passed = True
        total_files_checked = 0
        total_files_passed = 0

        for dataset_path in datasets:
            try:
                result = self.validate_dataset(dataset_path, permissive=is_lenient)
            except Exception as e:
                print(f"\n❌ Validation of {dataset_path.name} failed with error: {e}")
                import traceback

                traceback.print_exc()
                all_passed = False
                continue

            total_files_checked += result.files_checked
            total_files_passed += result.files_passed

            if len(datasets) > 1:
                status = "✅" if result.is_valid() else "❌"
                print(
                    f"  {status} {dataset_path.name}: {result.files_checked} files checked, "
                    f"{len(result.errors)} error(s)"
                )
                if result.errors:
                    for error in result.errors:
                        print(f"      - {error}")
            else:
                print("\n" + "=" * 60)
                print("VALIDATION RESULTS")
                print("=" * 60)
                print(result.summary())
                if result.warnings:
                    print(f"\n⚠️  Warnings ({len(result.warnings)}):")
                    for warning in result.warnings:
                        print(f"  - {warning}")
                if result.errors:
                    print(f"\n❌ Errors ({len(result.errors)}):")
                    for error in result.errors:
                        print(f"  - {error}")
                print("\n" + "=" * 60)

            if not result.is_valid() or (result.warnings and args.strict):
                all_passed = False

        if len(datasets) > 1:
            print("\n" + "=" * 60)
            print("VALIDATION RESULTS")
            print("=" * 60)
            print(f"Datasets checked: {len(datasets)}")
            print(f"Files   checked : {total_files_checked}")
            print(f"Files   passed  : {total_files_passed}")
            print("=" * 60)

        print("✅ VALIDATION PASSED" if all_passed else "❌ VALIDATION FAILED")
        return 0 if all_passed else 1

    # ------------------------------------------------------------------
    # Per-dataset validation
    # ------------------------------------------------------------------
    def validate_dataset(
        self,
        dataset_path: Path,
        *,
        permissive: bool = False,
        **kwargs: Any,
    ) -> ValidationResult:
        result = ValidationResult()
        dataset_id = dataset_path.name

        print(f"Validating dataset: {dataset_id} (format: {self.format})")

        print("  Checking directory structure...")
        self._validate_structure(dataset_path, result)

        print("  Validating schemas...")
        meta_data = self._validate_schemas(dataset_path, result)

        # Cross-references are only meaningful on schema-valid data: every
        # downstream check (set membership, path joining, orphan detection)
        # assumes the contract the schema enforces. If schema validation
        # failed, fix that first and re-run — chasing reference errors on
        # malformed data produces noise, not signal.
        if meta_data is not None:
            print("  Checking cross-references...")
            self._validate_cross_references(dataset_path, meta_data, result, permissive)
        else:
            print("  Skipping cross-references (schema validation failed).")

        return result

    # ------------------------------------------------------------------
    # Structure validation
    # ------------------------------------------------------------------
    def _validate_structure(self, dataset_path: Path, result: ValidationResult) -> None:
        """Check that the three required components of a cosmos-reason dataset exist.

        meta.json, media/, and text/ are all required by this format; missing
        any of them is an error regardless of ``--strict``. (``--strict`` only
        promotes *warnings* to errors; it does not change which items are
        required.)
        """
        if not (dataset_path / "meta.json").exists():
            result.add_error("Missing required file: meta.json")

        media_path = dataset_path / "media"
        if not media_path.exists():
            result.add_error("Missing required directory: media/")
        elif not media_path.is_dir():
            result.add_error("media/ exists but is not a directory")

        text_path = dataset_path / "text"
        if not text_path.exists():
            result.add_error("Missing required directory: text/")
        elif not text_path.is_dir():
            result.add_error("text/ exists but is not a directory")

    # ------------------------------------------------------------------
    # Schema validation
    # ------------------------------------------------------------------
    def _validate_schemas(self, dataset_path: Path, result: ValidationResult) -> Optional[dict]:
        meta_data: Optional[dict] = None

        meta_path = dataset_path / "meta.json"
        if meta_path.exists():
            result.files_checked += 1
            try:
                data = read_json_object(meta_path)
            except FormatError as e:
                result.add_error(f"meta.json: {e}")
            else:
                errors = self._validate_data(data, "meta.schema.json")
                if errors:
                    for err in errors:
                        result.add_error(f"meta.json: {err}")
                else:
                    result.files_passed += 1
                    meta_data = data

        text_path = dataset_path / "text"
        if text_path.exists() and text_path.is_dir():
            for conv_file in sorted(text_path.glob("*.json")):
                result.files_checked += 1
                try:
                    data = read_json_object(conv_file)
                except FormatError as e:
                    result.add_error(f"text/{conv_file.name}: {e}")
                    continue

                errors = self._validate_data(data, "conversation.schema.json")
                if errors:
                    for err in errors:
                        result.add_error(f"text/{conv_file.name}: {err}")
                else:
                    result.files_passed += 1

        return meta_data

    # ------------------------------------------------------------------
    # Cross-reference validation
    # ------------------------------------------------------------------
    def _validate_cross_references(
        self,
        dataset_path: Path,
        meta_data: dict,
        result: ValidationResult,
        permissive: bool,
    ) -> None:
        samples = meta_data.get("samples", [])

        seen_ids: set = set()
        for sample in samples:
            sid = sample.get("id")
            if sid is None:
                continue
            if sid in seen_ids:
                result.add_error(f"meta.json: Duplicate sample id '{sid}'")
            else:
                seen_ids.add(sid)

        referenced_conv_paths: set = set()
        for sample in samples:
            sid = sample.get("id", "?")

            conv_rel = sample.get("conversation")
            if conv_rel:
                referenced_conv_paths.add(conv_rel)
                if not (dataset_path / conv_rel).exists():
                    result.add_error(f"meta.json: sample '{sid}': conversation file not found: {conv_rel}")

            media_rel = sample.get("media")
            if media_rel:
                if not (dataset_path / media_rel).exists():
                    result.add_error(f"meta.json: sample '{sid}': media file not found: {media_rel}")

        text_path = dataset_path / "text"
        if text_path.exists() and text_path.is_dir():
            for conv_file in sorted(text_path.glob("*.json")):
                rel_path = f"text/{conv_file.name}"
                if rel_path not in referenced_conv_paths:
                    msg = f"text/{conv_file.name}: not referenced by any sample in meta.json"
                    result.add_warning(msg) if permissive else result.add_error(msg)
