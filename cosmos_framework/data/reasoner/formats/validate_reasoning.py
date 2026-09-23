# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validator for the ``cosmos-video-reasoning-v1.0`` annotation format.

A dataset is one or more ``*.json`` annotation files (filename free) plus the
media those annotations reference. Schema validation is per file; cross-reference
validation walks every item's ``video_id`` / ``image_id`` and verifies the
referenced media file exists on disk.
"""

import argparse
from pathlib import Path
from typing import Any, ClassVar, Optional

from cosmos_framework.data.reasoner.formats.reasoning import (
    FORMAT,
    find_datasets,
    resolve_media_path,
)
from cosmos_framework.data.reasoner.formats.utils import FormatError, read_json_object
from cosmos_framework.data.reasoner.formats.validation import BaseValidator
from cosmos_framework.data.reasoner.formats.validation_result import ValidationResult


class VideoReasoningV1_0Validator(BaseValidator):
    """Validator for the ``cosmos-video-reasoning-v1.0`` format.

    Validates the dataset structure:
    - one or more annotation files (``*.json`` with ``format: "cosmos-video-reasoning-v1.0"``)
    - media files reachable via ``media_root`` + each item's ``video_id``/``image_id``
    """

    format: ClassVar[str] = "cosmos-video-reasoning-v1.0"

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
            help="Path to dataset root (containing one or more annotation .json files)",
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
            print(
                f"❌ No {self.format} datasets (containing a *.json with "
                f"format='{self.format}') found under: {args.path}"
            )
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

        print("  Validating schemas...")
        annotations = self._validate_schemas(dataset_path, result)

        print("  Checking media references...")
        for ann_path, ann_data in annotations:
            self._validate_media_references(dataset_path, ann_path, ann_data, result)

        return result

    # ------------------------------------------------------------------
    # Schema validation
    # ------------------------------------------------------------------
    def _validate_schemas(
        self,
        dataset_path: Path,
        result: ValidationResult,
    ) -> list:
        """Find annotation files and validate each against the schema.

        Walks every ``*.json`` directly under ``dataset_path``, reading each
        file exactly once. A file is treated as an annotation when its
        top-level ``format`` equals ``"cosmos-video-reasoning-v1.0"``; anything else
        is silently skipped (it might be an unrelated JSON or a sibling
        format's file).

        Returns ``[(path, data), ...]`` for files that both look like
        annotations AND pass schema validation — downstream
        ``_validate_media_references`` consumes only those.
        """
        out = []
        for ann_path in sorted(dataset_path.glob("*.json")):
            try:
                data = read_json_object(ann_path)
            except FormatError:
                continue  # unparseable or non-object JSON — definitely not ours
            if data.get("format") != FORMAT:
                continue

            result.files_checked += 1
            errors = self._validate_data(data, "reasoning.schema.json")
            if errors:
                for err in errors:
                    result.add_error(f"{ann_path.name}: {err}")
            else:
                result.files_passed += 1
                out.append((ann_path, data))
        return out

    # ------------------------------------------------------------------
    # Media-reference validation
    # ------------------------------------------------------------------
    def _validate_media_references(
        self,
        dataset_path: Path,
        ann_path: Path,
        ann_data: dict,
        result: ValidationResult,
    ) -> None:
        """Resolve each item's media path and require the file to exist.

        ``ann_data`` is schema-validated (see ``_validate_schemas``), so
        every item has exactly one of ``video_id`` / ``image_id`` and the
        value is a string. A missing media file is always an error: the
        annotation claims the file exists, and that claim is part of the
        format's contract.
        """
        media_root: Optional[str] = ann_data.get("media_root")
        for idx, item in enumerate(ann_data.get("items", [])):
            rel_path = item.get("video_id") or item.get("image_id")
            resolved = resolve_media_path(dataset_path, media_root, rel_path)
            if not resolved.exists():
                result.add_error(f"{ann_path.name}: items[{idx}] media not found: {resolved}")
