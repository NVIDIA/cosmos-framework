# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validator for the ``metropolis-v3.0`` annotation format.

Owns the CLI surface for the ``metropolis-v3.0`` subcommand: subparser
arguments, scene discovery, raw-type auto-detection, the per-scene
validation loop, and the schema/cross-reference checks themselves.
"""

import argparse
from pathlib import Path
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple

from cosmos_framework.data.reasoner.formats.metropolis import (
    ContextualRequirements,
    RawType,
    detect_raw_type,
    find_scenes,
)
from cosmos_framework.data.reasoner.formats.utils import FormatError, get_metadata_type, read_json_object
from cosmos_framework.data.reasoner.formats.validation import BaseValidator
from cosmos_framework.data.reasoner.formats.validation_result import ValidationResult


class MetropolisV3_0Validator(BaseValidator):
    """Validator for the ``metropolis-v3.0`` format.

    - Flat contextual directory (no camera subdirectories)
    - Schema matching by metadata.type, not filename
    - Task type lives in metadata.type, not top-level task_type
    - Multiple annotation files per type (e.g. objects_annot1, objects_human)
    """

    format: ClassVar[str] = "metropolis-v3.0"

    # Timecodes that exceed the reported video duration by at most this many
    # seconds are reported as warnings rather than errors. The threshold
    # absorbs frame-rounding and encoder duration drift (typical 30 fps
    # frame ≈ 33 ms) while still catching genuinely out-of-range timestamps.
    _DURATION_TOLERANCE_SEC: ClassVar[float] = 1.0

    _ALLOWED_TASKS: ClassVar[Dict[RawType, List[str]]] = {
        RawType.VIDEO: [
            "video_summarization",
            "scene_description",
            "bcq",
            "bcq_openended",
            "mcq",
            "mcq_openended",
            "open_qa",
            "temporal_localization",
            "causal_linkage",
            "temporal_description",
        ],
        RawType.IMAGE: [
            "scene_description",
            "bcq",
            "bcq_openended",
            "mcq",
            "mcq_openended",
            "open_qa",
        ],
    }

    _CONTEXTUAL_SCHEMAS: ClassVar[Dict[str, str]] = {
        "calibration": "contextual/calibration.schema.json",
        "chunks": "contextual/chunks.schema.json",
        "events": "contextual/events.schema.json",
        "msted": "contextual/msted.schema.json",
        "image": "contextual/image.schema.json",
        "instances": "contextual/instances.schema.json",
        "objects": "contextual/objects.schema.json",
        "tracking": "contextual/tracking.schema.json",
        "video": "contextual/video.schema.json",
    }

    _TASK_SCHEMAS: ClassVar[Dict[str, str]] = {
        "video_summarization": "tasks/video_summarization.schema.json",
        "scene_description": "tasks/scene_description.schema.json",
        "bcq": "tasks/bcq.schema.json",
        "bcq_openended": "tasks/bcq_openended.schema.json",
        "mcq": "tasks/mcq.schema.json",
        "mcq_openended": "tasks/mcq_openended.schema.json",
        "open_qa": "tasks/open_qa.schema.json",
        "temporal_localization": "tasks/temporal_localization.schema.json",
        "causal_linkage": "tasks/causal_linkage.schema.json",
        "temporal_description": "tasks/temporal_description.schema.json",
    }

    @classmethod
    def get_allowed_tasks(cls, raw_type: RawType) -> List[str]:
        return cls._ALLOWED_TASKS.get(raw_type, [])

    @classmethod
    def is_valid_task_type(cls, raw_type: RawType, task_type: str) -> bool:
        return task_type in cls.get_allowed_tasks(raw_type)

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
            help="Path to scene directory or dataset root",
        )
        parser.add_argument(
            "--no-structure",
            action="store_true",
            help="Skip directory structure validation",
        )
        parser.add_argument(
            "--no-references",
            action="store_true",
            help="Skip cross-reference validation",
        )
        parser.add_argument(
            "--raw",
            type=str,
            choices=["image", "video", "auto"],
            default="auto",
            help="Raw data type. Default: auto (detect from dataset).",
        )
        parser.add_argument(
            "--contextual",
            type=str,
            nargs="*",
            help="Contextual annotation types to validate (objects, events, tracking, ...). "
            "Use 'all' or 'complete' for the default set per raw type.",
        )
        parser.add_argument(
            "--task",
            type=str,
            nargs="*",
            help="Task types to validate. If omitted, all present tasks are validated.",
        )
        parser.add_argument(
            "--strict",
            action="store_true",
            help="Strict mode: treat warnings as errors",
        )

    # ------------------------------------------------------------------
    # CLI execution loop
    # ------------------------------------------------------------------
    def run(self) -> int:  # noqa: C901 — single linear pipeline
        from cosmos_framework.data.reasoner.formats import __version__

        assert self.args is not None, "run() requires args; pass them at construction"
        args = self.args

        print(f"🔍 Cosmos dataset Validator v{__version__}")
        print(f"Format: {self.format}\n")
        print(f"Target: {args.path}\n")

        if args.raw == "image":
            raw_type = RawType.IMAGE
        elif args.raw == "video":
            raw_type = RawType.VIDEO
        else:
            raw_type = RawType.AUTO

        contextual_arg = args.contextual
        expand_contextual = contextual_arg in (["all"], ["complete"])
        contextual_types = None if expand_contextual else contextual_arg

        task_kwargs: dict = {"require_tasks": False}
        if args.task:
            task_kwargs["task_types"] = args.task

        scenes = find_scenes(args.path)
        if not scenes:
            print(f"❌ No scene directories (containing contextual/) found under: {args.path}")
            return 1
        if len(scenes) > 1:
            print(f"Found {len(scenes)} scene(s) to validate\n")

        is_lenient = not args.strict
        all_passed = True
        total_files_checked = 0
        total_files_passed = 0

        for scene_path in scenes:
            scene_raw_type = raw_type
            if scene_raw_type == RawType.AUTO:
                try:
                    scene_raw_type = detect_raw_type(scene_path)
                    print(f"Auto-detected raw type for {scene_path.name}: {scene_raw_type.value}")
                except Exception as e:
                    print(f"❌ Failed to auto-detect raw type for {scene_path.name}: {e}")
                    all_passed = False
                    continue

            scene_contextual = (
                ContextualRequirements.COMPLETE[scene_raw_type] if expand_contextual else contextual_types
            )

            try:
                result = self.validate_scene(
                    scene_path,
                    check_structure=not args.no_structure,
                    check_references=not args.no_references,
                    raw_type=scene_raw_type,
                    contextual_types=scene_contextual,
                    permissive=is_lenient,
                    **task_kwargs,
                )
            except Exception as e:
                print(f"\n❌ Validation of {scene_path.name} failed with error: {e}")
                import traceback

                traceback.print_exc()
                all_passed = False
                continue

            total_files_checked += result.files_checked
            total_files_passed += result.files_passed

            if len(scenes) > 1:
                status = "✅" if result.is_valid() else "❌"
                print(
                    f"  {status} {scene_path.name}: {result.files_checked} files checked, {len(result.errors)} error(s)"
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

            scene_passed = result.is_valid() and not (result.warnings and args.strict)
            if not scene_passed:
                all_passed = False

        if len(scenes) > 1:
            print("\n" + "=" * 60)
            print("VALIDATION RESULTS")
            print("=" * 60)
            print(f"Scenes  checked: {len(scenes)}")
            print(f"Files   checked: {total_files_checked}")
            print(f"Files   passed : {total_files_passed}")
            print("=" * 60)

        print("✅ VALIDATION PASSED" if all_passed else "❌ VALIDATION FAILED")
        return 0 if all_passed else 1

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Per-scene validation
    # ------------------------------------------------------------------
    def validate_scene(
        self,
        scene_path: Path,
        *,
        raw_type: Optional[RawType] = None,
        contextual_types: Optional[List[str]] = None,
        check_structure: bool = True,
        check_references: bool = True,
        check_tasks: bool = True,
        task_types: Optional[List[str]] = None,
        require_tasks: bool = False,
        permissive: bool = False,
        **kwargs: Any,
    ) -> ValidationResult:
        result = ValidationResult()
        scene_id = scene_path.name

        assert raw_type is not None, "raw_type must be specified"

        if contextual_types is None:
            contextual_types = ContextualRequirements.COMPLETE[raw_type]

        for ctx_type in contextual_types:
            if not ContextualRequirements.is_valid_combination(raw_type, ctx_type):
                result.add_error(f"Contextual type '{ctx_type}' not valid for raw type '{raw_type.value}'")
                return result

        mode_str = "strict" if not permissive else "lenient (default)"
        task_mode = f", tasks: {task_types if task_types else 'all'}" if check_tasks else ""
        print(
            f"Validating scene: {scene_id} (raw_type: {raw_type.value}, "
            f"contextual: {contextual_types}{task_mode}, mode: {mode_str})"
        )

        if check_structure:
            print("  Checking directory structure...")
            self._validate_structure(scene_path, result, permissive)

        print("  Validating contextual schemas...")
        valid_contextual = self._validate_contextual_schemas(scene_path, result, raw_type, permissive)

        # Cross-reference phases only consume schema-valid contextual files —
        # see ``_validate_contextual_schemas``. Files that failed schema
        # have already produced errors in ``result``; chasing references
        # through their malformed contents would just produce noise on top.
        if check_references:
            print("  Checking contextual cross-references...")
            self._validate_cross_references(scene_path, result, valid_contextual, permissive)

        if check_tasks:
            print("  Validating task annotations...")
            self._validate_task_annotations(
                scene_path,
                result,
                raw_type,
                valid_contextual,
                task_types,
                require_tasks,
                permissive,
            )

        return result

    # ------------------------------------------------------------------
    # Structure validation
    # ------------------------------------------------------------------
    def _validate_structure(self, scene_path: Path, result: ValidationResult, permissive: bool) -> None:
        contextual_path = scene_path / "contextual"
        if not contextual_path.exists():
            result.add_error("Missing required directory: contextual/")

        raw_path = scene_path / "raw"
        if not raw_path.exists():
            if permissive:
                result.add_warning("raw/ directory not found")
            else:
                result.add_error("Missing required directory: raw/")

    # ------------------------------------------------------------------
    # Contextual schema validation
    # ------------------------------------------------------------------
    def _validate_contextual_schemas(
        self,
        scene_path: Path,
        result: ValidationResult,
        raw_type: RawType,
        permissive: bool,
    ) -> Dict[Path, dict]:
        """Validate each contextual file and return ``{path: data}`` for the
        schema-valid ones.

        Files that failed JSON parsing, lack a recognized ``metadata.type``,
        or failed schema validation are omitted from the return value (their
        errors are appended to ``result``). Downstream cross-reference phases
        consume the returned dict and therefore see only schema-validated
        data — so they can trust that ``instances`` is a dict, ``video_id`` /
        ``image_id`` is a string, etc., without defensive ``isinstance``
        guards.
        """
        valid: Dict[Path, dict] = {}
        contextual_path = scene_path / "contextual"
        if not contextual_path.exists():
            return valid

        for json_file in sorted(contextual_path.glob("*.json")):
            result.files_checked += 1
            try:
                data = read_json_object(json_file)
            except FormatError as e:
                result.add_error(f"{json_file.name}: {e}")
                continue

            meta_type = get_metadata_type(data)
            if meta_type is None:
                result.add_error(f"{json_file.name}: Missing or invalid metadata.type field")
                continue

            schema_name = self._CONTEXTUAL_SCHEMAS.get(meta_type)
            if schema_name is None:
                result.add_error(
                    f"{json_file.name}: Unknown metadata.type '{meta_type}'. "
                    f"Valid types: {list(self._CONTEXTUAL_SCHEMAS.keys())}"
                )
                continue

            errors = self._validate_data(data, schema_name)
            if errors:
                for error in errors:
                    result.add_error(f"{json_file.name}: {error}")
                continue

            result.files_passed += 1
            valid[json_file] = data

        return valid

    # ------------------------------------------------------------------
    # Cross-reference validation
    # ------------------------------------------------------------------
    def _validate_cross_references(
        self,
        scene_path: Path,
        result: ValidationResult,
        valid_files: Dict[Path, dict],
        permissive: bool,
    ) -> None:
        """Cross-reference contextual files. ``valid_files`` only contains
        schema-validated entries (see ``_validate_contextual_schemas``), so
        every ``data`` below is guaranteed to satisfy its schema: ``instances``
        is a dict, ``frames`` is a dict, ``events`` is a list, ``video_id`` /
        ``image_id`` is a string, etc. No ``isinstance`` guards needed."""
        contextual_path = scene_path / "contextual"

        # Group schema-valid files by metadata.type for the per-type loops.
        files_by_type: Dict[str, List[Tuple[Path, dict]]] = {}
        for path, data in valid_files.items():
            meta_type = get_metadata_type(data)
            if meta_type:
                files_by_type.setdefault(meta_type, []).append((path, data))

        all_instance_ids: set = set()
        for inst_path, data in files_by_type.get("instances", []):
            instances = data["instances"]
            if not instances:
                result.add_warning(f"{inst_path.name}: 'instances' is empty")
            all_instance_ids.update(instances.keys())

        for obj_path, data in files_by_type.get("objects", []):
            if not data.get("frames"):
                result.add_warning(f"{obj_path.name}: 'frames' is empty")

            inst_source = data.get("instances_source")
            if inst_source:
                inst_path = contextual_path / inst_source
                if not inst_path.exists():
                    result.add_error(f"{obj_path.name}: instances_source '{inst_source}' not found")
                elif inst_path in valid_files:
                    source_ids = set(valid_files[inst_path]["instances"].keys())
                    for frame_id, frame in data.get("frames", {}).items():
                        for inst in frame.get("instances", []):
                            oid = inst.get("object_id")
                            if oid and oid not in source_ids:
                                result.add_error(
                                    f"{obj_path.name}: object_id '{oid}' in {frame_id} not found in {inst_source}"
                                )
                # else: referenced file exists but failed its own schema;
                # the user's already seeing those errors — no need to pile on.
            elif all_instance_ids:
                for frame_id, frame in data.get("frames", {}).items():
                    for inst in frame.get("instances", []):
                        oid = inst.get("object_id")
                        if oid and oid not in all_instance_ids:
                            result.add_error(
                                f"{obj_path.name}: object_id '{oid}' in {frame_id} not found in any instances file"
                            )

        for evt_path, data in files_by_type.get("events", []):
            if not data.get("events"):
                result.add_warning(f"{evt_path.name}: 'events' is empty")

            inst_source = data.get("instances_source")
            event_source_ids: Optional[set] = None

            if inst_source:
                inst_path = contextual_path / inst_source
                if not inst_path.exists():
                    result.add_error(f"{evt_path.name}: instances_source '{inst_source}' not found")
                elif inst_path in valid_files:
                    event_source_ids = set(valid_files[inst_path]["instances"].keys())
                # else: referenced file failed its own schema; skip silently.

            for event in data.get("events", []):
                event_instances = event.get("instances", [])
                if not event_instances:
                    continue

                if event_source_ids is not None:
                    for oid in event_instances:
                        if oid not in event_source_ids:
                            result.add_error(
                                f"{evt_path.name}: event '{event.get('event_id', '?')}' "
                                f"references object_id '{oid}' not found in {inst_source}"
                            )
                elif inst_source is None and all_instance_ids:
                    for oid in event_instances:
                        if oid not in all_instance_ids:
                            result.add_error(
                                f"{evt_path.name}: event '{event.get('event_id', '?')}' "
                                f"references object_id '{oid}' not found in any instances file"
                            )
                elif not all_instance_ids and event_instances:
                    msg = f"{evt_path.name}: events reference instances but no instances file found"
                    if permissive:
                        result.add_warning(msg)
                    else:
                        result.add_error(msg)

        for vid_path, data in files_by_type.get("video", []):
            if "camera_id" in data and not files_by_type.get("calibration"):
                msg = f"{vid_path.name}: references camera_id but no calibration file found"
                if permissive:
                    result.add_warning(msg)
                else:
                    result.add_error(msg)

        self._validate_timestamps_vs_duration(files_by_type, result)

    # ------------------------------------------------------------------
    # Timestamp ⇄ video duration cross-check
    # ------------------------------------------------------------------
    def _tc_to_sec(self, tc: str) -> Optional[float]:
        try:
            parts = tc.strip().split(":")
            if len(parts) == 3:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
            if len(parts) == 2:
                return int(parts[0]) * 60 + float(parts[1])
            return float(parts[0])
        except (ValueError, IndexError):
            return None

    def _validate_timestamps_vs_duration(
        self,
        files_by_type: Dict[str, List[Tuple[Path, dict]]],
        result: ValidationResult,
    ) -> None:
        """Flag timecodes in events/chunks/msted that exceed their video's
        ``duration``. All inputs are schema-valid, so ``video_id`` is a
        string and timecode fields hold the documented shape."""
        duration_by_id: Dict[str, float] = {}
        for _, data in files_by_type.get("video", []):
            duration_raw = data.get("duration")
            if duration_raw is None:
                continue
            try:
                dur = float(duration_raw)
            except (ValueError, TypeError):
                continue
            if dur > 0:
                duration_by_id[data["video_id"]] = dur

        if not duration_by_id:
            return

        # Per metadata.type: (list-key inside the JSON, the two timecode field
        # names on each record, function that builds a per-record label).
        sources: List[Tuple[str, str, Tuple[str, str], Callable[[int, dict], str]]] = [
            (
                "events",
                "events",
                ("start_time", "end_time"),
                lambda _i, e: f"event '{e.get('event_id', '?')}'",
            ),
            (
                "chunks",
                "chunks",
                ("start", "end"),
                lambda _i, c: f"chunk '{c.get('chunk_id', '?')}'",
            ),
            (
                "msted",
                "temporal_spatial_localization",
                ("start", "end"),
                lambda i, _r: f"tsl[{i}]",
            ),
        ]

        for meta_type, list_key, fields, label_fn in sources:
            for json_file, data in files_by_type.get(meta_type, []):
                duration = duration_by_id.get(data.get("video_id", ""))
                if duration is None:
                    continue
                for idx, record in enumerate(data.get(list_key, [])):
                    label = label_fn(idx, record)
                    for field in fields:
                        tc = record.get(field)
                        if not isinstance(tc, str):
                            continue
                        sec = self._tc_to_sec(tc)
                        if sec is None:
                            continue
                        overrun = sec - duration
                        if overrun <= 0:
                            continue
                        msg = (
                            f"{json_file.name}: {label} {field} timecode '{tc}' "
                            f"({sec:.3f}s) exceeds video duration ({duration:.3f}s)"
                        )
                        if overrun <= self._DURATION_TOLERANCE_SEC:
                            result.add_warning(
                                f"{msg} by {overrun:.3f}s (within {self._DURATION_TOLERANCE_SEC:.1f}s tolerance)"
                            )
                        else:
                            result.add_error(msg)

    # ------------------------------------------------------------------
    # Task annotation validation
    # ------------------------------------------------------------------
    def _validate_task_annotations(
        self,
        scene_path: Path,
        result: ValidationResult,
        raw_type: RawType,
        valid_contextual: Dict[Path, dict],
        task_types: Optional[List[str]],
        require_tasks: bool,
        permissive: bool,
    ) -> None:
        task_path = scene_path / "task"

        if not task_path.exists():
            if require_tasks:
                result.add_error("task/ directory required but not found")
            elif not permissive:
                result.add_error("task/ directory not found")
            else:
                result.add_warning("task/ directory not found (optional in metropolis-v3.0)")
            return

        if not task_path.is_dir():
            result.add_error(f"task/ exists but is not a directory: {task_path}")
            return

        task_files = list(task_path.glob("*.json"))
        if not task_files:
            if require_tasks:
                result.add_error("At least one task file required but task/ directory is empty")
            else:
                result.add_warning("task/ directory is empty")
            return

        valid_media_ids = self._collect_media_ids(valid_contextual)

        for task_file in sorted(task_files):
            result.files_checked += 1

            try:
                data = read_json_object(task_file)
            except FormatError as e:
                result.add_error(f"task/{task_file.name}: {e}")
                continue

            file_task_type = get_metadata_type(data)
            if file_task_type is None:
                file_task_type = data.get("task_type")

            if file_task_type is None:
                result.add_error(
                    f"task/{task_file.name}: Missing task type (expected metadata.type or task_type field)"
                )
                continue

            schema_name = self._TASK_SCHEMAS.get(file_task_type)
            if schema_name is None:
                result.add_error(
                    f"task/{task_file.name}: Unknown task type: '{file_task_type}'. "
                    f"Valid types: {list(self._TASK_SCHEMAS.keys())}"
                )
                continue

            errors = self._validate_data(data, schema_name)
            if errors:
                for error in errors:
                    result.add_error(f"task/{task_file.name}: {error}")
                continue

            if task_types is not None and "all" not in task_types:
                if file_task_type not in task_types:
                    result.add_error(
                        f"task/{task_file.name}: Task type '{file_task_type}' not requested. "
                        f"Requested types: {task_types}"
                    )
                    continue

            if not self.is_valid_task_type(raw_type, file_task_type):
                allowed = self.get_allowed_tasks(raw_type)
                result.add_error(
                    f"task/{task_file.name}: Task type '{file_task_type}' not valid for "
                    f"raw type '{raw_type.value}'. Allowed tasks: {allowed}"
                )
            else:
                result.files_passed += 1
                ref_errors = self._validate_task_references(data, task_file.name, file_task_type, valid_media_ids)
                for error in ref_errors:
                    result.add_error(f"task/{task_file.name}: {error}")

    def _collect_media_ids(self, valid_contextual: Dict[Path, dict]) -> set:
        """Return the set of ``video_id`` / ``image_id`` values declared in
        schema-valid contextual files. ``valid_contextual`` is the return
        value of ``_validate_contextual_schemas``, so every ``data`` here is
        schema-validated — ``video_id`` and ``image_id`` are guaranteed
        strings when present."""
        ids: set[str] = set()
        for data in valid_contextual.values():
            meta_type = get_metadata_type(data)
            if meta_type == "video":
                ids.add(data["video_id"])
            elif meta_type == "image":
                ids.add(data["image_id"])
        return ids

    def _validate_task_references(
        self,
        task_data: dict,
        task_filename: str,
        task_type: str,
        valid_media_ids: set,
    ) -> List[str]:
        errors = []
        items = task_data.get("items", [])

        for idx, item in enumerate(items):
            video_id = item.get("video_id")
            image_id = item.get("image_id")
            ref_id = video_id or image_id

            if ref_id and valid_media_ids and ref_id not in valid_media_ids:
                id_field = "video_id" if video_id else "image_id"
                errors.append(f"[items][{idx}] Invalid {id_field}: '{ref_id}' not found in contextual files")

        return errors
