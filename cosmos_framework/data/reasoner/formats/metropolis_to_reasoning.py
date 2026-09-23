# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Converter: ``metropolis-v3.0`` scene annotations → ``cosmos-video-reasoning-v1.0`` training format.

Items are aggregated by task type: each metropolis task type produces one
``<task_type>.json`` annotation file in the output. The task-specific framing
(MCQ option blocks, BCQ "Yes/No" prefix, temporal-localization JSON wrapping,
answer-format instructions) is composed directly into the free-form ``question``
and ``answer`` strings, since cosmos-video-reasoning-v1.0 has no per-task schema.
"""

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional

from cosmos_framework.data.reasoner.formats import metropolis as fmt
from cosmos_framework.data.reasoner.formats.conversion import BaseConverter, ConversionResult
from cosmos_framework.data.reasoner.formats.utils import media_dest_basename

DEFAULT_LICENSE = "CC BY-NC-ND 4.0"


class MetropolisV3_0ToVideoReasoningV1_0Converter(BaseConverter):
    """Converts metropolis-v3.0 task annotations to cosmos-video-reasoning-v1.0 training format.

    Output layout::

        {output}/
        ├── bcq.json          (one annotation file per source task type)
        ├── mcq.json
        ├── ...
        ├── videos/           (copied media; created on demand)
        └── images/           (copied media; created on demand)

    Each annotation file uses ``media_root: null`` and items reference media via
    relative paths (e.g. ``"videos/clip.mp4"``).
    """

    source_format: ClassVar[str] = "metropolis-v3.0"
    target_format: ClassVar[str] = "cosmos-video-reasoning-v1.0"

    SUPPORTED_TASKS = frozenset(
        [
            "open_qa",
            "bcq",
            "bcq_openended",
            "mcq",
            "mcq_openended",
            "video_summarization",
            "scene_description",
            "temporal_localization",
            "temporal_description",
            "causal_linkage",
        ]
    )

    _ANSWER_INSTRUCTIONS = {
        "bcq": "Answer with only Yes or No.",
        "bcq_openended": "Answer with Yes or No, followed by a brief explanation.",
        "mcq": "Choose the correct option by letter only.",
        "mcq_openended": "Choose the correct option and provide a brief explanation.",
        "temporal_localization": (
            "Provide the result in json format with 'mm:ss' for time depiction. "
            "Use keywords 'start', 'end' in the json output."
        ),
    }

    # ------------------------------------------------------------------
    # CLI plumbing
    # ------------------------------------------------------------------
    @classmethod
    def register_subparser(cls, target_subparsers: "argparse._SubParsersAction") -> None:
        parser = target_subparsers.add_parser(
            cls.target_format,
            help=f"Convert {cls.source_format} → {cls.target_format}",
        )
        cls._add_shared_arguments(parser)
        # Source-flavored: filters which task types the source emits.
        parser.add_argument(
            "--task",
            type=str,
            nargs="*",
            help="Task types to include (default: all supported). Example: --task open_qa video_summarization",
        )
        # Pair-specific: depends on both source layout (raw/) and target layout (videos/, images/).
        parser.add_argument(
            "--no-copy-media",
            action="store_true",
            help="Do not copy media files. media_root is set to the absolute source "
            "path; each item's video_id/image_id is the file's path relative to "
            "that root (e.g. 'raw/clip.mp4').",
        )
        parser.add_argument(
            "--emit-media-root",
            action="store_true",
            help="Only meaningful with --no-copy-media. Write media_root as null "
            "instead of the absolute source path so the dataset is portable; "
            "the consumer sets media_root at load time.",
        )
        # Target-flavored: lands in every emitted annotation file's metadata block.
        parser.add_argument(
            "--description",
            type=str,
            help="Human-readable description added to each annotation file's metadata block",
        )
        parser.add_argument(
            "--license",
            type=str,
            default=DEFAULT_LICENSE,
            help=f"License string added to each annotation file's metadata block (default: {DEFAULT_LICENSE})",
        )

    # ------------------------------------------------------------------
    # CLI execution
    # ------------------------------------------------------------------
    def run(self) -> int:
        from cosmos_framework.data.reasoner.formats import __version__

        assert self.args is not None, "run() requires args; pass them at construction"
        args = self.args

        print(f"🔄 Cosmos dataset Converter v{__version__}")
        print(f"   {self.source_format} → {self.target_format}")
        print(f"   Source : {args.path}")
        print(f"   Output : {args.output}\n")

        metadata: Dict[str, Any] = {}
        if args.description:
            metadata["description"] = args.description
        if args.license:
            metadata["license"] = args.license

        # convert_dataset handles both a single scene and a multi-scene root —
        # find_scenes(marker_dir="task") returns [root] when root itself is a
        # scene, otherwise walks recursively.
        result = self.convert_dataset(
            dataset_path=Path(args.path),
            output_path=args.output,
            task_types=args.task or None,
            copy_media=not args.no_copy_media,
            metadata=metadata or None,
            emit_media_root_as_null=args.emit_media_root,
        )

        print("=" * 60)
        print("CONVERSION RESULTS")
        print("=" * 60)
        print(result.summary())

        if result.warnings:
            print(f"\n⚠️  Warnings ({len(result.warnings)}):")
            for w in result.warnings:
                print(f"  - {w}")

        if result.errors:
            print(f"\n❌ Errors ({len(result.errors)}):")
            for e in result.errors:
                print(f"  - {e}")

        print("\n" + "=" * 60)

        if result.is_success():
            print("✅ CONVERSION COMPLETE")
            return 0
        if result.errors:
            print("❌ CONVERSION FAILED")
        else:
            # Output is valid but smaller than asked; CI gates that consume
            # the exit code still treat this as a failure (the conversion
            # didn't deliver every requested sample).
            print("⚠️  CONVERSION INCOMPLETE (some samples skipped — see warnings)")
        return 1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def convert_scene(
        self,
        scene_path: Path,
        output_path: Path,
        task_types: Optional[List[str]] = None,
        copy_media: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
        emit_media_root_as_null: bool = False,
    ) -> ConversionResult:
        """Convert a single metropolis-v3.0 scene to a cosmos-video-reasoning-v1.0 dataset."""
        scene_path = Path(scene_path).resolve()
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        result = ConversionResult()

        if not (scene_path / "task").is_dir():
            result.errors.append(f"task/ directory not found in {scene_path}")
            return result

        items_by_type: Dict[str, List[dict]] = defaultdict(list)
        self._collect_items_from_scene(
            scene_path, output_path, task_types, copy_media, scene_path, items_by_type, result
        )

        media_root_value = self._compute_media_root(copy_media, emit_media_root_as_null, scene_path)
        for task_type, items in items_by_type.items():
            self._write_annotation_file(output_path, task_type, items, metadata, media_root_value)
            result.samples_written += len(items)

        return result

    def convert_dataset(
        self,
        dataset_path: Path,
        output_path: Path,
        task_types: Optional[List[str]] = None,
        copy_media: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
        emit_media_root_as_null: bool = False,
    ) -> ConversionResult:
        """Convert all metropolis-v3.0 scenes under *dataset_path* into one dataset.

        Items from all scenes are aggregated by task type into the same per-type
        annotation files at the output root.
        """
        dataset_path = Path(dataset_path).resolve()
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        scenes = fmt.find_scenes(dataset_path, marker_dir="task")
        result = ConversionResult()

        if not scenes:
            result.errors.append(f"No metropolis-v3.0 scenes (containing task/) found under {dataset_path}")
            return result

        items_by_type: Dict[str, List[dict]] = defaultdict(list)
        for scene_path in scenes:
            self._collect_items_from_scene(
                scene_path,
                output_path,
                task_types,
                copy_media,
                dataset_path,
                items_by_type,
                result,
            )

        media_root_value = self._compute_media_root(copy_media, emit_media_root_as_null, dataset_path)
        for task_type, items in items_by_type.items():
            self._write_annotation_file(output_path, task_type, items, metadata, media_root_value)
            result.samples_written += len(items)

        return result

    # ------------------------------------------------------------------
    # Core conversion logic
    # ------------------------------------------------------------------

    def _collect_items_from_scene(
        self,
        scene_path: Path,
        output_path: Path,
        task_types: Optional[List[str]],
        copy_media: bool,
        media_anchor: Path,
        items_by_type: Dict[str, List[dict]],
        result: ConversionResult,
    ) -> None:
        """Walk one metropolis scene and append each item to ``items_by_type``.

        Source items whose media cannot be resolved through the schema
        contract (``raw/{media_id}.{format}``) are skipped with a warning
        and counted via ``result.samples_skipped`` — never written with a
        synthesized path that points nowhere.
        """
        media_formats = fmt.build_media_format_index(scene_path, result.warnings)

        for task_file, task_type, idx, source_item in fmt.iter_task_items(
            scene_path, self.SUPPORTED_TASKS, task_types, result.warnings
        ):
            media_id = source_item.get("video_id") or source_item.get("image_id")
            is_video = "video_id" in source_item

            if not media_id:
                result.warnings.append(f"{task_file.name}[{idx}]: missing video_id/image_id, skipping")
                result.samples_skipped += 1
                continue

            fmt_ext = media_formats.get(media_id)
            if fmt_ext is None:
                result.warnings.append(
                    f"{task_file.name}[{idx}]: no contextual entry for '{media_id}' declares a format, skipping"
                )
                result.samples_skipped += 1
                continue

            media_rel = self._handle_media(
                scene_path,
                output_path,
                media_id,
                fmt_ext,
                is_video,
                copy_media,
                media_anchor,
                result,
                task_file,
                idx,
            )
            if media_rel is None:
                result.samples_skipped += 1
                continue

            question_text = self._compose_question(task_type, source_item)
            answer_text = self._format_answer(task_type, source_item)
            if answer_text is None:
                result.warnings.append(f"{task_file.name}[{idx}]: could not build answer, skipping")
                result.samples_skipped += 1
                continue

            item: Dict[str, Any] = {
                ("video_id" if is_video else "image_id"): media_rel,
                "question": question_text,
                "answer": answer_text,
            }
            reasoning = source_item.get("reasoning")
            if reasoning:
                item["reasoning"] = str(reasoning)

            items_by_type[task_type].append(item)

    def _handle_media(
        self,
        scene_path: Path,
        output_path: Path,
        media_id: str,
        fmt_ext: str,
        is_video: bool,
        copy_media: bool,
        media_anchor: Path,
        result: ConversionResult,
        task_file: Path,
        idx: int,
    ) -> Optional[str]:
        """Resolve the source media file at ``raw/{media_id}.{fmt_ext}`` and
        return the path string to embed in the output item.

        Returns ``None`` when the source file is missing — the caller skips
        the sample (and counts it as skipped) rather than emitting an item
        that references a nonexistent file.

        - ``copy_media=True``: media is copied to ``output_path/<videos|images>/``
          under a flattened source-relative basename so scenes sharing a
          basename don't collide (see :func:`media_dest_basename`).
        - ``copy_media=False``: media is referenced in place; the item path
          is relative to ``media_anchor`` (the source scene root or
          dataset root).
        """
        media_src = fmt.find_media_file(scene_path, media_id, fmt_ext)
        if media_src is None:
            result.warnings.append(
                f"{task_file.name}[{idx}]: expected media file 'raw/{media_id}.{fmt_ext}' not found, skipping"
            )
            return None

        subdir = "videos" if is_video else "images"
        if copy_media:
            basename = media_dest_basename(media_src, media_anchor)
            media_out = output_path / subdir
            media_out.mkdir(parents=True, exist_ok=True)
            dest = media_out / basename
            if not dest.exists():
                shutil.copy2(media_src, dest)
            return f"{subdir}/{basename}"

        # --no-copy-media: leave the file in place; report its path relative to
        # media_anchor. Falls back to the absolute path if the resolved file
        # somehow lives outside the anchor (defensive — shouldn't happen for
        # files discovered via find_media_file).
        media_src_abs = media_src.resolve()
        try:
            return str(media_src_abs.relative_to(media_anchor))
        except ValueError:
            return str(media_src_abs)

    # ------------------------------------------------------------------
    # Question / answer composition
    # ------------------------------------------------------------------

    def _compose_question(self, task_type: str, item: dict) -> str:
        """Build the free-form ``question`` string for a cosmos-video-reasoning item.

        Embeds task-specific framing (MCQ options, answer-format instructions)
        directly into the prompt, since cosmos-video-reasoning has no per-task structure.
        """
        question = str(item.get("question", ""))
        if task_type in ("mcq", "mcq_openended"):
            options = item.get("options")
            if options:
                options_text = "\n".join(f"{k}) {v}" for k, v in sorted(options.items()))
                question = f"{question}\n\n{options_text}"
        instruction = self._ANSWER_INSTRUCTIONS.get(task_type)
        if instruction:
            question = f"{question}\n\n{instruction}"
        return question

    def _format_answer(self, task_type: str, item: dict) -> Optional[str]:
        """Format the free-form ``answer`` string for a cosmos-video-reasoning item."""
        if task_type in (
            "open_qa",
            "bcq_openended",
            "mcq_openended",
            "video_summarization",
            "scene_description",
            "temporal_description",
            "causal_linkage",
        ):
            return item.get("answer") or None

        if task_type == "bcq":
            answer = item.get("answer")
            if not answer:
                return None
            explanation = item.get("explanation", "")
            return f"{answer}. {explanation}" if explanation else answer

        if task_type == "mcq":
            letter = item.get("answer")
            if not letter:
                return None
            options = item.get("options", {})
            label = f"{letter}) {options[letter]}" if letter in options else letter
            explanation = item.get("explanation", "")
            return f"{label}. {explanation}" if explanation else label

        if task_type == "temporal_localization":
            answer = item.get("answer")
            if not answer:
                return None
            return f"```json\n{json.dumps(answer, indent=2)}\n```"

        return None

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def _compute_media_root(
        self,
        copy_media: bool,
        emit_media_root_as_null: bool,
        media_anchor: Path,
    ) -> Optional[str]:
        """Decide what to write into each annotation file's ``media_root`` field.

        - ``copy_media=True``: items reference paths under the output dataset
          (``videos/...``, ``images/...``); media_root is null (relative-to-output).
        - ``copy_media=False``: items reference paths under ``media_anchor``.
          ``media_root`` is ``str(media_anchor)`` so the dataset is readable in
          place. With ``emit_media_root_as_null=True`` the field is nulled out
          to make the dataset portable (consumer sets media_root at load time).
        """
        if copy_media:
            return None
        if emit_media_root_as_null:
            return None
        return str(media_anchor)

    def _write_annotation_file(
        self,
        output_path: Path,
        task_type: str,
        items: List[dict],
        metadata: Optional[Dict[str, Any]],
        media_root: Optional[str],
    ) -> None:
        """Emit ``output_path/<task_type>.json`` with the aggregated items.

        ``metadata.type`` is fixed to ``"annotation"`` (the schema discriminator);
        the source task type is recorded in ``metadata.task``.
        """
        meta_block: Dict[str, Any] = {"type": "annotation", "task": task_type}
        if metadata:
            meta_block.update({k: v for k, v in metadata.items() if k not in ("type", "task")})
        meta_block.setdefault("license", DEFAULT_LICENSE)
        annotation = {
            "format": self.target_format,
            "metadata": meta_block,
            "media_root": media_root,
            "items": items,
        }
        with open(output_path / f"{task_type}.json", "w") as f:
            json.dump(annotation, f, indent=2)
