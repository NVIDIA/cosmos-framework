# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Converter: ``metropolis-v3.0`` scene annotations → ``cosmos-reason-v1.0`` training format.

Each task item in a metropolis-v3.0 scene becomes one training sample:

  task/open_qa.json  ──►  text/<id>.json  +  entry in meta.json
  task/mcq.json      ──►  text/<id>.json  +  entry in meta.json
  ...

Media files are resolved from ``raw/`` and optionally copied to ``media/`` in
the output dataset.
"""

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional

from cosmos_framework.data.reasoner.formats import metropolis as fmt
from cosmos_framework.data.reasoner.formats.conversion import BaseConverter, ConversionResult
from cosmos_framework.data.reasoner.formats.utils import media_dest_basename


class MetropolisV3_0ToCosmosReasonV1_0Converter(BaseConverter):
    """Converts metropolis-v3.0 task annotations to cosmos-reason-v1.0 training format.

    Supported metropolis-v3.0 task types
    ------------------------------------
    open_qa, bcq, bcq_openended, mcq, mcq_openended, video_summarization,
    scene_description, temporal_localization, temporal_description, causal_linkage

    Each task item is converted to one cosmos-reason-v1.0 training sample:
    - A conversation file is written to ``text/``.
    - The media file is resolved from ``raw/`` and optionally copied
      to ``media/``.
    - All samples are indexed in ``meta.json``.

    Sample ID format
    ----------------
    ``{scene_id}__{task_type}__{media_id}__{index:04d}``

    ``scene_id`` is the scene's leaf directory name for ``convert_scene``;
    for ``convert_dataset`` it's the scene's dataset-relative path with
    ``/`` replaced by ``--`` so scenes that share a leaf name don't collide
    in ``text/`` or ``meta.json``.
    """

    source_format: ClassVar[str] = "metropolis-v3.0"
    target_format: ClassVar[str] = "cosmos-reason-v1.0"

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

    # ------------------------------------------------------------------
    # CLI plumbing
    # ------------------------------------------------------------------
    @classmethod
    def register_subparser(cls, target_subparsers: "argparse._SubParsersAction") -> None:
        """Register this pair's parser under its target format.

        ``target_subparsers`` is the subparser group of a *source* subparser
        (one such group exists per ``source_format``); the pair contributes a
        subparser keyed on ``cls.target_format``. Final invocation form is
        ``cosmos-dataset convert <source> <target> [pair flags]``.
        """
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
        # Pair-specific: depends on both source layout (raw/) and target layout (media/).
        parser.add_argument(
            "--no-copy-media",
            action="store_true",
            help="Do not copy media files; reference original raw/ paths instead",
        )
        # Target-flavored: lands in the target's meta.json metadata block.
        parser.add_argument(
            "--description",
            type=str,
            help="Human-readable description added to meta.json metadata block",
        )
        parser.add_argument(
            "--license",
            type=str,
            help="License string added to meta.json metadata block",
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

        source = Path(args.path)
        if (source / "task").is_dir():
            result = self.convert_scene(
                scene_path=source,
                output_path=args.output,
                task_types=args.task or None,
                copy_media=not args.no_copy_media,
                metadata=metadata or None,
            )
        else:
            result = self.convert_dataset(
                dataset_path=source,
                output_path=args.output,
                task_types=args.task or None,
                copy_media=not args.no_copy_media,
                metadata=metadata or None,
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
    ) -> ConversionResult:
        """Convert a single metropolis-v3.0 scene to a cosmos-reason-v1.0 dataset."""
        scene_path = Path(scene_path).resolve()
        output_path = Path(output_path)

        result = ConversionResult()

        if not (scene_path / "task").is_dir():
            result.errors.append(f"task/ directory not found in {scene_path}")
            return result

        media_out = output_path / "media"
        text_out = output_path / "text"
        media_out.mkdir(parents=True, exist_ok=True)
        text_out.mkdir(parents=True, exist_ok=True)

        items_result = self._convert_scene_items(scene_path, media_out, text_out, task_types, copy_media, scene_path)
        result.samples_written = items_result["written"]
        result.samples_skipped = items_result["skipped"]
        result.warnings.extend(items_result["warnings"])
        result.errors.extend(items_result["errors"])

        self._write_meta(output_path, items_result["samples"], metadata)
        return result

    def convert_dataset(
        self,
        dataset_path: Path,
        output_path: Path,
        task_types: Optional[List[str]] = None,
        copy_media: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ConversionResult:
        """Convert all metropolis-v3.0 scenes under *dataset_path* into one dataset."""
        dataset_path = Path(dataset_path).resolve()
        output_path = Path(output_path)

        scenes = fmt.find_scenes(dataset_path, marker_dir="task")
        combined = ConversionResult()

        if not scenes:
            combined.errors.append(f"No metropolis-v3.0 scenes (containing task/) found under {dataset_path}")
            return combined

        media_out = output_path / "media"
        text_out = output_path / "text"
        media_out.mkdir(parents=True, exist_ok=True)
        text_out.mkdir(parents=True, exist_ok=True)

        all_samples: List[Dict[str, str]] = []

        for scene_path in scenes:
            items_result = self._convert_scene_items(
                scene_path, media_out, text_out, task_types, copy_media, dataset_path
            )
            all_samples.extend(items_result["samples"])
            combined.samples_written += items_result["written"]
            combined.samples_skipped += items_result["skipped"]
            combined.warnings.extend(items_result["warnings"])
            combined.errors.extend(items_result["errors"])

        self._write_meta(output_path, all_samples, metadata)
        return combined

    # ------------------------------------------------------------------
    # Core conversion logic
    # ------------------------------------------------------------------

    def _convert_scene_items(
        self,
        scene_path: Path,
        media_out: Path,
        text_out: Path,
        task_types: Optional[List[str]],
        copy_media: bool,
        media_anchor: Path,
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "samples": [],
            "written": 0,
            "skipped": 0,
            "warnings": [],
            "errors": [],
        }

        # Scene id used in sample_id / conversation filenames. For
        # convert_scene the anchor equals scene_path, so the scene's leaf
        # name is unambiguous. For convert_dataset two scenes may share a
        # leaf name (``dataset/a/scene_001`` and ``dataset/b/scene_001``);
        # flatten the dataset-relative path with ``/`` → ``--`` so their
        # conversation files and meta.json entries don't collide.
        try:
            scene_rel = scene_path.relative_to(media_anchor)
            scene_id = scene_path.name if scene_rel == Path(".") else scene_rel.as_posix().replace("/", "--")
        except ValueError:
            scene_id = scene_path.name

        # Walk contextual/ once at scene entry to learn each media_id's
        # declared ``format``. The metropolis-v3.0 schema requires both
        # ``video_id``/``image_id`` and a sibling ``format`` for every
        # video/image contextual entry, so the file in ``raw/`` is at
        # ``{media_id}.{format}``. Trusting that contract directly removes
        # the multi-extension probing the old converter did (and the
        # ``.jpg.jpg`` bug that came with its fallback path synthesis).
        media_formats = fmt.build_media_format_index(scene_path, out["warnings"])

        for task_file, task_type, idx, item in fmt.iter_task_items(
            scene_path, self.SUPPORTED_TASKS, task_types, out["warnings"]
        ):
            media_id = item.get("video_id") or item.get("image_id")
            is_video = "video_id" in item

            if not media_id:
                out["warnings"].append(f"{task_file.name}[{idx}]: missing video_id/image_id, skipping")
                out["skipped"] += 1
                continue

            fmt_ext = media_formats.get(media_id)
            if fmt_ext is None:
                out["warnings"].append(
                    f"{task_file.name}[{idx}]: no contextual entry for '{media_id}' declares a format, skipping"
                )
                out["skipped"] += 1
                continue

            media_src = fmt.find_media_file(scene_path, media_id, fmt_ext)
            if media_src is None:
                out["warnings"].append(
                    f"{task_file.name}[{idx}]: expected media file 'raw/{media_id}.{fmt_ext}' not found, skipping"
                )
                out["skipped"] += 1
                continue

            if copy_media:
                # Flatten the source-relative path so scenes sharing a media
                # basename (e.g. ``camera_01.mp4``) don't collide under media/.
                basename = media_dest_basename(media_src, media_anchor)
                dest = media_out / basename
                if not dest.exists():
                    shutil.copy2(media_src, dest)
                media_rel = f"media/{basename}"
            else:
                media_rel = str(media_src)

            # Heuristic: warn if the source question appears to already embed
            # its MCQ options. Canonical form is options-in-dict-only; the
            # converter still emits the canonical options block, so duplicate
            # content in the prompt is the user's to clean up at the fmt.
            if task_type in ("mcq", "mcq_openended"):
                options = item.get("options")
                question = item.get("question", "")
                if options and self._question_has_inline_options(question, options):
                    out["warnings"].append(
                        f"{task_file.name}[{idx}]: question likely embeds MCQ "
                        f"options inline; canonical form keeps options only "
                        f"in the 'options' dict. Options will still be "
                        f"appended from the dict — clean the source to avoid "
                        f"duplicated choices in the generated prompt."
                    )

            conv = self._build_conversation(task_type, item, is_video)
            if conv is None:
                out["warnings"].append(f"{task_file.name}[{idx}]: could not build conversation, skipping")
                out["skipped"] += 1
                continue

            sample_id = f"{scene_id}__{task_type}__{media_id}__{idx:04d}"
            conv_filename = f"{sample_id}.json"
            with open(text_out / conv_filename, "w") as f:
                json.dump(conv, f, indent=2)

            out["samples"].append(
                {
                    "id": sample_id,
                    "conversation": f"text/{conv_filename}",
                    "media": media_rel,
                }
            )
            out["written"] += 1

        return out

    # ------------------------------------------------------------------
    # Conversation builder
    # ------------------------------------------------------------------

    def _build_conversation(self, task_type: str, item: dict, is_video: bool) -> Optional[dict]:
        """Build a cosmos-reason-v1.0 conversation from one task item."""
        question = item.get("question")
        if not question:
            return None

        media_type = "video" if is_video else "image"
        placeholder = "video_0" if is_video else "image_0"

        question_text = question
        if task_type in ("mcq", "mcq_openended"):
            options = item.get("options")
            if options:
                # Canonical form: ``question`` contains only the question;
                # ``options`` holds the choices. Always emit the options block
                # from the dict. A heuristic check for inline options lives at
                # the item-loop level and only surfaces a warning — it does not
                # gate behavior here.
                options_text = "\n".join(f"{k}) {v}" for k, v in sorted(options.items()))
                question_text = f"{question}\n\n{options_text}"

        _ANSWER_INSTRUCTIONS = {
            "bcq": "Answer with only Yes or No.",
            "bcq_openended": "Answer with Yes or No, followed by a brief explanation.",
            "mcq": "Choose the correct option by letter only.",
            "mcq_openended": "Choose the correct option and provide a brief explanation.",
            "temporal_localization": "Provide the result in json format with 'mm:ss' for time depiction. Use keywords 'start', 'end' in the json output.",
        }
        instruction = _ANSWER_INSTRUCTIONS.get(task_type)
        if instruction:
            question_text = f"{question_text}\n\n{instruction}"

        user_content = [
            {"type": media_type, media_type: placeholder},
            {"type": "text", "text": question_text},
        ]

        answer_text = self._format_answer(task_type, item)
        if answer_text is None:
            return None

        assistant_turn: Dict[str, Any] = {"role": "assistant"}
        reasoning = item.get("reasoning")
        if reasoning:
            assistant_turn["reasoning_content"] = [{"type": "text", "text": str(reasoning)}]
        assistant_turn["content"] = [{"type": "text", "text": answer_text}]

        return {
            "version": "cosmos-reason-v1.0",
            "metadata": {"type": "conversation", "tags": [task_type]},
            "conversations": [
                {"role": "user", "content": user_content},
                assistant_turn,
            ],
        }

    def _format_answer(self, task_type: str, item: dict) -> Optional[str]:
        """Format the assistant answer text for a given task type."""
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
    # I/O helpers
    # ------------------------------------------------------------------

    def _write_meta(
        self,
        output_path: Path,
        samples: List[Dict[str, str]],
        metadata: Optional[Dict[str, Any]],
    ) -> None:
        """Write the meta.json index file."""
        meta_block: Dict[str, Any] = {"type": "meta"}
        if metadata:
            meta_block.update({k: v for k, v in metadata.items() if k != "type"})

        meta = {
            "version": "cosmos-reason-v1.0",
            "metadata": meta_block,
            "samples": samples,
        }
        with open(output_path / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    @staticmethod
    def _question_has_inline_options(question: str, options: Dict[str, str]) -> bool:
        """Heuristic: does *question* look like it already embeds the options inline?

        Returns True when every option letter appears as a line-starting
        enumeration marker in one of the two common MCQ styles::

            A. X     A) X

        Advisory only — powers a user-facing warning. The caller always emits
        the canonical options block from the ``options`` dict regardless.
        """
        if not options:
            return False
        for letter in options:
            if not re.search(rf"(?:^|\n)\s*{re.escape(letter)}[.)]\s", question):
                return False
        return True
