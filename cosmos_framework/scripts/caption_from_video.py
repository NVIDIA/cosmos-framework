# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Generate structured-JSON and dense narrative captions from video files using a VLM.

Each video is passed directly to a VLM server via a ``video_url`` content part
using a ``file://`` path.  A structured prompt template guides the VLM through
a two-phase captioning process (Phase 1: structured-JSON scene analysis →
Phase 2: dense narrative rewrite).  Both outputs are persisted: ``caption.json``
(the canonical structured caption, with the dense narrative embedded as
``temporal_caption`` and the clip's real media fields) and ``caption.txt`` (the
dense narrative on its own).

The VLM server must support the OpenAI chat-completions API with vision and
must be started with ``--allowed-local-media-path`` pointing to the root of
your video storage so that it can read video files by path.  Compatible servers
include vLLM serving Qwen2-VL / Qwen3-VL, LLaVA-Next-Video, etc.

Example usage::

    # Caption videos listed in a JSONL file (each line has {"name": ..., "vision_path": ...})
    python -m cosmos_framework.scripts.caption_from_video \
        -i samples.jsonl -o /output/captions \
        --server http://localhost:8000/v1

    # Caption a single video directly
    python -m cosmos_framework.scripts.caption_from_video \
        --video /path/to/video.mp4 -o /output/captions \
        --server http://localhost:8000/v1

    # Caption a directory of videos
    python -m cosmos_framework.scripts.caption_from_video \
        --video /path/to/videos/ -o /output/captions \
        --server http://localhost:8000/v1
"""

import asyncio
import json
from pathlib import Path
from typing import Annotated

import openai
import pydantic
import tyro
from tqdm import tqdm

from cosmos_framework.inference.args import OmniSampleOverrides
from cosmos_framework.inference.common.args import VIDEO_EXTENSIONS
from cosmos_framework.inference.structured_caption import (
    assemble_caption_json,
    extract_xml_tag,
    media_fields_from_metadata,
    parse_structured_caption,
)
from cosmos_framework.scripts.video_metadata import probe_video_metadata
from cosmos_framework.utils import log

_PACKAGE_DIR = Path(__file__).parents[1].absolute()


class Args(pydantic.BaseModel):
    input_files: Annotated[list[Path] | None, tyro.conf.arg(aliases=("-i",))] = None
    """Path to input manifest files (JSON/JSONL).
    Each entry needs a 'vision_path' (a local path or an http(s)/data URL) and may
    include 'name' and a 'media' dict (resolution/aspect_ratio/duration/fps) — the
    latter is used as the caption's media fields when the video is a remote URL that
    ffprobe cannot read locally. Mutually exclusive with --video."""

    video: Annotated[Path | None, tyro.conf.arg(aliases=("-v",))] = None
    """Path to a single video file or a directory of videos.
    Mutually exclusive with --input_files."""

    output_dir: Annotated[Path, tyro.conf.arg(aliases=("-o",))]
    """Output directory for generated captions."""

    server: str = "http://localhost:8000/v1"
    """The URL of the OpenAI-compatible VLM API server."""
    model: str | None = None
    """The model to use. If not provided, the first model served will be used."""

    max_workers: int = 16
    """Maximum number of concurrent requests to the API."""
    max_retries: int = 5
    """Maximum number of retries for each request."""
    timeout: float = 600.0
    """Per-request client timeout in seconds; a hung request fails after this and is retried."""

    prompt_template_path: Path | None = None
    """Path to a custom prompt template. Defaults to the built-in video_captioner.txt."""

    debug: bool = False
    """If True, save raw API responses for debugging."""


def _is_remote_ref(ref: str) -> bool:
    """True if ``ref`` is something the server fetches itself (URL / data URI)."""
    return "://" in ref or ref.startswith("data:")


def _video_url(video_ref: str) -> str:
    """Map a local path or remote ref to the ``video_url`` string the server receives.

    Remote refs (``http(s)://`` or ``data:``) are passed through untouched, so the
    server fetches them itself — this is what makes captioning work against a remote
    VLM endpoint.  Local paths become ``file://`` URLs, which require a local server
    started with ``--allowed-local-media-path``.
    """
    if _is_remote_ref(video_ref):
        return video_ref
    return f"file://{Path(video_ref).absolute()}"


def _build_vlm_messages(video_ref: str, prompt_template: str) -> list[dict]:
    """Build an OpenAI-compatible multimodal message with a video + text prompt."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": _video_url(video_ref)}},
                {"type": "text", "text": prompt_template},
            ],
        }
    ]


async def _process_single(
    args: Args,
    client: openai.AsyncOpenAI,
    name: str,
    video_ref: str,
    media_override: dict | None,
    prompt_template: str,
) -> bool:
    assert args.model

    output_dir = args.output_dir / name
    messages = _build_vlm_messages(video_ref, prompt_template)

    for i_retry in range(args.max_retries):
        try:
            response = await client.chat.completions.create(
                model=args.model,
                messages=messages,
                max_tokens=2048,
                temperature=0.7,
                top_p=0.8,
                extra_body={"top_k": 20, "min_p": 0.0},
            )
        except Exception as e:
            log.warning(f"[{i_retry + 1}/{args.max_retries}] API Error for {name}: {e}")
            await asyncio.sleep(1)
            continue

        if args.debug:
            retry_dir = output_dir / f"{i_retry}"
            retry_dir.mkdir(parents=True, exist_ok=True)
            (retry_dir / "response.json").write_text(response.model_dump_json())

        assert len(response.choices) == 1
        choice = response.choices[0]
        if choice.finish_reason != "stop" or not choice.message.content:
            log.warning(f"[{i_retry + 1}/{args.max_retries}] Invalid response for {name}")
            continue

        text = choice.message.content.strip()
        final_prompt = extract_xml_tag(text, "final_prompt")
        if final_prompt is None:
            log.warning(f"[{i_retry + 1}/{args.max_retries}] Failed to extract final prompt for {name}")
            continue

        scene_draft = parse_structured_caption(text)
        if scene_draft is None:
            log.warning(f"[{i_retry + 1}/{args.max_retries}] Failed to parse scene_draft JSON for {name}")
            continue

        # Media fields: prefer a manifest-provided override; else ffprobe a local
        # file; else leave empty (e.g. a remote URL ffprobe cannot read).
        if media_override is not None:
            media = media_override
        elif not _is_remote_ref(video_ref):
            try:
                media = media_fields_from_metadata(probe_video_metadata(video_ref))
            except Exception as e:  # noqa: BLE001 - degrade gracefully, keep the caption
                log.warning(f"ffprobe failed for {name}: {e}; writing caption_json without media fields")
                media = {}
        else:
            media = {}

        try:
            caption_json = assemble_caption_json(scene_draft, final_prompt, media)
        except pydantic.ValidationError as e:
            log.warning(f"[{i_retry + 1}/{args.max_retries}] caption_json failed validation for {name}: {e}")
            continue

        output_dir.mkdir(parents=True, exist_ok=True)

        sample_overrides = OmniSampleOverrides(
            name=name,
            prompt=final_prompt,
            vision_path=video_ref,
            output_dir=output_dir,
        )
        (output_dir / "sample_args.json").write_text(sample_overrides.model_dump_json())
        (output_dir / "caption.txt").write_text(final_prompt)
        (output_dir / "caption.json").write_text(json.dumps(caption_json, indent=2, ensure_ascii=False))

        # Advisory: the SFT loader truncates very long prompts (see _MAX_CAPTION_TOKENS
        # in sft_dataset.py). ~4 chars/token is a rough guide; warn if the serialized
        # JSON looks large so it can be checked against the recipe's max_caption_tokens.
        approx_tokens = len(json.dumps(caption_json, ensure_ascii=False)) // 4
        if approx_tokens > 1024:
            log.warning(
                f"{name}: structured caption is ~{approx_tokens} tokens (rough estimate); "
                "ensure the SFT recipe's max_caption_tokens covers it to avoid truncation."
            )
        return True

    log.warning(f"Failed to get caption for {name}")
    return False


async def _process_with_semaphore(
    args: Args,
    client: openai.AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    name: str,
    video_ref: str,
    media_override: dict | None,
    prompt_template: str,
) -> bool:
    async with semaphore:
        return await _process_single(args, client, name, video_ref, media_override, prompt_template)


def _read_manifest_entries(input_files: list[Path]) -> list[tuple[str, str, dict | None]]:
    """Parse ``-i`` JSON/JSONL manifests into ``(name, video_ref, media)`` tuples.

    Each entry must have a ``vision_path`` (a local path or an ``http(s)``/``data``
    URL) and may carry an optional ``name`` and an optional ``media`` dict (the
    structured caption's media fields: resolution/aspect_ratio/duration/fps).  The
    ``media`` override lets remote-URL videos — which ffprobe cannot read — still
    get accurate media fields.
    """
    items: list[tuple[str, str, dict | None]] = []
    for path in input_files:
        text = path.read_text()
        if path.suffix == ".jsonl":
            entries = [json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            data = json.loads(text)
            entries = data if isinstance(data, list) else [data]
        for e in entries:
            vp = e.get("vision_path")
            name = e.get("name")
            if not vp:
                log.warning(f"Skipping entry with no vision_path: {name or '?'}")
                continue
            if Path(vp).suffix.lower() not in VIDEO_EXTENSIONS:
                log.warning(f"Skipping '{name or vp}': vision_path is not a video ({Path(vp).suffix})")
                continue
            items.append((name or Path(vp).stem, vp, e.get("media")))
    return items


def _collect_video_items(args: Args) -> list[tuple[str, str, dict | None]]:
    """Return ``(name, video_ref, media_override)`` items from the CLI arguments.

    ``video_ref`` is a local filesystem path or a remote URL (``http(s)``/``data``).
    """
    items: list[tuple[str, str, dict | None]] = []

    if args.input_files:
        items = _read_manifest_entries(args.input_files)
    elif args.video:
        if args.video.is_dir():
            for vp in sorted(args.video.iterdir()):
                if vp.suffix.lower() in VIDEO_EXTENSIONS:
                    items.append((vp.stem, str(vp), None))
        elif args.video.is_file():
            items.append((args.video.stem, str(args.video), None))
        else:
            raise FileNotFoundError(f"Video path does not exist: {args.video}")

    if not items:
        raise ValueError("No video inputs found. Provide --input_files (-i) or --video (-v).")
    return items


async def caption_from_video(args: Args):
    if args.input_files and args.video:
        raise ValueError("Provide either --input_files or --video, not both.")

    if args.prompt_template_path:
        prompt_template = args.prompt_template_path.read_text()
    else:
        prompt_template = (_PACKAGE_DIR / "inference/defaults/video_captioner.txt").read_text()

    items = _collect_video_items(args)

    client = openai.AsyncOpenAI(
        api_key="EMPTY",
        base_url=args.server,
        timeout=args.timeout,
    )
    if not args.model:
        models = await client.models.list()
        args.model = models.data[0].id
        log.info(f"Using model: {args.model}")

    semaphore = asyncio.Semaphore(args.max_workers)

    tasks = [
        _process_with_semaphore(
            args=args,
            client=client,
            semaphore=semaphore,
            name=name,
            video_ref=video_ref,
            media_override=media,
            prompt_template=prompt_template,
        )
        for name, video_ref, media in items
    ]
    n_success = 0
    for result in tqdm(asyncio.as_completed(tasks), desc="Captioning", total=len(tasks)):
        if await result:
            n_success += 1

    log.info(f"{n_success}/{len(tasks)} videos were successfully captioned")


def main():
    args = tyro.cli(Args, description=__doc__)
    asyncio.run(caption_from_video(args))


if __name__ == "__main__":
    main()
