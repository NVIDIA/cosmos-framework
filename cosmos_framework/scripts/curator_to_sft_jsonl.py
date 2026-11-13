# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Convert cosmos-curator splitting-pipeline outputs into the SFT training JSONL format.

The SFT dataset loader (``sft_dataset.py``) expects each JSONL line to have::

    uuid, duration, width, height, vision_path, t2w_windows

where ``t2w_windows`` is a list of ``{start_frame, end_frame, temporal_interval, caption}``.

Curator writes a richer schema per clip at
``<curator_output>/metas_jsonl/v0/*.jsonl``. This script renames and trims those
rows into the loader's format, applies the same default filters the loader applies
at train time (so dataset counts match), and writes a sidecar
``<output>.summary.json`` with per-reason drop counts.

Transfer (control-conditioned) training
----------------------------------------
Pass ``--control-type`` to produce a JSONL suitable for ``TransferSFTDataset``
(``transfer_sft_dataset.py``) in addition to the standard fields above.  Two
extra fields are written per row:

- ``control_type`` — the control signal to use (``edge``, ``blur``, ``depth``, ``seg``).
- ``control_path`` — path to the precomputed control video; only present for
  ``depth`` and ``seg`` (edge/blur are computed on-the-fly from ``vision_path``).

Usage
-----
    # Standard T2V/I2V/V2V SFT
    python -m cosmos_framework.scripts.curator_to_sft_jsonl \\
        --curator-output outputs/curator_split/ \\
        -o outputs/curator_split/cosmos3_sft.jsonl \\
        --caption-model qwen --enhanced-caption-model qwen_lm

    # Transfer SFT — edge control (no precomputed files needed)
    python -m cosmos_framework.scripts.curator_to_sft_jsonl \\
        --curator-output outputs/curator_split/ \\
        -o outputs/curator_split/cosmos3_transfer_sft.jsonl \\
        --control-type edge

    # Transfer SFT — depth control (requires precomputed depth videos)
    python -m cosmos_framework.scripts.curator_to_sft_jsonl \\
        --curator-output outputs/curator_split/ \\
        -o outputs/curator_split/cosmos3_transfer_sft.jsonl \\
        --control-type depth --control-path-root outputs/depth_maps/

Curator must have been run with ``--upload-clip-info-in-chunks`` so that
``metas_jsonl/v0/`` is populated; otherwise the converter has no input.
"""

import json
import os
import sys
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated, Any, Literal

import tyro

# Default filters mirror sft_dataset.py defaults so the converter drops the same
# rows the loader would drop at train time. Pass None through the CLI/config to
# disable a filter for short-task datasets.
MAX_VIDEO_DURATION_S: float = 61.0
MIN_WINDOW_FRAMES: int = 61
DEFAULT_TEMPORAL_INTERVAL: int = 1

# Suffixes curator writes for captions / enhanced captions in metas_jsonl rows.
_CAPTION_SUFFIX = "_caption"
_ENHANCED_SUFFIX = "_enhanced_caption"

# Control types for transfer training.
_ON_THE_FLY_CONTROL_TYPES = frozenset({"edge", "blur"})  # computed from vision_path at train time
_PRECOMPUTED_CONTROL_TYPES = frozenset({"depth", "seg"})  # require a separate control video
_ALL_CONTROL_TYPES = _ON_THE_FLY_CONTROL_TYPES | _PRECOMPUTED_CONTROL_TYPES


def _relativize_vision_path(vision_path: str, output_jsonl: Path) -> str:
    """Rewrite ``vision_path`` relative to the output JSONL's directory.

    The SFT loader resolves relative paths against the JSONL's directory
    (``sft_dataset.py:548-550``). Curator's ``clip_location`` is typically an
    absolute filesystem path, which the loader also accepts but which doesn't
    survive moving the dataset to a different mount or container.

    Behavior:

    - URIs containing ``://`` (e.g. ``s3://bucket/key``) pass through unchanged.
    - Absolute or relative filesystem paths are rewritten relative to
      ``output_jsonl.parent``. ``os.path.relpath`` will emit ``../`` segments if
      the clip lives outside the JSONL's parent tree; that still satisfies the
      loader, which simply joins the two.
    """
    if "://" in vision_path:
        return vision_path
    return os.path.relpath(vision_path, start=output_jsonl.parent)


def _iter_metas_jsonl_files(curator_output: Path) -> Iterator[Path]:
    """Find curator metas_jsonl files under the given curator output root.

    Accepts either the splitting-pipeline output root (recommended) or the
    ``metas_jsonl/v0/`` directory itself.
    """
    nested = curator_output / "metas_jsonl" / "v0"
    if nested.is_dir():
        yield from sorted(nested.glob("*.jsonl"))
        return
    if curator_output.is_dir():
        direct = sorted(curator_output.glob("*.jsonl"))
        if direct:
            yield from direct
            return
    # Nothing matched. Caller treats an empty stream as a fatal error.


def _resolve_window_caption(
    window: dict[str, Any],
    *,
    caption_model: str | None,
    enhanced_caption_model: str | None,
) -> str | None:
    """Pick a single caption text for a window using a deterministic chain.

    Resolution order:

    1. ``{enhanced_caption_model}_enhanced_caption`` when configured and non-empty.
    2. ``{caption_model}_caption`` when configured and non-empty.
    3. First non-empty ``*_enhanced_caption`` value (alphabetical key).
    4. First non-empty ``*_caption`` value (alphabetical key).
    5. ``None`` — caller drops the window.
    """
    if enhanced_caption_model:
        candidate = (window.get(f"{enhanced_caption_model}{_ENHANCED_SUFFIX}") or "").strip()
        if candidate:
            return candidate

    if caption_model:
        candidate = (window.get(f"{caption_model}{_CAPTION_SUFFIX}") or "").strip()
        if candidate:
            return candidate

    for key in sorted(window.keys()):
        if key.endswith(_ENHANCED_SUFFIX):
            candidate = (window.get(key) or "").strip()
            if candidate:
                return candidate

    for key in sorted(window.keys()):
        if key.endswith(_CAPTION_SUFFIX) and not key.endswith(_ENHANCED_SUFFIX):
            candidate = (window.get(key) or "").strip()
            if candidate:
                return candidate

    return None


def _find_control_path(
    clip_uuid: str,
    clip_location: str,
    control_type: str,
    control_path_root: Path | None,
) -> str | None:
    """Locate a precomputed control video for a clip.

    Search order:
    1. ``{control_path_root}/{uuid}.mp4``
    2. ``{control_path_root}/{uuid}/{uuid}.mp4``
    3. ``{control_path_root}/{uuid}_{control_type}.mp4``
    4. ``{clip_dir}/{uuid}_{control_type}.mp4`` (sibling convention)
    5. ``{clip_dir}/{uuid}_{control_type}.mkv``
    """
    if control_path_root is not None:
        for candidate in [
            control_path_root / f"{clip_uuid}.mp4",
            control_path_root / clip_uuid / f"{clip_uuid}.mp4",
            control_path_root / f"{clip_uuid}_{control_type}.mp4",
        ]:
            if candidate.is_file():
                return str(candidate)

    clip_dir = Path(clip_location).parent
    for suffix in (f"_{control_type}.mp4", f"_{control_type}.mkv"):
        sibling = clip_dir / f"{clip_uuid}{suffix}"
        if sibling.is_file():
            return str(sibling)

    return None


def _compute_duration(record: dict[str, Any]) -> float | None:
    """Derive clip duration in seconds from a curator row.

    Prefer ``num_frames / framerate`` for accuracy; fall back to
    ``duration_span[1] - duration_span[0]`` if the post-transcode metadata is
    missing.
    """
    num_frames = record.get("num_frames")
    framerate = record.get("framerate")
    if isinstance(num_frames, int) and isinstance(framerate, int | float) and framerate > 0:
        return float(num_frames) / float(framerate)
    span = record.get("duration_span")
    if isinstance(span, list | tuple) and len(span) == 2:
        try:
            return float(span[1]) - float(span[0])
        except (TypeError, ValueError):
            return None
    return None


def _build_sft_row(
    record: dict[str, Any],
    *,
    caption_model: str | None,
    enhanced_caption_model: str | None,
    min_short_edge: int,
    min_window_frames: int | None,
    max_duration_s: float | None,
    temporal_interval: int,
    control_type: str | None = None,
    control_path_root: Path | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Translate a curator metas_jsonl row into an SFT row, or report a drop reason.

    Returns ``(row, None)`` for kept records and ``(None, reason)`` for drops.

    When ``control_type`` is set, two extra fields are written:
    - ``control_type`` — always present.
    - ``control_path`` — only for ``depth``/``seg``; clips without a matching
      precomputed file are dropped with reason ``missing_control_path``.
    """
    clip_uuid = record.get("span_uuid")
    clip_location = record.get("clip_location")
    width = record.get("width")
    height = record.get("height")
    num_frames = record.get("num_frames")
    framerate = record.get("framerate")
    duration_s = _compute_duration(record)

    if not clip_uuid or not clip_location:
        return None, "missing_identity"
    if width is None or height is None or num_frames is None or framerate is None or duration_s is None:
        return None, "missing_clip_metadata"
    if max_duration_s is not None and duration_s > max_duration_s:
        return None, "duration_too_long"
    if min_short_edge > 0 and min(int(width), int(height)) < min_short_edge:
        return None, "short_edge_too_small"

    # For precomputed control types, require a matching control video.
    control_path: str | None = None
    if control_type in _PRECOMPUTED_CONTROL_TYPES:
        control_path = _find_control_path(str(clip_uuid), str(clip_location), control_type, control_path_root)
        if control_path is None:
            return None, "missing_control_path"

    windows = record.get("windows") or []
    t2w_windows: list[dict[str, Any]] = []
    for window in windows:
        start_frame = window.get("start_frame")
        end_frame = window.get("end_frame")
        if not isinstance(start_frame, int) or not isinstance(end_frame, int):
            continue
        frames_in_window = end_frame - start_frame + 1
        if min_window_frames is not None and frames_in_window < min_window_frames:
            continue
        caption_text = _resolve_window_caption(
            window,
            caption_model=caption_model,
            enhanced_caption_model=enhanced_caption_model,
        )
        if caption_text is None:
            continue
        t2w_windows.append(
            {
                "start_frame": start_frame,
                "end_frame": end_frame,
                "temporal_interval": temporal_interval,
                "caption": caption_text,
            }
        )

    if not t2w_windows:
        return None, "no_valid_window"

    row: dict[str, Any] = {
        "uuid": str(clip_uuid),
        "duration": float(duration_s),
        "width": int(width),
        "height": int(height),
        "nb_frames": int(num_frames),
        "framerate": float(framerate),
        "vision_path": str(clip_location),
        "t2w_windows": t2w_windows,
    }
    if control_type is not None:
        row["control_type"] = control_type
    if control_path is not None:
        row["control_path"] = control_path
    return row, None


def main(  # noqa: PLR0913
    curator_output: Annotated[
        Path,
        tyro.conf.arg(help="Curator splitting-pipeline output root (the dir containing metas_jsonl/v0/)."),
    ],
    output: Annotated[Path, tyro.conf.arg(aliases=("-o",), help="Output JSONL path.")],
    caption_model: Annotated[
        str | None,
        tyro.conf.arg(
            help="Curator caption-model name (e.g. 'qwen'); used to pick {model}_caption fields. "
            "Defaults to the first *_caption key encountered when unset.",
        ),
    ] = None,
    enhanced_caption_model: Annotated[
        str | None,
        tyro.conf.arg(
            help="Curator enhancement-model name (e.g. 'qwen_lm'); used to pick "
            "{model}_enhanced_caption fields. Preferred over caption_model when both are present.",
        ),
    ] = None,
    min_short_edge: Annotated[
        int,
        tyro.conf.arg(help="Drop clips whose shortest spatial edge is below this value. 0 disables."),
    ] = 0,
    min_window_frames: Annotated[
        int | None,
        tyro.conf.arg(
            help=(
                f"Drop windows shorter than this. Default {MIN_WINDOW_FRAMES} matches sft_dataset.py. "
                "Set to None for short-task datasets."
            ),
        ),
    ] = MIN_WINDOW_FRAMES,
    max_duration_s: Annotated[
        float | None,
        tyro.conf.arg(
            help=(
                f"Drop clips longer than this. Default {MAX_VIDEO_DURATION_S} matches sft_dataset.py. "
                "Set to None for short-task datasets."
            ),
        ),
    ] = MAX_VIDEO_DURATION_S,
    temporal_interval: Annotated[
        int,
        tyro.conf.arg(help="temporal_interval to record on every t2w_window."),
    ] = DEFAULT_TEMPORAL_INTERVAL,
    control_type: Annotated[
        Literal["edge", "blur", "depth", "seg"] | None,
        tyro.conf.arg(
            help=(
                "Produce a transfer SFT JSONL by adding 'control_type' (and optionally "
                "'control_path') to each row. 'edge' and 'blur' are computed on-the-fly "
                "at train time — no precomputed files needed. 'depth' and 'seg' require "
                "precomputed control videos; pass --control-path-root to locate them. "
                "Omit to produce a standard T2V/I2V/V2V SFT JSONL (default behaviour)."
            ),
        ),
    ] = None,
    control_path_root: Annotated[
        Path | None,
        tyro.conf.arg(
            help=(
                "Directory containing precomputed control videos for --control-type depth/seg. "
                "Searched as {root}/{uuid}.mp4 or {root}/{uuid}/{uuid}.mp4. "
                "Not needed for edge or blur."
            ),
        ),
    ] = None,
) -> None:
    """Build an SFT JSONL from a curator splitting-pipeline output directory."""
    if control_type in _PRECOMPUTED_CONTROL_TYPES and control_path_root is None:
        print(
            f"WARNING: --control-type={control_type!r} requires precomputed control files. "
            "Pass --control-path-root to locate them; clips without a matching file will be "
            "dropped with reason 'missing_control_path'.",
            file=sys.stderr,
        )

    jsonl_files = list(_iter_metas_jsonl_files(curator_output))
    if not jsonl_files:
        print(
            f"ERROR: No metas_jsonl files found under {curator_output}. "
            "Re-run the curator splitting pipeline with --upload-clip-info-in-chunks.",
            file=sys.stderr,
        )
        sys.exit(1)

    kept_rows: list[dict[str, Any]] = []
    drops: Counter[str] = Counter()
    seen_records = 0

    for jsonl_path in jsonl_files:
        with jsonl_path.open("r") as src:
            for line in src:
                line = line.strip()
                if not line:
                    continue
                seen_records += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    print(f"  SKIP malformed JSON in {jsonl_path}: {exc}")
                    drops["malformed_json"] += 1
                    continue
                row, reason = _build_sft_row(
                    record,
                    caption_model=caption_model,
                    enhanced_caption_model=enhanced_caption_model,
                    min_short_edge=min_short_edge,
                    min_window_frames=min_window_frames,
                    max_duration_s=max_duration_s,
                    temporal_interval=temporal_interval,
                    control_type=control_type,
                    control_path_root=control_path_root,
                )
                if row is None:
                    drops[reason or "unknown"] += 1
                    continue
                row["vision_path"] = _relativize_vision_path(row["vision_path"], output)
                if "control_path" in row:
                    row["control_path"] = _relativize_vision_path(row["control_path"], output)
                kept_rows.append(row)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as dst:
        for row in kept_rows:
            dst.write(json.dumps(row) + "\n")

    summary = {
        "curator_output": str(curator_output),
        "output_jsonl": str(output),
        "shards_read": len(jsonl_files),
        "records_seen": seen_records,
        "records_kept": len(kept_rows),
        "records_dropped": sum(drops.values()),
        "drops_by_reason": dict(drops),
        "filters": {
            "max_duration_s": max_duration_s,
            "min_window_frames": min_window_frames,
            "min_short_edge": min_short_edge,
        },
        "caption_model": caption_model,
        "enhanced_caption_model": enhanced_caption_model,
        "temporal_interval": temporal_interval,
        "control_type": control_type,
        "control_path_root": str(control_path_root) if control_path_root else None,
    }
    summary_path = output.with_suffix(output.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"Read {seen_records} records from {len(jsonl_files)} shard(s) under {curator_output}")
    print(f"Wrote {len(kept_rows)} records → {output}")
    if drops:
        print("Drops by reason:")
        for reason, count in sorted(drops.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  {reason}: {count}")
    print(f"Summary: {summary_path}")
    if not kept_rows:
        print("ERROR: No valid records written.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    tyro.cli(main)
