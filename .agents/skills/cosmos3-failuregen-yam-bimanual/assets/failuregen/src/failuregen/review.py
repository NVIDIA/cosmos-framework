# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Artifacts for looking at results: side-by-side clips and contact sheets.

The comparison that carries a result is the induced failure against the same episode untouched.
Which donor supplied the actions is provenance -- useful while debugging a transplant, noise on a
slide -- so it is off unless asked for.

Contact sheets exist because a numeric ratio says the pane changed, not that the object fell. The
sheet is where that gets confirmed, and it samples the wrist pane rather than the wide view for
the same reason scoring does.
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw

from . import annotations as ann
from .config import Profile
from .scoring import RunScore, wrist_pane


def _read(mp4: Path) -> list[Image.Image]:
    with av.open(str(mp4)) as c:
        return [f.to_image() for f in c.decode(c.streams.video[0])]


def _front_only(images: list[Image.Image], fraction: float) -> list[Image.Image]:
    w, h = images[0].size
    return [im.crop((0, 0, w, int(round(h * fraction)))) for im in images]


def _gt_front(profile: Profile, collection: str, episode: int, t0: float, n: int) -> list[Image.Image]:
    """Recorded wide-camera frames, clamped to the episode's last frame.

    A run placed in a final phase can extend past the end of its episode -- generation does not
    care, it only needs a start frame and actions, but asking a decoder for a timestamp that does
    not exist loses the whole comparison.
    """
    from lerobot.datasets.video_utils import decode_video_frames

    path = ann.video_path(profile, collection, episode, profile.cameras.front)
    with av.open(str(path)) as c:
        stream = c.streams.video[0]
        dur = float(stream.duration * stream.time_base) if stream.duration else 0.0
    fps = profile.dataset.target_fps
    last = max(0.0, dur - 1.0 / profile.dataset.source_fps) if dur else float("inf")
    times = [min(t0 + i / fps, last) for i in range(n)]
    dec = decode_video_frames(str(path), times, 1.0 / profile.dataset.source_fps, backend="pyav")
    return [Image.fromarray((dec[i].clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()) for i in range(n)]


def grid(profile: Profile, run_dir: Path, show_donor: bool = False) -> Path:
    """Induced failure beside the same episode untouched, with the injection marked."""
    record = json.loads((run_dir / "record.json").read_text())
    fps = int(record.get("fps", profile.dataset.target_fps))
    synth = _front_only(_read(run_dir / "synth.mp4"), profile.cameras.front_fraction)
    n = len(synth)
    host = record["host"]
    columns = [("FAILURE INDUCED (generated)", synth)]
    donor = record.get("donor")
    if show_donor and donor:
        columns.append(
            (
                f"DONOR ({donor['source']} ep{donor['episode_index']})",
                _gt_front(profile, donor["source"], int(donor["episode_index"]), float(donor["t_start"]), n),
            )
        )
    label = (
        "REAL (recorded, same episode)"
        if record.get("mode") == "reproduction"
        else "CONTROL (same episode, unmodified)"
    )
    columns.append((label, _gt_front(profile, host["source"], int(host["episode_index"]), float(host["t_start"]), n)))

    mark = int(round(float(record.get("failure_time_s", 0.0)) * fps))
    out = run_dir / "grid.mp4"
    cw, ch = columns[0][1][0].size
    width, height = cw * len(columns) - (cw * len(columns)) % 2, ch - ch % 2
    tb = Fraction(1, fps)
    with av.open(str(out), "w") as container:
        stream = container.add_stream("h264", rate=fps)
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        stream.options = {"crf": "18"}
        stream.codec_context.time_base = tb
        for i in range(min(len(c[1]) for c in columns)):
            canvas = Image.new("RGB", (cw * len(columns), ch), (12, 12, 12))
            for c, (text, frames) in enumerate(columns):
                im = frames[i].resize((cw, ch))
                d = ImageDraw.Draw(im)
                d.rectangle([0, 0, 8 * len(text) + 10, 16], fill=(0, 0, 0))
                d.text((3, 3), text, fill=(255, 255, 0))
                if i >= mark:
                    d.rectangle([0, 0, cw - 1, ch - 1], outline=(255, 60, 60), width=3)
                canvas.paste(im, (c * cw, 0))
            if canvas.size != (width, height):
                canvas = canvas.resize((width, height))
            frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(np.asarray(canvas)), format="rgb24")
            frame.pts, frame.time_base = i, tb
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    return out


def contact_sheet(
    profile: Profile, run_dirs: list[Path], out_jpg: Path, *, columns: int = 8, tile_width: int = 240
) -> Path:
    """One row per run of its holding-hand wrist pane, red-bordered after the injection."""
    rows = []
    for run_dir in run_dirs:
        record = json.loads((run_dir / "record.json").read_text())
        arm = record.get("arm") or next(iter(profile.robot.arms))
        frames = wrist_pane(profile, run_dir / "synth.mp4", arm)
        fps = int(record.get("fps", profile.dataset.target_fps))
        mark = int(round(float(record.get("failure_time_s", 0.0)) * fps))
        idx = np.linspace(0, len(frames) - 1, columns).round().astype(int)
        rows.append((record.get("name") or run_dir.name, [(Image.fromarray(frames[i]), int(i >= mark)) for i in idx]))
    if not rows:
        raise ValueError("nothing to render")

    w0, h0 = rows[0][1][0][0].size
    tw, th = tile_width, int(round(tile_width * h0 / w0))
    sheet = Image.new("RGB", (tw * columns, th * len(rows)), (16, 16, 16))
    for r, (name, tiles) in enumerate(rows):
        for c, (tile, after) in enumerate(tiles):
            im = tile.resize((tw, th))
            d = ImageDraw.Draw(im)
            if after:
                d.rectangle([0, 0, tw - 1, th - 1], outline=(255, 60, 60), width=3)
            if c == 0:
                text = name[:30]
                d.rectangle([0, 0, 7 * len(text) + 8, 14], fill=(0, 0, 0))
                d.text((3, 2), text, fill=(255, 255, 0))
            sheet.paste(im, (c * tw, r * th))
    out_jpg.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_jpg, quality=88)
    return out_jpg


def gallery(profile: Profile, root: Path, scores: list[RunScore], *, top_per_recipe: int = 8) -> Path:
    """A markdown index plus one contact sheet per recipe, ordered by effect size."""
    by_recipe: dict[str, list[RunScore]] = {}
    for s in scores:
        by_recipe.setdefault(s.recipe or "unknown", []).append(s)

    lines = [
        "# Induced failure gallery\n",
        f"{len(scores)} injections under `{root.name}`. Each row is one run; the red border "
        "starts at the injection. `ratio` is the wrist-pane change of the holding hand "
        "divided by the same clip untouched, so it is only meaningful against that control.\n",
    ]
    for recipe in sorted(by_recipe):
        items = [s for s in by_recipe[recipe] if s.ratio is not None]
        items.sort(key=lambda s: -(s.ratio or 0))
        if not items:
            continue
        picked = items[:top_per_recipe]
        sheet = contact_sheet(profile, [root / s.name for s in picked], root / f"{recipe}_sheet.jpg")
        hits = sum(1 for s in items if s.hit)
        lines.append(f"\n## {recipe}  ({hits}/{len(items)} above threshold)\n")
        lines.append(f"![{recipe}]({sheet.name})\n")
        for s in picked:
            lines.append(f"- `{s.name}` ratio {s.ratio:.2f} (post {s.post:.3f} vs control {s.control:.3f})\n")
    out = root / "GALLERY.md"
    out.write_text("".join(lines))
    return out
