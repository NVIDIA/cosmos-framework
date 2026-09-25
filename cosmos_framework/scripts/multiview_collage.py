# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Collage per-camera multiview outputs into a vehicle-spatial review video.

python -m cosmos_framework.scripts.multiview_collage results/my_run
python -m cosmos_framework.scripts.multiview_collage results/my_run --visualize_control_input
"""

from __future__ import annotations

import argparse
import glob
import math
import re
import shutil
import subprocess
from pathlib import Path

_VIEW_RE = re.compile(r"^(?P<kind>vision|control)_view(?P<idx>\d+)_(?P<camera>.+)\.mp4$")
_FFMPEG_SIZE_RE = re.compile(r"(\d{2,5})x(\d{2,5})")
MAX_GIF_BYTES = 10 * 1024 * 1024
_GIF_FPS_CANDIDATES = (12.0, 10.0, 8.0)
_GIF_MIN_WIDTH = 160

# Subset placement starts on the joint cam+lidar 4x4 ring, then empty rows
# and columns are dropped. The full 11-view set uses a 3-4-3-1 vehicle layout
# with shorter rows centered on the 4-wide middle row:
#
#        left_fish    front_fish    right_fish
#   cross_left    front_wide    front_tele    cross_right
#        rear_left    rear_tele     rear_right
#                      rear_fish
_SURROUND_ROWS = 4
_SURROUND_COLS = 4
_MADS_SURROUND_CAMERA_CELLS: dict[str, tuple[int, int]] = {
    "camera_left_fisheye_200fov": (0, 0),
    "camera_front_fisheye_200fov": (0, 1),
    "camera_front_wide_120fov": (0, 2),
    "camera_right_fisheye_200fov": (0, 3),
    "camera_cross_left_120fov": (1, 0),
    "camera_front_tele_30fov": (1, 2),
    "camera_cross_right_120fov": (1, 3),
    "camera_rear_left_70fov": (2, 0),
    "camera_rear_right_70fov": (2, 3),
    "camera_rear_fisheye_200fov": (3, 1),
    "camera_rear_tele_30fov": (3, 2),
}
_MADS_11VIEW_ROWS = 4
_MADS_11VIEW_COLS = 4
_MADS_11VIEW_CAMERA_CELLS: dict[str, tuple[float, float]] = {
    "camera_left_fisheye_200fov": (0, 0.5),
    "camera_front_fisheye_200fov": (0, 1.5),
    "camera_right_fisheye_200fov": (0, 2.5),
    "camera_cross_left_120fov": (1, 0),
    "camera_front_wide_120fov": (1, 1),
    "camera_front_tele_30fov": (1, 2),
    "camera_cross_right_120fov": (1, 3),
    "camera_rear_left_70fov": (2, 0.5),
    "camera_rear_tele_30fov": (2, 1.5),
    "camera_rear_right_70fov": (2, 2.5),
    "camera_rear_fisheye_200fov": (3, 1.5),
}


def choose_camera_grid(n: int) -> tuple[int, int]:
    """Pack n unknown cameras into a grid whose canvas is near 4:3."""
    if n < 1:
        raise ValueError(f"Need at least one camera, got {n}")
    cols = max(1, round(math.sqrt(n * 4 / 3)))
    return cols, math.ceil(n / cols)


def _iter_ring_cells(rows: int, cols: int) -> list[tuple[int, int]]:
    """Walk the outer ring clockwise from the top-left cell."""
    cells: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()

    def add(row: int, col: int) -> None:
        if 0 <= row < rows and 0 <= col < cols and (row, col) not in seen:
            seen.add((row, col))
            cells.append((row, col))

    for col in range(cols):
        add(0, col)
    for row in range(1, rows):
        add(row, cols - 1)
    if rows > 1:
        for col in range(cols - 2, -1, -1):
            add(rows - 1, col)
    if cols > 1:
        for row in range(rows - 2, 0, -1):
            add(row, 0)
    return cells


def _occupied(cells: dict[str, tuple[int, int]]) -> set[tuple[int, int]]:
    return set(cells.values())


def _shape(cells: dict[str, tuple[int, int]]) -> tuple[int, int]:
    return max(row for row, _ in cells.values()) + 1, max(col for _, col in cells.values()) + 1


def _drop_empty_axes(cells: dict[str, tuple[int, int]]) -> dict[str, tuple[int, int]]:
    """Remove rows and columns that contain no camera."""
    rows_used = sorted({row for row, _ in cells.values()})
    cols_used = sorted({col for _, col in cells.values()})
    row_map = {row: index for index, row in enumerate(rows_used)}
    col_map = {col: index for index, col in enumerate(cols_used)}
    return {camera: (row_map[row], col_map[col]) for camera, (row, col) in cells.items()}


def _can_merge_rows(cells: dict[str, tuple[int, int]], row_a: int, row_b: int) -> bool:
    cols_a = {col for row, col in cells.values() if row == row_a}
    cols_b = {col for row, col in cells.values() if row == row_b}
    return not (cols_a & cols_b)


def _can_merge_cols(cells: dict[str, tuple[int, int]], col_a: int, col_b: int) -> bool:
    rows_a = {row for row, col in cells.values() if col == col_a}
    rows_b = {row for row, col in cells.values() if col == col_b}
    return not (rows_a & rows_b)


def _merge_rows(cells: dict[str, tuple[int, int]], row_a: int, row_b: int) -> dict[str, tuple[int, int]]:
    merged: dict[str, tuple[int, int]] = {}
    for camera, (row, col) in cells.items():
        if row == row_b:
            row = row_a
        elif row > row_b:
            row -= 1
        merged[camera] = (row, col)
    return merged


def _merge_cols(cells: dict[str, tuple[int, int]], col_a: int, col_b: int) -> dict[str, tuple[int, int]]:
    merged: dict[str, tuple[int, int]] = {}
    for camera, (row, col) in cells.items():
        if col == col_b:
            col = col_a
        elif col > col_b:
            col -= 1
        merged[camera] = (row, col)
    return merged


def _compact_vehicle_cells(cells: dict[str, tuple[int, int]]) -> tuple[int, int, dict[str, tuple[int, int]]]:
    """Keep vehicle-spatial order but drop missing-camera black tiles."""
    cells = _drop_empty_axes(cells)
    while True:
        rows, cols = _shape(cells)
        merged = False
        for row in range(rows - 1):
            if _can_merge_rows(cells, row, row + 1):
                cells = _merge_rows(cells, row, row + 1)
                merged = True
                break
        if merged:
            continue
        for col in range(cols - 1):
            if _can_merge_cols(cells, col, col + 1):
                cells = _merge_cols(cells, col, col + 1)
                merged = True
                break
        if not merged:
            return rows, cols, cells


def choose_vehicle_cells(cameras: list[str]) -> tuple[int, int, dict[str, tuple[float, float]]]:
    """Return ``(rows, cols, camera -> (row, col))`` in vehicle-spatial order.

    ``col`` may be a half-tile (``0.5``, ``1.5``, …) so shorter 11-view rows can
    sit centered on the 4-wide middle row.
    """
    if not cameras:
        raise ValueError("Need at least one camera")
    if any(camera in _MADS_SURROUND_CAMERA_CELLS for camera in cameras):
        leftover = [camera for camera in cameras if camera not in _MADS_SURROUND_CAMERA_CELLS]
        present = [camera for camera in cameras if camera in _MADS_SURROUND_CAMERA_CELLS]
        if not leftover and set(present) == set(_MADS_11VIEW_CAMERA_CELLS):
            return _MADS_11VIEW_ROWS, _MADS_11VIEW_COLS, dict(_MADS_11VIEW_CAMERA_CELLS)
        rows, cols = _SURROUND_ROWS, _SURROUND_COLS
        cells: dict[str, tuple[int, int]] = {camera: _MADS_SURROUND_CAMERA_CELLS[camera] for camera in present}
        occupied = _occupied(cells)
        free = [cell for cell in _iter_ring_cells(rows, cols) if cell not in occupied]
        free += [
            (row, col)
            for row in range(rows)
            for col in range(cols)
            if (row, col) not in occupied and (row, col) not in free
        ]
        if len(leftover) > len(free):
            raise ValueError(f"Need {len(cameras)} cells but the 4x4 board only has {len(free) + len(cells)}")
        # The guard above rules out a short 'free', so no camera can be dropped here; 'free' is
        # normally longer than 'leftover', which is what strict=False allows.
        for camera, cell in zip(leftover, free, strict=False):
            cells[camera] = cell
        packed_rows, packed_cols, packed = _compact_vehicle_cells(cells)
        return packed_rows, packed_cols, {camera: (float(row), float(col)) for camera, (row, col) in packed.items()}

    cols, rows = choose_camera_grid(len(cameras))
    return rows, cols, {camera: (float(index // cols), float(index % cols)) for index, camera in enumerate(cameras)}


def cell_xy(row: float, col: float, tile_w: int, tile_h: int) -> tuple[int, int]:
    """Pixel origin of a tile; even so H.264 yuv420p accepts the stack."""
    x = int(col * tile_w)
    y = int(row * tile_h)
    return x - x % 2, y - y % 2


def tile_size(tile_width: int, src_w: int, src_h: int) -> tuple[int, int]:
    """Even WxH box from the first view, used for every camera tile."""
    w = max(2, tile_width - tile_width % 2)
    h = max(2, round(src_h * w / src_w))
    return w, h - h % 2


def tile_filter(w: int, h: int) -> str:
    """Fit any aspect ratio into a fixed box so stacked tiles share one size."""
    return f"scale={w}:{h}:force_original_aspect_ratio=decrease,setsar=1,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:black"


def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
    except ImportError as error:
        raise RuntimeError("No ffmpeg on PATH and imageio-ffmpeg is not installed.") from error
    return imageio_ffmpeg.get_ffmpeg_exe()


def _parse_ffprobe_size(out: str) -> tuple[int, int] | None:
    """First ``width,height`` record of ffprobe csv output, or None when there is none."""
    for line in out.splitlines():
        fields = line.strip().split(",")
        if len(fields) == 2 and all(field.isdigit() for field in fields):
            return int(fields[0]), int(fields[1])
    return None


def _video_size(path: Path) -> tuple[int, int]:
    probe = shutil.which("ffprobe")
    if probe:
        out = subprocess.run(
            [
                probe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "csv=p=0",
                str(path),
            ],
            check=False,
            capture_output=True,
            text=True,
        ).stdout
        size = _parse_ffprobe_size(out)
        if size is not None:
            return size
        # '-select_streams v:0' on a file with no video stream exits 0 with empty stdout, so
        # fall through to the ffmpeg probe, whose error names the offending file.
    err = subprocess.run([_ffmpeg(), "-i", str(path)], capture_output=True, text=True).stderr
    match = _FFMPEG_SIZE_RE.search(err)
    if match is None:
        raise RuntimeError(f"Could not read video size from {path}")
    return int(match.group(1)), int(match.group(2))


def find_sample_dirs(root: Path) -> list[Path]:
    if any(root.glob("vision_view*.mp4")):
        return [root]
    matches = sorted({p.parent for p in root.rglob("vision_view*.mp4")})
    if not matches:
        raise ValueError(f"No vision_view*.mp4 files under {root}")
    return matches


def collect_views(sample_dir: Path, kind: str = "vision") -> list[tuple[int, str, Path]]:
    views = []
    for path in sample_dir.glob(f"{kind}_view*.mp4"):
        match = _VIEW_RE.match(path.name)
        if match is None or match["kind"] != kind or path.stem.endswith("_preview"):
            continue
        views.append((int(match["idx"]), match["camera"], path))
    return sorted(views)


def _run_ffmpeg(cmd: list[str], *, inputs: list[str], filters: list[str]) -> None:
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            "ffmpeg collage failed\n"
            f"inputs: {inputs}\n"
            f"filter_complex: {';'.join(filters)}\n"
            f"{(error.stderr or error.stdout or '').strip()}"
        ) from error


def gif_target_width(src_width: int, encoded_bytes: int, max_bytes: int) -> int:
    """Next even width after an oversize GIF encode. Area is assumed to track file size."""
    even_src = max(2, src_width - src_width % 2)
    if encoded_bytes <= max_bytes:
        return even_src
    scale = math.sqrt(max_bytes * 0.92 / max(encoded_bytes, 1))
    width = max(_GIF_MIN_WIDTH, int(even_src * scale) // 2 * 2)
    if width >= even_src:
        width = max(_GIF_MIN_WIDTH, even_src - 2)
    return width


def _encode_palette_gif(ffmpeg: str, mp4: Path, gif: Path, *, width: int, fps: float) -> None:
    vf = (
        f"fps={fps:.3f},scale={width}:-2:flags=lanczos,split[s0][s1];"
        "[s0]palettegen=max_colors=256:stats_mode=full[p];"
        "[s1][p]paletteuse=dither=sierra2_4a"
    )
    cmd = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(mp4),
        "-filter_complex",
        vf,
        "-loop",
        "0",
        str(gif),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"ffmpeg gif encode failed for {mp4}\n{(error.stderr or error.stdout or '').strip()}"
        ) from error


def write_collage_gif(
    mp4: Path,
    gif: Path | None = None,
    *,
    max_bytes: int = MAX_GIF_BYTES,
) -> Path:
    """Write a palette GIF next to ``mp4``, keeping the highest quality that fits ``max_bytes``."""
    gif = gif or mp4.with_suffix(".gif")
    ffmpeg = _ffmpeg()
    src_w, _src_h = _video_size(mp4)
    width = max(2, src_w - src_w % 2)
    work = gif.with_name(f".{gif.stem}.enc.gif")
    chosen_fps = _GIF_FPS_CANDIDATES[-1]
    size = 0
    try:
        for fps in _GIF_FPS_CANDIDATES:
            _encode_palette_gif(ffmpeg, mp4, work, width=width, fps=fps)
            size = work.stat().st_size
            if size <= max_bytes:
                work.replace(gif)
                mb = size / (1024 * 1024)
                print(f"Collage gif ({mb:.1f}MB, {width}w @ {fps:.0f}fps): {gif}")
                return gif
        for _ in range(8):
            next_width = gif_target_width(width, size, max_bytes)
            if next_width >= width:
                break
            width = next_width
            _encode_palette_gif(ffmpeg, mp4, work, width=width, fps=chosen_fps)
            size = work.stat().st_size
            if size <= max_bytes:
                work.replace(gif)
                mb = size / (1024 * 1024)
                print(f"Collage gif ({mb:.1f}MB, {width}w @ {chosen_fps:.0f}fps): {gif}")
                return gif
        raise RuntimeError(
            f"Could not fit collage GIF under {max_bytes} bytes for {mp4} (last {size} bytes at {width}w)"
        )
    finally:
        if work.exists():
            work.unlink()


def _collage_one(
    sample_dir: Path,
    output: Path,
    tile_width: int,
    crf: int,
    labels: bool,
    kind: str = "vision",
) -> Path:
    views = collect_views(sample_dir, kind)
    if not views:
        raise ValueError(f"No {kind}_view*.mp4 files in {sample_dir}")

    src_w, src_h = _video_size(views[0][2])
    w, h = tile_size(tile_width, src_w, src_h)
    cameras = [camera for _, camera, _ in views]
    rows, cols, cells = choose_vehicle_cells(cameras)
    fonts = glob.glob("/usr/share/fonts/**/DejaVuSans.ttf", recursive=True)
    font = fonts[0] if labels and fonts else None
    font_size = max(10, w // 24)

    ffmpeg = _ffmpeg()
    inputs: list[str] = []
    filters: list[str] = []
    fit = tile_filter(w, h)
    tags: list[str] = []
    layout: list[str] = []

    for i, (_, camera, path) in enumerate(views):
        inputs.append(str(path))
        tag = f"[t{i}]"
        chain = f"[{i}:v]{fit}"
        if font:
            text = camera.removeprefix("camera_").replace("'", r"\'").replace(":", r"\:")
            chain += (
                f",drawtext=fontfile={font}:text='{text}':x=4:y=4:fontsize={font_size}"
                ":fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=3"
            )
        filters.append(chain + tag)
        tags.append(tag)
        row, col = cells[camera]
        x, y = cell_xy(row, col, w, h)
        layout.append(f"{x}_{y}")

    if len(tags) == 1:
        filters.append(f"{tags[0]}copy[out]")
    else:
        filters.append(f"{''.join(tags)}xstack=inputs={len(tags)}:layout={'|'.join(layout)}:fill=black[out]")

    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg, "-y", "-loglevel", "error"]
    for path in inputs:
        cmd += ["-i", path]
    cmd += [
        "-filter_complex",
        ";".join(filters),
        "-map",
        "[out]",
        "-c:v",
        "libx264",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    _run_ffmpeg(cmd, inputs=inputs, filters=filters)
    print(f"Collage ({kind}, {len(views)} cameras, {cols}x{rows}): {output}")
    return output


def make_collage(
    directory: Path,
    output: Path | None = None,
    tile_width: int = 320,
    crf: int = 20,
    labels: bool = True,
    visualize_control_input: bool = False,
    write_gif: bool = False,
    max_gif_bytes: int = MAX_GIF_BYTES,
) -> list[Path]:
    samples = find_sample_dirs(directory)
    if output is not None and len(samples) > 1:
        raise ValueError("--output requires a single sample directory")
    paths: list[Path] = []
    for sample in samples:
        out = output or sample / "collage.mp4"
        paths.append(_collage_one(sample, out, tile_width, crf, labels, kind="vision"))
        if write_gif:
            paths.append(write_collage_gif(out, max_bytes=max_gif_bytes))
        if visualize_control_input:
            control_out = out.with_name(f"{out.stem}_control_input.mp4")
            paths.append(_collage_one(sample, control_out, tile_width, crf, labels, kind="control"))
            if write_gif:
                paths.append(write_collage_gif(control_out, max_bytes=max_gif_bytes))
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--tile-width", type=int, default=320)
    parser.add_argument("--crf", type=int, default=20)
    parser.add_argument("--no-labels", action="store_true")
    parser.add_argument(
        "--visualize_control_input",
        action="store_true",
        help="Also collage control_view*.mp4 into <output_stem>_control_input.mp4",
    )
    parser.add_argument(
        "--save-extra-gif",
        action="store_true",
        help="Also write a palette GIF next to each MP4, capped at --max-gif-mb (default 10).",
    )
    parser.add_argument(
        "--max-gif-mb",
        type=float,
        default=MAX_GIF_BYTES / (1024 * 1024),
        help="Max GIF size in megabytes (default 10).",
    )
    args = parser.parse_args(argv)
    make_collage(
        args.directory,
        args.output,
        args.tile_width,
        args.crf,
        labels=not args.no_labels,
        visualize_control_input=args.visualize_control_input,
        write_gif=args.save_extra_gif,
        max_gif_bytes=max(1, int(args.max_gif_mb * 1024 * 1024)),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
