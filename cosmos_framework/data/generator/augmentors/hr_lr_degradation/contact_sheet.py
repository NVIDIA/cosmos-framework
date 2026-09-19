# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""E4: visual sanity sheet. For each input clip or image and each profile, render first / middle / last
frames of HR and LR plus a 4x zoom crop into one HTML page, with the degradation record alongside.

Inputs are video files (decoded with PyAV) or images (PNG/JPEG). Example::

    PYTHONPATH=. python -m cosmos_framework.data.generator.augmentors.hr_lr_degradation.contact_sheet \
        --inputs clips/*.mp4 --profiles p0_clean_bicubic p1_first_order p1_second_order p3_video_codec \
        --max-frames 33 --out sheet/index.html
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import sys
from pathlib import Path

import av
import numpy as np
import torch
from PIL import Image

from cosmos_framework.data.generator.augmentors.hr_lr_degradation.degrade import degrade_hr_to_lr

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def load_media(path: Path, max_frames: int, max_side: int | None) -> torch.Tensor:  # returns [3,T,H,W] uint8
    if path.suffix.lower() in _IMAGE_SUFFIXES:
        img = Image.open(path).convert("RGB")
        if max_side and max(img.size) > max_side:
            scale = max_side / max(img.size)
            img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
        arr = torch.from_numpy(np.array(img)).permute(2, 0, 1)  # [3,H,W]
        return arr.unsqueeze(1)  # [3,1,H,W]
    frames: list[np.ndarray] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        stream.thread_count = 4  # bounded: decoders otherwise size their pool to the whole host
        for frame in container.decode(stream):
            frames.append(frame.reformat(format="rgb24", threads=4).to_ndarray())  # [H,W,3], bounded swscale pool
            if len(frames) >= max_frames:
                break
    if not frames:
        raise ValueError(f"No frames decoded from {path}")
    clip = torch.from_numpy(np.stack(frames)).permute(3, 0, 1, 2)  # [3,T,H,W]
    if max_side and max(clip.shape[-2:]) > max_side:
        scale = max_side / max(clip.shape[-2:])
        size = (round(clip.shape[-2] * scale), round(clip.shape[-1] * scale))
        clip = torch.nn.functional.interpolate(
            clip.permute(1, 0, 2, 3).float(), size=size, mode="bicubic", antialias=True, align_corners=False
        )  # [T,3,h,w]
        clip = clip.clamp(0, 255).round().to(torch.uint8).permute(1, 0, 2, 3)  # [3,T,h,w]
    return clip


def _to_png_b64(frame: torch.Tensor, scale: float = 1.0) -> str:  # frame: [3,H,W] uint8
    img = Image.fromarray(frame.permute(1, 2, 0).numpy())
    if scale != 1.0:
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _crop(frame: torch.Tensor, frac: float, cy: float, cx: float) -> torch.Tensor:  # frame: [3,H,W]
    h, w = frame.shape[-2:]
    ch, cw = max(8, int(h * frac)), max(8, int(w * frac))
    top = min(max(0, int(cy * h - ch / 2)), h - ch)
    left = min(max(0, int(cx * w - cw / 2)), w - cw)
    return frame[:, top : top + ch, left : left + cw]


def render_row(name: str, hr: torch.Tensor, profile: str, seed: int, scale: float, display_width: int) -> str:
    result = degrade_hr_to_lr(hr, profile, scale=scale, seed=seed)
    lr = result.lr  # [3,T,h,w]
    t = hr.shape[1]
    idxs = sorted({0, t // 2, t - 1})
    cells: list[str] = []
    for i in idxs:
        hr_f, lr_f = hr[:, i], lr[:, i]  # [3,H,W], [3,h,w]
        disp_hr = display_width / hr_f.shape[-1]
        disp_lr = display_width / lr_f.shape[-1]  # LR is shown upscaled to the same width for comparison
        crop_hr = _crop(hr_f, 0.15, 0.5, 0.5)
        crop_lr = _crop(lr_f, 0.15, 0.5, 0.5)
        zoom_hr = display_width / 2 / crop_hr.shape[-1]
        zoom_lr = display_width / 2 / crop_lr.shape[-1]
        cells.append(
            f"<td><div class='lbl'>frame {i}: HR {hr_f.shape[-2]}x{hr_f.shape[-1]}</div>"
            f"<img src='data:image/png;base64,{_to_png_b64(hr_f, disp_hr)}'>"
            f"<div class='lbl'>LR {lr_f.shape[-2]}x{lr_f.shape[-1]} (shown at HR width)</div>"
            f"<img src='data:image/png;base64,{_to_png_b64(lr_f, disp_lr)}'>"
            f"<div class='lbl'>centre crop, HR | LR</div>"
            f"<img src='data:image/png;base64,{_to_png_b64(crop_hr, zoom_hr)}'>"
            f"<img src='data:image/png;base64,{_to_png_b64(crop_lr, zoom_lr)}'></td>"
        )
    record = html.escape(json.dumps(result.record, indent=1))
    return (
        f"<tr><th>{html.escape(name)}<br>{html.escape(profile)}<br>seed {seed}</th>"
        + "".join(cells)
        + f"<td><pre>{record}</pre></td></tr>"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs", nargs="+", required=True, help="video or image files")
    parser.add_argument("--profiles", nargs="+", default=["p0_clean_bicubic", "p1_first_order", "p1_second_order"])
    parser.add_argument("--scale", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=33)
    parser.add_argument("--max-side", type=int, default=None, help="downscale inputs whose longest side exceeds this")
    parser.add_argument("--display-width", type=int, default=480)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args(argv)

    rows: list[str] = []
    for path_str in args.inputs:
        path = Path(path_str)
        hr = load_media(path, args.max_frames, args.max_side)  # [3,T,H,W]
        for k, profile in enumerate(args.profiles):
            rows.append(render_row(path.name, hr, profile, args.seed + k, args.scale, args.display_width))
            print(f"rendered {path.name} / {profile}", flush=True)
    page = (
        "<html><head><meta charset='utf-8'><style>body{font-family:sans-serif;background:#111;color:#ddd}"
        "table{border-collapse:collapse}td,th{border:1px solid #444;vertical-align:top;padding:6px}"
        "img{display:block;margin:2px 0;image-rendering:pixelated}.lbl{font-size:11px;color:#9ad}"
        "pre{font-size:10px;max-width:320px;white-space:pre-wrap}</style></head><body>"
        f"<h2>HR-to-LR degradation contact sheet (scale {args.scale})</h2><table>{''.join(rows)}</table></body></html>"
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
