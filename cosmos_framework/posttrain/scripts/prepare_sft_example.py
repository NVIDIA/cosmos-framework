# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Generate tiny deterministic text/image/video conversations outside the source tree."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

_COLORS: tuple[tuple[str, tuple[int, int, int]], ...] = (
    ("red", (230, 30, 30)),
    ("green", (30, 200, 30)),
    ("blue", (30, 30, 230)),
    ("yellow", (220, 220, 30)),
)


def _write_video(path: Path, color: tuple[int, int, int], *, right: bool) -> None:
    import av
    from PIL import Image, ImageDraw

    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=4)
        stream.width = stream.height = 128
        stream.pix_fmt = "yuv420p"
        for index in range(8):
            frame = Image.new("RGB", (128, 128), "black")
            x = 8 + 12 * (index if right else 7 - index)
            ImageDraw.Draw(frame).rectangle((x, 48, x + 20, 68), fill=color)
            for packet in stream.encode(av.VideoFrame.from_image(frame)):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def prepare_example(output: Path, train_samples: int = 32, validation_samples: int = 8) -> None:
    """Write a new dataset directory; refuse replacement of existing files."""
    from PIL import Image

    if train_samples < 1 or validation_samples < 1:
        raise ValueError("both splits need at least one sample")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"refusing to overwrite non-empty example directory {output}")
    output.mkdir(parents=True, exist_ok=True)
    media = output / "media"
    media.mkdir()
    for name, rgb in _COLORS:
        Image.new("RGB", (128, 128), rgb).save(media / f"{name}.png")
        for right in (True, False):
            _write_video(media / f"{name}_{'right' if right else 'left'}.mp4", rgb, right=right)

    for split, count in (("train", train_samples), ("validation", validation_samples)):
        with (output / f"{split}.jsonl").open("w", encoding="utf-8") as stream:
            for index in range(count):
                color, _ = _COLORS[(index // 4) % len(_COLORS)]
                direction = "right" if index % 2 else "left"
                mode = index % 4
                content: list[dict[str, Any]] = []
                if mode in (1, 3):
                    content.append({"type": "image", "image": f"media/{color}.png"})
                if mode in (2, 3):
                    content.append({"type": "video", "video": f"media/{color}_{direction}.mp4"})
                question = f"Example {split}-{index}. "
                question += f"Remember the color {color}. What is the color?" if mode == 0 else "What color is shown?"
                content.append({"type": "text", "text": question})
                followup = (
                    "Which direction does the square move?" if mode in (2, 3) else "Repeat that color in uppercase."
                )
                answer = direction if mode in (2, 3) else color.upper()
                row = {
                    "id": f"{split}-{index}",
                    "messages": [
                        {"role": "system", "content": "Answer briefly using only the requested word."},
                        {"role": "user", "content": content},
                        {"role": "assistant", "content": color},
                        {"role": "user", "content": followup},
                        {"role": "assistant", "content": answer},
                    ],
                }
                stream.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-samples", type=int, default=32)
    parser.add_argument("--validation-samples", type=int, default=8)
    args = parser.parse_args()
    prepare_example(args.output, args.train_samples, args.validation_samples)
    print(f"Wrote {args.output}/{{train,validation}}.jsonl and local media")


if __name__ == "__main__":
    main()
