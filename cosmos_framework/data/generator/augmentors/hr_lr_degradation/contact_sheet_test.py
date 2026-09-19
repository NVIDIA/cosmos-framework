# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from pathlib import Path

import av
import numpy as np
import pytest
from PIL import Image

from cosmos_framework.data.generator.augmentors.hr_lr_degradation import contact_sheet

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def _write_clip(path: Path, frames: int = 9, h: int = 96, w: int = 128) -> None:
    rng = np.random.default_rng(0)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=24)
        stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
        stream.options = {"crf": "18", "threads": "1"}
        for i in range(frames):
            frame = np.full((h, w, 3), 40 + 20 * i, dtype=np.uint8)  # [H,W,3]
            frame[:, : w // 2] = rng.integers(0, 255, (h, w // 2, 3), dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def test_contact_sheet_renders_video_and_image_inputs(tmp_path: Path) -> None:
    clip = tmp_path / "clip.mp4"
    _write_clip(clip)
    image = tmp_path / "image.png"
    Image.fromarray(np.random.default_rng(1).integers(0, 255, (80, 120, 3), dtype=np.uint8)).save(image)
    out = tmp_path / "sheet" / "index.html"

    loaded = contact_sheet.load_media(clip, max_frames=5, max_side=None)  # [3,5,96,128]
    assert loaded.shape == (3, 5, 96, 128)
    assert contact_sheet.load_media(image, max_frames=5, max_side=60).shape == (3, 1, 40, 60)

    rc = contact_sheet.main(
        [
            "--inputs",
            str(clip),
            str(image),
            "--profiles",
            "p0_clean_bicubic",
            "p3_video_codec",
            "--max-frames",
            "5",
            "--out",
            str(out),
        ]
    )
    assert rc == 0 and out.exists()
    page = out.read_text()
    assert page.count("<tr>") == 4  # 2 inputs x 2 profiles
    assert "p3_video_codec" in page and "data:image/png;base64," in page and "profile_name" in page
