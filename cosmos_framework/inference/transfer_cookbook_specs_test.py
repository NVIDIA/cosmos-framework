# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""The Transfer cookbook specs must describe the videos they point at."""

import json
from pathlib import Path

import pytest

pytestmark = [pytest.mark.L0, pytest.mark.CPU]

_COOKBOOK_SPECS = Path(__file__).resolve().parents[2] / ".agents/skills/cookbooks/cosmos3/generator/transfer/specs"


def _is_lfs_pointer(path: Path) -> bool:
    return path.stat().st_size < 1024 and path.read_bytes().startswith(b"version https://git-lfs")


@pytest.mark.parametrize("spec", sorted(_COOKBOOK_SPECS.glob("*.json")), ids=lambda path: path.stem)
def test_cookbook_spec_fps_and_num_frames_match_its_video(spec: Path) -> None:
    """Runs where the Git LFS videos are checked out; CI's unit-test checkout does not fetch them."""
    import av

    raw = json.loads(spec.read_text())
    sources = [raw.get("vision_path")] + [hint.get("control_path") for hint in raw.values() if isinstance(hint, dict)]
    source = next(path for path in sources if path)
    if source.startswith("https://"):
        pytest.skip("spec downloads its input")
    path = (spec.parent / source).resolve()
    if _is_lfs_pointer(path):
        pytest.skip("Git LFS assets are not checked out")
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        assert round(float(stream.average_rate)) == raw["fps"]
        # Compared on the 4n+1 latent grid, as generation does.
        assert ((stream.frames - 1 + 3) // 4) * 4 + 1 == raw["num_frames"]
