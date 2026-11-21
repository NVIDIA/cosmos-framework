# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Check that the Diffusers policy guide builds the DROID concat_view the way training does."""

import __future__

import ast
import re
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from cosmos_framework.data.generator.action.datasets.droid_lerobot_dataset import DROIDLeRobotDataset

GUIDE = Path(__file__).resolve().parents[1] / ".agents/skills/cosmos3-inference-action-policy/references/diffusers.md"
# The guide's camera names, in the order DROIDLeRobotDataset._compose_multi_view tiles them.
CAMERAS = {"wrist": "wrist", "exterior_1": "left", "exterior_2": "right"}
# Imports the helpers under test do not use: the pipeline itself and the notebook's plotting and display.
SKIPPED_IMPORTS = {"diffusers", "IPython", "matplotlib", "mpl_toolkits"}


def _imported_modules(node: ast.Import | ast.ImportFrom) -> set[str]:
    if isinstance(node, ast.ImportFrom):
        return {(node.module or "").split(".")[0]}
    return {alias.name.split(".")[0] for alias in node.names}


def _guide_namespace(frames: dict[str, Image.Image], output_dir: Path) -> dict:
    """Run the guide's imports and define its helpers and camera table, then stub out video loading and output paths."""
    blocks = re.findall(r"```python\n(.*?)```", GUIDE.read_text(), flags=re.DOTALL)
    nodes = [
        node
        for block in blocks
        for node in ast.parse(block).body
        if isinstance(node, ast.FunctionDef)
        or (isinstance(node, (ast.Import, ast.ImportFrom)) and not _imported_modules(node) & SKIPPED_IMPORTS)
        or (
            isinstance(node, ast.Assign)
            and any(getattr(target, "id", None) in {"CAMERA_VIDEOS", "CONCAT_SIZE"} for target in node.targets)
        )
    ]
    namespace: dict = {}
    # Only imports and definitions run; postponed annotations keep e.g. ``-> Cosmos3OmniPipeline`` from needing diffusers.
    code = compile(ast.Module(body=nodes, type_ignores=[]), str(GUIDE), "exec", __future__.annotations.compiler_flag)
    exec(code, namespace)

    frames_by_path = {video: frames[name] for name, video in namespace["CAMERA_VIDEOS"].items()}
    namespace["asset_path"] = Path
    namespace["case_output_dir"] = lambda case: output_dir
    namespace["load_video"] = lambda path: [frames_by_path[path]]
    return namespace


def _training_composite(frames: dict[str, Image.Image]) -> np.ndarray:
    """The released DROID recipe's camera packing, without augmentation, as an RGB uint8 image."""
    dataset = object.__new__(DROIDLeRobotDataset)
    dataset._image_features = {role: role for role in CAMERAS.values()}
    dataset._use_image_augmentation = False
    dataset._jitter_after_compose = False
    dataset._image_augmentor = None
    dataset._color_augmentor = None
    sample = {
        role: torch.from_numpy(np.asarray(frames[name])).permute(2, 0, 1).unsqueeze(0).float()  # [T=1,C,H,W]
        for name, role in CAMERAS.items()
    }
    composite = dataset._compose_multi_view(sample)[0]  # [C,3H/2,W]
    return composite.permute(1, 2, 0).round().clamp(0, 255).to(torch.uint8).numpy()


@pytest.mark.L0
def test_build_concat_frame_matches_training_composite(tmp_path: Path) -> None:
    """Pixel-for-pixel: full wrist view on top, both exteriors at half size below, nothing cropped."""
    generator = np.random.default_rng(0)
    frames = {name: Image.fromarray(generator.integers(0, 256, size=(360, 640, 3), dtype=np.uint8)) for name in CAMERAS}

    namespace = _guide_namespace(frames, tmp_path)
    frame = np.asarray(Image.open(namespace["build_concat_frame"]("droid_policy")).convert("RGB"))

    expected = _training_composite(frames)
    assert frame.shape == expected.shape == (540, 640, 3)
    assert frame.shape[1::-1] == namespace["CONCAT_SIZE"]
    np.testing.assert_array_equal(frame[:360], np.asarray(frames["wrist"]))
    np.testing.assert_array_equal(frame, expected)
