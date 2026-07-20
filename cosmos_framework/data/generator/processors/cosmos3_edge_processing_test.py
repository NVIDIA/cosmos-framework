# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Feature-parity gate (G1) for the framework-native Cosmos3-Edge processor port.

The embedded goldens pin exact token ids AND pixel-tensor sha256 hashes; the
fixtures must byte-match ``outputs/audit/processor_audit.py``. A pixel sha256
mismatch with matching shapes means the patch layout regressed from the
original convention (raster patch-row order, ``(py, px, c)`` within-row) —
e.g. code "aligned" with transformers-main's native processor, which corrupts
vision features with these checkpoint weights; see
``outputs/audit/cosmos3_edge_native_vision_layout_bug.md``.
"""

import hashlib
import json
import os

import numpy as np
import pytest
import torch
from PIL import Image

from cosmos_framework.data.generator.processors.cosmos3_edge_processing import (
    build_cosmos3_edge_processor,
    is_cosmos3_edge_native_snapshot,
)

# Renewed (native-metadata, no remote code) snapshot of nvidia/Cosmos3-Edge.
# Point COSMOS3_EDGE_SNAPSHOT_DIR at a local snapshot dir to run these checks;
# unset (the default) auto-skips them.
_SNAPSHOT_DIR = os.environ.get("COSMOS3_EDGE_SNAPSHOT_DIR", "")

requires_snapshot = pytest.mark.skipif(
    not os.path.isdir(_SNAPSHOT_DIR),
    reason=f"renewed nvidia/Cosmos3-Edge snapshot not available at {_SNAPSHOT_DIR}",
)

# Captured from the old remote-code processor (rev 28a0b8e, transformers 4.57.6);
# embedded so the test does not depend on the outputs/ tree.
_GOLDEN = {
    "text_only": {
        "num_tokens": 35,
        "ids_sha256": "5916e7a832968ac9",
    },
    "image": {
        "num_tokens": 329,
        "ids_sha256": "9cc248c1c2260d79",
        "n_image_tokens": 300,
        "pixel_values": {"shape": [1200, 768], "sha256": "074c173f37463c24"},
        "image_grid_thw": [[1, 30, 40]],
    },
    "video": {
        "num_tokens": 730,
        "ids_sha256": "3c9bedc1730ecf67",
        "n_video_tokens": 640,
        "pixel_values_videos": {"shape": [2560, 768], "sha256": "8cf76eb9af3d0b5a"},
        "video_grid_thw": [[8, 16, 20]],
    },
}


def _ids_sha256(ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16]


def _tensor_sha256(t: torch.Tensor) -> str:
    a = np.asarray(t.float().cpu(), dtype=np.float32)
    return hashlib.sha256(a.tobytes()).hexdigest()[:16]


@pytest.fixture(scope="module")
def processor():
    return build_cosmos3_edge_processor(_SNAPSHOT_DIR)


def _apply(processor, messages, **kwargs):
    return processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_tensors="pt",
        **kwargs,
    )


@requires_snapshot
def test_detection_rule_accepts_native_snapshot() -> None:
    assert is_cosmos3_edge_native_snapshot(_SNAPSHOT_DIR)


def test_detection_rule_rejects_non_edge_dirs(tmp_path) -> None:
    # Empty dir, non-dir, and an old-style (remote-code) config all miss the rule.
    assert not is_cosmos3_edge_native_snapshot(str(tmp_path))
    assert not is_cosmos3_edge_native_snapshot(str(tmp_path / "missing"))
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "nemotron_siglip2"}))
    assert not is_cosmos3_edge_native_snapshot(str(tmp_path))


@requires_snapshot
def test_text_only_matches_golden(processor) -> None:
    inputs = _apply(
        processor,
        [
            {"role": "user", "content": [{"type": "text", "text": "Describe the physics of a bouncing ball."}]},
            {"role": "assistant", "content": [{"type": "text", "text": "It decelerates under gravity."}]},
        ],
    )
    ids = inputs["input_ids"][0].tolist()
    assert len(ids) == _GOLDEN["text_only"]["num_tokens"]
    assert _ids_sha256(ids) == _GOLDEN["text_only"]["ids_sha256"]
    assert inputs["attention_mask"][0].tolist() == [1] * len(ids)


@requires_snapshot
def test_image_matches_golden(processor) -> None:
    rng = np.random.RandomState(42)
    image = Image.fromarray(rng.randint(0, 255, (480, 640, 3), dtype=np.uint8))
    inputs = _apply(
        processor,
        [
            {
                "role": "user",
                "content": [{"type": "image", "image": image}, {"type": "text", "text": "What is shown?"}],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "A test pattern."}]},
        ],
    )
    golden = _GOLDEN["image"]
    ids = inputs["input_ids"][0].tolist()
    assert len(ids) == golden["num_tokens"]
    assert _ids_sha256(ids) == golden["ids_sha256"]
    image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    assert ids.count(image_token_id) == golden["n_image_tokens"]

    assert list(inputs["pixel_values"].shape) == golden["pixel_values"]["shape"]
    # Pins the exact float32 bytes, i.e. the OLD (py, px, c)/raster patch layout.
    assert _tensor_sha256(inputs["pixel_values"]) == golden["pixel_values"]["sha256"]
    assert inputs["image_grid_thw"].tolist() == golden["image_grid_thw"]


@requires_snapshot
def test_video_matches_golden(processor) -> None:
    rng = np.random.RandomState(7)
    frames = [Image.fromarray(rng.randint(0, 255, (256, 320, 3), dtype=np.uint8)) for _ in range(8)]
    metadata = {"fps": 2.0, "total_num_frames": len(frames), "frames_indices": list(range(len(frames)))}
    inputs = _apply(
        processor,
        [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": frames},
                    {"type": "text", "text": "Is this physically plausible?"},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "Yes."}]},
        ],
        videos_kwargs={"do_sample_frames": False, "video_metadata": metadata},
    )
    golden = _GOLDEN["video"]
    ids = inputs["input_ids"][0].tolist()
    assert len(ids) == golden["num_tokens"]
    assert _ids_sha256(ids) == golden["ids_sha256"]
    video_token_id = processor.tokenizer.convert_tokens_to_ids(processor.video_token)
    assert ids.count(video_token_id) == golden["n_video_tokens"]

    assert list(inputs["pixel_values_videos"].shape) == golden["pixel_values_videos"]["shape"]
    # Pins the exact float32 bytes, i.e. the OLD (py, px, c)/raster patch layout.
    assert _tensor_sha256(inputs["pixel_values_videos"]) == golden["pixel_values_videos"]["sha256"]
    assert inputs["video_grid_thw"].tolist() == golden["video_grid_thw"]

    # Per-frame "<{t:.1f} seconds>" timestamp interleaving (fps=2.0, indices 0..7).
    decoded = processor.tokenizer.decode(ids)
    assert "<0.0 seconds>" in decoded
    assert "<0.5 seconds>" in decoded


@requires_snapshot
def test_wrapper_interface_surface(processor) -> None:
    """The attributes the framework wrappers consume off the raw processor."""
    assert processor.image_token == "<|image_pad|>"
    assert processor.video_token == "<|video_pad|>"
    assert processor.image_processor.size == {"shortest_edge": 65536, "longest_edge": 16777216}
    assert processor.image_processor.patch_size == 16
    assert processor.image_processor.merge_size == 2
    assert processor.video_processor.patch_size == 16
    assert processor.video_processor.temporal_patch_size == 1
    assert processor.video_processor.merge_size == 2
    assert processor.tokenizer.eos_token_id is not None


@requires_snapshot
def test_build_processor_routes_local_edge_dir_to_nemotron_wrapper() -> None:
    from cosmos_framework.data.generator.processors import Nemotron3DenseVLProcessor, build_processor

    wrapper = build_processor(_SNAPSHOT_DIR)
    assert isinstance(wrapper, Nemotron3DenseVLProcessor)
    # Dataloader helper attributes resolved through the ported sub-processors.
    assert wrapper.patch_size == 16
    assert wrapper.temporal_patch_size == 1
    assert wrapper.merge_size == 2
    assert wrapper.image_token_id == 19
    assert wrapper.video_token_id == 18
