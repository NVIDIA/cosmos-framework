# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Native Edge timestamp rendering contracts using a real, local checkpoint processor.

These fixtures test preprocessing, not learned reasoning ability. No weights or
processor files are downloaded. Set COSMOS3_EDGE_SNAPSHOT_DIR to pinned metadata.
"""

import os
from copy import deepcopy
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image

from cosmos_framework.utils.generator.source_video_timing import (
    SOURCE_VIDEO_TIMING_KEY,
    SourceVideoTiming,
    build_source_video_timing,
)

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


@pytest.fixture(scope="module")
def native_processor() -> Any:
    path = os.environ.get("COSMOS3_EDGE_SNAPSHOT_DIR", "")
    if not os.path.isdir(path):
        pytest.skip("Set COSMOS3_EDGE_SNAPSHOT_DIR to real, pinned local Edge processor metadata")
    from cosmos_framework.data.generator.processors.cosmos3_edge_processing import build_cosmos3_edge_processor

    return build_cosmos3_edge_processor(path)


def _record(offset: float = 10.0) -> SourceVideoTiming:
    pts = torch.tensor([offset, offset + 0.2, offset + 0.2, offset + 0.9], dtype=torch.float64)  # [4]
    durations = torch.tensor([0.1, 0.1, 0.1, 0.1], dtype=torch.float64)  # [4]
    return build_source_video_timing([0, 2, 2, 7], pts, durations, "a" * 64)


def _frames() -> list[Image.Image]:
    rng = np.random.RandomState(71)
    return [Image.fromarray(rng.randint(0, 255, (64, 64, 3), dtype=np.uint8)) for _ in range(4)]


def _messages(timing: SourceVideoTiming | None) -> list[dict[str, Any]]:
    item: dict[str, Any] = {"type": "video", "video": _frames()}
    if timing is not None:
        item[SOURCE_VIDEO_TIMING_KEY] = timing
    return [
        {"role": "user", "content": [item, {"type": "text", "text": "Describe the motion."}]},
        {"role": "assistant", "content": [{"type": "text", "text": "The object moves."}]},
    ]


def _apply(processor: Any, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
    metadata = {"fps": 4.0, "total_num_frames": 4, "frames_indices": [0, 1, 2, 3]}
    return processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_tensors="pt",
        videos_kwargs={"do_sample_frames": False, "video_metadata": metadata},
        **kwargs,
    )


def test_source_pts_changes_only_timestamp_text_not_pixel_layout_or_rng(native_processor: Any) -> None:
    before = torch.get_rng_state()  # [S]
    legacy = _apply(native_processor, _messages(None))
    legacy_rng = torch.get_rng_state()  # [S]
    exact = _apply(native_processor, _messages(_record()))
    exact_rng = torch.get_rng_state()  # [S]
    assert torch.equal(before, legacy_rng) and torch.equal(before, exact_rng)
    assert set(legacy) == set(exact)
    for key in ("pixel_values_videos", "video_grid_thw"):
        assert torch.equal(legacy[key], exact[key])
    decoded = native_processor.tokenizer.decode(exact["input_ids"][0].tolist())
    assert decoded.count("<0.2 seconds>") == 2
    assert "<0.9 seconds>" in decoded and "<10.0 seconds>" not in decoded
    legacy_decoded = native_processor.tokenizer.decode(legacy["input_ids"][0].tolist())
    assert "<0.5 seconds>" in legacy_decoded
    assert not torch.equal(legacy["input_ids"], exact["input_ids"])
    # Absence of the opt-in record must retain the existing exact outputs.
    repeated = _apply(native_processor, _messages(None))
    for key in legacy:
        assert torch.equal(legacy[key], repeated[key])


def test_equal_legacy_timestamps_have_exact_token_and_pixel_parity(native_processor: Any) -> None:
    record = _record(0.0)
    record["pts_seconds"] = [0.0, 0.25, 0.5, 0.75]
    record["frame_indices"] = [0, 1, 2, 3]
    legacy = _apply(native_processor, _messages(None))
    exact = _apply(native_processor, _messages(record))
    for key in legacy:
        assert torch.equal(legacy[key], exact[key])


@pytest.mark.parametrize("field,value", [("pts_seconds", [10.0]), ("origin_pts_seconds", 0.0)])
def test_malformed_record_fails_before_processor_merge(native_processor: Any, field: str, value: object) -> None:
    record = dict(_record())
    record[field] = value
    messages = _messages(None)
    messages[0]["content"][0][SOURCE_VIDEO_TIMING_KEY] = record
    with pytest.raises(ValueError, match="source_pts"):
        _apply(native_processor, messages)


def test_video_encounter_order_preserves_independent_origins_with_images(native_processor: Any) -> None:
    messages = _messages(_record(10.0))
    second = _messages(_record(100.0))[0]
    second["content"].insert(0, {"type": "image", "image": _frames()[0]})
    messages.extend([second, {"role": "assistant", "content": [{"type": "text", "text": "Same motion."}]}])
    metadata = {"fps": 4.0, "total_num_frames": 4, "frames_indices": [0, 1, 2, 3]}
    exact = native_processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=False,
        videos_kwargs={"do_sample_frames": False, "video_metadata": [metadata, deepcopy(metadata)]},
    )
    assert exact["video_grid_thw"][:, 0].tolist() == [4, 4]
    decoded = native_processor.tokenizer.decode(exact["input_ids"][0].tolist())
    assert decoded.count("<0.9 seconds>") == 2
    assert "<100.0 seconds>" not in decoded


def test_partial_metadata_or_audio_is_rejected(native_processor: Any) -> None:
    messages = _messages(_record())
    messages[0]["content"].append({"type": "video", "video": _frames()})
    with pytest.raises(ValueError, match="every video"):
        _apply(native_processor, messages)
    messages = _messages(_record())
    messages[0]["content"].append({"type": "audio", "audio": "unsupported"})
    with pytest.raises(ValueError, match="audio"):
        _apply(native_processor, messages)


def test_direct_processor_rejects_count_mismatch_and_resampling(native_processor: Any) -> None:
    kwargs = {"videos": [_frames()], "text": [native_processor.video_token], "source_video_timing": [_record()]}
    with pytest.raises(ValueError, match="do_sample_frames=False"):
        native_processor(**kwargs)
    kwargs["source_video_timing"] = [_record(), _record()]
    with pytest.raises(ValueError, match="timing count"):
        native_processor(**kwargs, videos_kwargs={"do_sample_frames": False})


def test_native_processor_keeps_existing_remote_revision_goldens(native_processor: Any) -> None:
    from cosmos_framework.data.generator.processors.cosmos3_edge_processing_test import (
        test_image_matches_golden,
        test_text_only_matches_golden,
        test_video_matches_golden,
        test_wrapper_interface_surface,
    )

    # Execute the existing exact IDs/pixel hashes, recorded from remote revision 28a0b8e.
    test_text_only_matches_golden(native_processor)
    test_image_matches_golden(native_processor)
    test_video_matches_golden(native_processor)
    test_wrapper_interface_surface(native_processor)


def test_explicit_native_wrapper_accepts_pinned_legacy_metadata(native_processor: Any) -> None:
    from cosmos_framework.data.generator.processors import build_processor
    from cosmos_framework.data.generator.processors.nemotron3densevl_processor import Nemotron3DenseVLProcessor
    from cosmos_framework.utils.generator.source_video_timing import require_source_pts_processor

    path = os.environ["COSMOS3_EDGE_SNAPSHOT_DIR"]
    wrapper = build_processor(path, use_native_edge_processor=True)
    assert isinstance(wrapper, Nemotron3DenseVLProcessor)
    require_source_pts_processor(wrapper)
    messages = _messages(_record())
    messages[0]["content"][0]["fps"] = 4.0
    output = wrapper.apply_chat_template(messages)
    direct = _apply(native_processor, _messages(_record()))
    assert torch.equal(output["input_ids"], direct["input_ids"][0])
    assert torch.equal(output["pixel_values_videos"], direct["pixel_values_videos"])
    with pytest.raises(ValueError, match="staged local"):
        build_processor("not-a-local-model", use_native_edge_processor=True)
    with pytest.raises(ValueError, match="native Cosmos3-Edge"):
        require_source_pts_processor(object())
