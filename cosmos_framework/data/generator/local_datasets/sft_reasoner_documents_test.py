# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Tests for finite, deterministic SFT Reasoner document enumeration."""

from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest

from cosmos_framework.data.generator.local_datasets import sft_dataset as sft_dataset_module
from cosmos_framework.data.generator.local_datasets import sft_reasoner_documents as reasoner_documents_module
from cosmos_framework.data.generator.local_datasets.sft_dataset import (
    SFTDataset,
    enumerate_sft_captions,
    enumerate_sft_cfg_variants,
    render_sft_caption,
    resolve_sft_window_framing,
    sft_metadata_sort_key,
)
from cosmos_framework.data.generator.local_datasets.sft_reasoner_documents import (
    SFTVideoInfoProbe,
    iter_sft_reasoner_documents,
)


def _metadata(uuid: str, *windows: dict[str, Any]) -> dict[str, Any]:
    return {
        "uuid": uuid,
        "vision_path": f"/videos/{uuid}.mp4",
        "width": 256,
        "height": 144,
        "aspect_ratio": "test",
        "t2w_windows": list(windows),
    }


def _dataset(metadata: list[dict[str, Any]], **overrides: Any) -> SFTDataset:
    dataset = object.__new__(SFTDataset)
    defaults = {
        "metadata": metadata,
        "s3_credentials": {},
        "num_video_frames": -1,
        "temporal_interval_mode": "max_30fps",
        "frame_selection_mode": "first",
        "temporal_compression_factor": 4,
        "output_sizes": {"test": (256, 144)},
        "cfg_dropout_rate": 0.1,
        "cfg_dropout_keep_metadata": False,
        "caption_suffix": "quality suffix",
        "append_duration_fps_timestamps": True,
        "append_resolution_info": True,
        "conditioning_fps": -1,
        "conditioning_fps_noise_std": 0.0,
        "conditioning_config": None,
        "is_initialized": False,
        "s3_client": object(),
    }
    defaults.update(overrides)
    for name, value in defaults.items():
        setattr(dataset, name, value)

    def tokenize(caption: str) -> tuple[list[int], str]:
        return [len(caption), sum(ord(char) for char in caption)], caption

    dataset._tokenize_caption = tokenize
    return dataset


def _video_info(_: dict[str, Any]) -> Mapping[str, Any]:
    return {"fps": 20.0, "total_frames": 40, "decoded_total_frames": 40}


def test_reasoner_documents_are_finite_stably_sorted_and_expand_windows_and_cfg():
    metadata = [
        _metadata(
            "video-b",
            {"start_frame": 0, "end_frame": 8, "temporal_interval": 2, "caption": "first"},
            {"start_frame": 10, "end_frame": 18, "temporal_interval": 2, "caption": "second"},
        ),
        _metadata(
            "video-a",
            {"start_frame": 20, "end_frame": 28, "temporal_interval": 2, "caption": "third"},
        ),
    ]
    dataset = _dataset(metadata)

    first = list(iter_sft_reasoner_documents(dataset, video_info_resolver=_video_info))
    dataset.metadata = list(reversed(dataset.metadata))
    second = list(iter_sft_reasoner_documents(dataset, video_info_resolver=_video_info))

    assert first == second
    assert len(first) == 6
    expected_keys = [
        f"{item['uuid']}_w{window_index}"
        for item in sorted(metadata, key=sft_metadata_sort_key)
        for window_index in range(len(item["t2w_windows"]))
        for _ in range(2)
    ]
    assert [document.sample_key for document in first] == expected_keys
    assert [document.cfg_dropped for document in first] == [False, True] * 3
    assert all(document.window_index == document.framing.window_index for document in first)
    assert all(document.conditioning_fps == 10.0 for document in first)
    assert all(document.framing.num_frames == 5 for document in first)
    assert all(document.caption == "" for document in first if document.cfg_dropped)
    assert all("0.5 seconds" in document.caption for document in first if not document.cfg_dropped)


def test_reasoner_document_matches_training_sample_caption_window_and_tokens(monkeypatch):
    metadata = _metadata(
        "video",
        {"start_frame": 0, "end_frame": 8, "temporal_interval": 1, "caption": "same prompt"},
    )
    metadata["width"] = 8
    metadata["height"] = 4
    dataset = _dataset([metadata], cfg_dropout_rate=0.0, output_sizes={"test": (8, 4)})

    monkeypatch.setattr(sft_dataset_module, "download_from_s3", lambda *_args, **_kwargs: b"video")
    monkeypatch.setattr(
        sft_dataset_module,
        "get_video_metadata",
        lambda _path: {"fps": 20.0, "total_frames": 40},
    )
    monkeypatch.setattr(
        sft_dataset_module,
        "ffmpeg_decode_video",
        lambda *_args, **_kwargs: iter(np.zeros((7, 4, 8, 3), dtype=np.uint8)),
    )
    monkeypatch.setattr(sft_dataset_module.random, "randrange", lambda _size: 0)

    training_sample = dataset.process_one_sample(metadata)
    documents = list(
        iter_sft_reasoner_documents(
            dataset,
            video_info_resolver=lambda _metadata: {
                "fps": 20.0,
                "total_frames": 40,
                "decoded_total_frames": 7,
            },
        )
    )

    assert training_sample is not None
    assert len(documents) == 1
    document = documents[0]
    assert training_sample["__key__"] == document.sample_key
    assert training_sample["ai_caption"] == document.caption
    assert tuple(training_sample["text_token_ids"].tolist()) == document.text_token_ids
    assert training_sample["frame_start"] == document.framing.start_frame
    assert training_sample["frame_end"] == document.framing.end_frame
    assert training_sample["num_frames"] == document.framing.num_frames


def test_default_video_probe_counts_frames_from_the_real_decode_path(monkeypatch):
    metadata = _metadata(
        "video",
        {"start_frame": 0, "end_frame": 8, "temporal_interval": 1, "caption": "x"},
    )
    seen_decode_args: dict[str, Any] = {}
    monkeypatch.setattr(reasoner_documents_module.boto3, "client", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(reasoner_documents_module, "download_from_s3", lambda *_args, **_kwargs: b"video")
    monkeypatch.setattr(
        reasoner_documents_module,
        "get_video_metadata",
        lambda _path: {"width": 256, "height": 144, "fps": 20.0, "total_frames": 40},
    )

    def decode(_path, *, scale_hw, num_threads):
        seen_decode_args.update(scale_hw=scale_hw, num_threads=num_threads)
        return iter(range(7))

    monkeypatch.setattr(reasoner_documents_module, "ffmpeg_decode_video", decode)

    info = SFTVideoInfoProbe({}, {"test": (256, 144)})(metadata)

    assert info["decoded_total_frames"] == 7
    assert seen_decode_args == {"scale_hw": (144, 256), "num_threads": 2}


def test_reasoner_documents_enumerate_every_positive_weight_caption_in_stable_order():
    window = {
        "start_frame": 0,
        "end_frame": 8,
        "temporal_interval": 2,
        "qwen3_235b_dense": "dense",
        "qwen3_32b_short": "short",
        "qwen3_235b_temporal": "unreachable",
    }
    dataset = _dataset([_metadata("video", window)], cfg_dropout_rate=0.0)

    documents = list(iter_sft_reasoner_documents(dataset, video_info_resolver=_video_info))

    assert [document.caption_key for document in documents] == ["qwen3_32b_short", "qwen3_235b_dense"]
    assert [document.cfg_dropped for document in documents] == [False, False]


def test_caption_priority_shadows_weighted_fallbacks_during_enumeration():
    selections = enumerate_sft_captions(
        {
            "caption": "preferred",
            "qwen3_235b_dense": "fallback",
        }
    )

    assert selections == (("caption", "preferred.", False),)


def test_conditioning_fps_noise_fails_before_resolving_any_video():
    dataset = _dataset(
        [_metadata("video", {"start_frame": 0, "end_frame": 8, "temporal_interval": 2, "caption": "x"})],
        conditioning_fps_noise_std=0.1,
    )
    resolver_called = False

    def resolver(_: dict[str, Any]) -> Mapping[str, Any]:
        nonlocal resolver_called
        resolver_called = True
        return _video_info({})

    with pytest.raises(ValueError, match="conditioning_fps_noise_std=0"):
        list(iter_sft_reasoner_documents(dataset, video_info_resolver=resolver))
    assert not resolver_called


def test_custom_video_info_resolver_must_report_actual_decoded_frame_count():
    dataset = _dataset([_metadata("video", {"start_frame": 0, "end_frame": 8, "temporal_interval": 1, "caption": "x"})])

    with pytest.raises(ValueError, match="decoded_total_frames"):
        list(
            iter_sft_reasoner_documents(
                dataset,
                video_info_resolver=lambda _metadata: {"fps": 20.0, "total_frames": 40},
            )
        )


def test_random_fixed_length_selection_fails_closed():
    dataset = _dataset(
        [_metadata("video", {"start_frame": 0, "end_frame": 20, "temporal_interval": 1, "caption": "x"})],
        num_video_frames=5,
        frame_selection_mode="random",
    )

    with pytest.raises(ValueError, match="random fixed-length"):
        list(iter_sft_reasoner_documents(dataset, video_info_resolver=_video_info))


def test_cfg_endpoint_variants_and_flattened_window_key_match_training_semantics():
    metadata = _metadata(
        "video_w3",
        {"start_frame": 0, "end_frame": 8, "temporal_interval": 2, "caption": "x"},
    )
    dataset = _dataset([metadata], cfg_dropout_rate=1.0)

    documents = list(iter_sft_reasoner_documents(dataset, video_info_resolver=_video_info))

    assert enumerate_sft_cfg_variants(0.0) == (False,)
    assert enumerate_sft_cfg_variants(0.5) == (False, True)
    assert enumerate_sft_cfg_variants(1.0) == (True,)
    assert [(document.sample_key, document.cfg_dropped) for document in documents] == [("video_w3_w0", True)]


def test_enumerator_rejects_dataset_after_infinite_iterator_initialization():
    dataset = _dataset([], is_initialized=True)

    with pytest.raises(ValueError, match="before SFTDataset.__iter__"):
        list(iter_sft_reasoner_documents(dataset, video_info_resolver=_video_info))


def test_shared_caption_renderer_preserves_metadata_for_cfg_when_configured():
    caption = render_sft_caption(
        "description.",
        used_structured_json=False,
        cfg_dropped=True,
        cfg_dropout_keep_metadata=True,
        caption_suffix="quality suffix",
        append_duration_fps_timestamps=True,
        append_resolution_info=True,
        num_frames=5,
        conditioning_fps=10.0,
        target_height=144,
        target_width=256,
    )

    assert caption == ("The video is 0.5 seconds long and is of 10 FPS. This video is of 144x256 resolution.")


def test_shared_caption_renderer_keeps_structured_json_byte_stable():
    caption = render_sft_caption(
        '{"fps": 5}',
        used_structured_json=True,
        cfg_dropped=False,
        cfg_dropout_keep_metadata=False,
        caption_suffix="must not be appended",
        append_duration_fps_timestamps=True,
        append_resolution_info=True,
        num_frames=5,
        conditioning_fps=10.0,
        target_height=144,
        target_width=256,
    )

    assert caption == '{"fps": 5}'


def test_shared_window_framing_matches_native_window_clamp_and_truncation():
    metadata = _metadata(
        "video",
        {"start_frame": 5, "end_frame": 25, "temporal_interval": 3, "caption": "x"},
    )

    framing = resolve_sft_window_framing(
        metadata,
        0,
        original_fps=30.0,
        total_frames=20,
        num_video_frames=-1,
        temporal_interval_mode="max_30fps",
        frame_selection_mode="first",
        temporal_compression_factor=4,
        target_height=144,
        target_width=256,
    )

    assert framing is not None
    assert framing.sample_key == "video_w0"
    assert (framing.start_frame, framing.end_frame, framing.temporal_interval) == (5, 19, 3)
    assert framing.num_frames == 5


def test_shared_window_framing_counts_only_frames_the_training_decoder_can_reach():
    metadata = _metadata(
        "video",
        {"start_frame": 0, "end_frame": 99, "temporal_interval": 1, "caption": "x"},
    )

    framing = resolve_sft_window_framing(
        metadata,
        0,
        original_fps=60.0,
        total_frames=100,
        num_video_frames=93,
        temporal_interval_mode="max_30fps",
        frame_selection_mode="first",
        temporal_compression_factor=4,
        target_height=144,
        target_width=256,
    )

    assert framing is not None
    assert (framing.start_frame, framing.end_frame, framing.temporal_interval) == (0, 184, 2)
    assert framing.num_frames == 49
