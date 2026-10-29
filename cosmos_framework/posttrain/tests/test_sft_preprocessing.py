# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""Cache correctness and real spawned CPU preprocessing without cursor/model access."""

from __future__ import annotations

import hashlib
import json
import os
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers.feature_extraction_utils import BatchFeature
from transformers.video_utils import VideoMetadata

from cosmos_framework.posttrain.data.packer.sft import QwenSFTDataPacker, SFTPolicyInput, build_sft_packer
from cosmos_framework.posttrain.data.packer.sft_preprocess import (
    _ByteLRU,
    _cpu_bytes,
    _PreprocessingCache,
    build_sft_preprocessing,
)
from cosmos_framework.posttrain.schema import SFTTarget
from cosmos_framework.posttrain.tests.helpers.sft_packer import local_processor, local_tokenizer

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


class _VideoProcessor:
    def __init__(self):
        self.calls = 0

    def __call__(self, *, videos, num_frames, return_tensors):
        assert len(videos) == 1 and return_tensors == "pt"
        self.calls += 1
        return BatchFeature(
            {
                "pixel_values_videos": torch.arange(num_frames * 8).reshape(num_frames, 8).float(),
                "video_grid_thw": torch.tensor([[num_frames // 2, 2, 2]]),
                "video_metadata": [
                    VideoMetadata(total_num_frames=20, fps=10.0, frames_indices=list(range(num_frames)))
                ],
            }
        )


class _Packer:
    def __init__(self, *, template="same"):
        self.processor = SimpleNamespace(video_processor=_VideoProcessor())
        self.template, self.calls = template, 0
        self.frames = 4

    def validation_semantics_identity(self):
        return {"template": self.template}

    def sft_process_sample(self, target):
        self.calls += 1
        video = target.messages[0]["content"][0]["video"]
        outputs = self.processor.video_processor(videos=[video], num_frames=self.frames, return_tensors="pt")
        metadata = outputs.pop("video_metadata")[0]
        outputs["timestamps"] = metadata.timestamps
        # This mutation must never leak into the stored video cache entry.
        metadata.frames_indices[0] = 99
        answer = target.messages[-1]["content"]
        return SFTPolicyInput([1, len(answer), 3], [-100, len(answer), 3], outputs)


def _video_row(tmp_path, answer="A"):
    path = tmp_path / "video.mp4"
    path.write_bytes(b"deterministic fixture media")
    target = SFTTarget(
        [
            {"role": "user", "content": [{"type": "video", "video": str(path)}]},
            {"role": "assistant", "content": answer},
        ]
    )
    return target, {str(path): hashlib.sha256(path.read_bytes()).hexdigest()}


def test_video_cache_hits_across_distinct_questions_and_keeps_metadata(tmp_path):
    first, hashes = _video_row(tmp_path)
    second, _ = _video_row(tmp_path, answer="BB")
    packer = _Packer()
    original_video = packer.processor.video_processor
    cached = _PreprocessingCache(packer, cache_bytes=100_000, media_hashes=hashes)
    a, b = cached.process(first), cached.process(second)
    assert original_video.calls == 1 and packer.calls == 2
    assert a.input_ids != b.input_ids
    assert a.mm_kwargs["timestamps"] == b.mm_kwargs["timestamps"] == [0, 0.1, 0.2, 0.3]
    assert cached.stats()["videos"]["hits"] == 1
    torch.testing.assert_close(a.mm_kwargs["pixel_values_videos"], b.mm_kwargs["pixel_values_videos"])
    a.input_ids[0] = 99
    a.labels[0] = 99
    a.mm_kwargs["pixel_values_videos"].zero_()
    again = cached.process(first)
    assert again.input_ids[0] == 1 and again.labels[0] == -100
    assert again.mm_kwargs["pixel_values_videos"].count_nonzero() > 0
    assert original_video.calls == 1 and packer.calls == 2
    assert sum(value["bytes"] for value in cached.stats().values()) <= 100_000


def test_video_call_settings_and_processor_identity_are_in_cache_keys(tmp_path):
    first, hashes = _video_row(tmp_path)
    second, _ = _video_row(tmp_path, answer="BB")
    packer = _Packer()
    original_video = packer.processor.video_processor
    cached = _PreprocessingCache(packer, cache_bytes=100_000, media_hashes=hashes)
    cached.process(first)
    packer.frames = 6
    result = cached.process(second)
    assert original_video.calls == 2 and result.mm_kwargs["pixel_values_videos"].shape == (6, 8)
    other = _PreprocessingCache(_Packer(template="changed"), cache_bytes=100_000, media_hashes=hashes)
    assert cached.key(first) != other.key(first)


def test_cache_byte_limit_eviction_and_clone_isolation():
    item = {"pixels": torch.ones(8, 8)}
    cache = _ByteLRU(_cpu_bytes(item) + 1)
    cache.put("first", item)
    item["pixels"].zero_()
    assert cache.get("first")["pixels"].sum() == 64
    cache.put("second", {"pixels": torch.full((8, 8), 2.0)})
    assert cache.get("first") is None
    assert cache.get("second")["pixels"].sum() == 128
    cache.put("oversize", torch.ones(1024))
    assert cache.get("oversize") is None and cache.bytes <= cache.limit
    with pytest.raises(ValueError, match="detached CPU"):
        cache.put("graph", torch.ones(2, requires_grad=True))
    view = torch.ones(100_000)[:1]
    assert _cpu_bytes(view) >= 400_000
    cache.put("tiny_view_large_storage", view)
    assert cache.get("tiny_view_large_storage") is None


@pytest.mark.parametrize("failure", ["missing", "digest", "mutated", "mtime_restored"])
def test_cache_refuses_unverified_or_changed_media(tmp_path, failure):
    target, hashes = _video_row(tmp_path)
    if failure == "missing":
        hashes = {}
    elif failure == "digest":
        hashes = {path: "0" * 64 for path in hashes}
    cache = _PreprocessingCache(_Packer(), cache_bytes=100_000, media_hashes=hashes)
    if failure in {"mutated", "mtime_restored"}:
        cache.process(target)
        path = tmp_path / "video.mp4"
        before = path.stat()
        path.write_bytes(b"x" * before.st_size)
        if failure == "mtime_restored":
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(ValueError, match="manifest|digest|changed"):
        cache.process(target)


@pytest.mark.parametrize("source", ["python", "numpy", "torch"])
def test_cache_refuses_stochastic_preprocessing_and_restores_rng(tmp_path, source):
    target, hashes = _video_row(tmp_path)
    packer = _Packer()
    process = packer.sft_process_sample

    def stochastic(row):
        {"python": random.random, "numpy": np.random.random, "torch": lambda: torch.rand(1)}[source]()
        return process(row)

    packer.sft_process_sample = stochastic
    cache = _PreprocessingCache(packer, cache_bytes=100_000, media_hashes=hashes)
    before_python, before_numpy, before_torch = random.getstate(), np.random.get_state(), torch.get_rng_state()
    with pytest.raises(ValueError, match="must not consume RNG"):
        cache.process(target)
    assert random.getstate() == before_python
    assert np.array_equal(np.random.get_state()[1], before_numpy[1])
    assert torch.equal(torch.get_rng_state(), before_torch)
    assert cache.samples.bytes == 0


def _local_pipeline(tmp_path, *, prefetch):
    local_tokenizer().save_pretrained(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen2"}))
    packer = build_sft_packer(str(tmp_path), max_length=128)
    pipeline = build_sft_preprocessing(
        packer,
        name_or_path=str(tmp_path),
        cache_bytes=100_000,
        prefetch=prefetch,
        media_hashes={},
        max_pending=2,
    )
    return packer, pipeline


def _text_row(answer):
    return SFTTarget([{"role": "user", "content": "question"}, {"role": "assistant", "content": answer}])


def test_spawned_prefetch_preserves_order_identity_rng_and_cold_resume(tmp_path):
    packer, pipeline = _local_pipeline(tmp_path, prefetch=True)
    rows = [_text_row("A"), _text_row("BB")]
    expected = [packer.sft_process_sample(row) for row in rows]
    identity = packer.validation_semantics_identity()
    before_rng = torch.get_rng_state().clone()
    try:
        pipeline.prepare_window(rows)
        with pytest.raises(RuntimeError, match="another window"):
            pipeline.prepare_window(rows)
        with pytest.raises(ValueError, match="row order"):
            pipeline.sft_process_sample(rows[1])
        actual = [pipeline.sft_process_sample(row) for row in rows]
        pipeline.finish_window()
        assert actual == expected and pipeline.validation_semantics_identity() == identity
        assert torch.equal(before_rng, torch.get_rng_state())
        # Validation outside a training window shares the same worker and hits.
        assert pipeline.sft_process_sample(rows[0]) == expected[0]
        assert pipeline.preprocessing_stats["samples"]["hits"] == 1
    finally:
        pipeline.close()
    with pytest.raises(RuntimeError, match="closed"):
        pipeline.sft_process_sample(rows[0])
    _, resumed = _local_pipeline(tmp_path, prefetch=True)
    try:
        resumed.prepare_window(rows)
        assert [resumed.sft_process_sample(row) for row in rows] == expected
        resumed.finish_window()
    finally:
        resumed.close()


def test_prefetch_bounds_unconsumed_window_and_worker_failure(tmp_path):
    _, pipeline = _local_pipeline(tmp_path, prefetch=True)
    row = _text_row("A")
    try:
        with pytest.raises(ValueError, match="pending-row bound"):
            pipeline.prepare_window([row] * 3)
        pipeline.prepare_window([row])
        with pytest.raises(RuntimeError, match="completely consumed"):
            pipeline.finish_window()
        pipeline.sft_process_sample(row)
        pipeline.finish_window()
        broken = SFTTarget([{"role": "user", "content": "has no supervised answer"}])
        pipeline.prepare_window([broken])
        with pytest.raises(ValueError):
            pipeline.sft_process_sample(broken)
    finally:
        pipeline.close()


def test_synchronous_cache_preserves_processor_identity_and_no_window_prefetch(tmp_path):
    from cosmos_framework.posttrain.algorithm.sft import make_objective

    packer, pipeline = _local_pipeline(tmp_path, prefetch=False)
    row = _text_row("A")
    assert pipeline.validation_semantics_identity() == packer.validation_semantics_identity()
    assert make_objective(pipeline).identity == make_objective(packer).identity
    assert pipeline.sft_process_sample(row) == packer.sft_process_sample(row)
    assert pipeline.sft_process_sample(row) == packer.sft_process_sample(row)
    assert pipeline.preprocessing_stats["samples"]["hits"] == 1
    with pytest.raises(ValueError, match="prefetch=True"):
        pipeline.prepare_window([row])
    packer.max_length += 1
    with pytest.raises(ValueError, match="settings changed"):
        pipeline.sft_process_sample(row)
    pipeline.close()


def test_real_qwen_video_processor_cache_preserves_expanded_tokens_and_masks(tmp_path, monkeypatch):
    """Mock only file decoding; use the pinned real media/ChatML/label pipeline."""
    first, hashes = _video_row(tmp_path)
    second, _ = _video_row(tmp_path, answer="BB")
    reads = []

    def decode(path, **kwargs):
        reads.append(path)
        frames = np.arange(4 * 8 * 8 * 3, dtype=np.uint8).reshape(4, 8, 8, 3)
        return frames, VideoMetadata(total_num_frames=4, fps=10, frames_indices=[0, 1, 2, 3])

    monkeypatch.setattr("transformers.video_processing_utils.load_video", decode)
    tokenizer = local_tokenizer()
    packer = QwenSFTDataPacker(tokenizer, max_length=512, processor=local_processor(tokenizer))
    expected = [packer.sft_process_sample(row) for row in (first, second)]
    assert len(reads) == 2
    identity = packer.validation_semantics_identity()
    cache = _PreprocessingCache(packer, cache_bytes=1_000_000, media_hashes=hashes)
    actual = [cache.process(row) for row in (first, second)]
    assert len(reads) == 3
    assert packer.validation_semantics_identity() == identity
    for a, b in zip(actual, expected, strict=True):
        assert a.input_ids == b.input_ids and a.labels == b.labels and a.masking == b.masking
        assert a.mm_kwargs.keys() == b.mm_kwargs.keys()
        for key in a.mm_kwargs:
            torch.testing.assert_close(a.mm_kwargs[key], b.mm_kwargs[key], rtol=0, atol=0)
