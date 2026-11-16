# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Session orchestration with real CPU KV storage and a weight-free sampler."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from cosmos_framework.inference.bimanual import BimanualActionContract
from cosmos_framework.inference.inference import OmniInference
from cosmos_framework.model.generator.utils.action_stream import ARHistorySnapshot
from cosmos_framework.model.generator.utils.kv_cache import GenKVCache


class _SessionModel:
    def __init__(self):
        self.config = SimpleNamespace(
            video_temporal_causal=True,
            compile=SimpleNamespace(enabled=False, use_cuda_graphs=False),
            fixed_step_sampler_config=SimpleNamespace(t_list=[1.0, 0.5]),
            max_action_dim=64,
            teacher_forcing_frames_per_chunk=2,
        )
        self._bimanual_action_contract = BimanualActionContract(59, 64, {"agibotworld": 15, "camera_pose": 2})
        self.tensor_kwargs = {"device": "cpu"}
        self.input_video_key = "video"
        self.input_image_key = "images"
        self.input_caption_key = "caption"
        self.tokenizer_vision_gen = SimpleNamespace(
            temporal_compression_factor=4, use_cached_decoder=self.decoder_scope
        )
        self.entries = self.exits = self.calls = 0
        self.fail = None
        self.caches = []

    @contextmanager
    def decoder_scope(self):
        self.entries += 1
        self.decoded = 0
        try:
            yield
        finally:
            self.exits += 1

    def decode(self, latent):
        if self.fail == "decode":
            raise RuntimeError("decoder failure")
        rgb = latent.repeat_interleave(1 if self.decoded == 0 else 4, dim=2)
        self.decoded += 1
        return rgb

    def iter_samples_from_batch_autoregressive(self, batch, *, on_kv_cache_update, **kwargs):
        assert "is_preprocessed" not in batch  # reset must not reuse a mutated batch
        assert batch["caption"] == ["test task"]
        batch["is_preprocessed"] = True
        caches = [GenKVCache() for _ in range(2)]
        self.caches.append(caches)
        key = torch.ones(1, 2, 1, 2)
        frame = 1
        try:
            message = yield {"vision": torch.zeros(1, 3, 1, 2, 2)}
            for cache in caches:
                cache.store_kv(key, key, frame_idx=0)
            while True:
                self.calls += 1
                if self.fail == "sampler":
                    raise RuntimeError("sampler failure")
                assert message.actions.shape == (8, 64)
                assert not message.actions[:, 59:].any()
                assert message.domain_ids.tolist() == [15] * 4 + [2] * 4
                for index in range(frame, frame + 2):
                    for cache in caches:
                        cache.store_kv(key * self.calls, key, frame_idx=index)
                frame += 2
                on_kv_cache_update(ARHistorySnapshot(frame, tuple(c.retained_frame_indices(frame) for c in caches), ()))
                latent = torch.full((1, 3, 2, 2, 2), float(message.actions[:, 0].mean()))
                message = yield {"vision": latent}
        finally:
            for cache in caches:
                cache.reset()


@pytest.fixture
def runtime(tmp_path):
    image = tmp_path / "initial.png"
    Image.new("RGB", (832, 480)).save(image)
    return SimpleNamespace(model=_SessionModel(), vae_decode_stream=None), image


def _controls(value=0.0):
    return {
        "action_space": "normalized_unified_v1",
        "action": [[value] + [0.0] * 58 for _ in range(8)],
        "domain_names": ["agibotworld"] * 4 + ["camera_pose"] * 4,
    }


@pytest.mark.L0
@pytest.mark.CPU
def test_public_session_feedback_history_reset_and_exclusive_decoder(runtime):
    pipe, image = runtime
    with OmniInference.open_bimanual_session(pipe, vision_path=image, prompt="test task") as session:
        assert session.observation.shape == (1, 3, 1, 2, 2)
        original_id = session.inspect()["session_id"]
        with pytest.raises(RuntimeError, match="active Bimanual session"):
            OmniInference._generate_batch(pipe, [], {})
        with pytest.raises(RuntimeError, match="active Bimanual session"):
            OmniInference.open_bimanual_session(pipe, vision_path=image, prompt="test task")
        for index in range(5):
            # A delayed control decision uses the latest decoded observation.
            value = float(session.observation.mean()) if index == 4 else 0.0
            rgb = session.step(_controls(value), control_source="policy" if index < 3 else "human")
        assert float(rgb.mean()) > 0.5
        state = session.inspect()
        assert state["conditional_history"] == [list(range(11))] * 2
        assert state["minimum_retained_history_span_seconds"] > 1
        assert state["decoded_frames"] == 41 and state["control_source"] == "human"
        session.reset(seed=123)
        state = session.inspect()
        assert state["session_id"] != original_id and state["seed"] == 123
        assert state["conditional_history"] == [] and state["generated_actions"] == 0
        assert state["decoded_frames"] == 1
        assert all(not cache.retained_frame_indices(11) for cache in pipe.model.caches[0])
        session.step(_controls())
    assert pipe.model.entries == pipe.model.exits == 2
    assert session.closed and session.observation is None
    assert all(not cache.retained_frame_indices(11) for group in pipe.model.caches for cache in group)


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("failure", ["sampler", "decode"])
def test_invalid_control_can_retry_and_runtime_failure_releases_session(runtime, failure):
    pipe, image = runtime
    session = OmniInference.open_bimanual_session(pipe, vision_path=image, prompt="test task")
    bad = _controls()
    bad["action"][4][9] = 1.0  # robot-only slot in a camera row
    with pytest.raises(ValueError, match="invalid unified-v1"):
        session.step(bad)
    assert not session.closed and pipe.model.calls == 0
    pipe.model.fail = failure
    with pytest.raises(RuntimeError, match="failure"):
        session.step(_controls())
    assert session.closed and pipe.model.entries == pipe.model.exits == 1
    assert all(not cache.retained_frame_indices(3) for cache in pipe.model.caches[0])
    pipe.model.fail = None
    with OmniInference.open_bimanual_session(pipe, vision_path=image, prompt="test task") as fresh:
        assert fresh.inspect()["conditional_history"] == []
        fresh.step(_controls())
