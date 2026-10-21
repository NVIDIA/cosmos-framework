# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""CPU contracts for the rank-local decoded-image HPSv3 adapter."""

from __future__ import annotations

import io
import json
import time
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pytest
import torch

from cosmos_framework.posttrain.algorithm.reward import BoundRewardProvider, RewardProviderResult
from cosmos_framework.posttrain.config.rl import ImageRewardConfig
from cosmos_framework.posttrain.roles.reward.image import (
    HTTPImageRewardClient,
    ImageRewardRequest,
    build_image_reward_provider,
)
from cosmos_framework.posttrain.roles.reward.video import VideoRewardError, VideoRewardResult, _HTTPResponse
from cosmos_framework.posttrain.schema import (
    WFM_CONDITIONING_KEY,
    ObsBundle,
    Status,
    Trajectory,
    Transition,
    WFMConditioningInput,
)


def _config(**overrides: object) -> ImageRewardConfig:
    values: dict[str, object] = {
        "provider_revision": "hpsv3-model-2026-08",
        "poll_interval_s": 0.001,
    }
    values.update(overrides)
    return ImageRewardConfig(**values)


def _trajectory(*, modality: str = "image") -> Trajectory:
    conditioning = WFMConditioningInput(
        id="prompt-a",
        template_identity="image-reward-test-template-a",
        model_batch={"images": [[torch.zeros((3, 720, 720))]]},
        prompt="a red cube",
        modality=cast(Literal["image", "video"], modality),
    )
    return Trajectory(
        transitions=[
            Transition(
                obs=ObsBundle(
                    representations={WFM_CONDITIONING_KEY: conditioning},
                    id=conditioning.id,
                ),
                action=torch.zeros((1, 4, 1, 2, 2)),  # [1,C,T,H,W]
                policy_info={"seed": 7, "weight_version": 3},
                done=True,
            )
        ],
        id="sample-a",
        group_id=conditioning.id,
        status=Status.COMPLETED,
    )


class _PixelDecoder:
    output_range: str = "neg_one_to_one"

    def __init__(self, *, frames: int = 1, values: tuple[float, float, float] = (-1.0, 0.0, 1.0)) -> None:
        self.frames = frames
        self.values = values
        self.calls = 0

    def __call__(self, latents: list[torch.Tensor]) -> torch.Tensor:
        assert not torch.is_grad_enabled()
        assert len(latents) == 1
        self.calls += 1
        values = torch.tensor(self.values).reshape(1, 1, 1, 1, 3)  # [1,1,1,1,W=3]
        return values.expand(1, 3, self.frames, 1, 3).clone()


class _CaptureClient:
    poll_attempts: int = 1
    transport_retries: int = 0

    def __init__(self, score: float = 2.0) -> None:
        self.score_value = score
        self.requests: list[ImageRewardRequest] = []

    def score(self, request: ImageRewardRequest, *, deadline_monotonic: float) -> VideoRewardResult:
        assert deadline_monotonic > time.monotonic()
        self.requests.append(request)
        return VideoRewardResult(
            score=self.score_value,
            service_id="service-a",
        )


class _FaultClient:
    poll_attempts: int = 1
    transport_retries: int = 0

    def __init__(self, failure: str) -> None:
        self.failure = failure

    def score(self, request: ImageRewardRequest, *, deadline_monotonic: float) -> VideoRewardResult:
        assert deadline_monotonic > time.monotonic()
        del request
        return VideoRewardResult(
            score=cast(float, True if self.failure == "bool" else float("nan")),
            service_id="service-a",
        )


def _evaluate(provider: BoundRewardProvider, trajectory: Trajectory) -> RewardProviderResult:
    request = provider.prepare_group([trajectory])
    result = provider.evaluate_prepared_group(request)
    [item] = result.items
    return item.result


@pytest.mark.L0
@pytest.mark.CPU
def test_hpsv3_image_reward_decodes_once_and_sends_exact_uint8_payload() -> None:
    decoder = _PixelDecoder()
    client = _CaptureClient(score=100.0)
    provider = build_image_reward_provider(_config(), client, decoder, run_id="run-a")

    result = _evaluate(provider, _trajectory())

    assert result.terms == {"task": 50.0}
    assert provider.provider_revision == "hpsv3-model-2026-08"
    assert decoder.calls == 1
    [request] = client.requests
    assert request.image.dtype is torch.uint8
    assert request.image.device.type == "cpu"
    assert tuple(request.image.shape) == (1, 1, 3, 3)
    assert request.image[0, 0, 0].tolist() == [0, 0, 0]
    assert request.image[0, 0, 1].tolist() == [128, 128, 128]
    assert request.image[0, 0, 2].tolist() == [255, 255, 255]


@pytest.mark.L0
@pytest.mark.CPU
def test_hpsv3_image_reward_clamps_finite_decoder_overshoot() -> None:
    client = _CaptureClient()
    provider = build_image_reward_provider(
        _config(),
        client,
        _PixelDecoder(values=(-1.5, 0.0, 1.5)),
        run_id="run-a",
    )

    _evaluate(provider, _trajectory())

    [request] = client.requests
    assert request.image[0, 0, 0].tolist() == [0, 0, 0]
    assert request.image[0, 0, 1].tolist() == [128, 128, 128]
    assert request.image[0, 0, 2].tolist() == [255, 255, 255]


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    ("trajectory", "decoder", "message"),
    [
        pytest.param(
            _trajectory(modality="video"),
            _PixelDecoder(),
            "requires image conditioning",
            id="wrong-modality",
        ),
        pytest.param(
            _trajectory(),
            _PixelDecoder(frames=2),
            "temporal length T=1",
            id="temporal-shape",
        ),
    ],
)
def test_hpsv3_image_reward_rejects_wrong_modality_or_temporal_shape_before_client(
    trajectory: Trajectory,
    decoder: _PixelDecoder,
    message: str,
) -> None:
    client = _CaptureClient()
    provider = build_image_reward_provider(_config(), client, decoder, run_id="run-a")

    with pytest.raises(VideoRewardError, match=message):
        _evaluate(provider, trajectory)

    assert not client.requests


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    ("failure", "message"),
    [
        pytest.param("bool", "numeric and non-bool", id="bool-score"),
        pytest.param("nonfinite", "non-finite", id="nonfinite-score"),
    ],
)
def test_hpsv3_image_reward_rejects_invalid_scores(
    failure: str,
    message: str,
) -> None:
    provider = build_image_reward_provider(_config(), _FaultClient(failure), _PixelDecoder(), run_id="run-a")

    with pytest.raises(VideoRewardError, match=message):
        _evaluate(provider, _trajectory())


@pytest.mark.L0
@pytest.mark.CPU
def test_hpsv3_http_image_wire_omits_video_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REMOTE_REWARD_ENQUEUE_URL", "https://reward.invalid/enqueue")
    monkeypatch.setenv("REMOTE_REWARD_FETCH_URL", "https://reward.invalid/fetch")
    monkeypatch.setenv("REMOTE_REWARD_TOKEN", "secret")
    monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/curl")
    client = HTTPImageRewardClient(_config())
    payloads: list[bytes] = []

    def fake_post(
        _url: str,
        *,
        payload: bytes | Path,
        content_type: str,
        deadline_monotonic: float,
        headers: dict[str, str] | None = None,
    ) -> _HTTPResponse:
        del content_type, deadline_monotonic, headers
        payloads.append(payload if isinstance(payload, bytes) else payload.read_bytes())
        return _HTTPResponse(200, b'{"uuid":"service-a"}')

    monkeypatch.setattr(client, "_post", fake_post)
    monkeypatch.setattr(
        client,
        "_pull",
        lambda **_kwargs: _HTTPResponse(200, b'{"scores":{"hpsv3":[1.25]}}'),
    )
    image = torch.tensor([0, 128, 255], dtype=torch.uint8).reshape(1, 1, 1, 3)

    result = client.score(
        ImageRewardRequest("request-a", "sample-a", "a red cube", image, 7, 3),
        deadline_monotonic=time.monotonic() + 10.0,
    )

    assert result.score == 1.25
    metadata_raw, array_raw = payloads[0].split(b"\n", 1)
    metadata = json.loads(metadata_raw)
    assert metadata == {
        "media_type": "image",
        "prompts": ["a red cube"],
        "reward_fn": {"hpsv3": 1.0},
    }
    assert "video_infos" not in metadata
    loaded = np.load(io.BytesIO(array_raw), allow_pickle=False)
    np.testing.assert_array_equal(loaded, image.numpy())


@pytest.mark.L0
@pytest.mark.CPU
def test_hpsv3_http_image_accepts_correlated_pending_then_exact_singleton_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("REMOTE_REWARD_ENQUEUE_URL", "https://reward.invalid/enqueue")
    monkeypatch.setenv("REMOTE_REWARD_FETCH_URL", "https://reward.invalid/fetch")
    monkeypatch.setenv("REMOTE_REWARD_TOKEN", "secret")
    monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/curl")
    client = HTTPImageRewardClient(_config())
    responses = iter(
        (
            _HTTPResponse(200, b'{"uuid":"service-a"}'),
            _HTTPResponse(202, b'{"uuid":"service-a","status":"pending"}'),
            _HTTPResponse(400, b'{"message":"Not found scores for service-a"}'),
            _HTTPResponse(200, b'{"scores":{"hpsv3":[1.25]}}'),
        )
    )
    monkeypatch.setattr(client, "_post", lambda *_args, **_kwargs: next(responses))
    image = torch.tensor([0, 128, 255], dtype=torch.uint8).reshape(1, 1, 1, 3)

    result = client.score(
        ImageRewardRequest("request-a", "sample-a", "a red cube", image, 7, 3),
        deadline_monotonic=time.monotonic() + 2.0,
    )

    assert result == VideoRewardResult(score=1.25, service_id="service-a")
    assert client.poll_attempts == 3
    assert client.transport_retries == 0


@pytest.mark.L0
@pytest.mark.CPU
def test_hpsv3_http_image_pending_result_obeys_absolute_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REMOTE_REWARD_ENQUEUE_URL", "https://reward.invalid/enqueue")
    monkeypatch.setenv("REMOTE_REWARD_FETCH_URL", "https://reward.invalid/fetch")
    monkeypatch.setenv("REMOTE_REWARD_TOKEN", "secret")
    monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/curl")
    client = HTTPImageRewardClient(_config())
    responses = iter(
        (
            _HTTPResponse(200, b'{"uuid":"service-a"}'),
            _HTTPResponse(202, b'{"uuid":"service-a","status":"pending"}'),
        )
    )
    monkeypatch.setattr(client, "_post", lambda *_args, **_kwargs: next(responses))
    image = torch.tensor([0, 128, 255], dtype=torch.uint8).reshape(1, 1, 1, 3)

    with pytest.raises(VideoRewardError, match="absolute result deadline expired") as error_info:
        client.score(
            ImageRewardRequest("request-a", "sample-a", "a red cube", image, 7, 3),
            deadline_monotonic=0.0,
        )

    assert error_info.value.category == "timeout"
