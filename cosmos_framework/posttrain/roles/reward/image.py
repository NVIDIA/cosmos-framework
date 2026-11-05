# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Hard-bounded decoded-image reward adapter for rank-local online WFM."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, cast

import torch

from cosmos_framework.posttrain.algorithm.reward import (
    BoundRewardProvider,
    RewardCandidate,
    RewardCandidateResult,
    RewardContractError,
    RewardGroupRequest,
    RewardGroupResult,
    RewardProviderSpec,
)
from cosmos_framework.posttrain.config.rl import ImageRewardConfig
from cosmos_framework.posttrain.roles.reward.video import (
    DECODED_MEDIA_REWARD_DIAGNOSTICS,
    HTTPVideoRewardClient,
    VideoDecoder,
    VideoRewardClient,
    VideoRewardError,
    VideoRewardProvider,
    VideoRewardResult,
    _request_id,
)
from cosmos_framework.posttrain.schema import Status, Trajectory


@dataclass(frozen=True)
class ImageRewardRequest:
    """One decoded image submitted by one model rank."""

    request_id: str
    sample_id: str
    prompt: str
    image: torch.Tensor
    seed: int
    weight_version: int


class ImageRewardClient(Protocol):
    """Singular image client contract matching one trajectory per WFM rank."""

    poll_attempts: int
    transport_retries: int

    def score(self, request: ImageRewardRequest, *, deadline_monotonic: float) -> VideoRewardResult:
        """Return one exact result before the supplied absolute deadline."""
        ...


class HTTPImageRewardClient(HTTPVideoRewardClient):
    """Use the bounded decoded-media transport with an exact T=1 image payload."""

    def __init__(self, config: ImageRewardConfig) -> None:
        super().__init__(config)
        self.config = config

    def score(self, request: ImageRewardRequest, *, deadline_monotonic: float) -> VideoRewardResult:
        if request.image.dtype is not torch.uint8 or request.image.device.type != "cpu":
            raise VideoRewardError("protocol", "image reward wire tensor must be CPU uint8")
        if request.image.ndim != 4 or request.image.shape[0] != 1 or request.image.shape[-1] != 3:
            raise VideoRewardError(
                "protocol",
                f"image reward wire tensor must be [1,H,W,3], got {tuple(request.image.shape)}",
            )
        metadata = {
            "media_type": "image",
            "prompts": [request.prompt],
            "reward_fn": {self.config.provider_name: 1.0},
        }
        return self._score_payload(
            metadata=metadata,
            media=request.image.numpy(),
            deadline_monotonic=deadline_monotonic,
            media_type="image",
        )


class ImageRewardProvider(VideoRewardProvider):
    """Decode and score one rank-local T2I trajectory through HPSv3."""

    def __init__(
        self,
        config: ImageRewardConfig,
        client: ImageRewardClient,
        decoder: VideoDecoder,
        *,
        run_id: str,
    ) -> None:
        # The inherited evaluator is request-shape agnostic; this subclass replaces
        # ``_request`` and the concrete client consumes that exact image request.
        super().__init__(config, cast(VideoRewardClient, client), decoder, run_id=run_id)
        self.config = config

    def _request(self, trajectory: Trajectory) -> ImageRewardRequest:
        conditioning, latent, seed, weight_version = self._request_inputs(trajectory)
        if conditioning.modality != "image":
            raise VideoRewardError(
                "protocol",
                f"image reward provider requires image conditioning, got {conditioning.modality!r}",
            )
        # The production VAE can overshoot its nominal [-1,1] range slightly.
        # Image wire conversion below owns the exact saturation semantics.
        decoded = self._decode(latent, enforce_output_range=False)
        if decoded.shape[2] != 1:
            raise VideoRewardError(
                "protocol",
                f"image reward decoder requires temporal length T=1, got {decoded.shape[2]}",
            )
        image = (decoded.detach().to(device="cpu", dtype=torch.float32).clamp(-1.0, 1.0) + 1.0) * 127.5  # [1,3,1,H,W]
        image = image.round().to(dtype=torch.uint8).squeeze(2).permute(0, 2, 3, 1).contiguous()  # [1,H,W,3]
        return ImageRewardRequest(
            request_id=_request_id(
                self.run_id,
                self.config.provider_name,
                trajectory,
                seed,
                weight_version,
            ),
            sample_id=trajectory.id,
            prompt=conditioning.prompt,
            image=image,
            seed=seed,
            weight_version=weight_version,
        )


@dataclass(frozen=True)
class _PreparedImageRewardGroupRequest(RewardGroupRequest):
    """Canonical group envelope plus the local image trajectories to decode."""

    trajectories: tuple[Trajectory, ...] = ()

    def __post_init__(self) -> None:
        super().__post_init__()
        if len(self.trajectories) != len(self.candidates) or any(
            not isinstance(trajectory, Trajectory) for trajectory in self.trajectories
        ):
            raise RewardContractError(
                "decoded-image reward request must retain one trajectory per candidate, "
                f"got {len(self.trajectories)} trajectories and {len(self.candidates)} candidates"
            )
        trajectory_ids = tuple(trajectory.id for trajectory in self.trajectories)
        candidate_ids = tuple(candidate.trajectory_id for candidate in self.candidates)
        if trajectory_ids != candidate_ids:
            raise RewardContractError(
                "decoded-image reward request trajectory identities must match its candidate envelope, "
                f"got trajectories={trajectory_ids!r}, candidates={candidate_ids!r}"
            )


class _ImageGroupProvider:
    """Adapt the specialized image decoder to the canonical group result boundary."""

    def __init__(self, provider: ImageRewardProvider) -> None:
        self.provider = provider

    def evaluate(self, request: RewardGroupRequest) -> RewardGroupResult:
        if not isinstance(request, _PreparedImageRewardGroupRequest):
            raise RewardContractError("decoded-image reward provider requires its prepared trajectory group request")
        return RewardGroupResult(
            group_id=request.group_id,
            items=tuple(
                RewardCandidateResult(
                    trajectory_id=candidate.trajectory_id,
                    result=self.provider.evaluate(trajectory),
                )
                for candidate, trajectory in zip(request.candidates, request.trajectories, strict=True)
            ),
        )


class _BoundImageRewardProvider(BoundRewardProvider):
    """Bind latent-bearing image trajectories without widening the public group request schema."""

    def prepare_group(self, trajectories: Sequence[Trajectory]) -> RewardGroupRequest:
        group = tuple(trajectories)
        if not group:
            raise RewardContractError("decoded-image reward requires at least one trajectory")
        group_id: str | None = None
        for trajectory in group:
            if not isinstance(trajectory, Trajectory):
                raise RewardContractError(
                    f"decoded-image reward candidate must be a Trajectory, got {type(trajectory).__name__}"
                )
            if trajectory.status != Status.COMPLETED:
                raise RewardContractError(
                    f"decoded-image reward trajectory must be completed, got status={trajectory.status!r}"
                )
            if not trajectory.transitions or not trajectory.transitions[-1].done:
                raise RewardContractError("decoded-image reward trajectory must have a done terminal transition")
            if not isinstance(trajectory.id, str) or not trajectory.id.strip():
                raise RewardContractError(
                    f"decoded-image reward trajectory id must be a non-empty string, got {trajectory.id!r}"
                )
            if not isinstance(trajectory.group_id, str) or not trajectory.group_id.strip():
                raise RewardContractError(
                    f"decoded-image reward group_id must be a non-empty string, got {trajectory.group_id!r}"
                )
            if group_id is None:
                group_id = trajectory.group_id
            elif trajectory.group_id != group_id:
                raise RewardContractError(
                    f"decoded-image reward received mixed group_ids {group_id!r} and {trajectory.group_id!r}"
                )
            if trajectory.transitions[-1].obs.id != trajectory.group_id:
                raise RewardContractError("decoded-image reward terminal obs.id must match the trajectory group_id")
        if group_id is None:
            raise AssertionError("non-empty decoded-image group did not establish a group_id")
        return _PreparedImageRewardGroupRequest(
            group_id=group_id,
            candidates=tuple(RewardCandidate(trajectory_id=trajectory.id, completion="") for trajectory in group),
            trajectories=group,
        )


def build_image_reward_provider(
    config: ImageRewardConfig,
    client: ImageRewardClient,
    decoder: VideoDecoder,
    *,
    run_id: str,
) -> BoundRewardProvider:
    """Bind the T2I adapter to the canonical catalog result/provenance contract."""
    image_provider = ImageRewardProvider(config, client, decoder, run_id=run_id)
    group_provider = _ImageGroupProvider(image_provider)
    spec = RewardProviderSpec(
        factory=lambda: group_provider,
        revision=config.provider_revision,
        diagnostic_names=DECODED_MEDIA_REWARD_DIAGNOSTICS,
        term_names=("task",),
        semantic_identity={
            "adapter": "decoded_image_v1",
            "provider_name": config.provider_name,
            "provider_revision": config.provider_revision,
            "score_key": config.score_key,
            "connect_timeout_s": config.connect_timeout_s,
            "operation_timeout_s": config.operation_timeout_s,
            "total_deadline_s": config.total_deadline_s,
            "poll_interval_s": config.poll_interval_s,
            "pull_max_attempts": config.pull_max_attempts,
            "reward_clip_min": config.reward_clip_min,
            "reward_clip_max": config.reward_clip_max,
            "scale": config.scale,
        },
        is_remote=True,
    )
    return _BoundImageRewardProvider(config.provider_name, spec, spec.factory())


__all__ = [
    "HTTPImageRewardClient",
    "ImageRewardClient",
    "ImageRewardProvider",
    "ImageRewardRequest",
    "build_image_reward_provider",
]
