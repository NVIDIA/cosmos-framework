# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Hard-bounded decoded-video reward adapter for rank-local online WFM."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
import urllib.parse
from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
import torch

from cosmos_framework.posttrain.algorithm.reward import (
    BoundRewardProvider,
    RewardCandidate,
    RewardCandidateResult,
    RewardContractError,
    RewardGroupRequest,
    RewardGroupResult,
    RewardProviderResult,
    RewardProviderSpec,
)
from cosmos_framework.posttrain.config.rl import VideoRewardConfig
from cosmos_framework.posttrain.schema import (
    WFM_CONDITIONING_KEY,
    MetricContribution,
    Status,
    Trajectory,
    WFMConditioningInput,
)


class VideoDecoder(Protocol):
    """Model-owned decoder that explicitly attests its floating ``[-1,1]`` output range."""

    output_range: Literal["neg_one_to_one"]

    def __call__(self, latents: list[torch.Tensor]) -> torch.Tensor:
        """Decode one uniform latent batch to ``[1,3,T,H,W]``."""
        ...


VideoRewardFailureCategory = Literal["timeout", "transport", "protocol", "nonfinite"]

DECODED_MEDIA_REWARD_DIAGNOSTICS: tuple[str, ...] = (
    "evaluation_latency_s",
    "raw_score",
    "transformed_score",
    "clipped_count",
    "poll_attempts",
    "transport_retries",
    "clip_min",
    "clip_max",
    "scale",
)
# Compatibility alias for the video adapter's original public diagnostic inventory.
VIDEO_REWARD_DIAGNOSTICS = DECODED_MEDIA_REWARD_DIAGNOSTICS


class VideoRewardError(RuntimeError):
    """Stable categorized failure at the decoded-video reward boundary."""

    category: VideoRewardFailureCategory

    def __init__(self, category: VideoRewardFailureCategory, message: str) -> None:
        super().__init__(message)
        self.category = category


@dataclass(frozen=True)
class VideoRewardRequest:
    """One decoded sample submitted by one model rank."""

    request_id: str
    sample_id: str
    prompt: str
    video: torch.Tensor
    seed: int
    weight_version: int
    video_fps: float


@dataclass(frozen=True)
class VideoRewardResult:
    """One positional service score and its enqueue-assigned job identifier."""

    score: float
    service_id: str


class VideoRewardClient(Protocol):
    """Singular client contract matching one trajectory per distributed WFM rank."""

    poll_attempts: int
    transport_retries: int

    def score(self, request: VideoRewardRequest, *, deadline_monotonic: float) -> VideoRewardResult:
        """Return one exact result before the supplied absolute deadline."""
        ...


@dataclass(frozen=True)
class _HTTPResponse:
    status_code: int
    body: bytes

    def json_object(self, operation: str) -> dict[str, object]:
        try:
            value = json.loads(self.body)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise VideoRewardError("protocol", f"video reward {operation} returned invalid JSON") from error
        if not isinstance(value, dict):
            raise VideoRewardError(
                "protocol",
                f"video reward {operation} returned {type(value).__name__}, expected object",
            )
        return value


def _curl_config_value(value: str) -> str:
    """Quote one trusted config value without allowing a new curl directive."""
    if "\n" in value or "\r" in value or "\0" in value:
        raise ValueError("curl config values must not contain newlines or NUL bytes")
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _write_private_file(path: Path, payload: bytes) -> None:
    """Create one process-private file without a permissive umask window."""
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "wb") as output:
        output.write(payload)


def _scrub_private_file(path: Path) -> None:
    """Best-effort overwrite and unlink for a short-lived credential file."""
    try:
        with path.open("r+b", buffering=0) as output:
            size = os.fstat(output.fileno()).st_size
            output.write(b"\0" * size)
    except OSError:
        pass
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


class HTTPVideoRewardClient:
    """Hard-bounded client for the deployed DanceGRPO enqueue/pull service.

    Each HTTP operation runs in a separate curl process with both curl's ``--max-time`` and a
    parent-process timeout. A trickling peer therefore cannot hold a training rank indefinitely.
    Enqueue is attempted once because the service exposes no idempotency key; only pull retries.
    """

    def __init__(self, config: VideoRewardConfig) -> None:
        self.config = config
        self.enqueue_url = os.environ.get("REMOTE_REWARD_ENQUEUE_URL", "").strip()
        self.fetch_url = os.environ.get("REMOTE_REWARD_FETCH_URL", "").strip()
        self.token = os.environ.get("REMOTE_REWARD_TOKEN", "").strip()
        missing = [
            name
            for name, value in (
                ("REMOTE_REWARD_ENQUEUE_URL", self.enqueue_url),
                ("REMOTE_REWARD_FETCH_URL", self.fetch_url),
                ("REMOTE_REWARD_TOKEN", self.token),
            )
            if not value
        ]
        if missing:
            raise ValueError(f"video reward client needs environment variable(s): {', '.join(missing)}")
        for name, url in (("enqueue", self.enqueue_url), ("fetch", self.fetch_url)):
            parsed = urllib.parse.urlparse(url)
            if parsed.scheme not in ("http", "https") or not parsed.netloc:
                raise ValueError(f"video reward {name} URL must be absolute HTTP(S), got {url!r}")
        _curl_config_value(self.token)
        self.curl = shutil.which("curl")
        if self.curl is None:
            raise RuntimeError("video reward hard-deadline transport requires curl on PATH")
        self.poll_attempts = 0
        self.transport_retries = 0

    @staticmethod
    def _remaining(deadline_monotonic: float) -> float:
        remaining = deadline_monotonic - time.monotonic()
        if remaining <= 0.0:
            raise VideoRewardError("timeout", "video reward absolute result deadline expired")
        return remaining

    def _post(
        self,
        url: str,
        *,
        payload: bytes | Path,
        content_type: str,
        deadline_monotonic: float,
        headers: dict[str, str] | None = None,
    ) -> _HTTPResponse:
        remaining = self._remaining(deadline_monotonic)
        operation_timeout = min(self.config.operation_timeout_s, remaining)
        connect_timeout = min(self.config.connect_timeout_s, operation_timeout)
        request_headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": content_type,
            **({} if headers is None else headers),
        }
        with tempfile.TemporaryDirectory(prefix="posttrain-video-reward-") as tmp_dir:
            root = Path(tmp_dir)
            if isinstance(payload, bytes):
                payload_path = root / "payload.bin"
                _write_private_file(payload_path, payload)
            else:
                payload_path = payload
            config_path = root / "curl.conf"
            lines = [
                "silent",
                "show-error",
                'request = "POST"',
                f'url = "{_curl_config_value(url)}"',
                f'connect-timeout = "{connect_timeout:.9f}"',
                f'max-time = "{operation_timeout:.9f}"',
                f'upload-file = "{_curl_config_value(str(payload_path))}"',
                'header = "Expect:"',
                'write-out = "\\n%{http_code}"',
            ]
            lines.extend(
                f'header = "{_curl_config_value(name)}: {_curl_config_value(value)}"'
                for name, value in request_headers.items()
            )
            _write_private_file(config_path, ("\n".join(lines) + "\n").encode("utf-8"))
            try:
                try:
                    completed = subprocess.run(
                        [self.curl, "--config", str(config_path)],
                        check=False,
                        capture_output=True,
                        timeout=operation_timeout,
                    )
                except subprocess.TimeoutExpired as error:
                    raise VideoRewardError(
                        "timeout",
                        f"video reward HTTP operation exceeded {operation_timeout:.3f}s hard deadline",
                    ) from error
            finally:
                _scrub_private_file(config_path)
        if completed.returncode != 0:
            detail = completed.stderr.decode("utf-8", errors="replace").strip()
            category: VideoRewardFailureCategory = "timeout" if completed.returncode == 28 else "transport"
            raise VideoRewardError(
                category,
                f"video reward HTTP transport exited {completed.returncode}: {detail or 'no diagnostic'}",
            )
        body, separator, status = completed.stdout.rpartition(b"\n")
        if not separator or len(status) != 3 or not status.isdigit():
            raise VideoRewardError("protocol", "video reward curl response omitted an HTTP status code")
        return _HTTPResponse(status_code=int(status), body=body)

    @staticmethod
    def _pending(response: _HTTPResponse, *, service_id: str) -> bool:
        if response.status_code not in (202, 400):
            return False
        try:
            value = json.loads(response.body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return False
        if not isinstance(value, dict) or "scores" in value:
            return False
        if response.status_code == 202:
            return value.get("uuid") == service_id and value.get("status") == "pending"
        return value.get("message") == f"Not found scores for {service_id}"

    def _pull(self, *, service_id: str, replica_id: str, deadline_monotonic: float) -> _HTTPResponse:
        payload = urllib.parse.urlencode({"uuid": service_id, "type": self.config.provider_name}).encode("utf-8")
        headers = {}
        if replica_id and os.environ.get("COSMOS_DISABLE_REMOTE_REWARD_USE_REPLICA", "0") != "1":
            headers["X-Lepton-Replica-Target"] = replica_id
        last_error: VideoRewardError | None = None
        for attempt in range(self.config.pull_max_attempts):
            try:
                response = self._post(
                    self.fetch_url,
                    payload=payload,
                    content_type="application/x-www-form-urlencoded",
                    deadline_monotonic=deadline_monotonic,
                    headers=headers,
                )
            except VideoRewardError as error:
                if error.category not in ("timeout", "transport"):
                    raise
                last_error = error
            else:
                if response.status_code not in (408, 429) and response.status_code < 500:
                    return response
                last_error = VideoRewardError(
                    "transport",
                    f"video reward pull returned retryable HTTP {response.status_code}",
                )
            if attempt + 1 == self.config.pull_max_attempts:
                break
            backoff = self.config.poll_interval_s * (2 ** min(attempt, 3))
            time.sleep(min(backoff, self._remaining(deadline_monotonic)))
            self._remaining(deadline_monotonic)
            self.transport_retries += 1
        if last_error is None:
            raise RuntimeError("video reward pull retries ended without a response or error")
        raise last_error

    def _score_payload(
        self,
        *,
        metadata: dict[str, object],
        media: np.ndarray,
        deadline_monotonic: float,
        media_type: str,
    ) -> VideoRewardResult:
        """Submit and poll one decoded-media payload through the shared bounded state machine."""
        self.poll_attempts = 0
        self.transport_retries = 0
        with tempfile.TemporaryDirectory(
            prefix=f"posttrain-{media_type}-reward-payload-",
            dir=self.config.spool_dir,
        ) as tmp_dir:
            payload_path = Path(tmp_dir) / "payload.bin"
            descriptor = os.open(payload_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            with os.fdopen(descriptor, "wb") as output:
                output.write(json.dumps(metadata, sort_keys=True).encode("utf-8") + b"\n")
                np.save(output, media, allow_pickle=False)
            enqueue = self._post(
                self.enqueue_url,
                payload=payload_path,
                content_type="application/octet-stream",
                deadline_monotonic=deadline_monotonic,
            )
        if enqueue.status_code != 200:
            raise VideoRewardError("protocol", f"{media_type} reward enqueue returned HTTP {enqueue.status_code}")
        enqueue_body = enqueue.json_object("enqueue")
        service_id = enqueue_body.get("uuid")
        replica_id = enqueue_body.get("replica_id", "")
        if not isinstance(service_id, str) or not service_id:
            raise VideoRewardError("protocol", f"{media_type} reward enqueue response has no non-empty uuid")
        if not isinstance(replica_id, str):
            raise VideoRewardError(
                "protocol",
                f"{media_type} reward enqueue replica_id must be a string when present",
            )

        while True:
            self.poll_attempts += 1
            try:
                pull = self._pull(
                    service_id=service_id,
                    replica_id=replica_id,
                    deadline_monotonic=deadline_monotonic,
                )
            except VideoRewardError as error:
                if error.category not in ("timeout", "transport"):
                    raise
                time.sleep(min(self.config.poll_interval_s, self._remaining(deadline_monotonic)))
                self._remaining(deadline_monotonic)
                self.transport_retries += 1
                continue
            if pull.status_code == 200:
                result = pull.json_object("pull")
                scores = result.get("scores")
                values = scores.get(self.config.score_key) if isinstance(scores, dict) else None
                if not isinstance(values, list) or len(values) != 1:
                    raise VideoRewardError(
                        "protocol",
                        f"{media_type} reward score {self.config.score_key!r} must contain exactly one positional value",
                    )
                score = values[0]
                if isinstance(score, bool) or not isinstance(score, (int, float)):
                    raise VideoRewardError(
                        "protocol",
                        f"{media_type} reward score must be numeric and non-bool, got {score!r}",
                    )
                normalized = float(score)
                if not isfinite(normalized):
                    raise VideoRewardError(
                        "nonfinite",
                        f"{media_type} reward score is non-finite: {normalized!r}",
                    )
                self._remaining(deadline_monotonic)
                return VideoRewardResult(
                    score=normalized,
                    service_id=service_id,
                )
            if not self._pending(pull, service_id=service_id):
                raise VideoRewardError("protocol", f"{media_type} reward pull returned HTTP {pull.status_code}")
            time.sleep(min(self.config.poll_interval_s, self._remaining(deadline_monotonic)))

    def score(self, request: VideoRewardRequest, *, deadline_monotonic: float) -> VideoRewardResult:
        if request.video.dtype is not torch.uint8 or request.video.device.type != "cpu":
            raise VideoRewardError("protocol", "video reward wire tensor must be CPU uint8")
        if request.video.ndim != 5 or request.video.shape[0] != 1 or request.video.shape[1] != 3:
            raise VideoRewardError(
                "protocol",
                f"video reward wire tensor must be [1,3,T,H,W], got {tuple(request.video.shape)}",
            )
        metadata = {
            "media_type": "video",
            "prompts": [request.prompt],
            "reward_fn": {self.config.provider_name: 1.0},
            "video_infos": [{"video_fps": request.video_fps}],
        }
        return self._score_payload(
            metadata=metadata,
            media=request.video.numpy(),
            deadline_monotonic=deadline_monotonic,
            media_type="video",
        )


def _request_id(
    run_id: str,
    provider_name: str,
    trajectory: Trajectory,
    seed: int,
    weight_version: int,
) -> str:
    identity = f"{run_id}\0{provider_name}\0{trajectory.group_id}\0{trajectory.id}\0{seed}\0{weight_version}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


class VideoRewardProvider:
    """Decode and score one rank-local trajectory through a bounded external service."""

    scope: str = "trajectory"
    mode: str = "sync"

    def __init__(
        self,
        config: VideoRewardConfig,
        client: VideoRewardClient,
        decoder: VideoDecoder,
        *,
        run_id: str,
    ) -> None:
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("video reward run_id must be a non-empty string")
        if not callable(decoder):
            raise TypeError("video reward decoder must be callable")
        if getattr(decoder, "output_range", None) != "neg_one_to_one":
            raise ValueError(
                "video reward decoder must attest output_range='neg_one_to_one'; "
                f"got {getattr(decoder, 'output_range', None)!r}"
            )
        self.config = config
        self.client = client
        self.decoder = decoder
        self.run_id = run_id

    def _request_inputs(
        self,
        trajectory: Trajectory,
    ) -> tuple[WFMConditioningInput, torch.Tensor, int, int]:
        """Validate one trajectory before the modality-specific wire conversion."""
        if len(trajectory.transitions) != 1:
            raise VideoRewardError("protocol", "video reward trajectory must contain exactly one transition")
        transition = trajectory.transitions[0]
        conditioning = transition.obs.representations.get(WFM_CONDITIONING_KEY)
        if not isinstance(conditioning, WFMConditioningInput):
            raise VideoRewardError("protocol", "video reward trajectory has no WFMConditioningInput")
        latent = transition.action
        if not isinstance(latent, torch.Tensor):
            raise VideoRewardError("protocol", "video reward trajectory action must be a tensor")
        if latent.ndim != 5 or latent.shape[0] != 1:
            raise VideoRewardError(
                "protocol",
                f"video reward latent must be [1,C,T,H,W], got {tuple(latent.shape)}",
            )
        if not bool(torch.isfinite(latent).all()):
            raise VideoRewardError("nonfinite", "video reward latent contains a non-finite value")
        seed = transition.policy_info.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise VideoRewardError("protocol", f"video reward seed must be an integer, got {seed!r}")
        weight_version = transition.policy_info.get("weight_version")
        if isinstance(weight_version, bool) or not isinstance(weight_version, int):
            raise VideoRewardError(
                "protocol",
                f"video reward weight_version must be an integer, got {weight_version!r}",
            )
        return conditioning, latent, seed, weight_version

    def _decode(self, latent: torch.Tensor, *, enforce_output_range: bool = True) -> torch.Tensor:
        """Decode and validate the model-owned floating media contract."""
        try:
            with torch.no_grad():
                decoded = self.decoder([latent])  # [1,3,T,H,W]
        except VideoRewardError:
            raise
        except Exception as error:  # noqa: BLE001 -- normalize the injected model boundary
            raise VideoRewardError("protocol", f"video reward decoder failed: {error}") from error
        if not isinstance(decoded, torch.Tensor) or decoded.ndim != 5:
            raise VideoRewardError("protocol", "video reward decoder must return [1,3,T,H,W]")
        if decoded.shape[0] != 1 or decoded.shape[1] != 3:
            raise VideoRewardError(
                "protocol",
                f"video reward decoder must return [1,3,T,H,W], got {tuple(decoded.shape)}",
            )
        if not decoded.is_floating_point():
            raise VideoRewardError(
                "protocol",
                f"video reward decoder must return a floating [-1,1] tensor, got {decoded.dtype}",
            )
        if not bool(torch.isfinite(decoded).all()):
            raise VideoRewardError("nonfinite", "video reward decoded video contains a non-finite value")
        if enforce_output_range:
            decoded_min = decoded.amin()  # []
            decoded_max = decoded.amax()  # []
            if float(decoded_min) < -1.000001 or float(decoded_max) > 1.000001:
                raise VideoRewardError(
                    "protocol",
                    "video reward decoder violated its attested [-1,1] range: "
                    f"min={float(decoded_min):.6g}, max={float(decoded_max):.6g}",
                )
        return decoded

    def _request(self, trajectory: Trajectory) -> VideoRewardRequest:
        conditioning, latent, seed, weight_version = self._request_inputs(trajectory)
        if conditioning.modality != "video":
            raise VideoRewardError(
                "protocol",
                f"video reward provider requires video conditioning, got {conditioning.modality!r}",
            )
        decoded = self._decode(latent)
        video = (decoded.detach().to(device="cpu", dtype=torch.float32).clamp(-1.0, 1.0) + 1.0) * 127.5
        video = video.round().to(dtype=torch.uint8).contiguous()  # [1,3,T,H,W]
        return VideoRewardRequest(
            request_id=_request_id(
                self.run_id,
                self.config.provider_name,
                trajectory,
                seed,
                weight_version,
            ),
            sample_id=trajectory.id,
            prompt=conditioning.prompt,
            video=video,
            seed=seed,
            weight_version=weight_version,
            video_fps=float(self.config.video_fps),
        )

    def evaluate(self, trajectory: Trajectory) -> RewardProviderResult:
        started = time.monotonic()
        deadline = started + self.config.total_deadline_s
        request = self._request(trajectory)
        if time.monotonic() >= deadline:
            raise VideoRewardError(
                "timeout",
                f"video reward evaluation exceeded its {self.config.total_deadline_s:.3f}s absolute deadline",
            )
        try:
            result = self.client.score(
                request,
                deadline_monotonic=deadline,
            )
        except VideoRewardError:
            raise
        except Exception as error:  # noqa: BLE001 -- normalize the injected transport boundary
            raise VideoRewardError("transport", f"video reward client failed: {error}") from error
        if not isinstance(result, VideoRewardResult):
            raise VideoRewardError(
                "protocol",
                f"video reward client returned {type(result).__name__}, expected VideoRewardResult",
            )
        if time.monotonic() >= deadline:
            raise VideoRewardError(
                "timeout",
                f"video reward evaluation exceeded its {self.config.total_deadline_s:.3f}s absolute deadline",
            )
        if not isinstance(result.service_id, str) or not result.service_id:
            raise VideoRewardError("protocol", "video reward result must carry a non-empty service_id")
        if isinstance(result.score, bool) or not isinstance(result.score, Real):
            raise VideoRewardError(
                "protocol",
                f"video reward result score must be numeric and non-bool, got {result.score!r}",
            )
        normalized = float(result.score)
        if not isfinite(normalized):
            raise VideoRewardError("nonfinite", f"video reward result score is non-finite: {normalized!r}")
        raw = torch.tensor(normalized, dtype=torch.float32)  # []
        clipped = torch.clamp(  # []
            raw,
            min=self.config.reward_clip_min,
            max=self.config.reward_clip_max,
        )
        transformed = clipped * self.config.scale  # []
        if not bool(torch.isfinite(transformed)):
            raise VideoRewardError("nonfinite", "video reward clipping/scaling produced a non-finite value")
        for name in ("poll_attempts", "transport_retries"):
            value = getattr(self.client, name, None)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise VideoRewardError(
                    "protocol",
                    f"video reward client {name} must be a non-negative integer, got {value!r}",
                )
        clipped_count = int(bool(raw != clipped))
        elapsed = time.monotonic() - started
        return RewardProviderResult(
            terms={"task": float(transformed)},
            diagnostics={
                "evaluation_latency_s": MetricContribution.weighted_mean(elapsed),
                "raw_score": MetricContribution.weighted_mean(float(raw)),
                "transformed_score": MetricContribution.weighted_mean(float(transformed)),
                "clipped_count": MetricContribution.sum(clipped_count),
                "poll_attempts": MetricContribution.sum(self.client.poll_attempts),
                "transport_retries": MetricContribution.sum(self.client.transport_retries),
                "clip_min": MetricContribution.identical(self.config.reward_clip_min),
                "clip_max": MetricContribution.identical(self.config.reward_clip_max),
                "scale": MetricContribution.identical(self.config.scale),
            },
        )


@dataclass(frozen=True)
class _PreparedVideoRewardGroupRequest(RewardGroupRequest):
    """Canonical group envelope plus the local trajectories needed for pointwise decoding."""

    trajectories: tuple[Trajectory, ...] = ()

    def __post_init__(self) -> None:
        super().__post_init__()
        if len(self.trajectories) != len(self.candidates) or any(
            not isinstance(trajectory, Trajectory) for trajectory in self.trajectories
        ):
            raise RewardContractError(
                "decoded-video reward request must retain one trajectory per candidate, "
                f"got {len(self.trajectories)} trajectories and {len(self.candidates)} candidates"
            )
        trajectory_ids = tuple(trajectory.id for trajectory in self.trajectories)
        candidate_ids = tuple(candidate.trajectory_id for candidate in self.candidates)
        if trajectory_ids != candidate_ids:
            raise RewardContractError(
                "decoded-video reward request trajectory identities must match its candidate envelope, "
                f"got trajectories={trajectory_ids!r}, candidates={candidate_ids!r}"
            )


class _VideoGroupProvider:
    """Adapt the specialized trajectory decoder to the canonical group result boundary."""

    def __init__(self, provider: VideoRewardProvider) -> None:
        self.provider = provider

    def evaluate(self, request: RewardGroupRequest) -> RewardGroupResult:
        if not isinstance(request, _PreparedVideoRewardGroupRequest):
            raise RewardContractError("decoded-video reward provider requires its prepared trajectory group request")
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


class _BoundVideoRewardProvider(BoundRewardProvider):
    """Bind latent-bearing WFM trajectories without widening the public group request schema."""

    def prepare_group(self, trajectories: Sequence[Trajectory]) -> RewardGroupRequest:
        group = tuple(trajectories)
        if not group:
            raise RewardContractError("decoded-video reward requires at least one trajectory")
        group_id: str | None = None
        for trajectory in group:
            if not isinstance(trajectory, Trajectory):
                raise RewardContractError(
                    f"decoded-video reward candidate must be a Trajectory, got {type(trajectory).__name__}"
                )
            if trajectory.status != Status.COMPLETED:
                raise RewardContractError(
                    f"decoded-video reward trajectory must be completed, got status={trajectory.status!r}"
                )
            if not trajectory.transitions or not trajectory.transitions[-1].done:
                raise RewardContractError("decoded-video reward trajectory must have a done terminal transition")
            if not isinstance(trajectory.id, str) or not trajectory.id.strip():
                raise RewardContractError(
                    f"decoded-video reward trajectory id must be a non-empty string, got {trajectory.id!r}"
                )
            if not isinstance(trajectory.group_id, str) or not trajectory.group_id.strip():
                raise RewardContractError(
                    f"decoded-video reward group_id must be a non-empty string, got {trajectory.group_id!r}"
                )
            if group_id is None:
                group_id = trajectory.group_id
            elif trajectory.group_id != group_id:
                raise RewardContractError(
                    f"decoded-video reward received mixed group_ids {group_id!r} and {trajectory.group_id!r}"
                )
            if trajectory.transitions[-1].obs.id != trajectory.group_id:
                raise RewardContractError("decoded-video reward terminal obs.id must match the trajectory group_id")
        if group_id is None:
            raise AssertionError("non-empty decoded-video group did not establish a group_id")
        return _PreparedVideoRewardGroupRequest(
            group_id=group_id,
            candidates=tuple(RewardCandidate(trajectory_id=trajectory.id, completion="") for trajectory in group),
            trajectories=group,
        )


def build_video_reward_provider(
    config: VideoRewardConfig,
    client: VideoRewardClient,
    decoder: VideoDecoder,
    *,
    run_id: str,
) -> BoundRewardProvider:
    """Bind the specialized adapter to the canonical catalog result/provenance contract."""
    video_provider = VideoRewardProvider(config, client, decoder, run_id=run_id)
    group_provider = _VideoGroupProvider(video_provider)
    spec = RewardProviderSpec(
        factory=lambda: group_provider,
        revision=config.provider_revision,
        diagnostic_names=DECODED_MEDIA_REWARD_DIAGNOSTICS,
        term_names=("task",),
        semantic_identity={
            "adapter": "decoded_video_v1",
            "provider_name": config.provider_name,
            "provider_revision": config.provider_revision,
            "score_key": config.score_key,
            "video_fps": config.video_fps,
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
    return _BoundVideoRewardProvider(config.provider_name, spec, spec.factory())


__all__ = [
    "HTTPVideoRewardClient",
    "DECODED_MEDIA_REWARD_DIAGNOSTICS",
    "VIDEO_REWARD_DIAGNOSTICS",
    "VideoRewardClient",
    "VideoRewardConfig",
    "VideoRewardError",
    "VideoRewardProvider",
    "VideoRewardRequest",
    "VideoRewardResult",
    "build_video_reward_provider",
]
