# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""CPU contracts for the rank-local decoded-video reward adapter."""

from __future__ import annotations

import io
import json
import stat
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from cosmos_framework.posttrain.algorithm.annotators.advantage import (
    GROUP_REWARD_MEAN,
    GROUP_REWARD_NONZERO_FRACTION,
    GROUP_REWARD_SPREAD,
    distributed_group_relative,
)
from cosmos_framework.posttrain.config.rl import VideoRewardConfig
from cosmos_framework.posttrain.roles.reward import RewardWorker
from cosmos_framework.posttrain.roles.reward.video import (
    HTTPVideoRewardClient,
    VideoRewardError,
    VideoRewardProvider,
    VideoRewardRequest,
    VideoRewardResult,
    _HTTPResponse,
    build_video_reward_provider,
)
from cosmos_framework.posttrain.schema import (
    WFM_CONDITIONING_KEY,
    ObsBundle,
    Status,
    Trajectory,
    Transition,
    WFMConditioningInput,
)
from cosmos_framework.posttrain.schema.metrics import finalize_metric_report
from cosmos_framework.posttrain.tests.helpers.distributed import init_gloo_process_group
from cosmos_framework.posttrain.utils.dist import raise_if_any_rank_failed


def _config(**overrides: object) -> VideoRewardConfig:
    values: dict[str, object] = {
        "provider_revision": "dance-grpo-2026-08",
        "poll_interval_s": 0.001,
    }
    values.update(overrides)
    return VideoRewardConfig(**values)


def _trajectory(rank: int = 0) -> Trajectory:
    conditioning = WFMConditioningInput(
        id="prompt-a",
        template_identity="reward-test-template-a",
        model_batch={"conditioning": torch.ones(1)},
        prompt="a dancer",
    )
    transition = Transition(
        obs=ObsBundle(
            representations={WFM_CONDITIONING_KEY: conditioning},
            id=conditioning.id,
        ),
        action=torch.full((1, 4, 2, 2, 2), float(rank), requires_grad=True),  # [1,C,T,H,W]
        policy_info={"seed": 10 + rank, "weight_version": 3},
    )
    return Trajectory(
        transitions=[transition],
        id=f"sample-r{rank}",
        group_id=conditioning.id,
        status=Status.COMPLETED,
    )


class _MeanDecoder:
    output_range: str = "neg_one_to_one"

    def __call__(self, latents: list[torch.Tensor]) -> torch.Tensor:
        assert not torch.is_grad_enabled()
        assert len(latents) == 1
        value = latents[0].mean()
        return value.expand(1, 3, 2, 2, 2).clone()  # [1,3,T,H,W]


class _PixelOracleDecoder:
    output_range: str = "neg_one_to_one"

    def __call__(self, latents: list[torch.Tensor]) -> torch.Tensor:
        assert not torch.is_grad_enabled()
        assert len(latents) == 1
        values = torch.tensor([-1.0, 0.0, 1.0]).reshape(1, 1, 1, 1, 3)  # [1,1,T,H,W]
        return values.expand(1, 3, 1, 1, 3).clone()  # [1,3,T,H,W]


_decoder = _MeanDecoder()
_pixel_oracle_decoder = _PixelOracleDecoder()


class _SuccessClient:
    poll_attempts: int = 2
    transport_retries: int = 1

    def __init__(self, score: float) -> None:
        self.score_value = score
        self.requests: list[VideoRewardRequest] = []

    def score(self, request: VideoRewardRequest, *, deadline_monotonic: float) -> VideoRewardResult:
        assert deadline_monotonic > time.monotonic()
        self.requests.append(request)
        return VideoRewardResult(
            score=self.score_value,
            service_id="service-1",
        )


class _FailureClient(_SuccessClient):
    def score(self, request: VideoRewardRequest, *, deadline_monotonic: float) -> VideoRewardResult:
        del request, deadline_monotonic
        raise VideoRewardError("transport", "service unavailable")


@pytest.mark.L0
@pytest.mark.CPU
def test_video_reward_provider_returns_bound_result_and_typed_diagnostics() -> None:
    client = _SuccessClient(10.0)
    provider = build_video_reward_provider(
        _config(reward_clip_min=-1.0, reward_clip_max=1.0, scale=2.0),
        client,
        _pixel_oracle_decoder,
        run_id="run-a",
    )
    worker = RewardWorker([provider], training_group_size=1)
    trajectory = _trajectory()

    assert provider.is_remote is True
    assert provider.semantic_identity["adapter"] == "decoded_video_v1"
    assert provider.semantic_identity["provider_revision"] == "dance-grpo-2026-08"
    worker.score_with_provider_report([trajectory])
    report = worker.provider_report()
    metrics = finalize_metric_report(report.metrics)

    assert trajectory.final_reward == 2.0
    assert report.provenance == (("dance_grpo", "dance-grpo-2026-08"),)
    assert metrics["train/reward_provider/dance_grpo/sample_count"] == 1
    assert metrics["train/reward_provider/dance_grpo/raw_score"] == 10.0
    assert metrics["train/reward_provider/dance_grpo/transformed_score"] == 2.0
    assert metrics["train/reward_provider/dance_grpo/clipped_count"] == 1
    assert "train/reward_provider/dance_grpo/evaluation_latency_s" in metrics
    assert "train/reward_provider/dance_grpo/request_latency_s" not in metrics
    assert metrics["train/reward_provider/dance_grpo/poll_attempts"] == 2
    assert metrics["train/reward_provider/dance_grpo/transport_retries"] == 1
    [request] = client.requests
    assert request.video.dtype is torch.uint8
    assert request.video.device.type == "cpu"
    assert tuple(request.video.shape) == (1, 3, 1, 1, 3)
    assert request.video[0, 0, 0, 0].tolist() == [0, 128, 255]
    assert request.video_fps == 16.0


@pytest.mark.L0
@pytest.mark.CPU
def test_video_reward_failure_leaves_singleton_worker_input_unmodified() -> None:
    trajectory = _trajectory()
    provider = build_video_reward_provider(
        _config(),
        _FailureClient(0.0),
        _decoder,
        run_id="run-a",
    )
    worker = RewardWorker([provider], training_group_size=1)

    with pytest.raises(VideoRewardError, match="service unavailable"):
        worker.score_with_provider_report([trajectory])

    assert trajectory.transitions[-1].reward is None
    assert trajectory.transitions[-1].advantage is None
    with pytest.raises(RuntimeError, match="score_with_provider_report must succeed"):
        worker.provider_report()

    class _UnattestedDecoder:
        def __call__(self, _latents: list[torch.Tensor]) -> torch.Tensor:
            return torch.zeros((1, 3, 1, 1, 1))  # [1,3,T,H,W]

    with pytest.raises(ValueError, match="must attest"):
        VideoRewardProvider(_config(), _SuccessClient(0.0), _UnattestedDecoder(), run_id="run-a")

    class _IntegerDecoder:
        output_range: str = "neg_one_to_one"

        def __call__(self, _latents: list[torch.Tensor]) -> torch.Tensor:
            return torch.zeros((1, 3, 1, 1, 1), dtype=torch.uint8)  # [1,3,T,H,W]

    with pytest.raises(VideoRewardError, match="must return a floating"):
        VideoRewardProvider(_config(), _SuccessClient(0.0), _IntegerDecoder(), run_id="run-a").evaluate(_trajectory())

    class _OutOfRangeDecoder:
        output_range: str = "neg_one_to_one"

        def __call__(self, _latents: list[torch.Tensor]) -> torch.Tensor:
            return torch.full((1, 3, 1, 1, 1), 2.0)  # [1,3,T,H,W]

    with pytest.raises(VideoRewardError, match="violated its attested"):
        VideoRewardProvider(_config(), _SuccessClient(0.0), _OutOfRangeDecoder(), run_id="run-a").evaluate(
            _trajectory()
        )

    class _UnexpectedClient(_SuccessClient):
        def score(self, request: VideoRewardRequest, *, deadline_monotonic: float) -> VideoRewardResult:
            del request, deadline_monotonic
            raise ValueError("unexpected client bug")

    with pytest.raises(VideoRewardError, match="client failed") as client_error:
        VideoRewardProvider(_config(), _UnexpectedClient(0.0), _decoder, run_id="run-a").evaluate(_trajectory())
    assert client_error.value.category == "transport"


def _http_client(monkeypatch: pytest.MonkeyPatch, **overrides: object) -> HTTPVideoRewardClient:
    monkeypatch.setenv("REMOTE_REWARD_ENQUEUE_URL", "https://reward/enqueue")
    monkeypatch.setenv("REMOTE_REWARD_FETCH_URL", "https://reward/pull")
    monkeypatch.setenv("REMOTE_REWARD_TOKEN", "secret")
    return HTTPVideoRewardClient(_config(**overrides))


def _wire_request() -> VideoRewardRequest:
    return VideoRewardRequest(
        request_id="request-1",
        sample_id="sample-1",
        prompt="a dancer",
        video=torch.arange(24, dtype=torch.uint8).reshape(1, 3, 2, 2, 2),  # [1,3,T,H,W]
        seed=11,
        weight_version=3,
        video_fps=24.0,
    )


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.serial
def test_http_video_reward_post_success_recovers_exact_status_and_body(monkeypatch: pytest.MonkeyPatch) -> None:
    received: list[tuple[bytes, str | None, str | None]] = []

    class _SuccessHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers["Content-Length"])
            received.append(
                (
                    self.rfile.read(length),
                    self.headers.get("Expect"),
                    self.headers.get("Authorization"),
                )
            )
            body = b'{"uuid":"service-1"}'
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *_args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), _SuccessHandler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    client = _http_client(monkeypatch)
    try:
        response = client._post(
            f"http://{host}:{port}/enqueue",
            payload=b"exact-wire-body",
            content_type="application/octet-stream",
            deadline_monotonic=time.monotonic() + 2.0,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1.0)

    assert response == _HTTPResponse(200, b'{"uuid":"service-1"}')
    assert received == [(b"exact-wire-body", None, "Bearer secret")]


@pytest.mark.L0
@pytest.mark.CPU
def test_http_video_reward_uses_private_scrubbed_streaming_curl_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _http_client(monkeypatch)
    config_paths: list[Path] = []
    config_modes: list[int] = []
    configs: list[str] = []
    argvs: list[list[str]] = []

    def complete(
        args: list[str],
        *,
        check: bool,
        capture_output: bool,
        timeout: float,
    ) -> subprocess.CompletedProcess[bytes]:
        del check, capture_output, timeout
        config_path = Path(args[2])
        config = config_path.read_text(encoding="utf-8")
        config_paths.append(config_path)
        config_modes.append(stat.S_IMODE(config_path.stat().st_mode))
        configs.append(config)
        argvs.append(list(args))
        return subprocess.CompletedProcess(args, 0, b'{"ok":true}\n200', b"")

    monkeypatch.setattr(subprocess, "run", complete)
    response = client._post(
        client.fetch_url,
        payload=b"uuid=service-1",
        content_type="application/x-www-form-urlencoded",
        deadline_monotonic=time.monotonic() + 2.0,
    )

    assert response == _HTTPResponse(200, b'{"ok":true}')
    assert config_modes == [0o600]
    assert len(configs) == 1
    assert 'upload-file = "' in configs[0]
    assert 'header = "Expect:"' in configs[0]
    assert "data-binary" not in configs[0]
    assert len(argvs) == 1
    assert all("secret" not in argument for argument in argvs[0])
    assert len(config_paths) == 1
    assert not config_paths[0].exists()


@pytest.mark.L0
@pytest.mark.CPU
def test_http_video_reward_serializes_declared_decoded_video_and_fps_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client = _http_client(monkeypatch, spool_dir=str(tmp_path))
    calls: list[str] = []
    payload_paths: list[Path] = []

    def post(
        url: str,
        *,
        payload: bytes | Path,
        content_type: str,
        deadline_monotonic: float,
        headers: dict[str, str] | None = None,
    ) -> _HTTPResponse:
        del deadline_monotonic
        calls.append(url)
        if url.endswith("enqueue"):
            assert content_type == "application/octet-stream"
            assert isinstance(payload, Path)
            assert payload.is_relative_to(tmp_path)
            payload_paths.append(payload)
            assert stat.S_IMODE(payload.stat().st_mode) == 0o600
            metadata_raw, media_raw = payload.read_bytes().split(b"\n", 1)
            metadata = json.loads(metadata_raw)
            assert metadata == {
                "media_type": "video",
                "prompts": ["a dancer"],
                "reward_fn": {"dance_grpo": 1.0},
                "video_infos": [{"video_fps": 24.0}],
            }
            media = np.load(io.BytesIO(media_raw), allow_pickle=False)  # [1,3,T,H,W]
            assert media.dtype == np.uint8
            assert np.array_equal(media, _wire_request().video.numpy())
            return _HTTPResponse(200, b'{"uuid":"service-1","replica_id":"replica-a"}')
        assert content_type == "application/x-www-form-urlencoded"
        assert payload == b"uuid=service-1&type=dance_grpo"
        assert headers == {"X-Lepton-Replica-Target": "replica-a"}
        return _HTTPResponse(200, b'{"scores":{"overall_reward":[0]}}')

    monkeypatch.setattr(client, "_post", post)
    result = client.score(_wire_request(), deadline_monotonic=time.monotonic() + 2.0)

    assert result.score == 0.0
    assert result.service_id == "service-1"
    assert not hasattr(result, "request_id")
    assert not hasattr(result, "sample_id")
    assert calls == ["https://reward/enqueue", "https://reward/pull"]
    assert len(payload_paths) == 1
    assert not payload_paths[0].exists()


@pytest.mark.L0
@pytest.mark.CPU
def test_http_video_reward_accepts_correlated_pending_then_exact_singleton_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _http_client(monkeypatch)
    responses = iter(
        (
            _HTTPResponse(200, b'{"uuid":"service-1"}'),
            _HTTPResponse(202, b'{"uuid":"service-1","status":"pending"}'),
            _HTTPResponse(400, b'{"message":"Not found scores for service-1"}'),
            _HTTPResponse(200, b'{"scores":{"overall_reward":[1.25]}}'),
        )
    )

    def post(*_args: object, **_kwargs: object) -> _HTTPResponse:
        return next(responses)

    monkeypatch.setattr(client, "_post", post)
    result = client.score(_wire_request(), deadline_monotonic=time.monotonic() + 2.0)

    assert result.score == 1.25
    assert client.poll_attempts == 3
    assert client.transport_retries == 0


@pytest.mark.L0
@pytest.mark.CPU
def test_http_video_reward_treats_permanent_correlated_400_as_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _http_client(monkeypatch)
    responses = iter(
        (
            _HTTPResponse(200, b'{"uuid":"service-1"}'),
            _HTTPResponse(400, b'{"uuid":"service-1","message":"invalid reward type"}'),
        )
    )
    sleeps: list[float] = []

    monkeypatch.setattr(client, "_post", lambda *_args, **_kwargs: next(responses))
    monkeypatch.setattr(time, "sleep", sleeps.append)
    with pytest.raises(VideoRewardError, match="pull returned HTTP 400") as error:
        client.score(_wire_request(), deadline_monotonic=time.monotonic() + 2.0)

    assert error.value.category == "protocol"
    assert client.poll_attempts == 1
    assert client.transport_retries == 0
    assert sleeps == []


@pytest.mark.L0
@pytest.mark.CPU
def test_http_video_reward_rejects_malformed_late_or_nonfinite_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _http_client(
        monkeypatch,
        connect_timeout_s=0.1,
        operation_timeout_s=0.25,
        total_deadline_s=0.5,
    )
    assert not client._pending(_HTTPResponse(202, b'{"status":"pending"}'), service_id="service-1")
    assert not client._pending(
        _HTTPResponse(400, b'{"message":"Not found scores for another-service"}'),
        service_id="service-1",
    )

    subprocess_timeouts: list[float] = []

    def parent_timeout(
        args: list[str],
        *,
        check: bool,
        capture_output: bool,
        timeout: float,
    ) -> subprocess.CompletedProcess[bytes]:
        del check, capture_output
        subprocess_timeouts.append(timeout)
        raise subprocess.TimeoutExpired(args, timeout)

    monkeypatch.setattr(subprocess, "run", parent_timeout)
    with pytest.raises(VideoRewardError, match="hard deadline") as timeout_error:
        client._post(
            "https://reward/pull",
            payload=b"uuid=service-1",
            content_type="application/x-www-form-urlencoded",
            deadline_monotonic=time.monotonic() + 0.5,
        )
    assert timeout_error.value.category == "timeout"
    assert subprocess_timeouts == pytest.approx([0.25])

    responses = iter(
        (
            _HTTPResponse(200, b'{"uuid":"service-1"}'),
            _HTTPResponse(200, b'{"scores":{"overall_reward":[1.0,2.0]}}'),
        )
    )
    monkeypatch.setattr(client, "_post", lambda *_args, **_kwargs: next(responses))
    with pytest.raises(VideoRewardError, match="exactly one positional value"):
        client.score(_wire_request(), deadline_monotonic=time.monotonic() + 2.0)

    nonfinite = _SuccessClient(float("nan"))
    with pytest.raises(VideoRewardError, match="non-finite"):
        VideoRewardProvider(_config(), nonfinite, _decoder, run_id="run-a").evaluate(_trajectory())
    with pytest.raises(VideoRewardError, match="numeric and non-bool"):
        VideoRewardProvider(_config(), _SuccessClient(True), _decoder, run_id="run-a").evaluate(_trajectory())

    class _LateClient(_SuccessClient):
        def score(self, request: VideoRewardRequest, *, deadline_monotonic: float) -> VideoRewardResult:
            del deadline_monotonic
            time.sleep(0.01)
            return VideoRewardResult(
                score=self.score_value,
                service_id="service-1",
            )

    with pytest.raises(VideoRewardError, match="evaluation exceeded"):
        VideoRewardProvider(
            _config(connect_timeout_s=0.001, operation_timeout_s=0.001, total_deadline_s=0.001),
            _LateClient(1.0),
            _decoder,
            run_id="run-a",
        ).evaluate(_trajectory())

    class _LateDecoder:
        output_range: str = "neg_one_to_one"

        def __call__(self, _latents: list[torch.Tensor]) -> torch.Tensor:
            time.sleep(0.01)
            return torch.zeros((1, 3, 1, 1, 1))  # [1,3,T,H,W]

    with pytest.raises(VideoRewardError, match="evaluation exceeded"):
        VideoRewardProvider(
            _config(connect_timeout_s=0.001, operation_timeout_s=0.001, total_deadline_s=0.001),
            _SuccessClient(1.0),
            _LateDecoder(),
            run_id="run-a",
        ).evaluate(_trajectory())


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.serial
def test_http_video_reward_hard_bounds_a_blocking_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    class _BlockingHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            time.sleep(0.5)
            try:
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"{}")
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, _format: str, *_args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), _BlockingHandler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    monkeypatch.setenv("REMOTE_REWARD_ENQUEUE_URL", f"http://{host}:{port}/enqueue")
    monkeypatch.setenv("REMOTE_REWARD_FETCH_URL", f"http://{host}:{port}/pull")
    monkeypatch.setenv("REMOTE_REWARD_TOKEN", "secret")
    client = HTTPVideoRewardClient(_config(connect_timeout_s=0.05, operation_timeout_s=0.05, total_deadline_s=0.1))
    started = time.monotonic()
    try:
        with pytest.raises(VideoRewardError) as error:
            client._post(
                client.fetch_url,
                payload=b"uuid=service-1",
                content_type="application/x-www-form-urlencoded",
                deadline_monotonic=started + 0.1,
            )
        elapsed = time.monotonic() - started
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1.0)

    assert error.value.category == "timeout"
    assert elapsed < 0.4


@pytest.mark.L0
@pytest.mark.CPU
def test_http_video_reward_never_retries_ambiguous_enqueue(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _http_client(monkeypatch, pull_max_attempts=5)
    calls = 0

    def post(*_args: object, **_kwargs: object) -> _HTTPResponse:
        nonlocal calls
        calls += 1
        raise VideoRewardError("transport", "connection dropped after write")

    monkeypatch.setattr(client, "_post", post)
    with pytest.raises(VideoRewardError, match="connection dropped"):
        client.score(_wire_request(), deadline_monotonic=time.monotonic() + 2.0)
    assert calls == 1
    assert client.transport_retries == 0


@pytest.mark.L0
@pytest.mark.CPU
def test_http_video_reward_retries_only_idempotent_pull_within_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _http_client(monkeypatch, pull_max_attempts=3)
    calls = 0

    def post(*_args: object, **_kwargs: object) -> _HTTPResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            return _HTTPResponse(200, b'{"uuid":"service-1"}')
        if calls == 2:
            return _HTTPResponse(503, b"rolling replica")
        if calls == 3:
            return _HTTPResponse(429, b"busy")
        return _HTTPResponse(200, b'{"scores":{"overall_reward":[0.5]}}')

    monkeypatch.setattr(client, "_post", post)
    result = client.score(_wire_request(), deadline_monotonic=time.monotonic() + 2.0)

    assert result.score == 0.5
    assert calls == 4
    assert client.poll_attempts == 1
    assert client.transport_retries == 2


@pytest.mark.L0
@pytest.mark.CPU
def test_http_video_reward_retryable_status_error_is_single_wrapped_and_counted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _http_client(monkeypatch, pull_max_attempts=6)
    calls = 0
    sleeps: list[float] = []

    def unavailable(*_args: object, **_kwargs: object) -> _HTTPResponse:
        nonlocal calls
        calls += 1
        return _HTTPResponse(503, b"rolling replica")

    monkeypatch.setattr(client, "_post", unavailable)
    monkeypatch.setattr(time, "sleep", sleeps.append)
    with pytest.raises(VideoRewardError) as error:
        client._pull(service_id="service-1", replica_id="", deadline_monotonic=time.monotonic() + 2.0)

    assert error.value.category == "transport"
    assert str(error.value) == "video reward pull returned retryable HTTP 503"
    assert calls == 6
    assert client.transport_retries == 5
    assert sleeps == pytest.approx([0.001, 0.002, 0.004, 0.008, 0.008])


@pytest.mark.L0
@pytest.mark.CPU
def test_http_video_reward_transient_pull_failures_use_the_absolute_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _http_client(monkeypatch, pull_max_attempts=2, poll_interval_s=0.001)
    clock = 0.0
    calls = 0

    def monotonic() -> float:
        return clock

    def sleep(duration: float) -> None:
        nonlocal clock
        clock += duration

    def post(*_args: object, **_kwargs: object) -> _HTTPResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            return _HTTPResponse(200, b'{"uuid":"service-1"}')
        return _HTTPResponse(503, b"rolling replica")

    monkeypatch.setattr(time, "monotonic", monotonic)
    monkeypatch.setattr(time, "sleep", sleep)
    monkeypatch.setattr(client, "_post", post)
    with pytest.raises(VideoRewardError, match="absolute result deadline expired") as error:
        client.score(_wire_request(), deadline_monotonic=0.01)

    assert error.value.category == "timeout"
    assert clock == pytest.approx(0.01)
    assert calls > client.config.pull_max_attempts + 1
    # One enqueue plus N pull attempts yields exactly N - 1 transport retries. The increments
    # between attempts inside a burst and between exhausted bursts represent disjoint transitions.
    assert client.transport_retries == calls - 2


def _distributed_video_reward_worker(
    rank: int,
    world_size: int,
    rendezvous_path: str,
    fail_rank_zero: bool,
) -> None:
    init_gloo_process_group(rank, world_size, rendezvous_path)
    try:
        client = _FailureClient(0.0) if fail_rank_zero and rank == 0 else _SuccessClient(float(2 * rank - 1))
        provider = build_video_reward_provider(_config(), client, _decoder, run_id="run-a")
        estimator = distributed_group_relative(group_size=world_size, correction=1)
        worker = RewardWorker(
            [provider],
            annotators=(estimator,),
            training_group_size=1,
            failure_convergence=raise_if_any_rank_failed,
        )
        trajectory = _trajectory(rank)
        if fail_rank_zero:
            with pytest.raises(Exception, match="reward provider and combiner planning"):
                worker.score_with_provider_report([trajectory])
            assert trajectory.transitions[-1].reward is None
            assert trajectory.transitions[-1].advantage is None
            return

        worker.score_with_provider_report([trajectory])
        report = worker.provider_report()
        expected = (2 * rank - 1) / (2.0**0.5 + 1e-4)
        assert trajectory.final_reward == float(2 * rank - 1)
        assert trajectory.transitions[-1].advantage == pytest.approx(expected)
        assert trajectory.transitions[-1].algo_extra[GROUP_REWARD_MEAN] == 0.0
        assert trajectory.transitions[-1].algo_extra[GROUP_REWARD_SPREAD] == 2.0
        assert trajectory.transitions[-1].algo_extra[GROUP_REWARD_NONZERO_FRACTION] == 1.0
        assert report.provenance == (("dance_grpo", "dance-grpo-2026-08"),)

        client.score_value = 0.0
        tied = _trajectory(rank)
        tied.id = f"tied-r{rank}"
        worker.score([tied])
        assert tied.final_reward == 0.0
        assert tied.transitions[-1].advantage == 0.0
        assert tied.transitions[-1].algo_extra[GROUP_REWARD_MEAN] == 0.0
        assert tied.transitions[-1].algo_extra[GROUP_REWARD_SPREAD] == 0.0
        assert tied.transitions[-1].algo_extra[GROUP_REWARD_NONZERO_FRACTION] == 0.0
    finally:
        dist.destroy_process_group()


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.serial
@pytest.mark.no_xdist
def test_distributed_video_reward_failure_converges_before_advantage(tmp_path: Path) -> None:
    rendezvous_path = str(tmp_path / "reward-failure")
    mp.spawn(
        _distributed_video_reward_worker,
        args=(2, rendezvous_path, True),
        nprocs=2,
        join=True,
    )


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.serial
@pytest.mark.no_xdist
def test_distributed_video_reward_success_composes_transform_and_advantage(tmp_path: Path) -> None:
    rendezvous_path = str(tmp_path / "reward-success")
    mp.spawn(
        _distributed_video_reward_worker,
        args=(2, rendezvous_path, False),
        nprocs=2,
        join=True,
    )
