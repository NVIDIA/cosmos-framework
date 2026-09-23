# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

from collections.abc import Sequence

import pytest
import torch
from torch import nn

import cosmos_framework.model.generator.reasoner_runtime as reasoner_runtime_module
from cosmos_framework.model.generator.reasoner_features import (
    ReasonerFeatureBatch,
    ReasonerFeatureIdentity,
    ReasonerFeatureRequest,
    ReasonerFeatureSignature,
    compute_reasoner_feature_fingerprint,
)
from cosmos_framework.model.generator.reasoner_runtime import (
    ReasonerFeatureRuntime,
    ReasonerRuntimeInvariantError,
    ReasonerRuntimeSpec,
)

pytestmark = [pytest.mark.level(0), pytest.mark.gpus(0)]


class _FakeAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head_dim = 2
        self.k_proj = nn.Linear(4, 4, bias=False)


class _FakeLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _FakeAttention()


class _FakeReasonerModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.include_und_pathway = True
        self.include_gen_pathway = False
        self.embed_tokens = nn.Embedding(32, 4)
        self.layers = nn.ModuleList([_FakeLayer(), _FakeLayer()])

    def reasoner_forward(self, **_kwargs: object) -> None:
        raise AssertionError("fake reasoner_forward should not execute in runtime unit tests")


class _FakeReasoner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _FakeReasonerModel()
        self.requires_grad_(False)
        self.eval()


def _identity() -> ReasonerFeatureIdentity:
    return ReasonerFeatureIdentity(
        reasoner="reasoner-digest",
        tokenizer="tokenizer-digest",
        framing="framing-digest",
    )


def _request(
    sample_key: str,
    token_ids: Sequence[int],
    *,
    identity: ReasonerFeatureIdentity,
    fingerprint: str | None = None,
) -> ReasonerFeatureRequest:
    tokens = torch.tensor(token_ids, dtype=torch.int64)
    positions = torch.arange(tokens.numel(), dtype=torch.int64)
    offsets = torch.tensor([0, tokens.numel()], dtype=torch.int64)
    if fingerprint is None:
        fingerprint = compute_reasoner_feature_fingerprint(
            tokens,
            positions,
            offsets,
            identity=identity,
        )
    return ReasonerFeatureRequest(
        sample_key=sample_key,
        token_ids=tokens,
        position_ids=positions,
        causal_offsets=offsets,
        fingerprint=fingerprint,
    )


def _runtime(*, max_requests: int = 4, max_total_tokens: int = 16) -> ReasonerFeatureRuntime:
    return ReasonerFeatureRuntime(
        _FakeReasoner(),
        identity=_identity(),
        max_requests=max_requests,
        max_total_tokens=max_total_tokens,
    )


def _features_for(requests: Sequence[ReasonerFeatureRequest]) -> ReasonerFeatureBatch:
    total_tokens = sum(request.token_ids.numel() for request in requests)
    offsets = [0]
    for request in requests:
        offsets.append(offsets[-1] + request.token_ids.numel())
    return ReasonerFeatureBatch(
        cross_k=tuple(torch.full((total_tokens, 2, 2), layer_idx, dtype=torch.float32) for layer_idx in range(2)),
        cross_v=tuple(torch.full((total_tokens, 2, 2), layer_idx + 10, dtype=torch.float32) for layer_idx in range(2)),
        causal_offsets=torch.tensor(offsets, dtype=torch.int64),
        fingerprints=tuple(request.fingerprint for request in requests),
    )


def test_runtime_exposes_reasoner_feature_signature() -> None:
    runtime = _runtime()

    assert runtime.signature == ReasonerFeatureSignature(
        num_layers=2,
        num_kv_heads=2,
        head_dim=2,
        dtype=torch.float32,
    )
    assert runtime.vocab_size == 32
    assert runtime.device == torch.device("cpu")


def test_runtime_rejects_missing_reasoner_forward_at_startup() -> None:
    reasoner = _FakeReasoner()
    reasoner.model.reasoner_forward = None  # type: ignore[method-assign]

    with pytest.raises(TypeError, match="model.reasoner_forward to be callable"):
        ReasonerFeatureRuntime(reasoner, identity=_identity())


def test_runtime_rejects_generator_specific_k_normalization_at_startup() -> None:
    reasoner = _FakeReasoner()
    reasoner.model.layers[1].self_attn.k_norm_und_for_gen = nn.Identity()

    with pytest.raises(NotImplementedError, match=r"affected layers=\[1\]"):
        ReasonerFeatureRuntime(reasoner, identity=_identity())


def test_load_validates_execution_compatibility_before_checkpoint_restore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reasoner = _FakeReasoner()
    reasoner.model.reasoner_forward = None  # type: ignore[method-assign]
    checkpoint_loader_called = False

    monkeypatch.setattr(reasoner_runtime_module, "build_reasoner", lambda *_args, **_kwargs: reasoner)

    def checkpoint_loader(*_args: object, **_kwargs: object) -> None:
        nonlocal checkpoint_loader_called
        checkpoint_loader_called = True

    monkeypatch.setattr(reasoner_runtime_module, "load_reasoner_only_dcp", checkpoint_loader)
    spec = ReasonerRuntimeSpec(
        checkpoint="unused",
        checkpoint_source="regular",
        device="cpu",
        dtype=torch.float32,
        identity=_identity(),
    )

    with pytest.raises(TypeError, match="model.reasoner_forward to be callable"):
        ReasonerFeatureRuntime.load(object(), spec)
    assert not checkpoint_loader_called


def test_fingerprint_mismatch_fails_before_model_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = _runtime()
    request = _request("sample", [1, 2, 3], identity=runtime.identity, fingerprint="tampered")
    extractor_called = False

    def fail_if_called(*_args: object, **_kwargs: object) -> ReasonerFeatureBatch:
        nonlocal extractor_called
        extractor_called = True
        raise AssertionError("extractor must not run for an invalid fingerprint")

    monkeypatch.setattr(reasoner_runtime_module, "extract_reasoner_feature_batch", fail_if_called)

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        runtime.execute([request])
    assert not extractor_called


def test_token_budget_fails_before_model_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = _runtime(max_total_tokens=2)
    request = _request("too-long", [1, 2, 3], identity=runtime.identity)
    extractor_called = False

    def fail_if_called(*_args: object, **_kwargs: object) -> ReasonerFeatureBatch:
        nonlocal extractor_called
        extractor_called = True
        raise AssertionError("extractor must not run above the token budget")

    monkeypatch.setattr(reasoner_runtime_module, "extract_reasoner_feature_batch", fail_if_called)

    with pytest.raises(ValueError, match=r"token count 3 exceeds limit 2"):
        runtime.execute([request])
    assert not extractor_called


def test_estimated_response_bytes_uses_signature_and_total_tokens() -> None:
    runtime = _runtime()
    requests = (
        _request("first", [1, 2, 3], identity=runtime.identity),
        _request("second", [4, 5], identity=runtime.identity),
    )

    expected = 5 * 2 * 2 * 2 * 2 * torch.tensor([], dtype=torch.float32).element_size()
    assert runtime.estimated_response_bytes(requests) == expected


def test_execute_preserves_request_order(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = _runtime()
    requests = (
        _request("second", [4, 5], identity=runtime.identity),
        _request("first", [1, 2, 3], identity=runtime.identity),
    )
    seen_keys: list[str] = []

    def extract(
        reasoner: nn.Module,
        received: Sequence[ReasonerFeatureRequest],
    ) -> ReasonerFeatureBatch:
        assert reasoner is runtime.reasoner
        seen_keys.extend(request.sample_key for request in received)
        return _features_for(received)

    monkeypatch.setattr(reasoner_runtime_module, "extract_reasoner_feature_batch", extract)

    features = runtime.execute(requests)

    assert seen_keys == ["second", "first"]
    assert features.fingerprints == tuple(request.fingerprint for request in requests)
    assert features.causal_offsets.tolist() == [0, 2, 5]


def test_execute_rejects_extractor_output_in_a_different_request_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime()
    requests = (
        _request("first", [1, 2], identity=runtime.identity),
        _request("second", [3, 4], identity=runtime.identity),
    )

    def extract(
        _reasoner: nn.Module,
        received: Sequence[ReasonerFeatureRequest],
    ) -> ReasonerFeatureBatch:
        features = _features_for(received)
        return ReasonerFeatureBatch(
            cross_k=features.cross_k,
            cross_v=features.cross_v,
            causal_offsets=features.causal_offsets,
            fingerprints=tuple(reversed(features.fingerprints)),
        )

    monkeypatch.setattr(reasoner_runtime_module, "extract_reasoner_feature_batch", extract)

    with pytest.raises(ReasonerRuntimeInvariantError, match="do not preserve request order"):
        runtime.execute(requests)


def test_execute_raises_fatal_invariant_error_for_output_signature_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime()
    requests = (_request("sample", [1, 2], identity=runtime.identity),)

    def extract(
        _reasoner: nn.Module,
        received: Sequence[ReasonerFeatureRequest],
    ) -> ReasonerFeatureBatch:
        features = _features_for(received)
        return ReasonerFeatureBatch(
            cross_k=tuple(tensor.to(torch.bfloat16) for tensor in features.cross_k),
            cross_v=tuple(tensor.to(torch.bfloat16) for tensor in features.cross_v),
            causal_offsets=features.causal_offsets,
            fingerprints=features.fingerprints,
        )

    monkeypatch.setattr(reasoner_runtime_module, "extract_reasoner_feature_batch", extract)

    with pytest.raises(ReasonerRuntimeInvariantError, match="output signature changed"):
        runtime.execute(requests)
