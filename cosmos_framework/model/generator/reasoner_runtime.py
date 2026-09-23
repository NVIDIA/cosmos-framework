# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Reusable frozen-Reasoner runtime for offline extraction and remote serving.

The runtime owns one complete, unsharded Reasoner replica.  It deliberately
does not initialize a distributed process group: scale-out is achieved by
starting independent one-process/one-GPU replicas behind an external load
balancer.  Calls are serialized because the first implementation executes
variable-length requests one at a time inside
``extract_reasoner_feature_batch``.
"""

from __future__ import annotations

import copy
import threading
import time
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

import torch
from omegaconf import open_dict

from cosmos_framework.checkpoint.reasoner_only import load_reasoner_only_dcp
from cosmos_framework.model.generator.reasoner_features import (
    ReasonerFeatureBatch,
    ReasonerFeatureIdentity,
    ReasonerFeatureRequest,
    ReasonerFeatureSignature,
    compute_reasoner_feature_fingerprint,
    extract_reasoner_feature_batch,
)
from cosmos_framework.utils.lazy_config import instantiate as lazy_instantiate


@dataclass(frozen=True)
class ReasonerRuntimeSpec:
    """Immutable startup and admission limits for one Reasoner replica."""

    checkpoint: str | Path
    checkpoint_source: Literal["regular", "ema"]
    device: torch.device | str
    dtype: torch.dtype
    identity: ReasonerFeatureIdentity
    max_requests: int = 64
    max_total_tokens: int = 4_096

    def __post_init__(self) -> None:
        if self.checkpoint_source not in {"regular", "ema"}:
            raise ValueError(f"Unsupported checkpoint_source={self.checkpoint_source!r}")
        if self.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise TypeError(f"Unsupported Reasoner runtime dtype: {self.dtype}")
        for name in ("max_requests", "max_total_tokens"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")


@dataclass(frozen=True)
class ReasonerRuntimeLoadStats:
    construction_seconds: float
    checkpoint_load_seconds: float
    allocated_bytes: int
    peak_allocated_bytes: int
    reserved_bytes: int
    peak_reserved_bytes: int


class ReasonerRuntimeInvariantError(RuntimeError):
    """A permanent loaded-runtime/output contract violation requiring restart."""


def prepare_reasoner_model_config(config: object) -> object:
    """Copy the Nano LM lazy config and remove Generator/vision construction."""

    model = getattr(getattr(config, "model"), "config")
    model_instance = copy.deepcopy(model.vlm_config.model_instance)
    if model_instance is None:
        raise ValueError("Reasoner runtime requires model.config.vlm_config.model_instance")
    nested_config = model_instance["config"]
    if isinstance(nested_config, dict):
        nested_config.update(
            include_gen_pathway=False,
            include_und_pathway=True,
            include_visual=False,
        )
    else:
        with open_dict(nested_config):
            nested_config.include_gen_pathway = False
            nested_config.include_und_pathway = True
            nested_config.include_visual = False
    return model_instance


@contextmanager
def _temporary_default_dtype(dtype: torch.dtype) -> Iterator[None]:
    """Set the process-global construction dtype during single-threaded startup."""

    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def build_reasoner(config: object, *, device: torch.device, dtype: torch.dtype) -> torch.nn.Module:
    """Construct a materialized Reasoner replica directly in its compute dtype.

    Direct construction intentionally avoids meta ``to_empty``: HuggingFace
    rotary buffers are non-persistent and absent from DCP, so constructing on
    the final device initializes them before the strict checkpoint load.
    """

    model_instance = prepare_reasoner_model_config(config)
    with _temporary_default_dtype(dtype), torch.device(device):
        reasoner = lazy_instantiate(model_instance)
    generation_parameters = [name for name, _ in reasoner.named_parameters() if "moe_gen" in name]
    if generation_parameters:
        raise RuntimeError(f"Reasoner-only construction retained Generator parameters: {generation_parameters}")
    reasoner.requires_grad_(False)
    reasoner.eval()
    return reasoner


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _memory_stats(device: torch.device) -> tuple[int, int, int, int]:
    if device.type != "cuda":
        return 0, 0, 0, 0
    return (
        torch.cuda.memory_allocated(device),
        torch.cuda.max_memory_allocated(device),
        torch.cuda.memory_reserved(device),
        torch.cuda.max_memory_reserved(device),
    )


class ReasonerFeatureRuntime:
    """Loaded Reasoner plus strict request validation and serialized execution."""

    def __init__(
        self,
        reasoner: torch.nn.Module,
        *,
        identity: ReasonerFeatureIdentity,
        max_requests: int = 64,
        max_total_tokens: int = 4_096,
        load_stats: ReasonerRuntimeLoadStats | None = None,
    ) -> None:
        if reasoner.training:
            raise ValueError("ReasonerFeatureRuntime requires reasoner.eval()")
        if any(parameter.requires_grad for parameter in reasoner.parameters()):
            raise ValueError("ReasonerFeatureRuntime requires every Reasoner parameter to be frozen")
        if max_requests <= 0 or max_total_tokens <= 0:
            raise ValueError("Reasoner runtime request and token limits must be positive")
        try:
            model = reasoner.model
            layers = model.layers
            embedding = model.embed_tokens
            attentions = tuple(layer.self_attn for layer in layers)
            first_attention = attentions[0]
        except (AttributeError, IndexError) as error:
            raise TypeError(
                "Expected a *TextForCausalLM wrapper with model.embed_tokens and non-empty model.layers"
            ) from error
        if not getattr(model, "include_und_pathway", True):
            raise ValueError("Reasoner runtime requires include_und_pathway=True")
        if getattr(model, "include_gen_pathway", False):
            raise ValueError("Reasoner runtime must not retain the Generator pathway")
        if not callable(getattr(model, "reasoner_forward", None)):
            raise TypeError("Reasoner runtime requires model.reasoner_forward to be callable")
        unsupported_layers = [
            layer_idx
            for layer_idx, attention in enumerate(attentions)
            if getattr(attention, "k_norm_und_for_gen", None) is not None
        ]
        if unsupported_layers:
            raise NotImplementedError(
                "UND-only Reasoner execution does not support generator-specific K normalization; "
                f"affected layers={unsupported_layers}"
            )

        head_dim = int(getattr(first_attention, "head_dim"))
        key_projection = getattr(first_attention, "k_proj")
        if key_projection.out_features % head_dim:
            raise ValueError("Reasoner key projection width is not divisible by head_dim")
        num_kv_heads = int(key_projection.out_features // head_dim)
        self.reasoner = reasoner
        self.identity = identity
        self.max_requests = int(max_requests)
        self.max_total_tokens = int(max_total_tokens)
        self.signature = ReasonerFeatureSignature(
            num_layers=len(layers),
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=embedding.weight.dtype,
        )
        self.vocab_size = int(embedding.num_embeddings)
        self.device = embedding.weight.device
        self.load_stats = load_stats
        self._execution_lock = threading.Lock()

    @classmethod
    def load(cls, config: object, spec: ReasonerRuntimeSpec) -> ReasonerFeatureRuntime:
        """Construct and strictly restore one replica before reporting it ready."""

        device = torch.device(spec.device)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA Reasoner runtime requested, but CUDA is unavailable")
            if device.index is None:
                device = torch.device("cuda", torch.cuda.current_device())
            if device.index < 0 or device.index >= torch.cuda.device_count():
                raise ValueError(f"Reasoner runtime device {device} is not visible")
            torch.cuda.set_device(device)

        _synchronize(device)
        construction_started = time.perf_counter()
        reasoner = build_reasoner(config, device=device, dtype=spec.dtype)
        runtime = cls(
            reasoner,
            identity=spec.identity,
            max_requests=spec.max_requests,
            max_total_tokens=spec.max_total_tokens,
        )
        _synchronize(device)
        construction_seconds = time.perf_counter() - construction_started

        checkpoint_started = time.perf_counter()
        load_reasoner_only_dcp(reasoner, spec.checkpoint, source=spec.checkpoint_source)
        _synchronize(device)
        checkpoint_load_seconds = time.perf_counter() - checkpoint_started
        allocated, peak_allocated, reserved, peak_reserved = _memory_stats(device)
        runtime.load_stats = ReasonerRuntimeLoadStats(
            construction_seconds=construction_seconds,
            checkpoint_load_seconds=checkpoint_load_seconds,
            allocated_bytes=allocated,
            peak_allocated_bytes=peak_allocated,
            reserved_bytes=reserved,
            peak_reserved_bytes=peak_reserved,
        )
        return runtime

    def estimated_response_bytes(self, requests: Sequence[ReasonerFeatureRequest]) -> int:
        """Return the canonical K/V payload size before transport framing."""

        total_tokens = sum(request.token_ids.numel() for request in requests)
        return (
            total_tokens
            * self.signature.num_layers
            * 2
            * self.signature.num_kv_heads
            * self.signature.head_dim
            * self.signature.dtype.itemsize
        )

    def _validate_requests(self, requests: Sequence[ReasonerFeatureRequest]) -> tuple[ReasonerFeatureRequest, ...]:
        requests = tuple(requests)
        if not requests:
            raise ValueError("At least one Reasoner feature request is required")
        if len(requests) > self.max_requests:
            raise ValueError(f"Reasoner request count {len(requests)} exceeds limit {self.max_requests}")
        total_tokens = sum(request.token_ids.numel() for request in requests)
        if total_tokens > self.max_total_tokens:
            raise ValueError(f"Reasoner token count {total_tokens} exceeds limit {self.max_total_tokens}")
        for request in requests:
            if request.token_ids.numel() == 0:
                raise ValueError(f"Reasoner request {request.sample_key!r} contains no tokens")
            token_ids = request.token_ids.detach().to(device="cpu", dtype=torch.int64)
            minimum = int(token_ids.min())
            maximum = int(token_ids.max())
            if minimum < 0 or maximum >= self.vocab_size:
                raise ValueError(
                    f"Reasoner request {request.sample_key!r} has token range [{minimum}, {maximum}] "
                    f"outside vocabulary [0, {self.vocab_size})"
                )
            expected_fingerprint = compute_reasoner_feature_fingerprint(
                request.token_ids,
                request.position_ids,
                request.causal_offsets,
                identity=self.identity,
            )
            if request.fingerprint != expected_fingerprint:
                raise ValueError(
                    f"Reasoner request fingerprint mismatch for {request.sample_key!r}: "
                    f"request={request.fingerprint!r}, expected={expected_fingerprint!r}"
                )
        return requests

    @torch.inference_mode()
    def execute(self, requests: Sequence[ReasonerFeatureRequest]) -> ReasonerFeatureBatch:
        """Validate and execute requests in order on the owned Reasoner replica."""

        validated = self._validate_requests(requests)
        with self._execution_lock:
            features = extract_reasoner_feature_batch(self.reasoner, validated)
        actual = ReasonerFeatureSignature(
            num_layers=features.num_layers,
            num_kv_heads=features.layer(0).num_kv_heads,
            head_dim=features.layer(0).head_dim,
            dtype=features.layer(0).cross_k.dtype,
        )
        if actual != self.signature:
            raise ReasonerRuntimeInvariantError(
                f"Reasoner runtime output signature changed: output={actual!r}, ready={self.signature!r}"
            )
        expected_fingerprints = tuple(request.fingerprint for request in validated)
        if features.fingerprints != expected_fingerprints:
            raise ReasonerRuntimeInvariantError(
                "Reasoner runtime output fingerprints do not preserve request order: "
                f"output={features.fingerprints!r}, expected={expected_fingerprints!r}"
            )
        return features
