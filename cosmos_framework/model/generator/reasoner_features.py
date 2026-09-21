# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Provider-neutral Reasoner K/V features for generator-only execution.

The canonical boundary is the exact key/value pair consumed by the generator's
cross-attention, after all Reasoner-side normalization and RoPE.  Canonical
tensors are unpadded and use ``[sequence, kv_heads, head_dim]`` layout.  This
module deliberately lives outside :mod:`cosmos_framework.inference` so inline,
offline, and remote providers can share one training-safe contract.

Only ordinary two-way attention without context parallelism is supported here.
Unsupported attention layouts fail closed instead of silently applying a less
restrictive mask.
"""

from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

import torch

from cosmos_framework.data.generator.sequence_packing.runtime import (
    SequencePack,
    drop_pad_segment,
    from_und_gen_splits,
    get_gen_seq,
)
from cosmos_framework.model.attention import attention
from cosmos_framework.model.generator.mot.attention import SplitInfo, dispatch_attention
from cosmos_framework.model.generator.mot.unified_mot import ReasonerKVCache
from cosmos_framework.model.generator.utils.memory import KVToStore, MemoryState, MemoryValue

_CACHE_CROSS_K = "cross_k"
_CACHE_CROSS_V = "cross_v"
_CACHE_CAUSAL_OFFSETS = "causal_offsets"


def _detached_tensor(value: torch.Tensor, *, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}")
    if value.device.type == "meta":
        raise ValueError(f"{name} must contain materialized values, not a meta tensor")
    return value.detach()


def _validate_offsets(
    offsets: torch.Tensor,
    *,
    name: str,
    expected_total: int | None = None,
) -> torch.Tensor:
    offsets = _detached_tensor(offsets, name=name)
    if offsets.ndim != 1:
        raise ValueError(f"{name} must have shape [segments + 1], got {tuple(offsets.shape)}")
    if offsets.numel() < 2:
        raise ValueError(f"{name} must contain at least [0, end]")
    if offsets.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"{name} must use int32 or int64, got {offsets.dtype}")

    # Feature/state construction is deliberately outside torch.compile.  A host
    # copy gives useful validation errors and avoids data-dependent guards in the
    # compiled decoder layers.
    host_offsets = offsets.to(device="cpu", dtype=torch.int64)
    if int(host_offsets[0]) != 0:
        raise ValueError(f"{name} must start at 0, got {int(host_offsets[0])}")
    if bool(torch.any(host_offsets[1:] < host_offsets[:-1])):
        raise ValueError(f"{name} must be monotonically non-decreasing")
    total = int(host_offsets[-1])
    if expected_total is not None and total != expected_total:
        raise ValueError(f"{name} ends at {total}, expected {expected_total}")
    if total > torch.iinfo(torch.int32).max:
        raise ValueError(f"{name} exceeds the int32 attention-metadata limit")
    return offsets


@dataclass(frozen=True)
class ReasonerLayerKV:
    """Canonical K/V for one decoder layer.

    ``cross_k`` and ``cross_v`` are unpadded tensors with shape
    ``[S_und, num_kv_heads, head_dim]``.  Construction detaches both tensors;
    gradients must stop at the frozen Reasoner/generator boundary.
    """

    cross_k: torch.Tensor
    cross_v: torch.Tensor

    def __post_init__(self) -> None:
        cross_k = _detached_tensor(self.cross_k, name="cross_k")
        cross_v = _detached_tensor(self.cross_v, name="cross_v")
        if cross_k.ndim != 3 or cross_v.ndim != 3:
            raise ValueError(
                "Reasoner layer K/V must have shape [sequence, kv_heads, head_dim], "
                f"got K={tuple(cross_k.shape)} V={tuple(cross_v.shape)}"
            )
        if cross_k.shape != cross_v.shape:
            raise ValueError(
                f"Reasoner layer K/V shapes must match, got K={tuple(cross_k.shape)} V={tuple(cross_v.shape)}"
            )
        if any(size <= 0 for size in cross_k.shape):
            raise ValueError(f"Reasoner layer K/V dimensions must be positive, got {tuple(cross_k.shape)}")
        if cross_k.dtype != cross_v.dtype:
            raise TypeError(f"Reasoner layer K/V dtypes must match, got K={cross_k.dtype} V={cross_v.dtype}")
        if cross_k.device != cross_v.device:
            raise ValueError(f"Reasoner layer K/V devices must match, got K={cross_k.device} V={cross_v.device}")
        if not cross_k.is_floating_point():
            raise TypeError(f"Reasoner layer K/V must be floating point, got {cross_k.dtype}")
        object.__setattr__(self, "cross_k", cross_k)
        object.__setattr__(self, "cross_v", cross_v)

    @property
    def sequence_length(self) -> int:
        return self.cross_k.shape[0]

    @property
    def num_kv_heads(self) -> int:
        return self.cross_k.shape[1]

    @property
    def head_dim(self) -> int:
        return self.cross_k.shape[2]

    def to(
        self,
        device: torch.device | str,
        *,
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
    ) -> ReasonerLayerKV:
        """Return this layer on ``device`` while preserving the detached boundary."""
        return ReasonerLayerKV(
            self.cross_k.to(device=device, dtype=dtype, non_blocking=non_blocking),
            self.cross_v.to(device=device, dtype=dtype, non_blocking=non_blocking),
        )


@dataclass(frozen=True)
class ReasonerFeatureRequest:
    """Storage/provider-independent identity and inputs for one feature request."""

    sample_key: str
    token_ids: torch.Tensor
    position_ids: torch.Tensor
    causal_offsets: torch.Tensor
    fingerprint: str

    def __post_init__(self) -> None:
        if not self.sample_key:
            raise ValueError("sample_key must be non-empty")
        if not self.fingerprint:
            raise ValueError("fingerprint must be non-empty")
        token_ids = _detached_tensor(self.token_ids, name="token_ids")
        position_ids = _detached_tensor(self.position_ids, name="position_ids")
        if token_ids.ndim != 1:
            raise ValueError(f"token_ids must have shape [sequence], got {tuple(token_ids.shape)}")
        if token_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"token_ids must use int32 or int64, got {token_ids.dtype}")
        if position_ids.ndim not in (1, 2) or position_ids.shape[-1] != token_ids.numel():
            raise ValueError(
                "position_ids must have shape [sequence] or [axes, sequence] matching token_ids, "
                f"got {tuple(position_ids.shape)} for {token_ids.numel()} tokens"
            )
        if position_ids.dtype not in (torch.int32, torch.int64, torch.float32, torch.float64):
            raise TypeError(f"position_ids must use int32, int64, float32, or float64, got {position_ids.dtype}")
        causal_offsets = _validate_offsets(
            self.causal_offsets,
            name="causal_offsets",
            expected_total=token_ids.numel(),
        )
        object.__setattr__(self, "token_ids", token_ids)
        object.__setattr__(self, "position_ids", position_ids)
        object.__setattr__(self, "causal_offsets", causal_offsets)


@dataclass(frozen=True)
class ReasonerFeatureBatch:
    """Canonical unpadded Reasoner K/V for a packed batch.

    ``cross_k[layer]`` and ``cross_v[layer]`` each use the canonical
    ``[S_und, H_kv, D]`` layout.  ``causal_offsets`` partitions that shared
    sequence stream into samples.  ``fingerprints`` may be empty for ephemeral
    inline capture; provider-produced batches carry one fingerprint per sample.
    """

    cross_k: tuple[torch.Tensor, ...]
    cross_v: tuple[torch.Tensor, ...]
    causal_offsets: torch.Tensor
    fingerprints: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        cross_k = tuple(self.cross_k)
        cross_v = tuple(self.cross_v)
        fingerprints = tuple(self.fingerprints)
        if not cross_k:
            raise ValueError("ReasonerFeatureBatch must contain at least one layer")
        if len(cross_k) != len(cross_v):
            raise ValueError(f"ReasonerFeatureBatch layer counts disagree: K={len(cross_k)} V={len(cross_v)}")

        layers = tuple(ReasonerLayerKV(k, v) for k, v in zip(cross_k, cross_v))
        reference = layers[0]
        for layer_idx, layer in enumerate(layers[1:], start=1):
            if layer.cross_k.shape != reference.cross_k.shape:
                raise ValueError(
                    "All Reasoner layers must share [sequence, kv_heads, head_dim], "
                    f"layer 0={tuple(reference.cross_k.shape)} layer {layer_idx}={tuple(layer.cross_k.shape)}"
                )
            if layer.cross_k.dtype != reference.cross_k.dtype:
                raise TypeError(
                    f"All Reasoner layers must share a dtype, layer 0={reference.cross_k.dtype} "
                    f"layer {layer_idx}={layer.cross_k.dtype}"
                )
            if layer.cross_k.device != reference.cross_k.device:
                raise ValueError(
                    f"All Reasoner layers must share a device, layer 0={reference.cross_k.device} "
                    f"layer {layer_idx}={layer.cross_k.device}"
                )

        causal_offsets = _validate_offsets(
            self.causal_offsets,
            name="causal_offsets",
            expected_total=reference.sequence_length,
        )
        num_samples = causal_offsets.numel() - 1
        if fingerprints and len(fingerprints) != num_samples:
            raise ValueError(
                f"fingerprints must be empty or contain one entry per sample, got {len(fingerprints)} for "
                f"{num_samples} samples"
            )
        if any(not fingerprint for fingerprint in fingerprints):
            raise ValueError("fingerprints must not contain empty entries")

        object.__setattr__(self, "cross_k", tuple(layer.cross_k for layer in layers))
        object.__setattr__(self, "cross_v", tuple(layer.cross_v for layer in layers))
        object.__setattr__(self, "causal_offsets", causal_offsets)
        object.__setattr__(self, "fingerprints", fingerprints)

    @property
    def num_layers(self) -> int:
        return len(self.cross_k)

    @property
    def num_samples(self) -> int:
        return self.causal_offsets.numel() - 1

    @property
    def sequence_length(self) -> int:
        return self.cross_k[0].shape[0]

    def layer(self, layer_idx: int) -> ReasonerLayerKV:
        """Return one validated canonical layer."""
        return ReasonerLayerKV(self.cross_k[layer_idx], self.cross_v[layer_idx])

    @classmethod
    def from_stacked(
        cls,
        cross_k: torch.Tensor,
        cross_v: torch.Tensor,
        causal_offsets: torch.Tensor,
        fingerprints: Sequence[str] = (),
    ) -> ReasonerFeatureBatch:
        """Build from cache-friendly ``[layers, sequence, heads, dim]`` tensors."""
        cross_k = _detached_tensor(cross_k, name="cross_k")
        cross_v = _detached_tensor(cross_v, name="cross_v")
        if cross_k.ndim != 4 or cross_v.ndim != 4:
            raise ValueError(
                "Stacked Reasoner K/V must have shape [layers, sequence, kv_heads, head_dim], "
                f"got K={tuple(cross_k.shape)} V={tuple(cross_v.shape)}"
            )
        if cross_k.shape != cross_v.shape:
            raise ValueError(f"Stacked Reasoner K/V shapes must match, got K={cross_k.shape} V={cross_v.shape}")
        return cls(tuple(cross_k.unbind(0)), tuple(cross_v.unbind(0)), causal_offsets, tuple(fingerprints))

    def to_stacked(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cache-friendly ``[layers, sequence, heads, dim]`` K/V tensors."""
        return torch.stack(self.cross_k, dim=0), torch.stack(self.cross_v, dim=0)

    @classmethod
    def from_cache_tensors(
        cls,
        tensors: Mapping[str, torch.Tensor],
        *,
        fingerprints: Sequence[str] = (),
    ) -> ReasonerFeatureBatch:
        """Build from the tensor payload used by safetensors/cache providers."""
        missing = {_CACHE_CROSS_K, _CACHE_CROSS_V, _CACHE_CAUSAL_OFFSETS} - tensors.keys()
        if missing:
            raise KeyError(f"Reasoner feature cache payload is missing: {sorted(missing)}")
        return cls.from_stacked(
            tensors[_CACHE_CROSS_K],
            tensors[_CACHE_CROSS_V],
            tensors[_CACHE_CAUSAL_OFFSETS],
            fingerprints,
        )

    def to_cache_tensors(self) -> dict[str, torch.Tensor]:
        """Return a storage-friendly tensor mapping; fingerprints belong in the manifest."""
        cross_k, cross_v = self.to_stacked()
        return {
            _CACHE_CROSS_K: cross_k,
            _CACHE_CROSS_V: cross_v,
            _CACHE_CAUSAL_OFFSETS: self.causal_offsets,
        }

    def to(
        self,
        device: torch.device | str,
        *,
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
    ) -> ReasonerFeatureBatch:
        """Stage all layers and offsets on a device."""
        return ReasonerFeatureBatch(
            tuple(k.to(device=device, dtype=dtype, non_blocking=non_blocking) for k in self.cross_k),
            tuple(v.to(device=device, dtype=dtype, non_blocking=non_blocking) for v in self.cross_v),
            self.causal_offsets.to(device=device, non_blocking=non_blocking),
            self.fingerprints,
        )


class ReasonerFeatureProvider(Protocol):
    """Asynchronous provider contract shared by inline/offline/remote backends."""

    def submit(self, requests: Sequence[ReasonerFeatureRequest]) -> Future[ReasonerFeatureBatch]: ...


@torch.inference_mode()
def extract_reasoner_feature_batch(
    causal_lm: torch.nn.Module,
    requests: Sequence[ReasonerFeatureRequest],
) -> ReasonerFeatureBatch:
    """Run the UND-only prefill and return its exact generator-facing K/V.

    Requests are evaluated independently to preserve causal-document isolation.
    The initial implementation intentionally rejects architectures that apply a
    second generator-specific UND K normalization; Nano's Qwen dense model uses
    the same normalized/RoPE K for Reasoner self-attention and GEN cross-attention.
    """
    if not requests:
        raise ValueError("At least one Reasoner feature request is required")
    multi_document_requests = [request.sample_key for request in requests if request.causal_offsets.numel() != 2]
    if multi_document_requests:
        raise NotImplementedError(
            "Reasoner feature extraction currently requires one causal document per request; "
            f"multi-document requests={multi_document_requests}"
        )
    try:
        model = causal_lm.model
        layers = model.layers
        embedding = model.embed_tokens
    except AttributeError as error:
        raise TypeError("Expected a *TextForCausalLM wrapper with model.embed_tokens and model.layers") from error
    if not getattr(model, "include_und_pathway", True):
        raise ValueError("Reasoner feature extraction requires include_und_pathway=True")
    if causal_lm.training:
        raise ValueError("Reasoner feature extraction requires causal_lm.eval()")
    unsupported_layers = [
        layer_idx
        for layer_idx, layer in enumerate(layers)
        if getattr(layer.self_attn, "k_norm_und_for_gen", None) is not None
    ]
    if unsupported_layers:
        raise NotImplementedError(
            "UND-only extraction does not yet support generator-specific K normalization; "
            f"affected layers={unsupported_layers}"
        )

    device = embedding.weight.device
    per_request_keys: list[tuple[torch.Tensor, ...]] = []
    per_request_values: list[tuple[torch.Tensor, ...]] = []
    offsets = [0]
    for request in requests:
        token_ids = request.token_ids.to(device=device, dtype=torch.long).unsqueeze(0)
        positions = request.position_ids.to(device=device)
        position_ids = positions.unsqueeze(0) if positions.ndim == 1 else positions.unsqueeze(1)
        cache = ReasonerKVCache.empty(num_layers=len(layers))
        model.reasoner_forward(input_ids=token_ids, position_ids=position_ids, cache=cache)
        if any(key is None for key in cache.keys) or any(value is None for value in cache.values):
            raise RuntimeError("Reasoner prefill did not populate every K/V cache layer")
        keys = tuple(key.squeeze(0).detach() for key in cache.keys if key is not None)
        values = tuple(value.squeeze(0).detach() for value in cache.values if value is not None)
        if any(key.shape[0] != request.token_ids.numel() for key in keys):
            raise RuntimeError(f"Reasoner K/V length mismatch for request {request.sample_key!r}")
        per_request_keys.append(keys)
        per_request_values.append(values)
        offsets.append(offsets[-1] + request.token_ids.numel())

    cross_k = tuple(
        torch.cat([request_keys[layer_idx] for request_keys in per_request_keys], dim=0)
        for layer_idx in range(len(layers))
    )
    cross_v = tuple(
        torch.cat([request_values[layer_idx] for request_values in per_request_values], dim=0)
        for layer_idx in range(len(layers))
    )
    return ReasonerFeatureBatch(
        cross_k=cross_k,
        cross_v=cross_v,
        causal_offsets=torch.tensor(offsets, dtype=torch.int64, device=device),
        fingerprints=tuple(request.fingerprint for request in requests),
    )


@dataclass(frozen=True)
class ReasonerAttentionMetadata:
    """Precomputed packed-attention layout shared by every decoder layer."""

    gen_len: int
    kv_reorder_indices: torch.Tensor | None = None
    cumulative_seqlen_q: torch.Tensor | None = None
    cumulative_seqlen_kv: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_kv: int = 0


def _pack_offsets(pack: SequencePack, key: str, *, expected_total: int) -> torch.Tensor:
    offsets = pack.get(key)
    if not isinstance(offsets, torch.Tensor):
        raise ValueError(f"SequencePack must provide tensor metadata {key!r}")
    offsets = drop_pad_segment(pack, offsets)
    return _validate_offsets(offsets, name=f"SequencePack[{key!r}]", expected_total=expected_total)


def build_reasoner_attention_metadata(
    feature_batch: ReasonerFeatureBatch,
    pack: SequencePack,
    device: torch.device | str,
) -> ReasonerAttentionMetadata:
    """Validate a SequencePack and construct isolated GEN-to-(UND+GEN) ranges.

    Canonical UND and live GEN tensors are each sample-major streams.  For a
    multi-sample pack, ``kv_reorder_indices`` interleaves those two streams as
    ``[und_0, gen_0, und_1, gen_1, ...]`` and varlen offsets prevent leakage
    between samples.  A single sample uses the equivalent dense path.
    """
    target_device = torch.device(device)
    try:
        gen_len = int(pack["_num_full_tokens"])
        und_len = int(pack["_num_causal_tokens"])
    except KeyError as error:
        raise ValueError(f"SequencePack is missing real-token metadata {error.args[0]!r}") from error
    if gen_len <= 0:
        raise ValueError(f"SequencePack must contain at least one real GEN token, got {gen_len}")
    if und_len != feature_batch.sequence_length:
        raise ValueError(
            f"SequencePack UND token count {und_len} does not match Reasoner features {feature_batch.sequence_length}"
        )

    runtime_und_offsets = _pack_offsets(pack, "_causal_seq_offsets", expected_total=und_len)
    runtime_gen_offsets = _pack_offsets(pack, "_full_only_seq_offsets", expected_total=gen_len)
    feature_offsets_host = feature_batch.causal_offsets.to(device="cpu", dtype=torch.int64)
    runtime_und_offsets_host = runtime_und_offsets.to(device="cpu", dtype=torch.int64)
    if not torch.equal(feature_offsets_host, runtime_und_offsets_host):
        raise ValueError(
            "Reasoner feature causal offsets do not match SequencePack offsets: "
            f"features={feature_offsets_host.tolist()} pack={runtime_und_offsets_host.tolist()}"
        )
    if runtime_und_offsets.numel() != runtime_gen_offsets.numel():
        raise ValueError(
            "Reasoner/GEN sample counts disagree: "
            f"reasoner={runtime_und_offsets.numel() - 1} gen={runtime_gen_offsets.numel() - 1}"
        )

    und_offsets = feature_batch.causal_offsets.to(device=target_device, dtype=torch.int32)
    gen_offsets = runtime_gen_offsets.to(device=target_device, dtype=torch.int32)
    if und_offsets.numel() == 2:
        return ReasonerAttentionMetadata(gen_len=gen_len)

    und_lens = und_offsets[1:] - und_offsets[:-1]
    gen_lens = gen_offsets[1:] - gen_offsets[:-1]
    if bool(torch.any(und_lens <= 0)) or bool(torch.any(gen_lens <= 0)):
        raise ValueError("Every sample must contain at least one Reasoner token and one GEN token")
    sample_ids = torch.arange(und_lens.numel(), dtype=torch.int64, device=target_device)
    und_sample_ids = torch.repeat_interleave(sample_ids, und_lens.to(dtype=torch.int64))
    gen_sample_ids = torch.repeat_interleave(sample_ids, gen_lens.to(dtype=torch.int64))
    reorder = torch.argsort(torch.cat((und_sample_ids, gen_sample_ids)), stable=True)
    return ReasonerAttentionMetadata(
        gen_len=gen_len,
        kv_reorder_indices=reorder,
        cumulative_seqlen_q=gen_offsets,
        cumulative_seqlen_kv=und_offsets + gen_offsets,
        max_seqlen_q=int(gen_lens.max()),
        max_seqlen_kv=int((und_lens + gen_lens).max()),
    )


@dataclass
class StaticReasonerKVMemoryValue(MemoryValue):
    """One layer's external Reasoner K/V plus immutable attention metadata."""

    cross_k: torch.Tensor
    cross_v: torch.Tensor
    gen_len: int
    kv_reorder_indices: torch.Tensor | None = None
    cumulative_seqlen_q: torch.Tensor | None = None
    cumulative_seqlen_kv: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_kv: int = 0
    frame_idx: int = 1
    for_cuda_graphs: bool = False

    @property
    def supports_context_parallel_attention(self) -> bool:
        return False


@dataclass
class ReasonerKVCaptureMemoryValue(MemoryValue):
    """Marker requesting a normal joint pass whose UND K/V is captured."""

    frame_idx: int = 0
    for_cuda_graphs: bool = False

    @property
    def supports_context_parallel_attention(self) -> bool:
        return False


class StaticReasonerKVMemoryState(MemoryState):
    """Read-only memory state that starts in gen-only mode from prefilled K/V."""

    def __init__(self, feature_batch: ReasonerFeatureBatch) -> None:
        self.feature_batch = feature_batch
        self._active_batch = feature_batch
        self._metadata: ReasonerAttentionMetadata | None = None

    def init(self, hidden_states: dict, device: torch.device) -> None:
        self._active_batch = self.feature_batch.to(device, non_blocking=True)
        self._metadata = build_reasoner_attention_metadata(self._active_batch, hidden_states, device)

    def read_for_layer(self, layer_idx: int) -> StaticReasonerKVMemoryValue:
        if self._metadata is None:
            raise RuntimeError("StaticReasonerKVMemoryState.init() must be called before read_for_layer()")
        layer = self._active_batch.layer(layer_idx)
        metadata = self._metadata
        return StaticReasonerKVMemoryValue(
            cross_k=layer.cross_k,
            cross_v=layer.cross_v,
            gen_len=metadata.gen_len,
            kv_reorder_indices=metadata.kv_reorder_indices,
            cumulative_seqlen_q=metadata.cumulative_seqlen_q,
            cumulative_seqlen_kv=metadata.cumulative_seqlen_kv,
            max_seqlen_q=metadata.max_seqlen_q,
            max_seqlen_kv=metadata.max_seqlen_kv,
        )

    def write_for_layer(self, layer_idx: int, kv_to_store: KVToStore) -> None:
        # PackedAttentionMoT currently emits an empty-UND write even in gen-only
        # mode.  External features are immutable, so deliberately ignore it.
        del kv_to_store
        if not 0 <= layer_idx < self._active_batch.num_layers:
            raise IndexError(f"Reasoner layer index {layer_idx} is out of range")

    def is_gen_only(self) -> bool:
        return True

    def requires_natten_metadata(self) -> bool:
        return False


class CapturingReasonerKVMemoryState(MemoryState):
    """Capture inline Reasoner K/V once, then serve it through the static path.

    The first forward is a normal joint UND+GEN pass.  ``write_for_layer``
    detaches the exact RoPE-applied K/V emitted by each layer.  Once every layer
    is populated, the next forward reports ``is_gen_only()`` and reuses the
    captured canonical batch.
    """

    def __init__(self, num_layers: int, *, fingerprints: Sequence[str] = ()) -> None:
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        self._layers: list[ReasonerLayerKV | None] = [None] * num_layers
        self._fingerprints = tuple(fingerprints)
        self._causal_offsets: torch.Tensor | None = None
        self._static_state: StaticReasonerKVMemoryState | None = None

    def init(self, hidden_states: dict, device: torch.device) -> None:
        if self.is_gen_only():
            if self._static_state is None:
                self._static_state = StaticReasonerKVMemoryState(self.to_feature_batch())
            self._static_state.init(hidden_states, device)
            return

        try:
            und_len = int(hidden_states["_num_causal_tokens"])
        except KeyError as error:
            raise ValueError("SequencePack is missing real-token metadata '_num_causal_tokens'") from error
        offsets = _pack_offsets(hidden_states, "_causal_seq_offsets", expected_total=und_len).detach()
        if self._causal_offsets is None:
            if self._fingerprints and len(self._fingerprints) != offsets.numel() - 1:
                raise ValueError(
                    f"fingerprints contain {len(self._fingerprints)} entries for {offsets.numel() - 1} samples"
                )
            self._causal_offsets = offsets
        elif not torch.equal(
            self._causal_offsets.to(device="cpu", dtype=torch.int64),
            offsets.to(device="cpu", dtype=torch.int64),
        ):
            raise ValueError("SequencePack causal offsets changed during partial Reasoner K/V capture")

    def read_for_layer(self, layer_idx: int) -> MemoryValue:
        if not 0 <= layer_idx < len(self._layers):
            raise IndexError(f"Reasoner layer index {layer_idx} is out of range")
        if self.is_gen_only():
            if self._static_state is None:
                raise RuntimeError("CapturingReasonerKVMemoryState.init() must run before cached reads")
            return self._static_state.read_for_layer(layer_idx)
        return ReasonerKVCaptureMemoryValue()

    def write_for_layer(self, layer_idx: int, kv_to_store: KVToStore) -> None:
        if not 0 <= layer_idx < len(self._layers):
            raise IndexError(f"Reasoner layer index {layer_idx} is out of range")
        if self.is_gen_only():
            # See StaticReasonerKVMemoryState.write_for_layer.
            return
        if self._causal_offsets is None:
            raise RuntimeError("CapturingReasonerKVMemoryState.init() must be called before writes")
        _gen_k, _gen_v, und_k, und_v = kv_to_store
        expected_total = int(self._causal_offsets[-1].to(device="cpu"))
        self._layers[layer_idx] = ReasonerLayerKV(
            self._flatten_captured(und_k, expected_total, name="und_k").clone(),
            self._flatten_captured(und_v, expected_total, name="und_v").clone(),
        )

    def _flatten_captured(self, tensor: torch.Tensor, expected_total: int, *, name: str) -> torch.Tensor:
        tensor = _detached_tensor(tensor, name=name)
        if tensor.ndim == 3:
            if tensor.shape[0] < expected_total:
                raise ValueError(f"Captured {name} has {tensor.shape[0]} tokens, expected {expected_total}")
            return tensor[:expected_total]
        if tensor.ndim != 4:
            raise ValueError(f"Captured {name} must have shape [S,H,D] or [B,S,H,D], got {tensor.shape}")
        if tensor.shape[0] == 1:
            if tensor.shape[1] < expected_total:
                raise ValueError(f"Captured {name} has {tensor.shape[1]} tokens, expected {expected_total}")
            return tensor[0, :expected_total]

        assert self._causal_offsets is not None
        lengths = torch.diff(self._causal_offsets.to(device="cpu", dtype=torch.int64)).tolist()
        if tensor.shape[0] != len(lengths):
            raise ValueError(f"Captured {name} batch dimension {tensor.shape[0]} does not match {len(lengths)} samples")
        if any(length > tensor.shape[1] for length in lengths):
            raise ValueError(f"Captured {name} rows are too short for per-sample lengths {lengths}")
        return torch.cat([tensor[sample_idx, :length] for sample_idx, length in enumerate(lengths)], dim=0)

    def to_feature_batch(self) -> ReasonerFeatureBatch:
        """Return the completed canonical batch."""
        if self._causal_offsets is None or not self.is_gen_only():
            missing = [idx for idx, layer in enumerate(self._layers) if layer is None]
            raise RuntimeError(f"Reasoner K/V capture is incomplete; missing layers {missing}")
        layers = tuple(layer for layer in self._layers if layer is not None)
        return ReasonerFeatureBatch(
            tuple(layer.cross_k for layer in layers),
            tuple(layer.cross_v for layer in layers),
            self._causal_offsets,
            self._fingerprints,
        )

    def is_gen_only(self) -> bool:
        return all(layer is not None for layer in self._layers)

    def requires_natten_metadata(self) -> bool:
        return False


def _validate_cached_attention_mode(
    packed_query_states: SequencePack,
    packed_key_states: SequencePack,
    packed_value_states: SequencePack,
    attention_mask: object,
    natten_metadata: dict | None,
) -> None:
    if natten_metadata is not None:
        raise ValueError("Static Reasoner K/V supports only two-way attention; NATTEN metadata was provided")
    if any(pack.get("is_sharded", False) for pack in (packed_query_states, packed_key_states, packed_value_states)):
        raise ValueError("Static Reasoner K/V does not support context-parallel sharded SequencePacks")
    if not hasattr(attention_mask, "is_three_way"):
        raise TypeError(f"Unsupported attention metadata: {type(attention_mask)}")
    if bool(getattr(attention_mask, "is_three_way")):
        raise ValueError("Static Reasoner K/V supports only two-way attention, not three-way attention")
    unsupported_fields = (
        "control_stream_token_ranges",
        "flex_block_mask",
        "multiview_maskless",
    )
    enabled = [field for field in unsupported_fields if getattr(attention_mask, field, None) is not None]
    if enabled:
        raise ValueError(f"Static Reasoner K/V does not support specialized attention metadata: {enabled}")


def _attention_gen_with_reasoner_features(
    packed_query_states: SequencePack,
    packed_key_states: SequencePack,
    packed_value_states: SequencePack,
    memory_value: StaticReasonerKVMemoryValue,
) -> tuple[SequencePack, KVToStore | None]:
    q_gen = get_gen_seq(packed_query_states)
    k_gen = get_gen_seq(packed_key_states)
    v_gen = get_gen_seq(packed_value_states)
    if q_gen.ndim != 3 or k_gen.ndim != 3 or v_gen.ndim != 3:
        raise ValueError(
            "GEN Q/K/V must use [sequence, heads, head_dim] layout, "
            f"got Q={q_gen.shape} K={k_gen.shape} V={v_gen.shape}"
        )
    if k_gen.shape != v_gen.shape:
        raise ValueError(f"GEN K/V shapes must match, got K={k_gen.shape} V={v_gen.shape}")
    if memory_value.gen_len > min(q_gen.shape[0], k_gen.shape[0], v_gen.shape[0]):
        raise ValueError(
            f"GEN real-token count {memory_value.gen_len} exceeds Q/K/V stream lengths "
            f"{q_gen.shape[0]}/{k_gen.shape[0]}/{v_gen.shape[0]}"
        )
    cross_k = memory_value.cross_k
    cross_v = memory_value.cross_v
    if cross_k.ndim != 3 or cross_v.shape != cross_k.shape:
        raise ValueError(f"Cached Reasoner K/V have invalid shapes K={cross_k.shape} V={cross_v.shape}")
    if cross_k.shape[1:] != k_gen.shape[1:]:
        raise ValueError(
            f"Cached Reasoner and live GEN K/V head shapes disagree: {cross_k.shape[1:]} vs {k_gen.shape[1:]}"
        )
    if cross_k.device != k_gen.device or cross_v.device != v_gen.device:
        raise ValueError(
            f"Cached Reasoner and live GEN K/V devices disagree: {cross_k.device}/{cross_v.device} "
            f"vs {k_gen.device}/{v_gen.device}"
        )
    if cross_k.dtype != k_gen.dtype or cross_v.dtype != v_gen.dtype:
        raise TypeError(
            f"Cached Reasoner and live GEN K/V dtypes disagree: {cross_k.dtype}/{cross_v.dtype} "
            f"vs {k_gen.dtype}/{v_gen.dtype}"
        )

    gen_len = memory_value.gen_len
    q_real = q_gen[:gen_len].unsqueeze(0)
    k_real = k_gen[:gen_len]
    v_real = v_gen[:gen_len]
    k_full = torch.cat((cross_k, k_real), dim=0)
    v_full = torch.cat((cross_v, v_real), dim=0)

    if memory_value.kv_reorder_indices is None:
        attn_result = attention(
            query=q_real,
            key=k_full.unsqueeze(0),
            value=v_full.unsqueeze(0),
            is_causal=False,
            return_lse=False,
        )
    else:
        if (
            memory_value.cumulative_seqlen_q is None
            or memory_value.cumulative_seqlen_kv is None
            or memory_value.max_seqlen_q <= 0
            or memory_value.max_seqlen_kv <= 0
        ):
            raise ValueError("Multi-sample Reasoner K/V is missing varlen attention metadata")
        k_full = k_full.index_select(0, memory_value.kv_reorder_indices)
        v_full = v_full.index_select(0, memory_value.kv_reorder_indices)
        attn_result = attention(
            query=q_real,
            key=k_full.unsqueeze(0),
            value=v_full.unsqueeze(0),
            is_causal=False,
            return_lse=False,
            cumulative_seqlen_Q=memory_value.cumulative_seqlen_q,
            cumulative_seqlen_KV=memory_value.cumulative_seqlen_kv,
            max_seqlen_Q=memory_value.max_seqlen_q,
            max_seqlen_KV=memory_value.max_seqlen_kv,
        )

    if not isinstance(attn_result, torch.Tensor):
        raise TypeError(f"Attention returned {type(attn_result).__name__}, expected torch.Tensor")
    gen_out_real = attn_result.squeeze(0).flatten(-2, -1)
    gen_out = q_gen.new_zeros((q_gen.shape[0], gen_out_real.shape[-1]))
    gen_out[:gen_len] = gen_out_real
    empty_und = gen_out.new_empty((0, gen_out.shape[-1]))
    return from_und_gen_splits(empty_und, gen_out, packed_query_states), None


def dispatch_attention_with_reasoner_features(
    packed_query_states: SequencePack,
    packed_key_states: SequencePack,
    packed_value_states: SequencePack,
    attention_mask: object | SplitInfo,
    natten_metadata: dict | None = None,
    memory_value: MemoryValue | None = None,
    packed_key_states_normalized: SequencePack | None = None,
) -> tuple[SequencePack, KVToStore | None]:
    """Dispatch ordinary attention, inline capture, or cached GEN attention."""
    if isinstance(memory_value, (StaticReasonerKVMemoryValue, ReasonerKVCaptureMemoryValue)):
        _validate_cached_attention_mode(
            packed_query_states,
            packed_key_states,
            packed_value_states,
            attention_mask,
            natten_metadata,
        )
    if isinstance(memory_value, StaticReasonerKVMemoryValue):
        return _attention_gen_with_reasoner_features(
            packed_query_states,
            packed_key_states,
            packed_value_states,
            memory_value,
        )

    # The capture marker must reach PackedAttentionMoT (so it emits
    # kv_to_store), but the ordinary dispatcher does not consume memory.
    delegated_memory = None if isinstance(memory_value, ReasonerKVCaptureMemoryValue) else memory_value
    return dispatch_attention(
        packed_query_states,
        packed_key_states,
        packed_value_states,
        attention_mask,
        natten_metadata=natten_metadata,
        memory_value=delegated_memory,
        packed_key_states_normalized=packed_key_states_normalized,
    )


ReasonerDispatchSnapshot = list[tuple[torch.nn.Module, object]]


def install_reasoner_feature_attention_dispatch(net: torch.nn.Module) -> ReasonerDispatchSnapshot:
    """Install the cached dispatcher and return an exact restoration snapshot."""
    try:
        layers = net.language_model.model.layers
    except AttributeError as error:
        raise TypeError("Expected a model with net.language_model.model.layers") from error

    previous: ReasonerDispatchSnapshot = []
    try:
        for layer in layers:
            attn = layer.self_attn
            current = attn.dispatch_attention_fn
            if current is not dispatch_attention and current is not dispatch_attention_with_reasoner_features:
                current_name = getattr(current, "__name__", type(current).__name__)
                raise RuntimeError(
                    "Cannot install Reasoner feature attention over "
                    f"{current_name}; only the default non-CP dispatcher is supported"
                )
            previous.append((attn, current))
            attn.dispatch_attention_fn = dispatch_attention_with_reasoner_features
    except Exception:
        restore_reasoner_feature_attention_dispatch(previous)
        raise
    return previous


def restore_reasoner_feature_attention_dispatch(previous: ReasonerDispatchSnapshot) -> None:
    """Restore dispatchers returned by :func:`install_reasoner_feature_attention_dispatch`."""
    for attn, previous_fn in previous:
        attn.dispatch_attention_fn = previous_fn


__all__ = [
    "CapturingReasonerKVMemoryState",
    "ReasonerAttentionMetadata",
    "ReasonerFeatureBatch",
    "ReasonerFeatureProvider",
    "ReasonerFeatureRequest",
    "ReasonerKVCaptureMemoryValue",
    "ReasonerLayerKV",
    "StaticReasonerKVMemoryState",
    "StaticReasonerKVMemoryValue",
    "build_reasoner_attention_metadata",
    "dispatch_attention_with_reasoner_features",
    "extract_reasoner_feature_batch",
    "install_reasoner_feature_attention_dispatch",
    "restore_reasoner_feature_attention_dispatch",
]
