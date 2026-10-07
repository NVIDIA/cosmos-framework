# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Optional read-time temporal rephasing of an immutable frame-zero media sink."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch

from cosmos_framework.model.generator.reasoner.qwen3_vl.qwen3_vl import Qwen3VLTextRotaryEmbedding
from cosmos_framework.model.generator.utils.kv_cache import MultiviewARMemoryState, MultiviewARMemoryValue


@dataclass(frozen=True)
class MediaSinkKeyRotation:
    """Unit rotation shared by layers; original cache tensors are never changed."""

    token_count: int
    time_offset_seconds: float
    temporal_offset: float
    cos: torch.Tensor  # [1,1,1,D]
    sin: torch.Tensor  # [1,1,1,D]

    def apply(self, keys: torch.Tensor) -> torch.Tensor:  # keys, returns: [1,M,H_kv,D]
        if keys.ndim != 4 or keys.shape[-1] != self.cos.shape[-1] or not 0 < self.token_count <= keys.shape[1]:
            raise ValueError("Media sink rotation does not match the cached key geometry.")
        original = keys[:, : self.token_count].float()  # [1,S_sink,H_kv,D]
        half = original.shape[-1] // 2
        rotated = torch.cat((-original[..., half:], original[..., :half]), dim=-1)  # [1,S_sink,H_kv,D]
        shifted_ = original * self.cos + rotated * self.sin  # [1,S_sink,H_kv,D]
        result = keys.clone()  # [1,M,H_kv,D]
        result[:, : self.token_count] = shifted_.to(dtype=keys.dtype)  # [1,S_sink,H_kv,D]
        return result  # [1,M,H_kv,D]


@torch.no_grad()
def prepare_media_sink_key_rotation(
    rotary_embedding: Qwen3VLTextRotaryEmbedding,
    *,
    token_count: int,
    chunk_start_seconds: float,
    age_cap_seconds: float,
    temporal_units_per_second: float,
) -> MediaSinkKeyRotation | None:
    """Translate only temporal phase using the loaded native interleaved MRoPE.

    The cap is relative to the earliest query timestamp in a synchronized
    chunk. A later query in that same chunk can be older by its time offset.
    Static frequencies are required: dynamic RoPE could mutate the frequency
    basis of the already-rotated stored keys. Attention scaling is removed
    from the delta factors because the keys already contain that scaling.
    """
    if not isinstance(rotary_embedding, Qwen3VLTextRotaryEmbedding) or rotary_embedding.rope_type != "default":
        raise ValueError("Media sink age capping requires static Qwen3-VL rotary embeddings.")
    if isinstance(token_count, bool) or not isinstance(token_count, int) or token_count < 0:
        raise ValueError("Media sink token count must be a nonnegative integer.")
    if (
        not math.isfinite(chunk_start_seconds)
        or chunk_start_seconds < 0
        or isinstance(age_cap_seconds, bool)
        or not math.isfinite(age_cap_seconds)
        or age_cap_seconds <= 0
        or not math.isfinite(temporal_units_per_second)
        or temporal_units_per_second <= 0
    ):
        raise ValueError("Media sink age capping requires finite positive age and temporal scale.")
    scaling = float(rotary_embedding.attention_scaling)
    if not math.isfinite(scaling) or scaling <= 0:
        raise ValueError("The native rotary attention scaling must be finite and positive.")
    time_offset = max(0.0, chunk_start_seconds - age_cap_seconds)
    if token_count == 0 or time_offset == 0:
        return None
    temporal_offset = time_offset * temporal_units_per_second
    device = rotary_embedding.inv_freq.device
    delta = torch.tensor([[[temporal_offset]], [[0.0]], [[0.0]]], device=device, dtype=torch.float32)  # [3,1,1]
    sentinel = torch.empty((), device=device, dtype=torch.float32)  # []
    cos, sin = rotary_embedding(sentinel, delta)  # each [1,1,D]
    unit_cos = (cos / scaling).unsqueeze(2)  # [1,1,1,D]
    unit_sin = (sin / scaling).unsqueeze(2)  # [1,1,1,D]
    return MediaSinkKeyRotation(token_count, time_offset, temporal_offset, unit_cos, unit_sin)


class MediaSinkMemoryState(MultiviewARMemoryState):
    """Serve a temporary sink-key phase while inheriting the clean write path."""

    sink_key_rotation: MediaSinkKeyRotation | None = None

    def read_for_layer(self, layer_idx: int) -> MultiviewARMemoryValue:
        value = super().read_for_layer(layer_idx)
        if self.sink_key_rotation is None or value.cached_gen_k is None:
            return value
        keys_ = self.sink_key_rotation.apply(value.cached_gen_k)  # [1,M,H_kv,D]
        return replace(value, cached_gen_k=keys_)
