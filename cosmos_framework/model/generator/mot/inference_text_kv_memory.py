# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Request-local text K/V reuse for diffusion inference (cosmos3-local).

Caches understanding (text) K/V from the first denoising forward and reuses
them on later steps within the same request. Intentionally self-contained in
the public generator package.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from cosmos_framework.data.generator.sequence_packing.runtime import SequencePack, from_und_gen_splits, get_gen_seq
from cosmos_framework.model.attention import attention
from cosmos_framework.model.generator.mot.attention import SplitInfo, dispatch_attention
from cosmos_framework.model.generator.utils.memory import KVToStore, MemoryState, MemoryValue


class UndKVCache:
    """Fixed per-layer cache for understanding (text) tokens."""

    def __init__(self) -> None:
        self.k_und: torch.Tensor | None = None
        self.v_und: torch.Tensor | None = None
        self.is_initialized = False

    def store(self, k: torch.Tensor, v: torch.Tensor) -> None:
        """Store und K/V. ``k``/``v``: [B,S_und,H,D]."""
        self.k_und = k.detach().clone()
        self.v_und = v.detach().clone()
        self.is_initialized = True

    def get(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_initialized or self.k_und is None or self.v_und is None:
            raise AssertionError("UndKVCache not initialized — must call store() first")
        return self.k_und, self.v_und


@dataclass
class InferenceTextKVMemoryValue(MemoryValue):
    """Read-only snapshot passed into attention for one decoder layer."""

    und_k_cached: torch.Tensor | None
    und_v_cached: torch.Tensor | None
    frame_idx: int
    gen_len: int
    for_cuda_graphs: bool = False


class InferenceTextKVMemoryState(MemoryState):
    """Request-local memory state that reuses static text K/V during diffusion inference."""

    def __init__(self, text_kv_cache: list[UndKVCache]) -> None:
        self._text_kv_cache = text_kv_cache
        self._gen_len = 0

    def init(self, hidden_states: dict, device: torch.device) -> None:
        del device
        self._gen_len = int(hidden_states["_num_full_tokens"])

    def read_for_layer(self, layer_idx: int) -> InferenceTextKVMemoryValue:
        cache = self._text_kv_cache[layer_idx]
        if cache.is_initialized:
            und_k_cached, und_v_cached = cache.get()  # und_k/v: [1,N_und,H_kv,D]
        else:
            und_k_cached, und_v_cached = None, None

        return InferenceTextKVMemoryValue(
            und_k_cached=und_k_cached,
            und_v_cached=und_v_cached,
            frame_idx=1 if cache.is_initialized else 0,
            gen_len=self._gen_len,
            for_cuda_graphs=False,
        )

    def write_for_layer(self, layer_idx: int, kv_to_store: KVToStore) -> None:
        cache = self._text_kv_cache[layer_idx]
        if cache.is_initialized:
            return
        _gen_k, _gen_v, und_k, und_v = kv_to_store  # gen [1,N_gen,H_kv,D], und [1,N_und,H_kv,D]
        cache.store(und_k, und_v)

    def is_gen_only(self) -> bool:
        # Require every layer: layer-0 alone can be True after a partial first forward (e.g. OOM mid-stack).
        return all(cache.is_initialized for cache in self._text_kv_cache)

    def requires_natten_metadata(self) -> bool:
        return False


def make_inference_text_kv_cache(num_layers: int) -> list[UndKVCache]:
    """Create per-layer request-local text K/V caches for one CFG branch."""
    return [UndKVCache() for _ in range(num_layers)]


def _attention_gen_with_cached_text(
    packed_query_states: SequencePack,
    packed_key_states: SequencePack,
    packed_value_states: SequencePack,
    memory_value: InferenceTextKVMemoryValue,
) -> tuple[SequencePack, KVToStore | None]:
    """Gen-only attention attending to cached text K/V plus current gen K/V."""
    q_gen = get_gen_seq(packed_query_states)  # [S_curr, H, D]
    k_gen = get_gen_seq(packed_key_states)  # [S_curr, H_kv, D]
    v_gen = get_gen_seq(packed_value_states)  # [S_curr, H_kv, D]

    gen_len = memory_value.gen_len
    k_curr = k_gen[:gen_len].unsqueeze(0)  # [1, S_gen_real, H_kv, D]
    v_curr = v_gen[:gen_len].unsqueeze(0)  # [1, S_gen_real, H_kv, D]

    kv_parts_k = [k_curr]
    kv_parts_v = [v_curr]
    if memory_value.und_k_cached is not None:
        assert memory_value.und_v_cached is not None
        kv_parts_k.insert(0, memory_value.und_k_cached)
        kv_parts_v.insert(0, memory_value.und_v_cached)

    k_full = torch.cat(kv_parts_k, dim=1)  # [1, S_total, H_kv, D]
    v_full = torch.cat(kv_parts_v, dim=1)  # [1, S_total, H_kv, D]

    attn_result = attention(
        query=q_gen.unsqueeze(0),  # [1, S_curr, H, D]
        key=k_full,
        value=v_full,
        is_causal=False,
        return_lse=False,
    )
    assert isinstance(attn_result, torch.Tensor)
    gen_out = attn_result.squeeze(0).flatten(-2, -1)  # [S_curr, H*D]

    output = from_und_gen_splits(
        gen_out.new_empty(0, gen_out.shape[-1]),
        gen_out,
        packed_query_states,
    )
    return output, None


def dispatch_attention_with_text_kv_memory(
    packed_query_states: SequencePack,
    packed_key_states: SequencePack,
    packed_value_states: SequencePack,
    attention_mask: object | SplitInfo,
    natten_metadata: dict | None = None,
    memory_value: MemoryValue | None = None,
    packed_key_states_normalized: SequencePack | None = None,
) -> tuple[SequencePack, KVToStore | None]:
    """Dispatch attention with optional request-local text K/V reuse.

    Falls through to standard ``dispatch_attention`` when ``memory_value`` is
    ``None`` or still on the first (cache-fill) step.
    """
    if isinstance(memory_value, InferenceTextKVMemoryValue) and memory_value.frame_idx > 0:
        return _attention_gen_with_cached_text(
            packed_query_states,
            packed_key_states,
            packed_value_states,
            memory_value,
        )
    return dispatch_attention(
        packed_query_states,
        packed_key_states,
        packed_value_states,
        attention_mask,
        natten_metadata=natten_metadata,
        memory_value=None,
        packed_key_states_normalized=packed_key_states_normalized,
    )


def install_inference_memory_attention_dispatch(
    net: torch.nn.Module,
) -> list[tuple[torch.nn.Module, object]]:
    """Temporarily install text-KV memory attention dispatch.

    Returns prior ``(attn_module, dispatch_fn)`` pairs for restore. Refuses to
    overwrite an unexpected dispatcher (e.g. CP).
    """
    previous: list[tuple[torch.nn.Module, object]] = []
    try:
        for layer in net.language_model.model.layers:
            attn = layer.self_attn
            current = attn.dispatch_attention_fn
            if current is not dispatch_attention and current is not dispatch_attention_with_text_kv_memory:
                current_name = getattr(current, "__name__", type(current).__name__)
                raise RuntimeError(
                    "Cannot install memory-aware attention dispatch over "
                    f"{current_name}; request-local text K/V reuse requires the default "
                    "dispatch_attention (CP/CFGP paths are excluded by eligibility guards)."
                )
            previous.append((attn, current))
            attn.dispatch_attention_fn = dispatch_attention_with_text_kv_memory
    except Exception:
        for attn, previous_fn in previous:
            attn.dispatch_attention_fn = previous_fn
        raise
    return previous


def restore_inference_attention_dispatch(previous: list[tuple[torch.nn.Module, object]]) -> None:
    """Restore attention dispatch functions saved by ``install_inference_memory_attention_dispatch``."""
    for attn, previous_fn in previous:
        attn.dispatch_attention_fn = previous_fn
