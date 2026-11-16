# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Request-local text K/V reuse for diffusion inference (cosmos3-local).

Caches understanding (text) K/V from the first denoising forward and reuses
them on later steps within the same request. Intentionally self-contained in
``cosmos3`` — does not import ``projects.cosmos3.interactive``.

Supports single-sample packs (dense attention) and multi-sample packs such as
batched CFG (cond+uncond as B=2) via varlen attention with per-sample isolation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from cosmos_framework.model.attention import attention
from cosmos_framework.model.generator.mot.attention import (
    SplitInfo,
    build_packed_sequence,
    dispatch_attention,
    two_way_attention,
)
from cosmos_framework.model.generator.utils.memory import KVToStore, MemoryState, MemoryValue
from cosmos_framework.data.generator.sequence_packing.runtime import (
    SequencePack,
    drop_pad_segment,
    from_all_seq,
    from_und_gen_splits,
    get_gen_seq,
)


if TYPE_CHECKING:
    from cosmos_framework.data.generator.sequence_packing import PackedSequence


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
    """Total gen tokens across all samples (``_num_full_tokens``)."""
    kv_reorder_indices: torch.Tensor | None = None
    """Indices that turn ``[all und|all gen]`` into ``[und_i|gen_i|...]``."""
    cumulative_seqlen_q: torch.Tensor | None = None
    cumulative_seqlen_kv: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_kv: int = 0
    for_cuda_graphs: bool = False
    und_only: bool = False


class InferenceTextKVMemoryState(MemoryState):
    """Request-local memory state that reuses static text K/V during diffusion inference."""

    def __init__(
        self,
        text_kv_cache: list[UndKVCache],
        *,
        prefill: bool = False,
        prepare: Callable[[], None] | None = None,
    ) -> None:
        self._prepare = prepare
        self.prefill = prefill
        self._text_kv_cache = text_kv_cache
        self._gen_len = 0
        self._kv_reorder_indices: torch.Tensor | None = None
        self._cumulative_seqlen_q: torch.Tensor | None = None
        self._cumulative_seqlen_kv: torch.Tensor | None = None
        self._max_seqlen_q = 0
        self._max_seqlen_kv = 0

    def init(self, hidden_states: dict, device: torch.device) -> None:
        if self._prepare is not None:
            self._prepare()
        del device
        self._gen_len = int(hidden_states["_num_full_tokens"])
        # Precompute the multi-sample layout outside the compiled decoder layers.
        # Doing this here avoids converting CUDA offsets to Python lists and
        # rebuilding the same chunked K/V layout once per layer.
        # Real-sample offsets. Everything below counts samples off these -- the ``numel() > 2``
        # multi-sample test, the per-sample lengths, and the varlen ranges handed to attention --
        # and the towers carry the trailing padding as one extra segment, which would read as a
        # further sample: a single-sample pack would take the multi-sample path and give the
        # padding its own varlen range.
        und_off = hidden_states.get("_causal_seq_offsets")
        gen_off = hidden_states.get("_full_only_seq_offsets")
        if isinstance(und_off, torch.Tensor):
            und_off = drop_pad_segment(hidden_states, und_off)
        if isinstance(gen_off, torch.Tensor):
            gen_off = drop_pad_segment(hidden_states, gen_off)
        if isinstance(und_off, torch.Tensor) and isinstance(gen_off, torch.Tensor) and und_off.shape != gen_off.shape:
            raise RuntimeError(f"und/gen sample counts disagree: und={und_off.numel() - 1} gen={gen_off.numel() - 1}")
        if isinstance(und_off, torch.Tensor) and isinstance(gen_off, torch.Tensor) and und_off.numel() > 2:
            und_off = und_off.to(dtype=torch.int32)
            gen_off = gen_off.to(dtype=torch.int32)
            und_lens = und_off[1:] - und_off[:-1]
            gen_lens = gen_off[1:] - gen_off[:-1]
            sample_ids = torch.arange(und_lens.numel(), dtype=torch.int64, device=und_off.device)
            und_sample_ids = torch.repeat_interleave(sample_ids, und_lens.to(dtype=torch.int64))
            gen_sample_ids = torch.repeat_interleave(sample_ids, gen_lens.to(dtype=torch.int64))

            # Stable sorting preserves UND-before-GEN within each sample because
            # the source stream is [all UND | all GEN].
            self._kv_reorder_indices = torch.argsort(
                torch.cat((und_sample_ids, gen_sample_ids)),
                stable=True,
            )
            self._cumulative_seqlen_q = gen_off
            self._cumulative_seqlen_kv = und_off + gen_off
            self._max_seqlen_q = int(gen_lens.max())
            self._max_seqlen_kv = int((und_lens + gen_lens).max())
        else:
            # Single-sample: keep the legacy dense path.
            self._kv_reorder_indices = None
            self._cumulative_seqlen_q = None
            self._cumulative_seqlen_kv = None
            self._max_seqlen_q = 0
            self._max_seqlen_kv = 0

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
            kv_reorder_indices=self._kv_reorder_indices,
            cumulative_seqlen_q=self._cumulative_seqlen_q,
            cumulative_seqlen_kv=self._cumulative_seqlen_kv,
            max_seqlen_q=self._max_seqlen_q,
            max_seqlen_kv=self._max_seqlen_kv,
            for_cuda_graphs=False,
            und_only=self.prefill,
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


class InferenceTextKVPrefill:
    """Prepare every CFG branch inside the first generation forward, then swap once."""

    def __init__(
        self,
        net: torch.nn.Module,
        branches: list[tuple[PackedSequence | None, list[UndKVCache] | None]],
        activate: Callable[[str], None],
    ) -> None:
        self._net = net
        self._activate = activate
        self._pending: list[tuple[PackedSequence, list[UndKVCache]]] = []
        for template, cache in branches:
            if cache is not None:
                if template is None:
                    raise ValueError("Text prefill requires a packed template for each CFG branch")
                self._pending.append((template, cache))

    def __call__(self) -> None:
        if not self._pending:
            return
        self._activate("reasoner")
        for template, cache in self._pending:
            prefill_inference_text_kv(self._net, template, cache)
        self._activate("generator")
        self._pending.clear()


@torch.no_grad()
def prefill_inference_text_kv(net: torch.nn.Module, packed_seq: PackedSequence, caches: list[UndKVCache]) -> None:
    """Fill diffusion text caches with the ordinary packed decoder kernels.

    Preserve positions, padding and per-sample attention boundaries, including
    unequal conditional/unconditional prompts in batched CFG. Generation blocks
    are skipped; K/V writes stay outside the compiled decoder layers.
    """
    model = net.language_model.model
    if len(caches) != len(model.layers) or any(cache.is_initialized for cache in caches):
        raise ValueError("Text K/V prefill requires an empty cache for every decoder layer")
    if any(layer.self_attn.k_norm_und_for_gen is not None for layer in model.layers):
        raise ValueError("Text K/V prefill does not support generation-specific text K normalization")
    if packed_seq.text_ids.numel() == 0:
        for layer, cache in zip(model.layers, caches, strict=True):
            attn = layer.self_attn
            empty = model.embed_tokens.weight.new_empty((1, 0, attn.num_key_value_heads, attn.head_dim))
            cache.store(empty, empty)
        return
    hidden = model.embed_tokens(packed_seq.text_ids)
    packed = hidden.new_zeros((packed_seq.sequence_length, hidden.shape[-1]))
    packed[packed_seq.text_indexes] = hidden
    gen_mask = torch.ones(packed_seq.sequence_length, device=hidden.device, dtype=torch.bool)
    gen_mask[packed_seq.text_indexes] = False
    pack, mask, _ = build_packed_sequence(
        "two_way",
        packed_sequence=packed,
        attn_modes=packed_seq.attn_modes,
        split_lens=packed_seq.split_lens,
        sample_lens=packed_seq.sample_lens,
        packed_und_token_indexes=packed_seq.text_indexes,
        packed_gen_token_indexes=gen_mask.nonzero().flatten(),
        num_heads=model.config.num_attention_heads,
        head_dim=model.config.head_dim,
        num_layers=len(model.layers),
        skip_natten_metadata=True,
    )
    # Traverse only the reasoner layers inside the active generation forward.
    # No recursive model forward or generator final norm is needed for K/V.
    positions = packed_seq.position_ids
    cos, sin = model.rotary_emb(
        hidden,
        position_ids=positions.unsqueeze(0) if positions.ndim == 1 else positions.unsqueeze(1),
    )
    position_embeddings = (from_all_seq(cos.squeeze(0), pack), from_all_seq(sin.squeeze(0), pack))
    memory = InferenceTextKVMemoryState(caches, prefill=True)
    memory.init(pack, hidden.device)
    for index, layer in enumerate(model.layers):
        pack, _, kv = layer(
            pack,
            mask,
            position_embeddings,
            memory_value=memory.read_for_layer(index),
        )
        if kv is None:
            raise RuntimeError("Text prefill did not produce K/V for a decoder layer")
        memory.write_for_layer(index, kv)


def _attention_gen_with_cached_text(
    packed_query_states: SequencePack,
    packed_key_states: SequencePack,
    packed_value_states: SequencePack,
    memory_value: InferenceTextKVMemoryValue,
) -> tuple[SequencePack, KVToStore | None]:
    """Gen-only attention attending to cached text K/V plus current gen K/V.

    Single-sample packs use the dense attention API. Multi-sample packs (e.g.
    batched CFG with cond+uncond) rebuild KV in sample-major order
    ``[und_0|gen_0|und_1|gen_1|...]`` and call varlen attention so sample ``i``'s
    gen queries only see sample ``i``'s und+gen keys.
    """
    q_gen = get_gen_seq(packed_query_states)  # [S_curr, H, D]
    k_gen = get_gen_seq(packed_key_states)  # [S_curr, H_kv, D]
    v_gen = get_gen_seq(packed_value_states)  # [S_curr, H_kv, D]

    gen_len = memory_value.gen_len
    k_curr = k_gen[:gen_len].unsqueeze(0)  # [1, S_gen_real, H_kv, D]
    v_curr = v_gen[:gen_len].unsqueeze(0)  # [1, S_gen_real, H_kv, D]
    q_curr = q_gen.unsqueeze(0)  # [1, S_curr, H, D]

    kv_reorder_indices = memory_value.kv_reorder_indices
    cumulative_seqlen_q = memory_value.cumulative_seqlen_q
    cumulative_seqlen_kv = memory_value.cumulative_seqlen_kv
    multi_sample = (
        kv_reorder_indices is not None
        and cumulative_seqlen_q is not None
        and cumulative_seqlen_kv is not None
        and memory_value.und_k_cached is not None
    )

    if not multi_sample:
        kv_parts_k = [k_curr]
        kv_parts_v = [v_curr]
        if memory_value.und_k_cached is not None:
            assert memory_value.und_v_cached is not None
            kv_parts_k.insert(0, memory_value.und_k_cached)
            kv_parts_v.insert(0, memory_value.und_v_cached)

        k_full = torch.cat(kv_parts_k, dim=1)  # [1, S_total, H_kv, D]
        v_full = torch.cat(kv_parts_v, dim=1)  # [1, S_total, H_kv, D]

        attn_result = attention(
            query=q_curr,
            key=k_full,
            value=v_full,
            is_causal=False,
            return_lse=False,
        )
    else:
        assert memory_value.und_v_cached is not None
        assert kv_reorder_indices is not None
        assert cumulative_seqlen_q is not None and cumulative_seqlen_kv is not None
        und_k = memory_value.und_k_cached  # [1, N_und, H_kv, D]
        und_v = memory_value.und_v_cached
        k_full = torch.cat((und_k, k_curr), dim=1).index_select(1, kv_reorder_indices)
        v_full = torch.cat((und_v, v_curr), dim=1).index_select(1, kv_reorder_indices)

        attn_result = attention(
            query=q_curr[:, :gen_len],
            key=k_full,
            value=v_full,
            is_causal=False,
            return_lse=False,
            cumulative_seqlen_Q=cumulative_seqlen_q,
            cumulative_seqlen_KV=cumulative_seqlen_kv,
            max_seqlen_Q=memory_value.max_seqlen_q,
            max_seqlen_KV=memory_value.max_seqlen_kv,
        )

    assert isinstance(attn_result, torch.Tensor)
    gen_out = attn_result.squeeze(0).flatten(-2, -1)  # [S_gen_real, H*D]
    # Match legacy behavior: padded query rows (if any) stay in q_gen but only
    # the real prefix is produced by attention; zero-fill the tail so padded
    # slots stay finite (new_empty can leave NaN/Inf from recycled BF16 memory).
    if gen_out.shape[0] < q_gen.shape[0]:
        padded = q_gen.new_zeros(q_gen.shape[0], gen_out.shape[-1])
        padded[: gen_out.shape[0]] = gen_out
        gen_out = padded

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

    Use reasoner-only attention during prefill and cached text K/V during
    generation. Otherwise, delegate to standard ``dispatch_attention``.
    """
    if isinstance(memory_value, InferenceTextKVMemoryValue) and memory_value.und_only:
        return two_way_attention(packed_query_states, packed_key_states, packed_value_states, und_only=True), None
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
                    "dispatch_attention (the CP path is excluded by its eligibility guard)."
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
