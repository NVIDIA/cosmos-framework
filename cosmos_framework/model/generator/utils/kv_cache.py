# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""
Imaginaire4 Modules:
KV Cache support for efficient sequential/streaming inference.

This module provides utilities for caching key/value tensors across
sequential attention operations, enabling efficient autoregressive
and streaming inference patterns.
"""

from __future__ import annotations

import weakref
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

# Re-exported from memory.py for backward compatibility.
from cosmos_framework.model.generator.utils.memory import KVToStore, MemoryState, MemoryValue
from cosmos_framework.data.generator.sequence_packing.runtime import get_num_real_samples, to_device_nonblocking
from cosmos_framework.configs.base.defaults.replay_attention import TeacherForcingReplayPolicyConfig
from cosmos_framework.model.generator.utils.kv_storage_backend import (
    BF16StorageBackend,
    FP8StorageBackend,
    KVStorageBackend,
)

# See cosmos_framework/data/generator/sequence_packing/ for the canonical definition.
SequencePack = dict[str, Any]


MAX_CACHE_SIZE = 999999


def zero_null_action_values(
    gen_v: torch.Tensor,  # [B,S,H,D]
    vision_token_shapes: list[tuple[int, int, int]],
    num_action_tokens_per_supertoken: int,
    null_action_supertokens: bool,
) -> torch.Tensor:  # [B,S,H,D]
    """Zero value vectors for null action slots while preserving token layout."""
    if not null_action_supertokens or num_action_tokens_per_supertoken == 0:
        return gen_v

    gen_v = gen_v.clone()  # [B,S,H,D]
    starts: list[int] = []
    offset = 0
    for T, H_p, W_p in vision_token_shapes:
        starts.append(offset)
        offset += T * (num_action_tokens_per_supertoken + H_p * W_p)

    device = gen_v.device
    starts_tensor = torch.tensor(starts, device=device, dtype=torch.long)  # [B]
    action_offsets = torch.arange(num_action_tokens_per_supertoken, device=device, dtype=torch.long)  # [A]
    null_positions = (starts_tensor.unsqueeze(1) + action_offsets.unsqueeze(0)).reshape(-1)  # [B*A]
    gen_v[:, null_positions] = 0
    return gen_v  # [B,S,H,D]


class KVBufferPool:
    """Process-lifetime pool of reusable KV workspaces keyed by ``(name, slot)``.

    AR inference creates fresh caches for every generation.  Handing their large
    workspaces (per-layer K/V rings and the static history buffers read by the
    CUDA-graph path) out of one pool keeps the tensors, and therefore their
    addresses, stable across generations.  A stable address is what lets
    ``torch._dynamo.mark_static_address`` tell CUDA-graph trees to read a buffer
    in place instead of copying it into graph-owned inputs on every replay.

    The marking uses ``guard=False``, the same treatment Dynamo gives module
    parameters: a ``guard=True`` mark adds an object-identity guard, and since
    all decoder layers share one compiled code object while each layer owns a
    distinct buffer, that would force one recompile per layer and trip the
    recompile limit.  Without the guard, CUDA-graph trees still select the
    recorded graph by matching static-input addresses (per layer) and the
    tensor shape/dtype guards still cover reallocation.  A buffer is only ever
    reallocated when the requested shape, dtype or device changes, so a live
    address never silently moves.  Contents are never trusted across
    acquisitions: callers rewrite what they read.
    """

    def __init__(self) -> None:
        self._buffers: dict[tuple[str, int], torch.Tensor] = {}
        # Live owner per key so two concurrently active caches can never share a slot silently.
        self._owners: dict[tuple[str, int], weakref.ReferenceType[object]] = {}

    def acquire(
        self,
        name: str,
        slot: int,
        shape: tuple[int, ...],
        *,
        dtype: torch.dtype,
        device: torch.device,
        static_address: bool = False,
        owner: object | None = None,
    ) -> torch.Tensor:
        key = (name, int(slot))
        current_owner_ref = self._owners.get(key)
        current_owner = current_owner_ref() if current_owner_ref is not None else None
        if owner is not None and current_owner is not None and current_owner is not owner:
            raise RuntimeError(
                f"KV buffer {name!r} slot {slot} is still owned by a live cache; concurrent AR generations on one "
                "model need distinct pool slots (release the previous generation's caches first)"
            )
        buffer = self._buffers.get(key)
        if buffer is None or tuple(buffer.shape) != tuple(shape) or buffer.dtype != dtype or buffer.device != device:
            # A plain (non-inference) tensor stays writable in place from both
            # ``torch.inference_mode`` and ``torch.no_grad`` generations.
            with torch.inference_mode(mode=False):
                # Zero-filled on purpose: the varlen FMHA reads each sequence's last 128-token KV tile in
                # full and masks the overrun, so padding past ``cu_seqlens`` must hold finite values
                # (0 x finite = 0, 0 x NaN poisons the row).  One memset per allocation.
                buffer = torch.zeros(tuple(shape), dtype=dtype, device=device)
            if static_address:
                torch._dynamo.mark_static_address(buffer, guard=False)
            self._buffers[key] = buffer
        if owner is not None:
            self._owners[key] = weakref.ref(owner)
        return buffer

    def release_owner(self, owner: object) -> None:
        """Drop ``owner``'s claim on its slots (buffers stay pooled for the next generation)."""
        for key, ref in list(self._owners.items()):
            holder = ref()
            if holder is None or holder is owner:
                del self._owners[key]

    def discard(self, name: str, slot: int) -> None:
        """Forget a buffer so its memory is freed once no cache references it."""
        key = (name, int(slot))
        self._buffers.pop(key, None)
        self._owners.pop(key, None)

    def clear(self) -> None:
        self._buffers.clear()
        self._owners.clear()


class KVCache:
    """Manages key/value caching for efficient sequential attention using a circular buffer.

    This class stores K/V tensors from previous attention operations,
    enabling efficient autoregressive and streaming inference. Caches
    are stored as per-chunk/frame entries in a list-based circular buffer.

    Supports both uniform and variable chunk sizes - each cached entry can have
    a different number of tokens (e.g., for multimodal generation with vision + action tokens).

    Cache semantics:
        - Each cache entry corresponds to one chunk/frame/step.
        - The `cache_size` capacity refers to the number of chunks,
          NOT the number of individual tokens.
        - When `cache_size` is None, cache uses a very large fixed size.
        - When `cache_size` is set, operates as a CIRCULAR BUFFER (ring buffer):
          New entries overwrite oldest entries using modulo arithmetic.
        - When `attention_sink_size` is set, the first `attention_sink_size`
          chunks are pinned and later chunks roll through the remaining slots.
          This provides O(1) storage without array shifting or reallocation.

    Attributes:
        cache_size: Maximum number of chunks to cache (None = very large default).
        backend: Storage backend that encodes/decodes each cached K/V entry
            (defaults to a lossless BF16 passthrough).
        k_cache: List of cached, backend-encoded key entries (one per chunk). Acts as circular buffer.
        v_cache: List of cached, backend-encoded value entries (one per chunk). Acts as circular buffer.

    Important Properties:
        - **Gradient-free**: Cached K/V are detached, preventing gradient flow through cache.
        - **Activation checkpointing compatible**: Recomputation produces identical results.
        - **Training safe**: Works correctly with PyTorch gradient checkpointing.
        - **Explicit frame_idx**: Frame index passed as explicit parameter for clarity.

    Example:
        >>> cache = KVCache(cache_size=16)
        >>> # Frame 0: Store first frame — k0, v0 are [B,S,H,D]
        >>> cache.store_kv(k0, v0, frame_idx=0)
        >>>
        >>> # Frame 5: Fetch history [B,S_hist,H,D] and store current
        >>> k_hist, v_hist = cache.fetch_kv(frame_idx=5)  # [B,S_hist,H,D]
        >>> k_with_history = torch.cat([k_hist, k5], dim=1)  # [B,S_hist+S5,H,D]
        >>> cache.store_kv(k5, v5, frame_idx=5)

        Circular buffer example (cache_size=2):
        >>> cache = KVCache(cache_size=2)
        >>> # Frame 0 → cache[0], Frame 1 → cache[1]
        >>> # Frame 2 → cache[0] (overwrites Frame 0), Frame 3 → cache[1] (overwrites Frame 1)
    """

    def __init__(
        self,
        cache_size: int | None = None,
        backend: KVStorageBackend | None = None,
        attention_sink_size: int = 0,
    ) -> None:
        """Initialize KV cache with a fixed chunk capacity.

        Args:
            cache_size: Maximum number of frames to cache. Must be >= 2.
                        None uses very large default (999999).
            backend: Storage backend used to encode/decode each cached K/V
                entry. Defaults to ``BF16StorageBackend`` (a lossless
                detach-and-clone passthrough), which stores tensors exactly as
                a plain detached clone would.
            attention_sink_size: Number of initial frames to pin in cache.

        Raises:
            ValueError: If cache_size < 2, if attention_sink_size < 0, if
                attention_sink_size is set with an unbounded cache, or if
                attention_sink_size >= cache_size.
        """
        if cache_size is not None and cache_size < 2:
            raise ValueError(f"cache_size must be >= 2 to support history, got {cache_size}")
        if attention_sink_size < 0:
            raise ValueError(f"attention_sink_size must be >= 0, got {attention_sink_size}")
        if cache_size is None and attention_sink_size != 0:
            raise ValueError("attention_sink_size must be 0 when cache_size is None")
        if cache_size is not None and attention_sink_size >= cache_size:
            raise ValueError(
                f"attention_sink_size must be less than cache_size, got {attention_sink_size}>={cache_size}"
            )
        self.cache_size = MAX_CACHE_SIZE if cache_size is None else cache_size
        self.attention_sink_size = attention_sink_size
        self.backend: KVStorageBackend = backend if backend is not None else BF16StorageBackend()
        self.reset()

    def reset(self) -> None:
        """Reset cache state while keeping capacity."""
        self.k_cache: list[object | None] = [None] * self.cache_size
        self.v_cache: list[object | None] = [None] * self.cache_size
        self.backend.reset_kv_cache_state(self.cache_size)

    def _cache_index(self, frame_idx: int) -> int:
        """Map a logical frame index to its physical cache slot."""
        if self.attention_sink_size == 0:
            return frame_idx % self.cache_size
        if frame_idx < self.attention_sink_size:
            return frame_idx
        rolling_size = self.cache_size - self.attention_sink_size
        return self.attention_sink_size + ((frame_idx - self.attention_sink_size) % rolling_size)

    def _history_frame_indices(self, frame_idx: int) -> list[int]:
        """Return logical history frames in attention order for ``frame_idx``."""
        current_idx = int(frame_idx)
        if current_idx <= 0:
            return []
        if self.attention_sink_size == 0:
            start_idx = max(0, current_idx - self.cache_size + 1)
            return list(range(start_idx, current_idx))

        sink_end = min(self.attention_sink_size, current_idx)
        sink_indices = list(range(sink_end))
        rolling_history_size = self.cache_size - self.attention_sink_size - 1
        rolling_start = max(self.attention_sink_size, current_idx - rolling_history_size)
        rolling_indices = list(range(rolling_start, current_idx))
        return sink_indices + rolling_indices

    def store_kv(self, k: torch.Tensor, v: torch.Tensor, frame_idx: int) -> None:
        """Store K/V tensors into cache at the specified frame index.

        Tensor layout is BSHD (batch-first, heads-last):
            k, v: [B,S,H,D]
                - B: batch size
                - S: tokens in this frame; variable across frames (e.g. vision vs. action chunks)
                - H: number of attention heads
                - D: per-head dimension (head_dim)

        Args:
            k: Current key tensor [B,S,H,D].
            v: Current value tensor [B,S,H,D].
            frame_idx: Frame index where to store the K/V tensors.

        Note:
            - K/V are encoded by the storage backend before caching; the
              default backend detaches and clones, preventing gradient flow.
            - Uses circular buffer with modulo arithmetic.
        """
        # CIRCULAR BUFFER: use modulo to wrap index into the fixed-size buffer.
        # With attention sinks enabled, the pinned prefix is never overwritten
        # and only the suffix rolls.
        index = self._cache_index(int(frame_idx))

        # Encode through the storage backend, which owns how each entry is
        # stored.  The backend must return a representation that is detached
        # and backed by its own memory, preserving two properties that direct
        # tensor storage relied on:
        # - Detach prevents gradients flowing through the cache.  CRITICAL for
        #   activation checkpointing: recomputation during the backward pass
        #   then produces identical cached results.
        # - Owning its memory escapes CUDA-graph-managed storage: tensors
        #   produced inside a torch.compile(mode="reduce-overhead") region live
        #   in the CUDA graph's reusable memory pool and get overwritten on the
        #   next replay.  Storing a live reference would corrupt the cache on
        #   the next frame.
        # The default BF16 backend satisfies both via k.detach().clone().
        k_entry = self.backend.encode(k)  # [B,S,H,D]
        v_entry = self.backend.encode(v)  # [B,S,H,D]
        self.k_cache[index] = k_entry
        self.v_cache[index] = v_entry
        # For triton kernel, keep backend-side metadata aligned with the physical ring-buffer slot.
        self.backend.update_cached_kv_metadata(index, k_entry, v_entry)

    def fetch_kv(self, frame_idx: int) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
        """Fetch cached K/V history up to (but excluding) the specified frame index.

        Args:
            frame_idx: Current frame index. Returns history from start_idx to frame_idx-1.

        Returns:
            (k_history, v_history): cached tensors concatenated along the sequence dimension.
                k_history: [B,S_total,H,D]  where S_total = sum of S across cached frames
                v_history: [B,S_total,H,D]
            Returns (None, None) if frame_idx <= 0 or no history available.

        Raises:
            AssertionError: If cache contains None entries in the requested range.

        Note:
            - This method is read-only: it does not mutate any cache state, making it
              safe to call inside a torch.compile(fullgraph=True) region.
            - For circular buffer, maps logical frame indices to physical buffer indices.
        """
        if frame_idx <= 0:
            return None, None

        current_idx = int(frame_idx)

        # CIRCULAR BUFFER RETRIEVAL: map logical frame indices to physical
        # buffer indices. With attention sinks enabled, this returns pinned
        # sink frames first, followed by the chronological rolling tail.
        history_indices = [self._cache_index(i) for i in self._history_frame_indices(current_idx)]
        history_k_entries: list[object] = []
        history_v_entries: list[object] = []
        for cache_idx in history_indices:
            k_entry = self.k_cache[cache_idx]
            v_entry = self.v_cache[cache_idx]
            if k_entry is None:
                raise AssertionError(f"K cache contains None entries for frame_idx={current_idx}")
            if v_entry is None:
                raise AssertionError(f"V cache contains None entries for frame_idx={current_idx}")
            history_k_entries.append(k_entry)
            history_v_entries.append(v_entry)

        # Concatenate along sequence dimension (dim=1): S_total = S_0 + S_1 + ... + S_{frame_idx-1}
        # S can differ per frame (e.g. variable vision/action token counts)
        # B, H, D must match across all frames
        k_history, v_history = self.backend.decode_many(
            history_k_entries,
            history_v_entries,
            slots=history_indices,
        )  # [B,S_total,H,D] each
        return k_history, v_history


class UndKVCache:
    """Fixed cache for understanding (text) tokens.

    Unlike GenKVCache, this stores K/V for und tokens only once (at frame 0)
    and reuses them for all subsequent frames.  Does not use a circular buffer
    since und tokens generally remain constant throughout generation.  Calls
    to store() will overwrite the existing caption.

    Attributes:
        k_und: Cached key tensor for und tokens [B,S_und,H,D].
        v_und: Cached value tensor for und tokens [B,S_und,H,D].
        cached_len: Number of real (non-padding) und tokens stored.
        is_initialized: Whether the cache has been populated.

    Example:
        >>> und_cache = UndKVCache()
        >>> # Frame 0: Store und K/V after RoPE
        >>> und_cache.store(k_und_with_rope, v_und)
        >>> # Frame 1+: Retrieve cached und K/V
        >>> k_und, v_und = und_cache.get()
    """

    def __init__(self):
        """Initialize empty und cache."""
        self.k_und: torch.Tensor | None = None
        self.v_und: torch.Tensor | None = None
        self.cached_len: int = 0
        self.cached_lens: tuple[int, ...] = ()
        self.is_initialized = False

    def store(
        self,
        k: torch.Tensor,  # [B,S_und,H,D]
        v: torch.Tensor,  # [B,S_und,H,D]
        lengths: tuple[int, ...] | None = None,
    ) -> None:
        """Store und K/V tensors.

        The new caption will overwrite the existing caption in the cache.

        Args:
            k: Key tensor with RoPE applied [B,S_und,H,D].
                S_und is the number of understanding (text) tokens.
            v: Value tensor [B,S_und,H,D].
            lengths: Real text-token length for every batch row. When omitted,
                every row is assumed to occupy the full sequence dimension.
        """
        if lengths is None:
            lengths = (k.shape[1],) * k.shape[0]
        if len(lengths) != k.shape[0]:
            raise ValueError(f"Expected {k.shape[0]} understanding lengths, got {len(lengths)}")
        if any(length < 0 or length > k.shape[1] for length in lengths):
            raise ValueError(f"Understanding lengths {lengths} exceed cached tensor shape {tuple(k.shape)}")
        # Detach to prevent gradient flow (same as KVCache).
        # Clone to escape CUDA-graph-managed storage (see KVCache.store_kv).
        self.k_und = k.detach().clone()  # [B,S_und,H,D]
        self.v_und = v.detach().clone()  # [B,S_und,H,D]
        self.cached_len = k.shape[1]
        self.cached_lens = lengths
        self.is_initialized = True

    def get(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieve cached und K/V tensors.

        Returns:
            Tuple of (k_und, v_und): [B,S_und,H,D] each.

        Raises:
            AssertionError: If cache not initialized before retrieval.
        """
        if not self.is_initialized or self.k_und is None or self.v_und is None:
            raise AssertionError("UndKVCache not initialized - must call store() at frame 0")
        return self.k_und, self.v_und

    def get_padded(
        self,
        max_len: int,
        *,
        num_heads: int = 0,
        head_dim: int = 0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cached K/V zero-padded to max_len on the sequence dimension.

        Used by the branchless KV cache training path to produce fixed-shape
        tensors that don't trigger torch.compile recompilation.

        When the cache is not yet initialized (first segment), returns all-zero
        tensors of the correct shape. The branchless ``torch.where`` in
        attention will select the live values instead, so these zeros are never
        used for actual computation.

        Args:
            max_len: Target sequence length to pad to.
            num_heads: Number of KV heads (required if cache not initialized).
            head_dim: Per-head dimension (required if cache not initialized).
            device: Device for zero tensors (required if cache not initialized).
            dtype: Dtype for zero tensors (required if cache not initialized).

        Returns:
            (k_padded, v_padded): each [1, max_len, H, D].
        """
        if not self.is_initialized or self.k_und is None or self.v_und is None:
            return (
                torch.zeros(1, max_len, num_heads, head_dim, device=device, dtype=dtype),
                torch.zeros(1, max_len, num_heads, head_dim, device=device, dtype=dtype),
            )
        if self.cached_len > max_len:
            raise ValueError(f"cached und length {self.cached_len} exceeds requested max_len {max_len}")
        pad_amount = max_len - self.cached_len
        return (
            F.pad(self.k_und, (0, 0, 0, 0, 0, pad_amount)),
            F.pad(self.v_und, (0, 0, 0, 0, 0, pad_amount)),
        )

    def reset(self) -> None:
        """Reset cache to empty state."""
        self.k_und = None
        self.v_und = None
        self.cached_len = 0
        self.cached_lens = ()
        self.is_initialized = False


class GenKVCache(KVCache):
    """Rolling cache for generation (vision + action) tokens.

    Inherits from KVCache with circular buffer for rolling window.
    Each cache entry corresponds to one frame's gen tokens.

    GenKVCache is the rolling generation-token cache with a circular
    buffer as the source of truth for per-frame K/V and supports O(1)
    writes as the rolling window advances.
    Some inference paths like CUDA-graph AR need a fixed shape,
    chronologically ordered view of that rolling history.
    The static buffers provide an optimized read-side view for such paths.

    Example:
        >>> gen_cache = GenKVCache(cache_size=16)
        >>> # Frame 0: Store — k0, v0 are [B,S0,H,D]
        >>> gen_cache.store_kv(k0, v0, frame_idx=0)
        >>> # Frame 1: Fetch history, build full context, store — k1 is [B,S1,H,D]
        >>> k_hist, v_hist = gen_cache.fetch_kv(frame_idx=1)  # [B,S0,H,D] or None
        >>> k1_full = torch.cat([k_hist, k1], dim=1) if k_hist is not None else k1  # [B,S0+S1,H,D]
        >>> gen_cache.store_kv(k1, v1, frame_idx=1)
    """

    def __init__(
        self,
        cache_size: int | None = None,
        backend: KVStorageBackend | None = None,
        attention_sink_size: int = 0,
        *,
        buffer_pool: KVBufferPool | None = None,
        pool_slot: int = 0,
        preallocate_ring: bool = False,
    ) -> None:
        """Args beyond ``KVCache``:
        buffer_pool: Optional pool that owns this cache's large workspaces so
            they are reused across generations (stable addresses).
        pool_slot: Pool key for this cache (one per layer and CFG branch).
        preallocate_ring: Store entries in place into one preallocated ring
            tensor per K and V (no per-entry clone, no allocator churn).
            Only used with a finite cache, the BF16 backend and uniform entry
            shapes; otherwise the per-entry clone path is kept.
        """
        self._buffer_pool = buffer_pool
        self._pool_slot = int(pool_slot)
        self._preallocate_ring = preallocate_ring
        self._ring_k: torch.Tensor | None = None
        self._ring_v: torch.Tensor | None = None
        self._ring_disabled = False
        # initialize static KV cache used or AR inference use case.
        # This buffer hosts materialization of the rolling window in chronological order.
        self._static_k_buf: torch.Tensor | None = None
        self._static_v_buf: torch.Tensor | None = None
        self._static_valid_frame_idx: int | None = None
        self._static_real_len: int = 0
        self._static_tokens_per_frame: int = 0
        # The ``history_frame_indices`` request behind the current buffer contents (None = ring-native).
        self._static_history_indices: tuple[int, ...] | None = None
        # Composite layout: ``prefix_tokens`` = und + current-frame tokens ahead of the history region,
        # and the identity of the und K/V last copied into the und region (None = not primed).
        self._static_prefix_tokens: int = 0
        self._static_batch_rows: int = 1
        self._static_und_key: tuple | None = None
        self._static_real_lens: tuple[int, ...] = ()
        # Per-row episode restarts (batched Transfer): the global cache index at which row r last
        # restarted, or None when no row ever did.  A restarted row keeps storing into the shared
        # global slots but reads a fresh-cache history plan expressed in its local frame count, with
        # its first ``attention_sink_size`` local entries mirrored into the pinned sink slots of its
        # own ring row (see ``row_history_slots``).
        self._row_reset_frame: list[int] | None = None
        super().__init__(cache_size=cache_size, backend=backend, attention_sink_size=attention_sink_size)

    def reset(self) -> None:
        """Reset cache state, discard static inference workspaces and release pooled slots."""
        super().reset()
        self._static_k_buf = None
        self._static_v_buf = None
        self._static_valid_frame_idx = None
        self._static_real_len = 0
        self._static_tokens_per_frame = 0
        self._static_history_indices = None
        self._static_prefix_tokens = 0
        self._static_batch_rows = 1
        self._static_und_key = None
        self._static_real_lens = ()
        self._row_reset_frame = None
        self._ring_k = None
        self._ring_v = None
        self._ring_disabled = False
        if self._buffer_pool is not None:
            self._buffer_pool.release_owner(self)

    def store_kv(self, k: torch.Tensor, v: torch.Tensor, frame_idx: int) -> None:
        """Store K/V and invalidate static read workspaces."""
        if not self._store_kv_in_ring(k, v, frame_idx):
            super().store_kv(k, v, frame_idx)
        # A store can change the chronological history for the next AR frame,
        # and can also overwrite a physical ring-buffer slot that the static
        # buffer previously copied from.  Mark the workspace stale.
        # refresh happens once on the next static read.
        self._static_valid_frame_idx = None

    def _store_kv_in_ring(self, k: torch.Tensor, v: torch.Tensor, frame_idx: int) -> bool:
        """Write ``k``/``v`` in place into the preallocated ring; ``False`` = use the clone path."""
        if not self._preallocate_ring or self._ring_disabled:
            return False
        if self.cache_size >= MAX_CACHE_SIZE or not isinstance(self.backend, BF16StorageBackend):
            return False
        if k.shape != v.shape or k.dtype != v.dtype or k.device != v.device:
            return False
        ring_shape = (self.cache_size, *k.shape)
        if self._ring_k is None:
            if self._buffer_pool is not None:
                self._ring_k = self._buffer_pool.acquire(
                    "ring_k", self._pool_slot, ring_shape, dtype=k.dtype, device=k.device, owner=self
                )
                self._ring_v = self._buffer_pool.acquire(
                    "ring_v", self._pool_slot, ring_shape, dtype=k.dtype, device=k.device, owner=self
                )
            else:
                with torch.inference_mode(mode=False):
                    self._ring_k = torch.empty(ring_shape, dtype=k.dtype, device=k.device)  # [slots,B,S,H,D]
                    self._ring_v = torch.empty(ring_shape, dtype=k.dtype, device=k.device)  # [slots,B,S,H,D]
        assert self._ring_k is not None and self._ring_v is not None
        if tuple(self._ring_k.shape) != ring_shape or self._ring_k.dtype != k.dtype or self._ring_k.device != k.device:
            # Variable-size entries: keep the already-stored views (they hold the
            # ring alive) and fall back to per-entry clones for the rest of this
            # cache's lifetime.  Drop the ring from the pool so it is not pinned
            # for later generations that will never use it either.
            self._ring_disabled = True
            if self._buffer_pool is not None:
                self._buffer_pool.discard("ring_k", self._pool_slot)
                self._buffer_pool.discard("ring_v", self._pool_slot)
            return False
        index = self._cache_index(int(frame_idx))
        # In-place copy: detached from autograd and independent of the source
        # storage (which may live in a CUDA-graph pool), same as the clone path.
        self._ring_k[index].copy_(k.detach())
        self._ring_v[index].copy_(v.detach())
        self._mirror_reset_row_sinks(index, int(frame_idx))
        k_entry = self._ring_k[index]  # [B,S,H,D] view into the ring
        v_entry = self._ring_v[index]  # [B,S,H,D] view into the ring
        self.k_cache[index] = k_entry
        self.v_cache[index] = v_entry
        self.backend.update_cached_kv_metadata(index, k_entry, v_entry)
        return True

    def transfer_history_frame_indices(
        self,
        frame_idx: int,
        sink_entries: int,
        max_recent_entries: int,
    ) -> list[int]:
        """Return the logical cache entries a Transfer forward at ``frame_idx`` may attend to.

        Entry-level mirror of the token slicing in ``ARMemoryState.read_for_layer``:
        the first ``sink_entries`` entries of the ring history stay pinned (complete
        logical Transfer sink frames, one control + one RGB entry each) and only the
        newest ``max_recent_entries`` of the remaining history are kept.  Control seeds
        pass ``2 * recent`` and target forwards ``2 * recent + 1`` so that ``R_t`` also
        sees its aligned ``C_t``.  Requires every cached entry to hold the same token
        count (framewise Transfer), which the static AR path asserts when it copies.
        """
        if sink_entries < 0:
            raise ValueError(f"sink_entries must be >= 0, got {sink_entries}")
        if max_recent_entries < 0:
            raise ValueError(f"max_recent_entries must be >= 0, got {max_recent_entries}")
        history = self._history_frame_indices(int(frame_idx))
        sink = history[:sink_entries]
        rest = history[sink_entries:]
        recent = rest[-max_recent_entries:] if max_recent_entries > 0 else []
        return sink + recent

    def reset_rows(self, rows: Sequence[int], at_frame_idx: int) -> None:
        """Restart the episodes of ``rows`` at global cache index ``at_frame_idx`` (batched Transfer).

        The row keeps storing into the shared global ring slots (all rows step in lockstep) but its
        visible history becomes that of a fresh cache whose logical frame ``l`` is global
        ``at_frame_idx + l``; its first ``attention_sink_size`` entries are additionally mirrored
        into the pinned sink slots of its own ring row so the fresh-cache sink survives eviction.
        Only tensor values change afterwards, so captured CUDA graphs keep replaying.
        """
        if not self._preallocate_ring or self._ring_disabled or self.cache_size is None:
            raise RuntimeError("per-row resets require the in-place K/V ring (finite cache, BF16 storage)")
        if self._ring_k is not None:
            batch_rows = int(self._ring_k.shape[1])
            for row in rows:
                if row < 0 or row >= batch_rows:
                    raise IndexError(f"reset row {row} outside [0, {batch_rows})")
        if self._row_reset_frame is None:
            self._row_reset_frame = []
        for row in rows:
            while len(self._row_reset_frame) <= row:
                self._row_reset_frame.append(0)
            self._row_reset_frame[row] = int(at_frame_idx)
        self._static_valid_frame_idx = None

    def row_reset_frame(self, row: int) -> int:
        """Global cache index at which ``row`` last restarted (0 = never)."""
        if self._row_reset_frame is None or row >= len(self._row_reset_frame):
            return 0
        return self._row_reset_frame[row]

    @property
    def has_row_resets(self) -> bool:
        return self._row_reset_frame is not None and any(self._row_reset_frame)

    def _mirror_reset_row_sinks(self, slot: int, frame_idx: int) -> None:
        """After storing global entry ``frame_idx`` at ``slot``, copy restarted rows' early entries into their sink slots."""
        if self._row_reset_frame is None or self._ring_k is None or self._ring_v is None:
            return
        for row, reset_frame in enumerate(self._row_reset_frame):
            if reset_frame <= 0:
                continue
            local = frame_idx - reset_frame
            if 0 <= local < self.attention_sink_size and local != slot:
                self._ring_k[local, row].copy_(self._ring_k[slot, row])
                self._ring_v[local, row].copy_(self._ring_v[slot, row])

    def row_history_slots(self, row: int, frame_idx: int, sink_entries: int, max_recent_entries: int) -> list[int]:
        """Physical ring slots row ``row`` may attend to at global ``frame_idx`` (Transfer visibility).

        Without a restart this is ``transfer_history_frame_indices`` mapped through ``_cache_index``.
        After a restart the plan is that of a fresh cache at the row's local frame: its sink entries
        live in the pinned sink slots (mirrored at store time), the rest at the global slots.
        """
        reset_frame = self.row_reset_frame(row)
        local_frame = int(frame_idx) - reset_frame
        if local_frame < 0:
            raise ValueError(f"row {row} restarted at {reset_frame}, after frame {frame_idx}")
        slots: list[int] = []
        for logical_local in self.transfer_history_frame_indices(local_frame, sink_entries, max_recent_entries):
            if reset_frame > 0 and logical_local < self.attention_sink_size:
                slots.append(logical_local)  # mirrored sink slot of this row
            else:
                slots.append(self._cache_index(reset_frame + logical_local))
        return slots

    def _ensure_static_history_buffer_allocated(
        self,
        max_tokens: int,
        *,
        num_heads: int,
        head_dim: int,
        device: torch.device | None,
        dtype: torch.dtype | None,
        prefix_tokens: int = 0,
        batch_rows: int = 1,
    ) -> None:
        """Allocate the fixed-size K/V buffer used by CUDA Graph AR inference.

        ``prefix_tokens`` reserves ``[und | curr]`` room ahead of the ``max_tokens`` history
        region so the compiled block reads one contiguous ``[und | curr | hist | pad]`` buffer.
        ``batch_rows`` rows of that layout are laid end to end with a fixed row stride
        ``prefix_tokens + max_tokens`` (one row per batched sample; ``B=1`` is today's buffer).
        """
        if device is None:
            raise ValueError("device is required when allocating static gen KV buffers")
        if dtype is None:
            raise ValueError("dtype is required when allocating static gen KV buffers")
        if batch_rows < 1:
            raise ValueError(f"batch_rows must be >= 1, got {batch_rows}")

        expected_shape = (1, batch_rows * (prefix_tokens + max_tokens), num_heads, head_dim)
        needs_alloc = (
            self._static_k_buf is None
            or self._static_v_buf is None
            or self._static_prefix_tokens != prefix_tokens
            or self._static_batch_rows != batch_rows
            or tuple(self._static_k_buf.shape) != expected_shape
            or tuple(self._static_v_buf.shape) != expected_shape
            or self._static_k_buf.device != device
            or self._static_v_buf.device != device
            or self._static_k_buf.dtype != dtype
            or self._static_v_buf.dtype != dtype
        )
        if not needs_alloc:
            return

        if self._buffer_pool is not None:
            # Pooled + static address: the compiled AR block reads these in place
            # on every CUDA-graph replay instead of copying them into graph inputs.
            self._static_k_buf = self._buffer_pool.acquire(
                "static_k",
                self._pool_slot,
                expected_shape,
                dtype=dtype,
                device=device,
                static_address=True,
                owner=self,
            )  # [1,S_max,H,D]
            self._static_v_buf = self._buffer_pool.acquire(
                "static_v",
                self._pool_slot,
                expected_shape,
                dtype=dtype,
                device=device,
                static_address=True,
                owner=self,
            )  # [1,S_max,H,D]
        else:
            with torch.inference_mode(mode=False):
                # Zero-filled: the attention kernel reads past the real prefix within the last KV tile.
                self._static_k_buf = torch.zeros(expected_shape, device=device, dtype=dtype)  # [1,S_max,H,D]
                self._static_v_buf = torch.zeros(expected_shape, device=device, dtype=dtype)  # [1,S_max,H,D]
        self._static_valid_frame_idx = None
        self._static_real_len = 0
        self._static_tokens_per_frame = 0
        self._static_history_indices = None
        self._static_prefix_tokens = prefix_tokens
        self._static_batch_rows = batch_rows
        self._static_und_key = None

    def _first_cached_history_k(self, frame_idx: int) -> torch.Tensor | None:
        """Return the first cached K tensor that will contribute to ``frame_idx``."""
        current_idx = int(frame_idx)
        if current_idx <= 0:
            return None

        for logical_idx in self._history_frame_indices(current_idx):
            cache_idx = self._cache_index(logical_idx)
            # Stored entries use the backend's internal format (BF16: a plain
            # tensor; FP8: a (fp8, scale) tuple), so decode back to a tensor.
            k_entry = self.k_cache[cache_idx]
            if k_entry is not None:
                return self.backend.decode(k_entry)  # [B,S_frame,H,D]
        return None

    def _rebuild_static_history(
        self,
        frame_idx: int,
        max_tokens: int,
        tokens_per_frame: int,
        history_frame_indices: list[int] | None = None,
        row_slot_lists: list[list[int]] | None = None,
    ) -> int:
        """Rebuild the fixed history buffer from the list-backed circular cache.

        ``history_frame_indices`` overrides the ring-native chronological history
        with an explicit list of logical entries (Transfer sink + recent selection).
        ``row_slot_lists`` gives every buffer row its own physical slot list instead
        (batched Transfer with per-row episode restarts); rows may then differ in length.
        """
        assert self._static_k_buf is not None
        assert self._static_v_buf is not None
        if self._static_k_buf.is_cuda and torch.cuda.is_current_stream_capturing():
            # The copies below read the ring slots of *this* frame; recorded into a graph they
            # would replay the same slots for every later frame.  Callers refresh the buffer
            # before capture/replay (``prepare_for_coarse_cuda_graph_replay``).
            raise RuntimeError("static AR history rebuild requested inside a CUDA graph capture")

        current_idx = int(frame_idx)
        if row_slot_lists is not None:
            return self._rebuild_static_history_rows(current_idx, max_tokens, tokens_per_frame, row_slot_lists)
        requested_indices = None if history_frame_indices is None else tuple(history_frame_indices)
        logical_indices = (
            list(history_frame_indices)
            if history_frame_indices is not None
            else (self._history_frame_indices(current_idx) if current_idx > 0 else [])
        )
        if not logical_indices:
            # No visible gen history.  The returned buffer may contain
            # uninitialized tail data, but real_len=0 and the static AR path
            # will not expose any history tokens to attention.
            self._static_valid_frame_idx = current_idx
            self._static_real_len = 0
            self._static_tokens_per_frame = tokens_per_frame
            self._static_history_indices = requested_indices
            return 0

        # Chronological logical frames in history.  Without an override this
        # mirrors ``fetch_kv`` exactly, including the cache_size-1 history limit
        # that leaves one slot available for the current frame after it is stored.
        history_indices: list[int] = []
        history_k_entries: list[object] = []
        history_v_entries: list[object] = []
        for logical_idx in logical_indices:
            # Map logical frame indices to ring-buffer slots.  At wraparound,
            # this copies frames in logical order even though physical storage
            # is no longer contiguous.
            cache_idx = self._cache_index(logical_idx)
            # Stored entries use the backend's internal format (BF16: a plain
            # tensor; FP8: a (fp8, scale) tuple), so decode them together below.
            k_entry = self.k_cache[cache_idx]
            v_entry = self.v_cache[cache_idx]
            if k_entry is None:
                raise AssertionError(f"K cache contains None entry for frame_idx={current_idx}")
            if v_entry is None:
                raise AssertionError(f"V cache contains None entry for frame_idx={current_idx}")
            k_frame = self.backend.decode(k_entry)  # [B,S_frame,H,D]
            v_frame = self.backend.decode(v_entry)  # [B,S_frame,H,D]
            if k_frame.shape[1] != tokens_per_frame or v_frame.shape[1] != tokens_per_frame:
                raise AssertionError(
                    f"Static AR cache requires fixed frame tokens: expected {tokens_per_frame}, "
                    f"got k={k_frame.shape[1]}, v={v_frame.shape[1]} at logical frame {logical_idx}"
                )
            history_indices.append(cache_idx)
            history_k_entries.append(k_entry)
            history_v_entries.append(v_entry)

        dst_end = len(history_indices) * tokens_per_frame
        if dst_end > max_tokens:
            raise AssertionError(f"Static AR cache overflow: trying to write {dst_end} tokens into {max_tokens}")
        # Row r's history region starts at ``r*R + prefix`` (R = row stride); only the real
        # chronological prefix of it is written.  The padded suffix is deliberately left
        # untouched (zero-filled at allocation) and excluded by ``cu_seqlens_kv_t``.
        rows = self._static_batch_rows
        row_stride = self._static_prefix_tokens + max_tokens
        hist_start = self._static_prefix_tokens
        if self._ring_slot_runs_copyable(history_indices, history_k_entries, history_v_entries):
            # Single pass straight from the ring ``[slots,B,S,H,D]``: each run of consecutive
            # physical slots is one copy per row.  Same bytes in the same order as concatenating
            # the entries; at B=1 the source run is contiguous.
            assert self._ring_k is not None and self._ring_v is not None
            _, _, _, num_heads, head_dim = self._ring_k.shape
            for row in range(rows):
                offset = row * row_stride + hist_start
                for slot_start, slot_end in self._coalesce_slot_runs(history_indices):
                    run_entries = slot_end - slot_start
                    run_tokens = run_entries * tokens_per_frame
                    self._static_k_buf[0, offset : offset + run_tokens].view(
                        run_entries, tokens_per_frame, num_heads, head_dim
                    ).copy_(self._ring_k[slot_start:slot_end, row])  # [n,S,H,D]
                    self._static_v_buf[0, offset : offset + run_tokens].view(
                        run_entries, tokens_per_frame, num_heads, head_dim
                    ).copy_(self._ring_v[slot_start:slot_end, row])  # [n,S,H,D]
                    offset += run_tokens
        else:
            # Generic path (FP8 entries, clone storage, ragged rings): decode the whole
            # chronological window at once, then copy each row into its history region.
            k_history, v_history = self.backend.decode_many(
                history_k_entries,
                history_v_entries,
                slots=history_indices,
            )  # [B,S_hist,H,D] each
            if k_history.shape[1] != dst_end or v_history.shape[1] != dst_end:
                raise AssertionError(
                    f"Static AR cache requires {tokens_per_frame} tokens per frame: "
                    f"expected total={dst_end}, got k={k_history.shape[1]}, v={v_history.shape[1]}"
                )
            if k_history.shape[0] != rows:
                raise AssertionError(f"Static AR cache holds {rows} rows but the history has {k_history.shape[0]}")
            for row in range(rows):
                offset = row * row_stride + hist_start
                self._static_k_buf[0, offset : offset + dst_end].copy_(k_history[row].detach())  # [S_hist,H,D]
                self._static_v_buf[0, offset : offset + dst_end].copy_(v_history[row].detach())  # [S_hist,H,D]

        self._static_valid_frame_idx = current_idx
        self._static_real_len = dst_end
        self._static_real_lens = (dst_end,) * rows
        self._static_tokens_per_frame = tokens_per_frame
        # Remember the *request* (None = ring-native) so a later call with a different
        # request at the same frame rebuilds, including switching back to native.
        self._static_history_indices = requested_indices
        return dst_end

    def _rebuild_static_history_rows(
        self,
        current_idx: int,
        max_tokens: int,
        tokens_per_frame: int,
        row_slot_lists: list[list[int]],
    ) -> int:
        """Per-row rebuild: row r copies its own physical slots (single pass from the ring)."""
        assert self._static_k_buf is not None and self._static_v_buf is not None
        rows = self._static_batch_rows
        if len(row_slot_lists) != rows:
            raise AssertionError(f"{len(row_slot_lists)} slot lists for {rows} rows")
        if self._ring_k is None or self._ring_v is None or self._ring_disabled or self._ring_k.shape[1] != rows:
            raise RuntimeError("per-row static history requires the in-place K/V ring with one ring row per buffer row")
        _, _, ring_tokens, num_heads, head_dim = self._ring_k.shape
        if ring_tokens != tokens_per_frame:
            raise AssertionError(f"ring entries hold {ring_tokens} tokens, expected {tokens_per_frame}")
        row_stride = self._static_prefix_tokens + max_tokens
        real_lens: list[int] = []
        for row, slots in enumerate(row_slot_lists):
            for slot in slots:
                if self.k_cache[slot] is None or self.v_cache[slot] is None:
                    raise AssertionError(f"K/V cache slot {slot} is empty (row {row}, frame_idx={current_idx})")
            dst_end = len(slots) * tokens_per_frame
            if dst_end > max_tokens:
                raise AssertionError(f"Static AR cache overflow: trying to write {dst_end} tokens into {max_tokens}")
            offset = row * row_stride + self._static_prefix_tokens
            for slot_start, slot_end in self._coalesce_slot_runs(list(slots)):
                run_entries = slot_end - slot_start
                run_tokens = run_entries * tokens_per_frame
                self._static_k_buf[0, offset : offset + run_tokens].view(
                    run_entries, tokens_per_frame, num_heads, head_dim
                ).copy_(self._ring_k[slot_start:slot_end, row])  # [n,S,H,D]
                self._static_v_buf[0, offset : offset + run_tokens].view(
                    run_entries, tokens_per_frame, num_heads, head_dim
                ).copy_(self._ring_v[slot_start:slot_end, row])  # [n,S,H,D]
                offset += run_tokens
            real_lens.append(dst_end)
        self._static_valid_frame_idx = current_idx
        self._static_real_len = real_lens[0] if real_lens else 0
        self._static_real_lens = tuple(real_lens)
        self._static_tokens_per_frame = tokens_per_frame
        self._static_history_indices = tuple(tuple(slots) for slots in row_slot_lists)
        return self._static_real_len

    def _ring_slot_runs_copyable(
        self,
        slots: list[int],
        k_entries: list[object],
        v_entries: list[object],
    ) -> bool:
        """True when every entry is the ring view of its slot (BF16, in-place ring, one ring row per buffer row)."""
        if (
            self._ring_k is None
            or self._ring_v is None
            or self._ring_disabled
            or self._ring_k.shape[1] != self._static_batch_rows
        ):
            return False
        for slot, k_entry, v_entry in zip(slots, k_entries, v_entries, strict=True):
            if not isinstance(k_entry, torch.Tensor) or not isinstance(v_entry, torch.Tensor):
                return False
            if (
                k_entry.data_ptr() != self._ring_k[slot].data_ptr()
                or v_entry.data_ptr() != self._ring_v[slot].data_ptr()
            ):
                return False
        return True

    @staticmethod
    def _coalesce_slot_runs(slots: list[int]) -> list[tuple[int, int]]:
        """Group consecutive physical slots into ``[start, end)`` runs, preserving order."""
        runs: list[tuple[int, int]] = []
        for slot in slots:
            if runs and runs[-1][1] == slot:
                runs[-1] = (runs[-1][0], slot + 1)
            else:
                runs.append((slot, slot + 1))
        return runs

    def fetch_kv_static(
        self,
        frame_idx: int,
        max_tokens: int,
        tokens_per_frame: int,
        *,
        num_heads: int,
        head_dim: int,
        device: torch.device | None,
        dtype: torch.dtype | None,
        history_frame_indices: list[int] | None = None,
        prefix_tokens: int = 0,
        und_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        batch_rows: int = 1,
        und_lens: tuple[int, ...] | None = None,
        row_slot_lists: list[list[int]] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Fetch cached K/V history from a persistent fixed-size buffer.

        Unlike ``fetch_kv_padded``, this does not ``cat`` and ``pad`` on every
        read.  It rebuilds the persistent buffer only when the requested frame or
        the explicit ``history_frame_indices`` change, or the cache is invalidated
        by a store/reset.

        With ``prefix_tokens`` every row is ``[und | curr | hist | pad]`` with stride
        ``R = prefix_tokens + max_tokens`` and the buffer is ``[1, batch_rows * R, H, D]``; the
        history of row ``r`` lives at ``r*R + prefix_tokens``.  ``und_kv`` (``[B,U,H,D]`` each,
        ``U`` = padded und length) is copied once per distinct und tensor pair into the und
        region of each row, right-aligned so that it ends at ``r*R + U`` (``und_lens`` gives the
        real per-row lengths; default: every row full).  The returned ``real_len`` counts history
        tokens per row.
        """
        if self._ring_k is not None and not self._ring_disabled:
            # The ring carries the cached shape/device/dtype without walking the history.
            num_heads = self._ring_k.shape[3]
            head_dim = self._ring_k.shape[4]
            device = self._ring_k.device
            dtype = self._ring_k.dtype
        else:
            first_k = self._first_cached_history_k(frame_idx)  # [B,S_frame,H,D] or None
            if first_k is not None:
                # Prefer the cached tensor metadata over caller-provided defaults.
                # This preserves CP/head-sharded cache shapes and also picks up the
                # actual device/dtype after the first frame is cached.
                num_heads = first_k.shape[2]
                head_dim = first_k.shape[3]
                device = first_k.device
                dtype = first_k.dtype

        self._ensure_static_history_buffer_allocated(
            max_tokens,
            num_heads=num_heads,
            head_dim=head_dim,
            device=device,
            dtype=dtype,
            prefix_tokens=prefix_tokens,
            batch_rows=batch_rows,
        )
        if und_kv is not None:
            k_und, v_und = und_kv
            und_capacity = int(k_und.shape[1])
            lens = tuple(und_lens) if und_lens is not None else (und_capacity,) * batch_rows
            und_key = (k_und.data_ptr(), v_und.data_ptr(), und_capacity, lens)
            if self._static_und_key != und_key:
                assert self._static_k_buf is not None and self._static_v_buf is not None
                if self._static_k_buf.is_cuda and torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("static AR und-region priming requested inside a CUDA graph capture")
                if und_capacity > prefix_tokens:
                    raise AssertionError(
                        f"und length {und_capacity} exceeds the static prefix of {prefix_tokens} tokens"
                    )
                if k_und.shape[0] != batch_rows or len(lens) != batch_rows:
                    raise AssertionError(
                        f"und K/V carry {k_und.shape[0]} rows / {len(lens)} lengths for {batch_rows} rows"
                    )
                row_stride = prefix_tokens + max_tokens
                for row, real_len in enumerate(lens):
                    if real_len < 0 or real_len > und_capacity:
                        raise AssertionError(f"und length {real_len} outside [0, {und_capacity}] for row {row}")
                    end = row * row_stride + und_capacity
                    # Right-aligned: every row's current frame starts at the same offset ``U``.
                    self._static_k_buf[0, end - real_len : end].copy_(k_und[row, :real_len])  # [u_r,H,D]
                    self._static_v_buf[0, end - real_len : end].copy_(v_und[row, :real_len])  # [u_r,H,D]
                self._static_und_key = und_key
        if row_slot_lists is not None:
            requested_indices: tuple | None = tuple(tuple(slots) for slots in row_slot_lists)
        else:
            requested_indices = None if history_frame_indices is None else tuple(history_frame_indices)
        needs_rebuild = (
            self._static_valid_frame_idx != int(frame_idx)
            or self._static_tokens_per_frame != tokens_per_frame
            or self._static_history_indices != requested_indices
        )
        if needs_rebuild:
            self._rebuild_static_history(
                frame_idx,
                max_tokens,
                tokens_per_frame,
                history_frame_indices=history_frame_indices,
                row_slot_lists=row_slot_lists,
            )

        assert self._static_k_buf is not None
        assert self._static_v_buf is not None
        return self._static_k_buf, self._static_v_buf, self._static_real_len

    def fetch_kv_padded(
        self,
        frame_idx: int,
        max_tokens: int,
        *,
        num_heads: int = 0,
        head_dim: int = 0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Fetch cached K/V history, zero-padded to a fixed size.

        Always returns tensors of shape [1, max_tokens, H, D], regardless
        of how many segments are actually cached.  When the cache is empty
        (frame_idx <= 0), returns all-zero tensors.  This guarantees a
        constant tensor shape across segments, preventing torch.compile
        recompilation.

        Args:
            frame_idx: Current frame/segment index.
            max_tokens: Fixed output sequence length to pad to.
                Should be ``(cache_size - 1) * tokens_per_segment``.
            num_heads: Number of KV heads (used when cache is empty).
            head_dim: Per-head dimension (used when cache is empty).
            device: Device for zero tensors (used when cache is empty).
            dtype: Dtype for zero tensors (used when cache is empty).

        Returns:
            (k_padded, v_padded, real_len):
                k_padded: [1, max_tokens, H, D]
                v_padded: [1, max_tokens, H, D]
                real_len: number of non-padding tokens (0 when empty).
        """
        raw_k, raw_v = self.fetch_kv(frame_idx)
        if raw_k is None:
            return (
                torch.zeros(1, max_tokens, num_heads, head_dim, device=device, dtype=dtype),
                torch.zeros(1, max_tokens, num_heads, head_dim, device=device, dtype=dtype),
                0,
            )
        k_det, v_det = raw_k.detach(), raw_v.detach()
        real_len = k_det.shape[1]
        pad = max_tokens - real_len
        return (
            F.pad(k_det, (0, 0, 0, 0, 0, pad)),
            F.pad(v_det, (0, 0, 0, 0, 0, pad)),
            real_len,
        )


class DualKVCache:
    """Wrapper managing both und (fixed) and gen (rolling) caches.

    Used for optimized AR generation where:
    - Frame 0: Compute all tokens (und + gen), store both in respective caches
    - Frame 1+: Only compute new gen tokens, retrieve cached und K/V

    Attributes:
        und_cache: Fixed cache for und (text) tokens.
        gen_cache: Rolling cache for gen (vision+action) tokens.

    Example:
        >>> dual_cache = DualKVCache(gen_cache_size=16)
        >>> # Frame 0: Store und K/V [B,S_und,H,D] and gen K/V [B,S_gen,H,D]
        >>> dual_cache.und_cache.store(k_und, v_und)
        >>> dual_cache.gen_cache.store_kv(k_gen, v_gen, frame_idx=0)
        >>>
        >>> # Frame 1+: Retrieve cached und [B,S_und,H,D], fetch gen history
        >>> k_und, v_und = dual_cache.und_cache.get()
        >>> k_gen_hist, v_gen_hist = dual_cache.gen_cache.fetch_kv(frame_idx=1)  # [B,S_hist,H,D] or None
        >>> # Process new gen tokens [B,S_gen_new,H,D] and store
        >>> dual_cache.gen_cache.store_kv(k_gen_new, v_gen_new, frame_idx=1)
    """

    def __init__(
        self,
        gen_cache_size: int | None = None,
        kv_cache_dtype: str | None = None,
        kv_cache_kernel_impl: str = "triton",
        attention_sink_size: int = 0,
        *,
        buffer_pool: KVBufferPool | None = None,
        pool_slot: int = 0,
        preallocate_ring: bool = False,
    ) -> None:
        """Initialize dual cache.

        Args:
            gen_cache_size: Maximum number of gen frames to cache (None = very large).
            kv_cache_dtype: Storage format for the gen cache. None selects BF16
                (default); "fp8" selects tensor-scale e4m3 FP8. The und cache
                is always BF16.
            kv_cache_kernel_impl: FP8 gen-cache batch decode kernel backend.
                "triton" is the default fused decode path; "torch" uses the
                reference path. FP8 encode always uses the torch path.
            attention_sink_size: Number of initial gen frames to pin in cache.
            buffer_pool / pool_slot / preallocate_ring: See ``GenKVCache``.
        """
        self.und_cache = UndKVCache()
        if kv_cache_dtype is None:
            backend: KVStorageBackend = BF16StorageBackend()
        elif kv_cache_dtype == "fp8":
            backend = FP8StorageBackend(kv_cache_dtype="fp8", kernel_impl=kv_cache_kernel_impl)
        else:
            raise ValueError(f"kv_cache_dtype must be None or 'fp8'; got {kv_cache_dtype!r}")
        self.gen_cache = GenKVCache(
            cache_size=gen_cache_size,
            backend=backend,
            attention_sink_size=attention_sink_size,
            buffer_pool=buffer_pool,
            pool_slot=pool_slot,
            preallocate_ring=preallocate_ring,
        )

    def reset(self) -> None:
        """Reset both caches."""
        self.und_cache.reset()
        self.gen_cache.reset()


@dataclass
class KVTrainMemoryValue(MemoryValue):
    """Read-only tensor container for the KV-cache training path, passed into compiled regions.

    Originally introduced for KV-cache training (segment-based loop with a
    rolling gen cache).  Also reused by compile-safe AR inference under
    ``torch.compile`` (with ``gen_cache_size == num_frames``); the inference
    path imports this same type because the underlying buffer mechanics
    (rolling gen cache, padded und cache, varlen offsets) match.  A future MR
    will properly decouple the training and inference KV-cache abstractions.

    Carries the cached K/V tensors, boolean flags, and varlen offsets that
    ``three_way_attention_with_kv_cache`` needs.  All fields are tensors (or
    tensor-derived constants) whose types and shapes must be stable across
    steps to avoid ``torch.compile`` recompilation.

    Produced by ``KVCacheTrainMemoryState.read_for_layer()`` outside the
    compile boundary and consumed inside it.

    Attributes:
        vision_token_shapes: Per-sample ``(T, H_p, W_p)`` shapes used by
            ``multi_dimensional_attention`` for the temporal-causal gen SA
            reshape.
        num_action_tokens_per_supertoken: Number of action tokens prefixing
            each latent frame; also used for the gen SA reshape.

        has_new_caption: Scalar bool tensor.  ``True`` when the current
            segment carries a new text/und caption.
        has_caption: Scalar bool tensor.  ``True`` when there is any real
            text caption available — either a new caption in the current
            segment (``has_new_caption=True``) or a previously cached
            caption (``cached_und_len > 0``).  Masks the video-to-text
            CA LSE to ``-inf`` when no caption exists anywhere, so the
            merge gives that component zero weight.  Mirrors
            ``has_cached_gen`` on the cached-video side.
        has_cached_gen: Scalar bool tensor.  ``True`` when the gen cache is
            non-empty (``segment_idx > 0``).  Masks the gen-CA LSE
            to ``-inf`` when the cache is logically empty.
        und_kv_offsets: ``int32`` tensor ``[2]`` — cumulative seqlen for
            und KV in varlen attention.  Equals ``[0, real_und_len]``.
        clamp_empty_varlen_kv: When True, ``und_kv_offsets`` and
            ``gen_ca_cached_kv_offsets`` are clamped to length ``>= 1``.
            Currently needed for fp32, because the FA  kernel returns NaN for zero-length varlen); unnecessary and slower in bf16.
        gen_q_offsets: ``int32`` tensor ``[2]`` — cumulative seqlen for
            gen Q in varlen attention (e.g. ``[0, num_full]``).
        gen_ca_cached_kv_offsets: ``int32`` tensor ``[2]`` — cumulative
            seqlen for the cached gen K/V in the gen cross-attention to
            the KV-cache.  Equals ``[0, real_gen_cache_len]`` where
            ``real_gen_cache_len`` is the number of non-padding tokens in
            ``cached_gen_k/v``.  Always a tensor (never ``None``) so Dynamo
            compiles a single graph.

        cached_und_k: ``[1, padded_causal_len, H_kv, D]`` — padded cached
            und K.  Uses the actual padded ``causal_seq`` tensor size (not
            ``max_causal_len``, which is the real caption length).
            ``get_causal_seq()`` inside the compiled graph returns the
            padded tensor, so cached und KV must match that padded size
            for ``torch.where`` to broadcast.  The sequence dimension is
            marked static via ``torch._dynamo.mark_static`` so Dynamo
            specializes on the concrete size rather than assigning a
            symbolic variable.
        cached_und_v: ``[1, padded_causal_len, H_kv, D]`` — padded cached
            und V.  Same padding and static-marking as ``cached_und_k``.
        cached_gen_k: ``[1, max_gen_cache_tokens, H_kv, D]`` — padded
            cached gen K.  Always a tensor (never ``None``); zero-padded
            when the cache is empty.  Shape stays constant across
            segments (see ``max_gen_cache_tokens``) to avoid
            ``torch.compile`` recompilation.  The sequence dimension is
            marked static via ``torch._dynamo.mark_static``.
        cached_gen_v: ``[1, max_gen_cache_tokens, H_kv, D]`` — padded
            cached gen V.  Same padding and static-marking as
            ``cached_gen_k``.
        max_gen_cache_tokens: Constant padded sequence length of the gen
            history cache, equal to ``cache_size - 1`` times the token count
            of one segment. A transfer segment includes both aligned control
            and target layouts. Used as the static seq dim for ``cached_gen_k``
            / ``cached_gen_v`` and as the upper bound for
            ``gen_ca_cached_kv_offsets`` once the cache saturates.  Held
            as a Python ``int`` (not a tensor) because Dynamo specializes
            on it as a compile-time constant; it never changes after
            ``KVCacheTrainMemoryState`` initialization.
        uses_rolling_gen_cache: Python bool indicating whether attention
            should execute the cached-video component. Replay teacher forcing
            disables this explicitly because it starts without generated history
            and uses Pass-1 clean K/V directly.
    """

    vision_token_shapes: list[tuple[int, int, int]]
    num_action_tokens_per_supertoken: int

    has_new_caption: torch.Tensor
    has_caption: torch.Tensor
    has_cached_gen: torch.Tensor
    und_kv_offsets: torch.Tensor
    gen_q_offsets: torch.Tensor
    gen_ca_cached_kv_offsets: torch.Tensor

    cached_und_k: torch.Tensor
    cached_und_v: torch.Tensor
    cached_gen_k: torch.Tensor
    cached_gen_v: torch.Tensor
    max_gen_cache_tokens: int
    clamp_empty_varlen_kv: bool
    uses_rolling_gen_cache: bool = field(default=True, kw_only=True)

    @property
    def supports_context_parallel_attention(self) -> bool:
        return False


@dataclass
class TFReplayCleanMemoryValue(KVTrainMemoryValue):
    """Pass-1 replay teacher-forcing memory value.

    The tensor fields are identical to ``KVTrainMemoryValue``, but this
    container opts into CP dispatch.  Only replay teacher forcing produces this
    subclass; ordinary rolling KV-cache training keeps using
    ``KVTrainMemoryValue`` and remains CP-rejected.
    """

    # Backend-neutral visibility policy shared with the noisy replay pass.
    teacher_forcing_replay_policy: TeacherForcingReplayPolicyConfig = field(
        default_factory=TeacherForcingReplayPolicyConfig
    )
    frames_per_chunk: int = 1
    # Replay starts at segment zero; generic teacher forcing may carry history.
    uses_rolling_gen_cache: bool = field(default=False, kw_only=True)

    @property
    def supports_context_parallel_attention(self) -> bool:
        return True


@dataclass
class TFNoisyMemoryValue(KVTrainMemoryValue):
    """Read-only container for Pass 2 of teacher forcing.

    Inherits the rolling-cache and text-cache fields from ``KVTrainMemoryValue``
    and adds the current-segment clean gen K/V captured during Pass 1. Replay
    callers explicitly disable rolling history and use one-token placeholders
    for that unused cache; generic teacher forcing retains supplied history.
    """

    cached_clean_gen_k: torch.Tensor  # [1, S_clean, H_kv, D]
    cached_clean_gen_v: torch.Tensor  # [1, S_clean, H_kv, D]
    cached_clean_und_k: torch.Tensor | None = None  # [1,S_text,H_kv,D]
    cached_clean_und_v: torch.Tensor | None = None  # [1,S_text,H_kv,D]
    # Latent frames per causal chunk (chunk partition is [1, C, C, ...]; the
    # first chunk is always a single frame).  1 == framewise teacher forcing.
    frames_per_chunk: int = 1
    # Backend-neutral visibility policy shared with the clean replay pass.
    teacher_forcing_replay_policy: TeacherForcingReplayPolicyConfig = field(
        default_factory=TeacherForcingReplayPolicyConfig
    )
    # The optimized noisy pass projects only the target item's GEN rows. The
    # original full GEN layout is restored after each decoder layer.
    target_only_no_text: bool = False
    target_gen_start: int = 0
    target_gen_length: int = 0

    @property
    def supports_context_parallel_attention(self) -> bool:
        return True


class KVCacheTrainMemoryState(MemoryState):
    """Mutable memory state for the KV-cache training path (TransformerXL-style).

    Originally introduced for KV-cache training (segment-based loop with a
    rolling gen cache).  Also reused by compile-safe AR inference under
    ``torch.compile`` (with ``gen_cache_size == num_frames``); the inference
    path constructs this same class because the buffer mechanics match.
    A future MR will properly decouple the training and inference KV-cache
    abstractions (training is sized in segments; inference in frames).

    Wraps a ``list[DualKVCache]`` (one per transformer layer) together with
    the model-level constants needed to fetch and store cached K/V tensors.
    Constructed once per forward call in ``cosmos3_vfm_network.py``.

    Lifecycle within a single forward pass:

    1. ``init(hidden_states, device)`` — called once before any layer.
       Populates per-step tensor flags and offsets.
    2. ``read_for_layer(i)`` — called before each decoder layer.
       Returns a ``KVTrainMemoryValue`` snapshot of the cached K/V.
    3. ``write_for_layer(i, kv_to_store)`` — called after each decoder layer.
       Stores the newly-computed K/V back into the dual cache.

    Constructor attributes (fixed for the lifetime of training):
        vision_token_shapes: Per-sample ``(T, H_p, W_p)`` shapes from the
            packed sequence; needed for gen SA reshape.
        num_action_tokens_per_supertoken: Number of action tokens prefixing
            each latent frame.
        segment_idx: Index of the current segment within the video.  Used
            to query the gen cache circular buffer and to set
            ``has_cached_gen``.
        dual_kv_cache: Per-layer dual caches (``list[DualKVCache]``).
        num_kv_heads: Number of key/value attention heads (constant across
            layers).
        head_dim: Per-head dimension (constant across layers).
        context_parallel_size: Context-parallel degree for this memory state.
            CP-sharded packs carry local padded text tensors, while the
            attention kernel sees full sequence length after all-to-all, so
            empty cached-text buffers are sized as
            ``local_padded_causal_len * context_parallel_size``.

    Per-step attributes (set by ``init()``, read outside compile only):
        has_new_caption: Scalar bool tensor — ``True`` when the current
            segment has a new text/und caption.
        has_cached_gen: Scalar bool tensor — ``True`` when
            ``segment_idx > 0``.
        und_kv_offsets: ``int32[2]`` — cumulative seqlen for und KV.
        gen_q_offsets: ``int32[2]`` — cumulative seqlen for gen Q.
        has_new_caption_py: Python bool mirror of ``has_new_caption``.
        new_und_len: Number of real (unpadded) und tokens in the current
            segment (``0`` when no new caption).
        max_gen_cache_tokens: Constant padded size for gen cache tensors,
            computed once per step to avoid ``torch.compile`` recompilation.
    """

    def requires_natten_metadata(self) -> bool:
        return False

    def __init__(
        self,
        vision_token_shapes: list[tuple[int, int, int]],
        num_action_tokens_per_supertoken: int,
        null_action_supertokens: bool,
        segment_idx: int,
        dual_kv_cache: list[DualKVCache],
        num_kv_heads: int,
        head_dim: int,
        clamp_empty_varlen_kv: bool = True,
        context_parallel_size: int = 1,
    ) -> None:
        self.vision_token_shapes = vision_token_shapes
        self.num_action_tokens_per_supertoken = num_action_tokens_per_supertoken
        self.null_action_supertokens = null_action_supertokens
        self.segment_idx = segment_idx
        self.dual_kv_cache = dual_kv_cache
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.context_parallel_size = context_parallel_size

        # When True (the safe default), und_kv_offsets and
        # gen_ca_cached_kv_offsets are clamped to length >= 1 and the
        # corresponding LSE is masked to -inf in three_way_attention_with_kv_cache.
        self.clamp_empty_varlen_kv = clamp_empty_varlen_kv

        self.has_new_caption: torch.Tensor | None = None
        self.has_caption: torch.Tensor | None = None
        self.has_cached_gen: torch.Tensor | None = None
        self.und_kv_offsets: torch.Tensor | None = None
        self.gen_q_offsets: torch.Tensor | None = None
        self.gen_ca_cached_kv_offsets: torch.Tensor | None = None
        self.has_new_caption_py: bool = False
        self.new_und_len: int = 0
        self.max_gen_cache_tokens: int = 0
        self._padded_causal_len: int = 0
        self._device: torch.device = torch.device("cpu")
        self._dtype: torch.dtype = torch.float32

    def init(self, hidden_states: dict, device: torch.device) -> None:
        """Populate per-step tensor flags and offsets.

        Called once per forward training step, outside the ``torch.compile``
        boundary.

        Args:
            hidden_states: ``SequencePack`` for the current step.
            device: Device for newly-created tensors.
        """
        self._device = device
        self._dtype = hidden_states.get("causal_seq", torch.empty(0)).dtype or torch.float32

        # Use the actual padded causal_seq tensor size, not max_causal_len
        # (which, confusingly, is the real caption length, not the padded
        # length).  get_causal_seq() inside the compiled graph returns the
        # padded tensor, so the cached und KV must match that padded size
        # for torch.where to broadcast.
        self._padded_causal_len = hidden_states["causal_seq"].shape[0] * self.context_parallel_size

        has_new_caption_py = hidden_states["_num_causal_tokens"] > 0

        # Calculate the actual number of text/und tokens in the caption.
        # The caption can either be cached, or supplied as a new caption.
        new_und_len = hidden_states["_num_causal_tokens"] if has_new_caption_py else 0
        cached_und_len = (
            self.dual_kv_cache[0].und_cache.cached_len if self.dual_kv_cache[0].und_cache.is_initialized else 0
        )
        und_real_len = new_und_len if has_new_caption_py else cached_und_len

        self.has_new_caption = torch.tensor(has_new_caption_py, device=device)
        # has_caption: True when *any* real text exists (new in the current
        # pack or cached from a prior segment).  Used to mask the
        # video-to-text CA LSE to -inf when there is no caption anywhere.
        self.has_caption = torch.tensor(und_real_len > 0, device=device)
        self.has_cached_gen = torch.tensor(self.segment_idx > 0, device=device)

        # Clamp to >=1 so the FA varlen kernel never sees a zero-length
        # range (which returns NaN for some varlen kernels).
        clamp_min = 1 if self.clamp_empty_varlen_kv else 0
        self.und_kv_offsets = torch.tensor(
            [0, max(und_real_len, clamp_min)],
            device=device,
            dtype=torch.int32,
        )
        self.gen_q_offsets = torch.tensor([0, hidden_states["_num_full_tokens"]], device=device, dtype=torch.int32)

        self.has_new_caption_py = has_new_caption_py
        self.new_und_len = new_und_len

        # Constant padded size for gen cache tensors so the shape never changes
        # between segments, avoiding torch.compile recompilation.
        if len(self.vision_token_shapes) == 2:
            segment_gen_tokens = sum(
                T * (self.num_action_tokens_per_supertoken + H_p * W_p) for T, H_p, W_p in self.vision_token_shapes
            )
        else:
            T, H_p, W_p = self.vision_token_shapes[0]
            segment_gen_tokens = T * (self.num_action_tokens_per_supertoken + H_p * W_p)
        cache_size = self.dual_kv_cache[0].gen_cache.cache_size
        self.max_gen_cache_tokens = (cache_size - 1) * segment_gen_tokens

        # Real (non-padding) length of the cached gen history. Each cached
        # segment contributes one vision layout normally, or both aligned
        # control and target layouts for transfer teacher forcing. The rolling
        # cache holds up to cache_size-1 past segments.
        #
        # Always build the varlen-offsets tensor and always route through the
        # varlen attention path, regardless of cache-fill level.  When the
        # cache is saturated (real == max), ``[0, max]`` attends to every
        # real token, matching dense attention up to FA reduction order.
        # Using a single path keeps the compiled graph count stable: Dynamo
        # would otherwise specialize on ``Optional[Tensor]`` (None vs Tensor)
        # and produce an extra recompile + CUDA-graph recapture at the frame
        # where the cache first saturates.

        real_gen_cache_len = min(self.segment_idx, cache_size - 1) * segment_gen_tokens
        self.gen_ca_cached_kv_offsets = torch.tensor(
            [0, max(real_gen_cache_len, clamp_min)],
            device=device,
            dtype=torch.int32,
        )

    def read_for_layer(self, layer_idx: int) -> KVTrainMemoryValue:
        """Fetch cached K/V for *layer_idx* and return as a ``KVTrainMemoryValue``.

        Called once per layer, outside the ``torch.compile`` boundary.
        Uses ``get_padded`` / ``fetch_kv_padded`` so shapes are constant
        across segments, and marks the sequence dimension static for Dynamo.
        """
        assert self.has_new_caption is not None
        assert self.has_cached_gen is not None

        device = self._device
        dtype = self._dtype
        padded_causal_len = self._padded_causal_len

        # Retrieve cached text KVs.
        cached_und_k, cached_und_v = self.dual_kv_cache[layer_idx].und_cache.get_padded(
            padded_causal_len,
            num_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            device=device,
            dtype=dtype,
        )

        # Mark the sequence dimension as static so Dynamo specializes on
        # the concrete size rather than assigning a symbolic variable.
        torch._dynamo.mark_static(cached_und_k, 1)
        torch._dynamo.mark_static(cached_und_v, 1)

        # Retrieve cached video KVs.
        cached_gen_k, cached_gen_v, _ = self.dual_kv_cache[layer_idx].gen_cache.fetch_kv_padded(
            self.segment_idx,
            self.max_gen_cache_tokens,
            num_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            device=device,
            dtype=dtype,
        )

        # Mark the sequence dimension as static so Dynamo specializes on
        # the concrete size rather than assigning a symbolic variable.
        torch._dynamo.mark_static(cached_gen_k, 1)
        torch._dynamo.mark_static(cached_gen_v, 1)

        return KVTrainMemoryValue(
            vision_token_shapes=self.vision_token_shapes,
            num_action_tokens_per_supertoken=self.num_action_tokens_per_supertoken,
            has_new_caption=self.has_new_caption,
            has_caption=self.has_caption,
            has_cached_gen=self.has_cached_gen,
            und_kv_offsets=self.und_kv_offsets,
            gen_q_offsets=self.gen_q_offsets,
            gen_ca_cached_kv_offsets=self.gen_ca_cached_kv_offsets,
            cached_und_k=cached_und_k,
            cached_und_v=cached_und_v,
            cached_gen_k=cached_gen_k,
            cached_gen_v=cached_gen_v,
            max_gen_cache_tokens=self.max_gen_cache_tokens,
            clamp_empty_varlen_kv=self.clamp_empty_varlen_kv,
        )

    def write_for_layer(self, layer_idx: int, kv_to_store: KVToStore) -> None:
        """Write newly-computed K/V back into the dual cache for *layer_idx*.

        Called once per layer, outside the ``torch.compile`` boundary.
        The compiled layer returns ``kv_to_store`` instead of mutating the
        cache directly because ``torch.compile(fullgraph=True)`` cannot
        handle mutations to objects that live outside the compiled scope.

        Detaches and clones K/V for truncated BPTT: gradients should not
        flow across segment boundaries in rolling KV-cache training.

        Args:
            layer_idx: Index of the current transformer layer.
            kv_to_store: ``(gen_k, gen_v, und_k, und_v)`` from the layer.
        """
        gen_k, gen_v, und_k, und_v = kv_to_store
        gen_k = gen_k.detach().clone()
        gen_v = gen_v.detach().clone()
        gen_v = zero_null_action_values(
            gen_v,
            self.vision_token_shapes,
            self.num_action_tokens_per_supertoken,
            self.null_action_supertokens,
        )  # [B,S,H,D]
        und_k = und_k.detach().clone()
        und_v = und_v.detach().clone()
        # Append generated video frames to a rolling KV-cache.
        self.dual_kv_cache[layer_idx].gen_cache.store_kv(gen_k, gen_v, frame_idx=self.segment_idx)
        if self.has_new_caption_py:
            # Overwrite existing caption with new caption.
            self.dual_kv_cache[layer_idx].und_cache.store(und_k[:, : self.new_und_len], und_v[:, : self.new_und_len])

    def is_gen_only(self) -> bool:
        return False


class TeacherForcingMemoryState(KVCacheTrainMemoryState):
    """Memory state for the two-pass teacher forcing training path.

    Pass 1 captures clean gen K/V per layer in ``_clean_gen_kv`` and caches
    text K/V for reuse. Replay teacher forcing always starts at segment zero,
    so it does not allocate or write the rolling generated-video cache.

    Pass 2: ``read_for_layer`` returns ``TFNoisyMemoryValue`` (with the clean
    gen K/V attached).  ``write_for_layer`` is a no-op (clean data already
    written in Pass 1).
    """

    def __init__(
        self,
        vision_token_shapes: list[tuple[int, int, int]],
        num_action_tokens_per_supertoken: int,
        null_action_supertokens: bool,
        segment_idx: int,
        dual_kv_cache: list[DualKVCache],
        num_kv_heads: int,
        head_dim: int,
        detach_clean_kv: bool = False,
        clamp_empty_varlen_kv: bool = True,
        frames_per_chunk: int = 1,
        teacher_forcing_replay_policy: TeacherForcingReplayPolicyConfig | None = None,
        context_parallel_size: int = 1,
        selected_clean_gen_token_indexes: torch.Tensor | None = None,
        selected_clean_gen_padded_capacity: int = 0,
        target_only_no_text_pass2: bool = False,
        allow_detached_target_only_clean_kv: bool = False,
    ) -> None:
        super().__init__(
            vision_token_shapes=vision_token_shapes,
            num_action_tokens_per_supertoken=num_action_tokens_per_supertoken,
            null_action_supertokens=null_action_supertokens,
            segment_idx=segment_idx,
            dual_kv_cache=dual_kv_cache,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            clamp_empty_varlen_kv=clamp_empty_varlen_kv,
            context_parallel_size=context_parallel_size,
        )
        self.pass_number = 1
        self.detach_clean_kv = detach_clean_kv
        self.frames_per_chunk = frames_per_chunk
        self.teacher_forcing_replay_policy = (
            TeacherForcingReplayPolicyConfig()
            if teacher_forcing_replay_policy is None
            else teacher_forcing_replay_policy
        )
        if selected_clean_gen_token_indexes is None and selected_clean_gen_padded_capacity:
            raise ValueError("selected_clean_gen_padded_capacity requires selected_clean_gen_token_indexes.")
        if (
            selected_clean_gen_token_indexes is not None
            and selected_clean_gen_token_indexes.numel() > selected_clean_gen_padded_capacity
        ):
            raise ValueError(
                f"selected_clean_gen_padded_capacity={selected_clean_gen_padded_capacity} is smaller than "
                f"{selected_clean_gen_token_indexes.numel()} selected clean target tokens."
            )
        self.selected_clean_gen_token_indexes = selected_clean_gen_token_indexes
        self.selected_clean_gen_padded_capacity = selected_clean_gen_padded_capacity
        if target_only_no_text_pass2 and detach_clean_kv and not allow_detached_target_only_clean_kv:
            raise ValueError(
                "teacher_forcing_target_only_no_text_pass2 requires detach_clean_kv=False to preserve "
                "gradients through clean GEN and text K/V, unless the caller explicitly allows a detached "
                "frozen-teacher cache."
            )
        if target_only_no_text_pass2 and context_parallel_size != 1:
            raise ValueError(
                "teacher_forcing_target_only_no_text_pass2 currently requires context_parallel_size=1; "
                f"got {context_parallel_size}."
            )
        if target_only_no_text_pass2 and selected_clean_gen_token_indexes is not None:
            raise ValueError("teacher_forcing_target_only_no_text_pass2 does not support multiview Flex K/V selection.")
        if target_only_no_text_pass2 and len(vision_token_shapes) not in (1, 2):
            raise ValueError(
                "teacher_forcing_target_only_no_text_pass2 requires one target item or aligned "
                f"[control, target] items; got {len(vision_token_shapes)} items."
            )
        if target_only_no_text_pass2 and num_action_tokens_per_supertoken != 0:
            raise ValueError("teacher_forcing_target_only_no_text_pass2 currently supports vision-only GEN rows.")
        self.target_only_no_text_pass2 = target_only_no_text_pass2
        self.target_gen_length = self._vision_item_num_tokens(vision_token_shapes[-1])
        self.target_gen_start = (
            self._vision_item_num_tokens(vision_token_shapes[0]) if len(vision_token_shapes) == 2 else 0
        )
        self._target_gen_q_offsets: torch.Tensor | None = None
        self._clean_gen_kv: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * len(dual_kv_cache)
        self._clean_und_kv: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * len(dual_kv_cache)

    def _vision_item_num_tokens(self, shape: tuple[int, int, int]) -> int:
        """Return one vision item's flattened GEN length."""
        num_frames, height, width = shape
        return num_frames * (self.num_action_tokens_per_supertoken + height * width)

    def init(self, hidden_states: dict, device: torch.device) -> None:
        """Initialize replay metadata and validate the target-only GEN layout."""
        super().init(hidden_states, device)
        self._target_gen_q_offsets = None
        if not self.target_only_no_text_pass2:
            return

        expected_gen_tokens = self.target_gen_start + self.target_gen_length
        actual_gen_tokens = int(hidden_states["_num_full_tokens"])
        if actual_gen_tokens != expected_gen_tokens:
            raise ValueError(
                "Target-only teacher forcing requires the real GEN stream to contain exactly control plus target "
                f"vision rows; expected {expected_gen_tokens}, got {actual_gen_tokens}."
            )
        if self.pass_number == 2:
            self._target_gen_q_offsets = torch.tensor(
                [0, self.target_gen_length],
                device=device,
                dtype=torch.int32,
            )  # [2]

    def _read_teacher_forcing_base_value(self, layer_idx: int) -> KVTrainMemoryValue:
        """Read cached text K/V and build a one-token placeholder for unused video history."""
        assert self.has_new_caption is not None
        assert self.has_caption is not None
        assert self.has_cached_gen is not None
        assert self.und_kv_offsets is not None
        assert self.gen_q_offsets is not None
        assert self.gen_ca_cached_kv_offsets is not None

        cached_und_k, cached_und_v = self.dual_kv_cache[layer_idx].und_cache.get_padded(
            self._padded_causal_len,
            num_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            device=self._device,
            dtype=self._dtype,
        )  # [1,S_text,H_kv,D] each
        torch._dynamo.mark_static(cached_und_k, 1)
        torch._dynamo.mark_static(cached_und_v, 1)

        dummy_gen_kv = torch.zeros(
            1,
            1,
            self.num_kv_heads,
            self.head_dim,
            device=self._device,
            dtype=self._dtype,
        )  # [1,1,H_kv,D]
        torch._dynamo.mark_static(dummy_gen_kv, 1)
        return KVTrainMemoryValue(
            vision_token_shapes=self.vision_token_shapes,
            num_action_tokens_per_supertoken=self.num_action_tokens_per_supertoken,
            has_new_caption=self.has_new_caption,
            has_caption=self.has_caption,
            has_cached_gen=self.has_cached_gen,
            und_kv_offsets=self.und_kv_offsets,
            gen_q_offsets=self.gen_q_offsets,
            gen_ca_cached_kv_offsets=self.gen_ca_cached_kv_offsets,
            cached_und_k=cached_und_k,
            cached_und_v=cached_und_v,
            cached_gen_k=dummy_gen_kv,
            cached_gen_v=dummy_gen_kv,
            max_gen_cache_tokens=1,
            clamp_empty_varlen_kv=self.clamp_empty_varlen_kv,
        )

    def read_for_layer(self, layer_idx: int) -> KVTrainMemoryValue | TFNoisyMemoryValue:
        if self.pass_number == 1:
            base_value = self._read_teacher_forcing_base_value(layer_idx)
            return TFReplayCleanMemoryValue(
                vision_token_shapes=base_value.vision_token_shapes,
                num_action_tokens_per_supertoken=base_value.num_action_tokens_per_supertoken,
                has_new_caption=base_value.has_new_caption,
                has_caption=base_value.has_caption,
                has_cached_gen=base_value.has_cached_gen,
                und_kv_offsets=base_value.und_kv_offsets,
                gen_q_offsets=base_value.gen_q_offsets,
                gen_ca_cached_kv_offsets=base_value.gen_ca_cached_kv_offsets,
                cached_und_k=base_value.cached_und_k,
                cached_und_v=base_value.cached_und_v,
                cached_gen_k=base_value.cached_gen_k,
                cached_gen_v=base_value.cached_gen_v,
                max_gen_cache_tokens=base_value.max_gen_cache_tokens,
                clamp_empty_varlen_kv=base_value.clamp_empty_varlen_kv,
                teacher_forcing_replay_policy=self.teacher_forcing_replay_policy,
                frames_per_chunk=self.frames_per_chunk,
            )

        # Pass 2: wrap the parent's KVTrainMemoryValue with clean gen K/V.
        clean_kv = self._clean_gen_kv[layer_idx]
        assert clean_kv is not None, f"Clean gen K/V not captured for layer {layer_idx}"
        clean_k, clean_v = clean_kv
        clean_und_kv = self._clean_und_kv[layer_idx]
        if self.target_only_no_text_pass2:
            assert clean_und_kv is not None, f"Clean und K/V not captured for layer {layer_idx}"
            clean_und_k, clean_und_v = clean_und_kv
            assert self.has_new_caption is not None
            assert self.has_caption is not None
            assert self.has_cached_gen is not None
            assert self.und_kv_offsets is not None
            assert self.gen_ca_cached_kv_offsets is not None
            assert self._target_gen_q_offsets is not None, "Target-only Pass 2 offsets were not initialized"
            return TFNoisyMemoryValue(
                vision_token_shapes=self.vision_token_shapes,
                num_action_tokens_per_supertoken=self.num_action_tokens_per_supertoken,
                has_new_caption=self.has_new_caption,
                has_caption=self.has_caption,
                has_cached_gen=self.has_cached_gen,
                und_kv_offsets=self.und_kv_offsets,
                gen_q_offsets=self._target_gen_q_offsets,
                gen_ca_cached_kv_offsets=self.gen_ca_cached_kv_offsets,
                cached_und_k=clean_und_k,
                cached_und_v=clean_und_v,
                cached_gen_k=clean_k[:, :1],
                cached_gen_v=clean_v[:, :1],
                max_gen_cache_tokens=1,
                clamp_empty_varlen_kv=self.clamp_empty_varlen_kv,
                cached_clean_gen_k=clean_k,
                cached_clean_gen_v=clean_v,
                cached_clean_und_k=clean_und_k,
                cached_clean_und_v=clean_und_v,
                frames_per_chunk=self.frames_per_chunk,
                teacher_forcing_replay_policy=self.teacher_forcing_replay_policy,
                uses_rolling_gen_cache=False,
                target_only_no_text=True,
                target_gen_start=self.target_gen_start,
                target_gen_length=self.target_gen_length,
            )

        base_value = self._read_teacher_forcing_base_value(layer_idx)
        return TFNoisyMemoryValue(
            vision_token_shapes=base_value.vision_token_shapes,
            num_action_tokens_per_supertoken=base_value.num_action_tokens_per_supertoken,
            has_new_caption=base_value.has_new_caption,
            has_caption=base_value.has_caption,
            has_cached_gen=base_value.has_cached_gen,
            und_kv_offsets=base_value.und_kv_offsets,
            gen_q_offsets=base_value.gen_q_offsets,
            gen_ca_cached_kv_offsets=base_value.gen_ca_cached_kv_offsets,
            cached_und_k=base_value.cached_und_k,
            cached_und_v=base_value.cached_und_v,
            cached_gen_k=base_value.cached_gen_k,
            cached_gen_v=base_value.cached_gen_v,
            max_gen_cache_tokens=base_value.max_gen_cache_tokens,
            clamp_empty_varlen_kv=base_value.clamp_empty_varlen_kv,
            cached_clean_gen_k=clean_k,
            cached_clean_gen_v=clean_v,
            cached_clean_und_k=None,
            cached_clean_und_v=None,
            frames_per_chunk=self.frames_per_chunk,
            teacher_forcing_replay_policy=self.teacher_forcing_replay_policy,
            uses_rolling_gen_cache=False,
            target_only_no_text=False,
            target_gen_start=self.target_gen_start,
            target_gen_length=self.target_gen_length,
        )

    def write_for_layer(self, layer_idx: int, kv_to_store: KVToStore) -> None:
        if self.pass_number == 1:
            gen_k, gen_v, _und_k, _und_v = kv_to_store
            if self.selected_clean_gen_token_indexes is not None:
                selected_clean_indexes = self.selected_clean_gen_token_indexes.to(device=gen_k.device)  # [S_clean_real]
                gen_k = torch.index_select(gen_k, 1, selected_clean_indexes)  # [B,S_clean_real,H,D]
                gen_v = torch.index_select(gen_v, 1, selected_clean_indexes)  # [B,S_clean_real,H,D]
                clean_pad = self.selected_clean_gen_padded_capacity - gen_k.shape[1]
                if clean_pad:
                    gen_k = F.pad(gen_k, (0, 0, 0, 0, 0, clean_pad))  # [B,S_clean,H,D]
                    gen_v = F.pad(gen_v, (0, 0, 0, 0, 0, clean_pad))  # [B,S_clean,H,D]
            if self.detach_clean_kv:
                clean_gen_k = gen_k.detach().clone()  # [B,S,H,D]
                clean_gen_v = gen_v.detach().clone()  # [B,S,H,D]
            else:
                clean_gen_k = gen_k.clone()  # [B,S,H,D]
                clean_gen_v = gen_v.clone()  # [B,S,H,D]
            clean_gen_v = zero_null_action_values(
                clean_gen_v,
                self.vision_token_shapes,
                self.num_action_tokens_per_supertoken,
                self.null_action_supertokens,
            )  # [B,S,H,D]
            self._clean_gen_kv[layer_idx] = (clean_gen_k, clean_gen_v)
            if self.target_only_no_text_pass2:
                if self.detach_clean_kv:
                    clean_und_k = _und_k.detach().clone()  # [B,S_text,H,D]
                    clean_und_v = _und_v.detach().clone()  # [B,S_text,H,D]
                else:
                    clean_und_k = _und_k.clone()  # [B,S_text,H,D]
                    clean_und_v = _und_v.clone()  # [B,S_text,H,D]
                clean_und_capacity = max(self._padded_causal_len, 1 if self.clamp_empty_varlen_kv else 0)
                clean_und_pad = clean_und_capacity - clean_und_k.shape[1]
                if clean_und_pad < 0:
                    raise ValueError(
                        f"Clean text K/V has {clean_und_k.shape[1]} rows, exceeding padded capacity "
                        f"{clean_und_capacity}."
                    )
                if clean_und_pad:
                    clean_und_k = F.pad(  # [B,S_text_padded,H,D]
                        clean_und_k, (0, 0, 0, 0, 0, clean_und_pad)
                    )
                    clean_und_v = F.pad(  # [B,S_text_padded,H,D]
                        clean_und_v, (0, 0, 0, 0, 0, clean_und_pad)
                    )
                self._clean_und_kv[layer_idx] = (clean_und_k, clean_und_v)
            if self.has_new_caption_py:
                real_und_k = _und_k[:, : self.new_und_len]  # [B,S_text,H,D]
                real_und_v = _und_v[:, : self.new_und_len]  # [B,S_text,H,D]
                self.dual_kv_cache[layer_idx].und_cache.store(real_und_k, real_und_v)
            return

        # Pass 2: no-op. Clean KV already written in Pass 1.

    def is_gen_only(self) -> bool:
        return self.pass_number == 2 and self.target_only_no_text_pass2


@dataclass
class FlexARMemoryValue(MemoryValue):
    """Read-only fixed-capacity K/V suffix for multiview transfer AR inference."""

    cached_gen_k: torch.Tensor | None  # [1,S_memory,H_kv,D] or None during prefill
    cached_gen_v: torch.Tensor | None  # [1,S_memory,H_kv,D] or None during prefill


class FlexARMemoryState(MemoryState):
    """Capture and serve the fixed-size FlexAttention memory suffix.

    The cache list is shared by the prefill, denoising, and clean-refresh
    states of one CFG branch. Denoising states omit ``write_indexes`` and are
    read-only; prefill/refresh states select clean GEN K/V and write them into
    either a contiguous suffix range or explicitly indexed cache slots.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        memory_seq_len: int,
        cache: list[tuple[torch.Tensor, torch.Tensor] | None] | None = None,
        write_indexes: torch.Tensor | None = None,
        write_offset: int = 0,
        cache_write_indexes: torch.Tensor | None = None,
    ) -> None:
        if memory_seq_len < 1:
            raise ValueError(f"memory_seq_len must be >= 1, got {memory_seq_len}.")
        if write_offset < 0:
            raise ValueError(f"write_offset must be >= 0, got {write_offset}.")
        self.memory_seq_len = memory_seq_len
        self.cache = [None] * num_layers if cache is None else cache
        if len(self.cache) != num_layers:
            raise ValueError(f"Expected {num_layers} cache entries, got {len(self.cache)}.")
        if write_indexes is not None and write_indexes.ndim != 1:
            raise ValueError(f"write_indexes must be one-dimensional, got shape {tuple(write_indexes.shape)}.")
        if cache_write_indexes is not None:
            if cache_write_indexes.ndim != 1:
                raise ValueError(
                    f"cache_write_indexes must be one-dimensional, got shape {tuple(cache_write_indexes.shape)}."
                )
            if cache_write_indexes.dtype not in (torch.int32, torch.int64):
                raise ValueError(f"cache_write_indexes must use an integer dtype, got {cache_write_indexes.dtype}.")
            if write_indexes is None:
                raise ValueError("cache_write_indexes requires write_indexes.")
            if cache_write_indexes.numel() != write_indexes.numel():
                raise ValueError(
                    "cache_write_indexes and write_indexes must have the same length; "
                    f"got {cache_write_indexes.numel()} and {write_indexes.numel()}."
                )
            if cache_write_indexes.numel() and (
                int(cache_write_indexes.min().item()) < 0 or int(cache_write_indexes.max().item()) >= memory_seq_len
            ):
                raise ValueError(
                    f"cache_write_indexes must be in [0, {memory_seq_len}); got {cache_write_indexes.tolist()}."
                )
        self.write_indexes = write_indexes
        self.write_offset = write_offset
        self.cache_write_indexes = cache_write_indexes

    def requires_natten_metadata(self) -> bool:
        return False

    def init(self, hidden_states: dict, device: torch.device) -> None:
        del hidden_states, device

    def read_for_layer(self, layer_idx: int) -> FlexARMemoryValue:
        cached_kv = self.cache[layer_idx]
        if cached_kv is None:
            return FlexARMemoryValue(cached_gen_k=None, cached_gen_v=None)
        cached_k, cached_v = cached_kv
        return FlexARMemoryValue(cached_gen_k=cached_k, cached_gen_v=cached_v)

    def write_for_layer(self, layer_idx: int, kv_to_store: KVToStore) -> None:
        if self.write_indexes is None:
            return
        gen_k, gen_v, _und_k, _und_v = kv_to_store
        write_indexes = self.write_indexes.to(device=gen_k.device)  # [S_write]
        selected_k = torch.index_select(gen_k, 1, write_indexes).detach()  # [1,S_write,H_kv,D]
        selected_v = torch.index_select(gen_v, 1, write_indexes).detach()  # [1,S_write,H_kv,D]
        write_end = self.write_offset + selected_k.shape[1]
        if self.cache_write_indexes is None and write_end > self.memory_seq_len:
            raise ValueError(
                f"Flex AR cache write [{self.write_offset}, {write_end}) exceeds capacity {self.memory_seq_len}."
            )
        cached_kv = self.cache[layer_idx]
        if cached_kv is None:
            cached_k = selected_k.new_zeros(
                (selected_k.shape[0], self.memory_seq_len, selected_k.shape[2], selected_k.shape[3])
            )  # [1,S_memory,H_kv,D]
            cached_v = selected_v.new_zeros(
                (selected_v.shape[0], self.memory_seq_len, selected_v.shape[2], selected_v.shape[3])
            )  # [1,S_memory,H_kv,D]
            self.cache[layer_idx] = (cached_k, cached_v)
        else:
            cached_k, cached_v = cached_kv
        if self.cache_write_indexes is None:
            cached_k[:, self.write_offset : write_end].copy_(selected_k)  # [1,S_write,H_kv,D]
            cached_v[:, self.write_offset : write_end].copy_(selected_v)  # [1,S_write,H_kv,D]
            return
        cache_write_indexes = self.cache_write_indexes.to(device=cached_k.device, dtype=torch.long)  # [S_write]
        cached_k.index_copy_(1, cache_write_indexes, selected_k)  # [1,S_memory,H_kv,D]
        cached_v.index_copy_(1, cache_write_indexes, selected_v)  # [1,S_memory,H_kv,D]

    def is_gen_only(self) -> bool:
        return False


@dataclass
class ARMemoryValue(MemoryValue):
    """Read-only tensor container for AR inference, passed into compiled regions.

    Two mutually-exclusive flavors share this container:

    1. **Dynamic-shape (eager / compile-no-CG / frame 0 for compile-CG)**: ``gen_k_hist`` /
       ``gen_v_hist`` are sized to the real cached history
       (``frame_idx * S_per_frame``) and grow each frame.  The static-shape
       fields below are ``None``.
    2. **Static-shape (compile + CUDA Graphs)**: ``gen_k_buf_full`` /
       ``gen_v_buf_full`` carry the full preallocated gen-cache buffer at
       a constant max size, and ``cu_seqlens_q_t`` / ``cu_seqlens_kv_t``
       are ``[2]`` int32 tensors whose *values* change per frame but
       whose shapes stay constant.  ``gen_k_hist`` / ``gen_v_hist`` are
       ``None``.  The attention dispatch restricts the kernel to the
       real prefix at runtime via ``attention()``'s
       ``cumulative_seqlen_KV`` kwarg, giving CUDA Graphs a single
       capture that replays for every frame.

    The static flavor hands the block one pooled buffer per K and V,
    ``kv_k_static`` / ``kv_v_static`` ``[1, S_und + gen_len + S_hist_max, H_kv, D]``,
    laid out ``[und | curr | hist | pad]``: the und K/V are copied in once per
    generation, the block writes the current frame in place at
    ``static_curr_offset`` (= ``S_und``) and the history is rebuilt outside the
    compiled region at ``static_hist_offset`` (= ``S_und + gen_len``).  This is
    the same token order the block used to materialise with ``cat`` every
    forward, so attention reads identical bytes without the 81 MB copy.  For a
    single sample ``und_k_cached`` / ``gen_k_buf_full`` stay populated as *views* of
    that buffer (the und prefix and the ``[hist | pad]`` window of ``max_gen_tokens``),
    so readers of the pre-composite layout see the same bytes; a hand-built value with
    ``kv_k_static=None`` and those two set still takes the legacy ``cat`` path in
    ``attention_AR_gen_only``.  With ``batch_size > 1``
    the buffer holds one such row per sample at stride ``static_row_stride`` (``[1, B*R, H, D]``),
    the und of row ``r`` right-aligned to end at ``r*R + static_curr_offset``, and
    ``cu_seqlens_q_t`` / ``cu_seqlens_kv_t`` are ``[2B]``: entry ``2r`` is row ``r`` and entry
    ``2r+1`` a zero-query dummy sequence that owns the gap up to the next row (the varlen
    kernel wants adjacent sequences and skips zero-query entries).  ``B=1`` collapses to the
    ``[2]`` tensors above.

    All tensor sequence dimensions are fixed per-step.  For Context
    Parallelism the head dimension is ``H/cp`` (head-sharded); otherwise ``H``.

    Attributes:
        und_k_cached: ``[1, S_und, H_kv, D]`` — cached und K from frame 0.
            ``None`` at frame 0 before und cache is populated.
        und_v_cached: ``[1, S_und, H_kv, D]`` — cached und V from frame 0.
            ``None`` at frame 0 before und cache is populated.
        gen_k_hist: ``[1, S_hist, H_kv, D]`` or ``None`` — dynamic-shape gen
            K history from frames ``0..frame_idx-1``.  ``None`` in the
            static-shape flavor.
        gen_v_hist: ``[1, S_hist, H_kv, D]`` or ``None`` — dynamic-shape gen
            V history.  ``None`` in the static-shape flavor.
        frame_idx: Current frame index (0-based).
        gen_len: Number of real (unpadded) gen tokens in the current frame.
        gen_k_buf_full: ``[1, max_gen_tokens, H_kv, D]`` or ``None`` —
            static-shape gen K buffer (real prefix + zero-padded tail).
        gen_v_buf_full: ``[1, max_gen_tokens, H_kv, D]`` or ``None`` —
            static-shape gen V buffer.
        real_gen_cache_len_t: ``int32`` tensor of shape ``[1]`` or
            ``None`` — number of real (non-padding) tokens at the head of
            ``gen_k_buf_full`` / ``gen_v_buf_full``.  Tensor (not Python
            int) so its value can change across CUDA-graph replays without
            triggering recompilation.
        real_und_cache_len_t: ``int32`` tensor of shape ``[1]`` or
            ``None`` — number of real understanding tokens at the head of
            ``und_k_cached`` / ``und_v_cached`` when those tensors are padded
            to a static max length for post-saturation static compile.
        cu_seqlens_q_t: ``int32`` tensor of shape ``[2]`` or ``None`` —
            cumulative Q seqlens ``[0, gen_len]`` for varlen attention.
            Constant across frames (gen_len is fixed by resolution).
        cu_seqlens_kv_t: ``int32`` tensor of shape ``[2]`` or ``None`` —
            cumulative KV seqlens ``[0, S_und + gen_len + real_gen_cache_len]``
            for varlen attention.  Built outside the compiled region so
            Dynamo doesn't specialize on the captured Python ints; only
            the second element's *value* changes per frame, shape stays
            ``[2]``.
        max_seqlen_KV: Static maximum total KV length used by the varlen
            attention kernel in the static-shape branch (= S_und + gen_len
            + max_gen_cache_tokens).  Constant across frames.
        for_cuda_graphs: Static-shape flavor selector.  ``True`` ⇒
            static-shape buffers + varlen offsets are populated and the
            ``attention_AR_gen_only`` static-shape branch must be taken;
            ``False`` ⇒ dynamic-shape (eager / compile-no-CG / frame 0) flavor.
        post_saturation_static_compile: Marks dynamic-shape AR calls whose
            visible rolling KV history has reached a fixed window length, so
            the decoder layer may route to a separate static torch.compile path.
    """

    und_k_cached: torch.Tensor | None
    und_v_cached: torch.Tensor | None
    gen_k_hist: torch.Tensor | None
    gen_v_hist: torch.Tensor | None
    frame_idx: int
    gen_len: int
    batch_size: int = 1
    gen_lens: tuple[int, ...] = ()
    und_lens: tuple[int, ...] = ()
    gen_k_buf_full: torch.Tensor | None = None
    gen_v_buf_full: torch.Tensor | None = None
    kv_k_static: torch.Tensor | None = None
    kv_v_static: torch.Tensor | None = None
    static_curr_offset: int = 0
    static_hist_offset: int = 0
    static_row_stride: int = 0
    real_gen_cache_len_t: torch.Tensor | None = None
    real_und_cache_len_t: torch.Tensor | None = None
    cu_seqlens_q_t: torch.Tensor | None = None
    cu_seqlens_kv_t: torch.Tensor | None = None
    max_seqlen_KV: int = 0
    for_cuda_graphs: bool = False
    post_saturation_static_compile: bool = False


class ARMemoryState(MemoryState):
    """Mutable memory state for autoregressive inference.

    Wraps a ``list[DualKVCache]`` (one per transformer layer) and a
    ``frame_idx``.  Constructed once per AR generation step.

    Lifecycle within a single forward pass:

    1. ``init(hidden_states, device)`` — captures ``gen_len`` from the pack.
    2. ``read_for_layer(i)`` — fetches cached und K/V and gen history,
       returns an ``ARMemoryValue``.
    3. ``write_for_layer(i, kv_to_store)`` — stores the current frame's
       K/V into the dual cache.  At frame 0, also stores und K/V.

    Two flavors selected via ``for_cuda_graphs``:

    - **Dynamic-shape** (default): ``read_for_layer`` returns gen history
      via ``fetch_kv`` (variable seq length).  Used by eager and
      compile-no-CG paths where shape variance is fine.
    - **Static-shape** (``for_cuda_graphs=True``): ``read_for_layer``
      returns the full preallocated gen buffer + a real-length scalar
      tensor instead of a sized history slice, so every captured tensor
      has a constant shape across frames.  Required for a single CUDA
      Graph capture that replays across the AR loop.  Must be combined
      with the static-shape branch in ``attention_AR_gen_only``.

    Constructor attributes:
        dual_kv_cache: Per-layer dual caches.
        frame_idx: Current AR frame index (0-based).
        vision_token_shapes: Per-sample ``(T, H_p, W_p)``; used by the
            ``write_for_layer`` null-action zero-out and required when
            ``for_cuda_graphs=True`` to size the gen-buffer.
        num_action_tokens_per_supertoken: Action tokens per temporal
            super-token.  Used by ``write_for_layer`` for null-action
            zero-out and (when ``for_cuda_graphs=True``) for the
            static-shape gen-buffer math (``S_super = num_action_tokens
            + H_p * W_p``).
        null_action_supertokens: When ``True``, zero out V for null
            action slots in ``write_for_layer``.
        for_cuda_graphs: Enable the static-shape ``read_for_layer`` path.
        num_kv_heads: KV head count; required when ``for_cuda_graphs=True``
            for empty-cache zero buffers at frame 0 (unused in the
            standard flow since frame 0 takes the dynamic-shape
            ``ARMemoryState`` branch — ``for_cuda_graphs=False`` —
            even when ``torch.compile`` + CG is enabled).
        head_dim: Per-head dim; required when ``for_cuda_graphs=True``
            for the same reason as ``num_kv_heads``.
        write_gen_cache: Whether ``write_for_layer`` should append the
            current gen K/V to the rolling cache.  Sampler velocity forwards
            are read-only because their current-frame K/V corresponds to a
            noisy denoising step, not the finalized frame.  Prefill/seed
            forwards keep the default ``True`` and update the cache once.
        kv_head_shard_rank: Rank index for local KV-head cache storage.
        kv_head_shard_size: Number of KV-head cache shards.  ``1`` means the
            cache stores full-head K/V.
        static_und_cache_max_len: Fixed packed und-cache length used only by
            post-saturation static compile. ``read_for_layer`` pads cached
            text K/V to this length and carries the real text length in
            ``ARMemoryValue.real_und_cache_len_t``.
        coarse_cuda_graph: Keep generated-history tensors at fixed addresses
            so a full-model post-saturation CUDA Graph can replay across frames.
        stage_gen_cache_writes: Retain refresh K/V outputs at captured addresses
            for an explicit cache commit after graph replay.
        transfer_history_sink_tokens: Number of leading cached visual tokens to
            preserve before applying Transfer history limiting. This is the
            tokenized form of complete logical Transfer sink frames.
        transfer_history_max_tokens: Optional Transfer history limit. With a
            positive ``transfer_history_sink_tokens`` this limits only the
            recent suffix after the pinned prefix. With no sink tokens it keeps
            the legacy behavior of limiting the complete history to a suffix.
            Supported on the dynamic-shape and the CUDA-graph static-shape
            flavors (the static flavor selects whole cache entries via
            ``GenKVCache.transfer_history_frame_indices``); rejected with
            post-saturation static compile.
    """

    def requires_natten_metadata(self) -> bool:
        return False

    def __init__(
        self,
        dual_kv_cache: list[DualKVCache],
        frame_idx: int,
        vision_token_shapes: list[tuple[int, int, int]] | None = None,
        num_action_tokens_per_supertoken: int = 0,
        null_action_supertokens: bool = False,
        *,
        for_cuda_graphs: bool = False,
        num_kv_heads: int | None = None,
        head_dim: int | None = None,
        write_gen_cache: bool = True,
        kv_head_shard_rank: int = 0,
        kv_head_shard_size: int = 1,
        post_saturation_static_compile: bool = False,
        static_und_cache_max_len: int | None = None,
        coarse_cuda_graph: bool = False,
        stage_gen_cache_writes: bool = False,
        transfer_history_sink_tokens: int = 0,
        transfer_history_max_tokens: int | None = None,
        batched: bool = False,
    ) -> None:
        self.dual_kv_cache = dual_kv_cache
        self.frame_idx = frame_idx
        self.vision_token_shapes = vision_token_shapes or []
        self.num_action_tokens_per_supertoken = num_action_tokens_per_supertoken
        self.null_action_supertokens = null_action_supertokens
        self.write_gen_cache = write_gen_cache
        self.kv_head_shard_rank = kv_head_shard_rank
        self.kv_head_shard_size = kv_head_shard_size
        self._gen_len: int = 0
        self.for_cuda_graphs = for_cuda_graphs
        self.post_saturation_static_compile = post_saturation_static_compile
        self.static_und_cache_max_len = static_und_cache_max_len
        self.coarse_cuda_graph: bool = coarse_cuda_graph
        self.stage_gen_cache_writes: bool = stage_gen_cache_writes
        self.transfer_history_sink_tokens: int = transfer_history_sink_tokens
        self.transfer_history_max_tokens: int | None = transfer_history_max_tokens
        self.batched: bool = batched
        self._batch_size: int = 1
        self._gen_lens: tuple[int, ...] = ()
        self._current_und_lens: tuple[int, ...] = ()
        self._staged_gen_kv: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * len(dual_kv_cache)
        self._coarse_padded_und_kv: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * len(dual_kv_cache)
        if for_cuda_graphs:
            assert vision_token_shapes is not None, "for_cuda_graphs=True requires vision_token_shapes"
            assert num_kv_heads is not None, "for_cuda_graphs=True requires num_kv_heads"
            assert head_dim is not None, "for_cuda_graphs=True requires head_dim"
        if post_saturation_static_compile:
            assert static_und_cache_max_len is not None, (
                "post_saturation_static_compile=True requires static_und_cache_max_len"
            )
        if coarse_cuda_graph:
            assert post_saturation_static_compile or for_cuda_graphs, (
                "coarse_cuda_graph=True requires post-saturation static compile or the static-shape "
                "(for_cuda_graphs=True) flavor"
            )
        if transfer_history_sink_tokens < 0:
            raise ValueError(f"transfer_history_sink_tokens must be >= 0, got {transfer_history_sink_tokens}")
        if transfer_history_max_tokens is not None:
            if transfer_history_max_tokens < 0:
                raise ValueError(f"transfer_history_max_tokens must be >= 0, got {transfer_history_max_tokens}")
            if post_saturation_static_compile:
                raise ValueError(
                    "transfer history limiting supports dynamic-shape and CUDA-graph static-shape AR inference, "
                    "not post-saturation static compile"
                )
        if batched and post_saturation_static_compile:
            raise ValueError("Batched AR memory does not support post-saturation static compile")
        if kv_head_shard_size > 1:
            assert not for_cuda_graphs, "local KV-head cache storage does not support CUDA graph static-cache mode"
            assert num_kv_heads is not None, "local KV-head cache storage requires num_kv_heads"
            assert num_kv_heads % kv_head_shard_size == 0, (
                f"kv_head_shard_size({kv_head_shard_size}) must divide num_kv_heads({num_kv_heads})"
            )
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim
        # Populated in init() when for_cuda_graphs.
        self._max_gen_cache_tokens: int = 0
        self._tokens_per_frame: int = 0
        self._real_gen_cache_len_t: torch.Tensor | None = None
        self._cu_seqlens_q_t: torch.Tensor | None = None
        self._cu_seqlens_kv_t: torch.Tensor | None = None
        self._real_und_cache_len_t: torch.Tensor | None = None
        self._max_seqlen_KV: int = 0
        # Explicit logical history entries for the static Transfer path (None = ring-native history).
        self._static_history_indices: tuple[int, ...] | None = None
        # ``[und | curr]`` tokens ahead of the history region in the composite static buffer.
        self._static_prefix_tokens: int = 0
        # Per-row physical slot plans (batched Transfer / per-row restarts); None = shared plan.
        self._static_row_slots: list[list[int]] | None = None
        self._device: torch.device = torch.device("cpu")
        self._dtype: torch.dtype = torch.float32

    def init(self, hidden_states: dict, device: torch.device) -> None:
        if self.batched:
            # ``get_num_real_samples`` excludes the trailing CUDA-graph pad segment.  AR packs give
            # every sample one generation split of the same size, so the gen lengths follow from the
            # total without touching device memory.  Prompt lengths are read from the split offsets
            # only while the pack carries text (frame 0, the dynamic path); gen-only frames see no
            # sync, which keeps init() legal inside a capture.
            num_samples = int(get_num_real_samples(hidden_states))
            self._batch_size = num_samples
            num_full = int(hidden_states["_num_full_tokens"])
            if num_samples <= 0 or num_full <= 0 or num_full % num_samples:
                raise ValueError(
                    f"Batched AR requires equal generation lengths: {num_full} tokens over {num_samples} rows"
                )
            self._gen_lens = (num_full // num_samples,) * num_samples
            num_causal = int(hidden_states["_num_causal_tokens"])
            if num_causal == 0:
                self._current_und_lens = (0,) * num_samples
            else:
                causal_offsets = hidden_states["_causal_seq_offsets"][: num_samples + 1]  # [B+1]
                full_offsets = hidden_states["_full_only_seq_offsets"][: num_samples + 1]  # [B+1]
                if causal_offsets.is_cuda and torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("batched AR prompt lengths cannot be read inside a CUDA graph capture")
                self._current_und_lens = tuple(int(x) for x in torch.diff(causal_offsets).tolist())
                if tuple(int(x) for x in torch.diff(full_offsets).tolist()) != self._gen_lens:
                    raise ValueError(
                        f"Batched AR requires equal generation lengths, got {torch.diff(full_offsets).tolist()}"
                    )
            if any(length <= 0 for length in self._gen_lens):
                raise ValueError(f"Every batched AR sample must contain generation tokens, got {self._gen_lens}")
            if len(set(self._gen_lens)) != 1:
                raise ValueError(f"Batched AR requires equal generation lengths, got {self._gen_lens}")
            self._gen_len = self._gen_lens[0]
        else:
            self._gen_len = hidden_states["_num_full_tokens"]
        if self.post_saturation_static_compile:
            assert self.static_und_cache_max_len is not None
            s_und = self.dual_kv_cache[0].und_cache.cached_len
            if s_und > self.static_und_cache_max_len:
                raise ValueError(
                    f"[AR inference] cached S_und={s_und} exceeds "
                    f"ar_static_und_cache_max_len={self.static_und_cache_max_len}"
                )
            if self._real_und_cache_len_t is None:
                self._real_und_cache_len_t = torch.tensor([s_und], device=device, dtype=torch.int32)  # [1]

            if self.coarse_cuda_graph and self._max_gen_cache_tokens == 0:
                gen_k_hist, _ = self.dual_kv_cache[0].gen_cache.fetch_kv(self.frame_idx)
                if gen_k_hist is None:
                    raise AssertionError("Coarse post-saturation CUDA Graph capture requires populated gen history")
                cache_size = self.dual_kv_cache[0].gen_cache.cache_size
                history_frames = cache_size - 1
                if gen_k_hist.shape[1] % history_frames != 0:
                    raise AssertionError(
                        f"Saturated gen history length {gen_k_hist.shape[1]} is not divisible by {history_frames} frames"
                    )
                self._max_gen_cache_tokens = gen_k_hist.shape[1]
                self._tokens_per_frame = gen_k_hist.shape[1] // history_frames
                und_k = self.dual_kv_cache[0].und_cache.k_und
                self._device = device
                self._dtype = und_k.dtype if und_k is not None else torch.float32

            if self.coarse_cuda_graph and self._cu_seqlens_q_t is None:
                real_total_kv_len = s_und + self._max_gen_cache_tokens + self._gen_len
                self._cu_seqlens_q_t = torch.tensor([0, self._gen_len], device=device, dtype=torch.int32)  # [2]
                self._cu_seqlens_kv_t = torch.tensor(
                    [0, real_total_kv_len],
                    device=device,
                    dtype=torch.int32,
                )  # [2]
                self._max_seqlen_KV = self.static_und_cache_max_len + self._max_gen_cache_tokens + self._gen_len

        if not self.for_cuda_graphs:
            return

        # Constant padded size for the gen-cache buffer (compile-time const).
        T, H_p, W_p = self.vision_token_shapes[0]
        S_super = self.num_action_tokens_per_supertoken + H_p * W_p
        cache_size = self.dual_kv_cache[0].gen_cache.cache_size
        self._max_gen_cache_tokens = (cache_size - 1) * T * S_super
        self._tokens_per_frame = T * S_super
        self._device = device
        # Match the dtype of the populated und cache (set by the eager
        # frame-0 prefill).  At AR loop entry the und cache is always
        # initialized; for safety fall back to float32 if not.
        und_k = self.dual_kv_cache[0].und_cache.k_und
        self._dtype = und_k.dtype if und_k is not None else torch.float32
        self._static_prefix_tokens = self.dual_kv_cache[0].und_cache.cached_len + self._gen_len
        if self.coarse_cuda_graph and self._real_gen_cache_len_t is not None:
            # Whole-forward capture: ``init()`` runs inside the recorded region.  The
            # per-frame offset values are owned by ``prepare_for_coarse_cuda_graph_replay``;
            # a ``fill_`` recorded here would replay the capture-time constants over them.
            return
        self._refresh_static_offsets(self.frame_idx, device)

    def _static_history_plan(self, frame_idx: int) -> tuple[tuple[int, ...] | None, int]:
        """Return ``(explicit history entries or None, real history tokens)`` for ``frame_idx``.

        Transfer limits select whole cache entries (pinned sink pairs plus the newest
        ``max_tokens`` worth) so the static buffer copies exactly what the dynamic
        branch slices; otherwise the ring-native chronological window applies.
        """
        cache_size = self.dual_kv_cache[0].gen_cache.cache_size
        tokens_per_frame = self._tokens_per_frame
        if self.transfer_history_max_tokens is None:
            return None, min(int(frame_idx), cache_size - 1) * tokens_per_frame
        if (
            self.transfer_history_sink_tokens % tokens_per_frame != 0
            or self.transfer_history_max_tokens % tokens_per_frame != 0
        ):
            raise ValueError(
                "Transfer history limits must be whole cache entries on the static AR path: "
                f"sink_tokens={self.transfer_history_sink_tokens}, "
                f"max_tokens={self.transfer_history_max_tokens}, tokens_per_frame={tokens_per_frame}"
            )
        history_indices = self.dual_kv_cache[0].gen_cache.transfer_history_frame_indices(
            frame_idx,
            sink_entries=self.transfer_history_sink_tokens // tokens_per_frame,
            max_recent_entries=self.transfer_history_max_tokens // tokens_per_frame,
        )
        return tuple(history_indices), len(history_indices) * tokens_per_frame

    def _static_history_plan_rows(self, frame_idx: int) -> tuple[list[list[int]], list[int]] | None:
        """Per-row physical slot plans for batched Transfer (``None`` when the shared plan applies).

        Rows get their own plan when there is more than one row or any row restarted its episode
        (``GenKVCache.reset_rows``); each row's plan is a fresh-cache plan in its local frame count.
        """
        gen_cache = self.dual_kv_cache[0].gen_cache
        if self.transfer_history_max_tokens is None or (self._batch_size == 1 and not gen_cache.has_row_resets):
            return None
        tokens_per_frame = self._tokens_per_frame
        sink_entries = self.transfer_history_sink_tokens // tokens_per_frame
        max_recent_entries = self.transfer_history_max_tokens // tokens_per_frame
        row_slots = [
            gen_cache.row_history_slots(row, frame_idx, sink_entries, max_recent_entries)
            for row in range(self._batch_size)
        ]
        return row_slots, [len(slots) * tokens_per_frame for slots in row_slots]

    def _refresh_static_offsets(self, frame_idx: int, device: torch.device) -> None:
        """(Re)compute the static-flavor history plan and the varlen offset tensors for ``frame_idx``.

        The ``[1]`` / ``[2]`` int32 tensors are created once and afterwards updated in
        place: their *addresses* are baked into CUDA graphs (per-block trees or a
        forward-scope capture) and only their *values* change per frame.  The update
        must run *outside* any whole-forward capture: a recorded ``fill_`` replays its
        capture-time constants, so the coarse flavor refreshes only from
        ``prepare_for_coarse_cuda_graph_replay``.
        """
        self._static_history_indices, real_len = self._static_history_plan(frame_idx)
        self._static_row_slots = None
        real_lens = [real_len] * self._batch_size
        row_plan = self._static_history_plan_rows(frame_idx)
        if row_plan is not None:
            self._static_row_slots, real_lens = row_plan
            real_len = real_lens[0]
        und_cache = self.dual_kv_cache[0].und_cache
        s_und = und_cache.cached_len  # padded und length U (= the single prompt length at B=1)
        rows = self._batch_size
        row_stride = s_und + self._gen_len + self._max_gen_cache_tokens
        self._max_seqlen_KV = row_stride
        shared_single_row = rows == 1 and self._static_row_slots is None
        if shared_single_row:
            real_total_kv_len = s_und + self._gen_len + real_len
            cu_q_values = [0, self._gen_len]
            cu_kv_values = [0, real_total_kv_len]
        else:
            # Row r: [pad | und_r right-aligned to r*R+U | curr at r*R+U | hist_r | pad]; entry 2r is the
            # row, entry 2r+1 the zero-query dummy that owns the gap to the next row.
            und_lens = und_cache.cached_lens if len(und_cache.cached_lens) == rows else (s_und,) * rows
            cu_q_values, cu_kv_values = [0], []
            for row in range(rows):
                cu_kv_values += [
                    row * row_stride + s_und - und_lens[row],
                    row * row_stride + s_und + self._gen_len + real_lens[row],
                ]
                cu_q_values += [(row + 1) * self._gen_len] if row == rows - 1 else [(row + 1) * self._gen_len] * 2
        if self._real_gen_cache_len_t is None or self._cu_seqlens_q_t is None or self._cu_seqlens_kv_t is None:
            # Pre-build varlen offsets *outside* the compiled region.  Doing the
            # construction inside the captured graph forces Dynamo to specialize
            # on the contained Python ints (gen_len, S_und, ...) and emit
            # value-specific guards -- every frame retraces and blows past the
            # recompile limit.  Building them here keeps the captured region's
            # view as plain tensor inputs whose values change per frame but whose
            # shapes/addresses Dynamo never inspects.
            self._real_gen_cache_len_t = to_device_nonblocking(
                torch.tensor([real_len], dtype=torch.int32), device
            )  # [1]
            self._cu_seqlens_q_t = to_device_nonblocking(torch.tensor(cu_q_values, dtype=torch.int32), device)  # [2B]
            self._cu_seqlens_kv_t = to_device_nonblocking(torch.tensor(cu_kv_values, dtype=torch.int32), device)  # [2B]
        else:
            self._real_gen_cache_len_t.fill_(real_len)
            if shared_single_row:
                self._cu_seqlens_q_t[1:].fill_(self._gen_len)
                self._cu_seqlens_kv_t[1:].fill_(real_total_kv_len)
            else:
                # Same addresses, new values: an asynchronous copy from pinned host memory.
                self._cu_seqlens_kv_t.copy_(
                    to_device_nonblocking(torch.tensor(cu_kv_values, dtype=torch.int32), device)
                )

    def read_for_layer(self, layer_idx: int) -> ARMemoryValue:
        cache = self.dual_kv_cache[layer_idx]

        und_k_cached, und_v_cached = (
            cache.und_cache.get() if cache.und_cache.is_initialized else (None, None)
        )  # [B,S_und,H,D] each or None

        if not self.for_cuda_graphs:
            if self.coarse_cuda_graph:
                assert self._num_kv_heads is not None
                assert self._head_dim is not None
                gen_k_hist, gen_v_hist, _ = cache.gen_cache.fetch_kv_static(
                    self.frame_idx,
                    self._max_gen_cache_tokens,
                    self._tokens_per_frame,
                    num_heads=self._num_kv_heads,
                    head_dim=self._head_dim,
                    device=self._device,
                    dtype=self._dtype,
                )  # [1,S_hist_max,H,D] each
            else:
                gen_k_hist, gen_v_hist = cache.gen_cache.fetch_kv(self.frame_idx)  # [B,S_hist,H,D] each or None
            if self.transfer_history_max_tokens is not None and gen_k_hist is not None:
                if gen_v_hist is None or gen_v_hist.shape[1] != gen_k_hist.shape[1]:
                    raise AssertionError(
                        "transfer history K/V cache lengths differ: "
                        f"k={gen_k_hist.shape[1]}, v={None if gen_v_hist is None else gen_v_hist.shape[1]}"
                    )
                max_tokens = self.transfer_history_max_tokens
                sink_tokens = min(self.transfer_history_sink_tokens, gen_k_hist.shape[1])
                if sink_tokens == 0:
                    if max_tokens == 0:
                        gen_k_hist = None
                        gen_v_hist = None
                    elif gen_k_hist.shape[1] > max_tokens:
                        gen_k_hist = gen_k_hist[:, -max_tokens:]  # [B,S_history_limited,H,D]
                        gen_v_hist = gen_v_hist[:, -max_tokens:]  # [B,S_history_limited,H,D]
                else:
                    recent_available = gen_k_hist.shape[1] - sink_tokens
                    recent_tokens = min(max_tokens, recent_available)
                    sink_k = gen_k_hist[:, :sink_tokens]  # [B,S_sink,H,D]
                    sink_v = gen_v_hist[:, :sink_tokens]  # [B,S_sink,H,D]
                    if recent_tokens == 0:
                        gen_k_hist = sink_k  # [B,S_sink,H,D]
                        gen_v_hist = sink_v  # [B,S_sink,H,D]
                    else:
                        recent_k = gen_k_hist[:, -recent_tokens:]  # [B,S_recent,H,D]
                        recent_v = gen_v_hist[:, -recent_tokens:]  # [B,S_recent,H,D]
                        gen_k_hist = torch.cat((sink_k, recent_k), dim=1)  # [B,S_sink+S_recent,H,D]
                        gen_v_hist = torch.cat((sink_v, recent_v), dim=1)  # [B,S_sink+S_recent,H,D]
            if self.post_saturation_static_compile:
                assert self.static_und_cache_max_len is not None
                assert self._real_und_cache_len_t is not None
                assert cache.und_cache.is_initialized, (
                    "post-saturation static compile requires frame-0 und cache to be populated"
                )
                if self.coarse_cuda_graph:
                    padded_und_kv = self._coarse_padded_und_kv[layer_idx]
                    if padded_und_kv is None:
                        padded_und_kv = cache.und_cache.get_padded(
                            self.static_und_cache_max_len
                        )  # [B,S_und_max,H,D] each
                        self._coarse_padded_und_kv[layer_idx] = padded_und_kv
                    und_k_cached, und_v_cached = padded_und_kv
                else:
                    und_k_cached, und_v_cached = cache.und_cache.get_padded(
                        self.static_und_cache_max_len
                    )  # [B,S_und_max,H,D] each
                torch._dynamo.mark_static(und_k_cached, 1)
                torch._dynamo.mark_static(und_v_cached, 1)
                if gen_k_hist is not None and gen_v_hist is not None:
                    # After window saturation, generated-history length is fixed
                    # for this compile specialization. Mark it static explicitly
                    torch._dynamo.mark_static(gen_k_hist, 1)
                    torch._dynamo.mark_static(gen_v_hist, 1)
                gen_hist_len = gen_k_hist.shape[1] if gen_k_hist is not None else 0
                real_total_kv_len = cache.und_cache.cached_len + gen_hist_len + self._gen_len
                max_total_kv_len = self.static_und_cache_max_len + gen_hist_len + self._gen_len
                if self.coarse_cuda_graph:
                    assert self._cu_seqlens_q_t is not None
                    assert self._cu_seqlens_kv_t is not None
                    cu_seqlens_q_t = self._cu_seqlens_q_t
                    cu_seqlens_kv_t = self._cu_seqlens_kv_t
                    max_total_kv_len = self._max_seqlen_KV
                else:
                    cu_seqlens_q_t = torch.tensor(
                        [0, self._gen_len],
                        device=und_k_cached.device,
                        dtype=torch.int32,
                    )  # [2]
                    cu_seqlens_kv_t = torch.tensor(
                        [0, real_total_kv_len],
                        device=und_k_cached.device,
                        dtype=torch.int32,
                    )  # [2]
                return ARMemoryValue(
                    und_k_cached=und_k_cached,
                    und_v_cached=und_v_cached,
                    gen_k_hist=gen_k_hist,
                    gen_v_hist=gen_v_hist,
                    frame_idx=self.frame_idx,
                    gen_len=self._gen_len,
                    real_und_cache_len_t=self._real_und_cache_len_t,
                    cu_seqlens_q_t=cu_seqlens_q_t,
                    cu_seqlens_kv_t=cu_seqlens_kv_t,
                    max_seqlen_KV=max_total_kv_len,
                    for_cuda_graphs=False,
                    post_saturation_static_compile=True,
                )
            return ARMemoryValue(
                und_k_cached=und_k_cached,
                und_v_cached=und_v_cached,
                gen_k_hist=gen_k_hist,
                gen_v_hist=gen_v_hist,
                frame_idx=self.frame_idx,
                gen_len=self._gen_len,
                batch_size=self._batch_size,
                gen_lens=self._gen_lens,
                und_lens=(cache.und_cache.cached_lens if cache.und_cache.is_initialized else self._current_und_lens),
                for_cuda_graphs=False,
                post_saturation_static_compile=self.post_saturation_static_compile,
            )

        # Static-shape branch: hand the layer one composite ``[und | curr | hist | pad]``
        # buffer per K and V plus the varlen offset tensors.  Shapes and addresses are
        # constant across frames so a single CUDA-graph capture replays; the und region
        # is primed once per generation and the history region rebuilt per frame here
        # (per-block graphs) or in ``prepare_for_coarse_cuda_graph_replay`` (forward scope).
        assert und_k_cached is not None and und_v_cached is not None, (
            "ARMemoryState(for_cuda_graphs=True) requires the und cache to be "
            "populated by frame-0 prefill before entering the AR loop"
        )
        assert self._real_gen_cache_len_t is not None
        assert self._num_kv_heads is not None
        assert self._head_dim is not None
        s_und = cache.und_cache.cached_len
        assert s_und + self._gen_len == self._static_prefix_tokens, (
            f"static prefix mismatch: und={s_und} gen={self._gen_len} prefix={self._static_prefix_tokens}"
        )
        rows = self._batch_size
        und_lens = cache.und_cache.cached_lens if len(cache.und_cache.cached_lens) == rows else None
        kv_k_static, kv_v_static, _ = cache.gen_cache.fetch_kv_static(
            self.frame_idx,
            self._max_gen_cache_tokens,
            self._tokens_per_frame,
            num_heads=self._num_kv_heads,
            head_dim=self._head_dim,
            device=self._device,
            dtype=self._dtype,
            history_frame_indices=(
                None if self._static_history_indices is None else list(self._static_history_indices)
            ),
            prefix_tokens=self._static_prefix_tokens,
            und_kv=(und_k_cached, und_v_cached),
            batch_rows=rows,
            und_lens=und_lens,
            row_slot_lists=self._static_row_slots,
        )
        torch._dynamo.mark_static(kv_k_static, 1)
        torch._dynamo.mark_static(kv_v_static, 1)
        # Legacy single-sample views of the composite buffer: the und prefix and the
        # ``[hist | pad]`` window.  No copies; the block itself reads ``kv_k_static``.
        # Batched rows keep them ``None`` (per-row right-aligned und, strided rows).
        hist_start = self._static_prefix_tokens
        hist_end = hist_start + self._max_gen_cache_tokens
        legacy_und_k = kv_k_static[:, :s_und] if rows == 1 and s_und > 0 else None  # [1,S_und,H_kv,D]
        legacy_und_v = kv_v_static[:, :s_und] if rows == 1 and s_und > 0 else None  # [1,S_und,H_kv,D]
        legacy_gen_k = kv_k_static[:, hist_start:hist_end] if rows == 1 else None  # [1,max_gen_tokens,H_kv,D]
        legacy_gen_v = kv_v_static[:, hist_start:hist_end] if rows == 1 else None  # [1,max_gen_tokens,H_kv,D]
        return ARMemoryValue(
            und_k_cached=legacy_und_k,
            und_v_cached=legacy_und_v,
            gen_k_hist=None,
            gen_v_hist=None,
            frame_idx=self.frame_idx,
            gen_len=self._gen_len,
            batch_size=rows,
            gen_lens=self._gen_lens,
            und_lens=(cache.und_cache.cached_lens if self.batched else ()),
            gen_k_buf_full=legacy_gen_k,
            gen_v_buf_full=legacy_gen_v,
            kv_k_static=kv_k_static,
            kv_v_static=kv_v_static,
            static_curr_offset=s_und,
            static_hist_offset=self._static_prefix_tokens,
            static_row_stride=(self._max_seqlen_KV if rows > 1 else 0),
            real_gen_cache_len_t=self._real_gen_cache_len_t,
            cu_seqlens_q_t=self._cu_seqlens_q_t,
            cu_seqlens_kv_t=self._cu_seqlens_kv_t,
            max_seqlen_KV=self._max_seqlen_KV,
            for_cuda_graphs=True,
            post_saturation_static_compile=False,
        )

    def _slice_to_local_kv_heads(
        self,
        x: torch.Tensor,  # [B,S,H_kv,D]
    ) -> torch.Tensor:  # [B,S,H_kv_local,D]
        # KV cache storage is expressed in local KV-head shards.  Some callers
        # already pass local K/V [B,S,H_kv/CP,D] after attention-layout
        # conversion; replicated attention I/O can pass full K/V [B,S,H_kv,D]
        # during prefill/refresh.  Slice full K/V here so the cache
        # representation stays [B,S,H_kv/CP,D].
        if self.kv_head_shard_size == 1:
            return x
        assert self._num_kv_heads is not None
        local_heads = self._num_kv_heads // self.kv_head_shard_size
        if x.shape[-2] == local_heads:
            return x
        assert x.shape[-2] == self._num_kv_heads, (
            f"Expected full KV heads ({self._num_kv_heads}) or local KV heads ({local_heads}), got {x.shape[-2]}"
        )
        head_start = self.kv_head_shard_rank * local_heads
        head_end = head_start + local_heads
        return x[..., head_start:head_end, :].contiguous()  # [B,S,H_local,D]

    def write_for_layer(self, layer_idx: int, kv_to_store: KVToStore) -> None:
        gen_k, gen_v, und_k, und_v = kv_to_store
        cache = self.dual_kv_cache[layer_idx]
        if self.write_gen_cache:
            # Prefill/refresh forwards write finalized frame K/V to the gen cache
            gen_k_to_store = self._slice_to_local_kv_heads(gen_k).detach()  # [B,S,H_local,D]
            gen_v_to_store = self._slice_to_local_kv_heads(gen_v).detach()  # [B,S,H_local,D]
            gen_v_to_store = zero_null_action_values(
                gen_v_to_store,
                self.vision_token_shapes,
                self.num_action_tokens_per_supertoken,
                self.null_action_supertokens,
            )  # [B,S,H_local,D]
            if self.stage_gen_cache_writes:
                self._staged_gen_kv[layer_idx] = (gen_k_to_store, gen_v_to_store)
            else:
                cache.gen_cache.store_kv(gen_k_to_store, gen_v_to_store, frame_idx=self.frame_idx)
        if not cache.und_cache.is_initialized:
            und_k_to_store = self._slice_to_local_kv_heads(und_k)  # [B,S,H_local,D]
            und_v_to_store = self._slice_to_local_kv_heads(und_v)  # [B,S,H_local,D]
            cache.und_cache.store(
                und_k_to_store,
                und_v_to_store,
                lengths=self._current_und_lens if self.batched else None,
            )

    def prepare_for_coarse_cuda_graph_replay(self, frame_idx: int) -> None:
        """Refresh fixed-address history buffers (and, on the static flavor, the offset tensors) before replay."""
        if not self.coarse_cuda_graph:
            raise RuntimeError("prepare_for_coarse_cuda_graph_replay requires coarse_cuda_graph=True")
        assert self._num_kv_heads is not None
        assert self._head_dim is not None
        self.frame_idx = frame_idx
        if self.for_cuda_graphs:
            self._refresh_static_offsets(frame_idx, self._device)
        for cache in self.dual_kv_cache:
            cache.gen_cache.fetch_kv_static(
                frame_idx,
                self._max_gen_cache_tokens,
                self._tokens_per_frame,
                num_heads=self._num_kv_heads,
                head_dim=self._head_dim,
                device=self._device,
                dtype=self._dtype,
                history_frame_indices=(
                    None if self._static_history_indices is None else list(self._static_history_indices)
                ),
                prefix_tokens=self._static_prefix_tokens if self.for_cuda_graphs else 0,
                und_kv=cache.und_cache.get() if self.for_cuda_graphs else None,
                batch_rows=self._batch_size if self.for_cuda_graphs else 1,
                und_lens=(
                    cache.und_cache.cached_lens
                    if self.for_cuda_graphs and len(cache.und_cache.cached_lens) == self._batch_size
                    else None
                ),
                row_slot_lists=self._static_row_slots if self.for_cuda_graphs else None,
            )

    def commit_staged_gen_cache(self, frame_idx: int) -> None:
        """Commit refresh K/V after capture or replay completes."""
        if not self.stage_gen_cache_writes:
            raise RuntimeError("commit_staged_gen_cache requires stage_gen_cache_writes=True")
        self.frame_idx = frame_idx
        for layer_idx, staged_kv in enumerate(self._staged_gen_kv):
            if staged_kv is None:
                raise RuntimeError(f"CUDA Graph refresh did not stage K/V for layer {layer_idx}")
            gen_k_to_store, gen_v_to_store = staged_kv
            self.dual_kv_cache[layer_idx].gen_cache.store_kv(
                gen_k_to_store,
                gen_v_to_store,
                frame_idx=frame_idx,
            )

    def is_gen_only(self) -> bool:
        return self.frame_idx > 0 and self.dual_kv_cache[0].und_cache.is_initialized


__all__ = [
    "KVCache",
    "UndKVCache",
    "GenKVCache",
    "KVBufferPool",
    "DualKVCache",
    "KVCacheTrainMemoryState",
    "KVTrainMemoryValue",
    "TFReplayCleanMemoryValue",
    "TFNoisyMemoryValue",
    "TeacherForcingMemoryState",
    "FlexARMemoryState",
    "FlexARMemoryValue",
    "ARMemoryState",
    "ARMemoryValue",
    "zero_null_action_values",
    "KVToStore",
    "SequencePack",
    "MAX_CACHE_SIZE",
]
