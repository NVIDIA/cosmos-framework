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

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

# Re-exported from memory.py for backward compatibility.
from cosmos_framework.model.generator.utils.memory import KVToStore, MemoryState, MemoryValue
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
        self.is_initialized = False

    def store(self, k: torch.Tensor, v: torch.Tensor) -> None:
        """Store und K/V tensors.

        The new caption will overwrite the existing caption in the cache.

        Args:
            k: Key tensor with RoPE applied [B,S_und,H,D].
                S_und is the number of understanding (text) tokens.
            v: Value tensor [B,S_und,H,D].
        """
        # Detach to prevent gradient flow (same as KVCache).
        # Clone to escape CUDA-graph-managed storage (see KVCache.store_kv).
        self.k_und = k.detach().clone()  # [B,S_und,H,D]
        self.v_und = v.detach().clone()  # [B,S_und,H,D]
        self.cached_len = k.shape[1]
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
    ) -> None:
        # initialize static KV cache used or AR inference use case.
        # This buffer hosts materialization of the rolling window in chronological order.
        self._static_k_buf: torch.Tensor | None = None
        self._static_v_buf: torch.Tensor | None = None
        self._static_valid_frame_idx: int | None = None
        self._static_real_len: int = 0
        self._static_tokens_per_frame: int = 0
        super().__init__(cache_size=cache_size, backend=backend, attention_sink_size=attention_sink_size)

    def reset(self) -> None:
        """Reset cache state and discard static inference workspaces."""
        super().reset()
        self._static_k_buf = None
        self._static_v_buf = None
        self._static_valid_frame_idx = None
        self._static_real_len = 0
        self._static_tokens_per_frame = 0

    def store_kv(self, k: torch.Tensor, v: torch.Tensor, frame_idx: int) -> None:
        """Store K/V and invalidate static read workspaces."""
        super().store_kv(k, v, frame_idx)
        # A store can change the chronological history for the next AR frame,
        # and can also overwrite a physical ring-buffer slot that the static
        # buffer previously copied from.  Mark the workspace stale.
        # refresh happens once on the next static read.
        self._static_valid_frame_idx = None

    def _ensure_static_history_buffer_allocated(
        self,
        max_tokens: int,
        *,
        num_heads: int,
        head_dim: int,
        device: torch.device | None,
        dtype: torch.dtype | None,
    ) -> None:
        """Allocate the fixed-size history buffer used by Cuda Graph AR inference."""
        if device is None:
            raise ValueError("device is required when allocating static gen KV buffers")
        if dtype is None:
            raise ValueError("dtype is required when allocating static gen KV buffers")

        expected_shape = (1, max_tokens, num_heads, head_dim)
        needs_alloc = (
            self._static_k_buf is None
            or self._static_v_buf is None
            or tuple(self._static_k_buf.shape) != expected_shape
            or tuple(self._static_v_buf.shape) != expected_shape
            or self._static_k_buf.device != device
            or self._static_v_buf.device != device
            or self._static_k_buf.dtype != dtype
            or self._static_v_buf.dtype != dtype
        )
        if not needs_alloc:
            return

        self._static_k_buf = torch.empty(expected_shape, device=device, dtype=dtype)  # [1,S_max,H,D]
        self._static_v_buf = torch.empty(expected_shape, device=device, dtype=dtype)  # [1,S_max,H,D]
        self._static_valid_frame_idx = None
        self._static_real_len = 0
        self._static_tokens_per_frame = 0

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

    def _rebuild_static_history(self, frame_idx: int, max_tokens: int, tokens_per_frame: int) -> int:
        """Rebuild the fixed history buffer from the list-backed circular cache."""
        assert self._static_k_buf is not None
        assert self._static_v_buf is not None

        current_idx = int(frame_idx)
        if current_idx <= 0:
            # Frame 0 has no gen history.  The returned buffer may contain
            # uninitialized tail data, but real_len=0 and the static AR path
            # will not expose any history tokens to attention.
            self._static_valid_frame_idx = current_idx
            self._static_real_len = 0
            self._static_tokens_per_frame = tokens_per_frame
            return 0

        # Chronological logical frames in history.  This mirrors ``fetch_kv``
        # exactly, including the cache_size-1 history limit that leaves one
        # slot available for the current frame after it is stored.
        history_indices: list[int] = []
        history_k_entries: list[object] = []
        history_v_entries: list[object] = []
        for logical_idx in self._history_frame_indices(current_idx):
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

        # Decode/concatenate the whole chronological window at once.  This
        # avoids launching two copies per cached frame while staging inputs for
        # a coarse graph replay.
        k_history, v_history = self.backend.decode_many(
            history_k_entries,
            history_v_entries,
            slots=history_indices,
        )  # [B,S_hist,H,D] each
        dst_end = len(history_indices) * tokens_per_frame
        if k_history.shape[1] != dst_end or v_history.shape[1] != dst_end:
            raise AssertionError(
                f"Static AR cache requires {tokens_per_frame} tokens per frame: "
                f"expected total={dst_end}, got k={k_history.shape[1]}, v={v_history.shape[1]}"
            )
        if dst_end > max_tokens:
            raise AssertionError(f"Static AR cache overflow: trying to write {dst_end} tokens into {max_tokens}")
        # Copy only the real chronological prefix.  The padded suffix is
        # deliberately left untouched and excluded by ``cu_seqlens_kv_t``.
        self._static_k_buf[:, :dst_end].copy_(k_history.detach())  # [1,S_hist,H,D]
        self._static_v_buf[:, :dst_end].copy_(v_history.detach())  # [1,S_hist,H,D]

        self._static_valid_frame_idx = current_idx
        self._static_real_len = dst_end
        self._static_tokens_per_frame = tokens_per_frame
        return dst_end

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
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Fetch cached K/V history from a persistent fixed-size buffer.

        Unlike ``fetch_kv_padded``, this does not ``cat`` and ``pad`` on every
        read.  It rebuilds the persistent buffer only when the requested frame
        changes or the cache is invalidated by a store/reset.
        """
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
        )
        if self._static_valid_frame_idx != int(frame_idx) or self._static_tokens_per_frame != tokens_per_frame:
            self._rebuild_static_history(frame_idx, max_tokens, tokens_per_frame)

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

    # Transfer control visibility. ``global_control`` preserves the original
    # full-clip behavior; ``causal_control`` uses the [1, C, C, ...] chunk grid.
    transfer_control_attention_mode: str = "global_control"
    frames_per_chunk: int = 1

    @property
    def supports_context_parallel_attention(self) -> bool:
        return True


@dataclass
class TFNoisyMemoryValue(KVTrainMemoryValue):
    """Read-only container for Pass 2 of teacher forcing.

    Inherits all rolling-cache and text-cache fields from ``KVTrainMemoryValue``.
    Adds the current-segment clean gen K/V captured during Pass 1.
    """

    cached_clean_gen_k: torch.Tensor  # [1, S_clean, H_kv, D]
    cached_clean_gen_v: torch.Tensor  # [1, S_clean, H_kv, D]
    # Latent frames per causal chunk (chunk partition is [1, C, C, ...]; the
    # first chunk is always a single frame).  1 == framewise teacher forcing.
    frames_per_chunk: int = 1
    # Transfer control visibility; see ``TFReplayCleanMemoryValue``.
    transfer_control_attention_mode: str = "global_control"

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

    Pass 1: behaves identically to ``KVCacheTrainMemoryState`` (temporal-causal
    attention on clean data) and captures gen K/V per layer in
    ``_clean_gen_kv``. The legacy three-way path also writes the rolling cache;
    the two-way Flex path stores only the selected clean target K/V and skips
    the otherwise-unused full rolling-cache copy.

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
        transfer_control_attention_mode: str = "global_control",
        context_parallel_size: int = 1,
        selected_clean_gen_token_indexes: torch.Tensor | None = None,
        selected_clean_gen_padded_capacity: int = 0,
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
        self.transfer_control_attention_mode = transfer_control_attention_mode
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
        self._clean_gen_kv: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * len(dual_kv_cache)

    def _read_flex_base_value(self) -> KVTrainMemoryValue:
        """Build the shape-minimal base fields unused by two-way Flex attention."""
        assert self.has_new_caption is not None
        assert self.has_caption is not None
        assert self.has_cached_gen is not None
        assert self.und_kv_offsets is not None
        assert self.gen_q_offsets is not None
        assert self.gen_ca_cached_kv_offsets is not None
        dummy_kv = torch.zeros(
            1,
            1,
            self.num_kv_heads,
            self.head_dim,
            device=self._device,
            dtype=self._dtype,
        )  # [1,1,H_kv,D]
        return KVTrainMemoryValue(
            vision_token_shapes=self.vision_token_shapes,
            num_action_tokens_per_supertoken=self.num_action_tokens_per_supertoken,
            has_new_caption=self.has_new_caption,
            has_caption=self.has_caption,
            has_cached_gen=self.has_cached_gen,
            und_kv_offsets=self.und_kv_offsets,
            gen_q_offsets=self.gen_q_offsets,
            gen_ca_cached_kv_offsets=self.gen_ca_cached_kv_offsets,
            cached_und_k=dummy_kv,
            cached_und_v=dummy_kv,
            cached_gen_k=dummy_kv,
            cached_gen_v=dummy_kv,
            max_gen_cache_tokens=1,
            clamp_empty_varlen_kv=self.clamp_empty_varlen_kv,
        )

    def read_for_layer(self, layer_idx: int) -> KVTrainMemoryValue | TFNoisyMemoryValue:
        if self.pass_number == 1:
            base_value = (
                self._read_flex_base_value()
                if self.selected_clean_gen_token_indexes is not None
                else super().read_for_layer(layer_idx)
            )
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
                transfer_control_attention_mode=self.transfer_control_attention_mode,
                frames_per_chunk=self.frames_per_chunk,
            )

        # Pass 2: wrap the parent's KVTrainMemoryValue with clean gen K/V.
        base_value = (
            self._read_flex_base_value()
            if self.selected_clean_gen_token_indexes is not None
            else super().read_for_layer(layer_idx)
        )
        clean_kv = self._clean_gen_kv[layer_idx]
        assert clean_kv is not None, f"Clean gen K/V not captured for layer {layer_idx}"
        clean_k, clean_v = clean_kv
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
            frames_per_chunk=self.frames_per_chunk,
            transfer_control_attention_mode=self.transfer_control_attention_mode,
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
            if self.selected_clean_gen_token_indexes is not None:
                return
            super().write_for_layer(layer_idx, kv_to_store)
            return

        # Pass 2: no-op. Clean KV already written in Pass 1.

    def is_gen_only(self) -> bool:
        return False


@dataclass
class FlexARMemoryValue(MemoryValue):
    """Read-only fixed-capacity K/V suffix for multiview WSM AR inference."""

    cached_gen_k: torch.Tensor | None  # [1,S_memory,H_kv,D] or None during prefill
    cached_gen_v: torch.Tensor | None  # [1,S_memory,H_kv,D] or None during prefill


class FlexARMemoryState(MemoryState):
    """Capture and serve the fixed-size FlexAttention memory suffix.

    The cache list is shared by the prefill, denoising, and clean-refresh
    states of one CFG branch. Denoising states omit ``write_indexes`` and are
    read-only; prefill/refresh states select clean GEN K/V and write them into
    the requested suffix range.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        memory_seq_len: int,
        cache: list[tuple[torch.Tensor, torch.Tensor] | None] | None = None,
        write_indexes: torch.Tensor | None = None,
        write_offset: int = 0,
    ) -> None:
        if memory_seq_len < 1:
            raise ValueError(f"memory_seq_len must be >= 1, got {memory_seq_len}.")
        if write_offset < 0:
            raise ValueError(f"write_offset must be >= 0, got {write_offset}.")
        self.memory_seq_len = memory_seq_len
        self.cache = [None] * num_layers if cache is None else cache
        if len(self.cache) != num_layers:
            raise ValueError(f"Expected {num_layers} cache entries, got {len(self.cache)}.")
        self.write_indexes = write_indexes
        self.write_offset = write_offset

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
        if write_end > self.memory_seq_len:
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
        cached_k[:, self.write_offset : write_end].copy_(selected_k)  # [1,S_write,H_kv,D]
        cached_v[:, self.write_offset : write_end].copy_(selected_v)  # [1,S_write,H_kv,D]

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
    gen_k_buf_full: torch.Tensor | None = None
    gen_v_buf_full: torch.Tensor | None = None
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
            assert post_saturation_static_compile, "coarse_cuda_graph=True requires post-saturation static compile"
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
        self._device: torch.device = torch.device("cpu")
        self._dtype: torch.dtype = torch.float32

    def init(self, hidden_states: dict, device: torch.device) -> None:
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

        # Real (non-padding) length of the cached gen history at this
        # frame.  Same rolling-buffer math as KVCacheTrainMemoryState.
        # Held as an int32 tensor of shape ``[1]`` so its *value* can vary
        # across CUDA-graph replays while the *shape* stays static.
        real_len = min(self.frame_idx, cache_size - 1) * T * S_super
        self._real_gen_cache_len_t = torch.tensor([real_len], device=device, dtype=torch.int32)

        self._device = device
        # Match the dtype of the populated und cache (set by the eager
        # frame-0 prefill).  At AR loop entry the und cache is always
        # initialized; for safety fall back to float32 if not.
        und_k = self.dual_kv_cache[0].und_cache.k_und
        self._dtype = und_k.dtype if und_k is not None else torch.float32

        # Pre-build varlen offsets *outside* the compiled region.  Doing the
        # construction inside the captured graph forces Dynamo to specialize
        # on the contained Python ints (gen_len, S_und, ...) and emit
        # value-specific guards (e.g. ``memory_value.frame_idx == N``) — every
        # frame retraces and blows past the recompile limit.  Building them
        # here keeps the captured region's view as plain ``[2]`` tensor inputs
        # whose values change per frame but whose shapes/addresses Dynamo
        # never inspects.
        s_und = self.dual_kv_cache[0].und_cache.cached_len
        real_total_kv_len = s_und + self._gen_len + real_len
        self._cu_seqlens_q_t = torch.tensor([0, self._gen_len], device=device, dtype=torch.int32)
        self._cu_seqlens_kv_t = torch.tensor([0, real_total_kv_len], device=device, dtype=torch.int32)
        self._max_seqlen_KV = s_und + self._gen_len + self._max_gen_cache_tokens

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
                for_cuda_graphs=False,
                post_saturation_static_compile=self.post_saturation_static_compile,
            )

        # Static-shape branch: hand the layer the full preallocated gen
        # buffer + a scalar real-length tensor.  Shapes are constant
        # across frames so a single CUDA-graph capture replays.
        assert und_k_cached is not None, (
            "ARMemoryState(for_cuda_graphs=True) requires the und cache to be "
            "populated by frame-0 prefill before entering the AR loop"
        )
        assert self._real_gen_cache_len_t is not None
        assert self._num_kv_heads is not None
        assert self._head_dim is not None
        gen_k_buf, gen_v_buf, _ = cache.gen_cache.fetch_kv_static(
            self.frame_idx,
            self._max_gen_cache_tokens,
            self._tokens_per_frame,
            num_heads=self._num_kv_heads,
            head_dim=self._head_dim,
            device=self._device,
            dtype=self._dtype,
        )
        torch._dynamo.mark_static(gen_k_buf, 1)
        torch._dynamo.mark_static(gen_v_buf, 1)
        return ARMemoryValue(
            und_k_cached=und_k_cached,
            und_v_cached=und_v_cached,
            gen_k_hist=None,
            gen_v_hist=None,
            frame_idx=self.frame_idx,
            gen_len=self._gen_len,
            gen_k_buf_full=gen_k_buf,
            gen_v_buf_full=gen_v_buf,
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
            cache.und_cache.store(und_k_to_store, und_v_to_store)

    def prepare_for_coarse_cuda_graph_replay(self, frame_idx: int) -> None:
        """Refresh fixed-address history buffers before replaying a coarse graph."""
        if not self.coarse_cuda_graph:
            raise RuntimeError("prepare_for_coarse_cuda_graph_replay requires coarse_cuda_graph=True")
        assert self._num_kv_heads is not None
        assert self._head_dim is not None
        self.frame_idx = frame_idx
        for cache in self.dual_kv_cache:
            cache.gen_cache.fetch_kv_static(
                frame_idx,
                self._max_gen_cache_tokens,
                self._tokens_per_frame,
                num_heads=self._num_kv_heads,
                head_dim=self._head_dim,
                device=self._device,
                dtype=self._dtype,
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
