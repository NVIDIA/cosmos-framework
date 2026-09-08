# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""
Imaginaire4 Modules:
Pluggable storage formats for cached key/value tensors.

A storage backend converts a BF16 key or value tensor into some stored
representation and back, letting a cache trade memory for precision without
changing its own logic. Backends encode the key and value with separate calls.
They can decode one entry or a sequence of entries concatenated along the
token dimension.
"""

from __future__ import annotations

import abc
from typing import Any

import torch

try:
    import triton  # pyright: ignore[reportMissingImports]
    import triton.language as tl  # pyright: ignore[reportMissingImports]
except ImportError:
    triton = None
    tl = None


# Keep the Triton JIT kernel at module scope so decode calls reuse its compiled form.
_decode_fp8_kv_many_kernel: Any = None
if triton is not None and tl is not None:

    @triton.jit
    def _decode_fp8_kv_many_kernel_impl(
        k_value_ptrs,
        v_value_ptrs,
        k_scales,
        v_scales,
        ordered_slots,
        out_k,
        out_v,
        batch_size: tl.constexpr,  # pyright: ignore[reportInvalidTypeForm]
        tokens_per_entry: tl.constexpr,  # pyright: ignore[reportInvalidTypeForm]
        total_seq: tl.constexpr,  # pyright: ignore[reportInvalidTypeForm]
        num_heads: tl.constexpr,  # pyright: ignore[reportInvalidTypeForm]
        head_dim: tl.constexpr,  # pyright: ignore[reportInvalidTypeForm]
        block_size: tl.constexpr,  # pyright: ignore[reportInvalidTypeForm]
    ) -> None:
        entry_ord = tl.program_id(1)  # []
        slot = tl.load(ordered_slots + entry_ord)  # []
        offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)  # [block_size]
        entry_numel = batch_size * tokens_per_entry * num_heads * head_dim  # []
        mask = offsets < entry_numel  # [block_size]

        dim_idx = offsets % head_dim  # [block_size]
        row_idx = offsets // head_dim  # [block_size]
        head_idx = row_idx % num_heads  # [block_size]
        seq_idx = (row_idx // num_heads) % tokens_per_entry  # [block_size]
        batch_idx = row_idx // (tokens_per_entry * num_heads)  # [block_size]
        seq_start = entry_ord * tokens_per_entry  # []
        out_offsets = (
            (batch_idx * total_seq + seq_start + seq_idx) * num_heads + head_idx
        ) * head_dim + dim_idx  # [block_size]

        k_base = tl.load(k_value_ptrs + slot).to(tl.pointer_type(tl.float8e4nv))  # []
        v_base = tl.load(v_value_ptrs + slot).to(tl.pointer_type(tl.float8e4nv))  # []
        k_scale = tl.load(k_scales + slot).to(tl.bfloat16)  # []
        v_scale = tl.load(v_scales + slot).to(tl.bfloat16)  # []
        k_fp8 = tl.load(k_base + offsets, mask=mask).to(tl.bfloat16)  # [block_size]
        v_fp8 = tl.load(v_base + offsets, mask=mask).to(tl.bfloat16)  # [block_size]
        tl.store(out_k + out_offsets, k_fp8 * k_scale, mask=mask)
        tl.store(out_v + out_offsets, v_fp8 * v_scale, mask=mask)

    _decode_fp8_kv_many_kernel = _decode_fp8_kv_many_kernel_impl

VALID_KV_CACHE_DTYPES: frozenset[str] = frozenset({"fp8"})
VALID_FP8_KERNEL_IMPLS: frozenset[str] = frozenset({"torch", "triton"})


def validate_fp8_kernel_impl(kernel_impl: str, *, parameter_name: str = "kernel_impl") -> None:
    if kernel_impl not in VALID_FP8_KERNEL_IMPLS:
        valid_values = ", ".join(sorted(VALID_FP8_KERNEL_IMPLS))
        raise ValueError(f"{parameter_name} must be one of: {valid_values}; got {kernel_impl!r}")


def validate_kv_cache_dtype(kv_cache_dtype: str | None, *, parameter_name: str = "kv_cache_dtype") -> None:
    """Validate the opt-in KV cache storage dtype."""
    if kv_cache_dtype is None:
        return
    if kv_cache_dtype not in VALID_KV_CACHE_DTYPES:
        raise ValueError(f"{parameter_name} must be None or 'fp8'; got {kv_cache_dtype!r}")


def parse_fp8_entry(entry: object) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not isinstance(entry, tuple) or len(entry) != 2:
        return None
    fp8, scale = entry
    if not isinstance(fp8, torch.Tensor) or not isinstance(scale, torch.Tensor):
        return None
    if (
        fp8.ndim != 4
        or fp8.dtype != torch.float8_e4m3fn
        or not fp8.is_contiguous()
        or scale.numel() != 1
        or scale.device != fp8.device
    ):
        return None
    return fp8, scale


def decode_fp8_kv_many_triton(
    *,
    k_value_ptrs: torch.Tensor,  # [N_slots]
    v_value_ptrs: torch.Tensor,  # [N_slots]
    k_scales: torch.Tensor,  # [N_slots]
    v_scales: torch.Tensor,  # [N_slots]
    ordered_slots: torch.Tensor,  # [E]
    batch_size: int,
    tokens_per_entry: int,
    total_seq: int,
    num_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:  # returns [B,S_total,H,D] each
    """Decode fixed-size FP8 entries with the Triton fast path."""
    if triton is None or _decode_fp8_kv_many_kernel is None:
        return None
    if total_seq <= 0 or tokens_per_entry <= 0 or batch_size <= 0 or num_heads <= 0 or head_dim <= 0:
        return None
    if (
        k_value_ptrs.device.type != "cuda"
        or v_value_ptrs.device != k_value_ptrs.device
        or k_scales.device != k_value_ptrs.device
        or v_scales.device != k_value_ptrs.device
        or ordered_slots.device != k_value_ptrs.device
    ):
        return None

    out_k = torch.empty(
        (batch_size, total_seq, num_heads, head_dim),
        device=k_value_ptrs.device,
        dtype=torch.bfloat16,
    )  # [B,S_total,H,D]
    out_v = torch.empty(
        (batch_size, total_seq, num_heads, head_dim),
        device=k_value_ptrs.device,
        dtype=torch.bfloat16,
    )  # [B,S_total,H,D]

    block_size = 1024
    entry_numel = batch_size * tokens_per_entry * num_heads * head_dim
    grid = (triton.cdiv(entry_numel, block_size), ordered_slots.numel())
    try:
        _decode_fp8_kv_many_kernel[grid](
            k_value_ptrs,
            v_value_ptrs,
            k_scales,
            v_scales,
            ordered_slots,
            out_k,
            out_v,
            batch_size,
            tokens_per_entry,
            total_seq,
            num_heads,
            head_dim,
            block_size,
        )
    except Exception as exc:
        # Preserve the original launch error as the cause for kernel debugging.
        raise RuntimeError("FP8 Triton decode kernel launch failed") from exc
    return out_k, out_v


class KVStorageBackend(abc.ABC):
    """Storage format for a single cached key or value tensor.

    `encode` turns a BF16 tensor into an opaque stored representation and
    `decode` reconstructs the BF16 tensor from it. The representation is
    private to the backend: it may be the tensor itself, a quantized form, or
    a tuple bundling extra metadata such as scale tensors. The interface is
    per-tensor, so callers invoke it once for the key and once for the value.
    """

    @abc.abstractmethod
    def encode(self, tensor: torch.Tensor) -> object:
        """Convert a BF16 tensor into this backend's stored representation."""

    @abc.abstractmethod
    def decode(self, entry: object) -> torch.Tensor:
        """Reconstruct a BF16 tensor from this backend's stored representation."""

    def decode_many(
        self,
        k_entries: list[object],
        v_entries: list[object],
        slots: list[int] | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode paired K/V entries and concatenate them along the sequence dimension."""
        del slots
        assert len(k_entries) == len(v_entries), "K and V entries must have the same number of chunks"
        decoded_k_entries = [self.decode(entry) for entry in k_entries]  # each entry: [B,S_i,H,D]
        decoded_v_entries = [self.decode(entry) for entry in v_entries]  # each entry: [B,S_i,H,D]
        if dtype is not None:
            decoded_k_entries = [tensor.to(dtype) for tensor in decoded_k_entries]  # each entry: [B,S_i,H,D]
            decoded_v_entries = [tensor.to(dtype) for tensor in decoded_v_entries]  # each entry: [B,S_i,H,D]
        return torch.cat(decoded_k_entries, dim=1), torch.cat(decoded_v_entries, dim=1)  # [B,S_total,H,D] each

    def reset_kv_cache_state(self, cache_size: int) -> None:
        """Reset backend-owned state tied to a KV cache instance."""
        del cache_size

    def update_cached_kv_metadata(self, cache_idx: int, k_entry: object, v_entry: object) -> None:
        """Observe a stored K/V pair after the cache updates its slot."""
        del cache_idx, k_entry, v_entry


class BF16StorageBackend(KVStorageBackend):
    """Passthrough backend that stores the tensor unchanged.

    The stored representation is a plain ``torch.Tensor`` (a detached clone),
    so a round-trip is bit-exact. Keeping the entry a bare tensor rather than
    wrapping it in a tuple matters: it preserves the storage shape that other
    code paths may depend on, so swapping this backend in for direct tensor
    storage changes nothing observable.
    """

    def encode(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.detach().clone()

    def decode(self, entry: object) -> torch.Tensor:
        assert isinstance(entry, torch.Tensor), "BF16 entry must be a torch.Tensor"
        return entry


class FP8StorageBackend(KVStorageBackend):
    """Stores a tensor as FP8 (e4m3) values with one scale per cached tensor.

    The e4m3 format is chosen over e5m2 because key/value activations cluster
    near zero; e4m3's extra mantissa bit (3 vs 2) preserves more precision in
    that regime than e5m2's wider exponent range would buy.
    """

    @staticmethod
    def _fp8_tensor_scale(tensor: torch.Tensor) -> torch.Tensor:  # tensor: [B,S,H,D], returns [1]
        fp8_max = torch.finfo(torch.float8_e4m3fn).max
        scale = tensor.abs().amax().reshape(1) / fp8_max  # [1]
        return scale.clamp_min(torch.finfo(scale.dtype).tiny)  # [1]

    @staticmethod
    def _encode_fp8_torch(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # tensor: [B,S,H,D]
        detached = tensor.detach()  # [B,S,H,D]
        scale = FP8StorageBackend._fp8_tensor_scale(detached)  # [1]
        normalized = (detached / scale).to(torch.bfloat16)  # [B,S,H,D]
        fp8 = normalized.to(torch.float8_e4m3fn).clone()  # [B,S,H,D]
        return fp8, scale

    @staticmethod
    def _decode_fp8(entry: object) -> torch.Tensor:
        assert isinstance(entry, tuple), "FP8 entry must be a (fp8_values, scale) tuple"
        fp8, scale = entry
        assert isinstance(fp8, torch.Tensor), "FP8 values must be a torch.Tensor"
        assert isinstance(scale, torch.Tensor), "FP8 scale must be a torch.Tensor"
        assert scale.numel() == 1, f"FP8 scale must have one element, got {scale.numel()}"
        return fp8.to(torch.bfloat16) * scale  # [B,S,H,D]

    def __init__(self, kv_cache_dtype: str = "fp8", kernel_impl: str = "triton") -> None:
        if kv_cache_dtype != "fp8":
            raise ValueError(f"kv_cache_dtype must be 'fp8'; got {kv_cache_dtype!r}")
        validate_fp8_kernel_impl(kernel_impl)
        self.kv_cache_dtype: str = kv_cache_dtype
        self.kernel_impl: str = kernel_impl
        # Triton metadata stores raw tensor pointers, so slot validity must be explicit.
        self._metadata_cache_size: int = 0
        self._metadata_k_value_ptrs: torch.Tensor | None = None
        self._metadata_v_value_ptrs: torch.Tensor | None = None
        self._metadata_k_scales: torch.Tensor | None = None
        self._metadata_v_scales: torch.Tensor | None = None
        self._metadata_tokens_per_entry: int | None = None
        self._metadata_batch_size: int | None = None
        self._metadata_num_heads: int | None = None
        self._metadata_head_dim: int | None = None

    def _ensure_metadata(self, device: torch.device) -> bool:
        if self._metadata_cache_size <= 0:
            return False
        value_ptrs = self._metadata_k_value_ptrs
        needs_alloc = (
            value_ptrs is None or value_ptrs.device != device or value_ptrs.numel() != self._metadata_cache_size
        )
        if not needs_alloc:
            return True

        self._metadata_k_value_ptrs = torch.zeros(
            (self._metadata_cache_size,), device=device, dtype=torch.int64
        )  # [N_slots]
        self._metadata_v_value_ptrs = torch.zeros(
            (self._metadata_cache_size,), device=device, dtype=torch.int64
        )  # [N_slots]
        self._metadata_k_scales = torch.zeros(
            (self._metadata_cache_size,), device=device, dtype=torch.float32
        )  # [N_slots]
        self._metadata_v_scales = torch.zeros(
            (self._metadata_cache_size,), device=device, dtype=torch.float32
        )  # [N_slots]
        return True

    def _invalidate_metadata_slot(self, cache_idx: int) -> None:
        if (
            self._metadata_k_value_ptrs is None
            or self._metadata_v_value_ptrs is None
            or self._metadata_k_scales is None
            or self._metadata_v_scales is None
        ):
            return
        if cache_idx < 0 or cache_idx >= self._metadata_k_value_ptrs.numel():
            return
        self._metadata_k_value_ptrs[cache_idx] = 0
        self._metadata_v_value_ptrs[cache_idx] = 0
        self._metadata_k_scales[cache_idx] = 0
        self._metadata_v_scales[cache_idx] = 0

    def _store_metadata_pair(self, cache_idx: int, k_entry: object, v_entry: object) -> None:
        parsed_k = parse_fp8_entry(k_entry)
        parsed_v = parse_fp8_entry(v_entry)
        if parsed_k is None or parsed_v is None:
            self._invalidate_metadata_slot(cache_idx)
            return
        k_fp8, k_scale = parsed_k
        v_fp8, v_scale = parsed_v
        if k_fp8.shape != v_fp8.shape or k_fp8.device != v_fp8.device:
            self._invalidate_metadata_slot(cache_idx)
            return
        tokens_per_entry = int(k_fp8.shape[1])
        if self._metadata_tokens_per_entry is None:
            self._metadata_tokens_per_entry = tokens_per_entry
        elif tokens_per_entry != self._metadata_tokens_per_entry:
            # Variable-S entries cannot use this fixed-S fast path; clear the stale slot first.
            self._invalidate_metadata_slot(cache_idx)
            raise ValueError(
                'FP8 kernel_impl="triton" requires fixed tokens_per_entry; '
                f"expected {self._metadata_tokens_per_entry}, got {tokens_per_entry}"
            )
        if not self._ensure_metadata(k_fp8.device):
            self._invalidate_metadata_slot(cache_idx)
            return

        # Drop any stale pointer before publishing the replacement metadata.
        self._invalidate_metadata_slot(cache_idx)
        assert self._metadata_k_value_ptrs is not None
        assert self._metadata_v_value_ptrs is not None
        assert self._metadata_k_scales is not None
        assert self._metadata_v_scales is not None
        self._metadata_k_value_ptrs[cache_idx] = k_fp8.data_ptr()
        self._metadata_v_value_ptrs[cache_idx] = v_fp8.data_ptr()
        self._metadata_k_scales[cache_idx] = k_scale
        self._metadata_v_scales[cache_idx] = v_scale
        self._metadata_batch_size = k_fp8.shape[0]
        self._metadata_num_heads = k_fp8.shape[2]
        self._metadata_head_dim = k_fp8.shape[3]

    def _decode_pair_triton(
        self,
        dtype: torch.dtype | None,
        slots: list[int] | None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Return None unless the fixed-S Triton metadata path is ready."""
        if dtype not in (None, torch.bfloat16) or not slots:
            return None
        if triton is None or _decode_fp8_kv_many_kernel is None:
            return None

        ordered_slots_py = [int(slot) for slot in slots]
        if any(slot < 0 or slot >= self._metadata_cache_size for slot in ordered_slots_py):
            return None

        k_value_ptrs = self._metadata_k_value_ptrs
        v_value_ptrs = self._metadata_v_value_ptrs
        k_scales = self._metadata_k_scales
        v_scales = self._metadata_v_scales
        tokens_per_entry = self._metadata_tokens_per_entry
        batch_size = self._metadata_batch_size
        num_heads = self._metadata_num_heads
        head_dim = self._metadata_head_dim
        if (
            k_value_ptrs is None
            or v_value_ptrs is None
            or k_scales is None
            or v_scales is None
            or tokens_per_entry is None
            or batch_size is None
            or num_heads is None
            or head_dim is None
        ):
            return None
        device = k_value_ptrs.device
        ordered_slots = torch.tensor(ordered_slots_py, device=device, dtype=torch.int64)  # [E]
        selected_k_value_ptrs = k_value_ptrs[ordered_slots]  # [E]
        selected_v_value_ptrs = v_value_ptrs[ordered_slots]  # [E]
        missing_k_slot = bool((selected_k_value_ptrs == 0).any().item())
        missing_v_slot = bool((selected_v_value_ptrs == 0).any().item())
        if missing_k_slot or missing_v_slot:
            return None
        total_seq = len(ordered_slots_py) * tokens_per_entry

        return decode_fp8_kv_many_triton(
            k_value_ptrs=k_value_ptrs,
            v_value_ptrs=v_value_ptrs,
            k_scales=k_scales,
            v_scales=v_scales,
            ordered_slots=ordered_slots,
            batch_size=batch_size,
            tokens_per_entry=tokens_per_entry,
            total_seq=total_seq,
            num_heads=num_heads,
            head_dim=head_dim,
        )

    def encode(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode through the torch reference path; kernel_impl only affects batch decode."""
        return self._encode_fp8_torch(tensor)  # [B,S,H,D], [1]

    def decode(self, entry: object) -> torch.Tensor:
        return self._decode_fp8(entry)  # [B,S,H,D]

    def decode_many(
        self,
        k_entries: list[object],
        v_entries: list[object],
        slots: list[int] | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode paired FP8 entries and concatenate them along the sequence dimension."""
        if self.kernel_impl == "torch":
            return super().decode_many(k_entries, v_entries, slots=slots, dtype=dtype)  # [B,S_total,H,D] each

        decoded = self._decode_pair_triton(dtype=dtype, slots=slots)
        if decoded is None:
            raise RuntimeError(
                'FP8 kernel_impl="triton" requires the multi-entry Triton decode path, '
                'but its requirements were not met. Set kernel_impl="torch" '
                "to use the reference path."
            )
        return decoded

    def update_cached_kv_metadata(self, cache_idx: int, k_entry: object, v_entry: object) -> None:
        if self.kernel_impl != "triton":
            return
        cache_idx = int(cache_idx)
        self._store_metadata_pair(cache_idx, k_entry, v_entry)

    def reset_kv_cache_state(self, cache_size: int) -> None:
        self._metadata_cache_size = int(cache_size) if self.kernel_impl == "triton" else 0
        self._metadata_k_value_ptrs = None
        self._metadata_v_value_ptrs = None
        self._metadata_k_scales = None
        self._metadata_v_scales = None
        self._metadata_tokens_per_entry = None
        self._metadata_batch_size = None
        self._metadata_num_heads = None
        self._metadata_head_dim = None
