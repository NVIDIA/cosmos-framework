# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Tests for the key/value storage backends.

These tests live apart from `kv_cache_test.py` to keep their dependency
surface minimal: `kv_cache_test.py` pulls in `cosmos_framework.model.attention`,
`transformers`-backed tokenizers, and the rest of the autoregressive stack,
none of which the backend numerics need. Isolating them lets the round-trip
and byte-footprint checks run in a torch-only environment; CI still collects
them normally.

The tests describe observable behavior through the `encode` / `decode`
interface. They deliberately avoid asserting on internal scale dtype or which
quantization helper is used, so a future swap of the FP8 implementation
(e.g. to a wrapper tensor type) need not break them.
"""

import pytest
import torch

from cosmos_framework.model.generator.utils import kv_storage_backend as kv_storage_backend_module
from cosmos_framework.model.generator.utils.kv_storage_backend import (
    BF16StorageBackend,
    FP8StorageBackend,
)


def _tensor_bytes(obj: object) -> int:
    """Total bytes of every tensor inside `obj`, whatever the structure.

    Reads like a definition:
      - a tensor's bytes  = element_size * numel
      - a list/tuple's    = sum of its items' bytes
      - anything else     = 0

    This counts a plain BF16 tensor (BF16 backend) and a (fp8, scale) tuple
    (FP8 backend) the same way, so it measures the stored format's real
    footprint — scale included — and survives a future swap to a wrapper type.
    """
    if isinstance(obj, torch.Tensor):
        return obj.element_size() * obj.numel()
    if isinstance(obj, (tuple, list)):
        return sum(_tensor_bytes(item) for item in obj)
    return 0


@pytest.mark.L0
@pytest.mark.CPU
def test_fp8_backend_round_trip() -> None:
    """Round-trip a BF16 tensor and verify the shape/dtype/closeness contract.

    Behavior under test:
      - `decode(encode(t))` preserves shape and returns BF16.
      - Round-trip relative error stays in typical FP8 territory (< 5%).
      - The BF16 backend round-trips bit-exactly.
    """
    t = torch.randn(1, 100, 8, 128, dtype=torch.bfloat16)

    fp8_backend = FP8StorageBackend()
    t_back = fp8_backend.decode(fp8_backend.encode(t))

    assert t_back.shape == t.shape
    assert t_back.dtype == torch.bfloat16
    rel_err = ((t_back - t).abs().max() / t.abs().max()).item()
    assert rel_err < 0.05, f"FP8 round-trip rel_err too high: {rel_err}"

    bf16_backend = BF16StorageBackend()
    t_bf16_back = bf16_backend.decode(bf16_backend.encode(t))
    assert torch.equal(t_bf16_back, t)


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    "backend",
    [
        BF16StorageBackend(),
        FP8StorageBackend(kv_cache_dtype="fp8", kernel_impl="torch"),
    ],
    ids=["bf16", "fp8"],
)
def test_decode_many_non_fused_paths_return_kv_histories(backend: BF16StorageBackend | FP8StorageBackend) -> None:
    """BF16 and FP8 torch decode_many return paired K/V histories."""
    frames = [
        (
            torch.randn(1, 3, 2, 16, dtype=torch.bfloat16),  # [B,S0,H,D]
            torch.randn(1, 3, 2, 16, dtype=torch.bfloat16),  # [B,S0,H,D]
        ),
        (
            torch.randn(1, 5, 2, 16, dtype=torch.bfloat16),  # [B,S1,H,D]
            torch.randn(1, 5, 2, 16, dtype=torch.bfloat16),  # [B,S1,H,D]
        ),
    ]
    k_entries = [backend.encode(k) for k, _ in frames]
    v_entries = [backend.encode(v) for _, v in frames]

    expected_k = torch.cat([backend.decode(entry) for entry in k_entries], dim=1)  # [B,S_total,H,D]
    expected_v = torch.cat([backend.decode(entry) for entry in v_entries], dim=1)  # [B,S_total,H,D]
    decoded_k, decoded_v = backend.decode_many(k_entries, v_entries, slots=[1, 0])  # [B,S_total,H,D] each

    torch.testing.assert_close(decoded_k, expected_k)
    torch.testing.assert_close(decoded_v, expected_v)
    with pytest.raises(AssertionError, match="same number"):
        backend.decode_many(k_entries, v_entries[:1])


@pytest.mark.L0
@pytest.mark.CPU
def test_fp8_backend_default_triton_decode_many_fails_on_cpu() -> None:
    """Default FP8 batch decode requires the Triton path."""
    backend = FP8StorageBackend(kv_cache_dtype="fp8")
    backend.reset_kv_cache_state(3)
    frames = [
        (
            torch.randn(1, 4, 2, 16, dtype=torch.bfloat16),  # [B,S,H,D]
            torch.randn(1, 4, 2, 16, dtype=torch.bfloat16),  # [B,S,H,D]
        )
        for _ in range(2)
    ]
    k_entries = [backend.encode(k) for k, _ in frames]
    v_entries = [backend.encode(v) for _, v in frames]

    backend.reset_kv_cache_state(3)
    for cache_idx, (k_entry, v_entry) in enumerate(zip(k_entries, v_entries)):
        backend.update_cached_kv_metadata(cache_idx, k_entry, v_entry)

    with pytest.raises(RuntimeError, match='kernel_impl="torch"'):
        backend.decode_many(k_entries, v_entries, slots=[0, 1])


@pytest.mark.L0
@pytest.mark.CPU
def test_fp8_backend_triton_metadata_invalidates_reused_slot_on_variable_tokens_per_entry() -> None:
    """A fixed-S metadata failure clears the reused slot before raising."""
    backend = FP8StorageBackend(kv_cache_dtype="fp8", kernel_impl="triton")
    backend.reset_kv_cache_state(3)

    k0 = torch.randn(1, 3, 2, 16, dtype=torch.bfloat16)  # [B,S0,H,D]
    v0 = torch.randn(1, 3, 2, 16, dtype=torch.bfloat16)  # [B,S0,H,D]
    k0_entry = backend.encode(k0)  # [B,S0,H,D], [1]
    v0_entry = backend.encode(v0)  # [B,S0,H,D], [1]
    backend.update_cached_kv_metadata(0, k0_entry, v0_entry)

    assert backend._metadata_k_value_ptrs is not None
    assert backend._metadata_v_value_ptrs is not None
    assert backend._metadata_k_scales is not None
    assert backend._metadata_v_scales is not None
    assert int(backend._metadata_k_value_ptrs[0].item()) != 0
    assert int(backend._metadata_v_value_ptrs[0].item()) != 0

    k1 = torch.randn(1, 5, 2, 16, dtype=torch.bfloat16)  # [B,S1,H,D]
    v1 = torch.randn(1, 5, 2, 16, dtype=torch.bfloat16)  # [B,S1,H,D]
    k1_entry = backend.encode(k1)  # [B,S1,H,D], [1]
    v1_entry = backend.encode(v1)  # [B,S1,H,D], [1]
    with pytest.raises(ValueError, match="fixed tokens_per_entry"):
        backend.update_cached_kv_metadata(0, k1_entry, v1_entry)

    assert int(backend._metadata_k_value_ptrs[0].item()) == 0
    assert int(backend._metadata_v_value_ptrs[0].item()) == 0
    assert float(backend._metadata_k_scales[0].item()) == 0.0
    assert float(backend._metadata_v_scales[0].item()) == 0.0


@pytest.mark.L0
@pytest.mark.GPU
def test_fp8_backend_triton_decode_many_uses_fast_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Triton kernel setting uses torch encode and fused decode_many."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the Triton FP8 backend")
    pytest.importorskip("triton")

    backend = FP8StorageBackend(kv_cache_dtype="fp8", kernel_impl="triton")
    backend.reset_kv_cache_state(4)
    frames = [
        (
            torch.randn(1, 4, 2, 16, device="cuda", dtype=torch.bfloat16),  # [B,S,H,D]
            torch.randn(1, 4, 2, 16, device="cuda", dtype=torch.bfloat16),  # [B,S,H,D]
        )
        for _ in range(3)
    ]

    original_torch_encode = backend._encode_fp8_torch
    torch_encode_calls = 0

    def track_torch_encode(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # tensor: [B,S,H,D]
        nonlocal torch_encode_calls
        torch_encode_calls += 1
        encoded = original_torch_encode(tensor)  # [B,S,H,D], [1]
        return encoded

    monkeypatch.setattr(backend, "_encode_fp8_torch", track_torch_encode)
    k_entries: list[object] = []
    v_entries: list[object] = []
    for cache_idx, (k, v) in enumerate(frames):
        k_entry = backend.encode(k)
        v_entry = backend.encode(v)
        k_entries.append(k_entry)
        v_entries.append(v_entry)
        backend.update_cached_kv_metadata(cache_idx, k_entry, v_entry)

    assert torch_encode_calls == 2 * len(frames)
    expected_k = torch.cat([backend.decode(entry) for entry in k_entries], dim=1)  # [B,S_total,H,D]
    expected_v = torch.cat([backend.decode(entry) for entry in v_entries], dim=1)  # [B,S_total,H,D]

    def fail_on_fallback(_entry: object) -> torch.Tensor:
        raise AssertionError("triton unexpectedly fell back to per-entry decode")

    monkeypatch.setattr(backend, "decode", fail_on_fallback)
    actual_k, actual_v = backend.decode_many(k_entries, v_entries, slots=[0, 1, 2])  # [B,S_total,H,D] each

    torch.testing.assert_close(actual_k, expected_k)
    torch.testing.assert_close(actual_v, expected_v)


@pytest.mark.L0
@pytest.mark.GPU
def test_decode_fp8_kv_many_triton_preserves_launch_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """Kernel launch failures keep the original exception as the cause."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the Triton FP8 backend")
    pytest.importorskip("triton")

    class SentinelLaunchError(RuntimeError):
        pass

    def fail_launch(*_args: object) -> None:
        raise SentinelLaunchError("sentinel fp8 launch")

    class FailingKernel:
        def __getitem__(self, _grid: tuple[int, int]) -> object:
            return fail_launch

    monkeypatch.setattr(kv_storage_backend_module, "_decode_fp8_kv_many_kernel", FailingKernel())
    k_value_ptrs = torch.ones(1, device="cuda", dtype=torch.int64)  # [N_slots]
    v_value_ptrs = torch.ones(1, device="cuda", dtype=torch.int64)  # [N_slots]
    k_scales = torch.ones(1, device="cuda", dtype=torch.float32)  # [N_slots]
    v_scales = torch.ones(1, device="cuda", dtype=torch.float32)  # [N_slots]
    ordered_slots = torch.tensor([0], device="cuda", dtype=torch.int64)  # [E]

    with pytest.raises(RuntimeError, match="FP8 Triton decode kernel launch failed") as exc_info:
        kv_storage_backend_module.decode_fp8_kv_many_triton(
            k_value_ptrs=k_value_ptrs,
            v_value_ptrs=v_value_ptrs,
            k_scales=k_scales,
            v_scales=v_scales,
            ordered_slots=ordered_slots,
            batch_size=1,
            tokens_per_entry=4,
            total_seq=4,
            num_heads=2,
            head_dim=16,
        )

    assert isinstance(exc_info.value.__cause__, SentinelLaunchError)
    assert "sentinel fp8 launch" in str(exc_info.value.__cause__)


@pytest.mark.L0
@pytest.mark.CPU
def test_fp8_backend_round_trip_and_storage_shape() -> None:
    """FP8 stores one scale per entry and round-trips to BF16."""
    t = torch.randn(1, 100, 8, 128, dtype=torch.bfloat16)  # [B,S,H,D]

    backend = FP8StorageBackend(kv_cache_dtype="fp8")
    entry = backend.encode(t)
    t_back = backend.decode(entry)  # [B,S,H,D]

    assert isinstance(entry, tuple)
    fp8, scale = entry
    assert fp8.shape == t.shape
    assert fp8.dtype == torch.float8_e4m3fn
    assert scale.shape == (1,)
    assert t_back.shape == t.shape
    assert t_back.dtype == torch.bfloat16
    rel_err = ((t_back - t).abs().max() / t.abs().max()).item()
    assert rel_err < 0.08, f"FP8 round-trip rel_err too high: {rel_err}"

    fp8_bytes = _tensor_bytes(entry)
    bf16_bytes = _tensor_bytes(BF16StorageBackend().encode(t))
    assert 0.45 < fp8_bytes / bf16_bytes < 0.55


@pytest.mark.L0
@pytest.mark.CPU
def test_fp8_backend_decode_many_matches_decode_cat_with_explicit_dtype() -> None:
    """Batch decode preserves the same concat semantics as decode+cat."""
    backend = FP8StorageBackend(kv_cache_dtype="fp8", kernel_impl="torch")
    frames = [
        (
            torch.randn(1, 3, 2, 16, dtype=torch.bfloat16),  # [B,S0,H,D]
            torch.randn(1, 3, 2, 16, dtype=torch.bfloat16),  # [B,S0,H,D]
        ),
        (
            torch.randn(1, 5, 2, 16, dtype=torch.bfloat16),  # [B,S1,H,D]
            torch.randn(1, 5, 2, 16, dtype=torch.bfloat16),  # [B,S1,H,D]
        ),
        (
            torch.randn(1, 2, 2, 16, dtype=torch.bfloat16),  # [B,S2,H,D]
            torch.randn(1, 2, 2, 16, dtype=torch.bfloat16),  # [B,S2,H,D]
        ),
    ]
    k_entries = [backend.encode(k) for k, _ in frames]
    v_entries = [backend.encode(v) for _, v in frames]

    expected_k = torch.cat([backend.decode(entry) for entry in k_entries], dim=1)  # [B,S_total,H,D]
    expected_v = torch.cat([backend.decode(entry) for entry in v_entries], dim=1)  # [B,S_total,H,D]
    decoded_k, decoded_v = backend.decode_many(k_entries, v_entries, dtype=torch.bfloat16)  # [B,S_total,H,D] each

    assert decoded_k.dtype == torch.bfloat16
    assert decoded_v.dtype == torch.bfloat16
    torch.testing.assert_close(decoded_k, expected_k)
    torch.testing.assert_close(decoded_v, expected_v)


@pytest.mark.L0
@pytest.mark.CPU
def test_fp8_backend_outlier_sets_single_tensor_scale() -> None:
    """A large outlier controls the single scale stored with an FP8 entry."""
    torch.manual_seed(0)
    backend = FP8StorageBackend()
    k = torch.randn(1, 101, 8, 128, dtype=torch.bfloat16) * 0.5  # [B,S,H,D]
    k[0, 50, :, :] *= 10.0

    fp8, scale = backend.encode(k)

    assert fp8.shape == k.shape
    assert scale.shape == (1,)
    expected_scale = k.abs().amax().reshape(1) / torch.finfo(torch.float8_e4m3fn).max  # [1]
    torch.testing.assert_close(scale, expected_scale)


@pytest.mark.L0
@pytest.mark.CPU
def test_fp8_backend_stores_about_half_the_bytes() -> None:
    """FP8 stored footprint is ~50% of the BF16 stored footprint.

    Values drop from 2 bytes (BF16) to 1 byte (FP8 e4m3); each cache entry also
    stores one BF16 scale. The ratio stays very close to 0.5 for normal KV
    tensors.

    Measured from the stored format directly (element_size * numel) rather
    than allocator stats, so it runs on CPU free of pool/alignment noise.
    """
    t = torch.randn(1, 10000, 8, 128, dtype=torch.bfloat16)

    fp8_bytes = _tensor_bytes(FP8StorageBackend().encode(t))
    bf16_bytes = _tensor_bytes(BF16StorageBackend().encode(t))

    ratio = fp8_bytes / bf16_bytes
    assert 0.45 < ratio < 0.60, (
        f"FP8/BF16 stored-bytes ratio {ratio:.3f}, expected ~0.5 (fp8={fp8_bytes}, bf16={bf16_bytes})"
    )


@pytest.mark.L0
@pytest.mark.CPU
def test_bf16_backend_entry_is_plain_tensor() -> None:
    """The BF16 backend keeps its stored entry a plain tensor.

    This is a load-bearing contract, not an incidental detail. Code that
    stores entries directly into preallocated, fixed-shape buffers (for
    example a cuda-graph static capture path) relies on a BF16 entry being a
    bare ``torch.Tensor`` rather than a tuple wrapper. If this regressed to a
    tuple, such a path would break, so it is asserted explicitly.
    """
    t = torch.randn(1, 100, 8, 128, dtype=torch.bfloat16)
    entry = BF16StorageBackend().encode(t)
    assert isinstance(entry, torch.Tensor)
