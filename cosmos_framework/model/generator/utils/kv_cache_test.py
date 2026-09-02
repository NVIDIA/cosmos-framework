# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Tests for KV cache and non-CP AR inference logic."""

import pytest
import torch

from cosmos_framework.model.attention import attention
from cosmos_framework.model.generator.mot.attention import SplitInfo
from cosmos_framework.data.generator.sequence_packing.runtime import SequencePack, get_gen_seq
from cosmos_framework.model.generator.mot.causal_attention import (
    attention_AR_gen_only,
    three_way_attention_with_kv_cache,
)
from cosmos_framework.model.generator.utils.kv_cache import (
    ARMemoryState,
    ARMemoryValue,
    DualKVCache,
    FlexARMemoryState,
    FlexARMemoryValue,
    GenKVCache,
    KVCache,
    KVCacheTrainMemoryState,
    TeacherForcingMemoryState,
    TFNoisyMemoryValue,
    TFReplayCleanMemoryValue,
    UndKVCache,
)
from cosmos_framework.model.generator.utils.kv_storage_backend import (
    BF16StorageBackend,
    FP8StorageBackend,
)


@pytest.mark.L0
def test_flex_ar_memory_state_captures_selected_tokens_at_requested_offset() -> None:
    """Prefill and refresh writes share one fixed-capacity cache without exposing padding."""
    cache: list[tuple[torch.Tensor, torch.Tensor] | None] = [None]
    prefill = FlexARMemoryState(
        num_layers=1,
        memory_seq_len=8,
        cache=cache,
        write_indexes=torch.tensor([0, 2]),  # [2]
        write_offset=0,
    )
    gen_k = torch.arange(12, dtype=torch.float32).reshape(1, 3, 2, 2)  # [1,3,2,2]
    gen_v = gen_k + 100  # [1,3,2,2]
    und = torch.empty(1, 0, 2, 2)  # [1,0,2,2]
    prefill.write_for_layer(0, (gen_k, gen_v, und, und))

    refresh = FlexARMemoryState(
        num_layers=1,
        memory_seq_len=8,
        cache=cache,
        write_indexes=torch.tensor([1]),  # [1]
        write_offset=2,
    )
    refresh.write_for_layer(0, (gen_k, gen_v, und, und))
    value = refresh.read_for_layer(0)
    assert isinstance(value, FlexARMemoryValue)
    assert value.supports_context_parallel_attention
    assert value.cached_gen_k is not None and value.cached_gen_v is not None
    assert torch.equal(value.cached_gen_k[:, :3], gen_k[:, [0, 2, 1]])
    assert torch.equal(value.cached_gen_v[:, :3], gen_v[:, [0, 2, 1]])
    assert torch.count_nonzero(value.cached_gen_k[:, 3:]) == 0


@pytest.mark.L0
def test_kv_cache_validation():
    """KVCache raises ValueError when cache_size < 2."""
    with pytest.raises(ValueError, match="cache_size must be >= 2"):
        KVCache(cache_size=1)
    with pytest.raises(ValueError, match="cache_size must be >= 2"):
        KVCache(cache_size=0)
    # cache_size=2 is the minimum valid value; no error
    cache = KVCache(cache_size=2)
    assert cache.cache_size == 2


@pytest.mark.L0
def test_kv_cache_attention_sink_validation():
    """Attention sinks require a finite cache and at least one rolling slot."""
    with pytest.raises(ValueError, match="attention_sink_size must be 0 when cache_size is None"):
        KVCache(cache_size=None, attention_sink_size=1)
    with pytest.raises(ValueError, match="attention_sink_size must be less than cache_size"):
        KVCache(cache_size=4, attention_sink_size=4)
    with pytest.raises(ValueError, match="attention_sink_size must be less than cache_size"):
        KVCache(cache_size=4, attention_sink_size=5)
    with pytest.raises(ValueError, match="attention_sink_size must be >= 0"):
        KVCache(cache_size=4, attention_sink_size=-1)
    with pytest.raises(ValueError, match="attention_sink_size must be 0 when cache_size is None"):
        DualKVCache(gen_cache_size=None, attention_sink_size=1)

    cache = DualKVCache(gen_cache_size=4, attention_sink_size=3)
    assert cache.gen_cache.cache_size == 4
    assert cache.gen_cache.attention_sink_size == 3


@pytest.mark.L0
def test_und_kv_cache_lifecycle():
    """UndKVCache: get() before store() raises; store/get returns correct tensors; reset clears."""
    cache = UndKVCache()
    assert not cache.is_initialized

    # get() before store() raises AssertionError
    with pytest.raises(AssertionError):
        cache.get()

    # store and retrieve
    k = torch.randn(1, 6, 4, 16)
    v = torch.randn(1, 6, 4, 16)
    cache.store(k, v)
    assert cache.is_initialized

    k_out, v_out = cache.get()
    assert torch.equal(k_out, k)
    assert torch.equal(v_out, v)

    # reset clears state; subsequent get() raises again
    cache.reset()
    assert not cache.is_initialized
    with pytest.raises(AssertionError):
        cache.get()


@pytest.mark.L0
def test_dual_kv_cache_reset():
    """DualKVCache.reset() clears both und and gen caches independently."""
    cache = DualKVCache(gen_cache_size=4)
    B, S, H, D = 1, 4, 2, 8
    k = torch.randn(B, S, H, D)
    v = torch.randn(B, S, H, D)

    cache.und_cache.store(k, v)
    cache.gen_cache.store_kv(k, v, frame_idx=0)
    assert cache.und_cache.is_initialized
    assert cache.gen_cache.k_cache[0] is not None

    cache.reset()

    assert not cache.und_cache.is_initialized
    assert all(slot is None for slot in cache.gen_cache.k_cache)
    # fetch_kv(0) returns (None, None) since frame 0 has no prior history
    k_none, v_none = cache.gen_cache.fetch_kv(frame_idx=0)
    assert k_none is None and v_none is None


@pytest.mark.L0
def test_kv_cache_rolling_window_many_frames():
    """Rolling window: fetch_kv returns only the cache_size-1 most-recent frames.

    With cache_size=3 and frame_idx=fi, only frames [fi-2, fi-1] are returned.
    Frames further in the past have been evicted (start_idx = max(0, fi - cache_size + 1)).
    """
    B, H, D = 1, 2, 4

    # --- scenario A: fetch after 6 stores (rolling well past capacity) ---
    cache = KVCache(cache_size=3)
    for i in range(6):
        # Each frame stores 2 tokens with value == frame index
        k = torch.ones(B, 2, H, D) * i
        cache.store_kv(k, k, frame_idx=i)

    # fetch_kv(6): start_idx = max(0, 6-3+1) = 4 → history = [frame 4, frame 5]
    k_hist, _ = cache.fetch_kv(frame_idx=6)
    assert k_hist is not None and k_hist.shape == (B, 4, H, D)
    assert k_hist[:, :2, :, :].allclose(torch.ones(B, 2, H, D) * 4), "expected frame 4 data"
    assert k_hist[:, 2:, :, :].allclose(torch.ones(B, 2, H, D) * 5), "expected frame 5 data"

    # fetch_kv(5): start_idx = max(0, 5-3+1) = 3 → history = [frame 3, frame 4]
    k_hist5, _ = cache.fetch_kv(frame_idx=5)
    assert k_hist5 is not None and k_hist5.shape == (B, 4, H, D)
    assert k_hist5[:, :2, :, :].allclose(torch.ones(B, 2, H, D) * 3)
    assert k_hist5[:, 2:, :, :].allclose(torch.ones(B, 2, H, D) * 4)

    # --- scenario B: fetch near the start (before rolling kicks in fully) ---
    cache2 = KVCache(cache_size=3)
    for i in range(3):
        k = torch.ones(B, 2, H, D) * i
        cache2.store_kv(k, k, frame_idx=i)

    # fetch_kv(2): start_idx = max(0, 2-3+1) = 0 → history = [frame 0, frame 1]
    k_hist2, _ = cache2.fetch_kv(frame_idx=2)
    assert k_hist2 is not None and k_hist2.shape == (B, 4, H, D)
    assert k_hist2[:, :2, :, :].allclose(torch.ones(B, 2, H, D) * 0)
    assert k_hist2[:, 2:, :, :].allclose(torch.ones(B, 2, H, D) * 1)

    # fetch_kv(3): start_idx = max(0, 3-3+1) = 1 → history = [frame 1, frame 2] (frame 0 evicted)
    k_hist3, _ = cache2.fetch_kv(frame_idx=3)
    assert k_hist3 is not None and k_hist3.shape == (B, 4, H, D)
    assert k_hist3[:, :2, :, :].allclose(torch.ones(B, 2, H, D) * 1)
    assert k_hist3[:, 2:, :, :].allclose(torch.ones(B, 2, H, D) * 2)


@pytest.mark.L0
def test_kv_cache_attention_sink_keeps_prefix_and_rolls_tail():
    """Attention sink history is pinned prefix followed by recent rolling tail."""
    B, S, H, D = 1, 2, 2, 4
    cache = KVCache(cache_size=6, attention_sink_size=2)

    for frame_idx in range(6):
        k = torch.full((B, S, H, D), float(frame_idx))  # [B,S,H,D]
        cache.store_kv(k, k, frame_idx=frame_idx)

    # Frame 6 is the first fetch where the rolling history overflows: frame 2
    # is evicted while sink frames 0 and 1 remain pinned.
    first_eviction_k, first_eviction_v = cache.fetch_kv(frame_idx=6)
    assert first_eviction_k is not None and first_eviction_v is not None
    assert first_eviction_k.shape == (B, 5 * S, H, D)
    assert first_eviction_v.shape == (B, 5 * S, H, D)

    first_eviction_frames = [0, 1, 3, 4, 5]
    for offset, expected_frame_idx in enumerate(first_eviction_frames):
        start = offset * S
        end = start + S
        expected = torch.full((B, S, H, D), float(expected_frame_idx))  # [B,S,H,D]
        torch.testing.assert_close(first_eviction_k[:, start:end], expected)
        torch.testing.assert_close(first_eviction_v[:, start:end], expected)

    for frame_idx in range(6, 10):
        k = torch.full((B, S, H, D), float(frame_idx))  # [B,S,H,D]
        cache.store_kv(k, k, frame_idx=frame_idx)

    k_hist, v_hist = cache.fetch_kv(frame_idx=10)
    assert k_hist is not None and v_hist is not None
    assert k_hist.shape == (B, 5 * S, H, D)
    assert v_hist.shape == (B, 5 * S, H, D)

    expected_frames = [0, 1, 7, 8, 9]
    for offset, frame_idx in enumerate(expected_frames):
        start = offset * S
        end = start + S
        expected = torch.full((B, S, H, D), float(frame_idx))  # [B,S,H,D]
        torch.testing.assert_close(k_hist[:, start:end], expected)
        torch.testing.assert_close(v_hist[:, start:end], expected)


@pytest.mark.L0
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_non_cp_ar_inference_rolling_window():
    """AR inference with rolling gen cache: old frames are evicted; output matches trimmed baseline.

    With gen_cache_size=3, at frame fi the attention context is:
        [und | gen_max(0, fi-2) | ... | gen_{fi-1} | gen_fi]
    i.e. at most cache_size-1 = 2 past gen frames are included.
    Verifies each frame's output matches full attention with the same truncated context.
    """
    device = torch.device("cuda", 0)
    num_heads, head_dim = 4, 32
    S_und, S_gen = 6, 4
    gen_cache_size = 3  # keeps at most 2 past frames (cache_size - 1)
    num_frames = 6
    scale = head_dim**-0.5

    torch.manual_seed(13)
    k_und = torch.randn(S_und, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    v_und = torch.randn(S_und, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    q_frames = [torch.randn(S_gen, num_heads, head_dim, device=device, dtype=torch.bfloat16) for _ in range(num_frames)]
    k_frames = [torch.randn(S_gen, num_heads, head_dim, device=device, dtype=torch.bfloat16) for _ in range(num_frames)]
    v_frames = [torch.randn(S_gen, num_heads, head_dim, device=device, dtype=torch.bfloat16) for _ in range(num_frames)]

    dual_cache = DualKVCache(gen_cache_size=gen_cache_size)
    dual_cache.und_cache.store(k_und.unsqueeze(0), v_und.unsqueeze(0))
    dual_cache.gen_cache.store_kv(k_frames[0].unsqueeze(0), v_frames[0].unsqueeze(0), frame_idx=0)

    for fi in range(1, num_frames):
        # --- cache path ---
        k_gen_hist, v_gen_hist = dual_cache.gen_cache.fetch_kv(fi)  # rolling window
        dual_cache.gen_cache.store_kv(k_frames[fi].unsqueeze(0), v_frames[fi].unsqueeze(0), frame_idx=fi)
        k_und_cached, v_und_cached = dual_cache.und_cache.get()
        k_curr = k_frames[fi].unsqueeze(0)
        v_curr = v_frames[fi].unsqueeze(0)
        if k_gen_hist is not None:
            k_full = torch.cat([k_und_cached, k_gen_hist, k_curr], dim=1)
            v_full = torch.cat([v_und_cached, v_gen_hist, v_curr], dim=1)
        else:
            k_full = torch.cat([k_und_cached, k_curr], dim=1)
            v_full = torch.cat([v_und_cached, v_curr], dim=1)

        attn_result = attention(
            query=q_frames[fi].unsqueeze(0),  # [1,S_gen,H,D]
            key=k_full,
            value=v_full,
            is_causal=False,
            scale=scale,
            return_lse=False,
        )
        assert isinstance(attn_result, torch.Tensor)
        out_cache: torch.Tensor = attn_result  # [1,S_gen,H,D]
        out_cache = out_cache.squeeze(0)  # [S_gen,H,D]

        # --- baseline: manually build trimmed context matching the rolling window ---
        start_idx = max(0, fi - gen_cache_size + 1)
        k_hist_parts = [k_frames[j] for j in range(start_idx, fi)]
        v_hist_parts = [v_frames[j] for j in range(start_idx, fi)]
        k_und_b = k_und.unsqueeze(0)  # [1,S_und,H,D]
        v_und_b = v_und.unsqueeze(0)
        if k_hist_parts:
            k_hist_b = torch.cat([t.unsqueeze(0) for t in k_hist_parts], dim=1)  # [1,S_hist,H,D]
            v_hist_b = torch.cat([t.unsqueeze(0) for t in v_hist_parts], dim=1)
            k_full_b = torch.cat([k_und_b, k_hist_b, k_frames[fi].unsqueeze(0)], dim=1)
            v_full_b = torch.cat([v_und_b, v_hist_b, v_frames[fi].unsqueeze(0)], dim=1)
        else:
            k_full_b = torch.cat([k_und_b, k_frames[fi].unsqueeze(0)], dim=1)
            v_full_b = torch.cat([v_und_b, v_frames[fi].unsqueeze(0)], dim=1)

        baseline_result = attention(
            query=q_frames[fi].unsqueeze(0),
            key=k_full_b,
            value=v_full_b,
            is_causal=False,
            scale=scale,
            return_lse=False,
        )
        assert isinstance(baseline_result, torch.Tensor)
        out_baseline: torch.Tensor = baseline_result  # [1,S_gen,H,D]
        out_baseline = out_baseline.squeeze(0)  # [S_gen,H,D]

        torch.testing.assert_close(
            out_cache, out_baseline, rtol=1e-2, atol=1e-2, msg=f"Frame {fi}: rolling window mismatch"
        )


@pytest.mark.L0
def test_kv_cache_store_fetch():
    """fetch_kv() returns correct sequence-concatenated history after multi-frame stores."""
    cache = KVCache(cache_size=4)
    B, H, D = 1, 4, 8

    k0 = torch.randn(B, 3, H, D)
    v0 = torch.randn(B, 3, H, D)
    k1 = torch.randn(B, 5, H, D)
    v1 = torch.randn(B, 5, H, D)

    cache.store_kv(k0, v0, frame_idx=0)
    cache.store_kv(k1, v1, frame_idx=1)

    k_hist, v_hist = cache.fetch_kv(frame_idx=2)
    assert k_hist is not None and v_hist is not None
    assert k_hist.shape == (B, 8, H, D)  # 3+5 seq tokens
    assert v_hist.shape == (B, 8, H, D)

    # fetch at frame 1 returns only frame 0
    k_hist1, _ = cache.fetch_kv(frame_idx=1)
    assert k_hist1 is not None and k_hist1.shape == (B, 3, H, D)

    # fetch at frame 0 returns nothing
    k_none, v_none = cache.fetch_kv(frame_idx=0)
    assert k_none is None and v_none is None


@pytest.mark.L0
def test_kv_cache_circular_buffer():
    """Circular buffer wraps correctly: older entries are overwritten."""
    cache = KVCache(cache_size=2)
    B, H, D = 1, 2, 4

    k0 = torch.ones(B, 3, H, D) * 0
    k1 = torch.ones(B, 3, H, D) * 1
    k2 = torch.ones(B, 3, H, D) * 2  # overwrites frame 0

    cache.store_kv(k0, k0, frame_idx=0)
    cache.store_kv(k1, k1, frame_idx=1)
    cache.store_kv(k2, k2, frame_idx=2)

    # fetch at frame 3 should return frames 2 (start_idx=max(0,3-2+1)=2, history=[2])
    k_hist, _ = cache.fetch_kv(frame_idx=3)
    assert k_hist is not None
    assert k_hist.allclose(torch.ones(B, 3, H, D) * 2)


@pytest.mark.L0
def test_genkvcache_bf16_backend_round_trips_bit_exact() -> None:
    """Routing storage through the BF16 backend preserves K/V bit-exactly.

    The BF16 backend's encode/decode is a lossless identity, so storing a
    frame and fetching it back as history must return the exact stored
    tensors. This pins the default storage path to be indistinguishable from
    caching the tensors directly.
    """
    B, S, H, D = 1, 5, 4, 16
    cache = GenKVCache(cache_size=4, backend=BF16StorageBackend())

    k0 = torch.randn(B, S, H, D, dtype=torch.bfloat16)
    v0 = torch.randn(B, S, H, D, dtype=torch.bfloat16)
    cache.store_kv(k0, v0, frame_idx=0)

    k_hist, v_hist = cache.fetch_kv(frame_idx=1)  # [B,S,H,D] — frame 0 only
    assert k_hist is not None and v_hist is not None
    assert torch.equal(k_hist, k0)
    assert torch.equal(v_hist, v0)


@pytest.mark.L0
@pytest.mark.CPU
def test_dual_kv_cache_fp8_gen_round_trip() -> None:
    """DualKVCache(kv_cache_dtype="fp8") routes the gen cache through FP8.

    Storing two gen frames and fetching their history exercises the FP8
    encode/decode round-trip: the result comes back BF16 with the right
    shape, reconstruction stays within FP8's ~5% relative-error budget, and
    it is not bit-equal to the input (proof the values were quantized rather
    than passed through). The und cache is left untouched here.
    """
    B, S, H, D = 1, 100, 8, 128
    cache = DualKVCache(gen_cache_size=10, kv_cache_dtype="fp8", kv_cache_kernel_impl="torch")
    assert isinstance(cache.gen_cache.backend, FP8StorageBackend)
    assert cache.gen_cache.backend.kv_cache_dtype == "fp8"

    k = torch.randn(B, S, H, D, dtype=torch.bfloat16)
    v = torch.randn(B, S, H, D, dtype=torch.bfloat16)
    cache.gen_cache.store_kv(k, v, frame_idx=0)
    cache.gen_cache.store_kv(k, v, frame_idx=1)

    k_hist, v_hist = cache.gen_cache.fetch_kv(frame_idx=2)  # [B,2S,H,D]
    assert k_hist is not None and v_hist is not None
    assert k_hist.shape == (B, 2 * S, H, D)
    assert v_hist.shape == (B, 2 * S, H, D)
    assert k_hist.dtype == torch.bfloat16
    assert v_hist.dtype == torch.bfloat16

    rel_err = ((k_hist[:, :S] - k).abs().max() / k.abs().max()).item()
    assert rel_err < 0.05, f"FP8 gen round-trip rel_err too high: {rel_err}"
    # FP8 is lossy: a bit-exact result would mean the backend never quantized.
    assert not torch.equal(k_hist[:, :S], k)


@pytest.mark.L0
@pytest.mark.CPU
def test_dual_kv_cache_fp8_kernel_impl_defaults_allows_torch_and_rejects_unknown() -> None:
    """DualKVCache validates the FP8 batch decode kernel setting."""
    default_cache = DualKVCache(gen_cache_size=10, kv_cache_dtype="fp8")
    assert isinstance(default_cache.gen_cache.backend, FP8StorageBackend)
    assert default_cache.gen_cache.backend.kernel_impl == "triton"

    torch_cache = DualKVCache(gen_cache_size=10, kv_cache_dtype="fp8", kv_cache_kernel_impl="torch")
    assert isinstance(torch_cache.gen_cache.backend, FP8StorageBackend)
    assert torch_cache.gen_cache.backend.kernel_impl == "torch"

    with pytest.raises(ValueError, match="kernel_impl"):
        DualKVCache(gen_cache_size=10, kv_cache_dtype="fp8", kv_cache_kernel_impl="cuda")


@pytest.mark.L0
@pytest.mark.CPU
def test_dual_kv_cache_fp8_default_triton_fails_on_cpu_fetch() -> None:
    """Default FP8 fetch requires the Triton batch decode path."""
    cache = DualKVCache(
        gen_cache_size=5,
        kv_cache_dtype="fp8",
    )
    assert isinstance(cache.gen_cache.backend, FP8StorageBackend)
    assert cache.gen_cache.backend.kernel_impl == "triton"

    frames = [
        (
            torch.randn(1, 4, 2, 16, dtype=torch.bfloat16),  # [B,S,H,D]
            torch.randn(1, 4, 2, 16, dtype=torch.bfloat16),  # [B,S,H,D]
        )
        for _ in range(3)
    ]
    for frame_idx, (k, v) in enumerate(frames):
        cache.gen_cache.store_kv(k, v, frame_idx=frame_idx)

    with pytest.raises(RuntimeError, match='kernel_impl="torch"'):
        cache.gen_cache.fetch_kv(frame_idx=3)


@pytest.mark.L0
@pytest.mark.GPU
def test_gen_kv_cache_fp8_triton_fetches_kv_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dynamic fetch decodes K and V with the Triton path and no torch fallback."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the Triton FP8 backend")
    pytest.importorskip("triton")

    torch_cache = DualKVCache(
        gen_cache_size=5,
        kv_cache_dtype="fp8",
        kv_cache_kernel_impl="torch",
        attention_sink_size=1,
    )
    triton_cache = DualKVCache(
        gen_cache_size=5,
        kv_cache_dtype="fp8",
        kv_cache_kernel_impl="triton",
        attention_sink_size=1,
    )
    backend = triton_cache.gen_cache.backend
    assert isinstance(backend, FP8StorageBackend)

    for frame_idx in range(6):
        k = torch.randn(1, 4, 2, 16, device="cuda", dtype=torch.bfloat16)  # [B,S,H,D]
        v = torch.randn(1, 4, 2, 16, device="cuda", dtype=torch.bfloat16)  # [B,S,H,D]
        torch_cache.gen_cache.store_kv(k, v, frame_idx=frame_idx)
        triton_cache.gen_cache.store_kv(k, v, frame_idx=frame_idx)

    expected_k, expected_v = torch_cache.gen_cache.fetch_kv(frame_idx=6)

    def fail_on_fallback(_entry: object) -> torch.Tensor:
        raise AssertionError("triton unexpectedly fell back to per-entry decode")

    monkeypatch.setattr(backend, "decode", fail_on_fallback)
    actual_k, actual_v = triton_cache.gen_cache.fetch_kv(frame_idx=6)

    assert expected_k is not None and expected_v is not None
    assert actual_k is not None and actual_v is not None
    torch.testing.assert_close(actual_k, expected_k)
    torch.testing.assert_close(actual_v, expected_v)


@pytest.mark.L0
def test_gen_kv_cache_fp8_fetch_preserves_attention_sink_order() -> None:
    """FP8 fetch returns pinned sinks followed by the chronological rolling tail."""
    B, S, H, D = 1, 2, 2, 4
    cache = GenKVCache(cache_size=4, backend=FP8StorageBackend(kernel_impl="torch"), attention_sink_size=1)

    for frame_idx in range(6):
        k = torch.full((B, S, H, D), float(frame_idx), dtype=torch.bfloat16)  # [B,S,H,D]
        v = torch.full((B, S, H, D), float(frame_idx + 100), dtype=torch.bfloat16)  # [B,S,H,D]
        cache.store_kv(k, v, frame_idx=frame_idx)

    k_hist, v_hist = cache.fetch_kv(frame_idx=6)

    assert k_hist is not None and v_hist is not None
    assert k_hist.shape == (B, 3 * S, H, D)
    assert v_hist.shape == (B, 3 * S, H, D)
    expected_k_frames = [0, 4, 5]
    expected_v_frames = [100, 104, 105]
    for offset, (k_frame_idx, v_frame_idx) in enumerate(zip(expected_k_frames, expected_v_frames)):
        start = offset * S
        end = start + S
        expected_k = torch.full((B, S, H, D), float(k_frame_idx), dtype=torch.bfloat16)  # [B,S,H,D]
        expected_v = torch.full((B, S, H, D), float(v_frame_idx), dtype=torch.bfloat16)  # [B,S,H,D]
        torch.testing.assert_close(k_hist[:, start:end], expected_k, atol=0.05, rtol=0)
        torch.testing.assert_close(v_hist[:, start:end], expected_v, atol=0.05, rtol=0)


@pytest.mark.L0
def test_dual_kv_cache_default_gen_bit_equal() -> None:
    """The default DualKVCache (no kv_cache_dtype) keeps the gen cache BF16.

    Omitting the parameter must leave the gen cache on the lossless BF16
    backend, so a store/fetch round-trip is bit-identical to the input. This
    pins the default path as bit-exact regardless of the FP8 option.
    """
    B, S, H, D = 1, 100, 8, 128
    cache = DualKVCache(gen_cache_size=10, kv_cache_kernel_impl="not-used")  # default => BF16

    k = torch.randn(B, S, H, D, dtype=torch.bfloat16)
    v = torch.randn(B, S, H, D, dtype=torch.bfloat16)
    cache.gen_cache.store_kv(k, v, frame_idx=0)
    cache.gen_cache.store_kv(k, v, frame_idx=1)

    k_hist, v_hist = cache.gen_cache.fetch_kv(frame_idx=2)  # [B,2S,H,D]
    assert k_hist is not None and v_hist is not None
    assert torch.equal(k_hist, torch.cat([k, k], dim=1))
    assert torch.equal(v_hist, torch.cat([v, v], dim=1))


@pytest.mark.L0
def test_dual_kv_cache_unknown_dtype_raises() -> None:
    """An unrecognized kv_cache_dtype fails fast rather than silently degrading.

    Mapping an unknown format to a default backend would hide a typo and let
    a run proceed with the wrong storage precision, so the constructor raises
    ValueError naming the offending parameter.
    """
    with pytest.raises(ValueError, match="None or 'fp8'"):
        DualKVCache(gen_cache_size=10, kv_cache_dtype="fp9")


@pytest.mark.L0
def test_dual_kv_cache_und_stays_bf16_under_fp8() -> None:
    """The und (text) cache stays BF16 even when the gen cache is FP8.

    ``kv_cache_dtype="fp8"`` only threads the FP8 backend into the gen cache;
    ``UndKVCache`` takes no backend and stores K/V via a plain detach-and-clone,
    so an und store/get round-trip must be bit-identical to the input. This
    pins the structural contract that quantization never reaches the und cache,
    contrasting with the gen cache which is intentionally lossy under FP8.
    """
    B, S, H, D = 1, 50, 8, 128
    cache = DualKVCache(gen_cache_size=10, kv_cache_dtype="fp8")

    k_und = torch.randn(B, S, H, D, dtype=torch.bfloat16)
    v_und = torch.randn(B, S, H, D, dtype=torch.bfloat16)
    cache.und_cache.store(k_und, v_und)

    k_out, v_out = cache.und_cache.get()  # [B,S,H,D]
    assert torch.equal(k_out, k_und)
    assert torch.equal(v_out, v_und)


@pytest.mark.L0
def test_genkvcache_static_history_bf16_backend_bit_exact() -> None:
    """The static-history path is bit-exact under the default backend.

    ``fetch_kv_static`` rebuilds the fixed buffer through ``backend.decode``.
    With the lossless BF16 backend, the real prefix must equal the stored
    frames exactly, matching a plain direct-copy of the stored tensors.
    """
    B, S, H, D = 1, 3, 2, 4
    cache_size = 4
    max_tokens = (cache_size - 1) * S
    cache = GenKVCache(cache_size=cache_size, backend=BF16StorageBackend())

    frames: list[tuple[torch.Tensor, torch.Tensor]] = []
    for frame_idx in range(3):
        k = torch.randn(B, S, H, D, dtype=torch.bfloat16)  # [B,S,H,D]
        v = torch.randn(B, S, H, D, dtype=torch.bfloat16)  # [B,S,H,D]
        cache.store_kv(k, v, frame_idx=frame_idx)
        frames.append((k, v))

    k_static, v_static, real_len = cache.fetch_kv_static(
        3,
        max_tokens,
        S,
        num_heads=H,
        head_dim=D,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )

    assert real_len == 3 * S
    for i, (k, v) in enumerate(frames):
        assert torch.equal(k_static[:, i * S : (i + 1) * S], k)
        assert torch.equal(v_static[:, i * S : (i + 1) * S], v)


@pytest.mark.L0
def test_gen_kv_cache_static_fetch_matches_dynamic_prefix_and_reuses_buffer() -> None:
    """
    Test cache.fetch_kv_static and cache.fetch_kv_padded are equivalent when cache size is full.
    """
    B, S, H, D = 1, 3, 2, 4
    cache_size = 4
    max_tokens = (cache_size - 1) * S
    cache = GenKVCache(cache_size=cache_size)

    def _frame_kv(frame_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Use deterministic per-frame values so order mistakes are obvious:
        # if the ring buffer is copied in physical order instead of logical
        # chronological order, the prefix comparison below will fail.
        base = torch.arange(B * S * H * D, dtype=torch.float32).reshape(B, S, H, D)  # [B,S,H,D]
        k = base + frame_idx * 1000.0  # [B,S,H,D]
        v = base + frame_idx * 1000.0 + 100.0  # [B,S,H,D]
        return k, v

    # Prime an empty static buffer.  Frame 0 has no history, but the buffer
    # shape and pointer should already be stable for future reads.
    k_static, v_static, real_len = cache.fetch_kv_static(
        0,
        max_tokens,
        S,
        num_heads=H,
        head_dim=D,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert k_static.shape == (B, max_tokens, H, D)
    assert v_static.shape == (B, max_tokens, H, D)
    assert real_len == 0
    k_ptr = k_static.data_ptr()
    v_ptr = v_static.data_ptr()

    # Store enough frames to cover early fill, saturated history, and wrapped
    # ring-buffer cases.  With cache_size=4, frame 5 reads frames [3,4,5] in
    # logical order even though physical storage has wrapped.
    for frame_idx in range(6):
        k_frame, v_frame = _frame_kv(frame_idx)
        cache.store_kv(k_frame, v_frame, frame_idx=frame_idx)
        current_idx = frame_idx + 1

        # Compare against both old surfaces: dynamic fetch proves ordering and
        # eviction semantics, padded fetch proves the real prefix length.
        k_dyn, v_dyn = cache.fetch_kv(current_idx)
        k_padded, v_padded, padded_real_len = cache.fetch_kv_padded(
            current_idx,
            max_tokens,
            num_heads=H,
            head_dim=D,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        k_static, v_static, static_real_len = cache.fetch_kv_static(
            current_idx,
            max_tokens,
            S,
            num_heads=H,
            head_dim=D,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        # Second static read for the same frame should be a pure cache hit.
        k_static_again, v_static_again, static_real_len_again = cache.fetch_kv_static(
            current_idx,
            max_tokens,
            S,
            num_heads=H,
            head_dim=D,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        assert k_static.shape == (B, max_tokens, H, D)
        assert v_static.shape == (B, max_tokens, H, D)
        assert static_real_len == padded_real_len
        assert static_real_len_again == static_real_len
        assert k_static.data_ptr() == k_ptr
        assert v_static.data_ptr() == v_ptr
        assert k_static_again.data_ptr() == k_ptr
        assert v_static_again.data_ptr() == v_ptr
        assert k_dyn is not None and v_dyn is not None
        torch.testing.assert_close(k_static[:, :static_real_len], k_dyn)
        torch.testing.assert_close(v_static[:, :static_real_len], v_dyn)
        torch.testing.assert_close(k_static[:, :static_real_len], k_padded[:, :padded_real_len])
        torch.testing.assert_close(v_static[:, :static_real_len], v_padded[:, :padded_real_len])


@pytest.mark.L0
def test_gen_kv_cache_attention_sink_static_and_padded_match_dynamic() -> None:
    """Static and padded reads preserve attention-sink ordering."""
    B, S, H, D = 1, 2, 2, 4
    cache_size = 6
    attention_sink_size = 2
    max_tokens = (cache_size - 1) * S
    cache = GenKVCache(cache_size=cache_size, attention_sink_size=attention_sink_size)

    for frame_idx in range(10):
        k = torch.full((B, S, H, D), float(frame_idx))  # [B,S,H,D]
        v = torch.full((B, S, H, D), float(frame_idx + 100))  # [B,S,H,D]
        cache.store_kv(k, v, frame_idx=frame_idx)

    k_dyn, v_dyn = cache.fetch_kv(frame_idx=10)
    k_padded, v_padded, padded_real_len = cache.fetch_kv_padded(
        10,
        max_tokens,
        num_heads=H,
        head_dim=D,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    k_static, v_static, static_real_len = cache.fetch_kv_static(
        10,
        max_tokens,
        S,
        num_heads=H,
        head_dim=D,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert k_dyn is not None and v_dyn is not None
    assert padded_real_len == static_real_len == 5 * S
    expected_k_frames = [0, 1, 7, 8, 9]
    expected_v_frames = [100, 101, 107, 108, 109]
    for offset, (k_frame_idx, v_frame_idx) in enumerate(zip(expected_k_frames, expected_v_frames)):
        start = offset * S
        end = start + S
        expected_k = torch.full((B, S, H, D), float(k_frame_idx))  # [B,S,H,D]
        expected_v = torch.full((B, S, H, D), float(v_frame_idx))  # [B,S,H,D]
        torch.testing.assert_close(k_dyn[:, start:end], expected_k)
        torch.testing.assert_close(v_dyn[:, start:end], expected_v)

    torch.testing.assert_close(k_padded[:, :padded_real_len], k_dyn)
    torch.testing.assert_close(v_padded[:, :padded_real_len], v_dyn)
    torch.testing.assert_close(k_static[:, :static_real_len], k_dyn)
    torch.testing.assert_close(v_static[:, :static_real_len], v_dyn)


@pytest.mark.L0
def test_gen_kv_cache_static_fetch_invalidates_after_store() -> None:
    """A store after static fetch must invalidate stale history contents.

    This protects the invalidation rule in ``GenKVCache.store_kv``.  If the
    static buffer were reused after a store without rebuilding, a same-frame
    read would keep returning ``frame1_old`` after the cache slot is updated
    to ``frame1_new``.
    """
    B, S, H, D = 1, 2, 2, 4
    cache = GenKVCache(cache_size=3)
    max_tokens = 2 * S

    k0 = torch.full((B, S, H, D), 1.0)  # [B,S,H,D]
    v0 = torch.full((B, S, H, D), 10.0)  # [B,S,H,D]
    k1_old = torch.full((B, S, H, D), 2.0)  # [B,S,H,D]
    v1_old = torch.full((B, S, H, D), 20.0)  # [B,S,H,D]
    k1_new = torch.full((B, S, H, D), 3.0)  # [B,S,H,D]
    v1_new = torch.full((B, S, H, D), 30.0)  # [B,S,H,D]

    cache.store_kv(k0, v0, frame_idx=0)
    cache.store_kv(k1_old, v1_old, frame_idx=1)
    # Build a valid static history for frame 2.  This makes the internal
    # workspace valid for frame_idx=2 and populated with frame1_old.
    k_static, v_static, real_len = cache.fetch_kv_static(
        2,
        max_tokens,
        S,
        num_heads=H,
        head_dim=D,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert real_len == 2 * S
    torch.testing.assert_close(k_static[:, :S], k0)
    torch.testing.assert_close(v_static[:, :S], v0)
    torch.testing.assert_close(k_static[:, S : 2 * S], k1_old)
    torch.testing.assert_close(v_static[:, S : 2 * S], v1_old)

    # Overwrite the same logical frame while reading the same requested
    # frame_idx=2 again.  The frame index alone would not force a rebuild; this
    # only returns frame1_new if store_kv invalidated the static workspace.
    cache.store_kv(k1_new, v1_new, frame_idx=1)
    k_static, v_static, real_len = cache.fetch_kv_static(
        2,
        max_tokens,
        S,
        num_heads=H,
        head_dim=D,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert real_len == 2 * S
    torch.testing.assert_close(k_static[:, :S], k0)
    torch.testing.assert_close(v_static[:, :S], v0)
    torch.testing.assert_close(k_static[:, S : 2 * S], k1_new)
    torch.testing.assert_close(v_static[:, S : 2 * S], v1_new)


@pytest.mark.L0
def test_gen_kv_cache_static_fetch_uses_cached_head_shape() -> None:
    """Static fetch should size buffers from cached K/V for CP head-sharded caches.

    In context parallelism the cache may hold only the local head shard.  The
    static fetch API still receives model-level ``num_heads`` in some callers,
    so the implementation must prefer the cached tensor's actual ``H``.
    """
    B, S, H_cached, H_full, D = 1, 3, 2, 4, 8
    cache = GenKVCache(cache_size=4)
    max_tokens = 3 * S

    k = torch.randn(B, S, H_cached, D)  # [B,S,H_cp,D]
    v = torch.randn(B, S, H_cached, D)  # [B,S,H_cp,D]
    cache.store_kv(k, v, frame_idx=0)

    # Pass the full model head count on purpose.  The returned static buffer
    # should use H_cached, not H_full.
    k_static, v_static, real_len = cache.fetch_kv_static(
        1,
        max_tokens,
        S,
        num_heads=H_full,
        head_dim=D,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert real_len == S
    assert k_static.shape == (B, max_tokens, H_cached, D)
    assert v_static.shape == (B, max_tokens, H_cached, D)
    torch.testing.assert_close(k_static[:, :S], k)
    torch.testing.assert_close(v_static[:, :S], v)


@pytest.mark.L0
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_non_cp_ar_inference():
    """Non-CP AR inference: fetch+store+attend produces same result as full attention.

    Simulates the logic in PackedAttentionMoT.forward_with_kv_cache() (non-CP path):
      - Frame 0: store und K/V and gen K/V in cache as 4D [1,S,H,D]
      - Frame 1+: fetch history, store current, build full K/V, attend
    Verifies each frame's output matches single-pass full attention with the same context.
    """
    device = torch.device("cuda", 0)
    num_heads, head_dim = 4, 32
    S_und, S_gen = 6, 4
    num_frames = 3
    scale = head_dim**-0.5

    torch.manual_seed(7)
    k_und = torch.randn(S_und, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    v_und = torch.randn(S_und, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    q_frames = [torch.randn(S_gen, num_heads, head_dim, device=device, dtype=torch.bfloat16) for _ in range(num_frames)]
    k_frames = [torch.randn(S_gen, num_heads, head_dim, device=device, dtype=torch.bfloat16) for _ in range(num_frames)]
    v_frames = [torch.randn(S_gen, num_heads, head_dim, device=device, dtype=torch.bfloat16) for _ in range(num_frames)]

    # Baseline: frame-by-frame full attention (no cache)
    # Context: [und | frame0 | frame1 | ...]
    baseline_results = []
    k_ctx = torch.cat([k_und.unsqueeze(0), k_frames[0].unsqueeze(0)], dim=1)  # [1,S_und+S_gen,H,D]
    v_ctx = torch.cat([v_und.unsqueeze(0), v_frames[0].unsqueeze(0)], dim=1)
    for fi in range(1, num_frames):
        k_full = torch.cat([k_ctx, k_frames[fi].unsqueeze(0)], dim=1)  # [1,S_ctx+S_gen,H,D]
        v_full = torch.cat([v_ctx, v_frames[fi].unsqueeze(0)], dim=1)
        out = attention(
            query=q_frames[fi].unsqueeze(0), key=k_full, value=v_full, is_causal=False, scale=scale, return_lse=False
        )
        assert isinstance(out, torch.Tensor)
        out_t: torch.Tensor = out
        baseline_results.append(out_t.squeeze(0))  # [S_gen,H,D]
        k_ctx = k_full
        v_ctx = v_full

    # Non-CP path: pre-populate cache with frame 0 (as forward() does at frame 0)
    dual_cache = DualKVCache(gen_cache_size=num_frames + 2)
    dual_cache.und_cache.store(k_und.unsqueeze(0), v_und.unsqueeze(0))  # [1,S_und,H,D]
    dual_cache.gen_cache.store_kv(k_frames[0].unsqueeze(0), v_frames[0].unsqueeze(0), frame_idx=0)  # [1,S_gen,H,D]

    # Run frames 1..num_frames-1 using the non-CP forward_with_kv_cache() logic
    for fi in range(1, num_frames):
        q_gen_ = q_frames[fi]  # [S_gen,H,D]
        k_gen_ = k_frames[fi]  # [S_gen,H,D]
        v_gen = v_frames[fi]  # [S_gen,H,D]

        k_gen_hist, v_gen_hist = dual_cache.gen_cache.fetch_kv(fi)  # [1,S_hist,H,D] or None
        dual_cache.gen_cache.store_kv(k_gen_.unsqueeze(0), v_gen.unsqueeze(0), frame_idx=fi)  # [1,S_gen,H,D]

        k_und_cached, v_und_cached = dual_cache.und_cache.get()  # [1,S_und,H,D]

        k_curr = k_gen_.unsqueeze(0)  # [1,S_gen,H,D]
        v_curr = v_gen.unsqueeze(0)  # [1,S_gen,H,D]

        if k_gen_hist is not None:
            k_full = torch.cat([k_und_cached, k_gen_hist, k_curr], dim=1)  # [1,S_total,H,D]
            v_full = torch.cat([v_und_cached, v_gen_hist, v_curr], dim=1)
        else:
            k_full = torch.cat([k_und_cached, k_curr], dim=1)
            v_full = torch.cat([v_und_cached, v_curr], dim=1)

        attn_result = attention(
            query=q_gen_.unsqueeze(0), key=k_full, value=v_full, is_causal=False, scale=scale, return_lse=False
        )
        assert isinstance(attn_result, torch.Tensor)
        attn_out: torch.Tensor = attn_result
        out = attn_out.squeeze(0)  # [S_gen,H,D]

        torch.testing.assert_close(out, baseline_results[fi - 1], rtol=1e-2, atol=1e-2, msg=f"Frame {fi} mismatch")


@pytest.mark.L0
def test_ar_memory_state_read_for_layer():
    """ARMemoryState.read_for_layer returns correctly populated ARMemoryValue.

    Verifies:
    - frame_idx and gen_len are propagated correctly.
    - Und K/V are None when und_cache is not initialized (before frame 0 store).
    - Und K/V are present after und_cache.store().
    - Gen history is None at frame 0 and frame 1, and correct at frame 2+.
    - Multiple layers return independent values from their own caches.
    """
    B, H, D = 1, 4, 16
    S_und, S_gen = 6, 4
    num_layers = 3
    gen_cache_size = 5

    torch.manual_seed(42)
    und_k = torch.randn(B, S_und, H, D)
    und_v = torch.randn(B, S_und, H, D)

    # One DualKVCache per transformer layer, as in production.
    caches = [DualKVCache(gen_cache_size=gen_cache_size) for _ in range(num_layers)]

    # --- Frame 0: caches are empty ---
    # A new ARMemoryState is constructed each frame (mirroring cosmos3_vfm_network.py).
    # init() captures gen_len from the hidden_states metadata dict.
    state_f0 = ARMemoryState(caches, frame_idx=0)
    state_f0.init({"_num_full_tokens": S_gen}, torch.device("cpu"))
    mv0 = state_f0.read_for_layer(0)

    # At frame 0, nothing has been cached yet, so all cached fields are None.
    assert isinstance(mv0, ARMemoryValue)
    assert mv0.frame_idx == 0
    assert mv0.gen_len == S_gen
    assert mv0.und_k_cached is None
    assert mv0.und_v_cached is None
    assert mv0.gen_k_hist is None
    assert mv0.gen_v_hist is None

    # Simulate frame 0 write for all layers: stores both gen K/V and und K/V.
    # write_for_layer stores und K/V only when und_cache is not yet initialized.
    for i in range(num_layers):
        gen_k = torch.randn(B, S_gen, H, D)
        gen_v = torch.randn(B, S_gen, H, D)
        state_f0.write_for_layer(i, (gen_k, gen_v, und_k, und_v))

    # --- Frame 1: und_cache is now populated, gen history has frame 0 ---
    state_f1 = ARMemoryState(caches, frame_idx=1)
    state_f1.init({"_num_full_tokens": S_gen}, torch.device("cpu"))
    mv1 = state_f1.read_for_layer(0)

    assert mv1.frame_idx == 1
    assert mv1.gen_len == S_gen
    # Und K/V should be the exact tensors stored at frame 0.
    assert mv1.und_k_cached is not None
    assert mv1.und_v_cached is not None
    assert torch.equal(mv1.und_k_cached, und_k)
    assert torch.equal(mv1.und_v_cached, und_v)
    # Gen history at frame 1 = [frame 0], so shape is [B, S_gen, H, D].
    assert mv1.gen_k_hist is not None, "frame 1 should see frame 0's gen K"
    assert mv1.gen_k_hist.shape == (B, S_gen, H, D)

    # Simulate frame 1 write (gen K/V only — und K/V should be ignored).
    for i in range(num_layers):
        gen_k1 = torch.randn(B, S_gen, H, D)
        gen_v1 = torch.randn(B, S_gen, H, D)
        state_f1.write_for_layer(i, (gen_k1, gen_v1, und_k, und_v))

    # --- Frame 2: gen history should contain frames 0 and 1 concatenated ---
    state_f2 = ARMemoryState(caches, frame_idx=2)
    state_f2.init({"_num_full_tokens": S_gen}, torch.device("cpu"))
    mv2 = state_f2.read_for_layer(0)

    assert mv2.frame_idx == 2
    assert mv2.und_k_cached is not None
    assert mv2.gen_k_hist is not None
    assert mv2.gen_k_hist.shape == (B, 2 * S_gen, H, D), "should have history from frames 0 and 1"

    # --- Different layers should return independent cache state ---
    # Each layer wrote different random gen K/V at frames 0 and 1, so
    # their histories must differ even though shapes match.
    mv_layer0 = state_f2.read_for_layer(0)
    mv_layer2 = state_f2.read_for_layer(2)
    assert mv_layer0.gen_k_hist is not None
    assert mv_layer2.gen_k_hist is not None
    assert mv_layer0.gen_k_hist.shape == mv_layer2.gen_k_hist.shape
    assert not torch.equal(mv_layer0.gen_k_hist, mv_layer2.gen_k_hist), (
        "different layers should have different cached values"
    )

    # --- write_for_layer at frame 1+ must NOT overwrite und K/V ---
    # The guard `if not cache.und_cache.is_initialized` in ARMemoryState prevents
    # this. Verify by passing all-zeros as und K/V and confirming the cache is unchanged.
    original_und_k, _ = caches[0].und_cache.get()
    dummy_und_k = torch.zeros_like(und_k)
    gen_k_dummy = torch.randn(B, S_gen, H, D)
    gen_v_dummy = torch.randn(B, S_gen, H, D)
    state_f2.write_for_layer(0, (gen_k_dummy, gen_v_dummy, dummy_und_k, dummy_und_k))
    current_und_k, _ = caches[0].und_cache.get()
    assert torch.equal(current_und_k, original_und_k), "und K/V must not be overwritten after frame 0"


@pytest.mark.L0
def test_ar_memory_state_post_saturation_static_compile_keeps_true_frame_idx() -> None:
    """Static post-saturation compile keeps true frame_idx and a stable dispatch flag."""
    B, S_und, S_gen, H, D = 1, 2, 3, 2, 4
    cache = DualKVCache(gen_cache_size=4, attention_sink_size=1)
    und_k = torch.full((B, S_und, H, D), -1.0)  # [B,S_und,H,D]
    und_v = torch.full((B, S_und, H, D), -2.0)  # [B,S_und,H,D]
    cache.und_cache.store(und_k, und_v)

    for frame_idx in range(5):
        gen_k = torch.full((B, S_gen, H, D), float(frame_idx))  # [B,S_gen,H,D]
        gen_v = torch.full((B, S_gen, H, D), float(frame_idx + 10))  # [B,S_gen,H,D]
        cache.gen_cache.store_kv(gen_k, gen_v, frame_idx=frame_idx)

    state = ARMemoryState(
        [cache],
        frame_idx=5,
        post_saturation_static_compile=True,
        static_und_cache_max_len=4,
    )
    state.init({"_num_full_tokens": S_gen}, torch.device("cpu"))
    memory_value = state.read_for_layer(0)

    assert state.frame_idx == 5
    assert memory_value.frame_idx == 5
    assert memory_value.post_saturation_static_compile is True
    assert memory_value.und_k_cached is not None
    assert memory_value.und_v_cached is not None
    assert memory_value.und_k_cached.shape == (B, 4, H, D)
    assert memory_value.und_v_cached.shape == (B, 4, H, D)
    assert memory_value.real_und_cache_len_t is not None
    assert memory_value.real_und_cache_len_t.item() == S_und
    assert memory_value.cu_seqlens_q_t is not None
    assert memory_value.cu_seqlens_kv_t is not None
    assert memory_value.cu_seqlens_q_t.tolist() == [0, S_gen]
    assert memory_value.cu_seqlens_kv_t.tolist() == [0, S_und + 3 * S_gen + S_gen]
    assert memory_value.max_seqlen_KV == 4 + 3 * S_gen + S_gen
    assert memory_value.gen_k_hist is not None
    assert memory_value.gen_v_hist is not None

    expected_history_k = torch.cat(
        [
            torch.full((B, S_gen, H, D), 0.0),  # [B,S_gen,H,D]
            torch.full((B, S_gen, H, D), 3.0),  # [B,S_gen,H,D]
            torch.full((B, S_gen, H, D), 4.0),  # [B,S_gen,H,D]
        ],
        dim=1,
    )  # [B,S_hist,H,D]
    expected_history_v = torch.cat(
        [
            torch.full((B, S_gen, H, D), 10.0),  # [B,S_gen,H,D]
            torch.full((B, S_gen, H, D), 13.0),  # [B,S_gen,H,D]
            torch.full((B, S_gen, H, D), 14.0),  # [B,S_gen,H,D]
        ],
        dim=1,
    )  # [B,S_hist,H,D]
    torch.testing.assert_close(memory_value.gen_k_hist, expected_history_k)
    torch.testing.assert_close(memory_value.gen_v_hist, expected_history_v)


@pytest.mark.L0
def test_ar_memory_state_post_saturation_static_compile_rejects_oversized_und_cache() -> None:
    """Post-saturation static compile must fail before handing variable S_und to Dynamo."""
    B, S_und, S_gen, H, D = 1, 3, 2, 2, 4
    cache = DualKVCache(gen_cache_size=4)
    und_k = torch.randn(B, S_und, H, D)  # [B,S_und,H,D]
    und_v = torch.randn(B, S_und, H, D)  # [B,S_und,H,D]
    cache.und_cache.store(und_k, und_v)

    state = ARMemoryState(
        [cache],
        frame_idx=3,
        post_saturation_static_compile=True,
        static_und_cache_max_len=2,
    )

    with pytest.raises(ValueError, match="cached S_und=3 exceeds ar_static_und_cache_max_len=2"):
        state.init({"_num_full_tokens": S_gen}, torch.device("cpu"))


@pytest.mark.L0
def test_ar_memory_state_coarse_cuda_graph_refreshes_history_and_stages_writes() -> None:
    """Coarse graph state preserves history addresses and commits refresh K/V explicitly."""
    B, S_und, S_gen, H, D = 1, 2, 3, 2, 4
    cache = DualKVCache(gen_cache_size=4)
    und_k = torch.full((B, S_und, H, D), -1.0)  # [B,S_und,H,D]
    und_v = torch.full((B, S_und, H, D), -2.0)  # [B,S_und,H,D]
    cache.und_cache.store(und_k, und_v)
    for frame_idx in range(4):
        gen_k = torch.full((B, S_gen, H, D), float(frame_idx))  # [B,S_gen,H,D]
        gen_v = torch.full((B, S_gen, H, D), float(frame_idx + 10))  # [B,S_gen,H,D]
        cache.gen_cache.store_kv(gen_k, gen_v, frame_idx)

    state = ARMemoryState(
        [cache],
        frame_idx=4,
        num_kv_heads=H,
        head_dim=D,
        write_gen_cache=True,
        post_saturation_static_compile=True,
        static_und_cache_max_len=4,
        coarse_cuda_graph=True,
        stage_gen_cache_writes=True,
    )
    state.init({"_num_full_tokens": S_gen}, torch.device("cpu"))
    memory_value = state.read_for_layer(0)
    assert memory_value.gen_k_hist is not None
    captured_history_ptr = memory_value.gen_k_hist.data_ptr()

    gen_k4 = torch.full((B, S_gen, H, D), 4.0)  # [B,S_gen,H,D]
    gen_v4 = torch.full((B, S_gen, H, D), 14.0)  # [B,S_gen,H,D]
    state.write_for_layer(0, (gen_k4, gen_v4, und_k, und_v))
    cached_entry_before_commit = cache.gen_cache.k_cache[0]
    assert cached_entry_before_commit is not None
    cached_before_commit = cache.gen_cache.backend.decode(cached_entry_before_commit)  # [B,S_gen,H,D]
    assert torch.equal(cached_before_commit, torch.zeros_like(gen_k4))

    state.commit_staged_gen_cache(frame_idx=4)
    state.prepare_for_coarse_cuda_graph_replay(frame_idx=5)

    assert memory_value.gen_k_hist.data_ptr() == captured_history_ptr
    expected_history = torch.cat(
        [
            torch.full((B, S_gen, H, D), 2.0),  # [B,S_gen,H,D]
            torch.full((B, S_gen, H, D), 3.0),  # [B,S_gen,H,D]
            torch.full((B, S_gen, H, D), 4.0),  # [B,S_gen,H,D]
        ],
        dim=1,
    )  # [B,S_hist,H,D]
    torch.testing.assert_close(memory_value.gen_k_hist, expected_history)


# =============================================================================
# KVCacheTrainMemoryState: layer-loop equivalence vs eager AR + init invariants
# =============================================================================


def _mk_gen_only_pack(gen_seq: torch.Tensor, device: torch.device) -> SequencePack:
    """SequencePack with no causal (und) tokens — used for the eager
    ``attention_AR_gen_only`` golden."""
    S = gen_seq.shape[0]
    return {
        "causal_seq": gen_seq.new_empty(0, *gen_seq.shape[1:]),
        "full_only_seq": gen_seq,
        "is_sharded": False,
        "sample_offsets": torch.tensor([0, S], device=device, dtype=torch.int32),
        "max_sample_len": S,
        "max_causal_len": 0,
        "max_full_len": S,
        "_causal_indices": torch.empty(0, dtype=torch.long, device=device),
        "_full_indices": torch.arange(S, device=device),
        "_causal_seq_offsets": torch.tensor([0], device=device, dtype=torch.int32),
        "_full_only_seq_offsets": torch.tensor([0, S], device=device, dtype=torch.int32),
        "_num_causal_tokens": 0,
        "_num_full_tokens": S,
    }


def _mk_factored_pack_with_caption(
    und: torch.Tensor,
    gen: torch.Tensor,
    *,
    has_new_caption: bool,
    padded_causal_len: int,
    device: torch.device,
) -> SequencePack:
    """SequencePack for the rolling path.

    ``causal_seq`` is always padded to ``padded_causal_len`` (compile
    stability).  When ``has_new_caption=True`` the real und K/V live there;
    when ``False`` it is zero-padding and the rolling-path attention falls
    back to ``memory_value.cached_und_k/v``.
    """
    assert und.shape[0] <= padded_causal_len
    real_und_len = und.shape[0]
    if has_new_caption:
        pad_rows = padded_causal_len - real_und_len
        causal_seq = (
            und
            if pad_rows == 0
            else torch.cat([und, torch.zeros(pad_rows, *und.shape[1:], dtype=und.dtype, device=device)])
        )
        num_causal_tokens = real_und_len
    else:
        causal_seq = torch.zeros(padded_causal_len, *und.shape[1:], dtype=und.dtype, device=device)
        num_causal_tokens = 0
    S_g = gen.shape[0]
    return {
        "causal_seq": causal_seq,
        "full_only_seq": gen,
        "is_sharded": False,
        "sample_offsets": torch.tensor([0, padded_causal_len + S_g], device=device, dtype=torch.int32),
        "max_sample_len": padded_causal_len + S_g,
        "max_causal_len": num_causal_tokens,
        "max_full_len": S_g,
        "_causal_indices": torch.arange(padded_causal_len, device=device),
        "_full_indices": torch.arange(padded_causal_len, padded_causal_len + S_g, device=device),
        "_causal_seq_offsets": torch.tensor([0, num_causal_tokens], device=device, dtype=torch.int32),
        "_full_only_seq_offsets": torch.tensor([0, S_g], device=device, dtype=torch.int32),
        "_num_causal_tokens": num_causal_tokens,
        "_num_full_tokens": S_g,
    }


@pytest.mark.L0
@pytest.mark.CPU
def test_kv_cache_training_state_sizes_two_aligned_vision_layouts() -> None:
    """A transfer segment reserves cache space for both control and target tokens."""
    device = torch.device("cpu")
    T, H_p, W_p = 2, 1, 2
    num_action_tokens = 0
    tokens_per_item = T * (num_action_tokens + H_p * W_p)
    tokens_per_segment = 2 * tokens_per_item
    cache_size = 3
    state = KVCacheTrainMemoryState(
        vision_token_shapes=[(T, H_p, W_p), (T, H_p, W_p)],
        num_action_tokens_per_supertoken=num_action_tokens,
        null_action_supertokens=False,
        segment_idx=1,
        dual_kv_cache=[DualKVCache(gen_cache_size=cache_size)],
        num_kv_heads=1,
        head_dim=4,
    )
    und = torch.randn(2, 1, 4, device=device)  # [S_text,H_kv,D]
    gen = torch.randn(tokens_per_segment, 1, 4, device=device)  # [S_gen,H_kv,D]
    pack = _mk_factored_pack_with_caption(
        und,
        gen,
        has_new_caption=True,
        padded_causal_len=und.shape[0],
        device=device,
    )

    state.init(pack, device=device)

    assert state.max_gen_cache_tokens == (cache_size - 1) * tokens_per_segment
    assert state.gen_ca_cached_kv_offsets is not None
    expected_offsets = torch.tensor([0, tokens_per_segment], dtype=torch.int32)  # [2]
    torch.testing.assert_close(state.gen_ca_cached_kv_offsets, expected_offsets)


@pytest.mark.L0
@pytest.mark.CPU
def test_teacher_forcing_memory_state_supports_cp_head_sharded_cache() -> None:
    """Replay TF opts into CP and returns cache tensors with caller-provided local KV heads."""
    device = torch.device("cpu")
    dtype = torch.float32
    T, H_p, W_p, tcf = 2, 1, 2, 0
    S_super = tcf + H_p * W_p
    tokens_per_seg = T * S_super
    padded_causal_len = 4
    context_parallel_size = 2
    local_padded_causal_len = padded_causal_len // context_parallel_size
    local_tokens_per_seg = tokens_per_seg // context_parallel_size
    local_num_kv_heads = 2
    head_dim = 8
    dual_caches = [DualKVCache(gen_cache_size=2)]
    state = TeacherForcingMemoryState(
        vision_token_shapes=[(T, H_p, W_p)],
        num_action_tokens_per_supertoken=tcf,
        null_action_supertokens=False,
        segment_idx=0,
        dual_kv_cache=dual_caches,
        num_kv_heads=local_num_kv_heads,
        head_dim=head_dim,
        frames_per_chunk=4,
        transfer_control_attention_mode="causal_control",
        context_parallel_size=context_parallel_size,
    )
    pack = _mk_factored_pack_with_caption(
        torch.randn(
            local_padded_causal_len,
            local_num_kv_heads,
            head_dim,
            device=device,
            dtype=dtype,
        ),  # [S_text/cp,H_kv_local,D]
        torch.randn(
            local_tokens_per_seg,
            local_num_kv_heads,
            head_dim,
            device=device,
            dtype=dtype,
        ),  # [S_gen/cp,H_kv_local,D]
        has_new_caption=True,
        padded_causal_len=local_padded_causal_len,
        device=device,
    )
    pack["_num_causal_tokens"] = padded_causal_len
    pack["_num_full_tokens"] = tokens_per_seg

    state.init(pack, device=device)
    pass1_value = state.read_for_layer(0)

    assert isinstance(pass1_value, TFReplayCleanMemoryValue)
    assert pass1_value.supports_context_parallel_attention
    assert pass1_value.frames_per_chunk == 4
    assert pass1_value.transfer_control_attention_mode == "causal_control"
    assert pass1_value.cached_und_k.shape == (1, padded_causal_len, local_num_kv_heads, head_dim)
    assert pass1_value.cached_gen_k.shape == (1, tokens_per_seg, local_num_kv_heads, head_dim)

    gen_k = torch.randn(
        1, tokens_per_seg, local_num_kv_heads, head_dim, device=device, dtype=dtype
    )  # [1,S_gen,H_kv_local,D]
    gen_v = torch.randn(
        1, tokens_per_seg, local_num_kv_heads, head_dim, device=device, dtype=dtype
    )  # [1,S_gen,H_kv_local,D]
    und_k = torch.randn(
        1, padded_causal_len, local_num_kv_heads, head_dim, device=device, dtype=dtype
    )  # [1,S_text,H_kv_local,D]
    und_v = torch.randn(
        1, padded_causal_len, local_num_kv_heads, head_dim, device=device, dtype=dtype
    )  # [1,S_text,H_kv_local,D]
    state.write_for_layer(0, (gen_k, gen_v, und_k, und_v))

    state.pass_number = 2
    pass2_value = state.read_for_layer(0)

    assert isinstance(pass2_value, TFNoisyMemoryValue)
    assert pass2_value.supports_context_parallel_attention
    assert pass2_value.frames_per_chunk == 4
    assert pass2_value.transfer_control_attention_mode == "causal_control"
    assert pass2_value.cached_clean_gen_k.shape == (1, tokens_per_seg, local_num_kv_heads, head_dim)
    assert pass2_value.cached_clean_gen_v.shape == (1, tokens_per_seg, local_num_kv_heads, head_dim)


@pytest.mark.L0
@pytest.mark.CPU
def test_teacher_forcing_memory_state_compacts_clean_target_and_preserves_gradients() -> None:
    """Two-way Flex stores only selected target K/V and keeps Pass-2 gradients into Pass 1."""
    device = torch.device("cpu")
    num_kv_heads, head_dim = 2, 4
    clean_indexes = torch.tensor([1, 4])  # [S_clean_real]
    state = TeacherForcingMemoryState(
        vision_token_shapes=[(3, 1, 2), (3, 1, 2)],
        num_action_tokens_per_supertoken=0,
        null_action_supertokens=False,
        segment_idx=0,
        dual_kv_cache=[DualKVCache(gen_cache_size=2)],
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        detach_clean_kv=False,
        selected_clean_gen_token_indexes=clean_indexes,
        selected_clean_gen_padded_capacity=4,
    )
    pack = _mk_factored_pack_with_caption(
        torch.randn(2, num_kv_heads, head_dim),  # [S_text,H_kv,D]
        torch.randn(6, num_kv_heads, head_dim),  # [S_gen,H_kv,D]
        has_new_caption=True,
        padded_causal_len=2,
        device=device,
    )
    state.init(pack, device=device)

    gen_k = torch.randn(1, 6, num_kv_heads, head_dim, requires_grad=True)  # [1,S_gen,H_kv,D]
    gen_v = torch.randn(1, 6, num_kv_heads, head_dim, requires_grad=True)  # [1,S_gen,H_kv,D]
    und_k = torch.randn(1, 2, num_kv_heads, head_dim)  # [1,S_text,H_kv,D]
    und_v = torch.randn(1, 2, num_kv_heads, head_dim)  # [1,S_text,H_kv,D]
    state.write_for_layer(0, (gen_k, gen_v, und_k, und_v))
    state.pass_number = 2
    pass2_value = state.read_for_layer(0)

    assert isinstance(pass2_value, TFNoisyMemoryValue)
    assert pass2_value.cached_clean_gen_k.shape == (1, 4, num_kv_heads, head_dim)
    assert pass2_value.cached_clean_gen_v.shape == (1, 4, num_kv_heads, head_dim)
    expected_clean_k = gen_k[:, clean_indexes]  # [1,S_clean_real,H_kv,D]
    expected_clean_v = gen_v[:, clean_indexes]  # [1,S_clean_real,H_kv,D]
    torch.testing.assert_close(pass2_value.cached_clean_gen_k[:, :2], expected_clean_k)
    torch.testing.assert_close(pass2_value.cached_clean_gen_v[:, :2], expected_clean_v)
    assert torch.count_nonzero(pass2_value.cached_clean_gen_k[:, 2:]) == 0  # []
    assert torch.count_nonzero(pass2_value.cached_clean_gen_v[:, 2:]) == 0  # []

    loss = pass2_value.cached_clean_gen_k.sum() + pass2_value.cached_clean_gen_v.sum()  # []
    loss.backward()
    selected = torch.tensor([False, True, False, False, True, False])  # [S_gen]
    gen_k_has_grad = gen_k.grad.abs().sum(dim=(0, 2, 3)) > 0  # [S_gen]
    gen_v_has_grad = gen_v.grad.abs().sum(dim=(0, 2, 3)) > 0  # [S_gen]
    assert torch.equal(gen_k_has_grad, selected)
    assert torch.equal(gen_v_has_grad, selected)


@pytest.mark.L0
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_rolling_kv_cache_layer_loop_matches_eager_ar():
    """End-to-end: ``KVCacheTrainMemoryState.init/read_for_layer/write_for_layer``
    wrapped around ``three_way_attention_with_kv_cache`` matches a manually-
    maintained eager ``attention_AR_gen_only`` loop, segment by segment.

    Mirrors ``test_non_cp_ar_inference_rolling_window`` but:

    - The candidate runs the full ``KVCacheTrainMemoryState`` plumbing
      (init flags + read_for_layer padded fetch + write_for_layer rolling
      store), so bugs in flag/offset/padding construction are caught.
    - The golden runs a parallel manual tracking of gen history (same
      rolling window cut-off) through ``attention_AR_gen_only``.

    ``num_frames=5`` with ``gen_cache_size=3`` exercises empty (seg 0),
    partial (seg 1, 2), saturation boundary (seg 3 evicts seg 0), and
    post-saturation (seg 4).
    """
    device = torch.device("cuda")
    dtype = torch.bfloat16
    T, H_p, W_p, tcf = 1, 2, 2, 0
    S_super = tcf + H_p * W_p
    tokens_per_seg = T * S_super
    num_heads = 8
    num_kv_heads = 8
    head_dim = 32
    S_und = 6
    padded_causal_len = S_und  # no extra padding for simplicity
    gen_cache_size = 3  # holds up to gen_cache_size-1 = 2 past segments
    num_frames = 5

    torch.manual_seed(1337)
    und_k = torch.randn(S_und, num_kv_heads, head_dim, device=device, dtype=dtype)
    und_v = torch.randn(S_und, num_kv_heads, head_dim, device=device, dtype=dtype)
    q_frames = [torch.randn(tokens_per_seg, num_heads, head_dim, device=device, dtype=dtype) for _ in range(num_frames)]
    k_frames = [
        torch.randn(tokens_per_seg, num_kv_heads, head_dim, device=device, dtype=dtype) for _ in range(num_frames)
    ]
    v_frames = [
        torch.randn(tokens_per_seg, num_kv_heads, head_dim, device=device, dtype=dtype) for _ in range(num_frames)
    ]

    # Candidate: full KVCacheTrainMemoryState plumbing.
    dual_caches = [DualKVCache(gen_cache_size=gen_cache_size)]

    for fi in range(num_frames):
        has_new_caption = fi == 0  # only segment 0 carries the caption.
        pack_q = _mk_factored_pack_with_caption(
            und_k, q_frames[fi], has_new_caption=has_new_caption, padded_causal_len=padded_causal_len, device=device
        )
        pack_k = _mk_factored_pack_with_caption(
            und_k, k_frames[fi], has_new_caption=has_new_caption, padded_causal_len=padded_causal_len, device=device
        )
        pack_v = _mk_factored_pack_with_caption(
            und_v, v_frames[fi], has_new_caption=has_new_caption, padded_causal_len=padded_causal_len, device=device
        )
        state = KVCacheTrainMemoryState(
            vision_token_shapes=[(T, H_p, W_p)],
            num_action_tokens_per_supertoken=tcf,
            null_action_supertokens=False,
            segment_idx=fi,
            dual_kv_cache=dual_caches,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )
        state.init(pack_q, device=device)
        mv = state.read_for_layer(0)

        rolling_mask = SplitInfo(
            split_lens=[padded_causal_len, tokens_per_seg],
            attn_modes=["causal", "full"],
            sample_lens=[padded_causal_len + tokens_per_seg],
            actual_len=padded_causal_len + tokens_per_seg,
        )
        rolling_out_pack = three_way_attention_with_kv_cache(
            pack_q, pack_k, pack_v, memory_value=mv, attention_meta=rolling_mask
        )
        rolling_gen_out = get_gen_seq(rolling_out_pack).unflatten(-1, (num_heads, head_dim))

        # write_for_layer stores current-frame K/V (and und K/V on segment 0)
        # back into the rolling DualKVCache.  The live gen K/V live in the
        # factored pack as full_only_seq [S, H_kv, D] and the live und K/V
        # as causal_seq [padded_causal_len, H_kv, D]; both must be reshaped
        # to [1, S, H_kv, D] for DualKVCache.store*.
        state.write_for_layer(
            0,
            (
                k_frames[fi].unsqueeze(0),
                v_frames[fi].unsqueeze(0),
                pack_k["causal_seq"].unsqueeze(0),
                pack_v["causal_seq"].unsqueeze(0),
            ),
        )

        # Golden: manual rolling window cut-off, feed eager AR attention.
        window_start = max(0, fi - (gen_cache_size - 1))
        hist_k = [k_frames[j].unsqueeze(0) for j in range(window_start, fi)]
        hist_v = [v_frames[j].unsqueeze(0) for j in range(window_start, fi)]
        gen_k_hist = torch.cat(hist_k, dim=1) if hist_k else None
        gen_v_hist = torch.cat(hist_v, dim=1) if hist_v else None
        eager_mv = ARMemoryValue(
            und_k_cached=und_k.unsqueeze(0),
            und_v_cached=und_v.unsqueeze(0),
            gen_k_hist=gen_k_hist,
            gen_v_hist=gen_v_hist,
            frame_idx=max(fi, 1),
            gen_len=tokens_per_seg,
        )
        eager_mask = SplitInfo(
            split_lens=[tokens_per_seg],
            attn_modes=["full"],
            sample_lens=[tokens_per_seg],
            actual_len=tokens_per_seg,
        )
        eager_out_pack, _ = attention_AR_gen_only(
            _mk_gen_only_pack(q_frames[fi], device),
            _mk_gen_only_pack(k_frames[fi], device),
            _mk_gen_only_pack(v_frames[fi], device),
            eager_mask,
            memory_value=eager_mv,
        )
        eager_gen_out = get_gen_seq(eager_out_pack).unflatten(-1, (num_heads, head_dim))

        torch.testing.assert_close(
            rolling_gen_out, eager_gen_out, rtol=1e-2, atol=1e-2, msg=f"seg {fi} rolling vs eager mismatch"
        )


@pytest.mark.L0
def test_rolling_kv_cache_memory_state_init_invariants():
    """``KVCacheTrainMemoryState.init()`` produces shape/dtype-stable
    flags and offsets across every segment index.

    These invariants are what makes the compile path recompile-free:

    - ``gen_ca_cached_kv_offsets`` is always ``shape=(2,), dtype=int32``
      (no ``Optional[Tensor]`` branching that would force Dynamo to
      specialize on the ``None`` vs ``Tensor`` case).
    - ``gen_ca_cached_kv_offsets[1] == max(real_gen_cache_len, 1)`` — the
      kernel-safety clamp that lets FA3's varlen path accept an empty
      cache at ``segment_idx=0`` without raising.
    - ``has_cached_gen`` is a scalar bool tensor mirroring ``segment_idx > 0``.
    - ``max_gen_cache_tokens`` is constant across segments (the whole
      point of ``fetch_kv_padded``).

    This is the pure-Python mirror of the compile single-graph invariant
    tested (CUDA-gated, L1) in
    ``test_three_way_attention_with_kv_cache_compiles_single_graph``.
    CPU-only, no CUDA dependency, always runs in L0 CI.
    """
    device = torch.device("cpu")
    dtype = torch.float32
    T, H_p, W_p, tcf = 1, 2, 2, 0
    S_super = tcf + H_p * W_p
    tokens_per_seg = T * S_super
    num_kv_heads = 4
    head_dim = 16
    S_und = 5
    padded_causal_len = S_und
    gen_cache_size = 3

    dual_caches = [DualKVCache(gen_cache_size=gen_cache_size)]

    expected_max_tokens = (gen_cache_size - 1) * tokens_per_seg
    first_max_tokens: int | None = None

    # Iterate segments 0 .. cache_size (inclusive) to cross the saturation
    # boundary.  Post-saturation, real_gen_cache_len saturates at
    # (cache_size - 1) * tokens_per_seg.
    for seg_idx in range(gen_cache_size + 1):
        has_new_caption = seg_idx == 0
        dummy_gen = torch.zeros(tokens_per_seg, num_kv_heads, head_dim, dtype=dtype)
        dummy_und = torch.zeros(S_und, num_kv_heads, head_dim, dtype=dtype)
        pack = _mk_factored_pack_with_caption(
            dummy_und,
            dummy_gen,
            has_new_caption=has_new_caption,
            padded_causal_len=padded_causal_len,
            device=device,
        )
        state = KVCacheTrainMemoryState(
            vision_token_shapes=[(T, H_p, W_p)],
            num_action_tokens_per_supertoken=tcf,
            null_action_supertokens=False,
            segment_idx=seg_idx,
            dual_kv_cache=dual_caches,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )
        state.init(pack, device=device)

        # gen_ca_cached_kv_offsets shape/dtype invariants.
        assert state.gen_ca_cached_kv_offsets is not None
        assert state.gen_ca_cached_kv_offsets.shape == (2,), (
            f"seg {seg_idx}: gen_ca_cached_kv_offsets.shape={state.gen_ca_cached_kv_offsets.shape}, expected (2,)"
        )
        assert state.gen_ca_cached_kv_offsets.dtype == torch.int32

        # max(real_gen_cache_len, 1) clamp.
        real_gen_cache_len = min(seg_idx, gen_cache_size - 1) * tokens_per_seg
        expected_kv_hi = max(real_gen_cache_len, 1)
        assert state.gen_ca_cached_kv_offsets[0].item() == 0
        assert state.gen_ca_cached_kv_offsets[1].item() == expected_kv_hi, (
            f"seg {seg_idx}: gen_ca_cached_kv_offsets[1]={state.gen_ca_cached_kv_offsets[1].item()}, "
            f"expected max(real={real_gen_cache_len}, 1)={expected_kv_hi}"
        )

        # has_cached_gen scalar bool.
        assert state.has_cached_gen is not None
        assert state.has_cached_gen.dtype == torch.bool
        assert state.has_cached_gen.shape == ()
        assert bool(state.has_cached_gen.item()) == (seg_idx > 0)

        # has_new_caption scalar bool mirrors python flag.
        assert state.has_new_caption is not None
        assert state.has_new_caption.dtype == torch.bool
        assert state.has_new_caption.shape == ()
        assert bool(state.has_new_caption.item()) == has_new_caption

        # und_kv_offsets / gen_q_offsets shape invariants.
        assert state.und_kv_offsets is not None
        assert state.und_kv_offsets.shape == (2,)
        assert state.und_kv_offsets.dtype == torch.int32
        assert state.gen_q_offsets is not None
        assert state.gen_q_offsets.shape == (2,)
        assert state.gen_q_offsets.dtype == torch.int32
        assert state.gen_q_offsets[1].item() == tokens_per_seg

        # max_gen_cache_tokens is a constant — same int every segment.
        if first_max_tokens is None:
            first_max_tokens = state.max_gen_cache_tokens
            assert first_max_tokens == expected_max_tokens, (
                f"max_gen_cache_tokens={first_max_tokens}, expected {expected_max_tokens}"
            )
        else:
            assert state.max_gen_cache_tokens == first_max_tokens, (
                f"seg {seg_idx}: max_gen_cache_tokens drifted from {first_max_tokens} to {state.max_gen_cache_tokens}"
            )


# ARMemoryState — static-shape (CG) vs dynamic-shape parity at read_for_layer


@pytest.mark.L0
def test_ar_memory_state_static_shape_init_and_read():
    """``ARMemoryState(for_cuda_graphs=True)`` must:

    1. Reject construction without the required CG params (``vision_token_shapes``,
       ``num_kv_heads``, ``head_dim``).
    2. Allocate ``cu_seqlens_q_t`` / ``cu_seqlens_kv_t`` as ``int32`` ``[2]``
       tensors after ``init`` (constant shape across frames is the prerequisite
       for a single CUDA-graph capture).
    3. Return an ``ARMemoryValue(for_cuda_graphs=True)`` with constant-shape
       ``gen_k_buf_full`` / ``gen_v_buf_full`` (``[1, max_gen_cache_tokens, H, D]``)
       across every frame.
    4. Carry the und K/V cached at frame 0 unchanged into frames 1+.
    5. Match the dynamic-shape flavor's ``frame_idx``, ``gen_len``, and
       ``und_k/v_cached`` fields exactly (they're shared by both branches).
    """
    B, H, D = 1, 4, 16
    S_und, S_gen = 6, 4
    num_layers = 2
    gen_cache_size = 4
    expected_max_gen_tokens = (gen_cache_size - 1) * S_gen  # = 12

    torch.manual_seed(123)
    und_k = torch.randn(B, S_und, H, D)
    und_v = torch.randn(B, S_und, H, D)

    caches = [DualKVCache(gen_cache_size=gen_cache_size) for _ in range(num_layers)]
    # Eager frame-0 prefill: populate und cache + frame 0 of gen cache.
    for c in caches:
        c.und_cache.store(und_k, und_v)
        c.gen_cache.store_kv(torch.randn(B, S_gen, H, D), torch.randn(B, S_gen, H, D), frame_idx=0)

    # Construction guard: must reject missing CG params.
    with pytest.raises(AssertionError, match=r"requires vision_token_shapes"):
        ARMemoryState(caches, frame_idx=1, for_cuda_graphs=True, num_kv_heads=H, head_dim=D)
    with pytest.raises(AssertionError, match=r"requires num_kv_heads"):
        ARMemoryState(
            caches,
            frame_idx=1,
            vision_token_shapes=[(1, 2, 2)],
            for_cuda_graphs=True,
            head_dim=D,
        )
    with pytest.raises(AssertionError, match=r"requires head_dim"):
        ARMemoryState(
            caches,
            frame_idx=1,
            vision_token_shapes=[(1, 2, 2)],
            for_cuda_graphs=True,
            num_kv_heads=H,
        )

    # Construct both flavors at the same frame
    # vision_token_shapes=(T=1, H_p=2, W_p=2), num_action_tokens=0
    # → S_super = 0 + 2*2 = 4 = S_gen, max_gen_tokens = (cache_size-1) * 1 * 4 = 12.
    state_static = ARMemoryState(
        caches,
        frame_idx=1,
        vision_token_shapes=[(1, 2, 2)],
        num_action_tokens_per_supertoken=0,
        for_cuda_graphs=True,
        num_kv_heads=H,
        head_dim=D,
    )
    state_dyn = ARMemoryState(caches, frame_idx=1)
    state_static.init({"_num_full_tokens": S_gen}, torch.device("cpu"))
    state_dyn.init({"_num_full_tokens": S_gen}, torch.device("cpu"))

    # cu_seqlens shape invariants (constant across frames is what enables CG).
    assert state_static._cu_seqlens_q_t is not None
    assert state_static._cu_seqlens_q_t.shape == (2,)
    assert state_static._cu_seqlens_q_t.dtype == torch.int32
    assert state_static._cu_seqlens_kv_t is not None
    assert state_static._cu_seqlens_kv_t.shape == (2,)
    assert state_static._cu_seqlens_kv_t.dtype == torch.int32
    # cu_seqlens_kv = [0, S_und + S_gen + real_gen_cache_len]; real=1*4 at frame 1.
    assert state_static._cu_seqlens_kv_t[1].item() == S_und + S_gen + 1 * S_gen
    # max_seqlen_KV = S_und + S_gen + max_gen_cache_tokens (constant).
    assert state_static._max_seqlen_KV == S_und + S_gen + expected_max_gen_tokens

    # read_for_layer: static-buffer shape is constant.
    mv_static = state_static.read_for_layer(0)
    assert isinstance(mv_static, ARMemoryValue)
    assert mv_static.for_cuda_graphs is True
    assert mv_static.gen_k_hist is None and mv_static.gen_v_hist is None
    assert mv_static.gen_k_buf_full is not None and mv_static.gen_v_buf_full is not None
    assert mv_static.gen_k_buf_full.shape == (B, expected_max_gen_tokens, H, D)
    assert mv_static.gen_v_buf_full.shape == (B, expected_max_gen_tokens, H, D)
    assert mv_static.real_gen_cache_len_t is not None
    assert mv_static.real_gen_cache_len_t.shape == (1,)
    assert mv_static.real_gen_cache_len_t.item() == 1 * S_gen  # 1 prior frame

    # und K/V cached at frame 0 carried unchanged into frame 1.
    assert mv_static.und_k_cached is not None and mv_static.und_v_cached is not None
    assert torch.equal(mv_static.und_k_cached, und_k)
    assert torch.equal(mv_static.und_v_cached, und_v)

    # Shared fields match dynamic flavor exactly.
    mv_dyn = state_dyn.read_for_layer(0)
    assert mv_dyn.for_cuda_graphs is False
    assert mv_static.frame_idx == mv_dyn.frame_idx == 1
    assert mv_static.gen_len == mv_dyn.gen_len == S_gen
    assert mv_dyn.und_k_cached is not None
    torch.testing.assert_close(mv_static.und_k_cached, mv_dyn.und_k_cached)
    torch.testing.assert_close(mv_static.und_v_cached, mv_dyn.und_v_cached)

    # Static buffer's real prefix must match dynamic gen_k_hist exactly.
    real_len = mv_static.real_gen_cache_len_t.item()
    assert mv_dyn.gen_k_hist is not None and mv_dyn.gen_v_hist is not None
    torch.testing.assert_close(mv_static.gen_k_buf_full[:, :real_len], mv_dyn.gen_k_hist)
    torch.testing.assert_close(mv_static.gen_v_buf_full[:, :real_len], mv_dyn.gen_v_hist)
    # The padded tail is intentionally unspecified: static AR attention uses
    # ``cu_seqlens_kv_t`` to exclude it from the varlen kernel.


@pytest.mark.L0
def test_ar_memory_state_static_shape_constant_across_frames():
    """The whole point of the static-shape flavor is shape constancy across
    frames: ``cu_seqlens_*`` and ``gen_k_buf_full`` shapes must not change as
    ``frame_idx`` advances; only the *values* in ``cu_seqlens_kv_t`` and
    ``real_gen_cache_len_t`` should update.

    If this test ever fails, ``torch.compile`` will recompile every frame
    and CUDA graphs will recapture every frame, defeating the whole refactor.
    """
    B, H, D = 1, 4, 16
    S_und, S_gen = 6, 4
    gen_cache_size = 4
    num_frames = 5  # 0..4, with rolling eviction starting at frame 4

    torch.manual_seed(7)
    und_k = torch.randn(B, S_und, H, D)
    und_v = torch.randn(B, S_und, H, D)

    caches = [DualKVCache(gen_cache_size=gen_cache_size)]
    caches[0].und_cache.store(und_k, und_v)
    for fi in range(num_frames - 1):
        caches[0].gen_cache.store_kv(torch.randn(B, S_gen, H, D), torch.randn(B, S_gen, H, D), frame_idx=fi)

    fixed_shapes: dict[str, tuple] = {}
    for fi in range(1, num_frames):
        state = ARMemoryState(
            caches,
            frame_idx=fi,
            vision_token_shapes=[(1, 2, 2)],
            num_action_tokens_per_supertoken=0,
            for_cuda_graphs=True,
            num_kv_heads=H,
            head_dim=D,
        )
        state.init({"_num_full_tokens": S_gen}, torch.device("cpu"))
        mv = state.read_for_layer(0)
        assert mv.gen_k_buf_full is not None
        shapes = {
            "gen_k_buf_full": tuple(mv.gen_k_buf_full.shape),
            "cu_seqlens_q_t": tuple(state._cu_seqlens_q_t.shape),  # type: ignore[union-attr]
            "cu_seqlens_kv_t": tuple(state._cu_seqlens_kv_t.shape),  # type: ignore[union-attr]
            "real_gen_cache_len_t": tuple(mv.real_gen_cache_len_t.shape),  # type: ignore[union-attr]
        }
        if not fixed_shapes:
            fixed_shapes = shapes
        else:
            assert shapes == fixed_shapes, f"frame {fi}: static shape drifted: {shapes} vs {fixed_shapes}"

        # max_seqlen_KV is a Python int but it must be constant too (it's
        # captured into the compiled graph as a literal).
        assert state._max_seqlen_KV == S_und + S_gen + (gen_cache_size - 1) * S_gen


@pytest.mark.L0
def test_ar_memory_state_can_skip_gen_cache_write() -> None:
    """Sampler denoise forwards should not store temporary current-frame K/V."""
    B, S_und, S_gen, H, D = 1, 6, 4, 2, 8
    cache = DualKVCache(gen_cache_size=4)
    und_k = torch.randn(B, S_und, H, D)  # [B,S_und,H,D]
    und_v = torch.randn(B, S_und, H, D)  # [B,S_und,H,D]
    gen0_k = torch.randn(B, S_gen, H, D)  # [B,S_gen,H,D]
    gen0_v = torch.randn(B, S_gen, H, D)  # [B,S_gen,H,D]
    gen4_k = torch.randn(B, S_gen, H, D)  # [B,S_gen,H,D]
    gen4_v = torch.randn(B, S_gen, H, D)  # [B,S_gen,H,D]

    cache.und_cache.store(und_k, und_v)
    cache.gen_cache.store_kv(gen0_k, gen0_v, frame_idx=0)
    # Pick frame_idx=4 so modulo indexing would overwrite physical slot 0.
    cached_gen0_k_ref = cache.gen_cache.k_cache[0]  # [B,S_gen,H,D] or None
    cached_gen0_v_ref = cache.gen_cache.v_cache[0]  # [B,S_gen,H,D] or None
    assert cached_gen0_k_ref is not None
    assert cached_gen0_v_ref is not None
    cached_gen0_k = cached_gen0_k_ref.clone()  # [B,S_gen,H,D]
    cached_gen0_v = cached_gen0_v_ref.clone()  # [B,S_gen,H,D]
    state = ARMemoryState([cache], frame_idx=4, write_gen_cache=False)
    state.init({"_num_full_tokens": S_gen}, torch.device("cpu"))
    state.write_for_layer(0, (gen4_k, gen4_v, und_k, und_v))

    assert cache.gen_cache.k_cache[0] is not None
    assert cache.gen_cache.v_cache[0] is not None
    # Slot 0 should still contain frame-0 K/V. If write_gen_cache=False were ignored,
    # frame_idx=4 would map to slot 0 and overwrite it with gen4 K/V.
    torch.testing.assert_close(cache.gen_cache.k_cache[0], cached_gen0_k)
    torch.testing.assert_close(cache.gen_cache.v_cache[0], cached_gen0_v)
    torch.testing.assert_close(cache.und_cache.k_und, und_k)
    torch.testing.assert_close(cache.und_cache.v_und, und_v)


@pytest.mark.L0
def test_ar_memory_state_slices_full_kv_to_local_heads_before_cache_write() -> None:
    """AR memory state stores only the local KV-head shard when configured."""
    B, S_und, S_gen, H, D = 1, 3, 4, 8, 5
    kv_head_shard_size = 4
    kv_head_shard_rank = 2
    local_heads = H // kv_head_shard_size
    h_start = kv_head_shard_rank * local_heads
    h_end = h_start + local_heads

    cache = DualKVCache(gen_cache_size=4)
    gen_k = torch.arange(B * S_gen * H * D, dtype=torch.float32).reshape(B, S_gen, H, D)  # [B,S_gen,H,D]
    gen_v = (gen_k + 10_000).clone()  # [B,S_gen,H,D]
    und_k = torch.arange(B * S_und * H * D, dtype=torch.float32).reshape(B, S_und, H, D)  # [B,S_und,H,D]
    und_v = (und_k + 20_000).clone()  # [B,S_und,H,D]

    state = ARMemoryState(
        [cache],
        frame_idx=0,
        num_kv_heads=H,
        head_dim=D,
        kv_head_shard_rank=kv_head_shard_rank,
        kv_head_shard_size=kv_head_shard_size,
    )
    state.init({"_num_full_tokens": S_gen}, torch.device("cpu"))
    state.write_for_layer(0, (gen_k, gen_v, und_k, und_v))

    expected_gen_k = gen_k[:, :, h_start:h_end, :]  # [B,S_gen,H_local,D]
    expected_gen_v = gen_v[:, :, h_start:h_end, :]  # [B,S_gen,H_local,D]
    expected_und_k = und_k[:, :, h_start:h_end, :]  # [B,S_und,H_local,D]
    expected_und_v = und_v[:, :, h_start:h_end, :]  # [B,S_und,H_local,D]

    # frame_idx=0 stores generated K/V in physical slot 0 of the circular gen cache.
    cached_gen_k = cache.gen_cache.k_cache[0]  # [B,S_gen,H_local,D] or None
    cached_gen_v = cache.gen_cache.v_cache[0]  # [B,S_gen,H_local,D] or None
    assert cached_gen_k is not None
    assert cached_gen_v is not None
    torch.testing.assert_close(cached_gen_k, expected_gen_k)
    torch.testing.assert_close(cached_gen_v, expected_gen_v)
    torch.testing.assert_close(cache.und_cache.k_und, expected_und_k)
    torch.testing.assert_close(cache.und_cache.v_und, expected_und_v)


@pytest.mark.L0
def test_ar_memory_state_local_kv_head_cache_requires_divisible_kv_heads() -> None:
    """Local KV-head cache storage requires num_kv_heads % shard_size == 0."""
    with pytest.raises(AssertionError, match=r"kv_head_shard_size\(3\) must divide num_kv_heads\(8\)"):
        ARMemoryState(
            [DualKVCache(gen_cache_size=4)],
            frame_idx=0,
            num_kv_heads=8,
            head_dim=5,
            kv_head_shard_size=3,
        )
