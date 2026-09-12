# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import pytest
import torch
from torch import nn

from cosmos_framework.configs.base.defaults.quantization import QuantizationConfig
from cosmos_framework.utils.generator import qdq_sim_edges as edges
from cosmos_framework.utils.generator.quantization import QdqSimLinear, apply_quantization_inplace, fake_quant_int8


def _restore_runtime_defaults() -> None:
    # ``reset_sim_edges`` only re-applies the ``configure_sim_edges`` arguments; the experimental attention options
    # (``attn_q_smoothing``, ``attn_qk_balance``, ``attn_qk_hadamard``, ``attn_outlier_frac``, ``attn_qk_block_tokens``,
    # gating fields, ...) would otherwise leak from one test into the next.
    edges._runtime.__dict__.update(edges.QdqSimEdgeRuntime().__dict__)


@pytest.fixture(autouse=True)
def _reset_edges():
    _restore_runtime_defaults()
    edges.reset_sim_edges()
    yield
    _restore_runtime_defaults()
    edges.reset_sim_edges()


def _sdpa(q, k, v):  # [B,S,H,D] layouts, GQA-aware, fp32 reference
    rep = q.shape[2] // k.shape[2]
    out = torch.nn.functional.scaled_dot_product_attention(
        q.float().permute(0, 2, 1, 3),
        k.float().permute(0, 2, 1, 3).repeat_interleave(rep, dim=1),
        v.float().permute(0, 2, 1, 3).repeat_interleave(rep, dim=1),
    )
    return out.permute(0, 2, 1, 3)


def test_hooks_are_identity_when_no_edge_is_enabled() -> None:
    x = torch.randn(4, 128)
    assert edges.fake_quant_edge(x, "residual") is x
    assert edges.fake_quant_edge(x, "gemm_out") is x
    k, v = torch.randn(1, 5, 2, 8), torch.randn(1, 5, 2, 8)
    k2, v2 = edges.quantize_cached_kv(k, v)
    assert k2 is k and v2 is v


def test_configure_validates_edges_and_group_size() -> None:
    with pytest.raises(ValueError, match="Unknown sim edges"):
        edges.configure_sim_edges(["nope"], method="int8_sim", group_size=64)
    with pytest.raises(ValueError, match="qdq_group_size > 0"):
        edges.configure_sim_edges(["residual"], method="int8_sim", group_size=0)
    with pytest.raises(ValueError, match="residual_bits"):
        edges.configure_sim_edges(["residual"], method="int8_sim", group_size=64, residual_bits=12)
    edges.configure_sim_edges(["residual", "attn_qkv"], method="int8_sim", group_size=64)
    assert edges.sim_edges_enabled() == frozenset({"residual", "attn_qkv"})


def test_residual_edge_group_and_bits() -> None:
    x = torch.randn(3, 256).to(torch.bfloat16)
    edges.configure_sim_edges(["residual"], method="int8_sim", group_size=64)
    torch.testing.assert_close(edges.fake_quant_edge(x, "residual").float(), fake_quant_int8(x, group_size=64).float())
    edges.configure_sim_edges(["residual"], method="int8_sim", group_size=64, residual_group_size=16)
    torch.testing.assert_close(edges.fake_quant_edge(x, "residual").float(), fake_quant_int8(x, group_size=16).float())
    edges.configure_sim_edges(["residual"], method="int8_sim", group_size=64, residual_bits=16)
    err16 = (edges.fake_quant_edge(x, "residual").float() - x.float()).abs().max().item()
    err8 = (fake_quant_int8(x, group_size=64).float() - x.float()).abs().max().item()
    assert err16 < err8 / 50


def test_fake_quant_uint8_grid_and_partial_group() -> None:
    p = torch.rand(2, 150)  # 150 = 2 full groups of 64 + a partial group of 22
    q = edges.fake_quant_uint8(p, group_size=64)
    assert q.shape == p.shape and q.min() >= 0
    for lo, hi in ((0, 64), (64, 128), (128, 150)):
        blk, qb = p[:, lo:hi], q[:, lo:hi]
        scale = blk.amax(dim=-1, keepdim=True) / 255.0
        codes = qb / scale
        assert torch.allclose(codes, torch.round(codes), atol=1e-4) and codes.max() <= 255.0 + 1e-4


def test_k_smoothing_leaves_attention_unchanged() -> None:
    torch.manual_seed(0)
    q, k, v = torch.randn(1, 37, 8, 32), torch.randn(1, 53, 2, 32) + 3.0, torch.randn(1, 53, 2, 32)
    k_s = edges._smooth_k(k)
    assert torch.allclose(k_s.mean(dim=1), torch.zeros(1, 2, 32), atol=1e-5)
    torch.testing.assert_close(_sdpa(q, k_s, v), _sdpa(q, k, v), atol=1e-5, rtol=1e-5)


def test_quant_k_rows_is_one_scale_per_token_head() -> None:
    edges.configure_sim_edges(["attn_qkv"], method="int8_sim", group_size=64)
    k = torch.randn(1, 9, 2, 128)
    kq = edges.quant_k_rows(k)
    # every (token, head) row lies on a single integer grid spanning the whole head_dim
    scale = k.abs().amax(dim=-1, keepdim=True) / 127.0
    codes = kq / scale
    assert torch.allclose(codes, torch.round(codes), atol=1e-4)
    torch.testing.assert_close(kq, fake_quant_int8(k, per_row=True))


def test_quant_v_channels_scale_is_constant_along_keys() -> None:
    edges.configure_sim_edges(["attn_pv"], method="int8_sim", group_size=64)
    v = torch.randn(1, 100, 2, 16) * torch.linspace(0.1, 10, 16)  # channel-wise outliers
    vq = edges.quant_v_channels(v, seq_dim=1)
    scale = v.abs().amax(dim=1, keepdim=True) / 127.0  # one scale per (head, channel), over all keys
    codes = vq / scale
    assert torch.allclose(codes, torch.round(codes), atol=1e-4)
    # per-(key block, channel) variant: each 64-key block has its own per-channel scale
    edges.configure_sim_edges(["attn_pv"], method="int8_sim", group_size=64, attn_v_block_size=64)
    vb = edges.quant_v_channels(v, seq_dim=1)
    for lo, hi in ((0, 64), (64, 100)):
        blk = v[:, lo:hi]
        codes = vb[:, lo:hi] / (blk.abs().amax(dim=1, keepdim=True) / 127.0)
        assert torch.allclose(codes, torch.round(codes), atol=1e-4)
    # finer blocks are never worse than whole-sequence scales
    assert (vb - v).norm() <= (vq - v).norm() + 1e-6


def test_reference_attention_matches_sdpa_without_quantization() -> None:
    torch.manual_seed(1)
    q, k, v = torch.randn(1, 37, 8, 32), torch.randn(1, 53, 2, 32), torch.randn(1, 53, 2, 32)
    torch.testing.assert_close(
        edges.reference_attention_qdq(q, k, v, quantize_pv=False), _sdpa(q, k, v), atol=1e-5, rtol=1e-5
    )


def test_reference_attention_pv_quantization_with_smoothing_is_small_perturbation() -> None:
    torch.manual_seed(2)
    edges.configure_sim_edges(["attn_pv"], method="int8_sim", group_size=64)
    q, k = torch.randn(1, 64, 4, 64), torch.randn(1, 130, 4, 64)
    v = torch.randn(1, 130, 4, 64) + 5.0 * torch.randn(1, 1, 4, 64)  # channel means far from zero
    exact = _sdpa(q, k, v)
    quant = edges.reference_attention_qdq(q, k, v, quantize_pv=True)
    rel = ((quant - exact).norm() / exact.norm()).item()
    assert 0.0 < rel < 0.03
    # smoothing helps when channel means are large
    edges.configure_sim_edges(["attn_pv"], method="int8_sim", group_size=64, attn_smoothing=False)
    rel_ns = ((edges.reference_attention_qdq(q, k, v, quantize_pv=True) - exact).norm() / exact.norm()).item()
    assert rel < rel_ns


def test_sim_attention_routes_by_edges() -> None:
    torch.manual_seed(3)
    q, k, v = torch.randn(1, 40, 4, 32), torch.randn(1, 70, 2, 32), torch.randn(1, 70, 2, 32)
    exact = _sdpa(q, k, v)
    edges.configure_sim_edges(["attn_qkv"], method="int8_sim", group_size=64)
    # attn_qkv only: quantized Q/K (with K smoothing) through the dense reference without P/V quant
    out = edges.reference_attention_qdq(
        edges.quant_k_rows(q), edges.quant_k_rows(edges._smooth_k(k)), v, quantize_pv=False
    )
    rel = ((out - exact).norm() / exact.norm()).item()
    assert 0.0 < rel < 0.05
    edges.configure_sim_edges(["attn_qkv", "attn_pv"], method="int8_sim", group_size=64)
    out2 = edges.sim_attention(q, k, v)
    rel2 = ((out2 - exact).norm() / exact.norm()).item()
    assert 0.0 < rel2 < 0.06


def test_cached_kv_quantization_layout() -> None:
    edges.configure_sim_edges(["und_kv"], method="int8_sim", group_size=64)
    k, v = torch.randn(1, 33, 2, 128), torch.randn(1, 33, 2, 128)
    kq, vq = edges.quantize_cached_kv(k, v)
    torch.testing.assert_close(kq, fake_quant_int8(k, per_row=True))  # per (token, head)
    codes = vq / (v.abs().amax(dim=1, keepdim=True) / 127.0)  # per (head, channel) over keys
    assert torch.allclose(codes, torch.round(codes), atol=1e-4)


def test_gemm_out_edge_quantizes_linear_outputs() -> None:
    torch.manual_seed(4)
    model = nn.Sequential(nn.Linear(64, 128, bias=False, dtype=torch.bfloat16))
    x = torch.randn(5, 64).to(torch.bfloat16)
    apply_quantization_inplace(
        model, QuantizationConfig(method="int8_sim", qdq_group_size=64, include_regex=["0"], sim_edges=["gemm_out"])
    )
    lin = model[0]
    assert isinstance(lin, QdqSimLinear) and lin.qdq_output
    dense = nn.functional.linear(fake_quant_int8(x, group_size=64), lin.weight)
    torch.testing.assert_close(lin(x).float(), fake_quant_int8(dense, group_size=64).float())


def test_attention_operand_formats() -> None:
    torch.manual_seed(5)
    v = torch.randn(1, 100, 2, 16) * torch.linspace(0.1, 10, 16)
    p = torch.softmax(torch.randn(3, 150), dim=-1)
    # fp8 V: per-channel scale, values on the E4M3 grid
    edges.configure_sim_edges(["attn_pv"], method="int8_sim", group_size=64, attn_v_format="fp8")
    vq = edges.quant_v_channels(v, seq_dim=1)
    scale = v.abs().amax(dim=1, keepdim=True) / 448.0
    torch.testing.assert_close(vq, ((v / scale).to(torch.float8_e4m3fn).float() * scale))
    # fp8 P: fixed scale 448
    edges.configure_sim_edges(["attn_pv"], method="int8_sim", group_size=64, attn_p_format="fp8")
    pq = edges.quant_probs(p)
    torch.testing.assert_close(pq, (p * 448.0).to(torch.float8_e4m3fn).float() / 448.0)
    # none keeps the tensors
    edges.configure_sim_edges(["attn_pv"], method="int8_sim", group_size=64, attn_v_format="none", attn_p_format="none")
    assert edges.quant_v_channels(v, seq_dim=1) is v and edges.quant_probs(p) is p
    with pytest.raises(ValueError, match="attn_v_format"):
        edges.configure_sim_edges(["attn_pv"], method="int8_sim", group_size=64, attn_v_format="int4")


def test_fp16_accumulated_pv_is_close_to_fp32() -> None:
    torch.manual_seed(6)
    probs = torch.softmax(torch.randn(1, 2, 8, 200), dim=-1)
    v = torch.randn(1, 2, 200, 32)
    exact = torch.matmul(probs, v)
    approx = edges._pv_fp16_accumulate(probs, v, 16)
    rel = ((approx - exact).norm() / exact.norm()).item()
    assert 0.0 < rel < 2e-3
    edges.configure_sim_edges(
        ["attn_pv"], method="int8_sim", group_size=64, attn_v_format="none", attn_p_format="none", attn_pv_accum="fp16"
    )
    q, k, vv = torch.randn(1, 40, 4, 32), torch.randn(1, 70, 2, 32), torch.randn(1, 70, 2, 32)
    out = edges.sim_attention(q, k, vv)
    rel2 = ((out - _sdpa(q, k, vv)).norm() / _sdpa(q, k, vv).norm()).item()
    assert 0.0 < rel2 < 2e-3


def test_attn_k_scope_splits_text_and_gen_keys() -> None:
    torch.manual_seed(5)
    q, k = torch.randn(1, 12, 4, 32), torch.randn(1, 30, 2, 32)
    mask = torch.zeros(30, dtype=torch.bool)
    mask[:10] = True  # first 10 keys are text
    smoothed = edges._smooth_k(k)
    edges.configure_sim_edges(["attn_qkv"], method="int8_sim", group_size=64, attn_k_scope="gen")
    q_out, k_out = edges._quant_qk_scoped(q, k, mask)
    torch.testing.assert_close(q_out, edges.quant_k_rows(q))
    torch.testing.assert_close(k_out[:, :10], smoothed[:, :10])  # text keys: shifted, not quantized
    torch.testing.assert_close(k_out[:, 10:], edges.quant_k_rows(smoothed)[:, 10:])
    assert not torch.equal(k_out[:, 10:], smoothed[:, 10:])
    edges.configure_sim_edges(["attn_qkv"], method="int8_sim", group_size=64, attn_k_scope="und")
    q_out, k_out = edges._quant_qk_scoped(q, k, mask)
    assert q_out is q  # Q stays bf16 when only the text keys are quantized
    torch.testing.assert_close(k_out[:, :10], edges.quant_k_rows(smoothed)[:, :10])
    torch.testing.assert_close(k_out[:, 10:], smoothed[:, 10:])
    edges.configure_sim_edges(["attn_qkv"], method="int8_sim", group_size=64, attn_k_scope="all")
    q_out, k_out = edges._quant_qk_scoped(q, k, None)
    torch.testing.assert_close(k_out, edges.quant_k_rows(smoothed))
    edges.configure_sim_edges(["attn_qkv"], method="int8_sim", group_size=64, attn_k_scope="gen")
    with pytest.raises(ValueError, match="needs und_key_mask"):
        edges._quant_qk_scoped(q, k, None)
    with pytest.raises(ValueError, match="must be bool"):
        edges._quant_qk_scoped(q, k, mask[:5])
    with pytest.raises(ValueError, match="attn_k_scope"):
        edges.configure_sim_edges(["attn_qkv"], method="int8_sim", group_size=64, attn_k_scope="text")


def test_sim_attention_accepts_und_key_mask_with_scope_gen() -> None:
    torch.manual_seed(6)
    q, k, v = torch.randn(1, 40, 4, 32), torch.randn(1, 70, 2, 32), torch.randn(1, 70, 2, 32)
    mask = torch.zeros(70, dtype=torch.bool)
    mask[:20] = True
    exact = _sdpa(q, k, v)
    edges.configure_sim_edges(
        ["attn_qkv", "attn_pv"],
        method="int8_sim",
        group_size=64,
        attn_v_format="none",
        attn_p_format="none",
        attn_k_scope="gen",
    )
    out = edges.sim_attention(q, k, v, und_key_mask=mask)
    rel = ((out - exact).norm() / exact.norm()).item()
    assert 0.0 < rel < 0.05
    # text keys kept exact: quantizing them too (scope all) must move the output further
    edges.configure_sim_edges(
        ["attn_qkv", "attn_pv"],
        method="int8_sim",
        group_size=64,
        attn_v_format="none",
        attn_p_format="none",
        attn_k_scope="all",
    )
    out_all = edges.sim_attention(q, k, v, und_key_mask=mask)
    assert not torch.equal(out, out_all)


def _mask(n, n_text):
    m = torch.zeros(n, dtype=torch.bool)
    m[:n_text] = True
    return m


def test_outlier_keys_are_kept_bf16_for_k_and_v() -> None:
    torch.manual_seed(7)
    q, k, v = torch.randn(1, 16, 4, 32), torch.randn(1, 64, 2, 32), torch.randn(1, 64, 2, 32)
    k[0, 40, 1, 3] = 40.0  # one outlying gen key in K
    v[0, 50, 0, 5] = 60.0  # one outlying gen key in V
    edges.configure_sim_edges(["attn_qkv", "attn_pv"], method="int8_sim", group_size=64, attn_k_scope="gen")
    edges._runtime.attn_outlier_frac = 2 / 54  # 54 gen keys -> top 2
    k_s = edges._smooth_k(k)
    keep = edges._bf16_key_mask(k_s, v, _mask(64, 10))
    assert keep[:10].all() and keep[40] and keep[50] and int(keep.sum()) == 12
    q_out, k_out, _ = edges._quant_qk(q, k, keep, _mask(64, 10))
    torch.testing.assert_close(k_out[:, keep], k_s[:, keep])  # bf16 keys only shifted, never rounded
    assert not torch.equal(k_out[:, ~keep], k_s[:, ~keep])
    v_used, v_mean = edges._quant_v(v, keep)
    v_shift = (v.float() - v_mean).to(v.dtype)
    torch.testing.assert_close(v_used[:, keep], v_shift[:, keep])
    # the per-channel scale is set by the quantized keys only: without the outlier the codes stay fine
    codes = v_used[:, ~keep] / (v_shift[:, ~keep].abs().amax(dim=1, keepdim=True) / 127.0)
    assert torch.allclose(codes, torch.round(codes), atol=1e-3)


def test_q_smoothing_adds_mean_back_and_reduces_error() -> None:
    torch.manual_seed(8)
    q = torch.randn(1, 200, 4, 32) + torch.linspace(-3, 3, 32)  # strong per-channel offset
    k = torch.randn(1, 50, 2, 32)
    edges.configure_sim_edges(["attn_qkv"], method="int8_sim", group_size=64)
    q_plain, _ = edges._quant_qk_scoped(q, k, None)
    edges._runtime.attn_q_smoothing = True
    q_smooth, _ = edges._quant_qk_scoped(q, k, None)
    err_plain = ((q_plain - q).norm() / q.norm()).item()
    err_smooth = ((q_smooth - q).norm() / q.norm()).item()
    assert err_smooth < err_plain


def test_qk_block_scales_share_one_scale_per_block_and_head() -> None:
    torch.manual_seed(9)
    k = torch.randn(1, 100, 2, 16)  # 100 tokens -> blocks of 64 + 36
    edges.configure_sim_edges(["attn_qkv"], method="int8_sim", group_size=64, attn_smoothing=False)
    edges._runtime.attn_qk_block_tokens = 64
    kq = edges._quant_rows_or_blocks(k)
    for lo, hi in ((0, 64), (64, 100)):
        blk = k[:, lo:hi]
        scale = blk.abs().amax(dim=(1, 3), keepdim=True) / 127.0  # per head over (tokens, D)
        codes = kq[:, lo:hi] / scale
        assert torch.allclose(codes, torch.round(codes), atol=1e-3)


def test_fused_v_path_matches_dense_reference() -> None:
    torch.manual_seed(10)
    q, k, v = torch.randn(1, 24, 4, 32), torch.randn(1, 40, 2, 32), torch.randn(1, 40, 2, 32)
    edges.configure_sim_edges(["attn_pv"], method="int8_sim", group_size=64, attn_p_format="none")
    dense = edges.reference_attention_qdq(q, k, v, quantize_pv=True)
    v_used, v_mean = edges._quant_v(v, None)
    fused = _sdpa(q, k, v_used) + v_mean.repeat_interleave(2, dim=2)
    torch.testing.assert_close(fused.float(), dense.float(), atol=2e-4, rtol=1e-4)


def test_step_and_layer_gating_counters() -> None:
    edges.configure_sim_edges(["attn_qkv"], method="int8_sim", group_size=64)
    edges._runtime.attn_bf16_first_steps = 2
    edges._runtime.attn_bf16_layers = (3,)
    edges._runtime.attn_layers_per_forward = 4
    edges.set_sampler_step(1)
    assert edges._state.step == 1 and edges._state.call_idx == 0
    edges.set_sampler_step(5)
    assert edges._state.step == 5


def test_parse_attn_opts() -> None:
    opts = edges._parse_attn_opts(
        "outlier_frac=0.01, q_smoothing=1,qk_block=128,bf16_first_steps=6,bf16_layers=0-1-35,err_log=/tmp/x.csv"
    )
    assert opts == {
        "attn_outlier_frac": 0.01,
        "attn_q_smoothing": True,
        "attn_qk_block_tokens": 128,
        "attn_bf16_first_steps": 6,
        "attn_bf16_layers": (0, 1, 35),
        "attn_err_log": "/tmp/x.csv",
    }
    with pytest.raises(ValueError):
        edges._parse_attn_opts("nope=1")


def _logits(q, k):  # [B,S_q,H,D] x [B,S_kv,H_kv,D] -> softmax over keys, GQA-aware
    rep = q.shape[2] // k.shape[2]
    kk = k.float().repeat_interleave(rep, dim=2)
    return torch.softmax(torch.einsum("bqhd,bkhd->bhqk", q.float(), kk) / q.shape[-1] ** 0.5, dim=-1)


def test_qk_transforms_are_exact_without_rounding_and_reduce_error_with_it() -> None:
    torch.manual_seed(11)
    q = torch.randn(1, 64, 4, 32)
    k = torch.randn(1, 96, 2, 32)
    q[..., 5] *= 12.0  # outlier channels of different magnitude in Q and K
    k[..., 7] *= 9.0
    ref = _logits(q, k)
    edges.configure_sim_edges(["attn_qkv"], method="int8_sim", group_size=64)
    edges._runtime.attn_q_smoothing = True
    # exactness: transforms alone (no rounding) leave softmax unchanged
    q_c, q_mean = edges._smooth_q(q)
    k_c = edges._smooth_k(k)
    for bal, had in ((True, False), (False, True), (True, True)):
        edges._runtime.attn_qk_balance, edges._runtime.attn_qk_hadamard = bal, had
        t_q, t_k = edges._qk_transforms(q_c, k_c)
        out = _logits(t_q(q_c) + t_q(q_mean), t_k(k_c))
        torch.testing.assert_close(out, ref, atol=2e-5, rtol=1e-4)

    # with rounding: error must not grow, and Hadamard+balance should beat plain per-token INT8
    def err(bal, had):
        edges._runtime.attn_qk_balance, edges._runtime.attn_qk_hadamard = bal, had
        qo, ko, _ = edges._quant_qk(q, k, torch.zeros(96, dtype=torch.bool), None)
        return ((_logits(qo, ko) - ref).norm() / ref.norm()).item()

    e_plain, e_both = err(False, False), err(True, True)
    assert e_both < e_plain, (e_plain, e_both)


def test_hadamard_is_orthonormal() -> None:
    h = edges._hadamard(128, "cpu", torch.float64)
    torch.testing.assert_close(h @ h.T, torch.eye(128, dtype=torch.float64))


# ----------------------------------------------------------------------------- SageAttention-v1 layout + exact pre-transforms
def _fwht(x: torch.Tensor) -> torch.Tensor:
    """In-place style fast Walsh-Hadamard transform along the last dim (log2(n) butterfly stages), the
    form a kernel would run on a register tile. Sylvester ordering, unnormalized."""
    n = x.shape[-1]
    y = x.clone()
    h = 1
    while h < n:
        y = y.reshape(*x.shape[:-1], n // (2 * h), 2, h)
        a, b = y[..., 0, :], y[..., 1, :]
        y = torch.stack((a + b, a - b), dim=-2).reshape(*x.shape[:-1], n)
        h *= 2
    return y


def test_hadamard_matches_seven_stage_butterfly_and_normalization_is_a_power_of_two() -> None:
    torch.manual_seed(12)
    x = torch.randn(3, 5, 128, dtype=torch.float64)
    h = edges._hadamard(128, "cpu", torch.float64)
    torch.testing.assert_close(x @ h, _fwht(x) / 128**0.5)
    # applying the *normalized* rotation to both Q and K is exact; the unnormalized butterflies on both
    # sides differ by exactly 1/128, a power of two a kernel folds into the softmax scale without rounding
    q, k = torch.randn(7, 128, dtype=torch.float64), torch.randn(9, 128, dtype=torch.float64)
    torch.testing.assert_close((q @ h) @ (k @ h).T, q @ k.T)
    torch.testing.assert_close(_fwht(q) @ _fwht(k).T / 128, q @ k.T)
    # the fp32 copy used at runtime is orthonormal to fp32 precision and cached per (n, device, dtype)
    h32 = edges._hadamard(128, "cpu", torch.float32)
    torch.testing.assert_close(h32 @ h32.T, torch.eye(128), atol=1e-6, rtol=0)
    assert edges._hadamard(128, "cpu", torch.float32) is h32
    with pytest.raises(ValueError, match="power-of-two"):
        edges._hadamard(96, "cpu", torch.float32)


def test_qk_balance_uses_kv_head_major_gqa_mapping() -> None:
    """q heads [rep*g, rep*g+rep) share kv-head g's balance vector (the layout of ``view(-1, num_heads, head_dim)``
    and of the backends' ``repeat_interleave(rep, dim=head)``); after balancing, the per-channel amax of every GQA
    group's queries equals its kv-head's key amax (alpha = 0.5)."""
    torch.manual_seed(13)
    b, s_q, s_kv, heads, kv_heads, d = 2, 40, 64, 8, 2, 32
    rep = heads // kv_heads
    q = torch.randn(b, s_q, heads, d)
    k = torch.randn(b, s_kv, kv_heads, d)
    q[:, :, 0:rep, 3] *= 30.0  # outlier channel only in the q heads of kv-group 0
    k[:, :, 1, 9] *= 25.0  # outlier channel only in kv-head 1
    edges.configure_sim_edges(["attn_qkv"], method="int8_sim", group_size=64)
    edges._runtime.attn_qk_balance = True
    t_q, t_k = edges._qk_transforms(q, k)
    fac_q = t_q(torch.ones(1, 1, heads, d))  # the per-(q head, channel) factor actually applied
    fac_k = 1.0 / t_k(torch.ones(1, 1, kv_heads, d))
    for g in range(kv_heads):
        for r in range(rep):
            torch.testing.assert_close(fac_q[0, 0, g * rep + r], fac_k[0, 0, g])
    aq = t_q(q).abs().reshape(b, s_q, kv_heads, rep, d).amax(dim=(0, 1, 3))
    ak = t_k(k).abs().amax(dim=(0, 1))
    torch.testing.assert_close(aq, ak, rtol=1e-4, atol=1e-5)
    # sanity: a rep-major tiling of the same factors would not balance the groups
    s_bad = fac_k[0, 0].repeat(rep, 1).view(1, 1, heads, d)
    aq_bad = (q * s_bad).abs().reshape(b, s_q, kv_heads, rep, d).amax(dim=(0, 1, 3))
    assert not torch.allclose(aq_bad, ak, rtol=1e-2)


def test_quant_qk_pipeline_is_exact_when_rounding_is_disabled(monkeypatch) -> None:
    """The whole ``_quant_qk`` chain -- K mean, Q mean, balance, Hadamard, bf16-key split, q̄ add-back -- with the
    INT8 rounding replaced by the identity must reproduce softmax(QKᵀ) exactly (GQA rep 4, outlier channels,
    text + outlier keys kept unrounded)."""
    torch.manual_seed(14)
    b, s_q, s_kv, heads, kv_heads, d = 1, 48, 80, 8, 2, 64
    q = torch.randn(b, s_q, heads, d) + torch.linspace(-3, 3, d)
    k = torch.randn(b, s_kv, kv_heads, d) + 2.0 * torch.randn(1, 1, kv_heads, d)
    q[..., 5] *= 15.0
    k[..., 7] *= 12.0
    k[0, 60, 1, 2] = 50.0  # an outlying gen key
    mask = _mask(s_kv, 16)
    ref = _logits(q, k)
    edges.configure_sim_edges(["attn_qkv"], method="int8_sim", group_size=64, attn_k_scope="gen")
    edges._runtime.attn_q_smoothing = True
    edges._runtime.attn_qk_balance = True
    edges._runtime.attn_qk_hadamard = True
    edges._runtime.attn_outlier_frac = 1 / 64
    monkeypatch.setattr(edges, "_quant_rows_or_blocks", lambda t: t)
    keep = edges._bf16_key_mask(edges._smooth_k(k), k, mask)
    assert keep[:16].all() and keep[60]
    q_out, k_out, k_t = edges._quant_qk(q, k, keep, mask)
    torch.testing.assert_close(_logits(q_out, k_out), ref, atol=1e-5, rtol=1e-5)
    # every key -- rounded or not -- carries the same shift and transforms, so the bf16 segment is consistent
    torch.testing.assert_close(k_out, k_t)
    # ... and the transforms are exact for *any* shared shift / balance vector, not only the data statistics:
    # a kernel may use stale (previous step) statistics without losing exactness
    c = torch.randn(1, 1, kv_heads, d)
    s = torch.rand(1, 1, kv_heads, d) + 0.5
    torch.testing.assert_close(
        _logits(q * s.repeat_interleave(heads // kv_heads, dim=2), (k - c) / s), ref, atol=1e-5, rtol=1e-5
    )


def test_fake_quant_qk_matches_int8_kernel_with_per_token_scales_on_s() -> None:
    """A real kernel forms S from INT8 codes (int32 exact) and applies s_q[i]·s_k[j] to the tile. The simulator
    instead feeds code·scale back through a floating-point kernel; with fp32 operands that is identical to
    fp32 precision, with bf16 operands the extra re-rounding is a small fraction of the INT8 error."""
    torch.manual_seed(15)
    b, s_q, s_kv, heads, kv_heads, d = 1, 64, 96, 8, 2, 128
    rep = heads // kv_heads
    q = torch.randn(b, s_q, heads, d) + torch.linspace(-1, 1, d)
    k = torch.randn(b, s_kv, kv_heads, d) + torch.randn(1, 1, kv_heads, d)
    q[..., 5] *= 10.0
    k[..., 7] *= 8.0
    edges.configure_sim_edges(["attn_qkv"], method="int8_sim", group_size=64, attn_smoothing=False)

    def kernel_logits(qq, kk):  # int8 codes -> exact int32 dot -> fp32 scale product
        def codes(x):
            x32 = x.float()
            scale = x32.abs().amax(dim=-1, keepdim=True) / 127.0
            return torch.round(x32 / scale).clamp_(-127, 127), scale

        qc, qs = codes(qq)
        kc, ks = codes(kk)
        assert torch.equal(qc, qc.round()) and qc.abs().max() <= 127 and kc.abs().max() <= 127
        s_int = torch.einsum("bqhd,bkhd->bhqk", qc.double(), kc.double().repeat_interleave(rep, dim=2))
        assert s_int.abs().max() < 2**24  # 127*127*128 < 2^24: exact in fp32 as well as int32
        return (s_int * qs.double().permute(0, 2, 1, 3) * ks.double().repeat_interleave(rep, dim=2).permute(0, 2, 3, 1)).float()

    def sim_logits(qq, kk):
        return torch.einsum("bqhd,bkhd->bhqk", qq.float(), kk.float().repeat_interleave(rep, dim=2))

    exact = sim_logits(q, k)
    s_kernel = kernel_logits(q, k)
    q32, k32, _ = edges._quant_qk(q, k, torch.zeros(s_kv, dtype=torch.bool), None)
    torch.testing.assert_close(sim_logits(q32, k32), s_kernel, atol=1e-3, rtol=1e-5)  # fp32 operands: identical
    qb, kb = q.to(torch.bfloat16), k.to(torch.bfloat16)
    s_kernel_b = kernel_logits(qb, kb)
    q16, k16, _ = edges._quant_qk(qb, kb, torch.zeros(s_kv, dtype=torch.bool), None)
    assert q16.dtype == torch.bfloat16 and k16.dtype == torch.bfloat16
    # bf16(code * scale) moves a value by at most half a bf16 ulp: < 0.5 of the INT8 step for |code| <= 127
    scale_q = qb.float().abs().amax(dim=-1, keepdim=True) / 127.0
    codes_q = q16.float() / scale_q
    assert ((codes_q - codes_q.round()).abs() < 0.5).all()
    err_int8 = (s_kernel_b - sim_logits(qb, kb)).pow(2).mean().sqrt()
    err_bf16_dq = (sim_logits(q16, k16) - s_kernel_b).pow(2).mean().sqrt()
    # (adds in quadrature: a 0.3 ratio inflates the simulated error by < 5 %; measured ~0.1 on this data)
    assert err_bf16_dq < 0.3 * err_int8, (err_bf16_dq.item(), err_int8.item())


def test_q_mean_add_back_in_bf16_is_conservative_relative_to_an_fp32_bias() -> None:
    """The simulator adds T_q(q̄) to the dequantized bf16 Q (one more bf16 rounding); a kernel keeps q̄·kⱼ as an
    fp32 per-key bias on the S tile. The simulator's extra error must stay well below the INT8 error."""
    torch.manual_seed(16)
    q = (torch.randn(1, 128, 8, 64) + 4.0 * torch.linspace(-1, 1, 64)).to(torch.bfloat16)
    k = torch.randn(1, 96, 2, 64).to(torch.bfloat16)
    edges.configure_sim_edges(["attn_qkv"], method="int8_sim", group_size=64)
    edges._runtime.attn_q_smoothing = True
    q_out, k_out, _ = edges._quant_qk(q, k, torch.zeros(96, dtype=torch.bool), None)
    q_c, q_mean = edges._smooth_q(q)
    q_q = edges.quant_k_rows(q_c)
    kk = k_out.float().repeat_interleave(4, dim=2)
    s_sim = torch.einsum("bqhd,bkhd->bhqk", q_out.float(), kk)
    s_kernel = torch.einsum("bqhd,bkhd->bhqk", q_q.float(), kk) + torch.einsum(
        "bqhd,bkhd->bhqk", q_mean.float().expand_as(q), kk
    )
    s_exact = torch.einsum("bqhd,bkhd->bhqk", q.float(), k.float().repeat_interleave(4, dim=2))
    err_sim_vs_kernel = (s_sim - s_kernel).pow(2).mean().sqrt()
    err_int8 = (s_kernel - s_exact).pow(2).mean().sqrt()
    # the extra rounding is the bf16 representation noise of Q itself (the bf16 baseline carries the same);
    # it stays a fraction of the INT8 error and adds in quadrature (measured ~0.2-0.5 with |q̄| ~ amax(q - q̄))
    assert err_sim_vs_kernel < 0.6 * err_int8, (err_sim_vs_kernel.item(), err_int8.item())
    # and the add-back makes the simulator at least as accurate as plain per-token INT8 on this offset data
    edges._runtime.attn_q_smoothing = False
    q_plain, k_plain, _ = edges._quant_qk(q, k, torch.zeros(96, dtype=torch.bool), None)
    s_plain = torch.einsum("bqhd,bkhd->bhqk", q_plain.float(), k_plain.float().repeat_interleave(4, dim=2))
    assert (s_sim - s_exact).norm() < (s_plain - s_exact).norm()


def test_k_mean_shift_is_shared_by_bf16_and_int8_keys_under_scope_gen() -> None:
    torch.manual_seed(17)
    q, k = torch.randn(1, 12, 4, 32), torch.randn(1, 30, 2, 32) + 3.0
    mask = _mask(30, 10)
    edges.configure_sim_edges(["attn_qkv"], method="int8_sim", group_size=64, attn_k_scope="gen")
    edges._runtime.attn_qk_hadamard = True
    keep = edges._bf16_key_mask(edges._smooth_k(k), k, mask)
    q_out, k_out, k_t = edges._quant_qk(q, k, keep, mask)
    h = edges._hadamard(32, "cpu", torch.float32)
    # the mean is taken over *all* keys (text + gen) and the same shift + rotation reaches the unrounded text keys
    expected_text = (k - k.mean(dim=1, keepdim=True))[:, :10] @ h
    torch.testing.assert_close(k_out[:, :10], expected_text, atol=1e-5, rtol=1e-5)
    # rotating back the unrounded text keys recovers the shifted originals exactly (H is orthonormal)
    torch.testing.assert_close(k_out[:, :10] @ h.T, (k - k.mean(dim=1, keepdim=True))[:, :10], atol=1e-5, rtol=1e-5)
