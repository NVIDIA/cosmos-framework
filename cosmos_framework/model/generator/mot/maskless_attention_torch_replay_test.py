# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""The PyTorch gapped-replay forward (GPUs without FlashAttention-4 kernels) against a per-segment reference."""

import pytest
import torch

from cosmos_framework.model.generator.mot import maskless_attention as m

# (q start, q used), (k start, k used): gaps between segments, a segment with no keys,
# and rows past each segment's used length that the launch must leave untouched.
_SEGMENTS = [((0, 290), (0, 900)), ((300, 200), (900, 1200)), ((500, 40), (2100, 0)), ((550, 350), (2100, 1500))]
_Q, _K = 950, 3600


def _gather(device: torch.device) -> m._VarlenGather:
    return m._VarlenGather(
        [q[0] for q, _ in _SEGMENTS],
        [q[1] for q, _ in _SEGMENTS],
        [k[0] for _, k in _SEGMENTS],
        [k[1] for _, k in _SEGMENTS],
        device,
    )


def _reference(q, k, v, scale):  # [Q,H,D], [K,H,D], [K,H,Dv] -> rows -> (out [n,H,Dv], lse [H,n])
    result = {}
    for (qs, qn), (ks, kn) in _SEGMENTS:
        if kn == 0:
            result[(qs, qn)] = None
            continue
        scores = torch.einsum("qhd,khd->hqk", q[qs : qs + qn].double(), k[ks : ks + kn].double()) * scale
        lse = torch.logsumexp(scores, dim=-1)
        out = torch.einsum("hqk,khd->qhd", torch.exp(scores - lse[..., None]), v[ks : ks + kn].double())
        result[(qs, qn)] = (out, lse)
    return result


def _check(q, k, v, *, out_tol: float, lse_tol: float) -> None:
    scale = q.shape[-1] ** -0.5
    gather = _gather(q.device)
    out = q.new_full((_Q, q.shape[1], v.shape[-1]), 7.0)
    lse = q.new_full((q.shape[1], _Q), 7.0, dtype=torch.float32)
    m._torch_varlen_fwd(q, k, v, gather, scale, out, lse)
    reference = _reference(q, k, v, scale)
    written = torch.zeros(_Q, dtype=torch.bool)
    for (qs, qn), expected in reference.items():
        written[qs : qs + qn] = True
        if expected is None:  # a segment with no keys: zero output, -inf LSE
            assert (out[qs : qs + qn] == 0).all()
            assert torch.isneginf(lse[:, qs : qs + qn]).all()
            continue
        torch.testing.assert_close(out[qs : qs + qn].double(), expected[0], atol=out_tol, rtol=out_tol)
        torch.testing.assert_close(lse[:, qs : qs + qn].double(), expected[1], atol=lse_tol, rtol=lse_tol)
    assert (out[~written.to(out.device)] == 7.0).all()  # rows no segment uses are left as they are
    assert (lse[:, ~written.to(lse.device)] == 7.0).all()


@pytest.mark.L0
@pytest.mark.CPU
def test_gather_packs_used_rows() -> None:
    gather = _gather(torch.device("cpu"))
    assert gather.lengths == [(290, 900), (200, 1200), (350, 1500)]
    assert gather.cu_q.tolist() == [0, 290, 490, 840]
    assert gather.cu_k.tolist() == [0, 900, 2100, 3600]
    assert (gather.max_q, gather.max_k) == (350, 1500)
    assert not isinstance(gather.q, slice)  # q has gaps
    assert gather.k == slice(0, 3600)  # k does not
    assert gather.empty_q == slice(500, 540)  # the segment with no keys


@pytest.mark.L0
@pytest.mark.CPU
def test_fp32_path_matches_reference() -> None:
    torch.manual_seed(0)
    q, k, v = torch.randn(_Q, 4, 64), torch.randn(_K, 4, 64), torch.randn(_K, 4, 64)
    _check(q, k, v, out_tol=1e-4, lse_tol=1e-4)


@pytest.mark.L0
@pytest.mark.CPU
def test_fp32_path_small_score_budget_matches_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    # One query row per block: the blocks a byte budget picks must tile every segment exactly.
    monkeypatch.setattr(m, "_MATMUL_SCORE_BYTES", 1)
    torch.manual_seed(0)
    q, k, v = torch.randn(_Q, 2, 32), torch.randn(_K, 2, 32), torch.randn(_K, 2, 32)
    _check(q, k, v, out_tol=1e-4, lse_tol=1e-4)


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    ("capability", "torch_replay", "admitted"),
    [
        ((9, 0), False, True),
        ((10, 0), False, True),
        ((12, 0), True, True),
        ((8, 6), False, False),
        ((11, 0), False, False),
    ],
)
def test_device_gate(monkeypatch: pytest.MonkeyPatch, capability, torch_replay: bool, admitted: bool) -> None:
    monkeypatch.delenv("COSMOS_GAPPED_REPLAY_TORCH", raising=False)
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: capability)
    monkeypatch.setattr(m, "_flash_attn_interface", lambda: None)
    device = torch.device("cuda")
    assert m._torch_replay(device) is torch_replay
    assert (m.gapped_replay_unavailable_reason(device) is None) is admitted
    monkeypatch.setenv("COSMOS_GAPPED_REPLAY_TORCH", "1")
    assert m._torch_replay(device)
    assert m.gapped_replay_unavailable_reason(device) is None


@pytest.mark.L0
@pytest.mark.GPU
@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlashAttention-2 needs CUDA")
def test_flash_attention_2_path_matches_reference() -> None:
    torch.manual_seed(0)
    q, k, v = (torch.randn(n, 8, 128, device="cuda", dtype=torch.bfloat16) for n in (_Q, _K, _K))
    _check(q, k, v, out_tol=2e-2, lse_tol=1e-3)
