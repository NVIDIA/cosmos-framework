# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Manual SM120 forward regression; run normally and under Compute Sanitizer.

These small synthetic cases are not model inference, quality, training or cache
validation. Run this file with --manual --num-gpus=1 on an authorized SM120 host.
"""

import pytest
import torch

from cosmos_framework.model.generator.mot import maskless_attention as replay

pytestmark = [pytest.mark.level(0), pytest.mark.gpus(1), pytest.mark.manual]


def _segments(history_growth: int) -> tuple[list[replay.GappedSegment], int, int]:
    # Seven views; GQA4 folds these query lengths to [4,12,28,68,124,260,28].
    # The mixed lengths and gaps exercise varlen scheduler padding/partial tiles.
    segments = []
    q_row, k_row = 2, 3
    for q_len, k_len in zip((1, 3, 7, 17, 31, 65, 7), (3, 5, 17, 29, 43, 71, 11), strict=True):
        k_len += history_growth
        segments.append(replay.GappedSegment(q_row, q_len, k_row, k_len))
        q_row += q_len + 3
        k_row += k_len + 5
    return segments, q_row + 2, k_row + 3


def _reference(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, segments: list[replay.GappedSegment]
) -> tuple[torch.Tensor, torch.Tensor]:
    # Independent FP32 dense attention per view; no replay metadata or FA4 calls.
    out = torch.zeros_like(q, dtype=torch.float32)
    lse = torch.full(q.shape[:-1], -torch.inf, device=q.device, dtype=torch.float32)
    for q_row, q_len, k_row, k_len in segments:
        qs = q[:, q_row : q_row + q_len].float()
        ks = k[:, k_row : k_row + k_len].float().repeat_interleave(4, dim=2)
        vs = v[:, k_row : k_row + k_len].float().repeat_interleave(4, dim=2)
        scores = torch.einsum("bqhd,bkhd->bhqk", qs, ks) * q.shape[-1] ** -0.5
        out[:, q_row : q_row + q_len] = torch.einsum("bhqk,bkhd->bqhd", scores.softmax(-1), vs)
        lse[:, q_row : q_row + q_len] = scores.logsumexp(-1).transpose(1, 2)
    return out, lse


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("history_growth", [0, 5])
def test_sm120_seven_view_gapped_forward(
    dtype: torch.dtype, history_growth: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("Requires an authorized CUDA GPU")
    device = torch.device("cuda", torch.cuda.current_device())
    if torch.cuda.get_device_capability(device) != (12, 0):
        pytest.skip("This regression targets exact SM120, not other architectures")
    pytest.importorskip("flash_attn.cute.interface")
    reason = replay.gapped_replay_unavailable_reason(device)
    assert reason is None, reason
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    segments, q_rows, kv_rows = _segments(history_growth)
    generator = torch.Generator(device=device).manual_seed(20261021 + history_growth)
    q = torch.randn(1, q_rows, 32, 128, device=device, dtype=dtype, generator=generator)
    k = torch.randn(1, kv_rows, 8, 128, device=device, dtype=dtype, generator=generator)
    v = torch.randn(1, kv_rows, 8, 128, device=device, dtype=dtype, generator=generator)
    originals = [tensor.clone() for tensor in (q, k, v)]
    plan = replay.build_gapped_replay_plan(segments, (), device)
    q_active = torch.zeros(q_rows, dtype=torch.bool, device=device)
    k_active = torch.zeros(kv_rows, dtype=torch.bool, device=device)
    for q_row, q_len, k_row, k_len in segments:
        q_active[q_row : q_row + q_len] = True
        k_active[k_row : k_row + k_len] = True
    with torch.no_grad():
        out, lse = replay._gapped_replay_forward(q, k, v, plan.handle)
        expected, expected_lse = _reference(q, k, v, segments)
        torch.cuda.synchronize(device)
        assert torch.isfinite(out).all()
        assert torch.isfinite(lse[:, q_active]).all()
        torch.testing.assert_close(out.float(), expected, atol=0.005, rtol=0.015)
        torch.testing.assert_close(lse[:, q_active], expected_lse[:, q_active], atol=0.001, rtol=0.0001)
        assert torch.count_nonzero(out[:, ~q_active]).item() == 0
        assert torch.isneginf(lse[:, ~q_active]).all()
        for original, actual in zip(originals, (q, k, v), strict=True):
            torch.testing.assert_close(actual, original, atol=0, rtol=0)

        # Finite gap values may be read by partial tiles but must never contribute.
        gap_k, gap_v = k.clone(), v.clone()
        gap_k[:, ~k_active] = 23
        gap_v[:, ~k_active] = -19
        gap_out, gap_lse = replay._gapped_replay_forward(q, gap_k, gap_v, plan.handle)
        torch.cuda.synchronize(device)
        torch.testing.assert_close(gap_out, out, atol=0, rtol=0)
        torch.testing.assert_close(gap_lse, lse, atol=0, rtol=0)

        # Poison every other view while preserving view zero's own keys/values.
        own = torch.zeros(kv_rows, dtype=torch.bool, device=device)
        first = segments[0]
        own[first.k_row : first.k_row + first.k_len] = True
        foreign_k, foreign_v = k.clone(), v.clone()
        foreign_k[:, ~own] = -11
        foreign_v[:, ~own] = 17
        foreign_out, foreign_lse = replay._gapped_replay_forward(q, foreign_k, foreign_v, plan.handle)
        torch.cuda.synchronize(device)
        rows = slice(first.q_row, first.q_row + first.q_len)
        torch.testing.assert_close(foreign_out[:, rows], out[:, rows], atol=0, rtol=0)
        torch.testing.assert_close(foreign_lse[:, rows], lse[:, rows], atol=0, rtol=0)
        assert torch.count_nonzero(foreign_out[:, ~q_active]).item() == 0
        assert torch.isneginf(foreign_lse[:, ~q_active]).all()
