# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""SM120 replay compatibility against independent dense attention references."""

import math
from types import ModuleType

import pytest
import torch

from cosmos_framework.model.generator.mot import maskless_attention as replay


@pytest.mark.L0
@pytest.mark.gpus(0)
@pytest.mark.parametrize("capability", [(9, 0), (10, 0), (10, 3), (12, 0)])
def test_replay_supported_architectures(monkeypatch: pytest.MonkeyPatch, capability: tuple[int, int]) -> None:
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: capability)
    monkeypatch.setattr(replay, "_flash_attn_interface", lambda: ModuleType("flash_attn"))
    assert replay.gapped_replay_unavailable_reason(torch.device("cuda")) is None


@pytest.mark.L0
@pytest.mark.gpus(0)
@pytest.mark.parametrize("capability", [(8, 0), (8, 9), (9, 1), (11, 0), (12, 1)])
def test_replay_rejects_unqualified_architectures(monkeypatch: pytest.MonkeyPatch, capability: tuple[int, int]) -> None:
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: capability)
    reason = replay.gapped_replay_unavailable_reason(torch.device("cuda"))
    assert reason is not None
    assert f"sm{capability[0]}{capability[1]}" in reason


@pytest.mark.L0
@pytest.mark.gpus(0)
@pytest.mark.parametrize(("device", "hip", "message"), [("cpu", None, "CUDA device"), ("cuda", "6.0", "not ROCm")])
def test_replay_rejects_non_nvidia_devices(
    monkeypatch: pytest.MonkeyPatch, device: str, hip: str | None, message: str
) -> None:
    monkeypatch.setattr(torch.version, "hip", hip)

    def unexpected_capability_query(device):
        pytest.fail("Non-NVIDIA devices must be rejected before querying CUDA capability")

    monkeypatch.setattr(torch.cuda, "get_device_capability", unexpected_capability_query)
    reason = replay.gapped_replay_unavailable_reason(torch.device(device))
    assert reason is not None
    assert message in reason


@pytest.mark.L0
@pytest.mark.gpus(0)
def test_sm120_replay_requires_flash_attention(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 0))

    def missing_backend():
        raise ImportError("FA4 is not installed")

    monkeypatch.setattr(replay, "_flash_attn_interface", missing_backend)
    with pytest.raises(RuntimeError, match="could not be imported: FA4 is not installed"):
        replay.require_gapped_replay(torch.device("cuda"))


def _require_sm120() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("Requires an SM120 GPU")
    pytest.importorskip("flash_attn.cute.interface")


@pytest.mark.L0
@pytest.mark.gpus(1)
@pytest.mark.parametrize("dim", [64, 128])
@pytest.mark.parametrize(("heads", "kv_heads"), [(8, 2), (8, 8), (16, 1)])
def test_sm120_replay_matches_dense_partition_reference(dim: int, heads: int, kv_heads: int) -> None:
    _require_sm120()
    torch.manual_seed(20261026)
    q = torch.randn(1, 257, heads, dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 389, kv_heads, dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    segments = [
        replay.GappedSegment(0, 31, 10, 71),
        replay.GappedSegment(55, 74, 100, 101),
        replay.GappedSegment(55, 74, 240, 100),
        replay.GappedSegment(199, 41, 3, 93),
        replay.GappedSegment(140, 31, 310, 71),
    ]
    caption = replay.ReplayMasklessPass(
        "caption",
        torch.tensor([0, 5, 55, 100, 199, 225], device="cuda"),
        torch.tensor([1, 8, 12, 190, 388], device="cuda"),
        (0, 6),
        (0, 5),
    )
    plan = replay.build_gapped_replay_plan(segments, [caption], torch.device("cuda"))
    out, lse = replay._gapped_replay_forward(q, k, v, plan.handle)

    # Count each partition's edges independently, including duplicated keys.
    count = torch.zeros(257, 389, device="cuda")
    for segment in segments:
        count[
            segment.q_row : segment.q_row + segment.q_len,
            segment.k_row : segment.k_row + segment.k_len,
        ] += 1
    count[caption.q_gather[:, None], caption.kv_gather[None, :]] += 1
    keys = k.float().repeat_interleave(heads // kv_heads, dim=2)
    values = v.float().repeat_interleave(heads // kv_heads, dim=2)
    scores = torch.einsum("bqhd,bkhd->bhqk", q.float(), keys) / math.sqrt(dim) + count.log()[None, None]
    valid = count.sum(-1) > 0
    weights = torch.nan_to_num(scores.softmax(-1))
    expected = torch.einsum("bhqk,bkhd->bqhd", weights, values)
    expected_lse = scores.logsumexp(-1).transpose(1, 2)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out.float(), expected, atol=0.025, rtol=0)
    torch.testing.assert_close(lse[:, valid], expected_lse[:, valid], atol=0.01, rtol=0)
    assert torch.count_nonzero(out[:, ~valid]) == 0
    assert torch.isneginf(lse[:, ~valid]).all()


@pytest.mark.L0
@pytest.mark.gpus(1)
def test_sm120_replay_backward_is_explicitly_unsupported() -> None:
    _require_sm120()
    q = torch.zeros(1, 1, 1, 64, device="cuda", dtype=torch.bfloat16)
    lse = torch.zeros(1, 1, 1, device="cuda")
    # An unregistered handle proves the guard runs before plan lookup or a kernel launch.
    handle = torch.tensor([-1], dtype=torch.int64)
    with pytest.raises(NotImplementedError, match="inference-only"):
        replay._gapped_replay_backward(q, q, q, q, lse, q, handle)
