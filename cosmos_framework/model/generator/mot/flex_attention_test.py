# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import math
from typing import cast

import pytest
import torch

from cosmos_framework.model.generator.mot.flex_attention import (
    _FLEX_BLOCK_SIZE,
    FlexMetadata,
    _build_gen_sample_ids,
    _from_flex_layout,
    _multiview_mask_mod_factory,
    _to_flex_layout,
    build_flex_metadata,
    flex_attention_varlen,
)

# Conditioning-stream type ids used throughout the tests (matches the layout in
# assets/mv_attn_mask.py: 0 = cond type A, 1 = cond type B, -1 = noisy/visual).
_COND_A = 0
_COND_B = 1


def _metadata_from_tokens(tokens: list[dict], seq_len: int | None = None, device: str = "cpu") -> FlexMetadata:
    """Build a :class:`FlexMetadata` from an explicit list of token descriptors.

    Each token dict has ``s`` (sample), ``t`` (frame), ``v`` (view), ``noisy``
    (bool) and ``ct`` (cond type id, ``-1`` for noisy). Positions beyond
    ``len(tokens)`` are padding and get the ``-1`` / ``False`` sentinels.
    """
    n = len(tokens)
    if seq_len is None:
        seq_len = n
    pad = seq_len - n
    assert pad >= 0

    def col(key: str) -> torch.Tensor:
        return torch.tensor([tok[key] for tok in tokens] + [-1] * pad, dtype=torch.long, device=device)

    is_noisy = torch.tensor([tok["noisy"] for tok in tokens] + [False] * pad, dtype=torch.bool, device=device)
    return FlexMetadata(
        seq_len=seq_len,
        sample_id=col("s"),
        frame_id=col("t"),
        view_id=col("v"),
        is_noisy=is_noisy,
        cond_type_id=col("ct"),
    )


def _make_multiview_tokens() -> list[dict]:
    """A small multi-sample multiview layout with cond-A, cond-B and noisy tokens."""
    tokens: list[dict] = []
    # Sample 0: 2 frames x 2 views; per (t, v): 1 cond-A, 1 cond-B, 2 noisy.
    for t in (0, 1):
        for v in (0, 1):
            tokens.append(dict(s=0, t=t, v=v, noisy=False, ct=_COND_A))
            tokens.append(dict(s=0, t=t, v=v, noisy=False, ct=_COND_B))
            tokens.append(dict(s=0, t=t, v=v, noisy=True, ct=-1))
            tokens.append(dict(s=0, t=t, v=v, noisy=True, ct=-1))
    # Sample 1: 1 frame, 1 view; 1 cond-A, 1 cond-B, 3 noisy.
    tokens.append(dict(s=1, t=0, v=0, noisy=False, ct=_COND_A))
    tokens.append(dict(s=1, t=0, v=0, noisy=False, ct=_COND_B))
    for _ in range(3):
        tokens.append(dict(s=1, t=0, v=0, noisy=True, ct=-1))
    return tokens


def _reference_visibility(tokens: list[dict], seq_len: int) -> torch.Tensor:
    """Ground-truth ``[seq_len, seq_len]`` bool matrix ``M[q, k] = q attends to k``.

    Encodes exactly the documented multiview rules; padding positions (index >=
    len(tokens)) share the ``-1`` sample so they only attend to each other.
    """

    def desc(i: int) -> dict:
        if i < len(tokens):
            return tokens[i]
        return dict(s=-1, t=-1, v=-1, noisy=False, ct=-1)

    m = torch.zeros(seq_len, seq_len, dtype=torch.bool)
    for q in range(seq_len):
        dq = desc(q)
        for k in range(seq_len):
            dk = desc(k)
            if dq["s"] != dk["s"]:
                continue
            same_fv = dq["t"] == dk["t"] and dq["v"] == dk["v"]
            if not dq["noisy"] and not dk["noisy"]:
                ok = same_fv and dq["ct"] == dk["ct"]
            elif dq["noisy"] and dk["noisy"]:
                ok = True
            elif dq["noisy"] and not dk["noisy"]:
                ok = same_fv
            else:  # cond query -> noisy key: never
                ok = False
            m[q, k] = ok
    return m


def _mask_mod_to_dense(metadata: FlexMetadata) -> torch.Tensor:
    """Evaluate the metadata's ``mask_mod`` on every (q, k) pair -> ``[S, S]`` bool."""
    mask_mod = _multiview_mask_mod_factory(metadata)
    s = metadata.seq_len
    q_idx = torch.arange(s).view(-1, 1).expand(s, s)
    kv_idx = torch.arange(s).view(1, -1).expand(s, s)
    zero = torch.tensor(0)
    return mask_mod(zero, zero, q_idx, kv_idx)


@pytest.mark.L0
def test_build_gen_sample_ids_marks_padding() -> None:
    offsets = torch.tensor([0, 3, 7], dtype=torch.long)
    sample_id = _build_gen_sample_ids(offsets, seq_len=10, device=torch.device("cpu"))
    expected = torch.tensor([0, 0, 0, 1, 1, 1, 1, -1, -1, -1], dtype=torch.long)
    assert torch.equal(sample_id, expected)


@pytest.mark.L0
def test_build_gen_sample_ids_no_padding() -> None:
    offsets = torch.tensor([0, 2, 5], dtype=torch.long)
    sample_id = _build_gen_sample_ids(offsets, seq_len=5, device=torch.device("cpu"))
    assert torch.equal(sample_id, torch.tensor([0, 0, 1, 1, 1], dtype=torch.long))


@pytest.mark.L0
def test_build_flex_metadata_populates_all_fields() -> None:
    seq_len = 4
    sample_id = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    frame_id = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    view_id = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    is_noisy = torch.tensor([False, True, False, True])
    cond_type_id = torch.tensor([0, -1, 1, -1], dtype=torch.long)

    meta = build_flex_metadata(
        seq_len,
        sample_id=sample_id,
        frame_id=frame_id,
        view_id=view_id,
        is_noisy=is_noisy,
        cond_type_id=cond_type_id,
    )

    assert meta.seq_len == seq_len
    assert torch.equal(meta.sample_id, sample_id)
    assert torch.equal(meta.frame_id, frame_id)
    assert torch.equal(meta.view_id, view_id)
    assert torch.equal(meta.is_noisy, is_noisy)
    assert torch.equal(meta.cond_type_id, cond_type_id)


@pytest.mark.L0
def test_flex_layout_roundtrip() -> None:
    x4 = torch.randn(1, 6, 4, 8)  # [1,S,H,D]
    assert _to_flex_layout(x4).shape == (1, 4, 6, 8)  # [1,H,S,D]
    assert torch.equal(_from_flex_layout(_to_flex_layout(x4)), x4)

    x3 = torch.randn(1, 4, 6)  # [1,H,S] (LSE layout)
    assert _from_flex_layout(x3).shape == (1, 6, 4)  # [1,S,H]


@pytest.mark.L0
def test_multiview_mask_mod_matches_reference() -> None:
    tokens = _make_multiview_tokens()
    seq_len = len(tokens)
    metadata = _metadata_from_tokens(tokens, seq_len=seq_len)

    got = _mask_mod_to_dense(metadata)
    expected = _reference_visibility(tokens, seq_len)
    assert torch.equal(got, expected)


@pytest.mark.L0
def test_multiview_mask_mod_specific_rules() -> None:
    # Two views, two frames, one sample. Layout index -> token:
    #  0: condA (t0,v0)   1: condB (t0,v0)   2: noisy (t0,v0)
    #  3: condA (t0,v1)   4: noisy (t0,v1)
    #  5: condA (t1,v0)   6: noisy (t1,v0)
    tokens = [
        dict(s=0, t=0, v=0, noisy=False, ct=_COND_A),
        dict(s=0, t=0, v=0, noisy=False, ct=_COND_B),
        dict(s=0, t=0, v=0, noisy=True, ct=-1),
        dict(s=0, t=0, v=1, noisy=False, ct=_COND_A),
        dict(s=0, t=0, v=1, noisy=True, ct=-1),
        dict(s=0, t=1, v=0, noisy=False, ct=_COND_A),
        dict(s=0, t=1, v=0, noisy=True, ct=-1),
    ]
    m = _mask_mod_to_dense(_metadata_from_tokens(tokens))

    # cond-A (t0,v0) attends only to itself among these (same type/frame/view).
    assert m[0, 0]
    assert not m[0, 1]  # different cond type
    assert not m[0, 2]  # cond -> noisy never
    assert not m[0, 3]  # different view
    assert not m[0, 5]  # different frame

    # noisy (t0,v0) attends to all noisy in the sample + cond of same (t,v).
    assert m[2, 2] and m[2, 4] and m[2, 6]  # all noisy tokens
    assert m[2, 0] and m[2, 1]  # cond A & B in same (t0,v0)
    assert not m[2, 3]  # cond in different view
    assert not m[2, 5]  # cond in different frame


@pytest.mark.L0
def test_multiview_mask_mod_block_diagonal_across_samples() -> None:
    tokens = _make_multiview_tokens()
    metadata = _metadata_from_tokens(tokens)
    m = _mask_mod_to_dense(metadata)

    sample_id = metadata.sample_id
    cross = sample_id.view(-1, 1) != sample_id.view(1, -1)
    assert not m[cross].any(), "attention must never cross sample boundaries"


@pytest.mark.L0
def test_multiview_mask_mod_padding_isolated() -> None:
    tokens = _make_multiview_tokens()
    n = len(tokens)
    seq_len = n + 5  # add padding positions
    metadata = _metadata_from_tokens(tokens, seq_len=seq_len)
    m = _mask_mod_to_dense(metadata)

    # Real queries never attend to padding keys, and vice versa.
    assert not m[:n, n:].any()
    assert not m[n:, :n].any()
    # Padded queries attend only to padding (non-empty softmax -> no NaN).
    assert m[n:, n:].all()


@pytest.mark.L0
def test_flex_attention_varlen_rejects_unaligned_seq_len() -> None:
    seq_len = _FLEX_BLOCK_SIZE + 1  # not a multiple of the block size
    q = torch.randn(1, seq_len, 2, 8)
    meta = _metadata_from_tokens([dict(s=0, t=0, v=0, noisy=True, ct=-1)], seq_len=seq_len)
    with pytest.raises(AssertionError, match="pre-padded to a multiple"):
        flex_attention_varlen(q, q, q, meta)


@pytest.mark.L0
def test_flex_attention_varlen_rejects_seq_len_mismatch() -> None:
    seq_len = _FLEX_BLOCK_SIZE
    q = torch.randn(1, seq_len, 2, 8)
    meta = _metadata_from_tokens([dict(s=0, t=0, v=0, noisy=True, ct=-1)], seq_len=seq_len - 1)
    with pytest.raises(AssertionError, match="must match the GEN sequence length"):
        flex_attention_varlen(q, q, q, meta)


def _reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dense masked attention reference. q/k/v: ``[1,S,H,D]`` / ``[1,S,Hkv,D]``.

    Returns ``(out [1,S,H,D], lse [1,S,H])`` with the natural-log LSE and the
    default ``1/sqrt(D)`` scale, matching the FlexAttention convention.
    """
    qh = q[0]  # [S,H,D]
    kh = k[0]  # [S,Hkv,D]
    vh = v[0]
    num_q_heads = qh.shape[1]
    num_kv_heads = kh.shape[1]
    if num_q_heads != num_kv_heads:
        factor = num_q_heads // num_kv_heads
        kh = kh.repeat_interleave(factor, dim=1)
        vh = vh.repeat_interleave(factor, dim=1)
    scale = 1.0 / math.sqrt(qh.shape[-1])
    scores = torch.einsum("shd,thd->hst", qh, kh) * scale  # [H,S,S]
    neg_inf = torch.finfo(scores.dtype).min
    scores = scores.masked_fill(~mask.view(1, *mask.shape), neg_inf)
    weights = torch.softmax(scores, dim=-1)
    out = torch.einsum("hst,thd->shd", weights, vh)  # [S,H,D]
    lse = torch.logsumexp(scores, dim=-1)  # [H,S]
    return out.unsqueeze(0), lse.transpose(0, 1).unsqueeze(0)


@pytest.mark.L0
@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlexAttention kernels require a GPU.")
@pytest.mark.parametrize("num_kv_heads", [4, 1])
@pytest.mark.parametrize("return_lse", [True, False])
def test_flex_attention_varlen_matches_reference(num_kv_heads: int, return_lse: bool) -> None:
    torch.manual_seed(0)
    torch.compiler.reset()
    device = "cuda"
    dtype = torch.float32
    num_q_heads = 4
    head_dim = 32
    seq_len = _FLEX_BLOCK_SIZE  # single block

    tokens = _make_multiview_tokens()
    n_real = len(tokens)
    metadata = _metadata_from_tokens(tokens, seq_len=seq_len, device=device)

    q = torch.randn(1, seq_len, num_q_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(1, seq_len, num_kv_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(1, seq_len, num_kv_heads, head_dim, device=device, dtype=dtype)

    result = flex_attention_varlen(q, k, v, metadata, return_lse=return_lse)
    lse: torch.Tensor | None = None
    if return_lse:
        assert isinstance(result, tuple)
        out = cast(torch.Tensor, result[0])
        lse = cast(torch.Tensor, result[1])
        assert lse.shape == (1, seq_len, num_q_heads)
    else:
        assert isinstance(result, torch.Tensor)
        out = cast(torch.Tensor, result)
    assert out.shape == (1, seq_len, num_q_heads, head_dim)

    mask = _mask_mod_to_dense(metadata).to(device)
    ref_out, ref_lse = _reference_attention(q, k, v, mask)

    # Only compare real (non-padding) token rows.
    real = slice(0, n_real)
    torch.testing.assert_close(out[:, real], ref_out[:, real], atol=2e-2, rtol=2e-2)
    if lse is not None:
        torch.testing.assert_close(lse[:, real], ref_lse[:, real], atol=2e-2, rtol=2e-2)
