# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import math
from typing import cast

import pytest
import torch
from torch.nn.attention.flex_attention import BlockMask, create_block_mask

from cosmos_framework.model.generator.mot.flex_attention import (
    FLEX_BLOCK_SIZE,
    FlexMetadata,
    _build_gen_sample_ids,
    _from_flex_layout,
    _multiview_mask_mod_factory,
    _to_flex_layout,
    build_block_mask,
    build_flex_metadata,
    build_multiview_block_mask,
    build_multiview_flex_metadata,
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


def _eager_block_mask(metadata: FlexMetadata) -> BlockMask:
    """A CPU ``BlockMask`` over the metadata's ``mask_mod``, built without torch.compile.

    ``build_block_mask`` goes through the compiled ``create_block_mask``, which is only
    exercised in the GPU test. The guard tests below just need a real ``BlockMask`` of a
    given size, so they use the eager builder and stay on CPU.
    """
    return create_block_mask(
        _multiview_mask_mod_factory(metadata),
        B=None,
        H=None,
        Q_LEN=metadata.seq_len,
        KV_LEN=metadata.seq_len,
        device="cpu",
        BLOCK_SIZE=FLEX_BLOCK_SIZE,
    )


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


# ── build_multiview_flex_metadata ────────────────────────────────────────────
# Each case mirrors what the packer hands the network: ``token_shapes`` holds
# ``(latent_t, patch_h, patch_w)`` per vision item, the ``latent_t`` axis is
# camera-major (all latent frames of view 0, then view 1, ...) because the
# multiview dataset concatenates the per-camera clips along T, and
# ``condition_frames`` indexes that same camera-major axis.
_MULTIVIEW_CASES: dict[str, dict] = {
    # One item, 2 views x 2 frames/view x 2 spatial tokens, first latent frame
    # of each camera kept clean.
    "two_views_first_frame_cond": dict(
        num_vision_items_per_sample=[1],
        token_shapes=[(4, 2, 1)],
        num_views_per_vision_item=[2],
        condition_frames=[[0, 2]],
        pad=4,
    ),
    # Transfer layout: every sample carries a noisy video plus a fully
    # conditioning control item sharing its (frame, view) grid.
    "transfer_two_items_two_samples": dict(
        num_vision_items_per_sample=[2, 2],
        token_shapes=[(4, 1, 2), (4, 1, 2), (2, 2, 1), (2, 2, 1)],
        num_views_per_vision_item=[2, 2, 1, 1],
        condition_frames=[[0], [0, 1, 2, 3], [], [0, 1]],
        pad=5,
    ),
    # Degenerate single-frame single-view item, exactly filling the sequence.
    "single_frame_single_view": dict(
        num_vision_items_per_sample=[1],
        token_shapes=[(1, 3, 3)],
        num_views_per_vision_item=[1],
        condition_frames=[[]],
        pad=0,
    ),
}


def _condition_mask(latent_t: int, condition_frames: list[int]) -> torch.Tensor:
    """A ``[T,1,1]`` float mask with ``1.0`` at conditioning frames.

    Matches what ``SequencePacker.pack_vision_tokens`` appends to
    ``vision.condition_mask`` (latent dtype, ``1`` = conditioning/clean frame).
    """
    mask = torch.zeros(latent_t, 1, 1)
    for frame_idx in condition_frames:
        mask[frame_idx, 0, 0] = 1.0
    return mask


def _case_offsets(case: dict) -> torch.Tensor:
    """Per-sample cumulative GEN offsets, ``int32`` as the packer emits them."""
    offsets = [0]
    item_idx = 0
    for num_items in case["num_vision_items_per_sample"]:
        sample_tokens = 0
        for _ in range(num_items):
            latent_t, patch_h, patch_w = case["token_shapes"][item_idx]
            sample_tokens += latent_t * patch_h * patch_w
            item_idx += 1
        offsets.append(offsets[-1] + sample_tokens)
    return torch.tensor(offsets, dtype=torch.int32)


def _expected_tokens(case: dict) -> list[dict]:
    """Per-token descriptors derived independently from the camera-major layout."""
    tokens: list[dict] = []
    item_idx = 0
    for sample, num_items in enumerate(case["num_vision_items_per_sample"]):
        for item_in_sample in range(num_items):
            latent_t, patch_h, patch_w = case["token_shapes"][item_idx]
            num_views = case["num_views_per_vision_item"][item_idx]
            frames_per_view = latent_t // num_views
            condition_frames = set(case["condition_frames"][item_idx])
            for view in range(num_views):
                for frame in range(frames_per_view):
                    is_cond = (view * frames_per_view + frame) in condition_frames
                    for _ in range(patch_h * patch_w):
                        tokens.append(
                            dict(
                                s=sample,
                                t=frame,
                                v=view,
                                noisy=not is_cond,
                                ct=item_in_sample if is_cond else -1,
                            )
                        )
            item_idx += 1
    return tokens


def _build_case_metadata(case: dict) -> tuple[FlexMetadata, int]:
    """Run the builder on a case; returns the metadata and the real token count."""
    offsets = _case_offsets(case)
    num_real = int(offsets[-1])
    condition_masks = [
        _condition_mask(token_shape[0], condition_frames)
        for token_shape, condition_frames in zip(case["token_shapes"], case["condition_frames"])
    ]
    metadata = build_multiview_flex_metadata(
        seq_len=num_real + case["pad"],
        full_q_offsets=offsets,
        token_shapes=case["token_shapes"],
        condition_masks=condition_masks,
        num_vision_items_per_sample=case["num_vision_items_per_sample"],
        num_views_per_vision_item=case["num_views_per_vision_item"],
        device=torch.device("cpu"),
    )
    return metadata, num_real


@pytest.mark.L0
def test_build_multiview_flex_metadata_camera_major_layout() -> None:
    metadata, num_real = _build_case_metadata(_MULTIVIEW_CASES["two_views_first_frame_cond"])

    assert (num_real, metadata.seq_len) == (8, 12)
    # View-outer, frame-inner, spatial-innermost: [v0f0 x2, v0f1 x2, v1f0 x2, v1f1 x2].
    assert torch.equal(metadata.frame_id, torch.tensor([0, 0, 1, 1, 0, 0, 1, 1] + [-1] * 4))
    assert torch.equal(metadata.view_id, torch.tensor([0, 0, 0, 0, 1, 1, 1, 1] + [-1] * 4))
    # Camera-major condition frames {0, 2} are (v0,f0) and (v1,f0).
    assert torch.equal(
        metadata.is_noisy,
        torch.tensor([False, False, True, True, False, False, True, True] + [False] * 4),
    )
    assert torch.equal(metadata.cond_type_id, torch.tensor([0, 0, -1, -1, 0, 0, -1, -1] + [-1] * 4))
    assert torch.equal(metadata.sample_id, torch.tensor([0] * 8 + [-1] * 4))

    assert metadata.sample_id.dtype == torch.long
    assert metadata.frame_id.dtype == torch.long
    assert metadata.view_id.dtype == torch.long
    assert metadata.cond_type_id.dtype == torch.long
    assert metadata.is_noisy.dtype == torch.bool


@pytest.mark.L0
@pytest.mark.parametrize("case_name", sorted(_MULTIVIEW_CASES))
def test_build_multiview_flex_metadata_matches_reference_layout(case_name: str) -> None:
    case = _MULTIVIEW_CASES[case_name]
    metadata, num_real = _build_case_metadata(case)
    tokens = _expected_tokens(case)
    assert len(tokens) == num_real, "the reference layout must cover exactly the packed GEN tokens"

    expected = _metadata_from_tokens(tokens, seq_len=metadata.seq_len)
    assert torch.equal(metadata.sample_id, expected.sample_id)
    assert torch.equal(metadata.frame_id, expected.frame_id)
    assert torch.equal(metadata.view_id, expected.view_id)
    assert torch.equal(metadata.is_noisy, expected.is_noisy)
    assert torch.equal(metadata.cond_type_id, expected.cond_type_id)


@pytest.mark.L0
@pytest.mark.parametrize("case_name", sorted(_MULTIVIEW_CASES))
def test_build_multiview_flex_metadata_mask_matches_reference(case_name: str) -> None:
    """The built metadata drives exactly the documented visibility rules."""
    case = _MULTIVIEW_CASES[case_name]
    metadata, _ = _build_case_metadata(case)

    got = _mask_mod_to_dense(metadata)
    expected = _reference_visibility(_expected_tokens(case), metadata.seq_len)
    assert torch.equal(got, expected)


@pytest.mark.L0
def test_build_multiview_flex_metadata_pads_with_sentinels() -> None:
    case = _MULTIVIEW_CASES["transfer_two_items_two_samples"]
    metadata, num_real = _build_case_metadata(case)
    pad = case["pad"]
    tail = slice(num_real, metadata.seq_len)

    sentinel = torch.full((pad,), -1)
    assert torch.equal(metadata.sample_id[tail], sentinel)
    assert torch.equal(metadata.frame_id[tail], sentinel)
    assert torch.equal(metadata.view_id[tail], sentinel)
    assert torch.equal(metadata.cond_type_id[tail], sentinel)
    assert not metadata.is_noisy[tail].any()

    m = _mask_mod_to_dense(metadata)
    assert not m[:num_real, num_real:].any()
    assert not m[num_real:, :num_real].any()
    # Padded queries attend to padding only, which keeps their softmax non-empty.
    assert m[num_real:, num_real:].all()


@pytest.mark.L0
def test_build_multiview_flex_metadata_aligns_control_item_by_frame_and_view() -> None:
    """Noisy tokens see the control item's tokens at the same (frame, view) only."""
    metadata, _ = _build_case_metadata(_MULTIVIEW_CASES["transfer_two_items_two_samples"])
    m = _mask_mod_to_dense(metadata)

    # Sample 0 holds two 2-view x 2-frame x 2-spatial items: the noisy video at
    # tokens 0..7 and the fully conditioning control at tokens 8..15.
    noisy_v0f1 = 2
    cond_v0f1 = 8 + 2
    cond_v1f1 = 8 + 6
    assert metadata.is_noisy[noisy_v0f1]
    assert not metadata.is_noisy[cond_v0f1]

    assert m[noisy_v0f1, cond_v0f1]
    assert not m[noisy_v0f1, cond_v1f1], "conditioning from another camera must stay masked"
    assert not m[cond_v0f1, noisy_v0f1], "conditioning queries never attend to noisy keys"


@pytest.mark.L0
def test_build_multiview_flex_metadata_cond_types_are_per_sample() -> None:
    metadata, _ = _build_case_metadata(_MULTIVIEW_CASES["transfer_two_items_two_samples"])

    # Conditioning ids are the item's index within its own sample, which is
    # enough to separate the streams because every rule is gated on same_sample.
    assert metadata.cond_type_id[0] == 0  # sample 0, item 0, conditioning frame
    assert metadata.cond_type_id[8] == 1  # sample 0, item 1 (control)
    assert metadata.cond_type_id[20] == 1  # sample 1, item 1 (control)


@pytest.mark.L0
def test_build_multiview_flex_metadata_accepts_flat_bool_condition_mask() -> None:
    """A flat bool mask must be equivalent to the packer's ``[T,1,1]`` float mask."""
    case = _MULTIVIEW_CASES["two_views_first_frame_cond"]
    metadata, _ = _build_case_metadata(case)

    from_bool = build_multiview_flex_metadata(
        seq_len=metadata.seq_len,
        full_q_offsets=_case_offsets(case),
        token_shapes=case["token_shapes"],
        condition_masks=[torch.tensor([True, False, True, False])],
        num_vision_items_per_sample=case["num_vision_items_per_sample"],
        num_views_per_vision_item=case["num_views_per_vision_item"],
        device=torch.device("cpu"),
    )
    assert torch.equal(from_bool.is_noisy, metadata.is_noisy)
    assert torch.equal(from_bool.cond_type_id, metadata.cond_type_id)


@pytest.mark.L0
@pytest.mark.parametrize("num_views", [0, 3])
def test_build_multiview_flex_metadata_rejects_bad_view_count(num_views: int) -> None:
    with pytest.raises(ValueError, match="not divisible by num_views"):
        build_multiview_flex_metadata(
            seq_len=16,
            full_q_offsets=torch.tensor([0, 8], dtype=torch.int32),
            token_shapes=[(4, 1, 2)],
            condition_masks=[_condition_mask(4, [])],
            num_vision_items_per_sample=[1],
            num_views_per_vision_item=[num_views],
            device=torch.device("cpu"),
        )


@pytest.mark.L0
def test_build_multiview_flex_metadata_rejects_condition_mask_length() -> None:
    with pytest.raises(ValueError, match="expected 4"):
        build_multiview_flex_metadata(
            seq_len=16,
            full_q_offsets=torch.tensor([0, 8], dtype=torch.int32),
            token_shapes=[(4, 1, 2)],
            condition_masks=[_condition_mask(3, [])],
            num_vision_items_per_sample=[1],
            num_views_per_vision_item=[2],
            device=torch.device("cpu"),
        )


@pytest.mark.L0
def test_build_multiview_flex_metadata_rejects_item_count_mismatch() -> None:
    with pytest.raises(ValueError, match="must align with flattened vision items"):
        build_multiview_flex_metadata(
            seq_len=16,
            full_q_offsets=torch.tensor([0, 8], dtype=torch.int32),
            token_shapes=[(4, 1, 2)],
            condition_masks=[_condition_mask(4, [])],
            num_vision_items_per_sample=[2],  # two items claimed, one supplied
            num_views_per_vision_item=[2],
            device=torch.device("cpu"),
        )


@pytest.mark.L0
def test_build_multiview_flex_metadata_rejects_packed_token_count_mismatch() -> None:
    with pytest.raises(ValueError, match="packed full-attention splits disagree"):
        build_multiview_flex_metadata(
            seq_len=16,
            full_q_offsets=torch.tensor([0, 7], dtype=torch.int32),  # item contributes 8
            token_shapes=[(4, 1, 2)],
            condition_masks=[_condition_mask(4, [])],
            num_vision_items_per_sample=[1],
            num_views_per_vision_item=[2],
            device=torch.device("cpu"),
        )


@pytest.mark.L0
def test_build_multiview_flex_metadata_rejects_mixed_grids_in_a_sample() -> None:
    with pytest.raises(ValueError, match=r"same \(num_views, frames_per_view\) grid"):
        build_multiview_flex_metadata(
            seq_len=32,
            full_q_offsets=torch.tensor([0, 16], dtype=torch.int32),
            # Same token count, but 2 views x 2 frames against 1 view x 4 frames.
            token_shapes=[(4, 1, 2), (4, 1, 2)],
            condition_masks=[_condition_mask(4, [0]), _condition_mask(4, [0, 1, 2, 3])],
            num_vision_items_per_sample=[2],
            num_views_per_vision_item=[2, 1],
            device=torch.device("cpu"),
        )


@pytest.mark.L0
def test_build_multiview_flex_metadata_rejects_overlong_metadata() -> None:
    with pytest.raises(ValueError, match="exceeding GEN sequence length"):
        build_multiview_flex_metadata(
            seq_len=4,  # smaller than the 8 tokens the item contributes
            full_q_offsets=torch.tensor([0, 8], dtype=torch.int32),
            token_shapes=[(4, 1, 2)],
            condition_masks=[_condition_mask(4, [])],
            num_vision_items_per_sample=[1],
            num_views_per_vision_item=[2],
            device=torch.device("cpu"),
        )


@pytest.mark.L0
def test_flex_attention_varlen_rejects_unaligned_seq_len() -> None:
    seq_len = FLEX_BLOCK_SIZE + 1  # not a multiple of the block size
    q = torch.randn(1, seq_len, 2, 16)
    meta = _metadata_from_tokens([dict(s=0, t=0, v=0, noisy=True, ct=-1)], seq_len=seq_len)
    with pytest.raises(ValueError, match="block-aligned GEN sequence length"):
        flex_attention_varlen(q, q, q, _eager_block_mask(meta))


@pytest.mark.L0
def test_flex_attention_varlen_rejects_seq_len_mismatch() -> None:
    """A mask built for another pack must be rejected, not silently applied."""
    q = torch.randn(1, FLEX_BLOCK_SIZE, 2, 8)  # one block of GEN tokens
    # Mask for a two-block pack, i.e. one that q/k/v did not come from.
    other_pack = _metadata_from_tokens(
        [dict(s=0, t=0, v=0, noisy=True, ct=-1)],
        seq_len=2 * FLEX_BLOCK_SIZE,
    )
    with pytest.raises(ValueError, match="but the GEN sequence is"):
        flex_attention_varlen(q, q, q, _eager_block_mask(other_pack))


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
    seq_len = FLEX_BLOCK_SIZE  # single block

    tokens = _make_multiview_tokens()
    n_real = len(tokens)
    metadata = _metadata_from_tokens(tokens, seq_len=seq_len, device=device)
    block_mask = build_block_mask(metadata, torch.device(device))

    q = torch.randn(1, seq_len, num_q_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(1, seq_len, num_kv_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(1, seq_len, num_kv_heads, head_dim, device=device, dtype=dtype)

    result = flex_attention_varlen(q, k, v, block_mask, return_lse=return_lse)
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


@pytest.mark.L0
@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlexAttention kernels require a GPU.")
def test_build_multiview_block_mask_covers_the_padded_gen_stream() -> None:
    """The composed entry point returns a mask sized for the block-padded GEN stream.

    That size is what ``flex_attention_varlen`` checks q/k/v against, so a mask built
    for the real token count instead of the padded one would be rejected there.
    """
    device = torch.device("cuda")
    case = _MULTIVIEW_CASES["transfer_two_items_two_samples"]
    condition_masks = [
        _condition_mask(token_shape[0], condition_frames)
        for token_shape, condition_frames in zip(case["token_shapes"], case["condition_frames"])
    ]

    block_mask = build_multiview_block_mask(
        seq_len=FLEX_BLOCK_SIZE,
        full_q_offsets=_case_offsets(case).to(device),
        token_shapes=case["token_shapes"],
        condition_masks=condition_masks,
        num_vision_items_per_sample=case["num_vision_items_per_sample"],
        num_views_per_vision_item=case["num_views_per_vision_item"],
        device=device,
    )

    assert block_mask.shape[-2:] == (FLEX_BLOCK_SIZE, FLEX_BLOCK_SIZE)
