# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import contextlib
import dataclasses
import math
import unittest.mock
from collections.abc import Iterator
from typing import cast

import pytest
import torch
from torch.nn.attention.flex_attention import BlockMask, create_block_mask

from cosmos_framework.configs.base.defaults.flex_attention import (
    ATTENTION_SCOPES,
    AttentionScope,
)
from cosmos_framework.model.generator.mot import flex_attention as flex_attention_module
from cosmos_framework.model.generator.mot.attention import build_packed_sequence
from cosmos_framework.model.generator.mot.flex_attention import (
    FlexBackend,
    FlexMetadata,
    MaskItem,
    _build_stream_sample_ids,
    _from_flex_layout,
    _get_triton_flex_backend,
    _metadata_groups,
    _multiview_mask_mod,
    _to_flex_layout,
    build_block_mask,
    build_multiview_block_mask,
    build_multiview_flex_metadata,
    flash_backend_block_size,
    flex_attention,
    resolve_flex_backend,
    triton_backend_block_size,
)
from cosmos_framework.data.generator.sequence_packing.runtime import (
    SequencePack,
    get_causal_seq,
    get_full_only_seq,
)
from cosmos_framework.data.generator.sequence_packing.sequence import ModalityData, PackedSequence

# The two mask geometries the cases below are held to. Triton's block is square and
# device-independent. FlashAttention-4's is the Blackwell one, where each CTA owns two query
# tiles (``q_stage=2``) and the mask is therefore 256x128 -- the only configuration in which the
# two stream alignments differ, which is what makes it the one worth covering. It is spelled out
# rather than resolved because :func:`_get_flash_flex_backend` needs the GPU while every mask
# here is built on CPU, and because FA4 on Hopper steps 128x128, i.e. the Triton geometry
# already covered. ``test_resolve_flex_backend_takes_flash_when_it_is_available`` holds it to
# what the real constructor produces.
_TRITON_BACKEND = _get_triton_flex_backend()
_FLASH_BACKEND = FlexBackend(name="flash", block_size=(256, 128), kernel_options={"BACKEND": "FLASH"})

# Building a mask is device-independent, so every case that does not run a kernel is held to
# both geometries. The kernels are a separate matter -- they need the hardware and the
# ``flash-attn-4`` package, which :func:`_flash_fused_case` probes for.
_GEOMETRIES = pytest.mark.parametrize("backend", (_TRITON_BACKEND, _FLASH_BACKEND), ids=("triton", "flash"))

# How far noisy tokens reach among the noisy tokens of their sample. Held to all three wherever
# a case has more than one view and frame, since the scope reshapes the largest quadrant of the
# mask and the two narrow ones are the only rules keyed on the view or the frame alone.
_NOISY_SCOPES = pytest.mark.parametrize("attention_scope", ATTENTION_SCOPES)

# Padded stream lengths that tile at either geometry, so one pack serves a parametrized case
# without being resized per backend: the coarsest query block for the GEN stream, and the key
# block both backends step for the UND prefix.
_GEN_ALIGNMENT = math.lcm(_TRITON_BACKEND.full_seq_alignment, _FLASH_BACKEND.full_seq_alignment)
_UND_ALIGNMENT = math.lcm(_TRITON_BACKEND.causal_seq_alignment, _FLASH_BACKEND.causal_seq_alignment)


def _metadata_from_tokens(
    tokens: list[dict],
    seq_len: int | None = None,
    device: str = "cpu",
    und_samples: list[int] | None = None,
    attention_scope: str = "all_views",
) -> FlexMetadata:
    """Build a :class:`FlexMetadata` from an explicit list of GEN token descriptors.

    Each token dict has ``s`` (sample), ``t`` (frame), ``v`` (view), ``noisy`` (bool),
    and optionally ``control`` (bool, defaults ``False``) marking a control (e.g. WSM)
    token. Positions beyond ``len(tokens)`` are padding and get the ``-1`` / ``False``
    sentinels.

    ``und_samples`` prepends the UND half of the fused key stream: one sample id per
    UND token, ``-1`` for UND padding. That id is the only field the gen->und rule
    reads, so the multiview fields are sentinels there. Left ``None``, the metadata is
    GEN-only and the mask it drives is square.

    ``attention_scope`` is typed loosely here so the case below that hands it an
    unrecognised scope can reach the metadata's own check.
    """
    n = len(tokens)
    if seq_len is None:
        seq_len = n
    pad = seq_len - n
    assert pad >= 0
    und_samples = list(und_samples or [])
    num_und = len(und_samples)

    def col(key: str) -> torch.Tensor:
        return torch.tensor([-1] * num_und + [tok[key] for tok in tokens] + [-1] * pad, dtype=torch.long, device=device)

    is_noisy = torch.tensor(
        [False] * num_und + [tok["noisy"] for tok in tokens] + [False] * pad, dtype=torch.bool, device=device
    )
    is_control = torch.tensor(
        [False] * num_und + [tok.get("control", False) for tok in tokens] + [False] * pad,
        dtype=torch.bool,
        device=device,
    )
    return FlexMetadata(
        seq_len=num_und + seq_len,
        sample_id=torch.tensor(
            und_samples + [tok["s"] for tok in tokens] + [-1] * pad, dtype=torch.long, device=device
        ),
        frame_id=col("t"),
        view_id=col("v"),
        is_noisy=is_noisy,
        is_control=is_control,
        num_und=num_und,
        attention_scope=cast(AttentionScope, attention_scope),
    )


def _und_samples(*sample_lens: int, length: int) -> list[int]:
    """One padded UND prefix: ``sample_lens[i]`` tokens of sample ``i``, then ``-1`` padding.

    ``length`` is the key block the prefix answers to, so that the UND/GEN boundary lands on a
    block boundary of whichever backend runs the mask.
    """
    ids = [sample for sample, count in enumerate(sample_lens) for _ in range(count)]
    assert len(ids) <= length
    return ids + [-1] * (length - len(ids))


def _make_multiview_tokens() -> list[dict]:
    """A small multi-sample multiview layout with two conditioning tokens and two noisy per cell.

    The two conditioning tokens per ``(t, v)`` cell used to carry distinct cond-type ids
    (before that field existed); they are now indistinguishable RGB conditioning tokens,
    kept as two per cell to still exercise a multi-token conditioning group at fixed
    token indices, which several tests below key on directly.
    """
    tokens: list[dict] = []
    # Sample 0: 2 frames x 2 views; per (t, v): 2 conditioning, 2 noisy.
    for t in (0, 1):
        for v in (0, 1):
            tokens.append(dict(s=0, t=t, v=v, noisy=False))
            tokens.append(dict(s=0, t=t, v=v, noisy=False))
            tokens.append(dict(s=0, t=t, v=v, noisy=True))
            tokens.append(dict(s=0, t=t, v=v, noisy=True))
    # Sample 1: 1 frame, 1 view; 2 conditioning, 3 noisy.
    tokens.append(dict(s=1, t=0, v=0, noisy=False))
    tokens.append(dict(s=1, t=0, v=0, noisy=False))
    for _ in range(3):
        tokens.append(dict(s=1, t=0, v=0, noisy=True))
    return tokens


def _reference_visibility(
    tokens: list[dict],
    seq_len: int,
    und_samples: list[int] | None = None,
    attention_scope: AttentionScope = "all_views",
) -> torch.Tensor:
    """Ground-truth ``[seq_len, num_und + seq_len]`` bool ``M[q, k] = q attends to k``.

    Encodes exactly the documented multiview rules; padding positions (index >=
    len(tokens)) share the ``-1`` sample so they only attend to each other. With
    ``und_samples`` the matrix gains the gen->und columns on the left, where the rule is
    "same sample" alone, and is rectangular as the fused mask is.

    ``attention_scope`` narrows the RGB<->RGB reach (conditioning<->conditioning and
    noisy->any RGB alike); it never widens a control token's reach, which is always its
    own view, any frame. Spelled out per scope rather than derived from the
    implementation's gates, so a rule that changes shape has to be restated here.
    """
    und_samples = list(und_samples or [])
    num_und = len(und_samples)

    def desc(i: int) -> dict:
        if i < len(tokens):
            return tokens[i]
        return dict(s=-1, t=-1, v=-1, noisy=False, control=False)

    def in_scope(dq: dict, dk: dict) -> bool:
        if attention_scope == "all_views":
            return True
        if attention_scope == "same_view":
            return dq["v"] == dk["v"]
        return dq["v"] == dk["v"] or dq["t"] == dk["t"]

    m = torch.zeros(seq_len, num_und + seq_len, dtype=torch.bool)
    for q in range(seq_len):
        dq = desc(q)
        for k, und_sample in enumerate(und_samples):
            m[q, k] = dq["s"] == und_sample
        for k in range(seq_len):
            dk = desc(k)
            if dq["s"] != dk["s"]:
                continue
            q_control = dq.get("control", False)
            k_control = dk.get("control", False)
            same_view = dq["v"] == dk["v"]
            if q_control:
                ok = k_control and same_view  # control Q -> control K: own view; never RGB K.
            elif k_control:
                ok = same_view  # RGB Q (any) -> control K: own view, any frame.
            elif not dq["noisy"] and not dk["noisy"]:
                ok = in_scope(dq, dk)  # RGB conditioning -> RGB conditioning: within scope.
            elif dq["noisy"]:
                ok = in_scope(dq, dk)  # RGB noisy -> any RGB (noisy or conditioning): within scope.
            else:  # RGB conditioning query -> RGB noisy key: never
                ok = False
            m[q, num_und + k] = ok
    return m


def _eager_block_mask(metadata: FlexMetadata, block_size: tuple[int, int]) -> BlockMask:
    """A CPU ``BlockMask`` over the metadata's ``mask_mod`` at ``block_size``, built by torch itself.

    ``build_block_mask`` derives the same mask from the metadata runs instead of from a
    dense ``[S, S]`` evaluation, so ``create_block_mask`` serves both as the reference
    it is compared against and as a cheap way for the guard tests below to obtain a real
    ``BlockMask`` of a given size.

    ``block_size`` is passed rather than assumed because it is what a backend dictates, and the
    reference has to be built at the same granularity as the mask under test.
    """
    return create_block_mask(
        _multiview_mask_mod(metadata),
        B=None,
        H=None,
        Q_LEN=metadata.q_len,
        KV_LEN=metadata.seq_len,
        device="cpu",
        BLOCK_SIZE=block_size,
    )


def _mask_mod_to_dense(metadata: FlexMetadata) -> torch.Tensor:
    """Evaluate the metadata's ``mask_mod`` on every (q, k) pair -> ``[q_len, S]`` bool."""
    mask_mod = _multiview_mask_mod(metadata)
    q_len, kv_len = metadata.q_len, metadata.seq_len
    q_idx = torch.arange(q_len).view(-1, 1).expand(q_len, kv_len)
    kv_idx = torch.arange(kv_len).view(1, -1).expand(q_len, kv_len)
    zero = torch.tensor(0)
    return mask_mod(zero, zero, q_idx, kv_idx)


def _multi_block_tokens() -> list[dict]:
    """A layout several blocks wide: 2 samples x (noisy + control) x 2 views x 7 frames.

    Each ``(item, frame, view)`` cell holds 40 tokens, so blocks straddle cell
    boundaries (partial blocks) while the leading blocks of each noisy item fall
    entirely inside the always-visible noisy region (full blocks). Both kinds have to
    appear at either geometry, which is what sets the frame count: an item spans 560
    tokens here, so whole blocks fit inside one even at FA4's 256-row query tile.
    2240 real tokens, leaving 64 padding positions in the padded sequence below.
    """
    tokens: list[dict] = []
    for sample in range(2):
        for item in range(2):
            noisy_item = item == 0
            for view in range(2):
                for frame in range(7):
                    for _ in range(40):
                        tokens.append(dict(s=sample, t=frame, v=view, noisy=noisy_item, ct=-1 if noisy_item else item))
    return tokens


# The padded GEN length that layout is used at, block-aligned for either geometry.
_MULTI_BLOCK_SEQ_LEN = math.ceil(len(_multi_block_tokens()) / _GEN_ALIGNMENT) * _GEN_ALIGNMENT


def _blocks_to_dense(
    counts: torch.Tensor,
    indices: torch.Tensor,
    num_q_blocks: int,
    num_kv_blocks: int | None = None,
) -> torch.Tensor:
    """Rebuild the ``[nQ, nKV]`` bool block matrix from FlexAttention's (count, index) form.

    Comparing masks this way ignores how the unselected indices are ordered, which is
    not part of the mask's meaning. ``num_kv_blocks`` defaults to ``num_q_blocks``, the
    square case; the FA4 block size is asymmetric.
    """
    num_kv_blocks = num_q_blocks if num_kv_blocks is None else num_kv_blocks
    dense = torch.zeros(num_q_blocks, num_kv_blocks, dtype=torch.bool)
    row_counts = counts.reshape(num_q_blocks)
    row_indices = indices.reshape(num_q_blocks, num_kv_blocks)
    for q_block in range(num_q_blocks):
        dense[q_block, row_indices[q_block, : row_counts[q_block]].long()] = True
    return dense


def _block_view(mask: torch.Tensor, block_size: tuple[int, int]) -> torch.Tensor:
    """Regroup a ``[q_len, kv_len]`` token mask into ``[nQ, nKV, q_block, kv_block]``."""
    q_block, kv_block = block_size
    q_len, kv_len = mask.shape
    return mask.view(q_len // q_block, q_block, kv_len // kv_block, kv_block).permute(0, 2, 1, 3)


@pytest.mark.L0
def test_metadata_groups_splits_runs_of_equal_metadata() -> None:
    tokens = [
        dict(s=0, t=0, v=0, noisy=True, ct=-1),
        dict(s=0, t=0, v=0, noisy=True, ct=-1),
        dict(s=0, t=1, v=0, noisy=True, ct=-1),  # new frame
        dict(s=0, t=1, v=0, noisy=False, ct=0),  # same cell, now conditioning
        dict(s=1, t=1, v=0, noisy=False, ct=0),  # new sample
    ]
    metadata = _metadata_from_tokens(tokens)
    group_id, representatives = _metadata_groups(metadata, torch.device("cpu"))

    assert torch.equal(group_id, torch.tensor([0, 0, 1, 2, 3]))
    assert torch.equal(representatives, torch.tensor([0, 2, 3, 4]))


@pytest.mark.L0
def test_metadata_groups_gives_repeated_tuples_separate_ids() -> None:
    """A tuple that reappears after another run is a second group, which is harmless."""
    tokens = [
        dict(s=0, t=0, v=0, noisy=True, ct=-1),
        dict(s=0, t=1, v=0, noisy=True, ct=-1),
        dict(s=0, t=0, v=0, noisy=True, ct=-1),  # same tuple as token 0
    ]
    group_id, representatives = _metadata_groups(_metadata_from_tokens(tokens), torch.device("cpu"))

    assert torch.equal(group_id, torch.tensor([0, 1, 2]))
    assert representatives.numel() == 3


@pytest.mark.L0
@_GEOMETRIES
def test_build_block_mask_matches_the_mask_mod_it_was_built_from(backend: FlexBackend) -> None:
    """Blocks are marked partial or full exactly as the token-level predicate says."""
    metadata = _metadata_from_tokens(_multi_block_tokens(), seq_len=_MULTI_BLOCK_SEQ_LEN)
    q_block, kv_block = backend.block_size
    num_q_blocks, num_kv_blocks = metadata.q_len // q_block, metadata.seq_len // kv_block

    block_mask = build_block_mask(metadata, torch.device("cpu"), backend.block_size)
    blocks = _block_view(_mask_mod_to_dense(metadata), backend.block_size)
    expected_full = blocks.all(dim=-1).all(dim=-1)
    expected_partial = blocks.any(dim=-1).any(dim=-1) & ~expected_full

    assert expected_full.any() and expected_partial.any(), "the case must exercise both block kinds"
    assert block_mask.full_kv_num_blocks is not None
    assert torch.equal(
        _blocks_to_dense(block_mask.kv_num_blocks, block_mask.kv_indices, num_q_blocks, num_kv_blocks),
        expected_partial,
    )
    assert torch.equal(
        _blocks_to_dense(block_mask.full_kv_num_blocks, block_mask.full_kv_indices, num_q_blocks, num_kv_blocks),
        expected_full,
    )


@pytest.mark.L0
@_GEOMETRIES
def test_build_block_mask_matches_create_block_mask(backend: FlexBackend) -> None:
    """The run-collapsed build agrees with torch's dense-evaluation builder.

    Run at both geometries because the collapse is where a coarser tile can go wrong: a block
    is full only if every run it covers is, and FA4's asymmetric 256x128 makes each block span
    twice the rows.
    """
    metadata = _metadata_from_tokens(_multi_block_tokens(), seq_len=_MULTI_BLOCK_SEQ_LEN)
    q_block, kv_block = backend.block_size
    num_q_blocks, num_kv_blocks = metadata.q_len // q_block, metadata.seq_len // kv_block

    got = build_block_mask(metadata, torch.device("cpu"), backend.block_size)
    expected = _eager_block_mask(metadata, backend.block_size)

    assert got.shape[-2:] == expected.shape[-2:]
    assert got.BLOCK_SIZE == expected.BLOCK_SIZE == backend.block_size
    assert expected.full_kv_num_blocks is not None and got.full_kv_num_blocks is not None
    assert torch.equal(
        _blocks_to_dense(got.kv_num_blocks, got.kv_indices, num_q_blocks, num_kv_blocks),
        _blocks_to_dense(expected.kv_num_blocks, expected.kv_indices, num_q_blocks, num_kv_blocks),
    )
    assert torch.equal(
        _blocks_to_dense(got.full_kv_num_blocks, got.full_kv_indices, num_q_blocks, num_kv_blocks),
        _blocks_to_dense(expected.full_kv_num_blocks, expected.full_kv_indices, num_q_blocks, num_kv_blocks),
    )


@pytest.mark.L0
def test_build_block_mask_rejects_seq_len_not_divisible_by_the_block_size() -> None:
    """A 256-row Q tile needs a 256-aligned stream, or the trailing rows have no mask entry.

    Deliberately one geometry against the other: the stream is aligned to the Triton block but
    not to FA4's coarser query tile, so it is the tile that has to be the one reported. That is
    the multiple the caller has to pad to, and the one ``FlexBackend.full_seq_alignment`` would
    have handed the packer.
    """
    seq_len = 3 * _TRITON_BACKEND.full_seq_alignment  # aligned for Triton, half a tile short for FA4
    metadata = _metadata_from_tokens(_multi_block_tokens()[:100], seq_len=seq_len)
    tile = _FLASH_BACKEND.full_seq_alignment
    with pytest.raises(ValueError, match=f"GEN sequence length, got {seq_len}, which is not a multiple of {tile}"):
        build_block_mask(metadata, torch.device("cpu"), _FLASH_BACKEND.block_size)


@pytest.mark.L0
def test_flash_backend_block_size_rejects_non_cuda_device() -> None:
    with pytest.raises(ValueError, match="CUDA device"):
        flash_backend_block_size(torch.device("cpu"))


@pytest.mark.L0
def test_resolve_flex_backend_falls_back_to_triton_where_flash_is_unavailable() -> None:
    """``auto`` is the default, so it has to degrade rather than raise. A CPU device stands in
    for any host the FA4 kernels do not exist on, since it fails the same first check."""
    backend = resolve_flex_backend(torch.device("cpu"), "auto")
    assert backend == _get_triton_flex_backend()
    assert backend.kernel_options is None  # i.e. FlexAttention's own choice, the Triton kernels
    assert (backend.full_seq_alignment, backend.causal_seq_alignment) == triton_backend_block_size()


@pytest.mark.L0
def test_resolve_flex_backend_pins_triton_without_consulting_the_host() -> None:
    """Pinning is what keeps a run comparable with one from a host that lacks the kernels, so it
    must not depend on whether this one has them."""

    def _fail(device: torch.device) -> tuple[int, int]:
        raise AssertionError("A pinned backend must not be probed for.")

    with unittest.mock.patch.object(flex_attention_module, "flash_backend_unavailable_reason", _fail):
        assert resolve_flex_backend(torch.device("cuda"), "triton") == _get_triton_flex_backend()


@pytest.mark.L0
def test_resolve_flex_backend_takes_flash_when_it_is_available() -> None:
    """The point of the default: available means used, with the mask geometry that implies.

    Both are faked, because a CPU test box has neither the kernels nor the hardware. The
    Blackwell block size is the interesting one -- 256 query rows against 128 keys is what
    makes the two alignments differ, and it is the padding, not the block size, that the
    packer is handed.
    """
    with (
        unittest.mock.patch.object(flex_attention_module, "flash_backend_unavailable_reason", lambda device: None),
        unittest.mock.patch.object(
            flex_attention_module, "flash_backend_block_size", lambda device: _FLASH_BACKEND.block_size
        ),
    ):
        backend = resolve_flex_backend(torch.device("cuda"), "auto")
    # Equality covers the kernel options too, which is what torch reads to pick the FA4 kernels,
    # and is what lets the cases above stand ``_FLASH_BACKEND`` in for a resolved backend.
    assert backend == _FLASH_BACKEND
    assert backend.full_seq_alignment == 256  # the GEN stream supplies the mask's rows
    assert backend.causal_seq_alignment == 128  # the UND stream only supplies keys


@pytest.mark.L0
def test_resolve_flex_backend_demanding_flash_reports_why_it_is_unavailable() -> None:
    """A run that pins ``flash`` wants the reason, not a silent downgrade to Triton."""
    with pytest.raises(ValueError, match="FlashAttention-4 backend, but .*CUDA device"):
        resolve_flex_backend(torch.device("cpu"), "flash")


@pytest.mark.L0
def test_resolve_flex_backend_rejects_an_unknown_preference() -> None:
    """A typo in the config would otherwise read as 'whatever the host has'."""
    with pytest.raises(ValueError, match="Unknown flex_attention_backend 'FLASH'"):
        resolve_flex_backend(torch.device("cpu"), "FLASH")


@pytest.mark.L0
@_GEOMETRIES
def test_build_block_mask_covers_padding_only_blocks(backend: FlexBackend) -> None:
    """Padding blocks stay computed: padded queries must keep a non-empty softmax."""
    q_block, kv_block = backend.block_size
    tokens = [dict(s=0, t=0, v=0, noisy=True, ct=-1)] * q_block
    metadata = _metadata_from_tokens(tokens, seq_len=2 * q_block)
    num_kv_blocks = metadata.seq_len // kv_block

    block_mask = build_block_mask(metadata, torch.device("cpu"), backend.block_size)
    assert block_mask.full_kv_num_blocks is not None
    computed = _blocks_to_dense(block_mask.kv_num_blocks, block_mask.kv_indices, 2, num_kv_blocks) | _blocks_to_dense(
        block_mask.full_kv_num_blocks, block_mask.full_kv_indices, 2, num_kv_blocks
    )
    # The first query block holds the real tokens and the second the padding; neither may see
    # the other, however many key blocks each of them spans.
    real_q = torch.tensor([True, False])
    real_kv = real_q.repeat_interleave(q_block // kv_block)
    assert torch.equal(computed, real_q.unsqueeze(-1) == real_kv.unsqueeze(0))


@pytest.mark.L0
@_GEOMETRIES
def test_build_block_mask_rejects_unaligned_seq_len(backend: FlexBackend) -> None:
    seq_len = backend.full_seq_alignment + 1
    metadata = _metadata_from_tokens([dict(s=0, t=0, v=0, noisy=True, ct=-1)], seq_len=seq_len)
    with pytest.raises(ValueError, match="block-aligned GEN sequence length"):
        build_block_mask(metadata, torch.device("cpu"), backend.block_size)


@pytest.mark.L0
def test_build_stream_sample_ids_marks_padding() -> None:
    offsets = torch.tensor([0, 3, 7], dtype=torch.long)
    sample_id = _build_stream_sample_ids(offsets, seq_len=10, device=torch.device("cpu"))
    expected = torch.tensor([0, 0, 0, 1, 1, 1, 1, -1, -1, -1], dtype=torch.long)
    assert torch.equal(sample_id, expected)


@pytest.mark.L0
def test_build_stream_sample_ids_no_padding() -> None:
    offsets = torch.tensor([0, 2, 5], dtype=torch.long)
    sample_id = _build_stream_sample_ids(offsets, seq_len=5, device=torch.device("cpu"))
    assert torch.equal(sample_id, torch.tensor([0, 0, 1, 1, 1], dtype=torch.long))


@pytest.mark.L0
def test_flex_layout_roundtrip() -> None:
    x4 = torch.randn(1, 6, 4, 8)  # [1,S,H,D]
    assert _to_flex_layout(x4).shape == (1, 4, 6, 8)  # [1,H,S,D]
    assert torch.equal(_from_flex_layout(_to_flex_layout(x4)), x4)

    x3 = torch.randn(1, 4, 6)  # [1,H,S] (LSE layout)
    assert _from_flex_layout(x3).shape == (1, 6, 4)  # [1,S,H]


def _mask_mod_captures(metadata: FlexMetadata) -> list[object]:
    """Everything the traced ``mask_mod`` closes over, with the per-side bundles opened up."""
    captured: list[object] = []
    for cell in _multiview_mask_mod(metadata).__closure__ or ():
        value = cell.cell_contents
        # The fields arrive bundled per side; what matters is what the bundles hold.
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            captured.extend(getattr(value, field.name) for field in dataclasses.fields(value))
        else:
            captured.append(value)
    return captured


@pytest.mark.L0
def test_multiview_mask_mod_captures_only_tensors() -> None:
    """The traced ``mask_mod`` must close over tensors alone, or FlashAttention-4 refuses the graph.

    Inductor lifts a captured Python int into a symbol, and rejects the FLASH lowering for
    it (``NYI: score_mod or mask_mod captures a dynamic scalar``) as soon as the enclosing
    region compiles with dynamic shapes. The shift a key-stream predicate would need
    (``q_idx + num_und``) is exactly one such int, which is why the fields are sliced per
    side instead. Symbolically shaped captures are fine by comparison -- a training run
    lowers onto that backend with the pack length symbolic -- so it is the scalars that this
    has to keep out.

    Structural because the failure is not reachable from here: the mask evaluates correctly
    either way, on CPU and on a static compile alike, so nothing short of a dynamic
    compiled run on Hopper or Blackwell would notice.
    """
    und_samples = _und_samples(12, 12, length=_UND_ALIGNMENT)
    metadata = _metadata_from_tokens(_make_multiview_tokens(), seq_len=_GEN_ALIGNMENT, und_samples=und_samples)
    assert metadata.num_und > 0  # the fused stream, i.e. the case that would need a shift

    captured = _mask_mod_captures(metadata)
    assert captured, "The mask_mod reads per-token metadata, so it has to capture something."
    scalars = [value for value in captured if not isinstance(value, torch.Tensor)]
    assert not scalars, f"mask_mod captures non-tensors the FLASH backend cannot inline: {scalars}"


@pytest.mark.L0
def test_multiview_mask_mod_captures_no_offset_views() -> None:
    """Every captured field must start at offset zero, a scalar-free capture being the point.

    Dropping the captured ``num_und`` is only half of it. The query side reads the GEN tail of
    each field, and a view of that tail carries ``num_und`` as its storage offset -- which
    Inductor lifts into the same kind of symbol the int became, so the FLASH lowering fails
    with the same ``NYI: score_mod or mask_mod captures a dynamic scalar``. The offsets are
    the observable half of that; :func:`_query_stream_fields` copies to keep them zero.

    Structural for the same reason as the test above: an offset view masks identically, so
    only a dynamic compiled run on Hopper or Blackwell would ever object.
    """
    und_samples = _und_samples(12, 12, length=_UND_ALIGNMENT)
    metadata = _metadata_from_tokens(_make_multiview_tokens(), seq_len=_GEN_ALIGNMENT, und_samples=und_samples)
    assert metadata.num_und > 0  # otherwise the tail is the whole stream and every offset is 0 anyway

    offsets = [value.storage_offset() for value in _mask_mod_captures(metadata) if isinstance(value, torch.Tensor)]
    assert offsets, "The mask_mod captures the fields as tensors, per the test above."
    assert not any(offsets), f"mask_mod captures views into a larger buffer, at offsets {offsets}"


@pytest.mark.L0
@_NOISY_SCOPES
def test_multiview_mask_mod_matches_reference(attention_scope: AttentionScope) -> None:
    tokens = _make_multiview_tokens()
    seq_len = len(tokens)
    metadata = _metadata_from_tokens(tokens, seq_len=seq_len, attention_scope=attention_scope)

    got = _mask_mod_to_dense(metadata)
    expected = _reference_visibility(tokens, seq_len, attention_scope=attention_scope)
    assert torch.equal(got, expected)


@pytest.mark.L0
def test_multiview_mask_mod_noisy_scopes_cut_the_camera_grid() -> None:
    """Each scope's noisy->noisy rule, on the smallest grid that tells the three apart.

    Four noisy tokens, one per cell of a 2-frame by 2-view grid, and one query at
    ``(t0, v0)``: the cell it sits in, the same instant in the other view, its own view at
    the other instant, and the cell that shares neither. The reference the cases above are
    held to encodes these same rules, so this is where they are stated by hand instead.
    """
    tokens = [
        dict(s=0, t=0, v=0, noisy=True, ct=-1),
        dict(s=0, t=0, v=1, noisy=True, ct=-1),
        dict(s=0, t=1, v=0, noisy=True, ct=-1),
        dict(s=0, t=1, v=1, noisy=True, ct=-1),
    ]
    seen = {
        scope: _mask_mod_to_dense(_metadata_from_tokens(tokens, attention_scope=scope))[0].tolist()
        for scope in ATTENTION_SCOPES
    }

    #                                                  (t0,v0) (t0,v1) (t1,v0) (t1,v1)
    assert seen["all_views"] == [True, True, True, True]
    assert seen["same_view"] == [True, False, True, False]
    assert seen["same_view_or_frame"] == [True, True, True, False]


@pytest.mark.L0
def test_flex_metadata_rejects_an_unknown_noisy_scope() -> None:
    """The scope is a string a config reaches, and the ``Literal`` bounding it is not enforced.

    Nothing at runtime distinguishes an unrecognised scope from a narrow one: the predicate
    reads the scope by equality, so a typo would leave both of its gates off and mask as
    ``"same_view"`` -- a quieter wrong answer than the permissive default would be.
    """
    tokens = [dict(s=0, t=0, v=0, noisy=True, ct=-1)]

    with pytest.raises(ValueError, match="Unknown attention_scope 'per_view'"):
        _metadata_from_tokens(tokens, attention_scope="per_view")


@pytest.mark.L0
def test_multiview_mask_mod_specific_rules() -> None:
    """RGB<->RGB and RGB<->control rules, under ``attention_scope="same_view"``.

    Two views, two frames, one sample. Layout index -> token:
      0: RGB cond (t0,v0)   1: RGB noisy (t0,v0)   2: control (t0,v0)
      3: RGB cond (t0,v1)                          7: control (t0,v1)
      4: RGB cond (t1,v0)   5: RGB noisy (t1,v0)    6: control (t1,v0)

    ``same_view`` is deliberately not the default ``all_views``: it is what makes the
    RGB<->RGB reach and the RGB<->control reach comparable (both view-gated), so the
    only thing distinguishing them below is that control ignores the frame RGB<->RGB
    would otherwise need ``same_view_or_frame`` to cross.
    """
    tokens = [
        dict(s=0, t=0, v=0, noisy=False),  # 0
        dict(s=0, t=0, v=0, noisy=True),  # 1
        dict(s=0, t=0, v=0, noisy=False, control=True),  # 2
        dict(s=0, t=0, v=1, noisy=False),  # 3: different view
        dict(s=0, t=1, v=0, noisy=False),  # 4: same view, different frame
        dict(s=0, t=1, v=0, noisy=True),  # 5: same view, different frame
        dict(s=0, t=1, v=0, noisy=False, control=True),  # 6: same view as control@2, different frame
        dict(s=0, t=0, v=1, noisy=False, control=True),  # 7: different view from control@2
    ]
    m = _mask_mod_to_dense(_metadata_from_tokens(tokens, attention_scope="same_view"))

    # RGB conditioning (t0,v0) reaches RGB conditioning of its own view at any frame
    # (index 4, a different frame) and control of its own view at any frame (indices
    # 2, 6), but not a different view (indices 3, 7), and never a noisy key (index 1).
    assert m[0, 0] and m[0, 4]
    assert m[0, 2] and m[0, 6]
    assert not m[0, 3] and not m[0, 7]
    assert not m[0, 1]

    # RGB noisy (t0,v0) reaches every RGB key of its own view, conditioning or noisy
    # alike, at any frame (indices 0, 1, 4, 5), and control of its own view at any
    # frame (indices 2, 6), but never the other view (indices 3, 7).
    assert m[1, 0] and m[1, 1] and m[1, 4] and m[1, 5]
    assert m[1, 2] and m[1, 6]
    assert not m[1, 3] and not m[1, 7]

    # A control query reaches only control keys of its own view, at any frame
    # (indices 2, 6), and never an RGB key or another view's control token (index 7).
    assert m[2, 2] and m[2, 6]
    assert not m[2, 7]
    for rgb_k in (0, 1, 3, 4, 5):
        assert not m[2, rgb_k], "a control query must never reach an RGB key"

    # A different view's RGB conditioning token (index 3) mirrors the same rules,
    # scoped to its own view: reaches itself and its own view's control (index 7),
    # not the other view's RGB or control tokens.
    assert m[3, 3] and m[3, 7]
    assert not m[3, 0] and not m[3, 2] and not m[3, 4] and not m[3, 6]


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


# ── the fused [UND | GEN] key stream ─────────────────────────────────────────
# GEN tokens are the queries, so the mask is rectangular: the GEN rules above cover its
# right half, and the left half is the gen->und pass, where the only rule is "same
# sample". The three UND tokens of sample 0, two of sample 1 and one padding row below
# are the miniature of what the packer's padded causal stream holds.
_UND_SAMPLES = [0, 0, 0, 1, 1, -1]


@pytest.mark.L0
def test_fused_mask_mod_matches_reference() -> None:
    tokens = _make_multiview_tokens()
    metadata = _metadata_from_tokens(tokens, und_samples=_UND_SAMPLES)

    got = _mask_mod_to_dense(metadata)
    assert got.shape == (len(tokens), len(_UND_SAMPLES) + len(tokens))
    assert torch.equal(got, _reference_visibility(tokens, len(tokens), _UND_SAMPLES))


@pytest.mark.L0
def test_fused_mask_mod_gives_every_gen_token_the_und_stream_of_its_own_sample() -> None:
    """Conditioning and noisy GEN tokens alike read the whole caption, and only theirs."""
    tokens = _make_multiview_tokens()
    num_und = len(_UND_SAMPLES)
    metadata = _metadata_from_tokens(tokens, seq_len=len(tokens) + 2, und_samples=_UND_SAMPLES)
    m = _mask_mod_to_dense(metadata)

    cond_of_sample_0 = 0  # cond-A (t0,v0), see _make_multiview_tokens
    noisy_of_sample_0 = 2
    cond_of_sample_1 = 16
    assert not metadata.is_noisy[num_und + cond_of_sample_0]
    assert metadata.is_noisy[num_und + noisy_of_sample_0]

    for gen_row in (cond_of_sample_0, noisy_of_sample_0):
        assert m[gen_row, :3].all(), "a GEN token reads every UND token of its sample"
        assert not m[gen_row, 3:num_und].any(), "and no UND token of another sample, nor UND padding"
    assert m[cond_of_sample_1, 3:5].all()
    assert not m[cond_of_sample_1, :3].any()

    # Padded GEN queries keep the UND padding to attend to, so their softmax is non-empty.
    assert m[len(tokens) :, 5].all()
    assert not m[len(tokens) :, :5].any()


@pytest.mark.L0
@_GEOMETRIES
def test_build_block_mask_is_rectangular_over_a_fused_stream(backend: FlexBackend) -> None:
    """The block mask spans q_len by kv_len, and agrees with torch's dense builder.

    At the FA4 geometry this is also the case for the two padding multiples that backend asks
    for on Blackwell: the GEN stream is padded to the 256-row query block while the UND stream
    stops at the 128-wide key block, the one combination where :class:`FlexBackend`'s two
    alignments differ. Everything about it that does not need the kernels is checked here,
    because the mask is built on CPU while the kernels that consume it are not.
    """
    q_len = _MULTI_BLOCK_SEQ_LEN
    # One key block of UND tokens: two samples' worth of real ones, then padding.
    und_samples = _und_samples(60, 60, length=_UND_ALIGNMENT)
    metadata = _metadata_from_tokens(_multi_block_tokens(), seq_len=q_len, und_samples=und_samples)
    q_block, kv_block = backend.block_size
    # What flex_attention requires of the pair, and the reason the UND stream may stop at the
    # narrower multiple: only the queries are tiled 256 at a time.
    assert metadata.q_len % q_block == 0 and metadata.seq_len % kv_block == 0
    assert metadata.num_und == _UND_ALIGNMENT, "the UND prefix is one key block, so the mask is rectangular"
    num_q_blocks, num_kv_blocks = q_len // q_block, metadata.seq_len // kv_block

    got = build_block_mask(metadata, torch.device("cpu"), backend.block_size)
    expected = _eager_block_mask(metadata, backend.block_size)

    assert got.BLOCK_SIZE == backend.block_size
    assert got.shape[-2:] == (q_len, metadata.seq_len) == expected.shape[-2:]
    assert expected.full_kv_num_blocks is not None and got.full_kv_num_blocks is not None
    assert torch.equal(
        _blocks_to_dense(got.kv_num_blocks, got.kv_indices, num_q_blocks, num_kv_blocks),
        _blocks_to_dense(expected.kv_num_blocks, expected.kv_indices, num_q_blocks, num_kv_blocks),
    )
    assert torch.equal(
        _blocks_to_dense(got.full_kv_num_blocks, got.full_kv_indices, num_q_blocks, num_kv_blocks),
        _blocks_to_dense(expected.full_kv_num_blocks, expected.full_kv_indices, num_q_blocks, num_kv_blocks),
    )


@pytest.mark.L0
@_GEOMETRIES
def test_build_block_mask_rejects_an_unaligned_und_stream(backend: FlexBackend) -> None:
    """The UND/GEN boundary has to fall on a block boundary, not just the total length."""
    seq_len = backend.full_seq_alignment
    metadata = _metadata_from_tokens(
        _multi_block_tokens()[:seq_len],
        seq_len=seq_len,
        und_samples=[0] * (backend.causal_seq_alignment + 1),
    )
    with pytest.raises(ValueError, match="block-aligned UND sequence length"):
        build_block_mask(metadata, torch.device("cpu"), backend.block_size)


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
        num_items_per_sample=[1],
        token_shapes=[(4, 2, 1)],
        num_views_per_item=[2],
        condition_frames=[[0, 2]],
        pad=4,
    ),
    # Transfer layout: every sample carries a fully conditioning control item beside a
    # partly noisy RGB target, the WSM pairing the network reports as one control item and
    # one target per sample.
    "transfer_two_items_two_samples": dict(
        num_items_per_sample=[2, 2],
        token_shapes=[(4, 1, 2), (4, 1, 2), (2, 2, 1), (2, 2, 1)],
        num_views_per_item=[2, 2, 1, 1],
        condition_frames=[[0, 1, 2, 3], [0], [0, 1], []],
        is_control_per_item=[True, False, True, False],
        pad=5,
    ),
    # Degenerate single-frame single-view item, exactly filling the sequence.
    "single_frame_single_view": dict(
        num_items_per_sample=[1],
        token_shapes=[(1, 3, 3)],
        num_views_per_item=[1],
        condition_frames=[[]],
        pad=0,
    ),
    # Joint camera + LiDAR: one sample owning a camera pair on a 3-frame grid and a range
    # pair on a 4-frame one, each pair a fully conditioning control item beside its noisy
    # target, as JointCamLidarToTrainingFormat emits them. The range pair sits on view 1,
    # which is how the LiDAR sweeps stay off the camera's grid; the two grids differ in
    # both length and spatial extent, which is the case a per-sample grid check rejects.
    "joint_camera_and_lidar": dict(
        num_items_per_sample=[4],
        token_shapes=[(3, 1, 2), (3, 1, 2), (4, 2, 1), (4, 2, 1)],
        num_views_per_item=[1, 1, 1, 1],
        condition_frames=[[0, 1, 2], [], [0, 1, 2, 3], []],
        view_offsets_per_item=[0, 0, 1, 1],
        # Control is marked per stream: each sensor's first item conditions its second, so
        # a camera item's position never decides a LiDAR item's role.
        is_control_per_item=[True, False, True, False],
        pad=4,
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


def _multiview_mask_items_for_test(packed_seq: PackedSequence) -> list[list[MaskItem]]:
    pytest.importorskip("transformers", reason="cosmos3_vfm_network requires the Cosmos3 network dependencies.")
    from cosmos_framework.model.generator.mot.cosmos3_vfm_network import _multiview_mask_items

    return _multiview_mask_items(packed_seq)


def _mask_items_pack(*, num_views: int, with_lidar: bool, with_view_metadata: bool = True) -> PackedSequence:
    """A one-sample pack of two camera items, optionally beside two LiDAR items."""

    def stream(token_shapes: list[tuple[int, int, int]]) -> ModalityData:
        return ModalityData(
            tokens=[torch.zeros(1) for _ in token_shapes],  # list[[1]]
            token_shapes=token_shapes,
            condition_mask=[torch.ones(shape[0]) for shape in token_shapes],  # list[[T]]
        )

    camera_shape = (2 * num_views, 1, 1)
    return PackedSequence(
        sample_lens=[1],
        vision=stream([camera_shape, camera_shape]),
        num_vision_items_per_sample=[2],
        num_views_per_vision_item=[num_views, num_views] if with_view_metadata else None,
        lidar=stream([(3, 1, 1), (3, 1, 1)]) if with_lidar else None,
        num_lidar_items_per_sample=[2] if with_lidar else None,
    )


@pytest.mark.L0
def test_mask_items_leave_a_camera_only_pack_on_view_zero() -> None:
    """No LiDAR means every item stays on view 0, which keeps the camera mask untouched."""
    items = _multiview_mask_items_for_test(_mask_items_pack(num_views=2, with_lidar=False))

    assert [len(sample_items) for sample_items in items] == [2]
    assert [item.num_views for item in items[0]] == [2, 2]
    assert [item.view_offset for item in items[0]] == [0, 0]
    assert [item.is_control for item in items[0]] == [True, False]


@pytest.mark.L0
def test_mask_items_place_the_lidar_stream_past_the_widest_camera_item() -> None:
    """A six-camera rig owns views 0-5, so the range clips start at view 6, not view 1."""
    items = _multiview_mask_items_for_test(_mask_items_pack(num_views=6, with_lidar=True))

    # The sample reads [camera, camera, lidar, lidar], the order the packer lays down.
    assert [len(sample_items) for sample_items in items] == [4]
    assert [item.num_views for item in items[0]] == [6, 6, 1, 1]
    assert [item.view_offset for item in items[0]] == [0, 0, 6, 6]
    assert [item.latent_t for item in items[0]] == [12, 12, 3, 3]
    # Per stream: each sensor's first item conditions its second.
    assert [item.is_control for item in items[0]] == [True, False, True, False]


@pytest.mark.L0
def test_mask_items_read_a_joint_pack_without_per_camera_metadata_as_single_camera() -> None:
    """The joint recipe's camera clips skip the per-camera encode path, so they carry no counts."""
    items = _multiview_mask_items_for_test(_mask_items_pack(num_views=1, with_lidar=True, with_view_metadata=False))

    assert [item.num_views for item in items[0]] == [1, 1, 1, 1]
    assert [item.view_offset for item in items[0]] == [0, 0, 1, 1]
    assert [item.is_control for item in items[0]] == [True, False, True, False]


@pytest.mark.L0
def test_mask_items_reject_a_camera_only_pack_without_per_camera_metadata() -> None:
    """Without a second stream to vouch for one view per item, a missing count stays an error."""
    with pytest.raises(ValueError, match="per-camera VAE metadata"):
        _multiview_mask_items_for_test(_mask_items_pack(num_views=1, with_lidar=False, with_view_metadata=False))


@pytest.mark.L0
def test_mask_items_accept_a_lidar_only_pack() -> None:
    """A range-only attention batch has no cameras, so every item stays on view 0."""
    lidar = ModalityData(
        tokens=[torch.zeros(1), torch.zeros(1)],  # list[[1]]
        token_shapes=[(3, 1, 1), (3, 1, 1)],
        condition_mask=[torch.ones(3), torch.ones(3)],  # list[[3]]
    )
    items = _multiview_mask_items_for_test(
        PackedSequence(
            sample_lens=[1],
            vision=None,
            lidar=lidar,
            num_lidar_items_per_sample=[2],
        )
    )

    assert [len(sample_items) for sample_items in items] == [2]
    assert [item.num_views for item in items[0]] == [1, 1]
    assert [item.view_offset for item in items[0]] == [0, 0]
    assert [item.is_control for item in items[0]] == [True, False]
    assert [item.latent_t for item in items[0]] == [3, 3]


@pytest.mark.L0
def test_mask_items_reject_a_pack_with_neither_stream() -> None:
    with pytest.raises(ValueError, match="vision or LiDAR"):
        _multiview_mask_items_for_test(PackedSequence(sample_lens=[1], vision=None, lidar=None))


def _case_offsets(case: dict) -> torch.Tensor:
    """Per-sample cumulative GEN offsets, ``int32`` as the packer emits them."""
    offsets = [0]
    item_idx = 0
    for num_items in case["num_items_per_sample"]:
        sample_tokens = 0
        for _ in range(num_items):
            latent_t, patch_h, patch_w = case["token_shapes"][item_idx]
            sample_tokens += latent_t * patch_h * patch_w
            item_idx += 1
        offsets.append(offsets[-1] + sample_tokens)
    return torch.tensor(offsets, dtype=torch.int32)


def _expected_tokens(case: dict) -> list[dict]:
    """Per-token descriptors derived independently from the camera-major layout.

    ``is_control_per_item`` and ``view_offsets_per_item`` are read straight off the case,
    as :func:`build_multiview_flex_metadata` reads them off its arguments; both default to
    the single-stream, no-control layout when a case omits them.
    """
    tokens: list[dict] = []
    item_idx = 0
    num_case_items = len(case["token_shapes"])
    view_offsets = case.get("view_offsets_per_item") or [0] * num_case_items
    is_control = case.get("is_control_per_item") or [False] * num_case_items
    for sample, num_items in enumerate(case["num_items_per_sample"]):
        for _ in range(num_items):
            is_control_item = is_control[item_idx]
            latent_t, patch_h, patch_w = case["token_shapes"][item_idx]
            num_views = case["num_views_per_item"][item_idx]
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
                                v=view_offsets[item_idx] + view,
                                noisy=not is_cond,
                                control=is_control_item,
                            )
                        )
            item_idx += 1
    return tokens


def _case_items(case: dict) -> list[list[MaskItem]]:
    """The case's flat per-item columns as the nested ``MaskItem`` lists the builder takes.

    The cases stay column-shaped because that reads better as table data; this is the one
    place that transposes them, so a case omitting ``view_offsets_per_item`` or
    ``is_control_per_item`` gets those fields' own defaults.
    """
    num_items = len(case["token_shapes"])
    view_offsets = case.get("view_offsets_per_item") or [0] * num_items
    is_control = case.get("is_control_per_item") or [False] * num_items
    items = [
        MaskItem(
            token_shape=case["token_shapes"][idx],
            condition_mask=_condition_mask(case["token_shapes"][idx][0], case["condition_frames"][idx]),
            num_views=case["num_views_per_item"][idx],
            view_offset=view_offsets[idx],
            is_control=is_control[idx],
        )
        for idx in range(num_items)
    ]
    items_per_sample: list[list[MaskItem]] = []
    first_item = 0
    for sample_num_items in case["num_items_per_sample"]:
        items_per_sample.append(items[first_item : first_item + sample_num_items])
        first_item += sample_num_items
    return items_per_sample


def _build_case_metadata(case: dict, attention_scope: AttentionScope = "all_views") -> tuple[FlexMetadata, int]:
    """Run the builder on a case; returns the metadata and the real token count."""
    offsets = _case_offsets(case)
    num_real = int(offsets[-1])
    metadata = build_multiview_flex_metadata(
        seq_len=num_real + case["pad"],
        full_q_offsets=offsets,
        items_per_sample=_case_items(case),
        device=torch.device("cpu"),
        attention_scope=attention_scope,
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
    # A single-item sample has no control item: is_control is all False.
    assert not metadata.is_control.any()
    assert torch.equal(metadata.sample_id, torch.tensor([0] * 8 + [-1] * 4))

    assert metadata.sample_id.dtype == torch.long
    assert metadata.frame_id.dtype == torch.long
    assert metadata.view_id.dtype == torch.long
    assert metadata.is_noisy.dtype == torch.bool
    assert metadata.is_control.dtype == torch.bool


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
    assert torch.equal(metadata.is_control, expected.is_control)


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
    assert not metadata.is_noisy[tail].any()
    assert not metadata.is_control[tail].any()

    m = _mask_mod_to_dense(metadata)
    assert not m[:num_real, num_real:].any()
    assert not m[num_real:, :num_real].any()
    # Padded queries attend to padding only, which keeps their softmax non-empty.
    assert m[num_real:, num_real:].all()


@pytest.mark.L0
def test_build_multiview_flex_metadata_reaches_the_control_item_by_view() -> None:
    """RGB tokens see the control item's tokens at the same view, any frame."""
    metadata, _ = _build_case_metadata(_MULTIVIEW_CASES["transfer_two_items_two_samples"])
    m = _mask_mod_to_dense(metadata)

    # Sample 0 holds two 2-view x 2-frame x 2-spatial items: the fully conditioning
    # control item at tokens 0..7 (item 0, positionally first) and the RGB target,
    # conditioning only at (v0,f0), at tokens 8..15 (item 1, positionally last).
    control_v0f0 = 0
    control_v0f1 = 2
    control_v1f0 = 4
    target_cond_v0f0 = 8
    target_noisy_v0f1 = 8 + 2
    assert metadata.is_control[control_v0f0] and metadata.is_control[control_v0f1]
    assert not metadata.is_control[target_cond_v0f0] and not metadata.is_control[target_noisy_v0f1]
    assert not metadata.is_noisy[target_cond_v0f0]
    assert metadata.is_noisy[target_noisy_v0f1]

    # A noisy RGB target token reaches control of its own view at any frame --
    # v0f0 (a different frame from the query) as well as v0f1 (the same frame) --
    # but not a different view's control token.
    assert m[target_noisy_v0f1, control_v0f0]
    assert m[target_noisy_v0f1, control_v0f1]
    assert not m[target_noisy_v0f1, control_v1f0], "a different view's control token must stay masked"

    # A conditioning RGB target token reaches control the same way, but never a
    # noisy RGB key of its own item.
    assert m[target_cond_v0f0, control_v0f0] and m[target_cond_v0f0, control_v0f1]
    assert not m[target_cond_v0f0, target_noisy_v0f1], "conditioning queries never attend to noisy keys"

    # Control never reaches RGB, in either direction.
    assert not m[control_v0f0, target_cond_v0f0]
    assert not m[control_v0f0, target_noisy_v0f1]


@pytest.mark.L0
def test_build_multiview_flex_metadata_expands_the_control_flags_over_each_items_tokens() -> None:
    """``is_control_per_item`` is per item; the field it drives is per token."""
    metadata, _ = _build_case_metadata(_MULTIVIEW_CASES["transfer_two_items_two_samples"])

    # Sample 0: control item over tokens 0..7, target over 8..15. Sample 1: control over
    # 16..19, target over 20..23.
    assert metadata.is_control[0:8].all()
    assert not metadata.is_control[8:16].any()
    assert metadata.is_control[16:20].all()
    assert not metadata.is_control[20:24].any()


@pytest.mark.L0
def test_build_multiview_flex_metadata_defaults_to_no_control_items() -> None:
    """Omitting ``is_control_per_item`` marks nothing, rather than inferring from item order.

    The transfer case's own flags say its first item of each sample is control; dropping
    them has to leave every token an RGB one, so a caller that never mentions control
    streams cannot acquire them from the way its items happen to be ordered.
    """
    case = dict(_MULTIVIEW_CASES["transfer_two_items_two_samples"])
    case.pop("is_control_per_item")

    metadata, _ = _build_case_metadata(case)

    assert not metadata.is_control.any()


@pytest.mark.L0
def test_build_multiview_flex_metadata_accepts_flat_bool_condition_mask() -> None:
    """A flat bool mask must be equivalent to the packer's ``[T,1,1]`` float mask."""
    case = _MULTIVIEW_CASES["two_views_first_frame_cond"]
    metadata, _ = _build_case_metadata(case)

    from_bool = build_multiview_flex_metadata(
        seq_len=metadata.seq_len,
        full_q_offsets=_case_offsets(case),
        items_per_sample=[
            [
                MaskItem(
                    token_shape=case["token_shapes"][0],
                    condition_mask=torch.tensor([True, False, True, False]),
                    num_views=case["num_views_per_item"][0],
                )
            ]
        ],
        device=torch.device("cpu"),
    )
    assert torch.equal(from_bool.is_noisy, metadata.is_noisy)
    assert torch.equal(from_bool.is_control, metadata.is_control)


@pytest.mark.L0
@pytest.mark.parametrize("num_views", [0, 3])
def test_mask_item_rejects_bad_view_count(num_views: int) -> None:
    """An item whose latent axis does not divide into its views is malformed on its own."""
    with pytest.raises(ValueError, match="not divisible by num_views"):
        MaskItem(token_shape=(4, 1, 2), condition_mask=_condition_mask(4, []), num_views=num_views)


@pytest.mark.L0
def test_mask_item_rejects_condition_mask_length() -> None:
    """A mask that does not cover the item's latent axis is malformed on its own."""
    with pytest.raises(ValueError, match="expected 4"):
        MaskItem(token_shape=(4, 1, 2), condition_mask=_condition_mask(3, []), num_views=2)


@pytest.mark.L0
def test_build_multiview_flex_metadata_rejects_packed_token_count_mismatch() -> None:
    with pytest.raises(ValueError, match="packed full-attention splits disagree"):
        build_multiview_flex_metadata(
            seq_len=16,
            full_q_offsets=torch.tensor([0, 7], dtype=torch.int32),  # item contributes 8
            items_per_sample=[[MaskItem(token_shape=(4, 1, 2), condition_mask=_condition_mask(4, []), num_views=2)]],
            device=torch.device("cpu"),
        )


@pytest.mark.L0
def test_build_multiview_flex_metadata_rejects_mixed_grids_in_a_sample() -> None:
    with pytest.raises(ValueError, match=r"same \(num_views, frames_per_view\) grid"):
        build_multiview_flex_metadata(
            seq_len=32,
            full_q_offsets=torch.tensor([0, 16], dtype=torch.int32),
            # Same token count, but 2 views x 2 frames against 1 view x 4 frames.
            items_per_sample=[
                [
                    MaskItem(token_shape=(4, 1, 2), condition_mask=_condition_mask(4, [0]), num_views=2),
                    MaskItem(token_shape=(4, 1, 2), condition_mask=_condition_mask(4, [0, 1, 2, 3]), num_views=1),
                ]
            ],
            device=torch.device("cpu"),
        )


@pytest.mark.L0
def test_build_multiview_flex_metadata_rejects_mixed_grids_within_one_view_range() -> None:
    """Items on their own views may disagree on the grid; items sharing views may not.

    The relaxation that lets a camera pair sit beside a range pair has to stay confined to
    the view boundary: on one view, conditioning still reaches noisy by matching the frame,
    so a control item on another grid would condition the wrong moments silently.
    """
    with pytest.raises(ValueError, match=r"same \(num_views, frames_per_view\) grid"):
        build_multiview_flex_metadata(
            seq_len=64,
            full_q_offsets=torch.tensor([0, 48], dtype=torch.int32),
            # The first two share view offset 0 and disagree: 2 views x 2 frames against 1 x 4.
            items_per_sample=[
                [
                    MaskItem(token_shape=(4, 1, 2), condition_mask=_condition_mask(4, [0]), num_views=2),
                    MaskItem(token_shape=(4, 1, 2), condition_mask=_condition_mask(4, [0, 1, 2, 3]), num_views=1),
                    MaskItem(token_shape=(4, 1, 2), condition_mask=_condition_mask(4, []), num_views=1, view_offset=2),
                ]
            ],
            device=torch.device("cpu"),
        )


@pytest.mark.L0
def test_build_multiview_flex_metadata_keeps_a_second_sensor_off_the_camera_grid() -> None:
    """The joint layout: a camera pair on view 0 and a range pair on view 1.

    Both are what the joint recipe packs and what a per-sample grid check rejected, the two
    pairs running to different lengths. The view offset is what stops the rules that match on
    ``(frame, view)`` from pairing a camera latent with the sweep of the same frame index --
    1.33 s against 1.0 s, the streams running at 7.5 and 10 Hz.
    """
    case = _MULTIVIEW_CASES["joint_camera_and_lidar"]
    metadata, num_real = _build_case_metadata(case)
    camera_tokens = 2 * (3 * 2)

    assert num_real == camera_tokens + 2 * (4 * 2)
    assert torch.equal(
        metadata.view_id[:num_real], torch.tensor([0] * camera_tokens + [1] * (num_real - camera_tokens))
    )
    # Each pair numbers its own frames, so both runs start at 0 and stop at their own length.
    assert metadata.frame_id[:camera_tokens].max() == 2
    assert metadata.frame_id[camera_tokens:num_real].max() == 3

    m = _mask_mod_to_dense(metadata)
    camera_noisy = int(metadata.is_noisy[:camera_tokens].nonzero()[0])
    lidar_control = int((~metadata.is_noisy[camera_tokens:num_real]).nonzero()[0]) + camera_tokens
    lidar_noisy = int(metadata.is_noisy[camera_tokens:num_real].nonzero()[0]) + camera_tokens
    assert m[camera_noisy, lidar_noisy], "the sensors couple through their noisy tokens"
    assert not m[camera_noisy, lidar_control], "and not through the other sensor's control clip"


@pytest.mark.L0
def test_joint_camera_lidar_rejects_factorized_same_view_or_frame() -> None:
    """same_view_or_frame pairs tokens by frame index; camera and LiDAR do not share one."""
    case = _MULTIVIEW_CASES["joint_camera_and_lidar"]
    with pytest.raises(ValueError, match="same_view_or_frame"):
        _build_case_metadata(case, attention_scope="same_view_or_frame")


@pytest.mark.L0
def test_build_multiview_flex_metadata_without_view_offsets_starts_every_item_at_view_zero() -> None:
    """Omitting the offsets is the same batch as passing zero for every item.

    This is the invariance that keeps every camera-only recipe on the mask it trained with:
    the argument is new, and a pack that leaves it out has to build what it built before.
    """
    case = _MULTIVIEW_CASES["transfer_two_items_two_samples"]
    without_offsets, _ = _build_case_metadata(case)
    with_zero_offsets, _ = _build_case_metadata({**case, "view_offsets_per_item": [0] * len(case["token_shapes"])})

    assert torch.equal(without_offsets.view_id, with_zero_offsets.view_id)
    assert torch.equal(_mask_mod_to_dense(without_offsets), _mask_mod_to_dense(with_zero_offsets))


@pytest.mark.L0
def test_build_multiview_flex_metadata_prepends_the_und_stream() -> None:
    """With ``num_und`` the fields cover ``[UND | GEN]``: sample ids left, sentinels elsewhere."""
    case = _MULTIVIEW_CASES["transfer_two_items_two_samples"]
    gen_only, num_real = _build_case_metadata(case)
    num_und = 8

    fused = build_multiview_flex_metadata(
        seq_len=gen_only.seq_len,
        full_q_offsets=_case_offsets(case),
        items_per_sample=_case_items(case),
        device=torch.device("cpu"),
        num_und=num_und,
        causal_offsets=torch.tensor([0, 3, 5], dtype=torch.int32),  # 3 + 2 real UND tokens, 3 padded
    )

    assert (fused.num_und, fused.q_len, fused.seq_len) == (num_und, gen_only.seq_len, num_und + gen_only.seq_len)
    assert torch.equal(fused.sample_id[:num_und], torch.tensor([0, 0, 0, 1, 1, -1, -1, -1]))
    for field in (fused.frame_id, fused.view_id):
        assert torch.equal(field[:num_und], torch.full((num_und,), -1))
    assert not fused.is_noisy[:num_und].any()
    assert not fused.is_control[:num_und].any()
    # The GEN half is untouched by the prefix.
    assert torch.equal(fused.sample_id[num_und:], gen_only.sample_id)
    assert torch.equal(fused.frame_id[num_und:], gen_only.frame_id)
    assert torch.equal(fused.is_noisy[num_und:], gen_only.is_noisy)

    # Real GEN tokens read their own sample's UND tokens and nothing else on that side.
    m = _mask_mod_to_dense(fused)
    assert m[0, :3].all() and not m[0, 3:num_und].any()
    assert m[num_real - 1, 3:5].all() and not m[num_real - 1, :3].any()


@pytest.mark.L0
def test_build_multiview_flex_metadata_requires_causal_offsets_for_a_fused_stream() -> None:
    with pytest.raises(ValueError, match="needs causal_offsets"):
        build_multiview_flex_metadata(
            seq_len=16,
            full_q_offsets=torch.tensor([0, 8], dtype=torch.int32),
            items_per_sample=[[MaskItem(token_shape=(4, 1, 2), condition_mask=_condition_mask(4, []), num_views=2)]],
            device=torch.device("cpu"),
            num_und=8,
        )


@pytest.mark.L0
def test_build_multiview_flex_metadata_rejects_overlong_metadata() -> None:
    with pytest.raises(ValueError, match="exceeding GEN sequence length"):
        build_multiview_flex_metadata(
            seq_len=4,  # smaller than the 8 tokens the item contributes
            full_q_offsets=torch.tensor([0, 8], dtype=torch.int32),
            items_per_sample=[[MaskItem(token_shape=(4, 1, 2), condition_mask=_condition_mask(4, []), num_views=2)]],
            device=torch.device("cpu"),
        )


@pytest.mark.L0
@_GEOMETRIES
def test_flex_attention_rejects_unaligned_seq_len(backend: FlexBackend) -> None:
    seq_len = backend.full_seq_alignment + 1  # not a multiple of the backend's query block
    q = torch.randn(1, seq_len, 2, 16)
    meta = _metadata_from_tokens([dict(s=0, t=0, v=0, noisy=True, ct=-1)], seq_len=seq_len)
    with pytest.raises(ValueError, match="block-aligned GEN sequence length"):
        flex_attention(q, q, q, _eager_block_mask(meta, backend.block_size), backend)


@pytest.mark.L0
@_GEOMETRIES
def test_flex_attention_rejects_seq_len_mismatch(backend: FlexBackend) -> None:
    """A mask built for another pack must be rejected, not silently applied."""
    seq_len = backend.full_seq_alignment
    q = torch.randn(1, seq_len, 2, 8)  # one query block of GEN tokens
    # Mask for a two-block pack, i.e. one that q/k/v did not come from.
    other_pack = _metadata_from_tokens(
        [dict(s=0, t=0, v=0, noisy=True, ct=-1)],
        seq_len=2 * seq_len,
    )
    with pytest.raises(ValueError, match="but the GEN sequence is"):
        flex_attention(q, q, q, _eager_block_mask(other_pack, backend.block_size), backend)


@pytest.mark.L0
def test_flex_attention_rejects_a_mask_built_for_another_backend() -> None:
    """A mask at a granularity other than the backend's is the one mismatch nothing else reports.

    FlashAttention-4 drives its tile scheduler from the block mask, so a mask finer than the
    blocks it steps makes it attend to the wrong tokens rather than raise. Both lengths here are
    block-aligned and the mask covers the right pack, so the block size is all that disagrees --
    which is why this case is the one pairing of the two geometries rather than a parametrization
    over them.
    """
    seq_len = _FLASH_BACKEND.full_seq_alignment  # aligned for both, so only the granularity differs
    q = torch.randn(1, seq_len, 2, 8)
    meta = _metadata_from_tokens([dict(s=0, t=0, v=0, noisy=True, ct=-1)], seq_len=seq_len)
    with pytest.raises(ValueError, match="but the flash backend steps"):
        flex_attention(q, q, q, _eager_block_mask(meta, _TRITON_BACKEND.block_size), _FLASH_BACKEND)


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
def test_flex_attention_matches_reference(num_kv_heads: int, return_lse: bool) -> None:
    torch.manual_seed(0)
    torch.compiler.reset()
    device = "cuda"
    dtype = torch.float32
    num_q_heads = 4
    head_dim = 32
    seq_len = _TRITON_BACKEND.full_seq_alignment  # single block

    tokens = _make_multiview_tokens()
    n_real = len(tokens)
    metadata = _metadata_from_tokens(tokens, seq_len=seq_len, device=device)
    block_mask = build_block_mask(metadata, torch.device(device), _TRITON_BACKEND.block_size)

    q = torch.randn(1, seq_len, num_q_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(1, seq_len, num_kv_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(1, seq_len, num_kv_heads, head_dim, device=device, dtype=dtype)

    result = flex_attention(q, k, v, block_mask, _TRITON_BACKEND, return_lse=return_lse)
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
def test_flex_attention_fused_gradients_match_dense_reference() -> None:
    """The fused call must match a dense attention over ``[UND | GEN]``, forward and backward.

    This is what ``two_way_attention``'s full branch computes: one kernel, one softmax, GEN
    queries against the concatenated key stream. The gradients are the point of the test.
    The alternative -- running the two quadrants separately and recombining them with
    ``merge_attentions`` -- gives the same forward but not the same backward, because NATTEN's
    merge repairs the branch gradients by overwriting the ``(out, lse)`` storage the attention
    kernel saved, and the heads-last conversion in ``flex_attention`` hands it copies
    rather than views (the transpose has to be made contiguous for the merge to accept it).
    A branch then differentiates through its *local* softmax normalization against an
    unweighted upstream gradient, which the earlier merged version of this test caught as an
    O(1) error on ``dq`` while its forward assertion passed. Fusing removes the contract
    instead of repairing it.
    """
    torch.manual_seed(0)
    torch.compiler.reset()
    device = "cuda"
    num_heads, head_dim = 4, 32
    seq_len = _TRITON_BACKEND.full_seq_alignment  # one block of GEN tokens
    und_samples = _und_samples(12, 12, length=_TRITON_BACKEND.causal_seq_alignment)  # one block of UND tokens

    tokens = _make_multiview_tokens()
    metadata = _metadata_from_tokens(tokens, seq_len=seq_len, device=device, und_samples=und_samples)
    block_mask = build_block_mask(metadata, torch.device(device), _TRITON_BACKEND.block_size)
    mask = _mask_mod_to_dense(metadata).to(device)  # [seq_len, num_und+seq_len] bool

    generator = torch.Generator(device=device).manual_seed(7)

    def _leaf(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, device=device, dtype=torch.float32, generator=generator).requires_grad_(True)

    q = _leaf(1, seq_len, num_heads, head_dim)  # [1,N_full,H,D]
    k = _leaf(1, metadata.seq_len, num_heads, head_dim)  # [1,N_und+N_full,H,D]
    v = _leaf(1, metadata.seq_len, num_heads, head_dim)
    leaves = [q, k, v]

    # Every query row carries a loss, padding included: the reference runs the very same
    # mask, so padded rows are not a special case here.
    grad_seed = torch.randn(1, seq_len, num_heads, head_dim, device=device, generator=generator)

    got = cast(torch.Tensor, flex_attention(q, k, v, block_mask, _TRITON_BACKEND))
    expected, _lse = _reference_attention(q, k, v, mask)
    assert got.shape == expected.shape == (1, seq_len, num_heads, head_dim)
    torch.testing.assert_close(got, expected, atol=2e-2, rtol=2e-2)

    # autograd.grad rather than .backward() so the two graphs never share a .grad accumulator.
    got_grads = torch.autograd.grad((got * grad_seed).sum(), leaves, materialize_grads=True)
    expected_grads = torch.autograd.grad((expected * grad_seed).sum(), leaves, materialize_grads=True)
    for name, got_grad, expected_grad in zip(("dq", "dk", "dv"), got_grads, expected_grads):
        torch.testing.assert_close(got_grad, expected_grad, atol=2e-2, rtol=2e-2, msg=lambda m, n=name: f"{n}: {m}")


# Failure text that means this box cannot lower FlexAttention onto FlashAttention-4 at all,
# as opposed to the backend computing something wrong. resolve_flex_backend has already
# ruled out the missing package and the wrong GPU by the time these run, which leaves what
# only the compiler can reject: a torch whose CuTeDSL codegen disagrees with the installed
# kernels, or an Inductor check on the traced graph itself. Neither is visible before the
# graph is compiled, so the call under test doubles as the last availability probe.
#
# attention_test imports this rather than keeping its own copy: the two files wrap it in
# different guards, but what counts as "the backend is unavailable here" is one answer, and a
# new torch error string has to reach both at once or one suite starts failing what the other
# skips.
_FLASH_UNAVAILABLE_MARKERS = ("BACKEND", "FLASH", "flash_attn", "cutlass", "cute", "CuTe")


@contextlib.contextmanager
def _flash_backend_or_skip() -> Iterator[None]:
    """Skip the test if FlashAttention-4 cannot be lowered here, rather than failing it.

    Anything that does not name the backend is re-raised: a wrong *result*, a rejected
    shape or a mask the kernel disagrees with all have to fail. ``dLSE`` is re-raised too --
    asking the FA4 backward to differentiate the log-sum-exp is a misuse of the API these
    tests avoid, so seeing it means a test regressed, not that the box is short a package.
    """
    try:
        yield
    except Exception as e:  # noqa: BLE001 - narrowed by the marker check below
        message = f"{type(e).__name__}: {e}"
        if "dLSE" in message or not any(marker in message for marker in _FLASH_UNAVAILABLE_MARKERS):
            raise
        pytest.skip(f"FlexAttention cannot lower onto the FlashAttention-4 backend here -- {message}")


def _flash_fused_case(device: torch.device) -> tuple[FlexMetadata, BlockMask, FlexBackend]:
    """The fused ``[UND | GEN]`` case of the tests above, sized for the FlashAttention-4 backend.

    Two things differ from the Triton version. The mask is built at
    :func:`flash_backend_block_size`, because FA4 drives its tile scheduler from the block
    mask and cannot express a finer one; and the GEN stream is padded to that coarser query
    block, 256 rows on Blackwell against 128 on Hopper. The token layout is otherwise
    identical, which is what makes a mismatch attributable to the backend.

    The geometry comes from :func:`resolve_flex_backend` rather than being spelled out, so
    these tests measure the same combination of block size, padding and kernel options that a
    run with ``flex_attention_backend="auto"`` picks up. Demanding ``"flash"`` turns a box
    without the kernels, or without the hardware, into a skip with the reason attached.
    """
    try:
        backend = resolve_flex_backend(device, "flash")
    except ValueError as e:
        pytest.skip(str(e))
    und_samples = _und_samples(12, 12, length=backend.causal_seq_alignment)  # one block of UND tokens
    metadata = _metadata_from_tokens(
        _make_multiview_tokens(),
        seq_len=backend.full_seq_alignment,
        device=str(device),
        und_samples=und_samples,
    )
    block_mask = build_block_mask(metadata, device, backend.block_size)
    assert block_mask.BLOCK_SIZE == backend.block_size  # the mask the backend requires
    return metadata, block_mask, backend


@pytest.mark.L0
@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlexAttention kernels require a GPU.")
@pytest.mark.parametrize("return_lse", [False, True])
def test_flash_backend_forward_matches_dense_reference(return_lse: bool) -> None:
    """FlashAttention-4's forward over the fused ``[UND | GEN]`` stream must match a dense reference.

    One ``FlexBackend`` apart from the Triton fused test above, so what this
    isolates is the backend. The failure it guards against is silent: FA4 decides which
    256-row query tiles to skip from the block mask, so a mask at the wrong granularity, or
    a stream padded to the wrong multiple, drops or over-includes tokens instead of raising.

    bf16 because the CuTeDSL kernels are bf16/fp16 only; the reference runs in fp32 over
    the same rounded inputs, leaving the kernel's own accumulation as the residual. Padded
    query rows are compared alongside the real ones -- the reference applies the very same
    mask, so they are not a special case.

    ``return_lse`` is covered because forward-only is the one place FA4 can produce the
    log-sum-exp: its backward cannot differentiate through it, which is why the fused path
    that trains does not ask for it.
    """
    torch.manual_seed(0)
    torch.compiler.reset()
    device = torch.device("cuda")
    num_heads, head_dim = 4, 64  # the FA4 kernels are compiled for 64/128-wide heads
    metadata, block_mask, backend = _flash_fused_case(device)
    generator = torch.Generator(device=device).manual_seed(7)

    def _tensor(seq_len: int) -> torch.Tensor:
        return torch.randn(1, seq_len, num_heads, head_dim, device=device, dtype=torch.bfloat16, generator=generator)

    q = _tensor(metadata.q_len)  # [1,N_full,H,D]
    k = _tensor(metadata.seq_len)  # [1,N_und+N_full,H,D]
    v = _tensor(metadata.seq_len)

    with _flash_backend_or_skip():
        result = flex_attention(q, k, v, block_mask, backend, return_lse=return_lse)
    lse: torch.Tensor | None = None
    if return_lse:
        assert isinstance(result, tuple)
        out = cast(torch.Tensor, result[0])
        lse = cast(torch.Tensor, result[1])
        assert lse.shape == (1, metadata.q_len, num_heads)
    else:
        assert isinstance(result, torch.Tensor)
        out = cast(torch.Tensor, result)
    assert out.shape == (1, metadata.q_len, num_heads, head_dim)

    mask = _mask_mod_to_dense(metadata).to(device)  # [q_len, num_und+q_len] bool
    ref_out, ref_lse = _reference_attention(q.float(), k.float(), v.float(), mask)
    torch.testing.assert_close(out.float(), ref_out, atol=2e-2, rtol=2e-2)
    if lse is not None:
        torch.testing.assert_close(lse.float(), ref_lse, atol=2e-2, rtol=2e-2)


@pytest.mark.L0
@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlexAttention kernels require a GPU.")
def test_flash_backend_gradients_match_dense_reference() -> None:
    """FlashAttention-4's backward over the fused ``[UND | GEN]`` stream must match a dense reference.

    This is the shape training takes: GEN queries against the concatenated key stream, no
    log-sum-exp requested. The omission is deliberate and is what makes the backend usable
    for training at all -- FA4's backward raises on ``dLSE``, and the fused branch of
    ``two_way_attention`` has nothing left to merge, so it never asks. The dense reference
    is the same one the Triton fused test uses, so both backends are held to one standard.

    Tolerances are sized for bf16 inputs and stay far below the errors that matter here: a
    query tile the kernel skipped, or a key block the mask placed wrong, moves whole rows
    by O(1) rather than by roundoff.
    """
    torch.manual_seed(0)
    torch.compiler.reset()
    device = torch.device("cuda")
    num_heads, head_dim = 4, 64  # the FA4 kernels are compiled for 64/128-wide heads
    metadata, block_mask, backend = _flash_fused_case(device)
    generator = torch.Generator(device=device).manual_seed(7)

    def _leaf(seq_len: int) -> torch.Tensor:
        tensor = torch.randn(1, seq_len, num_heads, head_dim, device=device, dtype=torch.bfloat16, generator=generator)
        return tensor.requires_grad_(True)

    q = _leaf(metadata.q_len)  # [1,N_full,H,D]
    k = _leaf(metadata.seq_len)  # [1,N_und+N_full,H,D]
    v = _leaf(metadata.seq_len)
    leaves = [q, k, v]
    # The reference differentiates fp32 copies rather than these bf16 leaves, so its
    # gradients are not rounded twice on the way out.
    ref_leaves = [leaf.detach().float().requires_grad_(True) for leaf in leaves]
    ref_q, ref_k, ref_v = ref_leaves

    # bf16 values carried in fp32: the flash path rounds the upstream gradient to bf16, and
    # drawing it in bf16 first makes that rounding a no-op, so both paths differentiate the
    # same loss to the bit.
    grad_seed = torch.randn(
        1, metadata.q_len, num_heads, head_dim, device=device, dtype=torch.bfloat16, generator=generator
    ).float()

    mask = _mask_mod_to_dense(metadata).to(device)  # [q_len, num_und+q_len] bool
    with _flash_backend_or_skip():
        got = cast(torch.Tensor, flex_attention(q, k, v, block_mask, backend))
    expected, _lse = _reference_attention(ref_q, ref_k, ref_v, mask)
    assert got.shape == expected.shape == (1, metadata.q_len, num_heads, head_dim)
    torch.testing.assert_close(got.float(), expected, atol=2e-2, rtol=2e-2)

    # autograd.grad rather than .backward() so the two graphs never share a .grad accumulator.
    with _flash_backend_or_skip():
        got_grads = torch.autograd.grad((got.float() * grad_seed).sum(), leaves, materialize_grads=True)
    expected_grads = torch.autograd.grad((expected * grad_seed).sum(), ref_leaves, materialize_grads=True)
    for name, got_grad, expected_grad in zip(("dq", "dk", "dv"), got_grads, expected_grads):
        torch.testing.assert_close(
            got_grad.float(), expected_grad, atol=2e-2, rtol=2e-2, msg=lambda m, n=name: f"{n}: {m}"
        )


@pytest.mark.L0
@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlexAttention kernels require a GPU.")
@_GEOMETRIES
def test_build_multiview_block_mask_covers_the_padded_gen_stream(backend: FlexBackend) -> None:
    """The composed entry point returns a mask sized for the block-padded GEN stream.

    That size is what ``flex_attention`` checks q/k/v against, so a mask built
    for the real token count instead of the padded one would be rejected there.
    """
    device = torch.device("cuda")
    case = _MULTIVIEW_CASES["transfer_two_items_two_samples"]
    seq_len = backend.full_seq_alignment

    block_mask = build_multiview_block_mask(
        seq_len=seq_len,
        full_q_offsets=_case_offsets(case).to(device),
        items_per_sample=_case_items(case),
        device=device,
        block_size=backend.block_size,
    )

    assert block_mask.shape[-2:] == (seq_len, seq_len)


@pytest.mark.L0
@_GEOMETRIES
@_NOISY_SCOPES
def test_build_multiview_block_mask_matches_create_block_mask_on_a_camera_pack(
    backend: FlexBackend, attention_scope: AttentionScope
) -> None:
    """End-to-end agreement on a transfer pack shaped like the real one.

    Eleven cameras, a noisy item plus a fully conditioning control item, and cells
    wider than a block -- the geometry the multiview runs use, with the spatial extent
    scaled down so torch's dense ``[S, S]`` reference still fits in memory.

    Every noisy scope runs, because this is where the two forms of the predicate meet:
    ``build_multiview_block_mask`` collapses the key-stream form onto the metadata runs,
    while the reference evaluates the ``mask_mod`` form densely. Should one of them ever stop
    reading the scope, the disagreement lands here rather than in a training run, where a
    block wrongly called fully unmasked skips ``mask_mod`` and attends across views in
    silence. The narrow scopes are also what make the collapsing's granularity load-bearing:
    a grouping any coarser than one run per ``(item, frame, view)`` cell could not express
    them.
    """
    num_views, frames_per_view, spatial_tokens = 11, 3, 92
    latent_t = num_views * frames_per_view
    item_tokens = latent_t * spatial_tokens
    q_block, kv_block = backend.block_size
    seq_len = math.ceil(2 * item_tokens / q_block) * q_block
    num_q_blocks, num_kv_blocks = seq_len // q_block, seq_len // kv_block
    kwargs = dict(
        seq_len=seq_len,
        full_q_offsets=torch.tensor([0, 2 * item_tokens], dtype=torch.int32),
        items_per_sample=[
            [
                MaskItem(
                    token_shape=(latent_t, spatial_tokens, 1),
                    condition_mask=_condition_mask(latent_t, frames),
                    num_views=num_views,
                )
                for frames in ([], list(range(latent_t)))
            ]
        ],
        device=torch.device("cpu"),
        attention_scope=attention_scope,
    )

    got = build_multiview_block_mask(**kwargs, block_size=backend.block_size)  # type: ignore[arg-type]
    expected = _eager_block_mask(build_multiview_flex_metadata(**kwargs), backend.block_size)  # type: ignore[arg-type]

    assert expected.full_kv_num_blocks is not None and got.full_kv_num_blocks is not None
    assert torch.equal(
        _blocks_to_dense(got.kv_num_blocks, got.kv_indices, num_q_blocks, num_kv_blocks),
        _blocks_to_dense(expected.kv_num_blocks, expected.kv_indices, num_q_blocks, num_kv_blocks),
    )
    assert torch.equal(
        _blocks_to_dense(got.full_kv_num_blocks, got.full_kv_indices, num_q_blocks, num_kv_blocks),
        _blocks_to_dense(expected.full_kv_num_blocks, expected.full_kv_indices, num_q_blocks, num_kv_blocks),
    )
    assert got.sparsity() == pytest.approx(expected.sparsity())


@pytest.mark.L0
@_GEOMETRIES
@_NOISY_SCOPES
def test_a_noisy_scope_reaches_the_block_mask_as_sparsity(
    backend: FlexBackend, attention_scope: AttentionScope
) -> None:
    """A narrow scope has to survive the collapsing as skipped blocks, not just as masked pairs.

    A scope that holds per token but lands inside partially-masked blocks buys nothing: the
    kernel visits whole blocks, and the pairs inside one are ``mask_mod``'s business alone. So
    this holds the mask's sparsity to the fraction of the noisy quadrant its scope drops --
    all but ``1/V`` of it for the query's own view, all but ``(F + V - 1) / (V*F)`` for its own
    view or frame. The fractions are exact rather than approached because a ``(frame, view)``
    cell fills a whole query block at either geometry, which the assertion below insists on.

    One conditioning-free item at a length that needs no padding, so the noisy quadrant is the
    entire mask and its fraction is the mask's.
    """
    num_views, frames_per_view, spatial_tokens = 11, 4, 256
    latent_t = num_views * frames_per_view
    item_tokens = latent_t * spatial_tokens
    q_block = backend.block_size[0]
    assert spatial_tokens % q_block == 0 and item_tokens % q_block == 0, "cells and blocks have to line up"
    kept = {
        "all_views": 1.0,
        "same_view": 1 / num_views,
        "same_view_or_frame": (frames_per_view + num_views - 1) / (num_views * frames_per_view),
    }[attention_scope]

    block_mask = build_multiview_block_mask(
        seq_len=item_tokens,
        full_q_offsets=torch.tensor([0, item_tokens], dtype=torch.int32),
        items_per_sample=[
            [
                MaskItem(
                    token_shape=(latent_t, spatial_tokens, 1),
                    condition_mask=_condition_mask(latent_t, []),
                    num_views=num_views,
                )
            ]
        ],
        device=torch.device("cpu"),
        block_size=backend.block_size,
        attention_scope=attention_scope,
    )

    assert block_mask.sparsity() == pytest.approx(100.0 * (1.0 - kept))


# ── The packer / mask wiring ─────────────────────────────────────────────────
# Everything above hands the builders metadata assembled by the test. These two run the
# real packer instead and derive the mask the way cosmos3_vfm_network does, which is
# the only place the two stream lengths, the UND offsets and the block size meet. A
# mask can be perfectly correct for the metadata it was given and still be wrong for
# the pack the kernel runs on, and no test above would notice. The cases above pin the
# supertoken rules; these pin the wiring that decides which tokens those rules see.
#
# Both check the mask's contents rather than running attention on it, which is why they
# stay on CPU. ``attention_test`` takes the same packs the other way, comparing the
# kernel's output and gradients against the dense two-way path across several shapes
# under torch.compile.

# Two samples, unequal GEN lengths so both streams end up padded: sample 0 is
# 4 views x 2 frames x 16 spatial = 128 GEN tokens, sample 1 is 3 views x 2 frames = 96.
_WIRING_UND_LENS = [7, 5]
_WIRING_TOKEN_SHAPES = [(8, 4, 4), (6, 4, 4)]  # (latent_t, patch_h, patch_w) per vision item
_WIRING_NUM_VIEWS = [4, 3]


def _reference_stream_sample_ids(offsets: torch.Tensor, length: int) -> torch.Tensor:
    """Per-token sample ids for one padded stream, derived independently of the builder."""
    ids = torch.full((length,), -1, dtype=torch.long)
    for sample in range(len(offsets) - 1):
        ids[int(offsets[sample]) : int(offsets[sample + 1])] = sample
    return ids


def _effective_token_mask(block_mask: BlockMask) -> torch.Tensor:
    """Expand a ``BlockMask`` to the ``[q_len, kv_len]`` visibility it actually encodes.

    Full blocks admit their whole tile; partial blocks admit whatever ``mask_mod`` says
    within it; unlisted blocks admit nothing. That combination is what the kernel reads,
    so it is what a reference has to be compared against.
    """
    q_len, kv_len = block_mask.shape[-2:]
    q_block, kv_block = block_mask.BLOCK_SIZE
    num_q_blocks, num_kv_blocks = q_len // q_block, kv_len // kv_block

    partial = _blocks_to_dense(block_mask.kv_num_blocks, block_mask.kv_indices, num_q_blocks, num_kv_blocks)
    if block_mask.full_kv_num_blocks is not None:
        full = _blocks_to_dense(block_mask.full_kv_num_blocks, block_mask.full_kv_indices, num_q_blocks, num_kv_blocks)
    else:
        full = torch.zeros_like(partial)

    device = block_mask.kv_indices.device
    zero = torch.zeros((), dtype=torch.long, device=device)
    q_idx = torch.arange(q_len, device=device).unsqueeze(-1)
    kv_idx = torch.arange(kv_len, device=device).unsqueeze(0)
    elementwise = block_mask.mask_mod(zero, zero, q_idx, kv_idx).cpu()

    def _to_tokens(blocks: torch.Tensor) -> torch.Tensor:
        return blocks.repeat_interleave(q_block, dim=0).repeat_interleave(kv_block, dim=1)

    return _to_tokens(full) | (_to_tokens(partial) & elementwise)


def _wiring_pack(x: torch.Tensor, *, backend: FlexBackend) -> SequencePack:
    """Pack ``x`` through the real packer as the network does for a multiview batch.

    The two streams are padded to what ``backend`` asks of each of them, which is the pairing
    the network hands ``build_packed_sequence``: they differ on FA4, where only the queries are
    tiled 256 at a time.
    """
    gen_lens = [latent_t * patch_h * patch_w for latent_t, patch_h, patch_w in _WIRING_TOKEN_SHAPES]
    split_lens: list[int] = []
    und_indexes: list[int] = []
    gen_indexes: list[int] = []
    start = 0
    for und_len, gen_len in zip(_WIRING_UND_LENS, gen_lens):
        split_lens.extend((und_len, gen_len))
        und_indexes.extend(range(start, start + und_len))
        gen_indexes.extend(range(start + und_len, start + und_len + gen_len))
        start += und_len + gen_len

    return build_packed_sequence(
        "two_way",
        packed_sequence=x,
        attn_modes=["causal", "full"] * len(_WIRING_UND_LENS),
        split_lens=split_lens,
        sample_lens=[und + gen for und, gen in zip(_WIRING_UND_LENS, gen_lens)],
        packed_und_token_indexes=cast(torch.LongTensor, torch.tensor(und_indexes, dtype=torch.long)),
        packed_gen_token_indexes=cast(torch.LongTensor, torch.tensor(gen_indexes, dtype=torch.long)),
        num_heads=x.shape[-2],
        head_dim=x.shape[-1],
        num_layers=1,
        full_seq_alignment=backend.full_seq_alignment,
        causal_seq_alignment=backend.causal_seq_alignment,
    )[0]


def _wiring_block_mask(
    pack: SequencePack,
    *,
    block_size: tuple[int, int],
    condition_frames: list[list[int]],
    attention_scope: AttentionScope = "all_views",
) -> BlockMask:
    """Build the GEN mask from ``pack`` exactly as ``cosmos3_vfm_network`` does."""
    full_only_seq, full_q_offsets = get_full_only_seq(pack)
    causal_seq, causal_offsets = get_causal_seq(pack)
    return build_multiview_block_mask(
        seq_len=full_only_seq.shape[0],
        full_q_offsets=full_q_offsets,
        items_per_sample=[
            # One item per sample, which is what the wiring pack carries.
            [
                MaskItem(
                    token_shape=token_shape,
                    condition_mask=_condition_mask(token_shape[0], frames),
                    num_views=num_views,
                )
            ]
            for token_shape, num_views, frames in zip(_WIRING_TOKEN_SHAPES, _WIRING_NUM_VIEWS, condition_frames)
        ],
        device=full_only_seq.device,
        block_size=block_size,
        num_und=causal_seq.shape[0],
        causal_offsets=causal_offsets,
        attention_scope=attention_scope,
    )


@pytest.mark.L0
@_GEOMETRIES
def test_network_wiring_gives_the_dense_same_sample_mask_without_conditioning(backend: FlexBackend) -> None:
    """A mask built off a real pack sees exactly one sample's ``[UND | GEN]`` per query.

    Catches the wiring the unit cases cannot: ``seq_len`` taken from the padded GEN
    stream, ``num_und`` from the padded UND stream, and ``causal_offsets`` labelling UND
    keys with their sample. Getting any of those from the wrong stream still yields a
    well-formed mask, just one that lets a query read another sample's tokens or drops
    its own.
    """
    pack = _wiring_pack(torch.zeros(sum(_WIRING_UND_LENS) + 224, 4, 8), backend=backend)
    _, full_q_offsets = get_full_only_seq(pack)
    causal_seq, causal_offsets = get_causal_seq(pack)

    # The pack keeps its padding in separate `_pad_segment` offsets, so these hold one
    # entry per sample boundary. Were a trailing pad entry to appear here, the reference
    # below would silently start treating padding as a real sample.
    assert len(causal_offsets) == len(_WIRING_UND_LENS) + 1
    assert len(full_q_offsets) == len(_WIRING_UND_LENS) + 1

    block_mask = _wiring_block_mask(pack, block_size=backend.block_size, condition_frames=[[], []])
    q_len, kv_len = block_mask.shape[-2:]
    num_und = causal_seq.shape[0]
    assert (num_und, kv_len - num_und) == (backend.causal_seq_alignment, q_len), "each stream is padded on its own"

    gen_ids = _reference_stream_sample_ids(full_q_offsets, q_len)
    key_ids = torch.cat((_reference_stream_sample_ids(causal_offsets, num_und), gen_ids))
    # Padding carries -1 on both sides, so it forms its own "sample": real queries never
    # reach it and padded queries have it to themselves, which is what keeps the softmax
    # over a padded row non-empty.
    expected = key_ids.unsqueeze(0) == gen_ids.unsqueeze(-1)

    assert torch.equal(_effective_token_mask(block_mask), expected)


@pytest.mark.L0
@_GEOMETRIES
def test_network_wiring_matches_the_reference_visibility_with_conditioning(backend: FlexBackend) -> None:
    """The same pack with conditioning frames, against the full per-token reference.

    The no-conditioning test above only sees sample boundaries, because noisy->noisy is
    full within a sample. Conditioning is what makes the frame layout observable, so this
    is the case that ties ``token_shapes`` and the conditioning quadrants to the tokens
    the packer actually laid out.

    It still cannot observe ``num_views_per_item``, because the default
    ``attention_scope="all_views"`` this test runs at makes the conditioning<->conditioning
    and noisy->RGB rules it exercises unrestricted by view or frame -- any view divides
    ``latent_t`` just as well as any other. The test below is the one that does observe
    it: the narrow scopes key a rule on the view or the frame alone, and the view count is
    then what says where each view ends.
    """
    condition_frames = [[0, 2], [0]]  # camera-major latent indices: view 0/1 frame 0, and view 0 frame 0
    pack = _wiring_pack(torch.zeros(sum(_WIRING_UND_LENS) + 224, 4, 8), backend=backend)
    causal_seq, causal_offsets = get_causal_seq(pack)

    block_mask = _wiring_block_mask(pack, block_size=backend.block_size, condition_frames=condition_frames)
    q_len = block_mask.shape[-2]

    tokens = _expected_tokens(
        dict(
            num_items_per_sample=[1] * len(_WIRING_UND_LENS),
            token_shapes=_WIRING_TOKEN_SHAPES,
            num_views_per_item=_WIRING_NUM_VIEWS,
            condition_frames=condition_frames,
        )
    )
    und_samples = _reference_stream_sample_ids(causal_offsets, causal_seq.shape[0]).tolist()
    expected = _reference_visibility(tokens, q_len, und_samples=und_samples)

    assert torch.equal(_effective_token_mask(block_mask), expected)


@pytest.mark.L0
@_GEOMETRIES
@_NOISY_SCOPES
def test_network_wiring_honours_the_attention_scope(backend: FlexBackend, attention_scope: AttentionScope) -> None:
    """``attention_scope`` reaches the mask the network builds, and only that rule.

    Conditioning is left out so the scope is the only thing separating the cases: every GEN
    token is noisy, so a query sees either its whole sample, exactly its own view of it, or
    its own view plus its own frame. The samples here carry 4 and 3 views over 2 frames each,
    so all three scopes differ, and a mask that read the view count from the wrong item would
    match none of the references.

    The last assertion is what keeps the rest honest. Every other check compares the mask
    against a reference built from the same scope, which would still pass if the scope were
    threaded through both and dropped on the floor.
    """
    pack = _wiring_pack(torch.zeros(sum(_WIRING_UND_LENS) + 224, 4, 8), backend=backend)
    causal_seq, causal_offsets = get_causal_seq(pack)

    block_mask = _wiring_block_mask(
        pack,
        block_size=backend.block_size,
        condition_frames=[[], []],
        attention_scope=attention_scope,
    )
    q_len = block_mask.shape[-2]

    tokens = _expected_tokens(
        dict(
            num_items_per_sample=[1] * len(_WIRING_UND_LENS),
            token_shapes=_WIRING_TOKEN_SHAPES,
            num_views_per_item=_WIRING_NUM_VIEWS,
            condition_frames=[[], []],
        )
    )
    und_samples = _reference_stream_sample_ids(causal_offsets, causal_seq.shape[0]).tolist()
    expected = _reference_visibility(tokens, q_len, und_samples=und_samples, attention_scope=attention_scope)
    got = _effective_token_mask(block_mask)

    assert torch.equal(got, expected)
    for other_scope in set(ATTENTION_SCOPES) - {attention_scope}:
        other = _reference_visibility(tokens, q_len, und_samples=und_samples, attention_scope=other_scope)
        assert not torch.equal(expected, other), f"{attention_scope} and {other_scope} agree on this pack"
