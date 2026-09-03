# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch.nn.attention.flex_attention import BlockMask

from cosmos_framework.model.attention import (
    attention,
    merge_attentions,
    multi_dimensional_attention_varlen,
)
from cosmos_framework.model.attention.masks import CausalType
from cosmos_framework.model.generator.utils.memory import KVToStore, MemoryValue


class SplitInfo:
    def __init__(
        self,
        split_lens: list[int],
        attn_modes: list[str],
        sample_lens: list[int],
        actual_len: int,
        is_three_way: bool = False,
        vision_token_shapes: list[tuple[int, int, int]] | None = None,
        action_token_shapes: list[tuple[int, ...]] | None = None,
        num_action_tokens_per_supertoken: int = 0,
        null_action_supertokens: bool = False,
    ):
        """
        Actual len is the actual non-padded length of the packed sequence.
        It's used to trim split_lens, attn_modes and sample_lens, which may
        be padded to max sequence length by upstream packers.
        """
        assert sum(sample_lens) == sum(split_lens), (
            f"Sum of new sample lens {sum(sample_lens)} is not equal to sum of new split lens {sum(split_lens)}"
        )

        max_causal_len = 0
        max_full_len = 0
        for split_len, attn_mode in zip(split_lens, attn_modes):
            if attn_mode == "causal":
                max_causal_len = max(max_causal_len, split_len)
            elif attn_mode == "full":
                max_full_len = max(max_full_len, split_len)

        self.max_causal_len = max_causal_len
        self.max_full_len = max_full_len
        self.max_sample_len = max(sample_lens)

        self.split_lens = split_lens
        self.attn_modes = attn_modes
        self.sample_lens = sample_lens

        self.is_three_way = is_three_way
        self.vision_token_shapes = vision_token_shapes
        self.action_token_shapes = action_token_shapes
        self.num_action_tokens_per_supertoken = num_action_tokens_per_supertoken
        self.null_action_supertokens = null_action_supertokens

        # Multi-control transfer fields (set post-construction in cosmos3_vfm_network.py).
        # Gen-relative token ranges for each control stream, one tuple (start, end) per control.
        self.control_stream_token_ranges: list[tuple[int, int]] | None = None
        # Gen-relative token range (start, end) for the noisy target tokens.
        self.noisy_token_range: tuple[int, int] | None = None
        # Per-control scalar weights; parallel to control_stream_token_ranges.
        self.control_weights: list[float] | None = None
        # Multiview GEN-query mask, set post-construction in cosmos3_vfm_network.py when
        # use_multiview_flex_attention is on. When populated, two_way_attention computes the
        # generator's full attention with FlexAttention over the fused [UND | GEN] stream
        # under the multiview supertoken mask. Only the mask is carried here; the per-token
        # fields it was derived from are an implementation detail of
        # flex_attention.build_multiview_block_mask.
        self.flex_block_mask: BlockMask | None = None
        # The backend that mask was built for, from the flex_attention.resolve_flex_backend call
        # that fixed its block size. They travel together because they have to agree: the
        # FlashAttention-4 kernels are only correct for a mask built at that backend's coarser
        # block size, which is also what the packer padded the two streams to.
        self.flex_backend: FlexBackend | None = None


AttentionMaskType = SplitInfo


_SPLIT_INFO_ATTRIBUTES = (
    "max_causal_len",
    "max_full_len",
    "max_sample_len",
    "split_lens",
    "attn_modes",
    "sample_lens",
    "is_three_way",
    "vision_token_shapes",
    "action_token_shapes",
    "num_action_tokens_per_supertoken",
    "null_action_supertokens",
    "control_stream_token_ranges",
    "noisy_token_range",
    "control_weights",
)


def _is_split_info_compatible(attention_mask: object) -> bool:
    return isinstance(attention_mask, SplitInfo) or all(
        hasattr(attention_mask, attribute) for attribute in _SPLIT_INFO_ATTRIBUTES
    )


_dotproduct_attention_cache = {}


from cosmos_framework.model.generator.mot.flex_attention import FlexBackend, flex_attention
from cosmos_framework.data.generator.sequence_packing.natten import (
    generate_natten_metadata,
    generate_temporal_causal_natten_metadata,
)
from cosmos_framework.data.generator.sequence_packing.runtime import (
    SequencePack,
    SequencePackMetadata,
    from_mode_splits,
    get_all_seq,
    get_all_seq_padded,
    get_causal_seq,
    get_causal_seq_padded,
    get_full_only_seq,
    get_full_only_seq_padded,
    sequence_pack_from_packed_sequence,
)


def _use_varlen(sample_offsets: torch.Tensor) -> bool:
    """Whether a pass over this pack needs the varlen (sequence-packed) attention API.

    True means the caller passes the ``cumulative_seqlen_*``/``max_seqlen_*`` metadata to
    :func:`attention`; False means it calls the dense API instead, with no ranges at all.

    With a single sample there is exactly one sequence in the pack, so that metadata is
    redundant and the dense API computes the same thing. This remains correct in the presence
    of trailing padding: for causal self-attention the mask never lets a real query attend to
    padded keys (padding is appended after all real tokens), the full path keeps the unpadded
    ``get_all_seq`` KV whenever this returns False (the dense API has no ranges to fence
    padding off with), and any padded query rows are independent of the real rows and simply
    discarded downstream.

    The dense path is gated to forward-only (inference) execution via ``torch.is_grad_enabled``,
    which is False under ``torch.no_grad()``/``torch.inference_mode()`` and True during training.
    This avoids branching on the sample count during training, where batch composition varies
    between single- and multi-sample packs; keeping a single code path there prevents
    torch.compile from specializing on both shapes and incurring the associated recompilation
    overhead.

    The grad-mode test comes first so that training never reaches the sample count, since ``or``
    stops at the first true operand. Reading that count costs no kernel and no device sync --
    ``sample_offsets`` has shape ``[num_samples + 1]``, so the count is a shape -- but comparing
    a shape against a constant makes torch.compile specialize the enclosing graph on that count,
    and a training run whose packs hold one sample sometimes and several other times then
    recompiles every layer on each count it meets. Inference wants the specialization and has a
    stable count.

    Args:
        sample_offsets: the pack's per-sample offsets, shape ``[num_samples + 1]``, whose length
            gives the sample count.
    Returns:
        bool: True to pass varlen metadata to :func:`attention`, False to use the dense API.
    """
    return torch.is_grad_enabled() or sample_offsets.shape[0] > 2


def two_way_attention(
    packed_query_states: SequencePack,
    packed_key_states: SequencePack,
    packed_value_states: SequencePack,
    packed_key_states_normalized: SequencePack | None = None,
    flex_block_mask: BlockMask | None = None,
    flex_backend: FlexBackend | None = None,
):
    """
    Performs two-way attention with causal and full attention.

    ``packed_key_states_normalized``: optional alternative K pack used for the generator's
    full attention (gen→all).  When provided, the generator attends to these keys
    instead of ``packed_key_states``, allowing the und K tokens to be normalised for
    the gen cross-attention path while keeping raw K tokens for the reasoner's own
    causal self-attention.  If ``None``, ``packed_key_states`` is used for both paths.

    ``flex_block_mask``: when provided, the generator's full attention runs on
    FlexAttention under the multiview supertoken mask instead of the maskless dense
    kernel. Both express "every GEN token attends to its whole sample"; the mask adds
    the supertoken restriction on the GEN→GEN quadrant, which no dense kernel can
    encode. The reasoner's causal self-attention is untouched either way.

    ``flex_backend``: which FlexAttention backend runs that mask, decided by
    ``flex_attention.resolve_flex_backend`` when the mask was built.  Required alongside
    ``flex_block_mask`` and meaningless without it: the backend's kernels are only correct
    at the block size the mask was built at, which ``flex_attention`` checks the two against
    each other for.
    """
    # For gen full-attention, use normed keys when provided,
    # otherwise fall back to the standard packed keys.
    packed_key_normalized = (
        packed_key_states_normalized if packed_key_states_normalized is not None else packed_key_states
    )

    # get_causal_seq_padded/get_full_only_seq_padded prefer the pack's pad-segment offsets when it
    # carries one: trailing padding rows belong to no sample, and varlen attention leaves rows
    # outside its cumulative ranges unwritten in both directions, so every pass that may see
    # padding needs to switch to it. The FlexAttention branch below needs no offsets at all,
    # because its mask marks padding with the -1 sentinel.
    #
    # Only the offsets and max lengths differ between the plain and the _padded accessors -- the
    # stream each returns is the pack's own tower tensor, already padded at pack time, so the two
    # families hand back the same object. That is why the FlexAttention branch below can mix them
    # freely when it concatenates the towers, and why ``causal_k_normalized`` here belongs to that
    # branch alone: the dense full pass takes its und keys from the interleaved stream instead.
    causal_q, causal_q_offsets, max_causal_len = get_causal_seq_padded(packed_query_states)
    causal_k, causal_k_offsets, _ = get_causal_seq_padded(packed_key_states)
    causal_v, _, _ = get_causal_seq_padded(packed_value_states)

    # NOTE: we can only use the don't care causal mask when we know seqlen_Q == seqlen_KV.
    # Since this is a varlen use case, we would need to statically check all Q and KV offsets
    # are the same.
    # We don't want to launch a kernel just to perform this check and slow down our model, and
    # we definitely don't want to complicate the sequence_packing code so that it performs a
    # static check when creating the packed sequence and metadata. Instead, we just rely
    # on causal_q_offsets and causal_k_offsets being the same tensor.
    use_dont_care_mask = causal_q_offsets is causal_k_offsets

    sample_offsets = packed_query_states["sample_offsets"]
    use_varlen = _use_varlen(sample_offsets)

    if use_varlen:
        causal_varlen_kwargs: dict[str, Any] = dict(
            cumulative_seqlen_Q=causal_q_offsets,
            cumulative_seqlen_KV=causal_k_offsets,
            max_seqlen_Q=max_causal_len,
            max_seqlen_KV=max_causal_len,
        )
    else:
        causal_varlen_kwargs = dict()

    causal_res = attention(
        causal_q.unsqueeze(0),  # [1,N_und,heads,head_dim]
        causal_k.unsqueeze(0),  # [1,N_und,heads,head_dim]
        causal_v.unsqueeze(0),  # [1,N_und,heads,head_dim]
        is_causal=True,
        causal_type=CausalType.DontCare if use_dont_care_mask else CausalType.TopLeft,
        **causal_varlen_kwargs,
    )  # [1,N_und,heads,head_dim]

    # [1,N_und,heads,head_dim] -> [N_und,heads,head_dim] -> [N_und,heads*head_dim]
    causal_out = causal_res.squeeze(0).flatten(-2, -1)  # type: ignore  # [N_und,heads*head_dim]

    full_q, full_q_offsets, max_full_len = get_full_only_seq_padded(packed_query_states)

    if flex_block_mask is not None:
        if flex_backend is None:
            raise ValueError(
                "flex_block_mask needs the FlexBackend it was built for: which kernels run the mask "
                "is only correct at the block size it was built at, so the two are set together."
            )
        # FlexAttention: the multiview supertoken mask encoded in flex_block_mask. It keys
        # GEN queries against [UND | GEN], so the two block-padded streams are concatenated
        # in that order rather than gathered back into the interleaved pack order that
        # get_all_seq produces. The mask was built over exactly these two lengths, which is why
        # the towers go in whole, padding included. Both halves come from the packs the dense
        # path below reads: the normalized keys, since und normalization is exactly what the gen
        # pass wants, and the raw values -- the UND halves are the tensors already fetched above,
        # and only the GEN halves are new here. This path needs no varlen offsets and no separate
        # cross-attention term: padding carries the -1 sentinel in the mask, so every row is
        # written and only padding attends to padding.
        causal_k_normalized, _ = get_causal_seq(packed_key_normalized)
        full_k, _ = get_full_only_seq(packed_key_normalized)  # [N_full,heads,head_dim]
        full_v, _ = get_full_only_seq(packed_value_states)  # [N_full,heads,head_dim]
        full_res = flex_attention(
            full_q.unsqueeze(0),  # [1,N_full,heads,head_dim]
            torch.cat((causal_k_normalized, full_k)).unsqueeze(0),  # [1,N_und+N_full,heads,head_dim]
            torch.cat((causal_v, full_v)).unsqueeze(0),  # [1,N_und+N_full,heads,head_dim]
            flex_block_mask,
            flex_backend,
        )  # [1,N_full,heads,head_dim]
    else:
        # Same treatment the causal pass gets above, on the stream this pass actually keys
        # against. full_q is the padded GEN stream, so without a pad segment its trailing rows
        # sit outside every cumulative range, and a varlen kernel leaves such rows -- and their
        # dq/dk/dv -- exactly as it found them (flash3 demonstrably does; see
        # ``attention_test.test_varlen_attention_backward_writes_query_grad_rows_past_its_ranges``).
        # Covering them takes a segment on both sides: the query side already has one (from
        # get_full_only_seq_padded above), and get_all_seq_padded gives the key side the matching
        # one, which is why this pass could not simply reuse the towers' offsets.
        if use_varlen:
            sample_k, sample_kv_offsets, max_sample_len = get_all_seq_padded(packed_key_normalized)
            sample_v, _, _ = get_all_seq_padded(packed_value_states)
            full_varlen_kwargs: dict[str, Any] = dict(
                cumulative_seqlen_Q=full_q_offsets,
                cumulative_seqlen_KV=sample_kv_offsets,
                max_seqlen_Q=max_full_len,
                max_seqlen_KV=max_sample_len,
            )
        else:
            # The padded key stream is only correct alongside the offsets that fence it off, so it
            # is not an option here even when the pack carries one: the dense API has no ranges to
            # confine a query to, padding K is zero, and exp(q.0) = 1 would give every real query
            # softmax mass on every padding row. get_all_seq's real-token stream has no padding to
            # attend to in the first place, which is what makes this branch correct -- and what
            # makes get_all_seq_padded impossible to hoist out of this if, however much its own
            # fallback to get_all_seq (taken only when the pack has no pad segment) invites it.
            sample_k = get_all_seq(packed_key_normalized)
            sample_v = get_all_seq(packed_value_states)
            full_varlen_kwargs = dict()

        full_res = attention(
            full_q.unsqueeze(0),  # [1,N_full,heads,head_dim]
            sample_k.unsqueeze(0),  # [1,N_all,heads,head_dim]  normed und K for gen
            sample_v.unsqueeze(0),  # [1,N_all,heads,head_dim]
            **full_varlen_kwargs,
        )  # [1,N_full,heads,head_dim]

    # [1,N_full,heads,head_dim] -> [N_full,heads,head_dim] -> [N_full,heads*head_dim]
    full_out = full_res.squeeze(0).flatten(-2, -1)  # type: ignore  # [N_full,heads*head_dim]

    out_all = from_mode_splits(causal_out, full_out, packed_query_states)
    return out_all


def three_way_attention(
    packed_query_states: SequencePack,
    packed_key_states: SequencePack,
    packed_value_states: SequencePack,
    natten_metadata: dict | None,
    attention_meta: SplitInfo | None = None,
    packed_key_states_normalized: SequencePack | None = None,
):
    """
    Performs three-way attention, with understanding and generations attentions fully decomposed,
    and allows sparsity / multi-dimensional masking in the generation tower.

    The generation-tower self-attention (``full_sa``) is computed by NATTEN when
    ``natten_metadata`` is provided and by dense self-attention otherwise, then merged
    by log-sum-exp with the gen→und cross-attention (``full_ca``).

    FlexAttention is deliberately not one of those paths. Its output has to be copied
    into the heads-last layout, which breaks the data-pointer contract that
    ``merge_attentions`` relies on to fix up the branch backward, so the merged
    gradients came out wrong while the forward looked fine. The multiview supertoken
    mask lives on ``two_way_attention`` instead, where GEN queries take the whole
    ``[UND | GEN]`` stream in a single kernel and there is nothing to merge.

    When attention_meta is provided with null_action_supertokens=True, zeros V for the first
    num_action_tokens_per_supertoken tokens of each sample's GEN sequence (null action
    supertokens for temporal causal training). The metadata encodes is_causal=(True, False):
    causal across T supertokens, full within each supertoken S.

    NOTE: the three-way decomposition is only done so we can handle sparsity in the gen tower,
    but a KEY assumption is that the "full" tokens all correspond to the same modality!
    We should be careful when extending this to beyond t2i and t2v.

    ``packed_key_states_normalized``: optional alternative K pack for the gen→und cross-attention
    (``full_ca``).  When provided, its causal (und) stream supplies the und K tokens seen by the
    generator, while ``packed_key_states``' own causal stream (raw und K) is still used for the
    reasoner's own causal self-attention.  If ``None``, both paths share ``packed_key_states``.
    """

    # get_causal_seq_padded/get_full_only_seq_padded prefer the pack's pad-segment offsets when a
    # pack carries one: trailing padding rows belong to no sample, and varlen attention leaves
    # rows outside its cumulative ranges unwritten in both directions, so every pass that may see
    # padding needs to switch to it. Both streams gain the same extra segment, which is what keeps
    # the query and key segment counts equal for the gen->und pass below.
    causal_q, causal_q_offsets, max_causal_len = get_causal_seq_padded(packed_query_states)
    causal_k, causal_k_offsets, _ = get_causal_seq_padded(packed_key_states)

    # For gen→und cross-attention use normed keys when provided,
    # otherwise fall back to the standard causal keys.
    if packed_key_states_normalized is not None:
        causal_k_normalized, causal_k_normalized_offsets, _ = get_causal_seq_padded(packed_key_states_normalized)
    else:
        causal_k_normalized, causal_k_normalized_offsets = causal_k, causal_k_offsets
    causal_v, _, _ = get_causal_seq_padded(packed_value_states)

    full_q, full_q_offsets, max_full_len = get_full_only_seq_padded(packed_query_states)
    full_k, _, _ = get_full_only_seq_padded(packed_key_states)
    full_v, _, _ = get_full_only_seq_padded(packed_value_states)

    if attention_meta is not None and attention_meta.null_action_supertokens:
        # Zero V for the first num_action_tokens_per_supertoken tokens of each
        # sample's GEN sequence (null action supertokens at t=0).
        # out_i = Σ_j softmax(QKᵀ/√d)_j · V_j — terms with V_j=0 contribute exactly 0 to the output,
        # regardless of attention weights. Softmax mass is still allocated to these positions (not
        # redistributed), so this differs from hard key masking, but the output contribution is 0.
        full_v = full_v.clone()
        # Deliberately the plain (non-pad-segment) get_full_only_seq offsets, not full_q_offsets
        # above: this indexes into the start of each real sample's GEN sequence, and the
        # pad-segment variant appends the padding region's own start as a spurious extra "sample".
        # Zeroing from that spurious start could walk num_action_tokens_per_supertoken rows past
        # the tensor's end, since the pad segment is only guaranteed non-empty, not that long.
        _, full_q_offsets_unpadded = get_full_only_seq(packed_query_states)
        starts = full_q_offsets_unpadded[:-1].long()  # [B]
        null_positions = (
            starts.unsqueeze(1) + torch.arange(attention_meta.num_action_tokens_per_supertoken, device=starts.device)
        ).reshape(-1)
        full_v[null_positions] = 0

    use_dont_care_mask = causal_q_offsets is causal_k_offsets

    # NOTE: cosmos_framework attention is BSHD in, BSHD out
    causal_res = attention(
        causal_q.unsqueeze(0),  # [1,N_und,heads,head_dim]
        causal_k.unsqueeze(0),  # [1,N_und,heads,head_dim]
        causal_v.unsqueeze(0),  # [1,N_und,heads,head_dim]
        cumulative_seqlen_Q=causal_q_offsets,
        cumulative_seqlen_KV=causal_k_offsets,
        max_seqlen_Q=max_causal_len,
        max_seqlen_KV=max_causal_len,
        is_causal=True,
        causal_type=CausalType.DontCare if use_dont_care_mask else CausalType.TopLeft,
    )  # [1,N_und,heads,head_dim]
    # [1,N_und,heads,head_dim] -> [N_und,heads,head_dim] -> [N_und,heads*head_dim]
    causal_out = causal_res.squeeze(0).flatten(-2, -1)  # type: ignore  # [N_und,heads*head_dim]

    # GEN-tower self-attention (full_sa), NATTEN when it has metadata and dense otherwise.
    if natten_metadata is not None:
        full_sa, full_sa_lse = multi_dimensional_attention_varlen(
            full_q.unsqueeze(0),  # [1,N_full,heads,head_dim]
            full_k.unsqueeze(0),  # [1,N_full,heads,head_dim]
            full_v.unsqueeze(0),  # [1,N_full,heads,head_dim]
            metadata=natten_metadata,
            return_lse=True,
        )  # full_sa: [1,N_full,heads,head_dim], full_sa_lse: [1,N_full,heads]
    else:
        # Dense layer: each GEN token attends to every GEN token within its own
        # packed sample (block-diagonal, bidirectional). Self-attention, so the
        # KV offsets equal the Q offsets.
        full_sa, full_sa_lse = attention(
            full_q.unsqueeze(0),  # [1,N_full,heads,head_dim]
            full_k.unsqueeze(0),  # [1,N_full,heads,head_dim]
            full_v.unsqueeze(0),  # [1,N_full,heads,head_dim]
            cumulative_seqlen_Q=full_q_offsets,
            cumulative_seqlen_KV=full_q_offsets,
            max_seqlen_Q=max_full_len,
            max_seqlen_KV=max_full_len,
            return_lse=True,
        )  # full_sa: [1,N_full,heads,head_dim], full_sa_lse: [1,N_full,heads]

    full_ca, full_ca_lse = attention(
        full_q.unsqueeze(0),  # [1,N_full,heads,head_dim]
        causal_k_normalized.unsqueeze(0),  # [1,N_und,heads,head_dim]  normed und K for gen→und
        causal_v.unsqueeze(0),  # [1,N_und,heads,head_dim]
        cumulative_seqlen_Q=full_q_offsets,
        cumulative_seqlen_KV=causal_k_normalized_offsets,
        max_seqlen_Q=max_full_len,
        max_seqlen_KV=max_causal_len,
        return_lse=True,
    )  # full_ca: [1,N_full,heads,head_dim], full_ca_lse: [1,N_full,heads]

    assert full_sa.shape == full_ca.shape
    full_res, _ = merge_attentions(
        outputs=[full_sa, full_ca], lse_tensors=[full_sa_lse, full_ca_lse], torch_compile=False
    )  # [1,N_full,heads,head_dim]

    # [1,N_full,heads,head_dim] -> [N_full,heads,head_dim] -> [N_full,heads*head_dim]
    full_out = full_res.squeeze(0).flatten(-2, -1)  # type: ignore  # [N_full,heads*head_dim]

    out_all = from_mode_splits(causal_out, full_out, packed_query_states)
    return out_all


def multi_control_two_way_attention(
    packed_query_states: SequencePack,
    packed_key_states: SequencePack,
    packed_value_states: SequencePack,
    split_info: SplitInfo,
) -> SequencePack:
    """Two-way attention for multi-control transfer inference.

    N independent single-control attention passes; noisy output = weighted sum.

    Layout of the "full/gen" segment (mirrors the packed batch built by ``build_transfer_batch``):

        full = [ctrl_1 | ctrl_2 | ... | ctrl_N | noisy]

    For each control i, one independent maskless SDPA is computed:

        ctrl_i and noisy both attend to KV = [text | ctrl_i | noisy]

    The final outputs are:
      - ctrl_i output: from pass i only
      - noisy output:  w_1 * noisy_out_1 + ... + w_N * noisy_out_N  (weighted sum)

    All SDPA calls are maskless → Flash Attention is always active.
    N=1, w=1.0 → identical to ``two_way_attention``.

    Padding safety:
      Both ``get_causal_seq`` and ``get_full_only_seq`` can return padded rows.
      We unpad to valid token counts before each SDPA so that padded rows
      never enter the softmax denominator.

    Args:
        packed_query/key/value_states: SequencePack for a single sample.
        split_info: SplitInfo carrying ``control_stream_token_ranges``,
            ``noisy_token_range``, and ``control_weights`` (all must be non-None).
    """
    assert not torch.is_grad_enabled(), "Multi-control attention does not support grad mode"
    assert split_info.control_stream_token_ranges is not None
    assert split_info.noisy_token_range is not None
    assert split_info.control_weights is not None

    ctrl_ranges = split_info.control_stream_token_ranges
    noisy_s, noisy_e = split_info.noisy_token_range
    weights = split_info.control_weights

    # ── 1. Text self-attention (causal) ──────────────────────────────────────
    causal_q, causal_q_offsets = get_causal_seq(packed_query_states)
    causal_k, causal_k_offsets = get_causal_seq(packed_key_states)
    causal_v, _ = get_causal_seq(packed_value_states)

    use_dont_care_mask = causal_q_offsets is causal_k_offsets

    # No varlen metadata: this pack holds one sample, so the offsets are a single
    # [0, n_text].
    #
    # Unlike _sdpa these streams are still padded -- nothing unpads them -- so dense keys
    # over the padded length rather than over [0, n_text]. That costs nothing here (the text
    # stream is small enough that both forms are launch-bound) and changes no real row:
    # padding is appended after every real token, so a real query i < n_text only ever
    # attends keys <= i, all of them real.
    causal_res = attention(
        causal_q.unsqueeze(0),
        causal_k.unsqueeze(0),
        causal_v.unsqueeze(0),
        is_causal=True,
        causal_type=CausalType.DontCare if use_dont_care_mask else CausalType.TopLeft,
    )
    causal_out = causal_res.squeeze(0).flatten(-2, -1)  # type: ignore  # [N_text, Hq*D]

    # ── 2. Extract unpadded full/gen tokens ──────────────────────────────────
    full_q, full_q_offsets = get_full_only_seq(packed_query_states)
    full_k, _ = get_full_only_seq(packed_key_states)
    full_v, _ = get_full_only_seq(packed_value_states)

    n_text = int(causal_k_offsets[-1])
    n_full = int(full_q_offsets[-1])

    # `n_full` comes from int(full_q_offsets[-1]) → an unbacked symint under
    # torch.compile. The control ranges + noisy range partition the full/gen
    # segment with noisy last, so `noisy_e` (a concrete int from SplitInfo) is
    # exactly the number of valid gen tokens == n_full. Binding them lets Dynamo
    # treat the per-segment `full_*_v[cs:ce]` slices below as concrete-length, so
    # the in-place writes `full_out_v[cs:ce] = _sdpa(...)` don't raise
    # data-dependent `Eq(slice_len, out_len)` guards.
    torch._check(n_full == noisy_e)

    # Unpad to avoid padded rows entering the softmax denominator.
    causal_k_v = causal_k[:n_text]  # [N_text, Hkv, D]
    causal_v_v = causal_v[:n_text]  # [N_text, Hkv, D]
    full_q_v = full_q[:n_full]  # [N_full, Hq,  D]
    full_k_v = full_k[:n_full]  # [N_full, Hkv, D]
    full_v_v = full_v[:n_full]  # [N_full, Hkv, D]

    noisy_q = full_q_v[noisy_s:noisy_e]  # [N_noisy, Hq,  D]
    noisy_k = full_k_v[noisy_s:noisy_e]  # [N_noisy, Hkv, D]
    noisy_v = full_v_v[noisy_s:noisy_e]  # [N_noisy, Hkv, D]

    def _sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Maskless attention using cosmos_framework.model.attention() → [N_q, Hq*D]."""
        # K and V are built by concatenating the SAME [text | ctrl_i | noisy]
        # slices, so their sequence lengths are always equal. Under
        # torch.compile (fullgraph=True) those lengths are unbacked symints
        # (from data-dependent unpadding), and the attention frontend's
        # `if key_shape[1] != value_shape[1]` guard (attention/checks.py) cannot
        # be resolved symbolically. Assert the invariant so Dynamo can discharge
        # the guard statically instead of raising a data-dependent error.
        torch._check(k.shape[0] == v.shape[0])
        n_q, n_kv = q.shape[0], k.shape[0]

        # These lengths come from data-dependent unpadding, so they are unbacked
        # symints under torch.compile. Backend validation checks require positive
        # lengths, and cuDNN specifically rejects KV length 1. This path builds
        # KV as [text | ctrl_i | noisy], where ctrl_i and noisy are non-empty for
        # valid multi-control packs, so assert the stronger invariant. Without
        # these, Dynamo cannot discharge them against unbacked symints.
        torch._check(n_q > 0)
        torch._check(n_kv > 1)

        # No varlen metadata on purpose. Every tensor here was unpadded above and
        # each pass is a single (batch=1) sequence, so cumulative offsets would be
        # exactly [0, n] -- one range spanning the whole tensor, constraining
        # nothing its shape does not already. Passing them is not free: the
        # frontend derives `is_varlen` from their presence and feeds it to
        # `choose_backend`, and cuDNN declines varlen outright, so the varlen form
        # silently fell through to NATTEN. Dropping it lets cuDNN take this path:
        # measured on GB200, 1.3x-3.7x faster per call and 1.2x-1.9x over the
        # whole function, and no further from a float64 reference than the NATTEN
        # path it replaces.
        res = attention(
            q.unsqueeze(0),  # [1, N_q,  Hq,  D]
            k.unsqueeze(0),  # [1, N_kv, Hkv, D]
            v.unsqueeze(0),  # [1, N_kv, Hkv, D]
        )  # [1, N_q, Hq, D]
        return res.squeeze(0).flatten(-2, -1)  # type: ignore  # [N_q, Hq*D]

    # ── 3. N independent single-control passes ────────────────────────────────
    # For each control i: KV = [text | ctrl_i | noisy] — maskless SDPA.
    # ctrl_i attends to [text, ctrl_i, noisy] → stored directly in full_out.
    # noisy  attends to [text, ctrl_i, noisy] → accumulated as weighted sum.
    full_out_v = full_q_v.new_zeros(n_full, causal_out.shape[-1])
    noisy_out_acc: torch.Tensor | None = None

    for i, (cs, ce) in enumerate(ctrl_ranges):
        ctrl_k_i = full_k_v[cs:ce]
        ctrl_v_i = full_v_v[cs:ce]
        ctrl_q_i = full_q_v[cs:ce]

        # KV context for this pass: [text | ctrl_i | noisy]
        kv_k_i = torch.cat([causal_k_v, ctrl_k_i, noisy_k], dim=0)
        kv_v_i = torch.cat([causal_v_v, ctrl_v_i, noisy_v], dim=0)

        # ctrl_i output — stored directly
        full_out_v[cs:ce] = _sdpa(ctrl_q_i, kv_k_i, kv_v_i)

        # noisy output for pass i — accumulate weighted sum
        noisy_out_i = _sdpa(noisy_q, kv_k_i, kv_v_i)
        if noisy_out_acc is None:
            noisy_out_acc = weights[i] * noisy_out_i
        else:
            noisy_out_acc = noisy_out_acc + weights[i] * noisy_out_i

    assert noisy_out_acc is not None
    full_out_v[noisy_s:noisy_e] = noisy_out_acc

    # Re-pad to original shape so downstream layers see consistent tensor sizes.
    full_out = full_q.new_zeros(full_q.shape[0], full_out_v.shape[-1])
    full_out[:n_full] = full_out_v

    return from_mode_splits(causal_out, full_out, packed_query_states)


def dispatch_attention(
    packed_query_states: SequencePack,
    packed_key_states: SequencePack,
    packed_value_states: SequencePack,
    attention_mask: SplitInfo,
    natten_metadata: dict | None = None,
    memory_value: MemoryValue | None = None,
    packed_key_states_normalized: SequencePack | None = None,
) -> tuple[SequencePack, KVToStore | None]:
    assert memory_value is None, "Base dispatch_attention does not handle MemoryValue"
    if not _is_split_info_compatible(attention_mask):
        raise TypeError(f"Unsupported attention metadata: {type(attention_mask)}")
    if attention_mask.control_stream_token_ranges is not None:
        output = multi_control_two_way_attention(
            packed_query_states,
            packed_key_states,
            packed_value_states,
            attention_mask,
        )
    elif attention_mask.is_three_way:
        output = three_way_attention(
            packed_query_states,
            packed_key_states,
            packed_value_states,
            natten_metadata=natten_metadata,
            attention_meta=attention_mask,
            packed_key_states_normalized=packed_key_states_normalized,
        )
    else:
        output = two_way_attention(
            packed_query_states,
            packed_key_states,
            packed_value_states,
            packed_key_states_normalized=packed_key_states_normalized,
            # getattr because _is_split_info_compatible also accepts duck-typed metadata that
            # predates these fields.
            flex_block_mask=getattr(attention_mask, "flex_block_mask", None),
            flex_backend=getattr(attention_mask, "flex_backend", None),
        )
    return output, None


def build_packed_sequence(
    joint_attn_implementation: str,
    *,
    packed_sequence: torch.Tensor,
    attn_modes: list[str],
    split_lens: list[int],
    sample_lens: list[int],
    packed_und_token_indexes: torch.LongTensor,
    packed_gen_token_indexes: torch.LongTensor,
    num_heads: int,
    head_dim: int,
    num_layers: int,
    token_shapes: Sequence[tuple[int, ...]] | None = None,
    natten_parameter_list: list | None = None,
    is_image_batch: bool = False,
    cp_world_size: int = 1,
    video_temporal_causal: bool = False,
    skip_natten_metadata: bool = False,
    vision_token_shapes: list[tuple[int, int, int]] | None = None,
    action_token_shapes: list[tuple[int, ...]] | None = None,
    num_action_tokens_per_supertoken: int = 0,
    null_action_supertokens: bool = False,
    pad_for_cuda_graphs: bool = False,
    full_seq_alignment: int = 1,
    causal_seq_alignment: int = 1,
    prepared_metadata: SequencePackMetadata | None = None,
) -> tuple[SequencePack, AttentionMaskType, list | None]:
    """
    Build the model input pack and attention meta for joint attention.
    Returns a tuple: (input_pack, attention_meta, natten_metadata_list).

    ``full_seq_alignment`` and ``causal_seq_alignment`` pad the full (GEN) and causal (UND)
    streams up to a multiple of themselves; pass the matching two properties of the
    ``FlexBackend`` when the GEN tower runs FlexAttention, which keys GEN queries against
    the fused ``[UND | GEN]`` stream and so needs each half aligned to the block that
    tiles it.
    """
    device = packed_sequence.device
    natten_metadata_list = None
    if joint_attn_implementation == "two_way":
        attention_meta = SplitInfo(
            split_lens=split_lens,
            attn_modes=attn_modes,
            sample_lens=sample_lens,
            actual_len=int(packed_sequence.shape[0]),
        )
    elif joint_attn_implementation == "three_way":
        attention_meta = SplitInfo(
            split_lens=split_lens,
            attn_modes=attn_modes,
            sample_lens=sample_lens,
            actual_len=int(packed_sequence.shape[0]),
            is_three_way=True,
            vision_token_shapes=vision_token_shapes,
            action_token_shapes=action_token_shapes,
            num_action_tokens_per_supertoken=num_action_tokens_per_supertoken,
            null_action_supertokens=null_action_supertokens,
        )
        # Some memory-driven attention paths implement temporal visibility in
        # their own attention kernels; skip NATTEN metadata for those paths.
        if not skip_natten_metadata:
            # Temporal causal: encode (T, S) supertoken layout; spatial NATTEN: encode (H, W) layout.
            if video_temporal_causal:
                if vision_token_shapes is None:
                    raise ValueError(
                        "video_temporal_causal needs vision_token_shapes: the (T, H, W) layout per vision "
                        "item is what defines the supertoken boundaries the temporal mask is built from."
                    )
                natten_metadata_list = generate_temporal_causal_natten_metadata(
                    vision_token_shapes=vision_token_shapes,
                    num_action_tokens_per_supertoken=num_action_tokens_per_supertoken,
                    num_layers=num_layers,
                    head_dim=head_dim,
                    device=device,
                    dtype=packed_sequence.dtype,
                    requires_grad=packed_sequence.requires_grad,
                )
            else:
                natten_metadata_list = generate_natten_metadata(
                    token_shapes=token_shapes,
                    head_dim=head_dim,
                    num_layers=num_layers,
                    device=device,
                    dtype=packed_sequence.dtype,
                    requires_grad=packed_sequence.requires_grad,
                    natten_parameter_list=natten_parameter_list,
                )
    else:
        raise ValueError(
            f"Invalid joint_attn_implementation: {joint_attn_implementation}. Must be 'two_way' or 'three_way'."
        )

    input_pack = sequence_pack_from_packed_sequence(
        packed_sequence=packed_sequence,
        attn_modes=attn_modes,
        split_lens=split_lens,
        sample_lens=sample_lens,
        packed_und_token_indexes=packed_und_token_indexes.to(device),
        packed_gen_token_indexes=packed_gen_token_indexes.to(device),
        is_image_batch=is_image_batch,
        cp_world_size=cp_world_size,
        pad_for_cuda_graphs=pad_for_cuda_graphs,
        full_seq_alignment=full_seq_alignment,
        causal_seq_alignment=causal_seq_alignment,
        prepared_metadata=prepared_metadata,
    )
    # Not needed anymore, can cause recompilations.
    input_pack.pop("split_lens", None)
    input_pack.pop("attn_modes", None)
    return input_pack, attention_meta, natten_metadata_list
