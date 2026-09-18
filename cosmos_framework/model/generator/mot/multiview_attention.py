# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""The multiview attention pathway: its shared UND half, and the GEN half it delegates.

A multiview run's UND self-attention is the same computation whichever GEN attention it pairs
with, per-view caption boundaries included, so it lives here once and both GEN halves are handed
its result. What differs is the GEN pass alone: :mod:`~...models.mot.flex_attention`'s masked
call over the fused ``[UND | GEN]`` stream, or :mod:`~...models.mot.multiview_maskless_attention`'s
maskless folds. Keeping the shared half here is what lets those two modules each implement only
the part that is actually theirs.

The dependency runs one way -- this imports both of them and neither imports this -- which is
also why the GEN halves take and return plain tensors rather than a ``SplitInfo``: that type
lives in ``attention``, which imports this.

This module also owns which of the two a run takes, in ``resolve_multiview_backend``, and the
properties a pack has to have before either is chosen: that its captions state a single layout
(``reject_mixed_caption_layouts``) and that no sample of it is cut off from the captions
entirely (``reject_samples_reading_no_caption``). Both are properties of the pack rather than of
the attention either backend runs, and both would otherwise be answered twice, differently.

Its own module because the choice spans them: ``"maskless"`` is
:mod:`~...models.mot.multiview_maskless_attention`'s maskless folds and the ``flex_*`` backends are
:mod:`~...models.mot.flex_attention`'s masked call, so the decision belongs to neither. Keeping
it in ``flex_attention`` made that module the arbiter of a backend it does not implement.

The dependency runs one way -- this imports ``flex_attention`` for the mask geometry and nothing
imports this back -- and only through its public surface, so the two stay separable.
"""

from collections.abc import Sequence
from typing import Any

import torch
from torch.nn.attention.flex_attention import BlockMask

from cosmos_framework.model.attention import attention
from cosmos_framework.model.attention.masks import CausalType
from cosmos_framework.configs.base.defaults.multiview_attention import (
    BACKEND_PREFERENCES,
    MultiviewAttentionConfig,
    ResolvedBackend,
)
from cosmos_framework.model.generator.mot.flex_attention import (
    FlexBackend,
    SensorMaskItem,
    flash_backend_unavailable_reason,
    flex_attention,
    resolve_flex_backend,
)
from cosmos_framework.model.generator.mot.multiview_maskless_attention import (
    MultiviewMasklessPlan,
    maskless_unavailable_reason,
    multiview_maskless_gen_attention,
)
from cosmos_framework.data.generator.sequence_packing.runtime import (
    SequencePack,
    from_mode_splits,
    get_caption_seq_offsets,
    get_causal_seq,
    get_full_only_seq,
)


def reject_mixed_caption_layouts(caption_view_ids: Sequence[Sequence[int]]) -> None:
    """Refuse a pack that is captioned per view and per sample at once.

    A caption names the camera view it describes, or ``-1`` for the sample-level caption that
    describes the whole rig -- and those are layouts of a *batch*, not choices a sample makes on
    its own. :func:`resolve_caption_scope` takes one ``per_view_captions`` for the pack, the
    causal stream's per-caption boundaries are built for the pack, and both backends label every
    token from that single decision. A pack carrying both kinds has no one answer to give them:
    the folds would let the ``-1`` caption reach every view while the mask refuses the same
    sample outright, which is the two backends disagreeing about one batch.

    Here rather than in either backend for the same reason the choice between them is here: it
    is a property of the pack, settled before anything is built from the captions, and the
    caller runs it before the mask's items and the folds' plan alike.

    Nothing that packs a sequence produces such a mix -- ``_apply_per_view_caption_plan`` sets
    the layout for every sample of the batch or none of them -- so this states that invariant
    where a pack built by hand or by a future packer would meet it, rather than leaving it to be
    discovered as a backend disagreement.

    Raises:
        ValueError: when one sample carries both kinds, or when the batch carries a sample-level
            sample beside a per-view one.
    """
    sample_level: list[int] = []
    per_view: list[int] = []
    for sample_idx, view_ids in enumerate(caption_view_ids):
        own_sample_level = [view_id for view_id in view_ids if view_id < 0]
        own_per_view = [view_id for view_id in view_ids if view_id >= 0]
        if own_sample_level and own_per_view:
            raise ValueError(
                f"Sample {sample_idx} carries both a sample-level caption (view id -1) and "
                f"captions written for views {own_per_view}. A caption layout is the batch's, so "
                "a sample is described either by one caption covering its whole rig or by one "
                "caption per camera view -- not by both."
            )
        if own_sample_level:
            sample_level.append(sample_idx)
        elif own_per_view:
            # A sample with no caption at all states no layout, so it is neither side of this
            # comparison; the backends refuse it on their own terms.
            per_view.append(sample_idx)
    if sample_level and per_view:
        raise ValueError(
            f"This pack mixes caption layouts: sample(s) {sample_level} carry a sample-level "
            f"caption (view id -1) while sample(s) {per_view} carry captions written for "
            "individual views. The layout decides how every token of the pack is keyed against "
            "the captions, so it has to be the same for every sample: pack the whole batch with "
            "separate_view_text_tokenization, or none of it."
        )


def reject_samples_reading_no_caption(sensor_mask_items: Sequence[Sequence[SensorMaskItem]]) -> None:
    """Refuse a sample whose every item is cut off from the captions.

    ``lidar_attends_captions=False`` withholds from a sweep the text its *cameras* are described
    by. A sample with no camera item has no such text to withhold: every one of its items is
    text-free, so none of its tokens reads a caption and it trains as unconditioned generation --
    a data/config mismatch rather than anything either backend should express.

    Checked before the backend is chosen, because it is a property of the items both backends
    read rather than of the attention either of them runs. The mask would express it by masking
    every gen->und edge of those rows away and the folds by leaving the pass no rows at all; both
    are silent, and silently training without conditioning is the failure worth naming.
    """
    for sample_idx, sample_items in enumerate(sensor_mask_items):
        if sample_items and all(item.caption_access == "no_captions" for item in sample_items):
            raise ValueError(
                f"Sample {sample_idx} of this pack would generate with no text conditioning at "
                "all: every one of its items is cut off from the captions by "
                "lidar_attends_captions=False, so no token of it reads one. That flag withholds "
                "from a LiDAR sweep the captions its cameras are described by, and this sample "
                "carries no camera item for any caption to describe. Either set "
                "model.config.multiview_attention.mask.lidar_attends_captions=true for this "
                "data, or pack the LiDAR stream alongside the camera stream it was captured "
                "with."
            )


def resolve_multiview_backend(
    device: torch.device,
    preference: str = "auto",
    *,
    config: MultiviewAttentionConfig,
) -> tuple[ResolvedBackend, FlexBackend | None]:
    """Which multiview attention this run takes, and the mask geometry that choice forces.

        ``preference`` is a run's policy, not its outcome:

        * ``"auto"`` ranks the masks first: FA4 where the host has it, else Triton, and the folds
          last. That ordering keeps ``"auto"`` a choice of *kernels* -- a host that has FA4
          installed uses it -- and never a choice of attention, because ``"maskless"`` is a different
          pattern and a default that switched what a run trains on nothing more than which package
          an image carries would not be a safe default. Triton is available by construction, so the
          folds are never actually reached here: ``"maskless"`` is opt-in, by name. The flip side of
          the kernel choice remains, that the same config on a host without FA4 runs different
          kernels at a different padded length with different rounding, so a run that has to stay
          bit-comparable with another pins ``"flex_triton"`` rather than trusting the environments
          to match.
        * ``"maskless"`` demands the maskless folds and raises if the config rules them out.
        * ``"flex_triton"`` pins FlexAttention's Triton kernels, ignoring what is installed.
        * ``"flex_flash"`` demands FA4 and raises if it cannot be used, for a benchmark or a test
          that is meaningless on the other backend.

    Whether the folds can serve ``config`` at all is asked of them directly, through
        :func:`~...models.mot.multiview_maskless_attention.maskless_unavailable_reason`, which answers with
        its reason rather than a bool so a pinned ``"maskless"`` can fail with the cause.

        That verdict is deliberately a property of the config and not of a batch: ``"maskless"`` is a different
        attention pattern rather than a faster one, so which one a run trains under is fixed here,
        once. A batch that then cannot be expressed without a mask raises in
        ``_multiview_maskless_geometry`` rather than quietly taking the mask.

    ``"maskless"`` comes back with no geometry at all. A ``FlexBackend`` describes the block a
        mask is built at and the padding that block needs, and the folds build no mask: their
        partitions cover whatever padding the pack has, so they impose no alignment of their own.
        Returning one anyway would hand callers a block size that describes nothing they run, and
        would pad the GEN stream to a mask boundary no kernel ever reads. Context-parallel
        divisibility and CUDA-graph bucketing do not come from here -- ``_get_padded_size`` folds
        those in separately -- so dropping it costs the folds nothing they need.

        Returns:
            ``(backend, flex_geometry)`` -- the attention this run takes, and the geometry the
            packer and any mask must agree with, or ``None`` under ``"maskless"``, which has neither.

        Raises:
            ValueError: for an unknown ``preference``; for ``"maskless"`` when the config rules it
                out; or for ``"flex_flash"`` when the backend is unavailable -- each with the
                reason.
    """
    if preference not in BACKEND_PREFERENCES:
        raise ValueError(f"Unknown multiview attention backend {preference!r}; expected one of {BACKEND_PREFERENCES}.")

    # Every geometry below comes from ``resolve_flex_backend``, whose own ``"auto"`` is exactly
    # the "FA4 where the host has it" rule this one wants for the mask half. The reason is read
    # separately because the *name* of the chosen backend, not just its geometry, is returned.
    flash_reason = flash_backend_unavailable_reason(device)
    maskless_reason = maskless_unavailable_reason(config)

    if preference == "maskless":
        if maskless_reason is not None:
            raise ValueError(
                f"backend='maskless' asks for the maskless folds, but {maskless_reason}. "
                "Pin a flex_* backend to run the mask instead -- but note that is different attention, "
                "not the same attention computed differently."
            )
        return "maskless", None
    if preference == "flex_triton":
        return "flex_triton", resolve_flex_backend(device, "flex_triton")
    if preference == "flex_flash":
        # Raises with the reason where FA4 is unavailable, which is this preference's contract.
        return "flex_flash", resolve_flex_backend(device, "flex_flash")
    # "auto", in rank order: FA4, then Triton, then the folds. The third rank is vacuous --
    # Triton always resolves -- which is the point rather than an oversight: it is what keeps
    # "auto" from ever changing which attention a run trains. ``maskless_reason`` is read above for
    # an explicit ``"maskless"`` and deliberately not consulted here.
    if flash_reason is None:
        return "flex_flash", resolve_flex_backend(device, "flex_flash")
    return "flex_triton", resolve_flex_backend(device, "flex_triton")


def und_self_attention(
    packed_query_states: SequencePack,
    packed_key_states: SequencePack,
    packed_value_states: SequencePack,
) -> torch.Tensor:
    """The UND stream attending causally within itself, shared by both GEN halves.

    A single sample with no per-caption boundaries is one document, so the offsets would be a
    single ``[0, n_und]`` range and the dense API computes the same thing without them. Padding
    is appended after every real token, so a real causal query at ``i`` only ever reaches keys
    ``<= i``, all real. Several samples need the ranges to stay separate documents, and take the
    pad-segment offsets for the reason ``has_pad_segment`` gives: varlen leaves rows outside
    every range unwritten in both directions.

    Per-view captions cut the causal stream finer than one document per sample: each caption is
    its own, so no caption attends another. One tensor is handed to both sides on purpose --
    ``use_dont_care_mask`` is an identity check, and two equal tensors would silently drop the
    pass to ``CausalType.TopLeft``.

    Returns:
        ``[N_und, heads * head_dim]``, in packed order.
    """
    causal_q, causal_q_offsets = get_causal_seq(packed_query_states)
    causal_k, causal_k_offsets = get_causal_seq(packed_key_states)
    causal_v, _ = get_causal_seq(packed_value_states)
    max_causal_len = packed_query_states["max_causal_len"]

    caption_self_offsets = get_caption_seq_offsets(packed_query_states)
    if caption_self_offsets is not None:
        causal_q_offsets, max_causal_len = caption_self_offsets
        causal_k_offsets = causal_q_offsets
    use_dont_care_mask = causal_q_offsets is causal_k_offsets

    causal_varlen_kwargs: dict[str, Any] = dict(
        cumulative_seqlen_Q=causal_q_offsets,
        cumulative_seqlen_KV=causal_k_offsets,
        max_seqlen_Q=max_causal_len,
        max_seqlen_KV=max_causal_len,
    )
    causal_res = attention(
        causal_q.unsqueeze(0),  # [1,N_und,heads,head_dim]
        causal_k.unsqueeze(0),  # [1,N_und,kv_heads,head_dim]
        causal_v.unsqueeze(0),  # [1,N_und,kv_heads,head_dim]
        is_causal=True,
        causal_type=CausalType.DontCare if use_dont_care_mask else CausalType.TopLeft,
        **causal_varlen_kwargs,
    )  # [1,N_und,heads,head_dim]
    return causal_res.squeeze(0).flatten(-2, -1)  # type: ignore  # [N_und,heads*head_dim]


def _masked_gen_attention(
    packed_query_states: SequencePack,
    packed_key_normalized: SequencePack,
    packed_value_states: SequencePack,
    *,
    causal_v: torch.Tensor,
    flex_block_mask: BlockMask,
    flex_backend: FlexBackend,
) -> torch.Tensor:
    """The GEN stream under the multiview mask, as one FlexAttention call.

    The mask keys GEN queries against ``[UND | GEN]``, so the two block-padded streams are
    concatenated in that order rather than gathered back into the interleaved pack order. The
    keys are the normalised ones -- UND normalisation is exactly what the GEN pass wants -- and
    the values the raw ones.

    No varlen offsets and no separate cross-attention term: padding carries the ``-1`` sentinel
    in the mask, so every row is written and only padding attends to padding.
    """
    causal_k_normalized, _ = get_causal_seq(packed_key_normalized)
    full_q, _ = get_full_only_seq(packed_query_states)
    full_k, _ = get_full_only_seq(packed_key_normalized)  # [N_full,heads,head_dim]
    full_v, _ = get_full_only_seq(packed_value_states)  # [N_full,heads,head_dim]
    full_res = flex_attention(
        full_q.unsqueeze(0),  # [1,N_full,heads,head_dim]
        torch.cat((causal_k_normalized, full_k)).unsqueeze(0),  # [1,N_und+N_full,heads,head_dim]
        torch.cat((causal_v, full_v)).unsqueeze(0),  # [1,N_und+N_full,heads,head_dim]
        flex_block_mask,
        flex_backend,
    )  # [1,N_full,heads,head_dim]
    return full_res.squeeze(0).flatten(-2, -1)  # [N_full,heads*head_dim]


def multiview_attention(
    packed_query_states: SequencePack,
    packed_key_states: SequencePack,
    packed_value_states: SequencePack,
    *,
    maskless_plan: MultiviewMasklessPlan | None = None,
    flex_block_mask: BlockMask | None = None,
    flex_backend: FlexBackend | None = None,
    packed_key_states_normalized: SequencePack | None = None,
) -> SequencePack:
    """Multiview attention over one pack: the shared UND half, then the GEN half it was given.

    Exactly one of ``maskless_plan`` and ``flex_block_mask`` describes the GEN pass, which is the
    only thing the two multiview backends disagree about. Which one arrives was decided once,
    per run, by :func:`resolve_multiview_backend` -- not per batch, because the two are
    different attention rather than two speeds of one.

    Raises:
        ValueError: when the GEN half is described by neither or by both, or when a mask arrives
            without the backend it was built for.
    """
    if (maskless_plan is None) == (flex_block_mask is None):
        raise ValueError(
            "Multiview attention takes exactly one description of its GEN pass: a maskless_plan for "
            "the maskless folds or a flex_block_mask for the masked call, and got "
            f"{'both' if maskless_plan is not None else 'neither'}."
        )
    # The generator's full attention takes the normed keys when provided, else the standard ones.
    packed_key_normalized = (
        packed_key_states_normalized if packed_key_states_normalized is not None else packed_key_states
    )

    # The GEN half first. The two are independent -- the UND output feeds nothing here -- and
    # this order lets a plan that does not describe this pack be refused by the folds, which own
    # that invariant, rather than surfacing as whatever the UND pass makes of the same mismatch.
    if maskless_plan is not None:
        full_out = multiview_maskless_gen_attention(
            packed_query_states,
            packed_key_states,
            packed_value_states,
            plan=maskless_plan,
            packed_key_states_normalized=packed_key_states_normalized,
        )
    else:
        if flex_backend is None:
            raise ValueError(
                "flex_block_mask needs the FlexBackend it was built for: which kernels run the "
                "mask is only correct at the block size it was built at, so the two are set "
                "together."
            )
        assert flex_block_mask is not None  # narrowed by the exclusivity check above
        causal_v, _ = get_causal_seq(packed_value_states)
        full_out = _masked_gen_attention(
            packed_query_states,
            packed_key_normalized,
            packed_value_states,
            causal_v=causal_v,
            flex_block_mask=flex_block_mask,
            flex_backend=flex_backend,
        )

    causal_out = und_self_attention(packed_query_states, packed_key_states, packed_value_states)
    return from_mode_splits(causal_out, full_out, packed_query_states)
