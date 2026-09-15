# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""The multiview attention pathway: its shared UND half, and the GEN half it delegates.

A multiview run's UND self-attention is the same computation whichever GEN attention it pairs
with, per-view caption boundaries included, so it lives here once and both GEN halves are handed
its result. What differs is the GEN pass alone: :mod:`~...models.mot.flex_attention`'s masked
call over the fused ``[UND | GEN]`` stream, or :mod:`~...models.mot.multiview_dense_attention`'s
maskless folds. Keeping the shared half here is what lets those two modules each implement only
the part that is actually theirs.

The dependency runs one way -- this imports both of them and neither imports this -- which is
also why the GEN halves take and return plain tensors rather than a ``SplitInfo``: that type
lives in ``attention``, which imports this.

This module also owns which of the two a run takes, in ``resolve_multiview_backend``.

Its own module because the choice spans them: ``"dense"`` is
:mod:`~...models.mot.multiview_dense_attention`'s maskless folds and the ``flex_*`` backends are
:mod:`~...models.mot.flex_attention`'s masked call, so the decision belongs to neither. Keeping
it in ``flex_attention`` made that module the arbiter of a backend it does not implement.

The dependency runs one way -- this imports ``flex_attention`` for the mask geometry and nothing
imports this back -- and only through its public surface, so the two stay separable.
"""

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
    flash_backend_unavailable_reason,
    flex_attention,
    resolve_flex_backend,
)
from cosmos_framework.model.generator.mot.multiview_dense_attention import (
    MultiviewDensePlan,
    dense_unavailable_reason,
    multiview_dense_gen_attention,
)
from cosmos_framework.data.generator.sequence_packing.runtime import (
    SequencePack,
    from_mode_splits,
    get_caption_seq_offsets,
    get_causal_seq,
    get_full_only_seq,
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
          installed uses it -- and never a choice of attention, because ``"dense"`` is a different
          pattern and a default that switched what a run trains on nothing more than which package
          an image carries would not be a safe default. Triton is available by construction, so the
          folds are never actually reached here: ``"dense"`` is opt-in, by name. The flip side of
          the kernel choice remains, that the same config on a host without FA4 runs different
          kernels at a different padded length with different rounding, so a run that has to stay
          bit-comparable with another pins ``"flex_triton"`` rather than trusting the environments
          to match.
        * ``"dense"`` demands the maskless folds and raises if the config rules them out.
        * ``"flex_triton"`` pins FlexAttention's Triton kernels, ignoring what is installed.
        * ``"flex_flash"`` demands FA4 and raises if it cannot be used, for a benchmark or a test
          that is meaningless on the other backend.

    Whether the folds can serve ``config`` at all is asked of them directly, through
        :func:`~...models.mot.multiview_dense_attention.dense_unavailable_reason`, which answers with
        its reason rather than a bool so a pinned ``"dense"`` can fail with the cause.

        That verdict is deliberately a property of the config and not of a batch: ``"dense"`` is a different
        attention pattern rather than a faster one, so which one a run trains under is fixed here,
        once. A batch that then cannot be expressed without a mask raises in
        ``_multiview_dense_geometry`` rather than quietly taking the mask.

    ``"dense"`` comes back with no geometry at all. A ``FlexBackend`` describes the block a
        mask is built at and the padding that block needs, and the folds build no mask: their
        partitions cover whatever padding the pack has, so they impose no alignment of their own.
        Returning one anyway would hand callers a block size that describes nothing they run, and
        would pad the GEN stream to a mask boundary no kernel ever reads. Context-parallel
        divisibility and CUDA-graph bucketing do not come from here -- ``_get_padded_size`` folds
        those in separately -- so dropping it costs the folds nothing they need.

        Returns:
            ``(backend, flex_geometry)`` -- the attention this run takes, and the geometry the
            packer and any mask must agree with, or ``None`` under ``"dense"``, which has neither.

        Raises:
            ValueError: for an unknown ``preference``; for ``"dense"`` when the config rules it
                out; or for ``"flex_flash"`` when the backend is unavailable -- each with the
                reason.
    """
    if preference not in BACKEND_PREFERENCES:
        raise ValueError(f"Unknown multiview attention backend {preference!r}; expected one of {BACKEND_PREFERENCES}.")

    # Every geometry below comes from ``resolve_flex_backend``, whose own ``"auto"`` is exactly
    # the "FA4 where the host has it" rule this one wants for the mask half. The reason is read
    # separately because the *name* of the chosen backend, not just its geometry, is returned.
    flash_reason = flash_backend_unavailable_reason(device)
    dense_reason = dense_unavailable_reason(config)

    if preference == "dense":
        if dense_reason is not None:
            raise ValueError(
                f"backend='dense' asks for the maskless folds, but {dense_reason}. "
                "Pin a flex_* backend to run the mask instead -- but note that is different attention, "
                "not the same attention computed differently."
            )
        return "dense", None
    if preference == "flex_triton":
        return "flex_triton", resolve_flex_backend(device, "flex_triton")
    if preference == "flex_flash":
        # Raises with the reason where FA4 is unavailable, which is this preference's contract.
        return "flex_flash", resolve_flex_backend(device, "flex_flash")
    # "auto", in rank order: FA4, then Triton, then the folds. The third rank is vacuous --
    # Triton always resolves -- which is the point rather than an oversight: it is what keeps
    # "auto" from ever changing which attention a run trains. ``dense_reason`` is read above for
    # an explicit ``"dense"`` and deliberately not consulted here.
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
    dense_plan: MultiviewDensePlan | None = None,
    flex_block_mask: BlockMask | None = None,
    flex_backend: FlexBackend | None = None,
    packed_key_states_normalized: SequencePack | None = None,
) -> SequencePack:
    """Multiview attention over one pack: the shared UND half, then the GEN half it was given.

    Exactly one of ``dense_plan`` and ``flex_block_mask`` describes the GEN pass, which is the
    only thing the two multiview backends disagree about. Which one arrives was decided once,
    per run, by :func:`resolve_multiview_backend` -- not per batch, because the two are
    different attention rather than two speeds of one.

    Raises:
        ValueError: when the GEN half is described by neither or by both, or when a mask arrives
            without the backend it was built for.
    """
    if (dense_plan is None) == (flex_block_mask is None):
        raise ValueError(
            "Multiview attention takes exactly one description of its GEN pass: a dense_plan for "
            "the maskless folds or a flex_block_mask for the masked call, and got "
            f"{'both' if dense_plan is not None else 'neither'}."
        )
    # The generator's full attention takes the normed keys when provided, else the standard ones.
    packed_key_normalized = (
        packed_key_states_normalized if packed_key_states_normalized is not None else packed_key_states
    )

    # The GEN half first. The two are independent -- the UND output feeds nothing here -- and
    # this order lets a plan that does not describe this pack be refused by the folds, which own
    # that invariant, rather than surfacing as whatever the UND pass makes of the same mismatch.
    if dense_plan is not None:
        full_out = multiview_dense_gen_attention(
            packed_query_states,
            packed_key_states,
            packed_value_states,
            plan=dense_plan,
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
