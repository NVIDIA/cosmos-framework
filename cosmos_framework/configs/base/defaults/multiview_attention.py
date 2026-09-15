# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Whether the multiview generation attention runs, on which backend, and under what mask.

Its own module rather than a section of ``model_config`` because the attention code reads these
values too, and importing ``model_config`` from there is not possible: it pulls in the reasoner
defaults, which reach the MoT attention modules, which reach this module itself. Keeping the
values here leaves one definition with nothing but ``attrs`` behind it, which either side can
import.
"""

from typing import Literal, get_args

import attrs

# Which RGB tokens an RGB token attends to, among those of its own sample.
#
# The noisy square is the largest quadrant of the multiview mask -- every other rule is already
# confined to a single ``(frame, view)`` cell -- so this is the choice that sets what the mask
# costs. For a sample of ``V`` views by ``F`` frames per view by ``S`` spatial tokens per cell,
# that quadrant holds ``(V*F*S)**2`` pairs, of which each scope keeps:
#
# * ``"all_views"``: all of them. Every camera sees every other one at every instant. This is
#   the default. Cost is (FVS)^2.
# * ``"same_view"``: ``1/V`` of them. Each camera only attends to its own noisy tokens. Cost
#   is V*(FS)^2.
# * ``"decomposed"``: Each camera attends to its own noisy tokens plus the same frame in
#   every other camera, which decomposes the square into a temporal half and a spatial one.
#   Cost is V*(FS)^2 + F*(VS)^2. Rejected on a joint camera + LiDAR pack unless
#   ``decomposed_temporal_window_seconds`` is set: the two streams do not share a frame
#   index, but they do share real capture time, which the window compares instead.
#
# Read by the ``flex_*`` backends only. The ``"dense"`` backend is its own attention pattern
# and does not take a scope -- see ``BackendPreference``.
AttentionScope = Literal["all_views", "same_view", "decomposed"]

# The scopes of ``AttentionScope`` at runtime, which the annotation itself is not.
ATTENTION_SCOPES = get_args(AttentionScope)

# Which attention the multiview generation stream runs as.
#
# * ``"dense"``: the maskless three-pass decomposition -- same view across all frames, same
#   frame across all views, and gen->und -- merged by log-sum-exp, with no mask anywhere. See
#   ``models.mot.multiview_dense_attention.multiview_dense_attention``.
# * ``"flex_flash"``: the masked FlexAttention call on FlashAttention-4 (CuTeDSL) kernels.
# * ``"flex_triton"``: the same masked call on FlexAttention's Triton kernels. Available by
#   construction, so it is what everything else falls back to.
# * ``"auto"``: ``"flex_flash"`` where FA4 is available, else ``"flex_triton"``. The folds
#   rank last and are never reached, Triton always resolving, which is what keeps "auto" a
#   choice of kernels rather than of attention. ``"dense"`` is opt-in, by name.
#
# ``"dense"`` is NOT a faster spelling of ``mask.attention_scope="decomposed"``: its two sensor
# passes overlap on the query's own ``(view, frame)`` cell, and merging double-weights it where
# the mask counts it once -- measured ~22% apart on a small sample. It is a distinct attention
# pattern, so it is a choice about what to train and not only about how fast to run: a
# checkpoint trained under a mask is not one ``"dense"`` can serve, and the reverse holds too.
#
# That is why availability here is a property of the *config* rather than of the batch, and why
# a batch the folds cannot express raises instead of quietly taking a mask. Naming ``"dense"``
# fixes what a run trains for its whole duration. See ``models.mot.multiview_dense_attention.dense_unavailable_reason``, which owns
# that verdict because every condition in it is a fact about what the folds express, and
# ``models.mot.multiview_attention.resolve_multiview_backend``, which acts on it.
BackendPreference = Literal["auto", "dense", "flex_triton", "flex_flash"]

# The preferences of ``BackendPreference`` at runtime, which the annotation itself is not.
BACKEND_PREFERENCES = get_args(BackendPreference)

# The backends ``BackendPreference`` resolves to, i.e. everything but ``"auto"``.
ResolvedBackend = Literal["dense", "flex_triton", "flex_flash"]

# The subset of ``BackendPreference`` that names a *mask geometry*. ``"dense"`` is absent
# because it builds no mask: it is an attention pattern, and its padding alignments come from
# whichever flex geometry the host admits. Callers that only ever build a mask -- the benchmark's
# flex rows, and ``flex_attention.resolve_flex_backend`` -- take this rather than the full
# preference, so ``"dense"`` is rejected by the type instead of at the call.
FlexGeometryPreference = Literal["auto", "flex_triton", "flex_flash"]

# The preferences of ``FlexGeometryPreference`` at runtime, which the annotation itself is not.
FLEX_GEOMETRY_PREFERENCES = get_args(FlexGeometryPreference)


@attrs.define(slots=False)
class MultiviewAttentionMaskConfig:
    """What the multiview attention mask lets the generated tokens see.

    Only read when ``enabled`` is on and a ``flex_*`` backend is selected; ordinary dense
    attention has no notion of a view, and the ``"dense"`` multiview backend expresses its
    rules as a partition rather than as a mask.
    """

    # Which RGB tokens of its sample an RGB token attends to, independent of whether the
    # query or key is conditioning. Cross-view attention is what lets the rig agree with
    # itself, so the full square is the default; the narrower scopes buy attention that grows
    # with the rig rather than with its square, per the comment above. Never widens a WSM
    # (World Scenario Map) control token's reach, which is always its own view -- see
    # flex_attention.build_multiview_flex_metadata's ``is_control_per_item``, which the
    # network derives per generation stream, and which a batch without a control stream
    # leaves empty.
    attention_scope: AttentionScope = attrs.field(
        default="all_views",
        validator=attrs.validators.in_(ATTENTION_SCOPES),
    )

    # Only read under attention_scope="decomposed". Replaces that scope's temporal half --
    # "the query's own frame index" -- with "any key within this many seconds at or before the
    # query's own capture time", i.e. 0 <= query_timestamp - key_timestamp <= this value. None
    # (the default) keeps the frame-index form, which only agrees across sensors that share one
    # clock; a joint camera + LiDAR pack needs a window instead; see
    # flex_attention.build_multiview_flex_metadata and ._multiview_pair_predicate.
    #
    # Setting it rules the ``"dense"`` backend out: a sliding window is directed and
    # overlapping, and only an equivalence relation can be an unmasked pass. The decomposition
    # reaches across sensors by quantising capture time onto the anchor item's frame grid
    # instead, which needs no window.
    decomposed_temporal_window_seconds: float | None = attrs.field(
        default=None,
        validator=attrs.validators.optional(attrs.validators.ge(0)),
    )

    # Let a control query attend to every non-control sensor key in the same view,
    # across all frames. False preserves the existing one-way sensor-to-control
    # connectivity. Applies equally to camera and LiDAR control/target streams.
    control_attends_sensor: bool = False


@attrs.define(slots=False)
class MultiviewAttentionConfig:
    """How the multiview GEN attention is computed, for a run that asked for it.

    Whether it runs at all is ``ModelConfig.joint_attn_implementation == "multiview"``. This
    config is read only then, so none of its fields mean anything on their own -- which is why
    there is no ``enabled`` here to disagree with the pathway.
    """

    # Which attention the multiview GEN pass runs as; see ``BackendPreference`` for what each
    # name means. Read only under ``joint_attn_implementation="multiview"``, which is what
    # selects multiview attention at all -- this config describes *how* it runs, never whether.
    #
    # "auto" makes the choice a property of the host as much as of the config: the same
    # experiment can get different kernels, a different padded sequence length and different
    # rounding depending on the image it runs in, so a run that has to stay bit-comparable with
    # an earlier one pins "flex_triton" instead. "flex_flash" and "dense" both fail the run
    # where they are unavailable rather than falling back, since a silent fall back from "dense"
    # would train a different distribution under the same config.
    backend: BackendPreference = attrs.field(
        default="auto",
        validator=attrs.validators.in_(BACKEND_PREFERENCES),
    )

    # What a ``flex_*`` backend's mask lets the noisy tokens attend to.
    mask: MultiviewAttentionMaskConfig = MultiviewAttentionMaskConfig()
