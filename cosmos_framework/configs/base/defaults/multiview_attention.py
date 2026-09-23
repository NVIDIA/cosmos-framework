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

# What an item is to its sample's captions, stated without reference to how many of them there
# are. Both backends describe their items with this, and both resolve it through
# :func:`resolve_caption_scope` below, so one rule cannot be expressed two ways -- which is what
# let the maskless folds ignore ``lidar_attends_captions`` when they derived the rule themselves.
#
# * ``"camera"``: one of the rig's cameras. It reads the caption written for its view where
#   there is one per view, and the sample's single caption otherwise. It is also the answer to
#   "which views must a per-view caption layout cover".
# * ``"all_captions"``: not one of the rig's cameras, and reads every caption of its sample -- a
#   LiDAR sweep, which fuses the whole rig rather than looking through one camera.
# * ``"no_captions"``: not one of the rig's cameras, and reads none of them -- the same sweep
#   under ``lidar_attends_captions=False``.
CaptionAccess = Literal["camera", "all_captions", "no_captions"]

# The accesses of ``CaptionAccess`` at runtime, which the annotation itself is not.
CAPTION_ACCESSES = get_args(CaptionAccess)

# Which of its sample's captions a token reads, once an access has been resolved against the
# batch's caption layout. Plain ints rather than an enum because the mask compares them against a
# tensor inside its traced ``mask_mod``, which may capture neither a Python scalar nor an object.
CAPTION_SCOPE_NONE = 0  # reads no caption at all
CAPTION_SCOPE_SAME_VIEW = 1  # reads the caption written for its own view, and no other
CAPTION_SCOPE_ALL = 2  # reads every caption of its sample


def resolve_caption_scope(access: CaptionAccess, *, per_view_captions: bool) -> int:
    """The ``CAPTION_SCOPE_*`` an item's access comes to under this batch's caption layout.

    Only a camera's answer depends on the layout, and only because the two layouts describe the
    same caption differently: with one caption per view it reads the one written for its own
    view, and with a single sample-level caption it reads that one, which describes the whole
    rig. A sweep reads every caption or none in either layout, by config.

    Shared by both backends deliberately. The mask resolves this per token into
    ``FlexMetadata.caption_scope`` and the maskless folds per same-view group into the run of
    captions that group is keyed against; going through one function is what keeps those two
    from drifting into different rules, as they had.
    """
    if access not in CAPTION_ACCESSES:
        raise ValueError(f"Unknown caption access {access!r}; expected one of {CAPTION_ACCESSES}.")
    if access == "no_captions":
        return CAPTION_SCOPE_NONE
    if access == "all_captions":
        return CAPTION_SCOPE_ALL
    return CAPTION_SCOPE_SAME_VIEW if per_view_captions else CAPTION_SCOPE_ALL


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
# Read by every backend, but not the same way. A ``flex_*`` backend expresses the scope as a mask.
# The ``"maskless"`` backend expresses ``"same_view"`` and ``"decomposed"`` as partitions of the
# GEN stream and refuses ``"all_views"``, which is not a partition at all -- so there the scope
# decides whether the cross-instant pass exists rather than describing one attention two ways. See
# ``BackendPreference`` and ``models.mot.multiview_maskless_attention.MASKLESS_ATTENTION_SCOPES``.
AttentionScope = Literal["all_views", "same_view", "decomposed"]

# The scopes of ``AttentionScope`` at runtime, which the annotation itself is not.
ATTENTION_SCOPES = get_args(AttentionScope)

# Which attention the multiview generation stream runs as.
#
# * ``"maskless"``: the maskless three-pass decomposition -- same view across all frames, same
#   frame across all views, and gen->und -- merged by log-sum-exp, with no mask anywhere. See
#   ``models.mot.multiview_maskless_attention.multiview_maskless_gen_attention``.
# * ``"flex_flash"``: the masked FlexAttention call on FlashAttention-4 (CuTeDSL) kernels.
# * ``"flex_triton"``: the same masked call on FlexAttention's Triton kernels. Available by
#   construction, so it is what everything else falls back to.
# * ``"auto"``: ``"flex_flash"`` where FA4 is available, else ``"flex_triton"``. The folds
#   rank last and are never reached, Triton always resolving, which is what keeps "auto" a
#   choice of kernels rather than of attention. ``"maskless"`` is opt-in, by name.
#
# ``"maskless"`` is NOT a faster spelling of ``mask.attention_scope="decomposed"``: its two sensor
# passes overlap on the query's own ``(view, frame)`` cell, and merging double-weights it where
# the mask counts it once -- measured ~22% apart on a small sample. It is a distinct attention
# pattern, so it is a choice about what to train and not only about how fast to run: a
# checkpoint trained under a mask is not one ``"maskless"`` can serve, and the reverse holds too.
#
# That is why availability here is a property of the *config* rather than of the batch, and why
# a batch the folds cannot express raises instead of quietly taking a mask. Naming ``"maskless"``
# fixes what a run trains for its whole duration. See ``models.mot.multiview_maskless_attention.maskless_unavailable_reason``, which owns
# that verdict because every condition in it is a fact about what the folds express, and
# ``models.mot.multiview_attention.resolve_multiview_backend``, which acts on it.
BackendPreference = Literal["auto", "maskless", "flex_triton", "flex_flash"]

# The preferences of ``BackendPreference`` at runtime, which the annotation itself is not.
BACKEND_PREFERENCES = get_args(BackendPreference)

# The backends ``BackendPreference`` resolves to, i.e. everything but ``"auto"``.
ResolvedBackend = Literal["maskless", "flex_triton", "flex_flash"]

# The subset of ``BackendPreference`` that names a *mask geometry*. ``"maskless"`` is absent
# because it builds no mask: it is an attention pattern, and its padding alignments come from
# whichever flex geometry the host admits. Callers that only ever build a mask -- the benchmark's
# flex rows, and ``flex_attention.resolve_flex_backend`` -- take this rather than the full
# preference, so ``"maskless"`` is rejected by the type instead of at the call.
FlexGeometryPreference = Literal["auto", "flex_triton", "flex_flash"]

# The preferences of ``FlexGeometryPreference`` at runtime, which the annotation itself is not.
FLEX_GEOMETRY_PREFERENCES = get_args(FlexGeometryPreference)


@attrs.define(slots=False)
class MultiviewAttentionMaskConfig:
    """What the multiview attention mask lets the generated tokens see.

    Only read when ``enabled`` is on and a ``flex_*`` backend is selected; ordinary dense
    attention has no notion of a view, and the ``"maskless"`` multiview backend expresses its
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
    # Setting it rules the ``"maskless"`` backend out: a sliding window is directed and
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

    # Whether the LiDAR stream of a joint camera + LiDAR pack reads its sample's captions.
    # True (the default) is the existing behaviour: a sweep is not one of the rig's cameras, so
    # under per-view captions it reads every camera's caption, and under the single sample-level
    # caption it reads that one. False drops the gen->und edges for LiDAR tokens entirely, so a
    # sweep is conditioned on the camera stream and its own control stream alone, with no text --
    # the ablation for whether the captions, which describe what the cameras see, help or
    # mislead the range prediction. Camera tokens keep their captions either way, and a pack
    # with no LiDAR stream is unaffected. See flex_attention.SensorMaskItem.attends_captions and
    # ._multiview_pair_predicate.
    lidar_attends_captions: bool = True


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
    # an earlier one pins "flex_triton" instead. "flex_flash" and "maskless" both fail the run
    # where they are unavailable rather than falling back, since a silent fall back from "maskless"
    # would train a different distribution under the same config.
    backend: BackendPreference = attrs.field(
        default="auto",
        validator=attrs.validators.in_(BACKEND_PREFERENCES),
    )

    # What a ``flex_*`` backend's mask lets the noisy tokens attend to.
    mask: MultiviewAttentionMaskConfig = MultiviewAttentionMaskConfig()
