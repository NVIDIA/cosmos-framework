# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Whether the multiview generation attention runs, on which backend, and under what mask.

Its own module rather than a section of ``model_config`` because the attention code reads these
values too, and importing ``model_config`` from there is not possible: it pulls in the reasoner
defaults, which reach the MoT attention modules, which reach this module itself. Keeping the
values here leaves one definition with nothing but ``attrs`` behind it, which either side can
import.
"""

import math
from collections.abc import Sequence
from typing import Any, Literal, get_args

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

# Shared by Flex's pair predicate and maskless window planning. Capture
# times are float32; include pairs that round just beyond either window boundary.
DECOMPOSED_TEMPORAL_WINDOW_EPS: float = 1e-4

# Signed key-time offsets from the query, in seconds: always explicit (start, end).
TemporalWindow = tuple[float, float]


def temporal_window_bounds(window: TemporalWindow | None) -> tuple[float, float] | None:
    """Validate explicit (start, end) bounds, including their YAML list representation."""
    if window is None:
        return None
    if (
        not isinstance(window, Sequence)
        or len(window) != 2
        or not all(isinstance(bound, (int, float)) and math.isfinite(bound) for bound in window)
    ):
        raise ValueError("decomposed_temporal_window_seconds requires two finite bounds (start, end)")
    start, end = window
    if start > end:
        raise ValueError("decomposed_temporal_window_seconds requires start <= end")
    return float(start), float(end)


def load_temporal_window(window: TemporalWindow | float | None) -> TemporalWindow | None:
    """Migrate legacy saved configs at construction, never inside attention code.

    The old scalar N admitted keys at or before the query within N seconds.
    Converting it to (-N, N) would silently add future keys to old checkpoints.
    """
    if isinstance(window, (int, float)):
        if not math.isfinite(window) or window < 0:
            raise ValueError("decomposed_temporal_window_seconds legacy scalar must be finite and non-negative")
        window = (-float(window), 0.0)
    return temporal_window_bounds(window)


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
# passes by default overlap on the query's own ``(view, frame)`` cell, and merging double-weights it where
# the mask counts it once -- measured ~22% apart on a small sample. It is a distinct attention
# pattern, so it is a choice about what to train and not only about how fast to run: a
# checkpoint trained under a mask is not one ``"maskless"`` can serve, and the reverse holds too.
# Opting into ``deduplicate_cross_view`` removes this overlap using disjoint dense
# cross-view segments. Without a temporal window it retains the maskless instant
# assignment; with a window it uses Flex's directed capture-time rule.
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

TemporalFrameWindow = tuple[int, int]


def _convert_temporal_frame_window(
    value: TemporalFrameWindow | Sequence[int] | None,
) -> TemporalFrameWindow | None:
    """Convert a two-integer sequence to a tuple, leaving ``None`` unchanged.

    Strings, booleans, non-integer bounds, and sequences of other lengths are
    rejected. Bound ordering is checked by the field validator.

    Examples:
        ``None`` remains ``None``.
        ``[-8, 8]`` becomes ``(-8, 8)``.
    """
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(
            "A temporal frame-window experiment override must be null or a two-integer list such as [-8,8]; "
            f"got {value!r}."
        )
    if len(value) != 2:
        raise ValueError(
            f"A temporal frame-window experiment override needs exactly two bounds, such as [-8,8]; got {value!r}."
        )
    if any(isinstance(bound, bool) or not isinstance(bound, int) for bound in value):
        raise TypeError(f"Temporal frame-window bounds must be integers; got {value!r}.")
    return value[0], value[1]


def _validate_temporal_frame_window(
    _instance: object,
    attribute: attrs.Attribute,
    value: TemporalFrameWindow | None,
) -> None:
    """Require ordered inclusive temporal-window bounds."""
    if value is None:
        return
    lower, upper = value
    if lower > upper:
        raise ValueError(f"{attribute.name} lower bound must not exceed its upper bound; got {value!r}.")


@attrs.define(slots=False)
class MultiviewAttentionMaskConfig:
    """What the multiview attention mask lets the generated tokens see.

    Read by both multiview backends. Ordinary dense attention has no notion of a
    view; the ``"maskless"`` backend expresses these rules as dense rectangles
    rather than materializing a token-level mask.
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
    # "the query's own frame index" -- with an inclusive (start, end) interval of key capture
    # times relative to the query: start <= key_timestamp - query_timestamp <= end.
    # Configs spell out both bounds: (-0.4, 0.0) is past-only,
    # (-0.2, 0.2) is symmetric with 0.4 s total width, (-0.4, 0.2) is asymmetric.
    # Saved legacy scalar N is converted at config construction to (-N, 0.0). None
    # (the default) keeps the frame-index form, which only agrees across sensors that share one
    # clock; a joint camera + LiDAR pack needs a window instead; see
    # flex_attention.build_multiview_flex_metadata and ._multiview_pair_predicate.
    #
    # Both maskless counting modes group queries by capture time. Exact mode
    # excludes own-view keys from the window pass; overlapping mode includes them,
    # counting own-view sensor keys in that window twice after merging with the
    # full-temporal same-view pass. None preserves midpoint-quantised instant groups.
    # OmegaConf rejects unions containing tuples even when the value is None.
    # Keep the serialized field untyped; the converter enforces TemporalWindow.
    decomposed_temporal_window_seconds: Any = attrs.field(default=None, converter=load_temporal_window)

    # Also let every cross-view query read its sample's first frame (capture time 0, the given
    # frame under I2V) on the other views, whatever the window says. It widens the window's key
    # times rather than adding a pass, so under ``deduplicate_cross_view`` a first frame that
    # already sits inside the window is still counted once and own-view keys stay out. Maskless
    # only, and only with a window: the mask has no such rule, and without a window the folds
    # group by instant rather than comparing capture times.
    decomposed_temporal_window_includes_first_frame: bool = False

    # Let a control query attend to every non-control sensor key in the same view,
    # across all frames. False preserves the existing one-way sensor-to-control
    # connectivity. Applies equally to camera and LiDAR control/target streams.
    control_attends_sensor: bool = False

    # Optional inclusive temporal windows for each same-view generation edge.
    # Sensor denotes the generated, non-control side for both camera and LiDAR streams.
    # A centered [-N, N] window (inclusive) uses up to 2N+1 frames and shifts inward at the
    # sequence boundaries to preserve that width. A causal [-N, 0] window includes
    # the current frame and up to N earlier frames, for at most N+1 frames, and clips
    # at the beginning of the sequence. Passing None disables temporal windowing
    # for that edge, so every frame in the same view is used. Cross-view
    # same-instant attention is unchanged.
    # Experiments using selective activation checkpointing must also override
    # model.config.activation_checkpointing.save_only_marked_ops=False when enabling
    # any temporal window; marked-only checkpointing is not implemented for this path.
    sensor_to_sensor_window: TemporalFrameWindow | None = attrs.field(
        default=None,
        converter=_convert_temporal_frame_window,
        validator=_validate_temporal_frame_window,
    )
    sensor_to_control_window: TemporalFrameWindow | None = attrs.field(
        default=None,
        converter=_convert_temporal_frame_window,
        validator=_validate_temporal_frame_window,
    )
    control_to_control_window: TemporalFrameWindow | None = attrs.field(
        default=None,
        converter=_convert_temporal_frame_window,
        validator=_validate_temporal_frame_window,
    )
    control_to_sensor_window: TemporalFrameWindow | None = attrs.field(
        default=None,
        converter=_convert_temporal_frame_window,
        validator=_validate_temporal_frame_window,
    )

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

    # The same switch for the radar stream of a joint camera + radar pack, read the same way.
    # Radar has a stronger prior claim to needing it than LiDAR does: the captions describe what
    # the cameras see, and a BEV occupancy grid shares even less of that vocabulary than a range
    # image does. Independent of ``lidar_attends_captions`` so a three-sensor pack can drop text
    # for one stream and keep it for the other.
    radar_attends_captions: bool = True


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

    # What the multiview backend lets the noisy tokens attend to.
    mask: MultiviewAttentionMaskConfig = MultiviewAttentionMaskConfig()

    # Count each visual key once by excluding same-view keys from the cross-instant
    # pass. Only used by maskless decomposed attention; same_view is already exact.
    # False preserves existing recipes.
    deduplicate_cross_view: bool = False
