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
from collections.abc import Mapping, Sequence
from typing import Any, Literal, get_args, overload

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
RigRopeMissingGeometry = Literal["error", "mrope", "no_rotation", "time_only"]
RigRopeChannels = Literal["all", "even_pairs"]

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
#   ``cross_view_temporal_window.bounds`` is set: the two streams do not share a frame
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

# Applied by temporal_window_bounds for Flex's pair predicate and maskless window planning.
# Capture times are float32; include pairs that round just beyond either window boundary.
DECOMPOSED_TEMPORAL_WINDOW_EPS: float = 1e-4

# Signed key-time offsets from the query, in seconds: always explicit (start, end).
TemporalWindow = tuple[float, float]


def validate_temporal_window_bounds(window: TemporalWindow | None) -> TemporalWindow | None:
    """Validate explicit (start, end) bounds, including their YAML list representation."""
    if window is None:
        return None
    if (
        not isinstance(window, Sequence)
        or len(window) != 2
        or not all(
            isinstance(bound, (int, float)) and not isinstance(bound, bool) and math.isfinite(bound) for bound in window
        )
    ):
        raise ValueError("Temporal window requires two finite bounds (start, end)")
    start, end = window
    if start > end:
        raise ValueError("Temporal window requires start <= end")
    return float(start), float(end)


@overload
def temporal_window_bounds(window: TemporalWindow) -> TemporalWindow: ...


@overload
def temporal_window_bounds(window: None) -> None: ...


def temporal_window_bounds(window: TemporalWindow | None) -> TemporalWindow | None:
    """Expand validated bounds for comparisons without changing the stored config."""
    if window is None:
        return None
    start, end = window
    return start - DECOMPOSED_TEMPORAL_WINDOW_EPS, end + DECOMPOSED_TEMPORAL_WINDOW_EPS


# Which attention the multiview generation stream runs as.
#
# * ``"maskless"``: disjoint same-view, cross-view, and gen->und partitions,
#   merged by log-sum-exp, with no mask anywhere. See
#   ``models.mot.multiview_maskless_attention.multiview_maskless_gen_attention``.
# * ``"flex_flash"``: the masked FlexAttention call on FlashAttention-4 (CuTeDSL) kernels.
# * ``"flex_triton"``: the same masked call on FlexAttention's Triton kernels. Available by
#   construction, so it is what everything else falls back to.
# * ``"auto"``: ``"maskless"`` when the config supports it, then ``"flex_flash"`` where
#   FA4 is available, else ``"flex_triton"``.
#
# Maskless cross-view segments always exclude same-view keys, so every allowed key
# is counted once. Without a temporal window it retains midpoint-quantised instant
# assignment; with a window it uses Flex's directed capture-time rule. Mixed-rate
# streams therefore need an explicit window for the two backends to share their geometry.
#
# Availability here is a property of the *config* rather than of the batch. A batch the
# selected folds cannot express raises instead of quietly taking a mask. See
# ``models.mot.multiview_maskless_attention.maskless_unavailable_reason``, which owns
# that verdict because every condition in it is a fact about what the folds express, and
# ``models.mot.multiview_attention.resolve_multiview_backend``, which acts on it.
BackendPreference = Literal["auto", "maskless", "flex_triton", "flex_flash"]

# The preferences of ``BackendPreference`` at runtime, which the annotation itself is not.
BACKEND_PREFERENCES = get_args(BackendPreference)

# The backends ``BackendPreference`` resolves to, i.e. everything but ``"auto"``.
ResolvedBackend = Literal["maskless", "flex_triton", "flex_flash"]

# The subset of ``BackendPreference`` that names a *mask geometry*. ``"maskless"`` is absent
# because it builds no mask and imposes no Flex padding alignment. Callers that only ever build a mask -- the benchmark's
# flex rows, and ``flex_attention.resolve_flex_backend`` -- take this rather than the full
# preference, so ``"maskless"`` is rejected by the type instead of at the call.
FlexGeometryPreference = Literal["auto", "flex_triton", "flex_flash"]

# The preferences of ``FlexGeometryPreference`` at runtime, which the annotation itself is not.
FLEX_GEOMETRY_PREFERENCES = get_args(FlexGeometryPreference)


@attrs.define(slots=False)
class CrossViewTemporalWindowConfig:
    """Cross-view key capture-time bounds, in seconds relative to each query."""

    # Only read under attention_scope="decomposed". Replaces that scope's temporal half --
    # "the query's own frame index" -- with an inclusive (start, end) interval of key capture
    # times relative to the query: start <= key_timestamp - query_timestamp <= end.
    # Configs spell out both bounds: (-0.4, 0.0) is past-only,
    # (-0.2, 0.2) is symmetric with 0.4 s total width, (-0.4, 0.2) is asymmetric.
    # None (the default) keeps the frame-index form, which only agrees across sensors that share one
    # clock; a joint camera + LiDAR pack needs a window instead; see
    # flex_attention.build_multiview_flex_metadata and ._multiview_pair_predicate.
    #
    # Maskless groups queries by capture time and excludes own-view keys from the
    # window pass, counting each key once after merging with the same-view pass.
    # None preserves midpoint-quantised instant groups.
    bounds: TemporalWindow | None = attrs.field(default=None, converter=validate_temporal_window_bounds)

    # Also let every cross-view query read its sample's first frame (capture time 0, the given
    # frame under I2V) on the other views, whatever the window says. It widens the window's key
    # times rather than adding a pass, so a first frame that
    # already sits inside the window is still counted once and own-view keys stay out. Maskless
    # only, and only with a window: the mask has no such rule, and without a window the folds
    # group by instant rather than comparing capture times.
    includes_first_frame: bool = False


@attrs.define(slots=False)
class SameViewTemporalWindowConfig:
    """Same-view sensor/control neighborhood bounds, in seconds relative to each query."""

    # Optional inclusive temporal windows for each same-view generation edge, in seconds.
    # Sensor denotes the generated, non-control side for both camera and LiDAR streams.
    # A centered (-N, N) window uses the frame neighborhood within N seconds on either
    # side and shifts inward at sequence boundaries to preserve its width. A causal
    # (-N, 0) window includes the current frame and frames up to N seconds earlier,
    # clipping at the beginning of the sequence. Bounds are converted using each item's
    # seconds_per_frame when the attention plan is built. None disables temporal
    # windowing for that edge, so every frame in the same view is used. Cross-view
    # same-instant attention is unchanged.
    # A windowed sensor/control edge requires matching seconds_per_frame for that pair;
    # independent view groups may use different clocks when their frame windows agree.
    # Experiments using selective activation checkpointing must also override
    # model.config.activation_checkpointing.save_only_marked_ops=False when enabling
    # any temporal window; marked-only checkpointing is not implemented for this path.
    sensor_to_sensor_bounds: TemporalWindow | None = attrs.field(
        default=None, converter=validate_temporal_window_bounds
    )
    sensor_to_control_bounds: TemporalWindow | None = attrs.field(
        default=None, converter=validate_temporal_window_bounds
    )
    control_to_control_bounds: TemporalWindow | None = attrs.field(
        default=None, converter=validate_temporal_window_bounds
    )
    control_to_sensor_bounds: TemporalWindow | None = attrs.field(
        default=None, converter=validate_temporal_window_bounds
    )


def _load_cross_view_temporal_window(
    value: CrossViewTemporalWindowConfig | Mapping[str, Any],
) -> CrossViewTemporalWindowConfig:
    """Construct the nested config from serialized mappings when needed."""
    return value if isinstance(value, CrossViewTemporalWindowConfig) else CrossViewTemporalWindowConfig(**value)


def _load_same_view_temporal_window(
    value: SameViewTemporalWindowConfig | Mapping[str, Any],
) -> SameViewTemporalWindowConfig:
    """Construct the nested config from serialized mappings when needed."""
    return value if isinstance(value, SameViewTemporalWindowConfig) else SameViewTemporalWindowConfig(**value)


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

    cross_view_temporal_window: CrossViewTemporalWindowConfig = attrs.field(
        factory=CrossViewTemporalWindowConfig, converter=_load_cross_view_temporal_window
    )
    same_view_temporal_window: SameViewTemporalWindowConfig = attrs.field(
        factory=SameViewTemporalWindowConfig, converter=_load_same_view_temporal_window
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

    # The same switch for the radar stream of a joint camera + radar pack, read the same way.
    # Radar has a stronger prior claim to needing it than LiDAR does: the captions describe what
    # the cameras see, and a BEV occupancy grid shares even less of that vocabulary than a range
    # image does. Independent of ``lidar_attends_captions`` so a three-sensor pack can drop text
    # for one stream and keep it for the other.
    radar_attends_captions: bool = True


def normalize_multiview_temporal_windows(mask_config: Any) -> None:
    """Normalize temporal windows in saved DictConfig masks; attrs configs are already validated."""
    if isinstance(mask_config, MultiviewAttentionMaskConfig):
        return

    # Exported checkpoint configs can arrive as plain DictConfig objects, bypassing
    # the attrs converter. Normalize durations before building attention.
    cross_view_window = mask_config.cross_view_temporal_window
    cross_view_window.bounds = validate_temporal_window_bounds(cross_view_window.bounds)
    same_view_window = mask_config.same_view_temporal_window
    for name in (
        "sensor_to_sensor_bounds",
        "sensor_to_control_bounds",
        "control_to_control_bounds",
        "control_to_sensor_bounds",
    ):
        setattr(same_view_window, name, validate_temporal_window_bounds(getattr(same_view_window, name)))


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
    # an earlier one pins its backend. "auto" prefers supported maskless attention,
    # then Flex Flash and Flex Triton. Explicit "flex_flash" and "maskless" both fail
    # the run where they are unavailable rather than falling back.
    backend: BackendPreference = attrs.field(
        default="auto",
        validator=attrs.validators.in_(BACKEND_PREFERENCES),
    )

    # Decomposed-only geometry modes. PRoPE composes camera-relative extrinsics with
    # pretrained mRoPE in the direct camera cross-view pass. RigRoPE instead leaves captions
    # and same-view attention on pretrained mRoPE and rotates raw cross-view Q/K from
    # calibrated rays. Both are parameter-free and fail closed when geometry is incomplete.
    # "prope_cross_view" runs PRoPE's camera-relative extrinsics inside RigRoPE's cross-view
    # pass, so it takes the band, the head split and the missing-geometry fallback: on the
    # geometry heads, every other split-half mRoPE pair carries a 4x4 reference-to-camera
    # transform (Q by P^T, K and V by P^-1, the output by P) and the remaining pairs keep
    # mRoPE, so both stay exactly relative and the heads keep capture time and pixel position.
    # It reads ``rigrope_head_fraction``, ``rigrope_missing_geometry`` (where "no_rotation" and
    # "time_only" both mean an identity transform, since the mRoPE pairs already carry time;
    # "mrope" is rejected), ``rigrope_moment_normalization`` with ``rigrope_moment_scale_m`` and
    # ``rigrope_moment_scale_floor_m`` as the translation unit, and ``rigrope_pose_world_frame``
    # for the frame poses are anchored in. It uses intrinsics only under ``prope_intrinsics``.
    geometry_position_encoding: Literal["baseline", "prope", "rigrope_cross_view", "prope_cross_view"] = attrs.field(
        default="baseline",
        validator=attrs.validators.in_(("baseline", "prope", "rigrope_cross_view", "prope_cross_view")),
    )

    # Full PRoPE for "prope" and "prope_cross_view": each camera's transform becomes the
    # projection P = lift(K) @ reference-to-camera, so a query sees K_i T_i T_j^-1 K_j^-1 rather
    # than the extrinsics alone. K is the calibration's linear fx, fy, cx, cy carried through
    # ``image_from_calibration`` into the encoded frame and normalized by its width and height
    # (PRoPE's convention), for every lens model: distortion and F-theta polynomials are
    # ignored, so wide lenses get only their central pinhole approximation. Off is the
    # extrinsics-only (GTA-style) transform. Ignored by the other geometry modes.
    prope_intrinsics: bool = False

    # Unit ray directions and Pluecker moments in units of this many metres; the default is
    # 25 m. The moments' slowest frequency turns by pi per unit, so a
    # scale far above the rig's extent leaves them nearly unrotated. No alpha/ramp is used by
    # this mode, and no time coordinate unless ``rigrope_include_time``. Unused under
    # ``rigrope_moment_normalization="per_sample_rms"``.
    rigrope_moment_scale_m: float = attrs.field(
        default=25.0,
        validator=lambda _instance, _attribute, value: (
            None
            if math.isfinite(value) and value > 0.0
            else (_ for _ in ()).throw(ValueError("rigrope_moment_scale_m must be finite and positive"))
        ),
    )

    # How the moment unit is chosen.
    # - "fixed": ``rigrope_moment_scale_m`` for every sample.
    # - "per_sample_rms": each sample's RMS camera distance from the moment origin, over all of
    #   its views and encoded frames, floored at ``rigrope_moment_scale_floor_m``. A 1 m car rig
    #   and a 10 cm robot head then span the same angles, at the cost of the moments no longer
    #   being metric across samples.
    rigrope_moment_normalization: Literal["fixed", "per_sample_rms"] = attrs.field(
        default="fixed", validator=attrs.validators.in_(("fixed", "per_sample_rms"))
    )

    # Smallest per-sample moment unit, in metres, so a rig whose cameras nearly coincide (a
    # stereo pair, or one camera that barely moves) does not blow its moments up to noise.
    rigrope_moment_scale_floor_m: float = attrs.field(
        default=0.05,
        validator=lambda _instance, _attribute, value: (
            None
            if math.isfinite(value) and value > 0.0
            else (_ for _ in ()).throw(ValueError("rigrope_moment_scale_floor_m must be finite and positive"))
        ),
    )

    # Express every sample's descriptors in a sample-local frame. RigRoPE rotates each axis of
    # the directions and moments separately, so it is not invariant to a global rotation, and a
    # pose-world sample's scene frame is otherwise arbitrary. On, a pose-world sample is rotated
    # into its first view's camera at the first encoded frame, with the OpenCV axes relabelled
    # forward-left-up -- the axes of the AV ego rig, in which that camera would be a front
    # camera. A static rig keeps its (ego) axes. Both put the origin at the cameras' centroid
    # at the first encoded frame, which pose-world samples already use.
    rigrope_canonical_frame: bool = False

    # Which frame a pose-world sample's descriptors are read in, at each encoded frame.
    # - "scene": one frame for the whole sample, the scene's or (``rigrope_canonical_frame``)
    #   the first encoded frame's reference camera. A rig moving through the scene then carries
    #   its ego-motion into the descriptors: moments are not translation-invariant, and a rig
    #   that moves by e shifts two views' relative moment by e x (d_a - d_b); under the per-sample
    #   RMS unit, a half-metre car rig that drives 40 m also gets a unit of about 23 m.
    # - "rig_per_frame": every encoded frame in its own first-view camera, relabelled
    #   forward-left-up, about its own camera centroid. A rigid rig then reads the same at every
    #   frame, as a static rig does, and the RMS unit is the rig's spread; cameras that move
    #   against each other (a wrist camera against an exterior one) keep that motion.
    #   ``rigrope_canonical_frame`` is moot for pose-world samples here.
    # Static rigs have no per-frame poses and are unaffected.
    rigrope_pose_world_frame: Literal["scene", "rig_per_frame"] = attrs.field(
        default="scene", validator=attrs.validators.in_(("scene", "rig_per_frame"))
    )

    # Whether RigRoPE descriptors carry capture time in seconds from the sample's first frame.
    # Off (the default) leaves the time frequency slots at zero, which costs nothing
    # while every key shares the query's instant. A cross-view band keys other instants too, and
    # a static rig's rays are the same at every frame, so without time the RigRoPE heads cannot
    # tell a key five frames away from one at the query's own instant.
    rigrope_include_time: bool = False

    # Seconds per unit of the RigRoPE time channel. The channel's slowest frequency turns by pi
    # per unit, so two tokens up to this many seconds apart stay within half a turn of it and
    # their offset is unambiguous; farther apart it wraps, and only the faster frequencies,
    # which never line up exactly, keep them apart. Raising it trades that range for
    # resolution: at 16 s one 4/30 s latent step turns the fastest frequency by ~0.2 rad.
    rigrope_time_scale_s: float = attrs.field(
        default=1.0,
        validator=lambda _instance, _attribute, value: (
            None
            if math.isfinite(value) and value > 0.0
            else (_ for _ in ()).throw(ValueError("rigrope_time_scale_s must be finite and positive"))
        ),
    )

    # What RigRoPE's cross-view pass does for a multiview sample without usable calibrated
    # geometry, so a recipe can train on rows with and without camera sidecars. Each
    # (sample, frame) group of that pass is one sample's -- or, under a cross-view band, one
    # sample's run of instants -- so no softmax mixes encodings.
    # - "error": the sample fails the batch.
    # - "mrope": every head keeps the pretrained mRoPE. The RigRoPE heads then see mRoPE on
    #   some samples and RigRoPE on others, with nothing telling the model which.
    # - "no_rotation": the RigRoPE heads apply no rotation (content-only cross-view attention,
    #   a "geometry unknown" state) while the other heads keep mRoPE, so each head sees one
    #   encoding family. Meant for ``rigrope_head_fraction < 1``; at 1.0 a sample without
    #   geometry loses cross-view positions entirely.
    # - "time_only": like "no_rotation", but the RigRoPE heads rotate by the descriptor's time
    #   channel alone (rays and moments zero). Identical to "no_rotation" when every key shares
    #   the query's instant; under ``maskless_cross_view_band_radius`` or
    #   ``maskless_cross_view_include_frame_zero`` it keeps the relative capture time between
    #   frames, in the same seconds a posed sample's descriptor carries.
    rigrope_missing_geometry: RigRopeMissingGeometry = attrs.field(
        default="error", validator=attrs.validators.in_(get_args(RigRopeMissingGeometry))
    )

    # Share of the KV head groups (with the query heads they serve) whose cross-view pass uses
    # RigRoPE; the remaining groups keep pretrained mRoPE on every sample. 1.0 is every head.
    # The network rejects a share that does not name a whole number of groups.
    rigrope_head_fraction: float = attrs.field(
        default=1.0,
        validator=lambda _instance, _attribute, value: (
            None if 0.0 < value <= 1.0 else (_ for _ in ()).throw(ValueError("rigrope_head_fraction must be in (0, 1]"))
        ),
    )
    # Which channels of a RigRoPE head its cross-view pass rotates by geometry.
    # - "all": every channel, so the head keeps no mRoPE there.
    # - "even_pairs": the even split-half pairs (channel c with c mod D/2 even), as
    #   ``prope_cross_view`` does, at half the frequencies per descriptor component; the odd
    #   pairs keep mRoPE, so the head keeps position offsets even without geometry. The two
    #   act on disjoint pairs, so each stays exactly relative. Needs full-head mRoPE.
    rigrope_channels: RigRopeChannels = attrs.field(
        default="all", validator=attrs.validators.in_(get_args(RigRopeChannels))
    )

    # RigRoPE frequencies are ``pi * 2^e`` for e evenly spaced over [min, max] in every
    # descriptor channel; the defaults give [pi, 8 pi]. Two views seeing one point at depth z
    # from baseline b differ in ray direction by about b / z, so the fastest frequency turns a
    # near-field pair (b / z ~ 0.3) by several radians and its phase carries no correspondence.
    # A unit-direction component differs by up to 2 between views, which the slowest frequency
    # turns by 2 pi * 2^min: at min = 0 cameras facing opposite ways read alike there, and
    # min <= -1 keeps that span within one half turn.
    rigrope_min_freq_exponent: float = attrs.field(
        default=0.0,
        validator=lambda _instance, _attribute, value: (
            None
            if math.isfinite(value)
            else (_ for _ in ()).throw(ValueError("rigrope_min_freq_exponent must be finite"))
        ),
    )
    rigrope_max_freq_exponent: float = attrs.field(
        default=3.0,
        validator=lambda instance, _attribute, value: (
            None
            if math.isfinite(value) and value >= instance.rigrope_min_freq_exponent
            else (_ for _ in ()).throw(
                ValueError("rigrope_max_freq_exponent must be finite and at least rigrope_min_freq_exponent")
            )
        ),
    )

    # Maskless expansion of the cross-view fold. Zero with frame zero disabled preserves the
    # ordinary aligned-frame decomposition. A positive radius keys every query instant against
    # that symmetric range of instants, and ``include_frame_zero`` additionally includes the
    # sample's first instant in every key run. Only "other_views" band keys are supported for
    # expansion: they keep the cross-view and same-view passes disjoint, counting each key once.
    maskless_cross_view_band_radius: int = attrs.field(default=0, validator=attrs.validators.ge(0))
    maskless_cross_view_include_frame_zero: bool = False
    # Which views the band keys. Without an expansion this setting is ignored, and the ordinary
    # decomposition remains exact regardless of a value saved in an older checkpoint.
    # - "all_views" and "own_view": accepted for older configs without expansion, but rejected
    #   when a radius or frame-zero expansion is requested, since they would overlap the
    #   same-view pass. Use "other_views" for those expansions.
    # - "other_views": every other view at every band instant, the query's own instant included,
    #   and never the query's own view, which the same-view pass keys at every frame. Each key
    #   then counts once, so this is exact attention over its key set, at radius zero too, and
    #   single-view samples still never enter the cross pass. Gathering the other views once per
    #   query view would repeat each key ten times on an 11-view clip, so it runs one pass per
    #   level of a binary split of the views instead: ceil(log2(views)) times the keys.
    maskless_cross_view_band_keys: Literal["all_views", "own_view", "other_views"] = attrs.field(
        default="other_views", validator=attrs.validators.in_(("all_views", "own_view", "other_views"))
    )

    # What the multiview backend lets the noisy tokens attend to.
    mask: MultiviewAttentionMaskConfig = attrs.field(factory=MultiviewAttentionMaskConfig)


# Same-view windows of masks saved before the temporal-window configs, by their current names.
_SAVED_SAME_VIEW_FRAME_WINDOWS: dict[str, str] = {
    "sensor_to_sensor_window": "sensor_to_sensor_bounds",
    "sensor_to_control_window": "sensor_to_control_bounds",
    "control_to_control_window": "control_to_control_bounds",
    "control_to_sensor_window": "control_to_sensor_bounds",
}


def _drop(config: Any, name: str) -> None:
    """Remove a saved key if present."""
    if hasattr(config, name):
        delattr(config, name)


def _restore_missing_fields(config: Any, default: Any) -> None:
    """Add the fields a config saved before they existed lacks, as plain values."""
    for field in attrs.fields(type(default)):
        if not hasattr(config, field.name):
            value = getattr(default, field.name)
            setattr(config, field.name, attrs.asdict(value) if attrs.has(type(value)) else value)


def _saved_cross_view_window(mask: Any) -> dict[str, Any]:
    """The cross-view window equivalent to a mask's earlier ``decomposed_temporal_window_*`` keys."""
    window = getattr(mask, "decomposed_temporal_window_seconds", None)
    if isinstance(window, (int, float)) and not isinstance(window, bool):
        # The old scalar N admitted keys at or before the query within N seconds.
        if not math.isfinite(window) or window < 0:
            raise ValueError("Saved decomposed_temporal_window_seconds must be finite and non-negative")
        window = (-float(window), 0.0)
    return {
        "bounds": validate_temporal_window_bounds(window),
        "includes_first_frame": bool(getattr(mask, "decomposed_temporal_window_includes_first_frame", False)),
    }


def upgrade_saved_multiview_attention(config: Any) -> None:
    """Bring a saved ``multiview_attention`` config to the current schema, in place.

    Exported checkpoint configs arrive as plain DictConfig objects, bypassing the attrs
    converters, and can predate fields or use the earlier temporal-window keys. attrs configs
    are already current. A config that cannot be upgraded is left unchanged.
    """
    if config is None or isinstance(config, MultiviewAttentionConfig):
        return
    mask = getattr(config, "mask", None)
    saved_mask = mask is not None and not isinstance(mask, MultiviewAttentionMaskConfig)
    # Convert and check before changing anything, so a failed upgrade keeps the saved values.
    cross_view = None
    same_view = None
    if saved_mask and not hasattr(mask, "cross_view_temporal_window"):
        cross_view = _saved_cross_view_window(mask)
    if saved_mask and not hasattr(mask, "same_view_temporal_window"):
        # Saved windows counted frames, which cannot become seconds without the clip's frame rate.
        configured = sorted(name for name in _SAVED_SAME_VIEW_FRAME_WINDOWS if getattr(mask, name, None) is not None)
        if configured:
            raise ValueError(
                f"Saved multiview mask sets frame-count windows {configured}. Set the matching "
                "model.config.multiview_attention.mask.same_view_temporal_window.*_bounds in seconds instead."
            )
        same_view = dict.fromkeys(_SAVED_SAME_VIEW_FRAME_WINDOWS.values())

    # Maskless cross-view segments now always exclude same-view keys, so nothing reads this.
    _drop(config, "deduplicate_cross_view")
    _restore_missing_fields(config, MultiviewAttentionConfig())
    if not saved_mask:
        return
    for name in ("decomposed_temporal_window_seconds", "decomposed_temporal_window_includes_first_frame"):
        _drop(mask, name)
    for name in _SAVED_SAME_VIEW_FRAME_WINDOWS:
        _drop(mask, name)
    if cross_view is not None:
        mask.cross_view_temporal_window = cross_view
    if same_view is not None:
        mask.same_view_temporal_window = same_view
    _restore_missing_fields(mask, MultiviewAttentionMaskConfig())
