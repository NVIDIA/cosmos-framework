# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Backend-neutral teacher-forcing replay attention configuration."""

from typing import Literal, get_args

import attrs

from cosmos_framework.configs.base.defaults.flex_attention import (
    ATTENTION_SCOPES,
    AttentionScope,
)

TeacherForcingKVImplementation = Literal["multiview_flex_kv", "singleview_threeway_kv"]
TEACHER_FORCING_KV_IMPLEMENTATIONS = get_args(TeacherForcingKVImplementation)

TeacherForcingControlVisibility = Literal["global", "causal", "current"]
TEACHER_FORCING_CONTROL_VISIBILITIES = get_args(TeacherForcingControlVisibility)

TeacherForcingCleanPassCausality = Literal["frame", "chunk"]
TEACHER_FORCING_CLEAN_PASS_CAUSALITIES = get_args(TeacherForcingCleanPassCausality)


@attrs.define(slots=False)
class TeacherForcingReplayPolicyConfig:
    """Connectivity shared by every teacher-forcing K/V implementation."""

    # Which same-view control frames control, target-condition, and generated-target queries may read.
    control_visibility: TeacherForcingControlVisibility = attrs.field(
        default="global",
        validator=attrs.validators.in_(TEACHER_FORCING_CONTROL_VISIBILITIES),
    )

    # Let control queries additionally read target-conditioning RGB and cached/generated
    # clean RGB from strictly earlier replay chunks, intersected with the multiview scope.
    controls_read_strict_past_clean_rgb: bool = attrs.field(
        default=False,
        validator=attrs.validators.instance_of(bool),
    )

    # Whether clean-pass targets use frame-level or replay-chunk-level causal ordering.
    clean_pass_causality: TeacherForcingCleanPassCausality = attrs.field(
        default="frame",
        validator=attrs.validators.in_(TEACHER_FORCING_CLEAN_PASS_CAUSALITIES),
    )

    # The view/time footprint shared by replay target and control-to-clean-RGB edges.
    multiview_attention_scope: AttentionScope = attrs.field(
        default="all_views",
        validator=attrs.validators.in_(ATTENTION_SCOPES),
    )

    # Under the decomposed scope, replace equal frame indexes with a bounded
    # non-negative capture-time gap so sensors with different rates can align.
    decomposed_temporal_window_seconds: float | None = attrs.field(
        default=None,
        validator=attrs.validators.optional(attrs.validators.ge(0)),
    )

    def __attrs_post_init__(self) -> None:
        """Reject replay graphs that can relay future RGB through global control K/V."""
        self.validate()

    def validate(self) -> None:
        """Recheck cross-field invariants after config resolution or mutation."""
        if self.control_visibility == "global" and self.controls_read_strict_past_clean_rgb:
            raise ValueError(
                "controls_read_strict_past_clean_rgb requires causal or current control_visibility; "
                "global control visibility can relay future clean RGB through control states."
            )
