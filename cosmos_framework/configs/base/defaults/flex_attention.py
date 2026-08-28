# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Whether the multiview generation attention runs on FlexAttention, and under what mask.

Its own module rather than a section of ``model_config`` because the mask code reads these
values too, and importing ``model_config`` from there is not possible: it pulls in the reasoner
defaults, which reach the MoT attention modules, which reach ``flex_attention`` itself. Keeping
the values here leaves one definition with nothing but ``attrs`` behind it, which either side
can import.
"""

from typing import Literal, get_args

import attrs

# Which noisy tokens a noisy token attends to, among those of its own sample.
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
# * ``"same_view_or_frame"``: Each camera attends to its own noisy tokens plus the same frame in
#   every other camera. Cost is V*(FS)^2 + F*(VS)^2. Not allowed on a joint camera + LiDAR pack:
#   the two streams do not share a frame index.
AttentionScope = Literal["all_views", "same_view", "same_view_or_frame"]

# The scopes of ``AttentionScope`` at runtime, which the annotation itself is not.
ATTENTION_SCOPES = get_args(AttentionScope)

# Which kernels the multiview mask runs on. See flex_attention.resolve_flex_backend for what
# each one resolves to and the comment on ``FlexAttentionConfig.backend`` for when to pin one.
FlexBackendPreference = Literal["auto", "triton", "flash"]

# The preferences of ``FlexBackendPreference`` at runtime, which the annotation itself is not.
FLEX_BACKEND_PREFERENCES = get_args(FlexBackendPreference)


@attrs.define(slots=False)
class FlexAttentionMaskConfig:
    """What the multiview FlexAttention mask lets the generated tokens see.

    Only read when ``flex_attention.enabled`` is on; the dense attention path has no notion of a
    view.
    """

    # Which same-kind (RGB) tokens of its sample a token attends to: conditioning tokens
    # reach conditioning tokens, and noisy tokens reach both noisy and conditioning tokens
    # alike. Cross-view attention is what lets the rig agree with itself, so the full square
    # is the default; the narrower scopes buy attention that grows with the rig rather than
    # with its square, per the comment above. Never widens a WSM (World Scenario Map) control
    # token's reach, which is always its own view -- see
    # flex_attention.build_multiview_flex_metadata's ``is_control_per_item``, which the
    # network derives per generation stream, and which a batch without a control stream
    # leaves empty.
    attention_scope: AttentionScope = attrs.field(
        default="all_views",
        validator=attrs.validators.in_(ATTENTION_SCOPES),
    )


@attrs.define(slots=False)
class FlexAttentionConfig:
    """Whether the within-sample GEN attention runs as one masked FlexAttention call."""

    # Replace the dense within-sample GEN attention with the multiview FlexAttention mask, which
    # keeps each conditioning stream to its own (frame, view) and lets noisy tokens read the
    # conditioning of their own (frame, view) as well as other noisy tokens from the same sample.
    # GEN tokens still see the whole caption, so the mask spans the fused [UND | GEN] key stream
    # and one kernel covers what the three-way decomposition split across two and merged.
    # Requires joint_attn_implementation="two_way", a vision-only generation batch, and a dataset
    # with enable_per_camera_vae_encoding so the per-camera view counts are available.
    # Mutually exclusive with NATTEN.
    enabled: bool = False

    # Which kernels the mask above runs on. "auto" takes FlashAttention-4 (CuTeDSL) wherever the
    # flash-attn-4 package is installed and the GPU supports it, and FlexAttention's Triton kernels
    # everywhere else. That makes the choice a property of the host rather than of the config: the
    # same experiment gets different kernels, a different padded sequence length, and different
    # rounding depending on the image it runs in, so a run that has to stay bit-comparable with an
    # earlier one wants "triton" instead. "flash" fails the run where FA4 is unavailable, which is
    # only useful for a benchmark that is meaningless without it. See
    # flex_attention.resolve_flex_backend.
    backend: FlexBackendPreference = attrs.field(
        default="auto",
        validator=attrs.validators.in_(FLEX_BACKEND_PREFERENCES),
    )

    # What that mask lets the noisy tokens attend to.
    mask: FlexAttentionMaskConfig = FlexAttentionMaskConfig()
