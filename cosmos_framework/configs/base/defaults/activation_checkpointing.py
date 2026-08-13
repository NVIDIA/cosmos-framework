# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Shared activation checkpointing schema for the MoT and VLM paths.

``ActivationCheckpointingConfig`` is referenced from both
``OmniMoTModelConfig.activation_checkpointing`` (MoT) and
``PolicyConfig.activation_checkpointing`` (VLM, in
vfm/configs/base/vlm/defaults/training.py). Both read sites consume every
field, because both apply AC with ``ptd_checkpoint_wrapper`` — see
``parallelize_unified_mot.apply_ac`` and ``parallelize_vlm.apply_ac``.

Historically the VLM path used HF's binary ``gradient_checkpointing_enable``
and therefore honoured only ``mode``, silently degrading ``"selective"`` to
no checkpointing and ignoring the SAC fields. That is no longer the case.
"""

import attrs


@attrs.define(slots=False)
class ActivationCheckpointingConfig:
    """Activation checkpointing (AC) policy shared by MoT and VLM training.

    Mirrors the torchtitan SAC design: a single ``mode`` knob switches between
    full-block recompute, and per-op selective AC. The remaining fields are
    knobs for the per-op selective policy or the underlying
    ``torch.utils.checkpoint`` plumbing.

    Read sites, both consuming every field:

    - MoT path — cosmos_framework/model/generator/mot/parallelize_unified_mot.py.
    - VLM path — cosmos_framework/model/generator/parallelize_vlm.py.
    """

    # AC mode:
    #   - "selective":     per-op SAC. Save expensive matmuls/attention
    #                      ops, recompute the rest.
    #   - "full":          checkpoint each whole transformer block.
    #   - "none":          no activation checkpointing.
    mode: str = attrs.field(
        default="full",
        validator=attrs.validators.in_({"selective", "full", "none"}),
    )

    # Regex patterns for ops to save when using selective AC. Ignored if
    # mode is "full" or "none".
    save_ops_regex: list[str] = attrs.field(
        factory=lambda: ["fmha"],
    )

    # Stash and restore RNG state across recompute boundaries. Required for
    # deterministic output vs. non-checkpointed passes; slower otherwise.
    preserve_rng_state: bool = True

    # Determinism check forwarded to ``ptd_checkpoint_wrapper`` /
    # ``torch.utils.checkpoint.checkpoint``.
    determinism_check: str = "default"
