# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Shared activation checkpointing schema for the MoT and VLM paths.

``ActivationCheckpointingConfig`` is referenced from both
``OmniMoTModelConfig.activation_checkpointing`` (MoT) and
``PolicyConfig.activation_checkpointing`` (VLM, in
vfm/configs/base/vlm/defaults/training.py), but the two read sites
consume different subsets of the schema:

- MoT (vfm/models/mot/parallelize_unified_mot.py): every field is
  consumed.
- VLM (vfm/models/vlm_model.py): only ``mode`` is consumed, and only
  ``"full"`` actually enables checkpointing. The HF backbone exposes a
  single binary ``gradient_checkpointing_enable`` API and has no
  per-op SAC support, so ``"selective"`` is accepted at the type level
  but degrades to no checkpointing on the VLM path; the SAC-specific
  fields (``save_ops_regex``, ``preserve_rng_state``,
  ``determinism_check``) are ignored entirely.
"""

import attrs


@attrs.define(slots=False)
class ActivationCheckpointingConfig:
    """Activation checkpointing (AC) policy shared by MoT and VLM training.

    Mirrors the torchtitan SAC design: a single ``mode`` knob switches between
    full-block recompute, and per-op selective AC. The remaining fields are
    knobs for the per-op selective policy or the underlying
    ``torch.utils.checkpoint`` plumbing.

    Read sites:

    - MoT path consumes every field — see
      cosmos_framework/model/generator/mot/parallelize_unified_mot.py.
    - VLM path consumes only ``mode`` (and only ``"full"`` enables
      checkpointing) — see cosmos_framework/model/generator/vlm_model.py.
    """

    # AC mode:
    #   - "selective":     per-op SAC. Save expensive matmuls/attention
    #                      ops, recompute the rest. MoT only — on the VLM
    #                      path this degrades to no checkpointing because
    #                      the HF backbone has no per-op SAC support.
    #   - "full":          checkpoint each whole transformer block.
    #   - "none":          no activation checkpointing.
    mode: str = attrs.field(
        default="full",
        validator=attrs.validators.in_({"selective", "full", "none"}),
    )

    # Regex patterns for ops to save when using selective AC. Ignored if
    # mode is "full" or "none". MoT only — unused on the VLM path.
    save_ops_regex: list[str] = attrs.field(
        factory=lambda: ["fmha"],
    )

    # Stash and restore RNG state across recompute boundaries. Required for
    # deterministic output vs. non-checkpointed passes; slower otherwise.
    # MoT only — unused on the VLM path.
    preserve_rng_state: bool = True

    # Determinism check forwarded to ``ptd_checkpoint_wrapper`` /
    # ``torch.utils.checkpoint.checkpoint``. MoT only — unused on the
    # VLM path.
    determinism_check: str = "default"
