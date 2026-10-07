# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Shared activation checkpointing schema for the MoT and VLM paths.

``ActivationCheckpointingConfig`` is referenced from both
``OmniMoTModelConfig.activation_checkpointing`` (MoT) and
``PolicyConfig.activation_checkpointing`` (VLM, in
vfm/configs/base/vlm/defaults/training.py). Both apply AC with
``ptd_checkpoint_wrapper``; CPU offload settings apply to the MoT path — see
``parallelize_unified_mot.apply_ac`` and ``parallelize_vlm.apply_ac``.

Historically the VLM path used HF's binary ``gradient_checkpointing_enable``
and therefore honoured only ``mode``, silently degrading ``"selective"`` to
no checkpointing and ignoring the SAC fields. That is no longer the case.
"""

import attrs

# The forward attention op of every backend the frontend can dispatch, by the name
# the checkpoint policy matches against.
#
# * natten: fmha_forward
# * flash2: _flash_attn_varlen_forward (varlen), _flash_attn_forward (dense)
# * flash3: _flash_attn_forward (dense, varlen)
# * flash4: fmha_fwd (imaginaire_flash4 custom op, dense and varlen)
#
# cuDNN is deliberately absent since it does not support varlen attention.
ATTENTION_FORWARD_OPS_REGEX = [
    "fmha_forward",
    "_flash_attn_varlen_forward",
    "_flash_attn_forward",
    "fmha_fwd",
]


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
    #
    # Defaults to attention on whichever backend the frontend picks, which is what
    # every config asking for selective AC wants and is not what a name like "fmha"
    # delivers: that covers NATTEN and our Flash4 wrapper, but misses flash2/flash3.
    # On sm100, NATTEN remains preferred for compatible varlen calls because cuDNN
    # rejects varlen and Flash4 follows NATTEN, so "fmha" matches; on sm90 flash3 takes it,
    # leaving nothing in the region named "fmha", so selective AC there kept nothing
    # and silently recomputed every attention it was configured to save.
    #
    # Copied because attrs hands this list to the instance, and a config mutating it
    # would edit the module constant for every other config in the process.
    save_ops_regex: list[str] = attrs.field(
        factory=lambda: list(ATTENTION_FORWARD_OPS_REGEX),
    )

    # Narrow ``save_ops_regex`` to the call sites that asked to be kept.
    #
    # The regex matches dispatched op names, which cannot separate calls running
    # the same kernel. The decomposed multiview attention runs four FMHA calls per
    # layer worth very different amounts to keep -- the same-view fold is ~96% of
    # forward attention time and ~94% of backward, the rest are cheap to recompute.
    #
    # With this on, a matching op is kept only where the model marked it with
    # ``activation_marks.mark_next_activation``. The regex still says which ops are
    # eligible; the mark says which of them are worth it. Off by default, which is
    # the behaviour every existing config already has.
    save_only_marked_ops: bool = False

    # Stash and restore RNG state across recompute boundaries. Required for
    # deterministic output vs. non-checkpointed passes; slower otherwise.
    preserve_rng_state: bool = True

    # Determinism check forwarded to ``ptd_checkpoint_wrapper`` /
    # ``torch.utils.checkpoint.checkpoint``.
    determinism_check: str = "default"

    # Offload large AC checkpoint inputs to CPU memory during forward
    # and fetch them back synchronously during backward recompute. Other denoise
    # autograd saves stay on their original device. Reduces GPU peak by the
    # full-AC checkpoint ramp at the cost of PCIe round-trips.
    # Only meaningful when ``mode`` is ``"full"`` or ``"selective"``.
    offload_to_cpu: bool = False

    # Keep an activation on GPU if its CPU copy would leave less than this
    # fraction of any finite job-cgroup RAM limit free. Warn and keep it on GPU
    # when no finite cgroup limit or live usage can be measured.
    min_cgroup_memory_free_fraction: float = attrs.field(
        default=0.10,
        validator=[attrs.validators.ge(0.0), attrs.validators.lt(1.0)],
    )
