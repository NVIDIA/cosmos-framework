# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""State-free helpers for LongLive-style streaming long tuning.

The implementation follows LongLive v1.0's streaming contract: one optimizer
iteration extends a persistent AR rollout, teacher supervision is restricted to
a short suffix window, and prior generated frames are detached. Every teacher
window contains one re-encoded clean boundary frame followed by complete causal
chunks. SLT uses the same ``[1, C, C, ...]`` partition as Cosmos3 teacher
forcing and inference: latent frame 0 is a singleton, followed by complete
``C``-frame chunks.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class StreamingLongTuningSpec:
    """Validated temporal and attention settings for one streaming sequence."""

    max_sequence_frames: int
    supervision_window_frames: int
    new_frames_per_step: int
    causal_chunk_size: int
    attention_window_frames: int
    attention_sink_frames: int
    allow_partial_final_step: bool = False

    def __post_init__(self) -> None:
        positive_fields = {
            "max_sequence_frames": self.max_sequence_frames,
            "supervision_window_frames": self.supervision_window_frames,
            "new_frames_per_step": self.new_frames_per_step,
            "causal_chunk_size": self.causal_chunk_size,
            "attention_window_frames": self.attention_window_frames,
        }
        for name, value in positive_fields.items():
            if value < 1:
                raise ValueError(f"{name} must be >= 1, got {value}")
        if self.attention_sink_frames < 0:
            raise ValueError(f"attention_sink_frames must be >= 0, got {self.attention_sink_frames}")
        if self.supervision_window_frames > self.max_sequence_frames:
            raise ValueError(
                "supervision_window_frames must not exceed max_sequence_frames, "
                f"got {self.supervision_window_frames}>{self.max_sequence_frames}"
            )
        if self.new_frames_per_step + 1 != self.supervision_window_frames:
            raise ValueError(
                "supervision_window_frames must equal one clean boundary frame plus "
                "new_frames_per_step, got "
                f"{self.supervision_window_frames}!={self.new_frames_per_step}+1"
            )
        if self.new_frames_per_step % self.causal_chunk_size != 0:
            raise ValueError(
                "new_frames_per_step must be divisible by causal_chunk_size, "
                f"got {self.new_frames_per_step}%{self.causal_chunk_size}"
            )
        if (self.supervision_window_frames - 1) % self.causal_chunk_size != 0:
            raise ValueError(
                "The supervised suffix must be divisible by causal_chunk_size, got "
                f"({self.supervision_window_frames}-1)%{self.causal_chunk_size}"
            )
        if (self.max_sequence_frames - 1) % self.causal_chunk_size != 0:
            raise ValueError(
                "max_sequence_frames must satisfy (max_sequence_frames - 1) % causal_chunk_size == 0, "
                f"got ({self.max_sequence_frames} - 1)%{self.causal_chunk_size}"
            )
        if (
            not self.allow_partial_final_step
            and (self.max_sequence_frames - self.supervision_window_frames) % self.new_frames_per_step != 0
        ):
            raise ValueError(
                "max_sequence_frames must contain one full first window followed by complete "
                "new-frame steps unless allow_partial_final_step is enabled, got "
                f"({self.max_sequence_frames}-{self.supervision_window_frames})%"
                f"{self.new_frames_per_step}"
            )
        if self.first_rollout_frames < self.new_frames_per_step:
            raise ValueError(
                "The singleton-prefixed first rollout must be at least one regular rollout, "
                f"got {self.first_rollout_frames}<{self.new_frames_per_step}"
            )
        if self.first_rollout_frames > self.max_sequence_frames:
            raise ValueError(
                "The singleton-prefixed first rollout must not exceed max_sequence_frames, "
                f"got {self.first_rollout_frames}>{self.max_sequence_frames}"
            )
        if self.attention_window_frames <= self.causal_chunk_size:
            raise ValueError(
                "attention_window_frames must exceed causal_chunk_size so generated chunks have history, "
                f"got {self.attention_window_frames}<={self.causal_chunk_size}"
            )
        if self.attention_sink_frames >= self.kv_cache_capacity:
            raise ValueError(
                "attention_sink_frames must be smaller than the derived KV-cache capacity, "
                f"got {self.attention_sink_frames}>={self.kv_cache_capacity}"
            )

    @property
    def first_rollout_frames(self) -> int:
        """Generate the singleton and supervised suffix in the first step."""
        return self.supervision_window_frames

    @property
    def kv_cache_capacity(self) -> int:
        """Ring slots needed for ``attention_window_frames`` including the live chunk.

        ``GenKVCache`` stores history only and reserves one physical slot for the
        current write. A live ``C``-frame chunk contributes its K/V directly, so
        retaining ``W-C`` historical frames requires a capacity of ``W-C+1``.
        """
        return self.attention_window_frames - self.causal_chunk_size + 1

    @property
    def local_attention_frames(self) -> int:
        """Non-sink portion of the effective attention window."""
        return self.attention_window_frames - self.attention_sink_frames

    def plan_step(self, current_frame: int, *, initial_condition_frames: int = 0) -> "StreamingStepPlan":
        """Plan the next rollout and local supervision window.

        ``initial_condition_frames=1`` starts camera forward dynamics after an
        already-seeded clean frame; the default preserves prompt-only behavior.
        """
        if current_frame < 0 or current_frame > self.max_sequence_frames:
            raise ValueError(f"current_frame must be in [0, {self.max_sequence_frames}], got {current_frame}")
        if initial_condition_frames not in (0, 1):
            raise ValueError(f"initial_condition_frames must be 0 or 1, got {initial_condition_frames}")
        if current_frame < initial_condition_frames:
            raise ValueError(
                f"current_frame must be >= initial_condition_frames, got {current_frame} and {initial_condition_frames}"
            )
        if current_frame == self.max_sequence_frames:
            raise ValueError("The streaming sequence is already complete")
        if current_frame > 0 and (current_frame - 1) % self.causal_chunk_size != 0:
            raise ValueError(
                "Continuation must start on a [1,C,C,...] causal chunk boundary, "
                f"got current_frame={current_frame}, causal_chunk_size={self.causal_chunk_size}"
            )

        requested_frames = self.first_rollout_frames if current_frame == 0 else self.new_frames_per_step
        rollout_frames = min(requested_frames, self.max_sequence_frames - current_frame)
        if current_frame > 0 and rollout_frames % self.causal_chunk_size != 0:
            raise ValueError(
                "Final continuation is not chunk aligned; choose a chunk-aligned max_sequence_frames, "
                f"got rollout_frames={rollout_frames}"
            )
        if rollout_frames < requested_frames and not self.allow_partial_final_step:
            raise ValueError(
                "Final continuation is shorter than new_frames_per_step; enable allow_partial_final_step "
                f"or choose an aligned sequence length, got rollout_frames={rollout_frames}"
            )

        end_frame = current_frame + rollout_frames
        window_frames = rollout_frames if current_frame == 0 else rollout_frames + 1
        window_start = 0 if current_frame == 0 else current_frame - 1
        overlap_frames = max(0, current_frame - window_start)
        conditioned_frames = 1
        supervised_frames = window_frames - conditioned_frames
        expected_supervised_frames = rollout_frames - conditioned_frames if current_frame == 0 else rollout_frames
        if supervised_frames != expected_supervised_frames:
            raise RuntimeError(
                "Streaming supervision accounting mismatch: "
                f"supervised_frames={supervised_frames}, expected={expected_supervised_frames}"
            )

        return StreamingStepPlan(
            rollout_start=current_frame,
            rollout_end=end_frame,
            window_start=window_start,
            window_frames=window_frames,
            overlap_frames=overlap_frames,
            conditioned_frames=conditioned_frames,
            supervised_frames=supervised_frames,
            requires_first_frame_reencode=True,
            is_final=end_frame == self.max_sequence_frames,
        )


@dataclass(frozen=True)
class StreamingStepPlan:
    """Frame ranges and masks for one streaming optimizer iteration."""

    rollout_start: int
    rollout_end: int
    window_start: int
    window_frames: int
    overlap_frames: int
    conditioned_frames: int
    supervised_frames: int
    requires_first_frame_reencode: bool
    is_final: bool

    @property
    def rollout_frames(self) -> int:
        return self.rollout_end - self.rollout_start

    @property
    def supervised_slice(self) -> slice:
        return slice(self.conditioned_frames, self.window_frames)


def build_streaming_supervision_window(
    history_tail: torch.Tensor | None,
    new_frames: torch.Tensor,
    plan: StreamingStepPlan,
    reencode_first_frame: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the short teacher window and detached tail for the following step.

    Args:
        history_tail: Detached raw generated history ``[1,C,T_hist,H,W]``.
        new_frames: Current rollout ``[1,C,T_new,H,W]``; may carry gradients.
        plan: Temporal plan produced by :meth:`StreamingLongTuningSpec.plan_step`.
        reencode_first_frame: Callback that decodes a prefix and returns the
            final RGB frame re-encoded as one image latent ``[1,C,1,H,W]``.

    Returns:
        ``(supervision_window, next_history_tail)``. The first frame of the
        supervision window is always detached and re-encoded; gradients are
        retained only through the following generated frames. The second
        tensor is detached raw history.
    """
    if new_frames.ndim != 5:
        raise ValueError(f"new_frames must have shape [B,C,T,H,W], got {tuple(new_frames.shape)}")
    if new_frames.shape[2] != plan.rollout_frames:
        raise ValueError(f"new_frames contains {new_frames.shape[2]} frames, expected {plan.rollout_frames}")

    if history_tail is None:
        combined = new_frames
    else:
        if history_tail.ndim != 5:
            raise ValueError(f"history_tail must have shape [B,C,T,H,W], got {tuple(history_tail.shape)}")
        if history_tail.shape[:2] + history_tail.shape[3:] != new_frames.shape[:2] + new_frames.shape[3:]:
            raise ValueError(
                "history_tail and new_frames must share batch/channel/spatial shapes, "
                f"got {tuple(history_tail.shape)} and {tuple(new_frames.shape)}"
            )
        combined = torch.cat([history_tail.detach(), new_frames], dim=2)

    if combined.shape[2] < plan.window_frames:
        raise ValueError(
            f"Combined history has {combined.shape[2]} frames, fewer than planned window {plan.window_frames}"
        )

    if plan.requires_first_frame_reencode:
        suffix_frames = plan.window_frames - 1
        prefix_end = combined.shape[2] - suffix_frames
        if prefix_end < 1:
            raise ValueError("First-frame re-encoding requires at least one prefix frame")
        image_latent = reencode_first_frame(combined[:, :, :prefix_end].detach())
        expected_shape = (*combined.shape[:2], 1, *combined.shape[3:])
        if tuple(image_latent.shape) != expected_shape:
            raise ValueError(
                "reencode_first_frame must return one image latent with shape "
                f"{expected_shape}, got {tuple(image_latent.shape)}"
            )
        suffix = combined[:, :, -suffix_frames:] if suffix_frames > 0 else combined[:, :, :0]
        window = torch.cat([image_latent.detach(), suffix], dim=2)
    else:
        window = combined[:, :, -plan.window_frames :]

    next_history_tail = combined[:, :, -plan.window_frames :].detach().clone()
    return window, next_history_tail


def build_streaming_supervision_mask(
    plan: StreamingStepPlan,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return ``[T,1,1]`` mask with one only on newly generated frames."""
    mask = torch.zeros((plan.window_frames, 1, 1), device=device, dtype=dtype)
    mask[plan.supervised_slice] = 1
    return mask
