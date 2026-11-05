# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Self-forcing DMD model for Cosmos3 interactive VFM."""

from __future__ import annotations

import collections
import copy
import json
import math
from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import attrs
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict, set_model_state_dict
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy
from torch.nn.modules.module import _IncompatibleKeys

from cosmos_framework.utils.lazy_config import LazyDict
from cosmos_framework.utils import log
from cosmos_framework.model.generator.distillation.common_loss import (
    make_vsd_pseudo_target,
    variational_score_distillation_loss,
)
from cosmos_framework.checkpoint.storage_reader import get_storage_reader
from cosmos_framework.configs.base.defaults.compile import CompileConfig
from cosmos_framework.configs.base.defaults.model_config import FixedStepSamplerConfig
from cosmos_framework.configs.base.defaults.reasoner import PretrainedWeightsConfig, VLMConfig
from cosmos_framework.configs.base.experiment.distillation.dmd2_config import DMD2OptimizerConfig
from cosmos_framework.data.generator.augmentors.duration_fps_text_timestamps import (
    DEFAULT_TEMPLATE as DEFAULT_DURATION_FPS_TEMPLATE,
)
from cosmos_framework.data.generator.augmentors.resolution_text_info import DEFAULT_VIDEO_TEMPLATE
from cosmos_framework.model.generator.distillation.dmd_common import (
    DMDTrainingBase,
    generated_vision_item_indices,
    get_data_resolutions,
    vision_token_count,
)
from cosmos_framework.model.generator.distillation.streaming_long_tuning import (
    StreamingLongTuningSpec,
    StreamingStepPlan,
    build_streaming_supervision_mask,
    build_streaming_supervision_window,
)
from cosmos_framework.model.generator.mot.causal_attention import dispatch_attention_with_memory
from cosmos_framework.model.generator.mot.context_parallel_utils import context_parallel_broadcast_tensor_list
from cosmos_framework.model.generator.omni_mot_causal_model import (
    OmniMoTCausalModel,
    OmniMoTCausalModelConfig,
    _iter_ar_chunk_ranges,
)
from cosmos_framework.model.generator.omni_mot_model import _get_master_dtype_and_mp_policy
from cosmos_framework.model.generator.reasoner.qwen3_vl.utils import tokenize_caption
from cosmos_framework.model.generator.utils.data_and_condition import (
    GenerationDataClean,
    GenerationDataNoised,
    _expand_per_sample_to_per_vision_item,
)
from cosmos_framework.model.generator.utils.fsdp_reshard import defer_fsdp_reshard
from cosmos_framework.model.generator.utils.kv_cache import ARMemoryState, DualKVCache
from cosmos_framework.model.generator.utils.memory import MemoryState
from cosmos_framework.model.generator.utils.multiview_ar import (
    MultiviewTransferARBackend,
    MultiviewTransferARReplayContext,
    multiview_conditioned_prefix_length,
)
from cosmos_framework.data.generator.sequence_packing import (
    PackedSequence,
    SequencePlan,
    build_sequence_plans_from_data_batch,
)
from cosmos_framework.data.generator.sequence_packing.autoregressive import (
    pack_input_sequence_autoregressive,
    resolve_text_system_prompt,
)
from cosmos_framework.data.generator.sequence_packing.modality import compute_text_split_length
from cosmos_framework.model.generator.distillation.optimizer import (
    PhaseOptimizer,
    PhaseScheduler,
)


@attrs.define(slots=False)
class SelfForcingDMDConfig(OmniMoTCausalModelConfig):
    """Config for self-forcing DMD with native AR/KV student rollout."""

    vlm_config: VLMConfig = VLMConfig(pretrained_weights=PretrainedWeightsConfig(enabled=False))
    vlm_config_teacher: VLMConfig = VLMConfig(pretrained_weights=PretrainedWeightsConfig(enabled=False))
    vlm_config_fake_score: VLMConfig = VLMConfig(pretrained_weights=PretrainedWeightsConfig(enabled=False))
    # Override compilation for the trainable scorer while preserving the student policy.
    # None inherits the existing model-wide policy.
    fake_score_compile: CompileConfig | None = None
    fixed_step_sampler_config: FixedStepSamplerConfig = FixedStepSamplerConfig()
    loss_scale_fake_score: float = 1.0
    loss_scale_sid: float = 1.0
    student_update_freq: int = 5
    teacher_guidance: float = 1.0
    # Caption for the teacher CFG negative branch; retain each sample's system
    # prompt. The empty default preserves existing non-transfer recipes.
    teacher_negative_prompt: str = ""
    # Match transfer teacher inference's "same" metadata mode for single-view videos.
    teacher_negative_prompt_keep_metadata: bool = False
    # Joint multiview recipes opt into camera-local metadata and exact regular
    # initialization. Defaults retain the existing generic SF-DMD behavior.
    teacher_negative_prompt_per_view_metadata: bool = False
    strict_regular_checkpoint_loading: bool = False
    joint_sensor_modalities: tuple[str, ...] = ("vision", "lidar")
    action_conditioned_teacher_guidance: float | None = None
    warmup_student_steps: int = 0
    warmup_critic_steps: int = 0
    simulation_mode: Literal["self_forcing"] = "self_forcing"
    sf_rollout_backend: Literal["causal_ar", "multiview_wsm"] = "causal_ar"
    sf_replayed_training: bool = False
    sf_replay_block_size: int = 1
    sf_replay_include_conditioned_frames: bool = False
    # Joint no-grad rollouts reuse detached clean K/V when the sample layout supports it.
    # Unsupported layouts retain replay; student backward keeps the full-prefix noisy gradient pass.
    sf_critic_use_persistent_kv: bool = True
    sf_student_use_persistent_kv: bool = True
    # Reuse rollout clean target K/V during backward instead of rebuilding its clean prefix.
    # Requires the student rollout to use persistent K/V; otherwise falls back to clean replay.
    sf_student_backward_use_persistent_kv: bool = True
    # Optional logical rollout horizon for compact, first-frame-only input.
    sf_rollout_num_pixel_frames: int | None = None
    backward_grad_steps: int = 1
    sf_simulation_steps_per_frame: list[int] = []
    sf_backward_timestep_schedules: list[list[float]] = []
    sf_rollout_step_mode: Literal["cycle", "max"] = "cycle"
    teacher_load_from: Any = None
    student_load_from: Any = None
    load_teacher_weights: bool = True
    vis_debug: bool = False
    sf_streaming_long_tuning: bool = False
    sf_streaming_conditioning_mode: Literal["prompt_only", "camera_action"] = "prompt_only"
    sf_streaming_sequence_pixel_frames: int = 1281
    sf_streaming_step_latent_frames: int = 32
    sf_streaming_attention_window_latent_frames: int = 33
    sf_streaming_attention_sink_latent_frames: int = 1
    sf_streaming_allow_partial_final_step: bool = False
    sf_streaming_debug_rollout_dir: str = ""
    sf_keep_student_fsdp_unsharded_during_rollout: bool = False
    # Reuse the gathered frozen teacher weights across both CFG branches.
    sf_defer_teacher_fsdp_reshard: bool = False
    # Aligned recipes opt in to rejecting inconsistent horizons before replay;
    # truncation can leave their tokenized metadata and conditioning misaligned.
    # Preserve minimum-length synchronization for existing variable-length recipes.
    sf_require_equal_rollout_lengths: bool = False

    def __attrs_post_init__(self) -> None:
        if self.warmup_student_steps > 0 and self.warmup_critic_steps > 0:
            raise ValueError(
                "Only one of warmup_student_steps and warmup_critic_steps can be nonzero, "
                f"got warmup_student_steps={self.warmup_student_steps}, "
                f"warmup_critic_steps={self.warmup_critic_steps}"
            )
        if self.sf_replay_block_size < 1:
            raise ValueError(f"sf_replay_block_size must be >= 1, got {self.sf_replay_block_size}")
        if self.sf_rollout_num_pixel_frames is not None and self.sf_rollout_num_pixel_frames < 1:
            raise ValueError("sf_rollout_num_pixel_frames must be positive when set")
        if self.sf_rollout_step_mode not in ("cycle", "max"):
            raise ValueError(f"sf_rollout_step_mode must be 'cycle' or 'max', got {self.sf_rollout_step_mode!r}")
        if self.sf_rollout_backend not in ("causal_ar", "multiview_wsm"):
            raise ValueError(
                f"sf_rollout_backend must be 'causal_ar' or 'multiview_wsm', got {self.sf_rollout_backend!r}"
            )
        if self.sf_streaming_conditioning_mode not in ("prompt_only", "camera_action"):
            raise ValueError(
                "sf_streaming_conditioning_mode must be 'prompt_only' or 'camera_action', "
                f"got {self.sf_streaming_conditioning_mode!r}"
            )
        if self.sf_streaming_long_tuning and self.sf_replayed_training:
            raise ValueError(
                "sf_streaming_long_tuning keeps only a bounded rolling KV cache and is incompatible with "
                "sf_replayed_training, which replays against the complete rollout cache."
            )
        if self.sf_streaming_sequence_pixel_frames < 1:
            raise ValueError("sf_streaming_sequence_pixel_frames must be positive")
        if self.sf_streaming_step_latent_frames < 1:
            raise ValueError("sf_streaming_step_latent_frames must be positive")


@dataclass(frozen=True)
class _SelfForcingCacheMetadata:
    """Physical cache index and logical Transfer history limits for one AR unit."""

    cache_frame_idx: int
    transfer_history_sink_tokens: int = 0
    transfer_history_max_tokens: int | None = None

    def to_memory_info(self, dual_kv_cache: list[DualKVCache], *, write_gen_cache: bool) -> dict[str, Any]:
        """Build memory metadata for a self-forcing denoise or cache-refresh pass."""
        return {
            "dual_kv_cache": dual_kv_cache,
            "frame_idx": self.cache_frame_idx,
            "write_gen_cache": write_gen_cache,
            "transfer_history_sink_tokens": self.transfer_history_sink_tokens,
            "transfer_history_max_tokens": self.transfer_history_max_tokens,
        }


@dataclass
class _CausalARReplayFrame:
    """Replay input for one ordinary causal-AR frame or chunk."""

    frame_idx: int  # chunk start latent-frame index
    xt: torch.Tensor  # [1,C,chunk_len,H,W]
    sigma: float
    cache_metadata: _SelfForcingCacheMetadata
    action_latent: torch.Tensor | None  # [chunk_len*TCF,D] or None
    text_tokens: list[int] | None
    chunk_len: int = 1
    condition_frame_indexes_vision: tuple[int, ...] = ()


@dataclass
class _MultiviewWSMReplayFrame:
    """Replay input whose cache and layout are owned by the multiview backend."""

    frame_idx: int  # synchronized chunk start latent-frame index
    xt: torch.Tensor  # [1,C,V*chunk_len,H,W]
    sigma: float
    chunk_len: int
    backend_context: MultiviewTransferARReplayContext


@dataclass
class _CausalARReplayState:
    """Backend-owned replay state for ordinary causal AR rollout."""

    frames: list[_CausalARReplayFrame]
    dual_kv_cache: list[DualKVCache]
    fps_vision_list: list[float]
    fps_action_list: list[float] | None
    cached_text_offset: int
    force_action_tokens: bool


@dataclass
class _MultiviewWSMReplayState:
    """Backend-owned replay state for synchronized multiview WSM rollout."""

    frames: list[_MultiviewWSMReplayFrame]


_SelfForcingReplayState = _CausalARReplayState | _MultiviewWSMReplayState


@dataclass
class _SelfForcingSimulationResult:
    """DMD-facing rollout outputs plus one explicitly typed backend replay state."""

    gen_data_student: GenerationDataClean
    packed_reference: PackedSequence
    sigma_student: torch.Tensor  # [B,1]
    out_student: dict[str, Any]
    replay_state: _SelfForcingReplayState | None


@dataclass
class _SelfForcingReplayContext:
    """Detached context used by replayed SF-DMD student closures."""

    sim: _SelfForcingSimulationResult
    target_vision: torch.Tensor  # [1,C,T,H,W]
    iteration: int
    sigmas_critic: torch.Tensor | None = None  # [B,T_vis] or None
    teacher_guidance: float = 1.0


@dataclass
class _StreamingLongState:
    """Persistent detached rollout state shared by consecutive optimizer iterations."""

    input_text_indexes: list[list[int]]
    unconditional_text_indexes: list[list[int]] | None
    sequence_plans: list[SequencePlan]
    gen_data_template: GenerationDataClean
    dual_kv_cache: list[DualKVCache]
    fps_vision_list: list[float]
    fps_action_list: list[float] | None
    force_action_tokens: bool
    cached_text_offset: int
    history_tail: torch.Tensor | None
    current_frame: int
    sequence_index: int
    prompt: str


@dataclass
class _StreamingLongStep:
    """Current short rollout plus its local teacher-scoring view."""

    plan: StreamingStepPlan
    input_text_indexes: list[list[int]]
    unconditional_text_indexes: list[list[int]] | None
    sequence_plans: list[SequencePlan]
    gen_data_student: GenerationDataClean
    packed_reference: PackedSequence
    supervision_mask: torch.Tensor
    out_student: dict[str, Any]


class SelfForcingDMDModel(DMDTrainingBase, OmniMoTCausalModel):
    """Self-forcing DMD model with student samples from causal AR/KV rollout."""

    config: SelfForcingDMDConfig

    def __init__(self, config: SelfForcingDMDConfig):
        self._install_causal_attention_dispatch: bool = True
        # Hold the frozen teacher and the fake-score critic OUTSIDE the nn.Module
        # registry (mirroring TF-dCM's non-persistent teacher), so the routine
        # ``get_model_state_dict`` that VFM inference loading reads sees a
        # student-only ("net.") state dict. Training checkpoints still persist
        # the fake-score net explicitly via the distillation checkpointer.
        self._net_teacher_holder: list[torch.nn.Module] = []
        self._net_fake_score_holder: list[torch.nn.Module] = []
        self._phase_trainable_parameter_masks: dict[str, dict[str, bool]] | None = None
        super().__init__(config)
        self._streaming_long_state: _StreamingLongState | None = None
        self._streaming_critic_step: _StreamingLongStep | None = None
        self._streaming_sequence_index: int = 0
        self._streaming_long_spec: StreamingLongTuningSpec | None = (
            self._build_streaming_long_spec() if self.config.sf_streaming_long_tuning else None
        )

    @property
    def net_teacher(self) -> torch.nn.Module:
        """Frozen teacher kept out of the module registry / routine state dict."""
        if not self._net_teacher_holder:
            raise AttributeError("SF-DMD teacher has not been initialized.")
        return self._net_teacher_holder[0]

    @property
    def net_fake_score(self) -> torch.nn.Module:
        """Fake-score critic kept out of the module registry; persisted explicitly."""
        if not self._net_fake_score_holder:
            raise AttributeError("SF-DMD fake-score network has not been initialized.")
        return self._net_fake_score_holder[0]

    def _expand_self_forcing_steps_per_frame(self, num_frames: int) -> list[int]:
        """Return per-frame backward rollout lengths, repeating the last entry."""
        if num_frames < 1:
            raise ValueError(f"num_frames must be >= 1, got {num_frames}")
        configured_steps = self.config.sf_simulation_steps_per_frame
        if not configured_steps:
            full_t_list = self._default_self_forcing_timestep_schedule()  # [N_steps+1]
            return [len(full_t_list) - 1] * num_frames

        step_counts = [int(step) for step in configured_steps[:num_frames]]
        if len(step_counts) < num_frames:
            step_counts.extend([step_counts[-1]] * (num_frames - len(step_counts)))
        if any(step < 1 for step in step_counts):
            raise ValueError(f"Self-forcing step counts must be positive, got {step_counts}")
        if any(prev < cur for prev, cur in zip(step_counts, step_counts[1:])):
            raise ValueError(f"sf_simulation_steps_per_frame must be non-increasing, got {step_counts}")
        return step_counts

    def _self_forcing_step_counts_for_iteration(
        self,
        num_frames: int,
        effective_iteration: int | None,
    ) -> list[int]:
        """Return rollout step counts for one student/fake-score phase iteration."""
        max_step_counts = self._expand_self_forcing_steps_per_frame(num_frames)
        if self.config.sf_rollout_step_mode == "max" or effective_iteration is None:
            return max_step_counts
        if self.config.sf_rollout_step_mode != "cycle":
            raise ValueError(f"Unsupported sf_rollout_step_mode {self.config.sf_rollout_step_mode!r}")
        global_steps = int(effective_iteration) % max_step_counts[0] + 1
        return [min(global_steps, max_steps) for max_steps in max_step_counts]

    def _expand_self_forcing_midpoint_schedules(self, num_frames: int) -> list[list[float]]:
        """Return per-frame midpoint schedules, repeating the last configured schedule."""
        schedules_config = self.config.sf_backward_timestep_schedules
        if not schedules_config:
            return []
        if not self.config.sf_simulation_steps_per_frame:
            raise ValueError("sf_backward_timestep_schedules requires sf_simulation_steps_per_frame")
        if len(schedules_config) != len(self.config.sf_simulation_steps_per_frame):
            raise ValueError(
                "sf_backward_timestep_schedules must align with sf_simulation_steps_per_frame: "
                f"got {len(schedules_config)} schedules for "
                f"{len(self.config.sf_simulation_steps_per_frame)} frame step entries"
            )

        schedules = []
        for max_steps, schedule in zip(self.config.sf_simulation_steps_per_frame, schedules_config):
            expected_len = int(max_steps) - 1
            if len(schedule) != expected_len:
                raise ValueError(
                    f"Schedule {schedule} has length {len(schedule)}, expected {expected_len} for {max_steps} steps"
                )
            schedules.append([float(timestep) for timestep in schedule])
        if len(schedules) < num_frames:
            schedules.extend([schedules[-1]] * (num_frames - len(schedules)))
        return schedules[:num_frames]

    def _default_self_forcing_timestep_schedule(self) -> list[float]:
        """Return the global self-forcing schedule with the terminal clean step."""
        t_list = [float(timestep) for timestep in self.config.fixed_step_sampler_config.t_list]
        if not t_list:
            raise ValueError("fixed_step_sampler_config.t_list must not be empty")
        if t_list[-1] != 0.0:
            t_list = t_list + [0.0]
        if len(t_list) < 2:
            raise ValueError("fixed_step_sampler_config.t_list must contain a nonzero sigma")
        return t_list

    def _self_forcing_timestep_schedule_for_frame(
        self,
        frame_idx: int,
        num_frames: int,
        step_counts: list[int] | None = None,
    ) -> list[float]:
        """Return the full descending sigma schedule for one AR frame."""
        if frame_idx < 0 or frame_idx >= num_frames:
            raise ValueError(f"frame_idx must be in [0, {num_frames}), got {frame_idx}")
        if step_counts is None:
            step_counts = self._expand_self_forcing_steps_per_frame(num_frames)
        if len(step_counts) != num_frames:
            raise ValueError(f"Expected {num_frames} step counts, got {len(step_counts)}")
        n_steps = step_counts[frame_idx]
        midpoint_schedules = self._expand_self_forcing_midpoint_schedules(num_frames)
        if midpoint_schedules:
            max_midpoints = midpoint_schedules[frame_idx]
            if n_steps > len(max_midpoints) + 1:
                raise ValueError(
                    f"Frame {frame_idx} requests {n_steps} backward steps, but "
                    f"sf_backward_timestep_schedules only provides {len(max_midpoints) + 1}"
                )
            return [1.0, *max_midpoints[: n_steps - 1], 0.0]

        default_schedule = self._default_self_forcing_timestep_schedule()  # [N_steps+1]
        if n_steps > len(default_schedule) - 1:
            raise ValueError(
                f"Frame {frame_idx} requests {n_steps} backward steps, but "
                f"fixed_step_sampler_config.t_list only provides {len(default_schedule) - 1}"
            )
        if n_steps == len(default_schedule) - 1:
            return default_schedule
        return default_schedule[:n_steps] + [0.0]

    def _get_ar_distilled_timestep_schedule(
        self,
        distilled_num_steps: int | None = None,
        frame_idx: int | None = None,
        num_frames: int | None = None,
    ) -> list[float]:
        """Return the SF-DMD frame-aware AR sampling schedule."""
        if frame_idx is None or num_frames is None:
            return super()._get_ar_distilled_timestep_schedule(
                distilled_num_steps=distilled_num_steps,
                frame_idx=frame_idx,
                num_frames=num_frames,
            )
        full_t_list = self._self_forcing_timestep_schedule_for_frame(frame_idx, num_frames)  # [N_steps+1]
        if distilled_num_steps is not None:
            if distilled_num_steps < 1:
                raise ValueError(f"distilled_num_steps must be >= 1, got {distilled_num_steps}")
            if distilled_num_steps < len(full_t_list) - 1:
                full_t_list = full_t_list[:distilled_num_steps] + [0.0]
        return full_t_list

    def _inference_mode_enabled(self) -> bool:
        """Return whether the optional parallelism config enables inference-only setup."""
        parallelism = getattr(self.config, "parallelism", None)
        return bool(getattr(parallelism, "enable_inference_mode", False))

    def _scorer_video_temporal_causal(self) -> bool:
        """Retain bidirectional scoring unless a replay-aware subclass opts in."""
        return False

    def set_up_model(self) -> None:
        """Build a causal student and the selected teacher/fake-score denoisers."""
        is_inference_mode = self._inference_mode_enabled()
        if not is_inference_mode:
            self.config.vlm_config.pretrained_weights.enabled = False
        super().set_up_model()

        if is_inference_mode:
            log.info("[SF-DMD] inference mode: skipping teacher and fake-score networks.")
            self.denoiser_nets = {"student": self.net}
            return

        teacher_load_path = self._get_load_from_value(self.config.teacher_load_from, "load_path")
        if self.config.load_teacher_weights and not teacher_load_path:
            raise ValueError("A pretrained teacher model checkpoint is required for self-forcing distillation")

        saved_vlm_config = self.vlm_config
        # Typed DictConfig assignment can update the existing node in place.
        # Preserve its values before temporarily applying the fake-score policy.
        saved_compile = copy.deepcopy(self.config.compile)
        saved_video_temporal_causal = self.config.video_temporal_causal
        self.config.video_temporal_causal = self._scorer_video_temporal_causal()
        try:
            self.config.vlm_config_teacher.pretrained_weights.enabled = False
            self.vlm_config = self.config.vlm_config_teacher
            no_cast_mp_policy = MixedPrecisionPolicy()
            self._net_teacher_holder = [self.build_net(self.precision, mp_policy=no_cast_mp_policy)]

            self.config.vlm_config_fake_score.pretrained_weights.enabled = False
            self.vlm_config = self.config.vlm_config_fake_score
            fake_score_compile = getattr(self.config, "fake_score_compile", None)
            if fake_score_compile is not None:
                self.config.compile = fake_score_compile
            master_dtype, mp_policy = _get_master_dtype_and_mp_policy(self.config, self.precision)
            self._net_fake_score_holder = [self.build_net(master_dtype, mp_policy=mp_policy)]
        finally:
            self.vlm_config = saved_vlm_config
            self.config.compile = saved_compile
            self.config.video_temporal_causal = saved_video_temporal_causal

        if self.config.load_teacher_weights:
            assert self.config.teacher_load_from is not None
            # Load the FP32 fake-score parameters directly; copying the BF16
            # teacher would discard checkpoint precision before training starts.
            for net in (self.net_teacher, self.net_fake_score):
                self._load_checkpoint_to_net(
                    net,
                    teacher_load_path,
                    credential_path=self._get_load_from_value(self.config.teacher_load_from, "credentials"),
                )

        student_load_from = self.config.student_load_from
        if not self._get_load_from_value(student_load_from, "load_path") and self.config.load_teacher_weights:
            student_load_from = self.config.teacher_load_from
        student_load_path = self._get_load_from_value(student_load_from, "load_path")
        if student_load_path:
            self._load_checkpoint_to_net(
                self.net,
                student_load_path,
                credential_path=self._get_load_from_value(student_load_from, "credentials"),
            )
            if self.config.ema.enabled:
                self.net_ema_worker.copy_to(src_model=self.net, tgt_model=self.net_ema)

        self.net_teacher.requires_grad_(False)
        self.net_teacher.eval()
        self.denoiser_nets = {
            "teacher": self.net_teacher,
            "fake_score": self.net_fake_score,
            "student": self.net,
        }

    def install_attention_dispatch(self, net: torch.nn.Module) -> None:
        """Install the memory-aware causal attention dispatcher on every tower layer."""
        if not getattr(self, "_install_causal_attention_dispatch", True):
            return
        net.language_model.model.pad_for_cuda_graphs = getattr(net, "pad_for_cuda_graphs", False)
        for layer in net.language_model.model.layers:
            layer.self_attn.dispatch_attention_fn = dispatch_attention_with_memory

    def on_train_start(self, memory_format: torch.memory_format = torch.preserve_format) -> None:
        """Initialize phase-trainable FSDP roots before either network can be frozen.

        FSDP2 derives a parameter group's original and reduction dtypes during its
        one-shot lazy initialization. If the first forward happens while every
        parameter in that group is frozen, PyTorch leaves both dtypes unset. A
        later student phase then reduces BF16 gradients without restoring them to
        the FP32 master-parameter dtype and fails while assigning the shard grad.
        Distillation alternates frozen student and critic networks, so eagerly
        initializing both roots while they are trainable is required for resumes
        whose first phase is not a student update.
        """
        super().on_train_start(memory_format)
        if bool(getattr(self.config.parallelism, "enable_inference_mode", False)):
            return

        for network_name, network in (("student", self.net), ("fake_score", self.net_fake_score)):
            if not isinstance(network, FSDPModule):
                continue
            if not any(parameter.requires_grad for parameter in network.parameters()):
                raise RuntimeError(f"Cannot initialize the {network_name} FSDP root while all parameters are frozen.")
            network._get_fsdp_state()._lazy_init()
            log.info(f"[SF-DMD] eagerly initialized the {network_name} FSDP root before phase freezing")

    def build_memory_state(
        self,
        packed_seq: PackedSequence,
        memory_info: dict[str, Any],
    ) -> MemoryState | None:
        """Use causal inference memory or build training self-forcing memory."""
        if self._inference_mode_enabled():
            return super().build_memory_state(packed_seq, memory_info)

        dual_kv_cache: list[DualKVCache] | None = memory_info["dual_kv_cache"]
        if dual_kv_cache is None:
            return None

        vision_token_shapes = packed_seq.vision.token_shapes if packed_seq.vision else None
        replicated_attention_io_cp = (
            self.config.attention_io_layout == "replicated"
            and self.parallel_dims is not None
            and self.parallel_dims.cp_enabled
        )
        post_saturation_static_compile = memory_info.get("post_saturation_static_compile", False)
        coarse_cuda_graph = memory_info.get("coarse_cuda_graph", False)
        stage_gen_cache_writes = memory_info.get("stage_gen_cache_writes", False)
        static_und_cache_max_len = memory_info.get("static_und_cache_max_len")
        if static_und_cache_max_len is None and post_saturation_static_compile:
            static_und_cache_max_len = getattr(getattr(self, "config", None), "ar_static_und_cache_max_len", 512)
        return ARMemoryState(
            dual_kv_cache=dual_kv_cache,
            frame_idx=memory_info["frame_idx"],
            vision_token_shapes=vision_token_shapes,
            num_action_tokens_per_supertoken=packed_seq.num_action_tokens_per_supertoken,
            null_action_supertokens=packed_seq.null_action_supertokens,
            num_kv_heads=self.net.num_kv_heads if replicated_attention_io_cp or coarse_cuda_graph else None,
            head_dim=self.net.head_dim if replicated_attention_io_cp or coarse_cuda_graph else None,
            write_gen_cache=memory_info.get("write_gen_cache", True),
            kv_head_shard_rank=self.parallel_dims.cp_rank if replicated_attention_io_cp else 0,
            kv_head_shard_size=self.parallel_dims.cp_size if replicated_attention_io_cp else 1,
            post_saturation_static_compile=post_saturation_static_compile,
            static_und_cache_max_len=static_und_cache_max_len if post_saturation_static_compile else None,
            coarse_cuda_graph=coarse_cuda_graph,
            stage_gen_cache_writes=stage_gen_cache_writes,
            transfer_history_sink_tokens=memory_info.get("transfer_history_sink_tokens", 0),
            transfer_history_max_tokens=memory_info.get("transfer_history_max_tokens"),
        )

    def _build_streaming_long_spec(self) -> StreamingLongTuningSpec:
        """Resolve pixel durations to latent frames and validate the LongLive schedule."""
        tokenizer = self.tokenizer_vision_gen
        if tokenizer is None:
            raise RuntimeError("Streaming long tuning requires a vision tokenizer")
        temporal_factor = int(tokenizer.temporal_compression_factor)
        sequence_pixel_frames = int(self.config.sf_streaming_sequence_pixel_frames)
        if (sequence_pixel_frames - 1) % temporal_factor != 0:
            raise ValueError(
                "sf_streaming_sequence_pixel_frames must map exactly to causal VAE latents, "
                f"got ({sequence_pixel_frames} - 1) % {temporal_factor}"
            )
        step_latent_frames = int(self.config.sf_streaming_step_latent_frames)
        supervision_window_frames = step_latent_frames + 1
        supervision_window_pixel_frames = (supervision_window_frames - 1) * temporal_factor + 1

        spec = StreamingLongTuningSpec(
            max_sequence_frames=tokenizer.get_latent_num_frames(sequence_pixel_frames),
            supervision_window_frames=supervision_window_frames,
            new_frames_per_step=step_latent_frames,
            causal_chunk_size=max(int(self.config.teacher_forcing_frames_per_chunk), 1),
            attention_window_frames=int(self.config.sf_streaming_attention_window_latent_frames),
            attention_sink_frames=int(self.config.sf_streaming_attention_sink_latent_frames),
            allow_partial_final_step=bool(self.config.sf_streaming_allow_partial_final_step),
        )
        log.info(
            "[SLT] enabled: "
            f"window={spec.supervision_window_frames} latent/{supervision_window_pixel_frames} pixel frames, "
            f"new={spec.new_frames_per_step}, "
            f"max={spec.max_sequence_frames} latent/{sequence_pixel_frames} pixel frames, "
            f"attention={spec.local_attention_frames} local + {spec.attention_sink_frames} sink, "
            f"cache_capacity={spec.kv_cache_capacity}"
        )
        return spec

    @staticmethod
    def _batch_scalar(data_batch: dict[str, Any], key: str, expected_type: type[int] | type[float]) -> int | float:
        value = data_batch[key]
        if isinstance(value, list):
            value = value[0]
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError(f"{key} must contain one scalar, got shape {tuple(value.shape)}")
            value = value.item()
        return expected_type(value)

    @staticmethod
    def _batch_string(data_batch: dict[str, Any], key: str) -> str:
        value = data_batch.get(key, "")
        if isinstance(value, (list, tuple)):
            value = value[0] if value else ""
        return str(value)

    @staticmethod
    def _is_prompt_only_batch(data_batch: dict[str, Any]) -> bool:
        marker = data_batch.get("sf_prompt_only", False)
        if isinstance(marker, list):
            return len(marker) > 0 and all(
                bool(item.item() if isinstance(item, torch.Tensor) else item) for item in marker
            )
        if isinstance(marker, torch.Tensor):
            return bool(marker.numel() > 0 and marker.bool().all().item())
        return bool(marker)

    def _make_prompt_only_generation_template(self, data_batch: dict[str, Any]) -> GenerationDataClean:
        """Create one latent reference frame without loading or encoding a real video."""
        if not self._is_prompt_only_batch(data_batch):
            raise ValueError("Streaming long tuning currently requires an sf_prompt_only data batch")
        if self._streaming_long_spec is None:
            raise RuntimeError("Streaming long tuning spec has not been initialized")

        sequence_pixel_frames = int(self.config.sf_streaming_sequence_pixel_frames)
        batch_pixel_frames = int(self._batch_scalar(data_batch, "sf_num_pixel_frames", int))
        if batch_pixel_frames != sequence_pixel_frames:
            raise ValueError(
                f"Prompt-only batch sf_num_pixel_frames={batch_pixel_frames} does not match "
                f"sf_streaming_sequence_pixel_frames={sequence_pixel_frames}"
            )

        tokenizer = self.tokenizer_vision_gen
        assert tokenizer is not None
        spatial_factor = int(tokenizer.spatial_compression_factor)
        height = int(self._batch_scalar(data_batch, "sf_frame_height", int))
        width = int(self._batch_scalar(data_batch, "sf_frame_width", int))
        if height < 1 or width < 1:
            raise ValueError(f"Prompt-only height/width must be positive, got {(height, width)}")
        if height % spatial_factor != 0 or width % spatial_factor != 0:
            raise ValueError(
                "Prompt-only height/width must be divisible by the tokenizer spatial factor, "
                f"got {(height, width)} and factor {spatial_factor}"
            )
        latent_height = height // spatial_factor
        latent_width = width // spatial_factor
        latent_channels = int(tokenizer.latent_ch)
        reference_frame = torch.zeros(
            (1, latent_channels, 1, latent_height, latent_width),
            **self.tensor_kwargs_fp32,
        )  # [1,C,1,H_lat,W_lat]
        fps = float(self._batch_scalar(data_batch, "conditioning_fps", float))
        if fps <= 0:
            raise ValueError(f"Prompt-only fps must be positive, got {fps}")

        return GenerationDataClean(
            batch_size=1,
            is_image_batch=False,
            raw_state_vision=None,
            x0_tokens_vision=[reference_frame],
            fps_vision=torch.tensor([fps], **self.tensor_kwargs_fp32),
            num_vision_items_per_sample=None,
            raw_state_sound=None,
            x0_tokens_sound=None,
            fps_sound=None,
            raw_state_action=None,
            x0_tokens_action=None,
            fps_action=None,
            action_domain_id=None,
            raw_action_dim=None,
        )

    def _start_streaming_long_sequence(
        self,
        data_batch: dict[str, Any],
        iteration: int,
    ) -> _StreamingLongState:
        """Initialize one prompt-only or camera-action streaming sequence."""
        if self.config.sound_gen:
            raise NotImplementedError("Streaming long tuning does not support sound")
        if self.parallel_dims is not None and (self.parallel_dims.cp_enabled or self.parallel_dims.cfgp_enabled):
            raise ValueError("Streaming long tuning does not support CP/CFGP yet")
        if self._streaming_long_spec is None:
            raise RuntimeError("Streaming long tuning spec has not been initialized")

        if self.parallel_dims is None or self.parallel_dims.cp_rank == 0:
            self._update_train_stats(data_batch)
        input_text_indexes = self._load_and_tokenize_text_data(data_batch, iteration)
        sequence_plans = build_sequence_plans_from_data_batch(
            data_batch=data_batch,
            input_video_key=self.input_video_key,
            input_image_key=self.input_image_key,
        )
        if len(input_text_indexes) != 1 or len(sequence_plans) != 1:
            raise ValueError("Streaming long tuning currently supports exactly one prompt per rank")
        plan = sequence_plans[0]
        mode = self.config.sf_streaming_conditioning_mode
        if mode == "prompt_only":
            if self.config.action_gen or plan.has_action or plan.has_sound or plan.condition_frame_indexes_vision:
                raise ValueError(
                    "Prompt-only streaming long tuning requires unconditioned text-to-video sequence plans"
                )
            gen_data_template = self._make_prompt_only_generation_template(data_batch)
            current_frame = 0
            history_tail = None
        else:
            input_text_indexes = [list(tokens) for tokens in input_text_indexes]
            if not self.config.action_gen or not plan.has_action or plan.has_sound:
                raise ValueError("Camera-action streaming requires action_gen and an action-conditioned vision plan")
            if plan.condition_frame_indexes_vision != [0]:
                raise ValueError(
                    "Camera-action streaming requires exactly latent vision frame 0 as conditioning, "
                    f"got {plan.condition_frame_indexes_vision}"
                )
            # Camera SF-DMD consumes only the clean initial frame from the source
            # video. Actions define the rollout horizon independently of source
            # video length, so avoid retaining or encoding target video frames.
            with torch.no_grad():
                gen_data_template = self.get_data_and_condition(data_batch, vision_condition_indexes=[[0]])
            if gen_data_template.batch_size != 1:
                raise ValueError("Camera-action streaming supports exactly one sample per rank")
            if gen_data_template.x0_tokens_vision is None or gen_data_template.x0_tokens_action is None:
                raise ValueError("Camera-action streaming requires encoded vision and action tokens")
            vision_tokens = gen_data_template.x0_tokens_vision[0]
            expected_frames = self._streaming_long_spec.max_sequence_frames
            tcf = self.tokenizer_vision_gen.temporal_compression_factor or 4
            expected_actions = (expected_frames - 1) * tcf
            action_tokens = gen_data_template.x0_tokens_action[0]
            actual_actions = action_tokens.shape[-2]
            if actual_actions < expected_actions:
                raise ValueError(
                    f"Camera-action streaming needs at least {expected_actions} action tokens, got {actual_actions}"
                )
            initial_frame = self._slice_vision_frame(vision_tokens, 0).detach().clone()
            gen_data_template.raw_state_vision = None
            gen_data_template.raw_state_action = None
            gen_data_template.x0_tokens_vision = [initial_frame]
            gen_data_template.x0_tokens_action = [action_tokens[..., :expected_actions, :].contiguous()]
            current_frame = 1
            history_tail = initial_frame

        num_layers: int = self.net.num_hidden_layers  # type: ignore[attr-defined]
        dual_kv_cache = [
            DualKVCache(
                gen_cache_size=self._streaming_long_spec.kv_cache_capacity,
                attention_sink_size=self._streaming_long_spec.attention_sink_frames,
            )
            for _ in range(num_layers)
        ]
        fps_vision_list = gen_data_template.fps_vision.tolist() if gen_data_template.fps_vision is not None else [24.0]
        fps_action_list = (
            gen_data_template.fps_action.tolist()
            if gen_data_template.fps_action is not None and plan.has_action
            else None
        )
        force_action_tokens = bool(self.config.action_gen and plan.has_action)
        text_tokens = input_text_indexes[0]
        cached_text_offset = compute_text_split_length(
            len(text_tokens),
            self.llm_special_tokens,
            has_generation=True,
        )
        state = _StreamingLongState(
            input_text_indexes=input_text_indexes,
            unconditional_text_indexes=(
                self._make_unconditional_text_indexes(
                    gen_data_template.batch_size, data_batch, is_image_batch=gen_data_template.is_image_batch
                )
                if self._teacher_guidance_for_batch(sequence_plans) > 1.0
                else None
            ),
            sequence_plans=sequence_plans,
            gen_data_template=gen_data_template,
            dual_kv_cache=dual_kv_cache,
            fps_vision_list=fps_vision_list,
            fps_action_list=fps_action_list,
            force_action_tokens=force_action_tokens,
            cached_text_offset=cached_text_offset,
            history_tail=history_tail,
            current_frame=current_frame,
            sequence_index=self._streaming_sequence_index,
            prompt=self._batch_string(data_batch, "prompt") or self._batch_string(data_batch, "ai_caption"),
        )
        if force_action_tokens:
            self._seed_self_forcing_history(
                frame_latent=state.history_tail,
                action_latent=None,
                frame_idx=0,
                dual_kv_cache=dual_kv_cache,
                text_tokens=text_tokens,
                cached_text_offset=cached_text_offset,
                gen_data_clean=gen_data_template,
                fps_vision_list=fps_vision_list,
                fps_action_list=fps_action_list,
                force_action_tokens=True,
            )
        self._streaming_sequence_index += 1
        self._streaming_long_state = state
        log.info(f"[SLT] starting sequence {state.sequence_index} at iteration {iteration}")
        return state

    @staticmethod
    def _is_streaming_continuation(data_batch: dict[str, Any]) -> bool:
        marker = data_batch.get("sf_streaming_continuation", False)
        if isinstance(marker, (list, tuple)):
            return bool(marker) and all(
                bool(item.item() if isinstance(item, torch.Tensor) else item) for item in marker
            )
        if isinstance(marker, torch.Tensor):
            return bool(marker.numel() > 0 and marker.bool().all().item())
        return bool(marker)

    def _get_or_start_streaming_long_sequence(
        self,
        data_batch: dict[str, Any],
        iteration: int,
    ) -> _StreamingLongState:
        state = self._streaming_long_state
        if state is not None and self._streaming_long_spec is not None:
            if state.current_frame >= self._streaming_long_spec.max_sequence_frames:
                self._streaming_long_state = None
                state = None
        is_continuation = self._is_streaming_continuation(data_batch)
        if state is None:
            if is_continuation:
                raise ValueError("Streaming continuation received without an active incomplete sequence")
            return self._start_streaming_long_sequence(data_batch, iteration)
        if state.force_action_tokens and not is_continuation:
            raise ValueError("Active camera-action streaming sequence requires a continuation batch")
        return state

    @torch.no_grad()
    def _reencode_streaming_boundary_frame(self, prefix_latents: torch.Tensor) -> torch.Tensor:
        """Decode a generated prefix and image-encode its final RGB frame."""
        if prefix_latents.ndim != 5 or prefix_latents.shape[2] < 1:
            raise ValueError(f"Expected non-empty [B,C,T,H,W] prefix, got {tuple(prefix_latents.shape)}")
        pixels = self.decode(prefix_latents)  # [B,3,T_pixel,H,W]
        last_pixel_frame = pixels[:, :, -1:].contiguous()  # [B,3,1,H,W]
        image_latent = self.encode(last_pixel_frame)  # [B,C,1,H_lat,W_lat]
        return image_latent.to(device=prefix_latents.device, dtype=prefix_latents.dtype)

    def _rollout_streaming_long_step(
        self,
        state: _StreamingLongState,
        plan: StreamingStepPlan,
        *,
        allow_grad: bool,
    ) -> _StreamingLongStep:
        """Roll out only the current short clip and construct its teacher window."""
        if self._streaming_long_spec is None:
            raise RuntimeError("Streaming long tuning spec has not been initialized")
        reference_frame = state.gen_data_template.x0_tokens_vision[0]
        generated_chunks: list[torch.Tensor] = []
        out_student: dict[str, Any] = {}
        chunk_size = self._streaming_long_spec.causal_chunk_size
        action_tokens = (
            state.gen_data_template.x0_tokens_action[0]
            if state.force_action_tokens and state.gen_data_template.x0_tokens_action is not None
            else None
        )
        tcf = self.tokenizer_vision_gen.temporal_compression_factor or 4

        with self._student_rollout_reshard_context():
            for chunk_start, chunk_end in _iter_ar_chunk_ranges(
                plan.rollout_start,
                plan.rollout_end,
                chunk_size,
            ):
                chunk_len = chunk_end - chunk_start
                noise = torch.randn(
                    (1, reference_frame.shape[1], chunk_len, reference_frame.shape[3], reference_frame.shape[4]),
                    **self.tensor_kwargs_fp32,
                )  # [1,C,T_chunk,H,W]
                text_tokens = state.input_text_indexes[0] if chunk_start == 0 else None
                action_chunk = self._slice_action_for_chunk(action_tokens, chunk_start, chunk_end, tcf)
                x0_chunk, out_student, _replay_xt, _replay_sigma = self._denoise_self_forcing_frame(
                    frame_latent=noise,
                    action_latent=action_chunk,
                    frame_idx=chunk_start,
                    num_frames=self._streaming_long_spec.max_sequence_frames,
                    dual_kv_cache=state.dual_kv_cache,
                    text_tokens=text_tokens,
                    cached_text_offset=state.cached_text_offset,
                    gen_data_clean=state.gen_data_template,
                    fps_vision_list=state.fps_vision_list,
                    fps_action_list=state.fps_action_list,
                    force_action_tokens=state.force_action_tokens,
                    allow_grad=allow_grad and chunk_start > 0,
                    collect_replay=False,
                )
                generated_chunks.append(x0_chunk)

                # Store finalized, detached per-frame K/V. This is the only cache
                # state carried into the following optimizer iteration.
                for offset, frame_idx in enumerate(range(chunk_start, chunk_end)):
                    self._seed_self_forcing_history(
                        frame_latent=x0_chunk[:, :, offset : offset + 1],
                        action_latent=self._slice_action_for_frame(action_tokens, frame_idx, tcf),
                        frame_idx=frame_idx,
                        dual_kv_cache=state.dual_kv_cache,
                        text_tokens=state.input_text_indexes[0] if frame_idx == 0 else None,
                        cached_text_offset=state.cached_text_offset,
                        gen_data_clean=state.gen_data_template,
                        fps_vision_list=state.fps_vision_list,
                        fps_action_list=state.fps_action_list,
                        force_action_tokens=state.force_action_tokens,
                    )

        new_frames = torch.cat(generated_chunks, dim=2)  # [1,C,T_new,H,W]
        supervision_window, next_history_tail = build_streaming_supervision_window(
            history_tail=state.history_tail,
            new_frames=new_frames,
            plan=plan,
            reencode_first_frame=self._reencode_streaming_boundary_frame,
        )
        state.history_tail = next_history_tail
        state.current_frame = plan.rollout_end

        local_action_tokens = None
        local_action_domain_id = None
        local_action_condition_indexes: list[int] = []
        if state.force_action_tokens:
            local_action = self._slice_action_for_chunk(
                action_tokens,
                plan.window_start + 1,
                plan.rollout_end,
                tcf,
            )
            if local_action is None:
                raise RuntimeError("Camera-action streaming produced an empty local action window")
            local_action_tokens = [local_action]
            local_domain_ids = self._slice_action_domain_for_ar_unit(
                OmniMoTCausalModel._first_action_domain_id(state.gen_data_template),
                local_action,
                plan.window_start + 1,
                tcf,
            )
            local_action_domain_id = [local_domain_ids] if local_domain_ids is not None else None
            local_action_condition_indexes = list(range(local_action.shape[0]))

        gen_data_student = GenerationDataClean(
            batch_size=1,
            is_image_batch=False,
            raw_state_vision=None,
            x0_tokens_vision=[supervision_window],
            fps_vision=state.gen_data_template.fps_vision,
            num_vision_items_per_sample=None,
            raw_state_sound=None,
            x0_tokens_sound=None,
            fps_sound=None,
            raw_state_action=None,
            x0_tokens_action=local_action_tokens,
            fps_action=state.gen_data_template.fps_action if state.force_action_tokens else None,
            action_domain_id=local_action_domain_id,
            raw_action_dim=state.gen_data_template.raw_action_dim if state.force_action_tokens else None,
        )
        local_sequence_plans = [
            SequencePlan(
                has_text=True,
                has_vision=True,
                condition_frame_indexes_vision=[0],
                has_action=state.force_action_tokens,
                condition_frame_indexes_action=local_action_condition_indexes,
            )
        ]
        packed_reference = self._pack_input_sequence(
            local_sequence_plans,
            state.input_text_indexes,
            gen_data_student,
            torch.zeros((1, 1), dtype=torch.float32),
        )
        supervision_mask = build_streaming_supervision_mask(
            plan,
            device=supervision_window.device,
            dtype=supervision_window.dtype,
        )
        log.info(
            f"[SLT] sequence={state.sequence_index}, rollout=[{plan.rollout_start},{plan.rollout_end}), "
            f"teacher_window=[{plan.window_start},{plan.rollout_end}), overlap={plan.overlap_frames}, "
            f"final={plan.is_final}"
        )
        return _StreamingLongStep(
            plan=plan,
            input_text_indexes=state.input_text_indexes,
            unconditional_text_indexes=state.unconditional_text_indexes,
            sequence_plans=local_sequence_plans,
            gen_data_student=gen_data_student,
            packed_reference=packed_reference,
            supervision_mask=supervision_mask,
            out_student=out_student,
        )

    def _save_streaming_debug_rollout(
        self,
        state: _StreamingLongState,
        step: _StreamingLongStep,
        iteration: int,
    ) -> None:
        """Persist the exact pre-teacher student window for offline VAE decoding."""
        output_dir = self.config.sf_streaming_debug_rollout_dir.strip()
        if not output_dir:
            return
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return

        assert step.gen_data_student.x0_tokens_vision is not None
        latents = step.gen_data_student.x0_tokens_vision[0].detach().to(device="cpu", dtype=torch.bfloat16)
        mask = step.supervision_mask.detach().to(device="cpu", dtype=torch.float32)
        phase = self.get_phase(iteration)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        stem = (
            f"iter_{iteration:03d}_{phase}_seq_{state.sequence_index:03d}_"
            f"window_{step.plan.window_start:04d}_{step.plan.rollout_end:04d}"
        )
        latent_path = output_path / f"{stem}.pt"
        metadata = {
            "iteration": iteration,
            "phase": phase,
            "sequence_index": state.sequence_index,
            "rollout_start": step.plan.rollout_start,
            "rollout_end": step.plan.rollout_end,
            "window_start": step.plan.window_start,
            "window_frames": step.plan.window_frames,
            "conditioned_frames": step.plan.conditioned_frames,
            "supervised_frames": step.plan.supervised_frames,
            "first_frame_reencoded": step.plan.requires_first_frame_reencode,
            "prompt": state.prompt,
        }
        torch.save({"latents": latents, "supervision_mask": mask, "metadata": metadata}, latent_path)
        latent_path.with_suffix(".json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        log.info(f"[SLT debug] saved pre-teacher rollout to {latent_path}")

    def _prepare_streaming_long_step(
        self,
        data_batch: dict[str, Any],
        iteration: int,
        *,
        allow_grad: bool,
    ) -> _StreamingLongStep:
        state = self._get_or_start_streaming_long_sequence(data_batch, iteration)
        assert self._streaming_long_spec is not None
        plan = self._streaming_long_spec.plan_step(
            state.current_frame,
            initial_condition_frames=1 if state.force_action_tokens else 0,
        )
        step = self._rollout_streaming_long_step(state, plan, allow_grad=allow_grad)
        self._save_streaming_debug_rollout(state, step, iteration)
        return step

    def _detach_streaming_long_step(self, step: _StreamingLongStep) -> _StreamingLongStep:
        """Create a graph-free copy of a student window for its critic updates."""
        gen_data_student = replace(
            step.gen_data_student,
            x0_tokens_vision=(
                [tokens.detach() for tokens in step.gen_data_student.x0_tokens_vision]
                if step.gen_data_student.x0_tokens_vision is not None
                else None
            ),
            x0_tokens_action=(
                [tokens.detach() for tokens in step.gen_data_student.x0_tokens_action]
                if step.gen_data_student.x0_tokens_action is not None
                else None
            ),
        )
        packed_reference = self._pack_input_sequence(
            step.sequence_plans,
            step.input_text_indexes,
            gen_data_student,
            torch.zeros((1, 1), dtype=torch.float32),
        )
        return replace(
            step,
            gen_data_student=gen_data_student,
            packed_reference=packed_reference,
            supervision_mask=step.supervision_mask.detach(),
            out_student={},
        )

    def _get_streaming_critic_step(self, data_batch: dict[str, Any]) -> _StreamingLongStep:
        """Return the current student window without advancing persistent rollout state."""
        step = self._streaming_critic_step
        state = self._streaming_long_state
        if step is None or state is None:
            raise ValueError("Streaming critic update requires a preceding student rollout window")
        if state.force_action_tokens and not self._is_streaming_continuation(data_batch):
            raise ValueError("Active camera-action streaming sequence requires a continuation batch")
        return step

    def _streaming_step_for_phase(
        self,
        data_batch: dict[str, Any],
        iteration: int,
        *,
        student_phase: bool,
    ) -> _StreamingLongStep:
        """Advance on a student update, then reuse that window for its critic updates."""
        if not student_phase:
            return self._get_streaming_critic_step(data_batch)
        self._streaming_critic_step = None
        step = self._prepare_streaming_long_step(data_batch, iteration, allow_grad=True)
        self._streaming_critic_step = self._detach_streaming_long_step(step)
        return step

    def _setup_grad_requirements(self, iteration: int) -> None:
        masks = getattr(self, "_phase_trainable_parameter_masks", None)
        if masks is None:
            raise RuntimeError("Initialize SF-DMD optimizers before switching training phases.")
        active_phase = self.get_phase(iteration)
        for phase, network in (("student", self.net), ("critic", self.net_fake_score)):
            active = phase == active_phase
            # Detached forwards can leave root or resident child parameters
            # unsharded. Restore the optimizer-owned parameters before changing
            # trainability; changing only temporary full parameters would be
            # undone by the next unshard after the first replay backward.
            for module in network.modules():
                if isinstance(module, FSDPModule):
                    module.reshard()
            network.train(active)
            # FSDP can swap parameter objects; names retain the optimizer's
            # original trainability mask across unshard/reshard transitions.
            for name, parameter in network.named_parameters():
                parameter.requires_grad_(active and masks[phase][name])  # [*parameter.shape]
        self.net_teacher.eval().requires_grad_(False)

    def _student_rollout_reshard_context(self) -> AbstractContextManager[None]:
        """Retain rollout weights while preserving the original FSDP policies."""
        enabled = self.config.sf_keep_student_fsdp_unsharded_during_rollout
        return defer_fsdp_reshard(
            self.net,
            after_forward=enabled,
            reshard_root_on_exit=enabled,
        )

    @staticmethod
    def _get_load_from_value(load_from: Any, key: str) -> Any:
        if load_from is None:
            return None
        if isinstance(load_from, dict):
            return load_from.get(key)
        return getattr(load_from, key, None)

    def _load_checkpoint_to_net(
        self,
        net: torch.nn.Module,
        ckpt_path: str,
        prefix: str = "net_ema",
        credential_path: str | None = None,
    ) -> None:
        """Load a DCP checkpoint into one denoiser network."""
        storage_reader = get_storage_reader(ckpt_path, credential_path)
        if ckpt_path.endswith(".dcp/model"):
            prefix = "net"
        state_dict = get_model_state_dict(net)

        metadata = storage_reader.read_metadata()
        checkpoint_keys = set(metadata.state_dict_metadata.keys())
        prefixed_model_keys = {f"{prefix}.{key}" for key in state_dict.keys()}
        missing_keys = sorted(key for key in prefixed_model_keys - checkpoint_keys if not key.endswith("_extra_state"))
        if missing_keys:
            raise RuntimeError(
                f"Self-forcing DMD checkpoint load is missing model keys; first missing keys: {missing_keys[:20]}"
            )

        unexpected_keys = sorted(checkpoint_keys - prefixed_model_keys)
        if prefix == "net_ema":
            unexpected_keys = [key for key in unexpected_keys if "net." not in key]
        else:
            unexpected_keys = [key for key in unexpected_keys if "net_ema." not in key]
        unexpected_keys = [key for key in unexpected_keys if "_extra_state" not in key]
        if unexpected_keys:
            log.warning(f"Unexpected keys in checkpoint: {unexpected_keys[:20]}")

        prefixed_state_dict = collections.OrderedDict()
        for key in state_dict.keys():
            prefixed_state_dict[f"{prefix}.{key}"] = state_dict[key]
        dcp.load(
            prefixed_state_dict, storage_reader=storage_reader, planner=DefaultLoadPlanner(allow_partial_load=True)
        )
        for key in state_dict.keys():
            state_dict[key] = prefixed_state_dict[f"{prefix}.{key}"]

        log.info(set_model_state_dict(net, state_dict, options=StateDictOptions(strict=False)))
        del state_dict, prefixed_state_dict

    def init_optimizer_scheduler(
        self,
        optimizer_config: DMD2OptimizerConfig,
        scheduler_config: LazyDict,
    ) -> tuple[PhaseOptimizer, PhaseScheduler]:
        """Create phase-aware optimizers for the student and fake-score denoisers."""
        optimizer, scheduler = super().init_optimizer_scheduler(optimizer_config, scheduler_config)
        # Optimizer construction freezes parameters excluded by keys_to_select.
        # Save that result before either phase temporarily freezes its network.
        self._phase_trainable_parameter_masks = {
            phase: {name: parameter.requires_grad for name, parameter in network.named_parameters()}
            for phase, network in (("student", self.net), ("critic", self.net_fake_score))
        }
        return optimizer, scheduler

    def get_fake_score_iteration(self, iteration: int) -> int:
        """Return the effective fake-score update index used by the SF rollout curriculum."""
        if iteration < self.config.warmup_critic_steps:
            return iteration
        return max(iteration - self.get_student_iteration(iteration) - 1, 0)

    # NOTE: ``state_dict`` is intentionally NOT overridden to add net_fake_score.
    # net_fake_score is held outside the module registry, and ``get_model_state_dict``
    # (used by VFM inference loading) calls ``state_dict()``; re-adding net_fake_score
    # here would reintroduce non-"net." keys and break ``load_ema_to_reg``. The
    # fake-score net is persisted/restored explicitly by the distillation
    # checkpointer (see dcp_distill_cosmos3.ModelWrapper).

    def model_dict(self) -> dict[str, torch.nn.Module]:
        return {"net": self.net, "fake_score": self.net_fake_score}

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        # net_fake_score is held outside the module registry, so route its keys to
        # the fake-score net explicitly and keep them out of the registry load.
        # OmniMoTModel.load_state_dict only supports non-strict loading; strictness
        # is enforced by the distillation checkpointer (ModelWrapper.strict_resume)
        # via the returned missing/unexpected keys.
        del strict
        main_state_dict = collections.OrderedDict(
            (key, value) for key, value in state_dict.items() if not key.startswith("net_fake_score.")
        )
        base_results: _IncompatibleKeys = super().load_state_dict(main_state_dict, strict=False, assign=assign)
        fake_score_state_dict = collections.OrderedDict()
        for key, value in state_dict.items():
            if key.startswith("net_fake_score."):
                fake_score_state_dict[key.replace("net_fake_score.", "")] = value

        missing_keys = list(base_results.missing_keys)
        unexpected_keys = list(base_results.unexpected_keys)
        # Only load net_fake_score here when its keys are present (e.g. a full
        # state dict). The distillation checkpointer loads it separately.
        if fake_score_state_dict:
            fake_score_results: _IncompatibleKeys = self.net_fake_score.load_state_dict(
                fake_score_state_dict, strict=False, assign=assign
            )
            missing_keys += [f"net_fake_score.{key}" for key in fake_score_results.missing_keys]
            unexpected_keys += [f"net_fake_score.{key}" for key in fake_score_results.unexpected_keys]
        return _IncompatibleKeys(missing_keys=missing_keys, unexpected_keys=unexpected_keys)

    @staticmethod
    def _velocity_to_x0(
        x_t: list[torch.Tensor],  # list of [C,...]
        v_pred: list[torch.Tensor],  # list of [C,...]
        sigma: list[torch.Tensor],  # list broadcastable to [C,...]
    ) -> list[torch.Tensor]:
        """Convert RF velocity predictions to clean endpoints."""
        return [
            x_t[i].float() - sigma[i].to(device=x_t[i].device, dtype=torch.float32) * v_pred[i].float()  # [C,...]
            for i in range(len(x_t))
        ]  # list of [C,...]

    @staticmethod
    def _generated_vision_item_indices(gen_data_clean: GenerationDataClean) -> list[int]:
        """Return the flat vision-item index of each logical sample's generated target."""
        vision_items = gen_data_clean.x0_tokens_vision
        if vision_items is None:
            raise ValueError("Self-forcing DMD requires vision tokens.")
        num_items = len(vision_items)
        item_counts = getattr(gen_data_clean, "num_vision_items_per_sample", None)
        if item_counts is None:
            if num_items != gen_data_clean.batch_size:
                raise ValueError(
                    "Single-item vision data must have one flat item per logical sample, "
                    f"got batch_size={gen_data_clean.batch_size} and {num_items} items."
                )
        elif len(item_counts) != gen_data_clean.batch_size or sum(item_counts) != num_items:
            raise ValueError(
                "num_vision_items_per_sample must align with the logical batch and flat vision items, "
                f"got batch_size={gen_data_clean.batch_size}, counts={item_counts}, and {num_items} items."
            )

        return generated_vision_item_indices(item_counts, num_items)

    @staticmethod
    def _select_vision_items(items: list[Any], indices: list[int]) -> list[Any]:
        """Select flat vision items without changing their order."""
        return [items[index] for index in indices]

    @staticmethod
    def _expand_vision_schedule(schedule: torch.Tensor, gen_data_clean: GenerationDataClean) -> torch.Tensor:
        """Expand a per-logical-sample timestep or sigma tensor to flat vision items."""
        return _expand_per_sample_to_per_vision_item(
            schedule,
            getattr(gen_data_clean, "num_vision_items_per_sample", None),
        )  # [N_vision_items,...]

    @classmethod
    def _num_vision_tokens_per_sample(cls, gen_data_clean: GenerationDataClean) -> list[int]:
        """Return generated-target token counts, one per logical sample."""
        vision_items = gen_data_clean.x0_tokens_vision
        if vision_items is None:
            raise ValueError("Self-forcing DMD requires vision tokens.")
        generated_indices = cls._generated_vision_item_indices(gen_data_clean)
        return [vision_token_count(x0) for x0 in cls._select_vision_items(vision_items, generated_indices)]

    def _sample_critic_noise_level(
        self,
        gen_data_clean: GenerationDataClean,
        resolutions: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        vision_items = gen_data_clean.x0_tokens_vision
        if vision_items is None:
            raise ValueError("Self-forcing DMD requires vision tokens.")
        generated_indices = self._generated_vision_item_indices(gen_data_clean)
        generated_items = self._select_vision_items(vision_items, generated_indices)
        num_vision_latent_frames = [x.shape[2] if x.ndim == 5 else x.shape[1] for x in generated_items]  # [B]
        timesteps, sigmas = self._get_train_noise_level_vision(
            batch_size=gen_data_clean.batch_size,
            is_image_batch=gen_data_clean.is_image_batch,
            num_vision_latent_frames=num_vision_latent_frames,
            resolutions=resolutions,
            num_tokens=self._num_vision_tokens_per_sample(gen_data_clean),
        )  # [B,T_vis], [B,T_vis]
        # CP invariant: any RNG feeding a packed sequence must be identical across CP
        # ranks — sequence shards are mixed by all-to-all, so divergence corrupts
        # activations, not just samples.
        timesteps, sigmas = timesteps.contiguous(), sigmas.contiguous()
        context_parallel_broadcast_tensor_list([timesteps, sigmas], getattr(self, "parallel_dims", None))
        return timesteps, sigmas

    def _get_data_resolutions(self, data_batch: dict[str, Any], batch_size: int) -> list[str]:
        """Return per-sample resolution strings for RF shift lookup."""
        return get_data_resolutions(data_batch, batch_size, self.config.resolution)

    @staticmethod
    def _action_sample_indices(sequence_plans: list[SequencePlan]) -> list[int]:
        return [i for i, plan in enumerate(sequence_plans) if plan.has_action]

    def _teacher_guidance_for_batch(self, sequence_plans: list[SequencePlan]) -> float:
        """Return the teacher CFG scale for the current batch."""
        if self.config.action_conditioned_teacher_guidance is not None and any(
            plan.has_action for plan in sequence_plans
        ):
            return float(self.config.action_conditioned_teacher_guidance)
        return float(self.config.teacher_guidance)

    @staticmethod
    def _packed_action_has_noisy_tokens(packed_sequence: PackedSequence) -> bool:
        return packed_sequence.action is not None and any(
            nfi.numel() > 0 for nfi in packed_sequence.action.noisy_frame_indexes
        )

    @staticmethod
    def _sum_optional_preds(output: dict[str, Any], key: str, reference: torch.Tensor) -> torch.Tensor:
        """Sum optional denoiser predictions for zero-valued graph edges."""
        preds = output.get(key)
        if preds is None:
            return reference.new_zeros(())  # []
        total = reference.new_zeros(())  # []
        for pred in preds:
            total = total + pred.sum()  # []
        return total  # []

    def _sample_critic_action_noise_level(
        self,
        sequence_plans: list[SequencePlan],
        packed_sequence: PackedSequence,
        batch_size: int,
        timesteps_vision: torch.Tensor,
        sigmas_vision: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if packed_sequence.action is None:
            return None, None
        action_sample_indices = self._action_sample_indices(sequence_plans)  # [n_action]
        if not action_sample_indices or not self._packed_action_has_noisy_tokens(packed_sequence):
            return None, None

        if not self.config.rectified_flow_training_config.independent_action_schedule:
            idx = torch.tensor(action_sample_indices, dtype=torch.long, device=timesteps_vision.device)  # [n_action]
            return timesteps_vision[idx], sigmas_vision[idx]  # [n_action,T_vis] each

        timesteps_full, sigmas_full = self._get_train_noise_level_action(batch_size=batch_size)  # [B,1] each
        timesteps_full, sigmas_full = timesteps_full.contiguous(), sigmas_full.contiguous()
        context_parallel_broadcast_tensor_list([timesteps_full, sigmas_full], getattr(self, "parallel_dims", None))
        idx = torch.tensor(action_sample_indices, dtype=torch.long, device=timesteps_full.device)  # [n_action]
        return timesteps_full[idx], sigmas_full[idx]  # [n_action,1] each

    @staticmethod
    def _overwrite_action_timesteps(
        packed_sequence: PackedSequence,
        timesteps_action: torch.Tensor | None,
    ) -> None:
        if timesteps_action is None or packed_sequence.action is None:
            return
        if not SelfForcingDMDModel._packed_action_has_noisy_tokens(packed_sequence):
            return

        sample_ts = timesteps_action.detach().cpu()  # [n_action,T_action]
        packed_sequence.action.timesteps = torch.cat(
            [
                (
                    sample_ts[i].flatten()[: nfi.numel()]
                    if sample_ts[i].numel() > 1
                    else sample_ts[i].flatten()[0:1].expand(nfi.numel())
                )
                for i, nfi in enumerate(packed_sequence.action.noisy_frame_indexes)
            ]
        ).to(dtype=torch.float32)  # [N_action_noisy]

    def _pack_and_denoise(
        self,
        gen_data_clean: GenerationDataClean,
        gen_data_noised: GenerationDataNoised,
        timesteps: torch.Tensor,
        input_text_indexes: list[list[int]],
        sequence_plans: list[SequencePlan],
        net_type: str,
        timesteps_action: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | object]:
        # As in TF-dCM, build the layout first, then install the live noisy
        # payload below. Vision packing only inspects shape/dtype/device, so
        # scalar-backed CPU views avoid downloading complete GPU latent clips
        # while keeping mask and position construction on the CPU.
        vision_layout_cpu = [
            torch.empty((), dtype=vt.dtype, device="cpu").expand(vt.shape) for vt in gen_data_noised.xt_tokens_vision
        ]  # list[[1,C,T,H,W]], each backed by a single CPU scalar
        noised_sound_cpu = (
            [st.cpu() for st in gen_data_noised.xt_tokens_sound]
            if gen_data_noised.xt_tokens_sound is not None
            else None
        )  # list[[C,T,H,W]] or None
        noised_action_cpu = (
            [at.cpu() for at in gen_data_noised.xt_tokens_action]
            if gen_data_noised.xt_tokens_action is not None
            else None
        )  # list[[T_action,D]] or None
        gen_data_for_packing = replace(
            gen_data_clean,
            x0_tokens_vision=vision_layout_cpu,
            x0_tokens_sound=noised_sound_cpu,
            x0_tokens_action=noised_action_cpu,
        )

        # The DMD teacher/fake-score nets are constructed with ``video_temporal_causal=False``.
        # Pack their scoring inputs the same way; the causal student rollout uses AR packs elsewhere.
        saved_video_temporal_causal = self.config.video_temporal_causal
        if net_type in {"teacher", "fake_score"}:
            self.config.video_temporal_causal = False
        try:
            packed_sequence = self._pack_input_sequence(
                sequence_plans, input_text_indexes, gen_data_for_packing, timesteps.cpu()
            )
        finally:
            self.config.video_temporal_causal = saved_video_temporal_causal
        self._overwrite_action_timesteps(packed_sequence, timesteps_action)

        assert packed_sequence.vision is not None, "Packed vision data is required"
        packed_sequence.vision.tokens = gen_data_noised.xt_tokens_vision
        if packed_sequence.action is not None and gen_data_noised.xt_tokens_action is not None:
            action_all_conditioning = all(
                torch.all(condition_mask == 1).item() for condition_mask in packed_sequence.action.condition_mask
            )
            if not action_all_conditioning:
                packed_sequence.action.tokens = gen_data_noised.xt_tokens_action
        if packed_sequence.sound is not None and gen_data_noised.xt_tokens_sound is not None:
            packed_sequence.sound.tokens = gen_data_noised.xt_tokens_sound

        packed_sequence.to_cuda()
        return self.denoise(
            net=self.denoiser_nets[net_type],
            data_batch_packed=packed_sequence,
        )

    def _make_teacher_negative_captions(self, batch_size: int, data_batch: dict[str, Any]) -> list[str]:
        """Format negative captions from the owner's pixels before CP data-window rotation."""
        negative_prompt = self.config.teacher_negative_prompt
        if not isinstance(negative_prompt, str):
            raise TypeError("SF-DMD teacher_negative_prompt must be a string.")
        if not self.config.teacher_negative_prompt_keep_metadata:
            return [negative_prompt] * batch_size
        if "enable_per_camera_vae_encoding" in data_batch:
            raise ValueError("SF-DMD negative metadata currently supports single-view video batches only.")

        # get_data_and_condition has flattened each sample's [control, target]
        # pixels. The final item belongs to the target; image_size is flattened
        # too, so indexing that field by logical sample would mix sample owners.
        videos = data_batch.get(self.input_video_key)
        item_counts = data_batch.get("num_vision_items_per_sample")
        if item_counts is None:
            item_counts = [1] * batch_size
        if (
            not isinstance(videos, (list, tuple))
            or not isinstance(item_counts, (list, tuple))
            or len(item_counts) != batch_size
            or any(isinstance(count, bool) or not isinstance(count, int) or count < 1 for count in item_counts)
            or sum(item_counts) != len(videos)
        ):
            raise ValueError("SF-DMD negative metadata requires aligned video items and counts for each sample.")

        # conditioning_fps includes the sampler's frame stride; the source fps
        # may describe a different rate. Both collated tensors and joint-loader
        # lists of singleton tensors preserve one effective FPS per sample.
        fps_values = data_batch.get("conditioning_fps")
        if isinstance(fps_values, torch.Tensor):
            fps_values = fps_values.reshape(-1).tolist()  # [B] -> list[float]
        if not isinstance(fps_values, (list, tuple)) or len(fps_values) != batch_size:
            raise ValueError("SF-DMD negative metadata requires one conditioning_fps value per sample.")

        captions: list[str] = []
        item_offset = 0
        for sample_idx, item_count in enumerate(item_counts):
            target = videos[item_offset + item_count - 1]  # [1,C,T,H,W]
            item_offset += item_count
            if not isinstance(target, torch.Tensor) or target.ndim != 5 or target.shape[0] != 1:
                raise ValueError("SF-DMD negative metadata requires target pixels with shape [1,C,T,H,W].")
            num_frames, height, width = target.shape[-3:]
            if min(target.shape[1:]) < 1:
                raise ValueError("SF-DMD negative metadata requires nonempty target pixels.")
            fps_value = fps_values[sample_idx]
            if isinstance(fps_value, torch.Tensor):
                if fps_value.numel() != 1:
                    raise ValueError("SF-DMD negative metadata requires one conditioning_fps value per sample.")
                fps_value = fps_value.item()
            if isinstance(fps_value, bool):
                raise ValueError("SF-DMD negative metadata requires numeric conditioning_fps, not a boolean.")
            try:
                fps = float(fps_value)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError("SF-DMD negative metadata requires numeric conditioning_fps.") from error
            if not math.isfinite(fps) or fps <= 0 or round(fps) < 1:
                raise ValueError(f"SF-DMD negative metadata requires positive finite conditioning_fps; got {fps}.")
            # Match the positive caption's duration using the sampled FPS;
            # round only the displayed FPS after fractional frame resampling.
            # Use the full sampled pixel clip, not the latent generation chunk.
            metadata_fps = int(round(fps))
            caption = negative_prompt.strip()
            if num_frames > 1:
                duration_text = DEFAULT_DURATION_FPS_TEMPLATE.format(duration=num_frames / fps, fps=metadata_fps)
                caption = caption.rstrip(".") + ". " + duration_text
            resolution_text = DEFAULT_VIDEO_TEMPLATE.format(height=height, width=width)
            caption = caption.strip().rstrip(".") + ". " + resolution_text
            captions.append(caption.lstrip(".").strip())
        return captions

    def _make_unconditional_text_indexes(
        self,
        batch_size: int,
        data_batch: dict[str, Any],
        *,
        is_image_batch: bool,
    ) -> list[list[int]]:
        """Tokenize the teacher's negative caption with caption dropout's system prompts."""
        system_prompt = resolve_text_system_prompt(data_batch)
        if system_prompt is None or isinstance(system_prompt, str):
            system_prompts = [system_prompt] * batch_size
        elif isinstance(system_prompt, (list, tuple)):
            if len(system_prompt) != batch_size:
                raise ValueError(
                    f"SF-DMD requires one system prompt per sample; got {len(system_prompt)} for {batch_size} samples."
                )
            system_prompts = list(system_prompt)
        else:
            raise TypeError(f"SF-DMD system prompts must be strings, got {type(system_prompt).__name__}.")
        if any(prompt is not None and not isinstance(prompt, str) for prompt in system_prompts):
            raise TypeError("SF-DMD system prompts must contain only strings or None.")
        negative_captions = self._make_teacher_negative_captions(batch_size, data_batch)
        return [
            tokenize_caption(
                negative_caption,
                self.vlm_tokenizer,
                is_video=not is_image_batch,
                use_system_prompt=self.vlm_config.use_system_prompt,
                system_prompt=prompt,
            )
            for negative_caption, prompt in zip(negative_captions, system_prompts, strict=True)
        ]

    def memory_init_training(
        self,
        gen_data_clean: GenerationDataClean,
        data_batch: dict[str, Any],
        input_text_indexes: list[list[int]],
    ) -> tuple[GenerationDataClean, dict]:
        """Keep teacher CFG tokens paired with the sample through CP data-window rotation."""
        # Capture metadata from the sampled clip before chunk alignment can
        # shorten raw_state_vision, which aliases the batch's pixel list.
        unconditional_text_indexes = None
        if self.config.teacher_guidance > 1.0 or (self.config.action_conditioned_teacher_guidance or 0.0) > 1.0:
            unconditional_text_indexes = self._make_unconditional_text_indexes(
                gen_data_clean.batch_size, data_batch, is_image_batch=gen_data_clean.is_image_batch
            )
        gen_data_clean, memory_info = super().memory_init_training(gen_data_clean, data_batch, input_text_indexes)
        if unconditional_text_indexes is not None:
            memory_info["_sf_dmd_unconditional_text_indexes"] = unconditional_text_indexes
        return gen_data_clean, memory_info

    def _prepare_self_forcing_inputs(
        self,
        data_batch: dict[str, Any],
        iteration: int,
    ) -> tuple[list[list[int]], list[SequencePlan], GenerationDataClean, list[str], list[list[int]] | None]:
        """Return the CP-synchronized tokenized inputs for one self-forcing step.

        Under context parallelism the AR rollout and its Ulysses collectives must run
        the same sequence on every CP peer, but each rank draws an independent batch
        whose clip length and latent spatial shape differ. Delegating to
        :meth:`_get_training_inputs` reuses the base CP data-window rotation, so one
        owner rank's tokenized payload is broadcast to the whole CP group. Without CP
        it is the plain rank-local path.
        """
        input_text_indexes, sequence_plans, gen_data_clean, memory_info, data_resolutions, _ = (
            self._get_training_inputs(data_batch, iteration)
        )
        sequence_plans = self._sync_transfer_conditioning_plans(sequence_plans, gen_data_clean)
        if data_resolutions is None:
            data_resolutions = [self.config.resolution] * gen_data_clean.batch_size
        if self.config.sf_rollout_num_pixel_frames is not None:
            if gen_data_clean.x0_tokens_vision is None or len(gen_data_clean.x0_tokens_vision) != 1:
                raise ValueError("Fixed-horizon SF-DMD requires exactly one vision item")
            vision_tokens = gen_data_clean.x0_tokens_vision[0]
            temporal_compression_factor = self.tokenizer_vision_gen.temporal_compression_factor
            if temporal_compression_factor is None:
                raise ValueError("Fixed-horizon SF-DMD requires a temporal compression factor")
            if temporal_compression_factor < 1:
                raise ValueError("Fixed-horizon SF-DMD requires a positive temporal compression factor")
            if (self.config.sf_rollout_num_pixel_frames - 1) % temporal_compression_factor != 0:
                raise ValueError(
                    "sf_rollout_num_pixel_frames must align with the tokenizer temporal compression factor"
                )
            latent_frames = (self.config.sf_rollout_num_pixel_frames - 1) // temporal_compression_factor + 1
            if vision_tokens.shape[2] != 1:
                raise ValueError(
                    "Fixed-horizon SF-DMD expects a compact first-frame source, "
                    f"got {vision_tokens.shape[2]} latent frames"
                )
            # Only frame zero is conditioned; these are placeholder target slots
            # overwritten by self-forcing before their clean values can be consumed.
            gen_data_clean.x0_tokens_vision[0] = vision_tokens.expand(-1, -1, latent_frames, -1, -1)
        # Expand the CP owner's cached negative tokens over its packed caption slots.
        # Tokenization captured the full clip metadata before memory initialization.
        unconditional_text_indexes = memory_info.get("_sf_dmd_unconditional_text_indexes")
        if unconditional_text_indexes is not None:
            unconditional_text_indexes = [
                list(tokens)
                for plan, tokens in zip(sequence_plans, unconditional_text_indexes, strict=True)
                for _ in range(len(plan.text_view_ids) if plan.text_view_ids is not None else 1)
            ]
        return input_text_indexes, sequence_plans, gen_data_clean, data_resolutions, unconditional_text_indexes

    def training_step(self, data_batch: dict[str, Any], iteration: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        self._setup_grad_requirements(iteration)
        if self.config.sf_streaming_long_tuning:
            if self.get_phase(iteration) == "student":
                return self.training_step_generator_streaming(data_batch, iteration)
            return self.training_step_critic_streaming(data_batch, iteration)

        input_text_indexes, sequence_plans, gen_data_clean, data_resolutions, unconditional_text_indexes = (
            self._prepare_self_forcing_inputs(data_batch, iteration)
        )

        if self.get_phase(iteration) == "student":
            return self.training_step_generator(
                input_text_indexes,
                sequence_plans,
                gen_data_clean,
                data_resolutions,
                iteration,
                unconditional_text_indexes=unconditional_text_indexes,
            )
        return self.training_step_critic(
            input_text_indexes, sequence_plans, gen_data_clean, data_resolutions, iteration
        )

    @staticmethod
    def _streaming_step_metrics(step: _StreamingLongStep) -> dict[str, torch.Tensor]:
        """Return scalar temporal diagnostics for one streaming update."""
        tensor_kwargs = {"device": step.supervision_mask.device, "dtype": torch.float32}
        return {
            "slt_rollout_start": torch.tensor(step.plan.rollout_start, **tensor_kwargs),
            "slt_rollout_end": torch.tensor(step.plan.rollout_end, **tensor_kwargs),
            "slt_window_frames": torch.tensor(step.plan.window_frames, **tensor_kwargs),
            "slt_overlap_frames": torch.tensor(step.plan.overlap_frames, **tensor_kwargs),
            "slt_supervised_frames": torch.tensor(step.plan.supervised_frames, **tensor_kwargs),
            "slt_is_final": torch.tensor(float(step.plan.is_final), **tensor_kwargs),
        }

    def _score_student_samples(
        self,
        gen_data_student: GenerationDataClean,
        gen_data_noised_critic: GenerationDataNoised,
        timesteps_critic: torch.Tensor,  # [B,T_vis]
        input_text_indexes: list[list[int]],
        sequence_plans: list[SequencePlan],
        teacher_guidance: float,
        unconditional_text_indexes: list[list[int]] | None,
        timesteps_action: torch.Tensor | None = None,  # [n_action,T_action] or None
    ) -> tuple[
        list[torch.Tensor],  # teacher endpoints, each [C,...]
        list[torch.Tensor],  # fake-score endpoints, each [C,...]
        dict[str, Any],
        dict[str, Any],
        dict[str, Any] | None,
    ]:
        """Score one noisy rollout, retaining raw action predictions for the caller's loss."""
        with torch.no_grad():
            out_fake = self._pack_and_denoise(
                gen_data_student,
                gen_data_noised_critic,
                timesteps_critic,
                input_text_indexes,
                sequence_plans,
                net_type="fake_score",
                timesteps_action=timesteps_action,
            )
            fake_x0 = self._velocity_to_x0(
                gen_data_noised_critic.xt_tokens_vision,
                out_fake["preds_vision"],  # type: ignore[arg-type]
                gen_data_noised_critic.sigmas_vision,  # type: ignore[arg-type]
            )  # list of [C,...]

        with (
            torch.no_grad(),
            defer_fsdp_reshard(
                self.net_teacher,
                after_forward=self.config.sf_defer_teacher_fsdp_reshard,
                reshard_root_on_exit=self.config.sf_defer_teacher_fsdp_reshard,
            ),
        ):
            out_teacher = self._pack_and_denoise(
                gen_data_student,
                gen_data_noised_critic,
                timesteps_critic,
                input_text_indexes,
                sequence_plans,
                net_type="teacher",
                timesteps_action=timesteps_action,
            )
            teacher_x0 = self._velocity_to_x0(
                gen_data_noised_critic.xt_tokens_vision,
                out_teacher["preds_vision"],  # type: ignore[arg-type]
                gen_data_noised_critic.sigmas_vision,  # type: ignore[arg-type]
            )  # list of [C,...]
            if teacher_guidance > 1.0:
                if unconditional_text_indexes is None:
                    raise ValueError("Teacher CFG requires unconditional text prepared with the training sample.")
                out_teacher_uncond = self._pack_and_denoise(
                    gen_data_student,
                    gen_data_noised_critic,
                    timesteps_critic,
                    unconditional_text_indexes,
                    sequence_plans,
                    net_type="teacher",
                    timesteps_action=timesteps_action,
                )
                teacher_x0_uncond = self._velocity_to_x0(
                    gen_data_noised_critic.xt_tokens_vision,
                    out_teacher_uncond["preds_vision"],  # type: ignore[arg-type]
                    gen_data_noised_critic.sigmas_vision,  # type: ignore[arg-type]
                )  # list of [C,...]
                teacher_x0 = [
                    teacher_x0[i] + (teacher_guidance - 1.0) * (teacher_x0[i] - teacher_x0_uncond[i])
                    for i in range(len(teacher_x0))
                ]  # list of [C,...]
            else:
                out_teacher_uncond = None

        return teacher_x0, fake_x0, out_fake, out_teacher, out_teacher_uncond

    def training_step_generator_streaming(
        self,
        data_batch: dict[str, Any],
        iteration: int,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Advance one bounded rollout window and run its student DMD update."""
        step = self._streaming_step_for_phase(data_batch, iteration, student_phase=True)
        gen_data_student = step.gen_data_student
        batch_size = gen_data_student.batch_size
        data_resolutions = self._get_data_resolutions(data_batch, batch_size)
        timesteps_critic, sigmas_critic = self._sample_critic_noise_level(gen_data_student, data_resolutions)
        sigmas_critic_vision = self._expand_vision_schedule(sigmas_critic, gen_data_student)  # [B_items,T_vis]
        gen_data_noised_critic = self._add_noise_to_input(
            gen_data_student,
            step.packed_reference,
            sigmas_critic_vision,
        )
        teacher_guidance = self._teacher_guidance_for_batch(step.sequence_plans)

        teacher_x0, fake_x0, _, _, _ = self._score_student_samples(
            gen_data_student,
            gen_data_noised_critic,
            timesteps_critic,
            step.input_text_indexes,
            step.sequence_plans,
            teacher_guidance,
            step.unconditional_text_indexes,
        )

        assert gen_data_student.x0_tokens_vision is not None
        loss_mask = [step.supervision_mask]
        vsd_loss = self._vsd_frame_sum_loss(
            gen_data_student.x0_tokens_vision,
            teacher_x0,
            fake_x0,
            loss_mask=loss_mask,
        )
        total_loss = self.config.loss_scale_sid * vsd_loss
        log.info(
            f"Iteration {iteration}, SLT student phase, rollout=[{step.plan.rollout_start},"
            f"{step.plan.rollout_end}), vsd_loss={vsd_loss.item():.6f}"
        )
        output_batch: dict[str, torch.Tensor] = {
            "vsd_loss": vsd_loss.detach(),
            "total_generator_loss": total_loss.detach(),
            "sigma": sigmas_critic.detach(),
            "teacher_guidance": torch.tensor(teacher_guidance, **self.tensor_kwargs_fp32),
            "flow_matching_loss_vision_per_instance": vsd_loss.detach().expand(batch_size),
            "dmd_loss_generator": total_loss.detach(),
            "dmd_loss": vsd_loss.detach(),
            **self._streaming_step_metrics(step),
        }
        return output_batch, total_loss

    def training_step_critic_streaming(
        self,
        data_batch: dict[str, Any],
        iteration: int,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Reuse the latest student rollout window for a fake-score update."""
        with torch.no_grad():
            step = self._streaming_step_for_phase(data_batch, iteration, student_phase=False)
            gen_data_student = step.gen_data_student
            data_resolutions = self._get_data_resolutions(data_batch, gen_data_student.batch_size)
            timesteps_critic, sigmas_critic = self._sample_critic_noise_level(gen_data_student, data_resolutions)
            sigmas_critic_vision = self._expand_vision_schedule(sigmas_critic, gen_data_student)  # [B_items,T_vis]
            gen_data_noised_critic = self._add_noise_to_input(
                gen_data_student,
                step.packed_reference,
                sigmas_critic_vision,
            )

        out_fake = self._pack_and_denoise(
            gen_data_student,
            gen_data_noised_critic,
            timesteps_critic,
            step.input_text_indexes,
            step.sequence_plans,
            net_type="fake_score",
        )
        condition_mask = [1.0 - step.supervision_mask]
        timesteps_critic_vision = self._expand_vision_schedule(timesteps_critic, gen_data_student)  # [B_items,T_vis]
        _, loss_per_instance = self._compute_flow_matching_loss(
            pred=out_fake["preds_vision"],  # type: ignore[arg-type]
            target=gen_data_noised_critic.vt_target_vision,  # type: ignore[arg-type]
            condition_mask=condition_mask,
            timesteps=timesteps_critic_vision,
            has_valid_tokens=True,
            rectified_flow=self.rectified_flow_video,
            normalize_by_active=True,
        )
        loss_fake_score = loss_per_instance.mean() * self.config.loss_scale_fake_score
        total_loss = loss_fake_score
        log.info(
            f"Iteration {iteration}, SLT critic phase, rollout=[{step.plan.rollout_start},"
            f"{step.plan.rollout_end}), fake_score_loss={loss_fake_score.item():.6f}"
        )
        output_batch: dict[str, torch.Tensor] = {
            "loss_fake_score": loss_fake_score.detach(),
            "total_critic_loss": total_loss.detach(),
            "sigma": sigmas_critic.detach(),
            "flow_matching_loss_vision_per_instance": loss_per_instance.detach(),
            "dmd_loss_critic": total_loss.detach(),
            "dmd_loss": loss_fake_score.detach(),
            **self._streaming_step_metrics(step),
        }
        return output_batch, total_loss

    def training_step_closures(
        self,
        data_batch: dict[str, Any],
        iteration: int,
    ) -> Iterator[tuple[str, Callable[[], tuple[dict[str, torch.Tensor], torch.Tensor]], bool]]:
        """Yield one or more closures for a training iteration."""
        if self.get_phase(iteration) != "student" or not self.config.sf_replayed_training:
            yield "sf_dmd", lambda: self.training_step(data_batch, iteration), True
            return

        self._setup_grad_requirements(iteration)
        input_text_indexes, sequence_plans, gen_data_clean, data_resolutions, unconditional_text_indexes = (
            self._prepare_self_forcing_inputs(data_batch, iteration)
        )
        ctx = self._make_replayed_sf_ctx(
            input_text_indexes,
            sequence_plans,
            gen_data_clean,
            data_resolutions,
            iteration,
            unconditional_text_indexes=unconditional_text_indexes,
        )
        if ctx.sim.replay_state is None:
            raise RuntimeError("Replayed SF-DMD requires a backend replay state.")
        last_index = len(ctx.sim.replay_state.frames) - 1
        for replay_index in range(len(ctx.sim.replay_state.frames)):
            yield (
                f"sf_dmd_replayed_{replay_index}",
                lambda replay_index=replay_index: self._replayed_sf_step(replay_index, ctx),
                replay_index == last_index,
            )

    def training_step_generator(
        self,
        input_text_indexes: list[list[int]],
        sequence_plans: list[SequencePlan],
        gen_data_clean: GenerationDataClean,
        data_resolutions: list[str],
        iteration: int,
        *,
        unconditional_text_indexes: list[list[int]] | None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        B = gen_data_clean.batch_size
        sim = self._self_forcing_simulation(
            input_text_indexes,
            sequence_plans,
            gen_data_clean,
            effective_iteration=self.get_student_iteration(iteration),
        )
        gen_data_student = sim.gen_data_student
        packed_student = sim.packed_reference
        out_student = sim.out_student

        timesteps_critic, sigmas_critic = self._sample_critic_noise_level(
            gen_data_student, data_resolutions
        )  # [B,T_vis]
        timesteps_action, sigmas_action = self._sample_critic_action_noise_level(
            sequence_plans,
            packed_student,
            B,
            timesteps_critic,
            sigmas_critic,
        )  # [n_action,T_action] or None

        sigmas_critic_vision = self._expand_vision_schedule(sigmas_critic, gen_data_student)  # [B_items,T_vis]
        gen_data_noised_critic = self._add_noise_to_input(
            gen_data_student,
            packed_student,
            sigmas_critic_vision,
            sigmas_action=sigmas_action,
        )
        teacher_guidance = self._teacher_guidance_for_batch(sequence_plans)

        teacher_x0, fake_x0, out_fake, out_teacher, out_teacher_uncond = self._score_student_samples(
            gen_data_student,
            gen_data_noised_critic,
            timesteps_critic,
            input_text_indexes,
            sequence_plans,
            teacher_guidance,
            unconditional_text_indexes,
            timesteps_action=timesteps_action,
        )

        log.info(f"student phase, after pack_and_denoise teacher, teacher_x0: {teacher_x0[0].shape}")

        assert gen_data_student.x0_tokens_vision is not None
        assert packed_student.vision is not None
        generated_vision_item_indices = self._generated_vision_item_indices(gen_data_student)
        generated_x0_tokens_vision = self._select_vision_items(
            gen_data_student.x0_tokens_vision, generated_vision_item_indices
        )
        generated_teacher_x0 = self._select_vision_items(teacher_x0, generated_vision_item_indices)
        generated_fake_x0 = self._select_vision_items(fake_x0, generated_vision_item_indices)
        noisy_mask_vision = [
            (1.0 - cm).to(gen_data_student.x0_tokens_vision[0].dtype)
            for cm in self._select_vision_items(
                packed_student.vision.condition_mask,  # type: ignore[union-attr]
                generated_vision_item_indices,
            )
        ]  # list of [T_i,1,1]
        vsd_loss = self._vsd_frame_sum_loss(
            generated_x0_tokens_vision,
            generated_teacher_x0,
            generated_fake_x0,
            loss_mask=noisy_mask_vision,
        )
        total_loss = self.config.loss_scale_sid * vsd_loss

        if self.config.action_gen:
            if gen_data_student.x0_tokens_action is not None and self._packed_action_has_noisy_tokens(packed_student):
                fake_x0_action = self._velocity_to_x0(
                    gen_data_noised_critic.xt_tokens_action,  # type: ignore[arg-type]
                    out_fake["preds_action"],  # type: ignore[arg-type]
                    gen_data_noised_critic.sigmas_action,  # type: ignore[arg-type]
                )
                teacher_x0_action = self._velocity_to_x0(
                    gen_data_noised_critic.xt_tokens_action,  # type: ignore[arg-type]
                    out_teacher["preds_action"],  # type: ignore[arg-type]
                    gen_data_noised_critic.sigmas_action,  # type: ignore[arg-type]
                )
                if out_teacher_uncond is not None:
                    teacher_x0_action_uncond = self._velocity_to_x0(
                        gen_data_noised_critic.xt_tokens_action,  # type: ignore[arg-type]
                        out_teacher_uncond["preds_action"],  # type: ignore[arg-type]
                        gen_data_noised_critic.sigmas_action,  # type: ignore[arg-type]
                    )
                    teacher_x0_action = [
                        teacher_x0_action[i]
                        + (teacher_guidance - 1.0) * (teacher_x0_action[i] - teacher_x0_action_uncond[i])
                        for i in range(len(teacher_x0_action))
                    ]
                noisy_mask_action = [
                    (1.0 - cm).to(gen_data_student.x0_tokens_action[0].dtype)
                    for cm in packed_student.action.condition_mask  # type: ignore[union-attr]
                ]  # list of [T_i,1]
                total_loss += self.config.loss_scale_sid * variational_score_distillation_loss(
                    gen_data_student.x0_tokens_action,
                    teacher_x0_action,
                    fake_x0_action,
                    loss_mask=noisy_mask_action,
                )
            else:
                total_loss = (
                    total_loss
                    + 0.0 * self._sum_optional_preds(out_student, "preds_action", total_loss)
                    + 0.0 * self._sum_optional_preds(out_fake, "preds_action", total_loss)
                    + 0.0 * self._sum_optional_preds(out_teacher, "preds_action", total_loss)
                )

        log.info(
            f"Iteration {iteration}, student phase, after vsd_loss, "
            f"vsd loss: {vsd_loss.item():.6f}, total loss: {total_loss.item():.6f}"
        )
        output_batch: dict[str, torch.Tensor] = {
            "vsd_loss": vsd_loss.detach(),
            "total_generator_loss": total_loss.detach(),
            "sigma": sigmas_critic.detach(),
            "teacher_guidance": torch.tensor(teacher_guidance, **self.tensor_kwargs_fp32),  # []
            "flow_matching_loss_vision_per_instance": vsd_loss.detach().expand(B),
            "dmd_loss_generator": total_loss.detach(),
            "dmd_loss": vsd_loss.detach(),
        }
        return output_batch, total_loss

    @staticmethod
    def _make_vsd_pseudo_target(
        gen_data: list[torch.Tensor],
        teacher_x0: list[torch.Tensor],
        fake_score_x0: list[torch.Tensor],
        loss_mask: list[torch.Tensor] | None,
    ) -> list[torch.Tensor]:
        """Return the detached pseudo-target that induces the VSD gradient."""
        with torch.no_grad():
            return [
                make_vsd_pseudo_target(
                    gen_data[idx],
                    fake_score_x0[idx] - teacher_x0[idx],  # [C,...]
                    teacher_x0[idx],
                    loss_mask=loss_mask[idx] if loss_mask is not None else None,
                )  # [C,...], detached
                for idx in range(len(gen_data))
            ]

    @staticmethod
    def _vsd_frame_sum_loss(
        gen_data: list[torch.Tensor],
        teacher_x0: list[torch.Tensor],
        fake_score_x0: list[torch.Tensor],
        loss_mask: list[torch.Tensor] | None,
    ) -> torch.Tensor:
        """Return VSD loss as a sum of per-frame spatial means."""
        targets = SelfForcingDMDModel._make_vsd_pseudo_target(gen_data, teacher_x0, fake_score_x0, loss_mask)
        per_sample_losses = []
        for idx, gen_i in enumerate(gen_data):
            if gen_i.ndim not in (4, 5):
                raise ValueError(f"Expected vision data with 4 or 5 dims, got shape {tuple(gen_i.shape)}")
            target_i = targets[idx].to(device=gen_i.device, dtype=gen_i.dtype)  # [1,C,T,H,W] or [C,T,H,W]
            frame_error = 0.5 * (gen_i - target_i).square()  # [1,C,T,H,W] or [C,T,H,W]
            temporal_dim = gen_i.ndim - 3
            reduce_dims = tuple(dim for dim in range(gen_i.ndim) if dim != temporal_dim)
            if loss_mask is not None:
                mask_i = loss_mask[idx].to(device=gen_i.device, dtype=gen_i.dtype).expand_as(gen_i)
                frame_numerators = (frame_error * mask_i).sum(dim=reduce_dims)  # [T]
                frame_denominators = mask_i.sum(dim=reduce_dims).clamp(min=1)  # [T]
                per_frame_loss = frame_numerators / frame_denominators  # [T]
            else:
                per_frame_loss = frame_error.mean(dim=reduce_dims)  # [T]
            per_sample_losses.append(per_frame_loss.sum())  # []
        return torch.stack(per_sample_losses).mean()  # []

    def _make_replayed_sf_ctx(
        self,
        input_text_indexes: list[list[int]],
        sequence_plans: list[SequencePlan],
        gen_data_clean: GenerationDataClean,
        data_resolutions: list[str],
        iteration: int,
        *,
        unconditional_text_indexes: list[list[int]] | None,
    ) -> _SelfForcingReplayContext:
        """Build detached replay context for rCM-style replayed SF-DMD."""
        if self.config.backward_grad_steps != 1:
            raise ValueError("Replayed SF-DMD currently requires backward_grad_steps=1.")
        if self.config.sf_replay_block_size != 1:
            raise ValueError("Replayed SF-DMD currently supports only sf_replay_block_size=1.")

        with torch.no_grad():
            sim = self._self_forcing_simulation(
                input_text_indexes,
                sequence_plans,
                gen_data_clean,
                effective_iteration=self.get_student_iteration(iteration),
                allow_grad=False,
                collect_replay=True,
            )
            if sim.replay_state is None or len(sim.replay_state.frames) == 0:
                raise ValueError("Replayed SF-DMD requires at least one generated replay frame.")
            gen_data_student = sim.gen_data_student
            packed_student = sim.packed_reference
            batch_size = gen_data_student.batch_size
            timesteps_critic, sigmas_critic = self._sample_critic_noise_level(
                gen_data_student, data_resolutions
            )  # [B,T_vis]
            timesteps_action, sigmas_action = self._sample_critic_action_noise_level(
                sequence_plans,
                packed_student,
                batch_size,
                timesteps_critic,
                sigmas_critic,
            )  # [n_action,T_action] or None
            sigmas_critic_vision = self._expand_vision_schedule(sigmas_critic, gen_data_student)  # [B_items,T_vis]
            gen_data_noised_critic = self._add_noise_to_input(
                gen_data_student,
                packed_student,
                sigmas_critic_vision,
                sigmas_action=sigmas_action,
            )
            teacher_guidance = self._teacher_guidance_for_batch(sequence_plans)
            teacher_x0, fake_x0, _, _, _ = self._score_student_samples(
                gen_data_student,
                gen_data_noised_critic,
                timesteps_critic,
                input_text_indexes,
                sequence_plans,
                teacher_guidance,
                unconditional_text_indexes,
                timesteps_action=timesteps_action,
            )

            assert gen_data_student.x0_tokens_vision is not None
            assert packed_student.vision is not None
            if gen_data_student.x0_tokens_action is not None and self._packed_action_has_noisy_tokens(packed_student):
                raise NotImplementedError("Replayed SF-DMD currently supports action tokens as conditioning only.")
            noisy_mask_vision = [
                (1.0 - cm).to(gen_data_student.x0_tokens_vision[0].dtype)
                for cm in self._select_vision_items(
                    packed_student.vision.condition_mask,  # type: ignore[union-attr]
                    self._generated_vision_item_indices(gen_data_student),
                )
            ]  # list of [T_i,1,1]
            generated_vision_item_indices = self._generated_vision_item_indices(gen_data_student)
            target_vision = self._make_vsd_pseudo_target(
                self._select_vision_items(gen_data_student.x0_tokens_vision, generated_vision_item_indices),
                self._select_vision_items(teacher_x0, generated_vision_item_indices),
                self._select_vision_items(fake_x0, generated_vision_item_indices),
                noisy_mask_vision,
            )[0]  # [1,C,T,H,W]
        return _SelfForcingReplayContext(
            sim=sim,
            target_vision=target_vision,
            iteration=iteration,
            sigmas_critic=sigmas_critic.detach(),
            teacher_guidance=teacher_guidance,
        )

    def _replayed_sf_step(
        self,
        replay_index: int,
        ctx: _SelfForcingReplayContext,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Replay one generated frame and backpropagate its DMD scoring loss."""
        replay_state = ctx.sim.replay_state
        if replay_state is None:
            raise RuntimeError("Replayed SF-DMD requires a backend replay state.")
        replay_frame = replay_state.frames[replay_index]
        timestep = replay_frame.sigma * float(self.config.rectified_flow_inference_config.num_train_timesteps)
        if isinstance(replay_frame, _MultiviewWSMReplayFrame):
            if not isinstance(replay_state, _MultiviewWSMReplayState):
                raise TypeError("Multiview replay frame requires multiview backend replay state.")
            backend = self._make_multiview_transfer_ar_backend()
            pack, memory = backend.build_replay_pack_and_memory(
                context=replay_frame.backend_context,
                vision_latent=replay_frame.xt.to(**self.tensor_kwargs),  # [1,C,V*chunk_len,H,W]
                chunk_start=replay_frame.frame_idx,
                timestep=timestep,
            )
        else:
            if not isinstance(replay_state, _CausalARReplayState):
                raise TypeError("Causal AR replay frame requires causal AR backend replay state.")
            action_latent = (
                replay_frame.action_latent.to(**self.tensor_kwargs) if replay_frame.action_latent is not None else None
            )  # [TCF,D] or None
            pack = self._make_ar_pack(
                vision_latent=replay_frame.xt.to(**self.tensor_kwargs),  # [1,C,chunk_len,H,W]
                action_latent=action_latent,
                text_tokens=replay_frame.text_tokens,
                timestep=timestep,
                frame_idx=replay_frame.frame_idx,
                cached_text_offset=replay_state.cached_text_offset,
                fps_vision_list=replay_state.fps_vision_list,
                fps_action_list=replay_state.fps_action_list,
                force_action_tokens=replay_state.force_action_tokens,
                action_domain_id=(
                    self._slice_action_domain_for_ar_unit(
                        OmniMoTCausalModel._first_action_domain_id(ctx.sim.gen_data_student),
                        action_latent,
                        replay_frame.frame_idx,
                        self.tokenizer_vision_gen.temporal_compression_factor or 4,
                    )
                    if replay_state.force_action_tokens
                    else None
                ),
                raw_action_dim=OmniMoTCausalModel._first_raw_action_dim(ctx.sim.gen_data_student),
                condition_frame_indexes_vision=list(replay_frame.condition_frame_indexes_vision),
            )
            self._cast_ar_action_tokens_to_projection_dtype(pack)
            pack.to_cuda()
            memory = self.build_memory_state(
                pack,
                replay_frame.cache_metadata.to_memory_info(replay_state.dual_kv_cache, write_gen_cache=False),
            )
        out_student = self.denoise(
            net=self.denoiser_nets["student"],
            data_batch_packed=pack,
            memory=memory,
        )
        velocity = self._single_vision_prediction(
            out_student["preds_vision"]
        ).float()  # [1,C,chunk_len,H,W] or [1,C,V*chunk_len,H,W]
        sigma_tensor = torch.as_tensor(replay_frame.sigma, dtype=torch.float32, device=velocity.device)  # []
        x0_pred = (
            replay_frame.xt.to(device=velocity.device, dtype=torch.float32) - sigma_tensor * velocity
        )  # [1,C,chunk_len,H,W] or [1,C,V*chunk_len,H,W]
        chunk_start = replay_frame.frame_idx
        chunk_end = chunk_start + replay_frame.chunk_len
        if isinstance(replay_frame, _MultiviewWSMReplayFrame):
            target_chunk = MultiviewTransferARBackend.slice_chunk(
                ctx.target_vision,
                num_views=replay_frame.backend_context.num_views,
                frames_per_view=replay_frame.backend_context.frames_per_view,
                chunk_start=chunk_start,
                chunk_end=chunk_end,
            ).to(device=x0_pred.device, dtype=x0_pred.dtype)  # [1,C,V*chunk_len,H,W]
        else:
            target_chunk = ctx.target_vision[:, :, chunk_start:chunk_end].to(
                device=x0_pred.device,
                dtype=x0_pred.dtype,
            )  # [1,C,chunk_len,H,W]
        # Match main and full-video VSD: mean channel/spatial error per frame, then sum every
        # generated camera-major frame without view-count normalization.
        per_camera_frame_loss = (0.5 * (x0_pred - target_chunk).square()).mean(
            dim=(0, 1, 3, 4)
        )  # [chunk_len] or [V*chunk_len]
        if isinstance(replay_frame, _CausalARReplayFrame) and replay_frame.condition_frame_indexes_vision:
            replay_loss_mask = torch.ones_like(per_camera_frame_loss)  # [chunk_len]
            replay_loss_mask[list(replay_frame.condition_frame_indexes_vision)] = 0.0
            per_camera_frame_loss = per_camera_frame_loss * replay_loss_mask  # [chunk_len]
        frame_loss = per_camera_frame_loss.sum()  # []
        total_loss = self.config.loss_scale_sid * frame_loss  # []
        batch_size = int(ctx.sim.sigma_student.shape[0])
        output_batch = {
            "vsd_loss": total_loss.detach(),
            "total_generator_loss": total_loss.detach(),
            "dmd_loss_generator": total_loss.detach(),
            "dmd_loss": total_loss.detach(),
            "flow_matching_loss_vision_per_instance": total_loss.detach().expand(batch_size),
        }
        if replay_index == 0 and ctx.sigmas_critic is not None:
            output_batch["sigma"] = ctx.sigmas_critic.detach()
            output_batch["teacher_guidance"] = torch.tensor(
                ctx.teacher_guidance,
                **self.tensor_kwargs_fp32,
            )  # []
        return output_batch, total_loss

    def training_step_critic(
        self,
        input_text_indexes: list[list[int]],
        sequence_plans: list[SequencePlan],
        gen_data_clean: GenerationDataClean,
        data_resolutions: list[str],
        iteration: int,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        B = gen_data_clean.batch_size
        with torch.no_grad():
            sim = self._self_forcing_simulation(
                input_text_indexes,
                sequence_plans,
                gen_data_clean,
                effective_iteration=self.get_fake_score_iteration(iteration),
            )
            gen_data_student = sim.gen_data_student
            packed_student = sim.packed_reference

        timesteps_critic, sigmas_critic = self._sample_critic_noise_level(
            gen_data_student, data_resolutions
        )  # [B,T_vis]
        timesteps_action, sigmas_action = self._sample_critic_action_noise_level(
            sequence_plans,
            packed_student,
            B,
            timesteps_critic,
            sigmas_critic,
        )  # [n_action,T_action] or None

        sigmas_critic_vision = self._expand_vision_schedule(sigmas_critic, gen_data_student)  # [B_items,T_vis]
        with torch.no_grad():
            gen_data_noised_critic = self._add_noise_to_input(
                gen_data_student,
                packed_student,
                sigmas_critic_vision,
                sigmas_action=sigmas_action,
            )

        out_fake = self._pack_and_denoise(
            gen_data_student,
            gen_data_noised_critic,
            timesteps_critic,
            input_text_indexes,
            sequence_plans,
            net_type="fake_score",
            timesteps_action=timesteps_action,
        )

        rectified_flow = self.rectified_flow_image if gen_data_student.is_image_batch else self.rectified_flow_video
        timesteps_critic_vision = self._expand_vision_schedule(timesteps_critic, gen_data_student)  # [B_items,T_vis]
        _, loss_per_instance = self._compute_flow_matching_loss(
            pred=out_fake["preds_vision"],  # type: ignore[arg-type]
            target=gen_data_noised_critic.vt_target_vision,  # type: ignore[arg-type]
            condition_mask=packed_student.vision.condition_mask,  # type: ignore[union-attr]
            timesteps=timesteps_critic_vision,
            has_valid_tokens=True,
            rectified_flow=rectified_flow,
            normalize_by_active=True,
        )
        generated_vision_item_indices = self._generated_vision_item_indices(gen_data_student)
        generated_loss_per_instance = loss_per_instance[generated_vision_item_indices]  # [B]
        loss_fake_score = generated_loss_per_instance.mean() * self.config.loss_scale_fake_score
        total_loss = loss_fake_score

        if (
            self.config.action_gen
            and gen_data_student.x0_tokens_action is not None
            and gen_data_noised_critic.xt_tokens_action is not None
            and packed_student.action is not None
            and self._packed_action_has_noisy_tokens(packed_student)
        ):
            _, action_per_instance = self._compute_flow_matching_loss(
                pred=out_fake["preds_action"],  # type: ignore[arg-type]
                target=gen_data_noised_critic.vt_target_action,  # type: ignore[arg-type]
                condition_mask=packed_student.action.condition_mask,  # type: ignore[union-attr]
                timesteps=timesteps_action if timesteps_action is not None else timesteps_critic,
                has_valid_tokens=True,
                rectified_flow=self.rectified_flow_action,
                raw_action_dim=gen_data_noised_critic.raw_action_dim,
                normalize_by_active=True,
            )
            total_loss += action_per_instance.mean() * self.config.loss_scale_fake_score

        log.info(
            f"Iteration {iteration}, critic phase, after flow_matching_loss, "
            f"fake_score_loss: {loss_fake_score.item():.6f}, total_loss: {total_loss.item():.6f}"
        )
        output_batch: dict[str, torch.Tensor] = {
            "loss_fake_score": loss_fake_score.detach(),
            "total_critic_loss": total_loss.detach(),
            "sigma": sigmas_critic.detach(),
            "flow_matching_loss_vision_per_instance": generated_loss_per_instance.detach(),
            "dmd_loss_critic": total_loss.detach(),
            "dmd_loss": loss_fake_score.detach(),
        }
        return output_batch, total_loss

    @staticmethod
    def _conditioned_vision_frames(sequence_plan: SequencePlan) -> set[int]:
        """Return latent-frame indexes that should be copied from clean conditioning."""
        return set(sequence_plan.condition_frame_indexes_vision or [])

    @staticmethod
    def _slice_vision_frame(vision_tokens: torch.Tensor, frame_idx: int) -> torch.Tensor:
        """Slice one latent frame as ``[1,C,1,H,W]``."""
        if vision_tokens.ndim == 5:
            return vision_tokens[:, :, frame_idx : frame_idx + 1]  # [1,C,1,H,W]
        if vision_tokens.ndim == 4:
            return vision_tokens[:, frame_idx : frame_idx + 1].unsqueeze(0)  # [1,C,1,H,W]
        raise ValueError(f"Expected vision tokens with 4 or 5 dims, got shape {tuple(vision_tokens.shape)}")

    @staticmethod
    def _slice_action_for_frame(
        action_tokens: torch.Tensor | None,
        frame_idx: int,
        temporal_compression_factor: int,
    ) -> torch.Tensor | None:
        """Slice action tokens controlling ``V{frame_idx - 1} -> V{frame_idx}``."""
        if action_tokens is None or frame_idx == 0:
            return None
        if action_tokens.ndim == 3:
            if action_tokens.shape[0] != 1:
                raise ValueError(f"Expected action batch size 1, got shape {tuple(action_tokens.shape)}")
            action_tokens = action_tokens.squeeze(0)  # [T_action,D]
        if action_tokens.ndim != 2:
            raise ValueError(f"Expected action tokens with 2 or 3 dims, got shape {tuple(action_tokens.shape)}")

        start = (frame_idx - 1) * temporal_compression_factor
        end = frame_idx * temporal_compression_factor
        if action_tokens.shape[0] < end:
            raise ValueError(
                f"Not enough action tokens for frame {frame_idx}: need {end}, got {action_tokens.shape[0]}"
            )
        return action_tokens[start:end]  # [TCF,D]

    @staticmethod
    def _slice_vision_chunk(vision_tokens: torch.Tensor, chunk_start: int, chunk_end: int) -> torch.Tensor:
        """Slice latent frames ``[chunk_start, chunk_end)`` as ``[1,C,chunk_len,H,W]``."""
        if vision_tokens.ndim == 5:
            return vision_tokens[:, :, chunk_start:chunk_end]  # [1,C,chunk_len,H,W]
        if vision_tokens.ndim == 4:
            return vision_tokens[:, chunk_start:chunk_end].unsqueeze(0)  # [1,C,chunk_len,H,W]
        raise ValueError(f"Expected vision tokens with 4 or 5 dims, got shape {tuple(vision_tokens.shape)}")

    @staticmethod
    def _multiview_wsm_shape(
        gen_data_clean: GenerationDataClean,
    ) -> tuple[torch.Tensor, torch.Tensor, int, int]:  # [1,C,V*T,H,W], [1,C,V*T,H,W], V, T
        """Validate and return ``control, target, views, frames_per_view``."""
        vision_items = gen_data_clean.x0_tokens_vision
        if vision_items is None or len(vision_items) != 2:
            raise ValueError("Multiview WSM self-forcing requires [WSM control, RGB target] vision items.")
        if gen_data_clean.batch_size != 1 or gen_data_clean.num_vision_items_per_sample != [2]:
            raise ValueError(
                "Multiview WSM self-forcing requires one sample with exactly two vision items; "
                f"got batch_size={gen_data_clean.batch_size} and "
                f"num_vision_items_per_sample={gen_data_clean.num_vision_items_per_sample}."
            )
        num_views_per_item = gen_data_clean.num_views_per_vision_item
        if num_views_per_item is None or len(num_views_per_item) != 2:
            raise ValueError("Multiview WSM self-forcing requires per-item camera-view metadata.")
        num_views = num_views_per_item[0]
        if num_views < 1 or num_views_per_item != [num_views, num_views]:
            raise ValueError(f"WSM control and RGB target must use the same views, got {num_views_per_item}.")
        control, target = vision_items
        if control.ndim != 5 or target.ndim != 5 or control.shape[0] != 1 or target.shape[0] != 1:
            raise ValueError(
                "Multiview WSM self-forcing requires batched [1,C,V*T,H,W] items, "
                f"got control={tuple(control.shape)} and target={tuple(target.shape)}."
            )
        if control.shape != target.shape or target.shape[2] % num_views != 0:
            raise ValueError(
                "WSM control and RGB target must have matching camera-major shapes divisible by views; "
                f"got control={tuple(control.shape)}, target={tuple(target.shape)}, views={num_views}."
            )
        return control, target, num_views, target.shape[2] // num_views

    def _truncate_multiview_wsm_gen_data(
        self,
        gen_data_clean: GenerationDataClean,
        frames_per_view: int,
    ) -> GenerationDataClean:
        """Truncate every camera in lockstep without slicing a flattened tail."""
        _control, _target, num_views, original_frames = self._multiview_wsm_shape(gen_data_clean)
        if frames_per_view > original_frames or frames_per_view < 1:
            raise ValueError(f"Cannot truncate {original_frames} frames per view to {frames_per_view}.")
        if frames_per_view == original_frames:
            return gen_data_clean

        truncated_latents = [
            MultiviewTransferARBackend.slice_chunk(
                item,
                num_views=num_views,
                frames_per_view=original_frames,
                chunk_start=0,
                chunk_end=frames_per_view,
            ).contiguous()
            for item in gen_data_clean.x0_tokens_vision or []
        ]  # list[[1,C,V*T_sync,H,W]]
        truncated_positions = None
        if gen_data_clean.temporal_positions_vision is not None:
            if len(gen_data_clean.temporal_positions_vision) != len(truncated_latents):
                raise ValueError(
                    "Multiview temporal-position items must align with vision items, "
                    f"got {len(gen_data_clean.temporal_positions_vision)} position items and "
                    f"{len(truncated_latents)} vision items."
                )
            for positions in gen_data_clean.temporal_positions_vision:
                if positions.shape[0] != num_views * original_frames:
                    raise ValueError(
                        "Multiview temporal positions must match the camera-major latent length, "
                        f"got {positions.shape[0]} positions for {num_views} views and "
                        f"{original_frames} frames per view."
                    )
            truncated_positions = [
                torch.cat(
                    [
                        positions[view_idx * original_frames : view_idx * original_frames + frames_per_view]
                        for view_idx in range(num_views)
                    ]
                ).contiguous()
                for positions in gen_data_clean.temporal_positions_vision
            ]  # list[[V*T_sync]]

        truncated_raw = gen_data_clean.raw_state_vision
        if truncated_raw is not None:
            if len(truncated_raw) != len(truncated_latents):
                raise ValueError(
                    "Multiview raw pixel items must align with vision items, "
                    f"got {len(truncated_raw)} raw items and {len(truncated_latents)} vision items."
                )
            pixel_frames = self.tokenizer_vision_gen.get_pixel_num_frames(frames_per_view)
            raw_frames: int | None = None
            for item in truncated_raw:
                if item.ndim not in (4, 5):
                    raise ValueError(f"Expected multiview raw pixels with 4 or 5 dims, got {tuple(item.shape)}.")
                flat_raw_frames = int(item.shape[-3])
                if flat_raw_frames % num_views != 0:
                    raise ValueError(
                        "Multiview raw pixel length must be divisible by views, "
                        f"got length={flat_raw_frames} and views={num_views}."
                    )
                item_raw_frames = flat_raw_frames // num_views
                if raw_frames is None:
                    raw_frames = item_raw_frames
                elif item_raw_frames != raw_frames:
                    raise ValueError(
                        "WSM control and RGB target must have equal raw frames per view, "
                        f"got {raw_frames} and {item_raw_frames}."
                    )
            if raw_frames is None or pixel_frames > raw_frames:
                raise ValueError(
                    "Requested synchronized raw prefix exceeds the available frames, "
                    f"requested={pixel_frames}, available={raw_frames}."
                )
            truncated_raw = [
                torch.cat(
                    [
                        item[
                            ...,
                            view_idx * raw_frames : view_idx * raw_frames + pixel_frames,
                            :,
                            :,
                        ]
                        for view_idx in range(num_views)
                    ],
                    dim=-3,
                ).contiguous()
                for item in truncated_raw
            ]  # list[[1,C,V*T_pixel_sync,H,W]]

        return replace(
            gen_data_clean,
            raw_state_vision=truncated_raw,
            x0_tokens_vision=truncated_latents,
            temporal_positions_vision=truncated_positions,
        )

    @staticmethod
    def _slice_action_for_chunk(
        action_tokens: torch.Tensor | None,
        chunk_start: int,
        chunk_end: int,
        temporal_compression_factor: int,
    ) -> torch.Tensor | None:
        """Concatenate per-frame action slices for latent frames ``[chunk_start, chunk_end)``.

        Frame 0 is always its own singleton chunk (no previous action), so any chunk with
        ``chunk_start >= 1`` carries the contiguous action span ``a{chunk_start-1}..a{chunk_end-2}``.
        """
        if action_tokens is None or chunk_start == 0:
            return None
        if action_tokens.ndim == 3:
            if action_tokens.shape[0] != 1:
                raise ValueError(f"Expected action batch size 1, got shape {tuple(action_tokens.shape)}")
            action_tokens = action_tokens.squeeze(0)  # [T_action,D]
        if action_tokens.ndim != 2:
            raise ValueError(f"Expected action tokens with 2 or 3 dims, got shape {tuple(action_tokens.shape)}")

        start = (chunk_start - 1) * temporal_compression_factor
        end = (chunk_end - 1) * temporal_compression_factor
        if action_tokens.shape[0] < end:
            raise ValueError(
                f"Not enough action tokens for chunk [{chunk_start},{chunk_end}): need {end}, "
                f"got {action_tokens.shape[0]}"
            )
        return action_tokens[start:end]  # [chunk_len*TCF,D]

    @staticmethod
    def _slice_action_domain_for_ar_unit(
        action_domain_id: torch.Tensor | None,
        action_latent: torch.Tensor | None,
        frame_idx: int,
        temporal_compression_factor: int,
    ) -> torch.Tensor | None:
        """Align scalar or framewise domain metadata with one AR frame/chunk.

        ``action_latent`` already carries the action rows for the unit beginning at
        ``frame_idx``. Framewise IDs must therefore use the identical absolute row
        range; scalar metadata remains scalar. Frame 0 has only null action tokens,
        whose domain follows the first forthcoming real action just as in inference.
        """
        if action_domain_id is None:
            return None
        if action_latent is None:
            return OmniMoTCausalModel._null_action_domain_id(action_domain_id)
        if frame_idx < 1:
            raise ValueError("A real self-forcing action unit cannot begin at latent frame 0.")
        if action_latent.ndim not in (2, 3):
            raise ValueError(
                f"Expected self-forcing action latents with 2 or 3 dims, got shape {tuple(action_latent.shape)}"
            )
        action_token_count = int(action_latent.shape[-2])
        if action_token_count < 1:
            raise ValueError("A real self-forcing action unit must contain at least one action token.")
        start = (frame_idx - 1) * temporal_compression_factor
        return OmniMoTCausalModel._slice_action_domain_id(
            action_domain_id,
            start,
            start + action_token_count,
        )

    @staticmethod
    def _single_vision_prediction(preds_vision: list[torch.Tensor]) -> torch.Tensor:
        """Return the only per-sample vision prediction for single-sample AR packs."""
        if len(preds_vision) != 1:
            raise ValueError(f"Self-forcing expected one vision prediction, got {len(preds_vision)}")
        return preds_vision[0]  # [1,C,1,H,W]

    @staticmethod
    def _concat_vision_frames(frame_outputs: list[torch.Tensor]) -> torch.Tensor:
        """Concatenate per-frame outputs without dropping the singleton batch dim."""
        return torch.cat(frame_outputs, dim=2)  # [1,C,T,H,W]

    @staticmethod
    def _truncate_sequence_plan_for_frames(
        sequence_plan: SequencePlan,
        num_frames: int,
        action_steps: int,
    ) -> SequencePlan:
        """Drop conditioning indexes that no longer exist after distributed frame truncation."""
        return replace(
            sequence_plan,
            condition_frame_indexes_vision=[
                frame_idx for frame_idx in sequence_plan.condition_frame_indexes_vision if frame_idx < num_frames
            ],
            condition_frame_indexes_action=[
                step_idx for step_idx in sequence_plan.condition_frame_indexes_action if step_idx < action_steps
            ],
        )

    @staticmethod
    def _truncate_self_forcing_gen_data(
        gen_data_clean: GenerationDataClean,
        num_frames: int,
        temporal_compression_factor: int,
    ) -> GenerationDataClean:
        """Trim every aligned item and temporal metadata to one latent length."""
        if gen_data_clean.x0_tokens_vision is None:
            raise ValueError("Self-forcing DMD requires vision tokens.")

        truncated_vision: list[torch.Tensor] = []
        for vision_tokens in gen_data_clean.x0_tokens_vision:
            if vision_tokens.ndim == 5:
                vision_item = vision_tokens[:, :, :num_frames].contiguous()  # [1,C,T_sync,H,W]
            elif vision_tokens.ndim == 4:
                vision_item = vision_tokens[:, :num_frames].contiguous()  # [C,T_sync,H,W]
            else:
                raise ValueError(f"Expected vision tokens with 4 or 5 dims, got shape {tuple(vision_tokens.shape)}")
            truncated_vision.append(vision_item)

        pixel_frames = max(num_frames - 1, 0) * temporal_compression_factor + 1
        truncated_raw_vision = None
        if gen_data_clean.raw_state_vision is not None:
            truncated_raw_vision = [
                raw_vision[..., :pixel_frames, :, :].contiguous()  # [...,C,T_pixel_sync,H,W]
                for raw_vision in gen_data_clean.raw_state_vision
            ]

        truncated_temporal_positions = None
        if gen_data_clean.temporal_positions_vision is not None:
            truncated_temporal_positions = [
                positions[:num_frames].contiguous() for positions in gen_data_clean.temporal_positions_vision
            ]  # list[[T_sync]]

        action_steps = max(num_frames - 1, 0) * temporal_compression_factor
        truncated_actions = None
        truncated_action_domain_id = gen_data_clean.action_domain_id
        if gen_data_clean.x0_tokens_action is not None:
            truncated_actions = [
                action_tokens[..., :action_steps, :].contiguous()  # [...,T_action_sync,D]
                for action_tokens in gen_data_clean.x0_tokens_action
            ]

            if gen_data_clean.action_domain_id is not None:
                if len(gen_data_clean.action_domain_id) != len(gen_data_clean.x0_tokens_action):
                    raise ValueError(
                        "Self-forcing action-domain metadata must have one entry per action tensor; "
                        f"got {len(gen_data_clean.action_domain_id)} domain entries for "
                        f"{len(gen_data_clean.x0_tokens_action)} action tensors."
                    )
                truncated_action_domain_id = []
                for item_index, (domain_ids, source_actions) in enumerate(
                    zip(gen_data_clean.action_domain_id, gen_data_clean.x0_tokens_action)
                ):
                    source_rows = int(source_actions.shape[-2])
                    flat_domain_ids = domain_ids.reshape(-1)
                    if flat_domain_ids.numel() not in (1, source_rows):
                        raise ValueError(
                            f"Self-forcing action-domain entry {item_index} must be scalar or cover every real "
                            f"action row; got {flat_domain_ids.numel()} IDs for {source_rows} rows."
                        )
                    truncated_action_domain_id.append(
                        domain_ids if flat_domain_ids.numel() == 1 else flat_domain_ids[:action_steps].contiguous()
                    )

        truncated_raw_actions = None
        if gen_data_clean.raw_state_action is not None:
            truncated_raw_actions = [
                action_state[..., :action_steps, :].contiguous()  # [...,T_action_sync,D]
                for action_state in gen_data_clean.raw_state_action
            ]

        return replace(
            gen_data_clean,
            raw_state_vision=truncated_raw_vision,
            x0_tokens_vision=truncated_vision,
            temporal_positions_vision=truncated_temporal_positions,
            raw_state_action=truncated_raw_actions,
            x0_tokens_action=truncated_actions,
            action_domain_id=truncated_action_domain_id,
        )

    def _sync_self_forcing_num_frames(self, num_frames: int, device: torch.device) -> int:
        """Validate equal shard-group rollout lengths or return their legacy minimum."""
        if not dist.is_available() or not dist.is_initialized():
            return num_frames
        require_equal = self.config.sf_require_equal_rollout_lengths
        # MIN([T, -T]) gives both group bounds so every rank rejects a mismatch.
        frame_counts = [num_frames, -num_frames] if require_equal else [num_frames]
        num_frames_tensor = torch.tensor(frame_counts, dtype=torch.long, device=device)  # [2] or [1]
        dist.all_reduce(num_frames_tensor, op=dist.ReduceOp.MIN, group=self._self_forcing_sync_group())
        synced_num_frames = int(num_frames_tensor[0].item())
        if synced_num_frames < 1:
            raise ValueError(f"Synchronized self-forcing frame count must be >= 1, got {synced_num_frames}")
        if require_equal:
            max_num_frames = -int(num_frames_tensor[1].item())
            if synced_num_frames != max_num_frames:
                raise ValueError(
                    "SF-DMD requires equal rollout lengths within each FSDP shard group when "
                    "sf_require_equal_rollout_lengths=True; "
                    f"got min={synced_num_frames}, max={max_num_frames} latent frames. "
                    "Use length-aligned batches before tokenization; refusing to truncate the rollout."
                )
        return synced_num_frames

    def _self_forcing_sync_group(self) -> Any:
        """Return the FSDP shard process group when model parallel dims expose one."""
        parallel_dims = getattr(self, "parallel_dims", None)
        dp_shard_mesh = getattr(parallel_dims, "dp_shard_mesh", None)
        get_group = getattr(dp_shard_mesh, "get_group", None)
        if not callable(get_group):
            return None
        group = get_group()
        process_group_type = getattr(dist, "ProcessGroup", None)
        if process_group_type is not None and not isinstance(group, process_group_type):
            return None
        return group

    @staticmethod
    def _sync_transfer_conditioning_plans(
        sequence_plans: list[SequencePlan],
        gen_data_clean: GenerationDataClean,
    ) -> list[SequencePlan]:
        """Broadcast one Transfer prefix-conditioning plan across the full HSDP mesh."""
        if gen_data_clean.num_vision_items_per_sample != [2]:
            return sequence_plans
        if gen_data_clean.batch_size != 1 or len(sequence_plans) != 1:
            raise ValueError(
                "Transfer self-forcing conditioning synchronization requires one logical sample per rank; "
                f"got batch_size={gen_data_clean.batch_size} and {len(sequence_plans)} sequence plans."
            )

        plan = sequence_plans[0]
        conditioned_frames = list(plan.condition_frame_indexes_vision)
        if conditioned_frames != list(range(len(conditioned_frames))):
            raise ValueError(
                f"Transfer self-forcing requires prefix conditioning indexes [0, ..., K-1]; got {conditioned_frames}."
            )

        vision_items = gen_data_clean.x0_tokens_vision
        if vision_items is None or len(vision_items) != 2:
            raise ValueError("Transfer self-forcing requires one aligned [control, target] vision pair.")
        target = vision_items[1]
        num_frames = target.shape[2] if target.ndim == 5 else target.shape[1]

        conditioned_count = len(conditioned_frames)
        if dist.is_available() and dist.is_initialized():
            count_tensor = torch.tensor([conditioned_count], dtype=torch.long, device=target.device)  # [1]
            # HSDP backward collectives span both shard and replicate axes. Use
            # the world group so every rank builds the same trainable graph.
            dist.broadcast(count_tensor, src=0)
            conditioned_count = int(count_tensor.item())
        if conditioned_count < 0 or num_frames < 1:
            raise ValueError(
                "Synchronized Transfer conditioning must leave at least one generated frame; "
                f"got {conditioned_count} conditioned frames for {num_frames} total frames."
            )
        # The data augmentor samples the same prefix options for all temporal
        # buckets. Short buckets must retain at least one latent to generate;
        # the one-frame bucket therefore uses control-only conditioning.
        conditioned_count = min(conditioned_count, num_frames - 1)

        synced_conditioned_frames = list(range(conditioned_count))
        return [replace(plan, condition_frame_indexes_vision=synced_conditioned_frames)]

    def _make_ar_pack(
        self,
        *,
        vision_latent: torch.Tensor,
        action_latent: torch.Tensor | None,
        text_tokens: list[int] | None,
        timestep: float,
        frame_idx: int,
        cached_text_offset: int,
        fps_vision_list: list[float],
        fps_action_list: list[float] | None,
        force_action_tokens: bool,
        action_domain_id: torch.Tensor | None,
        raw_action_dim: torch.Tensor | None,
        condition_frame_indexes_vision: list[int] | None = None,
    ) -> PackedSequence:
        """Pack one self-forcing AR frame using the interactive inference layout."""
        diffusion_cfg = self.config.diffusion_expert_config
        return pack_input_sequence_autoregressive(
            vision_latent=vision_latent,
            action_latent=action_latent,
            text_tokens=text_tokens,
            timestep=timestep,
            fps_vision=fps_vision_list,
            fps_action=fps_action_list if force_action_tokens else None,
            special_tokens=self.llm_special_tokens,
            latent_patch_size=diffusion_cfg.patch_spatial,
            condition_frame_indexes_vision=condition_frame_indexes_vision or [],
            condition_frame_indexes_action=[],
            frame_idx=frame_idx,
            temporal_compression_factor=self.tokenizer_vision_gen.temporal_compression_factor or 4,
            video_temporal_causal=self.config.video_temporal_causal,
            action_dim=self.config.max_action_dim,
            enable_fps_modulation=diffusion_cfg.enable_fps_modulation,
            base_fps=float(diffusion_cfg.base_fps),
            cached_text_offset=None if text_tokens is not None else cached_text_offset,
            unified_3d_mrope_temporal_modality_margin=diffusion_cfg.unified_3d_mrope_temporal_modality_margin,
            force_action_tokens=force_action_tokens,
            action_domain_id=action_domain_id,
            raw_action_dim=raw_action_dim,
        )

    def _denoise_self_forcing_frame(
        self,
        *,
        frame_latent: torch.Tensor,
        action_latent: torch.Tensor | None,
        frame_idx: int,
        num_frames: int,
        dual_kv_cache: list[DualKVCache],
        text_tokens: list[int] | None,
        cached_text_offset: int,
        gen_data_clean: GenerationDataClean,
        fps_vision_list: list[float],
        fps_action_list: list[float] | None,
        force_action_tokens: bool,
        cache_metadata: _SelfForcingCacheMetadata | None = None,
        condition_latent: torch.Tensor | None = None,
        condition_frame_indexes_vision: list[int] | None = None,
        step_counts: list[int] | None = None,
        allow_grad: bool = True,
        collect_replay: bool = False,
    ) -> tuple[torch.Tensor, dict, torch.Tensor | None, float | None]:
        """Denoise one AR chunk, keeping gradients only for configured final steps."""
        full_t_list = self._self_forcing_timestep_schedule_for_frame(frame_idx, num_frames, step_counts=step_counts)
        n_steps = len(full_t_list) - 1
        backward_grad_steps = self.config.backward_grad_steps
        grad_steps = n_steps if backward_grad_steps == -1 else backward_grad_steps
        if backward_grad_steps == 0:
            raise ValueError("backward_grad_steps must be -1 or >= 1 for self-forcing rollout.")

        local_condition_indexes = condition_frame_indexes_vision or []
        condition_mask = torch.zeros(
            (1, 1, frame_latent.shape[2], 1, 1),
            dtype=torch.float32,
            device=frame_latent.device,
        )  # [1,1,chunk_len,1,1]
        if local_condition_indexes:
            if condition_latent is None:
                raise ValueError("Conditioned self-forcing chunks require clean condition latents.")
            condition_mask[:, :, local_condition_indexes] = 1.0  # [1,1,chunk_len,1,1]
        clean_condition = (
            condition_latent.float() if condition_latent is not None else frame_latent.float()
        )  # [1,C,chunk_len,H,W]
        x_frame = (
            condition_mask * clean_condition + (1.0 - condition_mask) * frame_latent.float()
        )  # [1,C,chunk_len,H,W]
        out_student: dict[str, Any] = {}
        x0_pred = x_frame  # [1,C,chunk_len,H,W]
        max_timestep = float(self.config.rectified_flow_inference_config.num_train_timesteps)
        sample_type = self.config.fixed_step_sampler_config.sample_type
        replay_xt: torch.Tensor | None = None
        replay_sigma: float | None = None
        action_domain_id = (
            self._slice_action_domain_for_ar_unit(
                OmniMoTCausalModel._first_action_domain_id(gen_data_clean),
                action_latent,
                frame_idx,
                self.tokenizer_vision_gen.temporal_compression_factor or 4,
            )
            if force_action_tokens
            else None
        )

        for step_idx, (sigma_cur, sigma_next) in enumerate(zip(full_t_list[:-1], full_t_list[1:])):
            timestep = float(sigma_cur) * max_timestep
            enable_grad = allow_grad and step_idx >= n_steps - grad_steps
            if collect_replay and step_idx == n_steps - grad_steps:
                replay_xt = x_frame.detach().clone()  # [1,C,1,H,W]
                replay_sigma = float(sigma_cur)
            action_frame_latent = (
                action_latent.to(**self.tensor_kwargs) if action_latent is not None else None
            )  # [TCF,D] or None
            pack = self._make_ar_pack(
                vision_latent=x_frame.to(**self.tensor_kwargs),
                action_latent=action_frame_latent,
                text_tokens=text_tokens,
                timestep=timestep,
                frame_idx=frame_idx,
                cached_text_offset=cached_text_offset,
                fps_vision_list=fps_vision_list,
                fps_action_list=fps_action_list,
                force_action_tokens=force_action_tokens,
                action_domain_id=action_domain_id,
                raw_action_dim=OmniMoTCausalModel._first_raw_action_dim(gen_data_clean),
                condition_frame_indexes_vision=local_condition_indexes,
            )
            self._cast_ar_action_tokens_to_projection_dtype(pack)
            pack.to_cuda()
            resolved_cache_metadata = cache_metadata or _SelfForcingCacheMetadata(cache_frame_idx=frame_idx)
            memory = self.build_memory_state(
                pack, resolved_cache_metadata.to_memory_info(dual_kv_cache, write_gen_cache=False)
            )
            with torch.set_grad_enabled(enable_grad):
                out_student = self.denoise(
                    net=self.denoiser_nets["student"],
                    data_batch_packed=pack,
                    memory=memory,
                )
            velocity = self._single_vision_prediction(out_student["preds_vision"]).float()  # [1,C,1,H,W]
            sigma_cur_tensor = torch.as_tensor(sigma_cur, dtype=torch.float32, device=x_frame.device)  # []
            x0_pred = x_frame - sigma_cur_tensor * velocity  # [1,C,chunk_len,H,W]
            x0_pred = condition_mask * clean_condition + (1.0 - condition_mask) * x0_pred  # [1,C,chunk_len,H,W]

            if sigma_next > 0.0:
                if sample_type == "ode":
                    delta_sigma = torch.as_tensor(
                        sigma_next - sigma_cur, dtype=torch.float32, device=x_frame.device
                    )  # []
                    x_frame = x_frame + delta_sigma * velocity  # [1,C,chunk_len,H,W]
                else:
                    noise = torch.randn_like(x0_pred, **self.tensor_kwargs_fp32)  # [1,C,chunk_len,H,W]
                    context_parallel_broadcast_tensor_list([noise], getattr(self, "parallel_dims", None))
                    sigma_next_tensor = torch.as_tensor(
                        sigma_next,
                        dtype=torch.float32,
                        device=x_frame.device,
                    )  # []
                    x_frame = (1.0 - sigma_next_tensor) * x0_pred + sigma_next_tensor * noise  # [1,C,chunk_len,H,W]
                x_frame = condition_mask * clean_condition + (1.0 - condition_mask) * x_frame  # [1,C,chunk_len,H,W]

        return x0_pred, out_student, replay_xt, replay_sigma

    def _denoise_self_forcing_multiview_wsm_chunk(
        self,
        *,
        chunk_latent: torch.Tensor,  # [1,C,V*chunk_len,H,W]
        chunk_start: int,
        chunk_len: int,
        frames_per_view: int,
        backend: MultiviewTransferARBackend,
        backend_context: MultiviewTransferARReplayContext,
        step_counts: list[int],
        allow_grad: bool,
        collect_replay: bool,
    ) -> tuple[torch.Tensor, dict[str, Any], torch.Tensor | None, float | None]:
        """Denoise one synchronized camera chunk against detached Flex history."""
        full_t_list = self._self_forcing_timestep_schedule_for_frame(
            chunk_start,
            frames_per_view,
            step_counts=step_counts,
        )
        n_steps = len(full_t_list) - 1
        backward_grad_steps = self.config.backward_grad_steps
        grad_steps = n_steps if backward_grad_steps == -1 else backward_grad_steps
        if backward_grad_steps == 0:
            raise ValueError("backward_grad_steps must be -1 or >= 1 for self-forcing rollout.")

        x_chunk = chunk_latent.float()  # [1,C,V*chunk_len,H,W]
        x0_pred = x_chunk  # [1,C,V*chunk_len,H,W]
        out_student: dict[str, Any] = {}
        max_timestep = float(self.config.rectified_flow_inference_config.num_train_timesteps)
        sample_type = self.config.fixed_step_sampler_config.sample_type
        replay_xt: torch.Tensor | None = None  # [1,C,V*chunk_len,H,W] or None
        replay_sigma: float | None = None

        for step_idx, (sigma_cur, sigma_next) in enumerate(zip(full_t_list[:-1], full_t_list[1:])):
            enable_grad = allow_grad and step_idx >= n_steps - grad_steps
            if collect_replay and step_idx == n_steps - grad_steps:
                replay_xt = x_chunk.detach().clone()  # [1,C,V*chunk_len,H,W]
                replay_sigma = float(sigma_cur)
            timestep = float(sigma_cur) * max_timestep
            pack, memory = backend.build_replay_pack_and_memory(
                context=backend_context,
                vision_latent=x_chunk.to(**self.tensor_kwargs),  # [1,C,V*chunk_len,H,W]
                chunk_start=chunk_start,
                timestep=timestep,
            )
            assert pack.vision is not None and len(pack.vision.tokens) == 1
            pack.vision.tokens = [x_chunk.to(**self.tensor_kwargs)]  # list[[1,C,V*chunk_len,H,W]]
            with torch.set_grad_enabled(enable_grad):
                out_student = self.denoise(
                    net=self.denoiser_nets["student"],
                    data_batch_packed=pack,
                    memory=memory,
                )
            velocity = self._single_vision_prediction(out_student["preds_vision"]).float()  # [1,C,V*chunk_len,H,W]
            sigma_cur_tensor = torch.as_tensor(
                sigma_cur,
                dtype=torch.float32,
                device=x_chunk.device,
            )  # []
            x0_pred = x_chunk - sigma_cur_tensor * velocity  # [1,C,V*chunk_len,H,W]

            if sigma_next > 0.0:
                if sample_type == "ode":
                    delta_sigma = torch.as_tensor(
                        sigma_next - sigma_cur,
                        dtype=torch.float32,
                        device=x_chunk.device,
                    )  # []
                    x_chunk = x_chunk + delta_sigma * velocity  # [1,C,V*chunk_len,H,W]
                else:
                    noise = torch.randn_like(x0_pred, **self.tensor_kwargs_fp32)  # [1,C,V*chunk_len,H,W]
                    context_parallel_broadcast_tensor_list([noise], getattr(self, "parallel_dims", None))
                    sigma_next_tensor = torch.as_tensor(
                        sigma_next,
                        dtype=torch.float32,
                        device=x_chunk.device,
                    )  # []
                    x_chunk = (1.0 - sigma_next_tensor) * x0_pred + sigma_next_tensor * noise
                    # [1,C,V*chunk_len,H,W]

        if x0_pred.shape[2] != backend_context.num_views * chunk_len:
            raise ValueError(
                f"Expected denoised camera-major chunk length {backend_context.num_views * chunk_len}, "
                f"got {x0_pred.shape[2]}."
            )
        return x0_pred, out_student, replay_xt, replay_sigma

    def _seed_self_forcing_history(
        self,
        *,
        frame_latent: torch.Tensor,
        action_latent: torch.Tensor | None,
        frame_idx: int,
        dual_kv_cache: list[DualKVCache],
        text_tokens: list[int] | None,
        cached_text_offset: int,
        gen_data_clean: GenerationDataClean,
        fps_vision_list: list[float],
        fps_action_list: list[float] | None,
        force_action_tokens: bool,
        cache_metadata: _SelfForcingCacheMetadata | None = None,
        position_frame_idx: int | None = None,
        condition_frame_indexes_vision: list[int] | None = None,
    ) -> None:
        """Write detached clean generated history into AR K/V cache."""
        with torch.no_grad():
            action_frame_latent = (
                action_latent.detach().to(**self.tensor_kwargs) if action_latent is not None else None
            )  # [TCF,D] or None
            resolved_cache_metadata = cache_metadata or _SelfForcingCacheMetadata(cache_frame_idx=frame_idx)
            OmniMoTCausalModel._seed_frame_into_kv_cache(
                self,
                frame_latent=frame_latent.detach().to(**self.tensor_kwargs),
                frame_idx=resolved_cache_metadata.cache_frame_idx,
                dual_kv_cache=dual_kv_cache,
                dual_kv_cache_uncond=None,
                cond_text_tokens=text_tokens,
                uncond_text_tokens=None,
                cond_cached_text_offset=cached_text_offset,
                uncond_cached_text_offset=cached_text_offset,
                curr_action_latent=action_frame_latent,
                action_domain_id=(
                    self._slice_action_domain_for_ar_unit(
                        OmniMoTCausalModel._first_action_domain_id(gen_data_clean),
                        action_frame_latent,
                        frame_idx,
                        self.tokenizer_vision_gen.temporal_compression_factor or 4,
                    )
                    if force_action_tokens
                    else None
                ),
                gen_data_clean=gen_data_clean,
                fps_vision_list=fps_vision_list,
                fps_action_list=fps_action_list or [24.0],
                seed=0,
                cfg_active=False,
                cfgp_enabled=False,
                tcf=self.tokenizer_vision_gen.temporal_compression_factor or 4,
                patch_size=self.config.diffusion_expert_config.patch_spatial,
                action_dim=self.config.max_action_dim,
                video_tc=self.config.video_temporal_causal,
                enable_fps_mod=self.config.diffusion_expert_config.enable_fps_modulation,
                base_fps=float(self.config.diffusion_expert_config.base_fps),
                modality_margin=self.config.diffusion_expert_config.unified_3d_mrope_temporal_modality_margin,
                position_frame_idx=position_frame_idx,
                condition_frame_indexes_vision=condition_frame_indexes_vision,
                transfer_history_sink_tokens=resolved_cache_metadata.transfer_history_sink_tokens,
                transfer_history_max_tokens=resolved_cache_metadata.transfer_history_max_tokens,
            )

    def _transfer_history_cache_spec(
        self,
        *,
        num_frames: int,
        vision_item: torch.Tensor,
    ) -> tuple[int, int, int, int | None, int | None]:
        """Resolve physical cache capacity and logical Transfer token visibility."""
        logical_window = self.config.kv_cache_inference_size
        logical_sink = int(self.config.attention_sink_size)
        if logical_window is None:
            if logical_sink != 0:
                raise ValueError("Transfer attention sinks require a finite kv_cache_inference_size.")
            return 2 * num_frames + 1, 0, 0, None, None
        if logical_window < 1:
            raise ValueError(f"Transfer kv_cache_inference_size must be >= 1, got {logical_window}.")
        if logical_sink < 0 or logical_sink >= logical_window:
            raise ValueError(
                "Transfer attention_sink_size must satisfy 0 <= sink < kv_cache_inference_size; "
                f"got sink={logical_sink}, window={logical_window}."
            )

        patch_size = int(self.config.diffusion_expert_config.patch_spatial)
        latent_h = int(vision_item.shape[-2])
        latent_w = int(vision_item.shape[-1])
        # Cosmos3VFMNetwork zero-pads non-divisible latent grids before
        # patchifying. The padding patches are real sequence tokens (for
        # example, 720p is 45x80 latents -> 23x40 patches), so mirror its ceil
        # division here when deriving the rolling-history visibility budget.
        patch_h = (latent_h + patch_size - 1) // patch_size
        patch_w = (latent_w + patch_size - 1) // patch_size
        tokens_per_frame = patch_h * patch_w
        recent_past_frames = logical_window - logical_sink - 1
        sink_tokens = 2 * logical_sink * tokens_per_frame
        control_max_tokens = 2 * recent_past_frames * tokens_per_frame
        target_max_tokens = (2 * recent_past_frames + 1) * tokens_per_frame
        return 2 * logical_window, 2 * logical_sink, sink_tokens, control_max_tokens, target_max_tokens

    def _self_forcing_multiview_wsm_simulation(
        self,
        input_text_indexes: list[list[int]],
        sequence_plans: list[SequencePlan],
        gen_data_clean: GenerationDataClean,
        *,
        effective_iteration: int | None,
        allow_grad: bool,
        collect_replay: bool,
    ) -> _SelfForcingSimulationResult:
        """Roll out only the RGB target with synchronized camera-major Flex history."""
        if not self._uses_multiview_replay_kv():
            raise ValueError("Multiview WSM self-forcing requires the multiview Flex K/V replay implementation.")
        if self.config.action_gen or self.config.sound_gen:
            raise ValueError("Multiview WSM self-forcing supports vision-only transfer batches.")
        replay_policy = self._get_teacher_forcing_replay_policy()
        if replay_policy.controls_read_strict_past_clean_rgb:
            raise ValueError(
                "Multiview WSM self-forcing requires controls_read_strict_past_clean_rgb=False; "
                "the rollout captures control K/V once and does not refresh it after generated target chunks."
            )
        if len(sequence_plans) != 1:
            raise ValueError("Multiview WSM self-forcing requires exactly one sequence per rank.")
        text_view_ids = sequence_plans[0].text_view_ids
        expected_text_count = len(text_view_ids) if text_view_ids is not None else 1
        if len(input_text_indexes) != expected_text_count:
            raise ValueError(
                f"Multiview WSM self-forcing requires {expected_text_count} caption token sequences, "
                f"got {len(input_text_indexes)}."
            )

        control, target, num_views, frames_per_view = self._multiview_wsm_shape(gen_data_clean)
        synced_frames_per_view = self._sync_self_forcing_num_frames(frames_per_view, target.device)
        if synced_frames_per_view != frames_per_view:
            log.info(
                "[SF-DMD] truncating synchronized multiview frames per view "
                f"{frames_per_view} -> {synced_frames_per_view}"
            )
            gen_data_clean = self._truncate_multiview_wsm_gen_data(
                gen_data_clean,
                synced_frames_per_view,
            )
            sequence_plans = [
                self._truncate_sequence_plan_for_frames(
                    sequence_plan,
                    synced_frames_per_view,
                    action_steps=0,
                )
                for sequence_plan in sequence_plans
            ]
            control, target, num_views, frames_per_view = self._multiview_wsm_shape(gen_data_clean)

        step_counts = self._self_forcing_step_counts_for_iteration(
            frames_per_view,
            effective_iteration,
        )
        timesteps_zero = torch.zeros((1, 1), dtype=torch.float32)  # [B,1]
        packed_reference = self._pack_input_sequence(
            sequence_plans,
            input_text_indexes,
            gen_data_clean,
            timesteps_zero,
        )
        if packed_reference.vision is None or len(packed_reference.vision.condition_mask) != 2:
            raise ValueError("Multiview WSM self-forcing requires packed [control, target] vision masks.")
        target_condition_mask = packed_reference.vision.condition_mask[1]  # [V*T,1,1]
        condition_count = multiview_conditioned_prefix_length(
            target_condition_mask,
            num_views=num_views,
            frames_per_view=frames_per_view,
        )
        if condition_count >= frames_per_view:
            raise ValueError(
                "Multiview WSM self-forcing requires at least one generated target frame; "
                f"got condition_count={condition_count}, frames_per_view={frames_per_view}."
            )

        fps_vision_list = gen_data_clean.fps_vision.tolist() if gen_data_clean.fps_vision is not None else [24.0]
        backend = self._make_multiview_transfer_ar_backend()
        prefill_pack = backend.build_prefill_pack(
            sequence_plans=sequence_plans,
            gen_data_clean=gen_data_clean,
            text_tokens=input_text_indexes,
            materialized_target_frame_ranges=[],
        )
        session = backend.create_session(
            prefill_pack=prefill_pack,
            num_views=num_views,
            frames_per_view=frames_per_view,
            condition_count=condition_count,
            cfg_active=False,
            cfgp_enabled=False,
            text_view_ids=sequence_plans[0].text_view_ids,
        )
        generated_target = target.to(device=self.tensor_kwargs["device"], dtype=torch.float32).clone()
        replay_frames: list[_MultiviewWSMReplayFrame] = []
        out_student: dict[str, Any] = {}

        with self._student_rollout_reshard_context():
            with torch.no_grad():
                backend.capture_control_cache(
                    session=session,
                    conditional_pack=prefill_pack,
                    unconditional_pack=None,
                )

            for chunk_start, chunk_end in _iter_ar_chunk_ranges(
                condition_count,
                frames_per_view,
                self.config.teacher_forcing_frames_per_chunk,
            ):
                chunk_len = chunk_end - chunk_start
                backend_context = backend.snapshot_replay_context(
                    session,
                    text_tokens=input_text_indexes,
                    fps_vision=fps_vision_list,
                )
                clean_chunk = backend.slice_chunk(
                    target,
                    num_views=num_views,
                    frames_per_view=frames_per_view,
                    chunk_start=chunk_start,
                    chunk_end=chunk_end,
                ).to(device=self.tensor_kwargs["device"], dtype=torch.float32)  # [1,C,V*chunk_len,H,W]
                init_noise = torch.randn_like(clean_chunk, **self.tensor_kwargs_fp32)  # [1,C,V*chunk_len,H,W]
                context_parallel_broadcast_tensor_list([init_noise], getattr(self, "parallel_dims", None))
                x0_chunk, out_student, replay_xt, replay_sigma = self._denoise_self_forcing_multiview_wsm_chunk(
                    chunk_latent=init_noise,
                    chunk_start=chunk_start,
                    chunk_len=chunk_len,
                    frames_per_view=frames_per_view,
                    backend=backend,
                    backend_context=backend_context,
                    step_counts=step_counts,
                    allow_grad=allow_grad,
                    collect_replay=collect_replay,
                )
                backend.scatter_chunk(
                    generated_target,
                    x0_chunk,
                    num_views=num_views,
                    frames_per_view=frames_per_view,
                    chunk_start=chunk_start,
                    chunk_end=chunk_end,
                )
                if collect_replay:
                    if replay_xt is None or replay_sigma is None:
                        raise RuntimeError(f"Missing replay state for multiview WSM chunk [{chunk_start},{chunk_end}).")
                    replay_frames.append(
                        _MultiviewWSMReplayFrame(
                            frame_idx=chunk_start,
                            xt=replay_xt,
                            sigma=replay_sigma,
                            chunk_len=chunk_len,
                            backend_context=backend_context,
                        )
                    )

                if chunk_end < frames_per_view:
                    with torch.no_grad():
                        backend.commit_clean_chunk(
                            session=session,
                            denoised_chunk=x0_chunk.detach().to(**self.tensor_kwargs),  # [1,C,V*chunk_len,H,W]
                            chunk_start=chunk_start,
                            chunk_end=chunk_end,
                            conditional_text_tokens=input_text_indexes,
                            unconditional_text_tokens=None,
                            fps_vision=fps_vision_list,
                        )

        conditioned_control = control.to(
            device=self.tensor_kwargs["device"], dtype=torch.float32
        ).detach()  # [1,C,V*T,H,W]
        gen_data_student = replace(
            gen_data_clean,
            x0_tokens_vision=[
                conditioned_control,  # [1,C,V*T,H,W]
                generated_target,  # [1,C,V*T,H,W]
            ],
        )
        sigma_student = torch.full(
            (1, 1),
            float(self.config.fixed_step_sampler_config.t_list[0]),
            **self.tensor_kwargs_fp32,
        )  # [B,1]
        return _SelfForcingSimulationResult(
            gen_data_student=gen_data_student,
            packed_reference=packed_reference,
            sigma_student=sigma_student,
            out_student=out_student,
            replay_state=_MultiviewWSMReplayState(frames=replay_frames) if collect_replay else None,
        )

    def _self_forcing_simulation(
        self,
        input_text_indexes: list[list[int]],
        sequence_plans: list[SequencePlan],
        gen_data_clean: GenerationDataClean,
        *,
        effective_iteration: int | None = None,
        allow_grad: bool = True,
        collect_replay: bool = False,
    ) -> _SelfForcingSimulationResult:
        """Generate student samples with per-frame AR/KV self-forcing."""
        if gen_data_clean.batch_size != 1:
            raise ValueError("Self-forcing currently supports one packed sample per rank.")
        if gen_data_clean.x0_tokens_lidar:
            # The RGB rollout replaces only vision tokens. Retaining real LiDAR
            # targets would silently feed ground truth into the student scores.
            raise ValueError(
                "Self-forcing DMD does not implement joint RGB/LiDAR rollout and loss; "
                "refusing to retain ground-truth LiDAR in generated student samples."
            )
        if self.config.sound_gen:
            raise ValueError("Self-forcing DMD is currently vision-only; sound requires a separate design.")
        if self.config.action_gen and not self.config.video_temporal_causal:
            raise ValueError("Action-conditioned self-forcing requires video_temporal_causal=True.")
        if self.parallel_dims is not None and self.parallel_dims.cfgp_enabled:
            raise ValueError("Self-forcing DMD does not support CFGP.")
        if (
            self.parallel_dims is not None
            and self.parallel_dims.cp_enabled
            and self.config.attention_io_layout != "sequence_sharded"
        ):
            raise ValueError(
                "Self-forcing DMD under CP requires attention_io_layout='sequence_sharded' "
                "(Ulysses); the 'replicated' head-sliced AR layout is not supported."
            )

        if self.config.sf_rollout_backend == "multiview_wsm":
            return self._self_forcing_multiview_wsm_simulation(
                input_text_indexes,
                sequence_plans,
                gen_data_clean,
                effective_iteration=effective_iteration,
                allow_grad=allow_grad,
                collect_replay=collect_replay,
            )

        vision_items = gen_data_clean.x0_tokens_vision
        if vision_items is None:
            raise ValueError("Self-forcing DMD requires vision tokens.")
        item_counts = getattr(gen_data_clean, "num_vision_items_per_sample", None)
        is_transfer = item_counts == [2]
        if item_counts not in (None, [1], [2]):
            raise ValueError(
                "Self-forcing DMD supports one generated vision item or one aligned [control, target] pair; "
                f"got num_vision_items_per_sample={item_counts}."
            )
        if not is_transfer and len(vision_items) != 1:
            raise ValueError(
                f"Non-Transfer self-forcing requires exactly one flat vision item; got {len(vision_items)} items."
            )
        if is_transfer:
            if len(vision_items) != 2 or not sequence_plans[0].share_vision_temporal_positions:
                raise ValueError(
                    "Transfer self-forcing requires one logical [control, target] pair with shared temporal "
                    f"positions; got {len(vision_items)} items and "
                    f"share_vision_temporal_positions={sequence_plans[0].share_vision_temporal_positions}."
                )
            if vision_items[0].shape != vision_items[1].shape:
                raise ValueError(
                    "Transfer self-forcing requires aligned control and target latent shapes; "
                    f"got {tuple(vision_items[0].shape)} and {tuple(vision_items[1].shape)}."
                )
            if gen_data_clean.x0_tokens_action is not None:
                raise ValueError("Transfer self-forcing does not support action tokens.")
            if self.parallel_dims is not None and self.parallel_dims.cp_enabled:
                raise ValueError("Transfer self-forcing does not support context parallelism.")
            if self.config.compile.enabled:
                raise ValueError("Transfer self-forcing requires compile.enabled=False.")
            if self.config.kv_cache_dtype is not None:
                raise ValueError("Transfer self-forcing requires kv_cache_dtype=None for BF16 cache storage.")
            replay_policy = self._get_teacher_forcing_replay_policy()
            if (
                replay_policy.control_visibility != "causal"
                or not replay_policy.controls_read_strict_past_clean_rgb
                or replay_policy.clean_pass_causality != "frame"
            ):
                raise ValueError(
                    "Transfer self-forcing requires "
                    "teacher_forcing_replay_policy.control_visibility='causal' and "
                    "controls_read_strict_past_clean_rgb=True and "
                    "clean_pass_causality='frame'."
                )

        generation_vision_item_idx = 1 if is_transfer else 0
        vision_clean = vision_items[generation_vision_item_idx]  # [1,C,T,H,W] or [C,T,H,W]
        num_frames = vision_clean.shape[2] if vision_clean.ndim == 5 else vision_clean.shape[1]
        conditioned_frames = self._conditioned_vision_frames(sequence_plans[0])
        generated_frame_indexes = [idx for idx in range(num_frames) if idx not in conditioned_frames]
        if not generated_frame_indexes:
            raise ValueError("Self-forcing DMD requires at least one generated vision frame.")

        tcf = self.tokenizer_vision_gen.temporal_compression_factor or 4
        chunk_size = max(int(self.config.teacher_forcing_frames_per_chunk), 1)
        if is_transfer and chunk_size != 1:
            raise ValueError(
                "causal_control_with_rgb_history Transfer self-forcing requires teacher_forcing_frames_per_chunk=1."
            )
        synced_num_frames = self._sync_self_forcing_num_frames(num_frames, vision_clean.device)
        if synced_num_frames != num_frames:
            log.info(f"[SF-DMD] truncating local latent frames {num_frames} -> {synced_num_frames}")
            gen_data_clean = self._truncate_self_forcing_gen_data(gen_data_clean, synced_num_frames, tcf)
            action_steps = max(synced_num_frames - 1, 0) * tcf
            sequence_plans = [
                self._truncate_sequence_plan_for_frames(sequence_plan, synced_num_frames, action_steps)
                for sequence_plan in sequence_plans
            ]
            assert gen_data_clean.x0_tokens_vision is not None
            vision_items = gen_data_clean.x0_tokens_vision
            vision_clean = vision_items[generation_vision_item_idx]  # [1,C,T_sync,H,W] or [C,T_sync,H,W]
            num_frames = synced_num_frames
        conditioned_frames = self._conditioned_vision_frames(sequence_plans[0])
        generated_frame_indexes = [idx for idx in range(num_frames) if idx not in conditioned_frames]
        if not generated_frame_indexes:
            raise ValueError("Self-forcing DMD requires at least one generated vision frame.")
        step_counts = self._self_forcing_step_counts_for_iteration(num_frames, effective_iteration)

        if is_transfer:
            cache_size, physical_sink_size, sink_tokens, control_max_tokens, target_max_tokens = (
                self._transfer_history_cache_spec(num_frames=num_frames, vision_item=vision_clean)
            )
        else:
            cache_size = max(num_frames, 2)
            physical_sink_size = 0
            sink_tokens = 0
            control_max_tokens = None
            target_max_tokens = None
        num_layers: int = self.net.num_hidden_layers  # type: ignore[attr-defined]
        dual_kv_cache = [
            DualKVCache(gen_cache_size=cache_size, attention_sink_size=physical_sink_size) for _ in range(num_layers)
        ]
        fps_vision_list = gen_data_clean.fps_vision.tolist() if gen_data_clean.fps_vision is not None else [24.0]
        fps_action_list = gen_data_clean.fps_action.tolist() if gen_data_clean.fps_action is not None else [24.0]
        text_tokens = input_text_indexes[0]
        cached_text_offset = compute_text_split_length(
            len(text_tokens),
            self.llm_special_tokens,
            has_generation=True,
        )
        action_tokens = (
            gen_data_clean.x0_tokens_action[0]
            if gen_data_clean.x0_tokens_action is not None and len(gen_data_clean.x0_tokens_action) > 0
            else None
        )  # [T_action,D] or [1,T_action,D] or None
        force_action_tokens = self.config.action_gen and sequence_plans[0].has_action
        if force_action_tokens and action_tokens is None:
            raise ValueError("Action-conditioned self-forcing requires action tokens in the batch.")

        with self._student_rollout_reshard_context():
            if is_transfer:
                return self._run_transfer_self_forcing_chunks(
                    num_frames=num_frames,
                    conditioned_frames=conditioned_frames,
                    control_clean=vision_items[0],
                    target_clean=vision_clean,
                    dual_kv_cache=dual_kv_cache,
                    text_tokens=text_tokens,
                    cached_text_offset=cached_text_offset,
                    gen_data_clean=gen_data_clean,
                    fps_vision_list=fps_vision_list,
                    fps_action_list=fps_action_list,
                    step_counts=step_counts,
                    allow_grad=allow_grad,
                    collect_replay=collect_replay,
                    sequence_plans=sequence_plans,
                    input_text_indexes=input_text_indexes,
                    sink_tokens=sink_tokens,
                    control_max_tokens=control_max_tokens,
                    target_max_tokens=target_max_tokens,
                )
            return self._run_self_forcing_chunks(
                chunk_size=chunk_size,
                num_frames=num_frames,
                conditioned_frames=conditioned_frames,
                vision_clean=vision_clean,
                action_tokens=action_tokens,
                tcf=tcf,
                dual_kv_cache=dual_kv_cache,
                text_tokens=text_tokens,
                cached_text_offset=cached_text_offset,
                gen_data_clean=gen_data_clean,
                fps_vision_list=fps_vision_list,
                fps_action_list=fps_action_list,
                force_action_tokens=force_action_tokens,
                step_counts=step_counts,
                allow_grad=allow_grad,
                collect_replay=collect_replay,
                sequence_plans=sequence_plans,
                input_text_indexes=input_text_indexes,
            )

    def _run_transfer_self_forcing_chunks(
        self,
        *,
        num_frames: int,
        conditioned_frames: set[int],
        control_clean: torch.Tensor,
        target_clean: torch.Tensor,
        dual_kv_cache: list[DualKVCache],
        text_tokens: list[int],
        cached_text_offset: int,
        gen_data_clean: GenerationDataClean,
        fps_vision_list: list[float],
        fps_action_list: list[float] | None,
        step_counts: list[int],
        allow_grad: bool,
        collect_replay: bool,
        sequence_plans: list[SequencePlan],
        input_text_indexes: list[list[int]],
        sink_tokens: int,
        control_max_tokens: int | None,
        target_max_tokens: int | None,
    ) -> _SelfForcingSimulationResult:
        """Roll out Transfer by interleaving clean control and finalized RGB cache entries."""
        frame_outputs: list[torch.Tensor] = []
        replay_frames: list[_CausalARReplayFrame] = []
        out_student: dict[str, Any] = {}
        cache_frame_idx = 0

        for frame_idx in range(num_frames):
            control_frame = self._slice_vision_frame(control_clean, frame_idx).to(
                device=self.tensor_kwargs["device"],
                dtype=torch.float32,
            )  # [1,C,1,H,W]
            control_cache_metadata = _SelfForcingCacheMetadata(
                cache_frame_idx=cache_frame_idx,
                transfer_history_sink_tokens=sink_tokens,
                transfer_history_max_tokens=control_max_tokens,
            )
            self._seed_self_forcing_history(
                frame_latent=control_frame,
                action_latent=None,
                frame_idx=frame_idx,
                dual_kv_cache=dual_kv_cache,
                text_tokens=text_tokens if cache_frame_idx == 0 else None,
                cached_text_offset=cached_text_offset,
                gen_data_clean=gen_data_clean,
                fps_vision_list=fps_vision_list,
                fps_action_list=fps_action_list,
                force_action_tokens=False,
                cache_metadata=control_cache_metadata,
                position_frame_idx=frame_idx,
                condition_frame_indexes_vision=[0],
            )
            cache_frame_idx += 1

            target_cache_metadata = _SelfForcingCacheMetadata(
                cache_frame_idx=cache_frame_idx,
                transfer_history_sink_tokens=sink_tokens,
                transfer_history_max_tokens=target_max_tokens,
            )
            clean_target_frame = self._slice_vision_frame(target_clean, frame_idx).to(
                device=self.tensor_kwargs["device"],
                dtype=torch.float32,
            )  # [1,C,1,H,W]
            if frame_idx in conditioned_frames:
                x0_frame = clean_target_frame
            else:
                init_noise = torch.randn_like(clean_target_frame)  # [1,C,1,H,W]
                context_parallel_broadcast_tensor_list([init_noise], getattr(self, "parallel_dims", None))
                x0_frame, out_student, replay_xt, replay_sigma = self._denoise_self_forcing_frame(
                    frame_latent=init_noise,
                    action_latent=None,
                    frame_idx=frame_idx,
                    num_frames=num_frames,
                    dual_kv_cache=dual_kv_cache,
                    text_tokens=None,
                    cached_text_offset=cached_text_offset,
                    gen_data_clean=gen_data_clean,
                    fps_vision_list=fps_vision_list,
                    fps_action_list=fps_action_list,
                    force_action_tokens=False,
                    cache_metadata=target_cache_metadata,
                    step_counts=step_counts,
                    allow_grad=allow_grad,
                    collect_replay=collect_replay,
                )  # x0_frame: [1,C,1,H,W]
                if collect_replay:
                    if replay_xt is None or replay_sigma is None:
                        raise RuntimeError(f"Missing replay state for generated Transfer frame {frame_idx}.")
                    replay_frames.append(
                        _CausalARReplayFrame(
                            frame_idx=frame_idx,
                            xt=replay_xt,
                            sigma=replay_sigma,
                            action_latent=None,
                            text_tokens=None,
                            cache_metadata=target_cache_metadata,
                        )
                    )
            frame_outputs.append(x0_frame)

            # This shared path also seeds non-final conditioned targets so later frames
            # can attend to the clean prefix.
            if frame_idx < num_frames - 1:
                self._seed_self_forcing_history(
                    frame_latent=x0_frame,
                    action_latent=None,
                    frame_idx=frame_idx,
                    dual_kv_cache=dual_kv_cache,
                    text_tokens=None,
                    cached_text_offset=cached_text_offset,
                    gen_data_clean=gen_data_clean,
                    fps_vision_list=fps_vision_list,
                    fps_action_list=fps_action_list,
                    force_action_tokens=False,
                    cache_metadata=target_cache_metadata,
                    position_frame_idx=frame_idx,
                    condition_frame_indexes_vision=[0],
                )
                cache_frame_idx += 1

        generated_vision = self._concat_vision_frames(frame_outputs)  # [1,C,T,H,W]
        timesteps_zero = torch.zeros((gen_data_clean.batch_size, 1), dtype=torch.float32)  # [B,1]
        packed_reference = self._pack_input_sequence(
            sequence_plans,
            input_text_indexes,
            gen_data_clean,
            timesteps_zero,
        )
        sigma_student = torch.full(
            (gen_data_clean.batch_size, 1),
            float(self.config.fixed_step_sampler_config.t_list[0]),
            **self.tensor_kwargs_fp32,
        )  # [B,1]
        assert gen_data_clean.x0_tokens_vision is not None
        student_vision_items = list(gen_data_clean.x0_tokens_vision)
        student_vision_items[1] = generated_vision
        gen_data_student = replace(
            gen_data_clean,
            x0_tokens_vision=student_vision_items,
            x0_tokens_sound=None,
            x0_tokens_action=None,
        )
        return _SelfForcingSimulationResult(
            gen_data_student=gen_data_student,
            packed_reference=packed_reference,
            sigma_student=sigma_student,
            out_student=out_student,
            replay_state=(
                _CausalARReplayState(
                    frames=replay_frames,
                    dual_kv_cache=dual_kv_cache,
                    fps_vision_list=fps_vision_list,
                    fps_action_list=fps_action_list,
                    cached_text_offset=cached_text_offset,
                    force_action_tokens=False,
                )
                if collect_replay
                else None
            ),
        )

    def _run_self_forcing_chunks(
        self,
        *,
        chunk_size: int,
        num_frames: int,
        conditioned_frames: set[int],
        vision_clean: torch.Tensor,
        action_tokens: torch.Tensor | None,
        tcf: int,
        dual_kv_cache: list[DualKVCache],
        text_tokens: list[int],
        cached_text_offset: int,
        gen_data_clean: GenerationDataClean,
        fps_vision_list: list[float],
        fps_action_list: list[float] | None,
        force_action_tokens: bool,
        step_counts: list[int],
        allow_grad: bool,
        collect_replay: bool,
        sequence_plans: list[SequencePlan],
        input_text_indexes: list[list[int]],
    ) -> _SelfForcingSimulationResult:
        frame_outputs: list[torch.Tensor] = []
        replay_frames: list[_CausalARReplayFrame] = []
        out_student: dict[str, Any] = {}

        # Seed the contiguous clean prefix before generating. Starting the chunk iterator at
        # the prefix boundary lets a multi-frame prefix consume only the partial tail of its
        # [1, C, C, ...] partition chunk instead of mixing clean and generated frames.
        generation_start = 0
        while generation_start < num_frames and generation_start in conditioned_frames:
            clean_frame = self._slice_vision_frame(vision_clean, generation_start).to(
                device=self.tensor_kwargs["device"],
                dtype=torch.float32,
            )  # [1,C,1,H,W]
            frame_outputs.append(clean_frame)
            if generation_start < num_frames - 1:
                self._seed_self_forcing_history(
                    frame_latent=clean_frame,
                    action_latent=self._slice_action_for_frame(action_tokens, generation_start, tcf),
                    frame_idx=generation_start,
                    dual_kv_cache=dual_kv_cache,
                    text_tokens=text_tokens if generation_start == 0 else None,
                    cached_text_offset=cached_text_offset,
                    gen_data_clean=gen_data_clean,
                    fps_vision_list=fps_vision_list,
                    fps_action_list=fps_action_list,
                    force_action_tokens=force_action_tokens,
                )
            generation_start += 1

        # Iterate the remaining chunkwise teacher-forcing partition. ``chunk_size == 1``
        # recovers the framewise rollout exactly (one frame per range).
        for chunk_start, chunk_end in _iter_ar_chunk_ranges(generation_start, num_frames, chunk_size):
            chunk_frames = list(range(chunk_start, chunk_end))
            chunk_len = chunk_end - chunk_start
            conditioned_flags = [f in conditioned_frames for f in chunk_frames]
            include_text = chunk_start == 0
            chunk_text_tokens = text_tokens if include_text else None

            if all(conditioned_flags):
                # Whole chunk is clean conditioning: copy clean frames and seed each.
                for f in chunk_frames:
                    clean_frame = self._slice_vision_frame(vision_clean, f).to(
                        device=self.tensor_kwargs["device"],
                        dtype=torch.float32,
                    )  # [1,C,1,H,W]
                    frame_outputs.append(clean_frame)
                    if f < num_frames - 1:
                        self._seed_self_forcing_history(
                            frame_latent=clean_frame,
                            action_latent=self._slice_action_for_frame(action_tokens, f, tcf),
                            frame_idx=f,
                            dual_kv_cache=dual_kv_cache,
                            text_tokens=text_tokens if f == 0 else None,
                            cached_text_offset=cached_text_offset,
                            gen_data_clean=gen_data_clean,
                            fps_vision_list=fps_vision_list,
                            fps_action_list=fps_action_list,
                            force_action_tokens=force_action_tokens,
                        )
                continue
            if any(conditioned_flags):
                raise NotImplementedError(
                    "Chunkwise SF-DMD does not support a chunk that mixes conditioned and generated frames; "
                    f"chunk [{chunk_start},{chunk_end}) conditioned flags {conditioned_flags}."
                )

            clean_chunk = self._slice_vision_chunk(vision_clean, chunk_start, chunk_end).to(
                device=self.tensor_kwargs["device"],
                dtype=torch.float32,
            )  # [1,C,chunk_len,H,W]
            action_chunk = self._slice_action_for_chunk(
                action_tokens, chunk_start, chunk_end, tcf
            )  # [chunk_len*TCF,D] or None
            init_noise = torch.randn_like(clean_chunk)  # [1,C,chunk_len,H,W]
            context_parallel_broadcast_tensor_list([init_noise], getattr(self, "parallel_dims", None))
            x0_chunk, out_student, replay_xt, replay_sigma = self._denoise_self_forcing_frame(
                frame_latent=init_noise,
                action_latent=action_chunk,
                frame_idx=chunk_start,
                num_frames=num_frames,
                dual_kv_cache=dual_kv_cache,
                text_tokens=chunk_text_tokens,
                cached_text_offset=cached_text_offset,
                gen_data_clean=gen_data_clean,
                fps_vision_list=fps_vision_list,
                fps_action_list=fps_action_list,
                force_action_tokens=force_action_tokens,
                step_counts=step_counts,
                allow_grad=allow_grad,
                collect_replay=collect_replay,
            )  # x0_chunk: [1,C,chunk_len,H,W]
            if collect_replay:
                if replay_xt is None or replay_sigma is None:
                    raise RuntimeError(f"Missing replay state for generated chunk [{chunk_start},{chunk_end}).")
                replay_frames.append(
                    _CausalARReplayFrame(
                        frame_idx=chunk_start,
                        xt=replay_xt,
                        sigma=replay_sigma,
                        action_latent=action_chunk.detach().clone() if action_chunk is not None else None,
                        text_tokens=chunk_text_tokens,
                        chunk_len=chunk_len,
                        cache_metadata=_SelfForcingCacheMetadata(cache_frame_idx=chunk_start),
                    )
                )
            # Split the joint chunk prediction back into per-frame outputs and seed each clean
            # frame into the rolling cache so the next chunk attends to finalized history.
            for offset, f in enumerate(chunk_frames):
                x0_frame = x0_chunk[:, :, offset : offset + 1]  # [1,C,1,H,W]
                frame_outputs.append(x0_frame)
                if f < num_frames - 1:
                    self._seed_self_forcing_history(
                        frame_latent=x0_frame,
                        action_latent=self._slice_action_for_frame(action_tokens, f, tcf),
                        frame_idx=f,
                        dual_kv_cache=dual_kv_cache,
                        text_tokens=text_tokens if f == 0 else None,
                        cached_text_offset=cached_text_offset,
                        gen_data_clean=gen_data_clean,
                        fps_vision_list=fps_vision_list,
                        fps_action_list=fps_action_list,
                        force_action_tokens=force_action_tokens,
                    )

        generated_vision = self._concat_vision_frames(frame_outputs)  # [1,C,T,H,W]
        timesteps_zero = torch.zeros((gen_data_clean.batch_size, 1), dtype=torch.float32)  # [B,1]
        packed_reference = self._pack_input_sequence(
            sequence_plans,
            input_text_indexes,
            gen_data_clean,
            timesteps_zero,
        )
        sigma_student = torch.full(
            (gen_data_clean.batch_size, 1),
            float(self.config.fixed_step_sampler_config.t_list[0]),
            **self.tensor_kwargs_fp32,
        )  # [B,1]
        gen_data_student = replace(
            gen_data_clean,
            x0_tokens_vision=[generated_vision],
            x0_tokens_sound=None,
            x0_tokens_action=gen_data_clean.x0_tokens_action if force_action_tokens else None,
        )
        return _SelfForcingSimulationResult(
            gen_data_student=gen_data_student,
            packed_reference=packed_reference,
            sigma_student=sigma_student,
            out_student=out_student,
            replay_state=(
                _CausalARReplayState(
                    frames=replay_frames,
                    dual_kv_cache=dual_kv_cache,
                    fps_vision_list=fps_vision_list,
                    fps_action_list=fps_action_list,
                    cached_text_offset=cached_text_offset,
                    force_action_tokens=force_action_tokens,
                )
                if collect_replay
                else None
            ),
        )
