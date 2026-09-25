# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch

from cosmos_framework.model.generator.mot.context_parallel_utils import context_parallel_broadcast_tensor_list
from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean
from cosmos_framework.data.generator.sequence_packing import PackedSequence, SequencePlan
from cosmos_framework.model.generator.utils.kv_cache import TeacherForcingMemoryState

if TYPE_CHECKING:
    from cosmos_framework.model.generator.omni_mot_causal_model import OmniMoTCausalModel


@dataclass(frozen=True)
class JointARChunk:
    """A complete training-time causal chunk on the shared RGB/LiDAR clock."""

    step: int
    vision_frames: tuple[int, ...]
    lidar_frames: tuple[int, ...]
    vision_prefix_end: int
    lidar_prefix_end: int


def joint_ar_chunks(
    *,
    vision_frames: int,
    lidar_frames: int,
    vision_seconds_per_frame: float,
    lidar_seconds_per_frame: float,
    frames_per_chunk: int,
) -> list[JointARChunk]:
    """Match the teacher-forcing singleton-zero and shared-time chunk boundaries."""
    if min(vision_frames, lidar_frames, frames_per_chunk) < 1:
        raise ValueError("Joint AR requires positive frame counts and chunk size")
    if not all(math.isfinite(value) and value > 0 for value in (vision_seconds_per_frame, lidar_seconds_per_frame)):
        raise ValueError("Joint AR requires finite positive sensor periods")
    chunk_seconds = frames_per_chunk * vision_seconds_per_frame
    vision_steps = [math.ceil(frame / frames_per_chunk - 1e-5) for frame in range(vision_frames)]
    lidar_steps = [math.ceil(frame * lidar_seconds_per_frame / chunk_seconds - 1e-5) for frame in range(lidar_frames)]
    chunks: list[JointARChunk] = []
    for step in sorted(set(vision_steps) | set(lidar_steps)):
        chunks.append(
            JointARChunk(
                step=step,
                vision_frames=tuple(frame for frame, value in enumerate(vision_steps) if value == step),
                lidar_frames=tuple(frame for frame, value in enumerate(lidar_steps) if value == step),
                vision_prefix_end=sum(value <= step for value in vision_steps),
                lidar_prefix_end=sum(value <= step for value in lidar_steps),
            )
        )
    return chunks


def _camera_prefix(
    latent: torch.Tensor,  # [1,C,V*T,H,W]
    num_views: int,
    end: int,
) -> torch.Tensor:  # [1,C,V*end,H,W]
    batch, channels, total_frames, height, width = latent.shape
    grid = latent.reshape(batch, channels, num_views, total_frames // num_views, height, width)  # [1,C,V,T,H,W]
    return grid[:, :, :, :end].reshape(batch, channels, num_views * end, height, width)  # [1,C,V*end,H,W]


def _conditioned_target(
    latent: torch.Tensor,  # [1,C,T,H,W]
    mask: torch.Tensor,  # [T,1,1]
) -> torch.Tensor:  # [1,C,T,H,W]
    """Discard every unconditioned target value before any model forward."""
    condition = mask.to(device=latent.device, dtype=torch.bool).reshape(1, 1, -1, 1, 1)  # [1,1,T,1,1]
    return torch.where(condition, latent, torch.zeros_like(latent))  # [1,C,T,H,W]


def _prefix_condition_count(mask: torch.Tensor, num_views: int) -> int:  # mask: [V*T,1,1]
    """Reject unsynchronized or non-prefix conditioning in this joint sampler."""
    grid = mask.to(dtype=torch.bool).reshape(num_views, -1)  # [V,T]
    if not torch.equal(grid, grid[:1].expand_as(grid)):
        raise ValueError("Joint AR currently requires the same conditioned prefix for every RGB view")
    count = int(grid[0].sum().item())
    expected = torch.arange(grid.shape[1], device=grid.device) < count  # [T]
    if not torch.equal(grid[0], expected):
        raise ValueError("Joint AR requires a contiguous conditioned target prefix")
    return count


def _prefix_data(
    data: GenerationDataClean,
    *,
    vision_target: torch.Tensor,  # [1,Cv,V*Tv,Hv,Wv]
    lidar_target: torch.Tensor,  # [1,Cl,Tl,Hl,Wl]
    num_views: int,
    chunk: JointARChunk,
) -> GenerationDataClean:
    """Keep complete sensor chunks, with no future target or control positions."""
    assert data.x0_tokens_vision is not None and data.x0_tokens_lidar is not None
    if data.temporal_positions_vision is not None:
        raise ValueError("Joint AR currently requires the training latent-index temporal positions")
    return replace(
        data,
        raw_state_vision=None,
        raw_state_lidar=None,
        x0_tokens_vision=[
            _camera_prefix(data.x0_tokens_vision[0], num_views, chunk.vision_prefix_end),  # [1,Cv,V*Tv_prefix,Hv,Wv]
            _camera_prefix(vision_target, num_views, chunk.vision_prefix_end),  # [1,Cv,V*Tv_prefix,Hv,Wv]
        ],
        x0_tokens_lidar=[
            data.x0_tokens_lidar[0][:, :, : chunk.lidar_prefix_end],  # [1,Cl,Tl_prefix,Hl,Wl]
            lidar_target[:, :, : chunk.lidar_prefix_end],  # [1,Cl,Tl_prefix,Hl,Wl]
        ],
    )


def _build_replay_branch(
    host: OmniMoTCausalModel,
    plans: list[SequencePlan],
    data: GenerationDataClean,
    text_tokens: list[list[int]],
) -> tuple[PackedSequence, TeacherForcingMemoryState]:
    """Use the actual joint training clean/noisy passes, including caption isolation."""
    pack = host._pack_input_sequence(plans, text_tokens, data, torch.zeros(1, dtype=torch.float32))  # timestep: [1]
    pack.to_cuda()
    host._cast_generated_tokens_to_precision(pack)
    host._validate_teacher_forcing_pack(pack)
    memory = host._build_clean_tf_cache(
        net=host.net,
        packed_sequence=pack,
        gen_data_clean=data,
        memory_info={},
        detach_clean_kv=True,
    )
    return pack, memory


def sample_joint_transfer_ar(
    host: OmniMoTCausalModel,
    *,
    plans: list[SequencePlan],
    data: GenerationDataClean,
    conditional_text: list[list[int]],
    unconditional_text: list[list[int]] | None,
    guidance: float,
    seed: int,
    num_steps: int,
    shift: float,
) -> dict[str, list[torch.Tensor]]:  # returns vision: [1,Cv,V*Tv,Hv,Wv], lidar: [1,Cl,Tl,Hl,Wl]
    """Joint AR reference sampling with training-time replay rebuilt at each chunk.

    Both targets are denoised together. Each clean replay contains only conditions
    and previously generated values; ungenerated targets are zero placeholders.
    The existing noisy-pass mask reads only strictly earlier clean chunks, and its
    current-target edges stay inside the current chunk. A complete prefix is used
    so controls can recompute their training-time history-dependent K/V. This is
    deliberately a separate inference path: it does not alter training or the
    optimized RGB-only AR cache.
    """
    if getattr(host.config, "fixed_step_sampler_config", None) is not None:
        raise ValueError("Joint AR supports RF checkpoints only; student checkpoints are unsupported.")
    if len(plans) != 1 or not plans[0].has_lidar or not plans[0].has_vision:
        raise ValueError("Joint AR requires exactly one RGB+LiDAR sequence")
    if plans[0].has_action or plans[0].has_sound:
        raise ValueError("Joint AR supports only RGB and LiDAR targets")
    if not host._uses_multiview_replay_kv() or host.config.compile.enabled:
        raise ValueError("Joint AR requires eager multiview Flex teacher forcing")
    if host.parallel_dims is not None and host.parallel_dims.cfgp_enabled:
        raise ValueError("Joint AR currently uses serial CFG branches; set cfg_parallel_shard_degree=1")
    policy = host._get_teacher_forcing_replay_policy()
    if policy.control_visibility != "causal":
        raise ValueError("Joint AR prefix replay requires causal controls")
    if data.num_vision_items_per_sample != [2] or data.num_lidar_items_per_sample != [2]:
        raise ValueError("Joint AR requires control and target items for each sensor modality")
    if data.x0_tokens_vision is None or data.x0_tokens_lidar is None:
        raise ValueError("Joint AR requires encoded camera and LiDAR streams")
    views = data.num_views_per_vision_item
    if views is None or len(views) != 2 or views[0] < 1 or views[0] != views[1]:
        raise ValueError("Joint AR requires matching per-camera VAE metadata")
    num_views = views[0]
    if len(data.x0_tokens_vision) != 2 or len(data.x0_tokens_lidar) != 2:
        raise ValueError("Joint AR requires exactly two encoded items per modality")
    for items in (data.x0_tokens_vision, data.x0_tokens_lidar):
        if items[0].ndim != 5 or items[0].shape[0] != 1 or items[0].shape != items[1].shape:
            raise ValueError("Joint AR requires aligned [1,C,T,H,W] control and target latents")
    if data.x0_tokens_vision[1].shape[2] % num_views:
        raise ValueError("Joint AR camera-major latent count must be divisible by its view count")
    if guidance != 1.0 and unconditional_text is None:
        raise ValueError("Joint AR guidance requires unconditional captions")
    if seed < 0 or num_steps < 1 or not math.isfinite(guidance) or not math.isfinite(shift) or shift <= 0:
        raise ValueError("Joint AR requires a nonnegative seed and finite valid sampling settings")

    # Rank-local VAE encoding can differ even with identical pixels and seeds.
    # CP shards must read one canonical set of controls and conditioned targets
    # before the initial pack and every subsequent clean/noisy replay.
    # Keep the existing Flex inference input behavior unchanged.
    if host._get_teacher_forcing_kv_implementation() == "multiview_maskless_kv":
        context_parallel_broadcast_tensor_list(data.x0_tokens_vision, host.parallel_dims)  # each [1,Cv,V*Tv,Hv,Wv]
        context_parallel_broadcast_tensor_list(data.x0_tokens_lidar, host.parallel_dims)  # each [1,Cl,Tl,Hl,Wl]

    initial_pack = host._pack_input_sequence(plans, conditional_text, data, torch.zeros(1))  # timestep: [1]
    if initial_pack.vision is None or initial_pack.lidar is None:
        raise ValueError("The joint inference pack omitted a required modality")
    # Keep clock and prefix helpers independent of full transformer construction.
    from cosmos_framework.model.generator.mot.causal_cosmos3_vfm_network import build_interactive_multiview_mask_items

    sensor_items = build_interactive_multiview_mask_items(initial_pack)
    if len(sensor_items) != 1 or len(sensor_items[0]) != 4:
        raise ValueError("Joint AR requires the training [RGB control, RGB target, LiDAR control, LiDAR target] layout")
    rgb_control, rgb_item, lidar_control, lidar_item = sensor_items[0]
    if not (rgb_control.is_control and lidar_control.is_control) or rgb_item.is_control or lidar_item.is_control:
        raise ValueError("Joint AR sensor control/target roles differ from the training layout")
    vision_condition = _prefix_condition_count(initial_pack.vision.condition_mask[1], num_views)
    lidar_condition = _prefix_condition_count(initial_pack.lidar.condition_mask[1], 1)
    vision_target = _conditioned_target(data.x0_tokens_vision[1], initial_pack.vision.condition_mask[1]).to(
        device=host.tensor_kwargs["device"], dtype=torch.float32
    )  # [1,Cv,V*Tv,Hv,Wv]
    lidar_target = _conditioned_target(data.x0_tokens_lidar[1], initial_pack.lidar.condition_mask[1]).to(
        device=host.tensor_kwargs["device"], dtype=torch.float32
    )  # [1,Cl,Tl,Hl,Wl]
    frames_per_view = vision_target.shape[2] // num_views
    chunks = joint_ar_chunks(
        vision_frames=frames_per_view,
        lidar_frames=lidar_target.shape[2],
        vision_seconds_per_frame=rgb_item.seconds_per_frame,
        lidar_seconds_per_frame=lidar_item.seconds_per_frame,
        frames_per_chunk=int(host.config.teacher_forcing_frames_per_chunk),
    )
    del initial_pack
    for chunk in chunks:
        _sample_joint_chunk(
            host,
            plans=plans,
            data=data,
            conditional_text=conditional_text,
            unconditional_text=unconditional_text,
            guidance=guidance,
            seed=seed,
            num_steps=num_steps,
            shift=shift,
            vision_target=vision_target,
            lidar_target=lidar_target,
            num_views=num_views,
            frames_per_view=frames_per_view,
            vision_condition=vision_condition,
            lidar_condition=lidar_condition,
            chunk=chunk,
        )
    return {"vision": [vision_target], "lidar": [lidar_target]}


def _sample_joint_chunk(
    host: OmniMoTCausalModel,
    *,
    plans: list[SequencePlan],
    data: GenerationDataClean,
    conditional_text: list[list[int]],
    unconditional_text: list[list[int]] | None,
    guidance: float,
    seed: int,
    num_steps: int,
    shift: float,
    vision_target: torch.Tensor,  # [1,Cv,V*Tv,Hv,Wv]
    lidar_target: torch.Tensor,  # [1,Cl,Tl,Hl,Wl]
    num_views: int,
    frames_per_view: int,
    vision_condition: int,
    lidar_condition: int,
    chunk: JointARChunk,
) -> None:
    """Denoise and commit one chunk, releasing its branch caches on return."""
    vision_frames = [frame for frame in chunk.vision_frames if frame >= vision_condition]
    lidar_frames = [frame for frame in chunk.lidar_frames if frame >= lidar_condition]
    if not vision_frames and not lidar_frames:
        return
    prefix = _prefix_data(
        data, vision_target=vision_target, lidar_target=lidar_target, num_views=num_views, chunk=chunk
    )
    assert prefix.x0_tokens_vision is not None and prefix.x0_tokens_lidar is not None
    vision_indexes = [view * chunk.vision_prefix_end + frame for view in range(num_views) for frame in vision_frames]
    final_vision_indexes = [view * frames_per_view + frame for view in range(num_views) for frame in vision_frames]
    vision_shape = (1, vision_target.shape[1], len(vision_indexes), *vision_target.shape[-2:])
    lidar_shape = (1, lidar_target.shape[1], len(lidar_frames), *lidar_target.shape[-2:])
    vision_size = math.prod(vision_shape)
    lidar_size = math.prod(lidar_shape)
    generator = torch.Generator(device=vision_target.device).manual_seed(seed + chunk.step)
    initial_noise = torch.randn(
        (1, vision_size + lidar_size), device=vision_target.device, dtype=torch.float32, generator=generator
    )  # [1,D_vision+D_lidar]
    cond_pack, cond_memory = _build_replay_branch(host, plans, prefix, conditional_text)
    uncond_branch = (
        _build_replay_branch(host, plans, prefix, unconditional_text)
        if guidance != 1.0 and unconditional_text is not None
        else None
    )

    def branch_velocity(
        pack: PackedSequence,
        memory: TeacherForcingMemoryState,
        noise: torch.Tensor,  # [1,D_vision+D_lidar]
        timestep: torch.Tensor,  # [1,1]
    ) -> torch.Tensor:  # [1,D_vision+D_lidar]
        assert pack.vision is not None and pack.lidar is not None
        noisy_vision = prefix.x0_tokens_vision[1].clone()  # [1,Cv,V*Tv_prefix,Hv,Wv]
        noisy_lidar = prefix.x0_tokens_lidar[1].clone()  # [1,Cl,Tl_prefix,Hl,Wl]
        noisy_vision[:, :, vision_indexes] = noise[:, :vision_size].reshape(vision_shape)  # [1,Cv,V*Tv_chunk,Hv,Wv]
        noisy_lidar[:, :, lidar_frames] = noise[:, vision_size:].reshape(lidar_shape)  # [1,Cl,Tl_chunk,Hl,Wl]
        host._update_inference_pack_template(
            pack,
            [prefix.x0_tokens_vision[0], noisy_vision],  # list of [1,Cv,V*Tv_prefix,Hv,Wv]
            [prefix.x0_tokens_lidar[0], noisy_lidar],  # list of [1,Cl,Tl_prefix,Hl,Wl]
            None,
            None,
            timestep,
        )
        prediction = host.denoise(data_batch_packed=pack, memory=memory)
        vision_velocity = prediction["preds_vision"][1]  # [1,Cv,V*Tv_prefix,Hv,Wv]
        lidar_velocity = prediction["preds_lidar"][1]  # [1,Cl,Tl_prefix,Hl,Wl]
        if vision_velocity.shape != noisy_vision.shape or lidar_velocity.shape != noisy_lidar.shape:
            raise ValueError("Joint model predictions must retain each input item's [1,C,T,H,W] shape")
        # Grid-stream decoding retains its singleton batch dimension. Select time,
        # preserving every channel before flattening for the shared RF solver.
        return torch.cat(
            [vision_velocity[:, :, vision_indexes].reshape(1, -1), lidar_velocity[:, :, lidar_frames].reshape(1, -1)],
            dim=1,
        )  # [1,D_vision+D_lidar]

    def velocity(noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:  # [1,D], [1,1] -> [1,D]
        conditional = branch_velocity(cond_pack, cond_memory, noise, timestep)  # [1,D]
        if uncond_branch is None:
            return conditional  # [1,D]
        unconditional = branch_velocity(*uncond_branch, noise, timestep)  # [1,D]
        return unconditional + guidance * (conditional - unconditional)  # [1,D]

    denoised = host._run_ar_sampler(
        velocity,
        initial_noise,
        sampler_mode="rf",
        num_steps=num_steps,
        shift=shift,
        seed=seed,
        sample_idx=chunk.step,
        num_frames=frames_per_view,
        distilled_num_steps=None,
    )  # [1,D_vision+D_lidar]
    if not torch.isfinite(denoised).all():
        raise FloatingPointError(f"Joint AR produced nonfinite latents in causal chunk {chunk.step}")
    vision_target[:, :, final_vision_indexes] = denoised[:, :vision_size].reshape(
        vision_shape
    )  # [1,Cv,V*Tv_chunk,Hv,Wv]
    lidar_target[:, :, lidar_frames] = denoised[:, vision_size:].reshape(lidar_shape)  # [1,Cl,Tl_chunk,Hl,Wl]
