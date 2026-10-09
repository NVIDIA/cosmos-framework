# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Opt-in incremental RGB and joint RGB/LiDAR inference with bounded replay KV."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from dataclasses import replace
from typing import TYPE_CHECKING, Literal

import torch

from cosmos_framework.utils import log
from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean
from cosmos_framework.data.generator.sequence_packing import PackedSequence, SequencePlan
from cosmos_framework.utils.generator.multiview import slice_camera_major_frames
from cosmos_framework.model.generator.teacher_forcing import mark_modality_as_clean_condition
from cosmos_framework.model.generator.utils.multiview_ar import JointARChunk, joint_ar_chunks
from cosmos_framework.model.generator.utils.rolling_kv.rolling_prompt import (
    RollingPromptPack,
    RollingPromptSchedule,
    RollingPromptSegment,
    RollingTextSinkLayer,
)
from cosmos_framework.model.generator.utils.rolling_kv.rolling_replay import RollingReplayCache, RollingReplayRequest
from cosmos_framework.model.generator.utils.rolling_kv.rolling_sink_rope import prepare_media_sink_key_rotation

if TYPE_CHECKING:
    from cosmos_framework.model.generator.omni_mot_causal_model import OmniMoTCausalModel


def _initial_target(
    target: torch.Tensor,
    num_views: int,
    condition_count: int,
    device: torch.device,  # target: [1,C,V*T,H,W]
) -> torch.Tensor:  # [1,C,V*T,H,W]
    """Read only explicitly observed targets, never future ground truth."""
    result = torch.zeros(target.shape, device=device, dtype=torch.float32)  # [1,C,V*T,H,W]
    if condition_count:
        frames = target.shape[2] // num_views
        indexes = [view * frames for view in range(num_views)]
        result[:, :, indexes] = target[:, :, indexes].to(device=device, dtype=torch.float32)  # [1,C,V,H,W]
    return result


def _chunk_data(
    data: GenerationDataClean,
    vision: torch.Tensor,  # [1,Cv,V*Tv,Hv,Wv]
    lidar: torch.Tensor | None,  # [1,Cl,Tl,Hl,Wl] or None
    num_views: int,
    chunk: JointARChunk,
) -> GenerationDataClean:
    assert data.x0_tokens_vision is not None
    start, stop = chunk.vision_frames[0], chunk.vision_frames[-1] + 1
    lidar_items = None
    if lidar is not None:
        assert data.x0_tokens_lidar is not None
        lidar_start, lidar_stop = chunk.lidar_frames[0], chunk.lidar_frames[-1] + 1
        lidar_items = [
            data.x0_tokens_lidar[0][:, :, lidar_start:lidar_stop],
            lidar[:, :, lidar_start:lidar_stop],
        ]  # each [1,Cl,Tlc,Hl,Wl]
    return replace(
        data,
        raw_state_vision=None,
        raw_state_lidar=None,
        x0_tokens_vision=[
            slice_camera_major_frames(data.x0_tokens_vision[0], num_views, start, stop - start),
            slice_camera_major_frames(vision, num_views, start, stop - start),
        ],  # each [1,Cv,V*Tvc,Hv,Wv]
        x0_tokens_lidar=lidar_items,
    )


def shift_rolling_positions(
    pack: PackedSequence,
    *,
    vision_start: int,
    lidar_start: int,
    enable_fps_modulation: bool,
    base_fps: float,
    vision_temporal_compression: int,
    temporal_scale: float = 1.0,
) -> None:
    """Offset sensor RoPE only; caption positions and spatial grids stay fixed.

    The caller requires shared control/target vision positions, so each sensor
    starts at the caption boundary regardless of the local chunk length. Both
    sensor RoPE grids use the vision base temporal compression, as in training.
    """
    for modality, start in ((pack.vision, vision_start), (pack.lidar, lidar_start)):
        if modality is None:
            continue
        if not modality.seconds_per_frame or len(set(modality.seconds_per_frame)) != 1:
            raise ValueError("Rolling replay requires one sensor period per modality.")
        scale = modality.seconds_per_frame[0] * base_fps / vision_temporal_compression if enable_fps_modulation else 1
        if temporal_scale == 1.0:
            pack.position_ids[0, modality.sequence_indexes] += start * scale  # [S_modality]
            continue
        if not pack.position_ids.is_floating_point():
            pack.position_ids = pack.position_ids.float()  # [3,S]
        local_time = pack.position_ids[0, modality.sequence_indexes]  # [S_modality]
        origin = local_time.min()  # []
        pack.position_ids[0, modality.sequence_indexes] = (
            origin + (local_time - origin + start * scale) * temporal_scale
        )  # [S_modality]


def _build_branch_pack(
    host: OmniMoTCausalModel,
    *,
    plan: SequencePlan,
    data: GenerationDataClean,
    text: list[list[int]],
    chunk: JointARChunk,
    cache: RollingReplayCache,
    clean: bool,
    prompt: RollingPromptPack | None = None,
) -> tuple[PackedSequence, RollingReplayRequest]:
    pack = host._pack_input_sequence([plan], text, data, torch.zeros(1))  # timestep: [1]
    expert = host.config.diffusion_expert_config
    shift_rolling_positions(
        pack,
        vision_start=chunk.vision_frames[0],
        lidar_start=chunk.lidar_frames[0] if plan.has_lidar else 0,
        enable_fps_modulation=expert.enable_fps_modulation,
        base_fps=float(expert.base_fps),
        vision_temporal_compression=host.tokenizer_vision_gen.temporal_compression_factor,
        temporal_scale=getattr(host.config, "rolling_media_rope_scale", 1.0),
    )
    pack.to_cuda()
    age_cap = getattr(host.config, "rolling_media_sink_age_cap_seconds", None)
    cache.sink_key_rotation = None
    if age_cap is not None:
        assert pack.vision is not None
        starts_seconds = [chunk.vision_frames[0] * pack.vision.seconds_per_frame[0]]
        if pack.lidar is not None:
            starts_seconds.append(chunk.lidar_frames[0] * pack.lidar.seconds_per_frame[0])
        cache.sink_key_rotation = prepare_media_sink_key_rotation(
            host.net.language_model.model.rotary_emb,
            token_count=cache.initial_sink_token_count,
            chunk_start_seconds=min(starts_seconds),
            age_cap_seconds=age_cap,
            temporal_units_per_second=(
                float(expert.base_fps)
                / host.tokenizer_vision_gen.temporal_compression_factor
                * getattr(host.config, "rolling_media_rope_scale", 1.0)
            ),
        )
    cache.prompt_layout = None
    if prompt is not None:
        assert pack.vision is not None
        vision_period = pack.vision.seconds_per_frame[0]
        temporal_units = (
            float(expert.base_fps) / host.tokenizer_vision_gen.temporal_compression_factor
            if expert.enable_fps_modulation
            else 1.0 / vision_period
        )
        cache.prompt_layout = prompt.prepare(
            pack,
            position_mode=getattr(host.config, "rolling_text_position_mode", "fixed"),
            temporal_units_per_second=temporal_units * getattr(host.config, "rolling_media_rope_scale", 1.0),
            chunk_start_seconds=chunk.vision_frames[0] * vision_period,
        )
    host._cast_generated_tokens_to_precision(pack)
    host._validate_teacher_forcing_pack(pack)
    condition_masks = [
        mask.clone()
        for modality in (pack.vision, pack.lidar)
        if modality is not None
        for mask in modality.condition_mask
    ]  # each [V*T,1,1]
    starts = (chunk.vision_frames[0],) * 2
    if plan.has_lidar:
        starts += (chunk.lidar_frames[0],) * 2
    request = RollingReplayRequest(
        step=chunk.step,
        frame_starts=starts,
        frames_per_chunk=int(host.config.teacher_forcing_frames_per_chunk),
        pass_kind="clean" if clean else "noisy",
        condition_masks=condition_masks,
        history=cache.metadata,
        prompt_layout=cache.prompt_layout,
    )
    if clean:
        for modality in (pack.vision, pack.lidar):
            if modality is not None:
                mark_modality_as_clean_condition(modality)
    setattr(pack, "rolling_replay_request", request)
    return pack, request


def _validate_rolling_transfer(
    host: OmniMoTCausalModel,
    plans: list[SequencePlan],
    data: GenerationDataClean,
    sampler_mode: str,
) -> None:
    if len(plans) != 1 or not plans[0].has_vision or any((plans[0].has_action, plans[0].has_sound, plans[0].has_radar)):
        raise ValueError("Rolling replay requires one RGB or RGB/LiDAR transfer sample.")
    if not host._uses_multiview_replay_kv() or host.config.compile.enabled:
        raise ValueError("Rolling replay requires eager multiview teacher forcing.")
    if host.parallel_dims is not None and host.parallel_dims.cfgp_enabled:
        raise ValueError("Rolling replay uses serial CFG; set cfg_parallel_shard_degree=1.")
    if host._get_teacher_forcing_replay_policy().control_visibility != "causal":
        raise ValueError("Rolling replay requires causal control visibility.")
    if (
        host.config.kv_cache_dtype is not None
        or host.config.kv_cache_inference_size is not None
        or host.config.attention_sink_size
    ):
        raise ValueError(
            "Use rolling_kv_cache_chunks/rolling_kv_sink_chunks without framewise or quantized KV options."
        )
    groups = plans[0].vision_temporal_position_groups
    shared_grid = plans[0].share_vision_temporal_positions and groups is None
    # The native joint input builder expresses the same shared RGB clock with
    # one temporal group instead of the legacy all-items shared-grid flag.
    shared_group = (
        not plans[0].share_vision_temporal_positions
        and groups is not None
        and len(groups) == 2
        and groups[0] is not None
        and groups[0] == groups[1]
    )
    if data.temporal_positions_vision is not None or not (shared_grid or shared_group):
        raise ValueError("Rolling transfer requires shared latent-index control/target positions.")
    if plans[0].condition_view_indexes_vision or any(
        indexes not in ([], [0])
        for indexes in (plans[0].condition_frame_indexes_vision, plans[0].condition_frame_indexes_lidar)
    ):
        raise ValueError(
            "Rolling replay currently supports zero or one initial target frame per sensor, in every view."
        )
    if sampler_mode not in ("rf", "distilled"):
        raise ValueError("Rolling replay sampler_mode must be rf or distilled.")
    student = host.config.fixed_step_sampler_config is not None
    if student != (sampler_mode == "distilled"):
        raise ValueError("Rolling replay sampler_mode must match the checkpoint's fixed-step sampler configuration.")
    if data.num_vision_items_per_sample != [2] or data.x0_tokens_vision is None or len(data.x0_tokens_vision) != 2:
        raise ValueError("Rolling replay requires RGB control and target items.")
    views = data.num_views_per_vision_item
    if views is None or len(views) != 2 or views[0] < 1 or views[0] != views[1]:
        raise ValueError("Rolling replay requires matching camera-major view counts.")
    if plans[0].has_lidar and (
        data.num_lidar_items_per_sample != [2] or data.x0_tokens_lidar is None or len(data.x0_tokens_lidar) != 2
    ):
        raise ValueError("Rolling joint replay requires LiDAR control and target items.")
    for items in (data.x0_tokens_vision, data.x0_tokens_lidar if plans[0].has_lidar else None):
        if items is not None and (items[0].ndim != 5 or items[0].shape[0] != 1 or items[0].shape != items[1].shape):
            raise ValueError("Rolling replay requires aligned [1,C,T,H,W] control/target latents.")
    if data.x0_tokens_vision[0].shape[2] % views[0]:
        raise ValueError("Rolling replay camera latents must be divisible by the view count.")


@torch.no_grad()
def iter_rolling_transfer_ar(
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
    sampler_mode: Literal["rf", "distilled"],
    distilled_num_steps: int | None = None,
    output_frames: int | None = None,
    on_cache_update: Callable[[RollingReplayCache], None] | None = None,
    on_clean_vision_chunk: Callable[[torch.Tensor], None] | None = None,
    prompt_schedule: RollingPromptSchedule | None = None,
) -> Iterator[dict[str, torch.Tensor]]:  # each vision [1,Cv,V*Tvc,Hv,Wv], optional lidar [1,Cl,Tlc,Hl,Wl]
    """Notify the decoder, refresh KV for later chunks, then yield the current chunk.

    Only the current chunk is packed or forwarded. Encoded input controls and
    the output backing latents still scale with the input clip; this bounds the
    transformer cache and query work, not VAE/media memory or rollout quality.
    Input canonicalization and seed broadcast belong to the public AR entry.
    """
    _validate_rolling_transfer(host, plans, data, sampler_mode)
    policy = getattr(host.config, "rolling_prompt_policy", None)
    if policy not in (None, "segmented") or (policy is None and prompt_schedule is not None):
        raise ValueError("Segmented prompts require rolling_prompt_policy='segmented'.")
    rope_scale = getattr(host.config, "rolling_media_rope_scale", 1.0)
    if not math.isfinite(rope_scale) or rope_scale <= 0 or (policy is None and rope_scale != 1.0):
        raise ValueError("A positive temporal RoPE scale is supported only with segmented prompts.")
    age_cap = getattr(host.config, "rolling_media_sink_age_cap_seconds", None)
    if age_cap is not None and (
        isinstance(age_cap, bool)
        or not math.isfinite(age_cap)
        or age_cap <= 0
        or policy != "segmented"
        or host.config.rolling_kv_sink_chunks != 1
        or not host.config.diffusion_expert_config.enable_fps_modulation
    ):
        raise ValueError(
            "Media sink age capping requires segmented prompts, one initial sink, FPS modulation and a positive age."
        )
    text_position_mode = getattr(host.config, "rolling_text_position_mode", "fixed")
    if text_position_mode not in ("fixed", "segment_start", "chunk_start") or (
        policy is None and text_position_mode != "fixed"
    ):
        raise ValueError("Nonfixed text temporal positioning requires segmented prompts and a supported position mode.")
    if seed < 0 or num_steps < 1 or not math.isfinite(guidance) or not math.isfinite(shift) or shift <= 0:
        raise ValueError("Rolling replay requires finite valid sampling settings.")
    if guidance != 1.0 and unconditional_text is None:
        raise ValueError("Rolling replay CFG requires unconditional captions.")
    assert data.x0_tokens_vision is not None and data.num_views_per_vision_item is not None
    plan, num_views = plans[0], data.num_views_per_vision_item[0]
    if policy == "segmented":
        if plan.text_view_ids is None:
            if len(conditional_text) != 1 or (unconditional_text is not None and len(unconditional_text) != 1):
                raise ValueError("A shared prompt must have one text stream.")
            conditional_text = conditional_text * num_views
            unconditional_text = unconditional_text * num_views if unconditional_text is not None else None
            plan = replace(plan, text_view_ids=list(range(num_views)))
            plans = [plan]
        if plan.text_view_ids != list(range(num_views)) or len(conditional_text) != num_views:
            raise ValueError("Segmented prompting requires camera-ordered text streams.")
    if prompt_schedule is not None and (
        len(prompt_schedule.views) != num_views or plan.text_view_ids != list(range(num_views))
    ):
        raise ValueError("Segmented prompting requires one scheduled caption stream per camera, in native view order.")
    device = torch.device(host.tensor_kwargs["device"])
    vision_condition, lidar_condition = (
        len(plan.condition_frame_indexes_vision),
        len(plan.condition_frame_indexes_lidar),
    )
    vision = _initial_target(data.x0_tokens_vision[1], num_views, vision_condition, device)  # [1,Cv,V*Tv,Hv,Wv]
    lidar = (
        _initial_target(data.x0_tokens_lidar[1], 1, lidar_condition, device)
        if plan.has_lidar and data.x0_tokens_lidar is not None
        else None
    )  # [1,Cl,Tl,Hl,Wl] or None
    total_frames = vision.shape[2] // num_views
    frame_count = total_frames if output_frames is None else min(output_frames, total_frames)
    # A singleton probe resolves the same sensor periods and patch geometry as
    # production packing, without materializing an all-horizon token pack.
    first = JointARChunk(0, (0,), (0,), 1, 1)
    probe = host._pack_input_sequence(
        plans, conditional_text, _chunk_data(data, vision, lidar, num_views, first), torch.zeros(1)
    )  # timestep: [1]
    assert probe.vision is not None
    vision_period = probe.vision.seconds_per_frame[0]
    lidar_period = probe.lidar.seconds_per_frame[0] if probe.lidar is not None else vision_period
    source_fps = host.tokenizer_vision_gen.temporal_compression_factor / vision_period
    if prompt_schedule is not None and not math.isclose(prompt_schedule.fps, source_fps, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f"Prompt schedule FPS {prompt_schedule.fps} does not match the source clock {source_fps}.")
    negative_schedule = None
    if policy == "segmented":
        if prompt_schedule is None:
            # Native evaluation callbacks may supply one static caption per
            # camera. Give it the actual generated horizon, without requiring
            # a separate schedule file or changing the input media length.
            fps = source_fps
            num_frames = round((frame_count - 1) * vision_period * fps) + 1
            prompt_schedule = RollingPromptSchedule(
                fps=fps,
                num_frames=num_frames,
                views=tuple((RollingPromptSegment(0, num_frames, tuple(tokens)),) for tokens in conditional_text),
                sink_tokens=host.config.rolling_text_sink_tokens,
                special_tokens=host.llm_special_tokens,
                context_before=host.config.rolling_prompt_context_before,
                context_after=host.config.rolling_prompt_context_after,
                media_text_extent=host.config.rolling_media_text_extent,
            )
        if guidance != 1.0:
            assert unconditional_text is not None
            if len(unconditional_text) != num_views:
                raise ValueError("Segmented prompting requires one unconditional text stream per camera.")
            # RF and distilled samplers share serial CFG. The negative branch owns separate
            # KV and a static negative caption; it never reads positive sinks.
            negative_schedule = RollingPromptSchedule(
                fps=prompt_schedule.fps,
                num_frames=prompt_schedule.num_frames,
                views=tuple(
                    (RollingPromptSegment(0, prompt_schedule.num_frames, tuple(tokens)),)
                    for tokens in unconditional_text
                ),
                sink_tokens=0,
                special_tokens=host.llm_special_tokens,
            )
    chunk_size = int(host.config.teacher_forcing_frames_per_chunk)
    chunks = joint_ar_chunks(
        vision_frames=frame_count,
        lidar_frames=lidar.shape[2] if lidar is not None else 1,
        vision_seconds_per_frame=vision_period,
        lidar_seconds_per_frame=lidar_period,
        frames_per_chunk=chunk_size,
    )
    if any(not chunk.vision_frames or (lidar is not None and not chunk.lidar_frames) for chunk in chunks):
        raise ValueError("Rolling joint replay requires an aligned horizon with both sensors in every chunk.")
    if prompt_schedule is not None:
        last_timestamp = max(
            (frame_count - 1) * vision_period,
            chunks[-1].lidar_frames[-1] * lidar_period if lidar is not None else 0.0,
        )
        # Validate the complete finite horizon before forwarding its first chunk.
        # Longer schedules may still serve a requested prefix of the input clip.
        prompt_schedule.for_times((last_timestamp,))
    max_vision_frames = max(len(chunk.vision_frames) for chunk in chunks)
    token_capacity = sum(math.prod(shape[1:]) * num_views * max_vision_frames for shape in probe.vision.token_shapes)
    if probe.lidar is not None:
        token_capacity += sum(
            math.prod(shape[1:]) * max(len(chunk.lidar_frames) for chunk in chunks)
            for shape in probe.lidar.token_shapes
        )
    flex_backend = getattr(host.net, "flex_backend", None)
    if flex_backend is not None and not getattr(host.net, "teacher_forcing_maskless", False):
        # Flex appends every reserved history slot to the aligned current stream.
        # Pad storage only; cache writes and metadata retain the real token counts.
        kv_alignment = int(flex_backend.block_size[1])
        token_capacity = ((token_capacity + kv_alignment - 1) // kv_alignment) * kv_alignment
    del probe
    cache_chunks = host.config.rolling_kv_cache_chunks
    assert cache_chunks is not None
    texts = [conditional_text] if guidance == 1.0 else [conditional_text, unconditional_text]
    caches = [
        RollingReplayCache(
            num_layers=int(host.net.num_hidden_layers),
            cache_chunks=cache_chunks,
            sink_chunks=host.config.rolling_kv_sink_chunks,
            tokens_per_chunk=token_capacity,
        )
        for _ in texts
    ]
    if prompt_schedule is not None:
        for cache in caches:
            cache.text_sink_layers = [RollingTextSinkLayer() for _ in cache.cache]
    log.info(
        f"Rolling replay: {cache_chunks} history chunks, {host.config.rolling_kv_sink_chunks} sinks, {caches[0].capacity} KV tokens/layer/branch."
    )
    for chunk in chunks:
        prompts: list[RollingPromptPack | None] = [None] * len(caches)
        if prompt_schedule is not None:
            timestamps = tuple(frame * vision_period for frame in chunk.vision_frames)
            if lidar is not None:
                timestamps += tuple(frame * lidar_period for frame in chunk.lidar_frames)
            prompts = [prompt_schedule.for_times(timestamps)]
            if negative_schedule is not None:
                prompts.append(negative_schedule.for_times(timestamps))
            texts = [prompt.text_tokens for prompt in prompts if prompt is not None]
        chunk_plan = replace(
            plan,
            condition_frame_indexes_vision=[0] if vision_condition and chunk.step == 0 else [],
            condition_frame_indexes_lidar=[0] if lidar_condition and chunk.step == 0 else [],
        )
        current = _chunk_data(data, vision, lidar, num_views, chunk)
        assert current.x0_tokens_vision is not None
        vision_indexes = [
            view * len(chunk.vision_frames) + index
            for view in range(num_views)
            for index, frame in enumerate(chunk.vision_frames)
            if frame >= vision_condition
        ]
        lidar_indexes = (
            [index for index, frame in enumerate(chunk.lidar_frames) if frame >= lidar_condition]
            if lidar is not None
            else []
        )
        vision_shape = (1, vision.shape[1], len(vision_indexes), *vision.shape[-2:])
        lidar_shape = (1, lidar.shape[1], len(lidar_indexes), *lidar.shape[-2:]) if lidar is not None else (1, 0)
        vision_size, lidar_size = math.prod(vision_shape), math.prod(lidar_shape)
        if vision_size + lidar_size:
            # Match the reference joint seed clock; preserve RGB's frame-start
            # seed convention for its separate pre-existing public entry.
            sample_index = chunk.step if lidar is not None else chunk.vision_frames[0]
            generator = torch.Generator(device=device).manual_seed(seed + sample_index)
            noise = torch.randn(
                (1, vision_size + lidar_size), device=device, dtype=torch.float32, generator=generator
            )  # [1,Dv+Dl]
            branches = [
                _build_branch_pack(
                    host, plan=chunk_plan, data=current, text=text, chunk=chunk, cache=cache, clean=False, prompt=prompt
                )[0]
                for text, cache, prompt in zip(texts, caches, prompts, strict=True)
                if text is not None
            ]

            def velocity(noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:  # [1,Dv+Dl], [1,1] -> [1,Dv+Dl]
                predictions: list[torch.Tensor] = []  # each [1,Dv+Dl]
                for pack, cache in zip(branches, caches, strict=True):
                    assert current.x0_tokens_vision is not None
                    noisy_vision = current.x0_tokens_vision[1].clone()  # [1,Cv,V*Tvc,Hv,Wv]
                    noisy_vision[:, :, vision_indexes] = noise[:, :vision_size].reshape(
                        vision_shape
                    )  # [1,Cv,V*Tvc_noisy,Hv,Wv]
                    lidar_items = None
                    if current.x0_tokens_lidar is not None:
                        noisy_lidar = current.x0_tokens_lidar[1].clone()  # [1,Cl,Tlc,Hl,Wl]
                        noisy_lidar[:, :, lidar_indexes] = noise[:, vision_size:].reshape(
                            lidar_shape
                        )  # [1,Cl,Tlc_noisy,Hl,Wl]
                        lidar_items = [current.x0_tokens_lidar[0], noisy_lidar]
                    host._update_inference_pack_template(
                        packed_sequence=pack,
                        noise_x_vision=[current.x0_tokens_vision[0], noisy_vision],
                        noise_x_lidar=lidar_items,
                        noise_x_radar=None,
                        noise_x_action=None,
                        noise_x_sound=None,
                        timestep=timestep,  # [1,1]
                    )
                    output = host.denoise(data_batch_packed=pack, memory=cache.read())
                    values = [output["preds_vision"][1][:, :, vision_indexes].reshape(1, -1)]  # each [1,Dv]
                    if lidar is not None:
                        values.append(output["preds_lidar"][1][:, :, lidar_indexes].reshape(1, -1))  # [1,Dl]
                    predictions.append(torch.cat(values, dim=1))  # [1,Dv+Dl]
                if len(predictions) == 1:
                    return predictions[0]  # [1,Dv+Dl]
                return predictions[1] + guidance * (predictions[0] - predictions[1])  # [1,Dv+Dl]

            denoised = host._run_ar_sampler(
                velocity,
                noise,
                sampler_mode=sampler_mode,
                num_steps=num_steps,
                shift=shift,
                seed=seed,
                sample_idx=sample_index,
                num_frames=frame_count,
                distilled_num_steps=distilled_num_steps,
                schedule_frame_idx=chunk.vision_frames[0],
            )  # [1,Dv+Dl]
            if not torch.isfinite(denoised).all():
                raise FloatingPointError(f"Rolling replay produced nonfinite latents in chunk {chunk.step}.")
            final_vision_indexes = [
                view * total_frames + frame
                for view in range(num_views)
                for frame in chunk.vision_frames
                if frame >= vision_condition
            ]
            vision[:, :, final_vision_indexes] = denoised[:, :vision_size].reshape(
                vision_shape
            )  # [1,Cv,V*Tvc_noisy,Hv,Wv]
            if lidar is not None:
                final_lidar_indexes = [frame for frame in chunk.lidar_frames if frame >= lidar_condition]
                lidar[:, :, final_lidar_indexes] = denoised[:, vision_size:].reshape(
                    lidar_shape
                )  # [1,Cl,Tlc_noisy,Hl,Wl]
            del branches
        clean_data = _chunk_data(data, vision, lidar, num_views, chunk)
        assert clean_data.x0_tokens_vision is not None
        if on_clean_vision_chunk is not None:
            on_clean_vision_chunk(clean_data.x0_tokens_vision[1])
        if chunk.step != chunks[-1].step:
            for text, cache, prompt in zip(texts, caches, prompts, strict=True):
                assert text is not None
                clean_pack, request = _build_branch_pack(
                    host,
                    plan=chunk_plan,
                    data=clean_data,
                    text=text,
                    chunk=chunk,
                    cache=cache,
                    clean=True,
                    prompt=prompt,
                )
                token_count = sum(
                    math.prod(shape)
                    for modality in (clean_pack.vision, clean_pack.lidar)
                    if modality is not None
                    for shape in modality.token_shapes
                )
                host.denoise(
                    data_batch_packed=clean_pack,
                    memory=cache.writer(step=chunk.step, token_count=token_count, device=device),
                )
                cache.commit(step=chunk.step, token_count=token_count, request=request)
                del clean_pack, request
            if on_cache_update is not None:
                on_cache_update(caches[0])
        assert clean_data.x0_tokens_vision is not None
        result = {"vision": clean_data.x0_tokens_vision[1]}  # [1,Cv,V*Tvc,Hv,Wv]
        if clean_data.x0_tokens_lidar is not None:
            result["lidar"] = clean_data.x0_tokens_lidar[1]  # [1,Cl,Tlc,Hl,Wl]
        yield result


def sample_rolling_joint_transfer_ar(
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
    sampler_mode: Literal["rf", "distilled"],
    distilled_num_steps: int | None,
    prompt_schedule: RollingPromptSchedule | None = None,
) -> dict[str, list[torch.Tensor]]:  # vision [1,Cv,V*Tv,Hv,Wv], lidar [1,Cl,Tl,Hl,Wl]
    """Collect chunks in the existing joint API's camera-major output layout."""
    assert data.num_views_per_vision_item is not None
    views = data.num_views_per_vision_item[0]
    rgb_parts: list[torch.Tensor] = []  # each [1,Cv,V,Tvc,Hv,Wv]
    lidar_parts: list[torch.Tensor] = []  # each [1,Cl,Tlc,Hl,Wl]
    for part in iter_rolling_transfer_ar(
        host,
        plans=plans,
        data=data,
        conditional_text=conditional_text,
        unconditional_text=unconditional_text,
        guidance=guidance,
        seed=seed,
        num_steps=num_steps,
        shift=shift,
        sampler_mode=sampler_mode,
        distilled_num_steps=distilled_num_steps,
        prompt_schedule=prompt_schedule,
    ):
        rgb = part["vision"]  # [1,Cv,V*Tvc,Hv,Wv]
        rgb_parts.append(rgb.reshape(1, rgb.shape[1], views, -1, *rgb.shape[-2:]))  # [1,Cv,V,Tvc,Hv,Wv]
        lidar_parts.append(part["lidar"])  # [1,Cl,Tlc,Hl,Wl]
    vision = torch.cat(rgb_parts, dim=3).flatten(2, 3)  # [1,Cv,V*Tv,Hv,Wv]
    return {"vision": [vision], "lidar": [torch.cat(lidar_parts, dim=2)]}  # lidar [1,Cl,Tl,Hl,Wl]
