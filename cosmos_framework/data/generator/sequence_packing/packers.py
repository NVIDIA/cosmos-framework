# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Top-level input sequence packing orchestration."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from cosmos_framework.data.generator.sequence_packing.modalities import (
    pack_action_tokens,
    pack_sound_tokens,
    pack_text_tokens,
    pack_vision_tokens,
)
from cosmos_framework.data.generator.sequence_packing.temporal_causal import pack_supertokens_temporal_causal
from cosmos_framework.data.generator.sequence_packing.types import PackedSequence, SequencePlan

if TYPE_CHECKING:
    from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean


def pack_input_sequence(
    sequence_plans: list[SequencePlan],
    input_text_indexes: list[list[int]],
    gen_data_clean: GenerationDataClean,
    input_timesteps: torch.Tensor,
    special_tokens: dict[str, int],
    max_num_tokens: int | None = None,
    latent_patch_size: int = 1,
    skip_text_tokens: bool = False,
    include_end_of_generation_token: bool = False,
    unified_3d_mrope_reset_spatial_ids: bool = True,
    unified_3d_mrope_temporal_modality_margin: int = 0,
    enable_fps_modulation: bool = False,
    base_fps: float = 24.0,
    sound_base_temporal_compression_factor: int | None = None,
    temporal_compression_factor: int = 4,
    vision_temporal_position_mode: str = "latent_index",
    video_temporal_causal: bool = False,
    action_dim: int = 32,
    initial_mrope_temporal_offset: int | float = 0,
) -> PackedSequence:
    """
    Pack a sequence of input strings and VAE latents into a packed tensor format.
    Uses SequencePlan to determine which modalities are present for each sample,
    and maintains separate indices for text, vision, action, and sound to handle variable modality presence.

    Args:
        sequence_plans: List of SequencePlan items describing which modalities are present.
        input_text_indexes: List of text token ID sequences (only for samples where has_text=True).
        gen_data_clean: GenerationDataClean containing vision, action, and sound tensors.
            - x0_tokens_vision: Vision tensors for samples where has_vision=True
            - x0_tokens_action: Action tensors for samples where has_action=True
            - x0_tokens_sound: Sound tensors (list of [C, T]) for samples where has_sound=True
        input_timesteps: Diffusion timesteps for each sample. Shape (B,) or (B, 1) for
            teacher_forcing/none (all frames share the same sigma), or (B, T_max) for
            diffusion_forcing (per-frame independent sigma). Entries are extracted per
            sample as a float (numel==1) or Tensor(T_max,) for per-frame indexing.
        special_tokens: Dictionary containing special token IDs (eos_token_id, start_of_generation, end_of_generation)
        max_num_tokens: Maximum number of tokens in the packed sequence
        latent_patch_size: Patch size used by the network to pack latents
        skip_text_tokens: If True, skip packing text tokens
        include_end_of_generation_token: If True, append end-of-generation token
        unified_3d_mrope_reset_spatial_ids: If True (default), spatial (H, W) indices
            start from 0 for each vision segment. If False, spatial indices are offset
            by the temporal offset (Qwen2VL-style).
        enable_fps_modulation: If True, scale temporal position IDs based on video FPS
            to reflect real time. Requires fps_vision in gen_data_clean.
            Uses the same flag as diffusion_expert_config.enable_fps_modulation.
        base_fps: Base FPS for normalization (default 24.0).
            Uses the same value as diffusion_expert_config.base_fps.
        sound_base_temporal_compression_factor: Base temporal compression factor for sound FPS scaling.
            ``None`` preserves the current behavior where sound advances at ``base_fps`` positions/sec.
        temporal_compression_factor: VAE temporal compression factor (default 4).
            Obtained from the VAE tokenizer at runtime.
        vision_temporal_position_mode: Temporal coordinates used for unified_3d_mrope vision tokens.
            "latent_index" keeps legacy positions; "uniae_source_right_edge" uses
            per-latent positions from gen_data_clean.temporal_positions_vision.
    Returns:
        PackedSequence containing all packed tensors and metadata. See PackedSequence for field details.
    """
    del max_num_tokens

    assert special_tokens is not None, "Special tokens must be provided"
    assert isinstance(input_timesteps, torch.Tensor), "input_timesteps must be a tensor"
    if input_timesteps.is_cuda:
        raise ValueError("input_timesteps must be on CPU, not CUDA")
    if isinstance(input_text_indexes, torch.Tensor):
        raise ValueError("input_text_tokens must be a list, not a tensor")

    supported_vision_temporal_position_modes = {"latent_index", "uniae_source_right_edge"}
    if vision_temporal_position_mode not in supported_vision_temporal_position_modes:
        raise ValueError(
            "Unsupported vision_temporal_position_mode: "
            f"{vision_temporal_position_mode}. Supported modes: {supported_vision_temporal_position_modes}."
        )
    has_any_vision = any(plan.has_vision for plan in sequence_plans)
    explicit_vision_temporal_positions_active = vision_temporal_position_mode != "latent_index" and has_any_vision
    if explicit_vision_temporal_positions_active:
        if gen_data_clean.temporal_positions_vision is None:
            raise ValueError(
                f"vision_temporal_position_mode={vision_temporal_position_mode} requires "
                "gen_data_clean.temporal_positions_vision."
            )
        if gen_data_clean.x0_tokens_vision is not None and len(gen_data_clean.temporal_positions_vision) != len(
            gen_data_clean.x0_tokens_vision
        ):
            raise ValueError(
                "temporal_positions_vision must have one entry per x0_tokens_vision item, "
                f"got {len(gen_data_clean.temporal_positions_vision)} positions for "
                f"{len(gen_data_clean.x0_tokens_vision)} vision items."
            )
        if video_temporal_causal:
            raise NotImplementedError(
                "video_temporal_causal=True is not wired for explicit UniAE vision temporal positions yet."
            )
        if any(plan.has_action for plan in sequence_plans):
            raise NotImplementedError("Action packing is not wired for explicit UniAE vision temporal positions yet.")
        if initial_mrope_temporal_offset != 0:
            raise NotImplementedError(
                "Autoregressive mRoPE temporal offsets are not wired for explicit UniAE vision temporal positions yet."
            )
    use_float_mrope_positions = enable_fps_modulation or explicit_vision_temporal_positions_active

    # Initialize packed sequence (acts as builder during packing)
    packed_seq = PackedSequence()

    # Configure 3D mRoPE on the builder.
    packed_seq._mrope_reset_spatial = unified_3d_mrope_reset_spatial_ids

    # Maintain separate indices for each modality
    idx_text = 0
    idx_vision = 0
    idx_action = 0
    idx_sound = 0
    null_action_flags: list[bool] = []  # collected from TC path; asserted consistent after the loop

    # Validate: all samples must have text (causal split is always required for two-way attention).
    # CFG dropout only drops text *content*, not the structural text split.
    if not skip_text_tokens:
        for plan in sequence_plans:
            assert plan.has_text, "All sequence plans must have has_text=True when skip_text_tokens=False"

    # Pack each sample based on its sequence plan
    for sample_idx, sequence_plan in enumerate(sequence_plans):
        sample_len = 0

        # mRoPE temporal offset resets per sample.
        # initial_mrope_temporal_offset is non-zero only for AR inference (frame N seeds at N*tcf).
        packed_seq._mrope_temporal_offset = initial_mrope_temporal_offset

        _ts = input_timesteps[sample_idx]
        input_timestep = _ts.item() if _ts.numel() == 1 else _ts  # float (TF) or Tensor(T_max,) (DF)

        # Pack text tokens if has_text=True and not skipped
        if sequence_plan.has_text and not skip_text_tokens:
            text_ids = input_text_indexes[idx_text]
            idx_text += 1

            has_generation_for_sample = sequence_plan.has_vision or sequence_plan.has_action or sequence_plan.has_sound
            text_sample_len = pack_text_tokens(
                packed_seq,
                text_ids,
                special_tokens,
                has_generation=has_generation_for_sample,
                use_float_positions=use_float_mrope_positions,
            )
            sample_len += text_sample_len

            # End of text modality, add an offset as the boundary between text and vision.
            packed_seq._mrope_temporal_offset += unified_3d_mrope_temporal_modality_margin

        # Save temporal offset before vision for action tokens (action uses same offset as vision start)
        vision_start_temporal_offset = packed_seq._mrope_temporal_offset

        # Pack vision (and optionally action) tokens
        if video_temporal_causal and sequence_plan.has_vision:
            # Temporal causal path: when sequence_plan.has_action=True, interleaved supertokens
            # [action_t, vision_t]; when False, supertokens are just vision patches.
            input_vision_tokens = gen_data_clean.x0_tokens_vision[idx_vision]
            idx_vision += 1

            vision_fps = None
            if (
                enable_fps_modulation
                and gen_data_clean.fps_vision is not None
                and idx_vision - 1 < len(gen_data_clean.fps_vision)
            ):
                vision_fps = float(gen_data_clean.fps_vision[idx_vision - 1].item())

            input_action_tokens_tc: torch.Tensor | None = None
            action_fps_tc: float | None = None
            if sequence_plan.has_action:
                input_action_tokens_tc = gen_data_clean.x0_tokens_action[idx_action]
                if (
                    enable_fps_modulation
                    and gen_data_clean.fps_action is not None
                    and idx_action < len(gen_data_clean.fps_action)
                ):
                    action_fps_tc = float(gen_data_clean.fps_action[idx_action].item())
                idx_action += 1

            supertoken_split_len, null_flag = pack_supertokens_temporal_causal(
                packed_seq=packed_seq,
                input_vision_tokens=input_vision_tokens,
                input_action_tokens=input_action_tokens_tc,
                condition_frame_indexes_vision=sequence_plan.condition_frame_indexes_vision,
                input_timestep=input_timestep,
                latent_patch_size=latent_patch_size,
                temporal_compression_factor=temporal_compression_factor,
                action_dim=action_dim,
                vision_fps=vision_fps,
                action_fps=action_fps_tc,
                enable_fps_modulation=enable_fps_modulation,
                base_fps=base_fps,
                pack_action_tokens=sequence_plan.has_action,
            )
            null_action_flags.append(null_flag)
            # We assume all samples in a batch share the same has_action layout, so
            # stamp the supertoken layout constant directly here. This is the
            # single source of truth read by downstream attention / KV-cache
            # code (no recomputation in the network).
            packed_seq.num_action_tokens_per_supertoken = temporal_compression_factor if sequence_plan.has_action else 0
            sample_len += supertoken_split_len
            vision_split_len = supertoken_split_len
            action_split_len = 0  # Already absorbed into supertoken_split_len

        else:
            # Standard path: vision and action packed separately
            if sequence_plan.has_vision:
                # Determine how many vision items this sample owns.
                # For multi-item samples (e.g. image editing), num_vision_items_per_sample
                # records [2, 2, ...]; for standard T2I/T2V it is None (1 item per sample).
                num_vis = (
                    gen_data_clean.num_vision_items_per_sample[sample_idx]
                    if gen_data_clean.num_vision_items_per_sample is not None
                    else 1
                )

                vision_split_len = 0
                # Per-item split lengths for multi-control attention routing.
                # Only tracked when control_weights are present (inference-only);
                # skipped during training to avoid unnecessary side effects.
                track_item_split_lens = gen_data_clean.control_weights is not None
                sample_item_split_lens: list[int] = []
                # Controlnet-style transfer: when set, all vision items share the same
                # temporal mRoPE grid. We snapshot the offset before the loop and
                # rewind to it before each item, so every item produces identical
                # temporal IDs. Each pack_vision_tokens call still advances the
                # offset by latent_t internally; in shared-grid mode the post-loop
                # offset equals snapshot + latent_t (single-clip semantics for
                # downstream EOV / next-modality tokens).
                shared_grid = sequence_plan.share_vision_temporal_positions and num_vis > 1
                items_temporal_offset_snapshot = packed_seq._mrope_temporal_offset
                shared_latent_t: int | None = None
                shared_patch_h: int | None = None
                shared_patch_w: int | None = None
                shared_temporal_positions: torch.Tensor | None = None
                # FPS is recorded per-sample (shape [B]); for multi-item samples
                # (transfer / image-edit) every vision item in this sample shares
                # the same conditioning FPS, so we read by sample_idx, not by the
                # flat idx_vision counter (which would alias to a neighbor sample's
                # fps and corrupt RoPE FPS modulation).
                sample_vision_fps: float | None = None
                if (
                    enable_fps_modulation
                    and gen_data_clean.fps_vision is not None
                    and sample_idx < len(gen_data_clean.fps_vision)
                ):
                    sample_vision_fps = float(gen_data_clean.fps_vision[sample_idx].item())

                for item_idx in range(num_vis):
                    flat_vision_idx = idx_vision
                    input_vision_tokens = gen_data_clean.x0_tokens_vision[flat_vision_idx]
                    vision_temporal_positions: torch.Tensor | None = None
                    if explicit_vision_temporal_positions_active:
                        assert gen_data_clean.temporal_positions_vision is not None
                        vision_temporal_positions = gen_data_clean.temporal_positions_vision[flat_vision_idx]
                        if vision_temporal_positions.shape[0] != input_vision_tokens.shape[2]:
                            raise ValueError(
                                "vision_temporal_positions must match latent_t for each vision item, "
                                f"got {vision_temporal_positions.shape[0]} positions and "
                                f"latent_t={input_vision_tokens.shape[2]} for item {flat_vision_idx}."
                            )
                    vision_fps = sample_vision_fps
                    idx_vision += 1

                    # Determine conditioning for this vision item.
                    # For multi-item mode: all items except the last are fully conditioned
                    # (all frames are clean); the last item uses the SequencePlan's
                    # condition_frame_indexes_vision (typically [] = fully generated).
                    if num_vis > 1 and item_idx < num_vis - 1:
                        # Conditioning item (e.g. source image): mark all frames as clean
                        latent_t = input_vision_tokens.shape[2]
                        item_condition_frames = list(range(latent_t))
                    else:
                        # Generation item (single-item mode or last item in multi-item)
                        item_condition_frames = sequence_plan.condition_frame_indexes_vision

                    if shared_grid:
                        item_latent_t = input_vision_tokens.shape[2]
                        item_latent_h = input_vision_tokens.shape[3]
                        item_latent_w = input_vision_tokens.shape[4]
                        if shared_latent_t is None:
                            shared_latent_t = item_latent_t
                            shared_patch_h = item_latent_h
                            shared_patch_w = item_latent_w
                        else:
                            assert item_latent_t == shared_latent_t, (
                                f"share_vision_temporal_positions requires equal latent_t across items, "
                                f"got item {item_idx} latent_t={item_latent_t} vs first={shared_latent_t}"
                            )
                            assert item_latent_h == shared_patch_h and item_latent_w == shared_patch_w, (
                                f"share_vision_temporal_positions requires equal spatial grid across items, "
                                f"got item {item_idx} (H,W)=({item_latent_h},{item_latent_w}) "
                                f"vs first=({shared_patch_h},{shared_patch_w})"
                            )
                        if vision_temporal_positions is not None:
                            if shared_temporal_positions is None:
                                shared_temporal_positions = vision_temporal_positions
                            else:
                                comparison_temporal_positions = vision_temporal_positions.to(
                                    device=shared_temporal_positions.device
                                )  # [T]
                                assert torch.allclose(comparison_temporal_positions, shared_temporal_positions), (
                                    "share_vision_temporal_positions requires equal explicit temporal positions "
                                    f"across vision items, got item {item_idx} positions "
                                    f"{vision_temporal_positions.tolist()} vs first "
                                    f"{shared_temporal_positions.tolist()}."
                                )
                        # Rewind so this item starts at the same temporal offset as item 0.
                        packed_seq._mrope_temporal_offset = items_temporal_offset_snapshot

                    item_split_len = pack_vision_tokens(
                        packed_seq=packed_seq,
                        input_vision_tokens=input_vision_tokens,
                        condition_frame_indexes_vision=item_condition_frames,
                        input_timestep=input_timestep,
                        latent_patch_size=latent_patch_size,
                        vision_fps=vision_fps,
                        enable_fps_modulation=enable_fps_modulation,
                        base_fps=base_fps,
                        temporal_compression_factor=temporal_compression_factor,
                        vision_temporal_positions=vision_temporal_positions,
                    )
                    vision_split_len += item_split_len
                    if track_item_split_lens:
                        sample_item_split_lens.append(item_split_len)
                if track_item_split_lens:
                    packed_seq.vision_item_split_lens.append(sample_item_split_lens)
                sample_len += vision_split_len

            else:
                vision_split_len = 0

            # Pack action tokens if has_action=True
            if sequence_plan.has_action:
                input_action_tokens = gen_data_clean.x0_tokens_action[idx_action]

                # Get FPS for action (action may have its own FPS independent of vision)
                action_fps: float | None = None
                if (
                    enable_fps_modulation
                    and gen_data_clean.fps_action is not None
                    and idx_action < len(gen_data_clean.fps_action)
                ):
                    action_fps = float(gen_data_clean.fps_action[idx_action].item())

                idx_action += 1

                action_split_len = pack_action_tokens(
                    packed_seq=packed_seq,
                    input_action_tokens=input_action_tokens,
                    condition_frame_indexes_action=sequence_plan.condition_frame_indexes_action,
                    input_timestep=input_timestep,
                    action_temporal_offset=vision_start_temporal_offset,
                    enable_fps_modulation=enable_fps_modulation,
                    base_fps=base_fps,
                    action_fps=action_fps,
                    base_temporal_compression_factor=temporal_compression_factor,
                    action_start_frame_offset=sequence_plan.action_start_frame_offset,
                )
                sample_len += action_split_len
            else:
                action_split_len = 0

        # Pack sound tokens if has_sound=True
        if sequence_plan.has_sound:
            input_sound_tokens = gen_data_clean.x0_tokens_sound[idx_sound]

            # Get FPS for sound (from gen_data_clean, like vision and action)
            sound_fps: float | None = None
            if (
                enable_fps_modulation
                and gen_data_clean.fps_sound is not None
                and idx_sound < len(gen_data_clean.fps_sound)
            ):
                sound_fps = float(gen_data_clean.fps_sound[idx_sound].item())

            idx_sound += 1

            sound_split_len = pack_sound_tokens(
                packed_seq=packed_seq,
                input_sound_tokens=input_sound_tokens,
                condition_frame_indexes_sound=sequence_plan.condition_frame_indexes_sound,
                input_timestep=input_timestep,
                sound_temporal_offset=vision_start_temporal_offset,
                enable_fps_modulation=enable_fps_modulation,
                base_fps=base_fps,
                sound_fps=sound_fps,
                sound_base_temporal_compression_factor=sound_base_temporal_compression_factor,
            )
            sample_len += sound_split_len
        else:
            sound_split_len = 0

        # Add end-of-generation token if needed
        eov_len = 0
        has_any_generation = sequence_plan.has_vision or sequence_plan.has_action or sequence_plan.has_sound
        if include_end_of_generation_token and has_any_generation:
            # Type narrowing: we're in build mode, fields are lists
            assert isinstance(packed_seq.text_ids, list)
            assert isinstance(packed_seq.text_indexes, list)
            assert isinstance(packed_seq.position_ids, list)

            packed_seq.text_ids.append(special_tokens["end_of_generation"])
            packed_seq.text_indexes.append(packed_seq.curr)

            # Use float dtype when any vision mRoPE positions are fractional.
            eov_dtype = torch.float32 if use_float_mrope_positions else torch.long
            eov_mrope_ids = torch.full((3, 1), packed_seq._mrope_temporal_offset, dtype=eov_dtype)  # [3,1]
            packed_seq.position_ids.append(eov_mrope_ids)  # type: ignore[arg-type]
            packed_seq._mrope_temporal_offset += 1

            packed_seq.curr += 1
            eov_len = 1
            sample_len += 1

        combined_split_len = vision_split_len + action_split_len + sound_split_len + eov_len
        packed_seq.attn_modes.append("full")
        packed_seq.split_lens.append(combined_split_len)
        packed_seq.sample_lens.append(sample_len)

    # Assert consistent null_action_supertokens across all TC samples, then set once
    if null_action_flags:
        assert len(set(null_action_flags)) == 1, (
            f"Inconsistent null_action_supertokens across samples: {null_action_flags}. "
            "All samples in a batch must have the same structure (all training or all AR inference)."
        )
        packed_seq.null_action_supertokens = null_action_flags[0]

    # Finalize and return packed data
    return packed_seq.finalize(
        gen_data_clean=gen_data_clean,
    )
