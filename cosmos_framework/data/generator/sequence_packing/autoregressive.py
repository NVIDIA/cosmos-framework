# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Autoregressive sequence packing for framewise and chunkwise AR generation."""

import torch

from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean
from cosmos_framework.data.generator.sequence_packing import (
    PackedSequence,
    SequencePlan,
    pack_input_sequence,
)


def pack_input_sequence_autoregressive(
    vision_latent: torch.Tensor | None,
    action_latent: torch.Tensor | None,
    text_tokens: list[int] | None,
    timestep: float,
    fps_vision: list[float],
    fps_action: list[float] | None,
    special_tokens: dict[str, int],
    latent_patch_size: int = 1,
    condition_frame_indexes_vision: list[int] | None = None,
    condition_frame_indexes_action: list[int] | None = None,
    frame_idx: int = 0,
    temporal_compression_factor: int = 4,
    video_temporal_causal: bool = False,
    action_dim: int = 32,
    enable_fps_modulation: bool = True,
    base_fps: float = 24.0,
    cached_text_offset: int | None = None,
    unified_3d_mrope_temporal_modality_margin: int = 0,
    force_action_tokens: bool = False,
    action_domain_id: torch.Tensor | int | None = None,
    raw_action_dim: torch.Tensor | int | None = None,
    vision_temporal_positions: torch.Tensor | None = None,
    num_views: int | None = None,
) -> PackedSequence:
    """
    Pack input sequence for autoregressive video generation (one AR unit at a time).

    An "AR unit" is a single latent frame (framewise AR) or a chunk of ``chunk_len``
    frames denoised jointly (chunkwise AR, ``teacher_forcing_frames_per_chunk > 1``).
    This function returns a finalized PackedSequence containing only the current
    unit's tokens, avoiding the overhead of repacking entire sequences during AR
    generation. Earlier text/generated prefixes are represented by KV-cache state;
    this packed delta carries the offsets needed to align its positions with that
    cached history.
    Used for ALL units (frame 0, then each subsequent frame or chunk) in AR generation.

    Text2video frame 0:            [text]  (no vision yet)
    Image2video frame 0:           [text] [v0]  (v0 is conditioned)
    Framewise unit N (N>0):        [aN-1] [vN]  (no text, only action + vision)
    Chunkwise unit [N, N+C):       [aN-1 vN] [aN vN+1] ... [aN+C-2 vN+C-1]
                                   (per-frame [action, vision] supertokens, no text)

    Args:
        vision_latent: Vision latent for the current AR unit. Shape: (1, C, T, H, W),
            where ``T`` is 1 for framewise AR and ``chunk_len`` for chunkwise AR. Or None.
        action_latent: Action latent for the current AR unit. Temporal causal: (T*tcf, D)
            (T action frames a_{N-1}..a_{N+T-2}, tcf sub-tokens each). Standard: (1, 1, D).
            Or None.
        text_tokens: List of text token IDs, or None for units after frame 0
        timestep: Diffusion timestep for noise schedule (single float)
        fps_vision: FPS for vision modality (list with single element)
        fps_action: FPS for action modality (list with single element), or None
        special_tokens: Dictionary of special token IDs (eos_token_id, start_of_generation, etc.)
        latent_patch_size: Patch size for latent patchification
        condition_frame_indexes_vision: Indexes of clean vision frames (for conditioning)
        condition_frame_indexes_action: Indexes of clean action steps (for conditioning)
        frame_idx: Global latent-frame index of the FIRST frame in this AR unit
            (``N``). For framewise AR it is the frame index; for chunkwise AR it is the
            chunk's starting frame (``chunk_start``). Used to seed the mRoPE temporal
            offset so the unit lands at its absolute training position.
        enable_fps_modulation: If True (default), the unit is packed at
            ``frame_stride = base_fps / fps`` mRoPE units per frame, matching training's
            fps-modulated positions. Production AR should always leave this True;
            False is only for tests that want integer positions.
        base_fps: Reference fps when enable_fps_modulation=True. Must match
            ``diffusion_expert_config.base_fps``.
        cached_text_offset: Length of the cached text supertoken (in mRoPE temporal
            units) for frame N>=1, when text is already in und_cache and ``text_tokens``
            is None. Pass ``None`` (default) for frame 0 / prefill, where text is
            packed inline and the offset is computed internally. The modality margin
            is added on top of this value internally — callers should pass the raw
            ``compute_text_split_length(...)`` without baking the margin in.
        unified_3d_mrope_temporal_modality_margin: Margin added between text and
            vision modalities on the mRoPE temporal axis. Must match the value used
            during training (``diffusion_expert_config.unified_3d_mrope_temporal_modality_margin``).
            Applied uniformly across all frames: at frame 0 it is consumed by the
            inline text packing inside ``pack_input_sequence``; at frame N>=1 it is
            added to ``cached_text_offset`` here so vision lands at training's
            absolute positions.
        force_action_tokens: Pack the temporal-causal action prefix even when
            ``action_latent`` is ``None``. This is required for frame-0 null
            action supertokens in action-conditioned AR/FDM layouts.
        action_domain_id: Optional domain id for DomainAwareLinear action projection.
        raw_action_dim: Optional raw action width before model-space padding.
        vision_temporal_positions: Optional absolute flattened-item latent
            coordinates for every frame in ``vision_latent``. Multiview WSM AR
            uses this to preserve each camera's position in the full camera-major
            training item while packing only the current chunk.
        num_views: Number of camera views concatenated along the latent temporal
            axis. Required by multiview FlexAttention metadata.

    Returns:
        Finalized PackedSequence containing the supertoken(s) for this AR unit
        (one per frame).

    Example:
        # Text2video frame 0: Pack text only (no vision)
        pack = pack_input_sequence_autoregressive(
            vision_latent=None,  # No vision yet
            action_latent=None,
            text_tokens=[1, 2, 3, ...],  # Text token IDs
            timestep=0.0,
            fps_vision=[24.0],
            fps_action=None,
            special_tokens=special_tokens,
        )

        # Text2video frame 1+: Pack only new vision frame (no text)
        pack = pack_input_sequence_autoregressive(
            vision_latent=v1,  # (1, C, 1, H, W) - new frame only
            action_latent=None,
            text_tokens=None,  # No text for frame 1+
            timestep=t,
            fps_vision=[24.0],
            fps_action=None,
            special_tokens=special_tokens,
        )

        # Chunkwise forward-dynamics unit [N, N+C): pack C frames + their actions
        pack = pack_input_sequence_autoregressive(
            vision_latent=chunk,           # (1, C, chunk_len, H, W) - denoised jointly
            action_latent=chunk_actions,   # (chunk_len*tcf, D) - a_{N-1}..a_{N+C-2}
            text_tokens=None,
            timestep=t,
            fps_vision=[24.0],
            fps_action=[24.0],
            special_tokens=special_tokens,
            frame_idx=N,                   # chunk_start
            video_temporal_causal=True,
        )

        # Image2video frame 0: Pack text + conditioned v0
        pack = pack_input_sequence_autoregressive(
            vision_latent=v0,  # (1, C, 1, H, W) - conditioned
            action_latent=None,
            text_tokens=[1, 2, 3, ...],
            timestep=0.0,
            fps_vision=[24.0],
            fps_action=None,
            special_tokens=special_tokens,
            condition_frame_indexes_vision=[0],  # Mark v0 as conditioned
        )
    """
    # Validate inputs
    if vision_latent is not None:
        assert vision_latent.shape[0] == 1, "pack_input_sequence_autoregressive only supports batch_size=1"
        # ``vision_latent.shape[2]`` (number of latent frames) is 1 for framewise AR
        # and ``chunk_size`` for chunkwise AR (chunk denoised jointly).
        if vision_temporal_positions is not None and vision_temporal_positions.shape != (vision_latent.shape[2],):
            raise ValueError(
                "vision_temporal_positions must match the packed latent time axis, "
                f"got {tuple(vision_temporal_positions.shape)} for latent_t={vision_latent.shape[2]}."
            )
    elif vision_temporal_positions is not None:
        raise ValueError("vision_temporal_positions requires vision_latent.")
    if num_views is not None and num_views < 1:
        raise ValueError(f"num_views must be >= 1, got {num_views}.")
    if action_latent is not None:
        if video_temporal_causal:
            assert action_latent.dim() == 2, (
                f"Temporal causal action_latent should be (num_frames*tcf, D), got {tuple(action_latent.shape)}"
            )
        else:
            assert action_latent.shape[0] == 1, f"action_latent batch_size must be 1, got {action_latent.shape[0]}"
            assert action_latent.shape[1] == 1, (
                f"action_latent must be a single step (T=1) for AR inference, got {action_latent.shape[1]}"
            )

    # For temporal causal AR: frame 0 can carry null action tokens before any
    # real action exists. Callers opt into that architectural action layout
    # with force_action_tokens=True.

    # Build SequencePlan
    has_text = text_tokens is not None
    has_vision = vision_latent is not None
    has_action = action_latent is not None or force_action_tokens

    sequence_plan = SequencePlan(
        has_text=has_text,
        has_vision=has_vision,
        has_action=has_action,
        condition_frame_indexes_vision=condition_frame_indexes_vision or [],
        condition_frame_indexes_action=condition_frame_indexes_action or [],
    )

    # Create GenerationDataClean for this AR unit (single frame or chunk)
    # Note: x0_tokens_vision and x0_tokens_action must be lists of tensors
    # since pack_input_sequence indexes them per sample. fps_vision/fps_action
    # are tensors (matching GenerationDataClean's type); pack_input_sequence
    # pulls per-sample fps via ``.item()`` on each entry.
    fps_vision_t = torch.as_tensor(fps_vision, dtype=torch.float32) if fps_vision is not None else None
    fps_action_t = (
        torch.as_tensor(fps_action, dtype=torch.float32) if (fps_action is not None and has_action) else None
    )  # [B_action] or None
    action_domain_id_tensor = (
        torch.as_tensor(action_domain_id, dtype=torch.long).reshape(1)
        if has_action and action_domain_id is not None
        else None
    )  # [1] or None
    action_domain_id_list = [action_domain_id_tensor] if action_domain_id_tensor is not None else None
    raw_action_dim_list = (
        [torch.as_tensor(raw_action_dim, dtype=torch.long)] if has_action and raw_action_dim is not None else None
    )  # list[[]] or None
    gen_data_clean = GenerationDataClean(
        batch_size=1,
        is_image_batch=False,
        raw_state_vision=None,
        x0_tokens_vision=[vision_latent],  # List with single 5D tensor (1, C, 1, H, W)
        x0_tokens_action=[action_latent] if has_action else None,  # List with single tensor/None for null actions
        fps_vision=fps_vision_t,
        fps_action=fps_action_t,
        temporal_positions_vision=[vision_temporal_positions] if vision_temporal_positions is not None else None,
        action_domain_id=action_domain_id_list,
        raw_action_dim=raw_action_dim_list,
        num_views_per_vision_item=[num_views] if num_views is not None else None,
    )

    # Prepare text indexes
    input_text_indexes = [text_tokens] if has_text else [[]]  # Empty list for no text

    # Prepare timestep
    input_timesteps = torch.tensor([timestep], dtype=torch.float32)  # [1]

    # Per-frame mRoPE temporal stride. Training's one-shot fps-modulated pack
    # spaces consecutive latent frames by ``base_fps / fps`` (= 1.0 at 24fps,
    # 0.8 at 30fps, 1.5 at 16fps); per-frame AR packs must use the same stride so
    # vision K/V lands at training's absolute positions.
    assert enable_fps_modulation, "enable_fps_modulation must be True for autoregressive packing"
    assert len(fps_vision) >= 1, "fps_vision must contain the per-clip fps"
    _fps = float(fps_vision[0])
    assert _fps > 0, f"fps must be > 0 for fps-modulated AR packing, got {_fps}"
    frame_stride: int | float = base_fps / _fps

    # AR generation (single frame OR multi-frame chunk) with real actions uses
    # start_frame_offset=1 inside the TC packer for both vision and action
    # (matching whole-clip training, where action a_{g-1}'s last sub-token
    # co-locates with vision frame g), so the pack itself advances one
    # frame-stride; seed (N-1) frames back to compensate. The packer infers this
    # layout from the action shape (every frame carries a real action), so no flag
    # is needed -- it holds for any chunk size, including latent_t > 1.
    ar_with_real_actions = video_temporal_causal and action_latent is not None
    seed_frames = (frame_idx - 1) if ar_with_real_actions else frame_idx

    # Frame 0 (cached_text_offset is None): text is packed inline and the inner
    # ``pack_input_sequence`` advances through text + margin itself, so we seed at 0.
    # Frame N>=1 (cached_text_offset is an int): text was packed once into und_cache;
    # bake the cached length + modality margin in here so vision Q/K lands at
    # training's absolute positions.
    if cached_text_offset is None:
        text_offset_for_pack: int | float = 0
    else:
        text_offset_for_pack = cached_text_offset + unified_3d_mrope_temporal_modality_margin

    # Call pack_input_sequence
    if vision_temporal_positions is not None and cached_text_offset is not None:
        raise ValueError("Explicit multiview AR positions require text tokens to remain inline in the pack.")
    initial_temporal_offset = (
        0 if vision_temporal_positions is not None else text_offset_for_pack + seed_frames * frame_stride
    )
    packed_seq = pack_input_sequence(
        sequence_plans=[sequence_plan],
        input_text_indexes=input_text_indexes,
        gen_data_clean=gen_data_clean,
        input_timesteps=input_timesteps,
        special_tokens=special_tokens,
        latent_patch_size=latent_patch_size,
        skip_text_tokens=(not has_text),
        unified_3d_mrope_temporal_modality_margin=unified_3d_mrope_temporal_modality_margin,
        enable_fps_modulation=enable_fps_modulation,
        base_fps=base_fps,
        temporal_compression_factor=temporal_compression_factor,
        vision_temporal_position_mode=(
            "uniae_source_right_edge" if vision_temporal_positions is not None else "latent_index"
        ),
        video_temporal_causal=video_temporal_causal,
        action_dim=action_dim,
        initial_mrope_temporal_offset=initial_temporal_offset,
    )

    return packed_seq
