# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Temporal-causal supertoken packing helpers."""

import math

import torch

from cosmos_framework.data.generator.sequence_packing.mrope import get_3d_mrope_ids_vae_tokens
from cosmos_framework.data.generator.sequence_packing.types import ModalityData, PackedSequence


def pack_supertokens_temporal_causal(
    packed_seq: "PackedSequence",
    input_vision_tokens: torch.Tensor,
    input_action_tokens: torch.Tensor | None,
    condition_frame_indexes_vision: list[int],
    input_timestep: float | torch.Tensor,
    latent_patch_size: int,
    temporal_compression_factor: int,
    action_dim: int,
    vision_fps: float | None = None,
    action_fps: float | None = None,
    enable_fps_modulation: bool = False,
    base_fps: float = 24.0,
    pack_action_tokens: bool = True,
) -> tuple[int, bool]:
    """Pack vision and (optionally) action tokens in supertoken order for temporal causal attention.

    Buffer layout per frame:
        pack_action_tokens=True:  [action_t (tcf), vision_t (H*W)]  — supertoken size tcf + H*W
        pack_action_tokens=False: [vision_t (H*W)]                  — supertoken size H*W

    Use ``pack_action_tokens=False`` when ``config.action_gen=False``; the resulting
    ``num_action_tokens_per_supertoken=0`` is stamped on the pack and read by the
    attention builder so NATTEN metadata stays in sync automatically.

    mRoPE layout (with actions, unified_3d_mrope only). The layout is inferred from the
    action tensor shape:
        - Whole-clip training (frame 0 is the clean conditioning frame, so
          ``real_actions`` has ``(T-1)*tcf`` rows): null action for supertoken 0, real
          actions for frames 1..T-1 with ``start_frame_offset=1`` so the last action in
          group i co-locates with vision frame i; vision uses ``start_frame_offset=0``.
        - AR generation, single frame OR chunk (every frame carries a real action, so
          ``real_actions`` has ``latent_t*tcf`` rows): vision AND action both use
          ``start_frame_offset=1``, generalizing the single-frame AR supertoken to
          ``latent_t`` frames. The caller (``pack_input_sequence_autoregressive``)
          seeds ``temporal_offset`` one frame-stride back to compensate, so the unit
          lands at the same absolute positions as the whole-clip training pack.
        - Interleaved per frame as cat([action_ids, vision_ids]).

    ``input_timestep`` is float (TF/none) or Tensor(T_max,) (DF, per-frame sigma).
    Conditioning frames are excluded from mse_loss_indexes either way.

    Returns (total_split_len, null_action_flag); null_action_flag is False when
    pack_action_tokens=False.
    """
    assert isinstance(packed_seq.position_ids, list), "PackedSequence must be in build mode"

    _, _, latent_t, latent_h, latent_w = input_vision_tokens.shape
    patch_h = math.ceil(latent_h / latent_patch_size)
    patch_w = math.ceil(latent_w / latent_patch_size)
    tcf = temporal_compression_factor
    patches_per_frame = patch_h * patch_w
    supertoken_len = tcf + patches_per_frame if pack_action_tokens else patches_per_frame  # S

    # Initialize modalities if needed
    if packed_seq.vision is None:
        packed_seq.vision = ModalityData()
    if pack_action_tokens and packed_seq.action is None:
        packed_seq.action = ModalityData()

    assert isinstance(packed_seq.vision.sequence_indexes, list)
    assert isinstance(packed_seq.vision.mse_loss_indexes, list)
    assert isinstance(packed_seq.vision.timesteps, list)
    assert isinstance(packed_seq.vision.tokens, list)
    assert isinstance(packed_seq.vision.condition_mask, list)
    if pack_action_tokens:
        assert isinstance(packed_seq.action.sequence_indexes, list)
        assert isinstance(packed_seq.action.mse_loss_indexes, list)
        assert isinstance(packed_seq.action.timesteps, list)
        assert isinstance(packed_seq.action.tokens, list)
        assert isinstance(packed_seq.action.condition_mask, list)

    device = input_vision_tokens.device
    dtype = input_vision_tokens.dtype

    null_action_flag: bool
    if pack_action_tokens:
        # Build all_action_tokens: shape (latent_t * tcf, action_dim)
        #
        # Cases (token assembly; mRoPE start_frame_offset is chosen separately below,
        # inferred from the same action shape):
        #   1. Whole-clip training with conditioning frame (latent_t > 1, real_actions
        #      has (T-1)*tcf rows): prepend tcf null tokens for frame 0, then real
        #      actions for frames 1..T-1.
        #   2. AR generation (every frame has a real action, real_actions has
        #      latent_t*tcf rows — single frame OR chunk): no null prefix.
        #   3. AR frame 0 / image2video (action is None): all null tokens.
        if input_action_tokens is not None:
            # input_action_tokens shape: (1, T*tcf, D) or (T*tcf, D) for training; (T*tcf, D) for AR units
            if input_action_tokens.dim() == 3:
                real_actions = input_action_tokens.squeeze(0)  # [T*tcf,action_dim] or [N,action_dim]
            else:
                real_actions = input_action_tokens  # [N,action_dim]
            null_tokens = torch.zeros(tcf, action_dim, device=device, dtype=real_actions.dtype)  # [tcf,action_dim]
            if real_actions.shape[0] == latent_t * tcf:
                # AR generation (single frame: tcf == 1*tcf, or chunk: latent_t*tcf):
                # every supertoken carries a real action, no null prefix.
                all_action_tokens = real_actions
                null_action_flag = False
            elif real_actions.shape[0] == (latent_t - 1) * tcf:
                # Conditioning frame present: null for supertoken 0, real for 1..T-1
                all_action_tokens = torch.cat([null_tokens, real_actions], dim=0)  # [T*tcf,action_dim]
                null_action_flag = True
            else:
                raise ValueError(
                    "Temporal-causal action tokens must have either latent_t*tcf rows for AR chunks "
                    f"or (latent_t-1)*tcf rows for whole-clip training; got {real_actions.shape[0]} rows "
                    f"for latent_t={latent_t}, tcf={tcf}."
                )
        else:
            # AR frame 0 or image2video: all action tokens are null
            all_action_tokens = torch.zeros(
                latent_t * tcf, action_dim, device=device, dtype=dtype
            )  # [T*tcf,action_dim]
            null_action_flag = True
    else:
        # pack_action_tokens=False: action tokens must not be supplied.
        assert input_action_tokens is None, (
            "pack_action_tokens=False requires input_action_tokens=None; got a non-None tensor."
        )
        null_action_flag = False

    # Record vision token shapes and tokens
    packed_seq.vision.token_shapes.append((latent_t, patch_h, patch_w))
    packed_seq.vision.tokens.append(input_vision_tokens)

    # Vision conditioning mask: (T, 1, 1)
    condition_set_vision = {idx for idx in condition_frame_indexes_vision if 0 <= idx < latent_t}
    vision_condition_mask = torch.zeros((latent_t, 1, 1), device=device, dtype=dtype)  # [T,1,1]
    for fidx in condition_set_vision:
        vision_condition_mask[fidx, 0, 0] = 1.0
    packed_seq.vision.condition_mask.append(vision_condition_mask)

    vision_noisy_frame_indexes = torch.tensor(
        [idx for idx in range(latent_t) if idx not in condition_set_vision],
        device=device,
        dtype=torch.long,
    )  # [N_noisy_frames]
    packed_seq.vision.noisy_frame_indexes.append(vision_noisy_frame_indexes)

    if pack_action_tokens:
        # Action token shapes: latent_t * tcf total (including null tokens)
        packed_seq.action.token_shapes.append((latent_t * tcf,))
        packed_seq.action.tokens.append(all_action_tokens)

        # Action conditioning mask: all action tokens are conditioning (not supervised)
        # Null tokens are always conditioning; real actions are conditioning too (they are inputs)
        action_condition_mask = torch.ones((latent_t * tcf, 1), device=device, dtype=dtype)  # [T*tcf,1]
        packed_seq.action.condition_mask.append(action_condition_mask)

    # Pack in interleaved supertoken order: [action_t, vision_t] for each frame t
    # (or just [vision_t] per frame when pack_action_tokens=False)
    curr = packed_seq.curr
    total_split_len = 0

    # Snapshot the offset before this sample and compute mRoPE IDs.
    temporal_offset = packed_seq._mrope_temporal_offset
    effective_vision_fps = vision_fps if enable_fps_modulation else None

    # AR generation (single frame OR chunk) is detected by every frame carrying a
    # real action (``real_actions`` has ``latent_t*tcf`` rows). There, vision AND
    # action both use start_frame_offset=1 so the last action in each group
    # co-locates with its vision frame, mirroring whole-clip training; the caller
    # (pack_input_sequence_autoregressive) seeds temporal_offset one frame-stride
    # back to compensate. Whole-clip training (frame 0 is the null conditioning
    # frame, ``real_actions`` has ``(T-1)*tcf`` rows) keeps vision start_frame_offset=0.
    all_frames_have_real_action = (
        pack_action_tokens and input_action_tokens is not None and real_actions.shape[0] == latent_t * tcf
    )
    vision_sfo = 1 if all_frames_have_real_action else 0

    vision_ids_flat, new_offset = get_3d_mrope_ids_vae_tokens(
        grid_t=latent_t,
        grid_h=patch_h,
        grid_w=patch_w,
        temporal_offset=temporal_offset,
        reset_spatial_indices=packed_seq._mrope_reset_spatial,
        fps=effective_vision_fps,
        base_fps=base_fps,
        temporal_compression_factor=tcf,
        start_frame_offset=vision_sfo,
    )  # vision_ids_flat: [3,T*patch_h*patch_w]

    if pack_action_tokens:
        effective_action_fps = action_fps if enable_fps_modulation else None

        # Action IDs. Real action tokens use start_frame_offset=1 so the last
        # sub-token of a group co-locates with its vision frame. Whole-clip training
        # has a null action at frame 0 (the conditioning frame); AR units have a real
        # action for every frame.
        fps_active = effective_action_fps is not None
        t_dtype = torch.float32 if fps_active else torch.long
        t_offset = float(temporal_offset) if fps_active else int(temporal_offset)
        null_t = torch.full((tcf,), t_offset, dtype=t_dtype)  # [tcf]
        null_hw = torch.zeros(tcf, dtype=t_dtype)  # [tcf]
        null_ids = torch.stack([null_t, null_hw, null_hw])  # [3,tcf]

        def _real_action_ids(n_frames: int, start_frame_offset: int) -> torch.Tensor:
            flat, _ = get_3d_mrope_ids_vae_tokens(
                grid_t=n_frames * tcf,
                grid_h=1,
                grid_w=1,
                temporal_offset=temporal_offset,
                reset_spatial_indices=packed_seq._mrope_reset_spatial,
                fps=effective_action_fps,
                base_fps=base_fps,
                temporal_compression_factor=1,
                base_temporal_compression_factor=tcf,
                start_frame_offset=start_frame_offset,
            )
            return flat.reshape(3, n_frames, tcf)  # [3,n_frames,tcf]

        if all_frames_have_real_action:
            # AR generation (single frame: tcf == 1*tcf, or chunk: latent_t*tcf):
            # every supertoken carries a real action. start_frame_offset=1 puts
            # a_{j-1}'s last sub-token on vision frame j -- the whole-clip TF
            # training layout. The caller seeds temporal_offset (N-1) frame-strides
            # back to compensate.
            action_ids_3d = _real_action_ids(latent_t, start_frame_offset=1)  # [3,T,tcf]
        elif latent_t > 1:
            # Whole-clip training: supertoken 0 = null (conditioning frame), frames
            # 1..T-1 = real with start_frame_offset=1. Covers real-action training
            # (real_actions has (T-1)*tcf rows) and the architectural all-null layout
            # (input_action_tokens is None); the tokens differ but the IDs match.
            null_ids_3d = null_ids.reshape(3, 1, tcf)  # [3,1,tcf]
            real_ids_3d = _real_action_ids(latent_t - 1, start_frame_offset=1)  # [3,T-1,tcf]
            action_ids_3d = torch.cat([null_ids_3d, real_ids_3d], dim=1)  # [3,T,tcf]
        else:
            # AR frame 0 / image2video (latent_t == 1, no action): only null.
            action_ids_3d = null_ids.reshape(3, 1, tcf)  # [3,1,tcf]

        # (3, T*H*W) -> (3, T, H*W)
        vision_ids_3d = vision_ids_flat.reshape(3, latent_t, patches_per_frame)  # [3,T,patch_h*patch_w]

        # Interleave per frame: (3, T, tcf+H*W) -> (3, T*S)
        interleaved_ids = torch.cat([action_ids_3d, vision_ids_3d], dim=2).reshape(
            3, latent_t * supertoken_len
        )  # [3,T*S]
        packed_seq.position_ids.append(interleaved_ids)
    else:
        # No action tokens: just vision IDs, already in (3, T*H*W) order.
        packed_seq.position_ids.append(vision_ids_flat)

    packed_seq._mrope_temporal_offset = new_offset

    for frame_t in range(latent_t):
        if pack_action_tokens:
            # Pack action tokens for this frame (indexes only; tokens already stored in packed_seq.action.tokens)
            action_indexes = list(range(curr, curr + tcf))
            packed_seq.action.sequence_indexes.extend(action_indexes)
            # Action tokens are never in MSE loss (always conditioning)
            curr += tcf
            total_split_len += tcf

        # Pack vision tokens for this frame
        frame_indexes = list(range(curr, curr + patches_per_frame))
        packed_seq.vision.sequence_indexes.extend(frame_indexes)
        curr += patches_per_frame
        total_split_len += patches_per_frame

        # Vision MSE loss: supervise non-conditioning frames
        if frame_t not in condition_set_vision:
            packed_seq.vision.mse_loss_indexes.extend(frame_indexes)
            frame_ts = input_timestep[frame_t].item() if isinstance(input_timestep, torch.Tensor) else input_timestep
            packed_seq.vision.timesteps.extend([frame_ts] * patches_per_frame)

    packed_seq.curr = curr
    return total_split_len, null_action_flag
