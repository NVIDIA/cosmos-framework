# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Registration-free causal teacher-forcing model defaults."""

import copy
from typing import Any

_MODEL_COMMON: dict[str, Any] = dict(
    video_temporal_causal=True,
    joint_attn_implementation="three_way",
    resolution="480",
    state_ch=48,
    latent_downsample_factor=16,
    state_t=300,
    max_num_tokens_after_packing=30000,
    rectified_flow_training_config=dict(
        train_time_weight="uniform",
        train_time_video_distribution="waver",
        loss_scale=10.0,
        use_discrete_rf=False,
        shift=5,
    ),
    diffusion_expert_config=dict(
        patch_spatial=2,
        unified_3d_mrope_reset_spatial_ids=True,
        unified_3d_mrope_temporal_modality_margin=15000,
        enable_fps_modulation=True,
        base_fps=24,
    ),
    tokenizer=dict(
        bucket_name="",
        object_store_credential_path_pretrained="",
        encode_chunk_frames={"256": 68, "480": 24, "720": 8},
    ),
    parallelism=dict(
        data_parallel_shard_degree=8,
    ),
    compile=dict(
        enabled=True,
    ),
    vlm_config=dict(pretrained_weights=dict(enabled=False), use_system_prompt=False),
)

_MODEL_VIDEO: dict[str, Any] = copy.deepcopy(_MODEL_COMMON)
