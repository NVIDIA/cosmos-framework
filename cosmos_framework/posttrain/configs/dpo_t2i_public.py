# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Self-contained public Diffusion-DPO image recipe with synthetic preference pairs."""

from __future__ import annotations

import os
from copy import deepcopy

from cosmos_framework.configs.base.defaults.model import MOT_FSDP_CONFIG
from cosmos_framework.configs.base.defaults.reasoner import Qwen3VLMoT_VLM_8b_Instruct_Config
from cosmos_framework.configs.base.defaults.tokenizer import Wan2pt2VAEConfig
from cosmos_framework.posttrain.config.config import build_root
from cosmos_framework.posttrain.config.config_dpo import PosttrainDPOConfig
from cosmos_framework.posttrain.config.rl import DPOConfig, OptimizerConfig, PreferenceEnvConfig, RefConfig


def build_config() -> PosttrainDPOConfig:
    """Build a catalog-free four-rank public T2I DPO functional recipe."""
    config = build_root(
        PosttrainDPOConfig,
        "posttrain_dpo_public",
        register_experiment_configs=False,
        register_base_configs=False,
    )
    config.algorithm = "dpo"
    config.job.group = "posttrain_dpo_public"
    config.job.name = "dpo_t2i_public"
    config.job.wandb_mode = "disabled"

    config.grid.backend = "local"
    config.grid.grouping = "single"
    config.grid.ranks = 4

    config.trainer.distributed_parallelism = "fsdp"
    config.model = deepcopy(MOT_FSDP_CONFIG["model"])
    model = config.model.config
    model.sound_gen = False
    model.action_gen = False
    model.joint_attn_implementation = "two_way"
    model.resolution = "256"
    model.state_ch = 48
    model.state_t = 1
    model.latent_downsample_factor = 16
    model.max_num_tokens_after_packing = 8192
    model.parallelism.data_parallel_shard_degree = 4
    model.compile.enabled = False
    model.ema.enabled = False
    model.rectified_flow_training_config.shift = 1
    model.rectified_flow_training_config.use_dynamic_shift = False
    model.rectified_flow_training_config.train_time_image_distribution = "logitnormal"
    model.rectified_flow_training_config.train_time_weight = "uniform"
    model.rectified_flow_training_config.loss_scale = 10.0
    model.rectified_flow_training_config.use_discrete_rf = False
    model.diffusion_expert_config.patch_spatial = 2
    model.diffusion_expert_config.max_vae_latent_side_after_patchify = 45
    model.diffusion_expert_config.unified_3d_mrope_temporal_modality_margin = 15000
    model.diffusion_expert_config.unified_3d_mrope_reset_spatial_ids = True
    model.diffusion_expert_config.enable_fps_modulation = True
    model.diffusion_expert_config.base_fps = 24

    model.tokenizer = deepcopy(Wan2pt2VAEConfig)
    model.tokenizer.bucket_name = ""
    model.tokenizer.object_store_credential_path_pretrained = None
    model.tokenizer.vae_path = os.environ.get(
        "DPO_TOKENIZER_PATH",
        "pretrained/tokenizers/video/wan2pt2/Wan2.2_VAE.pth",
    )

    model.vlm_config = deepcopy(Qwen3VLMoT_VLM_8b_Instruct_Config)
    # Cosmos3-Nano carries the Qwen3-VL vision tower; build it so the base DCP loads key for key.
    model.vlm_config.model_instance.config.include_visual = True
    model.vlm_config.tokenizer.config_variant = "hf"
    model.vlm_config.pretrained_weights.enabled = False
    model.vlm_config.pretrained_weights.backbone_path = ""
    model.vlm_config.pretrained_weights.credentials_path = ""

    config.rl = DPOConfig(
        beta=5000.0,
        grad_accum=1,
        pairs_per_step=1,
        save_every=0,
        init_from=os.environ.get("DPO_BASE_CHECKPOINT", ""),
        # The sound and action heads exist only in the Cosmos3-Nano base; T2I DPO does not build them.
        init_keys_to_skip_loading=[
            "llm2sound.weight",
            "llm2sound.bias",
            "sound2llm.weight",
            "sound2llm.bias",
            "sound_modality_embed",
            "action2llm.bias.weight",
            "action2llm.fc.weight",
            "action_modality_embed",
            "llm2action.bias.weight",
            "llm2action.fc.weight",
        ],
        optimizer=OptimizerConfig(
            name="adamw",
            lr=1e-6,
            betas=[0.9, 0.99],
            weight_decay=0.0,
            grad_clip_norm=1.0,
            train_keys=["moe_gen", "time_embedder", "vae2llm", "llm2vae"],
        ),
        ref=RefConfig(build_ref=True, update_every=0),
        data=PreferenceEnvConfig(
            synthetic_pairs=64,
            media="image",
            resolution="256",
            num_frames=1,
            shuffle=False,
        ),
    )
    return config


def make_config() -> PosttrainDPOConfig:
    """Return the config consumed by ``entrypoints.dpo --config``."""
    return build_config()
