# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Shared Nano-tier (Qwen3-VL-8B) ``model.config`` baseline for SFT experiments.

Consumers must ``copy.deepcopy`` this constant before mutating it. Baseline
mirrors ``vision_sft_nano`` (HF-cluster deployment with empty tokenizer/vlm
paths, video-style loss scales, ``load_weights_from_pretrained=True``).
"""

from cosmos_framework.configs.base.defaults.reasoner import (
    create_qwen2_tokenizer_with_download,
    create_vlm_config,
)
from cosmos_framework.model.generator.mot.unified_mot import Qwen3VLMoTConfig, Qwen3VLTextForCausalLM
from cosmos_framework.utils.lazy_config import LazyCall as L

NANO_MODEL_CONFIG = dict(
    action_gen=True,
    causal_training_strategy="none",
    input_caption_key="ai_caption",
    input_image_key="images",
    input_video_key="video",
    joint_attn_implementation="two_way",
    latent_downsample_factor=16,
    log_enc_time_every_n=100,
    max_action_dim=64,
    max_num_tokens_after_packing=45056,
    num_embodiment_domains=32,
    resolution="720",
    sound_gen=False,
    sound_latent_fps=25,
    state_ch=48,
    state_t=300,
    video_temporal_causal=False,
    vision_gen=True,
    diffusion_expert_config=dict(
        base_fps=24,
        enable_fps_modulation=True,
        load_weights_from_pretrained=True,
        max_vae_latent_side_after_patchify=20,
        patch_spatial=2,
        timestep_range=1.0,
        unified_3d_mrope_reset_spatial_ids=True,
        unified_3d_mrope_temporal_modality_margin=15000,
    ),
    ema=dict(
        enabled=True,
        iteration_shift=0,
        rate=0.1,
    ),
    lbl=dict(
        coeff_gen=None,
        coeff_und=None,
        method="local",
    ),
    parallelism=dict(
        cfg_parallel_shard_degree=1,
        context_parallel_shard_degree=1,
        data_parallel_shard_degree=8,
        enable_inference_mode=False,
        fsdp_master_dtype="float32",
    ),
    compile=dict(
        compile_dynamic=True,
        compiled_region="language",
        coordinate_descent_tuning=False,
        max_autotune_pointwise=False,
        use_cuda_graphs=False,
        enabled=True,
    ),
    precision="bfloat16",
    activation_checkpointing=dict(
        mode="full",
    ),
    rectified_flow_inference_config=dict(
        num_train_timesteps=1000,
        scheduler_type="unipc",
        shift=1,
        use_dynamic_shifting=False,
    ),
    rectified_flow_training_config=dict(
        action_loss_weight=10.0,
        image_loss_scale=1.0,
        independent_action_schedule=False,
        loss_scale=1.0,
        normalize_loss_by_active=False,
        shift={"256": 3, "480": 5, "720": 10},
        train_time_action_distribution="logitnormal",
        train_time_image_distribution="logitnormal",
        train_time_sound_distribution="logitnormal",
        train_time_video_distribution="waver",
        train_time_weight="uniform",
        use_discrete_rf=False,
        use_dynamic_shift=False,
    ),
    tokenizer=dict(
        bucket_name="",
        chunk_duration=93,
        encode_chunk_frames={"256": 68, "480": 24, "720": 12},
        encode_exact_durations=None,
        keep_decoder_cache=False,
        object_store_credential_path_pretrained="",
        spatial_compression_factor=16,
        temporal_compression_factor=4,
        use_streaming_encode=False,
        vae_path="pretrained/tokenizers/video/wan2pt2/Wan2.2_VAE.pth",
    ),
    vlm_config=dict(
        layer_module="Qwen2MoTDecoderLayer",
        model_name="Qwen/Qwen3-VL-8B-Instruct",
        tie_word_embeddings=False,
        use_system_prompt=False,
        pretrained_weights=dict(
            enabled=False,
            backbone_path=(
                "s3://bucket0/cosmos3/pretrained/huggingface/"
                "Qwen/Qwen3-VL-8B-Instruct/"
            ),
            credentials_path="",
            enable_gcs_patch_in_boto3=True,
        ),
        model_instance=L(Qwen3VLTextForCausalLM)(
            config=L(create_vlm_config)(
                base_config=L(Qwen3VLMoTConfig.from_json_file)(
                    json_file=(
                        "cosmos_framework/model/generator/reasoner/qwen3_vl/configs/"
                        "Qwen3-VL-8B-Instruct.json"
                    ),
                ),
                freeze_und=False,
                layer_module="MoTDecoderLayer",
                qk_norm_for_text=True,
                tie_word_embeddings=True,
            ),
        ),
        tokenizer=L(create_qwen2_tokenizer_with_download)(
            config_variant="hf",
            pretrained_model_name="Qwen/Qwen3-VL-8B-Instruct",
        ),
    ),
)
