# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Shared Super-tier (Qwen3-VL-32B) ``model.config`` baseline for SFT experiments.

Consumers must ``copy.deepcopy`` this constant before mutating it. Baseline
mirrors ``vision_sft_super`` (LoRA-only fine-tune, EMA off, no torch.compile,
DP=4 / CP=2 FSDP topology, action gen off) with the deployment-env
canonicalization applied — bucket / credentials paths empty, the VLM
tokenizer uses ``config_variant="hf"``, and ``backbone_path`` carries the
sanitized Qwen3-VL-32B URI (matching the ``NANO_MODEL_CONFIG`` policy and
keeping ``export_model --vit`` resolvable via the checkpoint catalog).

Two differences from a literal extraction of the legacy inline block:

- ``base_config`` uses ``Qwen3VLMoTConfig.from_json_file`` rather than the
  raw HF ``Qwen3VLTextConfig.from_json_file``. The MoT wrapper exposes the
  ``full_config`` property that ``OmniMoTModel.build_net`` reads at
  ``omni_mot_model.py:327`` — pointing at the raw HF config crashes training
  with ``AttributeError: 'Qwen3VLTextConfig' object has no attribute
  'full_config'``.
- ``bucket_name``, ``object_store_credential_path_pretrained``,
  ``pretrained_weights.credentials_path`` are the empty string, and
  ``vlm_config.tokenizer.config_variant="hf"``.
"""

from cosmos_framework.utils.lazy_config import LazyCall as L

from cosmos_framework.configs.base.defaults.vlm import (
    create_qwen2_tokenizer_with_download,
    create_vlm_config,
)
from cosmos_framework.model.vfm.mot.unified_mot import Qwen3VLMoTConfig, Qwen3VLTextForCausalLM


SUPER_MODEL_CONFIG = dict(
    action_gen=False,
    causal_training_strategy="none",
    input_caption_key="ai_caption",
    input_image_key="images",
    input_video_key="video",
    joint_attn_implementation="two_way",
    latent_downsample_factor=16,
    log_enc_time_every_n=100,
    lora_alpha=32,
    lora_enabled=True,
    lora_rank=16,
    lora_target_modules="q_proj_moe_gen,k_proj_moe_gen,v_proj_moe_gen,o_proj_moe_gen",
    max_action_dim=32,
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
        position_embedding_type="unified_3d_mrope",
        rope_h_extrapolation_ratio=1.0,
        rope_t_extrapolation_ratio=1.0,
        rope_w_extrapolation_ratio=1.0,
        timestep_range=1.0,
        unified_3d_mrope_reset_spatial_ids=True,
        unified_3d_mrope_temporal_modality_margin=15000,
    ),
    ema=dict(
        enabled=False,
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
        context_parallel_shard_degree=2,
        data_parallel_shard_degree=4,
        enable_inference_mode=False,
        fsdp_master_dtype="float32",
    ),
    compile=dict(
        compile_dynamic=True,
        compiled_region="language",
        coordinate_descent_tuning=False,
        max_autotune_pointwise=False,
        use_cuda_graphs=False,
        enabled=False,
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
        high_sigma_ratio=0.05,
        high_sigma_timesteps_max=1000,
        high_sigma_timesteps_min=995,
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
        use_high_sigma_strategy=False,
        use_high_sigma_strategy_action=False,
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
        model_name="Qwen/Qwen3-VL-32B-Instruct",
        tie_word_embeddings=False,
        use_system_prompt=False,
        pretrained_weights=dict(
            enabled=False,
            backbone_path=(
                "s3://bucket0/cosmos3/pretrained/huggingface/"
                "Qwen/Qwen3-VL-32B-Instruct/"
            ),
            credentials_path="",
            enable_gcs_patch_in_boto3=True,
        ),
        model_instance=L(Qwen3VLTextForCausalLM)(
            config=L(create_vlm_config)(
                base_config=L(Qwen3VLMoTConfig.from_json_file)(
                    json_file=(
                        "cosmos_framework/model/vfm/vlm/qwen3_vl/configs/"
                        "Qwen3-VL-32B-Instruct.json"
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
            pretrained_model_name="Qwen/Qwen3-VL-32B-Instruct",
        ),
    ),
)
