# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Shared Edge-tier (Nemotron-2B-Dense-VL) ``model.config`` baseline for SFT experiments.

Consumers must ``copy.deepcopy`` this constant before mutating it. Baseline
mirrors ``vision_sft_edge`` (HF-cluster deployment with empty tokenizer/vlm
paths, video-style loss scales, ``load_weights_from_pretrained=True``), except
``action_gen``: the baseline keeps the Cosmos3-Edge.yaml value (``True``) and
``vision_sft_edge`` overrides it to ``False`` (no action tokens in vision SFT).

Derived from ``nano_model_config.NANO_MODEL_CONFIG``; every field is identical to
the Nano baseline except the Cosmos3-Edge deltas sourced verbatim from
``inference/configs/model/Cosmos3-Edge.yaml``:
  * ``vlm_config`` swapped from the Qwen3-VL backbone to the Nemotron-3 Dense VL
    backbone (``Nemotron3DenseVLTextForCausalLM`` / ``Nemotron3DenseVLMoTConfig``,
    ``Nemotron-2B-Dense-VL.json``, ``nvidia/Cosmos3-Edge-Reasoner``,
    ``build_processor_lazy(repository="nvidia/Cosmos3-Edge")`` tokenizer,
    ``qk_norm_for_text=False``, ``include_visual=None``, ``layer_module=None``).
  * ``resolution`` ``"720"`` -> ``"480"`` (Edge native inference res).
  * ``rectified_flow_training_config``: ``loss_scale`` ``1.0`` -> ``10.0``,
    ``image_loss_scale`` ``1.0`` -> ``None``, plus the Edge-only ``sound_loss_scale``
    delta (dataclass-supported keys only).
"""

from cosmos_framework.configs.base.defaults.reasoner import create_vlm_config
from cosmos_framework.data.generator.processors import build_processor_lazy
from cosmos_framework.model.generator.mot.unified_mot import (
    Nemotron3DenseVLMoTConfig,
    Nemotron3DenseVLTextForCausalLM,
)
from cosmos_framework.utils.lazy_config import LazyCall as L

EDGE_MODEL_CONFIG = dict(
    # Mirrors Cosmos3-Edge.yaml (action_gen: true), same as the nano baseline.
    # Safe since the renewed (2026-07-16) checkpoint ships trained action-head
    # weights (pre-renewal ones lacked them and NaN'd — see vision_sft_edge.py);
    # recipes that don't train action data still override to False.
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
    resolution="480",
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
        image_loss_scale=None,  # Edge delta (nano=1.0); Cosmos3-Edge.yaml image_loss_scale: null
        independent_action_schedule=False,
        # Edge-only keys accepted by RectifiedFlowTrainingConfig (nano omits them):
        independent_sound_schedule=False,
        loss_scale=10.0,  # Edge delta (nano=1.0); Cosmos3-Edge.yaml loss_scale: 10.0
        normalize_loss_by_active=False,
        shift={"256": 3, "480": 5, "720": 10},
        shift_action=None,
        shift_sound=None,
        sound_loss_scale=2.0,  # Edge-only (nano leaves default None); unused while sound_gen=False
        train_time_action_distribution="logitnormal",
        train_time_image_distribution="logitnormal",
        train_time_sound_distribution="logitnormal",
        train_time_video_distribution="waver",
        train_time_weight="uniform",
        use_discrete_rf=False,
        use_dynamic_shift=False,
        # Dropped Cosmos3-Edge.yaml keys — NOT fields of RectifiedFlowTrainingConfig:
        #   high_sigma_ratio, high_sigma_timesteps_max, high_sigma_timesteps_min,
        #   use_high_sigma_strategy, use_high_sigma_strategy_action,
        #   use_high_sigma_strategy_sound (high-sigma strategy unsupported in the
        #   training-package dataclass; nano omits them and trains fine).
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
        # Top-level VLMConfig fields (Cosmos3-Edge.yaml vlm_config.*):
        layer_module=None,  # Edge.yaml vlm_config.layer_module: null (nano="Qwen2MoTDecoderLayer")
        model_name="nvidia/Cosmos3-Edge-Reasoner",
        tie_word_embeddings=False,
        use_system_prompt=False,
        # Edge.yaml sets vlm_config.qk_norm=False; VLMConfig.qk_norm already defaults
        # to False (nano omits it too), so it is intentionally left unset.
        pretrained_weights=dict(
            enabled=False,  # SFT loads from the DCP, not the HF backbone
            backbone_path=(
                "s3://bucket/cosmos3/pretrained/huggingface/"
                "nvidia/Cosmos3-Edge-Reasoner-590c1c0/"
            ),  # kept for parity, unused while enabled=False
            credentials_path="",
            enable_gcs_patch_in_boto3=True,
        ),
        model_instance=L(Nemotron3DenseVLTextForCausalLM)(
            config=L(create_vlm_config)(
                base_config=L(Nemotron3DenseVLMoTConfig.from_json_file)(
                    json_file=(
                        "cosmos_framework/model/generator/reasoner/nemotron_3_dense_vl/configs/"
                        "Nemotron-2B-Dense-VL.json"
                    ),
                ),
                freeze_und=False,
                layer_module="MoTDecoderLayer",
                qk_norm_for_text=False,  # Edge delta (nano=True); Edge.yaml create_vlm_config.qk_norm_for_text: false
                include_visual=None,  # Edge delta (nano omits -> default); Edge.yaml create_vlm_config.include_visual: null
                tie_word_embeddings=True,
            ),
        ),
        tokenizer=L(build_processor_lazy)(
            repository="nvidia/Cosmos3-Edge",
            revision="main",
        ),
    ),
)
