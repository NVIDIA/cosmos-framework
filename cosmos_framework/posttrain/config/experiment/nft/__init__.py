# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Online Cosmos3 DiffusionNFT experiment registrations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType

NFT_T2I_HPSV3_LORA_720P_PROFILE = "t2i_hpsv3_lora_720p_v1"
NFT_T2I_PUBLIC_HPSV3_LORA_720P_PROFILE = "t2i_hpsv3_lora_720p_public_v1"
NFT_T2I_PUBLIC_TOY_4GPU_PROFILE = "t2i_toy_4gpu_public_v1"




@dataclass(frozen=True)
class NFTT2IRecipeContract:
    """Qualified source, model, execution, and continuation coordinates for T2I NFT."""

    profile: str = NFT_T2I_HPSV3_LORA_720P_PROFILE
    base_checkpoint: str = (
        "s3://<checkpoint-bucket>/cosmos3_vfm/cosmos3_ga_image_only/"
        "cosmos3_ga_64bm32b_from_v2_image_only_v1p5_hq_distill/checkpoints/iter_000108000"
    )
    source_only_keys: tuple[str, ...] = (
        "llm2sound.weight",
        "llm2sound.bias",
        "sound2llm.weight",
        "sound2llm.bias",
        "sound_modality_embed",
    )
    prompt_key: str = "prompt"
    expected_rows: int = 10_000
    height: int = 720
    width: int = 720
    fps: float = 24.0
    shuffle: bool = True
    shuffle_seed: int = 0
    num_replicas: int = 8
    source_rank: int = 0
    drop_last: bool = True
    group_size: int = 8
    data_parallel_shard_degree: int = 8
    grad_accum: int = 4
    sample_steps: int = 20
    sampler_shift: float = 5.0
    resolution: str = "720"
    state_t: int = 24
    training_shift: int = 5
    max_vae_latent_side_after_patchify: int = 20
    compile_enabled: bool = False
    ema_enabled: bool = False
    sound_gen: bool = False
    action_gen: bool = False
    lora_enabled: bool = True
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_target_modules: str = "q_proj_moe_gen,k_proj_moe_gen,v_proj_moe_gen,o_proj_moe_gen"
    beta: float = 1.0
    kl_beta: float = 0.0
    policy_loss_scale: float = 5.0
    build_ref: bool = False
    old_update_every: int = 10
    advantage_low: float = -5.0
    advantage_high: float = 5.0
    advantage_eps: float = 1e-4
    r_map: str = "sigmoid"
    r_tau: float = 0.3
    uniform_timestep: bool = True
    optimizer_name: str = "adamw"
    optimizer_lr: float = 3e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-6
    optimizer_weight_decay: float = 0.0
    optimizer_train_keys: tuple[str, ...] = ("lora_",)
    optimizer_grad_clip_norm: float = 1.0
    optimizer_force_finite_grads: bool = True
    scheduler_warm_up_steps: int = 50
    scheduler_cycle_length: int = 500_000
    scheduler_f_start: float = 1e-6
    scheduler_f_max: float = 1.0
    scheduler_f_min: float = 0.5
    provider_name: str = "hpsv3"
    score_key: str = "hpsv3"
    reward_clip_min: float = -50.0
    reward_clip_max: float = 50.0
    reward_scale: float = 1.0
    guidance: float = 1.0
    rollout_seed_clock: str = "optimizer_step"
    use_system_prompt: bool = False
    tokenizer_type: str = "Qwen/Qwen3-VL-32B-Instruct"
    vlm_json_file: str = "cosmos_framework/model/generator/reasoner/qwen3_vl/configs/Qwen3-VL-32B-Instruct.json"
    qk_norm_for_text: bool = True
    joint_attn_implementation: str = "two_way"
    backbone_path: str = (
        "s3://<checkpoint-bucket>/cosmos3/pretrained/huggingface/Cosmos-Reason/Cosmos3-Reasoner-32B-Private/"
    )


NFT_T2I_HPSV3_CONTRACT = NFTT2IRecipeContract()
NFT_T2I_PUBLIC_HPSV3_CONTRACT = replace(
    NFT_T2I_HPSV3_CONTRACT,
    profile=NFT_T2I_PUBLIC_HPSV3_LORA_720P_PROFILE,
    base_checkpoint="",
    backbone_path="",
    source_only_keys=(
        *NFT_T2I_HPSV3_CONTRACT.source_only_keys,
        "action2llm.bias.weight",
        "action2llm.fc.weight",
        "action_modality_embed",
        "llm2action.bias.weight",
        "llm2action.fc.weight",
    ),
    tokenizer_type="Qwen/Qwen3-VL-8B-Instruct",
    vlm_json_file="cosmos_framework/model/generator/reasoner/qwen3_vl/configs/Qwen3-VL-8B-Instruct.json",
)
NFT_T2I_PUBLIC_TOY_4GPU_CONTRACT = replace(
    NFT_T2I_PUBLIC_HPSV3_CONTRACT,
    profile=NFT_T2I_PUBLIC_TOY_4GPU_PROFILE,
    expected_rows=4,
    num_replicas=4,
    group_size=4,
    data_parallel_shard_degree=4,
    grad_accum=1,
    sample_steps=2,
    provider_name="toy_image_hash",
    score_key="toy_image_hash",
)
NFT_T2I_PROFILES: Mapping[str, NFTT2IRecipeContract] = MappingProxyType(
    {
        NFT_T2I_PUBLIC_HPSV3_CONTRACT.profile: NFT_T2I_PUBLIC_HPSV3_CONTRACT,
        NFT_T2I_PUBLIC_TOY_4GPU_CONTRACT.profile: NFT_T2I_PUBLIC_TOY_4GPU_CONTRACT,
    }
)
