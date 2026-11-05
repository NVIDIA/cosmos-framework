# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Public 720p T2I DiffusionNFT recipe with external model, tokenizer, prompt, and reward inputs."""

from __future__ import annotations

from hydra.core.config_store import ConfigStore

from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import LazyDict
from cosmos_framework.posttrain.config.experiment.nft import NFT_T2I_PUBLIC_HPSV3_CONTRACT
from cosmos_framework.posttrain.config.rl import (
    ImageRewardConfig,
    NFTConfig,
    OldPolicyConfig,
    OptimizerConfig,
    RefConfig,
    WFMLRSchedulerConfig,
)
from cosmos_framework.posttrain.data.datasets import NFTPromptImageDataLoader

_PROFILE = NFT_T2I_PUBLIC_HPSV3_CONTRACT

# Unlike the internal qualified recipe, this public recipe does not inherit an
# experiment carrying cluster, object-store, dataset, or checkpoint defaults.
# Every artifact outside the framework is an explicit environment input.
cosmos3_nft_t2i_public = LazyDict(
    dict(
        defaults=[
            {"override /data_train": None},
            {"override /data_val": None},
            {"override /model": "mot_fsdp"},
            {"override /optimizer": "fusedadamw"},
            {"override /scheduler": "lambdalinear"},
            {"override /tokenizer": "wan2pt2_tokenizer"},
            {"override /vlm_config": "qwen3_vl_mot_vlm_8b_instruct"},
            "_self_",
        ],
        algorithm="nft",
        grid=dict(
            backend="ray",
            grouping="single",
            gpu_placement="colocated",
            num_replicas=1,
            ranks=_PROFILE.group_size,
        ),
        trainer=dict(
            cudnn=dict(
                deterministic=True,
                benchmark=False,
            )
        ),
        checkpoint=dict(load_path=""),
        model=dict(
            config=dict(
                sound_gen=_PROFILE.sound_gen,
                action_gen=_PROFILE.action_gen,
                joint_attn_implementation=_PROFILE.joint_attn_implementation,
                resolution=_PROFILE.resolution,
                state_ch=48,
                state_t=_PROFILE.state_t,
                latent_downsample_factor=16,
                max_num_tokens_after_packing=8192,
                lora_enabled=_PROFILE.lora_enabled,
                lora_rank=_PROFILE.lora_rank,
                lora_alpha=_PROFILE.lora_alpha,
                lora_target_modules=_PROFILE.lora_target_modules,
                parallelism=dict(data_parallel_shard_degree=_PROFILE.data_parallel_shard_degree),
                compile=dict(enabled=_PROFILE.compile_enabled),
                ema=dict(enabled=_PROFILE.ema_enabled),
                tokenizer=dict(
                    bucket_name="",
                    object_store_credential_path_pretrained=None,
                    vae_path="${oc.env:NFT_TOKENIZER_PATH}",
                ),
                vlm_config=dict(
                    use_system_prompt=_PROFILE.use_system_prompt,
                    model_instance=dict(config=dict(include_visual=True)),
                    tokenizer=dict(config_variant="hf"),
                    pretrained_weights=dict(
                        enabled=False,
                        backbone_path=_PROFILE.backbone_path,
                        credentials_path="",
                    ),
                ),
                rectified_flow_training_config=dict(
                    shift=int(_PROFILE.training_shift),
                    use_dynamic_shift=False,
                    train_time_image_distribution="logitnormal",
                    train_time_weight="uniform",
                    loss_scale=10.0,
                    use_discrete_rf=False,
                ),
                rectified_flow_inference_config=dict(
                    scheduler_type="unipc",
                    num_train_timesteps=1000,
                    shift=1,
                    use_dynamic_shifting=False,
                ),
                diffusion_expert_config=dict(
                    patch_spatial=2,
                    max_vae_latent_side_after_patchify=_PROFILE.max_vae_latent_side_after_patchify,
                    unified_3d_mrope_temporal_modality_margin=15000,
                    unified_3d_mrope_reset_spatial_ids=True,
                    enable_fps_modulation=True,
                    base_fps=24,
                ),
            )
        ),
        dataloader_train=L(NFTPromptImageDataLoader)(
            path="${oc.env:NFT_T2I_PROMPTS_JSONL}",
            prompt_key=_PROFILE.prompt_key,
            height=_PROFILE.height,
            width=_PROFILE.width,
            fps=_PROFILE.fps,
            expected_sha256="${oc.env:NFT_T2I_PROMPTS_SHA256}",
            expected_rows=_PROFILE.expected_rows,
            shuffle=_PROFILE.shuffle,
            shuffle_seed=_PROFILE.shuffle_seed,
            num_replicas=_PROFILE.num_replicas,
            source_rank=_PROFILE.source_rank,
            drop_last=_PROFILE.drop_last,
        ),
        rl=NFTConfig(
            max_steps=400,
            save_every=10,
            grad_accum=_PROFILE.grad_accum,
            checkpoint_format="dcp",
            save_async=True,
            keep_optim_last=1,
            keep_model_last=2,
            seed=0,
            beta=_PROFILE.beta,
            kl_beta=_PROFILE.kl_beta,
            policy_loss_scale=_PROFILE.policy_loss_scale,
            advantage_high=_PROFILE.advantage_high,
            advantage_low=_PROFILE.advantage_low,
            r_map=_PROFILE.r_map,
            r_tau=_PROFILE.r_tau,
            uniform_timestep=_PROFILE.uniform_timestep,
            modality="image",
            profile=_PROFILE.profile,
            group_size=_PROFILE.group_size,
            sample_steps=_PROFILE.sample_steps,
            sampler_shift=_PROFILE.sampler_shift,
            guidance=_PROFILE.guidance,
            advantage_eps=_PROFILE.advantage_eps,
            init_from="${oc.env:NFT_BASE_CHECKPOINT}",
            init_keys_to_skip_loading=list(_PROFILE.source_only_keys),
            rollout_seed_clock=_PROFILE.rollout_seed_clock,
            optimizer=OptimizerConfig(
                name=_PROFILE.optimizer_name,
                lr=_PROFILE.optimizer_lr,
                betas=list(_PROFILE.optimizer_betas),
                weight_decay=_PROFILE.optimizer_weight_decay,
                eps=_PROFILE.optimizer_eps,
                train_keys=list(_PROFILE.optimizer_train_keys),
                grad_clip_norm=_PROFILE.optimizer_grad_clip_norm,
                force_finite_grads=_PROFILE.optimizer_force_finite_grads,
            ),
            old=OldPolicyConfig(build_old=True, update_every=_PROFILE.old_update_every),
            ref=RefConfig(build_ref=_PROFILE.build_ref, update_every=0),
            scheduler=WFMLRSchedulerConfig(
                warm_up_steps=_PROFILE.scheduler_warm_up_steps,
                cycle_length=_PROFILE.scheduler_cycle_length,
                f_start=_PROFILE.scheduler_f_start,
                f_max=_PROFILE.scheduler_f_max,
                f_min=_PROFILE.scheduler_f_min,
            ),
            reward=ImageRewardConfig(
                provider_revision="${oc.env:NFT_REWARD_REVISION}",
                provider_name=_PROFILE.provider_name,
                score_key=_PROFILE.score_key,
                connect_timeout_s=30.0,
                operation_timeout_s=60.0,
                total_deadline_s=600.0,
                poll_interval_s=1.0,
                pull_max_attempts=3,
                reward_clip_min=_PROFILE.reward_clip_min,
                reward_clip_max=_PROFILE.reward_clip_max,
                scale=_PROFILE.reward_scale,
            ),
        ),
        job=dict(
            project="cosmos3_vfm",
            group="cosmos3_nft_t2i_public",
            name="cosmos3_nft_t2i_public",
            wandb_mode="disabled",
        ),
    ),
    flags={"allow_objects": True},
)

ConfigStore.instance().store(
    group="experiment",
    package="_global_",
    name="cosmos3_nft_t2i_public",
    node=cosmos3_nft_t2i_public,
)
