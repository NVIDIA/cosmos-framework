# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""``action_policy_robocasa_nano`` — RoboCasa mobile-base action policy.

Trains Cosmos3-Nano into a mobile-manipulation policy on the 18 RoboCasa ``target/atomic``
tasks. The action contract is 15-dimensional and carries RoboCasa's native base command
unchanged (``base_encoding="raw"``)::

    [base_motion(4), control_mode(1), eef_pos(3), eef_rot6d(6), gripper(1)]

``base_motion`` is the command sent to the base controller, so the closed-loop client writes it
straight back (``env[7:11] = action[0:4]``) and a recorded demonstration replays exactly. It is
already normalised to [-1, 1], i.e. the same scale as the arm block, which matters because
``action_normalization=None``: the state token is absolute, so delta statistics do not apply and
the raw channel scales are what the loss sees.

Observation is ``camera_set="left_wrist"`` — ``agentview_left`` and ``eye_in_hand`` concatenated
horizontally at their native 256x256 each — plus end-effector proprioception prepended as a
clean condition token (``use_state=True``). Action chunk length 32 at 20 fps, with
``tokenizer.encode_exact_durations=[33]`` pinned to the matching 33-frame observation window
(chunk + 1).

Closed-loop evaluation must use the same contract::

    ACTION_HORIZON=32 CAMERA_SET=left_wrist USE_STATE=1
    USE_BASE_ACTION=1 BASE_ENCODING=raw RAW_ACTION_DIM=15
"""

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.experiment.sft.models.nano_model_config import NANO_MODEL_CONFIG
from cosmos_framework.data.generator.action.datasets.action_sft_dataset import get_action_robocasa_sft_dataset
from cosmos_framework.data.generator.action.datasets.robocasa_lerobot_dataset import DEFAULT_ALL_ATOMIC_TASKS
from cosmos_framework.data.generator.joint_dataloader import (
    PackingDataLoader,
    RankPartitionedDataLoader,
)
from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import LazyDict

cs = ConfigStore.instance()


def _lw_state_model_config() -> dict:
    cfg = copy.deepcopy(NANO_MODEL_CONFIG)  # action_gen=True, max_action_dim=64
    cfg["max_num_tokens_after_packing"] = 74000
    cfg["activation_checkpointing"]["mode"] = "selective"
    cfg["diffusion_expert_config"]["load_weights_from_pretrained"] = False
    cfg["rectified_flow_training_config"]["loss_scale"] = 10.0
    cfg["rectified_flow_training_config"]["image_loss_scale"] = None
    # chunk_length=32 -> 33 observation frames; pin the VAE encode duration to match
    # (mirrors action_policy_droid_nano, which also trains with a 32-step chunk).
    cfg["tokenizer"]["encode_exact_durations"] = [33]
    return cfg


action_policy_robocasa_nano = LazyDict(
    dict(
        defaults=[
            {"override /model": "mot_fsdp"},
            {"override /data_train": None},
            {"override /data_val": None},
            {"override /optimizer": "fusedadamw"},
            {"override /scheduler": "lambdalinear"},
            {"override /checkpoint": "s3"},
            {"override /callbacks": ["basic", "optimization", "job_monitor"]},
            {"override /ema": "power"},
            {"override /tokenizer": "wan2pt2_tokenizer"},
            {"override /sound_tokenizer": None},
            {"override /vlm_config": None},
            {"override /ckpt_type": "dcp"},
            "_self_",
        ],
        job=dict(
            project="cosmos3",
            group="action_sft",
            name="action_policy_robocasa_nano",
            wandb_mode="disabled",
        ),
        model=dict(config=_lw_state_model_config()),
        optimizer=dict(
            betas=[0.9, 0.99],
            eps=1.0e-08,
            fused=True,
            keys_to_select=[
                "moe_gen", "time_embedder", "vae2llm", "llm2vae",
                "action2llm", "llm2action", "action_modality_embed",
            ],
            lr=5.0e-05,
            lr_multipliers={"action2llm": 5.0, "llm2action": 5.0, "action_modality_embed": 5.0},
            optimizer_type="FusedAdam",
            weight_decay=0.05,
        ),
        scheduler=dict(
            lr_scheduler_type="LambdaLinear",
            cycle_lengths=[100],
            f_max=[1.0], f_min=[0.0], f_start=[1.0e-06],
            verbosity_interval=0, warm_up_steps=[0],
        ),
        trainer=dict(
            distributed_parallelism="fsdp",
            grad_accum_iter=1, logging_iter=1, max_iter=100, max_val_iter=None,
            run_validation=False, run_validation_on_start=False,
            save_zero_checkpoint=False, seed=42, timeout_period=999999999,
            validation_iter=100,
            compile_config=dict(recompile_limit=8, use_duck_shape=False),
            cudnn=dict(benchmark=True, deterministic=False),
            ddp=dict(broadcast_buffers=True, find_unused_parameters=False, static_graph=True),
            grad_scaler_args=dict(enabled=False),
            callbacks=dict(
                dataloader_speed=dict(every_n=100, save_s3=False, step_size=1),
                device_monitor=dict(every_n=200, log_memory_detail=True, save_s3=False, step_size=1, upload_every_n_mul=5),
                grad_clip=dict(clip_norm=1.0, force_finite=True),
                heart_beat=dict(every_n=200, save_s3=False, step_size=1, update_interval_in_minute=20),
                iter_speed=dict(every_n=1, hit_thres=50, save_s3=False, save_s3_every_log_n=500),
                low_precision=dict(update_iter=1),
                manual_gc=dict(every_n=5, gc_level=1, warm_up=1),
                param_count=dict(save_s3=False),
                skip_nan_step=dict(max_consecutive_nan=100),
                training_stats=dict(log_freq=100),
            ),
        ),
        checkpoint=dict(
            broadcast_via_filesystem=False,
            dcp_async_mode_enabled=False,
            enable_gcs_patch_in_boto3=True,
            keys_not_to_resume=[],
            keys_to_skip_loading=[
                "net_ema.", "action2llm", "llm2action", "action_modality_embed", "action_pos_embed",
            ],
            load_ema_to_reg=False,
            load_path="???",
            load_training_state=False,
            only_load_scheduler_state=False,
            save_iter=100,
            strict_resume=False,
            verbose=True,
            hf_export=dict(enabled=False, export_every_n=1, hf_repo_id=None,
                           upload_to_object_store=dict(bucket="", credentials="", enabled=False)),
            jit=dict(device="cuda", dtype="bfloat16", enabled=False, input_shape=None, strict=True),
            load_from_object_store=dict(bucket="", credentials="", enabled=False),
            save_to_object_store=dict(bucket="", credentials="", enabled=False),
        ),
        dataloader_train=L(PackingDataLoader)(
            audio_sample_rate=48000,
            dataset_name="action_robocasa",
            max_samples_per_batch=128,
            max_sequence_length=None,
            patch_spatial=2,
            sound_latent_fps=0,
            tokenizer_spatial_compression_factor=16,
            tokenizer_temporal_compression_factor=4,
            dataloader=L(RankPartitionedDataLoader)(
                batch_size=1, in_order=False, num_workers=4,
                persistent_workers=True, pin_memory=True, prefetch_factor=4, sampler=None,
                datasets=dict(
                    robocasa=dict(
                        ratio=1,
                        dataset=L(get_action_robocasa_sft_dataset)(
                            root="${oc.env:ROBOCASA_ROOT}",
                            task_names=DEFAULT_ALL_ATOMIC_TASKS,   # all 18 atomic tasks (NavigateKitchen included)
                            use_base_action=True,
                            base_encoding="raw",   # 15D: native base_motion(4) + control_mode + arm
                            fps=20,
                            chunk_length=32,   # 32-step action chunk (DROID recipe uses 32); eval must use ACTION_HORIZON=32
                            mode="wam",
                            viewpoint="concat_view",
                            camera_set="left_wrist",   # agentview_left + wrist, full-res 256x512
                            use_state=True,   # EEF proprioception -> prepended clean condition token
                            action_normalization=None,   # absolute state token: delta-stats do not apply
                            split="train",
                            split_val_ratio=0.01,
                            iterable_shuffle=True,
                            episode_shuffle_seed=42,
                            resolution=None,   # keep 256x512 horizontal concat un-squished
                            max_action_dim="${model.config.max_action_dim}",
                            cfg_dropout_rate=0.1,
                            format_prompt_as_json=True,
                            tokenizer_config="${model.config.vlm_config.tokenizer}",
                        ),
                    ),
                ),
            ),
        ),
        dataloader_val=None,
        upload_reproducible_setup=False,
    ),
    flags={"allow_objects": True},
)


for _item in [action_policy_robocasa_nano]:
    _name = [k for k, v in globals().items() if v is _item][0]
    cs.store(group="experiment", package="_global_", name=_name, node=_item)
