# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Single-device native Qwen3-VL-8B LoRA recipe with a frozen BF16 base.

``sft_qwen3_vl_8b_qlora.py`` is the same recipe with a frozen NF4 base.
Set SFT_MODEL_PATH to a local snapshot prepared by
``cosmos_framework.posttrain.scripts.prepare_qlora_source`` and SFT_DATA_ROOT to a
directory with train.jsonl and validation.jsonl. Export CUBLAS_WORKSPACE_CONFIG=:4096:8
and PYTHONHASHSEED before the process starts (determinism must precede CUDA init).
Launch with torchrun --nproc-per-node=1 through cosmos_framework.posttrain.entrypoints.sft.
Short runs must override rl.max_steps, rl.lr_schedule.cycle_steps and
rl.optimizer.warmup_steps together using the entrypoint's dotlist.
"""

from __future__ import annotations

import os

from cosmos_framework.posttrain.config.config_sft import PosttrainSFTConfig
from cosmos_framework.posttrain.config.config_sft import make_config as make_root_config
from cosmos_framework.posttrain.config.model import (
    NativeLoRAConfig,
    NativePolicyBuildSpec,
    NativePolicyParallelismConfig,
    QuantizedBaseTrainingConfig,
    Qwen3VLModelConfig,
)
from cosmos_framework.posttrain.config.rl import OptimizerConfig
from cosmos_framework.posttrain.config.sft import SFTDatasetConfig, SFTLRScheduleConfig, SFTValidationConfig
from cosmos_framework.posttrain.utils.adapter_source import SOURCE_REPOSITORY, SOURCE_REVISION

NATIVE_POLICY_BUILDER = "cosmos_framework.posttrain.model.qwen3_vl.build_qwen3_vl_native_policy"


def build_config(*, quantized: bool = False) -> PosttrainSFTConfig:
    """The BF16 and NF4 recipes differ only in the frozen language base format and job name."""
    config = make_root_config()
    # The source preflight requires a local snapshot; the default is deliberately a path,
    # never a remote model ID that could download dense weights on the training device.
    model_id = os.environ.get("SFT_MODEL_PATH", f"checkpoints/Qwen3-VL-8B-Instruct/{SOURCE_REVISION}")
    data_root = os.environ.get("SFT_DATA_ROOT", "data/sft")
    config.job.project = "cosmos3_reasoner"
    config.job.group = "qwen3_vl_8b_native_adapter"
    config.job.name = "qlora_nf4_seed42" if quantized else "lora_bf16_seed42"
    config.job.wandb_mode = "disabled"
    config.model_backend = "native"
    config.model = NativePolicyBuildSpec(
        builder=NATIVE_POLICY_BUILDER,
        model_id=model_id,
        model_config=Qwen3VLModelConfig(
            revision=SOURCE_REVISION,
            dtype="bfloat16",
            attn_impl="sdpa",
            activation_checkpointing="full",
            fsdp_reduce_dtype="float32",
            adapter=NativeLoRAConfig(),
            quantized_base=QuantizedBaseTrainingConfig() if quantized else None,
            source_identity={"repository": SOURCE_REPOSITORY, "revision": SOURCE_REVISION},
        ),
        parallelism=NativePolicyParallelismConfig(dp_replicate=1, dp_shard=1),
        requires_collated_policy_forward=True,
    )
    config.rl.seed = 42
    config.rl.deterministic = True
    # These describe the native base. The builder owns FP32 adapter parameters and
    # casts adapter inputs/output explicitly, without a model-wide dtype conversion.
    config.rl.master_dtype = "bfloat16"
    config.rl.param_dtype = "bfloat16"
    config.rl.reduce_dtype = "float32"
    config.rl.mixed_precision_mode = "fsdp"
    config.rl.max_steps = 300
    config.rl.micro_batch_size = 1
    config.rl.grad_accum = 8
    config.rl.max_length = 4096
    config.rl.checkpoint_format = "dcp"
    config.rl.save_async = False
    config.rl.save_every = 50
    config.rl.optimizer = OptimizerConfig(
        name="adamw",
        impl="fused",
        train_keys=[],
        lr=1e-4,
        betas=[0.9, 0.999],
        eps=1e-8,
        weight_decay=0.01,
        grad_clip_norm=1.0,
        warmup_steps=30,
    )
    config.rl.lr_schedule = SFTLRScheduleConfig(kind="cosine", f_start=0.01, f_min=0.1, cycle_steps=300)
    config.rl.ce_exponent = 0.0
    config.rl.mask_think_opener = False
    config.rl.serve_contract = "tools_no_prefill"
    config.rl.moe_lbl_coeff = 0.0
    config.rl.data = SFTDatasetConfig(path=os.path.join(data_root, "train.jsonl"))
    config.rl.validation = SFTValidationConfig(
        enabled=True,
        before_training=True,
        every_steps=50,
        batch_size=1,
        seed=42,
        data=SFTDatasetConfig(path=os.path.join(data_root, "validation.jsonl"), shuffle=False),
    )
    video_frames = 8
    config.rl.processor_kwargs = {
        # Image size values are pixel areas. The pinned Transformers 5.8.1 Qwen3-VL
        # video processor compares T*H*W with size, so multiply per-frame area by T.
        "images_kwargs": {"size": {"shortest_edge": 65_536, "longest_edge": 262_144}},
        "videos_kwargs": {
            "size": {"shortest_edge": video_frames * 4096, "longest_edge": video_frames * 65_536},
            "num_frames": video_frames,
            "fps": None,
        },
    }
    return config


def make_config() -> PosttrainSFTConfig:
    return build_config()
