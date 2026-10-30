# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Differentiable public Alpamayo prefix/action forward, preserving model names."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import torch
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1

from cosmos_framework.posttrain.config.build import Precision
from cosmos_framework.posttrain.model.adapters import assert_trainability, prepare_adapter_model, resolve_targets
from cosmos_framework.posttrain.model.alpamayo.sft import REFERENCE_REVISION, require_runtime, source_config
from cosmos_framework.posttrain.model.protocol import FlowMatchingLossOutput


class AlpamayoSFTNet(AlpamayoR1):
    def forward(self, tokenized, history, actions, action_mask, *, time=None, noise=None):
        if (
            actions.ndim != 3
            or actions.shape[1:] != self.action_space.get_action_space_dims()
            or action_mask.shape != actions.shape
        ):
            raise ValueError("Alpamayo flow inputs disagree with its action-space dimensions")
        ids = self.fuse_traj_tokens(tokenized["input_ids"], history)
        prefix = self.vlm.model(**{**tokenized, "input_ids": ids}, use_cache=True, return_dict=True)
        cache = prefix.past_key_values
        if cache is None:
            raise ValueError(
                "Alpamayo flow requires differentiable prefix K/V; generic language gradient checkpointing is unsupported"
            )
        length = cache.get_seq_length()
        horizon = self.action_space.get_action_space_dims()[0]
        positions = torch.arange(horizon, device=actions.device)[None, None, :].expand(3, actions.shape[0], -1)
        positions = positions + (prefix.rope_deltas + length)[None]
        attention = torch.zeros(
            (actions.shape[0], 1, horizon, length + horizon), device=actions.device, dtype=torch.float32
        )
        if "attention_mask" in tokenized:
            attention[..., :length].masked_fill_(
                ~tokenized["attention_mask"].bool()[:, None, None, :], torch.finfo(torch.float32).min
            )
        if time is None and noise is None:
            training = self.diffusion.construct_training_data(actions)
        elif time is not None and noise is not None:
            t = time[:, None, None]
            training = {"x": actions, "noise": noise, "timesteps": t, "noisy_x": t * actions + (1 - t) * noise}
        else:
            raise ValueError("controlled Alpamayo flow requires both time and noise")
        embedding = self.action_in_proj(training["noisy_x"], training["timesteps"])
        output = self.expert(
            inputs_embeds=embedding,
            position_ids=positions,
            past_key_values=cache,
            attention_mask=attention,
            use_cache=True,
            is_causal=False,
        )
        prediction = self.action_out_proj(output.last_hidden_state[:, -horizon:]).float()
        # One mask weight per whole physical trajectory; native flow loss owns
        # its target/direction. No text-generation loss is implied by this seat.
        weights = action_mask[:, 0, 0]
        if not torch.equal(action_mask, weights[:, None, None].expand_as(action_mask)):
            raise ValueError("Alpamayo SFT mask must supervise whole 64x2 trajectories or validation padding")
        losses = torch.stack(
            [
                self.diffusion.compute_loss_from_pred(
                    {k: v[i : i + 1] for k, v in training.items() if isinstance(v, torch.Tensor)}, prediction[i : i + 1]
                )
                for i in range(actions.shape[0])
            ]
        )
        loss = (losses * weights).sum() / weights.sum().clamp_min(1)
        return FlowMatchingLossOutput(loss, action_mask, prediction)


class AlpamayoSFTPolicy:
    precision = Precision(master="bfloat16", param="bfloat16", reduce="float32", transfer=None, mode="autocast")

    def __init__(self, net):
        self.net = net

    def flow_matching_loss(self, inputs):
        device = next(self.net.parameters()).device

        def move(value):
            if isinstance(value, dict):
                return {k: move(v) for k, v in value.items()}
            return value.to(device) if isinstance(value, torch.Tensor) else value

        # The official action projection produces FP32 Fourier features; base
        # Linear autocast and FP32 adapter accumulation retain separate policies.
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=next(self.net.parameters()).dtype != torch.float32
        ):
            return self.net(**move(inputs))


def validate_targets(net, config):
    resolved = resolve_targets(net, config, family="alpamayo_r1")
    root = "vlm.model.language_model."
    last = len(net.vlm.model.language_model.layers) - 1
    prefix = f"{root}layers.{last}."
    for row in resolved["targets"]:
        if row["name"].startswith(prefix) and row["name"][len(prefix) :] not in (
            "self_attn.k_proj",
            "self_attn.v_proj",
        ):
            raise ValueError(
                "Alpamayo action loss reads final-prefix K/V only; select k_proj/v_proj or an earlier layer"
            )
    if any(row["name"].startswith(root + "norm.") for row in resolved["extra_parameters"]):
        raise ValueError("Alpamayo action loss does not consume the final prefix norm")


def external_identity():
    import alpamayo_r1.models.alpamayo_r1 as native_module

    from cosmos_framework.posttrain.utils.identity import canonical_json_sha256
    from cosmos_framework.posttrain.utils.object_store import file_sha256

    # The public distribution is a namespace package without __init__.py.
    root = Path(native_module.__file__).parents[1]
    files = {str(p.relative_to(root)): file_sha256(str(p)) for p in root.rglob("*.py")}
    digest = canonical_json_sha256(files, context="Alpamayo source")
    if digest != "4363fee75791d3c465e7b1374451fafe5ea9b93a1af582ebe1251d60fb6371eb":
        raise ValueError(f"Alpamayo SFT requires the public source at {REFERENCE_REVISION}; installed source differs")
    return {
        "reference_revision": REFERENCE_REVISION,
        "source_sha256": digest,
        "files": files,
    }


def load_base(model_id, config, *, dtype=torch.bfloat16):
    """Strict native construction shared by training and private merged export."""
    from cosmos_framework.posttrain.checkpoint.adapters import snapshot_identity

    if (Path(model_id) / "merged_model.json").exists():
        raise ValueError("Alpamayo merged artifacts require model.alpamayo.adapter_export.load_merged")
    cfg = source_config(model_id, config)
    cfg.model_dtype = str(dtype).removeprefix("torch.")
    cfg.expert_cfg = {**(cfg.expert_cfg or {}), "dtype": cfg.model_dtype}
    base = snapshot_identity(model_id, weights=True)
    # The outer public class does not own an attention implementation flag;
    # its model-owned config selects SDPA on the actual VLM/expert children.
    net, loading = AlpamayoSFTNet.from_pretrained(
        model_id, config=cfg, dtype=dtype, local_files_only=True, output_loading_info=True
    )
    if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"incomplete/incompatible Alpamayo base: {loading}")
    if any(value.is_meta for value in [*net.parameters(), *net.buffers()]):
        raise ValueError("Alpamayo base loading left meta state")
    if any(p.dtype != dtype for p in net.parameters() if p.is_floating_point()):
        raise ValueError("Alpamayo loader did not honor the declared base precision")
    if snapshot_identity(model_id, weights=True) != base:
        raise ValueError("Alpamayo base changed during materialization")
    identity = {
        "base": base,
        "processor": snapshot_identity(config.processor_path, weights=False),
        "model_contract": {
            "family": "alpamayo-r1-trajectory-only-v1",
            "external": external_identity(),
            "forward": "fresh-differentiable-prefix-noncausal-expert",
            "reasoning_supervision": False,
            "min_pixels": config.min_pixels,
            "max_pixels": config.max_pixels,
        },
    }
    return net, identity


def build_policy(
    model_id, config, pdims=None, *, adapters=None, adapter_resume=False, adapter_media_identity=None, device="cuda"
):
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointImpl,
        apply_activation_checkpointing,
        checkpoint_wrapper,
    )
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionBlock

    from cosmos_framework.posttrain.model.parallelize import parallelize
    from cosmos_framework.posttrain.model.specs import ShardPlan
    from cosmos_framework.posttrain.utils.dist import run_rank_converged

    require_runtime()
    config.__post_init__()
    if adapters is None:
        raise ValueError("Alpamayo trajectory SFT currently requires explicit adapter components")
    if pdims is not None and (
        pdims.dp_replicate != 1 or any(getattr(pdims, k, 1) != 1 for k in ("tp", "cp", "pp", "ep"))
    ):
        raise ValueError("Alpamayo adapters require pure DP sharding; TP/CP/PP/EP/replication are unsupported")

    def construct():
        net, identity = load_base(model_id, config)
        return net.to(device), identity

    net, identity = run_rank_converged(construct, "Alpamayo base materialization")
    # Prefix K/V leaves decoder blocks through a mutable cache, not their tensor
    # outputs. Keep language and expert in the root FSDP unit so backward hooks
    # observe the action loss; wrapping those decoder blocks would hide their
    # cache-only gradient paths from FSDP's output traversal. Vision returns
    # ordinary tensors and can use independent units and checkpointing.
    blocks = (Qwen3VLVisionBlock,)
    run_rank_converged(lambda: validate_targets(net, adapters), "Alpamayo adapter gradient targets")
    run_rank_converged(
        lambda: prepare_adapter_model(
            net,
            adapters,
            base=model_id,
            processor=config.processor_path,
            base_dtype="bfloat16",
            media=adapter_media_identity or {},
            resume=adapter_resume,
            distributed=pdims is not None and pdims.world_size > 1,
            family="alpamayo_r1",
            fsdp_block_types=blocks,
            source_identity=identity,
        ),
        "Alpamayo adapter installation",
    )
    if config.activation_checkpointing == "vision":
        apply_activation_checkpointing(
            net,
            checkpoint_wrapper_fn=partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT),
            check_fn=lambda m: isinstance(m, Qwen3VLVisionBlock),
        )
    parallelize(net, ShardPlan(blocks, None, torch.float32), pdims)
    assert_trainability(net)
    return AlpamayoSFTPolicy(net)
