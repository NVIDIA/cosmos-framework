# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Differentiable public Alpamayo prefix/action forward, preserving model names."""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path

import torch

from cosmos_framework.posttrain.config.build import Precision
from cosmos_framework.posttrain.model.alpamayo.sft import (
    INFERENCE_STEPS,
    PHYSICALAI_PROFILE,
    REFERENCE_REVISION,
    source_config,
)
from cosmos_framework.posttrain.model.protocol import FlowMatchingLossOutput


@dataclass(frozen=True)
class AlpamayoFlowOutput(FlowMatchingLossOutput):
    prediction: torch.Tensor


@functools.cache
def sft_net_class() -> type:
    """Build the trainable subclass on first use; the optional ``alpamayo_r1`` package is imported here."""
    try:
        from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
    except ImportError as error:
        raise ImportError(
            "Native Alpamayo trajectory training requires the optional alpamayo_r1 package "
            f"from NVlabs/alpamayo@{REFERENCE_REVISION}; see docker/posttrain/native_action/README.md"
        ) from error

    class AlpamayoSFTNet(AlpamayoR1):
        def __init__(self, config, *args, **kwargs):
            super().__init__(config, *args, **kwargs)
            if config.model_dtype == "float32":
                # The public processor has text_config.dtype=BF16. Transformers
                # constructs that child in BF16 even with an outer FP32 request,
                # then casts incoming checkpoint tensors to the child's dtype.
                # Establish FP32 on the empty model BEFORE from_pretrained loads
                # merged weights; casting after loading would irreversibly round
                # the merged adapter contribution. Native BF16 training is unchanged.
                self.float()

        def forward(self, tokenized, history, actions, action_mask, *, time=None, noise=None):
            if tokenized.get("past_key_values") is not None:
                raise ValueError("Alpamayo SFT requires a fresh differentiable prefix, not an externally cached prefix")
            if getattr(self, "_full_activation_checkpointing", False) and self.training and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint

                # Replay the whole prefix -> expert -> flow graph. Every invocation
                # constructs a NEW DynamicCache internally; none crosses the replay
                # boundary. Generic decoder checkpointing would instead replay into
                # the already-extended mutable K/V cache and lose/corrupt gradients.
                return checkpoint(
                    self._flow_forward,
                    tokenized,
                    history,
                    actions,
                    action_mask,
                    time=time,
                    noise=noise,
                    use_reentrant=False,
                    preserve_rng_state=True,
                )
            return self._flow_forward(tokenized, history, actions, action_mask, time=time, noise=noise)

        def _flow_forward(self, tokenized, history, actions, action_mask, *, time=None, noise=None):
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
            if not torch.isfinite(action_mask).all() or (action_mask < 0).any():
                raise ValueError("Alpamayo trajectory supervision weights must be finite and nonnegative")
            weights = action_mask[:, 0, 0]
            if not torch.equal(action_mask, weights[:, None, None].expand_as(action_mask)):
                raise ValueError("Alpamayo SFT mask must supervise whole 64x2 trajectories or validation padding")
            losses = torch.stack(
                [
                    self.diffusion.compute_loss_from_pred(
                        {k: v[i : i + 1] for k, v in training.items() if isinstance(v, torch.Tensor)},
                        prediction[i : i + 1],
                    )
                    for i in range(actions.shape[0])
                ]
            )
            loss = (losses * weights).sum() / weights.sum().clamp_min(1)
            return AlpamayoFlowOutput(loss, action_mask, prediction)

    AlpamayoSFTNet.__qualname__ = "AlpamayoSFTNet"
    return AlpamayoSFTNet


def __getattr__(name: str):
    if name == "AlpamayoSFTNet":
        return sft_net_class()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda" or next(self.net.parameters()).dtype == torch.bfloat16,
        ):
            return self.net(**move(inputs))


@functools.cache
def verify_external_source() -> None:
    """Runtime preflight, once per process: alpamayo_r1 is the pinned NVlabs/alpamayo source checkout."""
    # The public distribution is a namespace package without __init__.py.
    import alpamayo_r1.models.alpamayo_r1 as native_module

    from cosmos_framework.posttrain.utils.native_source import require_source_revision

    require_source_revision(native_module.__file__, REFERENCE_REVISION, "NVlabs/alpamayo")


def model_contract(config) -> dict:
    """Identity of the native prefix/expert forward and its observation preparation."""
    return {
        "family": "alpamayo-r1-trajectory-only-v1",
        "external": {"reference_revision": REFERENCE_REVISION},
        "forward": "fresh-differentiable-prefix-noncausal-expert",
        "reasoning_supervision": False,
        "observation_profile": PHYSICALAI_PROFILE,
        "min_pixels": config.min_pixels,
        "max_pixels": config.max_pixels,
    }


def task_contract(config) -> dict:
    """Identity of the supervised trajectory task and its open-loop sampler."""
    return {
        "family": "alpamayo-r1-trajectory-only-v1",
        "loss": "native-flow-matching",
        "train_action_head": config.train_action_head,
        "inference_steps": INFERENCE_STEPS,
        "integrator": "euler",
        "reasoning_supervision": False,
    }


def load_base(model_id, config, *, dtype=torch.bfloat16):
    """Strict native construction shared by training and merged export."""
    from cosmos_framework.posttrain.model.alpamayo.source import snapshot_identity

    if (Path(model_id) / "merged_model.json").exists():
        raise ValueError("Alpamayo merged artifacts require model.alpamayo.adapter_export.load_merged")
    cfg = source_config(model_id, config)
    cfg.model_dtype = str(dtype).removeprefix("torch.")
    cfg.expert_cfg = {**(cfg.expert_cfg or {}), "dtype": cfg.model_dtype}
    base = snapshot_identity(model_id, weights=True)
    # The outer public class does not own an attention implementation flag;
    # its model-owned config selects SDPA on the actual VLM/expert children.
    net, loading = sft_net_class().from_pretrained(
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
        "model_contract": model_contract(config),
    }
    return net, identity


def build_policy(config, *, device, output=None):
    """Native action-SFT factory: CPU base -> NF4/BF16 -> adapters -> optimizer."""
    from cosmos_framework.posttrain.model.alpamayo.runtime import bind_rotary_aliases
    from cosmos_framework.posttrain.model.alpamayo.sft import AlpamayoSFTPacker
    from cosmos_framework.posttrain.model.alpamayo.source import auxiliary_parameters, backbone_targets
    from cosmos_framework.posttrain.model.quantization.family import install_family_backbone
    from cosmos_framework.utils.lazy_config import instantiate

    cfg = instantiate(config.model)
    cfg.__post_init__()
    verify_external_source()  # Fail before expensive loading when the optional runtime drifted.
    net, identity = load_base(cfg.checkpoint_path, cfg)
    # Language/vision remain frozen BF16. Expert + action projections are the
    # explicit auxiliary FP32 masters; CUDA autocast controls their compute.
    state = install_family_backbone(
        net,
        config=cfg.backbone_adaptation,
        targets=backbone_targets(net),
        auxiliary_parameters=auxiliary_parameters(net, train_action_head=cfg.train_action_head),
        source_identity=identity,
        task_identity=task_contract(cfg),
        device=torch.device(device),
    )
    bind_rotary_aliases(net)
    if cfg.activation_checkpointing == "vision":
        from functools import partial

        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            CheckpointImpl,
            apply_activation_checkpointing,
            checkpoint_wrapper,
        )
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionBlock

        apply_activation_checkpointing(
            net,
            checkpoint_wrapper_fn=partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT),
            check_fn=lambda module: isinstance(module, Qwen3VLVisionBlock),
        )
    net._full_activation_checkpointing = cfg.activation_checkpointing == "full"
    policy = AlpamayoSFTPolicy(net)
    policy.family_state = state
    policy.packer = AlpamayoSFTPacker(cfg.checkpoint_path, cfg, training_prefix=64, action_dim=2)
    policy.config = cfg
    state.codec.bind_processor_identity(policy.packer.validation_semantics_identity())
    return policy


def sample_trajectories(policy, inputs):
    """Native empty-reasoning prefix + ten-step Euler action rollout; open loop only."""
    net = policy.net
    device = next(net.parameters()).device

    def move(value):
        if isinstance(value, dict):
            return {key: move(item) for key, item in value.items()}
        return value.to(device) if isinstance(value, torch.Tensor) else value

    values = move(inputs)
    tokenized, history = values["tokenized"], values["history"]
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        ids = net.fuse_traj_tokens(tokenized["input_ids"], history)
        prefix = net.vlm.model(**{**tokenized, "input_ids": ids}, use_cache=True, return_dict=True)
        cache = prefix.past_key_values
        length, horizon = cache.get_seq_length(), 64
        positions = torch.arange(horizon, device=device)[None, None].expand(3, ids.shape[0], -1)
        positions = positions + (prefix.rope_deltas + length)[None]
        attention = torch.zeros((ids.shape[0], 1, horizon, length + horizon), device=device, dtype=torch.float32)
        if "attention_mask" in tokenized:
            attention[..., :length].masked_fill_(
                ~tokenized["attention_mask"].bool()[:, None, None], torch.finfo(torch.float32).min
            )

        def step_fn(x, t):
            embedding = net.action_in_proj(x, t)
            out = net.expert(
                inputs_embeds=embedding,
                position_ids=positions,
                past_key_values=cache,
                attention_mask=attention,
                use_cache=True,
                is_causal=False,
            )
            prediction = net.action_out_proj(out.last_hidden_state[:, -horizon:]).float()
            cache.crop(length)
            return prediction

        actions = net.diffusion.sample(
            batch_size=ids.shape[0],
            step_fn=step_fn,
            device=device,
            inference_step=INFERENCE_STEPS,
            int_method="euler",
        )
    # Native action denormalization/integration runs in FP32, with checkpoint statistics.
    return net.action_space.action_to_traj(
        actions.float(), history["ego_history_xyz"][:, 0], history["ego_history_rot"][:, 0]
    )[0]
