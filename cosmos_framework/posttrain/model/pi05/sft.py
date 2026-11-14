# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Full-model π0.5 supervised adapters, separate from the cached-prefix PPO seat."""

from __future__ import annotations

from dataclasses import dataclass

from cosmos_framework.posttrain.model.sft_source import (
    register_flow_sft_gradient_window,
    register_flow_sft_packer,
    register_native_sft_source,
)

BUILDER = "cosmos_framework.posttrain.model.pi05.sft.build_pi05_sft_policy"


def require_sft_runtime():
    import transformers

    if transformers.__version__ != "4.57.6":
        raise ValueError(
            "pi05 full-model SFT/merged reload requires Transformers 4.57.6; Transformers 5 changes the SigLIP checkpoint namespace"
        )


@dataclass
class Pi05SFTModelConfig:
    tokenizer_path: str
    norm_stats_path: str
    activation_checkpointing: str = "full"

    def __post_init__(self):
        if self.activation_checkpointing not in ("none", "full"):
            raise ValueError("pi05 SFT activation_checkpointing must be none or full")


def _model_config(model_id, config):
    from cosmos_framework.posttrain.model.pi05.config import Pi05ModelConfig

    return Pi05ModelConfig(model_id, config.tokenizer_path, config.norm_stats_path, action_chunk=10)


@register_native_sft_source(BUILDER)
def resolve_source(spec):
    from cosmos_framework.posttrain.model.pi05.source import Pi05Source
    from cosmos_framework.utils.lazy_config import instantiate

    source = Pi05Source.resolve(_model_config(spec.model_id, instantiate(spec.model_config)))
    return str(source.checkpoint.parent), False


class Pi05SFTPacker:
    def __init__(self, config, *, training_prefix, action_dim):
        from cosmos_framework.posttrain.model.pi05.processing import Pi05Processor
        from cosmos_framework.posttrain.model.pi05.source import Pi05Source

        if not 1 <= training_prefix <= 10 or action_dim != 7:
            raise ValueError("pi05 LIBERO flow SFT requires a prefix in [1,10] and action_dim=7")
        self.processor = Pi05Processor(config)
        self.source = Pi05Source.resolve(config)
        self.training_prefix, self.action_dim = training_prefix, action_dim

    def validation_semantics_identity(self):
        from cosmos_framework.posttrain.utils.object_store import file_sha256

        return {
            "processor": "pi05-libero-offline-v1",
            "tokenizer_sha256": file_sha256(str(self.source.tokenizer)),
            "normalization_sha256": file_sha256(str(self.source.normalization)),
            "images": "two-RGB-PIL-bilinear-pad-224-third-view-masked",
            "tokens": 200,
            "action_normalization": "quantile-q01-q99-epsilon-1e-6",
            "action_convention": "LIBERO-demonstration-unchanged",
            "training_prefix": self.training_prefix,
            "action_dim": self.action_dim,
            "augmentation": "none",
            "time": "Beta(1.5,1.0)*0.999+0.001",
        }

    def __call__(self, trajectories):
        import numpy as np
        import torch

        from cosmos_framework.posttrain.data.flow_media import open_flow_image
        from cosmos_framework.posttrain.model.gr00t.packer import action_mask_for

        observations, actions, weights = [], [], []
        for trajectory in trajectories:
            for step in trajectory.transitions:
                sample = step.obs.representations["flow_sample"]
                observation = dict(sample["observation"])
                for key in ("image", "wrist_image"):
                    with open_flow_image(sample["media"][key]) as image:
                        observation[key] = np.asarray(image.convert("RGB")).copy()
                observations.append(observation)
                actions.append(sample["actions"])
                weights.append(step.algo_extra.get("validation_weight", 1.0))
        actions = self.processor.encode_actions(torch.tensor(actions, dtype=torch.float32))
        mask = action_mask_for(actions, action_dim=self.action_dim, training_prefix=self.training_prefix)
        mask *= torch.tensor(weights)[:, None, None]
        return {"inputs": self.processor.encode(observations), "actions": actions, "action_mask": mask}


@register_flow_sft_packer(BUILDER)
def build_packer(spec, rl):
    from cosmos_framework.utils.lazy_config import instantiate

    return Pi05SFTPacker(
        _model_config(spec.model_id, instantiate(spec.model_config)),
        training_prefix=rl.flow_training_prefix,
        action_dim=rl.flow_action_dim,
    )


def source_identity(model, source):
    """The source has no HF config.json; its existing native contract is authoritative."""
    from cosmos_framework.posttrain.model.pi05.source import model_identity
    from cosmos_framework.posttrain.utils.identity import canonical_json_sha256
    from cosmos_framework.posttrain.utils.object_store import file_sha256

    def inventory(paths):
        files = {name: file_sha256(str(path)) for name, path in paths.items()}
        return {"files": files, "sha256": canonical_json_sha256(files, context="pi05 adapter source")}

    contract = model_identity(model.config, source.identity())
    contract["training_replay"] = "offline-fresh-differentiable-prefix-v1"
    contract["parameter_dtypes"] = {
        name: str(p.dtype).removeprefix("torch.") for name, p in model.named_parameters(remove_duplicate=False)
    }
    return {
        "base": inventory({"model.safetensors": source.checkpoint}),
        "processor": inventory({"tokenizer.model": source.tokenizer, "norm_stats.json": source.normalization}),
        "model_contract": contract,
    }


def validate_prefix_targets(model, adapters):
    """The action expert reads per-layer K/V, not the final prefix output.

    Query/output/MLP projections in the last prefix layer affect only that unused
    final output. Selecting them would create apparently trainable dead branches.
    Earlier blocks feed later K/V and remain valid targets.
    """
    from cosmos_framework.posttrain.model.adapters import resolve_targets

    resolved = resolve_targets(model, adapters, family="pi05")
    last = len(model.paligemma_with_expert.paligemma.model.language_model.layers) - 1
    prefix = f"paligemma_with_expert.paligemma.model.language_model.layers.{last}."
    for row in resolved["targets"]:
        if row["name"].startswith(prefix) and row["name"][len(prefix) :] not in (
            "self_attn.k_proj",
            "self_attn.v_proj",
        ):
            raise ValueError(
                f"pi05 action loss does not consume final-prefix output target {row['name']}; select its k_proj/v_proj or an earlier layer"
            )


class Pi05SFTPolicy:
    def __init__(self, model, processor):
        from cosmos_framework.posttrain.model.pi05.policy import NATIVE_PRECISION

        self.net, self.processor, self.precision = model, processor, NATIVE_PRECISION

    def flow_matching_loss(self, inputs):
        device = next(self.net.parameters()).device
        return self.net(
            self.processor.materialize(inputs["inputs"], device=device),
            inputs["actions"].to(device),
            inputs["action_mask"].to(device),
            **{k: inputs[k].to(device) for k in ("time", "noise") if k in inputs},
        )


def build_pi05_sft_policy(
    model_id,
    config,
    pdims=None,
    *,
    adapters=None,
    adapter_resume=False,
    adapter_media_identity=None,
    device="cuda",
    gradient_policy=None,
):
    import torch
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointImpl,
        apply_activation_checkpointing,
        checkpoint_wrapper,
    )
    from transformers.models.gemma.modeling_gemma import GemmaMLP
    from transformers.models.siglip.modeling_siglip import SiglipEncoderLayer

    from cosmos_framework.posttrain.model.adapters import (
        assert_trainability,
        prepare_adapter_model,
        validate_adapter_parallelism,
    )
    from cosmos_framework.posttrain.model.parallelize import parallelize
    from cosmos_framework.posttrain.model.pi05.compat import Pi05GemmaAttention, Pi05GemmaDecoderLayer
    from cosmos_framework.posttrain.model.pi05.modeling import Pi05Model
    from cosmos_framework.posttrain.model.pi05.processing import Pi05Processor
    from cosmos_framework.posttrain.model.pi05.source import Pi05Source, load_model_weights
    from cosmos_framework.posttrain.model.specs import ShardPlan
    from cosmos_framework.posttrain.utils.dist import run_rank_converged

    config.__post_init__()
    if gradient_policy is not None:
        from cosmos_framework.posttrain.roles.trainer.gradient_window import PROFILE

        if (
            gradient_policy != PROFILE
            or pdims is None
            or pdims.world_size != 4
            or (pdims.dp_shard, pdims.dp_replicate) not in ((4, 1), (2, 2))
        ):
            raise ValueError("Pi0.5 owned FP64 gradients require fp64_owned_shard_window_v1 on FSDP4 or HSDP2x2")
    require_sft_runtime()
    if adapters is None:
        raise ValueError("pi05 offline SFT currently requires explicit adapters; use the existing PPO seat for PPO")
    validate_adapter_parallelism(pdims)
    native = _model_config(model_id, config)
    source = run_rank_converged(lambda: Pi05Source.resolve(native), "pi05 SFT source")

    def construct():
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(42)
            net = Pi05Model(native, device="cpu")
            load_model_weights(net, source)
            net.apply_reference_precision()
        identity = source_identity(net, source)
        net.to(device=device)
        return net, identity

    net, identity = run_rank_converged(construct, "pi05 SFT base materialization")
    blocks = (Pi05GemmaAttention, GemmaMLP, SiglipEncoderLayer)
    run_rank_converged(lambda: validate_prefix_targets(net, adapters), "pi05 SFT prefix gradient targets")
    run_rank_converged(
        lambda: prepare_adapter_model(
            net,
            adapters,
            base=model_id,
            processor=model_id,
            base_dtype="model_defined",
            media=adapter_media_identity or {},
            resume=adapter_resume,
            distributed=pdims is not None and pdims.world_size > 1,
            family="pi05",
            fsdp_block_types=blocks,
            source_identity=identity,
        ),
        "pi05 SFT adapter installation",
    )
    if config.activation_checkpointing == "full":
        from functools import partial

        apply_activation_checkpointing(
            net,
            checkpoint_wrapper_fn=partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT),
            check_fn=lambda module: isinstance(module, (Pi05GemmaDecoderLayer, SiglipEncoderLayer)),
        )
    parallelize(net, ShardPlan(blocks, None, torch.float64 if gradient_policy is not None else torch.float32), pdims)
    assert_trainability(net)
    policy = Pi05SFTPolicy(net, Pi05Processor(native))
    if gradient_policy is not None:
        from dataclasses import replace

        policy.precision = replace(policy.precision, reduce="float64")
    return policy


@register_flow_sft_gradient_window(BUILDER)
def build_gradient_window(policy, optimizer, rl):
    """Install after adapter/shard registration and final optimizer ownership."""
    from cosmos_framework.posttrain.model.adapters import assert_trainability
    from cosmos_framework.posttrain.model.pi05.modeling import Pi05Model
    from cosmos_framework.posttrain.roles.trainer.gradient_window import PROFILE, install_fsdp
    from cosmos_framework.posttrain.utils.dist import run_rank_converged

    def admit():
        if (
            rl.gradient_policy != PROFILE
            or not isinstance(policy, Pi05SFTPolicy)
            or not isinstance(policy.net, Pi05Model)
        ):
            raise ValueError("owned FP64 gradient policy requires the native Pi0.5 SFT model")
        if rl.task != "flow_bc" or rl.adapters is None:
            raise ValueError("owned FP64 gradient policy requires flow adapter SFT")
        if (
            policy.precision.reduce != "float64"
            or policy.precision.mode != "fsdp"
            or policy.precision.param is not None
        ):
            raise ValueError("Pi0.5 gradient policy requires native compute and registered FP64 reduction")
        assert_trainability(policy.net, optimizer)

    run_rank_converged(admit, "Pi0.5 owned gradient-policy model admission")
    return install_fsdp(policy.net, optimizer, microbatches=rl.grad_accum)
