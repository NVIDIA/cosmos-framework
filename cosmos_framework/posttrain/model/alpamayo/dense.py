# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Native effective-base dense Alpamayo artifacts; no original weight reload required."""

from __future__ import annotations

from dataclasses import asdict, replace
from importlib.metadata import version
from pathlib import Path

from cosmos_framework.posttrain.model.alpamayo.source import (
    BASE_CONFIG_SHA256,
    BASE_REVISION,
    BASE_WEIGHTS,
    PROCESSOR_FILES,
    auxiliary_parameters,
    backbone_targets,
    snapshot_identity,
)
from cosmos_framework.posttrain.utils.identity import canonical_json_sha256
from cosmos_framework.posttrain.utils.object_store import file_sha256


def dense_sidecars(config):
    """Copy only required configuration/processor bytes, never the original base weights."""
    from cosmos_framework.utils.lazy_config import instantiate

    cfg = instantiate(config.model)
    cfg.__post_init__()
    processor = snapshot_identity(cfg.processor_path, weights=False)
    base_config = Path(cfg.checkpoint_path) / "config.json"
    if file_sha256(str(base_config)) != BASE_CONFIG_SHA256:
        raise ValueError("Alpamayo dense export requires the exact source architecture config")
    return {
        "config.json": str(base_config),
        **{f"processor/{name}": str(Path(cfg.processor_path) / name) for name in processor["files"]},
    }


def _recorded_base(identity):
    """Validate the pinned source attested by export, without allocating/rereading it."""
    base = identity["base"]["base"]
    files = base.get("files", {})
    if (
        base.get("repository") != "nvidia/Alpamayo-R1-10B"
        or base.get("revision") != BASE_REVISION
        or files.get("config.json") != BASE_CONFIG_SHA256
        or any(files.get(name) != digest for name, digest in BASE_WEIGHTS.items())
        or {name for name in files if name.endswith(".safetensors")} != set(BASE_WEIGHTS)
        or base.get("sha256") != canonical_json_sha256(files, context="Alpamayo snapshot")
    ):
        raise ValueError("dense Alpamayo artifact does not attest the supported immutable base")
    return base


def build_dense_policy(config, artifact, *, device, output=None, unmerged=False):
    """Native action_eval factory; materialize one dense parameter at a time from meta."""
    import torch
    from accelerate import init_empty_weights

    from cosmos_framework.posttrain.checkpoint.family_dense import (
        load_family_dense,
        load_family_unmerged,
        read_family_dense,
        read_family_unmerged,
    )
    from cosmos_framework.posttrain.model.alpamayo.policy import AlpamayoSFTNet, AlpamayoSFTPolicy, external_identity
    from cosmos_framework.posttrain.model.alpamayo.runtime import bind_rotary_aliases
    from cosmos_framework.posttrain.model.alpamayo.sft import AlpamayoSFTPacker, source_config
    from cosmos_framework.utils.lazy_config import instantiate

    cfg = instantiate(config.model)
    cfg.__post_init__()
    root = Path(artifact)
    read_artifact = read_family_unmerged if unmerged else read_family_dense
    load_artifact = load_family_unmerged if unmerged else load_family_dense
    manifest = read_artifact(root)
    identity = manifest["source_metadata"]["identity"]
    if identity.get("task_contract", {}).get("family") != "alpamayo-r1-trajectory-only-v1":
        raise ValueError("dense artifact is not an Alpamayo native trajectory policy")
    required = {"config.json", *{f"processor/{name}" for name in PROCESSOR_FILES}}
    if not required.issubset(manifest.get("sidecars", {})):
        raise ValueError("dense Alpamayo artifact is missing native configuration/processor sidecars")
    if file_sha256(str(root / "config.json")) != BASE_CONFIG_SHA256:
        raise ValueError("dense Alpamayo native architecture differs from its supported checkpoint")
    local_cfg = replace(cfg, checkpoint_path=str(root), processor_path=str(root / "processor"))
    processor = snapshot_identity(local_cfg.processor_path, weights=False)
    base = _recorded_base(identity)
    model_contract = {
        "family": "alpamayo-r1-trajectory-only-v1",
        "external": external_identity(),
        "forward": "fresh-differentiable-prefix-noncausal-expert",
        "reasoning_supervision": False,
        "observation_profile": cfg.observation_profile,
        "min_pixels": cfg.min_pixels,
        "max_pixels": cfg.max_pixels,
    }
    task_contract = {
        "family": "alpamayo-r1-trajectory-only-v1",
        "loss": "native-flow-matching",
        "train_action_head": cfg.train_action_head,
        "activation_checkpointing": cfg.activation_checkpointing,
        "inference_steps": cfg.inference_steps,
        "integrator": "euler",
        "reasoning_supervision": False,
    }
    native_config = source_config(str(root), local_cfg)
    with init_empty_weights(include_buffers=False):
        net = AlpamayoSFTNet(native_config)
    packer = AlpamayoSFTPacker(str(root), local_cfg, training_prefix=64, action_dim=2)
    expected = {
        "format": "cosmos.family-backbone.v1",
        "base": {"base": base, "processor": processor, "model_contract": model_contract},
        "task_contract": task_contract,
        "adapter": asdict(cfg.backbone_adaptation),
        "targets": {name: list(shape) for name, shape in backbone_targets(net).items()},
        "auxiliary_parameters": sorted(auxiliary_parameters(net, train_action_head=cfg.train_action_head)),
        "processor_contract": packer.validation_semantics_identity(),
        "torch_version": torch.__version__,
        "dependencies": {name: version(name) for name in ("transformers", "safetensors")},
    }
    load_artifact(net, root, device=device, expected_identity=expected, manifest=manifest)
    bind_rotary_aliases(net)
    if any(parameter.requires_grad or parameter.is_meta for parameter in net.parameters()):
        raise RuntimeError("dense Alpamayo inference load left trainable or unmaterialized parameters")
    policy = AlpamayoSFTPolicy(net)
    policy.config, policy.packer = local_cfg, packer
    if unmerged:
        policy.unmerged_manifest = manifest
    else:
        policy.dense_manifest = manifest
    return policy
