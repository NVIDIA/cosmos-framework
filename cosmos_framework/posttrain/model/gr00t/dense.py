# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Native GR00T dense policy reconstruction, including processor/statistics and action head.

This is an inference artifact reconstructed from effective quantized weights plus
adapters. It does not reload the original dense base or imply merge/requantization parity.
"""

from __future__ import annotations

import copy
from pathlib import Path


def sidecars(config):
    """Explicit native preprocessing files, no source weights and no arbitrary directory copy."""
    from cosmos_framework.posttrain.model.gr00t.qlora_source import verify_metadata
    from cosmos_framework.posttrain.model.gr00t.source import metadata_files, resolve_source
    from cosmos_framework.utils.lazy_config import instantiate

    model = instantiate(config.model)
    checkpoint, _ = resolve_source(model.checkpoint_path, model.revision)
    backbone = Path(model.backbone_adaptation.backbone_path).expanduser().resolve()
    verify_metadata(checkpoint, backbone)
    return {
        **{f"native/{name}": str(checkpoint / name) for name in metadata_files(checkpoint, gr00t=True)},
        **{f"backbone/{name}": str(backbone / name) for name in metadata_files(backbone)},
    }


def build_policy(config, artifact, *, device, output, unmerged=False):
    """Restore complete native policy on the framework-owned device without source-base loading."""
    import torch

    from cosmos_framework.posttrain.checkpoint.adapter import require_identity_subset
    from cosmos_framework.posttrain.checkpoint.family_dense import (
        head_adapter_targets,
        load_family_dense,
        load_family_unmerged,
        read_family_dense,
        read_family_unmerged,
    )
    from cosmos_framework.posttrain.model.gr00t.action_sft import Gr00tActionPacker
    from cosmos_framework.posttrain.model.gr00t.adaptation import (
        construct_meta,
        resolve_language_targets,
        task_identity,
        validate_task_config,
        verify_external_source,
    )
    from cosmos_framework.posttrain.model.gr00t.head_adaptation import configured_head_targets
    from cosmos_framework.posttrain.model.gr00t.policy import OFFICIAL_PRECISION, Gr00tN1d7TrainPolicy, load_processor
    from cosmos_framework.posttrain.model.gr00t.qlora_source import recorded_source, verify_metadata
    from cosmos_framework.posttrain.model.precision import compute_context
    from cosmos_framework.utils.lazy_config import instantiate

    model = copy.deepcopy(instantiate(config.model))
    model.backbone_adaptation.__post_init__()
    if model.backbone_adaptation.action_head_adapter_profile is not None and not unmerged:
        raise ValueError("GR00T head adapters require standalone unmerged artifacts")
    root = Path(artifact).resolve()
    # Verification occurs before any tokenizer/model constructor reads artifact sidecars.
    read_artifact = read_family_unmerged if unmerged else read_family_dense
    load_artifact = load_family_unmerged if unmerged else load_family_dense
    manifest = read_artifact(str(root))
    # Model-defining fields only, as for adapter reload: no library versions, process math
    # flags, or adapter seed/dropout.
    validate_task_config(model)
    verify_external_source()
    verify_metadata(root / "native", root / "backbone")
    expected = {
        "base": recorded_source(),
        "task_contract": task_identity(model, root / "native", root / "backbone"),
        "adapter": {name: getattr(model.backbone_adaptation, name) for name in ("mode", "rank", "alpha")},
    }
    require_identity_subset(manifest["source_metadata"]["identity"], expected)
    task = manifest["source_metadata"]["identity"]["task_contract"]
    model.checkpoint_path = str(root / "native")
    model.revision = ""
    model.backbone_adaptation.backbone_path = str(root / "backbone")
    net = construct_meta(task["native_model_config"], model.backbone_adaptation.backbone_path)
    head_targets = configured_head_targets(net, model.backbone_adaptation)
    identity = manifest["source_metadata"]["identity"]
    if head_targets is None:
        if "head_adapters" in identity:
            raise ValueError("GR00T artifact contains an unrequested head adapter profile")
    elif {name: list(shape) for name, shape in head_targets.items()} != head_adapter_targets(identity):
        raise ValueError("GR00T artifact differs from the explicit head adapter profile")
    if {name: list(shape) for name, shape in resolve_language_targets(net).items()} != manifest["source_metadata"][
        "identity"
    ]["targets"]:
        raise ValueError("dense GR00T architecture differs from the exported language target profile")
    expected["auxiliary_parameters"] = sorted(
        name
        for name, parameter in net.named_parameters()
        if name.startswith("action_head.") and parameter.requires_grad and model.backbone_adaptation.train_action_head
    )
    # Preserve the upstream Conv3d layout; the shared dense loader records and restores canonical values.
    channels_last = [
        name
        for name, parameter in net.named_parameters()
        if parameter.ndim == 5 and parameter.is_contiguous(memory_format=torch.channels_last_3d)
    ]
    load_artifact(net, str(root), device=torch.device(device), expected_identity=expected, manifest=manifest)
    with torch.no_grad():
        for name in channels_last:
            parameter = net.get_parameter(name)
            parameter.data = parameter.contiguous(memory_format=torch.channels_last_3d)
    rotary = net.backbone.model.language_model.rotary_emb
    if hasattr(rotary, "original_inv_freq"):
        rotary.original_inv_freq = rotary.inv_freq
    net.eval()
    policy = object.__new__(Gr00tN1d7TrainPolicy)
    policy.config, policy.precision, policy._ppo_enabled = model, OFFICIAL_PRECISION, False
    policy.gr00t = policy.net = net
    policy.backbone = net.backbone
    policy.compute_ctx = compute_context("autocast", torch.bfloat16, torch.device(device).type)
    policy.processor = load_processor(model.checkpoint_path, model_name=model.backbone_adaptation.backbone_path)
    policy.packer = Gr00tActionPacker(
        model.checkpoint_path, model, training_prefix=model.training_prefix, action_dim=model.action_dim
    )
    if unmerged:
        policy.unmerged_manifest = manifest
    else:
        policy.dense_manifest = manifest
    return policy
