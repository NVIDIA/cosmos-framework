# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Native GR00T dense policy reconstruction, including processor/statistics and action head.

This is an inference artifact reconstructed from effective quantized weights plus
adapters. It does not reload the original dense base or imply merge/requantization parity.
"""

from __future__ import annotations

import copy
import json
from importlib.metadata import version
from pathlib import Path


def sidecars(config):
    """Explicit native preprocessing files, no source weights and no arbitrary directory copy."""
    from cosmos_framework.posttrain.model.gr00t.qlora_source import verify_metadata
    from cosmos_framework.posttrain.model.gr00t.source import _metadata, _resolve
    from cosmos_framework.utils.lazy_config import instantiate

    model = instantiate(config.model)
    checkpoint, _ = _resolve(model.checkpoint_path, model.revision)
    backbone = Path(model.backbone_adaptation.backbone_path).expanduser().resolve()
    verify_metadata(checkpoint, backbone)
    return {
        **{f"native/{name}": str(checkpoint / name) for name in _metadata(checkpoint, gr00t=True)},
        **{f"backbone/{name}": str(backbone / name) for name in _metadata(backbone)},
    }


def _task_identity(config, root):
    """Validate source/task identities using only exported metadata; source weights are unnecessary."""
    from cosmos_framework.posttrain.model.gr00t.adaptation import _external_identity, validate_task_config
    from cosmos_framework.posttrain.model.gr00t.qlora_source import verify_metadata
    from cosmos_framework.posttrain.model.gr00t.source import _metadata
    from cosmos_framework.posttrain.utils.object_store import file_sha256

    validate_task_config(config)
    identity = config.identity()
    identity.pop("checkpoint_path")
    identity.pop("revision")
    identity["backbone_adaptation"].pop("backbone_path")
    native = root / "native"
    verify_metadata(native, root / "backbone")
    result = {
        "family": "gr00t_n1d7",
        "external": _external_identity(),
        "config": identity,
        "implementation": {
            name: file_sha256(str(Path(__file__).parent / name))
            for name in (
                "adaptation.py",
                "head_adaptation.py",
                "action_sft.py",
                "policy.py",
                "packer.py",
                "qlora_source.py",
            )
        },
        "native_model_config": json.loads((native / "config.json").read_text()),
        "processor_config": json.loads((native / "processor_config.json").read_text()),
        "statistics": json.loads((native / "statistics.json").read_text()),
        "backbone_metadata": _metadata(root / "backbone"),
        "precision": {
            "backbone_parameters": "bfloat16",
            "adapter_parameters": "float32",
            "action_head_parameters": "float32",
            "compute": "bfloat16-autocast-including-adapter-gemm",
            "task_state_action_noise": "float32",
            "matmul_allow_tf32": True,
            "cudnn_allow_tf32": True,
        },
    }
    embodiment = native / "embodiment_id.json"
    if embodiment.exists():
        result["embodiment_id"] = json.loads(embodiment.read_text())
    return result


def build_policy(config, artifact, *, device, output, unmerged=False):
    """Restore complete native policy on the framework-owned device without source-base loading."""
    import torch

    from cosmos_framework.posttrain.checkpoint.adapter import _require_subset
    from cosmos_framework.posttrain.checkpoint.family_dense import (
        _head_targets,
        load_family_dense,
        load_family_unmerged,
        read_family_dense,
        read_family_unmerged,
    )
    from cosmos_framework.posttrain.model.gr00t.action_sft import Gr00tActionPacker
    from cosmos_framework.posttrain.model.gr00t.adaptation import _construct_meta, resolve_language_targets
    from cosmos_framework.posttrain.model.gr00t.head_adaptation import configured_head_targets
    from cosmos_framework.posttrain.model.gr00t.policy import OFFICIAL_PRECISION, Gr00tN1d7TrainPolicy, load_processor
    from cosmos_framework.posttrain.model.gr00t.qlora_source import recorded_source
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
    expected = {
        "base": recorded_source(),
        "task_contract": _task_identity(model, root),
        "torch_version": torch.__version__,
        "dependencies": {name: version(name) for name in ("transformers", "safetensors")},
    }
    if unmerged:
        expected["adapter"] = {
            name: getattr(model.backbone_adaptation, name) for name in ("mode", "rank", "alpha", "dropout", "seed")
        }
    compatibility = None
    if unmerged:
        from cosmos_framework.posttrain.checkpoint.family_inference import (
            INFERENCE_COMPATIBILITY,
            gr00t_unmerged_inference_identity_for_read,
        )
        from cosmos_framework.posttrain.model.quantization.family import _implementation_identity

        adjusted = gr00t_unmerged_inference_identity_for_read(
            manifest["source_metadata"]["identity"], expected, implementation=_implementation_identity()
        )
        if adjusted is not expected:
            compatibility = INFERENCE_COMPATIBILITY
        expected = adjusted
    _require_subset(manifest["source_metadata"]["identity"], expected)
    task = manifest["source_metadata"]["identity"]["task_contract"]
    precision = task["precision"]
    torch.backends.cuda.matmul.allow_tf32 = precision["matmul_allow_tf32"]
    torch.backends.cudnn.allow_tf32 = precision["cudnn_allow_tf32"]
    model.checkpoint_path = str(root / "native")
    model.revision = ""
    model.backbone_adaptation.backbone_path = str(root / "backbone")
    net = _construct_meta(task["native_model_config"], model.backbone_adaptation.backbone_path)
    head_targets = configured_head_targets(net, model.backbone_adaptation)
    identity = manifest["source_metadata"]["identity"]
    if head_targets is None:
        if "head_adapters" in identity:
            raise ValueError("GR00T artifact contains an unrequested head adapter profile")
    elif {name: list(shape) for name, shape in head_targets.items()} != _head_targets(identity):
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
        policy.inference_compatibility = compatibility
    else:
        policy.dense_manifest = manifest
    return policy
