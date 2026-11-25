# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""SFT adapter glue for ``entrypoints/sft.py``: the configuration preflight and the adapter build path.

The adapter mathematics, installation and artifacts live in ``cosmos_framework/utils/generator/adapter*.py``.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from cosmos_framework.posttrain.utils.log import info0

if TYPE_CHECKING:
    from cosmos_framework.posttrain.config.sft import SFTConfig


def validate_sft_adapters(config: Any, rl: SFTConfig, policy_source: tuple[str, bool]) -> None:
    """Refuse an unsupported adapter configuration before any model is built.

    Checks the backend (HF, or a native builder that accepts ``adapters``), the snapshot's architecture,
    the dtypes, the optimizer settings (adapters own membership, learning rates and weight decay), and a
    configured ``rl.adapter_warm_start`` artifact. ``policy_source`` is ``(name_or_path, trust_remote_code)``
    of the model snapshot, as ``entrypoints/sft.py::sft_policy_source`` resolves it.
    """
    import importlib
    import inspect

    from transformers import PretrainedConfig

    from cosmos_framework.posttrain.config.build import resolve_precision, select_policy_backend
    from cosmos_framework.posttrain.config.model import HF
    from cosmos_framework.utils.generator.adapter_artifacts import read_adapter_manifest
    from cosmos_framework.utils.generator.adapter_training import validate_adapter_dtypes

    selection = select_policy_backend(config)
    if selection.backend == HF:
        precision = resolve_precision(config)
        dtypes = {"storage_dtype": precision.master, "compute_dtype": precision.param, "reduce_dtype": precision.reduce}
    elif selection.native_spec is not None:
        builder = selection.native_spec.builder
        module, _, name = builder.rpartition(".")
        parameters = inspect.signature(getattr(importlib.import_module(module), name)).parameters.values()
        if not any(p.name == "adapters" or p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters):
            raise ValueError(f"SFT adapters: the native builder {builder} does not accept adapters")
        model_config = selection.native_spec.model_config
        missing = [field for field in ("dtype", "fsdp_reduce_dtype") if getattr(model_config, field, None) is None]
        if missing:
            raise ValueError(f"SFT adapters: the native model_config of {builder} has no {', '.join(missing)}")
        dtypes = {
            "storage_dtype": model_config.dtype,
            "compute_dtype": None,
            "reduce_dtype": model_config.fsdp_reduce_dtype,
        }
    else:
        raise ValueError(f"SFT adapters require model_backend={HF!r} or a native policy builder that accepts adapters")
    name_or_path, trust_remote_code = policy_source
    if os.path.isdir(name_or_path) and not os.path.isfile(os.path.join(name_or_path, "config.json")):
        raise ValueError(
            f"SFT adapters: {name_or_path!r} has no config.json; point to a complete Hugging Face snapshot"
        )
    model_type = PretrainedConfig.get_config_dict(name_or_path, trust_remote_code=trust_remote_code)[0].get(
        "model_type"
    )
    validate_adapter_dtypes(model_type, **dtypes)
    optimizer = rl.optimizer
    if optimizer.train_keys or optimizer.decay_groups or optimizer.lr_multipliers:
        raise ValueError(
            "SFT adapters set optimizer membership, lr and weight decay per component; clear optimizer "
            "train_keys, decay_groups and lr_multipliers"
        )
    if optimizer.name != "adamw":
        raise ValueError("SFT adapter runs support only optimizer.name='adamw'")
    if rl.adapter_warm_start:
        identity = read_adapter_manifest(rl.adapter_warm_start)
        if identity["architecture"] != model_type:
            raise ValueError(
                f"rl.adapter_warm_start {rl.adapter_warm_start!r} was trained on {identity['architecture']!r}, "
                f"not {model_type!r}"
            )


def build_sft_adapter_model(
    config: Any, rl: SFTConfig, resume_checkpoint: str | None
) -> tuple[Any, Any, dict[str, Any]]:
    """Build the model with adapters and its optimizer; returns ``(model, optimizer, objective identity additions)``.

    A run resuming its own checkpoint ignores ``rl.adapter_warm_start`` and must keep the checkpoint's
    adapters and resume settings, which are checked before the trainer's own identity comparison.
    """
    from cosmos_framework.posttrain.checkpoint.adapters import require_resumable_adapters
    from cosmos_framework.posttrain.config.build import OptimizerParameterGroup, build_backbone, build_optimizer
    from cosmos_framework.utils.generator.adapter_config import AdapterSetup
    from cosmos_framework.utils.generator.adapter_training import adapter_optimizer_groups
    from cosmos_framework.utils.generator.adapters import (
        assert_trainability,
        installed_adapter_identity,
        installed_adapter_manifest,
        training_identity,
    )

    if resume_checkpoint and rl.adapter_warm_start:
        info0(f"SFT adapter_warm_start {rl.adapter_warm_start!r} ignored: resuming {resume_checkpoint}")
    setup = AdapterSetup(
        rl.adapters,
        from_checkpoint=bool(resume_checkpoint),
        warm_start=None if resume_checkpoint else rl.adapter_warm_start,
    )
    model = build_backbone(config, adapters=setup)
    identity, manifest = installed_adapter_identity(model.net), installed_adapter_manifest(model.net)
    training = training_identity(manifest)
    if resume_checkpoint:
        require_resumable_adapters(resume_checkpoint, identity, training)
    groups = tuple(
        OptimizerParameterGroup(
            name=group.name, parameter_names=group.parameter_names, lr=group.lr, weight_decay=group.weight_decay
        )
        for group in adapter_optimizer_groups(model.net, rl.optimizer.lr, rl.optimizer.weight_decay)
    )
    optimizer = build_optimizer(model.net, rl.optimizer, parameter_groups=groups)
    assert_trainability(model.net, manifest, optimizer)
    # Exact resume also compares both as part of the objective identity.
    return model, optimizer, {"adapters": identity, "adapter_training": training}
