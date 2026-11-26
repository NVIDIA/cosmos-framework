# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Helpers shared by the native action SFT, evaluation, export and GR00T rollout entrypoints."""

from __future__ import annotations

import importlib
import json
import math
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

FROZEN_BACKBONE_MODES = ("bf16_frozen", "nf4_frozen")


def load_factory(path: str):
    """Resolve an explicit ``module.attribute`` factory path."""
    if not isinstance(path, str) or "." not in path:
        raise ValueError("action SFT factories must be explicit import paths")
    module, name = path.rsplit(".", 1)
    result = getattr(importlib.import_module(module), name)
    if not callable(result):
        raise TypeError(f"action SFT factory is not callable: {path}")
    return result


def factory_identity(config) -> dict[str, str]:
    """The configured factory import paths; recorded in training and evaluation identities."""
    return {
        name: getattr(config, name)
        for name in ("policy_factory", "data_factory", "objective_factory", "evaluation_callback")
        if getattr(config, name)
    }


def configure_tf32(config) -> None:
    """Set process-wide TF32 once, before any model is built; the policy identity records the result.

    GR00T follows its upstream finetune recipe (``TrainingArguments(tf32=True)``); other families keep
    PyTorch's defaults.
    """
    import torch

    from cosmos_framework.posttrain.config.model import Gr00tN1d7ModelConfig

    if isinstance(config.model, Gr00tN1d7ModelConfig):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def write_json(path: Path, value: Any) -> None:
    from cosmos_framework.posttrain.checkpoint.io import atomic_write_bytes

    atomic_write_bytes(str(path), (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())


class ActionSource:
    """Give the shared IndexedCursor a content-bound family dataset."""

    def __init__(self, dataset):
        from cosmos_framework.posttrain.utils.identity import canonical_json_sha256

        self.dataset = dataset
        self.identity = json.loads(json.dumps(dataset.identity, allow_nan=False))
        if not self.identity or not len(dataset):
            raise ValueError("action SFT requires a nonempty, content-bound map-style dataset")
        self.fingerprint = canonical_json_sha256(self.identity, context="action SFT source")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    def cursor_fingerprint(self):
        return self.fingerprint


def validate_config(config):
    from cosmos_framework.utils.lazy_config import instantiate

    rl = instantiate(config.rl)
    rl.__post_init__()
    if config.grid.backend != "local" or config.grid.grouping != "single":
        raise ValueError("action SFT requires the native local single-process torchrun path")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("action SFT does not support distributed execution; launch one torchrun process")
    if rl.checkpoint_store.backend != "local" or rl.upload.enabled:
        raise ValueError("action SFT requires a local checkpoint store without upload")
    for name in ("policy_factory", "data_factory", "objective_factory"):
        if not isinstance(getattr(config, name), str) or "." not in getattr(config, name):
            raise ValueError(f"action SFT requires {name}")
    from cosmos_framework.posttrain.utils.action_evaluation_inputs import validate_evaluation_input_reuse
    from cosmos_framework.posttrain.utils.gr00t_evaluation_backbone import validate_reuse

    validate_evaluation_input_reuse(config, rl)
    validate_reuse(config, rl)
    return rl


def validate_policy_source(config, *, checkpoint, base, dense=False, unmerged=False) -> None:
    """Require exactly one of a trained checkpoint or the untrained base; adapters need a checkpoint."""
    if bool(checkpoint) == bool(base) or ((dense or unmerged) and base) or (dense and unmerged):
        raise ValueError("choose exactly one of --checkpoint or --base; --dense/--unmerged require --checkpoint")
    if not base:
        return
    adaptation = getattr(config.model, "backbone_adaptation", None)
    if getattr(adaptation, "mode", None) not in FROZEN_BACKBONE_MODES:
        raise ValueError("--base requires backbone_adaptation.mode=bf16_frozen or nf4_frozen; LoRA needs a checkpoint")


@contextmanager
def single_process_group():
    """Initialize the one-process NCCL group if needed; destroy it on exit if this call created it."""
    import torch.distributed as dist

    owned = not dist.is_initialized()
    if owned:
        dist.init_process_group("nccl")
    try:
        yield
    finally:
        if owned:
            dist.destroy_process_group()


@contextmanager
def preserved_evaluation(policy, seed: int):
    """Restore mixed module modes and all training RNG streams, including on failure."""
    import random

    import numpy as np
    import torch

    from cosmos_framework.posttrain.checkpoint.rng import collect_rng_state, restore_rng_state

    modes = [(module, module.training) for module in policy.net.modules()]
    rng = collect_rng_state()
    try:
        policy.net.eval()
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        with torch.no_grad():
            yield
    finally:
        try:
            for module, mode in modes:
                module.training = mode
        finally:
            restore_rng_state(rng)


def evaluate(policy, objective, source, config, callback=None):
    """Full finite evaluation; loss is weighted by the objective's native denominator."""
    from cosmos_framework.posttrain.algorithm.primitives.objective import OBJECTIVE_WEIGHT_SUM
    from cosmos_framework.posttrain.utils.action_evaluation_inputs import evaluation_batch_scope

    limit = config.rl.evaluation_max_samples or len(source)
    count = min(limit, len(source))
    totals: dict[str, list[float]] = {}
    rows = []
    with preserved_evaluation(policy, config.rl.evaluation_seed):
        for start in range(0, count, config.rl.micro_batch_size):
            batch = [source[index] for index in range(start, min(start + config.rl.micro_batch_size, count))]
            with evaluation_batch_scope(policy, batch):
                loss, metrics = objective(policy, batch)
                weight = float(metrics.get(OBJECTIVE_WEIGHT_SUM, len(batch)))
                values = {"loss": {"sum": float(loss.detach()) * weight, "count": weight}}
                if callback is not None:
                    extra = callback(policy, batch, config)
                    if "loss" in extra:
                        raise ValueError("task evaluation callback must not replace the native loss")
                    values.update(extra)
            for key, item in values.items():
                if set(item) != {"sum", "count"}:
                    raise ValueError(f"evaluation metric {key} requires explicit sum and count")
                numerator, denominator = float(item["sum"]), float(item["count"])
                if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator <= 0:
                    raise ValueError(f"nonfinite or empty action evaluation metric {key}")
                total = totals.setdefault(key, [0.0, 0.0])
                total[0] += numerator
                total[1] += denominator
            rows.append({"ids": [item.id for item in batch], "metrics": values})
    return {key: numerator / denominator for key, (numerator, denominator) in totals.items()}, rows
