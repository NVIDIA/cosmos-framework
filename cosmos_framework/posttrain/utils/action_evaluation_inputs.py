# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Opt-in, one-batch CPU preparation reuse for deterministic Alpamayo evaluation.

Only the second preparation is reused. Model forwards, mutable KV caches, noise,
loss reductions and task metrics remain owned by the original native functions.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import time
from contextlib import contextmanager, nullcontext
from dataclasses import asdict

MAX_CACHED_BYTES = 128 * 1024**2


def validate_evaluation_input_reuse(config, rl, source=None):
    if not rl.reuse_evaluation_inputs:
        return
    from cosmos_framework.posttrain.model.alpamayo.sft import AlpamayoSFTModelConfig

    if rl.micro_batch_size != 1 or not isinstance(config.model, AlpamayoSFTModelConfig):
        raise ValueError("reuse_evaluation_inputs requires an Alpamayo action SFT model config and micro_batch_size=1")
    if source is not None:
        media = getattr(source.dataset, "media_hashes", None)
        if not isinstance(media, dict) or not media:
            raise ValueError("evaluation input reuse requires content-bound local media")
        for path in media:
            _local_path(path)


def _local_path(path):
    if not isinstance(path, str) or not os.path.isabs(path) or "://" in path:
        raise ValueError(
            "evaluation input reuse supports absolute local image files only; remote media is not supported"
        )
    return path


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _file_version(path):
    value = os.stat(_local_path(path))
    if not stat.S_ISREG(value.st_mode):
        raise ValueError("evaluation input reuse requires regular local image files")
    return (path, value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _processor_identity(packer):
    from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessorKwargs

    from cosmos_framework.posttrain.utils.identity import canonical_config_value

    processor = packer.processor
    components = {}
    for name in ("image_processor", "video_processor"):
        component = getattr(processor, name, None)
        components[name] = (
            None
            if component is None
            else {
                "object": id(component),
                "class": f"{type(component).__module__}.{type(component).__qualname__}",
                "settings": component.to_dict(),
            }
        )
    tokenizer = processor.tokenizer
    components["tokenizer"] = {
        "object": id(tokenizer),
        "class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
        "size": len(tokenizer),
        "added_tokens": tokenizer.get_added_vocab(),
        "settings": {
            name: getattr(tokenizer, name, None)
            for name in (
                "padding_side",
                "truncation_side",
                "model_max_length",
                "chat_template",
                "special_tokens_map",
                "bos_token_id",
                "eos_token_id",
                "pad_token_id",
                "add_bos_token",
                "add_eos_token",
            )
        },
    }
    action_state = {}
    for name, tensor in packer.action_space.named_buffers():
        if tensor.device.type != "cpu" or tensor.numel() > 1024:
            raise ValueError("evaluation input reuse requires bounded CPU action-normalization state")
        action_state[name] = {"dtype": str(tensor.dtype), "shape": list(tensor.shape), "value": tensor.tolist()}
    return canonical_config_value(
        {
            "config": asdict(packer.config),
            "processor_object": id(processor),
            "processor_class": f"{type(processor).__module__}.{type(processor).__qualname__}",
            "chat_template": processor.chat_template,
            # Qwen3VL owns these separately from tokenizer.special_tokens_map.
            # Its __call__ uses them for image expansion and multimodal checks.
            "multimodal_tokens": {
                name: getattr(processor, name, None)
                for token in ("image_token", "video_token", "vision_start_token", "vision_end_token")
                for name in (token, token + "_id")
            },
            # Match the pinned processor's actual precedence, including tokenizer
            # init defaults and live attributes. Snapshot only resolved modality
            # kwargs, not vocabulary files or the mutable tokenizer backend.
            "resolved_kwargs": processor._merge_kwargs(
                Qwen3VLProcessorKwargs,
                tokenizer_init_kwargs=tokenizer.init_kwargs,
                return_tensors="pt",
            ),
            "components": components,
            "action_space_object": id(packer.action_space),
            "action_space_config": packer.model_config.action_space_cfg,
            "action_space_settings": {
                key: value for key, value in vars(packer.action_space).items() if not key.startswith("_")
            },
            "action_space_buffers": action_state,
        }
    )


def _batch_identity(batch, packer):
    if not isinstance(batch, list) or len(batch) != 1 or len(batch[0].transitions) != 1:
        raise ValueError("evaluation input reuse requires exactly one unchanged physical demonstration")
    step = batch[0].transitions[0]
    sample = step.obs.representations["flow_sample"]
    media = sample["media"]
    if not isinstance(media, dict) or any(not isinstance(paths, list) for paths in media.values()):
        raise ValueError("evaluation input reuse requires ordered local image lists")
    files = tuple(_file_version(path) for key in sorted(media) for path in media[key])
    value = {
        "trajectory_id": batch[0].id,
        "sample": sample,
        "validation_weight": step.algo_extra.get("validation_weight", 1.0),
        "preparation": _processor_identity(packer),
    }
    return _digest(value), files


def _clone_cpu_tree(value, *, clone=True):
    """Strict plain container tree; no cached CUDA/autograd/custom-object state."""
    import torch

    if type(value) is torch.Tensor:
        if value.device.type != "cpu" or value.requires_grad or value.layout != torch.strided:
            raise ValueError("evaluation input reuse requires detached dense CPU tensors")
        size = value.numel() * value.element_size()
        if not value.is_contiguous() or value.untyped_storage().nbytes() != size:
            raise ValueError("evaluation input reuse requires complete contiguous CPU tensor storage")
        return value.clone() if clone else value, size
    if type(value) is dict and all(isinstance(key, str) for key in value):
        pairs = {key: _clone_cpu_tree(item, clone=clone) for key, item in value.items()}
        return {key: pair[0] for key, pair in pairs.items()}, sum(pair[1] for pair in pairs.values())
    if type(value) in (tuple, list):
        pairs = [_clone_cpu_tree(item, clone=clone) for item in value]
        return type(value)(pair[0] for pair in pairs), sum(pair[1] for pair in pairs)
    if value is None or type(value) in (str, int, float, bool):
        return value, 0
    raise ValueError("evaluation input reuse refuses unsupported prepared objects")


class _EvaluationPacker:
    def __init__(self, packer):
        self.original = packer
        self._batch = self._prepared = self._identity = None
        self._calls = 0
        self.stats = {
            "format": "alpamayo-evaluation-preparation-observations-v1",
            "scope": "current process only; no checkpoint state",
            "misses": 0,
            "hits": 0,
            "cached_tensor_bytes_peak": 0,
            "identity_seconds": 0.0,
            "clone_seconds": 0.0,
        }

    def __getattr__(self, name):
        return getattr(self.original, name)

    @property
    def pending(self):
        return self._batch is not None or self._prepared is not None

    @contextmanager
    def batch_scope(self, batch):
        if self.pending:
            raise RuntimeError("nested evaluation preparation scopes are unsupported")
        self._batch, self._calls = batch, 0
        try:
            yield
            if self._calls != 2 or self._prepared is not None:
                raise RuntimeError("native evaluation must consume exactly two preparations for each row")
        finally:
            self._batch = self._prepared = self._identity = None
            self._calls = 0

    def _snapshot(self, batch):
        started = time.perf_counter()
        try:
            return _batch_identity(batch, self.original)
        finally:
            self.stats["identity_seconds"] += time.perf_counter() - started

    def __call__(self, batch):
        if self._batch is None:
            return self.original(batch)  # Training and calls outside native evaluation never cache.
        if batch is not self._batch or self._calls >= 2:
            raise RuntimeError("evaluation input reuse requires the original batch and exactly two preparations")
        self._calls += 1
        before = self._snapshot(batch)
        if self._calls == 2:
            if self._prepared is None or before != self._identity:
                raise RuntimeError("evaluation row, mask, processor settings or media changed before reuse")
            result, self._prepared = self._prepared, None  # Consume once; never share mutable outputs across calls.
            self.stats["hits"] += 1
            return result
        prepared = self.original(batch)
        if self._snapshot(batch) != before:
            raise RuntimeError("evaluation row, mask, processor settings or media changed during preparation")
        started = time.perf_counter()
        _, size = _clone_cpu_tree(prepared, clone=False)
        if size > MAX_CACHED_BYTES:
            raise ValueError("evaluation prepared CPU tensors exceed the 128 MiB cache limit")
        returned, _ = _clone_cpu_tree(prepared)
        self.stats["clone_seconds"] += time.perf_counter() - started
        self._prepared, self._identity = prepared, before
        self.stats["misses"] += 1
        self.stats["cached_tensor_bytes_peak"] = max(self.stats["cached_tensor_bytes_peak"], size)
        return returned


def install_evaluation_input_reuse(policy, config, rl):
    validate_evaluation_input_reuse(config, rl)
    if not rl.reuse_evaluation_inputs:
        return
    from cosmos_framework.posttrain.model.alpamayo.sft import AlpamayoSFTPacker

    if type(policy.packer) is not AlpamayoSFTPacker:
        raise ValueError("evaluation input reuse requires the native Alpamayo packer")
    policy.packer = _EvaluationPacker(policy.packer)


@contextmanager
def evaluation_batch_scope(policy, batch):
    from cosmos_framework.posttrain.utils.gr00t_evaluation_backbone import evaluation_scope

    packer = getattr(policy, "packer", None)
    prepared = packer.batch_scope(batch) if isinstance(packer, _EvaluationPacker) else nullcontext()
    with prepared, evaluation_scope(policy, batch):
        yield


def preparation_observations(policy):
    packer = getattr(policy, "packer", None)
    if not isinstance(packer, _EvaluationPacker):
        return None
    if packer.pending:
        raise RuntimeError("evaluation preparation state must be cleared before publication")
    return dict(packer.stats)
