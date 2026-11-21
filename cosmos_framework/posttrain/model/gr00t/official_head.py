# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Existing GR00T head-finetuning control with explicit resource-adjusted native SFT.

This does not quantize or inject LoRA. It preserves FP32 source storage and the
pinned finetune optimizer/augmentation policy. Shared QLoRA constructors and
arithmetic identities are deliberately unchanged.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

from cosmos_framework.posttrain.config.gr00t_official import OFFICIAL_HEAD_PROFILE
from cosmos_framework.posttrain.model.gr00t.action_sft import Gr00tActionDataConfig, Gr00tActionPacker


@dataclass
class OfficialHeadDataConfig(Gr00tActionDataConfig):
    backbone_path: str = ""


@contextmanager
def processor_mode(processor, training):
    """Processor augmentation is not an nn.Module mode; restore it even on failure."""
    previous = processor.training
    (processor.train if training else processor.eval)()
    try:
        yield
    finally:
        (processor.train if previous else processor.eval)()


class OfficialHeadPacker(Gr00tActionPacker):
    def __init__(self, model, backbone_path, net):
        # Only the existing packer's processor-path argument needs an adaptation-shaped view.
        # The actual model remains a distinct no-adapter official-head policy.
        processor_config = SimpleNamespace(backbone_adaptation=SimpleNamespace(backbone_path=backbone_path))
        super().__init__(
            model.checkpoint_path,
            processor_config,
            training_prefix=model.training_prefix,
            action_dim=model.action_dim,
        )
        self.config = model
        self.net = net
        self.processor.state_dropout_prob = 0.2  # pinned FinetuneConfig override passed to official processor

    def __call__(self, trajectories):
        with processor_mode(self.processor, self.net.training):
            return super().__call__(trajectories)


def materialize_official_parameters(net, read_tensor, device):
    """Stream one CPU source tensor at a time into FP32 native storage; preserve aliases/layout."""
    import torch
    from torch import nn

    from cosmos_framework.posttrain.model.quantization.loader import TensorSlot, parameter_slots

    slots = parameter_slots(net)
    buffers = {
        f"{name}.{attribute}" if name else attribute: TensorSlot(owner, attribute)
        for name, owner in net.named_modules()
        for attribute, value in owner._buffers.items()
        if value is not None
    }
    if any(not slot.tensor.is_meta for slot in slots.values()):
        raise ValueError("official head source must start on meta, without a dense GPU loading copy")
    if any(slot.tensor.is_meta or slot.tensor.device.type != "cpu" for slot in buffers.values()):
        raise ValueError("official head rotary buffers must be initialized on CPU")
    originals = {name: slot.tensor for name, slot in slots.items()}
    moved = {}
    for name, slot in slots.items():
        original = originals[name]
        if not original.is_floating_point():
            raise ValueError(f"unsupported official GR00T parameter dtype: {name}")
        if id(original) not in moved:
            value = read_tensor(name)
            if (
                value.device.type != "cpu"
                or tuple(value.shape) != tuple(original.shape)
                or not value.is_floating_point()
            ):
                raise ValueError(f"official source tensor shape/device/dtype differs: {name}")
            value = value.to(dtype=torch.float32)
            if not torch.isfinite(value).all().item():
                raise ValueError(f"nonfinite official source tensor: {name}")
            if value.ndim == 5 and original.is_contiguous(memory_format=torch.channels_last_3d):
                value = value.contiguous(memory_format=torch.channels_last_3d)
            moved[id(original)] = nn.Parameter(
                value.to(device=device), requires_grad=name.startswith("action_head.") and original.requires_grad
            )
        setattr(slot.owner, slot.attribute, moved[id(original)])
    for slot in buffers.values():
        setattr(slot.owner, slot.attribute, slot.tensor.to(device=device))
    return buffers


def attach_official_codec(net, identity):
    from cosmos_framework.posttrain.model.checkpoint import AdapterCheckpointCodec
    from cosmos_framework.posttrain.model.quantization.family import _family_tensor_record
    from cosmos_framework.posttrain.model.quantization.loader import parameter_slots

    parameters = {name: value for name, value in net.named_parameters() if value.requires_grad}
    if not parameters or any(not name.startswith("action_head.") or ".lora_" in name for name in parameters):
        raise ValueError("official GR00T control permits only native action-head trainables")
    slots = parameter_slots(net)
    frozen = {name: _family_tensor_record(slot.tensor) for name, slot in slots.items() if not slot.tensor.requires_grad}
    buffers = {name: _family_tensor_record(value) for name, value in net.named_buffers()}
    identity = {
        **identity,
        "format": "cosmos.gr00t-official-head.v1",
        "optimizer_profile": OFFICIAL_HEAD_PROFILE,
        "auxiliary_parameters": sorted(parameters),
        "frozen_tensors": frozen,
        "base_buffers": buffers,
    }

    def validate_frozen():
        current = parameter_slots(net)
        for name, expected in frozen.items():
            if _family_tensor_record(current[name].tensor) != expected:
                raise RuntimeError(f"official frozen backbone changed: {name}")
        current_buffers = dict(net.named_buffers())
        for name, expected in buffers.items():
            if _family_tensor_record(current_buffers[name]) != expected:
                raise RuntimeError(f"official frozen buffer changed: {name}")

    net._cosmos_adapter_parameters = dict(sorted(parameters.items()))
    net._native_adapter_parameter_names = tuple(sorted(parameters))
    net._cosmos_adapter_metadata = identity
    net._cosmos_adapter_optimizer_profile = OFFICIAL_HEAD_PROFILE
    net._cosmos_validate_frozen_base = validate_frozen
    net._cosmos_adapter_checkpoint_codec = AdapterCheckpointCodec(net, identity=identity)
    return net._cosmos_adapter_checkpoint_codec


def build_policy(config, *, device, output):
    import torch
    import torch.distributed as dist
    from safetensors import safe_open

    from cosmos_framework.posttrain.config.gr00t_official import validate_official_head_optimizer
    from cosmos_framework.posttrain.model.gr00t.adaptation import (
        _construct_meta,
        _external_identity,
        checkpoint_weight_map,
        resolve_language_targets,
        validate_task_config,
    )
    from cosmos_framework.posttrain.model.gr00t.policy import OFFICIAL_PRECISION, Gr00tN1d7TrainPolicy
    from cosmos_framework.posttrain.model.gr00t.qlora_source import pinned_source_identity
    from cosmos_framework.posttrain.model.precision import compute_context
    from cosmos_framework.posttrain.model.quantization.loader import guard_resident_movement
    from cosmos_framework.posttrain.utils.object_store import file_sha256
    from cosmos_framework.utils.lazy_config import instantiate

    model, data = instantiate(config.model), instantiate(config.data)
    model.__post_init__()
    validate_task_config(model)
    if config.rl.optimizer_profile != OFFICIAL_HEAD_PROFILE or model.backbone_adaptation is not None:
        raise ValueError("official head factory requires its explicit optimizer profile and no backbone adaptation")
    validate_official_head_optimizer(config.rl.optimizer)
    device = torch.device(device)
    if (
        device.type != "cuda"
        or not dist.is_initialized()
        or dist.get_world_size() != 1
        or device.index != torch.cuda.current_device()
        or not torch.cuda.is_bf16_supported()
    ):
        raise ValueError("official head control requires one native BF16-capable CUDA torchrun process")
    if version("albumentations") != "1.4.18":
        raise ValueError("official head augmentation recovery requires pinned Albumentations 1.4.18 global RNG")
    external = _external_identity()
    checkpoint, backbone = Path(model.checkpoint_path).resolve(), Path(data.backbone_path).resolve()
    source = pinned_source_identity(checkpoint, backbone)
    payload = json.loads((checkpoint / "config.json").read_text())
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    net = _construct_meta(payload, str(backbone)).float()
    resolve_language_targets(net)  # architecture assertion only; no quantization or adapters
    weight_map = checkpoint_weight_map(checkpoint)
    if set(net.state_dict()) != set(weight_map):
        raise ValueError("official head source inventory differs from native architecture")

    def read_tensor(name):
        shard = weight_map[name]
        if Path(shard).name != shard:
            raise ValueError("invalid source safetensors shard path")
        with safe_open(str(checkpoint / shard), framework="pt", device="cpu") as reader:
            return reader.get_tensor(name)

    materialize_official_parameters(net, read_tensor, device)
    rotary = net.backbone.model.language_model.rotary_emb
    if hasattr(rotary, "original_inv_freq"):
        rotary.original_inv_freq = rotary.inv_freq
    policy = Gr00tN1d7TrainPolicy.__new__(Gr00tN1d7TrainPolicy)
    policy.config, policy._ppo_enabled, policy.precision = model, False, OFFICIAL_PRECISION
    policy.gr00t = policy.net = net
    policy.backbone = net.backbone
    policy.compute_ctx = compute_context("autocast", torch.bfloat16, device.type)
    policy.packer = OfficialHeadPacker(model, str(backbone), net)
    policy.processor = policy.packer.processor
    task = model.identity()
    task.pop("checkpoint_path")
    task.pop("revision")
    identity = {
        "base": source,
        "task": task,
        "external": external,
        "precision": {"all_parameters": "float32", "forward": "bfloat16-autocast", "optimizer_state": "float32"},
        "processor_policy": {
            "training": "native-train-transforms",
            "validation": "native-eval-transforms",
            "state_dropout_prob": 0.2,
        },
        "processor_config": json.loads((checkpoint / "processor_config.json").read_text()),
        "statistics": json.loads((checkpoint / "statistics.json").read_text()),
        "implementation": {
            name: file_sha256(str(Path(__file__).parent / name))
            for name in (
                "official_head.py",
                "adaptation.py",
                "action_sft.py",
                "policy.py",
                "packer.py",
                "qlora_source.py",
            )
        },
        "dependencies": {
            name: version(name) for name in ("torch", "transformers", "safetensors", "albumentations", "torchvision")
        },
    }
    embodiment = checkpoint / "embodiment_id.json"
    if embodiment.exists():
        identity["embodiment_id"] = json.loads(embodiment.read_text())
    attach_official_codec(net, identity)
    guard_resident_movement(net, device)
    if pinned_source_identity(checkpoint, backbone) != source:
        raise ValueError("official head source changed during materialization")
    Path(output).mkdir(parents=True, exist_ok=True)
    return policy
