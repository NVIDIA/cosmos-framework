# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""GR00T's explicit architecture boundary for shared native backbone adaptation."""

from __future__ import annotations

import functools
import json
from contextlib import ExitStack
from pathlib import Path
from typing import Any

ISAAC_GR00T_REVISION = "51d4c89f72fda44cbf77285c6a8114b52676b8a1"
LANGUAGE_PROJECTIONS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


def resolve_language_targets(model: Any) -> dict[str, tuple[int, int]]:
    """Resolve by module ownership, never broad substring matching across vision/action modules."""
    import torch.nn as nn

    config = model.config
    if (
        getattr(config, "model_type", None) != "Gr00tN1d7"
        or getattr(config, "select_layer", None) != 16
        or getattr(config, "backbone_embedding_dim", None) != 2048
        or getattr(config, "action_horizon", None) != 40
        or getattr(config, "max_action_dim", None) != 132
        or getattr(config, "num_inference_timesteps", None) != 4
    ):
        raise ValueError("GR00T adaptation requires the pinned N1.7 Reason2-2B layer-16/action-40x132 profile")
    language = model.backbone.model.language_model
    if len(language.layers) != 16:
        raise ValueError("GR00T language model must keep exactly 16 layers (select_layer=16)")
    occurrences: dict[int, list[str]] = {}
    for name, module in model.named_modules(remove_duplicate=False):
        occurrences.setdefault(id(module), []).append(name)
    targets = {}
    for layer in language.layers:
        for relative in LANGUAGE_PROJECTIONS:
            module = layer.get_submodule(relative)
            names = occurrences.get(id(module), [])
            if len(names) != 1 or not names[0].startswith("backbone."):
                raise ValueError(f"GR00T target is aliased or outside the backbone: {names}")
            if type(module) is not nn.Linear or module.bias is not None:
                raise ValueError(f"GR00T target {names[0]} must be an ordinary bias-free Linear")
            if module.in_features != 2048 and module.out_features != 2048:
                raise ValueError(f"GR00T target {names[0]} does not touch the pinned hidden width")
            targets[names[0]] = tuple(module.weight.shape)
    if len(targets) != 112:
        raise ValueError("GR00T adaptation must resolve exactly 112 language projections")
    return targets


def validate_task_config(config):
    """Reject settings that the selected official checkpoint/processor cannot honor."""
    keys = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
    if (
        config.embodiment_tag != "libero_sim"
        or config.action_dim != 7
        or config.model_action_horizon != 40
        or config.denoising_steps != 4
        or not 1 <= config.training_prefix <= 16
        or config.camera_map != {"image": "image", "wrist_image": "wrist_image"}
        or config.state_map != {key: key for key in keys}
        or config.prompt_key != "prompt"
        or config.language_key != "annotation.language.language_instruction"
    ):
        raise ValueError(
            "GR00T adaptation requires the explicit LIBERO camera/state schema, 7 actions, 40 head slots, 1..16 supervised slots and four sampling steps"
        )
    if config.backbone_buffer_dtype is not None:
        raise ValueError(
            "GR00T adaptation preserves native FP32 rotary buffers; backbone_buffer_dtype override is unsupported"
        )


@functools.cache
def verify_external_source() -> None:
    """Runtime preflight, once per process: Isaac-GR00T is the pinned source checkout."""
    import gr00t

    from cosmos_framework.posttrain.utils.native_source import require_source_revision

    require_source_revision(gr00t.__file__, ISAAC_GR00T_REVISION, "Isaac-GR00T")


def prepare_task_inputs(model, inputs, native_prepare):
    """Preserve official FP32 state/action/noise math with a mixed-precision backbone.

    Isaac routes every floating input through the first parameter's dtype. A BF16
    frozen embedding must not silently cast the demonstration/noise schedule to BF16.
    Restore directly from processor FP32 values, never from an already-rounded copy.
    """
    import torch

    fields = {}
    for name in ("state", "action", "action_mask"):
        if name in inputs:
            value = inputs[name]
            if not torch.is_tensor(value) or not value.is_floating_point():
                raise TypeError(f"GR00T native task input {name} must be floating-point")
            fields[name] = value.to(device=model.device, dtype=torch.float32)
    backbone, action = native_prepare(dict(inputs))
    for name, value in fields.items():
        action[name] = value
    return backbone, action


def construct_meta(payload, backbone_path):
    """Use Isaac's unchanged forward and exact HF architecture without nested weight loading.

    Parameters stay on the meta device until the shared one-tensor source reader installs
    their final representation.
    """
    import torch
    from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7, Gr00tN1d7ActionHead
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7DataCollator
    from gr00t.model.modules.qwen3_backbone import Qwen3Backbone
    from torch import nn
    from transformers import PreTrainedModel, Qwen3VLConfig, Qwen3VLForConditionalGeneration

    class Backbone(Qwen3Backbone):
        def __init__(self, config):
            nn.Module.__init__(self)
            backbone_config = Qwen3VLConfig.from_pretrained(backbone_path, local_files_only=True)
            if not 1 <= config.select_layer <= backbone_config.text_config.num_hidden_layers:
                raise ValueError("GR00T selected layer differs from its source backbone")
            backbone_config._attn_implementation = "sdpa"
            backbone_config.text_config.use_cache = False
            self.model = Qwen3VLForConditionalGeneration(backbone_config).bfloat16()
            while len(self.model.language_model.layers) > config.select_layer:
                self.model.language_model.layers.pop(-1)
            self.select_layer = config.select_layer
            self.set_trainable_parameters(False, False, 0)
            self._reset_rotary_inv_freq()
            self._apply_vision_patch_embed_channels_last()

    class Net(Gr00tN1d7):
        def prepare_input(self, inputs):
            return prepare_task_inputs(self, inputs, super().prepare_input)

        def __init__(self, config):
            PreTrainedModel.__init__(self, config)
            self.backbone = Backbone(config)
            self.action_head = Gr00tN1d7ActionHead(config).float()
            self.collator = Gr00tN1d7DataCollator(backbone_path, transformers_loading_kwargs={"local_files_only": True})

    # Official FinetuneConfig head selection/state dropout, not pretraining defaults.
    from gr00t.configs.finetune_config import FinetuneConfig

    finetune = FinetuneConfig(base_model_path="", dataset_path="", embodiment_tag="")
    payload = {
        **payload,
        "tune_llm": False,
        "tune_visual": False,
        "tune_projector": finetune.tune_projector,
        "tune_diffusion_model": finetune.tune_diffusion_model,
        "state_dropout_prob": finetune.state_dropout_prob,
        "use_flash_attention": False,
    }
    with torch.device("meta"):
        return Net(Gr00tN1d7Config(**payload))


def checkpoint_weight_map(checkpoint):
    """Match HF/source-provenance precedence when both storage layouts are present."""
    from safetensors import safe_open

    checkpoint = Path(checkpoint)
    single = checkpoint / "model.safetensors"
    if single.is_file():
        with safe_open(str(single), framework="pt") as handle:
            return dict.fromkeys(handle.keys(), "model.safetensors")
    return json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]


def task_identity(config: Any, checkpoint: Path, backbone: Path) -> dict[str, Any]:
    """Model-defining task contract from the native checkpoint and backbone metadata files.

    Machine paths, process math flags, the supervised ``training_prefix`` and the adapter
    seed/dropout (training-only, bound by the run's training contract) and the rollout-only
    ``execution_horizon`` are not part of it.
    """
    from cosmos_framework.posttrain.model.gr00t.source import metadata_files

    config_identity = config.identity()
    for name in ("checkpoint_path", "revision", "training_prefix", "execution_horizon"):
        config_identity.pop(name)
    for name in ("backbone_path", "seed", "dropout"):
        config_identity["backbone_adaptation"].pop(name)
    result = {
        "family": "gr00t_n1d7",
        "external": {"revision": ISAAC_GR00T_REVISION},
        "config": config_identity,
        "native_model_config": json.loads((checkpoint / "config.json").read_text()),
        "precision": {
            "backbone_parameters": "bfloat16",
            "adapter_parameters": "float32",
            "action_head_parameters": "float32",
            "compute": "bfloat16-autocast-including-adapter-gemm",
            "task_state_action_noise": "float32",
        },
        "backbone_metadata": metadata_files(backbone),
        "processor_config": json.loads((checkpoint / "processor_config.json").read_text()),
        "statistics": json.loads((checkpoint / "statistics.json").read_text()),
    }
    embodiment = checkpoint / "embodiment_id.json"
    if embodiment.exists():
        result["embodiment_id"] = json.loads(embodiment.read_text())
    return result


def build_adapted_gr00t(config: Any, *, device: Any, save_cfg_dir: Any) -> tuple[Any, Any, Any]:
    """Stream each full-GR00T source tensor into final storage; no dense CPU/GPU base copy."""
    import torch
    from safetensors import safe_open

    from cosmos_framework.posttrain.model.gr00t.head_adaptation import configured_head_targets
    from cosmos_framework.posttrain.model.gr00t.policy import load_processor
    from cosmos_framework.posttrain.model.gr00t.qlora_source import pinned_source_identity
    from cosmos_framework.posttrain.model.gr00t.source import resolve_source
    from cosmos_framework.posttrain.model.quantization.family import install_family_backbone

    validate_task_config(config)
    adaptation = config.backbone_adaptation
    adaptation.__post_init__()
    verify_external_source()
    checkpoint, _ = resolve_source(config.checkpoint_path, config.revision)
    backbone = Path(adaptation.backbone_path).expanduser().resolve()
    if not (backbone / "config.json").is_file():
        raise ValueError("GR00T adaptation needs its exact staged Cosmos-Reason2-2B config/tokenizer/processor")
    source_identity = pinned_source_identity(checkpoint, backbone)
    payload = json.loads((checkpoint / "config.json").read_text())
    model = construct_meta(payload, str(backbone))
    targets = resolve_language_targets(model)
    head_targets = configured_head_targets(model, adaptation)
    auxiliary = tuple(
        name
        for name, parameter in model.named_parameters()
        if name.startswith("action_head.") and parameter.requires_grad and adaptation.train_action_head
    )
    weight_map = checkpoint_weight_map(checkpoint)
    expected = model.state_dict()
    if set(expected) != set(weight_map):
        raise ValueError(
            f"GR00T checkpoint inventory differs: missing={sorted(set(expected) - set(weight_map))[:8]}, "
            f"unexpected={sorted(set(weight_map) - set(expected))[:8]}"
        )

    shards = ExitStack()
    handles = {}

    def read_tensor(name):
        shard = weight_map[name]
        if Path(shard).name != shard:
            raise ValueError("GR00T source contains an invalid safetensors shard path")
        if shard not in handles:
            handles[shard] = shards.enter_context(safe_open(str(checkpoint / shard), framework="pt", device="cpu"))
        value = handles[shard].get_tensor(name)
        if tuple(value.shape) != tuple(expected[name].shape) or not value.is_floating_point():
            raise ValueError(f"GR00T checkpoint tensor has incompatible shape/type: {name}")
        value = value.to(expected[name].dtype)
        # Preserve Isaac's A100 Conv3d layout without registering its input hook a second time.
        if value.ndim == 5 and expected[name].is_contiguous(memory_format=torch.channels_last_3d):
            value = value.contiguous(memory_format=torch.channels_last_3d)
        return value

    with shards:
        state = install_family_backbone(
            model,
            config=adaptation,
            targets=targets,
            auxiliary_parameters=auxiliary,
            source_identity=dict(source_identity),
            task_identity=task_identity(config, checkpoint, backbone),
            device=torch.device(device),
            read_tensor=read_tensor,
            head_targets=head_targets,
        )
    # `original_inv_freq` is upstream's unregistered alias, not a weight or packed quantization
    # buffer. Bind it to the moved registered buffer without recomputing arithmetic.
    rotary = model.backbone.model.language_model.rotary_emb
    if hasattr(rotary, "original_inv_freq"):
        rotary.original_inv_freq = rotary.inv_freq
    model.backbone.tune_llm = adaptation.mode.endswith("_lora")
    model.backbone.tune_visual = False
    model.config.tune_llm = model.backbone.tune_llm
    processor = load_processor(str(checkpoint), model_name=str(backbone))
    if pinned_source_identity(checkpoint, backbone) != source_identity:
        raise ValueError("GR00T source changed during streamed construction")
    Path(save_cfg_dir).mkdir(parents=True, exist_ok=True)
    return model, processor, state
