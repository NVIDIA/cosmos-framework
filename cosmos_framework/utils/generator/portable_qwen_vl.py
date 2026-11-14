# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Public FP32 deployment of the recorded ordered HF Qwen-VL adapter path.

This is ordinary recorded adapter execution, not a cache-consistent numerical
profile. Base construction shares the pretrained loader with posttrain, without
importing a training wrapper, trainer, optimizer or distributed setup.
"""

import copy
import re
from functools import partial
from inspect import signature
from pathlib import Path

PROFILE = "qwen_vl_hf_ordered_v1"
STOCK_FACTORIZED_PROFILE = "qwen3_vl_factorized_fp32_v1"
PROFILES = (PROFILE, STOCK_FACTORIZED_PROFILE)
FAMILIES = ("qwen2_vl", "qwen2_5_vl", "qwen3_vl")


def validate_processor(processor, family):
    import transformers

    types = {
        "qwen2_vl": transformers.Qwen2VLProcessor,
        "qwen2_5_vl": transformers.Qwen2_5_VLProcessor,
        "qwen3_vl": transformers.Qwen3VLProcessor,
    }
    if family not in types or type(processor) is not types[family]:
        raise ValueError("ordered portable processor must match the recorded stock Qwen-VL architecture")


def validate_identity(identity, *, profile=PROFILE):
    from cosmos_framework.posttrain.checkpoint.adapters import state_spec

    if profile not in PROFILES:
        raise ValueError(f"unknown original-factorized VLM deployment profile: {profile!r}")
    state_spec(identity)
    stock = profile == STOCK_FACTORIZED_PROFILE
    execution = "qwen3_vl_hf_stock_v1" if stock else PROFILE
    if (
        identity["format"] != "cosmos-posttrain-adapter-v3"
        or identity["architecture"] not in (("qwen3_vl",) if stock else FAMILIES)
        or identity["base_storage_dtype"] != "bfloat16"
        or identity["resolved"].get("model_execution") != execution
        or not identity["resolved"]["targets"]
        or any(row["settings"]["dtype"] != "float32" for row in identity["resolved"]["targets"])
        or (
            stock
            and any(
                row["settings"].get("execution", "factorized_v1") != "factorized_v1"
                for row in identity["resolved"]["targets"]
            )
        )
    ):
        raise ValueError(f"{profile} requires its recorded BF16-base/FP32-adapter HF VLM artifact")


def load_base(base_path, hf_config, device, identity):
    """Materialize the original checkpoint namespace using the shared LoadPlan.

    Inspect every declared source tensor before copying. Preserve embedding ties
    by identity, and reject ambiguous aliases instead of silently picking a shard.
    Ordered forward/adapters are installed only after the caller resolves and
    checks the entire artifact identity against this unadapted model.
    """
    import torch
    from accelerate import init_on_device
    from safetensors import safe_open
    from transformers import AutoModelForImageTextToText, GenerationConfig

    from cosmos_framework.posttrain.checkpoint.adapters import adapter_configuration
    from cosmos_framework.posttrain.model.adapters import resolve_targets
    from cosmos_framework.posttrain.model.qwen_vl_hf_execution import canonical_weight_name
    from cosmos_framework.posttrain.model.specs import LoadPlan
    from cosmos_framework.posttrain.model.weight_loader import load_weights

    validate_identity(identity)
    if hf_config.model_type != identity["architecture"] or getattr(hf_config, "auto_map", None):
        raise ValueError("ordered portable inference requires the recorded stock HF architecture, without remote code")
    with init_on_device("meta", include_buffers=False):
        net = AutoModelForImageTextToText.from_config(
            hf_config, dtype=torch.bfloat16, attn_implementation="sdpa", trust_remote_code=False
        )
    # Match posttrain's uniform frozen storage, without rounding its real rotary
    # buffers. Re-tie after materialization because _apply can replace Parameters.
    for parameter in net.parameters():
        if parameter.is_floating_point() and parameter.dtype != torch.bfloat16:
            parameter.data = parameter.data.to(torch.bfloat16)
    net.tie_weights()
    if resolve_targets(net, adapter_configuration(identity), family=identity["architecture"]) != identity["resolved"]:
        raise ValueError("ordered portable target/extra identity mismatch before base materialization")
    state = net.state_dict(keep_vars=True)
    aliases = {}
    for name, value in state.items():
        aliases.setdefault(id(value), []).append(name)
    rename = partial(canonical_weight_name, hf_config.model_type)
    root = Path(base_path)
    shards = sorted(name for name in identity["base"]["files"] if name.endswith(".safetensors"))
    if set(shards) != {path.name for path in root.glob("*.safetensors")}:
        raise ValueError("portable base weight shards must exactly match the recorded local snapshot")
    source = {}
    for shard in shards:
        with safe_open(root / shard, framework="pt", device="cpu") as reader:
            for key in reader.keys():
                name = rename(key)
                if name in source or name not in state:
                    raise ValueError(f"unexpected or conflicting portable base tensor: {key!r} -> {name!r}")
                if tuple(reader.get_slice(key).get_shape()) != tuple(state[name].shape):
                    raise ValueError(f"portable base tensor shape mismatch: {name!r}")
                source[name] = (shard, key)
    skipped = []
    for names in aliases.values():
        present = [name for name in names if name in source]
        if not present:
            raise ValueError(f"missing portable base tensor: {names!r}")
        owner = present[0]
        # A checkpoint may store both members of a tied pair. They must describe
        # one value; a different alias may not silently replace its owner.
        if len(present) > 1:
            shard, key = source[owner]
            with safe_open(root / shard, framework="pt", device="cpu") as reader:
                expected = reader.get_tensor(key)
            for name in present[1:]:
                shard, key = source[name]
                with safe_open(root / shard, framework="pt", device="cpu") as reader:
                    actual = reader.get_tensor(key)
                if expected.dtype != actual.dtype or not torch.equal(expected, actual):
                    raise ValueError(f"conflicting tied portable base tensors: {owner!r}, {name!r}")
        skipped.extend(name for name in names if name != owner)
    net._apply(lambda value: torch.empty_like(value, device=device) if value.is_meta else value.to(device))
    net.tie_weights()
    load_weights(net, str(root), LoadPlan(rename=rename, skip_patterns=tuple(f"^{re.escape(n)}$" for n in skipped)))
    if any(value.is_meta for value in net.state_dict().values()):
        raise ValueError("portable base loading left unmaterialized tensors")
    if (root / "generation_config.json").is_file():
        net.generation_config = GenerationConfig.from_pretrained(root, local_files_only=True)
    return net


def install_inference_guard(net, identity, *, profile=PROFILE):
    """Declare FP32 eager inference without changing adapter/attention arithmetic."""
    import torch

    from cosmos_framework.posttrain.model.qwen_vl_hf_execution import _validate_owner

    validate_identity(identity, profile=profile)
    _validate_owner(net, identity["architecture"])
    if getattr(net, "_cosmos_deployment_profile", None) is not None:
        raise ValueError("ordered portable inference conflicts with an already installed deployment profile")
    stock = profile == STOCK_FACTORIZED_PROFILE
    execution = "qwen3_vl_hf_stock_v1" if stock else PROFILE
    if stock and any(module.training for module in net.modules()):
        raise ValueError(f"{profile} requires every module, including configured dropout, in eval mode")
    captured_identity = copy.deepcopy(identity)
    captured_config = copy.deepcopy(net.config.to_dict())
    forward_signature = signature(net.forward)

    def tensor_owner(name, tensor):
        return name, id(tensor), tensor.dtype, tensor.device, tuple(tensor.shape)

    owners = tuple(tensor_owner(name, p) for name, p in net.named_parameters(remove_duplicate=False))
    initials = tuple(
        tensor_owner(name, b) for name, b in net.named_buffers() if name.endswith(("initial_A", "initial_B"))
    )
    if any(dtype != torch.float32 for _, _, dtype, _, _ in owners):
        raise ValueError("ordered portable inference requires explicit FP32 parameter promotion")
    forward = net.model.forward.__func__

    def settings(model):
        return tuple(
            (
                name,
                id(module),
                module.method,
                module.execution,
                module._lora_scale,
                getattr(module.lora_dropout, "p", 0.0),
                id(module.lora_dropout),
            )
            for name, module in model.named_modules()
            if hasattr(module, "lora_A")
        )

    adapter_settings = settings(net)

    def callable_owner(fn):
        # Attribute access creates a fresh bound-method object. Its function and
        # receiver, rather than the transient method object, own the execution.
        return id(getattr(fn, "__self__", None)), id(getattr(fn, "__func__", fn))

    def execution_owners(model):
        return tuple(
            (
                name,
                id(module),
                callable_owner(module.forward),
                callable_owner(getattr(module, "_old_forward", None)),
                id(getattr(module, "_hf_hook", None)),
            )
            for name, module in model.named_modules()
        )

    if stock:
        from cosmos_framework.posttrain.model.adapters import PosttrainAdapterLinear

        for module in net.modules():
            if isinstance(module, PosttrainAdapterLinear):
                original = getattr(module, "_old_forward", module.forward)
                if (
                    getattr(original, "__self__", None) is not module
                    or getattr(original, "__func__", None) is not PosttrainAdapterLinear.forward
                ):
                    raise ValueError(f"{profile} requires original factorized adapter forward ownership")
    captured_execution = execution_owners(net) if stock else None

    def guard(model, args, kwargs):
        if (
            model.training
            or torch.is_grad_enabled()
            or torch.is_autocast_enabled("cuda")
            or torch.is_autocast_enabled("cpu")
            or torch.compiler.is_compiling()
            or (stock and any(module.training for module in model.modules()))
        ):
            raise ValueError(f"{profile} is eager FP32 eval/no_grad only, including dropout, without autocast/compile")
        if stock and (torch.backends.cuda.matmul.allow_tf32 or torch.cuda.is_current_stream_capturing()):
            raise ValueError(f"{profile} requires FP32 matmul without TF32 or CUDA graph capture")
        if stock:
            from transformers.cache_utils import DynamicCache, DynamicLayer

            call = forward_signature.bind_partial(*args, **kwargs).arguments
            for name, ndim in (("input_ids", 2), ("inputs_embeds", 3)):
                value = call.get(name)
                if value is not None and (value.ndim != ndim or value.shape[0] != 1):
                    raise ValueError(f"{profile} requires batch-one {name}, without beam expansion")
            cache = call.get("past_key_values")
            if cache is not None and (
                type(cache) is not DynamicCache
                or getattr(cache, "offloading", False)
                or any(type(layer) is not DynamicLayer for layer in cache.layers)
                or cache.layer_class_to_replicate not in (None, DynamicLayer)
                or (
                    cache.layer_class_to_replicate is None
                    and len(cache.layers) != model.config.text_config.num_hidden_layers
                )
            ):
                raise ValueError(f"{profile} supports ordinary full-attention DynamicCache without offload")
        inputs = kwargs.get("input_ids", args[0] if args else None)
        if inputs is None:
            inputs = kwargs.get("inputs_embeds")
        if inputs is not None and int(inputs.shape[0]) != 1:
            raise ValueError("ordered portable deployment requires batch-one inference, without beam expansion")
        if (
            model._cosmos_deployment_profile != profile
            or model._posttrain_model_execution != execution
            or model._posttrain_adapter_identity != captured_identity
            or model.config.to_dict() != captured_config
            or getattr(model.model.forward, "__func__", None) is not forward
            or settings(model) != adapter_settings
            or (stock and execution_owners(model) != captured_execution)
            or tuple(tensor_owner(n, p) for n, p in model.named_parameters(remove_duplicate=False)) != owners
            or tuple(tensor_owner(n, b) for n, b in model.named_buffers() if n.endswith(("initial_A", "initial_B")))
            != initials
        ):
            raise ValueError(f"{profile} parameter/profile ownership changed; reload the declared artifact")
        _validate_owner(model, identity["architecture"])

    net._cosmos_deployment_profile = profile
    net.register_forward_pre_hook(guard, with_kwargs=True)
