# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Experimental fixed projection reductions for the stock Qwen3-VL-2B binding.

Every language projection reduces in the existing fixed A100 FP64 MMA order,
then rounds to its original FP32 boundary. Native normalization, SDPA, RoPE,
vision and adapter branch order are retained. This is an explicit inference
profile, not a training change or a claim that whole-model cache gates pass.
"""

import copy
import hashlib
import json
import math
from importlib.metadata import version
from inspect import signature
from pathlib import Path
from types import MethodType
from uuid import uuid4

import torch

from cosmos_framework.utils.generator import qwen3_vl_projection_contract as contract

PROFILE = contract.PROFILE


def _project(inputs, weight, bias=None):
    from cosmos_framework.utils.generator.smollm_fixed_mma import linear

    # Each native projection returns FP32, including the intermediate A branch.
    # Do not silently acquire the FP64 composite adapter algebra of other profiles.
    return linear(inputs, weight, bias).to(inputs.dtype)


def _guard(module):
    if (
        module.training
        or torch.is_grad_enabled()
        or torch.is_autocast_enabled("cuda")
        or torch.is_autocast_enabled("cpu")
        or torch.compiler.is_compiling()
        or torch.jit.is_tracing()
        or torch.cuda.is_current_stream_capturing()
        or torch.backends.cuda.matmul.allow_tf32
    ):
        raise RuntimeError(f"{PROFILE} requires eager FP32 eval/no_grad without TF32, autocast or graph capture")


def _linear(module, inputs):
    _guard(module)
    if inputs.dtype != torch.float32:
        raise ValueError(f"{PROFILE} requires FP32 projection inputs")
    return _project(inputs, module.weight, module.bias)


def _adapter(module, inputs):
    """Keep the native FP32 boundaries while replacing direct projection calls.

    LoRA uses the existing adapter_output method. DoRA uses the existing shared
    merged_weight method solely for its unchanged FP32 directional row norm.
    PiSSA retains separate trained and frozen initial projections. The patched
    A/B children round independently to FP32, as the original factorized path.
    """
    _guard(module)
    if inputs.dtype != torch.float32:
        raise ValueError(f"{PROFILE} requires FP32 adapter inputs")
    base = _project(inputs, module.weight, module.bias)
    if module.method == "lora":
        return base + module._lora_scale * module.adapter_output(inputs).to(base.dtype)
    dropped = module.lora_dropout(inputs)
    without_bias = _project(dropped, module.weight) if module.method == "dora" else None
    delta = module.lora_B(module.lora_A(dropped.float())) * module._lora_scale
    if module.method == "pissa":
        initial = _project(_project(inputs.float(), module.initial_A), module.initial_B) * module._lora_scale
        return (base.float() + (delta - initial)).to(base.dtype)
    if module.method != "dora":
        raise ValueError(f"{PROFILE} adapter method changed")
    direction = module.merged_weight(module.weight, module.lora_A.weight, module.lora_B.weight, dtype=torch.float32)
    norm = direction.norm(dim=1).clamp_min(torch.finfo(torch.float32).tiny)
    ratio = module.magnitude / norm
    return (base.float() + (ratio - 1) * without_bias.float() + ratio * delta).to(base.dtype)


def _callable(fn):
    return id(getattr(fn, "__self__", None)), id(getattr(fn, "__func__", fn))


def _executions(model):
    return tuple(
        (
            name,
            id(module),
            _callable(module.forward),
            _callable(getattr(module, "_old_forward", None)),
            id(getattr(module, "_hf_hook", None)),
            _callable(getattr(module, "adapter_output", None)),
            _callable(getattr(module, "merged_weight", None)),
        )
        for name, module in model.named_modules(remove_duplicate=False)
    )


def _ownership(model):
    values = list(model.named_parameters(remove_duplicate=False)) + list(model.named_buffers(remove_duplicate=False))
    return tuple(
        (name, id(t), t.data_ptr(), t._version, t.dtype, t.device, tuple(t.shape), tuple(t.stride()), t.requires_grad)
        for name, t in values
    )


def _settings(module):
    return (
        module.method,
        module.execution,
        module._lora_rank,
        module._lora_alpha,
        module._lora_use_rslora,
        module._lora_scale,
        id(module.lora_dropout),
        getattr(module.lora_dropout, "p", 0.0),
        getattr(module, "pissa_input_gradient", "separate_casts_v1"),
    )


def _native_settings(model):
    fields = (
        "variance_epsilon",
        "eps",
        "head_dim",
        "num_key_value_groups",
        "scaling",
        "attention_dropout",
        "is_causal",
        "layer_idx",
        "rope_type",
        "attention_scaling",
        "mrope_section",
        "max_seq_len_cached",
        "original_max_seq_len",
    )
    return tuple(
        (
            name,
            id(getattr(module, "config", None)),
            tuple((field, copy.deepcopy(getattr(module, field))) for field in fields if hasattr(module, field)),
        )
        for name, module in model.named_modules()
    )


def _device(model):
    parameters = list(model.parameters())
    devices = {p.device for p in parameters}
    if (
        len(devices) != 1
        or next(iter(devices)).type != "cuda"
        or torch.cuda.get_device_capability(next(iter(devices))) != (8, 0)
        or "A100" not in torch.cuda.get_device_name(next(iter(devices)))
        or any(p.dtype != torch.float32 or p.is_inference() for p in parameters)
    ):
        raise ValueError(f"{PROFILE} requires FP32 parameters on one A100, constructed outside inference_mode")


def _check_owned(model, record):
    if (
        getattr(model, "_cosmos_deployment_profile", None) != PROFILE
        or getattr(model, "_cosmos_qwen3_fixed_invalidated", False)
        or getattr(model, "_cosmos_unadapted_state_invalidated", False)
    ):
        raise ValueError(f"{PROFILE} was invalidated; reload the immutable artifact")
    if _ownership(model) != record["ownership"] or _executions(model) != record["executions"]:
        raise ValueError(f"{PROFILE} tensor or callable ownership changed; reload the immutable artifact")
    if model.config.to_dict() != record["config"] or any(m.training for m in model.modules()):
        raise ValueError(f"{PROFILE} configuration or eval state changed")
    if any(_settings(module) != settings for module, settings in record["adapters"]):
        raise ValueError(f"{PROFILE} adapter settings changed")
    if _native_settings(model) != record["native_settings"]:
        raise ValueError(f"{PROFILE} native attention/norm/rotary settings changed")
    if any(
        getattr(module.config, "_attn_implementation", None) != "sdpa"
        for module in model.modules()
        if hasattr(module, "config")
    ):
        raise ValueError(f"{PROFILE} requires unchanged native SDPA")
    for module, attribute, old, installed in record["forwards"]:
        if getattr(module, attribute) != installed:
            raise ValueError(f"{PROFILE} projection ownership changed")


def _before_forward(model, args, kwargs):
    from transformers.cache_utils import DynamicCache, DynamicLayer

    _guard(model)
    record = model._cosmos_qwen3_fixed_mma
    _check_owned(model, record)
    call = record["signature"].bind_partial(*args, **kwargs).arguments
    ids, embeds = call.get("input_ids"), call.get("inputs_embeds")
    for value, ndim in ((ids, 2), (embeds, 3)):
        if value is not None and (value.ndim != ndim or value.shape[0] != 1):
            raise ValueError(f"{PROFILE} requires batch-one IDs/embeddings without beams")
    if ids is None and embeds is None:
        raise ValueError(f"{PROFILE} requires IDs or embeddings")
    if embeds is not None and embeds.dtype != torch.float32:
        raise ValueError(f"{PROFILE} requires FP32 embeddings")
    cache = call.get("past_key_values")
    if cache is not None and (
        type(cache) is not DynamicCache
        or getattr(cache, "offloading", False)
        or cache.layer_class_to_replicate not in (None, DynamicLayer)
        or any(type(layer) is not DynamicLayer for layer in cache.layers)
        or len(cache.layers) > model.config.text_config.num_hidden_layers
        or (cache.layer_class_to_replicate is None and len(cache.layers) != model.config.text_config.num_hidden_layers)
    ):
        raise ValueError(f"{PROFILE} supports ordinary full-attention DynamicCache only")
    if (
        cache is not None
        and cache.get_seq_length()
        and getattr(cache, "_cosmos_qwen3_fixed_owner", None) != record["cache_owner"]
    ):
        raise ValueError(f"{PROFILE} requires a prefix owned by this profile installation")


def _after_forward(model, args, kwargs, output):
    from transformers.cache_utils import DynamicCache

    record = model._cosmos_qwen3_fixed_mma
    call = record["signature"].bind_partial(*args, **kwargs).arguments
    caches = [call.get("past_key_values"), getattr(output, "past_key_values", None)]
    if isinstance(output, tuple):
        caches.extend(x for x in output if isinstance(x, DynamicCache))
    for cache in caches:
        if cache is not None:
            cache._cosmos_qwen3_fixed_owner = record["cache_owner"]


def install(model, *, identity, adapter_type=None, assembly_forward=None, extra_norm_forward=None):
    """Validate complete identity/placement before installing owned projections."""
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration

    base = identity.get("format") == contract.BASE_FORMAT
    (contract.validate_base_identity if base else contract.validate_identity)(identity)
    contract.validate_configuration(model.config)
    if type(model) is not Qwen3VLForConditionalGeneration or any(m.training for m in model.modules()):
        raise ValueError(f"{PROFILE} requires the stock Qwen3-VL-2B eval model")
    if any(version(name).split("+")[0] != wanted for name, wanted in contract.DEPENDENCIES.items()):
        raise ValueError(f"{PROFILE} requires the pinned Torch/Transformers/Triton dependencies")
    if any(
        getattr(section, "_attn_implementation", None) != "sdpa"
        for section in (model.config, model.config.text_config, model.config.vision_config)
    ):
        raise ValueError(f"{PROFILE} requires unchanged native SDPA throughout")
    if model.get_input_embeddings().weight is not model.get_output_embeddings().weight:
        raise ValueError(f"{PROFILE} requires the original tied embedding/output weight")
    _device(model)
    if getattr(model, "_cosmos_qwen3_fixed_invalidated", False) or getattr(
        model, "_cosmos_unadapted_state_invalidated", False
    ):
        raise ValueError(f"{PROFILE} was invalidated; reload before installation")
    installed = getattr(model, "_cosmos_deployment_profile", None)
    if installed is not None:
        if installed != PROFILE or not hasattr(model, "_cosmos_qwen3_fixed_mma"):
            raise ValueError(f"{PROFILE} conflicts with an already installed profile")
        record = model._cosmos_qwen3_fixed_mma
        if record["identity"] != identity:
            raise ValueError(f"{PROFILE} identity changed")
        _check_owned(model, record)
        return list(record["names"])
    targets = [(n, m) for n, m in model.named_modules(remove_duplicate=False) if hasattr(m, "lora_A")]
    if base and (targets or adapter_type is not None):
        raise ValueError("unadapted base cannot contain adapters")
    if targets:
        if (
            adapter_type is None
            or adapter_type.__module__ != "cosmos_framework.posttrain.model.adapters"
            or adapter_type.__name__ != "PosttrainAdapterLinear"
        ):
            raise ValueError(f"{PROFILE} requires native recorded adapter ownership")
        expected = {r["name"]: r for r in identity["resolved"]["targets"]}
        if {n for n, _ in targets} != set(expected):
            raise ValueError(f"{PROFILE} adapter placement changed")
        for name, module in targets:
            row = expected[name]
            settings = row["settings"]
            if (
                type(module) is not adapter_type
                or module.execution != "factorized_v1"
                or module._lora_scale != row["scaling"]
            ):
                raise ValueError(f"{PROFILE} requires exact factorized adapter settings")
            actual = dict(
                method=module.method,
                rank=module._lora_rank,
                alpha=module._lora_alpha,
                use_rslora=module._lora_use_rslora,
                dropout=getattr(module.lora_dropout, "p", 0.0),
            )
            if any(settings.get(k) != v for k, v in actual.items()) or tuple(module.weight.shape) != tuple(
                row["shape"]
            ):
                raise ValueError(f"{PROFILE} adapter settings/shape differ from the artifact")
            if getattr(module, "_cosmos_effective_weight_cache", None) is not None:
                raise ValueError(f"{PROFILE} does not support a dense effective-weight cache")
            original = getattr(module, "_old_forward", module.forward)
            if (
                getattr(original, "__func__", None) is not adapter_type.forward
                or getattr(original, "__self__", None) is not module
            ):
                raise ValueError(f"{PROFILE} requires the original factorized forward before installation")
    elif adapter_type is not None:
        raise ValueError(f"{PROFILE} expected portable adapter modules")
    extras = (
        {row["name"].rsplit(".", 1)[0] for row in identity.get("resolved", {}).get("extra_parameters", [])}
        if targets
        else set()
    )
    if targets:
        # The existing public portable builder owns this exact collective-order
        # assembly. Pass its callable through the construction seam rather than
        # importing training code here or replacing its modality semantics.
        if (
            getattr(assembly_forward, "__module__", None) != "cosmos_framework.posttrain.model.qwen3_vl_hf"
            or getattr(assembly_forward, "__name__", None) != "_forward"
        ):
            raise ValueError(f"{PROFILE} requires the recorded portable modality assembly")
        if extras and (
            getattr(extra_norm_forward, "__module__", None) != "cosmos_framework.posttrain.model.adapter_bf16_execution"
            or getattr(extra_norm_forward, "__name__", None) != "_effective_norm"
        ):
            raise ValueError(f"{PROFILE} requires the recorded extra-normalization callable")
    elif assembly_forward is not None or extra_norm_forward is not None:
        raise ValueError(f"{PROFILE} base/merged inference requires native stock modality assembly")
    # An initially foreign norm/attention/child forward cannot acquire this
    # profile merely by retaining a compatible class and parameter shape.
    for name, module in model.named_modules():
        original = getattr(module, "_old_forward", module.forward)
        expected_forward = assembly_forward if name == "model" and targets else type(module).forward
        if name in extras:
            expected_forward = extra_norm_forward
        if (
            getattr(original, "__func__", None) is not expected_forward
            or getattr(original, "__self__", None) is not module
            or getattr(getattr(module, "_hf_hook", None), "offload", False)
        ):
            raise ValueError(f"{PROFILE} requires native callable ownership before installation: {name}")
        if hasattr(module, "lora_A") and any(
            getattr(getattr(module, key), "__func__", None) is not getattr(type(module), key)
            for key in ("adapter_output", "merged_weight")
        ):
            raise ValueError(f"{PROFILE} adapter algebra ownership changed before installation")
    replacements = []
    names = []
    parent_count = 0
    for name, module in model.named_modules():
        if not (name.startswith("model.language_model.layers.") or name == "lm_head"):
            continue
        if type(module) is torch.nn.Linear:
            function = _linear
            expected_forward = torch.nn.Linear.forward
        elif adapter_type is not None and type(module) is adapter_type:
            function = _adapter
            expected_forward = adapter_type.forward
            parent_count += 1
        else:
            continue
        attribute = "_old_forward" if hasattr(module, "_hf_hook") else "forward"
        original = getattr(module, attribute)
        if (
            getattr(original, "__func__", None) is not expected_forward
            or getattr(original, "__self__", None) is not module
            or getattr(getattr(module, "_hf_hook", None), "offload", False)
        ):
            raise ValueError(f"{PROFILE} requires original projection functions without offload: {name}")
        if module.weight.dtype != torch.float32 or not module.weight.is_contiguous():
            raise ValueError(f"{PROFILE} requires contiguous FP32 projection weights")
        if module.bias is not None and (module.bias.dtype != torch.float32 or not module.bias.is_contiguous()):
            raise ValueError(f"{PROFILE} requires contiguous FP32 biases")
        replacements.append((module, attribute, original, MethodType(function, module)))
        names.append(name)
    expected_count = model.config.text_config.num_hidden_layers * 7 + 1 + parent_count * 2
    if len(replacements) != expected_count:
        raise ValueError(f"{PROFILE} requires all native language projections and the tied output head")
    record = {
        "identity": copy.deepcopy(identity),
        "config": copy.deepcopy(model.config.to_dict()),
        "ownership": _ownership(model),
        "signature": signature(model.forward),
        "adapters": [(m, _settings(m)) for _, m in targets],
        "forwards": replacements,
        "names": names,
        "cache_owner": (PROFILE, uuid4().hex),
        "native_settings": _native_settings(model),
    }
    # Mutation starts only after the complete prevalidation above.
    for module, attribute, old, replacement in replacements:
        setattr(module, attribute, replacement)
    model._cosmos_qwen3_fixed_mma = record
    model._cosmos_deployment_profile = PROFILE
    record["executions"] = _executions(model)
    record["hooks"] = [
        model.register_forward_pre_hook(_before_forward, with_kwargs=True),
        model.register_forward_hook(_after_forward, with_kwargs=True),
    ]
    return list(names)


def restore(model):
    record = getattr(model, "_cosmos_qwen3_fixed_mma", None)
    if record is None or getattr(model, "_cosmos_deployment_profile", None) != PROFILE:
        raise ValueError(f"{PROFILE} is not installed")
    # Restore only our own forwards. Untracked-write invalidation intentionally
    # survives restoration, so a stale external prefix cannot acquire a new seal.
    if _executions(model) != record["executions"]:
        raise ValueError(f"{PROFILE} callable ownership changed; reload instead of restoring foreign functions")
    for module, attribute, old, replacement in record["forwards"]:
        setattr(module, attribute, old)
    for hook in record["hooks"]:
        hook.remove()
    del model._cosmos_qwen3_fixed_mma
    del model._cosmos_deployment_profile


def _target_device(device):
    if (
        device.type != "cuda"
        or torch.cuda.get_device_capability(device) != (8, 0)
        or "A100" not in torch.cuda.get_device_name(device)
    ):
        raise ValueError(f"{PROFILE} requires a single rank-local A100 SM80")


def merged_source_identity(path, *, verify_tensors=False):
    """Bind this profile to the complete export and unchanged processor bytes.

    The materializer records per-tensor digests. Verify those before allocation,
    in bounded CPU row chunks, and capture whole-file hashes for the post-load
    immutability check. This does not import a checkpoint or training module.
    """
    from safetensors import safe_open

    from cosmos_framework.utils.hf_snapshot import snapshot_identity

    root = Path(path)
    marker_bytes = (root / ".materialized").read_bytes()
    marker = json.loads(marker_bytes)
    export = marker["identity"]
    identity = export["export_metadata"]["adapter_identity"]
    contract.validate_identity(identity)
    if (
        marker.get("protocol") != "posttrain.materialized-hf.v1"
        or marker.get("identity_digest") != hashlib.sha256(json.dumps(export, sort_keys=True).encode()).hexdigest()
        or marker.get("options", {}).get("no_verify") is not False
    ):
        raise ValueError(f"{PROFILE} requires an unchanged verified merged-export commitment")
    snapshot = snapshot_identity(str(root), weights=True)
    processor = snapshot_identity(str(root), weights=False)["files"]
    if {k: v for k, v in processor.items() if k != "config.json"} != {
        k: v for k, v in identity["processor"]["files"].items() if k != "config.json"
    }:
        raise ValueError(f"{PROFILE} merged processor differs from its immutable source")
    index = json.loads((root / "model.safetensors.index.json").read_bytes())["weight_map"]
    inventory = marker.get("shards", [])
    named = {row["name"]: row for row in inventory}
    digests = marker.get("verify", {}).get("tensor_sha256", {})
    if (
        not index
        or len(named) != len(inventory)
        or set(named) != set(index.values())
        or set(named) != {p.name for p in root.glob("*.safetensors")}
        or set(index) != set(digests)
        or marker["verify"].get("bitwise_probe") != "all_tensors"
        or marker["verify"].get("n_tensors") != len(index)
        or marker["verify"].get("bitwise_equal_tensors") != len(index)
        or marker["verify"].get("n_shards") != len(named)
        or marker["verify"].get("bytes") != sum(row["bytes"] for row in inventory)
    ):
        raise ValueError(f"{PROFILE} requires the complete verified merged tensor inventory")
    for name, row in named.items():
        if Path(name).name != name or (root / name).stat().st_size != row["bytes"]:
            raise ValueError(f"{PROFILE} merged shard inventory changed")
        with safe_open(root / name, framework="pt", device="cpu") as reader:
            keys = {key for key, shard in index.items() if shard == name}
            if set(reader.keys()) != keys or len(keys) != row["n_tensors"]:
                raise ValueError(f"{PROFILE} merged shard tensor coverage changed")
            if verify_tensors:
                for key in sorted(keys):
                    view = reader.get_slice(key)
                    shape = view.get_shape()
                    if view.get_dtype() != "F32":
                        raise ValueError(f"{PROFILE} merged tensor is not FP32: {key}")
                    digest = hashlib.sha256()
                    if not shape:
                        digest.update(reader.get_tensor(key).reshape(-1).view(torch.uint8).numpy())
                    else:
                        rows = max(1, (16 * 1024 * 1024) // max(1, 4 * math.prod(shape[1:])))
                        for start in range(0, shape[0], rows):
                            digest.update(view[start : start + rows].contiguous().view(torch.uint8).numpy())
                    if digest.hexdigest() != digests[key]:
                        raise ValueError(f"{PROFILE} merged tensor differs from its committed digest: {key}")
    return {"snapshot": snapshot, "materialized_sha256": hashlib.sha256(marker_bytes).hexdigest()}


def load_base_model(base_path, *, dtype, device, attn_implementation=None):
    """Public unadapted control; no fabricated adapter identity or training import."""
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor, Qwen3VLProcessor

    from cosmos_framework.utils.hf_snapshot import snapshot_identity

    if dtype not in ("auto", "float32", "fp32") or attn_implementation not in (None, "sdpa"):
        raise ValueError(f"{PROFILE} requires dtype='float32' and native SDPA")
    if torch.is_inference_mode_enabled():
        raise ValueError("construct the base runtime outside inference_mode")
    device = torch.device(device)
    _target_device(device)
    sources = {
        "base": snapshot_identity(base_path, weights=True),
        "processor": snapshot_identity(base_path, weights=False),
    }
    identity = {
        "format": contract.BASE_FORMAT,
        "architecture": "qwen3_vl",
        "base_storage_dtype": "bfloat16",
        **sources,
        "dependencies": {name: version(name).split("+")[0] for name in contract.DEPENDENCIES},
        "execution_profile": PROFILE,
        "dtype": "float32",
    }
    contract.validate_base_identity(identity)
    if any((Path(base_path) / name).exists() for name in ("adapter_manifest.json", ".materialized")):
        raise ValueError("base_execution_profile requires an unadapted snapshot, not an adapter/export")
    config = AutoConfig.from_pretrained(base_path, local_files_only=True, trust_remote_code=False)
    contract.validate_configuration(config)
    if (
        getattr(config, "cosmos_inference_profile", None) is not None
        or getattr(config, "cosmos_model_execution", None) is not None
    ):
        raise ValueError("base_execution_profile cannot reinterpret a tagged export")
    processor = AutoProcessor.from_pretrained(base_path, local_files_only=True, trust_remote_code=False)
    if type(processor) is not Qwen3VLProcessor:
        raise ValueError("fixed Qwen3-VL projections require the recorded processor")
    net, info = AutoModelForImageTextToText.from_pretrained(
        base_path,
        config=config,
        dtype=torch.bfloat16,
        device_map={"": str(device)},
        attn_implementation="sdpa",
        local_files_only=True,
        trust_remote_code=False,
        output_loading_info=True,
    )
    problems = {k: info[k] for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs") if info.get(k)}
    if problems:
        raise ValueError(f"incomplete unadapted base; no weights may be initialized: {problems}")
    net.float().eval().requires_grad_(False)
    if sources != {
        "base": snapshot_identity(base_path, weights=True),
        "processor": snapshot_identity(base_path, weights=False),
    }:
        raise ValueError("base/processor changed during construction")
    install(net, identity=identity)
    processor.tokenizer.padding_side = "left"
    return (
        net,
        processor,
        {
            "kind": "unadapted_model_deployment",
            "execution_profile": PROFILE,
            "identity": copy.deepcopy(identity),
            "dtype": "float32",
            "base_path": str(base_path),
            "processor_path": str(base_path),
            "parameter_policy": "all base parameters frozen; no adapters; eval/no_grad",
            "generation_config": net.generation_config.to_dict(),
            "precision_contract": "BF16 stored values promoted to FP32; fixed FP64 language projections with separate original FP32 boundaries; native norms/SDPA/vision unchanged",
            "backend": "Transformers native SDPA, single A100 SM80, eager batch one; experimental profile requires qualification",
        },
    )
