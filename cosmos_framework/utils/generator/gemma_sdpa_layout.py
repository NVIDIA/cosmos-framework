# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Experimental, versioned Gemma1 FP32 inference with contiguous masked-SDPA K/V.

Transformers 5.8.1 repeats Gemma's single KV head into an expanded, zero-head-
stride view when a mask disables native GQA. The recorded Torch 2.10 SM80
environment has a layout-sensitive SDPA error for that view. Materialize only
those repeated attention operands. Keep native unmasked GQA, projections, RoPE,
cache storage, masks, scales and FP32 reduction behavior. This does not promise
shape-independent arithmetic or turn legacy cache/merge failures into passes.

No parameter, buffer, adapter branch, or serialized tensor is replaced. This
profile is inference-only and must be explicitly installed by a public loader.

The separately named fixed-MMA candidate reuses the attention and ownership
contracts here, with explicit projection/RMSNorm operators from gemma_fixed_mma.
The original contiguous-K/V profile retains its native projection arithmetic.
The separately versioned SDPA64 candidate retains fixed-MMA projections and
FP32 persistent caches, but reduces attention in FP64 before rounding its
output to FP32. Its attention temporaries and latency require qualification;
neither prior profile acquires these execution semantics.
"""

import copy
from importlib.metadata import version
from inspect import signature
from types import MethodType, SimpleNamespace
from uuid import uuid4

import torch

PROFILE = "gemma_fp32_sdpa_contiguous_kv_v1"
FIXED_MMA_PROFILE = "gemma_fp32_fixed_mma_v1"
SDPA64_PROFILE = "gemma_fp32_fixed_mma_sdpa64_v1"
PROFILES = (PROFILE, FIXED_MMA_PROFILE, SDPA64_PROFILE)
BASE_FORMAT = "cosmos-unadapted-gemma-deployment-v1"
BASE_SHA256 = "d69463aaf679e21e544bbd1b77095232ae5b420177f11a4c53a727fc246b5922"
PROCESSOR_SHA256 = "4931454373d18ead3f0cc4de9953b83eaafc7a47059c623c651233ec572042fa"
EXPORT_EQUIVALENCE = "versioned_contiguous_kv_sdpa_inference_requires_qualification"
BUFFER_CONTRACT = {
    "nonpersistent_buffer_policy": "gemma-fp32-embedding-scale-v1",
    "buffers": {"model.embed_tokens.embed_scale": {"dtype": "float32", "value": 45.25483322143555}},
}
ROPE_PARAMETERS = {"rope_theta": 10000.0, "rope_type": "default"}
CACHE_CONFIGURATION = {
    "sliding_window": None,
    "attention_chunk_size": None,
    "layer_types": None,
    "num_kv_shared_layers": None,
}
ARCHITECTURE = {
    "model_type": "gemma",
    "hidden_size": 2048,
    "intermediate_size": 16384,
    "num_hidden_layers": 18,
    "num_attention_heads": 8,
    "num_key_value_heads": 1,
    "head_dim": 256,
    "vocab_size": 256000,
    "hidden_act": "gelu",
    "rms_norm_eps": 1e-6,
    "tie_word_embeddings": True,
    "attention_bias": False,
    "attention_dropout": 0.0,
    "max_position_embeddings": 8192,
}


def validate_configuration(config):
    """Require the fixed base's positional and ordinary-cache semantics."""
    if isinstance(config, dict):
        from transformers import GemmaConfig

        # Resolve legacy rope_theta/rope_scaling fields through the same pinned
        # constructor as the stock loader, without mutating serialized input.
        config = GemmaConfig.from_dict(copy.deepcopy(config))
    if getattr(config, "rope_parameters", None) != ROPE_PARAMETERS:
        raise ValueError(f"{PROFILE} requires the recorded default RoPE parameters {ROPE_PARAMETERS}")
    if any(getattr(config, field, None) != expected for field, expected in CACHE_CONFIGURATION.items()):
        raise ValueError(f"{PROFILE} requires the recorded ordinary full-attention cache configuration")


def export_equivalence(profile):
    if profile == PROFILE:
        return EXPORT_EQUIVALENCE
    if profile == FIXED_MMA_PROFILE:
        return "versioned_fixed_mma_gemma_inference_requires_qualification"
    if profile == SDPA64_PROFILE:
        return "versioned_fixed_mma_sdpa64_gemma_inference_requires_qualification"
    raise ValueError(f"unknown Gemma inference profile: {profile!r}")


def validate_identity(identity, *, profile=PROFILE):
    """Pin this candidate to the recorded Gemma1 base and factorized contract."""
    export_equivalence(profile)
    resolved = identity.get("resolved", {})
    targets = resolved.get("targets", [])
    if (
        identity.get("format") != "cosmos-posttrain-adapter-v1"
        or identity.get("architecture") != "gemma"
        or identity.get("base_storage_dtype") != "bfloat16"
        or identity.get("base", {}).get("sha256") != BASE_SHA256
        or identity.get("processor", {}).get("sha256") != PROCESSOR_SHA256
        or identity.get("model_contract") != BUFFER_CONTRACT
        or resolved.get("model_execution", "default_v1") != "default_v1"
        or not targets
        or any(
            row.get("settings", {}).get("execution", "factorized_v1") != "factorized_v1"
            or row.get("settings", {}).get("method") not in ("lora", "dora", "pissa")
            for row in targets
        )
        or identity.get("dependencies", {}).get("torch") != "2.10.0"
        or identity.get("dependencies", {}).get("transformers") != "5.8.1"
    ):
        raise ValueError(f"{profile} requires the recorded Gemma1-2B base/processor and factorized adapter identity")


def validate_base_identity(identity, *, profile):
    """Admit the unadapted control separately; never manufacture adapter targets."""
    if (
        profile != SDPA64_PROFILE
        or set(identity)
        != {
            "format",
            "architecture",
            "base_storage_dtype",
            "base",
            "processor",
            "model_contract",
            "dependencies",
            "execution_profile",
            "dtype",
        }
        or identity.get("format") != BASE_FORMAT
        or identity.get("architecture") != "gemma"
        or identity.get("execution_profile") != profile
        or identity.get("dtype") != "float32"
        or identity.get("base_storage_dtype") != "bfloat16"
        or identity.get("base", {}).get("sha256") != BASE_SHA256
        or identity.get("processor", {}).get("sha256") != PROCESSOR_SHA256
        or identity.get("model_contract") != BUFFER_CONTRACT
        or identity.get("dependencies") != {"torch": "2.10.0", "transformers": "5.8.1", "triton": "3.6.0"}
    ):
        raise ValueError(f"{profile} requires the exact unadapted Gemma1 base/processor and FP32 deployment identity")


def _guard(module):
    if (
        module.training
        or torch.is_grad_enabled()
        or torch.is_autocast_enabled("cuda")
        or torch.is_autocast_enabled("cpu")
        or torch.compiler.is_compiling()
        or torch.jit.is_tracing()
        or torch.cuda.is_current_stream_capturing()
    ):
        raise RuntimeError(f"{PROFILE} requires eager eval and no_grad/inference_mode, without autocast or graphs")
    if torch.backends.cuda.matmul.allow_tf32:
        raise RuntimeError(f"{PROFILE} requires float32_matmul_precision('highest')")


def attention_interface(module, query, key, value, attention_mask, **kwargs):
    """Reuse the pinned native SDPA wrapper; only materialize its repeated K/V."""
    from transformers.integrations.sdpa_attention import repeat_kv, sdpa_attention_forward, use_gqa_in_sdpa

    if not use_gqa_in_sdpa(attention_mask, key):
        key = repeat_kv(key, module.num_key_value_groups).contiguous()
        value = repeat_kv(value, module.num_key_value_groups).contiguous()
        # The native wrapper must not repeat the already expanded heads again.
        # It consults only num_key_value_groups and is_causal on this argument;
        # omission of groups retains every other native SDPA choice unchanged.
        module = SimpleNamespace(is_causal=module.is_causal)
    return sdpa_attention_forward(module, query, key, value, attention_mask, **kwargs)


def _attention(module, hidden_states, position_embeddings=None, attention_mask=None, past_key_values=None, **kwargs):
    # Preserve pinned GemmaAttention.forward's projection/RoPE/cache order.
    # Avoid any process-wide attention-registry or torch functional mutation.
    from transformers.models.gemma.modeling_gemma import apply_rotary_pos_emb

    _guard(module)
    if kwargs.get("output_attentions"):
        raise ValueError(f"{PROFILE} does not expose attention probabilities")
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, module.head_dim)
    query = module.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key = module.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value = module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    query, key = apply_rotary_pos_emb(query, key, *position_embeddings)
    if past_key_values is not None:
        key, value = past_key_values.update(key, value, module.layer_idx)
    output, weights = attention_interface(
        module, query, key, value, attention_mask, dropout=0.0, scaling=module.scaling, **kwargs
    )
    output = output.reshape(*input_shape, -1).contiguous()
    return module.o_proj(output), weights


def _attention_fp64(
    module, hidden_states, position_embeddings=None, attention_mask=None, past_key_values=None, **kwargs
):
    """Keep native FP32 RoPE/cache order, using FP64 attention temporaries only."""
    from transformers.models.gemma.modeling_gemma import apply_rotary_pos_emb

    _guard(module)
    if kwargs.get("output_attentions"):
        raise ValueError(f"{SDPA64_PROFILE} does not expose attention probabilities")
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, module.head_dim)
    query = module.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key = module.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value = module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    query, key = apply_rotary_pos_emb(query, key, *position_embeddings)
    if past_key_values is not None:
        key, value = past_key_values.update(key, value, module.layer_idx)
    mask = (
        attention_mask.double() if attention_mask is not None and attention_mask.is_floating_point() else attention_mask
    )
    output, weights = attention_interface(
        module, query.double(), key.double(), value.double(), mask, dropout=0.0, scaling=module.scaling, **kwargs
    )
    output = output.to(hidden_states.dtype).reshape(*input_shape, -1).contiguous()
    return module.o_proj(output), weights


def _ownership(model):
    tensors = list(model.named_parameters(remove_duplicate=False))
    tensors.extend(
        (n, t)
        for n, t in model.named_buffers()
        if n.endswith(("initial_A", "initial_B")) or n == "model.embed_tokens.embed_scale"
    )
    return tuple((n, id(t), t.data_ptr(), t._version, t.dtype, t.device, tuple(t.shape)) for n, t in tensors)


def _adapter_settings(module):
    return (
        module.method,
        module.execution,
        module._lora_rank,
        module._lora_alpha,
        module._lora_use_rslora,
        module._lora_scale,
        getattr(module.lora_dropout, "p", 0.0),
        module.forward,
    )


def _attention_settings(module):
    return (
        module.head_dim,
        module.num_key_value_groups,
        module.scaling,
        module.is_causal,
        module.attention_dropout,
        module.layer_idx,
        id(module.config),
    )


def _configuration(config):
    return {
        name: copy.deepcopy(getattr(config, name, None))
        for name in (*ARCHITECTURE, *CACHE_CONFIGURATION, "rope_parameters", "use_bidirectional_attention")
    }


def _check_wrappers(model, record):
    for name, module, hook, forward in record["wrappers"]:
        if (
            model.get_submodule(name) is not module
            or getattr(module, "_hf_hook", None) is not hook
            or module.forward != forward
            or getattr(hook, "offload", False)
        ):
            raise ValueError(f"{PROFILE} attention wrapper/hook changed: {name}; reload the immutable artifact")


def _check_owned(model, record):
    if record["identity"].get("format") == BASE_FORMAT and getattr(model, "_cosmos_unadapted_state_invalidated", False):
        raise ValueError(
            "unadapted base may have untracked writes; reload the immutable base before forward/prefix reuse"
        )
    if _ownership(model) != record["ownership"]:
        raise ValueError(f"{PROFILE} tensors changed; restore and reload before model/prefix reuse")
    if model.config._attn_implementation != "sdpa" or _configuration(model.config) != record["configuration"]:
        raise ValueError(f"{PROFILE} configuration changed; restore and reload")
    if getattr(model, "_cosmos_deployment_profile", None) != record["profile"]:
        raise ValueError(f"{PROFILE} conflicts with the active deployment profile")
    _check_wrappers(model, record)
    for module, settings in record["attention_settings"]:
        if _attention_settings(module) != settings:
            raise ValueError(f"{PROFILE} attention settings changed; restore and reload")
    for name, module, attribute, old, replacement in record["forwards"]:
        if model.get_submodule(name) is not module or getattr(module, attribute) != replacement:
            raise ValueError(f"{PROFILE} module/forward changed: {name}")
    for module, settings in record["adapters"]:
        if _adapter_settings(module) != settings:
            raise ValueError(f"{PROFILE} adapter settings changed; restore and reload")
    for module, epsilon in record["norm_settings"]:
        if module.eps != epsilon:
            raise ValueError(f"{record['profile']} normalization settings changed; restore and reload")


def _before_forward(model, args, kwargs):
    from transformers.cache_utils import DynamicCache, DynamicLayer

    _guard(model)
    record = model._cosmos_gemma_sdpa_layout
    _check_owned(model, record)
    call = record["signature"].bind_partial(*args, **kwargs).arguments
    ids, embeds = call.get("input_ids"), call.get("inputs_embeds")
    inputs = ids if ids is not None else embeds
    if inputs is None or inputs.shape[0] != 1:
        raise ValueError(f"{PROFILE} requires batch-one inputs")
    if embeds is not None and embeds.dtype != torch.float32:
        raise ValueError(f"{PROFILE} requires FP32 embeddings")
    cache = call.get("past_key_values")
    if cache is not None and (
        type(cache) is not DynamicCache
        or getattr(cache, "offloading", False)
        or cache.layer_class_to_replicate not in (None, DynamicLayer)
        or any(type(layer) is not DynamicLayer for layer in cache.layers)
        or len(cache.layers) > model.config.num_hidden_layers
        or (cache.layer_class_to_replicate is None and len(cache.layers) != model.config.num_hidden_layers)
    ):
        raise ValueError(f"{PROFILE} supports ordinary full-attention DynamicCache layers without offload")
    if (
        cache is not None
        and cache.get_seq_length()
        and getattr(cache, "_cosmos_gemma_kv_owner", None) != record["cache_owner"]
    ):
        raise ValueError(f"{PROFILE} requires a prefix from this installation; rebuild the prefix")
    if inputs.shape[1] + (cache.get_seq_length() if cache is not None else 0) > 8224:
        raise ValueError(f"{PROFILE} is bounded to 8192 prefix tokens plus 32 decoding tokens")


def _after_forward(model, args, kwargs, output):
    from transformers.cache_utils import DynamicCache

    record = model._cosmos_gemma_sdpa_layout
    call = record["signature"].bind_partial(*args, **kwargs).arguments
    caches = [call.get("past_key_values"), getattr(output, "past_key_values", None)]
    if isinstance(output, tuple):
        caches.extend(value for value in output if isinstance(value, DynamicCache))
    for cache in caches:
        if cache is not None:
            cache._cosmos_gemma_kv_owner = record["cache_owner"]


def _validate_device(model):
    parameters = list(model.parameters())
    devices = {p.device for p in parameters}
    if (
        len(devices) != 1
        or next(iter(devices)).type != "cuda"
        or torch.cuda.get_device_capability(next(iter(devices))) != (8, 0)
        or any(p.dtype != torch.float32 or p.is_inference() for p in parameters)
    ):
        raise ValueError(f"{PROFILE} requires FP32 parameters on one A100, constructed outside inference_mode")


def install(model, *, identity, adapter_type=None, profile=PROFILE):
    """Prevalidate all placement, then install only the selected inference operators."""
    from transformers.models.gemma.modeling_gemma import GemmaAttention, GemmaForCausalLM, GemmaRMSNorm

    unadapted = identity.get("format") == BASE_FORMAT
    if unadapted:
        validate_base_identity(identity, profile=profile)
        if getattr(model, "_cosmos_unadapted_state_invalidated", False):
            raise ValueError("unadapted base may have untracked writes; reload before profile installation")
        if adapter_type is not None or any(hasattr(m, "lora_A") for m in model.modules()):
            raise ValueError("unadapted base deployment cannot contain adapters")
    else:
        validate_identity(identity, profile=profile)
    fixed = profile in (FIXED_MMA_PROFILE, SDPA64_PROFILE)
    if fixed and version("triton") != "3.6.0":
        raise ValueError(f"{profile} requires Triton 3.6.0")
    validate_configuration(model.config)
    if (
        type(model) is not GemmaForCausalLM
        or model.training
        or model.config._attn_implementation != "sdpa"
        or version("torch").split("+")[0] != "2.10.0"
        or version("transformers") != "5.8.1"
        or any(getattr(model.config, k, None) != v for k, v in ARCHITECTURE.items())
        or getattr(model.config, "use_bidirectional_attention", False)
    ):
        raise ValueError(f"{PROFILE} requires the pinned stock Gemma1-2B eval architecture, SDPA and dependencies")
    if model.get_input_embeddings().weight is not model.get_output_embeddings().weight:
        raise ValueError(f"{PROFILE} requires the original tied embedding/output weight")
    embedding = model.model.embed_tokens
    if (
        "embed_scale" not in embedding._non_persistent_buffers_set
        or embedding.embed_scale.dtype != torch.float32
        or embedding.embed_scale.item() != BUFFER_CONTRACT["buffers"]["model.embed_tokens.embed_scale"]["value"]
    ):
        raise ValueError(f"{PROFILE} requires the artifact's canonical FP32 embedding-scale buffer")
    installed = getattr(model, "_cosmos_deployment_profile", None)
    if installed not in (None, profile):
        raise ValueError("conflicting deployment profile; restore before installation")
    _validate_device(model)
    if installed:
        record = model._cosmos_gemma_sdpa_layout
        if record["identity"] != identity:
            raise ValueError("installed profile has a different adapter identity")
        _check_owned(model, record)
        return [row[0] for row in record["forwards"]]
    targets = [(n, m) for n, m in model.named_modules(remove_duplicate=False) if hasattr(m, "lora_A")]
    if targets:
        if adapter_type is None or {n for n, _ in targets} != {r["name"] for r in identity["resolved"]["targets"]}:
            raise ValueError("portable placement differs from resolved adapter targets")
        if any(type(m) is not adapter_type or m.execution != "factorized_v1" for _, m in targets):
            raise ValueError(f"{PROFILE} requires native factorized adapters")
        if any(getattr(m, "_cosmos_effective_weight_cache", None) is not None for _, m in targets):
            raise ValueError(f"{PROFILE} preserves separate branches; effective-weight caches are unsupported")
        if fixed:
            declared = {row["name"]: row["settings"] for row in identity["resolved"]["targets"]}
            for name, module in targets:
                settings = declared[name]
                actual = {
                    "method": module.method,
                    "execution": module.execution,
                    "rank": module._lora_rank,
                    "alpha": module._lora_alpha,
                    "use_rslora": module._lora_use_rslora,
                    "dropout": getattr(module.lora_dropout, "p", 0.0),
                }
                if module.method not in ("lora", "dora", "pissa") or any(
                    settings[key] != value for key, value in actual.items() if key in settings
                ):
                    raise ValueError(f"{profile} adapter settings differ from the declared artifact: {name}")
    forwards = []
    for name, module in model.named_modules():
        if type(module) is GemmaAttention:
            attribute = "_old_forward" if hasattr(module, "_hf_hook") else "forward"
            original = getattr(module, attribute)
            if getattr(original, "__func__", None) is not GemmaAttention.forward or getattr(
                getattr(module, "_hf_hook", None), "offload", False
            ):
                raise ValueError(f"{PROFILE} requires unmodified stock attention without offload: {name}")
            attention_forward = _attention_fp64 if profile == SDPA64_PROFILE else _attention
            forwards.append((name, module, attribute, original, MethodType(attention_forward, module)))
    expected = {f"model.layers.{i}.self_attn" for i in range(model.config.num_hidden_layers)}
    if {row[0] for row in forwards} != expected:
        raise ValueError(f"{PROFILE} requires all native Gemma attention modules")
    attention_settings = [(row[1], _attention_settings(row[1])) for row in forwards]
    norm_settings = []
    if fixed:
        from cosmos_framework.utils.generator import gemma_fixed_mma

        # Adapter children remain separate state owners, but their enclosing
        # factorized forward explicitly calls the shared low-rank algebra.
        descendants = {id(child) for _, module in targets for child in module.modules() if child is not module}
        operators = []
        for name, module in model.named_modules():
            if id(module) in descendants:
                continue
            function = {torch.nn.Linear: gemma_fixed_mma.linear, GemmaRMSNorm: gemma_fixed_mma.norm}.get(type(module))
            if adapter_type is not None and type(module) is adapter_type:
                function = gemma_fixed_mma.adapter
            if function is None:
                continue
            attribute = "_old_forward" if hasattr(module, "_hf_hook") else "forward"
            original = getattr(module, attribute)
            if getattr(original, "__func__", None) is not type(module).forward or getattr(
                getattr(module, "_hf_hook", None), "offload", False
            ):
                raise ValueError(f"{profile} requires unmodified native operators without offload: {name}")
            if type(module) is GemmaRMSNorm:
                if module.eps != model.config.rms_norm_eps:
                    raise ValueError(f"{profile} requires the recorded Gemma RMSNorm epsilon")
                norm_settings.append((module, module.eps))
            operators.append((name, module, attribute, original, MethodType(function, module)))
        expected_operators = {"lm_head", "model.norm"}
        for index in range(model.config.num_hidden_layers):
            prefix = f"model.layers.{index}."
            expected_operators.update(prefix + "self_attn." + name for name in ("q_proj", "k_proj", "v_proj", "o_proj"))
            expected_operators.update(prefix + "mlp." + name for name in ("gate_proj", "up_proj", "down_proj"))
            expected_operators.update(prefix + name for name in ("input_layernorm", "post_attention_layernorm"))
        if {row[0] for row in operators} != expected_operators:
            raise ValueError(f"{profile} requires all native Gemma projection and normalization operators")
        forwards.extend(operators)
    record = {
        "identity": copy.deepcopy(identity),
        "profile": profile,
        "forwards": forwards,
        "attention_settings": attention_settings,
        "norm_settings": norm_settings,
        "ownership": _ownership(model),
        "configuration": _configuration(model.config),
        "signature": signature(model.forward),
        "cache_owner": (profile, uuid4().hex),
    }
    model._cosmos_gemma_sdpa_layout = record
    for _, module, attribute, _, replacement in forwards:
        setattr(module, attribute, replacement)
    record["adapters"] = [(module, _adapter_settings(module)) for _, module in targets]
    # Accelerate owns the outer forward when this profile replaces _old_forward.
    # Neither that wrapper nor its hook may change unnoticed during cache reuse.
    record["wrappers"] = [
        (name, module, getattr(module, "_hf_hook", None), module.forward) for name, module, *_ in forwards
    ]
    record["hook"] = model.register_forward_pre_hook(_before_forward, with_kwargs=True)
    record["output_hook"] = model.register_forward_hook(_after_forward, with_kwargs=True)
    model._cosmos_deployment_profile = profile
    return [row[0] for row in forwards]


def restore(model):
    record = getattr(model, "_cosmos_gemma_sdpa_layout", None)
    if (
        record is None
        or record.get("profile") not in PROFILES
        or getattr(model, "_cosmos_deployment_profile", None) != record["profile"]
    ):
        raise ValueError(f"{PROFILE} is not installed; refusing to restore another profile")
    _check_wrappers(model, record)
    if any(model.get_submodule(n) is not m or getattr(m, a) != new for n, m, a, _, new in record["forwards"]):
        raise ValueError("profile modules/forwards changed; reload the immutable artifact")
    for _, module, attribute, old, _ in record["forwards"]:
        setattr(module, attribute, old)
    record["hook"].remove()
    record["output_hook"].remove()
    del model._cosmos_gemma_sdpa_layout
    del model._cosmos_deployment_profile
