# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Opt-in SmolLM inference with FP32 storage and FP64 reduction transients.

This is a new inference oracle, not a reinterpretation of factorized training or
plain stock inference. Linear/RMSNorm/SDPA results round back to FP32 at their
original boundaries. Adapter branches remain separate; no owned tensor changes.
The cache, RoPE, mask construction and residual/activation operations stay native.
"""

from importlib.metadata import version
from inspect import signature
from types import MethodType
from uuid import uuid4

import torch
from torch.nn import functional as F

PROFILE = "smollm_fp32_consistent_v1"
BASE_SHA256 = "377bf183dd7e0f58651d83d0eff98c88f2d3ccea1a009874a805eebfa56f6e29"


def validate_identity(identity):
    resolved = identity.get("resolved", {})
    if (
        identity.get("format") != "cosmos-posttrain-adapter-v1"
        or identity.get("architecture") != "llama"
        or identity.get("base_storage_dtype") != "bfloat16"
        or identity.get("base", {}).get("sha256") != BASE_SHA256
        or resolved.get("model_execution", "default_v1") != "default_v1"
        or not resolved.get("targets")
        or any(
            row["settings"].get("execution", "factorized_v1") != "factorized_v1"
            or row["settings"]["method"] not in ("lora", "dora", "pissa")
            for row in resolved["targets"]
        )
        or identity.get("dependencies", {}).get("torch") != "2.10.0"
        or identity.get("dependencies", {}).get("transformers") != "5.8.1"
    ):
        raise ValueError(f"{PROFILE} requires the recorded SmolLM2-135M v1 factorized adapter/base identity")


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
        raise RuntimeError(
            f"{PROFILE} requires eager eval with no_grad/inference_mode; no training, autocast or graphs"
        )
    if torch.backends.cuda.matmul.allow_tf32:
        raise RuntimeError(f"{PROFILE} requires float32_matmul_precision('highest')")


def _linear(module, inputs):
    _guard(module)
    return F.linear(inputs.double(), module.weight.double(), None if module.bias is None else module.bias.double()).to(
        inputs.dtype
    )


def _adapter(module, inputs):
    _guard(module)
    # Dropout is disabled by eval, including independently configured dropout.
    x, a, b = inputs.double(), module.lora_A.weight.double(), module.lora_B.weight.double()
    base = F.linear(x, module.weight.double(), None if module.bias is None else module.bias.double())
    delta = F.linear(F.linear(x, a), b) * module._lora_scale
    if module.method == "pissa":
        initial = F.linear(F.linear(x, module.initial_A.double()), module.initial_B.double()) * module._lora_scale
        result = base + (delta - initial)
    elif module.method == "dora":
        direction = module.weight.double() + module._lora_scale * (b @ a)
        ratio = module.magnitude.double() / direction.norm(dim=1).clamp_min(torch.finfo(torch.float64).tiny)
        # The bias is not normalized. The dense direction only defines the row
        # norm; it is never used as a dense effective-weight forward.
        result = base + (ratio - 1) * F.linear(x, module.weight.double(), None) + ratio * delta
    elif module.method == "lora":
        result = base + delta
    else:
        raise ValueError(f"{PROFILE}: adapter method changed after installation")
    return result.to(inputs.dtype)


def _norm(module, inputs):
    _guard(module)
    x = inputs.double()
    return (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + module.variance_epsilon) * module.weight.double()).to(
        inputs.dtype
    )


def _attention(module, hidden_states, position_embeddings=None, attention_mask=None, past_key_values=None, **kwargs):
    # Preserve Transformers 5.8.1 LlamaAttention's projections, RoPE and cache
    # update exactly. Only the SDPA reduction dtype differs. Do not modify the
    # process-wide attention registry (other models may share this process).
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

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
    mask = (
        attention_mask.double() if attention_mask is not None and attention_mask.is_floating_point() else attention_mask
    )
    output, weights = sdpa_attention_forward(
        module, query.double(), key.double(), value.double(), mask, dropout=0.0, scaling=module.scaling, **kwargs
    )
    output = output.to(hidden_states.dtype).reshape(*input_shape, -1).contiguous()
    return module.o_proj(output), weights


def _ownership(model):
    tensors = list(model.named_parameters(remove_duplicate=False))
    tensors.extend((n, t) for n, t in model.named_buffers() if n.endswith(("initial_A", "initial_B")))
    return tuple((n, id(t), t.data_ptr(), t._version, t.dtype, t.device, tuple(t.shape)) for n, t in tensors)


def _before_forward(model, args, kwargs):
    from transformers.cache_utils import DynamicCache

    _guard(model)
    record = model._cosmos_smollm_consistent
    if _ownership(model) != record["ownership"]:
        raise ValueError(f"{PROFILE} tensors changed; restore and reload before reusing the model or a prefix")
    if model.config._attn_implementation != "sdpa":
        raise ValueError(f"{PROFILE} requires its recorded SDPA implementation")
    for name, module, attribute, old, replacement in record["forwards"]:
        if model.get_submodule(name) is not module or getattr(module, attribute) != replacement:
            raise ValueError(f"{PROFILE} module/forward changed: {name}; restore before mutation")
    for module, method, execution, scale, dropout in record["adapters"]:
        if (module.method, module.execution, module._lora_scale, module.lora_dropout.p) != (
            method,
            execution,
            scale,
            dropout,
        ):
            raise ValueError(f"{PROFILE} adapter settings changed; restore and reload")
    call = record["signature"].bind_partial(*args, **kwargs).arguments
    ids, embeds = call.get("input_ids"), call.get("inputs_embeds")
    inputs = ids if ids is not None else embeds
    if inputs is None or inputs.shape[0] != 1:
        raise ValueError(f"{PROFILE} requires batch-one inputs")
    if embeds is not None and embeds.dtype != torch.float32:
        raise ValueError(f"{PROFILE} requires FP32 embeddings")
    cache = call.get("past_key_values")
    if cache is not None and (type(cache) is not DynamicCache or getattr(cache, "offloading", False)):
        raise ValueError(f"{PROFILE} supports ordinary DynamicCache without offload")
    if (
        cache is not None
        and cache.get_seq_length()
        and getattr(cache, "_cosmos_smollm_owner", None) != record["cache_owner"]
    ):
        raise ValueError(f"{PROFILE} requires a prefix produced by this model/profile installation; rebuild the prefix")
    length = inputs.shape[1] + (cache.get_seq_length() if cache is not None else 0)
    if length > 8224:
        raise ValueError(f"{PROFILE} is bounded to 8192 prefix tokens plus 32 decoding tokens")


def _after_forward(model, args, kwargs, output):
    from transformers.cache_utils import DynamicCache

    record = model._cosmos_smollm_consistent
    call = record["signature"].bind_partial(*args, **kwargs).arguments
    caches = [call.get("past_key_values"), getattr(output, "past_key_values", None)]
    if isinstance(output, tuple):
        caches.extend(value for value in output if isinstance(value, DynamicCache))
    for cache in caches:
        if cache is not None:
            cache._cosmos_smollm_owner = record["cache_owner"]


def install(model, *, identity, adapter_type=None):
    """Prevalidate the complete model, then replace forwards without owning state."""
    from transformers.models.llama.modeling_llama import LlamaAttention, LlamaForCausalLM, LlamaRMSNorm

    validate_identity(identity)
    if (
        type(model) is not LlamaForCausalLM
        or model.training
        or model.config._attn_implementation != "sdpa"
        or version("torch").split("+")[0] != "2.10.0"
        or version("transformers") != "5.8.1"
    ):
        raise ValueError(f"{PROFILE} requires stock SmolLM/Llama eval with SDPA and Torch 2.10.0 / Transformers 5.8.1")
    expected_config = {
        "hidden_size": 576,
        "intermediate_size": 1536,
        "num_hidden_layers": 30,
        "num_attention_heads": 9,
        "num_key_value_heads": 3,
        "vocab_size": 49152,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings": True,
        "attention_bias": False,
        "mlp_bias": False,
        "max_position_embeddings": 8192,
    }
    if any(getattr(model.config, k, None) != v for k, v in expected_config.items()):
        raise ValueError(f"{PROFILE} requires the recorded SmolLM2-135M architecture")
    if model.get_input_embeddings().weight is not model.get_output_embeddings().weight:
        raise ValueError(f"{PROFILE} requires the original tied embedding/output weight")
    installed = getattr(model, "_cosmos_deployment_profile", None)
    if installed not in (None, PROFILE):
        raise ValueError("conflicting deployment profile; restore it before installing a different profile")
    parameters = list(model.parameters())
    devices = {p.device for p in parameters}
    if (
        len(devices) != 1
        or next(iter(devices)).type != "cuda"
        or torch.cuda.get_device_capability(next(iter(devices))) != (8, 0)
        or any(p.dtype != torch.float32 or p.is_inference() for p in parameters)
    ):
        raise ValueError(f"{PROFILE} requires FP32 parameters on one A100, constructed outside inference_mode")
    if installed:
        if model._cosmos_smollm_consistent["identity"] != identity:
            raise ValueError("installed profile has a different adapter identity")
        return [r[0] for r in model._cosmos_smollm_consistent["forwards"]]
    targets = [(n, m) for n, m in model.named_modules(remove_duplicate=False) if hasattr(m, "lora_A")]
    if targets:
        if adapter_type is None or {n for n, _ in targets} != {r["name"] for r in identity["resolved"]["targets"]}:
            raise ValueError("portable adapter placement differs from the recorded resolved targets")
        if any(type(m) is not adapter_type or m.execution != "factorized_v1" for _, m in targets):
            raise ValueError(f"{PROFILE} requires the native factorized adapter implementation")
        if any(getattr(m, "_cosmos_effective_weight_cache", None) is not None for _, m in targets):
            raise ValueError(f"{PROFILE} preserves separate branches; derived effective-weight caches are unsupported")
    descendants = {id(child) for _, module in targets for child in module.modules() if child is not module}
    forwards = []
    for name, module in model.named_modules():
        if id(module) in descendants:
            continue
        function = {torch.nn.Linear: _linear, LlamaRMSNorm: _norm, LlamaAttention: _attention}.get(type(module))
        if adapter_type is not None and type(module) is adapter_type:
            function = _adapter
        if function is not None:
            attribute = "_old_forward" if hasattr(module, "_hf_hook") else "forward"
            forwards.append((name, module, attribute, getattr(module, attribute), MethodType(function, module)))
    model._cosmos_smollm_consistent = {
        "identity": identity,
        "forwards": forwards,
        "adapters": [(m, m.method, m.execution, m._lora_scale, m.lora_dropout.p) for _, m in targets],
        "ownership": _ownership(model),
        "signature": signature(model.forward),
        "cache_owner": (PROFILE, uuid4().hex),
    }
    for _, module, attribute, _, replacement in forwards:
        setattr(module, attribute, replacement)
    model._cosmos_smollm_consistent["hook"] = model.register_forward_pre_hook(_before_forward, with_kwargs=True)
    model._cosmos_smollm_consistent["output_hook"] = model.register_forward_hook(_after_forward, with_kwargs=True)
    model._cosmos_deployment_profile = PROFILE
    return [r[0] for r in forwards]


def restore(model):
    record = getattr(model, "_cosmos_smollm_consistent", None)
    if record is None or getattr(model, "_cosmos_deployment_profile", None) != PROFILE:
        raise ValueError(f"{PROFILE} is not installed; refusing to restore another profile")
    # Fail before making a partial restoration if another owner changed a module.
    if any(model.get_submodule(n) is not m or getattr(m, a) != new for n, m, a, _, new in record["forwards"]):
        raise ValueError("profile modules/forwards changed; reload the immutable artifact")
    for _, module, attribute, old, _ in record["forwards"]:
        setattr(module, attribute, old)
    record["hook"].remove()
    record["output_hook"].remove()
    del model._cosmos_smollm_consistent
    del model._cosmos_deployment_profile
