# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Hugging Face value model and shared PPO response-token alignment."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn

from cosmos_framework.posttrain.model.parallel_dims import ParallelDims
from cosmos_framework.posttrain.model.parallelize import parallelize
from cosmos_framework.posttrain.model.specs import ShardPlan


def _base_net(model: Any) -> nn.Module:
    net = getattr(model, "net", model)
    if not isinstance(net, nn.Module):
        raise TypeError(f"model.net must be an nn.Module, got {type(net).__name__}")
    return net


def _final_hidden_state(output: Any) -> "torch.Tensor | None":
    """Return a causal-model output's final hidden state, or None when it carries none."""
    hidden_states = getattr(output, "hidden_states", None)
    if hidden_states:
        return hidden_states[-1]  # [B,L,H]
    decoder_hidden_states = getattr(output, "decoder_hidden_states", None)
    if decoder_hidden_states:
        return decoder_hidden_states[-1]  # [B,L,H]
    last = getattr(output, "last_hidden_state", None)
    return last if isinstance(last, torch.Tensor) else None


def _decoder_inputs(model_inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Drop the language-head-only knob before entering a decoder that has no head."""
    return {name: value for name, value in model_inputs.items() if name != "logits_to_keep"}


def _accepts_logits_to_keep(net: nn.Module) -> bool:
    """Whether this head exposes the transformers knob that trims the language-head projection."""
    try:
        return "logits_to_keep" in inspect.signature(net.forward).parameters
    except (TypeError, ValueError):
        return False


def model_device(model: Any) -> torch.device:
    """Return the device of a model's first parameter."""
    try:
        return next(_base_net(model).parameters()).device
    except StopIteration as error:
        raise ValueError("PPO model must have at least one parameter") from error


def forward_causal_model(model: Any, inputs: Mapping[str, Any]) -> Any:
    """Run only the model-forward phase of an HF-shaped causal model."""
    model_inputs = dict(inputs)
    model_inputs["use_cache"] = False
    callable_model = model if isinstance(model, nn.Module) else _base_net(model)
    return callable_model(**model_inputs)


def extract_causal_logits(output: Any) -> torch.Tensor:  # returns [B,L,V]
    """Extract and validate logits from an already-completed causal-model forward."""
    logits = output if isinstance(output, torch.Tensor) else getattr(output, "logits", None)
    if logits is None and isinstance(output, (tuple, list)) and output:
        logits = output[0]
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
        shape = None if not isinstance(logits, torch.Tensor) else tuple(logits.shape)
        raise ValueError(f"causal model must return logits shaped [B,L,V], got {shape}")
    return logits  # [B,L,V]


def forward_logits(model: Any, inputs: dict[str, Any]) -> torch.Tensor:  # returns [B,L,V]
    """Run an HF-shaped causal model and extract its logits."""
    return extract_causal_logits(forward_causal_model(model, inputs))  # [B,L,V]


def response_action_logits(
    logits: torch.Tensor,
    prompt_length: int,
    action_length: int,
) -> torch.Tensor:  # logits: [...,L,V], returns [...,T,V]
    """Select policy/reference positions ``P-1 ... P+T-2``."""
    if logits.ndim < 2:
        raise ValueError(f"logits must have sequence and vocabulary dimensions, got {tuple(logits.shape)}")
    if prompt_length < 1 or action_length < 1:
        raise ValueError("prompt_length and action_length must be positive")
    required_length = prompt_length + action_length - 1
    if logits.shape[-2] < required_length:
        raise ValueError(f"logit sequence length {logits.shape[-2]} is shorter than required length {required_length}")
    start = prompt_length - 1
    stop = start + action_length
    return logits[..., start:stop, :]  # [...,T,V]


def action_state_values(
    values: torch.Tensor,
    prompt_length: int,
    action_length: int,
) -> torch.Tensor:  # values: [...,L], returns [...,T+1]
    """Select critic positions ``P-1 ... P+T-1``, including bootstrap."""
    if values.ndim < 1:
        raise ValueError("values must have a sequence dimension")
    if prompt_length < 1 or action_length < 1:
        raise ValueError("prompt_length and action_length must be positive")
    required_length = prompt_length + action_length
    if values.shape[-1] < required_length:
        raise ValueError(f"value sequence length {values.shape[-1]} is shorter than required length {required_length}")
    start = prompt_length - 1
    stop = start + action_length + 1
    return values[..., start:stop]  # [...,T+1]


def sampled_action_logprobs(
    action_logits: torch.Tensor,
    action_token_ids: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:  # logits: [...,T,V], ids: [...,T], returns [...,T]
    """Gather sampled-action log-probabilities from aligned response logits."""
    if not isinstance(temperature, (int, float)) or not 0.0 < float(temperature) < float("inf"):
        raise ValueError(f"temperature must be finite and positive, got {temperature}")
    if action_logits.shape[:-1] != action_token_ids.shape:
        raise ValueError(
            f"action token shape {tuple(action_token_ids.shape)} must match logits prefix "
            f"{tuple(action_logits.shape[:-1])}"
        )
    if action_token_ids.dtype != torch.long:
        action_token_ids = action_token_ids.to(dtype=torch.long)  # [...,T]
    logprobs = torch.log_softmax(action_logits.float() / float(temperature), dim=-1)  # [...,T,V]
    return logprobs.gather(dim=-1, index=action_token_ids.unsqueeze(-1)).squeeze(-1)  # [...,T]


def _hidden_size(base: Any, net: nn.Module) -> int:
    config = getattr(base, "hf_config", None) or getattr(net, "config", None)
    text_config = getattr(config, "text_config", None) or config
    hidden_size = getattr(text_config, "hidden_size", None)
    if isinstance(hidden_size, bool) or not isinstance(hidden_size, int) or hidden_size < 1:
        raise ValueError("critic base config must expose a positive hidden_size")
    return hidden_size


def _parameter_device(net: nn.Module) -> torch.device:
    first_floating: torch.nn.Parameter | None = None
    for parameter in net.parameters():
        if parameter.device.type == "meta":
            raise ValueError(
                "DenseValueModel must be attached after base-weight loading; base contains meta parameters"
            )
        if first_floating is None and parameter.is_floating_point():
            first_floating = parameter
    if first_floating is None:
        raise ValueError("critic base must have floating-point parameters")
    return first_floating.device


class DenseValueModel(nn.Module):
    """A separately loaded causal LLM/VLM plus a scalar value head.

    Construct this wrapper only after the base model has completed its ordinary HF
    weight-loading lifecycle. The value head is then attached and zero initialized,
    so strict base checkpoint loading never sees an unexpected critic parameter.
    """

    base_model: Any
    value_head: nn.Linear
    # Latched on the first forward: whether this base hands its final hidden state to the causal
    # root or only to the decoder beneath it. See `forward`.
    _hidden_from_inner: bool

    def __init__(self, loaded_base_model: Any, *, pdims: ParallelDims | None = None) -> None:
        super().__init__()
        self._hidden_from_inner = False
        net = _base_net(loaded_base_model)
        device = _parameter_device(net)
        self.base_model = loaded_base_model
        self.value_head = nn.Linear(
            _hidden_size(loaded_base_model, net),
            1,
            bias=True,
            device=device,
            dtype=torch.float32,
        )
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)
        # The HF base has already been FSDP-wrapped and loaded. Shard the newly
        # attached head as its own FSDP unit so its gradients and optimizer state
        # participate in the critic's data-parallel mesh too.
        reduce_dtype = getattr(getattr(loaded_base_model, "shard_plan", None), "reduce_dtype", torch.float32)
        head_plan = ShardPlan(block_types=(), param_dtype=torch.float32, reduce_dtype=reduce_dtype)
        parallelize(self.value_head, head_plan, pdims)

    def forward(self, input_ids: torch.Tensor | None = None, **inputs: Any) -> torch.Tensor:  # returns [B,L]
        """Return one scalar value at every causal sequence position."""
        if input_ids is not None:
            if "input_ids" in inputs:
                raise ValueError("input_ids was provided both positionally and by keyword")
            inputs["input_ids"] = input_ids
        inputs["output_hidden_states"] = True
        inputs["return_dict"] = True
        inputs["use_cache"] = False
        # Call the FSDP ROOT, never its inner `.model`. `parallelize()` wraps the root as its own
        # FSDP unit holding the embeddings, final norm, and head, so the root's pre-forward hook is
        # what all-gathers `embed_tokens.weight`. Entering at the inner decoder skips that hook and
        # the embedding sees a still-sharded DTensor against plain `input_ids`, which raises
        # "got mixed torch.Tensor and DTensor". A replicate-only topology never calls `fully_shard`,
        # so that path stayed green while any real sharded critic died on its first forward.
        net = _base_net(self.base_model)
        config = getattr(self.base_model, "hf_config", None) or getattr(net, "config", None)
        non_model_keys = set(getattr(net, "_COLLATE_NON_MODEL_KEYS", ())) | {"logprob_masks"}
        if getattr(config, "model_type", None) in {"qwen2_vl", "qwen2_5_vl", "qwen3_vl", "qwen3_vl_moe"}:
            non_model_keys.add("image_sizes")
        model_inputs = {name: value for name, value in inputs.items() if name not in non_model_keys}
        # The critic reads hidden states only. Keeping one position of logits makes the language
        # head's [B,L,V] projection negligible instead of the dominant cost of a value forward.
        if "logits_to_keep" not in model_inputs and _accepts_logits_to_keep(net):
            model_inputs["logits_to_keep"] = 1
        inner = getattr(net, "model", None)
        if self._hidden_from_inner and isinstance(inner, nn.Module):
            hidden = _final_hidden_state(inner(**_decoder_inputs(model_inputs)))
        else:
            hidden = _final_hidden_state(net(**model_inputs))
            if hidden is None and isinstance(inner, nn.Module):
                # Some roots take ``**kwargs``, drop ``output_hidden_states``, and return logits
                # only; their hidden state exists solely inside the decoder. Entering the decoder is
                # safe ONLY while the root is not its own FSDP unit -- otherwise it skips the
                # pre-forward hook that all-gathers ``embed_tokens.weight`` and the embedding sees a
                # sharded DTensor. An HF root honors the flag and never reaches this branch.
                if hasattr(net, "unshard"):
                    raise ValueError(
                        f"critic base {type(net).__name__} returns no hidden state and is FSDP-sharded, "
                        "so its decoder cannot be entered directly; the base must return its final "
                        "hidden state (e.g. populate `hidden_states` on its output)"
                    )
                self._hidden_from_inner = True
                hidden = _final_hidden_state(inner(**_decoder_inputs(model_inputs)))
        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
            shape = None if not isinstance(hidden, torch.Tensor) else tuple(hidden.shape)
            raise ValueError(f"critic base must return a final hidden state shaped [B,L,H], got {shape}")
        hidden_float = hidden.float()  # [B,L,H]
        values = self.value_head(hidden_float).squeeze(-1)  # [B,L]
        return values


__all__ = [
    "DenseValueModel",
    "model_device",
    "forward_logits",
    "response_action_logits",
    "action_state_values",
    "sampled_action_logprobs",
]
