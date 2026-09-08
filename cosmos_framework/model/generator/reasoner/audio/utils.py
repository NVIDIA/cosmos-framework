# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Backend-neutral placeholder validation and raw-HF forward integration for audio tokens."""

import inspect
from functools import wraps
from numbers import Integral
from typing import Any

import torch


def _validate_audio_merge_inputs(
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    audio_embeddings: torch.Tensor,
    audio_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    token_mask = input_ids == int(audio_token_id)
    flattened_audio_embeddings = audio_embeddings.flatten(0, -2)
    num_placeholders = int(token_mask.sum().item())
    if num_placeholders != flattened_audio_embeddings.shape[0]:
        raise ValueError(
            "audio features and audio placeholder tokens do not match: "
            f"tokens={num_placeholders}, features={flattened_audio_embeddings.shape[0]}"
        )

    expanded_mask = token_mask.unsqueeze(-1).expand_as(inputs_embeds)
    return expanded_mask, flattened_audio_embeddings


def merge_audio_embeddings(
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    audio_embeddings: torch.Tensor,
    audio_token_id: int,
) -> torch.Tensor:
    """Return token embeddings with audio placeholders replaced in row-major order."""
    expanded_mask, flattened_audio_embeddings = _validate_audio_merge_inputs(
        input_ids,
        inputs_embeds,
        audio_embeddings,
        audio_token_id,
    )
    return inputs_embeds.masked_scatter(expanded_mask, flattened_audio_embeddings)


def _restrict_token_embedding_gradients(
    input_ids: torch.Tensor,
    token_embeddings: torch.Tensor,
    trainable_token_ids: tuple[int, ...],
) -> torch.Tensor:  # input_ids: [B,N], token_embeddings: [B,N,D], returns: [B,N,D]
    """Preserve embedding values while limiting gradients to selected token positions."""
    if not torch.is_grad_enabled() or not token_embeddings.requires_grad:
        return token_embeddings

    trainable_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)  # [B,N]
    for token_id in trainable_token_ids:
        trainable_token_mask = trainable_token_mask | (input_ids == token_id)  # [B,N]
    gradient_mask = trainable_token_mask.unsqueeze(-1)  # [B,N,1]
    return torch.where(gradient_mask, token_embeddings, token_embeddings.detach())  # [B,N,D]


def merge_projected_audio_embeddings(
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    projected_audio_embeddings: torch.Tensor,
    audio_embedding_lengths: torch.Tensor,
    audio_sample_token_lengths: torch.Tensor,
    audio_token_id: int,
) -> torch.Tensor:
    """Flatten valid projected clips and merge them into independent text rows.

    ``projected_audio_embeddings`` and ``audio_embedding_lengths`` are ordered
    clip row-major: all clips belonging to text row 0, then row 1, and so on.
    ``audio_sample_token_lengths`` records the total valid audio tokens assigned
    to every text row, including zeros for rows without audio. The per-row check
    prevents a globally balanced batch from silently moving audio features
    across sample boundaries.
    """
    num_clips, padded_audio_time, _ = projected_audio_embeddings.shape
    batch_size = input_ids.shape[0]
    expected_length_shapes = {
        "audio_embedding_lengths": (num_clips,),
        "audio_sample_token_lengths": (batch_size,),
    }
    for name, lengths in (
        ("audio_embedding_lengths", audio_embedding_lengths),
        ("audio_sample_token_lengths", audio_sample_token_lengths),
    ):
        if lengths.shape != expected_length_shapes[name]:
            raise ValueError(f"{name} must have shape {expected_length_shapes[name]}, got {tuple(lengths.shape)}")
    if torch.any(audio_embedding_lengths <= 0):
        raise ValueError("audio_embedding_lengths values must be positive")
    if torch.any(audio_embedding_lengths > padded_audio_time):
        raise ValueError("audio_embedding_lengths cannot exceed the padded audio time dimension")
    if torch.any(audio_sample_token_lengths < 0):
        raise ValueError("audio_sample_token_lengths values must be non-negative")

    total_clip_tokens = int(audio_embedding_lengths.sum().item())
    total_sample_tokens = int(audio_sample_token_lengths.sum().item())
    if total_clip_tokens != total_sample_tokens:
        raise ValueError(
            "audio_embedding_lengths and audio_sample_token_lengths must have the same total: "
            f"{total_clip_tokens} != {total_sample_tokens}"
        )

    actual_sample_tokens = (input_ids == int(audio_token_id)).sum(dim=1)
    expected_sample_tokens = audio_sample_token_lengths.to(dtype=actual_sample_tokens.dtype)
    if not torch.equal(actual_sample_tokens, expected_sample_tokens):
        raise ValueError(
            "audio placeholder count must match audio_sample_token_lengths for every text row: "
            f"placeholders={actual_sample_tokens.tolist()}, "
            f"audio_sample_token_lengths={audio_sample_token_lengths.tolist()}"
        )

    valid_audio_mask = torch.arange(padded_audio_time, device=projected_audio_embeddings.device).unsqueeze(0) < (
        audio_embedding_lengths.unsqueeze(1)
    )
    flattened_audio_embeddings = projected_audio_embeddings[valid_audio_mask]

    return merge_audio_embeddings(
        input_ids,
        inputs_embeds,
        flattened_audio_embeddings,
        int(audio_token_id),
    )


_AUDIO_FORWARD_ARGUMENTS = (
    "audio_features",
    "audio_feature_lengths",
    "audio_token_lengths",
    "audio_sample_token_lengths",
)
_SUPPORTED_AUDIO_REASONER_TYPES = frozenset({"nemotron_siglip2", "qwen3_vl"})
_PATCH_MARKER = "_cosmos_reasoner_audio_forward_patch"
_TIED_OUTPUT_PATCH_MARKER = "_cosmos_reasoner_boundary_only_output_patch"


def _patch_tied_output_embedding_forward(model: torch.nn.Module) -> None:
    """Prevent the tied output projection from updating the shared embedding table."""
    get_output_embeddings = getattr(model, "get_output_embeddings", None)
    output_embeddings = get_output_embeddings() if callable(get_output_embeddings) else None
    if output_embeddings is None:
        output_embeddings = getattr(model, "lm_head", None)
    if not isinstance(output_embeddings, torch.nn.Linear):
        raise ValueError(
            "Boundary-token-only training with tied embeddings requires a torch.nn.Linear output embedding module"
        )
    if getattr(output_embeddings, _TIED_OUTPUT_PATCH_MARKER, False):
        return

    def detached_weight_forward(self: torch.nn.Linear, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(hidden_states, self.weight.detach(), self.bias)  # [...,V]

    output_embeddings.forward = detached_weight_forward.__get__(output_embeddings, type(output_embeddings))
    setattr(output_embeddings, _TIED_OUTPUT_PATCH_MARKER, True)


def _get_bound_argument(
    bound_arguments: inspect.BoundArguments,
    signature: inspect.Signature,
    name: str,
) -> Any:
    if name in bound_arguments.arguments:
        return bound_arguments.arguments[name]
    for parameter_name, parameter in signature.parameters.items():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            return bound_arguments.arguments.get(parameter_name, {}).get(name)
    return None


def _set_bound_argument(
    bound_arguments: inspect.BoundArguments,
    signature: inspect.Signature,
    name: str,
    value: Any,
) -> None:
    if name in signature.parameters:
        bound_arguments.arguments[name] = value
        return
    for parameter_name, parameter in signature.parameters.items():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            extra_kwargs = bound_arguments.arguments.setdefault(parameter_name, {})
            extra_kwargs[name] = value
            return
    raise TypeError(f"The wrapped reasoner forward does not accept {name!r}")


def _compute_native_position_ids(
    model: torch.nn.Module,
    model_type: str,
    input_ids: torch.Tensor,
    bound_arguments: inspect.BoundArguments,
    signature: inspect.Signature,
) -> torch.Tensor:
    """Compute native multimodal positions while original token IDs still exist."""
    inner_model = getattr(model, "model", None)
    if inner_model is None or not callable(getattr(inner_model, "get_rope_index", None)):
        raise RuntimeError(f"Audio understanding requires {type(model).__name__}.model.get_rope_index for {model_type}")

    rope_kwargs = {
        "input_ids": input_ids,
        "image_grid_thw": _get_bound_argument(bound_arguments, signature, "image_grid_thw"),
        "video_grid_thw": _get_bound_argument(bound_arguments, signature, "video_grid_thw"),
        "attention_mask": _get_bound_argument(bound_arguments, signature, "attention_mask"),
    }
    if model_type == "nemotron_siglip2":
        cu_seqlens = _get_bound_argument(bound_arguments, signature, "cu_seqlens")
        if cu_seqlens is not None:
            rope_kwargs["cu_seqlens"] = cu_seqlens

    position_ids, rope_deltas = inner_model.get_rope_index(**rope_kwargs)
    inner_model.rope_deltas = rope_deltas
    return position_ids


def _projector_dummy_zero(model: torch.nn.Module) -> torch.Tensor | None:
    """Return a zero connected only to a trainable projector, never the encoder."""
    projector = model.sound_und_model.projector
    trainable_parameters = [parameter for parameter in projector.parameters() if parameter.requires_grad]
    if not torch.is_grad_enabled() or not trainable_parameters:
        return None

    input_hidden_size = getattr(projector, "input_hidden_size", None)
    if isinstance(input_hidden_size, bool) or not isinstance(input_hidden_size, Integral) or input_hidden_size <= 0:
        raise RuntimeError("The audio projector must expose a positive integer input_hidden_size")
    reference = trainable_parameters[0]
    dummy_input = torch.zeros(
        1,
        int(input_hidden_size),
        device=reference.device,
        dtype=reference.dtype,
    )
    return model.sound_und_model.projector(dummy_input).sum() * 0


def _connect_projector_zero_to_logits(output: Any, projector_zero: torch.Tensor, labels: Any) -> Any:
    """Connect a no-op projector graph to logits for mixed-modality FSDP ranks."""

    def connected(logits: torch.Tensor) -> torch.Tensor:
        return logits + projector_zero.to(device=logits.device, dtype=logits.dtype)

    logits = getattr(output, "logits", None)
    if isinstance(logits, torch.Tensor):
        output.logits = connected(logits)
        return output
    if isinstance(output, dict) and isinstance(output.get("logits"), torch.Tensor):
        output["logits"] = connected(output["logits"])
        return output
    if isinstance(output, tuple):
        logits_index = 1 if labels is not None else 0
        if len(output) <= logits_index or not isinstance(output[logits_index], torch.Tensor):
            raise RuntimeError("Could not locate logits in the tuple returned by the reasoner")
        return output[:logits_index] + (connected(output[logits_index]),) + output[logits_index + 1 :]
    if isinstance(output, torch.Tensor):
        return connected(output)
    raise RuntimeError(f"Could not locate logits in reasoner output of type {type(output).__name__}")


def patch_reasoner_audio_forward(
    model: torch.nn.Module,
    *,
    audio_token_id: int,
    model_type: str,
    trainable_token_ids: tuple[int, ...] | None = None,
) -> None:
    """Add optional audio inputs at a raw HF reasoner's embedding boundary.

    The patch deliberately wraps the raw conditional-generation model rather
    than its text tower. This keeps the audio projector owned by (and executed
    inside) the same FSDP root as the native vision merger. With audio absent,
    the original forward receives the exact original arguments. With audio
    present, native multimodal positions are computed from the untouched token
    IDs before only audio placeholder embeddings are replaced; the original
    forward then performs its normal image/video encoding and masked scatter.
    When ``trainable_token_ids`` is set, gradient-enabled training requires
    audio inputs on every batch. For tied models, the output projection reads a
    detached view of the shared embedding weight so only selected input lookup
    positions can update that table.
    """
    if model_type not in _SUPPORTED_AUDIO_REASONER_TYPES:
        raise NotImplementedError(
            f"Reasoner audio forwarding does not support model_type={model_type!r}; "
            f"supported types are {sorted(_SUPPORTED_AUDIO_REASONER_TYPES)}"
        )
    if isinstance(audio_token_id, bool) or not isinstance(audio_token_id, Integral) or audio_token_id < 0:
        raise ValueError(f"audio_token_id must be a non-negative integer, got {audio_token_id!r}")
    normalized_trainable_token_ids = tuple(trainable_token_ids or ())
    if any(
        isinstance(token_id, bool) or not isinstance(token_id, Integral) or token_id < 0
        for token_id in normalized_trainable_token_ids
    ):
        raise ValueError(f"trainable_token_ids must contain non-negative integers, got {trainable_token_ids!r}")
    if len(set(normalized_trainable_token_ids)) != len(normalized_trainable_token_ids):
        raise ValueError(f"trainable_token_ids must be unique, got {trainable_token_ids!r}")
    if audio_token_id in normalized_trainable_token_ids:
        raise ValueError("The replaced audio placeholder embedding cannot be selected as a trainable boundary token")
    sound_und_model = getattr(model, "sound_und_model", None)
    if sound_und_model is None or not all(hasattr(sound_und_model, name) for name in ("encoder", "projector")):
        raise ValueError("The raw reasoner must own sound_und_model with encoder and projector submodules")
    if not callable(getattr(model, "get_input_embeddings", None)):
        raise ValueError("The raw reasoner must expose get_input_embeddings()")

    patch_identity = (model_type, int(audio_token_id), normalized_trainable_token_ids)
    existing_patch = getattr(model, _PATCH_MARKER, None)
    if existing_patch is not None:
        if existing_patch == patch_identity:
            return
        raise ValueError(
            "Reasoner audio forward is already patched with a different configuration: "
            f"existing={existing_patch}, requested={patch_identity}"
        )
    if normalized_trainable_token_ids and bool(getattr(getattr(model, "config", None), "tie_word_embeddings", False)):
        _patch_tied_output_embedding_forward(model)

    original_forward = model.forward
    original_signature = inspect.signature(original_forward)

    @wraps(original_forward)
    def patched_forward(self: torch.nn.Module, *args: Any, **kwargs: Any) -> Any:
        audio_arguments = {name: kwargs.pop(name, None) for name in _AUDIO_FORWARD_ARGUMENTS}
        has_any_audio_argument = any(value is not None for value in audio_arguments.values())
        has_all_audio_arguments = all(value is not None for value in audio_arguments.values())
        if has_any_audio_argument and not has_all_audio_arguments:
            raise ValueError(f"{', '.join(_AUDIO_FORWARD_ARGUMENTS)} must be provided together")

        bound_arguments = original_signature.bind_partial(*args, **kwargs)
        input_ids = _get_bound_argument(bound_arguments, original_signature, "input_ids")
        inputs_embeds = _get_bound_argument(bound_arguments, original_signature, "inputs_embeds")

        if not has_all_audio_arguments:
            if isinstance(input_ids, torch.Tensor) and torch.any(input_ids == int(audio_token_id)):
                raise ValueError("Reasoner input contains audio placeholders, but audio inputs were not provided")
            if normalized_trainable_token_ids and self.training and torch.is_grad_enabled():
                raise ValueError("Boundary-token-only training requires audio inputs on every training batch")
            output = original_forward(*args, **kwargs)
            projector_zero = _projector_dummy_zero(self)
            if projector_zero is None:
                return output
            labels = _get_bound_argument(bound_arguments, original_signature, "labels")
            return _connect_projector_zero_to_logits(output, projector_zero, labels)

        if not isinstance(input_ids, torch.Tensor):
            raise ValueError("input_ids must be provided when audio inputs are provided")
        if inputs_embeds is not None:
            raise ValueError("Audio inputs cannot be combined with caller-provided inputs_embeds")

        token_embeddings = self.get_input_embeddings()(input_ids)  # [B,N,D]
        if normalized_trainable_token_ids:
            token_embeddings = _restrict_token_embedding_gradients(  # [B,N,D]
                input_ids,
                token_embeddings,
                normalized_trainable_token_ids,
            )
        encoder_parameter = next(self.sound_und_model.encoder.parameters(), None)
        encoder_dtype = encoder_parameter.dtype if encoder_parameter is not None else token_embeddings.dtype
        projected_audio, audio_embedding_lengths = self.sound_und_model(
            audio_features=audio_arguments["audio_features"].to(
                device=token_embeddings.device,
                dtype=encoder_dtype,
            ),
            audio_feature_lengths=audio_arguments["audio_feature_lengths"].to(device=token_embeddings.device),
            audio_token_lengths=audio_arguments["audio_token_lengths"].to(device=token_embeddings.device),
        )
        projected_audio = projected_audio.to(device=token_embeddings.device, dtype=token_embeddings.dtype)
        merged_embeddings = merge_projected_audio_embeddings(
            input_ids,
            token_embeddings,
            projected_audio,
            audio_embedding_lengths.to(device=token_embeddings.device),
            audio_arguments["audio_sample_token_lengths"].to(device=token_embeddings.device),
            int(audio_token_id),
        )

        position_ids = _get_bound_argument(bound_arguments, original_signature, "position_ids")
        if position_ids is None:
            position_ids = _compute_native_position_ids(
                self,
                model_type,
                input_ids,
                bound_arguments,
                original_signature,
            )
        _set_bound_argument(bound_arguments, original_signature, "input_ids", None)
        _set_bound_argument(bound_arguments, original_signature, "inputs_embeds", merged_embeddings)
        _set_bound_argument(bound_arguments, original_signature, "position_ids", position_ids)
        return original_forward(*bound_arguments.args, **bound_arguments.kwargs)

    # ``original_forward`` is already bound, so plain ``@wraps`` would make
    # ``inspect.signature(model.forward)`` drop its first real argument when
    # ``patched_forward`` is bound again. Publish an explicit unbound signature
    # and include the four newly supported keyword-only inputs.
    patched_parameters: list[inspect.Parameter] = [inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    inserted_audio_parameters = False
    for parameter in original_signature.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            patched_parameters.extend(
                inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, default=None)
                for name in _AUDIO_FORWARD_ARGUMENTS
            )
            inserted_audio_parameters = True
        patched_parameters.append(parameter)
    if not inserted_audio_parameters:
        patched_parameters.extend(
            inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, default=None) for name in _AUDIO_FORWARD_ARGUMENTS
        )
    setattr(
        patched_forward,
        "__signature__",
        original_signature.replace(parameters=patched_parameters),
    )
    model.forward = patched_forward.__get__(model, type(model))
    setattr(model, _PATCH_MARKER, patch_identity)


__all__ = [
    "merge_audio_embeddings",
    "merge_projected_audio_embeddings",
    "patch_reasoner_audio_forward",
]
