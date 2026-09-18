# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from collections.abc import Iterator
from functools import wraps
from types import MethodType
from typing import Any, NamedTuple

import torch
import torch.nn as nn

from cosmos_framework.model.generator.algorithm.loss.cross_entropy import LossStatistics, fused_weighted_cross_entropy_loss


class Qwen35FP32Decay(nn.Module):
    """Own recurrent A_log in a separately shardable FP32 module."""

    def __init__(self, a_log: nn.Parameter) -> None:
        super().__init__()
        self.A_log = a_log

    def forward(self, unused: None = None) -> torch.Tensor:
        del unused
        # FSDP may free the holder's gathered storage as this getter returns.
        # HF consumes A_log afterwards, so return an independent autograd value.
        return self.A_log.float().clone()


_FP32_ACCESS_CLASSES: dict[type[nn.Module], type[nn.Module]] = {}


def _fp32_fqn_modifiers(module: nn.Module) -> dict[str, str]:
    # DCP traverses canonical state-dict keys. Resolve the original owner directly
    # so checkpoint inspection neither calls forward nor launches an all-gather.
    del module
    return {"A_log": "_fp32_params"}


def _fp32_a_log(module: nn.Module) -> torch.Tensor:
    return module._fp32_params(None)


def _strip_fp32_holder_from_state_dict(
    module: nn.Module, state_dict: dict[str, torch.Tensor], prefix: str, local_metadata: dict[str, Any]
) -> None:
    del module, local_metadata
    key = f"{prefix}_fp32_params.A_log"
    if key in state_dict:
        state_dict[f"{prefix}A_log"] = state_dict.pop(key)


def _route_fp32_holder_in_state_dict(
    module: nn.Module,
    state_dict: dict[str, torch.Tensor],
    prefix: str,
    local_metadata: dict[str, Any],
    strict: bool,
    missing_keys: list[str],
    unexpected_keys: list[str],
    error_msgs: list[str],
) -> None:
    del module, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    key = f"{prefix}A_log"
    if key in state_dict:
        state_dict[f"{prefix}_fp32_params.A_log"] = state_dict.pop(key).float()


def qwen35_fp32_decay_modules(model: nn.Module) -> list[Qwen35FP32Decay]:
    """Return FP32 units to shard before their parent decoder blocks."""
    return [module for module in model.modules() if isinstance(module, Qwen35FP32Decay)]


def named_parameters_with_qwen35_decay(
    module: nn.Module, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
) -> Iterator[tuple[str, nn.Parameter]]:
    """Keep optimizer FQNs consistent with canonical model checkpoint keys.

    DCP's model traversal honors _fqn_modifiers, but its optimizer exporter uses
    named_parameters. Both must expose A_log without the private holder segment.
    Parameter objects and the actual module hierarchy are unchanged.
    """
    for name, parameter in nn.Module.named_parameters(module, prefix, recurse, remove_duplicate):
        yield name.replace("_fp32_params.", ""), parameter


def install_qwen35_fp32_decay(model: nn.Module) -> int:
    """Route HF's existing A_log reads through an FP32 FSDP module.

    The small descriptor subclass inherits the original recurrent forward.
    This avoids copying a version-specific Transformers implementation. A real
    nn.Module owns the parameter because fully_shard rejects ModuleList-based
    parametrization containers. State hooks preserve the pretrained A_log key.
    """
    candidates = [
        module
        for module in model.modules()
        if "A_log" in module._parameters and hasattr(module, "in_proj_a") and hasattr(module, "conv1d")
    ]
    for module in candidates:
        a_log = module._parameters.pop("A_log")
        a_log.data = a_log.data.float()
        module.add_module("_fp32_params", Qwen35FP32Decay(a_log))
        original_class = type(module)
        if original_class not in _FP32_ACCESS_CLASSES:
            _FP32_ACCESS_CLASSES[original_class] = type(
                f"{original_class.__name__}WithFP32Decay",
                (original_class,),
                {"A_log": property(_fp32_a_log), "_fqn_modifiers": _fp32_fqn_modifiers},
            )
        module.__class__ = _FP32_ACCESS_CLASSES[original_class]
        module.register_state_dict_post_hook(_strip_fp32_holder_from_state_dict)
        module.register_load_state_dict_pre_hook(_route_fp32_holder_in_state_dict)
    if candidates:
        model.named_parameters = MethodType(named_parameters_with_qwen35_decay, model)
    return len(candidates)


class Qwen35CaptionLoss(NamedTuple):
    """Scalar objective produced inside the FSDP root, before its weights reshard."""

    loss: torch.Tensor
    stats: LossStatistics


def _text_only_vision_dependency(model: nn.Module) -> torch.Tensor:
    """Keep vision gradients connected for text-only ranks in a mixed-modality step."""
    visual = model.model.visual
    config = model.config.vision_config
    parameter = next(visual.parameters())
    merge = config.spatial_merge_size
    width = config.in_channels * config.temporal_patch_size * config.patch_size**2
    pixels = torch.zeros((merge * merge, width), device=parameter.device, dtype=parameter.dtype)
    grid = torch.tensor([[1, merge, merge]], device=parameter.device, dtype=torch.long)
    output = visual(pixels, grid_thw=grid)
    features = getattr(output, "pooler_output", None)
    if features is None:
        features = output if isinstance(output, torch.Tensor) else output[0]
    return features.sum() * 0.0


def configure_qwen35_caption_model(
    model: nn.Module,
    *,
    fp32_recurrent_a_log: bool = False,
    fused_weighted_ce: bool = False,
    weighted_ce_exponent: float = 0.5,
) -> None:
    """Install the MR !11447 options on the unified dense Qwen3.5 HF model.

    The conditional vision tower stays in the root FSDP unit (parallelize_vlm),
    so text, image and video ranks execute the same collective schedule. The
    scalar fused loss also executes within this root while lm_head is gathered.
    """
    if model.config.model_type != "qwen3_5":
        raise ValueError("Unified caption options currently support dense qwen3_5 models")
    if getattr(model, "_unified_qwen35_caption", False):
        raise ValueError("Qwen3.5 caption model has already been configured")
    if fp32_recurrent_a_log:
        count = install_qwen35_fp32_decay(model)
        if count == 0:
            raise ValueError("Qwen3.5 FP32 decay requires at least one GatedDeltaNet layer")
    original_forward = model.forward

    @wraps(original_forward.__func__)
    def forward(self: nn.Module, *args: Any, **kwargs: Any) -> Any:
        text_only = kwargs.get("pixel_values") is None and kwargs.get("pixel_values_videos") is None
        dependency = _text_only_vision_dependency(self) if text_only else None
        if fused_weighted_ce:
            labels = kwargs.pop("labels", None)
            if labels is None:
                raise ValueError("Fused caption training requires labels; use an unfused model for generation")
            kwargs.pop("logits_to_keep", None)
            kwargs["use_cache"] = False
            hidden = self.model(*args, **kwargs).last_hidden_state
            if dependency is not None:
                hidden = hidden + dependency.to(hidden.dtype)
            loss, stats = fused_weighted_cross_entropy_loss(
                hidden,
                labels,
                self.lm_head.weight,
                exponent=weighted_ce_exponent,
            )
            return Qwen35CaptionLoss(loss, stats)
        output = original_forward(*args, **kwargs)
        if dependency is not None:
            output.logits = output.logits + dependency.to(output.logits.dtype)
        return output

    model.forward = MethodType(forward, model)
    model._unified_qwen35_caption = True
