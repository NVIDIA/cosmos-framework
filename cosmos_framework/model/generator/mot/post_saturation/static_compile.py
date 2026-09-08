# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Static torch.compile routing for post-saturation AR inference."""

from typing import Any

import torch
import torch.nn as nn
from loguru import logger as log

from cosmos_framework.configs.base.defaults.compile import CompileConfig
from cosmos_framework.data.generator.sequence_packing.modality import compute_text_split_length


def _compile_options_from_config(config: CompileConfig) -> dict[str, bool]:
    compile_options: dict[str, bool] = {}
    if config.max_autotune_pointwise:
        compile_options["max_autotune_pointwise"] = True
    if config.coordinate_descent_tuning:
        compile_options["coordinate_descent_tuning"] = True
    return compile_options


def _is_post_saturation_static_compile_frame(frame_idx: int, kv_cache_inference_size: int | None) -> bool:
    """Return whether AR has reached the fixed-size rolling KV history phase."""
    if kv_cache_inference_size is None:
        raise ValueError("A static post-saturation mode requires kv_cache_inference_size to be set.")
    if kv_cache_inference_size < 2:
        raise ValueError(f"kv_cache_inference_size must be >= 2, got {kv_cache_inference_size}")
    return frame_idx >= kv_cache_inference_size - 1


def _ar_packed_text_len(text_tokens: list[int] | None, special_tokens: dict[str, int]) -> int:
    """Return packed AR text length including special text/generation tokens."""
    return compute_text_split_length(
        len(text_tokens) if text_tokens is not None else 0,
        special_tokens,
        has_generation=True,
    )


def _validate_ar_static_und_cache_len(
    *,
    branch_name: str,
    s_und: int,
    max_s_und: int,
) -> None:
    """Fail loudly when a prompt cannot fit the fixed post-saturation und cache."""
    if s_und > max_s_und:
        raise ValueError(
            f"[AR inference] {branch_name} S_und={s_und} exceeds "
            f"ar_static_und_cache_max_len={max_s_und}. Increase "
            "model.config.ar_static_und_cache_max_len or truncate/tokenize prompts to a smaller max length."
        )


def validate_ar_static_und_cache_lengths(
    *,
    cond_text_tokens: list[int] | None,
    uncond_text_tokens: list[int] | None,
    cfg_active: bool,
    special_tokens: dict[str, int],
    max_s_und: int,
) -> None:
    """Validate frame-0 packed text lengths for post-saturation static compile."""
    cond_s_und = _ar_packed_text_len(cond_text_tokens, special_tokens)
    _validate_ar_static_und_cache_len(
        branch_name="conditional",
        s_und=cond_s_und,
        max_s_und=max_s_und,
    )

    if cfg_active:
        uncond_s_und = _ar_packed_text_len(uncond_text_tokens, special_tokens)
        _validate_ar_static_und_cache_len(
            branch_name="unconditional",
            s_und=uncond_s_und,
            max_s_und=max_s_und,
        )
    else:
        uncond_s_und = None

    log.info(
        "[AR inference] post-saturation static compile und cache lengths: "
        f"cond S_und={cond_s_und}, uncond S_und={uncond_s_und}, "
        f"ar_static_und_cache_max_len={max_s_und}"
    )


class ARPostSaturationStaticCompileRouter(nn.Module):
    """Route AR post-saturation calls to a separate static compiled layer."""

    def __init__(self, default_layer: nn.Module, compile_options: dict[str, bool]) -> None:
        super().__init__()
        self.default_layer = default_layer
        eager_layer = getattr(default_layer, "_orig_mod", default_layer)
        self._post_saturation_forward = torch.compile(
            eager_layer.forward,
            fullgraph=True,
            dynamic=False,
            mode="max-autotune-no-cudagraphs",
            options=compile_options or None,
        )

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        memory_value = kwargs.get("memory_value")
        if getattr(memory_value, "post_saturation_static_compile", False):
            return self._post_saturation_forward(*args, **kwargs)
        return self.default_layer(*args, **kwargs)

    def reasoner_forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.default_layer.reasoner_forward(*args, **kwargs)


def apply_ar_post_saturation_static_compile(model: nn.Module, config: CompileConfig) -> None:
    """Install AR post-saturation static-compile routing on decoder layers."""
    compile_options = _compile_options_from_config(config)
    for layer_id, block in model.model.layers.named_children():
        if isinstance(block, ARPostSaturationStaticCompileRouter):
            continue
        block = ARPostSaturationStaticCompileRouter(block, compile_options=compile_options)
        model.model.layers.register_module(layer_id, block)
