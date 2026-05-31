# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Load the understanding tower of a Cosmos3 checkpoint."""

import re
from collections.abc import Iterable

import torch
from transformers_cosmos3.model import DROP_PATTERNS, KEY_MAPPING
from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration

_DROP_RE = re.compile("|".join(DROP_PATTERNS))
_KEY_MAPPING_RES: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(src), tgt) for src, tgt in KEY_MAPPING.items()
)
_UNDERSTANDING_PREFIXES: tuple[str, ...] = (
    "lm_head.",
    "model.language_model.",
    "model.visual.",
)


def _to_hf_name(name: str) -> str:
    for pattern, replacement in _KEY_MAPPING_RES:
        name = pattern.sub(replacement, name)
    return name


def _is_und_tower_weight(name: str) -> bool:
    if _DROP_RE.search(name) is not None:
        return False
    return _to_hf_name(name).startswith(_UNDERSTANDING_PREFIXES)


class Cosmos3ReasonerForConditionalGeneration(Qwen3VLForConditionalGeneration):
    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        overrides = getattr(vllm_config.model_config.hf_config, "allow_patterns_overrides", None)
        if overrides:
            self.allow_patterns_overrides = list(overrides)
            if any(p.endswith(".safetensors") for p in self.allow_patterns_overrides):
                vllm_config.load_config.load_format = "safetensors"

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        def _iter() -> Iterable[tuple[str, torch.Tensor]]:
            for name, tensor in weights:
                if not _is_und_tower_weight(name):
                    continue
                yield _to_hf_name(name), tensor

        loaded = super().load_weights(_iter())
        return loaded
