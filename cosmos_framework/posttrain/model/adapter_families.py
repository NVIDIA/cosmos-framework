# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Architecture-owned adapter boundaries, independent of adapter mathematics.

An entry binds names to components, not a blanket qualification claim. Operators
still pass the resolver's type/shape/ownership checks; GPU qualification is tracked
separately. Unknown architectures must supply a reviewed binding, never a heuristic
substring match that could silently train the wrong part of a model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class AdapterFamily:
    name: str
    # (component, anchored module prefix regex). A named `layer` group identifies
    # the actual repeated block; its prefix also identifies a native FSDP unit.
    boundaries: tuple[tuple[str, str], ...]
    normalization_types: tuple[str, ...] = ()
    hf_auto_class: str | None = None
    embedding: str | None = None

    def locate(self, name: str) -> tuple[str, int | None, str] | None:
        found = []
        for component, pattern in self.boundaries:
            match = re.match(pattern, name)
            if match:
                layer = match.groupdict().get("layer")
                unit = match.group(0).rstrip(".") if layer is not None else "root"
                found.append((component, int(layer) if layer is not None else None, unit))
        if len(found) > 1:
            raise ValueError(f"ambiguous {self.name} adapter component for {name}")
        return found[0] if found else None


_VLM = (
    ("language", r"^model\.language_model\.layers\.(?P<layer>\d+)\."),
    ("language", r"^model\.language_model\.norm(?:\.|$)"),
    ("projector", r"^model\.visual\.(?:merger|deepstack_merger_list\.\d+)(?:\.|$)"),
    ("vision", r"^model\.visual\.blocks\.(?P<layer>\d+)\."),
)
_TEXT = (
    ("language", r"^model\.layers\.(?P<layer>\d+)\."),
    ("language", r"^model\.norm(?:\.|$)"),
)
_FAMILIES = {
    name: AdapterFamily(name, _VLM, norms, "AutoModelForImageTextToText", "model.language_model.embed_tokens.weight")
    for name, norms in (
        ("qwen3_vl", ("Qwen3VLTextRMSNorm",)),
        ("qwen3_vl_moe", ("Qwen3VLMoeTextRMSNorm",)),
        ("qwen2_vl", ("Qwen2RMSNorm",)),
        ("qwen2_5_vl", ("Qwen2RMSNorm",)),
        ("qwen3_5", ("Qwen3_5RMSNorm",)),
    )
}
_FAMILIES.update(
    {
        name: AdapterFamily(name, _TEXT, (norm,), "AutoModelForCausalLM", "model.embed_tokens.weight")
        for name, norm in (
            ("qwen2", "Qwen2RMSNorm"),
            ("qwen3", "Qwen3RMSNorm"),
            ("llama", "LlamaRMSNorm"),
            ("mistral", "MistralRMSNorm"),
            ("gemma", "GemmaRMSNorm"),
            ("gemma2", "Gemma2RMSNorm"),
        )
    }
)


def adapter_family(name: str) -> AdapterFamily:
    try:
        return _FAMILIES[name]
    except KeyError as error:
        raise ValueError(
            f"no adapter component binding for architecture {name!r}; available: {sorted(_FAMILIES)}"
        ) from error


def family_for_model(net, family: str | None = None) -> AdapterFamily:
    # Explicit family is used by owners whose `.net` is only a trainable subtree.
    # The fallback preserves the original standalone native-Qwen fixture API.
    declared = getattr(getattr(net, "config", None), "model_type", None)
    if family is not None and declared is not None and family != declared:
        raise ValueError(f"adapter family {family!r} disagrees with model architecture {declared!r}")
    return adapter_family(family or declared or "qwen3_vl")
