# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Shared construction constants for BF16 sources and FP32 training/deployment."""

from __future__ import annotations

import torch


def buffer_contract(config) -> dict | None:
    if config.model_type not in ("gemma", "gemma2"):
        return None
    return {
        "nonpersistent_buffer_policy": "gemma-fp32-embedding-scale-v1",
        "buffers": {
            "model.embed_tokens.embed_scale": {
                "dtype": "float32",
                "value": torch.tensor(config.hidden_size**0.5, dtype=torch.float32).item(),
            }
        },
    }


def prepare_adapter_buffers(net) -> dict | None:
    """Keep config-derived scales in FP32; the native forward casts at use time.

    HF from_config(dtype=BF16) otherwise constructs Gemma's nonpersistent scale
    in BF16. Promoting that rounded buffer cannot recover the constant a fresh
    FP32 HF loader regenerates. Reconstruct it before distributed registration.
    BF16 forward values and embedding/head parameter tying remain unchanged.
    """
    contract = buffer_contract(net.config)
    if contract is None:
        return None
    embedding = net.model.embed_tokens
    expected_type = {"gemma": "GemmaTextScaledWordEmbedding", "gemma2": "Gemma2TextScaledWordEmbedding"}
    if (
        type(embedding).__name__ != expected_type[net.config.model_type]
        or "embed_scale" not in embedding._non_persistent_buffers_set
        or embedding.scalar_embed_scale != net.config.hidden_size**0.5
        or embedding.embed_scale.is_meta
    ):
        raise ValueError(
            "Gemma adapter embedding-scale implementation changed; qualify the installed Transformers version"
        )
    embedding.register_buffer(
        "embed_scale",
        torch.tensor(embedding.scalar_embed_scale, dtype=torch.float32, device=embedding.embed_scale.device),
        persistent=False,
    )
    return contract


def validate_buffer_contract(identity: dict, config) -> None:
    contract = buffer_contract(config)
    if contract is not None and identity.get("model_contract") != contract:
        raise ValueError(
            "Gemma adapter artifact lacks the canonical FP32 embedding-scale contract; "
            "train/export with the current builder before requesting a merged HF model"
        )
