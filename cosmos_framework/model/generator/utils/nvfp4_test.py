# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Light CPU tests for the NVFP4 linear conversion (no GPU / CUDA extensions needed)."""

import pytest

from cosmos_framework.model.generator.utils.nvfp4 import (
    _is_skipped,
    convert_decoder_linears_to_nvfp4,
    resolve_legacy_nvfp4_mode,
)


@pytest.mark.L0
@pytest.mark.parametrize(
    ("config_mode", "w4a16_env", "w4a4_env", "expected_mode"),
    [
        pytest.param("w4a4", "true", "yes", "w4a4", id="config-priority"),
        pytest.param(None, "YeS", "true", "w4a16", id="w4a16-env-priority"),
        pytest.param(None, None, "1", "w4a4", id="w4a4-env-fallback"),
    ],
)
def test_resolve_legacy_nvfp4_mode_precedence(
    monkeypatch: pytest.MonkeyPatch,
    config_mode: str | None,
    w4a16_env: str | None,
    w4a4_env: str | None,
    expected_mode: str,
) -> None:
    for name, value in (
        ("COSMOS3_W4A16_TORCHAO", w4a16_env),
        ("COSMOS3_NVFP4", w4a4_env),
    ):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)

    assert resolve_legacy_nvfp4_mode(config_mode) == expected_mode


@pytest.mark.L0
def test_skip_patterns_keep_sensitive_layers_high_precision():
    # quantized (projections)
    assert not _is_skipped("self_attn.q_proj")
    assert not _is_skipped("mlp.gate_proj")
    # kept high precision
    for name in ("self_attn.gate", "input_layernorm", "lm_head", "time_embed", "vae2llm"):
        assert _is_skipped(name), name


@pytest.mark.L0
def test_convert_rejects_unknown_mode():
    # An unknown mode must fail fast (before touching the model / CUDA ops).
    with pytest.raises(ValueError, match="unknown nvfp4 linear mode"):
        convert_decoder_linears_to_nvfp4(object(), "bogus")
