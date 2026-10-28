# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Backend-exclusive model and VLM packer selection for online GRPO."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from cosmos_framework.posttrain.config.model import DataPackerConfig, HFModelConfig  # noqa: E402
from cosmos_framework.posttrain.config.rl import GRPOConfig, RolloutConfig  # noqa: E402

_QWEN3_VL_MODEL = "Qwen/Qwen3-VL-8B-Instruct"


@pytest.mark.L0
@pytest.mark.CPU
def test_hf_qwen3_vl_backend_selects_generic_hf_model_and_packer(monkeypatch: pytest.MonkeyPatch) -> None:
    """The HF backend must win over Qwen3-VL's native architecture registration."""
    import cosmos_framework.posttrain.config.build as build
    import cosmos_framework.posttrain.data.packer as packer_module
    import cosmos_framework.posttrain.utils.dist as dist
    from cosmos_framework.posttrain.config.build import select_policy_backend
    from cosmos_framework.posttrain.data.packer.hf_vlm import HFVLMDataPacker
    from cosmos_framework.posttrain.entrypoints.grpo import build_rollout

    class _Qwen3VLConfig:
        model_type = "qwen3_vl"
        vision_config = object()

    class _FakeAutoConfig:
        @staticmethod
        def from_pretrained(*_args: object, **_kwargs: object) -> _Qwen3VLConfig:
            return _Qwen3VLConfig()

    config = SimpleNamespace(
        model_backend="hf",
        model_hf=HFModelConfig(name_or_path=_QWEN3_VL_MODEL, trust_remote_code=True),
        rollout=RolloutConfig(name_or_path=_QWEN3_VL_MODEL),
        rl=GRPOConfig(),
    )
    selection = select_policy_backend(config)
    assert selection.backend == "hf"
    assert selection.model_config is config.model_hf
    assert selection.native_spec is None

    monkeypatch.setattr(packer_module, "AutoConfig", _FakeAutoConfig)
    setup_trust: list[bool] = []
    captured: dict[str, object] = {}

    def _setup(_self: object, *_args: object, **kwargs: object) -> None:
        setup_trust.append(bool(kwargs["trust_remote_code"]))

    monkeypatch.setattr(HFVLMDataPacker, "setup", _setup)
    monkeypatch.setattr(dist, "run_rank_converged", lambda fn, _operation: fn())
    monkeypatch.setattr(
        build,
        "build_vllm_rollout",
        lambda _config, *, packer, trust_remote_code, return_routed_experts, enable_sleep_mode: captured.update(
            packer=packer,
            trust_remote_code=trust_remote_code,
        )
        or packer,
    )

    selected = build_rollout(config)

    assert isinstance(selected, HFVLMDataPacker)
    assert setup_trust == [True]
    assert captured == {"packer": selected, "trust_remote_code": True}


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    ("model_type", "has_vision", "backend", "expected_name"),
    [
        ("qwen3_vl", True, "i4", "Qwen3VLDataPacker"),
        ("qwen3_vl", True, "hf", "HFVLMDataPacker"),
        ("unregistered_vision_language_model", True, "hf", "HFVLMDataPacker"),
        ("qwen3", False, "hf", "DecoderOnlyLLMDataPacker"),
        ("llama", False, "i4", "DecoderOnlyLLMDataPacker"),
    ],
)
def test_backend_and_modality_select_one_packer(
    model_type: str,
    has_vision: bool,
    backend: str,
    expected_name: str,
) -> None:
    """Backend and model modality, rather than a Qwen-only branch, own packer selection."""
    from cosmos_framework.posttrain.data.packer import _select_packer_cls

    config = SimpleNamespace(model_type=model_type)
    if has_vision:
        config.vision_config = object()
    assert _select_packer_cls(config, model_backend=backend).__name__ == expected_name


@pytest.mark.L0
@pytest.mark.CPU
def test_policy_packer_build_spec_is_shared_and_immutable() -> None:
    """Every actor resolves the same serializable packer identity, then builds its own instance."""
    from cosmos_framework.posttrain.config.build import policy_data_packer_build_spec

    config = SimpleNamespace(
        model_backend="hf",
        model_hf=HFModelConfig(name_or_path=_QWEN3_VL_MODEL, trust_remote_code=True),
    )
    packer_config = DataPackerConfig(factory="package.factory")

    first = policy_data_packer_build_spec(config, packer_config)
    second = policy_data_packer_build_spec(config, packer_config)

    assert first == second
    assert first.model_id == _QWEN3_VL_MODEL
    assert first.factory == "package.factory"
    assert first.model_backend == "hf"
    assert first.trust_remote_code is True
    with pytest.raises(FrozenInstanceError):
        setattr(first, "model_id", "other/model")


@pytest.mark.L0
@pytest.mark.CPU
def test_i4_qwen3_vl_backend_selects_native_packer_without_importing_hf() -> None:
    """A fresh i4 process must neither import nor select the generic HF VLM packer."""
    script = f"""
import sys
from types import SimpleNamespace

from cosmos_framework.posttrain.config.model import (
    NativePolicyBuildSpec,
    Qwen3VLModelConfig,
)
from cosmos_framework.posttrain.config.rl import GRPOConfig, RolloutConfig
import cosmos_framework.posttrain.config.build as build
import cosmos_framework.posttrain.data.packer as packer_module
from cosmos_framework.posttrain.data.packer.qwen3_vl import Qwen3VLDataPacker
import cosmos_framework.posttrain.utils.dist as dist
from cosmos_framework.posttrain.entrypoints.grpo import build_rollout

hf_module = "cosmos_framework.posttrain.data.packer.hf_vlm"
assert hf_module not in sys.modules, "importing the native packer package imported HFVLMDataPacker"

class Qwen3VLConfig:
    model_type = "qwen3_vl"
    vision_config = object()

model_id = {_QWEN3_VL_MODEL!r}
model_config = Qwen3VLModelConfig(name_or_path=model_id)
config = SimpleNamespace(
    model_backend="i4",
    model=NativePolicyBuildSpec(
        builder="cosmos_framework.posttrain.model.qwen3_vl.build_qwen3_vl_native_policy",
        model_id=model_id,
        model_config=model_config,
        requires_collated_policy_forward=True,
    ),
    rollout=RolloutConfig(name_or_path=model_id),
    rl=GRPOConfig(),
)
packer_module.AutoConfig = SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: Qwen3VLConfig())
Qwen3VLDataPacker.setup = lambda self, *_args, **_kwargs: None
dist.run_rank_converged = lambda fn, _operation: fn()
build.build_vllm_rollout = lambda _config, *, packer, trust_remote_code, return_routed_experts, enable_sleep_mode: packer

packer = build_rollout(config)
assert isinstance(packer, Qwen3VLDataPacker)
assert config.model.requires_collated_policy_forward is True
assert hf_module not in sys.modules, "the native rollout imported the HF VLM packer"
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
