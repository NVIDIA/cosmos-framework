# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Focused contracts for rolling decoder optimization."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from cosmos_framework.model.generator.mot.post_saturation.installer import install_ar_post_saturation_mode
from cosmos_framework.model.generator.mot.post_saturation.rolling import RollingPostSaturation
from cosmos_framework.model.generator.mot.post_saturation.static_compile import ARPostSaturationStaticCompileRouter
from cosmos_framework.model.generator.utils.kv_cache import MultiviewARMemoryState


@pytest.mark.L0
@pytest.mark.CPU
def test_static_compile_combines_mode_and_precision_options(monkeypatch: pytest.MonkeyPatch) -> None:
    options: dict[str, Any] = {}

    def compile_forward(forward: Any, **kwargs: Any) -> Any:
        options.update(kwargs)
        return forward

    monkeypatch.setattr(torch, "compile", compile_forward)
    ARPostSaturationStaticCompileRouter(torch.nn.Identity(), {"emulate_precision_casts": True})
    assert "mode" not in options  # torch.compile accepts mode or options, not both.
    assert options["options"]["max_autotune"]
    assert not options["options"].get("triton.cudagraphs", False)
    assert options["options"]["emulate_precision_casts"]
    assert options["fullgraph"] and not options["dynamic"]


@pytest.mark.L0
@pytest.mark.CPU
def test_compile_only_saturated_noisy_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: list[bool] = []
    installed: list[bool] = []

    def install(model: Any, config: Any, *, emulate_precision_casts: bool) -> None:
        installed.append(emulate_precision_casts)

    monkeypatch.setattr(
        "cosmos_framework.model.generator.mot.post_saturation.rolling.apply_ar_post_saturation_static_compile",
        install,
    )

    def denoise(*, data_batch_packed: Any, memory: MultiviewARMemoryState) -> None:
        observed.append(memory.read_for_layer(0).post_saturation_static_compile)

    host = SimpleNamespace(
        config=SimpleNamespace(
            compile=SimpleNamespace(ar_post_saturation_mode="static-compile", enabled=False, use_cuda_graphs=False),
            teacher_forcing_kv_implementation="multiview_maskless_kv",
            rolling_kv_cache_chunks=2,
        ),
        net=SimpleNamespace(language_model=object(), teacher_forcing_maskless=True),
        parallel_dims=None,
        denoise=denoise,
    )
    install_ar_post_saturation_mode(host)
    assert installed == []  # Shared loading must leave installation to the rolling sampler.
    runtime = RollingPostSaturation(host)
    for saturated in (False, True):
        runtime.denoise(None, MultiviewARMemoryState(num_layers=1, memory_seq_len=4), saturated=saturated)
    assert observed == [False, True]
    RollingPostSaturation(host)  # Subsequent samples reuse the installed routers.
    assert installed == [True]
    # Clean refresh creates a separate state and must retain eager routing.
    assert not MultiviewARMemoryState(num_layers=1, memory_seq_len=4).read_for_layer(0).post_saturation_static_compile


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    "override, message",
    [
        ({"implementation": "multiview_flex_kv"}, "Multiview FlexAttention does not support"),
        ({"mode": "cuda-graph"}, "only static-compile"),
        ({"cache_chunks": None}, "bounded rolling maskless"),
        ({"maskless": False}, "bounded rolling maskless"),
        ({"global_compile": True}, "--no-use-torch-compile"),
        ({"cuda_graphs": True}, "--no-use-cuda-graphs"),
    ],
)
def test_shared_loader_rejects_unsupported_multiview_compile(override: dict[str, Any], message: str) -> None:
    host = SimpleNamespace(
        config=SimpleNamespace(
            teacher_forcing_kv_implementation=override.get("implementation", "multiview_maskless_kv"),
            rolling_kv_cache_chunks=override.get("cache_chunks", 2),
            compile=SimpleNamespace(
                ar_post_saturation_mode=override.get("mode", "static-compile"),
                enabled=override.get("global_compile", False),
                use_cuda_graphs=override.get("cuda_graphs", False),
            ),
        ),
        net=SimpleNamespace(teacher_forcing_maskless=override.get("maskless", True)),
    )
    with pytest.raises(ValueError, match=message):
        install_ar_post_saturation_mode(host)
