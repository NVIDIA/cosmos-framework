# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Backend selection must check sparse FA4 backward, not just package importability."""

import pytest
import torch

from cosmos_framework.model.generator.mot import flex_attention as module

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


@pytest.fixture(autouse=True)
def _clear_capability_cache():
    probe = getattr(module, "_flash_sparse_backward_unavailable_reason", None)
    if probe is not None:
        probe.cache_clear()
    yield
    if probe is not None:
        probe.cache_clear()


@pytest.fixture
def available_flash(monkeypatch):
    from torch._inductor.kernel.flex import flex_flash_attention

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (9, 0))
    monkeypatch.setattr(flex_flash_attention, "ensure_flash_available", lambda: True)
    return flex_flash_attention


def test_auto_uses_triton_when_torch_rejects_sparse_flash_backward(monkeypatch, available_flash):
    monkeypatch.setattr(
        available_flash,
        "_can_use_flex_flash_attention_backward",
        lambda score, mask: (False, "NYI: block_sparsity"),
    )
    backend = module.resolve_flex_backend(torch.device("cuda"), "auto")
    assert backend == module._get_triton_flex_backend()
    assert backend.kernel_options is None


def test_explicit_flash_reports_unsupported_sparse_backward(monkeypatch, available_flash):
    monkeypatch.setattr(
        available_flash,
        "_can_use_flex_flash_attention_backward",
        lambda score, mask: (False, "NYI: block_sparsity"),
    )
    with pytest.raises(ValueError, match="nontrivial mask in backward.*block_sparsity"):
        module.resolve_flex_backend(torch.device("cuda"), "flex_flash")


@pytest.mark.parametrize("capability", [(9, 0), (10, 0), (10, 3), (11, 0)])
def test_auto_keeps_flash_when_torch_supports_sparse_backward(monkeypatch, available_flash, capability):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: capability)
    monkeypatch.setattr(available_flash, "_can_use_flex_flash_attention_backward", lambda score, mask: (True, ""))
    backend = module.resolve_flex_backend(torch.device("cuda"), "auto")
    assert backend.name == "flash"
    assert backend.kernel_options == {"BACKEND": "FLASH"}
    assert backend.block_size == ((128, 128) if capability[0] == 9 else (256, 128))


@pytest.mark.parametrize("capability", [(12, 0), (12, 1)])
def test_auto_uses_triton_on_sm12_even_when_torch_supports_sparse_backward(monkeypatch, available_flash, capability):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: capability)
    monkeypatch.setattr(available_flash, "_can_use_flex_flash_attention_backward", lambda score, mask: (True, ""))
    backend = module.resolve_flex_backend(torch.device("cuda"), "auto")
    assert backend == module._get_triton_flex_backend()
    assert backend.kernel_options is None
    assert backend.block_size == (128, 128)


@pytest.mark.parametrize("capability", [(12, 0), (12, 1)])
def test_explicit_flash_reports_sm12_sparse_attention_limitation(monkeypatch, available_flash, capability):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: capability)
    monkeypatch.setattr(available_flash, "_can_use_flex_flash_attention_backward", lambda score, mask: (True, ""))
    with pytest.raises(ValueError, match=rf"4\.0\.0b24.*block-sparse.*sm{capability[0]}{capability[1]}"):
        module.resolve_flex_backend(torch.device("cuda"), "flex_flash")
