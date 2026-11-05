# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Derived-weight invalidation and ownership, independently of CUDA inference kernels."""

import pytest
import torch
from torch import nn

from cosmos_framework.posttrain.config.adapters import AdapterComponentConfig
from cosmos_framework.posttrain.model.adapter_inference_cache import (
    adapter_inference_cache_info,
    bind_adapter_inference_cache_optimizer,
    clear_adapter_inference_cache,
    configure_weight_cache,
    effective_weight_for_inference,
)
from cosmos_framework.posttrain.model.adapters import PosttrainAdapterLinear


def layer(method="lora"):
    base = nn.Linear(7, 9).requires_grad_(False)
    config = AdapterComponentConfig(
        component="language",
        targets=["projection"],
        method=method,
        initialization="svd_lowrank" if method == "pissa" else "default",
        rank=2,
        alpha=3,
        use_rslora=True,
        execution="merged_weight_bf16_v1",
    )
    return PosttrainAdapterLinear(base, config).eval()


@pytest.mark.parametrize("method", ["lora", "dora", "pissa"])
def test_cache_matches_uncached_preserves_ownership_and_invalidates_every_source(method):
    module = layer(method)
    parameters = {name: (id(p), p.requires_grad) for name, p in module.named_parameters()}
    keys = set(module.state_dict())
    optimizer = torch.optim.AdamW([p for p in module.parameters() if p.requires_grad])
    members = [id(p) for group in optimizer.param_groups for p in group["params"]]
    configure_weight_cache(module, True)
    with torch.no_grad():
        value = effective_weight_for_inference(module)
        assert effective_weight_for_inference(module) is value
        assert torch.equal(value, module.effective_weight())
        assert not value.requires_grad and value.grad_fn is None
        # Includes DoRA magnitude, PiSSA initial buffers, base and bias.
        for name, source in module.state_dict(keep_vars=True).items():
            previous = effective_weight_for_inference(module)
            source.add_(0.01)
            current = effective_weight_for_inference(module)
            assert current is not previous, name
            assert torch.equal(current, module.effective_weight()), name
    assert set(module.state_dict()) == keys
    assert parameters == {name: (id(p), p.requires_grad) for name, p in module.named_parameters()}
    assert members == [id(p) for group in optimizer.param_groups for p in group["params"]]
    assert len(members) == len(set(members))
    assert adapter_inference_cache_info(module)["cached_bytes"] == value.numel() * value.element_size()


def test_cache_replacement_scaling_load_and_dtype_lifecycle():
    module = layer("pissa")
    configure_weight_cache(module, True)
    saved = {name: value.clone() for name, value in module.state_dict().items()}
    with torch.no_grad():
        previous = effective_weight_for_inference(module)
        module.lora_A.weight = nn.Parameter(module.lora_A.weight.clone() + 0.1)
        current = effective_weight_for_inference(module)
        assert current is not previous and torch.equal(current, module.effective_weight())
        module._lora_alpha *= 2
        assert effective_weight_for_inference(module) is not current
        module.load_state_dict(saved)
        assert adapter_inference_cache_info(module)["cached_bytes"] == 0
        effective_weight_for_inference(module)
        module.double()
        assert adapter_inference_cache_info(module)["cached_bytes"] == 0
        assert effective_weight_for_inference(module).dtype == torch.float64
        module.train()
        assert adapter_inference_cache_info(module)["cached_bytes"] == 0
        with pytest.raises(RuntimeError, match="eager eval"):
            effective_weight_for_inference(module)


def test_shared_parameters_update_both_caches_without_optimizer_duplication():
    left, right = layer("dora"), layer("dora")
    right.weight, right.bias = left.weight, left.bias
    right.lora_A, right.lora_B, right.magnitude = left.lora_A, left.lora_B, left.magnitude
    net = nn.ModuleList([left, right])
    ownership = [id(p) for p in net.parameters()]
    for module in net:
        configure_weight_cache(module, True)
    with torch.no_grad():
        before = [effective_weight_for_inference(x) for x in net]
        left.lora_B.weight.add_(0.2)
        after = [effective_weight_for_inference(x) for x in net]
        assert all(a is not b for a, b in zip(before, after, strict=True))
        assert torch.equal(after[0], after[1])
    assert ownership == [id(p) for p in net.parameters()]
    assert len(ownership) == len(set(ownership))


def test_clear_for_unsafe_data_writes_and_disable_does_not_serialize_cache():
    module = layer()
    configure_weight_cache(module, True)
    keys = set(module.state_dict())
    with torch.no_grad():
        value = effective_weight_for_inference(module)
        module.lora_B.weight.data.add_(0.5)  # Explicitly outside PyTorch version tracking.
        clear_adapter_inference_cache(module)
        assert adapter_inference_cache_info(module)["cached_bytes"] == 0
        assert not torch.equal(value, effective_weight_for_inference(module))
        configure_weight_cache(module, False)
        assert adapter_inference_cache_info(module)["enabled_modules"] == 0
        assert effective_weight_for_inference(module) is not effective_weight_for_inference(module)
    assert set(module.state_dict()) == keys


def test_forbidden_contexts_and_unversioned_sources_fail_before_cache_mutation():
    module = layer()
    configure_weight_cache(module, True)
    with pytest.raises(RuntimeError, match="no_grad"):
        effective_weight_for_inference(module)
    with torch.no_grad(), torch.autocast("cpu"):
        with pytest.raises(RuntimeError, match="autocast"):
            effective_weight_for_inference(module)
    assert adapter_inference_cache_info(module)["cached_bytes"] == 0
    with torch.inference_mode():
        module.lora_A.weight = nn.Parameter(torch.zeros_like(module.lora_A.weight))
    configure_weight_cache(module, False)
    with pytest.raises(ValueError, match="version-tracked"):
        configure_weight_cache(module, True)
    assert adapter_inference_cache_info(module)["enabled_modules"] == 0


@pytest.mark.parametrize("options", [{}, {"foreach": True}, {"fused": True}])
def test_bound_optimizer_invalidates_even_without_tensor_version_change(options):
    module = layer("dora")
    optimizer = torch.optim.AdamW([p for p in module.parameters() if p.requires_grad], lr=0.1, **options)
    members = [id(p) for group in optimizer.param_groups for p in group["params"]]
    configure_weight_cache(module, True)
    handle = bind_adapter_inference_cache_optimizer(module, optimizer)
    with torch.no_grad():
        before = effective_weight_for_inference(module)
        module.lora_B.weight.grad = torch.ones_like(module.lora_B.weight)
        optimizer.step()
        assert adapter_inference_cache_info(module)["cached_bytes"] == 0
        after = effective_weight_for_inference(module)
        assert not torch.equal(before, after)
        assert torch.equal(after, module.effective_weight())
    assert members == [id(p) for group in optimizer.param_groups for p in group["params"]]
    handle.remove()


def test_optimizer_rebind_disable_partial_failure_and_model_lifetime():
    import gc
    import weakref

    module = layer()

    class PartialUpdate(torch.optim.Optimizer):
        def step(self, closure=None):
            self.param_groups[0]["params"][0].data.add_(1)
            raise RuntimeError("partial update")

    optimizer = PartialUpdate([module.lora_B.weight], {})
    configure_weight_cache(module, True)
    first = bind_adapter_inference_cache_optimizer(module, optimizer)
    second = bind_adapter_inference_cache_optimizer(module, optimizer)
    assert len(optimizer._optimizer_step_pre_hooks) == len(optimizer._optimizer_step_post_hooks) == 1
    first.remove()  # Removing the replaced binding must leave the new one intact.
    with torch.no_grad():
        previous = effective_weight_for_inference(module)
        with pytest.raises(RuntimeError, match="partial update"):
            optimizer.step()
        assert adapter_inference_cache_info(module)["cached_bytes"] == 0
        assert not torch.equal(previous, effective_weight_for_inference(module))
    configure_weight_cache(module, False)
    assert not optimizer._optimizer_step_pre_hooks and not optimizer._optimizer_step_post_hooks
    second.remove()
    configure_weight_cache(module, True)
    handle = bind_adapter_inference_cache_optimizer(module, optimizer)
    reference = weakref.ref(module)
    del module
    gc.collect()
    assert reference() is None
    handle.remove()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA device migration")
def test_device_migration_clears_without_registering_derived_weights():
    module = layer("dora")
    configure_weight_cache(module, True)
    with torch.no_grad():
        effective_weight_for_inference(module)
        module.cuda()
        assert adapter_inference_cache_info(module)["cached_bytes"] == 0
        assert effective_weight_for_inference(module).is_cuda
        module.cpu()
        assert adapter_inference_cache_info(module)["cached_bytes"] == 0
        assert effective_weight_for_inference(module).device.type == "cpu"
