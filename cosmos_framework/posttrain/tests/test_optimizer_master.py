# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""CPU numerical/ownership checks; CUDA/FSDP gates run separately."""

from copy import deepcopy

import numpy as np
import pytest
import torch

from cosmos_framework.posttrain.config.build import build_optimizer
from cosmos_framework.posttrain.config.rl import OptimizerConfig


def model():
    net = torch.nn.Linear(4, 3, bias=False, dtype=torch.bfloat16)
    with torch.no_grad():
        net.weight.fill_(1)
    net.register_parameter("tied", net.weight)
    net.register_parameter("unused", torch.nn.Parameter(torch.ones(3, dtype=torch.bfloat16)))
    net.register_parameter("frozen", torch.nn.Parameter(torch.ones(3), requires_grad=False))
    return net


def make(net, warmup=0):
    return build_optimizer(
        net, OptimizerConfig(lr=1e-4, weight_decay=0, warmup_steps=warmup, update_policy="fp32_master")
    )


def assert_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            assert_equal(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_equal(x, y)
    elif isinstance(a, np.ndarray):  # NumPy RNG state needs array equality.
        assert np.array_equal(a, b)
    else:
        assert a == b


def test_master_retains_sub_bf16_updates_and_exact_membership():
    net = model()
    opt = make(net)
    reference = torch.nn.Parameter(net.weight.detach().float().clone())
    control = torch.optim.AdamW([reference], lr=1e-4, weight_decay=0)
    assert opt.param_groups[0]["params"][0] is net.weight
    assert opt.names == ("weight", "unused")
    for _ in range(40):
        net.weight.grad = torch.ones_like(net.weight)
        reference.grad = torch.ones_like(reference)
        opt.step()
        control.step()
        assert torch.equal(opt.masters[0], reference)
        assert torch.equal(net.weight, reference.bfloat16())
    assert (net.weight < 1).all()  # Direct BF16 updates of this size would vanish.
    assert not opt.state.get(net.unused)
    assert torch.equal(net.frozen, torch.ones(3))


def test_master_resume_next_updates_and_rejected_state_is_transactional():
    net = model()
    opt = make(net)
    for _ in range(7):
        net.weight.grad = torch.full_like(net.weight, 0.5)
        opt.step()
    saved = opt.state_dict()
    other = model()
    other.load_state_dict(net.state_dict())
    restored = make(other)
    restored.load_state_dict(saved)
    for _ in range(4):
        net.weight.grad = torch.full_like(net.weight, -0.3)
        other.weight.grad = net.weight.grad.clone()
        opt.step()
        restored.step()
        assert_equal(opt.state_dict(), restored.state_dict())
        assert torch.equal(net.weight, other.weight)
    before = restored.state_dict()
    for corruption in ("dtype", "topology", "missing"):
        bad = deepcopy(saved)
        if corruption == "dtype":
            bad["masters"] = (bad["masters"][0].bfloat16(), *bad["masters"][1:])
        elif corruption == "topology":
            bad["identity"]["world_size"] = 2
        else:
            del bad["masters"]
        with pytest.raises(ValueError):
            restored.load_state_dict(bad)
        assert_equal(before, restored.state_dict())


def test_nonfinite_does_not_advance_master_moments_or_model():
    net = model()
    opt = make(net)
    net.weight.grad = torch.ones_like(net.weight)
    opt.step()
    before, weights = opt.state_dict(), deepcopy(net.state_dict())
    net.unused.grad = torch.full_like(net.unused, float("nan"))
    with pytest.raises(Exception, match="nonfinite"):
        opt.step()
    assert_equal(before, opt.state_dict())
    assert_equal(weights, net.state_dict())


def test_fp32_storage_is_already_the_master_and_rejects_rebinding():
    net = model().float()
    opt = make(net)
    assert opt.masters[0].data_ptr() == net.weight.data_ptr()
    assert opt.state_report()["master_bytes"] == 0
    net.weight.grad = torch.ones_like(net.weight)
    opt.step()
    saved = opt.state_dict()
    net.weight.data = net.weight.detach().clone()
    with pytest.raises(Exception, match="storage changed"):
        opt.step()
    # Reload rebinds the final live storage, with no second FP32 master allocation.
    opt.load_state_dict(saved)
    assert opt.masters[0].data_ptr() == net.weight.data_ptr()
    assert opt.state_report()["master_bytes"] == 0


def test_fused_control_memory_counts_initialized_step_tensors():
    net = model()
    optimizer = make(net)
    assert optimizer.state_report()["metadata_bytes"] == 0
    net.weight.grad = torch.ones_like(net.weight)
    optimizer.step()
    counters = [state["step"] for state in optimizer.backend.state.values()]
    assert len(counters) == 1  # The unused parameter must not invent a counter.
    expected = sum(value.numel() * value.element_size() for value in counters)
    assert expected > 0
    assert optimizer.state_report()["metadata_bytes"] == expected


def test_warm_start_load_after_construction_reseeds_masters_before_the_first_step():
    """Trainers may build the optimizer before a model-only (warm-start/EMA) load."""
    net = model()
    opt = make(net)
    loaded = torch.full_like(net.weight, 0.25)
    net.load_state_dict({**net.state_dict(), "weight": loaded})  # model-only load, optimizer untouched
    reference = torch.nn.Parameter(loaded.float().clone())
    control = torch.optim.AdamW([reference], lr=1e-4, weight_decay=0)
    for _ in range(3):
        net.weight.grad = torch.ones_like(net.weight)
        reference.grad = torch.ones_like(reference)
        opt.step()
        control.step()
        # The update starts from the loaded weights, never from the stale construction-time copy.
        assert torch.equal(opt.masters[0], reference)
        assert torch.equal(net.weight, reference.to(net.weight.dtype))


def test_model_only_load_under_initialized_moments_is_refused_without_mutation():
    net = model()
    opt = make(net)
    net.weight.grad = torch.ones_like(net.weight)
    opt.step()
    before = deepcopy(opt.state_dict())
    with torch.no_grad():
        net.weight.fill_(3)
    opt._verify_masters = True  # as after any re-binding; a later load must not be silently overwritten
    net.weight.grad = torch.ones_like(net.weight)
    with pytest.raises(Exception, match="model weights changed under an initialized"):
        opt.step()
    # A save would write masters that disagree with the saved model: refused the same way.
    with pytest.raises(ValueError, match="model weights changed under an initialized"):
        opt.state_dict()
    assert_equal(before["masters"], [master.detach() for master in opt.masters])
    assert_equal(before["backend"], opt.backend.state_dict())
    assert torch.equal(net.weight, torch.full_like(net.weight, 3))


def test_fp32_master_requires_the_trainers_finite_gradient_check():
    with pytest.raises(ValueError, match="fp32_master requires force_finite_grads=True"):
        OptimizerConfig(update_policy="fp32_master", force_finite_grads=False)
    OptimizerConfig(force_finite_grads=False)  # the legacy opt-out of the parameter policy is unchanged


def test_bf16_master_is_one_fp32_copy_and_fp32_storage_aliases():
    net = model()
    opt = make(net)
    assert opt.masters[0].dtype == torch.float32 and opt.masters[0].data_ptr() != net.weight.data_ptr()
    assert torch.equal(opt.masters[0], net.weight.float())
    fp32 = torch.nn.Linear(4, 3)
    assert all(m.data_ptr() == p.data_ptr() for m, p in zip(make(fp32).masters, fp32.parameters()))


def test_model_facing_state_is_rebuilt_only_when_backend_state_appears():
    net = model()
    opt = make(net)

    def step(*parameters):
        for parameter in parameters:
            parameter.grad = torch.ones_like(parameter)
        opt.step()

    step(net.weight)
    mapping = opt.state
    assert mapping[net.weight] is opt.backend.state[opt.masters[0]]
    step(net.weight)
    assert opt.state is mapping  # no new backend state: the mapping is reused, not rebuilt
    step(net.weight, net.unused)
    assert opt.state is not mapping and opt.state[net.unused] is opt.backend.state[opt.masters[1]]
