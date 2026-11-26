# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Dead or diagnostic tensors must not occupy accelerator memory across a Joint PPO update.

Regression for the fixed 55 GiB GR00T pressure continuation (round 10): the nonpaged
eight-bit learner ran out of memory in backward while (a) a full pre-update clone of every
local model shard and (b) a consumed FP32 master-gradient copy were device-resident.
"""

from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.posttrain.config.build import build_optimizer
from cosmos_framework.posttrain.config.rl import JointPPOConfig, OptimizerConfig
from cosmos_framework.posttrain.roles.trainer import joint_ppo
from cosmos_framework.posttrain.roles.trainer.joint_ppo import JointPPOTrainer


def _bf16_net(device="cpu", width=4):
    net = torch.nn.Sequential(torch.nn.Linear(width, width, bias=False), torch.nn.Linear(width, 3, bias=False))
    return net.to(device=device, dtype=torch.bfloat16)


@pytest.mark.L0
@pytest.mark.CPU
def test_master_step_releases_consumed_fp32_staging_but_keeps_model_gradients():
    net = _bf16_net()
    opt = build_optimizer(net, OptimizerConfig(lr=1e-3, weight_decay=0, update_policy="fp32_master"))
    for parameter in net.parameters():
        parameter.grad = torch.ones_like(parameter)
    opt.step()
    report = opt.state_report()
    # Before the fix the FP32 copy (4 bytes/element) survived until the next window.
    assert report["gradient_staging_bytes"] == 0
    assert all(parameter.grad is not None for parameter in net.parameters())
    assert all(master.grad is None for master in opt.masters)
    # A second step stages afresh; the update arithmetic is unchanged.
    before = [master.clone() for master in opt.masters]
    for parameter in net.parameters():
        parameter.grad = torch.ones_like(parameter)
    opt.step()
    assert all(not torch.equal(a, b) for a, b in zip(before, opt.masters))


@pytest.mark.L0
@pytest.mark.CPU
def test_fp32_alias_gradient_is_not_released_by_master_step():
    net = _bf16_net().float()
    opt = build_optimizer(net, OptimizerConfig(lr=1e-3, weight_decay=0, update_policy="fp32_master"))
    for parameter in net.parameters():
        parameter.grad = torch.ones_like(parameter)
    opt.step()
    assert opt.state_report()["fp32_gradient_alias_bytes"] > 0


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("host", [False, True])
def test_pre_update_snapshot_is_an_independent_copy(host):
    net = _bf16_net()
    owner = SimpleNamespace(model=SimpleNamespace(net=net), _host_snapshot_buffers={})
    with JointPPOTrainer._parameter_state(owner, clone=True, host=host) as before:
        assert before and all(value.device.type == "cpu" for value in before.values())
        live = {p.data_ptr() for p in net.parameters()}
        assert not live & {value.data_ptr() for value in before.values()}
        with torch.no_grad():
            for parameter in net.parameters():
                parameter.add_(1)
        assert all(
            not torch.equal(value, parameter.detach()) for value, parameter in zip(before.values(), net.parameters())
        )


@pytest.mark.L0
@pytest.mark.CPU
def test_allocator_receipt_is_absent_off_device_and_update_still_reports_it(monkeypatch):
    from cosmos_framework.posttrain.tests.helpers.ppo_continuous import (
        GaussianPolicy,
        LocalWindowGateway,
        collect_scored,
        make_collector,
    )

    assert joint_ppo._cuda_allocator_window(_bf16_net(), reset=True) is None
    windows = []
    real_window = joint_ppo._cuda_allocator_window
    monkeypatch.setattr(
        joint_ppo, "_cuda_allocator_window", lambda net, reset: windows.append(reset) or real_window(net, reset=reset)
    )
    for update_policy, receipts in (("parameter", False), ("fp32_master", False), ("fp32_master", True)):
        windows.clear()
        recipe = JointPPOConfig(
            num_envs=2, window_decisions=2, global_minibatch=2, microbatch=1, cuda_allocator_receipts=receipts
        )
        recipe.optimizer.update_policy = update_policy
        model = GaussianPolicy()
        collector = make_collector(model, LocalWindowGateway(), recipe, 0, 2)
        learner = JointPPOTrainer(model, recipe)
        receipt = learner.prepare_ppo_targets(collect_scored(collector, 0))
        result = learner.train_batch(receipt["batch_id"])
        # Without the explicit diagnostics switch (any optimizer) the receipt keeps the upstream
        # shape: no allocator reset, gather or key.
        assert ("cuda_allocator" in result) is receipts
        assert result.get("cuda_allocator") is None
        # Only the switch touches (and resets) the process-wide CUDA peak counters.
        assert windows == ([True, False] if receipts else [])
        assert result["parameter_update_l2"]["actor"] > 0 and result["parameter_update_l2"]["value"] > 0
        collector.close()
    with pytest.raises(ValueError, match="cuda_allocator_receipts must be a boolean"):
        JointPPOConfig(cuda_allocator_receipts=1)


@pytest.mark.L0
@pytest.mark.GPU
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_pre_update_snapshot_occupies_no_device_memory_during_the_update():
    device = torch.device("cuda")
    net = _bf16_net(device, width=2048)  # 2 x 8 MiB of BF16 parameters
    owner = SimpleNamespace(model=SimpleNamespace(net=net), _host_snapshot_buffers={})
    parameter_bytes = sum(p.numel() * p.element_size() for p in net.parameters())
    torch.cuda.synchronize(device)
    baseline = torch.cuda.memory_allocated(device)
    original = [p.detach().clone() for p in net.parameters()]
    with JointPPOTrainer._parameter_state(owner, clone=True, host=True) as before:
        held = torch.cuda.memory_allocated(device) - baseline - parameter_bytes  # minus `original`
        assert all(value.device.type == "cpu" and value.is_pinned() for value in before.values())
        # The asynchronous copies are stream-ordered before later in-place updates.
        with torch.no_grad():
            for parameter in net.parameters():
                parameter.add_(1)
        restored = [value.to(device, non_blocking=True) for value in before.values()]
        assert all(torch.equal(a, b) for a, b in zip(restored, original))
    # Before the fix `held` equalled every local shard (the clone was device-resident).
    assert held < parameter_bytes // 4
    # The default (non-master) path keeps the historical device-resident clone.
    with JointPPOTrainer._parameter_state(owner, clone=True) as before:
        assert all(value.is_cuda for value in before.values())


@pytest.mark.L0
@pytest.mark.GPU
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_master_step_returns_fp32_staging_to_the_device_allocator():
    device = torch.device("cuda")
    net = _bf16_net(device, width=2048)
    opt = build_optimizer(net, OptimizerConfig(lr=1e-3, weight_decay=0, update_policy="fp32_master"))
    for parameter in net.parameters():
        parameter.grad = torch.ones_like(parameter)
    opt.step()  # first step initializes moments
    for parameter in net.parameters():
        parameter.grad = torch.ones_like(parameter)
    torch.cuda.synchronize(device)
    before_step = torch.cuda.memory_allocated(device)
    opt.step()
    torch.cuda.synchronize(device)
    staging_bytes = sum(p.numel() * 4 for p in net.parameters())
    assert torch.cuda.memory_allocated(device) - before_step < staging_bytes // 4
    receipt = joint_ppo._cuda_allocator_window(net, reset=True)
    assert receipt is not None
    assert {"inactive_split_bytes.all.current", "reserved_bytes.all.peak", "device_free_bytes"} <= set(receipt)


@pytest.mark.L0
@pytest.mark.GPU
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_backend_step_device_time_is_recorded_without_changing_arithmetic():
    device = torch.device("cuda")
    nets = [_bf16_net(device, width=512) for _ in range(2)]
    nets[1].load_state_dict(nets[0].state_dict())
    opts = [build_optimizer(net, OptimizerConfig(lr=1e-3, weight_decay=0, update_policy="fp32_master")) for net in nets]
    for net, opt in zip(nets, opts):
        for parameter in net.parameters():
            parameter.grad = torch.full_like(parameter, 0.5)
        opt.step()
    assert joint_ppo._backend_step_milliseconds(opts[0]) > 0
    for left, right in zip(nets[0].parameters(), nets[1].parameters()):
        assert torch.equal(left, right)


@pytest.mark.L0
@pytest.mark.CPU
def test_backend_step_time_is_absent_on_cpu():
    net = _bf16_net()
    opt = build_optimizer(net, OptimizerConfig(lr=1e-3, weight_decay=0, update_policy="fp32_master"))
    for parameter in net.parameters():
        parameter.grad = torch.ones_like(parameter)
    opt.step()
    assert joint_ppo._backend_step_milliseconds(opt) is None


@pytest.mark.L0
@pytest.mark.CPU
def test_master_pre_update_snapshot_copies_only_trainable_parameters(monkeypatch):
    from contextlib import contextmanager

    from cosmos_framework.posttrain.tests.helpers.ppo_continuous import (
        GaussianPolicy,
        LocalWindowGateway,
        collect_scored,
        make_collector,
    )
    from cosmos_framework.utils import quantized_optimizer

    net = _bf16_net()
    net[0].weight.requires_grad_(False)  # a frozen backbone stand-in
    owner = SimpleNamespace(model=SimpleNamespace(net=net), _host_snapshot_buffers={})
    with JointPPOTrainer._parameter_state(owner, clone=True, host=True, names=frozenset({"1.weight"})) as before:
        assert set(before) == {"1.weight"}

    snapshots = []
    real = JointPPOTrainer._parameter_state

    @contextmanager
    def spy(self, **kwargs):
        with real(self, **kwargs) as state:
            if kwargs.get("clone"):
                snapshots.append(set(state))
            yield state

    monkeypatch.setattr(JointPPOTrainer, "_parameter_state", spy)
    for update_policy in ("parameter", "fp32_master"):
        snapshots.clear()
        recipe = JointPPOConfig(num_envs=2, window_decisions=2, global_minibatch=2, microbatch=1)
        recipe.optimizer.update_policy = update_policy
        model = GaussianPolicy()
        model.net.register_buffer("frozen_statistic", torch.zeros(3))
        collector = make_collector(model, LocalWindowGateway(), recipe, 0, 2)
        learner = JointPPOTrainer(model, recipe)
        receipt = learner.prepare_ppo_targets(collect_scored(collector, 0))
        with monkeypatch.context() as patch:
            # The trainable name set is fixed at construction, not recomputed every update.
            patch.setattr(
                quantized_optimizer, "canonical_parameter_names", lambda net: pytest.fail("recomputed per update")
            )
            result = learner.train_batch(receipt["batch_id"])
        trainable = {name for name, p in model.net.named_parameters() if p.requires_grad}
        # The default path keeps its historical whole-state-dict snapshot; the master path copies
        # only the trainable PPO parameters, which are all the update-norm diagnostics read.
        expected = trainable if update_policy == "fp32_master" else set(model.net.state_dict())
        assert snapshots == [expected] and "frozen_statistic" not in trainable
        assert result["parameter_update_l2"]["actor"] > 0 and result["parameter_update_l2"]["value"] > 0
        collector.close()


@pytest.mark.L0
@pytest.mark.GPU
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_post_step_finite_check_reads_paged_fp32_moments_on_the_device(monkeypatch):
    from cosmos_framework.posttrain.roles.trainer.finite import all_ranks_model_optimizer_finite
    from cosmos_framework.utils.quantized_optimizer import device_view

    net = _bf16_net("cuda", width=512)  # 0.weight: 262,144 elements, above the 100k paging floor
    config = OptimizerConfig(
        name="paged_adamw_8bit", update_policy="fp32_master", bnb_min_8bit_size=1, bnb_fp32_state_names=["0.weight"]
    )
    opt = build_optimizer(net, config)
    for parameter in net.parameters():
        parameter.grad = torch.ones_like(parameter)
    opt.step()
    paged = opt.state[net[0].weight]["state1"]
    assert paged.is_paged and paged.dtype == torch.float32 and paged.device.type == "cpu"
    devices = []
    real = torch.isfinite
    monkeypatch.setattr(torch, "isfinite", lambda tensor: devices.append(tensor.device.type) or real(tensor))
    assert all_ranks_model_optimizer_finite(net, opt)
    # Before the fix the managed moment was scanned on the host, migrating its pages every step.
    assert devices and "cpu" not in devices
    device_view(paged)[0, 0] = float("nan")
    assert not all_ranks_model_optimizer_finite(net, opt)


@pytest.mark.L0
@pytest.mark.CPU
def test_master_snapshot_names_must_be_model_state_dict_keys():
    net = _bf16_net()
    assert joint_ppo._update_snapshot_names(SimpleNamespace(names=("0.weight", "1.weight")), net) == {
        "0.weight",
        "1.weight",
    }
    # Before the check, a mismatch silently emptied the pre-update snapshot and the update check
    # then reported that the update changed nothing.
    with pytest.raises(ValueError, match=r"\['net.0.weight'\] are not state-dict keys"):
        joint_ppo._update_snapshot_names(SimpleNamespace(names=("net.0.weight", "1.weight")), net)


def _evaluation_owner(net, opt):
    owner = SimpleNamespace(
        model=SimpleNamespace(net=net, ppo_frozen_state=lambda: {}),
        opt=opt,
        lr_scheduler=None,
        optimizer_step=1,
        publication_version=lambda: 1,
        _frozen_model_state={},
    )
    owner._parameter_state = lambda **kwargs: JointPPOTrainer._parameter_state(owner, **kwargs)
    return owner


@pytest.mark.L0
@pytest.mark.CPU
def test_evaluation_digest_marks_fp32_master_aliases_instead_of_rehashing_them():
    from cosmos_framework.posttrain.algorithm.ppo.wire import canonical_digest

    net = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 3).to(torch.bfloat16))
    opt = build_optimizer(net, OptimizerConfig(lr=1e-3, update_policy="fp32_master"))
    state = JointPPOTrainer.evaluation_state(_evaluation_owner(net, opt))
    expected = [
        "fp32-parameter-alias" if live.dtype == torch.float32 else canonical_digest(master.detach())
        for live, master in zip(opt.live, opt.masters)
    ]
    assert expected.count("fp32-parameter-alias") == 2  # the FP32 Linear's weight and bias
    assert state["optimizer_masters"] == canonical_digest(expected)


@pytest.mark.L0
@pytest.mark.GPU
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_evaluation_digest_reads_paged_state_on_the_device_with_the_same_digest(monkeypatch):
    from cosmos_framework.posttrain.algorithm.ppo.wire import canonical_digest
    from cosmos_framework.utils.quantized_optimizer import device_view

    net = _bf16_net("cuda", width=512)
    opt = build_optimizer(net, OptimizerConfig(name="paged_adamw_8bit", update_policy="fp32_master"))
    for parameter in net.parameters():
        parameter.grad = torch.ones_like(parameter)
    opt.step()
    paged = [value for state in opt.state.values() for value in state.values() if getattr(value, "is_paged", False)]
    assert paged
    expected = canonical_digest(
        [
            canonical_digest(
                {
                    key: canonical_digest(device_view(value).cpu() if getattr(value, "is_paged", False) else value)
                    for key, value in state.items()
                }
            )
            for state in opt.state.values()
        ]
    )
    hashed, synchronized = [], []
    monkeypatch.setattr(joint_ppo, "canonical_digest", lambda value: hashed.append(value) or canonical_digest(value))
    real_synchronize = torch.cuda.synchronize
    monkeypatch.setattr(
        torch.cuda, "synchronize", lambda device=None: synchronized.append(device) or real_synchronize(device)
    )
    state = JointPPOTrainer.evaluation_state(_evaluation_owner(net, opt))
    # Before the fix the managed moments were hashed on the host, migrating their pages there.
    assert not any(getattr(value, "is_paged", False) for value in hashed)
    assert state["optimizer"] == expected
    assert len(paged) > 1 and len(synchronized) == 1  # one wait per device, not one per paged tensor


@pytest.mark.L0
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.GPU)])
def test_host_pre_update_snapshot_reuses_its_buffers_across_updates(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    net = _bf16_net(device)
    owner = SimpleNamespace(model=SimpleNamespace(net=net), _host_snapshot_buffers={})
    pointers = []
    for update in range(3):
        with torch.no_grad():
            for parameter in net.parameters():
                parameter.fill_(update)
        with JointPPOTrainer._parameter_state(owner, clone=True, host=True) as before:
            if device == "cuda":
                torch.cuda.synchronize()
                assert all(value.is_pinned() for value in before.values())
            # Each update's snapshot holds that update's values, in the same host buffers.
            assert all(torch.equal(value, torch.full_like(value, update)) for value in before.values())
            pointers.append({name: value.data_ptr() for name, value in before.items()})
    assert pointers[0] == pointers[1] == pointers[2]
