# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Unit tests for the FP32-master update owner (``update_policy=fp32_master``)."""

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from cosmos_framework.posttrain.config.build import build_optimizer
from cosmos_framework.posttrain.config.rl import OptimizerConfig


def _master(dtype=torch.bfloat16):
    torch.manual_seed(0)
    net = torch.nn.Linear(4, 3).to(dtype)
    return net, build_optimizer(net, OptimizerConfig(update_policy="fp32_master", lr=1e-2))


def _gradients(net, value=1.0):
    for parameter in net.parameters():
        parameter.grad = torch.full_like(parameter, value)


def test_skip_policy_leaves_every_state_untouched_and_reports_no_timing():
    net, opt = _master()
    _gradients(net)
    opt.step()
    opt.last_backend_step_events = ("stale", "events")
    before = opt.state_dict()
    model = [p.detach().clone() for p in net.parameters()]
    opt.nonfinite_policy = "skip"
    _gradients(net, float("nan"))
    assert opt.step() is None
    assert opt.skipped_nonfinite_steps == 1
    assert opt.last_backend_step_events is None
    after = opt.state_dict()
    assert all(torch.equal(a, b) for a, b in zip(before["masters"], after["masters"]))
    assert before["backend"]["state"][0]["step"] == after["backend"]["state"][0]["step"]
    assert all(torch.equal(a, b) for a, b in zip(model, net.parameters()))
    _gradients(net)
    opt.step()  # a later finite step proceeds normally
    assert after["backend"]["state"][0]["step"] < opt.state_dict()["backend"]["state"][0]["step"]


def test_raise_policy_is_the_posttrain_default():
    net, opt = _master()
    assert opt.nonfinite_policy == "raise"
    _gradients(net, float("inf"))
    with pytest.raises(ValueError, match="nonfinite"):
        opt.step()


def test_identity_omits_the_torch_version_and_records_it_as_metadata():
    _, opt = _master()
    assert "torch_version" not in opt.identity
    state = opt.state_dict()
    assert state["versions"] == {"torch": str(torch.__version__)}
    state["versions"] = {"torch": "0.0.0"}  # informational: a different torch still resumes exactly
    opt.load_state_dict(state)


def test_restore_validates_on_host_and_commits_into_the_live_tensors():
    net, opt = _master()
    _gradients(net)
    opt.step()
    saved = opt.state_dict()
    masters = opt.masters
    pointers = [m.data_ptr() for m in masters]
    _gradients(net)
    opt.step()
    staged = opt.stage_state_dict(saved)
    assert staged is saved and all(m.device.type == "cpu" for m in staged["masters"])
    opt.commit_state_dict(staged)
    assert opt.masters is masters and [m.data_ptr() for m in opt.masters] == pointers
    assert all(torch.equal(m, s) for m, s in zip(opt.masters, saved["masters"]))
    # (The model still holds the later step, so a save here would be refused; damage a copy instead.)
    damaged = {**saved, "masters": list(saved["masters"])}
    damaged["masters"] = (damaged["masters"][0][:1], *damaged["masters"][1:])
    with pytest.raises(ValueError, match="master weight"):
        opt.stage_state_dict(damaged)


def test_reset_state_returns_to_a_fresh_optimizer_seeded_from_the_model():
    net, opt = _master()
    _gradients(net)
    opt.step()
    with torch.no_grad():
        net.weight.add_(1)
    opt.reset_state()
    assert not opt.backend.state
    assert torch.equal(opt.masters[0].to(torch.bfloat16), net.weight)


class _CopiesInto(TorchDispatchMode):
    def __init__(self, pointers):
        super().__init__()
        self.pointers, self.count = pointers, 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func is torch.ops.aten.copy_.default and args[0].data_ptr() in self.pointers:
            self.count += 1
        return func(*args, **(kwargs or {}))


def test_fp32_shards_are_not_copied_back_onto_themselves():
    net, opt = _master(torch.float32)
    _gradients(net)
    before = net.weight.detach().clone()
    with _CopiesInto({p.data_ptr() for p in net.parameters()}) as mode:
        opt.step()
    assert mode.count == 0
    assert not torch.equal(before, net.weight)
    assert opt.masters[0].data_ptr() == net.weight.data_ptr()


def test_trainer_checked_steps_skip_the_owner_collective(monkeypatch):
    from cosmos_framework.posttrain.utils import dist

    net, opt = _master()
    calls = []
    real = dist.run_rank_converged
    monkeypatch.setattr(dist, "run_rank_converged", lambda fn, context: calls.append(context) or real(fn, context))
    _gradients(net)
    opt.note_gradients_checked_finite()
    opt.step()  # the first step after binding still verifies masters collectively
    assert len(calls) == 1
    _gradients(net)
    opt.note_gradients_checked_finite()
    opt.step()
    assert len(calls) == 1
    _gradients(net)
    opt.step()  # an unchecked step converges its own decision
    assert len(calls) == 2


def test_skip_policy_steady_state_decides_with_one_collective(monkeypatch):
    from cosmos_framework.posttrain.utils import dist

    net, opt = _master()
    opt.nonfinite_policy = "skip"
    converged, reduced = [], []
    real_converged, real_reduce = dist.run_rank_converged, dist.all_ranks_tensor_min
    monkeypatch.setattr(
        dist, "run_rank_converged", lambda fn, context: converged.append(context) or real_converged(fn, context)
    )
    monkeypatch.setattr(dist, "all_ranks_tensor_min", lambda flag: reduced.append(flag) or real_reduce(flag))
    _gradients(net)
    opt.step()  # first step after binding: converged staging (verifies masters) plus the decision
    assert (len(converged), len(reduced)) == (1, 1)
    for value in (1.0, float("nan"), 1.0):
        _gradients(net, value)
        opt.step()
    assert len(converged) == 1 and len(reduced) == 4  # steady state: one all-reduce per step, nothing else
    assert opt.skipped_nonfinite_steps == 1


def test_skip_policy_steady_state_staging_failure_is_raised_after_the_collective(monkeypatch):
    """A rank-local staging failure joins the step's single all-reduce, so its peers raise with it."""
    from cosmos_framework.posttrain.utils import dist

    real = dist.all_ranks_tensor_min
    reduced = []
    monkeypatch.setattr(dist, "all_ranks_tensor_min", lambda value: reduced.append(int(value)) or real(value))
    net, opt = _master(torch.float32)
    opt.nonfinite_policy = "skip"
    _gradients(net)
    opt.step()  # binds and allocates moments; steady state from here
    net.weight.data = net.weight.data.clone()  # this rank's FP32 storage rebinding, refused under moments
    _gradients(net)
    reduced.clear()
    with pytest.raises(ValueError, match="staging failed on at least one rank") as failing:
        opt.step()
    # Before the fix this rank raised before the all-reduce, leaving its peers waiting in it.
    assert reduced == [0]
    assert "FP32 model shard storage changed" in str(failing.value.__cause__)

    # A peer whose own staging succeeded learns of the failure from the same all-reduce.
    peer_net, peer = _master(torch.float32)
    peer.nonfinite_policy = "skip"
    _gradients(peer_net)
    peer.step()
    before = peer.state_dict()
    model = [p.detach().clone() for p in peer_net.parameters()]

    def failing_peer(value):
        reduced.append(int(value))
        return 0  # the MIN over ranks, one of which failed staging

    monkeypatch.setattr(dist, "all_ranks_tensor_min", failing_peer)
    _gradients(peer_net)
    reduced.clear()
    with pytest.raises(ValueError, match="staging failed on at least one rank"):
        peer.step()
    assert reduced == [2]  # finite and staged here; the reduced status (0) came from the failing rank
    assert all(torch.equal(a, b) for a, b in zip(model, peer_net.parameters()))
    assert before["backend"]["state"][0]["step"] == peer.state_dict()["backend"]["state"][0]["step"]


def test_save_before_the_first_step_writes_masters_of_the_loaded_model():
    """An iteration-0 save (``save_zero_checkpoint``) after a model-only load must be resumable."""
    net, opt = _master()
    torch.manual_seed(1)
    loaded = {name: torch.randn_like(value) for name, value in net.state_dict().items()}
    net.load_state_dict(loaded)  # warm start after the optimizer was built
    saved = opt.state_dict()
    for master, name in zip(saved["masters"], ("weight", "bias")):
        assert torch.equal(master.to(loaded[name].dtype), loaded[name])
    # The payload resumes into a fresh optimizer over the same loaded model.
    resumed_net, resumed = _master()
    resumed_net.load_state_dict(loaded)
    resumed.load_state_dict(saved)
    _gradients(resumed_net)
    resumed.step()


def test_save_refuses_masters_that_disagree_with_a_model_changed_under_moments():
    net, opt = _master()
    _gradients(net)
    opt.step()
    opt.load_state_dict(opt.state_dict())  # verification pending again, with moments
    with torch.no_grad():
        net.weight.add_(1)
    with pytest.raises(ValueError, match="changed under an initialized FP32-master optimizer"):
        opt.state_dict()


def test_reset_after_fp32_storage_rebinding_aliases_the_new_storage():
    net, opt = _master(torch.float32)
    _gradients(net)
    opt.step()
    for parameter in net.parameters():
        parameter.data = parameter.data.clone()  # the model rebinds its FP32 storage
    opt.reset_state()
    assert all(m.data_ptr() == p.data_ptr() for m, p in zip(opt.masters, net.parameters()))
    before = [p.detach().clone() for p in net.parameters()]
    _gradients(net)
    opt.step()  # before the fix: "FP32 model shard storage changed after optimizer binding"
    assert all(not torch.equal(a, p) for a, p in zip(before, net.parameters()))


def test_build_optimizer_duck_types_a_config_without_update_policy():
    from dataclasses import asdict
    from types import SimpleNamespace

    fields = {k: v for k, v in asdict(OptimizerConfig()).items() if k != "update_policy"}
    optimizer = build_optimizer(torch.nn.Linear(2, 2), SimpleNamespace(**fields))
    assert type(optimizer) is torch.optim.AdamW


def test_train_keys_matching_only_frozen_parameters_names_the_real_cause():
    net = torch.nn.Linear(4, 3).to(torch.bfloat16)
    net.weight.requires_grad_(False)
    with pytest.raises(ValueError, match="matched only parameters .* already frozen"):
        build_optimizer(net, OptimizerConfig(update_policy="fp32_master", train_keys=["weight"]))
    with pytest.raises(ValueError, match="matched no parameter"):
        build_optimizer(net, OptimizerConfig(update_policy="fp32_master", train_keys=["missing"]))


def test_hsdp_validation_outside_torchrun_leaves_inferred_replication_to_the_runtime_mesh(monkeypatch):
    from types import SimpleNamespace

    from cosmos_framework.posttrain.config.build import _effective_dp_replicate

    monkeypatch.delenv("WORLD_SIZE", raising=False)
    for shard in (-1, 1, 8):
        assert _effective_dp_replicate(SimpleNamespace(dp_replicate=-1, dp_shard=shard)) == 1
    assert _effective_dp_replicate(SimpleNamespace(dp_replicate=2, dp_shard=4)) == 2
    monkeypatch.setenv("WORLD_SIZE", "16")
    assert _effective_dp_replicate(SimpleNamespace(dp_replicate=-1, dp_shard=8)) == 2


def _rebind_fp32_storage(net, rebind):
    """New storage under the same Parameter objects, as a model load after binding may leave it."""
    loaded = {name: torch.randn_like(value) for name, value in net.named_parameters()}
    for name, parameter in net.named_parameters():
        if rebind == "swap":  # e.g. module._apply / load_state_dict(assign=True) with swapped tensors
            torch.utils.swap_tensors(parameter, torch.nn.Parameter(loaded[name].clone()))
        else:
            parameter.data = loaded[name].clone()
    return loaded


@pytest.mark.parametrize("rebind", ["swap", "data"])
def test_first_step_after_fp32_storage_rebinding_adopts_the_new_storage(rebind):
    net, opt = _master(torch.float32)
    loaded = _rebind_fp32_storage(net, rebind)
    _gradients(net)
    opt.step()  # before the fix: "FP32 model shard storage changed after optimizer binding"
    assert all(m.data_ptr() == p.data_ptr() for m, p in zip(opt.masters, net.parameters()))
    assert all(not torch.equal(p, loaded[name]) for name, p in net.named_parameters())  # the loaded values stepped
    # With moments, a rebinding is still refused.
    _rebind_fp32_storage(net, rebind)
    _gradients(net)
    with pytest.raises(ValueError, match="FP32 model shard storage changed"):
        opt.step()


def test_save_before_the_first_step_follows_rebound_fp32_storage():
    net, opt = _master(torch.float32)
    loaded = _rebind_fp32_storage(net, "data")
    saved = opt.state_dict()
    assert all(torch.equal(m, loaded[name]) for m, name in zip(saved["masters"], ("weight", "bias")))


@pytest.mark.parametrize(
    "lr",
    [lambda: 0.5, lambda: __import__("numpy").float32(0.5), lambda: torch.tensor(0.5, dtype=torch.float64)],
    ids=["float", "numpy", "tensor"],
)
def test_numeric_scalar_learning_rates_are_saved_as_floats_and_restored(lr):
    net, opt = _master()
    for group in opt.param_groups:
        group["lr"] = lr()  # e.g. a generator scheduler's NumPy or 0-dim tensor rate
    _gradients(net)
    opt.step()
    saved = opt.state_dict()
    assert all(type(group["lr"]) is float and group["lr"] == 0.5 for group in saved["backend"]["param_groups"])
    _, resumed = _master()
    for group in saved["backend"]["param_groups"]:
        group["lr"] = lr()  # a payload that still stores one is accepted too, as a float
    resumed.load_state_dict(saved)
    assert all(type(group["lr"]) is float and group["lr"] == 0.5 for group in resumed.param_groups)
    for invalid in (True, torch.tensor([0.5]), float("nan"), -1.0, "0.5"):
        saved["backend"]["param_groups"][0]["lr"] = invalid
        with pytest.raises(ValueError, match="learning rate"):
            resumed.stage_state_dict(saved)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_save_before_the_first_step_keeps_the_first_step_check_for_a_later_model_load(dtype):
    net, opt = _master(dtype)
    opt.state_dict()  # e.g. an iteration-0 checkpoint, before any moments exist
    loaded = {name: torch.full_like(value, 0.25) for name, value in net.state_dict().items()}
    net.load_state_dict(loaded)  # a model-only load after that save
    control_net, control = _master(dtype)
    control_net.load_state_dict(loaded)  # the same load, with no save in between
    for model, optimizer in ((net, opt), (control_net, control)):
        _gradients(model)
        optimizer.step()  # before the fix: the save's pre-load masters were written over the load
    assert all(torch.equal(a, b) for a, b in zip(net.parameters(), control_net.parameters()))
    assert all(torch.equal(a, b) for a, b in zip(opt.masters, control.masters))
    assert not all(torch.equal(p, loaded[name]) for name, p in net.named_parameters())  # it stepped


def test_state_log_line_has_counters_and_an_identity_digest_not_the_per_parameter_identity(monkeypatch):
    import logging

    lines = []
    monkeypatch.setattr(logging.getLogger("posttrain"), "info", lambda message, *args: lines.append(message % args))
    net, opt = _master()
    _gradients(net)
    opt.step()
    (line,) = [line for line in lines if "master optimizer state" in line]
    assert "master_bytes" in line and "identity_digest" in line
    # Before the fix the line carried the whole identity: every parameter's name, shape and layout.
    assert "weight" not in line and "local_shape" not in line
