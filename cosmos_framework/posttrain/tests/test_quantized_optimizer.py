# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
import copy
import json
import subprocess
import sys
from dataclasses import asdict

import pytest
import torch

from cosmos_framework.posttrain.config.build import build_lr_scheduler, build_optimizer
from cosmos_framework.posttrain.config.rl import OptimizerConfig


def test_config():
    assert OptimizerConfig("adamw").name == "adamw"
    assert asdict(OptimizerConfig(name="adamw_bnb_8bit"))["name"] == "adamw_bnb_8bit"
    with pytest.raises(ValueError):
        OptimizerConfig(name="adamw_8bit")
    for kwargs in ({"impl": "fused"}, {"bnb_min_8bit_size": 0}, {"bnb_fp32_state_names": ["x", "x"]}):
        with pytest.raises(ValueError):
            OptimizerConfig(name="adamw_bnb_8bit", **kwargs)


def test_adam_does_not_import_bnb():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, torch; from cosmos_framework.posttrain.config.build import build_optimizer; from cosmos_framework.posttrain.config.rl import OptimizerConfig; build_optimizer(torch.nn.Linear(2,2), OptimizerConfig()); assert 'bitsandbytes' not in sys.modules",
        ],
        check=True,
    )


@pytest.mark.parametrize("name", ["adamw_bnb_8bit", "paged_adamw_8bit"])
@pytest.mark.parametrize(
    "invalid",
    [
        {"lr": float("nan")},
        {"lr": -1.0},
        {"eps": float("inf")},
        {"eps": -1.0},
        {"weight_decay": -1.0},
        {"weight_decay": True},
        {"betas": [0.9]},
        {"betas": [0.9, 1.0]},
        {"betas": [0.9, float("nan")]},
    ],
)
def test_invalid_backend_hyperparameters_fail_in_config(name, invalid):
    with pytest.raises(ValueError, match="bitsandbytes"):
        OptimizerConfig(name=name, **invalid)
    # Zero learning rate/epsilon/decay and beta zero are supported backend values.
    OptimizerConfig(name=name, lr=0.0, eps=0.0, weight_decay=0.0, betas=[0.0, 0.99])


@pytest.mark.parametrize("name", ["adamw_bnb_8bit", "paged_adamw_8bit"])
def test_gpu_resume_and_membership(name, tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA kernel gate")
    torch.manual_seed(17)
    net = torch.nn.Linear(512, 256, device="cuda")
    net.register_parameter("unused", torch.nn.Parameter(torch.ones(8192, device="cuda")))
    net.register_parameter("frozen", torch.nn.Parameter(torch.ones(8192, device="cuda"), requires_grad=False))
    net.register_parameter("tied", net.weight)
    cfg = OptimizerConfig(
        name=name, lr=0.001, warmup_steps=3, lr_multipliers={"bias": 2.0}, bnb_fp32_state_names=["bias"]
    )
    opt = build_optimizer(net, cfg)
    scheduler = build_lr_scheduler(opt, cfg)
    assert len([p for g in opt.param_groups for p in g["params"]]) == 3
    assert not opt.state_dict()["state"]
    for _ in range(3):
        net.weight.grad = torch.ones_like(net.weight)
        net.bias.grad = torch.ones_like(net.bias)
        opt.step()
        scheduler.step()
    report = opt.state_report()
    assert report["quantized_elements"] == net.weight.numel()
    assert report["uninitialized_elements"] == net.unused.numel()
    assert bool(report["paged_moment_bytes"]) == (name == "paged_adamw_8bit")
    saved = opt.state_dict()
    before = net.weight.detach().clone()
    net.bias.grad.fill_(float("nan"))
    with pytest.raises(FloatingPointError):
        opt.step()
    assert torch.equal(before, net.weight)
    assert saved["state"][0]["step"] == opt.state[net.weight]["step"]
    net.bias.grad.fill_(1)
    ckpt = tmp_path / "resume.pt"
    torch.save({"net": net.state_dict(), "opt": saved, "scheduler": scheduler.state_dict(), "cfg": asdict(cfg)}, ckpt)
    # A separate interpreter exercises CPU deserialization -> managed allocation.
    code = """
import torch, sys
from cosmos_framework.posttrain.config.build import build_optimizer, build_lr_scheduler
from cosmos_framework.posttrain.config.rl import OptimizerConfig
s=torch.load(sys.argv[1], weights_only=False)
n=torch.nn.Linear(512,256,device='cuda')
n.register_parameter('unused',torch.nn.Parameter(torch.ones(8192,device='cuda')))
n.register_parameter('frozen',torch.nn.Parameter(torch.ones(8192,device='cuda'),requires_grad=False))
n.register_parameter('tied',n.weight)
n.load_state_dict(s['net'])
c=OptimizerConfig(**s['cfg']);o=build_optimizer(n,c);lr=build_lr_scheduler(o,c)
o.load_state_dict(s['opt']);lr.load_state_dict(s['scheduler'])
assert n.unused not in o.state
assert bool(o.state_report()['paged_moment_bytes']) == (c.name == 'paged_adamw_8bit')
for index in range(4):
 if index == 2: n.unused.grad=torch.ones_like(n.unused)
 n.weight.grad=torch.ones_like(n.weight);n.bias.grad=torch.ones_like(n.bias);o.step();lr.step()
torch.save({'net':n.state_dict(),'opt':o.state_dict(),'scheduler':lr.state_dict()},sys.argv[2])
"""
    output = tmp_path / "continued.pt"
    subprocess.run([sys.executable, "-c", code, str(ckpt), str(output)], check=True)
    for index in range(4):
        if index == 2:
            net.unused.grad = torch.ones_like(net.unused)
        opt.step()
        scheduler.step()
    resumed = torch.load(output, weights_only=False)
    assert torch.equal(net.weight.cpu(), resumed["net"]["weight"].cpu())
    assert torch.equal(net.bias.cpu(), resumed["net"]["bias"].cpu())
    assert torch.equal(net.unused.cpu(), resumed["net"]["unused"].cpu())
    assert opt.state[net.unused]["step"] == 2
    assert scheduler.state_dict() == resumed["scheduler"]
    assert saved["state"][0]["step"] == 3  # snapshot remains immutable
    broken = copy.deepcopy(saved)
    broken["cosmos_quantized_optimizer"]["name"] = "adamw"
    with pytest.raises(ValueError, match="identity"):
        opt.load_state_dict(broken)
    changed = copy.deepcopy(cfg)
    changed.lr *= 2
    with pytest.raises(ValueError, match="identity"):
        build_optimizer(net, changed).load_state_dict(saved)
    (tmp_path / "report.json").write_text(json.dumps(report))


def test_reject_cpu_and_missing_selector():
    cfg = OptimizerConfig(name="adamw_bnb_8bit")
    with pytest.raises(ValueError, match="CUDA FP32"):
        build_optimizer(torch.nn.Linear(4, 4), cfg)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="bitsandbytes eight-bit AdamW requires CUDA")
@pytest.mark.parametrize("name", ["adamw_bnb_8bit", "paged_adamw_8bit"])
def test_step_syncs_at_most_once_and_matches_upstream_numerics(name, monkeypatch):
    import bitsandbytes.optim.optimizer as upstream

    def build():
        torch.manual_seed(0)
        net = torch.nn.Sequential(torch.nn.Linear(512, 256), torch.nn.Linear(256, 4)).cuda()  # 8-bit and FP32 moments
        net.extra = torch.nn.Parameter(torch.linspace(-1, 1, 131072, device="cuda"))  # paged when the name pages
        return net, build_optimizer(net, OptimizerConfig(name=name, lr=1e-3))

    (ours_net, ours), (reference_net, reference) = build(), build()
    syncs = []
    real = torch.cuda.synchronize
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: syncs.append(device) or real(device))
    for step in range(3):
        for net in (ours_net, reference_net):
            for index, parameter in enumerate(net.parameters()):
                parameter.grad = torch.full_like(parameter, 0.1 * (step + 1)).sin_().mul_(index + 1)
        syncs.clear()
        ours.step()
        if step:  # the first step also allocates state; filling managed buffers synchronizes once each
            assert len(syncs) == (1 if name == "paged_adamw_8bit" else 0)
        upstream.Optimizer8bit.step(reference)  # bitsandbytes' own loop, one device sync per parameter
        assert len(syncs) > 3
    torch.cuda.synchronize()
    assert all(torch.equal(a, b) for a, b in zip(ours_net.parameters(), reference_net.parameters()))
    from cosmos_framework.utils.quantized_optimizer import device_view

    for a, b in zip(ours.state.values(), reference.state.values()):
        for key in a:
            if isinstance(a[key], torch.Tensor):
                assert torch.equal(device_view(a[key]), device_view(b[key])), key
            else:
                assert a[key] == b[key], key


def test_cuda_paging_preserves_updates_with_large_fp32_exception():
    if not torch.cuda.is_available():
        pytest.skip("real CUDA required")
    count = 131072
    initial = torch.linspace(-0.2, 0.3, count, device="cuda")
    models, optimizers = [], []
    for name in ("adamw_bnb_8bit", "paged_adamw_8bit"):
        model = torch.nn.Module()
        model.register_parameter("quantized", torch.nn.Parameter(initial.clone()))
        model.register_parameter("exception", torch.nn.Parameter(initial.clone()))
        optimizer = build_optimizer(model, OptimizerConfig(name=name, lr=1e-3, bnb_fp32_state_names=["exception"]))
        models.append(model)
        optimizers.append(optimizer)
    for step in range(6):
        if step == 4:
            # Rebuild both uint8 and FP32 managed arrays, without a model update.
            optimizers[1].load_state_dict(optimizers[1].state_dict())
        gradient = (initial * (step + 1)).cos()
        for model, optimizer in zip(models, optimizers):
            for parameter in model.parameters():
                parameter.grad = gradient.clone()
            optimizer.step()
        assert all(torch.equal(a, b) for a, b in zip(models[0].parameters(), models[1].parameters()))
    report = optimizers[1].state_report()
    assert report["quantized_elements"] == report["fp32_elements"] == count
    assert report["paged_moment_bytes"] == count * (2 + 8)
    assert report["exceptions"] == {"exception": "explicit"}


def test_missing_dependency(monkeypatch):
    import importlib.metadata

    from cosmos_framework.utils.quantized_optimizer import require_backend

    def missing(_):
        raise importlib.metadata.PackageNotFoundError("bitsandbytes")

    monkeypatch.setattr(importlib.metadata, "version", missing)
    with pytest.raises(ImportError, match="quantized-optimizers"):
        require_backend()


def test_backend_class_does_not_own_parameters_after_instance_release(monkeypatch):
    """A cached optimizer class must not retain a retired model/master allocation."""
    import weakref
    from contextlib import nullcontext
    from types import SimpleNamespace

    import cosmos_framework.utils.quantized_optimizer as backend

    class Parameter(torch.nn.Parameter):
        @property
        def device(self):
            return torch.device("cuda:0")

    class Upstream(torch.optim.Optimizer):
        def __init__(self, groups, **options):
            super().__init__(groups, options)

    monkeypatch.setattr(torch.nn, "Parameter", Parameter)
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(
        backend, "require_backend", lambda **_: SimpleNamespace(optim=SimpleNamespace(AdamW8bit=Upstream))
    )

    def construct():
        parameter = Parameter(torch.ones(8))
        net = SimpleNamespace(named_parameters=lambda: iter((("weight", parameter),)))
        optimizer = backend.build_quantized_optimizer(net, [parameter], OptimizerConfig(name="adamw_bnb_8bit"))
        return optimizer, weakref.ref(parameter)

    optimizer, parameter = construct()
    retained_class = type(optimizer)
    del optimizer
    # No gc.collect(): old GPU storage must be freed at ownership replacement,
    # even when tooling retains the class or Python's cyclic collector is delayed.
    assert parameter() is None
    other, _ = construct()
    assert type(other) is retained_class


def test_cuda_exceptions_are_local_and_restore_is_transactional():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    a = torch.nn.Linear(128, 128, device="cuda")
    b = torch.nn.Linear(128, 128, device="cuda")
    first = build_optimizer(a, OptimizerConfig(name="adamw_bnb_8bit", bnb_fp32_state_names=["weight"]))
    second = build_optimizer(b, OptimizerConfig(name="adamw_bnb_8bit"))
    for net, opt in [(a, first), (b, second)]:
        net.weight.grad = torch.ones_like(net.weight)
        net.bias.grad = torch.ones_like(net.bias)
        opt.step()
    assert first.state[a.weight]["state1"].dtype == torch.float32
    assert second.state[b.weight]["state1"].dtype == torch.uint8
    assert second.state[b.bias]["state1"].dtype == torch.float32
    assert second.state_report()["exceptions"]["bias"] == "small_tensor"
    saved = second.state_dict()
    broken = copy.deepcopy(saved)
    broken["state"][0]["absmax1"] = torch.zeros(1)
    previous = second.state
    with pytest.raises(ValueError, match="shape/dtype"):
        second.load_state_dict(broken)
    assert second.state is previous
    b.weight.grad = torch.ones_like(b.weight).to_sparse()
    with pytest.raises(ValueError, match="sparse"):
        second.step()
    assert second.state[b.weight]["step"] == 1
    for cfg in [
        OptimizerConfig(name="adamw_bnb_8bit", bnb_fp32_state_names=["missing"]),
        OptimizerConfig(name="adamw_bnb_8bit", bnb_fp32_state_names=["bias", "missing"]),
    ]:
        with pytest.raises(ValueError, match="unmatched"):
            build_optimizer(b, cfg)
    with pytest.raises(ValueError, match="FP32"):
        build_optimizer(b.to(torch.bfloat16), OptimizerConfig(name="adamw_bnb_8bit"))


@pytest.mark.parametrize("name", ["adamw_bnb_8bit", "paged_adamw_8bit"])
@pytest.mark.parametrize("failure", ["moment_copy", "group_snapshot"])
def test_failed_restore_keeps_live_values_and_next_update(name, failure, monkeypatch):
    import cosmos_framework.utils.quantized_optimizer as backend

    if not torch.cuda.is_available():
        pytest.skip("CUDA allocation and copy failure gate")
    net = torch.nn.Linear(512, 256, bias=False, device="cuda")
    control_net = copy.deepcopy(net)
    cfg = OptimizerConfig(name=name, lr=0.001)
    opt, control = (build_optimizer(model, cfg) for model in (net, control_net))
    for index in range(2):
        for model, optimizer in ((net, opt), (control_net, control)):
            model.weight.grad = torch.full_like(model.weight, float(index + 1))
            optimizer.step()
        if index == 0:
            saved = opt.state_dict()
    state, groups, pages = opt.state, opt.param_groups, opt.page_mng.paged_tensors
    before = opt.state_dict()
    with monkeypatch.context() as patch:
        if failure == "moment_copy":
            original = torch.Tensor.copy_

            def fail_copy(target, value, *args, **kwargs):
                result = original(target, value, *args, **kwargs)
                if target.dtype == torch.uint8 and target.numel() == net.weight.numel():
                    raise RuntimeError("injected moment copy failure after writing replacement storage")
                return result

            patch.setattr(torch.Tensor, "copy_", fail_copy)
        else:
            original = backend._snapshot

            def fail_snapshot(value):
                if value is saved["param_groups"][0]:
                    raise RuntimeError("injected group snapshot failure")
                return original(value)

            patch.setattr(backend, "_snapshot", fail_snapshot)
        with pytest.raises(RuntimeError, match="injected"):
            opt.load_state_dict(saved)
    assert opt.state is state and opt.param_groups is groups and opt.page_mng.paged_tensors is pages
    after = opt.state_dict()
    for key, value in before["state"][0].items():
        actual = after["state"][0][key]
        assert torch.equal(value, actual) if isinstance(value, torch.Tensor) else value == actual
    assert before["param_groups"] == after["param_groups"]
    for model, optimizer in ((net, opt), (control_net, control)):
        model.weight.grad = torch.full_like(model.weight, 3.0)
        optimizer.step()
    assert torch.equal(net.weight, control_net.weight)
    for key, value in opt.state_dict()["state"][0].items():
        actual = control.state_dict()["state"][0][key]
        assert torch.equal(value, actual) if isinstance(value, torch.Tensor) else value == actual


def test_paged_buffer_lifetime_follows_storage_aliases(monkeypatch):
    import gc

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from bitsandbytes.cextension import lib

    from cosmos_framework.utils import quantized_optimizer as backend

    monkeypatch.setattr(backend, "_PROBED", set())  # exercise the (normally cached) scratch probe too
    actual_free = lib._lib.cudaFree
    results = []

    def recorded_free(pointer):
        result = actual_free(pointer)
        results.append(result)
        return result

    monkeypatch.setattr(lib._lib, "cudaFree", recorded_free)
    net = torch.nn.Linear(512, 256, device="cuda", bias=False)
    opt = build_optimizer(net, OptimizerConfig(name="paged_adamw_8bit"))
    net.weight.grad = torch.ones_like(net.weight)
    opt.step()
    buffer = opt.state[net.weight]["state1"]
    owner = buffer._cosmos_managed_owner
    alias = buffer.detach()
    saved = opt.state_dict()
    assert not hasattr(saved["state"][0]["state1"], "is_paged")
    del buffer, opt
    gc.collect()
    assert owner() is not None
    # The detached storage alias is still valid after optimizer destruction.
    torch.testing.assert_close(alias, saved["state"][0]["state1"])
    del alias
    gc.collect()
    assert owner() is None
    assert len(results) >= 4  # scratch preflight + two live moments
    assert results == [0] * len(results)  # real cudaFree succeeded for every allocation


def test_backend_probe_runs_once_per_device(monkeypatch):
    """Entry preflight and optimizer construction share one cached CUDA probe per device/mode."""
    from cosmos_framework.utils import quantized_optimizer as backend

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    backend._PROBED.clear()
    backend.require_backend()
    probes = []
    original = backend._owned_state_buffer
    monkeypatch.setattr(backend, "_owned_state_buffer", lambda *a, **k: probes.append(1) or original(*a, **k))
    backend.require_backend()
    assert probes == []


def test_empty_and_noncontiguous_trainables_fail():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    for tensor in (torch.empty(0, device="cuda"), torch.ones(8, 16, device="cuda").T):
        net = torch.nn.Module()
        net.register_parameter("weight", torch.nn.Parameter(tensor))
        with pytest.raises(ValueError, match="nonempty contiguous"):
            build_optimizer(net, OptimizerConfig(name="adamw_bnb_8bit"))


@pytest.mark.parametrize("name", ["adamw_bnb_8bit", "paged_adamw_8bit"])
def test_mainline_bf16_base_lora_uses_only_fp32_adapters(name):
    from cosmos_framework.utils.generator.lora import LoraInjectedLinear

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(29)
    base = torch.nn.Linear(1024, 1024, bias=False, dtype=torch.bfloat16, device="cuda")
    base.weight.requires_grad_(False)
    net = LoraInjectedLinear(base, 128, 128, adapter_dtype=torch.float32)
    torch.nn.init.zeros_(net.lora_B.weight)
    frozen = net.weight.detach().clone()
    cfg = OptimizerConfig(name=name, lr=1e-3)
    opt = build_optimizer(net, cfg)
    assert {id(p) for group in opt.param_groups for p in group["params"]} == {
        id(net.lora_A.weight),
        id(net.lora_B.weight),
    }
    x = torch.randn(4, 16, 1024, device="cuda", dtype=torch.bfloat16)
    for _ in range(5):
        opt.zero_grad()
        net(x).float().square().mean().backward()
        opt.step()
    assert torch.equal(frozen, net.weight)
    assert net.lora_B.weight.count_nonzero() > 0
    assert opt.state_report()["quantized_elements"] == 262144
    assert bool(opt.state_report()["paged_moment_bytes"]) == (name == "paged_adamw_8bit")


def test_parameter_names_match_model_state_dict_under_checkpoint_and_compile_wrappers():
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
    from torch.distributed.checkpoint.state_dict import get_model_state_dict

    from cosmos_framework.utils.quantized_optimizer import named_parameter_groups

    net = torch.nn.Sequential(checkpoint_wrapper(torch.nn.Linear(2, 2)), torch.nn.Linear(2, 1))
    compiled = torch.compile(net)
    for model in (net, compiled):
        raw = [name for name, _ in model.named_parameters()]
        assert any("_checkpoint_wrapped_module" in name for name in raw)
        groups, _, names = named_parameter_groups(model, list(model.parameters()), context="test")
        assert (
            groups[0]["param_names"]
            == list(get_model_state_dict(model))
            == ["0.weight", "0.bias", "1.weight", "1.bias"]
        )
        assert set(names.values()) == set(groups[0]["param_names"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="bitsandbytes eight-bit AdamW requires CUDA")
def test_state_report_is_logged_when_state_is_allocated_not_rescanned_every_step(monkeypatch):
    net = torch.nn.Linear(64, 64).cuda()
    opt = build_optimizer(net, OptimizerConfig(name="adamw_bnb_8bit", bnb_min_8bit_size=1))
    reports = []
    monkeypatch.setattr(opt, "state_report", lambda: reports.append(len(opt.state)) or {})

    def step(*parameters):
        for parameter in parameters:
            parameter.grad = torch.ones_like(parameter)
        opt.step()
        opt.zero_grad()

    step(net.weight)
    step(net.weight)
    assert reports == [1]  # first allocation only; an unchanged state set is not reported again
    step(net.weight, net.bias)
    assert reports == [1, 2]  # init_state for the bias marks the report stale
    opt.reset_state()
    step(net.weight)
    assert reports == [1, 2, 1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="bitsandbytes eight-bit AdamW requires CUDA")
def test_restore_reads_the_constant_quantization_maps_once_not_per_parameter(monkeypatch):
    net = torch.nn.Sequential(*[torch.nn.Linear(64, 64) for _ in range(3)]).cuda()
    opt = build_optimizer(net, OptimizerConfig(name="adamw_bnb_8bit", bnb_min_8bit_size=1))
    for parameter in net.parameters():
        parameter.grad = torch.ones_like(parameter)
    opt.step()
    saved = opt.state_dict()
    assert sum("qmap1" in state for state in saved["state"].values()) == 6
    calls = []
    real = type(opt)._host_quantization_maps
    monkeypatch.setattr(type(opt), "_host_quantization_maps", lambda self: calls.append(1) or real(self))
    opt.load_state_dict(saved)
    assert calls == [1]  # one host copy of the shared tables for all six eight-bit parameters
    broken = copy.deepcopy(saved)
    broken["state"][5]["qmap2"] = broken["state"][5]["qmap2"].flip(0)
    with pytest.raises(ValueError, match="quantization map mismatch"):
        opt.load_state_dict(broken)
