# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Real native factory/entry routing, with CPU backend doubles only at the CUDA boundary."""

import json
import sys
from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.posttrain.tests.helpers.optimizer_backend import backend_capture  # noqa: F401
from cosmos_framework.utils.optimizer_master import MasterWeightOptimizer


@pytest.mark.parametrize(
    "name,update_policy",
    [
        ("adamw", "parameter"),
        ("adamw", "fp32_master"),
        ("adamw_bnb_8bit", "fp32_master"),
        ("paged_adamw_8bit", "fp32_master"),
    ],
)
@pytest.mark.parametrize("override", [False, True])
def test_alpagym_actual_entry_constructs_after_setup(name, update_policy, override, monkeypatch, backend_capture):
    from cosmos_framework.posttrain.algorithm.primitives.objective import Objective, Reduction
    from cosmos_framework.posttrain.entrypoints.alpagym_clrl import _build_trainer, alpagym_optimizer_config

    class Model:
        def __init__(self, path, dtype):
            self.net = torch.nn.Linear(4, 4, dtype=dtype)
            self.before_setup = self.net.weight
            self.net.register_parameter("frozen", torch.nn.Parameter(torch.zeros(1), requires_grad=False))

        def setup(self, world_size):
            self.width = world_size
            self.net.weight = torch.nn.Parameter(self.net.weight.detach().clone())

    bundle = SimpleNamespace(install_runtime_bridge=lambda: None, build_model_inputs=lambda config: None)
    modules = {
        "alpagym_runtime.policies.registry": SimpleNamespace(get_policy_bundle=lambda _: bundle),
        "alpagym_runtime.posttrain.model": SimpleNamespace(AlpamayoTrainModel=Model),
        "alpagym_runtime.posttrain.objective": SimpleNamespace(
            make_alpagym_objective=lambda *a, **kw: Objective(
                lambda m, b: (m.net(b).sum(), {}), reduction=Reduction.MEAN
            )
        ),
    }
    for key, value in modules.items():
        monkeypatch.setitem(sys.modules, key, value)
    train = SimpleNamespace(
        optm_lr=1e-4,
        optm_warmup_steps=4,
        train_policy=SimpleNamespace(
            grpo_ratio_clip_low=0.2, grpo_ratio_clip_high=0.2, kl_beta=0, reference_reset_interval=0
        ),
    )
    config = SimpleNamespace(
        cosmos=SimpleNamespace(train=train),
        policy=SimpleNamespace(model=SimpleNamespace(kind="alpamayo", path="unallocated")),
    )
    selected = (
        None
        if update_policy == "parameter"
        else alpagym_optimizer_config(
            train, name=name, update_policy=update_policy, impl="fused" if name == "adamw" else ""
        )
    )
    if override:
        selected = selected or alpagym_optimizer_config(train)
        selected.lr, selected.warmup_steps, selected.grad_clip_norm = 6e-5, 2, 0.5
        if name != "adamw":
            selected.bnb_min_8bit_size = 8192
            selected.bnb_fp32_state_names = ["weight"]
    trainer = _build_trainer(config, 3, optimizer=selected)
    params = [p for g in trainer.opt.param_groups for p in g["params"]]
    assert {id(p) for p in params} == {id(p) for p in trainer.model.net.parameters() if p.requires_grad}
    assert all(p is not trainer.model.before_setup for p in params)
    group = trainer.opt.param_groups[0]
    assert (group["lr"], group["eps"], group["betas"], group["weight_decay"]) == (
        3e-5 if override else 2.5e-5,
        1e-6,
        (0.9, 0.999),
        0.01,
    )
    assert trainer.max_grad_norm == (0.5 if override else 1.0)
    assert trainer.n_mini == 3 and trainer.skip_zero_grad
    if update_policy == "parameter":
        assert type(trainer.opt) is torch.optim.AdamW and not backend_capture
    else:
        assert isinstance(trainer.opt, MasterWeightOptimizer)
        if name == "adamw":
            assert not backend_capture
            assert type(trainer.opt.backend) is torch.optim.AdamW
            assert all(g["fused"] for g in trainer.opt.backend.param_groups)
            assert all(p.dtype == torch.float32 for p in trainer.opt.masters)
            assert all(p.dtype == torch.bfloat16 for p in params)
        else:
            assert backend_capture[0][2].name == name
            assert backend_capture[0][2].bnb_min_8bit_size == (8192 if override else 4096)
            assert backend_capture[0][2].bnb_fp32_state_names == (["weight"] if override else [])


@pytest.mark.parametrize(
    "name,impl", [("adamw", ""), ("adamw", "fused"), ("adamw_bnb_8bit", ""), ("paged_adamw_8bit", "")]
)
def test_alpagym_cli_resolves_shared_state_options_without_allocating_model(name, impl, monkeypatch, tmp_path):
    from cosmos_framework.posttrain.entrypoints import alpagym_clrl as entry

    train = SimpleNamespace(
        optm_lr=2e-5,
        optm_warmup_steps=5,
        optm_decay_type="none",
        optm_decay_ratio=0.0,
        train_batch_per_replica=4,
    )
    config = SimpleNamespace(expected_valid_steps=8, cosmos=SimpleNamespace(train=train))
    monkeypatch.setitem(sys.modules, "alpagym_host.config", SimpleNamespace(load_run_config=lambda _: config))
    monkeypatch.setattr(entry.logging, "basicConfig", lambda **_: None)
    captured = []
    monkeypatch.setattr(entry, "run", captured.append)
    argv = [
        "alpagym",
        "--resolved-config",
        str(tmp_path / "resolved_config.yaml"),
        "--gpus",
        "4",
        "--steps",
        "2",
        "--group-size",
        "2",
    ]
    if name != "adamw":
        argv += [
            "--optimizer",
            name,
            "--optimizer-update-policy",
            "fp32_master",
            "--optimizer-bnb-min-8bit-size",
            "8192",
            "--optimizer-bnb-fp32-state-name",
            "weight",
            "--optimizer-bnb-fp32-state-name",
            "bias",
        ]
    if impl:
        argv += ["--optimizer-impl", impl, "--optimizer-update-policy", "fp32_master"]
    monkeypatch.setattr(sys, "argv", argv)
    # The host owns the resolved config's directory (here read-only); the run owns its checkpoints.
    (tmp_path / "checkpoints").mkdir()
    tmp_path.chmod(0o555)
    try:
        entry.main()
    finally:
        tmp_path.chmod(0o755)
    selected = captured[0].optimizer
    assert selected.name == name
    assert selected.impl == impl
    assert selected.update_policy == ("fp32_master" if impl or name != "adamw" else "parameter")
    assert selected.bnb_min_8bit_size == (4096 if name == "adamw" else 8192)
    assert selected.bnb_fp32_state_names == ([] if name == "adamw" else ["weight", "bias"])
    assert (selected.lr, selected.eps, selected.betas, selected.weight_decay) == (2e-5, 1e-6, [0.9, 0.999], 0.01)
    assert captured[0].prompts_per_step == 2 and captured[0].ticks_per_episode == 8
    # Written into the run's checkpoint directory, not next to the host-owned resolved config.
    assert not (tmp_path / "optimizer_config_resolved.json").exists()
    persisted = json.loads((tmp_path / "checkpoints" / "optimizer_config_resolved.json").read_text())
    assert persisted["name"] == name and persisted["impl"] == impl
    assert persisted["update_policy"] == selected.update_policy
    assert persisted["bnb_min_8bit_size"] == selected.bnb_min_8bit_size
    assert persisted["bnb_fp32_state_names"] == selected.bnb_fp32_state_names
    assert (persisted["lr"], persisted["eps"], persisted["betas"], persisted["weight_decay"]) == (
        2e-5,
        1e-6,
        [0.9, 0.999],
        0.01,
    )


@pytest.mark.parametrize(
    "options",
    [
        {"bnb_min_8bit_size": 0},
        {"bnb_fp32_state_names": ["weight", "weight"]},
        {"name": "adamw", "bnb_min_8bit_size": 8192},
        {"impl": "fused"},
    ],
)
def test_alpagym_state_options_use_shared_validation(options):
    from cosmos_framework.posttrain.entrypoints.alpagym_clrl import alpagym_optimizer_config

    train = SimpleNamespace(optm_lr=2e-5, optm_warmup_steps=5)
    with pytest.raises(ValueError):
        alpagym_optimizer_config(train, **{"name": "adamw_bnb_8bit", **options})


@pytest.mark.parametrize("name", ["adamw_bnb_8bit", "paged_adamw_8bit"])
def test_alpagym_cli_defaults_eight_bit_to_fp32_master_and_rejects_parameter_before_workers(
    name, monkeypatch, tmp_path, capsys
):
    from cosmos_framework.posttrain.entrypoints import alpagym_clrl as entry

    train = SimpleNamespace(
        optm_lr=2e-5, optm_warmup_steps=5, optm_decay_type="none", optm_decay_ratio=0.0, train_batch_per_replica=4
    )
    config = SimpleNamespace(expected_valid_steps=8, cosmos=SimpleNamespace(train=train))
    loads = []
    monkeypatch.setitem(
        sys.modules, "alpagym_host.config", SimpleNamespace(load_run_config=lambda _: loads.append(1) or config)
    )
    monkeypatch.setattr(entry.logging, "basicConfig", lambda **_: None)
    captured = []
    monkeypatch.setattr(entry, "run", captured.append)
    argv = ["alpagym", "--resolved-config", str(tmp_path / "r.yaml"), "--gpus", "4", "--steps", "2"]
    argv += ["--group-size", "2", "--optimizer", name]
    monkeypatch.setattr(sys, "argv", argv)
    entry.main()
    assert captured[0].optimizer.update_policy == "fp32_master"
    monkeypatch.setattr(sys, "argv", [*argv, "--optimizer-update-policy", "parameter"])
    with pytest.raises(SystemExit):
        entry.main()
    assert "requires --optimizer-update-policy fp32_master" in capsys.readouterr().err
    assert len(captured) == 1 and len(loads) == 1  # refused before reading the run or allocating workers


def test_optimizer_record_is_rewritten_by_every_launch(tmp_path):
    from cosmos_framework.posttrain.entrypoints.alpagym_clrl import alpagym_optimizer_config, record_optimizer_config

    train = SimpleNamespace(optm_lr=1e-4, optm_warmup_steps=1)
    eight_bit = alpagym_optimizer_config(train, name="adamw_bnb_8bit", update_policy="fp32_master")
    adamw = alpagym_optimizer_config(train)
    checkpoints = tmp_path / "checkpoints"
    record = checkpoints / "optimizer_config_resolved.json"
    record_optimizer_config(str(checkpoints), eight_bit)  # first launch: creates the directory and record
    assert json.loads(record.read_text())["name"] == "adamw_bnb_8bit"
    (checkpoints / "iter_000000008").mkdir()
    # The entry never resumes from its checkpoints, so a fresh launch with another optimizer is
    # legitimate (before the fix it was refused) and the record follows it.
    record_optimizer_config(str(checkpoints), adamw)
    assert json.loads(record.read_text())["name"] == "adamw"
