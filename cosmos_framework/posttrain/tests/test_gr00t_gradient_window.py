# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""GR00T construction/admission controls; CPU stand-ins do not qualify CUDA FSDP."""

import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("gr00t")
import transformers

from cosmos_framework.posttrain.model.gr00t import sft, sft_policy
from cosmos_framework.posttrain.model.gr00t.policy import OFFICIAL_PRECISION
from cosmos_framework.posttrain.roles.trainer.gradient_window import PROFILE
from cosmos_framework.posttrain.tests.test_gradient_window_builder import dimensions, rl_config

pytestmark = [
    pytest.mark.L0,
    pytest.mark.CPU,
    pytest.mark.skipif(transformers.__version__ != "4.57.6", reason="native GR00T binds Transformers 4.57.6"),
]


@pytest.fixture
def native(monkeypatch):
    from torch.distributed.algorithms._checkpoint import checkpoint_wrapper

    from cosmos_framework.posttrain.model import adapters, parallelize
    from cosmos_framework.posttrain.roles.trainer import gradient_window

    events = []

    class TinyNative(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2))
            self.alias = self.weight
            self.extra = torch.nn.Parameter(torch.ones(2))
            self.base = torch.nn.Parameter(torch.zeros(2), requires_grad=False)

    def load(*args):
        events.append("load")
        return TinyNative(), {"fixture": "native-gr00t"}

    def install_adapters(net, cfg, **kwargs):
        assert kwargs["family"] == "Gr00tN1d7" and kwargs["base_dtype"] == "float32"
        assert kwargs["source_identity"] == {"fixture": "native-gr00t"}
        events.append(("adapters", kwargs["resume"]))

    def shard(net, plan, pdims):
        events.append(("shard", plan.reduce_dtype, plan.param_dtype, pdims.dp_shard, pdims.dp_replicate))
        # A transformation can replace owners; the optimizer must use these.
        net.weight = torch.nn.Parameter(net.weight.detach().clone())
        net.alias = net.weight
        net._fixture_sharded = True

    def trainability(net, optimizer=None):
        if optimizer is not None:
            actual = [p for group in optimizer.param_groups for p in group["params"]]
            assert list(map(id, actual)) == [id(net.weight), id(net.extra)]
        events.append("trainability")

    def install(net, optimizer, *, microbatches):
        assert net._fixture_sharded and microbatches == 2
        events.append("policy")
        return SimpleNamespace(fixture=True)

    monkeypatch.setattr(sft_policy, "Gr00tSFTNet", TinyNative)
    monkeypatch.setattr(sft_policy, "load_base", load)
    monkeypatch.setattr(adapters, "prepare_adapter_model", install_adapters)
    monkeypatch.setattr(adapters, "assert_trainability", trainability)
    monkeypatch.setattr(adapters, "validate_adapter_parallelism", lambda _: None)
    monkeypatch.setattr(parallelize, "parallelize", shard)
    monkeypatch.setattr(gradient_window, "install_fsdp", install)
    monkeypatch.setattr(checkpoint_wrapper, "apply_activation_checkpointing", lambda *a, **kw: events.append("ac"))
    return events


@pytest.mark.parametrize("layout", [(4, 1), (2, 2)])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("resume,ac", [(False, "none"), (True, "full")])
def test_gr00t_registers_reducer_before_final_optimizer(native, layout, enabled, resume, ac):
    rl = rl_config(enabled)
    policy = sft.build_gr00t_sft_policy(
        "fixture",
        sft.Gr00tSFTModelConfig("fixture", ac),
        dimensions(*layout),
        adapters=rl.adapters,
        adapter_resume=resume,
        gradient_policy=rl.gradient_policy,
        device="cpu",
    )
    optimizer = torch.optim.AdamW([policy.net.weight, policy.net.extra], lr=1e-3)
    native.append("optimizer")
    expected = replace(OFFICIAL_PRECISION, reduce="float64") if enabled else OFFICIAL_PRECISION
    assert policy.precision == expected and policy.autocast is True
    assert policy.net.alias is policy.net.weight and not policy.net.base.requires_grad
    if enabled:
        assert sft.build_gradient_window(policy, optimizer, rl).fixture
    assert native == [
        "load",
        ("adapters", resume),
        *(["ac"] if ac == "full" else []),
        ("shard", torch.float64 if enabled else torch.float32, None, *layout),
        "trainability",
        "optimizer",
        *(["trainability", "policy"] if enabled else []),
    ]
    assert OFFICIAL_PRECISION.reduce == "float32"  # The default class preset is immutable.


@pytest.mark.parametrize("layout", [(1, 1), (2, 1), (1, 4), (8, 1)])
def test_unsupported_gr00t_topology_fails_before_materialization(native, layout):
    with pytest.raises(ValueError, match="FSDP4 or HSDP2x2"):
        sft.build_gr00t_sft_policy(
            "fixture",
            sft.Gr00tSFTModelConfig("fixture", "none"),
            dimensions(*layout),
            adapters=rl_config().adapters,
            gradient_policy=PROFILE,
            device="cpu",
        )
    assert not native


@pytest.mark.parametrize(
    "failure", ["policy", "net", "autocast", "reducer", "param", "master", "task", "adapters", "owners"]
)
def test_factory_rejects_incompatible_gr00t_before_install(native, failure):
    rl = rl_config()
    policy = sft.build_gr00t_sft_policy(
        "fixture",
        sft.Gr00tSFTModelConfig("fixture", "none"),
        dimensions(),
        adapters=rl.adapters,
        gradient_policy=PROFILE,
        device="cpu",
    )
    optimizer = torch.optim.AdamW([policy.net.weight, policy.net.extra], lr=1e-3)
    if failure == "policy":
        policy = SimpleNamespace(net=policy.net, precision=policy.precision, autocast=True)
    elif failure == "net":
        policy.net = torch.nn.Linear(2, 2)
    elif failure == "autocast":
        policy.autocast = False
    elif failure in ("reducer", "param", "master"):
        field = "reduce" if failure == "reducer" else failure
        policy.precision = replace(policy.precision, **{field: "float32" if field != "master" else "bfloat16"})
    elif failure == "task":
        rl.task = "conversation"
    elif failure == "adapters":
        rl.adapters = None
    else:
        optimizer.param_groups[0]["params"] = [policy.net.weight, policy.net.alias]
    with pytest.raises((ValueError, AssertionError)):
        sft.build_gradient_window(policy, optimizer, rl)
    assert "policy" not in native


@pytest.mark.parametrize("activation_checkpointing", ["none", "full"])
def test_real_gr00t_graph_global_objective_resume_and_portable_warm_start(
    tmp_path, monkeypatch, activation_checkpointing
):
    """Native stochastic forward/backward, real AdamW/DCP, simulated SUM hooks.

    This is one CPU process, not a substitute for the four-rank installed test.
    The independent branch owns its continuous parameters and optimizer; no
    update is rebased to the trainer. Only the collective transport is mocked.
    """
    from cosmos_framework.posttrain.algorithm.flow_bc import make_objective
    from cosmos_framework.posttrain.checkpoint.adapters import save_adapter_artifact
    from cosmos_framework.posttrain.config.adapters import AdapterExtraModuleConfig
    from cosmos_framework.posttrain.config.build import build_optimizer
    from cosmos_framework.posttrain.config.rl import OptimizerConfig
    from cosmos_framework.posttrain.model import parallelize
    from cosmos_framework.posttrain.model.adapters import assert_trainability, optimizer_groups
    from cosmos_framework.posttrain.roles.trainer import gradient_window
    from cosmos_framework.posttrain.roles.trainer.llm import LLMTrainer
    from cosmos_framework.posttrain.tests.test_gr00t_adapters import adapters, batch, prediction, source
    from cosmos_framework.posttrain.tests.test_gradient_window_trainer import assert_tree

    base, model_config = source(tmp_path)
    model_config.activation_checkpointing = activation_checkpointing
    adapter_config = adapters("lora")
    adapter_config.components[1].method = "pissa"
    adapter_config.components[1].execution = "pissa_dense_v1"
    adapter_config.components[1].rank = 3
    adapter_config.components[1].lr_multiplier = 0.5
    adapter_config.components[2].method = "dora"
    adapter_config.components[2].use_rslora = True
    adapter_config.components[3].dropout = 0.1
    adapter_config.extra_modules = [
        AdapterExtraModuleConfig(
            component="language",
            targets=["backbone.model.model.language_model.layers.0.input_layernorm"],
            lr_multiplier=0.25,
            weight_decay=0.1,
        )
    ]
    adapter_config.__post_init__()
    rl = rl_config()
    rl.adapters = adapter_config
    monkeypatch.setattr(parallelize, "parallelize", lambda *args: None)

    def simulated_transport(model, optimizer, *, microbatches):
        assert_trainability(model, optimizer)
        groups = {
            name: [gradient_window.Owner(name, p, tuple(p.shape), 1, 1, (name,))]
            for name, p in model.named_parameters()
            if p.requires_grad
        }
        window = gradient_window.OwnedGradientWindow(
            groups,
            microbatches=microbatches,
            world_size=1,
            count_sum=lambda n: n,
        )
        for name, owners in groups.items():

            def capture(gradient, group=name):
                window.hook(group)(gradient.detach().double().flatten())
                return torch.zeros_like(gradient)

            owners[0].parameter.register_hook(capture)
        return window

    monkeypatch.setattr(gradient_window, "install_fsdp", simulated_transport)

    def build(*, owned, warm=""):
        torch.manual_seed(195)
        cfg = copy.deepcopy(adapter_config)
        cfg.warm_start = warm
        policy = sft.build_gr00t_sft_policy(
            base,
            model_config,
            dimensions() if owned else None,
            adapters=cfg,
            gradient_policy=PROFILE if owned else None,
            device="cpu",
        )
        optimizer = build_optimizer(
            policy.net,
            OptimizerConfig(lr=1e-3, impl="for-loop"),
            parameter_groups=optimizer_groups(policy.net, 1e-3),
        )
        window = sft.build_gradient_window(policy, optimizer, rl) if owned else None
        return policy, optimizer, window

    policy, optimizer, window = build(owned=True)
    reference, independent_optimizer, _ = build(owned=False)
    before = {n: p.detach().clone() for n, p in policy.net.named_parameters()}
    assert_tree(policy.net.state_dict(), reference.net.state_dict())
    data = batch()
    inputs = []
    for steps in (2, 4, 1, 3, 4, 2):
        item = copy.deepcopy(data)
        item["action_mask"][:, steps:] = 0
        inputs.append(item)

    def trainer(model, opt, owned):
        objective = make_objective(lambda value: value, training_prefix=4, action_dim=7)
        objective.identity = {**objective.identity, "adapters": model.net._posttrain_adapter_identity}
        return LLMTrainer(
            model,
            objective,
            opt,
            grad_accum=2,
            require_gradients=True,
            max_grad_norm=1e6,
            gradient_window=owned,
            checkpoint_format="dcp",
            save_async=False,
        )

    primary = trainer(policy, optimizer, window)
    owners = dict(reference.net.named_parameters())
    owners = {n: p for n, p in owners.items() if p.requires_grad}
    checkpoint = str(tmp_path / "iter_000001")
    for update in range(3):
        numerators = {n: [] for n in owners}
        count = 0
        for item in inputs[update * 2 : update * 2 + 2]:
            rng = torch.get_rng_state()
            primary.train(item)
            after = torch.get_rng_state()
            torch.set_rng_state(rng)
            out = reference.flow_matching_loss(item)
            valid = int(out.action_mask.sum())
            derivatives = torch.autograd.grad(out.loss * valid, tuple(owners.values()))
            for (name, _), value in zip(owners.items(), derivatives, strict=True):
                numerators[name].append(value.double())
            count += valid
            torch.set_rng_state(after)
        for name, p in owners.items():
            p.grad = (sum(numerators[name]) / count).float()
        # The large explicit clip bound is inactive, avoiding a shared norm oracle.
        assert sum(p.grad.double().square().sum() for p in owners.values()).sqrt() < 1e6
        independent_optimizer.step()
        independent_optimizer.zero_grad(set_to_none=True)
        assert_tree(policy.net.state_dict(), reference.net.state_dict())
        assert_tree(optimizer.state_dict(), independent_optimizer.state_dict())
        assert window.completed_updates == primary.iteration() == update + 1
        if update == 0:
            primary.save(checkpoint, data_state={"next_microbatch": 2})
            primary.finalize_checkpoints()
    assert_trainability(policy.net, optimizer)
    for name, p in policy.net.named_parameters():
        if not p.requires_grad:
            assert torch.equal(p, before[name]), name  # PiSSA's initialized residual is the frozen base here.
    assert any(not torch.equal(p, before[n]) for n, p in policy.net.named_parameters() if p.requires_grad)
    restored, restored_opt, restored_window = build(owned=True)
    resumed = trainer(restored, restored_opt, restored_window)
    assert resumed.load(checkpoint) == 1
    for item in inputs[2:]:
        resumed.train(item)
    assert_tree(restored.net.state_dict(), policy.net.state_dict())
    assert_tree(restored_opt.state_dict(), optimizer.state_dict())
    assert_tree(restored_window.state_dict(), window.state_dict())
    portable = str(tmp_path / "portable")
    save_adapter_artifact(policy.net, portable)
    warm, warm_optimizer, warm_window = build(owned=True, warm=portable)
    assert warm_window.completed_updates == 0 and not warm_optimizer.state
    assert torch.equal(prediction(warm, data), prediction(policy, data))
