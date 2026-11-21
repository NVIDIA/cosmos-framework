# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Real portable-IO dispatch contracts; tiny CPU fixtures do not qualify GR00T task behavior."""

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from cosmos_framework.posttrain.checkpoint.adapter import save_adapter_artifact
from cosmos_framework.posttrain.checkpoint.manager import CheckpointManager
from cosmos_framework.posttrain.entrypoints.gr00t_rollout import load_rollout_policy
from cosmos_framework.posttrain.model.checkpoint import AdapterCheckpointCodec
from cosmos_framework.posttrain.model.gr00t.qlora_source import recorded_source

MATCHED = "cosmos_framework.posttrain.model.gr00t.action_sft.build_policy"
OFFICIAL = "cosmos_framework.posttrain.model.gr00t.official_head.build_policy"
DENSE = "cosmos_framework.posttrain.model.gr00t.dense.build_policy"


def configuration(mode):
    return SimpleNamespace(policy_factory=OFFICIAL if mode == "official" else MATCHED, dense_policy_factory=DENSE)


def fixture(mode, *, mutate_identity=None):
    net = nn.Module()
    net.register_parameter("frozen", nn.Parameter(torch.ones(2), requires_grad=False))
    net.action_head = nn.Linear(2, 2)
    identity = {"base": recorded_source(), "processor": "fixed-native-processor"}
    if mode == "official":
        identity["format"] = "cosmos.gr00t-official-head.v1"
    else:
        identity.update(
            format="cosmos.family-backbone.v1", task_contract={"family": "gr00t_n1d7"}, adapter={"mode": mode}
        )
        if mode.endswith("lora"):
            net.register_parameter("lora_A", nn.Parameter(torch.ones(2, 2)))
            net.register_parameter("lora_B", nn.Parameter(torch.zeros(2, 2)))
    if mutate_identity:
        mutate_identity(identity)
    net._cosmos_adapter_parameters = {name: value for name, value in net.named_parameters() if value.requires_grad}
    codec = AdapterCheckpointCodec(net, identity=identity)
    net._cosmos_adapter_checkpoint_codec = codec
    return SimpleNamespace(net=net), codec


def portable(tmp_path, mode, *, mutate_identity=None):
    policy, codec = fixture(mode, mutate_identity=mutate_identity)
    for value in codec.parameters.values():
        with torch.no_grad():
            value.fill_(3.25)
    path = save_adapter_artifact(
        str(tmp_path / "checkpoints/iter_000003"),
        codec.parameters,
        codec.metadata(),
        manager=CheckpointManager(checkpoint_format="dcp"),
    )
    return path, copy.deepcopy(policy.net.state_dict())


@pytest.mark.parametrize("mode", ["official", "bf16_lora"])
def test_native_committed_checkpoint_loads_complete_policy_through_declared_factory(tmp_path, monkeypatch, mode):
    from cosmos_framework.posttrain.algorithm.primitives.objective import Objective, Reduction
    from cosmos_framework.posttrain.entrypoints import action_sft
    from cosmos_framework.posttrain.roles.trainer.llm import LLMTrainer
    from cosmos_framework.posttrain.utils.rank_state import collect_rank_state

    policy, codec = fixture(mode)
    optimizer = torch.optim.AdamW(list(codec.parameters.values()), lr=0.01)

    def objective(model, batch):
        return model.net.action_head(torch.ones(1, 2)).square().mean(), {}

    trainer = LLMTrainer(
        policy,
        Objective(objective, reduction=Reduction.MEAN, identity={"objective": "rollout-loading-fixture"}),
        optimizer,
        checkpoint_manager=CheckpointManager(checkpoint_format="dcp"),
    )
    trainer.train([None])
    checkpoint = str(tmp_path / "checkpoints/iter_000001")
    trainer.save(checkpoint, data_state=collect_rank_state({"cursor": 1, "source": "fixture-row"}))
    trainer.finalize_checkpoints()
    expected = copy.deepcopy(policy.net.state_dict())
    monkeypatch.setattr(action_sft, "_factory", lambda path: lambda *args, **kwargs: fixture(mode)[0])
    restored = load_rollout_policy(configuration(mode), checkpoint, device="cpu", output=tmp_path / "loaded")
    for name, tensor in restored.net.state_dict().items():
        torch.testing.assert_close(tensor, expected[name], rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["official", "nf4_lora", "bf16_lora", "nf4_frozen", "bf16_frozen"])
def test_all_native_policies_reload_exact_complete_portable_inventory(tmp_path, monkeypatch, mode):
    from cosmos_framework.posttrain.entrypoints import action_sft

    artifact, expected = portable(tmp_path, mode)
    config = configuration(mode)
    calls = []

    def factory(path):
        calls.append(path)
        assert path == config.policy_factory
        return lambda *args, **kwargs: fixture(mode)[0]

    monkeypatch.setattr(action_sft, "_factory", factory)
    policy = load_rollout_policy(config, artifact, device=torch.device("cpu"), output=tmp_path / "loaded")
    assert calls == [OFFICIAL if mode == "official" else MATCHED]
    assert not policy.net.training
    # These policies intentionally have no family_state. The common native codec
    # includes all trainable head tensors and, where configured, both LoRA factors.
    assert not hasattr(policy, "family_state")
    assert set(policy.net.state_dict()) == set(expected)
    for name, value in policy.net.state_dict().items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)


@pytest.mark.parametrize("failure", ["source", "format", "family", "uncommitted"])
def test_invalid_artifact_is_rejected_before_base_construction(tmp_path, monkeypatch, failure):
    from cosmos_framework.posttrain.entrypoints import action_sft

    def corrupt(identity):
        if failure == "source":
            identity["base"]["base"]["revision"] = "different"
        elif failure == "format":
            identity["format"] = "cosmos.gr00t-official-head.v1"
        elif failure == "family":
            identity["task_contract"]["family"] = "alpamayo"

    artifact, _ = portable(tmp_path, "nf4_lora", mutate_identity=corrupt)
    if failure == "uncommitted":
        artifact = str(tmp_path / "missing")
    monkeypatch.setattr(action_sft, "_factory", lambda *args: pytest.fail("invalid artifact allocated its base"))
    with pytest.raises((ValueError, RuntimeError, FileNotFoundError)):
        load_rollout_policy(configuration("nf4_lora"), artifact, device="cpu", output=tmp_path / "loaded")


@pytest.mark.parametrize("failure", ["mode", "processor", "missing_codec", "foreign_codec"])
def test_constructed_policy_requires_matching_numerical_identity_and_live_codec(tmp_path, monkeypatch, failure):
    from cosmos_framework.posttrain.entrypoints import action_sft

    artifact, _ = portable(tmp_path, "nf4_lora")
    policy, _ = fixture("bf16_lora" if failure == "mode" else "nf4_lora")
    if failure == "processor":
        policy.net._cosmos_adapter_checkpoint_codec.identity["processor"] = "changed"
    elif failure == "missing_codec":
        del policy.net._cosmos_adapter_checkpoint_codec
    elif failure == "foreign_codec":
        other, codec = fixture("nf4_lora")
        policy.net._cosmos_adapter_checkpoint_codec = codec
    monkeypatch.setattr(action_sft, "_factory", lambda path: lambda *args, **kwargs: policy)
    with pytest.raises((ValueError, RuntimeError)):
        load_rollout_policy(configuration("nf4_lora"), artifact, device="cpu", output=tmp_path / "loaded")


def test_dense_dispatch_remains_native_and_refuses_official_head_before_loading(tmp_path, monkeypatch):
    from cosmos_framework.posttrain.entrypoints import action_sft

    calls = []
    expected = object()

    def factory(path):
        calls.append(path)
        return lambda *args, **kwargs: expected

    monkeypatch.setattr(action_sft, "_factory", factory)
    assert (
        load_rollout_policy(configuration("nf4_lora"), "dense", device="cpu", output=tmp_path, dense=True) is expected
    )
    assert calls == [DENSE]
    with pytest.raises(ValueError, match="unavailable"):
        load_rollout_policy(configuration("official"), "dense", device="cpu", output=tmp_path, dense=True)
    assert calls == [DENSE]


def test_unmerged_dispatch_is_explicit_and_does_not_load_original_base_or_portable(tmp_path, monkeypatch):
    from cosmos_framework.posttrain.checkpoint import adapter
    from cosmos_framework.posttrain.entrypoints import action_sft

    calls, expected = [], object()

    def factory(path):
        assert path == DENSE

        def load(config, artifact, **kwargs):
            calls.append((artifact, kwargs))
            return expected

        return load

    monkeypatch.setattr(action_sft, "_factory", factory)
    monkeypatch.setattr(
        adapter, "read_adapter_metadata", lambda *a, **k: pytest.fail("unmerged read a portable artifact")
    )
    monkeypatch.setattr(adapter, "load_adapter_weights", lambda *a, **k: pytest.fail("unmerged loaded original base"))
    assert (
        load_rollout_policy(configuration("nf4_lora"), "artifact", device="cpu", output=tmp_path, unmerged=True)
        is expected
    )
    assert calls == [("artifact", {"device": "cpu", "output": tmp_path, "unmerged": True})]
    with pytest.raises(ValueError, match="unavailable"):
        load_rollout_policy(configuration("official"), "artifact", device="cpu", output=tmp_path, unmerged=True)
    for kwargs in ({"dense": True}, {"base": True}):
        with pytest.raises(ValueError, match="exactly one"):
            load_rollout_policy(
                configuration("nf4_lora"), "artifact", device="cpu", output=tmp_path, unmerged=True, **kwargs
            )


def test_unknown_policy_factory_fails_before_artifact_or_model_access(tmp_path):
    config = SimpleNamespace(policy_factory="unknown.policy", dense_policy_factory="")
    with pytest.raises(ValueError, match="explicit native"):
        load_rollout_policy(config, "missing", device="cpu", output=tmp_path)


@pytest.mark.parametrize("mode", ["official", "bf16_frozen", "nf4_frozen"])
def test_direct_source_keeps_native_weights_without_artifact_io(tmp_path, monkeypatch, mode):
    from cosmos_framework.posttrain.checkpoint import adapter
    from cosmos_framework.posttrain.entrypoints import action_sft

    config = configuration(mode)
    config.model = SimpleNamespace(backbone_adaptation=SimpleNamespace(mode=mode))
    policy, codec = fixture(mode)
    expected = copy.deepcopy(policy.net.state_dict())
    checked = []
    policy.net._cosmos_validate_frozen_base = lambda: checked.append(True)
    monkeypatch.setattr(action_sft, "_factory", lambda path: lambda *args, **kwargs: policy)
    monkeypatch.setattr(adapter, "read_adapter_metadata", lambda *a, **k: pytest.fail("base read an artifact"))
    monkeypatch.setattr(adapter, "load_adapter_weights", lambda *a, **k: pytest.fail("base loaded trained tensors"))
    restored = load_rollout_policy(config, base=True, device="cpu", output=tmp_path / "source")
    assert restored is policy and checked == [True] and not policy.net.training
    assert codec.identity["base"] == recorded_source()
    assert "training_contract" not in codec.identity
    for name, value in policy.net.state_dict().items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
    assert all(value.grad is None for value in policy.net.parameters())
    assert not (tmp_path / "source/checkpoints").exists()
    assert not (tmp_path / "source/checkpoint_artifacts").exists()


@pytest.mark.parametrize("mode", ["nf4_lora", "bf16_lora", None])
def test_source_refuses_adaptation_recipe_before_constructing_model(tmp_path, monkeypatch, mode):
    from cosmos_framework.posttrain.entrypoints import action_sft

    config = configuration("nf4_lora")
    config.model = SimpleNamespace(backbone_adaptation=SimpleNamespace(mode=mode))
    monkeypatch.setattr(action_sft, "_factory", lambda *args: pytest.fail("invalid source allocated its model"))
    with pytest.raises(ValueError, match="LoRA requires an artifact"):
        load_rollout_policy(config, base=True, device="cpu", output=tmp_path)


@pytest.mark.parametrize("failure", ["revision", "trained", "format", "mode"])
def test_source_rejects_incompatible_constructed_identity(tmp_path, monkeypatch, failure):
    from cosmos_framework.posttrain.entrypoints import action_sft

    config = configuration("nf4_frozen")
    config.model = SimpleNamespace(backbone_adaptation=SimpleNamespace(mode="nf4_frozen"))
    policy, codec = fixture("nf4_frozen")
    if failure == "revision":
        codec.identity["base"]["base"]["revision"] = "different"
    elif failure == "trained":
        codec.bind_training_identity({"data": "trained-source"})
    elif failure == "format":
        codec.identity["format"] = "cosmos.gr00t-official-head.v1"
    else:
        codec.identity["adapter"]["mode"] = "bf16_frozen"
    monkeypatch.setattr(action_sft, "_factory", lambda path: lambda *args, **kwargs: policy)
    with pytest.raises(ValueError, match="immutable untrained source"):
        load_rollout_policy(config, base=True, device="cpu", output=tmp_path)


@pytest.mark.parametrize("checkpoint,base,dense", [(None, False, False), ("artifact", True, False), (None, True, True)])
def test_source_selection_is_explicit_and_exclusive(tmp_path, checkpoint, base, dense):
    with pytest.raises(ValueError, match="exactly one"):
        load_rollout_policy(
            configuration("official"), checkpoint, base=base, dense=dense, device="cpu", output=tmp_path
        )


@pytest.mark.parametrize(
    "source_args",
    [
        [],
        ["--base", "--checkpoint", "artifact"],
        ["--base", "--dense"],
        ["--base", "--unmerged"],
        ["--checkpoint", "artifact", "--dense", "--unmerged"],
    ],
)
def test_cli_rejects_ambiguous_source_without_initializing_cuda(monkeypatch, source_args):
    import sys

    from cosmos_framework.posttrain.entrypoints import gr00t_rollout

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gr00t_rollout",
            "--config",
            "unused",
            "--sim-python",
            "unused",
            "--cases",
            "unused",
            "--seed",
            "17",
            "--output",
            "unused",
            *source_args,
        ],
    )
    with pytest.raises(SystemExit) as error:
        gr00t_rollout.main()
    assert error.value.code == 2
