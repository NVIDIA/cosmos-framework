# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Bounded raw reuse and actual pinned native-head semantics; no CUDA speed claim."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from cosmos_framework.posttrain.utils import gr00t_evaluation_backbone as reuse


def batch():
    return [SimpleNamespace(transitions=[object()])]


def inputs():
    return {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
        "attention_mask": torch.ones(1, 5, dtype=torch.long),
        "pixel_values": torch.arange(16).reshape(2, 8).bfloat16(),
        "image_grid_thw": torch.ones(2, 3, dtype=torch.long),
    }


def raw():
    return {
        "backbone_features": torch.arange(40).reshape(1, 5, 8).bfloat16() / 40,
        "backbone_attention_mask": torch.ones(1, 5, dtype=torch.bool),
        "image_mask": torch.tensor([[True, True, False, False, False]]),
    }


def state():
    return reuse.EvaluationBackboneReuse(SimpleNamespace(gr00t=nn.Linear(1, 1).eval()))


@pytest.mark.parametrize("mode", ["nf4_lora", "bf16_lora", "nf4_frozen", "bf16_frozen", "unknown", None])
def test_only_explicit_native_lora_modes_are_admitted(mode):
    prefix = "cosmos_framework.posttrain.model.gr00t.action_sft."
    config = SimpleNamespace(
        policy_factory=prefix + "build_policy",
        data_factory=prefix + "build_dataset",
        objective_factory=prefix + "build_objective",
        evaluation_callback=prefix + "evaluate_actions",
        model=SimpleNamespace(
            backbone_adaptation=None
            if mode is None
            else SimpleNamespace(mode=mode, profile="gr00t_n1d7_reason2_2b_layer16_v1")
        ),
    )
    rl = SimpleNamespace(reuse_gr00t_evaluation_backbone=True, micro_batch_size=1, reuse_evaluation_inputs=False)
    if mode in {"nf4_lora", "bf16_lora"}:
        reuse.validate_reuse(config, rl)
    else:
        with pytest.raises(ValueError, match="pinned native"):
            reuse.validate_reuse(config, rl)


def test_disabled_path_needs_no_gr00t_configuration_or_state():
    reuse.validate_reuse(object(), SimpleNamespace(reuse_gr00t_evaluation_backbone=False))
    policy = object()
    with reuse.batch_scope(policy, object()):
        pass
    assert reuse.observations(policy) is None


@pytest.mark.parametrize("kind", ["dense", "unmerged"])
def test_unsupported_standalone_loading_fails_before_cuda_or_model(monkeypatch, kind, tmp_path):
    from cosmos_framework.posttrain.configs.gr00t_n1d7_libero_qlora import make_config
    from cosmos_framework.posttrain.entrypoints import action_eval

    config = make_config()
    config.rl.reuse_gr00t_evaluation_backbone = True

    def forbidden(*args, **kwargs):
        pytest.fail("unsupported reuse must fail before expensive CUDA/model construction")

    with monkeypatch.context() as patched:
        patched.setattr(torch.cuda, "is_available", forbidden)
        with pytest.raises(ValueError, match="does not support dense or unmerged"):
            action_eval.run(config, checkpoint="unchanged-artifact", output=str(tmp_path / "absent"), **{kind: True})
    assert not (tmp_path / "absent").exists()


@pytest.mark.parametrize("change", ["dtype", "mask", "key", "grad"])
def test_unsupported_raw_outputs_fail_instead_of_caching(change):
    cache = state()
    with pytest.raises(ValueError):
        with torch.no_grad(), cache.batch_scope(batch()):
            prepared, value = inputs(), raw()
            cache.begin_loss(prepared)
            if change == "dtype":
                value["backbone_features"] = value["backbone_features"].float()
            elif change == "mask":
                value["image_mask"] = value["image_mask"].long()
            elif change == "key":
                value["unexpected"] = torch.ones(1)
            else:
                value["backbone_features"].requires_grad_(True)
            cache.finish_loss(prepared, value)
    assert cache.inputs is cache.raw is None and not cache.active


def test_raw_storage_and_both_head_mappings_have_independent_ownership():
    cache = state()
    prepared, original = inputs(), raw()
    with torch.no_grad(), cache.batch_scope(batch()):
        rng = torch.random.get_rng_state().clone()
        cache.begin_loss(prepared)
        first = cache.finish_loss(prepared, original)
        original["backbone_features"].add_(3)
        first["backbone_features"].mul_(5)
        first["image_mask"].zero_()
        second = cache.prediction(inputs())
        assert all(torch.equal(second[key], value) for key, value in raw().items())
        assert first is not second and second["backbone_features"].data_ptr() != first["backbone_features"].data_ptr()
        assert torch.equal(rng, torch.random.get_rng_state())
        assert cache.inputs is cache.raw is None
    assert not cache.active and cache.stats["misses"] == cache.stats["hits"] == 1


@pytest.mark.parametrize("key", reuse.INPUT_KEYS)
@pytest.mark.parametrize("change", ["value", "dtype", "shape"])
def test_actual_prepared_input_changes_fail_and_clear_cache(key, change):
    cache = state()
    with pytest.raises(ValueError, match="prepared backbone inputs (differ|require)"):
        with torch.no_grad(), cache.batch_scope(batch()):
            before = inputs()
            cache.begin_loss(before)
            cache.finish_loss(before, raw())
            after = inputs()
            if change == "value":
                after[key].view(-1)[0] += 1
            elif change == "dtype":
                after[key] = after[key].float()
            else:
                after[key] = after[key].flatten()
            cache.prediction(after)
    assert cache.inputs is cache.raw is None and not cache.active


def test_backbone_inplace_input_mutation_is_rejected():
    cache = state()
    with pytest.raises(ValueError, match="mutated its prepared input"):
        with torch.no_grad(), cache.batch_scope(batch()):
            value = inputs()
            cache.begin_loss(value)
            value["pixel_values"].zero_()
            cache.finish_loss(value, raw())
    assert cache.inputs is cache.raw is None


@pytest.mark.parametrize("phase", ["before_loss", "after_loss", "after_prediction"])
def test_any_native_failure_clears_state_without_consuming_rng(phase):
    cache = state()
    rng = torch.random.get_rng_state().clone()
    with pytest.raises(LookupError, match="native failure"):
        with torch.no_grad(), cache.batch_scope(batch()):
            if phase != "before_loss":
                value = inputs()
                cache.begin_loss(value)
                cache.finish_loss(value, raw())
            if phase == "after_prediction":
                cache.prediction(inputs())
            raise LookupError("native failure")
    assert cache.inputs is cache.raw is None and not cache.active
    assert torch.equal(rng, torch.random.get_rng_state())


@pytest.mark.parametrize("violation", ["grad", "train", "nest", "missing_prediction", "prediction_first"])
def test_scope_is_no_grad_eval_only_and_requires_exact_call_order(violation):
    cache = state()
    if violation == "train":
        cache.policy.gr00t.train()
    with pytest.raises(RuntimeError):
        with nullcontext() if violation == "grad" else torch.no_grad(), cache.batch_scope(batch()):
            if violation == "nest":
                with cache.batch_scope(batch()):
                    pass
            elif violation == "prediction_first":
                cache.prediction(inputs())
            elif violation == "missing_prediction":
                value = inputs()
                cache.begin_loss(value)
                cache.finish_loss(value, raw())
    assert cache.inputs is cache.raw is None and not cache.active


@pytest.mark.parametrize("phase", ["inputs", "raw"])
def test_cap_is_checked_before_retaining_oversized_storage(monkeypatch, phase):
    cache = state()
    limit = sum(x.numel() * x.element_size() for x in inputs().values())
    monkeypatch.setattr(reuse, "MAX_RETAINED_BYTES", limit - 1 if phase == "inputs" else limit)
    with pytest.raises(ValueError, match="exceed 16 MiB"):
        with torch.no_grad(), cache.batch_scope(batch()):
            value = inputs()
            cache.begin_loss(value)
            cache.finish_loss(value, raw())
    assert cache.inputs is cache.raw is None and cache.stats["retained_tensor_bytes_peak"] == 0


def native_head():
    pytest.importorskip("gr00t")
    from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead

    config = Gr00tN1d7Config(
        hidden_size=16,
        input_embedding_dim=16,
        backbone_embedding_dim=8,
        max_state_dim=4,
        max_action_dim=3,
        action_horizon=4,
        max_num_embodiments=1,
        max_seq_len=16,
        num_timestep_buckets=8,
        num_inference_timesteps=4,
        vl_self_attention_cfg={"num_layers": 0},
        diffusion_model_cfg={
            "positional_embeddings": None,
            "num_layers": 2,
            "num_attention_heads": 2,
            "attention_head_dim": 8,
            "norm_type": "ada_norm",
            "dropout": 0.2,
            "final_dropout": True,
            "output_dim": 16,
            "interleave_self_attention": True,
        },
    )
    return Gr00tN1d7ActionHead(config).eval()


def test_actual_pinned_head_loss_and_sampler_match_with_raw_reuse():
    from transformers.feature_extraction_utils import BatchFeature

    head = native_head()
    action = {
        "state": torch.tensor([[[0.123456, -0.234567, 0.345678, 0.456789]]]),
        "action": torch.linspace(-0.5, 0.5, 12).reshape(1, 4, 3),
        "action_mask": torch.ones(1, 4, 3),
        "embodiment_id": torch.zeros(1, dtype=torch.long),
    }

    def action_input(sample=False):
        return BatchFeature(
            data={k: v.clone() for k, v in action.items() if not sample or k not in {"action", "action_mask"}}
        )

    cache = state()
    rng = torch.random.get_rng_state().clone()
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        original_mapping = BatchFeature(data=raw())
        original_feature = original_mapping["backbone_features"]
        expected_loss = head(original_mapping, action_input())
        assert original_mapping["backbone_features"] is not original_feature
        expected_prediction = head.get_action(BatchFeature(data=raw()), action_input(True))
        expected_rng = torch.random.get_rng_state().clone()
        torch.random.set_rng_state(rng)
        with cache.batch_scope(batch()):
            prepared = inputs()
            cache.begin_loss(prepared)
            actual_loss = head(cache.finish_loss(prepared, raw()), action_input())
            actual_prediction = head.get_action(cache.prediction(inputs()), action_input(True))
        assert torch.equal(torch.random.get_rng_state(), expected_rng)
    assert all(torch.equal(actual_loss[k], v) for k, v in expected_loss.items())
    assert all(torch.equal(actual_prediction[k], v) for k, v in expected_prediction.items())


def native_policy():
    pytest.importorskip("gr00t")
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7
    from transformers.feature_extraction_utils import BatchFeature

    from cosmos_framework.posttrain.configs.gr00t_n1d7_libero_qlora import make_config
    from cosmos_framework.posttrain.model.gr00t.action_sft import Gr00tActionPacker
    from cosmos_framework.posttrain.model.gr00t.adaptation import prepare_task_inputs
    from cosmos_framework.posttrain.model.gr00t.policy import OFFICIAL_PRECISION, Gr00tN1d7TrainPolicy

    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(8, 8).bfloat16().requires_grad_(False)
            self.calls = 0

        def prepare_input(self, value):
            return BatchFeature(data=value)

        def forward(self, value):
            self.calls += 1
            return BatchFeature(
                data={
                    "backbone_features": self.embedding(value["input_ids"]) + value["pixel_values"].mean(),
                    "backbone_attention_mask": value["attention_mask"] == 1,
                    "image_mask": value["input_ids"] < 3,
                }
            )

    class Native(Gr00tN1d7):
        def __init__(self):
            nn.Module.__init__(self)
            self.backbone = Backbone()
            self.action_head = native_head()
            self.prepared_actions = []

        def prepare_input(self, value):
            backbone, action = prepare_task_inputs(self, value, super().prepare_input)
            self.prepared_actions.append(
                {k: action[k].clone() for k in ("state", "action", "action_mask") if k in action}
            )
            return backbone, action

    config = make_config()
    policy = object.__new__(Gr00tN1d7TrainPolicy)
    policy.config, policy.precision, policy._ppo_enabled = config.model, OFFICIAL_PRECISION, False
    policy.gr00t = policy.net = Native()
    policy.backbone = policy.net.backbone
    policy.compute_ctx = torch.autocast("cpu", dtype=torch.bfloat16)
    policy.family_state = SimpleNamespace()
    policy.packer = object.__new__(Gr00tActionPacker)
    return policy, config


def task_inputs():
    return {
        **inputs(),
        "state": torch.tensor([[[0.123456, -0.234567, 0.345678, 0.456789]]]),
        "action": torch.linspace(-0.5, 0.5, 12).reshape(1, 4, 3),
        "action_mask": torch.ones(1, 4, 3),
        "embodiment_id": torch.zeros(1, dtype=torch.long),
    }


def objective(policy, _batch):
    value = policy.flow_matching_loss(task_inputs())
    return value.loss, {}


def callback(policy, _batch, _config):
    value = {k: v for k, v in task_inputs().items() if k not in {"action", "action_mask"}}
    prediction = policy.predict_actions(value)["action_pred"]
    return {"prediction": {"sum": float(prediction.float().sum()), "count": prediction.numel()}}


def test_native_evaluator_off_on_exact_and_patched_prepare_keeps_fp32():
    from cosmos_framework.posttrain.checkpoint.rng import collect_rng_state
    from cosmos_framework.posttrain.entrypoints.action_sft import evaluate
    from cosmos_framework.posttrain.tests.test_action_sft import assert_state_equal

    policy, config = native_policy()
    rows = [SimpleNamespace(id=str(i), transitions=[object()]) for i in range(2)]
    before = collect_rng_state()
    expected = evaluate(policy, objective, rows, config, callback)
    assert policy.backbone.calls == 4
    config.rl.reuse_gr00t_evaluation_backbone = True
    reuse.install_reuse(policy, config, config.rl)
    modes = [(m, m.training) for m in policy.net.modules()]
    actual = evaluate(policy, objective, rows, config, callback)
    assert actual == expected and policy.backbone.calls == 6
    assert_state_equal(before, collect_rng_state())
    assert all(m.training == flag for m, flag in modes)
    assert reuse.observations(policy)["hits"] == reuse.observations(policy)["misses"] == 2
    for prepared in policy.net.prepared_actions:
        assert all(x.dtype == torch.float32 for x in prepared.values())
        assert torch.equal(prepared["state"], task_inputs()["state"])
        if "action" in prepared:
            assert torch.equal(prepared["action"], task_inputs()["action"])
    assert policy._evaluation_backbone_reuse.raw is None


def test_real_head_failure_consuming_rng_restores_outer_rng_and_mixed_modes(monkeypatch):
    import random

    import numpy as np

    from cosmos_framework.posttrain.checkpoint.rng import collect_rng_state
    from cosmos_framework.posttrain.entrypoints.action_sft import evaluate
    from cosmos_framework.posttrain.tests.test_action_sft import assert_state_equal

    policy, config = native_policy()
    config.rl.reuse_gr00t_evaluation_backbone = True
    reuse.install_reuse(policy, config, config.rl)
    policy.net.train()
    policy.net.action_head.vlln.eval()
    modes = [(m, m.training) for m in policy.net.modules()]
    before = collect_rng_state()

    def fail(*args, **kwargs):
        torch.rand(7)
        np.random.rand(3)
        random.random()
        raise LookupError("actual native sampling failure")

    monkeypatch.setattr(policy.net.action_head, "get_action", fail)
    with pytest.raises(LookupError, match="actual native sampling failure"):
        evaluate(policy, objective, [SimpleNamespace(id="row", transitions=[object()])], config, callback)
    assert_state_equal(before, collect_rng_state())
    assert all(m.training == flag for m, flag in modes)
    assert policy._evaluation_backbone_reuse.inputs is policy._evaluation_backbone_reuse.raw is None
    assert not policy._evaluation_backbone_reuse.active


def test_training_and_unscoped_prediction_keep_original_native_path(monkeypatch):
    policy, config = native_policy()
    config.rl.reuse_gr00t_evaluation_backbone = True
    reuse.install_reuse(policy, config, config.rl)
    called = []
    forward = policy.gr00t.forward
    sample = policy.gr00t.get_action

    def traced_forward(value):
        called.append("forward")
        return forward(value)

    def traced_sample(value):
        called.append("get_action")
        return sample(value)

    monkeypatch.setattr(policy.gr00t, "forward", traced_forward)
    monkeypatch.setattr(policy.gr00t, "get_action", traced_sample)
    policy.net.train()
    rng = torch.random.get_rng_state().clone()
    with policy.compute_ctx:
        expected = forward(task_inputs())
    expected["loss"].backward()
    gradients = {k: p.grad.clone() for k, p in policy.net.named_parameters() if p.grad is not None}
    policy.net.zero_grad(set_to_none=True)
    torch.random.set_rng_state(rng)
    result = policy.flow_matching_loss(task_inputs())
    assert result.loss.requires_grad
    result.loss.backward()
    assert torch.equal(result.loss, expected["loss"])
    assert all(torch.equal(dict(policy.net.named_parameters())[k].grad, value) for k, value in gradients.items())
    policy.net.eval()
    with torch.no_grad():
        callback(policy, None, None)
    assert called == ["forward", "get_action"]
    assert reuse.observations(policy)["hits"] == reuse.observations(policy)["misses"] == 0


def test_typed_flag_defaults_off_and_training_contract_refuses_changed_resume():
    import copy

    from cosmos_framework.posttrain.config.action_sft import ActionSFTConfig
    from cosmos_framework.posttrain.configs.gr00t_n1d7_libero_qlora import make_config
    from cosmos_framework.posttrain.entrypoints.action_sft import training_identity
    from cosmos_framework.posttrain.model.checkpoint import AdapterCheckpointCodec

    assert ActionSFTConfig().reuse_gr00t_evaluation_backbone is False
    with pytest.raises(ValueError, match="reuse_gr00t_evaluation_backbone must be a boolean"):
        ActionSFTConfig(reuse_gr00t_evaluation_backbone=1)
    config = make_config()
    off = training_identity(config, config.rl, SimpleNamespace(identity={}), SimpleNamespace(identity={}))
    config.rl.reuse_gr00t_evaluation_backbone = True
    on = training_identity(config, config.rl, SimpleNamespace(identity={}), SimpleNamespace(identity={}))
    changed = copy.deepcopy(off)
    changed["evaluation_backbone_reuse"]["enabled"] = True
    assert on == changed
    assert "utils/gr00t_evaluation_backbone.py" in on["implementation"]["files"]

    def codec(contract):
        net = nn.Linear(2, 2)
        net._cosmos_adapter_parameters = dict(net.named_parameters())
        result = AdapterCheckpointCodec(net, identity={"family": "same-new-source-fixture"})
        result.bind_training_identity(contract)
        return net, result, torch.optim.AdamW(net.parameters())

    _saved_net, saved, saved_optimizer = codec(off)
    _current_net, current, current_optimizer = codec(on)
    with pytest.raises(ValueError, match="identity differs"):
        current.validate_metadata(saved.metadata(saved_optimizer), purpose="resume", optimizer=current_optimizer)
    current.validate_metadata(saved.metadata(saved_optimizer), purpose="reload")
