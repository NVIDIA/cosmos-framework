# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""GR00T action-head adapter targets and installation on CPU/meta modules."""

import copy
from dataclasses import asdict

import pytest
import torch
from torch import nn

from cosmos_framework.posttrain.config.gr00t_bc import HEAD_ADAPTER_PROFILE, Gr00tBackboneAdaptation
from cosmos_framework.posttrain.model.gr00t.head_adaptation import (
    configured_head_targets,
    resolve_head_attention_targets,
)
from cosmos_framework.posttrain.tests.qlora.action_helpers import require_native_action

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def native_config(*, tiny=False):
    require_native_action("gr00t")
    from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config

    return Gr00tN1d7Config(
        select_layer=16,
        state_dropout_prob=0.2,
        hidden_size=8 if tiny else 1024,
        input_embedding_dim=8 if tiny else 1536,
        backbone_embedding_dim=8 if tiny else 2048,
        max_action_dim=3 if tiny else 132,
        max_state_dim=4 if tiny else 132,
        action_horizon=2 if tiny else 40,
        max_seq_len=4 if tiny else 1024,
        diffusion_model_cfg={
            "num_layers": 2 if tiny else 32,
            "num_attention_heads": 2 if tiny else 32,
            "attention_head_dim": 4 if tiny else 48,
            "output_dim": 8 if tiny else 1024,
            "norm_type": "ada_norm",
            "dropout": 0.2,
            "final_dropout": True,
            "interleave_self_attention": True,
            "positional_embeddings": None,
        },
        vl_self_attention_cfg={
            "num_layers": 1 if tiny else 4,
            "num_attention_heads": 2 if tiny else 32,
            "attention_head_dim": 4 if tiny else 64,
            "dropout": 0.2,
            "final_dropout": True,
            "positional_embeddings": None,
        },
    )


def native_model(*, tiny=False):
    config = native_config(tiny=tiny)
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead

    model = nn.Module()
    model.config = config
    with torch.device("cpu" if tiny else "meta"):
        model.action_head = Gr00tN1d7ActionHead(config).float()
    return model


@pytest.fixture(scope="module")
def pinned_head():
    return native_model()


def selected_config(**kwargs):
    return Gr00tBackboneAdaptation(
        **{"train_action_head": False, "action_head_adapter_profile": HEAD_ADAPTER_PROFILE, **kwargs}
    )


@pytest.mark.parametrize("mode", ["nf4_lora", "bf16_lora"])
def test_profile_is_explicit_and_preserves_legacy_config_identity(mode):
    from cosmos_framework.posttrain.configs.gr00t_n1d7_libero_qlora import make_config

    model = make_config().model
    legacy = model.identity()
    assert "action_head_adapter_profile" not in legacy["backbone_adaptation"]
    model.backbone_adaptation = selected_config(mode=mode)
    assert model.identity()["backbone_adaptation"]["action_head_adapter_profile"] == HEAD_ADAPTER_PROFILE
    assert model.identity()["backbone_adaptation"]["train_action_head"] is False
    old_fields = asdict(Gr00tBackboneAdaptation())
    old_fields.pop("action_head_adapter_profile")
    assert {k: v for k, v in legacy["backbone_adaptation"].items() if k != "backbone_path"} == {
        k: v for k, v in old_fields.items() if k != "backbone_path"
    }


@pytest.mark.parametrize(
    "change",
    [
        {"train_action_head": True},
        {"mode": "bf16_frozen"},
        {"mode": "nf4_frozen"},
        {"action_head_adapter_profile": "all-head-linears"},
    ],
)
def test_unsupported_head_policy_is_rejected(change):
    with pytest.raises(ValueError):
        selected_config(**change)


def test_actual_native_meta_head_resolves_only_128_dit_attention_projections(pinned_head):
    targets = configured_head_targets(pinned_head, selected_config())
    assert len(targets) == 128
    assert len(list(pinned_head.action_head.parameters())) == 537
    assert sum(32 * (rows + columns) for rows, columns in targets.values()) == 13_107_200
    assert all(name.startswith("action_head.model.transformer_blocks.") for name in targets)
    assert not any(part in name for name in targets for part in ("vl_self", "encoder", "decoder", ".ff.", ".norm"))
    assert set(targets.values()) == {(1536, 1536), (1536, 2048)}
    assert configured_head_targets(pinned_head, Gr00tBackboneAdaptation()) is None
    assert configured_head_targets(pinned_head, Gr00tBackboneAdaptation(train_action_head=False)) is None


@pytest.mark.parametrize(
    "change",
    [
        "category",
        "category_shape",
        "fused",
        "processor",
        "custom",
        "alias",
        "weight_alias",
        "bias",
        "dtype",
        "shape",
        "mode",
        "dropout",
        "blocks",
    ],
)
def test_actual_native_head_mismatches_refused_before_source_read(pinned_head, change):
    from diffusers.models.attention_processor import FusedAttnProcessor2_0

    model = copy.deepcopy(pinned_head)
    head = model.action_head
    attention = head.model.transformer_blocks[0].attn1
    if change == "category":
        head.state_encoder.layer1 = nn.Linear(132, 1024, device="meta")
    elif change == "category_shape":
        head.state_encoder.layer1.W = nn.Parameter(torch.empty(31, 132, 1024, device="meta"))
    elif change == "fused":
        attention.fuse_projections()
    elif change == "processor":
        attention.set_processor(FusedAttnProcessor2_0())
    elif change == "custom":

        class Custom(nn.Linear):
            pass

        attention.to_q = Custom(1536, 1536, device="meta")
    elif change == "alias":
        head.alias = attention.to_q
    elif change == "weight_alias":
        attention.to_q.weight = attention.to_out[0].weight
    elif change == "bias":
        attention.to_q.bias = None
    elif change == "dtype":
        head.action_decoder.layer2.W = nn.Parameter(head.action_decoder.layer2.W.bfloat16())
    elif change == "shape":
        attention.to_q = nn.Linear(2048, 1536, device="meta")
    elif change == "mode":
        head.set_trainable_parameters(True, False, True)
    elif change == "dropout":
        attention.to_out[1].p = 0.0
    else:
        head.model.transformer_blocks.pop(-1)
    with pytest.raises(ValueError):
        resolve_head_attention_targets(model)


def test_frozen_original_parameters_preserve_actual_native_dropout_modes(pinned_head):
    # This is also the existing train_action_head=False policy: requires_grad alone
    # never enters Isaac's tune_projector/tune_diffusion_model=False eval branches.
    model = copy.deepcopy(pinned_head)
    model.action_head.requires_grad_(False)
    model.train()
    before = [(name, module.training) for name, module in model.action_head.named_modules()]
    model.action_head.set_frozen_modules_to_eval_mode()
    assert [(name, module.training) for name, module in model.action_head.named_modules()] == before
    assert model.action_head.state_dropout_prob == 0.2
    assert all(
        module.training and module.p == 0.2 for module in model.action_head.modules() if type(module) is nn.Dropout
    )
    assert len(configured_head_targets(model, selected_config())) == 128


def test_tiny_actual_native_head_zero_init_rng_gradient_and_frozen_updates():
    from transformers.feature_extraction_utils import BatchFeature

    from cosmos_framework.posttrain.model.quantization.family import install_family_backbone

    torch.manual_seed(349)
    original = native_model(tiny=True)
    original.backbone = nn.Linear(8, 8, bias=False).bfloat16()
    models = [copy.deepcopy(original) for _ in range(3)]
    states = []
    for model, policy in zip(models, ("full", "frozen", "head_lora"), strict=True):
        heads = {
            name: tuple(module.weight.shape)
            for name, module in model.named_modules()
            if name.startswith("action_head.model.transformer_blocks.")
            and type(module) is nn.Linear
            and any(name.endswith(leaf) for leaf in ("attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0"))
        }
        before_rng = torch.get_rng_state().clone()
        state = install_family_backbone(
            model,
            config=selected_config(mode="bf16_lora")
            if policy == "head_lora"
            else Gr00tBackboneAdaptation(mode="bf16_lora", train_action_head=policy == "full"),
            targets={"backbone": (8, 8)},
            auxiliary_parameters=tuple(name for name, _ in model.named_parameters() if name.startswith("action_head."))
            if policy == "full"
            else (),
            head_targets=heads if policy == "head_lora" else None,
            source_identity={"fixture": "actual_native_reduced_head"},
            task_identity={"family": "gr00t_n1d7"},
            device=torch.device("cpu"),
        )
        assert torch.equal(before_rng, torch.get_rng_state())
        states.append(state)
    for model in models[1:]:
        for leaf in ("lora_A", "lora_B"):
            assert torch.equal(getattr(models[0].backbone, leaf).weight, getattr(model.backbone, leaf).weight)
    inputs = torch.randn(1, 3, 8).bfloat16()
    action = {
        "state": torch.randn(1, 1, 4),
        "action": torch.randn(1, 2, 3),
        "action_mask": torch.ones(1, 2, 3),
        "embodiment_id": torch.tensor([2]),
    }

    def raw(model):
        return BatchFeature(
            data={
                "backbone_features": model.backbone(inputs),
                "backbone_attention_mask": torch.ones(1, 3, dtype=torch.bool),
                "image_mask": torch.tensor([[True, False, True]]),
            }
        )

    rng = torch.get_rng_state().clone()
    outputs, gradients, endings = [], [], []
    for model in models:
        model.train()
        torch.set_rng_state(rng)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            result = model.action_head(raw(model), BatchFeature(data=copy.deepcopy(action)))
        result["loss"].backward()
        outputs.append(result["loss"].detach())
        gradients.append(model.backbone.lora_B.weight.grad.clone())
        endings.append(torch.get_rng_state().clone())
    assert all(torch.equal(outputs[0], value) for value in outputs[1:])
    assert all(torch.equal(gradients[0], value) for value in gradients[1:])
    assert all(torch.equal(endings[0], value) for value in endings[1:])
    sampled, endings = [], []
    for model in models:
        model.eval()
        torch.set_rng_state(rng)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            sampled.append(
                model.action_head.get_action(
                    raw(model),
                    BatchFeature(
                        data={
                            key: value.clone() for key, value in action.items() if key not in {"action", "action_mask"}
                        }
                    ),
                )["action_pred"]
            )
        endings.append(torch.get_rng_state().clone())
    assert all(torch.equal(sampled[0], value) for value in sampled[1:])
    assert all(torch.equal(endings[0], value) for value in endings[1:])
    models[-1].train()
    new = models[-1]
    factors = {name: parameter for name, parameter in new.named_parameters() if parameter.requires_grad}
    assert len(factors) == 18  # two language + 2 blocks * 4 projections * 2 head factors
    assert all(parameter.dtype == torch.float32 for parameter in factors.values())
    head_b = [p for n, p in factors.items() if n.startswith("action_head.") and ".lora_B." in n]
    head_a = [p for n, p in factors.items() if n.startswith("action_head.") and ".lora_A." in n]
    assert all(p.grad is not None and torch.count_nonzero(p.grad) == 0 for p in head_a)
    assert any(torch.count_nonzero(p.grad) for p in head_b)
    optimizer = torch.optim.AdamW(factors.values(), lr=1e-5)
    states[-1].assert_optimizer(optimizer)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        new.action_head(raw(new), BatchFeature(data=copy.deepcopy(action)))["loss"].backward()
    assert any(torch.count_nonzero(p.grad) for p in head_a)
    optimizer.step()
    states[-1].validate_frozen_base()


@pytest.mark.parametrize("mode", ["bf16_lora", "nf4_lora"])
def test_native_builder_passes_final_head_allowlist_before_one_codec(mode, pinned_head, tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    from cosmos_framework.posttrain.configs.gr00t_n1d7_libero_qlora import make_config
    from cosmos_framework.posttrain.model.gr00t import adaptation, policy, qlora_source, source
    from cosmos_framework.posttrain.model.quantization import family

    model = copy.deepcopy(pinned_head)
    model.backbone = nn.Module()
    model.backbone.model = nn.Module()
    language = model.backbone.model.language_model = nn.Module()
    language.rotary_emb = nn.Module()
    language.layers = nn.ModuleList()
    with torch.device("meta"):
        for _ in range(16):
            layer = nn.Module()
            layer.self_attn, layer.mlp = nn.Module(), nn.Module()
            for relative in adaptation.LANGUAGE_PROJECTIONS:
                parent, name = relative.split(".")
                setattr(getattr(layer, parent), name, nn.Linear(2048, 128, bias=False))
            language.layers.append(layer)
    config = make_config().model
    config.backbone_adaptation = selected_config(mode=mode, backbone_path=str(tmp_path))
    (tmp_path / "config.json").write_text(json.dumps({"native": "fixture"}))
    for name in ("processor_config.json", "statistics.json"):
        (tmp_path / name).write_text("{}")
    original_keys = set(model.state_dict())
    monkeypatch.setattr(adaptation, "verify_external_source", lambda: None)
    monkeypatch.setattr(source, "resolve_source", lambda *args: (tmp_path, ""))
    monkeypatch.setattr(source, "metadata_files", lambda *args: {})
    monkeypatch.setattr(qlora_source, "pinned_source_identity", lambda *args: {"pinned": True})
    monkeypatch.setattr(adaptation, "construct_meta", lambda *args: model)
    monkeypatch.setattr(
        adaptation, "checkpoint_weight_map", lambda *args: dict.fromkeys(original_keys, "model.safetensors")
    )
    monkeypatch.setattr(policy, "load_processor", lambda *args, **kwargs: object())
    calls = []

    def install(actual, **kwargs):
        # The source inventory is admitted before factors exist. The shared
        # installer then streams bases and creates both groups before its codec.
        assert set(actual.state_dict()) == original_keys
        assert len(kwargs["targets"]) == 112
        assert len(kwargs["head_targets"]) == 128
        assert kwargs["auxiliary_parameters"] == ()
        # Model-defining identity records the pinned upstream revision, not source-file hashes.
        assert kwargs["task_identity"]["external"] == {"revision": adaptation.ISAAC_GR00T_REVISION}
        assert (
            kwargs["task_identity"]["config"]["backbone_adaptation"]["action_head_adapter_profile"]
            == HEAD_ADAPTER_PROFILE
        )
        calls.append(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(family, "install_family_backbone", install)
    built, _, _ = adaptation.build_adapted_gr00t(config, device="cpu", save_cfg_dir=tmp_path / "out")
    assert built is model and len(calls) == 1
    assert built.action_head.tune_projector and built.action_head.tune_diffusion_model and built.action_head.tune_vlln
    # A source with preinserted factors is refused before the installer/codec.
    monkeypatch.setattr(
        adaptation,
        "checkpoint_weight_map",
        lambda *args: {
            **dict.fromkeys(original_keys, "model.safetensors"),
            "action_head.model.transformer_blocks.0.attn1.to_q.lora_A.weight": "model.safetensors",
        },
    )
    with pytest.raises(ValueError, match="checkpoint inventory differs"):
        adaptation.build_adapted_gr00t(config, device="cpu", save_cfg_dir=tmp_path / "out2")
    assert len(calls) == 1


def test_head_profile_merged_reload_refused_before_artifact_reads(tmp_path, monkeypatch):
    from cosmos_framework.posttrain.checkpoint import family_dense
    from cosmos_framework.posttrain.configs.gr00t_n1d7_libero_qlora import make_config
    from cosmos_framework.posttrain.model.gr00t.dense import build_policy

    config = make_config()
    config.model.backbone_adaptation = selected_config(mode="bf16_lora")

    def unexpected(*args, **kwargs):
        pytest.fail("merged head policy must fail before loading any artifact")

    monkeypatch.setattr(family_dense, "read_family_dense", unexpected)
    with pytest.raises(ValueError, match="standalone unmerged"):
        build_policy(config, tmp_path / "absent", device="cpu", output=tmp_path, unmerged=False)
