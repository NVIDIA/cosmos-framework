# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Actual tiny public Alpamayo/Qwen prefix + action expert flow graph."""

from __future__ import annotations

import copy
import json

import pytest
import torch

pytest.importorskip("alpamayo_r1")
import transformers

if transformers.__version__ != "4.57.6":
    pytest.skip("pinned Alpamayo runtime", allow_module_level=True)
from cosmos_framework.posttrain.model.alpamayo.policy import AlpamayoSFTPolicy
from cosmos_framework.posttrain.model.quantization.family import FamilyBackboneConfig, install_family_backbone


def source(tmp_path):
    from alpamayo_r1.config import AlpamayoR1Config
    from safetensors.torch import save_file
    from tokenizers import Tokenizer, models
    from transformers import (
        Qwen2TokenizerFast,
        Qwen2VLImageProcessorFast,
        Qwen3VLConfig,
        Qwen3VLProcessor,
        Qwen3VLVideoProcessor,
    )

    from cosmos_framework.posttrain.model.alpamayo.policy import AlpamayoSFTNet
    from cosmos_framework.posttrain.tests.test_model_qwen3_vl_gpu import _dense_config

    processor = tmp_path / "processor"
    processor.mkdir()
    vocab = {f"token{i}": i for i in range(128)}
    for name, index in [
        ("<unk>", 0),
        ("<|vision_start|>", 118),
        ("<|vision_end|>", 119),
        ("<|image_pad|>", 120),
        ("<|video_pad|>", 121),
    ]:
        del vocab[f"token{index}"]
        vocab[name] = index
    tokenizer = Qwen2TokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel(vocab, unk_token="<unk>")), unk_token="<unk>", pad_token="<unk>"
    )
    tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    media = dict(min_pixels=16, max_pixels=64, patch_size=2, temporal_patch_size=2, merge_size=2)
    Qwen3VLProcessor(
        tokenizer=tokenizer,
        image_processor=Qwen2VLImageProcessorFast(**media),
        video_processor=Qwen3VLVideoProcessor(**media),
    ).save_pretrained(processor)
    qwen = Qwen3VLConfig(**_dense_config().to_dict())
    # The public 8B processor declares a nested BF16 text dtype. Transformers
    # honors it even inside an outer from_pretrained(dtype=float32) call.
    qwen.text_config.dtype = torch.bfloat16
    qwen.text_config.rope_scaling = {"rope_type": "default", "mrope_section": [1, 1, 2]}
    qwen.save_pretrained(processor)
    cfg = AlpamayoR1Config(
        vlm_name_or_path=str(processor),
        traj_vocab_size=64,
        add_special_tokens=True,
        attn_implementation="sdpa",
        model_dtype="float32",
        min_pixels=16,
        max_pixels=64,
        action_space_cfg={"_target_": "alpamayo_r1.action_space.UnicycleAccelCurvatureActionSpace", "n_waypoints": 4},
        diffusion_cfg={"_target_": "alpamayo_r1.diffusion.flow_matching.FlowMatching"},
        action_in_proj_cfg={
            "_target_": "alpamayo_r1.models.action_in_proj.PerWaypointActionInProjV2",
            "hidden_size": 32,
            "num_enc_layers": 2,
            "num_fourier_feats": 4,
        },
        action_out_proj_cfg={"_target_": "torch.nn.Linear"},
        expert_cfg={"hidden_size": 32, "intermediate_size": 64, "dtype": "float32"},
    )
    torch.manual_seed(13)
    model = AlpamayoSFTNet(cfg)
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text(json.dumps(cfg.to_dict()))
    save_file({n: v.contiguous() for n, v in model.state_dict().items()}, str(base / "model.safetensors"))
    return model.bfloat16()


def batch():
    from cosmos_framework.posttrain.tests.test_model_qwen3_vl_gpu import _batch

    torch.manual_seed(27)
    actions = torch.randn(1, 4, 2)
    return dict(
        tokenized=_batch(),
        history={},
        actions=actions,
        action_mask=torch.ones_like(actions),
        time=torch.tensor([0.3]),
        noise=torch.randn_like(actions),
    )


@pytest.mark.parametrize("checkpointing", [False, True])
@pytest.mark.parametrize("mode", ["bf16_lora", "nf4_lora", "bf16_frozen"])
def test_actual_flow_reaches_adapters_and_auxiliary_heads(tmp_path, checkpointing, mode):
    import os

    import torch.distributed as dist

    device = torch.device("cpu")
    if mode == "nf4_lora":
        if not torch.cuda.is_available() or "LOCAL_RANK" not in os.environ:
            pytest.skip("NF4 requires real CUDA kernels under one-rank torchrun")
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        torch.cuda.set_device(device)
        if not dist.is_initialized():
            dist.init_process_group("nccl")
    net = source(tmp_path)
    reference = copy.deepcopy(net)
    targets = {
        name: tuple(module.weight.shape)
        for name, module in net.named_modules()
        if type(module) is torch.nn.Linear
        and name.startswith("vlm.model.language_model.layers.")
        and (".layers.0." in name or name.endswith(("k_proj", "v_proj")))
    }
    auxiliary = tuple(
        name
        for name, _ in net.named_parameters()
        if name.startswith(("expert.", "action_in_proj.", "action_out_proj."))
    )
    settings = FamilyBackboneConfig(mode=mode, rank=2, alpha=4, seed=9)

    def install(model):
        return install_family_backbone(
            model,
            config=settings,
            targets=targets,
            auxiliary_parameters=auxiliary,
            source_identity={"fixture": "actual-public-Alpamayo"},
            task_identity={"loss": "native-flow"},
            device=device,
        )

    state = install(net)
    from cosmos_framework.posttrain.model.alpamayo.runtime import bind_rotary_aliases

    bind_rotary_aliases(net)
    for name in ("vlm.model.language_model.rotary_emb", "expert.rotary_emb"):
        rotary = net.get_submodule(name)
        assert rotary.original_inv_freq is rotary.inv_freq
    net._full_activation_checkpointing = checkpointing
    policy = AlpamayoSFTPolicy(net)
    data = batch()
    optimizer = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=1e-3)
    state.assert_optimizer(optimizer)
    trainables = set(state.codec.parameters)
    assert set(auxiliary) <= trainables
    if mode == "bf16_frozen":
        assert trainables == set(auxiliary)
        assert not any("lora_" in name for name, _ in net.named_parameters())
    else:
        assert trainables - set(auxiliary) == {
            f"{name}.{factor}.weight" for name in targets for factor in ("lora_A", "lora_B")
        }
    before = {name: p.detach().clone() for name, p in state.codec.parameters.items()}
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        out = policy.flow_matching_loss(data)
        assert torch.isfinite(out.loss)
        out.loss.backward()
        for name, p in state.codec.parameters.items():
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
            if "lora_A" not in name or step > 0:
                assert p.grad.abs().sum() > 0, name
        optimizer.step()
    assert all(not torch.equal(before[name], p) for name, p in state.codec.parameters.items())
    state.validate_frozen_base()
    restored = install(reference)
    restored.codec.validate_metadata(state.codec.metadata(), purpose="reload")
    restored.codec.view.load_state_dict(state.codec.view.state_dict())
    torch.testing.assert_close(
        AlpamayoSFTPolicy(reference).flow_matching_loss(data).prediction,
        policy.flow_matching_loss(data).prediction,
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("unmerged", [False, True])
def test_native_dense_meta_reconstruction_after_real_task_updates(tmp_path, unmerged):
    from accelerate import init_empty_weights

    from cosmos_framework.posttrain.checkpoint.adapter import save_adapter_artifact
    from cosmos_framework.posttrain.checkpoint.family_dense import (
        export_family_dense,
        export_family_unmerged,
        load_family_dense,
        load_family_unmerged,
    )
    from cosmos_framework.posttrain.checkpoint.manager import CheckpointManager
    from cosmos_framework.posttrain.model.alpamayo.policy import AlpamayoSFTNet

    net = source(tmp_path)
    targets = {
        name: tuple(module.weight.shape)
        for name, module in net.named_modules()
        if type(module) is torch.nn.Linear
        and name.startswith("vlm.model.language_model.layers.")
        and (".layers.0." in name or name.endswith(("k_proj", "v_proj")))
    }
    auxiliary = tuple(
        name
        for name, _ in net.named_parameters()
        if name.startswith(("expert.", "action_in_proj.", "action_out_proj."))
    )
    state = install_family_backbone(
        net,
        config=FamilyBackboneConfig(mode="bf16_lora", rank=2, alpha=4, seed=9),
        targets=targets,
        auxiliary_parameters=auxiliary,
        source_identity={"fixture": "actual-public-Alpamayo"},
        task_identity={"loss": "native-flow"},
        device=torch.device("cpu"),
    )
    policy, data = AlpamayoSFTPolicy(net), batch()
    optimizer = torch.optim.AdamW(state.codec.parameters.values(), lr=1e-3)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        policy.flow_matching_loss(data).loss.backward()
        optimizer.step()
    reference = policy.flow_matching_loss(data).prediction.detach()
    checkpoint = save_adapter_artifact(
        str(tmp_path / "portable" / "iter_000003"),
        state.codec.view.state_dict(),
        state.codec.metadata(),
        manager=CheckpointManager(checkpoint_format="consolidated"),
    )
    destination = tmp_path / "dense"
    export = export_family_unmerged if unmerged else export_family_dense
    load = load_family_unmerged if unmerged else load_family_dense
    export(
        net,
        checkpoint=checkpoint,
        output=str(destination),
        **({} if unmerged else {"merged_backbone_dtype": "bfloat16"}),
        exporter_build={"code_revision": "a" * 40, "wheel_sha256": "b" * 64},
        max_shard_size_bytes=65536,
    )
    with init_empty_weights(include_buffers=False):
        reconstructed = AlpamayoSFTNet(net.config)
    assert all(parameter.is_meta for parameter in reconstructed.parameters())
    rng = torch.get_rng_state().clone()
    load(reconstructed, destination, device="cpu", expected_identity=state.codec.identity)
    assert torch.equal(rng, torch.get_rng_state())
    assert not any(parameter.is_meta or parameter.requires_grad for parameter in reconstructed.parameters())
    for name in auxiliary:
        torch.testing.assert_close(reconstructed.get_parameter(name), net.get_parameter(name), atol=0, rtol=0)
    prediction = AlpamayoSFTPolicy(reconstructed).flow_matching_loss(data).prediction.detach()
    # BF16 weight merge changes arithmetic: this is a bounded tiny-output check,
    # not exact adapter parity and not a real driving task-quality qualification.
    torch.testing.assert_close(prediction, reference, atol=0 if unmerged else 2e-3, rtol=0 if unmerged else 2e-2)


# The exact keys accepted by the CLI override ++rl.optimizer.lr_multipliers={expert:0.1,...}.
EXPERT_LR_MULTIPLIERS = {"expert": 0.1, "action_in_proj": 0.1, "action_out_proj": 0.1}


def test_expert_lr_multiplier_groups_schedule_and_checkpoint_identity(tmp_path):
    """The 0.1x expert/projection screen uses only the existing native optimizer grouping."""
    from cosmos_framework.posttrain.config.build import build_optimizer, build_warmup_cosine_lr_scheduler
    from cosmos_framework.posttrain.config.rl import OptimizerConfig

    net = source(tmp_path)
    targets = {
        name: tuple(module.weight.shape)
        for name, module in net.named_modules()
        if type(module) is torch.nn.Linear
        and name.startswith("vlm.model.language_model.layers.")
        and (".layers.0." in name or name.endswith(("k_proj", "v_proj")))
    }
    auxiliary = tuple(name for name, _ in net.named_parameters() if name.startswith(tuple(EXPERT_LR_MULTIPLIERS)))
    state = install_family_backbone(
        net,
        config=FamilyBackboneConfig(mode="bf16_lora", rank=2, alpha=4, seed=9),
        targets=targets,
        auxiliary_parameters=auxiliary,
        source_identity={"fixture": "actual-public-Alpamayo"},
        task_identity={"loss": "native-flow"},
        device=torch.device("cpu"),
    )
    from cosmos_framework.posttrain.model.alpamayo.runtime import bind_rotary_aliases

    bind_rotary_aliases(net)
    cfg = OptimizerConfig(
        name="adamw",
        impl="fused",
        lr=1e-4,
        warmup_steps=2,
        weight_decay=0.0,
        lr_multipliers=dict(EXPERT_LR_MULTIPLIERS),
    )
    for refused, message in (
        ({"layers.0.": 0.1}, "only trained auxiliary"),
        ({"lora_": 0.1}, "only trained auxiliary"),
        ({**EXPERT_LR_MULTIPLIERS, "exprt.": 0.1}, "selects no trained auxiliary"),
    ):
        with pytest.raises(ValueError, match=message):
            build_optimizer(net, OptimizerConfig(name="adamw", impl="fused", lr=1e-4, lr_multipliers=refused))
    optimizer = build_optimizer(net, cfg)
    state.assert_optimizer(optimizer)
    names = {id(p): n for n, p in net.named_parameters()}
    groups = {g["lr_multiplier"]: {names[id(p)] for p in g["params"]} for g in optimizer.param_groups}
    adapters = set(state.codec.parameters) - set(auxiliary)
    assert set(groups) == {1.0, 0.1}
    assert groups[0.1] == set(auxiliary) and groups[1.0] == adapters
    assert all("lora_" in name for name in groups[1.0])
    assert optimizer.param_groups[0]["lr_multiplier"] == 1.0  # reported train/learning_rate stays the base lr
    scheduler = build_warmup_cosine_lr_scheduler(optimizer, warmup_steps=2, cycle_steps=6, f_start=0.01, f_min=0.01)
    policy = AlpamayoSFTPolicy(net)
    data = batch()
    before = {name: p.detach().clone() for name, p in state.codec.parameters.items()}
    for _ in range(4):
        lrs = {g["lr_multiplier"]: g["lr"] for g in optimizer.param_groups}
        assert lrs[0.1] == pytest.approx(0.1 * lrs[1.0], rel=1e-12)
        optimizer.zero_grad(set_to_none=True)
        policy.flow_matching_loss(data).loss.backward()
        optimizer.step()
        scheduler.step()
    assert all(not torch.equal(before[n], p) for n, p in state.codec.parameters.items())
    identity = state.codec.optimizer_identity(optimizer)
    assert [g["options"]["lr_multiplier"] for g in identity["groups"]] == [1.0, 0.1]
    assert {n for g in identity["groups"] for n in g["parameters"]} == set(state.codec.parameters)
    restored = build_optimizer(net, cfg)
    restored.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    assert [(g["lr"], g["lr_multiplier"]) for g in restored.param_groups] == [
        (g["lr"], g["lr_multiplier"]) for g in optimizer.param_groups
    ]
    assert state.codec.optimizer_identity(restored) == identity


@pytest.mark.parametrize("unmerged_reload", [False, True])
def test_frozen_expert_trains_only_language_adapters_and_reloads_exactly(tmp_path, unmerged_reload):
    """train_action_head=false: the source expert/projections stay frozen in their source dtype."""
    from accelerate import init_empty_weights

    from cosmos_framework.posttrain.checkpoint.adapter import save_adapter_artifact
    from cosmos_framework.posttrain.checkpoint.family_dense import export_family_unmerged, load_family_unmerged
    from cosmos_framework.posttrain.checkpoint.manager import CheckpointManager
    from cosmos_framework.posttrain.config.build import build_optimizer
    from cosmos_framework.posttrain.config.rl import OptimizerConfig
    from cosmos_framework.posttrain.model.alpamayo.policy import AlpamayoSFTNet
    from cosmos_framework.posttrain.model.alpamayo.runtime import bind_rotary_aliases
    from cosmos_framework.posttrain.model.alpamayo.source import auxiliary_parameters

    net = source(tmp_path)
    targets = {
        name: tuple(module.weight.shape)
        for name, module in net.named_modules()
        if type(module) is torch.nn.Linear
        and name.startswith("vlm.model.language_model.layers.")
        and (".layers.0." in name or name.endswith(("k_proj", "v_proj")))
    }
    assert auxiliary_parameters(net, train_action_head=False) == ()
    head = {n: p.detach().clone() for n, p in net.named_parameters() if n.startswith(tuple(EXPERT_LR_MULTIPLIERS))}
    state = install_family_backbone(
        net,
        config=FamilyBackboneConfig(mode="bf16_lora", rank=2, alpha=4, seed=9),
        targets=targets,
        auxiliary_parameters=auxiliary_parameters(net, train_action_head=False),
        source_identity={"fixture": "actual-public-Alpamayo"},
        task_identity={"loss": "native-flow", "train_action_head": False},
        device=torch.device("cpu"),
    )
    bind_rotary_aliases(net)
    assert set(state.codec.parameters) == {
        f"{name}.{factor}.weight" for name in targets for factor in ("lora_A", "lora_B")
    }
    for name, value in head.items():
        parameter = net.get_parameter(name)
        assert not parameter.requires_grad and parameter.dtype == torch.bfloat16, name
    with pytest.raises(ValueError, match="explicitly trained auxiliary"):
        build_optimizer(net, OptimizerConfig(name="adamw", impl="fused", lr=1e-4, lr_multipliers={"expert": 0.1}))
    optimizer = build_optimizer(net, OptimizerConfig(name="adamw", impl="fused", lr=1e-3, weight_decay=0.0))
    state.assert_optimizer(optimizer)
    policy, data = AlpamayoSFTPolicy(net), batch()
    before = {name: p.detach().clone() for name, p in state.codec.parameters.items()}
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        out = policy.flow_matching_loss(data)
        assert torch.isfinite(out.loss)
        out.loss.backward()
        for name, p in state.codec.parameters.items():
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
            if "lora_A" not in name or step > 0:
                assert p.grad.abs().sum() > 0, name  # the flow loss reaches adapters through the frozen expert
        optimizer.step()
    assert all(not torch.equal(before[name], p) for name, p in state.codec.parameters.items())
    for name, value in head.items():
        assert torch.equal(net.get_parameter(name), value), name
    state.validate_frozen_base()
    reference = policy.flow_matching_loss(data).prediction.detach()
    checkpoint = save_adapter_artifact(
        str(tmp_path / "portable" / "iter_000003"),
        state.codec.view.state_dict(),
        state.codec.metadata(),
        manager=CheckpointManager(checkpoint_format="consolidated"),
    )
    assert state.codec.metadata()["identity"]["auxiliary_parameters"] == []
    if not unmerged_reload:
        (tmp_path / "again").mkdir()
        fresh = source(tmp_path / "again")
        restored = install_family_backbone(
            fresh,
            config=FamilyBackboneConfig(mode="bf16_lora", rank=2, alpha=4, seed=9),
            targets=targets,
            auxiliary_parameters=(),
            source_identity={"fixture": "actual-public-Alpamayo"},
            task_identity={"loss": "native-flow", "train_action_head": False},
            device=torch.device("cpu"),
        )
        bind_rotary_aliases(fresh)
        restored.codec.validate_metadata(state.codec.metadata(), purpose="reload")
        restored.codec.view.load_state_dict(state.codec.view.state_dict())
        prediction = AlpamayoSFTPolicy(fresh).flow_matching_loss(data).prediction.detach()
        torch.testing.assert_close(prediction, reference, atol=0, rtol=0)
        return
    destination = tmp_path / "unmerged"
    export_family_unmerged(
        net,
        checkpoint=checkpoint,
        output=str(destination),
        exporter_build={"code_revision": "a" * 40, "wheel_sha256": "b" * 64},
        max_shard_size_bytes=65536,
    )
    with init_empty_weights(include_buffers=False):
        reconstructed = AlpamayoSFTNet(net.config)
    load_family_unmerged(reconstructed, destination, device="cpu", expected_identity=state.codec.identity)
    for name, value in head.items():
        assert torch.equal(reconstructed.get_parameter(name), value), name
    prediction = AlpamayoSFTPolicy(reconstructed).flow_matching_loss(data).prediction.detach()
    torch.testing.assert_close(prediction, reference, atol=0, rtol=0)
