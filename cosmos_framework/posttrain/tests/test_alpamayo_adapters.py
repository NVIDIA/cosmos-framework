# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Actual small public R1/Qwen3-VL/expert flow graphs; no external trainer."""

from __future__ import annotations

import copy
import json

import pytest
import torch

pytest.importorskip("alpamayo_r1")
import transformers

pytestmark = pytest.mark.skipif(transformers.__version__ != "4.57.6", reason="pinned robotics runtime")

from cosmos_framework.posttrain.checkpoint.adapters import load_adapter_artifact, save_adapter_artifact
from cosmos_framework.posttrain.config.adapters import AdapterComponentConfig as C
from cosmos_framework.posttrain.config.adapters import AdapterConfig
from cosmos_framework.posttrain.model.adapters import assert_trainability
from cosmos_framework.posttrain.model.alpamayo.policy import AlpamayoSFTPolicy, build_policy, load_base
from cosmos_framework.posttrain.model.alpamayo.sft import AlpamayoSFTModelConfig


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
    return str(base), AlpamayoSFTModelConfig(str(processor), "none", 16, 64)


def config(method, rs=False):
    common = dict(method=method, use_rslora=rs, rank=2, alpha=4)
    return AdapterConfig(
        components=[
            C(
                component="language",
                targets=[
                    "vlm.model.language_model.layers.*.self_attn.k_proj",
                    "vlm.model.language_model.layers.*.self_attn.v_proj",
                ],
                **common,
            ),
            C(component="vision", targets=["vlm.model.visual.blocks.*.attn.qkv"], layers=[0], **common),
            C(
                component="projector",
                targets=["vlm.model.visual.merger.linear_fc2", "vlm.model.visual.deepstack_merger_list.*.linear_fc2"],
                **common,
            ),
            C(
                component="action",
                targets=[
                    "expert.layers.*.self_attn.q_proj",
                    "expert.layers.*.self_attn.v_proj",
                    "action_in_proj.encoder.trunk.0",
                    "action_out_proj",
                ],
                **common,
            ),
        ]
    )


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


@pytest.mark.parametrize("method,rs", [("lora", False), ("lora", True), ("dora", False), ("pissa", False)])
@pytest.mark.parametrize("ac", ["none", "vision"])
def test_r1_components_flow_updates_and_reload(tmp_path, method, rs, ac):
    base, model_config = source(tmp_path)
    model_config.activation_checkpointing = ac
    cfg = config(method, rs)
    policy = build_policy(base, model_config, adapters=cfg, device="cpu")
    data = batch()
    reference = AlpamayoSFTPolicy(load_base(base, model_config)[0].eval())
    policy.net.eval()
    with torch.no_grad():
        torch.testing.assert_close(
            policy.flow_matching_loss(data).prediction,
            reference.flow_matching_loss(data).prediction,
            atol=2e-5,
            rtol=2e-5,
        )
    policy.net.train()
    before = {n: p.detach().clone() for n, p in policy.net.named_parameters()}
    initial_factors = {n: v.clone() for n, v in policy.net.named_buffers() if n.endswith(("initial_A", "initial_B"))}
    optimizer = torch.optim.AdamW([p for p in policy.net.parameters() if p.requires_grad], lr=0.003, weight_decay=0)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        output = policy.flow_matching_loss(data)
        assert torch.isfinite(output.loss)
        output.loss.backward()
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all() for p in policy.net.parameters() if p.requires_grad
        )
        optimizer.step()
    assert_trainability(policy.net, optimizer)
    assert all(torch.equal(v, dict(policy.net.named_buffers())[n]) for n, v in initial_factors.items())
    for name, p in policy.net.named_parameters():
        if p.requires_grad:
            assert not torch.equal(p, before[name]), name
        else:
            assert torch.equal(p, before[name]), name
    policy.net.eval()
    with torch.no_grad():
        expected = policy.flow_matching_loss(data).prediction
    portable = tmp_path / "portable"
    save_adapter_artifact(policy.net, str(portable))
    warm = copy.deepcopy(cfg)
    warm.warm_start = str(portable)
    model_config.activation_checkpointing = "none"
    loaded = build_policy(base, model_config, adapters=warm, device="cpu")
    loaded.net.eval()
    with torch.no_grad():
        torch.testing.assert_close(loaded.flow_matching_loss(data).prediction, expected, atol=2e-5, rtol=2e-5)
    # An explicit load repeats no method initialization/decomposition.
    load_adapter_artifact(loaded.net, str(portable))


@pytest.mark.parametrize("method,rs", [("lora", False), ("lora", True), ("dora", False), ("pissa", False)])
def test_r1_merged_file_precision_reload_and_identity(tmp_path, method, rs):
    from cosmos_framework.posttrain.model.alpamayo.adapter_export import export_merged, load_merged

    base, model_config = source(tmp_path)
    policy = build_policy(base, model_config, adapters=config(method, rs), device="cpu")
    with torch.no_grad():
        for parameter in policy.net.parameters():
            if parameter.requires_grad:
                parameter.add_(0.001)
    policy.net.eval()
    portable = tmp_path / "portable"
    save_adapter_artifact(policy.net, str(portable))
    before = {n: v.clone() for n, v in policy.net.state_dict().items()}
    export_merged(str(portable), base, str(tmp_path / "merged"), processor=model_config.processor_path)
    assert all(torch.equal(v, policy.net.state_dict()[n]) for n, v in before.items())
    loaded = load_merged(str(tmp_path / "merged"))
    policy.net.float()
    with torch.no_grad():
        torch.testing.assert_close(
            loaded.flow_matching_loss(batch()).prediction,
            policy.flow_matching_loss(batch()).prediction,
            atol=5e-5,
            rtol=5e-5,
        )
    with pytest.raises(ValueError, match="merged artifacts require"):
        build_policy(str(tmp_path / "merged"), model_config, adapters=config(method, rs), device="cpu")
    (tmp_path / "merged/processor/config.json").write_text("{}")
    with pytest.raises(ValueError, match="missing or changed"):
        load_merged(str(tmp_path / "merged"))
    metadata = json.loads((tmp_path / "base/config.json").read_text())
    metadata["wrong_base_marker"] = 1
    (tmp_path / "base/config.json").write_text(json.dumps(metadata))
    cfg = config(method, rs)
    cfg.warm_start = str(portable)
    with pytest.raises(ValueError, match="compatibility mismatch"):
        build_policy(base, model_config, adapters=cfg, device="cpu")


def test_r1_rejects_dead_prefix_targets_and_cache_checkpoint_replay(tmp_path):
    base, model_config = source(tmp_path)
    cfg = AdapterConfig(components=[C(targets=["vlm.model.language_model.layers.1.self_attn.q_proj"], rank=2)])
    with pytest.raises(ValueError, match="final-prefix K/V"):
        build_policy(base, model_config, adapters=cfg, device="cpu")
    model_config.activation_checkpointing = "full"
    with pytest.raises(ValueError, match="mutable"):
        build_policy(base, model_config, adapters=config("lora"), device="cpu")


@pytest.mark.parametrize("method,rs", [("lora", False), ("lora", True), ("dora", False), ("pissa", False)])
def test_r1_native_stochastic_checkpoint_next_update(tmp_path, monkeypatch, method, rs):
    from cosmos_framework.posttrain.algorithm.flow_bc import make_objective
    from cosmos_framework.posttrain.checkpoint.io import load_component
    from cosmos_framework.posttrain.roles.trainer.llm import LLMTrainer

    base, model_config = source(tmp_path)
    cfg = config(method, rs)
    data = batch()
    del data["noise"], data["time"]

    def trainer(policy):
        objective = make_objective(lambda rows: data, training_prefix=4, action_dim=2)
        objective.identity = {**objective.identity, "adapters": policy.net._posttrain_adapter_identity}
        optimizer = torch.optim.AdamW([p for p in policy.net.parameters() if p.requires_grad], lr=0.003, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        return LLMTrainer(
            policy,
            objective,
            optimizer,
            checkpoint_format="dcp",
            save_async=False,
            lr_scheduler=scheduler,
            require_gradients=True,
        )

    original = trainer(build_policy(base, model_config, adapters=cfg, device="cpu"))
    original.train([0])
    original.train([0])
    checkpoint = str(tmp_path / "checkpoint")
    original.save(checkpoint, data_state={"next_row": 2})
    original.finalize_checkpoints()
    expected_report = original.train([0])
    expected = copy.deepcopy(original.model.net.state_dict())
    expected_opt = copy.deepcopy(original.opt.state_dict())
    monkeypatch.setattr(torch, "svd_lowrank", lambda *a, **kw: pytest.fail("resume repeated PiSSA decomposition"))
    resumed = trainer(build_policy(base, model_config, adapters=cfg, adapter_resume=True, device="cpu"))
    assert resumed.load(checkpoint) == 2
    actual_report = resumed.train([0])
    assert actual_report["train/loss_avg"] == expected_report["train/loss_avg"]
    assert load_component(checkpoint, "data") == {"next_row": 2}
    assert resumed.lr_scheduler.state_dict() == original.lr_scheduler.state_dict()
    for name, value in resumed.model.net.state_dict().items():
        torch.testing.assert_close(value, expected[name], atol=0, rtol=0)
    for index, values in resumed.opt.state_dict()["state"].items():
        for name, value in values.items():
            torch.testing.assert_close(value, expected_opt["state"][index][name], atol=0, rtol=0)
    assert_trainability(resumed.model.net, resumed.opt)
