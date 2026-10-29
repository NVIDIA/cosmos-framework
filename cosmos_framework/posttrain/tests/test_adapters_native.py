# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Actual native dense Qwen3-VL blocks, multimodal gradients and trainer recovery."""

from __future__ import annotations

import copy
import json
import subprocess
import sys

import pytest
import torch

from cosmos_framework.posttrain.checkpoint.adapters import checkpoint_adapter_state, prepare_merged_state
from cosmos_framework.posttrain.config.adapters import AdapterComponentConfig as Component
from cosmos_framework.posttrain.config.adapters import AdapterConfig
from cosmos_framework.posttrain.model.adapters import assert_trainability, install_adapters
from cosmos_framework.posttrain.model.qwen3_vl import (
    Qwen3VLBuildHooks,
    build_qwen3_vl_policy,
    override_qwen3_vl_forward,
)
from cosmos_framework.posttrain.tests.test_adapters import identity
from cosmos_framework.posttrain.tests.test_model_qwen3_vl_gpu import _batch, _dense_config


def adapter_config(method: str, *, dropout: float = 0) -> AdapterConfig:
    return AdapterConfig(
        components=[
            Component(
                component="language",
                targets=[
                    "model.language_model.layers.*.self_attn.q_proj",
                    "model.language_model.layers.*.self_attn.v_proj",
                ],
                rank=2,
                method=method,
                dropout=dropout,
            ),
            Component(
                component="projector",
                targets=["model.visual.merger.linear_fc1", "model.visual.deepstack_merger_list.*.linear_fc2"],
                rank=2,
                method=method,
                dropout=dropout,
            ),
            Component(
                component="vision",
                targets=["model.visual.blocks.*.attn.qkv"],
                layers=[0],
                rank=2,
                method=method,
                dropout=dropout,
            ),
        ]
    )


def make_policy(*, dtype=torch.float32):
    return build_qwen3_vl_policy(
        model_type="qwen3_vl",
        model_config=_dense_config(),
        dtype=dtype,
        hooks=Qwen3VLBuildHooks(prepare_model=lambda net: override_qwen3_vl_forward(net.model)),
        initialize_on_meta=False,
    )


def batches():
    image = _batch()
    text = {"input_ids": torch.tensor([[1, 2, 3, 4, 5]]), "attention_mask": torch.ones(1, 5, dtype=torch.bool)}
    video = {
        "input_ids": image["input_ids"].clone(),
        "attention_mask": image["attention_mask"],
        "pixel_values_videos": image["pixel_values"].clone(),
        "video_grid_thw": image["image_grid_thw"].clone(),
    }
    video["input_ids"][0, 2] = 121
    return [text, image, video]


@pytest.mark.parametrize("method", ["lora", "dora", "pissa"])
@pytest.mark.parametrize("activation_checkpointing", [False, True])
def test_native_all_components_media_gradients_and_merge(method, activation_checkpointing):
    torch.manual_seed(37)
    policy = make_policy()
    data = batches()
    policy.net.eval()
    with torch.no_grad():
        baseline = [policy.forward_policy_batch(b).logits.clone() for b in data]
    manifest = install_adapters(policy.net, adapter_config(method))
    ident = identity(manifest)
    with torch.no_grad():
        for batch, expected in zip(data, baseline):
            torch.testing.assert_close(policy.forward_policy_batch(batch).logits, expected, atol=2e-6, rtol=2e-5)
    if activation_checkpointing:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

        for blocks in (policy.net.model.language_model.layers, policy.net.model.visual.blocks):
            for index in range(len(blocks)):
                blocks[index] = checkpoint_wrapper(blocks[index])  # same non-reentrant wrapper as apply_ac
    policy.net.train()
    opt = torch.optim.AdamW([p for p in policy.net.parameters() if p.requires_grad], lr=0.003, weight_decay=0)
    frozen = {n: p.detach().clone() for n, p in policy.net.named_parameters() if not p.requires_grad}
    before = {n: p.detach().clone() for n, p in policy.net.named_parameters() if p.requires_grad}
    for batch in data * 2:
        opt.zero_grad(set_to_none=True)
        logits = policy.forward_policy_batch(batch).logits
        loss = torch.nn.functional.cross_entropy(logits[:, -1].float(), torch.tensor([7]))
        loss.backward()
        assert torch.isfinite(loss)
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all() for p in policy.net.parameters() if p.requires_grad
        )
        opt.step()
    assert_trainability(policy.net, opt)
    for name, parameter in policy.net.named_parameters():
        if parameter.requires_grad:
            assert not torch.equal(parameter, before[name]), name
        else:
            assert torch.equal(parameter, frozen[name]), name
    policy.net.eval()
    merged_policy = make_policy()
    merged_policy.net.load_state_dict(prepare_merged_state(policy.net.state_dict(), ident))
    merged_policy.net.eval()
    with torch.no_grad():
        for batch in data:
            torch.testing.assert_close(
                policy.forward_policy_batch(batch).logits,
                merged_policy.forward_policy_batch(batch).logits,
                atol=5e-6,
                rtol=5e-5,
            )


@pytest.mark.parametrize("method", ["lora", "dora", "pissa"])
@pytest.mark.parametrize("checkpoint_format", ["consolidated", "dcp"])
def test_native_trainer_checkpoint_matches_next_update(tmp_path, monkeypatch, method, checkpoint_format):
    from cosmos_framework.posttrain.algorithm.primitives.objective import Objective, Reduction
    from cosmos_framework.posttrain.checkpoint.io import load_component
    from cosmos_framework.posttrain.roles.trainer.llm import LLMTrainer

    torch.manual_seed(17)
    data = batches()
    cfg = adapter_config(method, dropout=0.1 if method != "pissa" else 0)
    original = make_policy()
    base = copy.deepcopy(original.net.state_dict())
    manifest = install_adapters(original.net, cfg)
    ident = identity(manifest)

    def trainer(policy):
        objective = Objective(
            lambda model, batch: (model.forward_policy_batch(data[batch[0]]).logits.square().mean(), {}),
            reduction=Reduction.TOKEN_SUM,
            identity={"adapters": ident},
        )
        opt = torch.optim.AdamW([p for p in policy.net.parameters() if p.requires_grad], lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.9)
        return LLMTrainer(
            policy,
            objective,
            opt,
            checkpoint_format=checkpoint_format,
            save_async=False,
            lr_scheduler=scheduler,
            require_gradients=True,
        )

    uninterrupted = trainer(original)
    uninterrupted.train([1])
    uninterrupted.train([2])
    checkpoint = str(tmp_path / "iter_000002")
    uninterrupted.save(checkpoint, data_state={"next_row": 2})
    uninterrupted.train([1])
    expected = copy.deepcopy(original.net.state_dict())
    expected_opt = copy.deepcopy(uninterrupted.opt.state_dict())
    restored = make_policy()
    restored.net.load_state_dict(base)
    monkeypatch.setattr(torch, "svd_lowrank", lambda *a, **kw: pytest.fail("resume repeated PiSSA initialization"))
    install_adapters(restored.net, cfg, initialize=False)
    resumed = trainer(restored)
    assert resumed.load(checkpoint) == 2
    assert load_component(checkpoint, "data") == {"next_row": 2}
    resumed.train([1])
    assert resumed.lr_scheduler.state_dict() == uninterrupted.lr_scheduler.state_dict()
    for name, tensor in restored.net.state_dict().items():
        torch.testing.assert_close(tensor, expected[name], atol=0, rtol=0)
    for key, values in resumed.opt.state_dict()["state"].items():
        for name, tensor in values.items():
            torch.testing.assert_close(tensor, expected_opt["state"][key][name], atol=0, rtol=0)
    artifact_identity, state, provenance = checkpoint_adapter_state(checkpoint)
    assert artifact_identity == ident and provenance
    assert set(state) == set(expected)
    assert_trainability(restored.net, resumed.opt)


def test_loaded_base_hook_runs_before_distributed_registration():
    events = []
    hooks = Qwen3VLBuildHooks(
        model_factory=lambda config: torch.nn.Linear(4, 4),
        materialize=lambda net: events.append("materialize"),
        load_weights=lambda *a: events.append("load"),
        after_load_before_parallelize=lambda net: events.append("adapters"),
        parallelize=lambda *a: events.append("distributed"),
    )
    build_qwen3_vl_policy(model_type="qwen3_vl", model_config=_dense_config(), hooks=hooks, initialize_on_meta=False)
    assert events == ["materialize", "load", "adapters", "distributed"]


def test_bf16_extra_norm_keeps_activation_dtype_and_merged_reload():
    from cosmos_framework.posttrain.config.adapters import AdapterExtraModuleConfig

    torch.manual_seed(19)
    policy = make_policy(dtype=torch.bfloat16)
    cfg = adapter_config("lora")
    cfg.extra_modules = [AdapterExtraModuleConfig(targets=["model.language_model.norm"])]
    manifest = install_adapters(policy.net, cfg)
    ident = {**identity(manifest), "base_storage_dtype": "bfloat16"}
    opt = torch.optim.AdamW([p for p in policy.net.parameters() if p.requires_grad], lr=0.01)
    norm = policy.net.model.language_model.norm
    before = norm.weight.detach().clone()
    batch = _batch()
    batch["pixel_values"] = batch["pixel_values"].bfloat16()
    for _ in range(3):
        opt.zero_grad()
        loss = policy.forward_policy_batch(batch).logits.float().square().mean()
        loss.backward()
        opt.step()
    assert norm.weight.dtype == torch.float32 and not torch.equal(before, norm.weight)
    assert norm(torch.randn(2, 32, dtype=torch.bfloat16)).dtype == torch.bfloat16
    merged = make_policy(dtype=torch.bfloat16)
    merged.net.load_state_dict(prepare_merged_state(policy.net.state_dict(), ident), strict=True)
    with torch.no_grad():
        torch.testing.assert_close(
            policy.forward_policy_batch(batch).logits, merged.forward_policy_batch(batch).logits, atol=0.015, rtol=0.02
        )


def test_real_meta_builder_loads_base_before_pissa_and_preserves_resume(tmp_path, monkeypatch):
    from cosmos_framework.posttrain.config.model import Qwen3VLModelConfig
    from cosmos_framework.posttrain.model.qwen3_vl import build_qwen3_vl_model

    base = make_policy()
    from safetensors.torch import save_file

    base.net.config.save_pretrained(tmp_path)
    save_file(base.net.state_dict(), str(tmp_path / "model.safetensors"))
    config = Qwen3VLModelConfig(name_or_path=str(tmp_path), dtype="float32", activation_checkpointing="full")
    net = build_qwen3_vl_model(config, adapters=adapter_config("pissa")).net
    assert not any(t.is_meta for t in net.parameters())
    assert_trainability(net)
    monkeypatch.setattr(torch, "svd_lowrank", lambda *a, **kw: pytest.fail("exact resume decomposed base"))
    restored = build_qwen3_vl_model(config, adapters=adapter_config("pissa"), adapter_resume=True).net
    restored.load_state_dict(net.state_dict(), strict=True)
    assert net._posttrain_adapter_identity == restored._posttrain_adapter_identity


def tiny_snapshot(path, *, tied=False):
    """Real tokenizer/processor, native model, indexed safetensors for HF conversion."""
    from safetensors.torch import save_file

    from cosmos_framework.posttrain.tests.helpers.sft_packer import local_processor, local_tokenizer

    path.mkdir()
    tokenizer = local_tokenizer()
    config = _dense_config()
    config.tie_word_embeddings = config.text_config.tie_word_embeddings = tied
    config.text_config.vocab_size = len(tokenizer)
    config.image_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    config.video_token_id = tokenizer.convert_tokens_to_ids("<|video_pad|>")
    config.vision_start_token_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    config.vision_end_token_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")
    config.architectures = ["Qwen3VLForConditionalGeneration"]
    policy = build_qwen3_vl_policy(
        model_type="qwen3_vl", model_config=config, dtype=torch.float32, initialize_on_meta=False
    )
    policy.net.config.save_pretrained(path)
    local_processor(tokenizer).save_pretrained(path)
    state = policy.net.state_dict()
    if tied:
        del state["lm_head.weight"]
    shard = "model-00001-of-00001.safetensors"
    save_file(state, str(path / shard))
    (path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": sum(t.numel() * t.element_size() for t in state.values())},
                "weight_map": {n: shard for n in state},
            }
        )
    )


def test_architecture_embedding_tie_survives_portable_and_merged_reload(tmp_path):
    from transformers import AutoModelForImageTextToText

    from cosmos_framework.posttrain.checkpoint.adapters import save_adapter_artifact
    from cosmos_framework.posttrain.checkpoint.tools.adapters import export_merged_adapter
    from cosmos_framework.posttrain.config.model import Qwen3VLModelConfig
    from cosmos_framework.posttrain.model.qwen3_vl import build_qwen3_vl_model

    base = tmp_path / "base"
    tiny_snapshot(base, tied=True)
    cfg = Qwen3VLModelConfig(name_or_path=str(base), dtype="float32", activation_checkpointing="none")
    adapters = adapter_config("dora")
    original = build_qwen3_vl_model(cfg, adapters=adapters)
    assert original.net.lm_head.weight is original.net.model.language_model.embed_tokens.weight
    assert not original.net.lm_head.weight.requires_grad
    with torch.no_grad():
        original.net.model.language_model.layers[0].self_attn.q_proj.lora_B.weight.add_(0.01)
    portable = tmp_path / "portable"
    save_adapter_artifact(original.net, str(portable))
    adapters.warm_start = str(portable)
    reloaded = build_qwen3_vl_model(cfg, adapters=adapters)
    assert reloaded.net.lm_head.weight is reloaded.net.model.language_model.embed_tokens.weight
    merged = tmp_path / "merged"
    export_merged_adapter(str(portable), str(base), str(merged))
    dense = AutoModelForImageTextToText.from_pretrained(merged, dtype=torch.float32).eval()
    assert dense.lm_head.weight is dense.model.language_model.embed_tokens.weight
    batch = {"input_ids": torch.tensor([[30, 31, 32, 33]])}
    original.net.eval()
    reloaded.net.eval()
    with torch.no_grad():
        expected = original.forward_policy_batch(batch).logits
        torch.testing.assert_close(reloaded.forward_policy_batch(batch).logits, expected, atol=0, rtol=0)
        torch.testing.assert_close(dense(**batch).logits, expected, atol=3e-5, rtol=3e-5)
    from cosmos_framework.posttrain.tests.adapter_numerics import verify_promoted_merge

    control = verify_promoted_merge(reloaded, reloaded.net._posttrain_adapter_identity, [batch], str(merged))
    assert control["serialized_tensors_verified"] > 0


def test_promoted_merge_control_bf16_independent_methods_and_extra_norm(tmp_path):
    from cosmos_framework.posttrain.checkpoint.adapters import save_adapter_artifact
    from cosmos_framework.posttrain.checkpoint.tools.adapters import export_merged_adapter
    from cosmos_framework.posttrain.config.adapters import AdapterExtraModuleConfig
    from cosmos_framework.posttrain.config.model import Qwen3VLModelConfig
    from cosmos_framework.posttrain.model.qwen3_vl import build_qwen3_vl_model
    from cosmos_framework.posttrain.tests.adapter_numerics import verify_promoted_merge

    base = tmp_path / "base"
    tiny_snapshot(base)
    config = adapter_config("lora")
    config.components[1].method = "dora"
    config.components[2].method = "pissa"
    config.extra_modules = [AdapterExtraModuleConfig(targets=["model.language_model.norm"])]
    policy = build_qwen3_vl_model(
        Qwen3VLModelConfig(name_or_path=str(base), dtype="bfloat16", activation_checkpointing="none"), adapters=config
    )
    with torch.no_grad():
        for parameter in policy.net.parameters():
            if parameter.requires_grad:
                parameter.add_(0.002)
    identity = policy.net._posttrain_adapter_identity
    save_adapter_artifact(policy.net, str(tmp_path / "portable"))
    export_merged_adapter(str(tmp_path / "portable"), str(base), str(tmp_path / "merged"))
    from transformers import AutoModelForImageTextToText

    from cosmos_framework.posttrain.checkpoint.conversion.hf import config_with_export_dtype

    merged_path = tmp_path / "merged"
    original_config = json.loads((base / "config.json").read_text())
    assert json.loads((merged_path / "config.json").read_text()) == config_with_export_dtype(original_config, "float32")
    dense = AutoModelForImageTextToText.from_pretrained(merged_path, dtype="auto")
    assert all(p.dtype == torch.float32 for p in dense.parameters())
    # Repeated export remains identity-bound and does not rewrite the original base.
    assert export_merged_adapter(str(tmp_path / "portable"), str(base), str(merged_path)) == "skipped"
    assert json.loads((base / "config.json").read_text()) == original_config
    control = verify_promoted_merge(
        policy, identity, [{"input_ids": torch.tensor([[30, 31, 32, 33]])}], str(tmp_path / "merged")
    )
    assert control["serialized_tensors_verified"] > 0
    assert control["hf_fp32"]["maximum_logit_error"] < 3e-5


@pytest.mark.parametrize("component", ["all", "language", "projector", "vision"])
def test_qualification_harness_reaches_training_resume_and_export(tmp_path, monkeypatch, component):
    from cosmos_framework.posttrain.scripts.prepare_sft_example import prepare_example
    from cosmos_framework.posttrain.tests import qualify_adapters as harness

    base, data, output = tmp_path / "base", tmp_path / "data", tmp_path / "output"
    tiny_snapshot(base)
    prepare_example(data, train_samples=8, validation_samples=8)
    monkeypatch.setenv("SFT_MODEL_PATH", str(base))
    monkeypatch.setenv("SFT_DATA_ROOT", str(data))
    monkeypatch.setenv("SFT_QUALIFY_COMPONENT", "all")
    for key, value in [("SFT_ADAPTER_METHOD", "lora"), ("SFT_ADAPTER_RSLORA", "0"), ("SFT_QUALIFY_EXTRA_NORM", "0")]:
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(harness, "attest", lambda *_: {"container_sha256": "0" * 64})
    monkeypatch.setattr(harness, "init_local_process_group", lambda: None)
    monkeypatch.setattr(harness.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(harness.dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(harness.dist, "barrier", lambda: None)
    monkeypatch.setattr(harness.dist, "destroy_process_group", lambda: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    original_load = harness.load_config

    def tiny_recipe(recipe, opts):
        config = original_load(recipe, opts)
        config.model.model_config.dtype = "float32"
        config.rl.adapters.components[2].layers = [0]
        for group in ("images_kwargs", "videos_kwargs"):
            config.rl.processor_kwargs[group]["size"] = {"shortest_edge": 16, "longest_edge": 64}
        config.rl.processor_kwargs["videos_kwargs"]["num_frames"] = 2
        return config

    monkeypatch.setattr(harness, "load_config", tiny_recipe)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qualify",
            "--wheel",
            "fixture",
            "--build-record",
            "fixture",
            "--out",
            str(output),
            "--method",
            "mixed" if component == "all" else "lora",
            "--component",
            component,
            "--override",
            "rl.grad_accum=2" if component == "all" else "rl.grad_accum=1",
        ],
    )
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        harness.main()
    finally:
        torch.use_deterministic_algorithms(previous)
    report = json.loads((output / "training-rank0.json").read_text())
    assert report["changed_tensors"] == report["trainable_tensors"]
    assert {row["component"] for row in report["resolved"]["targets"]} == (
        {"language", "projector", "vision"} if component == "all" else {component}
    )
    assert report["resume_max_abs_parameter_error"] == 0
    assert len(report["train"]) == 9
    assert (output / "portable/adapter_manifest.json").is_file()


@pytest.mark.parametrize("method", ["lora", "dora", "pissa"])
def test_sft_recipe_real_media_and_checkpoint_exports(tmp_path, monkeypatch, method):
    from cosmos_framework.posttrain.checkpoint.adapters import read_adapter_artifact
    from cosmos_framework.posttrain.checkpoint.tools.adapters import main as export_main
    from cosmos_framework.posttrain.entrypoints.sft import run
    from cosmos_framework.posttrain.scripts.prepare_sft_example import prepare_example
    from cosmos_framework.posttrain.tests.test_entrypoints_sft import _quiet_loop
    from cosmos_framework.posttrain.utils.entry import out_dir
    from cosmos_framework.utils.config import load_config

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr("cosmos_framework.posttrain.utils.dist.init_local_process_group", lambda: None)
    _quiet_loop(monkeypatch)
    model_path, data_path = tmp_path / "model", tmp_path / "data"
    tiny_snapshot(model_path)
    prepare_example(data_path, train_samples=4, validation_samples=4)
    monkeypatch.setenv("SFT_MODEL_PATH", str(model_path))
    monkeypatch.setenv("SFT_DATA_ROOT", str(data_path))
    monkeypatch.setenv("SFT_ADAPTER_METHOD", method)
    monkeypatch.setenv("IMAGINAIRE_OUTPUT_ROOT", str(tmp_path / "outputs"))
    recipe = "cosmos_framework/posttrain/configs/sft_cosmos3_nano_adapters.py"
    opts = [
        "rl.max_steps=2",
        "rl.micro_batch_size=4",
        "rl.save_every=1",
        "rl.resume=off",
        "rl.keep_optim_last=0",
        "rl.checkpoint_format=consolidated",
        "model.model_config.dtype=float32",
        "rl.adapters.components.2.layers=[0]",
        "rl.validation.every_steps=1",
        "rl.validation.batch_size=4",
        "rl.processor_kwargs.images_kwargs.size.shortest_edge=16",
        "rl.processor_kwargs.images_kwargs.size.longest_edge=64",
        "rl.processor_kwargs.videos_kwargs.size.shortest_edge=16",
        "rl.processor_kwargs.videos_kwargs.size.longest_edge=64",
        "rl.processor_kwargs.videos_kwargs.num_frames=2",
    ]
    config = load_config(recipe, ["--", *opts])
    result = run(config, recipe, opts, out_dir(config))
    assert result["final_step"] == 2 and result["final_loss"] > 0
    checkpoint = next((tmp_path / "outputs").rglob("iter_000002"))
    first_checkpoint = checkpoint.with_name("iter_000001")
    resume_opts = [x for x in opts if not x.startswith("rl.resume=")] + [
        f"rl.resume={first_checkpoint}",
        "job.name=resume",
    ]
    resumed_config = load_config(recipe, ["--", *resume_opts])
    continued = run(resumed_config, recipe, resume_opts, out_dir(resumed_config))
    assert continued["start"] == 1 and continued["final_step"] == 2
    from cosmos_framework.posttrain.checkpoint.io import load_component

    resumed_checkpoint = next(p for p in (tmp_path / "outputs").rglob("iter_000002") if p != checkpoint)
    expected_state = load_component(str(checkpoint), "model")["net"]
    actual_state = load_component(str(resumed_checkpoint), "model")["net"]
    assert all(torch.equal(expected_state[n], actual_state[n]) for n in expected_state)
    assert load_component(str(checkpoint), "data") == load_component(str(resumed_checkpoint), "data")
    portable = tmp_path / "portable"
    export_main(["--checkpoint", str(checkpoint), "--out", str(portable)])
    ident, state = read_adapter_artifact(str(portable))
    assert ident["media"]["processor_kwargs"]["videos_kwargs"]["num_frames"] == 2
    assert {r["component"] for r in ident["resolved"]["targets"]} == {"language", "vision", "projector"}
    # Existing converter is exercised with complete sidecars and a real HF loader.
    merged = tmp_path / "merged"
    export_main(["--adapter", str(portable), "--base", str(model_path), "--merged", "--out", str(merged)])
    assert (merged / "config.json").is_file()
    from transformers import AutoModelForImageTextToText

    from cosmos_framework.posttrain.model.qwen3_vl import build_qwen3_vl_model

    dense = AutoModelForImageTextToText.from_pretrained(merged, dtype=torch.float32).eval()
    cfg = config.model.model_config
    cfg.name_or_path = str(model_path)
    cfg.dtype = "float32"
    cfg.activation_checkpointing = "none"
    # Test reload through the public builder using the exact recorded packer identity.
    config.rl.adapters.warm_start = str(portable)
    adapted = build_qwen3_vl_model(cfg, adapters=config.rl.adapters, adapter_media_identity=ident["media"])
    adapted.net.eval()
    from cosmos_framework.posttrain.algorithm.sft import make_objective
    from cosmos_framework.posttrain.data.packer.sft import build_sft_packer
    from cosmos_framework.posttrain.roles.env.sft import build_sft_env

    packer = build_sft_packer(
        str(model_path), max_length=config.rl.max_length, processor_kwargs=config.rl.processor_kwargs
    )
    objective = make_objective(packer)
    samples = build_sft_env(config.rl.data, config.rl.seed).get_samples(4)
    from cosmos_framework.posttrain.model.qwen3_vl import Qwen3VLPolicy

    with torch.no_grad():
        a, _ = objective(adapted, samples)
        b, _ = objective(Qwen3VLPolicy(dense, "qwen3_vl"), samples)
    torch.testing.assert_close(a, b, atol=3e-5, rtol=3e-5)
    probes = batches()
    probes[0]["input_ids"] = torch.tensor([[30, 31, 32, 33, 34]])
    for batch in probes:
        for old, new in [
            (118, adapted.net.config.vision_start_token_id),
            (119, adapted.net.config.vision_end_token_id),
            (120, adapted.net.config.image_token_id),
            (121, adapted.net.config.video_token_id),
        ]:
            batch["input_ids"][batch["input_ids"] == old] = new
        batch["mm_token_type_ids"] = torch.zeros_like(batch["input_ids"])
        batch["mm_token_type_ids"][batch["input_ids"] == adapted.net.config.image_token_id] = 1
        batch["mm_token_type_ids"][batch["input_ids"] == adapted.net.config.video_token_id] = 2
    with torch.no_grad():
        expected = [adapted.forward_policy_batch(batch).logits for batch in probes]
    fixture = tmp_path / "merge_probes.pt"
    torch.save((probes, expected), fixture)
    script = """
import sys, torch
from transformers import AutoModelForImageTextToText
model = AutoModelForImageTextToText.from_pretrained(sys.argv[1], dtype=torch.float32).eval()
data, expected = torch.load(sys.argv[2], weights_only=True)
with torch.no_grad():
    for batch, reference in zip(data, expected):
        torch.testing.assert_close(model(**batch, use_cache=False).logits, reference, atol=3e-5, rtol=3e-5)
"""
    check = subprocess.run([sys.executable, "-c", script, str(merged), str(fixture)], capture_output=True, text=True)
    assert check.returncode == 0, check.stderr
    from cosmos_framework.posttrain.tests.adapter_numerics import verify_promoted_merge

    control = verify_promoted_merge(adapted, ident, probes, str(merged))
    assert control["serialized_tensors_verified"] > 0
    assert control["hf_fp32"]["argmax_equal"] == control["hf_fp32"]["positions"]
