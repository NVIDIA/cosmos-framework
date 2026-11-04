# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Observe final optimizer gradients under both FSDP2 data-parallel layouts."""

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def test_execution_identity_uses_effective_hf_mesh_and_checkpointing(monkeypatch):
    from types import SimpleNamespace

    from cosmos_framework.posttrain.config.build import resolve_policy_parallelism
    from cosmos_framework.posttrain.config.model import HFModelConfig, HFParallelismConfig
    from cosmos_framework.posttrain.entrypoints.sft import adapter_execution_identity

    monkeypatch.setattr("cosmos_framework.posttrain.utils.dist.world_size_or_1", lambda: 4)
    model = HFModelConfig(
        attn_impl="eager", grad_ckpt=True, parallelism=HFParallelismConfig(dp_replicate=2, dp_shard=-1)
    )
    config = SimpleNamespace(model_backend="hf", model_hf=model)
    assert resolve_policy_parallelism(config, 4) == (2, 2)
    initial = adapter_execution_identity(config)
    assert initial["activation_checkpointing"] == "full"
    assert (initial["dp_replicate"], initial["dp_shard"]) == (2, 2)
    model.parallelism.dp_shard = 2
    assert adapter_execution_identity(config) == initial
    model.parallelism.dp_replicate, model.parallelism.dp_shard = 1, 4
    assert adapter_execution_identity(config) != initial
    model.grad_ckpt = False
    assert adapter_execution_identity(config)["activation_checkpointing"] == "none"
    model.attn_impl = "sdpa"
    assert adapter_execution_identity(config)["attention"] == "sdpa"


def test_adapter_checkpoint_binds_execution_but_portable_warm_start_does_not(tmp_path, monkeypatch):
    from cosmos_framework.posttrain.checkpoint.adapters import save_adapter_artifact
    from cosmos_framework.posttrain.scripts.prepare_sft_example import prepare_example
    from cosmos_framework.posttrain.tests.helpers.distributed import init_gloo_process_group
    from cosmos_framework.posttrain.tests.qualify_adapters import runtime, train_step, trainable_state
    from cosmos_framework.posttrain.tests.test_adapters_native import tiny_snapshot

    torch.manual_seed(42)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    tiny_snapshot(tmp_path / "base")
    prepare_example(tmp_path / "data", train_samples=8, validation_samples=8)
    for name, value in {
        "SFT_MODEL_PATH": str(tmp_path / "base"),
        "SFT_DATA_ROOT": str(tmp_path / "data"),
        "SFT_ADAPTER_METHOD": "lora",
        "SFT_ADAPTER_RSLORA": "0",
        "SFT_QUALIFY_EXTRA_NORM": "0",
        "SFT_QUALIFY_COMPONENT": "all",
    }.items():
        monkeypatch.setenv(name, value)
    recipe = "cosmos_framework/posttrain/configs/sft_cosmos3_nano_adapters.py"
    opts = [
        "rl.max_steps=8",
        "rl.data.shuffle=False",
        "rl.keep_optim_last=0",
        "rl.validation.every_steps=0",
        "rl.validation.before_training=False",
        "rl.checkpoint_format=consolidated",
        "model.model_config.dtype=bfloat16",
        "model.model_config.activation_checkpointing=none",
        "model.model_config.attn_impl=eager",
        "rl.adapters.components.1.method=dora",
        "rl.adapters.components.2.method=pissa",
        "rl.adapters.components.2.layers=[0]",
        "rl.processor_kwargs.images_kwargs.size.shortest_edge=16",
        "rl.processor_kwargs.images_kwargs.size.longest_edge=64",
        "rl.processor_kwargs.videos_kwargs.size.shortest_edge=16",
        "rl.processor_kwargs.videos_kwargs.size.longest_edge=64",
        "rl.processor_kwargs.videos_kwargs.num_frames=2",
    ]
    init_gloo_process_group(0, 1, str(tmp_path / "rendezvous"))
    try:
        _, rl, data, _, trainer = runtime(recipe, [*opts, "rl.resume=off"], str(tmp_path / "train"))
        for _ in range(2):
            train_step(data, trainer, rl)
        checkpoint = str(tmp_path / "bound")
        trainer.save(checkpoint, data_state=data.state_dict())
        trainer.finalize_checkpoints()
        train_step(data, trainer, rl)
        expected = trainable_state(trainer.model.net)
        _, _, resumed_data, _, resumed = runtime(recipe, [*opts, f"rl.resume={checkpoint}"], str(tmp_path / "resume"))
        train_step(resumed_data, resumed, rl)
        assert all(torch.equal(value, trainable_state(resumed.model.net)[name]) for name, value in expected.items())
        for changed in ("model.model_config.attn_impl=sdpa", "model.model_config.activation_checkpointing=full"):
            with pytest.raises(RuntimeError, match="objective exact-resume identity differs"):
                runtime(recipe, [*opts, f"rl.resume={checkpoint}", changed], str(tmp_path / "refused"))
        # A valid legacy transaction has no trustworthy execution binding. Do not
        # silently assume it used SDPA or rewrite its saved identity to match.
        resumed.objective_identity.pop("adapter_execution")
        legacy = str(tmp_path / "legacy")
        resumed.save(legacy, data_state=resumed_data.state_dict())
        resumed.finalize_checkpoints()
        with pytest.raises(RuntimeError, match="original runtime/configuration"):
            runtime(recipe, [*opts, f"rl.resume={legacy}"], str(tmp_path / "legacy-refused"))
        portable = str(tmp_path / "portable")
        save_adapter_artifact(resumed.model.net, portable)
        _, _, _, _, warm = runtime(
            recipe,
            [*opts, "rl.resume=off", f"rl.adapters.warm_start={portable}", "model.model_config.attn_impl=sdpa"],
            str(tmp_path / "warm"),
        )
        assert warm.optimizer_is_pristine() and warm._update_state_dict()["optimizer_step"] == 0
        assert all(torch.equal(value, trainable_state(warm.model.net)[name]) for name, value in expected.items())
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("attention", [None, "sdpa", "eager"])
def test_reload_preserves_recorded_attention(tmp_path, attention):
    import json

    from cosmos_framework.posttrain.tests.reload_adapters import recorded_attention

    record = {} if attention is None else {"configured_execution": {"attention": attention}}
    for rank in range(2):
        (tmp_path / f"training-rank{rank}.json").write_text(json.dumps(record))
    assert recorded_attention(tmp_path) == (attention or "sdpa")
    (tmp_path / "training-rank1.json").write_text(json.dumps({"configured_execution": {"attention": "invalid"}}))
    with pytest.raises(ValueError, match="recorded attention"):
        recorded_attention(tmp_path)


def _gradient_observer_worker(rank, world, replicate, path):
    from cosmos_framework.posttrain.model.parallel_dims import ParallelDims
    from cosmos_framework.posttrain.model.parallelize import parallelize
    from cosmos_framework.posttrain.model.specs import ShardPlan
    from cosmos_framework.posttrain.tests.helpers.distributed import init_gloo_process_group
    from cosmos_framework.posttrain.tests.qualify_adapters import local_value, observe_optimizer_gradients

    init_gloo_process_group(rank, world, path)
    try:
        torch.manual_seed(42)
        net = torch.nn.Linear(4, 4)
        dims = ParallelDims(dp_replicate=replicate, dp_shard=world // replicate, world_size=world)
        dims.build_device_mesh("cpu")
        parallelize(net, ShardPlan((), None, torch.float32), dims)
        optimizer = torch.optim.AdamW(net.parameters(), lr=0.01, weight_decay=0)
        before = {name: local_value(p).clone() for name, p in net.named_parameters()}
        records, handle = observe_optimizer_gradients(net, optimizer)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            net(torch.full((2, 4), float(rank + 1))).square().mean().backward()
            optimizer.step()
        handle.remove()
        assert all(row["observed"] and row["finite"] and row["nonzero_local"] for row in records.values()), records
        assert all(not torch.equal(before[name], local_value(p)) for name, p in net.named_parameters())
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(("replicate", "shard"), [(1, 2), (2, 1), (2, 2)])
def test_gradient_observer_uses_final_reduced_parameters(tmp_path, replicate, shard):
    world = replicate * shard
    mp.spawn(_gradient_observer_worker, args=(world, replicate, str(tmp_path / "rendezvous")), nprocs=world, join=True)
