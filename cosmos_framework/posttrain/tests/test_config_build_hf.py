# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""build_hf_model FSDP path (world>1 raise removed) + the current_pdims accessor default.

Single-device (`pdims=None`) must build a working HFModel with no raise — proving the old
`world_size>1 -> NotImplementedError` gate is gone — and `current_pdims()` must return None
outside any Ray actor's setup window.
"""

import pytest

torch = pytest.importorskip("torch")


@pytest.mark.L0
@pytest.mark.CPU
def test_build_hf_model_single_device_no_raise(tmp_path):
    pytest.importorskip("transformers")
    from transformers import GPT2Config, GPT2LMHeadModel

    from cosmos_framework.posttrain.config.model import HFModelConfig
    from cosmos_framework.posttrain.model.hf import build_hf_model

    # Save a tiny GPT2 to disk so build_hf_model can AutoConfig + load it.
    cfg = GPT2Config(n_layer=2, n_head=2, n_embd=32, vocab_size=256, n_positions=32)
    GPT2LMHeadModel(cfg).save_pretrained(tmp_path)

    hf_cfg = HFModelConfig(
        name_or_path=str(tmp_path),
        dtype="float32",
        attn_impl="eager",
    )
    # pdims=None -> single-device; must NOT raise (old world_size>1 gate is gone).
    model = build_hf_model(hf_cfg, pdims=None)

    import torch.nn as nn

    assert isinstance(model.net, nn.Module)  # .net present
    # single-device materializes to cuda on a GPU host, cpu otherwise — feed inputs on the
    # model's own device.
    dev = next(model.net.parameters()).device
    out = model(input_ids=torch.zeros(1, 4, dtype=torch.long, device=dev))  # forward produces logits
    assert out.shape[-1] == cfg.vocab_size


@pytest.mark.L0
@pytest.mark.CPU
def test_current_pdims_default_none():
    # Outside any Ray actor's run_setup window the module-global accessor is None,
    # so build_backbone falls through to its pure-FSDP / single-device resolution.
    pytest.importorskip("ray")  # workers.ray imports ray at module import
    from cosmos_framework.posttrain.workers.ray import current_pdims

    assert current_pdims() is None


@pytest.mark.L1
@pytest.mark.GPU
def test_build_backbone_hf_shards_over_actual_dist_world(monkeypatch):
    """Regression: the local/torchrun HF path must size the FSDP mesh from the ACTUAL distributed
    world (`dist.get_world_size()`), NOT the `world_size` parameter. A caller invoking
    `build_backbone(config)` (world_size defaulting to 1) under a >1-rank torchrun job must still
    get a sharded ParallelDims spanning the whole group — otherwise FSDP is silently skipped and a
    single-device model is built on every rank."""
    if int(__import__("os").environ.get("WORLD_SIZE", "1")) < 2:
        # torchrun-mode test: needs a real multi-rank env:// world (torchrun --nproc_per_node>=2).
        # WORLD_SIZE-based (not RANK) because other tests leak single-rank env defaults.
        pytest.skip("torchrun-mode test: run under torchrun --nproc_per_node>=2 (WORLD_SIZE>=2)")
    pytest.importorskip("ray")  # build_backbone imports workers.ray (current_pdims)
    import torch
    import torch.distributed as dist

    if not dist.is_initialized():
        dist.init_process_group("nccl")
    world = dist.get_world_size()
    if world < 2:
        pytest.skip("needs >=2 GPUs")
    torch.cuda.set_device(dist.get_rank())

    from types import SimpleNamespace

    import cosmos_framework.posttrain.config.build as build_mod
    from cosmos_framework.posttrain.config.model import HFModelConfig

    # Identity-instantiate + capture the pdims handed to build_hf_model, so we test the world-size
    # RESOLUTION (the bug) without paying for a real model build. The real mesh is still built over
    # the live process group inside build_backbone before build_hf_model is called.
    captured = {}
    monkeypatch.setattr("cosmos_framework.utils.lazy_config.instantiate", lambda x: x)
    monkeypatch.setattr(
        "cosmos_framework.posttrain.model.hf.build_hf_model",
        lambda cfg, pdims: captured.setdefault("pdims", pdims),
    )

    config = SimpleNamespace(model_backend="hf", model_hf=HFModelConfig())
    build_mod.build_backbone(config)  # world_size NOT passed -> defaults to 1 (the bug trigger)

    pd = captured.get("pdims")
    assert pd is not None, "FSDP silently skipped: build_backbone(config) built single-device under torchrun world>1"
    assert pd.world_size == world and pd.dp_shard == world  # mesh spans the ACTUAL world, not the param
    assert pd.dp_shard_enabled and pd.dp_shard_mesh.size() == world


@pytest.mark.L0
@pytest.mark.CPU
def test_build_lr_scheduler_linear_warmup_then_constant():
    """`OptimizerConfig.warmup_steps` (cosmos-rl `optm_warmup_steps`) must actually reach the
    optimizer. It was set to 20 in the GSM8K recipe and advertised as TOML parity while NOTHING
    consumed it, so that run trained at full LR from step 1 while cosmos-rl — which does build LR
    schedulers — ramped."""
    torch = pytest.importorskip("torch")
    from cosmos_framework.posttrain.config.build import build_lr_scheduler
    from cosmos_framework.posttrain.config.rl import OptimizerConfig

    m = torch.nn.Linear(2, 2)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-6)
    sched = build_lr_scheduler(opt, OptimizerConfig(lr=1e-6, warmup_steps=5))
    seen = []
    for _ in range(8):
        seen.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    # 1-based ramp: the FIRST step is 1/5 of target, not zero (a zero-LR step wastes an update).
    assert seen[0] == pytest.approx(2e-7, rel=1e-6)
    assert seen[4] == pytest.approx(1e-6, rel=1e-6)
    assert seen[7] == pytest.approx(1e-6, rel=1e-6)  # constant after warmup
    assert seen == sorted(seen)  # monotone non-decreasing


@pytest.mark.L0
@pytest.mark.CPU
def test_build_lr_scheduler_disabled_returns_none():
    """warmup_steps<=0 keeps the pre-existing no-scheduler path exactly as it was."""
    torch = pytest.importorskip("torch")
    from cosmos_framework.posttrain.config.build import build_lr_scheduler
    from cosmos_framework.posttrain.config.rl import OptimizerConfig

    opt = torch.optim.AdamW(torch.nn.Linear(2, 2).parameters(), lr=1e-6)
    assert build_lr_scheduler(opt, OptimizerConfig(warmup_steps=0)) is None
    assert OptimizerConfig().warmup_steps == 0  # off by default
