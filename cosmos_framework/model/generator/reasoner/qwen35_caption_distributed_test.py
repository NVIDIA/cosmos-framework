# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Bounded single- or multi-node unified Qwen3.5 smoke, using a tiny random HF checkpoint.

Run with torchrun --standalone --nproc_per_node=4 -m
cosmos_framework.model.generator.reasoner.qwen35_caption_distributed_test.
For multiple nodes, use the normal torchrun rendezvous arguments and set
CAPTION_TEST_EXPECT_NODES and CAPTION_TEST_CHECKPOINT_DIR (shared by all nodes).
No hub downloads, database reads, or production weights are used.
"""

from __future__ import annotations

import os
import socket
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from safetensors.torch import save_file
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict, set_state_dict

from cosmos_framework.utils.config import CheckpointConfig, ObjectStoreConfig
from cosmos_framework.callbacks.qwen35_compile_warmup import warmup_caption_kernels
from cosmos_framework.configs.base.defaults.parallelism import ParallelismConfig
from cosmos_framework.configs.base.defaults.reasoner import VLMConfig
from cosmos_framework.configs.base.reasoner.defaults.policy_config import PolicyConfig, VLMModelConfig
from cosmos_framework.model.generator.reasoner.qwen35_caption import qwen35_fp32_decay_modules
from cosmos_framework.model.generator.vlm_model import VLMModel

if TYPE_CHECKING:
    from transformers import Qwen3_5Config


def _tiny_config() -> Qwen3_5Config:
    from transformers import Qwen3_5Config

    return Qwen3_5Config(
        text_config=dict(
            vocab_size=256,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=32,
            linear_num_key_heads=2,
            linear_num_value_heads=2,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            layer_types=["linear_attention", "full_attention"],
            rope_parameters=dict(
                rope_type="default",
                rope_theta=10000.0,
                partial_rotary_factor=1.0,
                mrope_section=[4, 4, 8],
                mrope_interleaved=True,
            ),
            pad_token_id=0,
        ),
        vision_config=dict(
            depth=1,
            hidden_size=32,
            intermediate_size=64,
            num_heads=2,
            patch_size=2,
            spatial_merge_size=2,
            temporal_patch_size=2,
            out_hidden_size=64,
            num_position_embeddings=16,
        ),
        image_token_id=250,
        video_token_id=251,
        vision_start_token_id=252,
        vision_end_token_id=253,
    )


def _batch(rank: int, step: int) -> dict[str, Any]:
    # Rotate modality by rank and step, including a rank that calls both towers.
    modality = (rank + step) % 4
    ids = [10, 11]
    mm = [0, 0]
    inputs = {}
    for active, token, kind in ((modality in (1, 3), 250, "image"), (modality in (2, 3), 251, "video")):
        if active:
            ids += [252] + [token] * 4 + [253, 12]
            mm += [0] + [1 if kind == "image" else 2] * 4 + [0, 0]
            inputs["pixel_values" if kind == "image" else "pixel_values_videos"] = torch.randn(16, 24, device="cuda")
            inputs[f"{kind}_grid_thw"] = torch.tensor([[1, 4, 4]], device="cuda")
    ids += [13, 14, 15, 16]
    mm += [0] * 4
    input_ids = torch.tensor([ids], device="cuda")
    labels = torch.full_like(input_ids, -100)
    labels[:, -4:] = input_ids[:, -4:]
    inputs.update(
        input_ids=input_ids,
        labels=labels,
        mm_token_type_ids=torch.tensor([mm], device="cuda"),
        attention_mask=torch.ones_like(input_ids),
    )
    return inputs


def _check_accumulation_gradient_parity(
    model: VLMModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    rank: int,
) -> None:
    batches = [_batch(rank, micro) for micro in range(2)]
    for batch, count in zip(batches, (1, 4), strict=True):
        batch["labels"].fill_(-100)
        batch["labels"][:, -count:] = batch["input_ids"][:, -count:]
    denominator = sum((batch["labels"][:, 1:] != -100).sum().float().sqrt() for batch in batches)
    dist.all_reduce(denominator)

    # Independent reference: sum sample numerators with one denominator across
    # both microbatches and every DP rank, without the VLM accumulation hooks.
    optimizer.zero_grad(set_to_none=True)
    for batch in batches:
        result = model.model(**batch)
        (result.loss * result.stats.global_objective_denominator / denominator).backward()
    expected = {name: parameter.grad.to_local().clone() for name, parameter in model.named_parameters()}
    assert model._weighted_ce_window_denominator is None

    optimizer.zero_grad(set_to_none=True)
    for batch in batches:
        output, loss = model.training_step(dict(batch), iteration=0)
        (output.get("_backward_loss", loss) / len(batches)).backward()
    model.on_before_optimizer_step(optimizer, scheduler, iteration=0)
    assert model._weighted_ce_window_denominator is None
    assert model._weighted_ce_window_microbatches == 0
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter.grad.to_local(), expected[name], atol=0.003, rtol=0.05, msg=name)
    optimizer.zero_grad(set_to_none=True)
    if rank == 0:
        print("QWEN35_ACCUMULATION_PARITY_PASS microbatches=2 supervised_counts=1,4 all_parameters", flush=True)


def main() -> None:
    # CI collects this module with an older Transformers image. Load Qwen3.5
    # only when this explicit torchrun smoke is invoked in its supported image.
    from transformers import Qwen3_5ForConditionalGeneration

    rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(minutes=5))
    hosts: list[str | None] = [None] * dist.get_world_size()
    dist.all_gather_object(hosts, socket.gethostname())
    node_count = len(set(hosts))
    expected_nodes = int(os.environ.get("CAPTION_TEST_EXPECT_NODES", "1"))
    assert node_count == expected_nodes, (hosts, expected_nodes)
    if rank == 0:
        print(f"QWEN35_TOPOLOGY nodes={node_count} world_size={dist.get_world_size()} hosts={hosts}", flush=True)
    checkpoint_root = os.environ.get("CAPTION_TEST_CHECKPOINT_DIR")
    if node_count > 1 and not checkpoint_root:
        raise ValueError("Multi-node checkpoint validation requires a shared CAPTION_TEST_CHECKPOINT_DIR")
    root = Path(os.environ.get("TMPDIR", "/tmp")) / "qwen35-caption-tiny"
    # Each node has independent temporary storage. Stage identical weights once
    # per node before any rank initializes the unified model from that local path.
    if int(os.environ["LOCAL_RANK"]) == 0:
        torch.manual_seed(88)
        model = Qwen3_5ForConditionalGeneration(_tiny_config())
        # Match the published 27B checkpoint's canonical model.language_model keys.
        # Avoid save_pretrained's version-dependent conversion of freshly built models.
        model.config.save_pretrained(root)
        canonical_state = {key: value.contiguous() for key, value in model.state_dict().items()}
        save_file(canonical_state, str(root / "model.safetensors"))
        del model
    dist.barrier()
    config = VLMModelConfig(
        policy=PolicyConfig(
            backbone=VLMConfig(model_name=str(root)),
            use_weighted_ce=True,
            weighted_ce_exponent=0.5,
            normalize_weighted_ce_over_accumulation_window=True,
            enable_fused_weighted_ce=True,
            qwen35_fp32_recurrent_a_log=True,
            attn_implementation="sdpa",
        ),
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=int(os.environ.get("CAPTION_TEST_DP_SHARD", dist.get_world_size())),
            data_parallel_replicate_degree=-1,
        ),
    )
    checkpoint = CheckpointConfig(load_from_object_store=ObjectStoreConfig(enabled=False), load_path="")
    model = VLMModel(config, checkpoint)
    before_warmup = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone()
    warmup_caption_kernels(model.model, max_length=128, bucket_multiple=32)
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter.to_local(), before_warmup[name].to_local(), rtol=0, atol=0)
        assert parameter.grad is None
    assert torch.equal(cpu_rng, torch.get_rng_state())
    assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
    assert model._weighted_ce_window_denominator is None
    del before_warmup
    if rank == 0:
        print("QWEN35_COMPILE_WARMUP_PASS buckets=4 parameters_and_rng_unchanged", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    _check_accumulation_gradient_parity(model, optimizer, scheduler, rank)
    for step in range(4):
        optimizer.zero_grad()
        for micro in range(2):
            output, loss = model.training_step(_batch(rank, step + micro), step)
            assert torch.isfinite(loss), (rank, step, loss)
            (output.get("_backward_loss", loss) / 2).backward()
        model.on_before_optimizer_step(optimizer, scheduler, iteration=step)
        decay = qwen35_fp32_decay_modules(model)
        assert decay and all(holder.A_log.dtype == torch.float32 for holder in decay)
        assert all(holder.A_log.grad is not None for holder in decay)
        assert all(p.grad is not None for p in model.model.model.model.visual.parameters())
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None, (rank, name, "missing gradient")
            gradient = parameter.grad.to_local() if hasattr(parameter.grad, "to_local") else parameter.grad
            assert torch.isfinite(gradient).all(), (rank, step, name, "non-finite gradient")
        optimizer.step()
        if rank == 0:
            print(f"QWEN35_UNIFIED_STEP={step} loss={loss.item():.6f}", flush=True)
    state, optimizer_state = get_state_dict(model, optimizer)
    state = {key: value.clone() for key, value in state.items()}
    expected_decay = [holder.A_log.clone() for holder in qwen35_fp32_decay_modules(model)]
    expected_optimizer = {
        name: {key: value.clone() for key, value in values.items()} for name, values in optimizer_state["state"].items()
    }
    disk_state = {"model": state, "optimizer": optimizer_state}
    checkpoint_path = Path(checkpoint_root) if checkpoint_root else root / "dcp"
    dcp.save(disk_state, checkpoint_id=checkpoint_path)
    with torch.no_grad():
        for holder in qwen35_fp32_decay_modules(model):
            holder.A_log.zero_()
        for value in state.values():
            value.zero_()
        for values in optimizer_state["state"].values():
            for value in values.values():
                value.zero_()
    dcp.load(disk_state, checkpoint_id=checkpoint_path)
    set_state_dict(model, optimizer, model_state_dict=state, optim_state_dict=optimizer_state)
    for holder, expected in zip(qwen35_fp32_decay_modules(model), expected_decay, strict=True):
        torch.testing.assert_close(holder.A_log.to_local(), expected.to_local())
    for name, values in optimizer_state["state"].items():
        for key, actual in values.items():
            expected = expected_optimizer[name][key]
            if hasattr(actual, "to_local"):
                actual, expected = actual.to_local(), expected.to_local()
            torch.testing.assert_close(actual, expected)
    # Full-state APIs are used by HF export and must use the same canonical keys.
    full_options = StateDictOptions(full_state_dict=True)
    full_state, full_optimizer_state = get_state_dict(model, optimizer, options=full_options)
    set_state_dict(
        model, optimizer, model_state_dict=full_state, optim_state_dict=full_optimizer_state, options=full_options
    )
    optimizer.zero_grad()
    output, loss = model.training_step(_batch(rank, 5), 5)
    assert torch.isfinite(loss)
    output.get("_backward_loss", loss).backward()
    model.on_before_optimizer_step(optimizer, scheduler, iteration=5)
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, (rank, name, "missing resumed gradient")
        gradient = parameter.grad.to_local() if hasattr(parameter.grad, "to_local") else parameter.grad
        assert torch.isfinite(gradient).all(), (rank, name, "non-finite resumed gradient")
    optimizer.step()
    dist.barrier()
    if rank == 0:
        print(
            f"QWEN35_UNIFIED_DISTRIBUTED_PASS: nodes={node_count}, mixed modalities, AC, finite gradients, "
            "FP32 decay, accumulated fused CE gradient parity, disk DCP and full-state optimizer restore",
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
