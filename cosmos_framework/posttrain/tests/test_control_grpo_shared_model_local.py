# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Production guard (local/torchrun): colocated shared-model GRPO on an FSDP2-sharded HFModel,
via typed ActorGroup binding. Trains a sharded model through the full
GRPO loop and checks the optimizer moved the gathered weights, then round-trips both consolidated
async and collective DCP checkpoints. Runs under torchrun on >=2 GPUs; the Ray-backend variant lives
in test_control_grpo_shared_model_ray.py.

    torchrun --standalone --nproc_per_node=2 -m pytest -q --L1 \
        cosmos_framework/posttrain/tests/test_control_grpo_shared_model_local.py
"""

from __future__ import annotations

import inspect
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")


@pytest.mark.L0
@pytest.mark.CPU
def test_runtime_group_size_configures_reward_worker_and_annotator() -> None:
    from cosmos_framework.posttrain.schema import ObsBundle, Status, Trajectory, Transition
    from cosmos_framework.posttrain.tests.helpers.hf_fsdp_grpo import build_reward_worker, grpo_loop

    group_size = 4
    worker = build_reward_worker(group_size)
    trajectories = [
        Trajectory(
            transitions=[
                Transition(
                    obs=ObsBundle(id="g", representations={}),
                    policy_info={"completion_text": "x" * (index + 1)},
                    done=True,
                )
            ],
            id=f"t{index}",
            group_id="g",
            status=Status.COMPLETED,
        )
        for index in range(group_size)
    ]

    scored = worker.score(trajectories)

    assert worker.training_group_size == group_size
    assert worker.configured_group_size() == group_size
    assert "G" not in inspect.signature(grpo_loop).parameters
    assert len(scored) == group_size
    assert all(trajectory.transitions[-1].advantage is not None for trajectory in scored)


def _save_tiny_gpt2(path: str) -> None:
    from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast

    cfg = GPT2Config(n_layer=2, n_head=2, n_embd=32, vocab_size=50257, n_positions=64)
    GPT2LMHeadModel(cfg).save_pretrained(path)
    GPT2TokenizerFast.from_pretrained("gpt2").save_pretrained(path)


def _optimizer_tensor_state(optimizer: torch.optim.Optimizer) -> list[tuple[int, str, torch.Tensor]]:
    """Gather deterministic CPU copies of every optimizer tensor for exact resume checks."""
    from torch.distributed.tensor import DTensor

    state: list[tuple[int, str, torch.Tensor]] = []
    for parameter_index, parameter_state in enumerate(optimizer.state.values()):
        for name, value in sorted(parameter_state.items()):
            if not isinstance(value, torch.Tensor):
                continue
            full = value.full_tensor() if isinstance(value, DTensor) else value  # [*optimizer_state]
            state.append((parameter_index, name, full.detach().cpu().clone()))  # [*optimizer_state]
    return state


def _optimizer_states_equal(
    first: list[tuple[int, str, torch.Tensor]],
    second: list[tuple[int, str, torch.Tensor]],
) -> bool:
    """Return whether two optimizer tensor snapshots are exactly identical."""
    first_keys = [(index, name) for index, name, _tensor in first]
    second_keys = [(index, name) for index, name, _tensor in second]
    if first_keys != second_keys:
        return False
    return all(
        torch.equal(first_item[2], second_item[2]) for first_item, second_item in zip(first, second, strict=True)
    )


@pytest.mark.L1
@pytest.mark.GPU
@pytest.mark.parametrize("checkpoint_format", ["consolidated", "dcp"])
def test_grpo_hf_fsdp_colocated_local(tmp_path: Path, checkpoint_format: str) -> None:
    if int(__import__("os").environ.get("WORLD_SIZE", "1")) < 2:
        # torchrun-mode test: needs a real multi-rank env:// world (torchrun --nproc_per_node>=2).
        # WORLD_SIZE-based (not RANK) because other tests leak single-rank env defaults.
        pytest.skip("torchrun-mode test: run under torchrun --nproc_per_node>=2 (WORLD_SIZE>=2)")
    import torch.distributed as dist

    if torch.cuda.device_count() < 2:
        pytest.skip("needs >=2 GPUs")
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    world, rank = dist.get_world_size(), dist.get_rank()
    if world < 2:
        pytest.skip("needs >=2 GPUs")
    torch.cuda.set_device(rank)

    from cosmos_framework.posttrain.tests.helpers.hf_fsdp_grpo import bind, grpo_loop, param_signature

    # torchrun gives each rank its own tmp_path -> broadcast rank 0's shared dir (single node).
    _obj = [str(tmp_path / "gpt2")] if rank == 0 else [None]
    dist.broadcast_object_list(_obj, src=0)
    model_dir = _obj[0]
    if rank == 0:
        _save_tiny_gpt2(model_dir)
    dist.barrier()

    # local backend: pure-FSDP mesh comes from the torchrun world (current_pdims() is None here).
    group_size = 4
    wgs = bind(
        model_dir,
        "local",
        group_size=group_size,
        checkpoint_format=checkpoint_format,
        save_async=True,
    )
    net = wgs["policy"].body.model.net  # the FSDP-sharded HF module (this rank's shard)

    before = param_signature(net)
    losses = grpo_loop(wgs, steps=2, P=2, backend="local")
    after = param_signature(net)

    assert len(losses) == 2 and all(x == x and abs(x) != float("inf") for x in losses)
    assert int(wgs["policy"].iteration().result()) == 2
    assert abs(after - before) > 0.0, "optimizer did not move the sharded weights"

    # Both formats use the same lifecycle: save, finalize, mutate, restore, then continue.
    _c = [str(tmp_path / "ckpt" / checkpoint_format / "iter_000002")] if rank == 0 else [None]
    dist.broadcast_object_list(_c, src=0)
    ckpt = _c[0]
    wgs["policy"].save(ckpt).result()
    wgs["policy"].finalize_checkpoints().result()
    saved_optimizer_state = _optimizer_tensor_state(wgs["policy"].body.opt)
    assert saved_optimizer_state, "checkpoint gate requires real optimizer tensor state"
    dist.barrier()
    if rank == 0:
        expected_model_path = (
            os.path.join(ckpt, "model", ".metadata") if checkpoint_format == "dcp" else os.path.join(ckpt, "model.pt")
        )
        assert os.path.exists(expected_model_path), f"{checkpoint_format} save wrote no model component"

    grpo_loop(wgs, steps=1, P=2, base_seed=8_000, backend="local")
    assert int(wgs["policy"].iteration().result()) == 3
    mutated_optimizer_state = _optimizer_tensor_state(wgs["policy"].body.opt)
    assert not _optimizer_states_equal(mutated_optimizer_state, saved_optimizer_state)
    assert wgs["policy"].load(ckpt).result() == 2
    restored = param_signature(net)
    assert restored == pytest.approx(after, rel=0.0, abs=1e-8)
    restored_optimizer_state = _optimizer_tensor_state(wgs["policy"].body.opt)
    assert _optimizer_states_equal(restored_optimizer_state, saved_optimizer_state)

    continued = grpo_loop(wgs, steps=1, P=2, base_seed=9_000, backend="local")
    assert len(continued) == 1 and int(wgs["policy"].iteration().result()) == 3
