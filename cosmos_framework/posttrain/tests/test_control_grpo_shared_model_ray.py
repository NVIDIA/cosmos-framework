# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Production guard (ray): colocated shared-model GRPO on an FSDP2-sharded HFModel, via the
ActorGroup/DeviceGroup style. This is the first END-TO-END exercise of
the placement-layer mesh injection (workers.ray.set_mesh -> current_pdims, Tasks 6+11): the Ray
grid builds + injects ParallelDims(world=K, dp_shard=K), each actor builds its FSDP shard of the
shared HFModel, and the GRPO loop trains across the actors + a collective DCP save/resume.

Plain pytest (Ray manages its own K GPU worker processes) on >=2 GPUs; single node.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")


def _save_tiny_gpt2(path: str) -> None:
    from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast

    cfg = GPT2Config(n_layer=2, n_head=2, n_embd=32, vocab_size=50257, n_positions=64)
    GPT2LMHeadModel(cfg).save_pretrained(path)
    GPT2TokenizerFast.from_pretrained("gpt2").save_pretrained(path)


@pytest.mark.L1
@pytest.mark.GPU
def test_grpo_hf_fsdp_colocated_ray(tmp_path: Path) -> None:
    ray = pytest.importorskip("ray")
    if torch.cuda.device_count() < 2:
        pytest.skip("needs >=2 GPUs")

    from cosmos_framework.posttrain.tests.helpers.hf_fsdp_grpo import bind, grpo_loop

    model_dir = str(tmp_path / "gpt2")  # driver-local dir; the actors are on this same node
    _save_tiny_gpt2(model_dir)

    try:
        # replicas=1, ranks=2: one pure-FSDP group of 2 GPU actors; the placement layer injects the
        # mesh (current_pdims) which build_shared_bodies reads to FSDP-shard the shared HFModel.
        group_size = 3
        wgs = bind(model_dir, "ray", ranks=2, group_size=group_size)
        losses = grpo_loop(wgs, steps=2, P=2, backend="ray")

        assert len(losses) == 2 and all(x == x and abs(x) != float("inf") for x in losses)
        assert int(wgs["policy"].iteration().result()) == 2

        # Collective DCP save across the 2 actors (shared path on this node).
        ckpt = str(tmp_path / "ckpt" / "iter_000002")
        wgs["policy"].save(ckpt).all()  # wait on EVERY cell, not just (0,0)
        wgs["policy"].finalize_checkpoints().all()
        assert os.path.exists(os.path.join(ckpt, "model", ".metadata")), "DCP save wrote no metadata"

        grpo_loop(wgs, steps=1, P=2, base_seed=8_000, backend="ray")
        assert int(wgs["policy"].iteration().result()) == 3
        restored_iterations = wgs["policy"].load(ckpt).all()
        assert set(restored_iterations.values()) == {2}

        continued = grpo_loop(wgs, steps=1, P=2, base_seed=9_000, backend="ray")
        assert len(continued) == 1 and int(wgs["policy"].iteration().result()) == 3
    finally:
        ray.shutdown()
