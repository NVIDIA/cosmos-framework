# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Real CPU collectives for failure consensus and uneven finite-tail weighting."""

import copy
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from cosmos_framework.posttrain.algorithm.flow_bc import make_objective as flow_objective
from cosmos_framework.posttrain.algorithm.sft import make_objective
from cosmos_framework.posttrain.roles.env.sft import SFTDatasetEnv
from cosmos_framework.posttrain.roles.trainer.llm import LLMTrainer
from cosmos_framework.posttrain.tests.helpers.sft_objective import MemoryPacker, Policy
from cosmos_framework.posttrain.tests.test_sft_finite_epochs import Rows

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def _worker(rank, rendezvous, micro):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=40))
    try:

        class NeverForward:
            def flow_matching_loss(self, _inputs):
                raise AssertionError("no peer may enter the collective-bearing policy")

        def collate(_batch):
            if rank == 1:
                raise ValueError("bad physical media on rank one")
            return {}

        with pytest.raises(RuntimeError, match="flow_bc batch preparation.*rank 1.*bad physical media"):
            flow_objective(collate, training_prefix=1, action_dim=1)(NeverForward(), [object()])

        torch.manual_seed(31)
        policy = Policy()
        reference = copy.deepcopy(policy)
        packer = MemoryPacker()
        examples = []
        for i in range(5):
            ids = list(range(1, i + 3))
            examples.append(packer.add(ids, [-100, *ids[1:]]))
        samples = [packer.sft_process_sample(row.transitions[0].targets) for row in examples]
        full = packer.sft_collate_fn(samples, packer.sft_compute_max_len(samples))
        labels = full.pop("labels")[:, 1:]
        logits = reference.net(**full).logits[:, :-1]
        valid = labels != -100
        ref_loss = F.cross_entropy(logits[valid], labels[valid], reduction="sum") / valid.sum()
        ref_opt = torch.optim.AdamW(reference.parameters(), lr=0.01, weight_decay=0.1)
        ref_loss.backward()
        reference_grads = [p.grad.detach().clone() for p in reference.parameters()]
        ref_opt.step()

        policy.net = DistributedDataParallel(policy.net)
        opt = torch.optim.AdamW(policy.parameters(), lr=0.01, weight_decay=0.1)
        before_step = []
        opt.register_step_pre_hook(
            lambda optimizer, args, kwargs: before_step.extend(p.grad.detach().clone() for p in policy.parameters())
        )
        accum = 4 // micro
        trainer = LLMTrainer(
            policy, make_objective(packer, allow_empty_training_padding=True), opt, grad_accum=accum, max_grad_norm=0
        )
        data = SFTDatasetEnv(Rows(5), rank=rank, world_size=2, shuffle=False, training_epochs=1, samples_per_update=4)
        local_ids, losses = [], []
        for _ in range(accum):
            rows = data.get_samples(micro)
            local_ids.extend(row.id for row in rows if row.transitions[0].algo_extra["validation_weight"])
            report = trainer.train(rows)
            losses.append((report["train/loss_avg"], report["train/valid_token_count"]))
        assert trainer.iteration() == 1
        for actual, expected in zip(before_step, reference_grads):
            torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-6)
        for actual, expected in zip(policy.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-6)
        ids = [None, None]
        dist.all_gather_object(ids, local_ids)
        assert sorted(x for part in ids for x in part) == [f"finite-test-5:{i}" for i in range(5)]
        # Trainer reports are already DP-reduced; aggregate the actual token sums.
        loss = sum(value * count for value, count in losses) / sum(count for _, count in losses)
        assert loss == pytest.approx(float(ref_loss.detach()), abs=2e-6)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("micro", [1, 2])
def test_gloo_finite_tail_matches_independent_global_examples(tmp_path, micro):
    # micro=1 includes a globally all-padding final chunk. micro=2 also mixes
    # real and padded trajectories within one rank's physical microbatch.
    mp.spawn(_worker, args=(f"file://{tmp_path / 'rendezvous'}", micro), nprocs=2, join=True)
