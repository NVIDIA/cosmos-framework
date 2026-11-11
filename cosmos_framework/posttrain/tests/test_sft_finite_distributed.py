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


class _FlowNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(2, 2)

    def forward(self, inputs):
        from cosmos_framework.posttrain.model.protocol import FlowMatchingLossOutput

        prediction = self.projection(inputs["noise"] + inputs["time"] * inputs["actions"])
        error = (prediction - (inputs["noise"] - inputs["actions"])) ** 2
        mask = inputs["mask"]
        return FlowMatchingLossOutput((error * mask).sum() / mask.sum().clamp_min(1), mask, prediction)


class _FlowPolicy:
    def __init__(self, net):
        self.net = net

    def flow_matching_loss(self, inputs):
        return self.net(inputs)


def _flow_inputs(rows):
    # Fixed sample-owned time/noise: compares the same mathematical objective
    # across microbatch and rank partitions, independent of RNG consumption order.
    values, masks, noise, time = [], [], [], []
    for row in rows:
        index = int(row.id.rsplit(":", 1)[1])
        values.append(torch.arange(10, dtype=torch.float32).reshape(5, 2) * (index + 1) / 10)
        mask = torch.zeros(5, 2)
        mask[: index + 1] = row.transitions[0].algo_extra.get("validation_weight", 1)
        masks.append(mask)
        noise.append(torch.randn(5, 2, generator=torch.Generator().manual_seed(42 + index)))
        time.append(0.1 + 0.1 * index)
    return {
        "actions": torch.stack(values),
        "mask": torch.stack(masks),
        "noise": torch.stack(noise),
        "time": torch.tensor(time)[:, None, None],
    }


def _flow_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=40))
    try:
        torch.manual_seed(37)
        net = _FlowNet()
        reference = copy.deepcopy(net)
        data = SFTDatasetEnv(Rows(5), rank=rank, world_size=2, shuffle=False, training_epochs=1, samples_per_update=4)
        full_data = SFTDatasetEnv(Rows(5), shuffle=False, training_epochs=1, samples_per_update=5)
        inputs = _flow_inputs(full_data.get_samples(5))
        # Independent reference: no Objective, trainer, rank denominator or microbatch loop.
        prediction = F.linear(
            inputs["noise"] + inputs["time"] * inputs["actions"], reference.projection.weight, reference.projection.bias
        )
        loss = (((prediction - (inputs["noise"] - inputs["actions"])) ** 2) * inputs["mask"]).sum() / inputs[
            "mask"
        ].sum()
        opt_ref = torch.optim.AdamW(reference.parameters(), lr=0.01, weight_decay=0.1)
        loss.backward()
        grads = [p.grad.detach().clone() for p in reference.parameters()]
        opt_ref.step()
        policy = _FlowPolicy(DistributedDataParallel(net))
        opt = torch.optim.AdamW(policy.net.parameters(), lr=0.01, weight_decay=0.1)
        actual_grads = []
        opt.register_step_pre_hook(
            lambda optimizer, args, kwargs: actual_grads.extend(
                p.grad.detach().clone() for p in policy.net.parameters()
            )
        )
        trainer = LLMTrainer(
            policy,
            flow_objective(_flow_inputs, training_prefix=5, action_dim=2, allow_empty_training_padding=True),
            opt,
            grad_accum=4,
            max_grad_norm=0,
        )
        for _ in range(4):
            trainer.train(data.get_samples(1))
        assert trainer.iteration() == 1
        for a, b in zip(actual_grads, grads):
            torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-6)
        for a, b in zip(policy.net.parameters(), reference.parameters()):
            torch.testing.assert_close(a, b, atol=2e-7, rtol=2e-6)
        for a, b in zip(opt.state.values(), opt_ref.state.values()):
            for key in a:
                torch.testing.assert_close(a[key], b[key], atol=2e-7, rtol=2e-6)
    finally:
        dist.destroy_process_group()


def test_gloo_accumulated_flow_tail_matches_controlled_global_action_reference(tmp_path):
    mp.spawn(_flow_worker, args=(f"file://{tmp_path / 'flow-rendezvous'}",), nprocs=2, join=True)
