# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Exercise the real pinned Omni engine, mixed-dtype publication and reconstruction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    import numpy as np
    import torch

    from cosmos_framework.posttrain.comm.transfer.movers import CopyMover
    from cosmos_framework.posttrain.comm.weight_transfer import WeightSync, push_stream_to_peer
    from cosmos_framework.posttrain.config.rl import JointPPOConfig
    from cosmos_framework.posttrain.control.loop.weight_sync import confirm_weight_version
    from cosmos_framework.posttrain.model.pi05.config import Pi05ModelConfig
    from cosmos_framework.posttrain.model.pi05.policy import Pi05TrainPolicy
    from cosmos_framework.posttrain.model.pi05.serving.body import Pi05PolicyBody
    from cosmos_framework.posttrain.roles.trainer.joint_ppo import JointPPOTrainer
    from cosmos_framework.posttrain.tests.helpers.pi05_parity import observations, read_request

    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("request", "checkpoint", "tokenizer", "norm-stats", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = Pi05ModelConfig(str(args.checkpoint), str(args.tokenizer), str(args.norm_stats))
    request = read_request(args.request)
    obs = observations(request)
    seed = int(request["seed"])
    policy = Pi05TrainPolicy(config)
    producer = JointPPOTrainer(policy, JointPPOConfig())

    # The first hop stays inside this test process. CUDA cannot import its own
    # exported IPC allocation; deliver a real CPU snapshot to the CUDA receiver.
    class DeviceCopyMover(CopyMover):
        def fetch(self, handle):
            return {name: value.to(device="cuda") for name, value in super().fetch(handle).items()}

    backend = WeightSync(CopyMover())
    body = Pi05PolicyBody(config, output=str(args.output), verify_weight_values=True)
    try:
        body.bind_weight_backend(WeightSync(DeviceCopyMover()))
        push_stream_to_peer(producer, body, backend)
        confirm_weight_version(body, producer, producer.publication_version())
        original_pid = body.worker_device["pid"]
        actions = body.infer_batch(obs, seed=seed, weight_version=0)
        batch_actions = body.infer_batch(obs + obs, seed=seed, weight_version=0)
        if len(batch_actions) != len(obs) * 2:
            raise AssertionError("Omni failed to preserve the observation batch")
        sources = [
            {
                "sample_id": f"probe-{i}",
                "episode_id": f"episode-{i}",
                "fragment_id": "window-0",
                "step_idx": 0,
                "session": "omni-probe",
            }
            for i in range(len(obs))
        ]
        decisions = body.sample_decisions(obs, sources=sources, seed=seed, weight_version=0)
        deltas = []
        for decision in decisions:
            step = decision.transitions[0]
            policy.validate_ppo_record(step)
            with torch.no_grad():
                current = policy.score_ppo_decision(step)
            from cosmos_framework.posttrain.algorithm.joint_ppo.records import STATS_KEY

            delta = abs(float(current.logprobs[0]) - step.algo_extra[STATS_KEY].old_logprob)
            if delta > policy.ppo_logprob_tolerance:
                raise AssertionError(f"Omni/learner log probability mismatch: {delta}")
            deltas.append(delta)
        # A controlled parameter perturbation isolates publication from the PPO update test.
        with torch.no_grad():
            policy.net.action_out_proj.bias.add_(0.01)
        producer.set_publication_version(1)
        push_stream_to_peer(producer, body, backend)
        confirm_weight_version(body, producer, producer.publication_version())
        changed = body.infer_batch(obs, seed=seed, weight_version=1)
        if body.worker_device["pid"] != original_pid:
            raise AssertionError("ordinary weight publication rebuilt the engine")
        if all(np.array_equal(a["action"], b["action"]) for a, b in zip(actions, changed, strict=True)):
            raise AssertionError("published action-head change did not affect sampling")
        body.rebuild_engine()
        try:
            body.infer_batch(obs, seed=seed, weight_version=1)
        except RuntimeError:
            pass
        else:
            raise AssertionError("reconstructed engine served before full resynchronization")
        push_stream_to_peer(producer, body, backend)
        confirm_weight_version(body, producer, producer.publication_version())
        rebuilt = body.infer_batch(obs, seed=seed, weight_version=1)
        for before, after in zip(changed, rebuilt, strict=True):
            np.testing.assert_array_equal(before["action"], after["action"])
        report = {
            "status": "PASS",
            "batch_sizes": [len(obs), 2 * len(obs)],
            "max_logprob_delta": max(deltas),
            "initial_worker_pid": original_pid,
            "rebuilt_worker_pid": body.worker_device["pid"],
            "publication_version": body.weight_version(),
            "identity": body.ppo_identity(),
        }
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
    finally:
        body.close()
        backend.mover.close()


if __name__ == "__main__":
    main()
