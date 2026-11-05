# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Two-rank FSDP2 probe using the production π0.5 policy and an immutable parity trace.

Run with torchrun --nproc-per-node=2. --reference-trace is a record produced by
pi05_parity_cli.py, preferably its unsharded posttrain phase. This checks one
deliberately weighted gradient reduction and one AdamW update; it is not a
LIBERO training or evaluation result.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _run(args: argparse.Namespace) -> dict:
    import torch
    import torch.distributed as dist
    from torch.distributed.tensor import DTensor, Shard

    from cosmos_framework.posttrain.model.parallel_dims import ParallelDims
    from cosmos_framework.posttrain.model.pi05.config import Pi05ModelConfig
    from cosmos_framework.posttrain.model.pi05.policy import Pi05TrainPolicy
    from cosmos_framework.posttrain.model.pi05.ppo import Pi05PPOTrace
    from cosmos_framework.posttrain.tests.helpers.pi05_parity import observations, read_request, sha256
    from cosmos_framework.posttrain.utils.dist import rank_identical, run_rank_converged

    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 2:
        raise ValueError("pi05_fsdp_cli requires exactly two torchrun ranks")
    device = torch.device("cuda", torch.cuda.current_device())

    def load_inputs():
        request = read_request(args.request)
        record = torch.load(args.reference_trace, map_location="cpu", weights_only=True, mmap=True)
        for key, path in (
            ("request_sha256", args.request),
            ("checkpoint_sha256", args.checkpoint / "model.safetensors"),
            ("tokenizer_sha256", args.tokenizer),
            ("norm_stats_sha256", args.norm_stats),
        ):
            if record["metadata"][key] != sha256(path):
                raise ValueError(f"FSDP reference source mismatch: {key}")
        for name in ("before", "after", "index"):
            if f"score.trace.{name}" not in record["tensors"]:
                raise ValueError(f"reference record is missing score.trace.{name}")
        return request, record

    request, reference = run_rank_converged(load_inputs, "pi05 FSDP input validation")
    dimensions = ParallelDims(world_size=world, dp_shard=world)
    dimensions.build_device_mesh("cuda")
    config = Pi05ModelConfig(str(args.checkpoint), str(args.tokenizer), str(args.norm_stats))
    policy = Pi05TrainPolicy(config, device=device, parallel_dims=dimensions)
    parameters = dict(policy.net.named_parameters())

    def check_shards() -> None:
        if {parameter.dtype for parameter in parameters.values()} != {torch.float32, torch.bfloat16}:
            raise AssertionError("FSDP changed the native mixed parameter dtypes")
        for name, parameter in parameters.items():
            if not isinstance(parameter, DTensor) or parameter.placements != (Shard(0),):
                raise AssertionError(f"{name} is not a production FSDP parameter shard")

    run_rank_converged(check_shards, "pi05 FSDP shard validation")
    frozen_before = run_rank_converged(policy.ppo_frozen_state, "pi05 frozen state before FSDP update")
    rank_identical(frozen_before, "pi05 initial frozen state")
    condition = run_rank_converged(
        lambda: policy.ppo_condition(observations(request)), "pi05 FSDP condition construction"
    )
    saved = reference["tensors"]
    trace = Pi05PPOTrace(*(saved[f"score.trace.{name}"].to(device) for name in ("before", "after", "index")))
    scores = policy.ppo_score(condition, trace)

    def check_scores() -> dict:
        result = {}
        for name, value in (("logprobs", scores.logprobs), ("values", scores.values)):
            expected = saved[f"score.{name}"].to(device)
            torch.testing.assert_close(value, expected, rtol=0, atol=policy.ppo_logprob_tolerance)
            result[name] = float((value.detach() - expected).abs().max())
        return result

    score_errors = run_rank_converged(check_scores, "pi05 FSDP score parity")
    actor_names, value_names = policy.ppo_parameters.actor, policy.ppo_parameters.value
    optimizer = torch.optim.AdamW(
        [{"params": [parameters[name] for name in names]} for names in (actor_names, value_names)],
        lr=1e-4,
        weight_decay=0,
        foreach=False,
    )
    optimizer.zero_grad(set_to_none=True)
    if args.disable_gradient_sync:
        # Explicit negative control: the normal gradient checks must reject this.
        policy.net.set_requires_gradient_sync(False, recurse=True)
    # Ranks contribute coefficients 1 and 2. A missing average or missing rank
    # cannot match the unsharded gradient scaled by their mean (1.5).
    objective = (-scores.logprobs.mean() + scores.values.square().mean() / 2) * (rank + 1)
    objective.backward()

    def check_gradients() -> list[dict]:
        rows = []
        for name in (*actor_names, *value_names):
            gradient = parameters[name].grad
            if not isinstance(gradient, DTensor):
                raise AssertionError(f"missing distributed gradient for {name}")
            local = gradient.to_local().detach()
            if not torch.isfinite(local).all():
                raise AssertionError(f"non-finite gradient for {name}")
            full_name = (
                "paligemma_with_expert.gemma_expert.model." + name[len("expert.") :]
                if name.startswith("expert.")
                else name
            )
            expected_full = saved[f"gradient.{full_name}"]
            width = (expected_full.shape[0] + world - 1) // world
            expected = expected_full[rank * width : (rank + 1) * width].to(device) * ((world + 1) / 2)
            if expected.shape != local.shape or expected.dtype != local.dtype:
                raise AssertionError(f"gradient shard shape/dtype differs for {name}")
            difference = float((local.double() - expected.double()).abs().max()) if local.numel() else 0.0
            scale = float(expected.abs().max()) if expected.numel() else 0.0
            # This is a reduction rounding bound, not a tolerance for different
            # forward kernels. BF16 gradients are reduced in their native dtype.
            bound = 4 * torch.finfo(local.dtype).eps * scale
            if difference > bound:
                raise AssertionError(f"gradient synchronization differs at {name}: max_abs={difference}, bound={bound}")
            rows.append({"name": name, "dtype": str(local.dtype), "max_abs": difference, "bound": bound})
        return rows

    gradients = run_rank_converged(check_gradients, "pi05 FSDP synchronized gradient parity")

    def capture_probes() -> dict:
        probes = {}
        for group, names in (("actor", actor_names), ("value", value_names)):
            for name in names:
                parameter = parameters[name]
                local = parameter.to_local()
                if (
                    parameter.dtype == torch.float32
                    and local.numel()
                    and torch.count_nonzero(parameter.grad.to_local())
                ):
                    probes[group] = (name, local.detach().clone())
                    break
            if group not in probes:
                raise AssertionError(f"{group} has no nonzero FP32 gradient to verify its update")
        return probes

    probes = run_rank_converged(capture_probes, "pi05 optimizer update probes")
    run_rank_converged(optimizer.step, "pi05 AdamW step")

    def check_updated() -> None:
        for group, (name, before) in probes.items():
            current = parameters[name].to_local().detach()
            if torch.equal(current, before) or not torch.isfinite(current).all():
                raise AssertionError(f"AdamW did not produce a finite {group} update at {name}")

    run_rank_converged(check_updated, "pi05 optimizer update validation")
    with torch.no_grad():
        after = policy.ppo_score(condition, trace)
    values = torch.cat((after.logprobs, after.values))
    peers = [torch.empty_like(values) for _ in range(world)]
    dist.all_gather(peers, values)

    def check_rank_scores() -> None:
        for peer in peers:
            torch.testing.assert_close(values, peer, rtol=0, atol=policy.ppo_logprob_tolerance)

    run_rank_converged(check_rank_scores, "pi05 updated score agreement")
    frozen_after = run_rank_converged(policy.ppo_frozen_state, "pi05 frozen state after FSDP update")

    def check_frozen() -> None:
        if frozen_after != frozen_before:
            raise AssertionError("FSDP/AdamW changed the frozen VLM")

    run_rank_converged(check_frozen, "pi05 frozen VLM validation")
    return {
        "passed": True,
        "rank": rank,
        "world_size": world,
        "gpu": torch.cuda.get_device_name(),
        "request_sha256": reference["metadata"]["request_sha256"],
        "reference_trace": str(args.reference_trace),
        "reference_phase": reference["metadata"]["phase"],
        "score_max_abs": score_errors,
        "actor_gradients": len(actor_names),
        "value_gradients": len(value_names),
        "gradients": gradients,
        "optimizer_updates": 1,
        "test_step_learning_rate": 1e-4,
        "frozen_before": frozen_before,
        "frozen_after": frozen_after,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("request", "checkpoint", "tokenizer", "norm-stats", "reference-trace", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument(
        "--disable-gradient-sync",
        action="store_true",
        help="Negative control: disable FSDP gradient synchronization; the probe must fail.",
    )
    args = parser.parse_args(argv)
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    import torch
    import torch.distributed as dist

    from cosmos_framework.posttrain.utils.dist import (
        destroy_process_group_if_initialized,
        init_local_process_group,
        run_rank_converged,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("pi05_fsdp_cli requires real CUDA GPUs")
    try:
        init_local_process_group()
        report = _run(args)
        reports = [None] * dist.get_world_size()
        dist.all_gather_object(reports, report)

        def write_report() -> None:
            if dist.get_rank() == 0:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps({"passed": True, "ranks": reports}, indent=2) + "\n")
                print(
                    json.dumps(
                        {"passed": True, "world_size": len(reports), "optimizer_updates": 1, "output": str(args.output)}
                    )
                )

        run_rank_converged(write_report, "pi05 FSDP report writing")
        return 0
    finally:
        destroy_process_group_if_initialized()


if __name__ == "__main__":
    raise SystemExit(main())
