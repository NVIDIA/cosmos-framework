# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""GPU forward diagnostics for actual Pi0.5 scalar and batched PPO replay.

Only exported compact prepared16 records are accepted, preserving real targets.
Scalar/B16 and padded-tail differences are diagnostic; the independent RLinf
qualification is provided by pi05_recompute_parity_cli. No gradient/update
numerical tolerance is invented and no model tensors are saved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from cosmos_framework.posttrain.algorithm.joint_ppo.records import STATS_KEY, TARGET_KEY
from cosmos_framework.posttrain.model.pi05.config import Pi05ModelConfig
from cosmos_framework.posttrain.model.pi05.policy import Pi05TrainPolicy
from cosmos_framework.posttrain.tests.helpers.pi05_prepared import load_prepared16, parity_recipe
from cosmos_framework.posttrain.utils.object_store import file_sha256


def tensor_difference(reference, actual):
    reference, actual = reference.detach().cpu(), actual.detach().cpu()
    if reference.shape != actual.shape or reference.dtype != actual.dtype:
        raise ValueError("comparison requires equal tensor shape and dtype")
    if not bool(torch.isfinite(reference).all() and torch.isfinite(actual).all()):
        raise ValueError("comparison encountered a nonfinite tensor")
    left, right = reference.double(), actual.double()
    difference = right - left
    reference_squared = float(left.square().sum())
    difference_squared = float(difference.square().sum())
    return {
        "dtype": str(reference.dtype),
        "numel": reference.numel(),
        "different_elements": int((reference != actual).sum()),
        "exact": bool(torch.equal(reference, actual)),
        "max_abs": float(difference.abs().max()),
        "reference_l2": math.sqrt(reference_squared),
        "difference_l2": math.sqrt(difference_squared),
        "relative_l2": math.sqrt(difference_squared / reference_squared) if reference_squared else None,
        "reference_sha256": hashlib.sha256(reference.contiguous().reshape(-1).view(torch.uint8).numpy()).hexdigest(),
        "actual_sha256": hashlib.sha256(actual.contiguous().reshape(-1).view(torch.uint8).numpy()).hexdigest(),
    }


@torch.no_grad()
def compare_forward(policy, samples, batch_sizes, policy_clip):
    steps = [sample.transitions[0] for sample in samples]
    scalar = [policy.score_ppo_decision(step) for step in steps]
    logprobs = torch.cat([score.logprobs for score in scalar])
    values = torch.cat([score.values for score in scalar])
    rows = []
    for size, count in ((size, count) for size in batch_sizes for count in (16, 13)):
        # The same real source also exercises an incomplete fixed-size tail.
        selected = steps[:count] + [steps[0]] * (size - count)
        scored = policy.score_ppo_decisions(selected)
        expected_logprobs = torch.cat([logprobs[:count], logprobs[:1].repeat(size - count)])
        expected_values = torch.cat([values[:count], values[:1].repeat(size - count)])
        differences = {
            "logprobs": tensor_difference(expected_logprobs, scored.logprobs),
            "values": tensor_difference(expected_values, scored.values),
        }
        old = torch.tensor([step.algo_extra[STATS_KEY].old_logprob for step in steps[:count]], device=logprobs.device)
        delta = float((scored.logprobs[:count] - old).abs().max())
        clipped = bool(((scored.logprobs[:count] - old).exp().sub(1).abs() > policy_clip).any())
        rows.append(
            {
                "batch_size": size,
                "real_decisions": count,
                "loss_mask": [step.algo_extra[TARGET_KEY].loss_mask for step in steps[:count]],
                "loss_weight": [step.algo_extra[TARGET_KEY].loss_weight for step in steps[:count]],
                "passed": all(row["max_abs"] <= policy.ppo_logprob_tolerance for row in differences.values()),
                "public_on_policy_passed": delta <= policy.ppo_logprob_tolerance and not clipped,
                "max_logprob_delta": delta,
                "initial_clip_fraction": float(clipped),
                **differences,
            }
        )
    old = torch.tensor([step.algo_extra[STATS_KEY].old_logprob for step in steps], device=logprobs.device)
    old_values = torch.tensor([step.algo_extra[STATS_KEY].old_value for step in steps], device=values.device)
    return {
        "passed": all(row["passed"] for row in rows),
        "public_on_policy_passed": all(row["public_on_policy_passed"] for row in rows),
        "batches": rows,
        "original_replay_vs_sampling": {
            "logprobs": tensor_difference(old, logprobs),
            "values": tensor_difference(old_values, values),
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records", type=Path, required=True, help="Exported compact prepared16.pt with its same-stem provenance JSON."
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("/reference/models/common_init"))
    parser.add_argument(
        "--tokenizer", type=Path, default=Path("/reference/cache/openpi/big_vision/paligemma_tokenizer.model")
    )
    parser.add_argument(
        "--norm-stats",
        type=Path,
        default=Path("/reference/models/common_init/physical-intelligence/libero/norm_stats.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[16, 128])
    args = parser.parse_args(argv)
    if any(size < 16 for size in args.batch_sizes):
        parser.error("batch sizes must be at least 16")
    report = {
        "passed": False,
        "runs": [],
        "pass_criterion": "both scalar/batch logprob-value threshold and unchanged-weight logprob/clip gate",
        "scope": "Forward only. The independent before/current/RLinf B16 recompute probe owns gradient and update validation.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("pi05 batch parity requires CUDA")
        config = Pi05ModelConfig(str(args.checkpoint), str(args.tokenizer), str(args.norm_stats))
        assets = {
            "checkpoint_sha256": file_sha256(args.checkpoint / "model.safetensors"),
            "tokenizer_sha256": file_sha256(args.tokenizer),
            "norm_stats_sha256": file_sha256(args.norm_stats),
        }
        samples, manifest = load_prepared16(args.records, config, assets)
        policy = Pi05TrainPolicy(config)
        recipe = parity_recipe()
        report.update(
            source_identity=policy.ppo_identity(),
            gpu=torch.cuda.get_device_name(),
            torch_version=torch.__version__,
            scoring_tolerance=policy.ppo_logprob_tolerance,
            batch_sizes=args.batch_sizes,
        )
        report["input"] = {
            "kind": "real compact prepared16",
            "sha256": manifest["records_sha256"],
            "source_capture": manifest["source_capture"],
            "selection": manifest["selection"],
        }
        result = {"case": "records", "forward": compare_forward(policy, samples, args.batch_sizes, recipe.policy_clip)}
        report["runs"].append(result)
        print(json.dumps({"stage": "forward_complete", **result}), flush=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        report["forward_passed"] = all(row["forward"]["passed"] for row in report["runs"])
        report["comparison_exact"] = all(
            difference["exact"]
            for row in report["runs"]
            for batch in row["forward"]["batches"]
            for difference in (batch["logprobs"], batch["values"])
        )
        report["passed"] = report["forward_passed"] and all(
            row["forward"]["public_on_policy_passed"] for row in report["runs"]
        )
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "forward_passed": report.get("forward_passed"),
                "output": str(args.output),
                "error": report.get("error"),
            }
        ),
        flush=True,
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
