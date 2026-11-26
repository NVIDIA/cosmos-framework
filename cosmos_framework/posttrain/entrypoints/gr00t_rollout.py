# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Native GR00T LIBERO Spatial rollout of a base or trained policy through the isolated simulator gateway."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path


def load_rollout_policy(config, checkpoint=None, *, device, output, dense=False, base=False, unmerged=False):
    """Load the untrained base or a trained checkpoint through the configured GR00T factories.

    The checkpoint metadata is verified before the base model is allocated; the constructed
    policy's checkpoint codec then verifies the model identity and every trained tensor.
    """
    from cosmos_framework.posttrain.config.model import Gr00tN1d7ModelConfig
    from cosmos_framework.posttrain.entrypoints.action_common import load_factory, validate_policy_source

    if not isinstance(config.model, Gr00tN1d7ModelConfig) or config.model.backbone_adaptation is None:
        raise ValueError("GR00T rollout requires a GR00T N1.7 model config with backbone_adaptation")
    validate_policy_source(config, checkpoint=checkpoint, base=base, dense=dense, unmerged=unmerged)
    from cosmos_framework.posttrain.checkpoint.adapter import load_adapter_weights, read_adapter_metadata
    from cosmos_framework.posttrain.checkpoint.manager import CheckpointManager
    from cosmos_framework.posttrain.model.checkpoint import AdapterCheckpointCodec
    from cosmos_framework.posttrain.model.gr00t.qlora_source import recorded_source

    if dense or unmerged:
        if not config.dense_policy_factory:
            raise ValueError("GR00T dense/unmerged rollout requires dense_policy_factory")
        options = {"unmerged": True} if unmerged else {}
        return load_factory(config.dense_policy_factory)(config, checkpoint, device=device, output=output, **options)
    expected = {
        "format": "cosmos.family-backbone.v1",
        "base": recorded_source(),
        "task_contract": {"family": "gr00t_n1d7"},
    }
    manager = CheckpointManager(checkpoint_format="dcp")
    metadata = None if base else read_adapter_metadata(checkpoint, manager=manager, expected_identity=expected)
    policy = load_factory(config.policy_factory)(config, device=device, output=output)
    codec = getattr(policy.net, "_cosmos_adapter_checkpoint_codec", None)
    if not isinstance(codec, AdapterCheckpointCodec):
        raise ValueError("GR00T rollout policy must expose the native adapter checkpoint codec")
    codec.assert_live(policy.net)
    if base:
        identity = codec.identity
        if (
            identity.get("format") != expected["format"]
            or identity.get("base") != expected["base"]
            or "training_contract" in identity
            or identity.get("task_contract", {}).get("family") != "gr00t_n1d7"
            or identity.get("adapter", {}).get("mode") != config.model.backbone_adaptation.mode
        ):
            raise ValueError("constructed GR00T base policy differs from the configured untrained source")
        codec.validate_frozen_base()
    else:
        codec.validate_metadata(metadata, purpose="inference")
        load_adapter_weights(checkpoint, codec, manager=manager, purpose="inference")
    policy.net.eval()
    return policy


def load_cases(path):
    """Read and validate the requested LIBERO Spatial cases before any model allocation."""
    from cosmos_framework.posttrain.embodied.sims.libero.protocol import LiberoCase

    selected = [LiberoCase(**row) for row in json.loads(Path(path).read_text())]
    if not selected or len(set(selected)) != len(selected) or any(case.suite != "libero_spatial" for case in selected):
        raise ValueError("GR00T rollout requires a nonempty list of distinct LIBERO Spatial cases")
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint")
    source.add_argument("--base", action="store_true", help="Evaluate the untrained base without a checkpoint")
    representation = parser.add_mutually_exclusive_group()
    representation.add_argument("--dense", action="store_true", help="Load a merged dense export")
    representation.add_argument("--unmerged", action="store_true", help="Load a dequantized unmerged LoRA export")
    parser.add_argument("--sim-python", required=True)
    parser.add_argument("--cases", required=True, help="JSON list of LiberoCase mappings")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--latency-warmup-calls", type=int, default=5)
    parser.add_argument("--output", required=True)
    args, rest = parser.parse_known_args()
    if args.latency_warmup_calls < 0:
        parser.error("--latency-warmup-calls must be nonnegative")
    if (args.dense or args.unmerged) and args.base:
        parser.error("--dense/--unmerged requires --checkpoint")
    opts = rest[1:] if rest and rest[0] == "--" else rest
    if any("=" not in item for item in opts):
        parser.error("configuration overrides require key=value")
    if os.environ.get("WORLD_SIZE") != "1":
        raise ValueError("GR00T rollout requires a single-process torchrun (WORLD_SIZE=1)")
    sim_gpu = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not sim_gpu:
        raise ValueError("GR00T rollout requires CUDA_VISIBLE_DEVICES to name the GPU shared with the simulator")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    if not Path(args.sim_python).is_file():
        raise ValueError("the configured isolated LIBERO simulator Python is unavailable")
    from cosmos_framework.posttrain.embodied.sims.libero.protocol import (
        LIBERO_RUNTIME,
        SPATIAL_H8_EXECUTION_PROFILE,
        execution_protocol,
    )
    from cosmos_framework.posttrain.entrypoints.action_common import configure_tf32, validate_config, write_json
    from cosmos_framework.posttrain.model.gr00t.rollout import NativeArtifactRollout
    from cosmos_framework.posttrain.roles.env.sim_gateway import SimGatewayBody
    from cosmos_framework.posttrain.utils.determinism import enable_determinism
    from cosmos_framework.utils.config import load_config

    cases = load_cases(args.cases)
    config = load_config(args.config, ["--", *opts])
    validate_config(config)
    protocol = execution_protocol("libero_spatial", SPATIAL_H8_EXECUTION_PROFILE)
    horizon, max_steps = protocol.execution_horizon, protocol.max_episode_steps
    if config.model.execution_horizon != horizon:
        raise ValueError(f"GR00T LIBERO Spatial rollout requires model.execution_horizon={horizon}")
    decisions = max_steps // horizon
    import torch
    import torch.distributed as dist

    enable_determinism(strict=True)
    configure_tf32(config)
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl")
    started = time.monotonic()
    gateway = None
    try:
        output.mkdir(parents=True)
        policy = load_rollout_policy(
            config,
            args.checkpoint,
            device=device,
            output=output,
            dense=args.dense,
            base=args.base,
            unmerged=args.unmerged,
        )
        policy_loading_seconds = time.monotonic() - started
        rollout = NativeArtifactRollout(policy, latency_capacity=decisions * len(cases))
        gateway = SimGatewayBody(
            sim="libero",
            sink="null",
            embodiment="libero_sim",
            sim_python=args.sim_python,
            sim_gpu=sim_gpu,
            sim_options={
                "n_lanes": 1,
                "suite": "libero_spatial",
                "execution_horizon": horizon,
                "max_steps": max_steps,
                "execution_profile": SPATIAL_H8_EXECUTION_PROFILE,
            },
        )
        probe = gateway.probe()
        result = gateway.run_evaluation(
            policy=rollout, cases=[asdict(case) for case in cases], version=0, seed=args.seed, decisions=decisions
        )
        episodes = result["episodes"]
        expected = {(case.task_id, case.init_state_id) for case in cases}
        actual = {(row["task_id"], row["init_state_id"]) for row in episodes}
        if (
            len(episodes) != len(cases)
            or actual != expected
            or any(
                type(row["success"]) is not bool
                or not horizon <= row["steps"] <= max_steps
                or row["steps"] % horizon != 0
                or (not row["success"] and row["steps"] != max_steps)
                for row in episodes
            )
        ):
            raise RuntimeError("LIBERO evaluation did not return exactly one complete episode per requested case")
        if args.unmerged:
            artifact_kind, identity = "unmerged", policy.unmerged_manifest["source_metadata"]["identity"]
        elif args.dense:
            artifact_kind, identity = "dense", policy.dense_manifest["source_metadata"]["identity"]
        else:
            artifact_kind = "base" if args.base else "checkpoint"
            identity = policy.net._cosmos_adapter_checkpoint_codec.identity
        write_json(
            output / "rollout.json",
            {
                "format": "gr00t-native-artifact-rollout-v1",
                "checkpoint": args.checkpoint,
                "artifact_kind": artifact_kind,
                "model_identity": identity,
                "cases": [asdict(case) for case in cases],
                "seed": args.seed,
                "execution_profile": SPATIAL_H8_EXECUTION_PROFILE,
                "protocol": protocol.identity(),
                "runtime": LIBERO_RUNTIME.identity(),
                "simulator_probe": probe,
                "result": result,
                "successes": sum(row["success"] for row in episodes),
                "episodes": len(episodes),
                "workflow_seconds": time.monotonic() - started,
                "policy_loading_seconds": policy_loading_seconds,
                "response_latency": rollout.latency_report(warmup_calls=args.latency_warmup_calls),
            },
        )
    finally:
        if gateway is not None:
            gateway.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
