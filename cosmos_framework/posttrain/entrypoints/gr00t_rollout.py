# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Native GR00T source/artifact evaluation through the existing isolated LIBERO simulator gateway."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path


def load_rollout_policy(config, checkpoint=None, *, device, output, dense=False, base=False, unmerged=False):
    """Load a declared native source or complete trained policy through its factory.

    Read and verify the committed artifact before allocating the immutable base.
    The constructed codec then checks every numerical identity and trained tensor;
    selecting the official head control must never dispatch the LoRA constructor.
    """
    if bool(checkpoint) == bool(base) or ((dense or unmerged) and base) or (dense and unmerged):
        raise ValueError("choose exactly one native checkpoint or explicit --base; --dense requires a checkpoint")
    from cosmos_framework.posttrain.checkpoint.adapter import load_adapter_weights, read_adapter_metadata
    from cosmos_framework.posttrain.checkpoint.manager import CheckpointManager
    from cosmos_framework.posttrain.entrypoints.action_sft import _factory
    from cosmos_framework.posttrain.model.checkpoint import AdapterCheckpointCodec
    from cosmos_framework.posttrain.model.gr00t.qlora_source import recorded_source

    matched = "cosmos_framework.posttrain.model.gr00t.action_sft.build_policy"
    official = "cosmos_framework.posttrain.model.gr00t.official_head.build_policy"
    if config.policy_factory not in {matched, official}:
        raise ValueError("GR00T rollout requires its explicit native matched or official-head policy factory")
    mode = None
    if base and config.policy_factory == matched:
        adaptation = getattr(config.model, "backbone_adaptation", None)
        mode = getattr(adaptation, "mode", None)
        if mode not in {"bf16_frozen", "nf4_frozen"}:
            raise ValueError("GR00T source rollout requires bf16_frozen or nf4_frozen; LoRA requires an artifact")
    if dense or unmerged:
        if config.policy_factory != matched or config.dense_policy_factory != (
            "cosmos_framework.posttrain.model.gr00t.dense.build_policy"
        ):
            raise ValueError("GR00T dense rollout is unavailable for the official-head control")
        options = {"unmerged": True} if unmerged else {}
        return _factory(config.dense_policy_factory)(config, checkpoint, device=device, output=output, **options)
    expected = {
        "format": "cosmos.family-backbone.v1" if config.policy_factory == matched else "cosmos.gr00t-official-head.v1",
        "base": recorded_source(),
    }
    if config.policy_factory == matched:
        expected["task_contract"] = {"family": "gr00t_n1d7"}
    manager = CheckpointManager(checkpoint_format="dcp")
    metadata = None if base else read_adapter_metadata(checkpoint, manager=manager, expected_identity=expected)
    policy = _factory(config.policy_factory)(config, device=device, output=output)
    codec = getattr(policy.net, "_cosmos_adapter_checkpoint_codec", None)
    if not isinstance(codec, AdapterCheckpointCodec):
        raise ValueError("GR00T rollout policy must expose the native complete trained-parameter checkpoint codec")
    codec.assert_live(policy.net)
    if base:
        if (
            codec.identity.get("format") != expected["format"]
            or codec.identity.get("base") != expected["base"]
            or "training_contract" in codec.identity
            or (
                config.policy_factory == matched
                and (
                    codec.identity.get("task_contract", {}).get("family") != "gr00t_n1d7"
                    or codec.identity.get("adapter", {}).get("mode") != mode
                )
            )
        ):
            raise ValueError("native GR00T source policy differs from its immutable untrained source contract")
        codec.validate_frozen_base()
    else:
        codec.validate_metadata(metadata, purpose="reload")
        load_adapter_weights(checkpoint, codec, manager=manager)
    policy.net.eval()
    return policy


def base_policy_binding(identity):
    """Bind existing native source/quantization/config identity; never fabricate a trained artifact."""
    from cosmos_framework.posttrain.embodied.sims.libero.generated import digest
    from cosmos_framework.posttrain.model.gr00t.qlora_source import recorded_source

    if (
        identity.get("format") != "cosmos.family-backbone.v1"
        or "training_contract" in identity
        or identity.get("base") != recorded_source()
        or identity.get("adapter", {}).get("mode") not in ("bf16_frozen", "nf4_frozen")
    ):
        raise ValueError("generated base control requires the pinned untrained native BF16/NF4 policy")
    return {
        "kind": "native-base",
        "source": identity["base"],
        "adapter": identity["adapter"],
        "quantization": identity["quantization"],
        "task_config": identity["task_contract"]["config"],
        "identity_sha256": digest(identity),
    }


def artifact_policy_binding(args):
    from cosmos_framework.posttrain.embodied.sims.libero.generated import file_sha

    marker = (
        "cosmos_family_unmerged_export.json"
        if args.unmerged
        else "cosmos_family_dense_export.json"
        if args.dense
        else ".posttrain-checkpoint-artifact.json"
    )
    return {"kind": "artifact", "manifest_sha256": file_sha(Path(args.checkpoint) / marker)}


def require_generated_policy(args, policy, manifest):
    binding = (
        base_policy_binding(policy.net._cosmos_adapter_checkpoint_codec.identity)
        if args.base
        else artifact_policy_binding(args)
    )
    if binding not in manifest["plan"]["policies"]:
        raise ValueError("loaded policy differs from the prospectively frozen generated-scene policy set")


def load_cases(args):
    """Validate frozen episode inputs before model allocation; public serialization stays unchanged."""
    from cosmos_framework.posttrain.embodied.sims.libero.protocol import LiberoCase

    if bool(args.cases) == bool(args.generated_scenes):
        raise ValueError("choose published --cases or --generated-scenes, exclusively")
    if args.generated_scenes:
        from cosmos_framework.posttrain.embodied.sims.libero.generated import cases, read_manifest

        manifest = read_manifest(args.generated_scenes, args.generated_scenes_sha256)
        if args.base:
            if not any(row["kind"] == "native-base" for row in manifest["plan"]["policies"]):
                raise ValueError("generated scenes do not bind a native base control")
        elif not args.checkpoint or artifact_policy_binding(args) not in manifest["plan"]["policies"]:
            raise ValueError("generated scenes were prepared for a different frozen policy artifact")
        return cases(manifest), manifest
    if args.generated_scenes_sha256:
        raise ValueError("generated manifest digest requires --generated-scenes")
    selected = [LiberoCase(**row) for row in json.loads(Path(args.cases).read_text())]
    if not selected or len(set(selected)) != len(selected) or any(case.suite != "libero_spatial" for case in selected):
        raise ValueError("GR00T artifact rollout requires distinct, declared LIBERO spatial cases")
    return selected, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint")
    source.add_argument(
        "--base", action="store_true", help="Evaluate the pinned source without an adapter/head artifact"
    )
    representation = parser.add_mutually_exclusive_group()
    representation.add_argument("--dense", action="store_true")
    representation.add_argument(
        "--unmerged", action="store_true", help="Load a self-contained dequantized-unmerged native LoRA artifact"
    )
    parser.add_argument("--sim-python", required=True)
    scenes = parser.add_mutually_exclusive_group(required=True)
    scenes.add_argument("--cases", help="Prospectively frozen JSON list of published LiberoCase mappings")
    scenes.add_argument("--generated-scenes", help="Prepared native generated-scene manifest")
    parser.add_argument("--generated-scenes-sha256", help="Required trusted digest for --generated-scenes")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--latency-warmup-calls", type=int, default=5)
    parser.add_argument("--output", required=True)
    args, rest = parser.parse_known_args()
    if args.latency_warmup_calls < 0:
        parser.error("--latency-warmup-calls must be nonnegative")
    if (args.dense or args.unmerged) and args.base:
        parser.error(
            "--dense/--unmerged requires --checkpoint; source evaluation does not export or fabricate an artifact"
        )
    opts = rest[1:] if rest and rest[0] == "--" else rest
    if any("=" not in item for item in opts):
        parser.error("configuration overrides require key=value")
    if os.environ.get("WORLD_SIZE") != "1" or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError("native GR00T rollout requires single-process torchrun with explicit allocated CUDA device")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    if not Path(args.sim_python).is_file():
        raise ValueError("the configured isolated LIBERO simulator Python is unavailable")
    from cosmos_framework.posttrain.embodied.sims.libero.protocol import (
        GR00T_QLORA_SPATIAL_EXECUTION_PROFILE,
        LIBERO_RUNTIME,
        execution_protocol,
    )
    from cosmos_framework.posttrain.entrypoints.action_sft import _json, validate_config
    from cosmos_framework.posttrain.model.gr00t.rollout import NativeArtifactRollout
    from cosmos_framework.posttrain.roles.env.sim_gateway import SimGatewayBody
    from cosmos_framework.posttrain.utils.determinism import enable_determinism
    from cosmos_framework.utils.config import load_config

    cases, generated = load_cases(args)
    config = load_config(args.config, ["--", *opts])
    validate_config(config)
    import torch
    import torch.distributed as dist

    enable_determinism(strict=True)
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
        if generated is not None:
            require_generated_policy(args, policy, generated)
        policy_loading_seconds = time.monotonic() - started
        rollout = NativeArtifactRollout(policy, latency_capacity=32 * len(cases))
        protocol = execution_protocol("libero_spatial", GR00T_QLORA_SPATIAL_EXECUTION_PROFILE)

        def new_gateway():
            return SimGatewayBody(
                sim="libero",
                sink="null",
                embodiment="libero_sim",
                sim_python=args.sim_python,
                sim_gpu=os.environ["CUDA_VISIBLE_DEVICES"],
                sim_options={
                    "n_lanes": 1,
                    "suite": "libero_spatial",
                    "execution_horizon": 8,
                    "max_steps": 256,
                    "execution_profile": GR00T_QLORA_SPATIAL_EXECUTION_PROFILE,
                    **(
                        {
                            "generated_scenes": str(Path(args.generated_scenes).resolve()),
                            "generated_scenes_sha256": args.generated_scenes_sha256,
                        }
                        if generated is not None
                        else {}
                    ),
                },
            )

        if generated is None:
            gateway = new_gateway()
            probe = gateway.probe()
            result = gateway.run_evaluation(
                policy=rollout, cases=[asdict(case) for case in cases], version=0, seed=args.seed, decisions=32
            )
        else:
            probe, result = [], {"episodes": [], "policy_version": 0}
            for case in cases:
                # Each scene is reconstructed and exactly verified in a fresh simulator process.
                gateway = new_gateway()
                probe.append(gateway.probe())
                current = gateway.run_evaluation(
                    policy=rollout, cases=[asdict(case)], version=0, seed=args.seed, decisions=32
                )
                result["episodes"].extend(current["episodes"])
                gateway.close()
                gateway = None
        episodes = result["episodes"]
        id_field = "scene_id" if generated is not None else "init_state_id"
        expected = {(case.task_id, getattr(case, id_field)) for case in cases}
        actual = {(row["task_id"], row[id_field]) for row in episodes}
        if (
            len(episodes) != len(cases)
            or actual != expected
            or any(
                type(row["success"]) is not bool
                or not 8 <= row["steps"] <= 256
                or row["steps"] % 8 != 0
                or (not row["success"] and row["steps"] != 256)
                for row in episodes
            )
        ):
            raise RuntimeError("native LIBERO evaluator did not complete the exact declared episode inventory")
        _json(
            output / "rollout.json",
            {
                "format": "gr00t-native-artifact-rollout-v1",
                "checkpoint": args.checkpoint,
                "dense": args.dense,
                "artifact_kind": "fresh-frozen-base"
                if args.base
                else "dequantized-unmerged-native-lora"
                if args.unmerged
                else "effective-base-dense"
                if args.dense
                else "same-base-trained-policy",
                "model_identity": policy.unmerged_manifest["source_metadata"]["identity"]
                if args.unmerged
                else policy.dense_manifest["source_metadata"]["identity"]
                if args.dense
                else policy.net._cosmos_adapter_checkpoint_codec.identity,
                "cases": [asdict(case) for case in cases],
                **(
                    {
                        "generated_scenes": {
                            "sha256": args.generated_scenes_sha256,
                            "format": generated["format"],
                            "reset_order": generated["reset_order"],
                            "policy_noise": generated["policy_noise"],
                            "source": generated["plan"]["source"],
                        }
                    }
                    if generated is not None
                    else {}
                ),
                "seed": args.seed,
                "execution_profile": GR00T_QLORA_SPATIAL_EXECUTION_PROFILE,
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
