# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Start one independent GPU Reasoner feature-service replica.

Start one process per service GPU.  The processes do not create a PyTorch
distributed process group and must remain outside the Generator FSDP world.
Use an external TCP/gRPC load balancer when multiple replicas serve training
ranks.
"""

from __future__ import annotations

import argparse
import json
import signal
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import torch

from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml
from cosmos_framework.model.generator.reasoner_features import ReasonerFeatureIdentity
from cosmos_framework.model.generator.reasoner_remote_server import (
    ReasonerFeatureService,
    create_reasoner_grpc_server,
)
from cosmos_framework.model.generator.reasoner_runtime import ReasonerFeatureRuntime, ReasonerRuntimeSpec

# The generic SFT loader resolves the complete training config, including
# dataloaders, the video VAE and the trainer's checkpoint input.  None of those
# objects are used by the Reasoner-only runtime, but shipped recipes commonly
# populate them with ``${oc.env:...}`` expressions.  Replace the unused
# subtrees before OmegaConf resolves the config so a service replica does not
# require training-only environment variables such as DATASET_PATH,
# WAN_VAE_PATH or BASE_CHECKPOINT_PATH.
_UNUSED_REASONER_SERVICE_PATH = "/unused/by-reasoner-service"
_REASONER_SERVICE_CONFIG_OVERRIDES = (
    "dataloader_train=null",
    "dataloader_val=null",
    f"checkpoint.load_path={_UNUSED_REASONER_SERVICE_PATH}",
    f"model.config.tokenizer.vae_path={_UNUSED_REASONER_SERVICE_PATH}",
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sft-toml", required=True, help="Nano SFT recipe used to construct the Reasoner")
    parser.add_argument("--checkpoint", required=True, help="Local DCP model component or iteration directory")
    parser.add_argument("--checkpoint-source", choices=("regular", "ema"), default="regular")
    parser.add_argument("--reasoner-fingerprint", required=True)
    parser.add_argument("--tokenizer-fingerprint", required=True)
    parser.add_argument("--framing-fingerprint", required=True)
    parser.add_argument("--device", default="cuda", help="One visible service device, for example cuda or cuda:0")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address (default: loopback; use a non-loopback address only on a trusted private network)",
    )
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--max-requests", type=int, default=64)
    parser.add_argument("--max-batch-tokens", type=int, default=4_096)
    parser.add_argument("--max-queued-tokens", type=int, default=8_192)
    parser.add_argument("--max-chunk-bytes", type=int, default=2 * 1024**2)
    parser.add_argument("--max-rpc-workers", type=int, default=16)
    parser.add_argument("--shutdown-grace-seconds", type=float, default=30.0)
    parser.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="Hydra overrides applied after TOML; prefix the list with --",
    )
    args = parser.parse_args(argv)
    args.overrides = [item for item in args.overrides if item != "--"]
    if not 1 <= args.port <= 65_535:
        parser.error("--port must be in [1, 65535]")
    for name in ("max_requests", "max_batch_tokens", "max_queued_tokens", "max_chunk_bytes", "max_rpc_workers"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_queued_tokens < args.max_batch_tokens:
        parser.error("--max-queued-tokens must be at least --max-batch-tokens")
    if args.max_chunk_bytes > 4 * 1024**2:
        parser.error("--max-chunk-bytes must not exceed 4 MiB")
    if args.shutdown_grace_seconds < 0:
        parser.error("--shutdown-grace-seconds must be non-negative")
    return args


def _load_reasoner_service_config(args: argparse.Namespace) -> object:
    """Load only service-relevant settings from a complete SFT recipe.

    Internal pruning overrides intentionally follow user overrides.  The
    corresponding training subtrees cannot affect Reasoner construction, and
    keeping the pruning last prevents an accidental CLI override from
    reintroducing an unrelated environment interpolation.
    """

    return load_experiment_from_toml(
        Path(args.sft_toml),
        extra_overrides=[*args.overrides, *_REASONER_SERVICE_CONFIG_OVERRIDES],
    )


def _run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Reasoner feature serving requires CUDA")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("Reasoner feature serving requires a CUDA --device")
    config = _load_reasoner_service_config(args)
    identity = ReasonerFeatureIdentity(
        reasoner=args.reasoner_fingerprint,
        tokenizer=args.tokenizer_fingerprint,
        framing=args.framing_fingerprint,
    )
    if device.index is not None:
        torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    runtime = ReasonerFeatureRuntime.load(
        config,
        ReasonerRuntimeSpec(
            checkpoint=args.checkpoint,
            checkpoint_source=args.checkpoint_source,
            device=device,
            dtype=torch.bfloat16,
            identity=identity,
            max_requests=args.max_requests,
            max_total_tokens=args.max_batch_tokens,
        ),
    )
    service = ReasonerFeatureService(
        runtime,
        max_queued_tokens=args.max_queued_tokens,
        max_chunk_bytes=args.max_chunk_bytes,
    )
    server, bound_port = create_reasoner_grpc_server(
        service,
        address=f"{args.host}:{args.port}",
        max_rpc_workers=args.max_rpc_workers,
    )
    server.start()
    startup = {
        "status": "ready",
        "address": f"{args.host}:{bound_port}",
        "service_instance_id": service.service_instance_id,
        "identity": asdict(identity),
        "signature": {
            "num_layers": runtime.signature.num_layers,
            "num_kv_heads": runtime.signature.num_kv_heads,
            "head_dim": runtime.signature.head_dim,
            "dtype": str(runtime.signature.dtype),
        },
        "limits": {
            "max_requests": runtime.max_requests,
            "max_batch_tokens": runtime.max_total_tokens,
            "max_queued_tokens": args.max_queued_tokens,
            "max_chunk_bytes": args.max_chunk_bytes,
        },
        "load_stats": None if runtime.load_stats is None else asdict(runtime.load_stats),
    }
    print(json.dumps(startup, indent=2, sort_keys=True), flush=True)
    stop_requested = threading.Event()

    def request_stop(_signum: int, _frame: object) -> None:
        stop_requested.set()

    previous_sigterm = signal.signal(signal.SIGTERM, request_stop)
    try:
        while not stop_requested.wait(timeout=1.0):
            if not server.wait_for_termination(timeout=0):
                break
    except KeyboardInterrupt:
        pass
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        server.stop(args.shutdown_grace_seconds).wait()


def main(argv: Sequence[str] | None = None) -> int:
    _run(_parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
