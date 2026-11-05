# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Bounded DDP continuation loop with exact same-topology checkpoint resume."""

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from cosmos_framework.configs.tokenizer.lidar_v1p2 import LidarV1P2Recipe
from cosmos_framework.data.tokenizer.lidar import LidarTarDataset
from cosmos_framework.model.generator.tokenizers.lidar.lidar_tokenizer_v1 import infer_lidar_compression
from cosmos_framework.model.tokenizer.lidar.model import LidarTokenizerModel


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def save_training_state(
    model: LidarTokenizerModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    identity: dict[str, Any],
    path: Path,
) -> None:
    """All ranks participate; only rank zero writes the complete checkpoint."""
    rng: dict[str, Any] = {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state()}  # [N_cpu], [N_cuda]
    rank_states: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(rank_states, rng)
    if dist.get_rank() == 0:
        atomic_save(
            {
                "model": model.state_dict(),
                "parameter_names": [name for name, parameter in model.named_parameters() if parameter.requires_grad],
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": step,
                "identity": identity,
                "rng_by_rank": rank_states,
            },
            path,
        )
    dist.barrier()


def restore_training_state(
    payload: dict[str, Any],
    model: LidarTokenizerModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    identity: dict[str, Any],
) -> int:
    if payload["identity"] != identity:
        raise ValueError("Resume requires identical recipe, input contents, output directory and world size")
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    return int(payload["step"])


def train(
    recipe: LidarV1P2Recipe,
    samples: list[Path],
    output: Path,
    *,
    heldout: list[Path],
    warm_start: Path | None,
    resume: Path | None,
    stop_after: int,
    workers: int,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("LiDAR V1.2 training requires a CUDA allocation; use --dryrun on CPU")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    try:
        if warm_start is None and resume is None:
            raise ValueError("A pinned warm start or full-state resume is required")
        world_size, rank = dist.get_world_size(), dist.get_rank()
        torch.manual_seed(recipe.seed)
        torch.cuda.manual_seed(recipe.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        payload = torch.load(resume, map_location="cpu", weights_only=False) if resume else None
        if payload is None and (output.exists() and any(output.iterdir())):
            raise ValueError("A new run needs an empty output directory")
        output.mkdir(parents=True, exist_ok=True)
        identity = {
            "recipe": recipe.manifest(),
            "world_size": world_size,
            "output": str(output.resolve()),
            "samples": [{"path": str(path), "sha256": sha256(path)} for path in samples],
            "heldout": [{"path": str(path), "sha256": sha256(path)} for path in heldout],
            "initialization_sha256": payload["identity"]["initialization_sha256"] if payload else sha256(warm_start),
        }
        # JSON normalization makes tuples and saved manifest values comparable.
        identity = json.loads(json.dumps(identity))
        model = LidarTokenizerModel(recipe.model_config(str(warm_start) if warm_start else None)).cuda()
        optimizer, scheduler = model.init_optimizer_scheduler(recipe.optimizer_config(), recipe.scheduler_config())
        step = restore_training_state(payload, model, optimizer, scheduler, identity) if payload else 0
        if not step < stop_after <= recipe.max_steps:
            raise ValueError(f"Require saved step ({step}) < stop-after <= {recipe.max_steps}")
        ddp = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=True, static_graph=True)
        dataset = LidarTarDataset(samples, recipe.num_source_frames)
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=recipe.seed, drop_last=True
        )
        # A private generator prevents worker/iterator initialization from consuming the model RNG.
        loader = DataLoader(
            dataset,
            batch_size=recipe.batch_size,
            sampler=sampler,
            drop_last=True,
            num_workers=workers,
            pin_memory=True,
            generator=torch.Generator().manual_seed(recipe.seed),
        )
        if len(loader) == 0:
            raise ValueError("Need at least batch_size * world_size training clips")
        if rank == 0:
            (output / "resolved.json").write_text(json.dumps(identity, indent=2) + "\n")
        epoch, offset = divmod(step, len(loader))
        sampler.set_epoch(epoch)
        iterator = iter(loader)
        for _ in range(offset):
            next(iterator)
        # Rewind RNG after DDP construction and skipped deterministic input batches.
        if payload:
            rng = payload["rng_by_rank"][rank]
            torch.set_rng_state(rng["cpu"])
            torch.cuda.set_rng_state(rng["cuda"])
        while step < stop_after:
            try:
                batch = next(iterator)
            except StopIteration:
                epoch += 1
                sampler.set_epoch(epoch)
                iterator = iter(loader)
                batch = next(iterator)
            optimizer.zero_grad(set_to_none=True)
            metrics, loss = ddp(batch, step)  # loss: []; metric tensors: batch-dependent
            finite = torch.isfinite(loss).to(dtype=torch.int32)  # []
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not bool(finite.item()):
                raise FloatingPointError(f"Non-finite loss at step {step}; optimizer was not updated")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), recipe.max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            if model.ema is not None:
                model.ema.update_average(model, iteration=step)
            step += 1
            if rank == 0 and (step == 1 or step % 10 == 0):
                print(
                    json.dumps(
                        {
                            "step": step,
                            "loss": loss.item(),
                            "lr": scheduler.get_last_lr()[0],
                            "range_mae_meters": metrics["loss_dict"]["train-video_rec/range_mae_meters"].item(),
                        }
                    ),
                    flush=True,
                )
            if step % recipe.save_every == 0 or step == stop_after:
                save_training_state(model, optimizer, scheduler, step, identity, output / f"iter_{step:09d}.pt")
    finally:
        dist.destroy_process_group()


def export_checkpoint(checkpoint: Path, output: Path, weights: str) -> None:
    """Create a serving payload without optimizer state or machine-local input paths."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload["model"]
    if not bool(state["_latent_stats_initialized"].item()):
        raise ValueError("Cannot export a tokenizer without accumulated latent statistics")
    exported = {name: value for name, value in state.items() if not name.startswith("ema.")}
    if weights == "ema":
        for name in payload["parameter_names"]:
            if name.startswith("network."):
                ema_name = "ema." + name.replace(".", "-")
                if ema_name in state:
                    exported[name] = state[ema_name]
                else:
                    raise ValueError(f"Missing EMA tensor: {ema_name}")
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    recipe = payload["identity"]["recipe"]
    atomic_save({"model": exported}, output / "tokenizer.pt")
    stats = {
        "mean": exported["latent_mean"].reshape(-1),  # [C]
        "std": exported["latent_std"].reshape(-1),  # [C]
        "min_range": float(exported["min_range"]),
        "max_range": float(exported["max_range"]),
    }
    atomic_save(stats, output / "latent_stats.pt")
    spatial, temporal = infer_lidar_compression(recipe["network"])
    (output / "config.json").write_text(
        json.dumps(
            {
                "network_config": recipe["network"],
                "range_projection": recipe["range_projection"],
                "spatial_compression": list(spatial),
                "temporal_compression": temporal,
                "weights": weights,
                "step": payload["step"],
                "source_sha256": sha256(checkpoint),
                "checkpoint_sha256": sha256(output / "tokenizer.pt"),
                "precision": "float32",
            },
            indent=2,
        )
        + "\n"
    )
