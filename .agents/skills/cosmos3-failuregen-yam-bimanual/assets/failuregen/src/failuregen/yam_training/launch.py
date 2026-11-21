# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Prepare original YAM recordings and launch one-node Cosmos training."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from .contract import file_sha256, load_prepared, training_contract


def run(command, *, env=None):
    print("Running:", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), check=True, env=env)


def training_overrides(run_name: str, *, smoke: bool) -> list[str]:
    # Remove this diagnostic AFTER Hydra merges the basic callback defaults and
    # experiment. Packed FD/ID batches can have unequal sigma/loss lengths.
    # Cosmos requires '=' in extra overrides. An explicit null gives Hydra an
    # unconditional deletion; an empty value would compare against an empty string.
    overrides = ["--", f"job.name={run_name}", "~trainer.callbacks.sigma_loss_analysis=null"]
    if smoke:
        overrides += ["checkpoint.save_iter=5"]
    return overrides


def validate_resolved_config(path: Path):
    """Reject unsupported callbacks and validation before distributed training."""
    config = yaml.safe_load(path.read_text())
    callbacks = config["trainer"]["callbacks"]
    if not isinstance(callbacks, dict):
        raise ValueError(f"Expected a callback mapping in {path}")
    for name, callback in callbacks.items():
        target = callback.get("_target_", "") if isinstance(callback, dict) else ""
        if name == "sigma_loss_analysis" or str(target).endswith(".SigmaLossAnalysis"):
            raise ValueError(f"SigmaLossAnalysis must be removed from the resolved YAM config: {path}")
    trainer = config["trainer"]
    if trainer.get("run_validation") is not False or trainer.get("run_validation_on_start") is not False:
        raise ValueError(
            "OmniMoTModel.validation_step is unimplemented; disable trainer validation and use held-out replay"
        )
    if config.get("dataloader_val") is not None:
        raise ValueError("YAM training must leave dataloader_val unset; use held-out checkpoint replay")


def write_contract(path: Path, value: dict):
    """Refuse incompatible resumes; preserve the original contract on mismatch."""
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Run inputs changed: {path}. Choose a new WORK_DIR/RUN_NAME.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def validate_source(prepared: Path, dataset: Path, splits: Path, *, action_space="joint", eef_root=None) -> dict:
    manifest, _ = load_prepared(prepared)
    if manifest["contract"] != training_contract(action_space):
        raise ValueError("Prepared action space differs from the requested run; use a new WORK_DIR")
    if action_space == "eef":
        eef_root = eef_root or dataset / ".labeling/cosmos_yam_report_5fps"
        if file_sha256(eef_root / "manifest.json") != manifest["eef_export_manifest_sha256"]:
            raise ValueError("EEF export manifest changed after preparation")
    if manifest["source_info_sha256"] != file_sha256(dataset / "meta/info.json"):
        raise ValueError("Dataset metadata changed after preparation")
    if manifest["annotations_sha256"] != file_sha256(dataset / manifest["annotations"]):
        raise ValueError("Annotations changed after preparation")
    if manifest["splits"] != json.loads(splits.read_text()):
        raise ValueError("Prepared episode splits differ from the selected split file")
    info = json.loads((dataset / "meta/info.json").read_text())
    for episode in manifest["episodes"]:
        if action_space == "eef" and file_sha256(eef_root / episode["eef_source"]) != episode["eef_source_sha256"]:
            raise ValueError(f"Source EEF actions changed: episode {episode['episode']}")
        index = episode["episode"]
        source = dataset / info["data_path"].format(episode_chunk=index // info["chunks_size"], episode_index=index)
        if file_sha256(source) != episode["source_sha256"]:
            raise ValueError(f"Source joint states changed: {source}")
        if file_sha256(prepared / episode["states"]) != episode["states_sha256"]:
            raise ValueError(f"Prepared states changed: episode {index}")
        for camera, relative in episode["videos"].items():
            if (dataset / relative).stat().st_size != episode["video_sizes"][camera]:
                raise ValueError(f"Source video changed: {relative}")
    return manifest


def build_plan(
    prepared: Path,
    checkpoint: Path,
    *,
    samples_per_rank: int,
    epochs: float,
    smoke: bool,
    world_size: int = 4,
    action_space: str = "joint",
) -> dict:
    if world_size != 4 or samples_per_rank not in (1, 2, 4) or not (0 < epochs <= 100):
        raise ValueError("Use four GPUs, 1/2/4 clips per rank, and a positive epoch budget <= 100")
    manifest, _ = load_prepared(prepared)
    contract = training_contract(action_space)
    if manifest["contract"] != contract:
        raise ValueError("Prepared action representation differs from the requested run")
    domains = 32 if action_space == "eef" else 33
    migration_path = checkpoint / "yam_checkpoint_migration.json"
    migration = json.loads(migration_path.read_text())
    if (
        migration.get("contract") != contract
        or migration.get("new_domain_count") != domains
        or not migration.get("existing_rows_preserved")
        or not (checkpoint / "model/.metadata").is_file()
    ):
        raise ValueError(f"Initialization is not a complete {domains}-domain {action_space} YAM checkpoint")
    if action_space == "eef" and migration.get("initialization") != "reuse_existing_domain":
        raise ValueError("EEF initialization must preserve the existing domain-16 projector")
    config = json.loads((checkpoint / "model/config.json").read_text())
    if config["model"]["config"]["num_embodiment_domains"] != domains:
        raise ValueError(f"Initialization config must have {domains} domains")
    batch = world_size * samples_per_rank
    count = manifest["window_counts"]["train"]
    if count <= 0 or any(manifest["window_counts"][key] <= 0 for key in ("val", "test")):
        raise ValueError("All splits must contain eligible windows")
    return {
        "contract": contract,
        "prepared_manifest_sha256": file_sha256(prepared / "manifest.json"),
        "stats_sha256": manifest["stats_sha256"],
        "migration_sha256": file_sha256(migration_path),
        "base_checkpoint": str(checkpoint.resolve()),
        "world_size": world_size,
        "samples_per_rank": samples_per_rank,
        "global_batch": batch,
        "target_passes": epochs,
        "train_windows": count,
        "max_iter": 10 if smoke else math.ceil(epochs * count / batch),
        "smoke": smoke,
    }


def source_fingerprint(framework: Path, recipe: Path) -> dict:
    """Fingerprint executable source, including local edits, without storing credentials."""
    files = {"recipe": recipe, **{f"failuregen/{p.name}": p for p in Path(__file__).parent.glob("*.py")}}
    for base in (framework / "cosmos_framework", framework / "packages"):
        for path in sorted(base.rglob("*.py")):
            if not any(part in {".venv", "__pycache__"} for part in path.parts):
                files[str(path.relative_to(framework))] = path
    for name in ("pyproject.toml", "uv.lock", ".python-version"):
        if (framework / name).is_file():
            files[name] = framework / name
    digest = hashlib.sha256()
    for name, path in sorted(files.items()):
        digest.update(f"{name}\0{file_sha256(path)}\n".encode())
    commit = subprocess.check_output(["git", "-C", str(framework), "rev-parse", "HEAD"], text=True).strip()
    return {"source_sha256": digest.hexdigest(), "framework_commit": commit}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--framework", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--action-space", choices=("joint", "eef"), default="joint")
    parser.add_argument("--eef-root", type=Path)
    parser.add_argument("--mode", choices=("prepare", "dryrun", "smoke", "train"), default="smoke")
    parser.add_argument("--run-name", default="yam_joint_gb200_v1")
    parser.add_argument("--samples-per-rank", type=int, choices=(1, 2, 4), default=4)
    parser.add_argument("--epochs", type=float, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.run_name or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.run_name
    ):
        parser.error("Use letters, numbers, underscores and hyphens in --run-name")
    if not (0 < args.epochs <= 100):
        parser.error("--epochs must be positive and <= 100")
    for name in ("dataset", "checkpoint", "vae", "work_dir", "framework", "recipe", "splits"):
        setattr(args, name, getattr(args, name).resolve())
    if args.eef_root:
        args.eef_root = args.eef_root.resolve()
    for path in (
        args.dataset / "meta/info.json",
        args.dataset / "failuregen_annotations.json",
        args.checkpoint / "config.json",
        args.vae,
        args.recipe,
        args.splits,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if shutil.which("ffmpeg") is None:
        raise FileNotFoundError("FFmpeg is required in the Cosmos container")

    args.work_dir.mkdir(parents=True, exist_ok=True)
    # One preparation/conversion/training process owns this work directory.
    with (args.work_dir / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another job is using this WORK_DIR") from exc
        execute(args)


def execute(args):
    action_space = getattr(args, "action_space", "joint")
    eef_root = getattr(args, "eef_root", None)
    prepared, checkpoint = args.work_dir / "prepared", args.work_dir / "initialization"
    if not prepared.exists():
        if action_space == "eef":
            from .prepare_eef import prepare

            prepare(args.dataset, prepared, eef_root=eef_root, splits_path=args.splits)
        else:
            from .prepare import prepare

            prepare(args.dataset, prepared, splits_path=args.splits)
    manifest = validate_source(prepared, args.dataset, args.splits, action_space=action_space, eef_root=eef_root)
    print("Prepared windows:", manifest["window_counts"], flush=True)
    # Prevent reusing an initialization with a different base model or converter.
    init_identity = {
        "action_space": action_space,
        "source": str(args.checkpoint),
        "config_sha256": file_sha256(args.checkpoint / "config.json"),
        "convert_sha256": file_sha256(Path(__file__).with_name("convert.py")),
        "expand_sha256": file_sha256(Path(__file__).with_name("expand.py")),
        # Model files are large: record size/mtime instead of reading all weights twice.
        "source_files": {
            str(p.relative_to(args.checkpoint)): [p.stat().st_size, p.stat().st_mtime_ns]
            for p in sorted(args.checkpoint.rglob("*"))
            if p.is_file() and not any(part.startswith(".") for part in p.relative_to(args.checkpoint).parts)
        },
    }
    write_contract(args.work_dir / "initialization_source.json", init_identity)
    if not checkpoint.exists():
        run(
            [
                sys.executable,
                "-m",
                "failuregen.yam_training.convert",
                "--checkpoint-path",
                args.checkpoint,
                "--reuse-yam-eef-domain" if action_space == "eef" else "--append-yam-domain",
                "-o",
                checkpoint,
            ]
        )
    plan = build_plan(
        prepared,
        checkpoint,
        samples_per_rank=args.samples_per_rank,
        epochs=args.epochs,
        smoke=args.mode == "smoke",
        action_space=action_space,
    )
    if args.mode == "prepare":
        print(f"Data and initialization ready under {args.work_dir}")
        return
    run_name = args.run_name + (f"_{args.mode}" if args.mode != "train" else "")
    output_root = args.work_dir / "training"
    run_dir = output_root / "cosmos3_yam/fd_id" / run_name
    started = run_dir / "training_started.json"
    if started.exists() and not args.resume:
        raise ValueError(f"Training was already started in {run_dir}; use RESUME=1 or a new RUN_NAME")
    plan.update(source_fingerprint(args.framework, args.recipe))
    plan.update(
        vae_sha256=file_sha256(args.vae),
        dataset=str(args.dataset),
        container_image=os.environ.get("CONTAINER_IMAGE", "unspecified"),
    )
    write_contract(run_dir / "yam_training_contract.json", plan)
    env = dict(
        os.environ,
        YAM_DATASET_ROOT=str(args.dataset),
        YAM_PREPARED_ROOT=str(prepared),
        BASE_CHECKPOINT_PATH=str(checkpoint),
        WAN_VAE_PATH=str(args.vae),
        IMAGINAIRE_OUTPUT_ROOT=str(output_root),
        YAM_RUN_NAME=run_name,
        YAM_ACTION_SPACE=action_space,
        YAM_SAMPLES_PER_RANK=str(args.samples_per_rank),
        YAM_MAX_ITER=str(plan["max_iter"]),
    )
    print(
        f"YAM {action_space} domain {plan['contract']['domain_id']}: 4 GPUs, "
        f"batch {plan['global_batch']}, {plan['max_iter']} updates",
        flush=True,
    )
    command = [sys.executable, "-m", "failuregen.yam_training.train", "--sft-toml", args.recipe]
    tail = training_overrides(run_name, smoke=args.mode == "smoke")
    # Every mode writes a resolved config before allocating the distributed model.
    run([*command, "--dryrun", *tail], env=env)
    validate_resolved_config(run_dir / "config.yaml")
    print(
        "Resolved config verified: sigma diagnostic and unsupported trainer validation disabled; "
        "evaluate saved checkpoints with held-out replay",
        flush=True,
    )
    if args.mode == "dryrun":
        return
    if not started.exists():
        write_contract(started, {"slurm_job_id": os.environ.get("SLURM_JOB_ID"), "mode": args.mode})
    run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            "--nproc-per-node=4",
            "-m",
            "failuregen.yam_training.train",
            "--sft-toml",
            args.recipe,
            *tail,
        ],
        env=env,
    )
    latest = run_dir / "checkpoints/latest_checkpoint.txt"
    if not latest.is_file():
        raise RuntimeError(f"Training exited without a saved checkpoint: {latest}")
    saved = run_dir / "checkpoints" / latest.read_text().strip()
    if not (saved / "model/.metadata").is_file():
        raise RuntimeError(f"Latest checkpoint has no model metadata: {saved}")
    print(f"Training finished. Saved checkpoint: {latest.read_text().strip()}\nRun: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
