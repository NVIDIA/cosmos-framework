# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Evaluation entrypoint."""

from cosmos_framework.inference.common.init import init_script, is_rank0

init_script(
    env={
        "COSMOS_TRAINING": "1",
    }
)

import json
from pathlib import Path

import pydantic
import torch
import tyro

from cosmos_framework.inference.args import OmniSetupOverrides
from cosmos_framework.inference.common.args import SampleOutputs, SetupOverrides, tyro_cli
from cosmos_framework.inference.common.checkpoints import register_checkpoints
from cosmos_framework.inference.common.init import init_output_dir
from cosmos_framework.inference.dataset import DatasetArgs, create_dataset
from cosmos_framework.inference.scripts.eval_utils import (
    aggregate_metrics,
    compute_sample_metrics,
    compute_video_metrics,
    derive_match_key_and_group,
    extract_gt_action,
    extract_gt_video,
)
from cosmos_framework.inference.vision import read_media_frames
from cosmos_framework.utils import log


class EvalArgs(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid", use_attribute_docstrings=True)

    setup: tyro.conf.OmitArgPrefixes[SetupOverrides] = OmniSetupOverrides.model_construct()
    """Model and parallelism configuration."""
    dataset: DatasetArgs = DatasetArgs.model_construct()
    """Dataset loading configuration. ``dataset.model_mode`` selects the eval path:
    ``joint`` / ``forward_dynamics`` / ``inverse_dynamics`` / ``policy`` → action eval
    (dataset-driven inference + scoring); ``vision`` → score pre-generated videos vs GT
    (predictions must already exist; generate them with ``cosmos3.scripts.inference``)."""
    compute_metrics: bool = True
    """Compute per-sample metrics and write metrics.json sidecars + metrics_aggregate.json."""
    gt_dir: Path | None = None
    """Directory of ground-truth videos. Required for vision eval."""
    predictions_dir: Path | None = None
    """Root containing pre-generated prediction videos. Required for vision eval."""
    predictions_glob: str = "**/vision.mp4"
    """Glob (relative to ``predictions_dir``) for the predicted videos to score."""
    gt_extension: str = ".mp4"
    """File extension on the GT side (e.g. ``.mp4``, ``.mov``)."""


def eval_action(args: EvalArgs) -> list[SampleOutputs]:
    """Run action-policy dataset inference: load dataset in memory, run inference, save outputs."""
    if args.setup.output_dir is None:
        raise ValueError("'output_dir' is required")
    if args.dataset.model_mode == "vision":
        raise ValueError("eval_action requires an action mode; got dataset.model_mode='vision'")

    setup = args.setup.build_setup()
    init_output_dir(setup.output_dir)
    log.debug(f"{args.__class__.__name__}({args})")

    register_checkpoints()
    samples = create_dataset(
        args.dataset,
        config_args=setup,
    )
    log.info(f"Loaded {len(samples)} samples in memory")

    pipe = setup.get_inference_cls().create(setup)

    output_dir = setup.output_dir
    all_outputs: list[SampleOutputs] = []
    for i, (sample_args, data_batch) in enumerate(samples):
        assert sample_args.name
        sample_args.output_dir = output_dir / sample_args.name
        sample_args = sample_args.build_sample(model_config=pipe.model_config)
        log.info(f"[{i + 1}/{len(samples)}] Processing: {sample_args.name}")

        gt_video: torch.Tensor | None = None
        gt_action: torch.Tensor | None = None
        if args.compute_metrics:
            gt_video = extract_gt_video(data_batch)
            gt_action = extract_gt_action(data_batch)

        batch_outputs = pipe.generate_batch([sample_args], data_batch)
        all_outputs.extend(batch_outputs)

        if args.compute_metrics and batch_outputs:
            sample_output = batch_outputs[0]
            if sample_output.status == "success":
                metrics = compute_sample_metrics(
                    sample_args.name,
                    gt_video,
                    gt_action,
                    sample_output,
                    sample_args.output_dir,
                    sample_args.vision_extension,
                )
                (sample_args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
                log.info(f"Metrics for {sample_args.name}: {metrics}")

    if setup.benchmark and is_rank0():
        benchmark_file = output_dir / "benchmark.json"
        benchmark_file.write_text(json.dumps(pipe.get_timer_results(), indent=2, sort_keys=True))
        log.success(f"Saved benchmark to '{benchmark_file}'")

    if args.compute_metrics and is_rank0():
        aggregate = aggregate_metrics(output_dir)
        aggregate_file = output_dir / "metrics_aggregate.json"
        aggregate_file.write_text(json.dumps(aggregate, indent=2, sort_keys=True))
        log.success(f"Saved aggregated metrics to '{aggregate_file}'")

    return all_outputs


def eval_vision(args: EvalArgs) -> None:
    """Score pre-generated videos against a ground-truth directory. CPU-only.

    Predictions must already exist on disk (generate them with
    ``cosmos3.scripts.inference``; see training.md). This function only pairs
    each prediction with a GT video and computes per-clip PSNR / SSIM.
    """
    if args.dataset.model_mode != "vision":
        raise ValueError(f"eval_vision requires dataset.model_mode='vision'; got {args.dataset.model_mode!r}")
    if args.setup.output_dir is None:
        raise ValueError("'setup.output_dir' is required")
    if args.gt_dir is None:
        raise ValueError("'gt_dir' is required for vision eval")
    if args.predictions_dir is None:
        raise ValueError("'predictions_dir' is required for vision eval")
    if not args.gt_dir.exists():
        raise FileNotFoundError(f"gt_dir does not exist: {args.gt_dir}")
    if not args.predictions_dir.exists():
        raise FileNotFoundError(f"predictions_dir does not exist: {args.predictions_dir}")

    output_dir = args.setup.output_dir
    init_output_dir(output_dir)
    log.debug(f"{args.__class__.__name__}({args})")

    pred_paths = sorted(args.predictions_dir.glob(args.predictions_glob))
    log.info(f"Found {len(pred_paths)} prediction file(s) under {args.predictions_dir} / {args.predictions_glob!r}")
    if not pred_paths:
        raise ValueError(f"no prediction files matched {args.predictions_glob!r} under {args.predictions_dir}")

    scored = 0
    skipped_missing_gt = 0
    for i, pred_path in enumerate(pred_paths):
        match_key, group = derive_match_key_and_group(pred_path, args.predictions_dir)
        bucket = group or "default"
        gt_path = args.gt_dir / f"{match_key}{args.gt_extension}"
        if not gt_path.exists():
            log.warning(f"[{i + 1}/{len(pred_paths)}] missing GT for match_key={match_key!r} at {gt_path}; skipping")
            skipped_missing_gt += 1
            continue

        # Load GT — read all frames; `compute_video_metrics` caps the pred read with +1.
        gt_video, _ = read_media_frames(gt_path, max_frames=10**9)

        try:
            video_metrics = compute_video_metrics(gt_video, pred_path, mode="vision")
        except ValueError as e:
            log.warning(f"[{i + 1}/{len(pred_paths)}] skip {pred_path} (shape mismatch): {e}")
            continue

        sample_dir = output_dir / bucket / match_key
        sample_dir.mkdir(parents=True, exist_ok=True)
        metrics = {"mode": bucket, "name": match_key, **video_metrics}
        (sample_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
        log.info(f"[{i + 1}/{len(pred_paths)}] {bucket}/{match_key}: {video_metrics}")

        scored += 1

    log.info(f"scored {scored} / {len(pred_paths)} samples; skipped {skipped_missing_gt} for missing GT")

    if is_rank0():
        aggregate = aggregate_metrics(output_dir)
        aggregate_file = output_dir / "metrics_aggregate.json"
        aggregate_file.write_text(json.dumps(aggregate, indent=2, sort_keys=True))
        log.success(f"Saved aggregated metrics to '{aggregate_file}'")


def main() -> None:
    args = tyro_cli(EvalArgs, description=__doc__)
    if args.dataset.model_mode == "vision":
        eval_vision(args)
    else:
        eval_action(args)


if __name__ == "__main__":
    main()
