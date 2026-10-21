# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Helpers for `eval.py`: per-sample metric computation and aggregation."""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cosmos_framework.data.generator.action.utils.transforms import remove_reflection_padding
from cosmos_framework.inference.common.args import SampleOutputs
from cosmos_framework.inference.metrics import compute_action_mse, compute_psnr
from cosmos_framework.inference.vision import read_media_frames
from cosmos_framework.utils import log

VIDEO_MODES = {"forward_dynamics"}
ACTION_MODES = {"inverse_dynamics"}
BOTH_MODES = {"wam"}
ALL_MODES = VIDEO_MODES | ACTION_MODES | BOTH_MODES


def extract_gt_video(data_batch: dict, spatial_compression_factor: int) -> torch.Tensor | None:
    """Snapshot GT video as uint8, matching the generated video's spatial alignment.

    Must be called BEFORE the inference pipeline runs — the model normalizes
    `data_batch["video"]` in place from uint8 [0, 255] to float [-1, 1]. The model
    removes padding in latent space with floor division by the tokenizer's spatial
    compression factor, so use that same aligned content extent for the pixel GT.
    """
    video = data_batch.get("video")
    if video is None:
        return None
    gt_video = video[0].detach().clone()
    image_size = data_batch.get("image_size")
    if image_size is None:
        return gt_video
    content = remove_reflection_padding(gt_video, image_size)
    aligned_h = content.shape[-2] // spatial_compression_factor * spatial_compression_factor
    aligned_w = content.shape[-1] // spatial_compression_factor * spatial_compression_factor
    return content[..., :aligned_h, :aligned_w].contiguous()  # [C,T,H_aligned,W_aligned]


def extract_gt_action(data_batch: dict) -> tuple[torch.Tensor, torch.Tensor | None] | None:
    """Snapshot the GT action and optional valid-channel mask, or None when absent."""
    action = data_batch.get("action_raw", [None])[0]
    if action is None:
        return None
    action = action.detach().clone().float()  # [T,D_raw]
    valid_mask = data_batch.get("action_valid_mask", [None])[0]
    if valid_mask is not None:
        valid_mask = valid_mask[: action.shape[-1]].detach().clone().bool()  # [D_raw]
    return action, valid_mask


def _parse_mode_from_name(name: str) -> str:
    parts = name.split("/")
    if len(parts) < 2:
        raise ValueError(f"unexpected sample name: {name!r}")
    mode = parts[-2]
    if mode not in ALL_MODES:
        raise ValueError(f"unexpected mode {mode!r} in sample name {name!r}; expected one of {sorted(ALL_MODES)}")
    return mode


def derive_match_key_and_group(
    pred_path: Path,
    predictions_dir: Path,
) -> tuple[str, str]:
    """Path → ``(match_key, group)``. Used by vision eval to pair predictions with GT.

    ``vision.*`` paths use the canonical ``<group>/<sample>/<output_index>/vision.*``
    layout. Otherwise ``match_key`` is the filename stem.

    Examples (with ``predictions_dir=/root``):
        ``/root/t2v/episode_0/0/vision.mp4`` → ``("episode_0", "t2v")``
        ``/root/sub/foo.mp4`` → ``("foo", "sub")``
    """
    pred_path = pred_path.resolve()
    predictions_dir = predictions_dir.resolve()
    if not pred_path.is_relative_to(predictions_dir):
        raise ValueError(f"pred_path {pred_path} is not under predictions_dir {predictions_dir}")
    rel = pred_path.relative_to(predictions_dir)
    parts = rel.parts
    if pred_path.name.startswith("vision."):
        if len(parts) < 3 or not parts[-2].isdigit():
            raise ValueError(f"expected <group>/<key>/<output_index>/vision.* under predictions_dir, got rel={rel}")
        match_key = parts[-3]
        group_parts = parts[:-3]
    else:
        match_key = pred_path.stem
        group_parts = parts[:-1]
    group = "/".join(group_parts)
    return match_key, group


def compute_video_metrics(
    gt_video_cthw_uint8: torch.Tensor,
    pred_path: Path,
    mode: str,
) -> dict[str, float]:
    """Compute per-clip PSNR. Temporal-mismatch policy depends on *mode*:

    - ``"vision"``: lenient. If ``T_gt != T_pred`` (after the H/W top-left crop), trim both to
      ``min(T_gt, T_pred)`` from the start (VFM generation is aligned to ``4k+1`` frames —
      latent temporal factor 4 + 1 conditioning frame — while the on-disk GT keeps the raw
      clip length, so small T deltas (e.g. GT=96, pred=93) are expected and treating them as
      hard errors loses the entire eval) and log an info line.
    - any other mode (``forward_dynamics``, ``wam``, ...): strict. Pred T is fixed by
      the action chunk size; a mismatch indicates a real bug.

    Spatial mismatch always errors. The GT extraction path is responsible for applying
    the same tokenizer alignment as generation.
    """
    # +1 so an over-long prediction surfaces as a shape mismatch instead of silent truncation.
    pred, _ = read_media_frames(pred_path, max_frames=gt_video_cthw_uint8.shape[1] + 1)
    gt = gt_video_cthw_uint8
    if pred.shape != gt.shape:
        if pred.shape[-2:] != gt.shape[-2:]:
            raise ValueError(f"video spatial mismatch: gt {tuple(gt.shape)} vs pred {tuple(pred.shape)} ({pred_path})")
        # Temporal mismatch: lenient for vision eval, strict otherwise.
        if mode == "vision":
            min_t = min(gt.shape[1], pred.shape[1])
            log.info(
                f"vision frame-count mismatch trimmed to {min_t} (gt T={gt.shape[1]}, pred T={pred.shape[1]}, "
                f"{pred_path}); likely due to generation aligned to 4k+1 frames "
                "(latent temporal factor 4 + 1 conditioning frame), while the on-disk GT "
                "keeps the raw clip length."
            )
            gt = gt[:, :min_t]
            pred = pred[:, :min_t]
        else:
            raise ValueError(f"video shape mismatch: gt {tuple(gt.shape)} vs pred {tuple(pred.shape)} ({pred_path})")

    return {"psnr": compute_psnr(gt, pred)}


def _compute_action_metrics(
    gt_action_td: torch.Tensor,
    pred_action_list: list,
    valid_mask: torch.Tensor | None,
) -> dict[str, Any]:
    pred = torch.tensor(pred_action_list, dtype=torch.float32)
    if pred.shape != gt_action_td.shape:
        raise ValueError(f"action shape mismatch: gt {tuple(gt_action_td.shape)} vs pred {tuple(pred.shape)}")
    if valid_mask is not None:
        gt_action_td = gt_action_td[..., valid_mask]
        pred = pred[..., valid_mask]
    return {"action_mse": compute_action_mse(gt_action_td, pred)}


def compute_sample_metrics(
    name: str,
    gt_video_cthw: torch.Tensor | None,
    gt_action: tuple[torch.Tensor, torch.Tensor | None] | None,
    sample_output: SampleOutputs,
    sample_dir: Path,
    vision_extension: str,
) -> dict[str, Any]:
    """Compute metrics for a single sample, dispatched by the mode parsed from `name`."""
    mode = _parse_mode_from_name(name)
    out: dict[str, Any] = {"mode": mode, "name": sample_dir.name}
    if mode in VIDEO_MODES | BOTH_MODES:
        if gt_video_cthw is None:
            raise ValueError(f"mode={mode!r} requires GT video but data_batch had none")
        out.update(compute_video_metrics(gt_video_cthw, sample_dir / f"vision{vision_extension}", mode))
    if mode in ACTION_MODES | BOTH_MODES:
        pred_action = sample_output.outputs[0].content.get("action") if sample_output.outputs else None
        if pred_action is None:
            raise ValueError(f"mode={mode!r} requires predicted action but content has none")
        if gt_action is None:
            raise ValueError(f"mode={mode!r} requires GT action but data_batch had none")
        gt_action_td, valid_mask = gt_action
        out.update(_compute_action_metrics(gt_action_td, pred_action, valid_mask))
    return out


def aggregate_metrics(output_dir: Path) -> dict[str, Any]:
    """Walk `output_dir` for per-sample `metrics.json` files; emit per-mode/metric summary.

    Each scalar metric is summarised as ``{mean, count}``.
    """
    totals: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for f in output_dir.rglob("metrics.json"):
        m = json.loads(f.read_text())
        mode = m.pop("mode", None)
        m.pop("name", None)
        if mode is None:
            continue
        for k, v in m.items():
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    totals[mode][f"{k}/{sub_k}"].append(float(sub_v))
            else:
                totals[mode][k].append(float(v))
    return {
        mode: {metric: {"mean": float(np.mean(vals)), "count": len(vals)} for metric, vals in metrics.items()}
        for mode, metrics in totals.items()
    }
