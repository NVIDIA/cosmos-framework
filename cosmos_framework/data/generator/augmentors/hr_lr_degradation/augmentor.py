# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Dataloader augmentors that turn a single-stream sample into an (LR, HR) super-resolution sample.

Two stages, meant for the cosmos3 Lance video/image pipelines:

``AddLowRes`` runs right after the media is at its final HR resolution and before reflection
padding. It writes ``data_dict[output_key]`` (uint8, same layout as the input) at ``1/scale`` of
the HR size using a seeded degradation profile, plus ``data_dict[record_key]``, a JSON string with every sampled
parameter (a string collates cleanly; a dict of variable-length lists does not). The seed derives from the sample key so a sample always gets the same LR.

``SRToTrainingFormat`` runs last. It pads LR to half the HR padding bucket, packs
``data_dict[media_key] = [lr, hr]`` (the joint dataloader treats every item before the last as
pure conditioning, as in ``TransferToTrainingFormat``), writes one ``image_size`` entry per item so
``OmniMoTModel._remove_padding_from_latent`` crops each latent correctly, and marks the
``SequencePlan`` with ``share_vision_temporal_positions=False`` because the two items have
different latent grids (``sequence_packing/packers.py`` asserts equal grids when sharing).
"""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any, Mapping, Optional

import numpy as np
import torch
import torchvision.transforms.functional as transforms_F

from cosmos_framework.data.imaginaire.webdataset.augmentors.augmentor import Augmentor
from cosmos_framework.utils import log
from cosmos_framework.data.generator.augmentors.hr_lr_degradation.degrade import degrade_hr_to_lr
from cosmos_framework.data.generator.augmentors.hr_lr_degradation.diffjpeg import DiffJPEG
from cosmos_framework.data.generator.augmentors.hr_lr_degradation.profiles import get_profile
from cosmos_framework.data.generator.sequence_packing import SequencePlan

DEFAULT_RECORD_KEY = "degradation_record"


def seed_from_sample(data_dict: Mapping[str, Any], salt: str = "") -> int:
    """Stable 63-bit seed from the sample identity (``__key__``), or a random one when absent."""
    key = data_dict.get("__key__")
    if key is None:
        return random.getrandbits(63)
    digest = hashlib.blake2b(f"{key}|{salt}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") & 0x7FFF_FFFF_FFFF_FFFF


DEFAULT_FPS = 24.0


def clip_fps(data_dict: Mapping[str, Any]) -> float:
    """Effective frame rate of the frames in the sample, for the codec stage.

    ``conditioning_fps`` is the native rate divided by the sampled stride, i.e. the rate at which the kept
    frames actually play and the rate the model, captions and mRoPE use. ``fps`` is the source file's
    native rate and is only right when the stride is 1. Images carry neither and fall back to a default
    (the codec is skipped for them anyway).
    """
    for key in ("conditioning_fps", "fps"):
        value = data_dict.get(key)
        if value is None:
            continue
        if isinstance(value, torch.Tensor):
            value = value.reshape(-1)[0].item()
        if float(value) > 0:
            return float(value)
    return DEFAULT_FPS


def _as_uint8_media(frames: Any) -> torch.Tensor:  # returns [C,T,H,W] or [C,H,W] uint8
    if isinstance(frames, np.ndarray):
        frames = torch.from_numpy(frames)
    if not isinstance(frames, torch.Tensor):
        raise TypeError(f"AddLowRes expects a tensor or ndarray, got {type(frames).__name__}")
    if frames.dtype == torch.uint8:
        return frames
    if frames.is_floating_point():
        if frames.numel() > 0 and frames.min() < 0.0:
            raise ValueError("AddLowRes must run before normalisation (got values below 0)")
        return (frames.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
    raise TypeError(f"Unsupported media dtype {frames.dtype}")


class AddLowRes(Augmentor):
    """Add a degraded low-resolution copy of ``input_keys[0]`` under ``output_keys[0]``.

    args:
        scale: HR-to-LR factor (default 2).
        profiles: mapping profile name -> sampling weight, or a single profile name string.
        seed_salt: extra string mixed into the per-sample seed (use to decorrelate ablation arms).
        chunk_frames: frames per degradation step (peak-memory bound).
        device: ``"cpu"`` (dataloader workers) or a CUDA device string for GPU-side use.
        jpeg_backend / poisson_mode: forwarded to ``degrade_hr_to_lr``.
        modality: ``"image"`` or ``"video"``; when set, profiles declared for the other modality are rejected
            at construction (a video regime on single images would have no compression term at all).
        record_key: where the parameter record is written.
    """

    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)
        args = dict(args or {})
        if len(self.input_keys) != 1:
            raise ValueError("AddLowRes takes exactly one input key")
        self.output_key = (output_keys or [f"{self.input_keys[0]}_lr"])[0]
        self.scale = float(args.get("scale", 2.0))
        profiles = args.get("profiles", "p1_first_order")
        if isinstance(profiles, str):
            profiles = {profiles: 1.0}
        self.profile_names = list(profiles.keys())
        weights = np.asarray([float(profiles[n]) for n in self.profile_names], dtype=np.float64)  # [P]
        if weights.sum() <= 0:
            raise ValueError("profile weights must sum to a positive number")
        self.profile_weights = weights / weights.sum()  # [P]
        self.modality = args.get("modality")
        for name in self.profile_names:
            prof = get_profile(name)  # fail early on typos
            if self.modality is not None and prof.modality not in ("any", self.modality):
                raise ValueError(f"profile {name!r} is for {prof.modality} data, this AddLowRes serves {self.modality}")
        self.seed_salt = str(args.get("seed_salt", ""))
        self.chunk_frames = int(args.get("chunk_frames", 8))
        self.device = torch.device(args.get("device", "cpu"))
        self.jpeg_backend = str(args.get("jpeg_backend", "auto"))
        self.poisson_mode = str(args.get("poisson_mode", "auto"))
        self.record_key = str(args.get("record_key", DEFAULT_RECORD_KEY))
        self._jpeger: DiffJPEG | None = None

    def _jpeger_for(self, device: torch.device) -> DiffJPEG:
        if self._jpeger is None:
            self._jpeger = DiffJPEG(differentiable=False)
        if self._jpeger.y_table.device != device:
            self._jpeger.to(device)
        return self._jpeger

    def __call__(self, data_dict: dict) -> dict | None:
        media = data_dict.get(self.input_keys[0])
        if media is None:
            log.warning(f"AddLowRes: missing {self.input_keys[0]} in {data_dict.get('__key__', 'unknown')}")
            return None
        hr = _as_uint8_media(media)  # [C,T,H,W] or [C,H,W]
        seed = seed_from_sample(data_dict, self.seed_salt)
        # The profile draw must not share a stream with the plan: degrade_hr_to_lr re-creates default_rng(seed),
        # so drawing from default_rng(seed) here would make the mixture choice and the plan's first probability
        # gate the same uniform (e.g. a 30% clean regime whose only gate then fires 100% of the time).
        profile_rng = np.random.default_rng(seed_from_sample(data_dict, self.seed_salt + "|profile"))
        profile_name = self.profile_names[int(profile_rng.choice(len(self.profile_names), p=self.profile_weights))]
        result = degrade_hr_to_lr(
            hr.to(self.device, non_blocking=False),
            profile_name,
            scale=self.scale,
            seed=seed,
            chunk_frames=self.chunk_frames,
            jpeger=self._jpeger_for(self.device),
            jpeg_backend=self.jpeg_backend,
            poisson_mode=self.poisson_mode,
            fps=clip_fps(data_dict),  # only the codec stage (P3) uses it
        )
        data_dict[self.output_key] = result.lr.cpu()  # [C,T,h,w] or [C,h,w] uint8
        # JSON string: records differ in length between samples, so a dict would break default_collate.
        data_dict[self.record_key] = json.dumps(result.record)
        return data_dict


def _pad_to(frames: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:  # frames: [...,H,W]
    """One-sided reflect padding (bottom/right), edge padding when the pad exceeds the content."""
    h, w = frames.shape[-2:]
    pad_right, pad_bottom = target_w - w, target_h - h
    if pad_right < 0 or pad_bottom < 0:
        raise ValueError(f"Cannot pad {(h, w)} to smaller target {(target_h, target_w)}")
    if pad_right == 0 and pad_bottom == 0:
        return frames
    mode = "edge" if (pad_right >= w or pad_bottom >= h) else "reflect"
    return transforms_F.pad(frames, [0, 0, pad_right, pad_bottom], padding_mode=mode)  # [...,tH,tW]


class SRToTrainingFormat(Augmentor):
    """Pack (LR, HR) into the two-item conditioning format with per-item ``image_size``.

    args:
        media_key: ``"video"`` or ``"images"`` (the HR key; the LR key defaults to ``f"{media_key}_lr"``).
        lr_key: override for the LR key.
        scale: HR-to-LR factor; the LR padding bucket is the HR bucket divided by this.
        share_vision_temporal_positions: keep False for native-resolution LR (default).
        dataset_name: value written to ``data_dict["dataset_name"]``.
        drop_lr_key: remove the standalone LR key after packing (default True).
    """

    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)
        args = dict(args or {})
        self.media_key = str(args.get("media_key", "video"))
        self.lr_key = str(args.get("lr_key", f"{self.media_key}_lr"))
        self.scale = float(args.get("scale", 2.0))
        self.share_vision_temporal_positions = bool(args.get("share_vision_temporal_positions", False))
        default_name = "image_sr" if self.media_key == "images" else f"{self.media_key}_sr"
        self.dataset_name = str(args.get("dataset_name", default_name))
        self.drop_lr_key = bool(args.get("drop_lr_key", True))

    def __call__(self, data_dict: dict) -> dict | None:
        hr = data_dict.get(self.media_key)
        lr = data_dict.get(self.lr_key)
        if hr is None or lr is None or not isinstance(hr, torch.Tensor) or not isinstance(lr, torch.Tensor):
            log.warning(
                f"SRToTrainingFormat: missing {self.media_key} or {self.lr_key} in {data_dict.get('__key__', 'unknown')}",
                rank0_only=False,
            )
            return None
        hr_size = data_dict.get("image_size")
        if hr_size is None:
            hr_size = torch.tensor([hr.shape[-2], hr.shape[-1], hr.shape[-2], hr.shape[-1]], dtype=torch.float)  # [4]
        hr_size = torch.as_tensor(hr_size, dtype=torch.float).reshape(-1)  # [4] = [tH,tW,oH,oW]
        target_h, target_w = int(hr_size[0].item()), int(hr_size[1].item())
        lr_target_h = int(np.ceil(target_h / self.scale))
        lr_target_w = int(np.ceil(target_w / self.scale))
        lr_orig_h, lr_orig_w = int(lr.shape[-2]), int(lr.shape[-1])
        lr_padded = _pad_to(lr, lr_target_h, lr_target_w)  # [C,T,th,tw] or [C,th,tw]
        if lr_padded.dtype != hr.dtype:
            # HR may already be normalised float (image pipeline); match dtype so the model treats both alike.
            if hr.is_floating_point() and lr_padded.dtype == torch.uint8:
                raise ValueError(
                    "HR is float but LR is uint8: add a Normalize stage for the LR key before SRToTrainingFormat"
                )
        lr_size = torch.tensor([lr_target_h, lr_target_w, lr_orig_h, lr_orig_w], dtype=torch.float)  # [4]

        data_dict[self.media_key] = [lr_padded, hr]
        data_dict["image_size"] = [lr_size, hr_size]
        data_dict["dataset_name"] = self.dataset_name
        plan = data_dict.get("sequence_plan")
        if plan is None:
            plan = SequencePlan(has_text=True, has_vision=True, condition_frame_indexes_vision=[])
        plan.share_vision_temporal_positions = self.share_vision_temporal_positions
        data_dict["sequence_plan"] = plan
        if self.drop_lr_key:
            del data_dict[self.lr_key]
        return data_dict
