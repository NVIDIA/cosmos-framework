# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Entry point: turn an HR clip or image into its degraded LR counterpart.

Two-phase design so the result is reproducible and cheap to log:

1. ``plan_degradation`` samples every parameter for the clip from ``numpy.random.default_rng(seed)``
   and resolves concrete intermediate sizes. The plan doubles as the degradation record.
2. ``apply_plan`` executes the plan chunk by chunk over time. Only per-pixel noise is drawn
   here, from a torch generator seeded with ``seed`` on the input's device: the sampled
   parameters are shared across devices, the noise realisation is device specific.

Parameters are sampled once per clip (clip-consistent), matching video SR practice
(RealBasicVSR, Upscale-A-Video, SeedVR). Transfer1 drew one set per batch and JPEG per frame.
"""

from __future__ import annotations

import dataclasses
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import numpy as np
import torch

from cosmos_framework.data.generator.augmentors.hr_lr_degradation import ops
from cosmos_framework.data.generator.augmentors.hr_lr_degradation.codec import codec_round_trip
from cosmos_framework.data.generator.augmentors.hr_lr_degradation.diffjpeg import DiffJPEG
from cosmos_framework.data.generator.augmentors.hr_lr_degradation.kernels import (
    random_mixed_kernel,
    random_sinc_kernel,
    scale_kernel_size,
)
from cosmos_framework.data.generator.augmentors.hr_lr_degradation.profiles import (
    BlurConfig,
    CleanResizeProfile,
    CodecConfig,
    DegradationStage,
    FinalBlockConfig,
    JPEGConfig,
    NoiseConfig,
    Profile,
    RealESRGANProfile,
    ResizeConfig,
    get_profile,
)


@dataclass
class PlannedOp:
    """One primitive with fully resolved parameters. ``kernel`` is kept out of the record; ``stage`` names the
    block that produced the op (``stage1`` / ``stage2`` / ``final`` / ``codec``) so records can be sliced per block."""

    op: str
    params: dict[str, Any] = field(default_factory=dict)
    kernel: np.ndarray | None = field(default=None, repr=False, compare=False)
    stage: str = ""

    def record(self) -> dict[str, Any]:
        if {"op", "stage"} & self.params.keys():
            raise ValueError("'op' and 'stage' are reserved record keys")
        return {"op": self.op, "stage": self.stage, **self.params}


@dataclass
class DegradationPlan:
    profile_name: str
    seed: int
    scale: float
    hr_size: tuple[int, int]
    lr_size: tuple[int, int]
    ops: list[PlannedOp]
    codec: PlannedOp | None = None  # whole-clip op applied after the per-frame ops (P3)

    def record(self) -> dict[str, Any]:
        """JSON-serialisable summary of every sampled parameter."""
        out = asdict(self)
        out["hr_size"] = list(self.hr_size)
        out["lr_size"] = list(self.lr_size)
        out["ops"] = [op.record() for op in self.ops]
        out["codec"] = None if self.codec is None else self.codec.record()
        return out


@dataclass
class DegradationResult:
    lr: torch.Tensor  # uint8, same layout as the input ([C,T,h,w] or [C,h,w])
    record: dict[str, Any]


def _lr_size(hr_size: tuple[int, int], scale: float, target_size: tuple[int, int] | None) -> tuple[int, int]:
    if target_size is not None:
        return int(target_size[0]), int(target_size[1])
    return max(1, int(round(hr_size[0] / scale))), max(1, int(round(hr_size[1] / scale)))


def _plan_blur(cfg: BlurConfig, rng: np.random.Generator, res_factor: float, max_kernel: int) -> PlannedOp | None:
    if rng.uniform() >= cfg.prob:
        return None
    base_size = int(rng.choice(np.asarray(cfg.kernel_range)))
    kernel_size = scale_kernel_size(base_size, res_factor, max_kernel)
    if rng.uniform() < cfg.sinc_prob:
        kernel, info = random_sinc_kernel(rng, kernel_size, cfg.sinc_cutoff_range)  # [K,K]
    else:
        sigma_range = (cfg.sigma_range[0] * res_factor, cfg.sigma_range[1] * res_factor)
        kernel, info = random_mixed_kernel(
            rng,
            cfg.kernel_list,
            cfg.kernel_prob,
            kernel_size,
            sigma_range,
            sigma_range,
            betag_range=cfg.betag_range,
            betap_range=cfg.betap_range,
        )  # [K,K]
    return PlannedOp("blur", info, kernel)


def _clamp_size(size: tuple[int, int], floor: tuple[int, int]) -> tuple[int, int]:
    return max(size[0], floor[0]), max(size[1], floor[1])


def _plan_resize(
    cfg: ResizeConfig,
    rng: np.random.Generator,
    current: tuple[int, int],
    target: tuple[int, int],
    floor: tuple[int, int],
) -> PlannedOp | None:
    if rng.uniform() >= cfg.prob:
        return None
    prob = np.asarray(cfg.updown_prob, dtype=np.float64)  # [3]
    updown = str(rng.choice(np.asarray(["up", "down", "keep"]), p=prob / prob.sum()))
    if updown == "up":
        factor = float(rng.uniform(1.0, cfg.scale_range[1]))
    elif updown == "down":
        factor = float(rng.uniform(cfg.scale_range[0], 1.0))
    else:
        factor = 1.0
    reference = current if cfg.relative_to == "current" else target
    size = (max(1, int(round(reference[0] * factor))), max(1, int(round(reference[1] * factor))))
    size = _clamp_size(size, floor)
    mode = str(rng.choice(np.asarray(cfg.modes)))
    # ``factor`` is the sampled value; ``factor_effective`` is the height ratio the emitted size realises after the
    # intermediate floor, so records can be sliced on what actually happened.
    return PlannedOp(
        "resize",
        {
            "updown": updown,
            "factor": factor,
            "factor_effective": size[0] / reference[0],
            "size": list(size),
            "mode": mode,
        },
    )


def _plan_noise(cfg: NoiseConfig, rng: np.random.Generator) -> PlannedOp | None:
    if rng.uniform() >= cfg.prob:
        return None
    gray = bool(rng.uniform() < cfg.gray_noise_prob)
    if rng.uniform() < cfg.gaussian_prob:
        sigma = float(rng.uniform(*cfg.gaussian_sigma_range))
        return PlannedOp("gaussian_noise", {"sigma": sigma, "gray": gray})
    scale = float(rng.uniform(*cfg.poisson_scale_range))
    return PlannedOp("poisson_noise", {"scale": scale, "gray": gray})


def _plan_jpeg(cfg: JPEGConfig, rng: np.random.Generator) -> PlannedOp | None:
    if rng.uniform() >= cfg.prob:
        return None
    return PlannedOp("jpeg", {"quality": float(rng.uniform(*cfg.quality_range))})


def _plan_stage(
    stage: DegradationStage,
    rng: np.random.Generator,
    current: tuple[int, int],
    target: tuple[int, int],
    floor: tuple[int, int],
    res_factor: float,
    max_kernel: int,
    stage_name: str,
) -> tuple[list[PlannedOp], tuple[int, int]]:
    planned: list[PlannedOp] = []
    if stage.blur is not None:
        op = _plan_blur(stage.blur, rng, res_factor, max_kernel)
        if op is not None:
            planned.append(op)
    if stage.resize is not None:
        op = _plan_resize(stage.resize, rng, current, target, floor)
        if op is not None:
            planned.append(op)
            current = tuple(op.params["size"])
    if stage.noise is not None:
        op = _plan_noise(stage.noise, rng)
        if op is not None:
            planned.append(op)
    if stage.jpeg is not None:
        op = _plan_jpeg(stage.jpeg, rng)
        if op is not None:
            planned.append(op)
    for op in planned:
        op.stage = stage_name
    return planned, current


def _plan_final(
    cfg: FinalBlockConfig,
    rng: np.random.Generator,
    current: tuple[int, int],
    target: tuple[int, int],
    res_factor: float,
    max_kernel: int,
) -> list[PlannedOp]:
    planned: list[PlannedOp] = []
    sinc: PlannedOp | None = None
    if rng.uniform() < cfg.sinc_prob:
        base_size = int(rng.choice(np.asarray(cfg.kernel_range)))
        kernel_size = scale_kernel_size(base_size, res_factor, max_kernel)
        kernel, info = random_sinc_kernel(rng, kernel_size, cfg.sinc_cutoff_range)  # [K,K]
        sinc = PlannedOp("blur", info, kernel)
    mode = str(rng.choice(np.asarray(cfg.modes)))
    final_resize = PlannedOp(  # nothing is sampled for the final resize; factor_effective keeps the schema uniform
        "resize",
        {
            "updown": "final",
            "factor": None,
            "factor_effective": target[0] / current[0],
            "size": list(target),
            "mode": mode,
        },
    )
    jpeg_op = _plan_jpeg(cfg.jpeg, rng)
    if rng.uniform() < 0.5:
        planned.append(final_resize)
        if sinc is not None:
            planned.append(sinc)
        if jpeg_op is not None:
            planned.append(jpeg_op)
    else:
        if jpeg_op is not None:
            planned.append(jpeg_op)
        planned.append(final_resize)
        if sinc is not None:
            planned.append(sinc)
    for op in planned:
        op.stage = "final"
    return planned


def _stage2_gate(seed: int) -> float:
    """Uniform in [0, 1) for the optional second stage, on a stream separate from the plan's."""
    return float(np.random.default_rng([seed, 2]).uniform())


def plan_degradation(
    profile: str | Profile,
    hr_size: tuple[int, int],
    scale: float = 2.0,
    seed: int = 0,
    target_size: tuple[int, int] | None = None,
) -> DegradationPlan:
    """Sample all clip-level parameters for ``profile`` on a clip of spatial size ``hr_size``."""
    prof = get_profile(profile)
    hr_size = (int(hr_size[0]), int(hr_size[1]))
    lr_size = _lr_size(hr_size, scale, target_size)
    rng = np.random.default_rng(seed)
    planned: list[PlannedOp] = []

    if isinstance(prof, CleanResizeProfile):
        planned.append(PlannedOp("resize_clean", {"size": list(lr_size), "kernel": prof.kernel}, stage="final"))
        codec_op = None if prof.codec is None else _plan_codec(prof.codec, rng)
        return DegradationPlan(prof.name, seed, scale, hr_size, lr_size, planned, codec=codec_op)

    assert isinstance(prof, RealESRGANProfile)
    res_factor = 1.0
    if prof.scale_kernels_with_resolution:
        res_factor = max(hr_size) / prof.reference_longest_side
        if prof.min_resolution_factor is not None:
            res_factor = max(res_factor, prof.min_resolution_factor)
        if prof.max_resolution_factor is not None:
            res_factor = min(res_factor, prof.max_resolution_factor)
    floor = (
        max(1, int(round(lr_size[0] * prof.min_intermediate_scale))),
        max(1, int(round(lr_size[1] * prof.min_intermediate_scale))),
    )
    current = hr_size
    stage_ops, current = _plan_stage(
        prof.stage1, rng, current, lr_size, floor, res_factor, prof.max_kernel_size, "stage1"
    )
    planned.extend(stage_ops)
    # The stage-2 gate draws from its own stream, so profiles with stage2_prob = 1.0 keep their pre-existing
    # seeded plans and stage 1 never depends on the gate. Stage 2's own draws come from the main stream, so on
    # seeds where the gate skips it the final block and codec are re-rolled.
    if prof.stage2 is not None and _stage2_gate(seed) < prof.stage2_prob:
        stage_ops, current = _plan_stage(
            prof.stage2, rng, current, lr_size, floor, res_factor, prof.max_kernel_size, "stage2"
        )
        planned.extend(stage_ops)
    planned.extend(_plan_final(prof.final, rng, current, lr_size, res_factor, prof.max_kernel_size))
    codec_op = None if prof.codec is None else _plan_codec(prof.codec, rng)
    return DegradationPlan(prof.name, seed, scale, hr_size, lr_size, planned, codec=codec_op)


def _plan_codec(cfg: CodecConfig, rng: np.random.Generator) -> PlannedOp | None:
    if rng.uniform() >= cfg.prob:
        return None
    prob = np.asarray(cfg.codec_prob, dtype=np.float64)  # [N]
    codec = str(rng.choice(np.asarray(cfg.codecs), p=prob / prob.sum()))
    crf = float(rng.uniform(*cfg.crf_range))
    preset = str(rng.choice(np.asarray(cfg.presets)))
    return PlannedOp("codec", {"codec": codec, "crf": crf, "preset": preset}, stage="codec")


def apply_plan(
    frames: torch.Tensor,
    plan: DegradationPlan,
    gen: torch.Generator,
    jpeger: DiffJPEG,
    jpeg_backend: str = "auto",
    poisson_mode: str = "auto",
) -> torch.Tensor:  # frames: [T,C,H,W] float in [0,1]; returns [T,C,h,w] float in [0,1]
    """Run every planned op on one chunk of frames."""
    x = frames
    for op in plan.ops:
        if op.op == "blur":
            assert op.kernel is not None
            x = ops.blur(x, op.kernel)  # [T,C,H,W]
        elif op.op == "resize":
            x = ops.resize(x, tuple(op.params["size"]), op.params["mode"])  # [T,C,h,w]
        elif op.op == "resize_clean":
            x = ops.resize_clean(x, tuple(op.params["size"]), op.params["kernel"])  # [T,C,h,w]
        elif op.op == "gaussian_noise":
            x = ops.add_gaussian_noise(x, op.params["sigma"], op.params["gray"], gen)  # [T,C,h,w]
        elif op.op == "poisson_noise":
            x = ops.add_poisson_noise(x, op.params["scale"], op.params["gray"], gen, mode=poisson_mode)  # [T,C,h,w]
        elif op.op == "jpeg":
            x = ops.jpeg(x, op.params["quality"], jpeger, backend=jpeg_backend)  # [T,C,h,w]
        else:
            raise ValueError(f"Unknown planned op {op.op!r}")
    if tuple(x.shape[-2:]) != plan.lr_size:
        raise RuntimeError(f"Plan ended at size {tuple(x.shape[-2:])}, expected {plan.lr_size}")
    return x


def _to_tchw(hr: torch.Tensor) -> tuple[torch.Tensor, bool]:  # returns ([T,C,H,W] uint8 or float in [0,1], is_image)
    """Reorder to time-major without changing dtype; float conversion happens per chunk to bound memory."""
    if hr.dim() == 3:
        hr = hr.unsqueeze(1)  # [C,1,H,W]
        is_image = True
    elif hr.dim() == 4:
        is_image = False
    else:
        raise ValueError(f"Expected [C,T,H,W] or [C,H,W], got shape {tuple(hr.shape)}")
    x = hr.permute(1, 0, 2, 3)  # [T,C,H,W]
    if x.is_floating_point():
        if x.numel() > 0 and x.min() < 0.0:
            raise ValueError("Float input must be in [0, 1]; got negative values (is it normalised to [-1, 1]?)")
    elif x.dtype != torch.uint8:
        raise TypeError(f"Unsupported dtype {x.dtype}")
    return x, is_image


def _chunk_to_float(chunk: torch.Tensor) -> torch.Tensor:  # chunk: [t,C,H,W] uint8 or float, returns float32 in [0,1]
    if chunk.dtype == torch.uint8:
        return chunk.float() / 255.0  # [t,C,H,W]
    return chunk.float()  # [t,C,H,W]


def degrade_hr_to_lr(
    hr: torch.Tensor,
    profile: str | Profile,
    scale: float = 2.0,
    seed: int = 0,
    target_size: tuple[int, int] | None = None,
    chunk_frames: int = 8,
    jpeger: DiffJPEG | None = None,
    jpeg_backend: str = "auto",
    poisson_mode: str = "auto",
    fps: float = 24.0,
) -> DegradationResult:
    """Degrade an HR clip or image into LR with a fully seeded, clip-consistent parameter set.

    Args:
        hr: ``[C,T,H,W]`` video or ``[C,H,W]`` image, uint8 or float in [0, 1], any device.
        profile: profile name from ``PROFILES`` or a profile dataclass.
        scale: HR-to-LR downscale factor; LR is ``round(H/scale) x round(W/scale)`` unless ``target_size``.
        seed: seeds both parameter sampling and noise realisation.
        target_size: explicit LR ``(h, w)``; overrides ``scale`` for the output size.
        chunk_frames: frames processed per step; bounds peak memory (float32 intermediates).
        jpeger: optional reusable ``DiffJPEG`` module (avoids re-creating buffers per call).
        jpeg_backend: ``"auto"`` (libjpeg via cv2 on CPU, DiffJPEG on GPU), ``"cv2"`` or ``"diffjpeg"``.
        poisson_mode: ``"auto"`` (exact on GPU, Gaussian approximation on CPU), ``"exact"`` or
            ``"gaussian_approx"``. Exact Poisson sampling costs about 0.3 s per 1080p frame on CPU.
        fps: frame rate of the clip, used only by the codec stage (P3) as the encoder's stream rate, which
            feeds x264/x265 rate control. Pass the sample's real fps; ignored for images and codec-free profiles.

    Returns:
        ``DegradationResult`` with ``lr`` as uint8 in the input layout and the parameter ``record``.
        The record also states the resolved ``jpeg_backend`` and ``poisson_mode`` and the device.
    """
    x, is_image = _to_tchw(hr)  # [T,C,H,W], input dtype
    plan = plan_degradation(profile, tuple(x.shape[-2:]), scale=scale, seed=seed, target_size=target_size)
    codec_skipped = None
    if plan.codec is not None and (is_image or x.shape[0] < 2):
        # A video codec needs a clip; on an image or a one-frame clip the op cannot run, so it leaves the plan
        # (the record must not list an op that never happened) and the record says why.
        codec_skipped = "single_frame"
        plan = dataclasses.replace(plan, codec=None)
    gen = ops.make_generator(seed, x.device)
    resolved_jpeg = ops.resolve_jpeg_backend(jpeg_backend, x.device)
    resolved_poisson = ops.resolve_poisson_mode(poisson_mode, x.device)
    if jpeger is None:
        jpeger = DiffJPEG(differentiable=False)
    chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, x.shape[0], max(1, chunk_frames)):
            chunk = _chunk_to_float(x[start : start + chunk_frames])  # [t,C,H,W] float32
            out = apply_plan(chunk, plan, gen, jpeger, jpeg_backend=resolved_jpeg, poisson_mode=resolved_poisson)
            chunks.append(ops.to_uint8(out))  # [t,C,h,w]
        lr_tchw = torch.cat(chunks, dim=0)  # [T,C,h,w] uint8
        codec_applied = plan.codec is not None  # single-frame inputs had the codec removed from the plan above
        if plan.codec is not None:
            # Whole-clip op: needs every frame at once, runs on CPU (software encoders), returns to device.
            params = plan.codec.params
            lr_tchw = codec_round_trip(
                lr_tchw, codec=params["codec"], crf=params["crf"], preset=params["preset"], fps=fps
            )
    lr = lr_tchw.permute(1, 0, 2, 3).contiguous()  # [C,T,h,w]
    if is_image:
        lr = lr[:, 0]  # [C,h,w]
    record = plan.record()
    record.update(
        {
            "jpeg_backend": resolved_jpeg,
            "poisson_mode": resolved_poisson,
            "device": x.device.type,
            "codec_applied": codec_applied,
            "codec_skipped": codec_skipped,
            "codec_fps": float(fps) if codec_applied else None,
        }
    )
    return DegradationResult(lr=lr, record=record)


def degrade_batch(
    hr_list: Sequence[torch.Tensor], profile: str | Profile, scale: float, seeds: Sequence[int], **kwargs: Any
) -> list[DegradationResult]:
    """Convenience wrapper for a list of clips with one seed each."""
    if len(hr_list) != len(seeds):
        raise ValueError("hr_list and seeds must have equal length")
    jpeger = kwargs.pop("jpeger", None) or DiffJPEG(differentiable=False)
    return [
        degrade_hr_to_lr(hr, profile, scale=scale, seed=s, jpeger=jpeger, **kwargs) for hr, s in zip(hr_list, seeds)
    ]
