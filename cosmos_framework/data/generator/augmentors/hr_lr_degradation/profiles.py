# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Degradation profile definitions.

A profile is a plain dataclass tree so it can be built from a LazyCall config, copied with
``dataclasses.replace`` for ablation arms, and serialised into the degradation record.

Numeric defaults follow the Cosmos Transfer1 corruptor configs
(``projects/cosmos/diffusion/v1/config/base/corruptor.py`` on branch
``qianlim/transfer1_upscaler_gbuffer``), which in turn follow Real-ESRGAN
``options/train_realesrnet_x2plus.yml``. The noise stage, which Transfer1 left out, uses
the Real-ESRGAN x2plus values.

Two families are registered in ``PROFILES``: the inherited ``p0_*`` / ``p1_*`` / ``p3_*`` profiles with those
Real-ESRGAN ranges, and the ``img_*`` / ``vid_*`` regime ladder (clean / mild / moderate / harsh) budgeted to the
x2 task, drawn per sample through ``IMAGE_SR_DEFAULT_MIX`` and ``VIDEO_SR_DEFAULT_MIX``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Sequence

RESIZE_MODES = ("area", "bilinear", "bicubic")
CLEAN_RESIZE_KERNELS = ("bicubic_antialias", "bilinear_antialias", "area")

# Real-ESRGAN kernel sizes are odd values from 7 to 21 and were tuned for roughly 400 px crops.
_DEFAULT_KERNEL_RANGE = tuple(2 * v + 1 for v in range(3, 11))
_DEFAULT_KERNEL_LIST = ("iso", "aniso", "generalized_iso", "generalized_aniso", "plateau_iso", "plateau_aniso")
_DEFAULT_KERNEL_PROB = (0.45, 0.25, 0.12, 0.03, 0.12, 0.03)


def _check_cutoff_range(cutoff: Sequence[float]) -> None:
    """Sinc cutoffs are angular frequencies: 0 < low <= high <= pi (omega_c = 0 is an all-NaN kernel)."""
    try:
        lo, hi = (float(v) for v in cutoff)
    except (TypeError, ValueError):
        raise ValueError(f"sinc_cutoff_range must be a (low, high) pair, got {cutoff!r}") from None
    if not 0.0 < lo <= hi <= math.pi:
        raise ValueError(f"sinc_cutoff_range must satisfy 0 < low <= high <= pi, got {(lo, hi)}")


@dataclass(frozen=True)
class BlurConfig:
    """Mixed-kernel or sinc blur applied once per clip."""

    prob: float = 1.0
    kernel_range: Sequence[int] = _DEFAULT_KERNEL_RANGE
    sigma_range: Sequence[float] = (0.2, 3.0)
    sinc_prob: float = 0.1
    sinc_cutoff_range: Sequence[float] | None = None  # omega_c bounds; None = Real-ESRGAN's size-dependent prior
    kernel_list: Sequence[str] = _DEFAULT_KERNEL_LIST
    kernel_prob: Sequence[float] = _DEFAULT_KERNEL_PROB
    betag_range: Sequence[float] = (0.5, 4.0)
    betap_range: Sequence[float] = (1.0, 2.0)

    def __post_init__(self) -> None:
        if self.sinc_cutoff_range is not None:
            _check_cutoff_range(self.sinc_cutoff_range)


@dataclass(frozen=True)
class ResizeConfig:
    """Random up / down / keep resize.

    ``relative_to`` selects the reference size the factor multiplies: ``"current"`` (stage 1,
    Real-ESRGAN first resize) or ``"target"`` (stage 2, Real-ESRGAN second resize is relative to
    the final LR size).
    """

    prob: float = 1.0
    updown_prob: Sequence[float] = (0.2, 0.7, 0.1)
    scale_range: Sequence[float] = (0.15, 1.5)
    modes: Sequence[str] = RESIZE_MODES
    relative_to: str = "current"


@dataclass(frozen=True)
class NoiseConfig:
    """Gaussian or Poisson noise, optionally grey (shared across channels)."""

    prob: float = 1.0
    gaussian_prob: float = 0.5
    gaussian_sigma_range: Sequence[float] = (1.0, 30.0)
    poisson_scale_range: Sequence[float] = (0.05, 3.0)
    gray_noise_prob: float = 0.4


@dataclass(frozen=True)
class JPEGConfig:
    prob: float = 1.0
    quality_range: Sequence[float] = (30.0, 95.0)


@dataclass(frozen=True)
class FinalBlockConfig:
    """Resize to the exact LR size, then sinc filter and JPEG in random order (Real-ESRGAN final block)."""

    sinc_prob: float = 0.8
    kernel_range: Sequence[int] = _DEFAULT_KERNEL_RANGE
    sinc_cutoff_range: Sequence[float] | None = (math.pi / 3, math.pi)  # None = Real-ESRGAN's size-dependent prior
    modes: Sequence[str] = RESIZE_MODES
    jpeg: JPEGConfig = field(default_factory=JPEGConfig)

    def __post_init__(self) -> None:
        if self.sinc_cutoff_range is not None:
            _check_cutoff_range(self.sinc_cutoff_range)


@dataclass(frozen=True)
class CodecConfig:
    """Whole-clip video codec round trip on the final LR (P3). Skipped for single images.

    ``crf_range`` follows RealBasicVSR / Upscale-A-Video (18 to 35). ODVista-style streaming at fixed
    low bitrates is harsher; widen the upper end for that benchmark. ``presets`` are x264 names; NVENC
    encoders (``h264_nvenc`` / ``hevc_nvenc``) map them to ``p1``..``p7`` and CRF to ``cq`` (see ``codec.py``).
    """

    prob: float = 0.6
    codecs: Sequence[str] = ("libx264", "libx265")
    codec_prob: Sequence[float] = (0.7, 0.3)
    crf_range: Sequence[float] = (18.0, 35.0)
    presets: Sequence[str] = ("veryfast", "medium")


@dataclass(frozen=True)
class DegradationStage:
    blur: BlurConfig | None = field(default_factory=BlurConfig)
    resize: ResizeConfig | None = field(default_factory=ResizeConfig)
    noise: NoiseConfig | None = field(default_factory=NoiseConfig)
    jpeg: JPEGConfig | None = field(default_factory=JPEGConfig)


@dataclass(frozen=True)
class RealESRGANProfile:
    """Real-ESRGAN style pipeline with one or two stages and a final block.

    Attributes:
        reference_longest_side: kernel sizes and sigmas are scaled by ``longest_side / reference``
            so published ranges tuned near 400 to 720 px stay meaningful at 1080p and above.
        min_resolution_factor / max_resolution_factor: clamps on that scale factor. ``[1.0, 1.5]`` keeps inputs
            below the reference at the base ranges and stops proportional growth past 1.5x (unbounded scaling
            produced 61 px kernels at 1080p). ``None`` leaves that side unbounded, as in the inherited profiles.
        stage2_prob: probability of applying ``stage2`` when defined (1.0 = always, Real-ESRGAN style).
        max_kernel_size: cap for the scaled kernel size (odd).
        min_intermediate_scale: intermediate frames never shrink below this fraction of the
            target LR size, which keeps a x2 task from becoming a x4 to x6 task.
    """

    name: str = "p1_first_order"
    stage1: DegradationStage = field(default_factory=DegradationStage)
    stage2: DegradationStage | None = None
    final: FinalBlockConfig = field(default_factory=FinalBlockConfig)
    codec: CodecConfig | None = None
    modality: str = "any"  # "image" / "video" / "any": AddLowRes rejects a profile built for the other stream
    stage2_prob: float = 1.0  # probability of running stage2 when it is defined
    scale_kernels_with_resolution: bool = True
    reference_longest_side: int = 720
    min_resolution_factor: float | None = None  # lower clamp on longest_side / reference; None = unbounded
    max_resolution_factor: float | None = None  # upper clamp on longest_side / reference; None = unbounded
    max_kernel_size: int = 61
    min_intermediate_scale: float = 0.75

    def __post_init__(self) -> None:
        if not 0.0 <= self.stage2_prob <= 1.0:
            raise ValueError(f"stage2_prob must be in [0, 1], got {self.stage2_prob}")
        if self.stage2 is None and self.stage2_prob != 1.0:
            raise ValueError(f"stage2_prob={self.stage2_prob} has no effect: the profile defines no stage2")
        if not 0.0 <= self.min_intermediate_scale <= 1.0:  # 0 = no floor (published Real-ESRGAN ranges)
            raise ValueError(f"min_intermediate_scale must be in [0, 1], got {self.min_intermediate_scale}")
        if self.reference_longest_side <= 0:
            raise ValueError(f"reference_longest_side must be positive, got {self.reference_longest_side}")
        lo, hi = self.min_resolution_factor, self.max_resolution_factor
        for label, value in (("min_resolution_factor", lo), ("max_resolution_factor", hi)):
            if value is not None and value <= 0:
                raise ValueError(f"{label} must be positive, got {value}")
        if lo is not None and hi is not None and lo > hi:
            raise ValueError(f"min_resolution_factor {lo} exceeds max_resolution_factor {hi}")


@dataclass(frozen=True)
class CleanResizeProfile:
    """P0: deterministic antialiased resize to the exact LR size, no other degradation."""

    name: str = "p0_clean_bicubic"
    kernel: str = "bicubic_antialias"
    codec: CodecConfig | None = None  # optional whole-clip re-encode on the clean LR (video only)
    modality: str = "any"  # "image" / "video" / "any": AddLowRes rejects a profile built for the other stream


Profile = RealESRGANProfile | CleanResizeProfile


def _second_order_stage2() -> DegradationStage:
    return DegradationStage(
        blur=BlurConfig(prob=0.8),
        resize=ResizeConfig(updown_prob=(0.3, 0.4, 0.3), scale_range=(0.3, 1.2), relative_to="target"),
        noise=NoiseConfig(gaussian_sigma_range=(1.0, 25.0), poisson_scale_range=(0.05, 2.5)),
        jpeg=JPEGConfig(),
    )


PROFILES: dict[str, Profile] = {
    "p0_clean_bicubic": CleanResizeProfile(name="p0_clean_bicubic", kernel="bicubic_antialias"),
    "p0_clean_area": CleanResizeProfile(name="p0_clean_area", kernel="area"),
    "p1_first_order": RealESRGANProfile(name="p1_first_order"),
    "p1_first_order_no_noise": RealESRGANProfile(name="p1_first_order_no_noise", stage1=DegradationStage(noise=None)),
    "p1_second_order": RealESRGANProfile(name="p1_second_order", stage2=_second_order_stage2()),
    # P3: video terms. First-order pixel pipeline plus an H.264 / H.265 round trip on the LR clip.
    "p3_video_codec": RealESRGANProfile(name="p3_video_codec", codec=CodecConfig()),
    "p3_video_codec_second_order": RealESRGANProfile(
        name="p3_video_codec_second_order", stage2=_second_order_stage2(), codec=CodecConfig()
    ),
    # Published Real-ESRGAN ranges without resolution scaling, for the calibration ablation (E2).
    "p1_second_order_published": RealESRGANProfile(
        name="p1_second_order_published",
        stage2=_second_order_stage2(),
        scale_kernels_with_resolution=False,
        min_intermediate_scale=0.0,
    ),
}


# ---------------------------------------------------------------------------------------------------------------
# Regime profiles for x2 SR: a clean / mild / moderate / harsh ladder per modality, drawn per sample through the
# mixes at the bottom. Ranges are budgeted to the x2 task (the LR already loses 4x the pixels), so extra blur stays
# within about one LR pixel except in the harsh tail. Sigma values are HR pixels at 720p HR and scale with
# resolution up to 1.5x; noise sigma is in 8-bit units. Images are JPEG-first: the JPEG sits in the final block, so
# in Real-ESRGAN's random order it lands on the LR grid half the time and just before the final resize otherwise.
# Video is codec-first (no JPEG under the codec except the rare "re-saved frames" term). Each regime declares its
# modality so AddLowRes can reject a mix handed to the wrong stream.
# ---------------------------------------------------------------------------------------------------------------
_REGIME_KERNEL_PROB = (0.50, 0.30, 0.07, 0.03, 0.07, 0.03)  # iso / aniso / gen-iso / gen-aniso / plateau-iso / -aniso
# Regime ranges are specified at 720p HR (longest side 1280): r = clamp(longest / 1280, 1.0, 1.5), so 1080p HR
# gets x1.5 and QHD / 4K stay at x1.5, while anything below 720p keeps the 720p ranges.
_REGIME_COMMON = dict(
    reference_longest_side=1280,
    min_resolution_factor=1.0,
    max_resolution_factor=1.5,
    max_kernel_size=41,
    min_intermediate_scale=0.75,
)
_IMG = dict(modality="image", **_REGIME_COMMON)
_VID = dict(modality="video", **_REGIME_COMMON)
_NO_FINAL_JPEG = JPEGConfig(prob=0.0)


def _down(prob: float, low: float) -> ResizeConfig:
    """Intermediate downscale to [low, 1.0] of the LR size (never the Real-ESRGAN 'up' branch). ``low`` stays at or
    above ``min_intermediate_scale``: below it the floor would turn the tail of the range into a point mass."""
    return ResizeConfig(prob=prob, updown_prob=(0.0, 1.0, 0.0), scale_range=(low, 1.0), relative_to="target")


def _noise(
    prob: float, gauss: tuple[float, float], poisson: tuple[float, float] | None, gray: float = 0.3
) -> NoiseConfig:
    if poisson is None:
        return NoiseConfig(prob=prob, gaussian_prob=1.0, gaussian_sigma_range=gauss, gray_noise_prob=gray)
    return NoiseConfig(
        prob=prob, gaussian_prob=0.5, gaussian_sigma_range=gauss, poisson_scale_range=poisson, gray_noise_prob=gray
    )


def _blur(
    prob: float, sigma: tuple[float, float], sinc_prob: float, cutoff: tuple[float, float] = (math.pi / 3, math.pi)
) -> BlurConfig:
    return BlurConfig(
        prob=prob, sigma_range=sigma, sinc_prob=sinc_prob, sinc_cutoff_range=cutoff, kernel_prob=_REGIME_KERNEL_PROB
    )


def _final(
    sinc_prob: float, cutoff: tuple[float, float] = (math.pi / 3, math.pi), jpeg: JPEGConfig = _NO_FINAL_JPEG
) -> FinalBlockConfig:
    """Final block: resize to LR plus sinc / JPEG in Real-ESRGAN's random order, so an image regime's JPEG lands on
    the LR grid half the time (a photo saved at its own resolution) and just before the final resize otherwise.
    Video regimes leave the JPEG off because the codec is the compression term."""
    return FinalBlockConfig(sinc_prob=sinc_prob, sinc_cutoff_range=cutoff, jpeg=jpeg)


# Video regimes keep the CodecConfig defaults for codecs (H.264 0.7 / H.265 0.3) and presets (veryfast / medium).
REGIME_PROFILES: dict[str, Profile] = {
    # ---- images: JPEG-first
    "img_clean": CleanResizeProfile(name="img_clean", kernel="bicubic_antialias", modality="image"),
    "img_mild": RealESRGANProfile(
        name="img_mild",
        stage1=DegradationStage(
            blur=_blur(0.8, (0.2, 1.0), 0.05, cutoff=(math.pi / 2, math.pi)),
            resize=_down(0.5, 0.85),
            noise=_noise(0.6, (1.0, 6.0), (0.05, 0.8)),
            jpeg=None,
        ),
        final=_final(0.2, (math.pi / 2, math.pi), jpeg=JPEGConfig(prob=0.7, quality_range=(70.0, 95.0))),
        **_IMG,
    ),
    "img_moderate": RealESRGANProfile(
        name="img_moderate",
        stage1=DegradationStage(
            blur=_blur(1.0, (0.5, 2.0), 0.1),
            resize=_down(0.7, 0.75),
            noise=_noise(0.8, (3.0, 12.0), (0.3, 1.5)),
            jpeg=None,
        ),
        stage2=DegradationStage(  # an earlier generation: re-saved at some intermediate size, then re-processed
            blur=_blur(1.0, (0.25, 1.0), 0.05),
            resize=None,
            noise=_noise(0.8, (1.5, 6.0), (0.15, 0.75)),
            jpeg=JPEGConfig(prob=0.9, quality_range=(60.0, 90.0)),
        ),
        stage2_prob=0.3,
        final=_final(0.4, jpeg=JPEGConfig(prob=0.9, quality_range=(45.0, 80.0))),
        **_IMG,
    ),
    "img_harsh": RealESRGANProfile(
        name="img_harsh",
        stage1=DegradationStage(
            blur=_blur(1.0, (1.0, 3.0), 0.1),
            resize=_down(0.9, 0.75),
            noise=_noise(1.0, (5.0, 20.0), (1.0, 3.0)),
            jpeg=None,
        ),
        final=_final(0.5, jpeg=JPEGConfig(prob=1.0, quality_range=(30.0, 60.0))),
        **_IMG,
    ),
    # ---- video: codec-first
    "vid_clean": CleanResizeProfile(
        name="vid_clean",
        kernel="bicubic_antialias",
        modality="video",
        codec=CodecConfig(
            prob=0.3, codecs=("libx264",), codec_prob=(1.0,), crf_range=(16.0, 20.0), presets=("medium",)
        ),
    ),
    "vid_mild": RealESRGANProfile(
        name="vid_mild",
        stage1=DegradationStage(
            blur=_blur(0.8, (0.2, 1.0), 0.05),
            resize=_down(0.4, 0.85),
            noise=_noise(0.5, (1.0, 5.0), (0.05, 0.6)),
            jpeg=None,
        ),
        final=_final(0.1),
        codec=CodecConfig(prob=0.9, crf_range=(20.0, 28.0)),
        **_VID,
    ),
    "vid_moderate": RealESRGANProfile(
        name="vid_moderate",
        stage1=DegradationStage(
            blur=_blur(1.0, (0.5, 1.8), 0.1),
            resize=_down(0.6, 0.75),
            noise=_noise(0.7, (2.0, 10.0), (0.3, 1.2)),
            jpeg=JPEGConfig(prob=0.2, quality_range=(60.0, 90.0)),  # frames re-saved before re-encoding
        ),
        final=_final(0.2),
        codec=CodecConfig(prob=1.0, crf_range=(26.0, 34.0)),
        **_VID,
    ),
    "vid_harsh": RealESRGANProfile(
        name="vid_harsh",
        stage1=DegradationStage(
            blur=_blur(1.0, (1.0, 2.5), 0.1),
            resize=_down(0.9, 0.75),
            noise=_noise(1.0, (5.0, 15.0), None),
            jpeg=None,
        ),
        final=_final(0.3),
        codec=CodecConfig(prob=1.0, crf_range=(32.0, 40.0), presets=("veryfast",)),  # low-bitrate streaming look
        **_VID,
    ),
}
PROFILES.update(REGIME_PROFILES)

# Regime mixtures for AddLowRes ``profiles=`` / dataset ``sr_profiles=``. The weights are the calibration knob
# (against real 720p inventory and Cosmos 720p outputs); the per-regime ranges are fixed for physical plausibility.
IMAGE_SR_DEFAULT_MIX: dict[str, float] = {"img_clean": 0.30, "img_mild": 0.45, "img_moderate": 0.20, "img_harsh": 0.05}
VIDEO_SR_DEFAULT_MIX: dict[str, float] = {"vid_clean": 0.30, "vid_mild": 0.40, "vid_moderate": 0.25, "vid_harsh": 0.05}


def get_profile(name_or_profile: str | Profile) -> Profile:
    if isinstance(name_or_profile, (RealESRGANProfile, CleanResizeProfile)):
        return name_or_profile
    if name_or_profile not in PROFILES:
        raise KeyError(f"Unknown degradation profile {name_or_profile!r}; known: {sorted(PROFILES)}")
    return PROFILES[name_or_profile]


def profile_to_dict(profile: Any) -> Any:
    """Recursively convert a profile dataclass tree into JSON-serialisable primitives."""
    if is_dataclass(profile) and not isinstance(profile, type):
        return {f.name: profile_to_dict(getattr(profile, f.name)) for f in fields(profile)}
    if isinstance(profile, (list, tuple)):
        return [profile_to_dict(v) for v in profile]
    return profile
