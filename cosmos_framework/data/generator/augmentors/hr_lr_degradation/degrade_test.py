# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import dataclasses
import json
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from cosmos_framework.data.generator.augmentors.hr_lr_degradation import (
    PROFILES,
    RealESRGANProfile,
    degrade_hr_to_lr,
    get_profile,
    ops,
)
from cosmos_framework.data.generator.augmentors.hr_lr_degradation.degrade import plan_degradation
from cosmos_framework.data.generator.augmentors.hr_lr_degradation.diffjpeg import DiffJPEG
from cosmos_framework.data.generator.augmentors.hr_lr_degradation.kernels import (
    circular_lowpass_kernel,
    random_mixed_kernel,
    scale_kernel_size,
)
from cosmos_framework.data.generator.augmentors.hr_lr_degradation.profiles import (
    IMAGE_SR_DEFAULT_MIX,
    REGIME_PROFILES,
    VIDEO_SR_DEFAULT_MIX,
    BlurConfig,
    DegradationStage,
    FinalBlockConfig,
    JPEGConfig,
    Profile,
    profile_to_dict,
)

pytestmark = [pytest.mark.L0, pytest.mark.CPU]

_ALL_PROFILES = sorted(PROFILES)


def _synthetic_clip(t: int = 5, h: int = 96, w: int = 128, seed: int = 0) -> torch.Tensor:  # returns [3,T,H,W] uint8
    """Smooth gradients plus a moving edge so blur, resize and JPEG all have something to act on."""
    g = torch.Generator().manual_seed(seed)
    ys = torch.linspace(0, 1, h).view(1, 1, h, 1)  # [1,1,H,1]
    xs = torch.linspace(0, 1, w).view(1, 1, 1, w)  # [1,1,1,W]
    ts = torch.linspace(0, 1, t).view(1, t, 1, 1)  # [1,T,1,1]
    r = ys.expand(1, t, h, w)  # [1,T,H,W]
    gch = xs.expand(1, t, h, w)  # [1,T,H,W]
    b = ((xs + 0.3 * ts) % 1.0 > 0.5).float().expand(1, t, h, w)  # [1,T,H,W]
    clip = torch.cat([r, gch, b], dim=0)  # [3,T,H,W]
    clip = clip + 0.02 * torch.rand(clip.shape, generator=g)  # [3,T,H,W]
    return (clip.clamp(0, 1) * 255).round().to(torch.uint8)  # [3,T,H,W]


@pytest.mark.parametrize("profile_name", _ALL_PROFILES)
def test_video_output_shape_dtype_and_range(profile_name: str) -> None:
    hr = _synthetic_clip(t=5, h=96, w=128)  # [3,5,96,128]
    result = degrade_hr_to_lr(hr, profile_name, scale=2, seed=123, chunk_frames=2)
    assert result.lr.shape == (3, 5, 48, 64)
    assert result.lr.dtype == torch.uint8
    assert result.record["profile_name"] == profile_name
    assert result.record["lr_size"] == [48, 64]
    json.dumps(result.record)  # record must be serialisable


@pytest.mark.parametrize("profile_name", ["p0_clean_bicubic", "p1_second_order"])
def test_image_input_keeps_layout(profile_name: str) -> None:
    hr = _synthetic_clip(t=1, h=64, w=80)[:, 0]  # [3,64,80]
    result = degrade_hr_to_lr(hr, profile_name, scale=2, seed=7)
    assert result.lr.shape == (3, 32, 40)
    assert result.lr.dtype == torch.uint8


def test_float_input_matches_uint8_input() -> None:
    hr_u8 = _synthetic_clip(t=2, h=64, w=64)  # [3,2,64,64]
    hr_f = hr_u8.float() / 255.0  # [3,2,64,64]
    out_u8 = degrade_hr_to_lr(hr_u8, "p1_second_order", seed=3).lr
    out_f = degrade_hr_to_lr(hr_f, "p1_second_order", seed=3).lr
    assert torch.equal(out_u8, out_f)


def test_normalised_float_input_is_rejected() -> None:
    hr = torch.rand(3, 2, 32, 32) * 2 - 1  # [3,2,32,32] in [-1,1]
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        degrade_hr_to_lr(hr, "p0_clean_bicubic")


@pytest.mark.parametrize("profile_name", ["p1_first_order", "p1_second_order"])
def test_same_seed_is_deterministic_and_chunking_invariant(profile_name: str) -> None:
    hr = _synthetic_clip(t=6, h=64, w=96)  # [3,6,64,96]
    a = degrade_hr_to_lr(hr, profile_name, seed=42, chunk_frames=6)
    b = degrade_hr_to_lr(hr, profile_name, seed=42, chunk_frames=6)
    assert torch.equal(a.lr, b.lr)
    assert a.record == b.record


def test_chunking_only_changes_noise_realisation_not_parameters() -> None:
    """Chunking must not change the sampled plan, and may only perturb pixels through float rounding.

    Without a noise stage the pipeline is a per-frame map, but not a bitwise-reproducible one across chunk sizes:
    the FFT convolution picks a different plan for a batch of 6 frames than for a batch of 2 (observed on aarch64
    CI runners; x86 happened to agree), which can move a pixel by one 8-bit level. A JPEG stage after the blur
    amplifies such one-level input changes into several output levels on a small fraction of pixels. So:
    compression-free plans must agree to within one level; plans with JPEG must agree statistically.
    """
    hr = _synthetic_clip(t=6, h=64, w=96)  # [3,6,64,96]

    # 1) Same sampled parameters regardless of chunking (the property the training pipeline relies on).
    a = degrade_hr_to_lr(hr, "p1_first_order_no_noise", seed=42, chunk_frames=6)
    b = degrade_hr_to_lr(hr, "p1_first_order_no_noise", seed=42, chunk_frames=2)
    assert a.record == b.record

    # 2) Blur + resize only: differences are pure float rounding, at most one 8-bit level on few pixels.
    base = get_profile("p1_first_order_no_noise")
    assert isinstance(base, RealESRGANProfile)
    no_compression = dataclasses.replace(
        base,
        name="chunking_probe_no_compression",
        stage1=DegradationStage(noise=None, jpeg=None),
        final=dataclasses.replace(base.final, jpeg=JPEGConfig(prob=0.0)),
    )
    c = degrade_hr_to_lr(hr, no_compression, seed=42, chunk_frames=6)
    d = degrade_hr_to_lr(hr, no_compression, seed=42, chunk_frames=2)
    assert c.record == d.record
    diff = (c.lr.int() - d.lr.int()).abs()
    assert diff.max() <= 1, f"chunking changed compression-free pixels by up to {int(diff.max())} levels"
    assert (diff > 0).float().mean() < 0.02, f"{(diff > 0).float().mean():.3%} of pixels differ across chunkings"

    # 3) With JPEG in the chain, a handful of pixels may move by several levels; the images stay the same picture.
    diff_jpeg = (a.lr.int() - b.lr.int()).abs()
    assert diff_jpeg.float().mean() < 0.1, f"mean abs diff {diff_jpeg.float().mean():.4f} levels across chunkings"
    assert (diff_jpeg > 0).float().mean() < 0.05, f"{(diff_jpeg > 0).float().mean():.3%} of pixels differ"


def test_different_seeds_give_different_parameters() -> None:
    plans = {plan_degradation("p1_second_order", (720, 1280), seed=s).record()["ops"].__repr__() for s in range(8)}
    assert len(plans) > 1


def test_p0_matches_torch_antialiased_bicubic_reference() -> None:
    hr = _synthetic_clip(t=3, h=96, w=128)  # [3,3,96,128]
    out = degrade_hr_to_lr(hr, "p0_clean_bicubic", scale=2, seed=0).lr  # [3,3,48,64]
    ref = F.interpolate(
        hr.permute(1, 0, 2, 3).float() / 255.0, size=(48, 64), mode="bicubic", align_corners=False, antialias=True
    )  # [3,3,48,64]
    ref_u8 = (ref.clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 0, 2, 3)  # [3,3,48,64]
    assert torch.equal(out, ref_u8)


def test_p0_is_seed_independent() -> None:
    hr = _synthetic_clip(t=2, h=64, w=64)  # [3,2,64,64]
    assert torch.equal(
        degrade_hr_to_lr(hr, "p0_clean_bicubic", seed=1).lr, degrade_hr_to_lr(hr, "p0_clean_bicubic", seed=2).lr
    )


def test_target_size_overrides_scale() -> None:
    hr = _synthetic_clip(t=2, h=90, w=160)  # [3,2,90,160]
    out = degrade_hr_to_lr(hr, "p1_first_order", scale=2, seed=0, target_size=(40, 72)).lr
    assert out.shape == (3, 2, 40, 72)


def test_plan_respects_intermediate_floor_and_ends_at_lr_size() -> None:
    profile = get_profile("p1_second_order")
    assert isinstance(profile, RealESRGANProfile)
    for seed in range(50):
        plan = plan_degradation(profile, (720, 1280), scale=2, seed=seed)
        floor_h = round(plan.lr_size[0] * profile.min_intermediate_scale)
        floor_w = round(plan.lr_size[1] * profile.min_intermediate_scale)
        sizes = [tuple(op.params["size"]) for op in plan.ops if op.op == "resize"]
        assert sizes[-1] == plan.lr_size
        for h, w in sizes:
            assert h >= floor_h and w >= floor_w
        for op in plan.ops:
            if op.op == "blur":
                assert op.kernel is not None and op.kernel.shape[0] % 2 == 1
                assert op.kernel.shape[0] <= profile.max_kernel_size
                assert abs(float(op.kernel.sum()) - 1.0) < 1e-4


def test_kernel_sizes_scale_with_resolution() -> None:
    small = [
        op.params["kernel_size"]
        for s in range(40)
        for op in plan_degradation("p1_first_order", (360, 640), seed=s).ops
        if op.op == "blur"
    ]
    large = [
        op.params["kernel_size"]
        for s in range(40)
        for op in plan_degradation("p1_first_order", (1080, 1920), seed=s).ops
        if op.op == "blur"
    ]
    assert np.mean(large) > np.mean(small)
    published = [
        op.params["kernel_size"]
        for s in range(40)
        for op in plan_degradation("p1_second_order_published", (1080, 1920), seed=s).ops
        if op.op == "blur"
    ]
    assert max(published) <= 21


def test_scale_kernel_size_is_odd_and_bounded() -> None:
    for base in (7, 9, 21):
        for factor in (0.3, 1.0, 2.67, 10.0):
            k = scale_kernel_size(base, factor, 61)
            assert k % 2 == 1 and 3 <= k <= 61


def test_mixed_and_sinc_kernels_are_normalised() -> None:
    rng = np.random.default_rng(0)
    for _ in range(20):
        kernel, info = random_mixed_kernel(
            rng,
            get_profile("p1_first_order").stage1.blur.kernel_list,
            get_profile("p1_first_order").stage1.blur.kernel_prob,
            13,
            (0.2, 3),
            (0.2, 3),
        )
        assert kernel.shape == (13, 13) and abs(kernel.sum() - 1) < 1e-6 and info["kernel_type"]
    sinc = circular_lowpass_kernel(np.pi / 2, 11, pad_to=21)
    assert sinc.shape == (21, 21) and abs(sinc.sum() - 1) < 1e-6


def test_diffjpeg_degrades_more_at_lower_quality_and_handles_odd_sizes() -> None:
    jpeger = DiffJPEG()
    x = _synthetic_clip(t=2, h=45, w=67).permute(1, 0, 2, 3).float() / 255.0  # [2,3,45,67]
    hi = jpeger(x, quality=95.0)  # [2,3,45,67]
    lo = jpeger(x, quality=10.0)  # [2,3,45,67]
    assert hi.shape == x.shape and lo.shape == x.shape
    assert (hi - x).abs().mean() < (lo - x).abs().mean()
    assert (hi - x).abs().mean() < 0.02
    per_frame = jpeger(x, quality=torch.tensor([95.0, 10.0]))  # [2,3,45,67]
    assert torch.allclose(per_frame[0], hi[0]) and torch.allclose(per_frame[1], lo[1])


def test_degradation_actually_changes_content_relative_to_clean() -> None:
    hr = _synthetic_clip(t=2, h=96, w=128)  # [3,2,96,128]
    clean = degrade_hr_to_lr(hr, "p0_clean_bicubic").lr.float()
    degraded = degrade_hr_to_lr(hr, "p1_second_order", seed=5).lr.float()
    assert (clean - degraded).abs().mean() > 0.5  # 8-bit units


def test_profiles_are_frozen_copyable_and_serialisable() -> None:
    base = get_profile("p1_first_order")
    arm = dataclasses.replace(base, name="arm", stage1=DegradationStage(noise=None))
    assert arm.stage1.noise is None and base.stage1.noise is not None
    json.dumps(profile_to_dict(arm))
    with pytest.raises(KeyError):
        get_profile("does_not_exist")


def test_fft_filter_matches_direct_convolution() -> None:
    x = _synthetic_clip(t=2, h=64, w=80).permute(1, 0, 2, 3).float() / 255.0  # [2,3,64,80]
    rng = np.random.default_rng(1)
    for k in (11, 21, 33):
        kernel = torch.from_numpy(rng.random((k, k)).astype(np.float32))  # asymmetric on purpose
        kernel = kernel / kernel.sum()
        direct = ops._filter2d_direct(x, kernel)  # [2,3,64,80]
        via_fft = ops._filter2d_fft(x, kernel)  # [2,3,64,80]
        assert torch.allclose(direct, via_fft, atol=1e-5), f"k={k}: max err {(direct - via_fft).abs().max()}"


def test_cv2_and_diffjpeg_backends_behave_alike() -> None:
    x = _synthetic_clip(t=2, h=45, w=67).permute(1, 0, 2, 3).float() / 255.0  # [2,3,45,67]
    jpeger = DiffJPEG()
    for quality in (90.0, 20.0):
        via_cv2 = ops.jpeg(x, quality, jpeger, backend="cv2")  # [2,3,45,67]
        via_torch = ops.jpeg(x, quality, jpeger, backend="diffjpeg")  # [2,3,45,67]
        assert via_cv2.shape == x.shape and via_torch.shape == x.shape
        # Both are lossy codecs of the same picture: they should agree with each other about as well as with the input.
        assert (via_cv2 - via_torch).abs().mean() < 2.0 * max((via_cv2 - x).abs().mean(), (via_torch - x).abs().mean())
    err_hi = (ops.jpeg(x, 90.0, jpeger, backend="cv2") - x).abs().mean()
    err_lo = (ops.jpeg(x, 20.0, jpeger, backend="cv2") - x).abs().mean()
    assert err_hi < err_lo
    with pytest.raises(ValueError):
        ops.resolve_jpeg_backend("nope", torch.device("cpu"))


def test_poisson_modes_have_signal_dependent_variance() -> None:
    gen = ops.make_generator(0, "cpu")
    dark = torch.full((4, 3, 64, 64), 0.05)  # [4,3,64,64]
    bright = torch.full((4, 3, 64, 64), 0.6)  # [4,3,64,64]
    for mode in ("exact", "gaussian_approx"):
        noise_dark = ops.add_poisson_noise(dark, 1.0, False, gen, mode=mode) - dark
        noise_bright = ops.add_poisson_noise(bright, 1.0, False, gen, mode=mode) - bright
        assert noise_dark.std() < noise_bright.std()
        assert noise_bright.std() > 0.0
    assert ops.resolve_poisson_mode("auto", torch.device("cpu")) == "gaussian_approx"
    assert ops.resolve_poisson_mode("auto", torch.device("cuda")) == "exact"


def test_record_states_resolved_runtime_choices() -> None:
    hr = _synthetic_clip(t=2, h=64, w=64)  # [3,2,64,64]
    record = degrade_hr_to_lr(hr, "p1_first_order", seed=1).record
    assert record["jpeg_backend"] == "cv2" and record["poisson_mode"] == "gaussian_approx" and record["device"] == "cpu"
    record_torch_jpeg = degrade_hr_to_lr(hr, "p1_first_order", seed=1, jpeg_backend="diffjpeg").record
    assert record_torch_jpeg["jpeg_backend"] == "diffjpeg"


@pytest.mark.GPU
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_gpu_shares_parameters_with_cpu_and_is_deterministic() -> None:
    hr = _synthetic_clip(t=4, h=96, w=128)  # [3,4,96,128]
    cpu = degrade_hr_to_lr(hr, "p1_second_order", seed=11)
    gpu_a = degrade_hr_to_lr(hr.cuda(), "p1_second_order", seed=11)
    gpu_b = degrade_hr_to_lr(hr.cuda(), "p1_second_order", seed=11)
    assert gpu_a.lr.device.type == "cuda"
    assert torch.equal(gpu_a.lr, gpu_b.lr)
    # Sampled parameters are device independent; only noise realisation, JPEG codec and Poisson mode differ.
    keys = ("profile_name", "seed", "scale", "hr_size", "lr_size", "ops")
    assert {k: cpu.record[k] for k in keys} == {k: gpu_a.record[k] for k in keys}
    assert (cpu.lr.float() - gpu_a.lr.float().cpu()).abs().mean() < 12.0  # 8-bit units; noise + JPEG backend differ
    # Without noise (its realisation is device specific and dominates the residual) what remains is the
    # cv2-vs-DiffJPEG gap plus resize / FFT rounding, a few 8-bit units on this content.
    base = get_profile("p1_second_order")
    assert isinstance(base, RealESRGANProfile) and base.stage2 is not None
    noise_free = dataclasses.replace(
        base,
        name="p1_second_order_noise_free",
        stage1=dataclasses.replace(base.stage1, noise=None),
        stage2=dataclasses.replace(base.stage2, noise=None),
    )
    cpu_nf = degrade_hr_to_lr(hr, noise_free, seed=11)
    gpu_nf = degrade_hr_to_lr(hr.cuda(), noise_free, seed=11)
    assert (cpu_nf.lr.float() - gpu_nf.lr.float().cpu()).abs().mean() < 10.0  # 8-bit units


def test_codec_round_trip_keeps_shape_dtype_and_is_lossy() -> None:
    from cosmos_framework.data.generator.augmentors.hr_lr_degradation.codec import codec_available, codec_round_trip

    assert codec_available("libx264") and codec_available("libx265")
    clip = _synthetic_clip(t=6, h=45, w=67).permute(1, 0, 2, 3)  # [6,3,45,67] uint8, odd sizes
    for codec in ("libx264", "libx265"):
        hi = codec_round_trip(clip, codec=codec, crf=18, preset="veryfast")  # [6,3,45,67]
        lo = codec_round_trip(clip, codec=codec, crf=40, preset="veryfast")  # [6,3,45,67]
        assert hi.shape == clip.shape and hi.dtype == torch.uint8
        err_hi = (hi.float() - clip.float()).abs().mean()
        err_lo = (lo.float() - clip.float()).abs().mean()
        assert 0.0 < err_hi < err_lo, f"{codec}: {err_hi} vs {err_lo}"
    as_float = codec_round_trip(clip.float() / 255.0, codec="libx264", crf=23)  # [6,3,45,67] float
    assert as_float.is_floating_point() and 0.0 <= as_float.min() and as_float.max() <= 1.0


def test_p3_profile_applies_codec_to_video_but_not_images() -> None:
    hr = _synthetic_clip(t=6, h=96, w=128)  # [3,6,96,128]
    applied = [degrade_hr_to_lr(hr, "p3_video_codec", seed=s).record for s in range(12)]
    assert any(r["codec_applied"] for r in applied) and any(r["codec"] is None for r in applied)  # prob 0.6
    with_codec = next(r for r in applied if r["codec_applied"])
    assert with_codec["codec"]["codec"] in ("libx264", "libx265") and 18 <= with_codec["codec"]["crf"] <= 35
    image = degrade_hr_to_lr(hr[:, 0], "p3_video_codec", seed=0).record
    assert image["codec_applied"] is False
    a = degrade_hr_to_lr(hr, "p3_video_codec", seed=int(with_codec["seed"]))
    b = degrade_hr_to_lr(hr, "p3_video_codec", seed=int(with_codec["seed"]))
    assert torch.equal(a.lr, b.lr)  # codec round trip is deterministic for the same input and settings


def test_codec_fps_is_recorded_only_when_the_codec_ran() -> None:
    hr = _synthetic_clip(t=6, h=64, w=64)  # [3,6,64,64]
    records = [degrade_hr_to_lr(hr, "p3_video_codec", seed=s, fps=30.0).record for s in range(12)]
    with_codec = [r for r in records if r["codec_applied"]]
    without = [r for r in records if not r["codec_applied"]]
    assert with_codec and without
    assert all(r["codec_fps"] == 30.0 for r in with_codec) and all(r["codec_fps"] is None for r in without)
    assert degrade_hr_to_lr(hr, "p1_first_order", seed=0, fps=30.0).record["codec_fps"] is None


def test_nvenc_options_align_with_x264_semantics() -> None:
    from cosmos_framework.data.generator.augmentors.hr_lr_degradation.codec import (
        _encoder_options,
        codec_available,
        codec_round_trip,
        nvenc_preset,
    )

    assert nvenc_preset("veryfast") == "p1" and nvenc_preset("medium") == "p4" and nvenc_preset("veryslow") == "p7"
    order = [
        nvenc_preset(p) for p in ("ultrafast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow")
    ]
    assert order == sorted(order)  # monotone in speed
    assert nvenc_preset("p3") == "p3"
    with pytest.raises(ValueError):
        nvenc_preset("turbo")
    opts = _encoder_options("h264_nvenc", 27.6, "medium")
    assert opts == {"rc": "vbr", "cq": "28", "b": "0", "preset": "p4"}  # CRF analogue, not constqp
    assert _encoder_options("libx264", 27.6, "medium", threads=2) == {"crf": "28", "preset": "medium", "threads": "2"}
    if codec_available("h264_nvenc") and torch.cuda.is_available():
        with pytest.raises(ValueError, match="at least"):
            codec_round_trip(_synthetic_clip(t=2, h=96, w=128).permute(1, 0, 2, 3), codec="h264_nvenc")
        clip = _synthetic_clip(t=6, h=240, w=320).permute(1, 0, 2, 3)  # [6,3,240,320]
        hi = codec_round_trip(clip, codec="h264_nvenc", crf=18, preset="veryfast")
        lo = codec_round_trip(clip, codec="h264_nvenc", crf=45, preset="medium")
        assert hi.shape == clip.shape
        assert (hi.float() - clip.float()).abs().mean() < (lo.float() - clip.float()).abs().mean()


def test_codec_round_trip_bounds_encoder_and_decoder_threads() -> None:
    """x264/x265 default to machine-sized thread pools; under 8 xdist workers in CI that exhausted the container's
    thread limit and stalled the CPU test phase, so the round trip must keep its thread count small."""
    import threading
    import time

    from cosmos_framework.data.generator.augmentors.hr_lr_degradation.codec import (
        DEFAULT_CODEC_THREADS,
        _encoder_options,
        codec_round_trip,
    )

    assert _encoder_options("libx264", 30, "veryfast", threads=2)["threads"] == "2"
    assert "pools=2" in _encoder_options("libx265", 30, "veryfast", threads=2)["x265-params"]
    assert DEFAULT_CODEC_THREADS <= 4

    def thread_count() -> int:
        return int(next(line for line in open("/proc/self/status") if line.startswith("Threads")).split()[1])

    clip = _synthetic_clip(t=6, h=96, w=128).permute(1, 0, 2, 3)  # [6,3,96,128]
    for codec in ("libx264", "libx265"):
        baseline = thread_count()
        peak = [baseline]
        stop = threading.Event()

        def sample() -> None:
            while not stop.is_set():
                peak.append(thread_count())
                time.sleep(0.001)

        sampler = threading.Thread(target=sample)
        sampler.start()
        codec_round_trip(clip, codec=codec, crf=30, preset="veryfast", threads=2)
        stop.set()
        sampler.join()
        extra = max(peak) - baseline - 1  # minus the sampler thread itself
        # Unbounded this is ~40 (x264) to ~70 (x265) on 16 cores and scales with the host; bounded it is a handful.
        assert extra <= 8, f"{codec} spawned {extra} extra threads with threads=2"


def _stage1_kernels(profile: str | Profile, hr_size: tuple[int, int], seed: int) -> list[int]:
    ops_ = plan_degradation(profile, hr_size, seed=seed).ops
    return [o.params["kernel_size"] for o in ops_ if o.op == "blur" and o.stage == "stage1"]


def test_plan_records_carry_stage_tags_and_effective_resize_factors() -> None:
    base = get_profile("p1_second_order")
    assert isinstance(base, RealESRGANProfile) and base.stage1.resize is not None
    order = {"stage1": 0, "stage2": 1, "final": 2}
    clamped = 0
    for seed in range(50):
        plan = plan_degradation(base, (720, 1280), seed=seed)
        tags = [o.stage for o in plan.ops]
        assert set(tags) <= set(order) and "final" in tags and tags == sorted(tags, key=order.__getitem__)
        reference = (720, 1280) if base.stage1.resize.relative_to == "current" else plan.lr_size
        for o in plan.ops:
            if o.op == "resize":  # one schema for every resize op, the final one included
                assert {"updown", "factor", "factor_effective", "size", "mode"} <= set(o.params)
                if o.stage == "stage1":
                    assert o.params["factor_effective"] == o.params["size"][0] / reference[0]
                    assert o.params["size"][0] >= round(0.75 * plan.lr_size[0]) - 1
                    clamped += o.params["factor_effective"] > o.params["factor"] + 1e-6
    assert clamped > 0  # the inherited range dips below the 0.75 floor; the record shows the draw and the outcome
    for name in PROFILES:  # profiles stay strict-JSON serialisable (no inf / NaN defaults)
        json.dumps(profile_to_dict(get_profile(name)), allow_nan=False)


def test_codec_is_dropped_from_the_plan_and_record_for_single_frame_inputs() -> None:
    hr = _synthetic_clip(t=1, h=64, w=80)  # [3,1,64,80]
    seed = next(s for s in range(20) if plan_degradation("p3_video_codec", (64, 80), seed=s).codec is not None)
    for hr_in in (hr[:, 0], hr):  # image layout and one-frame clip
        rec = degrade_hr_to_lr(hr_in, "p3_video_codec", seed=seed).record
        assert rec["codec"] is None and rec["codec_applied"] is False and rec["codec_skipped"] == "single_frame"
    rec = degrade_hr_to_lr(_synthetic_clip(t=6, h=64, w=80), "p3_video_codec", seed=seed).record
    assert rec["codec"]["stage"] == "codec" and rec["codec_applied"] is True and rec["codec_skipped"] is None


def test_sinc_cutoff_range_is_honoured_and_defaults_to_the_real_esrgan_prior() -> None:
    base = get_profile("p1_first_order")
    assert isinstance(base, RealESRGANProfile) and base.stage1.blur is not None
    bounded_blur = dataclasses.replace(base.stage1.blur, sinc_prob=1.0, sinc_cutoff_range=(math.pi / 2, math.pi))
    bounded = dataclasses.replace(base, name="p1_bounded", stage1=dataclasses.replace(base.stage1, blur=bounded_blur))

    def cutoffs(profile: Profile, seeds: int) -> list[float]:
        return [
            o.params["omega_c"]
            for s in range(seeds)
            for o in plan_degradation(profile, (1080, 1920), seed=s).ops
            if o.op == "blur" and o.params["kernel_type"] == "sinc" and o.stage == "stage1"
        ]

    explicit = cutoffs(bounded, 100)
    assert explicit and math.pi / 2 - 1e-9 <= min(explicit) and max(explicit) <= math.pi + 1e-9
    assert min(cutoffs(base, 500)) < math.pi / 2  # None keeps Real-ESRGAN's prior, which reaches down to pi/5


def test_stage2_gate_has_its_own_stream() -> None:
    base = get_profile("p1_second_order")
    assert isinstance(base, RealESRGANProfile) and base.stage2 is not None
    almost = dataclasses.replace(base, stage2_prob=0.999999)
    sometimes = dataclasses.replace(base, stage2_prob=0.3)
    skipped = 0
    for s in range(200):
        always_plan = plan_degradation(base, (720, 1280), seed=s)
        gated_plan = plan_degradation(sometimes, (720, 1280), seed=s)
        if s < 5:  # prob 1.0 and 0.999999 give identical plans: no plan-shifting draw on the main stream
            assert always_plan.record() == plan_degradation(almost, (720, 1280), seed=s).record()
        # Stage 1 never depends on the gate; only whether stage 2 (and what follows it) is drawn changes.
        assert [o.record() for o in always_plan.ops if o.stage == "stage1"] == [
            o.record() for o in gated_plan.ops if o.stage == "stage1"
        ]
        skipped += not any(o.stage == "stage2" for o in gated_plan.ops)
    assert 120 < skipped < 160, skipped  # 0.7 * 200 = 140 expected


def test_profile_validation_and_resolution_clamps() -> None:
    base = get_profile("p1_first_order")
    assert isinstance(base, RealESRGANProfile)
    for bad in (
        dict(stage2_prob=1.5),
        dict(stage2_prob=0.3),  # no stage2 to gate: would be silently ignored
        dict(min_intermediate_scale=1.5),
        dict(min_resolution_factor=2.0, max_resolution_factor=1.5),
        dict(max_resolution_factor=0.0),
        dict(reference_longest_side=0),
    ):
        with pytest.raises(ValueError):
            dataclasses.replace(base, **bad)
    with pytest.raises(ValueError):
        BlurConfig(sinc_cutoff_range=(math.pi, math.pi / 2))  # reversed
    with pytest.raises(ValueError):
        FinalBlockConfig(sinc_cutoff_range=(0.0, math.pi))  # omega_c = 0 is an all-NaN kernel
    # [1.0, 1.5] keeps sub-reference inputs at the base ranges and stops growth past 1.5x; the inherited profile
    # keeps scaling; with scaling off the clamps are ignored.
    clamped = dataclasses.replace(base, name="p1_clamped", min_resolution_factor=1.0, max_resolution_factor=1.5)
    unscaled = dataclasses.replace(
        base, name="p1_unscaled", scale_kernels_with_resolution=False, max_resolution_factor=0.5
    )
    for seed in range(5):
        assert _stage1_kernels(clamped, (360, 640), seed) == _stage1_kernels(clamped, (405, 720), seed)
        assert _stage1_kernels(clamped, (1080, 1920), seed) == _stage1_kernels(clamped, (2160, 3840), seed)
        assert _stage1_kernels(unscaled, (2160, 3840), seed) == _stage1_kernels(unscaled, (360, 640), seed)
    assert any(_stage1_kernels(base, (1080, 1920), s) != _stage1_kernels(base, (2160, 3840), s) for s in range(5))


def _blur_kernels(profile: str | Profile, hr_size: tuple[int, int], seed: int) -> list[int]:
    return [o.params["kernel_size"] for o in plan_degradation(profile, hr_size, seed=seed).ops if o.op == "blur"]


def test_regime_profiles_match_their_specification() -> None:
    assert set(REGIME_PROFILES) == {f"{k}_{r}" for k in ("img", "vid") for r in ("clean", "mild", "moderate", "harsh")}
    for mix, prefix in ((IMAGE_SR_DEFAULT_MIX, "img_"), (VIDEO_SR_DEFAULT_MIX, "vid_")):
        assert abs(sum(mix.values()) - 1) < 1e-9 and all(k.startswith(prefix) and k in PROFILES for k in mix)
    hr = _synthetic_clip(t=6, h=96, w=128)  # [3,6,96,128]
    for name, prof in REGIME_PROFILES.items():
        out = degrade_hr_to_lr(hr, name, seed=1)
        assert out.lr.shape == (3, 6, 48, 64) and out.lr.dtype == torch.uint8
        if isinstance(prof, RealESRGANProfile):  # declared resize ranges respect the floor, so it never binds
            for stage in (prof.stage1, prof.stage2):
                if stage is not None and stage.resize is not None:
                    assert stage.resize.scale_range[0] >= prof.min_intermediate_scale, name
    # Video regimes never JPEG after the final resize (the codec is the compression term); vid_clean is a clean
    # resize with an occasional clean H.264 re-encode.
    for name in ("vid_mild", "vid_moderate", "vid_harsh"):
        for seed in range(30):
            ops_ = [o.op for o in plan_degradation(name, (720, 1280), seed=seed).ops]
            assert "jpeg" not in ops_[max(i for i, o in enumerate(ops_) if o == "resize") :], (name, seed, ops_)
    clean = [plan_degradation("vid_clean", (720, 1280), seed=s) for s in range(40)]
    assert all([o.op for o in p.ops] == ["resize_clean"] for p in clean)
    codecs = [p.codec.params for p in clean if p.codec is not None]
    assert codecs and all(c["codec"] == "libx264" and 16 <= c["crf"] <= 20 for c in codecs)
    # img_moderate runs its second stage 30% of the time.
    plans = [plan_degradation("img_moderate", (720, 1280), seed=s) for s in range(200)]
    n_stage2 = sum(any(o.stage == "stage2" for o in p.ops) for p in plans)
    assert 40 < n_stage2 < 80, n_stage2
    # Resolution scaling: 1280 px reference clamped to [1.0, 1.5], so 360p == 720p, 1080p is 1.5x, 4K == 1080p.
    for seed in range(10):
        k_720, k_1080 = _stage1_kernels("img_mild", (720, 1280), seed), _stage1_kernels("img_mild", (1080, 1920), seed)
        assert k_1080 == [scale_kernel_size(k, 1.5, 41) for k in k_720]
        assert _blur_kernels("img_mild", (360, 640), seed) == _blur_kernels("img_mild", (720, 1280), seed)
        assert _blur_kernels("img_mild", (1080, 1920), seed) == _blur_kernels("img_mild", (2160, 3840), seed)


def test_image_regimes_jpeg_in_the_final_block_with_declared_sinc_cutoffs() -> None:
    # The JPEG sits in the final block: Real-ESRGAN's random order puts it after the final resize (on the LR grid)
    # about half the time and just before it otherwise. Stage and final sinc cutoffs stay in the declared range.
    at_lr = with_jpeg = 0
    cutoffs: list[float] = []
    for s in range(300):
        plan = plan_degradation("img_mild", (1080, 1920), seed=s)
        cutoffs += [o.params["omega_c"] for o in plan.ops if o.op == "blur" and o.params["kernel_type"] == "sinc"]
        jpegs = [i for i, o in enumerate(plan.ops) if o.op == "jpeg"]
        if jpegs:
            with_jpeg += 1
            at_lr += jpegs[-1] > max(i for i, o in enumerate(plan.ops) if o.op == "resize")
    assert 0.3 < at_lr / with_jpeg < 0.7
    assert cutoffs and min(cutoffs) >= math.pi / 2 - 1e-9  # img_mild declares (pi/2, pi)
