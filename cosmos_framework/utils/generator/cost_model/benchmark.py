# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Time real training steps and write the calibration the cost model prices against.

Run under ``torchrun`` on a GPU node. A single shape::

    torchrun --nproc_per_node=4 -m cosmos_framework.utils.generator.cost_model.benchmark \\
        --experiment cosmos3_training_ablations_ga_16bm8b_baseline \\
        --resolutions 480 --frames 81 \\
        --output-dir ./calibration_run

or a full cost-versus-length curve, which is the cross product of
``--resolutions`` and ``--frames`` measured under ONE model init::

    torchrun --nproc_per_node=4 -m cosmos_framework.utils.generator.cost_model.benchmark \\
        --experiment cosmos3_training_ablations_ga_16bm8b_baseline \\
        --resolutions 256 480 720 --frames 1 5 17 33 61 121 181 241 297 \\
        --output-dir ./curve_run

Each writes its calibration to ``<output_dir>/benchmark.json``, alongside the AOT-compiled
VAE cache (``vae_aot_cache/``, when ``--compile-vae`` is on) and any ``torch.profiler``
trace (``torch_profiles/``, when ``--profile`` is on).

Sweeping inside one process is not a convenience: building the model and paying
``torch.compile`` warm-up costs minutes, while a timed point costs seconds, so
re-launching per shape would spend almost all of its wall time on setup.

The benchmark drives the model directly rather than through
:class:`~cosmos_framework.trainer.ImaginaireTrainer`: it builds the model from a
registered experiment, feeds it SYNTHETIC batches, and skips the dataloader,
checkpointing and W&B. Synthetic input is the point, not a shortcut -- step size
has to be set exactly to sweep it, and real data would add loader stalls to a
number that is meant to measure compute.

The sweep varies one thing: ``--frames`` and ``--resolutions`` (and
``--text-tokens``) move the token count per sample, giving cost as a function of
sequence length. The batch size is not an axis but a consequence -- at every
shape the sweep runs as many whole samples per rank as fit within
``--tokens-per-step`` tokens. That still moves the batch across rows, since the
count that fits varies inversely with the sample's own length, which is what
lets the fit tell the per-sample cost from the per-step one. It moves it in
lockstep with the shape, though, so a step's total token count barely varies
among the shapes short enough to be batched, and separating the two rests on the
long clips that exceed the target on their own and on the caption sweep shifting
the split between the streams.

``max_num_tokens_after_packing`` is raised automatically to admit the largest step
the sweep builds, which is ``--tokens-per-step`` itself, or one sample where a
sample alone is longer than that, because the experiment ships a budget sized for
its own training mix and the packer refuses anything longer. Points run in sweep order,
each resolution in the order given with ascending clip length within it and
ascending caption length within that, and when one runs out of memory the rest
of that resolution's frame ladder is skipped rather than retried: longer clips
at the same resolution can only need more memory.

Weights are randomly initialized (pretrained loading is disabled) because step
time does not depend on their values, and loading them would add minutes per
run.

The timed step never carries an EMA copy, and carries the parameter update only
under ``--optimizer``. Both are elementwise over parameters, so neither can depend
on what the batch held: they shift every step time by the same constant, which the
fit absorbs into ``c0`` and no per-token coefficient. Dropping them buys memory --
the optimizer states and an fp32 ``net_ema`` are several bytes per parameter each
-- which is what lets a larger model be measured at all. The cost is that ``c0`` is
then a lower bound, so ``peak_mem_gb`` and any price that charges a whole step
(``newstep``, the packer's reserved fixed cost) understate training that enables
them, while incremental prices are exact either way.
"""

from __future__ import annotations

import contextlib
import math
import os
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import tyro

from cosmos_framework.utils.config import load_config
from cosmos_framework.utils.lazy_config import instantiate
from cosmos_framework.utils import distributed, log, misc
from cosmos_framework.utils.generator.cost_model.estimator import (
    PEAK_TFLOPS_BY_DEVICE,
    BucketMeasurement,
    Calibration,
    descriptor_from_model,
    dtypes_from_model,
    fit_token_cost_model,
    flops_per_sample,
)
from cosmos_framework.utils.generator.cost_model.spec import SampleSpec
from cosmos_framework.data.generator.sequence_packing import SequencePlan
from cosmos_framework.utils.generator.method_timer import MethodTimer

# NCCL reads this at communicator creation (inside distributed.init() below), so it must
# be set before that call, not just before the timed region. Production sets the same
# value for real jobs via submit_helper.py's SlurmExecutor env (NVLink SHARP causes job
# failures on GB200 clusters); this benchmark is launched directly with torchrun rather
# than through that launcher, so without this it would default to NVLS enabled and time
# a communication pattern real training never runs. setdefault so an explicit
# NCCL_NVLS_ENABLE in the launching shell still wins.
os.environ.setdefault("NCCL_NVLS_ENABLE", "0")

CONFIG_PATH = "cosmos_framework/configs/base/config.py"
DEFAULT_EXPERIMENT = "t2w_mot_exp306_006_qwen3_vl_8b_ratio_sample_balancing"


def _peak_tflops(device_name: str) -> float:
    """Dense BF16 peak for a CUDA device name, or 0.0 when it is not in the table."""
    for known, peak in PEAK_TFLOPS_BY_DEVICE.items():
        if known.lower() in device_name.lower():
            return peak
    return 0.0


def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def build_sweep_specs(
    resolutions: list[str],
    frame_counts: list[int],
    aspect_ratio: str,
    text_token_counts: list[int],
) -> list[SampleSpec]:
    """Expand the resolution x frames x caption-length cross product into specs.

    A single frame is an image rather than a one-frame video: that is how a
    still actually reaches the generation tower, and ``SampleSpec`` rejects
    video shorter than one VAE temporal window anyway.

    Caption length is a real sweep axis, not a constant, because the cost model
    fits separate coefficients for the understanding and generation streams and
    cannot identify the understanding ones unless the caption varies. With a
    single caption length the intercept and both understanding columns are three
    constants, and the fit correctly refuses to report them -- so a sweep meant to
    calibrate the full model has to move this axis.

    Raises:
        ValueError: If a frame count is neither 1 nor a valid ``1 + 4k`` video
            length, or a resolution bucket does not exist. Both are caught here
            rather than after the model has spent minutes initializing.
    """
    specs: list[SampleSpec] = []
    for resolution in resolutions:
        for frames in frame_counts:
            modality = "image" if frames == 1 else "video"
            for text_tokens in text_token_counts:
                try:
                    specs.append(
                        SampleSpec(
                            modality=modality,
                            resolution=resolution,
                            aspect_ratio=aspect_ratio,
                            num_pixel_frames=frames,
                            text_tokens=text_tokens,
                        )
                    )
                except (ValueError, KeyError) as error:
                    raise ValueError(
                        f"Bad sweep point {resolution}p x {frames}f x {text_tokens} text tokens: {error}"
                    ) from error
    return specs


def any_rank(flag: bool) -> bool:
    """Whether ANY rank set ``flag``.

    Ranks have to abandon a sweep point together. If one skipped a point the
    others measured, they would enter the next point's collectives a different
    number of calls apart and the job would hang instead of failing.
    """
    if not dist.is_initialized():
        return flag
    vote = torch.tensor([1.0 if flag else 0.0], device="cuda")
    dist.all_reduce(vote, op=dist.ReduceOp.MAX)
    return bool(vote.item() > 0)


def build_model(
    experiment: str,
    world_size: int,
    extra_overrides: list[str],
    context_parallel_shard_degree: int = 1,
) -> tuple[Any, Any, Any]:
    """Instantiate the experiment's model plus its optimizer and scheduler.

    ``data_parallel_shard_degree`` is pinned to the live world size. The
    experiment ships a cluster-scale value and ``ParallelDims`` would otherwise
    clamp it with a warning; pinning states the intent for a single node.

    ``context_parallel_shard_degree`` is an OVERLAY axis on top of that, not an
    additional partition of it: ``ParallelDims`` validates
    ``dp_replicate * dp_shard == world_size`` independently of ``cp``, so pinning
    ``dp_shard`` to the full world size stays correct whatever CP degree is
    requested -- CP ranks share the same rank slots as dp, splitting one sample's
    packed sequence across the group rather than consuming extra ranks.

    Args:
        experiment: Registered Hydra experiment name.
        world_size: Ranks in this job.
        extra_overrides: Additional OmegaConf overrides, applied last.
        context_parallel_shard_degree: Context-parallel group size. Must divide
            ``world_size``; validated by the caller before this is reached.

    Returns:
        ``(model, optimizer, scheduler)``, model already on CUDA.

    Raises:
        ValueError: If an override tries to configure the EMA copy, which this
            function pins off and the calibration records as untimed.
    """
    ema_override = next((override for override in extra_overrides if "ema.enabled" in override), None)
    if ema_override is not None:
        raise ValueError(
            f"The benchmark never builds the EMA copy and records the EMA average as untimed in every "
            f"calibration it writes, so {ema_override!r} is either redundant or would make that record wrong. "
            "Remove it."
        )
    overrides = [
        "--",
        f"experiment={experiment}",
        # Timing does not depend on weight values, and loading costs minutes.
        "model.config.vlm_config.pretrained_weights.enabled=False",
        f"model.config.parallelism.data_parallel_shard_degree={world_size}",
        f"model.config.parallelism.context_parallel_shard_degree={context_parallel_shard_degree}",
        "callbacks=[basic,optimization,job_monitor]",
        "model.config.ema.enabled=False",
    ]
    # Applied last so an explicit override on the command line still wins.
    overrides.extend(extra_overrides)
    log.info(f"Loading {experiment} with overrides {overrides}")
    config = load_config(CONFIG_PATH, overrides)

    # ImaginaireTrainer.__init__ applies this before building the model; this benchmark
    # never constructs a trainer (see the module docstring), so without this call
    # torch.fx.experimental._config.use_duck_shape stays at PyTorch's default (True)
    # instead of the experiment's own config.trainer.compile_config.use_duck_shape --
    # almost always False in this repo's real configs, because True lets dynamo tie two
    # unrelated tensors to the same symbolic size purely because they were equal on the
    # first trace, which then forces a recompile (possibly mid-bucket, corrupting that
    # bucket's timed sec_per_step) the moment they diverge. Setting it here makes the
    # benchmark recompile on the same guards production does, not fewer or more.
    misc.set_torch_compile_options(
        config.trainer.compile_config.recompile_limit, config.trainer.compile_config.use_duck_shape
    )

    model = instantiate(config.model).cuda()
    optimizer, scheduler = model.init_optimizer_scheduler(config.optimizer, config.scheduler)
    return model, optimizer, scheduler


def synthetic_batch(
    modality: str,
    height: int,
    width: int,
    num_pixel_frames: int,
    batch_size: int,
    text_token_ids: int,
) -> dict[str, Any]:
    """Build one synthetic packed-training batch of ``batch_size`` samples.

    Follows the batch contract exercised by
    ``cosmos_framework/model/generator/omni_mot_model_test.py``: raw uint8 pixels
    (the VAE encode is part of the timed step) and raw text token ids (the
    network does its own embedding).

    The two modalities want DIFFERENT ranks. Video items are ``[C, T, H, W]``,
    but image items are ``[C, H, W]``: ``_augment_image_dim_inplace`` rearranges
    them with ``"c h w -> 1 c 1 h w"``, adding the batch and temporal axes
    itself, so handing it a length-1 temporal axis fails on rank rather than
    being silently accepted.

    ``condition_frame_indexes_vision`` is left empty so every latent frame is
    generated. A conditioning frame would be clean context rather than a
    generation token, which is not what the cost model prices.
    """
    is_image = modality == "image"
    media_key = "images" if is_image else "video"
    pixel_shape = (3, height, width) if is_image else (3, num_pixel_frames, height, width)
    return {
        "ai_caption": ["Synthetic cost-model benchmark sample."] * batch_size,
        media_key: [[torch.randint(0, 256, pixel_shape, dtype=torch.uint8).cuda()] for _ in range(batch_size)],
        "image_size": [
            torch.tensor([[height, width, height, width]], dtype=torch.float32).cuda() for _ in range(batch_size)
        ],
        "text_token_ids": [
            [torch.randint(0, 1000, (text_token_ids,), dtype=torch.long).cuda()] for _ in range(batch_size)
        ],
        "conditioning_fps": [torch.tensor([24.0]).cuda().to(torch.bfloat16) for _ in range(batch_size)],
        "sequence_plan": [
            SequencePlan(has_text=True, has_vision=True, condition_frame_indexes_vision=[]) for _ in range(batch_size)
        ],
    }


def run_step(
    model: Any,
    optimizer: Any,
    scheduler: Any,
    batch: dict[str, Any],
    iteration: int,
    step_optimizer: bool = True,
) -> dict[str, Any]:
    """One full training step: forward, backward, optimizer, zero_grad.

    Mirrors ``ImaginaireTrainer.training_step`` minus the callbacks, the grad
    scaler (a no-op in bf16) and gradient accumulation, which this experiment
    leaves at 1. The model's own hooks stay in, because work like the MoE bias
    update is real per-step cost that the fixed term should include. The EMA
    average, which would otherwise run from inside ``on_before_zero_grad``, is
    absent because ``build_model`` never builds ``net_ema``.

    ``step_optimizer=False`` runs forward and backward but not the update. The
    parameter update is elementwise over PARAMETERS, so it costs the same whatever
    the batch contains: it contributes only to ``c0``, and not one nanosecond to
    any per-token coefficient. Gradient reduce-scatter still happens inside
    ``backward``, so the token-dependent communication is still measured.

    Two consequences worth being deliberate about. The fitted ``c0`` comes out
    lower by the update's cost, which matters for the cost of a sample that forces
    a new step and for the packer's reserved fixed cost, though not for the
    incremental price that is quoted. And Adam allocates its moments lazily on the first
    step, so never stepping means never allocating them -- which is what lets a
    model fit that otherwise would not, and equally means ``peak_mem_gb`` and the
    out-of-memory ceiling no longer describe real training.
    """
    output_batch, loss = model.training_step(batch, iteration)
    loss.backward()
    model.on_after_backward()
    model.on_before_optimizer_step(optimizer, scheduler, iteration=iteration)
    if step_optimizer:
        optimizer.step()
        scheduler.step()
    model.on_before_zero_grad(optimizer, scheduler, iteration=iteration)
    # Cleared either way: gradients would otherwise accumulate across steps, and
    # set_to_none frees and reallocates them exactly as a real step does.
    optimizer.zero_grad(set_to_none=True)
    return output_batch


def time_bucket(
    model: Any,
    optimizer: Any,
    scheduler: Any,
    batch: dict[str, Any],
    steps: int,
    warmup: int,
    step_optimizer: bool = True,
    vae_timer: MethodTimer | None = None,
    profiler: torch.profiler.profile | None = None,
) -> tuple[float, dict[str, Any], float, int]:
    """Time ``steps`` steps (rounded up, see below) after ``warmup``, returning
    ``(sec_per_step, last_output, vae_sec_per_step, steps_run)``.

    Warm-up matters more than usual here: this experiment enables
    ``torch.compile``, so the first step of every new shape pays compilation.

    The reported time is the MAX across ranks, not the local mean. Ranks run in
    lockstep through collectives, so the slowest one sets the step time; taking
    a mean would report a step nobody experienced. ``vae_sec_per_step`` follows the
    same MAX convention when ``vae_timer`` is given; it is ``0.0`` otherwise.

    ``profiler``, when given, is stepped once per TIMED iteration only, never during
    warm-up. Its ``wait`` window (``--profile-wait-steps``) is documented as skipping
    steps on top of this bucket's own ``--warmup``; stepping it during warm-up too
    would run that wait window (and possibly its active window) out against
    compiling/warm-up steps instead, so the recorded trace would show compilation
    or warm-up noise rather than the steady-state timed steps it is meant to show.
    """
    for iteration in range(warmup):
        run_step(model, optimizer, scheduler, batch, iteration, step_optimizer=step_optimizer)

    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()

    # OmniMoTModel only calls VAE encode on CP window slot 0 (see
    # OmniMoTModel._get_training_inputs) -- once every cp_size calls to
    # training_step, with the other cp_size-1 calls broadcasting a cached payload
    # instead. Warm-up leaves the model's window slot wherever it happened to land,
    # so without correcting it here the timed region starts mid-window and, unless
    # ``steps`` is itself a multiple of cp_size, clips a fractional window at the
    # far end too -- biasing vae_sec_per_step (and adding small-N noise to
    # sec_per_step itself) by how far ``steps`` sits from a cp_size multiple.
    # Forcing the slot to 0 and rounding steps up to the next whole multiple makes
    # the timed region exactly ``steps // cp_size`` complete windows, each
    # contributing exactly one encode -- the same cadence production trains at.
    # Safe to poke the model's counter directly: this benchmark never runs
    # ImaginaireTrainer (run_step calls the model directly, see its docstring), so
    # there is no ContextParallelDataWindow tracking this rank's slot to desync
    # from.
    cp_size = getattr(getattr(model, "parallel_dims", None), "cp_size", 1)
    if cp_size > 1:
        model._cp_window_slot = 0
        steps = -(-steps // cp_size) * cp_size  # ceil to the next multiple of cp_size

    if vae_timer is not None:
        vae_timer.reset()

    start = time.perf_counter()
    output_batch: dict[str, Any] = {}
    for iteration in range(steps):
        output_batch = run_step(model, optimizer, scheduler, batch, warmup + iteration, step_optimizer=step_optimizer)
        if profiler is not None:
            profiler.step()
    torch.cuda.synchronize()
    sec_per_step = (time.perf_counter() - start) / steps

    vae_sec_per_step = vae_timer.stop_and_read_ms() / 1000.0 / steps if vae_timer is not None else 0.0

    if dist.is_initialized():
        worst = torch.tensor([sec_per_step, vae_sec_per_step], device="cuda")
        dist.all_reduce(worst, op=dist.ReduceOp.MAX)
        sec_per_step, vae_sec_per_step = float(worst[0].item()), float(worst[1].item())
    return sec_per_step, output_batch, vae_sec_per_step, steps


def observed_spec(
    output_batch: dict[str, Any],
    batch_size: int,
    modality: str,
    resolution: str,
    aspect_ratio: str,
    num_pixel_frames: int,
) -> SampleSpec:
    """Build the priced :class:`SampleSpec` from what the packer actually produced.

    The requested text length is raw token ids, but the packer also inserts
    BOS/EOS/begin-generation markers, so the packed understanding length is a
    few tokens longer. Reading it back keeps the calibration's token accounting
    identical to the run's, instead of off by the specials.
    """
    und_tokens = int(output_batch.get("und_token_length", 0))
    return SampleSpec(
        modality=modality,  # type: ignore[arg-type]
        resolution=resolution,
        aspect_ratio=aspect_ratio,
        num_pixel_frames=num_pixel_frames,
        text_tokens=round(und_tokens / batch_size),
    )


def check_vision_tokens(spec: SampleSpec, output_batch: dict[str, Any], batch_size: int) -> str:
    """Compare the analytic vision-token count against the packed one.

    ``spec.vision_tokens`` re-derives the VAE and patchify arithmetic offline;
    if it disagrees with what the model packed, every extrapolated estimate is
    wrong in the same way. Returns a note for the measurement record.
    """
    packed = int(output_batch.get("vision_token_length", 0))
    if batch_size <= 0 or packed <= 0:
        return ""
    packed_per_sample = packed / batch_size
    if abs(packed_per_sample - spec.vision_tokens) < 0.5:
        return ""
    message = f"vision tokens disagree: spec says {spec.vision_tokens}, model packed {packed_per_sample:.1f} per sample"
    log.warning(message)
    return message


def batch_size_for_token_target(spec: SampleSpec, tokens_per_step: int) -> int:
    """Samples per rank that fill one step of ``spec`` to the token target.

    A short step is not a cheap version of a long one. The per-step floor -- every
    parameter and optimizer-state byte through HBM, the FSDP collectives -- is paid
    in full whether the step carries one sample or a thousand, so a step of a few
    hundred tokens spends nearly all of its time on work no sample caused and
    retires the tokens it does carry far more slowly than a full step does. Rows
    like that do not just carry noise: they sit on a different efficiency curve
    from the long clips beside them, and a fit that has to reconcile the two reads
    the difference as a token cost, which is where negative coefficients come from.
    Filling every step to the same target takes that drift out of the design.

    It also buys the one thing the shape axis cannot: because the count needed
    varies inversely with the sample's own length, the per-sample column varies
    across rows while the intercept does not, which is exactly what separates ``F``
    from ``c0``. A sweep run at one fixed batch size leaves those two proportional
    and only their sum identified.

    Whole samples rarely divide the target, and the count is rounded DOWN, making
    the target a ceiling that a step approaches from below rather than a floor it
    clears. Which way to round only matters where a sample is a sizeable fraction
    of the target, and there rounding up is the worse of the two: a 69.5k-token
    720p clip against a 148k target wants 2.13 samples, and taking 3 runs a step
    41% over -- both a different step size from every other row in the sweep, which
    is what the target exists to prevent, and the peak memory and packing budget
    that go with it. Rounding down holds that step to 139k, 6% under. The cost is
    the mirror case, a count landing just above an integer: 4.97 samples becomes 4
    and the step falls 20% short. Nothing is rounded below one sample, so a shape
    longer than the target runs alone and overruns it by its own length.
    """
    if spec.total_tokens <= 0:
        return 1
    return max(1, math.floor(tokens_per_step / spec.total_tokens))


@dataclass
class BenchmarkConfig:
    """What to measure, and how thoroughly.

    The sweep is the cross product of ``resolutions``, ``frames`` and ``text_tokens``,
    every point of it measured under one model init. Each is a list taken space
    separated, as ``--frames 1 5 17``.
    """

    output_dir: str
    """Directory to write this run's output into (rank 0 only): the calibration JSON as
    ``benchmark.json``, the AOT-compiled VAE cache (when --compile-vae is on) under
    ``vae_aot_cache/``, and the torch.profiler trace (when --profile is on) under
    ``torch_profiles/``."""

    experiment: str = DEFAULT_EXPERIMENT
    """Registered Hydra experiment name."""

    resolutions: list[str] = field(default_factory=lambda: ["480"])
    """Resolution tiers to sweep, e.g. ``--resolutions 256 480 720``."""

    aspect_ratio: str = "16,9"
    """Aspect bucket within the tier."""

    frames: list[int] = field(default_factory=lambda: [81])
    """Pixel frame counts to sweep. Each must be 1, which is measured as an image, or a
    video length of 1 + 4k, since the VAE encodes a first frame plus groups of 4."""

    text_tokens: list[int] = field(default_factory=lambda: [512])
    """Caption lengths, as raw token ids per sample. Sweep at least three, e.g.
    ``--text-tokens 64 256 512 1024 2048``, or the fit cannot separate the
    understanding-stream coefficients from the fixed per-step cost and will drop them."""

    tokens_per_step: int = 222_000
    """Tokens per rank every timed step aims to carry, and the only thing setting the batch
    size: each shape runs as many whole samples as fit without exceeding it. Keeps short
    shapes out of the bandwidth-bound regime, where they would sit on a different efficiency
    curve from the long clips they are fitted alongside, and makes the per-sample column vary
    so F can be told apart from c0. A shape longer than this runs one sample and carries its
    own full length, the only case where a step goes over."""

    context_parallel_shard_degree: int = 1
    """Context-parallel group size. CP is an overlay axis: a group of this many ranks
    splits ONE sample's packed sequence across itself (Ulysses-style all-to-all)
    rather than each rank contributing an independent sample, so it changes both the
    communication a step pays and how many distinct samples a step actually processes.
    Must evenly divide the launched world size. --optimizer aside, a calibration
    measured at cp>1 cannot be pooled with one measured at cp=1: see
    Calibration.load_all."""

    steps: int = 8
    """Timed steps per bucket. Under CP (``--context-parallel-shard-degree`` > 1),
    ``time_bucket`` rounds this up to the next multiple of the CP degree so the timed
    region covers whole encode windows -- see its docstring -- so the actual step count
    a bucket ran at (``BucketMeasurement.steps``) may exceed this value."""

    warmup: int = 3
    """Untimed steps per bucket, which is what covers torch.compile."""

    max_tokens: int = 0
    """Skip any point whose packed sequence exceeds this many tokens. 0 attempts every point."""

    compile: bool = True
    """Whether the model is compiled. ``--no-compile`` cuts warm-up to seconds, which
    suits a smoke test of the sweep and not a calibration meant to be trained against."""

    compile_vae: bool = True
    """AOT-compile the VAE encoder (torch.export + aoti_compile_and_package, see
    Wan2pt2VAEInterface.compile_encode) before the sweep starts. On by default, to match how
    production experiments that set compile_tokenizer.enabled=True actually train. This
    benchmark never builds an ImaginaireTrainer and never runs the Callback system at all
    (run_step calls the model directly), so the CompileTokenizer callback that normally
    triggers this in real training never fires here regardless of what the experiment config
    sets. As a side effect, compile_encode also benchmarks every AOT chunk shape as it
    compiles it (see its docstring), populating Wan2pt2VAEInterface.predicted_encode_seconds
    for later use by VAE load balancing -- but that is not why the timed steps below are
    accurate: main() always MEASURES this run's own VAE-encode time via MethodTimer and
    subtracts that, not a prediction, so --no-compile-vae still produces a valid (just
    eager-VAE-timed, less production-representative) calibration."""

    profile: bool = False
    """Wrap the timed steps in ``torch.profiler`` and write a trace per rank to
    ``<output_dir>/torch_profiles/rank<N>``. Off by default: the profiler's own overhead
    lands on whichever bucket is profiled, so a calibration run should not combine
    ``--profile`` with the points it means to fit against -- use it to diagnose a single
    shape, not to run the sweep. Only the FIRST measured bucket is profiled (see
    ``--profile-active-steps``); every later bucket runs exactly as it would without
    ``--profile``, since a warm profiler with an exhausted schedule costs only a per-step
    counter increment. Traces open in TensorBoard's PyTorch Profiler plugin or
    chrome://tracing, written gzip-compressed (see ``tensorboard_trace_handler``'s
    ``use_gzip``) since an uncompressed per-op CUDA trace across every rank is the largest
    thing this script writes and both consumers read ``.gz`` natively; the
    ``.pt.trace.json`` name -- ``.pt`` for "profiler trace", not a saved model -- is
    ``tensorboard_trace_handler``'s own naming, not this script's, so it stays fixed
    regardless of compression."""

    profile_wait_steps: int = 1
    """Untimed steps the profiler skips before it starts recording, on top of this
    bucket's own ``--warmup``. Lets the first step or two of the profiled window itself
    settle (e.g. caching allocator warm-up) before the trace that gets written starts."""

    profile_active_steps: int = 3
    """Steps the profiler actively records once its wait window elapses. Bounds the trace
    to a handful of steps regardless of ``--steps``, since a full bucket's worth of
    per-op CUDA events is far more than needed to read off a kernel breakdown."""

    profile_rank: int = 0
    """Which rank's ``torch.profiler`` context is active; every other rank runs the
    profiled bucket with ``--profile`` effectively off. All ranks still execute the
    same steps and the same collectives -- profiling is local CPU/GPU instrumentation,
    not a collective itself -- so this only cuts the per-op recording overhead (and
    trace file size) on the ranks that would otherwise write a trace nobody asked for.
    It does not remove the profiled rank's own overhead from the timed steps, since
    that rank still runs the sweep and still contributes to the MAX-across-ranks
    ``sec_per_step``; see ``--profile``'s docstring for why a profiled run is a
    diagnostic, not a calibration point, regardless of how many ranks are traced."""

    log_recompiles: bool = True
    """Enable dynamo's TORCH_LOGS=recompiles logging for the run. On by default: a
    recompile mid-sweep is exactly the kind of thing that silently inflates sec_per_step
    for whichever bucket it lands in."""

    optimizer: bool = False
    """Include the parameter update in the timed step. Off by default: the update is
    elementwise over parameters, so skipping it leaves every per-token coefficient
    untouched and lowers only the fixed per-step cost c0, understating the cost of a
    sample that forces a new step and the fixed_seconds the packer reserves out of its
    iteration-time target. It also means Adam never allocates its moments, so peak memory
    and the out-of-memory ceiling stop describing real training. Pass --optimizer to
    measure a full step. Calibrations measured with and without the update cannot be
    pooled."""

    override: list[str] = field(default_factory=list)
    """Extra OmegaConf overrides, applied after the benchmark's own."""

    def __post_init__(self) -> None:
        """Normalize the sweep axes and refuse a sweep that cannot be run.

        Repeats are dropped rather than measured twice: two identical points share one
        calibration key, so the second would overwrite the first and the run would have
        paid GPU-minutes for a measurement it then discards.

        Clip and caption lengths are sorted because these lists are also the run order:
        the sweep walks resolutions in the order given, ascending frames within each,
        ascending captions within each of those. Ascending frames is what lets a
        resolution's first out-of-memory point stand for every longer clip at it.

        Checked here rather than in ``main`` so a typo costs nothing: the whole sweep is
        validated before ``torchrun`` has allocated a node, let alone built a model.
        """
        self.resolutions = list(dict.fromkeys(value.strip() for value in self.resolutions))
        self.frames = sorted(set(self.frames))
        self.text_tokens = sorted(set(self.text_tokens))
        for flag, values in (
            ("--resolutions", self.resolutions),
            ("--frames", self.frames),
            ("--text-tokens", self.text_tokens),
        ):
            if not values:
                raise ValueError(f"{flag} needs at least one value.")
        if self.tokens_per_step <= 0:
            raise ValueError(f"--tokens-per-step must be positive, got {self.tokens_per_step}.")
        if self.steps <= 0:
            raise ValueError(f"--steps must be positive, got {self.steps}.")
        if self.warmup < 0:
            raise ValueError(f"--warmup cannot be negative, got {self.warmup}.")
        if self.context_parallel_shard_degree <= 0:
            raise ValueError(
                f"--context-parallel-shard-degree must be positive, got {self.context_parallel_shard_degree}."
            )
        if self.profile_wait_steps < 0:
            raise ValueError(f"--profile-wait-steps cannot be negative, got {self.profile_wait_steps}.")
        if self.profile_active_steps <= 0:
            raise ValueError(f"--profile-active-steps must be positive, got {self.profile_active_steps}.")
        if self.profile_rank < 0:
            raise ValueError(f"--profile-rank cannot be negative, got {self.profile_rank}.")


def main(args: BenchmarkConfig) -> None:
    if args.log_recompiles:
        torch._logging.set_logs(recompiles=True)

    distributed.init()
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0

    # ParallelDims validates this the same way (cfgp=1 during training, so its
    # ``rest * cfgp * cp == world_size`` check reduces to plain divisibility);
    # caught here so a bad launch fails before torchrun has built a model.
    if world_size % args.context_parallel_shard_degree != 0:
        raise ValueError(
            f"--context-parallel-shard-degree={args.context_parallel_shard_degree} does not divide "
            f"world_size={world_size}: CP ranks overlay onto dp rank slots in equal-sized groups, so the "
            "degree must evenly partition the launched ranks."
        )
    if args.profile_rank >= world_size:
        raise ValueError(f"--profile-rank={args.profile_rank} is out of range for world_size={world_size}.")

    frame_counts = args.frames
    text_token_counts = args.text_tokens
    resolutions = args.resolutions
    specs = build_sweep_specs(resolutions, frame_counts, args.aspect_ratio, text_token_counts)
    points = [(spec, batch_size_for_token_target(spec, args.tokens_per_step)) for spec in specs]
    if args.max_tokens > 0:
        points = [(spec, size) for spec, size in points if spec.total_tokens <= args.max_tokens]
        if not points:
            raise ValueError(f"--max-tokens={args.max_tokens} excluded every requested point")
    # Run order is the order build_sweep_specs produced them in: each resolution in the
    # order given, ascending clip length within it, ascending caption length within
    # that. Deliberately not re-sorted by cost, so the log reads as the ladder it was
    # asked for. Ascending frames is also what the memory skip relies on: the first
    # point to fail at a resolution is the shortest clip that cannot fit, and every
    # longer one there can be abandoned unattempted.

    if len(points) < 2:
        log.warning(
            "Only one point requested. The fixed per-step cost cannot be fitted from a single point, so the "
            "cost model will fall back to charging every sample a full share of it."
        )
    if len({size for _, size in points}) < 2:
        log.warning(
            f"Every point runs at {points[0][1]} sample(s) per rank, so the per-sample column is proportional to "
            "the intercept and only their sum c0 + n*F is identified. The fit will drop F and fold it into c0, "
            "which incremental pricing never charges, understating every short shape. Every shape requested "
            "takes the same number of samples to fill --tokens-per-step, so the target cannot vary the batch: "
            "add a shorter shape, or raise --tokens-per-step well above the longest one."
        )
    if len(text_token_counts) < 3:
        log.warning(
            f"Only {len(text_token_counts)} caption length(s) requested ({text_token_counts}). The understanding-stream "
            "coefficients are not identifiable from that: with the caption fixed, the intercept and both "
            "understanding columns are constants. The fit will drop them and fold their cost into c0 and the "
            "generation terms, which is correct at this caption length and wrong at any other. Sweep at least "
            "three, e.g. --text-tokens 64 256 512 1024 2048."
        )

    overrides = list(args.override)
    if not args.compile:
        overrides.append("model.config.compile.enabled=False")
    # The packer rejects a batch longer than the experiment's own budget, which is
    # sized for its training mix rather than for this sweep. Rounding the batch down
    # holds every step under the token target, so the target IS the largest step --
    # bar a sample longer than it, which runs alone and overruns by its own length.
    # The 1% covers the specials the packer inserts on top of each sample's tokens,
    # a couple per sample and so a few hundred across the widest batch here.
    if not any("max_num_tokens_after_packing" in override for override in overrides):
        longest_sample = max(spec.total_tokens for spec, _ in points)
        budget = int(math.ceil(max(args.tokens_per_step, longest_sample) * 1.01 / 1000.0) * 1000)
        overrides.append(f"model.config.max_num_tokens_after_packing={budget}")
        log.info(f"Sizing max_num_tokens_after_packing={budget} for a {args.tokens_per_step:,}-token step")
    # Every distinct shape is a fresh Dynamo graph, and the experiment's limit is
    # set for a handful of training shapes rather than a sweep of them.
    if not any("recompile_limit" in override for override in overrides):
        overrides.append(f"trainer.compile_config.recompile_limit={max(100, 8 * len(points))}")

    def bucket_label(spec: SampleSpec) -> str:
        """Unique calibration key for one measured point.

        The caption suffix appears only when the sweep varies that axis, so a
        plain resolution/frames sweep keeps the short names it has always had.
        Uniqueness matters more than brevity once one shape can be measured at
        several caption lengths: without the suffix those rows would silently
        overwrite each other, and the calibration would keep only the last. The
        batch size needs no suffix, being a function of the shape and the token
        target rather than an axis of its own.
        """
        label = spec.name
        if len(text_token_counts) > 1:
            label += f"@txt{spec.text_tokens}"
        return label

    log.info(
        f"Sweeping {len(points)} point(s) on {world_size} rank(s) at batch sizes filling a "
        f"{args.tokens_per_step:,}-token step per rank, and caption lengths {text_token_counts}"
    )
    for spec, size in points:
        log.info(f"  queued bs={size} ({size * spec.total_tokens:,} tokens/rank): {spec.describe()}")

    # The update shifts every step time by a constant, so it changes c0 and nothing
    # else -- but it also decides whether the run fits in memory, so state it. No EMA
    # copy is ever built, hence nothing to state about it here.
    carried = "with the parameter update" if args.optimizer else "forward and backward only"
    log.info(f"Timed step: {carried}, no EMA copy, VAE AOTI-compiled")

    model, optimizer, scheduler = build_model(
        args.experiment, world_size, overrides, context_parallel_shard_degree=args.context_parallel_shard_degree
    )
    model.train()
    descriptor, flags = descriptor_from_model(model)
    # Recorded, not chosen: the compute precision scales every coefficient and the
    # two FSDP dtypes move c0, so a fit is only valid at the dtypes it was measured
    # at, and Calibration.load_all refuses to pool across them.
    precision, master_dtype, reduce_dtype = dtypes_from_model(model)
    log.info(f"Dtypes: compute {precision}, FSDP master {master_dtype}, gradient reduce {reduce_dtype}")
    # Authoritative rather than args.context_parallel_shard_degree: this is what
    # ParallelDims actually built the cp mesh at, read the same way any other
    # consumer of model.parallel_dims would.
    cp_size = model.parallel_dims.cp_size
    if cp_size > 1:
        log.info(
            f"Context parallelism: cp={cp_size}. Ranks within a cp group of {cp_size} split ONE sample's packed "
            "sequence rather than each contributing a distinct sample, so samples_per_step below is divided by "
            "cp_size to count distinct samples, not rank contributions."
        )

    if args.compile_vae:
        tokenizer = model.tokenizer_vision_gen
        if isinstance(tokenizer, torch.jit.ScriptModule):
            raise ValueError("model.tokenizer_vision_gen is a JIT ScriptModule; AOTI compile does not apply to it.")

        aoti_cache_dir = str(Path(args.output_dir) / "vae_aot_cache")
        log.info(f"AOT-compiling the VAE encoder for resolutions {resolutions} -> {aoti_cache_dir} ...")
        compile_start = time.perf_counter()
        tokenizer.compile_encode(resolutions, output_dir=aoti_cache_dir)
        log.info(f"VAE AOT compile finished in {time.perf_counter() - compile_start:.1f}s")

    # Always measured, not predicted: predicted_encode_seconds (populated as a side effect of
    # compile_encode above) is for decisions that have to be made BEFORE a step runs -- like
    # VAE load balancing -- but here the step is actually running, so the real per-step VAE
    # time via CUDA events is available and strictly more accurate (it captures dispatch
    # overhead and any eager fallback the isolated per-chunk benchmark wouldn't see).
    vae_timer = MethodTimer(model, "encode")

    profile_dir = str(Path(args.output_dir) / "torch_profiles")
    profile_this_rank = args.profile and rank == args.profile_rank
    profiler_cm = (
        torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(
                wait=args.profile_wait_steps, warmup=0, active=args.profile_active_steps, repeat=1
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(f"{profile_dir}/rank{rank}", use_gzip=True),
        )
        if profile_this_rank
        else contextlib.nullcontext()
    )
    if profile_this_rank:
        log.info(
            f"Profiling the first measured bucket on rank {rank} only: {args.profile_wait_steps} step(s) "
            f"skipped, then {args.profile_active_steps} recorded, trace written to {profile_dir}. Every other "
            "rank runs the same steps and collectives without the profiler attached."
        )
    elif args.profile:
        log.info(f"--profile is on but this is rank {rank}; only rank {args.profile_rank} attaches a profiler.")

    measurements: dict[str, BucketMeasurement] = {}
    # Smallest packed length that ran out of memory, per resolution. Longer
    # clips at that resolution cannot fit either, so they are never attempted.
    oom_floor: dict[str, int] = {}
    skipped: dict[str, str] = {}
    with profiler_cm as profiler:
        for point_index, (spec, batch_size) in enumerate(points):
            label = bucket_label(spec)
            floor = oom_floor.get(spec.resolution)
            if floor is not None and spec.total_tokens >= floor:
                reason = f"skipped: {spec.resolution}p ran out of memory at {floor} tokens"
                log.warning(f"{label}: {reason}")
                skipped[label] = reason
                continue

            height, width = spec.pixel_shape
            modality = spec.modality
            failed = False
            sec_per_step = 0.0
            vae_sec_per_step = 0.0
            steps_run = args.steps
            output_batch: dict[str, Any] = {}
            try:
                batch = synthetic_batch(modality, height, width, spec.num_pixel_frames, batch_size, spec.text_tokens)
                torch.cuda.reset_peak_memory_stats()
                sec_per_step, output_batch, vae_sec_per_step, steps_run = time_bucket(
                    model,
                    optimizer,
                    scheduler,
                    batch,
                    args.steps,
                    args.warmup,
                    step_optimizer=args.optimizer,
                    vae_timer=vae_timer,
                    # Only the first bucket carries the profiler's overhead -- see
                    # BenchmarkConfig.profile.
                    profiler=profiler if point_index == 0 else None,
                )
            except torch.OutOfMemoryError:
                failed = True
                log.warning(f"{label}: out of memory at {spec.total_tokens} tokens x bs{batch_size}")

            # Ranks vote so they abandon the point together; a one-sided skip would
            # desync the next point's collectives.
            if any_rank(failed):
                floor = min(spec.total_tokens, oom_floor.get(spec.resolution, spec.total_tokens))
                oom_floor[spec.resolution] = floor
                skipped[label] = f"out of memory at {spec.total_tokens} tokens x bs{batch_size}"
                continue

            spec = observed_spec(
                output_batch, batch_size, modality, spec.resolution, args.aspect_ratio, spec.num_pixel_frames
            )
            note = check_vision_tokens(spec, output_batch, batch_size)

            # Subtract this run's own MEASURED VAE-encode time (via vae_timer / CUDA events)
            # rather than a predicted one, so the fitted cost model prices transformer compute
            # only. Measuring beats predicting here because the step is actually running --
            # predicted_encode_seconds exists for decisions that have to be made before a step
            # runs (VAE load balancing), not for after-the-fact accounting where the real
            # number is directly available.
            sec_per_step = max(sec_per_step - vae_sec_per_step, 0.0)

            # A cp group of cp_size ranks cooperates on ONE logical batch -- each rank
            # narrows its own locally built sequence down to its 1/cp_size shard rather
            # than contributing a distinct sample (see context_parallel_utils.py). So
            # only one cp group's worth of batch_size samples is real work per group,
            # and there are world_size / cp_size such groups in the step.
            samples_per_step = batch_size * world_size // cp_size
            gpu_sec_per_step = sec_per_step * world_size
            sample_flops = flops_per_sample(descriptor, spec, flags)
            measurements[label] = BucketMeasurement(
                name=label,
                spec=spec,
                steps=steps_run,
                samples_per_step=samples_per_step,
                sec_per_step=sec_per_step,
                gpu_sec_per_sample=gpu_sec_per_step / samples_per_step,
                flops_per_sample=sample_flops,
                achieved_tflops_per_gpu=sample_flops * samples_per_step / gpu_sec_per_step / 1e12,
                peak_mem_gb=torch.cuda.max_memory_allocated() / 2**30,
                world_size=world_size,
                note=note,
            )
            # Both units, because the two numbers that matter live in different ones: the
            # token target is per rank, while samples_per_step is global and is what the fit
            # scales its per-sample columns by. Printing the per-sample token count against
            # the global sample count invites multiplying them into a figure that describes
            # neither a rank nor the target.
            cp_suffix = f", cp={cp_size}" if cp_size > 1 else ""
            log.info(
                f"{label}: {sec_per_step * 1000:.1f} ms/step (transformer-only, VAE "
                f"{vae_sec_per_step * 1000:.1f} ms/step measured & subtracted), "
                f"{spec.total_tokens} tokens x {batch_size} samples/rank = {batch_size * spec.total_tokens:,} "
                f"tokens/rank ({samples_per_step} distinct samples/step{cp_suffix}), "
                f"{measurements[label].achieved_tflops_per_gpu:.1f} TFLOP/s/GPU, "
                f"{measurements[label].peak_mem_gb:.1f} GiB peak"
            )

    if not measurements:
        raise RuntimeError("Every sweep point failed or was skipped; nothing to calibrate against.")

    device_name = torch.cuda.get_device_name()
    calibration = Calibration(
        model_name=args.experiment,
        device_name=device_name,
        peak_tflops_per_gpu=_peak_tflops(device_name),
        world_size=world_size,
        context_parallel_shard_degree=cp_size,
        gpus_per_node=min(world_size, torch.cuda.device_count()),
        descriptor_fields=descriptor._asdict(),
        flags=flags,
        measurements=measurements,
        measured_optimizer_step=args.optimizer,
        precision=precision,
        fsdp_master_dtype=master_dtype,
        fsdp_reduce_dtype=reduce_dtype,
        metadata={
            "git_sha": _git_sha(),
            "experiment": args.experiment,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "overrides": overrides,
            "steps": args.steps,
            "warmup": args.warmup,
            "swept_resolutions": resolutions,
            "swept_frames": frame_counts,
            # The batch sizes the target worked out to, so a calibration says how each
            # step was filled without having to re-derive it from the target and shapes.
            "swept_samples_per_step": sorted({size for _, size in points}),
            "tokens_per_step": args.tokens_per_step,
            "swept_text_tokens": text_token_counts,
            # Recorded so a gap in the curve reads as a measured memory limit
            # rather than as a point nobody thought to ask for.
            "skipped": skipped,
        },
    )

    if rank == 0:
        output = str(Path(args.output_dir) / "benchmark.json")
        calibration.save(output)
        log.info(f"Wrote {output}")
        cost_model = fit_token_cost_model(calibration)
        log.info(f"Token cost model ({cost_model.source}):")
        log.info(f"  c0 per step       {cost_model.fixed_gpu_sec_per_step:.4f} GPU-s/step")
        # The same cost as a duration. c0 is the one coefficient summed over ranks, so
        # this is the figure a time target is denominated in, at any training scale.
        log.info(
            f"                    {cost_model.fixed_sec_per_step_per_rank:.6f} s per rank "
            f"(over {cost_model.world_size} ranks)"
        )
        log.info(f"  F  per sample     {cost_model.fixed_gpu_sec_per_sample * 1e3:.3f} ms/sample")
        log.info(f"  A  und attention  {cost_model.gpu_sec_per_und_area * 1e9:.6f} ns/und token^2")
        log.info(f"  B  gen attention  {cost_model.gpu_sec_per_gen_area * 1e9:.6f} ns/gen token^2")
        log.info(f"  C  cross attn     {cost_model.gpu_sec_per_cross_area * 1e9:.6f} ns/und*gen")
        log.info(f"  D  und linear     {cost_model.gpu_sec_per_und_token * 1e6:.3f} us/und token")
        log.info(f"  E  gen linear     {cost_model.gpu_sec_per_gen_token * 1e6:.3f} us/gen token")
        log.info(
            f"  quality           r^2={cost_model.r_squared:.4f}, worst point off by "
            f"{cost_model.max_residual_fraction * 100:.1f}%"
        )
        if cost_model.crossover_gen_tokens > 0:
            log.info(
                f"  crossover         {cost_model.crossover_gen_tokens:,.0f} gen tokens "
                "(generation attention overtakes linear generation work)"
            )
        if not cost_model.und_identifiable:
            log.warning(
                "The understanding-stream coefficients were not identifiable, so this calibration prices only "
                f"captions of exactly {cost_model.und_token_range[0]} tokens. Re-run with three or more "
                "--text-tokens values to fit them."
            )

    if dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    # The module docstring is the description, the dataclass supplying only the per-flag
    # help; the sweep's rationale is what someone reaching for --help needs.
    main(tyro.cli(BenchmarkConfig, description=__doc__))
