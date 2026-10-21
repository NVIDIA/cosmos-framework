# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Causal multiview inference: one segmented rolling-KV rollout per sample, and its outputs.

A causal multiview student (``video_temporal_causal`` with multiview replay K/V, e.g.
Cosmos3-Nano-Sim-Auto) is not served by the one-shot multiview builders: it is sampled chunk by
chunk, with per-camera captions scheduled on the video clock. This module is that frontend. It is a
port of the reference implementation's causal simulation entry point, together with the AR chunk
decode helpers of its shared inference module and the joint decode of its transfer module. That
source is not released, so this is cosmos-framework's own copy, written as glue around the
released helpers it calls. Function names follow the source so the two can be read side by side.

Only the segmented rolling path is ported. The reference falls back to a non-rolling chunk loop when
the exported config carries no rolling settings; here that configuration is refused instead
(:func:`require_segmented_rolling_config`), because the student is published for the rolling
sampler and the fallback would produce a different video with no error.

Deliberate differences from the reference:

- A checkpoint that reads one caption per camera is refused without
  ``multiview.load_caption_from_data``. The reference silently falls back to one merged prompt for
  every camera there, which conditions each view on text it was never trained with.
- A checkpoint loaded from a training experiment is refused: only an export records the caption
  layout here, while the reference also reads it off the training dataloaders.
- Input preparation raises a rank's error on the whole replica in two steps, before and after the
  camera media loads. Those loads, like the joint camera+LiDAR decode, are split over the replica's
  ranks by its partitioner, as in the reference, which uses one step: there, a rank that fails while
  reading its captions waits for its peers while they wait in the partitioner's exchange.
- The joint batch comes from :func:`~cosmos_framework.inference.lidar_transfer.build_joint_cam_lidar_batch`,
  which emits no ``lidar_frame_indices`` (read only by RigRoPE geometry). A measured LiDAR prefix
  sweep is read from ``lidar.condition_path`` only; the reference's top-level
  ``lidar_condition_path``/``num_lidar_condition_frames`` tensor options are not carried.
- The saver also writes ``control_view*.mp4`` from the controls the batch consumed, and writes no
  LiDAR preview videos. The writing rank encodes every camera itself, where the reference spreads
  them over the replica, with the vision and control sets running concurrently and every encode at
  the reference's x264 thread count.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory, mkdtemp
from typing import TYPE_CHECKING, Any, cast

import torch

from cosmos_framework.data.generator.augmentors.duration_fps_text_timestamps import DurationFPSTextTimeStamps
from cosmos_framework.data.generator.augmentors.resolution_text_info import ResolutionTextInfo
from cosmos_framework.data.generator.augmentors.text_tokenizer import (
    _SYSTEM_PROMPT_AV_MULTIVIEW_TRANSFER,
    _SYSTEM_PROMPT_TRANSFER,
)
from cosmos_framework.data.generator.multiview.camera_attributes import CAMERA_VIEW_MAPPING
from cosmos_framework.data.generator.multiview.caption_chunks import (
    MultiviewCaptionChunk,
    view_caption_texts,
)
from cosmos_framework.data.generator.multiview.caption_chunks import (
    load_multiview_caption_chunks_per_view as load_caption_chunks,
)
from cosmos_framework.inference.args import CAUSAL_MULTIVIEW_ROLLING_EXPORT_OVERRIDES, OmniSampleArgs
from cosmos_framework.inference.common.args import SampleOutput, SampleOutputs
from cosmos_framework.inference.inference import (
    _available_cpu_count,
    _fallback_seed,
    _finalize_data_batch,
    _get_prompt_sample_data,
    _loads_training_experiment,
    _multiview_save_thread_budget,
    _parse_json_object_prompt,
)
from cosmos_framework.inference.lidar_transfer import (
    JOINT_DEFAULT_LIDAR_FPS,
    build_joint_cam_lidar_batch,
    joint_lidar_num_sweeps,
    lidar_range_projection,
    load_joint_lidar_condition,
    load_lidar_control,
    save_lidar_range_view,
)
from cosmos_framework.inference.multiview import (
    _get_multiview_condition_pixel_frames,
    _get_multiview_control_pixel_frames,
    get_multiview_sample_data,
)
from cosmos_framework.model.generator.mot.replica_partition import WorkUnit, run_units_locally
from cosmos_framework.model.generator.utils.rolling_kv.rolling_prompt import (
    RollingPromptSchedule,
    RollingPromptSegment,
)
from cosmos_framework.utils import log
from cosmos_framework.utils.generator.multiview import (
    MultiviewDecode,
    iter_multiview_video_by_view,
    pad_multiview_view_video,
    require_multiview_tensor_by_view,
    slice_multiview_view_frames,
)

if TYPE_CHECKING:
    from cosmos_framework.inference.inference import OmniInference
    from cosmos_framework.model.generator.omni_mot_causal_model import OmniMoTCausalModel

__all__ = [
    "CausalMultiviewResult",
    "generate_causal_multiview_sample",
    "generate_rolling_sample",
    "joint_lidar_sweep_ids",
    "require_segmented_rolling_config",
    "save_causal_multiview_outputs",
]

Timer = Callable[[str], AbstractContextManager[Any]]


@dataclass
class CausalMultiviewResult:
    """Everything one causal multiview rollout produced, on the host, in camera order."""

    vision: list[torch.Tensor]  # per camera: [3,F,H,W] in [0,1]
    controls: list[torch.Tensor]  # per camera: [3,F_source,H,W] uint8, before padding to F
    lidar: torch.Tensor | None  # joint only: decoded [1,3,F_lidar,H_l,W_model]
    lidar_control: torch.Tensor | None  # joint only: [1,C,F_lidar,H_l,W_model]
    chunks: list[dict[str, Any]]
    """Every consumed caption segment: frame ranges, the formatted caption per camera, the seed."""
    rolling_prompt: dict[str, Any]
    """The schedule's settings, as written to ``rolling_prompt.json``."""
    latents: list[dict[str, torch.Tensor]]
    """``[]`` unless ``save_latents``; otherwise one dict of ``*_latent`` tensors, as ``latents.pt`` holds."""
    save_latents: bool


def _timed(timer: Timer | None, name: str) -> AbstractContextManager[Any]:
    return nullcontext() if timer is None else timer(name)


def _config_value(config: Any, key: str) -> Any:
    """Read *key* off the model config object, or off the ``config.json`` mapping it was built from."""
    return config.get(key) if isinstance(config, Mapping) else getattr(config, key)


def require_segmented_rolling_config(model_config: Any) -> None:
    """Refuse a causal multiview checkpoint that was not exported for the segmented rolling sampler.

    ``model_config`` is the model's config object or the ``model.config`` mapping of an exported
    ``config.json``. Without the rolling settings the model has no bounded cache and no caption
    schedule, and the framewise or quantized KV options are rejected by the rolling sampler itself,
    but only after every camera's media has been decoded.
    """
    cache_chunks = _config_value(model_config, "rolling_kv_cache_chunks")
    policy = _config_value(model_config, "rolling_prompt_policy")
    if cache_chunks is None or policy != "segmented":
        raise ValueError(
            "Causal multiview inference runs the segmented rolling-KV sampler, but this checkpoint's config has "
            f"rolling_kv_cache_chunks={cache_chunks!r} and rolling_prompt_policy={policy!r}. Re-export the "
            f"checkpoint with --experiment-overrides {CAUSAL_MULTIVIEW_ROLLING_EXPORT_OVERRIDES}."
        )
    framewise = {
        key: value
        for key in ("kv_cache_dtype", "kv_cache_inference_size", "attention_sink_size")
        if (value := _config_value(model_config, key))
    }
    if framewise:
        raise ValueError(
            f"Rolling-KV inference cannot be combined with framewise or quantized KV options, got {framewise}. "
            "Drop --kv-cache-inference-size/--attention-sink-size; the rolling cache is fixed by the export "
            f"(--experiment-overrides {CAUSAL_MULTIVIEW_ROLLING_EXPORT_OVERRIDES})."
        )


def _inference_replica_groups(model: OmniMoTCausalModel) -> tuple[torch.distributed.ProcessGroup, ...]:
    """Reuse the CP and CFG axes; their Cartesian product is one inference replica."""
    parallel_dims = getattr(model, "parallel_dims", None)
    if parallel_dims is None:
        return ()
    return tuple(mesh.get_group() for mesh in (parallel_dims.cp_mesh, parallel_dims.cfgp_mesh) if mesh is not None)


@contextmanager
def _sync_inference_replica_errors(process_groups: Sequence[torch.distributed.ProcessGroup]) -> Iterator[None]:
    """Raise a preparation error on every rank of the replica, before any rank enters model collectives.

    Independent replicas may be preparing different samples, so this must not synchronize WORLD.
    Without replica groups the error is already local and simply propagates.
    """
    if not process_groups:
        yield
        return
    error_flag = torch.zeros(1, dtype=torch.int32, device="cuda")  # [1]
    local_error: Exception | None = None
    try:
        yield
    except Exception as error:
        error_flag += 1  # [1]
        local_error = error
    for group in process_groups:
        torch.distributed.all_reduce(error_flag, op=torch.distributed.ReduceOp.SUM, group=group)  # [1]
    if error_flag.item() > 0:
        # Ranks that failed raise their own error; the others abort with a generic one.
        raise local_error if local_error else RuntimeError("A different GPU rank failed.")


def joint_lidar_sweep_ids(num_sweeps: int, *, camera_fps: float, lidar_fps: float) -> torch.Tensor:  # [T]
    """Camera-clock source ID of every sweep.

    A 10 Hz sweep against a 30 fps camera is an exact stride, so the IDs follow from the two rates and
    a control that carries no IDs of its own can still be placed on the camera clock.
    """
    if num_sweeps < 1:
        raise ValueError(f"num_sweeps must be positive, got {num_sweeps}")
    if camera_fps <= 0 or lidar_fps <= 0:
        raise ValueError(f"camera_fps and lidar_fps must be positive, got {camera_fps}, {lidar_fps}")
    stride = camera_fps / lidar_fps
    return torch.tensor([round(index * stride) for index in range(num_sweeps)], dtype=torch.int64)  # [T]


def per_view_prompt_data(sample: OmniSampleArgs, model: OmniMoTCausalModel, captions: Sequence[str]) -> dict[str, Any]:
    """Match training punctuation, integer duration and structured metadata."""
    width, height = sample.vision_size
    parsed = [obj if (obj := _parse_json_object_prompt(caption)) is not None else caption for caption in captions]
    for caption in parsed:
        if isinstance(caption, dict):
            caption.pop("duration", None)
            caption.pop("fps", None)
    data: dict[str, Any] = {
        "ai_caption": parsed,
        "conditioning_fps": float(sample.fps),
        "num_video_frames_per_view": sample.num_frames,
        "video": None,
        "image_size": (height, width),
        "aspect_ratio": sample.aspect_ratio,
    }
    if sample.duration_template is not None and sample.num_frames > 1:
        DurationFPSTextTimeStamps(
            args={
                "template": sample.duration_template,
                "num_frames_key": "num_video_frames_per_view",
            }
        )(data)
    if sample.resolution_template is not None:
        ResolutionTextInfo(args={"video_template": sample.resolution_template})(data)
    # Training drops each camera caption to an empty caption for CFG. An extra
    # inference-only control directive would change the trained text contract.
    return {model.input_caption_key: [view_caption_texts(data["ai_caption"])]}


def _autoregressive_prompt_sample(sample: OmniSampleArgs) -> OmniSampleArgs:
    """The sample as the reference formats an autoregressive sample's sample-level text: no duration.

    The reference's prompt helper drops both duration templates for an autoregressive sample, so the
    batch builders' captions and the shared negative carry no duration sentence there; this repo's
    shared helper keeps them. Pass this copy wherever that helper formats text for the rollout.
    Per-camera captions are formatted from the sample itself (:func:`per_view_prompt_data`), with
    their duration, as training wrote them.
    """
    return sample.model_copy(update={"duration_template": None, "inverse_duration_template": None})


def _causal_rgb_system_prompt(view_prompts: Sequence[str] | None) -> str:
    """The system prompt of the RGB-only rolling batch: whatever the reference sends today.

    The reference builds this batch with its multiview builder, which picks the AV multiview prompt
    whenever per-camera prompts are packed (always, on the rolling path) and the plain transfer
    prompt otherwise. cosmos-framework's builder instead follows the exported caption task, so the
    choice is pinned here rather than inherited, keeping both runtimes on the same text tokens.

    Open question, not decided here: the Sim-Auto per-view caption streams were trained under
    ``av_joint_camera_lidar_transfer`` (the export records it as ``per_view_caption_task``), samples
    whose LiDAR stream was dropped included, which argues for the joint prompt on this route too.
    Switching breaks parity with the reference until it switches as well. The joint route already
    sends the joint prompt in both runtimes.
    """
    return _SYSTEM_PROMPT_AV_MULTIVIEW_TRANSFER if view_prompts is not None else _SYSTEM_PROMPT_TRANSFER


def _decode_rgb(model: OmniMoTCausalModel, latent: torch.Tensor) -> torch.Tensor:  # [B,C,T,H,W] -> [B,3,F,H,W]
    video = model.decode(latent)  # [B,3,F,H,W]
    output_range = model.tokenizer_vision_gen.get_output_range()
    if output_range == "[-1,1]":
        return ((1 + video) / 2).clamp(0, 1)  # [B,3,F,H,W]
    if output_range == "[0,1]":
        return video  # [B,3,F,H,W]
    raise ValueError(f"Unsupported decoder output range: {output_range}")


def _prepare_ar_vision_chunk_for_decode(
    vision_latent: torch.Tensor, *, sample_n_views: int
) -> torch.Tensor:  # [1,C,V*T,H,W] -> [V,C,T,H,W]
    """Turn a camera-major AR chunk into a view batch for independent streaming decode."""
    if sample_n_views < 1:
        raise ValueError(f"sample_n_views must be positive, got {sample_n_views}.")
    if vision_latent.ndim != 5:
        raise ValueError(f"AR vision chunks must have rank 5, got shape {tuple(vision_latent.shape)}.")
    if vision_latent.shape[0] != 1:
        raise ValueError(
            "Multiview transfer autoregressive decode requires batch_size=1 before views become decode batch rows, "
            f"got shape {tuple(vision_latent.shape)}."
        )
    flat_latent_frames = int(vision_latent.shape[2])
    if flat_latent_frames % sample_n_views != 0:
        raise ValueError(
            f"Camera-major AR chunk length {flat_latent_frames} is not divisible by {sample_n_views} views."
        )
    return require_multiview_tensor_by_view(  # [V,C,T,H,W]
        vision_latent, sample_n_views, flat_latent_frames // sample_n_views
    )


def _assemble_decoded_ar_vision_chunks(
    decoded_vision_chunks: list[torch.Tensor], *, sample_n_views: int
) -> torch.Tensor:  # chunks: list[[V,C,F_chunk,H,W]], returns [1,C,V*F,H,W]
    """Concatenate streaming decode chunks per view, then restore camera-major layout."""
    if not decoded_vision_chunks:
        raise RuntimeError("Autoregressive generation produced no decoded vision chunks.")
    decoded_by_stream = torch.cat(decoded_vision_chunks, dim=2)  # [V,C,F,H,W]
    if decoded_by_stream.shape[0] != sample_n_views:
        raise ValueError(
            f"Expected one decoded stream per view, got {decoded_by_stream.shape[0]} streams for "
            f"sample_n_views={sample_n_views}."
        )
    decoded_camera_major = decoded_by_stream.permute(1, 0, 2, 3, 4).contiguous()  # [C,V,F,H,W]
    return decoded_camera_major.reshape(  # [1,C,V*F,H,W]
        1,
        decoded_camera_major.shape[0],
        sample_n_views * decoded_camera_major.shape[2],
        decoded_camera_major.shape[3],
        decoded_camera_major.shape[4],
    )


@torch.no_grad()
def generate_rgb_chunk(
    model: OmniMoTCausalModel, sample: OmniSampleArgs, data: dict[str, Any], *, save_latents: bool = False
) -> dict[str, torch.Tensor]:  # vision: [1,3,V*F,H,W], optional latent: [1,C,V*T,h,w]
    assert sample.multiview is not None
    num_views = len(sample.multiview.views)
    is_student = model.config.fixed_step_sampler_config is not None
    decoded: list[torch.Tensor] = []  # list[[V,3,F_chunk,H,W]]
    latents: list[torch.Tensor] = []  # list[[V,C,T_chunk,h,w]]
    # Each camera is one row of the decode batch, so the cached decoder carries every camera's
    # causal state from one chunk to the next instead of decoding the camera-major chunk as one clip.
    with model.tokenizer_vision_gen.use_cached_decoder():
        for output in model.iter_samples_from_batch_autoregressive(
            data,
            mode="video_transfer",
            guidance=1.0 if is_student else sample.guidance,
            seed=sample.seed if sample.seed is not None else _fallback_seed(sample),
            num_steps=sample.num_steps,
            shift=sample.shift,
            normalize_cfg=sample.normalize_cfg,
            sampler_mode="distilled" if is_student else "rf",
            has_negative_prompt=f"neg_{model.input_caption_key}" in data,
        ):
            latent = _prepare_ar_vision_chunk_for_decode(
                output["vision"], sample_n_views=num_views
            )  # [V,C,T_chunk,h,w]
            decoded.append(_decode_rgb(model, latent).cpu())  # [V,3,F_chunk,H,W]
            if save_latents:
                latents.append(latent.detach().cpu())  # [V,C,T_chunk,h,w]
    result = {"vision": _assemble_decoded_ar_vision_chunks(decoded, sample_n_views=num_views)}  # [1,3,V*F,H,W]
    if save_latents:
        result["vision_latent"] = _assemble_decoded_ar_vision_chunks(latents, sample_n_views=num_views)  # [1,C,V*T,h,w]
    return result


def prepare_rolling_prompt_schedule(
    sample: OmniSampleArgs,
    model: OmniMoTCausalModel,
    chunks: list[MultiviewCaptionChunk],
    *,
    per_view_captions: bool,
    system_prompt: str,
) -> tuple[RollingPromptSchedule, list[dict[str, Any]]]:
    """Tokenize consumed native captions on one finite, continuous video timeline."""
    assert sample.multiview is not None
    num_views = len(sample.multiview.views)
    start = chunks[0].frame_start
    end = start + sample.num_frames
    consumed = [chunk for chunk in chunks if chunk.frame_start < end]
    views: list[list[RollingPromptSegment]] = [[] for _ in range(num_views)]
    metadata: list[dict[str, Any]] = []
    cursor = start
    for index, chunk in enumerate(consumed):
        if chunk.frame_start != cursor:
            raise ValueError(
                "Segmented caption spans must be contiguous; gaps and overlaps cannot change the control clock."
            )
        # Keep the existing first-caption/last-frame padding contract. Holding
        # the terminal caption does not replay earlier captions or control frames.
        stop = end if index == len(consumed) - 1 else min(end, chunk.frame_end)
        prompts = (
            chunk.view_prompts if per_view_captions and chunk.view_prompts is not None else [chunk.prompt] * num_views
        )
        segment_sample = sample.model_copy(update={"num_frames": stop - cursor})
        formatted = per_view_prompt_data(segment_sample, model, prompts)[model.input_caption_key][0]
        tokenized = model._tokenize_captions(
            formatted, use_system_prompt=model.vlm_config.use_system_prompt, system_prompt=system_prompt, is_video=False
        )
        if len(tokenized) != num_views:
            raise ValueError("Rolling prompt tokenization must preserve the input camera order.")
        for segments, tokens in zip(views, tokenized, strict=True):
            segments.append(RollingPromptSegment(cursor - start, stop - start, tuple(tokens)))
        metadata.append(
            {
                "frame_range": [cursor - start, stop - start - 1],
                "source_frame_range": [chunk.frame_start, chunk.frame_end - 1],
                "prompt": formatted,
                "seed": sample.seed,
            }
        )
        cursor = stop
    schedule = RollingPromptSchedule(
        fps=float(sample.fps),
        num_frames=sample.num_frames,
        views=tuple(tuple(segments) for segments in views),
        sink_tokens=model.config.rolling_text_sink_tokens,
        special_tokens=model.llm_special_tokens,
        context_before=model.config.rolling_prompt_context_before,
        context_after=model.config.rolling_prompt_context_after,
        media_text_extent=model.config.rolling_media_text_extent,
    )
    return schedule, metadata


@torch.no_grad()
def generate_rolling_sample(
    model: OmniMoTCausalModel,
    sample: OmniSampleArgs,
    *,
    per_view_captions: bool,
    save_latents: bool = False,
    timer: Timer | None = None,
) -> CausalMultiviewResult:
    """Run segmented rolling inference once for the requested horizon, retaining KV across caption changes.

    ``sample.num_frames`` is the horizon. Captions only schedule text along it; they never set it.
    ``timer`` (e.g. ``OmniInference._get_timer``) times input preparation, which media decoding
    dominates, apart from the rollout and its decode.
    """
    assert sample.multiview is not None
    if sample.seed is None:
        seed = _fallback_seed(sample)
        # The fallback hashes the output directory, which the reference lays out differently (one
        # subdirectory per output), so an unseeded sample draws different noise in each runtime.
        log.warning(
            f"Sample '{sample.name}' sets no seed; using seed={seed}, derived from its name and output "
            "directory. Set seed to reproduce this sample under another output directory or runtime."
        )
        sample = sample.model_copy(update={"seed": seed})
    replica_groups = _inference_replica_groups(model)
    with _timed(timer, "inference.prepare_inputs"):
        # Two error barriers: the media loads in the second block enter the replica partitioner's
        # own collectives. With one barrier, a rank that failed earlier (e.g. reading its caption
        # files) would wait in the barrier while its peers wait in the partitioner, and hang.
        with _sync_inference_replica_errors(replica_groups):
            require_segmented_rolling_config(model.config)
            if per_view_captions and not sample.multiview.load_caption_from_data:
                # The reference packs one merged prompt per camera here. These weights read one caption
                # per camera, so that conditions every view on text training never paired with it.
                raise ValueError(
                    "This checkpoint tokenizes one caption per camera; set multiview.load_caption_from_data=True "
                    "with a caption_path per camera."
                )
            chunks = (
                load_caption_chunks(
                    sample.multiview.views,
                    first_only=sample.multiview.use_first_chunk_only,
                    first_caption_paragraph_only=sample.multiview.first_caption_paragraph_only,
                )
                if sample.multiview.load_caption_from_data
                else [MultiviewCaptionChunk(index=0, frame_start=0, frame_end=sample.num_frames, prompt=sample.prompt)]
            )
            source_start = chunks[0].frame_start
            # Spans outside the requested prefix cannot establish its caption coverage.
            chunks = [chunk for chunk in chunks if chunk.frame_start < source_start + sample.num_frames]
            source_end = min(source_start + sample.num_frames, chunks[-1].frame_end)
            if source_start != 0:
                raise ValueError("Rolling RGB/LiDAR inputs must share a source window starting at frame zero.")
            # Full caption schedules must cover the requested media window. Preserve
            # the explicit first-caption mode's native terminal-frame padding contract.
            if not sample.multiview.use_first_chunk_only and source_end < sample.num_frames:
                raise ValueError(
                    f"Rolling caption spans must cover the requested {sample.num_frames}-frame horizon, "
                    f"got {source_end} frames."
                )
        with _sync_inference_replica_errors(replica_groups):
            width, height = sample.vision_size
            # Inside a CP/CFG replica each rank decodes its share of the cameras and receives the rest.
            partitioner = getattr(model, "replica_partitioner", None)
            conditions = _get_multiview_condition_pixel_frames(
                sample, model, height=height, width=width, partitioner=partitioner
            )  # list[[3,F_cond,H,W]|None]
            source_controls = _get_multiview_control_pixel_frames(
                sample.model_copy(update={"num_frames": source_end}),
                height=height,
                width=width,
                partitioner=partitioner,
            )  # list[[3,F_source,H,W]]
            controls = [
                slice_multiview_view_frames(frames, start_frame=source_start, end_frame=source_end)
                for frames in source_controls
            ]  # list[[3,F_source,H,W]]
            initial = chunks[0]
            prompts = (
                initial.view_prompts
                if per_view_captions and initial.view_prompts is not None
                else [initial.prompt] * len(controls)
            )
            prompt_sample = _autoregressive_prompt_sample(sample)
            if sample.is_joint_cam_lidar_transfer:
                data = prepare_joint_batch(model, sample, controls, conditions, prompts)
                # The native joint builder has one shared caption. Match the
                # scheduled camera documents before the model tokenizes its probe.
                data.update(per_view_prompt_data(sample, model, prompts))
            else:
                data = get_multiview_sample_data(
                    prompt_sample,
                    model,
                    device=model.tensor_kwargs["device"],
                    condition_frames=conditions,
                    control_frames=controls,
                    view_prompts=prompts,
                )
                # Replaces the builder's captions, and with them its control emphasis.
                data.update(per_view_prompt_data(sample, model, prompts))
                data["system_prompt"] = _causal_rgb_system_prompt(prompts)
                data = _finalize_data_batch(data_batch=data, batch_size=1, model=model)
            if not per_view_captions:
                # Expanding shared positive captions into camera documents must retain
                # the native shared negative prompt and its metadata formatting for CFG.
                shared_prompt_data = _get_prompt_sample_data(
                    prompt_sample, model, h=height, w=width, device=model.tensor_kwargs["device"]
                )
                negative_key = "neg_" + model.input_caption_key
                if negative_key in shared_prompt_data:
                    data[negative_key] = shared_prompt_data[negative_key]
            schedule, metadata = prepare_rolling_prompt_schedule(
                sample, model, chunks, per_view_captions=per_view_captions, system_prompt=data["system_prompt"]
            )
            data["rolling_prompt_schedule"] = schedule
    # Sampling runs collectives: a rank-local failure must reach the launcher, not a sync barrier.
    sampler = generate_joint_chunk if sample.is_joint_cam_lidar_transfer else generate_rgb_chunk
    with _timed(timer, "inference.generate_decode"):
        result = sampler(model, sample, data, save_latents=save_latents)
    video = result["vision"][0].detach().cpu()  # [3,V*F,H,W]
    videos = list(
        iter_multiview_video_by_view(video, sample_n_views=len(controls), num_video_frames_per_view=sample.num_frames)
    )  # list[[3,F,H,W]]
    return CausalMultiviewResult(
        vision=videos,
        controls=controls,
        lidar=result.get("lidar"),
        lidar_control=result.get("lidar_control"),
        chunks=metadata,
        rolling_prompt={
            "policy": "segmented",
            "num_frames": sample.num_frames,
            "fps": sample.fps,
            "source_frame_range": [source_start, source_end - 1],
            "prompt_repeat": False,
            "control_repeat": False,
            "sink_tokens": schedule.sink_tokens,
            "context_before": schedule.context_before,
            "context_after": schedule.context_after,
            "media_text_extent": schedule.fixed_text_extent,
        },
        latents=[{key: value for key, value in result.items() if key.endswith("_latent")}] if save_latents else [],
        save_latents=save_latents,
    )


def prepare_joint_batch(
    model: OmniMoTCausalModel,
    sample: OmniSampleArgs,
    controls: list[torch.Tensor],  # list[[3,F,H,W]]
    conditions: list[torch.Tensor | None],  # list[[3,F_cond,H,W]|None]
    prompts: list[str] | None,
) -> dict[str, Any]:
    assert sample.lidar is not None
    assert sample.multiview is not None
    camera_view_ids: list[int] | None = None
    if model.config.diffusion_expert_config.num_view_embeddings:
        camera_keys = [view.camera_key for view in sample.multiview.views]
        unknown_keys = [key for key in camera_keys if key not in CAMERA_VIEW_MAPPING]
        if unknown_keys:
            raise ValueError(f"Joint camera+LiDAR rig view embeddings require supported camera keys: {unknown_keys}")
        # Subsets and reordered views retain the physical IDs used during training.
        camera_view_ids = [CAMERA_VIEW_MAPPING[key] for key in camera_keys]
    width, height = sample.vision_size
    lidar_fps = float(model.config.lidar_fps or JOINT_DEFAULT_LIDAR_FPS)
    sweeps = joint_lidar_num_sweeps(sample.num_frames, camera_fps=float(sample.fps), lidar_fps=lidar_fps)
    _, lidar_control, frame_ids = load_lidar_control(
        model, sample.lidar, max_frames=sweeps
    )  # control: [Cl,F_lidar,Hl,Wl], IDs: [F_lidar]|None
    if lidar_control.shape[1] != sweeps:
        raise ValueError(f"Joint LiDAR control needs {sweeps} sweeps, got {lidar_control.shape[1]}.")
    if frame_ids is not None:
        expected_ids = torch.arange(sweeps, device=frame_ids.device) * (float(sample.fps) / lidar_fps)  # [F_lidar]
        if not torch.equal(frame_ids, expected_ids):
            raise ValueError("Joint LiDAR source IDs must align with the RGB source window and sensor clocks.")
    prefix = load_joint_lidar_condition(
        model, sample.lidar, num_lidar_sweeps=sweeps, control_frame_ids=frame_ids
    )  # [Cl,1,Hl,Wl]|None
    if prefix is not None and (prefix.shape != lidar_control[:, :1].shape or not torch.isfinite(prefix).all()):
        raise ValueError("LiDAR prefix must be a finite sweep with the same channels and spatial shape as control.")
    prompt_data = (
        per_view_prompt_data(sample, model, prompts)
        if prompts is not None
        else _get_prompt_sample_data(
            _autoregressive_prompt_sample(sample),
            model,
            h=height,
            w=width,
            device=str(model.tensor_kwargs["device"]),
        )
    )
    prompt = prompt_data[model.input_caption_key][0]
    if prompts is None and sample.emphasize_control_in_prompt:
        prompt = prompt.rstrip() + (
            " Follow the wsm and lidar control videos precisely: every camera view must"
            " align with its world-scenario map, and the LiDAR rangemap must align with the"
            " HD-map rangemap, at every frame."
        )
    return build_joint_cam_lidar_batch(
        camera_control=controls,
        camera_conditions=cast(list[torch.Tensor], conditions)
        if all(value is not None for value in conditions)
        else None,
        camera_view_ids=camera_view_ids,
        lidar_control=lidar_control,
        lidar_condition=prefix,
        num_camera_frames=sample.num_frames,
        num_lidar_sweeps=sweeps,
        camera_height=height,
        camera_width=width,
        camera_fps=float(sample.fps),
        prompt_key=model.input_caption_key,
        prompt=prompt,
        negative_prompt=prompt_data.get("neg_" + model.input_caption_key, [None])[0],
        condition_frame_indexes_vision=sample.condition_frame_indexes_vision,
        control_weight=1.0 if sample.wsm is None or sample.wsm.weight is None else float(sample.wsm.weight),
        device=model.tensor_kwargs["device"],
        optimize_raw_pixel_memory=getattr(sample.multiview, "optimize_raw_pixel_memory", False),
    )


def _decode_joint_outputs(
    model: OmniMoTCausalModel,
    vision_latent: torch.Tensor,  # [1,C,V*T_latent,H,W]
    lidar_latent: torch.Tensor,  # [1,C,T_lidar,H,W]
    *,
    sample_n_views: int,
    output_camera_frames: int,
) -> tuple[torch.Tensor, torch.Tensor]:  # [1,3,V*F,H,W], [1,C,T_lidar,H,W], both on CPU
    """Decode every camera view on its own, then the LiDAR clip, assembling the cameras on the host.

    Inside a CP/CFG replica the views and the LiDAR clip are units of the replica partitioner,
    assigned round-robin in that order, and every rank receives every decoded clip. Without one,
    this rank decodes them all, cameras first, then LiDAR.
    """
    if vision_latent.ndim != 5:
        raise ValueError(f"Joint camera latents must have shape [B,C,V*T,H,W], got {tuple(vision_latent.shape)}.")
    partitioner = getattr(model, "replica_partitioner", None)

    def decode_view(view_latent: torch.Tensor) -> torch.Tensor:  # [1,C,T_latent,H,W] -> [1,3,F,H,W]
        return _decode_rgb(model, view_latent)[:, :, :output_camera_frames]

    camera = MultiviewDecode(decode_view, vision_latent, sample_n_views, output_camera_frames, assemble_on_cpu=True)
    units = [*camera.units, WorkUnit(run=lambda: model.decode_lidar(lidar_latent))]  # LiDAR: [1,C,T_lidar,H,W]
    decoded_lidar: torch.Tensor | None = None

    def collect(index: int, decoded: torch.Tensor) -> None:
        nonlocal decoded_lidar
        if index == sample_n_views:
            decoded_lidar = decoded.cpu()  # [1,C,T_lidar,H,W]
        else:
            camera.collect(index, decoded)  # [1,3,F,H,W]

    if partitioner is None:
        run_units_locally(units, collect)
    else:
        partitioner.run(units, collect_on=None, on_collected=collect, label="joint decode")
    video = camera.result()  # [1,3,V*F,H,W]
    assert video is not None and decoded_lidar is not None
    return video, decoded_lidar


@torch.no_grad()
def generate_joint_chunk(
    model: OmniMoTCausalModel, sample: OmniSampleArgs, data: dict[str, Any], *, save_latents: bool = False
) -> dict[str, torch.Tensor]:
    assert sample.multiview is not None
    is_student = model.config.fixed_step_sampler_config is not None
    # Sampling rewrites data["lidar"] into batched clips; keep the original control for the result.
    lidar_control = data["lidar"][0].detach().cpu()  # [Cl,F_lidar,Hl,Wl]
    outputs = model.generate_joint_samples_from_batch_autoregressive(
        data,
        guidance=sample.guidance,
        seed=sample.seed if sample.seed is not None else _fallback_seed(sample),
        num_steps=sample.num_steps,
        shift=sample.shift,
        has_negative_prompt="neg_" + model.input_caption_key in data,
        sampler_mode="distilled" if is_student else "rf",
    )
    vision = outputs["vision"][-1]  # [1,Cv,V*Tv,Hv,Wv]
    lidar = outputs["lidar"][-1]  # [1,Cl,Tl,Hl,Wl]
    video, decoded_lidar = _decode_joint_outputs(
        model, vision, lidar, sample_n_views=len(sample.multiview.views), output_camera_frames=sample.num_frames
    )  # [1,3,V*F,H,W], [1,Cl,F_lidar,Hl,Wl]
    result = {"vision": video, "lidar": decoded_lidar, "lidar_control": lidar_control.unsqueeze(0)}
    if save_latents:
        result.update(
            vision_latent=vision.detach().cpu(), lidar_latent=lidar.detach().cpu()
        )  # [1,Cv,V*Tv,Hv,Wv], [1,Cl,Tl,Hl,Wl]
    return result


def generate_causal_multiview_sample(
    pipe: OmniInference, sample: OmniSampleArgs, *, save_latents: bool
) -> CausalMultiviewResult:
    """Generate one causal multiview sample with the pipeline's model, caption layout and timer.

    Every rank of the replica must call this: the rollout runs model collectives. Both refusals below
    read the setup, which every rank shares, so the ranks refuse together.
    """
    # The setup contract refuses this for an exported checkpoint before the model loads. The decode
    # stream itself exists on the decode-owning rank only, so it cannot be what the ranks check.
    if pipe.setup_args.use_separate_pipeline_vision_decode_gpu:
        raise ValueError("Causal multiview inference does not support a separate VAE decode GPU.")
    if _loads_training_experiment(pipe.setup_args):
        # pipe.per_view_captions is False here whatever the weights were trained with, and the
        # rolling schedule would then give every camera the merged prompt, with no error.
        raise ValueError(
            "Causal multiview inference reads the checkpoint's caption layout from its export: the "
            "per_view_captions key of an exported checkpoint's config.json. A checkpoint loaded from a "
            "training experiment does not record it; run the exported checkpoint instead."
        )
    return generate_rolling_sample(
        cast("OmniMoTCausalModel", pipe.model),
        sample,
        per_view_captions=pipe.per_view_captions,
        save_latents=save_latents,
        timer=pipe._get_timer,
    )


def _save_outputs_transactionally(
    output_dir: Path,
    save_local: Callable[[Path], dict[str, list[Path]]],
    publish: Callable[[dict[str, list[Path]]], SampleOutputs],
    *,
    manifest_name: str = "sample_outputs.json",
) -> SampleOutputs:
    """Stage every file, move the set into ``output_dir``, and write the manifest last.

    ``save_local`` writes into a staging directory and returns the staged paths by artifact;
    ``publish`` receives the same mapping with final paths. If anything fails after a file was
    replaced, the previous files and manifest are restored before the error propagates.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".causal-", dir=str(output_dir)) as staging:
        written = save_local(Path(staging))
        staged_files = [path for paths in written.values() for path in paths]
        final_paths = {key: [output_dir / path.name for path in paths] for key, paths in written.items()}
        manifest = output_dir / manifest_name
        backups: dict[Path, Path] = {}
        published: list[Path] = []
        backup_dir = Path(mkdtemp(prefix=".causal-previous-", dir=str(output_dir)))
        keep_backups = False
        try:
            # Hide any old manifest before replacing media, and retain replaced files for rollback.
            for path in [manifest, *(output_dir / staged.name for staged in staged_files)]:
                if path.exists():
                    backup = backup_dir / path.name
                    path.replace(backup)
                    backups[path] = backup
            for path in staged_files:
                destination = output_dir / path.name
                path.replace(destination)
                published.append(destination)
            published.append(manifest)  # A failed manifest write can also leave a partial file.
            return publish(final_paths)
        except Exception as error:
            try:
                for path in reversed(published):
                    path.unlink(missing_ok=True)
                for path, backup in reversed(backups.items()):
                    backup.replace(path)
            except Exception as rollback_error:
                keep_backups = True
                raise RuntimeError(
                    f"{error}; causal multiview output rollback failed: {rollback_error}; "
                    f"backups retained in {backup_dir}"
                ) from error
            raise
        finally:
            if not keep_backups:
                try:
                    shutil.rmtree(backup_dir)
                except OSError as cleanup_error:
                    # Cleanup must not change the publication result or hide its original error.
                    log.warning(
                        f"Could not remove causal multiview backup directory '{backup_dir}': {cleanup_error}",
                        rank0_only=False,
                    )


def save_causal_multiview_outputs(
    pipe: OmniInference, sample: OmniSampleArgs, result: CausalMultiviewResult
) -> SampleOutputs:
    """Write per-camera videos, LiDAR and metadata, and publish the manifest only once all succeeded.

    Call on the one rank that writes this sample (``pipe.should_process_sample``); nothing here
    synchronizes ranks. ``sample_args.json`` is written here, after generation.
    """
    assert sample.multiview is not None and sample.output_dir is not None
    output_dir = sample.output_dir
    cameras = sample.multiview.camera_keys
    # x264's bitstream depends on its thread count: identical pixels encode to different mp4 bytes
    # under a different count. So every camera is encoded with the count the reference's budget gives
    # for one set of cameras, never with the count a budget over all vision and control files would.
    # This rank encodes every camera, while the reference spreads them over a CP replica; the count
    # is the same whenever the rank may schedule on at least four CPUs per camera.
    _, encoder_threads = _multiview_save_thread_budget(len(cameras))
    # The vision and control sets encode side by side, each at most this many cameras at a time,
    # so that together they stay within the CPUs this rank may schedule on.
    workers_per_set = max(1, min(len(cameras), _available_cpu_count() // (2 * encoder_threads)))

    def save_local(staging_dir: Path) -> dict[str, list[Path]]:
        staged = sample.model_copy(update={"output_dir": staging_dir})
        vision = [
            pipe._run_video_guardrail(f"{output_dir}/{camera}", view)  # [3,F,H,W]
            for camera, view in zip(cameras, result.vision, strict=True)
        ]
        # The control each camera followed, padded to the horizon exactly as the batch padded it.
        width, height = sample.vision_size
        controls = [
            pad_multiview_view_video(frames, num_frames=sample.num_frames, height=height, width=width)
            for frames in result.controls
        ]  # list[[3,F,H,W]] uint8
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="causal-multiview-save") as executor:
            pending = {
                prefix: executor.submit(
                    pipe._save_multiview_view_videos,
                    staged,
                    views,
                    prefix=prefix,
                    fps=sample.fps,
                    encoder_threads=encoder_threads,
                    max_workers=workers_per_set,
                )
                for prefix, views in (("vision", vision), ("control", controls))
            }
            del vision, controls  # Each set's frames are freed once that set is written.
            # Both sets finish before the pool exits, so a failure leaves no encode writing into staging.
            written: dict[str, list[Path]] = {prefix: future.result()[0] for prefix, future in pending.items()}
        written["lidar"] = []
        if result.lidar is not None:
            lidar_fps = float(pipe.model.config.lidar_fps or JOINT_DEFAULT_LIDAR_FPS)
            # Joint preparation validated a zero-origin camera clock, including any source sweep IDs.
            frame_ids = joint_lidar_sweep_ids(
                result.lidar.shape[2], camera_fps=float(sample.fps), lidar_fps=lidar_fps
            )  # [F_lidar]
            written["lidar"].append(
                save_lidar_range_view(
                    result.lidar,
                    staging_dir / "lidar.safetensors",
                    range_projection=lidar_range_projection(pipe.model),
                    frame_ids=frame_ids,
                )
            )
        written["artifacts"] = []
        for name, value in (
            ("lidar.pt", result.lidar),
            ("latents.pt", result.latents if result.save_latents else None),
        ):
            if value is not None:
                path = staging_dir / name
                torch.save(value, path)
                written["artifacts"].append(path)
        metadata = {
            "sample_args.json": sample.model_dump_json(indent=2),
            "caption_chunks.json": json.dumps(result.chunks, indent=2),
            "rolling_prompt.json": json.dumps(result.rolling_prompt, indent=2),
        }
        for name, content in metadata.items():
            path = staging_dir / name
            path.write_text(content)
            written["artifacts"].append(path)
        return written

    def publish(written: dict[str, list[Path]]) -> SampleOutputs:
        content: dict[str, Any] = {
            "multiview_cameras": cameras,
            "vision_files_by_camera": {
                camera: [str(path)] for camera, path in zip(cameras, written["vision"], strict=True)
            },
            "control_files_by_camera": {
                camera: [str(path)] for camera, path in zip(cameras, written["control"], strict=True)
            },
            "caption_chunks": result.chunks,
            "rolling_prompt": result.rolling_prompt,
        }
        if written["lidar"]:
            content["lidar_file"] = str(written["lidar"][0])
            content["lidar_range_projection"] = lidar_range_projection(pipe.model).to_dict()
        sample_output = SampleOutputs(
            args=sample.model_dump(mode="json"),
            outputs=[
                SampleOutput(
                    content=content,
                    files=[*written["vision"], *written["control"], *written["lidar"], *written["artifacts"]],
                )
            ],
        )
        sample_outputs_file = output_dir / "sample_outputs.json"
        sample_outputs_file.write_text(sample_output.model_dump_json())
        log.success(f"Saved causal multiview outputs to '{sample_outputs_file}'", rank0_only=False)
        return sample_output

    return _save_outputs_transactionally(output_dir, save_local, publish)
