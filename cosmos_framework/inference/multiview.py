# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Multiview WSM inference: camera batch assembly, caption chunks, and output layout.

Shared pixel operations remain in ``cosmos_framework.utils.generator.multiview``.
RGB-only generation dispatches here; joint RGB/LiDAR generation uses
``inference.lidar_transfer`` with the same camera order.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any

import torch

from cosmos_framework.data.generator.augmentors.text_tokenizer import (
    _SYSTEM_PROMPT_AV_MULTIVIEW_TRANSFER,
    _SYSTEM_PROMPTS,
)
from cosmos_framework.data.generator.multiview.camera_attributes import CAMERA_ATTRIBUTES, CAMERA_VIEW_MAPPING
from cosmos_framework.data.generator.multiview.caption_chunks import (
    MultiviewCaptionChunk,
    caption_chunk_frame_count,
    load_multiview_caption_chunks,
)
from cosmos_framework.data.generator.multiview.caption_format import (
    DEFAULT_CAPTION_PREFIXES,
    first_caption_paragraph,
    format_multiview_caption,
    format_separate_view_captions,
    format_view_caption,
)
from cosmos_framework.data.generator.sequence_packing import SequencePlan
from cosmos_framework.inference.args import ConditionVisionMode, OmniSampleArgs
from cosmos_framework.inference.common.args import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
from cosmos_framework.inference.common.multiview_captions import exported_caption_task
from cosmos_framework.inference.vision import load_conditioning_image_pixels, load_conditioning_video
from cosmos_framework.model.generator.mot.replica_partition import ReplicaPartitioner, WorkUnit
from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel
from cosmos_framework.model.generator.reasoner.qwen3_vl.utils import _SYSTEM_PROMPT_TRANSFER

# Shared pixel-maths helpers — present after the next release sync.
# These imports will fail until the release that ships the full shared module; see module docstring.
from cosmos_framework.utils.generator.multiview import (
    build_camera_major_video,
    decode_multiview_latent_per_view,
    generated_multiview_condition_frames,
    load_multiview_media_pixels,
    load_multiview_media_units,
    normalize_multiview_control_weights,
    pad_multiview_view_video,
    slice_multiview_view_frames,
    split_multiview_video_by_view,
)


def _get_multiview_condition_pixel_frames(
    sample_args: OmniSampleArgs,
    model: OmniMoTModel,
    *,
    height: int,
    width: int,
    partitioner: ReplicaPartitioner | None = None,
) -> list[torch.Tensor | None]:
    """Return per-view conditioning pixel frames, or ``[None, ...]`` when no conditioning is set.

    Each element is either ``None`` (no conditioning for this view) or a CPU uint8 tensor of shape
    ``[3, T_cond, H, W]``, in view order.

    With a ``partitioner``, each camera is loaded on one rank of the inference replica and
    broadcast to the others, so every rank still returns every view. The load then enters
    collectives: every replica rank must call this with identical arguments.
    """
    assert sample_args.multiview is not None

    condition_mode = sample_args.multiview.condition_vision_mode
    if condition_mode is None:
        return [None for _ in sample_args.multiview.views]

    if sample_args.condition_frame_indexes_vision:
        num_condition_latent_frames = max(sample_args.condition_frame_indexes_vision) + 1
    else:
        num_condition_latent_frames = 1

    if condition_mode == ConditionVisionMode.VIDEO:
        max_frames = model.tokenizer_vision_gen.get_pixel_num_frames(
            num_condition_latent_frames,
            resolution=getattr(model.config, "resolution", None),
        )
    else:
        max_frames = 1

    # Every view is checked before the load enters a collective. The input is the same on every
    # replica rank, so each raises here before decoding anything, rather than only the rank that
    # owns the bad view raising mid-load.
    vision_paths: list[Path] = []
    for view in sample_args.multiview.views:
        assert view.vision_path is not None
        vision_paths.append(Path(view.vision_path))
    keep = sample_args.condition_video_keep or "first"

    def load_view(view_index: int) -> torch.Tensor:  # returns [3,T_cond,H,W]
        return load_multiview_media_pixels(
            vision_paths[view_index],
            target_h=height,
            target_w=width,
            max_frames=max_frames,
            keep=keep,
            video_loader=load_conditioning_video,
            image_loader=load_conditioning_image_pixels,
            video_extensions=VIDEO_EXTENSIONS,
            image_extensions=IMAGE_EXTENSIONS,
        )  # [3,T_cond,H,W]

    condition_frames: list[torch.Tensor | None] = list(
        load_multiview_media_units(
            [WorkUnit(run=partial(load_view, index)) for index in range(len(vision_paths))],
            partitioner=partitioner,
            label="camera condition media load",
        )
    )  # list[[3,T_cond,H,W]]
    return condition_frames


def _get_multiview_control_pixel_frames(
    sample_args: OmniSampleArgs,
    *,
    height: int,
    width: int,
    partitioner: ReplicaPartitioner | None = None,
) -> list[torch.Tensor]:
    """Return per-view control pixel frames for WSM-transfer inference.

    Each element is a CPU uint8 tensor of shape ``[3, T_control, H, W]``, in view order.
    Raises ``ValueError`` if any view is missing ``control_path``, before anything is loaded.

    With a ``partitioner``, each camera is loaded on one rank of the inference replica and
    broadcast to the others, so every rank still returns every view. The load then enters
    collectives: every replica rank must call this with identical arguments.
    """
    assert sample_args.multiview is not None

    # Every view is checked before the load enters a collective; see the condition loader.
    control_paths: list[Path] = []
    for view in sample_args.multiview.views:
        if view.control_path is None:
            raise ValueError("Multiview transfer inference requires control_path for every view.")
        control_paths.append(Path(view.control_path))

    def load_view(view_index: int) -> torch.Tensor:  # returns [3,T_control,H,W]
        return load_multiview_media_pixels(
            control_paths[view_index],
            target_h=height,
            target_w=width,
            max_frames=sample_args.num_frames,
            video_loader=load_conditioning_video,
            image_loader=load_conditioning_image_pixels,
            video_extensions=VIDEO_EXTENSIONS,
            image_extensions=IMAGE_EXTENSIONS,
        )  # [3,T_control,H,W]

    return load_multiview_media_units(
        [WorkUnit(run=partial(load_view, index)) for index in range(len(control_paths))],
        partitioner=partitioner,
        label="camera control media load",
    )  # list[[3,T_control,H,W]]


def _multiview_system_prompt(model: OmniMoTModel, *, view_prompts: list[str] | None) -> str:
    """The system prompt this checkpoint's captions were trained against.

    ``exported_caption_task`` returns ``None`` when the exporter could not determine the task --
    an older checkpoint, or streams that disagreed -- so fall back to the shape-based choice that
    predates the key rather than guessing a task.
    """
    task = exported_caption_task(getattr(model, "config", None))
    if task is not None and task in _SYSTEM_PROMPTS:
        return _SYSTEM_PROMPTS[task]
    return _SYSTEM_PROMPT_AV_MULTIVIEW_TRANSFER if view_prompts is not None else _SYSTEM_PROMPT_TRANSFER


def get_multiview_sample_data(
    sample_args: OmniSampleArgs,
    model: OmniMoTModel,
    *,
    device: Any = "cuda",
    condition_frames: list[torch.Tensor | None] | None = None,
    control_frames: list[torch.Tensor] | None = None,
    view_prompts: list[str] | None = None,
) -> dict[str, Any]:
    """Build the model-visible batch dict for multiview (WSM-transfer) inference.

    ``condition_frames`` and ``control_frames`` may be supplied by the caller (e.g. when
    running back-to-back AR chunks) or left as ``None`` to be loaded from
    ``sample_args.multiview.views``.

    Returns a dict whose keys match the training-time batch schema expected by ``OmniMoTModel``.

    NOTE: ``sample_args.multiview`` does not yet exist in CF's arg schema; see module docstring
    for the field shape this function expects and the list of wiring tasks that remain.
    """
    # Import here to avoid a potential import cycle; _get_prompt_sample_data is a private
    # helper in inference.py that couples to OmniSampleArgs + OmniMoTModel.  If this import
    # later causes a cycle, promote the call to a caller-supplied argument.
    from cosmos_framework.inference.inference import _get_prompt_sample_data

    assert sample_args.multiview is not None

    sample_n_views = len(sample_args.multiview.views)
    if sample_n_views < 1:
        raise ValueError("Multiview inference requires at least one camera view.")

    num_video_frames_per_view = sample_args.num_frames
    width, height = sample_args.vision_size

    if condition_frames is None:
        condition_frames = _get_multiview_condition_pixel_frames(
            sample_args,
            model,
            height=height,
            width=width,
        )
    if len(condition_frames) != sample_n_views:
        raise ValueError(f"Expected {sample_n_views} multiview condition frame tensors, got {len(condition_frames)}.")

    target_view_videos = [
        pad_multiview_view_video(frames, num_frames=num_video_frames_per_view, height=height, width=width)
        for frames in condition_frames
    ]  # list[[3,F,H,W]]
    target_video = build_camera_major_video(target_view_videos, device=device)  # [1,3,V*F,H,W]

    video_items = [target_video]
    is_transfer = bool(sample_args.transfer_hints)
    control_weights: list[float] | None = None

    if sample_args.multiview.has_control and not is_transfer:
        raise ValueError("Multiview transfer inference requires an explicit transfer hint such as wsm={}.")

    if is_transfer:
        if len(sample_args.transfer_hints) > 1:
            raise ValueError("Multiview transfer inference currently supports one transfer hint/control tensor.")
        control_weights = normalize_multiview_control_weights(
            [float(h.weight) for h in sample_args.transfer_hints.values()]
        )
        if not sample_args.multiview.has_control:
            raise ValueError("Multiview transfer inference requires control_path for every view.")
        if control_frames is None:
            control_frames = _get_multiview_control_pixel_frames(sample_args, height=height, width=width)
        if len(control_frames) != sample_n_views:
            raise ValueError(f"Expected {sample_n_views} multiview control frame tensors, got {len(control_frames)}.")
        control_view_videos = [
            pad_multiview_view_video(frames, num_frames=num_video_frames_per_view, height=height, width=width)
            for frames in control_frames
        ]  # list[[3,F,H,W]]
        control_video = build_camera_major_video(control_view_videos, device=device)  # [1,3,V*F,H,W]
        video_items = [control_video, target_video]

    has_condition_frames = [frames is not None for frames in condition_frames]
    if any(has_condition_frames) and not all(has_condition_frames):
        raise ValueError("All multiview camera views must provide conditioning frames, or none should.")
    condition_frame_indexes = list(sample_args.condition_frame_indexes_vision) if all(has_condition_frames) else []

    share_temporal = sample_args.share_vision_temporal_positions
    if share_temporal is None:
        share_temporal = True
    sequence_plan = SequencePlan(
        has_text=True,
        has_vision=True,
        condition_frame_indexes_vision=condition_frame_indexes,
        share_vision_temporal_positions=share_temporal,
    )

    size = torch.tensor([[height, width, height, width]], dtype=torch.float32, device=device)  # [1,4]
    flat_num_frames = sample_n_views * num_video_frames_per_view
    batch: dict[str, Any] = {
        "dataset_name": "video_transfer" if is_transfer else "video_data",
        "video": video_items,
        "image_size": [size] * len(video_items),
        "num_frames": torch.tensor([flat_num_frames], device=device),  # [B]
        "sample_n_views": torch.tensor([sample_n_views], dtype=torch.int64, device=device),  # [B]
        "num_video_frames_per_view": torch.tensor(
            [num_video_frames_per_view],
            dtype=torch.int64,
            device=device,
        ),  # [B]
        "enable_per_camera_vae_encoding": True,
        "sequence_plan": [sequence_plan],
        "fps": torch.tensor([float(sample_args.fps)], device=device),  # [B]
        "conditioning_fps": torch.tensor([float(sample_args.fps)], device=device),  # [B]
    }

    if is_transfer:
        assert control_weights is not None
        # Not cosmetic. An explicit batch["system_prompt"] overrides use_system_prompt inside
        # tokenize_caption, so this string is always part of what the model reads -- a mismatch
        # here changes the text token ids deterministically, for every sample, with no other
        # symptom. A checkpoint that packs one caption per camera was trained against the AV
        # multiview prompt; one reading a merged prompt was not.
        # When the export recorded which tokenizer task the per-view captions were trained under,
        # follow it. A joint camera+LiDAR checkpoint run on the RGB-only route is the case that
        # matters: it was trained against the joint prompt, and defaulting to the plain AV
        # multiview one silently shifts every text token id with no other symptom.
        batch["system_prompt"] = _multiview_system_prompt(model, view_prompts=view_prompts)
        batch["num_vision_items_per_sample"] = [len(video_items)]
        batch["control_weights"] = [control_weights]

    if getattr(model.config.diffusion_expert_config, "num_view_embeddings", 0):
        camera_keys = [view.camera_key for view in sample_args.multiview.views]
        unknown_keys = [key for key in camera_keys if key not in CAMERA_VIEW_MAPPING]
        if unknown_keys:
            raise ValueError(f"Rig view embeddings require supported camera keys: {unknown_keys}")
        # Preserve training identities for subsets and reordered camera views.
        batch["view_indices_selection"] = torch.tensor(
            [[CAMERA_VIEW_MAPPING[key] for key in camera_keys]], dtype=torch.int64, device=device
        )  # [1,V]

    batch |= _get_prompt_sample_data(sample_args, model, w=width, h=height, device=device, view_prompts=view_prompts)

    if is_transfer and sample_args.emphasize_control_in_prompt:
        hint_names = ", ".join(hint_key.value for hint_key in sample_args.transfer_hints) or "transfer"
        emphasis = (
            f" Follow the {hint_names} control videos precisely for every camera view: shape, contour,"
            f" position, and motion must align with the {hint_names} signal at every frame."
        )
        prompt_key = model.input_caption_key
        # Per-view captions nest one level, and every camera's caption gets the emphasis, so each
        # view's own text carries it into the tokens that view attends to.
        batch[prompt_key] = [
            [caption.rstrip() + emphasis for caption in prompt]
            if isinstance(prompt, list)
            else prompt.rstrip() + emphasis
            for prompt in batch[prompt_key]
        ]

    return batch


# --- MADS caption-chunk generation ------------------------------------------------
#
# Ported from the reference implementation (`_MultiviewCaptionChunk`, `_load_multiview_caption_chunks`,
# `_generate_multiview_caption_chunks_batch`). Before this existed, cosmos-framework
# declared `multiview.load_caption_from_data`, validated it, downloaded the caption
# JSON, and then ignored it: generation fell through to the one-shot builder with the
# sample's raw prompt, full `num_frames`, and the head of the unsliced control video.
# The run produced a different video from the reference implementation with no error at all.
#
# Generation mirrors `transfer.generate_transfer_sample`: a standalone chunk loop that
# calls `model.generate_samples_from_batch` directly rather than re-entering
# `generate_batch`.


# The parser lives in the released data module rather than here. It is the same code the reference implementation
# reads these files with, so the chunk boundaries and the caption text are identical by
# construction instead of by review -- and neither is something a wrong answer would show up as,
# since a chunk covering the wrong frames or carrying a different caption still generates a
# well-formed video.
#
# The previous local parser accepted only the legacy style-keyed list. Every caption file in the
# MADS eval bundles is a ``caption_structured`` object, so it raised
#   ValueError: Expected multiview caption JSON to contain a list, got dict.
# on all 1100 of them, and no multiview caption-chunk run against those bundles was possible.


def _label_view_captions(view_captions: Sequence[str], *, camera_keys: Sequence[str]) -> list[str | dict[str, Any]]:
    """Label each caption as the legacy merged-caption training dataloader does."""
    unlabeled = [key for key in camera_keys if key not in DEFAULT_CAPTION_PREFIXES]
    if unlabeled:
        raise ValueError(
            f"No caption prefix is defined for camera(s) {', '.join(unlabeled)}; captions can only be "
            f"assembled for {', '.join(sorted(DEFAULT_CAPTION_PREFIXES))}."
        )
    return [
        format_view_caption(
            caption=caption,
            camera_name=camera_key,
            view_index=view_index,
            add_view_prefix=True,
            camera_prefixes=DEFAULT_CAPTION_PREFIXES,
        )
        for view_index, (camera_key, caption) in enumerate(zip(camera_keys, view_captions, strict=True))
    ]


def _view_caption_texts(labeled_captions: Sequence[str | dict[str, Any]]) -> list[str]:
    """One caption string per camera, matching how the training tokenizer takes a caption payload.

    ``TextTokenizerTransform`` JSON-serializes a dict caption and passes a string through, so
    doing the same here keeps the per-view text identical to training's.
    """
    return [caption if isinstance(caption, str) else json.dumps(caption) for caption in labeled_captions]


def load_multiview_caption_chunks_per_view(
    views: Sequence[Any], *, first_caption_paragraph_only: bool = False
) -> list[MultiviewCaptionChunk]:
    """Read every camera's caption file and assemble one prompt per chunk.

    Each chunk carries both forms: ``prompt`` is the merged text a model reading one prompt gets,
    and ``view_prompts`` is one caption per camera for a checkpoint trained with
    ``separate_view_text_tokenization``. Which one is packed is decided at generation time from
    the checkpoint's own recorded layout, not here.

    ``first_caption_paragraph_only`` matches short-window LiDAR training. It trims the raw text
    before adding camera headers; the default preserves full captions.

    Chunkwise rollout generates all cameras together and slices the model output by one shared
    frame count, so the cameras have to agree on the chunk boundaries: a caption file written
    against a different frame span would silently describe frames other than the generated ones.
    """
    per_view_chunks: list[list[MultiviewCaptionChunk]] = []
    for view in views:
        if view.caption_path is None:
            raise ValueError(
                f"multiview.load_caption_from_data=True needs a caption for camera {view.camera_key!r}. "
                "Set its caption_path, multiview.caption_path, or multiview.caption_root."
            )
        per_view_chunks.append(load_multiview_caption_chunks(Path(view.caption_path)))

    chunk_counts = {view.camera_key: len(chunks) for view, chunks in zip(views, per_view_chunks, strict=True)}
    if len(set(chunk_counts.values())) > 1:
        raise ValueError(f"Cameras disagree on their caption chunk count: {chunk_counts}.")

    camera_keys = [view.camera_key for view in views]
    chunks: list[MultiviewCaptionChunk] = []
    for index, view_chunks in enumerate(zip(*per_view_chunks, strict=True)):
        spans = {(chunk.frame_start, chunk.frame_end) for chunk in view_chunks}
        if len(spans) > 1:
            raise ValueError(
                f"Caption chunk {index} covers a different frame span per camera: "
                f"{ {key: (chunk.frame_start, chunk.frame_end) for key, chunk in zip(camera_keys, view_chunks, strict=True)} }."
            )
        frame_start, frame_end = spans.pop()
        view_captions = [chunk.prompt for chunk in view_chunks]
        if first_caption_paragraph_only:
            view_captions = [first_caption_paragraph(caption) for caption in view_captions]
        labeled = _label_view_captions(view_captions, camera_keys=camera_keys)
        separate_view_captions = format_separate_view_captions(
            view_captions,
            camera_names=camera_keys,
            add_camera_rig_prefix=True,
            camera_attributes=CAMERA_ATTRIBUTES,
        )
        merged = format_multiview_caption(labeled, use_explicit_view_format=False, use_two_pass_format=False)
        chunks.append(
            MultiviewCaptionChunk(
                index=index,
                frame_start=frame_start,
                frame_end=frame_end,
                prompt=merged if isinstance(merged, str) else json.dumps(merged),
                view_prompts=_view_caption_texts(separate_view_captions),
            )
        )
    return chunks


@dataclass
class MultiviewCaptionChunksOutput:
    """Assembled result of a chunkwise multiview run."""

    vision_cthw: torch.Tensor
    """Camera-major generated video, ``[C, V*F_total, H, W]``."""
    control_video: torch.Tensor | None
    """Camera-major re-encoded control, ``[1, 3, V*F_total, H, W]``, or ``None``."""
    sample_n_views: int
    total_frames_per_view: int
    joined_prompt: str
    chunk_content: list[dict[str, Any]]


@torch.no_grad()
def generate_multiview_caption_chunks(
    sample_args: OmniSampleArgs,
    model: OmniMoTModel,
    *,
    debug_save: Any = None,
    per_view_captions: bool = False,
) -> MultiviewCaptionChunksOutput:
    """Run chunkwise multiview generation driven by a MADS caption file.

    Each chunk replaces the prompt with that chunk's caption, generates
    ``num_video_frames_per_chunk`` frames seeded at ``base_seed + chunk.index``, and
    conditions on the tail of the previous chunk. Control videos are sliced to the
    chunk's frame range. Chunks are then concatenated per view.

    ``debug_save`` is an optional ``callable(data_batch, name)`` used to dump per-chunk
    batches when the pipeline is in debug mode.
    """
    assert sample_args.multiview is not None
    if sample_args.control_guidance != 1.0:
        # This loop calls generate_samples_from_batch directly, so it never reaches the
        # velocity_postprocess_builder that generate_batch installs -- control_guidance was
        # accepted, validated, and then had no effect at all.
        #
        # Wiring the existing builder in would trade a silent no-op for a silently wrong answer:
        # _build_no_control_inference_state drops num_views_per_vision_item (and
        # temporal_positions_vision) when it builds the target-only state, so the extra forward
        # pass would treat an 11-camera vision item as a single view. The reference implementation's counterpart
        # carries both, plus the LiDAR stream. Until that is ported, refuse.
        #
        # 1.0 is the value both runtimes resolve for a multiview WSM sample (measured), and at
        # 1.0 control-CFG is a no-op by construction, so this turns nobody away today.
        raise NotImplementedError(
            f"control_guidance={sample_args.control_guidance} is not supported on the multiview "
            "caption-chunk path: cosmos-framework's control-CFG state builder is not view-aware, "
            "so the extra forward pass would treat a multi-camera item as a single view. Run this "
            "with control_guidance=1.0."
        )

    # A sample-level caption_path is the "one shared caption drives every view" case, which is how
    # lidar transfer works; otherwise each camera reads its own file and the chunk carries both
    # forms -- the merged prompt AND one caption per camera.
    #
    # `view_prompts` is populated but not yet consumed: the checkpoints that read it are still
    # refused by inference.py, and the builder that packs per-view text is a separate change.
    # Assembling it here first is deliberate -- it makes the text diffable against the reference implementation as
    # strings, before any generation is attempted.
    #
    # The merged prompt changes too, and that is the point: it is now the labelled, formatter-merged
    # text the reference implementation produces across all cameras, rather than the raw contents of the first
    # camera's file. Reading one camera's caption and presenting it as the whole rig's was wrong in
    # the same invisible way -- a plausible video, conditioned on text describing one view.
    first_caption_paragraph_only = sample_args.multiview.first_caption_paragraph_only
    if sample_args.multiview.caption_path is not None:
        caption_chunks = load_multiview_caption_chunks(Path(sample_args.multiview.caption_path))
        if first_caption_paragraph_only:
            caption_chunks = [replace(chunk, prompt=first_caption_paragraph(chunk.prompt)) for chunk in caption_chunks]
    elif all(view.caption_path is not None for view in sample_args.multiview.views):
        caption_chunks = load_multiview_caption_chunks_per_view(
            sample_args.multiview.views, first_caption_paragraph_only=first_caption_paragraph_only
        )
    else:
        raise ValueError(
            "multiview.load_caption_from_data=True requires a caption path: set multiview.caption_path, "
            "set caption_path on each camera, or call download() to resolve them."
        )

    if sample_args.multiview.use_first_chunk_only:
        caption_chunks = caption_chunks[:1]

    width, height = sample_args.vision_size
    sample_n_views = len(sample_args.multiview.views)
    # Captions are written against a fixed frame span, so the span decides the chunk length and
    # `num_frames` does not apply here. Deriving it -- rather than falling back to num_frames, or
    # to the generic 93-frame transfer chunk the arg defaults used to supply -- is what makes this
    # agree with the reference implementation: for the MADS bundles, whose captions all cover 0-300, it gives 297.
    # A wrong count generates a different amount of video from the same caption and raises nothing.
    chunk_frame_count = sample_args.num_video_frames_per_chunk or caption_chunk_frame_count(
        caption_chunks,
        tokenizer=model.tokenizer_vision_gen,
    )
    num_feedback_frames = sample_args.num_conditional_frames if sample_args.num_conditional_frames is not None else 1
    base_seed = sample_args.seed if sample_args.seed is not None else 0
    device = "cuda" if torch.cuda.is_available() else "cpu"

    is_distilled = model.config.fixed_step_sampler_config is not None
    sampler = model.fixed_step_sampler if is_distilled else None
    guidance = 1.0 if is_distilled else sample_args.guidance

    first_condition_frames = _get_multiview_condition_pixel_frames(sample_args, model, height=height, width=width)
    has_first_condition = all(frames is not None for frames in first_condition_frames)
    first_condition_indexes = list(sample_args.condition_frame_indexes_vision)
    if has_first_condition and not first_condition_indexes:
        first_condition_indexes = [0]

    control_source_frames: list[torch.Tensor] | None = None
    if sample_args.transfer_hints:
        max_caption_frame = max(chunk.frame_end for chunk in caption_chunks)
        control_source_frames = []
        for view in sample_args.multiview.views:
            if view.control_path is None:
                raise ValueError("Multiview transfer caption chunking requires control_path for every view.")
            control_source_frames.append(
                load_multiview_media_pixels(
                    Path(view.control_path),
                    target_h=height,
                    target_w=width,
                    max_frames=max_caption_frame,
                    video_loader=load_conditioning_video,
                    image_loader=load_conditioning_image_pixels,
                    video_extensions=VIDEO_EXTENSIONS,
                    image_extensions=IMAGE_EXTENSIONS,
                )
            )

    output_chunks_by_view: list[list[torch.Tensor]] = [[] for _ in range(sample_n_views)]
    control_chunks_by_view: list[list[torch.Tensor]] = [[] for _ in range(sample_n_views)]
    previous_view_videos: list[torch.Tensor] | None = None
    chunk_content: list[dict[str, Any]] = []

    for chunk in caption_chunks:
        if previous_view_videos is None:
            condition_frames = first_condition_frames
            condition_frame_indexes = first_condition_indexes
            current_condition_frame_count = max(condition_frame_indexes) + 1 if condition_frame_indexes else 0
        else:
            current_condition_frame_count = min(num_feedback_frames, chunk_frame_count)
            condition_frames = generated_multiview_condition_frames(
                previous_view_videos,
                num_condition_frames=current_condition_frame_count,
            )
            condition_frame_indexes = list(range(current_condition_frame_count))

        control_frames: list[torch.Tensor] | None = None
        padded_control_views: list[torch.Tensor] | None = None
        if control_source_frames is not None:
            control_frames = [
                slice_multiview_view_frames(frames, start_frame=chunk.frame_start, end_frame=chunk.frame_end)
                for frames in control_source_frames
            ]
            padded_control_views = [
                pad_multiview_view_video(frames, num_frames=chunk_frame_count, height=height, width=width)
                for frames in control_frames
            ]

        chunk_sample_args = sample_args.model_copy(
            deep=True,
            update={
                "prompt": chunk.prompt,
                "num_frames": chunk_frame_count,
                "condition_frame_indexes_vision": condition_frame_indexes,
                "seed": base_seed + chunk.index,
            },
        )
        chunk_data_batch = get_multiview_sample_data(
            chunk_sample_args,
            model,
            device=device,
            condition_frames=condition_frames,
            control_frames=control_frames,
            # The checkpoint decides, not the input: the chunk carries both forms, and which one
            # is packed comes from the layout the export recorded.
            view_prompts=chunk.view_prompts if per_view_captions else None,
        )
        if debug_save is not None:
            debug_save(chunk_data_batch, f"sample_data_chunk_{chunk.index:03d}")

        negative_prompt = chunk_data_batch.get("neg_" + model.input_caption_key, [None])[0]
        outputs = model.generate_samples_from_batch(
            chunk_data_batch,
            sampler=sampler,
            guidance=guidance,
            guidance_interval=chunk_sample_args.guidance_interval,
            seed=[chunk_sample_args.seed],
            n_sample=1,
            has_negative_prompt=negative_prompt is not None,
            num_steps=chunk_sample_args.num_steps,
            shift=chunk_sample_args.shift,
            sigma_max=chunk_sample_args.sigma_max,
            normalize_cfg=chunk_sample_args.normalize_cfg,
        )
        # Decode each camera's clip separately, as the one-shot multiview path does
        # (see inference.py, under the "enable_per_camera_vae_encoding" guard). The VAE drops
        # factor_t - 1 frames once per CALL -- get_pixel_num_frames(T) = (T-1)*4 + 1 -- so one
        # call over the whole camera-major latent returns 4*V*t - 3 frames while the split below
        # requires V*(4*t - 3). Those agree only at V=1; for 11 cameras at t=75 it is 3297 vs
        # 3267, and require_multiview_tensor_by_view raises.
        # The Wan decoder emits its native [-1, 1]; everything downstream of here -- the video
        # writer, the guardrail, and generated_multiview_condition_frames, which documents and
        # requires it -- works in [0, 1]. Clamping to [-1, 1] and stopping, as this did, hands a
        # [-1, 1] tensor to a [0, 1] consumer: every value the model put below 0.5 in [0, 1] terms
        # lands at or under zero and is crushed to black, which measured as 70.7% of all pixels on
        # an 11-camera run. The rest of this repo already normalises the same way at
        # inference.py:1863, :1870, :2429 and :2442; this path was the one that did not.
        vision_cthw = (
            (
                (
                    1.0
                    + decode_multiview_latent_per_view(
                        model.decode,
                        outputs["vision"][-1],
                        sample_n_views,
                        chunk_frame_count,
                    )
                )
                / 2
            )
            .clamp(0, 1)
            .cpu()[0]
        )  # [C,V*F,H,W]
        view_videos = split_multiview_video_by_view(
            vision_cthw,
            sample_n_views=sample_n_views,
            num_video_frames_per_view=chunk_frame_count,
        )

        keep_start = 0 if previous_view_videos is None else current_condition_frame_count
        for view_index, view_video in enumerate(view_videos):
            output_chunks_by_view[view_index].append(view_video[:, keep_start:])
        if padded_control_views is not None:
            for view_index, control_view in enumerate(padded_control_views):
                control_chunks_by_view[view_index].append(control_view[:, keep_start:])

        previous_view_videos = view_videos
        chunk_content.append(
            {
                "index": chunk.index,
                "frame_range": [chunk.frame_start, chunk.frame_end - 1],
                "prompt": chunk.prompt,
                "condition_frame_indexes_vision": condition_frame_indexes,
                "num_frames": chunk_frame_count,
            }
        )

    full_view_videos = [torch.cat(view_chunks, dim=1) for view_chunks in output_chunks_by_view]
    total_frames = int(full_view_videos[0].shape[1])
    full_vision = torch.cat(full_view_videos, dim=1)  # [C,V*F_total,H,W]

    control_video: torch.Tensor | None = None
    if control_chunks_by_view and control_chunks_by_view[0]:
        full_control_views = [torch.cat(view_chunks, dim=1) for view_chunks in control_chunks_by_view]
        control_video = build_camera_major_video(full_control_views, device=full_vision.device)

    return MultiviewCaptionChunksOutput(
        vision_cthw=full_vision,
        control_video=control_video,
        sample_n_views=sample_n_views,
        total_frames_per_view=total_frames,
        joined_prompt="\n\n".join(chunk.prompt for chunk in caption_chunks),
        chunk_content=chunk_content,
    )
