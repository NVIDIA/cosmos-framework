# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Transfer inference pipeline for the Omni model."""

import math
import random
from dataclasses import dataclass
from pathlib import Path

import torch

from cosmos_framework.inference.args import (
    BlurTransferArgs,
    EdgeTransferArgs,
    OmniSampleArgs,
    PresetBlurStrength,
    PresetEdgeThreshold,
    TransferArgs,
    TransferHintKey,
)
from cosmos_framework.inference.vision import (
    pad_temporal_frames,
    read_and_resize_media,
    uint8_to_normalized_float,
)
from cosmos_framework.utils import log
from cosmos_framework.data.vfm.sequence_packing import SequencePlan
from cosmos_framework.model.vfm.omni_mot_model import OmniMoTModel
from cosmos_framework.model.vfm.vlm.qwen3_vl.utils import _SYSTEM_PROMPT_TRANSFER


@dataclass
class TransferGenerationOutput:
    output_video: torch.Tensor
    control_videos: dict[TransferHintKey, torch.Tensor]
    fps: float
    original_hw: tuple[int, int]


def _get_num_chunks(total_frames: int, frames_per_chunk: int, conditional_frames: int) -> tuple[int, int]:
    """Return ``(num_chunks, stride)`` for autoregressive chunking."""
    if frames_per_chunk <= 0:
        raise ValueError("frames_per_chunk must be positive")
    if total_frames <= frames_per_chunk:
        return 1, frames_per_chunk
    stride = frames_per_chunk - conditional_frames
    if stride <= 0:
        raise ValueError("num_conditional_frames must be smaller than num_video_frames_per_chunk")
    remaining = total_frames - frames_per_chunk
    extra_chunks = remaining // stride + (1 if remaining % stride else 0)
    return 1 + extra_chunks, stride


def apply_transfer_control_augmentor(
    input_frames: torch.Tensor,
    *,
    hint_key: TransferHintKey,
    preset_edge_threshold: PresetEdgeThreshold,
    preset_blur_strength: PresetBlurStrength,
) -> torch.Tensor:
    """Compute edge/blur transfer controls on the fly from uint8 input frames."""
    from cosmos_framework.data.vfm.augmentors.transfer_control_input.control_input import (
        AddControlInputBlur,
        AddControlInputEdge,
    )

    data_dict = {"input_video": input_frames}
    if hint_key == TransferHintKey.EDGE:
        augmentor = AddControlInputEdge(
            input_keys=["input_video"],
            output_keys=["control_input_edge"],
            use_random=False,
            preset_strength=preset_edge_threshold,
        )
    elif hint_key == TransferHintKey.BLUR:
        augmentor = AddControlInputBlur(
            input_keys=["input_video"],
            output_keys=["control_input_blur"],
            use_random=False,
            downup_preset=preset_blur_strength,
        )
    else:
        raise ValueError(f"On-the-fly control generation is unsupported for '{hint_key}'")
    output = augmentor(data_dict)
    return output[f"control_input_{hint_key}"]


def load_transfer_control_frames(
    *,
    hint_key: TransferHintKey,
    transfer: TransferArgs,
    resolution: str,
    aspect_ratio: str | None,
    max_frames: int | None,
    input_frames: torch.Tensor | None = None,
) -> torch.Tensor:
    """Load pre-computed control frames or compute edge/blur on the fly.

    When *input_frames* is provided, on-the-fly computation reuses those frames
    instead of re-reading from disk.
    """
    control_path = Path(transfer.control_path) if transfer.control_path else None
    if control_path is not None and control_path.exists():
        control_frames, _, _, _ = read_and_resize_media(
            control_path,
            resolution=resolution,
            aspect_ratio=aspect_ratio,
            max_frames=max_frames,
        )
        log.info(f"Loaded pre-computed {hint_key} control from {control_path}")
        return control_frames

    if hint_key not in {TransferHintKey.EDGE, TransferHintKey.BLUR}:
        raise FileNotFoundError(
            f"Missing pre-computed control input for '{hint_key}'. Provide a control_path in the transfer config."
        )

    if input_frames is None:
        raise ValueError(
            "input_frames must be provided for on-the-fly control computation when no control_path is specified."
        )

    if hint_key == TransferHintKey.EDGE:
        assert isinstance(transfer, EdgeTransferArgs)
        preset_edge_threshold = transfer.preset_edge_threshold
        preset_blur_strength = PresetBlurStrength.MEDIUM
    else:
        assert isinstance(transfer, BlurTransferArgs)
        preset_edge_threshold = PresetEdgeThreshold.MEDIUM
        preset_blur_strength = transfer.preset_blur_strength

    log.info(f"Computing {hint_key} control input on the fly")
    return apply_transfer_control_augmentor(
        input_frames,
        hint_key=hint_key,
        preset_edge_threshold=preset_edge_threshold,
        preset_blur_strength=preset_blur_strength,
    )


def build_transfer_batch(
    *,
    control_videos: list[torch.Tensor],
    target_video: torch.Tensor,
    num_frames: int,
    height: int,
    width: int,
    fps: float,
    num_conditional_frames: int,
    temporal_compression_factor: int,
    prompt_key: str,
    prompt: str,
    negative_prompt: str | None,
    share_vision_temporal_positions: bool,
) -> dict[str, object]:
    """Build the ``[ctrl_1, ..., ctrl_N, target]`` batch for transfer inference."""
    control_5ds = [cv.unsqueeze(0).cuda().to(dtype=torch.bfloat16) for cv in control_videos]
    target_5d = target_video.unsqueeze(0).cuda().to(dtype=torch.bfloat16)
    num_vision_items = len(control_5ds) + 1
    if num_conditional_frames > 0:
        condition_frame_indexes = list(range((num_conditional_frames - 1) // temporal_compression_factor + 1))
    else:
        condition_frame_indexes = []

    size = torch.tensor([[height, width, height, width]], dtype=torch.float32).cuda()
    batch: dict[str, object] = {
        "dataset_name": "video_transfer",
        "system_prompt": _SYSTEM_PROMPT_TRANSFER,
        "video": [*control_5ds, target_5d],
        "image_size": [size] * num_vision_items,
        "padding_mask": torch.zeros(1, 1, height, width).cuda(),
        "num_frames": torch.tensor([num_frames]).cuda(),
        "num_vision_items_per_sample": [num_vision_items],
        "is_preprocessed": True,
        # share_vision_temporal_positions must match the trained checkpoint's
        # SequencePlan regime; mismatched flag → frame-drift between control and
        # target. See projects/cosmos3/vfm/docs/transfer_temporal_id_fix.md.
        "sequence_plan": [
            SequencePlan(
                has_text=True,
                has_vision=True,
                condition_frame_indexes_vision=condition_frame_indexes,
                share_vision_temporal_positions=share_vision_temporal_positions,
            )
        ],
        "fps": torch.tensor([fps]).cuda(),
        "conditioning_fps": torch.tensor([fps]).cuda(),
        prompt_key: [prompt],
    }
    if negative_prompt:
        batch[f"neg_{prompt_key}"] = [negative_prompt]
    return batch


def generate_transfer_sample(
    sample_args: OmniSampleArgs,
    model: OmniMoTModel,
) -> TransferGenerationOutput:
    """Run autoregressive transfer inference for a single sample."""
    from cosmos_framework.inference.inference import _get_prompt_sample_data

    hints = sample_args.transfer_hints
    assert hints, "transfer_hints must be set (caller should check before this call)"

    if sample_args.resolution is None:
        raise ValueError("resolution is required for transfer inference")

    max_frames = sample_args.max_frames
    num_video_frames_per_chunk = sample_args.num_video_frames_per_chunk
    num_conditional_frames = sample_args.num_conditional_frames
    num_first_chunk_conditional_frames = sample_args.num_first_chunk_conditional_frames

    input_frames: torch.Tensor | None = None
    input_fps: float = 0
    original_hw: tuple[int, int] = (0, 0)

    if sample_args.vision_path is not None:
        input_frames, input_fps, detected_aspect_ratio, original_hw = read_and_resize_media(
            Path(sample_args.vision_path),
            resolution=sample_args.resolution,
            aspect_ratio=sample_args.aspect_ratio,
            max_frames=max_frames,
        )
        final_aspect_ratio = sample_args.aspect_ratio or detected_aspect_ratio
    else:
        # No vision_path — auto-detect aspect ratio from the first hint's pre-computed control.
        first_control = next((h.control_path for h in hints.values() if h.control_path is not None), None)
        assert first_control is not None, "_build_transfer_data should have rejected this case"
        _, _, final_aspect_ratio, original_hw = read_and_resize_media(
            Path(first_control),
            resolution=sample_args.resolution,
            aspect_ratio=None,
            max_frames=max_frames,
        )

    if num_first_chunk_conditional_frames > 0 and input_frames is None:
        raise ValueError("num_first_chunk_conditional_frames > 0 requires 'vision_path' for first-chunk conditioning")

    # Load control frames for each hint independently — no averaging.
    # Sequence layout: [text, ctrl_1_tokens, ..., ctrl_N_tokens, noisy_target_tokens]
    per_hint_frames: dict[TransferHintKey, torch.Tensor] = {
        hint_key: load_transfer_control_frames(
            hint_key=hint_key,
            transfer=transfer,
            resolution=sample_args.resolution,
            aspect_ratio=final_aspect_ratio,
            max_frames=max_frames,
            input_frames=input_frames,
        )
        for hint_key, transfer in hints.items()
    }

    first_frames = next(iter(per_hint_frames.values()))
    output_fps = input_fps if input_fps > 0 else float(sample_args.fps)
    height, width = first_frames.shape[2], first_frames.shape[3]

    total_frames = first_frames.shape[1]
    temporal_compression_factor = model.config.tokenizer.temporal_compression_factor
    chunk_frames = 1 if total_frames == 1 else num_video_frames_per_chunk
    chunk_frames = math.ceil((chunk_frames - 1) / temporal_compression_factor) * temporal_compression_factor + 1
    num_chunks, stride = _get_num_chunks(total_frames, chunk_frames, num_conditional_frames)

    per_hint_frames = {k: pad_temporal_frames(f, max(total_frames, chunk_frames)) for k, f in per_hint_frames.items()}
    if input_frames is not None:
        input_frames = pad_temporal_frames(input_frames, max(total_frames, chunk_frames))

    output_chunks: list[torch.Tensor] = []
    control_chunks_per_hint: dict[TransferHintKey, list[torch.Tensor]] = {k: [] for k in per_hint_frames}
    previous_output: torch.Tensor | None = None

    is_distilled = model.config.fixed_step_sampler_config is not None
    if is_distilled:
        sampler = model.fixed_step_sampler
        guidance = 1.0
    else:
        sampler = None
        guidance = sample_args.guidance

    prompt_sample_args = sample_args.model_copy(update={"num_frames": chunk_frames, "fps": int(round(output_fps))})
    chunk_prompt_data = _get_prompt_sample_data(prompt_sample_args, model, h=height, w=width, device="cuda")
    prompt = chunk_prompt_data[model.input_caption_key][0]
    negative_prompt = chunk_prompt_data.get("neg_" + model.input_caption_key, [None])[0]

    model.eval()
    seed = sample_args.seed if sample_args.seed is not None else random.randint(0, 10000)
    for chunk_id in range(num_chunks):
        start_frame = chunk_id * stride
        end_frame = min(start_frame + chunk_frames, total_frames)

        # Build normalised control tensor for each hint independently.
        control_norms: dict[TransferHintKey, torch.Tensor] = {
            hint_key: uint8_to_normalized_float(pad_temporal_frames(frames[:, start_frame:end_frame], chunk_frames))
            for hint_key, frames in per_hint_frames.items()
        }

        target_norm = torch.zeros_like(next(iter(control_norms.values())))
        current_conditional_frames = 0

        if chunk_id == 0 and num_first_chunk_conditional_frames > 0:
            assert input_frames is not None
            current_conditional_frames = min(num_first_chunk_conditional_frames, input_frames.shape[1])
            if current_conditional_frames > 0:
                input_cond = uint8_to_normalized_float(input_frames[:, :current_conditional_frames])
                target_norm[:, :current_conditional_frames] = input_cond
                if current_conditional_frames < chunk_frames:
                    fill_value = target_norm[:, current_conditional_frames - 1 : current_conditional_frames]
                    target_norm[:, current_conditional_frames:] = fill_value.expand(
                        -1,
                        chunk_frames - current_conditional_frames,
                        -1,
                        -1,
                    )
        elif chunk_id > 0 and previous_output is not None:
            current_conditional_frames = min(num_conditional_frames, previous_output.shape[2])
            if current_conditional_frames > 0:
                target_norm[:, :current_conditional_frames] = previous_output[0, :, -current_conditional_frames:]
                if current_conditional_frames < chunk_frames:
                    fill_value = target_norm[:, current_conditional_frames - 1 : current_conditional_frames]
                    target_norm[:, current_conditional_frames:] = fill_value.expand(
                        -1,
                        chunk_frames - current_conditional_frames,
                        -1,
                        -1,
                    )

        # `share_vision_temporal_positions` is populated by `_build_transfer_data`
        # via `_TRANSFER_SAMPLE_DEFAULTS` (default True) and may be overridden by
        # the input JSON. None should not reach here for a transfer sample, but
        # fall back to the post-fix default to keep behaviour predictable.
        share_temporal = sample_args.share_vision_temporal_positions
        if share_temporal is None:
            share_temporal = True

        data_batch = build_transfer_batch(
            control_videos=list(control_norms.values()),
            target_video=target_norm,
            num_frames=chunk_frames,
            height=height,
            width=width,
            fps=output_fps,
            num_conditional_frames=current_conditional_frames,
            temporal_compression_factor=temporal_compression_factor,
            prompt_key=model.input_caption_key,
            prompt=prompt,
            negative_prompt=negative_prompt,
            share_vision_temporal_positions=share_temporal,
        )
        outputs = model.generate_samples_from_batch(
            data_batch,
            sampler=sampler,
            guidance=guidance,
            guidance_interval=sample_args.guidance_interval,
            seed=[seed + chunk_id],
            n_sample=1,
            has_negative_prompt=negative_prompt is not None,
            num_steps=sample_args.num_steps,
            shift=sample_args.shift,
            sigma_max=sample_args.sigma_max,
            normalize_cfg=sample_args.normalize_cfg,
        )
        generated_latent = outputs["vision"][-1]
        output_video = model.decode(generated_latent).clamp(-1, 1).cpu()

        if chunk_id == 0:
            output_chunks.append(output_video)
            for hint_key, cn in control_norms.items():
                control_chunks_per_hint[hint_key].append(cn.unsqueeze(0).cpu())
        else:
            output_chunks.append(output_video[:, :, current_conditional_frames:])
            for hint_key, cn in control_norms.items():
                control_chunks_per_hint[hint_key].append(cn[:, current_conditional_frames:].unsqueeze(0).cpu())
        previous_output = output_video

    full_output = torch.cat(output_chunks, dim=2)[:, :, :total_frames]
    full_controls = {
        hint_key: torch.cat(chunks, dim=2)[:, :, :total_frames] for hint_key, chunks in control_chunks_per_hint.items()
    }

    if sample_args.show_control_condition:
        all_controls = torch.cat(list(full_controls.values()), dim=-1)
        full_output = torch.cat([all_controls, full_output], dim=-1)
    if sample_args.show_input and input_frames is not None:
        normalized_input = uint8_to_normalized_float(input_frames[:, :total_frames], dtype=torch.float32).unsqueeze(0)
        full_output = torch.cat([normalized_input, full_output], dim=-1)

    return TransferGenerationOutput(
        output_video=full_output,
        control_videos=full_controls,
        fps=output_fps,
        original_hw=original_hw,
    )
