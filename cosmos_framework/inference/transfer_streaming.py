# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Causal Transfer inference with incrementally encoded control chunks."""

import math
from collections.abc import Callable, Iterator, Sequence
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

import av
import torch

from cosmos_framework.inference.args import OmniSampleArgs, TransferHintKey
from cosmos_framework.inference.inference import _get_prompt_sample_data
from cosmos_framework.inference.transfer import (
    TransferGenerationOutput,
    _latent_frames_for_pixel_frames,
    _require_causal_transfer_pyav,
    _resolve_transfer_seed,
    _warn_unused_length_and_fps,
    load_transfer_control_frames,
)
from cosmos_framework.inference.vision import (
    detect_aspect_ratio,
    pad_temporal_frames,
    probe_media_hw_fps,
    read_and_resize_media,
    uint8_to_normalized_float,
)
from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel
from cosmos_framework.model.generator.reasoner.qwen3_vl.utils import _SYSTEM_PROMPT_TRANSFER
from cosmos_framework.utils import log

# Longest trained framewise Transfer horizon; larger clips retain the same AR contract.
_MAX_TRAINED_AR_TRANSFER_LATENT_FRAMES = 51
_MAX_TRAINED_AR_TRANSFER_LATENT_FRAMES_CHUNKWISE = 151


@dataclass
class _StreamingTransferInput:
    """CPU-side inputs for one member of a streaming Transfer batch.

    Built by ``_prepare_streaming_transfer_input`` after validating the causal
    checkpoint and resolving media, prompts, and seed. ``control_frames`` holds
    one edge/blur/depth/seg stream as uint8 [0, 255] pixels, not encoded latents.
    ``input_frames`` is the optional source RGB preview in the same format,
    padded to at least ``total_frames``; it is not used as target-frame conditioning.
    ``total_frames``, ``height``, and ``width`` describe the control pixel stream,
    before any extra padding needed by the VAE's latent-chunk partition.

    ``sample_args`` retains the per-sample sampling/preview options; ``prompt``
    and ``negative_prompt`` are the formatted texts consumed by the model.
    ``seed`` is explicit or deterministically derived from the sample identity.
    ``output_fps`` uses the source RGB or precomputed control video's FPS when
    available, else the requested FPS; ``original_hw`` is the source size before
    resizing, for output saving.

    ``generate_transfer_batch`` moves and encodes only the current control chunk
    on the GPU. Treat this object as consumed after generation: materializing
    results clears ``control_frames`` to release its full-video CPU storage.
    """

    sample_args: OmniSampleArgs
    hint_key: TransferHintKey
    control_frames: torch.Tensor  # [3,T,H,W]
    input_frames: torch.Tensor | None  # [3,T_input,H,W] or None; T_input >= total_frames
    prompt: str
    negative_prompt: str | None
    total_frames: int
    height: int
    width: int
    output_fps: float
    original_hw: tuple[int, int]
    seed: int


@dataclass
class _StreamingTransferMedia:
    """Decoded CPU media needed to prepare one streaming Transfer sample.

    Shared by scheduling and input preparation so both use the actual decoded
    control dimensions. Pixels are uint8 [0, 255]; unlike ``_StreamingTransferInput``,
    the optional source video has not been temporally padded. This loading stage
    does not perform model-specific validation, prompt formatting, or seed resolution.
    """

    hint_key: TransferHintKey
    control_frames: torch.Tensor  # [3,T,H,W]
    input_frames: torch.Tensor | None  # [3,T_input,H,W] or None
    total_frames: int
    height: int
    width: int
    output_fps: float
    original_hw: tuple[int, int]


def _load_streaming_transfer_media(sample_args: OmniSampleArgs) -> _StreamingTransferMedia:
    """Decode the exact media whose resolved shape defines batch compatibility."""
    if getattr(sample_args, "transfer_resize_mode", "center_crop") == "direct_resize":
        raise ValueError(
            "Direct Transfer resizing requires condition_stream_encode=false; "
            "the batched condition-streaming path uses legacy spatial preprocessing"
        )
    if sample_args.resolution is None:
        raise ValueError("resolution is required for transfer inference")
    if len(sample_args.transfer_hints) != 1:
        raise ValueError(
            f"Streaming causal Transfer requires exactly one control hint; got {list(sample_args.transfer_hints)}"
        )

    _require_causal_transfer_pyav()
    log.info(f"[transfer] causal control decoder: PyAV {av.__version__}")
    hint_key, transfer = next(iter(sample_args.transfer_hints.items()))
    supported_hints = {
        TransferHintKey.EDGE,
        TransferHintKey.BLUR,
        TransferHintKey.DEPTH,
        TransferHintKey.SEG,
    }
    if hint_key not in supported_hints:
        raise ValueError(f"Streaming causal Transfer does not support control hint {hint_key.value!r}")

    input_frames: torch.Tensor | None = None  # [3,T,H,W] or None
    input_fps = 0.0
    original_hw = (0, 0)
    final_aspect_ratio = sample_args.aspect_ratio
    if sample_args.vision_path is not None:
        input_frames, input_fps, detected_aspect_ratio, original_hw = read_and_resize_media(
            Path(sample_args.vision_path),
            resolution=sample_args.resolution,
            aspect_ratio=sample_args.aspect_ratio,
            max_frames=sample_args.max_frames,
            video_decoder="pyav",
        )  # input_frames: [3,T,H,W]
        final_aspect_ratio = sample_args.aspect_ratio or detected_aspect_ratio
    else:
        if transfer.control_path is None:
            raise ValueError("Streaming Transfer requires a control_path when vision_path is omitted")
        original_hw, input_fps = probe_media_hw_fps(Path(transfer.control_path), video_decoder="pyav")
        final_aspect_ratio = sample_args.aspect_ratio or detect_aspect_ratio(original_hw[1], original_hw[0])

    control_frames = load_transfer_control_frames(
        hint_key=hint_key,
        transfer=transfer,
        resolution=sample_args.resolution,
        aspect_ratio=final_aspect_ratio,
        max_frames=sample_args.max_frames,
        input_frames=input_frames,
        video_decoder="pyav",
    )  # [3,T,H,W]
    total_frames = int(control_frames.shape[1])
    height, width = int(control_frames.shape[2]), int(control_frames.shape[3])
    output_fps = input_fps if input_fps > 0 else float(sample_args.fps)
    return _StreamingTransferMedia(
        hint_key=hint_key,
        control_frames=control_frames,
        input_frames=input_frames,
        total_frames=total_frames,
        height=height,
        width=width,
        output_fps=output_fps,
        original_hw=original_hw,
    )


def _prepare_streaming_transfer_input(
    sample_args: OmniSampleArgs,
    model: OmniMoTModel,
) -> _StreamingTransferInput:
    """Load and format one condition stream without encoding its full video."""
    if not sample_args.autoregressive:
        raise ValueError("condition_stream_encode requires autoregressive Transfer inference")
    if sample_args.num_first_chunk_conditional_frames != 0:
        raise ValueError("Streaming causal Transfer requires num_first_chunk_conditional_frames=0")
    if sample_args.control_guidance != 1.0:
        raise ValueError("Streaming causal Transfer requires control_guidance=1.0")
    if not sample_args.share_vision_temporal_positions:
        raise ValueError("Streaming causal Transfer requires share_vision_temporal_positions=true")

    strategy = getattr(model.config, "causal_training_strategy", None)
    simulation_mode = getattr(model.config, "simulation_mode", None)
    video_temporal_causal = bool(getattr(model.config, "video_temporal_causal", False))
    is_self_forcing = simulation_mode == "self_forcing" and video_temporal_causal
    # Student artifacts omit the training-only simulation mode.
    is_distilled_causal = video_temporal_causal and getattr(model.config, "fixed_step_sampler_config", None) is not None
    if strategy not in ("teacher_forcing", "teacher_forcing_dcm") and not (is_self_forcing or is_distilled_causal):
        raise ValueError(
            "Autoregressive transfer requires a teacher-forcing, self-forcing, or distilled causal checkpoint; "
            f"got causal_training_strategy={strategy!r}, simulation_mode={simulation_mode!r}, and "
            f"video_temporal_causal={video_temporal_causal!r}."
        )

    media = _load_streaming_transfer_media(sample_args)
    hint_key = media.hint_key
    control_frames = media.control_frames  # [3,T,H,W]
    input_frames = media.input_frames  # [3,T_input,H,W] or None
    total_frames = media.total_frames
    height = media.height
    width = media.width
    temporal_compression_factor = int(model.config.tokenizer.temporal_compression_factor)
    # Same check as the non-streaming path; an image source reports its sample fps, so it never warns.
    _warn_unused_length_and_fps(sample_args, total_frames, media.output_fps, temporal_compression_factor)
    chunk_size = int(getattr(model.config, "teacher_forcing_frames_per_chunk", 1) or 1)
    if input_frames is not None:
        input_frames = pad_temporal_frames(input_frames, total_frames)  # [3,T,H,W]

    latent_frames = _latent_frames_for_pixel_frames(total_frames, temporal_compression_factor)
    max_trained_latent_frames = (
        _MAX_TRAINED_AR_TRANSFER_LATENT_FRAMES_CHUNKWISE if chunk_size > 1 else _MAX_TRAINED_AR_TRANSFER_LATENT_FRAMES
    )
    if latent_frames > max_trained_latent_frames:
        log.warning(
            f"Requested {latent_frames} latent frames but AR transfer checkpoints were trained on clips of at "
            f"most {max_trained_latent_frames} latent frames; quality may degrade past the trained horizon."
        )

    output_fps = media.output_fps
    prompt_sample_args = sample_args.model_copy(
        update={
            "num_frames": total_frames,
            "fps": int(round(output_fps)),
            "autoregressive": False,
        }
    )
    prompt_data = _get_prompt_sample_data(prompt_sample_args, model, h=height, w=width, device="cuda")
    prompt = prompt_data[model.input_caption_key][0]
    negative_prompt = prompt_data.get("neg_" + model.input_caption_key, [None])[0]
    if sample_args.emphasize_control_in_prompt:
        prompt = (
            prompt.rstrip()
            + f" Follow the {hint_key.value} control video precisely: shape, contour, silhouette, position, and "
            f"motion of every visible structure must align with the {hint_key.value} signal at every frame."
        )
    log.info(f"[transfer] final user prompt: {prompt}")

    return _StreamingTransferInput(
        sample_args=sample_args,
        hint_key=hint_key,
        control_frames=control_frames,
        input_frames=input_frames,
        prompt=prompt,
        negative_prompt=negative_prompt,
        total_frames=total_frames,
        height=height,
        width=width,
        output_fps=output_fps,
        original_hw=media.original_hw,
        seed=_resolve_transfer_seed(sample_args),
    )


def _prepare_streaming_transfer_batch_inputs(
    sample_args_list: Sequence[OmniSampleArgs],
    model: OmniMoTModel,
) -> list[_StreamingTransferInput]:
    """Prepare every batch member before entering distributed model collectives."""
    prepared = [_prepare_streaming_transfer_input(sample_args, model) for sample_args in sample_args_list]
    _validate_streaming_transfer_batch_inputs(prepared)
    return prepared


def _validate_streaming_transfer_batch_inputs(prepared: Sequence[_StreamingTransferInput]) -> None:
    """Validate the batch contract before any rank enters model collectives."""
    if not prepared:
        return
    homogeneous_fields = ("hint_key", "total_frames", "height", "width")
    for field_name in homogeneous_fields:
        values = [getattr(item, field_name) for item in prepared]
        if any(value != values[0] for value in values[1:]):
            raise ValueError(f"Streaming Transfer batches require one {field_name}; got {values}")
    show_control_values = [item.sample_args.show_control_condition for item in prepared]
    show_input_values = [item.sample_args.show_input for item in prepared]
    if len(set(show_control_values)) != 1 or len(set(show_input_values)) != 1:
        raise ValueError("Streaming Transfer batch members must use identical show_control_condition/show_input")
    if show_input_values[0] and any(item.input_frames is None for item in prepared):
        raise ValueError("show_input requires vision_path for every streaming Transfer batch member")
    shared_sampling_fields = ("num_steps", "shift", "normalize_cfg", "guidance")
    for field_name in shared_sampling_fields:
        values = [getattr(item.sample_args, field_name) for item in prepared]
        if any(value != values[0] for value in values[1:]):
            raise ValueError(f"Streaming Transfer batches require one {field_name}; got {values}")


def _streaming_control_pixel_ranges(
    latent_frames: int,
    chunk_size: int,
    temporal_compression_factor: int,
) -> list[tuple[int, int]]:
    """Map the latent ``[1,C,C,...]`` partition to causal pixel chunks."""
    if latent_frames < 1:
        raise ValueError(f"latent_frames must be positive, got {latent_frames}")
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    ranges = [(0, 1)]
    for latent_start in range(1, latent_frames, chunk_size):
        latent_end = min(latent_start + chunk_size, latent_frames)
        if latent_end - latent_start != chunk_size:
            raise ValueError(
                f"Streaming causal Transfer requires a complete final latent chunk; got "
                f"[{latent_start}, {latent_end}) with C={chunk_size}"
            )
        pixel_start = 1 + (latent_start - 1) * temporal_compression_factor
        pixel_end = 1 + (latent_end - 1) * temporal_compression_factor
        ranges.append((pixel_start, pixel_end))
    return ranges


def generate_transfer_batch(
    *,
    sample_args_list: Sequence[OmniSampleArgs],
    model: OmniMoTModel,
    on_output_video_chunk: Callable[[torch.Tensor], None] | None = None,
    on_total_frames_resolved: Callable[[int], None] | None = None,
    materialize_outputs: bool = True,
    _prepared_inputs: Sequence[_StreamingTransferInput] | None = None,
) -> list[TransferGenerationOutput]:
    """Generate a homogeneous batch while VAE-encoding each condition chunk once.

    The non-streaming single-sample entry point remains separate so disabling
    ``condition_stream_encode`` retains its original execution exactly.
    """
    if not sample_args_list:
        return []
    prepared = (
        _prepare_streaming_transfer_batch_inputs(sample_args_list, model)
        if _prepared_inputs is None
        else list(_prepared_inputs)
    )
    if len(prepared) != len(sample_args_list):
        raise ValueError(
            f"Prepared Transfer input count {len(prepared)} does not match sample count {len(sample_args_list)}"
        )
    reference = prepared[0]
    show_control_values = [item.sample_args.show_control_condition for item in prepared]
    show_input_values = [item.sample_args.show_input for item in prepared]
    if on_total_frames_resolved is not None:
        on_total_frames_resolved(reference.total_frames)

    iter_ar = getattr(model, "iter_samples_from_batch_autoregressive_streaming_transfer", None)
    if not callable(iter_ar):
        raise TypeError(f"{type(model).__name__} does not implement batched streaming Transfer inference")
    use_cached_encoder = getattr(model.tokenizer_vision_gen, "use_cached_encoder", None)
    if not callable(use_cached_encoder):
        raise TypeError(f"{type(model.tokenizer_vision_gen).__name__} does not support cached condition encoding")
    temporal_compression_factor = int(model.config.tokenizer.temporal_compression_factor)
    latent_frames = _latent_frames_for_pixel_frames(reference.total_frames, temporal_compression_factor)
    chunk_size = int(getattr(model.config, "teacher_forcing_frames_per_chunk", 1) or 1)
    padded_latent_frames = 1 + math.ceil((latent_frames - 1) / chunk_size) * chunk_size
    pixel_ranges = _streaming_control_pixel_ranges(padded_latent_frames, chunk_size, temporal_compression_factor)
    batch_size = len(prepared)
    prompts = [item.prompt for item in prepared]
    negative_prompts = [item.negative_prompt for item in prepared]
    has_negative_prompt = any(prompt is not None for prompt in negative_prompts)
    data_batch: dict[str, object] = {
        # Keep the marker per request even when one model serves mixed workloads.
        "_preserve_transfer_numerics": all(
            item.hint_key in {TransferHintKey.EDGE, TransferHintKey.BLUR, TransferHintKey.DEPTH, TransferHintKey.SEG}
            for item in prepared
        ),
        "dataset_name": "video_transfer",
        "system_prompt": _SYSTEM_PROMPT_TRANSFER,
        model.input_caption_key: prompts,
        "fps": torch.tensor([item.output_fps for item in prepared], dtype=torch.float32),  # [B]
    }
    if has_negative_prompt:
        data_batch[f"neg_{model.input_caption_key}"] = [prompt or "" for prompt in negative_prompts]

    def control_latent_chunks() -> Iterator[torch.Tensor]:
        for pixel_start, pixel_end in pixel_ranges:
            pixels = torch.stack(
                [
                    pad_temporal_frames(
                        item.control_frames[:, pixel_start : min(pixel_end, item.total_frames)],
                        pixel_end - pixel_start,
                    )
                    for item in prepared
                ],
                dim=0,
            ).cuda()  # [B,3,T_pixel,H,W]
            pixels = uint8_to_normalized_float(pixels)  # [B,3,T_pixel,H,W]
            yield model.encode(pixels)  # [B,C_latent,T_latent,H_latent,W_latent]

    full_output: torch.Tensor | None = None  # [B,3,T,H,W] or None
    pending_output_chunks: list[torch.Tensor] = []
    decoded_frame_count = 0

    def decode_clean_chunk(vision_latent: torch.Tensor) -> None:  # [B,C_latent,T_latent,H_latent,W_latent]
        nonlocal decoded_frame_count, full_output
        decoded = model.decode(vision_latent).clamp(-1, 1).cpu()  # [B,3,T_pixel,H,W]
        frame_start = decoded_frame_count
        remaining_frames = reference.total_frames - decoded_frame_count
        decoded = decoded[:, :, :remaining_frames]  # [B,3,T_stream,H,W]
        if decoded.shape[2] == 0:
            return
        decoded_frame_count += int(decoded.shape[2])
        if materialize_outputs:
            if full_output is None:
                full_output = decoded.new_empty(
                    (batch_size, decoded.shape[1], reference.total_frames, decoded.shape[3], decoded.shape[4])
                )  # [B,3,T,H,W]
            full_output[:, :, frame_start:decoded_frame_count].copy_(decoded)  # [B,3,T_stream,H,W]
        if on_output_video_chunk is None:
            return
        output_chunk = decoded
        if show_control_values[0]:
            control_chunk = torch.stack(
                [
                    uint8_to_normalized_float(item.control_frames[:, frame_start:decoded_frame_count]).cpu()
                    for item in prepared
                ],
                dim=0,
            )  # [B,3,T_stream,H,W]
            output_chunk = torch.cat([control_chunk, output_chunk], dim=-1)  # [B,3,T_stream,H,2W]
        if show_input_values[0]:
            input_chunk = torch.stack(
                [
                    uint8_to_normalized_float(item.input_frames[:, frame_start:decoded_frame_count]).cpu()
                    for item in prepared
                    if item.input_frames is not None
                ],
                dim=0,
            )  # [B,3,T_stream,H,W]
            output_chunk = torch.cat([input_chunk, output_chunk], dim=-1)  # [B,3,T_stream,H,W_out]
        pending_output_chunks.append(output_chunk)

    model.eval()
    sampler_mode = "distilled" if model.config.fixed_step_sampler_config is not None else "rf"
    guidance = 1.0 if sampler_mode == "distilled" else reference.sample_args.guidance
    with (
        model.tokenizer_vision_gen.use_cached_encoder(),
        model.tokenizer_vision_gen.use_cached_decoder(),
        closing(
            iter_ar(
                data_batch=data_batch,
                control_latent_chunks=control_latent_chunks(),
                num_frames=padded_latent_frames,
                seeds=[item.seed for item in prepared],
                guidance=guidance,
                num_steps=reference.sample_args.num_steps,
                shift=reference.sample_args.shift,
                normalize_cfg=reference.sample_args.normalize_cfg,
                sampler_mode=sampler_mode,
                on_clean_vision_chunk=decode_clean_chunk,
                has_negative_prompt=has_negative_prompt,
            )
        ) as ar_chunks,
    ):
        for _ in ar_chunks:
            if on_output_video_chunk is not None and pending_output_chunks:
                for output_chunk in pending_output_chunks:
                    on_output_video_chunk(output_chunk)
                pending_output_chunks.clear()
    if on_output_video_chunk is not None and pending_output_chunks:
        for output_chunk in pending_output_chunks:
            on_output_video_chunk(output_chunk)
        pending_output_chunks.clear()
    if decoded_frame_count == 0:
        raise RuntimeError("Streaming causal Transfer inference produced no target chunks")
    if decoded_frame_count != reference.total_frames:
        raise RuntimeError(
            f"Streaming causal Transfer produced {decoded_frame_count} decoded frames; "
            f"expected {reference.total_frames}"
        )
    if not materialize_outputs:
        return []
    if full_output is None:
        raise RuntimeError("Streaming causal Transfer did not materialize its target video")

    outputs: list[TransferGenerationOutput] = []
    for sample_idx, item in enumerate(prepared):
        output_video = full_output[sample_idx : sample_idx + 1]  # [1,3,T,H,W]
        control_video = uint8_to_normalized_float(item.control_frames).unsqueeze(0).cpu()  # [1,3,T,H,W]
        item.control_frames = torch.empty(0, dtype=torch.uint8)  # [0]
        if item.sample_args.show_control_condition:
            output_video = torch.cat([control_video, output_video], dim=-1)  # [1,3,T,H,2W]
        if item.sample_args.show_input:
            assert item.input_frames is not None
            normalized_input = uint8_to_normalized_float(item.input_frames).unsqueeze(0).cpu()  # [1,3,T,H,W]
            output_video = torch.cat([normalized_input, output_video], dim=-1)  # [1,3,T,H,W_out]
        outputs.append(
            TransferGenerationOutput(
                output_video=output_video,
                control_videos={item.hint_key: control_video},
                fps=item.output_fps,
                original_hw=item.original_hw,
            )
        )
    return outputs
