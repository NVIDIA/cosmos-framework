# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Transfer inference pipeline for the Omni model."""

import math
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, cast

import av
import torch
from torchvision.transforms import InterpolationMode

from cosmos_framework.data.generator.sequence_packing import SequencePlan
from cosmos_framework.inference.args import (
    BlurTransferArgs,
    EdgeTransferArgs,
    OmniSampleArgs,
    PresetBlurStrength,
    PresetEdgeThreshold,
    TransferArgs,
    TransferHintKey,
)
from cosmos_framework.inference.inference import _fallback_seed, _get_prompt_sample_data
from cosmos_framework.inference.transfer_preprocessing import (
    TransferGeometry,
    TransferMedia,
    TransferPixelModality,
    read_direct_transfer_media,
    select_transfer_geometry,
    validate_transfer_media_alignment,
)
from cosmos_framework.inference.vision import (
    VideoDecoderBackend,
    detect_aspect_ratio,
    pad_temporal_frames,
    probe_media_hw,
    probe_media_hw_fps,
    read_and_resize_media,
    uint8_to_normalized_float,
)
from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel, VelocityPostprocess
from cosmos_framework.model.generator.reasoner.qwen3_vl.utils import _SYSTEM_PROMPT_TRANSFER
from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean
from cosmos_framework.utils import log


@dataclass
class TransferGenerationOutput:
    output_video: torch.Tensor
    control_videos: dict[TransferHintKey, torch.Tensor]
    fps: float
    original_hw: tuple[int, int]
    spatial_geometry: TransferGeometry | None = None


@dataclass(frozen=True)
class DirectTransferInputs:
    """Validated media prepared before distributed model collectives begin."""

    input_frames: torch.Tensor | None  # [3,T,H,W] or None
    control_frames: dict[TransferHintKey, torch.Tensor]  # Each [3,T,H,W]
    fps: float
    geometry: TransferGeometry


def _require_causal_transfer_pyav() -> None:
    """Require the decoder major used to establish causal Transfer parity."""
    version = str(av.__version__)
    major = version.partition(".")[0]
    if not major.isdecimal() or int(major) != 18:
        raise RuntimeError(
            "Causal Transfer inference requires PyAV >=18,<19 for bit-exact control decoding; "
            f"found PyAV {version}. Use Python >=3.11 and reinstall the project dependencies."
        )


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


def _latent_frames_for_pixel_frames(pixel_frames: int, temporal_compression_factor: int) -> int:
    """Return the latent count ``T = 1 + ceil((F - 1) / tcf)`` produced by the VAE."""
    return 1 + math.ceil((pixel_frames - 1) / temporal_compression_factor)


def _align_ar_frames_to_chunk_partition(
    total_frames: int,
    temporal_compression_factor: int,
    frames_per_chunk: int,
) -> int:
    """Trim to the largest pixel-frame count valid for ``[1, C, C, ...]`` AR."""
    if total_frames < 1:
        raise ValueError(f"total_frames must be positive, got {total_frames}")
    if frames_per_chunk <= 1:
        return total_frames
    latent_frames = _latent_frames_for_pixel_frames(total_frames, temporal_compression_factor)
    if latent_frames < 1 + frames_per_chunk:
        raise ValueError(
            "Chunkwise AR transfer inference requires at least one full latent chunk after the "
            f"first latent frame: need T >= 1 + C, got T={latent_frames} latent frames "
            f"({total_frames} pixel frames) with C={frames_per_chunk}. Provide a longer input."
        )
    aligned_latent_frames = 1 + ((latent_frames - 1) // frames_per_chunk) * frames_per_chunk
    if aligned_latent_frames == latent_frames:
        return total_frames
    return 1 + (aligned_latent_frames - 1) * temporal_compression_factor


def apply_transfer_control_augmentor(
    input_frames: torch.Tensor,
    *,
    hint_key: TransferHintKey,
    preset_edge_threshold: PresetEdgeThreshold,
    preset_blur_strength: PresetBlurStrength,
) -> torch.Tensor:
    """Compute edge/blur transfer controls on the fly from uint8 input frames."""
    from cosmos_framework.data.generator.augmentors.transfer_control_input.control_input import (
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
    video_decoder: VideoDecoderBackend = "torchcodec",
) -> torch.Tensor:
    """Load pre-computed control frames or compute edge/blur on the fly.

    When *input_frames* is provided, on-the-fly computation reuses those frames
    instead of re-reading from disk.
    """
    control_path = Path(transfer.control_path) if transfer.control_path else None
    if control_path is not None and control_path.exists():
        # Match training's categorical segmentation resize without changing legacy crop geometry.
        interpolation = InterpolationMode.NEAREST if hint_key == TransferHintKey.SEG else InterpolationMode.BILINEAR
        control_frames, _, _, _ = read_and_resize_media(
            control_path,
            resolution=resolution,
            aspect_ratio=aspect_ratio,
            max_frames=max_frames,
            video_decoder=video_decoder,
            interpolation=interpolation,
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
    control_weights: list[float] | None = None,
) -> dict[str, object]:
    """Build the ``[ctrl_1, ..., ctrl_N, target]`` batch for transfer inference.

    ``control_weights`` is a per-control scalar (default 1.0 each).  Weights are
    normalised to sum to 1 before use.  In ``multi_control_two_way_attention`` N
    independent maskless SDPA passes are computed (one per control), each with
    KV = [text | ctrl_i | noisy].  The final noisy output is the weighted sum:

        noisy_out = w_1 * noisy_out_1 + ... + w_N * noisy_out_N

    All SDPA calls are maskless so Flash Attention is always active.
    When ``N=1`` the single weight normalises to 1.0, reproducing the original
    ``two_way_attention`` behaviour exactly.
    """
    control_5ds = [cv.unsqueeze(0).cuda().to(dtype=torch.bfloat16) for cv in control_videos]
    target_5d = target_video.unsqueeze(0).cuda().to(dtype=torch.bfloat16)
    num_vision_items = len(control_5ds) + 1
    if num_conditional_frames > 0:
        condition_frame_indexes = list(range((num_conditional_frames - 1) // temporal_compression_factor + 1))
    else:
        condition_frame_indexes = []

    if control_weights is None:
        control_weights = [1.0] * len(control_5ds)
    assert len(control_weights) == len(control_5ds), (
        f"control_weights length {len(control_weights)} must match number of controls {len(control_5ds)}"
    )
    assert all(w >= 0 for w in control_weights), f"control_weights must all be non-negative, got {control_weights}"
    total = sum(control_weights)
    assert total > 0, f"control_weights must have a positive sum, got {control_weights}"
    control_weights = [w / total for w in control_weights]

    size = torch.tensor([[height, width, height, width]], dtype=torch.float32).cuda()
    batch: dict[str, object] = {
        "dataset_name": "video_transfer",
        "system_prompt": _SYSTEM_PROMPT_TRANSFER,
        "video": [*control_5ds, target_5d],
        "image_size": [size] * num_vision_items,
        "padding_mask": torch.zeros(1, 1, height, width).cuda(),
        "num_frames": torch.tensor([num_frames]).cuda(),
        "num_vision_items_per_sample": [num_vision_items],
        # The control and target are separate single-view items, matching the
        # native Transfer batch layout consumed by replay-aware attention.
        "num_views_per_vision_item": [1] * num_vision_items,
        # Per-control weights for multi-control weighted attention aggregation.
        # Shape: [num_samples], each element is a list of floats (one per control).
        "control_weights": [control_weights],
        "is_preprocessed": True,
        # share_vision_temporal_positions must match the trained checkpoint's
        # SequencePlan regime; mismatched flag → frame-drift between control and
        # target; both streams must use aligned temporal positions.
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


def _build_no_control_inference_state(
    sequence_plans: list[SequencePlan],
    gen_data_clean: GenerationDataClean,
) -> tuple[list[SequencePlan], GenerationDataClean, list[int]] | None:
    """Build a target-only counterpart of ``(sequence_plans, gen_data_clean)`` for
    control-CFG. Drops all but the last vision item per sample (the target).

    Returns ``None`` when no sample has multiple vision items (nothing to drop).

    Also returns ``ctrl_dims_per_sample`` — the flattened control-token dimension
    per sample, used to slice ``noise_x`` and mix velocities.

    NOT VIEW-AWARE. The rebuilt ``GenerationDataClean`` carries neither
    ``num_views_per_vision_item`` nor ``temporal_positions_vision``, so the target-only forward
    pass reads a multi-camera vision item as a single view; there is no LiDAR stream here either,
    which the reference implementation's counterpart keeps alongside vision. Single-view transfer is unaffected,
    since the view count is 1 either way. The multiview caption-chunk path refuses
    ``control_guidance != 1.0`` rather than reach this (see ``multiview.py``); lifting that
    refusal means porting the reference implementation's ``_build_no_control_inference_state`` together with its
    ``_keep_last_items`` / ``_target_ranges_for_streams`` / ``_scatter_mix_flat_ranges`` helpers,
    none of which exist here.
    """
    num_items_per_sample = gen_data_clean.num_vision_items_per_sample
    if num_items_per_sample is None or all(n <= 1 for n in num_items_per_sample):
        return None

    assert gen_data_clean.x0_tokens_vision is not None

    new_x0_tokens_vision: list[torch.Tensor] = []
    new_raw_state_vision: list[torch.Tensor] | None = [] if gen_data_clean.raw_state_vision is not None else None
    ctrl_dims_per_sample: list[int] = []
    vis_offset = 0
    for n_vis in num_items_per_sample:
        ctrl_dim_i = 0
        for j in range(n_vis - 1):
            sh = gen_data_clean.x0_tokens_vision[vis_offset + j].shape
            ctrl_dim_i += math.prod(sh)
        ctrl_dims_per_sample.append(ctrl_dim_i)
        tgt_idx = vis_offset + n_vis - 1
        new_x0_tokens_vision.append(gen_data_clean.x0_tokens_vision[tgt_idx])
        if new_raw_state_vision is not None:
            new_raw_state_vision.append(gen_data_clean.raw_state_vision[tgt_idx])  # type: ignore[index]
        vis_offset += n_vis

    gdc_nc = GenerationDataClean(
        batch_size=gen_data_clean.batch_size,
        is_image_batch=gen_data_clean.is_image_batch,
        raw_state_vision=new_raw_state_vision,
        x0_tokens_vision=new_x0_tokens_vision,
        fps_vision=gen_data_clean.fps_vision,
        num_vision_items_per_sample=None,
        raw_state_action=gen_data_clean.raw_state_action,
        x0_tokens_action=gen_data_clean.x0_tokens_action,
        action_domain_id=gen_data_clean.action_domain_id,
        fps_action=gen_data_clean.fps_action,
        raw_action_dim=gen_data_clean.raw_action_dim,
        raw_state_sound=gen_data_clean.raw_state_sound,
        x0_tokens_sound=gen_data_clean.x0_tokens_sound,
        fps_sound=gen_data_clean.fps_sound,
    )

    sp_nc = [
        SequencePlan(
            has_text=sp.has_text,
            has_vision=sp.has_vision,
            condition_frame_indexes_vision=sp.condition_frame_indexes_vision,
            share_vision_temporal_positions=False,
            has_action=sp.has_action,
            condition_frame_indexes_action=sp.condition_frame_indexes_action,
            action_start_frame_offset=sp.action_start_frame_offset,
            has_sound=sp.has_sound,
            condition_frame_indexes_sound=sp.condition_frame_indexes_sound,
        )
        for sp in sequence_plans
    ]

    return sp_nc, gdc_nc, ctrl_dims_per_sample


def build_control_cfg_postprocess(
    *,
    control_guidance: float,
    control_guidance_interval: Optional[list[float]] = None,
) -> Callable[..., VelocityPostprocess | None] | None:
    """Return a ``velocity_postprocess_builder`` that injects control-CFG.

    Pass the returned builder to ``OmniMoTModel.generate_samples_from_batch``.
    The builder is invoked once at the start of sampling with the prepared
    inference state; it builds the alternate (target-only) state and returns a
    per-step closure that mixes the conditional velocity with an extra forward
    pass that has all control items dropped. The returned ``VelocityPostprocess``
    declares that extra branch to the sampler for diffusion-cache bookkeeping
    and FSDP forward-count alignment.

    Returns ``None`` when control-CFG is a no-op (``control_guidance == 1.0``),
    so the model takes its fast single-forward path.
    """
    if control_guidance == 1.0:
        return None

    def builder(
        *,
        model: OmniMoTModel,
        net: torch.nn.Module | None = None,
        cond_tokens: list[list[int]],
        sequence_plans: list[SequencePlan],
        gen_data_clean: GenerationDataClean,
    ) -> VelocityPostprocess | None:
        nc_state = _build_no_control_inference_state(sequence_plans, gen_data_clean)
        if nc_state is None:
            log.warning(
                "control_guidance != 1.0 but no multi-vision sample found; falling back to single-branch inference."
            )
            return None

        if any(sp.has_action or sp.has_sound for sp in sequence_plans):
            raise ValueError("control_guidance currently supports video transfer only, not action/sound generation.")

        sp_nc, gdc_nc, ctrl_dims = nc_state
        control_guidance_bounds: tuple[float, float] | None = None
        if control_guidance_interval is not None:
            if len(control_guidance_interval) != 2:
                raise ValueError(f"control_guidance_interval must be [lo, hi], got {control_guidance_interval}")
            control_guidance_bounds = (control_guidance_interval[0], control_guidance_interval[1])

        def control_active(timestep: torch.Tensor) -> bool:  # timestep: [B,1]
            return control_guidance_bounds is None or (
                control_guidance_bounds[0] < timestep[0].item() < control_guidance_bounds[1]
            )

        def cfg_branches(timestep: torch.Tensor) -> tuple[str, ...]:  # timestep: [B,1]
            # Declare the no-control forward while active. Changing the branch set
            # resets all diffusion-cache histories at the interval boundary.
            return ("cond_no_control",) if control_active(timestep) else ()

        def postprocess(
            cond_v_full: list[torch.Tensor],
            noise_x: list[torch.Tensor],
            timestep: torch.Tensor,
            text_guidance_scale: float,
        ) -> list[torch.Tensor]:
            if not control_active(timestep):
                return cond_v_full

            if text_guidance_scale == 0.0:
                raise ValueError(
                    "control_guidance != 1.0 cannot be composed with text guidance scale 0.0 because "
                    "the text-CFG blend would discard every conditional control increment."
                )

            # Text CFG subsequently multiplies every change to the conditional
            # branch by its guidance scale. Compensate here so the final control
            # increment remains exactly (control_guidance - 1) * (full - no-control).
            effective_control_guidance = 1.0 + (control_guidance - 1.0) / text_guidance_scale

            noise_x_nc = [nx[c:] for nx, c in zip(noise_x, ctrl_dims, strict=True)]  # [[N_target],...]
            cond_v_nc = model._get_velocity(
                net=net,
                noise_x=noise_x_nc,
                timestep=timestep,
                text_tokens=cond_tokens,
                sequence_plans=sp_nc,
                gen_data_clean=gdc_nc,
                skip_text_tokens=False,
                has_noisy_actions=False,  # control-CFG is video-only (action/sound rejected above)
            )

            # Mix only the suffix (target vision). The control-token portion
            # of cond_v_full is already zeroed by the model's velocity mask
            # (control items are fully conditioned), so leave it untouched.
            mixed: list[torch.Tensor] = []
            for v_full_i, v_nc_i, c in zip(cond_v_full, cond_v_nc, ctrl_dims, strict=True):
                suffix_full = v_full_i[c:]  # [N_target]
                assert suffix_full.shape == v_nc_i.shape, (
                    f"shape mismatch in control-CFG mix: full suffix {suffix_full.shape} vs no-control {v_nc_i.shape}"
                )
                mixed_suffix = v_nc_i + effective_control_guidance * (suffix_full - v_nc_i)  # [N_target]
                mixed.append(torch.cat([v_full_i[:c], mixed_suffix], dim=0))  # [N_full]
            return mixed

        return VelocityPostprocess(apply=postprocess, cfg_branches=cfg_branches)

    return builder


def _generate_causal_transfer_sample(
    *,
    sample_args: OmniSampleArgs,
    model: OmniMoTModel,
    per_hint_frames: dict[TransferHintKey, torch.Tensor],
    input_frames: torch.Tensor | None,
    prompt: str,
    negative_prompt: str | None,
    total_frames: int,
    height: int,
    width: int,
    output_fps: float,
    original_hw: tuple[int, int],
    temporal_compression_factor: int,
    seed: int,
    on_output_video_chunk: Callable[[torch.Tensor], None] | None = None,
) -> TransferGenerationOutput:
    """Generate one causal target and optionally stream CPU chunks in final-output layout."""
    if len(per_hint_frames) != 1:
        raise ValueError(
            "Causal transfer AR inference requires exactly one control hint because the "
            f"checkpoint was trained with [control, target]; got {list(per_hint_frames)}."
        )
    if sample_args.num_first_chunk_conditional_frames != 0:
        raise ValueError(
            "Causal transfer AR inference currently supports target generation from noise only; "
            "set num_first_chunk_conditional_frames=0."
        )
    if sample_args.control_guidance != 1.0:
        raise ValueError("Causal transfer AR inference requires control_guidance=1.0.")
    strategy = getattr(model.config, "causal_training_strategy", None)
    simulation_mode = getattr(model.config, "simulation_mode", None)
    video_temporal_causal = bool(getattr(model.config, "video_temporal_causal", False))
    is_self_forcing = simulation_mode == "self_forcing" and video_temporal_causal
    # Student exports project onto the causal base config, which intentionally
    # omits training-only simulation_mode. Its causal flag and fixed sampler
    # preserve the inference contract without requiring the distillation class.
    is_distilled_causal = video_temporal_causal and getattr(model.config, "fixed_step_sampler_config", None) is not None
    if strategy not in ("teacher_forcing", "teacher_forcing_dcm") and not (is_self_forcing or is_distilled_causal):
        raise ValueError(
            "Autoregressive transfer requires a teacher-forcing, self-forcing, or distilled causal checkpoint; "
            f"got causal_training_strategy={strategy!r}, simulation_mode={simulation_mode!r}, and "
            f"video_temporal_causal={video_temporal_causal!r}."
        )

    hint_key, control_frames = next(iter(per_hint_frames.items()))
    supported_hints = {
        TransferHintKey.EDGE,
        TransferHintKey.BLUR,
        TransferHintKey.DEPTH,
        TransferHintKey.SEG,
    }
    if hint_key not in supported_hints:
        raise ValueError(
            f"This causal transfer checkpoint supports edge, blur, depth, and seg controls; got {hint_key.value!r}."
        )

    if not sample_args.share_vision_temporal_positions:
        raise ValueError(
            "Causal transfer AR inference requires share_vision_temporal_positions=true to match training."
        )

    iter_ar = getattr(model, "iter_samples_from_batch_autoregressive", None)
    if not callable(iter_ar):
        raise TypeError(f"{type(model).__name__} does not implement causal autoregressive inference")

    # The VAE requires 1 + n * temporal_compression_factor pixel frames.
    # Pad only the encoder inputs; decoded outputs retain the requested length.
    encoded_frames = (
        1
        + (_latent_frames_for_pixel_frames(total_frames, temporal_compression_factor) - 1) * temporal_compression_factor
    )
    # Direct depth resizing retains fractional float32 pixels, as in training.
    # Keep that precision until the tokenizer; preserve the legacy path's dtype.
    control_norm = uint8_to_normalized_float(  # [C,T,H,W]
        pad_temporal_frames(control_frames[:, :total_frames], encoded_frames),
        dtype=(
            torch.float32
            if getattr(sample_args, "transfer_resize_mode", "center_crop") == "direct_resize"
            else torch.bfloat16
        ),
    )
    target_norm = torch.zeros_like(control_norm)  # [C,T,H,W]
    data_batch = build_transfer_batch(
        control_videos=[control_norm],
        target_video=target_norm,
        num_frames=encoded_frames,
        height=height,
        width=width,
        fps=output_fps,
        num_conditional_frames=0,
        temporal_compression_factor=temporal_compression_factor,
        prompt_key=model.input_caption_key,
        prompt=prompt,
        negative_prompt=negative_prompt,
        share_vision_temporal_positions=True,
        control_weights=[1.0],
    )
    # Internal request metadata, after edge/blur/depth/seg validation above.
    # Shared models retain their normal path on every other request.
    data_batch["_preserve_transfer_numerics"] = True

    decoded_chunks: list[torch.Tensor] = []
    pending_output_chunks: list[torch.Tensor] = []
    pending_host_copies: list[torch.cuda.Event] = []
    decoded_frame_count = 0

    def to_host_async(tensor: torch.Tensor) -> torch.Tensor:  # tensor: [1,3,T,H,W]
        """Start a pinned, non-blocking device-to-host copy; ``wait_for_host_copies`` completes it.

        The synchronous ``.cpu()`` it replaces stalled the CPU for the whole VAE
        decode plus a pageable copy on every latent frame; the AR loop now
        keeps issuing GPU work while the frame drains to the host.
        """
        if not tensor.is_cuda:
            return tensor.cpu()
        host = torch.empty(tensor.shape, dtype=tensor.dtype, device="cpu", pin_memory=True)  # [1,3,T,H,W]
        host.copy_(tensor, non_blocking=True)
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(tensor.device))
        pending_host_copies.append(event)
        return host

    def wait_for_host_copies() -> None:
        while pending_host_copies:
            pending_host_copies.pop().synchronize()

    def decode_clean_chunk(vision_latent: torch.Tensor) -> None:  # vision_latent: [1,C,1,H_lat,W_lat]
        nonlocal decoded_frame_count
        decoded = to_host_async(model.decode(vision_latent).clamp(-1, 1))  # [1,3,T_pixel,H,W]
        frame_start = decoded_frame_count
        remaining_frames = total_frames - decoded_frame_count
        decoded = decoded[:, :, :remaining_frames]  # [1,3,T_stream,H,W]
        if decoded.shape[2] == 0:
            return
        decoded_chunks.append(decoded)
        decoded_frame_count += decoded.shape[2]
        if on_output_video_chunk is not None:
            wait_for_host_copies()  # streaming consumers read the pixels right away
            output_chunk = decoded
            if sample_args.show_control_condition:
                control_chunk = control_norm[:, frame_start:decoded_frame_count].unsqueeze(0).cpu()  # [1,3,T,H,W]
                output_chunk = torch.cat([control_chunk, output_chunk], dim=-1)  # [1,3,T,H,2W]
            if sample_args.show_input and input_frames is not None:
                input_chunk = uint8_to_normalized_float(  # [1,3,T,H,W]
                    input_frames[:, frame_start:decoded_frame_count], dtype=torch.float32
                ).unsqueeze(0)
                output_chunk = torch.cat([input_chunk.cpu(), output_chunk], dim=-1)  # [1,3,T,H,W_out]
            pending_output_chunks.append(output_chunk)

    sampler_mode = "distilled" if model.config.fixed_step_sampler_config is not None else "rf"
    guidance = 1.0 if sampler_mode == "distilled" else sample_args.guidance
    with (
        model.tokenizer_vision_gen.use_cached_decoder(),
        closing(
            iter_ar(
                data_batch=data_batch,
                mode="video_transfer",
                guidance=guidance,
                seed=seed,
                num_steps=sample_args.num_steps,
                shift=sample_args.shift,
                normalize_cfg=sample_args.normalize_cfg,
                sampler_mode=sampler_mode,
                on_clean_vision_chunk=decode_clean_chunk,
                has_negative_prompt=f"neg_{model.input_caption_key}" in data_batch,
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
    if not decoded_chunks:
        raise RuntimeError("Causal transfer AR inference produced no target chunks")

    wait_for_host_copies()
    full_output = torch.cat(decoded_chunks, dim=2)[:, :, :total_frames]  # [1,3,T,H,W]
    full_controls = {hint_key: control_norm[:, :total_frames].unsqueeze(0).cpu()}  # {hint: [1,3,T,H,W]}
    if sample_args.show_control_condition:
        full_output = torch.cat([full_controls[hint_key], full_output], dim=-1)  # [1,3,T,H,2W]
    if sample_args.show_input and input_frames is not None:
        normalized_input = uint8_to_normalized_float(  # [1,3,T,H,W]
            input_frames[:, :total_frames], dtype=torch.float32
        ).unsqueeze(0)
        full_output = torch.cat([normalized_input.cpu(), full_output], dim=-1)  # [1,3,T,H,W_out]

    return TransferGenerationOutput(
        output_video=full_output,
        control_videos=full_controls,
        fps=output_fps,
        original_hw=original_hw,
    )


def _resolve_transfer_seed(sample_args: OmniSampleArgs) -> int:
    """Return an explicit seed or a stable fallback derived from sample identity.

    Every distributed rank prepares the same logical sample independently, so a
    process-local random draw can give ranks different initial noise. Reusing the
    generic inference fallback keeps seedless generation aligned across ranks.
    """
    if sample_args.seed is not None:
        return sample_args.seed
    return _fallback_seed(sample_args)


def _load_direct_transfer_inputs(
    sample_args: OmniSampleArgs,
) -> DirectTransferInputs:
    """Load aligned RGB/control pixels under the new teacher's full-frame geometry.

    Returns RGB [3,T,H,W] or None, controls with that same spatial shape, the
    decoded video FPS, and the shared source-to-model geometry.
    """
    if (sample_args.num_first_chunk_conditional_frames or 0) > 0 and sample_args.vision_path is None:
        raise ValueError("num_first_chunk_conditional_frames > 0 requires 'vision_path' for first-chunk conditioning")
    hints = sample_args.transfer_hints
    source_path = sample_args.vision_path or next(
        (hint.control_path for hint in hints.values() if hint.control_path is not None), None
    )
    if source_path is None:
        raise ValueError("Direct Transfer resizing requires a source RGB or precomputed control")
    assert sample_args.resolution is not None
    geometry = select_transfer_geometry(
        probe_media_hw(Path(source_path), video_decoder="pyav"),
        resolution=sample_args.resolution,
        aspect_ratio=sample_args.aspect_ratio,
    )
    input_media: TransferMedia | None = None
    if sample_args.vision_path is not None:
        input_media = read_direct_transfer_media(
            Path(sample_args.vision_path),
            geometry=geometry,
            modality="rgb",
            max_frames=sample_args.max_frames,
        )
    input_frames = None if input_media is None else input_media.frames  # [3,T,H,W] or None
    reference_media = input_media
    per_hint_frames: dict[TransferHintKey, torch.Tensor] = {}  # Each [3,T,H,W]
    for hint_key, transfer in hints.items():
        if transfer.control_path is not None:
            media = read_direct_transfer_media(
                Path(transfer.control_path),
                geometry=geometry,
                modality=cast(TransferPixelModality, hint_key.value),
                max_frames=sample_args.max_frames,
            )
            if reference_media is not None:
                validate_transfer_media_alignment(reference_media, media)
            if reference_media is None or not reference_media.is_video:
                reference_media = media
            per_hint_frames[hint_key] = media.frames  # [3,T,H,W]
        else:
            # Edge/blur augmentors must see the same directly resized uint8 RGB
            # as training, including the same image boundary and field of view.
            # build_sample validates OmniSampleArgs controls as TransferArgs
            # subclasses; the shared property's union also covers unbuilt overrides.
            per_hint_frames[hint_key] = load_transfer_control_frames(  # [3,T,H,W]
                hint_key=hint_key,
                transfer=cast(TransferArgs, transfer),
                resolution=sample_args.resolution,
                aspect_ratio=geometry.aspect_ratio,
                max_frames=sample_args.max_frames,
                input_frames=input_frames,
            )
    if reference_media is None:
        raise ValueError("Direct Transfer resizing loaded no media")
    frame_counts = {frames.shape[1] for frames in per_hint_frames.values()}
    if len(frame_counts) != 1:
        raise ValueError(f"Transfer controls have inconsistent frame counts: {sorted(frame_counts)}")
    log.info(
        f"[transfer] direct resize: source={geometry.original_hw}, target={geometry.target_hw}, "
        f"aspect={geometry.aspect_ratio}, distortion={geometry.distortion:.4%}, fps={reference_media.fps}"
    )
    return DirectTransferInputs(input_frames, per_hint_frames, reference_media.fps, geometry)


def _warn_unused_length_and_fps(
    sample_args: OmniSampleArgs, total_frames: int, input_fps: float, temporal_compression_factor: int
) -> None:
    """Warn when a requested num_frames or fps differs from what this sample actually uses.

    The length comes from the decoded control (capped by max_frames) and the frame rate from the
    source video; sample fps is only the fallback for image sources, which report 0 fps.
    """
    # Stand-ins and samples rebuilt from JSON carry no request record; stay silent for them.
    requested = getattr(sample_args, "requested_fields", None)
    if not isinstance(requested, frozenset):
        return
    if "fps" in requested and input_fps > 0 and abs(sample_args.fps - input_fps) >= 0.5:
        log.warning(
            f"[transfer] 'fps'={sample_args.fps} has no effect with a video input; the output keeps the input "
            f"video's {input_fps:g} fps. 'fps' is used only for image inputs."
        )
    # num_frames was rounded up to the latent grid when the sample was built; compare on that grid.
    length = math.ceil((total_frames - 1) / temporal_compression_factor) * temporal_compression_factor + 1
    if "num_frames" in requested and total_frames > 1 and sample_args.num_frames != length:
        log.warning(
            f"[transfer] 'num_frames'={sample_args.num_frames} does not set the Transfer output length; the "
            f"length comes from the control video ({total_frames} frames after the 'max_frames' cap). "
            "Set 'max_frames' to change it."
        )


def generate_transfer_sample(
    sample_args: OmniSampleArgs,
    model: OmniMoTModel,
    *,
    on_total_frames_resolved: Callable[[int], None] | None = None,
    prepared_direct_inputs: DirectTransferInputs | None = None,
) -> TransferGenerationOutput:
    """Run causal or chunked transfer inference with the trained prompt metadata."""

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
    # Preserve native decoded pixel values for the causal student; other models
    # retain the shared TorchCodec reader.
    video_decoder: VideoDecoderBackend = "pyav" if sample_args.autoregressive else "torchcodec"
    if video_decoder == "pyav":
        _require_causal_transfer_pyav()
        log.info(f"[transfer] causal control decoder: PyAV {av.__version__}")

    spatial_geometry: TransferGeometry | None = None
    direct_resize = getattr(sample_args, "transfer_resize_mode", "center_crop") == "direct_resize"
    if direct_resize:
        prepared = prepared_direct_inputs or _load_direct_transfer_inputs(sample_args)
        input_frames = prepared.input_frames  # [3,T,H,W] or None
        per_hint_frames = prepared.control_frames
        input_fps = prepared.fps
        spatial_geometry = prepared.geometry
        original_hw = spatial_geometry.original_hw
        final_aspect_ratio = spatial_geometry.aspect_ratio
    else:
        if sample_args.vision_path is not None:
            input_frames, input_fps, detected_aspect_ratio, original_hw = read_and_resize_media(
                Path(sample_args.vision_path),
                resolution=sample_args.resolution,
                aspect_ratio=sample_args.aspect_ratio,
                max_frames=max_frames,
                video_decoder=video_decoder,
            )
            final_aspect_ratio = sample_args.aspect_ratio or detected_aspect_ratio
        else:
            # No vision_path — derive source metadata from the first precomputed control.
            first_control = next((h.control_path for h in hints.values() if h.control_path is not None), None)
            assert first_control is not None, "_build_transfer_data should have rejected this case"
            # Preserve the precomputed video's clock, just as for source RGB.
            # The control pixels are decoded below by ``load_transfer_control_frames``.
            original_hw, input_fps = probe_media_hw_fps(Path(first_control), video_decoder=video_decoder)
            final_aspect_ratio = sample_args.aspect_ratio or detect_aspect_ratio(original_hw[1], original_hw[0])

        if num_first_chunk_conditional_frames > 0 and input_frames is None:
            raise ValueError(
                "num_first_chunk_conditional_frames > 0 requires 'vision_path' for first-chunk conditioning"
            )

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
                video_decoder=video_decoder,
            )
            for hint_key, transfer in hints.items()
        }

    first_frames = next(iter(per_hint_frames.values()))
    output_fps = input_fps if input_fps > 0 else float(sample_args.fps)
    height, width = first_frames.shape[2], first_frames.shape[3]

    total_frames = first_frames.shape[1]
    temporal_compression_factor = model.config.tokenizer.temporal_compression_factor
    _warn_unused_length_and_fps(sample_args, total_frames, input_fps, temporal_compression_factor)
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

    if sample_args.autoregressive:
        # Align before formatting the prompt so its duration matches generated frames.
        frames_per_chunk = int(getattr(model.config, "teacher_forcing_frames_per_chunk", 1) or 1)
        aligned_total_frames = _align_ar_frames_to_chunk_partition(
            total_frames, temporal_compression_factor, frames_per_chunk
        )
        if aligned_total_frames != total_frames:
            log.warning(
                f"Trimming AR transfer input from {total_frames} to {aligned_total_frames} pixel frames "
                f"to match the chunkwise latent partition (C={frames_per_chunk}, "
                f"temporal_compression_factor={temporal_compression_factor})."
            )
            total_frames = aligned_total_frames

    if on_total_frames_resolved is not None:
        on_total_frames_resolved(total_frames)

    prompt_sample_args = sample_args.model_copy(
        update={
            "num_frames": total_frames if sample_args.autoregressive else chunk_frames,
            "fps": int(round(output_fps)),
            # Transfer training includes duration/FPS/resolution metadata. The
            # generic AR prompt path suppresses it, so format this prompt as a
            # non-AR sample while retaining the full transfer duration.
            "autoregressive": False,
        }
    )
    chunk_prompt_data = _get_prompt_sample_data(prompt_sample_args, model, h=height, w=width, device="cuda")
    prompt = chunk_prompt_data[model.input_caption_key][0]
    negative_prompt = chunk_prompt_data.get("neg_" + model.input_caption_key, [None])[0]

    # Optionally append a one-sentence control-adherence directive to the user prompt.
    # Names the active hint modality (e.g. "edge", "depth, seg") so the VLM gets the
    # exact control type. System prompt is untouched (training-distribution safe).
    if sample_args.emphasize_control_in_prompt:
        hint_names = ", ".join(k.value for k in hints.keys())
        prompt = (
            prompt.rstrip() + f" Follow the {hint_names} control video precisely: shape, contour, silhouette,"
            f" position, and motion of every visible structure must align with the {hint_names}"
            f" signal at every frame."
        )
    log.info(f"[transfer] final user prompt: {prompt}")

    model.eval()
    seed = _resolve_transfer_seed(sample_args)
    if sample_args.autoregressive:
        output = _generate_causal_transfer_sample(
            sample_args=sample_args,
            model=model,
            per_hint_frames=per_hint_frames,
            input_frames=input_frames,
            prompt=prompt,
            negative_prompt=negative_prompt,
            total_frames=total_frames,
            height=height,
            width=width,
            output_fps=output_fps,
            original_hw=original_hw,
            temporal_compression_factor=temporal_compression_factor,
            seed=seed,
        )
        output.spatial_geometry = spatial_geometry
        return output
    for chunk_id in range(num_chunks):
        start_frame = chunk_id * stride
        end_frame = min(start_frame + chunk_frames, total_frames)

        # Build normalised control tensor for each hint independently.
        control_norms: dict[TransferHintKey, torch.Tensor] = {}
        for hint_key, frames in per_hint_frames.items():
            chunk = pad_temporal_frames(frames[:, start_frame:end_frame], chunk_frames)  # [C,T,H,W]
            if direct_resize:
                control_norms[hint_key] = (chunk.float() / 127.5 - 1).to(dtype=torch.bfloat16)
            else:
                control_norms[hint_key] = uint8_to_normalized_float(chunk)

        target_norm = torch.zeros_like(next(iter(control_norms.values())))
        current_conditional_frames = 0

        if chunk_id == 0 and num_first_chunk_conditional_frames > 0:
            assert input_frames is not None
            current_conditional_frames = min(num_first_chunk_conditional_frames, input_frames.shape[1])
            if current_conditional_frames > 0:
                input_cond = uint8_to_normalized_float(
                    input_frames[:, :current_conditional_frames],
                    dtype=torch.float32 if direct_resize else torch.bfloat16,
                )
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
            control_weights=[h.weight for h in hints.values()],
        )
        outputs = model.generate_samples_from_batch(
            data_batch,
            sampler=sampler,
            guidance=guidance,
            guidance_interval=sample_args.guidance_interval,
            velocity_postprocess_builder=build_control_cfg_postprocess(
                control_guidance=sample_args.control_guidance,
                control_guidance_interval=sample_args.control_guidance_interval,
            ),
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
        spatial_geometry=spatial_geometry,
    )
