# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Transfer inference pipeline for the Omni model."""

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

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
from cosmos_framework.data.generator.sequence_packing import SequencePlan
from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel
from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean
from cosmos_framework.model.generator.reasoner.qwen3_vl.utils import _SYSTEM_PROMPT_TRANSFER


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
        # Per-control weights for multi-control weighted attention aggregation.
        # Shape: [num_samples], each element is a list of floats (one per control).
        "control_weights": [control_weights],
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


def _build_no_control_inference_state(
    sequence_plans: list[SequencePlan],
    gen_data_clean: GenerationDataClean,
) -> tuple[list[SequencePlan], GenerationDataClean, list[int]] | None:
    """Build a target-only counterpart of ``(sequence_plans, gen_data_clean)`` for
    control-CFG. Drops all but the last vision item per sample (the target).

    Returns ``None`` when no sample has multiple vision items (nothing to drop).

    Also returns ``ctrl_dims_per_sample`` — the flattened control-token dimension
    per sample, used to slice ``noise_x`` and mix velocities.
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
) -> Optional[
    Callable[..., Optional[Callable[[list[torch.Tensor], list[torch.Tensor], torch.Tensor], list[torch.Tensor]]]]
]:
    """Return a ``velocity_postprocess_builder`` that injects control-CFG.

    Pass the returned builder to ``OmniMoTModel.generate_samples_from_batch``.
    The builder is invoked once at the start of sampling with the prepared
    inference state; it builds the alternate (target-only) state and returns a
    per-step closure that mixes the conditional velocity with an extra forward
    pass that has all control items dropped.

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
    ) -> Optional[Callable[[list[torch.Tensor], list[torch.Tensor], torch.Tensor], list[torch.Tensor]]]:
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

        def postprocess(
            cond_v_full: list[torch.Tensor],
            noise_x: list[torch.Tensor],
            timestep: torch.Tensor,
        ) -> list[torch.Tensor]:
            if control_guidance_bounds is not None:
                if not (control_guidance_bounds[0] < timestep[0].item() < control_guidance_bounds[1]):
                    return cond_v_full

            noise_x_nc = [nx[c:] for nx, c in zip(noise_x, ctrl_dims, strict=True)]  # [[N_target],...]
            cond_v_nc = model._get_velocity(
                net=net,
                noise_x=noise_x_nc,
                timestep=timestep,
                text_tokens=cond_tokens,
                sequence_plans=sp_nc,
                gen_data_clean=gdc_nc,
                skip_text_tokens=False,
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
                mixed_suffix = v_nc_i + control_guidance * (suffix_full - v_nc_i)  # [N_target]
                mixed.append(torch.cat([v_full_i[:c], mixed_suffix], dim=0))  # [N_full]
            return mixed

        return postprocess

    return builder


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
    )
