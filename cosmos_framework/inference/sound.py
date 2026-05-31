# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Shared sound inference helpers used by text2video, video2video, and text2videosound scripts."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from cosmos_framework.data.vfm.sequence_packing import SequencePlan
from cosmos_framework.data.vfm.sound_data_utils import build_sequence_plan_for_sound


@dataclass
class AudioTokenizerInfo:
    """Consolidated info about a model's sound tokenizer (None-safe)."""

    tokenizer: Any | None
    sample_rate: int
    max_audio_latent_t: int | None
    audio_latent_fps: float | None

    @property
    def has_sound(self) -> bool:
        return self.tokenizer is not None


def get_audio_tokenizer_info(model: Any) -> AudioTokenizerInfo:
    """Probe a model for its sound tokenizer and return consolidated info."""
    tokenizer = getattr(model, "tokenizer_sound_gen", None)
    if tokenizer is None:
        return AudioTokenizerInfo(tokenizer=None, sample_rate=48000, max_audio_latent_t=None, audio_latent_fps=None)
    return AudioTokenizerInfo(
        tokenizer=tokenizer,
        sample_rate=getattr(tokenizer, "sample_rate", 48000),
        max_audio_latent_t=getattr(model.config, "max_audio_latent_t", None),
        audio_latent_fps=float(model.config.sound_latent_fps) if hasattr(model.config, "sound_latent_fps") else None,
    )


def create_placeholder_audio(
    num_frames: int,
    conditioning_fps: float,
    audio_info: AudioTokenizerInfo,
) -> torch.Tensor:
    """Build a zero-filled audio placeholder matching the video duration.

    Used in t2vs mode so the model knows the target sound length after encoding.

    Returns:
        Audio tensor of shape (1, C, N).
    """
    sr = audio_info.sample_rate
    video_duration_sec = num_frames / conditioning_fps
    sound_num_samples = int(video_duration_sec * sr)
    sound_channels = getattr(audio_info.tokenizer, "audio_channels", 2)
    return torch.zeros(1, sound_channels, sound_num_samples)  # [1,C_audio,N_samples]


def inject_sound_into_batch(
    data_batch: dict[str, Any],
    audio_tensor: torch.Tensor | None,
    model: Any,
) -> dict[str, Any]:
    """Add sound data and upgrade the SequencePlan in an existing data batch.

    This is the core wiring function: it takes a batch that was built for
    video-only generation and adds the sound modality to it.

    Args:
        data_batch: Existing data batch (from get_video_sample_batch or build_conditioned_video_batch).
        audio_tensor: Audio waveform tensor (1, C, N) or None.
        model: The OmniMoTModel instance.

    Returns:
        The same data_batch dict, mutated in-place with sound fields added.
    """
    if audio_tensor is not None:
        assert audio_tensor.ndim == 3 and audio_tensor.shape[0] == 1, (
            f"Expected audio_tensor of shape (1, C, N), got {audio_tensor.shape}"
        )
        batch_size = len(data_batch["video"])
        data_batch["sound"] = [audio_tensor[0]] * batch_size

    # Capture existing vision conditioning before overwriting the plan.
    # Callers like build_conditioned_video_batch set condition_frame_indexes_vision
    # for image2video / video2video; we must preserve that.
    existing_vision_cond: list[int] | None = None
    existing_plans = data_batch.get("sequence_plan")
    if existing_plans and isinstance(existing_plans, list) and len(existing_plans) > 0:
        existing_vision_cond = list(existing_plans[0].condition_frame_indexes_vision)

    has_sound = audio_tensor is not None and getattr(model, "tokenizer_sound_gen", None) is not None
    if has_sound:
        video_list = data_batch["video"]
        video_t = video_list[0].shape[2] if video_list[0].ndim == 5 else video_list[0].shape[1]
        temporal_cf = model.tokenizer_vision_gen.temporal_compression_factor
        video_latent_t = max(1, video_t // temporal_cf)
        sound_sample = data_batch["sound"][0]
        sound_latent_t = int(model.tokenizer_sound_gen.get_latent_num_samples(sound_sample.shape[-1]))

        # existing vision conditioning is preserved in the sequence plan for i2v and v2v modes
        sequence_plan = build_sequence_plan_for_sound(
            mode="t2vs",
            video_latent_length=video_latent_t,
            sound_latent_length=sound_latent_t,
        )
        if existing_vision_cond is not None:
            sequence_plan.condition_frame_indexes_vision = existing_vision_cond
    else:
        if existing_plans and isinstance(existing_plans, list) and len(existing_plans) > 0:
            return data_batch
        sequence_plan = SequencePlan(
            has_text=True,
            has_vision=True,
            condition_frame_indexes_vision=[],
        )

    batch_size = len(data_batch["video"])
    data_batch["sequence_plan"] = [sequence_plan] * batch_size
    return data_batch


def mux_audio_into_video(
    video_path: Path,
    audio_waveform: torch.Tensor,
    sample_rate: int,
) -> None:
    """Mux a decoded audio waveform into an existing MP4's audio track.

    The video stream is copied without re-encoding; the audio is encoded as AAC.
    The input file is replaced in place via an atomic rename.

    Args:
        video_path: Path to an existing .mp4 (replaced in place).
        audio_waveform: Decoded audio tensor of shape (C, N) or (N,).
        sample_rate: Audio sample rate in Hz.
    """
    import av
    import numpy as np

    if audio_waveform.ndim == 1:
        audio_waveform = audio_waveform.unsqueeze(0)
    audio_np = audio_waveform.clamp(-1, 1).to(dtype=torch.float32).cpu().numpy()
    channels, total_samples = audio_np.shape

    if channels == 1:
        layout = "mono"
    elif channels == 2:
        layout = "stereo"
    else:
        raise ValueError(f"Unsupported channel count {channels} for AAC muxing")

    tmp_path = video_path.with_suffix(video_path.suffix + ".muxing")
    with av.open(str(video_path), mode="r") as in_container:
        in_video_stream = in_container.streams.video[0]
        with av.open(str(tmp_path), mode="w", format="mp4") as out_container:
            out_video_stream = out_container.add_stream_from_template(in_video_stream)
            out_audio_stream = out_container.add_stream("aac", rate=sample_rate)
            out_audio_stream.layout = layout
            out_audio_stream.bit_rate = 256_000

            for packet in in_container.demux(in_video_stream):
                if packet.dts is None:
                    continue
                packet.stream = out_video_stream
                out_container.mux(packet)

            frame_size = out_audio_stream.frame_size or 1024
            pts = 0
            for start in range(0, total_samples, frame_size):
                end = min(start + frame_size, total_samples)
                chunk = np.ascontiguousarray(audio_np[:, start:end])
                frame = av.AudioFrame.from_ndarray(chunk, format="fltp", layout=layout)
                frame.sample_rate = sample_rate
                frame.pts = pts
                pts += end - start
                for packet in out_audio_stream.encode(frame):
                    out_container.mux(packet)
            for packet in out_audio_stream.encode():
                out_container.mux(packet)

    tmp_path.replace(video_path)


def save_sound(
    audio_waveform: torch.Tensor,
    output_path: Path,
    sample_rate: int,
) -> None:
    """Save a decoded audio waveform as a WAV file.

    Args:
        audio_waveform: Decoded audio tensor (e.g. shape (C, N) or (N,)).
        output_path: Destination .wav path.
        sample_rate: Audio sample rate in Hz.
    """
    import soundfile as sf  # type: ignore[import-not-found]

    audio_np = audio_waveform.clamp(-1, 1).to(dtype=torch.float32).cpu().numpy()
    if audio_np.ndim == 2:
        audio_np = audio_np.T

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), audio_np, sample_rate)
