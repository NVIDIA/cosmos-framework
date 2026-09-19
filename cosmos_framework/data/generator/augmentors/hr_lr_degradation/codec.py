# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Video codec round trip (P3) for the LR stream.

Encodes the whole clip with H.264 or H.265 through PyAV and decodes it back, so the LR carries real
inter-frame compression artifacts (blocking, ringing, temporal flicker at low bitrates). This is a
whole-clip operation: it must run after the per-frame pipeline has produced the final LR clip, not
inside the frame-chunked loop. Software encoders run on CPU; if the clip lives on a GPU it is moved
to host memory for the round trip and moved back.

Encoders: ``libx264`` / ``libx265`` (always available in the PyAV wheel), plus ``h264_nvenc`` /
``hevc_nvenc`` when the FFmpeg build exposes them. Callers always speak x264 vocabulary: a CRF on the
0 to 51 scale and an x264 speed preset. For NVENC these map to ``rc=vbr`` with ``cq`` (the quality-
targeted mode that corresponds to CRF; ``constqp`` would be x264's fixed ``-qp``) and to the ``p1`` to
``p7`` speed ladder via ``nvenc_preset``.
"""

from __future__ import annotations

import io
import os
from functools import lru_cache

import av
import numpy as np
import torch

SOFTWARE_CODECS = ("libx264", "libx265")
HARDWARE_CODECS = ("h264_nvenc", "hevc_nvenc")
# Thread budget per round trip, applied to the encoder, the decoder and libswscale's colour conversion. All
# three default to pools sized to the whole host (38 threads for x264, 68 for x265 and 32 for the yuv420p to
# rgb24 conversion on a 16-core box). In a dataloader with many workers, or under pytest-xdist in a container
# with a pid limit, that exhausts the thread budget and stalls every process on the host (CI's CPU phase went
# from 9.5 min to a 30 min timeout). The clips are short, so a small budget costs nothing measurable.
# The conversion bound uses ``VideoFrame.reformat(threads=...)`` (PyAV >= 17); the docker requirements pin
# ``av>=17`` (18 needs Python >= 3.11); every current imaginaire4 image ships PyAV 18.
DEFAULT_CODEC_THREADS = int(os.environ.get("HR_LR_CODEC_THREADS", "2"))
# NVENC rejects very small frames (documented minimum 145x49 for H.264; 128x96 fails, 256x144 works on an L4).
NVENC_MIN_WIDTH, NVENC_MIN_HEIGHT = 145, 49


@lru_cache(maxsize=None)
def codec_available(name: str) -> bool:
    try:
        av.codec.Codec(name, "w")
        return True
    except Exception:  # av.codec.codec.UnknownCodecError and friends
        return False


X264_PRESETS = (
    "ultrafast",
    "superfast",
    "veryfast",
    "faster",
    "fast",
    "medium",
    "slow",
    "slower",
    "veryslow",
    "placebo",
)

# x264 speed preset -> NVENC p1 (fastest) .. p7 (slowest, best quality). Anchored at veryfast -> p1 and
# medium -> p4; the rest follow the speed ordering monotonically.
_NVENC_PRESET_FROM_X264 = {
    "ultrafast": "p1",
    "superfast": "p1",
    "veryfast": "p1",
    "faster": "p2",
    "fast": "p3",
    "medium": "p4",
    "slow": "p5",
    "slower": "p6",
    "veryslow": "p7",
    "placebo": "p7",
}


def nvenc_preset(preset: str) -> str:
    """Map an x264 speed preset to the NVENC ``p1``..``p7`` ladder (``pN`` values pass through)."""
    if preset in _NVENC_PRESET_FROM_X264:
        return _NVENC_PRESET_FROM_X264[preset]
    if len(preset) == 2 and preset[0] == "p" and preset[1] in "1234567":
        return preset
    raise ValueError(f"Unknown preset {preset!r}; expected one of {X264_PRESETS} or p1..p7")


def _encoder_options(codec: str, crf: float, preset: str, threads: int = DEFAULT_CODEC_THREADS) -> dict[str, str]:
    q = str(int(round(crf)))
    t = str(max(1, int(threads)))
    if codec == "libx264":
        return {"crf": q, "preset": preset, "threads": t}
    if codec == "libx265":
        # ``threads`` maps to x265 frame threads; ``pools`` bounds the worker-thread pool, which otherwise
        # defaults to one thread per host core.
        return {"x265-params": f"crf={q}:log-level=error:pools={t}", "preset": preset, "threads": t}
    if codec in HARDWARE_CODECS:
        # CRF analogue: quality-targeted VBR with a constant-quality level on the same 0..51 scale and no
        # bitrate cap (b=0). constqp would pin one QP for every frame, which is x264's -qp, not CRF.
        return {"rc": "vbr", "cq": q, "b": "0", "preset": nvenc_preset(preset)}
    raise ValueError(f"Unsupported codec {codec!r}")


def codec_round_trip(
    frames: torch.Tensor,  # [T,C,H,W] uint8 or float in [0,1], any device
    codec: str = "libx264",
    crf: float = 23.0,
    preset: str = "veryfast",
    fps: float = 24.0,
    threads: int = DEFAULT_CODEC_THREADS,
) -> torch.Tensor:  # returns [T,C,H,W] same dtype and device as the input
    """Encode the clip with ``codec`` at the given CRF/QP and decode it back.

    ``threads`` bounds both the encoder and the decoder (default ``HR_LR_CODEC_THREADS`` or 2); the clips are
    short and small, so more threads buy little and cost a lot when many processes encode at once.
    """
    if frames.dim() != 4 or frames.shape[1] != 3:
        raise ValueError(f"codec_round_trip expects [T,3,H,W], got {tuple(frames.shape)}")
    if not codec_available(codec):
        raise RuntimeError(f"Encoder {codec!r} is not available in this FFmpeg build")
    device, dtype = frames.device, frames.dtype
    t, _, h, w = frames.shape
    if codec in HARDWARE_CODECS and (w < NVENC_MIN_WIDTH or h < NVENC_MIN_HEIGHT):
        raise ValueError(
            f"{codec} needs frames of at least {NVENC_MIN_WIDTH}x{NVENC_MIN_HEIGHT} (WxH), got {w}x{h}; "
            "use libx264/libx265 for smaller clips"
        )
    if dtype == torch.uint8:
        rgb = frames.permute(0, 2, 3, 1).cpu().numpy()  # [T,H,W,3]
    else:
        rgb = (frames.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()  # [T,H,W,3]
    # yuv420p needs even dimensions; pad by edge replication and crop after decoding.
    pad_h, pad_w = h % 2, w % 2
    if pad_h or pad_w:
        rgb = np.pad(rgb, ((0, 0), (0, pad_h), (0, pad_w), (0, 0)), mode="edge")  # [T,H+ph,W+pw,3]

    buffer = io.BytesIO()
    container = av.open(buffer, mode="w", format="mp4")
    stream = container.add_stream(codec, rate=max(1, int(round(fps))))
    stream.width, stream.height = w + pad_w, h + pad_h
    stream.pix_fmt = "yuv420p"
    stream.options = _encoder_options(codec, crf, preset, threads)
    stream.thread_count = max(1, int(threads))
    n_threads = max(1, int(threads))
    for frame_rgb in rgb:
        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(frame_rgb), format="rgb24")
        frame = frame.reformat(format="yuv420p", threads=n_threads)  # explicit, thread-bounded rgb -> yuv
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()

    buffer.seek(0)
    decoded: list[np.ndarray] = []
    with av.open(buffer) as container_in:
        container_in.streams.video[0].thread_count = n_threads  # set before the first decode() opens the context
        for frame in container_in.decode(video=0):
            # ``to_ndarray(format=...)`` converts with an auto-sized swscale pool; reformat with an explicit budget.
            decoded.append(frame.reformat(format="rgb24", threads=n_threads).to_ndarray())  # [H+ph,W+pw,3]
    if len(decoded) != t:
        raise RuntimeError(f"Codec round trip returned {len(decoded)} frames for {t} input frames")
    out = np.stack(decoded, axis=0)[:, :h, :w]  # [T,H,W,3]
    out_t = torch.from_numpy(np.ascontiguousarray(out)).permute(0, 3, 1, 2)  # [T,3,H,W] uint8
    if dtype != torch.uint8:
        out_t = out_t.to(dtype) / 255.0  # [T,3,H,W]
    return out_t.to(device)
