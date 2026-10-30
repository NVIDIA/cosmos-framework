# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import math
from pathlib import Path
from typing import Literal

import av
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from cosmos_framework.data.generator.sequence_packing import SequencePlan
from cosmos_framework.data.generator.utils import VIDEO_RES_SIZE_INFO
from cosmos_framework.utils.generator.host_threads import host_threads_per_local_rank, torch_host_threads

VideoDecoderBackend = Literal["torchcodec", "pyav"]


def crop_media_to_content(
    frames: torch.Tensor,
    crop_box: tuple[int, int, int, int] | None,
) -> torch.Tensor:
    """Crop ``[C,T,H,W]`` media to an explicitly declared content rectangle.

    The box follows the PIL convention ``(left, top, right, bottom)`` with an
    exclusive right/bottom edge. Inference cannot reliably distinguish a real
    full-resolution image from a pre-padded canvas, so callers must declare the
    content rectangle when their transport asset includes padding.
    """
    if crop_box is None:
        return frames
    if frames.ndim != 4:
        raise ValueError(f"Expected media with shape [C,T,H,W], got {tuple(frames.shape)}")
    left, top, right, bottom = map(int, crop_box)
    height, width = frames.shape[-2:]
    if left < 0 or top < 0 or right <= left or bottom <= top or right > width or bottom > height:
        raise ValueError(
            "content_crop_box must be a non-empty (left, top, right, bottom) rectangle "
            f"inside the decoded {width}x{height} media, got {tuple(crop_box)}"
        )
    return frames[..., top:bottom, left:right].contiguous()


def resize_pil_image(image: Image.Image, max_size: int, padding_constant: int) -> Image.Image:
    """Resize a PIL image so the max side length is at most *max_size* and both
    dimensions are divisible by *padding_constant*.

    Args:
        image: Input PIL image.
        max_size: Maximum allowed side length (longest edge will be at most this).
        padding_constant: Both height and width are rounded down to the nearest
            multiple of this value.

    Returns:
        Resized PIL image.
    """
    orig_w, orig_h = image.size
    scale = max_size / max(orig_w, orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    new_w = (new_w // padding_constant) * padding_constant
    new_h = (new_h // padding_constant) * padding_constant
    new_w = max(new_w, padding_constant)
    new_h = max(new_h, padding_constant)
    return image.resize(
        (new_w, new_h),
        Image.LANCZOS,  # type: ignore
    )


def _resize_and_center_crop(
    frames: torch.Tensor,
    target_h: int,
    target_w: int,
    *,
    interpolation: TF.InterpolationMode = TF.InterpolationMode.BILINEAR,
) -> torch.Tensor:
    """Aspect-ratio-preserving resize followed by center crop."""
    orig_h, orig_w = frames.shape[2], frames.shape[3]
    scaling_ratio = max(target_w / orig_w, target_h / orig_h)
    resize_h = int(math.ceil(scaling_ratio * orig_h))
    resize_w = int(math.ceil(scaling_ratio * orig_w))
    frames = TF.resize(frames, [resize_h, resize_w], interpolation=interpolation)  # [...,resize_h,resize_w]
    frames = TF.center_crop(frames, [target_h, target_w])  # [...,target_h,target_w]
    return frames


def load_conditioning_image_pixels(image_path: Path, target_h: int, target_w: int) -> torch.Tensor:
    """Load an image as resized/cropped uint8 pixels in ``[3, H, W]``."""
    with image_path.open("rb") as f:
        img = Image.open(f).convert("RGB")
    img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float().unsqueeze(0)  # [1,3,H,W]
    img_tensor = _resize_and_center_crop(img_tensor, target_h, target_w)  # [1,3,target_h,target_w]
    return img_tensor.squeeze(0).round().clamp(0, 255).to(torch.uint8)  # [3,target_h,target_w]


def load_prompt_upsampling_image(image_path: Path, target_h: int, target_w: int) -> Image.Image:
    """Load an image as resized/cropped RGB PIL pixels for VLM prompt upsampling."""
    img_tensor = load_conditioning_image_pixels(image_path, target_h, target_w)  # [3,target_h,target_w]
    img_array = img_tensor.permute(1, 2, 0).contiguous().cpu().numpy()  # [target_h,target_w,3]
    return Image.fromarray(img_array, mode="RGB")


def load_conditioning_image(image_path: Path, target_h: int, target_w: int) -> torch.Tensor:
    """Load an image as conditioning frames from local or remote path; returns (3, 1, H, W) in [-1, 1]."""
    img_tensor = load_conditioning_image_pixels(image_path, target_h, target_w).float()  # [3,target_h,target_w]
    img_tensor = img_tensor / 127.5 - 1.0  # [3,target_h,target_w]
    return img_tensor.unsqueeze(1)  # [3,1,target_h,target_w]


def decode_video_thwc_uint8(path: Path) -> tuple[torch.Tensor, float]:
    """Read all frames of a video as a uint8 [T, H, W, C] tensor plus fps.

    TorchCodec replacement for ``torchvision.io.read_video``; mirrors its frame layout so
    callers are unchanged. As of torchvision 0.28 the public wheel no longer binds that
    symbol — ``torchvision/io/__init__.py`` imports it from the Meta-internal
    ``pytorch.vision.fb.io.video`` under ``except ImportError: pass``, so ``import
    torchvision.io`` still succeeds but calling ``read_video`` raises ``AttributeError``.
    """
    from cosmos_framework.utils.generator.torchcodec_video import decode_all_frames_nhwc_uint8

    # torchcodec's core ops are CPU-only. Force CPU tensor creation so an active default-CUDA
    # device context (torch.set_default_device during generation) doesn't route torchcodec's
    # internal frame-index tensor to CUDA and raise NotImplementedError.
    with torch.device("cpu"):
        frames_nhwc, meta = decode_all_frames_nhwc_uint8(path)
    return torch.from_numpy(frames_nhwc), float(meta.average_fps)


def decode_video_pyav_thwc_uint8(path: Path) -> tuple[torch.Tensor, float]:
    """Match native Transfer's TorchVision/PyAV RGB conversion without legacy APIs.

    TorchCodec's RGB conversion can differ by integer rounding from PyAV. Causal
    Transfer requires the same decoded controls as native inference, including
    presentation-timestamp ordering and PyAV's default ``to_rgb`` conversion.
    """
    with av.open(str(path), metadata_errors="ignore") as container:
        if not container.streams.video:
            raise ValueError(f"No video stream found in {path}")
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate is not None else 24.0
        # Convert while decoding so per-frame PyAV reformatters do not retain
        # hundreds of conversion contexts until the entire clip has been read.
        pixels_by_pts = {
            frame.pts: frame.to_rgb().to_ndarray()  # [H,W,3]
            for frame in container.decode(video=0)
            if frame.pts is not None and frame.pts >= 0
        }
        if not pixels_by_pts:
            raise ValueError(f"Decoded zero video frames from {path}")
        pixels = np.stack([pixels_by_pts[pts] for pts in sorted(pixels_by_pts)])  # [T,H,W,3]
    return torch.from_numpy(pixels), fps  # [T,H,W,3]


def load_conditioning_video(
    video_path: Path,
    target_h: int,
    target_w: int,
    max_frames: int,
    *,
    keep: Literal["first", "last"] = "first",
) -> torch.Tensor:
    """Load video frames for conditioning; returns (3, T, H, W) in [-1, 1].

    ``keep`` selects which ``max_frames`` to take when the input is longer, with Python slice
    semantics: ``frames[-max_frames:]`` for ``"last"``, otherwise ``frames[:max_frames]``.

    Only the kept frames are decoded, and the decode and the resize run on this rank's share of
    the host's CPUs. A WSM control clip is ~650 frames of which a 297-frame rollout keeps the
    first 297, and an i2v camera keeps only frame 0, so decoding every frame and slicing
    afterwards spent most of the media preparation on frames nobody reads. The pixels are
    unchanged: H.264 decoding is exact under threading and with a range decode, and the resize
    computes each output element independently of how its work is split across threads.
    """
    # Imported here rather than at module scope: every inference module imports this one, and
    # importing torchcodec loads FFmpeg's shared libraries.
    from cosmos_framework.utils.generator.torchcodec_dependencies import load_torchcodec

    load_torchcodec()
    from torchcodec.decoders import VideoDecoder

    # torchcodec's core ops are CPU-only, and this runs under generation's default-CUDA device
    # context; keep every tensor created here (the empty selection included) on the host.
    with torch.device("cpu"):
        num_threads = host_threads_per_local_rank()
        decoder = VideoDecoder(video_path, device="cpu", dimension_order="NHWC", num_ffmpeg_threads=num_threads)
        frame_indices = range(len(decoder))
        kept = frame_indices[-max_frames:] if keep == "last" else frame_indices[:max_frames]
        if len(kept) > 0:
            frames = decoder.get_frames_in_range(kept.start, kept.stop).data  # [T,H,W,3]
        else:
            frames = torch.empty(
                (0, decoder.metadata.height, decoder.metadata.width, 3), dtype=torch.uint8
            )  # [0,H,W,3]
        with torch_host_threads(num_threads):
            frames_tchw = frames.permute(0, 3, 1, 2).float()  # [T,3,H,W]
            frames_resized = _resize_and_center_crop(frames_tchw, target_h, target_w)  # [T,3,target_h,target_w]
            frames_normalized = frames_resized / 127.5 - 1.0  # [T,3,target_h,target_w]
        return frames_normalized.permute(1, 0, 2, 3)  # [3,T,target_h,target_w]


def pil_to_conditioning_frames(pil_img: Image.Image) -> tuple[torch.Tensor, int, int]:
    """Convert a PIL image to a conditioning tensor in [-1, 1] and return (frames, h, w)."""
    w, h = pil_img.size
    img_tensor = torch.from_numpy(np.array(pil_img)).permute(2, 0, 1).float()  # [3,H,W]
    return (img_tensor / 127.5 - 1.0).unsqueeze(1), h, w  # [3,1,H,W]


def build_conditioned_video_batch(
    conditioning_frames: torch.Tensor,
    condition_frames_vision: list[int],
    w: int,
    h: int,
    num_frames: int,
    fps: float,
    batch_size: int = 1,
) -> dict:
    """Build a data batch with conditioning frames and sequence plans for generation."""
    t_cond = conditioning_frames.shape[1]
    video_data = torch.zeros(1, 3, num_frames, h, w, dtype=torch.bfloat16)  # [1,3,num_frames,h,w]
    t_fill = min(t_cond, num_frames)
    video_data[0, :, :t_fill, :, :] = conditioning_frames[:, :t_fill, :, :].to(dtype=torch.bfloat16)  # [3,t_fill,h,w]
    if t_fill < num_frames:
        video_data[0, :, t_fill:, :, :] = video_data[0, :, t_fill - 1 : t_fill, :, :].expand(
            -1, num_frames - t_fill, -1, -1
        )  # [3,num_frames-t_fill,h,w]
    video_list = [video_data.cuda() for _ in range(batch_size)]  # list of [1,3,num_frames,h,w]
    image_size = [torch.tensor([[h, w, h, w]], dtype=torch.float32).cuda() for _ in range(batch_size)]  # list of [1,4]
    sequence_plans = [
        SequencePlan(has_text=True, has_vision=True, condition_frame_indexes_vision=list(condition_frames_vision))
        for _ in range(batch_size)
    ]
    return {
        "dataset_name": "video_data",
        "video": video_list,
        "image_size": image_size,
        "t5_text_embeddings": torch.randn(batch_size, 512, 1024).cuda().to(dtype=torch.bfloat16),  # [B,512,1024]
        "fps": torch.full((batch_size,), float(fps)).cuda(),  # [B]
        "conditioning_fps": torch.full((batch_size,), float(fps)).cuda(),  # [B]
        "num_frames": torch.full((batch_size,), num_frames).cuda(),  # [B]
        "is_preprocessed": True,
        "sequence_plan": sequence_plans,
    }


def build_image_edit_batch(
    conditioning_frames: torch.Tensor,
    h: int,
    w: int,
    batch_size: int = 1,
) -> dict:
    """Build a data batch for image-to-image editing."""
    image = conditioning_frames.unsqueeze(0).cuda().to(dtype=torch.bfloat16)  # [1,3,1,h,w]
    sequence_plans = [
        SequencePlan(has_text=True, has_vision=True, condition_frame_indexes_vision=[]) for _ in range(batch_size)
    ]
    image_size = torch.tensor([[h, w, h, w]], dtype=torch.float32).cuda()  # [1,4]
    return {
        "dataset_name": "image_data",
        "images": [image, image] * batch_size,
        "image_size": [image_size, image_size] * batch_size,
        "num_frames": [torch.tensor([2], dtype=torch.int64).cuda() for _ in range(batch_size)],  # list of [1]
        "num_vision_items_per_sample": [2 for _ in range(batch_size)],
        "is_preprocessed": True,
        "sequence_plan": sequence_plans,
    }


_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def detect_aspect_ratio(width: int, height: int) -> str:
    """Return the closest supported aspect-ratio key for a frame size."""
    aspect_ratios = np.array([16 / 9, 4 / 3, 1, 3 / 4, 9 / 16])
    aspect_ratio_keys = ["16,9", "4,3", "1,1", "3,4", "9,16"]
    current = width / height
    return aspect_ratio_keys[int(np.argmin((aspect_ratios - current) ** 2))]


def read_media_frames(
    path: Path, max_frames: int, *, video_decoder: VideoDecoderBackend = "torchcodec"
) -> tuple[torch.Tensor, float]:
    """Read an image or video into a uint8 tensor of shape (C, T, H, W)."""
    ext = path.suffix.lower()
    if ext in _IMAGE_EXTENSIONS:
        with path.open("rb") as f:
            image = Image.open(f).convert("RGB")
        frames = torch.from_numpy(np.array(image)).permute(2, 0, 1).unsqueeze(1)
        return frames, 1.0
    if ext not in _VIDEO_EXTENSIONS:
        raise ValueError(f"Unsupported media extension: {ext}")
    if video_decoder == "pyav":
        frames, fps = decode_video_pyav_thwc_uint8(path)  # [T,H,W,C]
    else:
        frames, fps = decode_video_thwc_uint8(path)  # [T,H,W,C]
    frames = frames[:max_frames].permute(0, 3, 1, 2).permute(1, 0, 2, 3)
    return frames, fps


def probe_media_hw_fps(
    path: Path, *, video_decoder: VideoDecoderBackend = "torchcodec"
) -> tuple[tuple[int, int], float]:
    """Return decoded ``(height, width)`` and video FPS without resizing pixels.

    Images return zero FPS so callers retain their requested clock. The PyAV
    probe decodes only the first presentable frame and uses the same FPS fallback
    as the full reader. The TorchCodec path retains its existing decode behavior.
    """
    ext = path.suffix.lower()
    if ext in _IMAGE_EXTENSIONS:
        with path.open("rb") as f:
            width, height = Image.open(f).size
        return (height, width), 0.0
    if ext not in _VIDEO_EXTENSIONS:
        raise ValueError(f"Unsupported media extension: {ext}")
    if video_decoder == "pyav":
        with av.open(str(path), metadata_errors="ignore") as container:
            if not container.streams.video:
                raise ValueError(f"No video stream found in {path}")
            stream = container.streams.video[0]
            fps = float(stream.average_rate) if stream.average_rate is not None else 24.0
            for frame in container.decode(video=0):
                if frame.pts is not None and frame.pts >= 0:
                    height, width, _ = frame.to_rgb().to_ndarray().shape  # [H,W,3]
                    return (height, width), fps
        raise ValueError(f"Decoded zero video frames from {path}")
    frames, fps = decode_video_thwc_uint8(path)  # [T,H,W,C]
    return (frames.shape[1], frames.shape[2]), fps


def probe_media_hw(path: Path, *, video_decoder: VideoDecoderBackend = "torchcodec") -> tuple[int, int]:
    """Return ``(height, width)`` using the decoder's first presentable frame."""
    hw, _ = probe_media_hw_fps(path, video_decoder=video_decoder)
    return hw


def read_and_resize_media(
    path: Path,
    *,
    resolution: str,
    aspect_ratio: str | None,
    max_frames: int,
    video_decoder: VideoDecoderBackend = "torchcodec",
    interpolation: TF.InterpolationMode = TF.InterpolationMode.BILINEAR,
) -> tuple[torch.Tensor, float, str, tuple[int, int]]:
    """Read an image/video and resize it to the requested resolution bucket."""
    raw_frames, fps = read_media_frames(path, max_frames=max_frames, video_decoder=video_decoder)  # [C,T,H,W]
    original_hw = (raw_frames.shape[2], raw_frames.shape[3])
    detected_aspect_ratio = detect_aspect_ratio(raw_frames.shape[3], raw_frames.shape[2])
    final_aspect_ratio = aspect_ratio or detected_aspect_ratio
    width, height = VIDEO_RES_SIZE_INFO[resolution][final_aspect_ratio]
    resized = _resize_and_center_crop(raw_frames.permute(1, 0, 2, 3), height, width, interpolation=interpolation)
    return resized.permute(1, 0, 2, 3), fps, final_aspect_ratio, original_hw


def uint8_to_normalized_float(tensor: torch.Tensor, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Convert uint8 [0, 255] frames into normalized [-1, 1] frames."""
    return tensor.to(dtype=dtype) / 127.5 - 1.0


def pad_temporal_frames(frames: torch.Tensor, target_frames: int) -> torch.Tensor:
    """Pad a (C, T, H, W) tensor along time using reflection/repeat behavior."""
    num_frames = frames.shape[1]
    if num_frames >= target_frames:
        return frames
    if num_frames == 0:
        raise ValueError("Cannot pad an empty frame tensor.")
    padded = frames
    while padded.shape[1] < target_frames:
        pad_len = min(padded.shape[1] - 1, target_frames - padded.shape[1])
        if pad_len <= 0:
            pad_frame = padded[:, -1:].repeat(1, target_frames - padded.shape[1], 1, 1)
            padded = torch.cat([padded, pad_frame], dim=1)
            break
        padded = torch.cat([padded, padded.flip(dims=[1])[:, :pad_len]], dim=1)
    return padded
