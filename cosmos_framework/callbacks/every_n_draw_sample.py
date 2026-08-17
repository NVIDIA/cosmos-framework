# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import math
import os
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
from typing import Any, Literal, NamedTuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as torchvision_F
import wandb
from einops import rearrange
from PIL import Image, ImageDraw, ImageFont

from cosmos_framework.callbacks.every_n import EveryN
from cosmos_framework.model._base import ImaginaireModel
from cosmos_framework.utils import distributed, log, misc
from cosmos_framework.utils.easy_io import easy_io
from cosmos_framework.tools.visualize.video import save_img_or_video

from cosmos_framework.model.generator.mot.context_parallel_utils import broadcast_context_parallel_object
from cosmos_framework.utils.generator.data_utils import slice_data_batch
from cosmos_framework.utils.generator.multiview import (
    decode_multiview_latent_per_view,
    split_multiview_tensor_by_view,
)


class WandbAnimation(NamedTuple):
    """A whole clip written as a looping GIF, for W&B to play inline."""

    path: str
    num_frames: int


# What a clip is previewed as in W&B: one still grid, one panel per sampled
# frame, or an animation of every frame.
WandbClipPreview = Literal["grid", "frames", "animation"]
WandbMedia = str | dict[str, str] | WandbAnimation


def resize_image(image: torch.Tensor, size: int = 1024) -> torch.Tensor:
    """
    Resize the image to the given size. This is done so that wandb can display the image correctly.
    """
    _, h, w = image.shape
    ratio = size / max(h, w)
    new_h, new_w = int(ratio * h), int(ratio * w)
    return torchvision_F.resize(image, (new_h, new_w))


def is_primitive(value):
    return isinstance(value, (int, float, str, bool, type(None)))


def convert_to_primitive(value):
    if isinstance(value, (list, tuple)):
        return [convert_to_primitive(v) for v in value if is_primitive(v) or isinstance(v, (list, dict))]
    elif isinstance(value, dict):
        return {k: convert_to_primitive(v) for k, v in value.items() if is_primitive(v) or isinstance(v, (list, dict))}
    elif is_primitive(value):
        return value
    else:
        return "non-primitive"  # Skip non-primitive types


def pad_images_and_cat(images: list[torch.Tensor], max_w: int, max_h: int, t_crop: int = 1) -> torch.Tensor:
    """
    Pad images to a common size and concatenate them along the batch dimension.

    This function is needed because different samples in a batch can have different resolutions.
    To create a unified visualization grid, all images must be padded to the same dimensions.
    Images are center-padded to preserve their visual content in the middle.

    Args:
        images: List of image/video tensors with shape [B, C, T, H, W].
        max_w: Target width to pad all images to.
        max_h: Target height to pad all images to.
        t_crop: Number of temporal frames to keep for videos. If > 1 and the image
            has more than 1 frame, only the first t_crop frames are retained.

    Returns:
        Concatenated tensor of padded images with shape [total_B, C, T, max_h, max_w].
    """
    padded_images = []
    for image in images:
        # Pad the image to the center
        padding_h = (max_h - image.shape[-2]) // 2
        padding_w = (max_w - image.shape[-1]) // 2
        padded_image = torch.nn.functional.pad(
            image, (padding_w, max_w - image.shape[-1] - padding_w, padding_h, max_h - image.shape[-2] - padding_h)
        )  # [B,C,T,max_h,max_w]
        # Handle video case
        if image.shape[2] > 1 and t_crop > 1:
            padded_image = padded_image[:, :, 0:t_crop, :, :]

        padded_images.append(padded_image)
    return torch.cat(padded_images, dim=0)  # [total_B,C,T,max_h,max_w]  (total_B = sum of batch dims)


@dataclass(frozen=True, slots=True)
class MultiviewTransferMetadata:
    num_vision_items: int
    sample_n_views: int
    num_video_frames_per_view: int


@dataclass(frozen=True, slots=True)
class MultiviewTransferSampleResult:
    handled: bool
    media: WandbMedia | None = None


@dataclass(frozen=True, slots=True)
class TransferGeneration:
    """Generated latents to draw, or the result the caller must return instead of drawing.

    ``stop`` is set only when generation failed uniformly on every rank (for example the
    model returned the wrong number of vision tensors). Ranks that only participate in
    sampling to keep CP/FSDP collectives aligned return empty ``latents`` without ``stop``;
    callers must still wait with those ranks at a post-materialize barrier before any
    rank enters the callback-level NCCL barrier.
    """

    latents: list[torch.Tensor]
    stop: MultiviewTransferSampleResult | None = None


def _flatten_int_metadata(value: Any) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        flat_value = value.detach().cpu().flatten()  # [N_metadata]
        return [int(item) for item in flat_value.tolist()]
    if isinstance(value, (list, tuple)):
        values: list[int] = []
        for item in value:
            if isinstance(item, torch.Tensor):
                flat_item = item.detach().cpu().flatten()  # [N_metadata]
                values.extend(int(v) for v in flat_item.tolist())
            elif isinstance(item, (int, np.integer)):
                values.append(int(item))
            else:
                return None
        return values
    if isinstance(value, (int, np.integer)):
        return [int(value)]
    return None


def _first_positive_metadata_value(data_batch: dict[str, Any], key: str) -> int | None:
    values = _flatten_int_metadata(data_batch.get(key))
    if not values:
        return None
    value = values[0]
    return value if value > 0 else None


def _get_multiview_visualization_item_counts(
    data_batch: dict[str, Any],
    num_vision_items_per_sample: Any,
    batch_size: int,
) -> list[int] | None:
    num_items = _flatten_int_metadata(num_vision_items_per_sample)
    if num_items:
        return num_items
    if "sample_n_views" not in data_batch or "num_video_frames_per_view" not in data_batch:
        return None
    return [1] * batch_size


def _get_multiview_transfer_metadata(
    data_batch: dict[str, Any],
    num_vision_items_per_sample: Any,
) -> MultiviewTransferMetadata | None:
    """Detect camera-major multiview samples after the caller normalizes one-item counts."""
    if "sample_n_views" not in data_batch or "num_video_frames_per_view" not in data_batch:
        return None

    num_items = _flatten_int_metadata(num_vision_items_per_sample)
    if not num_items:
        return None

    sample_n_views = _first_positive_metadata_value(data_batch, "sample_n_views")
    num_video_frames_per_view = _first_positive_metadata_value(data_batch, "num_video_frames_per_view")
    if num_items[0] < 1 or sample_n_views is None or num_video_frames_per_view is None:
        return None

    return MultiviewTransferMetadata(
        num_vision_items=num_items[0],
        sample_n_views=sample_n_views,
        num_video_frames_per_view=num_video_frames_per_view,
    )


def _first_metadata_string(value: Any) -> str | None:
    while isinstance(value, (list, tuple)):
        if not value:
            return None
        value = value[0]
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return value if isinstance(value, str) and value else None


def _is_lidar_visualization_batch(data_batch: dict[str, Any]) -> bool:
    """Return True when draw samples should be colorized as LiDAR range maps."""
    return (
        _first_metadata_string(data_batch.get("vision_tokenizer_type")) == "lidar"
        or _first_metadata_string(data_batch.get("dataset_name")) == "lidar_transfer"
    )




def _to_display_row(pixels: torch.Tensor, *, colorize_lidar: bool = False) -> torch.Tensor:
    """Turn decoded pixels into a host-side visualization row: uint8, on CPU.

    Quantizing on the device, before the copy to the host, is what makes a tiled multiview
    grid affordable: the transfer, the grid and every copy the encoder makes of the grid
    all inherit the reduction. LiDAR range maps are colorized to RGB instead.
    """
    return _to_display_uint8(pixels).cpu()  # [V,C,F,H,W]


def _get_first_multiview_transfer_rows(
    raw_data: list[torch.Tensor] | None,
    metadata: MultiviewTransferMetadata,
    *,
    colorize_lidar: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if metadata.num_vision_items < 2 or raw_data is None or len(raw_data) < metadata.num_vision_items:
        return None

    control_video = _prepare_multiview_video_for_visualization(
        raw_data[0],
        metadata.sample_n_views,
        metadata.num_video_frames_per_view,
        colorize_lidar=colorize_lidar,
    )  # [V,C,F,H,W] or None
    gt_target_video = _prepare_multiview_video_for_visualization(
        raw_data[metadata.num_vision_items - 1],
        metadata.sample_n_views,
        metadata.num_video_frames_per_view,
        colorize_lidar=colorize_lidar,
    )  # [V,C,F,H,W] or None
    if control_video is None or gt_target_video is None:
        return None
    return control_video, gt_target_video


def _get_first_multiview_target_row(
    raw_data: list[torch.Tensor] | None,
    metadata: MultiviewTransferMetadata,
    *,
    colorize_lidar: bool = False,
) -> torch.Tensor | None:
    if raw_data is None or len(raw_data) < metadata.num_vision_items:
        return None

    return _prepare_multiview_video_for_visualization(
        raw_data[metadata.num_vision_items - 1],
        metadata.sample_n_views,
        metadata.num_video_frames_per_view,
        colorize_lidar=colorize_lidar,
    )  # [V,C,F,H,W] or None


def _get_first_transfer_dataloader_rows(
    raw_data: list[torch.Tensor] | None,
    metadata: MultiviewTransferMetadata,
    *,
    colorize_lidar: bool = False,
) -> list[torch.Tensor] | None:
    """Return the dataloader rows to display: ``[control, target]``, or ``[target]`` alone.

    Target-only batches (no control item) still draw, so the target row is the one that has
    to be readable; a missing control row just means there is nothing to compare against.
    """
    target_row = _get_first_multiview_target_row(
        raw_data, metadata, colorize_lidar=colorize_lidar
    )  # [V,C,F,H,W] or None
    if target_row is None:
        return None
    control_and_target = _get_first_multiview_transfer_rows(raw_data, metadata, colorize_lidar=colorize_lidar)
    if control_and_target is None:
        return [target_row]  # list[[V,C,F,H,W]]
    return list(control_and_target)  # list[[V,C,F,H,W]]  (control, target)


def _decode_transfer_display_row(
    model: Any,
    latent: torch.Tensor,  # [1,C,V*T_latent,H,W] or [C,V*T_latent,H,W]
    data_batch: dict[str, Any],
    metadata: MultiviewTransferMetadata,
    *,
    colorize_lidar: bool = False,
) -> torch.Tensor | None:  # [V,C,F,H,W] or None
    """Decode one latent into a camera-major display row, or None if it cannot be split by view."""
    assert hasattr(model, "decode")
    if "enable_per_camera_vae_encoding" in data_batch:
        decoded = decode_multiview_latent_per_view(  # [1,C,V*F,H,W] or [C,V*F,H,W]
            model.decode,
            latent,
            metadata.sample_n_views,
            metadata.num_video_frames_per_view,
        )
    else:
        decoded = model.decode(latent)  # [1,C,V*F,H,W] or [C,V*F,H,W]
    decoded_by_view = split_multiview_tensor_by_view(
        decoded,
        metadata.sample_n_views,
        metadata.num_video_frames_per_view,
    )  # [V,C,F,H,W] or None
    if decoded_by_view is None:
        return None
    return _to_display_row(decoded_by_view, colorize_lidar=colorize_lidar)  # [V,C,F,H,W]


def _render_text_stamp(label: str, scale: int = 2) -> torch.Tensor:  # [3,h,w] uint8
    """Render ``label`` as white-on-black pixels.

    The default PIL bitmap font is ~11px tall, which is hard to read once W&B fits a stacked
    grid into a panel, so the glyphs are upscaled with nearest-neighbour to stay crisp.
    """
    font = ImageFont.load_default()
    left, top, right, bottom = ImageDraw.Draw(Image.new("RGB", (1, 1))).textbbox((0, 0), label, font=font)
    stamp = Image.new("RGB", (right - left + 4, bottom - top + 4), color=(0, 0, 0))
    ImageDraw.Draw(stamp).text((2 - left, 2 - top), label, fill=(255, 255, 255), font=font)
    if scale > 1:
        stamp = stamp.resize((stamp.width * scale, stamp.height * scale), Image.NEAREST)
    return torch.from_numpy(np.array(stamp)).permute(2, 0, 1).contiguous()  # [3,h,w]


def _stamp_row_label(row: torch.Tensor, label: str) -> torch.Tensor:  # row: [V,C,F,H,W] uint8
    """Burn ``label`` into the top-left corner of every frame of a display row, in place.

    Range-map rows all look alike, so a stacked grid is only readable if each row says what
    it is. The label is rendered once and broadcast over views and frames, which keeps the
    cost independent of clip length. Rows too small to hold the glyphs are left alone.
    """
    stamp = _render_text_stamp(label)  # [3,h,w]
    channels, height, width = row.shape[1], row.shape[3], row.shape[4]
    stamp_height, stamp_width = stamp.shape[-2:]
    if stamp_height + 2 > height or stamp_width + 2 > width:
        return row
    if channels != stamp.shape[0]:
        stamp = stamp[:1].expand(channels, -1, -1)  # [C,h,w]  (single-channel rows take the same glyphs)
    row[:, :, :, 2 : 2 + stamp_height, 2 : 2 + stamp_width] = stamp.unsqueeze(1).to(row.dtype)
    return row


def _mismatched_row_shapes(rows: list[torch.Tensor]) -> list[tuple[int, ...]] | None:
    """Return every row shape when the rows cannot be stacked into one grid, else None."""
    if any(row.shape != rows[0].shape for row in rows):
        return [tuple(row.shape) for row in rows]
    return None


def _has_first_multiview_transfer_rows(
    raw_data: list[torch.Tensor] | None,
    metadata: MultiviewTransferMetadata,
) -> bool:
    """Check whether the first multiview sample can be split without materializing its rows."""
    if raw_data is None or len(raw_data) < metadata.num_vision_items:
        return False

    expected_num_frames = metadata.sample_n_views * metadata.num_video_frames_per_view
    vision_item_indices = (0, metadata.num_vision_items - 1) if metadata.num_vision_items >= 2 else (0,)
    for vision_item_idx in vision_item_indices:
        vision_item = raw_data[vision_item_idx]  # [B,C,V*F,H,W] or [C,V*F,H,W]
        if vision_item.dim() == 5:
            if vision_item.shape[0] != 1 or vision_item.shape[2] != expected_num_frames:
                return False
        elif vision_item.dim() == 4:
            if vision_item.shape[1] != expected_num_frames:
                return False
        else:
            return False
    return True


def _to_display_uint8(pixels: torch.Tensor) -> torch.Tensor:
    """Map pixels in [-1, 1] onto the uint8 levels the image and video writers consume.

    Rounding rather than truncating avoids a systematic half-level darkening, and the clamp
    is what makes out-of-range decoder output saturate instead of wrapping around, since a
    float outside [0, 255] has no defined uint8 value. Already-uint8 pixels are display
    levels by construction (``v`` encodes ``v / 127.5 - 1``) and pass through untouched.
    """
    if pixels.dtype == torch.uint8:
        return pixels
    # clamp() copies, so the in-place steps that follow never touch the caller's tensor.
    return pixels.clamp(-1, 1).float().add_(1).mul_(127.5).round_().to(torch.uint8)


def _to_unit_float(pixels: torch.Tensor) -> torch.Tensor:
    """Return pixels as float in [0, 1], the range the W&B image writers expect."""
    if pixels.dtype == torch.uint8:
        return pixels.float().div_(255)
    return pixels


def _stack_rows_for_display(rows: list[torch.Tensor]) -> torch.Tensor:
    """Stack visualization rows into one tensor whose values are ready to display.

    uint8 rows already carry display levels and pass through. Float rows arrive in [-1, 1]
    and are mapped to [0, 1] in place: ``torch.stack`` returns fresh memory, so mutating it
    cannot touch the caller's rows, and each temporary skipped is a full copy of the grid.
    """
    stacked = torch.stack(rows, dim=0)  # [N_rows,B,C,T,H,W]
    if stacked.dtype == torch.uint8:
        return stacked
    return stacked.clamp_(-1, 1).add_(1).div_(2)  # [N_rows,B,C,T,H,W]  range [0,1]


def _prepare_multiview_video_for_visualization(
    tensor: torch.Tensor,
    sample_n_views: int,
    num_video_frames_per_view: int,
    *,
    colorize_lidar: bool = False,
) -> torch.Tensor | None:  # tensor: [B,C,V*F,H,W] or [C,V*F,H,W], returns [V,C,F,H,W] uint8
    """Prepare camera-major pixels for visualization, as display-level uint8 on the host.

    Dataloader pixels arrive as uint8 whose levels are already the display levels, so they
    are never expanded to float and renormalized -- at multiview grid sizes that expansion
    alone is gigabytes for rows that end up 8-bit again in the encoder.
    """
    if tensor.dtype == torch.uint8:
        tensor = tensor.cpu()  # [B,C,V*F,H,W] or [C,V*F,H,W]
    video_by_view = split_multiview_tensor_by_view(
        tensor,
        sample_n_views,
        num_video_frames_per_view,
    )  # [V,C,F,H,W] or None
    if video_by_view is None:
        return None
    return _to_display_row(video_by_view, colorize_lidar=colorize_lidar)  # [V,C,F,H,W]


def _write_gif(frames: torch.Tensor, path: str, fps: float, max_size: int = 1024) -> None:
    """Write ``[T,C,H,W]`` display pixels -- uint8, or float in [0, 1] -- as a looping GIF.

    uint8 frames are written as they are: a whole clip of display pixels is large enough
    that expanding it to float only to quantize it back to 8 bits is a copy worth skipping.
    """
    if max(frames.shape[-2:]) > max_size:  # only ever shrink: upscaling costs bytes, not detail
        frames = torch.stack([resize_image(frame, max_size) for frame in frames])  # [T,C,h,w]
    pixels = frames.detach().cpu()  # [T,C,h,w]
    if pixels.dtype != torch.uint8:
        pixels = (pixels.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)  # [T,C,h,w]
    if pixels.shape[1] == 1:
        pixels = pixels.expand(-1, 3, -1, -1)  # [T,3,h,w]  (GIF frames are RGB)
    images = [Image.fromarray(frame) for frame in pixels.permute(0, 2, 3, 1).contiguous().numpy()]  # [h,w,3] each
    os.makedirs(os.path.dirname(path), exist_ok=True)
    images[0].save(path, save_all=True, append_images=images[1:], duration=round(1000 / fps), loop=0)


def _add_wandb_media(
    info: dict[str, Any],
    key_prefix: str,
    media: WandbMedia | None,
    caption: str,
) -> None:
    if media is None:
        return
    if isinstance(media, WandbAnimation):
        info[key_prefix] = wandb.Video(media.path, caption=f"{caption} | {media.num_frames} frames")
        return
    if isinstance(media, dict):
        for key_suffix, image_path in media.items():
            info[f"{key_prefix}_{key_suffix}"] = wandb.Image(image_path, caption=caption)
        return
    info[key_prefix] = wandb.Image(media, caption=caption)


def _pixel_tensor_to_5d(t: torch.Tensor) -> torch.Tensor:
    """Ensure a pixel tensor has shape (B, C, T, H, W) for the visualization grid.

    Handles (C, H, W), (B, C, H, W), and (B, C, T, H, W) inputs.
    """
    if t.ndim == 3:
        return t.unsqueeze(0).unsqueeze(2)  # (C,H,W) -> (1,C,1,H,W)
    if t.ndim == 4:
        return t.unsqueeze(2)  # (B,C,H,W) -> (B,C,1,H,W)
    return t


def _resize_5d_to_width(img5d: torch.Tensor, target_width: int) -> torch.Tensor:
    """Resize a single-frame (1, C, 1, H, W) tensor so its width is exactly ``target_width``.

    Height is scaled proportionally to preserve the overall aspect ratio. Assumes a
    single temporal frame (T == 1).
    """
    h, w = img5d.shape[-2], img5d.shape[-1]
    if w == target_width:
        return img5d
    new_h = max(1, round(h * target_width / w))
    chw = img5d[0, :, 0]  # (C,H,W)
    resized = torchvision_F.resize(chw, [new_h, target_width], antialias=True)  # (C,new_h,target_width)
    return resized.unsqueeze(0).unsqueeze(2)  # (1,C,1,new_h,target_width)


def _resize_pad_to_square(img5d: torch.Tensor, cell: int) -> torch.Tensor:
    """Resize a single-frame (1, C, 1, H, W) tensor to fit inside a ``cell`` x ``cell`` square.

    Aspect ratio is preserved (the image is scaled so its longer side equals ``cell``), then
    the result is center-padded with zeros to exactly ``cell`` x ``cell``. Assumes T == 1.
    """
    h, w = img5d.shape[-2], img5d.shape[-1]
    scale = cell / max(h, w)
    new_h = max(1, round(h * scale))
    new_w = max(1, round(w * scale))
    chw = img5d[0, :, 0]  # (C,H,W)
    resized = torchvision_F.resize(chw, [new_h, new_w], antialias=True)  # (C,new_h,new_w)
    pad_h = cell - new_h
    pad_w = cell - new_w
    top = pad_h // 2
    left = pad_w // 2
    padded = torch.nn.functional.pad(resized, (left, pad_w - left, top, pad_h - top))  # (C,cell,cell)
    return padded.unsqueeze(0).unsqueeze(2)  # (1,C,1,cell,cell)


def _build_reference_grid(references: list[torch.Tensor], target_width: int) -> torch.Tensor:
    """Tile reference images into a compact near-square grid sized to ``target_width``.

    All references are arranged in a ``rows`` x ``cols`` grid (``cols = ceil(sqrt(n))``), each in
    an aspect-preserved square cell, then the whole grid is resized so its width equals
    ``target_width``. This keeps the condition montage aligned with the generated-image column
    width (better space utilization) instead of one overly-long row.

    Args:
        references: list of single-frame pixel tensors (each (1,C,1,H,W) or (C,H,W)), in [-1, 1].
        target_width: width of the generated/target image for this sample.

    Returns:
        A (1, C, 1, H_grid, target_width) tensor.
    """
    if not references:
        raise ValueError("Expected at least one reference image to build the condition grid.")

    refs = [_pixel_tensor_to_5d(r) for r in references]  # list[(1,C,1,H,W)]
    n = len(refs)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    cell = max(1, target_width // cols)

    cells = [_resize_pad_to_square(r, cell) for r in refs]  # list[(1,C,1,cell,cell)]
    blank = torch.zeros_like(cells[0])
    while len(cells) < rows * cols:
        cells.append(blank)

    row_imgs = []
    for r in range(rows):
        row = torch.cat(cells[r * cols : (r + 1) * cols], dim=-1)  # (1,C,1,cell,cols*cell)
        row_imgs.append(row)
    grid = torch.cat(row_imgs, dim=-2)  # (1,C,1,rows*cell,cols*cell)

    return _resize_5d_to_width(grid, target_width)  # (1,C,1,H_grid,target_width)


def _synchronize_context_parallel_sampling_batch(
    model: Any,
    data_batch: dict[str, Any],
    n_viz_sample: int,
) -> dict[str, Any]:
    """Give every CP rank the same raw batch slice for sampling.

    CP-windowed training keeps a different raw batch on every rank and
    broadcasts only the current owner's post-tokenizer training payload. The
    sampling callback consumes the raw batch directly, so it must synchronize
    that input before any rank tokenizes or enters CP attention collectives.

    The model advances ``_cp_window_slot`` during the training step, before this
    callback runs. Therefore, the batch used by the just-finished step belongs
    to the preceding slot. Models without CP-window rotation use CP rank 0 as a
    stable source, matching the legacy synchronized-batch behavior.
    """
    sampling_batch = slice_data_batch(data_batch, start=0, limit=n_viz_sample)
    parallel_dims = getattr(model, "parallel_dims", None)
    if parallel_dims is None or not parallel_dims.cp_enabled:
        return sampling_batch

    cp_size = parallel_dims.cp_mesh.size()
    cp_window_slot = getattr(model, "_cp_window_slot", None)
    if cp_window_slot is None:
        owner_rank = 0
    else:
        if not isinstance(cp_window_slot, int):
            raise TypeError(f"CP data window slot must be an integer, got {type(cp_window_slot).__name__}.")
        if not 0 <= cp_window_slot < cp_size:
            raise ValueError(f"CP data window slot must be in [0, {cp_size}), got {cp_window_slot}.")
        owner_rank = (cp_window_slot - 1) % cp_size

    synchronized_batch = broadcast_context_parallel_object(
        sampling_batch,
        parallel_dims,
        owner_rank=owner_rank,
    )
    if not isinstance(synchronized_batch, dict):
        raise TypeError(f"Expected a dictionary CP sampling batch, got {type(synchronized_batch).__name__}.")
    return synchronized_batch


def _replica_identity(model: Any, rank: int) -> tuple[int, bool]:
    """Return this rank's sample-replica index and whether it owns that replica.

    CP and CFGP are overlay axes over the same sample: sampling gives every rank
    of a CP x CFGP group the identical batch (see
    ``_synchronize_context_parallel_sampling_batch``), so without this every rank
    in the group decodes the same video to CPU float32, writes the same S3 keys
    and multiplies host memory by the group size. The overlay mesh is built as
    ``(rest, cfgp, cp)``, so the replica index is the rank divided by the group
    size and the owner is the group's rank-0 slot.
    """
    parallel_dims = getattr(model, "parallel_dims", None)
    if parallel_dims is None:
        return rank, True

    cp_size = parallel_dims.cp_size if parallel_dims.cp_enabled else 1
    cfgp_size = parallel_dims.cfgp_size if parallel_dims.cfgp_enabled else 1
    group_size = cp_size * cfgp_size
    if group_size <= 1:
        return rank, True

    cp_rank = parallel_dims.cp_rank if parallel_dims.cp_enabled else 0
    cfgp_rank = parallel_dims.cfgp_rank if parallel_dims.cfgp_enabled else 0
    return rank // group_size, cp_rank == 0 and cfgp_rank == 0


class EveryNDrawSample(EveryN):
    """
    This callback sample condition inputs from training data, run inference and save the results to wandb and s3.

    Args:
        every_n (int): The frequency at which the callback is invoked.
        step_size (int, optional): The step size for the callback. Defaults to 1.
        n_viz_sample (int, optional): for each batch, min(n_viz_sample, batch_size) samples will be saved to wandb. Defaults to 3.
        n_sample_to_save (int, optional): number of samples to save. The actual number of samples to save is min(n_sample_to_save, data parallel instances). Defaults to 128.
        num_sampling_step (int, optional): number of sampling steps. Defaults to 35.
        guidance (list[float], optional): guidance scale. Defaults to [0.0, 3.0, 7.0].
        do_x0_prediction (bool, optional): whether to do x0 prediction. Defaults to True.
        n_sigmas_for_x0_prediction (int, optional): number of sigmas to use for x0 prediction. Defaults to 4.
        save_s3 (bool, optional): whether to save to s3. Defaults to False.
        is_ema (bool, optional): whether the callback is run for ema model. Defaults to False.
        use_negative_prompt (bool, optional): whether to use negative prompt. Defaults to False.
        fps (int, optional): frames per second when saving the video. Defaults to 16.
    """

    def __init__(
        self,
        every_n: int,
        step_size: int = 1,
        n_viz_sample: int = 2,
        n_sample_to_save: int = 128,
        num_sampling_step: int = 35,
        guidance: list[float] = [0.0, 3.0, 7.0],
        do_x0_prediction: bool = True,
        n_sigmas_for_x0_prediction: int = 4,
        save_s3: bool = False,
        save_local: bool = False,
        is_ema: bool = False,
        use_negative_prompt: bool = False,
        prompt_type: str = "t5_xxl",
        fps: int = 16,
        run_at_start: bool = False,
    ) -> None:
        # s3: # files: min(n_sample_to_save, data instance)  # per file: min(batch_size, n_viz_sample)
        # wandb: normal paths log one preview; multiview samples log one preview per selected timestamp.
        super().__init__(every_n, step_size, run_at_start=run_at_start)

        self.n_viz_sample = n_viz_sample
        self.n_sample_to_save = n_sample_to_save
        self.save_s3 = save_s3
        self.save_local = save_local
        self.do_x0_prediction = do_x0_prediction
        self.n_sigmas_for_x0_prediction = n_sigmas_for_x0_prediction
        self.name = self.__class__.__name__
        self.is_ema = is_ema
        self.use_negative_prompt = use_negative_prompt
        self.prompt_type = prompt_type
        self.guidance = guidance
        self.num_sampling_step = num_sampling_step
        self.rank = distributed.get_rank()
        self.fps = fps
        self.data_parallel_id = self.rank
        # Overwritten in on_train_start once the model's meshes are known.
        self.is_replica_leader = True

    def on_train_start(self, model: ImaginaireModel, iteration: int = 0) -> None:
        config_job = self.config.job
        self.local_dir = f"{config_job.path_local}/{self.name}"
        if distributed.get_rank() == 0:
            os.makedirs(self.local_dir, exist_ok=True)
            log.info(f"Callback: local_dir: {self.local_dir}")

        self.data_parallel_id, self.is_replica_leader = _replica_identity(model, self.rank)

    def _should_materialize_sample(self) -> bool:
        """Return whether this rank needs decoded pixels for saving or W&B."""
        if self.rank == 0:
            return True
        return self._should_save_to_s3() or self._should_save_local()

    def _should_save_to_s3(self) -> bool:
        """Return whether this rank owns the S3 artifacts for its sample replica."""
        return self.save_s3 and self.is_replica_leader and self.data_parallel_id < self.n_sample_to_save

    def _should_save_local(self) -> bool:
        """Return whether this rank owns the local artifacts for its sample replica."""
        return self.save_local and self.is_replica_leader and self.data_parallel_id < self.n_sample_to_save

    def _skip_multiview_visualization(
        self,
        reason: str,
        metadata: MultiviewTransferMetadata,
        iteration: int,
    ) -> MultiviewTransferSampleResult:
        if self._should_materialize_sample():
            log.warning(
                "Skipping multiview sampling visualization "
                f"at iteration {iteration}: {reason}. "
                f"num_vision_items={metadata.num_vision_items}, "
                f"sample_n_views={metadata.sample_n_views}, "
                f"num_video_frames_per_view={metadata.num_video_frames_per_view}.",
                rank0_only=False,
            )
        return MultiviewTransferSampleResult(handled=True)

    @misc.timer("EveryNDrawSample: x0")
    @torch.no_grad()
    def x0_pred(self, trainer, model, data_batch, output_batch, loss, iteration):
        tag = "ema" if self.is_ema else "reg"

        log.debug("starting data and condition model", rank0_only=False)
        data_clean = model.get_data_and_condition(data_batch)
        raw_data = data_clean.raw_state_vision
        x0 = data_clean.x0_tokens_vision

        # Handle model parallelism if available (legacy models)
        if hasattr(model, "broadcast_split_for_model_parallelsim"):
            _, condition, x0, _ = model.broadcast_split_for_model_parallelsim(None, None, x0, None)

        log.debug("done data and condition model", rank0_only=False)
        batch_size = len(x0)
        sigmas = np.exp(
            np.linspace(
                math.log(model.sde.sigma_min), math.log(model.sde.sigma_max), self.n_sigmas_for_x0_prediction + 1
            )[1:]
        )

        to_show = []
        generator = torch.Generator(device="cuda")
        generator.manual_seed(0)
        random_noise = torch.randn(*x0.shape, generator=generator, **model.tensor_kwargs)  # same shape as x0
        _ones = torch.ones(batch_size, **model.tensor_kwargs)  # [B]
        mse_loss_list = []
        for _, sigma in enumerate(sigmas):
            x_sigma = sigma * random_noise + x0
            log.debug(f"starting denoising {sigma}", rank0_only=False)
            sample = model.denoise(x_sigma, None).x0
            log.debug(f"done denoising {sigma}", rank0_only=False)
            mse_loss = distributed.dist_reduce_tensor(F.mse_loss(sample, x0))
            mse_loss_list.append(mse_loss)
            if hasattr(model, "decode"):
                sample = model.decode(sample)
            to_show.append(sample.float().cpu())
        to_show.append(
            raw_data.float().cpu(),
        )

        base_fp_wo_ext = f"{tag}_ReplicateID{self.data_parallel_id:04d}_x0_Iter{iteration:09d}"

        local_path = self.run_save(to_show, batch_size, base_fp_wo_ext)
        return local_path, torch.tensor(mse_loss_list).cuda(), sigmas  # [N_sigmas]

    @torch.no_grad()
    def every_n_impl(
        self,
        trainer: Any,
        model: Any,
        data_batch: dict[str, Any],
        output_batch: Any,
        loss: Any,
        iteration: int,
    ) -> None:
        if self.is_ema:
            if not model.config.ema.enabled:
                return
            context = partial(model.ema_scope, "every_n_sampling")
        else:
            context = nullcontext

        data_batch = _synchronize_context_parallel_sampling_batch(model, data_batch, self.n_viz_sample)

        tag = "ema" if self.is_ema else "reg"
        sample_counter = getattr(trainer, "sample_counter", iteration)
        batch_info = {
            "data": {
                k: convert_to_primitive(v)
                for k, v in data_batch.items()
                if is_primitive(v) or isinstance(v, (list, dict))
            },
            "sample_counter": sample_counter,
            "iteration": iteration,
        }
        if self._should_save_to_s3():
            easy_io.dump(
                batch_info,
                f"s3://rundir/{self.name}/Iter{iteration:09d}/BatchInfo_ReplicateID{self.data_parallel_id:04d}_Iter{iteration:09d}.json",
            )

        log.debug("entering, every_n_impl", rank0_only=False)
        with context():
            if self.do_x0_prediction:
                log.debug("entering, x0_pred", rank0_only=False)
                x0_img_fp, mse_loss, sigmas = self.x0_pred(
                    trainer,
                    model,
                    data_batch,
                    output_batch,
                    loss,
                    iteration,
                )
                log.debug("done, x0_pred", rank0_only=False)
                if self.save_s3 and self.rank == 0:
                    easy_io.dump(
                        {
                            "mse_loss": mse_loss.tolist(),
                            "sigmas": sigmas.tolist(),
                            "iteration": iteration,
                        },
                        f"s3://rundir/{self.name}/{tag}_MSE_Iter{iteration:09d}.json",
                    )

            log.debug("entering, sample", rank0_only=False)
            sample_img_fp = self.sample(
                trainer,
                model,
                data_batch,
                output_batch,
                loss,
                iteration,
            )
            log.debug("done, sample", rank0_only=False)

            log.debug("waiting for all ranks to finish", rank0_only=False)
            dist.barrier()
        # Only rank 0 initializes W&B. Logging the GIF/video here while other ranks have
        # already left the callback used to hold the GIL across the EveryN barrier and
        # amplify NCCL stragglers after LiDAR transfer sampling.
        if distributed.is_rank0() and wandb.run:
            sample_counter = getattr(trainer, "sample_counter", iteration)
            data_type = "image" if model.is_image_batch(data_batch) else "video"
            tag += f"_{data_type}"
            info = {
                "trainer/global_step": iteration,
                "sample_counter": sample_counter,
            }
            if self.do_x0_prediction:
                _add_wandb_media(info, f"{self.name}/{tag}_x0", x0_img_fp, f"{sample_counter}")
                # convert mse_loss to a dict
                mse_loss = mse_loss.tolist()
                info.update({f"x0_pred_mse_{tag}/Sigma{sigmas[i]:0.5f}": mse_loss[i] for i in range(len(mse_loss))})

            _add_wandb_media(info, f"{self.name}/{tag}_sample", sample_img_fp, f"{sample_counter}")
            wandb.log(
                info,
                step=iteration,
            )
        torch.cuda.empty_cache()

    def _generate_transfer_latents(
        self,
        model: Any,
        data_batch: dict[str, Any],
        metadata: MultiviewTransferMetadata,
        iteration: int,
    ) -> TransferGeneration:
        """Sample the first clip of the batch once per guidance scale.

        IMPORTANT: this runs BEFORE any auxiliary VAE decode, and callers must keep it that
        way. generate_samples_from_batch drives the compiled net under
        torch.compiler.cudagraph_mark_step_begin(); interposing a large VAE decode (e.g. the
        clean-x0 reconstruction row) between the previous cudagraph step and the sampler
        perturbs the captured cudagraph memory pool and collapses the generated latents to
        ~undenoised noise. Decoding display rows only after this returns (matching the
        standard single-item sample() path ordering) keeps generation bit-for-bit as in the
        runs that produced good zero-shot output.
        """
        should_materialize_sample = self._should_materialize_sample()
        latents: list[torch.Tensor] = []
        generation_batch = slice_data_batch(data_batch, start=0, limit=1)
        for guidance in self.guidance:
            sample = model.generate_samples_from_batch(
                generation_batch,
                guidance=guidance,
                n_sample=1,
                num_steps=self.num_sampling_step,
                has_negative_prompt=True if self.use_negative_prompt else False,
                seed=[iteration],
            )
            sample_vision = sample["vision"]
            if len(sample_vision) != 1:
                return TransferGeneration(
                    latents=[],
                    stop=self._skip_multiview_visualization(
                        f"expected one generated vision tensor, got {len(sample_vision)}",
                        metadata,
                        iteration,
                    ),
                )
            if should_materialize_sample:
                latents.append(
                    sample_vision[0].clone()  # [1,C,V*T_latent,H,W] or [C,V*T_latent,H,W]
                )

        # Sampling drives CP/FSDP collectives, so every rank must finish every guidance call
        # above. Pixel decoding and visualization are local work for replica leaders only;
        # followers return empty latents and must still join the post-materialize barrier in
        # the caller before ``every_n_impl``'s NCCL barrier.
        if not should_materialize_sample:
            return TransferGeneration(latents=[])
        return TransferGeneration(latents=latents)

    def _transfer_artifact_path(self, tag: str, iteration: int) -> str:
        """Path stem shared by the S3, local and W&B artifacts of one drawn sample."""
        return f"Iter{iteration:09d}/{tag}_ReplicateID{self.data_parallel_id:04d}_Sample_Iter{iteration:09d}"

    def _barrier_after_transfer_materialize(self) -> None:
        """Wait until replica leaders finish decode/save before any rank leaves ``sample()``.

        CP followers used to return from transfer sampling immediately after
        ``generate_samples_from_batch`` and then hit ``every_n_impl``'s NCCL barrier while
        leaders were still decoding 96-frame LiDAR clips and writing GIFs. That barrier
        then timed out and aborted the communicator; the next training-step collective
        surfaced as ``DistBackendError`` inside ``broadcast_context_parallel_object``.
        """
        if dist.is_initialized():
            dist.barrier()

    def _sample_multiview_transfer(
        self,
        model: Any,
        data_batch: dict[str, Any],
        raw_data: list[torch.Tensor] | None,
        x0: list[torch.Tensor] | None,
        metadata: MultiviewTransferMetadata,
        iteration: int,
        tag: str,
    ) -> MultiviewTransferSampleResult:
        """Draw a camera-rig transfer sample: one grid column per view, three frames in W&B.

        Rows are [control, GT, clean recon, generated]; W&B gets one panel per sampled frame
        so that a rig of 7+ views stays legible at panel size.
        """
        if not _has_first_multiview_transfer_rows(raw_data, metadata):
            return self._skip_multiview_visualization(
                "raw multiview rows cannot be split into camera views",
                metadata,
                iteration,
            )

        generation = self._generate_transfer_latents(model, data_batch, metadata, iteration)
        if generation.stop is not None:
            return generation.stop

        media: WandbMedia | None = None
        skip_reason: str | None = None
        if self._should_materialize_sample():
            to_show = _get_first_transfer_dataloader_rows(raw_data, metadata)  # list[[V,C,F,H,W]] or None
            if to_show is None:
                skip_reason = "ground-truth target row cannot be split into camera views"
            else:
                generated_rows: list[torch.Tensor] = []
                for generated_latent in generation.latents:
                    generated_row = _decode_transfer_display_row(
                        model, generated_latent, data_batch, metadata
                    )  # [V,C,F,H,W] or None
                    if generated_row is None:
                        skip_reason = "generated video cannot be split into camera views"
                        break
                    generated_rows.append(generated_row)  # [V,C,F,H,W]

                if skip_reason is None:
                    # VAE reconstruction of the clean target latent (decode of the x0 tokens). This is the
                    # tokenizer reconstruction ceiling — the best the model could produce if generation were
                    # perfect — so it isolates VAE loss from diffusion generation quality. Decoded after the
                    # generated rows (see the note in _generate_transfer_latents) but shown before them, so
                    # the display order stays [control, GT, clean recon, generated].
                    if x0 is not None and len(x0) >= metadata.num_vision_items:
                        clean_target_row = _decode_transfer_display_row(  # [V,C,F,H,W] or None
                            model,
                            x0[metadata.num_vision_items - 1],
                            data_batch,
                            metadata,
                        )
                        if clean_target_row is not None:
                            to_show.append(clean_target_row)

                    to_show.extend(generated_rows)

                    row_shapes = _mismatched_row_shapes(to_show)
                    if row_shapes is not None:
                        skip_reason = f"visualization rows have inconsistent shapes: {row_shapes}"
                    else:
                        media = self.run_save(
                            to_show,
                            metadata.sample_n_views,
                            self._transfer_artifact_path(tag, iteration),
                            max_columns=metadata.sample_n_views,
                            wandb_clip_preview="frames",
                        )

        self._barrier_after_transfer_materialize()
        if skip_reason is not None:
            return self._skip_multiview_visualization(skip_reason, metadata, iteration)
        return MultiviewTransferSampleResult(handled=True, media=media)

    def _sample_lidar_transfer(
        self,
        model: Any,
        data_batch: dict[str, Any],
        raw_data: list[torch.Tensor] | None,
        x0: list[torch.Tensor] | None,
        metadata: MultiviewTransferMetadata,
        iteration: int,
        tag: str,
    ) -> MultiviewTransferSampleResult:
        """Draw an HD-map to LiDAR transfer sample: colorized range views, played as a clip.

        Rows are [control, GT, clean recon, generated] as for camera rigs, but a range clip is
        a single wide view whose whole point is how the sweep evolves, so the rows are
        colorized from one-channel range, labelled with what they hold, and W&B gets the full
        clip as an animation instead of three sampled frames.
        """
        if not _has_first_multiview_transfer_rows(raw_data, metadata):
            return self._skip_multiview_visualization(
                "raw range rows cannot be read from the batch",
                metadata,
                iteration,
            )

        generation = self._generate_transfer_latents(model, data_batch, metadata, iteration)
        if generation.stop is not None:
            return generation.stop

        media: WandbMedia | None = None
        skip_reason: str | None = None
        if self._should_materialize_sample():
            to_show = _get_first_transfer_dataloader_rows(  # list[[V,C,F,H,W]] or None
                raw_data,
                metadata,
                colorize_lidar=True,
            )
            if to_show is None:
                skip_reason = "ground-truth range clip cannot be read from the batch"
            else:
                # One label per row, in row order, so a stack of near-identical range maps stays readable.
                row_labels = ["control signal (HD map)", "real LiDAR"] if len(to_show) == 2 else ["real LiDAR"]

                generated_rows: list[torch.Tensor] = []
                generated_row_labels: list[str] = []
                for guidance, generated_latent in zip(self.guidance, generation.latents, strict=True):
                    generated_row = _decode_transfer_display_row(  # [V,C,F,H,W] or None
                        model,
                        generated_latent,
                        data_batch,
                        metadata,
                        colorize_lidar=True,
                    )
                    if generated_row is None:
                        skip_reason = "generated range clip cannot be read from the decoder"
                        break
                    generated_rows.append(generated_row)  # [V,C,F,H,W]
                    generated_row_labels.append(f"generated LiDAR (guidance {guidance})")

                if skip_reason is None:
                    # Tokenizer reconstruction ceiling for the target sweeps: how close the range VAE can
                    # get on its own, which is what separates tokenizer loss from generation quality.
                    # Decoded after the generated rows, shown before them (see _generate_transfer_latents).
                    if x0 is not None and len(x0) >= metadata.num_vision_items:
                        clean_target_row = _decode_transfer_display_row(  # [V,C,F,H,W] or None
                            model,
                            x0[metadata.num_vision_items - 1],
                            data_batch,
                            metadata,
                            colorize_lidar=True,
                        )
                        if clean_target_row is not None:
                            to_show.append(clean_target_row)
                            row_labels.append("decoded clean tokens")

                    to_show.extend(generated_rows)
                    row_labels.extend(generated_row_labels)

                    row_shapes = _mismatched_row_shapes(to_show)
                    if row_shapes is not None:
                        skip_reason = f"visualization rows have inconsistent shapes: {row_shapes}"
                    else:
                        for row, row_label in zip(to_show, row_labels, strict=True):
                            _stamp_row_label(row, row_label)

                        media = self.run_save(
                            to_show,
                            metadata.sample_n_views,
                            self._transfer_artifact_path(tag, iteration),
                            max_columns=metadata.sample_n_views,
                            wandb_clip_preview="animation",
                        )

        self._barrier_after_transfer_materialize()
        if skip_reason is not None:
            return self._skip_multiview_visualization(skip_reason, metadata, iteration)
        return MultiviewTransferSampleResult(handled=True, media=media)

    @misc.timer("EveryNDrawSample: sample")
    def sample(
        self,
        trainer: Any,
        model: Any,
        data_batch: dict[str, Any],
        output_batch: Any,
        loss: Any,
        iteration: int,
    ) -> WandbMedia | None:
        data_batch = slice_data_batch(data_batch, start=0, limit=self.n_viz_sample)

        tag = "ema" if self.is_ema else "reg"

        # Obtain text embeddings online
        text_encoder_config = getattr(model.config, "text_encoder_config", None)
        if text_encoder_config is not None and text_encoder_config.compute_online:
            text_embeddings = model.text_encoder.compute_text_embeddings_online(data_batch, model.input_caption_key)
            data_batch["t5_text_embeddings"] = text_embeddings
            data_batch["t5_text_mask"] = torch.ones(
                text_embeddings.shape[0], text_embeddings.shape[1], device="cuda"
            )  # [B,N_tokens]  (all tokens valid)

        per_camera_vae_encoding = "enable_per_camera_vae_encoding" in data_batch
        data_clean = model.get_data_and_condition(
            data_batch,
            retain_raw_state_vision=not per_camera_vae_encoding,
        )
        raw_data = data_clean.raw_state_vision
        x0 = data_clean.x0_tokens_vision

        # determine the number of visualized samples
        n_viz_sample = min(self.n_viz_sample, data_clean.batch_size)

        # Check if this is a multi-item vision batch (image editing)
        num_items = data_clean.num_vision_items_per_sample
        is_multi_item = num_items is not None
        multiview_num_items = _get_multiview_visualization_item_counts(data_batch, num_items, data_clean.batch_size)
        transfer_metadata = _get_multiview_transfer_metadata(data_batch, multiview_num_items)
        if transfer_metadata is not None:
            draw_transfer_sample = (
                self._sample_lidar_transfer
                if _is_lidar_visualization_batch(data_batch)
                else self._sample_multiview_transfer
            )
            transfer_result = draw_transfer_sample(
                model,
                data_batch,
                raw_data,
                x0,
                transfer_metadata,
                iteration,
                tag,
            )
            if transfer_result.handled:
                return transfer_result.media

        if is_multi_item:
            # Image editing: raw_data is flat [src1, tgt1, src2, tgt2, ...].
            # Split into per-sample condition (source) and GT target images.
            condition_images: list[torch.Tensor] = []
            gt_target_images: list[torch.Tensor] = []
            gt_target_latents: list[torch.Tensor] = []
            vis_offset = 0
            for sample_idx in range(data_clean.batch_size):
                n_vis = num_items[sample_idx]
                # First item(s) are condition references, last item is the generation target.
                refs = raw_data[vis_offset : vis_offset + n_vis - 1]  # all condition items
                target = raw_data[vis_offset + n_vis - 1]  # target image (1, C, 1, H, W)
                # Multi-reference generation (>1 single-frame image references): tile every
                # reference into a compact grid resized to the target width, so all references
                # are visible without blowing up the row width. For video editing/transfer
                # (T > 1) keep the existing behavior (first item = condition) so those tasks
                # render exactly as before and stay consistent with the t_crop frame cropping.
                refs_are_images = all(r.shape[-3] == 1 for r in refs) and target.shape[-3] == 1
                if refs_are_images and len(refs) > 1:
                    condition_images.append(_build_reference_grid(refs, target.shape[-1]))
                else:
                    condition_images.append(raw_data[vis_offset])  # source image (1, C, 1, H, W) / video clip
                gt_target_images.append(target)
                # x0 (clean vision latents) can be None when the model/training setup does not
                # populate x0_tokens_vision; only collect target latents when they are available.
                if x0 is not None:
                    gt_target_latents.append(x0[vis_offset + n_vis - 1])  # target latent (1, C, 1, H, W)
                vis_offset += n_vis

            # Use target images for max_w/max_h/t_crop (generated samples match target size)
            max_w = max(img.shape[-1] for img in gt_target_images)
            max_h = max(img.shape[-2] for img in gt_target_images)
            t_crop = min(img.shape[-3] for img in gt_target_images)
        else:
            max_w = max(image.shape[-1] for image in raw_data)
            max_h = max(image.shape[-2] for image in raw_data)
            t_crop = min(image.shape[-3] for image in raw_data)

        to_show = []

        # Row 0 (image editing only): condition (source) images
        if is_multi_item:
            to_show.append(pad_images_and_cat(condition_images[:n_viz_sample], max_w, max_h, t_crop).float().cpu())

        for guidance in self.guidance:
            sample = model.generate_samples_from_batch(
                data_batch,
                guidance=guidance,
                n_sample=n_viz_sample,
                num_steps=self.num_sampling_step,
                has_negative_prompt=True if self.use_negative_prompt else False,
                seed=list(range(iteration, iteration + n_viz_sample)),
            )
            sample_vision = sample["vision"]
            assert hasattr(model, "decode")
            sample_vision_decoded = [model.decode(sample_vision_i) for sample_vision_i in sample_vision]
            assert len(sample_vision_decoded) == n_viz_sample
            to_show.append(pad_images_and_cat(sample_vision_decoded, max_w, max_h, t_crop).float().cpu())

        # Penultimate row: VAE reconstruction of the clean latents (decode of the x0 tokens).
        # This is the tokenizer reconstruction ceiling — how much detail is lost by encode+decode
        # alone — so it separates VAE loss from the diffusion model's generation quality.
        # x0 (clean vision latents) can be None when the model/training setup does not populate
        # x0_tokens_vision; skip the clean-recon row entirely in that case.
        if x0 is not None:
            assert hasattr(model, "decode")
            if is_multi_item:
                clean_token_decoded = [model.decode(latent) for latent in gt_target_latents]
            else:
                clean_token_decoded = [model.decode(latent) for latent in x0[:n_viz_sample]]
            to_show.append(pad_images_and_cat(clean_token_decoded, max_w, max_h, t_crop).float().cpu())

        # Last row: ground truth
        if is_multi_item:
            # Image editing: show GT target images (not the flat raw_data which mixes src + tgt)
            assert len(gt_target_images) == n_viz_sample
            to_show.append(pad_images_and_cat(gt_target_images, max_w, max_h, t_crop).float().cpu())
        else:
            assert len(raw_data) == n_viz_sample
            to_show.append(pad_images_and_cat(raw_data, max_w, max_h, t_crop).float().cpu())

        base_fp_wo_ext = f"{tag}_ReplicateID{self.data_parallel_id:04d}_Sample_Iter{iteration:09d}"
        base_fp_wo_ext = f"Iter{iteration:09d}/{base_fp_wo_ext}"

        batch_size = data_clean.batch_size
        local_path = self.run_save(to_show, batch_size, base_fp_wo_ext)
        return local_path

    def run_save(
        self,
        to_show: list[torch.Tensor],
        batch_size: int,
        base_fp_wo_ext: str,
        max_columns: int | None = None,
        wandb_clip_preview: WandbClipPreview = "grid",
    ) -> WandbMedia | None:
        to_show = _stack_rows_for_display(to_show)  # [N_rows,B,C,T,H,W]  uint8, or float in [0,1]
        is_single_frame = to_show.shape[3] == 1
        max_columns = self.n_viz_sample if max_columns is None else max_columns
        n_columns = min(max_columns, batch_size)
        to_show = to_show[:, :n_columns]

        # ! we only save first n_sample_to_save video!
        video_grid = rearrange(to_show, "n b c t h w -> c t (n h) (b w)")  # [C,T,N_rows*H,B*W]
        if self._should_save_to_s3():
            save_img_or_video(
                video_grid,
                f"s3://rundir/{self.name}/{base_fp_wo_ext}",
                fps=self.fps,
            )
        if self._should_save_local():
            local_video_path = f"{self.local_dir}/{base_fp_wo_ext}"
            os.makedirs(os.path.dirname(local_video_path), exist_ok=True)
            save_img_or_video(video_grid, local_video_path, fps=self.fps)

        file_base_fp = f"{base_fp_wo_ext}_resize.jpg"
        local_path = f"{self.local_dir}/{file_base_fp}"

        if self.rank == 0 and wandb.run:
            if is_single_frame:  # image case
                to_show = rearrange(
                    to_show[:, :n_columns],
                    "n b c t h w -> t c (n h) (b w)",
                )  # [1,C,N_rows*H,B*W]  (t=1 for images)
                image_grid = torchvision.utils.make_grid(
                    _to_unit_float(to_show), nrow=1, padding=0, normalize=False
                )  # [C,N_rows*H,B*W]
                # resize so that wandb can handle it
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                resized_image = resize_image(image_grid, 1024)  # [C,H_resize,W_resize]
                torchvision.utils.save_image(resized_image, local_path, nrow=1, scale_each=True)
            else:
                to_show = to_show[:, :n_columns]  # [N_rows,B,C,T,H,W]

                # resize 3 frames frames so that we can display them on wandb
                _T = to_show.shape[3]
                three_frames_list = [0, _T // 2, _T - 1]
                log_image_size = 1024
                os.makedirs(os.path.dirname(local_path), exist_ok=True)

                if wandb_clip_preview == "animation":
                    animation_frames = rearrange(
                        to_show,
                        "n b c t h w -> t c (n h) (b w)",
                    )  # [T,C,N_rows*H,B*W]
                    animation_path = f"{self.local_dir}/{base_fp_wo_ext}.gif"
                    _write_gif(animation_frames, animation_path, fps=self.fps, max_size=log_image_size)
                    return WandbAnimation(animation_path, _T)

                if wandb_clip_preview == "frames":
                    frame_paths: dict[str, str] = {}
                    for frame_name, frame_idx in zip(
                        ("frame_first", "frame_mid", "frame_last"),
                        three_frames_list,
                        strict=True,
                    ):
                        frame_to_show = to_show[:, :, :, frame_idx]  # [N_rows,B,C,H,W]
                        frame_to_show = rearrange(
                            frame_to_show,
                            "n b c h w -> 1 c (n h) (b w)",
                        )  # [1,C,N_rows*H,B*W]
                        frame_image_grid = torchvision.utils.make_grid(
                            _to_unit_float(frame_to_show),
                            nrow=1,
                            padding=0,
                            normalize=False,
                        )  # [C,N_rows*H,B*W]
                        frame_path = f"{self.local_dir}/{base_fp_wo_ext}_{frame_name}_resize.jpg"
                        resized_frame = resize_image(frame_image_grid, log_image_size)  # [C,H_resize,W_resize]
                        torchvision.utils.save_image(resized_frame, frame_path, nrow=1, scale_each=True)
                        frame_paths[frame_name] = frame_path
                    return frame_paths

                to_show = to_show[:, :, :, three_frames_list]  # [N_rows,B,C,3,H,W]  (3 sampled frames)
                to_show = rearrange(
                    to_show,
                    "n b c t h w -> 1 c (n h) (b t w)",
                )  # [1,C,N_rows*H,B*3*W]  (t=3 sampled frames)

                # resize so that wandb can handle it
                image_grid = torchvision.utils.make_grid(
                    _to_unit_float(to_show),
                    nrow=1,
                    padding=0,
                    normalize=False,
                )  # [C,N_rows*H,B*3*W]
                resized_image = resize_image(image_grid, log_image_size)  # [C,H_resize,W_resize]
                torchvision.utils.save_image(resized_image, local_path, nrow=1, scale_each=True)

            return local_path
        return None
