# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Framework-native processor for renewed ``nvidia/Cosmos3-Edge`` snapshots.

Port of the deleted remote-code ``processing.py`` from ``nvidia/Cosmos3-Edge``
rev ``28a0b8e``. Renewed snapshots ship native ``cosmos3_edge`` metadata with no
remote code, and ``AutoProcessor`` on transformers 4.x silently degrades to a
bare tokenizer for them; this module rebuilds the full processor from the
snapshot files without ``trust_remote_code``.

WARNING — pixel patch layout: patches must stay in the original convention
(raster patch-row order, ``(py, px, c)`` within-row flatten). transformers-main's
native ``Cosmos3EdgeProcessor`` emits a different layout (2x2 block-major,
channel-major within-row) that corrupts vision features with these checkpoint
weights. Do NOT "align" the patchify code with transformers main. Details:
``outputs/audit/cosmos3_edge_native_vision_layout_bug.md``.
"""

# NOTE: no `from __future__ import annotations` here — ProcessorMixin._merge_kwargs
# introspects Qwen3VLProcessorKwargs.__annotations__ at runtime and PEP 563
# stringified annotations (ForwardRef) break it.
import json
import math
import os
from typing import Optional, Union

import numpy as np
import torch
from torchvision.transforms.v2 import functional as F
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_processing_utils_fast import SizeDict
from transformers.image_utils import ChannelDimension, ImageInput, PILImageResampling
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.qwen3_vl.video_processing_qwen3_vl import (
    Qwen3VLVideoProcessor,
    Qwen3VLVideoProcessorInitKwargs,
    get_image_size,
    smart_resize,
)
from transformers.models.siglip2.image_processing_siglip2_fast import (
    Siglip2ImageProcessorFast,
    convert_image_to_patches,
)
from transformers.processing_utils import MultiModalData, ProcessingKwargs, ProcessorMixin, Unpack
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.utils import TensorType, logging
from transformers.video_utils import VideoInput, group_videos_by_shape, reorder_videos

logger = logging.get_logger(__name__)

# Sub-processor config keys that name (remote-code or transformers-main) classes.
# Only the parameter VALUES are consumed; the framework picks the classes itself.
_CLASS_NAME_KEYS = ("processor_class", "image_processor_type", "video_processor_type", "auto_map")


def is_cosmos3_edge_native_snapshot(model_dir: str) -> bool:
    """Return True for a renewed (native-metadata, no remote code) Cosmos3-Edge snapshot.

    Detection rule: ``config.json`` says ``model_type == "cosmos3_edge"``, or a
    ``processor_config.json`` names the native ``Cosmos3EdgeProcessor``. Old
    snapshots (``model_type == "nemotron_siglip2"``, remote-code ``processing.py``)
    and all other models match neither and keep the ``AutoProcessor`` path.
    """
    if not os.path.isdir(model_dir):
        return False
    config_path = os.path.join(model_dir, "config.json")
    if os.path.isfile(config_path):
        try:
            with open(config_path) as f:
                if json.load(f).get("model_type") == "cosmos3_edge":
                    return True
        except (OSError, ValueError):
            pass
    processor_config_path = os.path.join(model_dir, "processor_config.json")
    if os.path.isfile(processor_config_path):
        try:
            with open(processor_config_path) as f:
                if json.load(f).get("processor_class") == "Cosmos3EdgeProcessor":
                    return True
        except (OSError, ValueError):
            pass
    return False


def build_cosmos3_edge_processor(model_dir: str) -> "NemotronNanoV3BridgeProcessor":
    """Construct the full Cosmos3-Edge processor from a snapshot dir, without remote code.

    Reproduces the old remote-code ``AutoProcessor`` object exactly (golden-pinned
    by ``cosmos3_edge_processing_test.py``): the tokenizer, chat template, and
    sub-processor parameter files are unchanged upstream since rev ``28a0b8e``.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    # The renewed tokenizer_config.json adds ``return_mm_token_type_ids: true``
    # (a transformers-main knob); left in init_kwargs it would override the
    # text_kwargs default in _merge_kwargs and add an ``mm_token_type_ids`` key
    # the old processor never returned.
    tokenizer.init_kwargs.pop("return_mm_token_type_ids", None)

    # Wire the processor-level template explicitly; the tokenizer does not
    # reliably pick up chat_template.jinja on its own.
    with open(os.path.join(model_dir, "chat_template.jinja")) as f:
        chat_template = f.read()

    def _load_sub_config(file_name: str) -> dict:
        with open(os.path.join(model_dir, file_name)) as f:
            config = json.load(f)
        for key in _CLASS_NAME_KEYS:
            config.pop(key, None)
        return config

    image_processor = Siglip2ImageProcessorCustom(**_load_sub_config("preprocessor_config.json"))
    video_processor = Qwen3VLVideoProcessorCustom(**_load_sub_config("video_preprocessor_config.json"))
    return NemotronNanoV3BridgeProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        video_processor=video_processor,
        chat_template=chat_template,
    )


# Everything below is vendored from the rev 28a0b8e remote code; kept faithful
# so outputs match the HF reference bit-exactly.


def round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor


class NemotronImagesKwargs(Siglip2ImageProcessorFast.valid_kwargs, total=False):
    # global setting for all images, can be overridden by per-image kwargs
    max_pixels: Optional[int]
    min_pixels: Optional[int]

    # per-image overrides, e.g. [{"min_pixels": 256*28*28, "max_pixels": 1280*28*28}, None, ...]
    per_image_kwargs: Optional[list[dict]]


class Qwen3VLProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: NemotronImagesKwargs
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "return_token_type_ids": False,
            "return_mm_token_type_ids": False,
        },
        "videos_kwargs": {"return_metadata": True},
    }


class Siglip2ImageProcessorCustom(Siglip2ImageProcessorFast):
    resample = PILImageResampling.BICUBIC
    valid_kwargs = NemotronImagesKwargs

    def __init__(self, **kwargs: Unpack[NemotronImagesKwargs]):
        super().__init__(**kwargs)

    def _resize_image(
        self, image: torch.Tensor, max_ratio=200, interpolation: Optional[PILImageResampling] = None, **kwargs
    ) -> torch.Tensor:
        """Resize so both dims are divisible by merge_size * patch_size and the
        pixel count lands in [min_pixels, max_pixels], preserving aspect ratio."""
        image_min_pixels = kwargs.get("min_pixels", None) or self.size.get("shortest_edge", None)
        image_max_pixels = kwargs.get("max_pixels", None) or self.size.get("longest_edge", None)
        assert image_min_pixels is not None and image_max_pixels is not None, (
            "When do_resize is True, min_pixels and max_pixels must be provided."
        )
        assert image_max_pixels >= image_min_pixels, (
            "The max_pixels of image must be greater than or equal to min_pixels."
        )

        _, height, width = image.shape
        if max(height, width) / min(height, width) > max_ratio:
            raise ValueError(
                f"absolute aspect ratio must be smaller than {max_ratio}, got {max(height, width) / min(height, width)}"
            )
        factor = self.merge_size * self.patch_size
        h_bar = max(factor, round_by_factor(height, factor))
        w_bar = max(factor, round_by_factor(width, factor))
        if h_bar * w_bar > image_max_pixels:
            beta = math.sqrt((height * width) / image_max_pixels)
            h_bar = floor_by_factor(height / beta, factor)
            w_bar = floor_by_factor(width / beta, factor)
        elif h_bar * w_bar < image_min_pixels:
            beta = math.sqrt(image_min_pixels / (height * width))
            h_bar = ceil_by_factor(height * beta, factor)
            w_bar = ceil_by_factor(width * beta, factor)

        image = self.resize(image, size=SizeDict(height=h_bar, width=w_bar), interpolation=interpolation)
        return image

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        patch_size: int,
        max_num_patches: int,
        interpolation: Optional["F.InterpolationMode"],
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: Optional[Union[float, list[float]]],
        image_std: Optional[Union[float, list[float]]],
        return_tensors: Optional[Union[str, TensorType]],
        **kwargs,
    ) -> BatchFeature:
        per_image_kwargs = kwargs.pop("per_image_kwargs", None)
        pixel_values = []
        spatial_shapes = []
        for idx, image in enumerate(images):
            image_kwargs = kwargs.copy()
            if per_image_kwargs is not None and idx < len(per_image_kwargs) and per_image_kwargs[idx]:
                image_kwargs.update(per_image_kwargs[idx])
            if do_resize:
                image = self._resize_image(image, max_ratio=200, interpolation=interpolation, **image_kwargs)

            image = self.rescale_and_normalize(image, do_rescale, rescale_factor, do_normalize, image_mean, image_std)

            # (num_channels, height, width) -> (num_patches, patch_size * patch_size * num_channels)
            # Raster patch-row order, (py, px, c) within-row flatten — the layout
            # the checkpoint's patch_embedding weight was trained for (see module
            # docstring); transformers-main's native processor differs here.
            patches = convert_image_to_patches(image, patch_size)

            num_patches_height = image.shape[1] // patch_size
            num_patches_width = image.shape[2] // patch_size

            spatial_shapes.append((num_patches_height, num_patches_width))
            pixel_values.append(patches)

        spatial_shapes = torch.tensor(spatial_shapes)

        batch_feature = BatchFeature(
            data={
                "pixel_values": torch.cat(pixel_values, dim=0),
                "spatial_shapes": spatial_shapes,
            },
            tensor_type=return_tensors,
        )
        return batch_feature


class Qwen3VLVideoProcessorCustom(Qwen3VLVideoProcessor):
    def __init__(self, **kwargs: Unpack[Qwen3VLVideoProcessorInitKwargs]):
        super().__init__(**kwargs)

    def _preprocess(
        self,
        videos: list[torch.Tensor],
        do_convert_rgb: bool = True,
        do_resize: bool = True,
        size: Optional[SizeDict] = None,
        interpolation: PILImageResampling = PILImageResampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255.0,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
        patch_size: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ):
        merge_size = self.merge_size
        grouped_videos, grouped_videos_index = group_videos_by_shape(videos)
        resized_videos_grouped = {}

        for shape, stacked_videos in grouped_videos.items():
            B, T, C, H, W = stacked_videos.shape
            num_frames, height, width = T, H, W
            if do_resize:
                resized_height, resized_width = smart_resize(
                    num_frames=num_frames,
                    height=height,
                    width=width,
                    temporal_factor=1,
                    factor=patch_size * merge_size,
                    min_pixels=size.shortest_edge,
                    max_pixels=size.longest_edge,
                )
                stacked_videos = stacked_videos.view(B * T, C, H, W)
                stacked_videos = self.resize(
                    stacked_videos,
                    size=SizeDict(height=resized_height, width=resized_width),
                    interpolation=interpolation,
                )
                stacked_videos = stacked_videos.view(B, T, C, resized_height, resized_width)
            resized_videos_grouped[shape] = stacked_videos
        resized_videos = reorder_videos(resized_videos_grouped, grouped_videos_index)

        # Re-group: sizes may still differ (do_resize False, or per-video resize targets)
        grouped_videos, grouped_videos_index = group_videos_by_shape(resized_videos)
        processed_videos_grouped = {}
        processed_grids = {}
        for shape, stacked_videos in grouped_videos.items():
            resized_height, resized_width = get_image_size(stacked_videos[0], channel_dim=ChannelDimension.FIRST)
            stacked_videos = self.rescale_and_normalize(
                stacked_videos, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            patches = stacked_videos

            batch_size, grid_t, channel = patches.shape[:3]
            grid_h, grid_w = resized_height // patch_size, resized_width // patch_size

            patches = patches.view(
                batch_size,  # 0
                grid_t,  # 1
                channel,  # 2
                grid_h,  # 3
                patch_size,  # 4
                grid_w,  # 5
                patch_size,  # 6
            )
            # -> [batch_size, grid_t, grid_h, grid_w, patch_size, patch_size, channel]:
            # raster patch rows, channel-LAST within-row — the old-convention layout
            # the checkpoint weights expect (see module docstring).
            patches = patches.permute(0, 1, 3, 5, 4, 6, 2)

            flatten_patches = patches.reshape(
                batch_size,
                grid_t * grid_h * grid_w,
                patch_size * patch_size * channel,
            )

            processed_videos_grouped[shape] = flatten_patches
            processed_grids[shape] = [[grid_t, grid_h, grid_w]] * batch_size

        processed_videos = reorder_videos(processed_videos_grouped, grouped_videos_index)
        processed_grids = reorder_videos(processed_grids, grouped_videos_index)
        pixel_values_videos = torch.cat(processed_videos, dim=0)
        video_grid_thw = torch.tensor(processed_grids)
        data = {
            "pixel_values_videos": pixel_values_videos,
            "video_grid_thw": video_grid_thw,
        }

        return BatchFeature(data=data, tensor_type=return_tensors)


class NemotronNanoV3BridgeProcessor(ProcessorMixin):
    """Cosmos3-Edge processor: SigLIP2 image processor + Qwen3VL video processor +
    tokenizer behind a Qwen3VLProcessor-style interface."""

    attributes = ["image_processor", "tokenizer", "video_processor"]

    image_processor_class = "AutoImageProcessor"
    video_processor_class = "AutoVideoProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, image_processor=None, tokenizer=None, video_processor=None, chat_template=None, **kwargs):
        self.image_token = "<|image_pad|>" if not hasattr(tokenizer, "image_token") else tokenizer.image_token
        self.video_token = "<|video_pad|>" if not hasattr(tokenizer, "video_token") else tokenizer.video_token
        self.image_token_id = (
            tokenizer.image_token_id
            if getattr(tokenizer, "image_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.image_token)
        )
        self.video_token_id = (
            tokenizer.video_token_id
            if getattr(tokenizer, "video_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.video_token)
        )
        super().__init__(image_processor, tokenizer, video_processor, chat_template=chat_template)
        self.vision_start_token = (
            "<|vision_start|>" if not hasattr(tokenizer, "vision_start_token") else tokenizer.vision_start_token
        )
        self.vision_end_token = (
            "<|vision_end|>" if not hasattr(tokenizer, "vision_end_token") else tokenizer.vision_end_token
        )
        self.vision_start_token_id = (
            tokenizer.vision_start_token_id
            if getattr(tokenizer, "vision_start_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.vision_start_token)
        )
        self.vision_end_token_id = (
            tokenizer.vision_end_token_id
            if getattr(tokenizer, "vision_end_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.vision_end_token)
        )

    def apply_chat_template(self, conversation, **kwargs):
        """Extract per-image ``min_pixels``/``max_pixels`` from message content items
        (e.g. ``{"type": "image", "image": ..., "min_pixels": 256*28*28}``) and route
        them to the image processor; items without them use the global defaults."""
        if isinstance(conversation, (list, tuple)) and conversation and isinstance(conversation[0], (list, tuple)):
            conversations = conversation
        else:
            conversations = [conversation]

        per_image_kwargs: list = []
        for conv in conversations:
            for message in conv:
                content = message.get("content")
                if not isinstance(content, list):
                    continue
                for item in content:
                    if item.get("type") != "image":
                        continue
                    img_override = {}
                    if "min_pixels" in item:
                        img_override["min_pixels"] = item["min_pixels"]
                    if "max_pixels" in item:
                        img_override["max_pixels"] = item["max_pixels"]
                    per_image_kwargs.append(img_override if img_override else None)

        if any(v is not None for v in per_image_kwargs):
            kwargs["per_image_kwargs"] = per_image_kwargs

        return super().apply_chat_template(conversation, **kwargs)

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]] = None,
        videos: VideoInput = None,
        **kwargs: Unpack[Qwen3VLProcessorKwargs],
    ) -> BatchFeature:
        """Tokenize text and process images/videos into one [`BatchFeature`]:
        ``input_ids``/``attention_mask``, plus ``pixel_values`` + ``image_grid_thw``
        and/or ``pixel_values_videos`` + ``video_grid_thw`` when vision inputs are given."""
        output_kwargs = self._merge_kwargs(
            Qwen3VLProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        if images is not None:
            image_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
            pixel_values = image_inputs.pop("pixel_values")
            spatial_shapes = image_inputs.pop("spatial_shapes")
            final_pixel_value = pixel_values.view(-1, pixel_values.shape[-1])
            t_dim = torch.ones((spatial_shapes.shape[0], 1), dtype=spatial_shapes.dtype, device=spatial_shapes.device)
            image_grid_thw = torch.cat([t_dim, spatial_shapes], dim=1)
            image_inputs = {
                "pixel_values": final_pixel_value,
                "image_grid_thw": image_grid_thw,
            }
        else:
            image_inputs = {}
            image_grid_thw = None

        if videos is not None:
            videos_inputs = self.video_processor(videos=videos, **output_kwargs["videos_kwargs"])
            video_grid_thw = videos_inputs["video_grid_thw"]
            if not kwargs.get("return_metadata"):
                video_metadata = videos_inputs.pop("video_metadata")
            else:
                video_metadata = videos_inputs["video_metadata"]
        else:
            videos_inputs = {}
            video_grid_thw = None

        if not isinstance(text, list):
            text = [text]

        text = text.copy()  # below lines change text in-place
        if image_grid_thw is not None:
            merge_length = self.image_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while self.image_token in text[i]:
                    num_image_tokens = image_grid_thw[index].prod() // merge_length
                    text[i] = text[i].replace(self.image_token, "<|placeholder|>" * num_image_tokens, 1)
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.image_token)

        if video_grid_thw is not None:
            merge_length = self.video_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while self.video_token in text[i]:
                    metadata = video_metadata[index]
                    if metadata.fps is None:
                        logger.warning_once(
                            "Qwen3VL requires frame timestamps to construct prompts, but the `fps` of the input video "
                            "could not be inferred. Probably `video_metadata` was missing from inputs and you passed "
                            "pre-sampled frames. Defaulting to `fps=24`. Please provide `video_metadata` for more "
                            "accurate results."
                        )
                        metadata.fps = 24 if metadata.fps is None else metadata.fps

                    curr_timestamp = self._calculate_timestamps(
                        metadata.frames_indices,
                        metadata.fps,
                        merge_size=1,
                    )

                    video_placeholder = ""
                    frame_seqlen = video_grid_thw[index][1:].prod() // merge_length
                    for frame_idx in range(video_grid_thw[index][0]):
                        curr_time = curr_timestamp[frame_idx]
                        video_placeholder += f"<{curr_time:.1f} seconds>"
                        video_placeholder += (
                            self.vision_start_token + "<|placeholder|>" * frame_seqlen + self.vision_end_token
                        )
                    if f"{self.vision_start_token}{self.video_token}{self.vision_end_token}" in text[i]:
                        text[i] = text[i].replace(
                            f"{self.vision_start_token}{self.video_token}{self.vision_end_token}", video_placeholder, 1
                        )
                    else:
                        # vllm may input video token directly
                        text[i] = text[i].replace(self.video_token, video_placeholder, 1)
                    index += 1

                text[i] = text[i].replace("<|placeholder|>", self.video_token)

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", None)
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        self._check_special_mm_tokens(text, text_inputs, modalities=["image", "video"])

        if return_mm_token_type_ids:
            array_ids = np.array(text_inputs["input_ids"])
            mm_token_type_ids = np.zeros_like(text_inputs["input_ids"])
            mm_token_type_ids[array_ids == self.image_token_id] = 1
            text_inputs["mm_token_type_ids"] = mm_token_type_ids.tolist()

        return BatchFeature(data={**text_inputs, **image_inputs, **videos_inputs}, tensor_type=return_tensors)

    def _get_num_multimodal_tokens(self, image_sizes=None, video_sizes=None, **kwargs):
        """Return `MultiModalData` with placeholder-token counts for the given
        (height, width) image sizes and/or (num_frames, height, width) video sizes."""

        vision_data = {}
        if image_sizes is not None:
            images_kwargs = Qwen3VLProcessorKwargs._defaults.get("images_kwargs", {})
            images_kwargs.update(kwargs)
            merge_size = images_kwargs.get("merge_size", None) or self.image_processor.merge_size

            num_image_patches = [
                self.image_processor.get_number_of_image_patches(*image_size, images_kwargs)
                for image_size in image_sizes
            ]
            num_image_tokens = [(num_patches // merge_size**2) for num_patches in num_image_patches]
            vision_data.update({"num_image_tokens": num_image_tokens, "num_image_patches": num_image_patches})

        if video_sizes is not None:
            videos_kwargs = Qwen3VLProcessorKwargs._defaults.get("videos_kwargs", {})
            videos_kwargs.update(kwargs)
            num_video_patches = [
                self.video_processor.get_number_of_video_patches(*video_size, videos_kwargs)
                for video_size in video_sizes
            ]
            num_video_tokens = [(num_patches // merge_size**2) for num_patches in num_video_patches]
            vision_data["num_video_tokens"] = num_video_tokens

        return MultiModalData(**vision_data)

    def post_process_image_text_to_text(
        self, generated_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False, **kwargs
    ):
        """Batch-decode generated token ids to text via the tokenizer."""
        return self.tokenizer.batch_decode(
            generated_outputs,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

    def _calculate_timestamps(self, indices: Union[list[int], np.ndarray], video_fps: float, merge_size: int = 2):
        if not isinstance(indices, list):
            indices = indices.tolist()
        if len(indices) % merge_size != 0:
            indices.extend(indices[-1] for _ in range(merge_size - len(indices) % merge_size))
        timestamps = [idx / video_fps for idx in indices]
        # frames are merged by self.merge_size, so we need to average the
        # timestamps between the first/last frame within the temporal patch
        timestamps = [
            (timestamps[i] + timestamps[i + merge_size - 1]) / 2 for i in range(0, len(timestamps), merge_size)
        ]
        return timestamps


__all__ = [
    "NemotronNanoV3BridgeProcessor",
    "Qwen3VLVideoProcessorCustom",
    "Siglip2ImageProcessorCustom",
    "build_cosmos3_edge_processor",
    "is_cosmos3_edge_native_snapshot",
]
